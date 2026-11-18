"""Step 4: Bayesian hyper-parameter optimisation (Optuna TPE) of the selected model.

  * Model: the best model of the comparison by CV Top-1 accuracy (override: python 04_bayes_opt.py LightGBM).
  * Search: group-aware CV on the (quality-filtered) training trips only. The student-test trips are used only for the
    final untuned-vs-tuned comparison, never inside the search. Evaluator trips are never loaded.
  * Objective (maximise): mean CV observation-level Top-1 accuracy, i.e. exactly what the route decoder consumes.
  * Outputs: results/hpo/{trials.csv, best_params.json, tuned_vs_untuned.json, oof_probs.parquet},
             models/final_model.joblib, models/model_meta.json.
"""
import _setup  # noqa: F401
import json
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.common import list_trips, load_config, read_trip, resolve
from pathsnap.data import load_split, xy
from pathsnap.features import FEATURES
from pathsnap.metrics import ArcGraph, candidate_metrics, evaluate_sequence, top1_arcs
from pathsnap.modelzoo import SEED, ZOO

N_TRIALS = 20
HPO_FOLDS = 3   # fewer folds inside the search to bound compute on 2 CPU cores; final comparison uses 5 folds


def space(name, trial):
    """Search space per model family (ranges justified in docs/decisions.md)."""
    if name == "LightGBM":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
    if name == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if name == "HistGradientBoosting":
        return dict(
            max_iter=trial.suggest_int("max_iter", 100, 400, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        )
    raise SystemExit(f"No search space defined for {name}; choose LightGBM, XGBoost or HistGradientBoosting.")


def build(name, params):
    if name == "LightGBM":
        return lgb.LGBMClassifier(subsample_freq=1, n_jobs=2, random_state=SEED, verbose=-1, **params)
    if name == "XGBoost":
        return xgb.XGBClassifier(tree_method="hist", n_jobs=2, random_state=SEED, eval_metric="logloss", **params)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=SEED, **params)
    raise ValueError(name)


def top1_acc(obs, cand, prob):
    pred = top1_arcs(cand, prob)
    m = obs[["trip", "obs", "ref_arc"]].merge(pred, on=["trip", "obs"], how="left")
    return float((m["pred_arc"].fillna(-1).to_numpy() == m["ref_arc"].to_numpy()).mean())


def cv_eval(factory, X, y, cand, obs, groups, folds, collect_oof=False):
    f1s, accs, lls, oof = [], [], [], np.zeros(len(y), dtype=np.float32)
    for tr, va in folds:
        m = factory().fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        oof[va] = p
        cm = candidate_metrics(y[va], p)
        f1s.append(cm["cand_macro_f1"]), lls.append(cm["log_loss"])
        accs.append(top1_acc(obs[obs["trip"].isin(np.unique(groups[va]))], cand.iloc[va], p))
    return f1s, accs, lls, oof


def main():
    cfg = load_config()
    res_dir = resolve(cfg, "results") / "hpo"
    mdir = resolve(cfg, "models")
    res_dir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    comp = [json.loads(l) for l in (resolve(cfg, "results") / "models" / "comparison.jsonl").read_text().splitlines() if l.strip()]
    tunable = [c for c in comp if c["model"] in ("LightGBM", "XGBoost", "HistGradientBoosting")]
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = max(comp, key=lambda c: c["cv_top1_mean"])["model"]
        if name not in ("LightGBM", "XGBoost", "HistGradientBoosting"):
            name = max(tunable, key=lambda c: c["cv_top1_mean"])["model"]
            print("best model overall has no search space defined here; using best tunable model", name)
    print("Selected model for Bayesian optimisation:", name, flush=True)
    untuned = next(c for c in comp if c["model"] == name)

    cand, obs = load_split(cfg, "train", quality_filter=True)
    scand, sobs = load_split(cfg, "student")
    X, y = xy(cand)
    Xs, ys = xy(scand)
    groups = cand["trip"].to_numpy()
    hpo_folds = list(GroupKFold(n_splits=HPO_FOLDS).split(X, y, groups))

    def objective(trial):
        params = space(name, trial)
        t0 = time.time()
        f1s, accs, lls, _ = cv_eval(lambda: build(name, params), X, y, cand, obs, groups, hpo_folds)
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_logloss", float(np.mean(lls)))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: top1 {np.mean(accs):.4f} macroF1 {np.mean(f1s):.4f} "
              f"({time.time() - t0:.0f}s) {params}", flush=True)
        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6),
                                study_name=f"pathsnap_{name}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_search = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    search_seconds = time.time() - t_search
    study.trials_dataframe().to_csv(res_dir / "trials.csv", index=False)
    best = study.best_params
    (res_dir / "best_params.json").write_text(json.dumps({
        "model": name, "best_params": best, "best_cv_top1": study.best_value, "n_trials": N_TRIALS,
        "cv_folds_in_search": HPO_FOLDS, "sampler": "TPE (seed 42, 6 random start-up trials)",
        "objective": "mean group-CV observation-level Top-1 accuracy", "search_seconds": search_seconds,
        "search_space": {k: {"low": getattr(d, "low", None), "high": getattr(d, "high", None),
                             "log": bool(getattr(d, "log", False))}
                         for k, d in study.trials[0].distributions.items()},
    }, indent=2))
    print("best", study.best_value, best, flush=True)

    # --- tuned model: 5-fold CV with the SAME folds/protocol as the untuned comparison, then refit + student test
    folds5 = list(GroupKFold(n_splits=cfg["cv"]["n_splits"]).split(X, y, groups))
    f1s, accs, lls, oof = cv_eval(lambda: build(name, best), X, y, cand, obs, groups, folds5)
    t0 = time.time()
    model = build(name, best).fit(X, y)
    train_time = time.time() - t0
    t0 = time.time()
    ps = model.predict_proba(Xs)[:, 1]
    infer = time.time() - t0
    graphs = {}
    for d in list_trips(resolve(cfg, "raw_root") / cfg["paths"]["student_dir"]):
        t = read_trip(d)
        graphs[t.trip_id] = ArcGraph(t.arcs[:, 0], t.arcs[:, 1])
    seq = evaluate_sequence(sobs, top1_arcs(scand, ps), graphs)
    seq.pop("per_trip")
    cms = candidate_metrics(ys, ps)
    tuned = {
        "cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s)),
        "cv_top1_mean": float(np.mean(accs)), "cv_logloss_mean": float(np.mean(lls)),
        "student_top1": seq["top1_acc"], "student_macro_f1": cms["cand_macro_f1"], "student_logloss": cms["log_loss"],
        "student_obs_macro_f1": seq["obs_macro_f1"], "student_route_continuity_h3": seq["route_continuity_h3"],
        "inference_ms_per_1000_fixes": infer / len(sobs) * 1e6, "train_time_s": train_time,
        "search_seconds": search_seconds,
    }
    base = {
        "cv_macro_f1_mean": untuned["cv_macro_f1_mean"], "cv_macro_f1_std": untuned["cv_macro_f1_std"],
        "cv_top1_mean": untuned["cv_top1_mean"], "cv_logloss_mean": untuned["cv_logloss_mean"],
        "student_top1": untuned["student"]["top1_acc"], "student_macro_f1": untuned["student"]["cand_macro_f1"],
        "student_logloss": untuned["student"]["log_loss"], "student_obs_macro_f1": untuned["student"]["obs_macro_f1"],
        "student_route_continuity_h3": untuned["student"]["route_continuity_h3"],
        "inference_ms_per_1000_fixes": untuned["inference_ms_per_1000_fixes"], "train_time_s": untuned["train_time_s"],
    }
    (res_dir / "tuned_vs_untuned.json").write_text(json.dumps({"model": name, "untuned": base, "tuned": tuned}, indent=2))

    oof_df = cand[["trip", "obs", "arc", "t_frac"]].copy()
    oof_df["p"] = oof
    oof_df.to_parquet(res_dir / "oof_probs.parquet", index=False)
    np.save(resolve(cfg, "results") / "models" / "student_probs_tuned.npy", ps.astype(np.float32))
    joblib.dump(model, mdir / "final_model.joblib", compress=3)
    (mdir / "model_meta.json").write_text(json.dumps({
        "model": name, "params": best, "features": FEATURES, "seed": SEED,
        "candidates": cfg["candidates"], "features_cfg": cfg["features"], "trained_on": "train partition (quality-filtered)",
    }, indent=2))
    print("tuned:", json.dumps(tuned, indent=1), flush=True)
    print("untuned:", json.dumps(base, indent=1), flush=True)


if __name__ == "__main__":
    main()
