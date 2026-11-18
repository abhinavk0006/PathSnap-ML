"""Step 3: train and compare multiple models.

Protocol
  * Fit/CV data: the (quality-filtered) 80 training trips, GroupKFold with trips as groups (no trip is ever split
    across folds).
  * Development comparison: the 10 student-test trips, evaluated unfiltered.
  * Evaluator trips are never loaded.
Results are appended to results/models/comparison.jsonl (one JSON record per model; finished models are skipped on
re-run). Usage: python 03_train_compare.py [ModelName ...]
"""
import _setup  # noqa: F401
import json
import sys
import time

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from pathsnap.common import list_trips, load_config, read_trip, resolve
from pathsnap.data import load_split, xy
from pathsnap.metrics import ArcGraph, candidate_metrics, evaluate_sequence, top1_arcs
from pathsnap.modelzoo import SEED, ZOO


def top1_acc(obs, cand, prob):
    pred = top1_arcs(cand, prob)
    m = obs[["trip", "obs", "ref_arc"]].merge(pred, on=["trip", "obs"], how="left")
    return float((m["pred_arc"].fillna(-1).to_numpy() == m["ref_arc"].to_numpy()).mean())


def fit(name, model, X, y, Xv=None, yv=None, curves=None):
    """Fit one model; for models with epoch-wise histories, record train/validation loss curves."""
    if name == "LightGBM" and Xv is not None and curves is not None:
        model.fit(X, y, eval_set=[(X, y), (Xv, yv)], eval_names=["train", "valid"], eval_metric="binary_logloss")
        h = model.evals_result_
        curves[name] = {"train": h["train"]["binary_logloss"], "valid": h["valid"]["binary_logloss"]}
    elif name == "XGBoost" and Xv is not None and curves is not None:
        model.fit(X, y, eval_set=[(X, y), (Xv, yv)], verbose=False)
        h = model.evals_result()
        curves[name] = {"train": h["validation_0"]["logloss"], "valid": h["validation_1"]["logloss"]}
    else:
        model.fit(X, y)
        if name == "MLP" and curves is not None:
            curves[name] = {"train": [float(v) for v in model[-1].loss_curve_]}
    return model


def main():
    cfg = load_config()
    res_dir = resolve(cfg, "results") / "models"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / "comparison.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["model"] for l in out_path.read_text().splitlines() if l.strip()}
    names = sys.argv[1:] or list(ZOO)

    cand, obs = load_split(cfg, "train", quality_filter=True)
    scand, sobs = load_split(cfg, "student")
    print(f"train rows {len(cand)} fixes {len(obs)} trips {obs.trip.nunique()} "
          f"(dropped trips {obs.attrs.get('dropped_trips')}); student rows {len(scand)} fixes {len(sobs)}", flush=True)
    raw = resolve(cfg, "raw_root")
    graphs = {}
    for d in list_trips(raw / cfg["paths"]["student_dir"]):
        t = read_trip(d)
        graphs[t.trip_id] = ArcGraph(t.arcs[:, 0], t.arcs[:, 1])

    X, y = xy(cand)
    Xs, ys = xy(scand)
    groups = cand["trip"].to_numpy()
    folds = list(GroupKFold(n_splits=cfg["cv"]["n_splits"]).split(X, y, groups))
    rng = np.random.RandomState(SEED)
    curves = {}
    cpath = res_dir / "curves.json"
    if cpath.exists():
        curves = json.loads(cpath.read_text())

    for name in names:
        if name in done:
            print("skip", name, flush=True)
            continue
        family, conf, factory, max_rows = ZOO[name]
        t_start = time.time()
        fold_f1, fold_top1, fold_ll, fold_fit = [], [], [], []
        for k, (tr, va) in enumerate(folds):
            tr_use = tr if (max_rows is None or len(tr) <= max_rows) else rng.choice(tr, max_rows, replace=False)
            t0 = time.time()
            model = fit(name, factory(), X[tr_use], y[tr_use], X[va], y[va], curves if k == 0 else None)
            fold_fit.append(time.time() - t0)
            p = model.predict_proba(X[va])[:, 1]
            cm = candidate_metrics(y[va], p)
            va_trips = np.unique(groups[va])
            ob = obs[obs["trip"].isin(va_trips)]
            fold_f1.append(cm["cand_macro_f1"])
            fold_ll.append(cm["log_loss"])
            fold_top1.append(top1_acc(ob, cand.iloc[va], p))
            print(f"  {name} fold{k}: macroF1 {fold_f1[-1]:.4f} top1 {fold_top1[-1]:.4f} logloss {fold_ll[-1]:.4f} "
                  f"fit {fold_fit[-1]:.0f}s", flush=True)
        # final fit on all training rows, evaluation on the student-test trips
        tr_use = np.arange(len(X)) if (max_rows is None or len(X) <= max_rows) else rng.choice(len(X), max_rows, replace=False)
        t0 = time.time()
        model = fit(name, factory(), X[tr_use], y[tr_use])
        train_time = time.time() - t0
        t0 = time.time()
        ps = model.predict_proba(Xs)[:, 1]
        infer = time.time() - t0
        p_tr = model.predict_proba(X[tr_use])[:, 1]  # training-set metrics, used only for the over/under-fitting check
        cm_tr = candidate_metrics(y[tr_use], p_tr)
        cms = candidate_metrics(ys, ps)
        seq = evaluate_sequence(sobs, top1_arcs(scand, ps), graphs)
        per_trip = seq.pop("per_trip")
        conf_m = confusion_matrix(ys, (ps >= 0.5).astype(int), labels=[0, 1]).tolist()
        rec = {
            "model": name, "family": family, "config": conf,
            "cv_macro_f1_mean": float(np.mean(fold_f1)), "cv_macro_f1_std": float(np.std(fold_f1)),
            "cv_top1_mean": float(np.mean(fold_top1)), "cv_top1_std": float(np.std(fold_top1)),
            "cv_logloss_mean": float(np.mean(fold_ll)),
            "cv_fold_macro_f1": fold_f1, "cv_fold_top1": fold_top1,
            "train_macro_f1": cm_tr["cand_macro_f1"], "train_logloss": cm_tr["log_loss"],
            "train_time_s": train_time, "cv_total_time_s": time.time() - t_start,
            "inference_ms_per_1000_fixes": infer / len(sobs) * 1000 * 1000,
            "student": {**cms, **seq}, "student_confusion_0.5": conf_m, "student_per_trip": per_trip,
            "n_train_rows": int(len(tr_use)), "n_features": int(X.shape[1]), "seed": SEED,
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        cpath.write_text(json.dumps(curves))
        np.save(res_dir / f"student_probs_{name}.npy", ps.astype(np.float32))
        print(f"DONE {name}: CV macroF1 {rec['cv_macro_f1_mean']:.4f}+-{rec['cv_macro_f1_std']:.4f} "
              f"CV top1 {rec['cv_top1_mean']:.4f} | student top1 {seq['top1_acc']:.4f} macroF1 {cms['cand_macro_f1']:.4f} "
              f"logloss {cms['log_loss']:.4f} cont3 {seq['route_continuity_h3']:.4f} | train {train_time:.0f}s "
              f"infer {rec['inference_ms_per_1000_fixes']:.0f}ms/1000 fixes", flush=True)


if __name__ == "__main__":
    main()
