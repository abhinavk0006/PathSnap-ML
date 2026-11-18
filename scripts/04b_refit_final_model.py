"""Step 4b (optional): rebuild everything step 4 hands on, from the saved Bayesian-optimisation result, without
re-running the 20-trial search. Use it when the saved model file does not load on a machine (pickled XGBoost
models depend on the library build) or when step 4 could not be run.

Reads results/hpo/best_params.json (written by 04_bayes_opt.py) and, with those parameters, on the
quality-filtered training trips (same data, folds and seed as step 4):
  * 5-fold grouped out-of-fold probabilities -> results/hpo/oof_probs.parquet   (needed by 05_hmm_decode.py)
  * a final fit on all training rows         -> models/final_model.joblib, models/model_meta.json
  * student-test probabilities               -> results/models/student_probs_tuned.npy (needed by step 5)
About 6 model fits.
"""
import _setup  # noqa: F401
import json
import time

import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.common import load_config, resolve
from pathsnap.data import load_split, xy
from pathsnap.features import FEATURES
from pathsnap.modelzoo import SEED


def build(name, params):
    if name == "XGBoost":
        return xgb.XGBClassifier(tree_method="hist", n_jobs=2, random_state=SEED, eval_metric="logloss", **params)
    if name == "LightGBM":
        return lgb.LGBMClassifier(subsample_freq=1, n_jobs=2, random_state=SEED, verbose=-1, **params)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=SEED, **params)
    raise ValueError(name)


def main():
    cfg = load_config()
    best = json.loads((resolve(cfg, "results") / "hpo" / "best_params.json").read_text())
    name, params = best["model"], best["best_params"]
    cand, _ = load_split(cfg, "train", quality_filter=True)
    X, y = xy(cand)
    groups = cand["trip"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=cfg["cv"]["n_splits"]).split(X, y, groups)):
        t0 = time.time()
        oof[va] = build(name, params).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        print(f"out-of-fold fold {k} done in {time.time() - t0:.0f}s", flush=True)
    oof_df = cand[["trip", "obs", "arc", "t_frac"]].copy()
    oof_df["p"] = oof
    hpo_dir = resolve(cfg, "results") / "hpo"
    oof_df.to_parquet(hpo_dir / "oof_probs.parquet", index=False)
    print("saved", hpo_dir / "oof_probs.parquet", flush=True)
    t0 = time.time()
    model = build(name, params).fit(X, y)
    print(f"fitted {name} on {len(X)} rows in {time.time() - t0:.0f}s with {params}")
    mdir = resolve(cfg, "models")
    joblib.dump(model, mdir / "final_model.joblib", compress=3)
    (mdir / "model_meta.json").write_text(json.dumps({
        "model": name, "params": params, "features": FEATURES, "seed": SEED,
        "candidates": cfg["candidates"], "features_cfg": cfg["features"],
        "trained_on": "train partition (quality-filtered)", "rebuilt_by": "04b_refit_final_model.py",
    }, indent=2))
    print("saved", mdir / "final_model.joblib")
    scand, _ = load_split(cfg, "student")
    Xs, _ = xy(scand)
    out = resolve(cfg, "results") / "models"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "student_probs_tuned.npy", model.predict_proba(Xs)[:, 1].astype(np.float32))
    print("saved", out / "student_probs_tuned.npy")


if __name__ == "__main__":
    main()
