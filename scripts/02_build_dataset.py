"""Step 2: candidate generation + feature engineering + reference-route alignment for train and student_test.

Outputs (data/processed/):
  {split}_candidates.parquet : one row per (GPS fix, candidate arc): features + label
  {split}_obs.parquet        : one row per GPS fix: reference arc, distance to it, whether it is among the candidates
The evaluator partition is deliberately NOT touched here.
"""
import _setup  # noqa: F401
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.common import list_trips, load_config, read_trip, resolve, seg_dist
from pathsnap.features import FEATURES, align_to_route, build_features


def process(trip_dir, cfg):
    trip = read_trip(trip_dir, with_route=True)
    df, P, net = build_features(trip, cfg)
    ref = align_to_route(P, net, trip.route, cfg["alignment"]["max_cost_m"])
    df["label"] = (df["arc"].to_numpy() == ref[df["obs"].to_numpy()]).astype(np.int8)
    df["trip"] = trip.trip_id
    dref, _ = seg_dist(P, net.A[ref], net.B[ref])
    in_cand = np.zeros(len(P), bool)
    in_cand[df.loc[df["label"] == 1, "obs"].to_numpy()] = True
    obs = pd.DataFrame({"trip": trip.trip_id, "obs": np.arange(len(P)), "ref_arc": ref,
                        "ref_dist": dref, "in_cand": in_cand, "t": trip.t})
    return df, obs


def main():
    cfg = load_config()
    raw, out = resolve(cfg, "raw_root"), resolve(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    for split, key in (("train", "train_dir"), ("student", "student_dir")):
        trips = list_trips(raw / cfg["paths"][key])
        t0 = time.time()
        res = Parallel(n_jobs=2)(delayed(process)(d, cfg) for d in trips)
        cand = pd.concat([r[0] for r in res], ignore_index=True)
        obs = pd.concat([r[1] for r in res], ignore_index=True)
        cand = cand[["trip", "obs", "arc"] + FEATURES + ["label"]]
        cand.to_parquet(out / f"{split}_candidates.parquet", index=False)
        obs.to_parquet(out / f"{split}_obs.parquet", index=False)
        print(f"{split}: {len(trips)} trips, {len(obs)} fixes, {len(cand)} candidate rows, "
              f"positive rate {cand['label'].mean():.3f}, ref-in-candidates {obs['in_cand'].mean():.4f}, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
