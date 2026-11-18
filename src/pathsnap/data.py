"""Loading of the processed candidate tables with the documented training-label quality filter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import resolve
from .features import FEATURES


def load_split(cfg: dict, split: str, quality_filter: bool = False):
    """Return (candidates, obs). split in {"train", "student"}.

    quality_filter=True removes unreliable reference labels (trips whose reference route does not follow the track,
    fixes far from their reference arc). It is applied to the training partition only.
    """
    d = resolve(cfg, "processed")
    cand = pd.read_parquet(d / f"{split}_candidates.parquet")
    obs = pd.read_parquet(d / f"{split}_obs.parquet")
    if quality_filter:
        q = cfg["quality"]
        med = obs.groupby("trip")["ref_dist"].median()
        bad_trips = med[med > q["max_trip_median_ref_dist_m"]].index
        obs = obs[~obs["trip"].isin(bad_trips) & (obs["ref_dist"] <= q["max_fix_ref_dist_m"])].reset_index(drop=True)
        key_o = obs["trip"].to_numpy(np.int64) * 1_000_000 + obs["obs"].to_numpy(np.int64)
        key_c = cand["trip"].to_numpy(np.int64) * 1_000_000 + cand["obs"].to_numpy(np.int64)
        cand = cand[np.isin(key_c, key_o)].reset_index(drop=True)
        obs.attrs["dropped_trips"] = [int(x) for x in bad_trips]
    return cand, obs


def xy(cand: pd.DataFrame):
    return cand[FEATURES].to_numpy(np.float32), cand["label"].to_numpy(np.int8)
