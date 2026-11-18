"""Candidate generation, feature engineering and reference-route alignment.

For every GPS fix we generate candidate road arcs (all arcs within a radius, capped at the K nearest). Each
(fix, arc) pair becomes one row; a classifier scores how likely the arc is the road the vehicle was on.
Features only use the track and the road network. Labels use the reference route and are produced by a
separate function (`align_to_route`) that the prediction pipeline never calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from .common import Network, Trip, build_network, project, seg_dist, trip_epsg

FEATURES = [
    "d", "d_rel", "d_rank", "n_cand", "t_frac", "end_clamped", "dist_to_end",
    "arc_len", "deg_to_out", "deg_from_in", "has_rev",
    "hcos", "hsin", "moving", "speed", "dt_prev", "dt_next",
    "hcos_b", "hsin_b", "hb_valid", "hcos_f", "hsin_f", "hf_valid",
    "dp1", "dn1", "dp2", "dn2", "dp4", "dn4", "dwin", "dwin_rel", "dwin_max",
    "prog_prev", "prog_next", "step_prev", "step_next", "prog_ratio_next", "prog_ratio_prev",
]


def long_baseline_headings(P: np.ndarray, min_disp: float, cap: int = 400):
    """Heading of the vehicle measured over a LONG baseline: the displacement from the nearest earlier fix that is at
    least `min_disp` metres away (backward) and to the nearest later such fix (forward). When the vehicle crawls or is
    stopped, a +-2 fix heading is pure noise and the two travel directions of a road become indistinguishable; a longer
    baseline recovers the direction of travel. Returns (hb, hf, hb_valid, hf_valid), headings in radians."""
    n = len(P)
    hb = np.zeros(n)
    hf = np.zeros(n)
    vb = np.zeros(n, bool)
    vf = np.zeros(n, bool)
    for i in range(n):
        lo = max(0, i - cap)
        if i > lo:
            d = np.linalg.norm(P[lo:i] - P[i], axis=1)
            idx = np.flatnonzero(d >= min_disp)
            if len(idx):
                v = P[i] - P[lo + idx[-1]]
                hb[i], vb[i] = np.arctan2(v[1], v[0]), True
        hi = min(n, i + 1 + cap)
        if hi > i + 1:
            d = np.linalg.norm(P[i + 1:hi] - P[i], axis=1)
            idx = np.flatnonzero(d >= min_disp)
            if len(idx):
                v = P[i + 1 + idx[0]] - P[i]
                hf[i], vf[i] = np.arctan2(v[1], v[0]), True
    return hb, hf, vb, vf


# --------------------------------------------------------------------------- candidates
def generate_candidates(P: np.ndarray, net: Network, cfg: dict):
    """Return (obs_idx, arc_id, distance, t_frac) for all candidate pairs (capped to K nearest per fix)."""
    R = cfg["candidates"]["radius_m"]
    K = cfg["candidates"]["max_k"]
    FB = cfg["candidates"]["fallback_radius_m"]
    pad = max(R, FB) + 10
    lo, hi = P.min(0) - pad, P.max(0) + pad
    amin, amax = np.minimum(net.A, net.B), np.maximum(net.A, net.B)
    mask = (amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) & (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1])
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        z = np.zeros(0)
        return z.astype(int), z.astype(int), z, z
    lines = shapely.linestrings(np.stack([net.A[ids], net.B[ids]], axis=1))
    tree = shapely.STRtree(lines)
    pts = shapely.points(P)
    qi, ti = tree.query(pts, predicate="dwithin", distance=R)
    have = np.zeros(len(P), bool)
    have[qi] = True
    missing = np.flatnonzero(~have)
    if len(missing):  # fallback: single nearest arc within the fallback radius
        res = tree.query_nearest(pts[missing], max_distance=FB, all_matches=False)
        if res.shape[1]:
            qi = np.concatenate([qi, missing[res[0]]])
            ti = np.concatenate([ti, res[1]])
    arc = ids[ti]
    d, t = seg_dist(P[qi], net.A[arc], net.B[arc])
    order = np.lexsort((d, qi))
    qi, arc, d, t = qi[order], arc[order], d[order], t[order]
    starts = np.r_[0, np.flatnonzero(np.diff(qi)) + 1]
    counts = np.diff(np.r_[starts, len(qi)])
    rank = np.arange(len(qi)) - np.repeat(starts, counts)
    keep = rank < K
    return qi[keep], arc[keep], d[keep], t[keep]


# --------------------------------------------------------------------------- features
