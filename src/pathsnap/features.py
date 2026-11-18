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
def build_features(trip: Trip, cfg: dict, net: Network | None = None):
    """Compute the candidate/feature table for one trip. Returns (DataFrame, P, net)."""
    epsg = trip_epsg(trip)
    net = net or build_network(trip, epsg)
    P = project(trip.lon, trip.lat, epsg)
    t = trip.t.astype(float)
    n = len(P)
    qi, arc, d, tf = generate_candidates(P, net, cfg)
    if len(qi) == 0:
        return pd.DataFrame(columns=["obs", "arc"] + FEATURES), P, net
    fc = cfg["features"]
    span = fc["heading_span"]

    # per-fix motion descriptors
    i0, i1 = np.clip(np.arange(n) - span, 0, n - 1), np.clip(np.arange(n) + span, 0, n - 1)
    disp = P[i1] - P[i0]
    dn_ = np.linalg.norm(disp, axis=1)
    dts = np.maximum(t[i1] - t[i0], 1e-3)
    heading = np.arctan2(disp[:, 1], disp[:, 0])
    moving = (dn_ > fc["moving_threshold_m"]).astype(np.float32)
    speed = dn_ / dts
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]  # step[i] = |P[i]-P[i-1]|
    step_next = np.r_[step[1:], 0.0]

    hb, hf, vb, vf = long_baseline_headings(P, fc.get("long_heading_min_disp_m", 15.0))
    A, B, L = net.A[arc], net.B[arc], net.L[arc]
    dh = heading[qi] - net.bearing[arc]
    dhb, dhf = hb[qi] - net.bearing[arc], hf[qi] - net.bearing[arc]
    out = {
        "obs": qi, "arc": arc, "d": d, "t_frac": tf,
        "end_clamped": ((tf < 0.02) | (tf > 0.98)).astype(np.float32),
        "dist_to_end": np.minimum(tf, 1 - tf) * L,
        "arc_len": L,
        "deg_to_out": net.deg_out[net.to[arc]], "deg_from_in": net.deg_in[net.frm[arc]],
        "has_rev": net.has_rev[arc].astype(np.float32),
        "hcos": np.cos(dh) * moving[qi], "hsin": np.sin(dh) * moving[qi],
        "moving": moving[qi], "speed": speed[qi],
        "hcos_b": np.cos(dhb) * vb[qi], "hsin_b": np.sin(dhb) * vb[qi], "hb_valid": vb[qi].astype(np.float32),
        "hcos_f": np.cos(dhf) * vf[qi], "hsin_f": np.sin(dhf) * vf[qi], "hf_valid": vf[qi].astype(np.float32),
        "dt_prev": dt_prev[qi], "dt_next": dt_next[qi],
        "step_prev": step[qi], "step_next": step_next[qi],
    }
    # context: how well do the neighbouring fixes fit the same arc?
    nb = {}
    for lag in fc["lags"]:
        for name, sign in (("p", -1), ("n", 1)):
            j = np.clip(qi + sign * lag, 0, n - 1)
            dj, tj = seg_dist(P[j], A, B)
            out[f"d{name}{lag}"] = dj
            nb[(name, lag)] = (dj, tj)
    out["dwin"] = (d + sum(out[f"d{s}{l}"] for l in (1, 2) for s in "pn")) / 5.0
    out["dwin_max"] = np.max([d] + [out[f"d{s}{l}"] for l in (1, 2) for s in "pn"], axis=0)
    tp1, tn1 = nb[("p", 1)][1], nb[("n", 1)][1]
    out["prog_prev"] = (tf - tp1) * L
    out["prog_next"] = (tn1 - tf) * L
    out["prog_ratio_prev"] = out["prog_prev"] / np.maximum(out["step_prev"], 1.0)
    out["prog_ratio_next"] = out["prog_next"] / np.maximum(out["step_next"], 1.0)

    df = pd.DataFrame(out)
    g = df.groupby("obs")
    df["n_cand"] = g["d"].transform("size")
    df["d_rank"] = g["d"].rank(method="first") - 1
    df["d_rel"] = df["d"] - g["d"].transform("min")
    df["dwin_rel"] = df["dwin"] - g["dwin"].transform("min")
    df = df[["obs", "arc"] + FEATURES].copy()
    for c in FEATURES:
        df[c] = df[c].astype(np.float32)
    df["obs"] = df["obs"].astype(np.int32)
    df["arc"] = df["arc"].astype(np.int64)
    return df, P, net


# --------------------------------------------------------------------------- labels (training only)
def align_to_route(P: np.ndarray, net: Network, route: np.ndarray, max_cost: float = 60.0) -> np.ndarray:
    """Assign every GPS fix a reference arc by monotone alignment of the fixes to the ordered reference route.

    The reference route lists arcs in travel order but does not say which fix lies on which arc. We find the
    non-decreasing assignment j(i) (index into the route) minimising the summed (capped) distance between each
    fix and its assigned arc. Dynamic programming, O(n * m).
    """
    n, m = len(P), len(route)
    A, B = net.A[route], net.B[route]
    cost = np.empty((n, m), dtype=np.float32)
    for i in range(n):
        d, _ = seg_dist(np.broadcast_to(P[i], (m, 2)), A, B)
        cost[i] = np.minimum(d, max_cost)
    back = np.zeros((n, m), dtype=np.int32)
    ar = np.arange(m)
    dp = cost[0].copy()
    for i in range(1, n):
        pm = np.minimum.accumulate(dp)
        idx = np.maximum.accumulate(np.where(dp == pm, ar, 0))
        back[i] = idx
        dp = cost[i] + pm
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return route[j]
