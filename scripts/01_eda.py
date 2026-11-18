"""Step 1: dataset inventory and exploratory data analysis (train + student_test partitions only).

Writes results/eda/*.csv and results/figures/eda_*.png. Run AFTER 02_build_dataset.py if you want the
reference-route distance statistics (they come from the processed observation tables).
"""
import _setup  # noqa: F401
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.common import build_network, list_trips, load_config, project, read_trip, resolve, trip_epsg, trip_paths


def trip_row(split, d):
    trip = read_trip(d, with_route=True)
    p = trip_paths(d)
    n_nodes, n_arcs = len(trip.nodes_ll), len(trip.arcs)
    t = trip.t
    dt = np.diff(t)
    epsg = trip_epsg(trip)
    P = project(trip.lon, trip.lat, epsg)
    net = build_network(trip, epsg)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    speed = step / np.maximum(dt, 1e-3)
    g = coo_matrix((np.ones(n_arcs), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(n_nodes, n_nodes))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    used = np.zeros(n_nodes, bool)
    used[trip.arcs.ravel()] = True
    route = trip.route
    return {
        "split": split, "trip": trip.trip_id, "epsg": epsg,
        "file_kb_track": p["track"].stat().st_size / 1024, "file_kb_nodes": p["nodes"].stat().st_size / 1024,
        "file_kb_arcs": p["arcs"].stat().st_size / 1024, "file_kb_route": p["route"].stat().st_size / 1024,
        "n_fixes": len(t), "n_nodes": n_nodes, "n_arcs": n_arcs, "n_route_arcs": len(route),
        "lon_mean": trip.lon.mean(), "lat_mean": trip.lat.mean(),
        "extent_km": float(np.linalg.norm(P.max(0) - P.min(0)) / 1000),
        "track_len_km": float(step.sum() / 1000), "duration_s": float(t[-1] - t[0]),
        # missing / invalid values
        "nan_track": int(np.isnan(np.c_[trip.lon, trip.lat, trip.t]).sum()),
        "nan_nodes": int(np.isnan(trip.nodes_ll).sum()),
        "bad_lonlat": int(((np.abs(trip.lon) > 180) | (np.abs(trip.lat) > 90)).sum()),
        "arc_index_out_of_range": int(((trip.arcs < 0) | (trip.arcs >= n_nodes)).sum()),
        "route_index_out_of_range": int(((route < 0) | (route >= n_arcs)).sum()),
        # timestamps
        "dt_median": float(np.median(dt)), "dt_max": float(dt.max()), "pct_dt_eq_1": float((dt == 1).mean() * 100),
        "n_dt_gt_1": int((dt > 1).sum()), "n_dt_gt_10": int((dt > 10).sum()),
        "non_increasing_t": int((dt <= 0).sum()), "duplicate_consecutive_fixes": int((step == 0).sum()),
        "speed_p50": float(np.median(speed)), "speed_p99": float(np.percentile(speed, 99)), "speed_max": float(speed.max()),
        # network
        "components": int(ncomp), "largest_component_share": float(sizes.max() / n_nodes),
        "isolated_nodes": int((~used).sum()), "self_loops": int((trip.arcs[:, 0] == trip.arcs[:, 1]).sum()),
        "pct_arcs_with_reverse": float(net.has_rev.mean() * 100), "arc_len_median_m": float(np.median(net.L)),
        "arc_len_max_m": float(net.L.max()),
        # reference route
        "route_len_km": float(net.L[route].sum() / 1000),
        "route_is_connected_pct": float((net.to[route[:-1]] == net.frm[route[1:]]).mean() * 100) if len(route) > 1 else 100.0,
        "route_repeats_arc": int(len(route) - len(np.unique(route))),
        "route_to_track_length_ratio": float(net.L[route].sum() / max(step.sum(), 1.0)),
    }


