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


def main():
    cfg = load_config()
    raw = resolve(cfg, "raw_root")
    out = resolve(cfg, "results") / "eda"
    fig_dir = resolve(cfg, "results") / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, key in (("train", "train_dir"), ("student_test", "student_dir")):
        for d in list_trips(raw / cfg["paths"][key]):
            rows.append(trip_row(split, d))
    inv = pd.DataFrame(rows)
    # reference-route distance statistics from the processed tables (if built)
    proc = resolve(cfg, "processed")
    refd = []
    for sp, name in (("train", "train"), ("student", "student_test")):
        f = proc / f"{sp}_obs.parquet"
        if f.exists():
            o = pd.read_parquet(f)
            g = o.groupby("trip")["ref_dist"].agg(ref_dist_median="median", ref_dist_p95=lambda x: np.percentile(x, 95),
                                                  ref_dist_max="max")
            g["candidate_coverage"] = o.groupby("trip")["in_cand"].mean()
            g["split"] = name
            refd.append(g.reset_index())
    if refd:
        inv = inv.merge(pd.concat(refd), on=["split", "trip"], how="left")
        q = cfg["quality"]
        inv["label_quality_flag"] = np.where(inv["ref_dist_median"] > q["max_trip_median_ref_dist_m"], "EXCLUDED_FROM_TRAINING", "ok")
    inv.to_csv(out / "dataset_inventory.csv", index=False)

    # ---- summary tables
    summ = inv.groupby("split").agg(
        trips=("trip", "size"), fixes_total=("n_fixes", "sum"), fixes_median=("n_fixes", "median"),
        fixes_min=("n_fixes", "min"), fixes_max=("n_fixes", "max"), nodes_median=("n_nodes", "median"),
        arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"), extent_km_median=("extent_km", "median"),
        extent_km_max=("extent_km", "max"), pct_dt1_mean=("pct_dt_eq_1", "mean"), dt_max=("dt_max", "max"),
        trips_with_gap_gt_10s=("n_dt_gt_10", lambda x: int((x > 0).sum())),
        multi_component_trips=("components", lambda x: int((x > 1).sum())),
        isolated_nodes_total=("isolated_nodes", "sum"), nan_total=("nan_track", "sum"),
        non_increasing_t_total=("non_increasing_t", "sum"))
    summ.T.to_csv(out / "dataset_summary_by_partition.csv")
    missing = inv.groupby("split")[["nan_track", "nan_nodes", "bad_lonlat", "arc_index_out_of_range",
                                    "route_index_out_of_range", "non_increasing_t", "duplicate_consecutive_fixes",
                                    "self_loops"]].sum()
    missing.to_csv(out / "missing_and_invalid_values.csv")
    anomalies = inv[(inv.dt_max > 60) | (inv.route_to_track_length_ratio < 0.5) | (inv.route_to_track_length_ratio > 1.5)
                    | (inv.get("ref_dist_median", pd.Series(0, index=inv.index)) > 30)]
    anomalies.to_csv(out / "anomalous_trips.csv", index=False)
    print(summ.T.to_string())
    print("\nmissing / invalid:\n", missing.to_string())
    print(f"\nanomalous trips: {len(anomalies)} -> {sorted(anomalies.trip.tolist())}")

    # ---- figures
    tr = inv[inv.split == "train"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].hist(inv.n_fixes, bins=30, color="#3b6fb6"); ax[0, 0].set_title("GPS fixes per trip (train + student_test)")
    ax[0, 0].set_xlabel("fixes"); ax[0, 0].set_ylabel("trips"); ax[0, 0].set_xscale("log")
    for s, c in (("train", "#3b6fb6"), ("student_test", "#d9822b")):
        sub = inv[inv.split == s]
        ax[0, 1].scatter(sub.lon_mean, sub.lat_mean, s=18, c=c, label=f"{s} ({len(sub)})", alpha=.8)
    ax[0, 1].set_title("Trip locations (dataset is global)"); ax[0, 1].set_xlabel("longitude"); ax[0, 1].set_ylabel("latitude"); ax[0, 1].legend()
    ax[0, 2].hist(inv.pct_dt_eq_1, bins=20, color="#3b6fb6"); ax[0, 2].set_title("% of steps with dt == 1 s, per trip"); ax[0, 2].set_xlabel("%")
    ax[1, 0].scatter(inv.n_arcs, inv.n_fixes, s=18, c="#3b6fb6"); ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("Road-network size vs trip length"); ax[1, 0].set_xlabel("arcs in network"); ax[1, 0].set_ylabel("GPS fixes")
    ax[1, 1].hist(inv.speed_p50 * 3.6, bins=25, color="#3b6fb6", alpha=.8, label="median"); ax[1, 1].hist(inv.speed_p99 * 3.6, bins=25, color="#d9822b", alpha=.6, label="99th pct")
    ax[1, 1].set_title("Per-trip speed from consecutive fixes"); ax[1, 1].set_xlabel("km/h"); ax[1, 1].legend()
    if "ref_dist_median" in inv:
        ax[1, 2].scatter(inv.ref_dist_median, inv.route_to_track_length_ratio, s=22, c=np.where(inv.ref_dist_median > 30, "#c0392b", "#3b6fb6"))
        ax[1, 2].axvline(30, ls="--", c="grey"); ax[1, 2].set_xscale("log")
        ax[1, 2].set_title("Reference-route quality (red = excluded from training)")
        ax[1, 2].set_xlabel("median distance fix -> reference arc (m)"); ax[1, 2].set_ylabel("route length / track length")
    plt.tight_layout(); plt.savefig(fig_dir / "eda_overview.png", dpi=130); plt.close()

    # trajectory + network panels for a few trips
    pick = [tr.sort_values("n_fixes").trip.iloc[len(tr) // 2], tr.sort_values("n_arcs").trip.iloc[-1],
            tr.sort_values("n_fixes").trip.iloc[3]]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, tid in zip(ax, pick):
        trip = read_trip(raw / cfg["paths"]["train_dir"] / f"{tid:08d}", with_route=True)
        net = build_network(trip)
        P = project(trip.lon, trip.lat, trip_epsg(trip))
        lo, hi = P.min(0) - 300, P.max(0) + 300
        m = ((np.maximum(net.A, net.B) >= lo) & (np.minimum(net.A, net.B) <= hi)).all(1)
        idx = np.flatnonzero(m)
        if len(idx) > 40000:
            idx = np.random.RandomState(0).choice(idx, 40000, replace=False)
        for i in idx:
            a.plot([net.A[i, 0], net.B[i, 0]], [net.A[i, 1], net.B[i, 1]], c="#cfcfcf", lw=.4)
        for i in trip.route:
            a.plot([net.A[i, 0], net.B[i, 0]], [net.A[i, 1], net.B[i, 1]], c="#2e9e44", lw=1.6)
        a.scatter(P[:, 0], P[:, 1], s=2, c="#1f5fbf", zorder=3)
        a.set_aspect("equal"); a.set_title(f"train trip {tid:08d}: {len(P)} fixes, {len(trip.route)} route arcs", fontsize=10)
        a.set_xlabel("easting (m, UTM)"); a.set_ylabel("northing (m, UTM)")
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], c="#cfcfcf", label="road network"), Line2D([], [], c="#2e9e44", lw=2, label="reference route"),
                          Line2D([], [], marker="o", ls="", c="#1f5fbf", ms=3, label="raw GPS fixes")], loc="best")
    plt.tight_layout(); plt.savefig(fig_dir / "eda_trajectories_network.png", dpi=130); plt.close()
    json.dump({"example_trips": [int(x) for x in pick], "partition": "train"}, open(out / "eda_example_trips.json", "w"))
    print("EDA done ->", out)


if __name__ == "__main__":
    main()
