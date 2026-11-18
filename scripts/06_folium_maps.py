"""Step 6: interactive Folium maps (+ a static PNG export of the same layers) from the real prediction pipeline.

Usage: python 06_folium_maps.py [trip_dir ...]      (default: three student_test trips)
Layers: raw GPS fixes, road network around the track, reference route, classifier-only route, classifier + HMM route,
fixes where the HMM arc differs from the reference. Outputs: results/maps/map_<split>_<trip>.html and
results/figures/map_static_<split>_<trip>.png (the PNG is a matplotlib export of the same data, for the PDF report).
Reference routes are drawn only here (offline analysis); the API/UI never receives them.
"""
import _setup  # noqa: F401
import sys
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from folium.plugins import MiniMap

from pathsnap.common import list_trips, load_config, read_reference, resolve
from pathsnap.features import align_to_route
from pathsnap.pipeline import PathSnapPredictor, evaluate_prediction

LEGEND = """
<div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: white; padding: 10px 12px;
 border: 1px solid #888; border-radius: 6px; font-size: 13px; line-height: 1.5;">
<b>PathSnap legend</b><br>
<span style="color:#1f5fbf">&#9679;</span> raw GPS fixes<br>
<span style="color:#9a9a9a">&#9644;</span> road network<br>
<span style="color:#2e9e44">&#9644;</span> reference route<br>
<span style="color:#f08c00">&#9644;</span> classifier-only (top-1 arcs)<br>
<span style="color:#d6281e">&#9644;</span> classifier + HMM/Viterbi route<br>
<span style="color:#000">&#10005;</span> fix where HMM arc != reference arc
</div>"""


def arc_line(trip, a):
    n = trip.nodes_ll
    return [[float(n[trip.arcs[a, 0], 1]), float(n[trip.arcs[a, 0], 0])], [float(n[trip.arcs[a, 1], 1]), float(n[trip.arcs[a, 1], 0])]]


def dedup(seq):
    seq = np.asarray(seq)
    seq = seq[seq >= 0]
    return seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq


def make_map(pred: PathSnapPredictor, trip_dir: Path, out_html: Path, out_png: Path):
    res = pred.predict(trip_dir)
    trip, net, P = res["_trip"], res["_net"], res["_P"]
    ev = evaluate_prediction(trip_dir, res, pred.cfg)
    ref_route = read_reference(trip_dir)
    ref_fix = align_to_route(P, net, ref_route, pred.cfg["alignment"]["max_cost_m"])
    clf_arcs, hmm_route = dedup(res["clf_arc"]), res["route_arcs"]
    c = [float(np.mean(trip.lat)), float(np.mean(trip.lon))]
    m = folium.Map(location=c, zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[float(trip.lat.min()), float(trip.lon.min())], [float(trip.lat.max()), float(trip.lon.max())]])

    # road network near the track (capped for map size)
    lo, hi = P.min(0) - 150, P.max(0) + 150
    near = ((np.maximum(net.A, net.B) >= lo) & (np.minimum(net.A, net.B) <= hi)).all(1)
    idx = np.flatnonzero(near)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(0.5 * (net.A[idx] + net.B[idx]), distance_upper_bound=150)
    idx = idx[np.isfinite(d)]
    if len(idx) > 6000:
        idx = np.random.RandomState(0).choice(idx, 6000, replace=False)
    g_net = folium.FeatureGroup("Road network (arcs within 150 m of the track)", show=True)
    folium.PolyLine([arc_line(trip, a) for a in idx], color="#9a9a9a", weight=2, opacity=.7).add_to(g_net)
    g_ref = folium.FeatureGroup("Reference route", show=True)
    folium.PolyLine([arc_line(trip, a) for a in ref_route], color="#2e9e44", weight=7, opacity=.55).add_to(g_ref)
    g_clf = folium.FeatureGroup("Classifier-only (top-1 arcs)", show=False)
    folium.PolyLine([arc_line(trip, a) for a in clf_arcs], color="#f08c00", weight=4, opacity=.9, dash_array="6 6").add_to(g_clf)
    g_hmm = folium.FeatureGroup("Classifier + HMM route", show=True)
    folium.PolyLine([arc_line(trip, a) for a in hmm_route], color="#d6281e", weight=3, opacity=.95).add_to(g_hmm)
    g_gps = folium.FeatureGroup("Raw GPS fixes", show=True)
    step = max(1, len(P) // 2500)
    for i in range(0, len(P), step):
        folium.CircleMarker([float(trip.lat[i]), float(trip.lon[i])], radius=2, color="#1f5fbf", fill=True, fill_opacity=.9,
                            weight=0, tooltip=f"fix {i}, t={trip.t[i]:.0f}s").add_to(g_gps)
    g_err = folium.FeatureGroup("Fixes where HMM arc != reference arc", show=False)
    wrong = np.flatnonzero(res["hmm_arc"] != ref_fix)
    for i in wrong[:3000]:
        folium.Marker([float(trip.lat[i]), float(trip.lon[i])], icon=folium.DivIcon(html='<div style="font-size:12px">&#10005;</div>'),
                      tooltip=f"fix {i}: HMM arc {res['hmm_arc'][i]} vs reference {ref_fix[i]}").add_to(g_err)
    for g in (g_net, g_ref, g_clf, g_hmm, g_gps, g_err):
        g.add_to(m)
    folium.Marker([float(trip.lat[0]), float(trip.lon[0])], icon=folium.Icon(color="green", icon="play"), tooltip="start").add_to(m)
    folium.Marker([float(trip.lat[-1]), float(trip.lon[-1])], icon=folium.Icon(color="black", icon="stop"), tooltip="end").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND))
    h, k = ev["classifier_plus_hmm"], ev["classifier_only"]
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px 10px;border:1px solid #888;'
        f'font-size:13px;border-radius:6px"><b>Trip {trip.trip_id:08d}</b> &mdash; Top-1: classifier {k["top1_acc"]:.3f}, '
        f'+HMM {h["top1_acc"]:.3f}; continuity(h=3): {k["route_continuity_h3"]:.3f} &rarr; {h["route_continuity_h3"]:.3f}</div>'))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(out_html)

    # static export of the same layers
    fig, ax = plt.subplots(figsize=(9, 8))
    for a in idx:
        ax.plot([net.A[a, 0], net.B[a, 0]], [net.A[a, 1], net.B[a, 1]], c="#c8c8c8", lw=.6, zorder=1)
    for a in ref_route:
        ax.plot([net.A[a, 0], net.B[a, 0]], [net.A[a, 1], net.B[a, 1]], c="#2e9e44", lw=6, alpha=.45, zorder=2)
    for a in hmm_route:
        ax.plot([net.A[a, 0], net.B[a, 0]], [net.A[a, 1], net.B[a, 1]], c="#d6281e", lw=1.8, zorder=3)
    ax.scatter(P[::step, 0], P[::step, 1], s=3, c="#1f5fbf", zorder=4)
    ax.scatter(P[wrong, 0], P[wrong, 1], s=14, marker="x", c="k", zorder=5, linewidths=.7)
    ax.set_aspect("equal"); ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    ax.set_title(f"Trip {trip.trip_id:08d}: GPS (blue), reference (green), HMM route (red), HMM errors (x)\n"
                 f"Top-1 classifier {k['top1_acc']:.3f} -> classifier+HMM {h['top1_acc']:.3f}", fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], c="#c8c8c8", label="road network"), Line2D([], [], c="#2e9e44", lw=5, alpha=.5, label="reference route"),
                       Line2D([], [], c="#d6281e", label="classifier + HMM route"),
                       Line2D([], [], marker="o", ls="", c="#1f5fbf", ms=3, label="raw GPS fixes"),
                       Line2D([], [], marker="x", ls="", c="k", ms=5, label="HMM arc != reference")], loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    return ev


def main():
    cfg = load_config()
    raw = resolve(cfg, "raw_root")
    pred = PathSnapPredictor(cfg)
    if len(sys.argv) > 1:
        dirs = [Path(x) for x in sys.argv[1:]]
    else:
        st = list_trips(raw / cfg["paths"]["student_dir"])
        dirs = [st[0], st[3], st[6]]
    for d in dirs:
        split = d.parent.name
        h = resolve(cfg, "results") / "maps" / f"map_{split}_{d.name}.html"
        p = resolve(cfg, "results") / "figures" / f"map_static_{split}_{d.name}.png"
        ev = make_map(pred, d, h, p)
        print(d.name, "top1 clf %.4f -> hmm %.4f" % (ev["classifier_only"]["top1_acc"], ev["classifier_plus_hmm"]["top1_acc"]), "->", h.name)


if __name__ == "__main__":
    main()
