"""End-to-end inference: raw trip folder -> candidates -> classifier -> HMM/Viterbi -> route.

Used by the FastAPI backend and by the evaluation scripts, so the demo runs exactly the code that was evaluated.
The reference route is NOT read here; evaluation against it is a separate, server-side step (`evaluate_prediction`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import ROOT, build_network, list_trips, load_config, project, read_reference, read_trip, resolve, trip_epsg
from .features import FEATURES, align_to_route, build_features
from .hmm import Decoder, HMMParams
from .metrics import ArcGraph, evaluate_sequence


class PathSnapPredictor:
    def __init__(self, cfg: dict | None = None, models_dir: Path | None = None):
        self.cfg = cfg or load_config()
        mdir = Path(models_dir) if models_dir else resolve(self.cfg, "models")
        self.model = joblib.load(mdir / "final_model.joblib")
        self.meta = json.loads((mdir / "model_meta.json").read_text())
        hp = mdir / "hmm_config.json"
        self.hmm = HMMParams.from_dict(json.loads(hp.read_text())["params"]) if hp.exists() else HMMParams.from_dict(self.cfg["hmm"])
        assert self.meta["features"] == FEATURES, "feature list of the saved model differs from the code"

    def predict(self, trip_dir: str | Path, use_hmm: bool = True) -> dict:
        t0 = time.time()
        trip = read_trip(trip_dir)                       # track + network only: no labels are read
        df, P, net = build_features(trip, self.cfg)
        n = len(P)
        timing = {"features_s": time.time() - t0}
        t1 = time.time()
        prob = self.model.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1] if len(df) else np.zeros(0)
        timing["classifier_s"] = time.time() - t1
        d = df[["obs", "arc", "t_frac"]].copy()
        d["p"] = prob
        clf = np.full(n, -1, dtype=np.int64)
        if len(d):
            top = d.loc[d.groupby("obs")["p"].idxmax()]
            clf[top["obs"].to_numpy()] = top["arc"].to_numpy()
        hmm_arc, route, stats = clf.copy(), np.array([], dtype=np.int64), {}
        t2 = time.time()
        if use_hmm and len(d):
            dec = Decoder(net, P, trip.t, d["arc"].to_numpy(), self.hmm)
            out = dec.decode(d, d["p"].to_numpy())
            hmm_arc = np.full(n, -1, dtype=np.int64)
            hmm_arc[out["obs"].to_numpy()] = out["arc"].to_numpy()
            route = dec.fill_route(hmm_arc)
            stats = dict(dec.stats)
        timing["hmm_s"] = time.time() - t2
        timing["total_s"] = time.time() - t0
        return {"trip_id": trip.trip_id, "n_fixes": n, "clf_arc": clf, "hmm_arc": hmm_arc, "route_arcs": route,
                "stats": stats, "timing": timing, "_df": df, "_net": net, "_trip": trip, "_P": P}


def route_geometry(res: dict) -> list[list[list[float]]]:
    """Predicted route as a list of [lon, lat] polylines (one per arc), for the UI."""
    trip, net = res["_trip"], res["_net"]
    nodes = trip.nodes_ll
    arcs = trip.arcs
    return [[[float(nodes[arcs[a, 0], 0]), float(nodes[arcs[a, 0], 1])],
             [float(nodes[arcs[a, 1], 0]), float(nodes[arcs[a, 1], 1])]] for a in res["route_arcs"]]


def evaluate_prediction(trip_dir: str | Path, res: dict, cfg: dict | None = None) -> dict:
    """Server-side evaluation of one prediction against the reference route (labels stay on the server)."""
    cfg = cfg or load_config()
    trip, net, df, P = res["_trip"], res["_net"], res["_df"], res["_P"]
    ref_route = read_reference(trip_dir)
    ref = align_to_route(P, net, ref_route, cfg["alignment"]["max_cost_m"])
    n = len(P)
    in_cand = np.zeros(n, bool)
    lab = df["arc"].to_numpy() == ref[df["obs"].to_numpy()]
    in_cand[df["obs"].to_numpy()[lab]] = True
    obs = pd.DataFrame({"trip": trip.trip_id, "obs": np.arange(n), "ref_arc": ref, "in_cand": in_cand})
    graphs = {trip.trip_id: ArcGraph(trip.arcs[:, 0], trip.arcs[:, 1])}

    def run(arcs, routes=None):
        pred = pd.DataFrame({"trip": trip.trip_id, "obs": np.arange(n), "pred_arc": arcs})
        r = evaluate_sequence(obs, pred, graphs, routes=routes)
        r.pop("per_trip")
        return r

    clf = run(res["clf_arc"])
    hmm = run(res["hmm_arc"], routes={trip.trip_id: res["route_arcs"]} if len(res["route_arcs"]) else None)
    return {"classifier_only": clf, "classifier_plus_hmm": hmm}
