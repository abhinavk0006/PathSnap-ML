"""Evaluation metrics: candidate-level, observation-level and sequence-level.

Definitions (also written up in docs/metrics.md):
  * Candidate-level metrics treat every (fix, arc) row as a binary example (1 = the reference arc of that fix).
      - Macro F1 : mean of the F1 of class 0 and class 1 at threshold 0.5.
      - Log loss : binary cross-entropy of the predicted probability.
  * Observation-level Top-1 accuracy: per fix, take the highest-scoring candidate and compare it with the
    reference arc. A fix whose reference arc is not among its candidates counts as wrong (candidate miss).
  * Candidate miss rate: share of fixes whose reference arc is not in the candidate set.
  * Observation-level Macro F1: multiclass F1 (arc id as class, trip-qualified) averaged over the reference arcs.
  * Route continuity: share of consecutive-fix pairs whose predicted arcs are identical or connected within
    `hops` arc steps (hops=1 is strict adjacency). The same measure computed on the reference arcs is the ceiling,
    because at ~35 m/s a vehicle can skip over short arcs between two 1 Hz fixes.
  * Sequence-level: arc-set precision / recall / F1 of the unique predicted arcs vs the unique reference arcs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

EPS = 1e-7


def candidate_metrics(y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    pred = (p >= thr).astype(int)
    return {
        "cand_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "cand_precision": precision_score(y, pred, zero_division=0),
        "cand_recall": recall_score(y, pred, zero_division=0),
        "cand_f1_pos": f1_score(y, pred, zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def top1_arcs(cand: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """Per (trip, obs): arc with the highest probability. Returns DataFrame[trip, obs, pred_arc]."""
    d = cand[["trip", "obs", "arc"]].copy()
    d["p"] = prob
    idx = d.groupby(["trip", "obs"])["p"].idxmax()
    out = d.loc[idx, ["trip", "obs", "arc"]].rename(columns={"arc": "pred_arc"})
    return out.reset_index(drop=True)


def _out_csr(frm: np.ndarray, n_nodes: int):
    order = np.argsort(frm, kind="stable")
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(indptr, frm + 1, 1)
    return order, np.cumsum(indptr)


class ArcGraph:
    """Arc adjacency (arc a -> arcs leaving a's end node) for continuity checks."""

    def __init__(self, frm: np.ndarray, to: np.ndarray):
        self.frm, self.to = frm, to
        self.order, self.indptr = _out_csr(frm, int(max(frm.max(), to.max())) + 1)

    def successors(self, a: int) -> np.ndarray:
        n = self.to[a]
        return self.order[self.indptr[n]:self.indptr[n + 1]]

    def connected(self, a: int, b: int, hops: int) -> bool:
        if a == b:
            return True
        frontier = {a}
        for _ in range(hops):
            nxt = set()
            for x in frontier:
                s = self.successors(x)
                if b in s:
                    return True
                nxt.update(s.tolist())
            frontier = nxt
            if not frontier:
                break
        return False


def continuity(arcs: np.ndarray, graph: ArcGraph, hops: int) -> tuple[int, int]:
    """Returns (continuous_pairs, total_pairs) over consecutive fixes (pairs with a missing arc count as breaks)."""
    ok = tot = 0
    cache: dict = {}
    for a, b in zip(arcs[:-1], arcs[1:]):
        tot += 1
        if a < 0 or b < 0:
            continue
        key = (a, b)
        v = cache.get(key)
        if v is None:
            v = cache[key] = graph.connected(int(a), int(b), hops)
        ok += v
    return ok, tot


def lcs_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Ordered similarity of two de-duplicated arc sequences (difflib ratio)."""
    from difflib import SequenceMatcher

    def dedup(x):
        x = np.asarray(x)
        return x[np.r_[True, x[1:] != x[:-1]]].tolist() if len(x) else []

    return SequenceMatcher(None, dedup(a), dedup(b), autojunk=False).ratio()


def evaluate_sequence(obs: pd.DataFrame, pred: pd.DataFrame, graphs: dict[int, ArcGraph],
                      hops: tuple[int, ...] = (1, 3), routes: dict[int, np.ndarray] | None = None) -> dict:
    """obs: [trip, obs, ref_arc, in_cand]; pred: [trip, obs, pred_arc] (pred_arc = -1 when no candidate).

    Returns observation-level and sequence-level metrics pooled over all trips in `obs`. If `routes` (trip -> arc
    sequence, e.g. the shortest-path-filled HMM route) is given, the sequence-level arc-set / LCS metrics use it
    instead of the per-fix arcs.
    """
    m = obs.merge(pred, on=["trip", "obs"], how="left")
    m["pred_arc"] = m["pred_arc"].fillna(-1).astype(np.int64)
    correct = m["pred_arc"].to_numpy() == m["ref_arc"].to_numpy()
    key_ref = m["trip"].to_numpy(np.int64) * 10_000_000 + m["ref_arc"].to_numpy()
    key_pred = np.where(m["pred_arc"].to_numpy() >= 0,
                        m["trip"].to_numpy(np.int64) * 10_000_000 + m["pred_arc"].to_numpy(), -1)
    res = {
        "top1_acc": float(correct.mean()),
        "top1_acc_given_ref_in_cand": float(correct[m["in_cand"].to_numpy()].mean()),
        "candidate_miss_rate": float(1.0 - m["in_cand"].mean()),
        "obs_macro_f1": float(f1_score(key_ref, key_pred, labels=np.unique(key_ref), average="macro", zero_division=0)),
    }
    cont = {h: [0, 0] for h in hops}
    ref_cont = {h: [0, 0] for h in hops}
    prec, rec, f1s, lcs = [], [], [], []
    per_trip = []
    for trip, g in m.groupby("trip", sort=True):
        g = g.sort_values("obs")
        pa, ra = g["pred_arc"].to_numpy(), g["ref_arc"].to_numpy()
        gr = graphs[int(trip)]
        row = {"trip": int(trip), "top1_acc": float((pa == ra).mean())}
        for h in hops:
            ok, tot = continuity(pa, gr, h)
            cont[h][0] += ok
            cont[h][1] += tot
            ok2, tot2 = continuity(ra, gr, h)
            ref_cont[h][0] += ok2
            ref_cont[h][1] += tot2
            row[f"continuity_h{h}"] = ok / max(tot, 1)
        route_seq = routes[int(trip)] if routes is not None and int(trip) in routes else pa[pa >= 0]
        sp, sr = set(np.asarray(route_seq).tolist()), set(ra.tolist())
        tp = len(sp & sr)
        p_ = tp / max(len(sp), 1)
        r_ = tp / max(len(sr), 1)
        f_ = 2 * p_ * r_ / max(p_ + r_, 1e-12)
        l_ = lcs_ratio(np.asarray(route_seq), ra)
        prec.append(p_), rec.append(r_), f1s.append(f_), lcs.append(l_)
        row.update(set_precision=p_, set_recall=r_, set_f1=f_, lcs_ratio=l_)
        per_trip.append(row)
    for h in hops:
        res[f"route_continuity_h{h}"] = cont[h][0] / max(cont[h][1], 1)
        res[f"reference_continuity_h{h}"] = ref_cont[h][0] / max(ref_cont[h][1], 1)
    res.update(seq_set_precision=float(np.mean(prec)), seq_set_recall=float(np.mean(rec)),
               seq_set_f1=float(np.mean(f1s)), seq_lcs_ratio=float(np.mean(lcs)))
    res["per_trip"] = per_trip
    return res
