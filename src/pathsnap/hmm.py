"""HMM / Viterbi route reconstruction on top of the classifier's candidate probabilities.

Model
  * Hidden state at GPS fix i : one of the candidate road arcs generated for that fix (the candidate-to-arc
    correspondence is kept, so the decoded state IS a road arc id).
  * Emission  : log P(arc | fix) = emission_weight * log( p_arc / sum_k p_k ), with p the classifier probability.
    Normalising over the candidates of one fix turns the independent binary scores into a per-fix distribution.
  * Transition: network-distance model in the spirit of Newson & Krumm (2009). The driving distance d_route between
    the projections of two consecutive fixes on their candidate arcs (remaining length on the previous arc +
    shortest path between the arcs + offset on the next arc) should be close to the straight-line distance
    d_gc of the two fixes:  log T = -|d_route - d_gc| / beta_eff, with beta_eff = beta * sqrt(dt).
  * Invalid transitions / disconnected segments: if the next arc is not reachable inside the search budget
    (ratio * d_gc + base_m, widened by 40 m/s * dt for gaps longer than max_gap_s), the transition is NOT forbidden;
    it gets a fixed log penalty -gap_penalty (half of it across long gaps). Viterbi therefore never breaks, and every
    such jump is counted in the decoding statistics. Reachable but poor transitions are capped at -gap_penalty so a
    disconnected jump is never cheaper than a connected-but-inconsistent one.
  * U-turns onto the opposite-direction twin arc pay uturn_penalty; moving backwards on the same arc by more than
    GPS jitter pays back_penalty.
  * Fixes with no candidate at all (nothing within the fallback radius) are skipped by the chain; they get arc -1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .common import Network


@dataclass
class HMMParams:
    beta_m: float = 15.0
    emission_weight: float = 1.0
    gap_penalty: float = 12.0
    uturn_penalty: float = 4.0
    back_penalty: float = 1.0
    ratio: float = 2.5
    base_m: float = 150.0
    corridor_m: float = 400.0
    max_gap_s: float = 10.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Decoder:
    def __init__(self, net: Network, P: np.ndarray, t: np.ndarray, cand_arcs: np.ndarray, params: HMMParams):
        self.net, self.P, self.t, self.p = net, P, t, params
        # corridor sub-graph: arcs near the track (+ every candidate arc) -> keeps shortest-path searches small
        mid = 0.5 * (net.A + net.B)
        near = cKDTree(P).query(mid, distance_upper_bound=params.corridor_m)[0] < np.inf
        near[np.unique(cand_arcs)] = True
        self.arc_ids = np.flatnonzero(near)
        fr, to = net.frm[self.arc_ids], net.to[self.arc_ids]
        nodes, inv = np.unique(np.concatenate([fr, to]), return_inverse=True)
        self.node_ids = nodes
        self.node_local = {int(n): i for i, n in enumerate(nodes)}
        lf, lt = inv[: len(fr)], inv[len(fr):]
        w = net.L[self.arc_ids]
        key = lf.astype(np.int64) * len(nodes) + lt
        order = np.lexsort((w, key))  # keep the shortest of parallel arcs
        key_s = key[order]
        first = np.r_[True, key_s[1:] != key_s[:-1]]
        sel = order[first]
        self.G = csr_matrix((np.maximum(w[sel], 1e-3), (lf[sel], lt[sel])), shape=(len(nodes), len(nodes)))
        self.arc_of = {(int(a), int(b)): int(i) for a, b, i in zip(fr, to, self.arc_ids)}
        self.stats = {"disconnected_jumps": 0, "transitions": 0, "skipped_fixes": 0}

    # ------------------------------------------------------------------ transitions
    def _trans(self, i, k, a_prev, t_prev, a_next, t_next):
        net, p = self.net, self.p
        gc = float(np.linalg.norm(self.P[k] - self.P[i]))
        dt = max(float(self.t[k] - self.t[i]), 1.0)
        gap = dt > p.max_gap_s
        lim = p.ratio * max(gc, 5.0) + p.base_m + (40.0 * dt if gap else 0.0)
        La, Lb = net.L[a_prev], net.L[a_next]
        rem_a = (1.0 - t_prev) * La
        head_b = t_next * Lb
        src_nodes = np.array([self.node_local[int(n)] for n in net.to[a_prev]])
        dst_nodes = np.array([self.node_local[int(n)] for n in net.frm[a_next]])
        usrc, inv = np.unique(src_nodes, return_inverse=True)
        D = dijkstra(self.G, directed=True, indices=usrc, limit=lim)
        sp = D[inv][:, dst_nodes]                                   # (Kp, Kn) path length between arc end/start nodes
        d_route = rem_a[:, None] + sp + head_b[None, :]
        reach = np.isfinite(sp)
        # same arc: progress along the arc
        same = a_prev[:, None] == a_next[None, :]
        if same.any():
            delta = (t_next[None, :] - t_prev[:, None]) * La[:, None]
            d_route = np.where(same, np.abs(delta), d_route)
            reach = reach | same
        beta = p.beta_m * np.sqrt(dt)
        logT = -np.abs(d_route - gc) / beta
        logT = np.maximum(logT, -p.gap_penalty)
        logT = np.where(reach, logT, -p.gap_penalty * (0.5 if gap else 1.0))
        if same.any():
            back = np.maximum(-(t_next[None, :] - t_prev[:, None]) * La[:, None] - 8.0, 0.0)  # tolerate 8 m of jitter
            logT = np.where(same, logT - p.back_penalty * np.minimum(back / 10.0, 3.0), logT)
        rev = (net.frm[a_prev][:, None] == net.to[a_next][None, :]) & (net.to[a_prev][:, None] == net.frm[a_next][None, :])
        logT = np.where(rev & ~same, logT - p.uturn_penalty, logT)
        return logT, reach

    # ------------------------------------------------------------------ Viterbi
    def decode(self, df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
        """df: candidate table sorted by obs with columns obs, arc, t_frac. Returns DataFrame[obs, arc]."""
        p = self.p
        obs = df["obs"].to_numpy()
        arcs = df["arc"].to_numpy()
        tf = df["t_frac"].to_numpy(np.float64)
        pr = np.asarray(prob, dtype=np.float64)
        starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
        ends = np.r_[starts[1:], len(obs)]
        n = len(starts)
        if n == 0:
            return pd.DataFrame({"obs": [], "arc": []})
        em = np.empty(len(obs))
        for s, e in zip(starts, ends):
            q = pr[s:e] + 1e-9
            em[s:e] = p.emission_weight * np.log(q / q.sum())
        delta = em[starts[0]:ends[0]].copy()
        back = [None]
        reach_log = [None]
        for s in range(1, n):
            ps, pe = starts[s - 1], ends[s - 1]
            cs, ce = starts[s], ends[s]
            logT, reach = self._trans(int(obs[ps]), int(obs[cs]), arcs[ps:pe], tf[ps:pe], arcs[cs:ce], tf[cs:ce])
            sc = delta[:, None] + logT
            bp = np.argmax(sc, axis=0)
            delta = sc[bp, np.arange(sc.shape[1])] + em[cs:ce]
            back.append(bp)
            reach_log.append(reach)
        path = np.empty(n, dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        for s in range(n - 1, 0, -1):
            path[s - 1] = back[s][path[s]]
        # statistics on the chosen path
        for s in range(1, n):
            self.stats["transitions"] += 1
            if not reach_log[s][path[s - 1], path[s]]:
                self.stats["disconnected_jumps"] += 1
        sel = starts + path
        return pd.DataFrame({"obs": obs[sel], "arc": arcs[sel]})

    # ------------------------------------------------------------------ route fill
    def fill_route(self, arcs_seq: np.ndarray) -> np.ndarray:
        """De-duplicate the decoded arc sequence and insert the shortest-path arcs between non-adjacent
        consecutive arcs, giving a continuous route wherever the network allows it."""
        seq = [int(a) for a in arcs_seq if a >= 0]
        if not seq:
            return np.array([], dtype=np.int64)
        out = [seq[0]]
        for b in seq[1:]:
            a = out[-1]
            if b == a:
                continue
            if self.net.to[a] != self.net.frm[b]:
                ua, ub = self.node_local.get(int(self.net.to[a])), self.node_local.get(int(self.net.frm[b]))
                if ua is not None and ub is not None:
                    dist, pred = dijkstra(self.G, directed=True, indices=ua, return_predecessors=True, limit=1500.0)
                    if np.isfinite(dist[ub]):
                        nodes = [ub]
                        while nodes[-1] != ua:
                            nodes.append(int(pred[nodes[-1]]))
                        nodes = nodes[::-1]
                        for u, v in zip(nodes[:-1], nodes[1:]):
                            arc = self.arc_of.get((int(self.node_ids[u]), int(self.node_ids[v])))
                            if arc is not None:
                                out.append(arc)
            out.append(b)
        return np.asarray(out, dtype=np.int64)
