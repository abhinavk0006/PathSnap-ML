"""Step 5: tune the HMM/Viterbi decoder and compare classifier-only vs classifier + HMM.

  * Decoder parameters are tuned (Optuna TPE) on out-of-fold classifier probabilities of a sample of TRAINING trips
    (probabilities produced by models that never saw those trips). Student-test and evaluator data are not used for
    tuning.
  * Evaluation: the 10 student-test trips, with the final tuned classifier from step 4.
Outputs: results/hmm/*.json|csv|parquet, models/hmm_config.json
"""
import _setup  # noqa: F401
import json
import time

import numpy as np
import optuna
import pandas as pd

from pathsnap.common import build_network, list_trips, load_config, project, read_trip, resolve, trip_epsg
from pathsnap.data import load_split
from pathsnap.hmm import Decoder, HMMParams
from pathsnap.metrics import ArcGraph, evaluate_sequence, top1_arcs

N_TRIALS = 40
N_TUNE_TRIPS = 20
MAX_TUNE_FIXES = 6000


def prep_trip(cfg, split_dir, tid):
    trip = read_trip(split_dir / f"{tid:08d}")
    net = build_network(trip)
    P = project(trip.lon, trip.lat, trip_epsg(trip))
    return trip, net, P


def decode_trip(dec, d):
    out = dec.decode(d, d["p"].to_numpy())
    return out


def main():
    cfg = load_config()
    raw = resolve(cfg, "raw_root")
    out_dir = resolve(cfg, "results") / "hmm"
    out_dir.mkdir(parents=True, exist_ok=True)
    mdir = resolve(cfg, "models")

    # ---------------- tuning on OOF probabilities of training trips
    oof = pd.read_parquet(resolve(cfg, "results") / "hpo" / "oof_probs.parquet")
    _, tobs = load_split(cfg, "train", quality_filter=True)
    sizes = tobs.groupby("trip").size()
    eligible = sizes[sizes <= MAX_TUNE_FIXES].index.to_numpy()
    rng = np.random.RandomState(cfg["seed"])
    tune_trips = sorted(rng.choice(eligible, min(N_TUNE_TRIPS, len(eligible)), replace=False).tolist())
    print("tuning trips:", tune_trips, flush=True)
    cache = {}
    for tid in tune_trips:
        trip, net, P = prep_trip(cfg, raw / cfg["paths"]["train_dir"], tid)
        d = oof[oof.trip == tid].sort_values("obs").reset_index(drop=True)
        dec = Decoder(net, P, trip.t, d["arc"].to_numpy(), HMMParams.from_dict(cfg["hmm"]))
        cache[tid] = (dec, d, tobs[tobs.trip == tid])

    def score(params: HMMParams):
        ok = tot = 0
        for tid, (dec, d, o) in cache.items():
            dec.p = params
            out = dec.decode(d, d["p"].to_numpy())
            m = o[["obs", "ref_arc"]].merge(out.rename(columns={"arc": "pred"}), on="obs", how="left")
            ok += int((m["pred"].fillna(-1).to_numpy() == m["ref_arc"].to_numpy()).sum())
            tot += len(m)
        return ok / tot

    # classifier-only reference on the tuning trips
    base_ok = tot = 0
    for tid, (dec, d, o) in cache.items():
        pred = top1_arcs(d.assign(trip=tid), d["p"].to_numpy()).rename(columns={"pred_arc": "pred"})
        m = o[["obs", "ref_arc"]].merge(pred[["obs", "pred"]], on="obs", how="left")
        base_ok += int((m["pred"].fillna(-1).to_numpy() == m["ref_arc"].to_numpy()).sum())
        tot += len(m)
    clf_tune_acc = base_ok / tot
    default_acc = score(HMMParams.from_dict(cfg["hmm"]))
    print(f"tuning set top-1: classifier-only {clf_tune_acc:.4f} | HMM defaults {default_acc:.4f}", flush=True)

    def objective(trial):
        p = HMMParams(
            beta_m=trial.suggest_float("beta_m", 3.0, 60.0, log=True),
            emission_weight=trial.suggest_float("emission_weight", 0.2, 4.0, log=True),
            gap_penalty=trial.suggest_float("gap_penalty", 3.0, 40.0),
            uturn_penalty=trial.suggest_float("uturn_penalty", 0.0, 15.0),
            back_penalty=trial.suggest_float("back_penalty", 0.0, 5.0),
        )
        return score(p)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    study.trials_dataframe().to_csv(out_dir / "decoder_trials.csv", index=False)
    best = HMMParams.from_dict({**cfg["hmm"], **study.best_params})
    print(f"best decoder params {study.best_params} -> tuning-set top-1 {study.best_value:.4f} ({time.time() - t0:.0f}s)", flush=True)
    (mdir / "hmm_config.json").write_text(json.dumps({
        "params": best.to_dict(), "tuned_on": f"{len(tune_trips)} training trips (out-of-fold probabilities)",
        "tuning_trips": tune_trips, "objective": "pooled observation-level Top-1 accuracy", "n_trials": N_TRIALS,
        "tuning_set_top1": {"classifier_only": clf_tune_acc, "hmm_defaults": default_acc, "hmm_tuned": study.best_value}}, indent=2))

    # ---------------- evaluation on the student-test trips
    scand, sobs = load_split(cfg, "student")
    ps = np.load(resolve(cfg, "results") / "models" / "student_probs_tuned.npy")
    scand = scand.assign(p=ps)
    graphs, pred_clf, pred_def, pred_tuned, routes_def, routes_tuned = {}, [], [], [], {}, {}
    stats = {}
    for d_ in list_trips(raw / cfg["paths"]["student_dir"]):
        tid = int(d_.name)
        trip, net, P = prep_trip(cfg, raw / cfg["paths"]["student_dir"], tid)
        graphs[tid] = ArcGraph(trip.arcs[:, 0], trip.arcs[:, 1])
        c = scand[scand.trip == tid].sort_values("obs").reset_index(drop=True)
        pred_clf.append(top1_arcs(c, c["p"].to_numpy()))
        for tag, params, store, rstore in (("default", HMMParams.from_dict(cfg["hmm"]), pred_def, routes_def),
                                           ("tuned", best, pred_tuned, routes_tuned)):
            dec = Decoder(net, P, trip.t, c["arc"].to_numpy(), params)
            t1 = time.time()
            o = dec.decode(c, c["p"].to_numpy())
            dt = time.time() - t1
            store.append(o.assign(trip=tid).rename(columns={"arc": "pred_arc"})[["trip", "obs", "pred_arc"]])
            rstore[tid] = dec.fill_route(o.set_index("obs").reindex(np.arange(len(P)))["arc"].fillna(-1).to_numpy().astype(np.int64))
            stats[f"{tag}_{tid}"] = {**dec.stats, "decode_s": dt}
    res = {}
    for tag, pred, routes in (("classifier_only", pd.concat(pred_clf), None),
                              ("hmm_default_params", pd.concat(pred_def), routes_def),
                              ("hmm_tuned_params", pd.concat(pred_tuned), routes_tuned)):
        r = evaluate_sequence(sobs, pred, graphs, routes=routes)
        res[tag] = r
    per_trip = []
    for tag in res:
        for row in res[tag]["per_trip"]:
            per_trip.append({"method": tag, **row})
    for tag in res:
        res[tag].pop("per_trip")
    pd.DataFrame(per_trip).to_csv(out_dir / "student_per_trip.csv", index=False)
    (out_dir / "student_summary.json").write_text(json.dumps({"summary": res, "decode_stats": stats}, indent=2))
    allp = pd.concat([pd.concat(pred_clf).rename(columns={"pred_arc": "clf_arc"})
                      .merge(pd.concat(pred_tuned).rename(columns={"pred_arc": "hmm_arc"}), on=["trip", "obs"], how="outer")])
    allp.to_parquet(out_dir / "student_predictions.parquet", index=False)
    print(json.dumps({k: {m: round(v, 4) for m, v in r.items()} for k, r in res.items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
