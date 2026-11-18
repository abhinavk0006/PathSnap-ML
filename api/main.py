"""PathSnap FastAPI backend (runs on Laptop 1).

  GET  /health          liveness + model info
  GET  /trips           ids of the trips available for evaluation (ids only, no labels)
  POST /predict         {"trip_id": "00000012"} -> full pipeline (features -> classifier -> HMM/Viterbi), then server-side
                        scoring against the reference route kept on this machine. Returns the predicted route, the raw
                        GPS fixes and the metrics. The reference route / labels are NEVER included in a response.
  GET  /predict/{id}    same as POST (convenient for a browser/curl check)

Configuration (environment variables):
  PATHSNAP_TRIP_DIR   folder that contains the trip sub-folders (default: data/evaluator from configs/config.yaml)
  PATHSNAP_MODELS_DIR folder with final_model.joblib / model_meta.json / hmm_config.json (default: models/)
  PATHSNAP_CORS       comma-separated allowed origins for the UI laptop (default "*")
Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Every request is printed to the console (uvicorn access log + one PathSnap line) and appended to logs/api_requests.log.
"""
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.common import list_trips, load_config, resolve  # noqa: E402
from pathsnap.pipeline import PathSnapPredictor, evaluate_prediction, route_geometry  # noqa: E402

(ROOT / "logs").mkdir(exist_ok=True)
log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / "logs" / "api_requests.log")):
        h.setFormatter(_fmt)
        log.addHandler(h)

STATE: dict = {}
TRIP_ID = re.compile(r"^\d{1,8}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    d = os.environ.get("PATHSNAP_TRIP_DIR")
    trip_dir = Path(d) if d else resolve(cfg, "evaluator_dir")
    if not trip_dir.is_absolute():
        trip_dir = ROOT / trip_dir
    STATE["cfg"] = cfg
    STATE["trip_dir"] = trip_dir
    md = os.environ.get("PATHSNAP_MODELS_DIR")
    STATE["predictor"] = PathSnapPredictor(cfg, Path(md) if md else None)
    log.info(f"startup | model={STATE['predictor'].meta['model']} | trip_dir={trip_dir} | "
             f"{len(list_trips(trip_dir)) if trip_dir.exists() else 0} trips found")
    yield


app = FastAPI(title="PathSnap map-matching API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("PATHSNAP_CORS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()
    client = request.client.host if request.client else "?"
    log.info(f"REQUEST  {rid} | {client} | {request.method} {request.url.path}")
    response = await call_next(request)
    log.info(f"RESPONSE {rid} | status={response.status_code} | {1000 * (time.time() - t0):.0f} ms")
    return response


class PredictRequest(BaseModel):
    trip_id: str


def _trip_path(trip_id: str) -> Path:
    if not TRIP_ID.match(trip_id):
        raise HTTPException(400, "trip_id must be 1-8 digits")
    p = STATE["trip_dir"] / f"{int(trip_id):08d}"
    if not p.is_dir():
        raise HTTPException(404, f"trip {trip_id} not found on this server")
    return p


def _run(trip_id: str) -> dict:
    d = _trip_path(trip_id)
    t0 = time.time()
    res = STATE["predictor"].predict(d)                      # no labels used
    ev = evaluate_prediction(d, res, STATE["cfg"])           # server-side scoring against the reference route
    trip = res["_trip"]
    n = res["n_fixes"]
    step = max(1, n // 4000)
    h, c = ev["classifier_plus_hmm"], ev["classifier_only"]
    keys = ["top1_acc", "obs_macro_f1", "candidate_miss_rate", "route_continuity_h1", "route_continuity_h3",
            "reference_continuity_h3", "seq_set_precision", "seq_set_recall", "seq_set_f1", "seq_lcs_ratio"]
    body = {
        "trip_id": f"{trip.trip_id:08d}", "n_fixes": n, "n_route_arcs": int(len(res["route_arcs"])),
        "metrics": {"classifier_plus_hmm": {k: h[k] for k in keys}, "classifier_only": {k: c[k] for k in keys}},
        "hmm_stats": res["stats"], "timing_s": {**res["timing"], "request_total_s": time.time() - t0},
        "gps": [[round(float(trip.lat[i]), 6), round(float(trip.lon[i]), 6)] for i in range(0, n, step)],
        "gps_downsample_step": step,
        "predicted_route": [[[p[1], p[0]] for p in seg] for seg in route_geometry(res)],  # [lat, lon] pairs for Leaflet
        "predicted_arc_per_fix": [int(a) for a in res["hmm_arc"]],
        "model": STATE["predictor"].meta["model"],
    }
    log.info(f"PREDICT  trip={body['trip_id']} fixes={n} | HMM top1={h['top1_acc']:.4f} (classifier-only {c['top1_acc']:.4f}) | "
             f"continuity(h=3)={h['route_continuity_h3']:.4f} | pipeline {res['timing']['total_s']:.2f}s")
    return body


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["predictor"].meta["model"], "hmm_params": STATE["predictor"].hmm.to_dict(),
            "trip_dir": str(STATE["trip_dir"])}


@app.get("/trips")
def trips():
    d = STATE["trip_dir"]
    ids = [p.name for p in list_trips(d)] if d.exists() else []
    return {"trips": ids, "count": len(ids)}


@app.post("/predict")
def predict(req: PredictRequest):
    return _run(req.trip_id)


@app.get("/predict/{trip_id}")
def predict_get(trip_id: str):
    return _run(trip_id)
