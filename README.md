# PathSnap: distributed map-matching and route reconstruction

PathSnap matches noisy 1 Hz GPS fixes to a road network. Pipeline:

```
raw trip folder (.track .nodes .arcs)
  -> candidate arcs per GPS fix (all arcs within 60 m, K<=12 nearest)
  -> per-(fix, arc) features (distance, heading, neighbouring-fix fit, arc topology, speed, ...)
  -> classifier: P(this arc is the road the vehicle is on)
  -> HMM / Viterbi over the candidate arcs (network-distance transitions) -> arc per fix -> route
```

It runs as two processes: a **FastAPI backend** (Laptop 1: model, HMM, road data, reference labels) and a
**browser UI** (Laptop 2: lists the evaluation trips, sends the chosen trip id, draws the predicted route and the
server-computed metrics).

## Repository layout

| Path | Contents |
|---|---|
| `configs/config.yaml` | all paths and parameters (candidate radius, features, CV, label-quality filter, HMM defaults) |
| `src/pathsnap/` | library: `common.py` (IO, projection, geometry), `features.py` (candidates, features, label alignment), `modelzoo.py`, `data.py`, `metrics.py`, `hmm.py` (HMM/Viterbi), `pipeline.py` (inference) |
| `scripts/01_eda.py` | dataset inventory, missing/invalid-value analysis, network connectivity, EDA figures |
| `scripts/02_build_dataset.py` | candidate generation + features + reference alignment for train and student_test |
| `scripts/03_train_compare.py` | 8 models, grouped 5-fold CV, student-test evaluation, loss curves |
| `scripts/04_bayes_opt.py` | Optuna (TPE) Bayesian optimisation of the selected model, refit, final model export |
| `scripts/05_hmm_decode.py` | HMM parameter tuning, classifier-only vs classifier + HMM comparison |
| `scripts/06_folium_maps.py` | interactive Folium maps (+ static PNG export) |
| `scripts/07_make_figures.py` | comparison / tuning / HMM tables and plots from saved results |
| `scripts/08_summary.py` | writes `results/SUMMARY.md` from the saved result files |
| `api/main.py` | FastAPI backend |
| `ui/` | evaluator UI (`index.html`, `serve.py`, vendored Leaflet, works offline except map tiles) |
| `models/` | `final_model.joblib`, `model_meta.json`, `hmm_config.json` |
| `results/` | `eda/`, `models/` (comparison table, curves), `hpo/` (trials, best params), `hmm/`, `maps/`, `figures/` |
| `docs/` | `metrics.md` (metric definitions), `decisions.md` (design decisions and data findings) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the provided dataset so the folders `train/` and `student_test/` sit under `data/raw/`
(`data/raw/train/00000001/00000001.track`, ...). The reserved evaluation trips go in `data/evaluator/` on Laptop 1
only; they are never read by any training or tuning script.

## Reproduce the experiments

```bash
cd scripts
python 02_build_dataset.py     # ~1 min
python 01_eda.py               # EDA tables + figures
python 03_train_compare.py     # model comparison (all seeds fixed to 42)
python 04_bayes_opt.py         # tunes the best boosting model, writes models/final_model.joblib
python 05_hmm_decode.py        # tunes the decoder, writes models/hmm_config.json
python 06_folium_maps.py       # maps -> results/maps/*.html
python 07_make_figures.py && python 08_summary.py
```

## Deployment (two laptops)

**Laptop 1 (API).** Put the reserved trips in `data/evaluator/<trip_id>/<trip_id>.{track,nodes,arcs,route}`, then:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# for a rehearsal with the student-test trips instead:
#   PATHSNAP_TRIP_DIR=data/raw/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open port 8000 in the firewall and note the laptop's IP (`ipconfig` / `ip a`). The console shows every request
(`REQUEST`, `PREDICT`, `RESPONSE` lines) and the same lines are appended to `logs/api_requests.log`; this is the log
to show the evaluator.

**Laptop 2 (UI).**

```bash
python ui/serve.py --port 8080 --api http://<LAPTOP-1-IP>:8000
# open http://localhost:8080
```

The UI lists the trips returned by `GET /trips`, sends `POST /predict {"trip_id": "..."}` and displays the route and the
metrics the API computed. Reference routes stay on Laptop 1 and are never part of an API response.

Quick checks from any machine: `curl http://<LAPTOP-1-IP>:8000/health`, `curl http://<LAPTOP-1-IP>:8000/trips`.

## Data findings that shaped the implementation

* **The dataset is global** (trips fall in UTM zones 14-53), so a single projection is wrong for most trips. Every trip
  is projected to the UTM zone of its own centroid (`common.utm_epsg`) and all distances are in metres.
* **Some reference routes do not follow their track** (trips 33 and 54: the route covers 6 km / 31 km while the track
  is 48 km / 170 km). Training-only filter: trips whose median fix-to-reference distance exceeds 30 m are dropped and
  individual fixes more than 60 m from their reference arc are ignored. Student-test trips are evaluated unfiltered.
* **Timestamp gaps are common** (40 of 80 training trips contain a gap above 10 s, the largest is 153,064 s), so the HMM
  has explicit gap handling (see `src/pathsnap/hmm.py`).
* Networks are per trip, usually disconnected into several components, with no isolated nodes and no NaNs.
* The `.route` file does not say which fix lies on which arc; labels come from a monotone alignment
  (`features.align_to_route`), documented in `docs/metrics.md`. These labels are reference information, not verified
  ground truth.

See `docs/decisions.md` for the reasoning behind the main choices.

## Integrity notes

* The evaluator partition is not used for training, feature selection, model choice, HMM tuning or thresholds.
* No predictions or answers are hard-coded; `api/main.py` runs the full pipeline for every request.
* No credentials or keys are stored in this repository.
* AI assistance: parts of this code base were drafted with an AI coding assistant. Add the disclosure required by the
  event's policy here, and make sure every team member can explain the code and results before submission.


## Team contributions

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the team responsibility allocation.
