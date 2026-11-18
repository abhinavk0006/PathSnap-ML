# Design decisions and data findings

This file records what was decided and why, with pointers to the result files that back each statement.

## 1. Problem formulation
Map matching is cast as **candidate ranking**. Each GPS fix gets candidate road arcs (all arcs within 60 m, at most the 12
nearest; if none is that close, the single nearest arc within 500 m). A classifier scores every (fix, arc) pair, the
highest-scoring arc is the classifier-only prediction (observation-level Top-1), and an HMM/Viterbi decoder then chooses
a connected sequence of arcs using these scores. Candidate coverage (reference arc inside the candidate set) is reported
per trip in `results/eda/dataset_inventory.csv` (`candidate_coverage`) and as the *candidate miss rate*.

## 2. Coordinates
The dataset is global (UTM zones 14-53). Each trip is projected to the UTM zone of its own centroid
(`common.utm_epsg`); each trip spans less than about 1 degree so scale distortion stays around 0.1%. A fixed zone
(first attempt: UTM 32N) produced non-finite values for trip 57 (Ecuador) and distorted distances elsewhere, which
is how the problem was found.

## 3. Reference labels
`.route` gives the ordered arcs but not the fix-to-arc assignment. `features.align_to_route` assigns fixes to route
arcs with a monotone dynamic programme (minimise summed capped distance). This is an assumption.

### Label-quality filter (training data only)
* Trips 33 and 54: the reference route is 6 km / 31 km long while the GPS track is 48 km / 170 km, and fixes sit a
  median 122 m / 142 m from the route (and 84 m from the nearest arc of *any* kind in trip 54). Their labels do not
  describe the track, so they are excluded from training and CV (`quality.max_trip_median_ref_dist_m = 30`).
* Individual fixes more than 60 m from their aligned reference arc are ignored in training/CV
  (`quality.max_fix_ref_dist_m`).
* The student-test partition is evaluated without this filter, so reported student-test numbers include any label noise.

## 4. Features
Distances to the arc, projection position, arc length/topology (degrees, reverse twin), speed, time gaps, heading
relative to the arc, the fit of neighbouring fixes (+-1, 2, 4) to the same arc, and progress along the arc.
**Long-baseline heading** (`hcos_b/hsin_b/hcos_f/hsin_f`: direction measured to the nearest fix at least 15 m behind /
ahead) was added after error analysis of the first model family: 44% of the student-test errors were *direction* errors
(the opposite-direction twin of the correct road), concentrated at low speed, where a heading taken from +-2 fixes is
noise. The first feature set's results are kept in `results/models/v1_basic_features/` for comparison.

## 5. Model comparison protocol
Eight models from six families (linear, probabilistic, tree, bagging, boosting x3, neural network). Grouped 5-fold CV
(trips never split across folds) on the quality-filtered training trips, then a final fit on all training rows and
evaluation on the ten student-test trips. Seeds are fixed (42). KNN/SVM were not run: with about 1.4 M candidate rows
and 38 features their prediction/training cost on the available hardware (2 CPU cores) was not practical.
Selection metric: **CV Top-1 accuracy** (what the decoder consumes), with CV Macro F1, log loss, stability across
folds, inference time and calibration as secondary criteria (`results/models/comparison_table.csv`).

## 6. Bayesian optimisation
Optuna TPE (seed 42, 6 random start-up trials), objective = mean CV Top-1, 3 group folds inside the search (to bound
compute on 2 cores), 20 trials. Search spaces: see `scripts/04_bayes_opt.py` and `results/hpo/best_params.json`.
Only training trips are used inside the search. The tuned model is then re-evaluated with the same 5-fold protocol
and on the student-test trips, and compared with the untuned model (`results/hpo/tuned_vs_untuned.json`).

## 7. HMM / Viterbi
See the docstring of `src/pathsnap/hmm.py`. Hidden state = candidate arc of the fix; emission = per-fix normalised
classifier probability; transition = network-distance consistency (`|d_route - d_straight| / beta`), U-turn and
backward-motion penalties; disconnected jumps are allowed with a fixed penalty and counted; long time gaps widen
the search budget and halve the penalty. Decoder parameters are tuned on out-of-fold classifier probabilities of 20
training trips (`scripts/05_hmm_decode.py`), never on student-test or evaluator data.

## 8. Known limitations
* Reference labels come from an alignment, not independent ground truth.
* Roads are straight arcs between nodes (no shape points), so curved roads add geometric error.
* Networks contain parallel carriageways; GPS noise can make the two sides hard to separate.
* HMM decoding is not guaranteed to help: see `results/hmm/student_per_trip.csv` for trips where it did not.
