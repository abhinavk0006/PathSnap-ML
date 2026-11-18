# Metric definitions (all computed in `src/pathsnap/metrics.py`)

| Metric | Definition |
|---|---|
| Candidate-level Macro F1 | Every (GPS fix, candidate arc) row is a binary example (1 = reference arc of that fix). Macro F1 = mean of the F1 of class 0 and class 1 at threshold 0.5. |
| Log loss | Binary cross-entropy of the candidate probability (probabilities clipped to [1e-7, 1-1e-7]). |
| Observation-level Top-1 accuracy | For each fix take the highest-scoring candidate and compare with the reference arc. If the reference arc is not among the candidates the fix counts as wrong (candidate miss). |
| Candidate miss rate | Share of fixes whose reference arc is absent from the candidate set. |
| Observation-level Macro F1 | Multiclass F1 (class = trip-qualified arc id) averaged over the reference arcs, predictions = Top-1 arcs. |
| Route continuity (h) | Share of consecutive fix pairs whose predicted arcs are identical or connected within h arc steps. h=1 is strict adjacency (end node of arc a = start node of arc b); h=3 tolerates arcs skipped between two fixes. The same measure on the reference arcs (`reference_continuity_h*`) is the ceiling, because at ~35 m/s a vehicle can skip short arcs between two 1 Hz fixes. |
| Sequence-level (arc-set P/R/F1) | Unique arcs of the predicted route versus the unique arcs of the reference route. For the HMM the predicted route is the decoded arc sequence with the shortest-path arcs between non-adjacent decoded arcs inserted (`Decoder.fill_route`). |
| Route LCS ratio | `difflib` similarity of the de-duplicated predicted and reference arc sequences (order-aware). |
| Training time / inference time | Wall-clock seconds for the final fit on all training rows; inference = `predict_proba` time per 1000 GPS fixes on the student-test trips. |

## How reference labels per fix are produced
The `.route` file lists the arcs travelled, in order, but not which fix lies on which arc. `features.align_to_route`
finds the non-decreasing assignment of fixes to route arcs that minimises the summed (capped at 60 m) distance
between each fix and its arc (dynamic programming). This is an assumption, not independent ground truth.
