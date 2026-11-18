# PathSnap architecture iteration 11: breadth-first and depth-first design review

18 November 2026. Proposal grounded in structural inspection of the supplied dataset. No model training or performance evaluation has occurred.

This applies BFS and DFS as design-review strategies, not as replacements for HMM/Viterbi or as road-graph traversal experiments. It extends PROPOSED_ARCHITECTURE.md; the first ten reviews remain the history of the initial proposal.

## Review method

**Breadth-first pass:** start from the system goal, visit every first-level subsystem, then inspect each subsystem's inputs, outputs, dependencies and failure policy. Do not optimise one component before checking the others. Maintain a queue of unresolved design questions.

**Depth-first pass:** take the highest-impact unresolved question, trace its dependencies down to a concrete failure or decision, compare plausible alternatives, then backtrack and propagate the chosen change to the wider design. Stop a branch when it has a supported design decision, a measurable implementation check, or an explicit dependency on organiser clarification.

The final breadth-first pass checks that deep changes have not broken another component or the guidelines. This is a structured qualitative review; no exhaustive search of every possible architecture is claimed.

## Evidence from the supplied archive

Source: `C:\Users\Abhinav Krishna\Downloads\Student folder.zip`. Files were read directly from the ZIP without extraction or executing any contents. Per-trip findings are saved in `dataset_audit/structural_inventory.json`.

| Check | Observed result |
|---|---|
| Partitions | 80 train trips and 10 student_test trips; no evaluator partition present |
| Raw files | 90 each of .track, .nodes, .arcs and .route |
| GPS observations | 208,417 training; 19,160 student-testing |
| Reference-route entries | 31,336 training; 3,477 student-testing |
| Column counts | All files have the documented 3/2/2/1 columns respectively |
| Invalid node references in arcs | 0 |
| Invalid arc IDs in routes | 0 |
| Non-increasing adjacent timestamps | 0 |
| Equal GPS and route row counts | 0 of 90 trips |
| Distinct network byte hashes | 90; this does not prove geographical disjointness |
| Largest network by arc count | Trip 00000013: 155,003 nodes, 307,392 arcs |
| Largest timestamp gap | Training trip 00000090: 153,064 seconds, about 42.5 hours |
| Other very large gap | Training trip 00000078: 52,810 seconds, about 14.7 hours |
| Direct endpoint mismatches in consecutive reference arcs | 14 pairs across 11 trips |
| Fraction of arcs with a reverse endpoint pair | Ranges from about 35.1% to 100% across trips |

The gap values assume the documented seconds unit; their cause has not been established. An endpoint mismatch is not proof that two arcs are unreachable: intervening arcs, repeated entries or reference conventions may explain it. Reverse-arc statistics do not establish legal traffic direction. These are findings to investigate, not grounds for silently repairing the source data.

## BFS pass: inspect all subsystems at the same level

| Subsystem | Existing design | New finding or review question | Disposition |
|---|---|---|---|
| Ingestion | Partition manifest and validated loaders | Archive already supplies the required development split | Preserve folder assignment; fingerprint archive and files |
| Labels | Separate alignment/provenance component | None of the route lengths matches its GPS length | Promote alignment to a prerequisite for supervised experiments |
| Geometry | Per-trip road graph and metric coordinates | Networks differ in byte content and can be large | Keep trip-scoped IDs; lazy graph loading; validate coordinate extent before choosing projection |
| Candidates | Indexed nearby roads with bounded expansion | Full networks reach hundreds of thousands of arcs | Query spatially; avoid testing every arc for every point |
| Features | Shared pipeline with temporal features | Large gaps make adjacent-row motion features unreliable | Add segment boundaries and feature-validity masks |
| Model comparison | Five models on trip-grouped folds | Derived labels can create misleadingly good metrics | Report label provenance/coverage; add label-policy sensitivity checks |
| Bayesian search | Training-only grouped CV | Expensive geometry could be recomputed for every trial | Cache deterministic raw features by data/configuration version; keep fitted transforms fold-local |
| Decoder | Graph-aware log-space Viterbi | Very long gaps and reference non-adjacency need explicit policies | Segment long gaps and preserve unresolved route breaks |
| Evaluation | Candidate, observation and sequence metrics | Pointwise reference labels are not directly supplied | Name derived-label metrics accurately; retain sequence diagnostics |
| API and UI | Two laptops with isolated scorer | Long computation, partial routes and map payloads | Bound work, expose partial status and preserve separate route segments |
| Submission | Logs, maps and report generation | New data-dependent decisions need reproducible evidence | Retain audit script, inventory and an assumptions register |

Breadth-first queue, ordered for the depth-first pass: label alignment -> gap policy -> network interpretation -> routing cost -> evaluation honesty -> deployment integration.

## DFS branch 1: labels -> alignment -> validity of experiments

**Dependency path:** model selection -> trustworthy metrics -> observation labels -> chronological route-to-GPS alignment.

1. A .route entry identifies a traversed arc, not an explicitly timestamped GPS label.
2. Treating row i in .route as the label for row i in .track fails structurally in every trip.
3. Marking any arc appearing anywhere in the route as positive loses temporal meaning, especially on revisits.
4. Assigning each observation its nearest reference arc ignores traversal order and can confuse parallel sections or repeated passes.
5. Proposed approach: first check for organiser alignment tools or scoring rules. If absent, build an explicitly derived, order-aware alignment against the permitted reference route, distinguishing repeated occurrences of an arc. Segment around large GPS gaps and unresolved reference discontinuities. Record ambiguous/unresolved observations instead of forcing labels.
6. This alignment is for development supervision and scoring only. Candidate generation and deployed prediction never see the reference route.

**Decision:** the label builder becomes a gated prerequisite. A plausible implementation is monotone alignment over ordered route occurrences, allowing multiple observations on an occurrence and skipped occurrences when sampling is sparse. Its costs, start/end handling and rejection policy still require investigation; it is not yet a validated ground-truth generator.

**Backtrack consequence:** all five models and the tuned model use the same frozen label policy and coverage accounting. A label-policy sensitivity analysis is needed because features based on distance can agree with geometry-derived labels without proving physical accuracy. Do not select an alignment policy simply because it makes classifier scores better.

**Branch exit:** require a documented alignment method, plots of representative easy/ambiguous development cases and agreement with any official scoring contract. Final evaluator alignment must follow organiser rules and stay inside isolated scoring.

## DFS branch 2: timestamps -> segmentation -> decoder behaviour

**Dependency path:** route continuity -> transition plausibility -> elapsed time and motion features -> timestamp gaps.

1. Strictly increasing timestamps do not imply a continuous recording session.
2. A 42.5-hour interval makes a single uninterrupted trip assumption questionable.
3. Increasing a transition distance allowance in proportion to that interval could admit almost arbitrary routes.
4. Forcing a shortest connector across the gap would invent an unobserved journey.

**Alternatives:** decode every adjacent observation continuously; discard the whole affected trip; or segment the observation sequence while retaining all usable observations.

**Decision:** propose explicit segmentation. Determine and justify the gap rule using training data and its actual sampling distribution. Reset heading/speed windows and Viterbi initialisation at a segment boundary. Do not join separate segments into one continuous polyline. Keep original observation order and IDs.

**Backtrack consequence:** API results carry segment IDs and break reasons. Folium draws separate route pieces. Metrics report gap counts and both continuity coverage and continuity quality. Segmenting must not silently remove failed transitions from the denominator: report the all-valid-pairs rate and a clearly labelled within-segment diagnostic. Group all segments of the same original trip into the same CV fold.

**Branch exit:** a frozen gap policy, audited feature masks, decoder restart checks and visible map breaks. No threshold has been selected yet.

## DFS branch 3: network representation -> reference continuity -> travel rules

**Dependency path:** feasible transition -> graph paths -> arc direction and geometry -> raw file conventions.

1. Node and arc references are in bounds, so basic indexing is sound.
2. Networks differ at the byte level; global arc IDs cannot be assumed.
3. Reverse endpoint pairs are not universal, so automatically adding every reverse edge would change the supplied graph.
4. Fourteen consecutive reference pairs fail direct endpoint adjacency. This could reflect omissions, duplicate arc entries, direction conventions or reference issues; it does not by itself establish disconnected components.

**Decision:** preserve the supplied from/to arcs as the initial directed representation and keep `(trip_id, arc_id)` identity. Inspect the 14 cases to categorise their cause before changing graph semantics. Do not reverse arcs or add connectors merely to make a reference route appear continuous. Keep raw references and any derived interpretation separately versioned.

**Backtrack consequence:** candidate metadata must retain direction; label alignment needs a discontinuity flag; decoder diagnostics must distinguish no direct adjacency from no feasible graph path. Different byte hashes do not mean validation trips cover new geographical areas, so do not claim geographical generalisation without a spatial-overlap analysis.

**Branch exit:** directionality and reference conventions established from the data/organisers, with unresolved exceptions recorded. No graph BFS/DFS was run as part of this design-method request.

## DFS branch 4: scale -> transition costs -> bounded inference

**Dependency path:** responsive two-laptop demo -> inference cost -> candidate-pair transitions -> graph path queries.

1. With T observations and K candidates per observation, a dense adjacent-step decoder considers roughly T times K squared candidate transitions.
2. Computing a fresh full-network route for every transition can dominate inference time.
3. Materialising all-pairs road-node distances is particularly unsuitable for a graph with 155,003 nodes: it entails over 24 billion entries before overhead.

**Decision:** keep sparse graph adjacency, spatially query candidate segments, load trip graphs lazily and use bounded on-demand weighted path queries. Reuse node-to-node path results with trip/network identity in the cache key; calculate projected endpoint offsets separately. Limit cache memory. Never reuse a complete transition value across different projected positions or elapsed times without including them in its key.

**Alternative rejected at design level:** unbounded all-pairs precomputation. The appropriate path-search implementation and limits remain performance decisions to measure on training data.

**Backtrack consequence:** cache deterministic geometry/features across experiments but invalidate on data, projection, feature, label or candidate-policy changes as appropriate. Scalers/calibration remain training-fold fitted. Do not prune solely for speed without measuring candidate coverage and decoder effects.

**Branch exit:** measured per-stage latency/memory and matching results with caching on/off. No speed claim or fixed latency target yet.

## DFS branch 5: metrics -> partial results -> evaluator isolation

**Dependency path:** meaningful final score -> scoring contract -> label provenance and denominators -> safe API response.

1. Observation-level scores depend on the unresolved alignment contract.
2. Candidate misses, ambiguous references, invalid inputs and predictions interrupted by gaps are different situations.
3. Treating all of them as dropped rows would reward incomplete predictions.

**Decision:** separately report reference-label coverage, prediction coverage, candidate miss rate, classifier accuracy and decoded accuracy. Count a missing prediction as wrong where a usable label exists. Record unresolved reference labels outside the pointwise denominator and disclose their number. Keep route-sequence diagnostics, including breaks, alongside pointwise scores.

**Backtrack consequence:** the scorer returns aggregate scores and coverage metadata; the UI receives no reference arc labels or geometry. Development maps can show permitted references; evaluator maps cannot. Missing evaluation references produce scoring-unavailable, not zero or a placeholder accuracy. A separate final-scoring adapter accommodates organiser rules without changing prediction.

**Branch exit:** explicit metric definitions and response-schema tests. The absent evaluator dataset remains absent from development and cannot be reconstructed from the missing trip IDs.

## Final BFS pass: propagate changes across the architecture

```text
Supplied development split
  -> validate and fingerprint
  -> segment GPS with original trip/observation identities preserved
  -> build directed per-trip graph and spatial index
  -> generate reference-free candidates and raw features
  -> train/evaluate five models with trip-grouped folds
  -> Bayesian tuning within training folds
  -> shared emission adapter and segment-aware HMM/Viterbi
  -> development comparisons and versioned inference bundle

Permitted reference routes
  -> separately versioned order-aware alignment with ambiguity flags
  -> development labels and metric computation only

Laptop 2 UI -> trip ID -> Laptop 1 FastAPI
  -> shared segmentation/candidates/features/classifier/decoder
  -> fixed predicted segments
  -> isolated scorer using official final reference/scoring contract
  -> predicted geometry, aggregate scores, coverage and break status -> UI
```

Final cross-component checks:

- Segmentation is shared by training and serving; it never splits a trip across CV folds.
- Label alignment remains outside the prediction dependency tree.
- Caching never fits transforms on held-out data or shares values across unrelated networks.
- Directionality and partial arc positions survive candidate selection and decoding.
- Missing candidates and sequence breaks remain visible in metrics and maps.
- Five-model comparison, Bayesian optimisation, HMM/Viterbi, Folium, FastAPI, two physical laptops, public source and a consolidated evidence-based report remain required.

## Changes accepted into the proposal

1. Make reference alignment a blocking design dependency for trustworthy supervised experiments.
2. Add an explicit segmentation/feature-validity component before candidate scoring and decoding.
3. Treat direct reference-arc mismatches as audit cases rather than automatically repairing routes.
4. Specify bounded on-demand graph routing, lazy loading and memory-limited caches.
5. Add label-policy sensitivity and spatial-overlap caveats to evaluation claims.
6. Extend result contracts with segment IDs, break reasons and separately defined coverage measures.

These are design improvements, not empirical proof of accuracy or implementation completion. Next work should inspect reference conventions, timestamp anomalies and alignment examples before selecting parameters or training models.
