# Chapter Overview — chapter_hyperparameter-optimization

Of the five book sections in this directory, only three carry a `## Exercises`
heading: `hyperopt-intro.md`, `hyperopt-api.md`, `rs-async.md`. `sh-intro.md`
and `sh-async.md` are slide-deck-only files (checked directly: no `##
Exercises` string anywhere) and are out of scope here. Best external sources
found: Bergstra & Bengio (2012) — already cited in-text — for the grid-vs-
random claim; Stanford CS231n's classification notes for train/val/test
discipline; Ray (Anyscale)'s own `tune_exercises/exercise_2_optimize.ipynb`
notebook, the single closest thing to a "graded HPO homework" we could
verify anywhere online; and MIT 6.5840's MapReduce lab for the
straggler/timeout problem `rs-async.md` motivates. The **AutoML book**
(Feurer & Hutter, Ch. 1) was fetched and searched directly (pdftotext) and
contains **zero exercises** — it is a research monograph, not a textbook.
The Freiburg/Hannover AutoML course runs weekly exercise sheets but they sit
behind a password-gated GitHub Classroom + AI-Campus platform we could not
read. Net finding: **there is essentially no public, verifiable course-
exercise tradition specific to HPO-API design, async random search, or
even random search itself** — the one rich exercise tradition we found
(Lattimore & Szepesvári's free bandit-algorithms textbook, Ch. 33) is about
*successive halving*, i.e. the topic of `sh-intro.md`/`sh-async.md`, which
have no `## Exercises` heading and are therefore out of scope. `hyperopt-
api.md`'s existing pair of exercises is already excellent (confirmed against
the prior style review) and is kept essentially as-is; the other two
sections each carry one weak sub-item (a bare "read this paper" pointer in
`hyperopt-intro.md`; an unmetriced comparison ask in `rs-async.md`) that we
rewrite. Totals: 3 sections, 7 current exercises (keep 5, rewrite 2, drop 0),
17 proposed problems.

---

## chapter_hyperparameter-optimization/hyperopt-intro.md — What Is Hyperparameter Optimization?

**Topic:** Defining HPO as a global black-box optimization problem (objective
function + configuration space) and implementing random search as the first
baseline algorithm, on the FashionMNIST softmax-regression example.
**Current exercises:** 3; disposition: keep 2, rewrite 1 — Exercises 1
(validation/test-set misuse) and 3 (grid-vs-random rationale) are solid,
concrete, and checkable; Exercise 2 is mostly strong but its final sub-item
("*Advanced*: Read :cite:`maclaurin-icml15` for an elegant... approach")
is a bare reading pointer with no task or deliverable (flagged in the prior
style review) and its first sub-item ("Convince yourself...") lacks a stated
check — both are folded into a rewritten version that keeps the good
sub-items (metric choice, computational-graph sketch, memory estimate,
gradient-pathology question) and replaces the two weak ones with a task that
still touches the same citation.

**External sources found:**
- Bergstra, J. & Bengio, Y., "Random Search for Hyper-Parameter Optimization," *JMLR* 13 (2012) — already cited in this section (`:cite:bergstra-jmlr12a`); shows empirically/theoretically that random search dominates grid search when only a few hyperparameters matter for a given dataset (Gaussian-process analysis of effective dimensionality) — no course exercises attached to the paper itself, but its central comparison is directly replicable at small scale. — https://jmlr.org/papers/v13/bergstra12a.html
- Fei-Fei Li, Andrej Karpathy et al., Stanford CS231n, "Image Classification: Data-driven Approach" course notes (ongoing) — teaches that hyperparameters (there, *k* for kNN) must be tuned against a held-out *validation* split carved from the *training* set, never against the test set; walks through a concrete CIFAR-10 example (49,000 train / 1,000 val) and a candidate list `k ∈ {1,3,5,10,20,50,100}`. — https://cs231n.github.io/classification/
- Feurer, M. & Hutter, F., "Hyperparameter Optimization," Ch. 1 of *Automated Machine Learning: Methods, Systems, Challenges* (Hutter, Kotthoff & Vanschoren, eds., Springer 2019, open access) — verified directly (downloaded PDF, ran `pdftotext`, grepped for "exercise") to contain **no end-of-chapter exercises** anywhere in the chapter; useful only as conceptual grounding (config-space types, log-scale parametrization) and as evidence for the "no tradition" finding. — https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book_Chapter1.pdf
- University of Freiburg/Hannover, "Automated Machine Learning" course (recurring; summer semesters 2022–2023 pages checked) — confirmed to run "roughly a new exercise sheet each week," but the sheets themselves live behind a password-protected GitHub Classroom + AI-Campus platform; no sheet text, numbering, or topics were retrievable. Noted as a finding (thin *public* tradition), not used as a source. — https://ml.informatik.uni-freiburg.de/teaching/summer-semester-2023/automated-machine-learning/

(Only 4 sources surfaced with verifiable content — this is itself the
chapter's headline finding: no strong external "HPO 101" exercise
tradition exists outside of general ML-course cross-validation exercises.)

**Proposed problem set** (our reference format):
1. [conceptual] **Validation Set vs. Test Set.** Read `Trainer.val_dataloader` and confirm (by tracing the code, not just re-stating the prose) that it wraps `FashionMNIST.val`, itself the original 10,000-example *test* set; write 3–4 sentences on why using it for HPO decisions contaminates the final accuracy estimate, referencing :numref:`sec_generalization_basics`, and state what a correct split would look like.
   *Provenance:* original (keep — this is the book's own Exercise 1, restated verbatim in spirit).
2. [conceptual] **Why Gradient-Based HPO Is Hard.** For a two-layer MLP on FashionMNIST (:numref:`sec_mlp-implementation`, batch size 256), tuning the SGD learning rate against a one-epoch validation metric: (a) name a validation metric usable in this graph and explain why validation *error* (a 0/1-style count) is unusable; (b) sketch the computational graph treating initial weights and the learning rate as input nodes (:numref:`sec_backprop`); (c) give an order-of-magnitude estimate of the number of activation values that must be stored across the forward pass on 60,000 training cases; (d) name one additional obstacle beyond compute/storage (:numref:`sec_numerical_stability`); (e) `:citet:maclaurin-icml15` proposes "hypergradients" to get around exactly this — state in your own words, in 2–3 sentences, which single step of your (b)–(d) answer their method sidesteps, and which one it does not.
   *Provenance:* adapted from original Exercise 2 (overlap high — same sub-items a–d unchanged; sub-item (e) replaces the bare "read this paper" pointer with a task that still requires engaging with `maclaurin-icml15`, addressing the style-review's clarity flag).
3. [conceptual] **Grid Search's Curse of Dimensionality.** Explain, referencing :citet:`bergstra-jmlr12a`, why random search can be far more sample-efficient than an equispaced grid once the number of tunable hyperparameters grows, even though both cover the same nominal range per hyperparameter.
   *Provenance:* original (keep — the book's own Exercise 3).
4. [short-code] **Uniform vs. Log-Uniform Priors, Empirically.** Replace `config_space["learning_rate"]` with `stats.uniform(1e-4, 1)` (linear-scale) and re-run the section's `num_iterations=5` random-search loop several times against the original `loguniform(1e-4, 1)` version; plot the sampled learning-rate values (histogram) and the resulting `errors` for both priors over, say, 20 repetitions each. State which prior more often lands a learning rate within one order of magnitude of the eventual incumbent.
   *Provenance:* inspired by Bergstra & Bengio (2012)'s point that hyperparameters spanning orders of magnitude need a log-scale search density (overlap low — the paper doesn't run this exact comparison).
5. [short-code] **Fixing the Validation Split.** Following your answer to Problem 1, carve a genuine validation split out of the FashionMNIST *training* set (e.g. a CS231n-style 90/10 split) instead of reusing the test set, adapt `HPOTrainer.validation_error` and `hpo_objective_softmax_classification` accordingly, and re-run the section's 5-trial random search. Report, in a small table, the chosen learning rate and the *true* held-out test accuracy under the original (leaky) split vs. your corrected split.
   *Provenance:* adapted from CS231n's train/val/test discipline (overlap med, cite on adoption) and the book's own Exercise 1 hint (overlap high, original).
6. [extended] **Grid Search Baseline.** Implement an equispaced grid-search baseline (per Problem 3's setup) over a 2-D space (`learning_rate` and one more hyperparameter of your choice, e.g. batch size, for the same softmax model), budget-match it against the section's random search (same number of total evaluations), plot both incumbent trajectories on one axis, and write a half-page verdict on whether Bergstra & Bengio's grid-vs-random claim replicates at this tiny scale.
   *Provenance:* adapted from Bergstra & Bengio (2012) (overlap med — replicates their central empirical claim at small scale; cite on adoption).

---

## chapter_hyperparameter-optimization/hyperopt-api.md — Hyperparameter Optimization API

**Topic:** Factoring HPO into `HPOSearcher` / `HPOScheduler` / `HPOTuner`
abstractions, implementing random search against this API, and comparing
algorithms via any-time (incumbent-vs-runtime) performance plots.
**Current exercises:** 2; disposition: keep 2, rewrite 0, drop 0 —
confirmed against the prior style review as defect-free: both exercises name
concrete code identifiers, exact hyperparameter ranges, and explicit hints;
this is the strongest exercise set in the chapter and is kept unchanged.

**External sources found:**
- Ray / Anyscale, official tutorial repository, `tune_exercises/exercise_2_optimize.ipynb` ("Search Algorithms and Trial Schedulers") — a real, hands-on notebook (fetched and read directly): define a grid-search space over `lr`/`momentum`, wrap it in an `ASHAScheduler` (`metric`, `mode`, `grace_period`), run it, then swap in a `HyperOptSearch` (Bayesian) search algorithm with `max_concurrent=1` and compare trial curves via TensorBoard. This is the closest thing to a graded "HPO homework" we found anywhere. — https://github.com/ray-project/tutorial/blob/master/tune_exercises/exercise_2_optimize.ipynb
- Benjamins, C. & Tornede, A., "Practical Hyperparameter Optimization with SMAC3," AutoML Fall School 2023 hands-on session — session abstract (verified) covers (1) SMAC's searcher/intensification principles and (2) a practical part: "setting up the configuration space, defining the range and types of hyperparameters," and "visualizing the optimization process" — the same three-part shape (define space → run optimizer → visualize) as this section, though we could not retrieve exact exercise wording beyond the abstract/slides. — https://sites.google.com/view/automl-fall-school-2023/schedule/hands-on-bayesian-optimization-with-smac
- Optuna documentation, "Efficient Optimization Algorithms" tutorial — documents `SuccessiveHalvingPruner`/`HyperbandPruner` and sampler choice as API reference, not as a posed task; included as a negative data point (frameworks ship docs, not homework). — https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html

**Proposed problem set** (our reference format):
1. [short-code] **A Harder Objective: DropoutMLP.** Modify `hpo_objective_lenet` into `hpo_objective_dropoutmlp` for the two-hidden-layer `DropoutMLP` (:numref:`sec_dropout`) with `max_epochs=50`, `num_gpus=0`; define `config_space` with `num_hiddens_1`, `num_hiddens_2` integers in $[8,1024]$, dropout values in $[0, 0.95]$, `batch_size` in $[16, 384]$; run `RandomSearcher`/`BasicScheduler`/`HPOTuner` with `number_of_trials=20`, first evaluating `initial_config = {'num_hiddens_1': 256, 'num_hiddens_2': 256, 'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1, 'batch_size': 256}`, and plot the incumbent trajectory.
   *Provenance:* original (keep — book's own Exercise 1).
2. [short-code] **A Local-Search Searcher.** Implement `LocalSearcher(HPOSearcher)` with parameters `probab_local`, `num_init_random`: for the first `num_init_random` calls sample uniformly at random; thereafter, with probability `1 - probab_local` sample uniformly at random, else perturb one randomly-chosen hyperparameter of the best-so-far configuration while holding the rest fixed. Re-run Problem 1's experiment with `LocalSearcher` in place of `RandomSearcher` for a couple of `(probab_local, num_init_random)` settings.
   *Provenance:* original (keep — book's own Exercise 2).
3. [conceptual] **Why Two Classes, Not One.** In 4–6 sentences, explain what capability the Searcher/Scheduler split buys over a single monolithic `HPOAlgorithm` class: name one new HPO method that only needs a new `HPOScheduler` (e.g., one that shortens `max_epochs` for configs performing poorly early) and requires no change to any `HPOSearcher`, and one that only needs a new `HPOSearcher` (e.g., Bayesian optimization) and requires no change to `BasicScheduler`.
   *Provenance:* inspired by the searcher/scheduler/tuner split documented (independently) by Ray Tune, Optuna, and Syne Tune, and by the AutoML Fall School SMAC session's "configure vs. optimize vs. visualize" framing (overlap low).
4. [conceptual] **Reading an Any-Time Plot by Hand.** Given a small hand-supplied table of `(cumulative_runtime, incumbent_error)` pairs for two algorithms A and B (construct your own 6–8-row example, modeled on :numref:`example_anytime_performance`), state at what wall-clock time B first becomes preferable to A, using only the fields `HPOTuner` already bookkeeps — no plotting required.
   *Provenance:* original — drills the exact fields defined in `bookkeeping()` in this section.
5. [short-code] **Does LocalSearcher's Edge Survive Noise?** Following the section's own warning that HPO comparisons need repetition across seeds, run both `RandomSearcher` and your Problem 2 `LocalSearcher`, each for ~10 repetitions of `number_of_trials=15` on a cheap stand-in objective (or `hpo_objective_lenet` with `max_epochs` lowered), and plot mean ± one standard deviation of `incumbent_trajectory` vs. `cumulative_runtime` for both. State whether any apparent advantage of `LocalSearcher` exceeds one standard deviation.
   *Provenance:* adapted from Ray Tune's `tune_exercises/exercise_2_optimize.ipynb`, which runs a grid/ASHA baseline against `HyperOptSearch` and compares curves (overlap med, cite on adoption); also grounded in this section's own "Comparing HPO Algorithms" discussion (overlap high, original).
6. [extended] **A Third Searcher: Grid Search Against This API.** Implement a deterministic `GridSearcher(HPOSearcher)` (Cartesian product over an equispaced grid per hyperparameter, per `hyperopt-intro.md`'s grid-search exercise) that plugs into the existing `BasicScheduler`/`HPOTuner` unmodified; run it on the LeNet example from this section and compare its any-time performance against `RandomSearcher`.
   *Provenance:* adapted from Bergstra & Bengio (2012)'s grid-vs-random comparison (overlap med, cite on adoption) and `hyperopt-intro.md`'s own grid-search exercise (overlap high, original), now implemented for the first time against this section's concrete Searcher/Scheduler/Tuner API.

---

## chapter_hyperparameter-optimization/rs-async.md — Asynchronous Random Search

**Topic:** Distributing random search across parallel workers using Syne
Tune, contrasting synchronous (straggler-blocked) vs. asynchronous
(always-busy) scheduling, and visualizing per-trial learning curves.
**Current exercises:** 2; disposition: keep 1, rewrite 1 — Exercise 1
(Syne Tune `DropoutMLP` objective, BO comparison, worker-count scaling
study) is solid and concrete throughout, and is kept. Exercise 2's final
sub-item ("Compare your new `LocalSearcher` with `RandomSearch` on the
`DropoutMLP` benchmark") names no metric or plot, unlike its sibling
sub-items and unlike Exercise 1(c) in the same section (flagged in the
prior style review); we rewrite it to add an explicit success criterion
while keeping the rest of the (Advanced-tagged) exercise as-is.

**External sources found:**
- MIT 6.5840 ("Distributed Systems," formerly 6.824), Lab 1: "MapReduce" — spec (fetched directly) requires the coordinator to detect a straggler worker via a fixed timeout ("notice if a worker hasn't completed its task in a reasonable amount of time (for this lab, use ten seconds), and give the same task to a different worker") since a busy coordinator "can't reliably distinguish between crashed... stalled... or too-slow" workers. Different domain (general distributed systems, not HPO), but it is the closest verifiable, freely-available assignment tackling exactly the sync/async + straggler problem this section motivates. — https://pdos.csail.mit.edu/6.824/labs/lab-mr.html
- Ray/Anyscale `tune_exercises/exercise_2_optimize.ipynb` (see `hyperopt-api.md` entry above) — reused here for its ASHA-scheduler half, which is about early-stopping trials under asynchronous scheduling rather than sequential comparison. — https://github.com/ray-project/tutorial/blob/master/tune_exercises/exercise_2_optimize.ipynb
- Lattimore, T. & Szepesvári, C., *Bandit Algorithms* (Cambridge University Press; free pre-publication PDF, verified by direct download + text extraction), Ch. 33 "Pure Exploration," §33.3 "Best arm identification with a budget," Exercise 33.7 — a rigorous 6-part guided proof of the Sequential Halving regret bound (algorithm originally due to Karnin, Koren & Somekh 2013). This is topically *successive halving* — i.e., `sh-intro.md`/`sh-async.md` territory in this same chapter, neither of which carries a `## Exercises` heading and so is out of scope for this catalog — but it is the single deepest, most rigorous exercise tradition we found anywhere adjacent to this chapter's algorithms, and is worth flagging for whoever eventually adds exercises to `sh-async.md`. — https://tor-lattimore.com/downloads/book/book.pdf

**Proposed problem set** (our reference format):
1. [short-code] **DropoutMLP on Syne Tune, at Scale.** Implement `hpo_objective_dropoutmlp_synetune` (reporting validation error after every epoch) for the `DropoutMLP` setup from `hyperopt-api.md` Exercise 1; compare `RandomSearch` against `syne_tune.optimizer.baselines.BayesianOptimization`; then, on a $\geq$4-core instance, run one of the two methods with `n_workers=1`, `2`, `4` and compare incumbent trajectories, expecting close-to-linear scaling for random search (average over repetitions for robustness).
   *Provenance:* original (keep — book's own Exercise 1).
2. [extended] **A Home-Grown Searcher, Inside Syne Tune.** Port the `LocalSearcher` from `hyperopt-api.md` Exercise 2 into Syne Tune as a custom searcher (build a dev environment with both `d2lbook` and `syne-tune` sources; follow Syne Tune's developer tutorial or its home-made-scheduler example). Compare it against `RandomSearch` on the `DropoutMLP` benchmark using an explicit criterion: the mean incumbent validation error after a fixed 15-minute wall-clock budget, averaged over at least 3 random seeds each, reported as a two-row table plus one sentence on which one wins.
   *Provenance:* adapted from original Exercise 2 (overlap high — same setup; the final comparison sub-item is rewritten to add the missing metric/success criterion flagged in the style review).
3. [conceptual] **Why Random Search Parallelizes for Free.** In 4–6 sentences, explain why `HPOSearcher.sample_configuration` (from `hyperopt-api.md`) can be called independently for every new trial under `RandomSearcher`, and identify which specific piece of state a Bayesian-optimization-style searcher would need to read before proposing its next configuration — the piece of state that forces some coordination and rules out trivially asynchronous parallelism for that method.
   *Provenance:* adapted from this section's own text ("random search... without exploiting observations... not straightforward with more sophisticated methods," overlap high, original), sharpened with the general "embarrassingly parallel" framing from parallel/distributed-systems teaching (overlap low, inspired by).
4. [short-code] **Injecting a Straggler.** Modify `hpo_objective_lenet_synetune` so that, with probability 0.1, it sleeps for a fixed extra delay (e.g. 3× a normal epoch's time) before returning — a simulated straggler trial. Re-run the asynchronous tuner with `n_workers` unchanged, and compare the resulting incumbent-trajectory-vs-wall-clock-time plot against a straggler-free run. Report, in seconds, how much wall-clock time the straggler adds before the incumbent trajectory catches back up.
   *Provenance:* inspired by MIT 6.5840 Lab 1 ("MapReduce"), which requires exactly this kind of timeout-based straggler detection and task reassignment (overlap low — different domain, same phenomenon; cite the lab if the timeout-based detection mechanic is adapted directly).
5. [short-code] **Does the Speed-Up Claim Hold?** The section claims close-to-linear speed-up with more workers. Using the section's own GPU-memory-based `n_workers` auto-detection cell, force `n_workers` to 1, 2, and the auto-detected value in three shortened (e.g. 3-minute) runs; record the wall-clock time at which each run's incumbent trajectory first drops below a fixed validation-error threshold (choose one from a scouting run), and report whether crossing times scale roughly as $1/n\_workers$.
   *Provenance:* adapted from this section's own Exercise 1(c), tightened with an explicit threshold-crossing success criterion (overlap high, original) — directly addresses the style review's note that this sub-item, unlike its sibling, previously named no metric.

---

### Summary of totals

| File | Existing exercises | keep | rewrite | drop | Proposed problems |
|---|---|---|---|---|---|
| hyperopt-intro.md | 3 | 2 | 1 | 0 | 6 |
| hyperopt-api.md | 2 | 2 | 0 | 0 | 6 |
| rs-async.md | 2 | 1 | 1 | 0 | 5 |
| **Total** | **7** | **5** | **2** | **0** | **17** |
