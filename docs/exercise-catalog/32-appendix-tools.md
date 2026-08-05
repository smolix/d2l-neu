# Chapter Overview — chapter_appendix-tools-for-deep-learning

Eight sections carry `## Exercises` (in book order: interactive-development,
hosted-notebooks, cloud-instances, hardware, software-ecosystem,
training-systems, model-serving, developers-guide); `utils.md`, `d2l.md`,
`index.md` have none. The prior style review already found all 32 existing
exercises uniformly strong — concrete deliverables, named success criteria,
zero defects — so this pass is additive, not corrective: every section keeps
its full existing set. Best external matches, by strength: **DTU MLOps**
(`skaftenicki.github.io/dtu_mlops`) is the standout source — its "Using the
Cloud" (M21) and "Distributed Training" (M20/M30) modules pose almost the
same provision/verify/benchmark tasks this book already does, so its value is
confirmatory plus one or two genuinely new angles (DataParallel-vs-DDP,
marketplace-not-hyperscaler provisioning). **Stanford CS336** assignment2-systems
(build DDP + optimizer-state sharding from scratch) is the strongest single
match for training-systems.md. **vLLM's own benchmark CLI** and **MLPerf
Inference** are the natural additions for model-serving.md — the section
already teaches the concepts vLLM's tooling exercises, so running the
project's own benchmark harness is a low-friction, high-value addition. The
weakest external match is **hosted-notebooks.md**: no course treats hosted
notebook session-lifecycle mechanics (Colab-pointer-vs-Kaggle-copy, quota
mechanics) as a pedagogical object — that's this book's own contribution, and
I say so rather than force-fitting a source. Chip Huyen's *ML Systems Design*
booklet (not the DMLS book itself — different, earlier work, same author)
supplies a clean baseline-before-leaderboard idea for software-ecosystem.md.
MIT Missing Semester supplies verified, close-fitting git-archaeology material
for developers-guide.md. Totals below.

---

## chapter_appendix-tools-for-deep-learning/interactive-development.md — Notebooks

**Topic:** Notebook document-vs-kernel state, restart-and-run-all, local/editor/remote workflows.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — all four
name a concrete comparison or measurement with an explicit "detect/explain"
closing; nothing to fix.

**External sources found:**
- Made With ML (Goku Mohandas), MLOps course, "Moving from Notebooks to
  Scripts" lesson, 2024 — argues notebooks should graduate to scripts on
  three grounds: statelessness (explicit vs. global state), linear execution
  (vs. out-of-order cells), and testability — the same failure mode this
  section's :numref:`fig_tools_kernel_state` illustrates, argued from the
  opposite direction (when to leave the notebook) — https://madewithml.com/courses/mlops/scripting/
- MIT, *Missing Semester of Your CS Education*, "Debugging and Profiling"
  lecture exercises, 2020/2026 — has students use `cProfile`/`line_profiler`
  to compare two algorithms' runtime and `memory_profiler` for memory, then
  cross-check with `perf` counters — https://missing.csail.mit.edu/2020/debugging-profiling/
- Full Stack Deep Learning, 2022 cohort, Lab 5 "Troubleshooting & Testing" —
  has students dissect a captured PyTorch training-step trace to locate a
  performance problem in GPU-accelerated code — https://fullstackdeeplearning.com/course/2022/lab-5-troubleshooting-and-testing/
- DTU MLOps course (Nicki Skafte Detlefsen) — surveyed for a notebook/editor
  module; the course's environment-and-reproducibility modules (Docker, M9)
  are close in spirit but operate at the container level, not the
  kernel/editor level this section teaches — noted as a near-miss, not
  adopted directly — https://skaftenicki.github.io/dtu_mlops/

**Proposed problem set** (6 problems):
1. [short-code] **Out-of-Order Dependency.** Create an intentional
   out-of-order dependency like the one in
   :numref:`fig_tools_kernel_state`, confirm that it works interactively,
   and then detect it with restart and run all.
   *Provenance:* original (kept from current set).
1. [short-code] **Executable Identity Check.** Compare `sys.executable` in a
   terminal, a JupyterLab kernel, and a VS Code kernel on your machine.
   Explain any difference.
   *Provenance:* original (kept from current set).
1. [short-code] **Sync-Aware Timing.** Time a matrix product with `%timeit`
   on CPU and, if available, on a GPU with and without synchronization.
   Explain the discrepancy.
   *Provenance:* original (kept from current set).
1. [conceptual] **Remote Layer Map.** Connect to a remote machine through an
   SSH tunnel and identify where the editor, server, kernel, and file system
   each run.
   *Provenance:* original (kept from current set).
1. [conceptual] **Notebook-to-Module Boundary.** Take one training notebook
   you have run in this book and sort its cells into two groups: "stays a
   notebook cell" and "belongs in a `.py` module," using Made With ML's three
   criteria (state, order, testability) as your rubric. Produce a one-page
   table naming each cell and its verdict, then check it against this
   section's own advice to "keep testable logic in `.py` modules."
   *Provenance:* adapted from Made With ML MLOps course, "Moving from
   Notebooks to Scripts" (overlap med; cite on adoption).
1. [short-code] **Profile a Slow Cell.** Pick a training or preprocessing
   cell from this book that takes more than a few seconds, profile it with
   `cProfile` (or `%prun`), and identify the single line responsible for the
   largest share of wall time. Apply one fix and confirm the profiled time
   drops by at least 30%.
   *Provenance:* inspired by MIT Missing Semester, "Debugging and Profiling"
   lecture exercises (overlap low), and Full Stack Deep Learning 2022 Lab 5
   (overlap low).

---

## chapter_appendix-tools-for-deep-learning/hosted-notebooks.md — Colab and Kaggle

**Topic:** Hosted-runtime lifecycle, Colab-vs-Kaggle persistence models, portable setup cells.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — all four
already ask for a specific platform action and observation; strong as is.

**External sources found:**
- No course or textbook surveyed treats *hosted-notebook session lifecycle*
  (Colab's live-GitHub-pointer vs. Kaggle's account-copy, weekly-quota
  mechanics) as an exercise topic in its own right — these platforms are
  used as free infrastructure inside other courses, never as the object of
  study. This is a genuine coverage gap in the external tradition, not a
  failure to search: Colab's own FAQ and Kaggle's docs are reference
  material, not pedagogy — https://research.google.com/colaboratory/faq.html
- DTU MLOps course — surveyed for a hosted-notebook-equivalent module; its
  closest analog is "Using the Cloud" (M21), which teaches the same
  provision/verify discipline but for self-managed cloud VMs, not
  quota-limited hosted notebooks — see the entry under cloud-instances.md
  below, where the overlap is real — https://skaftenicki.github.io/dtu_mlops/latest/s6_the_cloud/using_the_cloud/
- Made With ML — surveyed; the course assumes a local or Anyscale
  workspace setup throughout and never engages Colab/Kaggle's
  provider-controlled runtime model — https://madewithml.com/courses/mlops/setup/

**Proposed problem set** (5 problems):
1. [short-code] **Two Platforms, Two Lifetimes.** Open this section on
   both Colab and Kaggle via the **Run notebook** control. Where does your
   edited copy live in each case, and what happens to it when the session
   ends?
   *Provenance:* original (kept from current set).
1. [short-code] **Fingerprint Extension.** Extend the environment
   fingerprint with the framework version and accelerator name, without
   failing on a CPU-only runtime.
   *Provenance:* original (kept from current set).
1. [short-code] **Version Diff on Kaggle.** On Kaggle, produce two versions
   of a notebook with **Save & Run All** and compare them. What exactly does
   Kaggle store per version?
   *Provenance:* original (kept from current set).
1. [conceptual] **Quota-vs-Job Budget.** Estimate how long a free weekly
   Kaggle GPU quota would take to fine-tune the BERT model of
   :numref:`sec_bert-pretraining`, using the timings reported there.
   *Provenance:* original (kept from current set).
1. [short-code] **Cross-Provider Fingerprint.** Run this section's
   environment-fingerprint cell on Colab, on Kaggle, and on a local Jupyter
   kernel, and assemble the three results into one table. Flag any field
   (Python, NumPy, or machine architecture) that differs enough between
   providers to explain a "worked there, not here" bug report.
   *Provenance:* original — no external source treats this comparison as an
   exercise (see finding above); designed to fill that gap using the
   section's own fingerprint cell across three environments instead of one.

---

## chapter_appendix-tools-for-deep-learning/cloud-instances.md — Cloud Computing

**Topic:** Renting GPUs (three provider tiers), cost-per-result, disposable-instance discipline.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — all four
name a concrete, checkable artifact (priced comparison, measured memory
high-water mark, timed recovery drill, trust-boundary diagram).

**External sources found:**
- DTU MLOps course (Nicki Skafte Detlefsen), "Using the Cloud" (M21) — has
  students create a VM, `gcloud compute instances list`/SSH into it, confirm
  Python/PyTorch are *not* preinstalled on a bare image, then redeploy from a
  vendor deep-learning image and re-verify the stack, finishing with an
  explicit reminder to stop billing — the same boot/verify/teardown
  discipline this section teaches, on a hyperscaler specifically —
  https://skaftenicki.github.io/dtu_mlops/latest/s6_the_cloud/using_the_cloud/
- Chip Huyen, *Machine Learning Systems Design* (booklet; not the DMLS
  book) — its "Scaling" section poses cost/throughput tradeoff questions
  (e.g., balancing workload across machines, sync vs. async training cost)
  adjacent to but not overlapping this section's per-hour/per-result
  framing — https://github.com/chiphuyen/machine-learning-systems-design
- Full Stack Deep Learning — surveyed; its labs run on a fixed, provided
  compute budget and never pose a "shop the market" exercise, so no direct
  match — https://fullstackdeeplearning.com/course/2022/

**Proposed problem set** (6 problems):
1. [short-code] **Three-Provider Cost Model.** Price a fine-tuning job you
   care about on three providers from :numref:`tab_cloud_prices` using cost
   per completed run. State the date and your speed assumptions, then check
   how prices have moved since this table was written.
   *Provenance:* original (kept from current set).
1. [short-code] **Memory High-Water Mark.** Take whichever training
   notebook of this book you ran most recently and measure its actual GPU
   memory high-water mark. Which entries of :numref:`tab_cloud_prices` could
   run it?
   *Provenance:* original (kept from current set).
1. [short-code] **Interruption Drill.** Simulate an interruption: start a
   checkpointed training run, kill the process mid-epoch, and resume on a
   fresh machine from object storage. Time the recovery and identify what
   you forgot to save.
   *Provenance:* original (kept from current set).
1. [conceptual] **Trust Boundary Diagram.** Draw the trust boundary for (a)
   a marketplace host processing a public dataset and (b) a hyperscaler
   processing medical records. What changes?
   *Provenance:* original (kept from current set).
1. [short-code] **Marketplace Boot-and-Verify.** Rent the cheapest GPU
   listing from a marketplace or specialist provider (not a hyperscaler),
   boot its deep-learning image, and run this section's `nvidia-smi` /
   `df -h` / `torch.cuda.get_device_name(0)` smoke test. Confirm the
   assigned GPU matches the listing, then tear the instance down and report
   the total bill for the session.
   *Provenance:* adapted from DTU MLOps course, "Using the Cloud" (M21)
   (overlap high; cite on adoption) — same provision/verify/teardown
   sequence, retargeted from a hyperscaler console to the marketplace tier
   this section treats as the price leader.
1. [extended] **Real Cost-per-Result Audit.** Rerun this section's
   cost-per-result model (`#cloud-instances-cost-model`) with real numbers:
   rent a GPU, run an actual job from this book end to end, and record
   actual wall-clock time, actual storage/egress charges, and your own
   time spent on setup and debugging. Compare the real total to the
   model's estimate and identify which input was most wrong.
   *Provenance:* original — extends the section's own cost model from
   assumed to measured inputs; inspired by the general
   "measure before you trust the model" discipline in Chip Huyen's ML
   Systems Design booklet (overlap low).

---

## chapter_appendix-tools-for-deep-learning/hardware.md — Hardware

**Topic:** Capacity-vs-bandwidth-vs-compute reasoning for buying training/inference hardware.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — all four
compute a specific number against a specific table; no defects.

**External sources found:**
- MIT 6.5940 / 6.S965, *TinyML and Efficient Deep Learning Computing* (Song
  Han lab, efficientml.ai) — five problem sets on quantization, pruning,
  neural architecture search, and LLM compression/deployment, each requiring
  students to estimate memory/latency savings from a bit-width or sparsity
  choice against real hardware specs — the closest external match to this
  section's decode-bound and training-memory formulas —
  https://hanlab.mit.edu/courses/2022-fall-6s965
- r/LocalLLaMA hardware megathreads and llm-tracker.info benchmark tables —
  already cited by name inside this section's own "Keeping Current"
  paragraph as sources of newer numbers, not previously turned into an
  exercise — https://www.reddit.com/r/LocalLLaMA/, https://llm-tracker.info/
- Chip Huyen, *Machine Learning Systems Design* booklet, "Scaling" section —
  poses a mixed-precision-training tradeoff question (16-bit vs. 32-bit,
  memory footprint vs. batch size) adjacent to this section's bandwidth
  reasoning, but framed for training rather than the buy-side decision this
  section teaches — https://github.com/chiphuyen/machine-learning-systems-design

**Proposed problem set** (6 problems):
1. [short-code] **MoE Decode Sweep.** Compute the decode bound for a
   30B-A3B mixture-of-experts model (3B active parameters, 4-bit) on every
   machine in :numref:`tab_unified_memory`. Which become interactive
   (>20 tok/s)?
   *Provenance:* original (kept from current set).
1. [short-code] **LoRA Memory Fit.** Estimate LoRA fine-tuning memory for
   an 8B model on a 16 GB card: 4-bit frozen base, BF16 adapters at 1% of
   parameters, Adam. Does it fit, and what dominates?
   *Provenance:* original (kept from current set).
1. [short-code] **Build and Break-Even.** Spec a complete RTX 5070 Ti
   build at current local prices, then compute its break-even in hours
   against a rented 4090 from :numref:`tab_cloud_prices`.
   *Provenance:* original (kept from current set).
1. [short-code] **Used-Market Recheck.** Find this month's used 3090 price
   and recompute its \$/GB against the current 5070 Ti. Has the 2026
   anomaly (used cards appreciating) persisted?
   *Provenance:* original (kept from current set).
1. [conceptual] **Compute-Bound or Bandwidth-Bound.** For three workloads —
   a LoRA fine-tune, dense-70B decode, and MoE decode — classify each as
   compute-bound or bandwidth-bound using the arithmetic-intensity reasoning
   behind this section's decode-bound formula, and justify each
   classification in one paragraph with no GPU required.
   *Provenance:* inspired by MIT 6.5940's quantization/compression problem
   sets, which require the same bound-identification step before estimating
   a savings number (overlap med).
1. [extended] **Reproduce a Benchmark Claim.** Pick one throughput number
   from this section's community-benchmark discussion (or a current one
   from llm-tracker.info or the llama.cpp discussions board), reproduce it
   on hardware you can access, and check whether the decode-bound formula
   predicts your measured tok/s within the factor of two this section
   claims.
   *Provenance:* inspired by the community benchmark culture this section
   itself points to (r/LocalLLaMA, llm-tracker.info, llama.cpp
   discussions) (overlap low) — turns a citation this section already makes
   into a verification exercise.

---

## chapter_appendix-tools-for-deep-learning/software-ecosystem.md — Ecosystem

**Topic:** Finding, evaluating, and safely pinning models/datasets/leaderboards.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — every
exercise already names a concrete artifact (shortlist, file inventory,
eval set, license determination).

**External sources found:**
- Chip Huyen, *Machine Learning Systems Design* booklet (not the DMLS
  book), "Model Selection" section — argues every comparison needs a
  random-guess baseline, a human baseline, and a simple-heuristic baseline
  before trusting a ranking, plus ablation studies to check which components
  earn their complexity — directly transferable to this section's
  leaderboard-shortlisting exercises — https://github.com/chiphuyen/machine-learning-systems-design
- Artificial Analysis and OpenRouter rankings — already named and used
  inside this section's own text as leaderboard sources; surveyed for
  exercise framing but not an independent external course/textbook, so not
  double-counted as a new source here — https://artificialanalysis.ai/
- Made With ML — surveyed; its "foundations" track builds models from
  scratch and never poses a "discover and evaluate a released model"
  exercise, so no match found here — https://madewithml.com/courses/foundations/

**Proposed problem set** (5 problems):
1. [conceptual] **Leaderboard Disagreement.** Pick a task you care about
   and shortlist three models using at least two leaderboards plus
   Artificial Analysis. Where do the rankings disagree, and why might that
   be?
   *Provenance:* original (kept from current set).
1. [conceptual] **Executable-Risk Inventory.** Inspect a model repository
   of your choice and list every file needed for offline inference. Which
   of them could execute code on your machine?
   *Provenance:* original (kept from current set).
1. [short-code] **Your Own Eval Set.** Build a 25-example evaluation set
   for a task you know well and run your shortlist from Exercise 1 on it.
   Does your ranking match the leaderboards'?
   *Provenance:* original (kept from current set).
1. [conceptual] **License Determination.** Find the license of a popular
   open-weight model and determine: may you deploy it commercially,
   fine-tune it, and redistribute the fine-tune?
   *Provenance:* original (kept from current set).
1. [short-code] **Baseline Before Leaderboard.** Before trusting the
   ranking your shortlist produced in Exercise 3, add a random-guess
   baseline and a simple heuristic baseline to the same 25-example eval
   set. Report what fraction of the shortlist's advantage over the
   heuristic baseline survives — and whether the leaderboard-implied
   ranking still holds once the baselines are on the same scorecard.
   *Provenance:* adapted from Chip Huyen's ML Systems Design booklet,
   "Model Selection" baseline discussion (overlap med; cite on adoption).

---

## chapter_appendix-tools-for-deep-learning/training-systems.md — Distributed Model Training

**Topic:** Scaling ladder (DDP → FSDP → tensor/pipeline/expert parallelism), memory accounting, recovery.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — each
names a specific quantity or artifact (a world-size threshold, three
batch/world-size/accumulation combinations, a measured speedup, a
checkpoint/restore design).

**External sources found:**
- Stanford CS336, *Language Modeling from Scratch*, Assignment 2 (Systems)
  — has students build a benchmarking/profiling harness, implement
  distributed data-parallel training and optimizer-state sharding from
  scratch, and compare against framework baselines — the closest external
  match found for this section's whole topic — https://github.com/stanford-cs336/assignment2-systems
- DTU MLOps course, "Distributed Training" module (M20/M30) — has students
  wrap a model in `nn.DataParallel`, benchmark it across batch sizes, then
  explain the purpose of each DDP component (wrapper, `DistributedSampler`,
  `dist.barrier()`, environment variables) in provided example code, and
  measure the actual 1-vs-2-GPU speedup — https://skaftenicki.github.io/dtu_mlops/s9_scalable_applications/distributed_training/
- CMU 10-414/714, *Deep Learning Systems* (dlsyscourse.org, Kolter & Chen)
  — builds an autodiff/NN library (Needle) from scratch across five
  assignments, including hardware acceleration; surveyed for a
  parallel-primitives assignment but its distributed-training coverage is
  thinner than CS336's — noted as a near-miss —
  https://dlsyscourse.org/assignments/

**Proposed problem set** (6 problems):
1. [short-code] **FSDP Break-Even World Size.** Extend the memory model
   with a communication-buffer term and a LoRA configuration (frozen base
   weights, small trainable adapter). At what world size does FSDP stop
   paying for a 7B model?
   *Provenance:* original (kept from current set).
1. [conceptual] **Fixed-Batch Combinatorics.** For a fixed global batch of
   4M tokens, enumerate three combinations of device batch, world size, and
   accumulation steps, and reason about their relative throughput and
   optimizer behavior.
   *Provenance:* original (kept from current set).
1. [short-code] **Measured vs. Ideal Speedup.** Take a training notebook of
   roughly the scale of :numref:`sec_bert-pretraining` and run it once with
   `torchrun --nproc-per-node=2`. Measure the actual speedup over one GPU
   and explain the gap from 2×.
   *Provenance:* original (kept from current set).
1. [conceptual] **Elastic Checkpoint Design.** Design (on paper) the
   checkpoint contents and restore protocol for a sharded training job that
   must resume with a *different* number of workers. Which parts of the
   state are per-rank, and which are global?
   *Provenance:* original (kept from current set).
1. [short-code] **DataParallel vs. DDP.** Wrap the same model first in
   `nn.DataParallel` and then in this section's `torchrun` + DDP pattern.
   Benchmark both across two or three batch sizes and identify the batch
   size where `DataParallel`'s single-process, multi-thread design starts
   losing to DDP's replicated-process design.
   *Provenance:* adapted from DTU MLOps course, "Distributed Training"
   module (M20/M30) (overlap high; cite on adoption).
1. [extended] **Toy All-Reduce.** Implement a naive ring or tree all-reduce
   over N simulated workers (Python `multiprocessing`, no GPU required)
   that averages a fixed gradient vector. Verify correctness against
   `numpy.mean` across ranks, then compare its wall time to
   `torch.distributed.all_reduce` on the same vector at the same world size
   and explain the gap.
   *Provenance:* inspired by Stanford CS336 Assignment 2 (Systems), which
   builds distributed data-parallel training and optimizer-state sharding
   from scratch rather than importing them (overlap med) — scaled down to a
   single primitive so it runs without a multi-GPU machine.

---

## chapter_appendix-tools-for-deep-learning/model-serving.md — Model Serving

**Topic:** Serving engines (Ollama/vLLM/SGLang/TensorRT-LLM), continuous batching, paged/prefix KV cache, quantization.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — all four
name a concrete metric or artifact (p95 TTFT comparison, TTFT/TPOT at three
concurrencies, a KV-cache byte budget, a cache-key policy).

**External sources found:**
- vLLM project, official benchmark suite (`vllm bench serve` /
  `benchmarks/benchmark_serving.py`) — a maintained, primary-source
  benchmarking harness that measures exactly the TTFT/TPOT/goodput
  quantities this section defines, across configurable request rates and
  concurrency — https://docs.vllm.ai/en/stable/cli/bench/throughput/,
  https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md
- MLCommons, *MLPerf Inference* benchmark suite and Collective Mind
  tutorial — a fixed-methodology benchmark (fixed dataset, accuracy gate,
  percentile-latency reporting, offline/server scenarios) that formalizes
  exactly the ad hoc measurement this section's exercises ask students to
  do informally — https://github.com/mlcommons/inference,
  https://github.com/mlcommons/ck/blob/master/docs/tutorials/scc23-mlperf-inference-bert.md
- PyTorch/Serve (TorchServe) benchmark suite and handler tutorials —
  surveyed; TorchServe is now in limited maintenance and this section
  already treats vLLM/SGLang/TensorRT-LLM as the current-generation server
  engines, so TorchServe's benchmarking approach is noted but not adopted
  as an exercise source — https://github.com/pytorch/serve

**Proposed problem set** (6 problems):
1. [short-code] **Scheduler with Rejection.** Extend the toy scheduler with
   Poisson arrivals, a KV-memory budget, and rejection. Compare
   first-come-first-served against shortest-remaining-work on p95 TTFT.
   *Provenance:* original (kept from current set).
1. [short-code] **Ollama vs. vLLM.** Serve the same 8B model through
   Ollama (Q4_K_M) and vLLM (AWQ) on whatever hardware you have, and
   measure TTFT and TPOT at concurrency 1, 4, and 16. Which engine performs
   better for each workload, and why?
   *Provenance:* original (kept from current set).
1. [conceptual] **KV-Cache Budget.** Estimate KV-cache bytes per token for
   a model whose config you know (layers × 2 × kv-heads × head-dim ×
   bytes), then compute how many 8K-context users fit beside the weights on
   a 24 GB card.
   *Provenance:* original (kept from current set).
1. [conceptual] **Shared-Prompt Cache-Key Policy.** Design the cache-key
   policy for a service that reuses a long system prompt across users. What
   must be in the key, and what is the privacy obligation of caching at
   all?
   *Provenance:* original (kept from current set).
1. [short-code] **Reproduce a Goodput Curve.** Run vLLM's own benchmark
   tool (`vllm bench serve` or `benchmarks/benchmark_serving.py`) against a
   small model at three request rates. Plot raw throughput and goodput
   (requests meeting a stated TTFT SLO) against request rate, and identify
   the rate at which the two curves diverge.
   *Provenance:* adapted from vLLM's official benchmark CLI and
   `benchmarks/README.md` (overlap med; cite on adoption).
1. [extended] **MLPerf-Style Mini-Submission.** Pick one MLPerf Inference
   scenario (e.g., offline throughput) for a small open model on hardware
   you can access. Follow MLPerf's methodology — fixed dataset, an
   accuracy gate the run must clear, percentile-latency reporting — to
   produce a mini submission table, then compare its rigor (warm-up,
   accuracy gate, percentiles) to the ad hoc benchmark you ran in
   Exercise 2.
   *Provenance:* inspired by MLCommons' MLPerf Inference benchmark suite
   and its Collective Mind tutorial (overlap low).

---

## chapter_appendix-tools-for-deep-learning/developers-guide.md — Contributor Guide

**Topic:** Single-source build pipeline, Git contribution workflow, agent-assisted contributions.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — all four
are concrete build/contribution tasks with a measurable or verifiable
outcome; the meta exercise (using a coding agent) is appropriate rather than
a tone problem.

**External sources found:**
- MIT, *Missing Semester of Your CS Education*, "Version Control (Git)"
  lecture exercises, 2020/2026 — has students explore a real repository's
  version history as a graph, use `git log`/`git blame`/`git show` to find
  who last touched a specific line and why, and practice removing an
  accidentally committed secret or large file from history — the closest
  external match for git-archaeology-style exercises —
  https://missing.csail.mit.edu/2020/version-control/
- No course or textbook surveyed treats *this book's specific build model*
  — one Markdown source generating four notebook sets, slides, a library,
  and a website, with framework tabs and stable cell IDs — as an exercise
  topic; that pipeline is idiosyncratic to d2l-neu, so the external
  tradition here is necessarily thin and only the generic
  git-workflow layer transfers — https://github.com/smolix/d2l-neu
- Full Stack Deep Learning and Made With ML — surveyed; neither course
  is itself an open-source book with a generated-artifact pipeline, so
  neither poses a "contribute a fix and verify the rebuild" exercise —
  https://fullstackdeeplearning.com/, https://madewithml.com/

**Proposed problem set** (5 problems):
1. [short-code] **No-Op Rebuild Timing.** Fork the repository, run
   `make html`, and render the book locally. How long does a no-op rebuild
   of one page take?
   *Provenance:* original (kept from current set).
1. [short-code] **Docstring Round-Trip.** Find the `#@save` block that
   defines a `d2l` function you have used, change its docstring, rebuild
   the library, and verify the change is visible from a notebook import.
   *Provenance:* original (kept from current set).
1. [short-code] **Agent-Reviewed Micro-Fix.** Use a coding agent for a real
   micro-contribution: have it find a typo or broken link in a chapter of
   your choice, fix the source, and verify the render — then review its
   diff as if you were the maintainer.
   *Provenance:* original (kept from current set).
1. [conceptual] **Framework-View Diff.** Open the generated PyTorch and JAX
   notebooks of one section side by side (the VS Code extension makes this
   two keystrokes). What exactly differs, and where does that difference
   live in the source file?
   *Provenance:* original (kept from current set).
1. [short-code] **Git Archaeology on This Repo.** Using only `git log`,
   `git blame`, and `git show` (not `git status`/`git diff`), find who last
   modified a `#@save` function you use often and read the commit message
   that introduced it. Then find one commit in this repository's history
   that touched more than five files, and explain — from the message and
   diff alone — what property it was preserving (framework parity,
   `:numref:` resolution, or slide sync).
   *Provenance:* adapted from MIT Missing Semester, "Version Control (Git)"
   lecture exercises (overlap med; cite on adoption).

---

**Totals:** 8 sections, 32 current exercises (all 32 kept, 0 rewritten, 0
dropped), 13 new problems proposed, 45 problems total. Tag mix across the 13
new problems: 7 short-code, 2 conceptual, 4 extended (one each in
cloud-instances, hardware, training-systems, and model-serving — the four
sections judged to have enough external tooling/methodology depth to support
a project-scale addition). Every section retains at least one conceptual and
one short-code problem as required.
