# Ch. 13 review findings and fix spec (2026-07-20)

*Fable review of the ch. 13 rebuild (commit 9b5d3d4d), merged with the
validated findings of `performance_feedback.md` (an external review whose
factual claims were checked against the sources, the committed outputs
store, the pinned APIs, and vendor specs — its validity verdicts are in §5).
This file is the working spec for the fix pass. Fix agents: read §0 and §1
first, then your assigned per-file section in §2, then §3 (cross-cutting)
and §6 (do-not-touch). The binding build docs are `CLAUDE.md`,
`docs/build-system.md`, and `reviews/comp-perf-implementation-brief.md`.*

---

## 0. Operating rules for fix agents

1. **Source `.md` is truth; NEVER edit `.qmd`** (generated). Never edit files
   outside your assigned scope.
2. **Do not touch `#@save` bodies** — `Benchmark` (13.1), `split_batch`
   (13.5, both tabs), `resnet18`/`ResNet18` (13.6) — byte-identical bodies
   keep the d2l lib fingerprint fixed and the cross-chapter blast radius
   zero. If a fix seems to require changing one, STOP and report instead.
3. **New code cells**: write them without a `#id`, then run
   `python3 tools/add_cell_ids.py chapter_computational-performance/<file>.md`
   (idempotent; never edit existing IDs). Keep one imports cell per
   framework; new imports go there, not mid-file.
4. **Verify every code fix by running it** in the right venv
   (`.venv-pytorch/bin/python`, `.venv-jax/bin/python`) as a scratch script
   under your scratchpad — NOT via `jupyter nbconvert` on the real notebook,
   and NOT via the scheduler. Final execution/recapture happens centrally
   after all content lands (§7). Respect your assigned
   `CUDA_VISIBLE_DEVICES`.
5. **Prose numbers**: quote only re-execution-stable magnitudes (proposal
   §6.6). Where your fix changes what a cell will print, write the prose
   hedged to what your scratch run measured, and add the cell id to your
   status file under "RECONCILE AFTER CAPTURE" so the final pass re-checks
   prose against the blessed outputs.
6. **Tab-conditional prose is available**: `:begin_tab:`pytorch``…
   `:end_tab:` (see `chapter_builders-guide/saving-loading.md` for live
   examples). Use it where the two frameworks' measured stories genuinely
   diverge; prefer framework-neutral phrasing when it reads naturally.
7. **Figures**: edit only `tools/gen_mdl_perf_figures.py`, regenerate with
   `make figures`, render (`rsvg-convert -z 2 img/<f>.svg -o /tmp/f.png`)
   and LOOK at the PNG; re-run `make figures` and confirm `git diff img/`
   is byte-empty on the second run (idempotence).
8. **Slides**: `<!-- slides -->` sections use `@<id>` (or `@<id>@<fw>` to
   pin a framework variant). A slide referencing a cell that exists in only
   one framework MUST scope it, or the other deck silently drops it.
9. **Do not commit, do not push, do not run `make html`/`make lib`/the
   scheduler.** Maintain a status file
   `reviews/fixlog-<scope>.md` (create it; append as you go: done items,
   open items, RECONCILE list, anything you deferred) so work survives a
   pause/resume.
10. Exercise numbering stays `1.`-style; keep the book voice — fixes here
    replace absolutes with their conditions, they do not flatten the prose.

## 1. Ground truth — what the committed store actually printed

From `outputs/{pytorch,jax}/chapter_computational-performance/*.json`
(capture of 2026-07-20) and `reviews/comp-perf-pilot-notes.md` §MEASURED:

| Cell | pytorch | jax |
|---|---|---|
| 13.1 naive vs synced timer (10 mm) | 0.87 / 9.12 ms | 1.48 / **15.83 ms (bug-inflated; true ≈8.5)** |
| 13.1 Benchmark 4096-matmul | 0.91 ms/call | 0.85 ms/call |
| 13.1 read-every / read-once | 0.061 / 0.029 s | 0.842 / **2.217 s (inverted!)** |
| 13.1 sweep (256→8192, TF) | 2.2, 16.2, 116.7, 157.0, 150.7, 165.1 | 0.4, 2.8, 29.2, 136.9, 159.8, 161.7 |
| 13.1 add/mul/sin/sigmoid | all ≈0.14 ms | 0.21/0.21/0.16/0.16 ms |
| 13.1 1-op / chain | 0.15 / 1.28 ms | 0.23 / 1.54 ms |
| 13.2 HBM bandwidth | 0.80 TB/s | 0.91 TB/s |
| 13.2 H2D | pageable 13.4, pinned 24.3 GB/s | device_put **0.9 GB/s** |
| 13.3 fusion eager→compiled | 1.25→0.15 ms (8.3×) | 1.29→0.15 ms (8.6×) |
| 13.3 whole-step | first 0.4 s; 3.63→2.17 ms (1.67×) | AOT 1.8 s; 17.40→0.17 ms |
| 13.3 overhead demo | 1.78→0.43 ms | 7.71→0.27 ms |
| 13.4 predicted/measured MB | 403 / 524 | temp 15, arg+out 203 |
| 13.4 fp32 / tf32 / bf16 | 22.45 / 15.43 / 9.78 ms (1.58×) | 13.12 / — / 6.73 ms (1.95×) |
| 13.4 checkpointing peak | 4318→2507 MB | temp 3230→1216 MB |
| 13.5 LeNet sec/epoch (1→2 GPU) | 2.18 (acc .78) → 2.36 (acc .81) | 0.93 → 1.54 (+XLA timer warning) |
| 13.5 allreduce/psum | star copy 20.73 GB/s (25.9 ms) | psum 4.47 GB/s |
| 13.6 DDP samples/s k=1/2/4 | 2107 / 3710 (1.76×, **88%**) / 6911 (3.28×, **82%**) | (no jax run exists) |
| 13.7 rungs tokens/s | R0 221,455 → R1 229,778 (1.04×; first compile 1.9 s) → R2 388,715 (1.69×) → R3 621,021 (1.60×, 8.7 GiB) → R4 545,042 (0.88×, 3.1 GiB). **Cumulative R0→R3 = 2.80×** | (13.7 is pytorch-only) |

Also: NCCL allreduce busbw ≈ 2.2 GB/s on this box (framework audit); the
author-side pilot measured DDP comm ≈16 ms/step at k=2, ≈25 ms at k=4 via a
no_sync comparison (pilot notes — NOT in any notebook).

> **SUPERSEDED (fix pass, 2026-07-20):** the 13.7 rows above were later
> shown to be measurement artifacts — a dynamo retrace from the ragged
> final batch inside the timing window, plus torch.profiler's CUPTI
> instrumentation taxing subsequent timings ~16%. Clean measurements
> (shape-stable stream + CUPTI teardown): R0 ≈264k, R1 compile ≈1.31×,
> R2 bf16 ≈1.4×, R3 batch-up ≈1.3×, cumulative ≈2.4×. See
> `reviews/fixlog-agentC.md` for the forensics; final numbers come from
> the recapture.

---

## 2. Findings by file

### 2.1 `performance-model.md` (13.1) — agent A

- **PM-1 (blocker, code).** JAX timing-trap cell (`…-measuring-without-lying-1`):
  no `block_until_ready()` between the naive and the synchronized loop, so
  the "honest" window drains the ten in-flight naive matmuls → printed
  15.83 ms vs the true ≈8.5 (next cell's Benchmark: 0.85 ms/call). Add
  `jnp.dot(a, a).block_until_ready()` (or block on `b`) before the second
  `t0`. Re-run to confirm the two cells now agree.
- **PM-2 (blocker, code).** Read-once demo (`…-measuring-without-lying-2`):
  the JAX captured output is INVERTED (read-every 0.842 s, read-once
  2.217 s) because appending 1000 device scalars and `jnp.stack`-ing them is
  its own pathology. Redesign BOTH tabs to the identical experiment:
  accumulate a scalar **on device** (`s = s + y` / `s += y` with `s` a
  0-dim device tensor), one host read at the end; versus `.item()`/`float()`
  per step. Scratch-run both tabs; write the prose to what you measure. If
  JAX's win is small (unjitted dispatch dominates), say that — the rule
  ("keep reads out of the hot loop; sync at logging boundaries") survives
  either way. PyTorch's `s += y.item()` reads once per iteration (the
  external feedback's "twice" is wrong — don't "fix" that).
- **PM-3 (blocker, prose).** Sweep narrative vs data. The prose claims the
  measured crossover sits "near n ≈ 500" and the middle sizes climb
  "roughly in proportion to n". Measured: n=512 gives 16.2 TF (10% of
  roof) though its nominal intensity (~171) is already above the ridge; the
  knee is nearer 2048; 512→1024 jumps 7×. Rewrite "Read the plot": the
  roofline is a **ceiling**; the intensity model says n≈500 stops being
  bandwidth-limited *in principle*, but small shapes cannot fill 128 SMs
  and pay launch overhead, so the measured curve approaches the roof only
  around n≈2048 — the gap between the nominal crossover and the measured
  knee IS the utilization/overhead lesson, and it previews the profiler.
  Change :eqref:`eq_roofline` to a bound
  (`performance ≤ min(P, I·β)`) and keep the prose's "at most". Also
  "a few percent of peak" at n=256: true for pytorch (1.3%), not jax
  (0.24%) — say "a percent or less of peak".
- **PM-4 (prose).** Ridge-point rounding: printed 164, caption "about 165",
  prose "roughly two hundred" (twice: §Counting and Summary). Use "about
  165 FLOP/byte" for our card and "one-to-several-hundred on modern
  accelerators" for the general claim.
- **PM-5 (code, minor).** Profiler cell: (a) pytorch emits the captured
  "Profiler clears events" UserWarning — restructure (e.g. profile a fixed
  number of steps with `schedule=` or pass `acc_events=True`) so the
  rendered output is clean; (b) add one caveat sentence: CPU and CUDA
  totals overlap across threads/streams, so the ratio is a coarse regime
  heuristic, not a utilization figure — the timeline view is the real tool;
  (c) the JAX tab's only output is "trace files: 2", which teaches nothing —
  print something meaningful (e.g. the top of
  `jax.profiler`-collected data is hard; acceptable: state the trace was
  written and show how to open it in Perfetto, and move `import os` to the
  imports cell or drop it).
- **PM-6 (prose, minor).** "The `d2l` training loops … already do this" —
  verify or soften. The "synchronize once per minibatch at most" rule is
  fine; add the clause "and only when the host actually needs the value".
- **PM-7 (slides).** "Same Chip, 50× Apart" title: measured gap is 75×
  (pt) / 400× (jax). Retitle without the constant (e.g. "Same Chip, Two
  Orders of Magnitude Apart"). After PM-2, re-check the two slides that
  embed the affected cells still read correctly.
- **PM-8 (prose, minor).** Where the imports-cell comment cites
  `:numref:` inside code it renders literally — acceptable repo convention,
  leave as is (noted so nobody "fixes" it into broken markup).

### 2.2 `hardware.md` (13.2) — agent A

- **HW-1 (blocker, prose).** JAX H2D: the cell prints **0.9 GB/s** under
  shared prose claiming "tops out near PCIe's practical ceiling — tens of
  GB/s". Fix with tab-conditional prose: pytorch tab keeps the
  pinned-vs-pageable story (13.4/24.3 GB/s); jax tab states honestly that
  `device_put` from pageable NumPy includes JAX's host staging and lands
  far below the bus limit — the staging path, not the wire, sets the rate
  (and that is itself the lesson: control the staging or keep data
  resident). If you can find a *supported* faster path at jax 0.10.2
  (e.g. `jax.device_put` of a pinned buffer), you may demo it, but do not
  hack internals.
- **HW-2 (blocker, prose).** The generation "law" is refuted by the
  chapter's own table: H100→B200 is bf16 2.28×, BW 2.39×, capacity 2.4×,
  ridge **falls** 295→281; 4090→5090 ridge falls 165→117. Rewrite the
  "trend to memorize" and the shoreline section's closing claim: the
  4×/2×/1.7× figures are a long-run average across many generations; the
  ridge has *risen over the long run* but individual product steps can
  move either way (B200's packaging bought bandwidth faster than compute).
  Turn the contradiction into the lesson (packaging/market choices move
  either roof); update the Summary bullet + shoreline slide accordingly.
  Exercise 1 ("which direction has it moved… why is that the expected
  direction") must be re-posed — the honest answer is "it fell".
- **HW-3 (blocker, table).** Spec table `tab_gpu_specs`:
  (a) **MI355X: 5,000 TF bf16 is the sparse figure; dense is ≈2,500 TF**
  (AMD spec page) — fix the cross-vendor sentence;
  (b) **RTX 5090 fp4 1,676 TF breaks the table's own dense/fp32-acc
  convention** (4× its fp8 419 where every other pair is 2×) — verify
  against the RTX Blackwell whitepaper / MLSS dossier (`gh repo clone
  smolix/mlss-efficiency` if needed) and either correct (likely 838) or
  footnote the accumulate mode;
  (c) verify **B200 memory 192 GB vs the 180 GB shipping spec** and per-GPU
  power; cite what NVIDIA's current datasheet says;
  (d) add a one-line convention footnote under the table (dense, fp32
  accumulation, boost clocks, per-direction links) and re-check the ridge
  row after any edits. Web access is allowed for this.
- **HW-4 (prose, minor).** Package physics: "sits micrometers from the
  die" (§Where Bytes Live) contradicts the correct "millimeters … on an
  interposer" later — fix to on-package/millimeters. "Every chip boundary
  costs roughly an order of magnitude" — keep as stated rule of thumb but
  it's already labeled as such; fine. "Network storage extends without
  bound" → "effectively unbounded capacity, at NIC speed".
- **HW-5 (prose, minor).** GeForce P2P: "consumer GeForce cards have P2P
  disabled" is false historically (RTX 3090 had NVLink) — scope to
  "current GeForce generations (RTX 40/50)".
- **HW-6 (prose, minor).** Format ladder: exclude tf32 from "every halving
  wins twice" (tf32 keeps 32-bit storage: throughput win, no byte win —
  the figure already draws this; one clause in prose). Soften "can be
  swapped without overflow" → "same normal range; you lose precision, not
  magnitude". Mark E4M3/E5M2 roles as the common recipe, not a law.
  "every kernel the GPU runs was launched by a CPU thread" → add "(until
  the capture-and-replay of :numref:`sec_compilation`)".
- **HW-7 (prose, minor).** Prefill/decode: "Prefill is compute-bound" →
  "long, well-batched prefill is typically compute-bound"; keep the decode
  arithmetic as is (it's correct and conditioned).
- **HW-8 (slides).** "Two Ladders" slide shows one ladder — retitle or add
  the second image. Update any slide text touched by HW-2/HW-3.

### 2.3 `compilation.md` (13.3) — agent A

- **CO-1 (blocker, prose).** Summary bullet says fusion buys "roughly
  2×"; measured 8.3–8.6×, and the body/slide correctly say "close to an
  order of magnitude". Fix the bullet.
- **CO-2 (blocker, prose+code).** "Whole-Step Compilation, Measured"
  doesn't compile the whole step: pytorch compiles the model only
  (optimizer stays eager); jax compiles `value_and_grad` with **no
  update**. Either (preferred) make the JAX tab a genuine jitted train
  step (add the SGD update inside the jitted function via `jax.tree.map`)
  and have the PyTorch prose state plainly that `torch.compile(net)`
  captures forward+backward while `opt.step()` remains eager (that
  asymmetry is real and worth teaching), and retitle the subsection
  honestly (e.g. "Compiling the Training Step — Measured", with the text
  saying exactly what is captured in each framework); or rename to
  "Compiling Forward and Backward". Also fix the first-call prose: the
  captured pytorch first step is **0.4 s** (warm inductor cache), not
  "seconds" — write "a visible one-time cost — fractions of a second here,
  seconds on real models (:numref:`sec_fast_transformer` measures ~2 s)".
- **CO-3 (blocker, exercise).** Exercise 2 proposes `static_argnums` as a
  fix for varying input lengths — static args are cache keys, so every new
  value recompiles; it is not a fix. Rewrite: padding/bucketing as the fix;
  `static_argnums` reserved for genuinely constant configuration, with the
  cache-key behavior stated (a two-call trace-counter mini-experiment is a
  nice optional flourish).
- **CO-4 (blocker, slides).** "Breaks vs. Retraces" slide references
  `@compilation-capture-two-philosophies-1` and `-2` unscoped; -1 is
  pytorch-only, -2 jax-only, so each deck silently drops half the slide
  (verified in `_slides/*/…/compilation.qmd`). Scope them
  `@…-1@pytorch` / `@…-2@jax`.
- **CO-5 (code, minor).** Overhead demo: the pytorch 60-layer stack has no
  nonlinearity (jax tab uses tanh) — insert the same nonlinearity so the
  tabs match and the model isn't a single linear map.
- **CO-6 (code, minor).** Add a cheap parity invariant before the fusion
  timing: `torch.allclose(gelu_ish(x), compiled(x), …)` /
  `jnp.allclose(...)` with a stated tolerance — "same answer, then faster".
- **CO-7 (prose, minor).** (a) Triton is "one important Inductor backend"
  not "`torch.compile`'s own code generator" (Inductor also uses
  template/library kernels); (b) "compile it anyway (it costs nothing at
  steady state)" contradicts the section's own checklist — replace with
  "compiling it is usually harmless, but measure — compilation can also
  regress memory or time"; (c) `torch.export` and StableHLO are two export
  paths, not one shared representation — split the sentence; (d) "The
  Graph Was Always There": one clause acknowledging autograd records a
  backward tape (not a whole-program compiler IR) and that not every node
  is one kernel (views launch none, library calls fuse internally).
- **CO-8 (prose, minor).** Where the JAX whole-step gap is ~100×
  (17.40→0.17 ms), add the one-sentence real lesson for jax readers: never
  run an unjitted training step — eager JAX exists for debugging.

### 2.4 `index.md` — agent A

- **IX-1 (prose).** "cures the bandwidth and overhead regimes" → "targets"
  / "can collapse", plus one clause foreshadowing the capstone's honest
  null result ("and section 13.7 shows a case where it buys almost
  nothing — diagnosis first"). "There is one map" → "one map to start
  from" (the overhead regime lives beside the roofline, not on it).
  "the difference is almost never the algorithm" → keep the conditional
  explicit ("holding the model and its loss curve fixed, the difference is
  systems, not mathematics").
- **IX-2 (prose).** Add one sentence to the section tour or scope fence:
  the capstone case study is PyTorch-only (it exercises torch.compile,
  autocast, checkpointing, and DDP as one stack); JAX readers get the same
  method via 13.3–13.6 and the exercises.

### 2.5 `memory-precision.md` (13.4) — agent B

- **MP-1 (blocker, code+prose).** Mixed-precision demo: the shared prose
  narrates the pytorch three-timing story ("about one and a half times…
  the three timings"); the jax tab prints two timings at 1.95×, and its
  "fp32" baseline is almost certainly already tf32-class (13.12 ms beats
  pytorch's tf32 15.43). Fix by making the JAX tab tell the same
  fair-baseline story: time three configurations —
  `jax.default_matmul_precision('highest')` (true fp32),
  default/`'high'` (tf32-class), and bf16 — verify on-box which the
  default actually is, and state it. Then the shared prose can say "about
  1.5–2×, framework- and shape-dependent" and both tabs match the
  fair-baseline lesson. Note `default_matmul_precision` controls dot
  *compute* precision, not storage dtype — one clause.
- **MP-2 (blocker, code).** Checkpointing: the slide claims "~⅓ more time"
  and the prose "a modest increase in step time" but the cell measures
  memory only. Add a completion-timed fwd+bwd measurement to both tabs
  (d2l.Benchmark on `run(...)`+backward pytorch; jitted grad jax), then
  quote what you measure. Also: (a) clear grads/state between the two
  pytorch runs so the peak comparison is clean; (b) the jax init reuses
  the same key for W1 and W2 of each block (identical matrices!) — split
  32 keys; (c) "never for speed (it *costs* speed)" → "normally costs
  time; its purpose is memory" (rare cache-effect wins exist).
- **MP-3 (prose).** 403 vs 524 MB: add 2–3 sentences reconciling the gap
  (activations ≈13 MB, autograd/cuBLAS workspaces, allocator rounding —
  print `torch.cuda.max_memory_reserved()` too if it helps) and name
  allocated-vs-reserved once, cross-ref the builders-guide memory section.
  Frame 16P as "this configuration's floor" (fp32+Adam), not the anatomy;
  one clause that activations-dominate is workload-dependent.
- **MP-4 (prose).** Gradient accumulation: state the equivalence
  conditions in one sentence (additive loss, equal-size micro-batches or
  weighted means, controlled RNG/dropout, no batch-coupled state like
  BatchNorm, clip/step at the global point); replace "same work in the
  same time" with "same arithmetic, more launches and smaller GEMMs —
  usually slightly slower, by this chapter's own intensity logic".
  Update the Ladder list and slide wording to match (compile: "usually
  free"; bf16: "~1.5–2×"; accumulation: "≈ same wall-clock, never
  faster").
- **MP-5 (exercise).** Exercise 4 "watch the loss go to NaN" is not
  deterministic — re-pose: construct gradients small enough to underflow
  fp16 (or ask students to report whether/where failure appears and why it
  is workload-dependent).
- **MP-6 (prose, micro).** "they are what stand between" → "what stands";
  JAX `memory_analysis` is the compiler's *plan*, don't compare digits
  directly against pytorch's allocator peak (one clause; mostly already
  there).

### 2.6 `multiple-gpus.md` (13.5) — agent B

- **MG-1 (BLOCKER, code).** JAX `train_step` sums per-shard **mean**
  gradients with `psum` and never divides by k → effective lr is k·lr at
  k GPUs, while the prose claims "mathematically identical". Fix:
  `jax.lax.pmean(grads, 'data')` (semantics then match the pytorch tab:
  global-mean gradient × lr). Add a one-step equality invariant cell in
  BOTH tabs: same init, same batch — k=1 vs k=2 parameter deltas allclose
  (this is cheap and is the proof of "identical"). Also print test
  accuracy in the jax `train()` (pytorch already does) or restrict the
  accuracy sentence to the pytorch tab via :begin_tab:.
- **MG-2 (BLOCKER, prose).** Editing artifact left in the text
  (§Accounting): "big models (large $N$-relative-to-work... no: large
  *compute* per byte communicated)" — rewrite the sentence cleanly
  ("models with high compute per byte of gradient traffic, large
  per-device batches, and fast links all push in your favor").
- **MG-3 (blocker, prose+figure).** "independent of $k$" for ring traffic
  (prose, summary, slide, AND `fig_ring_allreduce` text in
  `gen_mdl_perf_figures.py`): $2(k{-}1)N/k$ *approaches* $2N$; say
  "nearly constant — bounded by $2N$ — as $k$ grows" everywhere, keep the
  exact factor. Regenerate the figure (idempotence check per §0.7). Add
  one sentence defining the convention (bytes sent per device per
  allreduce; NCCL's "busbw" is a different normalization) — it's needed
  before quoting 20.7 vs 2.2 GB/s. Optional single sentence acknowledging
  the latency (α) term for small messages; do not build the full α–β
  model.
- **MG-4 (blocker, prose).** Accounting paragraph vs jax tab: "the
  hand-rolled copy sustains on the order of ten GB/s" (a) understates the
  measured 20.7 ("tens of GB/s", as the summary already says) and (b) on
  the jax page the printed number is psum 4.47 GB/s, which is not a
  hand-rolled copy. Restructure with :begin_tab: or neutral phrasing so
  each tab's paragraph matches its own measurement; keep the
  NCCL-busbw-is-lower theory-vs-practice aside (pytorch tab).
- **MG-5 (prose+code, minor).** (a) Intro "buys both more compute and more
  memory at once" → "more arithmetic and more aggregate memory" (plain DP
  does not enlarge the model — the section itself says so later);
  (b) intro "on a datacenter rack … the accounting feels academic" →
  invert honestly (fast fabrics shrink one constant; at frontier scale the
  accounting matters more, not less — our slow fabric just makes it loud);
  (c) per-batch `torch.cuda.synchronize()` inside the epoch loop → sync
  once per epoch at the timer boundary; (d) "we elide them here (they live
  in the notebook)" — the loops are fully shown right below; delete the
  clause; (e) XLA "Delay kernel timed out" warning in the jax 1-GPU output
  — add a warmup epoch or note; try to make the rendered output clean.
- **MG-6 (prose).** Add the strong-vs-weak scaling definitions here, in
  §Accounting (eq_dp_cost is a strong-scaling model — fixed global B).
  Two sentences + pointer that 13.6's throughput sweep holds per-rank
  batch fixed (weak scaling) and must be read under that convention.
  (13.6 does the labeling on its side; agent C owns that file.)
- **MG-7 (prose, minor).** "every production collective library is built
  on rings (and trees…)" → "built on rings and trees, chosen per message
  size and topology, plus hardware-specific schemes"; parameter-server
  history: "lives on today mainly in recsys embeddings" → keep recsys as
  the book-relevant example, drop the exclusivity ("and wherever sparse or
  asynchronous state dominates").

### 2.7 `multi-gpu-practice.md` (13.6) — agent C

- **MGP-1 (BLOCKER, code).** DDP hygiene: (a) pre-download Fashion-MNIST
  once in the parent notebook (quiet — suppress the tqdm/percent stream)
  before any `torchrun`, so ranks never race and the captured output has
  no download noise (the current capture carries ~5.4 KB of "0.1%0.2%…");
  (b) suppress the bare `1470` echo (`_ = pathlib.Path(...).write_text(...)`);
  (c) write sidecar script + per-rank JSONs under a dedicated subdir
  (e.g. `ddp_scratch/`) and clean stale rank files before each launch so a
  crashed run can't serve old results; use context managers for file IO.
- **MGP-2 (BLOCKER, prose).** Strong vs weak scaling: the sweep fixes
  per-rank batch 256, so it is weak scaling, while 13.5's eq_dp_cost is a
  strong-scaling model. Name the convention where the sweep is introduced,
  present efficiency as weak-scaling efficiency, and either (preferred)
  add the small fixed-global-batch (strong-scaling) comparison at
  k∈{1,2,4} — it is one more torchrun sweep with per-rank batch 512/k —
  or explicitly derive how the weak-scaling numbers relate to the model.
  Also: the cell prints "82% efficiency"; prose/slide say "about 85%" —
  say 82% (the print is on the same page). Note global batch changes the
  optimization trajectory (one sentence + pointer to
  :numref:`sec_batch_size`).
- **MGP-3 (BLOCKER, code).** The `no_sync()` claim ("within about 20% of
  what a no_sync()-versus-synced measurement reports") references a
  measurement that exists only in pilot notes. Add it: in the sidecar
  script (or a variant), time synced steps vs `with model.no_sync():`
  steps at k=2, dump both, and let the notebook print measured comm-time
  vs the eq_dp_cost prediction (pilot saw ≈16 ms at k=2 — hedge prose to
  "within tens of percent"). State the caveat that the subtraction is an
  estimate (overlap changes). If this proves too fragile, delete the claim
  entirely — an invisible experiment must not carry the paragraph.
- **MGP-4 (BLOCKER, code).** JAX: the declarative-sharding climax is
  currently defined but never run. Make it real:
  (a) instantiate the nnx `ResNet18` + optimizer, run the jitted
  `train_step` on a sharded batch; (b) `visualize_array_sharding` of a
  weight before/after (replicated) as well as the batch (sharded);
  (c) the receipt: inspect the lowered/compiled step
  (`jax.jit(...).lower(...).compile().as_text()` or
  `…​.lower().as_text()`) and show/grep the inserted `all-reduce` — "the
  compiler wrote the collective" made visible; (d) a measured throughput
  sweep k∈{1,2,4} on the same ResNet/Fashion-MNIST-64 (weak scaling,
  matching the DDP convention) — the spec ("same ResNet, same k-sweep,
  measured") requires it, runtime ~few minutes; (e) one-step parameter
  delta vs a single-device reference (invariant), mirroring MG-1;
  (f) move `import optax` to the imports cell. Scratch-verify on your
  assigned GPUs; note the scheduler runs this notebook whole-box (all 4)
  at capture.
- **MGP-5 (blocker, prose).** FSDP: "the deprecated `FullyShardedDataParallel`
  wrapper is gone at our pin" is FALSE (verified importable at torch
  2.11.0) — say deprecated/legacy, `fully_shard` is the modern surface.
  "past a few billion parameters" → the memory inequality (when
  params+grads+states+activations at your precision exceed one GPU, or
  when the redundancy is worth trading for communication); keep the
  few-billion figure as the typical scale for this card class.
  Caption of `fig_fsdp_lifecycle`: add "one simplified
  reshard-after-forward policy".
- **MGP-6 (blocker, exercises+refs).** Exercise 2 references a "fork-based
  fallback harness from :numref:`sec_multi_gpu`'s discussion" and a
  "CUDA-context rule of :numref:`subsec_hw-interconnects`" — neither
  exists. Replace the exercise (good substitute: strong-vs-weak — rerun
  the sweep at fixed global batch and reconcile the two efficiency
  numbers; or bucket_cap_mb is ex. 1, so a no_sync-based comm-time
  measurement at k=4). Fix intro line "the reasoning behind it is in
  :numref:`subsec_hw-interconnects`" → point at the box topology (13.2)
  for *why the fabric is slow* and 13.5 for *what the collective costs*.
  Exercise 5: "~85%" → 82%.
- **MGP-7 (prose, minor).** (a) "the only lines that differ … are the
  three that set up DDP" — the script visibly also needs the
  DistributedSampler, set_epoch, rank-local device, teardown; say "the
  *loop body* is unchanged; the scaffolding around it is the launcher's";
  (b) PartitionSpec punchline: soften to "one sharding vocabulary spans
  what PyTorch exposes as three APIs — though an effective plan is still
  model-aware (layouts, constraints)"; (c) bridge: "the ideas are the same
  and only the scale changes" → name what multi-node adds (hierarchical
  topology, rendezvous, failure/restart, stragglers) in one sentence;
  (d) optional: the 4-row collective table (allreduce / reduce-scatter /
  all-gather / all-to-all) — cheap and clarifying, add if it fits;
  (e) update Summary + slides for weak-scaling labeling, 82%, the now-real
  JAX demo (slide can embed the sharding-visualization or sweep cell).

### 2.8 `fast-transformer.md` (13.7) — agent C

- **FT-1 (BLOCKER, code).** Rung 5 is a phantom: "the measurement … is
  left to the notebook" but no cell exists; the slide claims "the
  measurement confirms the prediction". Implement it: sidecar DDP script
  for `d2l.GPT` (d=512, same data; pre-download TimeMachine in parent),
  torchrun at k=2 and k=4 (guard on `d2l.num_gpus()`), tokens/s per
  config, printed beside the eq_dp_cost prediction (predict FIRST in
  prose, then measure — the agreement is the payoff). Bounded: ~1–2 min
  per k. Keep the waterfall's single-GPU rungs as the plot; add R5 as
  its own cell/plot or annotated line clearly marked multi-GPU. Update
  prose/slide/summary to the measured numbers (hedged per policy).
- **FT-2 (BLOCKER, code).** The "short real training run" is also a
  phantom ("the notebook trains a few hundred steps and watches the loss
  fall" — no such cell). Add it: a few hundred steps with the fast
  configuration (compiled+bf16+batch-up), print first/last smoothed loss
  (assert it fell); one line of prose: "speed that breaks the model is not
  speed".
- **FT-3 (BLOCKER, prose).** Cumulative: "roughly 2.5×" (prose, summary,
  slide) vs measured 2.80× baseline→batch-up. Say "about 2.8×" (or
  "nearly 3×") and prefer computing the ratio in the waterfall cell so
  prose can't drift again.
- **FT-4 (blocker, code).** Move `loss.backward()` OUT of the
  `torch.autocast` context in `step_bf16` and `step_ckpt` (13.4 already
  does it right; AMP docs recommend forward+loss only).
- **FT-5 (blocker, prose+code).** Rung 3's causal story ("bf16 freed the
  memory that buys batch 512") is unsupported — fp32 batch-512 fits
  comfortably on a 24 GB card (~14 GiB). Either add the cheap control
  (measure fp32-512 peak memory once) and reframe honestly ("bigger batch
  is its own rung; bf16 makes the headroom generous and will matter when
  activations are the constraint"), or rewrite the rung intro to the
  intensity story only. Also state once that `throughput` /
  `throughput_big` time the full pipeline including `next(stream)` and
  H2D — it is an end-to-end tokens/s, which is the honest metric here.
- **FT-6 (blocker, code).** Correctness gates: (a) parity assert
  `CheckpointedGPT(model)(X)` vs `model(X)` allclose (the mirror is exact
  today — `if m.pos == 'learned'`… `F.linear(m.norm(H), m.token_emb.weight)`
  — but pin it against drift); (b) eager vs compiled logits allclose with
  stated tolerance before R1's timing.
- **FT-7 (prose).** Re-profile after material rungs, as the section
  promises ("each re-profiled"): add compact profiler prints (row_limit≈5)
  after R1 and R3, or weaken the promise to match reality. Clean the
  captured torch.profiler UserWarning (same recipe as PM-5a).
- **FT-8 (prose, minor).** (a) Thermal paragraph: "can be twice as fast as
  its steady state" is an unverifiable author-side observation — either
  soften ("materially faster") or print measured clocks
  (`nvidia-smi --query-gpu=clocks.sm --format=csv` before/after warmup)
  once; (b) PyTorch-only signpost sentence in "The Subject" (pairs with
  IX-2); (c) label the waterfall cumulative ("each bar inherits every
  choice to its left") in caption/prose; (d) R1's "~4%" on the slide →
  "a few percent".

---

## 3. Cross-cutting calibration pass (whichever agent touches the file)

- Replace absolutes with their conditions where flagged: "arithmetic is
  free below the ridge" → "extra fused arithmetic hides under the memory
  time while bandwidth binds"; "every halving wins twice" (exclude tf32);
  "prefill is compute-bound" (add "typically, at real context lengths").
- Reduce the frequency of "honest/honestly" (~15 occurrences chapter-wide;
  keep it in 2–3 load-bearing places, elsewhere name the property:
  synchronized, completion-timed, measured-not-asserted).
- "physics" as rhetorical closure ("the same physics", "both report the
  same physics") — keep at most once where it is literally physics
  (energy); elsewhere say "the same design point / the same accounting".

## 4. Verified-valid external findings already folded in above

psum/pmean (MG-1) · phantom rung 5 + learning run (FT-1/2) · 2.8× (FT-3) ·
sweep narrative (PM-3) · read-once inversion (PM-2) · H2D jax (HW-1) ·
no_sync invisibility (MGP-3) · FSDP wording (MGP-5) · weak-vs-strong
(MGP-2/MG-6) · generation-law self-contradiction (HW-2) · MI355X dense
(HW-3) · whole-step naming (CO-2) · static_argnums exercise (CO-3) ·
backward-in-autocast (FT-4) · bf16→batch control (FT-5) · dup jax keys +
untimed checkpointing (MP-2) · accumulation conditions (MP-4) ·
ring k-dependence (MG-3) · micrometers (HW-4) · GeForce P2P scope (HW-5) ·
Triton/export/compile-anyway (CO-7) · profiler heuristic caveat (PM-5).

## 5. External feedback assessed and REJECTED (do not "fix" these)

1. "PyTorch loop reads `y.item()` twice per iteration" — false; once.
2. "Never advise synchronizing once per minibatch" — the chapter's
   "at most once, for metric reads" rule stands; refine only (PM-6).
3. `pmap` "compatibility shim" phrasing — accurate at our pin; keep.
4. Feedback's "P + P + 2P + 12P = 16P" arithmetic — garbled; the
   chapter's 4P+4P+8P is correct.
5. Wholesale bans ("remove 'honest' throughout") — apply the reduction of
   §3, not a ban.
6. Scope expansions (metrics/MFU table, hierarchical-roofline figure, α–β
   collective model, benchmark-provenance objects, sourced-CSV spec table,
   core/advanced track marks, shoreline second panel) — good ideas,
   **deferred to Alex**; do not implement in this pass beyond the single
   sentences specced above.

## 6. Do-not-touch list

- `#@save` bodies: `Benchmark`, `split_batch` (pt+jax), `resnet18`,
  `ResNet18` — byte-identical, no exceptions (§0.2).
- `legacy-multigpu-lib.md` — frozen.
- Existing cell `#id`s; `_quarto.yml`; `CHAPTER_NUMBERING`; labels
  (`sec_*`, `fig_*`, `eq_*` all have inbound references).
- Figures other than `fig_ring_allreduce` text (MG-3); figure *captions*
  in .md may change per spec.
- Other chapters (the three serving softenings etc. are done and out of
  scope).

## 7. Recapture and build plan (run centrally after content lands)

Code-cell changes → stale: pytorch 13.1, 13.3, 13.4, 13.5, 13.6, 13.7;
jax 13.1, 13.3, 13.4, 13.5, 13.6. 13.2 is prose/table-only in both tabs
(HW-1 via :begin_tab: prose — confirm no code cell changed; if one did,
add it). Then: `rm` the relevant `_notebooks/*/…/*.executed` stamps →
`tools/notebook_scheduler.py` run (13.6/13.7 whole-box; scheduler owns
concurrency) → require "done: 0 failed" → `make capture-outputs FILES=…` →
`tools/audit_outputs.py --verify-fresh` green → reconcile every prose
number in the RECONCILE lists against blessed outputs → `make figures`
idempotence → slides for the chapter → `make html` → `make pdfs` →
`make check-all-artifacts` → commit on `rnn-ssm-modernization`.
