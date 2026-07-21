# Fix log — agent A (performance-model.md, hardware.md, compilation.md, index.md)

Session start 2026-07-20. Spec: reviews/comp-perf-review-findings-2026-07-20.md §2.1–§2.4, §3.

## Status

- **PM-1 done** — added `b.block_until_ready()` before second timer in jax
  timing-trap cell + one prose sentence ("start and stop at a known-quiet
  device"). Scratch (GPU0, .venv-jax): naive 0.42 ms / honest 8.58 ms,
  Benchmark-style 0.86 ms/call — the two cells now agree.
- **PM-2 done** — both tabs redesigned to identical device-scalar
  accumulation (`s += y` / `s = s + y`, one `.item()`/`float()` at end).
  Scratch: pt 0.061 → 0.040 s (~1.5×), jax 0.867 → 0.317 s (~2.7×) —
  inversion gone, read-once wins in BOTH tabs. Prose hedged ("roughly a
  third … around two-thirds").
- **PM-3 done** — eq_roofline now `≤`; "Read the plot" rewritten around
  ceiling vs measured knee (n≈500 nominal vs n≈2048 measured; n=256 "a
  percent or less"); sweep intro + summary bullet + sweep slide updated.
- **PM-4 done** — "about 165 FLOP/byte" for our card; "one to several
  hundred" for the general claim (prose, summary, ridge slide).
- **PM-5 done** — (a) `acc_events=True` kills the "Profiler clears events"
  UserWarning (scratch-verified clean table); (b) coarse-ratio caveat
  sentence added; (c) jax tab now prints trace location + Perfetto pointer,
  `import os` dropped.
- **PM-6 done** — verified in d2l/torch.py (Module.plot hands a lazy
  callable to the board's background drawing thread); sentence now states
  that + "only when the host actually needs the value" added to the rule.
- **PM-7 done** — slide retitled "Same Chip, Two Orders of Magnitude
  Apart"; both slides embedding -1/-2 cells re-read post-fix, coherent.
- **PM-8** — left as is per spec.
- §3 calibration (perf-model): "adding FLOPs is free"→"extra arithmetic
  hides under the memory time" (×2 + slide); "honest" reduced to opener
  (+1 slide echo); "cures"→"targets"/"attacks".
- **HW-1 done** — prose-only (no code cell touched; 13.2 stays prose-only
  per §7). Split into :begin_tab: pytorch (pinned≈2× pageable, near PCIe
  ceiling) / jax (device_put from pageable NumPy ≈1 GB/s; staging path,
  not the wire, sets the rate; keep data resident). Re-verified on box:
  1.1 GB/s. No supported faster path demoed (pinned-buffer device_put not
  in jax 0.10 public API).
- **HW-2 done** — trend paragraph rewritten (long-run averages vs single
  steps; H100→B200 lockstep ~2.3×, ridge edges down; 4090→5090 ridge falls
  ~1/3; "engineering picks which wall to push"); intro line, summary
  bullet, "Where the Two Numbers Come From" + shoreline slides updated;
  Exercise 1 re-posed (which direction did it move + reconcile).
- **HW-3 done** — table: B200 192→180 GB, 8.0→7.7 TB/s, power→1,000 W,
  ridge ~281→~292; conventions footnote added (dense/fp32-acc/boost,
  HGX-vs-GB200 SKU, 5090 fp4 4× explanation); MI355X sentence → 2,500 TF
  dense bf16; 5090 fp4 stays 1,676 (whitepaper-confirmed fp32-acc dense);
  "The same physics"→"The same constraints". Sources in §Sources below.
- **HW-4 done** — on-package/millimeters + interposer wires; network
  storage "effectively unbounded capacity, at NIC speed"; "8 TB/s"→
  "nearly 8 TB/s" (B200 SXM is 7.7).
- **HW-5 done** — P2P claim scoped to RTX 40/50 with RTX 3090 NVLink
  parenthetical.
- **HW-6 done** — same-range/lose-precision-not-magnitude; E4M3/E5M2 as
  usual recipe; "every halving of storage wins twice" + tf32 exception
  (prose + slide); CPU-launch clause gains "(until capture-and-replay of
  sec_compilation)".
- **HW-7 done** — "long, well-batched prefill is typically compute-bound"
  + one-line-prompt counterexample; summary bullet + energy slide updated.
- **HW-8 done** — "Two Ladders" slide retitled "The Bandwidth Ladder";
  shoreline/format slides updated per HW-2/HW-6.
- **CO-1 done** — summary bullet "roughly 2×" → "close to an order of
  magnitude on our unfused elementwise chain".
- **CO-2 done (preferred option)** — jax tab is now a genuine jitted
  train step (`train_step` = value_and_grad + `jax.tree.map` SGD update,
  AOT-lowered). Scratch (GPU0/1): AOT 1.8 s, eager 17–19 ms → compiled
  0.18–0.19 ms — first-call-cost story intact, eager comparison kept.
  Section retitled "Compiling the Training Step, Measured" (label
  `subsec_comp-wholestep` unchanged; no inbound refs elsewhere — grepped).
  Intro states what each framework captures (torch.compile = fwd+bwd,
  opt.step eager; jax = whole step incl. update). First-call prose now "a
  fraction of a second to a couple of seconds on this toy step … about
  two seconds for :numref:`sec_fast_transformer`" (covers pt 0.3–0.4 s
  and jax 1.8 s prints). Slide retitled "Compile the Training Step" with
  the asymmetry named.
- **CO-3 done** — Exercise 2 re-posed: padding is the fix;
  `static_argnums` explained as a cache key (counter mini-experiment);
  ends "when IS static_argnums the right tool?". Body prose already said
  "genuinely-constant arguments" — untouched.
- **CO-4 done** — slide refs scoped `@…-1@pytorch` / `@…-2@jax`; syntax
  verified against tools/gen_slides.py placeholder grammar
  ("@cell-id@pytorch → force a specific framework variant") and live
  precedent in chapter_convolutional-modern/training-recipes.md.
- **CO-5 done, with a necessary twist** — pytorch overhead stack is now 60×
  (Linear+Tanh) matching the jax tab's tanh. Plain insertion regressed
  reduce-overhead to 10 ms/call: with autograd on, each Tanh's
  saved-for-backward activation escapes the CUDA-graph memory pool and
  forces re-recording EVERY call (TORCH_LOGS=cudagraphs showed per-call
  "Recording function=partition_0"; ReLU did not trigger it). Fix: time
  the forward under `torch.no_grad()` (also matches the jax tab, which
  records no autograd). Verified: eager 2.3 ms → reduce-overhead 0.31
  ms (~7×; story stronger than the old 4×). Prose explains the fixed-
  buffers requirement — turned the gotcha into the lesson.
- **CO-6 done** — `assert {torch,jnp}.allclose(gelu_ish(x), compiled(x),
  atol=1e-6)` added inside both fusion cells (doubles as the compile
  warmup; no new cell ids needed). Both pass on GPU. One prose sentence:
  same answer to 1e-6, then faster.
- **CO-7 done** — (a) Triton = "an important backend of torch.compile's
  Inductor, which also draws on template and library kernels"; (b)
  compile-anyway → "usually harmless, but measure — can also regress time
  or memory"; (c) torch.export and StableHLO split into two export paths;
  (d) tape-not-IR + not-every-node-is-a-kernel footnote added to "The
  Graph Was Always There".
- **CO-8 done** — :begin_tab:`jax` paragraph after the measurement:
  never run a training step un-jitted except to debug (measured ~90–100×).
- **IX-1 done** — "one map to start from"; "targets … can collapse both —
  though sec_fast_transformer shows a case where it buys almost nothing,
  which is why diagnosis comes first"; "holding the model and its loss
  curve fixed, that difference is systems, not mathematics"; "honest
  communication bill" → "measures the communication bill".
- **IX-2 done** — capstone-is-PyTorch-only sentence added to the section
  tour (JAX readers → JAX tabs of preceding sections + exercises).
- **§3 calibration across my files** — "honest/honestly" reduced to 3
  (perf-model opener + its slide echo, compilation "keep the picture
  honest"); "The same physics" → "the same constraints" (energy section
  untouched — it IS physics); absolutes conditioned as itemized above.
- **Verification** — full cell sequences of performance-model.md and
  compilation.md extracted from the edited .md and executed end-to-end in
  .venv-pytorch and .venv-jax (CUDA_VISIBLE_DEVICES=0/1): zero warnings,
  all asserts pass. `tools/add_cell_ids.py` run on all four files (0 new
  ids — all edits inside existing cells). `tools/lint_source.py` clean on
  all four. `#@save` Benchmark bodies byte-identical (diff-checked).
  hardware.md: NO code cell touched (13.2 stays prose/table-only per §7).

## Deviations / decisions

1. **HW-3b**: spec suggested 5090 fp4 "likely 838"; primary source (RTX
   Blackwell whitepaper Appendix A) shows 1,676 IS the dense fp32-acc
   figure — kept 1,676 and added the convention footnote explaining the
   4× step (GeForce halves fp8/fp16-with-fp32-acc; fp4 unpenalized).
2. **HW-3c**: B200 column now the HGX B200 shipping datasheet spec
   (180 GB, 7.7 TB/s, 1,000 W); ridge recomputed ~292 (spec's example
   said ~281 under the old 8.0 figure). HW-2 prose written to match
   (H100→B200 ridge "barely moved, edging slightly down" — 295→292 —
   with 4090→5090's ~1/3 drop carrying the visual).
3. **CO-5**: added `torch.no_grad()` around the reduce-overhead demo
   (see CO-5 above — required for correctness with tanh under
   cudagraphs, and aligns the tabs). Not in the spec's letter; in its
   spirit (tabs match, demo tells the true story).
4. **HW-1**: no faster jax H2D path demoed — pinned-host staging is not
   exposed in jax 0.10's public API; prose-only fix keeps 13.2 out of
   the recapture set.

## RECONCILE AFTER CAPTURE (agent A cells)

performance-model.md (pytorch + jax both stale):
- `#performance-model-measuring-without-lying-1` (jax code changed):
  prose says naive "close to nothing"; confirm jax honest ≈ Benchmark×10
  (scratch: 8.58 ms vs 0.86 ms/call).
- `#performance-model-measuring-without-lying-2` (both tabs redesigned):
  prose quotes "roughly a third" (pt ~1.5×) / "around two-thirds" (jax
  ~2.5–2.7×) wall-clock cut — re-check against blessed prints.
- `#performance-model-the-profiler` (both tabs changed): pt table must
  render warning-free (acc_events=True); jax now prints the
  trace-location line.
- `#performance-model-the-sweep-mapping-our-gpu` (code unchanged, prose
  re-anchored): knee ≈2048, n=512 "a tenth of peak or less" (pt ~11%,
  jax ~2%), n=256 "a percent or less".

compilation.md (pytorch + jax both stale):
- `#compilation-what-the-compiler-does-fusion` (both tabs: assert added):
  "close to an order of magnitude" (scratch pt 8.9×, jax 8.8×).
- `#compilation-whole-step-compilation-measured` (jax code changed; pt
  prose re-anchored): first-call prose "a fraction of a second to a
  couple of seconds" must cover pt first step (~0.3–0.4 s) and jax AOT
  (~1.8 s); jax tab "nearly two orders of magnitude" (scratch 90–107×).
- `#compilation-the-overhead-regime-capture-and-replay` (pt tab changed:
  tanh + no_grad): reduce-overhead must WIN (scratch: 2.3 → 0.31 ms,
  ~7×); if it comes back ~10 ms/call the cudagraph re-record pathology
  has returned — do not bless, ping me.

hardware.md / index.md: no code cells changed — 13.2 needs NO
re-execution (prose quotes store values: pinned ≈2× pageable; jax
device_put ≈1 GB/s, re-verified 1.1 GB/s on-box).

## Sources (HW-3)

- **RTX 5090 fp4**: NVIDIA RTX Blackwell GPU Architecture whitepaper V1.1
  (images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
  Appendix A, GB202 table: "Peak FP4 Tensor TFLOPS with FP32 Accumulate
  (FP4 AI TOPS): 1676/3352(sparse)". So **1,676 dense/fp32-acc is CORRECT**
  (not 838): GeForce halves fp8/fp16 tensor throughput with fp32 accumulate
  (fp8: 838 fp16-acc vs 419 fp32-acc) but fp4 carries no such halving →
  the fp8→fp4 step is 4× under the table's convention. Fixed via footnote,
  not a value change. Also confirms 4090 bf16 165.2 / fp8 330.3 dense
  fp32-acc; 5090 bf16 209.5 / fp8 419.
- **B200**: NVIDIA Blackwell datasheet ("NVIDIA Blackwell | Datasheet",
  mirrored at primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf),
  "Individual Blackwell GPU Specifications": HGX B200 GPU = **180 GB HBM3e |
  7.7 TB/s**, FP16/BF16 4.5 PF *with sparsity* (= 2.25 PF dense ✓), FP8 9 PF
  sparse (4.5 dense ✓), FP4 18 PF sparse (9 dense ✓), TDP **configurable up
  to 1,000 W**; GB200 NVL72 per-GPU variant = 186 GB | 8 TB/s, up to 1,200 W.
  Footnote: "All Tensor Core numbers except FP64 with sparsity."
  → table changed: memory 192→180 GB, bandwidth 8.0→7.7 TB/s, power
  1,000–1,200→1,000 W, ridge ~281→~292 FLOP/B; GB200 variant noted in the
  table footnote.
- **MI355X**: AMD Instinct MI355X GPU datasheet/brochure
  (amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf):
  BF16 **2.5166 PF dense**, 5.0332 PF w/ 2:4 sparsity; 288 GB HBM3E; 8 TB/s;
  TBP 1400 W. → cross-vendor sentence now says 2,500 TF dense bf16.
  (NB: the mlss-efficiency dossier's verification note claiming "5 PF dense
  confirmed" is wrong per AMD's own datasheet column layout; review spec §2.2
  HW-3a is right.)
- Dossier cross-check: smolix/mlss-efficiency ref/research/r1-gpu-specs.md
  (cloned to scratchpad) agrees on 5090 ladder + B200 SKU split.
