# Book coverage scan: what d2l-neu already teaches about compilation/performance/multi-GPU outside chapter_computational-performance

Scope: read-only scan of `chapter_*/*.md` (source of truth), `d2l/torch.py`, `d2l/jax.py`,
`_quarto.yml`, `tools/d2l_preprocess.py`, `reviews/ssm-chapter-review-and-proposal-2026-07-19.md`.
No repo files modified.

---

## 1. Compilation/JIT mentions (outside chapter_computational-performance)

### chapter_recurrent-modern/ssm.md (ch. 12.2 — the new SSM chapter)
- `:label:subsec_parallel-scans` "Parallel Scans" section (L142-343): teaches the
  Blelloch/Hillis-Steele parallel prefix scan as *the* mechanism that makes linear
  recurrence trainable on an accelerator. Implements `associative_scan` for both
  frameworks (`#@save`d into the d2l library): PyTorch hand-rolled doubling scan;
  JAX via the **primitive** `jax.lax.associative_scan(scan_combine, (a, b))`.
  `sequential_scan` uses `jax.lax.scan` as the reference/naive baseline.
- Benchmarks parallel scan vs. sequential loop from T=256 to T=16,384
  (`jax.jit(associative_scan)`, `jax.jit(sequential_scan)`, local `wall_clock`
  helper using `time.time()` + `.block_until_ready()`). Explicitly frames the
  result as the "trains like a CNN" moment for recurrence and is described as
  "everything else in this section and the next is built on top of it."
- Comment in a later cell: `# GPU matmuls default to TF32; run both schedules in
  full fp32 so the comparison isolates the reassociation error` (L1334) — teaches
  TF32 numerics awareness in passing.
- Chapter intro (`chapter_recurrent-modern/index.md`, L136-142) states explicitly:
  **"The chapter teaches algorithms, not kernels: the chunked forms here are
  twenty-line teaching implementations, and the Triton kernels, memory
  hierarchies, and serving systems that make them fast belong to
  `:numref:chap_performance`."** This is the master scope-fence/promise the new
  ch. 13 must honor.

### chapter_recurrent-modern/mamba.md (ch. 12.3)
- L477-490: a full paragraph on *why* Mamba shipped in 2023 not 2020 — because the
  authors' actual kernel "fuses discretization, scan, and read-out into a single
  pass that keeps intermediates in fast on-chip memory, and during backpropagation
  *recomputes* them from the small inputs instead of storing them" — explicitly
  named as the store-vs-recompute trade already met at `:numref:sec_bptt`, and
  states "it is half the reason the paper's title contains the word
  'hardware-aware.'" A slide is titled "Why 2023 and not 2020: hardware-awareness."
  This is a strong *conceptual* (not implemented) teaching of kernel fusion /
  memory-hierarchy awareness that the comp-perf chapter should build on, not repeat.

### chapter_recurrent-modern/hybrids.md (ch. 12.7, capstone) and matrix-state.md (12.4)
- hybrids.md L1010-1030: explicit deferral — "What this section did *not* cover is
  the systems story that makes the recurrent majority fast in practice — the
  chunked forms of `:numref:subsec_ms-chunked` living as fused kernels, and the
  serving stacks that exploit a mostly flat memory bill — which belongs to
  `:numref:chap_performance`."
- matrix-state.md L620-624: `jax.jit(matrix_state_recurrent)`,
  `jax.jit(matrix_state_dual)`, and a chunked form `forms[f'chunked ({C})'] =
  jax.jit(...)` — benchmarks recurrent/chunked/quadratic-dual formulations of the
  same computation against each other (a reassociation-of-arithmetic story, same
  genre as the ssm.md scan benchmark).

### chapter_attention/attention-at-scale.md (ch. 10.5) — the single richest overlap
- Full section "The Bottleneck is Memory Traffic" (L426-509): teaches that "on a
  modern GPU *moving bytes, not multiplying them, is the scarce resource*,"
  explains FlashAttention (`:cite:Dao.Fu.Ermon.ea.2022`) as chunked/online-softmax
  attention "engineered to the hardware: tile sizes matched to on-chip SRAM,
  softmax statistics kept in registers, and — during training — the backward pass
  *recomputing* the stripes instead of storing weights." States it "ships in every
  framework as a fused kernel: `torch.nn.functional.scaled_dot_product_attention`
  in PyTorch, `jax.nn.dot_product_attention` in JAX," and benchmarks the fused
  kernel against a naive implementation (local `wall_clock`/`peak_memory` helpers,
  L171-233) at B=2,H=8,n=8192,fp16 — reporting >10x speedup and orders-of-magnitude
  memory reduction (PyTorch), and a similar cuDNN-backed comparison in JAX using
  `jax.jit(f).lower(X).compile().memory_analysis().temp_size_in_bytes` (compiler
  memory introspection — real `jax.jit(...).lower(...).compile()` API usage, twice
  more at L198, L414).
- `chapter_attention/index.md` L116 cites FlashAttention explicitly as "the
  industrial form of `:numref:sec_attention-at-scale`'s chunked online-softmax
  cell."
- Summary (L981-1000) restates: "FlashAttention is this algorithm engineered to
  the memory hierarchy, and the fused kernels in both frameworks deliver its
  answer far faster and smaller than the naive schedule, because the true
  bottleneck of attention on modern hardware is memory traffic, not arithmetic."
- **Consequence for the new ch. 13**: FlashAttention/IO-awareness, kernel fusion,
  and "fused kernel" as a concept are already fully taught with a real benchmark.
  A rebuilt comp-perf chapter must not re-derive or re-benchmark FlashAttention —
  it can cite this section and build the *general* principle (fusion/memory-bound
  vs compute-bound reasoning) on top of it.

### chapter_transformers/scaling-laws.md (ch. 11.7)
- L218-220: `jax.jit(loss_fn).lower(params, X, Y).compile().cost_analysis()` and
  `jax.jit(jax.value_and_grad(loss_fn)).lower(...).compile().cost_analysis()` —
  uses XLA's compiled cost-analysis API to get real FLOP counts for a forward pass
  and a forward+backward pass, cross-checking the chapter's analytic $6ND$ FLOPs
  estimate against the compiler's own accounting.
- L670-674: explicit deferral list ending "...and the systems work of
  `:numref:chap_performance` — none of it a new body plan."

### chapter_transformers/gpt.md (ch. 11.2)
- L350-355: locates a training run on the FLOPs cost map ($6ND$), then: "Across
  those ten orders of magnitude the model definition barely changes — block, mask,
  embedding, head — but everything around it does: data pipelines, custom
  kernels, and the parallelism of `:numref:chap_performance`."

### chapter_transformers/transformer-block.md and encoders-decoders.md
- Use `nnx.jit(layernorm)` / `nnx.jit(rmsnorm)` and `nnx.jit(full)`/`nnx.jit(perceiver)`
  purely as a benchmarking harness to time architectural variants — JIT used as a
  measurement tool, not taught as a topic per se.

### chapter_optimization (ch. 9) — hardware/vectorization framing, and the deferral hub
- `chapter_optimization/minibatch-sgd.md` (`:label:sec_minibatch_sgd`), section
  "Vectorization and Caches" (L20-240): times elementwise vs. column-at-a-time vs.
  one-shot matrix multiply to teach *dispatch overhead* as the hardware reason
  batching helps (separate from the statistical noise-reduction reason). This is
  where the **`Timer` class is actually defined** (`#@save`, L105) — see §2.
  Explicitly separates "the *hardware* reason" (this section) from "the
  *statistical* reason" (`sec_sgd`), and defers gradient accumulation and
  data-parallel splitting to `:numref:chap_performance` (L278-280).
- `chapter_optimization/batch-size.md` L719-721: "turning the fewer, larger steps
  into less wall-clock time requires splitting each batch across many devices, and
  that is data parallelism, the machinery of `:numref:chap_performance`."
- `chapter_optimization/adamw.md` L559, L582-584: ZeRO-style optimizer-state
  sharding mentioned twice, citing `:cite:Rajbhandari.Rasley.Ruwase.ea.2020`
  (the ZeRO paper), pointing at **both** `:numref:chap_performance` *and*
  `:numref:sec_training_systems` (ch. 29, Tools appendix) together.
- `chapter_optimization/practice.md` L571-575: "Spreading gradients and state
  across a data-parallel group, ZeRO-style sharding, and the overlap of
  communication with computation belong to `:numref:chap_performance` and the
  training-systems material of `:numref:sec_training_systems`."

### chapter_builders-guide/gpus-devices-memory.md (ch. 5.7) — the big pre-existing foundation
See §2/§3 below — this file already teaches device placement, GPU memory
accounting, activation checkpointing (a store-vs-recompute tradeoff), a full
"Don't Break the Pipeline" section on **async dispatch** (PyTorch CUDA streams,
JAX's native async dispatch + `block_until_ready()`, TF `tf.function`/prefetch,
MXNet's engine), pinned memory / `non_blocking=True` transfers, and a
single-GPU-aware `Trainer`. Every one of these sub-topics ends with "The full
treatment of asynchrony, streams, and multi-device parallelism is in
`:numref:chap_performance`" (verbatim, 4x, once per framework tab).

### chapter_builders-guide/custom-layers.md (ch. 5.4)
- L1038-1041: "Writing new *kernels*, code that runs on the accelerator itself, is
  a separate craft that we do not cover in this book; `:numref:chap_performance`
  discusses how to get performance out of the operations you already have."
  (Sets reader expectations: comp-perf will NOT teach writing custom CUDA/Triton
  kernels — it discusses getting more out of existing ops.)

### chapter_appendix-tools-for-deep-learning/training-systems.md (ch. 29, `:label:sec_training_systems`) — important adjacent/non-duplicate territory
This is a **prose-only, display reference chapter** (per project convention,
ch. 29's python cells never execute — only `utils.md` is tracked/run). It already
covers, at a conceptual/reference level with no executed code:
- The "Scaling Ladder": one accelerator → DDP → FSDP/ZeRO stage 3 → tensor/
  pipeline/context/expert parallelism (`fig_tools_training_ladder`).
- `torchrun --standalone --nproc-per-node=4 train.py`, PyTorch's
  `DistributedDataParallel`, `fully_shard` (FSDP2/DTensor).
- JAX: "`jax.jit` with sharding specifications compiles the collectives for you,
  which is why the JAX code in this book has barely mentioned devices at all."
- The global-batch-size identity $B_\text{global} = B_\text{device} \cdot
  N_\text{ranks} \cdot N_\text{accumulation}$.
- Sections: "Where the Memory Goes," "The Library Landscape," "What to Use at
  Which Scale," "Keeping a Long Run Alive."
- **Implication**: this chapter is the *reference map of production libraries at
  scale* (DDP/FSDP/DeepSpeed names, torchrun, no runnable demos). Chapter 9's
  optimization sections already route "systems" promises through *both*
  `chap_performance` (hands-on, ch. 13) *and* `sec_training_systems` (reference,
  ch. 29) as a pair — the rebuilt ch. 13 should keep that division: ch. 13 teaches
  the mechanisms hands-on at notebook scale (streams, hybridization, one-node
  multi-GPU data parallelism, a toy parameter server); ch. 29 stays the
  practitioner's map of real-world libraries/scale. Don't let ch. 13 re-explain
  DDP/FSDP/ZeRO/torchrun prose — that's already ch. 29's job; ch. 13 should instead
  *earn* the concepts ch. 29 names (e.g. show why gradient all-reduce is needed by
  building a toy version, which ch. 29 does not do).

### chapter_appendix-tools-for-deep-learning/index.md — Resources list already curated
Cites (as background for `sec_hardware_buyers`, a *different*, ch.29-local
label — not to be confused with comp-perf's own `sec_hardware`):
"How to Scale Your Model" (Google DeepMind scaling book), "Making Deep Learning
Go Brrrr From First Principles" (Horace He — compute/bandwidth/overhead
taxonomy), "Which GPU for Deep Learning?" (Tim Dettmers). These are strong
candidate citations for the rebuilt comp-perf chapter's own Resources section,
already vetted once.

### Preliminaries / early device-placement teaching
- `chapter_preliminaries/ndarray.md` L878-880 (TF tab): "Compiling a computation
  with `tf.function` additionally lets TensorFlow prune and reuse allocations for
  you; we return to graph compilation and its performance benefits in
  `:numref:sec_hybridize`." — an explicit, direct forward-reference into
  comp-perf's very first section, from ch. 1.
- `chapter_linear-regression/synthetic-regression-data.md` and
  `chapter_linear-classification/image-classification-dataset.md`: both teach the
  idiom `drop_remainder=train` in the TF data pipeline specifically so that "a
  `@jax.jit`'d step function compiles once per epoch instead of recompiling for
  the smaller last batch" / "JAX does not retrace ... for a smaller last batch" —
  i.e., **shape-stability-for-JIT is already taught as a basic data-loading
  discipline in ch. 2/3**, well before any dedicated compilation chapter.
- `chapter_linear-classification/softmax-regression-scratch.md` L286-290: "Note
  that to make use of `jax.jit` to speed up JAX implementations... We refer
  interested readers to the JAX documentation on `jax.jit` and pure functions" —
  the book's very first explicit `jax.jit` mention, framed as a forward pointer to
  external docs, not a self-contained lesson.
- `chapter_builders-guide/model-construction.md` L1027-1049: teaches that
  data-dependent Python control flow is lost "once a model is wrapped in
  `jax.jit` for speed" — the tracing/staging mental model, taught early (ch. 5).

### Noise filtered out
Broad greps for `kernel`, `graph`, `fused`/`fusion`, `compile` turned up dozens of
false positives with unrelated meanings: "kernel" as in *convolution kernel*
(the whole CNN chapters), *reproducing kernel* (Gaussian processes, attention
scoring), Reinforcement Learning's Markov kernel; "graph" as in *computational
graph for autograd*, factor graphs, graphical models (probability chapters), or
literal `matplotlib` plots; "fused"/"fusion" mostly as *batch-norm fusion into
conv* (inference-time folding, taught in `convolutional-modern/batch-norm.md`
and `resnet.md`) and generic "combine/merge" prose. These are legitimate CNN/
inference-optimization teaching moments (BN-fold-into-conv is itself a real
compute-graph rewrite) but are pre-existing, chapter-local, and not directly
comp-perf's remit; flagged here only so the rebuild doesn't accidentally
re-claim BN folding as new territory — `convolutional-modern/resnet.md` /
`batch-norm.md` already own it.

---

## 2. Performance/timing idioms and d2l library helpers already in use

### `d2l.Timer` — defined in ch. 9, not in comp-perf
```
class Timer:  #@save         (in chapter_optimization/minibatch-sgd.md, :label:`sec_minibatch_sgd`)
```
`d2l/torch.py` L1198 and `d2l/jax.py` L1285 both carry the docstring "Defined in
`:numref:sec_minibatch_sgd`". Used pervasively outside comp-perf: `minibatch-sgd.md`
itself, `computer-vision/{kaggle-cifar10,image-augmentation,kaggle-dog,ssd}.md`,
`generative-adversarial-networks/{gan,dcgan}.md`,
`natural-language-processing-pretraining/{word2vec-pretraining,bert-pretraining}.md`,
`natural-language-processing-applications/natural-language-inference-attention.md`,
`recommender-systems/{autorec,mf,neumf}.md`, `appendix-tools-for-deep-learning/utils.md`.
**The comp-perf rebuild inherits `Timer` as a given — it should not redefine it.**

### `d2l.Benchmark` (context manager) — defined in comp-perf itself, genuinely unused elsewhere
`d2l/torch.py` L2022 / `d2l/jax.py` L2211: `class Benchmark`, docstring "Defined in
`:numref:sec_hybridize`". Grep for `d2l.Benchmark`/`Benchmark(` outside
`chapter_computational-performance/` returns **nothing** — this is comp-perf's own
tool, not yet reused. Note every other timing snippet in the book (ssm.md,
attention-at-scale.md, gpus-devices-memory.md) instead hand-rolls a local
`wall_clock`/`copy_time` helper with raw `time.time()` +
`torch.cuda.synchronize()`/`.block_until_ready()` rather than using
`d2l.Benchmark` — i.e. **there is no single canonical benchmarking idiom reused
book-wide**; each chapter reinvents a two-line timer. The rebuilt comp-perf
chapter could legitimately promote `d2l.Benchmark` (or a small successor) as
*the* pattern and note that earlier chapters used ad hoc versions of the same idea.

### `try_gpu` / `try_all_gpus` / `num_gpus` — defined in ch. 5 (builders-guide), not comp-perf
`d2l/torch.py` L635-654, `d2l/jax.py` L693-712, all docstring "Defined in
`:numref:sec_use_gpu`" → `chapter_builders-guide/gpus-devices-memory.md`
(`:label:sec_use_gpu`, ch. 5.7). Used everywhere from ch. 5 onward (ssm.md,
attention-at-scale.md's device placement, dozens of vision/NLP notebooks).
**Device-placement vocabulary is fully established by ch. 5; comp-perf ch. 13
inherits it and should not re-teach `try_gpu`.**

### `d2l.Trainer` GPU support — single-device only, explicitly deferred
`gpus-devices-memory.md`'s "The Trainer, Now with Devices" section (L1550-1810+)
adds `num_gpus` to `d2l.Trainer` for **one** device (`self.gpus[0]`), with the
`min(num_gpus, d2l.num_gpus())` ceiling pattern reused from ch. 6 onward. It is
explicit that this is not multi-device: every async/streams/multi-device
paragraph in that file ends "...is in `:numref:chap_performance`." So the
`Trainer` readers already know from ch. 5 does *not* do data-parallel multi-GPU;
comp-perf is where that gets added (or superseded).

### `train_batch_ch13` / `train_ch13` — real single-node multi-GPU training utility, but defined and used in ch. 19 (Image Models), NOT in comp-perf
`d2l/torch.py` L2925/L2945, `d2l/jax.py` L3133/L3149; docstring "Defined in
`:numref:sec_image_augmentation`" → `chapter_computer-vision/image-augmentation.md`
(ch. 19.1, old `#@tab` cell-tag style, not yet migrated to `%%tab`). This is a
genuinely-used multi-GPU training loop (PyTorch: `net.to(devices[0])` + manual
batch move, no `DataParallel` wrapper in the current pytorch tab shown; MXNet tab
uses `split_f=d2l.split_batch` sharding). It explicitly says "Recall the
introduction to multi-GPU training in `:numref:sec_multi_gpu_concise`" (comp-perf).
Reused by `computer-vision/{fcn,fine-tuning,kaggle-cifar10}.md`,
`natural-language-processing-pretraining/bert-pretraining.md` (see §5),
`natural-language-processing-applications/{sentiment-analysis-rnn,sentiment-
analysis-cnn,natural-language-inference-attention,natural-language-inference-
bert}.md`, `recommender-systems/{fm,deepfm}.md`. **This is a pre-existing,
book-wide multi-GPU training convention that predates and is independent of the
comp-perf rebuild — worth checking whether the rebuilt `sec_multi_gpu`/
`sec_multi_gpu_concise` should keep being its conceptual ancestor, since so much
downstream code already depends on `d2l.split_batch`/`d2l.resnet18` from those
exact comp-perf labels (`split_batch` "Defined in `:numref:sec_multi_gpu`",
`resnet18` "Defined in `:numref:sec_multi_gpu_concise`").**

### `d2l.split_batch`, `d2l.resnet18` — defined inside comp-perf, consumed downstream
`split_batch(X, y, devices)` docstring "Defined in `:numref:sec_multi_gpu`"; used
by `natural-language-processing-applications/natural-language-inference-
attention.md` L608 ("In contrast to the `split_batch` function in
`:numref:sec_multi_gpu` that takes single inputs..."). `resnet18(num_classes,
in_channels=1)` docstring "Defined in `:numref:sec_multi_gpu_concise`"; consumed
by `image-augmentation.md` (`d2l.resnet18(10)`, `d2l.resnet18(10, 3)`) and
`kaggle-cifar10.md` (`d2l.resnet18(num_classes, 3)`). **Any rebuild of
`multiple-gpus.md`/`multiple-gpus-concise.md` that renames or drops
`split_batch`/`resnet18` must update these downstream call sites or provide
compatible replacements — this is a real dependency, not just a doc promise.**

### Async dispatch — d2l.py has no dedicated helper; the pattern is taught inline
No `d2l.synchronize`/`d2l.block_until_ready` wrapper exists; `gpus-devices-memory.md`
and `ssm.md` each write raw `torch.cuda.synchronize()` / `.block_until_ready()` /
`tf.test.experimental.sync_devices()` / `npx.waitall()` inline, per framework.

---

## 3. Inbound `:numref:`/`:ref:` references into chapter_computational-performance labels (complete inventory)

Label stems that exist in comp-perf (via `grep -o ':label:\`[a-z_0-9-]*\`'
chapter_computational-performance/*.md`): `chap_performance`, `sec_hybridize`,
`sec_async`, `sec_auto_para`, `sec_hardware`, `sec_multi_gpu`,
`sec_multi_gpu_concise`, `sec_parameterserver`, plus `fig_*`/`table_*` labels
(none of which are ever referenced from outside the chapter).

**Result: essentially every inbound promise targets the whole-chapter label
`:numref:chap_performance`, not a specific subsection.** Exact-label hits for the
individual `sec_*` labels are rare:

| Label | Inbound refs outside comp-perf |
|---|---|
| `sec_hybridize` | 1 — `chapter_preliminaries/ndarray.md:880` (TF `tf.function` graph compilation) |
| `sec_async` | 0 |
| `sec_auto_para` | 0 |
| `sec_hardware` | 0 (note: `sec_hardware_buyers` is a *different*, ch. 29-local label — false-positive risk on substring grep) |
| `sec_multi_gpu` | 1 — `natural-language-processing-applications/natural-language-inference-attention.md:608` (re `split_batch`) |
| `sec_multi_gpu_concise` | 1 — `computer-vision/image-augmentation.md:699` ("Recall the introduction to multi-GPU training in...") |
| `sec_parameterserver` | 0 |
| **`chap_performance`** | **18 files, ~20 hits** (table below) |

### Full `chap_performance` inbound list (file : context)
1. `chapter_preface/index.md:296` — "Next, in `:numref:chap_performance`, we
   examine several key factors that influence the computational performance..."
   (top-level book roadmap sentence — stale/generic, pre-dates the SSM relocation
   and current chapter order; worth revisiting when index.md is rewritten).
2. `chapter_builders-guide/custom-layers.md:1040` — kernels/custom-op writing "is
   a separate craft that we do not cover in this book; `chap_performance`
   discusses how to get performance out of the operations you already have."
3. `chapter_builders-guide/saving-loading.md:967,975,984,991` (once per
   framework tab) — "returns to the machinery when models get that big" re
   sharded/lazy checkpoint loading at scale.
4. `chapter_builders-guide/gpus-devices-memory.md:1493,1505,1517,1547` (once per
   framework tab) — "The full treatment of asynchrony, streams, and multi-device
   parallelism is in `chap_performance`." (async dispatch / pinned-memory
   sections; see §1/§2.)
5. `chapter_optimization/minibatch-sgd.md:280` — gradient accumulation "a systems
   technique we return to in `chap_performance`."
6. `chapter_optimization/batch-size.md:721` — turning bigger batches into less
   wall-clock time "requires splitting each batch across many devices, and that
   is data parallelism, the machinery of `chap_performance`."
7. `chapter_optimization/adamw.md:584` — ZeRO-style sharding, "part of the
   systems story of `chap_performance` and `sec_training_systems`."
8. `chapter_optimization/practice.md:573` — data-parallel gradient/state
   spreading, ZeRO sharding, comm/compute overlap "belong to `chap_performance`
   and the training-systems material of `sec_training_systems`."
9. `chapter_recurrent-modern/index.md:140` — chapter-level scope fence: "the
   Triton kernels, memory hierarchies, and serving systems that make them fast
   belong to `chap_performance`." (chapter intro, see §1.)
10. `chapter_recurrent-modern/hybrids.md:1028` — capstone-section scope fence:
    fused kernels / serving stacks for the chunked recurrent forms "belongs to
    `chap_performance`."
11. `chapter_transformers/gpt.md:354` — "data pipelines, custom kernels, and the
    parallelism of `chap_performance`" (cost-map paragraph).
12. `chapter_transformers/kv-cache.md:1249` — "the systems-level story belongs to
    `chap_performance`" (re speculative decoding / serving).
13. `chapter_transformers/scaling-laws.md:672` — "...and the systems work of
    `chap_performance` — none of it a new body plan."

### Reading of this inventory
- **No chapter makes a promise about a *specific* comp-perf subsection except the
  two multi-GPU labels** (both concern `split_batch`/multi-GPU training,
  reflecting real code dependencies — see §2). Every other promise is diffuse:
  "systems/kernels/parallelism live in ch. 13," without committing to which
  subsection covers what. **This gives the rebuild real latitude to reorganize
  comp-perf's internal section structure** (rename/merge/reorder hybridize,
  async, auto-parallelism, hardware, multi-GPU, multi-GPU-concise, parameter
  server) as long as the chapter as a whole still delivers on: kernel fusion /
  custom kernels (custom-layers.md), async dispatch/streams/pinned memory
  (gpus-devices-memory.md, 4x), gradient accumulation and ZeRO-style sharding +
  comm/compute overlap (optimization chapter, 3x), data-parallel multi-GPU
  training (batch-size.md, image-augmentation.md's real `split_batch`/`resnet18`
  dependency), and "custom kernels"/Triton/serving-stack territory now explicitly
  claimed by the SSM chapter's scope fences (recurrent-modern/index.md,
  hybrids.md) and by gpt.md/kv-cache.md/scaling-laws.md.
- **Two must-preserve API contracts**: `d2l.split_batch` (from `sec_multi_gpu`)
  and `d2l.resnet18` (from `sec_multi_gpu_concise`) are live dependencies of
  `natural-language-processing-applications/natural-language-inference-
  attention.md` and `computer-vision/{image-augmentation,kaggle-cifar10}.md`
  respectively — renaming/removing them breaks those notebooks unless updated in
  lockstep.
- `chapter_preface/index.md:296` is the one place that narrates the book's
  chapter *sequence*, and it still describes comp-perf's old position/content in
  generic terms; flagged as stale prose (not asked to fix, but relevant if the
  preface rewrite mentioned in CLAUDE.md happens alongside this rebuild).

---

## 4. Chapter ordering (ground truth: `tools/d2l_preprocess.py` `CHAPTER_NUMBERING`, matches `_quarto.yml`)

**Advanced part = ch. 9-16, in this exact order:**

| # | Chapter | Directory |
|---|---|---|
| 9 | Optimization Algorithms | `chapter_optimization/` |
| 10 | Attention | `chapter_attention/` |
| 11 | Transformers | `chapter_transformers/` |
| 12 | Linear Recurrence and State Space Models | `chapter_recurrent-modern/` |
| **13** | **Computational Performance** | **`chapter_computational-performance/`** |
| 14 | Reinforcement Learning | `chapter_reinforcement-learning/` |
| 15 | Generative Adversarial Networks | `chapter_generative-adversarial-networks/` |
| 16 | Diffusion Models | `chapter_diffusion-models/` |

**Note on CLAUDE.md drift:** the "Current status" section of CLAUDE.md still
describes an older layout ("11 Computational Performance, 12 Transformers
(placeholder), 13 State Space Models"). That is now **stale** — `transformers` is
fully built out (index + 7 sections: transformer-block, gpt, kv-cache,
encoders-decoders, vision-transformer, moe, scaling-laws) and sits at ch. 11,
recurrent-modern/SSM is ch. 12, and comp-perf is ch. 13, confirmed by both
`_quarto.yml` chapter order and `CHAPTER_NUMBERING`. This matches the MEMORY.md
note "[ch12 SSM rebuild underway] — relocation DONE (ch12, after
attention/transformers)."

**Legitimate prerequisites available to comp-perf ch. 13 demos** (everything
before it in reading order): the full Basics part (ch. 1-8, incl. ch. 5
builders-guide's device placement/GPU-memory/async-dispatch/single-GPU Trainer),
**Optimization** (ch. 9: Timer, minibatch/vectorization/dispatch-overhead
framing, gradient accumulation, critical batch size, ZeRO mention, muP/scaling),
**Attention** (ch. 10: multi-head attention, FlashAttention/fused-kernel
benchmarking, linear attention), **Transformers** (ch. 11: full GPT-style model,
KV-cache, MoE, scaling laws with `jax.jit(...).lower(...).compile().cost_analysis()`),
and **State Space Models** (ch. 12: parallel scan, `jax.lax.associative_scan`,
Mamba's hardware-aware kernel discussion, chunked matrix-state forms). This is a
much richer, more modern demo surface than the original d2l book had at this
chapter's old position (AlexNet/ResNet were the only prior architecture); the
rebuild can legitimately time/compile/parallelize a transformer block, an
attention layer, or an SSM scan instead of (or alongside) classic CNN examples.
Transformers exist as a fully-realized chapter (not a placeholder) by the time
comp-perf is reached.

---

## 5. Language Models part (ch. 17-18) and distributed/large-scale training

`chapter_natural-language-processing-pretraining/` (ch. 17, incl.
`bert-pretraining.md`) and `chapter_natural-language-processing-applications/`
(ch. 18) contain **no mention** of multi-machine/cluster/distributed training
(`distributed`, `multi-machine`, `cluster`, `TPU pod`, `multi-node` all grep to
zero hits, aside from one unrelated "embeddings cluster semantically" in
word2vec-pretraining.md). What they *do* have is single-node, few-GPU training
using the pre-existing `train_ch13`/`train_batch_ch13`/`try_all_gpus` machinery
from ch. 19 (§2 above): `bert-pretraining.md` trains a small BERT on WikiText-2
with `nn.DataParallel(net, device_ids=devices)` in PyTorch, an explicit
JAX micro-batching loop (`num_micro=2`), and a TF-specific comment about working
around the default BFC allocator's memory growth
(`os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'`). The chapter's own
Summary section stays purely about model capability ("BERT has two versions...
representations are context-sensitive") — it never discusses infrastructure or
scale-up, consistent with the framing that real multi-machine BERT/LLM
pretraining infrastructure is out of scope here and deferred (conceptually, to
`chap_performance` for mechanisms and to ch. 29's `sec_training_systems` for the
production-library map — see §1).

---

## 6. House style checklist for the new chapter (verified from `chapter_optimization/index.md` and `chapter_recurrent-modern/ssm.md`)

1. **Chapter index Resources section.** Every part-level chapter `index.md`
   carries `## Resources and Further Reading {.unnumbered}` (confirmed in
   optimization, attention, transformers, recurrent-modern index files — a
   book-wide convention, not ch.9-specific). Organized into sub-bins with bold
   `**bin name**` headers: `**Books**`, `**Courses and video lectures**`,
   `**Foundational and current papers**`, `**Tutorials, notes, and interactive**`.
   Example: `- [Convex Optimization — Boyd & Vandenberghe](url) — free PDF; the
   standard reference behind the vocabulary this chapter uses informally —
   conditioning, convergence rates, duality, projections — and the right place to
   see the analyses that :numref:`sec_gd` and :numref:`sec_sgd` state without
   proof done properly.` Every entry: link, one-line access note (free/paywalled),
   then a sentence tying it explicitly back to a specific in-chapter `:numref:`.
2. **`%%tab pytorch` / `%%tab jax` per-cell tabs** (SSM chapter is PyTorch+JAX
   only, per Advanced-part policy). Example:
   ```
   ```{.python .input #ssm-implementation-1}
   %%tab pytorch
   def associative_scan(a, b, dim=0):  #@save
   ```
   ```{.python .input #ssm-implementation-1}
   %%tab jax
   def associative_scan(a, b):  #@save
   ```
   ```
   Note both cells of a pair **share the same `#<id>`** (the id names the
   *teaching point*, not the framework variant).
3. **Stable `#<id>` per code fence**, kebab-case, `<section-slug>-<n>` pattern,
   assigned once and never renumbered on edit (via `tools/add_cell_ids.py`).
   Example: `#ssm-implementation-1`, `#ssm-implementation-2`,
   `#ssm-a-mingru-language-model-3`.
4. **`:begin_tab:`/`:end_tab:`** prose blocks precede code when the explanation
   differs meaningfully by framework (e.g. JAX's native async dispatch vs.
   PyTorch's CUDA-stream model) — used heavily in ssm.md and
   gpus-devices-memory.md.
5. **`#@save`** marks cells whose definitions enter the `d2l` library (rebuilt
   into `d2l/*.py` by `make lib`); prose right after explicitly says so
   ("We save the helper in the `d2l` library: it is the computational core of
   this section and the next.").
6. **Figures are pre-generated SVGs, never drawn inline.** Example:
   `![A parallel prefix scan on eight elements...](../img/mdl-modernrnn-scan-tree.svg)`
   immediately followed by `:label:`fig_scan_tree``. No matplotlib figure-drawing
   code appears in the notebook cells.
7. **Display equations**: `$$` alone on its own line, content, `$$` alone on its
   own line, `:eqlabel:`eq_name`` on the line right after — e.g. the associative
   combine operator at ssm.md L165-169.
8. **Section labels**: every `##` section that is `:numref:`-targeted gets its
   own `:label:` line immediately under the heading (e.g.
   `## Parallel Scans` / `:label:`subsec_parallel-scans``); the whole chapter
   file gets one top `:label:`sec_xxx`` right under the `#` title.
9. **Exercises**: numbered list (`1.` repeated — renders as auto-numbered), each
   item a short **bolded/italicized name** for the exercise followed by the task.
   The SSM chapter additionally prefixes each with a difficulty/scope tag —
   `[short-code]` or `[extended]` — e.g. `1. [short-code] *Effective memory.*
   For a scalar recurrence...`. **Caveat: this tag convention is local to
   `chapter_recurrent-modern/` (7/6/5/4/3/6/4 tagged exercises across its files)
   and does not yet appear in ch. 9/10/11 — treat as optional/new, not an
   established book-wide requirement, when deciding whether the comp-perf
   rebuild should adopt it.**
10. **`[Discussions](https://d2l.discourse.group/)`** line right before the
    `<!-- slides -->` marker, at the very end of the body content.
11. **Slide section**: starts with a literal `<!-- slides -->` HTML comment,
    then one `::: {.slide title="..."} ... :::` div per slide. First slide is
    conventionally untitled-cover-style or a chapter-title recap; content uses
    plain Markdown/LaTeX, `**bold**` for emphasis, and `. . .` on its own line
    for a Reveal.js incremental-reveal pause.
12. **Slides reference notebook cells by `@<id>`** (bare, on its own line, no
    other markup) rather than re-embedding code — e.g. `@ssm-implementation-1`
    pulls in the exact cell already shown in the body. A slide can stack several
    `@<id>` refs with `. . .` between them to reveal outputs progressively.
13. **One imports cell per framework**, high up, no re-imports later (book-wide
    rule, not SSM-specific, but honored throughout).
14. **`:numref:`/`:ref:` never hardcoded** — every cross-chapter or
    cross-section pointer uses a label, confirmed throughout (e.g. ssm.md's
    "gate derived three times" motif references `sec_lstm`, `subsec_bptt-
    gradient-pathologies`, `chap_cnn` all by label).
15. **Kicker line on the slide cover**: `[Dive into Deep Learning · §N.M]{.kicker}`
    — auto-rewritten by `gen_slides.py` from `CHAPTER_NUMBERING`, never
    hand-maintained.

---

## 7. Structure of `reviews/ssm-chapter-review-and-proposal-2026-07-19.md` (template for the comp-perf proposal)

Top-level outline (8 numbered `##` sections):
1. **Verdict and the chapter at a glance** — one paragraph judgment + summary table.
2. **What the relocation changed** — subsections 2a (inbound obligations the
   chapter must now honor, "all grep-verified"), 2b (wrong-way references that
   must flip), 2c (standing content gaps, relocation-independent).
3. **What is preserved untouched** (the spine) — explicitly scopes what does
   *not* get rewritten.
4. **The structural call** — the one big organizational decision (here: moving
   `seq2seq.md` out to another part), argued at length as its own section.
5. **The new arc, section by section** — one `###` subsection per notebook file
   (`12.1 lstm.md — revise`, `12.2 ssm.md — revise`, ... `12.7 hybrids.md — NEW`),
   each a **numbered action list**: bold lead-in per item ("**Add `## Inference,
   One Token at a Time`** after ..."), concrete implementation detail (what to
   `#@save`, what figure/benchmark to add, which prose sentence to flip and why),
   explicit tie-backs to gap codes from §2 (e.g. "fixes G1", "fixes G4"), and
   forward/backward `:numref:` citations that will result. Ends with a full
   `index.md` rewrite entry.
6. **Cross-cutting specifications** — one bullet-block per concern that spans all
   sections: **Figures** (which new SVGs, house style, generator tool), **`#@save`
   contracts** (exact new library API surface), **Runtime budget** (CPU vs. GPU,
   minutes-scale), **Citations to add to `d2l.bib`** (verified arXiv IDs, one
   line each), **Resources and Further Reading** entries (click-verified), and a
   **Scope fences** bullet explicitly listing what the chapter will NOT cover and
   which other chapter owns that territory (this is where SSM's proposal states
   "kernels/Triton/serving (ch. 13)" belongs to comp-perf).
7. **Build/ops plan** — the mechanical execution order (per-file recapture,
   which moves require `CHAPTER_NUMBERING`/`_quarto.yml`/outputs-tree changes).
8. **Open questions for Alex — RESOLVED** — a running list of decisions that
   needed a human call, each marked resolved with the answer.

For the comp-perf proposal, the same template implies: a verdict/gap-analysis
opening grounded in *this* scan's inbound-reference inventory (§3) and the
already-covered-elsewhere inventory (§1/§2), a structural call (how to
reorganize hybridize/async/auto-parallelism/hardware/multi-gpu/multi-gpu-
concise/parameter-server given the new prerequisite chapters), a per-notebook
section-by-section plan in the same numbered-action-item format, and a
cross-cutting spec block whose "Scope fences" bullet must reconcile against the
SSM chapter's own scope fence (which already claims "kernels/Triton/serving" as
ch. 13's territory) and against ch. 29's `sec_training_systems` (which already
owns the DDP/FSDP/ZeRO/torchrun production-library map).
