# Review — §6 "Builders' Guide": the systems sections
### `read-write.md` (File I/O / saving & loading) + `use-gpu.md` (GPUs / device management)

*Landscape review (not a line-by-line change spec). Primary lens: PyTorch 2.x;
the book ships four framework tabs (pytorch primary). Scope guard: §6 is the
practical "how to build and use models" guide — flag the modern essentials and
forward-point the deep dives; do **not** turn this into a distributed-training
chapter.*

---

## 0. Headline verdict

These two sections are, as the author suspected, **the most dated in all of §6**,
and by a wide margin. They teach the 2021 mechanics correctly but they teach a
2021 *worldview*: serialization is `torch.save`/pickle and nothing else; a "device"
is a CUDA card; and the entire modern performance/safety/portability layer of
PyTorch 2.x — `weights_only`, **safetensors**, full **checkpointing**,
**device-agnostic code**, Apple **MPS**, `torch.accelerator`, **mixed precision /
AMP**, **`torch.compile`**, GPU-memory reasoning, and the **DataParallel→DDP/FSDP**
shift — is **absent from the body of both pages**.

The one nuance that makes this *worse than it looks*, not better: a 2024-era
modernization pass clearly happened, **but it only touched the `<!-- slides -->`
sections.** The slides for `read-write.md` already name `weights_only=True`,
safetensors, the full-checkpoint dict, and `register_buffer`; the `use-gpu.md`
slides already reframe the device discipline and warn about the inner-loop copy.
**None of that reached the prose or the executable notebook cells** — the HTML book
page and the runnable notebook a student actually reads are still the untouched
2021 content (I confirmed this against the executed
`_notebooks/pytorch/chapter_builders-guide/read-write.ipynb` and `use-gpu.ipynb`).
So the deck is ~70% modernized and the chapter is ~0% modernized, and they have
silently diverged. **Reconciling that gap is the single highest-leverage move for
this chapter**: most of the missing content already exists in draft form one
scroll down in the same file.

**Grades.**
- `read-write.md` — **C+.** Correct, clean, but teaches one third of the modern
  story. The pickle-security framing and the full-checkpoint pattern are the
  load-bearing omissions; safetensors is the load-bearing *addition*.
- `use-gpu.md` — **C / C-.** Correct on the core "tensors carry a device" lesson,
  but `cuda`-hardcoded end to end (including the committed `d2l` library helpers),
  contains a **factually dated claim** (the "global interpreter lock stalls all
  GPUs" framing), spends a disproportionate fraction of its length on a two-GPU
  data-shuffling demo that "most desktops" can't run, and omits *every* modern
  single-GPU performance topic. This page needs the most reworking in §6.

Both are assignable today only with an instructor's verbal "...but here's how we
actually do this now." The bar (Stanford/MIT/CMU/Berkeley) is that the page itself
*is* how we do it now.

---

## 1. Current-state assessment, section by section

### 1.1 `read-write.md` — "File I/O"

**What it teaches.** Three escalating moves, in all four frameworks:
1. Save/load a single tensor (`torch.save(x, 'x-file')` / `torch.load(...,
   weights_only=True)`).
2. Save/load a list and a `dict` of tensors.
3. Save/load **model parameters** via `state_dict()` / `load_state_dict()`, with
   the correct and well-made pedagogical point that **you save the *parameters*,
   not the model class** — the architecture is code, the file is weights — and a
   sanity check that `clone(X) == net(X)` after a round-trip.

**What's genuinely good and should survive any rewrite.**
- The state_dict-vs-class distinction is exactly right and is *still* the first
  thing the official PyTorch tutorial teaches
  (https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html — *"save
  the state_dict, not the whole model; pickling the module ties the file to the
  class definition and breaks on refactor"*). Keep this framing; it has aged well.
- The "save a dict of tensors" cell is the perfect on-ramp to both a `state_dict`
  *and* to safetensors (whose entire data model is "a dict of named tensors") — the
  chapter already has the right building block, it just stops one step early.
- Someone already added `weights_only=True` to every `torch.load` call. Correct,
  and forward-looking.

**What is dated / risky / thin (body, not slides).**
- **The pickle security story is told only in the deck.** The body never explains
  *why* `weights_only=True` is there. This is the single most important systems
  concept the section is missing: `torch.load` runs a pickle unpickler, and a
  malicious checkpoint can execute arbitrary code on load — a real, documented
  supply-chain attack (see §3 for the version facts and a CVE-grade citation).
  This is exactly the kind of "power user" knowledge §6 promises and it's
  one paragraph.
- **No `safetensors`.** The body has zero mention of the format that has become the
  de-facto standard for distributing weights (HF Hub default; used by
  `transformers`, `diffusers`, MLX, ComfyUI, timm, …). It appears only as a slide
  bullet. For a 2026 textbook this is the headline gap of the section.
- **No full training checkpoint.** The chapter's *own opening paragraph* motivates
  the section with checkpointing ("if we trip over the power cord… save
  intermediate results") — and then **never shows a checkpoint**. It saves weights
  only. The modern, resumable artifact is `{'model': …, 'optimizer':
  opt.state_dict(), 'epoch': …, 'rng_state': …}`; the section sets up the need and
  doesn't pay it off. (Again: the deck has it; the body doesn't.)
- **TensorFlow tab uses the legacy H5 path** (`save_weights('mlp.weights.h5')`)
  with an inline "prefer .keras in Keras v3" comment — i.e. the section knows it's
  dated and apologizes in a comment instead of fixing it.
- **JAX tab uses `flax.training.checkpoints`**, which is deprecated in favor of
  **Orbax** — again flagged by an inline comment rather than updated.
- **Weak exercises.** The three exercises are fine but pre-modern; none touches
  safe serialization, checkpoint/resume, or partial loading (the `strict=False`
  head-swap the slides mention).

**Net:** the section is a correct skeleton of the 2021 story with a 2024 deck
bolted on top. The body needs the security paragraph, safetensors, and a real
checkpoint to reach the bar.

### 1.2 `use-gpu.md` — "GPUs"

**What it teaches.** Device handles (`cpu()`, `gpu(i)`, `num_gpus()`, the
`try_gpu`/`try_all_gpus` helpers — all `#@save`d into `d2l`); tensors carry a
device (`.device`); creating tensors on a device; the cross-device-copy rule (you
must move tensors to a common device before an op, with the `copyto`/`fig_copyto`
illustration); a "side notes" passage on transfer cost; putting a model on the GPU
(`net.to(device)`); and patching the book's `Trainer` to move batches/params.

**What's good and should survive.**
- The central lesson — *every tensor and every parameter lives on a device;
  cross-device ops are an error; copies are expensive so keep the inner loop on one
  device* — is correct, important, and well-illustrated. This is the spine of the
  page and it's right.
- `try_gpu(i)` as a portability shim is a good instinct (it just needs to learn
  about non-CUDA accelerators — see below).
- The "many small transfers are worse than one big one" rule of thumb is sound.

**What is dated / risky / wrong.**
- **`cuda`-hardcoded end to end — including the committed library.** Every device
  path is `torch.device('cuda')` / `.cuda(i)`. The `d2l` helpers
  (`d2l/torch.py:574` `gpu()` → `torch.device(f'cuda:{i}')`,
  `num_gpus()` → `torch.cuda.device_count()`) bake CUDA into the *book's own
  library*, so the examples cannot run on the very machine this repo is built on (an
  Apple-silicon laptop, where the accelerator is `'mps'`, not `'cuda'`). This is the
  defining datedness of the section. The modern idiom is device-agnostic (see §2.2).
- **A factually dated claim.** The Summary and "Side Notes" assert that logging via
  `.numpy()` "will trigger a global interpreter lock which stalls all GPUs." This is
  imprecise-to-wrong by 2026 framing. The real cost is a **CPU↔GPU
  synchronization**: `.item()`/`.numpy()`/`print` forces the host to wait for the
  asynchronous CUDA stream to drain (a sync point), breaking compute/transfer
  overlap. It is a stream-stall, not fundamentally a GIL phenomenon (the PyTorch
  CUDA-semantics doc frames it as async execution + forced sync, never the GIL).
  This sentence should be corrected, not just modernized.
- **Opens by demanding two GPUs.** "To run the programs in this section, you need
  at least two GPUs… extravagant for most desktop computers." A large fraction of
  the page (the `X.cuda(1)` copy demo, the second-GPU `Y`) then teaches multi-GPU
  *data movement* — a niche most readers can't execute and that logically belongs
  with the actual multi-GPU chapter. The single-GPU essentials a 2026 reader needs
  (AMP, `compile`, memory) are absent, while two-GPU `copyto` choreography is
  over-weighted.
- **No mixed precision, no `torch.compile`, no GPU-memory reasoning.** The three
  things that actually make a single GPU fast in 2026 — `autocast`+`GradScaler`,
  `torch.compile`, and understanding allocated-vs-reserved memory / OOM — are
  entirely missing from a chapter whose stated promise (`index.md`) is "leveraging
  GPUs to achieve dramatic speedups." The chapter never delivers the speedups.
- **DataParallel-era multi-GPU mental model.** The TF tab uses
  `tf.distribute.MirroredStrategy`; the prose frames multi-GPU as "copy tensors
  between cards." There's no pointer to the modern reality (DDP = one process per
  GPU; FSDP for models that don't fit) — fine to keep pointer-level, but the pointer
  should exist and be *correct*.
- **JAX/TF "create on accelerator by default" claim is actually correct** (I
  verified: JAX places on the first accelerator; TF uses soft placement onto GPU) —
  so keep it, it's one of the few framework contrasts that's still accurate.

**Net:** correct core lesson, dated everywhere else, one wrong sentence, and
mis-weighted toward a two-GPU demo at the expense of the single-GPU performance
toolkit that defines the modern version of this exact topic.

---

## 2. Modernization gaps (what's there but dated) — with verified facts

These are corrections/updates to content the sections *already* cover. All version
facts below are verified against current (mid-2026) primary sources; PyTorch stable
is **2.12**, the repo pins torch **2.11.0** (which on darwin/arm64 ships MPS).

### 2.1 Serialization: pickle risk + `weights_only` (read-write)
- `torch.load` uses a pickle unpickler; a crafted checkpoint can execute arbitrary
  code **on load**. Not theoretical — JFrog documented ~100 weaponized models on the
  HF Hub with live backdoor payloads.
- **`weights_only=True` became the *default* in PyTorch 2.6** (released 2025-01-29;
  deprecation warnings since 2.4). The chapter's slide says "default since 2024" —
  **imprecise**: the *warning* started in 2.4 (2024), the *default flip* shipped in
  2.6 (Jan 2025). The body should state the version correctly. With
  `weights_only=True` the unpickler is restricted to tensors/primitives/dicts;
  trusted-source escape hatches are `weights_only=False` or
  `torch.serialization.add_safe_globals([...])`.

### 2.2 Devices: from `'cuda'` to device-agnostic (use-gpu)
- The 2021 idiom `device = torch.device('cuda' if torch.cuda.is_available() else
  'cpu')` is superseded. **`torch.accelerator` (new in PyTorch 2.6)** abstracts
  CUDA/MPS/XPU/etc. The **official PyTorch quickstart tutorial now leads with**:
  ```python
  device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
  ```
  This is the single cleanest "how it's taught now" citation for the section — the
  *official intro tutorial itself* dropped the `'cuda'` idiom. The `d2l`
  `gpu()`/`try_gpu()` helpers should learn about MPS/accelerator (decide whether to
  update the library or keep `try_gpu` and add an accelerator-aware variant).
- `torch.accelerator` also now exposes **device-agnostic memory introspection**
  (`memory_allocated`, `max_memory_allocated`, `empty_cache`), so the memory
  discussion (§3) can be written once for all backends.

### 2.3 The "GIL stalls GPUs" sentence (use-gpu) — correct it
Reframe as: `.item()`/`.numpy()`/`print` forces a **device→host synchronization
that drains the asynchronous CUDA stream**, killing compute/transfer overlap. Then
the *fix* is teachable and modern: pinned memory (`DataLoader(pin_memory=True)`) +
`tensor.to(device, non_blocking=True)`, and `torch.cuda.synchronize()` for correct
benchmarking. (Source: PyTorch "CUDA semantics" notes.)

### 2.4 Non-PyTorch tabs already self-flag as dated
- TF: replace H5 `save_weights('.h5')` with the Keras-v3 `.keras` path.
- JAX: `flax.training.checkpoints` → **Orbax** (`orbax.checkpoint`).
- These are *defer-to-a-later-framework-pass* items per the repo's PyTorch-first
  workflow, but they should be named so the multi-framework batch picks them up.

---

## 3. Missing topics (what a top-5 treatment covers and this chapter omits)

Each is verified and tagged with a recommended **home** (this chapter vs a
forward-pointer to a later one). The principle, confirmed by every strong resource
surveyed (§4): the topics this chapter omits are *exactly* the ones the modern
canon treats as core single-machine essentials. A pointer-level-but-present
treatment matches the bar; the deep dives forward-point.

| # | Topic | Verdict: where it belongs | One-line justification |
|---|---|---|---|
| M1 | **safetensors** (safe, zero-copy, framework-agnostic; `safetensors.torch.save_file`/`load_file`) | **IN `read-write.md`** (core) | The modern serialization standard; the section's "dict of tensors" cell is already the perfect setup. |
| M2 | **Pickle security / `weights_only` rationale** | **IN `read-write.md`** (core, 1 para) | Explains the `weights_only=True` already in the code; classic "power user" knowledge §6 promises. |
| M3 | **Full training checkpoint** (model+optimizer+epoch+RNG; resume) | **IN `read-write.md`** (core) | The section's own opening motivates it and never delivers it. |
| M4 | **Device-agnostic code** (`'cuda'/'mps'/'cpu'`, `torch.accelerator`) | **IN `use-gpu.md`** (core) | The official tutorial's current idiom; makes examples run on this repo's own Mac host. |
| M5 | **Apple MPS** (`'mps'`, `torch.backends.mps.is_available()`, fallback env var) | **IN `use-gpu.md`** (core, brief) | The build host is Apple-silicon; MPS is the relevant accelerator there, not CUDA. |
| M6 | **GPU memory** (allocated vs reserved, caching allocator, OOM, `empty_cache`) | **IN `use-gpu.md`** (core, brief) | OOM is the practical ceiling on batch/model size; the page promises GPU mastery. |
| M7 | **Async execution + transfer cost** (sync points, `pin_memory`, `non_blocking`) | **IN `use-gpu.md`** (replaces the GIL paragraph) | Corrects a wrong claim *and* upgrades it into a real, teachable optimization. |
| M8 | **Mixed precision / AMP** (`torch.autocast`, `GradScaler`, fp16 vs bf16) | **IN `use-gpu.md` (pointer-level, ~½ page) + forward-point** | The expected "make the GPU fast" tool in 2026; full treatment can live with optimization/training. |
| M9 | **`torch.compile`** (one-line wrap, ~1.3–2× speedup, warmup caveat) | **IN `use-gpu.md` (pointer-level, short) + forward-point** | The flagship PyTorch 2.0 feature; the chapter literally promises "dramatic speedups." |
| M10 | **Multi-GPU landscape** (DataParallel = legacy → **DDP**; **FSDP** for huge models) | **POINTER ONLY** here; deep dive → multi-GPU/distributed chapter | Keep in lane: correct one-paragraph map + URLs, not a distributed chapter. |
| M11 | **`torch.compile`/DDP `state_dict` key prefixes** (`_orig_mod.`, `module.`) | **Optional aside in `read-write.md`** | Real gotcha when saving compiled/wrapped models; one sentence + the `consume_prefix_in_state_dict_if_present` helper. |

**Verified facts behind the new topics** (all confirmed against primary sources):

- **safetensors** — *"a simple format for storing tensors safely (as opposed to
  pickle) and that is still fast (zero-copy)."* Save/load a state_dict with
  `from safetensors.torch import save_file, load_file`. **Limitation to state
  clearly:** tensors only — it **cannot** hold the arbitrary-Python-object training
  checkpoint (optimizer bookkeeping, epoch ints, schedulers). So the honest framing
  is two distinct artifacts: **weights → safetensors**; **resumable checkpoint →
  `torch.save` dict** (with the `weights_only` caveats). Trail-of-Bits audited
  safetensors (May 2023, no RCE). HF `transformers` defaults to it
  (`save_pretrained(..., safe_serialization=True)`) since v4.35 (Nov 2023).
- **Full checkpoint** — canonical pattern from the official tutorial:
  `torch.save({'epoch':…, 'model_state_dict':net.state_dict(),
  'optimizer_state_dict':opt.state_dict(), 'loss':loss}, PATH)`; load by
  constructing model+optimizer first, then `load_state_dict` each; call
  `model.eval()`/`.train()` appropriately. Convention: `.pt`/`.pth` for weights,
  `.tar` for multi-component checkpoints (≈2–3× larger).
- **MPS** — device string `'mps'`; check `torch.backends.mps.is_available()`; in
  PyTorch since 1.12 (Apple still labels it *beta*, and op coverage lags CUDA);
  caveats: incomplete op coverage (`PYTORCH_ENABLE_MPS_FALLBACK=1`), no float64, and
  a macOS floor that has risen by PyTorch version (historically 12.3; recent
  2.11/2.12 wheels state macOS 14.0+). `torch.accelerator` returns `mps` on Apple
  silicon, so the device-agnostic idiom "just works" on a Mac.
- **AMP** — modern device-agnostic API is `torch.autocast(device_type='cuda',
  dtype=…)` + `torch.amp.GradScaler('cuda')`; **`torch.cuda.amp.*` is deprecated
  (since 2.4)**. fp16 needs a `GradScaler` (gradient underflow); **bf16 does not**
  (same fp32-width exponent). ~1.5–3× throughput and ~½ activation memory on
  tensor-core GPUs. Canonical loop: `scaler.scale(loss).backward();
  scaler.step(opt); scaler.update()`.
- **`torch.compile`** — `model = torch.compile(model)`; TorchDynamo capture +
  TorchInductor (Triton) backend; **shipped in PyTorch 2.0 (March 2023)** and is now
  the framework's flagship, recommended optimization path (note: PyTorch tracks
  `torch.compile` maturity *per-capability* — there is no single documented Beta→GA
  version, so phrase it as "mature/recommended," not "GA in version X"). Official
  2.0 benchmark: works on 93% of 163 models, ~43% faster training on A100 (≈+21%
  fp32 / +51% with AMP); `mode=` ∈ {default, reduce-overhead, max-autotune}; first
  call is slow (trace + compile); shape changes can recompile.
- **DDP vs DataParallel** — official CUDA notes: *"It is recommended to use
  DistributedDataParallel, instead of DataParallel… even if there is only a single
  node."* DataParallel is single-process/multi-thread → GIL-bound + replicates each
  iteration; **DDP** is one process per GPU + gradient all-reduce; **FSDP/FSDP2**
  shards params+grads+optimizer state for models too big for one GPU.

---

## 4. How this is taught now — surveyed resources (verified, with URLs)

Every source below was checked live (2024–2026). Grouped by role. The throughline:
**the modern canon treats AMP, `torch.compile`, full checkpointing, safetensors,
device-agnostic/MPS, GPU memory, and DDP/FSDP as core** — precisely the set this
chapter omits.

**Official PyTorch (the primary "how to" authority — cite these in the body)**
- *Saving and Loading Models* — https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
  — canonical; state_dict-vs-module, the full-checkpoint dict, `.pt`/`.tar`
  conventions, `module.`-prefix handling. The recipe on general checkpoints now
  redirects into this consolidated page.
- *`torch.load` API ref* — https://docs.pytorch.org/docs/stable/generated/torch.load.html
  — authoritative for the `weights_only=True` default + security warning.
- *Learn the Basics → Quickstart* — https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
  — the *official* device-agnostic idiom (`torch.accelerator.current_accelerator()`);
  best single citation that `'cuda'`-hardcoding is now behind the times.
- *`torch.accelerator`* — https://docs.pytorch.org/docs/stable/accelerator.html
  — the device-agnostic runtime API (2.6+), incl. unified memory stats.
- *MPS backend notes* — https://docs.pytorch.org/docs/stable/notes/mps.html and Apple's
  *Accelerated PyTorch on Mac* — https://developer.apple.com/metal/pytorch/ — the
  `'mps'` story; Apple's page even lists "Stable (PyTorch 2.11.0)", the repo's pin.
- *CUDA semantics* — https://docs.pytorch.org/docs/stable/notes/cuda.html — async
  execution, sync points, the caching allocator (allocated vs reserved), pinned
  memory + `non_blocking`; the authority for correcting the "GIL" sentence.
- *AMP* — https://docs.pytorch.org/docs/stable/amp.html + recipe
  https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html — autocast,
  GradScaler, fp16-vs-bf16, the `torch.cuda.amp.*` deprecation.
- *`torch.compile`* — https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
  + https://pytorch.org/get-started/pytorch-2.0/ — what it is, modes, official speedups.
- *DDP* — https://docs.pytorch.org/docs/stable/notes/ddp.html (+ the "use DDP not
  DataParallel" line in the CUDA notes); *FSDP2* —
  https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html — the
  pointer-level multi-GPU map.

**safetensors / HuggingFace (the safe-serialization standard)**
- safetensors — https://github.com/huggingface/safetensors and docs
  https://huggingface.co/docs/safetensors/index — format, API, zero-copy/lazy
  loading; *the* reference for M1.
- *Safetensors audited as really safe* (HF, May 2023) —
  https://huggingface.co/blog/safetensors-security-audit — the citable proof
  (Trail of Bits audit) and the clearest statement of the pickle threat.
- JFrog, *malicious HF models with silent backdoors* —
  https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/
  — "this actually happened in the wild" for the M2 paragraph.
- HF *efficient training on a single GPU* —
  https://huggingface.co/docs/transformers/en/perf_train_gpu_one — the single best
  practical page mapping ~1:1 onto M6/M8/M9 (mixed precision, gradient
  checkpointing, `torch.compile`, memory tradeoffs).

**Top university courses / programs (establish the bar)**
- **CMU 10-414/10-714 "Deep Learning Systems"** (Tianqi Chen, with Zico Kolter /
  Tim Dettmers across offerings) — https://dlsyscourse.org/ — the gold-standard
  *DL-systems* course; students build a framework from scratch (autodiff + CPU/CUDA
  backends), "Hardware Acceleration + GPUs" lectures. Authority for the
  device/GPU-internals angle (less on checkpointing/AMP APIs).
- **MIT 6.5940 "TinyML and Efficient Deep Learning"** (Song Han) —
  https://hanlab.mit.edu/courses/2024-fall-65940 — the canonical academic treatment
  of precision/efficiency (quantization, mixed precision). Best anchor for M8.
- **Full Stack Deep Learning** (Karayev; originated at UC Berkeley) —
  https://fullstackdeeplearning.com/course/2022/ — the practitioner course on
  training infrastructure: GPUs, mixed precision, DDP, checkpointing. The closest
  match to this chapter's "builder's guide" framing.
- *(Caveat)* Stanford **CS231n** notes — https://cs231n.github.io/ — still the most
  cited intro-CV notes, but **dated** on these systems topics (predate AMP/compile);
  cite for general training framing only, not for the modern toolkit.

**Excellent modern hands-on references (the "real code" bar)**
- **Karpathy, nanoGPT** — https://github.com/karpathy/nanoGPT — `train.py` shows
  *every* missing topic in ~300 lines: `torch.compile`, bf16 `autocast`, gradient
  accumulation, DDP, and a full checkpoint dict with resume. The canonical concrete
  example.
- **Raschka, "Build a Large Language Model (From Scratch)"** —
  https://github.com/rasbt/LLMs-from-scratch — textbook-grade PyTorch for
  checkpointing, mixed precision, `torch.compile`, DDP/FSDP appendices.
- **PyTorch Lightning** — checkpointing
  https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html and
  precision https://lightning.ai/docs/pytorch/stable/common/precision_basic.html —
  shows what the manual checkpoint dict / AMP boilerplate *becomes* in a real
  framework (`precision="bf16-mixed"`, `strategy="ddp"`); the "you'll use a Trainer
  that does this" payoff.
- **HF "Ultra-Scale Playbook"** (2025) —
  https://huggingface.co/spaces/nanotron/ultrascale-playbook — best free deep dive
  on large-scale/distributed training; the forward-pointer target for M10.

---

## 5. Prioritized recommendations (direction, not a line-by-line spec)

This chapter likely needs the most reworking of all of §6. The good news (§0): a
large share of the missing content already exists in draft in the slide decks of
these same two files — the work is substantially *reconciliation + promotion to the
body + verification*, not greenfield writing.

**P0 — the load-bearing fixes (do these or the sections stay below the bar).**
1. **Reconcile slides ↔ body.** Treat the existing decks as the spec they implicitly
   are: promote `weights_only`-rationale, safetensors, the full-checkpoint dict
   (read-write), and the corrected device discipline (use-gpu) **into the prose and
   into runnable notebook cells**. Today the deck teaches a 2024 story the page
   never tells.
2. **`read-write.md`: add the three core topics** — (M2) one paragraph on the
   pickle/`weights_only` security story with the correct **2.6** version; (M1)
   **safetensors** as the modern safe/zero-copy standard, built directly on the
   existing "dict of tensors" cell, with its tensors-only limitation stated; (M3) a
   real **full checkpoint** (model+optimizer+epoch+RNG) + resume, paying off the
   section's own opening motivation.
3. **`use-gpu.md`: de-CUDA-fy and de-risk** — (M4) replace `'cuda'` hardcoding with
   the device-agnostic `torch.accelerator` idiom (and decide the `d2l`
   `gpu()`/`try_gpu` update); (M5) add **MPS** so examples run on the repo's own Mac
   host; (M7) **correct the "GIL stalls GPUs" sentence** to the CPU↔GPU sync-stall
   explanation, and upgrade it into the pinned-memory/`non_blocking` optimization.
4. **`use-gpu.md`: rebalance away from the two-GPU demo.** Demote the "you need two
   GPUs" `copyto` choreography to a short illustration (or fold it into the M10
   pointer); reclaim the space for the single-GPU performance toolkit below. Don't
   open the section by excluding most readers' hardware.

**P1 — the modern single-GPU toolkit (what makes this a 2026 chapter).**
5. **(M8) Mixed precision / AMP**, pointer-level (~½ page): the `autocast` +
   `GradScaler` loop, fp16-vs-bf16, the ~2× / ½-memory payoff; forward-point the
   full treatment.
6. **(M9) `torch.compile`**, short: one-line wrap, the speedup, the warmup caveat.
   This is how the chapter finally delivers the "dramatic speedups" `index.md`
   promises.
7. **(M6) GPU memory**, brief: allocated vs reserved, the caching allocator, OOM as
   the real ceiling, `empty_cache()` semantics — ideally device-agnostic via
   `torch.accelerator`.
8. **(M10) Multi-GPU pointer**, one correct paragraph: DataParallel is legacy → DDP
   (one process/GPU, all-reduce) → FSDP (shard for models too big for one GPU), with
   the URLs. **Stay in lane** — pointer, not chapter.

**P2 — polish and the framework batch.**
9. Refresh exercises in both sections to exercise the new material (save the same
   weights as safetensors and diff against the pickle file; build a checkpoint and
   resume; measure an AMP/`compile` speedup; OOM a batch and read the memory stats).
10. **Defer to the multi-framework pass** (name them, don't fix now): TF H5 → Keras-v3
    `.keras`; JAX `flax.training.checkpoints` → Orbax; JAX/MXNet device-agnostic
    equivalents. (M11) is an optional one-sentence aside on the
    `_orig_mod.`/`module.` state_dict prefix gotcha.

**Scope discipline reminder.** Everything above keeps §6 as the practical builder's
guide: AMP/`compile`/memory get a *present-but-pointer-level* treatment with deep
dives forward-pointed; the multi-GPU/distributed material stays a one-paragraph map
with citations, not a new chapter. The bar is met by the page *being* how we do
this now — device-agnostic, safety-aware, checkpoint-complete, and honest about the
performance tools — not by exhaustiveness.

---

## Appendix — provenance of version-critical claims (verified mid-2026)

| Claim | Status | Version / source |
|---|---|---|
| `torch.load` defaults `weights_only=True` | Confirmed | **PyTorch 2.6** (2025-01-29); warnings since 2.4 |
| pickle = arbitrary-code-exec on load; ~100 malicious HF models | Confirmed | JFrog research; HF safetensors audit (Trail of Bits, May 2023) |
| safetensors = tensors only (no full training checkpoint) | Confirmed | by design (HF docs) |
| HF `save_pretrained` defaults to safetensors | Confirmed | transformers **v4.35** (Nov 2023) |
| `torch.accelerator` device-agnostic API; quickstart uses it | Confirmed | **PyTorch 2.6**; official quickstart tutorial |
| MPS: `'mps'` device, beta, no fp64, fallback env var | Confirmed | since 1.12; Apple/PyTorch docs |
| "GIL stalls GPUs" is dated → CPU↔GPU sync-stall | Confirmed | PyTorch CUDA-semantics notes |
| `torch.cuda.amp.*` deprecated → `torch.autocast`/`torch.amp` | Confirmed | since **2.4** |
| fp16 needs GradScaler; bf16 does not | Confirmed | PyTorch AMP docs |
| `torch.compile` shipped, ~43% faster training on A100 (mature/recommended; no single documented Beta→GA version) | Confirmed | **PyTorch 2.0**, March 2023 |
| DDP recommended over DataParallel even single-node; FSDP for huge models | Confirmed | PyTorch CUDA/DDP/FSDP notes |
| `_orig_mod.` (compile) / `module.` (DDP) state_dict prefix gotcha | Confirmed | PyTorch issue #94575; `consume_prefix_in_state_dict_if_present` |
| Stable PyTorch = 2.12; repo pins 2.11.0 (MPS on darwin/arm64) | Confirmed | docs redirect + `pyproject.toml` |
