# Chapter 6 "Builders' Guide" — Cross-Cutting Landscape Review

**Reviewer role:** Big-picture / structure / missing-topics (whole chapter).
**Date:** 2026-06-13. **Repo:** `d2l-neu`. **Primary lens:** PyTorch 2.x (4-fw book).
**Scope reminder:** §6 is the practical "how to build/use models" guide *between* the
MLP chapter (Ch. 5) and the architecture chapters (CNNs onward). It is **not** the
performance chapter — the book already ships a separate **Computational Performance**
chapter (`chapter_computational-performance/`: Async, Auto-Parallelism, Hardware,
Compilers/Interpreters, Multiple GPUs ×2, Parameter Servers). That fact governs most
of the scope verdicts below.

This is a **landscape review, not a change spec.** It sets direction; the three
section-level agents (modules/params, init/lazy, custom-layer/IO/device) own the
line edits.

---

## 1. Executive read on the chapter's current shape

The chapter is **7 sections**, in this order:

1. `model-construction.md` — **Layers and Modules** (`nn.Module`, `Sequential`,
   custom module, `MySequential`, control flow in `forward`, nesting). 32 KB — the
   anchor section, and the strongest.
2. `parameters.md` — **Parameter Management** (access by index / `named_parameters`,
   `.grad`, tied/shared parameters).
3. `init-param.md` — **Parameter Initialization** (built-in + custom initializers,
   `net.apply`, Xavier/Kaiming in the slides).
4. `lazy-init.md` — **Lazy Initialization** (`LazyLinear`, deferred shape inference).
5. `custom-layer.md` — **Custom Layers** (stateless + with-`nn.Parameter`, buffers).
6. `read-write.md` — **File I/O** (tensor save/load, `state_dict`, checkpoint clone).
7. `use-gpu.md` — **GPUs** (device handles, tensor/model placement, copies, Trainer).

**Level and freshness.** The prose is 2021-vintage but the *code* has been carefully
modernized to current APIs: `nn.LazyLinear` throughout, `register_buffer` for the
fixed weight, `torch.load(..., weights_only=True)`, `assert ... is ...` for tying,
Gluon-2.0 weakref notes, JAX/Flax decoupled-params narration, and Keras-v3 caveats.
The **slides are markedly ahead of the prose** — they already name-drop safetensors,
full-checkpoint dicts (`optimizer`/`epoch`/`rng_state`), `strict=False` partial
loading, LoRA/low-rank as a custom-layer motivation, FixUp/Skip-init, tied embeddings,
and `.numpy()`-forces-a-sync. Several of the "missing topics" below already exist **as
one slide bullet** but have **no prose, no code cell, and no exercise**. The cheapest,
highest-value wins are to promote those bullets into the body.

**Overall:** solidly at the bar for *mechanics*, but it teaches the **2021 power-user
workflow**. A 2026 Stanford/MIT/CMU student leaves this chapter able to build and save
a model, but **unaware of the five things they will actually type first** in a modern
project: `torch.compile`, autocast/AMP, `safetensors`, a device-agnostic
(`cuda`/`mps`/`cpu`) accelerator handle, and a one-line model summary. The chapter's
own framing line — "move you from end user to power user" — is exactly right; the
definition of "power user" has just moved.

**Cross-cutting nits visible from the 30,000-ft view (hand to section agents):**
- **No through-line / running example.** Each section re-imports and rebuilds a fresh
  toy MLP from scratch (`nn.Sequential(nn.LazyLinear(8|256), ...)`). Seven cold starts.
  A single MLP carried across "build → inspect params → init → save → move to GPU"
  would make the chapter feel like one workflow instead of seven recipes.
- **Exercise depth is thin and uneven.** `init-param` has a one-line exercise ("Look
  up the docs"); `parameters`/`custom-layer` are good. Several sections have no
  "verify it / measure it" exercise, which is the d2l house move.
- **`use-gpu` is doing two jobs** (single-device placement *and* a `MirroredStrategy` /
  multi-GPU Trainer cameo) and is single-vendor ("at least two **NVIDIA** GPUs", CUDA
  10.0 install text). See §3 and §4.
- **Ordering micro-issue:** `init-param` (3) before `lazy-init` (4) forces the init
  section to lean on `net.apply` over already-materialized params, then §4 reveals the
  params didn't exist until a forward pass. Mild, but §4-before-§3 is more honest. (Low
  priority; flagged for the init/lazy agent.)

---

## 2. How the best current programs/resources teach this material (2026)

I surveyed the canonical sources for "building & using models." Key finding: **nobody
treats this as one monolith.** The strong resources split into (a) a tight "module +
params + save/load + device" core, and (b) a separate "make it fast / make it fit"
systems unit. d2l already mirrors this split (Ch. 6 vs. Computational-Performance) —
the opportunity is to add the **2026 connective tissue** to the core and forward-point
to the systems chapter.

**Official PyTorch — "Learn the Basics" + "Introduction to PyTorch (introyt)"**
([Build Model](https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html),
[Save/Load](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)).
*Sequence:* `nn.Module`+`Parameter` → common layer types → save/load via `state_dict`
→ **`model.eval()` before inference** (d2l's pytorch tab does call `.eval()` in the
clone; the prose never explains *why*). *Does well:* the canonical `state_dict`
mental model d2l's slides already adopt. This is d2l §6's direct analogue and the two
agree closely — d2l is actually broader (it has lazy-init, tying, device).

**Official PyTorch intermediate recipes — the "missing half."** These are the 2026
add-ons, all current (verified June/July 2025):
[`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
(JIT to fused kernels, `opt = torch.compile(model)`),
[AMP / `torch.amp`](https://docs.pytorch.org/docs/stable/amp.html) (`autocast` +
`GradScaler`; bf16 needs no scaler),
[Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
(self vs total CUDA time),
[Parametrizations](https://docs.pytorch.org/tutorials/intermediate/parametrizations.html)
(`torch.nn.utils.parametrize`; modern `weight_norm` is built on it),
[`torch.utils.checkpoint`](https://docs.pytorch.org/docs/main/checkpoint.html) (gradient
checkpointing, ~3× memory for ~20% compute).
*Does well:* each is a self-contained 1-page recipe — exactly the granularity a d2l
subsection wants.

**Stanford CS336 "Language Modeling from Scratch" (Spring 2025)**
([site](http://cs336.stanford.edu/spring2025/), [assignments](https://github.com/stanford-cs336)).
The single most useful structural data point. Its lecture order:
**L2 "PyTorch, resource accounting"**, L5 "GPUs", L6 "Kernels, Triton", L7–8
"Parallelism." **Assignment 2 (Systems)** = "profile and benchmark the model and
layers ... optimize attention with your own Triton FlashAttention2 ... build a
memory-efficient, **distributed** version." *Takeaway:* the elite 2026 treatment puts
**profiling/benchmarking + mixed precision + memory** right after the build-it
mechanics — but Triton/FSDP/scaling laws are a *separate systems course*, not the
build-a-model lesson. This validates: **profiling/AMP belong adjacent to §6; Triton &
real distributed training do not.**

**CMU 11-785 Intro to DL** ([S25](https://deeplearning.cs.cmu.edu/S25/index.html)).
NumPy-then-PyTorch; dedicated PyTorch recitations/bootcamps teach `nn.Module`,
`Dataset`/`DataLoader`, and the train loop as one workflow before architectures.
*Does well:* the "one continuous PyTorch walkthrough" framing d2l §6 lacks (see the
no-through-line nit).

**Stanford CS231n** ([site](https://cs231n.stanford.edu/)). Assignment 2 has students
build the same net three ways — bare tensors, `nn.Module`, then `nn.Sequential` —
making the abstraction *ladder* explicit. d2l's `MySequential` reaches for the same
"de-magic the framework" goal; CS231n's contrast is sharper. *Does well:* motivating
`nn.Module` by first suffering without it.

**fast.ai "Practical Deep Learning"** ([course.fast.ai](https://course.fast.ai/)).
Reimplements `nn.Module`/`Parameter`/`Learner` from scratch to demystify them — same
spirit as d2l's `MySequential` and `MyDense`. *Does well:* "no magic" pedagogy.

**HuggingFace** (course + `safetensors`/`transformers`). Cements **`safetensors` as
the default serialization format** in 2026 (zero-copy, no pickle/code-exec, framework-
agnostic) and `from_pretrained`/`save_pretrained` as the dominant save/load idiom.
d2l's read-write slide already says "HuggingFace standard, used by every modern
library" — but the prose still teaches only pickled `state_dict`.

**Prince, "Understanding Deep Learning"** ([udlbook](https://udlbook.github.io/udlbook/)).
Theory-first with per-chapter PyTorch notebooks; initialization (He/Glorot, variance
propagation) is a *first-class chapter*, not a subsection. d2l keeps the variance
argument in the init **slides** — promoting a compact version into the init prose would
match UDL's rigor and the d2l "intuition-first proof" house style.

**einops** ([einops.rocks](https://einops.rocks/), Imperial
[ReCoDE best-practices](https://imperialcollegelondon.github.io/ReCoDE-DeepLearning-Best-Practices/)).
`rearrange`/`reduce`/`repeat` are now a near-default readability layer in research code
and teaching materials. Relevant to §6 only as a *custom-layer / forward-method* tool
(shape-safe tensor ops); see verdict.

---

## 3. Missing-topics audit (researched; each with a verdict)

Legend: **ADD-§6** (belongs in this chapter) / **POINTER** (mention + forward-ref, home
is elsewhere) / **ELSEWHERE** (out of lane) / **EXPAND** (already present as a slide
bullet or stub — promote to prose/code).

| # | Topic | Verdict | Rationale / where |
|---|---|---|---|
| 1 | **`safetensors`** | **ADD-§6** (`read-write`) | The 2026 default save format (HF). Slide already sells it; prose teaches only pickle. Add a short subsection: `save_file(net.state_dict(), 'm.safetensors')` / `load_file`, why no-pickle matters (the section already uses `weights_only=True` for the same reason). Highest value/effort ratio in the chapter. |
| 2 | **`torch.compile`** | **POINTER from §6, home = Computational-Performance/`hybridize`** | One-liner that wraps any `nn.Module` — naturally belongs to the "model is built, now what" arc. But the book's **Compilers/Interpreters** section (`hybridize.md`) is its real home (it already covers the symbolic/imperative split MXNet `hybridize` + TF graph). Add a 2–3 sentence pointer in `model-construction` ("`opt = torch.compile(net)` to fuse kernels; see :numref:`...hybridize`") and do the depth there. |
| 3 | **Mixed precision / AMP** (`autocast`, `GradScaler`, bf16) | **POINTER from §6 → Optimization/Performance** | A 2026 student types `autocast` in week one. But it's a *training-loop* concern, and the book's training loop lives in Ch. 3/Optimization. Best as a pointer from `use-gpu`'s "side notes" ("modern GPUs run matmuls in bf16/fp16; wrap the forward in `torch.autocast`, see ..."). Do **not** build a full AMP subsection in §6. |
| 4 | **Device-agnostic code (CUDA/MPS/CPU, `torch.accelerator`)** | **EXPAND in-place (`use-gpu`)** | *Single biggest correctness gap.* The chapter is hard-wired to NVIDIA ("at least two **NVIDIA** GPUs", `pip install mxnet-cu100`, `try_gpu` only knows CUDA). The book itself renders/ runs on Apple-Silicon CPU (per CLAUDE.md). 2026 best practice = check `cuda` → `mps` → `cpu`, or use the new unified [`torch.accelerator`](https://runebook.dev/en/docs/pytorch/accelerator) API. Generalize `try_gpu` to an accelerator-aware handle and reword the NVIDIA-only framing. **Belongs squarely in §6.** |
| 5 | **Forward/backward hooks** (`register_forward_hook`) | **ADD-§6 (small), in `parameters` or `model-construction`** | The natural companion to "access parameters": hooks access *activations/grads* without editing `forward`. Core uses — feature extraction, activation/grad debugging, Grad-CAM — are exactly §6's "inspect your model" theme. Add a short subsection (one hook capturing a layer's output). High pedagogical value, low risk. |
| 6 | **`torch.nn.utils.parametrize`** | **EXPAND (one slide already gestures at it) — `custom-layer`** | The modern way to constrain/reparameterize weights (orthogonal, spectral-norm, the new `weight_norm`). Fits `custom-layer` precisely (it's "a custom layer *of the weight itself*"). Worth a compact subsection or a strong exercise; ties to the LoRA mention already in the slides. Medium priority. |
| 7 | **Model summary / inspection (`torchinfo`)** | **ADD-§6 (small), `parameters`** | Every course has students run a `summary(model, input_size)` to see shapes + param counts. d2l has `named_parameters()` listing shapes but no param-count / layer-by-layer view. A 4-line `torchinfo.summary` cell (or a hand-rolled `sum(p.numel() ...)`), framed as "the first thing you do to a model," is a cheap, universally-expected add. |
| 8 | **Reproducibility / seeding & determinism** | **ADD-§6 (short, cross-cutting)** | `torch.manual_seed`, `use_deterministic_algorithms`, why results wobble across runs/devices, and the **`rng_state` in a full checkpoint** (already in the read-write slide!). No single home today; a short box (likely in `init-param`, since init is the first place randomness bites, or `read-write` for the checkpoint angle) closes it. Expected at the bar. |
| 9 | **Basic profiling (`torch.profiler`)** | **POINTER → Computational-Performance** | CS336 puts profiling right after build-it, but d2l's `use-gpu` exercises *already* do hand-timed CPU-vs-GPU / log-on-GPU-vs-CPU benchmarks — the right altitude for §6. Keep those; add a one-line pointer to `torch.profiler` and the performance chapter for the real tool. Don't import the profiler API into §6. |
| 10 | **einops / tensor rearrangement** | **POINTER (optional), `custom-layer`** | Genuinely popular and readability-positive, but it's a *tensor-ops* utility, not a model-building primitive, and adding a non-stdlib dep to the teaching corpus has a cost. At most a sidebar in `custom-layer` ("for shape-juggling inside `forward`, `einops.rearrange` is far more readable than `.permute().reshape()`"). Not a section. |
| 11 | **Gradient checkpointing** (`torch.utils.checkpoint`) | **ELSEWHERE → Computational-Performance / large-model material** | A memory-vs-compute trade for *big* models. Out of lane for an MLP-era "how to build" chapter. At most a one-sentence forward-pointer from `use-gpu` common-mistakes ("activations dominate memory; `torch.utils.checkpoint` trades compute for memory — see ..."). |
| 12 | **DDP / FSDP (multi-GPU/-node)** | **ELSEWHERE (already exists)** | The book has **`multiple-gpus.md` + `multiple-gpus-concise.md` + `parameterserver.md`**. §6's job is the *single-device* mental model + one forward-pointer. The current `MirroredStrategy`/multi-GPU Trainer cameo in `use-gpu` should be **trimmed to a pointer** (see §4). |
| 13 | **Config / experiment tracking** (Hydra, W&B) | **ELSEWHERE / out of lane** | MLOps tooling, not core model-building. A book this framework-neutral shouldn't endorse a tracker. Skip (at most a one-line "in practice, track runs with a tool like W&B" aside in a later applied chapter). |
| 14 | **`model.eval()` / train-vs-eval mode** | **EXPAND (latent gap)** | The read-write pytorch tab *calls* `clone.eval()` but never explains it; dropout/BN behave differently in eval. This is a classic footgun. A two-sentence callout (likely `read-write` "loading: instantiate, then load" or a forward-ref to dropout/BN) is worth more than several of the flashier adds. |

**Net:** the chapter's biggest gaps, in priority order, are **(4) device-agnostic /
non-NVIDIA**, **(1) safetensors**, **(7) summary**, **(5) hooks**, **(8)
reproducibility**, **(14) `eval()`** — all genuinely in §6's lane and mostly cheap.
The flashy systems items (compile/AMP/profiling/checkpointing/DDP) are real but belong
**downstream**, reached from §6 by *pointers*, because the book already has the
chapters for them.

---

## 4. Restructure / reorder / split — recommendation

**Do NOT split the chapter.** Seven short, coherent sections at one altitude is the
right shape; it matches how PyTorch's own "Learn the Basics" and CMU's recitations
chunk the material. The fixes are **resequencing + scope-trimming + targeted adds**,
not surgery.

**Three structural moves (landscape level):**

1. **Add a 7th-vs-1st framing + a running example.** Open §6 with a single MLP and
   carry it through build → inspect → init → save → device, so the chapter reads as
   one workflow (the CMU/PyTorch-basics through-line). Today it's seven cold starts.

2. **Refocus `use-gpu` on the single-device model and make it vendor-neutral.** Pull
   the `MirroredStrategy` / multi-GPU Trainer block down to a forward-pointer to the
   Computational-Performance chapter; spend the reclaimed space on
   `cuda`/`mps`/`cpu` device-agnostic code (gap #4). This is the section most out of
   step with 2026 and with the book's own Apple-Silicon build.

3. **Promote the slide-only 2026 content into prose.** safetensors, full-checkpoint
   dict, `strict=False`, the variance/Xavier/Kaiming argument, and `register_buffer`
   already exist as polished *slide* bullets with no body text. Promoting them is
   low-risk and closes much of the gap without new research.

**Optional reorder (low priority, for the init/lazy agent):** consider
`lazy-init` *before* `init-param`, so "params don't exist until the first forward"
precedes "here's how to set their values." Defensible either way; not worth a fight.

---

## 5. Proposed modern §6 "north-star" outline

Section list with a one-line charter each. **Bold = changed from today.** This keeps 7
core sections + optional adds, stays in the build/use lane, and pushes systems topics
to pointers.

0. **`index.md` (cover)** — keep the "end user → power user" framing; **add one
   sentence** that the *definition* of power-user now includes compiling, mixed
   precision, and safe serialization, with forward-pointers to the Performance chapter.
1. **Layers and Modules** — `nn.Module`/`Sequential`/custom/`MySequential`/control-flow
   (keep ~as-is; strongest section). **Add:** one-paragraph pointer that a finished
   module can be `torch.compile`d (link Performance ch.). *Optional:* start the running
   MLP here.
2. **Parameter Management** — access, `named_parameters`, tying. **Add:** model
   **summary / param-count** (gap #7) and a short **forward/backward hooks** subsection
   (gap #5) — both are "inspect your model," the section's theme.
3. **Parameter Initialization** — built-in + custom + `net.apply`. **Add:** promote the
   slides' variance → Xavier/Kaiming argument into a short intuition-first prose block
   (UDL-style); add a real exercise (the current one is "read the docs"). Mention
   seeding here (gap #8) since init is where randomness first bites.
4. **Lazy Initialization** — keep; optionally sequence before §3.
5. **Custom Layers** — stateless + `nn.Parameter` + buffers. **Add:**
   `torch.nn.utils.parametrize` subsection or strong exercise (gap #6); optional einops
   sidebar for shape-ops in `forward` (gap #10).
6. **Saving and Loading** *(retitle from "File I/O")* — `state_dict`/clone (keep).
   **Add:** **safetensors** subsection (gap #1); **full-checkpoint dict** (model+opt+
   epoch+rng_state) promoted from slide to prose (gaps #8, partial); a sentence on
   **`model.eval()`** and `strict=False` (gaps #14). This becomes the most-improved
   section.
7. **Devices** *(retitle/refocus from "GPUs")* — single-device placement, copies,
   inner-loop hygiene (keep the good benchmark exercises). **Rework:** vendor-neutral
   `cuda`/`mps`/`cpu` (or `torch.accelerator`) device handle (gap #4); **trim** the
   multi-GPU/`MirroredStrategy` cameo to a forward-pointer (gap #12); one-line pointers
   to AMP (#3), profiler (#9), gradient checkpointing (#11).

*Everything systems-heavy (real `torch.compile` depth, AMP recipe, Triton, profiler,
multi-GPU, gradient checkpointing) stays in **Computational-Performance** and is
**reached from §6 by pointers**, not duplicated.*

---

## 6. Prioritized ADD / REWORK / CUT

**ADD (in priority order — all in-lane, mostly cheap):**
1. **safetensors** subsection in `read-write` (gap #1). *High value, low effort; slide already exists.*
2. **Vendor-neutral device handle** (`cuda`/`mps`/`cpu` or `torch.accelerator`) in `use-gpu` (gap #4). *Correctness + matches the book's own CPU/Apple-Silicon build.*
3. **Full-checkpoint dict** (model+optimizer+epoch+rng_state) + **`model.eval()`** note + `strict=False`, promoted to `read-write` prose (gaps #8/#14). *Slide already exists.*
4. **Model summary / param count** cell in `parameters` (gap #7). *Universally expected; 4 lines.*
5. **Forward/backward hooks** subsection in `parameters` (gap #5). *Fills the "inspect activations" hole.*
6. **Init variance → Xavier/Kaiming** prose + a real exercise in `init-param` (promote from slides; gap #8 seeding).
7. **`torch.nn.utils.parametrize`** subsection/exercise in `custom-layer` (gap #6).

**REWORK:**
- `use-gpu`: de-NVIDIA the framing; **trim** the `MirroredStrategy`/multi-GPU Trainer block to a pointer (the depth lives in `multiple-gpus*`).
- Add **pointers** (1–3 sentences each, not sections) from §6 to the Computational-Performance chapter for: `torch.compile` (#2), AMP (#3), profiler (#9), gradient checkpointing (#11), multi-GPU (#12).
- Introduce a **running-example through-line** and lift exercise depth in the thin sections (`init-param` especially).
- Retitle: "File I/O" → "Saving and Loading"; "GPUs" → "Devices."

**CUT / keep-out (do not add to §6):**
- Full AMP recipe, Triton kernels, real DDP/FSDP training, deep profiler API, gradient-checkpointing mechanics — **already downstream**; §6 only points to them.
- Config/experiment-tracking tooling (Hydra/W&B) — out of lane, and against the book's framework-neutral stance.
- einops as a full section — at most a `custom-layer` sidebar.

---

### Sources
- PyTorch — [Building Models](https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html) · [Save/Load Basics](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) · [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) · [torch.amp / AMP](https://docs.pytorch.org/docs/stable/amp.html) · [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) · [Parametrizations](https://docs.pytorch.org/tutorials/intermediate/parametrizations.html) · [torch.utils.checkpoint](https://docs.pytorch.org/docs/main/checkpoint.html) · [torch.accelerator (device-agnostic)](https://runebook.dev/en/docs/pytorch/accelerator)
- Stanford [CS336 Spring 2025](http://cs336.stanford.edu/spring2025/) · [CS336 assignments](https://github.com/stanford-cs336) · [CS231n](https://cs231n.stanford.edu/)
- CMU [11-785 S25](https://deeplearning.cs.cmu.edu/S25/index.html)
- [fast.ai Practical Deep Learning](https://course.fast.ai/)
- Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/)
- [einops](https://einops.rocks/) · Imperial [ReCoDE DL Best Practices](https://imperialcollegelondon.github.io/ReCoDE-DeepLearning-Best-Practices/)
