# Builders' Guide (v2 proposal)
:label:`chap_computation_v2`

> **Status: outline for review.** One notebook per section; each notebook lists
> its subsections and the topics each will cover, with indicative PyTorch code
> sketches (JAX follows once the structure is approved; TensorFlow/MXNet tabs
> are secondary). Markers: **[KEPT]** carried over from the current chapter,
> **[MOD]** carried over but modernized, **[NEW]** did not exist in the current
> chapter, **[CUT]** current material dropped or demoted.

Alongside giant datasets and powerful hardware, great software tools have
played an indispensable role in the rapid progress of deep learning. This
chapter keeps its original contract: no new models, no new datasets — it is
the chapter that turns you from an *end user* of a deep learning framework
into a *power user*. What changed since the first edition is what a power
user needs. In 2020 the working assumption was a small fp32 model, trained
from scratch on one GPU, saved with a one-liner. In 2026 the default unit of
work is a model built from a configuration object, measured in gigabytes,
run in reduced precision, checkpointed together with its optimizer state,
and — as often as not — *initialized from someone else's weights* rather
than from a random number generator.

Accordingly, this chapter covers, in order: how models are assembled from
modules and configs (:numref:`sec_model_construction_v2`); what a model's
state actually is — parameters, buffers, and the memory they occupy
(:numref:`sec_parameters_v2`); how state is initialized
(:numref:`sec_init_v2`); how to build layers the framework does not ship
(:numref:`sec_custom_layers_v2`); numerics — dtypes and mixed precision
(:numref:`sec_numerics_v2`); serialization, checkpointing, and pretrained
weights (:numref:`sec_read_write_v2`); GPUs, devices, and memory
(:numref:`sec_use_gpu_v2`); and reproducibility and inspection
(:numref:`sec_repro_v2`).

The boundaries with neighboring chapters are unchanged: the training-loop
scaffolding (`Module`/`DataModule`/`Trainer`) was introduced in
:numref:`sec_oo-design`; the *theory* of initialization lives in
:numref:`sec_numerical_stability`; anything involving more than one device
belongs to the computational-performance chapter; hardware procurement and
cloud tooling stay in the appendix.

**Section map (current → proposed):**

| Current (7 sections) | Proposed (8 sections) |
|---|---|
| 6.1 model-construction | 6.1 Modules and Model Construction **[MOD]** (+ lazy init folded in) |
| 6.2 parameters | 6.2 Parameters, State, and Memory **[MOD+NEW]** |
| 6.3 init-param | 6.3 Initialization **[MOD]** |
| 6.4 lazy-init | *(folded into 6.1)* **[CUT as standalone]** |
| 6.5 custom-layer | 6.4 Custom Layers and Functions **[MOD]** |
| — | 6.5 Numerics: Dtypes and Mixed Precision **[NEW]** |
| 6.6 read-write | 6.6 Saving, Loading, and Pretrained Weights **[MOD+NEW]** |
| 6.7 use-gpu | 6.7 GPUs, Devices, and Memory **[MOD]** (keeps `try_gpu` + `Trainer` patch) |
| — | 6.8 Reproducibility and Inspection **[NEW]** |

## Framework Coverage Summary

Verified per-framework (JAX/TF empirically against freshly built venvs —
jax 0.10.0 / flax 0.10.6 linen / optax 0.2.8 / orbax 0.11.24 and TF 2.21.0
/ Keras 3.14.0; MXNet against the pinned 2.0 wheel's source plus the
committed green-run output store). Per-section details in each notebook's
**Framework Coverage** footer. The matrix:

| Section | JAX | TensorFlow | MXNet |
|---|---|---|---|
| 6.1 Modules & Construction | ✓ (2× better) | ✓ | ✓ (lazy init best-in-class) |
| 6.2 Parameters, State, Memory | ✓ (tying/freezing/EMA cleaner) | ✓ (EMA built-in) | ✓ (freezing cleanest) |
| 6.3 Initialization | ✓ (different idiom: `*_init` args) | ✓ | ✓ (no TruncatedNormal → custom) |
| 6.4 Custom Layers & Functions | ✓ | ✓ | ✓ (`autograd.Function` single-use gotcha) |
| 6.5 Numerics & Mixed Precision | ✓ bf16-only (no GradScaler) | ✓ (no native fp8) | **reduced**: fp16+master weights; amp→pointer; tf32/fp8 SKIP |
| 6.6 Saving, Loading, Pretrained | ✓ (orbax cleaner; no model zoo) | ✓ (cleanest: `tf.train.Checkpoint`, `keras.applications`) | partial (numpy-bridge safetensors; zoo on archived S3; no RNG snapshot) |
| 6.7 GPUs, Devices, Memory | ✓ (`remat` first-class) | ✓ (`recompute_grad`) | **SKIP activation ckpt**; coarse memory info; `waitall` lesson best-in-class |
| 6.8 Reproducibility & Inspection | ✓ (star tab: PRNG + `capture_intermediates`) | partial (hooks → surgery/override) | ✓ (fwd hooks real; no bwd hooks; determinism = env var) |

**Cross-cutting facts.**

- `safetensors` becomes a new dependency for **all four** framework extras
  (verified: flax / tensorflow / numpy-bridge modules all work).
- **Keras 3 tripwires** (TF tab, both verified): `by_name=True` now raises
  on native `.weights.h5` (use `skip_mismatch=True` alone); a
  `tf.train.Checkpoint` restore needs `opt.build(vars)` first.
- **Where a non-PyTorch tab teaches it better** — highlight, don't hide:
  flax configs-are-dataclasses & `Embed.attend` tying; TF's dtype
  strictness, built-in EMA, `keras.applications`; gluon's native deferred
  init and `npx.waitall()` sync lesson.

**Decisions needing sign-off at rewrite time.**

1. JAX 6.5 teaches **bf16-only** (optax has no GradScaler; fp16 loss
   scaling skipped as legacy practice). Stay on **flax.linen** (nnx exists
   but would mean rewriting the book's jax core; one prose aside only).
2. MXNet 6.5 ships the **reduced variant** (fp16 + `multi_precision`
   verified path; `mxnet.amp` demoted to pointer unless GPU box validates).
3. 6.6's pretrained-weights demo is per-framework: torchvision (pt),
   `keras.applications` (tf), own-checkpoint + HF-in-prose (jax),
   `model_zoo` if its S3 still resolves (mxnet).

**d2l library surface (what ch6 exports, and its fate).** The freshness
gate fingerprints per-symbol shards stamped `Defined in :numref:<label>` —
so the rule is: *byte-identical block + kept label = zero downstream
recapture*.

| Symbol (`#@save`) | Downstream use | Fate |
|---|---|---|
| `try_gpu` / `try_all_gpus` / `num_gpus` / `cpu` / `gpu` | 28 / 18 / 25 / (internal) files | **Keep byte-identical** in 6.7; prose points at native `torch.accelerator` |
| `Trainer.__init__/prepare_batch/prepare_model` patches | every `Trainer(num_gpus=…)` cell (25 files) | **Keep byte-identical** in 6.7 |
| `Module.apply_init` (pt+jax) | 8+ files (`apply_init([X], init_cnn)`) | **Keep byte-identical**; subsection keeps `sec_lazy_init` label |
| `Module.set_scratch_params_device` (mxnet) | **0 files** | **Drop** (dead weight) |
| `RMSNorm` (new, 6.4) | future chapters | **No `#@save`** — natives exist in torch/flax/keras; build-then-compare |
| `save_checkpoint`/`load_checkpoint` (new, 6.6) | future chapters | `#@save` **pytorch+mxnet tabs only**; TF/JAX use natives |
| `Trainer` amp flag (6.5, deferred) | — | any Trainer-shard edit = book-wide recapture → keep amp local to 6.5 for now |

**GPU-box verification checklist** (claims that cannot be verified on a
CPU Mac; check before the actual rewrite): jax `memory_stats()` dict + XLA
determinism flag cost; TF memory_info plateaus, determinism slowdown,
`recompute_grad` timing; mxnet `amp` end-to-end, `cast('float16')`+
`multi_precision` training, bf16 arrays, `Trainer.save_states` round-trip,
model-zoo download, `autograd.Function` in a loop, dataloader-worker
seeding.

```toc
:maxdepth: 2

model-construction
parameters-state-memory
init
custom-layers
numerics
saving-loading
gpus-devices-memory
reproducibility-inspection
```
