# Reproducibility and Inspection
:label:`sec_repro_v2`

> **Role.** Entirely new, and the shortest section. Two power-user skills
> the current chapter never teaches: making runs repeatable (seeding and
> determinism, including why JAX handles randomness the way it does), and
> looking inside a running model (hooks). Both are prerequisites the book
> currently leaves to folklore; both will be quietly load-bearing later
> (debugging training runs; interpretability-flavored content).

## Seeds and Randomness **[NEW]**

*Topics.* Where randomness enters: init (:numref:`sec_init_v2`), dropout,
data shuffling/augmentation, and dataloader *workers* (the classic
"seeded everything, still not reproducible" hole). Seeding each source in
PyTorch (`torch.manual_seed`, generator objects, worker_init_fn in one
sentence). **The JAX contrast as a concept, not a quirk:** stateless,
explicitly-threaded PRNG keys (`random.split`) make randomness part of the
program's dataflow — verbose, but it eliminates the entire class of hidden
global-state bugs this subsection is about; PyTorch's global seed is the
convenient version of the same contract. *(This is where the current
chapter's scattered Flax `init(key, ...)` mechanics get their proper
conceptual home.)*

*Code (PyTorch).* Same script run twice unseeded vs seeded (weights and
loss trajectories compared); a dropout layer sampled under a fixed
generator.

## Determinism and Its Price **[NEW, short]**

*Topics.* Seeding makes the *program* repeatable, not the *arithmetic*:
nondeterministic kernels (atomics), `torch.use_deterministic_algorithms(True)`
and its cost/error behavior, cuDNN benchmark mode; and the honest rule —
bitwise reproducibility is a debugging tool, statistical reproducibility
(same curves, same conclusions) is the scientific goal. Ties back to
:numref:`sec_numerics_v2` (dtype changes results in the last bits by
design).

*Code (PyTorch).* Two seeded runs on GPU that differ without the
determinism flag and agree with it; measure the slowdown.

## Hooks: Looking Inside **[NEW]**

*Topics.* The `__call__` → `forward` gap from
:numref:`sec_model_construction_v2`, now cashed in:
`register_forward_hook` to capture per-layer activations without touching
model code; backward hooks for gradients in one paragraph. Three canonical
uses: (i) activation statistics per layer (revisits the init experiments of
:numref:`sec_init_v2` without editing the model), (ii) grabbing features
from a pretrained backbone (:numref:`sec_read_write_v2`'s resnet — the
"penultimate-layer embedding" workflow), (iii) debugging NaNs by hooking
every module and reporting the first offender. Removing hooks; hooks vs
editing `forward` (inspection vs architecture).

*Code (PyTorch).* Hook the config-built MLP, collect activation stds per
block, print the table — the init experiment of :numref:`sec_init_v2`
reproduced in five lines on an unmodified model; a NaN-finder hook as the
practical payoff.

## Summary and Exercises

*Exercises (sketch).* (1) Add a dataloader with `num_workers=4` to the
seeded script — reproducible? Fix it. (2) Use a forward hook to count
FLOPs-by-layer for the MLP (multiply-accumulate arithmetic from the shape
of each Linear). (3) Register a backward hook that clips per-layer
gradient norms and compare with global clipping. (4) In the JAX tab:
implement the same activation-statistics capture with
`flax.linen.intercept_methods` (or capture via `Module.sow`) and compare
ergonomics with hooks — which contract do you prefer, and why?

> **Downstream constraints.** None existing (new section). Provides the
> inspection vocabulary (hooks, activation capture) that later
> interpretability-adjacent content can assume, and gives JAX's PRNG model
> a conceptual anchor the JAX tabs can reference across the whole book.

## Framework Coverage

- **JAX** — the star tab for this section, as planned. Seeding: the
  explicit-PRNG story narrates what `d2l.get_key()`/`Trainer.fit(key=...)`
  already do. Hooks: `apply(..., capture_intermediates=True)` is a
  *zero-code-change* hook analogue (verified: full per-module intermediate
  dict) — lead with it; `Module.sow` as the opt-in fine-grained variant.
  Determinism: PRNG is bitwise-deterministic by construction (threefry
  counter-based, verified in source); XLA op-determinism is an env flag
  (`--xla_gpu_deterministic_ops`) [GPU-box: verify + cost].
- **TensorFlow** — seeding is *cleaner than PyTorch*:
  `keras.utils.set_random_seed()` seeds Python/NumPy/TF in one call
  (verified bit-identical draws); `enable_op_determinism()` verified
  callable [GPU-box: measure slowdown]. **Hooks: honest PARTIAL** — no
  post-hoc attach on an unmodified model; teach the two real idioms
  (functional surgery exposing intermediate tensors — verified; `call()`
  override stashing activations — verified) and say plainly that the
  "hook a black box without touching its code" pitch has no TF equivalent.
  No backward hooks either.
- **MXNet** — **full co-equal tab, surprisingly**: `register_forward_hook`
  / `register_forward_pre_hook` confirmed in source (with an internal
  precedent — `Block.summary()` is implemented with one); one-call
  multi-device seeding via `mx.random.seed(seed, device='all')` (verified).
  Reductions: **backward hooks SKIP** (post-hoc `.grad()` inspection
  only); determinism PARTIAL — no strict/error-raising mode, just the
  `MXNET_CUDNN_AUTOTUNE_DEFAULT` env lever; dataloader-worker seeding
  [UNVERIFIED — no `worker_init_fn` equivalent found].
