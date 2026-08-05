# Chapter Overview — chapter_builders-guide

Best external source, by a wide margin: CMU 10-714/714 "Deep Learning Systems" (dlsyscourse.org),
whose students build the `needle` framework from scratch — it has a direct homework analogue for
6 of 8 sections (autograd/custom ops in HW1, init/modules/optimizers in HW2, a CPU+CUDA NDArray
backend in HW3), because both books share a "build your own framework" premise. CMU 11-785 HW1P1
("MyTorch") is the second-best hit, independently re-deriving Linear/activation/BatchNorm
forward+backward without any autograd library. PyTorch's own tutorials/docs supplied the sharpest
single-topic matches — the AMP recipe's own suggested variation is nearly this book's numerics ex2;
the DataLoader docs describe reproducibility-inspection ex1's bug almost verbatim; CUDA semantics
and `torch.utils.checkpoint` docs map cleanly onto gpus-devices-memory. fast.ai Part 2 and JAX's
custom-derivative-rules notebook contributed real but thinner hits. Numerics and gpus-devices-memory
have essentially no textbook-style problem-set tradition anywhere (taught via vendor narrative and
production code, not homework) — the book's own exercises there are unusually load-bearing. Existing
exercises are excellent chapter-wide (echoing the prior style review): 30 of 32 current exercises are
kept as-is; only reproducibility-inspection.md's ex4/ex5 needed a rewrite, for a genuine content bug
(ex4's premise fails for TensorFlow/MXNet per the section's own text) plus a missing PyTorch variant.
Totals: 8 sections, 32 current exercises (keep 30, rewrite 2, drop 0), 54 proposed problems.

---

## chapter_builders-guide/model-construction.md — Modules and Model Construction

**Topic:** the module abstraction (parameters + child modules + forward computation) and building
models from configs and composable blocks.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — all four are concrete
diagnostic/build tasks with an explicit verification step and no clarity defects; this file's only
defect (non-canonical tab order, missing Discussions block) is structural, not a content problem.

**External sources found:**
- CMU 10-714/714 Deep Learning Systems (dlsyscourse.org), HW2 Question 2, 2023 — implement Linear,
  ReLU, Sequential, LayerNorm1d, BatchNorm1d, Dropout, and Residual as `needle.nn.Module`
  subclasses/instances, the same module vocabulary this section builds. — https://github.com/dlsyscourse/hw2
- CMU 11-785 Introduction to Deep Learning, HW1P1 "An Introduction to Neural Networks," Fall 2023 —
  without any autograd library, hand-implement a Linear layer's forward/backward and chain it into
  MLP models with 0, 1, and 4 hidden layers. — https://deeplearning.cs.cmu.edu/F23/document/homework/HW1/HW1P1_F23.pdf
- PyTorch official tutorial, "Learning PyTorch with Examples," Custom nn Modules section — subclasses
  `nn.Module`, defines learnable `nn.Parameter` fields in `__init__`, and a `forward` method: the
  minimal illustration of the module abstraction. — https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html
- PyTorch official tutorial, "Build the Neural Network" — nests a `Flatten` + `Sequential(Linear,
  ReLU, ...)` stack inside a subclassed `nn.Module`, then iterates `model.named_parameters()` to show
  child registration in action. — https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- fast.ai, Practical Deep Learning for Coders Part 2, Lessons 13-14 "Backpropagation & MLP," 2022 —
  builds a matmul-based MLP and its backward pass from first principles, motivating why a module
  abstraction is needed once one hand-written network stops being enough. — https://course.fast.ai/Lessons/lesson13.html

**Proposed problem set** (7 problems):
1. [conceptual] **Unregistered Module Diagnosis.** Take `PlainListMLP` and catalog everything that
   breaks besides the empty parameter list (state dict, dtype casting, `.eval()` reach); explain how
   each failure follows from the same missing registration.
   *Provenance:* original (book's existing ex1, kept).
1. [short-code] **Parallel Block Composition.** Implement a `ParallelBlock` that runs two child
   modules on the same input and concatenates their outputs; state what must hold for the
   concatenation to be valid.
   *Provenance:* original (book's existing ex2, kept).
1. [short-code] **Config-Driven Activation Switch.** Extend `MLPConfig` with an activation switch and
   make `build` honor it; argue which decisions belong in a config versus in code.
   *Provenance:* original (book's existing ex3, kept).
1. [conceptual] **Widening a Residual Block.** Give two standard fixes for a residual block whose
   output must be wider than its input, and the cost of each.
   *Provenance:* original (book's existing ex4, kept).
1. [short-code] **Minimal Module From Scratch.** In plain Python (no framework base class), write a
   bare-bones `Module` that supports registering child modules and parameters and a `.parameters()`
   that recurses into children; wrap two `Linear`-like objects in it as a 2-layer MLP and verify
   `.parameters()` finds every weight.
   *Provenance:* adapted from CMU 11-785 HW1P1 / PyTorch's Custom nn Modules tutorial (overlap med).
1. [conceptual] **Shared Child Aliasing.** Build a small network that reuses one child module at two
   sites in the tree (turning it into a graph, as `fig_blocks` notes). Predict, then verify, whether
   the shared child appears once or twice in the model's parameter listing, and what happens to it
   under a dtype-cast call.
   *Provenance:* original.
1. [extended] **MyTorch-Style MLP Family.** Using the section's own module system, build 0-hidden,
   1-hidden, and 4-hidden-layer variants of the residual MLP on a toy regression target; report a
   table of parameter count versus depth and versus achieved loss.
   *Provenance:* adapted from CMU 11-785 HW1P1's MLP0/MLP1/MLP4 progression (overlap med).

---

## chapter_builders-guide/parameters-state-memory.md — Parameters, State, and Memory

**Topic:** accessing a model's state, distinguishing trained parameters from non-trained buffers,
memory accounting, parameter tying, and freezing.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — well-verified and
section-grounded; the shared ex1/ex2 and framework-idiomatic tabbed ex3/ex4 are all strong.

**External sources found:**
- CMU 11-785 Introduction to Deep Learning, HW1P1 §10.1 "Batch Normalization," Fall 2023 —
  implement BatchNorm1d's training-mode and `eval=True` inference equations by hand, including the
  running-mean/variance update this section's ex2 interrogates as non-trained state. — https://deeplearning.cs.cmu.edu/F23/document/homework/HW1/HW1P1_F23.pdf
- CMU 10-714/714 Deep Learning Systems, HW2 Questions 2-3, 2023 — the same running-statistics
  mechanism plus SGD/Adam `step` implementations that a memory-accounting exercise must reason about
  term by term. — https://github.com/dlsyscourse/hw2
- PyTorch official tutorial, "Build the Neural Network" — demonstrates `model.named_parameters()` as
  the standard way to enumerate learnable state, the access pattern a byte-cost helper must walk. — https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- No good external exercise tradition found for the *tying-then-freezing interaction* specifically
  (ex3/ex4): tutorials document `state_dict`/`requires_grad`/`grad_req` mechanics narratively, but we
  found no source posing "tie two layers, then freeze one side, what trains" as a problem — this
  looks like an original contribution of the book.

**Proposed problem set** (7 problems):
1. [short-code] **Byte-Cost-Per-Block Helper.** Write a helper reporting the byte cost of fp32 Adam
   training separately for each top-level block; identify the dominating block and whether that
   holds if the residual blocks were 10x wider.
   *Provenance:* original (book's existing ex1, kept).
1. [conceptual] **Trainable Running Statistics.** Suppose BatchNorm's running mean/variance were made
   trainable parameters. Explain what breaks during training and why gradient descent on a running
   average isn't the forward-pass update rule it would replace.
   *Provenance:* original (book's existing ex2, kept).
1. [short-code] **Tying Survives Copying.** (Per framework.) Tie two layers as in `TinyLM`, then copy
   or round-trip the model through serialization; check whether the tying is preserved and where, if
   anywhere, it is recorded.
   *Provenance:* original (book's existing ex3, kept).
1. [conceptual] **Freezing One Side of a Tie.** (Per framework.) Freeze the embedding half of a tied
   `TinyLM` but leave the head trainable; count remaining trainable parameters and explain the
   interaction between tying and freezing.
   *Provenance:* original (book's existing ex4, kept).
1. [conceptual] **Buffers-Versus-Parameters Census.** Write a one-line rule classifying every entry
   of the full model state into {trained parameter, non-trained buffer}; verify it against the
   section's own listing API, then predict how many entries change during a single `.eval()`-mode
   forward pass with no backward call, and check by diffing the state before/after.
   *Provenance:* original.
1. [short-code] **Count Share Versus Byte Share.** Extend the byte-cost helper from ex1 to also report
   each block's share of total parameter *count* (not bytes); find a block whose count-rank and
   byte-rank disagree, and explain why from its buffer/parameter dtype mix.
   *Provenance:* inspired by PyTorch's `named_parameters()` enumeration pattern (overlap low).
1. [extended] **Tie Three Ways.** Extend `TinyLM` so the input embedding, a middle projection, and the
   output head all share one tensor; measure the parameter-count reduction versus untied, freeze only
   the shared tensor, and verify via the framework's own gradient inspection that no path feeds it a
   gradient.
   *Provenance:* original.

---

## chapter_builders-guide/init.md — Initialization

**Topic:** library-default initializers, explicit Xavier/He schemes, residual/Transformer init
conventions, and custom initializers.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — this file has the chapter's
heaviest crossref density and no clarity issues; all four are worth keeping as-is.

**External sources found:**
- CMU 10-714/714 Deep Learning Systems, HW2 Question 1 and HW4 Part 1, 2023 — implement
  `xavier_uniform`/`xavier_normal` with the exact fan_in/fan_out gain formulas, then (HW4) extend to
  Kaiming uniform/normal and fix `_calculate_fans` to include receptive-field size. — https://github.com/dlsyscourse/hw2 ; https://github.com/dlsyscourse/hw4
- fast.ai, Practical Deep Learning for Coders Part 2, Lesson 17 "Initialization/normalization," 2022
  — covers Glorot/Xavier, Kaiming/He, and LSUV initialization, tracking per-layer activation mean/std
  to see which scheme keeps signal stable through depth, directly paralleling this section's own
  depth-vs-std plot (ex1). — https://course.fast.ai/Lessons/lesson17.html
- No good external exercise tradition found for the truncated-normal clip-fraction/std-shrinkage
  calculation (ex4): it appears in library docstrings (e.g. `torch.nn.init.trunc_normal_`) as a
  narrative caveat, not posed as a problem elsewhere.

**Proposed problem set** (6 problems):
1. [short-code] **Depth-Versus-Std Curve.** Instrument the residual stack to record per-block
   activation std for the default and scaled treatments at $N=32$; plot against depth and identify
   which curve matches the geometric-growth prediction.
   *Provenance:* original (book's existing ex1, kept).
1. [conceptual] **All-Zero Initialization.** Zero-initialize every layer of every block instead of
   just the output projection; work out which parameters receive nonzero gradient and relate the
   answer to symmetry-breaking.
   *Provenance:* original (book's existing ex2, kept).
1. [short-code] **Name-Keyed Initializer.** Write an initializer that fills each parameter from a
   dictionary keyed by parameter name, walking the framework's own parameter-naming API.
   *Provenance:* original (book's existing ex3, kept).
1. [conceptual] **Truncation Clip Accounting.** For a normal distribution truncated at $\pm2\sigma$,
   compute the discarded fraction and the shrinkage in standard deviation; verify both against the
   truncation demo's printed output.
   *Provenance:* original (book's existing ex4, kept).
1. [short-code] **Kaiming for the Residual Stack.** Implement Kaiming (He) uniform/normal with an
   explicit `nonlinearity` argument switching the gain between 'relu' and 'linear', apply it to the
   residual stack in place of the default scheme, and reproduce ex1's depth-vs-std plot to confirm it
   also keeps std flat.
   *Provenance:* adapted from CMU 10-714/714 HW2 Q1 + HW4's Kaiming fix (overlap high on the formula).
1. [conceptual] **One-Pass Data-Driven Calibration.** Initialize the residual stack with any
   convenient scheme, run one forward mini-batch, and rescale each block's weight by the reciprocal
   of its own output std (an "LSUV-lite" calibration); compare the resulting depth-vs-std curve
   against the analytic scaled scheme.
   *Provenance:* inspired by fast.ai Lesson 17's LSUV coverage (overlap low).

---

## chapter_builders-guide/custom-layers.md — Custom Layers and Functions

**Topic:** writing a custom layer with its own forward computation, and a custom autograd op whose
backward differs from the automatic derivative.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — unusually clean bare-list writing;
every item pairs a build task with an explicit verification or comparison step.

**External sources found:**
- PyTorch official tutorial, "Learning PyTorch with Examples," "Defining New autograd Functions" +
  "Custom nn Modules" — implements `LegendrePolynomial3(torch.autograd.Function)` with a hand-derived
  forward and backward, the same forward/backward-pair pattern as this section's clamp exercise. — https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html
- JAX official tutorial notebook, "Custom derivative rules for JAX-transformable Python functions" —
  walks through `jax.custom_vjp`/`custom_jvp` with worked nonstandard-backward examples, directly
  underlying this section's JAX-tab `custom_vjp` clamp exercise. — https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
- CMU 10-714/714 Deep Learning Systems, HW1 Questions 1-2, 2023 — implement forward `compute()` and
  backward `gradient()` for ops (PowerScalar, EWiseDiv, ReLU, Log, Exp, Transpose, ...) as paired
  methods, the general pattern this section's clamp exercise specializes. — https://github.com/dlsyscourse/hw1
- CMU 11-785 Introduction to Deep Learning, HW1P1 §6 "Activation Functions," Fall 2023 — hand-derive
  and implement forward/backward for Sigmoid, Tanh, ReLU, GELU, Softmax: several more worked
  forward/backward pairs against which to check a custom gradient's boundary convention. — https://deeplearning.cs.cmu.edu/F23/document/homework/HW1/HW1P1_F23.pdf

**Proposed problem set** (7 problems):
1. [short-code] **RMSNorm Learned Bias.** Add an optional learned bias to `RMSNorm`; verify the model's
   state grows by the expected entry and that state saved without the bias fails a strict reload.
   *Provenance:* original (book's existing ex1, kept).
1. [short-code] **Dropout From Scratch.** Implement `Dropout` as a custom layer that zeroes entries
   with probability $p$ and rescales survivors in training only; trace where the training flag your
   `forward` consults lives and how an enclosing container's eval mode reaches it.
   *Provenance:* original (book's existing ex2, kept).
1. [short-code] **Clamp With Custom Gradient.** Implement a clamp whose custom backward passes
   gradient only strictly inside the bounds; compare against the native clamp's gradient at, inside,
   and on the boundary, and name the native convention.
   *Provenance:* original (book's existing ex3, kept).
1. [conceptual] **Parameterless Layer's Value.** Write a layer returning the leading half of the
   Fourier coefficients of its input; since it registers nothing, explain what wrapping it in a
   module still buys over calling the transform inline.
   *Provenance:* original (book's existing ex4, kept).
1. [short-code] **Gradient-Check a Custom Activation.** Implement a custom op for GELU or Softplus
   (either) with its own hand-derived backward; gradient-check it numerically against the framework's
   automatic derivative of an equivalent built-in expression and report the largest discrepancy.
   *Provenance:* adapted from PyTorch's autograd-Function tutorial + JAX's custom-derivative-rules
   notebook (overlap med).
1. [conceptual] **Why the Clamp Needs No Second Derivative.** For the clamp exercise (ex3), work out
   on paper what the *second* derivative would be at the clamp boundary, and explain why a custom
   `Function`/`custom_vjp` sidesteps ever forming it, whereas relying on automatic double-backward
   through a boundary-heavy composition would not.
   *Provenance:* inspired by CMU 11-785 HW1P1's activation-backward derivations (overlap low).
1. [extended] **Straight-Through Estimator Layer.** Implement a custom layer whose forward output is a
   non-differentiable integer bucket index from a learned score, paired with a straight-through
   custom backward that passes the incoming gradient unchanged to the pre-bucketing score; verify the
   model's loss still decreases despite the forward being locally flat almost everywhere.
   *Provenance:* original.

---

## chapter_builders-guide/numerics.md — Numerics: Dtypes and Mixed Precision

**Topic:** floating-point dtypes, mixed-precision training with fp32 master weights, loss scaling,
and normalization's fp32 requirement.
**Current exercises:** 3; disposition: keep 3, rewrite 0, drop 0 — concrete, checkable deliverables
(crossover point, finfo fields, RMSNorm overflow), no defects.

**External sources found:**
- PyTorch official recipe, "Automatic Mixed Precision" — its own suggested variation is "vary
  participating sizes and see how the mixed-precision speedup changes," essentially this section's
  ex2 crossover search. — https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- fast.ai, Practical Deep Learning for Coders Part 2, Lesson 20 "Mixed Precision," 2022 — builds a
  mixed-precision training callback around HuggingFace Accelerate rather than raw
  `autocast`/`GradScaler`, the same fp16/fp32 split motivation from a training-loop-engineering
  angle. — https://course.fast.ai/Lessons/lesson20.html
- No good external exercise tradition found for dtype `finfo`/`e4m3fn`-field literacy (ex3) or for
  localizing exactly which intermediate quantity overflows inside a normalization layer under fp16
  (ex4): vendor references (e.g. NVIDIA's mixed-precision guide, already in this chapter's Further
  Reading) explain these narratively but do not pose them as problems — as far as we found, this
  section's own exercises are the only problem-set treatment of either.

**Proposed problem set** (6 problems):
1. [conceptual] **Mixed-Precision Memory Arithmetic.** Redo the fp32 memory arithmetic for
   mixed-precision Adam (fp32 master weights/gradients/moments, bf16 activations); identify the
   dominating term and compare the total against all-fp32 training.
   *Provenance:* original (book's existing ex1, kept).
1. [short-code] **fp32-Versus-16-Bit Crossover.** Time fp32 and 16-bit runs while shrinking hidden
   width and batch size; find a model small enough that mixed precision is slower, and explain the
   crossover.
   *Provenance:* original (book's existing ex2, kept).
1. [short-code] **Comparing finfo Fields.** Print every field of the fp8 e4m3fn dtype and compare with
   fp16 and bf16; explain the name and why its `max` is 448 rather than what the exponent bits alone
   suggest.
   *Provenance:* original (book's existing ex3, kept).
1. [short-code] **RMSNorm Under fp16 Overflow.** Feed `RMSNorm` inputs with std around 100 under
   forced fp16 computation; identify which intermediate quantity fails first.
   *Provenance:* original (book's existing ex4, kept).
1. [short-code] **Batch-Size Speedup Sweep.** Holding model size fixed, sweep batch size from 8 to
   4096 and plot mixed-precision speedup versus batch size; identify the batch size below which
   mixed precision stops paying off.
   *Provenance:* adapted from the PyTorch AMP recipe's own suggested variation (overlap high).
1. [conceptual] **Loss-Scaling Failure Sketch.** Sketch, in prose or pseudocode, what a training loop
   must do differently under loss scaling when a NaN/Inf gradient is detected after unscaling — what
   is skipped, what is not updated, how the scale factor adapts — then check the sketch against the
   section's own loss-scaling description.
   *Provenance:* inspired by fast.ai Lesson 20's callback-based framing of mixed precision (overlap low).

---

## chapter_builders-guide/saving-loading.md — Saving, Loading, and Pretrained Weights

**Topic:** code versus state versus training state, atomic checkpoint writes, the safetensors format,
dtype-cast checkpoints, and partial/lenient loading.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — all verified, concrete
measurement/comparison tasks with no defects.

**External sources found:**
- PyTorch official tutorial, "Saving and Loading Models," "Warmstarting Model Using Parameters from a
  Different Model" — built around `load_state_dict(..., strict=False)`, warning that silently-ignored
  mismatches "can mask genuine errors," the same partial-load caution this section's
  `allow_missing`/`ignore_extra` discussion raises. — https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- Hugging Face safetensors documentation — documents the save/load API (`save_file`, `safe_open`,
  `get_slice` for loading part of a tensor) underlying this section's header-inspection exercise
  (ex2). — https://huggingface.co/docs/safetensors/index
- Google Orbax (JAX-ecosystem checkpoint library) README/quickstart — an `ocp.save`/`ocp.load`
  round trip on an arbitrary pytree is the closest external analogue to this section's JAX-tab
  checkpointing code, though it does not itself pose the atomic-write or dtype-casting questions. — https://github.com/google/orbax
- No good external exercise tradition found for weight averaging across checkpoints (ex4) as a posed
  problem: it's a widely used production technique but we found no course/tutorial turning "average
  two of your own checkpoints and evaluate" into an exercise.

**Proposed problem set** (7 problems):
1. [conceptual] **Why Checkpoint, and Why Atomically.** Name two reasons to checkpoint even without
   deploying elsewhere; then, for a non-atomic write, describe the failure a mid-write crash causes
   and why the atomic version avoids it.
   *Provenance:* original (book's existing ex1, kept).
1. [short-code] **Reading the Safetensors Header.** Read the first 8 bytes of a saved safetensors file
   as a little-endian integer; measure the JSON header size for the MLP and how it grows if the
   hidden width doubles.
   *Provenance:* original (book's existing ex2, kept).
1. [short-code] **bfloat16 Round Trip.** Save parameters cast to bfloat16 and load into a float32
   model; identify what is lost and whether that is acceptable for inference versus resuming
   training.
   *Provenance:* original (book's existing ex3, kept).
1. [short-code] **Averaging Two Checkpoints.** Take two checkpoints 50 steps apart, average their
   weight tensors into a third parameter set, and evaluate it.
   *Provenance:* original (book's existing ex4, kept).
1. [short-code] **Partial Load Onto a Bigger Model.** Load a saved checkpoint with `strict=False`
   (or the equivalent lenient flags) into a model with one extra layer the checkpoint doesn't cover;
   verify which parameters kept their fresh initialization and which loaded from file, and state the
   concrete error `strict=True` would have raised instead.
   *Provenance:* adapted from PyTorch's "Warmstarting" tutorial section (overlap high).
1. [conceptual] **Why Safetensors Permits Partial Reads.** Explain why safetensors' header-then-flat-
   bytes layout lets a reader load a single named tensor without reading the rest of the file, and
   contrast this with why a pickle-based checkpoint cannot offer the same guarantee without
   deserializing the whole object graph.
   *Provenance:* adapted from safetensors' `get_slice` partial-loading API (overlap med).
1. [extended] **Crash-Safe Checkpoint Manifest.** Build a utility that writes model state, optimizer
   state, and a JSON metadata file atomically as three files plus one final manifest rename; simulate
   a crash mid-write (e.g. truncate one file) and show reloading detects and rejects the incomplete
   checkpoint rather than silently loading stale or corrupt state.
   *Provenance:* original.

---

## chapter_builders-guide/gpus-devices-memory.md — GPUs, Devices, and Memory

**Topic:** device placement, what occupies GPU training memory and how checkpointing trades compute
for memory, and asynchronous execution / synchronization points.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — well-grounded and, where relevant,
self-gated (`num_gpus() >= 2`).

**External sources found:**
- CMU 10-714/714 Deep Learning Systems, HW3 "Building an NDArray Library," 2023 — students write
  their own CPU and CUDA backends (Compact, elementwise/reduction kernels, tiled matmul) from scratch,
  the closest external analogue to "what actually executes on a device," though it stops short of
  memory/precision accounting. — https://github.com/dlsyscourse/hw3
- PyTorch official docs, "CUDA semantics" — states that `.item()`, `.cpu()`, and printing are
  synchronization points and that correct GPU timing needs `torch.cuda.synchronize()` or
  `torch.cuda.Event`, exactly the mechanism this section's ex3 asks students to explain. — https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch official docs, `torch.utils.checkpoint` — documents activation checkpointing as trading
  compute for memory by discarding and recomputing intermediate activations, the same tradeoff this
  section's ex2 pushes to its out-of-memory limit. — https://docs.pytorch.org/docs/stable/checkpoint.html
- No good external exercise tradition found for peak-memory *prediction* against a hand-built
  accounting model (ex1) or for measuring real multi-GPU scaling deviation from linear (ex4): these
  are usually taught via profiler narrative or production benchmarking suites, not textbook problems.

**Proposed problem set** (7 problems):
1. [conceptual] **Predicting Peak Memory.** Using this section's accounting model, predict peak
   memory of the four-plateau cell at batch sizes 64/256/1024, then measure with the framework's
   peak-memory counter; identify where the prediction breaks down and what it omitted.
   *Provenance:* original (book's existing ex1, kept).
1. [short-code] **Checkpointing's Batch-Size Ceiling.** Increase batch size in the checkpointing
   comparison until the non-checkpointed run runs out of memory; measure how much further the
   checkpointed run goes, and explain the ratio from what each variant stores.
   *Provenance:* original (book's existing ex2, kept).
1. [short-code] **Per-Step Print's Synchronization Cost.** Time one epoch printing loss every step,
   and again printing once per epoch; explain the difference via synchronization points.
   *Provenance:* original (book's existing ex3, kept).
1. [short-code] **Two-GPU Scaling.** (Gated on `num_gpus() >= 2`.) Time 1000 matrix products on one
   GPU versus 500 on each of two GPUs from the same loop; measure the scaling and identify sources of
   deviation from linear.
   *Provenance:* original (book's existing ex4, kept).
1. [short-code] **Transfer-Versus-Compute Crossover.** Using only the section's own device/timing
   APIs, measure the cost of moving a tensor CPU->GPU->CPU versus performing an equivalent-FLOP
   computation on-device, and find the size at which transfer first exceeds compute.
   *Provenance:* inspired by dlsyscourse HW3's device-backend framing (overlap low; only the framing
   transfers, not the code).
1. [conceptual] **Auditing Every Sync Point.** List every synchronization point in the section's own
   training loop, not just `.item()`/print; classify each as unavoidable for a correct loop or an
   avoidable instrumentation artifact, and justify each classification.
   *Provenance:* adapted from PyTorch's CUDA semantics synchronization guidance (overlap high).
1. [extended] **Checkpointing-Granularity Pareto Frontier.** Reproduce the four-plateau memory
   breakdown with activation checkpointing applied at whole-model, per-block, and no granularity;
   plot the resulting memory/time Pareto frontier.
   *Provenance:* original.

---

## chapter_builders-guide/reproducibility-inspection.md — Reproducibility and Inspection

**Topic:** inventorying nondeterminism sources and determinism controls, and inspecting a model via
hooks without changing its computation.
**Current exercises:** 5 (4 shared + a 3-framework-only tabbed ex5); disposition: keep 3, rewrite 2,
drop 0 — ex4's premise (framework-agnostic forward-hook output replacement) is contradicted by the
section's own text for TensorFlow (no equivalent) and MXNet (observe-only), a genuine content bug per
the prior style review; ex5's tabbed block silently omits a PyTorch variant, unlike every other
tabbed group in this chapter. Both are rewritten together into one framework-honest problem rather
than dropped, since the underlying ablation idea is sound.

**External sources found:**
- PyTorch official docs, "Multi-process data loading," §"Randomness in multi-process data loading" —
  states that other libraries' seeds "may be duplicated upon initializing workers, causing each
  worker to return identical random numbers," and recommends reseeding from
  `get_worker_info().seed` inside `worker_init_fn` — close to verbatim the bug ex1 asks students to
  diagnose and fix. — https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
- fast.ai, Practical Deep Learning for Coders Part 2, Lesson 16 "The Learner framework" (HooksCallback
  / ActivationStats), 2022 — builds a callback attaching forward hooks across all layers to track
  per-layer activation mean/std/histograms for diagnosing training instability, the same idea as
  ex2/ex4. — https://course.fast.ai/Lessons/lesson16.html
- PyTorch official notes, "Reproducibility" — already cited in this chapter's own Further Reading as
  the standard reference on determinism flags and their costs/limits; narrative rather than
  exercise-posing, so it supports but does not itself supply a problem. — https://docs.pytorch.org/docs/stable/notes/randomness.html
- No good external exercise tradition found for backward-hook-based gradient-norm inspection (ex3)
  specifically as a posed problem, beyond the framework's own hook API reference.

**Proposed problem set** (7 problems):
1. [short-code] **Per-Worker Seeding Fix.** Extend `train_once` to load data through a parallel
   loader; give the dataset its own RNG, check whether workers copied identical generator state on a
   process-based loader, then replace it with one derived from the worker's own seed and verify
   agreement across two runs.
   *Provenance:* original (book's existing ex1, kept).
1. [short-code] **FLOP-Counting Forward Hook.** Write a forward hook counting multiply-accumulate
   operations for every linear layer from its input/weight shapes; report per-layer and total FLOPs
   for the residual stack and check against a hand count.
   *Provenance:* original (book's existing ex2, kept).
1. [conceptual] **Gradient Norms: Activation Versus Parameter.** Using a backward hook, record each
   block's output-gradient norm; compare against parameter-gradient norms in the same block and
   identify which first reveals an exploding backward signal.
   *Provenance:* original (book's existing ex3, kept).
1. [short-code] **Output-Replacing Hooks, Where They Exist.** For PyTorch and JAX, where a forward
   hook's/`sow`'s return value can replace or feed into a module's output, zero out a single residual
   block's body to turn it into the identity and measure the output change. For TensorFlow and MXNet,
   where the section's own text says hooks cannot modify or can only observe outputs, achieve the same
   ablation by editing the block's `call`/`forward` directly, and explain in one sentence why the
   *mechanism* differs even though the *experiment* is identical.
   *Provenance:* adapted from the book's own ex4/ex5 (overlap high; corrects the cross-framework
   premise and supplies the previously-missing PyTorch case rather than introducing new material).
1. [conceptual] **Predict What a Seed Does and Doesn't Fix.** Before touching code, predict which of
   the following a single top-level random seed makes reproducible: single-process data order,
   multi-worker data order, NumPy-based augmentation inside `__getitem__`, and nondeterministic GPU
   reduction order; then verify each prediction against the section's own seeding demo.
   *Provenance:* adapted from PyTorch's DataLoader worker-seeding documentation (overlap high).
1. [short-code] **NaN-Onset Training Doctor.** Extend the FLOP-counting hook into a lightweight
   monitor that also records, per step, whether any layer's activation contains a NaN/Inf; train
   deliberately with a learning rate 100x too large and report the first layer (by depth) where a NaN
   appears.
   *Provenance:* inspired by fast.ai's ActivationStats callback (overlap low).
1. [extended] **Comparing the Four Inspection Mechanisms.** On the section's own residual stack,
   compare PyTorch mutable hooks, JAX `nnx.sow`/`capture`, a TensorFlow functional-inspection
   submodel, and MXNet's observe-only hooks (or `Block.summary()`) on three axes: whether it requires
   owning the model's source, whether it can retain state/tensors, and whether it works on a model
   you did not write. Produce a short table and justify each cell from what you actually observed.
   *Provenance:* original.
