# Chapter 5 style review: Computation / Builders' Guide

## Scope

Reviewed all 19 tracked Markdown sources under `chapter_builders-guide`.

Rendered chapter sources: `index.md`, `model-construction.md`, `parameters-state-memory.md`, `init.md`, `custom-layers.md`, `numerics.md`, `saving-loading.md`, `gpus-devices-memory.md`, and `reproducibility-inspection.md`.

Tracked planning/archive sources: `_outline/index.md`, `_outline/model-construction.md`, `_outline/parameters-state-memory.md`, `_outline/init.md`, `_outline/custom-layers.md`, `_outline/numerics.md`, `_outline/saving-loading.md`, `_outline/gpus-devices-memory.md`, `_outline/reproducibility-inspection.md`, and `_outline/promotion-notes.md`.

The review covers prose, headings, equations, captions, code/experiment explanations, summaries, exercises, and planning notes. There are no slide blocks in any of these tracked sources. Line references are current.

## Executive assessment

The rendered chapter is substantially stronger than a conventional framework catalogue. It repeatedly builds a framework-neutral concept—module hierarchy, state, initialization, dtype, checkpoint, device—and then identifies the precise API consequences in each backend. The state/code/training-state distinction in `saving-loading.md`, the training-only fit of preprocessing ideas carried into state management, and the distinction between bitwise and statistical reproducibility are especially good. Captions are generally self-contained and technically useful.

The price of this completeness is scale and repetition. Several sections exceed what one recoverable question can support, and four framework tabs often repeat nearly identical prose. `gpus-devices-memory.md` alone covers placement, allocators, memory accounting, checkpointing, asynchronous timing, pinned transfer, and trainer integration. `reproducibility-inspection.md` combines control of random state with hook-based debugging, two different reader problems. More seriously, `model-construction.md` contains backend-name copy errors and oversimplifies Python control flow under compilation. The tracked `_outline` directory still describes the chapter as a proposal, contains unresolved decisions and `[UNVERIFIED]` items, and coexists with a “completed” promotion archive that lists pending GPU verification. Although these files are not rendered, they are now an unreliable maintenance reference.

## Scores

| Dimension | Score | Basis |
|---|---:|---|
| Writing quality | 7.8/10 | Direct, technical prose and good openings, reduced by four-way duplication, very long sections, false-intimacy phrases, and planning artifacts. |
| Explanation quality | 8.2/10 | Strong concept-to-API mapping and operational caveats; some sections answer several questions and bury their central procedure. |
| Technical quality | 7.9/10 | Many unusually careful security, state, and determinism qualifications; backend copy errors, compiled-control-flow simplifications, unstable version claims, and unresolved GPU verification require attention. |

## Chapter architecture and logical order

The opening roadmap (`chapter_builders-guide/index.md:18-29`) is explicit, but the order can better follow the state lifecycle. Once modules and state are defined, readers need initialization, saving/loading, device placement/numerics, and reproducibility before custom extensions. Currently saving/loading appears after custom layers and numerics, while reproducibility is last even though every experiment in the chapter depends on controlled randomness and trustworthy measurement.

A stronger dependency order is: modules/construction -> parameters and persistent state -> initialization -> saving/loading/checkpoint identity -> devices and dtypes -> custom layers -> reproducibility and inspection. Reproducibility could also be split: put random-state/determinism immediately after initialization/checkpointing and keep model inspection after custom modules.

At section level, `model-construction.md`, `parameters-state-memory.md`, `gpus-devices-memory.md`, and `reproducibility-inspection.md` should be divided or sharply compressed. Their concepts are related, but each currently carries multiple independent operational questions.

## Section- and file-level issues

| ID | Severity | Evidence | Violated style-guide rule | Diagnosis | Concrete revision direction |
|---|---|---|---|---|---|
| C5-01 | Medium | `chapter_builders-guide/index.md:18-29`: saving/loading is sixth and reproducibility/inspection eighth | Order by dependency and the lifecycle of the objects being taught. | State is defined early, but how it persists and how experiments remain repeatable is deferred until after many stateful demonstrations. | Move saving/loading directly after state/initialization. Split random-state reproducibility from hooks and introduce it before experiments whose outputs are compared. |
| C5-02 | High | `chapter_builders-guide/model-construction.md:1580-1613`: four near-identical config paragraphs; similar repetition throughout the chapter | Explain the shared concept once; tabs should isolate genuine API differences. | Framework parity is achieved by duplicating prose, making sections several times longer and obscuring the invariant. | Put the framework-neutral claim before tabs. Use a compact comparison table for construction/registration/serialization semantics and reserve tabs for executable code and exceptions. |
| C5-03 | High | `chapter_builders-guide/model-construction.md:382-405`: PyTorch and MXNet prose both say “`nnx.Sequential`” | Maintain technical names exactly across backend variants; audit copied tabs. | `nnx.Sequential` is the JAX/Flax name, not PyTorch's `nn.Sequential` or MXNet Gluon's `nn.Sequential`. This is a concrete copy artifact in reader-facing prose. | Correct both names and run a backend-token audit over every tab (`nnx`, `torch`, `tf`, `gluon`, device APIs) to catch cross-tab leakage. |
| C5-04 | High | `chapter_builders-guide/model-construction.md:749-777`: forward methods are “ordinary Python” and may branch/loop freely; compilation constraints are only partly noted | State execution assumptions and limitations before operational claims. | Python syntax is available, but data-dependent control flow, tracing, graph capture, JIT recompilation, and side effects differ materially across PyTorch, JAX, TensorFlow, and MXNet. “Nothing restricts it” is false once compiled. | Separate eager authoring syntax from compiled semantics. For each backend, distinguish static Python control flow from tensor/data-dependent control flow and state which constructs trigger tracing, recompilation, graph breaks, or errors. |
| C5-05 | Medium | `chapter_builders-guide/model-construction.md:1580-1613`: “the config is all you need to reconstruct” | Qualify sufficiency claims and name hidden dependencies. | Reconstruction also requires the model-building code, library versions, registered custom classes, and sometimes RNG/state schema. A config alone is not executable architecture. | Say that config plus versioned construction code determines the module graph; checkpoints should record schema/code version and migration expectations. |
| C5-06 | Medium | `chapter_builders-guide/parameters-state-memory.md:688-703`: “For fp32 ... 16 bytes per parameter” | State accounting assumptions beside formulas and distinguish lower bounds from measured memory. | The table is a useful base calculation, but it can be mistaken for total training memory. It excludes activations, temporary workspaces, allocator fragmentation, mixed-precision copies, EMA, and sharding effects. There is also a visible edit error at `:694-695` (“so in / For fp32”). | Label it “idealized persistent parameter-state bytes for dense Adam fp32.” Fix the sentence and point immediately to the measured allocator section for total peak memory. |
| C5-07 | Medium | `chapter_builders-guide/init.md:48-51`: “You can usually ignore initialization because the default is sensible.” | Avoid broad reassurance that conflicts with the section's conditions. | The section itself explains that depth, residual branches, custom tensors, activation choice, and paper reproduction require explicit initialization. | State the bounded claim: defaults are suitable for standard library layers in ordinary depths; inspect or override them when architecture or reproduction assumptions differ. |
| C5-08 | Low | `chapter_builders-guide/init.md:557-560`: after showing truncated variance changes, “practice ignores the difference” | Specify who/when and preserve consequences of the calculation. | The sentence dismisses a measurable variance change without saying when it is negligible or when fan-aware rescaling restores it. | State that fixed-std recipes such as BERT accept the reduced realized variance, while variance-scaling initializers compensate for truncation. |
| C5-09 | Medium | `chapter_builders-guide/custom-layers.md:612-616`: “build one to understand it, then use the native implementation in production” | Avoid universal prescriptions; state decision criteria. | Native layers may be faster and maintained, but custom semantics, research modifications, export requirements, or absent kernels can justify production custom code. | Recommend the native implementation when semantics match and verify equivalence, performance, gradients, serialization, and export for custom alternatives. |
| C5-10 | High | `chapter_builders-guide/numerics.md:175-186`: “bf16 became the default 16-bit format for training” | Scope hardware/software claims and avoid presenting ecosystem trends as universal facts. | fp16, bf16, tf32, and newer formats depend on accelerator generation, framework policy, workload, and deployment target. There is no single default across training systems. | Say bf16 is often preferred on hardware with native support because of its exponent range. Add the hardware/API version context for each recipe. |
| C5-11 | Medium | `chapter_builders-guide/numerics.md:188-204`, `:560-585`, summaries `:1278-1290` | Temporally unstable API behavior must be versioned and separated from the durable principle. | Default matmul precision, autocast eligibility, master-weight handling, and scaler behavior change across releases and devices. The chapter makes exact API-policy claims without a reader-visible tested-version box. | Add a compact environment/version table and label policy lists as current-library behavior. Keep the durable storage/compute/accumulation distinction in the main argument. |
| C5-12 | Low | `chapter_builders-guide/saving-loading.md:482-497`: safetensors “is what you use to hand a model to anyone else” | Avoid universal tool prescriptions; define the requirement. | The security argument is strong, but safetensors is one safe data-only interchange format, and framework/package compatibility can dictate alternatives. | Recommend a non-executable, documented, checksum-verified format; present safetensors as the chapter's chosen cross-framework example. |
| C5-13 | High | `chapter_builders-guide/gpus-devices-memory.md:9-16`, headings at `:680`, `:1037`, `:1298`, and trainer integration through `:1808` | One section should answer one main question; avoid packing a full systems chapter into one section. | Placement errors, allocator accounting, activation checkpointing, asynchronous timing, data transfer, and trainer design are independent operational problems. The nearly 1,900-line section makes its procedure difficult to reconstruct. | Split into “Devices and placement,” “Training-memory accounting and checkpointing,” and “Asynchronous execution and input transfer,” or move the latter two to computational performance. |
| C5-14 | Low | `chapter_builders-guide/gpus-devices-memory.md:683`, `:698`, `:715`, `:733`: “every ... user hits in their first week” repeated four times | Avoid false intimacy and duplicated rhetorical templates. | The phrase predicts reader experience and is repeated solely to maintain tab symmetry. | State the observable puzzle once outside tabs: process memory remains reserved after tensors die. Then compare allocator behavior in a table. |
| C5-15 | Medium | `chapter_builders-guide/gpus-devices-memory.md:1039-1055`, summaries `:1818-1856`: checkpointing costs “30–40%” / “roughly 1.3x” | Performance ratios require measured conditions and should not be universalized in captions/summaries. | Recompute overhead varies with segment size, operator mix, memory pressure, compiler, RNG handling, and device. The asymptotic memory argument does not imply a fixed runtime ratio. | Keep the `O(sqrt(N))` schedule under its equal-layer assumptions. Label 1.3x as an illustrative measurement/citation range and report the local benchmark's configuration. |
| C5-16 | Medium | `chapter_builders-guide/gpus-devices-memory.md:1298-1348`: four long paragraphs repeat the same queueing explanation | Shared mechanism once, backend differences second. | The repeated prose hides the actual differences: CPU behavior, synchronization primitive, and graph/eager dispatch. | Explain asynchronous dispatch once with the queue figure, then use a four-row table for backend scope and synchronization API before the code tabs. |
| C5-17 | High | `chapter_builders-guide/reproducibility-inspection.md:9-16`: one section explicitly covers random control and hook-based observation | One section should answer one recoverable question; preserve conceptual dependency. | Repeating an experiment and diagnosing an internal activation are related debugging activities but require different concepts, APIs, and conclusions. | Split into “Randomness and reproducibility” and “Inspecting model execution.” Let checkpoint RNG state link the former to saving/loading and module call wrappers link the latter to custom layers. |
| C5-18 | High | `chapter_builders-guide/reproducibility-inspection.md:1083-1089`: “In JAX the inventory collapses to one item, the key you pass” | State system boundaries; do not equate one library's PRNG discipline with a whole program. | JAX operations use explicit keys, but Python `random`, NumPy, data-loader libraries, host callbacks, external services, and nondeterministic kernels remain outside that key. The statement conflicts with the chapter's own inventory principle. | Say the *JAX PRNG portion* has explicit key state. Retain a program-level inventory for host preprocessing, loaders, and non-JAX dependencies. |
| C5-19 | High | `_outline/index.md:1-9` still says “v2 proposal” and “outline for review”; `_outline/numerics.md:121-128` contains “decision needed” and `[UNVERIFIED on GPU]`; `_outline/promotion-notes.md:1-18` says promotion is completed but lists pending GPU checks | Tracked documentation should have one authoritative state; remove stale generated/planning artifacts or label their lifecycle unambiguously. | The outline and archive disagree about whether the rewrite is proposed, completed, verified, or pending. Maintainers can no longer know whether unverified items were resolved in the rendered sources. | Archive the proposal outside the live chapter or update it to a dated design record. Resolve every verification item in a machine-readable matrix that names environment, command, result, and date; remove editorial markers from active references. |

## Mathematics and notation

- The state/memory arithmetic should be explicitly labeled as idealized persistent state. Use symbols for parameter count, bytes per dtype, optimizer slots, and activation batch scaling before mapping to framework counters.
- Initialization distinguishes fan-aware variance, truncation, residual-depth scaling, and zeroing the last residual projection. The independent-contribution approximation is stated in the residual-stream caption (`init.md:583`); keep that condition visible in experiment conclusions.
- Numerics correctly distinguishes range from precision and storage dtype from compute policy. Make accumulation dtype a third explicit axis wherever “mixed precision” is defined.
- Activation-checkpointing complexity assumes equally sized layers and a particular segment schedule. Do not combine that asymptotic statement with a fixed runtime multiplier.
- Reproducibility should distinguish pseudorandom stream identity, deterministic kernel selection, floating-point equivalence, and statistical replication. The prose at `reproducibility-inspection.md:592-601` does this very well and should become the section's organizing frame.

## Figures, captions, and slides

There are no slide blocks in the 19 tracked Markdown files, so no slide-level defects were found. This is worth preserving until a separate, deliberate deck is authored; auto-fragmenting these already large sections would make them less coherent.

The module-tree, checkpoint-contents, allocator, and activation-checkpoint figures are structurally useful. Their captions usually identify objects and relations without needing nearby prose. The activation-checkpoint caption should remove the fixed “roughly 1.3x” cost or label its source/conditions. The module-tree caption could mention shared-child aliases because the prose immediately qualifies the tree into an object graph.

## Code and experiment pedagogy

- `saving-loading.md` provides the strongest operational sequence: define state, demonstrate unsafe serialization, introduce a data-only interchange format, build a resumable checkpoint, then adapt foreign weights. Preserve it and move it earlier.
- Security guidance is concrete and actionable. Keep `weights_only=True`, atomic replacement, and explicit missing/unexpected-key diagnostics.
- Framework comparisons should start from one expected invariant and then show only differing API code. Many current tabs repeat intent and interpretation four times.
- GPU timing and memory demonstrations must name hardware, framework/CUDA versions, warm-up, synchronization, and allocator reset conditions. CPU fallbacks verify syntax but do not validate GPU performance claims.
- Initialization and dtype experiments should print the expectation being tested and compare observed values with tolerance, rather than only emit raw output.
- Reproducibility examples should culminate in a repeated-seed result table (mean and spread), not only show that one fixed seed repeats bitwise.
- The outline files record unverified GPU paths. Those checks should be resolved before treating guarded CPU cells as evidence about CUDA behavior.

## Recurring artifacts

- Four-way near-duplicate paragraphs surrounding code tabs.
- Universal reassurance/prescription: “usually ignore,” “what you use,” “every user,” “all you need.”
- Backend-copy leakage (`nnx` in PyTorch/MXNet prose).
- Exact performance ratios without visible environment or repeated measurement.
- Very long sections that accumulate related but independently actionable problems.
- Stale editorial metadata: `[NEW]`, `[MOD]`, `SKIP`, “decision needed,” and `[UNVERIFIED]` in tracked outlines.

## Positive patterns to preserve

- `model-construction.md` begins from repeated blocks in actual ResNet/Transformer-scale models and defines a recursive module hierarchy.
- `parameters-state-memory.md` distinguishes parameters, buffers, optimizer state, tied aliases, and freezing rather than treating “weights” as one undifferentiated object.
- `saving-loading.md` clearly separates code, model state, and full training state, then connects each checkpoint compartment to what it restores.
- The serialization security discussion explains *why* pickle is dangerous and provides a data-only alternative.
- `gpus-devices-memory.md` distinguishes live allocation from allocator reservation and external process reporting.
- `reproducibility-inspection.md:592-601` draws the essential line between bitwise reproducibility for debugging and statistical reproducibility for science.
- Captions are generally self-contained and code is tied to real failure modes.

## Prioritized revision plan

1. Correct the cross-backend naming errors in `model-construction.md` and perform a systematic token/API audit across all tabs.
2. Split or compress the oversized model-construction, GPU, and reproducibility sections; reorder saving/loading and random-state material by lifecycle dependency.
3. Rewrite shared framework concepts once, replacing duplicated prose with comparison tables and focused code tabs.
4. Qualify compiled control flow, mixed-precision defaults, memory formulas, checkpointing overhead, and JAX program-level reproducibility.
5. Resolve every GPU/version verification item and attach tested environments to temporally unstable systems claims.
6. Archive or reconcile the `_outline` proposal and promotion notes so tracked planning sources have one unambiguous status.
7. Run a final line edit for universal prescriptions, false-intimacy templates, and summary claims stronger than the body.

**Issue count: 19 total (8 high, 8 medium, 3 low).**
