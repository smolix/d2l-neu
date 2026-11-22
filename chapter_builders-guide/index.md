# Computation
:label:`chap_computation`

Up to now we have relied on frameworks to manage model construction, parameters,
initialization, and device execution behind the scenes. Moving from textbook
prototypes to production-scale systems requires opening that black box. Training
an 8-billion-parameter language model in 32-bit floating point with Adam
demands 128 GB of parameter, gradient, and optimizer state before storing a
single activation, and runs across thousands of GPUs where hardware failures
interrupt training every few hours. At that scale, understanding how frameworks
represent, initialize, compute with, and persist models is essential.

Everything in this chapter builds on one central data structure: a neural
network is a *tree of modules* whose leaves hold its *state*—trainable
parameters and fixed buffers alike—each addressed by its path from the root.
Every framework operation walks that same tree:

* :numref:`sec_model_construction` shows how modules register child modules into
  a hierarchy, why hiding layers in plain Python lists silently drops them from
  training, how residual blocks and compiled forward passes work, how lazy
  layers infer shapes on the fly, and how to separate architecture configs from
  builder code.
* :numref:`sec_parameters` opens the state tree to distinguish trainable
  parameters from non-trainable buffers, derives the 16-byte-per-parameter
  memory floor for fp32 Adam training, counts parameters abstractly without
  allocating memory, and covers weight tying and parameter freezing.
* :numref:`sec_init_param` moves from single-layer initialization to whole-tree
  and residual-network initialization, showing why unscaled residual blocks blow
  up activation variance by $(1 + g^2)^N$ and how scaling output projections by
  $1/\sqrt{N}$ or zeroing them keeps deep stacks stable.
* :numref:`sec_custom_layer` shows how to extend a framework with stateless
  layers, parameterized layers (`RMSNorm`), layers holding fixed buffers
  (`RandomFeatures`), and custom backward passes (the *straight-through
  estimator* for non-differentiable rounding).
* :numref:`sec_numerics` compares fp32, fp16, and bf16, showing why 16-bit
  weights round away small updates and how mixed-precision training pairs fp32
  master weights with 16-bit compute (and dynamic loss scaling for fp16).
* :numref:`sec_read_write` covers safe weight serialization (`safetensors` vs.
  arbitrary code execution in `pickle`), crash-safe atomic training checkpoints
  that resume bit for bit, and auditing missing or unexpected keys when loading
  pretrained weights.
* :numref:`sec_use_gpu` places tensors and models on GPUs, explains asynchronous
  command queues and accurate GPU benchmarking, dissects live vs. reserved vs.
  driver memory, uses activation checkpointing to cut activation memory from
  $O(K)$ to $O(\sqrt{K})$, and wires GPU support into `d2l.Trainer`.
* :numref:`sec_repro` shows how to control pseudorandom seeds across workers and
  devices, why parallel GPU atomic additions are non-associative, and how to
  attach forward and backward hooks to inspect activation statistics, gradient
  norms, and the first layer that emits a NaN.

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. All are freely accessible online except where noted.

**Books**

- [Deep Learning with PyTorch — Stevens, Antiga & Viehmann](https://web.archive.org/web/20211012030609/https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) — free PDF from the PyTorch team (archived copy; the original pytorch.org link has gone away); Part 1 covers tensors, storage, autograd, and `nn.Module` at the same level as :numref:`sec_model_construction` and :numref:`sec_parameters`.
- [Machine Learning Systems — Vijay Janapa Reddi](https://mlsysbook.ai/) — free online; the systems view around this chapter: frameworks, data pipelines, training infrastructure, and efficient deployment (:numref:`sec_use_gpu`, :numref:`sec_repro`).

**Courses and video lectures**

- [CMU 10-414/714: Deep Learning Systems — Chen & Kolter](https://dlsyscourse.org/) — free lectures and assignments; students build "needle", a miniature framework with autograd, modules, initialization, and GPU support, an implementer's view of :numref:`sec_model_construction`, :numref:`sec_init_param`, and :numref:`sec_use_gpu`.
- [Neural Networks: Zero to Hero — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html) — free video series; builds autograd, modules, and training loops from scratch in plain Python, the abstractions behind :numref:`sec_model_construction` and :numref:`sec_custom_layer`.
- [fast.ai Part 2: Deep Learning Foundations — Howard et al.](https://course.fast.ai/Lessons/part2.html) — free; rebuilds a training framework from tensor operations upward, including modules, initialization, and mixed precision (:numref:`sec_model_construction`, :numref:`sec_init_param`, :numref:`sec_numerics`).

**Tutorials, notes, and interactive**

- [PyTorch internals — Edward Yang](http://blog.ezyang.com/2019/05/pytorch-internals/) — free; a guided tour of tensors, strides, dispatch, and autograd inside PyTorch, one level below :numref:`sec_model_construction`.
- [JAX — The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) — free; pure functions, explicit PRNG keys, and jit constraints — the functional worldview behind this book's JAX tab, stated as a list of gotchas (:numref:`sec_parameters`, :numref:`sec_repro`).
- [Train With Mixed Precision — NVIDIA](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — free; the vendor guide to fp16/bf16 arithmetic and loss scaling that :numref:`sec_numerics` distills.
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic — David Goldberg](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — free; the standard reference beneath every rounding and overflow issue in :numref:`sec_numerics`.
- [Making Deep Learning Go Brrrr From First Principles — Horace He](https://horace.io/brrr_intro.html) — free; a first-principles account of compute, memory, and framework overhead covered in :numref:`sec_use_gpu`.
- [safetensors — Hugging Face](https://huggingface.co/docs/safetensors/index) — free; the file format for sharing weights without executing code on load, and why it replaced pickle-based checkpoints (:numref:`sec_read_write`).
- [Reproducibility — PyTorch notes](https://pytorch.org/docs/stable/notes/randomness.html) — free; the determinism flags, their costs, and their limits, complementing :numref:`sec_repro`.

**Foundational papers**

- [Automatic Differentiation in Machine Learning: a Survey — Baydin, Pearlmutter, Radul & Siskind (2018), *JMLR*](https://jmlr.org/papers/v18/17-468.html) — free; the definitive account of forward- and reverse-mode autodiff, the algorithm behind every custom gradient in :numref:`sec_custom_layer`.
- [Mixed Precision Training — Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) — free; the origin of the fp16 + master-weights + loss-scaling recipe in :numref:`sec_numerics`.
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library — Paszke et al. (2019)](https://arxiv.org/abs/1912.01703) — free; the design rationale (eager execution, autograd, memory allocator) for the imperative style of :numref:`sec_model_construction` and :numref:`sec_use_gpu`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §5]{.kicker}

Deep Learning Computation: From Prototype Code to Production Systems<br>**module trees · parameter state · initialization · custom layers · numerics · checkpoints · GPUs · reproducibility**
:::
:::

::: {.slide title="Scaling From a 4-Array MLP to an 8B Foundation Model"}
[The Systems Challenge]{.kicker}

::: {.cols .vc}
::: {.col}
- **Memory capacity (`128 GB` floor):** Storing 8B weights takes `16 GB` in `bfloat16`, yet training with Adam in `float32` requires `128 GB` of parameters, gradients, and optimizer moments before computing a single activation.
- **Numerical stability & precision:** Unscaled residual branches blow up activation variance by $2^{32}$ across 32 blocks, while 16-bit weights silently round away small updates.
- **Fault tolerance, security & async GPUs:** Atomic checkpoints (`96 GB`), `safetensors` vs. `pickle` RCE, synced GPU timers, and non-invasive layer hooks.
:::

::: {.col .fig .big}
@fig:ch05-bg-four-bottlenecks
:::
:::
:::

::: {.slide title="The Unifying Abstraction: A Model Is a Tree of Modules"}
[Core Mental Model]{.kicker}

::: {.cols .vc}
::: {.col}
- **Structured path addressing:** Every parameter and non-trainable buffer is uniquely addressed by its path from the root module (e.g., block 1, layer 0, weight).
- **Unified tree traversals:** Optimizer setup, initialization, `safetensors` serialization, dtype casting, GPU placement, and hooks all walk registered children recursively.
- **The silent registration invariant:** Any layer or tensor stored in a plain Python attribute or list outside the module registry is invisible to optimizers, checkpoints, and GPUs.
:::

::: {.col .fig .big}
@fig:ch05-bg-module-tree-hub
:::
:::
:::

::: {.slide title="Chapter 5 Roadmap: Architecture, State, and Execution"}
[Architectural & Systems Roadmap]{.kicker}

::: {.cols .vc}
::: {.col}
- **Model structure & state (§§5.1–5.4):** Module trees and `__setattr__` registration, the $16\text{ B/param}$ `fp32` Adam floor, weight tying, $(1+g^2)^N$ residual variance scaling ($g = 1/\sqrt{N}$), `RMSNorm`, fixed buffers, and the Straight-Through Estimator.
- **Precision, I/O & hardware (§§5.5–5.8):** `fp32` master weights with `bf16`/`fp16` compute, zero-copy `safetensors` and 5-part atomic checkpoints, asynchronous GPU queues, $O(\sqrt{K})$ activation checkpointing, PRNG seeding, and forward/backward hooks.
:::

::: {.col .fig .big}
@fig:ch05-bg-systems-roadmap
:::
:::
:::

::: {.slide title="Concrete Running Benchmarks Across the Chapter"}
[Quantitative Anchors]{.kicker}

::: {.cols .vc}
::: {.col}
- **Parameter & memory scaling (§§5.1–5.2, §5.5):** Toy 4-block Residual MLP ($729{,}866$ params) vs. Llama 3 8B ($291$ tensors, `16 GB` `bf16` inference, `128 GB` `fp32` Adam training floor).
- **Signal stability & I/O (§5.3, §5.6):** $g = 1/\sqrt{N}$ caps 32-block std growth at $\sqrt{e} \approx 1.65\times$ (vs. $65{,}536\times$ unscaled); 5-part `96 GB` checkpoints resume bit for bit.
- **Activation memory (§5.7):** Segmenting $K = 16$ blocks by $s = \sqrt{K} = 4$ cuts peak step RAM from $1.1\text{ GiB} \to 0.45\text{ GiB}$.
:::

::: {.col .fig .big}
@fig:ch05-bg-8b-benchmarks
:::
:::
:::

::: {.slide title="How to Navigate This Chapter"}
[Study & Engineering Guide]{.kicker}

Each section combines runnable multi-framework code with systems diagnostics that prevent silent training failures:

- **Core path for all readers:** Read **§5.1–§5.3** to master module registration, parameter/buffer accounting, and residual variance scaling, then **§5.5–§5.7** for mixed precision, safe checkpointing, and GPU memory management.
- **Framework-specific idioms:** Pay close attention to how your framework handles list containers, lazy shapes, PRNG state, and graph compilation.
- **Diagnostic habits to adopt immediately:** Count parameter and activation bytes before launching a job, verify initial activation standard deviations with a single forward pass, and audit missing/unexpected keys whenever loading external weights.
:::
