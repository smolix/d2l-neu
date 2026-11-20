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

What happens to a model between its definition and a trained result<br>
**modules · state · initialization · custom layers · number formats · checkpoints · devices · reproducibility**
:::
:::

::: {.slide title="An 8-Billion-Parameter Model Has to Fit, Train, Survive, and Be Shared"}
[The problem]{.kicker}

- **Fit:** 16 GB of bf16 weights, but about 128 GB of state to train
  with Adam in fp32.
- **Train:** a bad initial scale stops a deep network from learning; weights
  stored in 16 bits lose their small updates without any error.
- **Survive:** long runs are interrupted; resuming needs the weights, the
  optimizer state, and the state of the random-number generators.
- **Share:** a file someone else wrote can run code when it is loaded.

::: {.d2l-note}
None of this is a new model. It is the software around every model, and
this chapter opens it up.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- build models from modules and configs, and find any parameter by its name;
- count a model's parameters and the bytes that training it needs;
- initialize deep and residual networks so that they train;
- write a layer or a gradient that the library lacks;
- train in mixed precision, and explain loss scaling;
- save a training run and resume it bit for bit, and load pretrained
  weights safely;
- place work on devices, time it correctly, and trade compute for memory;
- make runs repeatable, measure how much they differ, and look inside a
  running model.

**Running example:** will an 8B-parameter model fit, train, and resume?
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question | adds |
|---|---|---|
| Modules and Model Construction | what is a module, and how is a model assembled? | a model is a tree of modules, built from a config |
| Parameters, State, and Memory | what state does a model hold, and what does training cost? | parameters, buffers, bytes per parameter |
| Initialization | how is the state set before training? | library defaults, custom initializers, residual depth |
| Custom Layers and Functions | what if the library lacks a layer? | new layers, state, and custom gradients |
| Numerical Formats | which formats can a model compute in? | dtypes, mixed precision, loss scaling |
| Saving, Loading, and Pretrained Weights | how do we save, resume, and share? | safe files, checkpoints, pretrained weights |
| GPUs, Devices, and Memory | where does the computation run, and what fills memory? | placement, asynchrony, training memory |
| Reproducibility and Inspection | when do two runs agree, and how do we look inside? | seeds, determinism, hooks |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Modules and Model Construction | the MLP of the last chapter has 4 parameter arrays; an 8B model has 291 |
| Parameters, State, and Memory | 8B parameters: 32 GB in fp32, about 128 GB to train with Adam |
| Initialization | 32 residual blocks grow the activations' scale by about $2^{16}$; a scaled last layer in each body keeps it below $\sqrt{e}$ |
| Custom Layers and Functions | RMSNorm maps (3, 4) and (300, 400) to the same (0.849, 1.131) |
| Numerical Formats | $300^2$ overflows in fp16 but not in bf16; bf16 weights fall behind, fp32 master weights do not |
| Saving, Loading, and Pretrained Weights | a 96 GB checkpoint for 32 GB of fp32 weights; a run resumed bit for bit |
| GPUs, Devices, and Memory | activation checkpointing cuts a step from about 1.1 GiB to 0.4–0.5 GiB for about a quarter more time |
| Reproducibility and Inspection | five seeds end one to four points apart; activations grow from about 0.7 to 3–4 over eight blocks |
:::
