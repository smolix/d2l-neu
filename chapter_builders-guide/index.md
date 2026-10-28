# Computation
:label:`chap_computation`

So far each model has been a few lines of code: a class with some layers, a
loss, and a call to `fit`. Real models are built from hundreds of such
pieces, hold billions of numbers, train for days on accelerators, and must
survive crashes, be shared, and be fine-tuned by others. What happens to a
model between its definition and a trained result we can trust is a matter
of software, and getting it wrong is costly: a model that is initialized
badly does not train, one that is saved incompletely cannot resume, one that
runs in the wrong number format silently loses precision, and one that is
placed on the wrong device runs ten times slower or not at all.

This chapter opens up the machinery that the previous chapters used without
comment. It introduces no new models or datasets; every later chapter relies
on what it teaches.

:numref:`sec_model_construction` asks how a model is assembled from modules,
and what a module is. :numref:`sec_parameters` asks what state a model
holds, and how much memory training it needs. :numref:`sec_init_param` asks
how to set that state before training, and how to change the defaults.
:numref:`sec_custom_layer` asks how to add a layer or a differentiable
function that the library does not provide. :numref:`sec_numerics` asks
which number formats a model can compute in, and what lower precision costs
and saves. :numref:`sec_read_write` asks how to save a model and its
training state, and how to load weights that someone else trained.
:numref:`sec_use_gpu` asks how computation is placed on devices and how GPU
memory is spent. :numref:`sec_repro` asks when two runs of the same code give
the same result, and how to look inside a running model.

The chapter covers one device at a time; training on several devices
follows in :numref:`chap_performance`.

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
- **Train:** a bad initial scale stops a deep network from learning; the
  wrong number format silently loses precision.
- **Survive:** long runs are interrupted; resuming needs weights *and*
  optimizer state.
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
- save and resume a training run, and load pretrained weights safely;
- place work on devices, time it correctly, and trade compute for memory;
- make runs repeatable, and look inside a running model.

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
| Modules and Model Construction | the §4.2 MLP has 4 parameter arrays; an 8B model has 291 |
| Parameters, State, and Memory | 8B parameters: 32 GB in fp32, about 128 GB to train with Adam |
| Initialization | 32 residual blocks grow the activations' scale by about $2^{16}$; scaled branches keep it below $\sqrt{e}$ |
| Custom Layers and Functions | RMSNorm maps (3, 4) and (300, 400) to the same (0.849, 1.131) |
| Numerical Formats | $300^2$ overflows in fp16 but not in bf16; loss scaling rescues underflowing gradients |
| Saving, Loading, and Pretrained Weights | 16 GB of bf16 weights against a 96 GB resumable checkpoint |
| GPUs, Devices, and Memory | activation checkpointing cuts a step from about 1.1 GiB to 0.4–0.5 GiB for about a quarter more time |
| Reproducibility and Inspection | Xavier activations grow from about 0.7 to 3–4 over eight blocks |
:::
