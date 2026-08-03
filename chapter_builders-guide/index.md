# Computation
:label:`chap_computation`

Deep learning libraries provide reusable layers and training components while
allowing models to define their own computation. Their central abstraction is
the *module*: a unit that can contain parameters, child modules, and a forward
computation. Modules range from individual layers to the repeated blocks from
which large models are assembled.

This chapter explains the software structure behind the models used so far.
It covers module construction, initialization, model state, serialization,
custom layers, reproducibility, numeric formats, and device placement. These
topics matter at any scale, but modern training makes their interaction
especially important: models are commonly built from configuration objects,
run in reduced precision, measured in gigabytes, initialized from pretrained
weights, and checkpointed together with optimizer state.

Accordingly, we proceed in eight steps. We start with how models are built
from modules and configs (:numref:`sec_model_construction`), what a model's
state is and what it costs in memory (:numref:`sec_parameters`), how that
state is initialized (:numref:`sec_init_param`), and how that state is saved,
restored, and adopted from pretrained models (:numref:`sec_read_write`). We then
turn to numeric formats (:numref:`sec_numerics`), devices and GPU memory
(:numref:`sec_use_gpu`), and layers the library does not provide
(:numref:`sec_custom_layer`). Finally, we distinguish repeatable experiments
from inspecting a model's execution (:numref:`sec_repro`). The chapter
introduces no new models or datasets; the advanced modeling chapters that
follow rely on these techniques throughout.

```toc
:maxdepth: 2

model-construction
parameters-state-memory
init
saving-loading
numerics
gpus-devices-memory
custom-layers
reproducibility-inspection
```

## Resources and Further Reading {.unnumbered}

The references below develop the mechanisms introduced in this chapter: how
frameworks represent models and state, automatic differentiation, numerics and
mixed precision, devices and memory, and reproducible training. All are freely
accessible online except where noted.

**Books**

- [Deep Learning with PyTorch — Stevens, Antiga & Viehmann](https://web.archive.org/web/20211012030609/https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) — free PDF from the PyTorch team (archived copy; the original pytorch.org link has gone away); Part 1 covers tensors, storage, autograd, and `nn.Module` at the same level as this chapter.
- [Machine Learning Systems — Vijay Janapa Reddi](https://mlsysbook.ai/) — free online; the systems view around this chapter: frameworks, data pipelines, training infrastructure, and efficient deployment.

**Courses and video lectures**

- [CMU 10-414/714: Deep Learning Systems — Chen & Kolter](https://dlsyscourse.org/) — free lectures and assignments; students build "needle", a miniature framework with autograd, modules, initialization, and GPU support, providing an implementer's view of these abstractions.
- [Neural Networks: Zero to Hero — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html) — free video series; builds autograd, modules, and training loops from scratch in plain Python, illustrating the abstractions used by deep learning libraries.
- [fast.ai Part 2: Deep Learning Foundations — Howard et al.](https://course.fast.ai/Lessons/part2.html) — free; rebuilds a training framework from tensor operations upward, including modules, initialization, mixed precision, and accelerated training.

**Tutorials, notes, and interactive**

- [PyTorch internals — Edward Yang](http://blog.ezyang.com/2019/05/pytorch-internals/) — free; a guided tour of tensors, strides, dispatch, and autograd inside PyTorch, one level below :numref:`sec_model_construction`.
- [JAX — The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) — free; pure functions, explicit PRNG keys, and jit constraints — the functional worldview behind this book's JAX tab, stated as a list of gotchas.
- [Train With Mixed Precision — NVIDIA](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — free; the vendor guide to fp16/bf16 arithmetic and loss scaling that :numref:`sec_numerics` distills.
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic — David Goldberg](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — free; the standard reference beneath every rounding and overflow issue in :numref:`sec_numerics`.
- [Making Deep Learning Go Brrrr From First Principles — Horace He](https://horace.io/brrr_intro.html) — free; a first-principles account of compute, memory, and framework overhead covered in :numref:`sec_use_gpu`.
- [Reproducibility — PyTorch notes](https://pytorch.org/docs/stable/notes/randomness.html) — free; the determinism flags, their costs, and their limits, complementing :numref:`sec_repro`.

**Foundational papers**

- [Automatic Differentiation in Machine Learning: a Survey — Baydin, Pearlmutter, Radul & Siskind (2018), *JMLR*](https://jmlr.org/papers/v18/17-468.html) — free; the definitive account of forward- and reverse-mode autodiff, the algorithm every framework in this book implements.
- [Mixed Precision Training — Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) — free; the origin of the fp16 + master-weights + loss-scaling recipe in :numref:`sec_numerics`.
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library — Paszke et al. (2019)](https://arxiv.org/abs/1912.01703) — free; the design rationale (eager execution, autograd, memory allocator) for the imperative style this chapter teaches.
