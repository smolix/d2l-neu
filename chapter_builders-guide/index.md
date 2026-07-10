# Builders' Guide
:label:`chap_computation`

Alongside giant datasets and powerful hardware, great software tools have
played an indispensable role in the rapid progress of deep learning. Deep
learning libraries let us recycle standard components while retaining the
ability to modify anything, and over time their abstractions have grown
coarser: from individual neurons, to layers, to the multi-layer *blocks* from
which today's models are assembled.

So far we called upon these libraries without asking how they work. We built
models from layers, initialized and trained them, and treated everything
between `net(X)` and the loss as machinery. This chapter opens the machinery.
In 2016 that meant learning to build, initialize, and save a small model
trained from scratch in 32-bit arithmetic on one device. Those skills remain,
and this chapter teaches them, but the working assumptions around them have
changed: a model is now assembled from a configuration object, measured in
gigabytes, run in reduced precision, checkpointed together with its optimizer
state, and as often as not initialized from someone else's weights rather than
from a random number generator.

Accordingly, we proceed in eight steps. We start with how models are built
from modules and configs (:numref:`sec_model_construction`), what a model's
state is and what it costs in memory (:numref:`sec_parameters`), how that
state is initialized (:numref:`sec_init_param`), and how to write layers the
library does not provide (:numref:`sec_custom_layer`). We then turn to the
numeric formats models compute in (:numref:`sec_numerics`), how state is
saved, restored, and adopted from pretrained models
(:numref:`sec_read_write`), how tensors and models live on GPUs and in GPU
memory (:numref:`sec_use_gpu`), and finally how to make runs repeatable and
inspect a model from the outside (:numref:`sec_repro`). The chapter
introduces no new models or datasets; the advanced modeling chapters that
follow rely on these techniques throughout.

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

The references below go deeper on the machinery this chapter opens up: how
frameworks represent models and state, automatic differentiation, numerics and
mixed precision, devices and memory, and reproducible training. All are freely
accessible online except where noted.

**Books**

- [Deep Learning with PyTorch — Stevens, Antiga & Viehmann](https://web.archive.org/web/20211012030609/https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) — free PDF from the PyTorch team (archived copy; the original pytorch.org link has gone away); Part 1 walks tensors, storage, autograd, and `nn.Module` mechanics at exactly this chapter's level of "open the machinery".
- [Machine Learning Systems — Vijay Janapa Reddi](https://mlsysbook.ai/) — free online; the systems view around this chapter: frameworks, data pipelines, training infrastructure, and efficient deployment.

**Courses and video lectures**

- [CMU 10-414/714: Deep Learning Systems — Chen & Kolter](https://dlsyscourse.org/) — free lectures and assignments; you build "needle", a miniature framework with autograd, modules, initialization, and GPU support — this entire chapter from the implementor's side.
- [Neural Networks: Zero to Hero — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html) — free video series; builds autograd, modules, and training loops from scratch in plain Python, making every abstraction in this chapter concrete before you rely on the library's version.
- [fast.ai Part 2: Deep Learning Foundations — Howard et al.](https://course.fast.ai/Lessons/part2.html) — free; rebuilds a training framework from tensor ops upward (modules, initialization, mixed precision, accelerated training), the practitioner's companion to this chapter.

**Tutorials, notes, and interactive**

- [PyTorch internals — Edward Yang](http://blog.ezyang.com/2019/05/pytorch-internals/) — free; the classic guided tour of tensors, strides, dispatch, and autograd inside PyTorch, one level below :numref:`sec_model_construction`.
- [JAX — The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) — free; pure functions, explicit PRNG keys, and jit constraints — the functional worldview behind this book's JAX tab, stated as a list of gotchas.
- [Train With Mixed Precision — NVIDIA](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — free; the vendor guide to fp16/bf16 arithmetic and loss scaling that :numref:`sec_numerics` distills.
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic — David Goldberg](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — free; the standard reference beneath every rounding and overflow issue in :numref:`sec_numerics`.
- [Making Deep Learning Go Brrrr From First Principles — Horace He](https://horace.io/brrr_intro.html) — free; the compute-, memory-, and overhead-bound mental model that :numref:`sec_use_gpu` builds on.
- [Reproducibility — PyTorch notes](https://pytorch.org/docs/stable/notes/randomness.html) — free; the determinism flags, their costs, and their limits — the fine print behind :numref:`sec_repro`.

**Foundational papers**

- [Automatic Differentiation in Machine Learning: a Survey — Baydin, Pearlmutter, Radul & Siskind (2018), *JMLR*](https://jmlr.org/papers/v18/17-468.html) — free; the definitive account of forward- and reverse-mode autodiff, the algorithm every framework in this book implements.
- [Mixed Precision Training — Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) — free; the origin of the fp16 + master-weights + loss-scaling recipe in :numref:`sec_numerics`.
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library — Paszke et al. (2019)](https://arxiv.org/abs/1912.01703) — free; the design rationale (eager execution, autograd, memory allocator) for the imperative style this chapter teaches.
