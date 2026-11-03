# Tools for Deep Learning
:label:`chap_appendix_tools`

Most readers first run this book's code on a laptop or in a browser
tab: open a notebook, run the cells, change the learning rate, and run them
again. Sooner or later something no longer fits. A model needs more memory
than the laptop has, a training run would take a week, or an open language
model downloaded for a weekend project writes three words per second. Each
of these looks like a question about which product to use. Underneath, the
same few questions recur: does the job fit in memory, what limits its speed,
and what will a finished result cost?

Products change within months, but the answers rest on a few rules that
last longer. Memory decides what can run at all, so we check that a job
fits before we make it faster. When a language model generates text, it
reads every weight it uses from memory once for each new token, so memory
bandwidth sets the speed of generation. The cheapest machine per hour does
not give the cheapest experiment if it runs the job more slowly or the job
fails partway, so we compare the cost of a completed result. And
a result that cannot be rerun from what was recorded cannot be checked or
built on, so reproducibility belongs to the workflow from the first cell.
In :numref:`chap_performance` a program was limited by its arithmetic, by
its memory bandwidth, or by the overhead of issuing operations. These rules
apply that model to the decisions of what to run, rent, buy, and deploy.

:numref:`sec_interactive_development` asks how to run the book's notebooks
locally so that a result does not depend on hidden state.
:numref:`sec_hosted_notebooks` asks how to run them with nothing installed,
on Colab or Kaggle. :numref:`sec_cloud_instances` asks when to rent an
accelerator and how to count what a run costs. :numref:`sec_hardware_buyers`
asks which hardware to buy, given what limits the workload.
:numref:`sec_software_ecosystem` asks where to find models, datasets, and
benchmarks, and how to judge and record them.
:numref:`sec_training_systems` asks how to spread training over several
accelerators when one is not enough. :numref:`sec_model_serving` asks how to
serve a model to many users at once. :numref:`sec_developers_guide` asks how
this book is built from its sources and how to propose a change to it.

Prices, product names, quotas, and library versions change quickly. The
sections that quote them give the date of each quote; check them before use.

```toc
:maxdepth: 2

interactive-development
hosted-notebooks
cloud-instances
hardware
software-ecosystem
training-systems
model-serving
developers-guide
```

The generated utility and `d2l` API documents remain searchable HTML
reference pages. They are not part of the teaching sequence or the PDF
edition.

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter, from working on
one machine to training and serving at scale. All are freely accessible
online.

**Systems and scaling**

- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/) — an online book on predicting performance before measuring it: rooflines, TPU and GPU anatomy, parallelism, and inference arithmetic, worked out with numbers for :numref:`sec_hardware_buyers`, :numref:`sec_training_systems`, and :numref:`sec_model_serving`.
- [The Ultra-Scale Playbook — Hugging Face (2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — a guide to training on GPU clusters, based on more than 4,000 measured runs on up to 512 GPUs; it extends :numref:`sec_training_systems` with detailed memory budgets and parallelism trade-offs.
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) — lectures on YouTube and assignments on GitHub that build a language model end to end, from tokenization through distributed training and inference; a course companion to :numref:`sec_training_systems` and :numref:`sec_model_serving`.
- [GPU MODE lectures](https://github.com/gpu-mode/lectures) — videos, slides, and notebooks from CUDA fundamentals through FlashAttention and quantization, for readers who want to write the kernels whose limits :numref:`sec_hardware_buyers` estimates.

**Performance thinking**

- [Making Deep Learning Go Brrrr From First Principles — Horace He (2022)](https://horace.io/brrr_intro.html) — a short essay that divides a program's time into compute, memory bandwidth, and overhead, the division behind :numref:`sec_hardware_buyers`.
- [Transformer Inference Arithmetic — kipply (2022)](https://kipp.ly/p/transformer-inference-arithmetic) — KV-cache sizes and the bandwidth limit of decoding worked out by hand, the estimates behind :numref:`sec_hardware_buyers` and :numref:`sec_model_serving`.
- [Accelerating Generative AI with PyTorch II: GPT, Fast — PyTorch team (2023)](https://pytorch.org/blog/accelerating-generative-ai-2/) — with the [gpt-fast](https://github.com/meta-pytorch/gpt-fast) code; an inference speedup of almost 10× built step by step in under 1,000 lines of PyTorch, using compilation, quantization, speculative decoding, and tensor parallelism.
- [Which GPU for Deep Learning? — Tim Dettmers (2023)](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) — last updated in January 2023, so its hardware examples are dated, but its method for choosing a GPU from the workload complements :numref:`sec_hardware_buyers`.

**Surveys**

- [A Survey on Efficient Inference for Large Language Models — Zhou et al. (2024)](https://arxiv.org/abs/2404.14294) — a taxonomy of data-, model-, and system-level inference optimizations, with comparative experiments on representative methods; background for :numref:`sec_model_serving`.
- [A Survey on Large Language Model Acceleration Based on KV Cache Management — Li et al. (2024)](https://arxiv.org/abs/2412.19442) — methods for storing, compressing, and evicting the KV cache that :numref:`sec_model_serving` introduces, with a curated paper list.

**Staying current**

- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) and the [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions) — release reports, quantization results, hardware measurements, and reproduction attempts; :numref:`sec_software_ecosystem` describes how to weigh such sources.
- [Pro Git — Chacon and Straub](https://git-scm.com/book/en/v2) — a free book whose chapters 1–3 and 6 cover the Git and pull-request workflow that :numref:`sec_developers_guide` assumes.
