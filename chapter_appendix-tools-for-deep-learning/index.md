# Tools for Deep Learning
:label:`chap_appendix_tools`

This chapter covers the tools and systems used around deep learning models.
It begins with reproducible notebook use
(:numref:`sec_interactive_development`), hosted notebooks
(:numref:`sec_hosted_notebooks`), rented accelerators
(:numref:`sec_cloud_instances`), and local hardware
(:numref:`sec_hardware_buyers`). It then surveys sources for models,
datasets, papers, and benchmarks (:numref:`sec_software_ecosystem`), before
introducing distributed training (:numref:`sec_training_systems`) and model
serving (:numref:`sec_model_serving`). The final section explains how this
book is built and how to contribute to it
(:numref:`sec_developers_guide`).

Prices, model names, quotas, and library versions are dated to mid-2026 and
must be checked before use. The durable principles are to establish memory
fit before optimizing speed, account for bandwidth limits during generation,
compare cost per completed result, and treat reproducibility as part of the
workflow.

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

The references below extend this chapter's practical arc — from working
effectively on one machine to training and serving at scale. All are
freely accessible online.

**Systems and scaling**

- [How to Scale Your Model — Austin et al. (Google DeepMind, 2025)](https://jax-ml.github.io/scaling-book/) — free online book; from rooflines and TPU/GPU anatomy to parallelism and inference arithmetic, this is the quantitative companion to :numref:`sec_hardware_buyers`, :numref:`sec_training_systems`, and :numref:`sec_model_serving` — a quantitative treatment of predicting performance before measuring it.
- [The Ultra-Scale Playbook — Hugging Face](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — free; a hands-on guide to GPU-cluster training, based on thousands of instrumented runs on up to 512 GPUs; it extends the discussion in :numref:`sec_training_systems` with detailed memory budgets and parallelism trade-offs.
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) — free lectures and assignments; builds a language model end to end including the systems layer — tokenization through distributed training and inference — the university-course companion to this chapter.
- [GPU MODE lecture series](https://github.com/gpu-mode/lectures) — free videos, slides, and notebooks, from CUDA fundamentals through FlashAttention and quantization; with material on kernel-level performance analysis and an active community.

**Performance thinking, in blog-post form**

- [Making Deep Learning Go Brrrr From First Principles — Horace He](https://horace.io/brrr_intro.html) — free; the compute-versus-bandwidth-versus-overhead taxonomy that underlies :numref:`sec_hardware_buyers`, in a concise article.
- [Transformer Inference Arithmetic — kipply](https://kipp.ly/transformer-inference-arithmetic/) — free; KV-cache sizing and bandwidth-bound decoding worked out by hand — its methods support the KV-cache and bandwidth calculations in :numref:`sec_model_serving`.
- [Accelerating Generative AI with PyTorch: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) — free, with the [gpt-fast](https://github.com/pytorch-labs/gpt-fast) code; a ~10× inference speedup built step by step in under 1,000 lines of native PyTorch — compilation, quantization, speculative decoding, and tensor parallelism made concrete.
- [Which GPU for Deep Learning? — Tim Dettmers](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) — free; last updated in 2023; its hardware examples are dated, but its method for reasoning about GPU choice complements :numref:`sec_hardware_buyers`.

**Surveys**

- [A Survey on Efficient Inference for Large Language Models — Zhou et al. (2024)](https://arxiv.org/abs/2404.14294) — free; a taxonomy of data-, model-, and system-level inference optimization, with comparative experiments.
- [A Survey on Large Language Model Acceleration Based on KV Cache Management — Li et al. (2025)](https://arxiv.org/abs/2412.19442) — free; a survey of KV-cache management methods introduced in :numref:`sec_model_serving`, with a maintained paper repository.

**Staying current**

- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) and the [llama.cpp benchmark discussions](https://github.com/ggml-org/llama.cpp/discussions) — free; a source of release reports, quantization results, hardware measurements, and reproduction attempts (see :numref:`sec_software_ecosystem` for the fuller information diet).
- [Pro Git — Chacon & Straub](https://git-scm.com/book/en/v2) — free book; chapters 1–3 and 6 cover the Git and pull-request concepts assumed by :numref:`sec_developers_guide`.
