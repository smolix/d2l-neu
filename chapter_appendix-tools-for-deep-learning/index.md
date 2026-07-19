# Tools for Deep Learning
:label:`chap_appendix_tools`

The rest of this book teaches models; this chapter teaches the workshop
around them. It follows the trajectory most readers actually take: run the
book's notebooks well (:numref:`sec_interactive_development`), for free in
the cloud (:numref:`sec_hosted_notebooks`), then on rented GPUs when the
free tiers run out (:numref:`sec_cloud_instances`) or on hardware of your
own (:numref:`sec_hardware_buyers`). From there it widens to the ecosystem
you will work inside — where models, datasets, papers, and benchmarks
actually live (:numref:`sec_software_ecosystem`) — and to the systems
questions that dominate practice at scale: training across many
accelerators (:numref:`sec_training_systems`) and serving models to users
(:numref:`sec_model_serving`). It closes with the machinery of this book
itself and how to contribute to it (:numref:`sec_developers_guide`).

Two warnings apply throughout. Prices, model names, and library versions
are quoted as of mid-2026 and will age; we date them so you can discount
them, and each section says where to find current numbers. The underlying
reasoning — memory fit before speed, bandwidth bounds on generation, cost
per result rather than per hour, reproducibility as a workflow rather
than a wish — is the part meant to last.

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

- [How to Scale Your Model — Austin et al. (Google DeepMind, 2025)](https://jax-ml.github.io/scaling-book/) — free online book; from rooflines and TPU/GPU anatomy to parallelism and inference arithmetic, this is the quantitative companion to :numref:`sec_hardware_buyers`, :numref:`sec_training_systems`, and :numref:`sec_model_serving` — the best single place to learn to *predict* performance before measuring it.
- [The Ultra-Scale Playbook — Hugging Face](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — free; the definitive hands-on guide to training on GPU clusters, distilled from thousands of instrumented runs on up to 512 GPUs; picks up exactly where :numref:`sec_training_systems` stops, with the memory budgets and parallelism trade-offs worked out in detail.
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) — free lectures and assignments; builds a language model end to end including the systems layer — tokenization through distributed training and inference — the university-course companion to this chapter.
- [GPU MODE lecture series](https://github.com/gpu-mode/lectures) — free videos, slides, and notebooks, from CUDA fundamentals through FlashAttention and quantization; where practitioners actually learn kernel-level thinking, with an active community attached.

**Performance thinking, in blog-post form**

- [Making Deep Learning Go Brrrr From First Principles — Horace He](https://horace.io/brrr_intro.html) — free; the compute-versus-bandwidth-versus-overhead taxonomy that underlies :numref:`sec_hardware_buyers`, in a few thousand vivid words.
- [Transformer Inference Arithmetic — kipply](https://kipp.ly/transformer-inference-arithmetic/) — free; KV-cache sizing and bandwidth-bound decoding worked out by hand — every number in :numref:`sec_model_serving` can be re-derived with the methods here.
- [Accelerating Generative AI with PyTorch: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) — free, with the [gpt-fast](https://github.com/pytorch-labs/gpt-fast) code; a ~10× inference speedup built step by step in under 1,000 lines of native PyTorch — compilation, quantization, speculative decoding, and tensor parallelism made concrete.
- [Which GPU for Deep Learning? — Tim Dettmers](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) — free; frozen since 2023, so ignore the part numbers, but still the classic exposition of *how to reason* about GPU choice — the ancestor of :numref:`sec_hardware_buyers`'s approach.

**Surveys**

- [A Survey on Efficient Inference for Large Language Models — Zhou et al. (2024)](https://arxiv.org/abs/2404.14294) — free; the standard taxonomy of data-, model-, and system-level inference optimization, with comparative experiments.
- [A Survey on Large Language Model Acceleration Based on KV Cache Management — Li et al. (2025)](https://arxiv.org/abs/2412.19442) — free; everything the field knows about the KV cache that :numref:`sec_model_serving` introduces, with a maintained paper repository.

**Staying current**

- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) and the [llama.cpp benchmark discussions](https://github.com/ggml-org/llama.cpp/discussions) — free; where new models, quantizations, and hardware numbers surface first, with reproduction attempts attached (see :numref:`sec_software_ecosystem` for the fuller information diet).
- [Pro Git — Chacon & Straub](https://git-scm.com/book/en/v2) — free book; chapters 1–3 and 6 cover everything :numref:`sec_developers_guide` assumes about Git and pull requests, properly.
