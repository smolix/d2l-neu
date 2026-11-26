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

The chapter moves from a single interactive notebook to multi-machine training and production serving. We begin with where code runs day to day: managing Jupyter kernel state and local `uv` environments so results do not depend on hidden execution history (:numref:`sec_interactive_development`), running notebooks with zero local setup on Colab and Kaggle (:numref:`sec_hosted_notebooks`), and pricing rented cloud accelerators by the total cost of a completed result (:numref:`sec_cloud_instances`). Next, we turn to hardware and external artifacts: choosing what to buy from memory capacity, bandwidth, and compute bounds (:numref:`sec_hardware_buyers`), and discovering, evaluating, pinning, and licensing pretrained models and datasets from public hubs (:numref:`sec_software_ecosystem`). Finally, we scale across machines and contributors, partitioning training across multiple accelerators (:numref:`sec_training_systems`), serving concurrent users under latency targets (:numref:`sec_model_serving`), and seeing how this book itself is built and maintained from single-source Markdown files (:numref:`sec_developers_guide`).

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

- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/): an online book on predicting performance before measuring it, covering rooflines, TPU and GPU anatomy, parallelism, and inference arithmetic worked out with numbers for :numref:`sec_hardware_buyers`, :numref:`sec_training_systems`, and :numref:`sec_model_serving`.
- [The Ultra-Scale Playbook — Hugging Face (2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook): a guide to training on GPU clusters, based on more than 4,000 measured runs on up to 512 GPUs; it extends :numref:`sec_training_systems` with detailed memory budgets and parallelism trade-offs.
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/): lectures on YouTube and assignments on GitHub that build a language model end to end, from tokenization through distributed training and inference; a course companion to :numref:`sec_training_systems` and :numref:`sec_model_serving`.
- [GPU MODE lectures](https://github.com/gpu-mode/lectures): videos, slides, and notebooks from CUDA fundamentals through FlashAttention and quantization, for readers who want to write the kernels whose limits :numref:`sec_hardware_buyers` estimates.

**Performance thinking**

- [Making Deep Learning Go Brrrr From First Principles — Horace He (2022)](https://horace.io/brrr_intro.html): a short essay that divides a program's time into compute, memory bandwidth, and overhead, the division behind :numref:`sec_hardware_buyers`.
- [Transformer Inference Arithmetic — kipply (2022)](https://kipp.ly/p/transformer-inference-arithmetic): KV-cache sizes and the bandwidth limit of decoding worked out by hand, the estimates behind :numref:`sec_hardware_buyers` and :numref:`sec_model_serving`.
- [Accelerating Generative AI with PyTorch II: GPT, Fast — PyTorch team (2023)](https://pytorch.org/blog/accelerating-generative-ai-2/): accompanied by the [gpt-fast](https://github.com/meta-pytorch/gpt-fast) code; an inference speedup of almost 10× built step by step in under 1,000 lines of PyTorch, using compilation, quantization, speculative decoding, and tensor parallelism.
- [Which GPU for Deep Learning? — Tim Dettmers (2023)](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/): last updated in January 2023, so its hardware examples are dated, but its method for choosing a GPU from the workload complements :numref:`sec_hardware_buyers`.

**Surveys**

- [A Survey on Efficient Inference for Large Language Models — Zhou et al. (2024)](https://arxiv.org/abs/2404.14294): a taxonomy of data-, model-, and system-level inference optimizations, with comparative experiments on representative methods; background for :numref:`sec_model_serving`.
- [A Survey on Large Language Model Acceleration Based on KV Cache Management — Li et al. (2024)](https://arxiv.org/abs/2412.19442): methods for storing, compressing, and evicting the KV cache that :numref:`sec_model_serving` introduces, with a curated paper list.

**Staying current**

- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) and the [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions): release reports, quantization results, hardware measurements, and reproduction attempts; :numref:`sec_software_ecosystem` describes how to weigh such sources.
- [Pro Git — Chacon and Straub](https://git-scm.com/book/en/v2): a free book whose chapters 1–3 and 6 cover the Git and pull-request workflow that :numref:`sec_developers_guide` assumes.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §39]{.kicker}

Tools for Deep Learning: From Local Notebooks to Distributed Clusters and Serving<br>**reproducible kernels · hosted runtimes · cloud economics · hardware rooflines · open-weight artifacts · distributed training · inference engines**
:::
:::

::: {.slide title="Four Systems Questions Behind Every Tooling Choice"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Does it fit in memory?** Training holds parameters, gradients, optimizer moments ($16N$), and activations in VRAM; OOM stops a run before FLOP/s matters.
- **2. What bounds throughput?** Training and prompt prefill are compute-bound, whereas batch-1 autoregressive decoding is memory-bandwidth bound.
- **3. Cost & Reproducibility:** Optimize total cost per completed result (including setup, egress, and spot recovery) and lock environments/SHAs from cell one.
:::

::: {.col .fig .big}
@fig:ch39-tools-four-questions
:::
:::
:::

::: {.slide title="Three First-Principles Equations for Systems Sizing"}
[Quantitative Rules]{.kicker}

::: {.cols .vc}
::: {.col}
$$M_{\text{train}} \approx 16 N + M_{\text{act}}, \qquad \text{tok/s} \lesssim \frac{\beta}{B_w + b n B_{\text{kv}}}$$

- **Training state & compute (§{sec_hosted_notebooks}, §{sec_hardware_buyers}, §{sec_training_systems}):** Mixed-precision Adam uses $16\text{ B/param}$ and $C \approx 6ND\text{ FLOPs}$ across $N$ parameters and $D$ tokens.
- **Decode & checkpoint intervals (§{sec_hosted_notebooks}–§{num:sec_model_serving}):** Batch-1 decode streams $B_w + B_{\text{kv}}$ at bandwidth $\beta$, and Young's law checkpoints every $\tau^* = \sqrt{2\delta M}$.
:::

::: {.col .fig}
@fig:ch39-tools-sizing-equations
:::
:::
:::

::: {.slide title="Roadmap of Chapter §{num:chap_appendix_tools}: From Laptop to Production"}
[Chapter Architecture]{.kicker}

::: {.cols .vc}
::: {.col}
- **§{sec_interactive_development}–§{num:sec_cloud_instances} Notebooks, Hosted VMs & Cloud:** Kernel state discipline, `uv` lock files, Colab/Kaggle persistence, and cloud cost per completed result.
- **§{sec_hardware_buyers}–§{num:sec_software_ecosystem} Hardware & Ecosystem:** Capacity vs. bandwidth rooflines, unified memory for MoE, Arena Elo resolution, and `safetensors`/SHA pinning.
- **§{sec_training_systems}–§{num:sec_utils} Scaling, Serving & D2L Internals:** DDP/FSDP2, 3D parallelism, continuous batching, PagedAttention, and the D2L single-source build.
:::

::: {.col .fig .big}
@fig:ch39-tools-lifecycle-roadmap
:::
:::
:::

::: {.slide title="Concrete Worked Benchmarks Across the Chapter"}
[Pedagogical Thread]{.kicker}

::: {.cols .vc}
::: {.col}
- **Reproducible setups & economics (§{sec_interactive_development}–§{num:sec_cloud_instances}):** `uv pip sync`, environment fingerprints, and engineer-inclusive cloud cost models across 4090 / A100 / H100.
- **Sizing & verification (§{sec_hardware_buyers}–§{num:sec_software_ecosystem}):** 7B training state ($78\text{ GiB}$ full vs. $\sim 4\text{ GiB}$ QLoRA), decode bounds, Bradley-Terry SE ($347/\sqrt{v}$), and frozen manifests.
- **Distributed training, serving & the build (§{sec_training_systems}–§{num:sec_developers_guide}):** 8-GPU FSDP2 ($22\text{ GiB}$ vs. $95\text{ GiB}$ DDP), continuous batching simulators, and prefix output hashing.
:::

::: {.col .fig .big}
@fig:ch39-tools-worked-benchmarks
:::
:::
:::

::: {.slide title="How to Navigate This Chapter and Stay Current"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Immediate setup (§{sec_interactive_development}–§{num:sec_hosted_notebooks}):** Start here to run the book's notebooks locally with `uv` or on Colab and Kaggle without losing work to ephemeral sessions.
- **Infrastructure decisions (§{sec_cloud_instances}–§{num:sec_software_ecosystem}):** Read these sections before renting cloud GPUs, building a workstation, or adopting an open-weight checkpoint from Hugging Face.
:::

::: {.col}
- **Scaling and contributing (§{sec_training_systems}–§{num:sec_utils}):** Study §{sec_training_systems}–§{num:sec_model_serving} alongside Chapter §{num:chap_performance} (*Computational Performance*) for multi-GPU training and LLM serving, §{sec_developers_guide} to contribute to D2L, and §{sec_utils} for `d2l` utilities.
- **Companion references:** Pair this chapter with DeepMind's *How to Scale Your Model*, Hugging Face's *Ultra-Scale Playbook*, and Stanford CS336.
:::
:::

::: {.d2l-note}
**Up next (§{sec_interactive_development} Notebooks):** How the Jupyter kernel execution model works, why hidden state breaks reproducibility, and how to set up deterministic local environments with `uv`.
:::
:::
