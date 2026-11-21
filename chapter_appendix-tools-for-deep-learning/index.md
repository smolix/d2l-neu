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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §39]{.kicker}

Tools for Deep Learning: From Local Notebooks to Distributed Clusters and Serving<br>**reproducible kernels · hosted runtimes · cloud economics · hardware rooflines · open-weight artifacts · distributed training · inference engines**
:::
:::

::: {.slide title="Four Systems Questions Behind Every Tooling Choice"}
[Motivation]{.kicker}

Product names, cloud prices, and library versions change every few months, but every decision about what to run, rent, buy, or deploy reduces to four durable systems questions:

::: {.cols}
::: {.col}
- **1. Does the workload fit in memory?** Training holds parameters, gradients, optimizer moments, and activations in device RAM; if peak memory exceeds capacity, the job fails before compute speed matters.
- **2. What physical limit bounds throughput?** Training and prompt prefill reuse weights across many tokens (compute-bound), whereas autoregressive decoding reads every active weight per token (bandwidth-bound).
:::

::: {.col}
- **3. What does a completed result cost?** The cheapest GPU per hour is rarely the cheapest experiment if slower iterations, egress fees, or uncheckpointed spot interruptions multiply wall time and engineering hours.
- **4. Can another person reproduce the output?** Hidden kernel state, unpinned container images, and floating Hub branches silently invalidate empirical claims unless reproducibility is built in from cell one.
:::
:::
:::

::: {.slide title="Three First-Principles Equations for Systems Sizing"}
[Quantitative Rules]{.kicker}

Before benchmarking a machine or launching a cluster, back-of-the-envelope arithmetic predicts whether a job fits, how fast it runs, and how often it must checkpoint:

::: {.cols}
::: {.col}
- **Peak mixed-precision training state (§39.4, §39.6):** With BF16 weights/gradients and FP32 master weights plus Adam moments ($2 + 2 + 4 + 8 = 16\text{ B/param}$):
  $$M_{\text{train}} \approx 16 N + M_{\text{activations}} + M_{\text{workspace}}$$
- **Autoregressive decode bandwidth bound (§39.4, §39.7):** Streaming active weights $B_w$ and KV cache $B_{\text{kv}}$ at memory bandwidth $\beta$:
  $$\text{tokens/s} \lesssim \frac{\beta}{B_w + b \, n \, B_{\text{kv}}}$$
:::

::: {.col}
- **Compute cost of training (§39.2, §39.6):** Forward ($2ND$) plus backward ($4ND$) floating-point operations across $N$ parameters and $D$ tokens:
  $$C \approx 6 N D \text{ FLOPs}$$
- **Young's optimal checkpoint interval (§39.2, §39.3):** Balancing checkpoint write duration $\delta$ against mean time between interruptions $M$:
  $$\tau^* = \sqrt{2 \delta M}$$
:::
:::
:::

::: {.slide title="Roadmap of Chapter 39: From Laptop to Production"}
[Chapter Architecture]{.kicker}

Each section addresses one stage of the engineering lifecycle as a project outgrows a single browser tab:

| Section | Core Engineering Question | Primary Bottleneck or Risk |
| :--- | :--- | :--- |
| **§39.1 Notebooks** | How do we run notebooks locally without hidden state? | Out-of-order kernel mutations & wrong-venv kernels |
| **§39.2 Colab & Kaggle** | How do we work on free, borrowed browser VMs? | Ephemeral disks, 12-hour disconnects & image drift |
| **§39.3 Cloud Computing** | When do we rent GPUs and how do we price a run? | Cost per completed result, spot preemption & egress |
| **§39.4 Hardware** | Which machine should we buy for training or local LLMs? | VRAM capacity vs. memory bandwidth vs. TCO break-even |
| **§39.5 Ecosystem** | How do we select, verify, and pin open models & data? | Benchmark contamination, commit drift & pickle exploits |
| **§39.6 Distributed Training** | How do we scale training across multiple GPUs/nodes? | State sharding (FSDP), interconnect bandwidth & faults |
| **§39.7 Model Serving** | How do we serve many concurrent users under an SLO? | KV-cache fragmentation, TTFT/TPOT & batch scheduling |
:::

::: {.slide title="Concrete Worked Benchmarks Across the Chapter"}
[Pedagogical Thread]{.kicker}

Every section pairs its systems principles with executable sizing models and dated empirical snapshots (November 2026):

| Section | Worked Quantitative Model / Live Code Cell | Key Systems Takeaway |
| :--- | :--- | :--- |
| **§39.1–39.2** | `uv` lock sync, environment fingerprint & path adapter | Reconstructible setups survive kernel restarts and VM resets |
| **§39.3** | Invoice vs. engineer-inclusive cost across 4090 / A100 / H100 | One extra 30-min debugging round outweighs a $6\times$ hourly gap |
| **§39.4** | 7B training memory ($86\text{ GiB}$ vs. $4\text{ GiB}$ QLoRA) & decode bounds | Discrete GPUs win on bandwidth; unified memory wins on capacity |
| **§39.5** | Bradley-Terry Arena SE ($347/\sqrt{v}$) & frozen `Artifact` manifest | Aggregate Elo hides domain reversals; pin 40-char commit SHAs |
| **§39.6** | 8-GPU 7B memory budget: DDP ($95\text{ GiB}$) vs. FSDP ($22\text{ GiB}$) | Shard state with FSDP2 inside nodes; checkpoint atomically |
| **§39.7–39.8** | Continuous batching simulator & D2L single-source build pipeline | Paged caches + continuous batching maximize goodput under SLOs |
:::

::: {.slide title="How to Navigate This Chapter and Stay Current"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Immediate setup (§39.1–39.2):** Start here to run the book's notebooks locally with `uv` or on Colab and Kaggle without losing work to ephemeral sessions.
- **Infrastructure decisions (§39.3–39.5):** Read these sections before renting cloud GPUs, building a workstation, or adopting an open-weight checkpoint from Hugging Face.
:::

::: {.col}
- **Scaling and contributing (§39.6–39.9):** Study §39.6–39.7 alongside Chapter 13 (*Computational Performance*) for multi-GPU training and LLM serving, §39.8 to contribute to D2L, and §39.9 for `d2l` utilities.
- **Companion references:** Pair this chapter with DeepMind's *How to Scale Your Model*, Hugging Face's *Ultra-Scale Playbook*, and Stanford CS336.
:::
:::

::: {.d2l-note}
**Up next (§39.1 Notebooks):** How the Jupyter kernel execution model works, why hidden state breaks reproducibility, and how to set up deterministic local environments with `uv`.
:::
:::
