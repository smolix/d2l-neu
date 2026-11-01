# Language Models Part Detailed Chapter and Notebook Plan

**Dive into Deep Learning — proposed replacement for the Language Models part**  
**Planning reference date: 28 October 2026**

This part develops the mathematical, modeling, data, and systems knowledge needed to build, adapt, evaluate, compress, and serve modern language models. Its endpoint is the ability to understand and critically assess a contemporary frontier-model report, implement important mechanisms, and apply many of them to a capable smaller model. It extends the existing Basics and Advanced parts rather than repeating their introductory treatment of attention, transformers, optimization, state-space models, and reinforcement learning.

The proposed structure has **16 chapters and 130 section-level units**. A unit is a planned explanation and notebook design, not a promise that every topic fits into one short notebook. Chapter numbers are local to this part. Ordering, final checkpoint selection, chapter titles, and file slugs remain adjustable; this plan does not create book scaffolding or lock architectural decisions.

## Scope and implementation policy

- **Language and its infrastructure:** text/code data, tokenization, model architectures, pretraining, distributed computation, adaptation, RL/post-training, OPD/OPSD, reasoning, compression, inference, serving, and evaluation.
- **Separate parts:** vision, audio, multimodal modeling, and agent environments/orchestration. Frontier-model case studies examine their language backbones and general training/serving mechanisms. Bounded code evaluation is included; autonomous tool-use loops are not.
- **PyTorch implementations:** explicit tensor/module/training code alongside practical PyTorch ecosystem implementations. Hugging Face is a source of checkpoints, tokenizers, datasets, and reference model code. External serving engines are case studies and optional comparisons. Readers do not need to write CUDA, Triton, JAX, or another framework.
- **Reference versus accelerated implementations:** a transparent PyTorch implementation may prove the algorithm correct without reproducing an optimized kernel's speed. Every performance claim must name the actual execution path, precision, workload, and hardware.
- **Text embeddings:** one substantial notebook in Section 2.7, with references back to it from data selection, adaptation, compression, and evaluation. Extended historical word2vec/GloVe treatment can remain optional/archival.
- **Frontier depth:** explain selected mechanisms in enough detail to implement and analyze them. Include current model reports as integrated case studies; do not turn the part into an unstructured catalogue of method names or leaderboard scores.

## Models compute and recurring artifacts

Use a compatible family where it improves continuity, but do not force every mechanism onto a model that does not support it. Qwen3 is a candidate practical family, not a locked choice. [Qwen3 model collection](https://huggingface.co/collections/Qwen/qwen3), [OLMo](https://github.com/allenai/OLMo), and [SmolLM](https://github.com/huggingface/smollm) are useful starting points for comparing accessibility, training transparency, and resource needs. Base, instruction-tuned, and reasoning checkpoints must be selected according to each experiment; their capabilities and training histories are not interchangeable.

| Tier | Intended use | Required treatment |
|---|---|---|
| CPU and reduced models | Tokenizers, data checks, correctness tests, schedule simulations, algebraic examples | Fast default tests and small deterministic fixtures |
| One GPU around 24GB | Most adaptation, evaluation, compression, and inference exercises using capable 1–8B models | Publish a tested configuration, measured peak memory, sequence/batch limits, and reduced fallback |
| Optional multiple GPUs | Distributed training, expert/context parallelism, and selected serving tests | Supply communication/schedule traces and a simulation or analysis path |
| Frontier-scale case study | Full architecture and cluster reasoning | Use report/configuration analysis and reduced mechanisms; distinguish measured results from projections |

An 8B model's BF16 weights alone occupy approximately 16GB in decimal units. That does not imply full fine-tuning, long-context inference, or simultaneous teacher/student execution fits in 24GB. Training exercises generally use smaller checkpoints, adapters, quantization, sequential teacher scoring, or reduced lengths. Exact configurations are chosen by measurement during notebook authoring.

The recurring artifacts are a corpus manifest, tokenizer comparison, small pretrained model, checkpoint inspector, resource calculator, distributed trainer, adapted checkpoint, RL/distillation comparison, compressed model, optimized generator, service workload replay, and evaluation report. Keep their interfaces compatible so later notebooks reuse real outputs instead of introducing unrelated toy pipelines.

## Opening walkthrough and reading paths

Before Chapter 1, a short orientation loads a capable pretrained model, generates text, measures memory, and runs a small evaluation. It introduces revision pinning and the difference between base and adapted behavior without requiring readers to understand every component. Sources: [Transformers quick tour](https://huggingface.co/docs/transformers/quicktour), [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), and [Raschka's from-scratch materials](https://www.sebastianraschka.com/from-scratch/). This walkthrough is outside the 130 numbered units.

**Model development path:** Chapters 1–5, 8–11, and 15–16. **Training systems path:** Chapters 4–7, with architecture/data prerequisites and Chapter 14's rollout systems. **Deployment path:** selected architecture sections, Chapters 4 and 12–15. The full sequence remains the recommended comprehensive path.

Evaluation is introduced immediately and accompanies every experiment. Chapter 15 consolidates and deepens the methodology; readers do not wait until then to measure quality. Hardware is placed before serious training, after readers have inspected the models and representations whose costs it explains.

## Chapter overview

| Chapter | Title | Units | Main artifact |
|---|---|---:|---|
| 1 | [Text Tokenization and Training Data](#chapter-1-text-tokenization-and-training-data) | 8 | Versioned corpus and tokenizer study |
| 2 | [Modern Language Model Architectures](#chapter-2-modern-language-model-architectures) | 8 | Checkpoint inspector and embedding notebook |
| 3 | [Sparse Recurrent and Hybrid Models](#chapter-3-sparse-recurrent-and-hybrid-models) | 9 | Reduced sparse/hybrid implementations |
| 4 | [Hardware and Performance Models for Language Models](#chapter-4-hardware-and-performance-models-for-language-models) | 8 | Resource model and profiler report |
| 5 | [Pretraining and Continued Training](#chapter-5-pretraining-and-continued-training) | 9 | Reproducible trainer and adaptation study |
| 6 | [Distributed Computation and Sharded Training](#chapter-6-distributed-computation-and-sharded-training) | 8 | Correct distributed training variants |
| 7 | [Parallelism at Scale](#chapter-7-parallelism-at-scale) | 8 | Schedule and topology design study |
| 8 | [Supervised Fine Tuning and Efficient Adaptation](#chapter-8-supervised-fine-tuning-and-efficient-adaptation) | 8 | Adapted checkpoints and capacity experiments |
| 9 | [Preference Learning and Reinforcement Learning](#chapter-9-preference-learning-and-reinforcement-learning) | 9 | Preference/RL comparison |
| 10 | [Distillation and Self-Improvement](#chapter-10-distillation-and-self-improvement) | 8 | SFT versus OPD versus OPSD study |
| 11 | [Reasoning and Inference-Time Computation](#chapter-11-reasoning-and-inference-time-computation) | 7 | Accuracy–compute evaluation |
| 12 | [Quantization and Model Compression](#chapter-12-quantization-and-model-compression) | 9 | Quality–memory–latency comparison |
| 13 | [Efficient Inference and Long Context State](#chapter-13-efficient-inference-and-long-context-state) | 9 | Optimized generator |
| 14 | [Serving and Post Training Systems](#chapter-14-serving-and-post-training-systems) | 8 | Scheduler and workload replay |
| 15 | [Evaluation and Experimental Methodology](#chapter-15-evaluation-and-experimental-methodology) | 8 | Versioned evaluation suite |
| 16 | [Anatomy of Frontier Language Models](#chapter-16-anatomy-of-frontier-language-models) | 6 | Source-linked model design dossiers |

## How to use the references

Each chapter names teaching sources, and each section contains annotated pointers to papers, implementations, datasets, documentation, or tutorials. The annotations explain the source's role in deriving the explanation or designing the notebook. These are an authoring bibliography: some references are for deeper reading, not dependencies that every learner must install or reproduce.

Use primary papers for algorithms and claims, official repositories/docs for implementation details, and courses/blogs for exposition and experimental organization. Pin code, dataset, and model revisions when authoring runnable notebooks. Living documentation links identify the maintained source; they are not a substitute for the notebook's tested dependency manifest. Newer report-specific mechanisms require the report's stated assumptions and should be compared against controlled baselines.

Source results are not universal promises. In particular, distinguish lossless weight reconstruction from changes in numerical execution, empirical LoRA recommendations from guarantees, speculative progress from wall-clock speedup, and reported final-model performance from causal evidence about each ingredient. Exact task-specific scores and hardware tables should live in updateable experiment outputs rather than be frozen into the conceptual outline.

## Detailed chapters

## Chapter 1 Text Tokenization and Training Data

**Purpose and prerequisites.** Explain how the representation, composition, and provenance of text determine what a language model can learn. Assume the elementary tokenizer and language-model objectives in Basics; revisit them only when production-scale choices change their behavior. The chapter produces a versioned corpus, tokenizer comparison, and reproducible data loader. Most work runs on CPU; quality-model experiments and proxy training use one GPU. Full-corpus processing is represented by bounded samples and recorded larger-run statistics.

**Chapter teaching sources.** [Stanford CS336](https://cs336.stanford.edu/) supplies tokenizer and Common Crawl assignments; [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter6/1) supplies tokenizer explanations; [FineWeb](https://arxiv.org/abs/2406.17557) and [DataComp-LM](https://arxiv.org/abs/2406.11794) supply experimental data-curation methodology. These complement, rather than replace, the existing D2L sequence-model material.

### 1.1 Text as data

**Scope.** Treat text as structured observations rather than interchangeable strings. Cover Unicode code points and bytes, normalization, whitespace, document boundaries, markup, mathematical notation, code indentation, and multilingual scripts. Explain how normalization can simplify representation while destroying distinctions. Track source identity and transformations so examples remain auditable throughout the pipeline.

**Notebook.** Build a multilingual and code-rich diagnostic corpus. Compare byte lengths, normalization changes, token counts, and round-trip decoding. Include adversarially ordinary cases such as combining marks, nonbreaking spaces, and mixed newline conventions; require reversible transformations unless a deliberate lossy rule is documented.

**References.**

- [Unicode normalization specification](https://www.unicode.org/reports/tr15/) — authoritative definitions and test cases for normalization.
- [Hugging Face tokenizer pipeline](https://huggingface.co/docs/tokenizers/pipeline) — normalizer, pre-tokenizer, model, and decoder responsibilities.
- [The Stack dataset](https://huggingface.co/datasets/bigcode/the-stack) — code-language diversity and dataset documentation; use an appropriately licensed subset.

### 1.2 Tokenization at scale

**Scope.** Compare byte BPE, WordPiece, and unigram tokenization through their objectives and inference algorithms. Explain pre-tokenization rules, vocabulary size, byte fallback, subword regularization, fertility, and the cost of longer sequences versus larger vocabulary projections. Separate unigram probability estimation from Viterbi segmentation. Do not compare raw perplexities across different tokenizers as if the prediction units were identical.

**Notebook.** Train BPE and unigram tokenizers on the same bounded corpus, then report compression, throughput, unknown-token behavior, and multilingual disparities. Use small models for downstream comparisons; retain the original tokenizer when loading pretrained weights.

**References.**

- [Sennrich et al. on subword units](https://aclanthology.org/P16-1162/) — BPE motivation and original translation setting.
- [SentencePiece](https://aclanthology.org/D18-2012/) — language-independent tokenization and practical training interface.
- [Subword Regularization](https://aclanthology.org/P18-1007/) — unigram segmentation and sampling.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — implementation-oriented comparison.

### 1.3 Changing the representation

**Scope.** Study vocabulary expansion and tokenizer transfer as changes to a trained system. Cover initialization of new input/output embeddings, preservation of old tokens, multilingual adaptation, multiword tokenization, and alternatives that allocate computation to byte patches. Explain which changes require retraining and why token counts alone do not establish a quality gain.

**Notebook.** Add a small domain vocabulary to a pretrained tokenizer and compare random versus decomposition-based embedding initialization during continued training. An optional experiment measures SuperBPE compression; a diagram and miniature byte-patching implementation introduce the architectural alternative without requiring a second full pretraining run.

**References.**

- [WECHSEL](https://aclanthology.org/2022.naacl-main.293/) — transferring pretrained models across languages through embedding initialization.
- [SuperBPE](https://arxiv.org/abs/2503.13423) and [project](https://superbpe.github.io/) — multiword vocabulary construction and controlled comparisons.
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) — entropy-model-driven byte patching and its architectural consequences.

### 1.4 Building a corpus

**Scope.** Follow a document from crawl record to training example. Explain WARC/WET distinctions, HTML extraction, boilerplate, language identification, source metadata, provenance, consent and permissions, privacy filtering, and dataset documentation. Present rights and policy metadata as engineering inputs, not as a blanket assertion that publicly reachable text is unrestricted training material.

**Notebook.** Process a small public crawl shard or supplied fixture into clean documents with source identifiers and transformation logs. Compare extractors on code, tables, and mathematics. Produce a dataset card and inspect false removals as well as retained noise.

**References.**

- [Common Crawl getting started](https://commoncrawl.org/get-started) — access and record formats.
- [Trafilatura](https://trafilatura.readthedocs.io/) — extraction implementation and configurable tradeoffs.
- [fastText language identification](https://fasttext.cc/docs/en/language-identification.html) — a reproducible language-filter baseline.
- [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) — provenance and intended-use documentation.

### 1.5 Filtering and deduplication

**Scope.** Compare heuristic rules, learned quality classifiers, exact document hashes, near-duplicate sketches, and substring deduplication. Derive MinHash/LSH collision behavior and distinguish document-level from corpus-level decisions. Include benchmark decontamination and evaluate whether a filter disproportionately removes particular languages, domains, or writing styles.

**Notebook.** Implement exact hashing and a small MinHash/LSH pipeline, then compare their matches against labeled pairs. Train a small quality scorer or use a documented pretrained scorer. Measure retained-token volume, duplicate leakage, and held-out proxy-model loss rather than optimizing a cleanliness score alone.

**References.**

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) — memorization and duplicate-removal experiments.
- [FineWeb paper](https://arxiv.org/abs/2406.17557) — filtering and deduplication ablations.
- [DataComp-LM](https://arxiv.org/abs/2406.11794) — controlled data-design comparisons.
- [Datatrove](https://github.com/huggingface/datatrove) — reference pipeline for filtering and deduplication at scale.

### 1.6 Data mixtures and curricula

**Scope.** Treat source weights as trainable design choices. Cover temperature sampling, multilingual/domain balance, proxy-model sweeps, robust domain weighting, and staged annealing. Distinguish choosing a mixture for average held-out loss from targeting a deployment distribution. Keep compute, tokenizer, and evaluation sets fixed when attributing gains to mixture selection.

**Notebook.** Train small proxy models on three domains under several fixed mixtures. Compare a simple regression-selected mixture with a uniform baseline and a staged schedule. Report per-domain losses and uncertainty; explicitly test whether proxy rankings transfer to a somewhat larger model.

**References.**

- [DoReMi](https://arxiv.org/abs/2305.10429) — robust optimization of domain weights.
- [RegMix](https://arxiv.org/abs/2407.01492) — regression over proxy mixture experiments.
- [OLMo 2](https://arxiv.org/abs/2501.00656) — staged data and training recipe as an integrated case study.
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) — practical experiment planning and data-mixture discussion.

### 1.7 Synthetic training data

**Scope.** Distinguish generating new tasks, rewriting existing text, producing demonstrations, and labeling quality. Examine verifier accuracy, diversity, source attribution, teacher biases, and recursive feedback. Synthetic data should be evaluated by its incremental effect on a held-out task distribution, not by the apparent fluency of generated examples.

**Notebook.** Create a bounded synthetic augmentation for a documented domain using a local pretrained model. Compare original-only, synthetic-only, and mixed training at equal token budgets. Measure factual errors and near-duplicates, and preserve the prompts and teacher revision used to create the data.

**References.**

- [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) — accessible synthetic corpus and generation documentation.
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) — a primary synthetic-data case study, with claims limited to its setting.
- [Nemotron-CC](https://arxiv.org/abs/2412.02595) — combining web curation with synthetic transformations.

### 1.8 Data infrastructure

**Scope.** Connect data quality to reliable execution. Cover shard formats, streaming, tokenization caches, sequence packing, document masks, sampling across workers, I/O overlap, and versioned manifests. Define what exact resumption means for shuffled streaming data and distinguish dataset reproducibility from deterministic numerical training.

**Notebook.** Build a streaming dataset with deterministic worker partitioning and a resumable cursor. Interrupt and restart a small training run, checking that the next example IDs and token batches match. Profile input starvation and compare uncompressed versus compressed shards.

**References.**

- [Hugging Face dataset streaming](https://huggingface.co/docs/datasets/stream) — iterable datasets, sharding, and streaming operations.
- [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html) — worker behavior and iterable-dataset responsibilities.
- [TorchData stateful data loader](https://github.com/pytorch/data) — stateful loading and restart implementation reference.
- [Datatrove](https://github.com/huggingface/datatrove) — pipeline metadata and scalable processing patterns.

## Chapter 2 Modern Language Model Architectures

**Purpose and prerequisites.** Turn the transformer knowledge from Advanced into the ability to read, inspect, and modify real model implementations. The main artifact is a checkpoint inspection notebook with verified component replacements. Most experiments use a 1–3B checkpoint on one GPU; larger models are inspected through configuration and tensor metadata. Architecture changes are distinguished from optimizations that preserve the existing function.

**Chapter teaching sources.** [CS336](https://cs336.stanford.edu/) provides architecture/resource-accounting lectures; [Raschka's from-scratch materials](https://www.sebastianraschka.com/from-scratch/) supply readable PyTorch components; [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) provides checkpoint implementations and loading conventions. Chapter 16 returns to integrated frontier-model designs.

### 2.1 Reading a model checkpoint

**Scope.** Interpret configuration fields, state-dict keys, tensor shapes, vocabulary projections, tied parameters, numerical formats, and generation defaults. Explain the relationships among model weights, tokenizer, chat template, architecture code, and revision identifiers. Teach safe inspection before loading large tensors or executing custom repository code.

**Notebook.** Inspect a small Qwen-family checkpoint and a larger model's metadata. Reconstruct the parameter budget from shapes and compare it with the model card. Replace one projection with an equivalent PyTorch implementation and verify logits within a documented numerical tolerance.

**References.**

- [Qwen3 model report](https://arxiv.org/abs/2505.09388) — family-level architectural context.
- [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B) — candidate practical checkpoint; pin a revision before authoring.
- [Safetensors documentation](https://huggingface.co/docs/safetensors/index) — weight-file representation and metadata.
- [PyTorch serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html) — state dictionaries and loading semantics.

### 2.2 Objectives and model families

**Scope.** Compare causal prediction, masked reconstruction, span corruption, and conditional sequence generation. Connect each factorization to visible context, output representation, training examples, and downstream tasks. Preserve encoders and encoder–decoders as useful alternatives rather than treating every language problem as chat completion.

**Notebook.** Construct masks and losses for the same tiny text batch under causal, masked, and span-corruption objectives. Fine-tune an accessible pretrained encoder or encoder–decoder on a bounded task and compare output structure and resource use with a causal model.

**References.**

- [BERT](https://arxiv.org/abs/1810.04805) — masked-language-model objective and encoder transfer.
- [T5](https://arxiv.org/abs/1910.10683) — text-to-text framing and span corruption.
- [Hugging Face language-task course](https://huggingface.co/learn/llm-course/en/chapter7/1) — complete task examples to adapt into explicit PyTorch loops.

### 2.3 The contemporary transformer block

**Scope.** Examine normalization placement, RMS normalization, gated MLPs, query/key normalization, residual branches, and initialization. Trace how numerical ranges and parameter allocation change with each choice. Attention Residuals and related depth-wise mixing belong as frontier extensions, with their proposed benefit separated from general architectural guarantees.

**Notebook.** Implement interchangeable block components and compare activation/gradient statistics in a small model. Verify an unchanged block against pretrained outputs before making interventions. Use controlled ablations for trainable changes rather than expecting arbitrary checkpoint surgery to preserve quality.

**References.**

- [RMSNorm](https://arxiv.org/abs/1910.07467) — normalization derivation.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — gated feed-forward alternatives.
- [OLMo 2](https://arxiv.org/abs/2501.00656) — normalization and stability in a complete recipe.
- [Kimi K3 report](https://arxiv.org/html/2607.24653v1) — Attention Residuals and associated design details.

### 2.4 Positional information and context extension

**Scope.** Develop RoPE geometry and distinguish interpolation, extrapolation, and frequency rescaling. Cover partial positional encoding and layer-specific attention patterns. Context extension requires both a positional mechanism and appropriate training/evaluation; accepting a longer tensor is not evidence of using distant information reliably.

**Notebook.** Visualize relative phases, compare rotary implementations, and evaluate a pretrained model near and beyond its training context. Perform a modest context-extension experiment on a small checkpoint, using retrieval and composition tasks in addition to language-model loss.

**References.**

- [RoFormer](https://arxiv.org/abs/2104.09864) — rotary positional embedding formulation.
- [Position Interpolation](https://arxiv.org/abs/2306.15595) — extension through position scaling.
- [YaRN](https://arxiv.org/abs/2309.00071) — frequency/attention adjustments and extension recipe.
- [Llama 3 report](https://arxiv.org/abs/2407.21783) — long-context training within a broader model recipe.

### 2.5 Local global and asymmetric computation

**Scope.** Compare local windows, occasional global layers, persistent sink positions, and asymmetric processing of input and output. Separate changes to attention connectivity from methods that reuse equivalent computations. Use causal encoder–decoder designs as a contemporary case study; avoid conflating them with the conventional bidirectional source encoder in translation.

**Notebook.** Implement window/global masks and trace the resulting receptive fields. Compare effective retrieval with theoretical reach. Build a reduced asymmetric model and count prefill, decode, and cached-state costs, leaving full-scale quality conclusions to published ablations.

**References.**

- [Mistral 7B](https://arxiv.org/abs/2310.06825) — sliding-window architecture case study.
- [StreamingLLM](https://arxiv.org/abs/2309.17453) — attention sinks and streaming cache behavior.
- [You Only Cache Once](https://arxiv.org/abs/2405.05254) — decoder–decoder cache-sharing architecture.
- [DeepSeek-V4.1-Flash report](https://arxiv.org/html/2609.19969v1) — causal encoder–decoder and cross-layer cache reuse.

### 2.6 Auxiliary prediction and conditional memory

**Scope.** Explore objectives that predict beyond the next token and conditional memory that retrieves learned representations without evaluating a full MLP. Distinguish auxiliary training supervision from the inference-time draft mechanism. Account for the memory capacity, lookup traffic, and synchronization of conditional memory rather than labeling lookup computation free.

**Notebook.** Add a small auxiliary prediction head to the reference model and evaluate whether its predictions support useful drafting. Implement a miniature hashed n-gram memory with gated integration, measuring collisions, parameter storage, and lookup latency separately from language-model quality.

**References.**

- [Better and Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) — auxiliary prediction objectives.
- [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) — multi-token prediction in a production training recipe.
- [DeepSeek-V4.1-Flash report](https://arxiv.org/html/2609.19969v1) — DSpark and Engram sections for advanced design comparison.

### 2.7 Text embeddings and representation models

**Scope.** One substantial notebook distinguishes token embeddings from whole-text representations. Cover contextual states, pooling, normalization, query/document asymmetry, contrastive objectives, in-batch and hard negatives, and dimensionality tradeoffs. Compare a pretrained encoder with decoder-derived representations. Historical word2vec/GloVe material is a sidebar and optional reading, not another required chapter.

**Notebook.** Retrieve passages from the book using a pretrained embedding model. Implement normalized similarity search in PyTorch and evaluate held-out queries with Recall@k and ranking metrics. Adapt on a small contrastive dataset, inspect hard negatives, and measure domain transfer and representation-size tradeoffs.

**References.**

- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — sentence representation and pairwise training baseline.
- [ModernBERT](https://arxiv.org/abs/2412.13663) — contemporary encoder architecture; distinguish a general encoder checkpoint from one trained for retrieval.
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — practical decoder-derived embedding checkpoint, with explicit PyTorch pooling/normalization and query-instruction examples.
- [LLM2Vec](https://arxiv.org/abs/2404.05961) — decoder-to-encoder adaptation, beyond changing a mask.
- [BEIR](https://github.com/beir-cellar/beir) — heterogeneous retrieval datasets and metrics.
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — nested representation dimensions as an extension exercise.

### 2.8 Alternative text generation objectives

**Scope.** Introduce masked/discrete diffusion language modeling by building on Advanced's diffusion treatment. Compare denoising objectives, visible context, token remasking, and parallel generation with autoregressive factorization. Examine likelihood estimation and fair compute comparisons. Published speed or reversal-task improvements are evaluated within their settings rather than treated as properties of all diffusion models.

**Notebook.** Implement a small masked denoiser and confidence-based decoding loop. Compare generation steps, model evaluations, wall time, and task quality at matched model scale. Use a pretrained diffusion checkpoint only as an optional heavier extension.

**References.**

- [D3PM](https://arxiv.org/abs/2107.03006) — structured discrete diffusion foundations.
- [LLaDA](https://arxiv.org/abs/2502.09992) — language-scale masked diffusion formulation.
- [LLaDA implementation](https://github.com/ML-GSAI/LLaDA) — model and sampling reference.

## Chapter 3 Sparse Recurrent and Hybrid Models

**Purpose and prerequisites.** Explain how architecture changes the amount of computation and state required for a given workload. Assume transformer, attention, and state-space foundations from Advanced and Chapter 2. Build small runnable components, then use configurations and reports for frontier-scale accounting. The principal artifact is a common comparison harness for dense, MoE, latent-attention, recurrent, and hybrid components; it does not claim small-model experiments reproduce frontier quality.

**Chapter teaching sources.** [CS336](https://cs336.stanford.edu/) covers attention alternatives and MoE; [Smola's efficiency slides](https://alex.smola.org/posts/45-mlss-efficiency/) connect architectural state to execution costs; [Schlag's supplied systems decks](https://drive.google.com/drive/folders/1pUG7TB5VRMQZSa9-9wxkGOZEzHjP8LC3) connect routing and expert communication. Use the Week 5 routing slides 37–48 and Week 6 state-budget slides 16–19.

### 3.1 Sparse capacity

**Scope.** Distinguish total parameters, active parameters, stored bytes, and executed operations. Explain fine-grained experts, shared experts, and how conditional computation changes capacity without eliminating storage or communication. Define matched-parameter, matched-compute, and matched-latency comparisons as different experiments rather than interchangeable fairness criteria.

**Notebook.** Implement an accounting tool for a dense model and a small MoE. Sweep expert count and selected experts, keeping active MLP width fixed where possible. Compare theoretical FLOPs with wall time and show how small expert batches can erase anticipated gains.

**References.**

- [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538) — conditional-computation formulation.
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — sparse capacity and simplified routing.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — fine-grained and shared-expert design.

### 3.2 Mixture of experts routing

**Scope.** Derive router scores, selection, gate weights, and the weighted combination of expert outputs. Explain token duplication, expert capacity, overflow handling, and dropless execution. Keep statistical routing decisions distinct from the dispatch implementation developed in Chapter 7. Introduce a clear accounting convention for tokens routed to multiple experts.

**Notebook.** Implement a transparent top-k MoE using ordinary PyTorch operations, then compare with grouped execution. Test routing permutation/inverse-permutation correctness, gate normalization, overflow behavior, and gradients on a tiny deterministic batch.

**References.**

- [Switch Transformers](https://arxiv.org/abs/2101.03961) — capacity and routing mechanics.
- [MegaBlocks](https://arxiv.org/abs/2211.15841) — dropless sparse computation.
- [MegaBlocks code](https://github.com/databricks/megablocks) — implementation reference, not a required notebook dependency.

### 3.3 Keeping experts useful and balanced

**Scope.** Study the feedback loop between expert selection, expert training, and future routing. Compare auxiliary losses, selection biases, and quantile-based balancing; distinguish load balance from logit stability and specialization. Explain local versus global balancing measurements and why perfectly equal token counts need not maximize model quality or execution efficiency.

**Notebook.** Train a small MoE on mixed domains, log expert utilization and routing entropy, and compare balancing rules. Include a controlled skewed-domain experiment and inspect whether balance hides poor specialization. Measure both overflow and the slowest expert's workload.

**References.**

- [ST-MoE](https://arxiv.org/abs/2202.08906) — router stability and training design.
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) — selection-bias approach.
- [Kimi K3 report](https://arxiv.org/html/2607.24653v1) — quantile balancing and Stable LatentMoE design.

### 3.4 Latent attention

**Scope.** Derive joint low-rank key/value representations, decoupled positional components, and absorption of projections into query/output matrices. Distinguish prefill execution from decode execution and account for temporary materialization. State the assumptions under which two implementations are algebraically equivalent; a smaller persistent cache can still change arithmetic and kernel efficiency.

**Notebook.** Build a small MLA layer with explicit and absorbed decode paths. Verify outputs and gradients in high precision, then quantify lower-precision deviations. Compare persistent state, temporary memory, and latency with MHA/GQA at several sequence lengths.

**References.**

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA formulation and decoupled RoPE.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — MLA within the larger architecture.
- [Smola efficiency slides](https://alex.smola.org/posts/45-mlss-efficiency/) — matrix absorption and byte-budget teaching examples.

### 3.5 Sparse attention

**Scope.** Separate local patterns, block selection, and learned indexing. Account for index construction/scoring as well as selected attention; selecting k positions does not automatically make the entire operation independent of context length. Discuss train–inference consistency, hardware alignment, and comparisons against exact attention.

**Notebook.** Implement window-plus-selected-block attention in PyTorch and compare with full attention on retrieval and compositional tasks. Profile selection overhead separately. Use a learned indexer only in a small controlled model; illustrate larger designs through their published execution diagrams.

**References.**

- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) — hardware-aligned sparse attention branches.
- [MoBA](https://arxiv.org/abs/2502.13189) — block routing over context.
- [DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1) — compressed sparse attention and hierarchical indexing as a frontier case.

### 3.6 Recurrent and delta rule language models

**Scope.** Extend the state-space material in Advanced to matrix-state recurrences, delta updates, and gating. Explain recurrent inference versus chunkwise parallel training, stability, and state-size accounting. Relate state compression to the information that must be retained, without treating fixed-state models as universally incapable of long-range tasks.

**Notebook.** Implement a small gated delta-rule layer in recurrent and chunked forms and test agreement. Measure state size, training throughput, and retrieval behavior under controlled changes in sequence length and memory demand.

**References.**

- [Mamba](https://arxiv.org/abs/2312.00752) — selective state-space design.
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) — state-space duality and parallel computation.
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — gating plus delta-rule updates.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) — efficient implementation reference and correctness comparisons.

### 3.7 Hybrid architectures

**Scope.** Combine recurrent layers with full, grouped-query, or latent attention. Analyze the allocation of attention layers, state capacities, and periodic global access. Treat layer ratios as design variables rather than a universal frontier recipe. Connect the design to training kernels and serving state management, revisited in Chapters 7 and 13.

**Notebook.** Compare small pure-attention, pure-recurrent, and hybrid models under matched compute and state budgets. Use both language-model loss and retrieval tasks, documenting the different conclusions each metric supports. Inspect a frontier hybrid configuration without loading its full weights.

**References.**

- [Jamba](https://arxiv.org/abs/2403.19887) — hybrid attention/state-space/MoE example.
- [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3) — inspectable hybrid configuration and precision metadata.
- [Kimi K3 technical report](https://arxiv.org/abs/2607.24653) — hybrid design and reported ablations.

### 3.8 Changing an existing architecture

**Scope.** Cover dense-to-MoE upcycling, attention conversion, and recovery training. Explain which weights can be copied, duplicated, factored, or newly initialized and how the resulting model changes capacity and inductive bias. Distinguish function-preserving initialization from long-run quality preservation and account for recovery data and compute.

**Notebook.** Upcycle a tiny dense MLP into several experts, test the initial function where the construction permits, and continue training with different routing initializations. An optional low-rank KV conversion demonstrates why checkpoint surgery needs held-out recovery evaluation.

**References.**

- [Sparse Upcycling](https://arxiv.org/abs/2212.05055) — reusing dense checkpoints to initialize sparse models.
- [MHA2MLA](https://arxiv.org/abs/2502.14837) — converting pretrained attention to latent attention.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — expert design choices to compare against duplicated dense blocks.

### 3.9 Architecture experiments

**Scope.** Establish experimental discipline for architectural comparisons. Track parameter count, active FLOPs, optimizer state, inference cache, real latency, and data exposure. Separate the effect of an architecture from its training recipe and kernel maturity. Discuss confidence intervals, multiple ablations, and the limits of extrapolating from tiny models.

**Notebook.** Build a common report for the chapter's models with quality–memory–latency plots and explicit matching rules. Include a counterexample where the FLOP-efficient variant is slower, and a context task where average loss conceals a capability regression.

**References.**

- [Stanford CS336 scaling assignment](https://cs336.stanford.edu/) — controlled component and scaling experiments.
- [HELM](https://arxiv.org/abs/2211.09110) — multi-metric evaluation discipline.
- [RULER](https://arxiv.org/abs/2404.06654) — effective-context diagnostic tasks.

## Chapter 4 Hardware and Performance Models for Language Models

**Purpose and prerequisites.** Teach readers to predict feasibility and identify bottlenecks before optimizing code. Assume the general computation/performance chapter in Advanced; deepen its treatment for language workloads. The artifact is a hardware/workload calculator paired with measured traces. CPU calculations and supplied profiles support every section; one GPU enables most measurements, and multi-GPU observations are optional.

**Chapter teaching sources.** [Smola's MLSS efficiency material](https://alex.smola.org/posts/45-mlss-efficiency/) supplies the prefill/decode and memory-budget narrative; [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) supplies systems cost models, translated into PyTorch experiments; [MIT 6.5940](https://hanlab.mit.edu/courses/2026-fall-65940) supplies efficiency/compression pedagogy. [Schlag's Week 3 deck](https://drive.google.com/file/d/1LeR_e8y9m7gosaGQ5Aeo-m23hG-aovJx/view), especially slides 3–9 and 25–36, supplies scaling and network measurements. Hardware tables should carry measurement dates and configurations.

### 4.1 The language model workload

**Scope.** Decompose pretraining, prefill, decode, teacher scoring, and rollout generation into matrix operations and state movement. Explain how batch size, sequence length, sparsity, and cache representation change the bottleneck. Prefill/decode labels are useful starting points, not unconditional assignments to compute-bound and bandwidth-bound regimes.

**Notebook.** Run a selected pretrained model at several prompt lengths and batch sizes. Separate tokenizer/loading time from model execution, then measure prefill and decode independently. Compare predicted work with kernel traces and identify the regime transitions.

**References.**

- [Smola efficiency tutorial](https://alex.smola.org/posts/45-mlss-efficiency/) — workload decomposition and worked inference examples.
- [Scaling Book](https://jax-ml.github.io/scaling-book/) — transformer operation and communication accounting.
- [PyTorch profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — practical tracing workflow.

### 4.2 Compute and memory accounting

**Scope.** Count weights, master weights where present, gradients, optimizer moments, saved activations, KV state, and temporary buffers. Distinguish allocated, reserved, peak, and persistent memory. Include logits and attention workspaces. Derive budgets for the actual numerical/optimizer configuration rather than imposing a universal bytes-per-parameter rule.

**Notebook.** Predict a model's inference footprint and training-step peak, then compare with a PyTorch memory snapshot. Vary context length, batch size, optimizer, and checkpointing. Show why fitting BF16 weights into 24GB does not imply full fine-tuning or long-context inference will fit.

**References.**

- [PyTorch CUDA memory notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) — allocator terminology and measurement APIs.
- [Understanding GPU memory](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — snapshots and memory-history analysis.
- [Schlag Week 4 deck](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — slides 2–5 and 13–16 connect predicted memory lifetimes to measurements.

### 4.3 Accelerator execution

**Scope.** Explain matrix tiling, tensor cores, data reuse, occupancy, launch overhead, and shape-dependent efficiency. Trace the difference between a large regular GEMM and many small expert GEMMs. Use diagrams and PyTorch operations for the required material; kernel source is an optional reference rather than a second implementation language requirement.

**Notebook.** Sweep matrix dimensions and batch/token counts, measuring achieved throughput after warmup and synchronization. Compare contiguous and unfavorable layouts. Relate the results to actual projection shapes in the pretrained model rather than reporting a single peak benchmark.

**References.**

- [NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) — shape, tiling, and arithmetic-intensity reasoning.
- [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — authoritative architecture concepts for optional deeper reading.
- [Schlag Week 4 deck](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — slides 6–7 show shape-dependent throughput.

### 4.4 The memory hierarchy

**Scope.** Connect registers, SRAM, caches, HBM, host memory, and storage through capacity, latency, and bandwidth. Explain sequential transfers, reuse, data layout, and asynchronous copies. Treat energy figures and silicon-geometric analogies as qualified explanatory models, not precise forecasts for every future chip.

**Notebook.** Measure representative copy and compute operations and build a dated memory ladder for the available device. Estimate the break-even point between retaining, transferring, and recomputing state. Use supplied measurements where access to host-interconnect or storage hardware is unavailable.

**References.**

- [Smola efficiency slides](https://alex.smola.org/posts/45-mlss-efficiency/) — memory and latency ladders with language-model examples.
- [FlashAttention](https://arxiv.org/abs/2205.14135) — a concrete algorithm designed around memory movement.
- [Scaling Book rooflines](https://jax-ml.github.io/scaling-book/roofline/) — resource-bound execution models.

### 4.5 Numerical formats

**Scope.** Describe exponent, mantissa, dynamic range, subnormals, rounding, and accumulation using FP32, TF32, BF16, FP16, FP8, and block-scaled low-bit formats. Separate storage format from multiply and accumulation formats. Introduce numerical error here; defer quantization algorithms and training recipes to Chapters 12 and 5.

**Notebook.** Implement small format simulators and block scaling in PyTorch. Compare reconstruction error, underflow/overflow, and matrix-product error for realistic activations. Label simulated arithmetic clearly; speed measurements require hardware and kernels supporting the chosen format.

**References.**

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — format design and range/precision tradeoffs.
- [Microscaling data formats](https://arxiv.org/abs/2310.10537) — shared scales and low-bit arithmetic.
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — precision and non-associativity in actual execution.
- [TorchAO](https://github.com/pytorch/ao) — supported PyTorch low-precision representations and operations.

### 4.6 Rooflines and sustained performance

**Scope.** Combine operation counts and byte movement into latency bounds, then examine why achieved performance falls below those bounds. Cover arithmetic intensity, launch overhead, power limits, communication exposure, and shape effects. Define model FLOPs utilization with its accounting assumptions, including extra attention work, sparsity, and recomputation.

**Notebook.** Plot measured kernels and model phases against a roofline based on measured sustained bandwidth and compute. Compare this with a specification-based roofline. Explain each major discrepancy using profiles, and avoid interpreting unused peak compute as a single universal overhead category.

**References.**

- [Roofline model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) — original performance-model exposition.
- [Scaling Book rooflines](https://jax-ml.github.io/scaling-book/roofline/) — modern accelerator examples.
- [Schlag Week 3 deck](https://drive.google.com/file/d/1LeR_e8y9m7gosaGQ5Aeo-m23hG-aovJx/view) — slides 4–9 contrast scaling, MFU, and sustained performance.

### 4.7 Nodes racks and networks

**Scope.** Introduce scale-up and scale-out domains, point-to-point versus switched links, network cards, oversubscription, and topology-sensitive communication. Explain why tensor/expert/context parallel placement depends on transfer frequency and payload. Detailed collective algorithms follow in Chapter 6; this section establishes the hardware constraints they must respect.

**Notebook.** Describe a small cluster as devices and links, then estimate the cost of alternative process-group placements. Where multiple GPUs are available, compare intra-node and inter-node transfers. Include a diagnostic example of an unintended slow transport path.

**References.**

- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — communication library and topology-aware execution.
- [Scaling Book sharding](https://jax-ml.github.io/scaling-book/sharding/) — placement and communication models.
- [Schlag Week 3 deck](https://drive.google.com/file/d/1LeR_e8y9m7gosaGQ5Aeo-m23hG-aovJx/view) — slides 34–47 connect topology, runtime setup, and measured collectives.

### 4.8 Choosing hardware for a workload

**Scope.** Translate quality, context, concurrency, latency, and training requirements into capacity and throughput constraints. Compare consumer GPUs, unified-memory systems, and datacenter accelerators through workload needs. Account for utilization and total operating cost without turning the chapter into a transient buying guide. Maintain dated device tables separately from the durable reasoning.

**Notebook.** Build a configurable capacity/performance calculator for dense and sparse models. Evaluate several hardware profiles using dated specifications and measured efficiencies. Produce separate choices for single-user decode, batched serving, adaptation, and distributed training, with sensitivity analysis rather than false precision.

**References.**

- [Smola efficiency tutorial](https://alex.smola.org/posts/45-mlss-efficiency/) — local inference and accelerator/model matching.
- [Scaling Book](https://jax-ml.github.io/scaling-book/) — inference and training cost models across configurations.
- [MLCommons inference benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/) — standardized results and scenario definitions for comparative measurement.

## Chapter 5 Pretraining and Continued Training

**Purpose.** Develop a reproducible training process and use controlled experiments to connect data, optimization, numerical precision, and downstream capability. Assume Chapters 1–4 and the optimization material in Advanced. The main artifacts are a small model trained from scratch, a continued-training run on a capable pretrained checkpoint, and a training report with loss, throughput, memory, and restart checks. Default exercises use one GPU; scaling studies use small runs plus supplied larger-run measurements.

**Chapter teaching sources.** [Stanford CS336](https://cs336.stanford.edu/) supplies a progression from implementation to scaling experiments. [OLMo 2](https://arxiv.org/abs/2501.00656) provides an open training-recipe case study. [TorchTitan](https://github.com/pytorch/torchtitan) supplies a maintained PyTorch training system to inspect after the explicit loop is understood. [Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt) is useful for the incremental implementation style, not as a current frontier recipe.

### 5.1 A complete PyTorch pretraining loop

**Scope.** Construct the causal objective, token shifts, document masks, packing, optimizer step, evaluation interval, and checkpoint boundary. Separate the mathematical loss from implementation choices such as reduction over tokens or sequences. Track consumed tokens and data order explicitly. The first implementation should remain readable enough to identify precisely which operation changes parameters or random state.

**Notebook.** Train a small decoder on a pinned corpus slice. Check shifted targets and document isolation on hand-constructed examples; overfit a tiny batch; then compare one accumulated update with an equivalent full-batch update. Save model, optimizer, scheduler, RNG, and sampler state.

**References.** [CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) — component and training-loop exercises; [build-nanogpt](https://github.com/karpathy/build-nanogpt) — progressive baseline; [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — candidate versioned training subset; [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) — optimizer semantics.

### 5.2 Initialization and optimizer choices

**Scope.** Apply AdamW and selected matrix-aware optimizer ideas to language-model parameter groups. Explain initialization, residual scaling, decay exclusions, epsilon, moment precision, and why embeddings or normalization parameters may require different treatment. Distinguish optimizer changes from learning-rate and batch changes; preserve the earlier optimization chapter's notation rather than rederive generic optimization.

**Notebook.** Compare a small set of optimizer configurations under matched tokens and compute. Plot update-to-weight ratios by module, gradient statistics, and held-out loss. Include a deliberately unstable configuration and diagnose it before changing several hyperparameters at once.

**References.** [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — AdamW objective and implementation distinction; [Muon implementation and explanation](https://github.com/KellerJordan/Muon) — matrix update reference; [OLMo 2](https://arxiv.org/abs/2501.00656) — language-model stability recipe; [Kimi K3 report](https://arxiv.org/html/2607.24653v1) — a frontier optimizer/architecture interaction to analyze.

### 5.3 Batch size and training schedules

**Scope.** Relate microbatch, accumulation, global batch, optimizer steps, and token budgets. Compare warmup, cosine decay, warmup–stable–decay, and batch ramp-up. Explain that throughput improvements and optimization improvements are different outcomes. Discuss the relationship between schedule horizon, branching experiments from a shared checkpoint, and changes in the training-data mixture.

**Notebook.** Run several small token-matched batch/schedule experiments. Report loss against tokens, steps, and elapsed time; inspect whether accumulation changes numerical behavior. Branch decay phases from a shared stable-phase checkpoint and compare their final quality at different budgets.

**References.** [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) — gradient-noise-scale perspective; [Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192) — schedule analysis; [ETH data and tensor parallelism slides](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — printed slides 8–12 for batch and accumulation; [PyTorch learning-rate schedulers](https://docs.pytorch.org/docs/stable/optim.html) — practical scheduler behavior.

### 5.4 Scaling experiments

**Scope.** Fit empirical relationships between model size, tokens, compute, and loss. Cover iso-compute design, uncertainty, held-out prediction, and the limits of fitting a few noisy small runs. Introduce maximum-update parameterization as a conditional hyperparameter-transfer method. Explicitly distinguish an explanatory toy scaling fit from a reliable forecast of frontier-scale capability.

**Notebook.** Train a grid of small models, fit a scaling relation, and test a withheld configuration. Show sensitivity to the fitting range and data mixture. Add width-transfer coordinate checks before attempting a learning-rate transfer experiment.

**References.** [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla experimental design; [Tensor Programs V](https://arxiv.org/abs/2203.03466) — hyperparameter transfer; [muP implementation](https://github.com/microsoft/mup) — coordinate-check and implementation guidance; [CS336 scaling assignment](https://github.com/stanford-cs336/assignment3-scaling) — constrained-budget exercise pattern.

### 5.5 Training for downstream cost

**Scope.** Extend training-compute optimization to include future inference volume, memory capacity, latency, and hardware utilization. Compare a smaller model trained longer with a larger model trained less. Treat expected deployment demand as an assumption to vary, and distinguish parameter-count proxies from measured serving costs. Link the eventual decision to Chapters 12–14.

**Notebook.** Build a lifecycle-cost calculator using measured inference rates for candidate checkpoints and fitted pretraining curves. Vary inference volume and latency requirements, then identify where the preferred model changes. Include sensitivity bands rather than one supposedly universal optimum.

**References.** [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) — inference-aware scaling; [Chinchilla](https://arxiv.org/abs/2203.15556) — training-only baseline; [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) — accessible small-model case; [Smola efficiency tutorial](https://alex.smola.org/posts/45-mlss-efficiency/) — deployment cost-model framing.

### 5.6 Low-precision training

**Scope.** Explain which values are stored and accumulated in each format, and how scaling, rounding, outliers, and reduction order affect gradients. Progress from BF16/FP16 mixed precision to selected FP8 and FP4 recipes. Separate numerical simulation from hardware-accelerated execution, and distinguish inference calibration from training-time error propagation and optimizer-state precision.

**Notebook.** Compare a high-precision reference step with lower-precision variants using activation, gradient, and update error measurements. Simulate block scaling on any device; run actual accelerated paths only on supported hardware. Include overflow, underflow, and rounding diagnostics rather than only final loss.

**References.** [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — foundational numerical recipe; [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) — fine-grained FP8 case study; [torchao](https://github.com/pytorch/ao) — PyTorch low-precision implementations; [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) — supported formats and practical recipe comparison.

### 5.7 Activation memory and recomputation

**Scope.** Follow saved activations and gradients through a complete step, including attention intermediates, MLP outputs, vocabulary logits, and temporary allocations. Contrast full-block and selective recomputation. Explain why wrapping an operation in a checkpoint does not necessarily eliminate its output when a downstream backward operation must retain that tensor.

**Notebook.** Record memory after each block in forward and backward, then predict the peak from tensor lifetimes. Compare no checkpointing, block checkpointing, and selective policies. Verify gradients within appropriate tolerances and report extra compute alongside peak-memory reductions.

**References.** [ETH data/tensor slides](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — printed slides 2–5 and 13–16, especially the selective-recomputation example; [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198) — model-level analysis; [PyTorch checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html) — current checkpoint semantics; [PyTorch activation-checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/) — selective policies and memory-budget discussion.

### 5.8 Continued training

**Scope.** Continue a pretrained base model on domain or language data, including staged mixtures and longer contexts. Explain learning-rate rewarming, replay, and forgetting as experimental choices rather than fixed recipes. Separate continued next-token training from instruction adaptation in Chapter 8, while showing how checkpoints and data mixtures connect the stages.

**Notebook.** Continue a small pretrained model on a licensed technical corpus slice. Evaluate domain loss, general-domain loss, and held-out tasks before and after adaptation; compare domain-only training with replay. Keep tokenizer and serialization fixed to isolate the training intervention.

**References.** [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) — domain/task adaptive pretraining; [OLMo 2](https://arxiv.org/abs/2501.00656) — staged data and training case study; [YaRN](https://arxiv.org/abs/2309.00071) — context-extension mechanism; [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — large-scale training and context-extension reference.

### 5.9 Diagnosing and recovering runs

**Scope.** Establish a diagnosis protocol for loss spikes, bad batches, NaNs, gradient anomalies, performance drops, and non-reproducible restarts. Track model, optimizer, data-loader, and random states separately. Explain how checkpoint selection can overfit evaluations and why an apparently resumed run may have silently changed its sample order or schedule.

**Notebook.** Inject a malformed batch, a numerical failure, and a mid-run interruption. Diagnose from logs, restore a checkpoint, and compare the resumed trajectory with uninterrupted training. Record tolerances and the sources of unavoidable nondeterminism on the selected backend.

**References.** [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) — determinism boundaries; [OLMo training code](https://github.com/allenai/OLMo) — practical state and diagnostics; [TorchTitan](https://github.com/pytorch/torchtitan) — training/checkpoint integration; [Llama 3 report](https://arxiv.org/abs/2407.21783) — operational failure case study.

## Chapter 6 Distributed Computation and Sharded Training

**Purpose.** Derive how tensors, gradients, and optimizer states move between devices. Assume Chapters 4–5 and linear algebra/autograd from Basics. The artifact is a correctness-checked distributed trainer with a communication and memory report. CPU multi-process examples teach semantics; two or more GPUs are optional for measurements. Do not imply CPU simulations predict accelerator performance.

**Chapter teaching sources.** [ETH collective-communication deck](https://drive.google.com/file/d/1LeR_e8y9m7gosaGQ5Aeo-m23hG-aovJx/view) supplies stepwise collective diagrams; [ETH data/tensor deck](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) connects tensor ownership to differentiation; [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) provides integrated training context; [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html) is the API reference, pinned when notebooks are authored.

### 6.1 Processes and collectives

**Scope.** Introduce rank, local rank, world size, device assignment, process groups, rendezvous, and point-to-point versus collective operations. Explain collective ordering and compatible shapes, including the difference between a programming deadlock and a slow transfer. Present a minimal launch script before any high-level distributed wrapper.

**Notebook.** Run broadcast, all-gather, reduce-scatter, and all-reduce on labeled tensors. Compare expected values across ranks and add explicit timeouts and rank-tagged logs. Demonstrate a safely bounded collective-order mismatch, then repair it.

**References.** [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html) — operation contracts; [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) — launcher/environment semantics; [ETH collective deck](https://drive.google.com/file/d/1LeR_e8y9m7gosaGQ5Aeo-m23hG-aovJx/view) — slides 11–23 and 42–45.

### 6.2 Communication cost models

**Scope.** Develop latency plus bytes-over-bandwidth models and derive ring all-reduce from reduce-scatter and all-gather. Contrast ring and tree behavior as message size and rank count change. Distinguish payload throughput, algorithm bandwidth, and normalized bus bandwidth; topology and contention prevent these from being interchangeable physical measurements.

**Notebook.** Benchmark a range of message sizes with warmup and synchronized timing. Fit latency/bandwidth terms, compare predicted and measured all-reduce times, and explain departures at small messages or topology boundaries. A supplied trace supports readers without multiple GPUs.

**References.** [ETH collective deck](https://drive.google.com/file/d/1LeR_e8y9m7gosaGQ5Aeo-m23hG-aovJx/view) — slides 25–32 for models and metrics; [NCCL tests](https://github.com/NVIDIA/nccl-tests) — benchmark methodology and bandwidth conventions; [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) — algorithms, environment, and troubleshooting.

### 6.3 Data parallelism from first principles

**Scope.** Replicate parameters, split examples, compute local gradients, and combine them into a global update. Derive normalization for unequal token counts, accumulation, and distributed sampling. Address repeated or dropped final batches and the distinction between averaging per-rank means and computing the desired global token-average objective.

**Notebook.** Compare one-device and multi-process training on identical examples, first with equal and then unequal token counts. Check loss, gradients, and post-update parameters. Add gradient accumulation and verify the effective batch calculation without changing the objective.

**References.** [PyTorch Distributed paper](https://arxiv.org/abs/2006.15704) — DDP design; [DistributedDataParallel API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — synchronization behavior; [ETH data/tensor deck](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — slides 17–21; [DistributedSampler](https://docs.pytorch.org/docs/stable/data.html) — sampling reference.

### 6.4 Overlapping communication

**Scope.** Connect gradient readiness to communication scheduling. Explain bucketing, kernel launch overhead, stream dependencies, accumulation synchronization, and the critical path. An asynchronous operation only permits overlap; it does not guarantee that the workload contains enough independent computation to hide the transfer.

**Notebook.** Compare synchronizing after backward with readiness-based asynchronous reductions and DDP bucketing. Inspect traces to locate exposed communication and vary bucket size. Check gradients before interpreting speedups, and use wall-clock throughput as well as GPU-event timing.

**References.** [PyTorch Distributed paper](https://arxiv.org/abs/2006.15704) — reduction overlap; [DDP API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — bucket and accumulation options; [ETH data/tensor deck](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — slides 22–24; [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — trace collection.

### 6.5 Tensor parallelism

**Scope.** Partition linear layers by rows and columns, then pair partitions to avoid gathering intermediates unnecessarily. Extend the reasoning to gated MLPs and attention heads. Account for parameters, activations, and communication frequency; a smaller matrix can also reduce local GPU efficiency, so memory savings alone do not establish speedups.

**Notebook.** Implement a two-rank MLP and attention block. Compare outputs and gradients with the unpartitioned block, then measure collective bytes and local matmul shapes. Contrast a naive gather-everywhere implementation with a communication-aware partition.

**References.** [Megatron-LM](https://arxiv.org/abs/1909.08053) — original tensor-parallel derivation; [PyTorch tensor parallelism](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) — implementation abstractions; [ETH data/tensor deck](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — slides 27–35.

### 6.6 Differentiation through distributed computation

**Scope.** Derive backward communication from tensor ownership and the global loss. Distinguish replicated tensors, independent local uses, partial sums, and shards. Explain why an identity-forward/all-reduce-backward mapping can be correct in one tensor-parallel layout without being the universal derivative of an identically named collective.

**Notebook.** Implement custom autograd mappings for explicitly declared layouts. Compare Jacobian-vector products and parameter gradients with a single-process reference. Include a counterexample where local losses are independent and copying a replicated-gradient recipe produces an incorrect result.

**References.** [Megatron-LM](https://arxiv.org/abs/1909.08053) — paired forward/backward mappings; [DTensor documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.html) — placement semantics; [PyTorch extending autograd](https://docs.pytorch.org/docs/stable/notes/extending.html) — custom-function contracts; [ETH data/tensor deck](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — slides 31–32.

### 6.7 State sharding

**Scope.** Explain ZeRO stages by following ownership of parameters, gradients, and optimizer moments through a step. Separate persistent shards from transient full-layer materialization. Derive byte budgets for specified dtypes and optimizer choices: the familiar 16-bytes-per-parameter budget assumes particular weights, gradients, master weights, and moments, rather than applying to every PyTorch configuration.

**Notebook.** Implement a tiny sharded optimizer update, then compare it with a full optimizer. Trace all-gathers and reduce-scatters for layer-wise sharding. Predict peak memory before measuring FSDP on a pretrained decoder or a representative reduced model.

**References.** [ZeRO](https://arxiv.org/abs/1910.02054) — state-sharding taxonomy; [PyTorch fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) — current FSDP interface; [ETH pipeline/ZeRO/MoE deck](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) — slides 23–29; [TorchTitan](https://github.com/pytorch/torchtitan) — complete integration.

### 6.8 Prefetching and practical implementation

**Scope.** Analyze resharding after forward, retaining gathered weights, prefetching future layers, and choosing sharding boundaries. Relate the overlap opportunity to layer compute time and network speed. Introduce practical PyTorch abstractions only after readers understand the state transitions; pin versions because distributed APIs and defaults evolve.

**Notebook.** Compare gather-then-compute, prefetch, and retained-parameter configurations. Draw the measured timeline and account for simultaneously live layers. Report where reduced waiting increases peak memory or fails to improve throughput because transfer remains on the critical path.

**References.** [ETH pipeline/ZeRO/MoE deck](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) — slides 29–33 explicitly cover prefetch and measurements; [fully_shard API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) — controls and semantics; [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) — sharding/overlap comparisons.

## Chapter 7 Parallelism at Scale

**Purpose.** Combine the mechanisms of Chapter 6 into configurations for deep, sparse, and long-context models. The artifact is a configuration study with predicted and measured bottlenecks, schedule diagrams, and a recovery demonstration. Simulations run on CPU; two-to-eight-GPU exercises are optional. Full-cluster claims require supplied measured traces, not extrapolation from a single GPU.

**Chapter teaching sources.** [ETH pipeline/ZeRO/MoE deck](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) gives explicit schedules; [ETH long-context deck](https://drive.google.com/file/d/19KFfqDVLdAI3UD90LEGt3_jzebjPUeLt/view) connects attention blocks to device groups. [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) and [TorchTitan](https://github.com/pytorch/torchtitan) provide broader configuration and implementation references.

### 7.1 Pipeline parallelism

**Scope.** Partition a model by depth and follow activations forward and gradients backward across stage boundaries. Explain microbatches, stage service times, optimizer-step consistency, and pipeline fill/drain. Include endpoint effects from embeddings and output heads; equal layer counts need not imply balanced time or memory.

**Notebook.** Simulate a four-stage pipeline, then implement two stages in PyTorch. Compare a full-batch reference update with the pipelined update. Vary microbatch count and stage partitioning, predicting bubbles and boundary-transfer volume before collecting measurements.

**References.** [GPipe](https://arxiv.org/abs/1811.06965) — microbatch pipeline foundation; [PyTorch pipelining](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) — stages and schedules; [ETH pipeline deck](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) — slides 4–9.

### 7.2 Pipeline schedules

**Scope.** Compare all-forward/all-backward with 1F1B through dependency graphs and activation lifetimes. Explain why, under the basic balanced assumptions, 1F1B primarily reduces activation residency rather than eliminating the bubble. Relate increasing microbatch count to global batch and optimization, and include stage imbalance in the performance model.

**Notebook.** Generate schedule diagrams that count live activations at every time step. Validate the expected residency of each stage, then perturb compute times to show imbalance. Optional execution compares simulated peaks with measured memory on a small decoder.

**References.** [Megatron large-scale training](https://arxiv.org/abs/2104.04473) — scheduling and combined parallelism; [PyTorch pipeline schedules](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) — executable variants; [ETH pipeline deck](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) — slides 10–13.

### 7.3 Advanced scheduling

**Scope.** Develop virtual stages and interleaving, then split backward into input-gradient and weight-gradient work. Show which computation unblocks another stage and which can be deferred. Discuss memory, communication, optimizer synchronization, and imbalanced operation times as constraints on zero-bubble or bidirectional schedules.

**Notebook.** Extend the scheduler simulator with separate backward tasks and virtual stages. Check dependencies and weight-version consistency, then compare bubble time, activation residency, and message counts. Do not label a schedule optimal without stating the timing and memory assumptions.

**References.** [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) — dependency-based scheduling; [DualPipe implementation](https://github.com/deepseek-ai/DualPipe) — bidirectional schedule reference; [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — deployment within a training system; [ETH pipeline deck](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) — slides 14–19.

### 7.4 Sequence and context parallelism

**Scope.** Distinguish sharding tokenwise operations around tensor-parallel blocks from distributing attention over a long sequence. Derive blockwise stable-softmax accumulation, ring exchange, causal load imbalance, and balanced partitions. Compare sequence–head transposition with ring communication, including head-count constraints and combinations with tensor parallelism.

**Notebook.** Verify blockwise attention against dense attention, including causal masks and extreme logits. Simulate or execute ring exchange and compare contiguous with balanced sequence partitions. Record per-rank work and numerical error before claiming scaling gains.

**References.** [Ring Attention](https://arxiv.org/abs/2310.01889) — distributed blockwise attention; [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) — all-to-all sequence/head exchange; [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198) — sequence parallelism context; [ETH long-context deck](https://drive.google.com/file/d/19KFfqDVLdAI3UD90LEGt3_jzebjPUeLt/view) — slides 6–13.

### 7.5 Expert parallelism

**Scope.** Implement routed token copies, permutations, dispatch, grouped expert computation, reverse dispatch, and combination. Handle variable token counts and distinguish capacity-limited from dropless execution. Explain why reading dynamic counts onto the CPU can synchronize each layer, and introduce GPU-initiated communication as a systems extension rather than a required kernel-writing exercise.

**Notebook.** Compare distributed expert output and gradients with a single-process MoE. Sweep routing skew, expert count, and token count, measuring imbalance and communication. Use a trace to identify a counts-to-host synchronization and explain the effect of a device-side implementation.

**References.** [MegaBlocks](https://arxiv.org/abs/2211.15841) — dropless expert execution; [DeepEP](https://github.com/deepseek-ai/DeepEP) — expert dispatch implementation; [ETH pipeline/ZeRO/MoE deck](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) — slides 37–48; [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — communication/architecture case study.

### 7.6 Multidimensional parallelism

**Scope.** Map data, tensor, pipeline, expert, and context dimensions to process groups and physical links. Explain that dimensions are not always independent Cartesian factors: framework-specific expert/data decompositions and sharding groups must be stated. Use communication frequency and message size to justify placement rather than prescribing one universal topology.

**Notebook.** Construct a device-mesh visualizer and group-membership checks. Compare candidate placements on a supplied node/rack topology, then validate one small mesh in PyTorch. Detect overlapping or incorrectly ordered groups before running the training graph.

**References.** [DeviceMesh tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html) — process-group construction; [TorchTitan](https://github.com/pytorch/torchtitan) — composition in a native PyTorch stack; [ETH long-context deck](https://drive.google.com/file/d/19KFfqDVLdAI3UD90LEGt3_jzebjPUeLt/view) — slides 11–13; [Megatron large-scale training](https://arxiv.org/abs/2104.04473) — topology-aware combination.

### 7.7 Operational reliability

**Scope.** Cover distributed checkpoint ownership, asynchronous persistence, atomic completion, restart, and resharding across changed world sizes. Distinguish recoverable process failure, hardware error, corrupted state, and data-order divergence. Include straggler monitoring and inexpensive consistency checks, without promising bitwise reproducibility across different layouts or hardware.

**Notebook.** Interrupt a small distributed run, recover it, and compare expected losses and optimizer steps. Restart at a different supported world size, verifying reconstructed tensors and consumed samples. Inject an incomplete checkpoint and ensure it is rejected rather than silently loaded.

**References.** [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) — save/load and resharding interface; [PyTorch elastic launcher](https://docs.pytorch.org/docs/stable/elastic/run.html) — process recovery model; [TorchTitan](https://github.com/pytorch/torchtitan) — integrated checkpoint practice; [Llama 3 report](https://arxiv.org/abs/2407.21783) — large-run reliability evidence.

### 7.8 Choosing a configuration

**Scope.** Combine capacity, compute efficiency, collective cost, pipeline bubbles, expert skew, and global batch constraints in a configuration decision. Distinguish strong and weak scaling and avoid optimizing utilization alone. The relevant outcome is useful training progress per resource budget, with quality comparisons held constant where possible.

**Notebook.** Given a model and a hardware topology, enumerate feasible parallel configurations and rank them using a simple model. Compare selected predictions against supplied or measured traces. Explain discrepancies and provide a recommendation with sensitivity to sequence length and network speed.

**References.** [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) — measured configuration tradeoffs; [ETH pipeline deck](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) — slides 32–35; [ETH collective deck](https://drive.google.com/file/d/1LeR_e8y9m7gosaGQ5Aeo-m23hG-aovJx/view) — slides 4–9; [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) — analytical reference, translating any implementation ideas into PyTorch.

## Chapter 8 Supervised Fine Tuning and Efficient Adaptation

**Purpose.** Specialize a capable pretrained model while measuring task improvement, retained ability, and resource cost. Assume tokenization, model loading, and the basic training loop; distributed training is optional. The artifact is a full fine-tuning versus LoRA/QLoRA comparison with pinned data, templates, adapter configuration, and evaluations. Use a 1–3B checkpoint for routine experiments and a quantized larger checkpoint only where measured memory permits.

**Chapter teaching sources.** [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/en/unit0/1) supplies accessible adaptation exercises. [torchtune](https://github.com/pytorch/torchtune) provides inspectable PyTorch recipes. [Tülu 3](https://arxiv.org/abs/2411.15124) connects data mixtures with a complete post-training process. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) motivates controlled rank/module/learning-rate experiments; its empirical guidance should not become universal prescriptions.

### 8.1 Task and instruction adaptation

**Scope.** Compare adding a task head, generating task outputs as text, and instruction fine-tuning. Cover classification, extraction, summarization, translation, and bounded question answering. Define task distributions and metrics before training. Distinguish improved task accuracy from merely changing response format or verbosity, and use base/instruct checkpoints deliberately.

**Notebook.** Adapt the same small model to a bounded task using two formulations, with matched data and an untouched evaluation set. Compare task accuracy, output validity, latency, and a small retained-capability suite. Include the unadapted checkpoint as a baseline.

**References.** [Hugging Face LLM Course task chapter](https://huggingface.co/learn/llm-course/en/chapter7/1) — task formulations; [Natural Instructions](https://github.com/allenai/natural-instructions) — diverse instruction-task examples; [torchtune](https://github.com/pytorch/torchtune) — training recipe reference; [SmolLM2 base checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) — practical candidate, not a fixed family decision.

### 8.2 Serialization correctness

**Scope.** Make message serialization part of the model interface: roles, separators, assistant prefixes, end-of-turn versus end-of-sequence tokens, and loss masks. Explain generation-prefix consistency and avoid accidental double special-token insertion. Separate full-sequence from completion-only training as objective choices; neither should be declared universally superior.

**Notebook.** Build fixtures for single-turn, multi-turn, empty-answer, and truncated conversations. Compare training prefixes with inference prefixes token by token. Print visible token/mask tables and ensure only intended positions contribute to loss, without relying on decoded text alone.

**References.** [Hugging Face chat templates](https://huggingface.co/docs/transformers/chat_templating) — serialization contract; [Chat-template writing guide](https://huggingface.co/docs/transformers/chat_templating_writing) — special-token and generation-prefix details; [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer) — practical objective/masking comparison; [Tülu 3 data](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) — concrete conversation formats.

### 8.3 Instruction data

**Scope.** Build an instruction mixture with explicit quality, difficulty, diversity, and domain criteria. Cover synthetic demonstrations, filtering, rejection selection, reasoning traces, and overlap detection. Treat teacher identity and generation settings as dataset metadata. Reserve evaluation examples before iterative data improvement so the selection process does not leak benchmark answers into training.

**Notebook.** Compare a random subset with a curated subset at matched example/token budgets. Examine the effect of answer length and duplicate clusters. Train on a modest licensed subset and evaluate both task quality and style-independent instruction compliance.

**References.** [Tülu 3 paper](https://arxiv.org/abs/2411.15124) — open post-training data design; [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) — source annotations; [Smol SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) — accessible smaller-model data; [LIMA](https://arxiv.org/abs/2305.11206) — quality-versus-quantity experiment to interpret cautiously.

### 8.4 Efficient batching and packing

**Scope.** Explain padding waste and distinguish packing independent examples from treating them as one conversational stream. Develop document-isolation masks, position handling, length bucketing, and loss normalization. Show how implementation choices affect both the objective and the kernel path; a theoretically sparse mask does not guarantee acceleration on every device.

**Notebook.** Compare unpacked, length-bucketed, and packed execution on the same examples. Verify logits and gradients for independent examples under matched positions and masks. Measure useful tokens per second and peak memory rather than only batch utilization.

**References.** [PyTorch FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html) — programmable attention/masking API; [FlexAttention introduction](https://pytorch.org/blog/flexattention/) — implementation rationale; [torchtune](https://github.com/pytorch/torchtune) — packing recipes; [ETH data/tensor slides](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) — slides 4–7 connect batch shape with memory and throughput.

### 8.5 Full fine tuning and parameter efficient methods

**Scope.** Compare full updates, inserted adapters, soft/prefix tuning, and low-rank updates. Derive LoRA's forward computation, parameter gradients, initialization, and merge operation. Account separately for trainable parameters, optimizer state, activations, and base weights. Explain when changing the parameterization changes optimization even if representational capacity appears similar.

**Notebook.** Implement a LoRA linear layer and test equality with the merged weight before and after training. Compare full fine-tuning and LoRA at a feasible model size, reporting optimizer-state memory, total peak memory, quality, and retained capabilities.

**References.** [LoRA](https://arxiv.org/abs/2106.09685) — low-rank update derivation; [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — adapters; [Prefix-Tuning](https://arxiv.org/abs/2101.00190) — learned context parameterization; [Hugging Face PEFT](https://huggingface.co/docs/peft/index) — practical implementation comparison.

### 8.6 LoRA design and capacity

**Scope.** Study rank, module selection, scaling, initialization, and learning-rate interactions, with selected variants used to isolate mechanisms rather than form a catalogue. Compare SFT and later reward-based adaptation under controlled conditions. Treat reported low-rank success and learning-rate multipliers as hypotheses to test on the chosen model and dataset.

**Notebook.** Sweep a small grid of ranks and target modules while recording quality, memory, and update spectra. Compare standard LoRA with one mechanism-focused alternative such as DoRA or rank-stabilized scaling. Reuse the protocol in the RL chapter rather than claiming reward-based updates always need low rank.

**References.** [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — empirical capacity and tuning questions; [DoRA](https://arxiv.org/abs/2402.09353) — magnitude/direction decomposition; [Rank-Stabilized LoRA](https://arxiv.org/abs/2312.03732) — scaling analysis; [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora) — configuration and supported variants.

### 8.7 Quantized adaptation

**Scope.** Explain how gradients update adapters while base weights remain quantized and frozen. Introduce NF4, block scales, nested quantization, and optimizer-memory techniques sufficiently for QLoRA, leaving the broader quantization taxonomy to Chapter 12. Include dequantization compute and activation memory when estimating whether a workload fits a 24GB GPU.

**Notebook.** Compare BF16 LoRA with quantized-base adaptation using identical data and adapter settings. Check that frozen base weights remain unchanged, adapter gradients are finite, and merged/exported inference is evaluated separately. Report peak memory at several sequence lengths.

**References.** [QLoRA](https://arxiv.org/abs/2305.14314) — numerical representation and training method; [QLoRA reference code](https://github.com/artidoro/qlora) — implementation details; [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) — practical integration; [torchao](https://github.com/pytorch/ao) — PyTorch-native quantized training paths to assess when selecting the final stack.

### 8.8 Combining and retaining capabilities

**Scope.** Evaluate forgetting, replay, adapter composition, and weight-space merging for models with compatible ancestry. Introduce task arithmetic and interference-aware merging, distinguishing merging from dense-to-MoE upcycling in Chapter 3. Add a bounded comparison of storing knowledge in adapted weights versus supplying retrieved text, without introducing agent orchestration.

**Notebook.** Train two small adapters on distinct tasks, then evaluate separate, combined, and merged variants on both tasks and a general suite. Include an incompatible or destructive merge example. For a small document corpus, compare adaptation with retrieval-conditioned answers using held-out factual questions.

**References.** [Task Arithmetic](https://arxiv.org/abs/2212.04089) — weight-space task vectors; [TIES-Merging](https://arxiv.org/abs/2306.01708) — interference reduction; [DARE](https://arxiv.org/abs/2311.03099) — sparse delta manipulation; [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) — weights-versus-context comparison; [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) — domain-adaptation baseline.

## Chapter 9 Preference Learning and Reinforcement Learning

**Purpose and prerequisites.** Develop language-model post-training from the policy-gradient and regularized-control foundations in Advanced, using the serialization, loss masking, and adapted checkpoints from Chapter 8. Readers should be able to distinguish an objective, its gradient estimator, and the systems implementation. The chapter preserves the book's RL notation rather than introducing another convention.

**Main artifact and resource tier.** A transparent PyTorch preference/RL trainer with logged rollouts, rewards, reference log-probabilities, and evaluation results. The default model is a capable 1–3B checkpoint with adapters, bounded responses, and small rollout groups. An 8B actor plus reference, critic, reward model, and optimizer is not assumed to fit on a 24GB GPU. Cached reference scores, sequential model loading, and supplied rollout traces provide accessible alternatives.

**Chapter teaching sources.** [Lambert's RLHF Book](https://rlhfbook.com/) provides the instructional sequence and terminology; [Stanford CS336](https://cs336.stanford.edu/) supplies implementation-oriented alignment/reasoning assignments; [Hugging Face TRL](https://huggingface.co/docs/trl/index) provides practical trainer comparisons; [verl](https://github.com/verl-project/verl) provides the bridge to the distributed systems in Chapter 14. These implementations are reference points, not substitutes for the explicit objective code.

### 9.1 Preference data and reward models

**Scope.** Explain pairwise comparisons, Bradley–Terry likelihoods, annotator disagreement, ties, heterogeneous preferences, and reward-model calibration. Separate reward-model accuracy on held-out comparisons from usefulness under policy optimization. Track how prompt distribution, response length, and annotator instructions influence the learned reward. Introduce outcome rewards here; process rewards receive fuller treatment in Chapter 11.

**Notebook.** Train a small reward head on a documented subset of preference pairs. Compare random and prompt-grouped splits, accuracy, calibration, and length correlations. Inspect failures with a fixed rubric and retain a held-out set generated by a different policy.

**References.**

- [InstructGPT](https://arxiv.org/abs/2203.02155) — original pipeline connecting demonstrations, preference comparisons, reward learning, and policy optimization.
- [Anthropic HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) — accessible chosen/rejected examples and dataset limitations to discuss before training.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — why optimizing a proxy reward can move beyond the region where it is reliable.

### 9.2 KL-regularized policy optimization

**Scope.** Derive the sequence-level regularized objective and its token decomposition. Distinguish the policy being optimized, the reference policy, and the behavior policy that generated samples. Compare exact categorical KL on a tiny vocabulary with sampled estimators, specifying the sampling distribution and whether a term is an objective, reward shaping, or monitoring statistic.

**Notebook.** Use a small categorical policy to verify expectations and gradients against enumeration. Then log KL estimators on real model rollouts, checking masks, response boundaries, reductions, and the effects of estimator variance and sequence length.

**References.**

- [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) — concise derivation of practical Monte Carlo estimators and their assumptions.
- [PPO](https://arxiv.org/abs/1707.06347) — policy-update constraints and the distinction between a surrogate and the underlying objective.
- [DPO](https://arxiv.org/abs/2305.18290) — the regularized optimal-policy relationship that connects this section to direct preference learning.

### 9.3 Direct preference optimization

**Scope.** Derive DPO by connecting a regularized optimum to a pairwise preference model. Explain reference-policy support, offline data coverage, reward identifiability, likelihood displacement, and why growing the preference margin does not guarantee improvement on every task. Present IPO, SimPO, or online preference updating as selected comparisons, organized by changed assumptions rather than acronyms.

**Notebook.** Implement the DPO loss in PyTorch, check it against a library implementation, and train on the same pairs as Section 9.1. Plot chosen and rejected log-probabilities separately alongside win rate, retained capability, response length, and reference divergence.

**References.**

- [DPO paper](https://arxiv.org/abs/2305.18290) — central derivation and original experimental design.
- [SimPO](https://arxiv.org/abs/2405.14734) — reference-free, length-normalized comparison of the objective design.
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) — practical handling of reference scores, datasets, and loss variants; pin the library release.

### 9.4 Online RLHF

**Scope.** Develop the actor, critic, reward, and reference roles, explaining which states require gradients and which can be cached or evaluated separately. Connect token-level advantages, termination, reward placement, value fitting, and minibatch reuse to the foundations in Advanced. Make rollout collection and optimization separate, observable phases.

**Notebook.** Run a small PPO-style experiment with short completions and a simple reward before substituting a learned reward. Check advantage and return calculations on hand-computable trajectories, and report time and peak memory by role rather than only total step time.

**References.**

- [InstructGPT](https://arxiv.org/abs/2203.02155) — reference for the complete learning pipeline.
- [PPO](https://arxiv.org/abs/1707.06347) — clipped policy objective and value-learning context.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — production-oriented organization of actor, reference, reward, and critic components.

### 9.5 Verifiable rewards

**Scope.** Replace preference rewards with explicit checks for mathematical equivalence, bounded code problems, or formal output constraints. Separate correctness of an answer from correctness of its extraction and verification. Discuss partial credit, malformed answers, timeouts, underspecified tests, and exploitation of the verifier. Avoid presenting programmatic rewards as automatically trustworthy.

**Notebook.** Build a deterministic arithmetic or Countdown generator with disjoint training/test instances and an exact checker. Add an optional fixed-function code task with isolated execution. Test the verifier independently using valid, invalid, ambiguous, and adversarially formatted responses.

**References.**

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — mathematical reasoning and the introduction of GRPO in this setting.
- [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k) — natural-language arithmetic examples for a separate evaluation set.
- [EvalPlus](https://github.com/evalplus/evalplus) — stronger code tests and evidence that evaluation depends on test completeness.

### 9.6 Group-based and leave-one-out methods

**Scope.** Derive group-relative and leave-one-out baselines from policy gradients. Examine group size, reward sparsity, variance, and the difference between subtracting a baseline and dividing by a group standard deviation. Explain why groups with identical outcomes may carry little learning signal for particular estimators, without equating them with universally useless prompts.

**Notebook.** Compare REINFORCE, leave-one-out, and GRPO-style estimators first on enumerated policies and then on the arithmetic task. Hold generated-token budgets fixed and measure variance, success rates, and the fraction of informative groups.

**References.**

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO formulation and context.
- [Back to Basics](https://arxiv.org/abs/2402.14740) — REINFORCE-style alternatives and leave-one-out reasoning.
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) — implementation details and configurable normalization choices to compare with the notebook.

### 9.7 Stability and bias

**Scope.** Study response-length weighting, token versus sequence reductions, clipping, reward normalization, entropy, and truncated outputs. Present Dr. GRPO and DAPO as diagnoses and remedies evaluated in particular settings. Require students to identify which distribution each statistic averages over and how filtering or dynamic sampling changes that distribution.

**Notebook.** Change one normalization or clipping choice at a time. Plot correctness against length, entropy over training, clipping fractions, and gradient norms. Include an intentionally flawed reduction and show its effect before implementing the correction.

**References.**

- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) — base-model effects and the Dr. GRPO analysis.
- [DAPO](https://arxiv.org/abs/2503.14476) — clipping, sampling, token-level learning, and long-response issues.
- [verl](https://github.com/verl-project/verl) — inspect configurable estimator and loss implementations rather than treating variant names as specifications.

### 9.8 Online and asynchronous learning

**Scope.** Explain policy lag between sampling and training, importance ratios, reuse limits, and changes in prompt or reward mixtures. Distinguish an on-policy objective from a system that approximates it with asynchronous samples. Defer deployment engineering to Chapter 14 while retaining the statistical consequences of stale or filtered trajectories here.

**Notebook.** Replay trajectories collected at successive checkpoints, varying their age and correction method. Compare effective sample sizes, importance-ratio tails, update stability, and held-out reward. A trace replay makes the central experiment runnable without a distributed cluster.

**References.**

- [AReaL paper](https://arxiv.org/abs/2505.24298) — asynchronous language-reasoning training and policy staleness.
- [AReaL implementation](https://github.com/areal-project/AReaL) — inspect versioning and rollout/update boundaries, excluding agent applications.
- [DeepSeek-V4.1-Flash report](https://arxiv.org/html/2609.19969v1) — Sections 5.2.1–5.2.4 provide a contemporary RL/OPD infrastructure case study.

### 9.9 Failure analysis

**Scope.** Separate rising training reward from genuine task improvement. Cover proxy exploitation, length/style shortcuts, verifier weaknesses, sycophancy, refusal calibration, and regressions on unrelated capabilities. Use observable errors and counterfactual tests rather than interpreting every verbalized explanation as evidence of the model's internal process.

**Notebook.** Optimize a deliberately imperfect reward, then evaluate against an independently designed checker and human rubric. Maintain a small regression suite throughout training and report both false acceptance and over-refusal on benign examples.

**References.**

- [Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — controlled study of divergence between proxy and target rewards.
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — rule-guided feedback and the design of preference criteria.
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) — examples for discussing helpfulness/harmlessness judgments and their scope.

## Chapter 10 Distillation and Self-Improvement

**Purpose and prerequisites.** Explain distillation as learning from a teacher's behavior or distribution, including methods that improve capability rather than merely reducing parameter count. Prerequisites are supervised adaptation, probability divergences, sequence generation, and Chapter 9's distinction between target and sampling policies.

**Main artifact and resource tier.** A common student-training harness comparing offline trace SFT, token-distribution distillation, OPD, OPSD, and a reward-based baseline. Use a 1–3B student, bounded responses, and either a smaller same-family teacher, sequential teacher scoring, or stored scores. Full-vocabulary scoring and external teacher access must have explicit storage, latency, licensing, and compute budgets; an API returning text alone does not supply teacher logits.

**Chapter teaching sources.** [GKD](https://arxiv.org/abs/2306.13649), [MiniLLM](https://arxiv.org/abs/2306.08543), and [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) anchor the algorithms. [RLHF Book](https://rlhfbook.com/) supplies a broader post-training account, while [verl's OPD documentation](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md) connects objectives to an implementation. Newer self-distillation results should be presented as research methods with stated evaluation conditions.

### 10.1 The distillation design space

**Scope.** Classify methods by teacher/student identity, trajectory source, extra context, target information, and update schedule. Distinguish a fixed teacher corpus, teacher distributions evaluated on fixed prefixes, student-generated trajectories, and a privileged-context self-teacher. Explain that neither “self” nor “on-policy” alone specifies a loss or guarantees improvement.

**Notebook.** Implement a shared data record containing prompt, sampled response, behavior-policy revision, teacher context, and supervision type. Trace one example through offline and online paths to expose what each method observes and what must be recomputed.

**References.**

- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) — foundational distinction between sequence and token supervision.
- [GKD](https://arxiv.org/abs/2306.13649) — student-generated prefixes and flexible distribution matching.
- [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) — same-model teacher/student with different available context.

### 10.2 Offline sequence distillation

**Scope.** Build synthetic demonstration datasets from teacher responses, with explicit prompt sampling, verification, diversity, and provenance. Discuss reasoning traces as training targets without assuming their verbal form faithfully exposes a teacher's computation. Connect coverage and filtering to the data chapter and distinguish distillation gains from simply adding more task data.

**Notebook.** Train on raw teacher samples, verified samples, and a diversity-preserving subset at matched token budgets. Evaluate on unseen problem templates and compare answer accuracy, response length, and retained language capabilities.

**References.**

- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) — the original sequence-target formulation.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — reasoning-trace distillation as part of a published model-development pipeline.
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) — candidate reasoning data with model/data provenance to inspect before selecting a subset.

### 10.3 Distribution matching

**Scope.** Derive forward and reverse KL and generalized alternatives on categorical token distributions, specifying temperature and normalization. Compare full-vocabulary targets, top-k approximations, and sampled-token estimates. Distinguish truncation bias from Monte Carlo variance and explain why inaccessible or differently tokenized teacher distributions require a different interface.

**Notebook.** Verify divergence values and gradients on synthetic distributions before applying them to a pretrained-model batch. Compare the effect of teacher-score truncation and temperature on student gradients, memory, and held-out loss.

**References.**

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — soft targets and temperature as a conceptual starting point.
- [GKD](https://arxiv.org/abs/2306.13649) — alternative divergences in autoregressive sequence distillation.
- [MiniLLM](https://arxiv.org/abs/2306.08543) — reverse-KL-based distillation and sampled-policy optimization.

### 10.4 On-policy distillation OPD

**Scope.** Generate responses from the current student and ask a teacher to evaluate the prefixes it actually visits. Explain exposure mismatch, trajectory refresh, mixing on/off-policy examples, and the distinction between differentiating a conditional matching loss and treating sampled trajectories as policy-gradient data. Preserve the estimator assumptions of the chosen formulation.

**Notebook.** Compare fixed-corpus distillation with periodically refreshed student rollouts. Track teacher–student disagreement on visited prefixes, held-out quality, rollout tokens, teacher scoring time, and total compute rather than reporting training steps alone.

**References.**

- [GKD paper](https://arxiv.org/abs/2306.13649) — primary algorithm and formulation of self-generated mistakes.
- [MiniLLM paper](https://arxiv.org/abs/2306.08543) — complementary objective and optimization choices.
- [verl OPD documentation](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md) — implementation organization and teacher-scoring interfaces.

### 10.5 On-policy self-distillation OPSD

**Scope.** Develop the Self-Distilled Reasoner formulation in which student and teacher share a model identity but receive different contexts. The student sees the problem; the teacher additionally sees privileged information such as a verified solution. Separate frozen snapshots, stop-gradient teacher evaluations, and any moving-teacher updates explicitly; prevent privileged information from leaking into evaluation.

**Notebook.** On a small verified arithmetic dataset, compare question-only teaching, answer-conditioned teaching, and solution-conditioned teaching. Generate student trajectories and score them under the teacher context. Audit prompt construction and evaluate only from the question.

**References.**

- [Self-Distilled Reasoner paper](https://arxiv.org/abs/2601.18734) — definitive formulation and experimental assumptions for this OPSD variant.
- [Official OPSD repository](https://github.com/siyan-zhao/OPSD) — source-linked implementation to inspect for sampling, masking, and teacher-update choices.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — possible supervised reasoning examples, with a separate held-out split and contamination discussion.

### 10.6 Context and reasoning distillation

**Scope.** Examine how knowledge, instructions, demonstrations, or concise behavior supplied in a teacher context can be transferred into a student policy. Compare removing dependence on context with shortening generated reasoning; these optimize different costs. Distinguish a correct answer with fewer tokens from a shorter but brittle solution.

**Notebook.** Distill a context-assisted or concision-conditioned teacher into a question-only student. Plot correctness against generated tokens and test robustness to problem difficulty, paraphrase, and missing cues. Keep the teacher's extra context out of every evaluation input.

**References.**

- [On-Policy Context Distillation](https://arxiv.org/abs/2602.12275) — research direction for transferring contextual assistance through student rollouts.
- [On-Policy Self-Distillation for Reasoning Compression](https://arxiv.org/abs/2603.05433) — recent formulation focused on shorter reasoning.
- [s1](https://arxiv.org/abs/2501.19393) — contrasting inference-time budget intervention, useful for separating training and decoding effects.

### 10.7 Multiple teachers and RL combinations

**Scope.** Explore specialist teachers, task-dependent selection, weighted objectives, and consolidation into one student. Explain why teacher quality is relative to domain and student compatibility rather than a single overall score. Compare alternating distillation and reward-based updates with combined objectives, checking whether one component undoes another's gains.

**Notebook.** Use two specialized checkpoints or two teacher contexts on separate task families. Compare single-teacher, mixed-teacher, and distillation-plus-verifier training at matched scoring budgets, with per-domain evaluation and an unrelated regression suite.

**References.**

- [GKD](https://arxiv.org/abs/2306.13649) — integration of distillation with reward-based optimization.
- [DeepSeek-V4.1-Flash report](https://arxiv.org/html/2609.19969v1) — Section 5.2.4 describes multi-teacher OPD as a frontier case study.
- [verl OPD documentation](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md) — practical configuration of teacher scoring and online trajectories.

### 10.8 Efficiency and limitations

**Scope.** Account for teacher inference, vocabulary logits, tokenization alignment, trajectory storage, and update latency. Analyze stale teachers, low-quality privileged information, self-confirming errors, and transfer failures. Distinguish the cost of collecting a reusable offline dataset from repeated online scoring, and state teacher access requirements before choosing an algorithm.

**Notebook.** Produce a complete cost ledger for the chapter's methods. Repeat one comparison with a weaker or mismatched teacher and vary rollout refresh frequency. Report failure cases and confidence intervals rather than assuming every distillation method improves the student.

**References.**

- [MiniLLM](https://arxiv.org/abs/2306.08543) — practical constraints of white-box teacher distributions.
- [OPSD paper and experiments](https://arxiv.org/abs/2601.18734) — teacher-context and capability conditions to examine.
- [AReaL](https://arxiv.org/abs/2505.24298) — systems perspective on expensive sampling and asynchronous updates; use only the language-reasoning material.

## Chapter 11 Reasoning and Inference-Time Computation

**Purpose and prerequisites.** Connect the learning methods of Chapters 8–10 to the computation used to answer a question. Readers should distinguish accuracy, reasoning-token use, latency, and total sampling cost. The chapter studies self-contained mathematics and fixed code problems, not tool-using agents or interactive environments.

**Main artifact and resource tier.** An inference-budget evaluator for a capable 1–8B model, with reproducible samples, verifiers, and token/time accounting. A 24GB setup can run samples sequentially; supplied traces support large-budget comparisons. Training exercises use the smaller checkpoint and controlled response lengths.

**Chapter teaching sources.** [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) provides the allocation question; [DeepSeek-R1](https://arxiv.org/abs/2501.12948) supplies a training-pipeline case study; [s1 code](https://github.com/simplescaling/s1) provides a compact budget-control example. [PRM800K](https://github.com/openai/prm800k) supports process-supervision exercises.

### 11.1 What reasoning evaluations measure

**Scope.** Separate final-answer correctness, consistency, robustness, and the apparent quality of generated explanations. Explain that a readable trace is an output and not automatically a faithful record of internal computation. Define difficulty strata and evaluation protocols before increasing the number of samples or the allowed reasoning length.

**Notebook.** Evaluate the same model with direct answers and reasoning prompts on disjoint arithmetic and mathematical sets. Introduce irrelevant or biased hints and inspect whether explanations acknowledge their influence. Report accuracy and confidence intervals without equating verbosity with reasoning.

**References.**

- [Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388) — experimental approach to explanation faithfulness.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — accessible arithmetic reasoning evaluation.
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — a compact mathematical evaluation set with different difficulty characteristics.

### 11.2 Training reasoning behavior

**Scope.** Connect cold-start demonstrations, verifiable rewards, rejection-sampled updates, and mixed-domain post-training into complete recipes. Study the dependence on the starting checkpoint and problem distribution. Report observed self-correction or longer reasoning as measured behavior, not an inevitable consequence of RL or evidence that a new internal algorithm has appeared.

**Notebook.** Compare SFT-only, RL-only from the same base, and SFT-then-RL on a bounded problem family. Hold training compute approximately fixed and evaluate unseen templates, retained capabilities, response length, and failure patterns.

**References.**

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — the staged reasoning-model development case study.
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) — evidence for examining starting-model and optimization effects.
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) — candidate traces for a reproducible small SFT subset.

### 11.3 Outcome and process supervision

**Scope.** Compare labels on final answers with feedback on intermediate steps. Explain manual labels, verifier-generated labels, and Monte Carlo estimates of continuation success. Cover ambiguous step boundaries, propagation of labeling errors, reward-model calibration, and selection bias when a process model sees only traces from one generator.

**Notebook.** Fit outcome and step-level scorers on a limited labeled dataset. Use them to rank identical candidate sets, evaluate calibration on new generators, and compare whether extra supervision improves selection enough to justify its labeling and inference cost.

**References.**

- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — outcome/process supervision comparison.
- [PRM800K repository](https://github.com/openai/prm800k) — step labels and data structures for the notebook.
- [Math-Shepherd](https://arxiv.org/abs/2312.08935) — automated step labeling and verification as an alternative design.

### 11.4 Parallel inference-time computation

**Scope.** Develop independent sampling, majority voting, best-of-N selection, and verifier-guided search over candidate solutions. Distinguish oracle pass@N from a deployable selector's success rate. Count all generated candidates and verifier calls, including discarded work, and examine correlation between samples rather than assuming independence.

**Notebook.** Generate a shared pool of candidates and compare self-consistency, learned scoring, and an exact verifier where available. Plot success against total tokens, wall time, and sample count; include an oracle ceiling only as a labeled diagnostic.

**References.**

- [Self-Consistency](https://arxiv.org/abs/2203.11171) — sampling and answer aggregation.
- [Test-Time Compute Scaling](https://arxiv.org/abs/2408.03314) — comparisons of selection/search and compute allocation.
- [PRM800K](https://github.com/openai/prm800k) — an executable basis for process-scored candidate selection.

### 11.5 Sequential computation

**Scope.** Study revisions, longer traces, stopping decisions, and explicit budget controls. Separate a model's learned ability to revise from prompting it to continue. Explain how truncation and answer extraction can confound evaluations, and compare extra computation on one trajectory with starting a fresh trajectory.

**Notebook.** Compare ordinary stopping, constrained budgets, and a documented continuation intervention on the same prompts. Record accuracy, malformed or missing answers, actual token counts, and latency; inspect cases where additional continuation harms a previously correct answer.

**References.**

- [s1 paper](https://arxiv.org/abs/2501.19393) — budget forcing and its particular experimental setting.
- [s1 repository](https://github.com/simplescaling/s1) — concrete generation and evaluation code.
- [Test-Time Compute Scaling](https://arxiv.org/abs/2408.03314) — revision as a compute-allocation option.

### 11.6 Adaptive compute allocation

**Scope.** Allocate sampling or reasoning budgets by estimated difficulty, uncertainty, or verifier disagreement. Distinguish an oracle policy that knows correctness from a practical decision rule. Connect budget allocation to latency objectives and consider whether a larger model answering once is preferable to repeated calls to a smaller model.

**Notebook.** Fit a budget policy on a validation set and freeze it before testing. Compare fixed budgets, difficulty-stratified budgets, and a sequential stopping rule under both average-compute and maximum-latency constraints.

**References.**

- [Test-Time Compute Scaling](https://arxiv.org/abs/2408.03314) — core compute-optimal allocation experiments.
- [Self-Consistency](https://arxiv.org/abs/2203.11171) — baseline aggregation mechanism.
- [s1](https://arxiv.org/abs/2501.19393) — a contrasting way to control sequential expenditure.

### 11.7 Reasoning efficiency

**Scope.** Construct accuracy–cost frontiers for reasoning systems. Examine overlong responses, concise solutions, distillation of expensive search, and early termination. Require quality evaluation across difficulty levels so that apparent efficiency does not merely reflect dropping hard examples or generating shorter incorrect outputs.

**Notebook.** Combine the chapter's runs into a frontier plot and identify dominated configurations. Compare a concise distilled checkpoint with its original model at matched token budgets, including the amortized teacher/training cost when deployment volume is specified.

**References.**

- [On-Policy Self-Distillation for Reasoning Compression](https://arxiv.org/abs/2603.05433) — recent training-side compression method.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — reasoning transfer to smaller models.
- [s1 code](https://github.com/simplescaling/s1) — reproducible budget sweeps and generation controls.

## Chapter 12 Quantization and Model Compression

**Purpose and prerequisites.** Reduce model storage, memory traffic, and computation while measuring the effect on capability. Prerequisites are numerical formats and rooflines from Chapter 4, representative calibration data from Chapter 1, and the distillation tools in Chapter 10. KV-state compression is developed separately in Chapter 13.

**Main artifact and resource tier.** A collection of compressed versions of the same useful pretrained model, accompanied by calibration manifests, exact storage accounting, numerical checks, task results, and measured latency. Use a smaller checkpoint for repeated recovery training and a 7–8B checkpoint for realistic inference studies when memory permits. Reference fake-quantization code demonstrates errors; accelerated execution requires compatible packed formats and kernels.

**Chapter teaching sources.** [Smola's efficiency slides](https://alex.smola.org/posts/45-mlss-efficiency/) provide the memory-traffic framing and weight-compression progression; [MIT 6.5940](https://hanlab.mit.edu/courses/2026-fall-65940) supplies compression-oriented pedagogy; [torchao](https://github.com/pytorch/ao) and [llm-compressor](https://github.com/vllm-project/llm-compressor) supply practical PyTorch ecosystem implementations. Pin hardware, software versions, and calibration inputs in every benchmark.

### 12.1 The compression design space

**Scope.** Distinguish fewer stored parameters, fewer active parameters, lower precision, sparsity, and lossless encoding. Map each change to capacity, bandwidth, arithmetic, and kernel requirements. Explain why MoE sparsity, weight quantization, and pruning operate on different axes and why multiplying published speedups usually gives an invalid combined prediction.

**Notebook.** Build a byte/FLOP ledger for dense, sparse, and low-bit variants of the selected model. Predict prefill and decode bottlenecks at several batch sizes, then compare available measured runs while holding task quality and context length explicit.

**References.**

- [Smola MLSS efficiency material](https://alex.smola.org/posts/45-mlss-efficiency/) — weight traffic and compression as part of the inference cost model.
- [MIT Efficient AI course](https://hanlab.mit.edu/courses/2026-fall-65940) — pruning, quantization, and distillation as complementary methods.
- [torchao repository](https://github.com/pytorch/ao) — supported representations and implementation examples.

### 12.2 Quantization fundamentals

**Scope.** Derive uniform quantization, symmetric/asymmetric grids, scales, zero points, clipping, and per-tensor/channel/group choices. Distinguish rounding error, saturation, calibration mismatch, and storage overhead. Use actual packed bytes, including scales and padding, when reporting effective bits per parameter.

**Notebook.** Implement reference quantizers in PyTorch and test reconstruction error, saturation, and output error for a linear layer. Sweep group size and clipping on held-out activations, then compare predicted checkpoint size with serialized packed tensors.

**References.**

- [Quantization and Training for Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — foundational affine quantization equations.
- [torchao documentation](https://docs.pytorch.org/ao/stable/index.html) — practical quantization configuration and supported tensor representations.
- [GPTQ](https://arxiv.org/abs/2210.17323) — a useful contrast between simple rounding and reconstruction-aware quantization.

### 12.3 Weight and activation outliers

**Scope.** Explain how activation distributions and channel scales affect quantization error. Develop equivalent rescalings across linear operations and distinguish weight-only quantization from joint weight/activation quantization. Avoid generalizing a particular fraction of salient channels or one calibration distribution to every architecture and task.

**Notebook.** Plot per-channel activation statistics on representative text. Compare round-to-nearest, SmoothQuant-style scaling, and AWQ-style calibration for a small layer stack; evaluate both layer reconstruction and end-to-end held-out quality.

**References.**

- [SmoothQuant](https://arxiv.org/abs/2211.10438) — shifting quantization difficulty through equivalent scaling.
- [AWQ](https://arxiv.org/abs/2306.00978) — activation-aware weight quantization and calibration design.
- [LLM.int8()](https://arxiv.org/abs/2208.07339) — activation outliers and mixed-precision decomposition.

### 12.4 Reconstruction-based quantization

**Scope.** Formulate layer-wise output reconstruction, derive the curvature approximation used by GPTQ-style methods, and explain damping, column ordering, error compensation, and block updates. Separate the local reconstruction objective from whole-model capability and examine how calibration size and domain affect the result.

**Notebook.** Implement the method first on a small matrix where the objective can be checked directly. Quantize selected pretrained layers, compare reconstruction with round-to-nearest, and then run whole-model quality evaluation with two calibration domains.

**References.**

- [GPTQ paper](https://arxiv.org/abs/2210.17323) — derivation, approximation choices, and quantization algorithm.
- [GPTQ reference repository](https://github.com/IST-DASLab/gptq) — implementation structure for Hessian accumulation and blockwise updates.
- [SparseGPT](https://arxiv.org/abs/2301.00774) — related reconstruction machinery, useful for connecting quantization and pruning.

### 12.5 Rotations and advanced low-bit representations

**Scope.** Develop orthogonal changes of basis, outlier spreading, learned rotations, and the conditions under which transformations can be absorbed into weights. Introduce vector/codebook quantization as an extension beyond independent scalar rounding. Account for runtime rotations, metadata, and dequantization rather than assuming these transformations are free.

**Notebook.** Verify an unquantized rotation transformation numerically, then compare scalar quantization before and after rotation. Add a small vector-codebook exercise to show the rate–distortion tradeoff without requiring a production low-bit kernel.

**References.**

- [QuaRot](https://arxiv.org/abs/2404.00456) — computational invariance and outlier reduction through rotations.
- [SpinQuant](https://arxiv.org/abs/2405.16406) — optimizing rotations using calibration data.
- [QuIP#](https://arxiv.org/abs/2402.04396) — incoherence processing and lattice-codebook quantization as advanced reading.

### 12.6 Quantization-aware training

**Scope.** Explain fake quantization, straight-through gradient approximations, clipping/scaling updates, and recovery training. Distinguish QAT from low-precision arithmetic throughout pretraining and from adapting frozen quantized weights with LoRA. Include native block-scaled formats as execution targets whose support depends on the selected hardware.

**Notebook.** Quantize a small pretrained model, perform a short QAT recovery run, and compare with post-training quantization under matched data and update budgets. Check fake-quantized versus converted-model outputs before comparing speed.

**References.**

- [torchao QAT documentation](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html) — entry point for current quantization/QAT APIs and conversion workflows.
- [QLoRA](https://arxiv.org/abs/2305.14314) — contrasting frozen-weight quantized adaptation.
- [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3) — contemporary low-precision deployment/QAT case study; model-specific details remain in Chapter 16.

### 12.7 Lossless weight compression

**Scope.** Measure entropy in the stored weight representation and develop exact coding of redundant fields. Explain why a format-dependent empirical entropy estimate is not a universal lower bound for all models. Lossless reconstruction guarantees the original weights; bit-identical inference additionally depends on the execution path and numerical settings.

**Notebook.** Inspect sign, exponent, and mantissa distributions, implement a small entropy-coding experiment, and verify byte-for-byte reconstruction. Measure decode cost and peak memory separately from compressed file size; compare offline checkpoint compression with on-the-fly execution.

**References.**

- [DFloat11](https://arxiv.org/abs/2504.11651) — exponent redundancy, exact reconstruction, and decompression-aware inference.
- [ZipNN](https://arxiv.org/abs/2411.05239) — lossless neural-weight compression and data representation.
- [Smola efficiency slides](https://alex.smola.org/posts/45-mlss-efficiency/) — comparison of lossless coding and lossy quantization.

### 12.8 Pruning and factorization

**Scope.** Compare unstructured sparsity, hardware-supported patterns, removal of complete heads/channels/layers, and low-rank factorization. Explain saliency estimates, output reconstruction, architecture surgery, and recovery by training or distillation. A sparse mask alone does not reduce dense-kernel work, so distinguish parameter removal from realized acceleration.

**Notebook.** Prune or factor selected model layers, evaluate without recovery, then distill briefly from the original checkpoint. Compare actual packed size and measured latency, including a case where parameter sparsity provides no speed improvement.

**References.**

- [SparseGPT](https://arxiv.org/abs/2301.00774) — reconstruction-aware one-shot pruning.
- [Wanda](https://arxiv.org/abs/2306.11695) — activation-aware saliency without the same curvature machinery.
- [Minitron](https://arxiv.org/abs/2407.14679) — structured pruning followed by distillation as a model-development recipe.

### 12.9 Deployment and evaluation

**Scope.** Connect calibrated models to packed weights, kernel dispatch, hardware support, and reproducible benchmarking. Separate cold loading, first-token time, decode throughput, batch scaling, and peak memory. Evaluate capability before declaring a compression win, and document configurations where decompression or conversion costs erase the expected benefit.

**Notebook.** Produce a common report for the chapter's compressed checkpoints: quality, effective bits, peak memory, load time, prefill/decode latency, and throughput. Use fixed prompts and decoding settings, and retain the uncompressed checkpoint as the reference.

**References.**

- [torchao](https://github.com/pytorch/ao) — PyTorch-native quantization and conversion examples.
- [llm-compressor](https://github.com/vllm-project/llm-compressor) — calibration and deployment workflows for supported serving stacks.
- [Marlin](https://github.com/IST-DASLab/marlin) — example of why a packed low-bit format and a specialized kernel must be considered together.

## Chapter 13 Efficient Inference and Long Context State

**Purpose.** Turn a correct pretrained-model implementation into an efficient generator. Separate transformations that preserve the target computation from approximations or architecture changes that need quality evaluation. Prerequisites are Chapters 2–4 and 12; Chapter 7 supplies the distributed counterpart. The main artifact is a benchmarked PyTorch generator with interchangeable cache and decoding policies. Core exercises use one GPU; long-context and multi-device results can use supplied traces.

**Chapter references.** [Smola, Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/) provides the weight-versus-KV accounting and prefill/decode teaching thread. [PyTorch gpt-fast](https://github.com/pytorch-labs/gpt-fast) supplies a compact practical implementation to inspect after building the reference generator. [FlashAttention](https://arxiv.org/abs/2205.14135) develops IO-aware exact attention. [ICML 2024 serving tutorial](https://icml.cc/virtual/2024/tutorial/35229) gives the wider algorithms-to-systems setting. [ETH Week 6](https://drive.google.com/file/d/19KFfqDVLdAI3UD90LEGt3_jzebjPUeLt/view), printed slides 6–10 and 15–19, connects blockwise attention and cache accounting.

### 13.1 A reference autoregressive generator

**Scope.** Implement prefill, incremental decode, position tracking, masks, EOS handling, and cache allocation. Compare dynamic concatenation with preallocated storage. Treat the tokenizer, model revision, decoding distribution, and positional conventions as part of the experiment. Separate correctness of cached execution from the numerical effects of changing kernels or precision.

**Notebook.** Load the selected small pretrained model and compare cached versus full-prefix logits over short sequences, multiple prompt lengths, and padded batches. Add regression cases for empty cache, cache reset, and position offsets. Measure allocations and tokens per second only after correctness passes.

**References.** [Transformers cache documentation](https://huggingface.co/docs/transformers/en/kv_cache) explains practical cache choices; [gpt-fast](https://github.com/pytorch-labs/gpt-fast) supplies an inspectable optimized generator; [Qwen3 model collection](https://huggingface.co/collections/Qwen/qwen3) is a candidate source of compatible checkpoints and configurations.

### 13.2 IO aware exact attention

**Scope.** Derive stable online softmax using a running maximum, normalization sum, and weighted-value accumulator. Explain tiling without materializing the score matrix. Connect blocks in SRAM to blocks exchanged across GPUs: FlashAttention and ring attention share an accumulation principle but have different scheduling and communication constraints.

**Notebook.** Implement a blockwise PyTorch reference and compare outputs and gradients with ordinary attention, including extreme logits and causal boundaries. Use optimized PyTorch attention for performance experiments; the reference implementation is not expected to achieve kernel-level speedups.

**References.** [FlashAttention](https://arxiv.org/abs/2205.14135) supplies the IO analysis; [FlashAttention-3](https://arxiv.org/abs/2407.08608) supplies the modern hardware case study; [Ring Attention](https://arxiv.org/abs/2310.01889) supplies the distributed connection; [ETH Week 6](https://drive.google.com/file/d/19KFfqDVLdAI3UD90LEGt3_jzebjPUeLt/view), slides 6–10, supplies an effective visual progression.

### 13.3 Compilation and execution overhead

**Scope.** Distinguish kernel time, Python/CPU overhead, dispatch, allocation, and synchronization. Introduce graph capture, compilation, shape specialization, and fusion through their effects on the generator. Explain warmup, compilation cost, recompilation, and why benchmarks must state whether setup is included.

**Notebook.** Profile eager and compiled decode over fixed and varying lengths. Report first-request latency separately from warmed steady-state performance. Inspect a trace to identify the next bottleneck rather than assuming every optimization composes multiplicatively.

**References.** [torch.compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) defines the execution surface; [PyTorch profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) supplies trace tooling; [gpt-fast](https://github.com/pytorch-labs/gpt-fast) connects compilation to real language-model inference.

### 13.4 Accounting for KV and recurrent state

**Scope.** Derive state bytes per token and per request for ordinary, grouped-query, and latent attention. Contrast sequence-growing KV storage with fixed-size recurrent state and mixed hybrids. Include positional state, scales, residual buffers, and metadata so storage formulas match the implementation. Distinguish stored state from bytes read at each decode step.

**Notebook.** Predict peak cache size for the chosen model and validate it at increasing context and batch sizes. Construct configurations from the DeepSeek and Kimi reports without allocating their full weights. Produce a state-budget calculator rather than a misleading universal KV formula.

**References.** [GQA](https://arxiv.org/abs/2305.13245) establishes grouped-query attention; [DeepSeek-V2](https://arxiv.org/abs/2405.04434) describes MLA; [Kimi K3 report](https://arxiv.org/html/2607.24653v1) supports hybrid state accounting; [ETH Week 6](https://drive.google.com/file/d/19KFfqDVLdAI3UD90LEGt3_jzebjPUeLt/view), slides 18–19, supplies a compact comparison.

### 13.5 Quantizing the KV cache

**Scope.** Explain why keys and values may favor different scaling axes, and how positional transformations affect distributions. Cover a simple calibrated baseline, residual high-precision windows, outlier handling, and rotation-based approaches. Separate vector reconstruction error, attention-score error, and downstream task quality; include scales and packing overhead in memory figures.

**Notebook.** Implement a reference cache quantizer and sweep precision and grouping. Measure reconstruction and attention-output error before running held-out retrieval and language tasks. Compare simulated savings with a supported packed implementation; do not claim speedup from fake quantization.

**References.** [KIVI](https://arxiv.org/abs/2402.02750) supplies asymmetric key/value quantization; [KVQuant](https://arxiv.org/abs/2401.18079) supplies pre-RoPE and outlier-aware designs; [KVQuant code](https://github.com/SqueezeAILab/KVQuant) supplies implementation detail; [TurboQuant](https://arxiv.org/abs/2504.19874) is an advanced rotation/vector-quantization reading, with claims evaluated under its stated assumptions.

### 13.6 Reading or retaining less state

**Scope.** Compare sliding windows, attention sinks, query-dependent sparse reads, and cache eviction. Distinguish reducing stored bytes from reducing reads. Introduce reconstruction/replay as another point in the memory–compute tradeoff, and explain why methods trained into an architecture cannot necessarily be attached to an arbitrary checkpoint unchanged.

**Notebook.** Compare full-cache, sink-plus-window, and a simple eviction policy on a fixed model. Sweep where supporting information occurs and test both retrieval and reasoning over retrieved content. Include a quality regression that a single needle test would miss.

**References.** [StreamingLLM](https://arxiv.org/abs/2309.17453) motivates sinks and bounded caches; [H2O](https://arxiv.org/abs/2306.14048) supplies a heavy-hitter eviction method; [Native Sparse Attention](https://arxiv.org/abs/2502.11089) supplies a trained sparse-access comparison; [DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1) supplies the bounded-replay and training-aware cache case study.

### 13.7 Exact speculative decoding

**Scope.** Derive proposal acceptance and the residual correction distribution for a precisely defined target sampler. Explain maximal coupling, acceptance, and expected progress under simplified assumptions. Then derive a wall-clock model including drafting, verification, rejected work, cache maintenance, and batch-dependent target cost. Distributional equivalence is not identical sampled strings or universal acceleration.

**Notebook.** First test empirical frequencies on a small categorical distribution, then implement draft–verify decoding with a compatible small model pair. Sweep draft length and report acceptance, target calls, memory, and measured latency, including a regime where speculation loses.

**References.** [Leviathan et al.](https://arxiv.org/abs/2211.17192) provides the algorithm and proof; [Chen et al.](https://arxiv.org/abs/2302.01318) provides an independent formulation; [MagicDec](https://arxiv.org/abs/2408.11049) explains why context and batching alter the tradeoff; [Smola slides](https://alex.smola.org/posts/45-mlss-efficiency/) provide the maximal-coupling teaching example.

### 13.8 Modern drafting mechanisms

**Scope.** Compare separate draft models, feature-based drafters, multiple prediction heads, and self-speculation. Connect drafting objectives to Chapter 2's auxiliary prediction objectives and Chapter 10's distillation. Explain train–inference mismatch in the drafter, acceptance variability across tasks, and the need to count draft-state overhead.

**Notebook.** Compare a small independent draft with a released compatible feature-based drafter, where available. A reduced head-training experiment can illustrate the objective without promising reproduction of a frontier speedup. Use common prompts and target sampling settings.

**References.** [EAGLE-3](https://arxiv.org/abs/2503.01840) supplies training-time test and multi-layer features; [EAGLE implementation](https://github.com/SafeAILab/EAGLE) supplies model compatibility and code; [DeepSeek-V3](https://arxiv.org/abs/2412.19437) connects multi-token prediction to deployment; [SpecDecode-Bench](https://specdecode-bench.github.io/) supplies workload-sensitive evaluation questions.

### 13.9 Constrained and structured generation

**Scope.** Develop finite-state constraints before introducing grammars and JSON schemas. Explain how subword tokens can span several grammar transitions, how masking changes the target distribution, and why valid syntax does not guarantee a correct answer. Discuss state management and interactions with speculative verification.

**Notebook.** Implement a tokenizer-aware finite-state masker for a small structured output format. Compare unconstrained and constrained outputs on validity, semantic correctness, token count, and latency. Add boundary cases where a token contains multiple characters or punctuation transitions.

**References.** [XGrammar](https://arxiv.org/abs/2411.15100) explains efficient grammar-guided generation; [XGrammar code](https://github.com/mlc-ai/xgrammar) supplies tokenizer/grammar engineering; [Outlines](https://github.com/dottxt-ai/outlines) supplies another implementation perspective; [SGLang](https://arxiv.org/abs/2312.07104) connects constrained execution to a serving runtime.

## Chapter 14 Serving and Post Training Systems

**Purpose.** Manage requests, state, and resources around the generator, then reuse these ideas for training-time rollouts and teacher scoring. Prerequisites are Chapters 6–7 and 13; the post-training section connects to Chapters 9–10. The main artifact is a small serving scheduler and workload replay harness. One-GPU load tests are the default; fleet and disaggregation exercises use simulation or supplied traces.

**Chapter references.** [CMU LLM Systems sample syllabus](https://www.cmu.edu/online/welcome/pdfs/11968-llm-systems-syllabus-f25.pdf) supplies a teaching checklist; [ICML 2024 serving tutorial](https://icml.cc/virtual/2024/tutorial/35229) supplies the algorithms-to-systems framing; [EMNLP 2025 efficient inference tutorial](https://aclanthology.org/2025.emnlp-tutorials.1/) supplies a complementary taxonomy. [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) are production case studies, not dependencies for understanding the simple scheduler.

### 14.1 Workloads and service objectives

**Scope.** Define arrival rate, concurrency, prompt/output distributions, time to first token, inter-token latency, completion latency, throughput, and goodput under service objectives. Distinguish an offline saturated benchmark from an interactive service. Discuss cold starts and the difference between client-observed and server-measured timing.

**Notebook.** Replay synthetic short/long prompt mixtures with fixed seeds and an open-loop arrival process. Plot latency percentiles and admitted/completed requests against load. Show how a throughput improvement can worsen first-token or tail latency.

**References.** [DistServe](https://arxiv.org/abs/2401.09670) supplies goodput and phase-specific service objectives; [Sarathi-Serve](https://arxiv.org/abs/2403.02310) supplies latency–throughput analysis; [vLLM documentation](https://docs.vllm.ai/) supplies practical benchmark and metric definitions to pin with the implementation version.

### 14.2 Batching and scheduling

**Scope.** Develop iteration-level scheduling, joining and leaving a batch, chunked prefill, admission control, cancellation, and backpressure. Explain head-of-line blocking and variable request sizes. Identify the scheduling objective explicitly instead of calling one policy universally optimal.

**Notebook.** Implement static and continuous batching around the Chapter 13 generator, then add prefill chunking. Evaluate mixed-length arrivals and cancellation. Report fairness and starvation alongside aggregate throughput; include a scheduler-only simulation for CPU users.

**References.** [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) establishes iteration-level scheduling; [Sarathi-Serve](https://arxiv.org/abs/2403.02310) supplies chunked-prefill scheduling; [SGLang runtime](https://github.com/sgl-project/sglang) supplies a real scheduler for comparison after the notebook.

### 14.3 Paged state management

**Scope.** Explain logical token blocks, physical KV blocks, block tables, allocation and reclamation, prefix sharing, and copy-on-write. Distinguish paging's reduction of fragmentation from compression of the values themselves. Connect allocation policy to maximum concurrency and cancellation correctness.

**Notebook.** Build a CPU block allocator with invariant tests, then a PyTorch paged-cache reference lookup. Replay ragged requests and compare reserved versus useful bytes with contiguous allocation. Keep the efficient gather/kernel implementation as an ecosystem inspection rather than a short-code speed claim.

**References.** [PagedAttention paper](https://arxiv.org/abs/2309.06180) supplies the memory-management model; [vLLM code](https://github.com/vllm-project/vllm) supplies allocator/runtime implementations; [FlashInfer](https://github.com/flashinfer-ai/flashinfer) supplies production attention primitives and kernel-interface context.

### 14.4 Prefix reuse and hierarchical caching

**Scope.** Define when two requests may safely share cached state: identical model/adapter, tokens, positions, and relevant execution configuration. Develop prefix trees, reference counts, eviction, routing, and cache isolation. Compare moving KV through memory tiers with recomputing prefill, including transfer setup and queueing costs.

**Notebook.** Add a radix/prefix cache to the scheduler and vary prefix repetition and request order. Compare cache-aware and arrival-order policies. Use measured transfer and prefill costs to estimate when host-memory caching helps; test invalidation when the adapter changes.

**References.** [SGLang paper](https://arxiv.org/abs/2312.07104) explains RadixAttention; [LMCache](https://github.com/LMCache/LMCache) supplies hierarchical KV reuse; [Mooncake](https://arxiv.org/abs/2407.00079) supplies distributed cache management; [Smola slides](https://alex.smola.org/posts/45-mlss-efficiency/) supply copy-versus-recompute examples.

### 14.5 Distributed and disaggregated serving

**Scope.** Revisit tensor, pipeline, and expert placement for inference rather than training. Explain when separate prefill/decode pools improve utilization and when KV transfer or queueing dominates. Include heterogeneous hardware, model replicas, locality, and request routing without assuming that a training configuration is also a good serving configuration.

**Notebook.** Simulate colocated and split pools using measured prefill/decode/transfer curves. Sweep workload mix and network bandwidth; identify a regime where disaggregation helps and one where it hurts. Optional multi-GPU execution validates selected cases.

**References.** [DistServe](https://arxiv.org/abs/2401.09670) supplies the disaggregation argument; [Mooncake](https://arxiv.org/abs/2407.00079) supplies KV-centric architecture; [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) supplies a contemporary implementation study, with version-specific details kept outside the core derivation.

### 14.6 Serving multiple adapted models

**Scope.** Separate shared base computation from request-specific low-rank updates. Discuss merged versus unmerged adapters, heterogeneous batches, adapter loading/eviction, rank variation, and cache correctness. Compare throughput and memory with independently loaded fine-tuned models.

**Notebook.** Serve several small adapters over one base and compare serial adapter switching with batched reference computation. Add adapter identity to prefix-cache keys. Inspect production grouped operations after verifying a transparent PyTorch implementation.

**References.** [S-LoRA](https://arxiv.org/abs/2311.03285) supplies adapter and paging design; [Punica](https://arxiv.org/abs/2310.18547) supplies multi-tenant computation; [vLLM](https://github.com/vllm-project/vllm) supplies deployable LoRA serving for a practical comparison.

### 14.7 Systems for rollouts and teacher scoring

**Scope.** Treat post-training as a pipeline of generation, reward/teacher evaluation, and parameter updates. Compare colocated and disaggregated workers, weight synchronization, parameter resharding, model versions, asynchronous sampling, and stale trajectories. OPD adds teacher-distribution access and scoring bandwidth; count these explicitly rather than treating teacher feedback as free.

**Notebook.** Instrument a small RL/OPD loop from Chapters 9–10 and attribute time and peak memory to each phase. Simulate delayed trajectories tagged by policy version. Compare serial versus overlapped execution without confusing scheduling gains with a changed optimization objective.

**References.** [verl](https://github.com/verl-project/verl) supplies a PyTorch post-training systems reference; [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) supplies an alternative worker layout; [AReaL](https://github.com/inclusionAI/AReaL) supplies asynchronous execution; [DeepSeek-V4.1-Flash report](https://arxiv.org/html/2609.19969v1), Section 5.2, supplies a frontier asynchronous OPD case study.

### 14.8 Operating and evaluating a service

**Scope.** Cover load tests, metrics, memory exhaustion, model loading, startup probes, checkpoint rollout/rollback, autoscaling, and fault recovery. Explain privacy-conscious logging and tenant isolation where they affect system design. Report cost per successful request under a stated quality and latency target, rather than raw token cost alone.

**Notebook.** Build a reproducible load-test report with cold/warm runs and injected worker failures. Record configuration, model/tokenizer revisions, queueing policy, hardware, arrival trace, and output limits. Demonstrate safe cancellation and allocator recovery after an interrupted request.

**References.** [vLLM documentation](https://docs.vllm.ai/) supplies serving and operational metrics; [SGLang documentation](https://docs.sglang.ai/) supplies benchmarking and deployment recipes; [MLPerf Inference](https://github.com/mlcommons/inference) supplies standardized workload/reporting concepts to adapt to text-only experiments.

## Chapter 15 Evaluation and Experimental Methodology

**Purpose.** Consolidate the evaluation tools used throughout the part and expose assumptions that can reverse a comparison. This chapter is read in stages: likelihood and task mechanics accompany early training; judges and efficiency comparisons follow adaptation. The main artifact is a versioned evaluation suite and report. CPU scoring and small-model subsets are core; expensive judging and long-context tests are optional.

**Chapter references.** [NeurIPS 2024 LLM evaluation tutorial](https://neurips.cc/virtual/2024/tutorial/99524) and [ICML 2024 evaluation tutorial](https://icml.cc/virtual/2024/tutorial/35227) supply teaching overviews. [HELM](https://arxiv.org/abs/2211.09110) supplies a multi-metric evaluation perspective. [EleutherAI evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Hugging Face LightEval](https://github.com/huggingface/lighteval) supply task implementations to inspect and pin, not unquestioned ground truth.

### 15.1 Likelihood metrics and calibration

**Scope.** Define token likelihood, perplexity, bits per byte, masking conventions, context-window evaluation, and domain-weighted aggregation. Explain why token perplexities are not comparable across tokenizers. BPB normalizes a practical tokenized-text score; it does not magically marginalize every possible segmentation or remove preprocessing differences. Separate next-token calibration from confidence in a final answer.

**Notebook.** Score identical raw texts under two tokenizers/models with transparent denominators and window policies. Compare per-domain and pooled results, then plot calibration under a precisely stated event definition. Add tests for prompt masking and shifted labels.

**References.** [Transformers perplexity guide](https://huggingface.co/docs/transformers/en/perplexity) explains fixed-length evaluation; [HELM](https://arxiv.org/abs/2211.09110) motivates multiple metrics; [evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) supplies executable likelihood tasks and aggregation conventions.

### 15.2 Task scoring mechanics

**Scope.** Compare choice scoring, length normalization, generative answers, answer extraction, and execution-based tests. Derive the pass@k estimator under its sampling assumptions and distinguish it from majority voting or a deployed selection policy. Explain why scoring only answer options may mischaracterize a model intended to reason before answering.

**Notebook.** Implement one multiple-choice and one bounded code/math evaluator. Change only the scoring convention or parser and inspect disagreements. Estimate pass@k from a common sample pool, report uncertainty, and keep sampling settings fixed.

**References.** [HumanEval paper](https://arxiv.org/abs/2107.03374) supplies pass@k; [HumanEval code](https://github.com/openai/human-eval) supplies the original evaluation protocol; [LightEval](https://github.com/huggingface/lighteval) supplies practical scoring implementations; [IFEval](https://arxiv.org/abs/2311.07911) supplies verifiable instruction constraints.

### 15.3 Building a useful evaluation suite

**Scope.** Select a small development suite and a separate reporting suite covering language tasks, knowledge, reasoning, code, instruction following, and multilingual transfer. Match tests to the intended use and model scale. Distinguish benchmark names, dataset versions, subsets, prompts, and evaluation dates. Avoid treating a single aggregate leaderboard rank as the learning objective.

**Notebook.** Build a task manifest with frozen revisions and explicit scoring. Include a small fast suite for every training/compression experiment and a slower held-out suite. Analyze which tasks change model ranking and which are too easy, too hard, or too noisy for the study.

**References.** [GPQA](https://arxiv.org/abs/2311.12022) supplies difficult knowledge/reasoning questions; [IFEval](https://arxiv.org/abs/2311.07911) supplies automatically checkable instructions; [LiveCodeBench](https://livecodebench.github.io/) supplies temporally organized code evaluation; [HELM](https://crfm.stanford.edu/helm/) supplies scenario and metric organization.

### 15.4 Evaluating long context

**Scope.** Distinguish accepted input length from retrieval, composition, instruction following, and reasoning over that input. Vary supporting-information position, distractors, query specificity, and task complexity. Test cache approximations and hybrid architectures on several task families; synthetic retrieval success alone is insufficient evidence of broad long-context capability.

**Notebook.** Combine a controllable synthetic retrieval test with document-based question answering or summarization. Sweep context length and evidence position. Repeat after KV quantization or eviction and compare task-specific degradation rather than only an average score.

**References.** [RULER](https://arxiv.org/abs/2404.06654) supplies controlled long-context tasks; [HELMET](https://arxiv.org/abs/2410.02694) supplies diverse application-oriented evaluation; [Lost in the Middle](https://arxiv.org/abs/2307.03172) motivates position sensitivity; [HELMET code](https://github.com/princeton-nlp/HELMET) supplies task construction and evaluation.

### 15.5 Human and model judges

**Scope.** Develop rubrics, blinded comparisons, ties, repeated ratings, and agreement. Examine position, verbosity, style, and self-preference biases. Fit pairwise preference models and distinguish estimated preference from factual correctness. Compare a small local judge against a human-labeled subset before scaling automated evaluation.

**Notebook.** Judge the same response pairs in both orders and with formatting normalized. Fit a Bradley–Terry model, bootstrap prompts, and compare ranking uncertainty. Include a test where polished but incorrect answers reveal judge limitations.

**References.** [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) establishes MT-Bench/Arena and judge concerns; [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) supplies automatic comparison tooling; [Chatbot Arena paper](https://arxiv.org/abs/2403.04132) supplies crowd preference methodology; [HELM](https://arxiv.org/abs/2211.09110) supplies broader evaluation dimensions.

### 15.6 Statistical comparisons and ablations

**Scope.** Explain paired sampling, confidence intervals, bootstrap units, seed variability, multiple comparisons, and minimum useful effect sizes. Separate sampling variability from training variability. Match compute or resource budgets when the scientific question requires it, and keep implementation changes separate from algorithmic changes.

**Notebook.** Reanalyze an earlier LoRA or quantization experiment with paired confidence intervals and several seeds. Show how treating tokens or repeated answers from one prompt as independent can overstate precision. Write an ablation table with a clear question for each row.

**References.** [Dror et al., A Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/) supplies test-selection guidance; [HELM](https://arxiv.org/abs/2211.09110) supplies multi-metric comparisons; [evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) supplies practical task-level aggregation to audit.

### 15.7 Contamination and memorization

**Scope.** Connect exact/fuzzy data overlap to Chapter 1, then examine paraphrases, temporal splits, task perturbations, and membership inference. Treat membership scores as imperfect evidence, not definitive proof that a benchmark example was in training. Distinguish data leakage, template familiarity, and genuine task generalization.

**Notebook.** Inject known duplicates and paraphrases into a small controlled training corpus and test several detection methods. Compare detection on a known training history with the uncertainty of an opaque pretrained model. Perturb task constants and templates to assess robustness.

**References.** [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) connects training duplication to behavior; [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) motivates memorization analysis; [GSM-Symbolic](https://arxiv.org/abs/2410.05229) supplies controlled math perturbations; [LiveCodeBench](https://livecodebench.github.io/) supplies temporal evaluation practice.

### 15.8 Capability and efficiency together

**Scope.** Build quality–memory–latency–cost frontiers using fixed tasks and explicit workload distributions. Separate capability measured at one reasoning budget from latency measured at another. Include adaptation/preparation costs where relevant, and distinguish an engineering improvement from statistical noise or a lower-quality operating point.

**Notebook.** Combine earlier baseline, adapted, quantized, and speculative models into a single report. Show Pareto frontiers and dominated configurations, with uncertainty and hardware details. Publish the manifest and measurement script so the comparison can be rerun after dependency changes.

**References.** [MLPerf Inference](https://github.com/mlcommons/inference) supplies reporting discipline; [DistServe](https://arxiv.org/abs/2401.09670) supplies quality-of-service-aware throughput concepts; [HELM](https://arxiv.org/abs/2211.09110) supplies multidimensional model evaluation; [SpecDecode-Bench](https://specdecode-bench.github.io/) supplies a workload-sensitive inference case study.

## Chapter 16 Anatomy of Frontier Language Models

**Purpose.** Assemble the preceding mechanisms into complete model and deployment stories. This is an analytical capstone, not a requirement to train or serve a frontier model locally. Prerequisites are the relevant earlier chapters, especially architectures, parallelism, post-training, and inference. The main artifacts are source-linked architecture/recipe maps, cost calculations, and reduced experiments. Model revisions and report versions must be explicit.

**Chapter references.** [DeepSeek-V4.1-Flash report](https://arxiv.org/html/2609.19969v1), [Kimi K3 report](https://arxiv.org/html/2607.24653v1), and [Qwen3 report](https://arxiv.org/abs/2505.09388) are primary case-study sources. [Smola's efficiency lecture](https://alex.smola.org/posts/45-mlss-efficiency/) supplies the common resource-accounting perspective. Reported ablations support narrower claims than final benchmark scores; architecture, implementation, and claimed performance should remain separate columns in the analysis.

### 16.1 Reading a technical report critically

**Scope.** Extract model configuration, objectives, training stages, data descriptions, numerical formats, parallelism, post-training, and serving assumptions. Distinguish disclosed facts from explanatory hypotheses and missing information. Learn to follow a claim to the relevant ablation rather than inferring causality from a strong final model.

**Notebook.** Build a report schema and fill it for two models. Trace several performance claims to experiments, record the compared baseline and resource budget, and mark which ingredients are supported by ablations. Check whether configuration files agree with the report version.

**References.** [Qwen3 report](https://arxiv.org/abs/2505.09388) supplies a model-family example; [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) supplies architecture/training/system connections; [Kimi K3 report](https://arxiv.org/html/2607.24653v1) supplies a frontier comparison; [HELM](https://arxiv.org/abs/2211.09110) supplies evaluation questions for interpreting reported results.

### 16.2 A dense model reference case

**Scope.** Trace an accessible model family from tokenizer and block configuration through pretraining, post-training, and deployment. Use the same small pretrained family as the practical notebooks where suitable, with base/instruction or reasoning variants only when actually available. Separate properties shared by the family from variant-specific training recipes.

**Notebook.** Generate a bill of materials directly from configuration and weights: parameter groups, active operations, state bytes, and supported precision. Reconcile predictions with the Chapter 4 profiler and link each training-stage claim to the report.

**References.** [Qwen3 report](https://arxiv.org/abs/2505.09388) is the main candidate case; [Qwen3 repository](https://github.com/QwenLM/Qwen3) supplies usage and model links; [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) supplies a concrete checkpoint; [OLMo 2 report](https://arxiv.org/abs/2501.00656) is a useful alternative when training transparency is the stronger teaching priority.

### 16.3 DeepSeek V4.1 Flash

**Scope.** Trace the disclosed causal encoder–decoder design, compressed sparse attention, cross-layer state reuse, low-precision cache, and bounded replay. Connect these to asynchronous post-training and on-policy distillation. Restrict the chapter to the language backbone and general optimization/system mechanisms; agent environments and vision components belong in other parts.

**Notebook.** Reconstruct state and communication budgets from the report/configuration. Implement a reduced cross-layer cache-sharing or replay example with correctness/approximation tests. Map each mechanism to the chapter where it was introduced and distinguish architectural savings from serving-policy savings.

**References.** [Technical report](https://arxiv.org/html/2609.19969v1), Sections 2–3 and 5, supplies architecture, infrastructure, and OPD; [official model card and files](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) supply configuration and deployment artifacts; [official release](https://www.deepseek.com/en/news/deepseek-v4-1-flash/) supplies release context; [DeepSeek-V3](https://arxiv.org/abs/2412.19437) supplies the earlier design baseline.

### 16.4 Kimi K3

**Scope.** Connect Kimi Delta Attention and gated MLA, Attention Residuals, Stable LatentMoE, balancing, optimizer choices, and low-precision execution. Use the disclosed hybrid layout to reason about state, active computation, and expert communication. Treat the report's full-system performance as evidence about the complete model, not proof of each component's contribution.

**Notebook.** Build a configuration-driven cost calculator and one reduced hybrid/residual experiment. Compare its predicted state growth with a full-attention baseline. Trace the routing/balancing design back to Chapters 3 and 7 and explain what a small experiment cannot reproduce.

**References.** [Kimi K3 report](https://arxiv.org/html/2607.24653v1), especially Sections 2–4, is the main technical source; [model card](https://huggingface.co/moonshotai/Kimi-K3) supplies configuration; [official repository](https://github.com/MoonshotAI/Kimi-K3) supplies the released report/artifacts; [ETH Week 5](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view), slides 41–42, supplies a teaching treatment of quantile balancing.

### 16.5 Comparing complete designs

**Scope.** Compare approaches to sparse capacity, long context, numerical precision, residual computation, optimization, and capability transfer. Use common accounting units and explicitly different workload regimes. Explain interactions: a smaller cache can enable larger batches, which can change whether a weight-quantization or speculation optimization helps.

**Notebook.** Construct a side-by-side design matrix and cost curves without inventing missing training details. Select a few mechanisms with public implementations for reduced ablations. Separate measured results from projected frontier-scale behavior.

**References.** [DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1), [Kimi K3](https://arxiv.org/html/2607.24653v1), and [Qwen3](https://arxiv.org/abs/2505.09388) supply the comparison cases; [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) supplies analytical cost-model techniques, translated into PyTorch exercises rather than adopting its implementation framework.

### 16.6 Designing under a fixed budget

**Scope.** Specify a task distribution, available data, training budget, hardware, context requirements, quality target, and serving objective. Choose a pretrained starting point or justify pretraining; select adaptation, compression, and deployment mechanisms. State uncertainties and define the experiments that would resolve them before committing substantial compute.

**Notebook.** Produce a reproducible design dossier: model/data manifest, capacity estimates, experiment matrix, quality gates, and serving replay. Require at least one measured baseline and one rejected design with evidence. The objective is defensible engineering judgment, not reproducing a frontier leaderboard.

**References.** [Stanford CS336](https://cs336.stanford.edu/) supplies end-to-end assignment discipline; [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) supplies training-budget reasoning; [Smola efficiency lecture](https://alex.smola.org/posts/45-mlss-efficiency/) supplies inference-budget reasoning; [MLPerf Inference](https://github.com/mlcommons/inference) and [evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) supply reproducible measurement practices.

## Shared source guide and authoring notes

### University teaching sources

These five institutions provide complementary examples of leading university teaching. This is a curriculum comparison, not a claim that one ranking establishes a unique “top five.” Their materials inform the sequence and exercises; frontier papers and implementations provide the newer technical detail.

| Institution and source | What to borrow | Where it enters this plan |
|---|---|---|
| [Stanford CS336: Language Modeling from Scratch, Spring 2026](https://cs336.stanford.edu/) | The progression through tokenization/model construction, systems, scaling, data, and alignment/reasoning; explicit implementation assignments and resource accounting | Chapters 1–7, 9, 11, 15; particularly the small trainer and data experiments |
| [MIT 6.5940: EfficientML.ai, Fall 2026](https://hanlab.mit.edu/courses/2026-fall-65940) | Hardware-aware efficiency, compression, quantization, and deployment experiments | Chapters 4, 12–14; use the published syllabus and available materials, without treating lectures scheduled after the reference date as already delivered |
| [CMU 11-968: LLM Systems, sample syllabus](https://www.cmu.edu/online/welcome/pdfs/11968-llm-systems-syllabus-f25.pdf) | The connection between GPU execution, training systems, and serving; performance-oriented assignments | Chapters 4, 6–7, 13–14; adapt kernel topics into PyTorch reference implementations and profiling |
| [UC Berkeley: Scalable AI, Spring 2026 syllabus](https://scalable-ai.eecs.berkeley.edu/assets/syllabus.pdf) | Distributed scaling, communication, hardware constraints, and project-oriented systems work | Chapters 4–7 and 14; replace cluster-dependent core exercises with reduced runs and traces |
| [Oxford: Large Language Models, professional programme](https://www.cs.ox.ac.uk/professionalprogramme/subjects/LLM.html) | A complementary application/adaptation perspective and an end-to-end view of language-model use | Chapters 2, 8–11, 15; this source is a professional-programme course, not a full public undergraduate lecture sequence |

### Conference tutorials and instructional surveys

| Source | Use in the book |
|---|---|
| [ICML 2024: Towards Efficient Generative Large Language Model Serving](https://icml.cc/virtual/2024/tutorial/35229), Xupeng Miao and Zhihao Jia | Organize the relationship between decoding algorithms and runtime systems in Chapters 13–14. Follow its links to source papers rather than treating the tutorial as the implementation specification. |
| [ICML 2024: Challenges in Language Model Evaluations](https://icml.cc/virtual/2024/tutorial/35227), Lintang Sutawika and Hailey Schoelkopf | Design Chapter 15's scoring-convention experiments and reporting requirements. |
| [ICML 2024: Physics of Language Models](https://icml.cc/virtual/2024/tutorial/35223), Zeyuan Allen-Zhu | Design controlled synthetic experiments on learning and reasoning in Chapters 1, 5, and 11; distinguish what a controlled experiment establishes from general frontier-model behavior. |
| [NeurIPS 2024: Evaluating Large Language Models—Principles, Approaches, and Applications](https://neurips.cc/virtual/2024/tutorial/99524) | Provide a broad evaluation taxonomy for Chapter 15 and the recurring evaluation harness. |
| [ACL 2024 tutorial programme](https://2024.aclweb.org/program/tutorials/) | Use computational expressivity as optional architecture context; text simplification/revision as adaptation tasks; adversarial vulnerabilities as failure-analysis examples. These are different tutorials, not a single comprehensive LLM-systems course. |
| [EMNLP 2025: Efficient Inference for Large Language Models—Algorithm, Model, and System](https://aclanthology.org/2025.emnlp-tutorials.1/) | Cross-check coverage of compression, decoding, and serving in Chapters 12–14. |

Conference provenance is recorded separately from technical content. The section bibliographies include original papers associated with ICLR and other venues; a conference paper or workshop talk is not relabeled as an official tutorial. Older tutorials remain useful for exposition, while the 2026 reports support the current frontier case studies.

### The two slide collections supplied for this project

[Alex Smola's MLSS efficiency post and slides](https://alex.smola.org/posts/45-mlss-efficiency/) supply a recurring explanation of compute, bandwidth, memory capacity, weights, KV state, speculation, and serving. Their main homes are Chapters 4 and 12–14, with connections to lifecycle cost in Chapter 5 and integrated designs in Chapter 16. Preserve the downloaded deck revision when authoring: the linked deck can be revised independently of the post date. A projected saving, a measured kernel result, and a full-service speedup must remain distinct.

The [four ETH slide decks in the supplied folder](https://drive.google.com/drive/folders/1pUG7TB5VRMQZSa9-9wxkGOZEzHjP8LC3) provide the distributed-computing progression. References below and within chapters use printed slide numbering, because animation builds can produce several PDF pages per numbered slide.

| Deck | Main content to use | Destination |
|---|---|---|
| [Week 3: collective communication](https://drive.google.com/file/d/1LeR_e8y9m7gosaGQ5Aeo-m23hG-aovJx/view) | Collective semantics, layouts, communication costs, and distributed computation foundations | Sections 6.1–6.2; revisit in Chapter 7's topology/configuration exercises |
| [Week 4: data and tensor parallelism](https://drive.google.com/file/d/1GHsvL3S8dfLGvjrDGb0z7GbzOWOWa4yU/view) | Replication, tensor partitioning, forward/backward communication, and overlap | Sections 6.3–6.6 |
| [Week 5: pipeline parallelism, ZeRO, and MoE](https://drive.google.com/file/d/17wRnF9oKWQbNeK1-oHjkRJI0o6lvhJmV/view) | Pipeline schedules; sharding and prefetching; expert placement and dispatch | Sections 6.7–6.8 and 7.1–7.3, 7.5; prefetching receives an explicit experiment rather than being a footnote |
| [Week 6: context parallelism and efficient attention](https://drive.google.com/file/d/19KFfqDVLdAI3UD90LEGt3_jzebjPUeLt/view) | Blockwise exact attention, context/sequence partitioning, and attention-state accounting | Sections 7.4, 13.2, and 13.4; connect the same accumulator mathematics across local and distributed execution |

The decks motivate two distributed chapters: first correctness and sharded computation, then schedules, topology, and multidimensional scaling. Production PyTorch code and primary papers remain necessary because a slide diagram usually omits collective ordering, autograd details, failure handling, and version-specific APIs.

### Popular tutorials and implementation-oriented references

| Source | Intended role |
|---|---|
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1) | Tokenization, dataset handling, pretrained-model interfaces, and language-task examples for Chapters 1–2 and 8 |
| [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/en/unit0/1) | Accessible adaptation workflow and small-model exercises for Chapters 8–10; omit its multimodal material |
| [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) | A visual/practical companion for Chapters 6–7; adapt implementation details to the tested PyTorch stack |
| [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) | Data-mixture selection, training experiments, and realistic small-model decisions in Chapters 1 and 5 |
| [Karpathy: Neural Networks—Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) and [build-nanogpt](https://github.com/karpathy/build-nanogpt) | Compact implementation progression; useful for the reduced trainer, without repeating all introductory neural-network material |
| [Sebastian Raschka: from-scratch materials](https://www.sebastianraschka.com/from-scratch/) | Readable PyTorch explanations and controlled construction of LM components |
| [RLHF Book](https://rlhfbook.com/) | Objective definitions, preference data, reward models, and post-training exposition for Chapters 9–11 |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Mathematical systems models for Chapters 4–7 and 14; borrow derivations, while all book implementations remain PyTorch |
| [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) | Design the rank, module-selection, and learning-rate experiments in Section 8.6; treat its findings as empirical evidence under its experimental conditions |
| [PyTorch torchtitan](https://github.com/pytorch/torchtitan), [torchtune](https://github.com/pytorch/torchtune), and [torchao](https://github.com/pytorch/ao) | Inspectable training, adaptation, and low-precision implementation references for Chapters 5–8 and 12 |

### Candidate model and data inventory

This inventory makes notebook design concrete without prematurely selecting one permanent model or dataset. Model cards, dataset cards, configuration files, and revision identifiers belong in the eventual reproducibility manifest. A large public corpus is a source from which to select a documented subset, not a download requirement for every learner.

| Candidate | Proposed use and constraints |
|---|---|
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) and [Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3) | Capable small-model experiments and compatible family comparisons. Choose base/adapted variants explicitly; check chat templates and thinking modes. Use larger members only where measured memory permits. |
| [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) and [SmolLM repository](https://github.com/huggingface/smollm) | An accessible alternative for pretraining/adaptation continuity and training-data transparency. |
| [OLMo](https://github.com/allenai/OLMo) | Training-code, configuration, and checkpoint-history studies; particularly useful when the experiment needs more provenance than a final checkpoint provides. |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | Frontier architecture/configuration inspection and source-based case studies, not single-24GB-GPU execution targets. |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Streamed, versioned pretraining and calibration subsets for Chapters 1, 5, and 12. Freeze sample selection and keep evaluation text separate. |
| [The Stack](https://huggingface.co/datasets/bigcode/the-stack) | Code tokenization and corpus-curation examples. Select a documented permitted subset and record any access requirements. |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Synthetic-data provenance, selection, diversity, and ablation examples. |
| [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) and [Smol SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Instruction serialization, data selection, and adapter experiments in Chapter 8. Preserve source labels when sampling. |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | Preference-pair and reward-model exercises. Explain the collection setting and distribution limits rather than treating the labels as universal preferences. |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | Candidate offline reasoning traces for Chapters 10–11, after checking source/model provenance and overlap with evaluation. |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) and [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Accessible evaluation examples at different difficulty levels; use explicit splits and acknowledge possible pretrained-model exposure. |
| [HumanEval](https://github.com/openai/human-eval), [IFEval](https://arxiv.org/abs/2311.07911), [RULER](https://arxiv.org/abs/2404.06654), and [HELMET](https://github.com/princeton-nlp/HELMET) | Bounded code, instruction-following, and long-context evaluation. Keep execution, parser, prompt, and generation settings under version control. |

For the embedding notebook, the model/paper/code choices are given in Section 2.7. Use a modest retrieval corpus with explicit query/document relevance judgments; the notebook should expose pooling, normalization, negatives, and ranking metrics, rather than reduce the exercise to an embedding API call.

### Integrating the part with D2L

Use the [current D2L structure](https://d2l.smola.org/) as the prerequisite map. During editing, audit actual chapter labels and cross-references before replacing the existing Language Models part. This plan uses conceptual dependencies instead of assuming that current chapter numbers are permanent.

| Earlier material | Reuse here | New material in this part |
|---|---|---|
| Probability, information theory, and numerical computation | Likelihood, divergences, precision, entropy coding | Tokenization-dependent metrics, policy/reference divergences, weight/state compression |
| Attention and transformers | Basic attention, masking, and backpropagation | Modern pretrained blocks, sparse/latent/hybrid designs, IO-aware execution, and frontier integration |
| Optimization | Gradient methods, regularization, schedules | Training stability at scale, data/compute allocation, low precision, and adaptation constraints |
| State-space and recurrent models | Recurrences and state representations | Language-specific delta/gated recurrences, hybrid attention, and state-budget comparisons |
| Reinforcement learning | Policy gradients, baselines, importance sampling | Preference objectives, verifiable rewards, group methods, asynchronous rollouts, OPD/OPSD |
| General model evaluation | Generalization and uncertainty | Chat/task scoring, judges, contamination, long-context behavior, and quality–cost frontiers |

Retain useful historical embedding material as optional background and redirect its main practical learning objectives into Section 2.7. Keep language-only retrieval examples bounded; the future agents part can build on retrieval and serving without moving agent orchestration into this part.

### Notebook authoring contract

Each section's notebook proposal should become an explicit experimental specification before implementation:

1. **Question and result:** state the mechanism or hypothesis and the figure/table/assertion that will answer it.
2. **Derivation and source:** name the objective, assumptions, key equations, and the paper section or implementation being followed. Cite borrowed code or figures separately from conceptual inspiration.
3. **Artifacts:** pin the checkpoint/tokenizer, dataset subset and split, software versions, and random seeds. Reuse previous chapter outputs where practical.
4. **Resource envelope:** provide a measured default configuration and a reduced fallback; state when a result uses a cluster trace, simulation, or externally produced teacher scores.
5. **Correctness:** compare against a simple baseline, numerical invariant, enumerated case, or trusted implementation before making performance claims.
6. **Comparison:** control data, decoding, and resource budgets; include quality together with memory, latency, throughput, or cost when relevant.
7. **Interpretation:** explain failures, scope of the evidence, and whether the result is a measured observation or an extrapolation.

Cross-reference the section that owns an algorithm rather than rederiving it repeatedly: Chapter 3 owns sparse/hybrid model mechanisms; Chapters 6–7 own distributed execution; Chapter 9 owns reward-based optimization; Chapter 10 owns distillation objectives; Chapter 12 owns weight compression; Chapter 13 owns token generation and state; Chapter 14 owns request/worker scheduling. Chapter 16 integrates these mechanisms and evaluates the evidence for their combined design.

### Source maintenance and alternative proposal

The Gemini proposal and its four research inventories supplied in the sibling `d2l-scan` directory are secondary planning inputs. Their university/tutorial inventories and implementation suggestions informed source discovery. The 10-chapter/78-section organization, directory blueprint, and model choices are not adopted as constraints. Technical claims in this document point to primary sources or explicitly identified teaching materials.

Local planning inputs (not in this repository): the Gemini structural proposal and its university, conference-tutorial, popular-tutorial and SOTA inventories.

At notebook authoring time, maintain a compact source manifest with URL, title, author/organization, paper version or repository commit, retrieval date, supported claims, and notebook usage. Keep model-report comparisons dated. Recheck living documentation and changed model cards, and retain the source revision behind a published experimental result. The outline is detailed enough to author from, while model choice, final ordering, and repository slugs remain editorial decisions.
