# Tools for Deep Learning appendix rewrite

Status: plan recorded before implementation; implementation completed pending publication.

## Goals and chapter order

Rebuild the final chapter as seven pedagogical notebooks, in this order:

1. **Interactive Development with JupyterLab and VS Code** (`interactive-development.md`, replacing `jupyter.md`)
2. **Hosted Notebooks: Colab and Kaggle** (`hosted-notebooks.md`, replacing `colab.md`)
3. **Renting Cloud Accelerators** (`cloud-instances.md`, replacing `aws.md`)
4. **Choosing Hardware for Deep Learning** (`hardware.md`, replacing `selecting-servers-gpus.md`)
5. **Models, Datasets, and the ML Software Ecosystem** (`software-ecosystem.md`, new)
6. **Training Systems** (`training-systems.md`, new)
7. **Model Serving** (`model-serving.md`, new)

Delete the SageMaker notebook and its navigation/numbering entries. Keep the generated
utility and `d2l` API pages as HTML-only reference material, outside the seven-notebook
pedagogical sequence. Move the contribution guide out of this chapter in a later
front-matter/developer-documentation pass; do not mix it into these seven notebooks.

## Hosted notebook publication

Adapt d2l.ai's stable page-to-GitHub-notebook mapping. Instead of separate repositories
per framework, publish a generated orphan branch named `notebooks` in this repository:

```text
pytorch/<chapter>/<notebook>.ipynb
jax/<chapter>/<notebook>.ipynb
numpy/<chapter>/<notebook>.ipynb
manifest.json
README.md
```

Policy:

- PyTorch is primary, JAX secondary.
- Do not publish TensorFlow or MXNet hosted notebooks.
- Publish one NumPy notebook for framework-independent sources rather than duplicating it
  under PyTorch and JAX.
- Preserve stable code-cell IDs and readable executed outputs, but normalize notebook
  metadata for hosted Python kernels.
- Insert a short, idempotent setup cell only where dependencies or the local `d2l`
  package are required. Pin the source revision recorded in the manifest.
- Do not commit generated notebooks to the authoring branch.
- Generate the branch in a temporary Git worktree and replace its contents atomically.
- Provide dry-run/check modes; pushing the branch is an explicit publication action.

The manifest maps each HTML page key to available variants and their GitHub/raw URLs.
The HTML build consumes a generated copy of this manifest.

## Dynamic site controls

Add a **Run notebook** control near the existing Slides control. It must:

- derive the current page key and selected framework;
- use the NumPy variant automatically for framework-independent pages;
- otherwise offer PyTorch or JAX according to the current framework selection;
- offer **Open in Colab**, **Open in Kaggle**, and **Download notebook**;
- disable unsupported combinations with an explanatory tooltip;
- update when the framework tab changes, including cross-window preference changes;
- remain usable on narrow/mobile pages and pages without a right-hand TOC;
- use accessible link text, focus states, and `rel="noopener noreferrer"`.

Colab URLs follow the supported GitHub form:

```text
https://colab.research.google.com/github/smolix/d2l-neu/blob/notebooks/<variant>/<chapter>/<file>.ipynb
```

Kaggle support is conditional on an end-to-end import test. The former public
`kernels/welcome?src=...` route was tested on July 15, 2026 and returned HTTP 404.
Canonical notebooks must therefore be published through a D2L Kaggle account with the
official Kaggle CLI; verified URLs are supplied through `hosted_notebooks_kaggle.json`.
Until then the visible Kaggle control is disabled with an explanation. The download link
points at the public raw notebook and remains the provider-independent fallback.

## Notebook content

### 1. Interactive Development with JupyterLab and VS Code

Teach kernel state and reproducible execution; running the downloaded `uv` environment;
the JupyterLab workspace, terminals, kernels, restart/run-all, and debugging; VS Code's
notebook editor, kernel selection, variables, diffs, and remote workflows; secure SSH
tunneling; and the D2L Tools extension for generated framework views, source sync,
capture, linting, and slide preview. Remove `notedown`, classic Notebook screenshots,
`jupyter_contrib_nbextensions`, and obsolete timing-extension instructions.

### 2. Hosted Notebooks: Colab and Kaggle

Teach ephemeral compute versus persistent notebooks/data; accelerator selection;
runtime/package drift; setup cells; saving and exporting work; Colab copies and custom
runtimes; Kaggle accelerators, datasets, internet access, versions, outputs, and CLI;
portability and secrets; a comparison table; and the site's launch controls.

### 3. Renting Cloud Accelerators

Teach provider categories (hyperscalers, specialist GPU clouds, marketplaces), workload
and GPU selection, VRAM/interconnect/CPU/RAM/storage/network constraints, full cost,
spot/interruptible capacity, trust and data sensitivity, images/containers, SSH and
tunneled Jupyter, persistent checkpoints, budget controls, and teardown. Use a
vendor-neutral worked path and short provider-specific reference boxes rather than
console screenshots or manual CUDA installation.

### 4. Choosing Hardware for Deep Learning

Start from training versus inference workloads. Teach model/training/KV memory estimates,
compute versus bandwidth, discrete VRAM versus unified memory, representative consumer
and edge systems, one-GPU workstation design, power/cooling/noise/circuit limits,
software compatibility, storage, used hardware, multi-GPU topology, and buy-versus-rent
economics. Keep volatile device tables dated and separate from durable principles.

### 5. Models, Datasets, and the ML Software Ecosystem

Map discover -> inspect -> pin -> download/cache -> load -> adapt -> evaluate -> publish
-> deploy. Use Hugging Face Hub as the running example: model/data cards, revisions,
Transformers, Datasets, Tokenizers, Accelerate, PEFT, TRL, Diffusers, and safetensors.
Teach artifact anatomy, formats (safetensors, adapters, ONNX, GGUF), licensing,
provenance, gated assets, `trust_remote_code`, tokens, caching, and alternatives such as
Kaggle, ModelScope, timm/PyTorch Hub, OpenMMLab, and MLX/GGUF communities. Briefly orient
readers to experiment/artifact tracking without turning the notebook into a logo catalog.

### 6. Training Systems

Teach the scaling ladder from one accelerator through DDP, FSDP2, tensor/pipeline/
context/expert parallelism; the memory budget; mixed precision, accumulation,
checkpointing, sharding, and offload; framework-native entry points; Accelerate,
DeepSpeed, Ray Train, Lightning/Fabric; Megatron-Core, NeMo, and Nanotron; data feeding,
distributed checkpoints, preemption recovery, reproducibility, and throughput/utilization
monitoring. Keep executable work modest and treat multi-GPU commands as inspected
examples rather than pretending they run everywhere.

### 7. Model Serving

Distinguish batch inference, local interactive use, one-user APIs, and multi-user serving.
Teach TTFT, TPOT, throughput, goodput, model/KV capacity, continuous batching, paged KV,
prefix caching, quantization, structured output, speculation, and scaling. Progress from
Ollama/llama.cpp/MLX to vLLM and SGLang, then explain TensorRT-LLM, Triton, NIM, and
Dynamo as distinct layers. Use one OpenAI-compatible client against multiple servers;
cover containers, health/auth/rate limits, observability, backpressure, cancellation,
revision pinning, privacy, and a use-case selection matrix.

## Visual standard and figure manifest

The visual bar is Chris Olah's explanatory clarity and Sebastian Raschka's architecture
gallery, using the repository's approved adaptation in
`docs/convnet-rewrite/figure-style.md`. All conceptual diagrams are pre-generated,
byte-idempotent committed SVGs; notebook code computes or demonstrates results and does
not contain illustration-only plotting plumbing.

Rules:

- One coherent figure style across the chapter.
- One idea and one visual hierarchy per diagram.
- Prefer mechanism, flow, spatial relationship, and meaningful annotation over decorated
  boxes.
- No gratuitous gradients, fake 3-D, clip-art, logo clouds, tiny labels, diagonal skip
  spaghetti, text-line collisions, dead whitespace, redundant titles, or screenshots of
  transient product UIs.
- Use one accent color, with a second only for an actual comparison.
- Use gallery-style vertical spines, pills, nested containers, rectilinear connections,
  restrained callouts, and explicit input/output anchors for architecture/dataflow.
- Keep labels at least 9 pt equivalent, normally 11--16 pt; captions carry titles.
- Render SVGs to PNG and inspect them after every iteration. Build contact sheets for the
  full chapter and verify byte idempotency by rendering twice.

Planned diagrams:

1. Jupyter/VS Code: notebook-document versus live-kernel state; local and remote editor /
   server / kernel / accelerator layers.
2. Hosted notebooks: persistent artifacts versus ephemeral runtime; D2L page -> generated
   branch -> Colab/Kaggle launch pipeline.
3. Cloud: provider spectrum and trust boundary; disposable compute plus persistent
   storage lifecycle; complete cost stack.
4. Hardware: training/inference decision map; model-memory fit; discrete VRAM versus
   unified-memory data path; one-GPU workstation anatomy; buy/rent break-even.
5. Ecosystem: complete model artifact anatomy; revision-pinned artifact lifecycle;
   adaptation and format/conversion relationships.
6. Training: parallelism ladder; training-memory budget; data/compute/communication
   pipeline; checkpoint and preemption recovery.
7. Serving: local-to-distributed serving stack; prefill/decode regimes; continuous
   batching and paged KV; prefix reuse; common client API over multiple engines.

Use `tools/arch_diagrams.py` and its pinned gallery grammar for architecture/dataflow,
extending shared primitives rather than styling individual figures. Use the same palette,
fonts, stroke weights, and save path for non-network system diagrams so the chapter does
not mix visual families. Data plots may use the mathematical figure infrastructure but
must match the chapter palette and typography.

## Implementation sequence and review gates

1. Record this plan before implementation.
2. Implement and test hosted-notebook classification, normalization, manifest generation,
   and dry-run branch staging.
3. Implement the dynamic site controls and unit/browser-test their URL/state logic.
4. Restructure chapter navigation and numbering; delete SageMaker from the sequence.
5. Produce two representative pilot diagrams (a system/dataflow diagram and a hardware
   comparison), render and inspect them against the approved visual grammar, then use
   their shared primitives for the remaining figures.
6. Rewrite the seven source `.md` notebooks in order, adding compact executable examples,
   high-quality diagrams, summaries, and exercises.
7. Generate PyTorch/JAX/NumPy notebook variants and test representative Colab/Kaggle
   launches. Never edit generated `.qmd` files directly.
8. Run source lint, figure lint, notebook generation/validation, hosted-manifest checks,
   focused HTML rendering, link/accessibility checks, and visual inspection. A full book
   rebuild is deferred unless explicitly requested.

## Acceptance criteria

- The seven notebooks appear in the agreed order and SageMaker is absent.
- PyTorch, JAX, and framework-independent NumPy notebook variants are classified and
  published deterministically; TF/MXNet are not offered as hosted variants.
- Every eligible page has correct Colab, Kaggle (if validated), and download actions that
  follow framework selection.
- Hosted notebooks contain setup, code, outputs, images/assets, and revision metadata
  needed to run independently of the local build tree.
- No conceptual figure is generated by notebook-only plotting code.
- Every planned structural concept has a legible, inspected, consistent diagram; no
  diagram violates the repository figure checklist.
- Source `.md` remains authoritative; generated `.qmd`/`.ipynb` artifacts do not pollute
  the authoring branch.
- Focused validation passes without rebuilding or uploading the complete book unless
  separately requested.
