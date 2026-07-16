# Ecosystem
:label:`sec_software_ecosystem`

Almost nothing in modern machine learning starts from an empty directory.
You begin from someone else's work: a pretrained checkpoint, a curated
dataset, a paper with released code, a benchmark that tells you what "good"
currently means. Knowing *where to look* — and how to judge what you find —
has quietly become a core skill, as load-bearing as knowing how to write a
training loop. This section is the map we wish someone had handed us: the
places that matter in 2026, the leaderboards worth trusting, how to stay
current in a field that turns over monthly, and the handful of habits that
keep borrowed artifacts from becoming liabilities.

## Where to Find Things

### Models

The [Hugging Face Hub](https://huggingface.co/models) is the default: over
two million public models as of spring 2026, with Git-style versioning,
model cards, and integration into effectively every library in this book.
Its scale cuts both ways — the top 200 repositories account for about half
of all downloads, and most of the rest is abandoned experiments — so search
by task, sort by downloads and recency, and read the model card before
committing bandwidth. Complements worth knowing:

* [ModelScope](https://modelscope.cn/), Alibaba's hub, is the primary home
  of the Chinese open-model ecosystem (Qwen and friends, 170,000+ models) —
  increasingly where open-weight state of the art appears first.
* [Kaggle Models](https://www.kaggle.com/models) hosts curated weights
  wired into Kaggle's competition and notebook infrastructure
  (:numref:`sec_hosted_notebooks`).
* The [Ollama library](https://ollama.com/library) is a short, curated
  menu of local-runtime models — less a discovery surface than a
  convenience layer (:numref:`sec_model_serving`).
* [`timm`](https://huggingface.co/timm) remains the reference collection
  of vision backbones; [Civitai](https://civitai.com/) dominates community
  image-generation checkpoints and LoRAs.

### Datasets

[Hugging Face Datasets](https://huggingface.co/datasets) (half a million
public datasets) and [Kaggle](https://www.kaggle.com/datasets) cover most
supervised needs. For pretraining-scale text, the lineage runs from raw
[Common Crawl](https://commoncrawl.org/) through filtered derivatives —
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (~15
trillion tokens, plus a 1,000-language successor) has become the standard
open baseline, the role C4 played in this book's era of BERT. For
vision–language pairs, DataComp superseded LAION as the recommended
starting point. Older and very large corpora sometimes live only on
[Academic Torrents](https://academictorrents.com/). Whatever the source:
datasets have versions, licenses, and documented failure modes exactly as
models do, and the data card deserves the same read as the model card.

### Papers and Code

New work appears on [arXiv](https://arxiv.org/list/cs.LG/recent) first;
the community's curated front page for ML is [Hugging Face
Papers](https://huggingface.co/papers), which absorbed that role when
Papers with Code — for years the standard index from papers to code and
benchmarks — was shut down without notice in mid-2025. (Its historical
leaderboard data survives only as a frozen archive; let that be a lesson
about free infrastructure.) [Semantic Scholar](https://www.semanticscholar.org/)
and [alphaXiv](https://alphaxiv.org/) help with search and discussion, and
GitHub remains where the code actually is — a repository's issue tracker
and commit recency tell you more about whether a method reproduces than
the paper's abstract does.

## Choosing a Model: Benchmarks and Leaderboards

"Which model should I use?" changed answers a dozen times while this book
was written; what stays stable is *how* to answer it. No single number
survived contact with optimization — static benchmark suites become
training data, and the once-canonical Open LLM Leaderboard was retired in
2025 for exactly that reason. Practitioners now triangulate:

* [LMArena](https://lmarena.ai/) (now Arena) — blind human pairwise
  preference; hard to game, but measures "pleasing an average user," which
  may not be your task.
* [LiveBench](https://livebench.ai/) — contamination-resistant by rotating
  fresh questions monthly.
* [SWE-bench Verified](https://www.swebench.com/) — the reference for
  agentic coding against real GitHub issues; representative of the
  benchmark-per-capability pattern (math, long context, safety all have
  their own).
* [Artificial Analysis](https://artificialanalysis.ai/) — the
  quality/price/latency triangulation across hundreds of models and
  providers; usually the first stop when cost matters.
* [OpenRouter rankings](https://openrouter.ai/rankings) — revealed
  preference by real token volume rather than scores; instructive
  precisely where it disagrees with the quality leaderboards.

Then the step that outranks all of the above: build a small evaluation set
from *your* task — even fifty examples — and run the shortlist on it. The
gap between leaderboard rank and performance on your distribution is
routinely larger than the gap between adjacent leaderboard entries. The
evaluation discipline this book has practiced throughout — held-out data,
honest baselines, error analysis — applies to *choosing* models exactly
as it does to training them.

## Staying Current

A field this fast rewards a deliberate information diet over doomscrolling.
A workable minimal set, as of 2026:

* [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — the town square
  of open-weight ML; new releases, quantizations, and hardware reports
  surface here first, usually with reproduction attempts attached.
* [Hugging Face Papers](https://huggingface.co/papers) daily — a curated
  dozen papers instead of arXiv's daily hundreds.
* One good newsletter — Andrew Ng's *The Batch*, Jack Clark's *Import AI*,
  or Sebastian Raschka's *Ahead of AI* — for the weekly synthesis.
* For systems depth, the [GPU MODE](https://github.com/gpu-mode/lectures)
  lecture series and community, and the
  [EleutherAI Discord](https://www.eleuther.ai/community) for open
  research.

X/Twitter remains where labs announce and researchers argue; treat it as a
discovery feed, not an archive. And the chapter-end
resources of this book (:numref:`chap_appendix_tools`) collect the
durable long-form references.

## Using What You Found

### Pin the Identity

![A reusable model artifact combines configuration, weights, preprocessing, documentation, and an immutable revision.](../img/tools-ecosystem-artifact.svg)
:label:`fig_tools_model_artifact`

A model is more than its weights: the tokenizer, preprocessing,
configuration, license, and revision all determine whether you can
reproduce a result (:numref:`fig_tools_model_artifact`). Repositories are
Git repositories — `main` moves. Pin the commit:

```text
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="organization/model-name",
    revision="0123456789abcdef",          # an immutable commit, not main
    allow_patterns=["*.json", "*.safetensors"],
)
```

A tiny manifest turns "which model was that?" from archaeology into a
lookup — in production you would add file hashes, library versions, and an
evaluation record:

```{.python .input #software-ecosystem-manifest}
from dataclasses import asdict, dataclass
import json

@dataclass(frozen=True)
class Artifact:
    repository: str
    revision: str
    task: str
    license: str
    parent: str | None = None

artifact = Artifact(repository="organization/model-name",
                    revision="0123456789abcdef",
                    task="text-generation", license="apache-2.0")
print(json.dumps(asdict(artifact), indent=2))
```

The same pinning discipline applies to *derived* artifacts: a LoRA adapter
without its base-model revision is incomplete, and a quantized conversion
(GGUF, AWQ, ONNX — see :numref:`sec_model_serving`) is a new artifact
whose numerical fidelity someone should have checked against the source.

### Trust and Licenses

Downloaded models are software supply-chain inputs, and the ecosystem has
real teeth:

* Prefer **safetensors** — a pure tensor container. Legacy pickle-based
  checkpoints can execute arbitrary code on load, and the Hub's scanners
  are a mitigation, not a guarantee.
* `trust_remote_code=True` runs Python from the repository on your
  machine. Read and pin that code; use the flag only when the architecture
  genuinely requires it.
* "Open" spans a wide range of licenses: permissive Apache/MIT weights,
  acceptable-use licenses with commercial thresholds, research-only
  releases, and gated models whose terms you accept per account. The
  license lives in the repository — record the one you actually accepted,
  since repositories can relicense between revisions.
* Every hub client caches aggressively (tens to hundreds of gigabytes in
  `~/.cache/huggingface` is routine). Learn your cache tool's `scan` and
  `delete` commands before your disk teaches you.

## Summary

* Discovery is a skill: Hugging Face is the center for models and
  datasets, with ModelScope, Kaggle, Ollama's library, and Civitai as the
  complements that matter; FineWeb-class corpora are the open pretraining
  baseline.
* Papers with Code is gone; arXiv plus Hugging Face Papers is the 2026
  reading pipeline, with GitHub as the ground truth for whether code
  exists and is alive.
* No leaderboard is trusted alone: triangulate Arena, LiveBench,
  task-specific benchmarks, and price/latency data — then decide on a
  small evaluation you built from your own task.
* Keep a deliberate information diet: one community, one curated paper
  feed, one newsletter.
* Pin revisions, prefer safetensors, read licenses, treat remote code as
  code review, and manage your caches — borrowed artifacts are supply
  chain, not just downloads.

## Exercises

1. Pick a task you care about and shortlist three models using at least
   two leaderboards plus Artificial Analysis. Where do the rankings
   disagree, and why might that be?
1. Inspect a model repository of your choice and list every file needed
   for offline inference. Which of them could execute code on your
   machine?
1. Build a 25-example evaluation set for a task you know well and run
   your shortlist from Exercise 1 on it. Does your ranking match the
   leaderboards'?
1. Find the license of a popular open-weight model and determine: may you
   deploy it commercially, fine-tune it, and redistribute the fine-tune?
