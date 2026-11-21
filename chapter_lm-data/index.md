# Data for Language Models
:label:`chap_lm_data`

<!-- For maintainers. Plan: docs/llm/language_models_part_plan.md, "Chapter 3
Data for Language Models"; blueprint docs/llm/blueprints/ch03_data.md. The
plan's Further Reading entry for this chapter is "none", so the chapter has
no further-reading notebook. Rights: Common Crawl text is never committed or
hosted, only pins, manifests and hashes. -->

By now you can download a pretrained model and measure what it does
(:numref:`chap_lm_pretrained`). The text it learned from was prepared by
someone else and, for many released models, never published; yet preparing
it is a large part of the work of training a language model. It starts from
billions of web pages, and every decision made before the first batch
changes what the model can ever learn: what to extract from a page, what to
throw away, what to read twice, and how to cut the rest into tokens.

For a long time such decisions were made by rules of thumb and seldom
published. The open corpora of the last few years publish the pipeline
together with the experiments behind it, so that others can check a choice
about data. This chapter builds one that a single machine can hold, from a
record of a web crawl to a token in a file on disk. It measures what each
step changes and writes every decision into a manifest, so that the corpus
can be rebuilt from hashes, audited, and corrected when a page has to be
removed.

The chapter moves in three steps. The first decides which documents exist:
it extracts text from pages (:numref:`sec_lm_crawl`), filters the text and
audits whose pages the filters remove (:numref:`sec_lm_quality_filtering`),
and takes out near-copies and leaked test items (:numref:`sec_lm_dedup`).
These steps are measured on the documents of a quarter of one Common Crawl
archive file, each set beside the text that FineWeb, a released corpus, kept
from the same record. The second step decides how often a model reads each
kind of text, and there the measure is a model: small proxy runs choose the
domain weights (:numref:`sec_lm_data_mixtures`) and test whether text
rewritten by a model can stretch a source that has run short
(:numref:`sec_lm_synthetic_data`). The third cuts the text into tokens and
writes the files on which :numref:`chap_lm_pretraining` trains the book's
reference model (:numref:`sec_lm_tokenizers`). The second and third steps start from
FineWeb-Edu as released, which holds far more text.

Streaming the token files into training batches belongs to the next chapter,
and the data that teach a model to follow instructions and preferences
belong to :numref:`chap_lm_finetuning` and :numref:`chap_lm_rl`. We assume
you know byte-level BPE (:numref:`sec_text-sequence`), why bits per byte can
compare models with different tokenizers (:numref:`subsec_lm_bpb`), how a
paired comparison puts an error bar on a difference
(:numref:`sec_lm_eval_suites`), and what a text embedding is
(:numref:`sec_lm_text_embeddings`).

The long computations, labelling pages with a model of 27 billion parameters
and training dozens of small language models, ran once on four 24 GB GPUs,
and their results ship with the book. Each notebook downloads them, checks
them against its configuration and repeats a small part; without a GPU it
skips what needs one and says so. Common Crawl grants no licence for the
pages it archives, so the book publishes only byte ranges and hashes, from
which the notebooks rebuild the documents. No notebook calls a paid service.

```toc
:maxdepth: 2

crawl
quality-filtering
dedup
data-mixtures
synthetic-data
tokenizers
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter, and each names
the section it goes with. All are freely accessible online. If you read only
three, read the FineWeb blog post, the data chapters of the Smol Training
Playbook and the pipeline guide of the `tokenizers` library: between them
they cover how a filter is tested by training on its output, how a mixture
is planned, and how a tokenizer is assembled. The chapter has no notebook of
further reading.

**Courses**

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) has lectures on data sources, filtering, deduplication, mixing and tokenization, and an assignment that turns Common Crawl into filtered, deduplicated training data, the pipeline of :numref:`sec_lm_crawl`, :numref:`sec_lm_quality_filtering` and :numref:`sec_lm_dedup`.
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) and [CMU 11-711: Advanced Natural Language Processing](https://cmu-l3.github.io/anlp-spring2026) each give lectures to pretraining and its data, a shorter route through the questions of this chapter.
- Chapter 6 of the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter6/1) trains tokenizers with the `tokenizers` library. Read it before :numref:`sec_lm_tokenizers` if the library is new to you.

**Building a web corpus**

- The [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) describes how FineWeb chose its extractor, filters and deduplication by training small models on the output of each candidate, the method behind :numref:`sec_lm_crawl` to :numref:`sec_lm_dedup` and the proxy runs of :numref:`sec_lm_data_mixtures`. Take the method; its numbers belong to its scale.
- Common Crawl's [getting-started page](https://commoncrawl.org/get-started) explains how its archives are organized and fetched, and the [trafilatura documentation](https://trafilatura.readthedocs.io/) lists the extraction options that :numref:`sec_lm_crawl` compares with FineWeb's settings.
- [Datatrove](https://github.com/huggingface/datatrove) is the library that built FineWeb: readers, extractors, the heuristic filters that :numref:`sec_lm_quality_filtering` checks its own rules against, and the MinHash stages of :numref:`sec_lm_dedup`.
- The [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) gives the rubric, the classifier and the share of documents kept, the recipe that :numref:`sec_lm_quality_filtering` rebuilds with a newer labeller. [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) is its multilingual counterpart, whose slices :numref:`sec_lm_quality_filtering` and :numref:`sec_lm_tokenizers` use.
- [datasketch](https://github.com/ekzhu/datasketch) implements MinHash and banding and serves :numref:`sec_lm_dedup` as an independent check, and [deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) is the suffix-array deduplicator whose method that section writes out on a small sample.
- The [dataset cards](https://huggingface.co/docs/hub/datasets-cards) of the Hugging Face Hub and [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) set the standard for documenting a corpus that the manifest and card of :numref:`sec_lm_crawl` follow.

**Mixtures and generated text**

- The [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) plans the data of a small model from start to finish, with chapters on mixtures, annealing and synthetic data that accompany :numref:`sec_lm_data_mixtures` and :numref:`sec_lm_synthetic_data`.
- The reference implementations of [RegMix](https://github.com/sail-sg/regmix) and [DoReMi](https://github.com/sangmichaelxie/doremi) are the full-scale versions of the two methods that :numref:`sec_lm_data_mixtures` builds in miniature, and the [Dolma 3 pool card](https://huggingface.co/datasets/allenai/dolma3_pool) lists the sources behind the released mixture that the section reads as a case study.
- The cards of [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) and [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) describe a rephrased crawl and a corpus generated from seeds, with their generators, prompts and licences, the two kinds of synthetic data of :numref:`sec_lm_synthetic_data`.

**Tokenizers**

- The [pipeline guide](https://huggingface.co/docs/tokenizers/pipeline) of the `tokenizers` library explains the normalizer, pre-tokenizer, model and decoder from which :numref:`sec_lm_tokenizers` assembles the reference tokenizer.
- [SentencePiece](https://github.com/google/sentencepiece) trains the unigram and BPE models that :numref:`sec_lm_tokenizers` compares, and its documentation states the normalization defaults that the section turns off.
- [minbpe](https://github.com/karpathy/minbpe) writes byte-level BPE in a few hundred lines, a second implementation to read beside the tokenizer of :numref:`sec_text-sequence`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §20]{.kicker}

Data for Language Models<br>
**extraction · quality filters · deduplication · mixtures · synthetic data · tokenizers**
:::
:::

::: {.slide title="Of 9,731 Crawled Pages, About One in Ten Reaches the Corpus"}
[The problem]{.kicker}

A quarter of one Common Crawl archive file, through the first three sections:

| stage | documents left | section |
|---|---|---|
| HTML responses with extracted text | 9,286 | Document Extraction |
| English, language score at least 0.65 | 3,288 | Document Extraction |
| after the heuristic rules | 1,032 | Quality Filtering |
| after exact and near-duplicate removal | 1,014 | Deduplication |

::: {.d2l-note}
Every row is a decision made before the first batch. A manifest records
each one, and the section that makes it measures what it removes.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- extract text from web archives and measure what an extractor loses;
- filter by rules and by a classifier on model labels, and audit whose
  pages a filter removes;
- find near-duplicates without comparing every pair, and keep test items
  out of training data;
- choose domain weights with small proxy runs and a noise floor;
- check text written by a model, and compare it with real text at equal
  cost;
- choose a vocabulary, and write verified token files for pretraining.

**Running example:** 256 MiB of one Common Crawl file beside FineWeb's text
for the same records; then FineWeb-Edu as released.

**Idea:** record every decision in a manifest and measure its effect: on
the documents for extraction, filters and deduplication; on small trained
models for mixtures and rewritten text.
:::

::: {.slide title="Three Steps: Choose Documents, Weigh Them, Cut Them into Tokens"}
[Chapter map]{.kicker}

| step | section | question |
|---|---|---|
| choose documents | Document Extraction from Web Crawls | what does extraction lose, and on which pages? |
| | Quality Filtering | what does a filter remove, and whose pages? |
| | Deduplication and Decontamination | how are near-copies found without comparing every pair? |
| weigh them | Data Mixtures | can small runs choose the domain weights of a large one? |
| | Synthetic and Rephrased Data | when does text written by a model help? |
| cut into tokens | Tokenizers at Scale | what does one vocabulary cost each language? |

- **Uses:** byte-level BPE; bits per byte; paired error bars; text embeddings.
- **Next:** the reference model trains on this chapter's token files.
:::
