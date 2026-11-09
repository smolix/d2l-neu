# Data for Language Models
:label:`chap_lm_data`

The reader has trained on a text file someone else prepared. Released models start from billions of pages, and every choice made before the first batch (what to extract, what to drop, what to repeat, how to cut it into tokens) changes what the model can ever learn.

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "Chapter 3 Data for Language Models").
Blueprint: docs/llm/blueprints/ch03_data.md.
The plan is a specification, not book text; this comment is replaced
when the chapter is written.

**Label:** `chap_lm_data`. Six sections, T0 to T2. The chapter follows one document from a crawl record to a token in a shard on disk: extraction, filtering, deduplication, mixing, synthesis and tokenization, each step written to one versioned manifest, so the corpus Chapter 4 trains on can be rebuilt from hashes.
**Hook:** The reader has trained on a text file someone else prepared. Released models start from billions of pages, and every choice made before the first batch (what to extract, what to drop, what to repeat, how to cut it into tokens) changes what the model can ever learn.
**Route:** What does extraction lose, and on which kinds of pages (3.1)? What does a quality filter remove, and who checks what it removed (3.2)? How are near-duplicates found without comparing every pair, and how are test items kept out (3.3)? How much of each domain goes in, and can small runs decide it (3.4)? When does text rewritten by a model help, and what does it cost in diversity (3.5)? How large should a vocabulary be, and what does one tokenizer cost each language (3.6)?
**Out of scope:** Byte-pair encoding itself (`sec_text-sequence`), n-gram statistics and Zipf's law (`sec_language-model`), distribution shift in general (`sec_environment-and-distribution-shift`), the data path from shards into packed batches (4.1), instruction and preference data (Chapter 7 and 8), and the 2026 governance of crawling (one paragraph in 3.1 and the Tools appendix).
**Writer notes:**
- Two data lines are kept apart on purpose. The teaching pipeline runs one pinned Common Crawl byte range through 3.1, 3.2 and 3.3 (small, inspectable, with FineWeb counterparts for every record); the reference model's data is FineWeb-Edu as released, tokenized with the reference tokenizer and written as shards by 3.6, which 4.1 streams and 4.2 trains on. The document buckets that split FineWeb-Edu (bucket 0 validation, 1 to 7 side pool for 3.4 and 3.5, 8 side validation, 9 to 255 reference training; `sha256(id)` mod 256) are fixed here and used by 3.4, 3.5, 3.6, 4.1 and 4.2 (blueprint ch03 "The corpus series").
- Rights rule (blueprint README Decision 13): Common Crawl text is never committed or hosted; 3.1 ships a pin (URL, byte range, SHA-256 per 16 MiB block) and manifests with hashes, and the notebooks rebuild the text; a takedown is honoured by deleting one manifest row. ODC-By data (FineWeb, FineWeb-Edu, FineWeb-2, FineMath, the FineWeb-Edu annotations) ships with attribution; Nemotron-CC-v2 is under the NVIDIA Data Agreement and is cited, not shipped; MIT and CC BY-SA benchmark items of 3.3 ship in a separate artifact with its licence stated.
- Shared artifacts and exports: `d2l.CorpusManifest` and `d2l.Doc` (3.1), `d2l.heuristic_filters` and `d2l.embed_docs` (3.2; a different name from 2.3's `d2l.embed_texts` on purpose), `d2l.near_dedup` (3.3; 7.3 imports it), `d2l.train_proxy_lm`, `d2l.ProxyCfg`, `d2l.bpb_from_nll` (3.4; 3.5 and 3.6 import them; 1.4 owns `d2l.bits_per_byte`), `d2l.r2_tokenizer`, `d2l.write_shards`, `d2l.TokenShards` (3.6; Chapter 4 reads them). The reference tokenizer (byte-level BPE, vocabulary 32,768, ids 0 to 31 reserved; blueprint README Decision 1) and the shard format are one-way doors.
- Running models: Qwen3.5-4B in bf16 as the labeller and judge, Qwen3.5-2B as the rephraser (3.5), Qwen3-Embedding-0.6B as the frozen embedder (3.2), a small released decoder for the perplexity filter, `d2l.GPT` as the proxy model (3.4, 3.5, 3.6). Current corpora named as exemplars (Table 8): FineWeb-2 (3.1, 3.6), Nemotron-CC-v2 and FinePhrase (3.5), Dolma 3 (3.4).
- Assignment A3 (instructor-facing): a data pipeline with a checkable manifest; deduplication F1 on labelled pairs within a margin of the staff solution; a manifest hash reproducible from the rebuilt documents; the MinHash specification is exact so signatures compare bit for bit (blueprint ch03 "Assignment A3"). T0 for everything except the embedding classifier, which has a T0 path.
- Compute: 3.1 and 3.3 are T0; 3.2, 3.5 and 3.6's experiment are T1; 3.4 is T2 with a T1 configuration of the same code and committed T2 logs. Authoring-time generation (the rephrased corpus, the paraphrased plants, the labels) is done once and committed.
- Consistency points: every proxy-run comparison is seed-paired with a noise floor from repeated seeds and a difference is called real only if every paired difference has the same sign and the mean exceeds two floor standard deviations; every per-domain loss is reported in bits per byte of the NFC text, never per-token; NFC is the one normalization, applied in 3.1 and before tokenization in 3.6, and NFKC is named only as the thing not to do.
**Teaching sources:**
- [CS336](https://cs336.stanford.edu/): the data lectures and assignment A4 (Common Crawl to filtered, deduplicated pretraining data), whose graded quantities shape A3.
- [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1): the ablation discipline (small models, equal tokens, per-dump against global deduplication); take the method, not the numbers.
- [FineWeb2](https://arxiv.org/abs/2506.20920) and its [dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2): the multilingual pipeline whose per-language slices feed 3.2's audit and 3.6's experiment.
- [Datatrove](https://github.com/huggingface/datatrove): the reference pipeline (readers, filters, MinHash stages); the production oracle for 3.1 to 3.3.
- [minbpe](https://github.com/karpathy/minbpe): minimal BPE, the from-scratch oracle beside `d2l.BPETokenizer` in 3.6; [Hugging Face LLM Course, chapter 6](https://huggingface.co/learn/llm-course/en/chapter6/1): the tokenizer chapter for readers new to the `tokenizers` library.
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook): the data chapters on mixtures, annealing and synthetic data (3.4, 3.5).
- [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) and [Datasheets for Datasets](https://arxiv.org/abs/1803.09010): the documentation standard the manifest's card follows (3.1).
- [CMU 11-711](https://cmu-l3.github.io/anlp-spring2026) and [CS224N](https://web.stanford.edu/class/cs224n/): the pretraining-data lectures (scaling, systems, data) that assign this chapter as reading.
**Further Reading:** none.
-->

```toc
:maxdepth: 2

crawl
quality-filtering
dedup
data-mixtures
synthetic-data
tokenizers
```
