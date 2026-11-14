# Encoders, Embeddings and Retrieval
:label:`chap_lm_encoders`

The reader has a decoder that answers questions but no way to find, among a million passages, the one that holds the answer, and no cheap way to label ten thousand sentences an hour.

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "Chapter 2 Encoders, Embeddings and Retrieval").
Blueprint: docs/llm/blueprints/ch02_encoders.md.
The plan is a specification, not book text; this comment is replaced
when the chapter is written.

**Label:** `chap_lm_encoders`. Five sections plus one Further Reading notebook, T1 throughout (2.1 also runs on CPU). The chapter covers the pretrained encoders and embedding models that still do the work in search, filtering, tagging and retrieval, and the retrieval systems built on them; it replaces the two retiring NLP chapters.
**Hook:** The reader has a decoder that answers questions but no way to find, among a million passages, the one that holds the answer, and no cheap way to label ten thousand sentences an hour.
**Route:** What statistic does a word vector factorize (2.1)? What does a masked-token encoder give per token, and when does fine-tuning it beat prompting a decoder (2.2)? Why do pooled encoder states retrieve poorly until contrastive training spreads them on the sphere (2.3)? What does an index trade for speed, and what floor must a dense index beat (2.4)? Where does a retrieval-augmented system fail, and how is the failing stage found (2.5)?
**Out of scope:** Encoder pretraining at scale (`sec_transformer` builds and trains the bidirectional encoder; 2.2 starts from a released checkpoint), agent loops that call retrieval as a tool (`chap_lm_agents`), multimodal retrieval (the Image Models part), learned sparse retrieval and late interaction (the Further Reading notebook).
**Writer notes:**
- Running models (Table 5): ModernBERT-base (encoder), Qwen3-Embedding-0.6B (embedder), Qwen3-Reranker-0.6B (reranker), Qwen3-1.7B (few-shot comparison in 2.2), Qwen3-4B (generator in 2.5), Qwen3.5-4B in bf16 (judge; not the blueprint's Qwen3.5-9B). Re-check the Hub for a 2026 embedding or reranking successor of Qwen3-Embedding before authoring 2.3; `sota.md` found none in budget.
- The one shared artifact this chapter owns is the book corpus (2.4): `d2l.book_corpus(chunking='sections')`, `d2l.BookRetriever`, and the question set `d2l.book_qa(split)` (2.5). Chapters 4, 7, 10, 12 and 13 consume them. Their specifications (cleaning rules, chunk ids, document splits, item schema, leakage control) are in blueprint ch02 2.4 (the corpus specification subsection) and 2.5 (the question-set subsection); keep them as written, with the name changes below. The corpus excludes the whole Language Models part so the notebooks' own examples cannot be retrieved.
- Exports and names (`01_interfaces.md`): `d2l.embed_texts` (2.3; 3.2 keeps its own `d2l.embed_docs` on purpose), `d2l.retrieval_metrics` (2.3), `d2l.Hit` (a NamedTuple, not a dict), `d2l.book_qa` items have no `gold_ids` field (derive gold chunks with `d2l.gold_chunk_ids(item, corpus)`), `d2l.rag_judge(judge, tokenizer, item, result, context_texts)` is the single route by which other chapters score answers to book questions. 2.1 keeps the legacy labels `subsec_negative-sampling` and `sec_word2vec_data`, cited by `sec_mdl-mi-estimation`.
- Book-side moves: `d2l.DATA_HUB['ptb']` is registered only in the retiring `word-embedding-dataset.md` and is needed by `d2l.PennTreebank` in `chapter_recurrent-modern/lstm.md`; move the registration there. The blueprint's claim that `sec_scaling-laws` imports `read_ptb` is wrong (it trains on `d2l.WikiText103`); 2.1 is free to drop the word2vec data helpers. `why-conv.md:274` cites the retiring sentiment CNN; 2.2 carries the one-sentence passage it can point to. `cloud-instances.md:212` cites `sec_bert-pretraining`; re-point that exercise to 2.2's fine-tuning run. Missed by the November 5-6 scan (found 2026-11-14): `legacy-attention-lib.md` (in `tools/build_lib.py`) lives in the retiring pretraining directory and defines `d2l.GRU` and the TensorFlow and MXNet copies of `masked_softmax`, `DotProductAttention` and `MultiHeadAttention`, so it moves to a surviving directory before the deletion; `discrete-diffusion.md` cites `sec_bert`; `d2l.TokenEmbedding` must be defined in 2.1. All are rows of "Changes outside this part".
- Assignment A2 (instructor-facing): a RAG system over the book with Recall@10 and answer accuracy above committed floors with paired-bootstrap intervals; the private grading split is regenerated per term and never shipped (blueprint ch02 "Assignment A2"). Judge at Qwen3.5-4B.
- Compute: every section is T1 within a 15 GiB slot; 2.5 loads the generator (Qwen3-4B, about 8 GB) and frees it before loading the judge (Qwen3.5-4B, about 8 GB). The 32K long-context arm of the blueprint is an exercise (Qwen3-4B's full-attention KV cache is about 147 KB per token, so 32K tokens is about 4.7 GB on top of the weights).
- Consistency points: every stage comparison uses `d2l.paired_bootstrap` with `cluster = doc_id`; tuning uses the `train` questions, analysis the `dev` split, and `test` is evaluated once; no section re-proves the InfoNCE ceiling (`sec_mdl-infonce`) or the shifted-PMI result (`sec_mdl-mi-estimation`).
**Teaching sources:**
- [CS224N](https://web.stanford.edu/class/cs224n/): the word-vector and pretraining lectures and problem sets (exercise seeds for 2.1 and 2.2); the agents, tool use and RAG lecture (2.5).
- [CMU 11-711](https://cmu-l3.github.io/anlp-spring2026): the retrieval and RAG lecture and assignment A2 (an end-to-end NLP system), the course-side counterpart of this chapter's A2.
- [Berkeley CS288](https://cal-cs288.github.io/sp26): text classification and the RAG lecture; its experimental-design lecture frames 2.5's stage-by-stage measurement.
- Jay Alammar, [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) and [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/): figure ideas for 2.1 (window and negatives) and 2.2 (input layout); redraw in the book's style, do not copy.
- [Sentence Transformers documentation](https://sbert.net/) and its [training overview](https://sbert.net/docs/sentence_transformer/training_overview.html): the production reference for 2.3's losses and quantization.
- [ModernBERT blog](https://huggingface.co/blog/modernbert): the design list for 2.2 (alternating local and global attention, unpadding, long context) and fine-tuning recipes.
- [Faiss wiki](https://github.com/facebookresearch/faiss/wiki): index factory strings and parameter guidance for 2.4; [Pyserini](https://github.com/castorini/pyserini) and [BEIR](https://github.com/beir-cellar/beir): sparse baselines and heterogeneous retrieval evaluation.
- Jason Liu, [Systematically Improving Your RAG](https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/) and Eugene Yan, [Patterns for Building LLM-based Systems](https://eugeneyan.com/writing/llm-patterns/): the retrieval-versus-generation decomposition that 2.5 measures; Hamel Husain, [Stop Saying RAG Is Dead](https://hamel.dev/notes/llm/rag/not_dead.html): the long-context-versus-retrieval exercise.
- Hugging Face cookbook, [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag): a runnable pipeline to compare against in 2.5.
**Further Reading:** `sec_lm_encoders_fr` (learned sparse retrieval, late interaction, late chunking and contextual headers, training for RAG, long context against retrieval).
-->

```toc
:maxdepth: 2

word-embeddings
bert
text-embeddings
retrieval
rag
further-reading
```
