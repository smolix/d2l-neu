# Further Reading
:label:`sec_lm_encoders_fr`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "FR Further Reading: Encoders and Retrieval · T0").
Blueprint: none (the plan adds this section).
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** The retrieval methods that fail the teachable-core test, each with the clause saying what it adds to the chapter's mechanisms.
**Opening:** none (reading).
**Teaches:**
- Learned sparse retrieval (SPLADE): term expansion in vocabulary space with a sparsity penalty, so an inverted index serves a learned model.
- Late interaction (ColBERT, ColBERTv2): per-token vectors matched at query time with a max-similarity sum; residual compression makes the index affordable.
- Late chunking and contextual headers: embedding the whole document before pooling chunks, or prepending a model-written context sentence per chunk, so a chunk carries what its neighbours say.
- Training for RAG: joint retriever and generator training (REALM, RAG), fine-tuning the generator on retrieved context with distractors (RAFT), learning when to retrieve and to critique (Self-RAG), what each one learns and costs.
- Long context against retrieval: when whole documents in the context win, and what the position effects of 2.5's exercise say about the comparison.
- Theoretical limits of single-vector retrieval: the number of top-k sets a fixed dimension can represent.
**Derivation:** none.
**Notebook:** none (reading); 1,000 to 2,000 words, no training run, at most a few cells that load an existing index.
**Figures:** none.
**Restate / forward:** none.
**Exercises:** none.
**Writer notes:** Each citation carries a clause saying what it adds; no survey tails elsewhere in the chapter. The notebook is placed last and nothing depends on it. Rows from these methods may appear on 2.4's trade-off plot as the open exercise, not here. Keep the long-context paragraph consistent with 2.5's exercise and with 4.5's long-context training. Do not list products or vector databases.
**Sources:**
- Papers: [SPLADE](https://arxiv.org/abs/2107.05720) (learned sparse expansion); [ColBERT](https://arxiv.org/abs/2004.12832) and [ColBERTv2](https://arxiv.org/abs/2112.01488) (late interaction and residual compression); [Late Chunking](https://arxiv.org/abs/2409.04701); [REALM](https://arxiv.org/abs/2002.08909) (retrieval as a latent variable in pretraining); [RAFT](https://arxiv.org/abs/2403.10131) (fine-tuning with distractors); [Self-RAG](https://arxiv.org/abs/2310.11511) (reflection tokens for when to retrieve); [Long-Context LLMs Meet RAG](https://arxiv.org/abs/2410.05983) (when long context helps and hurts); [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038) (dimension bounds on representable top-k sets; cite only what the abstract-level claim supports).
- Code and documentation: the [ColBERT repository](https://github.com/stanford-futuredata/ColBERT); [Pyserini](https://github.com/castorini/pyserini) (SPLADE and ColBERT baselines).
- Courses, tutorials and blogs: Anthropic, [Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval) (contextual headers and their reported gains, quoted with their datasets); Hamel Husain, [Stop Saying RAG Is Dead](https://hamel.dev/notes/llm/rag/not_dead.html).
-->
