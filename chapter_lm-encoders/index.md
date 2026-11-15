# Encoders, Embeddings and Retrieval
:label:`chap_lm_encoders`

The decoders of :numref:`chap_lm_pretrained` answer questions, and with a
few solved examples in the prompt they label text as well. Two ordinary jobs
strain them. Ask one how this book's old `load_array` helper decides whether
to shuffle its data, and it answers from whatever it absorbed in training,
although the paragraph that says so is somewhere in the book. Ask it to sort
the messages of a support desk by topic, ten thousand of them an hour, and
it reads its solved examples again for every message. The first job needs a
way to find the passage. The second needs a model that learns from labels
once and then labels cheaply.

Both jobs are done by models that turn text into vectors, so that texts with
similar meanings lie close together. The idea is old. Linguists of the 1950s
proposed that a word is known by the company it keeps, and word2vec made
vectors built on that idea cheap to learn from billions of words. BERT
computed a vector for every token from the words on both sides of it, and a
small layer on top turned those vectors into labels. Contrastive training
then taught such *encoders* to place a question next to the passage that
answers it. The word indexes of search engines still find the exact terms
that vectors miss. Encoders do much of the routine work of search, filtering
and tagging, and once they find the passage, a decoder can answer from
documents it never saw in training.

The chapter moves in three steps. The first builds vectors for words from
co-occurrence counts (:numref:`sec_lm_word_embeddings`), for tokens in
context from a pretrained encoder that we fine-tune and set against a
prompted decoder (:numref:`sec_lm_bert`), and for whole texts by contrastive
training (:numref:`sec_lm_text_embeddings`). The second finds passages with
a word index and a vector index together, and asks what an approximate index
gives up for speed (:numref:`sec_lm_retrieval`). The third hands what was
found to a generator and asks, when an answer comes back wrong, which stage
failed (:numref:`sec_lm_rag`). Everything is measured on this book, cut into
passages and paired with questions whose answers and sources we know.
Language models, not people, drafted and checked those questions, and later
chapters of this part reuse both.

We start from released encoders rather than pretraining one
(:numref:`sec_transformer` shows how that works). Agents that decide when to
search belong to :numref:`chap_lm_agents`, and the retrieval of images to
:numref:`chap_cv`.

```toc
:maxdepth: 2

word-embeddings
bert
text-embeddings
retrieval
rag
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter, and each names
the section it goes with. All are freely accessible online. Methods that
build on the chapter without being taught in it, such as learned sparse
retrieval, late interaction and the many ways of improving a
retrieval-augmented pipeline, are collected in :numref:`sec_lm_encoders_fr`.

**Courses**

- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) opens with word vectors and a first assignment on them, the material of :numref:`sec_lm_word_embeddings`. Its pretraining lecture covers the masked-token encoders of :numref:`sec_lm_bert`, and its lecture on agents, tool use and retrieval-augmented generation the system of :numref:`sec_lm_rag`.
- [Berkeley CS288](https://cal-cs288.github.io/sp26) has lectures on text classification, the task of :numref:`sec_lm_bert`, on retrieval and RAG, and on experimental design, which frames the stage-by-stage measurement of :numref:`sec_lm_rag`.
- [CMU 11-711: Advanced Natural Language Processing](https://cmu-l3.github.io/anlp-spring2026) gives a lecture to retrieval and RAG, with readings on retrieval-based language models that extend :numref:`sec_lm_retrieval` and :numref:`sec_lm_rag`.

**Illustrated introductions**

- Jay Alammar's [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) draws the sliding window and the negative samples of skip-gram, a picture to read beside :numref:`sec_lm_word_embeddings`.
- His [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) draws the input layout and the task heads of a masked-token encoder, as :numref:`sec_lm_bert` uses them.
- The [ModernBERT announcement](https://huggingface.co/blog/modernbert) by the model's authors explains the design choices that :numref:`sec_lm_bert` describes: alternating local and global attention, unpadding and long inputs.

**Libraries and their documentation**

- The [Sentence Transformers documentation](https://sbert.net/) and its [training overview](https://sbert.net/docs/sentence_transformer/training_overview.html) describe the library that :numref:`sec_lm_text_embeddings` checks its derivations against, with the contrastive losses, Matryoshka training and quantized vectors of that section.
- The [Faiss wiki](https://github.com/facebookresearch/faiss/wiki) explains the inverted files, product quantizers and graphs of :numref:`sec_lm_retrieval`, how to name a combination of them, and how to choose one for a collection of a given size.
- [Pyserini](https://github.com/castorini/pyserini) provides BM25 and other word-index baselines with reproducible settings, and [BEIR](https://github.com/beir-cellar/beir) evaluates retrievers on collections of many kinds, the test behind the warning in :numref:`sec_lm_retrieval` that a dense retriever trained on one collection can fall below BM25 on another.

**Building retrieval-augmented systems**

- Jason Liu's [Systematically Improving Your RAG](https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/) and Eugene Yan's [Patterns for Building LLM-based Systems and Products](https://eugeneyan.com/writing/llm-patterns/) separate the retrieval and the generation of a pipeline and measure each, the decomposition that :numref:`sec_lm_rag` carries out on the book.
- The Hugging Face cookbook's [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag) builds a complete pipeline in one notebook, to set beside the pipeline of :numref:`sec_lm_rag` and compare stage by stage.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §19]{.kicker}

Encoders, Embeddings and Retrieval<br>
**word vectors · pretrained encoders · text embeddings · sparse, dense and approximate retrieval · retrieval-augmented generation**
:::
:::

::: {.slide title="Finding a Passage and Labeling a Stream Are Jobs for Encoders"}
[The problem]{.kicker}

- A decoder asked about this book answers from its weights, right or
  wrong; the paragraph that answers sits among thousands of passages.
- A prompted decoder reads its solved examples again for every message;
  a support desk sorts ten thousand messages an hour.
- Both jobs need vectors: related texts close together, and a label read
  off by a small head.

::: {.d2l-note}
How do we compute such vectors, search them fast, and tell which stage
failed when an answer built on them is wrong?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- say which statistic of a text a word vector factorizes, and check it;
- fine-tune an encoder for sentences, pairs and tags, and decide when it
  beats a prompted decoder;
- train a text embedder contrastively, and explain why raw pooled vectors
  retrieve poorly;
- build BM25 and dense retrieval, fuse their rankings and rerank;
- choose an approximate index by recall, latency and bytes;
- measure a retrieval-augmented generator stage by stage, and locate its
  failures.

**Running example:** this book, cut into passages, with questions drafted
and checked by language models, not people.

**Idea:** turn text into vectors that place related texts close together,
then find the close ones fast.
:::

::: {.slide title="Three Movements: Vectors, Finding, Using What Is Found"}
[Chapter map]{.kicker}

| movement | section | question |
|---|---|---|
| vectors | Word Embeddings | which statistic does a word vector factorize? |
| | Pretrained Encoders | when does a fine-tuned encoder beat a prompt? |
| | Text Embeddings from Contrastive Training | why do pooled states retrieve poorly until trained? |
| finding | Sparse, Dense and Approximate Retrieval | what floor must vectors beat, and what does an index trade for speed? |
| using what is found | Retrieval-Augmented Generation | where does the system fail, and how is the stage found? |

- **Uses:** transformers, cross-entropy, the evaluation tools of the
  previous chapter.
- **Skips:** pretraining at scale, agents, the retrieval of images.
:::
