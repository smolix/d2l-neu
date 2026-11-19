# Encoders, Embeddings and Retrieval
:label:`chap_lm_encoders`

The previous chapter left us able to do a good deal with a released
language model. We can load one, check that it computes on our machine what
its authors computed on theirs, prompt it, and put an error bar on its
score. Two ordinary requests still defeat it. Ask it how this book's old
`load_array` helper decides whether to shuffle its data, and it answers from
whatever it absorbed in training. The answer we will meet is fluent and
wrong, although the paragraph that says so is somewhere in the book. Ask
it to sort the messages of a support desk by topic, ten thousand of them an
hour, and it reads its solved examples again for every message. The first
request needs a way to find a passage. The second needs a model that learns
from labels once and then labels cheaply.

Both requests are met by models that turn text into vectors, so that texts with
similar meanings lie close together. The idea is older than neural networks.
Linguists of the 1950s proposed that a word is known by the company it
keeps, and word2vec made vectors built on that idea cheap to learn from
billions of words. BERT computed a vector for every token from the words on
both sides of it, and a small layer on top turned those vectors into labels.
Contrastive training then taught such *encoders* to place a question next to
the passage that answers it. Encoders still do much of the routine work of
search, filtering and tagging, and once they have found the passage, a
decoder can answer from documents it never saw in training. One old tool
holds its ground through all of this. The word index of a search engine,
which does little more than count the words a question shares with a
passage, turns out to be the hardest baseline to beat on this book.

The chapter moves in three steps. First we compute vectors: for words, from
co-occurrence counts (:numref:`sec_lm_word_embeddings`); for tokens in
context, from a pretrained encoder that we fine-tune and set against a
prompted decoder (:numref:`sec_lm_bert`); and for whole texts, by
contrastive training (:numref:`sec_lm_text_embeddings`). Then we find
passages with a word index and a vector index together, and ask what an
approximate index gives up for speed (:numref:`sec_lm_retrieval`). Last we
hand what was found to a generator, and when an answer comes back wrong, we
ask which stage lost it (:numref:`sec_lm_rag`, :numref:`sec_lm_rag_fixes`).

From the third section on, every experiment runs on one collection: this
book without the Language Models part, cut into passages and paired with
questions whose answers and source passages we know. Language models, not
people, drafted and checked those questions, and :numref:`sec_lm_rag` says
what that can miss. The `load_array` question is one of them. We follow it
from a search that misses its passage, through a retriever that ranks the
passage first, to a generator that answers it from the page. The corpus,
the retriever and the questions return later in this part; the agents of
:numref:`chap_lm_agents`, for instance, search the book with this retriever
and decide for themselves when to do so.

We start from released encoders and do not pretrain one;
:numref:`sec_transformer` showed on a small scale how that works. Images,
which the same contrastive methods can place in one space with text, are
left out.

We assume the transformer of :numref:`chap_transformers` and the prompting,
scoring and paired error bars of :numref:`chap_lm_pretrained`. We ran the
notebooks on one GPU with 24 GB of memory and the judge that grades answers
on two, but no notebook requires a GPU. The long computations were done once
and their results published with the book. Each notebook fetches them and
recomputes a small part of each as a check; without a GPU it takes a
cheaper path, with fewer examples or a smaller search, and its output says
which. The questions were drafted and checked
through a paid API, and the generator's answers and the judge's verdicts
took hours of GPU time. All of these were recorded once, so the notebooks
replay them and call no API.

```toc
:maxdepth: 2

word-embeddings
bert
text-embeddings
retrieval
rag
rag-failures
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter, and each names
the section it goes with. All are freely accessible online. If you read
only three, read the training overview of Sentence Transformers, the Faiss
wiki and Jason Liu's essay on improving a RAG system: between them they
cover how an embedder is trained, how an index is chosen, and how to decide
what to fix next. Methods that build on the chapter without being taught in
it, such as learned sparse retrieval, late interaction and the many ways of
improving a retrieval-augmented pipeline, are collected in
:numref:`sec_lm_encoders_fr`.

**Courses**

- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) opens with word vectors and a first assignment on them, the material of :numref:`sec_lm_word_embeddings`. Its pretraining lecture covers the masked-token encoders of :numref:`sec_lm_bert`, and its lecture on agents, tool use and retrieval-augmented generation the system of :numref:`sec_lm_rag`.
- [Berkeley CS288](https://cal-cs288.github.io/sp26) has lectures on text classification, the task of :numref:`sec_lm_bert`, on retrieval and RAG, and on experimental design, which frames the one-change-at-a-time measurements of :numref:`sec_lm_rag_fixes`.
- [CMU 11-711: Advanced Natural Language Processing](https://cmu-l3.github.io/anlp-spring2026) gives a lecture to retrieval and RAG, with readings on retrieval-based language models that extend :numref:`sec_lm_retrieval` and :numref:`sec_lm_rag`.

**Illustrated introductions**

- Jay Alammar's [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) draws the sliding window and the negative samples of skip-gram, a picture to read beside :numref:`sec_lm_word_embeddings`.
- His [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) draws the input layout and the task heads of a masked-token encoder, as :numref:`sec_lm_bert` uses them.
- The [ModernBERT announcement](https://huggingface.co/blog/modernbert) by the model's authors explains the design of the encoder that :numref:`sec_lm_bert` uses, including two choices the section leaves out: unpadding and its attention kernels.

**Libraries and their documentation**

- The [Sentence Transformers documentation](https://sbert.net/) and its [training overview](https://sbert.net/docs/sentence_transformer/training_overview.html) describe the library that :numref:`sec_lm_text_embeddings` checks its derivations against, with the contrastive losses, Matryoshka training and quantized vectors of that section.
- The [Faiss wiki](https://github.com/facebookresearch/faiss/wiki) explains the inverted files, product quantizers and graphs of :numref:`sec_lm_retrieval`, how to name a combination of them, and how to choose one for a collection of a given size.
- [Pyserini](https://github.com/castorini/pyserini) provides BM25 and other word-index baselines with reproducible settings, and [BEIR](https://github.com/beir-cellar/beir) evaluates retrievers on collections of many kinds, the test behind the warning in :numref:`sec_lm_retrieval` that a dense retriever trained on one collection can fall below BM25 on another.

**Building retrieval-augmented systems**

- Jason Liu's [Systematically Improving Your RAG](https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/) and Eugene Yan's [Patterns for Building LLM-based Systems and Products](https://eugeneyan.com/writing/llm-patterns/) separate the retrieval and the generation of a pipeline and measure each, the decomposition that :numref:`sec_lm_rag` and :numref:`sec_lm_rag_fixes` carry out on the book.
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

| request | a prompted decoder |
|---|---|
| how does the book's `load_array` decide whether to shuffle? | without the book, Qwen3.5-9B invents a `shuffle` argument; the answer is one paragraph among 5,324 passages |
| label 10,000 support messages an hour | re-reads its 8 solved examples for every item: about 1,500 tokens per item, against 40 for a fine-tuned encoder |

::: {.d2l-note}
Both jobs need vectors that put related texts close together: one per
passage to search, one per token for a small labeling head.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- say which statistic of a text a word vector factorizes, and check it;
- fine-tune an encoder for sentences, pairs and tags, and say when it
  beats a prompt;
- train a text embedder contrastively, and explain why raw pooled vectors
  retrieve poorly;
- build BM25 and dense retrieval, fuse their rankings and rerank;
- choose an approximate index by recall, latency and bytes;
- measure a retrieval-augmented generator stage by stage, and locate its
  failures.

**Running example:** this book, 5,324 passages, and 220 questions with
known sources, drafted and checked by models; the `load_array` question
travels through the decks.

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
| using what is found | Retrieval-Augmented Generation | how is each stage of the system measured? |
| | Failures and Fixes in RAG | which stage lost each wrong answer, and does a fix chosen in advance help? |

- **Uses:** transformers; prompting, scoring and error bars.
- **Skips:** pretraining an encoder, agents that search, images.
:::
