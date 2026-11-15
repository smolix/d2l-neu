# Further Reading
:label:`sec_lm_encoders_fr`

The retrieval-augmented system of :numref:`sec_lm_rag` is plain on purpose.
It scores passages with BM25 and with one vector per passage, fuses the two
rankings, and hands the best five passages to a generator used as released.
On this book, plain worked well: a relevant passage reached the context for
213 of the 220 answerable dev questions, and most of the remaining errors
happened after retrieval had done its job. A reader who builds such a system
over other documents will soon want to change something. Should the index
learn which words matter? Should a passage be more than one vector? Should
the generator be trained for the job, or the context be so long that nothing
needs retrieving?

The literature has an answer to each of these questions, usually several.
This notebook collects the ones worth knowing. Each entry names the problem a
method addresses, the one thing it changes relative to the mechanisms of this
chapter, and what that change costs. Nothing else in the book depends on it.
The first half stays with retrieval; the second asks what happens once the
passages are found.

## Representations of Passages

### Learned Sparse Vectors

BM25 (:numref:`sec_lm_retrieval`) scores a passage only by the words it
shares with the question, with weights fixed by counts. A question that asks
about a "container of layers" misses a passage that only ever says
`nn.Sequential`. Dense vectors catch such paraphrases, but they need a
nearest-neighbour index and can lose an exact identifier. SPLADE
:cite:`Formal.Piwowarski.Clinchant.2021` keeps the inverted index and learns
its weights. Its encoder ends in a masked-token head like the one of
:numref:`sec_lm_bert`, which gives, at every position $i$ of a text $t$, a
logit $w_{ij}$ for every vocabulary entry $j$. The weight of entry $j$ for the
whole text is

$$
\phi_j(t) = \sum_{i \in t} \log\big(1 + \max(0, w_{ij})\big),
$$

and a passage $d$ is scored against a question $q$ by the sparse inner product
$\sum_j \phi_j(q)\, \phi_j(d)$. The logarithm damps a term that fires at many
positions, the job that saturation does in BM25. Because the head scores the
whole vocabulary, a passage can receive weight on words it never contains,
which is the *expansion* in the name. Training uses the contrastive loss of
:numref:`sec_lm_text_embeddings` plus a penalty on the squared batch average
of each $\phi_j$, which drives most weights to zero and keeps the
term-document matrix sparse. Compared with BM25, the cost is one encoder pass
per question and per passage and a larger index. Pyserini searches such
models with the same Lucene inverted index it uses for BM25.

### Late Interaction

One vector compresses a passage of hundreds of tokens into 1,024 numbers
before any question is known. The reranker of :numref:`sec_lm_retrieval`
avoids the compression by reading question and passage together, at the cost
of a forward pass per pair: reranking 50 chunks took about half a second per
question in :numref:`sec_lm_rag`. ColBERT :cite:`Khattab.Zaharia.2020` sits
between the two. It encodes every passage offline into one short vector per
token, $\mathbf{v}_1, \ldots, \mathbf{v}_{L_d}$, encodes the question at query
time into $\mathbf{u}_1, \ldots, \mathbf{u}_{L_q}$, and lets each question
token find its best match in the passage:

$$
s(q, d) = \sum_{i=1}^{L_q} \max_{1 \le j \le L_d} \mathbf{u}_i^\top \mathbf{v}_j.
$$

Question and passage meet only in this sum of maxima, called *MaxSim*, after
both were encoded separately. This is the "late" interaction, and it lets the
passage vectors be computed once. Storage grows instead. At 128 coordinates
per token, a passage of 500 tokens needs 64,000 numbers, about sixty times
the single vector of Qwen3-Embedding-0.6B. ColBERTv2
:cite:`Santhanam.Khattab.SaadFalcon.ea.2022` stores each token vector as the
index of its nearest centroid plus the residual at one or two bits per
coordinate: the coarse centroid of IVF and the short code of product
quantization from :numref:`sec_lm_retrieval`, applied to tokens. The paper
reports an index six to ten times smaller, and it also trains the encoder on
a cross-encoder's scores instead of binary labels. The authors' ColBERT
repository implements indexing and search for ColBERTv2.

### Context for a Chunk

A chunk cut from the middle of a section often leans on what came before it:
"this layer", "the loss above". The chunks of :numref:`sec_lm_retrieval` are
indexed with their heading path, "Chapter > Section > Subsection", which is a
context header written by a rule. Two methods give a chunk more of its
surroundings.

*Contextual retrieval* :cite:`Anthropic.2024` shows a language model the
whole document and one of its chunks, asks for a sentence or two (50 to 100
tokens) that situate the chunk, and prepends the answer before both embedding
and BM25 indexing. On the authors' collections of code, fiction, arXiv papers
and science papers, the share of questions whose relevant chunk was missing
from the top 20 (one minus Recall@20) fell from 5.7% to 3.7% with contextual
embeddings, to 2.9% with contextual BM25 as well, and to 1.9% with a reranker
on top. The cost is one generation per chunk at indexing time, each with the
whole document in its prompt, which prompt caching makes affordable, and
another run whenever a document changes.

*Late chunking* :cite:`Gunther.Mohr.Williams.ea.2024` needs no generation. A
long-context embedder reads the whole document once, and each chunk's vector
is the mean of the token vectors inside the chunk's span. Every token vector
has attended to the whole document, so the chunk vector carries what its
neighbours say, without any further training. The method needs two properties
that Qwen3-Embedding-0.6B lacks (:numref:`sec_lm_text_embeddings`). Under its
causal attention a token sees only the text before it, so an early chunk gains
nothing from later ones. And the model was trained to deliver one vector at
the last token, so the mean of a span's hidden states is not a vector it ever
learned to produce. The ModernBERT encoder that
:numref:`sec_lm_text_embeddings` trained with mean pooling has both
properties and accepts 8,192 tokens, which makes it the natural model on
which to try late chunking.

### Limits of a Single Vector

How many different answers can one vector per document give?
:citet:`Weller.Boratko.Naim.ea.2026` connect known results from learning
theory to retrieval. For embeddings of dimension $d$, the number of distinct
top-$k$ sets of documents that some query can return is limited by $d$. The
limit holds even when free vectors are optimized directly on the test
queries, and returning every pair of documents as a top-two result already
needs a relatively high dimension. Their LIMIT dataset, built from very simple
queries, defeats state-of-the-art embedders. The result concerns one vector
per document compared by an inner product. Learned sparse vectors have the
dimension of the vocabulary, and late interaction keeps a vector per token,
so the argument does not apply to them in the same form. On this book, where
BM25, the dense retriever and their fusion all reach a Recall@10 between 0.96
and 0.98, the bound is not what limits retrieval.

## Retrieval and Generation

### Training for Retrieval-Augmented Generation

In :numref:`sec_lm_rag` the generator and both retrievers stay frozen. The
marginalization derived there says what training would change: the
probability of an answer is a sum over retrieved passages, each weighted by
the retriever's probability, and the gradient that reaches the retriever's
score of a passage is the passage's posterior weight given the answer minus
its prior weight. Passages that make the right answer likely are pulled up.
Four lines of work use this freedom differently.

**Joint training.** RAG :cite:`Lewis.Perez.Piktus.ea.2020` trains the query
encoder and a sequence-to-sequence generator together on question answering
by this marginal likelihood. The passage encoder and its index stay fixed,
because re-encoding millions of passages after every step would cost more than
the training itself.

**Retrieval in pretraining.** REALM :cite:`Guu.Lee.Tung.ea.2020` moves
retrieval into masked-token pretraining: the model retrieves a passage before
it fills in a blank, and the passage is a latent variable trained by the same
gradient. REALM updates the passage encoder too, so its index goes stale and
is rebuilt asynchronously every few hundred steps.

**Distractors.** RAFT :cite:`Zhang.Patil.Jain.ea.2024` leaves retrieval alone
and fine-tunes the generator on questions shown with the relevant document
among distractors, and sometimes with distractors only, toward
chain-of-thought answers that quote the relevant passage. It targets the
`ignored` failures of :numref:`sec_lm_rag`, where a relevant passage was in
the context and the answer missed it. The fix chosen in advance there asked
Qwen3.5-9B to quote before answering by prompt alone, and it lowered dev
accuracy by 0.08 (interval $-0.12$ to $-0.05$). RAFT trains the behaviour into
the weights instead, at the cost of a fine-tuning run per domain, with
distractors drawn from one's own retriever.

**Reflection tokens.** Self-RAG :cite:`Asai.Wu.Wang.ea.2024` trains the
generator to emit special tokens that decide whether to retrieve, judge
whether a passage is relevant, and judge whether the generator's own sentence
is supported. A critic model labels the training data with these tokens
offline. At inference their probabilities rank candidate continuations, so
one answer costs several.

### Long Context and Retrieval

If a model can read a million tokens, why retrieve at all? Three findings
argue against dropping retrieval. First, every question re-reads its whole
context. In :numref:`sec_lm_rag`, raising the context budget from 3,000 to
6,000 tokens roughly doubled the time per answer and changed accuracy by
$+0.008$ (interval $-0.029$ to $+0.051$). Second, accepting a length is not
using it. :numref:`sec_lm_long_context` measures the effective context of a
model, which falls short of its window, and models use the start and the end
of a long context better than its middle :cite:`Liu.Lin.Hewitt.ea.2024`; an
exercise of :numref:`sec_lm_rag` measures this effect on the book. Third,
:citet:`Jin.Yoon.Han.ea.2025` find that for many long-context models the
quality of answers first rises and then falls as more retrieved passages are
added. They name retrieved hard negatives, passages that look relevant and are
not, as a main cause. When many passages are retrieved, placing the
highest-scoring ones at both ends of the context recovers part of the loss
without training (the `edges` order of `d2l.RAGPipeline` does this), and
fine-tuning on retrieved contexts recovers more.

Long context wins in two cases: a small collection read by many questions,
whose shared prefix can be cached, and a question that needs a whole document
at once, such as a summary of its themes. :citet:`Husain.Clavie.2025` make
the practical case for the rest: a model's knowledge is fixed when it is
trained, and putting everything into every query costs too much, so larger
windows do not retire retrieval.

### Improving the Pipeline

Each stage of :numref:`sec_lm_rag` can be improved, and the failure counts
measured there say where to start. On the book, retrieval rarely failed. Of
the 26 wrong dev answers, none lacked a relevant chunk in the ranking and 5
had it ranked too low to enter the context, while in 12 the relevant passage
was in the context and the answer missed it. Methods aimed at the query have
little to fix on such a collection, and methods aimed at the context and the
generator have more. On a collection where retrieval is harder the order
reverses, and the same counts would show it.

**The query.** HyDE :cite:`Gao.Ma.Lin.ea.2023` asks the generator for a
hypothetical answer passage and embeds that instead of the question, so the
dense side compares a passage with passages. RAG-Fusion :cite:`Rackauckas.2024`
writes several rewrites of the question and fuses their rankings with the
reciprocal-rank fusion of :numref:`sec_lm_retrieval`. Step-back prompting
:cite:`Zheng.Mishra.Chen.ea.2024` also retrieves for a more general version of
the question, and decomposition :cite:`zhou2023least` splits a question into
sub-questions that retrieve separately. All four target relevant chunks that
were not retrieved or were ranked too low. Each costs a generation before
retrieval and one more search per extra query. `d2l.RAGPipeline` has HyDE and
a two-question decomposition built in (`hyde=True`, `decompose=True`), which
:numref:`sec_lm_rag` leaves unmeasured.

**Several rounds of retrieval.** A multi-hop question needs a second passage
that only the first one reveals. Self-Ask :cite:`Press.Zhang.Min.ea.2023` has
the model state follow-up questions and answer each with a search. IRCoT
:cite:`Trivedi.Balasubramanian.Khot.ea.2023` uses each new sentence of a chain
of thought as the next query, and FLARE :cite:`Jiang.Xu.Gao.ea.2023` drafts
the next sentence and retrieves only when the draft contains tokens of low
probability. Adaptive-RAG :cite:`Jeong.Baek.Cho.ea.2024` trains a small
classifier that routes each question to no retrieval, one round or several.
CRAG :cite:`Yan.Gu.Zhu.ea.2024` scores the retrieved documents with a light
evaluator and searches further when the score is low. These methods target
missed passages on multi-hop questions, and their latency grows with the
number of rounds. Search-R1 :cite:`Jin.Zeng.Yue.ea.2025` and ReSearch
:cite:`Chen.Sun.Li.ea.2025` learn the same behaviour by reinforcement learning
from the correctness of the final answer, with no labelled search steps, so
the model itself decides when and what to search. At that point the pipeline
has become an agent with a search tool, the subject of
:numref:`chap_lm_agents`.

**The context.** RECOMP :cite:`Xu.Shi.Choi.2024` trains an extractive
compressor that keeps the useful sentences and an abstractive one that
summarizes several documents. Both learn from whether the generator then
answers well, and either may return nothing when retrieval did not help.
LongLLMLingua :cite:`Jiang.Wu.Luo.ea.2024` deletes the tokens a small language
model judges unnecessary for the question. Both target the `ignored` class and
the cost of long contexts; the risk is deleting the one sentence that held the
answer.

**The index.** Some questions are answered by no single chunk: what a whole
chapter covers, or which themes run through a collection. RAPTOR
:cite:`Sarthi.Abdullah.Tuli.ea.2024` clusters chunk vectors, has a language
model summarize each cluster, and repeats on the summaries, so that retrieval
can choose among the levels of a tree. GraphRAG :cite:`Edge.Trinh.Cheng.ea.2024`
has a language model extract a graph of entities and their relations,
summarizes groups of closely related entities in advance, and answers a global
question from those summaries. Both need language-model calls over the
whole collection at indexing time, repeated when it changes. This book already
has both structures, written by its authors: a tree of headings and a graph of
cross-references. The chunks of :numref:`sec_lm_retrieval` use only the tree,
as headers.

### Evaluating the Pipeline

:numref:`sec_lm_rag` grades answers against reference answers with a judge
model, Qwen3.8-27B, and checks the judge first: on the 100 calibration pairs
its correctness verdicts agree with the labels at $\kappa = 0.85$ (interval
0.70 to 0.97). Those labels were written by two frontier language models, not
by people, so a blind spot that labellers and judge share goes undetected.
Three other approaches each change one thing. RAGAS
:cite:`Es.James.EspinosaAnke.ea.2024` needs no reference answers. A language
model splits the answer into statements and checks each against the context
(faithfulness), and it generates questions back from the answer to compare
with the original one (answer relevance). These scores measure what the
context supports, not whether the answer is right. ARES
:cite:`SaadFalcon.Khattab.Potts.ea.2024` fine-tunes small judges on synthetic
data and then corrects their estimates with a few hundred human labels by
prediction-powered inference, which gives intervals that account for the
judge's errors; :numref:`sec_lm_rag` uses its labelled set only to measure
agreement. ALCE :cite:`Gao.Yen.Yu.ea.2023` scores citations with an entailment
model: the passages cited for a statement must together entail it (recall),
and each cited passage must contribute (precision). :numref:`sec_lm_rag`
computes citation precision from gold chunk ids instead, which counts a
correct citation of a passage outside the gold set as an error.

Every method in this notebook was evaluated in its paper on public
benchmarks. Whether it helps on a given collection is a question that the
measurements of :numref:`sec_lm_rag` can answer, one stage at a time.
