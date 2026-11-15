# Further Reading
:label:`sec_lm_encoders_fr`

The retrieval-augmented system of :numref:`sec_lm_rag` is plain on purpose.
It scores passages with BM25 and with one vector per passage, fuses the two
rankings, and hands the best five passages to a generator used as released.
On this book the plain design worked well: a relevant passage reached the
context for 213 of the 220 answerable dev questions, and most of the
remaining errors happened after retrieval had done its job. A reader who
builds such a system over other documents will soon want to change
something. Should the index learn which words matter? Should a passage be
more than one vector? Should the generator be trained for the job, or the
context be so long that nothing needs retrieving? And once retrieval works,
where should the effort go next, and how would one know that it helped?

The literature has an answer to each of these questions, usually several.
This notebook collects those that change one mechanism of this chapter. Each
entry names the problem a method addresses, the one thing it changes, and
what that change costs. The first half stays with the representation of
passages. The second takes retrieval and generation together: training,
long context, improving the pipeline stage by stage, and evaluating it.

## Representations of Passages

### Learned Sparse Vectors

BM25 (:numref:`sec_lm_retrieval`) scores a passage only by the words it
shares with the question, with weights fixed by counts. A question that asks
about a "container of layers" misses a passage that only ever says
`nn.Sequential`. Dense vectors catch such paraphrases, but they need a
nearest-neighbor index and can lose an exact identifier. SPLADE
:cite:`Formal.Piwowarski.Clinchant.2021` keeps the inverted index and learns
its weights.

Its encoder ends in a masked-token head like the one of
:numref:`sec_lm_bert`, which gives, at every position $i$ of a text $t$, a
logit $w_{ij}$ for every vocabulary entry $j$. The weight of entry $j$ for the
whole text is

$$
\phi_j(t) = \sum_{i \in t} \log\big(1 + \max(0, w_{ij})\big),
$$

and a passage $d$ is scored against a question $q$ by the sparse inner product
$\sum_j \phi_j(q)\, \phi_j(d)$. The logarithm compresses a large logit, so
that no single term dominates the vector. It is a soft cousin of BM25's
saturation, but it acts on each position's weight, not on how often the term
occurs. Because the head scores the whole vocabulary, a passage can receive
weight on words it never contains. This is *document expansion*, the
"expansion" of "sparse lexical and expansion model", which the name
abbreviates.

Training uses the contrastive loss of :numref:`sec_lm_text_embeddings` plus
a penalty, called FLOPS, on the squared batch average of each $\phi_j$. A
search through the inverted index does work for every term that is active
in both the question and the passage, so its cost grows with the square of
how often a term is active. The penalty is a smooth estimate of that cost
from the batch: it punishes a term active in many texts far more than one active in a few, and
drives most weights to zero. Compared with BM25, the cost is one encoder
pass per question and per passage and a larger index. Pyserini, an academic
retrieval toolkit built on Lucene, searches such models with the same
inverted index it uses for BM25. A released checkpoint is `naver/splade-v3`,
which the sentence-transformers library loads as a sparse encoder; its
licence is non-commercial, and the download is gated behind accepting it.

### Late Interaction

The single vector of Qwen3-Embedding-0.6B compresses a passage of hundreds of
tokens into 1,024 numbers before any question is known. The reranker of
:numref:`sec_lm_retrieval` avoids the compression by reading question and
passage together, at the cost of a forward pass per pair: reranking 50
chunks took about half a second per question in :numref:`sec_lm_rag_fixes`.
ColBERT :cite:`Khattab.Zaharia.2020` sits between the two. It encodes every
passage offline into one short vector per token,
$\mathbf{v}_1, \ldots, \mathbf{v}_{L_d}$, encodes the question at query time
into $\mathbf{u}_1, \ldots, \mathbf{u}_{L_q}$, and lets each question token
find its best match in the passage:

$$
s(q, d) = \sum_{i=1}^{L_q} \max_{1 \le j \le L_d} \mathbf{u}_i^\top \mathbf{v}_j.
$$

Question and passage meet only in this sum of maxima, called *MaxSim*, after
both were encoded separately. This is the "late" interaction, and it lets the
passage vectors be computed once. Storage grows instead. At 128 coordinates
per token, a passage of 500 tokens needs 64,000 numbers, about sixty times
the single vector. ColBERTv2 :cite:`Santhanam.Khattab.SaadFalcon.ea.2022`
clusters all token vectors and stores each one as the index of its nearest
centroid plus the residual, coded with one or two bits per coordinate. That
is the coarse centroid of IVF from :numref:`sec_lm_retrieval`, applied to
tokens, with a scalar code for the residual, a simpler relative of product
quantization. The paper reports an index six to ten times smaller. The
authors' ColBERT repository implements indexing and search for ColBERTv2,
and `lightonai/GTE-ModernColBERT-v1` is a released ColBERT model built on
ModernBERT, the encoder family of :numref:`sec_lm_bert`, which its model
card loads with the PyLate library.

On this book late interaction has little left to find: the dense retriever
alone reaches a Recall@10 of 0.98. Its extra storage pays where one vector
per passage is too coarse, a limit that the last subsection of this half
makes precise.

### Context around a Chunk

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
from the top 20 (one minus Recall@20) fell from 5.7% to 2.9%, about half,
when both the embeddings and BM25 saw the prepended context. A reranker on
top took it to 1.9%. The cost is one generation per chunk at indexing time,
each with the whole document in its prompt, which prompt caching makes
affordable, and another run whenever a document changes.

*Late chunking* :cite:`Gunther.Mohr.Williams.ea.2024` needs no generation. A
long-context embedder reads the whole document once, and each chunk's vector
is the mean of the token vectors inside the chunk's span. Every token vector
has attended to the whole document, so the chunk vector carries what its
neighbors say, without any further training. It has two costs. The
encoder runs over the whole document in one pass, whose attention cost grows
with the square of the document's length, and every chunk's vector now
depends on the whole document, so editing one paragraph re-encodes every
chunk of that document. The method needs two properties that
Qwen3-Embedding-0.6B lacks (:numref:`sec_lm_text_embeddings`). Under its
causal attention a token sees only the text before it, so an early chunk
gains nothing from later ones. And the model was trained to deliver one
vector at the last token, so the mean of a span's hidden states is not a
vector it ever learned to produce. The ModernBERT encoder that
:numref:`sec_lm_text_embeddings` trained with mean pooling has both
properties, and its architecture accepts 8,192 tokens. Its contrastive
training saw only short passages, though, so whether its vectors of long
documents hold up is the first thing to measure.

### Limits of a Single Vector

How many different answers can one vector per document give? The smallest
case shows the problem. Embed every document as a single number $x_i$ and
the question as a number $q$. The score $q x_i$ ranks the documents by $x_i$
when $q > 0$ and by $-x_i$ when $q < 0$, so however many documents there
are, only two top-$k$ sets can ever be returned: the $k$ largest and the $k$
smallest. In two dimensions the ranking of $n$ documents changes only where
the direction of $\mathbf{q}$ crosses a line perpendicular to some difference
$\mathbf{x}_i - \mathbf{x}_j$, at most $n(n-1)$ times around the circle. That
allows at most $n(n-1)$ rankings, while the number of possible top-$k$ sets,
$\binom{n}{k}$, grows like $n^k$.

:citet:`Weller.Boratko.Naim.ea.2026` prove a bound of this kind for any
dimension, with results from learning theory. For embeddings of dimension $d$, the number of
distinct top-$k$ sets of documents that some query can return is limited by
$d$. The limit shows already for top-two results, and even when free vectors
are optimized directly on the test queries. On their LIMIT dataset, built
from very simple queries, state-of-the-art embedders fail. The result
concerns one vector per document compared by an inner product. Learned
sparse vectors are single vectors too, but with the vocabulary's tens of
thousands of dimensions, so the bound bites far later. Late interaction
keeps a vector per token and is not covered by the argument. On this book,
where BM25, the dense retriever and their fusion all reach a Recall@10
between 0.96 and 0.98, the bound is not what limits retrieval.

## Retrieval and Generation

### Training for Retrieval-Augmented Generation

In :numref:`sec_lm_rag_fixes` the generator and both retrievers stay frozen. The
marginalization derived there says what training would change. With
$L_z = p(y \mid x, z)$ the probability that the generator gives the answer
$y$ when it reads document $z$, the answer's probability is a mixture over
retrieved documents, and its gradient with respect to a document's score is
the posterior weight minus the prior weight (:eqref:`eq_lm_rag_grad`):

$$
p_S(y \mid x) = \sum_z p(z \mid x)\, L_z, \qquad
\frac{\partial \log p_S(y \mid x)}{\partial s_z} = p(z \mid x, y) - p(z \mid x).
$$

A document gains score when it explains the answer better than the
retriever expected, and loses score otherwise. Two lines of work train the
retriever through this gradient. Two others leave the retriever alone and
train the generator to use what it is given.

**Joint training.** RAG :cite:`Lewis.Perez.Piktus.ea.2020` trains the query
encoder and a sequence-to-sequence generator together on question answering
by this marginal likelihood. The passage encoder and its index stay fixed:
re-encoding the passages during training is costly, and the authors found
it unnecessary for strong performance.

**Retrieval in pretraining.** REALM :cite:`Guu.Lee.Tung.ea.2020` moves
retrieval into masked-token pretraining: the model retrieves a passage
before it fills in a blank. The retrieved passage is a latent variable, and
the retriever is trained by the same gradient. REALM updates the passage
encoder too, so its index goes stale and is rebuilt asynchronously every few
hundred steps.

**Distractors.** RAFT :cite:`Zhang.Patil.Jain.ea.2024` fine-tunes the
generator on questions shown with the relevant document among distractors,
toward chain-of-thought answers that quote the relevant passage. It targets
the `ignored` failures of :numref:`sec_lm_rag_fixes`, where a relevant passage was
in the context and the answer missed it. The fix chosen in advance there
asked Qwen3.5-9B, by prompt alone, to quote before answering, with the best
passages at both ends. It lowered accuracy by 0.08 on dev and on test,
because the quoting format crowded out the sentence that declares a
question unanswerable: false answers to unanswerable dev questions rose from
2 to 15. RAFT trains the quoting into the weights instead, at the cost of a
fine-tuning run per domain, with distractors drawn from one's own retriever.
Its examples with distractors only still target the answer, which teaches
answering from memory, not abstaining, so whether RAFT avoids the failure
that our prompt created is a question for the procedure of
:numref:`sec_lm_rag_fixes`.

**Reflection tokens.** Self-RAG :cite:`Asai.Wu.Wang.ea.2024` trains the
generator to emit special tokens that decide whether to retrieve, judge
whether a passage is relevant, and judge whether the generator's own sentence
is supported. A critic model labels the training data with these tokens
offline. At inference their probabilities rank candidate continuations, so
each answer costs several candidate generations.

### Long Context and Retrieval

If a model can read a million tokens, why retrieve at all? Three findings
argue against dropping retrieval. First, every question re-reads its whole
context, and more context did not give better answers. In
:numref:`sec_lm_rag_fixes`, a 6,000-token budget without the five-chunk limit gave
the generator about 3.4 times the baseline's context. Accuracy changed by
$+0.01$ (interval $-0.03$ to $+0.05$), and the time per answer doubled,
partly because the longer prompts ran in smaller batches. Second, accepting
a length is not using it. Models use the start and the end of a long
context better than its middle :cite:`Liu.Lin.Hewitt.ea.2024`, and an
exercise of :numref:`sec_lm_rag` measures this effect on the book. How much
of its window a model actually uses, its effective context :cite:`Hsieh.Sun.Kriman.ea.2024`, we measure
ourselves in :numref:`sec_lm_long_context`. Third, :citet:`Jin.Yoon.Han.ea.2025` find that for many long-context models
the quality of answers first rises and then falls as more retrieved passages
are added. They name retrieved hard negatives, passages that look relevant
and are not, as a main cause. When many passages are retrieved, placing the
highest-scoring ones at both ends of the context recovers part of the loss
without training (the `edges` order of `d2l.RAGPipeline` does this), and
fine-tuning on retrieved contexts recovers more.

Long context wins in two cases: a small collection read by many questions,
whose shared prefix can be cached, and a question that needs a whole document
at once, such as a summary of its themes. Everywhere else retrieval stays,
because a model's knowledge is fixed when it is trained
:cite:`Husain.Clavie.2025`. A collection that changes after training, or
that is larger than any window, has to be searched.

### Improving the Pipeline

Each stage of :numref:`sec_lm_rag` can be improved, and the failure counts
of :numref:`sec_lm_rag_fixes` say where to start. On the book, retrieval rarely failed. Of
the 24 wrong answers to answerable dev questions, none lacked a relevant
chunk in the ranking and 5 had it ranked too low to enter the context. The
other 19 had a relevant passage in the context: in 12 the answer missed it,
in 6 it stated something its passages do not support, and in 1 the
generator abstained. Methods aimed at the query have little to fix on such a
collection, and methods aimed at the context and the generator have more.
On a collection where retrieval is harder the balance can reverse, and the
same counts would show it.

**The query.** HyDE :cite:`Gao.Ma.Lin.ea.2023` asks the generator for a
hypothetical answer passage and embeds that instead of the question, so the
dense side compares a passage with passages. RAG-Fusion :cite:`Rackauckas.2024`
asks the generator for several rewrites of the question and fuses their
rankings with the reciprocal-rank fusion of :numref:`sec_lm_retrieval`.
Step-back prompting :cite:`Zheng.Mishra.Chen.ea.2024` retrieves for a more
general version of the question as well as for the question itself. And
decomposition, as in least-to-most prompting :cite:`zhou2023least`, splits a
question into sub-questions, each of which retrieves separately. All four
target relevant chunks that were not retrieved or were ranked too low. Each
costs a generation before retrieval and one more search per extra query.
`d2l.RAGPipeline` has HyDE and a two-question decomposition built in
(`hyde=True`, `decompose=True`), which :numref:`sec_lm_rag` leaves
unmeasured.

**Several rounds of retrieval.** A multi-hop question needs a second passage
that only the first one reveals. The 20 multi-passage (`multihop`) dev
questions of :numref:`sec_lm_rag` each need more than one passage, so they
are where a reader can try these methods on the book. The methods differ in what
decides that another search is needed. In Self-Ask
:cite:`Press.Zhang.Min.ea.2023` it is the model's own follow-up question,
answered with a search. In IRCoT :cite:`Trivedi.Balasubramanian.Khot.ea.2023`
it is every step of a chain of thought, each new sentence serving as the
next query. FLARE :cite:`Jiang.Xu.Gao.ea.2023` drafts the next sentence and
retrieves only when the draft contains tokens of low probability.
Adaptive-RAG :cite:`Jeong.Baek.Cho.ea.2024` decides before retrieval, with a
small classifier that routes each question to no retrieval, one round or
several, and so also saves rounds on simple questions. CRAG
:cite:`Yan.Gu.Zhu.ea.2024` decides after retrieval: a light evaluator scores
the retrieved documents, and a low score triggers a further search. These
methods target passages that one search misses, most often on multi-hop
questions, and their latency grows with the number of rounds. In Search-R1
:cite:`Jin.Zeng.Yue.ea.2025` and ReSearch :cite:`Chen.Sun.Li.ea.2025` the
decision is a policy learned by reinforcement learning from the correctness
of the final answer, with no labeled search steps, so the model itself
decides when and what to search. At that point the pipeline has become an
agent with a search tool, the subject of :numref:`chap_lm_agents`.

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
model, Qwen3.8-27B, and checks the judge first. On the 93 of the 100
calibration pairs where the two labeling models agree, its correctness
verdicts match theirs at $\kappa = 0.85$ (interval 0.70 to 0.97), where
$\kappa$ is agreement corrected for chance. Against each labeler alone on
all 100 pairs it reaches 0.83 and 0.71, about as well as the labelers agree
with each other (0.81). Those labels were written by two frontier language
models, not by people, so a blind spot that labelers and judge share goes
undetected.

Three other approaches each change one thing. RAGAS
:cite:`Es.James.EspinosaAnke.ea.2024` needs no reference answers. A language
model splits the answer into statements and checks each against the context
(faithfulness), and it generates questions back from the answer to compare
with the original one (answer relevance). These scores measure what the
context supports, not whether the answer is right. ARES
:cite:`SaadFalcon.Khattab.Potts.ea.2024` fine-tunes small judges on synthetic
data and then corrects their estimates with a small set of labels written
by people, through prediction-powered inference. With $f(x)$ the judge's
verdict on an answer $x$ and $y_i$ the label of a labeled one, the judge's
average over all $N$ answers is corrected by its average error on the $n$
labeled answers,

$$
\hat\theta = \frac{1}{N}\sum_{j=1}^{N} f(x_j) - \frac{1}{n}\sum_{i=1}^{n} \big(f(x_i) - y_i\big),
$$

so the estimate stays unbiased however wrong the judge is, and its interval
widens with the judge's errors. :numref:`sec_lm_rag` uses its labeled set
only to measure the judge's agreement, and its labels come from models.
ALCE :cite:`Gao.Yen.Yu.ea.2023` scores citations with an entailment model,
the kind of classifier that :numref:`sec_lm_bert` fine-tunes on MNLI: the
passages cited for a statement must together entail it (recall), and each
cited passage must contribute (precision). :numref:`sec_lm_rag` computes
citation precision from gold chunk ids instead, which counts a correct
citation of a passage outside the gold set as an error.

Every grader in this subsection, the judge of :numref:`sec_lm_rag`
included, is a model, trusted only as far as it was checked against labels
from people or from stronger models. How to grade a pipeline on a new
collection when neither is at hand is an open problem.
