# Further Reading
:label:`sec_attention-further-reading`

The sections of this chapter teach one representative of each idea: one
soft lookup, one scoring function, one way to split attention into heads,
a few ways to encode position, one exact kernel and two ways to avoid the
quadratic cost, and one circuit traced through a small model. This section
points to topics they had no room for: other readings of the lookup and its
normalization, what trained heads do and how many are needed, further
positional schemes and their limits, older and newer ways to reduce the
cost, and the tools that analyze circuits in large models. Each topic states
the problem it addresses and what it changes relative to the chapter.
Implementations, tutorials, and courses for the material that the chapter
does teach are listed in its introduction, :numref:`chap_attention`. Nothing
else in the book depends on this section.

## The Lookup and Its Normalization

### Attention as Associative Memory

:numref:`sec_queries-keys-values` read attention as a smoothed lookup in a
table of key-value pairs, with a bandwidth that decides between retrieving
one entry and averaging many. :citet:`Ramsauer.Schafl.Lehner.ea.2021` read
the same computation as memory retrieval: softmax attention with one query
is one update step of a modern Hopfield network whose stored patterns are
the keys. The network's energy function makes the trade-off precise. At a
large inverse temperature the update converges to a single stored pattern;
at a small one it converges to an average of many, which is the role that
the bandwidth plays in :numref:`sec_queries-keys-values` and the factor
$1/\sqrt{d}$ in :numref:`sec_attention-scoring-functions`. They also show
that such a network can store a number of patterns exponential in the
dimension.

### Bandwidths That Follow the Data

The Nadaraya--Watson estimator of :numref:`sec_queries-keys-values` uses one
bandwidth everywhere, which is too narrow where the data are sparse and too
wide where they are dense. Adaptive kernel estimators let the bandwidth
follow the local density of the data :cite:`Silverman86`, a hand-built
analogue of a temperature that depends on the query. The same lookup also
works as a representation without any training: ASIF
:cite:`norelli2022asif` describes an image or a caption by its similarities
to a fixed set of image--caption anchor pairs, which lets two separately
trained encoders exchange information without a joint model.

### Monotone Alignment

Bahdanau's attention of :numref:`sec_attention-scoring-functions` was not
the first differentiable alignment. Graves's handwriting synthesis
:cite:`Graves.2013` aligned each pen stroke with a position in the text
through a soft window that could only move forward, a constraint inherited
from the left-to-right hidden Markov models of speech recognition
:cite:`rabiner1993fundamentals`. Dropping monotonicity is what lets
attention handle reordering between languages, as in the alignment of
"black cat" with "chat noir".

### Weights That Need Not Sum to One

The softmax forces every row of weights to sum to one, so a query that
finds nothing relevant must still put its weight somewhere. Trained models
often put it on the first token, the attention sink that
:numref:`sec_multihead-attention` and :numref:`sec_what-attention-computes`
mention and :numref:`subsec_kv-cache-sinks` measures. Several lines of work
ask why. :citet:`Sun.Chen.Kolter.ea.2024` find that sinks coincide with a
few activations in the residual stream that are orders of magnitude larger
than the rest and act as fixed biases. :citet:`Barbero.Arroyo.Gu.ea.2025`
argue that sinks are useful: they let a deep model with a long context avoid
mixing every token into every other. Two designs remove the need for them.
Sigmoid attention :cite:`Ramapuram.Danieli.Dhekane.ea.2025` replaces the
softmax by an elementwise sigmoid with a bias of $-\log n$, so that a query
can give little weight to every key. It trains stably once that bias is in
place and the attention output is kept small early in training, and its
fused kernel is about a sixth faster than FlashAttention-2 on an H100.
Gated attention :cite:`Qiu.Wang.Zheng.ea.2025`, which
:numref:`subsec_kv-cache-sinks` describes, keeps the softmax and
multiplies each head's output by a sigmoid gate computed from the layer
input.

## Heads

### Redundant Heads

:numref:`sec_multihead-attention` proved that one head cannot return two
values and that two heads can, and the trained heads of its experiment
nearly duplicated each other. Trained models at scale show the same
redundancy. :citet:`Michel.Levy.Neubig.2019` removed heads of trained
translation and BERT models at test time and found that a large fraction can
go with little loss, and that many layers work nearly as well with a single
head. :citet:`Voita.Talbot.Moiseev.ea.2019`
identified the few heads that carry the effect, among them heads that attend
to the previous token, to rare words, or along a syntactic relation, and
pruned most of the others with a learned gate at a small cost in quality.

### Sharing Keys and Values

:numref:`subsec_kv-head-sharing` quoted two results on grouped-query
attention that seem to disagree. :citet:`Ainslie.Lee-Thorp.Jong.ea.2023`
found eight key-value groups within a tenth of a point of full multi-head
attention, and DeepSeek-V2 :cite:`DeepSeek-AI.2024` reported a clear loss
for the same change. The two results come from different training. The
first converted a strong multi-head checkpoint and trained it for 5% of its
original budget. The second trained a 7B dense model from scratch on 1.33T
tokens. Uptraining starts from heads that have already learned distinct
weightings, while training from scratch must find them with fewer key-value
heads from the first step. Multi-head latent attention, DeepSeek-V2's answer,
is described in :numref:`sec_transformers-further-reading`.

### Subtracting Two Weightings

Differential attention :cite:`Ye.Dong.Xia.ea.2025` computes two softmax
weightings from two sets of queries and keys and subtracts a learned
multiple of the second from the first, so that weight that both place on
irrelevant tokens cancels. The authors report that it matches a standard
transformer with about two thirds of the parameters or training tokens. No
widely used open model has adopted it so far, so it is a candidate rather
than a standard.

## Positions

### Learned Relative Positions

Before RoPE, relative position entered the score as trained vectors, one
per clipped offset, added to the keys inside the dot product
:cite:`shaw2018self`. The Music Transformer :cite:`huang2018music` made this
affordable for long sequences by computing the relative logits for all
offsets at once and then shifting them into place, instead of storing a
vector for every pair of positions. RoPE in
:numref:`sec_positional-information` reaches scores that depend only on the
offset with no learned parameters at all.

### RoPE in Trained Models

Released models differ less in whether they use RoPE than in its base. Llama
2 used $10^4$, Llama 2 Long and Llama 3 use $5 \times 10^5$
:cite:`Xiong.Liu.Molybog.ea.2024`, and the global layers of Gemma 3 use
$10^6$ :cite:`Gemma.Team.2025`. A larger base slows every pair except the
fastest, and so moves more pairs into the slow group of
:numref:`subsec_pos-context-extension`.
:citet:`Barbero.Vitvitskyi.Perivolaropoulos.ea.2025` looked at which pairs a
trained model uses. Gemma 7B uses the fastest pairs for heads that attend by
position, such as previous-token heads, and the slowest pairs as nearly
position-free channels for content. These two roles are the trained
counterpart of the fast and slow pairs of
:numref:`sec_positional-information`. Removing the lowest frequencies
altogether, a variant they call p-RoPE, helped in their experiments.

### Per-Pair Rescaling

YaRN, in :numref:`subsec_pos-context-extension`, leaves the fast pairs alone,
interpolates the slow ones, and ramps between the two groups with a
hand-set rule. LongRoPE :cite:`Ding.Zhang.Zhang.ea.2024` replaces the rule by
an evolutionary search for one rescale factor per pair and leaves the first
tokens unscaled. With at most a thousand fine-tuning steps at training
lengths up to 256k it extends the context to 2048k tokens.

### Length Generalization of ALiBi and NoPE

:numref:`subsec_pos-experiment` found that ALiBi and NoPE keep their
perplexity beyond the training length. :citet:`Chi.Fan.Rudnicky.ea.2023`
explain the first. ALiBi's steepest slopes make most heads attend within a
window near the training length, so the extra tokens are effectively never
read; the exercise on where extrapolation fails measures this. NoPE has its
own limit. :citet:`Haviv.Ram.Press.ea.2022` showed that causal language
models without positional encoding are competitive in perplexity and learn
absolute position from the causal mask. :citet:`Wang.Ji.Wu.ea.2024` found
that their length generalization also breaks, because attention spreads
over too many tokens, and restored it with a softmax temperature per head.
Llama 4 :cite:`Meta.2025` applies the same fix at inference time to the
position-free global layers that it interleaves with RoPE layers.

### Perplexity and the Use of Long Contexts

A model whose perplexity stays flat at four times its training length need
not use the extra tokens. :citet:`Liu.Lin.Hewitt.ea.2024` placed the answer
to a question at different positions of a long context and found that
models retrieve it far better from the beginning or the end than from the
middle. How to train models that use long contexts, by continued
pretraining on long documents with a raised base and by schedules that grow
the length, belongs to the Language Models part of the book.

## Cost

### Approximating the Softmax Kernel

Before fused exact kernels, several methods approximated attention to reach
linear cost. Performer :cite:`Choromanski.Likhosherstov.Dohan.ea.2021`
replaces $\exp(\mathbf{q}^\top\mathbf{k})$ by an inner product of positive
random features whose expectation is the exponential, so that the
associativity of :numref:`sec_attention-at-scale` applies. Linformer
:cite:`Wang.Li.Khabsa.ea.2020` projects the keys and values along the
sequence axis to a fixed length, and Reformer
:cite:`Kitaev.Kaiser.Levskaya.2020` hashes similar queries and keys into
shared buckets and attends within each bucket. cosFormer
:cite:`Qin.Sun.Deng.ea.2022` uses ReLU features with a cosine reweighting
that favors nearby positions. None of them is exact, and in practice they
were displaced by exact attention in fused kernels, by windows, and by the
gated recurrences below. :citet:`Tay.Dehghani.Bahri.ea.2020` catalogue
dozens more.

### Global Tokens and Strided Patterns

The window of :numref:`subsec_attn-receptive-field` needs many layers before
distant tokens meet. Longformer :cite:`beltagy2020longformer` adds to the
window a few positions chosen for the task, such as a classification token,
that attend to every token and are attended by every token, so that the
whole document reaches them in one layer. The Sparse Transformer
:cite:`child2019generating` combines a local pattern with a strided one so
that two layers connect any pair of positions. Both patterns are fixed in
advance, like the window.

### Windows in Released Models

Mistral 7B :cite:`Jiang.Sablayrolles.Mensch.ea.2023` used a window of 4,096
tokens in all 32 layers, with a rolling cache that holds only the last $w$
keys and values. Its successor, Mistral-7B-Instruct-v0.2
:cite:`Mistral.AI.2023`, dropped the window for a full context of 32k tokens
and a RoPE base of $10^6$. Gemma 2 :cite:`Gemma.Team.2024` alternates windowed
and global layers one to one and reports almost unchanged perplexity when
its windows shrink from 4,096 to 1,024 tokens at inference. Gemma 3
:cite:`Gemma.Team.2025` uses five windowed layers of 1,024 tokens per global
layer, the ratio whose cache saving :numref:`sec_attention-at-scale`
computes, and finds little change in perplexity even at seven to one.

### Sparsity Learned from Content

A window decides by position which keys a query sees. Three recent designs
decide by content, and each is built to run as a fused kernel over blocks.
Natively sparse attention :cite:`Yuan.Gao.Dai.ea.2025` scores compressed
blocks of keys, keeps the highest-scoring blocks for each query, and adds a
sliding window, all trained end to end. MoBA :cite:`Lu.Jiang.Liu.ea.2025`
routes each query to the key blocks whose mean key it matches best, a
mixture-of-experts gate over blocks. DeepSeek Sparse Attention
:cite:`Liu.Mei.Lin.ea.2025` scores all past tokens with a small separate
indexer and runs attention over the top-scoring ones only.

### FlashAttention-3 and FlexAttention in Detail

:numref:`sec_attention-at-scale` describes FlashAttention-3
:cite:`Shah.Bikshandi.Zhang.ea.2024` as the same algorithm with a new
schedule. On H100 GPUs, some warps issue asynchronous tile loads while others
run the matrix products. Two groups of warps also take turns, so that the
exponentials of one group, which run on a much slower unit, overlap the
matrix product of the other. Its FP8 mode quantizes each block separately.
It first multiplies queries and keys by a random orthogonal matrix that
spreads outlying coordinates, which lowers the error of FP8 attention by a
factor of 2.6 against a single scale per tensor. FlexAttention
:cite:`Dong.Feng.Guessous.ea.2025` generates kernels for masks and score
modifications written as Python functions; on the variants that
FlashAttention-2 supports, its kernels run at 0.68 to 1.43 times the speed
of the handwritten one. Kernels for decoding, which split the cache across
thread blocks and read it from pages, are treated with the serving systems
of the Language Models part.

### Gated Recurrences and Hybrids

:numref:`sec_attention-at-scale` ended with the observation that plain
linear attention lost to softmax attention, and that two changes made it
competitive. Gated linear attention :cite:`Yang.Wang.Shen.ea.2024` adds a
decay that depends on the input, and Gated DeltaNet
:cite:`Yang.Kautz.Hatamizadeh.2024` replaces the additive write by a delta
rule that overwrites the value stored under a key; both train with
chunkwise kernels like the chunked form of the chapter. The models that use
them keep a few softmax layers for recall: Qwen3-Next
:cite:`Qwen.Team.2025` places one gated-attention layer after every three
Gated DeltaNet layers, Kimi Linear :cite:`Kimi.Team.2025b` mixes three of its
recurrent layers with one latent-attention layer, and MiniMax-01
:cite:`MiniMax.2025` uses one softmax layer per seven linear ones. Mamba-2
:cite:`Dao.Gu.2024` shows that a selective state space model with a scalar
decay computes the same output as a masked attention whose mask holds
products of the decays, so the chunked form with a decay is one of its
algorithms. :numref:`chap_modern_rnn` teaches these recurrences.

## Circuits in Large Models

### Patching and Automated Circuit Discovery

The ablation of :numref:`subsec_head-ablation` shows that a component is
needed, not what it carries or along which path. Activation patching copies
an activation from a run on a clean input into a run on a corrupted one and
measures how much of the clean behavior returns
:cite:`Meng.Bau.Andonian.ea.2022`. Path patching restricts the copy to one
path between components, which is how
:citet:`Wang.Variengien.Conmy.ea.2023` traced the circuit that GPT-2 small
uses to complete "When Mary and John went to the store, John gave a drink
to" with "Mary". ACDC :cite:`Conmy.MavorParker.Lynch.ea.2023` automates the
search over edges. Attribution patching :cite:`Nanda.2023` replaces each
patch by a first-order estimate from gradients, so that one backward pass
scores every component, and AtP* :cite:`Kramar.Lieberum.Shah.ea.2024`
corrects the cases where that estimate misses an important component.
:citet:`Zhang.Nanda.2024` and :citet:`Heimersheim.Nanda.2024` study how the
conclusions depend on the choice of corrupted input and on the value that
replaces an ablated component. The Language
Models part of the book treats these methods on large models.

### Superposition and Dictionary Features

The circuit algebra of :numref:`sec_what-attention-computes` assumes that
each head reads and writes identifiable directions of the residual stream.
In large models the features outnumber the dimensions and share directions,
a regime that :citet:`Elhage.Hume.Olsson.ea.2022` call superposition and
reproduce in small models. Sparse dictionaries learned on the activations
recover features one at a time, and attribution graphs built from them
trace which features cause which across layers
:cite:`Ameisen.Lindsey.Pearce.ea.2025`. These methods, too, belong to the
Language Models part.

### Induction Heads at Scale

:citet:`Olsson.Elhage.Nanda.ea.2022` found the induction heads of
:numref:`sec_what-attention-computes` in large models as well. Heads
identified on repeated random tokens also copy translated fragments and
support few-shot completion, and the authors give six lines of evidence that
induction heads carry most of in-context learning: causal in small
attention-only models, correlational in large ones.

### In-Context Learning as an Algorithm

:numref:`sec_what-attention-computes` showed that one head without a softmax
takes one gradient step on a regression problem stored in its context.
:citet:`Garg.Tsipras.Liang.ea.2022` trained transformers to regress in
context on whole function classes, such as linear functions, sparse linear
functions, and decision trees, and found errors close to those of the best
algorithm for each class. :citet:`Akyurek.Schuurmans.Andreas.ea.2023` asked
which algorithm a trained model implements and found a progression with
depth and width, from one gradient step towards ridge regression.

[Discussions](https://d2l.discourse.group/t/36852)
