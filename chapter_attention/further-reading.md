# Further Reading
:label:`sec_attention-further-reading`

The sections of this chapter teach one representative of each idea: one
soft lookup, one scoring function, self-attention on one sentence and one
way to split it into heads, a few ways to encode position, one exact kernel
and two ways to avoid the quadratic cost, and one circuit traced through a
small model. Each has relatives, and they are not equally important.

Three topics bear on models you are likely to build or fine-tune. Released
models differ in the base of their rotary encoding, and a trained model
gives its fast and slow pairs different jobs
(:numref:`subsec_attention-fr-rope`). Attention sinks are the tokens that a
cache must keep when it drops old ones, and several new designs aim to make
them unnecessary (:numref:`subsec_attention-fr-sinks`). And the linear
attention of :numref:`sec_windowed-linear-attention` returned, with gates
and a new write rule, inside several recent large models
(:numref:`subsec_attention-fr-gated`). The approximations of the softmax
kernel are history, kept for readers who meet them in older papers. The
rest is for the curious, or for the day a paper takes one of them for
granted. Resources for what the chapter itself teaches are listed in its
introduction, :numref:`chap_attention`. Nothing else in the book depends on
this section.

## The Lookup and Its Normalization

### Attention as Associative Memory

The lookup of :numref:`sec_queries-keys-values` has a bandwidth that
decides between retrieving one entry and averaging many.
:citet:`Ramsauer.Schafl.Lehner.ea.2021` read the same computation as memory
retrieval. Store some patterns as the rows of a matrix that serves as both
keys and values, attend to them with a query, feed the output back in as
the next query, and repeat: this is a modern Hopfield network, and one
softmax attention is one step of it. The inverse temperature of the softmax
plays the part of the reciprocal of the bandwidth. When it is large, the
iteration converges to a single stored pattern; when it is small, to an
average of many, which is the role of the bandwidth and of the factor
$1/\sqrt{d}$ in :numref:`sec_attention-scoring-functions`. The reading
gave a way to sort trained heads by how many tokens they average. In BERT,
the heads of the first layers mostly average over the whole sequence,
while heads of later layers average over a few tokens.

One bandwidth for every query is, of course, the simplest choice: too
narrow where the data are sparse and too wide where they are dense. Kernel
estimators can let the bandwidth follow the local density
:cite:`Silverman86`, either at each point where the estimate is evaluated,
as the nearest-neighbour estimate does, or at each data point, as the
adaptive kernel estimate does. In attention these become a temperature
that depends on the query or one that depends on the key. A temperature
per head is the coarse version, and the NoPE entry below meets it again.

### Monotone Alignment

A model that writes a text by hand must know, at every pen stroke, which
letter it is writing. Graves's handwriting synthesis :cite:`Graves.2013`
learned this alignment a year before attention reached translation, with a
soft window over the letters that can only move forward, like the
left-to-right hidden Markov models of speech recognition
:cite:`rabiner1993fundamentals`. Handwriting follows the text in order, so
the constraint costs nothing there. Translation does not: the toy aligner
of :numref:`subsec_learned-alignment` had to learn one swapped pair, the
kind of crossing that turns "black cat" into "chat noir". The alignment of
:citet:`Bahdanau.Cho.Bengio.2014` has no such constraint, and it is the one
that started the field.

### Weights That Need Not Sum to One
:label:`subsec_attention-fr-sinks`

The softmax forces every row of weights to sum to one, so a query that
finds nothing relevant must still put its weight somewhere. Under a causal
mask the first token is the one key that every query can see, and trained
models often park the weight there, the attention sink of
:numref:`subsec_cross-attention-alignment` and
:numref:`sec_what-attention-computes`. A cache that evicts old tokens must
therefore keep the sink, as :numref:`subsec_kv-cache-sinks` measures. Why
would a trained model want such a place at all? :citet:`Sun.Chen.Kolter.ea.2024` find that sinks
coincide with a few activations in the residual stream that are orders of
magnitude larger than the rest and act as fixed biases.
:citet:`Barbero.Arroyo.Gu.ea.2025` give a reason for having one: a head
that can idle on the sink keeps a deep model with a long context from
mixing every token into every other.

Several designs remove the need for a sink. Sigmoid attention
:cite:`Ramapuram.Danieli.Dhekane.ea.2025` replaces the softmax by an
elementwise sigmoid with a bias of $-\log n$, so that a query can give
little weight to every key. The bias starts each of the $n$ weights near
$1/(n+1)$, close to what a softmax gives at initialization; with it, and
with normalization that keeps the attention norms small early in training,
sigmoid attention matches softmax attention. Gated attention
:cite:`Qiu.Wang.Zheng.ea.2025` keeps the softmax and multiplies each head's
output by a sigmoid gate computed from the layer input. A head with nothing
to contribute can then close its gate instead of parking its weight on the
first token. Qwen3-Next uses it :cite:`Qwen.Team.2025`. gpt-oss takes a
third route :cite:`OpenAI.2025`: each head learns one extra score that
enters the normalizer of the softmax but has no value attached, so that
weight can go nowhere.

### Subtracting Two Weightings

What if the weight that a query wastes on irrelevant tokens could be
cancelled instead of parked? Differential attention
:cite:`Ye.Dong.Xia.ea.2025` computes two softmax weightings from two sets
of queries and keys and subtracts a learned multiple of the second from the
first. Like a differential amplifier, which subtracts two signals to remove
the noise they share, it removes the weight that both spread over
irrelevant tokens. The authors report that it matches a standard
transformer with about two thirds of the parameters or training tokens. No
widely used open model has adopted it so far, so it is a candidate rather
than a standard.

## Self-Attention and Heads

### Origins of Self-Attention

Attention began as a link between two sequences, a decoder reading an
encoder. Letting a sequence attend to itself made it a layer of its own.
:citet:`Cheng.Dong.Lapata.2016` added such *intra-attention* to an LSTM,
whose cell attends over its own earlier hidden states instead of relying
on the latest one. For deciding whether one sentence implies another,
:citet:`Parikh.Tackstrom.Das.ea.2016` dropped the recurrence altogether.
The words of each sentence attend to those of the other, the aligned pairs
are compared and summed, and an optional intra-sentence attention, in which
each sentence also attends to itself, improved the result. With about a
tenth of the parameters of the recurrent models before it, the model showed
that attention alone, without a recurrent encoder, could do the job.
:citet:`Lin.Feng.Santos.ea.2017` computed several weightings of one
sentence at once, the rows of a matrix, with a penalty that pushes the rows
apart so that they do not all pick the same words. Each row acts like a
head without projections of its own, and the penalty is an early remedy for
the redundancy of trained heads that the next entry describes. A few months
later the Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017` put the three
ideas, self-attention, no recurrence and several heads, into one layer.

### Redundant Heads

:numref:`sec_multihead-attention` proved that one head cannot return two
values and that two heads can; in its four-head layer the two extra heads
repeat the weightings of the other two or fall between them, adding nothing
the task needs. Models at scale show a similar redundancy. When
:citet:`Michel.Levy.Neubig.2019` removed heads of trained translation and
BERT models at test time, a large fraction could go with little loss.
:citet:`Voita.Talbot.Moiseev.ea.2019` identified the few heads that carry
the effect, among them heads that attend to a neighbouring token, to rare
words, or along a syntactic relation, and pruned most of the others at a
small cost in quality. Pruning heads after training never became a routine
step. Sharing keys and values among groups of heads did, as the
grouped-query attention of :numref:`subsec_kv-head-sharing`, because it
shrinks the cache of generation (:numref:`sec_kv-cache-reduction`).

## Positions

### Learned Relative Positions

Relative position entered the score as trained parameters long before
RoPE. In the first scheme :cite:`shaw2018self`, each offset gets a trained
vector that is added to the key inside the dot product, and all offsets
beyond a maximum share one vector. The Music Transformer
:cite:`huang2018music` made this affordable for long sequences: it computes
the relative logits for all offsets at once and shifts them into place, so
that it stores one vector per offset instead of one per pair of positions.
T5 :cite:`raffel2020exploring` reduced the trained vector to one scalar
per head, added to the score, for each bucket of offsets; the buckets hold
single offsets nearby and grow logarithmically with distance. ALiBi
replaces these learned scalars by a fixed linear penalty. RoPE makes
position enter the score only through the offset, with no learned
parameters at all.

### RoPE in Trained Models
:label:`subsec_attention-fr-rope`

Released models differ less in whether they use RoPE than in its base.
Llama 2 used $10^4$, Llama 2 Long :cite:`Xiong.Liu.Molybog.ea.2024` and
Llama 3 :cite:`Grattafiori.Dubey.Jauhri.ea.2024` use $5 \times 10^5$, and
the global layers of Gemma 3 use $10^6$ :cite:`Gemma.Team.2025`. A larger
base slows every pair except the fastest, and so moves more pairs into the
slow group of :numref:`subsec_pos-context-extension`, the pairs whose
period exceeds the training length and which never complete a turn in
training. :citet:`Barbero.Vitvitskyi.Perivolaropoulos.ea.2025` looked at
which pairs a trained model uses. Gemma 7B uses the fastest pairs for heads
that attend by position, such as previous-token heads, and the slowest
pairs as nearly position-free channels for content. These two roles are
the trained counterpart of the fast and slow pairs of
:numref:`sec_positional-information`. *Nearly* is the catch: over a long
enough distance even the slowest pair turns, and a channel meant for
content then drifts with position. Their variant p-RoPE removes the lowest
frequencies altogether, which makes those channels exactly position-free,
and gave a lower perplexity than full RoPE in their experiments.

### Per-Pair Rescaling

YaRN, in :numref:`subsec_pos-context-extension`, leaves the fast pairs
alone, interpolates the slow ones, and ramps between the two groups with a
hand-set rule. LongRoPE :cite:`Ding.Zhang.Zhang.ea.2024` replaces the rule
by an evolutionary search for one rescale factor per pair, leaves the first
tokens unscaled, and with a short fine-tuning extends a model's context to
2048k tokens, about two million.

### Length Generalization of ALiBi and NoPE

:numref:`sec_positional-information` found ALiBi's perplexity flat at four
times the training length and showed why: removing every key 128 or more
positions back leaves it unchanged. The curve is flat because the model
never reads the extra tokens. :citet:`Chi.Fan.Rudnicky.ea.2023` give the
same account through the receptive field of the heads. NoPE has a limit of
its own. :citet:`Haviv.Ram.Press.ea.2022` showed that causal language
models without positional encoding are competitive in perplexity and learn
absolute position from the causal mask. The length generalization of such
models breaks, according to :citet:`Wang.Ji.Wu.ea.2024`, because attention
spreads over too many tokens, and a softmax temperature per head restores
it. Llama 4 :cite:`Meta.2025` has global layers without positional
encoding, and its release scales the attention temperature at inference
time to reach longer contexts.

### Perplexity and the Use of Long Contexts

Even a model trained on long contexts may not use them. Perplexity averages
over all tokens, most of which need only the last few hundred, so it says
little about whether a model can find one fact far back.
:citet:`Liu.Lin.Hewitt.ea.2024` placed the answer to a question at
different positions of a long context and found that models retrieve it
far better from the beginning or the end than from the middle. Training
models that do use long contexts is the subject of
:numref:`sec_lm_long_context`.

## Cost

### Older Routes to Linear Cost

Before fused kernels made exact attention fast at the lengths that
mattered, several methods approximated it to reach linear cost. Performer
:cite:`Choromanski.Likhosherstov.Dohan.ea.2021` replaces
$\exp(\mathbf{q}^\top\mathbf{k})$ by an inner product of positive random
features whose expectation is the exponential, so that the associativity
of :numref:`sec_windowed-linear-attention` applies. Linformer
:cite:`Wang.Li.Khabsa.ea.2020` projects the keys and values along the
sequence axis to a fixed length, and Reformer
:cite:`Kitaev.Kaiser.Levskaya.2020` hashes similar queries and keys into
shared buckets. None of them is exact, all of them gave up quality on
language, and the fused kernels took away their reason to exist.
:citet:`Tay.Dehghani.Bahri.ea.2020` catalogue dozens more.

Fixed sparse patterns kept the exact softmax but restricted which keys each
query sees. Longformer :cite:`beltagy2020longformer` adds to a window a few
global positions, such as a classification token, that attend to every
token and are attended by every token. The Sparse Transformer
:cite:`child2019generating` combines a local pattern with a strided one so
that two layers connect any pair of positions. These patterns survived in a
simpler form, as whole layers that are either windowed or global.

### Windows in Released Models

A window caps the cache of a layer at $w$ entries, which is why released
models use windows at all; the question is how many layers can afford one.
Mistral 7B :cite:`Jiang.Sablayrolles.Mensch.ea.2023` answered "all of
them": it used a window in every layer and kept only the last $w$ keys and
values in a rolling cache. Its successor :cite:`Mistral.AI.2023` dropped
the window for a full context of 32k tokens. Later models mix the two
kinds of layer. Gemma 2 :cite:`Gemma.Team.2024` alternates windowed and
global layers one to one and loses almost no perplexity when its windows
shrink from 4,096 to 1,024 tokens. Gemma 3 :cite:`Gemma.Team.2025` uses
five windowed layers per global layer and finds little change even at
seven to one. gpt-oss :cite:`OpenAI.2025` alternates global layers with
windows of only 128 tokens. Llama 4 :cite:`Meta.2025` keeps RoPE only in
the layers that its released configuration restricts to fixed chunks of
the sequence; unlike a sliding window, a chunk gives a token at its start
nothing to look back at. Apart from the first Mistral, then, every model
here that uses windows interleaves them with global layers, and the windows
have narrowed from thousands of tokens to as few as 128.

### Sparsity Learned from Content

A window decides by position which keys a query sees. Three recent designs
decide by content, each built to run fast on GPUs. Natively sparse
attention :cite:`Yuan.Gao.Dai.ea.2025` scores compressed blocks of keys,
keeps the highest-scoring blocks for each query, and adds a sliding
window, all trained end to end. MoBA :cite:`Lu.Jiang.Liu.ea.2025` routes
each query to the key blocks whose mean key it matches best, a
mixture-of-experts gate over blocks. DeepSeek Sparse Attention
:cite:`Liu.Mei.Lin.ea.2025` scores all past tokens with a small separate
indexer and attends to the top-scoring ones only. None of the three removes
the quadratic term: the selector still compares each query with every block
or token, only far more cheaply than full attention would. DeepSeek Sparse
Attention is the one in a released open model, DeepSeek-V3.2, and MoBA
serves long requests in Kimi by its authors' account.
:numref:`sec_lm_sparse_attention` treats these designs.

### Later FlashAttention Versions and FlexAttention

The FlashAttention of :numref:`sec_attention-at-scale` was followed by
versions that keep the algorithm and change its schedule. FlashAttention-2
:cite:`Dao.2024` divides by the normalizer only once at the end, as
`chunked_attention` does, and runs blocks of queries in parallel, so that a
single long sequence still keeps the whole GPU busy. On H100 GPUs,
FlashAttention-3 :cite:`Shah.Bikshandi.Zhang.ea.2024` lets one group of
threads compute exponentials, on a unit much slower than the matrix units,
while another group runs matrix products. Its 8-bit mode first multiplies
queries and keys by a random orthogonal matrix, which leaves every score
unchanged but spreads outlying coordinates, so that they lose less to
quantization. In 16-bit precision every version computes exact attention;
the 8-bit mode does not. FlexAttention :cite:`Dong.Feng.Guessous.ea.2025`
takes the mask and any change of the scores as Python functions, such as
the ALiBi bias of :numref:`sec_positional-information` or a cap on the
scores, and compiles them into a fused kernel roughly as fast as
FlashAttention-2. Kernels for decoding split a long cache into pieces that
are processed in parallel and merge the partial results with the online
softmax of :numref:`sec_attention-at-scale`;
:numref:`sec_lm_decode_kernels` treats them.

### Gated Recurrences and Hybrids
:label:`subsec_attention-fr-gated`

Plain linear attention models language less well than softmax attention,
and :numref:`sec_windowed-linear-attention` named two changes that brought
it close: a decay computed from each input, so that the state can forget,
and a few softmax layers kept for recall. Gated linear attention
:cite:`Yang.Wang.Shen.ea.2024` is the standard form of the first. A third
change concerns the write itself. The additive write of linear attention
piles a new value on top of whatever the state already returns for a
similar key, so writing one key twice blurs the two values. DeltaNet
:cite:`Schlag.Irie.Schmidhuber.2021,Yang.Wang.Zhang.ea.2024` writes the
difference between the new value and what the state returns for the key,
which moves the stored value towards the new one, all the way when its
step size is one. Gated DeltaNet :cite:`Yang.Kautz.Hatamizadeh.2024`
combines this delta rule with the decay. The large models built on linear
recurrences keep a few softmax layers for recall, one per three to seven
recurrent layers :cite:`Qwen.Team.2025,Kimi.Team.2025b,MiniMax.2025`.
:numref:`chap_modern_rnn` teaches these layers: the decaying state in
:numref:`sec_matrix-state`, the delta rule in :numref:`sec_deltanet`, and
the hybrids in :numref:`sec_hybrids`.

## Circuits in Large Models

### Patching and Features

The ablation of :numref:`subsec_head-ablation` shows that a component is
needed, not what it carries. Activation patching copies an activation from
a run on a clean input into a run on a corrupted one and measures how much
of the clean behavior returns :cite:`Meng.Bau.Andonian.ea.2022`. Restricted
to single paths between components, it let
:citet:`Wang.Variengien.Conmy.ea.2023` trace the circuit by which GPT-2
small completes "When Mary and John went to the store, John gave a drink
to" with "Mary", a circuit that contains induction heads. ACDC
:cite:`Conmy.MavorParker.Lynch.ea.2023` automates the search, and
attribution patching :cite:`Nanda.2023,Kramar.Lieberum.Shah.ea.2024`
estimates every patch from one backward pass. The answers depend on the
corrupted input and on the value that replaces a component
:cite:`Zhang.Nanda.2024,Heimersheim.Nanda.2024`, a caution that applies to
the zeroing of :numref:`subsec_head-ablation` too.

A second difficulty concerns what flows along the paths. The circuit
algebra of :numref:`sec_what-attention-computes` assumes that each head
reads and writes identifiable directions of the residual stream. In large
models the features appear to outnumber the dimensions and to share
directions, a hypothesis that :citet:`Elhage.Hume.Olsson.ea.2022` call
*superposition* and demonstrate in toy models. Sparse dictionaries learned
on the activations, *sparse autoencoders*, recover the features as
separate directions
:cite:`Bricken.Templeton.Batson.ea.2023,Cunningham.Ewart.Riggs.ea.2024`,
and attribution graphs, built from a related sparse replacement of the MLP
layers, trace which features cause which
:cite:`Ameisen.Lindsey.Pearce.ea.2025`. :numref:`sec_lm_interpretability`
applies patching and sparse autoencoders to a released language model.

### The 2019 Debate on Explanation

This chapter drew many attention maps, and
:numref:`sec_what-attention-computes` showed that a map says where a head
looks, not what it moves. A debate of 2019 asked the same question of
trained text classifiers. :citet:`Jain.Wallace.2019` found that attention
weights agreed poorly with gradient-based measures of which words
mattered, and that quite different weights, searched for one example at a
time, left the predictions unchanged. :citet:`Wiegreffe.Pinter.2019`
replied that a weighting chosen per example after training is not one the
model could have produced, and that where uniform weights do as well, the
model does not use attention at all. An exercise of
:numref:`sec_what-attention-computes` works through both sides on the
chapter's small model. What the exchange established is modest and
durable: attention weights are not an explanation by default, and whether
they explain a prediction has to be tested by intervening on the model.

### In-Context Learning as an Algorithm

:citet:`Olsson.Elhage.Nanda.ea.2022` argue that induction heads carry most
of the in-context learning of language models, on evidence that is causal
in small attention-only models and correlational in large ones. Copying is not
the only way to learn from a context, though. One head without a softmax can
take a gradient step on a regression problem stored in its context (Exercise
:exref:`ex_what-attention-computes-in-context-regression`), and
:citet:`Oswald.Niklasson.Randazzo.ea.2023` found that deeper linear-attention
transformers trained on such problems learn refinements of that step that
correct for the curvature of the problem. Transformers trained on whole
function classes match least squares on linear functions and match or beat
standard learners on sparse linear functions, small networks and decision trees
:cite:`Garg.Tsipras.Liang.ea.2022`. With depth and with the noise of the data,
the learned algorithm moves between gradient descent, ridge regression and
exact least squares :cite:`Akyurek.Schuurmans.Andreas.ea.2023`. So a
transformer trained for the purpose can run a learning algorithm in its forward
pass; whether pretraining on ordinary text teaches a language model to do the
same is not settled :cite:`Shen.Mishra.Khashabi.2024`.

[Discussions](https://d2l.discourse.group/t/36852)
