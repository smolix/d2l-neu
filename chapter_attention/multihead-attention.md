# Self-Attention and Multiple Heads
:label:`sec_multihead-attention`

Every attention layer we have built so far has had two sides: the queries
came from one place, the keys and values from another, as when the letters
of "attention" queried those of "translation". A language model has no
second sequence. All it has is the text it has read so far, so its queries
must come from that text too.

Consider the sentence "Alice gave Bob the book". To make sense of *gave*,
a reader needs three other words: who gave, who received, and what was
given. If every word emits a query, a key, and a value, then *gave* can
put its question to the other words of its own sentence and collect what
it needs. This arrangement is called
*self-attention*. The idea of a sequence attending to itself goes back to
:citet:`Cheng.Dong.Lapata.2016` and :citet:`Lin.Feng.Santos.ea.2017`. In
2017 the Transformer of :citet:`Vaswani.Shazeer.Parmar.ea.2017` gave it the
query--key--value form used today and showed that a stack built around such
layers, with attention as its only way of mixing positions, translated
better than the best recurrent and convolutional models of the day. Nearly
every language model since has been built the same way.

We start with the simplest case: one self-attention layer on this sentence,
with maps small enough to follow by hand. The worked example then raises
the question of the section. *Gave* needs three words, and the layer gives
it one set of weights and one blended vector. How can one query read
several positions without blending what it reads? A small task makes the
question precise, and its answer is to run several copies of the layer,
each with maps of its own, side by side.

```{.python .input #multihead-attention-multi-head-and-cross-attention}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #multihead-attention-multi-head-and-cross-attention}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Self-Attention

In self-attention every token plays three roles. It asks: as a *query*, it
looks for the tokens it needs. It is found: as a *key*, it advertises what
it offers to the queries of others. And it is read: as a *value*, it hands
over its content once it has been selected. One vector could play all three
roles, as the letter embeddings did in
:numref:`sec_attention-scoring-functions`, but it would play them badly,
because what a verb looks for is not what a verb offers. The layer
therefore computes three linear maps of each input.

Let the $n$ tokens of a sequence be the rows of $\mathbf{X} \in
\mathbb{R}^{n \times d}$. Learned matrices $\mathbf{W}_q, \mathbf{W}_k \in
\mathbb{R}^{d \times d_h}$ and $\mathbf{W}_v \in \mathbb{R}^{d \times d_v}$
give the queries $\mathbf{X}\mathbf{W}_q$, the keys $\mathbf{X}\mathbf{W}_k$,
and the values $\mathbf{X}\mathbf{W}_v$, and scaled dot-product attention
does the rest:

$$
\mathrm{softmax}\left(\frac{(\mathbf{X}\mathbf{W}_q)(\mathbf{X}\mathbf{W}_k)^\top}{\sqrt{d_h}}\right)\mathbf{X}\mathbf{W}_v \in \mathbb{R}^{n \times d_v}.
$$
:eqlabel:`eq_self-attention`

Row $i$ of the result is the new representation of token $i$: an average
of the values of all $n$ tokens, weighted by how well the query of token $i$
matches each key. Nothing in the computation is new. It is the attention of
:numref:`sec_attention-scoring-functions`, with the queries, keys, and
values all drawn from the same sequence. Three maps and one softmax of this
kind make up one attention *head*, and $d_h$ is its width.

### One Head on One Sentence

Learned maps are hard to read, so let us set the maps by hand and check
every number. Each token's input has two parts: a one-hot code of its role
(person, action, thing, or function word), and a one-hot code of the word
itself, nine numbers in all. The query and key maps read only the role, and
they map it to $d_h = 2$ numbers:

| role | query | key |
|:--|:--|:--|
| person (Alice, Bob) | $(1, -1)$ | $(1, 1)$ |
| action (*gave*) | $(1, 1)$ | $(1, -1)$ |
| thing (book) | $(1, -1)$ | $(1, 1)$ |
| function word ("the") | $(0, 0)$ | $(0, 0)$ |

The query of the verb points along the keys of the nouns, the queries of
the nouns point along the key of the verb, and "the" neither asks nor is
found. People and things get the same maps because this head treats every
noun alike; telling them apart would take a second head. The value map
reads only the word part and copies it. Each value therefore marks its
word, and the output of a token can be read as amounts of each word.

Real embeddings, of course, come with no role code. A trained layer has to
find directions of this kind among hundreds of coordinates, and
:numref:`sec_what-attention-computes` looks at what trained heads actually
find. Here we hand the layer its roles so that nothing stands between us
and the arithmetic.

```{.python .input #multihead-attention-self-attention}
%%tab pytorch, jax
tokens = ['Alice', 'gave', 'Bob', 'the', 'book']
# Role of each token: person, action, thing, function word
roles = d2l.tensor([[1.0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
                    [0, 0, 0, 1], [0, 0, 1, 0]])
X = d2l.concat((roles, d2l.eye(5)), 1)  # (5, 9): the role, then the word
# Queries and keys read the role part (rows: person, action, thing,
# function word), values the word part
W_q = d2l.concat((d2l.tensor([[1.0, -1], [1, 1], [1, -1], [0, 0]]),
                  d2l.zeros((5, 2))))
W_k = d2l.concat((d2l.tensor([[1.0, 1], [1, -1], [1, 1], [0, 0]]),
                  d2l.zeros((5, 2))))
W_v = d2l.concat((d2l.zeros((4, 5)), d2l.eye(5)))
Q, K, V = X @ W_q, X @ W_k, X @ W_v  # (5, 2), (5, 2), (5, 5)
```

Before running the layer, we can predict the row of *gave*. Its query
$(1, 1)$ has dot product 2 with the key $(1, 1)$ of each noun and 0 with
the keys of *gave* and "the". Divided by $\sqrt{2}$, the scores are $1.41$
and $0$. The softmax gives each noun the weight $e^{1.41}/(3e^{1.41} + 2)
\approx 0.29$ and each of the other two tokens $1/(3e^{1.41} + 2) \approx
0.07$. :numref:`fig_self-attention` lays out the computation, from the
query of *gave* to its output.

![One head of self-attention on "Alice gave Bob the book", with the hand-set maps of the code above ($d_h = 2$). Every token emits a query, a key, and a value. The query of *gave*, $(1, 1)$, is scored against every key by $\mathbf{q} \cdot \mathbf{k}/\sqrt{2}$; the softmax turns the scores $1.41$ and $0$ into weights $0.29$ and $0.07$; the output of *gave* is the weighted sum of the values, which mark the five words.](../img/mdl-attention-self-attention.svg)
:label:`fig_self-attention`

The cell runs `d2l.DotProductAttention` on these queries, keys, and
values. It prints the scores and weights of *gave*, the weights of "the",
and the output of *gave*, then the weights of *gave* under a causal mask,
and it draws all 25 weights.

```{.python .input #multihead-attention-self-attention-1}
%%tab pytorch, jax
attention = d2l.DotProductAttention()
Y = attention(Q[None], K[None], V[None])  # A minibatch of one sentence
weights = attention.attention_weights[0]
scores = Q[1] @ K.T / math.sqrt(Q.shape[-1])  # Scaled by sqrt(d_h)
print('scores of gave: ', np.asarray(scores).round(2))
print('weights of gave:', np.asarray(weights[1]).round(2))
print('weights of the: ', np.asarray(weights[3]).round(2))
print('output of gave: ', np.asarray(Y[0, 1]).round(2))
attention(Q[None], K[None], V[None], d2l.causal_mask(5))
print('causal weights of gave:',
      np.asarray(attention.attention_weights[0, 1]).round(2))
d2l.show_heatmaps(weights[None, None], xlabel='keys', ylabel='queries',
                  figsize=(3.2, 3.2), cmap='Blues',
                  xticklabels=tokens, yticklabels=tokens)
```

The printed row matches the prediction. Since each value marks its word,
the output of *gave* is that same row: equal parts Alice, Bob, and book,
with a little of *gave* and "the". The query of "the" is zero, so all its
scores are equal and its weight spreads evenly, $0.2$ on every token. A
query that asks for nothing receives the average of everything. The third
kind of row belongs to the nouns. A noun's query scores $1.41$ on the verb
and $0$ elsewhere, so the verb gets $e^{1.41}/(e^{1.41} + 4) \approx 0.51$
and every other token about $0.12$: the dark cells of the heatmap.

A language model may not read ahead. Under the causal mask of
:numref:`sec_attention-scoring-functions`, each token sees only itself and
the tokens before it, so *gave* is left with Alice and itself. The scores
do not change. The softmax renormalizes the two that remain, $1.41$ and
$0$, into the weights $0.80$ and $0.20$ that the cell prints.

### Separate Maps for Queries and Keys

Why does the layer need two maps where one might seem to do? Suppose a
single map $\mathbf{W}$ produced both the queries and the keys, so that
token $i$ had one vector $\mathbf{u}_i = \mathbf{x}_i\mathbf{W}$ for both
roles. The scores $s_{ij} = \mathbf{u}_i \cdot \mathbf{u}_j/\sqrt{d_h}$
would then be symmetric, with token $i$ scoring token $j$ exactly as $j$
scores $i$, and by the Cauchy--Schwarz inequality $s_{ij} \le
\sqrt{s_{ii}\, s_{jj}}$. A token could seek another only as strongly as the
two, on geometric average, seek themselves. Our sentence shows what that
costs. Had we used the queries of our table as keys too, the query $(1, 1)$
of the verb would have dot product 2 with itself and 0 with every noun: the
verb would look for verbs. Separate maps let what a token asks for differ
from what it offers.

The value map splits the work once more. In our example the roles decide
who is read, and the word identities decide what is handed over. The output
has one row per token, the same shape as the input. A second
self-attention layer can therefore read the output of the first, and a
third the output of the second. The transformers of the next chapter are
such a stack, with other layers in between.

## Limits of a Single Head
:label:`subsec_one-mixture-bottleneck`

Let us return to the row of *gave*. It is one set of weights, and the
output is one weighted sum. In our example the values were indicators, one
coordinate per word, so the sum still records which words were read. Real
values are dense vectors of a fixed width. A weighted sum of three of them
is a single vector of that width, from which the three cannot in general be
recovered. Is this a limit of one head, or only of our hand-set weights? To
find out we need a task whose answer we know and whose best single-head
solution we can compute.

Here is one. Two source positions hold value vectors $\mathbf{v}_1,
\mathbf{v}_2 \in \mathbb{R}^{d_v}$, drawn independently from a standard
Gaussian for every example. Their keys mark only the positions, not the
content. A single query must report both values: the target is the
concatenation $\mathbf{t} = [\mathbf{v}_1; \mathbf{v}_2] \in
\mathbb{R}^{2d_v}$. We call this the *copy-both task*. It is artificial on
purpose: *gave* asking for two words at once, with everything stripped
away except the one thing a single head might fail at.

One head can produce only weights $(\alpha_1, \alpha_2)$ with $\alpha_1 +
\alpha_2 = 1$, the mixture $\mathbf{m} = \alpha_1 \mathbf{v}_1 + \alpha_2
\mathbf{v}_2$, and then a linear readout of $\mathbf{m}$. Since the keys
mark positions only, the weights cannot depend on the values. Whatever the
head learns, it applies the *same* weights to every example. How well can
it do?

**Proposition.** Consider one attention head with two fixed, position-only
keys, independent standard Gaussian values in $\mathbb{R}^{d_v}$, no
residual path, and a linear readout. On the copy-both task, the best
achievable relative squared error is $1/2$ for every softmax pair
$(\alpha_1,\alpha_2)$.

**Proof.** Given the mixture $\mathbf{m} = \alpha_1 \mathbf{v}_1 + \alpha_2
\mathbf{v}_2$, the best estimate of $\mathbf{v}_1$ is the conditional mean
$\mathbb{E}[\mathbf{v}_1 \mid \mathbf{m}] = \tfrac{\alpha_1}{\alpha_1^2 +
\alpha_2^2}\, \mathbf{m}$, which is linear, so a linear readout attains it.
Its error variance per coordinate is $1 - \tfrac{\alpha_1^2}{\alpha_1^2 +
\alpha_2^2} = \tfrac{\alpha_2^2}{\alpha_1^2 + \alpha_2^2}$, and the error for
$\mathbf{v}_2$ is $\tfrac{\alpha_1^2}{\alpha_1^2 + \alpha_2^2}$. The two
errors sum to $1$ per coordinate, while the target has total variance $2$ per
coordinate. The relative squared error is therefore $1/2$, however the head
splits its attention.
$\blacksquare$

The weights move the error between $\mathbf{v}_1$ and $\mathbf{v}_2$; they
cannot remove it. Counting shows why: the readout receives one mixture,
$d_v$ numbers, and must produce $2d_v$. Call one set of weights for one
query a *weighting*. A layer that solves the task needs several different
weightings for the same query, and it needs room in its output for every
value they retrieve. Neither suffices alone. A single head as wide as the
whole layer has room but one weighting, and when heads come to share their
values we will meet the opposite failure, several weightings and too little
room. Exercise :exref:`ex_multihead-what-the-bound-counts` turns the
counting into a proof.

We check the bound, and what a second head changes, by fitting the best
linear readout by least squares, with values of width $d_v = 8$. One head
supplies one mixture, here with weights $(0.7, 0.3)$. Two hand-built heads
supply two mixtures, whose outputs are concatenated before the readout. The
code reports the norm of the readout's error divided by the norm of the
target. This relative error is the square root of the relative squared
error, so the bound reads $\sqrt{1/2} \approx 0.707$.

```{.python .input #multihead-attention-one-head-must-average-1}
%%tab pytorch
N, d_v = 10000, 8
V1, V2 = torch.randn(N, d_v), torch.randn(N, d_v)
target = torch.cat([V1, V2], dim=1)

def readout_error(M, target):
    """Best linear readout of target from M, relative error."""
    W = torch.linalg.lstsq(M, target).solution
    return ((M @ W - target).norm() / target.norm()).item()

M = 0.7 * V1 + 0.3 * V2
print(f'one head, weights (0.70, 0.30): relative error '
      f'{readout_error(M, target):.3f}; sqrt(1/2) = {0.5 ** 0.5:.3f}')
for a in (1.0, 0.88):
    M2 = torch.cat([a * V1 + (1 - a) * V2, (1 - a) * V1 + a * V2], dim=1)
    print(f'two heads, weights ({a:.2f}, {1 - a:.2f}) and '
          f'({1 - a:.2f}, {a:.2f}): '
          f'relative error {readout_error(M2, target):.1e}')
```

```{.python .input #multihead-attention-one-head-must-average-1}
%%tab jax
N, d_v = 10000, 8
key1, key2 = jax.random.split(d2l.get_key())
V1, V2 = jax.random.normal(key1, (N, d_v)), jax.random.normal(key2, (N, d_v))
target = jnp.concatenate([V1, V2], axis=1)

def readout_error(M, target):
    """Best linear readout of target from M, relative error."""
    # Full fp32 products: TF32, the default on recent NVIDIA cards, would
    # stop the two-head errors near 1e-4
    with jax.default_matmul_precision('highest'):
        W = jnp.linalg.lstsq(M, target)[0]
        err = jnp.linalg.norm(M @ W - target) / jnp.linalg.norm(target)
    return float(err)

M = 0.7 * V1 + 0.3 * V2
print(f'one head, weights (0.70, 0.30): relative error '
      f'{readout_error(M, target):.3f}; sqrt(1/2) = {0.5 ** 0.5:.3f}')
for a in (1.0, 0.88):
    M2 = jnp.concatenate([a * V1 + (1 - a) * V2, (1 - a) * V1 + a * V2],
                         axis=1)
    print(f'two heads, weights ({a:.2f}, {1 - a:.2f}) and '
          f'({1 - a:.2f}, {a:.2f}): '
          f'relative error {readout_error(M2, target):.1e}')
```

The single head lands on the bound, at about $0.71$. Two heads bring the
error down to the rounding error of the least-squares solver, and the
second pair does so with *soft* weights $(0.88, 0.12)$. The heads need
different weightings, not one-hot ones. They supply two mixtures, the
readout inverts the $2 \times 2$ mixing matrix, and both values come back.
The inversion is well conditioned as long as the two weightings differ
appreciably; Exercise :exref:`ex_multihead-nearly-identical-heads` follows
what happens as they approach each other. :numref:`fig_one-head-averages`
summarizes the two cases.

![One weighting per query against two. A single head serves the copy-both task with one softmax weighting, so its output blends the two requested values. Two heads with different weightings supply two mixtures, from which a linear readout recovers both values.](../img/mdl-attention-one-head-averages.svg)
:label:`fig_one-head-averages`

## Multi-Head Attention

### Projected Heads

Rather than build the heads by hand, we learn them. A *multi-head* layer
runs $H$ heads side by side, each with its own three maps, and combines
their outputs with one more learned matrix
:cite:`Vaswani.Shazeer.Parmar.ea.2017`. Head $i$ computes

$$
\mathbf{Z}_i = \mathrm{softmax}\left(\frac{(\mathbf{X}\mathbf{W}_i^{(q)})(\mathbf{X}\mathbf{W}_i^{(k)})^\top}{\sqrt{d_h}}\right)\mathbf{X}\mathbf{W}_i^{(v)} \in \mathbb{R}^{n \times d_h}, \qquad i = 1, \ldots, H,
$$
:eqlabel:`eq_multihead-head`

with $\mathbf{W}_i^{(q)}, \mathbf{W}_i^{(k)}, \mathbf{W}_i^{(v)} \in
\mathbb{R}^{d \times d_h}$. This is :eqref:`eq_self-attention` with a head
index, and with values of the same width $d_h$ as queries and keys. Each
head scores with its own projected queries and keys, so it weights the
tokens by its own notion of relevance, and it moves its own projection of
the values. These are the separate mixtures of the construction above, now
learned. The only other new part is the output, a learned linear
recombination of the heads placed side by side:

$$
[\mathbf{Z}_1, \ldots, \mathbf{Z}_H]\, \mathbf{W}_o \in \mathbb{R}^{n \times d}, \qquad \mathbf{W}_o \in \mathbb{R}^{H d_h \times d}.
$$
:eqlabel:`eq_multihead-output`

The output projection $\mathbf{W}_o$ is the linear readout of the
proposition, now part of the layer. As before, $d$ is the width of the
model, the width of every token's vector between layers, and $d_h$ is the
width of one head. The output has width $d$ whatever the number of heads,
so a layer of width $d$ returns at most $d$ numbers per query. On the
copy-both task it can return both values only if $2d_v \le d$.
:numref:`fig_multi-head-attention` shows the layout.

![Multi-head attention. Every head reads the same queries, keys, and values, projects them into its own subspace, and attends there; the head outputs are concatenated and recombined by one output projection.](../img/mdl-attention-multi-head.svg)
:label:`fig_multi-head-attention`

### Parameter and FLOP Counts

The standard choice of head width makes the heads together no more
expensive than one wide head. Set $d_h = d/H$, so that the $H$ heads
produce $H d_h = d$ numbers per token, as many as a single head of width
$d$. The $H$ query matrices $\mathbf{W}_i^{(q)}$, placed side by side, then
form one $d \times d$ matrix, and so do the key, value, and output
matrices. For self-attention on $n$ tokens, counting one multiply--add as
two FLOPs, the layer costs

$$
\underbrace{8nd^2}_{\textrm{projections}} \;+\; \underbrace{4n^2 d}_{\textrm{scores and mixing}} \quad \textrm{FLOPs,} \qquad 4d^2 \ \textrm{parameters.}
$$
:eqlabel:`eq_multihead-flops`

For one layer of GPT-2 small, with $d = 768$ and its full context of
$n = 1{,}024$ tokens, that comes to 4.8 GFLOP for the projections, 3.2 GFLOP
for the scores and mixing, and 2.4 million parameters. Neither count
depends on $H$. The four projection matrices are $d \times d$ however their
outputs are sliced into heads, and each head's score and mixing products
shrink by the same factor $H$ by which their number grows. The number of
heads changes the representation, not the leading operation count.
Running time is not quite constant, though. With $H$ heads come $H n^2$
attention weights, each exponentiated, normalized, and stored, a cost the
count leaves out, and the heads' matrix products are smaller and run less
efficiently (Exercise :exref:`ex_multihead-narrow-heads`d). What does
change with $H$ is the attention map: each query now has $H$ sets of
weights over the $n$ keys instead of one.

If heads cost so little, why not take $H = d$, one head per coordinate?
Because a narrow head cannot point. With $d_h = 1$ every query and key is a
single number, the score $q k_j$ rises or falls monotonically with $k_j$,
and a query can concentrate its weight only on the largest key or on the
smallest (Exercise :exref:`ex_multihead-narrow-heads`). With two
coordinates the keys can sit on a circle, and then every query can single
out a key of its own choosing. Practice settles far above that. GPT-2 small
runs 12 heads of width 64, and its larger siblings keep the width at 64 and
add heads.

### Comparison with Convolution and Recurrence

Self-attention is the third layer in this book that maps a sequence of $n$
vectors to another sequence of $n$ vectors. The other two are the
one-dimensional convolution of :numref:`chap_cnn` and the recurrent layer
of :numref:`sec_rnn`. :numref:`fig_cnn-rnn-self-attention` draws the three
as graphs over five tokens. Two properties of these graphs matter beyond
the amount of arithmetic. The number of *sequential steps* says how much of
the work must wait for earlier work, and so how much of it a GPU can do at
once. The *path length* between two positions counts how many layer
applications a signal passes through on its way from one to the other.

![Sequence-layer connectivity for five tokens. Highlighted edges show how the inputs reach the third output in one layer: a convolution with kernel size $k = 3$ sees its neighbors, a recurrence passes information step by step along the chain, and self-attention connects every position directly. Below each panel are the path length between distant tokens and whether the outputs are computed in parallel.](../img/mdl-attention-cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`

A convolution with kernel size $k$ computes all positions at once, but each
layer widens its view by only $k - 1$ positions, so two tokens $n$ apart
are connected only through a stack of about $n/k$ layers. A recurrent layer
takes $n$ steps one after the other, and a signal from the first token
reaches the last through $n$ applications of the cell. In training, its
gradient is multiplied by $n$ Jacobians on the way back and tends to vanish
or explode (:numref:`sec_bptt`). Self-attention computes all outputs at
once and connects every pair of positions directly.
:numref:`tab_cnn-rnn-attn` collects the counts for layers of width $d$,
with the four projections of self-attention in its first term.

:Per-layer cost of mapping $n$ tokens of dimension $d$ to $n$ tokens of dimension $d$.
:label:`tab_cnn-rnn-attn`

| layer type | complexity | sequential operations | maximum path length |
|:--|:--|:--|:--|
| convolution (kernel $k$) | $O(knd^2)$ | $O(1)$ | $O(n/k)$ |
| recurrence | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| self-attention | $O(nd^2+n^2d)$ | $O(1)$ | $O(1)$ |

Self-attention wins the last two columns. The first is where it pays: on
top of the projections comes the term $n^2 d$, one score for every pair of
tokens. By :eqref:`eq_multihead-flops` that term overtakes the projections once
$4n^2 d > 8nd^2$, that is, once the sequence is longer than twice the
width. For GPT-2 small that happens at 1,536 tokens, just beyond the 1,024
it reads. A short path, moreover, makes a long-range dependency reachable,
not learned. Whether training finds it is a separate question.

### Implementation

The implementation rests on one idea: a head is just one more batch axis.
Because $d_h = d/H$, we can compute one query projection of width $d$ and
reshape its output into $H$ heads of width $d_h$. Queries, keys, and values
then become tensors of shape (batch, $H$, length, $d_h$). The scaled
dot-product attention of :numref:`sec_attention-scoring-functions` works
over any leading axes, so it runs all heads of all sequences in one batched
matrix product, with no loop over heads. A mask of shape (batch, queries,
keys) gets a head axis of size one and so applies to every head. The class
takes queries, keys, and values as three arguments, each with its own
projection; self-attention simply passes the same sequence three times.

The projections have no biases by default. A bias on the keys adds the same
amount to every score of a query, which the softmax ignores, and biases on
the values and on the output only shift the output by a constant. A bias on
the queries does change the weights, and dropping it is a convention rather
than a necessity: Llama and many later language models leave all four
biases out. The argument `num_kv_heads` lets groups of heads share keys and
values. It is the subject of :numref:`subsec_kv-head-sharing`, and until
then it equals the number of heads.

:begin_tab:`jax`
`Linear` is `nnx.Linear` with weights and biases drawn uniformly from
$(-1/\sqrt{n_\textrm{in}}, 1/\sqrt{n_\textrm{in}})$ for input width
$n_\textrm{in}$; the attention and transformer models of this and the next
chapter use it throughout.
:end_tab:

```{.python .input #multihead-attention-linear-init}
%%tab jax
class Linear(nnx.Linear):  #@save
    """nnx.Linear with weights and biases drawn from U(-a, a), where
    a = 1/sqrt(in_features)."""
    def __init__(self, in_features, out_features, *, use_bias=True, rngs,
                 **kwargs):
        a = in_features ** -0.5
        init = lambda key, shape, dtype=jnp.float32: jax.random.uniform(
            key, shape, dtype, -a, a)
        super().__init__(in_features, out_features, use_bias=use_bias,
                         kernel_init=init, bias_init=init, rngs=rngs,
                         **kwargs)
```

```{.python .input #multihead-attention-implementation-1}
%%tab pytorch
class MultiHeadAttention(nn.Module):  #@save
    """Multi-head attention with heads as an explicit axis; num_kv_heads <
    num_heads shares each key-value head across a group of query heads."""
    def __init__(self, num_hiddens, num_heads, dropout=0.0, bias=False,
                 num_kv_heads=None):
        super().__init__()
        assert num_hiddens % num_heads == 0, 'heads must divide num_hiddens'
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0
        kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, kv_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, kv_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, mask=None):
        # queries: (batch, no. of queries, num_hiddens); keys and values:
        # (batch, no. of key-value pairs, num_hiddens). mask: boolean,
        # broadcastable to (batch, no. of queries, no. of keys), shared by
        # all heads, or to (batch, num_heads, no. of queries, no. of keys)
        q = self.split_heads(self.W_q(queries), self.num_heads)
        k = self.split_heads(self.W_k(keys), self.num_kv_heads)
        v = self.split_heads(self.W_v(values), self.num_kv_heads)
        group = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        if mask is not None and mask.dim() == 3:
            mask = mask[:, None]           # The same mask for every head
        # Shape of Y: (batch, num_heads, no. of queries, head width)
        Y = self.attention(q, k, v, mask)
        return self.W_o(self.merge_heads(Y))

    @property
    def attention_weights(self):
        """Weights of the last call: (batch, num_heads, queries, keys)."""
        return self.attention.attention_weights
```

```{.python .input #multihead-attention-implementation-1}
%%tab jax
class MultiHeadAttention(nnx.Module):  #@save
    """Multi-head attention with heads as an explicit axis; num_kv_heads <
    num_heads shares each key-value head across a group of query heads."""
    def __init__(self, num_hiddens, num_heads, dropout=0.0, bias=False,
                 num_kv_heads=None, *, rngs):
        assert num_hiddens % num_heads == 0, 'heads must divide num_hiddens'
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0
        kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
        self.attention = d2l.DotProductAttention(dropout, rngs=rngs)
        self.W_q = Linear(num_hiddens, num_hiddens, use_bias=bias, rngs=rngs)
        self.W_k = Linear(num_hiddens, kv_hiddens, use_bias=bias, rngs=rngs)
        self.W_v = Linear(num_hiddens, kv_hiddens, use_bias=bias, rngs=rngs)
        self.W_o = Linear(num_hiddens, num_hiddens, use_bias=bias, rngs=rngs)

    def __call__(self, queries, keys, values, mask=None):
        # queries: (batch, no. of queries, num_hiddens); keys and values:
        # (batch, no. of key-value pairs, num_hiddens). mask: boolean,
        # broadcastable to (batch, no. of queries, no. of keys), shared by
        # all heads, or to (batch, num_heads, no. of queries, no. of keys)
        q = self.split_heads(self.W_q(queries), self.num_heads)
        k = self.split_heads(self.W_k(keys), self.num_kv_heads)
        v = self.split_heads(self.W_v(values), self.num_kv_heads)
        group = self.num_heads // self.num_kv_heads
        k, v = jnp.repeat(k, group, axis=1), jnp.repeat(v, group, axis=1)
        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]           # The same mask for every head
        # Shape of Y: (batch, num_heads, no. of queries, head width)
        Y = self.attention(q, k, v, mask)
        return self.W_o(self.merge_heads(Y))

    @property
    def attention_weights(self):
        """Weights of the last call: (batch, num_heads, queries, keys)."""
        return self.attention.attention_weights
```

Two reshapes do the work of splitting and merging, as
:numref:`fig_head-split` shows. `split_heads` cuts the last axis of width
$H d_h$ into $H$ slices of width $d_h$ and moves the head axis in front of the
length axis. `merge_heads` reverses both steps, so the output projection sees
the head outputs side by side, as in :eqref:`eq_multihead-output`. Head $i$
therefore owns columns $(i-1)d_h + 1$ to $i d_h$ of each projection's output.

![Heads as slices of one projection. The projected queries of $n$ tokens are cut into $H$ column slices of width $d_h = d/H$, which become a separate head axis; each head attends on its own, and the head outputs are placed side by side again before the output projection $\mathbf{W}_o$. Keys and values are split in the same way.](../img/mdl-attention-head-split.svg)
:label:`fig_head-split`

```{.python .input #multihead-attention-implementation-2}
%%tab pytorch
@d2l.add_to_class(MultiHeadAttention)  #@save
def split_heads(self, X, num_heads):
    """(batch, length, num_heads * width) -> (batch, num_heads, length,
    width)."""
    return X.reshape(X.shape[0], X.shape[1], num_heads, -1).transpose(1, 2)

@d2l.add_to_class(MultiHeadAttention)  #@save
def merge_heads(self, X):
    """Reverse split_heads: (batch, num_heads, length, width) -> (batch,
    length, num_heads * width)."""
    return X.transpose(1, 2).reshape(X.shape[0], X.shape[2], -1)
```

```{.python .input #multihead-attention-implementation-2}
%%tab jax
@d2l.add_to_class(MultiHeadAttention)  #@save
def split_heads(self, X, num_heads):
    """(batch, length, num_heads * width) -> (batch, num_heads, length,
    width)."""
    return X.reshape(X.shape[0], X.shape[1], num_heads, -1).swapaxes(1, 2)

@d2l.add_to_class(MultiHeadAttention)  #@save
def merge_heads(self, X):
    """Reverse split_heads: (batch, num_heads, length, width) -> (batch,
    length, num_heads * width)."""
    return X.swapaxes(1, 2).reshape(X.shape[0], X.shape[2], -1)
```

### Learned Heads on the Copy-Both Task
:label:`subsec_copy-both-learned`

We now return to the copy-both task with the layer itself instead of
hand-built mixtures. A layer of width $d = 16$ receives one query and two
key--value pairs. The keys are the standard basis vectors $\mathbf{e}_1$
and $\mathbf{e}_2$, so they mark position only, and the query is
$\mathbf{e}_3$ in every example. The values carry the content:
$\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^{8}$, drawn afresh for every
example and padded with zeros to width 16. So $d_v = 8$ is the width of
the content a value carries, while each head projects the padded values to
its own width $d_h = 16/H$. The target $[\mathbf{v}_1; \mathbf{v}_2]$ is
the layer's entire output, so the output has room for exactly two values
($2d_v = d$).

Three functions do the work. The data function draws a minibatch. The
training function fits a layer with Adam on the squared error, drawing a
fresh minibatch at every step. The evaluation function returns the
relative error on 10,000 fresh examples, together with the weight that
each head gives to the first position.

```{.python .input #multihead-attention-learned-heads-on-the-copy-both-task-1}
%%tab pytorch
def copy_both_data(n, d=16, d_v=8):
    """One query, two position-only keys, values of width d_v padded to d."""
    V = torch.randn(n, 2, d_v)
    keys = torch.eye(d)[:2].expand(n, 2, d)    # e_1, e_2 mark the positions
    query = torch.eye(d)[2].expand(n, 1, d)    # e_3 asks for both values
    values = F.pad(V, (0, d - d_v))
    return query, keys, values, V.reshape(n, 1, 2 * d_v)

def train_copy_both(num_heads, num_kv_heads=None, num_steps=2000):
    model = MultiHeadAttention(16, num_heads, num_kv_heads=num_kv_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(num_steps):
        query, keys, values, target = copy_both_data(256)
        loss = ((model(query, keys, values) - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def evaluate_copy_both(model, n=10000):
    """Relative error on fresh examples; each head's weight on v_1."""
    query, keys, values, target = copy_both_data(n)
    with torch.inference_mode():
        err = (model(query, keys, values) - target).norm() / target.norm()
    return err.item(), model.attention_weights[0, :, 0, 0].tolist()
```

```{.python .input #multihead-attention-learned-heads-on-the-copy-both-task-1}
%%tab jax
def copy_both_data(key, n, d=16, d_v=8):
    """One query, two position-only keys, values of width d_v padded to d."""
    V = jax.random.normal(key, (n, 2, d_v))
    keys = jnp.broadcast_to(jnp.eye(d)[:2], (n, 2, d))  # e_1, e_2: positions
    query = jnp.broadcast_to(jnp.eye(d)[2], (n, 1, d))  # e_3 asks for both
    values = jnp.pad(V, ((0, 0), (0, 0), (0, d - d_v)))
    return query, keys, values, V.reshape(n, 1, 2 * d_v)

@nnx.jit
def copy_both_step(model, optimizer, key):
    query, keys, values, target = copy_both_data(key, 256)
    loss = lambda model: ((model(query, keys, values) - target) ** 2).mean()
    optimizer.update(model, nnx.grad(loss)(model))

def train_copy_both(num_heads, num_kv_heads=None, num_steps=2000):
    model = MultiHeadAttention(16, num_heads, num_kv_heads=num_kv_heads,
                               rngs=nnx.Rngs(d2l.get_key()))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    # Full fp32 products: TF32, the default on recent NVIDIA cards, would
    # stop the error of a solved task near 1e-3 instead of fp32 rounding
    with jax.default_matmul_precision('highest'):
        for key in jax.random.split(d2l.get_key(), num_steps):
            copy_both_step(model, optimizer, key)
    return model

def evaluate_copy_both(model, n=10000):
    """Relative error on fresh examples; each head's weight on v_1."""
    query, keys, values, target = copy_both_data(d2l.get_key(), n)
    with jax.default_matmul_precision('highest'):
        Y = model(query, keys, values)
    err = jnp.linalg.norm(Y - target) / jnp.linalg.norm(target)
    return float(err), model.attention_weights[0, :, 0, 0].tolist()
```

We train layers with one, two, and four heads, twice each, so that we can
see whether an outcome depends on the run. The proposition predicts that
one head stays at $0.71$, while two heads have what the task needs. The
cell prints, for every run, the parameter count, the relative
error, and the weight that each head gives to the first position (its
weight on the second is one minus that).

```{.python .input #multihead-attention-learned-heads-on-the-copy-both-task-2}
%%tab pytorch, jax
print('H  run  parameters  error    weight on v_1, per head')
for num_heads in (1, 2, 4):
    for run in (1, 2):
        model = train_copy_both(num_heads)
        err, alpha_1 = evaluate_copy_both(model)
        print(f'{num_heads}  {run:3d}  {d2l.num_params(model):10d}  '
              f'{err:<#8.2g} ' + ' '.join(f'{a:.2f}' for a in alpha_1))
```

Every layer has 1,024 parameters, four $16 \times 16$ matrices, whatever
its number of heads. One head stays at $0.71$ in every run, the bound of
the proposition: all 16 outputs of a single head are linear functions of
the 8 numbers in one mixture. Its weight on the first position ends near
one half, close to where it started. Every split is equally bad, so
training has little reason to move it. With two or four heads the error
falls to about $10^{-7}$, the rounding error of single precision, and the
layer returns both values.

How do the heads manage it? In every run one head puts more than 0.8 of its
weight on the first position and another less than 0.2, and some heads stay
soft. Two different mixtures are all that the output projection needs, just
as in the hand-built construction. And the two extra heads of the four-head
layer? They add nothing the task requires; their weightings repeat those of
the other two or lie between them. A word of caution belongs here. The task
shows what one head cannot do. It does not show that trained language
models divide the work among their heads this way.

### Shared Keys and Values
:label:`subsec_kv-head-sharing`

In the copy-both layer the heads had to differ in how they weight the
positions, and that difference can come from their query projections
alone. Their keys could have been shared, and so could their values, as
long as the shared value heads together carry all $d_v$ coordinates of a
value. Sharing of this kind has a name. *Grouped-query attention* keeps $H$
query heads but computes only $H_\textrm{kv}$ key--value heads, each shared
by a group of $H/H_\textrm{kv}$ consecutive query heads
:cite:`Ainslie.Lee-Thorp.Jong.ea.2023`; with $H_\textrm{kv} = 1$ it is
*multi-query attention* :cite:`Shazeer.2019`.

Why share at all? A model that generates text one token at a time stores
the keys and values of all earlier tokens so that it need not recompute
them, and sharing shrinks that store by the factor $H/H_\textrm{kv}$.
:numref:`sec_kv-cache` builds the store, and :numref:`sec_kv-cache-reduction`
measures what sharing saves and reports how much quality large models give
up. In `MultiHeadAttention`,
`num_kv_heads` sets $H_\textrm{kv}$: `W_k` and `W_v` produce $H_\textrm{kv}$
heads, and the forward pass copies each of them for the query heads of its
group. Copying is the simplest correct code, and what it costs is
temporary memory inside one forward pass. The keys and values that a model
stores for generation, which are what sharing shrinks, hold each shared
head once.

What does the proposition predict? The shared value heads have
$H_\textrm{kv} d_h$ coordinates between them, and these must hold the $d_v$
coordinates of a value's content, so the room condition reads
$H_\textrm{kv} d_h \ge d_v$. With $d = 16$ and $d_v = 8$, the layouts
$(H, H_\textrm{kv}) = (2, 1)$ and $(4, 2)$ give the shared value heads 8
coordinates, as many as a value has, and should solve the task. The layout $(4, 1)$ has four query
heads, and so four weightings, but a single value head of width 4. Every
head then reads the same 4-dimensional projection of each value, the
readout can recover at most 8 of the 16 target numbers, and the error
should return to $\sqrt{1/2} \approx 0.71$. We train each layout twice.

```{.python .input #multihead-attention-sharing-keys-and-values}
%%tab pytorch, jax
print('H  H_kv  run  parameters  error    weight on v_1, per head')
for num_heads, num_kv_heads in ((2, 1), (4, 2), (4, 1)):
    for run in (1, 2):
        model = train_copy_both(num_heads, num_kv_heads)
        err, alpha_1 = evaluate_copy_both(model)
        print(f'{num_heads}  {num_kv_heads:4d}  {run:3d}  '
              f'{d2l.num_params(model):10d}  {err:<#8.2g} '
              + ' '.join(f'{a:.2f}' for a in alpha_1))
```

The prediction holds in every run. The two layouts whose value heads have
room solve the task to the same rounding error as the full layers, with 768
parameters instead of 1,024. The layout $(4, 1)$, with 640 parameters,
stays at $0.71$, although its four heads learn clearly different
weightings. It fails for lack of room, not for lack of weightings. Heads
may share what they read, then, as long as what they share is wide enough
to hold what they retrieve.

## Cross-Attention

Self-attention takes its queries, keys, and values from one sequence. In
*cross-attention* the queries come from one sequence $\mathbf{A}$ and the
keys and values from another sequence $\mathbf{B}$. The output has one row
for each token of $\mathbf{A}$ and gathers its content from $\mathbf{B}$.
The layer is the same; only its arguments differ, which is why
`MultiHeadAttention` takes three of them. We have used cross-attention all
along without the name: the letters of "attention" querying those of
"translation" and the alignment model of
:numref:`sec_attention-scoring-functions` were cross-attention, and so was
the copy-both layer above, whose one query read two stored key--value pairs.

Cross-attention is how one model reads the output of another. In the
original encoder--decoder Transformer, each position of the translation
queries the encoded source sentence. In vision--language models such as
Flamingo :cite:`alayrac2022flamingo`, text tokens query features of an
image. A language model that only continues text has no second sequence and
uses self-attention alone.

Let us return once more to *gave*. Two heads could give it one weighting
for the people and another for the book. No weighting computed from
content, however, can tell the giver from the receiver: swap the two names,
"Bob gave Alice the book", and the output at *gave* does not change. To
know who gave and who received, attention needs what it has ignored so
far, the order of the words.

## Summary

Self-attention lets every token of a sequence query every other token:
three learned maps turn each token into a query, a key, and a value, and
the output for a token is the softmax-weighted average of the values.
The layer connects any two positions in one step and computes all positions
at once, at a cost quadratic in the length. One head gives each query one
set of weights and so one mixture of the values: when a query needs two
values, the best linear readout of that mixture loses half of the target's
variance, whatever the weights. Multi-head attention runs $H$ heads of
width $d_h = d/H$ in learned projections and recombines them, with
parameter and leading FLOP counts that do not depend on $H$, and a trained
layer with two heads returns both values to rounding error. Heads may share
keys and values, which shrinks the keys and values stored during generation
by the factor $H/H_\textrm{kv}$, as long as the shared value heads have room for
what is retrieved. Cross-attention is the same layer with queries from one
sequence and keys and values from another.

## Exercises

1. **Self-attention by hand.** Three tokens have the inputs
   $\mathbf{x}_1 = (1, 0)$, $\mathbf{x}_2 = (0, 1)$, and $\mathbf{x}_3 =
   (1, 1)$. One head of width $d_h = 2$ uses
   $\mathbf{W}_q = \sqrt{2}\,\mathbf{I}$, the swap
   $\mathbf{W}_k = \left(\begin{smallmatrix} 0 & 1 \\ 1 & 0
   \end{smallmatrix}\right)$, and $\mathbf{W}_v = \mathbf{I}$, so that the
   factor $\sqrt{2}$ cancels the scale $1/\sqrt{d_h}$.
    1. Compute the queries, keys, and values, the $3 \times 3$ matrix of
       scaled scores, the weights, and the three outputs, each to two
       decimals. Which token does the first token attend to least, and why?
    1. Apply a causal mask and recompute. Which rows change, and why does
       the last row stay as it was?
    1. Replace $\mathbf{W}_k$ by the identity, so that each key points the
       same way as the query of its token. Recompute the weights of the
       first token. Why can no key of the same length as $\mathbf{x}_1$
       score higher than the first token's own key, and why does the key of
       the third token score as high anyway?
    1. Check your numbers with `d2l.DotProductAttention`.

1. [code] **Narrow heads.** With $d_h = d/H$, the leading count of
   :eqref:`eq_multihead-flops` does not depend on $H$, so the number of heads
   looks free up to $H = d$. Consider one head with scores
   $\mathbf{q}^\top \mathbf{k}_j / \sqrt{d_h}$ over keys $\mathbf{k}_1,
   \ldots, \mathbf{k}_m \in \mathbb{R}^{d_h}$ and no mask. Call a query sharp
   on key $j$ when its weights concentrate on $\mathbf{k}_j$ as the query
   is scaled by a factor that grows without bound.
    1. For $d_h = 1$ and distinct scalar keys, show that every query's weights
       are monotone in $k_j$. Conclude that the queries of such a head can
       be sharp only on the largest or the smallest key.
    1. Show that for $d_h = 2$ the keys and queries can be chosen so that
       each of $n$ queries is sharp on a key of its own choosing, for every
       assignment of queries to keys.
    1. Use these two results to state which attention patterns a layer
       gives up as $H$ approaches $d$.
    1. Time the forward pass of `MultiHeadAttention` with dropout 0 for
       self-attention with $d = 256$ and $n = 512$ on a minibatch of $8$
       sequences for $H \in \{1, 2, 4, 8, 16, 32, 64\}$. Make a warm-up
       call first and wait for the result before stopping the clock.
       Report the ratio of the slowest to the fastest configuration.
       Identify the cost that grows with $H$ and that
       :eqref:`eq_multihead-flops` leaves out, and check that it accounts
       for the trend.
   :exlabel:`ex_multihead-narrow-heads`

1. [code] **Heads as slices.** `MultiHeadAttention` computes one projection
   of width $d$ for each of the queries, keys, and values and splits it
   into heads with `split_heads`.
    1. Head $i$ owns a slice of the output features of `W_q`, `W_k`, and
       `W_v`. Determine which input features of `W_o` receive that head's
       output. Write a forward pass that loops over the heads, applies the
       sliced projections, and calls `d2l.DotProductAttention` once per
       head. Verify on random inputs with a padding mask that it matches
       `MultiHeadAttention` with dropout 0, and extend your loop to
       `num_kv_heads < num_heads`.
    1. Replace the body of `split_heads` by
       `X.reshape(X.shape[0], num_heads, X.shape[1], -1)`, which returns a
       tensor of the same shape. Describe which numbers each "head" now
       holds, show on random inputs that the output changes, and explain why
       no shape check can detect the mistake. Which input lengths and widths,
       if any, make the two versions agree?

1. **Permuting the source.** In cross-attention, the queries come from
   $\mathbf{A} \in \mathbb{R}^{n \times d}$ and the keys and values from
   $\mathbf{B} \in \mathbb{R}^{m \times d}$, neither carrying positional
   information. Let $\boldsymbol{\Pi}$ be an $m \times m$ permutation
   matrix.
    1. Show that replacing $\mathbf{B}$ by $\boldsymbol{\Pi}\mathbf{B}$ in
       both the key and the value argument leaves the output of
       `MultiHeadAttention` unchanged when no mask is applied, and that
       replacing it in the key argument alone changes the output in general.
    1. Now let a padding mask mark only the first $\ell$ rows of
       $\mathbf{B}$ as valid. Which permutations leave the output unchanged
       when the mask stays fixed?
   :exlabel:`ex_multihead-permuting-the-source`

1. [code] **Nonlinear readouts.** ● For Gaussian values, the proof of the
   proposition in :numref:`subsec_one-mixture-bottleneck` shows that the
   conditional mean $\mathbb{E}[\mathbf{v}_1 \mid \mathbf{m}]$ is linear in
   the mixture. Hence no readout, linear or not, improves on the relative
   error $\sqrt{1/2}$ that `readout_error` measures.
    1. Train a small multilayer perceptron to predict `target` from the
       mixture $\mathbf{m} = a\,\mathbf{v}_1 + (1 - a)\,\mathbf{v}_2$
       formed from `V1` and `V2`, for $a = 0.5$ and $a = 0.7$. Compare its
       relative error on held-out samples with that of the linear readout.
    1. Now draw the values uniformly from $\{-1, +1\}^{d_v}$. For $a = 0.5$,
       compute $\mathbb{E}[\mathbf{v}_1 \mid \mathbf{m}]$ coordinate by
       coordinate, show that it is again linear in $\mathbf{m}$, and
       conclude that the error stays at $\sqrt{1/2}$ for every readout.
    1. For $a = 0.7$ and the same discrete values, show that the map
       $(v_{1j}, v_{2j}) \mapsto m_j$ is injective. Train the perceptron
       again and report its held-out error. Explain why this escape from
       the bound does not contradict the proposition.

1. [code] **Nearly identical heads.** ● Two heads with weights
   $(0.5 + \epsilon, 0.5 - \epsilon)$ and $(0.5 - \epsilon, 0.5 +
   \epsilon)$ supply two distinct mixtures, so the linear readout inverts
   them exactly for every $\epsilon > 0$.
    1. Write the $2 \times 2$ mixing matrix, compute its eigenvalues and
       eigenvectors, and express its condition number as a function of
       $\epsilon$.
    1. Add independent Gaussian noise of standard deviation $\sigma = 0.01$
       to both mixtures and measure the error with `readout_error` for
       $\epsilon \in \{0.4, 0.1, 0.03, 0.01, 0.003, 0.001\}$. Plot the
       error against $\epsilon$ on logarithmic axes, read off the slope
       where the curve is straight, and explain it with the condition
       number from (a).
    1. Every head of a freshly initialized layer weights the two positions
       of the copy-both task close to $(0.5, 0.5)$. Modify
       `train_copy_both` to record the relative error and the weights of
       each head every 20 steps, and plot both for $H = 2$. Use the
       conditioning of the mixing matrix to explain the shape of the error
       curve in the first few hundred steps. Repeat for $H = 8$ and sort the
       final heads into those that are nearly one-hot, those that are soft
       but distinct, and those that nearly duplicate another head.
   :exlabel:`ex_multihead-nearly-identical-heads`

1. **What the bound counts.** ● The proposition of
   :numref:`subsec_one-mixture-bottleneck` fixes the softmax weights, but
   its error bound rests on a weaker fact: the readout receives a single
   vector $\mathbf{m} \in \mathbb{R}^{d_v}$ and must produce the target
   $\mathbf{t} \in \mathbb{R}^{2d_v}$.
    1. Let $\mathbf{m} \in \mathbb{R}^{d_v}$ be any function of
       $\mathbf{v}_1$ and $\mathbf{v}_2$, including a mixture whose weights
       depend on the values, and let $\mathbf{W}_o \in \mathbb{R}^{d_v
       \times 2d_v}$ be any matrix. Show that
       $$\mathbb{E}\,\|\mathbf{m}\mathbf{W}_o - \mathbf{t}\|^2 \geq \tfrac{1}{2}\, \mathbb{E}\,\|\mathbf{t}\|^2.$$
       Conclude that content-dependent keys, more layers, or a residual
       connection leave the bound in place as long as the readout is linear
       and receives $d_v$ numbers.
    1. A layer of width $d$ with two heads of width $d_h = d/2$ receives the
       values $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^{d/2}$ padded with
       zeros to width $d$. Give the attention weights of each head, value
       projections $\mathbf{W}_i^{(v)}$, and an output projection
       $\mathbf{W}_o$ that solve the task exactly. Compare your weights with
       those printed in :numref:`subsec_copy-both-learned`.
    1. Generalize the proposition to $k$ source positions with independent
       standard Gaussian values $\mathbf{v}_1, \ldots, \mathbf{v}_k \in
       \mathbb{R}^{d_v}$ and the target $[\mathbf{v}_1; \ldots;
       \mathbf{v}_k]$. Each of $H$ heads applies fixed weights to the
       values, the weights of head $i$ form row $i$ of a matrix
       $\mathbf{A} \in \mathbb{R}^{H \times k}$, and a linear readout sees
       all $H$ mixtures. Show that the best relative squared error is
       $$1 - \frac{\operatorname{rank}(\mathbf{A})}{k},$$
       and that softmax weights, whose entries are all positive, can attain
       $\operatorname{rank}(\mathbf{A}) = \min(H, k)$.
   :exlabel:`ex_multihead-what-the-bound-counts`

1. [code] **Sharing keys and values on the copy-both task.** ● Let a layer
   of width $d$ have $H$ query heads of width $d_h = d/H$ and
   $H_\textrm{kv} < H$ key--value heads.
    1. Use the counting of Exercise
       :exref:`ex_multihead-what-the-bound-counts` to show that it can solve
       the task only if $H_\textrm{kv} d_h \ge d_v$, and that this condition
       suffices.
    1. Train a layer with `train_copy_both(2, 1)`, in which two query heads
       share one key--value head. Find the parameters that differ between
       the two heads and those they share, and explain from the copy-both
       task why the difference must lie where you found it.
   :exlabel:`ex_multihead-sharing-keys-and-values`

[Discussions](https://d2l.discourse.group/t/36770)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.3]{.kicker}

Self-Attention and Multiple Heads<br>
**a sentence reads itself · what one head cannot do · heads in subspaces · shared keys and values**
:::
:::

::: {.slide title="One Head Hands 'gave' One Vector for Three Words"}
[The problem]{.kicker}

- *gave* needs three other words: who gave, who received, what was given.
- Self-attention: every word sends a query to the keys of its own
  sentence.
- One head returns **one** weighted average of the values: in GPT-2
  small, 64 numbers per token, whatever it read.

::: {.d2l-note .warn}
Can one query read several positions without blending what it reads?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute one head of self-attention by hand, with and without a causal
  mask;
- prove that one set of weights loses half of two requested values;
- write multi-head attention and count its FLOPs and parameters;
- predict when heads may share keys and values;
- tell self-attention from cross-attention by their inputs.

**Running example:** "Alice gave Bob the book" with $d_h = 2$, then a
GPT-2-small layer: $d = 768$, 12 heads of width 64.
:::

::: {.slide title="Self-Attention Is the Layer the Rest of the Book Is Built From"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Queries, Keys, and Values | attention as a soft lookup |
| Attention Scoring and Masking | scaled dot products, masks |
| **Self-Attention and Multiple Heads** | **a sequence reads itself; several heads** |
| Positional Information | where a token sits |
| The Cost of Attention | quadratic cost, fused kernels |
| Windowed and Linear Attention | linear cost, and what it gives up |
| Attention Circuits | what trained heads compute |
:::

::: {.col .narrow}
- **Uses:** scaled dot-product attention, causal masks, least squares.
- **Skips:** word order (next deck); the key-value cache itself (next
  chapter).
:::
:::
:::

::: {.slide title="A Token Asks, Is Found, and Is Read: Three Maps"}
[Self-attention]{.kicker}

- **query** $\mathbf{x}\mathbf{W}_q$: what the token looks for
- **key** $\mathbf{x}\mathbf{W}_k$: what it offers to other queries
- **value** $\mathbf{x}\mathbf{W}_v$: what it hands over once chosen

$$\mathrm{softmax}\left(\frac{(\mathbf{X}\mathbf{W}_q)(\mathbf{X}\mathbf{W}_k)^\top}{\sqrt{d_h}}\right)\mathbf{X}\mathbf{W}_v$$

$\mathbf{X}$: the $n$ tokens as rows; $d_h$: width of queries and keys.

::: {.d2l-note}
One sequence supplies the queries, the keys, and the values.
:::
:::

::: {.slide title="Check: Predict the Weights of 'gave'"}
[Check]{.kicker}

Hand-set maps, $d_h = 2$, score $\mathbf{q} \cdot \mathbf{k}/\sqrt{2}$:

| token | role | query | key |
|:--|:--|:--|:--|
| Alice, Bob | person | $(1, -1)$ | $(1, 1)$ |
| gave | action | $(1, 1)$ | $(1, -1)$ |
| book | thing | $(1, -1)$ | $(1, 1)$ |
| the | function word | $(0, 0)$ | $(0, 0)$ |

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Scores $2/\sqrt{2} \approx 1.41$ on the three nouns, $0$ on
*gave* and "the". Weights $e^{1.41}/(3e^{1.41}+2) \approx 0.29$ each and
$\approx 0.07$ for the other two.
:::
:::
:::

::: {.slide title="'Gave' Reads Equal Parts Alice, Bob, and Book"}
[Self-attention]{.kicker}

![](../img/mdl-attention-self-attention.svg){width=88%}

::: {.d2l-note}
Alice and Bob get the same weight: this head cannot say who gave and who
received.
:::
:::

::: {.slide title="The Layer Reproduces the Hand Computation; a Causal Mask Leaves 0.80 and 0.20"}
@!multihead-attention-self-attention-1

- Causal mask: the same scores, renormalized over Alice and *gave*.
:::

::: {.slide title="Separate Query and Key Maps Let a Token Seek What It Is Not"}
[Self-attention]{.kicker}

One shared map $\mathbf{u}_i = \mathbf{x}_i\mathbf{W}$ for queries and keys:

- scores symmetric, and $s_{ij} \le \sqrt{s_{ii}\, s_{jj}}$
  (Cauchy–Schwarz): no score exceeds the geometric mean of the two
  self-scores;
- the verb's $(1, 1)$ would have dot product 2 with itself and 0 with
  every noun.

::: {.d2l-note}
What a token asks for must be free to differ from what it offers.
:::
:::

::: {.slide title="One Head Returns One Mixture of Two Requested Values"}
[Limits of a single head]{.kicker}

**Copy-both task:** two positions hold random $\mathbf{v}_1,
\mathbf{v}_2 \in \mathbb{R}^{d_v}$; keys mark positions only; one query
must return $\mathbf{t} = [\mathbf{v}_1; \mathbf{v}_2]$.

- One head: weights $\alpha_1 + \alpha_2 = 1$, the same for every example.
- It hands the readout $\mathbf{m} = \alpha_1 \mathbf{v}_1 + \alpha_2
  \mathbf{v}_2$: $d_v$ numbers for a $2d_v$-number target.
:::

::: {.slide title="Any Single Head Has Relative Squared Error at Least 1/2"}
[Limits of a single head]{.kicker}

Values standard Gaussian. The best estimate is linear:

$$\mathbb{E}[\mathbf{v}_1 \mid \mathbf{m}] = \frac{\alpha_1}{\alpha_1^2 + \alpha_2^2}\,\mathbf{m}.$$

- Error per coordinate: $\frac{\alpha_2^2}{\alpha_1^2 + \alpha_2^2}$ for
  $\mathbf{v}_1$, $\frac{\alpha_1^2}{\alpha_1^2 + \alpha_2^2}$ for
  $\mathbf{v}_2$: sum $1$.
- Target variance $2$ per coordinate $\Rightarrow$ relative error $1/2$.

::: {.d2l-note}
The weights move the error between $\mathbf{v}_1$ and $\mathbf{v}_2$; they
cannot remove it.
:::
:::

::: {.slide title="A Solver Needs Several Weightings and Room for Every Value"}
[Limits of a single head]{.kicker}

![](../img/mdl-attention-one-head-averages.svg){width=70%}

1. **Weightings:** several different weightings for one query.
2. **Room:** output space for every value retrieved.

Neither suffices alone: one head as wide as the layer has room but one
weighting.
:::

::: {.slide title="One Mixture Stays at 0.71; Two Mixtures Reach Rounding Error"}
[Limits of a single head]{.kicker}

@!multihead-attention-one-head-must-average-1

- Values of width 8, 10,000 examples, best linear readout by least
  squares.
- Relative error $= \sqrt{\textrm{relative squared error}}$: the bound is
  $\sqrt{1/2} \approx 0.707$.
- Soft weights $(0.88, 0.12)$ work as well as one-hot ones.
:::

::: {.slide title="Check: Does a Residual Connection Lift the Bound?"}
[Check]{.kicker}

Add the query to the head's output and keep a linear readout of $d_v$
numbers. Can the error drop below $1/2$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** No. The query is the same in every example and carries nothing
about the values, and any $d_v$ numbers read out linearly leave error
$1/2$. Only more numbers (more heads, a wider output) help.
:::
:::
:::

::: {.slide title="Each Head Attends in Its Own Projection"}
[Multi-head attention]{.kicker}

$$\mathbf{Z}_i = \mathrm{softmax}\left(\frac{(\mathbf{X}\mathbf{W}_i^{(q)})(\mathbf{X}\mathbf{W}_i^{(k)})^\top}{\sqrt{d_h}}\right)\mathbf{X}\mathbf{W}_i^{(v)}, \qquad \textrm{output} = [\mathbf{Z}_1, \ldots, \mathbf{Z}_H]\,\mathbf{W}_o$$

![](../img/mdl-attention-multi-head.svg){width=36%}

New: the head index $i$ and the output projection $\mathbf{W}_o$.
:::

::: {.slide title="Heads Change the Attention Map, Not the Cost"}
[Multi-head attention]{.kicker}

Head width $d_h = d/H$, sequence length $n$:

$$\underbrace{8nd^2}_{\textrm{projections}} + \underbrace{4n^2 d}_{\textrm{scores, mixing}} \ \textrm{FLOPs}, \qquad 4d^2 \ \textrm{parameters.}$$

GPT-2 small layer, $n = 1024$: 4.8 GFLOP projections, 3.2 GFLOP scores
and mixing, 2.4M parameters, for 1 head or 12.

::: {.d2l-note}
Why not $H = d$? A head of width 1 can point only at the largest or the
smallest key.
:::
:::

::: {.slide title="Self-Attention Shortens Every Path to One Step and Pays a Quadratic Cost"}
[Multi-head attention]{.kicker}

![](../img/mdl-attention-cnn-rnn-self-attention.svg){width=56%}

$n$ tokens of width $d$, kernel size $k$:

| layer | cost | sequential | path |
|:--|:--|:--|:--|
| convolution | $knd^2$ | $1$ | $n/k$ |
| recurrence | $nd^2$ | $n$ | $n$ |
| self-attention | $nd^2 + n^2 d$ | $1$ | $1$ |

::: {.d2l-note}
Wins path and parallelism; the $n^2 d$ term overtakes the projections once
$n > 2d$ (GPT-2 small: 1,536 tokens).
:::
:::

::: {.slide title="Heads Are One More Tensor Axis"}
[Multi-head attention]{.kicker}

![](../img/mdl-attention-head-split.svg){width=92%}

- `split_heads`: reshape to `(batch, n, H, d_h)`, move the head axis
  forward; `merge_heads` undoes both.
- Attention runs over the last two axes: all heads in one batched product.
:::

::: {.slide title="A Trained Layer: One Head Stays at 0.71, Two Heads Reach Rounding Error"}
[Multi-head attention]{.kicker}

@!multihead-attention-learned-heads-on-the-copy-both-task-2

- $d = 16$, keys $\mathbf{e}_1, \mathbf{e}_2$, query $\mathbf{e}_3$,
  values of width 8: room for exactly two.
- One head: weight on $\mathbf{v}_1$ ends near one half; every split is
  equally bad.
- Every run of $H \ge 2$: one head above 0.8 on $\mathbf{v}_1$, one below
  0.2.
:::

::: {.slide title="Heads May Share Keys and Values"}
[Shared keys and values]{.kicker}

![](../img/mdl-transformers-gqa.svg){width=72%}

- $H$ query heads read $H_\textrm{kv}$ key-value heads: grouped-query
  attention; $H_\textrm{kv} = 1$: multi-query.
- Generation stores past keys and values: $H/H_\textrm{kv}$ times less.
  GPT-2 small, 16-bit, all 12 layers, 12 → 4 key-value heads: 36 → 12 KiB
  per token.
:::

::: {.slide title="Check: Which Layouts Solve Copy-Both?"}
[Check]{.kicker}

$d = 16$, $d_v = 8$, $d_h = d/H$. Which of $(H, H_\textrm{kv}) = (2, 1)$,
$(4, 2)$, $(4, 1)$ can return both values?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Room requires $H_\textrm{kv} d_h \ge d_v$. $(2, 1)$ and $(4, 2)$
give 8 shared value coordinates and succeed; $(4, 1)$ gives 4: four
weightings, no room, error back to $0.71$.
:::
:::
:::

::: {.slide title="Sharing Works While the Shared Value Heads Have Room"}
[Shared keys and values]{.kicker}

@!multihead-attention-sharing-keys-and-values

- $(2, 1)$, $(4, 2)$: rounding error with 768 parameters instead of 1,024.
- $(4, 1)$: 640 parameters, $0.71$, though its heads weight the positions
  differently.
:::

::: {.slide title="Self- and Cross-Attention Differ Only in Their Inputs"}
[Cross-attention]{.kicker}

- **Self-attention:** `attention(X, X, X)`. One sequence reads itself;
  output shape = input shape, so layers stack.
- **Cross-attention:** `attention(A, B, B)`. Sequence A asks, sequence B
  answers; output length = length of A.

Translation decoders and vision–language models read another sequence
this way.

::: {.d2l-note}
Cross-attention is how one model reads the output of another.
:::
:::

::: {.slide title="Check: Swap Alice and Bob"}
[Check]{.kicker}

"Bob gave Alice the book." Does the output of *gave* change, with one head
or with several?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** No. The weights depend on content only; swapping two tokens
swaps their values and their weights together. Content cannot tell the
giver from the receiver.
:::
:::
:::

::: {.slide title="Several Weightings per Query, Room for What They Read"}
[Wrap-up]{.kicker}

- One head on the sentence: *gave* gets $0.29$ on each noun, a blend.
- One head, one mixture: relative squared error $1/2$ on copy-both, for
  any weights.
- $H$ heads of width $d/H$: $8nd^2 + 4n^2d$ FLOPs, $4d^2$ parameters for
  every $H$; two trained heads reach rounding error.
- Shared keys and values: safe while $H_\textrm{kv} d_h \ge d_v$.

**Next:** attention ignores order; positional information restores it.
:::
