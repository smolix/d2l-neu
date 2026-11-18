# Queries, Keys, and Values
:label:`sec_queries-keys-values`

The recurrent networks of :numref:`sec_rnn` keep everything they know
about the past in a state of fixed size, and every prediction is made from
that state. The network has to decide what to write there before it knows
what a later prediction will need. Consider a recurrent language model as
wide as GPT-2 small that has read 1,024 tokens. Its hidden state holds
768 numbers. If the next prediction depends on the 17th token, it must be
made from those 768 numbers, all written before the network could know
that the 17th token would matter.

A database never has to guess in advance. It stores its records as they
are and decides what to read only when a query arrives, by comparing the
query with the stored keys. An exact lookup, however, either finds a
matching key or returns nothing, and a network trained by gradient descent
needs an answer that changes smoothly with the query. Take the smallest
possible case, the scalar database $\mathcal{D}=\{(2,10),(5,20)\}$ of
(key, value) pairs. The exact query
$q=5$ returns the value 20, and the query $q=4$ returns nothing at all,
because no key matches. A *soft* query instead scores every key, for
example by $a(q,k)=-(q-k)^2$, and turns the scores into weights that
sum to one. For $q=4$ the scores are $-4$ for key 2 and $-1$ for key 5. The
softmax turns them into the weights $e^{-4}/(e^{-4}+e^{-1}) \approx 0.05$
and $0.95$, so the answer is $0.05 \cdot 10 + 0.95 \cdot 20 \approx 19.5$.
The answer moves smoothly with the query, the same rule works for a
database of any size, and nothing has to be compressed before the query
arrives.

The *attention mechanism* :cite:`Bahdanau.Cho.Bengio.2014` is this soft
lookup, used as a layer of a neural network. We first write it for vectors
and work one lookup through by hand. We then meet it in its oldest form,
kernel regression, where the weights come from a similarity function fixed
in advance rather than learned. That setting is small enough to plot every
weight, and it lets us ask what decides whether a lookup picks out one
record or averages them all.

```{.python .input #queries-keys-values-queries-keys-and-values}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #queries-keys-values-queries-keys-and-values}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Attention as Differentiable Lookup

The soft query of the example used scalars and two stored pairs. In a
network the keys and values are vectors, and there is a pair for every
position of the input, so we now write the same rule in that generality.

Let the query and the keys be vectors $\mathbf{q},\mathbf{k}_i \in
\mathbb{R}^{d}$ and the values $\mathbf{v}_i \in \mathbb{R}^{d_v}$, and let
$\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1),
\ldots, (\mathbf{k}_m, \mathbf{v}_m)\}$ be a database of $m$ key--value
pairs. The keys are what the query is compared with and the values are
what comes back, so the two need not have the same width. The output of
attention is the weighted sum of the values

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$
:eqlabel:`eq_attention_pooling`

where the scalar weight $\alpha(\mathbf{q},\mathbf{k}_i)$ says how much
value $i$ contributes. Because the sum pools all the stored values into a
single vector, it is also called *attention pooling*. In this book the
weights are always nonnegative and sum to one, so the output is a convex
combination of the values, a point inside their convex hull. Exact lookup
is the special case in which one weight is $1$ and the others are $0$.
Uniform weights $1/m$ give the plain average of the values, *average
pooling*.

Weights with these two properties come from any scoring function
$a(\mathbf{q}, \mathbf{k})$ by exponentiating the scores, which makes them
positive, and dividing by their sum. This is the softmax operation of
multinomial logistic regression (:numref:`sec_softmax`):

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}.$$
:eqlabel:`eq_softmax_attention`

The notation $\alpha(\mathbf{q}, \mathbf{k}_i)$ names only the pair that
the weight belongs to. Through the denominator, the weight depends on all
the keys. Writing $a_i$ for $a(\mathbf{q}, \mathbf{k}_i)$, the weight is
$\alpha_i = 1 / \big(1 + \sum_{j \neq i} e^{a_j - a_i}\big)$, so it depends
only on how its score compares with the other scores, and adding the same
constant to every score changes nothing. The softmax and the weighted sum
are both differentiable, so gradients reach the query, the keys, and the
values. Attention can also make hard choices, keeping one key and
ignoring the rest, but such a choice has no gradient. Models that make it are trained
with reinforcement learning :cite:`Mnih.Heess.Graves.ea.2014`, which is
harder, so this book uses the soft form throughout.

Let us carry the scalar example over to vectors before going further, and
keep its score $a(\mathbf{q}, \mathbf{k}) = -\|\mathbf{q} -
\mathbf{k}\|^2$, so that a key scores higher the closer it lies to the
query. Take the query $\mathbf{q} = (1, 1)$ and three keys:
$\mathbf{k}_1 = (1, 1)$, which equals the query, $\mathbf{k}_2 = (2, 1)$ at
distance $1$, and $\mathbf{k}_3 = (0, 0)$ at distance $\sqrt{2}$. The
scores are $0$, $-1$, and $-2$. Exponentiating gives $1$, $0.368$, and
$0.135$, which sum to $1.503$, so the weights are $0.67$, $0.24$, and
$0.09$. With the values $\mathbf{v}_1 = (1, 3)$, $\mathbf{v}_2 = (0, 0)$,
and $\mathbf{v}_3 = (4, 1)$ the output is

$$\mathbf{o} = 0.67\,(1, 3) + 0.24\,(0, 0) + 0.09\,(4, 1) \approx (1.0, 2.1).$$

The first key equals the query, the best match there can be, and still
receives only two thirds of the weight. At these scores the softmax does
not commit fully, and how fully it commits depends on the size of the
scores, not only on their order. Multiply the three scores by ten, to $0$,
$-10$, and $-20$, and the exact match receives $99.995\%$ of the weight. We
will meet this factor again and give it a name. The second value is zero
and adds nothing to the sum, yet the quarter of the weight that it takes
pulls the output from $(1, 3)$ toward the origin. In a soft lookup every
record has a say, and a record that holds nothing still dilutes the
others. :numref:`fig_qkv` draws the computation.

![Soft lookup of the query $\mathbf{q} = (1, 1)$ in three key--value pairs, with keys $(1, 1)$, $(2, 1)$, $(0, 0)$ and values $(1, 3)$, $(0, 0)$, $(4, 1)$. The scores $s_i = -\|\mathbf{q} - \mathbf{k}_i\|^2$ become softmax weights $\alpha_i$ that sum to one (left), and the output $\mathbf{o} \approx (1.0, 2.1)$ is the weighted sum of the values, a point inside the triangle whose corners are the values (right).](../img/mdl-attention-soft-lookup.svg)
:label:`fig_qkv`

Nothing in this rule depends on how many pairs the database holds. The
same score and the same softmax serve three pairs or a thousand, so one
attention layer serves inputs of every length, and it reads all of them
directly rather than a compressed summary. What grows instead is memory. A
recurrent network keeps one state, while attention keeps a key and a value
for every position it may need to read. Each layer of GPT-2 small runs
twelve such lookups side by side, the heads of the chapter introduction,
and the keys and values of one head have 64 entries each. After 1,024
tokens one head holds $2 \times 1{,}024 \times 64 = 131{,}072$ numbers, and
the twelve heads of one layer hold about 1.6 million, where the recurrent
model of the opening kept 768.

## Attention Pooling with Fixed Kernels
:label:`sec_attention-pooling`

Our score so far has had no adjustable part: $-\|\mathbf{q} -
\mathbf{k}\|^2$ fixes, once and for all, how fast similarity fades with
distance. Long before deep learning, statisticians used the same weighted
sum for regression, with the training inputs as keys and the labels as
values, and gave the score one knob. The method is Nadaraya--Watson kernel
regression :cite:`Nadaraya.1964,Watson.1964`. As a regression method it is
simple and somewhat crude. As a first look at attention it is ideal,
because nothing in it is learned and every weight can be plotted. Its
limits show why the rest of the chapter learns the weights.

### Kernels and the Nadaraya--Watson Estimator

A *kernel* measures how similar a query is to a key. We write it
$K(\mathbf{q}, \mathbf{k})$ and keep the letter $\alpha$ for weights that
have been normalized over the keys. A kernel is nonnegative and typically
decreases with the distance between its arguments. The most common choice
is the Gaussian kernel

$$K(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{\|\mathbf{q} - \mathbf{k}\|^2}{2\sigma^2}\right),$$

whose *bandwidth* $\sigma$ sets the distance at which similarity fades. A
key at distance $\sigma$ from the query gets $e^{-1/2} \approx 0.61$ of
the kernel value of an exact match, and a key at distance $3\sigma$ only
about $0.01$. Boxcar and triangular windows, which give no weight at all
beyond a fixed distance, are also in use. We stay with the Gaussian.

Normalizing a kernel over the keys gives the *Nadaraya--Watson estimator*

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{K(\mathbf{q}, \mathbf{k}_i)}{\sum_j K(\mathbf{q}, \mathbf{k}_j)}.$$
:eqlabel:`eq_nadaraya-watson`

This is attention pooling :eqref:`eq_attention_pooling` with the weights
$\alpha(\mathbf{q}, \mathbf{k}_i) = K(\mathbf{q}, \mathbf{k}_i) / \sum_j
K(\mathbf{q}, \mathbf{k}_j)$, which are nonnegative and sum to one. They
are even softmax weights, for any kernel that is strictly positive: the
kernel is the exponential of the score $a = \log K$, and dividing by the
sum is the normalization of :eqref:`eq_softmax_attention`. For the
Gaussian kernel that score is the scaled squared distance $a(\mathbf{q},
\mathbf{k}) = -\|\mathbf{q} - \mathbf{k}\|^2 / 2\sigma^2$. It turns out
that we have met this estimator before. The soft query that opened the
section used the score $-(q-k)^2$, which is this one with $2\sigma^2 = 1$,
so it was Nadaraya--Watson regression with bandwidth $\sigma = 1/\sqrt{2}
\approx 0.71$.

For scalar regression with observations $(x_i, y_i)$, the keys are the
training inputs $x_i$, the values are the labels $y_i$, and the query is
the input at which we want a prediction. The estimate is a weighted
average of the labels of the training inputs near the query. It needs no
training: apart from the bandwidth, the data are the model. If the
bandwidth shrinks at a suitable rate as data accumulate, and the target
function and the density of the inputs are smooth enough, the estimate
converges to the conditional mean $\mathbb{E}[y \mid x]$, the best
predictor under squared error :cite:`mack1982weak`.

### The Bandwidth

To see what the bandwidth does, we fit a one-dimensional regression
problem. We draw $40$ training inputs uniformly
from $[0, 5)$, label them by $y = 2\sin(x) + x + \epsilon$ with standard
Gaussian noise $\epsilon$, and keep a grid of noise-free points for
comparison.

```{.python .input #queries-keys-values-nadaraya-watson-regression-in-action-1}
%%tab pytorch
n = 40
x_train, _ = torch.sort(torch.rand(n) * 5)
y_train = 2 * torch.sin(x_train) + x_train + torch.randn(n)
x_val = torch.arange(0, 5, 0.1)
y_val = 2 * torch.sin(x_val) + x_val
```

```{.python .input #queries-keys-values-nadaraya-watson-regression-in-action-1}
%%tab jax
n = 40
x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
y_train = (2 * jnp.sin(x_train) + x_train
           + jax.random.normal(d2l.get_key(), (n,)))
x_val = jnp.arange(0, 5, 0.1)
y_val = 2 * jnp.sin(x_val) + x_val
```

The estimator computes all query--key distances, applies a Gaussian kernel
of bandwidth $\sigma$, normalizes over the keys, and takes the weighted sum
of the labels. The normalized kernel matrix is the matrix of attention
weights, with one row per query, so we return it as well.

```{.python .input #queries-keys-values-nadaraya-watson-regression-in-action-2}
%%tab pytorch, jax
def nadaraya_watson(x_train, y_train, x_val, sigma):
    dists = x_val[:, None] - x_train[None, :]  # (queries, keys)
    k = d2l.exp(-dists**2 / (2 * sigma**2))
    attention_w = k / k.sum(1, keepdims=True)  # Normalize over the keys
    return attention_w @ y_train, attention_w
```

The two extremes of the bandwidth can be predicted before we run anything.
As $\sigma$ shrinks toward zero, all the weight goes to the training input
nearest the query, and the estimate copies that one noisy label. As
$\sigma$ grows without bound, every weight tends to $1/40$, and the
estimate becomes the mean of all the labels, a horizontal line. We compare
three bandwidths between these extremes, $0.1$, $0.5$, and $2$, each four
to five times the last, against the noise-free function. The dots are the
training data.

```{.python .input #queries-keys-values-nadaraya-watson-regression-1}
%%tab pytorch, jax
sigmas = (0.1, 0.5, 2.0)
estimates = [nadaraya_watson(x_train, y_train, x_val, s)[0] for s in sigmas]
d2l.plot([x_val] * 4 + [x_train], estimates + [y_val, y_train], 'x', 'y',
         legend=[rf'$\sigma$ = {s:g}' for s in sigmas] + ['truth'],
         fmts=('-', 'm--', 'g-.', 'k:', 'C7o'), ylim=(-2.5, 9),
         figsize=(5, 3), legend_loc='upper left')
```

The bandwidth decides the fit. With $\sigma = 0.1$ the estimate chases the
noise of individual observations. With $\sigma = 2$ it averages over most
of the data and flattens toward the global mean, missing both the bump and
the dip of the curve. The middle bandwidth, $\sigma = 0.5$, follows the
noise-free function closely except near the ends of the interval, where
all its neighbors lie on one side of the query and pull the estimate
toward them.

We judged the three fits against the noise-free curve, which no real
problem supplies. How would we choose $\sigma$ without it? As for any
other hyperparameter, we would use held-out error
(:numref:`subsec_generalization-model-selection`). Kernel regression makes
leave-one-out cross-validation cheap, because there is nothing to refit:
the prediction of each training label from all the others uses the same
weight matrix, with its diagonal removed and its rows normalized again.
Exercise :exref:`ex_queries-keys-values-learned-metric` chooses the
bandwidth this way and then learns a whole metric, a first sign that the
similarity itself can be learned.

The bandwidth has a second reading, and it is the one that matters for the
rest of the chapter. The Gaussian weights are the softmax of the scores
$-\tfrac{1}{2}(q - k)^2$ multiplied by $1/\sigma^2$, so the bandwidth acts
only through this one factor in front of every score. A large factor
stretches the gaps between the scores, and the weights approach a one-hot
vector: exact lookup of the nearest key. A small factor shrinks the gaps,
and the weights approach $1/m$: average pooling. The bandwidth is the
sharpness of the softmax, and the two extremes of attention pooling are
the two ends of one dial. Multiplying the scores of the vector example by
ten was such a turn of the dial: it took the exact match from two thirds
of the weight to $99.995\%$. Between $\sigma = 0.1$ and $\sigma = 2$ the
factor changes by $400$. Every attention layer has this dial, whether or
not anyone calls it a bandwidth.

## Attention Weights as a Diagnostic

Because the weights are nonnegative and sum to one, a row of them reads as
a distribution over keys. It tells us which stored pairs a query drew on.
Plotting the weight matrix over (query, key) pairs is therefore the
standard first look at any attention layer, and we will draw such
heatmaps throughout the chapter with `d2l.show_heatmaps`.

The narrow and the wide kernel put weights of very different sizes on
their keys, so each map gets its own color scale. The
queries are the test inputs and the keys are the training inputs, both in
increasing order.

```{.python .input #queries-keys-values-nadaraya-watson-regression-in-action-3}
%%tab pytorch, jax
w_narrow, w_wide = (nadaraya_watson(x_train, y_train, x_val, s)[1]
                    for s in (0.1, 2.0))
# show_heatmaps takes (rows of panels, columns of panels, queries, keys)
d2l.show_heatmaps(d2l.stack([w_narrow, w_wide])[None],
                  xlabel='training inputs (keys)',
                  ylabel='test inputs (queries)',
                  titles=[r'$\sigma$ = 0.1', r'$\sigma$ = 2'],
                  figsize=(7, 3), cbar='each')
```

With the narrow kernel, each query puts nearly all of its weight on the
few keys nearest to it, so the map is a thin diagonal band and the
estimate copies a few noisy labels. The wide kernel's largest weight is
only a few hundredths, not far above the uniform weight $1/40 = 0.025$
that $\sigma \to \infty$ would give, and it covers a broad band of keys,
so neighboring queries see nearly the same average. The two maps show the
two ends of the dial. In both maps the diagonal appears only
because queries and keys are sorted inputs on the same axis.

Here we know what the weights mean, because we chose the kernel. In a
trained network, such a heatmap is a hypothesis about what the layer does,
not a proof. A large weight on a key whose value the next layer ignores
explains nothing. The vector example already holds a small instance: a
heatmap of its query would show a quarter of the attention on the second
record, whose value adds nothing to the sum. A weight measures how well a
key matches, not what its value contributes. Testing such a hypothesis
takes an intervention rather than a picture, and the last section of this
chapter carries one out.

A heatmap also shows what Nadaraya--Watson regression cannot do. In the
narrow map every row is the same Gaussian bump over the input axis,
centered on its own query, and the band has no other structure. The kernel
fixes the shape, the bandwidth, and the space in which distances are
measured, so every query uses the same prescribed notion of similarity. In
a sequence, where the keys belong to tokens, a query may need the token
three positions back, or the one with the same grammatical role, and
neither is a matter of distance in the input space. A learned attention
layer would instead compute queries and keys from the data with trainable
maps, so that the task decides what counts as similar. Which score it
should compute from them is the next question.

## Summary

Attention answers a query with a weighted sum of stored values, called
attention pooling, in which each weight measures how well the query
matches the value's key. A softmax turns any scores into nonnegative
weights that sum to one, so the output lies inside the convex hull of the
values, with exact lookup and average pooling as the two extremes.
Attention reads every stored pair directly, so it needs no fixed-size
summary of its input, and nothing in its rule depends on how many pairs
there are, although its memory does. Nadaraya--Watson regression is
attention with a fixed Gaussian kernel, and its bandwidth is the sharpness
of the softmax: a narrow kernel copies the noise of a few labels, and a
wide one averages the signal away. A heatmap of the weights shows where
each query looks, which is a hypothesis about what a layer computes, not a
proof.

## Exercises

1. **Soft lookup by hand.** Use the database $\{(2, 10), (5, 20)\}$ and
   the score $a(q, k) = -(q - k)^2$ of the opening example.
    1. Compute the answer to the query $q = 3$.
    1. Add the record $(4, 30)$ and answer the query $q = 4$ again. Why
       does the answer move so far from $19.5$?
    1. Return to the two original records and multiply both scores by
       $10$. Answer the query $q = 4$ again, and find the limits of the
       answer as the factor tends to infinity and to zero. Which bandwidth
       $\sigma$ of the Gaussian kernel corresponds to the factor $10$?

1. **An average of two values.** Let the keys $\mathbf{k}_1, \ldots,
   \mathbf{k}_m$ be orthonormal and let the score be $a(\mathbf{q},
   \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$ with the weights
   of :eqref:`eq_softmax_attention`.
    1. For the query $\mathbf{q} = c\,(\mathbf{k}_a + \mathbf{k}_b)$ with
       $c > 0$, compute all $m$ weights and show that the output tends to
       $\tfrac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$ as $c \to \infty$.
    1. How large must $c$ be for the two selected keys to receive at least
       99% of the total weight when $m = 10$?
    1. Now let the keys be orthogonal with $\|\mathbf{k}_b\| =
       2\,\|\mathbf{k}_a\|$. Find a query that again averages
       $\mathbf{v}_a$ and $\mathbf{v}_b$, and state what
       $\mathbf{q} = c\,(\mathbf{k}_a + \mathbf{k}_b)$ returns instead as
       $c \to \infty$.

    *Adapted from Stanford CS224N,
    [Assignment 4](https://web.stanford.edu/class/cs224n/assignments_w25/a4.pdf),
    Problems 1(b) and 1(c).*
   :exlabel:`ex_queries-keys-values-average-two-values`

1. [code] **Effective number of keys.** For the weights $\alpha_i(q)$ of
   one query, define the effective number of keys
   $n_{\textrm{eff}}(q) = 1 / \sum_i \alpha_i(q)^2$.
    1. Let the values be $y_i = g(x_i) + \epsilon_i$ with independent noise
       of variance $s^2$, and hold the keys fixed. Show that the variance of
       the estimate :eqref:`eq_nadaraya-watson` at $q$ is
       $s^2 / n_{\textrm{eff}}(q)$.
    1. Compute $n_{\textrm{eff}}$ at every query in `x_val` from the weights
       that `nadaraya_watson` returns, for each bandwidth in `sigmas`, and
       plot the three curves. Then redraw the noise in `y_train` 200 times
       with `x_train` fixed, and compare the standard deviation of the
       estimate at each query with $1 / \sqrt{n_{\textrm{eff}}(q)}$.

1. [code] **Queries beyond the data.** The estimate
   :eqref:`eq_nadaraya-watson` is a convex combination of the values, and
   the inputs `x_train` lie in the interval $[0, 5)$.
    1. For distinct scalar keys and a Gaussian kernel of fixed bandwidth,
       find the limit of the estimate as $q \to \infty$. Find the same limit
       for the constant kernel.
    1. Evaluate `nadaraya_watson` with $\sigma = 0.5$ at the queries $5,
       5.5, \ldots, 10$ and plot the estimate against $y = 2\sin(x) + x$.
       Explain why no nonnegative kernel and no bandwidth let the estimate
       follow the trend of the truth.
    1. With $\sigma = 0.5$, `nadaraya_watson` returns NaN at the query
       $q = 15$. Explain why in single precision, and compute the weights
       instead as the softmax :eqref:`eq_softmax_attention` of the scores
       $-(q - x_i)^2 / (2\sigma^2)$, so that the estimate is finite at
       every query and every bandwidth.

1. [code] **The gradient of attention.** ● Suppose that $a(\mathbf{q},
   \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$ and that $\mathbf{k}_i =
   \mathbf{v}_i$ for $i = 1, \ldots, m$. Denote by $p(\mathbf{k}_i;
   \mathbf{q})$ the distribution over keys defined by the softmax
   normalization :eqref:`eq_softmax_attention`.
    1. Prove that
       $$\nabla_{\mathbf{q}} \mathop{\textrm{Attention}}(\mathbf{q}, \mathcal{D}) = \textrm{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i].$$
    1. Conclude that this Jacobian is positive semidefinite, and determine
       for which databases it is the zero matrix.
    1. Let $\mathbf{q} = c\,\mathbf{u}$ for a fixed direction $\mathbf{u}$
       for which exactly one key attains the largest score
       $\mathbf{u}^\top \mathbf{k}_i$. Find the limit of the Jacobian as
       $c \to \infty$, and state what it implies for training the query by
       gradient descent.
    1. Verify the identity and the limit with automatic differentiation on
       a random database with $m = 8$ and $d = 4$.

1. [code] **Soft lookup as retrieval.** ● Build a database of $m$ keys
   drawn uniformly from the unit sphere in $\mathbb{R}^{16}$, with value
   $\mathbf{v}_i$ the one-hot indicator of index $i$, and weight them with
   the Gaussian kernel of bandwidth $\sigma$ normalized as
   in :eqref:`eq_nadaraya-watson`, so that the output is the vector of
   weights. Query with a noisy copy $\mathbf{q} = \mathbf{k}_i +
   \boldsymbol{\epsilon}$ of a stored key, with $\boldsymbol{\epsilon} \sim
   \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$.
    1. Show that the largest weight goes to the key nearest the query for
       every $\sigma > 0$, so the bandwidth does not affect which index
       wins. It does affect how much of the weight the winner takes.
    1. Without noise, the weight on the correct key is
       $$\alpha_i = \frac{1}{1 + \sum_{j \neq i} \exp\big(-\|\mathbf{k}_i - \mathbf{k}_j\|^2 / 2\sigma^2\big)}.$$
       Approximate $\mathbf{k}_i^\top \mathbf{k}_j \approx 0$ for random
       unit vectors, so that $\|\mathbf{k}_i - \mathbf{k}_j\|^2 \approx 2$,
       and derive the bandwidth at which $\alpha_i$ falls to $0.9$ for
       $m = 20$ and for $m = 200$. How must $\sigma$ shrink as $m$ grows?
    1. A lookup whose output the next layer can use needs most of the
       weight on one key, not merely the largest share. Call the query
       retrieved when $\alpha_i \geq 0.5$. Sweep $\tau$ from $0$ to $1$ at
       $m = 20$ and $m = 200$, with the bandwidth from (b) for each $m$,
       and plot the retrieval accuracy against $\tau$. Compute the weights as the softmax
       :eqref:`eq_softmax_attention` of the scores
       $-\|\mathbf{q} - \mathbf{k}_j\|^2 / 2\sigma^2$. Explain why a larger
       $m$ lowers the accuracy at fixed $\tau$ although the distance from
       the query to the correct key is unchanged.

1. [extended] **A learned metric.** ● Kernel regression with a learned
   distance :cite:`Weinberger.Tesauro.2007` replaces the fixed similarity
   of :eqref:`eq_nadaraya-watson` by a learned one. Draw $n = 200$ inputs
   $\mathbf{x}_i$ uniformly from $[0, 5]^5$ and labels $y_i =
   2\sin(x_{i1}) + x_{i1} + \epsilon_i$ with standard Gaussian noise, so
   that only the first coordinate matters. Score a query against a key by
   $$a(\mathbf{q}, \mathbf{k}) = -\tfrac{1}{2} \|\mathbf{W}(\mathbf{q} - \mathbf{k})\|^2, \qquad \mathbf{W} = \textrm{diag}(w_1, \ldots, w_5),$$
   normalize as in :eqref:`eq_softmax_attention`, and fit $\mathbf{W}$ to
   the leave-one-out error, in which the estimate at $\mathbf{x}_i$
   excludes the pair $(\mathbf{x}_i, y_i)$. Measure the test error on
   $1000$ fresh inputs against the noise-free function.
    1. Fit the isotropic kernel $w_1 = \cdots = w_5 = 1/\sigma$ by a grid
       search over $\sigma$.
    1. Starting from the isotropic solution, fit the five scales by gradient
       descent. Compare the learned scales with the relevance of each
       coordinate, and the test error with that of the isotropic kernel and
       of an oracle that uses only the first coordinate at its best
       bandwidth.
    1. Compare the leave-one-out errors of the learned and the oracle
       metrics with their test errors, and repeat the comparison at
       $n = 1000$. What does the result say about learning a similarity from
       a finite sample?
    1. A full matrix $\mathbf{W} \in \mathbb{R}^{5 \times 5}$ maps queries
       and keys by the same linear map, whereas learned attention maps them
       by separate matrices $\mathbf{W}_q$ and $\mathbf{W}_k$. State two
       properties of the score $a$ that the shared map guarantees and
       separate maps lose.
   :exlabel:`ex_queries-keys-values-learned-metric`

[Discussions](https://d2l.discourse.group/t/36768)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.1]{.kicker}

Queries, Keys, and Values<br>
**soft lookup · softmax weights · Nadaraya–Watson regression · the bandwidth as sharpness**
:::
:::

::: {.slide title="A 768-Number State Must Answer for 1,024 Tokens"}
[The problem]{.kicker}

A recurrent language model with the width of GPT-2 small reads 1,024
tokens.

- Everything it knows about them sits in a state of 768 numbers.
- The state is written before the model knows which question comes next.
- The token representations themselves hold $1{,}024 \times 768 \approx
  786{,}000$ numbers.

::: {.d2l-note .warn}
Can a layer keep every position and read what it needs, by content, when
it needs it?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute a soft lookup by hand, for scalar and for vector keys: scores,
  softmax weights, output;
- say why the rule does not depend on the number of stored pairs, and
  what does grow;
- recognize Nadaraya–Watson regression as attention with a fixed kernel;
- read the bandwidth as the sharpness of the softmax and predict the fit
  and the heatmap it gives.

**Idea:** answer a query with an average of stored values, weighted by how
well each key matches.

**Running example:** one attention head of GPT-2 small, with keys and
values of width 64 and up to 1,024 stored pairs. A *head* is one such
lookup; each GPT-2 layer runs twelve side by side.
:::

::: {.slide title="Attention Looks Up Values by Content"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **Queries, Keys, and Values** | **attention as a soft lookup** |
| Attention Scoring and Masking | scaled dot products, masks |
| Self-Attention and Multiple Heads | a sequence reads itself; several heads |
| Positional Information | where a token sits |
| The Cost of Attention | quadratic cost, fused kernels |
| Windowed and Linear Attention | linear cost, and what it gives up |
| Attention Circuits | what trained heads compute |
:::

::: {.col .narrow}
- **Uses:** the softmax; the fixed-size state of an RNN.
- **Skips:** learned scores (next deck), heads, positions.
:::
:::
:::

::: {.slide title="An Exact Query Either Matches a Key or Returns Nothing"}
The scalar database $\mathcal{D} = \{(2, 10), (5, 20)\}$ of
(key, value) pairs:

- query $q = 5$ matches key 5 and returns 20;
- query $q = 4$ matches no key and returns nothing;
- a small change of the query changes the answer abruptly, so there is
  no gradient to learn from.

::: {.d2l-note}
We want an answer that moves smoothly with the query.
:::
:::

::: {.slide title="Soft Lookup Answers q = 4 with 19.5"}
[Worked example]{.kicker}

Score every key by $a(q, k) = -(q - k)^2$, then normalize with a softmax.

| key $k$ | value | score | weight |
|---|---|---|---|
| 2 | 10 | $-4$ | $e^{-4}/(e^{-4}+e^{-1}) \approx 0.05$ |
| 5 | 20 | $-1$ | $\approx 0.95$ |

Answer: $0.05 \cdot 10 + 0.95 \cdot 20 \approx 19.5$.

::: {.d2l-note .rule}
Every value contributes; the nearest key contributes most.
:::
:::

::: {.slide title="Attention Is a Weighted Sum of Values"}
Query $\mathbf{q} \in \mathbb{R}^d$; $m$ pairs of a key
$\mathbf{k}_i \in \mathbb{R}^d$ and a value
$\mathbf{v}_i \in \mathbb{R}^{d_v}$:

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\, \mathbf{v}_i.$$

- $\mathcal{D}$ = the $m$ pairs; weights $\alpha(\mathbf{q}, \mathbf{k}_i)
  \ge 0$ that sum to one.
- Keys are compared with the query; values are returned.
- The weighted sum is called *attention pooling*.
:::

::: {.slide title="Softmax Turns Any Score into Weights"}
$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}$$

- $a(\mathbf{q}, \mathbf{k})$ = score of the query against a key.
- Nonnegative and summing to one: the output is a convex combination of
  the values.
- One weight 1, the rest 0: exact lookup. All weights $1/m$: average
  pooling.
- Only score differences matter: adding a constant to every score
  changes nothing.
- Differentiable: gradients reach the query, the keys, and the values.
:::

::: {.slide title="An Exact Match Gets Only Two Thirds of the Weight"}
[Worked example]{.kicker}

Query $\mathbf{q} = (1, 1)$, score $s = -\|\mathbf{q} - \mathbf{k}\|^2$:

| key | $s$ | $e^{s}$ | weight | value |
|---|---|---|---|---|
| $(1, 1)$ | $0$ | $1$ | $0.67$ | $(1, 3)$ |
| $(2, 1)$ | $-1$ | $0.368$ | $0.24$ | $(0, 0)$ |
| $(0, 0)$ | $-2$ | $0.135$ | $0.09$ | $(4, 1)$ |

$\mathbf{o} = 0.67\,(1, 3) + 0.24\,(0, 0) + 0.09\,(4, 1) \approx (1.0, 2.1)$

- The exact match gets only $0.67$. Scores times ten ($0, -10, -20$):
  it gets $99.995\%$.
- The zero value adds nothing, yet takes a quarter of the weight.

::: {.d2l-note .rule}
Every record has a say, and how fully the softmax commits depends on the
size of the scores.
:::
:::

::: {.slide title="The Output Lies Inside the Triangle of Values"}
![Scores, weights that sum to one, and the output inside the triangle of values.](../img/mdl-attention-soft-lookup.svg){width=88%}

::: {.d2l-note}
Weights $\ge 0$ that sum to one: the output is a convex combination of the
values.
:::
:::

::: {.slide title="Check: q = 3.5"}
[Check]{.kicker}

Same database $\{(2, 10), (5, 20)\}$, same score $-(q - k)^2$. What does
the soft lookup return for $q = 3.5$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Both scores are $-2.25$, so both weights are $0.5$ and the
answer is $15$, the average. A query halfway between two keys cannot
prefer either one.
:::
:::
:::

::: {.slide title="The Rule Is Fixed, the Memory Grows with the Context"}
One GPT-2-small head at a context of 1,024 tokens:

- query width 64; 1,024 keys of width 64, 1,024 values of width 64;
- the score, the softmax and the weighted sum work for any number of
  pairs $m$;
- the output always has width 64.

::: {.d2l-note .rule}
Attention stores $2 \times 1{,}024 \times 64 = 131{,}072$ numbers for this
head, about 1.6 million for the twelve heads of one layer, where the RNN
keeps 768.
:::
:::

::: {.slide title="Nadaraya–Watson Regression Is Attention with a Fixed Kernel"}
[1964]{.kicker}

Gaussian kernel of bandwidth $\sigma$:
$K(q, k) = \exp\big(-(q - k)^2 / 2\sigma^2\big)$.

$$f(q) = \sum_i y_i \frac{K(q, x_i)}{\sum_j K(q, x_j)}$$

- keys = training inputs $x_i$; values = labels $y_i$; query = where to
  predict;
- no training: apart from $\sigma$, the data are the model.

::: {.d2l-note}
The lookup that gave 19.5 was Nadaraya–Watson with $\sigma = 1/\sqrt{2}$.
:::
:::

::: {.slide title="The Estimator Is Four Lines and Returns Its Weights"}
Distances → Gaussian kernel of bandwidth $\sigma$ → normalize over the
keys → weighted sum of the labels.

@queries-keys-values-nadaraya-watson-regression-in-action-2

The normalized kernel matrix is the attention-weight matrix.
:::

::: {.slide title="The Bandwidth Decides the Fit"}
40 noisy points from $y = 2\sin(x) + x + \epsilon$. Predicted extremes:
$\sigma \to 0$ copies the nearest label, $\sigma \to \infty$ gives the mean.

@!queries-keys-values-nadaraya-watson-regression-1

- $\sigma = 0.1$ chases noise; $\sigma = 2$ flattens; $\sigma = 0.5$ fits
  except near the ends.
- Without the true curve: choose $\sigma$ by leave-one-out error. Drop the
  diagonal of the weight matrix, renormalize the rows; nothing to refit.
:::

::: {.slide title="The Bandwidth Is the Sharpness of the Softmax"}
$$\alpha_i = \mathrm{softmax}_i\Big(\frac{1}{\sigma^2} \cdot \big(-\tfrac{1}{2}(q - k_i)^2\big)\Big)$$

- $1/\sigma^2$ large: score gaps stretch, weights near one-hot, exact
  lookup;
- $1/\sigma^2$ small: gaps shrink, weights near $1/m$, average pooling;
- from $\sigma = 0.1$ to $\sigma = 2$ the factor changes by 400;
- scores times ten in the vector example: exact match from $0.67$ to
  $99.995\%$.

::: {.d2l-note .rule}
Every attention layer has this dial. The Transformer sets it with
$1/\sqrt{d}$, $d$ = key width (next deck).
:::
:::

::: {.slide title="Check: Halve the Bandwidth"}
[Check]{.kicker}

Two keys get weights in the ratio $2 : 1$. The bandwidth $\sigma$ is
halved. What is the ratio now?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The factor $1/\sigma^2$ grows fourfold, so every score gap
grows fourfold. The ratio $e^{\Delta} = 2$ becomes $e^{4\Delta} = 2^4 =
16 : 1$.
:::
:::
:::

::: {.slide title="Narrow Kernels Read a Few Keys, Wide Kernels Read Many"}
Rows: test inputs (queries). Columns: training inputs (keys), sorted.
Each map has its own color scale.

@!queries-keys-values-nadaraya-watson-regression-in-action-3

- $\sigma = 0.1$: a thin band; each query uses a handful of nearby keys.
- $\sigma = 2$: broad rows; the largest weight is a few hundredths, not
  far above the uniform $1/40 = 0.025$.
:::

::: {.slide title="Heatmaps Suggest, They Do Not Prove"}
- A row of weights is a distribution over keys: which pairs did the
  query use?
- Here we chose the kernel, so we know what the weights mean.
- In a trained network a large weight on a value that later layers
  ignore explains nothing.
- Vector example: the record with value $(0, 0)$ takes $0.24$ of the
  weight and adds nothing to the sum. A weight measures the match of a
  key, not what its value contributes.
- The last deck of this chapter tests such readings by intervention.
:::

::: {.slide title="A Fixed Kernel Gives Every Query the Same Similarity"}
Nadaraya–Watson fixes the kernel shape, the bandwidth, and the space in
which distances are measured.

- Every row of the narrow heatmap is the same Gaussian bump, centered on
  its query.
- In a sequence, a query cannot ask for "the token three positions back"
  or "the subject of this verb": neither is a distance between inputs.
- Learned attention computes queries and keys with trainable maps, so the
  task decides what counts as similar.
:::

::: {.slide title="Check: A Query Far from the Data"}
[Check]{.kicker}

With $\sigma = 0.5$ and training inputs in $[0, 5)$, the estimator above
returns NaN at $q = 15$ in single precision. Why, and what fixes it?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Every kernel value is below $e^{-200}$, which underflows to
0 (float32 stops near $e^{-103}$), so the normalization computes $0/0$.
Computing the weights as a softmax of the scores $-(q - x_i)^2/2\sigma^2$
subtracts the largest score first and stays finite.
:::
:::
:::

::: {.slide title="Attention Is a Soft Lookup Whose Sharpness Is One Factor"}
[Wrap-up]{.kicker}

- $\textrm{Attention}(\mathbf{q}, \mathcal{D}) = \sum_i \alpha(\mathbf{q},
  \mathbf{k}_i)\, \mathbf{v}_i$, softmax weights: a point inside the hull
  of the values.
- $\{(2, 10), (5, 20)\}$: $q = 4$ gives 19.5, $q = 3.5$ gives 15. Three
  keys in the plane: weights 0.67, 0.24, 0.09; output $(1.0, 2.1)$.
- Fixed rule, growing memory: 131,072 numbers for one GPT-2-small head
  at 1,024 tokens.
- Nadaraya–Watson: $1/\sigma^2$ scales the scores; $\sigma = 0.1$ chases
  noise, $\sigma = 2$ flattens.

**Next:** learned scores, the factor $1/\sqrt{d}$, and masks.
:::
