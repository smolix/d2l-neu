# Queries, Keys, and Values
:label:`sec_queries-keys-values`

A recurrent neural network (:numref:`sec_rnn`) compresses everything it has
read into a fixed-size state vector before knowing which details a later
prediction will require. For example, a recurrent model with hidden dimension
768 reading 1,024 tokens must summarize all 1,024 positions in 768 numbers.

A database avoids premature compression by storing records verbatim and
deciding what to retrieve only when a query arrives. However, an exact
database lookup is discontinuous: given key--value pairs
$\mathcal{D}=\{(2,10),(5,20)\}$, the exact query $q=5$ returns $20$, whereas
$q=4$ matches no key. To train with gradient descent, we replace hard lookup
with a *soft* lookup that scores every key—say, by negative squared distance
$a(q,k)=-(q-k)^2$—and converts the scores into convex weights via a softmax.
At $q=4$, the scores for keys $2$ and $5$ are $-4$ and $-1$, giving weights
$e^{-4}/(e^{-4}+e^{-1}) \approx 0.05$ and $0.95$, and a weighted value
$0.05 \cdot 10 + 0.95 \cdot 20 \approx 19.5$. The output varies smoothly
with the query, scales to any number of stored records, and requires no
fixed-size bottleneck prior to querying.

This differentiable key--value lookup is the *attention mechanism*
:cite:`Bahdanau.Cho.Bengio.2014`. We first formulate attention pooling for
vector queries, keys, and values, and then examine its classical
nonparametric precursor, Nadaraya--Watson kernel regression, where a single
bandwidth parameter controls the sharpness of the lookup.

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

Let $\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1),
\ldots, (\mathbf{k}_m, \mathbf{v}_m)\}$ be a database of $m$ key--value pairs
with keys $\mathbf{k}_i \in \mathbb{R}^{d}$ and values $\mathbf{v}_i \in
\mathbb{R}^{d_v}$. Given a query $\mathbf{q} \in \mathbb{R}^{d}$, *attention
pooling* returns a weighted sum of the values:

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$
:eqlabel:`eq_attention_pooling`

where $\alpha(\mathbf{q},\mathbf{k}_i) \ge 0$ and $\sum_{i=1}^m
\alpha(\mathbf{q},\mathbf{k}_i) = 1$. The output is therefore a convex
combination of the values, interpolating between exact lookup (a one-hot
weight vector) and average pooling (uniform weights $1/m$).

To map arbitrary real-valued compatibility scores $a(\mathbf{q},
\mathbf{k}_i)$ to convex weights, we apply the softmax function
(:numref:`sec_softmax`):

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}.$$
:eqlabel:`eq_softmax_attention`

Because $\alpha_i = 1 / \big(1 + \sum_{j \neq i} e^{a_j - a_i}\big)$, the
weights depend only on score differences $a_j - a_i$ and are invariant to
adding a constant across all keys. Both the softmax and the linear
combination are differentiable with respect to $\mathbf{q}$, $\mathbf{k}_i$,
and $\mathbf{v}_i$.

As a concrete vector example, take $a(\mathbf{q}, \mathbf{k}) = -\|\mathbf{q}
- \mathbf{k}\|^2$ with query $\mathbf{q} = (1, 1)$, keys $\mathbf{k}_1 = (1,
1)$, $\mathbf{k}_2 = (2, 1)$, $\mathbf{k}_3 = (0, 0)$, and values
$\mathbf{v}_1 = (1, 3)$, $\mathbf{v}_2 = (0, 0)$, $\mathbf{v}_3 = (4, 1)$.
The scores are $0$, $-1$, and $-2$, yielding softmax weights
$(1, e^{-1}, e^{-2}) / (1 + e^{-1} + e^{-2}) \approx (0.67, 0.24, 0.09)$ and
output

$$\mathbf{o} = 0.67\,(1, 3) + 0.24\,(0, 0) + 0.09\,(4, 1) \approx (1.0, 2.1).$$

Notice two properties illustrated in :numref:`fig_qkv`. First, even though
$\mathbf{k}_1$ matches $\mathbf{q}$ exactly, it receives only two-thirds of
the weight; scaling the scores by a factor of $10$ (to $0, -10, -20$)
concentrates $99.995\%$ of the weight on $\mathbf{k}_1$. Thus the magnitude
of the scores governs how sharply the softmax commits. Second, although
$\mathbf{v}_2 = (0, 0)$ adds zero to the sum, its $0.24$ weight still pulls
the output toward the origin by diluting the other values.

![Soft lookup of the query $\mathbf{q} = (1, 1)$ in three key--value pairs, with keys $(1, 1)$, $(2, 1)$, $(0, 0)$ and values $(1, 3)$, $(0, 0)$, $(4, 1)$. The scores $s_i = -\|\mathbf{q} - \mathbf{k}_i\|^2$ become softmax weights $\alpha_i$ that sum to one (left), and the output $\mathbf{o} \approx (1.0, 2.1)$ is the weighted sum of the values, a point inside the triangle whose corners are the values (right).](../img/mdl-attention-soft-lookup.svg)
:label:`fig_qkv`

The attention rule :eqref:`eq_attention_pooling` applies unchanged to any
number of records $m$, allowing a single layer to process sequences of
arbitrary length without compressing them into a fixed-size state. The cost
shifts to memory: whereas a recurrent layer of width 768 keeps 768 numbers,
one 12-head attention layer of GPT-2 small ($d = 64$ per head) over 1,024
tokens stores $2 \times 1{,}024 \times 64 = 131{,}072$ key and value entries
per head, or roughly $1.6 \times 10^6$ numbers across all 12 heads.

## Attention Pooling with Fixed Kernels
:label:`sec_attention-pooling`

Long before deep learning, *Nadaraya--Watson kernel regression*
:cite:`Nadaraya.1964,Watson.1964` used :eqref:`eq_attention_pooling` for
nonparametric regression, treating training inputs as keys, training labels
as values, and a fixed similarity kernel as the scoring rule.

### Kernels and the Nadaraya--Watson Estimator

Given a nonnegative similarity kernel $K(\mathbf{q}, \mathbf{k})$, such as
the Gaussian kernel with *bandwidth* $\sigma > 0$,

$$K(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{\|\mathbf{q} - \mathbf{k}\|^2}{2\sigma^2}\right),$$

normalizing across keys yields the *Nadaraya--Watson estimator*:

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{K(\mathbf{q}, \mathbf{k}_i)}{\sum_j K(\mathbf{q}, \mathbf{k}_j)}.$$
:eqlabel:`eq_nadaraya-watson`

For any strictly positive kernel, :eqref:`eq_nadaraya-watson` is softmax
attention :eqref:`eq_softmax_attention` with score $a(\mathbf{q}, \mathbf{k})
= \log K(\mathbf{q}, \mathbf{k})$. For the Gaussian kernel, $a(\mathbf{q},
\mathbf{k}) = -\|\mathbf{q} - \mathbf{k}\|^2 / (2\sigma^2)$, which recovers
our opening example when $2\sigma^2 = 1$. Under regularity conditions on the
density and target function, as $m \to \infty$ and $\sigma \to 0$ at an
appropriate rate, $f(\mathbf{q})$ converges to the conditional expectation
$\mathbb{E}[y \mid \mathbf{q}]$ :cite:`mack1982weak`.

### The Bandwidth

To see how the bandwidth $\sigma$ controls the estimator, we draw $n = 40$
noisy training observations from $y = 2\sin(x) + x + \epsilon$ with $\epsilon
\sim \mathcal{N}(0, 1)$ on $[0, 5)$ and evaluate predictions on a dense grid.

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

The implementation computes all pairwise query--key distances, evaluates the
Gaussian kernel, normalizes each query row, and multiplies by `y_train`.

```{.python .input #queries-keys-values-nadaraya-watson-regression-in-action-2}
%%tab pytorch, jax
def nadaraya_watson(x_train, y_train, x_val, sigma):
    dists = x_val[:, None] - x_train[None, :]  # (queries, keys)
    k = d2l.exp(-dists**2 / (2 * sigma**2))
    attention_w = k / k.sum(1, keepdims=True)  # Normalize over the keys
    return attention_w @ y_train, attention_w
```

As $\sigma \to 0$, all weight concentrates on the nearest training point
(1-nearest-neighbor interpolation); as $\sigma \to \infty$, all weights
approach $1/n$ (the global sample mean). We plot $\sigma \in \{0.1, 0.5,
2.0\}$ below.

```{.python .input #queries-keys-values-nadaraya-watson-regression-1}
%%tab pytorch, jax
sigmas = (0.1, 0.5, 2.0)
estimates = [nadaraya_watson(x_train, y_train, x_val, s)[0] for s in sigmas]
d2l.plot([x_val] * 4 + [x_train], estimates + [y_val, y_train], 'x', 'y',
         legend=[rf'$\sigma$ = {s:g}' for s in sigmas] + ['truth'],
         fmts=('-', 'm--', 'g-.', 'k:', 'C7o'), ylim=(-2.5, 9),
         figsize=(5, 3), legend_loc='upper left')
```

A narrow bandwidth ($\sigma = 0.1$) overfits individual noisy labels, while a
wide bandwidth ($\sigma = 2.0$) oversmooths toward a horizontal line. The
intermediate value $\sigma = 0.5$ tracks the true curve closely except near
the boundaries, where all neighbors lie on one side. In practice $\sigma$ can
be chosen by leave-one-out cross-validation, which requires only zeroing the
diagonal of the weight matrix and renormalizing its rows (Exercise
:exref:`ex_queries-keys-values-learned-metric`).

More broadly, $1/\sigma^2$ acts as an inverse temperature multiplying the
raw scores $-\tfrac{1}{2}(q - k)^2$ before the softmax. Scaling scores up
stretches pairwise gaps and drives the softmax toward a one-hot lookup;
scaling scores down compresses gaps and drives the softmax toward uniform
averaging. We will see this same temperature scaling in dot-product attention
(:numref:`sec_attention-scoring-functions`).

## Attention Weights as a Diagnostic

Because each row of the attention matrix $\mathbf{A}$ is a probability
distribution over keys, plotting $\mathbf{A}$ as a heatmap shows which keys
each query consults.

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

At $\sigma = 0.1$, weight concentrates in a narrow diagonal band around the
nearest training inputs; at $\sigma = 2.0$, weight spreads broadly across the
dataset (peaking near $1/40 = 0.025$).

Two caveats apply when inspecting attention heatmaps in deep networks:

1. **Weights show where a layer looks, not what it transmits:** A key can
   receive a large attention weight even if its value vector is zero (as with
   $\mathbf{v}_2 = (0, 0)$ in our vector example) or lies in the null space
   of downstream projections. Verifying a head's causal role requires
   interventional experiments (:numref:`sec_what-attention-computes`).
2. **Fixed kernels enforce a static similarity metric:** In
   :eqref:`eq_nadaraya-watson`, every query uses the same Euclidean distance
   in input space. In sequence modeling, however, relevance is context- and
   task-dependent—a verb may need its syntactic subject several tokens away.
   Learned attention layers therefore project inputs into trainable query,
   key, and value spaces before scoring.

## Summary

* Attention pooling computes a convex combination of value vectors
  $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i$, where softmax
  weights $\alpha(\mathbf{q}, \mathbf{k}_i)$ interpolate smoothly between
  exact lookup and average pooling.
* Unlike a fixed-size recurrent state, attention accesses all stored
  key--value pairs directly; the rule is independent of the number of pairs
  $m$, while memory scales linearly with $m$.
* Nadaraya--Watson kernel regression is attention pooling with a fixed
  similarity kernel; its bandwidth $\sigma$ acts as a softmax temperature
  controlling the bias--variance and sharpness trade-off.
* Attention heatmaps visualize query--key alignments, but learned projections
  and scoring functions are required to go beyond fixed input-space distance.

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
**Answer.** Every distance is at least 10, so every kernel value
$\exp(-d^2/0.5) \le e^{-200}$ underflows to $0$ in float32, and $0/0$ is
NaN. Compute the weights with `softmax(-dists**2 / (2 * sigma**2))`,
which subtracts the row maximum before exponentiating.
:::
:::
:::

::: {.slide title="Soft Lookup, Softmax Sharpness, and Why Weights Must Be Learned"}
[Wrap-up]{.kicker}

- Attention answers a query with a softmax-weighted sum of stored values:
  a convex combination, from exact lookup to average pooling.
- The rule does not depend on the number of pairs; the memory grows with
  it.
- Nadaraya–Watson regression is attention with a fixed Gaussian kernel;
  its bandwidth is the sharpness of the softmax.
- A heatmap of weights shows where each query looks, a hypothesis about
  what the layer does, not a proof.

**Next:** a score that a network can learn, and masks for the pairs it
must ignore.
:::
