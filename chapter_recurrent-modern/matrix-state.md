# Matrix-State Recurrences
:label:`sec_matrix-state`

The selective models of :numref:`sec_mamba` decide, token by token, what to
keep and for how long. They leave open what the kept numbers can
represent. Much of what a language model must remember comes in pairs, a
name together with what it stands for. A program defines `rate = 0.05`
near its top and uses `rate` two hundred lines later. A story introduces
Mira as the ship's engineer, and three pages on, "Mira" must still bring
"engineer" with it. To predict such a token, a model must store a *pair*
and later retrieve the second half of the pair when the first half
reappears. This is *associative recall*, and transformers do it with
attention: the current token's query is compared with the keys of all
earlier tokens, and the values of the best matches are returned. Can a
recurrent state of fixed size store pairs too, and how many?

We have met a state that might do this twice, and both times it was a
matrix. The linear attention of :numref:`sec_windowed-linear-attention`
carries one from token to token, and a selective state space model keeps
$N$ numbers for each of its channels. This section makes the matrix the
subject. A pair is written by adding the outer product of its value and its
key, and read by multiplying the matrix by a query. Let us start with a
memory small enough to compute by hand and then find a law for how many
pairs a larger one can hold. A usable layer needs more: a decay that lets
old pairs fade, a way to train the decayed memory with matrix products,
and evidence that a trained network fills its memory the way the law
predicts. One limitation survives all of this, and it is where the next
section begins.

```{.python .input #matrix-state-matrix-state-recurrences}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #matrix-state-matrix-state-recurrences}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## An Associative Memory

### Writing and Reading Pairs

The simplest memory for pairs is a sum: it adds each new pair to what it
holds and reads a value back with its key. A small example shows how this
works and where it fails. Start with two keys in $\mathbb{R}^2$,
$\mathbf{k}_1 = (1, 0)^\top$ and $\mathbf{k}_2 = (0, 1)^\top$, with the
values $v_1 = 3$ and $v_2 = 5$. Store both in one row vector, the sum of
each value times its key, $\mathbf{S} = v_1 \mathbf{k}_1^\top + v_2
\mathbf{k}_2^\top = (3, 5)$. Multiplying by a key reads its value back:
$\mathbf{S}\mathbf{k}_1 = 3$ and $\mathbf{S}\mathbf{k}_2 = 5$. Now store a
third pair, $\mathbf{k}_3 = (1, 1)^\top / \sqrt{2}$ with $v_3 = 7$. The
memory becomes $(3 + 7/\sqrt 2, 5 + 7/\sqrt 2) \approx (7.95, 9.95)$, and
reading $\mathbf{k}_1$ now returns 7.95 instead of 3. Two dimensions hold
two pairs exactly; a third key must overlap with the others, and its value
leaks into theirs.

With keys $\mathbf{k}_t \in \mathbb{R}^{d_k}$ and values $\mathbf{v}_t \in
\mathbb{R}^{d_v}$, the memory becomes a matrix $\mathbf{S}_t \in
\mathbb{R}^{d_v \times d_k}$, with one row per value coordinate and one
column per key coordinate, so that a read is the product of the matrix with
a query $\mathbf{q}_t \in \mathbb{R}^{d_k}$ (the linear attention of
:numref:`sec_windowed-linear-attention` stores the transpose). Each step
writes one pair and reads the memory:

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top, \qquad
\mathbf{y}_t = \mathbf{S}_t \mathbf{q}_t.
$$
:eqlabel:`eq_ms-recurrence`

The outer product $\mathbf{v}_t \mathbf{k}_t^\top$ is a matrix of rank one
that maps the direction $\mathbf{k}_t$ to $\mathbf{v}_t$, and the state is
the sum of these maps. The keys, values, and queries are linear projections
of the input, as in attention, so the network learns what to store under
which key and what to look up.

### Linear Attention

Unrolling :eqref:`eq_ms-recurrence` from $\mathbf{S}_0 = \mathbf{0}$ shows
that the memory computes a form of attention:

$$\mathbf{y}_t = \Big(\sum_{s \le t} \mathbf{v}_s \mathbf{k}_s^\top\Big)
\mathbf{q}_t = \sum_{s \le t} (\mathbf{k}_s^\top \mathbf{q}_t)\,
\mathbf{v}_s.$$

The right side is causal attention with the score $\mathbf{k}_s^\top
\mathbf{q}_t$ and no softmax: every earlier value, weighted by how well its
key matches the query. It is the linear attention of
:numref:`sec_windowed-linear-attention` with the identity as the feature
map and without the normalizer; the layers of this chapter normalize their
output instead. Its cost, linear in the length, was counted there
:cite:`Katharopoulos.Vyas.Pappas.ea.2020`. Read from the right, the
computation is an attention layer; read from the left, it is a recurrent
network whose state is a memory of pairs.

### Interference

How many pairs can $\mathbf{S}$ hold? Store $n$ pairs, $\mathbf{S} =
\sum_{i=1}^n \mathbf{v}_i \mathbf{k}_i^\top$, with keys of unit length, and
read key $j$:

$$
\mathbf{S}\mathbf{k}_j = \mathbf{v}_j + \sum_{i \neq j}
(\mathbf{k}_i^\top \mathbf{k}_j)\, \mathbf{v}_i.
$$
:eqlabel:`eq_ms-retrieval-error`

The read returns the stored value plus every other value, weighted by the
overlap of its key with the target key. Only mutually orthogonal keys are
read back without error, and $\mathbb{R}^{d_k}$ holds at most $d_k$ of
them. For more pairs, or for keys that are not orthogonal, interference is
unavoidable. How large it is depends on the keys; for random keys we can
compute it.

**Proposition.** Let the keys $\mathbf{k}_1, \ldots, \mathbf{k}_n$ be
independent and uniformly distributed on the unit sphere of
$\mathbb{R}^{d_k}$, and let the values be unit vectors chosen independently
of the keys. Then the expected squared read error at every stored key is

$$
\mathbb{E}\,\big\|\mathbf{S}\mathbf{k}_j - \mathbf{v}_j\big\|^2 =
\frac{n - 1}{d_k}.
$$
:eqlabel:`eq_ms-capacity-law`

**Proof.** The error is $\sum_{i \neq j} (\mathbf{k}_i^\top \mathbf{k}_j)\,
\mathbf{v}_i$. Its squared norm is a double sum over $i$ and $l$. For $i
\neq l$, the expected cross term vanishes: given $\mathbf{k}_j$, the
overlaps $\mathbf{k}_i^\top \mathbf{k}_j$ and $\mathbf{k}_l^\top
\mathbf{k}_j$ are independent with mean zero. What remains is $\sum_{i \neq
j} \mathbb{E}[(\mathbf{k}_i^\top \mathbf{k}_j)^2]$, using $\|\mathbf{v}_i\|
= 1$. A random unit vector has the same expected squared coordinate in every
direction, and these $d_k$ expectations sum to $\|\mathbf{k}_i\|^2 = 1$, so
each term is $1/d_k$. There are $n - 1$ terms. $\blacksquare$

The error grows in proportion to the number of stored pairs and falls in
proportion to the key dimension. Once $n$ approaches $d_k$, the
interference is as large as the signal, whose squared norm is one. Time
does not appear: a pair written long ago is read back as well as one
written a moment ago. Memories are lost because other pairs interfere with
them, not because they are old.

The bold assumption in the proposition is that the keys are random and
independent. A trained network chooses its keys, and the keys of related
tokens are anything but independent. We check the formula with random
pairs first, and then break the assumption on purpose. The values come
from a codebook of 256 random unit vectors of the same width as the keys,
$d_v = d_k$, so we can also ask whether the codebook entry closest to the
read is the stored one. The fraction of reads for which it is, we call the
*recall*. The last curve uses correlated keys: every key shares a common
direction with weight $\rho = 0.5$. The computation needs only NumPy.

```{.python .input #matrix-state-interference-1}
%%tab pytorch, jax
def matrix_memory(d_k, nums, rho=0, num_values=256, trials=20, seed=0):
    """Read error and recall of S = sum_i v_i k_i^T over random unit keys."""
    rng = np.random.default_rng(seed)
    unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
    codebook = unit(rng.standard_normal((num_values, d_k)))
    errors, recalls = [], []
    for n in nums:
        error = recall = 0
        for _ in range(trials):
            K = unit(rng.standard_normal((n, d_k)))  # Keys as rows
            shared = unit(rng.standard_normal(d_k))  # A common direction
            K = unit(np.sqrt(1 - rho**2) * K + rho * shared)
            ids = rng.integers(0, num_values, n)
            S = codebook[ids].T @ K  # Sum of v_i k_i^T
            reads = K @ S.T  # Row j is S k_j
            error += ((reads - codebook[ids]) ** 2).sum(1).mean() / trials
            recall += ((reads @ codebook.T).argmax(1) == ids).mean() / trials
        errors.append(error)
        recalls.append(recall)
    return errors, recalls
```

```{.python .input #matrix-state-interference-2}
%%tab pytorch, jax
nums = [8, 16, 32, 64, 128, 256, 512]
results = {f'dₖ = {d_k}': matrix_memory(d_k, nums) for d_k in (32, 64, 128)}
results['dₖ = 64, ρ = 0.5'] = matrix_memory(64, nums, rho=0.5)
predictions = [[(n - 1) / d_k for n in nums] for d_k in (32, 64, 128)]
d2l.plot(nums, [err for err, _ in results.values()] + predictions,
         'stored pairs n', 'squared read error',
         legend=list(results) + ['$(n-1)/d_k$'], xscale='log', yscale='log',
         fmts=('-o', '-o', '-o', '-s', 'k:', 'k:', 'k:'), figsize=(5, 3.5))
print('recall at n =    ', ' '.join(f'{n:4d}' for n in nums))
for name, (_, recalls) in results.items():
    print(f'{name:17s}', ' '.join(f'{r:4.2f}' for r in recalls))
```

The measured errors lie on the predicted lines $(n-1)/d_k$ at all three
widths. The recall table holds a surprise. With $d_k = 64$, practically
every pair is recovered up to $n = 64$, 96% at $n = 128$, 72% at
$n = 256$, and 37% at $n = 512$. At $n = 64$ the expected squared error is
already about as large as the squared length of the signal, and yet the
table shows a recall of 1.00. How can that be? The error points in a
random direction of the $d_v$-dimensional value space. To decide which
codebook entry was stored, only the component of the error along each
candidate entry matters, and that component has a variance of about
$n/(d_k d_v)$. At $n = d_k = d_v = 64$ its typical size is an eighth of the
signal, small enough to pick the right entry out of 256.

Recall at a given level therefore depends mainly on $n / (d_k d_v)$, and
the squared error, which does not see $d_v$, cannot predict it alone. In
the cell above $d_v = d_k$, so doubling $d_k$ gives the state four times
the entries, and the table shows four times the pairs at the same recall:
0.96 at 32 pairs for $d_k = 32$ and at 128 pairs for $d_k = 64$, 0.94 at
512 pairs for $d_k = 128$. With values as wide as keys, then, the number of
pairs that a matrix memory can recall grows in proportion to $d_k d_v$, the
number of entries of its state. This codebook holds about one pair for
every 32 entries at a recall of 95%. A cell that keeps $d_v = d_k$ cannot,
of course, tell $d_k d_v$ from $d_k^2$. The variance argument says that
both widths count as long as the keys are not much narrower than the
values, and Exercise :exref:`ex_ms-value-width` separates the two.

Correlated keys do worse, and a wider key does not repair them. With a
shared direction of weight $\rho$, the overlap of two keys is about $\rho^2$
plus a random part of variance about $(1-\rho^2)^2/d_k$ from the
independent parts of the keys; the shared part adds a little more. Repeating
the proof with this overlap, for values of mean zero, gives

$$
\mathbb{E}\,\big\|\mathbf{S}\mathbf{k}_j - \mathbf{v}_j\big\|^2 \approx
(n - 1)\Big(\rho^4 + \frac{(1 - \rho^2)^2}{d_k}\Big).
$$

The error still grows in proportion to $n - 1$, but the term $\rho^4$ does
not shrink as $d_k$ grows. At $\rho = 0.5$ and $d_k = 64$ the formula
predicts an error 4.6 times that of independent keys, and the measured
curve starts about that far above the one for independent keys of the
same width; only half of 64 pairs are recovered (49%), against all of them
for independent keys. (At large $n$ the measured curve bends upward: the
mean of the codebook is not exactly zero, and the common overlap $\rho^2$
of correlated keys turns it into a term that grows like $n^2$; Exercise
:exref:`ex_ms-correlated-keys` finds it.) What a matrix memory can hold
depends on how different its keys are, not only on how many there are.

## Decay

### Forgetting Old Pairs

Linear attention never forgets. After $n$ writes all $n$ pairs share the
memory, and by :eqref:`eq_ms-capacity-law` the read error grows with $n$,
however old most of the pairs are. A decay before each write shrinks the
old pairs,

$$
\mathbf{S}_t = a_t\, \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top,
\qquad a_t \in (0, 1),
$$
:eqlabel:`eq_ms-decay`

so that a pair written $\ell$ steps ago has weight $a^\ell$ if the decay is
constant. These weights sum to $1/(1 - a)$, and the memory holds about the
last $1/(1-a)$ pairs by weight: 10 for $a = 0.9$, 100 for $a = 0.99$. This
is the number to remember. Interference counts differently, because it adds
squared weights, which sum to $1/(1 - a^2)$. By the argument of the
proposition, the interference at any key of a long-running decayed memory
stays below about $1/((1-a^2)\,d_k)$, as if the memory held $1/(1-a^2)$
pairs: about 5 at $a = 0.9$, roughly half of $1/(1-a)$ whenever $a$ is
close to one. The signal of an old pair shrinks as well, by $a^\ell$, and
Exercise :exref:`ex_ms-sliding-window` weighs the two.

A decay makes room for new pairs by giving up old ones, whether or not they
are still needed. A decay computed from the input, like the step sizes of
:numref:`sec_mamba`, at least lets the model choose when: hold the memory
through a paragraph, clear it at a new document. Such a decay is one of the
two changes that :numref:`sec_windowed-linear-attention` named as bringing
linear attention close to softmax attention on language benchmarks.

The decay can be shaped in several ways, each a one-line change of
:eqref:`eq_ms-decay`. RetNet :cite:`Sun.Dong.Huang.ea.2023` fixes one
scalar per head, so that different heads remember for different lengths.
Mamba-2 :cite:`Dao.Gu.2024` computes one scalar per head from the input,
as in :eqref:`eq_ms-decay`. Gated linear attention
:cite:`Yang.Wang.Shen.ea.2024` decays every key coordinate at its own rate,
$\mathbf{S}_t = \mathbf{S}_{t-1} \textrm{diag}(\boldsymbol{\alpha}_t) +
\mathbf{v}_t \mathbf{k}_t^\top$, so that some directions of the key space
are kept longer than others. We use one scalar per head throughout the
chapter. It is the simplest decay that a token can control, and, as we will
see, the one that turns the memory into masked attention with a single
mask.

### Selective State Space Models as Matrix Memories

The previous section ended by stacking the states of a Mamba layer into an
$H \times N$ matrix. Written out, that matrix is a decayed memory of the
kind above. Channel $c$ keeps a state $\mathbf{h}_t^{(c)} \in \mathbb{R}^N$,
decays it by $e^{\Delta_{t,c}\boldsymbol{\lambda}^{(c)}}$, where
$\boldsymbol{\lambda}^{(c)}$ holds the $N$ rates of the channel, and adds
$\Delta_{t,c}\, x_{t,c}\, \mathbf{B}_t$
(:numref:`subsec_selective-dynamics`). Put the states of the $H$ channels
into the rows of a matrix $\mathbf{S}_t \in \mathbb{R}^{H \times N}$ and
their rates into the rows of $\boldsymbol{\Lambda} \in \mathbb{R}^{H \times
N}$, and drop the skip term $D_c x_{t,c}$. The update and the readout
become

$$
\mathbf{S}_t = \mathbf{E}_t \odot \mathbf{S}_{t-1} + (\boldsymbol{\Delta}_t
\odot \mathbf{x}_t)\, \mathbf{B}_t^\top, \qquad
\mathbf{y}_t = \mathbf{S}_t \mathbf{C}_t, \qquad
\mathbf{E}_t = \exp(\textrm{diag}(\boldsymbol{\Delta}_t)\,
\boldsymbol{\Lambda}),
$$

with the exponential taken entry by entry. Read against
:eqref:`eq_ms-decay`, the value is the input scaled by its step sizes,
$\boldsymbol{\Delta}_t \odot \mathbf{x}_t$, the key is $\mathbf{B}_t$, the
query is $\mathbf{C}_t$, $d_k = N$, and $d_v = H$, the number of channels.
Only the decay does not fit: where :eqref:`eq_ms-decay` has one number,
$\mathbf{E}_t$ has a different rate for every channel and every state.
Mamba-2 restricts it to one number per head :cite:`Dao.Gu.2024`, which
gives up the spread of rates across the $N$ states and gains the matrix
products of the dual form below (Exercise :exref:`ex_ms-mamba` asks what
the restriction requires).
The state of Mamba was a matrix memory all along; only the names were
different.

## The Dual Form

### Unrolling with Decay

Written as a recurrence, the decayed memory processes one token at a time.
Without decay, linear attention can also be computed as attention, over
all positions at once (:numref:`sec_windowed-linear-attention`). Does this
survive the decay? With a scalar decay, unrolling :eqref:`eq_ms-decay` from
$\mathbf{S}_0 = \mathbf{0}$ gives each output as a weighted sum over the
past, in which a pair written at step $s$ is scaled by all decays met
since:

$$\mathbf{y}_t = \sum_{s \le t} L_{ts}\, (\mathbf{k}_s^\top \mathbf{q}_t)\,
\mathbf{v}_s, \qquad L_{ts} = a_{s+1} a_{s+2} \cdots a_t.$$

Stack the queries, keys, values, and outputs of a sequence as the rows of
$\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$, and $\mathbf{Y}$, and set $L_{ts}
= 0$ for $s > t$. Then all outputs together are

$$\mathbf{Y} = (\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top)\, \mathbf{V}.$$
:eqlabel:`eq_ms-semiseparable`

This is masked attention. With every $a_t = 1$, $\mathbf{L}$ is the causal
mask of a transformer, ones on and below the diagonal, and
:eqref:`eq_ms-semiseparable` is causal linear attention. With decays,
$\mathbf{L}$ is a *learned, decaying* mask: a decayed linear recurrence is
attention in which older tokens count less, by an amount the inputs choose.
:citet:`Dao.Gu.2024` call this correspondence *state space duality*. It
holds for linear attention, not for softmax attention, whose exponential
does not factor this way.

The two sides of the duality are two orders of computing the same
products. Forming $\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top$ first is
attention: all positions in parallel, at a cost and memory quadratic in
$T$. Summing from the left, one step at a time, is the recurrence: the
partial sums are the states $\mathbf{S}_t$, the cost is linear in $T$,
and the steps are sequential. We write both, starting with the recurrence.

```{.python .input #matrix-state-unrolling-with-decay-1}
%%tab pytorch
def linear_attention_recurrent(Q, K, V, a):
    """y_t = S_t q_t with S_t = a_t S_{t-1} + v_t k_t^T, step by step."""
    S, outputs = Q.new_zeros(V.shape[-1], K.shape[-1]), []
    for q, k, v, a_t in zip(Q, K, V, a):
        S = a_t * S + torch.outer(v, k)
        outputs.append(S @ q)
    return torch.stack(outputs)
```

```{.python .input #matrix-state-unrolling-with-decay-1}
%%tab jax
def linear_attention_recurrent(Q, K, V, a):
    """y_t = S_t q_t with S_t = a_t S_{t-1} + v_t k_t^T, step by step."""
    def step(S, inputs):
        q, k, v, a_t = inputs
        S = a_t * S + jnp.outer(v, k)
        return S, S @ q
    S0 = jnp.zeros((V.shape[-1], K.shape[-1]))
    return jax.lax.scan(step, S0, (Q, K, V, a))[1]
```

The dual form needs the mask $\mathbf{L}$. Its entries are products of
decays, which we compute from the logarithms of the decays: $\log L_{ts}$ is
the sum of $\log a_r$ over $s < r \le t$, a difference of two cumulative
sums. Above the diagonal these differences are positive and their
exponentials can overflow, so we set them to $-\infty$ before
exponentiating, which turns them into exact zeros. We save the mask,
because the chunked algorithms below and in the next section use it too.

```{.python .input #matrix-state-unrolling-with-decay-2}
%%tab pytorch
def decay_mask(log_a):  #@save
    """L[t, s] = a_{s+1} * ... * a_t for s <= t, and 0 for s > t."""
    cum = log_a.cumsum(-1)
    log_L = cum[..., :, None] - cum[..., None, :]  # Sum of log a over (s, t]
    causal = torch.ones(log_L.shape[-2:], dtype=torch.bool,
                        device=log_a.device).tril()
    return torch.exp(log_L.masked_fill(~causal, -torch.inf))

def linear_attention_dual(Q, K, V, a):
    """The same outputs as Y = (L * Q K^T) V, all positions at once."""
    return (decay_mask(torch.log(a)) * (Q @ K.T)) @ V
```

```{.python .input #matrix-state-unrolling-with-decay-2}
%%tab jax
def decay_mask(log_a):  #@save
    """L[t, s] = a_{s+1} * ... * a_t for s <= t, and 0 for s > t."""
    cum = log_a.cumsum(-1)
    log_L = cum[..., :, None] - cum[..., None, :]  # Sum of log a over (s, t]
    causal = jnp.tril(jnp.ones(log_L.shape[-2:], dtype=bool))
    return jnp.exp(jnp.where(causal, log_L, -jnp.inf))

def linear_attention_dual(Q, K, V, a):
    """The same outputs as Y = (L * Q K^T) V, all positions at once."""
    return (decay_mask(jnp.log(a)) * (Q @ K.T)) @ V
```

Neither order suits training as it stands. The recurrence takes $T$
sequential steps, each too small to keep a GPU busy, and the dual form
stores a $T \times T$ matrix per head, which grows with the square of the
length.

### Chunked Computation

Chunking combines the two orders, as :numref:`sec_windowed-linear-attention`
did for linear attention without decay: the quadratic form within each
chunk, and a state carried from chunk to chunk. Cut the sequence into
chunks of $m$ steps and partition the matrix $\mathbf{L} \odot
\mathbf{Q}\mathbf{K}^\top$ into $m \times m$ blocks, as in
:numref:`fig_ms-semiseparable`. The blocks on the diagonal are small masked
attention products, computed in parallel. The blocks below the diagonal are
never formed, and the decays are the reason they need not be. A query at
step $t$ reaches a key at step $s$ of an earlier chunk through the score
$\mathbf{k}_s^\top \mathbf{q}_t$ times $L_{ts}$, and $L_{ts}$ splits into
three factors: the decay of the key to the end of its chunk, the decays of
the chunks in between, and the decay of the query from the start of its
own chunk. Each factor belongs to one side. Scale every key of the past by
its factors, sum the outer products $\mathbf{v}_s\mathbf{k}_s^\top$, and
what comes out is a $d_v \times d_k$ matrix: the state at the start of the
chunk. Everything a chunk needs from the entire past is therefore that
state, read with queries scaled by their own decays. As a consequence,
every block below the diagonal has rank at most $d_k$ (Exercise
:exref:`ex_ms-rank-one`).

![The mask $\mathbf{L}$ of :eqref:`eq_ms-semiseparable` for random decays, partitioned into chunks. Diagonal blocks (outlined in blue) are small masked-attention products computed in parallel. Every block below the diagonal (outlined in orange) has rank one in $\mathbf{L}$ and rank at most $d_k$ in $\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top$, and is accounted for by the state passed from chunk to chunk (arrows); the shading fades as the decays compound.](../img/mdl-modernrnn-semiseparable.svg)
:label:`fig_ms-semiseparable`

Number the positions of a chunk from 1 to $m$. The chunked computation has
three parts:

1. *Within chunks*: the dual form :eqref:`eq_ms-semiseparable` on each
   chunk, with the chunk's own $m \times m$ mask.
2. *Across chunks*: let $\gamma^{(c)} = a_1 a_2 \cdots a_m$ be the product
   of the decays of chunk $c$. The state at the end of the chunk is the
   decayed state at the end of the previous one plus the chunk's writes,
   each decayed to the chunk's end,

   $$\mathbf{S}^{(c)} = \gamma^{(c)}\, \mathbf{S}^{(c-1)} + \sum_{t=1}^{m}
   L_{mt}\, \mathbf{v}_t \mathbf{k}_t^\top.$$

   This is a linear recurrence over the $T/m$ chunks with matrix-valued
   states, and `linear_scan` of :numref:`sec_linear-recurrence` computes
   it.
3. *Reading the past*: each position $t$ adds the read of the state at its
   chunk's start, decayed to $t$, $(a_1 \cdots a_t)\,
   \mathbf{S}^{(c-1)}\mathbf{q}_t$.

Step 1 is the dual form on a small scale, and step 3 is one read per
position. Step 2 is the only new idea, and it is the step to read slowly.

Almost all of the work is in matrix products, which GPUs do well, and the
memory holds $m \times m$ blocks and one state per chunk instead of a $T
\times T$ matrix. At $T = 4{,}096$ in 32-bit floats, the dual form's matrix
takes 64 MB per head. Chunks of 64 need $T \times 64$ mask entries, one
$64 \times 64$ block per chunk, and one $64 \times 64$ state per chunk,
about 2 MB, and this figure grows only in proportion to the length.
Mamba-2 trains this way, and so do the layers of the next section. With a
decay per key coordinate the scheme still works, but every key coordinate
then needs a mask of its own, $d_k$ masks per chunk instead of one.

The function takes a batch of sequences and the logarithms of their
decays, one per step. Its three parts follow the list above. The array
`cum` holds the cumulative log decays from the start of each chunk, so that
every factor the function exponentiates is the decay over a stretch of
time and lies between 0 and 1, however strong the decays get.

```{.python .input #matrix-state-chunked-computation-1}
%%tab pytorch
def chunked_linear_attention(Q, K, V, log_a, chunk_size):  #@save
    """y_t = S_t q_t with S_t = a_t S_{t-1} + v_t k_t^T, computed in chunks.

    Q, K: (batch_size, T, d_k); V: (batch_size, T, d_v); log_a: (batch_size, T)
    """
    B, T, m = Q.shape[0], Q.shape[1], chunk_size
    Q, K, V = (X.reshape(B, T // m, m, -1) for X in (Q, K, V))
    log_a = log_a.reshape(B, T // m, m)
    # 1. Within chunks: the dual form, one m x m mask per chunk
    Y = (decay_mask(log_a) * (Q @ K.mT)) @ V
    # 2. Across chunks: S^(c) = gamma^(c) S^(c-1) + sum_t L_mt v_t k_t^T
    cum = log_a.cumsum(-1).unsqueeze(-1)  # Log decay from the chunk's start
    writes = V.mT @ (K * torch.exp(cum[:, :, -1:] - cum))
    S = d2l.linear_scan(torch.exp(cum[:, :, -1:]), writes)  # At chunk ends
    S = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], 1)  # At the starts
    # 3. Every step reads the state at its chunk's start, decayed to it
    return (Y + (Q * torch.exp(cum)) @ S.mT).reshape(B, T, -1)
```

```{.python .input #matrix-state-chunked-computation-1}
%%tab jax
def chunked_linear_attention(Q, K, V, log_a, chunk_size):  #@save
    """y_t = S_t q_t with S_t = a_t S_{t-1} + v_t k_t^T, computed in chunks.

    Q, K: (batch_size, T, d_k); V: (batch_size, T, d_v); log_a: (batch_size, T)
    """
    B, T, m = Q.shape[0], Q.shape[1], chunk_size
    Q, K, V = (X.reshape(B, T // m, m, -1) for X in (Q, K, V))
    log_a = log_a.reshape(B, T // m, m)
    # 1. Within chunks: the dual form, one m x m mask per chunk
    Y = (decay_mask(log_a) * (Q @ K.mT)) @ V
    # 2. Across chunks: S^(c) = gamma^(c) S^(c-1) + sum_t L_mt v_t k_t^T
    cum = log_a.cumsum(-1)[..., None]  # Log decay from the chunk's start
    writes = V.mT @ (K * jnp.exp(cum[:, :, -1:] - cum))
    S = d2l.linear_scan(jnp.exp(cum[:, :, -1:]), writes)  # At chunk ends
    S = jnp.concatenate([jnp.zeros_like(S[:, :1]), S[:, :-1]], 1)  # Starts
    # 3. Every step reads the state at its chunk's start, decayed to it
    return (Y + (Q * jnp.exp(cum)) @ S.mT).reshape(B, T, -1)
```

Three orders of computation should give one answer. We check them on one
random sequence of 512 steps, with $d_k = d_v = 64$ and decays between 0.8
and 0.99, and print how far the dual form and chunks of 16, 64, and 256
steps deviate from the recurrence, relative to its largest output.

```{.python .input #matrix-state-chunked-computation-2}
%%tab pytorch
gen = torch.Generator().manual_seed(0)  # Local: later training stays unseeded
T, d_k, d_v = 512, 64, 64
Q, K, V = (torch.randn(T, d, generator=gen) for d in (d_k, d_k, d_v))
a = 0.8 + 0.19 * torch.rand(T, generator=gen)  # Decays in (0.8, 0.99)
Y = linear_attention_recurrent(Q, K, V, a)
forms = {'dual form': linear_attention_dual(Q, K, V, a)}
for m in (16, 64, 256):
    forms[f'chunks of {m}'] = chunked_linear_attention(
        Q[None], K[None], V[None], torch.log(a)[None], m)[0]
for name, Y_form in forms.items():
    print(f'{name:13s} relative deviation from the recurrence: '
          f'{(Y_form - Y).abs().max() / Y.abs().max():.1e}')
```

```{.python .input #matrix-state-chunked-computation-2}
%%tab jax
T, d_k, d_v = 512, 64, 64
key_q, key_k, key_v, key_a = jax.random.split(jax.random.key(0), 4)
Q, K = jax.random.normal(key_q, (T, d_k)), jax.random.normal(key_k, (T, d_k))
V = jax.random.normal(key_v, (T, d_v))
a = 0.8 + 0.19 * jax.random.uniform(key_a, (T,))  # Decays in (0.8, 0.99)
with jax.default_matmul_precision('highest'):  # Full float32, not TF32
    Y = linear_attention_recurrent(Q, K, V, a)
    forms = {'dual form': linear_attention_dual(Q, K, V, a)}
    for m in (16, 64, 256):
        forms[f'chunks of {m}'] = chunked_linear_attention(
            Q[None], K[None], V[None], jnp.log(a)[None], m)[0]
for name, Y_form in forms.items():
    print(f'{name:13s} relative deviation from the recurrence: '
          f'{jnp.abs(Y_form - Y).max() / jnp.abs(Y).max():.1e}')
```

All three agree to a few millionths of the output or better, the rounding
error of 32-bit floats, for every chunk size. The chunk size divides the
work between matrix products and the scan and leaves the result alone, so
choosing it is a matter of hardware alone. Exercise
:exref:`ex_ms-operations` asks which size is fastest.

## A Gated Linear Attention Layer

### The Layer

A layer built on the decayed matrix memory computes, for every head, a
query, a key, a value, and a decay from its input, runs the chunked
recurrence, and projects the outputs of all heads back to the model width.
We call it *gated linear attention*, the name that
:citet:`Yang.Wang.Shen.ea.2024` gave to linear attention with a learned
decay, although their layer decays every key coordinate and ours, as
explained above, one number per head. The decay of head $h$,
$a_{t,h} = \sigma(\mathbf{w}_h^\top \mathbf{x}_t + b_h)$, is computed from
the input by one linear map for all heads. One number per head from the
input is the choice of Mamba-2, and the sigmoid is its decay in another
form: since $\sigma(z) = \exp(-\textrm{softplus}(-z))$, it is $e^{\Delta
A}$ with the step $\Delta = \textrm{softplus}(-z)$ and the rate $A$ fixed
at $-1$. The biases start the heads with memories spread log-uniformly
between 4 and 256 steps, $a_{t,h} \approx e^{-1/\tau_h}$, just as the
timescales of S4D were spread in :numref:`sec_ssm`.

```{.python .input #matrix-state-the-layer-1}
%%tab pytorch
class GatedLinearAttention(nn.Module):  #@save
    """Multi-head linear attention with one input-dependent decay per head."""
    def __init__(self, num_hiddens, num_heads, chunk_size=16, conv_width=4,
                 timescales=(4, 256)):
        super().__init__()
        d = num_hiddens
        self.num_heads, self.chunk_size = num_heads, chunk_size
        self.W_qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(3 * d, 3 * d, conv_width, groups=3 * d,
                              padding=conv_width - 1)  # One filter per channel
        self.W_a = nn.Linear(d, num_heads)  # Decay gate, one per head
        self.W_g = nn.Linear(d, d)  # Output gate
        self.W_o = nn.Linear(d, d, bias=False)
        d2l.init_lecun(self)
        self.init_decays(*timescales)  # Memories of 4 to 256 steps by default

    def init_decays(self, tau_min, tau_max):
        """Start the decays at exp(-1/tau), tau log-uniform in range."""
        b = self.W_a.bias
        tau = torch.logspace(math.log10(tau_min), math.log10(tau_max), len(b))
        with torch.no_grad():  # log sigmoid(b) = -1 / tau
            b.copy_(-torch.log(torch.expm1(1 / tau)))
```

```{.python .input #matrix-state-the-layer-1}
%%tab jax
class GatedLinearAttention(nnx.Module):  #@save
    """Multi-head linear attention with one input-dependent decay per head."""
    def __init__(self, num_hiddens, num_heads, chunk_size=16, conv_width=4,
                 timescales=(4, 256), *, rngs):
        d = num_hiddens
        self.num_heads, self.chunk_size = num_heads, chunk_size
        self.W_qkv = nnx.Linear(d, 3 * d, use_bias=False, rngs=rngs)
        self.conv = nnx.Conv(3 * d, 3 * d, (conv_width,),  # Causal, one
                             feature_group_count=3 * d,  # filter per channel
                             padding=((conv_width - 1, 0),), rngs=rngs)
        self.W_a = nnx.Linear(d, num_heads, rngs=rngs)  # Decay, per head
        self.W_g = nnx.Linear(d, d, rngs=rngs)  # Output gate
        self.W_o = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.init_decays(*timescales)  # Memories of 4 to 256 steps by default

    def init_decays(self, tau_min, tau_max):
        """Start the decays at exp(-1/tau), tau log-uniform in range."""
        b = self.W_a.bias
        tau = jnp.logspace(math.log10(tau_min), math.log10(tau_max), len(b))
        b[...] = -jnp.log(jnp.expm1(1 / tau))  # log sigmoid(b) = -1 / tau
```

The memory sits in a method of its own, `memory`, so that the next section
can replace it, and `decays` returns the log decays of every head. Both see
the heads side by side: `heads_to_batch` moves the heads of a minibatch into
its batch axis, so that the chunked computation treats every head as a
sequence of its own, and `batch_to_heads` moves them back.

```{.python .input #matrix-state-the-layer-3}
%%tab pytorch
def heads_to_batch(Z, num_heads):  #@save
    """(batch_size, T, num_heads * e) -> (batch_size * num_heads, T, e)."""
    B, T, _ = Z.shape
    return Z.reshape(B, T, num_heads, -1).transpose(1, 2).flatten(0, 1)

def batch_to_heads(Y, batch_size):  #@save
    """(batch_size * num_heads, T, e) -> (batch_size, T, num_heads * e)."""
    _, T, e = Y.shape
    return Y.reshape(batch_size, -1, T, e).transpose(1, 2).flatten(2)

@d2l.add_to_class(GatedLinearAttention)  #@save
def decays(self, X):
    """Log decays, (batch_size * num_heads, T)."""
    return heads_to_batch(F.logsigmoid(self.W_a(X)), self.num_heads)[..., 0]

@d2l.add_to_class(GatedLinearAttention)  #@save
def memory(self, X, Q, K, V, log_a):
    """Outputs of the decayed matrix memory of every head. X is unused here;
    the delta rule of the next section reads its write strengths from it."""
    return chunked_linear_attention(Q, K, V, log_a, self.chunk_size)
```

```{.python .input #matrix-state-the-layer-3}
%%tab jax
def heads_to_batch(Z, num_heads):  #@save
    """(batch_size, T, num_heads * e) -> (batch_size * num_heads, T, e)."""
    B, T, _ = Z.shape
    return Z.reshape(B, T, num_heads, -1).swapaxes(1, 2).reshape(
        B * num_heads, T, -1)

def batch_to_heads(Y, batch_size):  #@save
    """(batch_size * num_heads, T, e) -> (batch_size, T, num_heads * e)."""
    _, T, e = Y.shape
    return Y.reshape(batch_size, -1, T, e).swapaxes(1, 2).reshape(
        batch_size, T, -1)

@d2l.add_to_class(GatedLinearAttention)  #@save
def decays(self, X):
    """Log decays, (batch_size * num_heads, T)."""
    return heads_to_batch(jax.nn.log_sigmoid(self.W_a(X)),
                          self.num_heads)[..., 0]

@d2l.add_to_class(GatedLinearAttention)  #@save
def memory(self, X, Q, K, V, log_a):
    """Outputs of the decayed matrix memory of every head. X is unused here;
    the delta rule of the next section reads its write strengths from it."""
    return chunked_linear_attention(Q, K, V, log_a, self.chunk_size)
```

The forward pass adds three parts around the memory, and the first matters
most. Before the queries, keys, and values reach the memory, they pass
through a causal convolution of width 4, one filter per channel, and a
SiLU, so that each query, key, and value also depends on the three tokens
before it. Without the convolution, the write at a token pairs that
token's key with that token's value. A name and its value sit on
neighboring tokens, so one layer alone could not store them as one pair. A
second layer could, but only after the first has learned to carry the name
forward; the convolution lets the token that carries the value carry the
key of the name before it from the first update on. In
:numref:`sec_hybrids` a related layer loses its convolution, and every run
without it recalls fewer pairs. H3 :cite:`Fu.Dao.Saab.ea.2023` used a
short filter of this kind to solve associative recall, and Mamba has one
too.

The other two parts are simpler. Queries and keys are scaled to unit
length, the setting of :eqref:`eq_ms-capacity-law`. The output of each head
is normalized to unit root mean square, which keeps its scale independent
of how many pairs the memory holds. A gate $\textrm{SiLU}(\mathbf{W}_g
\mathbf{x}_t)$ then lets each token scale, channel by channel, what it
takes from the memory before the output projection. The layer of
:citet:`Yang.Wang.Shen.ea.2024` normalizes and gates each head in the same
way, with a LayerNorm in place of the RMS normalization.

```{.python .input #matrix-state-the-layer-2}
%%tab pytorch
@d2l.add_to_class(GatedLinearAttention)  #@save
def forward(self, X):  # X: (batch_size, num_steps, num_hiddens)
    B, T = X.shape[:2]
    QKV = self.conv(self.W_qkv(X).mT)[..., :T].mT  # Causal: drop the tail
    Q, K, V = (heads_to_batch(Z, self.num_heads)
               for Z in F.silu(QKV).chunk(3, -1))
    Q, K = F.normalize(Q, dim=-1), F.normalize(K, dim=-1)  # Unit length
    Y = self.memory(X, Q, K, V, self.decays(X))  # (B * num_heads, T, d / h)
    Y = F.rms_norm(Y, Y.shape[-1:], eps=1e-6)  # Unit RMS per head
    return self.W_o(batch_to_heads(Y, B) * F.silu(self.W_g(X)))
```

```{.python .input #matrix-state-the-layer-2}
%%tab jax
@d2l.add_to_class(GatedLinearAttention)  #@save
def __call__(self, X):  # X: (batch_size, num_steps, num_hiddens)
    QKV = jax.nn.silu(self.conv(self.W_qkv(X)))
    Q, K, V = (heads_to_batch(Z, self.num_heads)
               for Z in jnp.split(QKV, 3, -1))
    Q, K = (Z / jnp.maximum(jnp.linalg.norm(Z, axis=-1, keepdims=True), 1e-12)
            for Z in (Q, K))  # Unit length
    Y = self.memory(X, Q, K, V, self.decays(X))  # (B * num_heads, T, d / h)
    Y = Y / jnp.sqrt((Y ** 2).mean(-1, keepdims=True) + 1e-6)  # Unit RMS
    return self.W_o(batch_to_heads(Y, X.shape[0]) * jax.nn.silu(self.W_g(X)))
```

### Associative Recall

The interference law was measured with random keys, which no network
uses. Does a trained layer store pairs in its matrix, and does what it
holds grow with the size of the state, as the law predicts? To find out we
use *multi-query associative recall*
:cite:`Arora.Eyuboglu.Timalsina.ea.2024`, the task of the opening reduced
to tokens. A sequence begins with $n$ pairs, each a key token followed by a
value token, with distinct keys and with values drawn from 4 possible
ones. Filler tokens, if the sequence has room for any, separate the pairs
from the end, where every key appears once more, in random order, as a
query. At each query the model must output the value that followed that
key. The options `num_rebound` and `gap` write some keys a second time with
a new value; the experiment of this section leaves them off, and the next
section needs them. We save the function, because the rest of the chapter
uses the task too.

```{.python .input #hybrids-associative-recall-1}
%%tab pytorch, jax
def associative_recall(rng, batch_size, num_pairs, num_keys=512,  #@save
                       num_values=4, num_steps=256, num_rebound=0, gap=None):
    """Key-value pairs, filler, then every key again as a query. The first
    num_rebound keys are written twice, the second time with a new value,
    at any later write or, given gap, gap writes after the first one.
    Returns the tokens (values, then keys, then the filler), the latest
    value of each query's key, and a mask of the queries for rebound keys."""
    n, r = num_pairs, num_rebound
    keys = np.argsort(rng.random((batch_size, num_keys)), 1)[:, :n]
    values = rng.integers(0, num_values, (batch_size, n + r))
    values[:, n:] = (values[:, :r] + rng.integers(  # A new value
        1, num_values, (batch_size, r))) % num_values
    if gap is None:  # Random write order, each second write after the first
        slots = np.argsort(rng.random((batch_size, n + r)), 1)
        slots[:, :r], slots[:, n:] = (np.minimum(slots[:, :r], slots[:, n:]),
                                      np.maximum(slots[:, :r], slots[:, n:]))
    else:  # First writes in random order, second ones gap first writes later
        first = np.argsort(rng.random((batch_size, n)), 1)
        times = np.concatenate([first, first[:, :r] + gap - 0.5], 1)
        slots = np.argsort(np.argsort(times, 1), 1)
    X = np.full((batch_size, num_steps), num_values + num_keys)  # Filler
    rows = np.arange(batch_size)[:, None]
    X[rows, 2 * slots] = num_values + np.concatenate([keys, keys[:, :r]], 1)
    X[rows, 2 * slots + 1] = values
    order = np.argsort(rng.random((batch_size, n)), 1)  # Queries
    X[:, -n:] = num_values + np.take_along_axis(keys, order, 1)
    latest = np.concatenate([values[:, n:], values[:, r:n]], 1)
    return X, np.take_along_axis(latest, order, 1), order < r
```

One short sequence shows the layout. It has four pairs, and to show the
option once, its first key is written twice. Token ids 0 to 3 are the
values, 4 to 515 the keys (512 possible keys by default), and 516 is the
filler.

```{.python .input #matrix-state-associative-recall-1}
%%tab pytorch, jax
X, Y, rebound = associative_recall(np.random.default_rng(0), 1, 4,
                                   num_steps=16, num_rebound=1)
print('tokens: ', X[0])
print('answers:', Y[0], ' written twice:', rebound[0])
```

The key 273 is written first with the value 3 and later with 0, and its
query, the third, expects the newer value, 0. The other three keys are
written once, and the four queries come last.

`train_recall` trains a model with the loss at the query positions only
(`d2l.train_queries` of :numref:`subsec_delayed-recall`), on fresh batches
of 256 sequences whose number of pairs cycles through `pairs`, for 2,000
updates. It uses AdamW with gradients clipped to norm 1 and the schedule
`d2l.cosine_warmup` of :numref:`sec_alexnet`, which warms up over a tenth
of the updates to a peak rate of 0.01 and then decays. It then measures the
accuracy at each number of pairs on 1,024 new sequences. Each sequence is
as short as its pairs and queries allow, padded with filler to a multiple
of 32 tokens.

```{.python .input #hybrids-the-recall-sweep-1}
%%tab pytorch
def train_recall(model, pairs=(4, 8, 16, 32, 64), num_updates=2000,  #@save
                 **task):
    """Train on associative recall (task: options of associative_recall)
    with a cycling number of pairs, 256 sequences per update, AdamW at peak
    rate 0.01; return the accuracy per number of pairs."""
    rng = np.random.default_rng()
    length = lambda n: 32 * math.ceil(
        (3 * n + 2 * task.get('num_rebound', 0)) / 32)
    data = lambda n, size: associative_recall(
        rng, size, n, num_steps=length(n), **task)[:2]
    model.to(d2l.try_gpu())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: (
        d2l.cosine_warmup(t, num_updates, num_updates // 10)))
    d2l.train_queries(model, lambda step: data(pairs[step % len(pairs)],
                      256), num_updates, optimizer, scheduler)
    return {n: float(d2l.query_accuracy(model, *data(n, 1024)))
            for n in pairs}
```

```{.python .input #hybrids-the-recall-sweep-1}
%%tab jax
def train_recall(model, pairs=(4, 8, 16, 32, 64), num_updates=2000,  #@save
                 **task):
    """Train on associative recall (task: options of associative_recall)
    with a cycling number of pairs, 256 sequences per update, AdamW at peak
    rate 0.01; return the accuracy per number of pairs."""
    rng = np.random.default_rng()
    length = lambda n: 32 * math.ceil(
        (3 * n + 2 * task.get('num_rebound', 0)) / 32)
    data = lambda n, size: associative_recall(
        rng, size, n, num_steps=length(n), **task)[:2]
    schedule = lambda t: 1e-2 * d2l.cosine_warmup(t, num_updates,
                                                  num_updates // 10)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.01)), wrt=nnx.Param)
    d2l.train_queries(model, lambda step: data(pairs[step % len(pairs)],
                      256), num_updates, optimizer)
    return {n: float(d2l.query_accuracy(model, *data(n, 1024)))
            for n in pairs}
```

The models are the stack `d2l.SeqLM` of :numref:`subsec_deep-rnn`, which
embeds the tokens, applies residual blocks around its sequence mixers, and
predicts a token at every position. Here it has two blocks of our layer at
width 32, and the keys come from a vocabulary of 128. We vary only how the
width is cut into heads: 8 heads of size 4, 4 of size 8, or 2 of size 16.
A head of size $e$ holds an $e \times e$ matrix, so a layer's state grows
from $8 \cdot 4^2 = 128$ entries to 256 and 512, while the number of
parameters barely changes. The decays start with memories of 256 to 2,048
steps, longer than any sequence (at most 192 tokens), because every pair
must survive until its query, and, as the recall experiments of
:numref:`sec_ssm` showed, a long memory that a model does not start with
is hard to learn.

```{.python .input #matrix-state-associative-recall-2}
%%tab pytorch
def memory_model(num_heads, num_hiddens=32, vocab_size=4 + 128 + 1):
    """Two blocks of gated linear attention; tokens: values, keys, filler."""
    return d2l.SeqLM(vocab_size, num_hiddens, [GatedLinearAttention(
        num_hiddens, num_heads, timescales=(256, 2048)) for _ in range(2)])
```

```{.python .input #matrix-state-associative-recall-2}
%%tab jax
def memory_model(num_heads, num_hiddens=32, vocab_size=4 + 128 + 1):
    """Two blocks of gated linear attention; tokens: values, keys, filler."""
    rngs = nnx.Rngs(d2l.get_key())  # A new initialization each call
    return d2l.SeqLM(vocab_size, num_hiddens, [GatedLinearAttention(
        num_hiddens, num_heads, timescales=(256, 2048), rngs=rngs)
        for _ in range(2)], rngs=rngs)
```

What should we expect? The law of the random keys was derived for one
memory, and here the state is cut into heads that all see the same pairs.
Each head has keys of its own, of width $e$, and holds an $e$-wide
slice of every value. If these keys behave like random ones, the variance
argument applies head by head, and the error along a candidate value
collects about $n/(32e)$ from all heads together. Now $32e = (32/e)\cdot
e^2$ is the number of entries of the state. So if a trained network fills
its memory the way random keys do, the number of pairs it holds at a
given recall grows in proportion to the entries: twice as many with 256
entries as with 128, four times as many with 512. Only this scaling is
predicted. The absolute numbers cannot be compared with those of the
random keys, which nobody had to learn and which were decoded among 256
values instead of 4. Each model is trained twice, from fresh weights and on
fresh data, so that a difference between the head sizes can be told apart
from the difference between two runs.

```{.python .input #matrix-state-associative-recall-3}
%%tab pytorch, jax
recall = {}
for num_heads in (8, 4, 2):
    d_h = 32 // num_heads
    name = f'{num_heads} heads of {d_h}, {num_heads * d_h ** 2} entries'
    models = [memory_model(num_heads) for _ in range(2)]
    recall[name] = [train_recall(model, num_keys=128) for model in models]
    print(f'{name:26s} {d2l.num_params(models[0]):,} parameters')
```

```{.python .input #matrix-state-associative-recall-4}
%%tab pytorch, jax
pairs = list(recall[name][0])
d2l.plot(pairs, [[np.mean([run[n] for run in runs]) for n in pairs]
                 for runs in recall.values()], 'pairs in the sequence',
         'recall, mean of two runs', legend=list(recall), xscale='log',
         ylim=(0.2, 1.02), fmts=('-o', '-s', '-^'), figsize=(4.5, 3))
d2l.plt.xticks(pairs, pairs)  # Ticks at the numbers of pairs
print(f'{"pairs":26s}', ''.join(f'{n:>11d}' for n in pairs))
for name, runs in recall.items():
    print(f'{name:26s}', ''.join(f'{runs[0][n]:6.2f}/{runs[1][n]:.2f}'
                                 for n in pairs))
```

The table prints both runs of each model as two numbers, and chance is
0.25. Every model learns the task: with four pairs, each recalls 98% or
more. As the pairs grow, recall falls, and the larger the state, the later
it falls. With 128 entries the memory recalls about half of 64 pairs; with
256 entries, about three quarters, a gap larger than the difference between
the two runs of either model. Doubling the state from 128 to 256 entries
roughly doubles what it holds: the larger state with $2n$ pairs recalls
about as well as the smaller with $n$. The parameter counts differ by about
1%; what changes is how the width is cut into heads, and with it the size
of the state. From 128 to 256 entries, a trained network follows the rule
that random keys obeyed.

The state of 512 entries is harder to read. It recalls the most on average
at 32 and 64 pairs, and from 16 pairs on every one of its runs is above
every run of the smallest state. At 32 and 64 pairs, however, its two runs
differ more than those of any other model, so this experiment does not say
how much more it holds than 256 entries. Whether 2,000 updates are too few
for heads this large is the question of Exercise
:exref:`ex_ms-training-length`.

One limitation is built into every model in this section. A write adds
$\mathbf{v}_t \mathbf{k}_t^\top$ to the memory, whatever the memory already
holds under $\mathbf{k}_t$. If a program sets `rate = 0.05` and later
`rate = 0.07`, the memory returns the sum of both values, scaled by the
decays, when queried with the key of `rate`. Key 273 of the example
sequence, written with 3 and then with 0, would be read back as a mixture
of both. A decay can make the old value fade, but it fades every other
pair as well. The next section changes the write so that it replaces the
value stored under a key.

## Summary

A matrix-valued state stores key–value pairs: a write adds the outer
product $\mathbf{v}_t\mathbf{k}_t^\top$, and a read multiplies the state by
a query. Unrolled, this recurrence is causal linear attention, and the
selective state space model of :numref:`sec_mamba` is a memory of the same
kind, with $\mathbf{B}_t$ as key and $\mathbf{C}_t$ as query. Stored pairs
interfere. For random unit keys the expected squared read error after $n$
writes is $(n-1)/d_k$, whatever the age of the pairs. The number of pairs
that can be recalled grows with $d_k d_v$, the number of entries of the
state, as long as the keys are not much narrower than the values, and
correlated keys add an error that no width removes. A decay makes room for
new pairs by fading old ones. With one decay per head, the decayed memory
equals masked attention with a learned mask, $(\mathbf{L} \odot
\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$. Computing it in chunks, attention
within a chunk and a recurrence across chunks, trains it with matrix
products in memory linear in the length. Trained on associative recall at
an equal parameter count, a layer built on this memory holds about twice
the pairs when its state doubles from 128 to 256 entries. Every write adds,
however, so a key written twice returns the sum of its values.

## Exercises

1. **Mamba as a matrix memory.** In the form
   $\mathbf{S}_t = \mathbf{E}_t \odot \mathbf{S}_{t-1} +
   (\boldsymbol{\Delta}_t \odot \mathbf{x}_t)\, \mathbf{B}_t^\top$ with
   $\mathbf{E}_t = \exp(\textrm{diag}(\boldsymbol{\Delta}_t)\,
   \boldsymbol{\Lambda})$, the logarithm of the decay is the matrix of
   rates with row $c$ scaled by the step size $\Delta_{t,c}$.
    1. Under which conditions on $\boldsymbol{\Delta}_t$ and on the rates
       $\boldsymbol{\Lambda}$ is $\mathbf{E}_t$ a scalar multiple of the
       matrix of ones, so that the update has the form
       :eqref:`eq_ms-decay`? Which of these conditions does Mamba-2
       impose, and what does it keep input-dependent?
    1. For each form of decay in this section (RetNet, Mamba-2, gated
       linear attention, Mamba), count the input-dependent numbers per
       token that determine the decay of one head of $d_v \times d_k$
       entries.
   :exlabel:`ex_ms-mamba`

1. **Rank-one blocks.** Let $p_t = a_1 a_2 \cdots a_t$ be the products of
   the decays in :eqref:`eq_ms-semiseparable`.
    1. Express $L_{ts}$ through the $p_t$, and show that every submatrix of
       $\mathbf{L}$ whose row indices are all at least as large as its
       column indices has rank one.
    1. Deduce that every block of $\mathbf{L} \odot
       \mathbf{Q}\mathbf{K}^\top$ that couples a chunk to an earlier chunk
       has rank at most $d_k$. Identify where `chunked_linear_attention`
       applies the query-side and the key-side decays of such a block.
    1. With a decay per key coordinate, $\mathbf{S}_t = \mathbf{S}_{t-1}
       \textrm{diag}(\boldsymbol{\alpha}_t) + \mathbf{v}_t
       \mathbf{k}_t^\top$, no single mask $\mathbf{L}$ multiplies
       $\mathbf{Q}\mathbf{K}^\top$. Write the decayed score of query $t$
       and key $s$, and show that a block coupling two chunks still has
       rank at most $d_k$.
   :exlabel:`ex_ms-rank-one`

1. **Counting operations.** Count one multiply-add as two floating-point
   operations, set $d_k = d_v = d$, and keep the leading terms.
    1. Count the operations of the recurrence, the dual form, and the
       chunked form at length $T$ and chunk size $m$. Split the chunked
       count into the three parts of the algorithm.
    1. Show that the chunked form costs at most a small constant times the
       recurrence while $m \le d$, and compute the ratio of chunked to dual
       operations at $T = 4096$, $d = 64$, and $m = 256$.
    1. Model the run time of a schedule as its operations divided by a
       throughput $P$, plus a fixed time $\tau$ per sequential step: $T$
       steps for the recurrence and $T/m$ for the chunked form. Find the
       chunk size that minimizes the chunked time, and evaluate it for
       $T = 4096$, $d = 64$, $P = 10^{13}$ operations per second, and
       $\tau = 5\,\mu\textrm{s}$. How much faster than the recurrence does
       the model predict the chunked form to be?
   :exlabel:`ex_ms-operations`

1. [code] **The normalizer as a value column.** The linear attention of
   :numref:`sec_windowed-linear-attention` applies a positive feature map
   $\phi$ to queries and keys and divides each output by
   $\phi(\mathbf{q}_t)^\top \mathbf{z}_t$, where $\mathbf{z}_t =
   \sum_{s \le t} \phi(\mathbf{k}_s)$. Give $\mathbf{z}_t$ the same decay
   as the state.
    1. Show that appending a constant 1 to every value vector makes the
       last coordinate of the output equal to the denominator.
    1. Compute the normalized outputs with one call of
       `chunked_linear_attention`, with $\phi = \textrm{elu} + 1$, and check
       them against a loop that carries $\mathbf{S}_t$ and $\mathbf{z}_t$.
    1. Which of the two changes, the feature map or the normalizer, alters
       the relative weights that a query gives to past positions, and which
       only rescales each output?

1. **Targeted forgetting.** Let a memory hold $\mathbf{S} =
   \mathbf{v}\mathbf{k}^\top + \sum_i \mathbf{v}_i \mathbf{k}_i^\top$,
   where $\mathbf{k}$ has unit length and every $\mathbf{k}_i$ is
   orthogonal to $\mathbf{k}$.
    1. Find a matrix $\mathbf{D}$ such that $\mathbf{S}\mathbf{D} =
       \sum_i \mathbf{v}_i \mathbf{k}_i^\top$ for all such memories. Is it
       unique?
    1. For which keys $\mathbf{k}$ is this transition one of the decays of
       this section?
    1. Apply the transition and then write a new value,
       $\mathbf{S}\mathbf{D} + \mathbf{v}'\mathbf{k}^\top$. Show that the
       memory now returns $\mathbf{v}'$ for $\mathbf{k}$ and is otherwise
       unchanged. The next section builds a layer on this update.

1. **Correlated keys.** ● The keys with $\rho > 0$ in `matrix_memory` are
   normalized versions of $\sqrt{1-\rho^2}\, \mathbf{g}_i + \rho\,
   \mathbf{c}$, with independent random unit vectors $\mathbf{g}_i$ and
   one shared unit vector $\mathbf{c}$. The formula in the text models
   the overlap of two distinct keys as $\rho^2 + \epsilon_{ij}$, with
   independent $\epsilon_{ij}$ of mean zero and variance $(1-\rho^2)^2/d_k$,
   and assumes values of mean zero.
    1. The values of `matrix_memory` come from a codebook of 256 random
       vectors, whose mean $\bar{\mathbf{v}}$ is not exactly zero. Show
       that this adds a term proportional to $(n-1)(n-2)\,
       \|\bar{\mathbf{v}}\|^2$ to the expected squared error, and compare
       your complete prediction with the measured errors.
    1. Center the codebook (subtract its mean, then normalize each entry
       again) and check that the upward bend of the curve disappears.
    1. Measure the error and the recall for $\rho \in \{0, 0.25, 0.5,
       0.75\}$ at $d_k = 64$ and $d_k = 256$. For which $\rho$ does the
       wider key still help, and where does the error approach $(n-1)
       \rho^4$?
   :exlabel:`ex_ms-correlated-keys`

1. [code] **Keys and values of different widths.** ● Give `matrix_memory`
   a value width `d_v` of its own, so that the codebook holds 256 random
   unit vectors in $\mathbb{R}^{d_v}$.
    1. Keep $n/(d_k d_v) = 1/32$ and measure recall for $(d_k, d_v) = (64,
       64)$, $(256, 16)$, and $(16, 256)$. Does recall depend on the product
       $d_k d_v$ alone?
    1. Show that the expected squared read error does not depend on $d_v$,
       so that it cannot explain what you measured.
    1. Every codebook entry is stored under about $n/256$ of the keys. Show
       that their overlaps with the read key add a component of variance
       about $n/(256\, d_k)$ along that entry, and use it to explain for
       which shapes the product law fails.
   :exlabel:`ex_ms-value-width`

1. [code] **Decay as a sliding window.** ● Add a constant decay $a$ to
   `matrix_memory`, so that the pair written $\ell$ steps before the end
   enters the final memory with weight $a^\ell$.
    1. At $d_k = 64$ and $n = 256$, measure recall separately for the
       oldest and the newest quarter of the pairs, for $a \in \{1, 0.99,
       0.95, 0.9\}$. What does the decay give the newest pairs, and what
       does it cost the oldest?
    1. Following the proof of the proposition, show that the expected
       squared interference at any key is at most $1 / ((1-a^2)\, d_k)$.
       Show that the number of pairs whose squared signal $a^{2\ell}$
       exceeds this bound is at most about $d_k / e$, whatever $a < 1$ is.
   :exlabel:`ex_ms-sliding-window`

1. [code] **One part at a time.** Train the model with 4 heads,
   `memory_model(4)`, on associative recall, with `train_recall(model,
   num_keys=128)`, once as it is and once each without the unit-length
   scaling of queries and keys, without the SiLU after the convolution,
   and without the output gate (subclass `GatedLinearAttention` and
   override `forward`, or `__call__` in JAX). The convolution is left out
   because :numref:`sec_hybrids` removes it. Which of the three parts
   matters most for recall at 32 and 64 pairs? Train each variant twice
   before you answer.

1. [code] **Training length and filler.** ●
    1. Train the model with 512 entries, `memory_model(2)`, five times
       with the default 2,000 updates and five times with
       `num_updates=4000`. Does the spread of recall at 64 pairs shrink,
       and does the model then hold clearly more than the one with 256
       entries?
    1. `train_recall` makes every sequence as short as its pairs and
       queries allow. Change it so that every sequence has 256 tokens, with
       a block of filler between the writes and the queries, and train the
       model with 128 entries. Compare its recall at 4 pairs with that of
       the experiment. A run of identical filler tokens writes the same
       outer product into the memory again and again; use this to explain
       what you see.
   :exlabel:`ex_ms-training-length`

[Discussions](https://d2l.discourse.group/t/36783)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.6]{.kicker}

Matrix-state recurrences<br>
**outer-product memory · interference · decay · attention with a decaying mask · chunks · a trained memory**
:::
:::

::: {.slide title="To Recall a Value, a Fixed-Size State Must Store Pairs"}
[The problem]{.kicker}

`rate = 0.05` near the top of a program; `rate` 200 lines later. A state
of $64 \times 64$ numbers: how many such pairs can it hold?

- Attention keeps every key and value: compare the query with each key,
  return the matching value.
- A recurrent state has a fixed size. To recall, it must store **pairs**.

::: {.d2l-note}
Make the state a **matrix** $\mathbf{S}$: add $\mathbf{v}\mathbf{k}^\top$
to write, multiply by a query $\mathbf{q}$ to read.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write and read key–value pairs with outer products;
- predict the read error $(n-1)/d_k$ of $n$ random pairs, and why recall
  grows with the number of entries of the state;
- show that a decayed matrix recurrence is masked attention;
- compute it in chunks, with matrix products and a short scan.

**Running example:** random unit keys of width 64 with a codebook of 256
values; a two-block model of width 32 trained on associative recall with
heads of size 4, 8, and 16.
:::

::: {.slide title="This Deck Turns the State Into a Table of Pairs"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | the state |
|---|---|
| Gated Recurrent Networks | a vector, gated |
| Sequence-to-Sequence Learning | one vector between two networks |
| Linear Recurrences and Parallel Scans | linear in the state, scanned |
| State Space Models | decays set by timescales |
| Selective State Space Models | decays chosen by the token |
| **Matrix-State Recurrences** | **a matrix of key–value pairs** |
| The Delta Rule | pairs corrected, not added |
| Hybrid Architectures | plus a few attention layers |
:::

::: {.col .narrow}
- **Uses:** linear attention as a recurrence; Mamba's state
  ($H$ channels × $N$ states); the linear scan.
- **Skips:** GPU kernels; a decay per key coordinate.
:::
:::
:::

::: {.slide title="Two Orthogonal Keys Store Two Values Exactly"}
$\mathbf{k}_1 = (1, 0)$, $\mathbf{k}_2 = (0, 1)$, values 3 and 5:

$$\mathbf{S} = 3\,\mathbf{k}_1^\top + 5\,\mathbf{k}_2^\top = (3, 5), \qquad \mathbf{S}\mathbf{k}_1 = 3,\ \mathbf{S}\mathbf{k}_2 = 5$$

Add $\mathbf{k}_3 = (1, 1)/\sqrt 2$ with value 7:
$\mathbf{S} \approx (7.95, 9.95)$, and $\mathbf{S}\mathbf{k}_1 = 7.95$.

::: {.d2l-note .warn}
Two dimensions hold two pairs exactly; the third leaks into both.
:::
:::

::: {.slide title="A Matrix Memory Is Linear Attention Read From the Left"}
Keys $\mathbf{k}_t \in \mathbb{R}^{d_k}$, values $\mathbf{v}_t \in
\mathbb{R}^{d_v}$, state $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$:

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t\mathbf{k}_t^\top,\qquad
\mathbf{y}_t = \mathbf{S}_t\mathbf{q}_t = \sum_{s \le t} (\mathbf{k}_s^\top\mathbf{q}_t)\,\mathbf{v}_s$$

- Right: causal attention without the softmax.
- Left: a recurrence with constant cost per token.
- $\exp(\mathbf{k}^\top\mathbf{q})$ does not fold into a matrix; a
  product of features does.
:::

::: {.slide title="Every Other Pair Leaks Into a Read"}
$n$ pairs, unit keys, read key $j$:

$$\mathbf{S}\mathbf{k}_j = \mathbf{v}_j + \sum_{i \neq j} (\mathbf{k}_i^\top\mathbf{k}_j)\,\mathbf{v}_i$$

Random unit keys, unit values:

$$\mathbb{E}\,\|\mathbf{S}\mathbf{k}_j - \mathbf{v}_j\|^2 = \frac{n - 1}{d_k}$$

Each overlap $\mathbf{k}_i^\top\mathbf{k}_j$ has mean 0 and variance
$1/d_k$; $n - 1$ of them add up.

::: {.d2l-note}
No age in the formula: pairs are lost to **interference**, not to time.
:::
:::

::: {.slide title="The Measured Error Sits on (n − 1)/d_k"}
@!matrix-state-interference-2
:::

::: {.slide title="Recall Survives an Error as Large as the Signal"}
At $d_k = d_v = 64$, $n = 64$: squared error $\approx 1$, recall 1.00.

- Decoding compares the read with 256 codebook values.
- Only the error's component **along a candidate** competes:
  variance $\approx n/(d_k d_v)$, here about $(1/8)^2$.

::: {.d2l-note}
Recall depends on $n/(d_k d_v)$, not on the squared error alone.
:::
:::

::: {.slide title="Four Times the Entries Hold Four Times the Pairs"}
Values as wide as keys, so the state has $d_k^2$ entries:

| $d_k$ | entries | recall 0.96 at |
|---|---|---|
| 32 | 1,024 | 32 pairs |
| 64 | 4,096 | 128 pairs |
| 128 | 16,384 | 512 pairs (0.94) |

::: {.d2l-note}
About one pair per 32 entries at 95% recall: capacity grows with the
entries $d_k d_v$, as long as keys are not much narrower than values.
:::
:::

::: {.slide title="Check: Where Does the Squared Error Reach 1?"}
$d_k = d_v = 64$, random unit keys and values.

1. At which $n$ is the expected squared error 1?
2. Why is recall still 0.96 at twice that?

::: {.fragment .fade-up}
1. $(n - 1)/64 = 1$: $n = 65$.
2. Only the error's component along a codebook value matters; at
   $n = 128$ it has a standard deviation of about $0.18$ against a signal
   of 1.
:::
:::

::: {.slide title="Correlated Keys Add an Error No Width Removes"}
Every key shares a direction with weight $\rho$: overlap $\approx \rho^2$ +
noise.

$$\mathbb{E}\,\|\mathbf{S}\mathbf{k}_j - \mathbf{v}_j\|^2 \approx (n - 1)\Big(\rho^4 + \frac{(1-\rho^2)^2}{d_k}\Big)$$

- $\rho = 0.5$, $d_k = 64$: 4.6 times the error of random keys.
- Recall of 64 pairs: 0.49 instead of 1.00.
- $\rho^4$ does not shrink as $d_k$ grows.
:::

::: {.slide title="Decay Keeps About the Last 1/(1 − a) Pairs"}
$$\mathbf{S}_t = a_t\,\mathbf{S}_{t-1} + \mathbf{v}_t\mathbf{k}_t^\top$$

- Constant $a$: weights $a^\ell$ sum to $1/(1-a)$: 10 pairs at $a = 0.9$,
  100 at $a = 0.99$.
- Interference adds squares: like $1/(1-a^2)$ pairs, about half as many.
- Fixed per head (RetNet); from the input, per head (Mamba-2); per key
  coordinate (gated linear attention).
:::

::: {.slide title="Mamba's State Is a Matrix Memory With a Decay per Entry"}
$H$ channels, $N$ states each, rows of $\mathbf{S}_t \in \mathbb{R}^{H \times N}$:

$$\mathbf{S}_t = \mathbf{E}_t \odot \mathbf{S}_{t-1} + (\boldsymbol{\Delta}_t \odot \mathbf{x}_t)\,\mathbf{B}_t^\top, \qquad \mathbf{y}_t = \mathbf{S}_t\mathbf{C}_t$$

| memory | Mamba |
|---|---|
| value | $\boldsymbol{\Delta}_t \odot \mathbf{x}_t$ (step-scaled input) |
| key, query | $\mathbf{B}_t$, $\mathbf{C}_t$ |
| decay | $\mathbf{E}_t = \exp(\textrm{diag}(\boldsymbol{\Delta}_t)\,\boldsymbol{\Lambda})$, per entry |

$\boldsymbol{\Delta}_t$: the $H$ step sizes; $\boldsymbol{\Lambda}$: the
$N$ rates of each channel, $H \times N$. Mamba-2: one decay per head.
:::

::: {.slide title="A Decayed Recurrence Is Attention With a Decaying Mask"}
$$\mathbf{Y} = (\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top)\,\mathbf{V},\qquad L_{ts} = a_{s+1}\cdots a_t \ (s \le t),\ 0 \textrm{ otherwise}$$

- All $a_t = 1$: $\mathbf{L}$ is the causal mask.
- Form $\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top$ first: attention,
  quadratic in $T$, parallel.
- Sum from the left: the recurrence, linear in $T$, sequential.

::: {.d2l-note}
Two orders of computing the same products.
:::
:::

::: {.slide title="Chunks: Attention Inside, a Recurrence Across"}
::: {.cols .vc}
::: {.col}
1. **Within** chunks of $m$: masked attention.
2. **Across:** $\mathbf{S}^{(c)} = \gamma^{(c)}\mathbf{S}^{(c-1)} + \sum_t L_{mt}\mathbf{v}_t\mathbf{k}_t^\top$, a scan over $T/m$ chunks.
3. **Read:** each step reads the state at its chunk's start.

$\gamma^{(c)}$: product of the chunk's decays; $L_{mt}$: decay from step
$t$ to the chunk's end. Step 2 is the new idea.
:::

::: {.col .fig .big}
![](../img/mdl-modernrnn-semiseparable.svg)
:::
:::
:::

::: {.slide title="Recurrence, Dual Form, and Chunks Agree to a Few Millionths"}
@!matrix-state-chunked-computation-2

One random sequence: $T = 512$, $d_k = d_v = 64$, decays in
$(0.8, 0.99)$. The chunk size changes the work, not the result.
:::

::: {.slide title="Check: How Much Memory Does Each Order Need?"}
One head, $T = 4096$, $d_k = d_v = 64$, 32-bit floats.

1. The dual form's $T \times T$ matrix?
2. Chunks of 64: $T \times 64$ mask entries plus one $64 \times 64$
   state per chunk?

::: {.fragment .fade-up}
1. $4096^2 \times 4$ bytes $= 64$ MB, growing with $T^2$.
2. $1$ MB $+ 64 \times 16$ KB $\approx 2$ MB, growing with $T$.
:::
:::

::: {.slide title="Gated Linear Attention: The Memory Plus Three Parts"}
Per head: query, key, value, decay $a_t = \sigma(\mathbf{w}^\top\mathbf{x}_t + b)$; chunked memory.

- **Short causal convolution (width 4) + SiLU** on $\mathbf{q}, \mathbf{k}, \mathbf{v}$: a name and its value, on neighboring tokens, become one pair.
- Unit-length queries and keys, as in the capacity law.
- Output: normalized per head (RMS), gated by $\textrm{SiLU}(\mathbf{W}_g\mathbf{x}_t)$.

::: {.d2l-note}
The convolution matters most: without it, a key and its value never meet
in one write, and every run of the hybrids section without it recalls
fewer pairs.
:::
:::

::: {.slide title="Associative Recall: Pairs, Then Every Key as a Query"}
@!matrix-state-associative-recall-1

- Key token, value token; filler; every key again as a query.
- Ids 0–3 values, 4–515 keys, 516 filler. Key 273 written twice: the
  answer is the newer value, 0.
:::

::: {.slide title="A Larger State Holds More Pairs at the Same Parameter Count"}
@!matrix-state-associative-recall-4

Width 32 cut into 8, 4, or 2 heads: about 37,000 parameters each, two
runs each.
:::

::: {.slide title="Twice the Entries, About Twice the Pairs"}
- 4 pairs: every model recalls 98% or more.
- 64 pairs: 128 entries about half, 256 entries about three quarters.
- 256 entries with $2n$ pairs $\approx$ 128 entries with $n$.
- 512 entries: best on average at 32 and 64 pairs, but its two runs
  differ the most.

::: {.d2l-note}
From 128 to 256 entries, a trained network follows the law of random
keys: twice the entries, about twice the pairs.
:::
:::

::: {.slide title="Check: A Key Written Twice"}
A program sets `rate = 0.05`, later `rate = 0.07`. What does the memory
return for the key of `rate`?

::: {.fragment .fade-up}
The **sum** of both values, each scaled by the decays since its write. A
decay fades the old value, and every other pair with it.
:::
:::

::: {.slide title="Recap"}
| | write | transition | computed as |
|---|---|---|---|
| linear attention | $\mathbf{v}\mathbf{k}^\top$ | $\mathbf{S}$ | recurrence or masked attention |
| RetNet | $\mathbf{v}\mathbf{k}^\top$ | $a\mathbf{S}$, $a$ fixed per head | chunks |
| Mamba-2, this section | $\mathbf{v}\mathbf{k}^\top$ | $a_t\mathbf{S}$, $a_t$ from the input | chunks |
| Mamba | $(\boldsymbol{\Delta}\odot\mathbf{x})\mathbf{B}^\top$ | $\mathbf{E}_t \odot \mathbf{S}$ | scan |

Read error $(n-1)/d_k$; recall grows with the entries $d_k d_v$: a
$64 \times 64$ state recalls 128 random pairs at 96%. Decay keeps about
$1/(1-a)$ pairs; trained, 128 → 256 entries hold about twice the pairs.

**Next:** correct a pair instead of adding to it.
:::
