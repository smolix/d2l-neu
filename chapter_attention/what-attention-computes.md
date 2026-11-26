# Attention Circuits
:label:`sec_what-attention-computes`

As noted in :numref:`sec_queries-keys-values`, an attention heatmap is a
hypothesis about what a layer computes, not a proof. The heatmap shows where
a head looks, which is only half of its computation; what the head actually
writes to the output depends on its value and output projections. A head that
attends sharply to a token may copy that token's identity, extract a single
subspace feature, or contribute a negligible vector.

To understand what trained attention heads compute, we adopt the mechanistic
circuit formulation of :citet:`Elhage.Nanda.Olsson.ea.2021` on the
attention-only language model `TinyCharLM` (:numref:`sec_positional-information`).
Each head factors into two rank-$d_h$ bilinear/linear operators on the
*residual stream*: a *QK circuit* that determines which positions are
attended to, and an *OV circuit* that determines how attended vectors alter
output logits. Using this decomposition, we show why a one-layer model can
express bigrams and skip-trigrams (including single-token copying) but cannot
perform in-context pattern completion ($[\mathrm{A}][\mathrm{B}] \ldots
[\mathrm{A}] \to [\mathrm{B}]$), whereas a two-layer model can compose a
*previous-token head* with a *match-and-copy head* to form an *induction
head* :cite:`Olsson.Elhage.Nanda.ea.2022`. Finally, we test this circuit
hypothesis via weight inspection and causal head ablations.

```{.python .input #what-attention-computes}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #what-attention-computes}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #what-attention-computes-attention-circuits-1}
%%tab pytorch
import copy
```

## Circuits and the Residual Stream

### QK and OV Circuits

In `TinyCharLM` (:numref:`subsec_tinycharlm`), token embeddings
$\mathbf{e}_{x_i} \in \mathbb{R}^d$ ($d = 128$) pass through $N$ residual
blocks, each consisting of $H = 4$ causal attention heads of width $d_h =
d/H = 32$ with no MLP, normalization, or bias terms, followed by a tied
unembedding projection $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$.

Writing token vectors as column vectors $\mathbf{h}_j \in \mathbb{R}^d$, each
head projects $\mathbf{h}_j$ via $\mathbf{W}_v \in \mathbb{R}^{d_h \times d}$
and forms the weighted sum $\sum_{j \le i} \alpha_{ij} \mathbf{W}_v
\mathbf{h}_j$. Partitioning the block's output matrix into $H$ column slices
$\mathbf{W}_o \in \mathbb{R}^{d \times d_h}$ (one per head) expresses the
concatenation-and-projection step as a sum of independent head contributions:

$$
\mathbf{W}_o \sum_{j \leq i} \alpha_{ij}\, \mathbf{W}_v \mathbf{h}_j = \sum_{j \leq i} \alpha_{ij}\, \mathbf{W}_o \mathbf{W}_v\, \mathbf{h}_j.
$$

Similarly, the pre-softmax score between query position $i$ and key position
$j$ is $(\mathbf{W}_q \mathbf{h}_i)^\top (\mathbf{W}_k \mathbf{h}_j) /
\sqrt{d_h}$ with $\mathbf{W}_q, \mathbf{W}_k \in \mathbb{R}^{d_h \times d}$.
Each head is therefore completely characterized by two $d \times d$ matrices
of rank at most $d_h$ :cite:`Elhage.Nanda.Olsson.ea.2021`:

$$
s_{ij} = \frac{\mathbf{h}_i^\top \mathbf{W}_{\mathrm{QK}}\, \mathbf{h}_j}{\sqrt{d_h}}, \qquad \mathbf{W}_{\mathrm{QK}} = \mathbf{W}_q^\top \mathbf{W}_k, \qquad \mathbf{W}_{\mathrm{OV}} = \mathbf{W}_o \mathbf{W}_v.
$$
:eqlabel:`eq_qkov`

* The **QK circuit** $\mathbf{W}_{\mathrm{QK}}$ (or $\mathbf{W}_q^\top
  \mathbf{R}_{j-i} \mathbf{W}_k$ under RoPE, :numref:`subsec_pos-rope`)
  governs *where* the head attends.
* The **OV circuit** $\mathbf{W}_{\mathrm{OV}}$ governs *what* vector the
  head writes into the destination position when a source position is
  attended to.

Studying the products $\mathbf{W}_{\mathrm{QK}}$ and
$\mathbf{W}_{\mathrm{OV}}$ rather than the individual factors eliminates
gauge freedom ($\mathbf{W}_o \mathbf{G}^{-1}$ and $\mathbf{G}\mathbf{W}_v$
yield the exact same $\mathbf{W}_{\mathrm{OV}}$ for any invertible
$\mathbf{G} \in \mathbb{R}^{d_h \times d_h}$) and makes the rank-$d_h$
subspace bottleneck explicit.

### Residual Representations Across Layers

Setting $\mathbf{h}_i^{(0)} = \mathbf{e}_{x_i}$, block $\ell \in \{1, \ldots,
N\}$ updates the representation at position $i$ additively:

$$
\mathbf{h}_i^{(\ell)} = \mathbf{h}_i^{(\ell-1)} + \sum_{m=1}^{H} \sum_{j \leq i} \alpha_{ij}^{\ell m}\, \mathbf{W}_{\mathrm{OV}}^{\ell m}\, \mathbf{h}_j^{(\ell-1)},
$$
:eqlabel:`eq_stream-update`

and the tied output head produces logits $\mathbf{z}_i = \mathbf{E}\,
\mathbf{h}_i^{(N)}$. The sequence $\mathbf{h}_i^{(0)}, \ldots,
\mathbf{h}_i^{(N)}$ is the *residual stream* at position $i$
(:numref:`fig_residual-stream`): a shared communication channel into which
every head reads and writes additively.

![The residual stream. Each position carries a vector from its embedding to the output logits. The QK circuit determines the attention weights $\alpha_{3j}$, and the OV circuit determines the contribution of each attended vector to the destination stream.](../img/mdl-attention-residual-stream.svg)
:label:`fig_residual-stream`

### Bigrams, Skip-Trigrams, and the Limits of One Layer

Expanding :eqref:`eq_stream-update` by depth reveals what functions $N$
attention layers can express:

* **Zero blocks ($N = 0$):** $\mathbf{z}_i = \mathbf{E}\,\mathbf{e}_{x_i}$,
  modeling only constrained token-bigram logits via the symmetric, positive
  semidefinite Gram matrix $\mathbf{E}\mathbf{E}^\top$ (Exercise :exref:`ex_what-attention-computes-tied-bigrams`).
* **One block ($N = 1$):**

  $$
  \mathbf{z}_i = \mathbf{E}\,\mathbf{e}_{x_i} + \sum_{m=1}^{H} \sum_{j \leq i} \alpha_{ij}^{m}\; \mathbf{E}\,\mathbf{W}_{\mathrm{OV}}^{m}\,\mathbf{e}_{x_j}.
  $$
  :eqlabel:`eq_one-layer-paths`

  Each head term implements a *skip-trigram* rule of the form $x_j \ldots
  x_i \to x_{\textrm{next}}$ (e.g., attending from *"in"* to an earlier
  *"keep"* and boosting *"mind"*, or *copying* by boosting $x_j$ itself).
* **Two blocks ($N = 2$) and the induction head:** Suppose a context contains
  $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}]$ and the model must predict
  $[\mathrm{B}]$ by in-context lookup. A one-block model cannot do this by
  content matching: to copy $[\mathrm{B}]$ from position $j$, a head must
  attend to position $j$, but in :eqref:`eq_one-layer-paths` the key at
  position $j$ depends only on $\mathbf{e}_{x_j} = \mathbf{e}_{[\mathrm{B}]}$
  and knows nothing about its predecessor $x_{j-1} = [\mathrm{A}]$.

  Two blocks solve this via *K-composition* (:numref:`fig_induction-circuit`,
  Exercise :exref:`ex_what-attention-computes-paths`): in block 1, a
  *previous-token head* attends to offset $-1$ and writes $x_{j-1} =
  [\mathrm{A}]$ into the residual stream $\mathbf{h}_j^{(1)}$ at position $j$;
  in block 2, an *induction head* matches its query $[\mathrm{A}]$ against
  that key feature in $\mathbf{h}_j^{(1)}$ and uses its OV circuit to copy
  $x_j = [\mathrm{B}]$ into the prediction logits.

![The induction circuit spans two layers. In layer 1 a previous-token head writes each token's predecessor into its stream, so the key at B's position can announce "follows A". In layer 2 the induction head's QK circuit matches the current A against that announcement, and its OV circuit copies B into the prediction.](../img/mdl-attention-induction-circuit.svg)
:label:`fig_induction-circuit`

## Copying and Induction

### A Repeated-Sequence Task

To test whether gradient descent discovers this two-layer circuit, we train on
synthetic sequences of *repeated random tokens*: each sequence draws a random
pattern of length $L \in [L_{\min}, L_{\max}]$ uniformly from a vocabulary of
size $64$ and tiles it to length $64$. Because every pattern is drawn fresh,
the first copy ($t < L$) is unpredictable (chance cross-entropy $\log 64
\approx 4.16$ nats). So is the token at position $t = L$, where the pattern
restarts: nothing in its context has repeated yet, and the model cannot know
$L$. We therefore count the targets $t \le L$ as the first copy. Later copies
($t > L$) can be predicted from the context by in-context lookup.

```{.python .input #what-attention-computes-repetition-as-a-task-1}
%%tab pytorch
def repeated_batch(batch_size, num_steps, vocab, Lmin, Lmax, device=None):
    """Random patterns of length Lmin..Lmax, tiled to num_steps tokens."""
    L = torch.randint(Lmin, Lmax + 1, (batch_size,), device=device)
    s = torch.randint(0, vocab, (batch_size, num_steps), device=device)
    pos = torch.arange(num_steps, device=device)
    return torch.gather(s, 1, pos[None, :] % L[:, None]), L

x, L = repeated_batch(batch_size=2, num_steps=16, vocab=10, Lmin=5, Lmax=8)
print(f'pattern lengths: {L.tolist()}')
print(x)
```

```{.python .input #what-attention-computes-repetition-as-a-task-1}
%%tab jax
def repeated_batch(key, batch_size, num_steps, vocab, Lmin, Lmax):
    """Random patterns of length Lmin..Lmax, tiled to num_steps tokens."""
    key_L, key_s = jax.random.split(key)
    L = jax.random.randint(key_L, (batch_size,), Lmin, Lmax + 1)
    s = jax.random.randint(key_s, (batch_size, num_steps), 0, vocab)
    pos = jnp.arange(num_steps)
    return jnp.take_along_axis(s, pos[None, :] % L[:, None], axis=1), L

x, L = repeated_batch(d2l.get_key(), batch_size=2, num_steps=16,
                      vocab=10, Lmin=5, Lmax=8)
print(f'pattern lengths: {L.tolist()}')
print(x)
```

`eval_copy` measures the cross-entropy on the first copy ($t \le L$), the
cross-entropy on later copies ($t > L$), and the top-1 accuracy on later
copies:

```{.python .input #what-attention-computes-repetition-as-a-task-2}
%%tab pytorch
@torch.inference_mode()
def eval_copy(model, x, L):
    """First-copy loss, later-copy loss, and later-copy accuracy."""
    logits, Y = model(x[:, :-1]), x[:, 1:]
    ce = F.cross_entropy(logits.transpose(1, 2), Y, reduction='none')
    later = torch.arange(1, Y.shape[1] + 1, device=Y.device) > L[:, None]
    acc = (logits.argmax(-1) == Y)[later].float().mean()
    return torch.stack((ce[~later].mean(), ce[later].mean(), acc)).tolist()

class RepeatedTokens(d2l.DataModule):
    """An endless stream of fresh repeated-token minibatches."""
    def __init__(self, Lmin, Lmax, batch_size=128, num_steps=64, vocab=64):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        while True:
            x, _ = repeated_batch(self.batch_size, self.num_steps,
                                  self.vocab, self.Lmin, self.Lmax)
            yield x[:, :-1], x[:, 1:]

def train_repeat(model, num_steps, Lmin, Lmax, eval_every=50):
    """Train with Adam; return (first, later) copy losses per round."""
    data, device = RepeatedTokens(Lmin, Lmax), d2l.try_gpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for _ in range(num_steps // eval_every):
        d2l.train_steps(model, data, optimizer, eval_every)
        x, L = repeated_batch(256, 64, 64, Lmin, Lmax, device)
        history.append(eval_copy(model, x, L)[:2])
    return history

def drop_step(hist, eval_every=50):
    """First step at which the later-copy loss is below 1 nat, or None."""
    return next((eval_every * (k + 1) for k, (_, rep) in enumerate(hist)
                 if rep < 1), None)
```

```{.python .input #what-attention-computes-repetition-as-a-task-2}
%%tab jax
def eval_copy(model, x, L):
    """First-copy loss, later-copy loss, and later-copy accuracy."""
    logits, Y = model(x[:, :-1]), x[:, 1:]
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    later = jnp.arange(1, Y.shape[1] + 1)[None, :] > L[:, None]
    acc = ((logits.argmax(-1) == Y) * later).sum() / later.sum()
    first = (ce * ~later).sum() / (~later).sum()
    rep = (ce * later).sum() / later.sum()
    return [float(v) for v in jax.device_get((first, rep, acc))]

class RepeatedTokens(d2l.DataModule):
    """An endless stream of fresh repeated-token minibatches."""
    def __init__(self, Lmin, Lmax, batch_size=128, num_steps=64, vocab=64):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        while True:
            x, _ = repeated_batch(d2l.get_key(), self.batch_size,
                                  self.num_steps, self.vocab, self.Lmin,
                                  self.Lmax)
            yield x[:, :-1], x[:, 1:]

def train_repeat(model, num_steps, Lmin, Lmax, eval_every=50):
    """Train with Adam; return (first, later) copy losses per round."""
    data = RepeatedTokens(Lmin, Lmax)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    history = []
    for _ in range(num_steps // eval_every):
        d2l.train_steps(model, data, optimizer, eval_every)
        x, L = repeated_batch(d2l.get_key(), 256, 64, 64, Lmin, Lmax)
        history.append(eval_copy(model, x, L)[:2])
    return history

def drop_step(hist, eval_every=50):
    """First step at which the later-copy loss is below 1 nat, or None."""
    return next((eval_every * (k + 1) for k, (_, rep) in enumerate(hist)
                 if rep < 1), None)
```

### A Fixed-Offset Solution
:label:`subsec_fixed-offset-shortcut`

Before testing variable periods, suppose the pattern period is fixed at
$L = 32$. A *one-block* model (`num_blks=1, bias=True`) easily achieves $100\%$
second-copy accuracy:

```{.python .input #what-attention-computes-the-positional-shortcut-1}
%%tab pytorch
model_shortcut = d2l.TinyCharLM(vocab_size=64, num_blks=1, pos='rope',
                                bias=True)
train_repeat(model_shortcut, 800, Lmin=32, Lmax=32)
first, rep, acc = eval_copy(model_shortcut, *repeated_batch(
    256, 64, 64, 32, 32, d2l.try_gpu()))
print(f'first copy {first:.2f} nats, second copy {rep:.2f} nats, '
      f'second-copy accuracy {acc:.2f}')
```

```{.python .input #what-attention-computes-the-positional-shortcut-1}
%%tab jax
model_shortcut = d2l.TinyCharLM(vocab_size=64, num_blks=1, pos='rope',
                                bias=True, rngs=nnx.Rngs(d2l.get_key()))
train_repeat(model_shortcut, 800, Lmin=32, Lmax=32)
first, rep, acc = eval_copy(model_shortcut, *repeated_batch(
    d2l.get_key(), 256, 64, 64, 32, 32))
print(f'first copy {first:.2f} nats, second copy {rep:.2f} nats, '
      f'second-copy accuracy {acc:.2f}')
```

How can a single layer copy the successor when our circuit analysis argued
that content-based induction requires two layers? When $L = 32$ is constant,
the target token at position $i + 1$ always equals the token at fixed offset
$j = i - 31$. A single RoPE head with constant bias queries and keys can
attend purely by relative offset $-31$ without matching content at all
(Exercise :exref:`ex_what-attention-computes-shortcut-bias`). Evaluating
`model_shortcut` on patterns of period $L = 24$ (where the target lies at
offset $-23$) immediately exposes the shortcut: attention remains locked at
offset $-31$ and accuracy collapses to chance ($1/64 \approx 0.02$):

```{.python .input #what-attention-computes-the-positional-shortcut-2}
%%tab pytorch
x, L = repeated_batch(256, 64, 64, 24, 24, d2l.try_gpu())
first, rep, acc = eval_copy(model_shortcut, x, L)
with torch.inference_mode():
    m = model_shortcut.attention_weights(x[:, :-1])[0]
print(f'patterns of length 24: second-copy accuracy {acc:.2f}')
for offset in (31, 23):
    mass = m.diagonal(-offset, dim1=2, dim2=3).mean((0, 2))
    print(f'mass at offset {offset}, per head: '
          + ' '.join(f'{v:.2f}' for v in mass.tolist()))
```

```{.python .input #what-attention-computes-the-positional-shortcut-2}
%%tab jax
x, L = repeated_batch(d2l.get_key(), 256, 64, 64, 24, 24)
first, rep, acc = eval_copy(model_shortcut, x, L)
m = model_shortcut.attention_weights(x[:, :-1])[0]
print(f'patterns of length 24: second-copy accuracy {acc:.2f}')
for offset in (31, 23):
    mass = jnp.diagonal(m, -offset, axis1=2, axis2=3).mean((0, 2))
    print(f'mass at offset {offset}, per head: '
          + ' '.join(f'{v:.2f}' for v in mass.tolist()))
```

### A Two-Block Induction Circuit
:label:`subsec_two-block-induction`

Sampling the pattern period uniformly from $L \in \{16, \ldots, 32\}$
eliminates any fixed-offset shortcut. We train three independent two-block
models and a one-block control (all with `bias=False`) for $5{,}000$ steps:

```{.python .input #what-attention-computes-two-blocks-learn-to-look-things-up-1}
%%tab pytorch
runs = []  # (blocks, model, history): three two-block runs, one control
for num_blks in (2, 2, 2, 1):
    model = d2l.TinyCharLM(vocab_size=64, num_blks=num_blks, pos='rope')
    runs.append((num_blks, model, train_repeat(model, 5000, 16, 32)))
models_two = [model for _, model, _ in runs[:3]]
_, model_two, hist_two = runs[0]   # Run 1: maps and curves shown for it
_, model_one, hist_one = runs[3]
x, L = repeated_batch(256, 64, 64, 16, 32, d2l.try_gpu())
print('blocks  first copy  later copies  accuracy  below 1 nat at step')
for num_blks, model, hist in runs:
    first, rep, acc = eval_copy(model, x, L)
    print(f'{num_blks:6d}{first:12.2f}{rep:14.2f}{acc:10.2f}'
          f'{drop_step(hist) or "never":>21}')
```

```{.python .input #what-attention-computes-two-blocks-learn-to-look-things-up-1}
%%tab jax
runs = []  # (blocks, model, history): three two-block runs, one control
for num_blks in (2, 2, 2, 1):
    model = d2l.TinyCharLM(vocab_size=64, num_blks=num_blks, pos='rope',
                           rngs=nnx.Rngs(d2l.get_key()))
    runs.append((num_blks, model, train_repeat(model, 5000, 16, 32)))
models_two = [model for _, model, _ in runs[:3]]
_, model_two, hist_two = runs[0]   # Run 1: maps and curves shown for it
_, model_one, hist_one = runs[3]
x, L = repeated_batch(d2l.get_key(), 256, 64, 64, 16, 32)
print('blocks  first copy  later copies  accuracy  below 1 nat at step')
for num_blks, model, hist in runs:
    first, rep, acc = eval_copy(model, x, L)
    print(f'{num_blks:6d}{first:12.2f}{rep:14.2f}{acc:10.2f}'
          f'{drop_step(hist) or "never":>21}')
```

All three two-block runs achieve more than $90\%$ later-copy accuracy (losses of
roughly $0.2$--$0.4$ nats), whereas the one-block model stalls above $3$ nats
($\sim 10\%$ accuracy, only boosting the unigram set of tokens present in the
context). The first-copy loss ends slightly *above* the chance value $\log 64$
in every run. All four models have learned to favor tokens that already
occurred in the context, and in the first period this confidence is misplaced:
when a token repeats there by chance, a two-block model predicts its earlier
successor. Furthermore, the training curve exhibits an abrupt phase transition
after an initial plateau:

```{.python .input #what-attention-computes-two-blocks-learn-to-look-things-up-2}
%%tab pytorch, jax
steps = list(range(50, 5001, 50))
d2l.plot(steps, [[h[1] for h in hist_two], [h[1] for h in hist_one],
                 [h[0] for h in hist_two], [math.log(64)] * len(steps)],
         'step', 'loss (nats)',
         legend=['later copies, 2 blocks', 'later copies, 1 block',
                 'first copy, 2 blocks', 'log 64'], figsize=(5, 3))
```

## Evidence for the Circuit

To verify whether `model_two` actually implements the two-step induction
circuit of :numref:`fig_induction-circuit`, we inspect (1) its attention maps,
(2) its OV copy matrices, and (3) head ablations.

### Attention Patterns of the Learned Heads

On a test sequence of period $L = 24$, our circuit predicts that at least one
block-1 head should attend one position back ($j = i - 1$), and block-2 heads
should attend to the induction target $j = i + 1 - L = i - 23$ once $i \ge
24$:

```{.python .input #what-attention-computes-the-heads-caught-in-the-act-1}
%%tab pytorch
x, L = repeated_batch(1, 64, 64, 24, 24, d2l.try_gpu())
with torch.inference_mode():
    maps = torch.stack(model_two.attention_weights(x[:, :-1]))
d2l.show_heatmaps(maps[:, 0], xlabel='key position',
                  ylabel='query position',
                  titles=[f'head {h}' for h in range(4)],
                  figsize=(9, 4.5), cmap='Blues')
```

```{.python .input #what-attention-computes-the-heads-caught-in-the-act-1}
%%tab jax
x, L = repeated_batch(d2l.get_key(), 1, 64, 64, 24, 24)
maps = jnp.stack(model_two.attention_weights(x[:, :-1]))
d2l.show_heatmaps(maps[:, 0], xlabel='key position',
                  ylabel='query position',
                  titles=[f'head {h}' for h in range(4)],
                  figsize=(9, 4.5), cmap='Blues')
```

To quantify both roles across variable pattern lengths $L \in \{16, \ldots,
32\}$ and across all three runs, `head_scores` computes each head's mean
attention weight on the previous token ($j = i - 1$) and on the induction
target ($j = i + 1 - L$ for $i \ge L$):

```{.python .input #what-attention-computes-the-heads-caught-in-the-act-2}
%%tab pytorch
@torch.inference_mode()
def head_scores(model, x, L):
    """Mean attention on the previous token and on the induction target:
    shape (blocks, 2, heads)."""
    maps = torch.stack(model.attention_weights(x[:, :-1]))
    i = torch.arange(maps.shape[-1], device=x.device)
    tgt = i + 1 - L[:, None]               # Induction target of query i
    is_tgt = i == tgt[..., None]           # (batch, query, key)
    valid = tgt >= 1                       # Query token occurred before
    prev = maps.diagonal(-1, dim1=3, dim2=4).mean((1, 3))
    ind = (maps * is_tgt[:, None]).sum(-1)  # (blocks, batch, heads, query)
    ind = (ind * valid[:, None]).sum((1, 3)) / valid.sum()
    return torch.stack((prev, ind), 1)

x, L = repeated_batch(256, 64, 64, 16, 32, d2l.try_gpu())
all_scores = [head_scores(model, x, L) for model in models_two]
scores = all_scores[0]  # Run 1, whose maps are shown above
for r, run_scores in enumerate(all_scores):
    for b, rows in enumerate(run_scores.tolist()):
        for name, row in zip(('previous', 'induction'), rows):
            print(f'run {r + 1}, block {b + 1}, {name:<9}: '
                  + ' '.join(f'{v:.2f}' for v in row))
```

```{.python .input #what-attention-computes-the-heads-caught-in-the-act-2}
%%tab jax
def head_scores(model, x, L):
    """Mean attention on the previous token and on the induction target:
    shape (blocks, 2, heads)."""
    maps = jnp.stack(model.attention_weights(x[:, :-1]))
    i = jnp.arange(maps.shape[-1])
    tgt = i + 1 - L[:, None]               # Induction target of query i
    is_tgt = i == tgt[..., None]           # (batch, query, key)
    valid = tgt >= 1                       # Query token occurred before
    prev = jnp.diagonal(maps, -1, axis1=3, axis2=4).mean((1, 3))
    ind = (maps * is_tgt[:, None]).sum(-1)  # (blocks, batch, heads, query)
    ind = (ind * valid[:, None]).sum((1, 3)) / valid.sum()
    return jnp.stack((prev, ind), 1)

x, L = repeated_batch(d2l.get_key(), 256, 64, 64, 16, 32)
all_scores = [head_scores(model, x, L) for model in models_two]
scores = all_scores[0]  # Run 1, whose maps are shown above
for r, run_scores in enumerate(all_scores):
    for b, rows in enumerate(run_scores.tolist()):
        for name, row in zip(('previous', 'induction'), rows):
            print(f'run {r + 1}, block {b + 1}, {name:<9}: '
                  + ' '.join(f'{v:.2f}' for v in row))
```

In every run, between one and three block-1 heads specialize as previous-token heads,
while block-2 heads concentrate attention on the induction target $j = i + 1
- L$.

### The Copy Matrix

Attending to position $j$ predicts $x_j$ only if block 2's OV circuit boosts
the logit of token $x_j$. Let

$$
C_{ab} = \mathbf{e}_b^\top \mathbf{W}_{\mathrm{OV}}\, \bar{\mathbf{h}}_a
$$
:eqlabel:`eq_copy-matrix`

denote the logit boost assigned to token $b$ when the summed OV circuit of a
block reads the mean residual-stream vector $\bar{\mathbf{h}}_a$ at positions
holding token $a$ (or the raw embedding $\mathbf{e}_a$). A copying OV circuit
has its row maximum on the diagonal ($b = a$) across nearly all $64$ rows
(versus $1/64 \approx 0.02$ by chance):

```{.python .input #what-attention-computes-copy-matrix-definitions}
%%tab pytorch
def circuits(model, blk):
    """W_QK and W_OV of every head of block blk, each (heads, D, D)."""
    # nn.Linear stores its weight as (output, input), the orientation of the
    # column-vector equations, so no transpose is needed
    W_qkv = model.blks[blk]['qkv'].weight           # (3D, D): W_q; W_k; W_v
    W_o = model.blks[blk]['proj'].weight            # (D, D): heads in columns
    H, D = model.num_heads, W_o.shape[0]
    W_q, W_k, W_v = W_qkv.reshape(3, H, D // H, D)  # Each (H, d_h, D)
    W_QK = W_q.mT @ W_k
    W_OV = W_o.reshape(D, H, D // H).transpose(0, 1) @ W_v
    return W_QK, W_OV

@torch.inference_mode()
def copy_matrix(model, blk, X=None):
    """C[a, b] = e_b^T W_OV h_a: W_OV summed over the heads of blk, h_a the
    mean stream vector entering blk at the positions of token a in X, or the
    embedding e_a if X is None."""
    E = h = model.token_emb.weight
    if X is not None:
        S = model.stream(X, blk)[0]            # The stream entering blk
        h = torch.zeros_like(E).index_add_(0, X.flatten(), S.flatten(0, 1))
        h = h / torch.bincount(X.flatten(), minlength=len(E))[:, None]
    W_OV = circuits(model, blk)[1].sum(0)
    return h @ W_OV.T @ E.T

def diagonal_share(C):
    """Fraction of the rows of C whose largest entry is on the diagonal."""
    rows = torch.arange(len(C), device=C.device)
    return float((C.argmax(-1) == rows).float().mean())
```

```{.python .input #what-attention-computes-copy-matrix-definitions}
%%tab jax
def circuits(model, blk):
    """W_QK and W_OV of every head of block blk, each (heads, D, D)."""
    # nnx.Linear stores its kernel as (input, output), the transpose of the
    # column-vector equations, so both kernels are transposed first
    W_qkv = model.blks[blk]['qkv'].kernel[...].T    # (3D, D): W_q; W_k; W_v
    W_o = model.blks[blk]['proj'].kernel[...].T     # (D, D): heads in columns
    H, D = model.num_heads, W_o.shape[0]
    W_q, W_k, W_v = W_qkv.reshape(3, H, D // H, D)  # Each (H, d_h, D)
    W_QK = W_q.mT @ W_k
    W_OV = W_o.reshape(D, H, D // H).swapaxes(0, 1) @ W_v
    return W_QK, W_OV

def copy_matrix(model, blk, X=None):
    """C[a, b] = e_b^T W_OV h_a: W_OV summed over the heads of blk, h_a the
    mean stream vector entering blk at the positions of token a in X, or the
    embedding e_a if X is None."""
    E = h = model.token_emb.embedding[...]
    if X is not None:
        S = model.stream(X, blk)[0]            # The stream entering blk
        h = jnp.zeros_like(E).at[X.flatten()].add(S.reshape(-1, E.shape[1]))
        h = h / jnp.bincount(X.flatten(), length=len(E))[:, None]
    W_OV = circuits(model, blk)[1].sum(0)
    return h @ W_OV.T @ E.T

def diagonal_share(C):
    """Fraction of the rows of C whose largest entry is on the diagonal."""
    return float((C.argmax(-1) == jnp.arange(len(C))).mean())
```

Evaluating `diagonal_share(copy_matrix(...))` across all three two-block training runs (both on the mean block-1 residual stream $\bar{\mathbf{h}}_a$ and on raw token embeddings $\mathbf{e}_a$) as well as on the one-block control model shows that the OV circuit forms a near-diagonal copying matrix:

```{.python .input #what-attention-computes-the-circuit-is-in-the-weights}
%%tab pytorch
x, L = repeated_batch(256, 64, 64, 16, 32, d2l.try_gpu())
print('rows whose largest entry is the diagonal (chance 1/64)')
print(f'{"":<32}   run 1   run 2   run 3')
for name, X in (('2 blocks, block 2, mean stream', x),
                ('2 blocks, block 2, embeddings', None)):
    shares = [diagonal_share(copy_matrix(m, 1, X)) for m in models_two]
    print(f'{name:<32}' + ''.join(f'{v:8.2f}' for v in shares))
print(f'{"1 block, mean stream":<32}'
      f'{diagonal_share(copy_matrix(model_one, 0, x)):8.2f}')
d2l.show_heatmaps(copy_matrix(model_two, 1, x)[None, None].cpu(),
                  xlabel='logit boosted', ylabel='token attended to',
                  figsize=(3.6, 3.6), cmap='Blues')
```

```{.python .input #what-attention-computes-the-circuit-is-in-the-weights}
%%tab jax
x, L = repeated_batch(d2l.get_key(), 256, 64, 64, 16, 32)
print('rows whose largest entry is the diagonal (chance 1/64)')
print(f'{"":<32}   run 1   run 2   run 3')
for name, X in (('2 blocks, block 2, mean stream', x),
                ('2 blocks, block 2, embeddings', None)):
    shares = [diagonal_share(copy_matrix(m, 1, X)) for m in models_two]
    print(f'{name:<32}' + ''.join(f'{v:8.2f}' for v in shares))
print(f'{"1 block, mean stream":<32}'
      f'{diagonal_share(copy_matrix(model_one, 0, x)):8.2f}')
d2l.show_heatmaps(copy_matrix(model_two, 1, x)[None, None],
                  xlabel='logit boosted', ylabel='token attended to',
                  figsize=(3.6, 3.6), cmap='Blues')
```

Notice two instructive subtleties: (1) the one-block control *also* learns a
diagonal OV copy matrix, showing that OV copying without QK K-composition
does not retrieve in-context successors (see Exercise
:exref:`ex_what-attention-computes-k-composition`); and (2) block 2 copies
more reliably from the block-1 stream $\bar{\mathbf{h}}_a$ than from raw
embeddings $\mathbf{e}_a$ alone, where the diagonal share varies widely from
run to run.

### Head Ablation
:label:`subsec_head-ablation`

To test causal necessity, `ablate` zeroes out the output-projection slice
$\mathbf{W}_o^m$ of selected heads in a block. We compare removing block 1's
previous-token heads (score $> 0.2$), removing block 1's remaining heads, and
cumulatively removing block 2's induction heads from strongest to weakest:

```{.python .input #what-attention-computes-head-ablation-1}
%%tab pytorch
def ablate(model, blk, heads):
    """Copy of model with the given heads of block blk removed."""
    model = copy.deepcopy(model)
    W_o = model.blks[blk]['proj'].weight   # (D, D): heads side by side
    with torch.no_grad():
        W_o.view(W_o.shape[0], model.num_heads, -1)[:, heads] = 0
    return model
```

```{.python .input #what-attention-computes-head-ablation-1}
%%tab jax
def ablate(model, blk, heads):
    """Copy of model with the given heads of block blk removed."""
    model = nnx.clone(model)
    W_o = model.blks[blk]['proj'].kernel   # (D, D): heads stacked in rows
    W = W_o[...].reshape(model.num_heads, -1, W_o.shape[1])
    W_o[...] = W.at[jnp.array(heads, dtype=int)].set(0).reshape(W_o.shape)
    return model
```

Applying `ablate` across all three trained two-block models quantifies how much later-copy cross-entropy and copy accuracy degrade when we remove block 1's previous-token heads versus block 2's top induction heads:

```{.python .input #what-attention-computes-head-ablation-2}
%%tab pytorch
x, L = repeated_batch(256, 64, 64, 16, 32, d2l.try_gpu())
H, table = model_two.num_heads, {}  # Variant: (loss, accuracy) per run
for r, (model, s) in enumerate(zip(models_two, all_scores)):
    prev = [h for h in range(H) if s[0, 0, h] > 0.2]  # Previous-token heads
    rest = [h for h in range(H) if h not in prev]
    order = s[1, 1].argsort(descending=True).tolist()  # Strongest first
    print(f'run {r + 1}: block 1 previous-token heads {prev}, others {rest}')
    variants = {'none': model,
                'block 1, previous-token heads': ablate(model, 0, prev),
                'block 1, all other heads': ablate(model, 0, rest)}
    for k in range(1, H + 1):
        variants[f'block 2, top {k} induction'] = ablate(model, 1, order[:k])
    for name, variant in variants.items():
        table.setdefault(name, []).append(eval_copy(variant, x, L)[1:])
table['one-block model'] = [eval_copy(model_one, x, L)[1:]]
print(f'{"heads removed":<31}later-copy loss / accuracy: run 1, 2, 3')
for name, cols in table.items():
    print(f'{name:<31}'
          + ''.join(f'{rep:7.2f} /{acc:5.2f}' for rep, acc in cols))
```

```{.python .input #what-attention-computes-head-ablation-2}
%%tab jax
x, L = repeated_batch(d2l.get_key(), 256, 64, 64, 16, 32)
H, table = model_two.num_heads, {}  # Variant: (loss, accuracy) per run
for r, (model, s) in enumerate(zip(models_two, all_scores)):
    prev = [h for h in range(H) if s[0, 0, h] > 0.2]  # Previous-token heads
    rest = [h for h in range(H) if h not in prev]
    order = jnp.argsort(-s[1, 1]).tolist()  # Strongest first
    print(f'run {r + 1}: block 1 previous-token heads {prev}, others {rest}')
    variants = {'none': model,
                'block 1, previous-token heads': ablate(model, 0, prev),
                'block 1, all other heads': ablate(model, 0, rest)}
    for k in range(1, H + 1):
        variants[f'block 2, top {k} induction'] = ablate(model, 1, order[:k])
    for name, variant in variants.items():
        table.setdefault(name, []).append(eval_copy(variant, x, L)[1:])
table['one-block model'] = [eval_copy(model_one, x, L)[1:]]
print(f'{"heads removed":<31}later-copy loss / accuracy: run 1, 2, 3')
for name, cols in table.items():
    print(f'{name:<31}'
          + ''.join(f'{rep:7.2f} /{acc:5.2f}' for rep, acc in cols))
```

Ablating block 1's previous-token heads collapses later-copy accuracy to
less than $0.2$, the level of the one-block model, confirming their necessity. Interestingly, ablating block 1's
*other* heads also breaks retrieval: because `copy_matrix` relied on block 1's
transformed stream (and zero-ablation shifts stream norms in an unnormalized
network; see Exercise :exref:`ex_what-attention-computes-circuit-ablation` on
mean ablation), block 2 depends on those heads as well. In block 2, the four
heads share the induction workload redundantly, requiring multiple heads to
be ablated before accuracy drops to chance.

## In-Context Learning

### Generalization to Unseen Patterns
:label:`subsec_unseen-patterns`

Because $64^{16} \approx 8 \times 10^{28}$ possible patterns exist, every test
sequence is novel and all token pairs are equally likely across the training
distribution. The induction circuit binds each token to its successor on the
fly from a single exposure in the context window—a minimal instance of
*in-context learning*:

```{.python .input #what-attention-computes-completing-patterns-it-has-never-seen}
%%tab pytorch
x, L = repeated_batch(512, 64, 64, 24, 24, d2l.try_gpu())
with torch.inference_mode():
    acc = (model_two(x[:, :-1]).argmax(-1) == x[:, 1:]).float().mean(0)
d2l.plot([torch.arange(1, 64), [24, 24]], [acc.cpu(), [0, 1]],
         'target position', 'accuracy', fmts=('-', 'k:'))
```

```{.python .input #what-attention-computes-completing-patterns-it-has-never-seen}
%%tab jax
x, L = repeated_batch(d2l.get_key(), 512, 64, 64, 24, 24)
acc = (model_two(x[:, :-1]).argmax(-1) == x[:, 1:]).mean(0)
d2l.plot([jnp.arange(1, 64), [24, 24]], [acc, [0, 1]],
         'target position', 'accuracy', fmts=('-', 'k:'))
```

Accuracy jumps from $1/64$ on the first copy (and at the restart boundary
$t = 24$, where content matching has not yet seen a repeated token) to
about $0.8$ at position $25$ and above $0.9$ a few positions later.

What accounts for the remaining error after the restart? When a
token appears $K > 1$ times within the $L$-token pattern with different
successors, matching on the current token alone is ambiguous: choosing
uniformly among the $K$ occurrences succeeds with probability $1/K$. For
$L = 24$ and $p = 1/64$, a token repeats in the pattern with probability
$1 - (63/64)^{23} \approx 0.30$, and uniform single-token matching predicts
an overall accuracy of

$$
\mathbb{E}\left[\frac{1}{K}\right] = \frac{1 - (1 - p)^L}{L p} \approx 0.84, \qquad p = \frac{1}{64},
$$

and $\approx 0.47$ on repeated tokens alone. Splitting test positions by
$K = 1$ (unique) versus $K > 1$ (repeated) shows that the three runs reach
$0.97$ or more on unique tokens and roughly $0.7$--$0.9$ on repeated tokens, well
above the $0.47$ single-token baseline. A plausible reason, which we do not
test here, is that the model also matches longer $n$-gram context and prefers
certain RoPE offsets (Exercise :exref:`ex_what-attention-computes-paths`):

```{.python .input #what-attention-computes-generalization-to-unseen-patterns-1}
%%tab pytorch
period, num_steps = 24, 64
x, _ = repeated_batch(512, num_steps, 64, period, period, d2l.try_gpu())
pattern = x[:, :period]
K = (pattern[:, :, None] == pattern[:, None, :]).sum(-1)  # Count in pattern
K = K[:, torch.arange(period, num_steps - 1) % period]  # Current token
parts = (K == 1, K > 1, K > 0)           # Unique, repeated, all
rows = {'share of targets': [p.float().mean() for p in parts],
        'single-token match': [(1 / K[p]).mean() for p in parts]}
for r, model in enumerate(models_two):
    with torch.inference_mode():
        hit = (model(x[:, :-1]).argmax(-1) == x[:, 1:])[:, period:]
    rows[f'run {r + 1} accuracy'] = [hit[p].float().mean() for p in parts]
print(f'{"current token":<18}   unique repeated      all')
for name, row in rows.items():
    print(f'{name:<18}' + ''.join(f'{v:9.2f}' for v in row))
```

```{.python .input #what-attention-computes-generalization-to-unseen-patterns-1}
%%tab jax
period, num_steps = 24, 64
x, _ = repeated_batch(d2l.get_key(), 512, num_steps, 64, period, period)
pattern = x[:, :period]
K = (pattern[:, :, None] == pattern[:, None, :]).sum(-1)  # Count in pattern
K = K[:, jnp.arange(period, num_steps - 1) % period]  # Current token
parts = (K == 1, K > 1, K > 0)           # Unique, repeated, all
rows = {'share of targets': [p.mean() for p in parts],
        'single-token match': [(1 / K[p]).mean() for p in parts]}
for r, model in enumerate(models_two):
    hit = (model(x[:, :-1]).argmax(-1) == x[:, 1:])[:, period:]
    rows[f'run {r + 1} accuracy'] = [hit[p].mean() for p in parts]
print(f'{"current token":<18}   unique repeated      all')
for name, row in rows.items():
    print(f'{name:<18}' + ''.join(f'{v:9.2f}' for v in row))
```

### Induction Heads in Larger Models
:label:`subsec_induction-larger-models`

:citet:`Olsson.Elhage.Nanda.ea.2022` demonstrated that induction heads form
during an abrupt phase transition in language models across scales, coinciding
with the emergence of general in-context learning (Exercise
:exref:`ex_what-attention-computes-text`). Beyond copying and fuzzy pattern
completion, attention layers can also implement in-context gradient steps on
regression tasks (Exercise
:exref:`ex_what-attention-computes-in-context-regression`,
:cite:`Oswald.Niklasson.Randazzo.ea.2023`), although whether pretrained
language models learn in context this way is disputed
:cite:`Shen.Mishra.Khashabi.2024`.

Our analysis leaned on a model with no normalization and no MLP, and a real
transformer block (:numref:`chap_transformers`) has both. Each changes the
method without changing the idea. Layer normalization rescales the stream by
its own norm, so the map from stream to logits is linear only once that scale
is held fixed; circuit analyses freeze it at its observed value and treat the
rest as before. MLPs add paths that pass through no attention pattern, and a
head's output may be rewritten by an MLP before a later head reads it. Finally,
induction heads in trained language models match fuzzily: they complete
$[\mathrm{A}^*][\mathrm{B}^*] \ldots [\mathrm{A}] \to [\mathrm{B}]$ for tokens
that are merely similar, not identical, to the ones seen earlier.

## Attention Weights as Explanations

Our investigation illustrates why attention heatmaps alone are insufficient as
mechanistic explanations:

* **Important heads can have unremarkable maps:** Block 1's non-previous-token
  heads looked inert in the heatmaps, yet ablating them broke block 2's
  retrieval.
* **Sharp maps can have negligible effect:** In pretrained language models,
  idle heads frequently place most of their weight on the first token (an
  *attention sink*), whose OV vector has near-zero norm
  :cite:`Xiao.Tian.Chen.ea.2024,Gu.Pang.Du.ea.2025`.

:numref:`tab_induction-evidence` summarizes the hierarchy of evidence used to
validate the induction circuit.

:Evidence for the induction-head interpretation, by level.
:label:`tab_induction-evidence`

| check in this section | evidence level | supported conclusion | remaining limitation |
|:--|:--|:--|:--|
| later-copy loss | behavioral | two blocks solve the repeated-token task, one does not | identifies no component |
| period change | input intervention | the fixed-period model uses position, not content | changes the task as well as the input |
| head scores | activation | heads attend with the predicted offsets | omits the value transformation |
| copy matrix | weights and mean activations | block 2's OV circuits copy tokens | also present in a model that does not retrieve |
| head ablation | component intervention | later-copy prediction needs all of block 1, not only its previous-token heads, and block 2's induction heads | zeroing moves the stream off its training distribution |

## Summary

Each attention head acts on the residual stream through two matrices of rank at most $d_h$: a **QK circuit** $\mathbf{W}_q^\top \mathbf{W}_k$ governing where it attends and an **OV circuit** $\mathbf{W}_o \mathbf{W}_v$ governing what vector it writes to the destination position. A single attention layer can represent bigrams and skip-trigrams (including single-token copying), whereas completing novel in-context patterns $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}] \to [\mathrm{B}]$ requires at least two layers so that a **previous-token head** and a match-and-copy **induction head** can compose through the residual stream. Trained on repeated random patterns of variable period, a two-block model undergoes a sharp phase transition to more than $90\%$ in-context accuracy, whereas a fixed period allows a one-block model to exploit a brittle positional shortcut. Establishing what a head computes in practice therefore requires combining attention-pattern probes, OV weight inspection, and causal head ablations.

## Exercises

1. **Tied bigrams.** With zero blocks, the logit of token $b$ after token
   $a$ is $\mathbf{e}_b^\top \mathbf{e}_a$, so the geometry of the
   embeddings alone fixes which successor the model prefers. Call $b$ the
   *top successor* of $a$ when $\mathbf{e}_b^\top \mathbf{e}_a$ is the
   largest logit of that row, and assume that there are no ties.
    1. Show that if all embeddings have the same norm, the top successor of
       every token is the token itself.
    1. Drop the equal-norm assumption. Show that a token $b \neq a$ can be
       the top successor of $a$ only if $\|\mathbf{e}_b\| >
       \|\mathbf{e}_a\|$, and conclude that the top successors cannot form a
       cycle $a_1 \to a_2 \to \cdots \to a_k \to a_1$ with $k \geq 2$.
    1. Suppose that in a corpus the most frequent successor of the
       character `t` is `h`, that of `h` is `e`, that of `e` is the space,
       and that of the space is `t`. Show that no zero-block model
       reproduces all four as top successors, however long it is trained.
    1. In a one-block model, every head at the first position attends only
       to that position. Write the logits there as $\mathbf{E}\mathbf{M}
       \mathbf{e}_{x_0}$ for a matrix $\mathbf{M}$ built from the OV
       circuits, and explain why neither constraint of the previous parts
       applies to $\mathbf{E}\mathbf{M}\mathbf{E}^\top$.
   :exlabel:`ex_what-attention-computes-tied-bigrams`

1. **Paths through two blocks.** Fix the attention patterns of a model with
   $N$ blocks of $H$ heads on a sequence of $T$ tokens, and write
   $\mathbf{A}^{\ell m}$ for the $T \times T$ matrix with entries
   $\alpha_{ij}^{\ell m}$.
    1. Expand the logits of :eqref:`eq_stream-update` for $N = 2$ into a
       sum of path terms. Show that the path through head $m$ of block 1 and
       head $m'$ of block 2 acts like a single head with pattern
       $\mathbf{A}^{2m'}\mathbf{A}^{1m}$ and OV circuit
       $\mathbf{W}_{\mathrm{OV}}^{2m'}\mathbf{W}_{\mathrm{OV}}^{1m}$, and
       bound the rank of that circuit.
    1. Count the path terms for $N$ blocks, and evaluate the count for
       `model_two`.
    1. In a trained model the patterns of block 2 are not fixed: they depend
       on $\mathbf{h}^{(1)}$ through the QK circuit of block 2. Substitute
       $\mathbf{h}_i^{(1)}$ and $\mathbf{h}_j^{(1)}$ into the score of
       :eqref:`eq_qkov` and sort the terms by whether the query side, the
       key side, or both contain the output of a block-1 head. These terms
       are called *Q-composition* and *K-composition*, and the path of the
       first part is *V-composition* :cite:`Elhage.Nanda.Olsson.ea.2021`.
       Which kind of term does the induction circuit of
       :numref:`fig_induction-circuit` rely on?
    1. Show that two blocks can match on the two-token prefix
       $(x_{i-1}, x_i)$ when block 1 contains heads that attend to offsets
       $-1$ and $-2$, and name the composition term that carries each half
       of the match.
   :exlabel:`ex_what-attention-computes-paths`

1. [code] **The shortcut without biases.** The fixed-offset model of
   :numref:`subsec_fixed-offset-shortcut` has projection biases.
    1. Let a head's query and key before rotation be the same constant
       vectors $\mathbf{q}$ and $\mathbf{k}$ at every position. Show with
       $\mathbf{R}_i^\top \mathbf{R}_j = \mathbf{R}_{j-i}$ that the score
       depends only on the offset $j - i$, and explain how $\mathbf{q}$ and $\mathbf{k}$ can be chosen
       so that the score is largest at offset $-31$.
    1. Without biases, every query and key is a linear function of the
       stream. What must the token embeddings contain for a head to produce
       nearly constant queries and keys?
    1. Train the one-block model of the text with `bias=False` three times
       and report the second-copy accuracy and each head's attention at
       offset 31. How reliably does the shortcut form without biases, and
       does it survive the period-24 probe?
   :exlabel:`ex_what-attention-computes-shortcut-bias`

1. **Weights that change nothing.** The output of a head at query $i$ is
   $\mathbf{W}_{\mathrm{OV}} \sum_j \alpha_{ij}\, \mathbf{h}_j$ by
   :eqref:`eq_stream-update`, and the softmax makes every weight
   $\alpha_{ij}$ on an unmasked key positive. :citet:`Jain.Wallace.2019`
   found that in trained text classifiers quite different attention
   distributions often leave the prediction unchanged, and concluded that
   attention weights do not explain predictions.
    1. Show that the attention distributions over $n$ keys that give the
       same head output as $\boldsymbol{\alpha}_i$ contain, around
       $\boldsymbol{\alpha}_i$, a piece of an affine subspace of dimension
       at least $n - 1 - d_h$. Evaluate the bound for the last query of a
       63-token input to `model_two`.
    1. The alternatives of the first part are constructed for one input at
       a time, as are those of :citet:`Jain.Wallace.2019`.
       :citet:`Wiegreffe.Pinter.2019` argued that an alternative counts
       against an explanation only if a single trained model produces such
       alternatives on every input while making nearly the same
       predictions. Explain why the dimension count of the first part says
       nothing about whether such a model exists, and describe an
       experiment that would decide it for `model_two`.

1. [code] **Zero and mean ablation.** The ablations of
   :numref:`subsec_head-ablation` set a head's output to zero.
    1. Explain why zeroing a head's columns of `proj` removes exactly that
       head's contribution to the residual stream. When a block-1 head is
       removed, which attention patterns stay unchanged and which can
       change?
    1. A head's output is never zero in training, so zeroing moves the
       stream to states the model has not seen. *Mean ablation* replaces
       the output $\mathbf{o}_m$ of head $m$ by its average
       $\bar{\mathbf{o}}_m$ over positions and sequences instead. Implement
       it: build a `TinyCharLM` with `bias=True`, copy the weights of
       `model_two` into it with zero `qkv` biases, zero head $m$'s columns
       of `proj`, and set the bias of `proj` to the product of these
       columns, before zeroing, with $\bar{\mathbf{o}}_m$. Repeat the
       ablations of the text. Does removing the heads of block 1 that are
       not previous-token heads still break retrieval, and which of the
       two readings of :numref:`subsec_head-ablation` does the answer
       support?
    1. Remove the second-strongest induction head of block 2 on its own.
       Compare the sum of the two single-head increases in later-copy loss
       with the increase when both heads are removed together. What does
       the comparison say about how the heads share the work?
    1. Which limitations in :numref:`tab_induction-evidence` does mean
       ablation remove, and which remain?
   :exlabel:`ex_what-attention-computes-circuit-ablation`

1. [code] **K-composition in the weights.** ● The match of the induction
   head works because the QK circuit of block 2 reads what the OV circuit
   of block 1 wrote into the stream.
    1. For every pair of head $m$ of block 1 and head $m'$ of block 2 of
       `model_two`, compute the $64 \times 64$ matrix
       $$\mathbf{E}\, \mathbf{W}_{\mathrm{QK}}^{2m'}\, \mathbf{W}_{\mathrm{OV}}^{1m}\, \mathbf{E}^\top,$$
       whose entry $(a, b)$ scores a query token $a$ against a key whose
       predecessor was $b$. Measure the fraction of rows whose largest
       entry lies on the diagonal, as the copy test of
       :eqref:`eq_copy-matrix` does for copying.
    1. Compare with the direct path $\mathbf{E}\,
       \mathbf{W}_{\mathrm{QK}}^{2m'}\, \mathbf{E}^\top$ of the same block-2
       heads. Which pairs of heads carry the match, and do they agree with
       the attention-based scores of `head_scores`?
    1. The computation ignores the rotation that RoPE applies inside the QK
       circuit. Include the relative rotation $\mathbf{R}_{j-i}$ for a
       fixed offset, and check whether doing so at offset $-23$ changes
       the ranking of the pairs.
   :exlabel:`ex_what-attention-computes-k-composition`

1. [code] **A head that takes a gradient step.** ● Induction heads learn
   from the context by copying. A model can also infer a rule from labeled
   examples in its context. Let a sequence hold pairs $(\mathbf{x}_i,
   y_i)$, $i = 1, \ldots, n$, with standard normal $\mathbf{x}_i \in
   \mathbb{R}^p$ and $y_i = \mathbf{w}^\top \mathbf{x}_i$ for a vector
   $\mathbf{w}$ drawn afresh for every sequence, followed by a query
   $\mathbf{x}_q$ whose label is to be predicted.
    1. Show that one step of gradient descent on $\frac{1}{2}\sum_i (y_i -
       \mathbf{w}^\top \mathbf{x}_i)^2$, started at $\mathbf{w} =
       \mathbf{0}$ with learning rate $\eta$, predicts $\hat{y}_q = \eta
       \sum_i y_i\, \mathbf{x}_i^\top \mathbf{x}_q$. Read this as
       attention without a softmax: what plays the query, the keys, and the
       values? Compare it with the linear attention of
       :numref:`sec_windowed-linear-attention` with feature map
       $\phi(\mathbf{x}) = \mathbf{x}$.
    1. Give each example the token $\mathbf{z}_i = [\mathbf{x}_i; y_i] \in
       \mathbb{R}^{p+1}$ and the query the token $\mathbf{z}_q =
       [\mathbf{x}_q; 0]$. Choose the query, key, and value maps of one
       head without a softmax so that its output at the query token is
       $\hat{y}_q$. Why does it not matter that the query token also
       attends to itself? Implement the head and check it against the
       gradient step. Do the head's weights depend on $\mathbf{w}$?
    1. With $\eta = 1/n$, show that the step gives $\mathbf{w}_1 =
       \big(\frac{1}{n}\sum_i \mathbf{x}_i \mathbf{x}_i^\top\big)
       \mathbf{w}$ and that $\mathbb{E}\,\|\mathbf{w}_1 - \mathbf{w}\|^2 =
       \frac{p+1}{n}\|\mathbf{w}\|^2$. Hint: for a standard normal
       $\mathbf{x}$, $\mathbb{E}[(\mathbf{x}^\top \mathbf{w})^2
       \|\mathbf{x}\|^2] = (p+2)\|\mathbf{w}\|^2$. Conclude that the
       expected squared error of $\hat{y}_q$, relative to the variance of
       $y_q$, is $(p+1)/n$.
    1. For $p = 8$, measure the relative error over 500 sequences at $n =
       10$, 100, and 1000 and compare it with the prediction $9/n$. At $n =
       10$, how does the head compare with always predicting zero?
   :exlabel:`ex_what-attention-computes-in-context-regression`

1. [extended] **Induction heads in text.** ●●
   :citet:`Olsson.Elhage.Nanda.ea.2022` tie the formation of induction
   heads in language models to an abrupt gain in in-context learning,
   measured as the drop in loss from early to late positions of the
   context. Test this on the byte-level model
   of :numref:`subsec_pos-experiment`.
    1. Train `d2l.TinyCharLM` with `pos='rope'` on the WikiText-103 bytes
       of :numref:`subsec_pos-experiment` in 30 rounds of 100 steps of
       `d2l.train_steps`. After each round, record two quantities. The
       first is the largest induction-target score of a block-2 head on
       repeated random sequences over the text vocabulary with pattern
       lengths 16 to 32, computed with `head_scores`. The second is the
       in-context gain, the validation loss at target positions 0 to 15
       minus that at positions 96 to 127.
    1. Plot both quantities against the training step. Do the induction
       score and the in-context gain rise in the same window?
    1. After training, remove the block-2 heads with the highest induction
       scores with `ablate`, and measure the in-context gain and the
       validation loss again. How much of the gain do these heads carry?
    1. The repeated random sequences contain no structure besides
       repetition, while text repeats words and phrases at irregular
       distances. Which of your results would change if the text contained
       no repeated words, and why?
   :exlabel:`ex_what-attention-computes-text`

[Discussions](https://d2l.discourse.group/t/36773)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.7]{.kicker}

Attention Circuits: The Residual Stream, QK/OV Decomposition, and Induction Heads<br>**residual stream · QK & OV circuits · skip-trigrams · two-layer induction heads · causal head ablation · in-context learning**
:::
:::

::: {.slide title="Why Attention Heatmaps Alone Cannot Explain a Head"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **A Positional Shortcut Can Mimic Retrieval:** A 1-block model trained on 64-token sequences repeating a random 32-token pattern reaches **$100\%$ accuracy** on the second copy by looking $31$ positions back.
- **Testing on Period 24 Exposes the Illusion:** Evaluated on 24-token patterns, its accuracy collapses to **$1.6\%$ (chance)** because its heads still stare blindly at offset $-31$.
:::

::: {.col .fig .big}
@fig:ch10-attn-circ-shortcut-illusion
:::
:::
:::

::: {.slide title="Decomposing Multi-Head Output into a Sum Over Heads"}
[Additive Head Decomposition]{.kicker}

::: {.cols .vc}
::: {.col}
- Partitioning $\mathbf{W}_o \in \mathbb{R}^{d \times H d_h}$ into $H$ column slices $\mathbf{W}_o^m \in \mathbb{R}^{d \times d_h}$ rewrites multi-head attention as a sum of $H$ independent updates:
  $$\sum_{m=1}^H \mathbf{W}_o^m \mathbf{u}_m = \sum_{m=1}^H \sum_{j \le i} \alpha_{ij}^m\,\big(\mathbf{W}_o^m \mathbf{W}_v^m\big)\,\mathbf{h}_j$$
- **Only the Product $\mathbf{W}_o^m \mathbf{W}_v^m$ Matters:** Value projection $\mathbf{W}_v^m$ and output slice $\mathbf{W}_o^m$ act exclusively through their rank-$d_h$ product.
:::

::: {.col .fig}
@fig:ch10-attn-circ-additive-heads
:::
:::
:::

::: {.slide title="Every Head Is Two Rank-d_h Matrices: QK and OV"}
[QK and OV Circuits]{.kicker}

::: {.cols .vc}
::: {.col}
- Each attention head of width $d_h$ is completely specified by two $d \times d$ matrices of rank $\le d_h$ (Elhage et al., 2021):
  $$\mathbf{W}_{\mathrm{QK}} = \mathbf{W}_q^\top \mathbf{W}_k, \qquad \mathbf{W}_{\mathrm{OV}} = \mathbf{W}_o \mathbf{W}_v$$
- **QK Circuit ($\mathbf{W}_{\mathrm{QK}}$):** Bilinear form governing *where* the head attends ($\alpha_{ij}$).
- **OV Circuit ($\mathbf{W}_{\mathrm{OV}}$):** Linear operator governing *what* vector $\mathbf{W}_{\mathrm{OV}}\mathbf{h}_j$ is added to destination $i$.
:::

::: {.col .fig}
@fig:ch10-attn-circ-qkov-factorization
:::
:::
:::

::: {.slide title="The Residual Stream as a Shared Communication Bus"}
[The Residual Stream]{.kicker}

::: {.cols .vc}
::: {.col}
- Starting from the token embedding $\mathbf{h}_i^{(0)} = \mathbf{e}_{x_i}$, each block $\ell = 1, \ldots, N$ adds its heads' outputs directly into a running vector at position $i$:
  $$\mathbf{h}_i^{(\ell)} = \mathbf{h}_i^{(\ell-1)} + \sum_{m=1}^H \sum_{j \le i} \alpha_{ij}^{\ell m}\,\mathbf{W}_{\mathrm{OV}}^{\ell m}\,\mathbf{h}_j^{(\ell-1)}$$
- **Shared Read/Write Channel:** Every head reads subspaces of $\mathbf{h}^{(\ell-1)}$ via its QK and OV circuits and adds its result back into $\mathbf{h}^{(\ell)}$, terminating at tied output logits $\mathbf{z}_i = \mathbf{E}\,\mathbf{h}_i^{(N)}$.
:::

::: {.col .fig}
![Each head reads earlier residual streams via its QK circuit and writes additively via its OV circuit.](../img/mdl-attention-residual-stream.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="What 0 and 1 Attention Layers Can Express"}
[Expressivity by Depth]{.kicker}

::: {.cols .vc}
::: {.col}
- Expanding the residual stream for $N = 1$ reveals 0-layer bigram and 1-layer skip-trigram terms:
  $$\mathbf{z}_i = \mathbf{E}\,\mathbf{e}_{x_i} + \sum_{m=1}^H \sum_{j \le i} \alpha_{ij}^m\,\mathbf{E}\,\mathbf{W}_{\mathrm{OV}}^m\,\mathbf{e}_{x_j}$$
- **Zero Blocks ($N = 0$):** Symmetric table $\mathbf{E}\mathbf{E}^\top$ (cannot express asymmetric cycles).
- **One Block ($N = 1$):** Skip-trigrams $x_j \ldots x_i \to x_{\textrm{next}}$, but **cannot** complete novel $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}] \to [\mathrm{B}]$ because $[\mathrm{B}]$'s key knows nothing about its predecessor $[\mathrm{A}]$.
:::

::: {.col .fig}
@fig:ch10-attn-circ-depth-hierarchy
:::
:::
:::

::: {.slide title="Two Layers Form an Induction Head via K-Composition"}
[The Two-Layer Induction Circuit]{.kicker}

::: {.cols .vc}
::: {.col}
- Completing $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}]$ $\to$ $[\mathrm{B}]$ requires **two layers composing through the key pathway (K-composition)**:
- **Layer 1 (Previous-Token Head):** Uses RoPE offset preference to attend $1$ step back ($j - 1$) and writes $\mathbf{W}_{\mathrm{OV}}^{1}\,\mathbf{e}_{[\mathrm{A}]}$ into the residual stream $\mathbf{h}_j^{(1)}$ at $[\mathrm{B}]$'s position $j$.
- **Layer 2 (Induction Head):** Its QK circuit matches the current query $\mathbf{e}_{[\mathrm{A}]}$ against that written key feature in $\mathbf{h}_j^{(1)}$ ($\mathbf{e}_{[\mathrm{A}]}^\top \mathbf{W}_{\mathrm{QK}}^{2}\mathbf{W}_{\mathrm{OV}}^{1}\mathbf{e}_{[\mathrm{A}]}$), and its OV circuit copies $x_j = [\mathrm{B}]$ into the output logits.
:::

::: {.col .fig}
![A two-layer induction circuit composes a layer-1 previous-token head with a layer-2 match-and-copy head.](../img/mdl-attention-induction-circuit.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Synthetic Benchmark: Repeated Random Token Patterns"}
[Isolating In-Context Retrieval]{.kicker}

Does gradient descent find this circuit? Each sequence tiles a random pattern of length $L \in [L_{\min}, L_{\max}]$ over $64$ tokens:

@-what-attention-computes-repetition-as-a-task-1

- **Unpredictable First Copy ($t \le L$):** Each pattern is fresh, so the first period stays at chance (cross-entropy $\log 64 \approx 4.16\text{ nats}$, accuracy $1/64 \approx 1.6\%$). So does the restart token at $t = L$: nothing has repeated yet.
- **Predictable Later Copies ($t > L$):** Once the pattern repeats, each token can be predicted from the context by looking up the earlier occurrence.
:::

::: {.slide title="Pitfall: The Fixed-Offset Positional Shortcut"}
[Why Fixed Periods Mislead]{.kicker}

What happens if we train a **one-block** model (`num_blks=1, bias=True`) on sequences with a *fixed* pattern period $L = 32$?

@!what-attention-computes-the-positional-shortcut-1

@!what-attention-computes-the-positional-shortcut-2

::: {.d2l-note .warn}
**Mechanistic Pitfall:** On $L = 32$, the 1-block model reaches $1.00$ accuracy (about $0.1\text{ nats}$) by using constant bias queries and keys under RoPE to attend purely to fixed offset $j - i = -31$. Tested on period $L = 24$ (target at offset $-23$), its heads still place more than $0.9$ of their mass at offset $-31$ and accuracy collapses to $0.02$, which is chance.
:::
:::

::: {.slide title="Variable Periods Force Content-Based Induction"}
[1-Block vs. 2-Block Experiment]{.kicker}

Sampling the period uniformly from $L \in \{16, \ldots, 32\}$ on every sequence destroys any fixed-offset shortcut. We train three 2-block runs and a 1-block control (`bias=False`) for $5{,}000$ Adam steps:

@!what-attention-computes-two-blocks-learn-to-look-things-up-1

- **Two Blocks Solve Retrieval:** All three 2-block seeds drop below $1\text{ nat}$ between steps $1{,}000$ and $3{,}000$, reaching roughly $0.2$–$0.4\text{ nats}$ and more than $90\%$ later-copy accuracy.
- **One Block Fails:** The 1-block control stalls above $3.5\text{ nats}$ (about $10\%$ accuracy)—all it can do is boost the unigram set of tokens that appeared anywhere in the prefix.
:::

::: {.slide title="Phase Transition During Induction Head Formation"}
[Training Dynamics]{.kicker}

::: {.cols .vc}
::: {.col}
- **Abrupt Loss Drop:** For well over a thousand steps, the 2-block model sits on a plateau near $4\text{ nats}$, barely below chance; once block 1's OV output aligns with block 2's QK circuit, later-copy loss plunges by more than $3\text{ nats}$.
- **First-Copy Side Effect:** First-copy loss *rises* slightly above $\log 64 \approx 4.16$ right at the phase transition: once induction activates, a token that repeats by chance inside the first period causes the model to predict its previous successor.
:::

::: {.col .fig}
@!what-attention-computes-two-blocks-learn-to-look-things-up-2
:::
:::
:::

::: {.slide title="Evidence Level 1: Inspecting Attention Heatmaps"}
[Activation Patterns]{.kicker}

::: {.cols .vc}
::: {.col}
- Evaluating `model_two` on a test sequence of period $L = 24$ reveals two distinct geometric signatures across the two blocks:
- **Block 1 (Top Row):** Several heads form a subdiagonal stripe at $j = i - 1$ (**previous-token heads**).
- **Block 2 (Bottom Row):** Once the second copy starts at $i \ge 24$, all four heads form a parallel diagonal stripe at $j = i + 1 - L = i - 23$ (**induction heads**).
:::

::: {.col .fig}
@!what-attention-computes-the-heads-caught-in-the-act-1
:::
:::
:::

::: {.slide title="Quantifying Previous-Token and Induction Head Scores" layout="tall-output"}
[Aggregated Attention Metrics]{.kicker}

`head_scores` averages each head's attention weight on the previous token ($j = i - 1$) and on the variable induction target ($j = i + 1 - L$ for $i \ge L$) over $256$ sequences with $L \in \{16, \ldots, 32\}$:

@!what-attention-computes-the-heads-caught-in-the-act-2

- **Clean Separation of Labor:** In every run, Block 1 has $1$–$3$ previous-token heads (scores from about $0.3$ to $0.99$) and at most $0.02$ induction score, whereas Block 2 heads concentrate up to $\approx 0.6$ of their mass directly on the moving target $i + 1 - L$ (vs. $\approx 0.025$ uniform chance).
:::

::: {.slide title="Evidence Level 2: Reading the OV Copy Matrix"}
[Weight-Space Circuit Analysis]{.kicker}

::: {.cols .vc}
::: {.col}
- Attending to position $j$ predicts token $x_j$ only if Block 2's OV circuit boosts the logit of $x_j$:
  $$C_{ab} \stackrel{\textrm{def}}{=} \mathbf{e}_b^\top \mathbf{W}_{\mathrm{OV}}\,\bar{\mathbf{h}}_a$$
- **Signature of Copying:** If the OV circuit copies token identity, the largest entry in row $a$ lies on the diagonal ($b = a$) across nearly $100\%$ of the $64$ tokens (versus $1/64 \approx 0.02$ at random initialization).
:::

::: {.col .fig}
@fig:ch10-attn-circ-copy-matrix-probe
:::
:::
:::

::: {.slide title="OV Copy Matrix Results and Two Crucial Subtleties" layout="tight"}
[Weight Inspection Results]{.kicker}

::: {.cols .vc}
::: {.col}
- **Subtlety 1 (Stream vs. Embeddings):** Block 2's OV matrix copies almost perfectly ($0.9$–$1.0$ diagonal share) on the Block-1 stream $\bar{\mathbf{h}}_a$, but less reliably (anywhere from $0.3$ to nearly $1.0$) on raw embeddings $\mathbf{e}_a$—Block 2 also reads features transformed by Block 1's OV circuits (**V-composition**).
- **Subtlety 2 (Copying $\neq$ Retrieval):** The 1-block model *also* has a near-perfect diagonal OV copy matrix (share $\approx 1.00$), yet gets only $10\%$ accuracy because its QK circuit cannot locate the target.

@!what-attention-computes-the-circuit-is-in-the-weights[out=1]
:::

::: {.col .fig .narrow}
@!what-attention-computes-the-circuit-is-in-the-weights[out=2]
:::
:::
:::

::: {.slide title="Evidence Level 3: Causal Head Ablations"}
[Component Interventions]{.kicker}

Correlation in attention maps or weights is not causation. To test which heads are causally necessary, `ablate` zeroes out the output projection slice $\mathbf{W}_o^m$ of selected heads:

@what-attention-computes-head-ablation-1

- **Exact Additive Removal:** Because block $\ell$ adds $\sum_{m=1}^H \mathbf{W}_o^m \mathbf{u}_m$ to the residual stream, zeroing $\mathbf{W}_o^m$ deletes head $m$'s contribution while leaving all other heads intact.
- **Experimental Design:** In Block 1, we compare removing the previous-token heads (score $> 0.2$) against removing all *other* Block-1 heads; in Block 2, we remove the top $k \in \{1, 2, 3, 4\}$ induction heads cumulatively.
:::

::: {.slide title="Ablation Results: Necessity, Redundancy, and Surprises" layout="tall-output"}
[Interpreting Interventions]{.kicker}

@!what-attention-computes-head-ablation-2

- **1. Block-1 Previous-Token Heads Are Necessary:** Removing them drops accuracy from above $0.9$ to below $0.2$.
- **2. Surprise—Block 1's "Inert" Heads Are Also Necessary:** Removing them drops accuracy below $0.15$: Block 2's OV copy circuit relies on their stream transformation (zero-ablation also shifts stream norms off-distribution).
- **3. Block-2 Redundancy:** Removing the single strongest induction head leaves accuracy at $0.7$–$0.9$; all four Block-2 heads share the induction workload and must be removed together to drop to chance.
:::

::: {.slide title="In-Context Learning: Generalizing to Unseen Patterns"}
[One-Shot Pattern Completion]{.kicker}

::: {.cols .vc}
::: {.col}
- With $64^{24} \approx 2.2 \times 10^{43}$ possible patterns of length $24$, the model has never seen any test pattern during training, and all token transitions are equally likely a priori.
- **First Copy ($t < 24$):** Accuracy sits at the chance baseline $1/64 \approx 0.016$.
- **Second Copy ($t \ge 25$):** Jumps to about $0.8$ at once and above $0.90$ within a few positions, from a *single* in-context exposure. (At the exact restart $t = 24$, content matching has not yet seen a repeated token.)
:::

::: {.col .fig}
@!what-attention-computes-completing-patterns-it-has-never-seen
:::
:::
:::

::: {.slide title="Dissecting the Remaining Error: Repeated Tokens"}
[Beyond Single-Token Matching]{.kicker}

Why does accuracy after the restart plateau a few points short of $100\%$? When a token occurs $K > 1$ times inside the $L$-token pattern with different successors, matching on the current token alone is ambiguous:

$$\mathbb{E}\!\left[\frac{1}{K}\right] = \frac{1 - (1 - p)^L}{L p} \approx 0.84 \quad\textrm{overall}\quad (\approx 0.47 \textrm{ on repeated tokens } K > 1), \qquad p = \frac{1}{64},\; L = 24$$

@!what-attention-computes-generalization-to-unseen-patterns-1

::: {.d2l-note}
On unique pattern tokens ($K = 1$, $70\%$ of positions), `model_two` achieves **$0.97$–$1.00$ accuracy**; on repeated tokens ($K > 1$, $30\%$ of positions), it reaches roughly **$0.7$–$0.9$**, beating the $0.47$ uniform single-token match by combining multi-token prefix context and RoPE distance preferences.
:::
:::

::: {.slide title="In-Context Gradient Descent via Linear Attention"}
[Beyond Copying]{.kicker}

::: {.cols .vc}
::: {.col}
- A single linear-attention head can also execute **one step of gradient descent** on an in-context regression task $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ (Oswald et al., 2023):
  $$\mathbf{w}_1 = \eta \sum_{i=1}^n y_i\,\mathbf{x}_i \;\Longrightarrow\; \hat{y}_q = \sum_{i=1}^n (\mathbf{x}_q^\top \mathbf{x}_i)\,(\eta y_i)$$
- Packing examples as $\mathbf{z}_i = [\mathbf{x}_i; y_i]$ and query as $\mathbf{z}_q = [\mathbf{x}_q; 0]$ computes $\hat{y}_q$ in a single forward pass without updating model weights.
:::

::: {.col .fig}
@fig:ch10-attn-circ-icl-gd
:::
:::
:::

::: {.slide title="Hierarchy of Mechanistic Evidence"}
[Interpretability Methodology]{.kicker}

Our investigation of induction heads illustrates why establishing what a circuit computes requires a multi-level hierarchy of evidence:

| Diagnostic Probe | Evidence Level | What It Establishes | What It Leaves Open / Limitation |
| :--- | :--- | :--- | :--- |
| **1. Later-Copy Loss** | Behavioral | Two blocks solve variable-period repetition; one cannot | Identifies no internal component or algorithm |
| **2. Period Shift ($32$ $\to$ $24$)** | Input intervention | Exposes the 1-block model's fixed-offset $-31$ shortcut | Alters the input distribution alongside the probe |
| **3. Attention Head Scores** | Activation probe | Block 1 attends to $i-1$; Block 2 attends to $i+1-L$ | Shows *where* heads look, not *what* their values transmit |
| **4. OV Copy Matrix ($\mathbf{C}$)** | Weights + stream | Block 2's OV circuit boosts the attended token's logit | Also diagonal ($\approx 1.00$) in the failing 1-block control |
| **5. Zero / Mean Ablation** | Causal intervention | Both Block-1 groups and Block-2 heads are causally required | Zero-ablation shifts stream norms (use mean ablation to verify) |
:::

::: {.slide title="Concept Check: Circuits, Composition, and Sinks"}
[Check Your Understanding]{.kicker}

Test your understanding of QK/OV circuits and mechanistic interventions:

1. **Q-, K-, and V-Composition:** When a Block-2 head scores $\mathbf{h}_i^{(1)\top}\mathbf{W}_{\mathrm{QK}}^{2}\mathbf{h}_j^{(1)}$, why does the induction circuit rely on **K-composition** ($\mathbf{e}_{x_i}^\top \mathbf{W}_{\mathrm{QK}}^{2} \mathbf{W}_{\mathrm{OV}}^{1} \mathbf{e}_{x_{j-1}}$) rather than Q-composition?
2. **Attention Sinks:** In pretrained LLMs, many heads place $> 80\%$ of their attention weight on the very first token (`<bos>`) across almost all queries. What two quantities should you measure before claiming that `<bos>` carries critical semantic information?
3. **Zero-Block Bigram Cycles:** Why can a 0-block model with tied embedding/unembedding logits $\mathbf{z}_i = \mathbf{E}\,\mathbf{e}_{x_i}$ never make `t` $\to$ `h` $\to$ `e` $\to$ `␣` $\to$ `t` each token's top predicted successor?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Because the query at destination $i$ is the current token $x_i = [\mathrm{A}]$ itself ($\mathbf{e}_{x_i}$), whereas the *key* at source position $j$ ($x_j = [\mathrm{B}]$) must contain the Block-1 OV output encoding that its predecessor was $x_{j-1} = [\mathrm{A}]$.
2. (1) The norm of the OV output vector $\|\mathbf{W}_{\mathrm{OV}}\mathbf{h}_{\langle\mathrm{bos}\rangle}\|$ (which is near zero for an **attention sink** absorbing unused softmax mass), and (2) the change in loss under head ablation.
3. For $b \neq a$ to beat $a$'s self-logit $\mathbf{e}_a^\top\mathbf{e}_a$, Cauchy–Schwarz requires $\|\mathbf{e}_b\| > \|\mathbf{e}_a\|$, which cannot hold around a closed cycle $\|\mathbf{e}_{\texttt{t}}\| < \|\mathbf{e}_{\texttt{h}}\| < \|\mathbf{e}_{\texttt{e}}\| < \|\mathbf{e}_{\texttt{␣}}\| < \|\mathbf{e}_{\texttt{t}}\|$.
:::
:::

::: {.slide title="Summary: Attention Circuits and Induction Heads"}
[Key Takeaways]{.kicker}

| Concept / Circuit | Mathematical Expression | Rank / Depth Requirement | Mechanistic Role in Sequence Models |
| :--- | :--- | :---: | :--- |
| **QK Circuit** | $\mathbf{W}_{\mathrm{QK}} = \mathbf{W}_q^\top \mathbf{W}_k$ (or $\mathbf{W}_q^\top \mathbf{R}_{j-i}\mathbf{W}_k$) | Rank $\le d_h$ ($d \times d$) | Determines *where* a head attends ($\alpha_{ij}$) |
| **OV Circuit** | $\mathbf{W}_{\mathrm{OV}} = \mathbf{W}_o \mathbf{W}_v$ | Rank $\le d_h$ ($d \times d$) | Determines *what* vector $\mathbf{W}_{\mathrm{OV}}\mathbf{h}_j$ is written to destination $i$ |
| **Residual Stream** | $\mathbf{h}_i^{(\ell)} = \mathbf{h}_i^{(\ell-1)} + \sum_{m, j \le i} \alpha_{ij}^{\ell m}\mathbf{W}_{\mathrm{OV}}^{\ell m}\mathbf{h}_j^{(\ell-1)}$ | Shared width-$d$ vector bus | Allows later layers to read and compose earlier heads' outputs |
| **Skip-Trigram** | $\mathbf{E}\,\mathbf{W}_{\mathrm{OV}}^m\,\mathbf{e}_{x_j}$ weighted by $\alpha_{ij}^m$ | $N = 1$ attention layer | Implements $x_j \ldots x_i \to x_{\textrm{next}}$ and single-token copying |
| **Induction Circuit** | K-composition $\mathbf{e}_{x_i}^\top \mathbf{W}_{\mathrm{QK}}^{2m'} \mathbf{W}_{\mathrm{OV}}^{1m} \mathbf{e}_{x_{j-1}}$ | **$N \ge 2$ attention layers** | Completes novel in-context patterns $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}]$ $\to$ $[\mathrm{B}]$ |
| **Copy Matrix & Ablation** | $C_{ab} = \mathbf{e}_b^\top \mathbf{W}_{\mathrm{OV}}\bar{\mathbf{h}}_a$; zero/mean $\mathbf{W}_o^m$ | Weights + intervention | Separates true causal circuits from attention sinks and shortcuts |
:::
