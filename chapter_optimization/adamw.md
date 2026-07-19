```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('pytorch', 'jax')
```

# AdamW
:label:`sec_adamw`

Adam ended the last section as the default optimizer of deep learning, but
the recipe card that practitioners actually run says *AdamW*: Adam with
weight decay, typically $\lambda = 0.1$, on most but not all of the
parameters. This section explains the W. Back in :numref:`sec_weight_decay`
we saw that adding an $\ell_2$ penalty to the loss and shrinking the weights
by a fixed factor each step are the same operation under stochastic gradient
descent, and we promised to return to the optimizer once we had better ones.
That section already warned, and :numref:`sec_training_recipes` repeated in
recipe form, that under Adam the two operations part ways: the penalty
version is rescaled coordinate-by-coordinate and quietly stops doing its
job. The one-line repair by :citet:`Loshchilov.Hutter.2019` became part of
the name of the method. What the earlier chapters asserted, this section
derives and measures.

The plan: first the two-line algebra that breaks the equivalence, then AdamW
from scratch, then an experiment on the language model of
:numref:`subsec_tinylm` showing what decoupling buys. With the mechanics
settled we turn to practice: what weight decay actually does in large-scale
training (not what the word "regularization" suggests), which parameters
should be exempt from it, and what the optimizer's state costs in memory,
the one piece of systems arithmetic this chapter owns.

```{.python .input #adamw}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch.nn import functional as F
```

```{.python .input #adamw}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import optax
```

## The Penalty Meets the Preconditioner

Add the penalty $\frac{\lambda}{2}\|\mathbf{x}\|^2$ to the loss and the
gradient becomes $\mathbf{g}_t + \lambda \mathbf{x}_t$. Under SGD the two
implementations of weight decay coincide, as we saw in
:numref:`sec_weight_decay`:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{x}_t)
= (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t.
$$
:eqlabel:`eq_l2-is-decay`

Penalizing the loss *is* shrinking the weights, by the uniform factor
$1 - \eta\lambda$. Now feed the same penalized gradient through Adam's
update :eqref:`eq_adam-update`. Everything in the gradient is divided by
$\sqrt{\hat{\mathbf{v}}_t}$, the penalty included. Schematically, treating
$\sqrt{\hat{\mathbf{v}}_t}$ as a common preconditioner for loss term and
penalty alike (both moments are in fact built from the penalized gradient),
the shrinkage applied to coordinate $i$ is no longer uniform:

$$
\underbrace{\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; x_{t,i}}_{\textrm{$\ell_2$ penalty through Adam}}
\qquad \textrm{versus} \qquad
\underbrace{\eta\,\lambda\; x_{t,i}\vphantom{\frac{\eta}{\sqrt{\hat{v}_{t,i}}}}}_{\textrm{weight decay}}.
$$
:eqlabel:`eq_coupled-decay`

The regularization strength is now rescaled per coordinate by the same
preconditioner that rescales the loss gradient, and backwards: a parameter
with a large gradient history (large $\hat{v}_i$) is barely decayed at all,
while a parameter whose gradients have gone quiet is decayed hard. Whatever
$\lambda$ you chose, Adam re-prices it per coordinate and over time, and
nobody chose those prices. The appendix works this out exactly and isolates
it in a two-coordinate experiment where the same $\lambda$ produces
effective decay rates $100\times$ apart
(:numref:`subsec_mdl-decoupled-weight-decay`).

AdamW restores the SGD semantics by *decoupling*: the loss gradient goes
through the preconditioner and the decay does not,

$$
\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
$$
:eqlabel:`eq_adamw`

with $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ computed from the *plain*
gradient $\mathbf{g}_t$ as in :eqref:`eq_adam-moments`. This is verbatim the
shrink factor of :eqref:`eq_l2-is-decay`, applied uniformly. When the
learning rate follows a schedule $\eta_t$, the decay follows the same
schedule: :citet:`Loshchilov.Hutter.2019` scale the decay term by the
schedule multiplier so that regularization does not outlive learning, and
the implementations in PyTorch and Optax fold the entire learning rate into
the decay, which is the convention we use throughout.

## AdamW from Scratch

The implementation is Adam from :numref:`sec_adam` plus one term in the
update. We keep the same state layout, two buffers per parameter, and add
the decay coefficient to the hyperparameter dictionary.

```{.python .input #adamw-adamw-from-scratch-1}
%%tab pytorch
def init_adamw_states(feature_dim):
    m_w, m_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((m_w, v_w), (m_b, v_b))

def adamw(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (m, v) in zip(params, states):
        with torch.no_grad():
            m[:] = beta1 * m + (1 - beta1) * p.grad
            v[:] = beta2 * v + (1 - beta2) * torch.square(p.grad)
            m_hat = m / (1 - beta1 ** hyperparams['t'])
            v_hat = v / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * (m_hat / (torch.sqrt(v_hat) + eps)
                                         + hyperparams['wd'] * p)
        p.grad.zero_()
    hyperparams['t'] += 1
```

```{.python .input #adamw-adamw-from-scratch-1}
%%tab jax
def init_adamw_states(feature_dim):
    m_w, m_b = jnp.zeros((feature_dim, 1)), jnp.zeros(1)
    v_w, v_b = jnp.zeros((feature_dim, 1)), jnp.zeros(1)
    return [(m_w, v_w), (m_b, v_b)]

def adamw(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for i, (p, (m, v), g) in enumerate(zip(params, states, grads)):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * jnp.square(g)
        m_hat = m / (1 - beta1 ** hyperparams['t'])
        v_hat = v / (1 - beta2 ** hyperparams['t'])
        params[i] = p - hyperparams['lr'] * (
            m_hat / (jnp.sqrt(v_hat) + eps) + hyperparams['wd'] * p)
        states[i] = (m, v)
    hyperparams['t'] += 1
    return params[0], params[1]
```

On the airfoil harness of :numref:`sec_minibatch_sgd` it behaves like Adam
with a mild pull toward small weights; on a well-conditioned regression
problem there is little for the decay to do, and that is the point:
decoupled decay is a knob you can leave gently on without disturbing the
optimizer.

```{.python .input #adamw-adamw-from-scratch-2}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adamw, init_adamw_states(feature_dim),
               {'lr': 0.01, 'wd': 0.01, 't': 1}, data_iter, feature_dim);
```

The framework implementations apply exactly :eqref:`eq_adamw`.

```{.python .input #adamw-adamw-from-scratch-3}
%%tab pytorch
trainer = torch.optim.AdamW
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'weight_decay': 0.01},
                       data_iter)
```

```{.python .input #adamw-adamw-from-scratch-3}
%%tab jax
trainer = optax.adamw
d2l.train_concise_ch11(trainer,
                       {'learning_rate': 0.01, 'weight_decay': 0.01},
                       data_iter)
```

A note on defaults before we move on. AdamW inherits Adam's
$(\beta_1, \beta_2) = (0.9, 0.999)$, but large language models are
commonly trained with $\beta_2 = 0.95$, a convention set by GPT-3
:cite:`brown2020language` and kept by many open recipes since, including
OLMo 2 :cite:`OLMo.2025` (not all: PaLM did not
:cite:`chowdhery2022palm`). The shorter second-moment window
(about twenty steps instead of a thousand) makes the scale estimate track
heavy-tailed gradient noise faster, at some cost in smoothing; we return to
what this buys in :numref:`sec_practice`. One thing that has *not* survived
into practice is Adam's original claim that the method is robust to its
hyperparameters; both $\eta$ and $\lambda$ still need to be chosen, and the
next experiment is about whether they can at least be chosen independently.

## Decoupling, Demonstrated

We return to the testbed of :numref:`subsec_tinylm`: `TinyLM` on the
character-level *Time Machine*, with the same helpers for the final loss
and a smoothed curve.

```{.python .input #adamw-decoupling-demonstrated-1}
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)

def final_loss(losses, k=100):
    v = sum(losses[-k:]) / k
    return v if math.isfinite(v) else float('inf')

def smooth(losses, k=25):
    return [sum(losses[i:i + k]) / k
            for i in range(0, len(losses) - k + 1, k)]
```

Weight decay is about generalization, so we also need a scoreboard the
training loss cannot see: the mean cross-entropy on held-out text.

```{.python .input #adamw-decoupling-demonstrated-2}
%%tab pytorch
def val_loss(model, data):
    device = d2l.try_gpu()
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for X, Y in data.val_dataloader():
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), Y.reshape(-1),
                reduction='sum'))
            count += Y.numel()
    return total / count
```

```{.python .input #adamw-decoupling-demonstrated-2}
%%tab jax
@nnx.jit
def eval_step(model, X, Y):
    logits = model(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).sum()

def val_loss(model, data):
    total, count = 0.0, 0
    for X, Y in data.val_dataloader():
        total += float(eval_step(model, jnp.asarray(X), jnp.asarray(Y)))
        count += Y.reshape(-1).shape[0]
    return total / count
```

### One Number, Two Meanings

First, the size of the problem. We train `TinyLM` twice, each framework at
its own tuned learning rate from :numref:`sec_adam`, with the standard
$\lambda = 0.1$,
once as an $\ell_2$ penalty inside Adam's gradient, and once decoupled as
AdamW. In PyTorch the coupled version is what `Adam`'s `weight_decay`
argument does; in Optax we build it by adding the decay to the gradient
*before* it enters `adam`. The number $\lambda$ is nominally identical in
both runs.

```{.python .input #adamw-one-number-two-meanings}
%%tab pytorch
curves = {}
for name, opt_cls in [('Adam + $\\ell_2$', torch.optim.Adam),
                      ('AdamW', torch.optim.AdamW)]:
    torch.manual_seed(0)
    model = d2l.TinyLM(len(data.vocab))
    optimizer = opt_cls(model.parameters(), lr=0.003, weight_decay=0.1)
    curves[name] = d2l.train_lm(model, data, optimizer, num_steps=1000)
d2l.plot(list(range(0, 1000, 25)), [smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

```{.python .input #adamw-one-number-two-meanings}
%%tab jax
def coupled(lr, wd):
    return optax.chain(optax.add_decayed_weights(wd), optax.adam(lr))

curves = {}
for name, tx in [('Adam + $\\ell_2$', coupled(0.001, 0.1)),
                 ('AdamW', optax.adamw(0.001, weight_decay=0.1))]:
    model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    curves[name] = d2l.train_lm(model, data, optimizer, num_steps=1000)
d2l.plot(list(range(0, 1000, 25)), [smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

The decoupled run trains essentially as if the decay were not there. The
coupled run gets stuck more than a full nat higher, barely below its
starting loss. The mechanism is :eqref:`eq_coupled-decay` read as an
amplifier: once training makes progress the loss gradients shrink,
$\sqrt{\hat{\mathbf{v}}}$ shrinks with them, and the penalty, divided by
that small number, comes back amplified by orders of magnitude. Worse, once
$\lambda \mathbf{x}$ dominates the gradient, Adam normalizes it like any
other gradient: every coordinate then shrinks at a rate near $\eta$
regardless of $\lambda$, so past this point the dial no longer responds.
"$\lambda = 0.1$" is simply not one amount of regularization; it is a
different amount for every coordinate, every step, and every problem.

### A Grid, Twice

The claim worth testing is not that coupled decay is too strong; you could
always compensate with a smaller $\lambda$. It is that the compensation
*depends on the learning rate*, so the two knobs must be tuned jointly,
whereas decoupling lets them be tuned separately. So we run the same
experiment twice: a $3 \times 3$ grid of learning rate against weight
decay, once coupled, once decoupled, scored by held-out loss.

To give the decay something to do we make overfitting easy: a training
slice of 8,000 windows that the model revisits about six times in 800
steps. (On the full corpus the model barely completes one pass, and in our
runs two orders of magnitude of $\lambda$ move the held-out loss by a few
hundredths of a nat: at this scale, weight decay is a knob for data you
repeat. What decay does in genuinely one-pass training at much larger
scale is a different story, taken up below.)
Each arm gets the $\lambda$ range that suits it, which is already half the
story: the coupled arm needs values two to three orders of magnitude
smaller to land anywhere near its sweet spot.

```{.python .input #adamw-a-grid-twice-1}
small = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                        num_train=8000, num_val=2000)
```

```{.python .input #adamw-a-grid-twice-2}
%%tab pytorch
lrs = [1e-3, 3e-3, 1e-2]
wds_coupled, wds_decoupled = [3e-4, 1e-3, 3e-3], [1, 3, 10]

def sweep(make_optimizer, wds, num_steps=800):
    tbl = []
    for lr in lrs:
        row = []
        for wd in wds:
            torch.manual_seed(0)
            model = d2l.TinyLM(len(small.vocab))
            d2l.train_lm(model, small, make_optimizer(model, lr, wd),
                         num_steps)
            row.append(val_loss(model, small))
        tbl.append(row)
        print(' '.join(f'{v:.2f}' for v in row))
    return tbl

coupled_tbl = sweep(lambda model, lr, wd: torch.optim.Adam(
    model.parameters(), lr, weight_decay=wd), wds_coupled)
```

```{.python .input #adamw-a-grid-twice-2}
%%tab jax
lrs = [3e-4, 1e-3, 3e-3]
wds_coupled, wds_decoupled = [3e-3, 1e-2, 3e-2], [1, 3, 10]

def sweep(make_tx, wds, num_steps=800):
    tbl = []
    for lr in lrs:
        row = []
        for wd in wds:
            model = d2l.TinyLM(len(small.vocab), rngs=nnx.Rngs(0))
            optimizer = nnx.Optimizer(model, make_tx(lr, wd),
                                      wrt=nnx.Param)
            d2l.train_lm(model, small, optimizer, num_steps)
            row.append(val_loss(model, small))
        tbl.append(row)
        print(' '.join(f'{v:.2f}' for v in row))
    return tbl

coupled_tbl = sweep(coupled, wds_coupled)
```

```{.python .input #adamw-a-grid-twice-3}
%%tab pytorch
decoupled_tbl = sweep(lambda model, lr, wd: torch.optim.AdamW(
    model.parameters(), lr, weight_decay=wd), wds_decoupled)
```

```{.python .input #adamw-a-grid-twice-3}
%%tab jax
decoupled_tbl = sweep(lambda lr, wd: optax.adamw(lr, weight_decay=wd),
                      wds_decoupled)
```

```{.python .input #adamw-a-grid-twice-4}
def show_grids(tables, wd_axes, titles):
    fig, axes = d2l.plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    lo = min(v for tbl in tables for row in tbl for v in row)
    hi = max(v for tbl in tables for row in tbl for v in row)
    for ax, tbl, wds, title in zip(axes, tables, wd_axes, titles):
        ax.imshow(tbl, cmap='viridis', vmin=lo, vmax=hi)
        for i, row in enumerate(tbl):
            best = min(range(len(row)), key=lambda j: row[j])
            for j, v in enumerate(row):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='black' if (v - lo) / (hi - lo) > 0.5
                        else 'white',
                        fontweight='bold' if j == best else 'normal')
        i, j = min(((i, j) for i, row in enumerate(tbl)
                    for j in range(len(row))),
                   key=lambda ij: tbl[ij[0]][ij[1]])
        ax.add_patch(d2l.plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='red', lw=2))
        ax.set_xticks(range(len(wds)), [f'{wd:g}' for wd in wds])
        ax.set_yticks(range(len(lrs)), [f'{lr:g}' for lr in lrs])
        ax.set_xlabel('weight decay $\\lambda$')
        ax.set_title(title)
    axes[0].set_ylabel('learning rate $\\eta$')

show_grids([coupled_tbl, decoupled_tbl],
           [wds_coupled, wds_decoupled],
           ['Adam + $\\ell_2$', 'AdamW'])
```

Read each heatmap row by row; the two panels share one color scale, the
best $\lambda$ in each row is set in bold, and each panel's best cell is
boxed in red. In the AdamW grid the bold entries line up in a single
column: the best weight decay is the same at every learning rate in the
grid, and in this single-seed run the same column wins in both frameworks,
whose different initializations shift everything else. In the
coupled grid the bold entries wander: change $\eta$ and the $\lambda$ you
tuned is no longer right, exactly the joint-tuning burden that
:citet:`Loshchilov.Hutter.2019` documented on image classifiers, where the
good region of the coupled $(\eta, \lambda)$ plane is a diagonal band and
the decoupled one is axis-aligned. Note what decoupling does *not* buy at
this scale: after tuning both arms fully, their best held-out losses are
close. The
practical difference is that one of the two searches was a line search and
the other genuinely needed the grid.

### What Weight Decay Is Actually Doing
:label:`subsec_wd-at-scale`

:numref:`sec_weight_decay` introduced weight decay as a regularizer, a
Gaussian prior shrinking the model toward simple functions, and our
overfitting experiment above used it that way. Modern large-scale practice
mostly does not. A language model trained for a single epoch on
non-repeated data has little classical overfitting to fight, yet
$\lambda = 0.1$ remains in every frontier recipe. What is it doing there?

The current understanding, assembled from careful ablations
:cite:`DAngelo.Andriushchenko.Varre.ea.2024,Kosson.Messmer.Jaggi.2024`, is
that weight decay at scale is a *training-dynamics* control, not an
explicit regularizer. The mechanism runs through the weight norms. Layers
followed by normalization (most of a transformer) are scale-invariant: only
the *direction* of the weight vector matters, and the effective step size
of an update is roughly the update divided by the weight norm. Gradient
noise pushes weight norms up; decay pulls them down; the two settle into an
equilibrium, a steady state that :citet:`Kosson.Messmer.Jaggi.2024`
describe as rotational, in which each weight vector turns by a roughly
constant angle per step. Through this equilibrium $\lambda$ sets the
*effective learning rate*, and sets it uniformly across layers. That is a
large part of why decayed LLM runs reach *lower training loss* rather
than trading training loss for validation loss, the signature
:citet:`DAngelo.Andriushchenko.Varre.ea.2024` document, along with a
second, mundane service: keeping parameters small enough that bfloat16
training does not wander into divergence.

Two consequences carry forward. First, since the decay term is multiplied
by the schedule, $\eta$ and $\lambda$ act on the equilibrium through
their product, and what the product controls is a timescale: an AdamW
iterate is approximately an average of its recent updates over a horizon
of $1/(\eta\lambda)$ steps. In epoch units the horizon is
$\tau = B/(\eta \lambda D)$ for batch size $B$ and dataset size $D$, and
recent scaling studies find that $\tau$ is the quantity to hold roughly
steady, so that $\lambda$ is co-varied with batch and dataset size rather
than re-tuned :cite:`Bergsma.Dey.Gosal.ea.2025b`. Second, none of this
mechanism applies to parameters that are not followed by normalization,
which is why the question of the next section, *which* parameters to
decay, has a sharper answer than "all of them".

## What Not to Decay

The parameter census of :numref:`subsec_tinylm` sorted `TinyLM` into three
populations: embeddings, two-dimensional matrices, and one-dimensional
vectors, the LayerNorm scales and biases. Standard practice decays only
the matrices.

The reasons differ by population. LayerNorm scales and biases are few,
set the model's normalization scales directly, and shrinking them toward
zero fights the very equilibrium that gives decay its meaning; they are
left alone, as biases were already in :numref:`sec_weight_decay`.
Embedding rows are sparse: a row receives a gradient only when its token
occurs, but decay is applied every step, so rare rows are all decay and no
signal. OLMo 2 traced a training instability to exactly this, decay
grinding the embedding norms down until the $1/\|\mathbf{x}\|$ factor in
LayerNorm's gradient blew the early layers up, and turned decay off for
embeddings to let their norms settle :cite:`OLMo.2025`.

The implementation pattern is two parameter groups, and the census already
computed the split. In PyTorch, `torch.optim.AdamW` takes a list of groups
with per-group settings; in Optax, `optax.adamw` takes a mask over the
parameter tree. The same split returns in :numref:`sec_muon`, where the
matrices get a different optimizer entirely rather than merely a different
decay.

```{.python .input #adamw-what-not-to-decay-1}
%%tab pytorch
torch.manual_seed(0)
model = d2l.TinyLM(len(data.vocab))
decay = [p for name, p in model.named_parameters()
         if p.ndim == 2 and 'emb' not in name]
no_decay = [p for name, p in model.named_parameters()
            if p.ndim != 2 or 'emb' in name]
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0}], lr=0.003)
n_decay = sum(p.numel() for p in decay)
n = n_decay + sum(p.numel() for p in no_decay)
print(f'decayed: {n_decay} of {n} parameters')
```

```{.python .input #adamw-what-not-to-decay-1}
%%tab jax
model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(0))

def decay_mask(params):
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

optimizer = nnx.Optimizer(
    model, optax.adamw(0.001, weight_decay=0.1, mask=decay_mask),
    wrt=nnx.Param)
params = nnx.state(model, nnx.Param)
n_decay = sum(int(p.size) for p, m in zip(
    jax.tree.leaves(params), jax.tree.leaves(decay_mask(params))) if m)
n = sum(int(p.size) for p in jax.tree.leaves(params))
print(f'decayed: {n_decay} of {n} parameters')
```

About 96% of the parameters are decayed and everything fragile is exempt.
Trained on the full corpus, the two-group configuration matches the tuned
Adam run of :numref:`sec_adam`:

```{.python .input #adamw-what-not-to-decay-2}
losses = d2l.train_lm(model, data, optimizer, num_steps=2000)
d2l.plot(list(range(0, 2000, 25)), [smooth(losses)],
         'step', 'training loss')
print(f'perplexity per character: {math.exp(final_loss(losses)):.2f}')
```

This is the modern default configuration in full: AdamW, decay on the
matrices only, $\lambda$ chosen jointly with the schedule. What remains is
to ask what it costs.

## Optimizer State and Memory

Adam and AdamW carry two extra numbers per parameter, $m$ and $v$. At this
chapter's scale that is invisible; at language-model scale it decides what
fits on a device, so we do the arithmetic once with real numbers, using
the census total $n$ we just computed.

In full fp32 training, a parameter costs 4 bytes each for the weight, the
gradient, $m$, and $v$: 16 bytes per parameter, three quarters of it the
optimizer's. Mixed-precision training in bf16 does not make this smaller,
it makes it larger: the weights and gradients used in the forward and
backward pass drop to 2 bytes, but stable training keeps an fp32 *master
copy* of the weights alongside the optimizer state, and usually an fp32
accumulator for gradients as well (:numref:`sec_training_systems`).

```{.python .input #adamw-optimizer-state-and-memory}
setups = {'fp32 weights, grads, m, v': (4, 4, 8),
          'bf16 weights + grads, fp32 master, m, v': (2, 2, 12),
          'the same + fp32 gradient accumulator': (2, 6, 12)}
print(f'{"":<42}{"B/param":>8}{"TinyLM":>10}{"7B model":>10}')
for name, (w, g, s) in setups.items():
    per = w + g + s
    print(f'{name:<42}{per:>8}{n * per / 1e6:>8.1f}MB'
          f'{7e9 * per / 1e9:>8.0f}GB')
```

For `TinyLM` the whole bill is a few megabytes. Scale the identical
arithmetic to a 7-billion-parameter model and the common bf16 pattern
costs about 140 GB before a single activation is stored, more than any
single 80-GB accelerator holds, and 12 of the 20 bytes per parameter, the
fp32 master, $m$, and $v$, belong to the optimizer. That ratio is why optimizer
state is the first target when memory runs out. Adafactor factors $v$ for
each matrix into a row and a column sum, replacing $mn$ numbers by
$m + n$ :cite:`Shazeer.Stern.2018`; 8-bit optimizers store $m$ and $v$
block-quantized at one byte each :cite:`Dettmers.Lewis.Shleifer.ea.2022`.
The other lever is not shrinking the state but not replicating it:
ZeRO-style sharding spreads the 20 bytes across the data-parallel group
:cite:`Rajbhandari.Rasley.Ruwase.ea.2020`, part of the systems story of
:numref:`chap_performance` and :numref:`sec_training_systems`.

## Summary

Under SGD, an $\ell_2$ penalty in the loss and a per-step shrinkage of the
weights are the same thing; under Adam they are not, because the penalty's
gradient is divided by the same $\sqrt{\hat{\mathbf{v}}}$ as everything
else, re-scaling the regularization per coordinate, backwards, and beyond
your control. AdamW applies the decay outside the preconditioner,
restoring one $\lambda$ with one meaning. Decoupling does not so much
improve the best attainable loss as make it findable: in our grid the best
$\lambda$ stopped depending on the learning rate. At scale, weight decay
is less a regularizer than a training-dynamics control that sets the
effective learning rate through an equilibrium of noise against decay, on
a timescale set by the product $\eta\lambda$. Decay the matrices; exempt
embeddings, norms, and biases. And remember the bill: with the standard
mixed-precision pattern, an AdamW parameter costs about 20 bytes, 12 of
them optimizer state.

## Exercises

1. Reproduce the coupled-versus-decoupled disparity on a two-parameter
   toy: let both coordinates' loss gradients be pure noise,
   $g_i \sim \mathcal{N}(0, \sigma_i^2)$ with
   $\sigma = (10, 0.1)$, so that decay is the only systematic force.
   Track $|x_i|$ over a few thousand steps under Adam-with-$\ell_2$ and
   under AdamW at the same $(\eta, \lambda)$, and verify AdamW's trajectory
   against the prediction $(1 - \eta\lambda)^t$. Compare with the
   experiment in :numref:`subsec_mdl-decoupled-weight-decay`.
1. Under SGD *with momentum*, is the $\ell_2$ penalty still exactly
   equivalent to decoupled decay? Trace where $\lambda \mathbf{x}_t$ ends
   up inside the momentum buffer of :numref:`sec_momentum`, then check
   your conclusion experimentally on the airfoil harness.
1. The timescale rule says $\eta$ and $\lambda$ act through their product,
   with $\tau = B/(\eta\lambda D)$. Fix $\eta\lambda = 3 \times 10^{-3}$
   and rerun the decoupled sweep along the fixed-product ridge, e.g.
   $(\eta, \lambda) \in \{(10^{-3}, 3), (3 \times 10^{-3}, 1),
   (10^{-2}, 0.3)\}$. How constant is the held-out loss along the ridge,
   and what breaks the equivalence at the extremes?
1. Verify the exemptions empirically: rerun the two-group configuration
   with decay applied to *everything*, and track the norm of a rare
   token's embedding row over training. Relate what you see to OLMo 2's
   embedding instability and to the LayerNorm gradient's dependence on
   $1/\|\mathbf{x}\|$.
1. Extend the accounting cell to activations: for `TinyLM` with batch
   size $B$ and sequence length $T$, estimate the bf16 activation memory
   that must be stored for the backward pass (per block: the two
   normalization outputs, the attention inputs and outputs, and the MLP
   hidden layer). At what batch size do activations overtake the
   optimizer state?
1. PyTorch and Optax multiply the decay by the full learning rate, so the
   per-step shrinkage is $\eta\lambda$; :citet:`Loshchilov.Hutter.2019`
   scale it by the schedule only. Rerun the decoupled grid with the decay
   applied at a rate $\lambda$ independent of $\eta$ (in PyTorch, pass
   `weight_decay=wd / lr`). Does the best column stay put?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.7]{.kicker}

Decoupled weight decay and the modern default<br>
**why $\ell_2$ through Adam fails · AdamW · what decay really does · what it costs**
:::
:::

::: {.slide title="A promise from §3.7"}
[Motivation]{.kicker}

Under SGD, penalty and decay are **the same operation**:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{x}_t)
= (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t.$$

. . .

Under Adam, the penalty goes **through the preconditioner**
(schematically — the appendix has the exact recurrence):

$$\underbrace{\frac{\eta\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\, x_{t,i}}_{\textrm{$\ell_2$ through Adam}}
\qquad\textrm{vs.}\qquad
\underbrace{\eta\lambda\, x_{t,i}}_{\textrm{weight decay}}$$

- Large gradient history → barely decayed; quiet coordinate → crushed.
- One $\lambda$, re-priced per coordinate, per step — **nobody chose those
  prices** (100× disparity demo: appendix ch. 25).
:::

::: {.slide title="AdamW: decay outside the preconditioner"}
$$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

Loss gradient → preconditioner. Decay → applied directly, scaled by the
schedule (Loshchilov & Hutter, 2019).

. . .

One term added to Adam from §9.6:

@adamw-adamw-from-scratch-1
:::

::: {.slide title="One number, two meanings"}
Same model, same tuned $\eta$, same $\lambda = 0.1$ — coupled vs.
decoupled:

@adamw-one-number-two-meanings

- Decoupled: trains as if the decay were absent.
- Coupled: stuck a full nat higher. Progress shrinks
  $\sqrt{\hat{\mathbf{v}}}$ → the penalty comes back **amplified**; once it
  dominates, Adam normalizes it and the dial stops responding.
:::

::: {.slide title="A grid, twice"}
$3\times 3$ learning rate × weight decay, held-out loss, on a training
slice small enough to overfit. Best $\lambda$ per row in bold; each
panel's best cell boxed in red:

@!adamw-a-grid-twice-4

- **AdamW: the bold column is the same at every $\eta$** — one line search.
- Coupled: the optimum wanders, and lives 2–3 orders of magnitude lower.
- Fully tuned, both reach similar loss — decoupling buys *tunability*.
:::

::: {.slide title="What weight decay is actually doing at scale"}
One-epoch LLM training has little classical overfitting — yet
$\lambda = 0.1$ is universal. Why?

- Normalized layers are scale-invariant → gradient noise pushes norms up,
  decay pulls down → **equilibrium**: constant rotation per step
  (Kosson et al., 2024).
- Through it, $\lambda$ sets the **effective learning rate** — decayed runs
  reach *lower training loss*, not a train/val trade
  (D'Angelo et al., 2024). Plus: keeps bf16 out of divergence.
- $\eta\lambda$ = a timescale: $\tau = B/(\eta\lambda D)$ epochs; scale
  $\lambda$ with $B$, $D$ instead of re-tuning (Bergsma et al., 2025).
:::

::: {.slide title="What not to decay"}
The census populations of §9.6, treated differently:

- **Matrices** — decay (96% of parameters).
- **Norms and biases** — exempt: they set the normalization scales.
- **Embeddings** — exempt: sparse gradients, decay every step; OLMo 2
  traced spikes to decay grinding embedding norms down
  ($1/\|\mathbf{x}\|$ in LayerNorm's gradient).

. . .

@adamw-what-not-to-decay-1

::: {.d2l-note}
The same matrices / non-matrices split returns in §9.9 — Muon gives the
matrices a different *optimizer*, not just a different decay.
:::
:::

::: {.slide title="The memory bill"}
Two extra numbers per parameter — do the arithmetic once:

@!adamw-optimizer-state-and-memory

- bf16 does not shrink the bill; it adds an fp32 **master copy**.
- 7B model ≈ **140 GB**, 12 of 20 B/param is optimizer state.
- Shrink it: Adafactor (factored $v$), 8-bit states. Or **shard** it:
  ZeRO (→ ch. 11, §29.6).
:::

::: {.slide title="Recap"}
- $\ell_2$ through Adam ≠ weight decay: the preconditioner re-prices
  $\lambda$ per coordinate, backwards. AdamW decouples: one $\lambda$,
  one meaning.
- Decoupled knobs tune **separately**: best $\lambda$ independent of
  $\eta$ in our grid.
- At scale, decay = effective-LR control via the noise–decay equilibrium,
  on timescale $1/(\eta\lambda)$.
- Decay matrices only; exempt embeddings, norms, biases.
- AdamW parameter ≈ 20 bytes in mixed precision; optimizer state
  dominates.
:::
