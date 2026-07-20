```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('pytorch', 'jax')
```

# Adam
:label:`sec_adam`

Momentum (:numref:`sec_momentum`) improved the *direction* of the step by
averaging gradients over time. It did nothing about the step's size per
coordinate: every parameter still moves by the same multiple of its gradient,
so a single learning rate must serve the stiffest direction and the flattest
one at once. This section builds the other half of the modern default. The
construction takes three moves, each repairing a defect of the previous one:
AdaGrad scales every coordinate by the history of its own gradients, RMSProp
makes that history forget, and Adam :cite:`Kingma.Ba.2014` adds momentum back
in and corrects the startup bias. The result has been the default optimizer of
deep learning for a decade.

Being the default earns a harder question: where does Adam actually win, and
why? The second half of the section answers it experimentally. We introduce
the testbed that the rest of the chapter trains on, a tiny transformer
language model, and race tuned Adam against tuned SGD with momentum on it and
on a convolutional network. The two races end very differently, and the
difference is one of the more instructive facts in modern optimization.

```{.python .input #adam}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #adam}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import optax
```

## From AdaGrad to Adam

### Per-Coordinate Learning Rates

Consider learning problems whose features occur with very different
frequencies. This is the normal state of affairs in language: we will see the
word *learning* far more often than the word *preconditioning*. The same
imbalance appears in computational advertising and collaborative filtering,
where most items interest only a few users. Parameters tied to rare features
receive meaningful gradients only when those features occur. Under the
decaying learning rate that stochastic convergence demands
(:numref:`sec_sgd`), the parameters of frequent features approach their
optima quickly while the rare ones are still waiting for evidence; by the
time it arrives, the shared step size has decayed too far to use it. One
clock cannot serve both populations.

AdaGrad :cite:`Duchi.Hazan.Singer.2011` gives every coordinate its own clock.
Instead of decaying the step with wall-clock time $t$, it decays each
coordinate's step with that coordinate's own accumulated activity, measured
by the running sum of its squared gradients:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t,
$$
:eqlabel:`eq_adagrad`

with all operations elementwise, $\mathbf{s}_0 = \mathbf{0}$, and a small
$\epsilon > 0$ guarding the division. A coordinate whose gradients have been
large moves cautiously from now on; a coordinate that has barely been touched
keeps a large step and can respond the moment its rare feature appears.

There is a second, deeper reading. For the ill-conditioned valley of
:numref:`sec_momentum`, the ideal fix is to rescale each eigendirection by
its curvature, which is a *preconditioner*. Computing curvature is out of
the question at deep-learning scale, but when the coordinate axes roughly
align with the eigendirections, the gradient's own magnitude history is a
usable proxy for the diagonal of the Hessian, and
:eqref:`eq_adagrad` is exactly a diagonal preconditioner estimated from
first-order information alone. Both readings can be made precise; the metric
and regret derivations live in :numref:`subsec_mdl-per-coordinate`.

Let's watch the per-coordinate scaling work on the quadratic
$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ from :numref:`sec_momentum`, with the
same learning rate $\eta = 0.4$ that plain gradient descent could only use
timidly.

```{.python .input #adam-per-coordinate-learning-rates-1}
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

The trajectory is smooth: the steep $x_2$ direction is scaled down
automatically, so there is none of the oscillation that limited gradient
descent. But the accumulation in $\mathbf{s}_t$ also shows its cost within
twenty steps: the effective learning rate keeps shrinking, and the iterate
crawls long before it reaches the minimum. Because the scaling is adaptive,
we can compensate with a learning rate that would have been unthinkable
before, $\eta = 2$:

```{.python .input #adam-per-coordinate-learning-rates-2}
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

The from-scratch implementation for a real model keeps one accumulator per
parameter. We train it on the airfoil dataset with the harness from
:numref:`sec_minibatch_sgd`, using a larger learning rate than SGD could
tolerate there.

```{.python .input #adam-per-coordinate-learning-rates-3}
%%tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.zero_()
```

```{.python .input #adam-per-coordinate-learning-rates-3}
%%tab jax
def init_adagrad_states(feature_dim):
    s_w = jnp.zeros((feature_dim, 1))
    s_b = jnp.zeros(1)
    return [s_w, s_b]

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for i, (p, s, g) in enumerate(zip(params, states, grads)):
        s = s + jnp.square(g)
        params[i] = p - hyperparams['lr'] * g / jnp.sqrt(s + eps)
        states[i] = s
    return params[0], params[1]
```

```{.python .input #adam-per-coordinate-learning-rates-4}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

### RMSProp: Forgetting on Purpose

AdaGrad's accumulator never forgets. Since $\mathbf{s}_t$ grows without
bound, roughly linearly under persistent gradient noise, the effective step
decays like $\mathcal{O}(t^{-1/2})$ whether or not the optimization is
anywhere near done. That schedule is exactly right for the convex problems
AdaGrad was invented for, and exactly wrong for a nonconvex landscape where
the model may need to cross a plateau late in training and arrives with its
steps ground down to nothing.

RMSProp :cite:`Tieleman.Hinton.2012` keeps the per-coordinate scaling and
discards the lifetime memory, replacing the sum with an exponential moving
average, the same leaky averaging that momentum applies to the gradient
itself:

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t.
$$
:eqlabel:`eq_rmsprop`

The average $\mathbf{v}_t$ now estimates the *current* squared-gradient
scale over a window of about $1/(1-\beta_2)$ steps, ten at the customary
$\beta_2 = 0.9$, rather than the lifetime total. The step size no longer
decays by construction, and the learning rate $\eta$ becomes a knob we
control separately, to be scheduled on its own terms
(:numref:`sec_scheduler`). On the toy quadratic, RMSProp at $\eta = 0.4$
makes steady progress where AdaGrad stalled:

```{.python .input #adam-rmsprop-forgetting-on-purpose-1}
def rmsprop_2d(x1, x2, v1, v2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    v1 = beta2 * v1 + (1 - beta2) * g1 ** 2
    v2 = beta2 * v2 + (1 - beta2) * g2 ** 2
    x1 -= eta / math.sqrt(v1 + eps) * g1
    x2 -= eta / math.sqrt(v2 + eps) * g2
    return x1, x2, v1, v2

eta, beta2 = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

The implementation changes one line relative to AdaGrad, and the state
layout is identical, so we reuse `init_adagrad_states`.

```{.python .input #adam-rmsprop-forgetting-on-purpose-2}
%%tab pytorch
def rmsprop(params, states, hyperparams):
    beta2, eps = hyperparams['beta2'], 1e-6
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = beta2 * v + (1 - beta2) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(v + eps)
        p.grad.zero_()
```

```{.python .input #adam-rmsprop-forgetting-on-purpose-2}
%%tab jax
def rmsprop(params, grads, states, hyperparams):
    beta2, eps = hyperparams['beta2'], 1e-6
    for i, (p, v, g) in enumerate(zip(params, states, grads)):
        v = beta2 * v + (1 - beta2) * jnp.square(g)
        params[i] = p - hyperparams['lr'] * g / jnp.sqrt(v + eps)
        states[i] = v
    return params[0], params[1]
```

```{.python .input #adam-rmsprop-forgetting-on-purpose-3}
d2l.train_ch11(rmsprop, init_adagrad_states(feature_dim),
               {'lr': 0.01, 'beta2': 0.9}, data_iter, feature_dim);
```

### Adam: Both Moments, Debiased

Adam assembles the pieces. It applies the same exponential averaging to the
gradient itself, giving a momentum estimate $\mathbf{m}_t$ (the averaged form
of the heavy-ball buffer from :numref:`sec_momentum`), keeps RMSProp's
second-moment estimate $\mathbf{v}_t$, and corrects both for their zero
initialization:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\, \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t,
\qquad &
\hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t), \\
\mathbf{v}_t &= \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad &
\hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t),
\end{aligned}
$$
:eqlabel:`eq_adam-moments`

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
$$
:eqlabel:`eq_adam-update`

The defaults are $\beta_1 = 0.9$ and $\beta_2 = 0.999$: the direction
estimate averages over roughly ten steps, while the scale estimate averages
over roughly a thousand, a hundred times longer than RMSProp's window.

The corrections in :eqref:`eq_adam-moments` deserve a sentence of respect,
because they are exact rather than heuristic. Starting from
$\mathbf{v}_0 = \mathbf{0}$, the average $\mathbf{v}_t$ is a weighted sum
whose weights total $1 - \beta_2^t$, not $1$; if the squared-gradient scale
is locally stationary, then $\mathbb{E}[\mathbf{v}_t]$ carries exactly the
fraction $1 - \beta_2^t$ of the true scale, and dividing by that fraction
cancels the deficit identically at every $t$. The transient being cancelled
is not small: at $\beta_2 = 0.999$, after ten steps $\mathbf{v}_t$ holds
about $1\%$ of its stationary value, so uncorrected Adam would take its
most aggressive steps precisely when its scale estimate is built from the
fewest samples. The derivation, and a picture of the transient, are in
:numref:`subsec_mdl-per-coordinate`.

One more difference from :eqref:`eq_rmsprop`: Adam adds $\epsilon$
*outside* the square root. At this demo's scales, where
$\sqrt{\hat{\mathbf{v}}_t}$ is of order one, the placement matters little;
it decides the update where $\sqrt{\hat{\mathbf{v}}_t}$ is small — sparse
gradients, mixed precision — and the $\eta/\epsilon$ step ceiling we meet
at the end of the section depends on it. We use $\epsilon = 10^{-6}$ in
the code below, while framework implementations default to $10^{-8}$.

The implementation carries two buffers per parameter and a step counter for
the bias correction, which we keep in the `hyperparams` dictionary.

```{.python .input #adam-adam-both-moments-debiased-1}
%%tab pytorch
def init_adam_states(feature_dim):
    m_w, m_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((m_w, v_w), (m_b, v_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (m, v) in zip(params, states):
        with torch.no_grad():
            m[:] = beta1 * m + (1 - beta1) * p.grad
            v[:] = beta2 * v + (1 - beta2) * torch.square(p.grad)
            m_hat = m / (1 - beta1 ** hyperparams['t'])
            v_hat = v / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * m_hat / (torch.sqrt(v_hat) + eps)
        p.grad.zero_()
    hyperparams['t'] += 1
```

```{.python .input #adam-adam-both-moments-debiased-1}
%%tab jax
def init_adam_states(feature_dim):
    m_w, m_b = jnp.zeros((feature_dim, 1)), jnp.zeros(1)
    v_w, v_b = jnp.zeros((feature_dim, 1)), jnp.zeros(1)
    return [(m_w, v_w), (m_b, v_b)]

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for i, (p, (m, v), g) in enumerate(zip(params, states, grads)):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * jnp.square(g)
        m_hat = m / (1 - beta1 ** hyperparams['t'])
        v_hat = v / (1 - beta2 ** hyperparams['t'])
        params[i] = p - hyperparams['lr'] * m_hat / (jnp.sqrt(v_hat) + eps)
        states[i] = (m, v)
    hyperparams['t'] += 1
    return params[0], params[1]
```

```{.python .input #adam-adam-both-moments-debiased-2}
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

In practice one calls the framework implementation, which applies exactly
:eqref:`eq_adam-moments` and :eqref:`eq_adam-update`.

```{.python .input #adam-adam-both-moments-debiased-3}
%%tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input #adam-adam-both-moments-debiased-3}
%%tab jax
trainer = optax.adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

Note what the two extra buffers cost: Adam stores two floats of state for
every parameter of the model, tripling the memory a parameter occupies
before activations and gradients are counted. At the scale of this chapter's
models the cost is invisible; at the scale of language models it dictates
hardware budgets, and we return to the accounting in the next section.

## A Tiny Language Model
:label:`subsec_tinylm`

On the airfoil problem every optimizer in this chapter looks fine, and on
small image models the differences stay small. The phenomena that separate
modern optimizers, and the reason Adam rather than SGD is the default, show
up clearly on *language models*. So we now build the smallest language model
that exhibits them, and it will serve as the chapter's testbed from here on.

The reader has trained language models on exactly this data before: *The
Time Machine*, tokenized and batched as in :numref:`sec_text-sequence`, with
quality measured in perplexity as in :numref:`sec_language-model`. Only the
architecture is new. `TinyLM` is a *decoder-only transformer*, the
architecture of :numref:`chap_transformers`, and we are
deliberately using it two chapters early: nothing in this chapter requires
knowing how attention works. For our purposes it is a black box, a
differentiable function with a particular *census* of parameters, and the
census rather than the mechanism is what optimization sees.

```{.python .input #adam-a-tiny-language-model-1}
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)
```

The model: learned token and position embeddings, two pre-norm transformer
blocks (attention plus a two-layer MLP, each behind a LayerNorm, each with a
residual connection), and a linear head that predicts the next character.
The attention primitive is a single library call.

```{.python .input #adam-a-tiny-language-model-2}
%%tab pytorch
class TinyLM(nn.Module):  #@save
    """A small decoder-only transformer language model."""
    def __init__(self, vocab_size, d_model=128, num_heads=2, num_blks=2,
                 max_len=64):
        super().__init__()
        self.num_heads = num_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blks = nn.ModuleList([nn.ModuleDict(dict(
            norm1=nn.LayerNorm(d_model),
            qkv=nn.Linear(d_model, 3 * d_model),
            proj=nn.Linear(d_model, d_model),
            norm2=nn.LayerNorm(d_model),
            mlp=nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                              nn.Linear(4 * d_model, d_model))))
            for _ in range(num_blks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def attention(self, blk, X):
        B, T, D = X.shape
        q, k, v = blk['qkv'](X).chunk(3, dim=-1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
                   for u in (q, k, v))
        Y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return blk['proj'](Y.transpose(1, 2).reshape(B, T, D))

    def forward(self, X):
        H = self.token_emb(X) + self.pos_emb(torch.arange(X.shape[1],
                                                          device=X.device))
        for blk in self.blks:
            H = H + self.attention(blk, blk['norm1'](H))
            H = H + blk['mlp'](blk['norm2'](H))
        return self.head(self.norm(H))
```

```{.python .input #adam-a-tiny-language-model-2}
%%tab jax
class TinyLM(nnx.Module):  #@save
    """A small decoder-only transformer language model."""
    def __init__(self, vocab_size, d_model=128, num_heads=2, num_blks=2,
                 max_len=64, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_heads = num_heads
        self.token_emb = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, d_model, rngs=rngs)
        self.blks = nnx.List([nnx.Dict(
            norm1=nnx.LayerNorm(d_model, rngs=rngs),
            qkv=nnx.Linear(d_model, 3 * d_model, rngs=rngs),
            proj=nnx.Linear(d_model, d_model, rngs=rngs),
            norm2=nnx.LayerNorm(d_model, rngs=rngs),
            mlp1=nnx.Linear(d_model, 4 * d_model, rngs=rngs),
            mlp2=nnx.Linear(4 * d_model, d_model, rngs=rngs))
            for _ in range(num_blks)])
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

    def attention(self, blk, X):
        B, T, D = X.shape
        q, k, v = jnp.split(blk['qkv'](X), 3, axis=-1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1) for u in (q, k, v))
        Y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        return blk['proj'](Y.reshape(B, T, D))

    def __call__(self, X):
        H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = H + self.attention(blk, blk['norm1'](H))
            H = H + blk['mlp2'](jax.nn.gelu(blk['mlp1'](blk['norm2'](H))))
        return self.head(self.norm(H))
```

Before training it, look at what it is made of. The census below groups the
parameters into three populations: *embeddings*, whose rows receive
gradients only when their token occurs, the sparse features of AdaGrad's
origin story; two-dimensional *matrices*, which hold nearly all of the
parameters; and one-dimensional *vectors*, the LayerNorm scales and biases.
Later sections treat these populations differently, deciding which ones to
weight-decay and which ones a matrix preconditioner should own, so this
table is worth a moment of study.

```{.python .input #adam-a-tiny-language-model-3}
%%tab pytorch
model = TinyLM(vocab_size=len(data.vocab))
kinds = {'embeddings': 0, 'matrices': 0, 'vectors': 0}
print(f'{"parameter":<22} {"shape":<14} {"count":>7}')
for name, p in model.named_parameters():
    kind = ('embeddings' if 'emb' in name else
            'matrices' if p.ndim == 2 else 'vectors')
    kinds[kind] += p.numel()
    print(f'{name:<22} {str(tuple(p.shape)):<14} {p.numel():>7}')
for kind, count in kinds.items():
    print(f'{kind:>37} {count:>7}')
print(f'{"total":>37} {sum(kinds.values()):>7}')
```

```{.python .input #adam-a-tiny-language-model-3}
%%tab jax
model = TinyLM(vocab_size=len(data.vocab))
kinds = {'embeddings': 0, 'matrices': 0, 'vectors': 0}
print(f'{"parameter":<22} {"shape":<14} {"count":>7}')
for path, p in nnx.to_flat_state(nnx.state(model, nnx.Param)):
    name = '.'.join(str(k) for k in path)
    kind = ('embeddings' if 'emb' in name else
            'matrices' if p.ndim == 2 else 'vectors')
    kinds[kind] += p.size
    print(f'{name:<22} {str(tuple(p.shape)):<14} {p.size:>7}')
for kind, count in kinds.items():
    print(f'{kind:>37} {count:>7}')
print(f'{"total":>37} {sum(kinds.values()):>7}')
```

About 0.4 million parameters, 96% of them in the matrices. Training is a
plain step-counted loop: stream minibatches, compute
the mean cross-entropy of the next-character prediction, step the optimizer,
record the loss. We save it for reuse by the rest of the chapter.

```{.python .input #adam-a-tiny-language-model-4}
%%tab pytorch
def train_lm(model, data, optimizer, num_steps):  #@save
    """Train a model on next-token prediction for a fixed number of steps."""
    device = d2l.try_gpu()
    model.to(device)
    losses, step = [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   Y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if step >= num_steps:
                return losses
```

```{.python .input #adam-a-tiny-language-model-4}
%%tab jax
def train_lm(model, data, optimizer, num_steps):  #@save
    """Train a model on next-token prediction for a fixed number of steps."""
    @nnx.jit
    def step_fn(model, optimizer, X, Y):
        def loss_fn(model):
            logits = model(X)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss
    losses, step = [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            loss = step_fn(model, optimizer, jnp.asarray(X), jnp.asarray(Y))
            losses.append(float(loss))
            step += 1
            if step >= num_steps:
                return losses
```

Two small helpers finish the toolkit: the final loss averaged over the last
hundred steps (with divergence mapped to infinity, which the experiments
below will need), and a smoothed curve for plotting.

```{.python .input #adam-a-tiny-language-model-5}
def final_loss(losses, k=100):
    v = sum(losses[-k:]) / k
    return v if math.isfinite(v) else float('inf')

def smooth(losses, k=25):
    return [sum(losses[i:i + k]) / k
            for i in range(0, len(losses) - k + 1, k)]
```

A first run with Adam at $\eta = 0.003$, for 2,000 steps, about twenty
seconds on one GPU:

```{.python .input #adam-a-tiny-language-model-6}
%%tab pytorch
model = TinyLM(len(data.vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
losses = train_lm(model, data, optimizer, num_steps=2000)
d2l.plot(list(range(0, 2000, 25)), [smooth(losses)],
         'step', 'training loss')
print(f'perplexity per character: {math.exp(final_loss(losses)):.2f}')
```

```{.python .input #adam-a-tiny-language-model-6}
%%tab jax
model = TinyLM(len(data.vocab))
optimizer = nnx.Optimizer(model, optax.adam(0.003), wrt=nnx.Param)
losses = train_lm(model, data, optimizer, num_steps=2000)
d2l.plot(list(range(0, 2000, 25)), [smooth(losses)],
         'step', 'training loss')
print(f'perplexity per character: {math.exp(final_loss(losses)):.2f}')
```

The model reaches a per-character perplexity below 3, against a uniform
baseline of about 28 (the vocabulary size): it has learned a good deal of
English spelling in those twenty seconds. Good enough to optimize; now the
question is how much of that speed belongs to Adam.

## Where Adam Wins

Practitioners reach for Adam on transformers because SGD, however tuned, does
not keep up. On convolutional networks that folk wisdom is far weaker; SGD
with momentum was the standard for a decade of vision. Rather than repeat
the folklore, we run the comparison.

The protocol matters more than the contestants, because "optimizer A beats
optimizer B" claims are notoriously sensitive to tuning effort
:cite:`Schmidt.Schneider.Hennig.2021`. Ours is symmetric and small: both
optimizers train the same model from the same initialization for the same
2,000 steps at a constant learning rate, with no schedule, clipping, or
weight decay. Each gets a four-point learning-rate grid spaced by factors of
about three, and the run with the best final training loss represents its
family. Training loss is the right scoreboard here: the question is who
optimizes faster, not who generalizes better.

### The Race on the Language Model

```{.python .input #adam-the-race-on-the-language-model-1}
%%tab pytorch
def run_lm(make_optimizer, lrs, num_steps=2000):
    curves = {}
    for lr in lrs:
        torch.manual_seed(0)
        model = TinyLM(len(data.vocab))
        curves[lr] = train_lm(model, data, make_optimizer(model, lr),
                              num_steps)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}')
    return curves

sgd_lm = run_lm(lambda model, lr: torch.optim.SGD(model.parameters(), lr,
                                                  momentum=0.9),
                lrs=[0.03, 0.1, 0.3, 1.0])
```

```{.python .input #adam-the-race-on-the-language-model-1}
%%tab jax
def run_lm(make_tx, lrs, num_steps=2000):
    curves = {}
    for lr in lrs:
        model = TinyLM(len(data.vocab), rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        curves[lr] = train_lm(model, data, optimizer, num_steps)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}')
    return curves

sgd_lm = run_lm(lambda lr: optax.sgd(lr, momentum=0.9),
                lrs=[0.03, 0.1, 0.3, 1.0])
```

```{.python .input #adam-the-race-on-the-language-model-2}
%%tab pytorch
adam_lm = run_lm(lambda model, lr: torch.optim.Adam(model.parameters(), lr),
                 lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #adam-the-race-on-the-language-model-2}
%%tab jax
adam_lm = run_lm(optax.adam, lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

The sweeps already tell most of the story. SGD's best learning rate sits
directly below the divergent one: push it one grid point higher and the loss
is NaN. That knife-edge is characteristic of SGD on transformers, whose
heavy-tailed gradients make the largest stable step the best one and the
next step fatal. Adam's optimum is interior, with stable neighbors on both
sides. Now the head-to-head comparison of the two winners:

```{.python .input #adam-the-race-on-the-language-model-3}
best_sgd = min(sgd_lm, key=lambda lr: final_loss(sgd_lm[lr]))
best_adam = min(adam_lm, key=lambda lr: final_loss(adam_lm[lr]))
d2l.plot(list(range(0, 2000, 25)),
         [smooth(sgd_lm[best_sgd]), smooth(adam_lm[best_adam])],
         'step', 'training loss',
         legend=[f'SGD + momentum, lr {best_sgd:g}',
                 f'Adam, lr {best_adam:g}'])
print(f'final perplexity: SGD {math.exp(final_loss(sgd_lm[best_sgd])):.2f}, '
      f'Adam {math.exp(final_loss(adam_lm[best_adam])):.2f}')
```

Tuned Adam beats tuned SGD ("tuned" here and below meaning the best of the
coarse four-point grid) by a wide margin, roughly 20% in final training
loss in our runs, and the gap is not a transient: Adam's curve is
below SGD's at every point after the first few dozen steps and is still
opening at the end. No rate in our grid rescued SGD: the sweep found its
best, and the next one diverged. On this problem, per-coordinate
scaling is worth more than any amount of step-size tuning.

### The Same Race on a CNN

Now the identical protocol on an image classifier: a compact convolutional
network on Fashion-MNIST, defined in a few lines. Note that `train_lm` never
asked its model to be a language model; it streams minibatches and computes
cross-entropy, so the same harness trains the CNN unchanged.

```{.python .input #adam-the-same-race-on-a-cnn-1}
%%tab pytorch
fashion = d2l.FashionMNIST(batch_size=256)

def make_cnn():
    return nn.Sequential(
        nn.LazyConv2d(32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.LazyConv2d(64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.LazyLinear(128), nn.ReLU(), nn.LazyLinear(10))

def test_accuracy(model, data):
    device = d2l.try_gpu()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in data.val_dataloader():
            X, y = X.to(device), y.to(device)
            correct += float((model(X).argmax(axis=1) == y).sum())
            total += y.numel()
    return correct / total
```

```{.python .input #adam-the-same-race-on-a-cnn-1}
%%tab jax
fashion = d2l.FashionMNIST(batch_size=256)

class FashionCNN(nnx.Module):
    def __init__(self, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.fc1 = nnx.Linear(64 * 7 * 7, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, X):
        X = nnx.max_pool(nnx.relu(self.conv1(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = nnx.max_pool(nnx.relu(self.conv2(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = X.reshape(X.shape[0], -1)
        return self.fc2(nnx.relu(self.fc1(X)))

def test_accuracy(model, data):
    correct = total = 0
    for X, y in data.val_dataloader():
        pred = model(jnp.asarray(X)).argmax(axis=1)
        correct += float((pred == jnp.asarray(y)).sum())
        total += y.shape[0]
    return correct / total
```

```{.python .input #adam-the-same-race-on-a-cnn-2}
%%tab pytorch
def run_cnn(make_optimizer, lrs, num_steps=2000):
    curves, accs = {}, {}
    for lr in lrs:
        torch.manual_seed(0)
        model = make_cnn()
        curves[lr] = train_lm(model, fashion, make_optimizer(model, lr),
                              num_steps)
        accs[lr] = test_accuracy(model, fashion)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}, '
              f'test accuracy {accs[lr]:.3f}')
    return curves, accs

sgd_cnn, sgd_cnn_acc = run_cnn(
    lambda model, lr: torch.optim.SGD(model.parameters(), lr, momentum=0.9),
    lrs=[0.01, 0.03, 0.1, 0.3])
```

```{.python .input #adam-the-same-race-on-a-cnn-2}
%%tab jax
def run_cnn(make_tx, lrs, num_steps=2000):
    curves, accs = {}, {}
    for lr in lrs:
        model = FashionCNN(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        curves[lr] = train_lm(model, fashion, optimizer, num_steps)
        accs[lr] = test_accuracy(model, fashion)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}, '
              f'test accuracy {accs[lr]:.3f}')
    return curves, accs

sgd_cnn, sgd_cnn_acc = run_cnn(lambda lr: optax.sgd(lr, momentum=0.9),
                               lrs=[0.01, 0.03, 0.1, 0.3])
```

```{.python .input #adam-the-same-race-on-a-cnn-3}
%%tab pytorch
adam_cnn, adam_cnn_acc = run_cnn(
    lambda model, lr: torch.optim.Adam(model.parameters(), lr),
    lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #adam-the-same-race-on-a-cnn-3}
%%tab jax
adam_cnn, adam_cnn_acc = run_cnn(optax.adam, lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #adam-the-same-race-on-a-cnn-4}
best_sgd = min(sgd_cnn, key=lambda lr: final_loss(sgd_cnn[lr]))
best_adam = min(adam_cnn, key=lambda lr: final_loss(adam_cnn[lr]))
d2l.plot(list(range(0, 2000, 25)),
         [smooth(sgd_cnn[best_sgd]), smooth(adam_cnn[best_adam])],
         'step', 'training loss',
         legend=[f'SGD + momentum, lr {best_sgd:g}',
                 f'Adam, lr {best_adam:g}'])
print(f'test accuracy: SGD {sgd_cnn_acc[best_sgd]:.3f}, '
      f'Adam {adam_cnn_acc[best_adam]:.3f}')
```

A different picture. Both tuned optimizers drive the training loss into the
same low regime along closely tracking curves, and their test accuracies
land within a point or two of each other, around 90–92%; whatever residual
edge remains is small and varies from run to run. Nothing here resembles the
language-model gap: on this problem SGD is a competitive choice, which is
why it carried computer vision for a decade. The interesting question is why
the same two optimizers behave so differently on the two tasks.

### Why the Gap Lives Where It Does

This is current research rather than settled textbook material, but its
outline is clear, and all of its threads run through the parameter census.

First, the gap is not about minibatch noise. :citet:`Kunstner.Chen.Lavington.ea.2023`
showed that it persists in *full-batch* training, where there is no noise to
manage, and that Adam's advantage tracks that of *sign descent*, the method
that keeps only the sign of each gradient coordinate. The connection is
visible in :eqref:`eq_adam-update`: switch off both averages
($\beta_1 = \beta_2 = 0$) and the update becomes
$\eta\, \mathbf{g}_t / (|\mathbf{g}_t| + \epsilon)$, which is
$\eta\, \mathrm{sign}(\mathbf{g}_t)$. Adam is a smoothed sign method: what
it mainly does on these problems is equalize the step across coordinates
whose gradients differ enormously in magnitude.

Second, the disparities being equalized are real and structural. Language
data is heavy-tailed: most tokens are rare, so the embedding rows and output
columns that serve them see small, infrequent gradients, while frequent
tokens see large, constant ones. :citet:`Kunstner.Yadav.Milligan.ea.2024`
showed that gradient descent stalls on the loss of the rare classes while
Adam keeps making progress on all of them, and that the effect reproduces
even in linear models with imbalanced classes. Third, the heterogeneity is
architectural as well as lexical: across the blocks of a transformer,
embedding, attention, and MLP parameters have curvature spectra so different
that no single learning rate suits them all, whereas the blocks of a CNN
look far more alike :cite:`Zhang.Chen.Ding.ea.2024`. A per-coordinate method
supplies every block its own effective step; a global method must serve the
stiffest and starve the rest. Our census is the coarse version of this
story: three populations with different geometry and different update
statistics, all sharing one $\eta$ under SGD.

None of these accounts is final, and they are not mutually exclusive; the
character-level vocabulary here has no starved rare tokens — its frequencies
are skewed, but unlike the rare words of a word-level vocabulary, even the
rarest character recurs thousands of times over the run — yet the
architectural heterogeneity alone sustains a wide gap. What is settled is
the practice: on transformers, use the method that scales per coordinate.

## When the Variance Estimate Misbehaves

Adam's preconditioner $\hat{\mathbf{v}}_t$ is an estimate built from a
single noisy gradient stream, and estimates can be wrong in load-bearing
ways. When gradients are sparse or heavy-tailed, the exponential average
forgets between informative events: a coordinate's $\hat{\mathbf{v}}$ decays
toward zero during a quiet stretch, and the rare large gradient then arrives
with an enormous effective step. :citet:`Reddi.Kale.Kumar.2019` turned this
into a theorem, exhibiting convex problems on which Adam converges to the
*worst* point; their construction, and the one-line fix that keeps a running
maximum of $\hat{\mathbf{v}}_t$ (AMSGrad), are worked through in
:numref:`subsec_mdl-per-coordinate`. A related repair, Yogi
:cite:`Zaheer.Reddi.Sachan.ea.2018`, controls how fast $\mathbf{v}_t$ can
shrink by making the update additive with a sign-controlled direction. Both
are implemented in the exercises; neither displaced Adam in practice, but
the failure mode they address is worth recognizing when a training run
inexplicably blows up on rare inputs.

This is also what $\epsilon$ is really for. Nominally a guard against
division by zero, it is in fact a floor on the denominator of
:eqref:`eq_adam-update`, and therefore a ceiling of $\eta/\epsilon$ on any
coordinate's effective step. A small $\epsilon$ says "trust the variance
estimate, even a tiny one"; a large $\epsilon$ says "do not", and in the
limit of large $\epsilon$ Adam degrades gracefully into SGD with momentum at
learning rate $\eta/\epsilon$. Between the extremes lies a real tuning knob:
values as large as $10^{-4}$ or even $10^{-2}$ are sometimes used where
second-moment estimates are unreliable. The exercise below maps the
interpolation.

## Summary

Adam is three stacked estimates: AdaGrad's insight that each coordinate
should be scaled by its own gradient history, RMSProp's correction that the
history must forget (an exponential average with window $1/(1-\beta_2)$),
and a momentum average of the gradient itself, with both averages debiased
exactly against their zero initialization. The result is a diagonal
preconditioner estimated from first-order information, cheap enough to run
at any scale, at the price of two extra state buffers per parameter.

Where it wins is not uniform. At matched tuning on a small transformer
language model, Adam beats SGD with momentum decisively, and no rate in
our grid closed the gap; on a comparable CNN the two are close. The best
current explanations point at heterogeneity, across token frequencies and
across architectural blocks, that a single global learning rate cannot
serve. The
variance estimate at Adam's heart can also misbehave; $\epsilon$, AMSGrad,
and Yogi are the standard responses.

## Exercises

1. Adjust the learning rate in the from-scratch Adam run on the airfoil
   data, and observe and analyze the results.
1. Rewrite the moment updates of :eqref:`eq_adam-moments` so that no
   explicit bias correction is required. Hint: initialize with the first
   gradient rather than with zero. What is lost?
1. Rerun the tuned Adam run on `TinyLM` with
   $\beta_2 \in \{0.9, 0.99, 0.999, 0.9999\}$. Which direction hurts more,
   forgetting too fast or too slowly? Relate your finding to the effective
   averaging window $1/(1-\beta_2)$.
1. Sweep $\epsilon \in \{10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}, 1\}$ for Adam
   on `TinyLM` at fixed $\eta$. Explain the trend at both ends using the
   $\eta/\epsilon$ step ceiling from the last section.
1. Adam's second-moment update can be rewritten as
   $\mathbf{v}_t = \mathbf{v}_{t-1} + (1 - \beta_2)(\mathbf{g}_t^2 - \mathbf{v}_{t-1})$,
   which shrinks $\mathbf{v}_t$ whenever $\mathbf{g}_t^2 < \mathbf{v}_{t-1}$.
   Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` caps the shrinkage rate by
   replacing the increment with
   $(1 - \beta_2)\, \mathbf{g}_t^2 \odot \mathrm{sgn}(\mathbf{g}_t^2 - \mathbf{v}_{t-1})$.
   Implement Yogi in the from-scratch harness and compare it with Adam on
   the airfoil data. Can you construct a stream of gradients on which Adam
   diverges and Yogi converges?
1. Implement AMSGrad by carrying the running maximum
   $\hat{\mathbf{v}}_t^{\max} = \max(\hat{\mathbf{v}}_{t-1}^{\max}, \hat{\mathbf{v}}_t)$
   and using it in place of $\hat{\mathbf{v}}_t$ in :eqref:`eq_adam-update`.
   Verify on the airfoil data that it behaves like Adam, then test it on the
   non-convergence construction of :numref:`subsec_mdl-per-coordinate`.
1. Adadelta :cite:`Zeiler.2012` extends RMSProp with a second exponential
   average, of the squared *updates*, whose square root replaces $\eta$, so
   the method nominally has no learning rate. Implement it in the
   from-scratch harness. Where did the scale of the very first update come
   from?
1. Per-coordinate methods are axis-aligned. Rotate the toy problem by 45°,
   $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$, and rerun
   `adagrad_2d` and `rmsprop_2d`. How much of the advantage over gradient
   descent survives?
1. Following :citet:`Kunstner.Yadav.Milligan.ea.2024`, log the training loss
   of `TinyLM` separately for frequent and rare characters (split the
   vocabulary by corpus frequency) under tuned SGD and tuned Adam. Which
   optimizer makes progress on the rare half?

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1078)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1079)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.6]{.kicker}

Per-coordinate learning rates and the modern default<br>
**AdaGrad → RMSProp → Adam · a tiny transformer testbed · where Adam wins**
:::
:::

::: {.slide title="One learning rate is not enough"}
[Motivation]{.kicker}

Momentum fixed the *direction*; the step *size per coordinate* is still
one global $\eta$.

- Rare features (rare words, rare users) get meaningful gradients rarely;
  by the time they arrive, a decayed $\eta$ is too small to use them.
- The ill-conditioned valley: the ideal fix is a *preconditioner*, one
  step size per direction. Curvature is unaffordable; the gradient's own
  history is not.

. . .

**AdaGrad** (Duchi, Hazan & Singer, 2011): decay each coordinate by its
*own* activity,

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.$$
:::

::: {.slide title="AdaGrad on the valley"}
Same quadratic as the momentum section, $\eta = 0.4$ — smooth, no
oscillation, but the accumulated $\mathbf{s}_t$ grinds the steps down:

@adam-per-coordinate-learning-rates-1

. . .

The scaling is adaptive, so a formerly unthinkable $\eta = 2$ is safe:

@adam-per-coordinate-learning-rates-2
:::

::: {.slide title="RMSProp: forgetting on purpose"}
AdaGrad never forgets: $\mathbf{s}_t$ grows forever, steps decay like
$t^{-1/2}$ *by construction* — right for convex problems, wrong for deep
nets that need to move late in training.

. . .

**RMSProp** (Hinton, 2012): same rule, leaky average instead of sum,

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\, \mathbf{g}_t^2,$$

with memory $\approx 1/(1-\beta_2)$ steps. The learning rate becomes an
independent knob (→ schedules).

@adam-rmsprop-forgetting-on-purpose-1
:::

::: {.slide title="Adam = both moments, debiased"}
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t,
\qquad
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2,$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},\quad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t},\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,
\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.$$

- Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$: direction averaged over ~10
  steps, scale over ~1000.
- Bias correction is *exact*: $\mathbb{E}[\mathbf{v}_t]$ carries the
  fraction $1-\beta_2^t$ of the true scale; division cancels it at every
  $t$. After 10 steps, $\mathbf{v}_t$ holds ~1% of its stationary value —
  uncorrected Adam takes its biggest steps on its worst estimates.
:::

::: {.slide title="From scratch"}
Two buffers per parameter plus a step counter:

@adam-adam-both-moments-debiased-1

. . .

@adam-adam-both-moments-debiased-2
:::

::: {.slide title="A tiny language model"}
The differences that matter show on *language models*. `TinyLM`: a
decoder-only transformer (subject of ch. 10 — here, a black box), on the
character-level *Time Machine* from ch. 8.

@adam-a-tiny-language-model-2

::: {.d2l-note}
A differentiable function with a particular **census of parameters** —
the census, not the mechanism, is what optimization sees.
:::
:::

::: {.slide title="The parameter census"}
Three populations — later sections treat them differently (decay
exclusions, matrix vs. non-matrix preconditioning):

@!adam-a-tiny-language-model-3

- **embeddings**: rows update only when their token occurs
- **matrices**: ~96% of all parameters
- **vectors**: LayerNorm scales and biases
:::

::: {.slide title="The race: tuned SGD vs. tuned Adam"}
Symmetric protocol: same model, same init, same 2,000 steps, constant
learning rate, four-point grid each, best final training loss wins.

. . .

On the language model:

@!adam-the-race-on-the-language-model-3

- SGD's best lr sits one grid point below divergence; Adam's optimum is
  interior.
- The gap opens early and no rate in our grid closes it.
:::

::: {.slide title="Same race, same harness, on a CNN"}
@!adam-the-same-race-on-a-cnn-4

- Curves nearly coincide; test accuracy within a point or two, either way.
- This is why SGD carried computer vision for a decade — and why "which
  optimizer wins" depends on the *model*, not just the tuning.
:::

::: {.slide title="Why: heterogeneity"}
Live research, but the threads agree (Kunstner et al. 2023, 2024; Zhang
et al. 2024):

- Not minibatch noise: the gap persists **full-batch**; Adam tracks
  **sign descent** ($\sqrt{\hat{\mathbf{v}}} \approx |\hat{\mathbf{m}}|$
  → step $\approx \eta\,\mathrm{sign}(\hat{\mathbf{m}})$).
- Language is heavy-tailed: GD stalls on **rare tokens**; Adam keeps
  moving on all of them.
- Transformer blocks have wildly different curvature; CNN blocks look
  alike. One global $\eta$ must serve the stiffest block and starve the
  rest.
:::

::: {.slide title="Recap"}
- Adam = per-coordinate scaling (AdaGrad) + forgetting (RMSProp) +
  momentum + exact bias correction.
- Cost: two state buffers per parameter — 3× parameter memory.
- Wins big on transformers (heterogeneity), little on CNNs — at
  *matched* tuning.
- $\epsilon$ is a step ceiling $\eta/\epsilon$, not just a numerical
  guard; AMSGrad/Yogi patch the variance estimate (exercises).
:::
