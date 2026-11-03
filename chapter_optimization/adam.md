# Adam and AdamW
:label:`sec_adam`

The modern training recipe of :numref:`sec_training_recipes` replaced SGD
with momentum by AdamW, and we used it there as one ingredient among many.
Adam differs from the optimizers so far in one respect: it gives every
coordinate of the parameter vector its own learning rate. This section asks
what such per-coordinate rates achieve, how Adam sets them without knowing
the curvature, and why weight decay must then bypass them.

The valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ of :numref:`sec_gd` shows
what a single rate costs. Gradient descent uses one learning rate for every
coordinate, and the stiffest direction decides how large that rate may be.
Here the curvatures are $0.2$ and $4$, so a step is stable only for $\eta <
2/4 = 0.5$. At $\eta = 0.4$ the flat coordinate shrinks by a factor of only
$1 - 0.4 \cdot 0.2 = 0.92$ per step, and twenty steps take $x_1$ from $-5$
to about $-0.94$. With a separate rate for each coordinate, the inverse of
that coordinate's curvature, a single step would reach the minimum. Momentum
averages gradients over time and so eases the problem, but it keeps the
single rate.

The curvatures of a neural network are unknown, and they change during
training. The methods of this section therefore estimate a scale for each
coordinate from the history of that coordinate's own gradients. AdaGrad
accumulates the squared gradients, RMSProp lets the accumulation forget, and
Adam :cite:`Kingma.Ba.2014` adds momentum and corrects both averages for
their start at zero. Adam is the standard optimizer for transformers. To see
where its advantage comes from, we build two small models, a transformer
language model and a convolutional network, and compare Adam with tuned SGD
with momentum on both. The section ends with AdamW, the form in which Adam
is used in practice. AdamW applies weight decay outside the per-coordinate
scaling. It also stores two extra numbers for every parameter, and we count
that memory at the end.

```{.python .input #adam}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import copy
from functools import partial
import math
import numpy as np
import os
import pandas as pd
import random
import textwrap
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
import numpy as np
import optax
import os
import pandas as pd
import random
import textwrap
```

## Per-Coordinate Learning Rates

### AdaGrad

Consider learning problems whose features occur with very different
frequencies. This is the normal state of affairs in language: we will see
the word *learning* far more often than the word *preconditioning*. The
same imbalance appears in advertising and recommendation, where most items
interest only a few users. Parameters tied to rare features receive
informative gradients only when those features occur. Under the decaying
learning rate that stochastic convergence demands (:numref:`sec_sgd`), the
parameters of frequent features approach their optima quickly while the
rare ones are still waiting for evidence, and by the time it arrives the
shared step size has decayed too far to use it. One clock cannot serve both
populations.

AdaGrad :cite:`Duchi.Hazan.Singer.2011` gives every coordinate its own
clock. Instead of decaying the step with the iteration count $t$, it decays
each coordinate's step with that coordinate's own activity, measured by the
running sum of its squared gradients:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t,
$$
:eqlabel:`eq_adagrad`

with all operations elementwise, $\mathbf{s}_0 = \mathbf{0}$, and a small
$\epsilon > 0$ guarding the division. A coordinate whose gradients have been
large moves cautiously from now on. A coordinate that has barely been
touched keeps a large step and can respond as soon as its rare feature
appears.

The same rule addresses the valley. The ideal fix there divides the step in
each direction by the curvature of that direction. A transformation that
rescales the directions of a problem in this way is called a
*preconditioner*. Curvature is too expensive to compute at the scale of deep
learning. When the coordinate axes roughly align with the directions of
curvature, however, the size of a coordinate's past gradients is a usable
stand-in for its curvature. Then :eqref:`eq_adagrad` is a diagonal
preconditioner estimated from gradients alone.
:numref:`subsec_mdl-per-coordinate` makes both readings precise, through a
change of metric and through a regret bound.

We run AdaGrad on the valley from the starting point $(-5, -2)$ at the
learning rate $\eta = 0.4$, close to the stability limit of gradient
descent. The
helper `d2l.train_2d` of :numref:`sec_gd` carries the state $(s_1, s_2)$
from step to step and passes extra keyword arguments, such as `eta`, to the
update rule.

```{.python .input #adam-per-coordinate-learning-rates-1}
%%tab pytorch, jax
def f_valley(x1, x2):  # Curvatures 0.2 and 4
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def valley_grad(x1, x2):
    return 0.2 * x1, 4 * x2

def adagrad_2d(x1, x2, s1, s2, f_grad, eta=0.4, eps=1e-6):
    g1, g2 = f_grad(x1, x2)
    s1, s2 = s1 + g1 ** 2, s2 + g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

d2l.show_trace_2d(f_valley, d2l.train_2d(adagrad_2d, f_grad=valley_grad))
```

The path is smooth: the steep direction $x_2$ is scaled down automatically,
so it does not oscillate. But the accumulation also has a drawback. On the
first step $\mathbf{s}_1 = \mathbf{g}_1^2$, so every coordinate moves by
exactly $\eta$, whatever its gradient. After that, a coordinate whose
gradient stays roughly the same size moves by about $\eta/\sqrt{t}$ at step
$t$, and twenty such steps add up to at most $\eta \sum_{t=1}^{20}
t^{-1/2} \approx 7.6\,\eta = 3.0$. That is not enough to cover the
distance of $5$ along $x_1$, and the iterate stalls far from the minimum.
Because the scaling adapts, we can compensate with a learning rate at which
gradient descent diverges, $\eta = 2$:

```{.python .input #adam-per-coordinate-learning-rates-2}
%%tab pytorch, jax
d2l.show_trace_2d(f_valley, d2l.train_2d(adagrad_2d, f_grad=valley_grad,
                                          eta=2))
```

### RMSProp

AdaGrad's sum never forgets. Under persistent gradient noise
$\mathbf{s}_t$ grows roughly linearly in $t$, so the effective step decays
like $t^{-1/2}$ whether or not the optimization is anywhere near done. This
decay suits the convex problems that AdaGrad was designed for, but it can
stop a nonconvex optimization that still needs to move late in training.

RMSProp :cite:`Tieleman.Hinton.2012` keeps the per-coordinate scaling and
replaces the sum with an exponential moving average, the same leaky
averaging that momentum applies to the gradient itself:

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t.
$$
:eqlabel:`eq_rmsprop`

The average $\mathbf{v}_t$ estimates the *current* squared-gradient scale
rather than the lifetime total, over a window of about $1/(1-\beta_2)$
steps, ten at the customary $\beta_2 = 0.9$. The step size no longer
decays by construction, and the learning rate $\eta$ becomes a setting we
control separately, to be changed over time by a schedule
(:numref:`sec_scheduler`). On the valley at $\eta = 0.4$, RMSProp gets
close to the minimum where AdaGrad stalled:

```{.python .input #adam-rmsprop-forgetting-on-purpose-1}
%%tab pytorch, jax
def rmsprop_2d(x1, x2, v1, v2, f_grad, eta=0.4, beta2=0.9, eps=1e-6):
    g1, g2 = f_grad(x1, x2)
    v1 = beta2 * v1 + (1 - beta2) * g1 ** 2
    v2 = beta2 * v2 + (1 - beta2) * g2 ** 2
    x1 -= eta / math.sqrt(v1 + eps) * g1
    x2 -= eta / math.sqrt(v2 + eps) * g2
    return x1, x2, v1, v2

d2l.show_trace_2d(f_valley, d2l.train_2d(rmsprop_2d, f_grad=valley_grad))
```

The first step of this run is larger than intended. Since
$\mathbf{v}_0 = \mathbf{0}$, the average after one step is
$\mathbf{v}_1 = 0.1\, \mathbf{g}_1^2$, a tenth of the squared gradient, and
each coordinate moves by $\eta/\sqrt{0.1} \approx 3.2\,\eta$ instead of
$\eta$. The average starts at zero and needs about ten steps to fill its
window. Adam corrects this startup error exactly.

### Adam

Adam applies the same exponential averaging to the gradient itself, giving
a momentum estimate $\mathbf{m}_t$, keeps RMSProp's second-moment estimate
$\mathbf{v}_t$, and divides each average by the total weight it has
accumulated so far:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\, \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t,
\qquad &
\hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t), \\
\mathbf{v}_t &= \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad &
\hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t).
\end{aligned}
$$
:eqlabel:`eq_adam-moments`

The update divides the corrected momentum by the square root of the
corrected scale:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
$$
:eqlabel:`eq_adam-update`

Each coordinate thus moves by about $\eta$ times the ratio of its average
gradient to its typical gradient size. The momentum method of
:numref:`sec_momentum` *adds* each gradient to $\beta$ times its buffer.
The factor $1 - \beta_1$ turns that sum into an *average*, which amounts to
the same method with the learning rate multiplied by $1 - \beta_1$. The
defaults are $\beta_1 = 0.9$ and $\beta_2 = 0.999$: the direction estimate
averages over about ten steps, the scale estimate over about a thousand.
Language-model recipes since GPT-3 :cite:`brown2020language` often use
$\beta_2 = 0.95$, a window of twenty steps. The scale estimate then follows
changes in the gradients more quickly, but with more noise.

The corrections in :eqref:`eq_adam-moments` are exact rather than
heuristic.
Unrolled from $\mathbf{v}_0 = \mathbf{0}$, the average
$\mathbf{v}_t = (1-\beta_2)\sum_{s=1}^{t} \beta_2^{t-s}\, \mathbf{g}_s^2$
is a weighted sum whose weights total $1 - \beta_2^t$, not $1$. If the
squared gradients have a steady mean, $\mathbf{v}_t$ therefore carries the
fraction $1 - \beta_2^t$ of it in expectation, and dividing by that
fraction removes the deficit at every $t$. The deficit is large: at
$\beta_2 = 0.999$, after ten steps $\mathbf{v}_t$ holds about $1\%$ of its
stationary value. Without the corrections, the first step would be
$(1-\beta_1)/\sqrt{1-\beta_2} = 0.1/\sqrt{0.001} \approx 3.2$ times
$\eta$, the same overshoot as RMSProp's. With them, $\hat{\mathbf{m}}_1 =
\mathbf{g}_1$ and $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so the first step
moves every coordinate by exactly $\eta$. :numref:`subsec_mdl-per-coordinate`
plots the transient and its cancellation.

Adam also adds $\epsilon$ *outside* the square root, unlike
:eqref:`eq_rmsprop`. Where $\sqrt{\hat{\mathbf{v}}_t}$ is of order one the
placement matters little. It decides the update where
$\sqrt{\hat{\mathbf{v}}_t}$ is small, and we return to it when we look at
the limits of the method. The usual default is $\epsilon = 10^{-8}$.

### Adam from Scratch

The update needs two buffers per parameter and a step counter for the
bias correction. We write it the way the framework writes its own
optimizers, so that it can train any model.

```{.python .input #adam-adam-both-moments-debiased-1}
%%tab pytorch
class Adam(torch.optim.Optimizer):
    """Adam from scratch: steps scaled by two debiased moving averages."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            (beta1, beta2), lr, eps = group['betas'], group['lr'], group['eps']
            for p in group['params']:
                if p.grad is None:  # A parameter that received no gradient
                    continue
                state = self.state[p]
                if not state:  # First step: t = 0 and m = v = 0
                    state.update(t=0, m=torch.zeros_like(p),
                                 v=torch.zeros_like(p))
                state['t'] += 1
                t, m, v = state['t'], state['m'], state['v']
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).add_(p.grad ** 2, alpha=1 - beta2)
                m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))
```

```{.python .input #adam-adam-both-moments-debiased-1}
%%tab jax
def scale_by_adam(b1=0.9, b2=0.999, eps=1e-8):
    """Adam's direction m_hat / (sqrt(v_hat) + eps), from scratch."""
    def init(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        return dict(t=jnp.zeros((), jnp.int32), m=zeros, v=zeros)

    def update(grads, state, params=None):
        t = state['t'] + 1
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g,
                         state['m'], grads)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2,
                         state['v'], grads)
        def direction(m, v):
            return m / (1 - b1 ** t) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        return jax.tree.map(direction, m, v), dict(t=t, m=m, v=v)
    return optax.GradientTransformation(init, update)

def adam(learning_rate):
    return optax.chain(scale_by_adam(), optax.scale(-learning_rate))
```

:begin_tab:`jax`
An Optax optimizer is a chain of gradient transformations. `scale_by_adam`
turns the gradient into Adam's direction, and `optax.scale` multiplies it
by $-\eta$. Keeping the two apart lets us insert weight decay between them
later in the section.
:end_tab:

Before using our version we check it against the library on a small
least-squares problem: two copies of the same linear model, the same data,
100 steps each. The first step should move every weight by exactly $\eta$.

```{.python .input #adam-adam-both-moments-debiased-2}
%%tab pytorch
def compare(ours, theirs, lr, num_steps=100):
    X, y = torch.randn(512, 20), torch.randn(512, 1)
    net = nn.Linear(20, 1)
    twin = copy.deepcopy(net)
    optimizers = [ours(net.parameters(), lr), theirs(twin.parameters(), lr)]
    w0 = net.weight.detach().clone()
    for t in range(num_steps):
        for model, optimizer in zip([net, twin], optimizers):
            loss = F.mse_loss(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if t == 0:
            first = (net.weight - w0).abs() / lr
            print(f'first step / lr: {first.min():.4f} to {first.max():.4f}')
    print(f'largest difference after {num_steps} steps: '
          f'{(net.weight - twin.weight).abs().max():.1e}')

compare(Adam, torch.optim.Adam, lr=0.01)
```

```{.python .input #adam-adam-both-moments-debiased-2}
%%tab jax
@nnx.jit
def least_squares_step(model, optimizer, X, y):
    grads = nnx.grad(lambda model: ((model(X) - y) ** 2).mean())(model)
    optimizer.update(model, grads)

def compare(ours, theirs, lr, num_steps=100):
    k_X, k_y, k_w = jax.random.split(jax.random.key(random.getrandbits(32)), 3)
    X, y = jax.random.normal(k_X, (512, 20)), jax.random.normal(k_y, (512, 1))
    net = nnx.Linear(20, 1, rngs=nnx.Rngs(k_w))
    twin = nnx.clone(net)
    optimizers = [nnx.Optimizer(net, ours, wrt=nnx.Param),
                  nnx.Optimizer(twin, theirs, wrt=nnx.Param)]
    w0 = net.kernel[...]
    for t in range(num_steps):
        for model, optimizer in zip([net, twin], optimizers):
            least_squares_step(model, optimizer, X, y)
        if t == 0:
            first = jnp.abs(net.kernel[...] - w0) / lr
            print(f'first step / lr: {first.min():.4f} to {first.max():.4f}')
    diff = jnp.abs(net.kernel[...] - twin.kernel[...]).max()
    print(f'largest difference after {num_steps} steps: {diff:.1e}')

compare(adam(0.01), optax.adam(0.01), lr=0.01)
```

The first step moves every weight by $\eta$, as the bias correction
promises, and after 100 steps the two copies agree to within rounding
error. From now on we call the library version.

## TinyLM and FashionCNN
:label:`subsec_tinylm`

A comparison of optimizers needs models on which the optimizers behave
differently, and models small enough to train many times. We use two of
about the same size: a transformer language model, the kind of model Adam
is used for in practice, and a convolutional image classifier, the kind of
model that SGD with momentum trained for a decade. Both go into the `d2l`
library, and the rest of this chapter trains them.

### The Language Model

The data is WikiText-103 :cite:`Merity.Xiong.Bradbury.ea.2016`, about 100
million words of Wikipedia articles kept as raw text, with case, digits, and
punctuation. A run of 2,000 steps on minibatches of 64 windows of 64 tokens
makes about 8 million predictions. On the 100,000 characters of *The Time
Machine* each character would be a target about 80 times, and the model
would memorize the text. Here the training windows start at random offsets
in the first of the two training files, about 270 million bytes, so a run
reads about 3% of it and rarely sees the same text twice. The tokens are the
UTF-8 bytes of the text: the byte-level tokenizer of
:numref:`sec_text-sequence` without any merges, a vocabulary of 256 tokens,
so the loss converts directly to bits per byte
(:numref:`sec_language-model`). Validation windows are consecutive,
non-overlapping pieces of the official validation split, about 1.1 million
bytes.

```{.python .input #adam-the-language-model-1}
%%tab pytorch
class WikiText103(d2l.DataModule):  #@save
    """WikiText-103 as UTF-8 bytes, cut into windows of num_steps bytes."""
    def __init__(self, batch_size, num_steps, num_train=None, num_val=None):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = d2l.BPETokenizer(vocab_size=256, specials=())  # Bytes
        self.train = self._read('train-00000-of-00002',
                                '3cbc792dfc0eeedf8d7333acdc15a9f536032d9a')
        self.val = self._read('validation-00000-of-00001',
                              '49d15019b0476afc752bd5f010b5a2aab2de89ca')
        self.train, self.val = self.train[:num_train], self.val[:num_val]

    def _read(self, split, sha1):
        url = ('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
               f'main/wikitext-103-raw-v1/{split}.parquet')
        fname = d2l.download(url, os.path.join(self.root, 'wikitext-103'),
                             sha1)
        lines = pd.read_parquet(fname)['text']  # Each line ends with '\n'
        return np.frombuffer(b''.join(s.encode() for s in lines), np.uint8)

    def windows(self, corpus, starts):
        """Inputs and next-byte targets of the windows at `starts`."""
        idx = np.asarray(starts)[:, None] + np.arange(self.num_steps + 1)
        array = torch.from_numpy(corpus[idx].astype(np.int64))
        return array[:, :-1], array[:, 1:]

    def train_dataloader(self):  # A pass of minibatches at random offsets
        n = len(self.train) - self.num_steps
        for _ in range(n // (self.batch_size * self.num_steps)):
            yield self.windows(self.train,
                               torch.randint(n, (self.batch_size,)).numpy())

    def val_dataloader(self):  # Consecutive, non-overlapping windows
        starts = np.arange(0, len(self.val) - self.num_steps, self.num_steps)
        for i in range(0, len(starts), self.batch_size):
            yield self.windows(self.val, starts[i:i + self.batch_size])
```

```{.python .input #adam-the-language-model-1}
%%tab jax
class WikiText103(d2l.DataModule):  #@save
    """WikiText-103 as UTF-8 bytes, cut into windows of num_steps bytes."""
    def __init__(self, batch_size, num_steps, num_train=None, num_val=None):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = d2l.BPETokenizer(vocab_size=256, specials=())  # Bytes
        self.train = self._read('train-00000-of-00002',
                                '3cbc792dfc0eeedf8d7333acdc15a9f536032d9a')
        self.val = self._read('validation-00000-of-00001',
                              '49d15019b0476afc752bd5f010b5a2aab2de89ca')
        self.train, self.val = self.train[:num_train], self.val[:num_val]

    def _read(self, split, sha1):
        url = ('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
               f'main/wikitext-103-raw-v1/{split}.parquet')
        fname = d2l.download(url, os.path.join(self.root, 'wikitext-103'),
                             sha1)
        lines = pd.read_parquet(fname)['text']  # Each line ends with '\n'
        return np.frombuffer(b''.join(s.encode() for s in lines), np.uint8)

    def windows(self, corpus, starts):
        """Inputs and next-byte targets of the windows at `starts`."""
        idx = np.asarray(starts)[:, None] + np.arange(self.num_steps + 1)
        array = corpus[idx].astype(np.int32)
        return array[:, :-1], array[:, 1:]

    def train_dataloader(self):  # A pass of minibatches at random offsets
        n = len(self.train) - self.num_steps
        rng = np.random.default_rng(random.getrandbits(32))
        for _ in range(n // (self.batch_size * self.num_steps)):
            yield self.windows(self.train,
                               rng.integers(n, size=self.batch_size))

    def val_dataloader(self):  # Consecutive, non-overlapping windows
        starts = np.arange(0, len(self.val) - self.num_steps, self.num_steps)
        for i in range(0, len(starts), self.batch_size):
            yield self.windows(self.val, starts[i:i + self.batch_size])
```

Both splits are held in memory as arrays of bytes, one byte per token, and
a minibatch is gathered from them on demand. A training pass draws as many
windows as the text holds non-overlapping ones, each at an independent
random offset, so `train_lm` below can run for any number of steps without
reusing a fixed set of windows.

```{.python .input #adam-a-tiny-language-model-1}
%%tab pytorch, jax
data = WikiText103(batch_size=64, num_steps=64)
print(f'{len(data.train):,} training bytes, '
      f'{len(data.val):,} validation bytes')
text = data.vocab.decode(data.train[:300])
print('\n'.join(textwrap.fill(line, 72) for line in text.splitlines()))
```

The model, `TinyLM`, is a *decoder-only transformer*, the architecture of
:numref:`chap_transformers`. We use it two chapters early because an
optimizer does not need to know what attention does: it sees a
differentiable function with parameters of several kinds, and only their
gradients. The model has learned token and position embeddings, two blocks,
and a linear head that predicts the next byte. Each block has an
attention layer and a two-layer MLP, each behind a LayerNorm and each with a
residual connection. The attention computation is a single library call.
Every linear layer draws its weights and biases uniformly from
$\pm 1/\sqrt{d_\textrm{in}}$, and every embedding row from a standard
normal distribution.

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
@jax.custom_vjp  #@save
def embedding_rows(table, X):
    """The rows X of table."""
    return table[X]

def _embedding_rows_fwd(table, X):  #@save
    return table[X], (table, X)

def _embedding_rows_bwd(residuals, grad):  #@save
    """Gradient of the table: the one-hot rows of X times the gradients."""
    table, X = residuals
    one_hot = jax.nn.one_hot(X, len(table), dtype=grad.dtype)
    return jnp.einsum('...v,...d->vd', one_hot, grad,
                      precision='highest').astype(table.dtype), None

embedding_rows.defvjp(_embedding_rows_fwd, _embedding_rows_bwd)  #@save

class Embed(nnx.Embed):  #@save
    """nnx.Embed whose gradient is a matrix product for small vocabularies."""
    def __call__(self, X):
        table = self.embedding[...]
        if len(table) > 1024:  # A one-hot row would cost too much
            return super().__call__(X)
        return embedding_rows(table, X).astype(self.dtype or table.dtype)

class TinyLM(nnx.Module):  #@save
    """A small decoder-only transformer language model."""
    def __init__(self, vocab_size, d_model=128, num_heads=2, num_blks=2,
                 max_len=64, *, rngs):
        self.num_heads = num_heads
        def linear(d_in, d_out):
            a = 1 / math.sqrt(d_in)  # Weights and biases ~ U(-a, a)
            init = lambda key, shape, dtype: jax.random.uniform(
                key, shape, dtype, -a, a)
            return nnx.Linear(d_in, d_out, kernel_init=init, bias_init=init,
                              rngs=rngs)
        def embed(n):  # Embedding rows ~ N(0, I)
            return Embed(n, d_model, rngs=rngs,
                         embedding_init=nnx.initializers.normal(1.0))
        def norm():
            return nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.token_emb, self.pos_emb = embed(vocab_size), embed(max_len)
        self.blks = nnx.List([nnx.Dict(
            norm1=norm(), qkv=linear(d_model, 3 * d_model),
            proj=linear(d_model, d_model), norm2=norm(),
            mlp1=linear(d_model, 4 * d_model),
            mlp2=linear(4 * d_model, d_model))
            for _ in range(num_blks)])
        self.norm = norm()
        self.head = linear(d_model, vocab_size)

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
            H = H + blk['mlp2'](jax.nn.gelu(blk['mlp1'](blk['norm2'](H)),
                                            approximate=False))
        return self.head(self.norm(H))
```

:begin_tab:`pytorch`
The initialization described above is the default of `nn.Linear` and
`nn.Embedding`, and `nn.GELU` computes the exact GELU.
:end_tab:

:begin_tab:`jax`
Flax's default initializers differ from the ones described above, so
`TinyLM` passes them explicitly: uniform $\pm 1/\sqrt{d_\textrm{in}}$ for
every kernel and bias, a standard normal for the embeddings, and
LayerNorm with $\epsilon = 10^{-5}$. `approximate=False` selects the exact
GELU. The random stream `rngs` is a required argument: every model states
where its initialization comes from.

The embeddings are `Embed`, an `nnx.Embed` with a faster gradient. The
gradient of a lookup adds the gradient at each position to the row of that
position's token, and XLA performs these additions with atomic operations.
In a minibatch of text, thousands of positions hold the same few frequent
bytes, such as the space and `e`. Their additions to the same row wait for
one another, and on our GPU the sum takes tens of times as long as for
random tokens. For vocabularies of up to 1,024 tokens, `Embed` computes
the same sum as one matrix product of the one-hot rows of the tokens with
the gradients.
:end_tab:

### Parameter Groups

The parameters of `TinyLM` fall into three groups. *Embeddings* are tables
whose rows receive a gradient only when their token or position occurs,
the sparse features that motivated AdaGrad. Two-dimensional *matrices*, the
weights of the linear layers, hold nearly all of the parameters.
One-dimensional *vectors* are the biases and the LayerNorm scales and
shifts. Weight decay at the end of this section and Muon in
:numref:`sec_muon` treat the three groups differently, so we count them
once here.

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
model = TinyLM(vocab_size=len(data.vocab),
               rngs=nnx.Rngs(random.getrandbits(32)))
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

The model has about 0.47 million parameters, 90% of them in the matrices.

### Training for a Fixed Number of Steps

Training is a loop counted in steps rather than epochs: draw a minibatch,
compute the mean cross-entropy of the next-token predictions, take an
optimizer step, record the loss. Nothing in it is specific to language: it
trains any model that maps a minibatch to logits, so we save it for the
rest of the book.

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
            losses.append(loss.detach())
            step += 1
            if step >= num_steps:
                return torch.stack(losses).tolist()
```

```{.python .input #adam-a-tiny-language-model-4}
%%tab jax
@nnx.jit  #@save
def _lm_train_step(model, optimizer, X, Y):
    def loss_fn(model):
        logits = model(X)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def train_lm(model, data, optimizer, num_steps):  #@save
    """Train a model on next-token prediction for a fixed number of steps."""
    losses, pending, step = [], [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            pending.append(_lm_train_step(model, optimizer, jnp.asarray(X),
                                          jnp.asarray(Y)))
            step += 1
            if len(pending) == 50 or step >= num_steps:  # Sync now and then
                losses.extend(float(l) for l in jax.device_get(pending))
                pending = []
            if step >= num_steps:
                return losses
```

Three helpers read a run: the final loss, averaged over the last hundred
steps and set to infinity if the run diverged; a curve smoothed over blocks
of 25 steps for plotting; and the first step at which the smoothed curve
reaches a given loss, or `None` if it never does.

```{.python .input #adam-a-tiny-language-model-5}
%%tab pytorch, jax
def final_loss(losses, k=100):
    v = sum(losses[-k:]) / k
    return v if math.isfinite(v) else float('inf')

def smooth(losses, k=25):
    return [sum(losses[i:i + k]) / k
            for i in range(0, len(losses) - k + 1, k)]

def steps_to_reach(losses, target, k=25):
    for i, v in enumerate(smooth(losses, k)):
        if v <= target:
            return k * (i + 1)
    return None
```

A first run trains `TinyLM` with Adam at $\eta = 0.003$ for 2,000 steps:

```{.python .input #adam-a-tiny-language-model-6}
%%tab pytorch
model = TinyLM(len(data.vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
losses = train_lm(model, data, optimizer, num_steps=2000)
d2l.plot(list(range(0, 2000, 25)), [smooth(losses)],
         'step', 'training loss')
print(f'training perplexity per byte: '
      f'{math.exp(final_loss(losses)):.2f}')
```

```{.python .input #adam-a-tiny-language-model-6}
%%tab jax
model = TinyLM(len(data.vocab), rngs=nnx.Rngs(random.getrandbits(32)))
optimizer = nnx.Optimizer(model, optax.adam(0.003), wrt=nnx.Param)
losses = train_lm(model, data, optimizer, num_steps=2000)
d2l.plot(list(range(0, 2000, 25)), [smooth(losses)],
         'step', 'training loss')
print(f'training perplexity per byte: '
      f'{math.exp(final_loss(losses)):.2f}')
```

The model reaches a training perplexity per byte of about 5, or about
2.3 bits per byte, against 256 for a uniform guess over the bytes: it has
learned a good deal of English spelling, and its loss is still falling
slowly when the run ends. Strong language models compress text like this
to below one bit per byte (:numref:`sec_language-model`).

### The Image Classifier
:label:`subsec_fashioncnn`

`FashionCNN` classifies the $28 \times 28$ grayscale images of
Fashion-MNIST (:numref:`sec_fashion_mnist`). Two stages of a $3 \times 3$
convolution, a ReLU, and $2 \times 2$ max-pooling, with 32 and 64 channels,
reduce each image to $64 \times 7 \times 7$ features. Two fully connected
layers map them to 128 hidden units and then to the ten classes. Every
weight and bias is drawn uniformly from $\pm 1/\sqrt{\textrm{fan-in}}$,
where the fan-in of a unit is the number of inputs it sums over.

```{.python .input #adam-the-same-race-on-a-cnn-1}
%%tab pytorch
class FashionCNN(nn.Module):  #@save
    """A compact CNN for Fashion-MNIST: two conv-pool stages, two linear."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), 2)
        X = F.max_pool2d(F.relu(self.conv2(X)), 2)
        return self.fc2(F.relu(self.fc1(X.flatten(1))))
```

```{.python .input #adam-the-same-race-on-a-cnn-1}
%%tab jax
class FashionCNN(nnx.Module):  #@save
    """A compact CNN for Fashion-MNIST: two conv-pool stages, two linear."""
    def __init__(self, *, rngs):
        def init(fan_in):  # Weights and biases ~ U(-a, a), a = 1/sqrt(fan_in)
            a = 1 / math.sqrt(fan_in)
            f = lambda key, shape, dtype: jax.random.uniform(
                key, shape, dtype, -a, a)
            return dict(kernel_init=f, bias_init=f, rngs=rngs)
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), **init(9))
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), **init(9 * 32))
        self.fc1 = nnx.Linear(64 * 7 * 7, 128, **init(64 * 7 * 7))
        self.fc2 = nnx.Linear(128, 10, **init(128))

    def __call__(self, X):
        X = nnx.max_pool(nnx.relu(self.conv1(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = nnx.max_pool(nnx.relu(self.conv2(X)), window_shape=(2, 2),
                         strides=(2, 2))
        return self.fc2(nnx.relu(self.fc1(X.reshape(X.shape[0], -1))))
```

:begin_tab:`pytorch`
The initialization described above is the default of `nn.Conv2d` and
`nn.Linear`. Images arrive as (batch, channel, height, width).
:end_tab:

:begin_tab:`jax`
Flax's default initializers draw kernels from a normal distribution scaled
by the fan-in and set biases to zero, so `FashionCNN` passes the uniform
initialization explicitly. Images arrive as (batch, height, width,
channel). As for `TinyLM`, the random stream `rngs` is required.
:end_tab:

With 421,642 parameters, `FashionCNN` is about the same size as
`TinyLM`. Since `train_lm` only asks a model for logits, it trains the CNN
unchanged; Fashion-MNIST minibatches of 256 images take the place of the
byte windows. The CNN also needs a test accuracy:

```{.python .input #adam-the-image-classifier-1}
%%tab pytorch
fashion = d2l.FashionMNIST(batch_size=256)

def test_accuracy(model, data):
    device = next(model.parameters()).device
    correct, total = 0, 0
    with torch.inference_mode():
        for X, y in data.val_dataloader():
            y_hat = model(X.to(device)).argmax(dim=1)
            correct += (y_hat == y.to(device)).sum()
            total += len(y)
    return float(correct) / total
```

```{.python .input #adam-the-image-classifier-1}
%%tab jax
fashion = d2l.FashionMNIST(batch_size=256)

@nnx.jit
def num_correct(model, X, y):
    return (model(X).argmax(axis=1) == y).sum()

def test_accuracy(model, data):
    correct, total = 0, 0
    for X, y in data.val_dataloader():
        correct += num_correct(model, jnp.asarray(X), jnp.asarray(y))
        total += len(y)
    return float(correct) / total
```

## Strengths and Limits of Adam
:label:`subsec_adam-where-wins`

Practitioners train transformers with Adam because SGD, however carefully
tuned, falls behind. On convolutional networks the case for Adam is far
weaker, and SGD with momentum was the standard for a decade of computer
vision. Claims that one optimizer beats another depend strongly on how much
effort went into tuning each :cite:`Schmidt.Schneider.Hennig.2021`, so the
protocol of the comparison matters as much as its result.

Ours is symmetric and small. Both optimizers train the same model from the
same initialization, a random draw shared by every run of the comparison,
for 2,000 steps at a constant learning rate, with no schedule, no clipping,
and no weight decay. Each gets a grid of four learning rates spaced by
factors of about three, and the run with the lowest final training loss
represents its method. Training loss is the right measure here, because
the question is which method optimizes faster, not which one generalizes
better. For the language model the two would agree anyway, since a run
rarely sees the same text twice. Below, "tuned" means the best of these
four rates.

### Comparison on the Language Model

The function `run_lm` trains one fresh `TinyLM` per learning rate and
prints its final loss.

```{.python .input #adam-the-race-on-the-language-model-1}
%%tab pytorch
def run_lm(make_optimizer, data, lrs, seed, num_steps=2000):
    curves = {}
    for lr in lrs:
        torch.manual_seed(seed)  # The same initialization for every lr
        model = TinyLM(len(data.vocab))
        curves[lr] = train_lm(model, data, make_optimizer(model, lr),
                              num_steps)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}')
    return curves

lm_seed = random.getrandbits(31)
sgd_lm = run_lm(lambda model, lr: torch.optim.SGD(model.parameters(), lr,
                                                  momentum=0.9),
                data, lrs=[0.03, 0.1, 0.3, 1.0], seed=lm_seed)
```

```{.python .input #adam-the-race-on-the-language-model-1}
%%tab jax
def run_lm(make_tx, data, lrs, seed, num_steps=2000):
    curves = {}
    for lr in lrs:  # The same initialization for every lr
        model = TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        curves[lr] = train_lm(model, data, optimizer, num_steps)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}')
    return curves

lm_seed = random.getrandbits(32)
sgd_lm = run_lm(lambda lr: optax.sgd(lr, momentum=0.9), data,
                lrs=[0.03, 0.1, 0.3, 1.0], seed=lm_seed)
```

```{.python .input #adam-the-race-on-the-language-model-2}
%%tab pytorch
adam_lm = run_lm(lambda model, lr: torch.optim.Adam(model.parameters(), lr),
                 data, lrs=[3e-4, 1e-3, 3e-3, 1e-2], seed=lm_seed)
```

```{.python .input #adam-the-race-on-the-language-model-2}
%%tab jax
adam_lm = run_lm(optax.adam, data, lrs=[3e-4, 1e-3, 3e-3, 1e-2],
                 seed=lm_seed)
```

SGD's best learning rate sits directly below one at which training
diverges, printed as `inf`. Adam's best rate is interior, with stable
neighbors on both sides. We plot the two tuned runs:

```{.python .input #adam-the-race-on-the-language-model-3}
%%tab pytorch, jax
best_sgd = min(sgd_lm, key=lambda lr: final_loss(sgd_lm[lr]))
best_adam = min(adam_lm, key=lambda lr: final_loss(adam_lm[lr]))
d2l.plot(list(range(0, 2000, 25)),
         [smooth(sgd_lm[best_sgd]), smooth(adam_lm[best_adam])],
         'step', 'training loss',
         legend=[f'SGD + momentum, lr {best_sgd:g}',
                 f'Adam, lr {best_adam:g}'])
loss_sgd = final_loss(sgd_lm[best_sgd])
loss_adam = final_loss(adam_lm[best_adam])
print(f'final loss: SGD {loss_sgd:.3f}, Adam {loss_adam:.3f}, '
      f'gap {loss_sgd - loss_adam:.3f} nats')
print(f"Adam reaches SGD's final loss at step "
      f'{steps_to_reach(adam_lm[best_adam], loss_sgd)} of 2000')
```

Tuned Adam ends about 0.03 to 0.2 nats below tuned SGD, a perplexity 3
to 20% lower, and its curve lies below SGD's from the first hundred steps
to the end. No learning rate in SGD's grid closes the gap: the best one is
the last stable one. Repeated with fresh random draws, both tuned losses
move by a few hundredths of a nat, and SGD, run just below the rate at
which it diverges, can end up to about a tenth of a nat higher. The last
line of the output measures the gap in time rather than in loss. Adam
reaches the loss on which SGD ends 40% to 80% of the way through the run,
so it optimizes this model about 1.3 to 2.5 times as fast.

### Comparison on a CNN

The same protocol trains `FashionCNN`. Adam's grid stays the same, SGD's
grid moves one step lower, and every run also records its test accuracy.

```{.python .input #adam-the-same-race-on-a-cnn-2}
%%tab pytorch
def run_cnn(make_optimizer, data, lrs, seed, num_steps=2000):
    curves, accs = {}, {}
    for lr in lrs:
        torch.manual_seed(seed)  # The same initialization for every lr
        model = FashionCNN()
        curves[lr] = train_lm(model, data, make_optimizer(model, lr),
                              num_steps)
        accs[lr] = test_accuracy(model, data)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}, '
              f'test accuracy {accs[lr]:.3f}')
    return curves, accs

seed = random.getrandbits(31)
sgd_cnn, sgd_cnn_acc = run_cnn(
    lambda model, lr: torch.optim.SGD(model.parameters(), lr, momentum=0.9),
    fashion, lrs=[0.01, 0.03, 0.1, 0.3], seed=seed)
```

```{.python .input #adam-the-same-race-on-a-cnn-2}
%%tab jax
def run_cnn(make_tx, data, lrs, seed, num_steps=2000):
    curves, accs = {}, {}
    for lr in lrs:  # The same initialization for every lr
        model = FashionCNN(rngs=nnx.Rngs(seed))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        curves[lr] = train_lm(model, data, optimizer, num_steps)
        accs[lr] = test_accuracy(model, data)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}, '
              f'test accuracy {accs[lr]:.3f}')
    return curves, accs

seed = random.getrandbits(32)
sgd_cnn, sgd_cnn_acc = run_cnn(lambda lr: optax.sgd(lr, momentum=0.9),
                               fashion, lrs=[0.01, 0.03, 0.1, 0.3],
                               seed=seed)
```

```{.python .input #adam-the-same-race-on-a-cnn-3}
%%tab pytorch
adam_cnn, adam_cnn_acc = run_cnn(
    lambda model, lr: torch.optim.Adam(model.parameters(), lr),
    fashion, lrs=[3e-4, 1e-3, 3e-3, 1e-2], seed=seed)
```

```{.python .input #adam-the-same-race-on-a-cnn-3}
%%tab jax
adam_cnn, adam_cnn_acc = run_cnn(optax.adam, fashion,
                                 lrs=[3e-4, 1e-3, 3e-3, 1e-2], seed=seed)
```

```{.python .input #adam-the-same-race-on-a-cnn-4}
%%tab pytorch, jax
best_sgd_cnn = min(sgd_cnn, key=lambda lr: final_loss(sgd_cnn[lr]))
best_adam_cnn = min(adam_cnn, key=lambda lr: final_loss(adam_cnn[lr]))
d2l.plot(list(range(0, 2000, 25)),
         [smooth(sgd_cnn[best_sgd_cnn]), smooth(adam_cnn[best_adam_cnn])],
         'step', 'training loss',
         legend=[f'SGD + momentum, lr {best_sgd_cnn:g}',
                 f'Adam, lr {best_adam_cnn:g}'])
loss_sgd = final_loss(sgd_cnn[best_sgd_cnn])
loss_adam = final_loss(adam_cnn[best_adam_cnn])
print(f'final loss: SGD {loss_sgd:.3f}, Adam {loss_adam:.3f}, '
      f'gap {loss_sgd - loss_adam:.3f} nats')
print(f"Adam reaches SGD's final loss at step "
      f'{steps_to_reach(adam_cnn[best_adam_cnn], loss_sgd)} of 2000')
print(f'test accuracy: SGD {sgd_cnn_acc[best_sgd_cnn]:.3f}, '
      f'Adam {adam_cnn_acc[best_adam_cnn]:.3f}')
```

Two things repeat from the language model. SGD's best rate again sits
close to the edge of stability: at $\eta = 0.3$ training is unreliable.
Often the loss stays at $\log 10 \approx 2.30$, the loss of a uniform
guess over the ten classes, and when the run does train, it ends well
above the run at $\eta = 0.1$. On both models SGD trains best at or just
below the largest rate that trains reliably. And tuned Adam again reaches
a lower training loss.

Both losses are cross-entropies in nats per prediction, so the two gaps
can be compared directly: a few hundredths to about a tenth of a nat on
the CNN, about 0.03 to 0.2 on the language model. A gap of $\Delta$ nats
means that the better model assigns, on average, $e^\Delta$ times as much
probability to each correct answer: a few percent to about 10% more on the
CNN, about 3 to 20% more on the language model.

Measured in time, the question our protocol asks, the two models are
close as well: on both, Adam reaches SGD's final loss about 1.3 to 2.5
times as fast. After the first hundred steps the two CNN curves run
within about a tenth of a nat of each other, and the test accuracies of
the two tuned runs lie within a point or two of each other, between 90%
and 93%. On this problem SGD with momentum is a competitive choice, which
is why it served computer vision for a decade.

### Sources of the Gap

On our two small models Adam's speedup is modest and of similar size. On
large transformers it is decisive: SGD with momentum falls far behind Adam
there, while on CNNs of comparable scale the two stay close
:cite:`Zhang.Chen.Ding.ea.2024`. The reasons are still being researched,
but three findings point in the same direction.

First, the gap is not about minibatch noise.
:citet:`Kunstner.Chen.Lavington.ea.2023` found that it persists in
*full-batch* training, where there is no noise to average away, and that
Adam's advantage follows that of *sign descent*, the method that moves each
coordinate by a fixed amount in the direction opposite to the sign of its
gradient. The connection is visible in :eqref:`eq_adam-update`: with both
averages switched off ($\beta_1 = \beta_2 = 0$) and $\epsilon = 0$, the
update becomes $\eta\, \mathbf{g}_t / |\mathbf{g}_t| = \eta\,
\mathrm{sign}(\mathbf{g}_t)$. Adam is a smoothed sign method. On these
problems it mostly equalizes the step across coordinates whose gradients
differ by orders of magnitude.

Second, the differences being equalized come from the data. Word
frequencies are heavy-tailed: most words are rare, so the embedding rows
and output weights that serve them see small, infrequent gradients, while
those of frequent words see large, constant ones.
:citet:`Kunstner.Yadav.Milligan.ea.2024` showed that gradient descent makes
almost no progress on the loss of the rare classes while Adam makes
progress on all of them, even in linear models with imbalanced classes.

Third, they come from the architecture. Across the blocks of a
transformer, the embedding, attention, and MLP parameters have curvatures
so different that no single learning rate suits them all, whereas the
blocks of a CNN look much more alike. A single learning rate must be small
enough for the stiffest block, which slows all the others. A
per-coordinate method gives every block its own effective step. The three
parameter groups of `TinyLM` are a coarse version of this division: groups
with different roles and different gradient statistics, all sharing one
$\eta$ under SGD.

How much each finding explains for `TinyLM` is open, and the model is a
mild case of the last two. Its bytes are skewed, but the rare ones carry
little of the loss. Every lowercase letter appears more than twice in an
average minibatch of 4,096 predictions. About a hundred byte values, mostly
pieces of the UTF-8 encodings of non-English characters, appear less than
once in ten minibatches, but together they make up less than 0.1% of the
text, whereas the rare words of a word-level vocabulary together make up a
sizable share of it. And with two blocks of width 128 the model is far
smaller than the transformers of these studies. The exercises test both
explanations on this model.

### Limits of the Variance Estimate
:label:`subsec_adam-variance-estimate`

Adam's denominator $\sqrt{\hat{\mathbf{v}}_t}$ is an estimate built from
one noisy stream of gradients, and its errors are not symmetric: an
estimate that is too small produces a step that is too large. The worst
case is a coordinate whose gradient is usually small and occasionally
large. Between the large gradients the average $\mathbf{v}_t$ decays toward
the small ones, so when the next large gradient arrives, its step is far
larger than the scale of that coordinate warrants.
:citet:`Reddi.Kale.Kumar.2019` built a convex online problem from this
pattern on which Adam converges to the *worst* feasible point. The problem
is adversarial and says nothing about ordinary training data, but it shows
that Adam has no general convergence guarantee. Their repair, AMSGrad,
divides by the running maximum of $\hat{\mathbf{v}}_t$, so that the
estimate can never shrink. :numref:`subsec_mdl-per-coordinate` works
through the construction and the repair. Neither AMSGrad nor later repairs
replaced Adam in practice.

The constant $\epsilon$ limits the damage. It is meant to guard against
division by zero, but it also acts as a floor on the denominator of
:eqref:`eq_adam-update`. The learning rate of a coordinate, the factor
$\eta/(\sqrt{\hat{v}_{t,i}} + \epsilon)$ that multiplies its momentum
$\hat{m}_{t,i}$, therefore never exceeds $\eta/\epsilon$. A small $\epsilon$
trusts the variance estimate even when it is tiny; a large $\epsilon$ does
not. In the limit of large $\epsilon$ the denominator is $\epsilon$ for
every coordinate, and Adam becomes $\mathbf{x}_{t+1} = \mathbf{x}_t -
(\eta/\epsilon)\, \hat{\mathbf{m}}_t$. Since $\mathbf{m}_t$ is $1 - \beta_1$
times the momentum buffer of :numref:`sec_momentum`, this is SGD with
momentum $\beta = \beta_1$ and learning rate $(1 - \beta_1)\eta/\epsilon$,
up to the bias correction of the first steps. Values of $\epsilon$ between
the default $10^{-8}$ and this limit interpolate between the two methods,
which makes $\epsilon$ a tuning knob rather than a numerical detail.

## Decoupled Weight Decay
:label:`sec_adamw`

Adam is almost always trained with weight decay, which shrinks the weights
toward zero at every step (:numref:`sec_weight_decay`). Under SGD it makes
no difference whether the shrinkage is written as a penalty
$\frac{\lambda}{2}\|\mathbf{x}\|^2$ added to the loss or applied to the
weights directly. Under Adam it does: the gradient of the penalty passes
through the per-coordinate scaling, and the same $\lambda$ then shrinks
every coordinate by a different amount. AdamW
:cite:`Loshchilov.Hutter.2019` applies the decay outside the scaling and
restores one meaning for $\lambda$.

### Penalty and Preconditioner

Adding the penalty to the loss adds $\lambda \mathbf{x}_t$ to the
gradient. Under SGD the update then splits into a uniform shrinkage and an
ordinary gradient step:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{x}_t)
= (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t.
$$
:eqlabel:`eq_l2-is-decay`

Penalizing the loss *is* shrinking the weights, by the factor
$1 - \eta\lambda$ for every coordinate. Now feed the same penalized
gradient to Adam. Both moments are built from it, so the penalty passes
through the preconditioner too. If we treat $\sqrt{\hat{\mathbf{v}}_t}$ as
fixed, the shrinkage of coordinate $i$ in one step compares with that of
weight decay as follows:

$$
\underbrace{\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; x_{t,i}}_{\textrm{$\ell_2$ penalty through Adam}}
\qquad \textrm{versus} \qquad
\underbrace{\eta\,\lambda\; x_{t,i}\vphantom{\frac{\eta}{\sqrt{\hat{v}_{t,i}}}}}_{\textrm{weight decay}}.
$$
:eqlabel:`eq_coupled-decay`

The penalty is rescaled per coordinate by the same factor that rescales
the loss gradient: a parameter with large gradients (large $\hat{v}_i$) is
barely decayed, while a parameter with small gradients is decayed
strongly. Whatever $\lambda$ we
choose, Adam changes its effective value by coordinate and over time.
:numref:`subsec_mdl-decoupled-weight-decay` follows the exact recurrence
and shows two coordinates with the same $\lambda$ whose effective decay
rates differ by a factor of 100.

AdamW *decouples* the two: the loss gradient goes through the
preconditioner, and the decay does not,

$$
\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
$$
:eqlabel:`eq_adamw`

with $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ computed from the plain
gradient $\mathbf{g}_t$ as in :eqref:`eq_adam-moments`. The shrinkage is
the uniform factor of :eqref:`eq_l2-is-decay` again. When the learning
rate follows a schedule $\eta_t$, the decay should follow it, so that the
weights stop shrinking when learning stops. :citet:`Loshchilov.Hutter.2019`
multiply the decay by the schedule's multiplier; the library implementations
we use multiply it by the whole learning rate, as in :eqref:`eq_adamw`, and
we follow them.

### AdamW from Scratch

The decoupled update adds one operation to Adam: shrink the weights, then
take the Adam step.

```{.python .input #adamw-adamw-from-scratch-1}
%%tab pytorch
class AdamW(Adam):
    """Adam with decoupled weight decay: shrink the weights, then step."""
    def __init__(self, params, lr=1e-3, weight_decay=0.01):
        super().__init__(params, lr)
        self.defaults['weight_decay'] = weight_decay  # For later groups
        for group in self.param_groups:
            group.setdefault('weight_decay', weight_decay)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
        super().step()
```

```{.python .input #adamw-adamw-from-scratch-1}
%%tab jax
def adamw(learning_rate, weight_decay):
    return optax.chain(scale_by_adam(),
                       optax.add_decayed_weights(weight_decay),
                       optax.scale(-learning_rate))
```

:begin_tab:`pytorch`
`torch.optim.Adam` also accepts a `weight_decay` argument. It adds
$\lambda \mathbf{x}$ to the gradient before the moments are updated, which
is the coupled form on the left of :eqref:`eq_coupled-decay`.
`torch.optim.AdamW` is the decoupled form.
:end_tab:

:begin_tab:`jax`
`optax.add_decayed_weights` adds $\lambda \mathbf{x}$ to whatever reaches
it. Placed after `scale_by_adam`, the decay bypasses the preconditioner.
Swapping the first two links,
`optax.chain(optax.add_decayed_weights(wd), scale_by_adam(), ...)`, gives
the coupled form instead: the position of one link decides which of the two
updates we get.
:end_tab:

We check it against the library's AdamW as before, at $\lambda = 0.1$:

```{.python .input #adamw-adamw-from-scratch-2}
%%tab pytorch
compare(partial(AdamW, weight_decay=0.1),
        partial(torch.optim.AdamW, weight_decay=0.1), lr=0.01)
```

```{.python .input #adamw-adamw-from-scratch-2}
%%tab jax
compare(adamw(0.01, weight_decay=0.1),
        optax.adamw(0.01, weight_decay=0.1), lr=0.01)
```

The first step now also shrinks every weight by the factor $1 -
\eta\lambda$, so it is no longer exactly $\eta$, and the two
implementations again agree to within rounding.

### Coupled and Decoupled Decay Compared
:label:`subsec_adamw-grid`

How large is the difference on a real model? We train `TinyLM` for 1,000
steps at the learning rate $\eta = 0.003$ and the common choice $\lambda =
0.1$, once with the penalty inside Adam's gradient and once decoupled, and
also once without any decay for reference. All three runs start from the
same random initialization.

```{.python .input #adamw-one-number-two-meanings}
%%tab pytorch
seed = random.getrandbits(31)
curves = {}
for name, make_optimizer in [
        ('no decay', lambda ps: torch.optim.Adam(ps, lr=0.003)),
        ('Adam + ℓ2',
         lambda ps: torch.optim.Adam(ps, lr=0.003, weight_decay=0.1)),
        ('AdamW',
         lambda ps: torch.optim.AdamW(ps, lr=0.003, weight_decay=0.1))]:
    torch.manual_seed(seed)
    model = TinyLM(len(data.vocab))
    optimizer = make_optimizer(model.parameters())
    curves[name] = train_lm(model, data, optimizer, num_steps=1000)
    print(f'{name}: final loss {final_loss(curves[name]):.2f}')
d2l.plot(list(range(0, 1000, 25)), [smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

```{.python .input #adamw-one-number-two-meanings}
%%tab jax
def coupled(lr, wd):  # The penalty's gradient enters Adam's moments
    return optax.chain(optax.add_decayed_weights(wd), optax.adam(lr))

seed = random.getrandbits(32)
curves = {}
for name, tx in [('no decay', optax.adam(0.003)),
                 ('Adam + ℓ2', coupled(0.003, 0.1)),
                 ('AdamW', optax.adamw(0.003, weight_decay=0.1))]:
    model = TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    curves[name] = train_lm(model, data, optimizer, num_steps=1000)
    print(f'{name}: final loss {final_loss(curves[name]):.2f}')
d2l.plot(list(range(0, 1000, 25)), [smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

The decoupled run trains almost as if there were no decay. The coupled run
ends more than a nat higher, and after an early dip its loss rises again.
To read its level, we compare it with a model that ignores the context and
predicts every byte from its frequency in the training text. The loss of
that model is the entropy of the byte frequencies:

```{.python .input #adam-coupled-and-decoupled-decay-compared-2}
%%tab pytorch, jax
freq = np.bincount(data.train, minlength=256) / len(data.train)
freq = freq[freq > 0]
print(f'entropy of the byte frequencies: {-(freq * np.log(freq)).sum():.2f}')
```

The coupled run ends close to this value: it has lost what it learned
about the context in its first hundred steps.
:eqref:`eq_coupled-decay` explains the rise. As training makes progress the
loss gradients shrink, $\sqrt{\hat{\mathbf{v}}}$ shrinks with them, and the
penalty, divided by that small number, grows by orders of magnitude. Once
$\lambda \mathbf{x}$ dominates the gradient, Adam normalizes it like any
other gradient, so every coordinate shrinks by about $\eta$ per step
whatever the value of $\lambda$: past this point, changing $\lambda$ hardly
changes the run.

The coupled form needs a much smaller $\lambda$, which is easy to find. The
harder question is how the best $\lambda$ moves when the learning rate
changes: if it moves, the two settings must be searched together on a
grid instead of one after the other. We measure the held-out loss on a
grid of three learning rates and several values of $\lambda$ for each
form, from $10^{-4}$ to $10^{-2}$ for the coupled form and from $0.3$ to
$10$ for the decoupled one. In a run this short, decay helps only where
there is overfitting to prevent, and on WikiText-103 there is none: 800
steps read about 1% of the training text. For this grid we therefore make
the model overfit on purpose, on the kind of text that it would memorize:
the first 8,000 windows of *The Time Machine*, split into characters as in
:numref:`sec_text-sequence`. In 800 steps the model sees each window about
six times, and we measure the held-out loss on the 2,000 windows that
follow.

```{.python .input #adamw-a-grid-twice-1}
%%tab pytorch, jax
small = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                        num_train=8000, num_val=2000)
```

The held-out loss is the mean cross-entropy over the validation windows:

```{.python .input #adamw-decoupling-demonstrated-2}
%%tab pytorch
def val_loss(model, data):
    device = next(model.parameters()).device
    total, count = 0, 0
    with torch.inference_mode():
        for X, Y in data.val_dataloader():
            logits = model(X.to(device))
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     Y.to(device).reshape(-1),
                                     reduction='sum')
            count += Y.numel()
    return float(total) / count
```

```{.python .input #adamw-decoupling-demonstrated-2}
%%tab jax
@nnx.jit
def loss_sum(model, X, Y):
    logits = model(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).sum()

def val_loss(model, data):
    total, count = 0, 0
    for X, Y in data.val_dataloader():
        total += loss_sum(model, jnp.asarray(X), jnp.asarray(Y))
        count += Y.size
    return float(total) / count
```

The function `sweep` trains one model per pair $(\eta, \lambda)$ and
returns the table of held-out losses, one row per learning rate.

```{.python .input #adamw-a-grid-twice-2}
%%tab pytorch
lrs = [1e-3, 3e-3, 1e-2]
wds_coupled, wds_decoupled = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], [0.3, 1, 3, 10]

def sweep(make_optimizer, data, lrs, wds, seed, num_steps=800):
    table = []
    for lr in lrs:
        row = []
        for wd in wds:
            torch.manual_seed(seed)  # The same initialization everywhere
            model = TinyLM(len(data.vocab))
            train_lm(model, data, make_optimizer(model, lr, wd), num_steps)
            row.append(val_loss(model, data))
        table.append(row)
        print(f'lr {lr:g}: ' + ' '.join(f'{v:.2f}' for v in row))
    return table

seed = random.getrandbits(31)
coupled_tbl = sweep(lambda model, lr, wd: torch.optim.Adam(
    model.parameters(), lr, weight_decay=wd), small, lrs, wds_coupled, seed)
```

```{.python .input #adamw-a-grid-twice-2}
%%tab jax
lrs = [1e-3, 3e-3, 1e-2]
wds_coupled, wds_decoupled = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], [0.3, 1, 3, 10]

def sweep(make_tx, data, lrs, wds, seed, num_steps=800):
    table = []
    for lr in lrs:
        row = []
        for wd in wds:  # The same initialization everywhere
            model = TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
            optimizer = nnx.Optimizer(model, make_tx(lr, wd), wrt=nnx.Param)
            train_lm(model, data, optimizer, num_steps)
            row.append(val_loss(model, data))
        table.append(row)
        print(f'lr {lr:g}: ' + ' '.join(f'{v:.2f}' for v in row))
    return table

seed = random.getrandbits(32)
coupled_tbl = sweep(coupled, small, lrs, wds_coupled, seed)
```

```{.python .input #adamw-a-grid-twice-3}
%%tab pytorch
decoupled_tbl = sweep(lambda model, lr, wd: torch.optim.AdamW(
    model.parameters(), lr, weight_decay=wd), small, lrs, wds_decoupled,
    seed)
```

```{.python .input #adamw-a-grid-twice-3}
%%tab jax
decoupled_tbl = sweep(lambda lr, wd: optax.adamw(lr, weight_decay=wd),
                      small, lrs, wds_decoupled, seed)
```

We plot the held-out loss against $\lambda$, one curve per learning rate,
for the coupled form and then for the decoupled form, on a common vertical
scale.

```{.python .input #adamw-a-grid-twice-4}
%%tab pytorch, jax
values = [v for table in (coupled_tbl, decoupled_tbl)
          for row in table for v in row]
ylim = (min(values) - 0.1, max(values) + 0.1)
legend = [f'lr {lr:g}' for lr in lrs]
d2l.plot(wds_coupled, coupled_tbl, '$\\lambda$ (Adam + $\\ell_2$)',
         'held-out loss', legend=legend, ylim=ylim, xscale='log')
```

```{.python .input #adam-coupled-and-decoupled-decay-compared-1}
%%tab pytorch, jax
d2l.plot(wds_decoupled, decoupled_tbl, '$\\lambda$ (AdamW)',
         'held-out loss', legend=legend, ylim=ylim, xscale='log')
```

In this grid the AdamW curves have their minimum at about the same
$\lambda$, 3 or 10, for all three learning rates: changing $\eta$ moves a curve up
or down but hardly sideways. The coupled form works with values of
$\lambda$ about three orders of magnitude smaller. At its largest
$\lambda$ the coupled curves end at nearly the same loss for every
learning rate, the saturation described above. Below that, the best
$\lambda$ differs among the three learning rates. Tuned fully, the
two forms reach similar held-out losses. They differ in the shape of the
search: a column for AdamW, a whole grid for the coupled form.

A fixed point explains why the best $\lambda$ of AdamW stays put here.
Where the weights stop moving on average, the shrinkage in
:eqref:`eq_adamw` balances the Adam step, $\eta\lambda\,\mathbf{x} =
-\eta\,\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}} + \epsilon)$. The learning
rate cancels: $\lambda$ alone selects the solution near which the run
settles, and $\eta$ sets how fast the run gets there, through the factor
$1 - \eta\lambda$ per step. Near the best $\lambda$, every run of the grid
lasts longer than $1/(\eta\lambda)$ steps, so every run gets close. The
argument depends on the parameterization, in which both terms scale with
$\eta$, and on the regime, as the next subsection shows.

### Weight Decay in Large-Scale Training
:label:`subsec_wd-at-scale`

Our grid used weight decay against overfitting, the role it played in
:numref:`sec_weight_decay`, where it acts as a prior that pulls the model
toward small weights. Large language models are trained for about one pass
over data that they never see twice, so there is little overfitting to
fight, yet most published recipes still use $\lambda = 0.1$. The decay
does a different job there
:cite:`DAngelo.Andriushchenko.Varre.ea.2024,Kosson.Messmer.Jaggi.2024`.

The mechanism runs through the norms of the weights. A weight matrix whose
output passes through a normalization layer is *scale-invariant*:
multiplying it by a constant does not change the network's function,
because the normalization divides the constant out. Only the direction of
such a matrix matters, and an update of fixed size turns it by an angle
roughly proportional to the update divided by the matrix's norm. In a
pre-norm transformer this holds only approximately, since the residual
stream carries each block's output around the next normalization, but it
describes enough of the parameters to set the dynamics. Gradient noise
makes the norms grow, decay makes them shrink, and the two settle into an
equilibrium in which each weight turns by a roughly constant angle per step
:cite:`Kosson.Messmer.Jaggi.2024`. Through this equilibrium the product
$\eta\lambda$ sets the *effective* learning rate, the angle per step, and
sets it alike for all layers. This is a large part of why decay in large
language-model runs lowers the training loss instead of trading training
loss for held-out loss :cite:`DAngelo.Andriushchenko.Varre.ea.2024`. Small
weights also keep low-precision training in a safe numerical range.

With decay multiplied by the learning rate, each step multiplies the
weights by $1 - \eta\lambda$, so a contribution made $k$ steps ago has
shrunk by the factor $(1 - \eta\lambda)^k \approx e^{-\eta\lambda k}$. An
AdamW iterate is therefore approximately an average of its recent updates
over a horizon of $1/(\eta\lambda)$ steps: with $\eta = 0.003$ and $\lambda
= 0.1$, about 3,300 steps. Measured in passes over a dataset of $D$
examples with batch size $B$, the horizon is $\tau = B/(\eta\lambda D)$.
:citet:`Bergsma.Dey.Gosal.ea.2025b` found that the best $\tau$ stays
roughly constant as models grow at a fixed ratio of training tokens to
parameters, and follows a power law in that ratio otherwise, so that
$\lambda$ can be set from $B$, $D$, and the model size rather than tuned
again. The horizon also explains why $\eta$ and $\lambda$ act on the
equilibrium through their product.

Keeping the product fixed when $\eta$ changes is the opposite of what our
grid showed. The two results describe different regimes. A scale-invariant
matrix has no fixed point of the kind found in the previous subsection: its
gradient is orthogonal to it, so decay and updates balance only in the norm,
which the network ignores. What remains is the angle per step, set by
$\eta\lambda$. When most of the decayed weights are of this kind, the
quantity to keep when $\eta$ is retuned is the product, which is also how
:citet:`Loshchilov.Hutter.2019` parameterize the decay: they multiply it
by the schedule only, not by $\eta$, and on image classifiers with batch
normalization they found its best value nearly independent of the
learning rate. When decay mainly fights overfitting, as in our grid,
$\lambda$ itself selects the solution. The exercise on the fixed-product
ridge measures which of the two holds for `TinyLM`.

### Parameters Exempt from Decay
:label:`subsec_adamw-what-not-to-decay`

The scale-invariance argument covers only matrices whose output is
normalized, and standard practice decays only the matrices among the three
parameter groups of `TinyLM`. The other two groups are exempt for different
reasons. Biases and LayerNorm scales and shifts are few, and they set the
scale and offset of each feature directly, so shrinking them toward zero
distorts the function instead of controlling the dynamics. Biases were
already exempt in :numref:`sec_weight_decay`. Embedding rows are sparse: a
row receives a gradient only when its token occurs, but the decay acts at
every step, so the rows of rare tokens are decayed without any signal to
balance it. OLMo 2 traced a training instability to this: decay shrank the
embeddings until the LayerNorm after them, which divides by their norm,
amplified their gradients :cite:`OLMo.2025`.

The implementation uses two parameter groups, one decayed and one not, and
the split is the one we counted for `TinyLM`.

```{.python .input #adamw-what-not-to-decay-1}
%%tab pytorch
torch.manual_seed(lm_seed)  # The initialization of the comparison's runs
model = TinyLM(len(data.vocab))
decay = [p for name, p in model.named_parameters()
         if p.ndim == 2 and 'emb' not in name]
no_decay = [p for name, p in model.named_parameters()
            if p.ndim != 2 or 'emb' in name]
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0}], lr=best_adam)
n_decay = sum(p.numel() for p in decay)
n = n_decay + sum(p.numel() for p in no_decay)
print(f'decayed: {n_decay} of {n} parameters')
```

```{.python .input #adamw-what-not-to-decay-1}
%%tab jax
model = TinyLM(len(data.vocab), rngs=nnx.Rngs(lm_seed))  # As in the comparison

def decay_mask(params):  # True for the matrices, False for the rest
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

optimizer = nnx.Optimizer(
    model, optax.adamw(best_adam, weight_decay=0.1, mask=decay_mask),
    wrt=nnx.Param)
params = nnx.state(model, nnx.Param)
n_decay = sum(int(p.size) for p, m in zip(
    jax.tree.leaves(params), jax.tree.leaves(decay_mask(params))) if m)
n = sum(int(p.size) for p in jax.tree.leaves(params))
print(f'decayed: {n_decay} of {n} parameters')
```

:begin_tab:`pytorch`
`torch.optim.AdamW` takes a list of parameter groups, each a dictionary
with its own settings; settings not given in a group fall back to the
keyword arguments.
:end_tab:

:begin_tab:`jax`
`optax.adamw` takes a `mask`, a function that maps the parameter tree to a
tree of booleans; only the parameters marked `True` are decayed.
:end_tab:

About 90% of the parameters are decayed. We train this configuration for
2,000 steps at Adam's tuned learning rate, from the same initialization as
the runs of the language-model comparison, and compare it with Adam's
tuned run there, which used no decay:

```{.python .input #adamw-what-not-to-decay-2}
%%tab pytorch, jax
losses = train_lm(model, data, optimizer, num_steps=2000)
d2l.plot(list(range(0, 2000, 25)), [smooth(losses)],
         'step', 'training loss')
print(f'perplexity: AdamW {math.exp(final_loss(losses)):.2f}, '
      f'Adam without decay {math.exp(final_loss(adam_lm[best_adam])):.2f}')
```

AdamW ends within a few percent of the run without decay, in either
direction. In
2,000 steps the model reads about 3% of the training text and does not
overfit, so the decay, whose horizon $1/(\eta\lambda)$ is longer than the
run, has nothing to prevent and cannot help.
This is the configuration in which Adam is used today: AdamW, decay on the
matrices only, and $\lambda$ chosen together with the schedule. What
remains is its cost in memory.

## Optimizer-State Memory
:label:`subsec_adamw-memory`

Adam and AdamW store two numbers per parameter, $m$ and $v$, on top of the
weights and the gradients. At the size of `TinyLM` that is invisible; at
the size of a language model it decides what fits on a device. We count
the bytes per parameter for three common setups and multiply by the
parameter count $n$ of `TinyLM`, which `d2l.num_params` returns.

In full fp32 training, a parameter costs 4 bytes each for the weight, the
gradient, $m$, and $v$: 16 bytes, half of them for $m$ and $v$. Mixed
precision does not reduce this. The forward and backward passes use bf16
weights and gradients, 2 bytes each, but stable training keeps an fp32
*master copy* of the weights for the optimizer to update, so the total is
again 16 bytes. Many setups also keep an fp32 copy of the gradients to
accumulate them over several minibatches, which brings the total to 20
(:numref:`sec_training_systems`).

```{.python .input #adamw-optimizer-state-and-memory}
%%tab pytorch, jax
n = d2l.num_params(model)  # The TinyLM just trained
setups = {'fp32 weights, grads, m, v': (4, 4, 8),
          'bf16 weights + grads, fp32 master, m, v': (2, 2, 12),
          'the same + fp32 gradient accumulator': (2, 6, 12)}
print(f'{"":<42}{"B/param":>8}{"TinyLM":>10}{"7B model":>10}')
for name, (w, g, s) in setups.items():
    per = w + g + s
    print(f'{name:<42}{per:>8}{n * per / 1e6:>8.1f}MB'
          f'{7e9 * per / 1e9:>8.0f}GB')
```

For `TinyLM` the total is a few megabytes. The same arithmetic for a model
with 7 billion parameters gives 112 to 140 GB before a single activation
is stored, more than an 80-GB accelerator holds. Of the 20 bytes, 12 are
the fp32 master copy, $m$, and $v$, which only the optimizer reads. This is
why optimizer state is the first target when memory runs out. It can be
stored in fewer bytes or split across the devices of a data-parallel group
instead of copied to each, the systems techniques of
:numref:`chap_performance`.

## Summary

A single learning rate must suit the stiffest direction of the loss. Adam
gives each coordinate its own step, scaled by an estimate of that
coordinate's gradient size. The estimate is AdaGrad's running sum of
squared gradients, made to forget by RMSProp's exponential average. Adam
combines it with momentum and corrects both averages exactly for their
start at zero. On a small transformer language model and on a CNN of the
same size, tuned Adam reaches tuned SGD's final training loss about 1.3
to 2.5 times as fast and ends a few hundredths to about two tenths of a nat
lower; on the CNN the test accuracies agree to within a point or two. On
large transformers
the gap is far wider, and the best current explanations point to gradient
scales that differ across coordinates, from heavy-tailed data and from
dissimilar blocks of the architecture. Under Adam an $\ell_2$ penalty is
not weight decay, because its gradient passes through the per-coordinate
scaling. AdamW applies the decay outside it, which gives $\lambda$ the
same meaning for every coordinate. At scale the product $\eta\lambda$
sets the effective learning rate of normalized layers. Adam stores two
extra numbers per parameter, which in mixed precision bring a parameter
to 16 to 20 bytes. What Adam does not remove is the choice of $\eta$ over
time, the subject of :numref:`sec_scheduler`.

## Exercises

1. [code] **Rotating the coordinates.** Per-coordinate methods are
   axis-aligned. Rotating the valley `f_valley` by $45^\circ$ gives
   $$f(\mathbf{x}) = 0.05 (x_1 + x_2)^2 + (x_1 - x_2)^2,$$
   whose Hessian has the same eigenvalues as that of `f_valley`, $0.2$
   and $4$.
    1. Write the gradient of the rotated valley as a function like
       `valley_grad`, and pass it as `f_grad` to `adagrad_2d`,
       `rmsprop_2d`, and the `gd_2d` of :numref:`sec_gd`. On both
       valleys, run AdaGrad at $\eta = 0.4$ and $\eta = 2$, RMSProp at
       $\eta = 0.4$, and gradient descent at $\eta = 0.4$ with
       `d2l.train_2d` and `steps=2000`. For each run, record the first
       step at which $\|\mathbf{x}\| \leq 10^{-3}$, or that no step within
       the 2,000 reaches it.
    1. How much of the adaptive methods' advantage over gradient descent
       survives the rotation? Explain the answer from the Hessian of the
       rotated valley.

1. **Bias correction by initialization.** Instead of the zero
   initialization and the corrections in :eqref:`eq_adam-moments`,
   initialize $\mathbf{m}_1 = \mathbf{g}_1$ and
   $\mathbf{v}_1 = \mathbf{g}_1^2$ and use the uncorrected averages from
   then on.
    1. Show that the two schemes coincide at $t = 1$ and derive the weights
       that each assigns to $\mathbf{g}_1, \ldots, \mathbf{g}_t$ for
       $t > 1$.
    1. What is lost relative to the exact correction, and for which range
       of $t$ does the difference matter?

1. [extended] **Frequent and rare tokens.** Following
   :citet:`Kunstner.Yadav.Milligan.ea.2024`, split the 256 byte values of
   `data` into a frequent and a rare group by their share of the training
   text, for instance at 0.1%. Under SGD with momentum and under Adam, each
   at the best learning rate of `sgd_lm` and `adam_lm`, record every 100
   steps the loss of `TinyLM` on the targets of each group, measured on a
   fixed set of validation windows.
    1. Which optimizer makes progress on the rare group, and does the gap
       of :numref:`subsec_adam-where-wins` come mainly from the rare group,
       from the frequent group, or from both?
    1. Repeat the measurement with subword tokens, among which rare
       tokens make up a larger share of the text. Train a
       `d2l.BPETokenizer` with 1,024 tokens, the GPT-2 pattern, and no
       special tokens (:numref:`sec_text-sequence`) on the
       first million bytes of the training text, encode the first ten
       million bytes and the validation split with it, store the results
       in `data.train` and `data.val`, and train `TinyLM` with a
       vocabulary of 1,024. Compare the two splits.

1. [code] **Per-group learning rates for SGD.** ● The three parameter
   groups of `TinyLM` in :numref:`subsec_tinylm`, embeddings, matrices, and
   vectors, share one learning rate under SGD with momentum.
    1. Give each group its own learning rate. Starting from the best rate
       of `sgd_lm`, scale one group's rate by each factor in
       $\{1/10, 1/3, 3, 10\}$ while holding the other two fixed, and keep
       the best factor for each group. Then train once with the three best
       factors combined.
    1. How much of the gap between the best runs of `sgd_lm` and `adam_lm`
       does per-group SGD close? What does the result say about whether
       Adam's advantage on this model comes from scale differences between
       the three groups or from differences among the coordinates within
       each group?

1. [code] **Two repairs of the variance estimate.** Adam's second-moment
   update can be written as
   $$\mathbf{v}_t = \mathbf{v}_{t-1} + (1 - \beta_2)(\mathbf{g}_t^2 - \mathbf{v}_{t-1}),$$
   which shrinks $\mathbf{v}_t$ whenever $\mathbf{g}_t^2 < \mathbf{v}_{t-1}$.
   Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` limits how fast it can shrink
   by replacing the increment with
   $$(1 - \beta_2)\, \mathbf{g}_t^2 \odot \mathrm{sgn}(\mathbf{g}_t^2 - \mathbf{v}_{t-1}).$$
   AMSGrad :cite:`Reddi.Kale.Kumar.2019` carries the running maximum
   $$\hat{\mathbf{v}}_t^{\max} = \max(\hat{\mathbf{v}}_{t-1}^{\max}, \hat{\mathbf{v}}_t)$$
   and uses it in place of $\hat{\mathbf{v}}_t$ in :eqref:`eq_adam-update`.
   (The function `online` below takes the maximum of the uncorrected
   $\mathbf{v}_t$ instead, as the original paper does; implement the
   version written here.)
    1. Implement both by changing the from-scratch Adam of this section,
       and check each against a hand computation of its first three steps
       on a scalar gradient sequence such as $(1, 0.1, 0.1)$.
    1. The function `online` of :numref:`subsec_mdl-per-coordinate` runs
       Adam and AMSGrad on the losses :eqref:`eq_mdl-opt-reddi`, on which
       Adam converges to the worst point $x = +1$. Add the Yogi update to
       it, run it at $C = 4$, and report where its iterate ends. Explain the
       outcome from how far the $v_t$ of Adam and of Yogi can fall during the
       two $-1$ gradients of each period.
   :exlabel:`ex_adam-two-repairs-variance-estimate`

1. [code] **The $\epsilon$ ceiling.** Sweep
   $\epsilon \in \{10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}, 1\}$ for Adam on
   `TinyLM` at the best learning rate of `adam_lm`. Explain the trend at
   both ends of the sweep with the ceiling $\eta/\epsilon$ on each
   coordinate's learning rate of :numref:`subsec_adam-variance-estimate`,
   and identify the $\epsilon$
   from which the run resembles SGD with momentum, as in `sgd_lm`, at
   learning rate $(1 - \beta_1)\eta/\epsilon$.

1. [code] **Saturation of coupled decay.** The experiment of
   :numref:`subsec_mdl-decoupled-weight-decay` starts two parameters at
   $x_i = 1$ and gives them pure-noise loss gradients
   $g_i \sim \mathcal{N}(0, \sigma_i^2)$ with $\sigma = (10, 0.1)$. At
   $\eta = 10^{-3}$ and $\lambda = 0.1$, Adam with an $\ell_2$ penalty
   decays them at the rates $\eta\lambda/\sigma_i$ that
   :eqref:`eq_coupled-decay` predicts.
    1. In the opposite limit $\sigma_i = 0$, the penalized gradient is
       $\lambda x_{t,i}$ alone. Show that for $\epsilon = 0$ the trajectory
       of $x_{t,i}$ under Adam with an $\ell_2$ penalty does not depend on
       $\lambda > 0$.
    1. Extend the experiment to
       $\sigma_i \in \{10, 1, 0.1, 0.01, 0.001\}$ at $\eta = 10^{-3}$ and
       record $|x_i|$ after 1,000 steps of Adam with an $\ell_2$ penalty
       for each $\lambda \in \{0.1, 1, 10\}$. Which coordinates no longer
       respond to $\lambda$, and how does the transition between the two
       regimes depend on $\sigma_i$ and $\lambda$?

1. [code] **The fixed-product ridge.** ● Both library implementations
   used in this section multiply the decay by the full learning rate, so
   the per-step shrinkage is $s = \eta\lambda$. :numref:`subsec_wd-at-scale`
   argues that at scale $\eta$ and $\lambda$ act through this product,
   which sets an averaging horizon of $1/s$ steps.
   :citet:`Loshchilov.Hutter.2019` instead scale the decay by the schedule
   multiplier only, so that under a constant schedule their decay
   coefficient is $s$ itself.
    1. Each column of a grid can fix $s$ instead of $\lambda$. Use
       `decoupled_tbl` to predict where the best entry of each row falls on
       the shrinkage axis `[1e-3, 3e-3, 1e-2, 3e-2]`. Then rerun the
       decoupled grid of :numref:`subsec_adamw-grid` with `sweep` on this
       axis, passing the decay coefficient to the optimizer as `wd / lr`.
       Does the best column stay put?
    1. How constant is the held-out loss down the column
       $s = 3 \times 10^{-3}$? Reconcile the answer with the argument of
       :numref:`subsec_wd-at-scale`, and state which quantity, $\lambda$ or
       $s$, the decay axis of a grid should carry when the learning rate is
       retuned in this setting.

1. [code] **Rotational equilibrium.** ● Following
   :citet:`Kosson.Messmer.Jaggi.2024`, let $\mathbf{x} \in \mathbb{R}^d$
   be a scale-invariant weight vector in the sense of
   :numref:`subsec_wd-at-scale`, trained by AdamW with $\beta_1 = 0$:
   $$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{u}_t,
   \qquad \mathbf{u}_t = \frac{\mathbf{g}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.$$
    1. Show that the loss gradient of a scale-invariant weight vector is
       orthogonal to $\mathbf{x}$. Assume from here on that $\mathbf{u}_t$
       is orthogonal to $\mathbf{x}_t$ as well and that
       $\|\mathbf{u}_t\|^2 \approx d$. Derive, to first order in
       $\eta\lambda$, the equilibrium norm at which decay balances the
       growth caused by the updates.
    1. Derive the angle through which $\mathbf{x}$ turns per step at
       equilibrium. How do the equilibrium norm and the angle each depend
       on $\eta$ and $\lambda$? Relate the result to the claim of
       :numref:`subsec_wd-at-scale` that $\eta$ and $\lambda$ act through
       their product.
    1. Check both predictions in NumPy with $d = 1{,}000$ over 20,000
       steps from a random unit vector, using pure-noise gradients
       projected orthogonal to $\mathbf{x}_t$, at $(\eta, \lambda) \in
       \{(10^{-3}, 0.1), (3 \times 10^{-3}, 0.1), (10^{-3}, 0.3)\}$.

1. **Activations and optimizer state.** Extend the accounting of
   :numref:`subsec_adamw-memory` beyond the `setups` table.
    1. For `TinyLM` with batch size $B$ and sequence length $T$, estimate
       the bf16 activation memory that must be stored for the backward pass
       (per block: the two normalization outputs, the attention inputs and
       outputs, and the MLP hidden layer). At what batch size do
       activations overtake the optimizer state?
    1. Count the floating-point operations of one AdamW update per
       parameter and compare them with the roughly $6BT$ operations per
       parameter of a forward and backward pass over one batch. At which
       batch size would the optimizer step cost one percent of a training
       step?

    *Adapted from Stanford CS336,
    [Assignment 1](https://github.com/stanford-cs336/spring2024-assignment1-basics),
    Problem adamwAccounting.*

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

Adam and AdamW<br>**per-coordinate learning rates · bias correction · Adam against SGD · decoupled weight decay · optimizer memory**
:::
:::

::: {.slide title="One Learning Rate Leaves x₁ at −0.94 After 20 Steps"}
[The problem]{.kicker}

The valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$: curvatures $0.2$ and
$4$, start $(-5, -2)$.

- Gradient descent is stable only for $\eta < 2/4 = 0.5$.
- At $\eta = 0.4$, $x_1$ shrinks by $1 - 0.4 \cdot 0.2 = 0.92$ per step.
- After 20 steps: $x_1 = -5 \cdot 0.92^{20} \approx -0.94$.
- A rate of $1/\textrm{curvature}$ per coordinate would finish in one
  step.

::: {.d2l-note .warn}
Curvatures are unknown in a network. Can each coordinate estimate its own
step from its own gradients?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute AdaGrad, RMSProp, and Adam steps on the valley by hand;
- show that bias correction makes Adam's first step exactly $\eta$;
- read a fair comparison of Adam and SGD on a transformer and a CNN;
- tell an $\ell_2$ penalty from decoupled weight decay under Adam;
- count the bytes of optimizer state per parameter.

**Running example:** the valley above, then `TinyLM`, a 0.47M-parameter
transformer on the bytes of WikiText-103.
:::

::: {.slide title="One Step Size per Coordinate"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Landscapes | curvature and noise slow training |
| Gradient Descent | step size against curvature |
| Stochastic and Minibatch GD | sampled gradients, noise ball, batching |
| Momentum | averaging over time |
| **Adam and AdamW** | **a step size per coordinate, decoupled decay** |
| Schedules | warmup and decay |
| Muon | orthogonalized updates for matrices |
| Batch Size | how many examples per step |
| Scaling Up | learning rates across widths |
| Practice | clipping, averaging, fair comparisons |
:::

::: {.col .narrow}
- **Uses:** the valley, momentum $\beta$, weight decay, the byte-level
  language model.
- **Skips:** convergence proofs (appendix), second-order methods.
:::
:::
:::

::: {.slide title="AdaGrad Divides Each Step by Its Coordinate's Gradient History"}
[Per-coordinate learning rates]{.kicker}

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$

(elementwise, $\mathbf{s}_0 = \mathbf{0}$)

- **Rare features:** a coordinate that is rarely touched keeps a large
  step for when its feature appears.
- **Ill-conditioning:** large past gradients stand in for large curvature,
  so $\sqrt{\mathbf{s}_t}$ is a diagonal preconditioner.

::: {.d2l-note .rule}
Each coordinate gets its own clock: its step decays with its own activity,
not with $t$.
:::
:::

::: {.slide title="AdaGrad Stalls at η = 0.4 and Converges at η = 2"}
::: {.cols}
::: {.col}
$\eta = 0.4$

@!adam-per-coordinate-learning-rates-1
:::

::: {.col}
$\eta = 2$ (gradient descent diverges)

@!adam-per-coordinate-learning-rates-2
:::
:::

- Step 1 moves every coordinate by exactly $\eta$: $\mathbf{s}_1 = \mathbf{g}_1^2$.
- Then about $\eta/\sqrt{t}$: 20 steps cover at most $7.6\,\eta = 3.0 < 5$.
:::

::: {.slide title="RMSProp Forgets: Its Average Spans 1/(1 − β₂) Steps"}
::: {.cols .vc}
::: {.col .fig}
@!adam-rmsprop-forgetting-on-purpose-1
:::

::: {.col}
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\, \mathbf{g}_t^2$$

- $\beta_2 = 0.9$: a window of 10 steps.
- Steps no longer shrink by construction.
- Step 1 overshoots: $\mathbf{v}_1 = 0.1\,\mathbf{g}_1^2$, so each
  coordinate moves $\eta/\sqrt{0.1} \approx 3.2\,\eta$.
:::
:::
:::

::: {.slide title="Adam Averages the Gradient and Its Square, Then Corrects Both"}
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t,
\qquad
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},\quad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t},\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,
\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

- Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$:
  direction over ~10 steps, scale over ~1,000.
- Language-model recipes often use $\beta_2 = 0.95$: a window of 20 steps.
- $\mathbf{m}_t$ is momentum, averaged instead of summed.
:::

::: {.slide title="Bias Correction Makes Adam's First Step Exactly η"}
[Worked example]{.kicker}

Step 1 from $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$:

- $\mathbf{m}_1 = 0.1\,\mathbf{g}_1$, $\mathbf{v}_1 = 0.001\,\mathbf{g}_1^2$.
- Uncorrected: $\mathbf{m}_1/\sqrt{\mathbf{v}_1} = (0.1/0.0316)\,
  \mathrm{sign}(\mathbf{g}_1) \approx 3.2\,\mathrm{sign}(\mathbf{g}_1)$.
- Corrected: $\hat{\mathbf{m}}_1 = \mathbf{g}_1$,
  $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so the step is
  $\eta\,\mathrm{sign}(\mathbf{g}_1)$.
- After 10 steps $\mathbf{v}_t$ still holds only $1 - 0.999^{10} \approx 1\%$
  of its stationary value.

::: {.d2l-note .rule}
The weights of $\mathbf{v}_t$ sum to $1 - \beta_2^t$; dividing by that sum
removes the deficit at every $t$.
:::
:::

::: {.slide title="Check: How Large Is the First Uncorrected Step at β₂ = 0.99?"}
[Check]{.kicker}

Adam without bias correction, $\beta_1 = 0.9$, $\beta_2 = 0.99$. How far
does the first step move a coordinate, in units of $\eta$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $(1-\beta_1)/\sqrt{1-\beta_2} = 0.1/0.1 = 1$: exactly $\eta$,
by coincidence. From step 2 on, $\mathbf{m}_t$ fills its window ten times
faster than $\mathbf{v}_t$, and the uncorrected steps grow too large.
:::
:::
:::

::: {.slide title="Adam from Scratch Matches the Library to Rounding Error"}
Two copies of a linear model, same data, 100 steps: our Adam against the
framework's.

@!adam-adam-both-moments-debiased-2

- Every weight moves by $\eta$ on step 1, as the bias correction promises.
- Written as a framework optimizer, the from-scratch version trains any
  model; afterwards we call the library's.
:::

::: {.slide title="TinyLM and FashionCNN: Two Models of About 0.45M Parameters"}
[Two models]{.kicker}

| | `TinyLM` | `FashionCNN` |
|---|---|---|
| task | next byte, WikiText-103 | 10 classes, Fashion-MNIST |
| layers | 2 pre-norm transformer blocks, width 128 | 2 conv-pool stages, 2 linear |
| parameters | 470,784 | 421,642 |
| minibatch | 64 windows of 64 bytes | 256 images |

`TinyLM`'s groups: embeddings 9%, matrices 90%, vectors 1%. A run reads
about 3% of the 270M training bytes and sees almost no text twice. Both
models are in `d2l` and train with `d2l.train_lm`.

::: {.d2l-note}
Protocol: same random initialization, 2,000 steps, constant rate, four
rates per method spaced by about 3×; "tuned" is the lowest final loss.
:::
:::

::: {.slide title="On Both Models, Adam Reaches SGD's Final Loss 1.3–2.5× Faster"}
::: {.cols}
::: {.col}
`TinyLM`: gap about 0.03–0.2 nats

@!adam-the-race-on-the-language-model-3
:::

::: {.col}
`FashionCNN`: gap about 0.03–0.1 nats

@!adam-the-same-race-on-a-cnn-4
:::
:::

- SGD trains best at or just below its largest stable rate; Adam's best
  rate lies inside its grid.
:::

::: {.slide title="Check: What Does a 0.1-Nat Gap Do to Perplexity?"}
[Check]{.kicker}

Adam's final loss is 0.1 nats below SGD's. By what factor is Adam's
perplexity lower?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Perplexity is $e^{\textrm{loss}}$, so the ratio is
$e^{-0.1} \approx 0.90$: about 10% lower.
:::
:::
:::

::: {.slide title="With β₁ = β₂ = 0, Adam Is Sign Descent"}
$$\beta_1 = \beta_2 = 0,\ \epsilon = 0:\qquad
\eta\,\frac{\mathbf{g}_t}{|\mathbf{g}_t|} = \eta\,\mathrm{sign}(\mathbf{g}_t)$$

Adam equalizes steps across coordinates whose gradients differ by orders
of magnitude. Three findings on why this helps transformers:

- The gap persists with full-batch gradients, so it is not noise.
- Rare tokens: gradient descent stalls on rare classes; Adam does not.
- Transformer blocks have very different curvatures; CNN blocks look
  alike.

[Kunstner et al. 2023, 2024; Zhang et al. 2024]{.small}
:::

::: {.slide title="ε Caps Every Coordinate's Learning Rate at η/ε"}
[Limits]{.kicker}

- Too small an estimate $\hat{\mathbf{v}}_t$ gives too large a step. A
  rare large gradient after many small ones hits exactly this case.
- Reddi et al. build a convex problem on which Adam converges to the worst
  point; AMSGrad divides by the running maximum of $\hat{\mathbf{v}}_t$.
- $\epsilon$ is a floor on the denominator, so a coordinate's learning
  rate $\eta/(\sqrt{\hat{v}_{t,i}} + \epsilon)$ is at most $\eta/\epsilon$.
- Large $\epsilon$: $\mathbf{x}_{t+1} \approx \mathbf{x}_t -
  (\eta/\epsilon)\hat{\mathbf{m}}_t$, SGD with momentum $\beta_1$ at rate
  $(1-\beta_1)\eta/\epsilon$.
:::

::: {.slide title="Under Adam, an ℓ2 Penalty Is Not Weight Decay"}
[Decoupled weight decay]{.kicker}

SGD: the penalty $\frac{\lambda}{2}\|\mathbf{x}\|^2$ *is* decay,

$$\mathbf{x}_t - \eta(\mathbf{g}_t + \lambda\mathbf{x}_t) = (1-\eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t.$$

Adam: the penalty passes through the preconditioner (schematically),

$$\underbrace{\frac{\eta\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\, x_{t,i}}_{\textrm{$\ell_2$ through Adam}}
\qquad\textrm{vs.}\qquad
\underbrace{\eta\lambda\, x_{t,i}}_{\textrm{weight decay}}$$

- Large gradients: almost no decay. Small gradients: strong decay.
:::

::: {.slide title="AdamW Shrinks the Weights, Then Takes the Adam Step"}
$$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
\qquad \hat{\mathbf{m}}_t, \hat{\mathbf{v}}_t \textrm{ from } \mathbf{g}_t \textrm{ alone}$$

- Coupled: add $\lambda\mathbf{x}_t$ to the gradient, *then* update
  the moments.
- Decoupled: update the moments from $\mathbf{g}_t$, and shrink the
  weights separately.
- From scratch this is one line on top of Adam, and it matches the
  library's AdamW to rounding error.
:::

::: {.slide title="At λ = 0.1, Coupled ℓ2 Costs Over a Nat; AdamW Tracks No Decay"}
@!adamw-one-number-two-meanings

- Coupled ends near 3.2 nats, the loss of byte frequencies alone: the
  early context is lost.
- Progress shrinks $\sqrt{\hat{\mathbf{v}}}$ until $\lambda\mathbf{x}$
  dominates; Adam then normalizes it, and every coordinate shrinks by
  about $\eta$ per step, whatever $\lambda$ is.
:::

::: {.slide title="Check: What Decay Does λ = 0.1 Give Where √v̂ = 10⁻³?"}
[Check]{.kicker}

Adam with an $\ell_2$ penalty, $\eta = 0.003$, $\lambda = 0.1$, on a
coordinate whose loss gradients have $\sqrt{\hat{v}_{t,i}} = 10^{-3}$.
What fraction of $x_{t,i}$ does the penalty remove per step, by the
formula with $\sqrt{\hat{v}}$ held fixed? And by AdamW?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta\lambda/\sqrt{\hat{v}} = 0.003 \cdot 0.1 / 10^{-3} = 0.3$,
30% per step, a thousand times AdamW's $\eta\lambda = 3 \times 10^{-4}$.
For any weight with $|x| > 0.01$, though, $\lambda|x|$ exceeds the loss
gradient, so the penalty dominates $\hat{v}$ itself and Adam normalizes
it: the coordinate moves about $\eta$ per step toward zero. That is the
saturation.
:::
:::
:::

::: {.slide title="In This Grid, AdamW's Best λ Barely Moves with η"}
Held-out loss after 800 steps on 8,000 windows of *The Time Machine*
(about six passes, overfitting on purpose), one curve per $\eta$:

::: {.cols}
::: {.col}
@!adamw-a-grid-twice-4
:::

::: {.col}
@!adam-coupled-and-decoupled-decay-compared-1
:::
:::

- AdamW: $\eta$ moves each curve up or down, hardly sideways. At the
  fixed point $\lambda\mathbf{x} = -\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}} + \epsilon)$,
  $\eta$ cancels.
- Coupled: $\lambda$ about three orders of magnitude smaller, saturated
  at the largest values.
:::

::: {.slide title="At Scale, Decay Sets a Horizon of 1/(ηλ) Steps"}
- A matrix followed by normalization is scale-invariant. Noise grows its
  norm, decay shrinks it: a constant turning angle per step, set by
  $\eta\lambda$.
- Horizon $1/(\eta\lambda)$: at $\eta = 0.003$, $\lambda = 0.1$, about
  3,300 steps; in passes, $\tau = B/(\eta\lambda D)$.
- So at scale, keep $\eta\lambda$ when $\eta$ changes; our short,
  overfitting grid kept $\lambda$.
- Decay the matrices only (90% of `TinyLM`). Exempt embeddings (sparse
  gradients, decay every step), biases, and LayerNorm.

[D'Angelo et al. 2024; Kosson et al. 2024; Bergsma et al. 2025]{.small}
:::

::: {.slide title="Adam Costs 16 to 20 Bytes per Parameter"}
[Memory]{.kicker}

@!adamw-optimizer-state-and-memory

- 7B parameters: 112 to 140 GB before any activation.
- 12 of the 20 bytes (fp32 master, $m$, $v$) are read only by the
  optimizer.
- Store less (Adafactor: row and column sums of $v$; 8-bit states) or
  shard the state across devices (ZeRO).
:::

::: {.slide title="Check: Does a 1B-Parameter Model Fit a 24 GB GPU?"}
[Check]{.kicker}

Training a 1B-parameter model with AdamW in fp32. Do the weights,
gradients, and optimizer state fit on a 24 GB GPU?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $16 \times 10^9$ bytes = 16 GB: yes, with 8 GB left for
activations. Half of the 16 GB is $m$ and $v$.
:::
:::
:::

::: {.slide title="Wrap-up"}
- AdaGrad: one clock per coordinate. On the valley it stalls at
  $\eta = 0.4$ and converges at $\eta = 2$.
- RMSProp forgets; Adam adds momentum and bias correction, so step 1 is
  exactly $\eta$.
- Tuned Adam reaches tuned SGD's final loss about 1.3–2.5× faster on
  both models; the gap is about 0.03–0.2 nats on `TinyLM`, about
  0.03–0.1 on `FashionCNN`.
- AdamW decays outside the preconditioner: one $\lambda$ for every
  coordinate. At scale, $\eta\lambda$ sets the effective learning rate.
- Cost: 16–20 bytes per parameter.

**Next:** how to change $\eta$ during training, in Schedules.
:::
