```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('pytorch', 'tensorflow', 'jax')
```

# Selective State Space Models
:label:`sec_mamba`

:begin_tab:`mxnet`
This section is intentionally not implemented in MXNet. Its models are built
on a parallel associative scan, a primitive that the MXNet 2.0 wheel used by
this book does not provide. See the PyTorch, TensorFlow, and JAX tabs.
:end_tab:

The previous section ended on a confession. Everything we gained by
linearizing the recurrence, parallel training by scan, stability by
construction, provably good memory, came at the price of *time invariance*:
the S4D applies the same dynamics at every step, so what it remembers is
decided before it ever sees the input. Its convolution kernel weights the
past by *position*, never by *content*. The gated cells of
:numref:`sec_lstm` had the opposite profile: their forget gates read the
data as it streamed past and decided, token by token, what deserved space
in the state, but their nonlinear recurrence trained sequentially. This
section closes the loop. We make the step size, and with it the dynamics,
a *function of the input*, following :citet:`Gu.Dao.2023`, and discover
that this one change re-derives the forget gate a third time while keeping
the parallel scan. The result, packaged into a residual block called
*Mamba*, returned recurrence to serious competition with transformers on
language for the first time since the LSTM's heyday, and it sets up the
honest question that hands this chapter over to the next: what can a
fixed-size state never do, no matter how cleverly it is updated?

```{.python .input #mamba-selective-state-space-models-1}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import numpy as np
import time
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #mamba-selective-state-space-models-1}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import numpy as np
import time
import tensorflow as tf
```

```{.python .input #mamba-selective-state-space-models-1}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import numpy as np
import optax
import time
```

Everything in this section is built on the associative scan of
:numref:`subsec_parallel-scans`, which evaluates the affine recurrence
$\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$ in
logarithmic depth.

:begin_tab:`pytorch`
That section saved the scan in the `d2l` library, so we simply pick it
back up.
:end_tab:

:begin_tab:`jax`
As in that section, JAX provides the scan as a primitive; we only restate
the two-line wrapper that applies the affine combine
:eqref:`eq_scan_combine`.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow has no associative-scan primitive, so we restate the short
doubling-scan helper from that section.
:end_tab:

```{.python .input #mamba-selective-state-space-models-2}
%%tab pytorch
associative_scan = d2l.associative_scan  # Saved in the previous section
```

```{.python .input #mamba-selective-state-space-models-2}
%%tab tensorflow
def associative_scan(a, b):
    """Parallel prefix scan for h_t = a_t * h_{t-1} + b_t with h_0 = 0."""
    step = 1
    while step < b.shape[0]:  # ceil(log2 T) rounds of combines
        a_prev, b_prev = a[:-step], b[:-step]
        a, b = (tf.concat([a[:step], a_prev * a[step:]], 0),
                tf.concat([b[:step], a[step:] * b_prev + b[step:]], 0))
        step *= 2
    return b
```

```{.python .input #mamba-selective-state-space-models-2}
%%tab jax
def scan_combine(prev, cur):
    a_prev, b_prev = prev
    a_cur, b_cur = cur
    return a_prev * a_cur, a_cur * b_prev + b_cur

def associative_scan(a, b):
    """Parallel prefix scan for h_t = a_t * h_{t-1} + b_t with h_0 = 0."""
    return jax.lax.associative_scan(scan_combine, (a, b))[1]
```

## The Selectivity Problem
:label:`subsec_selectivity`

What does it mean, concretely, for a model to be content-blind? The LTI
convolution view :eqref:`eq_ssm_kernel` makes it precise: an S4D's output
is $y_t = \sum_k \bar{K}_k u_{t-k}$, and the kernel $\bar{\mathbf{K}}$ is
computed from the model's parameters alone. Whether the token ten steps
ago was the key fact of the paragraph or a comma, its influence on the
present is the same fixed number $\bar{K}_{10}$. A gated RNN would consult
its input before deciding; the LTI system cannot, because *deciding based
on the input* is exactly the time-variance it gave up.

### A Task That Defeats Time Invariance

To turn this from an observation into a measurement we borrow the
*selective copying* task that :citet:`Gu.Dao.2023` used to motivate
selectivity (:numref:`fig_selective_copy`). Each input is a long stretch
of filler tokens in which a few *marked* symbols are scattered at random
positions; after the sequence, the model is prompted with query slots and
must reproduce the marked symbols, in order, ignoring everything else. A
solution has to do two content-dependent things: store a token *because
it is a symbol rather than filler*, and keep count of how many symbols it
has seen so far. Neither is expressible with position-based weights: the
same positions hold noise in one example and payload in the next.

![The selective copying task. A few marked symbols (color) are scattered among filler tokens (grey) at positions that change from example to example; prompted by query slots, the model must emit the symbols in order. Any fixed position-based kernel is defeated by design, since content decides what matters.](../img/mdl-modernrnn-selective-copy.svg)
:label:`fig_selective_copy`

This toy is not idle. Its language-scale counterpart is *associative
recall*: retrieving a value mentioned pages ago the next time its key
appears ("Mrs. Watchett ... later: Mrs. ___"). Transformer interpretability
work calls the circuits that do this *induction heads*, and
:citet:`Arora.Eyuboglu.Timalsina.ea.2024` showed that a small synthetic
recall benchmark of this kind predicts most of the language-modeling gap
between attention and efficient recurrent models. A sequence model that
cannot selectively copy has no business writing prose.

We generate the task synthetically: token $0$ is filler, token $1$ marks
the query slots, and the symbols occupy ids $2$ through $9$. The targets
are the eight-way symbol classes at the query positions.

```{.python .input #mamba-a-task-that-defeats-time-invariance}
%%tab pytorch, jax, tensorflow
def selective_copy(num_seqs, num_steps, num_marked, num_symbols, seed=42):
    """Sequences of filler (0) with num_marked symbols at random positions,
    followed by num_marked query slots (1); targets are the symbols."""
    rng = np.random.default_rng(seed)
    X = np.zeros((num_seqs, num_steps + num_marked), dtype=np.int64)
    X[:, num_steps:] = 1                          # Query slots
    Y = rng.integers(2, 2 + num_symbols, (num_seqs, num_marked))
    pos = np.argsort(rng.random((num_seqs, num_steps)), axis=1)
    pos = np.sort(pos[:, :num_marked], axis=1)    # Marked positions, in order
    np.put_along_axis(X, pos, Y, axis=1)
    return X, Y - 2                               # Classes 0..num_symbols-1

class SelectiveCopy(d2l.DataModule):
    def __init__(self, num_train=8192, num_val=1024, batch_size=128,
                 num_steps=256, num_marked=4, num_symbols=8):
        super().__init__()
        self.save_hyperparameters()
        X, Y = selective_copy(num_train + num_val, num_steps, num_marked,
                              num_symbols)
        self.X, self.Y = d2l.tensor(X), d2l.tensor(Y)

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.Y], train, idx)

copy_data = SelectiveCopy()
```

### An LTI Baseline and a Gated One

The prosecution calls two witnesses from earlier sections. The LTI
witness is the S4D stack exactly as we assembled it in
:numref:`subsec_s4d`; we restate the layer and its residual block
verbatim.

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-1}
%%tab pytorch
class S4D(nn.Module):
    """A diagonal state space layer: one SSM per feature channel."""
    def __init__(self, num_hiddens, num_states=4, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H, N = num_hiddens, num_states
        self.log_a = nn.Parameter(
            torch.log(torch.arange(1., N + 1)).repeat(H, 1))
        self.log_dt = nn.Parameter(
            torch.rand(H, 1) * math.log(dt_max / dt_min) + math.log(dt_min))
        self.C = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.ones(H))

    def forward(self, u):                    # (num_steps, batch, num_hiddens)
        a = -torch.exp(self.log_a)                    # (H, N), Re(a) < 0
        a_bar = torch.exp(torch.exp(self.log_dt) * a)
        b_bar = (a_bar - 1) / a                       # ZOH with B = 1
        a_elems = a_bar.expand(u.shape[0], 1, -1, -1) # Same at every step
        b_elems = b_bar * u.unsqueeze(-1)             # (T, batch, H, N)
        x = associative_scan(a_elems, b_elems)
        return (x * self.C).sum(-1) + self.D * u

class S4DBlock(nn.Module):
    def __init__(self, num_hiddens, num_states):
        super().__init__()
        self.ln1 = nn.LayerNorm(num_hiddens)
        self.ssm = S4D(num_hiddens, num_states)
        self.ln2 = nn.LayerNorm(num_hiddens)
        self.W_v = nn.Linear(num_hiddens, 2 * num_hiddens)
        self.W_g = nn.Linear(num_hiddens, 2 * num_hiddens)
        self.W_o = nn.Linear(2 * num_hiddens, num_hiddens)

    def forward(self, X):
        X = X + self.ssm(self.ln1(X))
        Y = self.ln2(X)
        return X + self.W_o(self.W_v(Y) * torch.sigmoid(self.W_g(Y)))
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-1}
%%tab tensorflow
class S4D(tf.keras.layers.Layer):
    """A diagonal state space layer: one SSM per feature channel."""
    def __init__(self, num_hiddens, num_states=4, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H, N = num_hiddens, num_states
        # Keras 3 tracks keras.Variable (not plain tf.Variable) attributes
        self.log_a = tf.keras.Variable(tf.tile(
            tf.math.log(tf.range(1., N + 1.))[None], (H, 1)))
        self.log_dt = tf.keras.Variable(
            tf.random.uniform((H, 1)) * math.log(dt_max / dt_min)
            + math.log(dt_min))
        self.C = tf.keras.Variable(tf.random.normal((H, N)) / math.sqrt(N))
        self.D = tf.keras.Variable(tf.ones(H))

    def call(self, u):                       # (num_steps, batch, num_hiddens)
        # Recompute the (T, batch, H, N) coefficients in the backward pass
        # instead of storing the whole scan (the store-vs-recompute trade)
        @tf.recompute_grad
        def ssm(u):
            a = -tf.exp(self.log_a)                   # (H, N), Re(a) < 0
            a_bar = tf.exp(tf.exp(self.log_dt) * a)
            b_bar = (a_bar - 1) / a                   # ZOH with B = 1
            a_elems = tf.tile(a_bar[None, None], (u.shape[0], 1, 1, 1))
            b_elems = b_bar * tf.expand_dims(u, -1)   # (T, batch, H, N)
            x = associative_scan(a_elems, b_elems)
            return tf.reduce_sum(x * self.C, -1) + self.D * u
        return ssm(u)

class S4DBlock(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, num_states):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ssm = S4D(num_hiddens, num_states)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.W_v = tf.keras.layers.Dense(2 * num_hiddens)
        self.W_g = tf.keras.layers.Dense(2 * num_hiddens)
        self.W_o = tf.keras.layers.Dense(num_hiddens)

    def call(self, X):
        X = X + self.ssm(self.ln1(X))
        Y = self.ln2(X)
        return X + self.W_o(self.W_v(Y) * tf.sigmoid(self.W_g(Y)))
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-1}
%%tab jax
class S4D(nnx.Module):
    """A diagonal state space layer: one SSM per feature channel."""
    def __init__(self, num_hiddens, num_states=4, dt_min=0.001, dt_max=0.1,
                 rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        H, N = num_hiddens, num_states
        self.log_a = nnx.Param(jnp.tile(jnp.log(jnp.arange(1., N + 1)),
                                        (H, 1)))
        self.log_dt = nnx.Param(
            rngs.params.uniform((H, 1)) * math.log(dt_max / dt_min)
            + math.log(dt_min))
        self.C = nnx.Param(rngs.params.normal((H, N)) / math.sqrt(N))
        self.D = nnx.Param(jnp.ones(H))

    def __call__(self, u):                   # (num_steps, batch, num_hiddens)
        a = -jnp.exp(self.log_a.value)                # (H, N), Re(a) < 0
        a_bar = jnp.exp(jnp.exp(self.log_dt.value) * a)
        b_bar = (a_bar - 1) / a                       # ZOH with B = 1
        a_elems = jnp.broadcast_to(                   # Same at every step
            a_bar[None, None], (u.shape[0], 1, *a_bar.shape))
        b_elems = b_bar * u[..., None]                # (T, batch, H, N)
        x = associative_scan(a_elems, b_elems)
        return (x * self.C).sum(-1) + self.D * u

class S4DBlock(nnx.Module):
    def __init__(self, num_hiddens, num_states, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.ln1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.ssm = S4D(num_hiddens, num_states, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.W_v = nnx.Linear(num_hiddens, 2 * num_hiddens, rngs=rngs)
        self.W_g = nnx.Linear(num_hiddens, 2 * num_hiddens, rngs=rngs)
        self.W_o = nnx.Linear(2 * num_hiddens, num_hiddens, rngs=rngs)

    def __call__(self, X):
        X = X + self.ssm(self.ln1(X))
        Y = self.ln2(X)
        return X + self.W_o(self.W_v(Y) * jax.nn.sigmoid(self.W_g(Y)))
```

The gated witness is the LSTM of :numref:`sec_lstm`, via the concise
`d2l.LSTM` layer. Both plug into the same harness: embed the tokens, run
the encoder, and classify each of the final query positions. As in
:numref:`subsec_s4d` we train with Adam, and we clip gradients to norm 1,
without which the LSTM destabilizes on these sequence lengths.

:begin_tab:`jax`
One implementation note: we embed tokens with a one-hot matrix product
rather than an embedding lookup. A lookup's gradient is a scatter-add,
and on this task every batch pushes 33,000 colliding updates into a
ten-row table, which XLA executes so slowly that it dominates the entire
training step; as a dense product, both passes are ordinary matrix
multiplications. Hold this thought: this section later argues that how a
computation maps onto hardware can matter as much as what it computes.
:end_tab:

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-2}
%%tab pytorch
class CopyModel(d2l.Classifier):
    """Read a token sequence; predict the symbols at the query slots."""
    def __init__(self, encoder, num_hiddens, vocab_size=10, num_marked=4,
                 num_symbols=8, lr=3e-3):
        super().__init__()
        self.save_hyperparameters()
        self.emb = nn.Embedding(vocab_size, num_hiddens)
        self.head = nn.LazyLinear(num_symbols)

    def forward(self, X):
        Y = self.encoder(self.emb(X.T))                 # (T, batch, hiddens)
        Y = Y[0] if isinstance(Y, tuple) else Y         # RNNs return a state
        return self.head(Y[-self.num_marked:]).movedim(0, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-2}
%%tab tensorflow
class CopyModel(d2l.Classifier):
    """Read a token sequence; predict the symbols at the query slots."""
    def __init__(self, encoder, num_hiddens, vocab_size=10, num_marked=4,
                 num_symbols=8, lr=3e-3):
        super().__init__()
        self.save_hyperparameters()
        self.emb = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.head = tf.keras.layers.Dense(num_symbols)

    def forward(self, X):
        Y = self.encoder(self.emb(tf.transpose(X)))     # (T, batch, hiddens)
        Y = Y[0] if isinstance(Y, tuple) else Y         # RNNs return a state
        return tf.transpose(self.head(Y[-self.num_marked:]), (1, 0, 2))

    def configure_optimizers(self):
        return tf.keras.optimizers.Adam(float(self.lr))
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-2}
%%tab jax
class CopyModel(d2l.Classifier):
    """Read a token sequence; predict the symbols at the query slots."""
    def __init__(self, encoder, num_hiddens, vocab_size=10, num_marked=4,
                 num_symbols=8, num_features=None, lr=3e-3, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'rngs'])
        rngs = nnx.Rngs(1) if rngs is None else rngs
        num_features = num_hiddens if num_features is None else num_features
        self.encoder = encoder
        # Embed by one-hot matmul: on a table this tiny (ten rows), the
        # scatter-add in an embedding lookup's gradient makes 33,000 colliding
        # updates to ten rows and serializes the whole training step on GPU;
        # as a dense product, forward and backward are ordinary matmuls
        self.emb = nnx.Linear(vocab_size, num_hiddens, use_bias=False,
                              rngs=rngs)
        self.head = nnx.Linear(num_features, num_symbols, rngs=rngs)

    def forward(self, X):
        E = self.emb(jax.nn.one_hot(X.T, self.vocab_size))
        Y = self.encoder(E)                             # (T, batch, hiddens)
        Y = Y[0] if isinstance(Y, tuple) else Y         # RNNs return a state
        return self.head(Y[-self.num_marked:]).swapaxes(0, 1)

    def configure_optimizers(self):
        return optax.adam(self.lr)
```

We give each model the same budget and record its validation accuracy
curve; `copy_curves` collects them, and a third contender will join in
:numref:`subsec_mamba-block`. The two encoders are sized to roughly 30,000
parameters each, the same pairing that :numref:`subsec_s4d` used for
sequential image classification, where the S4D *won*.

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-3}
%%tab pytorch
copy_curves = {}

def train_copy(name, model, data, epochs=32):
    trainer = d2l.Trainer(max_epochs=epochs, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)
    pts = model.board.data['val_acc']
    copy_curves[name] = ([p.x for p in pts], [float(p.y) for p in pts])
    print(f'{name}: final validation accuracy {copy_curves[name][1][-1]:.3f}')

s4d = CopyModel(nn.Sequential(*[S4DBlock(48, 4) for _ in range(2)]),
                num_hiddens=48)
train_copy('S4D', s4d, copy_data)
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-3}
%%tab tensorflow
copy_curves = {}

def train_copy(name, model, data, epochs=32):
    trainer = d2l.Trainer(max_epochs=epochs, gradient_clip_val=1)
    trainer.fit(model, data)
    pts = model.board.data['val_acc']
    copy_curves[name] = ([p.x for p in pts], [float(p.y) for p in pts])
    print(f'{name}: final validation accuracy {copy_curves[name][1][-1]:.3f}')

with d2l.try_gpu():
    s4d = CopyModel(tf.keras.Sequential([S4DBlock(48, 4) for _ in range(2)]),
                    num_hiddens=48)
    train_copy('S4D', s4d, copy_data)
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-3}
%%tab jax
copy_curves = {}

def train_copy(name, model, data, epochs=32):
    trainer = d2l.Trainer(max_epochs=epochs, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)
    pts = model.board.data['val_acc']
    copy_curves[name] = ([p.x for p in pts], [float(p.y) for p in pts])
    print(f'{name}: final validation accuracy {copy_curves[name][1][-1]:.3f}')

s4d = CopyModel(nnx.Sequential(*[S4DBlock(48, 4) for _ in range(2)]),
                num_hiddens=48)
train_copy('S4D', s4d, copy_data)
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-4}
%%tab pytorch, jax
if tab.selected('pytorch'):
    lstm = CopyModel(d2l.LSTM(num_inputs=48, num_hiddens=64), num_hiddens=48)
if tab.selected('jax'):
    lstm = CopyModel(d2l.LSTM(num_inputs=48, num_hiddens=64), num_hiddens=48,
                     num_features=64)
train_copy('LSTM', lstm, copy_data)
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-4}
%%tab tensorflow
with d2l.try_gpu():
    lstm = CopyModel(d2l.LSTM(num_inputs=48, num_hiddens=64), num_hiddens=48)
    train_copy('LSTM', lstm, copy_data)
```

```{.python .input #mamba-an-lti-baseline-and-a-gated-one-5}
%%tab pytorch, jax, tensorflow
names = list(copy_curves)
d2l.plot([copy_curves[n][0] for n in names],
         [copy_curves[n][1] for n in names], 'epoch',
         'validation accuracy', legend=names, figsize=(5, 3))
```

The curves separate exactly as the theory says they must. Both models
start at chance (one symbol in eight). The LSTM climbs toward a complete
solution, and in our runs ends the budget at or near perfect accuracy
(how fast depends on each framework's initialization defaults, the same
effect :numref:`subsec_s4d` dissected): its input and forget gates are
functions of the data, so "store this, it is a symbol" is a computation
it can learn, given enough epochs. The S4D crawls. Its pointwise
nonlinear blocks can learn to suppress filler locally, which earns it
partial credit well above chance, but the time-invariant state cannot
cleanly align "the third symbol I saw" with "the third query slot" when
the spacing between symbols changes from example to example, and it ends
the same budget far from a solution. This is the first task in the book
where the *older* architecture is simply the right tool. That should
feel like a cliffhanger: the LSTM's advantage is precisely the
input-dependence that :numref:`sec_ssm` deleted for the sake of the
scan. Can we put it back without giving up parallel training?

## Selective State Space Models
:label:`subsec_selective-ssm`

### Making the Dynamics Look at the Data

Recall from :numref:`subsec_zoh` where the S4D's dynamics come from: a
continuous system $(\mathbf{A}, \mathbf{B}, \mathbf{C})$ and a step size
$\Delta$, discretized by the zero-order hold into per-step coefficients.
Everything downstream of that box stays fixed; the *selective* state
space model of :citet:`Gu.Dao.2023` changes one thing. The step size, the
input matrix, and the read-out are no longer constants but functions of
the current input $\mathbf{u}_t \in \mathbb{R}^H$:

$$
\boldsymbol{\Delta}_t = \textrm{softplus}(\mathbf{u}_t \mathbf{W}_{\Delta} + \mathbf{b}_{\Delta}),
\qquad
\mathbf{B}_t = \mathbf{u}_t \mathbf{W}_{\textrm{B}},
\qquad
\mathbf{C}_t = \mathbf{u}_t \mathbf{W}_{\textrm{C}},
$$
:eqlabel:`eq_selective_heads`

with $\boldsymbol{\Delta}_t \in \mathbb{R}^H$ holding one step size per
channel (softplus keeps it positive) and
$\mathbf{B}_t, \mathbf{C}_t \in \mathbb{R}^N$ shared across channels. The
state matrix $\mathbf{A} = \textrm{diag}(a_1, \ldots, a_N)$ keeps its
fixed S4D parameterization; it sets the *menu* of decay rates, while
$\boldsymbol{\Delta}_t$ decides, per token, how far along that menu to
step. Discretizing exactly as before, channel $h$ of the layer now obeys

$$
\mathbf{x}_t = e^{\Delta_{t,h} \mathbf{a}} \odot \mathbf{x}_{t-1}
+ \Delta_{t,h}\, u_{t,h}\, \mathbf{B}_t,
\qquad
y_{t,h} = \mathbf{C}_t^\top \mathbf{x}_t + d_h\, u_{t,h},
$$
:eqlabel:`eq_selective_ssm`

where $\mathbf{x}_t \in \mathbb{R}^N$ is that channel's state. (Mamba
simplifies the zero-order hold on the input path to the first-order rule
$\bar{\mathbf{B}}_t = \Delta_{t,h} \mathbf{B}_t$; the exponential on the
state path, which controls stability, is kept exact.)

Now watch what the boxed correspondence of :numref:`subsec_zoh` does with
an input-dependent $\Delta$. There, a small $\Delta$ froze the state and
ignored the input; a large $\Delta$ flushed the state and admitted the
input. Make $\Delta$ a function of $\mathbf{u}_t$ and the model can
*choose per token*: filler should produce
$\Delta_t \approx 0$ (state glides through untouched, input contributes
nothing), while a marked symbol should produce a large $\Delta_t$ (reset
toward the new content). That is a forget gate and an input gate, fused
into one scalar, acting on a linear state. We have now derived the gate
three times, once by engineering (:numref:`sec_lstm`), once from
numerical integration (:numref:`subsec_zoh`), and now from the demand
that a linear recurrence be able to ignore what does not matter.
$\mathbf{B}_t$ and $\mathbf{C}_t$ extend the same courtesy to *where*
input enters the state and *which* state coordinates are read out; an
exercise asks how much they add over selectivity in $\Delta$ alone.

### What Selectivity Costs, and What Survives

There is no free lunch: with time-varying coefficients the model is no
longer LTI, and the convolutional view of :numref:`subsec_ssm-conv` dies
on the spot. There is no fixed kernel $\bar{\mathbf{K}}$ to materialize
and no FFT shortcut; of the three views in :numref:`fig_ssm_views`, only
the recurrence survives. This is why we built the scan rather than the
FFT in :numref:`sec_ssm`. The recurrence
:eqref:`eq_selective_ssm` is still an *affine* map of the state, just
with per-step coefficients, and the associative combine
:eqref:`eq_scan_combine` never assumed those coefficients were constant.
The same `associative_scan`, called with tensors whose leading axis now
varies per step, evaluates the selective recurrence in the same
logarithmic depth. Seeing is believing, one more time: a sequential loop
with time-varying decays against the scan.

```{.python .input #mamba-what-selectivity-costs-and-what-survives-1}
%%tab pytorch
num_steps, num_states = 100, 4
a_t = torch.rand(num_steps, num_states)      # Per-step decays in (0, 1)
b_t = torch.randn(num_steps, num_states)     # Per-step inputs
h, ys = torch.zeros(num_states), []
for t in range(num_steps):
    h = a_t[t] * h + b_t[t]
    ys.append(h)
err = (associative_scan(a_t, b_t) - torch.stack(ys)).abs().max()
print(f'time-varying scan vs loop: {float(err):.2e}')
```

```{.python .input #mamba-what-selectivity-costs-and-what-survives-1}
%%tab tensorflow
num_steps, num_states = 100, 4
a_t = tf.random.uniform((num_steps, num_states))  # Per-step decays in (0, 1)
b_t = tf.random.normal((num_steps, num_states))   # Per-step inputs
h, ys = tf.zeros(num_states), []
for t in range(num_steps):
    h = a_t[t] * h + b_t[t]
    ys.append(h)
err = tf.reduce_max(tf.abs(associative_scan(a_t, b_t) - tf.stack(ys)))
print(f'time-varying scan vs loop: {float(err):.2e}')
```

```{.python .input #mamba-what-selectivity-costs-and-what-survives-1}
%%tab jax
num_steps, num_states = 100, 4
key1, key2 = jax.random.split(d2l.get_key())
a_t = jax.random.uniform(key1, (num_steps, num_states))  # Decays in (0, 1)
b_t = jax.random.normal(key2, (num_steps, num_states))   # Per-step inputs
h, ys = jnp.zeros(num_states), []
for t in range(num_steps):
    h = a_t[t] * h + b_t[t]
    ys.append(h)
err = jnp.abs(associative_scan(a_t, b_t) - jnp.stack(ys)).max()
print(f'time-varying scan vs loop: {float(err):.2e}')
```

The layer below packages :eqref:`eq_selective_heads` and
:eqref:`eq_selective_ssm`. Next to the `S4D` class it differs in three
places: $\Delta$, $\mathbf{B}$, $\mathbf{C}$ are computed from `u` by
small linear heads rather than stored as parameters; the $\Delta$ head is
factored through rank $H/16$ (Mamba's `dt_rank`), which keeps its
parameter count negligible; and its bias is initialized so that
$\textrm{softplus}$ of it reproduces the log-uniform step sizes of
:numref:`subsec_s4d`, so an *untrained* selective layer starts as approximately
the multi-timescale S4D and must learn to deviate. The scan call is
unchanged, except that the decay tensor now genuinely spans
`(num_steps, batch, H, N)` instead of broadcasting one value.

```{.python .input #mamba-what-selectivity-costs-and-what-survives-2}
%%tab pytorch
class SelectiveSSM(nn.Module):
    """A diagonal SSM whose step size, input matrix, and read-out are
    functions of the input (Gu & Dao, 2023)."""
    def __init__(self, num_hiddens, num_states=4, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H, N, R = num_hiddens, num_states, max(2, num_hiddens // 16)
        self.log_a = nn.Parameter(
            torch.log(torch.arange(1., N + 1)).repeat(H, 1))
        self.W_dt = nn.Sequential(nn.Linear(H, R), nn.Linear(R, H, bias=False))
        dt = torch.exp(torch.rand(H) * math.log(dt_max / dt_min)
                       + math.log(dt_min))
        self.b_dt = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.W_B = nn.Linear(H, N, bias=False)
        self.W_C = nn.Linear(H, N, bias=False)
        self.D = nn.Parameter(torch.ones(H))

    def forward(self, u):                    # (num_steps, batch, num_hiddens)
        a = -torch.exp(self.log_a)                    # (H, N), Re(a) < 0
        dt = F.softplus(self.W_dt(u) + self.b_dt)     # (T, batch, H)
        B, C = self.W_B(u), self.W_C(u)               # (T, batch, N)
        a_bar = torch.exp(dt.unsqueeze(-1) * a)       # (T, batch, H, N)
        b_bar = (dt * u).unsqueeze(-1) * B.unsqueeze(-2)
        x = associative_scan(a_bar, b_bar)
        return (x * C.unsqueeze(-2)).sum(-1) + self.D * u
```

```{.python .input #mamba-what-selectivity-costs-and-what-survives-2}
%%tab tensorflow
class SelectiveSSM(tf.keras.layers.Layer):
    """A diagonal SSM whose step size, input matrix, and read-out are
    functions of the input (Gu & Dao, 2023)."""
    def __init__(self, num_hiddens, num_states=4, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H, N, R = num_hiddens, num_states, max(2, num_hiddens // 16)
        self.log_a = tf.keras.Variable(tf.tile(
            tf.math.log(tf.range(1., N + 1.))[None], (H, 1)))
        self.W_dt = tf.keras.Sequential([
            tf.keras.layers.Dense(R), tf.keras.layers.Dense(H, use_bias=False)])
        dt = tf.exp(tf.random.uniform((H,)) * math.log(dt_max / dt_min)
                    + math.log(dt_min))
        self.b_dt = tf.keras.Variable(dt + tf.math.log(-tf.math.expm1(-dt)))
        self.W_B = tf.keras.layers.Dense(num_states, use_bias=False)
        self.W_C = tf.keras.layers.Dense(num_states, use_bias=False)
        self.D = tf.keras.Variable(tf.ones(H))

    def call(self, u):                       # (num_steps, batch, num_hiddens)
        # Recompute the (T, batch, H, N) coefficients in the backward pass
        # instead of storing the whole scan (the store-vs-recompute trade)
        @tf.recompute_grad
        def ssm(u):
            a = -tf.exp(self.log_a)                   # (H, N), Re(a) < 0
            dt = tf.math.softplus(self.W_dt(u) + self.b_dt)   # (T, batch, H)
            B, C = self.W_B(u), self.W_C(u)           # (T, batch, N)
            a_bar = tf.exp(tf.expand_dims(dt, -1) * a)    # (T, batch, H, N)
            b_bar = tf.expand_dims(dt * u, -1) * tf.expand_dims(B, -2)
            x = associative_scan(a_bar, b_bar)
            return tf.reduce_sum(x * tf.expand_dims(C, -2), -1) + self.D * u
        return ssm(u)
```

```{.python .input #mamba-what-selectivity-costs-and-what-survives-2}
%%tab jax
class SelectiveSSM(nnx.Module):
    """A diagonal SSM whose step size, input matrix, and read-out are
    functions of the input (Gu & Dao, 2023)."""
    def __init__(self, num_hiddens, num_states=4, dt_min=0.001, dt_max=0.1,
                 rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        H, N, R = num_hiddens, num_states, max(2, num_hiddens // 16)
        self.log_a = nnx.Param(jnp.tile(jnp.log(jnp.arange(1., N + 1)),
                                        (H, 1)))
        self.W_dt = nnx.Sequential(
            nnx.Linear(H, R, rngs=rngs),
            nnx.Linear(R, H, use_bias=False, rngs=rngs))
        dt = jnp.exp(rngs.params.uniform((H,)) * math.log(dt_max / dt_min)
                     + math.log(dt_min))
        self.b_dt = nnx.Param(dt + jnp.log(-jnp.expm1(-dt)))
        self.W_B = nnx.Linear(H, N, use_bias=False, rngs=rngs)
        self.W_C = nnx.Linear(H, N, use_bias=False, rngs=rngs)
        self.D = nnx.Param(jnp.ones(H))

    def __call__(self, u):                   # (num_steps, batch, num_hiddens)
        a = -jnp.exp(self.log_a[...])                 # (H, N), Re(a) < 0
        dt = jax.nn.softplus(self.W_dt(u) + self.b_dt)    # (T, batch, H)
        B, C = self.W_B(u), self.W_C(u)               # (T, batch, N)
        a_bar = jnp.exp(dt[..., None] * a)            # (T, batch, H, N)
        b_bar = (dt * u)[..., None] * B[..., None, :]
        x = associative_scan(a_bar, b_bar)
        return (x * C[..., None, :]).sum(-1) + self.D * u
```

One prose remark on engineering, because it explains why this
architecture arrived in 2023 rather than 2020. Our implementation
materializes the `(num_steps, batch, H, N)` coefficient tensors in GPU
memory, which is fine at textbook scale and ruinous at model scale: the
computation is trivially cheap per element, so its speed is set entirely
by memory traffic. The Mamba authors' kernel never materializes those
tensors. It fuses discretization, scan, and read-out into a single pass
that keeps intermediates in fast on-chip memory, and during
backpropagation *recomputes* them from the small inputs instead of
storing them, trading a little arithmetic for a lot of bandwidth, the
same store-versus-recompute trade we met in :numref:`sec_bptt`. None of
this changes what is computed. It changed whether the architecture was
*worth* computing, and it is half the reason the paper's title contains
the word "hardware-aware".

## The Mamba Block
:label:`subsec_mamba-block`

The selective SSM mixes information across time. Like the S4D it needs a
scaffold around it that mixes channels and supplies nonlinearity, and
Mamba's block, :numref:`fig_mamba_block`, differs instructively from the
S4D block of :numref:`fig_ssm_block`. Instead of alternating a sequence
layer with a separate gated MLP, Mamba fuses the two: one linear
projection widens the input from $d$ to an expanded $2d$ and forks it
into two branches. The main branch is convolved with a short *causal*
convolution (width 4; a cheap way to let each token see a few immediate
predecessors before deciding its dynamics), passed through a SiLU
activation, and fed to the selective SSM. The other branch, after its own
SiLU, multiplies the SSM's output elementwise, one final gate, echoing
:numref:`sec_lstm` yet again, so that even the read-out is
content-controlled. A linear projection maps the expanded width back to
$d$, and the whole thing sits inside the usual pre-norm residual. Where a
transformer alternates attention blocks with MLP blocks, a Mamba language
model is simply this one homogeneous block stacked $L$ times; the full
comparison with attention must wait until we have built attention.

![The Mamba block. An input projection widens $d$ to $2d$ and forks: the main branch runs a short causal convolution, a SiLU, and the selective SSM, whose step size, input and read-out matrices are functions of its input; the gate branch applies a SiLU and multiplies the SSM output elementwise. An output projection returns to width $d$ inside a pre-norm residual.](../img/mdl-modernrnn-mamba-block.svg)
:label:`fig_mamba_block`

With `SelectiveSSM` in hand the block is a dozen lines, and a language
model is the block stacked plus the embedding and head that
`d2l.RNNLM` of :numref:`sec_rnn-scratch` already provides. The stack
exposes the `(inputs, state)` calling convention of our recurrent cells,
so it drops into the same scaffold as every other model in this chapter;
like the S4D, it trains with Adam.

```{.python .input #mamba-the-mamba-block}
%%tab pytorch
class MambaBlock(nn.Module):
    """Conv + SiLU + selective SSM, gated, inside a pre-norm residual."""
    def __init__(self, num_hiddens, num_states=4, expand=2, conv_width=4,
                 dropout=0):
        super().__init__()
        d = expand * num_hiddens
        self.ln = nn.LayerNorm(num_hiddens)
        self.W_in = nn.Linear(num_hiddens, 2 * d)
        self.conv = nn.Conv1d(d, d, conv_width, groups=d,
                              padding=conv_width - 1)
        self.ssm = SelectiveSSM(d, num_states)
        self.W_out = nn.Linear(d, num_hiddens)
        self.drop = nn.Dropout(dropout)

    def forward(self, X):                    # (num_steps, batch, num_hiddens)
        u, gate = self.W_in(self.ln(X)).chunk(2, -1)
        u = self.conv(u.permute(1, 2, 0))[..., :X.shape[0]]  # Causal: trim
        y = self.ssm(F.silu(u.permute(2, 0, 1)))
        return X + self.drop(self.W_out(y * F.silu(gate)))

class Mamba(d2l.Module):
    """A stack of Mamba blocks with the recurrent-cell interface."""
    def __init__(self, num_inputs, num_blocks=2, num_states=4, dropout=0):
        super().__init__()
        self.save_hyperparameters()
        self.num_hiddens = num_inputs                 # Output width, for heads
        self.blocks = nn.Sequential(*[
            MambaBlock(num_inputs, num_states, dropout=dropout)
            for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(num_inputs)

    def forward(self, X, state=None):
        return self.ln(self.blocks(X)), None
```

```{.python .input #mamba-the-mamba-block}
%%tab tensorflow
class MambaBlock(tf.keras.layers.Layer):
    """Conv + SiLU + selective SSM, gated, inside a pre-norm residual."""
    def __init__(self, num_hiddens, num_states=4, expand=2, conv_width=4,
                 dropout=0):
        super().__init__()
        d = expand * num_hiddens
        self.ln = tf.keras.layers.LayerNormalization()
        self.W_in = tf.keras.layers.Dense(2 * d)
        self.conv = tf.keras.layers.Conv1D(d, conv_width, groups=d,
                                           padding='causal')
        self.ssm = SelectiveSSM(d, num_states)
        self.W_out = tf.keras.layers.Dense(num_hiddens)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, X):                       # (num_steps, batch, num_hiddens)
        u, gate = tf.split(self.W_in(self.ln(X)), 2, axis=-1)
        u = tf.transpose(self.conv(tf.transpose(u, (1, 0, 2))), (1, 0, 2))
        y = self.ssm(tf.nn.silu(u))
        return X + self.drop(self.W_out(y * tf.nn.silu(gate)))

class Mamba(d2l.Module):
    """A stack of Mamba blocks with the recurrent-cell interface."""
    def __init__(self, num_inputs, num_blocks=2, num_states=4, dropout=0):
        super().__init__()
        self.save_hyperparameters()
        self.num_hiddens = num_inputs                 # Output width, for heads
        self.blocks = tf.keras.Sequential(
            [MambaBlock(num_inputs, num_states, dropout=dropout)
             for _ in range(num_blocks)])
        self.ln = tf.keras.layers.LayerNormalization()

    def forward(self, X, state=None):
        return self.ln(self.blocks(X)), None
```

```{.python .input #mamba-the-mamba-block}
%%tab jax
class MambaBlock(nnx.Module):
    """Conv + SiLU + selective SSM, gated, inside a pre-norm residual."""
    def __init__(self, num_hiddens, num_states=4, expand=2, conv_width=4,
                 dropout=0, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        d = expand * num_hiddens
        self.ln = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.W_in = nnx.Linear(num_hiddens, 2 * d, rngs=rngs)
        self.conv = nnx.Conv(d, d, kernel_size=(conv_width,),
                             feature_group_count=d, padding='CAUSAL',
                             rngs=rngs)
        self.ssm = SelectiveSSM(d, num_states, rngs=rngs)
        self.W_out = nnx.Linear(d, num_hiddens, rngs=rngs)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, X):                   # (num_steps, batch, num_hiddens)
        u, gate = jnp.split(self.W_in(self.ln(X)), 2, axis=-1)
        u = jnp.swapaxes(self.conv(jnp.swapaxes(u, 0, 1)), 0, 1)
        y = self.ssm(jax.nn.silu(u))
        return X + self.drop(self.W_out(y * jax.nn.silu(gate)))

class Mamba(nnx.Module):
    """A stack of Mamba blocks with the recurrent-cell interface."""
    def __init__(self, num_inputs, num_blocks=2, num_states=4, dropout=0,
                 rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_inputs = self.num_hiddens = num_inputs
        self.blocks = nnx.Sequential(*[
            MambaBlock(num_inputs, num_states, dropout=dropout, rngs=rngs)
            for _ in range(num_blocks)])
        self.ln = nnx.LayerNorm(num_inputs, rngs=rngs)

    def __call__(self, X, state=None):
        return self.ln(self.blocks(X)), None
```

### The Three Answers, Measured on One Task
:label:`subsec_capstone`

This chapter proposed three answers to its opening question of what a
hidden state should remember: *gate it* (the LSTM), *linearize it* (the
minGRU and the SSMs), *select it* (Mamba). Time to put all three on one
scoreboard. The task is the language-modeling recipe used throughout,
*The Time Machine* under the 1,024-token BPE tokenizer of
:numref:`sec_rnn-scratch`, 50,000 windows of 32 tokens, ten epochs,
gradients clipped to norm 1. Alongside perplexity we report bits per byte
(all three models share one tokenizer, but bpb keeps the scoreboard
comparable with the character-level models of :numref:`chap_rnn`), parameter
counts, and wall clock per epoch.

```{.python .input #mamba-the-three-answers-measured-on-one-task-1}
%%tab pytorch, jax, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32,
                       num_train=50000, num_val=5000)
ids = d2l.numpy(data.X[data.num_train:data.num_train + data.num_val,
                       0]).tolist()
bytes_per_token = len(data.tokenizer.decode(ids).encode('utf-8')) / len(ids)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-2}
%%tab pytorch
results = {}

def benchmark(name, model, epochs=10):
    trainer = d2l.Trainer(max_epochs=epochs, gradient_clip_val=1, num_gpus=1)
    model.board.yscale = 'log'
    start = time.time()
    trainer.fit(model, data)
    secs = (time.time() - start) / epochs
    ppl = float(model.board.data['val_ppl'][-1].y)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results[name] = (ppl, math.log2(ppl) / bytes_per_token, params, secs)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-2}
%%tab tensorflow
results = {}

def benchmark(name, model, epochs=10):
    trainer = d2l.Trainer(max_epochs=epochs, gradient_clip_val=1)
    model.board.yscale = 'log'
    start = time.time()
    trainer.fit(model, data)
    secs = (time.time() - start) / epochs
    ppl = float(model.board.data['val_ppl'][-1].y)
    params = sum(int(tf.size(v)) for v in model.trainable_variables)
    results[name] = (ppl, math.log2(ppl) / bytes_per_token, params, secs)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-2}
%%tab jax
results = {}

def val_ppl(model):
    model = nnx.view(model, deterministic=True, use_running_average=True,
                     raise_if_not_found=False)  # Dropout off for evaluation
    total_loss = num_tokens = 0
    for X_val, y_val in data.val_dataloader():
        losses = model.loss(model(X_val), y_val, averaged=False)
        total_loss += float(losses.sum())
        num_tokens += losses.size
    return math.exp(total_loss / num_tokens)

def benchmark(name, model, epochs=10):
    trainer = d2l.Trainer(max_epochs=epochs, gradient_clip_val=1, num_gpus=1)
    model.board.yscale = 'log'
    start = time.time()
    trainer.fit(model, data)
    secs = (time.time() - start) / epochs
    params = sum(p.size for p in
                 jax.tree.leaves(nnx.state(model, nnx.Param)))
    results[name] = (val_ppl(model), math.log2(val_ppl(model))
                     / bytes_per_token, params, secs)
```

The first answer, the LSTM of :numref:`sec_lstm`, and the second, the
minGRU of :numref:`subsec_mingru` (restated below), train with the exact
recipes of their home sections.

```{.python .input #mamba-the-three-answers-measured-on-one-task-3}
%%tab pytorch, jax
lstm_lm = d2l.RNNLM(d2l.LSTM(num_inputs=64, num_hiddens=128),
                    vocab_size=len(data.vocab), lr=4)
benchmark('LSTM', lstm_lm)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-3}
%%tab tensorflow
with d2l.try_gpu():
    lstm_lm = d2l.RNNLM(d2l.LSTM(num_inputs=64, num_hiddens=128),
                        vocab_size=len(data.vocab), lr=4)
    benchmark('LSTM', lstm_lm)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-4}
%%tab pytorch
class MinGRU(d2l.Module):
    """The minimal GRU of the previous section."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.W_xz = nn.Linear(num_inputs, num_hiddens)
        self.W_xh = nn.Linear(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        Z = torch.sigmoid(self.W_xz(inputs))     # (num_steps, batch, hiddens)
        H_tilde = torch.tanh(self.W_xh(inputs))
        a, b = 1 - Z, Z * H_tilde
        if H is not None:  # Fold the carried-in state into the first step
            b = torch.cat([b[:1] + a[:1] * H, b[1:]])
        outputs = associative_scan(a, b)
        return outputs, outputs[-1]

mingru_lm = d2l.RNNLM(MinGRU(num_inputs=64, num_hiddens=128),
                      vocab_size=len(data.vocab), lr=4)
benchmark('minGRU', mingru_lm)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-4}
%%tab tensorflow
class MinGRU(d2l.Module):
    """The minimal GRU of the previous section."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.W_xz = tf.keras.layers.Dense(num_hiddens)
        self.W_xh = tf.keras.layers.Dense(num_hiddens)

    def forward(self, inputs, H=None):
        Z = tf.sigmoid(self.W_xz(inputs))        # (num_steps, batch, hiddens)
        H_tilde = tf.tanh(self.W_xh(inputs))
        a, b = 1 - Z, Z * H_tilde
        if H is not None:  # Fold the carried-in state into the first step
            b = tf.concat([b[:1] + a[:1] * H, b[1:]], 0)
        outputs = associative_scan(a, b)
        return outputs, outputs[-1]

with d2l.try_gpu():
    mingru_lm = d2l.RNNLM(MinGRU(num_inputs=64, num_hiddens=128),
                          vocab_size=len(data.vocab), lr=4)
    benchmark('minGRU', mingru_lm)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-4}
%%tab jax
class MinGRU(nnx.Module):
    """The minimal GRU of the previous section."""
    def __init__(self, num_inputs, num_hiddens, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.W_xz = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)
        self.W_xh = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)

    def __call__(self, inputs, H=None):
        Z = jax.nn.sigmoid(self.W_xz(inputs))    # (num_steps, batch, hiddens)
        H_tilde = jnp.tanh(self.W_xh(inputs))
        a, b = 1 - Z, Z * H_tilde
        if H is not None:  # Fold the carried-in state into the first step
            b = jnp.concatenate([b[:1] + a[:1] * H, b[1:]])
        outputs = associative_scan(a, b)
        return outputs, outputs[-1]

mingru_lm = d2l.RNNLM(MinGRU(num_inputs=64, num_hiddens=128),
                      vocab_size=len(data.vocab), lr=4)
benchmark('minGRU', mingru_lm)
```

The third answer stacks two Mamba blocks at model width 128. Since
`d2l.RNNLM` supplies embedding and head around any module with the
`(inputs, state)` interface, only the optimizer needs overriding. One
adjustment is needed, and it is a compliment in disguise: trained like
the baselines, this model *overfits*. Its validation perplexity bottoms
out within a few epochs and then climbs while training perplexity keeps
falling toward single digits, which no gated model in this chapter comes
close to doing; a selective state is, among other things, an excellent
memorization device, and our corpus is one short novel. So Mamba, alone
in this chapter, gets the standard medicine of :numref:`sec_dropout`,
dropout on each block's residual branch, plus a gentler learning rate.

```{.python .input #mamba-the-three-answers-measured-on-one-task-5}
%%tab pytorch
class MambaLM(d2l.RNNLM):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

mamba_lm = MambaLM(Mamba(num_inputs=128, dropout=0.3),
                   vocab_size=len(data.vocab), lr=3e-4)
benchmark('Mamba', mamba_lm)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-5}
%%tab tensorflow
class MambaLM(d2l.RNNLM):
    def configure_optimizers(self):
        return tf.keras.optimizers.Adam(float(self.lr))

with d2l.try_gpu():
    mamba_lm = MambaLM(Mamba(num_inputs=128, dropout=0.3),
                       vocab_size=len(data.vocab), lr=3e-4)
    benchmark('Mamba', mamba_lm)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-5}
%%tab jax
class MambaLM(d2l.RNNLM):
    def configure_optimizers(self):
        return optax.adam(self.lr)

mamba_lm = MambaLM(Mamba(num_inputs=128, dropout=0.3),
                   vocab_size=len(data.vocab), lr=3e-4)
benchmark('Mamba', mamba_lm)
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-6}
%%tab pytorch, jax, tensorflow
print(f'{"model":>7} {"val ppl":>8} {"bpb":>6} {"params":>9} {"s/epoch":>8}')
for name, (ppl, bpb, params, secs) in results.items():
    print(f'{name:>7} {ppl:>8.1f} {bpb:>6.2f} {params:>9,} {secs:>8.1f}')
```

Numbers first, then caveats. In every framework we run, Mamba lands
clearly below the LSTM, by five to thirty points of perplexity in
our runs, at a parameter count larger than the LSTM's (most of the extra
sits in the block's expanded projections) and a slower epoch, our scan
being the teaching-grade version of the fused kernel discussed above. In
most runs it posts the best number of the chapter outright, though in
some the minGRU, at fewer than half the parameters, stays within a few
points or even edges it out: at this scale, framework initialization
defaults and optimizer choices move the scoreboard by amounts comparable
to the architectural gap. The honest caveats cut both ways: Mamba trains
with Adam and dropout while the gated baselines keep their sections'
plain SGD recipe, and at this corpus size a stronger model mostly buys
sharper memorization of Wells's prose. This scoreboard says the selective
architecture *can* be trained to better held-out prediction at comparable
scale, not that it dominates pound for pound; at research scale the
corresponding claim, matching transformers at small model sizes, is the
Mamba paper's central result.

Every language model in this book must also pass the smell test of
:numref:`sec_decoding`: generate something. We sample each model with the
same prefix, temperature, and min-$p$ filter, using the `d2l.generate`
helper built there, which needs only a function from a prefix to
next-token logits.

```{.python .input #mamba-the-three-answers-measured-on-one-task-7}
%%tab pytorch
def step_fn(model):
    def step(ids):  # Token ids in, numpy logits for the next token out
        with torch.no_grad():
            logits = model(d2l.tensor([ids], device=d2l.try_gpu()))
        return d2l.numpy(logits)[0, -1]
    return step

prefix = data.tokenizer.encode('the time traveller')
for name, model in [('LSTM', lstm_lm), ('minGRU', mingru_lm),
                    ('Mamba', mamba_lm)]:
    out = d2l.generate(step_fn(model), prefix, 25, strategy='sample',
                       temperature=1.0, min_p=0.1,
                       rng=np.random.default_rng(0))
    print(f'{name:>7}: {data.tokenizer.decode(out)!r}')
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-7}
%%tab tensorflow
def step_fn(model):
    def step(ids):  # Token ids in, numpy logits for the next token out
        return d2l.numpy(model(d2l.tensor([ids])))[0, -1]
    return step

# Every generated token changes the sequence length, so the conv re-traces;
# silence the (expected) retracing warnings for this cell
tf.get_logger().setLevel('ERROR')
prefix = data.tokenizer.encode('the time traveller')
for name, model in [('LSTM', lstm_lm), ('minGRU', mingru_lm),
                    ('Mamba', mamba_lm)]:
    out = d2l.generate(step_fn(model), prefix, 25, strategy='sample',
                       temperature=1.0, min_p=0.1,
                       rng=np.random.default_rng(0))
    print(f'{name:>7}: {data.tokenizer.decode(out)!r}')
tf.get_logger().setLevel('WARNING')
```

```{.python .input #mamba-the-three-answers-measured-on-one-task-7}
%%tab jax
def step_fn(model):
    model = nnx.view(model, deterministic=True, use_running_average=True,
                     raise_if_not_found=False)  # Dropout off for sampling
    def step(ids):  # Token ids in, numpy logits for the next token out
        return d2l.numpy(model(d2l.tensor([ids])))[0, -1]
    return step

prefix = data.tokenizer.encode('the time traveller')
for name, model in [('LSTM', lstm_lm), ('minGRU', mingru_lm),
                    ('Mamba', mamba_lm)]:
    out = d2l.generate(step_fn(model), prefix, 25, strategy='sample',
                       temperature=1.0, min_p=0.1,
                       rng=np.random.default_rng(0))
    print(f'{name:>7}: {data.tokenizer.decode(out)!r}')
```

### Selective Copying, Revisited

The scoreboard above is the everyday test; the section opened with the
diagnostic one. We now run the Mamba stack on the selective copying task
that the S4D could not solve, at the same parameter budget as before, and
replot all three curves together.

```{.python .input #mamba-selective-copying-revisited-1}
%%tab pytorch
mamba_copy = CopyModel(nn.Sequential(*[MambaBlock(48, 4) for _ in range(2)]),
                       num_hiddens=48)
train_copy('Mamba', mamba_copy, copy_data)
```

```{.python .input #mamba-selective-copying-revisited-1}
%%tab tensorflow
with d2l.try_gpu():
    mamba_copy = CopyModel(
        tf.keras.Sequential([MambaBlock(48, 4) for _ in range(2)]),
        num_hiddens=48)
    train_copy('Mamba', mamba_copy, copy_data)
```

```{.python .input #mamba-selective-copying-revisited-1}
%%tab jax
mamba_copy = CopyModel(
    nnx.Sequential(*[MambaBlock(48, 4) for _ in range(2)]), num_hiddens=48)
train_copy('Mamba', mamba_copy, copy_data)
```

```{.python .input #mamba-selective-copying-revisited-2}
%%tab pytorch, jax, tensorflow
names = list(copy_curves)
d2l.plot([copy_curves[n][0] for n in names],
         [copy_curves[n][1] for n in names], 'epoch',
         'validation accuracy', legend=names, figsize=(5, 3))
```

Mamba solves the task, and faster than the LSTM did. The mechanism is
exactly the one we built: on filler tokens the learned $\Delta_t$
collapses toward zero and the state carries the stored symbols forward
untouched; on a marked symbol $\Delta_t$ opens and writes. The
architectural circle closes here. The content-dependent gating that
:numref:`subsec_mingru` deleted to linearize the recurrence has been
restored, not by putting the state back inside a nonlinearity, but by
letting the input choose the coefficients of a linear map, and the scan
never noticed the difference.

## What a Fixed State Cannot Do
:label:`subsec_fixed-state-limits`

Selectivity fixed content-blindness. It did not, and cannot, fix
capacity. A Mamba layer's memory is still a fixed block of numbers, $N$
states for each of $H$ channels, and information theory does not care
how cleverly the update rule was chosen: to reproduce $k$ arbitrary
tokens from a vocabulary of size $V$, *something* in the model must hold
$k \log_2 V$ bits from the moment they appear to the moment they are
needed. Our selective-copy experiment lived comfortably inside that
budget, four symbols of three bits each against hundreds of state
dimensions. Scale the demand instead of the model and the wall is
mathematical: once what must be recalled exceeds what the state can
encode, no parameterization, gating, or training trick can help.

This is not hypothetical; it is measurable, and the measurements shaped
today's architectures. :citet:`Jelassi.Brandfonbrener.Kakade.ea.2024`
study copying itself, prove that a transformer can copy strings
exponentially longer than any fixed-state model at comparable size (its
"state", the attention window over everything so far, grows with the
sequence), and confirm empirically that copying and retrieval are where
SSM language models lag transformers even when perplexity is close. The
multi-query associative recall benchmark of
:citet:`Arora.Eyuboglu.Timalsina.ea.2024` reaches the same verdict from
the other side: recurrent models solve recall only while the number of
key-value pairs in play fits in their state, and the accuracy cliff
tracks state size almost exactly. We do not reproduce these experiments
here, honestly, because they only bite at scales beyond a textbook
notebook: our Time Machine perplexities cannot show a recall gap that
emerges when a model must retrieve one fact from tens of thousands of
tokens. An exercise asks you to find the cliff on selective copying
instead, where it is within reach.

:citet:`Gu.2025` offers a framing for this trade that is worth carrying
forward. A recurrent state is a *brain*: a compressed, always-on working
memory, constant cost per step, which must decide at write time what will
matter later, and therefore forgets. The transformer's growing key-value
cache is a *database*: a lossless log of everything, pay-per-query at
ever-growing cost, which never has to predict what will matter because it
keeps it all. Neither dominates. One number that should temper
enthusiasm for pure databases: a brain-style model reads a million-token
context into kilobytes-to-megabytes of state, while a database-style one
drags along a cache orders of magnitude larger, at which point serving
cost, not quality, decides architectures. The obvious synthesis, a brain
in front of a database, is where the field actually landed, as the next
subsection records.

## The Recurrent Frontier
:label:`subsec_recurrent-frontier`

Where does this leave recurrence in the age of the transformer? As of
this writing, three observations summarize the production landscape.

**Hybrids won.** Between the pure recurrent model and the pure
transformer, practice chose the mixture. Jamba interleaves one attention
block among each handful of Mamba blocks at 52B parameters
:cite:`Lieber.Lenz.Bata.ea.2024`; Griffin and its RecurrentGemma
derivatives alternate gated linear recurrences with *local* attention
:cite:`De.Smith.Fernando.ea.2024`; and several recent open-weight model
families ship attention-to-recurrence ratios between one-in-three and
one-in-seven. The design logic follows the brains-and-databases picture
directly: a few attention layers provide exact retrieval over a window,
recurrent layers provide cheap always-on context between them, and most
of the quality of full attention survives at a fraction of the inference
cost.

**The siblings converged.** Mamba was not alone. RWKV
:cite:`Peng.Alcaide.Anthony.ea.2023`, xLSTM
:cite:`Beck.Poppel.Spanring.ea.2024`, and gated linear attention
:cite:`Yang.Wang.Shen.ea.2024` all arrived, from RNN, LSTM, and
attention lineages respectively, at the same computational object: a
matrix-valued state updated by a learned decay plus an outer-product
write, evaluated in parallel by a scan, differing mainly in the update
rule and how the decay is parameterized. When three research lineages
meet at one design, the design is probably not an accident. The precise
statement of the meeting point, Mamba-2's *state space duality* showing
that a selective SSM is a form of masked attention :cite:`Dao.Gu.2024`,
needs the attention machinery of the next chapter, and we defer it
there; a third generation, Mamba-3, had just been announced as this
chapter was written.

**Where no attention is needed at all.** On modalities without natural
tokens, sampled at high rates, with information spread thinly and
locally, pure state space models remain the architecture of choice. In
raw-audio generation and understanding, SSM stacks operate directly on
waveform samples at rates where attention windows are hopeless. In
genomics, million-nucleotide DNA models are SSM-based for the same
reason, and the byte-level frontier of language modeling (an exercise
lets you probe it in miniature) leans the same way. The pattern matches
everything this chapter taught: the less a task depends on retrieving
exact distant items, and the longer its sequences, the better the
compressed always-on state fares against the growing log.

![One chapter, three answers to "what should a hidden state remember?": gate the state (LSTM), linearize the state path and scan (minGRU, S4D), make the linear dynamics select by content (Mamba).](../img/mdl-modernrnn-three-answers.svg)
:label:`fig_three_answers`

## Summary

This chapter asked what a hidden state should remember, and gave three
answers of increasing refinement (:numref:`fig_three_answers`). *Gate it*:
multiplicative gates let the data control writing, keeping, and exposing
memory, and made recurrent networks trainable over long ranges. *Linearize
it*: removing the nonlinearity from the state path turned the recurrence
into an associative scan, restoring parallel training, and the state space
view added principled step-size gates, stability by construction, and
provably good memory. *Select it*: making the step size and projections
functions of the input restored the content-awareness that linearization
lost, at unchanged scan cost, and the resulting Mamba block solved our
selective-copy task and topped our capstone scoreboard. What no update
rule can change is that a fixed-size state holds a fixed number of bits:
exact recall of an unbounded past demands memory that grows with the past.
The remaining move is to keep *everything* and learn what to look at.
That mechanism is attention, and it is the subject of the next chapter
(:numref:`chap_attention-and-transformers`).

## Exercises

1. *Selective in $\Delta$ only.* Modify `SelectiveSSM` so that only
   $\boldsymbol{\Delta}_t$ depends on the input, with
   $\mathbf{B} = \mathbf{1}$ fixed and $\mathbf{C}$ a plain parameter as
   in `S4D`, and rerun the selective copying experiment. Then try the
   converse (input-dependent $\mathbf{B}_t, \mathbf{C}_t$ with a fixed
   learned $\Delta$). Which ingredient carries the task, and why does
   that agree with the gate interpretation of :numref:`subsec_zoh`?
1. *Finding the capacity cliff.* Fix the Mamba copy model and grow the
   task: sweep the number of marked symbols (say 4, 8, 16, 32) at
   `num_states=4`, then repeat with `num_states=16`. Plot final accuracy
   against the number of symbols for both state sizes. Where does each
   model break, and how does the break point move with state size?
   Relate your finding to the arguments of
   :numref:`subsec_fixed-state-limits`.
1. *Ablating the block.* The Mamba block multiplies the SSM output by a
   SiLU gate branch. Remove the gate (pass the SSM output straight to
   `W_out`), and separately replace both SiLU activations by ReLU or by
   the identity. Retrain the capstone language model for each variant.
   Which change hurts most, and does the gate matter more for perplexity
   or for the selective-copy task?
1. *Bytes instead of BPE.* Rerun the capstone comparison at the byte
   level by constructing the dataset with
   `d2l.TimeMachine(..., tokenization='char')` and doubling `num_steps`
   so each window spans comparable text. Compare LSTM and Mamba by bits
   per byte, not perplexity. Does Mamba's advantage grow or shrink, and
   what does that suggest about the byte-level modeling mentioned in
   :numref:`subsec_recurrent-frontier`?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide title="Selective State Space Models"}
10.3 ended on a confession: the S4D is **LTI**. Its kernel weights the
past by *position*, never by *content*: the model decides what to
remember before it sees the input.

The LSTM had the opposite profile: gates that read the data, but a
nonlinear recurrence that trains sequentially.

. . .

This section: make the dynamics a **function of the input**
(Gu & Dao, 2023):

- the forget gate falls out a **third** time (engineering → calculus →
  selectivity),
- the scan survives; the kernel view does not,
- packaged as the **Mamba** block: the first recurrence since the LSTM
  to compete with transformers on language.
:::

::: {.slide title="A task that defeats time invariance"}
**Selective copying**: symbols scattered among filler at *random*
positions; reproduce them, in order, at the query slots.

![](../img/mdl-modernrnn-selective-copy.svg){width=100%}

. . .

- Store a token *because it is a symbol*, not because of where it sits.
- LM-scale counterpart: **associative recall** / induction heads; a
  synthetic recall benchmark predicts most of the attention-vs-SSM gap
  (Arora et al., 2024).
:::

::: {.slide title="Generating the task"}
Filler is token 0, queries are token 1, symbols are ids 2-9:

@mamba-a-task-that-defeats-time-invariance
:::

::: {.slide title="Two witnesses from earlier sections"}
The S4D stack of 10.3 (restated verbatim) vs. the LSTM of 10.1, both
~30k parameters, same harness: embed, encode, classify the query slots.

@mamba-an-lti-baseline-and-a-gated-one-2

. . .

@mamba-an-lti-baseline-and-a-gated-one-3
:::

::: {.slide title="The LTI model cannot; the gated one can"}
@mamba-an-lti-baseline-and-a-gated-one-5

. . .

- LSTM: climbs to solved. Gates read the data; "store this, it is a
  symbol" is learnable.
- S4D: **plateaus**. Pointwise nonlinearities can suppress filler
  locally, but a fixed kernel cannot align "third symbol seen" with
  "third query slot" under variable spacing.
- More epochs do not help: the failure is architectural.
:::

::: {.slide title="Selective SSMs: let the input choose the dynamics"}
One change to 10.3's recipe: step size, input matrix, read-out become
functions of the input,

$$\boldsymbol{\Delta}_t = \textrm{softplus}(\mathbf{u}_t \mathbf{W}_{\Delta} + \mathbf{b}_{\Delta}), \qquad \mathbf{B}_t = \mathbf{u}_t \mathbf{W}_B, \qquad \mathbf{C}_t = \mathbf{u}_t \mathbf{W}_C,$$

$$\mathbf{x}_t = e^{\Delta_{t,h} \mathbf{a}} \odot \mathbf{x}_{t-1} + \Delta_{t,h}\, u_{t,h}\, \mathbf{B}_t.$$

. . .

Recall the box from 10.3: $\Delta \to 0$ freezes the state,
$\Delta$ large overwrites it. Input-dependent $\Delta_t$ = **forget and
input gate in one scalar**, on a linear state:

- filler → $\Delta_t \approx 0$: glide through, untouched;
- symbol → $\Delta_t$ opens: write.

The gate, derived a third time.
:::

::: {.slide title="The price and the save"}
**Price**: time-varying coefficients kill the convolution view. No fixed
$\bar{\mathbf{K}}$, no FFT. Of 10.3's three views only the recurrence
survives.

**Save**: the recurrence is *still* an affine map of the state; the
combine $(\mathbf{a}_2, \mathbf{b}_2) \circ (\mathbf{a}_1, \mathbf{b}_1)$
never assumed constant coefficients. Same scan, per-step tensors:

@mamba-what-selectivity-costs-and-what-survives-1
:::

::: {.slide title="The selective SSM layer"}
Three changes vs. `S4D`: $\Delta, \mathbf{B}, \mathbf{C}$ from linear
heads; low-rank $\Delta$ head (`dt_rank`); bias init so the *untrained*
layer is approximately the multi-timescale S4D:

@mamba-what-selectivity-costs-and-what-survives-2
:::

::: {.slide title="Why 2023 and not 2020: hardware-awareness"}
Our implementation materializes `(T, batch, H, N)` coefficient tensors:
fine at textbook scale, ruinous at model scale (the compute is trivial;
**memory traffic** sets the speed).

. . .

The Mamba kernel:

- fuses discretization + scan + read-out in one pass, intermediates in
  on-chip SRAM,
- **recomputes** them in the backward pass instead of storing
  (store-vs-recompute, cf. 9.6),
- changes nothing about *what* is computed, only whether it was worth
  computing.
:::

::: {.slide title="The Mamba block"}
![](../img/mdl-modernrnn-mamba-block.svg){width=52%}
:::

::: {.slide title="Block and language model"}
A dozen lines around `SelectiveSSM`; the stack keeps the
`(inputs, state)` interface, so `d2l.RNNLM` wraps it unchanged:

@mamba-the-mamba-block
:::

::: {.slide title="Capstone: the chapter's three answers on one task"}
**Gate it** (LSTM) vs. **linearize it** (minGRU) vs. **select it**
(Mamba); Time Machine BPE, 50k windows of 32, ten epochs, clip 1:

@mamba-the-three-answers-measured-on-one-task-6

. . .

- Mamba: best validation perplexity of the chapter in most runs (minGRU
  edges it on some tabs).
- Caveats stated honestly: Adam vs. the baselines' SGD; more parameters;
  a small corpus rewards memorization.
:::

::: {.slide title="Sampling all three (9.7's toolkit)"}
@mamba-the-three-answers-measured-on-one-task-7
:::

::: {.slide title="Selective copying, revisited"}
@mamba-selective-copying-revisited-2

. . .

Mamba solves what the S4D could not, faster than the LSTM: on filler
$\Delta_t$ collapses to zero (state glides), on symbols it opens
(write). The content-dependent gating deleted in 10.3 is restored,
**without** giving up the scan.
:::

::: {.slide title="What a fixed state cannot do"}
Selectivity fixed content-blindness, not **capacity**: recalling $k$
arbitrary tokens needs $k \log_2 |\mathcal{V}|$ bits in the state, no
matter the update rule.

- "Repeat After Me": transformers copy strings exponentially longer
  than any fixed-state model (Jelassi et al., 2024).
- MQAR: recall works until the key-value pairs outgrow the state; the
  cliff tracks state size (Arora et al., 2024).
- (Not reproduced here: these bite beyond textbook scale.)

. . .

Gu's framing: recurrent state = **brain** (compressed, always-on,
forgets); KV cache = **database** (lossless, growing, pay-per-query).
Neither dominates.
:::

::: {.slide title="The recurrent frontier"}
- **Hybrids won**: Jamba, Griffin/RecurrentGemma; open-weight families
  at one attention layer per 3-7 recurrent ones. A few exact-retrieval
  layers + cheap always-on context between them.
- **Siblings converged**: RWKV, xLSTM, GLA: different update rules on a
  matrix state, all scanned. Mamba-2's SSM ≡ masked-attention duality:
  *deferred* until attention is taught (Mamba-3: just announced).
- **SSMs win outright** where sequences are long and tokens are raw:
  audio waveforms, DNA, byte-level text.
:::

::: {.slide title="Summary: one question, three answers"}
![](../img/mdl-modernrnn-three-answers.svg){width=100%}

. . .

What no update rule changes: a fixed state holds a fixed number of
bits. Exact recall of an unbounded past needs memory that **grows**.

Keep everything, and *learn what to look at*: **attention**, next
chapter.
:::
