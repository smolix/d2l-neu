# Gated Recurrent Networks
:label:`sec_lstm`

The plain recurrent network of :numref:`sec_rnn-scratch` generates recognizable words and punctuation, yet its grammar unravels over more than a few words. Consider predicting the verb after *"the keys to the cabinet in the hall"*: choosing *"are"* rather than *"is"* requires remembering that *"keys"*—30 characters back—is plural. Because a plain RNN overwrites its entire state at every step through the same recurrent matrix and $\tanh$ nonlinearity, both the forward signal and the backward gradient decay rapidly over tens of steps. How can a recurrent network hold what it has read across hundreds of steps?

The solution introduced in the late 1990s is to let the network decide, unit by unit and step by step, what fraction of its state to keep via a *gate*—a vector of values in $(0, 1)$ computed from the current input and previous state. A unit whose gate stays near $1$ passes both its stored content and the backward gradient across the step essentially unchanged. Below we build the two classic gated architectures—the long short-term memory (LSTM) and the gated recurrent unit (GRU)—and evaluate them on synthetic delayed recall and on the byte-level WikiText-103 benchmark of :numref:`chap_transformers`. While gates solve long-range retention, their dependence on the previous state still forces sequential step-by-step training; :numref:`sec_linear-recurrence` removes that bottleneck.

Throughout this chapter we write equations for a single sequence using column vectors, while the code processes minibatches of sequences with row vectors.

```{.python .input #lstm-gated-recurrent-networks}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #lstm-gated-recurrent-networks}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Gates

### Vanishing Gradients

Recall from :numref:`subsec_bptt_analysis` how gradients propagate through a plain RNN with hidden state $\mathbf{h}_t \in \mathbb{R}^d$:

$$\mathbf{h}_t = \tanh(\mathbf{W}_\textrm{hh} \mathbf{h}_{t-1} +
\mathbf{W}_\textrm{xh} \mathbf{x}_t + \mathbf{b}_\textrm{h}).$$

A perturbation to $\mathbf{h}_{t-k}$ reaches $\mathbf{h}_t$ multiplied by a chain of $k$ step Jacobians $\mathbf{J}_s = \textrm{diag}(1 - \mathbf{h}_s^2)\,\mathbf{W}_\textrm{hh}$:

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-k}} = \mathbf{J}_t
\mathbf{J}_{t-1} \cdots \mathbf{J}_{t-k+1}.
$$
:eqlabel:`eq_lstm-jacobian-product`

For a single scalar unit, each factor is $w(1 - h_s^2)$ of magnitude at most $|w|$. When $|w| < 1$, the product shrinks at least as fast as $|w|^k$ (at $w = 0.9$, to about $4\%$ over 30 steps and $\le 0.003\%$ over 100 steps).

For a matrix $\mathbf{W}_\textrm{hh}$, if its largest singular value is below $1$, every factor contracts every vector and the product must vanish. Over many steps, matrix powers scale as $\rho^k$, where the *spectral radius* $\rho$ (the largest eigenvalue magnitude) satisfies $\rho \le \|\mathbf{W}_\textrm{hh}\|_2$. If $\rho \approx 1$ and the state remains small so $\tanh$ is nearly linear, components along the leading eigenvectors can survive for many steps—a borderline regime we will observe in our experiments below. Away from $\rho \approx 1$, gradients either explode (which gradient clipping can cap) or vanish (which no post-hoc rescaling can recover).

### An Additive Memory

The simplest recurrence with unit Jacobian is a pure running sum, $c_t = c_{t-1} + x_t$, for which $\partial c_t / \partial c_{t-k} = 1$ at every lag $k$. However, an unweighted sum never forgets: every past input remains superposed with equal weight, and the state's variance grows with sequence length.

### Gates as Learned Decay

To retain relevant inputs while discarding irrelevant ones, we let the network choose at each step a retention fraction $f_t \in (0, 1)$ and an input fraction $i_t \in (0, 1)$ multiplying new candidate content $\tilde{c}_t$:

$$c_t = f_t\, c_{t-1} + i_t\, \tilde{c}_t.$$
:eqlabel:`eq_lstm-gated-sum`

Along the direct memory path, the $k$-step derivative is the product of the kept fractions, $\partial c_t / \partial c_{t-k} = \prod_{s=t-k+1}^t f_s$. At a constant keep fraction $f$, a stored value decays as $f^k$ with a *half-life* of $\ln 2 / \ln(1/f)$ steps: roughly 7 steps at $f = 0.9$, 69 at $f = 0.99$, and 693 at $f = 0.999$. Setting $f_t \to 1$ preserves a value and its gradient indefinitely; setting $f_t \to 0$ flushes the cell in a single step.

In a layer of width $d$, the memory becomes a vector $\mathbf{c}_t \in \mathbb{R}^d$ accompanied by an exposed hidden state $\mathbf{h}_t \in \mathbb{R}^d$. Each *gate* computes a vector in $(0, 1)^d$ via a sigmoid layer reading the current input $\mathbf{x}_t$ and previous output $\mathbf{h}_{t-1}$:

$$
\mathbf{f}_t = \sigma(\mathbf{W}_\textrm{xf} \mathbf{x}_t +
\mathbf{W}_\textrm{hf} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{f}).
$$
:eqlabel:`eq_gate`

Unlike a plain RNN—which multiplies the entire state by a dense matrix $\mathbf{W}_\textrm{hh}$ and applies $\tanh$ at every step—the gated memory updates each coordinate by elementwise multiplication with its own learned factor $f_{t,j} \in (0, 1)$. Because $f_{t,j} \le 1$, gradients along the direct memory path cannot explode, and holding $f_{t,j}$ near $1$ prevents them from vanishing.

Two practical details remain. First, while $f_t \approx 1$, repeated writes can still cause the stored value $c_t$ to grow, so the LSTM applies a $\tanh$ when reading $\mathbf{c}_t$ out to $\mathbf{h}_t$. Second, under zero bias initialization, sigmoid gates start near $\sigma(0) = 1/2$—a half-life of a single step—so a gated network can represent long memories but does not start with them by default.

## Long Short-Term Memory

### The Memory Cell

The *long short-term memory* (LSTM) of :citet:`Hochreiter.Schmidhuber.1997`, augmented with the forget gate of :citet:`Gers.Schmidhuber.Cummins.2000`, builds a recurrent layer around :eqref:`eq_lstm-gated-sum`. It maintains two vectors of width $d$: an internal *memory cell* $\mathbf{c}_t$ evolved purely by elementwise gating and addition, and an exposed *hidden state* $\mathbf{h}_t$ that provides a gated, squashed view of $\mathbf{c}_t$ to subsequent layers and to the next step's gates.

Three sigmoid gates—the *input gate* $\mathbf{i}_t$, *forget gate* $\mathbf{f}_t$, and *output gate* $\mathbf{o}_t$—control access to the cell:

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_\textrm{xi} \mathbf{x}_t + \mathbf{W}_\textrm{hi} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{i}),\\
\mathbf{f}_t &= \sigma(\mathbf{W}_\textrm{xf} \mathbf{x}_t + \mathbf{W}_\textrm{hf} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{f}),\\
\mathbf{o}_t &= \sigma(\mathbf{W}_\textrm{xo} \mathbf{x}_t + \mathbf{W}_\textrm{ho} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{o}).
\end{aligned}
$$
:eqlabel:`lstm_gates`

A fourth projection with $\tanh$ activation proposes the candidate *input node* $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_\textrm{xc} \mathbf{x}_t + \mathbf{W}_\textrm{hc} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{c})$. The cell and hidden state then update via elementwise multiplication ($\odot$):

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot
\tilde{\mathbf{c}}_t, \qquad \mathbf{h}_t = \mathbf{o}_t \odot
\tanh(\mathbf{c}_t).
$$
:eqlabel:`lstm_update`

Separating $\mathbf{c}_t$ from $\mathbf{h}_t$ lets a unit hold information silently until it is needed: for instance, if $c_{t-1} = 0.8$ with $f_t = 0.95$, $i_t = 0.1$, $\tilde{c}_t = -0.5$, and $o_t = 0.2$, the cell stays at $c_t = 0.71$ while exposing only $h_t = 0.2\tanh(0.71) = 0.12$ (:numref:`fig_lstm_cell`).

![The LSTM memory cell. The previous output $\mathbf{h}_{t-1}$ and the input $\mathbf{x}_t$ feed four layers: three sigmoid gates and a $\tanh$ input node. The cell $\mathbf{c}_t$ is updated additively, scaled by the forget gate, and the output gate controls what the rest of the network sees.](../img/mdl-modernrnn-lstm-cell.svg)
:label:`fig_lstm_cell`

With the gates treated as fixed, the direct cell-to-cell Jacobian $\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1}$ is simply $\textrm{diag}(\mathbf{f}_t)$ (the top rail in :numref:`fig_lstm_cell`). The full derivative also includes indirect terms through $\mathbf{h}_{t-1}$ and the gates, but the cell rail provides an unattenuated highway whenever $\mathbf{f}_t$ stays near $1$.

### Implementation

Because $\mathbf{i}_t$, $\mathbf{f}_t$, $\tilde{\mathbf{c}}_t$, and $\mathbf{o}_t$ all read $(\mathbf{x}_t, \mathbf{h}_{t-1})$, we stack their weights into two matrices $\mathbf{W}_\textrm{x} \in \mathbb{R}^{d_\textrm{in} \times 4d}$ and $\mathbf{W}_\textrm{h} \in \mathbb{R}^{d \times 4d}$, compute a single matrix multiplication per input, and split the result into four chunks.

```{.python .input #lstm-implementation-1}
%%tab pytorch
class LSTMScratch(nn.Module):
    """An LSTM layer computed one time step at a time."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.num_hiddens = num_hiddens
        # Weights of i, f, c_tilde, and o, stacked along the last axis, with
        # variance 1 / fan-in
        self.W_x = nn.Parameter(torch.randn(num_inputs, 4 * num_hiddens)
                                / math.sqrt(num_inputs))
        self.W_h = nn.Parameter(torch.randn(num_hiddens, 4 * num_hiddens)
                                / math.sqrt(num_hiddens))
        self.b = nn.Parameter(torch.zeros(4 * num_hiddens))

    def forward(self, X):  # X: (batch_size, num_steps, num_inputs)
        h = c = torch.zeros(X.shape[0], self.num_hiddens, device=X.device)
        outputs = []
        for x in X.unbind(1):  # One time step at a time
            i, f, c_tilde, o = (x @ self.W_x + h @ self.W_h + self.b).chunk(4, -1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(c_tilde)
            h = torch.sigmoid(o) * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, 1)  # (batch_size, num_steps, num_hiddens)
```

```{.python .input #lstm-implementation-1}
%%tab jax
class LSTMScratch(nnx.Module):
    """An LSTM layer computed one time step at a time."""
    def __init__(self, num_inputs, num_hiddens, *, rngs):
        self.num_hiddens = num_hiddens
        # Weights of i, f, c_tilde, and o, stacked along the last axis, with
        # variance 1 / fan-in
        self.W_x = nnx.Param(rngs.params.normal((num_inputs, 4 * num_hiddens))
                             / math.sqrt(num_inputs))
        self.W_h = nnx.Param(rngs.params.normal((num_hiddens, 4 * num_hiddens))
                             / math.sqrt(num_hiddens))
        self.b = nnx.Param(jnp.zeros(4 * num_hiddens))

    def __call__(self, X):  # X: (batch_size, num_steps, num_inputs)
        def step(state, x):
            h, c = state
            i, f, c_tilde, o = jnp.split(
                x @ self.W_x[...] + h @ self.W_h[...] + self.b[...], 4, -1)
            c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(c_tilde)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            return (h, c), h
        h = c = jnp.zeros((X.shape[0], self.num_hiddens))
        # lax.scan steps through the leading axis, so time goes first
        _, outputs = jax.lax.scan(step, (h, c), jnp.swapaxes(X, 0, 1))
        return jnp.swapaxes(outputs, 0, 1)  # (batch_size, num_steps, ...)
```

To verify that `LSTMScratch` computes the exact same recurrence as the framework's LSTM layer, we copy the library weights into our class and compare outputs on a random batch.

:begin_tab:`pytorch`
The library layer is the fused `nn.LSTM`.
:end_tab:

:begin_tab:`jax`
The library layer is `nnx.LSTMCell`.
:end_tab:

```{.python .input #lstm-implementation-2}
%%tab pytorch
lstm = nn.LSTM(16, 32, batch_first=True)  # Stacks its weights as i, f, c~, o
cell = LSTMScratch(16, 32)
with torch.no_grad():
    cell.W_x.copy_(lstm.weight_ih_l0.T)
    cell.W_h.copy_(lstm.weight_hh_l0.T)
    cell.b.copy_(lstm.bias_ih_l0 + lstm.bias_hh_l0)
X = torch.randn(4, 9, 16)  # (batch_size, num_steps, num_inputs)
print(f'max difference: {(cell(X) - lstm(X)[0]).abs().max():.1e}')
```

```{.python .input #lstm-implementation-2}
%%tab jax
lstm_cell = nnx.LSTMCell(16, 32, rngs=nnx.Rngs(0))  # A layer per gate
lstm = nnx.RNN(lstm_cell)
cell = LSTMScratch(16, 32, rngs=nnx.Rngs(1))
gates = [(lstm_cell.ii, lstm_cell.hi), (lstm_cell.if_, lstm_cell.hf),
         (lstm_cell.ig, lstm_cell.hg), (lstm_cell.io, lstm_cell.ho)]
cell.W_x[...] = jnp.concatenate([x.kernel[...] for x, _ in gates], 1)
cell.W_h[...] = jnp.concatenate([h.kernel[...] for _, h in gates], 1)
cell.b[...] = jnp.concatenate([h.bias[...] for _, h in gates])
X = jax.random.normal(jax.random.key(0), (4, 9, 16))
print(f'max difference: {jnp.abs(cell(X) - lstm(X)).max():.1e}')
```

The outputs agree to float32 rounding error ($\sim 10^{-7}$), so we use the faster fused library layers from here on.

## Gated Recurrent Units
:label:`sec_gru`

Does a gated layer really need two state vectors and three gates? The *gated recurrent unit* (GRU) of :citet:`Cho.Van-Merrienboer.Gulcehre.ea.2014` merges the cell and hidden state into a single vector $\mathbf{h}_t$ controlled by just two gates—a *reset gate* $\mathbf{r}_t$ and an *update gate* $\mathbf{z}_t$—while matching the LSTM on many sequence tasks :cite:`Chung.Gulcehre.Cho.ea.2014`:

$$
\mathbf{r}_t = \sigma(\mathbf{W}_\textrm{xr} \mathbf{x}_t +
\mathbf{W}_\textrm{hr} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{r}), \qquad
\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{xz} \mathbf{x}_t +
\mathbf{W}_\textrm{hz} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{z}).
$$
:eqlabel:`gru_gates`

The reset gate filters how much of $\mathbf{h}_{t-1}$ enters the proposed *candidate state*:

$$
\tilde{\mathbf{h}}_t = \tanh\big(\mathbf{W}_\textrm{xh} \mathbf{x}_t +
\mathbf{W}_\textrm{hh} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) +
\mathbf{b}_\textrm{h}\big).
$$
:eqlabel:`gru_tilde_H`

The update gate $\mathbf{z}_t$ then interpolates convexly between the old state and the candidate:

$$
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t)
\odot \tilde{\mathbf{h}}_t.
$$
:eqlabel:`gru_H`

Equation :eqref:`gru_H` is the gated sum :eqref:`eq_lstm-gated-sum` with coupled forget and input gates ($f_t = z_t, i_t = 1 - z_t$). Because each coordinate convexly blends two values in $[-1, 1]$, $\mathbf{h}_t$ stays bounded in $[-1, 1]$ without an extra output $\tanh$ (:numref:`fig_gru_cell`), and a GRU layer requires $6d^2$ recurrent/input weights compared to an LSTM's $8d^2$. (Fused library GRUs apply $\mathbf{r}_t$ *after* multiplying by $\mathbf{W}_\textrm{hh}$ so all recurrent matrix products can be batched into one call; see the exercises.)

![The GRU cell. The reset gate filters the previous state inside the candidate $\tilde{\mathbf{h}}_t$; the update gate blends the old state and the candidate.](../img/mdl-modernrnn-gru-cell.svg)
:label:`fig_gru_cell`

## Training Gated Networks
:label:`sec_gated-training`

### Delayed Recall
:label:`subsec_delayed-recall`

The gated equations *allow* memories lasting hundreds of steps, but does gradient descent actually find them? We test retention directly on a synthetic *delayed recall* task: a sequence begins with one of 8 target symbols (0–7), followed by $T - 1$ random filler tokens drawn from 8 other symbols (8–15), and at step $T$ the model must classify the initial symbol (chance accuracy $1/8 = 0.125$).

```{.python .input #lstm-delayed-recall-1}
%%tab pytorch, jax
def delayed_recall(rng, batch_size, num_steps, num_symbols=8,  #@save
                   blank_fillers=False):
    """A symbol, then num_steps - 1 fillers; the target is the symbol."""
    Y = rng.integers(0, num_symbols, (batch_size, 1))  # One query, at the end
    X = rng.integers(num_symbols, 2 * num_symbols, (batch_size, num_steps))
    if blank_fillers:  # The same filler token at every step
        X[:] = num_symbols
    X[:, :1] = Y
    return X, Y

X, Y = delayed_recall(np.random.default_rng(0), 2, 6)
print(X, Y)
```

Our testbed model (`RecallModel`) embeds tokens into 32 dimensions, runs a single recurrent layer of width 64, and applies a linear classifier at the final step. Throughout this chapter, we initialize all linear, convolutional, and recurrent weight matrices with LeCun normal initialization (variance $1 / n_\textrm{in}$) and zero biases (:numref:`sec_numerical_stability`).

:begin_tab:`pytorch`
`init_lecun` applies LeCun normal weights and zero biases across linear, convolutional, and recurrent modules, and `Recurrent` wraps PyTorch recurrent modules to return only the output sequence tensor.
:end_tab:

:begin_tab:`jax`
Flax linear layers and recurrent input projections already default to LeCun normal, while recurrent kernels default to orthogonal matrices; we pass `recurrent_kernel_init` to use LeCun normal there as well.
:end_tab:

```{.python .input #lstm-delayed-recall-6}
%%tab pytorch
def init_lecun(module):  #@save
    """LeCun normal weights and zero biases for all linear, convolutional,
    and recurrent layers."""
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            fan_in = layer.weight[0].numel()  # Inputs of one output unit
            nn.init.normal_(layer.weight, std=fan_in ** -0.5)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.RNNBase):  # Input and recurrent matrices
            for name, param in layer.named_parameters():
                if name.startswith('weight'):
                    nn.init.normal_(param, std=param.shape[1] ** -0.5)
                else:
                    nn.init.zeros_(param)
    return module

class Recurrent(nn.Module):  #@save
    """A batch-first recurrent layer that returns only its outputs."""
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self, X):
        return self.rnn(X)[0]
```

We compare six recurrent configurations in three pairs:

1. **Plain RNNs:** one with LeCun normal recurrent weights ($\sigma = 1/\sqrt{64} = 1/8$) and a control with uniform weights on $[-1/8, 1/8]$ (variance $1/3$ of LeCun normal).
2. **Default LSTM and GRU:** zero gate biases, so all gates start near $\sigma(0) = 1/2$ (half-life 1 step).
3. **Keep-gate LSTM and GRU (`init_keep_gate`):** the LSTM forget-gate bias and GRU update-gate bias are initialized to $+5$ ($\sigma(5) = 0.993$, half-life $\approx 100$ steps), and the LSTM input-gate bias is set to $-5$ ($\sigma(-5) = 1 - \sigma(5) = 0.007$). Pairing $b_\textrm{i} = -b_\textrm{f}$ bounds the LSTM cell magnitude by $\sigma(-5)/(1 - \sigma(5)) = 1$ across long filler runs so the output $\tanh$ never saturates, matching the GRU's built-in $1 - \mathbf{z}_t$ coupling (see :citet:`Jozefowicz.Zaremba.Sutskever.2015` and the *chrono initialization* of :citet:`Tallec.Ollivier.2018`).

:begin_tab:`pytorch`
The uniform control matches PyTorch's default `nn.RNN` initialization.
:end_tab:

```{.python .input #lstm-delayed-recall-2}
%%tab pytorch
class RecallModel(nn.Module):  #@save
    """Embedding, a sequence layer, and a classifier at every position."""
    def __init__(self, layer, num_hiddens, vocab_size=16, num_inputs=32,
                 num_outputs=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_inputs)
        nn.init.normal_(self.embedding.weight, std=num_inputs ** -0.5)
        self.layer = layer
        self.head = init_lecun(nn.Linear(num_hiddens, num_outputs))

    def forward(self, X):  # Logits, (batch_size, num_steps, num_outputs)
        return self.head(self.layer(self.embedding(X)))

def init_keep_gate(rnn, bias=5.0):
    """Start an LSTM's forget gate, or a GRU's update gate, near one."""
    h = rnn.hidden_size
    with torch.no_grad():
        rnn.bias_ih_l0.zero_()  # Independent of the earlier initialization
        rnn.bias_hh_l0.zero_()
        rnn.bias_ih_l0[h:2 * h] = bias  # LSTM: i, f, c~, o; GRU: r, z, n
        if isinstance(rnn, nn.LSTM):
            rnn.bias_ih_l0[:h] = -bias  # Input gate near zero
    return rnn
```

```{.python .input #lstm-delayed-recall-2}
%%tab jax
class RecallModel(nnx.Module):  #@save
    """Embedding, a sequence layer, and a classifier at every position."""
    def __init__(self, layer, num_hiddens, vocab_size=16, num_inputs=32,
                 num_outputs=8, *, rngs):
        # d2l.Embed computes its gradient by a matrix product: no atomic adds
        # that wait on one another when a few filler tokens fill every row
        self.embedding = d2l.Embed(vocab_size, num_inputs, rngs=rngs,
            embedding_init=nnx.initializers.normal(num_inputs ** -0.5))
        self.layer = layer
        self.head = nnx.Linear(num_hiddens, num_outputs, rngs=rngs)

    def __call__(self, X):  # Logits, (batch_size, num_steps, num_outputs)
        return self.head(self.layer(self.embedding(X)))

def init_keep_gate(cell, bias=5.0):
    """Start an LSTM's forget gate, or a GRU's update gate, near one."""
    if isinstance(cell, nnx.LSTMCell):
        cell.hf.bias[...] = jnp.full_like(cell.hf.bias[...], bias)
        cell.hi.bias[...] = jnp.full_like(cell.hi.bias[...], -bias)  # Input
    else:  # GRUCell: input biases ordered r, z, n
        h = cell.dense_i.bias.shape[0] // 3
        cell.dense_i.bias[...] = cell.dense_i.bias[...].at[h:2 * h].set(bias)
    return cell
```

Notice what simple arithmetic predicts for the default zero-bias gated networks: when every gate starts at $1/2$, the direct path between step $1$ and step $T$ is attenuated by $2^{-(T-1)}$ in both directions—roughly $10^{-6}$ at $T = 20$ (still resolvable in float32) but $10^{-15}$ at $T = 50$. The state at step $T$ adds the symbol's contribution to the $O(1)$ contributions of recent steps, so anything below float32's $\sim 10^{-7}$ relative precision is rounded away before the readout sees it, and the gradient that returns to the symbol's embedding is just as small. We therefore expect default LSTM/GRU to succeed at $T = 20$ and fail by $T = 50$, while the keep-gate variants should succeed across all delays up to $T = 200$.

We save the training helpers `train_queries`, `query_accuracy`, and `train_delayed_recall` for reuse across all synthetic recall tasks in this chapter.

```{.python .input #lstm-delayed-recall-4}
%%tab pytorch
def train_queries(model, batch, num_steps, optimizer,  #@save
                  scheduler=None):
    """Train on fresh batches; the queries are the last Y.shape[1] steps."""
    device = next(model.parameters()).device
    for step in range(num_steps):
        X, Y = (torch.as_tensor(A, device=device) for A in batch(step))
        logits = model(X)[:, -Y.shape[1]:]  # Predictions at the queries
        loss = F.cross_entropy(logits.flatten(0, 1), Y.flatten())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def query_accuracy(model, X, Y):  #@save
    """Fraction of queries answered correctly."""
    device = next(model.parameters()).device
    X, Y = (torch.as_tensor(A, device=device) for A in (X, Y))
    with torch.inference_mode():
        return (model(X)[:, -Y.shape[1]:].argmax(-1) == Y).float().mean()

def train_delayed_recall(model, num_steps, blank_fillers=False):  #@save
    """Train with Adam at rate 0.003 on 1,500 fresh batches of 128 sequences
    of delayed recall; return the accuracy on 2,048 new sequences."""
    task = lambda rng, n: delayed_recall(rng, n, num_steps,
                                         blank_fillers=blank_fillers)
    rng = np.random.default_rng(0)
    model.to(d2l.try_gpu())
    train_queries(model, lambda step: task(rng, 128), 1500,
                  torch.optim.Adam(model.parameters(), lr=3e-3))
    return float(query_accuracy(model, *task(np.random.default_rng(1), 2048)))
```

```{.python .input #lstm-delayed-recall-4}
%%tab jax
@nnx.jit  #@save
def query_step(model, optimizer, X, Y):
    """One optimizer step on the cross-entropy at the last Y.shape[1] steps."""
    def loss_fn(model):
        logits = model(X)[:, -Y.shape[1]:]  # Predictions at the queries
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, Y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def train_queries(model, batch, num_steps, optimizer):  #@save
    """Train on fresh batches; the queries are the last Y.shape[1] steps."""
    step = nnx.cached_partial(query_step, model, optimizer)  # Compile once
    for i in range(num_steps):
        step(*batch(i))

@nnx.jit  #@save
def query_accuracy(model, X, Y):
    """Fraction of queries answered correctly."""
    return (model(X)[:, -Y.shape[1]:].argmax(-1) == Y).mean()

def train_delayed_recall(model, num_steps, blank_fillers=False):  #@save
    """Train with Adam at rate 0.003 on 1,500 fresh batches of 128 sequences
    of delayed recall; return the accuracy on 2,048 new sequences."""
    task = lambda rng, n: delayed_recall(rng, n, num_steps,
                                         blank_fillers=blank_fillers)
    rng = np.random.default_rng(0)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(3e-3)), wrt=nnx.Param)
    train_queries(model, lambda step: task(rng, 128), 1500, optimizer)
    return float(query_accuracy(model, *task(np.random.default_rng(1), 2048)))
```

Before training, let us also check the spectral radius and largest singular value of the two $64 \times 64$ plain RNN initializations:

```{.python .input #lstm-delayed-recall-8}
%%tab pytorch, jax
rng, n = np.random.default_rng(0), 64  # n hidden units
for name, W in (('LeCun normal', rng.normal(0, n ** -0.5, (n, n))),
                ('default scale', rng.uniform(-n ** -0.5, n ** -0.5, (n, n)))):
    print(f'{name:>13}: spectral radius {np.abs(np.linalg.eigvals(W)).max():.2f},'
          f' largest singular value {np.linalg.norm(W, 2):.2f}')
```

The LeCun normal matrix has spectral radius $1.02$ (sitting right at the $\rho \approx 1$ borderline where leading eigenmodes neither vanish nor explode when states are small), whereas the uniform default-scale matrix has spectral radius $0.62$ (shrinking states by $\sim 0.6$ per step). Now we train two seeds of all six models across delays $T \in \{20, 50, 100, 200\}$.

```{.python .input #lstm-delayed-recall-5}
%%tab pytorch
def recurrent(layer, keep_gate=False):
    """A batch-first layer, 32 inputs to 64 hidden units, LeCun-initialized."""
    layer = init_lecun(layer(32, 64, batch_first=True))
    return init_keep_gate(layer) if keep_gate else layer

def rnn_default_scale():  # The recurrent matrix as nn.RNN draws it
    rnn = recurrent(nn.RNN)
    nn.init.uniform_(rnn.weight_hh_l0, -64 ** -0.5, 64 ** -0.5)
    return rnn

layers = {'RNN': lambda: recurrent(nn.RNN),
          'RNN, default scale': rnn_default_scale,
          'LSTM': lambda: recurrent(nn.LSTM), 'GRU': lambda: recurrent(nn.GRU),
          'LSTM, keep gate': lambda: recurrent(nn.LSTM, keep_gate=True),
          'GRU, keep gate': lambda: recurrent(nn.GRU, keep_gate=True)}
delays = [20, 50, 100, 200]
print(' ' * 18, ' '.join(f'{T:>9d}' for T in delays))
for name, layer in layers.items():
    runs = [[train_delayed_recall(RecallModel(Recurrent(layer()), 64), T)
             for _ in range(2)] for T in delays]
    print(f'{name:18s}', ' '.join(f'{a:.2f}/{b:.2f}' for a, b in runs))
```

```{.python .input #lstm-delayed-recall-5}
%%tab jax
lecun = {'recurrent_kernel_init': nnx.initializers.lecun_normal()}
default = {'recurrent_kernel_init': nnx.initializers.variance_scaling(
    1 / 3, 'fan_in', 'uniform')}  # Uniform on [-1/8, 1/8]
cells = {'RNN': lambda rngs: nnx.SimpleCell(32, 64, rngs=rngs, **lecun),
         'RNN, default scale':
             lambda rngs: nnx.SimpleCell(32, 64, rngs=rngs, **default),
         'LSTM': lambda rngs: nnx.LSTMCell(32, 64, rngs=rngs, **lecun),
         'GRU': lambda rngs: nnx.GRUCell(32, 64, rngs=rngs, **lecun),
         'LSTM, keep gate': lambda rngs: init_keep_gate(
             nnx.LSTMCell(32, 64, rngs=rngs, **lecun)),
         'GRU, keep gate': lambda rngs: init_keep_gate(
             nnx.GRUCell(32, 64, rngs=rngs, **lecun))}

def recall_model(cell):  # Fresh random weights for every model
    rngs = nnx.Rngs(d2l.get_key())
    return RecallModel(nnx.RNN(cell(rngs), unroll=8), 64, rngs=rngs)

delays = [20, 50, 100, 200]
print(' ' * 18, ' '.join(f'{T:>9d}' for T in delays))
for name, cell in cells.items():
    runs = [[train_delayed_recall(recall_model(cell), T) for _ in range(2)]
            for T in delays]
    print(f'{name:18s}', ' '.join(f'{a:.2f}/{b:.2f}' for a, b in runs))
```

The results confirm the analysis for the GRU and for every delay $T \ge 50$, and they correct it for the LSTM:

- **Default vs. keep-gate LSTM and GRU:** With zero gate biases, the GRU succeeds at $T = 20$ and both cells collapse to chance ($0.12$) at $T \ge 50$. The default LSTM, however, is already unreliable at $T = 20$, where the arithmetic predicted success. That estimate counted only the forget gates. We did not isolate the cause, but the likely one is that the LSTM's symbol must also enter the cell through a half-open input gate and leave it through a half-open output gate and a $\tanh$, which puts $T = 20$ at the edge of what training can recover. With `init_keep_gate`, both achieve $1.00$ accuracy across every run at every delay up to $T = 200$.
- **Borderline vs. contracting plain RNN:** Because the LeCun normal RNN starts with spectral radius $\rho \approx 1$, some seeds carry the symbol surprisingly far (even to $T = 100$), though unreliably across seeds, whereas the $\rho \approx 0.62$ default-scale RNN is hit-or-miss up to $T = 50$ and fails on every seed from $T = 100$ onward. Gates with well-initialized biases turn hit-or-miss spectral retention into dependable, per-unit memory control.

### Gated Networks on Text
:label:`subsec_deep-rnn`

In :numref:`subsec_gpt-rnn-baseline`, a single wide plain RNN fell behind a multi-block GPT after the first 16 bytes of context. To compare sequence mixers on equal footing—without confounding recurrence against residual depth and positionwise MLPs—we place every mixer in this chapter inside the same pre-norm residual backbone (`SeqLM`), matching :numref:`sec_transformer-block`: token embedding, a stack of blocks each applying `H = H + mixer(norm(H))` followed by a two-layer GELU MLP `H = H + mlp(H)`, and a final layer norm and linear output head (:numref:`fig_seqlm_backbone`).

![The shared backbone `SeqLM`. A block adds the output of a mixer, and then of an MLP, to the residual stream, each after its own layer normalization. Only the mixer changes from one model of this chapter to the next; the language models stack $L = 4$ blocks.](../img/mdl-modernrnn-seqlm-backbone.svg)
:label:`fig_seqlm_backbone`

```{.python .input #lstm-deep-recurrent-networks-1}
%%tab pytorch
class SeqLM(nn.Module):  #@save
    """A language model of residual blocks: a sequence mixer, then an MLP."""
    def __init__(self, vocab_size, num_hiddens, mixers):
        super().__init__()
        d = num_hiddens
        self.embedding = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embedding.weight, std=d ** -0.5)  # Unit length
        self.mixers = nn.ModuleList(mixers)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in mixers])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
            nn.Linear(4 * d, d)) for _ in mixers])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        for module in (self.mlps, self.head):  # Mixers initialize themselves
            init_lecun(module)

    def forward(self, X):
        H = self.embedding(X)  # (batch_size, num_steps, num_hiddens)
        for norm, mixer, mlp in zip(self.norms, self.mixers, self.mlps):
            H = H + mixer(norm(H))
            H = H + mlp(H)
        return self.head(self.norm(H))  # (batch_size, num_steps, vocab_size)
```

```{.python .input #lstm-deep-recurrent-networks-1}
%%tab jax
class SeqLM(nnx.Module):  #@save
    """A language model of residual blocks: a sequence mixer, then an MLP."""
    def __init__(self, vocab_size, num_hiddens, mixers, *, rngs):
        d = num_hiddens
        self.embedding = d2l.Embed(vocab_size, d, rngs=rngs,  # Unit length
            embedding_init=nnx.initializers.normal(d ** -0.5))
        self.mixers = nnx.List(mixers)
        norm = lambda: nnx.LayerNorm(d, epsilon=1e-5, rngs=rngs)
        self.norms = nnx.List([norm() for _ in mixers])
        self.mlps = nnx.List([nnx.Sequential(
            norm(), nnx.Linear(d, 4 * d, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),  # Exact GELU
            nnx.Linear(4 * d, d, rngs=rngs)) for _ in mixers])
        self.norm = norm()
        self.head = nnx.Linear(d, vocab_size, rngs=rngs)

    def __call__(self, X):
        H = self.embedding(X)  # (batch_size, num_steps, num_hiddens)
        for norm, mixer, mlp in zip(self.norms, self.mixers, self.mlps):
            H = H + mixer(norm(H))
            H = H + mlp(H)
        return self.head(self.norm(H))  # (batch_size, num_steps, vocab_size)
```

Our attention baseline within `SeqLM` is `AttentionMixer`, which wraps causal multi-head attention with RoPE (:numref:`sec_gpt`, :numref:`sec_positional-information`), LeCun normal initialization, and an optional depthwise causal convolution over $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ (used later in :numref:`sec_hybrids`).

```{.python .input #lstm-gated-networks-on-text-1}
%%tab pytorch
class AttentionMixer(d2l.CausalAttention):  #@save
    """Causal multi-head attention, with rotary positions unless rope is
    False, as a sequence mixer; with conv_width, a causal convolution per
    channel first mixes each query, key, and value with those of the
    previous steps."""
    def __init__(self, num_hiddens, num_heads, conv_width=None, rope=True):
        super().__init__(num_hiddens, num_heads, rope=rope)
        d = 3 * num_hiddens  # Channels of q, k, and v; one filter each
        self.conv = None if conv_width is None else nn.Conv1d(
            d, d, conv_width, groups=d, padding=conv_width - 1)
        init_lecun(self)  # Like every mixer of the chapter

    def qkv(self, X, offset=0, tables=None):
        B, T, _ = X.shape
        QKV = self.W_qkv(X)
        if self.conv is not None:  # Causal: drop the last conv_width - 1 steps
            QKV = self.conv(QKV.mT)[..., :T].mT
        q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
                   for u in QKV.chunk(3, -1))
        if self.rope:
            q, k = (d2l.rope(u, offset, tables=tables) for u in (q, k))
        return q, k, v  # Each (batch_size, num_heads, num_steps, head width)
```

```{.python .input #lstm-gated-networks-on-text-1}
%%tab jax
class AttentionMixer(d2l.CausalAttention):  #@save
    """Causal multi-head attention, with rotary positions unless rope is
    False, as a sequence mixer; with conv_width, a causal convolution per
    channel first mixes each query, key, and value with those of the
    previous steps."""
    def __init__(self, num_hiddens, num_heads, conv_width=None, rope=True,
                 *, rngs):
        super().__init__(num_hiddens, num_heads, rope=rope, rngs=rngs)
        lecun = nnx.initializers.lecun_normal()
        for layer in (self.W_qkv, self.W_o):  # Like every mixer of the chapter
            layer.kernel[...] = lecun(rngs.params(), layer.kernel.shape)
        d = 3 * num_hiddens  # Channels of q, k, and v; one filter each
        self.conv = None if conv_width is None else nnx.Conv(
            d, d, (conv_width,), feature_group_count=d,
            padding=((conv_width - 1, 0),), rngs=rngs)

    def qkv(self, X, offset=0, tables=None):
        B, T, _ = X.shape
        QKV = self.W_qkv(X)
        if self.conv is not None:  # Causal: padded on the left only
            QKV = self.conv(QKV)
        q, k, v = (u.reshape(B, T, self.num_heads, -1).swapaxes(1, 2)
                   for u in jnp.split(QKV, 3, -1))
        if self.rope:
            q, k = (d2l.rope(u, offset, tables=tables) for u in (q, k))
        return q, k, v  # Each (batch_size, num_heads, num_steps, head width)
```

We also save three evaluation helpers used across the chapter. First, `loss_by_position` computes the mean validation cross-entropy in nats at each position of the context window:

```{.python .input #lstm-gated-networks-on-text-2}
%%tab pytorch
@torch.inference_mode()  #@save
def loss_by_position(model, data):
    """Mean validation loss in nats at each position of the window."""
    device = next(model.parameters()).device
    total, count = 0, 0
    for X, Y in data.val_dataloader():
        logits = model(X.to(device))  # (batch, T, V)
        total += F.cross_entropy(logits.transpose(1, 2), Y.to(device),
                                 reduction='none').sum(0)
        count += len(X)
    return (total / count).tolist()
```

```{.python .input #lstm-gated-networks-on-text-2}
%%tab jax
@nnx.jit  #@save
def position_loss_sum(model, X, Y):
    """Loss in nats at each position, summed over the batch."""
    return optax.softmax_cross_entropy_with_integer_labels(model(X), Y).sum(0)

def loss_by_position(model, data):  #@save
    """Mean validation loss in nats at each position of the window."""
    total, count = 0, 0
    for X, Y in data.val_dataloader():
        total += position_loss_sum(model, jnp.asarray(X), jnp.asarray(Y))
        count += len(X)
    return (total / count).tolist()
```

Next, `train_text` trains a model for 1,000 steps of AdamW (100-step warmup to $2 \times 10^{-3}$ followed by cosine decay), and `text_table` runs two seeds per configuration to report parameter count, overall and position-band (`0-15`, `16-63`, `64-127`) validation losses, and milliseconds per step:

```{.python .input #lstm-gated-networks-on-text-3}
%%tab pytorch, jax
def train_text(model, data, lr=2e-3, num_steps=1000):  #@save
    """Train a language model with LMOptimizer; return its validation loss
    in nats at each position and its milliseconds per training step."""
    optimizer = d2l.LMOptimizer(model, lr=lr, num_steps=num_steps, warmup=100)
    # One untimed call of 50 steps warms up (JAX compiles), the rest is timed
    bench = d2l.Benchmark(lambda: d2l.train_steps(model, data, optimizer, 50),
                          warmup=1, repeats=num_steps // 50 - 1)
    return loss_by_position(model, data), 1000 * bench.time / 50

def text_table(make_model, arms, data, num_runs=2, num_steps=1000):  #@save
    """Train num_runs fresh models make_model(*args) for each arm (a name
    and its args) for num_steps steps; print the parameters, the validation loss of every run,
    its losses at early, middle, and late positions of the window, and the
    milliseconds per step. Return the trained models of each arm."""
    T = data.num_steps
    bands = [(0, T // 8), (T // 8, T // 2), (T // 2, T)]
    width = max(len(name) for name in arms)
    header = ['validation'] + [f'{a}-{b - 1}' for a, b in bands]
    print(' ' * (width + 12), *(f'{h:>11}' for h in header), '  ms/step')
    trained = {}
    for name, args in arms.items():
        models = [make_model(*args) for _ in range(num_runs)]
        runs = [train_text(model, data, num_steps=num_steps)  # (losses, ms)
                for model in models]
        columns = [[np.mean(loss) for loss, _ in runs]] + [
            [np.mean(loss[a:b]) for loss, _ in runs] for a, b in bands]
        print(f'{name:{width}s} {d2l.num_params(models[0]):11,d}',
              *('/'.join(f'{v:.3f}' for v in column) for column in columns),
              '/'.join(f'{ms:.0f}' for _, ms in runs))
        trained[name] = models
    return trained
```

We now benchmark our recurrent and attention mixers on byte-level WikiText-103 (`d2l.WikiText103`, windows of 128 bytes, batch size 64):

```{.python .input #lstm-gated-networks-on-text-4}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
```

Each `text_model` stacks 4 residual blocks at width $d = 256$—comparing a plain `RNN`, an `LSTM`, a `GRU`, and 8-head causal `attention`—plus a parameter-matched plain RNN widened to $d = 302$ ($3.81\textrm{M}$ parameters, matching the GRU stack):

```{.python .input #lstm-gated-networks-on-text-5}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32

def text_model(mixer, d):  # 256 byte values; four blocks of width d
    return SeqLM(256, d, [mixer(d) for _ in range(4)]).to(d2l.try_gpu())

rnn = lambda d: Recurrent(init_lecun(nn.RNN(d, d, batch_first=True)))
lstm = lambda d: Recurrent(init_lecun(nn.LSTM(d, d, batch_first=True)))
gru = lambda d: Recurrent(init_lecun(nn.GRU(d, d, batch_first=True)))
attention = lambda d: AttentionMixer(d, num_heads=8)
arms = {'RNN': (rnn, 256), 'LSTM': (lstm, 256), 'GRU': (gru, 256),
        'attention': (attention, 256),
        'RNN, d=302': (rnn, 302)}  # As many parameters as the GRU
```

```{.python .input #lstm-gated-networks-on-text-5}
%%tab jax
def text_model(mixer, d):  # 256 byte values; four blocks of width d
    rngs = nnx.Rngs(d2l.get_key())
    return SeqLM(256, d, [mixer(d, rngs) for _ in range(4)], rngs=rngs)

rnn = lambda d, rngs: nnx.RNN(nnx.SimpleCell(d, d, rngs=rngs, **lecun),
                              unroll=8)
lstm = lambda d, rngs: nnx.RNN(nnx.LSTMCell(d, d, rngs=rngs, **lecun),
                               unroll=8)
gru = lambda d, rngs: nnx.RNN(nnx.GRUCell(d, d, rngs=rngs, **lecun),
                              unroll=8)
attention = lambda d, rngs: AttentionMixer(d, num_heads=8, rngs=rngs)
arms = {'RNN': (rnn, 256), 'LSTM': (lstm, 256), 'GRU': (gru, 256),
        'attention': (attention, 256),
        'RNN, d=302': (rnn, 302)}  # As many parameters as the GRU
```

Calling `text_table` trains two runs of each arm and prints the side-by-side comparison:

```{.python .input #lstm-gated-networks-on-text-6}
%%tab pytorch, jax
trained = text_table(text_model, arms, data)
```

Three findings stand out from the table:

1. **Residual stacking + gating:** Simply placing 4 narrow $\tanh$ RNN layers inside a pre-norm residual + MLP backbone reaches $\sim 1.35$ nats/byte—already matching the 6-block SwiGLU GPT of :numref:`sec_gpt`. Replacing the plain RNN mixer with an LSTM or GRU lowers validation loss by a further $0.05$ nats/byte (to $\sim 1.30$), whereas widening the plain RNN to match the GRU's parameter count (`RNN, d=302`) closes less than a third of that gap (about $1.33$).
2. **Short-context utilization:** Across all models in this 1,000-step, 128-byte regime, loss drops sharply from band `0-15` to `16-63` and changes by less than $0.005$ beyond byte 16. Here the gates earn their lead by managing short-range byte context more effectively rather than by carrying 100-step memories; the attention stack (about $1.38$) trails the recurrent mixers already in the first 16 bytes at this small training budget.
3. **Step cost:** Despite fused library loops, the LSTM and GRU take $1.4\times$–$3\times$ longer per training step than the plain RNN. The cause is the work inside each step: three or four gate products where the plain RNN computes one. It is not the sequential evaluation, because the plain RNN waits for the previous step's hidden state $\mathbf{h}_{t-1}$ in the same way. At 128 steps that dependence is not yet the dominant cost, so this table cannot show it. The length sweep of :numref:`subsec_linear-recurrence-cost` isolates it: there the GRU's step time grows more than fifteenfold between 512 and 16,384 tokens.

## Summary

In a plain recurrent network, signals and gradients pass through a product of step Jacobians governed by a single shared matrix $\mathbf{W}_\textrm{hh}$, surviving over long horizons only when its spectral radius sits near $1$. Gated recurrent networks introduce an elementwise additive memory path $c_t = f_t c_{t-1} + i_t \tilde{c}_t$ controlled by learned sigmoid gates. The LSTM separates an internal memory cell $\mathbf{c}_t$ (guarded by input, forget, and output gates) from the exposed state $\mathbf{h}_t$, while the GRU couples the forget and input gates ($1 - \mathbf{z}_t$) and filters the recurrent state via a reset gate $\mathbf{r}_t$ on a single state vector. On synthetic delayed recall, initializing forget/update gates near $1$ (`init_keep_gate`) is decisive, turning chance accuracy at $T \ge 50$ into $100\%$ recall at $T = 200$. On byte-level WikiText-103 inside a residual backbone (`SeqLM`), LSTM and GRU mixers outperform both equal-width and parameter-matched plain RNNs by $0.03$–$0.05$ nats/byte, but their dependence on $\mathbf{h}_{t-1}$ inside the gates still forces $T$ sequential steps during training.

## Exercises

1. **Holding a value.** Suppose that the character at step $s$ determines
   the prediction at a much later step $t$, and that one LSTM unit is to
   carry it from $s$ to $t$.
    1. State the value that each gate of that unit in :eqref:`lstm_gates`
       should take at step $s$, at the steps strictly between $s$ and $t$,
       and at step $t$. Then do the same for the update gate of a GRU unit
       in :eqref:`gru_H`.
    1. A sigmoid gate never reaches 1. Suppose that the forget gate's
       pre-activation stays at a constant $b$, so that its value is
       $\sigma(b)$ at every step. What fraction of the stored value survives
       $k$ steps along the direct path? How large must $b$ be for 90% of
       the value to survive 1,000 steps?
    1. Show that a forget-gate bias of $\log(T - 1)$ gives an initial
       forget gate of $1 - 1/T$ when the weights are zero, so that for large
       $T$ a stored value decays by a factor of about $e$ over $T$ steps.
       The *chrono initialization* of :citet:`Tallec.Ollivier.2018` draws
       the forget-gate bias of each unit as $\log u$, with $u$ uniform on
       $[1, T_\textrm{max} - 1]$, and sets $\mathbf{b}_\textrm{i} =
       -\mathbf{b}_\textrm{f}$. Show that with zero weights this makes
       $\mathbf{f}_t + \mathbf{i}_t = 1$, and compare the resulting cell
       update with :eqref:`gru_H`.

1. **Output squashing.** The input node $\tilde{\mathbf{c}}_t$ is already
   squashed by a $\tanh$, yet :eqref:`lstm_update` applies $\tanh$ to
   $\mathbf{c}_t$ again before the output gate.
    1. Starting from $\mathbf{c}_0 = \mathbf{0}$, show that every entry of
       $\mathbf{c}_t$ is smaller than $t$ in magnitude. Show further that if
       every forget-gate value is at most $f_\textrm{max} < 1$ and every
       input-gate value at most $i_\textrm{max}$, every entry stays below
       $i_\textrm{max} / (1 - f_\textrm{max})$ in magnitude, whatever $t$
       is.
    1. Suppose that the output were $\mathbf{h}_t = \mathbf{o}_t \odot
       \mathbf{c}_t$ instead. After many steps of accumulation, what happens
       to the gate pre-activations at the next step, and what does that do
       to the gradients that flow through the gates?

1. **Reset before or after.** The fused GRU layers of deep learning
   frameworks apply the reset gate after the recurrent matrix product. Apart
   from the placement of the bias terms, they compute the candidate
   $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_\textrm{xh} \mathbf{x}_t + \mathbf{r}_t \odot (\mathbf{W}_\textrm{hh} \mathbf{h}_{t-1}) + \mathbf{b}_\textrm{h})$$
   in place of :eqref:`gru_tilde_H`.
    1. Show that the two candidates agree for every $\mathbf{r}_t$ and
       $\mathbf{h}_{t-1}$ if and only if $\mathbf{W}_\textrm{hh}$ is
       diagonal. For a general $\mathbf{W}_\textrm{hh}$, which reset-gate
       values still make them agree?
    1. The input products $\mathbf{W}_{\textrm{x}\cdot} \mathbf{x}_t$ of all
       steps can be computed as one large product before the recurrence
       starts. Under each form of the candidate, how many matrix products
       involving the previous state must each step perform one after the
       other? Why do fused GRU kernels use the form above?

:begin_tab:`pytorch`
4. [code] **GRU from scratch.** Write a class `GRUScratch` by editing
   `LSTMScratch`: three heads following :eqref:`gru_gates` and
   :eqref:`gru_tilde_H`, the update :eqref:`gru_H`, and a state that
   consists of $\mathbf{h}_t$ alone.
    1. Copy the weights of the library GRU layer into your layer, as we did
       for the LSTM, and compare the outputs. They will not agree. Use the
       previous exercise to change one line of your layer so that they do.
       (Also keep the last third of `bias_hh_l0`, the recurrent
       bias of the candidate, as a separate vector inside the reset gate,
       or zero it before copying.)
    1. Remove the reset gate by fixing $\mathbf{r}_t = \mathbf{1}$, and in
       a second layer remove the update gate by fixing $\mathbf{z}_t =
       \mathbf{0}$. Before training, predict from :eqref:`gru_H` which
       removal hurts delayed recall more. Then run the experiment of
       :numref:`subsec_delayed-recall` and check your prediction.
:end_tab:

:begin_tab:`jax`
4. [code] **GRU from scratch.** Write a class `GRUScratch` by editing
   `LSTMScratch`: three heads following :eqref:`gru_gates` and
   :eqref:`gru_tilde_H`, the update :eqref:`gru_H`, and a state that
   consists of $\mathbf{h}_t$ alone.
    1. Copy the weights of the library GRU layer into your layer, as we did
       for the LSTM, and compare the outputs. They will not agree. Use the
       previous exercise to change one line of your layer so that they do.
    1. Remove the reset gate by fixing $\mathbf{r}_t = \mathbf{1}$, and in
       a second layer remove the update gate by fixing $\mathbf{z}_t =
       \mathbf{0}$. Before training, predict from :eqref:`gru_H` which
       removal hurts delayed recall more. Then run the experiment of
       :numref:`subsec_delayed-recall` and check your prediction.
:end_tab:

5. [code] **The cell's other paths.** ● Consider an LSTM with one input and
   one hidden unit, so that every quantity in :eqref:`lstm_gates` and
   :eqref:`lstm_update` is a scalar, and write $w_\textrm{hf}$,
   $w_\textrm{hi}$, and $w_\textrm{hc}$ for the recurrent weights of the
   forget gate, the input gate, and the input node.
    1. The output gate $o_{t-1}$ depends on $h_{t-2}$ but not on $c_{t-1}$.
       Using $h_{t-1} = o_{t-1} \tanh(c_{t-1})$, derive the total derivative
       $\textrm{d}c_t / \textrm{d}c_{t-1}$. Write it as the direct-path value
       $f_t$ plus a correction, and factor the correction into a term from
       step $t$ and the derivative $\textrm{d}h_{t-1} / \textrm{d}c_{t-1}$.
    1. For which values of the gates, of $\tilde{c}_t$, and of $c_{t-1}$
       does the correction vanish? Which of these conditions can the network
       set through its gates while the cell is storing a value?
    1. Train an LSTM on delayed recall at a delay where it succeeds. For one
       test sequence, compute the norm of the gradient of the loss with
       respect to the cell $\mathbf{c}_{T-k}$ for every $k$, by automatic
       differentiation, and compare it with the gradient carried by the
       direct path alone, which multiplies the gradient at $\mathbf{c}_T$
       elementwise by the product of the forget-gate values over the last
       $k$ steps. Where do the two curves part, and which is larger?

1. [code] **Bidirectional leak.** ● A recurrent layer can also read the
   sequence backward. A *bidirectional* network
   :cite:`Schuster.Paliwal.1997` runs a second layer from the last step to
   the first and concatenates the outputs of the two, so that the
   representation of each position depends on the whole sequence. This
   helps whenever the whole input is given, as in tagging the words of a
   sentence or encoding a sentence for translation. A language model is
   another matter.
    1. Build a bidirectional mixer by running a second LSTM over the
       reversed sequence, reversing its outputs, concatenating them with
       those of a forward LSTM, and projecting the result back to width
       $d$. Train `SeqLM` with such mixers in place of the LSTMs of
       :numref:`subsec_deep-rnn`, with `train_text`, and compare its
       validation loss with that of the forward model. Explain the number:
       which input does the backward layer read at position $t$? What
       could the model do at generation time, when the following bytes do
       not exist yet?
    1. A leak-free bidirectional predictor of the byte $x_t$ combines the
       forward state after step $t-1$ with the backward state after step
       $t+1$, so it models the probability of $x_t$ given all other bytes
       of the window. Show with a sequence of two binary tokens that the
       product of these conditionals over $t$ need not sum to one over all
       sequences. What does this imply for using the product to score
       whole sequences, or to report a perplexity?

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1057)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/18016)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.1]{.kicker}

Gated Recurrent Networks: LSTM and GRU<br>**vanishing gradients · sigmoid decay gates · memory cells · gate bias initialization · residual sequence mixers**
:::
:::

::: {.slide title="Why Plain RNNs Lose Long-Range Grammatical Context"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{h}_t = \tanh(\mathbf{W}_\textrm{hh}\mathbf{h}_{t-1} + \mathbf{W}_\textrm{xh}\mathbf{x}_t + \mathbf{b}_\textrm{h})$$

- **Full-State Overwriting:** Every step multiplies $\mathbf{h}_{t-1}$ by $\mathbf{W}_\textrm{hh}$ and squashes it through $\tanh$.
- **Geometric Signal Decay:** At $w = 0.9$, perturbations shrink to $0.9^{30} \approx 4\%$ across $30$ steps and $0.9^{100} \approx 0.003\%$ across $100$ steps.
:::

::: {.col .fig}
@fig:ch12-mrnn-grammar-decay
:::
:::
:::

::: {.slide title="Jacobian Products and the Spectral Radius Borderline"}
[Vanishing & Exploding Gradients]{.kicker}

::: {.cols .vc}
::: {.col}
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-k}} = \prod_{s=t-k+1}^{t} \textrm{diag}(1 - \mathbf{h}_s^2)\,\mathbf{W}_\textrm{hh}$$

- **Contractive vs. Expansive Regimes:** Since $|1 - h_s^2| \le 1$, powers scale as $\rho^k$ ($\rho = \max_i |\lambda_i(\mathbf{W}_\textrm{hh})|$)—vanishing when $\rho < 1$ and exploding when $\rho > 1$.
- **Asymmetry of Failure:** Clipping caps explosions, but cannot recover underflowed long-lag gradients.
:::

::: {.col .fig}
@fig:ch12-mrnn-jacobian-chain
:::
:::
:::

::: {.slide title="From Unbounded Running Sums to Learned Decay Gates"}
[The Gating Principle]{.kicker}

::: {.cols .vc}
::: {.col}
$$c_t = f_t\, c_{t-1} + i_t\, \tilde{c}_t, \qquad \tau_{1/2} = \frac{\ln 2}{\ln(1/f)}$$

- **Elementwise Gradient Highway:** Along the direct memory rail, $\partial c_t / \partial c_{t-k} = \prod_{s=t-k+1}^t f_s \le 1$ with no $\mathbf{W}_\textrm{hh}$ matrix product.
- **Tunable Horizon:** Holding $f_t \approx 0.99$–$0.999$ stretches the memory half-life from $1$ step to $69$–$693$ steps.
:::

::: {.col .fig}
@fig:ch12-mrnn-gate-halflife
:::
:::
:::

::: {.slide title="The Long Short-Term Memory (LSTM) Architecture"}
[LSTM Mechanics]{.kicker}

The LSTM (Hochreiter & Schmidhuber, 1997; Gers et al., 2000) maintains an internal **memory cell** $\mathbf{c}_t \in \mathbb{R}^d$ and an exposed **hidden state** $\mathbf{h}_t \in \mathbb{R}^d$:

::: {.cols .vc}
::: {.col}
- **Three sigmoid gates in $(0, 1)^d$:** Input gate $\mathbf{i}_t$, forget gate $\mathbf{f}_t$, and output gate $\mathbf{o}_t$ each apply $\sigma(\mathbf{W}_{\textrm{x}\cdot}\mathbf{x}_t + \mathbf{W}_{\textrm{h}\cdot}\mathbf{h}_{t-1} + \mathbf{b}_\cdot)$.
- **Candidate input node:** $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_\textrm{xc}\mathbf{x}_t + \mathbf{W}_\textrm{hc}\mathbf{h}_{t-1} + \mathbf{b}_\textrm{c}) \in (-1, 1)^d$.
- **Additive cell update & gated readout:**
  $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
  with exposed state $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$.
:::

::: {.col .fig}
![](../img/mdl-modernrnn-lstm-cell.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Decoupling Internal Storage from Exposed Output"}
[Silent Retention & Gradient Highway]{.kicker}

::: {.cols .vc}
::: {.col}
- **Silent Storage Until Needed:** Holding $f_t = 0.95, i_t = 0.1$ preserves $c_t = 0.71$ while a closed output gate ($o_t = 0.2$) shields downstream layers ($h_t = 0.12$).
- **Bounding Cell Magnitude:** Squashing $\tanh(\mathbf{c}_t)$ at readout keeps $\mathbf{h}_t \in (-1, 1)^d$ even when repeated writes push $|c_t| > 1$.
- **Direct Cell Jacobian:** Along the top cell rail, $\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1} = \textrm{diag}(\mathbf{f}_t)$.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-silent-storage
:::
:::
:::

::: {.slide title="Fused 4-Gate Matrix Projection and Verification" only="pytorch"}
[LSTM Implementation]{.kicker}

Because $\mathbf{i}_t$, $\mathbf{f}_t$, $\tilde{\mathbf{c}}_t$, and $\mathbf{o}_t$ all read $(\mathbf{x}_t, \mathbf{h}_{t-1})$, stacking their weights into $\mathbf{W}_\textrm{x} \in \mathbb{R}^{d_\textrm{in} \times 4d}$ and $\mathbf{W}_\textrm{h} \in \mathbb{R}^{d \times 4d}$ computes all four pre-activations in one batched matrix product per step:

```{.python #lstm-exercises-1}
i, f, c_tilde, o = (x @ self.W_x + h @ self.W_h + self.b).chunk(4, -1)
c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(c_tilde)
h = torch.sigmoid(o) * torch.tanh(c)
```

@!lstm-implementation-2

::: {.d2l-note}
Our from-scratch loop matches fused `nn.LSTM` to `float32` rounding error ($\sim 10^{-7}$), so we use fused library kernels for training.
:::
:::

::: {.slide title="Fused 4-Gate Matrix Projection and Verification" only="jax"}
[LSTM Implementation]{.kicker}

Because $\mathbf{i}_t$, $\mathbf{f}_t$, $\tilde{\mathbf{c}}_t$, and $\mathbf{o}_t$ all read $(\mathbf{x}_t, \mathbf{h}_{t-1})$, stacking their weights into $\mathbf{W}_\textrm{x} \in \mathbb{R}^{d_\textrm{in} \times 4d}$ and $\mathbf{W}_\textrm{h} \in \mathbb{R}^{d \times 4d}$ computes all four pre-activations in one batched matrix product per step:

```{.python #lstm-exercises-2}
i, f, c_tilde, o = jnp.split(
    x @ self.W_x[...] + h @ self.W_h[...] + self.b[...], 4, -1)
c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(c_tilde)
h = jax.nn.sigmoid(o) * jnp.tanh(c)
```

@!lstm-implementation-2

::: {.d2l-note}
Our from-scratch loop matches `nnx.LSTMCell` to `float32` rounding error ($\sim 10^{-7}$), so we use fused library kernels for training.
:::
:::

::: {.slide title="Gated Recurrent Units (GRU): Coupled Reset and Update"}
[GRU Architecture]{.kicker}

The GRU (Cho et al., 2014) merges the cell and hidden state into a single vector $\mathbf{h}_t \in \mathbb{R}^d$ governed by two gates:

::: {.cols .vc}
::: {.col}
- **Reset gate $\mathbf{r}_t$ & update gate $\mathbf{z}_t$:** Both computed via $\sigma(\mathbf{W}_{\textrm{x}\cdot}\mathbf{x}_t + \mathbf{W}_{\textrm{h}\cdot}\mathbf{h}_{t-1} + \mathbf{b}_\cdot)$.
- **Reset-filtered candidate:** $\mathbf{r}_t$ gates how much past state enters the nonlinear proposal:
  $$\tilde{\mathbf{h}}_t = \tanh\big(\mathbf{W}_\textrm{xh}\mathbf{x}_t + \mathbf{W}_\textrm{hh}(\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_\textrm{h}\big)$$
- **Convex interpolation ($i_t = 1 - f_t$):**
  $$\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$$
:::

::: {.col .fig}
![](../img/mdl-modernrnn-gru-cell.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="LSTM vs. GRU: Structural and Systems Trade-Offs"}
[Architectural Comparison]{.kicker}

::: {.cols .vc}
::: {.col}
- **State & Gate Count:** LSTM carries two vectors $(\mathbf{c}_t, \mathbf{h}_t)$ controlled by 3 independent gates ($8d^2$ params); GRU carries one vector $\mathbf{h}_t$ with 2 gates ($6d^2$ params, $25\%$ smaller).
- **Bounding Mechanism:** LSTM bounds $\mathbf{h}_t$ via readout $\tanh(\mathbf{c}_t)$, whereas GRU's convex blend $\mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$ keeps $\mathbf{h}_t \in [-1, 1]^d$ automatically.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-lstm-vs-gru
:::
:::
:::

::: {.slide title="Synthetic Stress Test: The Delayed Recall Benchmark"}
[Experimental Setup]{.kicker}

Gated equations *represent* long memories, but does gradient descent actually discover them? We test retention on synthetic **delayed recall**:

- **Task structure:** Step $1$ presents a target symbol drawn from $\{0, \ldots, 7\}$, followed by $T - 1$ random filler tokens drawn from $\{8, \ldots, 15\}$. At step $T$, the network must classify the initial symbol (chance accuracy $= 1/8 = 0.125$).
- **Architecture (`RecallModel`):** Token embedding ($d_\textrm{in} = 32$) $\to$ single recurrent layer ($d = 64$, LeCun normal weights) $\to$ linear classifier at step $T$, trained for $1{,}500$ steps of Adam ($\text{lr} = 3 \times 10^{-3}$, batch size $128$).

@!lstm-delayed-recall-1
:::

::: {.slide title="Pitfall: Zero-Bias Gates Start with a 1-Step Memory"}
[Initialization Hazard]{.kicker}

::: {.cols .vc}
::: {.col}
- **One-Step Initial Half-Life:** With $\mathbf{b}_\textrm{f} = \mathbf{0}$, gates start at $\sigma(0) = 1/2$, attenuating the direct gradient from step $T$ to step $1$ by $2^{-(T-1)}$.
- **Underflow Wall at $T \ge 50$:** While $2^{-19} \approx 1.9 \times 10^{-6}$ at $T=20$ survives in `float32`, $2^{-49} \approx 1.8 \times 10^{-15}$ at $T=50$ is rounded away when the state adds it to the $O(1)$ contributions of recent steps.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-zero-bias-underflow
:::
:::
:::

::: {.slide title="Keep-Gate Initialization and Cell Magnitude Bounding"}
[Chrono & Keep-Gate Bias]{.kicker}

::: {.cols .vc}
::: {.col}
- **100-Step Initial Horizon:** Setting $b_\textrm{f} = +5$ starts gates at $\sigma(5) \approx 0.9933$ ($\tau_{1/2} \approx 103$ steps), preserving $0.9933^{50} \approx 71\%$ of the gradient across $50$ steps.
- **Paired Input Bias ($b_\textrm{i} = -5$):** Leaving $b_\textrm{i} = 0$ lets LSTM cells grow to $|c_t| \le 0.5 / 0.0067 \approx 75$; setting $b_\textrm{i} = -b_\textrm{f} = -5$ couples $i_t = 1 - f_t$ so $|c_t| \le 1$.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-keep-gate-init
:::
:::
:::

::: {.slide title="Plain RNN Initialization: Borderline vs. Contracting"}
[Spectral Radius Analysis]{.kicker}

Before training, we also compare two initializations of a $64 \times 64$ plain RNN recurrent matrix $\mathbf{W}_\textrm{hh}$:

@!lstm-delayed-recall-8

- **LeCun normal ($\mathcal{N}(0, 1/64)$):**By the circular law of random matrix theory, eigenvalues fill the unit disk ($\rho \approx 1.02$) even though the largest singular value is $\approx 2.0$. Small states along leading eigenmodes neither vanish nor explode.
- **Default uniform ($\mathcal{U}[-1/8, 1/8]$):** Has $1/3$ the variance of LeCun normal, giving spectral radius $\rho \approx 0.62$ and contracting every signal by $\sim 0.6^k$.
:::

::: {.slide title="Delayed Recall Results Across 20 to 200 Time Steps"}
[Empirical Results]{.kicker}

Accuracy across two random seeds at delays $T \in \{20, 50, 100, 200\}$ (chance $= 0.125$):

@!lstm-delayed-recall-5

::: {.d2l-note}
Initializing forget/update biases to $+5$ (`keep gate`) turns complete failure ($0.12$) at $T \ge 50$ into $1.00$ accuracy across every seed and delay up to $T = 200$.
:::
:::

::: {.slide title="Why Gate Initialization Decides Long-Range Recall"}
[Interpreting the Recall Sweep]{.kicker}

Comparing the three model pairs in the delayed recall table isolates architecture from initialization:

- **Zero-bias LSTM & GRU:** The GRU solves $T = 20$ ($1.00$) and the LSTM is unreliable there ($0.2$–$1.00$); both collapse to chance ($0.12$–$0.14$) at $T \in \{50, 100, 200\}$ because a contribution of $2^{-49}$ is rounded away in `float32`.
- **Keep-gate LSTM & GRU:** Identical architecture and parameter count, yet starting at $\sigma(5) = 0.993$ yields **$1.00$ accuracy on every run** from $T = 20$ to $T = 200$.
- **Borderline vs. contracting plain RNN:** The $\rho \approx 1.02$ LeCun RNN sporadically carries the symbol to $T = 100$ (with large variance between seeds), whereas the $\rho \approx 0.62$ uniform RNN is hit-or-miss up to $T = 50$ and fails from $T = 100$ onward.

::: {.d2l-note .rule}
Gates replace fragile whole-matrix eigenvalue tuning with **per-unit learned decay**, provided initial biases keep the gradient horizon open.
:::
:::

::: {.slide title="Fair Language Benchmarks via a Shared Residual Stack"}
[The SeqLM Backbone]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{H} \leftarrow \mathbf{H} + \textrm{mixer}(\textrm{LN}(\mathbf{H})), \quad \mathbf{H} \leftarrow \mathbf{H} + \textrm{MLP}(\textrm{LN}(\mathbf{H}))$$

- **Controlled Architecture:** Byte embedding $\to$ $4$ pre-norm residual blocks ($d = 256$, $4d$ GELU MLP) $\to$ LayerNorm + output head on 128-byte WikiText-103 windows.
- **Isolating the Mixer:** Only `mixer(·)` changes across §§{sec_lstm}–§{num:sec_hybrids}.
:::

::: {.col .fig}
@fig:ch12-mrnn-seqlm-backbone
:::
:::
:::

::: {.slide title="Byte-Level WikiText-103: Mixers Compared Head-to-Head"}
[Language Modeling Benchmark]{.kicker}

Two runs per configuration reporting parameters, overall validation loss (nats/byte), loss by window position band (`0-15`, `16-63`, `64-127`), and training step time:

@!lstm-gated-networks-on-text-6
:::

::: {.slide title="Dissecting the WikiText-103 Results and Step Latency"}
[Quality vs. Step Cost]{.kicker}

Three systems and modeling lessons emerge from the WikiText-103 benchmark:

1. **The residual stack closes the plain-RNN gap:** Simply placing $4$ narrow $\tanh$ RNN layers inside `SeqLM` reaches $\sim 1.35$ nats/byte—matching Chapter §{num:chap_transformers}'s 6-block SwiGLU GPT and outperforming the 4-block GELU `attention` stack ($1.38$) at this $1{,}000$-step budget.
2. **Gating beats raw parameter scaling:** Replacing the plain RNN with an `LSTM` or `GRU` lowers loss by $0.05$ nats/byte to $\sim 1.30$, whereas widening the plain RNN to match the GRU's $3.81\text{M}$ parameters (`RNN, d=302`) closes less than a third of that gap ($\sim 1.33$).
3. **Step cost:** `LSTM` and `GRU` take **$1.4\times$–$3\times$ longer per step** than the plain RNN because each step computes three or four gate products instead of one. All three wait for $\mathbf{h}_{t-1}$; the length sweep of §{sec_linear-recurrence} isolates that cost.
:::

::: {.slide title="Concept Check: Gate Half-Lives and Saturation Bounds"}
[Check Your Understanding]{.kicker}

Work out the following three design questions:

1. If an LSTM forget gate is held at $f_t = 0.99$, what is its memory half-life in steps, and what fraction of a direct cell gradient survives $k = 100$ steps?
2. Why does `init_keep_gate` set the LSTM input-gate bias to $b_\textrm{i} = -5$ alongside $b_\textrm{f} = +5$, whereas the GRU only needs $b_\textrm{z} = +5$?
3. Why can gradient clipping stabilize an RNN with spectral radius $\rho = 1.3$, but cannot fix a zero-bias LSTM on $100$-step delayed recall?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Half-life $\tau_{1/2} = \ln 2 / \ln(1/0.99) \approx \mathbf{69\text{ steps}}$; after $100$ steps, $0.99^{100} \approx \mathbf{36.6\%}$ of the direct gradient survives.
2. The GRU couples its input weight to $1 - z_t = 1 - \sigma(5) = \sigma(-5)$ automatically; in an LSTM, leaving $b_\textrm{i} = 0$ ($i_t = 0.5$) lets cell states grow up to $0.5 / (1 - 0.9933) \approx 75$, saturating $\tanh(c_t)$.
3. Clipping rescales overly large gradients downward, whereas $2^{-99} \approx 1.6 \times 10^{-30}$ underflows to zero in `float32` when summed with recent gradients—destroying directional information.
:::
:::

::: {.slide title="Summary: Plain vs. Gated Recurrent Architectures"}
[Key Takeaways]{.kicker}

| Architecture / Setup | Direct Memory Path | Recurrent Params ($d_\textrm{in}=d$) | $T=200$ Recall Accuracy | WikiText-103 (nats/byte) | Training Depth |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Plain RNN (uniform scale)** | $\tanh(\mathbf{W}_\textrm{hh}\mathbf{h}_{t-1} + \cdots)$, $\rho \approx 0.62$ | $2d^2$ | $0.12$ (chance) | — | $T$ sequential steps |
| **Plain RNN (LeCun normal)** | $\tanh(\mathbf{W}_\textrm{hh}\mathbf{h}_{t-1} + \cdots)$, $\rho \approx 1.02$ | $2d^2$ | $0.13$–$0.76$ (unstable) | $1.351$ ($1.334$ at $d=302$) | $T$ sequential steps |
| **LSTM (zero gate bias)** | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$, $f_0 = 0.5$ | $8d^2$ | $0.12$ (chance at $T \ge 50$) | **$1.295$–$1.298$** | $T$ sequential steps |
| **GRU (zero gate bias)** | $\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1-\mathbf{z}_t)\odot \tilde{\mathbf{h}}_t$ | $6d^2$ | $0.12$ (chance at $T \ge 50$) | **$1.300$–$1.302$** | $T$ sequential steps |
| **LSTM / GRU (keep gate $+5$)** | Initial gate $\sigma(5) = 0.993$ ($\tau_{1/2} \approx 103$) | $8d^2$ / $6d^2$ | **$1.00$** (all seeds) | — | $T$ sequential steps |
:::

