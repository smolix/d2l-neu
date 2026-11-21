# Linear Recurrences and Parallel Scans
:label:`sec_linear-recurrence`

The gated networks of :numref:`sec_lstm` can retain information over hundreds of steps, yet they still process a sequence one token after another because each step's gates and candidate depend on the previous state $\mathbf{h}_{t-1}$. On a sequence of $4{,}096$ tokens, a GRU executes $4{,}096$ sequential updates, leaving most of a GPU idle during training while a Transformer processes all positions in parallel via large matrix multiplications. Can a recurrent network retain $O(1)$-state autoregressive generation while training across all positions in parallel?

Yes—as soon as the gates and candidate depend only on the current input $\mathbf{x}_t$, making the recurrence *linear* (or more precisely, *affine*) in $\mathbf{h}_{t-1}$. Affine updates can be composed associatively just like prefix sums :cite:`Hillis.Steele.1986`, evaluating all $T$ states in $\lceil \log_2 T \rceil$ parallel rounds. Below we progress from a scalar decay to the minimal GRU (`MinGRU`) of :citet:`Feng.Tung.Ahmed.ea.2024`, implement the parallel associative scan, evaluate `MinGRU` on byte-level WikiText-103, and benchmark wall-clock scaling against both sequential GRUs and causal attention up to $16{,}384$ tokens.

```{.python .input #linear-recurrence-linear-recurrences-and-parallel-scans}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #linear-recurrence-linear-recurrences-and-parallel-scans}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Linear Recurrences
:label:`subsec_linear-recurrences`

### A Single Decaying Number

Start with a scalar state $h_t$ that retains a fixed fraction $a \in (0, 1)$ of its previous value and adds the input $x_t$:

$$h_t = a\, h_{t-1} + x_t, \qquad h_0 = 0.$$
:eqlabel:`eq_lr-scalar`

Unrolling :eqref:`eq_lr-scalar` expresses $h_t$ directly in terms of past inputs:

$$h_t = \sum_{s=1}^{t} a^{t-s} x_s.$$
:eqlabel:`eq_lr-unrolled`

Every $h_t$ is an exponentially weighted moving average of past inputs with half-life $\ln 2 / \ln(1/a)$ steps ($\approx 7$ steps at $a = 0.9$; $\approx 693$ steps at $a = 0.999$), and the gradient $\partial h_t / \partial x_s = a^{t-s}$ decays at the exact same rate.

### Input-Dependent Decay

Generalizing to a vector state $\mathbf{h}_t \in \mathbb{R}^d$ with input-dependent elementwise decay $\mathbf{a}_t \in (0, 1)^d$ and *write* vector $\mathbf{b}_t \in \mathbb{R}^d$ yields the affine recurrence:

$$
\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t,
\qquad \mathbf{a}_t = \mathbf{a}(\mathbf{x}_t), \quad
\mathbf{b}_t = \mathbf{b}(\mathbf{x}_t).
$$
:eqlabel:`eq_affine_recurrence`

Unrolling :eqref:`eq_affine_recurrence` replaces the power $a^{t-s}$ by the cumulative product of intermediate decays:

$$
\mathbf{h}_t = \sum_{s=1}^{t} \Big( \prod_{r=s+1}^{t} \mathbf{a}_r \Big)
\odot \mathbf{b}_s.
$$
:eqlabel:`eq_lr-unrolled-gated`

Because $\mathbf{a}_t$ and $\mathbf{b}_t$ depend only on $\mathbf{x}_t$, all $T$ decay and write vectors can be computed in parallel across the entire sequence before the recurrence runs. And because each $\mathbf{a}_r \in (0, 1)^d$, past writes can be selectively preserved ($\mathbf{a}_r \approx \mathbf{1}$) or flushed ($\mathbf{a}_r \approx \mathbf{0}$) without ever exploding.

### The Minimal GRU
:label:`subsec_mingru`

Look at the three places where the previous state $\mathbf{h}_{t-1}$ enters the standard GRU of :numref:`sec_gru`:

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(\mathbf{W}_{\textrm{xz}} \mathbf{x}_t + \underbrace{\mathbf{W}_{\textrm{hz}} \mathbf{h}_{t-1}}_{\textrm{(1)}} + \mathbf{b}_\textrm{z}),\\
\mathbf{r}_t &= \sigma(\mathbf{W}_{\textrm{xr}} \mathbf{x}_t + \underbrace{\mathbf{W}_{\textrm{hr}} \mathbf{h}_{t-1}}_{\textrm{(2)}} + \mathbf{b}_\textrm{r}),\\
\tilde{\mathbf{h}}_t &= \tanh\big(\mathbf{W}_{\textrm{xh}} \mathbf{x}_t + \underbrace{\mathbf{W}_{\textrm{hh}} (\mathbf{r}_t \odot \mathbf{h}_{t-1})}_{\textrm{(3)}} + \mathbf{b}_\textrm{h}\big),
\end{aligned}
$$

followed by $\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$. Dropping the three marked recurrent dependencies (which eliminates the reset gate $\mathbf{r}_t$ altogether) yields the *minimal GRU* (*minGRU*) of :citet:`Feng.Tung.Ahmed.ea.2024`:

$$
\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{z} \mathbf{x}_t + \mathbf{b}_\textrm{z}), \qquad
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_\textrm{h} \mathbf{x}_t + \mathbf{b}_\textrm{h}), \qquad
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t.
$$
:eqlabel:`eq_mingru`

This is an affine recurrence :eqref:`eq_affine_recurrence` with decay $\mathbf{a}_t = \mathbf{z}_t$ and write $\mathbf{b}_t = (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$, cutting the layer's parameter count from $6d^2$ to $2d^2$. While a single minGRU layer cannot condition its gate on what is already in $\mathbf{h}_{t-1}$, stacking multiple minGRU blocks with positionwise MLPs restores state-dependent gating across depth without sacrificing parallel evaluation across time.

## Parallel Scans
:label:`subsec_parallel-scans`

### Composing Affine Steps

Evaluating the unrolled sum :eqref:`eq_lr-unrolled-gated` naively for all $t = 1, \ldots, T$ would take $O(T^2)$ operations. Instead, we exploit the associativity of function composition. Represent step $t$'s affine map $\mathbf{h} \mapsto \mathbf{a}_t \odot \mathbf{h} + \mathbf{b}_t$ by the pair $(\mathbf{a}_t, \mathbf{b}_t)$. Applying step 1 followed by step 2 yields $\mathbf{a}_2 \odot (\mathbf{a}_1 \odot \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2$, which is another affine map with coefficients:

$$
(\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2) =
(\mathbf{a}_1 \odot \mathbf{a}_2,\ \mathbf{a}_2 \odot \mathbf{b}_1 +
\mathbf{b}_2).
$$
:eqlabel:`eq_scan_combine`

Because function composition is associative and the composite of any block of steps is still a single $(\mathbf{a}, \mathbf{b})$ pair of the same size, computing $\mathbf{h}_1, \ldots, \mathbf{h}_T$ from $\mathbf{h}_0 = \mathbf{0}$ reduces to an associative prefix *scan* under $\bullet$ (with $\mathbf{h}_t$ read from the second component of the $t$-th prefix).

### The Doubling Schedule

:numref:`fig_scan_tree` illustrates the parallel doubling schedule of :citet:`Hillis.Steele.1986` on $T = 8$ steps. In round $r = 0, 1, \ldots, \lceil \log_2 T \rceil - 1$, every position combines with the position at stride $2^r$ behind it. After $\lceil \log_2 T \rceil$ rounds, every position holds its complete prefix.

![A parallel scan on eight steps. Each box shows which steps its value
combines, $(1{:}4)$ meaning steps 1 to 4. In every round each position
combines with the position a fixed stride back (blue arrows), and the stride
doubles from round to round; dashed arrows copy a value unchanged. Teal
boxes hold complete prefixes. After $\log_2 8 = 3$ rounds every position
does.](../img/mdl-modernrnn-scan-tree.svg)
:label:`fig_scan_tree`

The Hillis–Steele doubling schedule performs $O(T \log_2 T)$ combines across $\lceil \log_2 T \rceil$ parallel rounds, whereas the *work-efficient* up-sweep/down-sweep tree schedule of :citet:`Blelloch.1990` performs $\sim 2T$ combines across $2\lceil \log_2 T \rceil$ shrinking rounds. We write the concise doubling schedule in PyTorch (`associative_scan`) and call XLA's built-in work-efficient `jax.lax.associative_scan` in JAX.

### Implementation

`linear_scan` computes all states $\mathbf{h}_1, \ldots, \mathbf{h}_T$ of :eqref:`eq_affine_recurrence` along the time axis (axis 1), folding an optional initial state $\mathbf{h}_0$ into the first write $\mathbf{b}_1 \leftarrow \mathbf{a}_1 \odot \mathbf{h}_0 + \mathbf{b}_1$.

```{.python .input #ssm-implementation-1}
%%tab pytorch
def associative_scan(combine, elems):  #@save
    """All prefixes of elems under an associative combine; time is axis 1."""
    s = 1
    # Autograd keeps the tensors of every round for the backward pass
    while s < elems[0].shape[1]:  # Combine every position with the one s back
        earlier = tuple(x[:, :-s] for x in elems)
        later = tuple(x[:, s:] for x in elems)
        elems = tuple(torch.cat([x[:, :s], y], 1)
                      for x, y in zip(elems, combine(earlier, later)))
        s *= 2
    return elems

def linear_scan(a, b, h0=None):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t; time is the second axis."""
    a = a.expand_as(b)  # Decays may omit axes on which they do not vary
    if h0 is not None:  # Fold the initial state into the first step
        b = torch.cat([a[:, :1] * h0.unsqueeze(1) + b[:, :1], b[:, 1:]], 1)
    def combine(earlier, later):
        (a1, b1), (a2, b2) = earlier, later
        return a1 * a2, a2 * b1 + b2
    return associative_scan(combine, (a, b))[1]
```

```{.python .input #ssm-implementation-1}
%%tab jax
def linear_scan(a, b, h0=None):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t; time is the second axis."""
    a = jnp.broadcast_to(a, b.shape)  # Decays may omit axes; broadcast them
    if h0 is not None:  # Fold the initial state into the first step
        b = b.at[:, 0].add(a[:, 0] * h0)
    def combine(earlier, later):
        (a1, b1), (a2, b2) = earlier, later
        return a1 * a2, a2 * b1 + b2
    # Autograd keeps the intermediate results of the scan for the backward pass
    return jax.lax.associative_scan(combine, (a, b), axis=1)[1]
```

Let us verify `linear_scan` against a sequential step-by-step loop on length $T = 100$ (a non-power of two).

```{.python .input #ssm-implementation-2}
%%tab pytorch
def sequential_scan(a, b):
    h, states = torch.zeros_like(b[:, 0]), []
    for a_t, b_t in zip(a.unbind(1), b.unbind(1)):
        h = a_t * h + b_t
        states.append(h)
    return torch.stack(states, 1)

a, b = torch.rand(4, 100, 8), torch.randn(4, 100, 8)
print(f'max deviation: '
      f'{(linear_scan(a, b) - sequential_scan(a, b)).abs().max():.1e}')
```

```{.python .input #ssm-implementation-2}
%%tab jax
def sequential_scan(a, b):
    def step(h, ab):
        h = ab[0] * h + ab[1]
        return h, h
    # lax.scan steps through the leading axis, so time goes first
    ab = (jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1))
    return jnp.swapaxes(jax.lax.scan(step, jnp.zeros_like(b[:, 0]), ab)[1], 0, 1)

key_a, key_b = jax.random.split(d2l.get_key())
a = jax.random.uniform(key_a, (4, 100, 8))
b = jax.random.normal(key_b, (4, 100, 8))
print(f'max deviation: '
      f'{jnp.abs(linear_scan(a, b) - sequential_scan(a, b)).max():.1e}')
```

The parallel scan and sequential loop agree to float32 rounding tolerance ($< 10^{-6}$).

## A Linear Recurrent Language Model

### The minGRU on Text

Wrapping `linear_scan` around two linear projections ($\mathbf{W}_\textrm{z}$ and $\mathbf{W}_\textrm{h}$) implements `MinGRU` :eqref:`eq_mingru` as a drop-in sequence mixer for `d2l.SeqLM`.

```{.python .input #ssm-a-linear-recurrent-language-model-1}
%%tab pytorch
class MinGRU(nn.Module):  #@save
    """The minimal GRU: its gate and candidate read only the input."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.W_z = nn.Linear(num_inputs, num_hiddens)  # Update gate (decay)
        self.W_h = nn.Linear(num_inputs, num_hiddens)  # Candidate state
        d2l.init_lecun(self)

    def coefficients(self, X):
        """Decays a_t and writes b_t of the linear recurrence."""
        Z = torch.sigmoid(self.W_z(X))
        return Z, (1 - Z) * torch.tanh(self.W_h(X))

    def forward(self, X):  # X: (batch_size, num_steps, num_inputs)
        return linear_scan(*self.coefficients(X))
```

```{.python .input #ssm-a-linear-recurrent-language-model-1}
%%tab jax
class MinGRU(nnx.Module):  #@save
    """The minimal GRU: its gate and candidate read only the input."""
    def __init__(self, num_inputs, num_hiddens, *, rngs):
        self.W_z = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)  # Decay
        self.W_h = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)

    def coefficients(self, X):
        """Decays a_t and writes b_t of the linear recurrence."""
        Z = jax.nn.sigmoid(self.W_z(X))
        return Z, (1 - Z) * jnp.tanh(self.W_h(X))

    def __call__(self, X):  # X: (batch_size, num_steps, num_inputs)
        return linear_scan(*self.coefficients(X))
```

How much modeling quality does dropping $\mathbf{h}_{t-1}$ from the gates cost on byte-level WikiText-103? Using the exact setup of :numref:`subsec_deep-rnn` (4 blocks of `d2l.SeqLM`, 128-byte windows, 1,000 steps), we compare `GRU` ($d = 256$, $3.81\textrm{M}$ parameters), `minGRU` at equal width ($d = 256$, $2.76\textrm{M}$ parameters), and `minGRU` widened to equal parameter count (`minGRU, d=302`, $3.81\textrm{M}$ parameters).

```{.python .input #linear-recurrence-a-linear-recurrent-language-model-1}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32

def text_model(mixer, d):  # 256 byte values; four blocks of width d
    return d2l.SeqLM(256, d, [mixer(d) for _ in range(4)]).to(d2l.try_gpu())

gru = lambda d: d2l.Recurrent(d2l.init_lecun(nn.GRU(d, d, batch_first=True)))
min_gru = lambda d: MinGRU(d, d)
arms = {'GRU': (gru, 256), 'minGRU': (min_gru, 256),
        'minGRU, d=302': (min_gru, 302)}  # As many parameters as the GRU
```

```{.python .input #linear-recurrence-a-linear-recurrent-language-model-1}
%%tab jax
def text_model(mixer, d):  # 256 byte values; four blocks of width d
    rngs = nnx.Rngs(d2l.get_key())
    return d2l.SeqLM(256, d, [mixer(d, rngs) for _ in range(4)], rngs=rngs)

lecun = nnx.initializers.lecun_normal()  # For the recurrent matrices
gru = lambda d, rngs: nnx.RNN(
    nnx.GRUCell(d, d, recurrent_kernel_init=lecun, rngs=rngs), unroll=8)
min_gru = lambda d, rngs: MinGRU(d, d, rngs=rngs)
arms = {'GRU': (gru, 256), 'minGRU': (min_gru, 256),
        'minGRU, d=302': (min_gru, 302)}  # As many parameters as the GRU
```

```{.python .input #linear-recurrence-a-linear-recurrent-language-model-2}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
trained = d2l.text_table(text_model, arms, data)
model = trained['minGRU'][0]  # The model we look inside below
```

At equal width ($d = 256$), `minGRU` finishes $\sim 0.05$ nats/byte behind `GRU` ($1.35$ vs. $1.30$), and at equal parameter count (`minGRU, d=302`) it closes about two fifths of the gap ($\sim 1.33$), landing right alongside the plain $\tanh$ RNN of :numref:`subsec_deep-rnn` while beating the attention baseline ($1.38$) and training via a parallel scan rather than a sequential loop.

### Inside the Trained Model

To check how much `minGRU` relies on its recurrent state, we evaluate the trained $d = 256$ model with its state zeroed before every step (`StateReset`, so $\mathbf{h}_t = \mathbf{b}_t$ depends only on the current byte).

```{.python .input #ssm-a-linear-recurrent-language-model-4}
%%tab pytorch
class StateReset(nn.Module):
    """A minGRU layer whose state is reset to zero before every step."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, X):
        return self.layer.coefficients(X)[1]

layers = model.mixers
loss = np.mean(d2l.loss_by_position(model, data))
model.mixers = nn.ModuleList([StateReset(layer) for layer in layers])
reset = np.mean(d2l.loss_by_position(model, data))
model.mixers = layers
print(f'validation loss: {loss:.3f} nats per byte, '
      f'with the state reset at every step: {reset:.3f}')
```

```{.python .input #ssm-a-linear-recurrent-language-model-4}
%%tab jax
class StateReset(nnx.Module):
    """A minGRU layer whose state is reset to zero before every step."""
    def __init__(self, layer):
        self.layer = layer

    def __call__(self, X):
        return self.layer.coefficients(X)[1]

layers = model.mixers
loss = np.mean(d2l.loss_by_position(model, data))
model.mixers = nnx.List([StateReset(layer) for layer in layers])
reset = np.mean(d2l.loss_by_position(model, data))
model.mixers = layers
print(f'validation loss: {loss:.3f} nats per byte, '
      f'with the state reset at every step: {reset:.3f}')
```

Zeroing the state raises validation loss from $1.35$ to $2.48$ nats/byte (essentially the $2.45$ nats/byte empirical byte-bigram baseline of :numref:`sec_gpt`), confirming that over $1.1$ nats/byte comes from context carried in the state.

We can also inspect the distribution of update-gate decays $\mathbf{z}_t$ in the first block before and after training. Adding `first_mixer_input` to `d2l.SeqLM` lets us extract the normalized token embeddings fed to the first mixer.

```{.python .input #ssm-a-linear-recurrent-language-model-5}
%%tab pytorch, jax
@d2l.add_to_class(d2l.SeqLM)  #@save
def first_mixer_input(self, X):
    """What the first mixer reads: the normalized token embeddings."""
    return self.norms[0](self.embedding(X))
```

```{.python .input #ssm-a-linear-recurrent-language-model-6}
%%tab pytorch
def first_gates(model, X):  # Gates of the first layer
    return model.mixers[0].coefficients(model.first_mixer_input(X))[0]

X, _ = next(iter(data.val_dataloader()))  # Validation text
X = X.to(next(model.parameters()).device)
with torch.inference_mode():
    Z_init = first_gates(text_model(min_gru, 256), X)  # Untrained
    Z = first_gates(model, X)
print('gate above  half-life in steps  share at initialization  trained')
for g in (0.5, 0.9, 0.99):
    print(f'{g:10}  {math.log(2) / -math.log(g):17.0f}  '
          f'{(Z_init > g).float().mean():23.1%}  '
          f'{(Z > g).float().mean():7.1%}')
```

```{.python .input #ssm-a-linear-recurrent-language-model-6}
%%tab jax
def first_gates(model, X):  # Gates of the first layer
    return model.mixers[0].coefficients(model.first_mixer_input(X))[0]

X = jnp.asarray(next(iter(data.val_dataloader()))[0])  # Validation text
Z_init = first_gates(text_model(min_gru, 256), X)  # Untrained
Z = first_gates(model, X)
print('gate above  half-life in steps  share at initialization  trained')
for g in (0.5, 0.9, 0.99):
    print(f'{g:10}  {math.log(2) / -math.log(g):17.0f}  '
          f'{(Z_init > g).mean():23.1%}  {(Z > g).mean():7.1%}')
```

Training increases the share of gates with half-life $\ge 7$ bytes ($g > 0.9$) more than tenfold (from $\sim 1\%$ to $\sim 16\%$), matching the word-level timescales needed over 128-byte windows, while almost no gates reach $g > 0.99$ (half-life 69 bytes) from zero-bias initialization.

## Cost against Attention
:label:`subsec_linear-recurrence-cost`

### Work, Depth, and Memory

:numref:`tab_linear-recurrence-cost` compares the asymptotic training work, sequential depth, and autoregressive generation state of one layer of width $d$ across $T$ tokens.

:Cost of one layer of width $d$ on $T$ tokens, up to constant factors.
:label:`tab_linear-recurrence-cost`

| Layer | Training work | Steps in a row | Memory at generation |
| :-- | :-- | :-- | :-- |
| GRU, one step at a time | $T d^2$ | $T$ | $d$ |
| minGRU, doubling scan | $T d^2 + T d \log_2 T$ | $\log_2 T$ | $d$ |
| minGRU, work-efficient scan | $T d^2 + T d$ | $2 \log_2 T$ | $d$ |
| Causal attention | $T d^2 + T^2 d$ | a few matrix products | $2 T d$ |

Both `GRU` and `minGRU` perform $O(Td^2)$ projection work and carry an $O(d)$ state vector during decoding, but `minGRU` reduces sequential training depth from $T$ matrix-vector steps to $O(\log_2 T)$ elementwise scan rounds. Causal attention has $O(1)$ sequential depth during training, but its attention score and mixing work grows quadratically as $O(T^2 d)$ (crossing its $8Td^2$ projection FLOPs around $T \approx 4d$) and its inference KV cache grows linearly as $2Td$.

### Measured Cost

To see where these scaling laws cross on real hardware, we time a forward-plus-backward pass in float16 at width $d = 256$ keeping the total token count per batch fixed at $B \times T = 65{,}536$ as sequence length $T$ sweeps from $512$ to $16{,}384$. With total tokens fixed, any layer whose cost is strictly linear in $T$ will have a flat curve, while a layer dominated by $O(T)$ sequential steps or $O(T^2)$ pairwise work will double its runtime each time $T$ doubles.

```{.python .input #linear-recurrence-cost-against-attention-1}
%%tab pytorch
d, tokens = 256, 65536  # Width; tokens per call, in sequences of length T
device, half = d2l.try_gpu(), torch.float16  # 16-bit floats throughout
timed = {'GRU': d2l.Recurrent(nn.GRU(d, d, batch_first=True)),
         'minGRU': torch.compile(MinGRU(d, d)),
         'attention': d2l.CausalAttention(d, num_heads=4)}
for layer in timed.values():
    layer.to(device, half)

def time_call(layer, T):
    """Milliseconds for a forward and backward pass over tokens / T
    sequences of length T."""
    X = torch.randn(tokens // T, T, d, device=device, dtype=half,
                    requires_grad=True)
    return 1000 * d2l.Benchmark(lambda: layer(X).float().sum().backward(),
                                warmup=1, repeats=3).time
```

```{.python .input #linear-recurrence-cost-against-attention-1}
%%tab jax
class FusedAttention(d2l.CausalAttention):
    """Causal attention through cuDNN's fused kernel."""
    @staticmethod
    def attend(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                            implementation='cudnn')

d, tokens = 256, 65536  # Width; tokens per call, in sequences of length T
half, rngs = jnp.float16, nnx.Rngs(d2l.get_key())  # 16-bit floats
timed = {'GRU': nnx.RNN(nnx.GRUCell(d, d, recurrent_kernel_init=lecun,
                                    param_dtype=half, rngs=rngs), unroll=8),
         'minGRU': MinGRU(d, d, rngs=rngs),
         'attention': FusedAttention(d, num_heads=4, rngs=rngs)}
for layer in timed.values():  # Every weight in 16 bits
    params = nnx.state(layer, nnx.Param)
    nnx.update(layer, jax.tree.map(lambda p: p.astype(half), params))

@nnx.jit
def forward_backward(layer, X):
    loss = lambda layer, X: layer(X).astype(jnp.float32).sum()
    return nnx.grad(loss, argnums=(0, 1))(layer, X)

def time_call(layer, T):
    """Milliseconds for a forward and backward pass over tokens / T
    sequences of length T."""
    X = jax.random.normal(d2l.get_key(), (tokens // T, T, d), half)
    return 1000 * d2l.Benchmark(lambda: forward_backward(layer, X),
                                warmup=1, repeats=3).time
```

```{.python .input #linear-recurrence-cost-against-attention-2}
%%tab pytorch, jax
lengths = [512, 1024, 2048, 4096, 8192, 16384]
times = {name: [time_call(layer, T) for T in lengths]
         for name, layer in timed.items()}
print('T        ', *(f'{T:>7d}' for T in lengths))
for name, ms in times.items():
    print(f'{name:9s}', *(f'{t:7.1f}' for t in ms))
d2l.plot(lengths, list(times.values()), 'sequence length',
         'milliseconds per call', legend=list(times), xscale='log',
         yscale='log', figsize=(4.5, 3))
```

The empirical scaling curves match :numref:`tab_linear-recurrence-cost`:

- **`GRU`:** slowest at every length, growing more than $15\times$ from $T = 512$ to $16{,}384$ because its $T$ sequential steps cannot be parallelized across time.
- **`minGRU`:** nearly flat across the entire 32-fold increase in sequence length (growing by under $50\%$ in PyTorch's doubling scan and virtually flat in JAX's work-efficient scan).
- **`attention`:** nearly flat up to $T \approx 2{,}048$ where $O(Td^2)$ projections and memory traffic dominate, then bending upward toward $2\times$ per doubling once the $O(T^2 d)$ attention term takes over, crossing PyTorch's doubling scan near $T = 4{,}096$ and trailing `minGRU` by $2.5\times$–$5\times$ at $T = 16{,}384$.

What `minGRU` still lacks is a structured way to initialize and control multi-timescale memories—the question :numref:`sec_ssm` addresses via continuous-time state space models.

## Summary

Removing the previous state $\mathbf{h}_{t-1}$ from a recurrent layer's gates yields an affine recurrence $\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$ whose steps compose associatively via $(\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2) = (\mathbf{a}_1 \odot \mathbf{a}_2, \mathbf{a}_2 \odot \mathbf{b}_1 + \mathbf{b}_2)$. An associative scan evaluates all $T$ states in $O(\log_2 T)$ parallel rounds during training while preserving $O(1)$-state sequential updates during decoding. Applied to the GRU, this simplification produces `MinGRU`, which matches a parameter-matched plain RNN on byte-level WikiText-103 (about $0.03$ nats/byte behind a full GRU) while training at nearly constant per-token time up to $16{,}384$ tokens—over an order of magnitude faster than a sequential GRU and several times faster than quadratic attention at long contexts.

## Exercises

1. **Closure under composition.** A parallel scan needs more than
   associativity: the composite of any run of steps must be described by as
   many numbers as a single step, so that every combine costs the same.
    1. Generalize :eqref:`eq_scan_combine` to the recurrence
       $\mathbf{h}_t = \mathbf{A}_t \mathbf{h}_{t-1} + \mathbf{b}_t$ with
       dense matrices $\mathbf{A}_t \in \mathbb{R}^{d \times d}$, keeping
       the factors of each product in the right order. Count the
       multiply-adds of one combine and of one sequential step. Compare the
       total work of the doubling scan, about $T \log_2 T$ combines, with
       that of the loop at $T = 4096$ and $d = 64$. What does the
       comparison suggest about dense transition matrices?
    1. For the scalar step $h \mapsto \tanh(w_t h + c_t)$, show that the
       composite of two steps with $w_1, w_2 \neq 0$ is not of the form
       $h \mapsto \tanh(w h + c)$ for any $w$ and $c$. What would a scan
       have to store to represent the composite of $k$ steps?

1. [code] **The direct route.** For scalar decays, the unrolled sum
   :eqref:`eq_lr-unrolled-gated` can be evaluated without a scan, from
   cumulative products $P_t = a_1 a_2 \cdots a_t$ and one cumulative sum.
    1. Show that $h_t = P_t \sum_{s=1}^{t} b_s / P_s$. Implement it with a
       library `cumprod` and `cumsum` and compare it with `linear_scan` in
       32-bit floats on decays drawn near 0.9. At what length does it fail,
       and why does the scan not fail there?
    1. :citet:`Feng.Tung.Ahmed.ea.2024` repair the formula by working with
       logarithms of the decays and of the writes, replacing the sum by a
       cumulative log-sum-exp. Implement the repair. What does it require
       of the signs of the writes, and how do the authors arrange it?

1. [code] **Candidate nonlinearity.** `MinGRU` keeps a $\tanh$ on the
   candidate $\tilde{\mathbf{h}}_t$, whereas
   :citet:`Feng.Tung.Ahmed.ea.2024` use a linear candidate.
    1. Starting from $\mathbf{h}_0 = \mathbf{0}$, write each coordinate of
       $\mathbf{h}_t$ as a weighted sum of the candidates
       $\tilde{\mathbf{h}}_1, \ldots, \tilde{\mathbf{h}}_t$. Show that the
       weights are nonnegative and sum to at most one, and derive a bound on
       $|\mathbf{h}_t|$ that holds at every length, with and without the
       $\tanh$.
    1. Remove the $\tanh$, train the language model of this section twice
       with `d2l.train_text`, and compare its validation loss with the two
       runs of `MinGRU`. Is the difference larger than the spread between
       runs? Relate what you observe to your bound.

1. [code] **State passing.** `linear_scan` accepts an initial state, so a
   long sequence can be processed in pieces.
    1. Split a sequence of length $T$ into chunks of length $\tau$, run the
       scan on each chunk starting from the final state of the previous
       one, and check that the concatenated states match a single call on
       the whole sequence.
    1. Now stop gradients through the carried state at every chunk
       boundary. The states do not change. Which contributions to the
       gradient of the loss at step $t$ disappear, and which dependencies
       can the trained model still use when it generates text? Compare with
       truncated backpropagation through time (:numref:`sec_bptt`).

1. [code] **Work-efficient scans.** ● A recursive schedule does less work
   than the doubling schedule: combine adjacent pairs, which halves the
   sequence; scan the half-length sequence recursively, which yields every
   second prefix; then obtain each remaining prefix with one more combine.
    1. For $T = 2^k$, count the combines and the rounds of combines of each
       schedule, and compare both with the $T - 1$ steps of the loop at
       $T = 16384$.
    1. Implement both schedules with slicing, check both against
       `sequential_scan`, and time them at the lengths of the measurement
       in :numref:`subsec_linear-recurrence-cost`. Does the ratio of times
       follow the ratio of combines? Explain why it might not at short
       lengths.

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.3]{.kicker}

Linear recurrences and parallel scans<br>
**a decaying number · affine steps compose · the minGRU on text · cost against attention**
:::
:::

::: {.slide title="A GRU Waits 4,095 Times on a 4,096-Token Sequence"}
[The problem]{.kicker}

- A GPU does tens of thousands of multiply-adds at once.
- A gated network cannot start step 2,000 before step 1,999 is done.
- A transformer layer processes all 4,096 positions in a few matrix products.

::: {.d2l-note}
The chain has one cause: the gates read $\mathbf{h}_{t-1}$ through a
nonlinearity. Make the update **linear in the state**, and
$\log_2 4096 = 12$ rounds suffice.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- unroll a linear recurrence into a weighted sum of its inputs;
- compose affine steps and compute all states in $\lceil\log_2 T\rceil$
  rounds;
- turn a GRU into a minGRU and say what it loses on text;
- predict how a GRU, a scan, and attention slow down as $T$ grows.

**Running example:** byte-level WikiText-103, windows of 128 bytes, four
blocks of width 256; the GRU reaches 1.30 nats per byte.
:::

::: {.slide title="Linear Recurrences Sit Between Gates and State Space Models"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Gated Recurrent Networks | gates as learned decay |
| Sequence-to-Sequence Learning | one state between two networks |
| **Linear Recurrences and Parallel Scans** | **$h_t = a_t h_{t-1} + b_t$, the scan, the minGRU** |
| State Space Models | decays from rates; kernels; convolution |
| Selective State Space Models | the input sets the step size |
| Matrix-State Recurrences | a state of key and value pairs |
| The Delta Rule | correct a pair instead of adding |
| Hybrid Architectures | a little attention back |
:::

::: {.col .narrow}
- **Uses:** the GRU, `SeqLM` and the byte benchmark, attention's FLOP
  count.
- **Skips:** how fast a state should forget (next deck).
:::
:::
:::

::: {.slide title="A Linear Recurrence Is a Weighted Sum of Its Inputs"}
$$h_t = a\,h_{t-1} + x_t \quad\Longrightarrow\quad h_t = \sum_{s \le t} a^{t-s} x_s$$

- $a = 0.9$, one input $x_1 = 1$: the state reads $1, 0.9, 0.81, \ldots$
- No $h_{t-1}$ on the right: every state follows from the inputs alone.
- Half-life $\ln 2/\ln(1/a)$: 7 steps at $a = 0.9$, 693 at $0.999$.

::: {.d2l-note}
$\partial h_t / \partial x_s = a^{t-s}$: the decay that forgets an input
also silences the gradient that would teach the model to use it.
:::
:::

::: {.slide title="Input-Chosen Decays Keep the Update Affine"}
$$\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t,
\qquad \mathbf{a}_t = \mathbf{a}(\mathbf{x}_t),\ \mathbf{b}_t = \mathbf{b}(\mathbf{x}_t)$$

$$\mathbf{h}_t = \sum_{s \le t} \Big(\prod_{r=s+1}^{t} \mathbf{a}_r\Big) \odot \mathbf{b}_s$$

- Decay $\mathbf{a}_t \in (0,1)^d$; the *write* $\mathbf{b}_t$ is what step
  $t$ adds (not a bias).
- *Affine* in $\mathbf{h}_{t-1}$: multiply, then add.
- An input survives as long as the inputs after it choose to keep it.
- All $\mathbf{a}_t, \mathbf{b}_t$ come from batched matrix products
  before the recurrence starts.
:::

::: {.slide title="The minGRU Is a GRU Whose Gates Read Only the Input"}
GRU: $\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{xz}\mathbf{x}_t + \mathbf{W}_\textrm{hz}\mathbf{h}_{t-1} + \mathbf{b}_\textrm{z})$;
its reset gate and candidate read $\mathbf{h}_{t-1}$ too. minGRU:

$$\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{z}\mathbf{x}_t + \mathbf{b}_\textrm{z}),\quad
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_\textrm{h}\mathbf{x}_t + \mathbf{b}_\textrm{h}),\quad
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$$

- Decay $\mathbf{a}_t = \mathbf{z}_t$, write $\mathbf{b}_t = (1 - \mathbf{z}_t)\odot\tilde{\mathbf{h}}_t$.
- The GRU's three uses of $\mathbf{h}_{t-1}$, in its gates and its
  candidate, are gone, and the reset gate with them.
- $2d^2$ weights per layer instead of the GRU's $6d^2$.
:::

::: {.slide title="A minGRU Unit Cannot Tell an Empty State From a Full One"}
Task for one unit: store the **first** digit of a text, ignore later ones.

| | update gate | first digit | second digit |
|:--|:--|:--|:--|
| GRU | $\sigma(\ldots + \mathbf{W}_\textrm{hz}\mathbf{h}_{t-1})$ | write | keep: the state is full |
| minGRU | $\sigma(\mathbf{W}_\textrm{z}\mathbf{x}_t + \mathbf{b}_\textrm{z})$ | write | write again |

::: {.d2l-note}
Gates that read only the token decide the same way whatever the state
holds. Stacking layers recovers some of this.
:::
:::

::: {.slide title="Affine Steps Compose Into Affine Steps"}
A step is the pair $(\mathbf{a}, \mathbf{b})$: the map
$\mathbf{h} \mapsto \mathbf{a} \odot \mathbf{h} + \mathbf{b}$.

$$(\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2) =
(\mathbf{a}_1 \odot \mathbf{a}_2,\ \mathbf{a}_2 \odot \mathbf{b}_1 + \mathbf{b}_2)$$

Example: $(0.5, 1) \bullet (0.5, 2) = (0.25,\ 2.5)$; stepping from
$h_0 = 0$ gives $h_1 = 1$, $h_2 = 2.5$.

- Summing the unrolled form term by term costs about $T^2/2$; composing
  pairs avoids the square.
- $\mathbf{h}_t$ is the $\mathbf{b}$-part of the first $t$ steps composed.
- The $\mathbf{a}$-part is carried so that a block can shrink what came
  before it.

::: {.d2l-note}
Associative, and a block of a thousand steps is still **one pair**.
:::
:::

::: {.slide title="A Scan Combines With a Doubling Stride"}
![](../img/mdl-modernrnn-scan-tree.svg){width=60%}

```{.python #ssm-exercises}
s = 1
while s < T:  # Combine each position with the one s steps back
    a, b = (cat(a[:s], a[:-s] * a[s:]),
            cat(b[:s], a[s:] * b[:-s] + b[s:]))
    s *= 2
```
A work-efficient scan needs about $2T$ combines in about $2\log_2 T$
rounds.

::: {.d2l-note}
$\lceil\log_2 T\rceil$ rounds instead of $T$, for about $\log_2 T$ times
the work.
:::
:::

::: {.slide title="Check: Rounds and Work"}
$T = 4096$.

1. How many rounds does the doubling scan need?
2. About how many combines, against the loop's steps?

::: {.fragment .fade-up}
1. $\log_2 4096 = 12$ rounds.
2. About $4096 \cdot 12 \approx 49{,}000$ combines against 4,095 steps:
   about 12 times the work, in 340 times fewer rounds.
:::
:::

::: {.slide title="At Equal Size the minGRU Trails the GRU by About 0.03 Nats per Byte"}
@!linear-recurrence-a-linear-recurrent-language-model-2

- Two runs each, nats per byte. At 2.76M parameters the minGRU is level
  with a plain tanh RNN of the same size (about 1.35).
- The gap is about 0.02 in the first 16 bytes, about 0.03 from byte 16 on.

::: {.d2l-note}
Input-only gates do no better than no gates. But the plain RNN's $\tanh$
chain cannot be scanned: the minGRU's case is speed at long lengths.
:::
:::

::: {.slide title="Memory Is Worth More Than a Nat per Byte"}
@!ssm-a-linear-recurrent-language-model-4

- Reset every state before every step: each prediction sees one byte.
- A one-byte model is a bigram model; the bigram model of the training
  bytes scores 2.45, even on those bytes.
- So about 1.1 nats per byte come from what the model remembers.
:::

::: {.slide title="After Training, One Gate in Six Keeps Its State for 7 Bytes or More"}
@!ssm-a-linear-recurrent-language-model-6

- Gate value $g$: half-life $\ln 2/\ln(1/g)$ steps.
- First layer, one validation batch, after 1,000 steps of training.
- Gates start near $1/2$, a half-life of about one step: before training
  about 1% exceed 0.9, after training about 18%.

::: {.d2l-note}
Windows of 128 bytes never ask for a memory of thousands; whether training
would find one is the next deck's question.
:::
:::

::: {.slide title="Attention Needs Few Steps in a Row, but Its Work Grows as T²"}
One layer of width $d$ on $T$ tokens, up to constant factors.

| layer | work | steps in a row | memory at generation |
|:--|:--|:--|:--|
| GRU | $T d^2$ | $T$ | $d$ |
| minGRU, doubling scan | $T d^2 + T d \log_2 T$ | $\log_2 T$ | $d$ |
| minGRU, work-efficient scan | $T d^2 + T d$ | $2\log_2 T$ | $d$ |
| causal attention | $T d^2 + T^2 d$ | a few matmuls | $2Td$ |

Attention per token: $8d^2$ FLOPs in projections, about $2Td$ in causal
scores and mixing.
:::

::: {.slide title="Check: Predict the Curves"}
Width $d = 256$; always 65,536 tokens per call; $T$ from 512 to 16,384.

1. Which layer's time stays flat?
2. Which grows nearly in proportion to $T$, and why?
3. From which $T$ on does attention's time start to double?

::: {.fragment .fade-up}
1. The scan: its work is proportional to the number of tokens.
2. The GRU: $T$ steps in a row, each with a fixed latency.
3. $8d^2 = 2Td$ at $T = 4d = 1{,}024$; beyond that, toward doubling.
:::
:::

::: {.slide title="At Long Lengths Attention Grows With T; the Scan Does Not"}
@!linear-recurrence-cost-against-attention-2

::: {.d2l-note}
Forward and backward, ms per 65,536 tokens. Attention turns later than the
count's 1,024: time per token that the FLOPs leave out dilutes the
quadratic term. From 4,096 on, about 1.7× per doubling; GRU slowest.
:::
:::

::: {.slide title="The Doubling Scan Makes 9 to 14 Passes Over the Sequence" only="pytorch"}
- Doubling schedule: $\lceil\log_2 T\rceil$ rounds, each reading and
  writing all 65,536 tokens; 9 rounds at 512, 14 at 16,384.
- Its time follows the rounds: slower than attention up to 2,048 tokens,
  level near 4,096, faster beyond.
- A work-efficient scan touches the sequence a few times, whatever $T$.
:::

::: {.slide title="A Work-Efficient Scan Stays Flat and Below Attention" only="jax"}
- `jax.lax.associative_scan`: about $2T$ combines, a few passes over the
  sequence, whatever $T$.
- Its time barely changes from 512 to 16,384 tokens and stays below
  attention's at every length.
- A doubling scan would make $\lceil\log_2 T\rceil$ passes instead.
:::

::: {.slide title="Recap"}
- Linear in the state: $\mathbf{h}_t$ is a sum of writes weighted by
  products of decays; forgetting and vanishing gradients are one number.
- Affine steps compose into one pair: all states in
  $\lceil\log_2 T\rceil$ rounds.
- minGRU on WikiText bytes: 0.03 nats per byte behind the GRU at equal
  size, level with a plain RNN, whose $\tanh$ chain cannot be scanned;
  memory worth more than a nat per byte.
- Cost: the GRU grows with $T$, attention with $T$ beyond a few thousand
  tokens at width 256, the scan hardly at all.

**Next:** the gates start near one half. How fast should a linear state
forget, and where should its decays start?
:::
