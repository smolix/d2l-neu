# Selective State Space Models
:label:`sec_mamba`

The S4D layer of :numref:`sec_ssm` treats every token at lag $k$ identically: an input from $k$ steps ago contributes to the present output with the fixed convolution weight $K_k$, whether that input was the key noun of a paragraph or a comma. Natural language requires *content-based selection*: most tokens can be discarded quickly, while a few critical tokens must be written strongly and held across hundreds of steps at positions that vary from sentence to sentence. Can a state space model decide, token by token, what to keep and for how long?

Two models from earlier sections each hold half of the answer:

- `MinGRU` (:numref:`sec_linear-recurrence`) computes its decay gate from the current input token, so it can keep one token and overwrite the next, but its gates start with a one-step half-life and each channel holds only a single scalar state.
- `S4D` (:numref:`sec_ssm`) starts with half-lives spanning $1$ to $\sim 700$ steps across $N$ states per channel, but its dynamics are time-invariant.

*Selective* state space models, introduced with the Mamba architecture by :citet:`Gu.Dao.2023`, unify the two: the step size $\Delta_t$ of the discretized SSM—along with the write vector $\mathbf{B}_t$ and read vector $\mathbf{C}_t$—becomes a function of the current input $\mathbf{x}_t$. Setting $\Delta_t \to 0$ freezes the state and ignores an uninformative filler token; setting $\Delta_t$ large flushes stale state and writes a salient token. Because the dynamics now vary by step, the FFT convolution no longer applies, but the parallel associative scan of :numref:`sec_linear-recurrence` still trains the layer in $O(\log_2 T)$ rounds. Below we benchmark `S4D` and `MinGRU` on selective copying, show that a 1-state selective SSM is mathematically equivalent to a `MinGRU` gate, build `SelectiveSSM` and `Mamba` with a memory-efficient custom backward scan (`adjoint_scan`), inspect how a trained Mamba layer populates its state, and evaluate it on byte-level WikiText-103.

```{.python .input #mamba-selective-state-space-models}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mamba-selective-state-space-models}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Selective Copying

The synthetic *selective copying* benchmark of :citet:`Gu.Dao.2023` isolates content-dependent retention (:numref:`fig_selective_copy`). In our setup, a sequence consists of $1{,}024$ filler tokens (`0`), into which $4$ marked symbols (drawn uniformly from tokens `2..9`) are inserted at *random* positions, followed by $4$ query tokens (`1`) at the end. At the $j$-th query token, the model must output the $j$-th marked symbol in order (chance accuracy $1/8 = 0.125$). Because the symbol positions vary randomly from sequence to sequence, any time-invariant filter that weights inputs solely by lag $T - s$ must fail.

![The selective copying task. A few marked symbols (color) are scattered among filler tokens (grey) at positions that change from sequence to sequence; at the query tokens, the model must output the symbols in order.](../img/mdl-modernrnn-selective-copy.svg)
:label:`fig_selective_copy`

```{.python .input #mamba-selective-copying-1}
%%tab pytorch
def copy_batch(batch_size, num_fillers=1024, num_marked=4, num_symbols=8,
               device=None):
    """Symbols at random positions among fillers, then query tokens."""
    X = torch.zeros(batch_size, num_fillers + num_marked, dtype=torch.long,
                    device=device)
    X[:, num_fillers:] = 1  # Query tokens
    Y = torch.randint(2, 2 + num_symbols, (batch_size, num_marked),
                      device=device)
    positions = torch.rand(batch_size, num_fillers, device=device).argsort(1)
    X.scatter_(1, positions[:, :num_marked].sort(1).values, Y)
    return X, Y

X, Y = copy_batch(2, num_fillers=12)
print(X, Y, sep='\n')
```

```{.python .input #mamba-selective-copying-1}
%%tab jax
def copy_batch(key, batch_size, num_fillers=1024, num_marked=4,
               num_symbols=8):
    """Symbols at random positions among fillers, then query tokens."""
    key_y, key_p = jax.random.split(key)
    Y = jax.random.randint(key_y, (batch_size, num_marked), 2, 2 + num_symbols)
    positions = jnp.argsort(
        jax.random.uniform(key_p, (batch_size, num_fillers)), 1)
    X = jnp.zeros((batch_size, num_fillers + num_marked), dtype=jnp.int32)
    X = X.at[:, num_fillers:].set(1)  # Query tokens
    rows = jnp.arange(batch_size)[:, None]
    X = X.at[rows, jnp.sort(positions[:, :num_marked], 1)].set(Y)
    return X, Y

X, Y = copy_batch(d2l.get_key(), 2, num_fillers=12)
print(X, Y, sep='\n')
```

We train two-block `d2l.SeqLM` models at width $d = 48$ for 1,500 steps of Adam (`lr=3e-3`, batch size 64) using `d2l.train_queries`, and evaluate accuracy across 1,024 fresh test sequences. `copy_runs` trains two seeds per architecture and stores the trained models.

```{.python .input #mamba-selective-copying-2}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32

def train_copy(model):
    """Train for 1,500 steps on fresh batches of selective copying; return
    the test accuracy."""
    num_steps = 1500
    device = d2l.try_gpu()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    d2l.train_queries(model, lambda step: copy_batch(64, device=device),
                      num_steps, optimizer)
    X, Y = copy_batch(1024, device=device)
    return float(torch.stack([d2l.query_accuracy(model, X_i, Y_i)  # In parts
                              for X_i, Y_i in zip(X.split(128), Y.split(128))
                              ]).mean())
```

```{.python .input #mamba-selective-copying-2}
%%tab jax
def train_copy(model):
    """Train for 1,500 steps on fresh batches of selective copying; return
    the test accuracy."""
    num_steps = 1500
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(3e-3)), wrt=nnx.Param)
    key = d2l.get_key()  # New data every run
    d2l.train_queries(model, lambda step: copy_batch(
        jax.random.fold_in(key, step), 64), num_steps, optimizer)
    X, Y = copy_batch(jax.random.fold_in(key, num_steps), 1024)
    return float(jnp.stack([d2l.query_accuracy(model, X[i:i + 128],  # In parts
                                               Y[i:i + 128])
                            for i in range(0, 1024, 128)]).mean())
```

```{.python .input #mamba-selective-copying-4}
%%tab pytorch, jax
def copy_runs(name, make_model, num_runs=2):
    """Train num_runs fresh models; print their accuracies; return them."""
    models = [make_model() for _ in range(num_runs)]
    accuracies = [train_copy(model) for model in models]
    print(f'{name:20s} {d2l.num_params(models[0]):6,} parameters, accuracy '
          + ', '.join(f'{acc:.2f}' for acc in accuracies))
    return models
```

First let us test `S4D` and `minGRU`:

```{.python .input #mamba-selective-copying-3}
%%tab pytorch
models = {}
for name, mixer in (('S4D', lambda: d2l.S4D(48)),
                    ('minGRU', lambda: d2l.MinGRU(48, 48))):
    models[name] = copy_runs(name, lambda: d2l.SeqLM(
        10, 48, [mixer() for _ in range(2)]))
```

```{.python .input #mamba-selective-copying-3}
%%tab jax
rngs = nnx.Rngs(d2l.get_key())
models = {}
for name, mixer in (('S4D', lambda: d2l.S4D(48, rngs=rngs)),
                    ('minGRU', lambda: d2l.MinGRU(48, 48, rngs=rngs))):
    models[name] = copy_runs(name, lambda: d2l.SeqLM(
        10, 48, [mixer() for _ in range(2)], rngs=rngs))
```

As predicted, time-invariant `S4D` stays near chance ($\le 0.20$), whereas input-selective `minGRU` reaches $0.75$–$0.97$ accuracy. Notice why `minGRU` succeeds here despite starting with one-step gate half-lives (which failed on 1,000-step fixed-delay recall in :numref:`sec_ssm`): in selective copying, marked symbols occasionally land close to the queries, providing short-horizon gradients that progressively teach the gates to hold symbols across longer filler spans (Exercise :exref:`ex_mamba-short-memories`).

## Selective Dynamics

### Input-Dependent Step Sizes
:label:`subsec_selective-dynamics`

We build the selective SSM in two steps. First, consider a single channel with a single state ($N = 1$) at rate $\lambda = -1$ under zero-order hold (:numref:`subsec_zoh`): $h_t = e^{-\Delta} h_{t-1} + (1 - e^{-\Delta}) x_t$. Instead of treating $\Delta$ as a fixed parameter, compute $\Delta_t = \textrm{softplus}(s_t) = \log(1 + e^{s_t}) > 0$ from a linear projection $s_t = \mathbf{w}^\top \mathbf{x}_t + b$ of the current token. Exponentiating $-\Delta_t$ yields an exact algebraic identity:

$$
e^{-\Delta_t} = e^{-\log(1 + e^{s_t})} = \frac{1}{1 + e^{s_t}} = \sigma(-s_t).
$$
:eqlabel:`eq_step-gate`

Setting $z_t = \sigma(-s_t)$ turns the ZOH update into $h_t = z_t h_{t-1} + (1 - z_t) x_t$—the exact convex blend of `MinGRU` (:numref:`subsec_mingru`, Theorem 1 of :citet:`Gu.Dao.2023`). When a filler token produces $s_t = -4.6$, $\Delta_t \approx 0.01$ and $z_t = 0.99$ (retaining $99\%$ of the state, half-life 69 steps); when a marked symbol produces $s_t = 2$, $\Delta_t \approx 2.1$ and $z_t \approx 0.12$ (overwriting the state with the new symbol). Parameterizing the gate through $\Delta_t = \textrm{softplus}(s_t)$ also makes S4D's multi-timescale initialization immediate: choosing the bias $b$ so $\textrm{softplus}(b) \in [0.001, 0.1]$ starts the gate at half-lives of $7$ to $693$ steps, and—crucially—a single channel step size $\Delta_{t,c}$ can drive $N$ states with distinct rates $\lambda_1, \ldots, \lambda_N$ simultaneously.

### Write and Read Vectors

Now give each of the layer's $H$ channels ($H = 2d$ in Mamba) an $N$-dimensional state vector $\mathbf{h}_t^{(c)} \in \mathbb{R}^N$ with rates $\boldsymbol{\lambda} = (\lambda_1, \ldots, \lambda_N)^\top$ initialized to $(-1, \ldots, -N)^\top$. From the current $H$-dimensional input $\mathbf{x}_t$, the layer computes per-channel step sizes $\boldsymbol{\Delta}_t \in \mathbb{R}_+^H$ together with a shared write direction $\mathbf{B}_t \in \mathbb{R}^N$ and read direction $\mathbf{C}_t \in \mathbb{R}^N$:

$$
\boldsymbol{\Delta}_t = \textrm{softplus}(\mathbf{W}_\Delta \mathbf{x}_t +
\mathbf{b}_\Delta), \qquad \mathbf{B}_t = \mathbf{W}_B \mathbf{x}_t, \qquad
\mathbf{C}_t = \mathbf{W}_C \mathbf{x}_t.
$$

Each channel $c \in \{1, \ldots, H\}$ then updates its $N$ states and reads out its output via:

$$
\mathbf{h}_t^{(c)} = e^{\Delta_{t,c} \boldsymbol{\lambda}} \odot
\mathbf{h}_{t-1}^{(c)} + \Delta_{t,c}\, x_{t,c}\, \mathbf{B}_t, \qquad
y_{t,c} = \mathbf{C}_t^\top \mathbf{h}_t^{(c)} + D_c\, x_{t,c}.
$$
:eqlabel:`eq_selective_ssm`

Following :citet:`Gu.Dao.2023`, :eqref:`eq_selective_ssm` retains the exact ZOH decay $e^{\Delta_{t,c}\boldsymbol{\lambda}}$ while approximating the ZOH input weight $(e^{\Delta\lambda} - 1)/\lambda$ by the first-order Euler weight $\Delta_{t,c}$ (which agrees when $\Delta|\lambda| \ll 1$ and allows large steps $\Delta_{t,c}$ to write with weight exceeding $1/|\lambda|$; see Exercise 1). The shapes of all quantities in :eqref:`eq_selective_ssm` are summarized below:

| Quantity | Shape | Computed from |
|---|---|---|
| input $\mathbf{x}_t$ | $H$ | the layer's input at step $t$ |
| step sizes $\boldsymbol{\Delta}_t$ | $H$, one per channel | $\mathbf{x}_t$, through rank-$R$ factored $\mathbf{W}_\Delta \in \mathbb{R}^{H \times H}$ |
| write and read vectors $\mathbf{B}_t$, $\mathbf{C}_t$ | $N$ each, shared by all channels | $\mathbf{x}_t$, through $\mathbf{W}_B, \mathbf{W}_C \in \mathbb{R}^{N \times H}$ |
| rates $\boldsymbol{\lambda}$ | $N$ per channel | learned parameters initialized to $-1, \ldots, -N$ |
| state $\mathbf{h}_t^{(1)}, \ldots, \mathbf{h}_t^{(H)}$ | $H \times N$ | the recurrence :eqref:`eq_selective_ssm` |
| output $\mathbf{y}_t$ | $H$ | readout with $\mathbf{C}_t$ plus skip $D_c x_{t,c}$ |

## The Mamba Layer

### Projections, Convolution, and Gate

The `Mamba` block :cite:`Gu.Dao.2023` wraps the selective SSM inside channel-mixing projections, a width-4 depthwise causal convolution, and a SwiGLU-style multiplicative output gate (:numref:`fig_mamba_block`):

1. An input projection $\mathbf{W}_\textrm{in}$ expands the $d$-dimensional input into two branches of width $H = 2d$.
2. The first branch passes through a width-4 depthwise causal convolution and SiLU activation before entering the selective SSM—allowing $\boldsymbol{\Delta}_t, \mathbf{B}_t, \mathbf{C}_t$ to condition on a 4-token local window rather than a single isolated token.
3. The SSM output is multiplied elementwise by the SiLU of the second branch and projected back to width $d$ via $\mathbf{W}_\textrm{out}$.
4. To save parameters, $\mathbf{W}_\Delta \in \mathbb{R}^{H \times H}$ is factored through bottleneck rank $R = \lceil d / 16 \rceil$.

![The Mamba layer. The input is projected to two branches of width $H = 2d$; one passes through a short causal convolution, a SiLU, and the selective SSM, whose step sizes and write and read vectors are computed from its input; the other, through a SiLU, gates the result. A final projection returns to width $d$.](../img/mdl-modernrnn-mamba-block.svg)
:label:`fig_mamba_block`

### Training with a Scan

Because :eqref:`eq_selective_ssm` is still an affine recurrence $\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$ over the $H \times N$ state at each step, `d2l.linear_scan` evaluates all $T$ states in parallel. However, expanding the state to $H \times N$ numbers per token multiplies activation memory by $2N$: differentiating naively through $\lceil \log_2 T \rceil$ rounds of a doubling scan retains $\sim 2\lceil \log_2 T \rceil$ tensors of shape $(B, T, H, N)$ (already $\sim 2\textrm{ GB}$ per layer in our small copying model, and hundreds of gigabytes at LLM scale).

We can avoid storing the intermediate scan rounds by noting that the backpropagated state gradient $\mathbf{g}_t = \partial \ell / \partial \mathbf{h}_t$ satisfies the reverse-time affine recurrence:

$$\mathbf{g}_t = \frac{\partial \ell}{\partial \mathbf{h}_t}\Big|_\textrm{direct}
+ \mathbf{a}_{t+1} \odot \mathbf{g}_{t+1},$$

with parameter gradients $\partial \ell / \partial \mathbf{b}_t = \mathbf{g}_t$ and $\partial \ell / \partial \mathbf{a}_t = \mathbf{g}_t \odot \mathbf{h}_{t-1}$. Thus the backward pass is simply a second call to `linear_scan` run in reverse time, requiring only the forward decays $\mathbf{a}$ and states $\mathbf{h}$ (two tensors instead of $\sim 20$; Mamba's fused CUDA kernel goes one step further by keeping $\mathbf{h}$ in on-chip SRAM and recomputing it during the backward pass, as discussed in :numref:`sec_memory_precision`). Let us implement `adjoint_scan` with a custom backward pass and verify it against standard autograd.

```{.python .input #mamba-training-with-a-scan-1}
%%tab pytorch
class AdjointScan(torch.autograd.Function):  #@save
    """linear_scan whose backward pass is a second scan, run backward."""
    @staticmethod
    def forward(ctx, a, b):
        h = d2l.linear_scan(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        a, h = ctx.saved_tensors
        # g_t = grad_h_t + a_{t+1} g_{t+1}: a linear recurrence from the end
        a_next = torch.cat([a[:, 1:], torch.ones_like(a[:, :1])], 1)
        g = d2l.linear_scan(a_next.flip(1), grad_h.flip(1)).flip(1)
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)
        return g * h_prev, g

def adjoint_scan(a, b):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t, differentiated by a scan."""
    return AdjointScan.apply(a.expand_as(b), b)
```

```{.python .input #mamba-training-with-a-scan-1}
%%tab jax
@jax.custom_vjp  #@save
def _adjoint_scan(a, b):
    return d2l.linear_scan(a, b)

def _adjoint_scan_forward(a, b):  #@save
    """Forward pass: the states, and what the backward pass needs."""
    h = d2l.linear_scan(a, b)
    return h, (a, h)

def _adjoint_scan_backward(saved, grad_h):  #@save
    """Backward pass: gradients of a and b by a scan in reverse time."""
    a, h = saved
    # g_t = grad_h_t + a_{t+1} g_{t+1}: a linear recurrence from the end
    a_next = jnp.concatenate([a[:, 1:], jnp.ones_like(a[:, :1])], 1)
    g = d2l.linear_scan(a_next[:, ::-1], grad_h[:, ::-1])[:, ::-1]
    h_prev = jnp.concatenate([jnp.zeros_like(h[:, :1]), h[:, :-1]], 1)
    return g * h_prev, g

_adjoint_scan.defvjp(_adjoint_scan_forward, _adjoint_scan_backward)  #@save

def adjoint_scan(a, b):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t, differentiated by a scan."""
    return _adjoint_scan(jnp.broadcast_to(a, b.shape), b)
```

```{.python .input #mamba-training-with-a-scan-2}
%%tab pytorch
a = torch.rand(2, 100, 8, requires_grad=True)
b = torch.randn(2, 100, 8, requires_grad=True)
w = torch.randn(2, 100, 8)  # A loss that weights every state
grads_scan = torch.autograd.grad((d2l.linear_scan(a, b) * w).sum(), (a, b))
grads_adjoint = torch.autograd.grad((adjoint_scan(a, b) * w).sum(), (a, b))
deviation = max(float((g1 - g2).abs().max())
                for g1, g2 in zip(grads_scan, grads_adjoint))
print(f'max deviation: {deviation:.1e}')
```

```{.python .input #mamba-training-with-a-scan-2}
%%tab jax
key_a, key_b, key_w = jax.random.split(d2l.get_key(), 3)
a = jax.random.uniform(key_a, (2, 100, 8))
b = jax.random.normal(key_b, (2, 100, 8))
w = jax.random.normal(key_w, (2, 100, 8))  # A loss that weights every state
grad = lambda scan: jax.grad(lambda a, b: (scan(a, b) * w).sum(), (0, 1))
deviation = max(float(jnp.abs(g1 - g2).max()) for g1, g2 in
                zip(grad(d2l.linear_scan)(a, b), grad(adjoint_scan)(a, b)))
print(f'max deviation: {deviation:.1e}')
```

The custom reverse-time scan gradient matches autograd to $\sim 10^{-6}$.

### Implementation

`SelectiveSSM` implements :eqref:`eq_selective_ssm`, initializing `dt_bias` so $\textrm{softplus}(\textrm{dt\_bias})$ is distributed log-uniformly on $[\textrm{dt\_min}, \textrm{dt\_max}] = [0.001, 0.1]$. Passing `selective=False` replaces $\boldsymbol{\Delta}_t, \mathbf{B}_t, \mathbf{C}_t$ with learned per-channel constants (an LTI control inside the exact same Mamba block).

```{.python .input #mamba-the-layer-1}
%%tab pytorch
class SelectiveSSM(nn.Module):  #@save
    """A diagonal SSM with N states for each of num_channels channels. Its
    step sizes and write and read vectors depend on the input, or, with
    selective=False, are learned constants of each channel."""
    def __init__(self, num_channels, dt_rank, num_states=16, dt_min=1e-3,
                 dt_max=1e-1, selective=True):
        super().__init__()
        H, N, R = num_channels, num_states, dt_rank
        self.selective = selective
        self.log_neg_rate = nn.Parameter(  # Rates -1, ..., -N in every channel
            torch.log(torch.arange(1., N + 1)).repeat(H, 1))
        dt = dt_min * (dt_max / dt_min) ** torch.rand(H)
        dt_bias = dt + torch.log(-torch.expm1(-dt))  # softplus(dt_bias) = dt
        if selective:
            self.W_dt = nn.Sequential(  # Step size per channel, rank R
                nn.Linear(H, R, bias=False), nn.Linear(R, H))
            self.W_B = nn.Linear(H, N, bias=False)  # Write vector
            self.W_C = nn.Linear(H, N, bias=False)  # Read vector
            d2l.init_lecun(self)
            with torch.no_grad():  # Small up-projection, step sizes dt
                nn.init.uniform_(self.W_dt[1].weight, -R ** -0.5, R ** -0.5)
                self.W_dt[1].bias.copy_(dt_bias)
        else:  # Fixed dynamics: S4D with a learned write vector
            self.dt_bias = nn.Parameter(dt_bias)
            self.B = nn.Parameter(torch.ones(H, N))
            self.C = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.ones(H))
```

```{.python .input #mamba-the-layer-1}
%%tab jax
class SelectiveSSM(nnx.Module):  #@save
    """A diagonal SSM with N states for each of num_channels channels. Its
    step sizes and write and read vectors depend on the input, or, with
    selective=False, are learned constants of each channel."""
    def __init__(self, num_channels, dt_rank, num_states=16, dt_min=1e-3,
                 dt_max=1e-1, selective=True, *, rngs):
        H, N, R = num_channels, num_states, dt_rank
        self.selective = selective
        self.log_neg_rate = nnx.Param(  # Rates -1, ..., -N in every channel
            jnp.tile(jnp.log(jnp.arange(1., N + 1)), (H, 1)))
        def dt_bias(key, shape, dtype=jnp.float32):  # softplus(bias) = dt
            dt = dt_min * (dt_max / dt_min) ** jax.random.uniform(key, shape)
            return (dt + jnp.log(-jnp.expm1(-dt))).astype(dtype)
        if selective:
            def small(key, shape, dtype=jnp.float32):  # U(-R^-1/2, R^-1/2)
                return jax.random.uniform(key, shape, dtype, -R ** -0.5,
                                          R ** -0.5)
            self.W_dt = nnx.Sequential(  # Step size per channel, rank R
                nnx.Linear(H, R, use_bias=False, rngs=rngs),
                nnx.Linear(R, H, kernel_init=small, bias_init=dt_bias,
                           rngs=rngs))
            self.W_B = nnx.Linear(H, N, use_bias=False, rngs=rngs)  # Write
            self.W_C = nnx.Linear(H, N, use_bias=False, rngs=rngs)  # Read
        else:  # Fixed dynamics: S4D with a learned write vector
            self.dt_bias = nnx.Param(dt_bias(rngs.params(), (H,)))
            self.B = nnx.Param(jnp.ones((H, N)))
            self.C = nnx.Param(rngs.params.normal((H, N)) / math.sqrt(N))
        self.D = nnx.Param(jnp.ones(H))
```

```{.python .input #mamba-the-layer-2}
%%tab pytorch
@d2l.add_to_class(SelectiveSSM)  #@save
def log_coefficients(self, U):
    """Log-decays and writes, each of shape (batch_size, T, H, N)."""
    if self.selective:  # Step sizes (batch_size, T, H), write vectors
        dt, B = F.softplus(self.W_dt(U)), self.W_B(U).unsqueeze(-2)
    else:  # One step size and one write vector per channel
        dt, B = F.softplus(self.dt_bias), self.B
    b = (dt * U).unsqueeze(-1) * B
    log_a = dt.unsqueeze(-1) * -torch.exp(self.log_neg_rate)
    return log_a.expand_as(b), b

@d2l.add_to_class(SelectiveSSM)  #@save
def forward(self, U):  # U: (batch_size, num_steps, num_channels)
    log_a, b = self.log_coefficients(U)
    H = adjoint_scan(torch.exp(log_a), b)  # States, (batch_size, T, H, N)
    C = self.W_C(U).unsqueeze(-2) if self.selective else self.C  # Read
    return torch.einsum('...n,...n->...', H, C) + self.D * U  # Sum over n
```

```{.python .input #mamba-the-layer-2}
%%tab jax
@d2l.add_to_class(SelectiveSSM)  #@save
def log_coefficients(self, U):
    """Log-decays and writes, each of shape (batch_size, T, H, N)."""
    if self.selective:  # Step sizes (batch_size, T, H), write vectors
        dt, B = jax.nn.softplus(self.W_dt(U)), self.W_B(U)[..., None, :]
    else:  # One step size and one write vector per channel
        dt, B = jax.nn.softplus(self.dt_bias[...]), self.B[...]
    b = (dt * U)[..., None] * B
    log_a = dt[..., None] * -jnp.exp(self.log_neg_rate[...])
    return jnp.broadcast_to(log_a, b.shape), b

@d2l.add_to_class(SelectiveSSM)  #@save
def __call__(self, U):  # U: (batch_size, num_steps, num_channels)
    log_a, b = self.log_coefficients(U)
    H = adjoint_scan(jnp.exp(log_a), b)  # States, (batch_size, T, H, N)
    C = self.W_C(U)[..., None, :] if self.selective else self.C[...]  # Read
    return jnp.einsum('...n,...n->...', H, C) + self.D[...] * U  # Sum over n
```

`Mamba` wraps `SelectiveSSM` in the two-branch architecture of :numref:`fig_mamba_block`.

```{.python .input #mamba-the-layer-3}
%%tab pytorch
class Mamba(nn.Module):  #@save
    """The Mamba layer: a gated selective SSM between two projections."""
    def __init__(self, num_hiddens, num_states=16, expand=2, conv_width=4,
                 **ssm_args):
        super().__init__()
        H = expand * num_hiddens
        self.W_in = nn.Linear(num_hiddens, 2 * H, bias=False)
        self.conv = nn.Conv1d(H, H, conv_width, groups=H,  # One per channel
                              padding=conv_width - 1)
        self.W_out = nn.Linear(H, num_hiddens, bias=False)
        d2l.init_lecun(self)  # The SSM, built next, has its own init
        self.ssm = SelectiveSSM(H, math.ceil(num_hiddens / 16), num_states,
                                **ssm_args)

    def ssm_input(self, X):
        """The input of the SSM and the gate branch."""
        U, Z = self.W_in(X).chunk(2, -1)
        U = self.conv(U.transpose(1, 2))[..., :X.shape[1]]  # Keep it causal
        return F.silu(U.transpose(1, 2)), Z

    def forward(self, X):  # X: (batch_size, num_steps, num_hiddens)
        U, Z = self.ssm_input(X)
        return self.W_out(self.ssm(U) * F.silu(Z))
```

```{.python .input #mamba-the-layer-3}
%%tab jax
class Mamba(nnx.Module):  #@save
    """The Mamba layer: a gated selective SSM between two projections."""
    def __init__(self, num_hiddens, num_states=16, expand=2, conv_width=4,
                 *, rngs, **ssm_args):
        H = expand * num_hiddens
        self.W_in = nnx.Linear(num_hiddens, 2 * H, use_bias=False, rngs=rngs)
        self.conv = nnx.Conv(H, H, (conv_width,), feature_group_count=H,
                             padding=((conv_width - 1, 0),),  # Causal
                             rngs=rngs)
        self.ssm = SelectiveSSM(H, math.ceil(num_hiddens / 16), num_states,
                                rngs=rngs, **ssm_args)
        self.W_out = nnx.Linear(H, num_hiddens, use_bias=False, rngs=rngs)

    def ssm_input(self, X):
        """The input of the SSM and the gate branch."""
        U, Z = jnp.split(self.W_in(X), 2, -1)
        return jax.nn.silu(self.conv(U)), Z

    def __call__(self, X):  # X: (batch_size, num_steps, num_hiddens)
        U, Z = self.ssm_input(X)
        return self.W_out(self.ssm(U) * jax.nn.silu(Z))
```

## Trained Mamba Models

### Selective Copying with Mamba

We now train `Mamba` ($N = 4$ states per channel) alongside its non-selective ablation `Mamba, not selective` (`selective=False`, keeping the exact same projections, width-4 convolution, and SiLU gate) on selective copying:

```{.python .input #mamba-selective-copying-with-mamba}
%%tab pytorch
for name, selective in (('Mamba', True), ('Mamba, not selective', False)):
    # Compiled, since eagerly each round of the scan would be its own kernel
    models[name] = copy_runs(name, lambda: d2l.SeqLM(10, 48, [
        torch.compile(Mamba(48, num_states=4, selective=selective))
        for _ in range(2)]))
```

```{.python .input #mamba-selective-copying-with-mamba}
%%tab jax
for name, selective in (('Mamba', True), ('Mamba, not selective', False)):
    models[name] = copy_runs(name, lambda: d2l.SeqLM(10, 48, [
        Mamba(48, num_states=4, selective=selective, rngs=rngs)
        for _ in range(2)], rngs=rngs))
```

`Mamba` achieves $99\%$–$100\%$ accuracy across both runs, outperforming both `minGRU` and the non-selective control (`Mamba, not selective`, which drops by more than $30$ percentage points and scatters widely across seeds, matching the ablation reported in :citet:`Gu.Dao.2023`). Exercise :exref:`ex_mamba-longer-sequences` tests zero-shot length extrapolation to $4{,}096$ fillers.

### Contributions to the State

To see *how* `Mamba` retains the four marked symbols while ignoring fillers, we decompose the final state at the first query token into the contribution of each earlier position $s \le T$. In the unrolled recurrence $\mathbf{h}_T = \sum_{s=1}^T (\prod_{r=s+1}^T \mathbf{a}_r) \odot \mathbf{b}_s$, the Euclidean norm of position $s$'s decayed term $\|\exp(\sum_{r=s+1}^T \log \mathbf{a}_r) \odot \mathbf{b}_s\|_2$ measures how much of position $s$'s write survives in the state at step $T$.

```{.python .input #mamba-where-the-state-comes-from-1}
%%tab pytorch
def state_contributions(log_a, b):
    """Size of each position's term in the final state of h = a h + b."""
    cum = log_a.cumsum(1)
    terms = torch.exp(cum[:, -1:] - cum) * b  # Decayed to the last step
    return torch.linalg.vector_norm(terms.flatten(2), dim=-1)  # (batch, T)
```

```{.python .input #mamba-where-the-state-comes-from-1}
%%tab jax
def state_contributions(log_a, b):
    """Size of each position's term in the final state of h = a h + b."""
    cum = log_a.cumsum(1)
    terms = jnp.exp(cum[:, -1:] - cum) * b  # Decayed to the last step
    return jnp.linalg.norm(terms.reshape(*terms.shape[:2], -1), axis=-1)
```

Adding `log_coefficients` to `d2l.S4D` lets us compute `first_layer_contributions` identically for `S4D` and both trained `Mamba` runs on a test sequence truncated at the first query token.

```{.python .input #mamba-contributions-to-the-state-1}
%%tab pytorch
@d2l.add_to_class(d2l.S4D)
def log_coefficients(self, X):
    """Log-decays and writes of the recurrence, each (batch_size, T, d, N)."""
    rate = -torch.exp(self.log_neg_rate)
    log_a = torch.exp(self.log_dt) * rate  # log a_bar = dt * lambda
    b_bar = torch.expm1(log_a) / rate
    return log_a.expand(*X.shape, -1), b_bar * X.unsqueeze(-1)
```

```{.python .input #mamba-contributions-to-the-state-1}
%%tab jax
@d2l.add_to_class(d2l.S4D)
def log_coefficients(self, X):
    """Log-decays and writes of the recurrence, each (batch_size, T, d, N)."""
    rate = -jnp.exp(self.log_neg_rate[...])
    log_a = jnp.exp(self.log_dt[...]) * rate  # log a_bar = dt * lambda
    b_bar = jnp.expm1(log_a) / rate
    return (jnp.broadcast_to(log_a, (*X.shape, log_a.shape[-1])),
            b_bar * X[..., None])
```

```{.python .input #mamba-contributions-to-the-state-2}
%%tab pytorch
def first_layer_contributions(model, X):
    """Contributions to the first layer's state at the last step of X."""
    mixer, X = model.mixers[0], model.first_mixer_input(X)
    if isinstance(mixer, d2l.S4D):
        return state_contributions(*mixer.log_coefficients(X))
    U, _ = mixer.ssm_input(X)  # Mamba: what its selective SSM reads
    return state_contributions(*mixer.ssm.log_coefficients(U))

X, _ = copy_batch(1, device=d2l.try_gpu())
X = X[:, :-3]  # Up to the first query token
probed = {'S4D': models['S4D'][0], 'Mamba, run 1': models['Mamba'][0],
          'Mamba, run 2': models['Mamba'][1]}  # Both trained Mamba runs
with torch.inference_mode():
    contributions = {name: first_layer_contributions(model, X)[0]
                     for name, model in probed.items()}
```

```{.python .input #mamba-contributions-to-the-state-2}
%%tab jax
def first_layer_contributions(model, X):
    """Contributions to the first layer's state at the last step of X."""
    mixer, X = model.mixers[0], model.first_mixer_input(X)
    if isinstance(mixer, d2l.S4D):
        return state_contributions(*mixer.log_coefficients(X))
    U, _ = mixer.ssm_input(X)  # Mamba: what its selective SSM reads
    return state_contributions(*mixer.ssm.log_coefficients(U))

X, _ = copy_batch(d2l.get_key(), 1)
X = X[:, :-3]  # Up to the first query token
probed = {'S4D': models['S4D'][0], 'Mamba, run 1': models['Mamba'][0],
          'Mamba, run 2': models['Mamba'][1]}  # Both trained Mamba runs
contributions = {name: first_layer_contributions(model, X)[0]
                 for name, model in probed.items()}
```

Because the width-4 causal convolution spreads each marked symbol across up to 4 consecutive positions, `symbol_windows` collects the 4-position window starting at each symbol ($\le 1.6\%$ of the sequence) and measures both the share of the final state originating in those windows and the full contribution profile across all $1{,}025$ positions:

```{.python .input #mamba-where-the-state-comes-from-2}
%%tab pytorch
def symbol_windows(X, width=4):
    """Positions of each symbol and the width - 1 after it, without repeats."""
    starts = (X[0] > 1).nonzero().flatten()  # The four symbols
    offsets = torch.arange(width, device=X.device)
    windows = (starts[:, None] + offsets).flatten()
    windows = windows.unique()  # Windows of nearby symbols may overlap
    return windows[windows < X.shape[1]]  # A window may reach past the end

windows = symbol_windows(X)
print(f'symbol windows: {len(windows) / X.shape[1]:.1%} of the positions')
for name, c in contributions.items():
    print(f'{name:12s} share of the state from them: '
          f'{c[windows].sum() / c.sum():.2f}')
d2l.plot(torch.arange(X.shape[1]), [(c / c.max()).cpu() for c in
                                    contributions.values()],
         'position', 'contribution to the state',
         legend=list(contributions), yscale='log', figsize=(6, 3),
         legend_loc='outside')  # The curves fill the axes
```

```{.python .input #mamba-where-the-state-comes-from-2}
%%tab jax
def symbol_windows(X, width=4):
    """Positions of each symbol and the width - 1 after it, without repeats."""
    starts = jnp.nonzero(X[0] > 1)[0]  # The four symbols
    windows = (starts[:, None] + jnp.arange(width)).flatten()
    windows = jnp.unique(windows)  # Windows of nearby symbols may overlap
    return windows[windows < X.shape[1]]  # A window may reach past the end

windows = symbol_windows(X)
print(f'symbol windows: {len(windows) / X.shape[1]:.1%} of the positions')
for name, c in contributions.items():
    print(f'{name:12s} share of the state from them: '
          f'{c[windows].sum() / c.sum():.2f}')
d2l.plot(jnp.arange(X.shape[1]), [c / c.max() for c in
                                  contributions.values()],
         'position', 'contribution to the state',
         legend=list(contributions), yscale='log', figsize=(6, 3),
         legend_loc='outside')  # The curves fill the axes
```

We also compute the median *spike ratio*—the maximum contribution inside each symbol's 4-step window divided by the contribution of the filler token immediately preceding it—across 64 fresh test sequences:

```{.python .input #mamba-contributions-to-the-state-3}
%%tab pytorch
X_test = copy_batch(64, device=X.device)[0][:, :-3]  # 64 new sequences
rows, starts = (X_test[:, 1:] > 1).nonzero(as_tuple=True)  # The symbols,
starts = starts + 1                                       # not at position 0
window = (starts[:, None] + torch.arange(4, device=X.device)).clamp(
    max=X_test.shape[1] - 1)  # A window may reach past the end
print(f'spike ratio, median over {len(rows)} symbols')
with torch.inference_mode():
    for name, model in probed.items():
        c = first_layer_contributions(model, X_test)
        spike = c[rows[:, None], window].amax(1) / c[rows, starts - 1]
        print(f'{name:12s} {spike.median():8,.1f}')
```

```{.python .input #mamba-contributions-to-the-state-3}
%%tab jax
X_test = copy_batch(d2l.get_key(), 64)[0][:, :-3]  # 64 new sequences
rows, starts = jnp.nonzero(X_test[:, 1:] > 1)  # The symbols,
starts = starts + 1                            # not at position 0
window = jnp.minimum(starts[:, None] + jnp.arange(4),
                     X_test.shape[1] - 1)  # A window may reach past the end
print(f'spike ratio, median over {len(rows)} symbols')
for name, model in probed.items():
    c = first_layer_contributions(model, X_test)
    spike = c[rows[:, None], window].max(1) / c[rows, starts - 1]
    print(f'{name:12s} {float(jnp.median(spike)):8,.1f}')
```

In `S4D`, contributions rise monotonically toward the end of the sequence because decay depends only on recency, whereas in `Mamba`, contributions spike by two to three orders of magnitude at the four symbol windows—both because symbols take a large write step $\Delta_t$ and because that same large $\Delta_t$ exponentially attenuates ($e^{\Delta_t\boldsymbol{\lambda}}$) the preceding filler tokens.

### Language Modeling
:label:`subsec_mamba-lm`

Finally, we compare `Mamba`, `minGRU`, and `attention` on our byte-level WikiText-103 benchmark (`d2l.SeqLM`, 4 blocks at width $d = 256$, 128-byte windows, 1,000 steps), alongside a parameter-matched `minGRU, d=309` ($3.99\textrm{M}$ parameters, matching `Mamba`'s expanded-channel parameter count).

```{.python .input #mamba-language-modeling-4}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
```

```{.python .input #mamba-language-modeling-3}
%%tab pytorch
def text_model(mixer, d):  # 256 byte values; four blocks of width d
    return d2l.SeqLM(256, d, [mixer(d) for _ in range(4)]).to(d2l.try_gpu())

mamba = lambda d: torch.compile(Mamba(d))  # Compiled, as in the copying
min_gru = lambda d: d2l.MinGRU(d, d)
attention = lambda d: d2l.AttentionMixer(d, num_heads=8)
arms = {'Mamba': (mamba, 256), 'minGRU': (min_gru, 256),
        'attention': (attention, 256),
        'minGRU, d=309': (min_gru, 309)}  # As many parameters as Mamba
```

```{.python .input #mamba-language-modeling-3}
%%tab jax
def text_model(mixer, d):  # 256 byte values; four blocks of width d
    rngs = nnx.Rngs(d2l.get_key())
    return d2l.SeqLM(256, d, [mixer(d, rngs) for _ in range(4)], rngs=rngs)

mamba = lambda d, rngs: Mamba(d, rngs=rngs)
min_gru = lambda d, rngs: d2l.MinGRU(d, d, rngs=rngs)
attention = lambda d, rngs: d2l.AttentionMixer(d, num_heads=8, rngs=rngs)
arms = {'Mamba': (mamba, 256), 'minGRU': (min_gru, 256),
        'attention': (attention, 256),
        'minGRU, d=309': (min_gru, 309)}  # As many parameters as Mamba
```

```{.python .input #mamba-language-modeling-2}
%%tab pytorch, jax
trained = d2l.text_table(text_model, arms, data)
```

`Mamba` reaches $1.30$ nats/byte—matching the full `LSTM` and `GRU` of :numref:`subsec_deep-rnn` while training via a parallel scan, and outperforming equal-width `minGRU` by $0.05$ nats/byte and parameter-matched `minGRU, d=309` by $0.03$ nats/byte. Because `Mamba` scans an $H \times N = 512 \times 16 = 8{,}192$-number state per token ($32\times$ larger than `minGRU`'s 256-number state), its un-fused training step takes $2\times$–$4\times$ longer than `minGRU`, though at inference time that $32d$-number state (plus $6d$ numbers for the width-4 convolution buffer) equals the KV cache of just 19 tokens of multi-head attention. Moreover, viewing Mamba's $H \times N$ state as an outer-product matrix memory $\mathbf{S}_t$ updated by $(\boldsymbol{\Delta}_t \odot \mathbf{x}_t)\mathbf{B}_t^\top$ and queried by $\mathbf{C}_t$ leads directly to the matrix-state recurrences and chunked matrix-multiply training of :numref:`sec_matrix-state`.

## Summary

Time-invariant state space models weight past inputs purely by lag and fail on content-selective tasks such as selective copying. Selective state space models compute per-channel step sizes $\boldsymbol{\Delta}_t = \textrm{softplus}(\mathbf{W}_\Delta \mathbf{x}_t + \mathbf{b}_\Delta)$ and shared write and read vectors $\mathbf{B}_t, \mathbf{C}_t$ from each input token. For a single state ($N = 1$), $e^{-\textrm{softplus}(s_t)} = \sigma(-s_t)$ recovers the `MinGRU` update gate; with $N$ states initialized at rates $-1, \ldots, -N$ and log-uniform $\Delta$ biases, `Mamba` combines content selection with multi-timescale memory. Differentiating the parallel scan via a reverse-time scan (`adjoint_scan`) avoids storing the $O(\log_2 T)$ intermediate doubling rounds. On selective copying, `Mamba` achieves $99\%$–$100\%$ accuracy (dropping by $>30$ points when selectivity is disabled), and on byte-level WikiText-103 it matches the full LSTM and GRU while outperforming a parameter-matched `MinGRU` by $0.03$ nats/byte.

## Exercises

1. **The approximate write.** Mamba keeps the decay $e^{\Delta\lambda}$ of
   the zero-order hold but writes with weight $\Delta$ instead of
   $(e^{\Delta\lambda} - 1)/\lambda$.
    1. For $\lambda = -1$, find the largest $\Delta$ for which the two
       weights differ by less than 10%.
    1. For one state with $\lambda = -1$, write vector 1, and the exact weight,
       show that a state that starts at zero stays within the interval spanned
       by zero and the inputs it has seen. Find a sequence of inputs in $[-1,
       1]$ for which the state with weight $\Delta$ leaves that interval. What
       keeps the values of a trained Mamba model in range?

1. **Memory of the scan.** For a Mamba layer with $H$ channels and $N$
   states, a minibatch of $B$ sequences of length $T$, and 32-bit numbers,
   count the bytes that backpropagation stores for the scan, (a) through the
   rounds of the doubling scan, (b) with the adjoint scan, and (c) with a
   kernel that stores only the inputs of the layer and recomputes the
   states. Evaluate the three for $H = 4096$, $N = 16$, $B = 8$, and $T =
   8192$, and compare with the memory of a GPU.

1. [code] **Longer sequences.** Evaluate the Mamba model trained with
   1,024 positions before the queries on sequences with 2,048 and 4,096
   (`num_fillers`), without retraining.
   :exlabel:`ex_mamba-longer-sequences`
    1. Does it still copy the symbols? Compare with the minGRU.
    1. For each channel of the first layer, compute the decay of its slowest
       state over 1,000 filler tokens. How many channels can carry a symbol
       across 4,096 fillers?

1. [code] **Which part must be selective?** ●
    1. Between the two Mamba models of this section lie two more:
       input-dependent step sizes with fixed write and read vectors
       $\mathbf{B}$ and $\mathbf{C}$, and fixed step sizes with
       input-dependent $\mathbf{B}_t$ and $\mathbf{C}_t$. Train both,
       twice each. Which of them solve the task? Relate the result to the
       spike ratios measured above.
    1. In the trained Mamba model, set the step sizes of the first layer to
       their median over filler positions at the four positions starting at
       each symbol (the windows measured above), so that the layer cannot write
       the symbols with a large step. How much does the accuracy drop? Repeat
       with the step sizes set to their median over these positions everywhere
       else, so that the layer cannot skip the fillers.

1. [code] **Generation.** Add a method `step` to `SelectiveSSM` and to
   `Mamba` that processes one token, given the state of the SSM and the
   last three inputs of the convolution, and returns the output and the
   new state. Check that stepping through a sequence reproduces the
   outputs of the forward pass. How many numbers does the layer carry from
   token to token?

1. [code] **Short initial memories.** The minGRU learned to copy although
   its gates start with a memory of about one step. Test the explanation
   that the symbols closest to the queries teach the gates.
   :exlabel:`ex_mamba-short-memories`
    1. Record the minGRU's gates on filler tokens before and after
       training, and, for the trained model, its accuracy on each query
       separately, sorted by the distance between the symbol and its
       query. Which symbols does it copy best?
    1. Train the Mamba model with step sizes initialized between 0.5 and 1
       (`dt_min=0.5, dt_max=1.0`), so that every state starts with a memory
       of at most two steps. Does it learn to copy the symbols within the
       training budget? If so, plot the step sizes of the first layer on
       filler tokens before and after training.
    1. Repeat both with 4,096 positions before the queries, where a symbol
       near a query is rarer, and with the minGRU initialized as in the
       long-range recall experiment of :numref:`sec_ssm`.

[Discussions](https://d2l.discourse.group/t/36782)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.5]{.kicker}

Selective state space models<br>
**selective copying · a step size as a gate · write and read vectors · the scan and its gradient · Mamba on text**
:::
:::

::: {.slide title="S4D Copies Barely More Symbols Than Chance"}
[The problem]{.kicker}

::: {.cols .vc}
::: {.col}
1,024 positions of filler, four of them symbols (of eight); at four
queries, repeat the symbols in order. Chance: 1/8.

- S4D: at most one in five. Its weights depend only on distance.
- One distance holds a symbol here, a filler there.
:::

::: {.col .fig}
![](../img/mdl-modernrnn-selective-copy.svg)
:::
:::

::: {.d2l-note}
Keep a token **because of what it is**, for up to a thousand steps.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
- show that a softplus step size is a GRU gate;
- write the selective SSM with the shape of every part;
- differentiate a scan with a second scan;
- read where a trained state comes from;
- weigh Mamba against the minGRU and attention.

**Running example:** selective copying; WikiText-103 bytes at width 256.
:::

::: {.slide title="This Deck Lets Each Token Choose the Step Size"}
[Scope and place]{.kicker}

gates → linear recurrence and scan → state space models →
**selective SSMs** → matrix memories → delta rule → hybrids

- Idea: the SSM's step size becomes a function of the token.
- Skipped: the fused GPU kernel; Mamba-2 (next deck).
:::

::: {.slide title="The minGRU and S4D Each Have Half a Solution"}
| | decay chosen by the token | long memories at the start | state per channel |
|---|---|---|---|
| minGRU | yes | no (gates at 1/2) | 1 number |
| S4D | no | yes | $N$ numbers |

::: {.d2l-note}
Mamba combines all three: the token chooses, the memories start
long, and each channel holds $N$ numbers.
:::
:::

::: {.slide title="S4D Fails; the minGRU Copies Most Symbols"}
Two blocks of width 48 in `d2l.SeqLM`, Adam, 1,500 steps of fresh
batches, two runs per model.

@!mamba-selective-copying-3

::: {.d2l-note}
Selectivity matters: the minGRU copies although its gates start with a
memory of one step.
:::
:::

::: {.slide title="A Step Size Through a Softplus Is a GRU Gate"}
One state, $\lambda = -1$, zero-order hold (input weight $1 - e^{-\Delta}$);
the token sets $s_t = \mathbf{w}^\top\mathbf{x}_t + b$:
$h_t = e^{-\Delta_t} h_{t-1} + (1 - e^{-\Delta_t})\, x_t$, with
$\Delta_t = \textrm{softplus}(s_t) = \log(1 + e^{s_t})$.

$$e^{-\textrm{softplus}(s)} = \frac{1}{1 + e^{s}} = \sigma(-s)$$

| $s_t$ | $\Delta_t$ | gate $z_t = \sigma(-s_t)$ | effect |
|---|---|---|---|
| $-4.6$ | 0.01 | 0.99 | keep: half-life 69 steps |
| $2$ | 2.1 | 0.12 | overwrite |

::: {.d2l-note}
With one state, the selective SSM is the minGRU's blend $h_t = z_t h_{t-1} + (1 - z_t) x_t$.
:::
:::

::: {.slide title="Mamba Gives Each Channel N States and a Write and Read Direction"}
$$\boldsymbol{\Delta}_t = \textrm{softplus}(\mathbf{W}_\Delta\mathbf{x}_t + \mathbf{b}_\Delta),\quad
\mathbf{B}_t = \mathbf{W}_B\mathbf{x}_t,\quad \mathbf{C}_t = \mathbf{W}_C\mathbf{x}_t$$

$$\mathbf{h}^{(c)}_t = e^{\Delta_{t,c}\boldsymbol{\lambda}} \odot \mathbf{h}^{(c)}_{t-1} + \Delta_{t,c}\, x_{t,c}\,\mathbf{B}_t,
\qquad y_{t,c} = \mathbf{C}_t^\top \mathbf{h}^{(c)}_t + D_c x_{t,c}$$

| $\mathbf{x}_t$, $\boldsymbol{\Delta}_t$ | $\mathbf{B}_t$, $\mathbf{C}_t$ | rates $\boldsymbol{\lambda}$ | state |
|---|---|---|---|
| $H$ channels | $N$, shared by channels | $-1, \ldots, -N$ per channel | $H \times N$ |

$D_c$: a learned skip weight per channel.

::: {.d2l-note}
With $N = 1$ and $\mathbf{B}_t = \mathbf{C}_t = 1$: the gate of the previous slide.
:::
:::

::: {.slide title="Check: How Long Can a Fresh Layer Remember?"}
$\Delta$ log-uniform in $[0.001, 0.1]$ across channels; rates $\lambda_n =
-1, \ldots, -16$. Half-life: $\ln 2/(\Delta|\lambda|)$ steps. What range
does a fresh layer cover?

::: {.fragment .fade-up}
$\ln 2/(0.1 \cdot 16) \approx 0.43$ to $\ln 2/0.001 = 693$ steps: a factor
of 16 from the rates, 100 from the step sizes.
:::
:::

::: {.slide title="The Mamba Layer Wraps the SSM in a Convolution and a Gate"}
::: {.cols .vc}
::: {.col}
- Project $d$ to $H = 2d$ channels and a gate branch.
- Causal convolution over 4 tokens: keep or skip from a short context.
- SiLU gate from the current token: how much of the past reaches the output.
- Step sizes through rank $\lceil d/16 \rceil$; the bias sets where memories start.
:::

::: {.col .fig .big}
![](../img/mdl-modernrnn-mamba-block.svg){width=72%}
:::
:::
:::

::: {.slide title="Selectivity Breaks the Convolution, Not the Scan"}
- Weight $k$ steps back: a product of $k$ **different** decays. No kernel.
- Still affine in $\mathbf{h}$: `linear_scan` applies.
- Copying model: $64 \times 1{,}028 \times 96 \times 4$ = 25 M numbers,
  100 MB per tensor of states.
- A large layer ($H = 4096$, $N = 16$, $8 \times 8{,}192$ tokens): 17 GB per tensor.

::: {.d2l-note .warn}
Autodiff through 11 doubling rounds keeps about 20 such tensors: 2 GB.
:::
:::

::: {.slide title="The Gradient of a Scan Is Another Scan"}
$$\mathbf{g}_t = \frac{\partial \ell}{\partial \mathbf{h}_t}\Big|_\textrm{direct} + \mathbf{a}_{t+1} \odot \mathbf{g}_{t+1},
\qquad \frac{\partial \ell}{\partial \mathbf{b}_t} = \mathbf{g}_t,\quad
\frac{\partial \ell}{\partial \mathbf{a}_t} = \mathbf{g}_t \odot \mathbf{h}_{t-1}$$

A linear recurrence run backward: store the decays and the states, two
tensors instead of about twenty.

@!mamba-training-with-a-scan-2

Mamba's fused kernel goes further: the states stay in on-chip memory and
are recomputed in the backward pass.
:::

::: {.slide title="Switching Selectivity Off Costs at Least 30 Points"}
Same block, same projections, convolution and gate; only
$\boldsymbol{\Delta}$, $\mathbf{B}$, $\mathbf{C}$ become constants.

@!mamba-selective-copying-with-mamba

::: {.d2l-note}
Every Mamba run beats every non-selective run by 30 points or more; the
non-selective runs scatter from about 0.2 to 0.7 across executions.
Selectivity, not the block, solves the task.
:::
:::

::: {.slide title="Mamba's State Spikes at the Symbols; S4D's Follows Distance"}
@!mamba-where-the-state-comes-from-2

::: {.d2l-note}
Size of each position's term in the first layer's state at the query: what
the state holds, not what $\mathbf{C}_t$ reads. The symbols' share of the
state varies from run to run.
:::
:::

::: {.slide title="Mamba's Symbols Stand Hundreds of Times Above the Fillers"}
Spike ratio: the largest contribution in a symbol's window over that of
the filler just before it; median over the symbols of 64 new sequences.

@!mamba-contributions-to-the-state-3

- Mamba: hundreds to thousands. S4D: a few tens at most.
- S4D can only write a symbol larger than a filler.
- A large selective step also decays everything written before it.

::: {.d2l-note}
Writing loudly S4D can learn; erasing on cue it cannot.
:::
:::

::: {.slide title="Check: How Big Is the State of a Mamba Layer?"}
$d = 256$: $H = 512$ channels, $N = 16$ states. Attention caches $2d$
numbers per token. How many numbers does a Mamba layer carry, in cached
tokens?

::: {.fragment .fade-up}
$HN = 8{,}192 = 32d$: 16 tokens of cache; with the convolution's
$3H = 6d$, 19. At any context length.
:::
:::

::: {.slide title="At Equal Size, Mamba Still Leads the minGRU on Text"}
WikiText-103 bytes, windows of 128, 1,000 steps; nats per byte, two runs.

@!mamba-language-modeling-2

::: {.d2l-note}
minGRU at $d = 309$ (Mamba's count): still 0.03 behind, runs within
0.003. About 0.02 of that lead is there in bytes 0 to 15: a short-range
gain. Mamba's step costs 2 to 4 times as much.
:::
:::

::: {.slide title="Recap"}
- Fixed dynamics weigh by distance: S4D copies at most one in five.
- $e^{-\textrm{softplus}(s)} = \sigma(-s)$: a step size is a gate; Mamba
  adds $N$ states, $\mathbf{B}_t$, $\mathbf{C}_t$.
- Scan forward, scan backward: two stored tensors.
- Copying: Mamba 0.98 or more; not selective, at least 30 points lower.
- Text: 0.03 nats per byte ahead of an equal-size minGRU.

**Next:** the $H \times N$ state as a matrix written by outer products:
how many writes does it hold?
:::
