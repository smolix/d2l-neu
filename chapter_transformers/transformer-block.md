# The Transformer Block
:label:`sec_transformer-block`

Self-attention (:numref:`chap_attention`) mixes information across sequence positions: each token's output is a weighted average of value vectors from the positions it attends to. On its own, however, an attention layer applies only linear projections ($W_Q, W_K, W_V, W_O$) around the weighted sum. Without a position-wise nonlinearity, residual connections, and normalization, stacking attention layers cannot build hierarchical per-token features or propagate signals reliably across dozens of layers.

The **Transformer block** :cite:`Vaswani.Shazeer.Parmar.ea.2017` pairs multi-head attention with a position-wise **feed-forward network (FFN)**, wraps each sublayer in an additive **residual connection** :cite:`He.Zhang.Ren.ea.2016`, and stabilizes activations with **layer normalization** (:numref:`fig_transformer-block`). Three design upgrades separate modern blocks from the original 2017 Transformer:

1. **Normalization placement:** Moving normalization from the residual stream itself (**post-norm**) to the input of each sublayer (**pre-norm**) preserves an unattenuated identity path across depth.
2. **Normalization form:** Replacing mean-centered `LayerNorm` with root-mean-square normalization (**`RMSNorm`**) and optional per-head **query--key normalization (`QK-Norm`)** bounds activations and attention logits with less memory-bandwidth overhead.
3. **Gated feed-forward networks:** Replacing the two-matrix ReLU/GELU MLP ($d_{\textrm{ff}} = 4d$) with a gated **`SwiGLU`** network ($d_{\textrm{ff}} \approx \frac{8}{3}d$) improves representation quality at the exact same $8d^2$ parameter and FLOP budget.

```{.python .input #transformer-block-the-transformer-block}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #transformer-block-the-transformer-block}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Components of a Transformer Block

A Transformer block operates on a shared **residual stream** $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$ (:numref:`sec_what-attention-computes`). Rather than overwriting $\mathbf{X}$, two consecutive sublayers read from the stream, compute an update, and add their result back:

- **Multi-head attention (token mixing)** is the only place where sequence positions interact: each token queries the sequence (:numref:`sec_multihead-attention`) and adds the retrieved context vector to its stream.
- **Position-wise feed-forward network (channel mixing)** applies the same MLP independently to each token vector $\mathbf{x}_i \in \mathbb{R}^d$, transforming the features accumulated by attention. Roughly two-thirds of a dense block's parameters reside in the FFN.

In the **pre-norm** arrangement used by modern Transformers (:numref:`fig_transformer-block`), each sublayer normalizes its own input before computing its update:

$$
\begin{aligned}
\mathbf{H} &= \mathbf{X} + \textrm{Attn}\!\bigl(\textrm{Norm}(\mathbf{X})\bigr), \\
\mathbf{Y} &= \mathbf{H} + \textrm{FFN}\!\bigl(\textrm{Norm}(\mathbf{H})\bigr).
\end{aligned}
$$
:eqlabel:`eq_preln-block`

![The transformer block: attention and a position-wise FFN read from the residual stream and add their results back. Left, the 2017 arrangement: normalization sits on the stream itself, after each addition. Right, the modern arrangement: normalization moves onto each branch, and the stream runs uninterrupted from input to output.](../img/mdl-transformers-block-anatomy.svg)
:label:`fig_transformer-block`

## Normalization

### Pre-Norm vs. Post-Norm

The original 2017 Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017` and BERT :cite:`Devlin.Chang.Lee.ea.2018` used **post-norm**, placing normalization *on the residual stream itself* after each addition:

$$
\begin{aligned}
\mathbf{H} &= \textrm{Norm}\!\bigl(\mathbf{X} + \textrm{Attn}(\mathbf{X})\bigr), \\
\mathbf{Y} &= \textrm{Norm}\!\bigl(\mathbf{H} + \textrm{FFN}(\mathbf{H})\bigr).
\end{aligned}
$$
:eqlabel:`eq_postln-block`

Why did modern architectures switch from :eqref:`eq_postln-block` to :eqref:`eq_preln-block`? We can see the difference directly at initialization without training a single step, using a minimal block `MiniBlock` that supports both `pre_norm=True` and `pre_norm=False` as well as custom output-projection weight scale `std`:

```{.python .input #transformer-block-two-arrangements}
%%tab pytorch
class MiniBlock(nn.Module):
    """Attention + MLP on a residual stream; norm placement as a flag."""
    def __init__(self, num_hiddens, pre_norm):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = d2l.MultiHeadAttention(num_hiddens, 4)
        self.W_1 = nn.Linear(num_hiddens, 4 * num_hiddens)
        self.W_2 = nn.Linear(4 * num_hiddens, num_hiddens)
        self.norm1 = nn.LayerNorm(num_hiddens, eps=1e-5)
        self.norm2 = nn.LayerNorm(num_hiddens, eps=1e-5)

    def ffn(self, X):  # exact GELU
        return self.W_2(F.gelu(self.W_1(X)))

    def forward(self, X):
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.attention(Y, Y, Y)
            return X + self.ffn(self.norm2(X))
        X = self.norm1(X + self.attention(X, X, X))
        return self.norm2(X + self.ffn(X))
```

```{.python .input #transformer-block-two-arrangements}
%%tab jax
class MiniBlock(nnx.Module):
    """Attention + MLP on a residual stream; norm placement as a flag."""
    def __init__(self, num_hiddens, pre_norm, *, rngs):
        self.pre_norm = pre_norm
        self.attention = d2l.MultiHeadAttention(num_hiddens, 4, rngs=rngs)
        self.W_1 = d2l.Linear(num_hiddens, 4 * num_hiddens, rngs=rngs)
        self.W_2 = d2l.Linear(4 * num_hiddens, num_hiddens, rngs=rngs)
        self.norm1 = nnx.LayerNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hiddens, epsilon=1e-5, rngs=rngs)

    def ffn(self, X):  # exact GELU
        return self.W_2(jax.nn.gelu(self.W_1(X), approximate=False))

    def __call__(self, X):
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.attention(Y, Y, Y)
            return X + self.ffn(self.norm2(X))
        X = self.norm1(X + self.attention(X, X, X))
        return self.norm2(X + self.ffn(X))
```

### Signal Propagation and Token Collapse at Initialization
:label:`subsec_signal-propagation`

Consider passing a batch of random unit-variance token vectors $\mathbf{X}^{(0)} \in \mathbb{R}^{B \times n \times d}$ through $L = 32$ freshly initialized blocks. At each block $\ell \in \{1, \dots, L\}$, we track three quantities:

1. **Stream standard deviation** across all entries of $\mathbf{X}^{(\ell)}$;
2. **Token spread**, measured as the standard deviation across the $n$ sequence positions (which vanishes if all $n$ tokens collapse onto the same vector); and
3. **Query gradient norm** $\|\partial \mathcal{L} / \partial W_q^{(\ell)}\|_F$ flowing back from a unit scalar loss at the top of the stack:

```{.python .input #transformer-block-signal-propagation-at-initialization-1}
%%tab pytorch
def signal_stats(pre_norm, std=None, num_blks=32, num_hiddens=256):
    """Token spread, stream RMS, and gradient norms of a fresh stack whose
    weight matrices are Xavier, or N(0, std^2) if std is given."""
    blks = nn.ModuleList([MiniBlock(num_hiddens, pre_norm)
                          for _ in range(num_blks)])
    init = (nn.init.xavier_uniform_ if std is None  # the 2017 initialization
            else lambda W: nn.init.normal_(W, std=std))
    for p in blks.parameters():
        if p.ndim == 2:
            init(p)
    norm = torch.linalg.vector_norm
    H = torch.randn(2, 64, num_hiddens)
    spread, rms = [], []
    for blk in blks:
        H = blk(H)
        spread.append(norm(H - H.mean(1, keepdim=True)) / norm(H))
        rms.append(H.pow(2).mean().sqrt())
    (H * torch.randn_like(H)).mean().backward()  # generic readout loss
    grad_q = [norm(blk.attention.W_q.weight.grad) for blk in blks]
    proj = {n: norm(getattr(blks[-1].attention, n).weight.grad).item()
            for n in ('W_q', 'W_k', 'W_v', 'W_o')}
    return {'spread': torch.stack(spread).tolist(),
            'rms': torch.stack(rms).tolist(),
            'grad_q': torch.stack(grad_q).tolist(), 'proj': proj}
```

```{.python .input #transformer-block-signal-propagation-at-initialization-1}
%%tab jax
def signal_stats(pre_norm, std=None, num_blks=32, num_hiddens=256):
    """Token spread, stream RMS, and gradient norms of a fresh stack whose
    weight matrices are Xavier, or N(0, std^2) if std is given."""
    rngs = nnx.Rngs(d2l.get_key())
    stack = nnx.Sequential(*[MiniBlock(num_hiddens, pre_norm, rngs=rngs)
                             for _ in range(num_blks)])
    init = (nnx.initializers.xavier_uniform() if std is None  # as in 2017
            else nnx.initializers.normal(std))
    for _, m in nnx.iter_graph(stack):
        if isinstance(m, nnx.Linear):
            m.kernel[...] = init(d2l.get_key(), m.kernel.shape)
    X = jax.random.normal(d2l.get_key(), (2, 64, num_hiddens))
    R = jax.random.normal(d2l.get_key(), X.shape)
    # Full fp32 products: on a GPU, JAX's default TF32 rounding (about 1e-3)
    # would swamp token differences of 1e-4
    with jax.default_matmul_precision('highest'):
        H, spread, rms = X, [], []
        for blk in stack.layers:
            H = blk(H)
            spread.append(jnp.linalg.norm(H - H.mean(1, keepdims=True))
                          / jnp.linalg.norm(H))
            rms.append(jnp.sqrt((H ** 2).mean()))
        g = nnx.grad(lambda m: (m(X) * R).mean())(stack)  # generic readout
    gnorm = lambda k, n: jnp.linalg.norm(
        g['layers'][k]['attention'][n]['kernel'][...])
    grad_q = [gnorm(k, 'W_q') for k in range(num_blks)]
    proj = {n: float(gnorm(num_blks - 1, n))
            for n in ('W_q', 'W_k', 'W_v', 'W_o')}
    return {'spread': jnp.stack(spread).tolist(),
            'rms': jnp.stack(rms).tolist(),
            'grad_q': jnp.stack(grad_q).tolist(), 'proj': proj}
```

Why does post-norm lose token diversity across depth? In each post-norm sublayer, the residual stream $\mathbf{x}_i$ (variance $1$) and the sublayer update $\mathbf{u}_i$ (variance $\sigma^2$) are added and then divided by $\sqrt{1 + \sigma^2}$ to restore unit variance. Because attention averages across tokens, the shared mean component is reinforced while the token-specific difference $\mathbf{x}_i - \mathbf{x}_j$ is scaled down at every block by roughly

$$
r \approx \frac{1}{\sqrt{1 + \sigma_{\textrm{attn}}^2}} \cdot \frac{1}{\sqrt{1 + \sigma_{\textrm{ffn}}^2}} < 1,
$$
:eqlabel:`eq_collapse-rate`

so token spread decays exponentially as $r^L$ across depth $L$.

Under pre-norm (:eqref:`eq_preln-block`), the residual stream is never divided by a normalization denominator: unrolling $L$ blocks gives a direct sum $\mathbf{X}^{(L)} = \mathbf{X}^{(0)} + \sum_{\ell=1}^{2L} \mathbf{U}^{(\ell)}$. Because $2L$ independent sublayer outputs add variance to the stream, the unscaled pre-norm stream RMS grows with depth as $\sqrt{1 + 2\ell\sigma^2}$ (to keep the total variance deposited by all $2L$ sublayers $O(1)$ at any depth $L$, GPT-2 :cite:`Radford.Wu.Child.ea.2019` scales the initialization standard deviation of each block's residual output projections `W_o` and `W_2` by $1/\sqrt{2L}$, as we implement in :numref:`sec_gpt` and test in Exercise 4). Meanwhile, in post-norm, BERT :cite:`Devlin.Chang.Lee.ea.2018` slows down the collapse rate :eqref:`eq_collapse-rate` by initializing all linear weights with a smaller standard deviation $\sigma = 0.02$ (so $g^2 = d\sigma^2 \approx 0.10$ at $d = 256$). Let us compare these three configurations (`post-norm`, `pre-norm`, and `post-norm, std 0.02`) across 32 blocks:

```{.python .input #transformer-block-signal-propagation-at-initialization}
%%tab pytorch, jax
arms = {'post-norm': (False, None), 'pre-norm': (True, None),
        'post-norm, std 0.02': (False, 0.02)}
stats, predicted = {}, {}
at = lambda xs: ' '.join(f'{xs[k - 1]:.1e}' for k in (8, 16, 32))
for name, (pre_norm, std) in arms.items():
    stats[name] = s = signal_stats(pre_norm, std)
    print(f'{name}: stream RMS at k=32 {s["rms"][-1]:.2f}')
    print(f'  token spread at k=8,16,32: {at(s["spread"])}')
    if not pre_norm:  # rho_0 = 1/63 for 64 random tokens; Xavier: g = 1
        g2 = 1 if std is None else 256 * std ** 2
        predicted[name] = [(1 + (1 + g2 ** 2) ** k / 63) ** -0.5
                           for k in range(1, 33)]
        print(f'  predicted:                 {at(predicted[name])}')
    print('  grad norms at k=32: ' + '  '.join(
        f'{n} {g:.1e}' for n, g in s['proj'].items()))
```

Plotting the empirical token spread across blocks $k = 1 \dots 32$ alongside the theoretical post-norm decay curve :eqref:`eq_collapse-rate` confirms how post-norm squeezes token representations together while pre-norm preserves unit spread:

```{.python .input #transformer-block-signal-propagation-at-initialization-3}
%%tab pytorch, jax
d2l.plot(list(range(1, 33)), [s['spread'] for s in stats.values()]
         + list(predicted.values()), 'block index k', 'token spread',
         legend=list(arms) + ['predicted'], yscale='log', ylim=(1e-6, 2),
         fmts=('C0-', 'C1-', 'C2-', 'k:', 'k:'), figsize=(4.5, 3.3),
         legend_loc='lower left')
```

Under standard Xavier post-norm (`post-norm`), token spread collapses to nearly zero ($\sim 10^{-4}$) by layer 32, closely tracking our theoretical curve :eqref:`eq_collapse-rate`: all tokens in the sequence have been squeezed into nearly identical vectors. Shrinking the post-norm initialization (`post-norm, std 0.02`) slows the per-block contraction enough to preserve token spread across 32 blocks, while pre-norm (`pre-norm`) keeps token spread $O(1)$ automatically via its unattenuated identity residual path (with stream RMS growing as $\sqrt{L}$ unless paired with $1/\sqrt{2L}$ output-projection scaling; Exercise 4).

### Gradients at Initialization

Plotting the backward gradient norm $\|\partial \mathcal{L} / \partial W_q^{(\ell)}\|_F$ across block index $\ell \in \{1, \dots, 32\}$ reveals the dual problem during backpropagation :cite:`xiong2020layer`:

```{.python .input #transformer-block-signal-propagation-at-initialization-2}
%%tab pytorch, jax
d2l.plot(list(range(1, 33)), [s['grad_q'] for s in stats.values()],
         'block index k', 'query projection gradient norm',
         legend=list(arms), yscale='log', ylim=(1e-12, 1),
         fmts=('C0-', 'C1-', 'C2-'), figsize=(4.5, 3.3),
         legend_loc='lower left')
```

In `post-norm`, differentiating through $2L$ consecutive normalization divisions attenuates the gradient exponentially from layer 32 down to layer 1, starving early layers while leaving top layers with huge gradients (which forces post-norm training to rely on a careful learning-rate warmup schedule). In `pre-norm`, the identity term $I$ in $\partial \mathbf{X}^{(L)} / \partial \mathbf{X}^{(\ell)} = I + \cdots$ delivers a uniform, well-scaled gradient directly from the loss to every block in the stack.

### RMSNorm
:label:`subsec_rmsnorm`

Standard LayerNorm (:numref:`sec_batch_norm`) centers each token vector $\mathbf{x} \in \mathbb{R}^d$ by its mean $\mu = \frac{1}{d}\sum_j x_j$, scales by its standard deviation, and applies learned gain $\boldsymbol{\gamma}$ and bias $\boldsymbol{\beta}$. :citet:`Zhang.Sennrich.2019` showed that mean-centering and the additive bias contribute little to training stability: what matters is normalizing the root-mean-square magnitude. Dropping $\mu$ and $\boldsymbol{\beta}$ yields **RMSNorm**:

$$
\begin{aligned}
\textrm{RMSNorm}(\mathbf{x}) &= \frac{\mathbf{x}}{\textrm{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}, \\
\textrm{RMS}(\mathbf{x}) &= \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon},
\end{aligned}
$$
:eqlabel:`eq_rmsnorm`

where $\boldsymbol{\gamma} \in \mathbb{R}^d$ is initialized to ones and $\epsilon = 10^{-5}$. Because RMSNorm computes only a single mean-of-squares reduction $\frac{1}{d}\sum_j x_j^2$ per token instead of both mean and variance, it performs fewer memory reductions while matching LayerNorm accuracy:

```{.python .input #transformer-block-rmsnorm}
%%tab pytorch
def rms_norm(X, weight=1, eps=1e-5):
    return X / torch.sqrt((X * X).mean(-1, keepdim=True) + eps) * weight

x = torch.tensor([1.0, 2.0, 3.0, 6.0])
rmsnorm, layernorm = nn.RMSNorm(4, eps=1e-5), nn.LayerNorm(4, eps=1e-5)
show = lambda v: ' '.join(f'{a:5.2f}' for a in v.tolist())
print(f'ours vs. nn.RMSNorm: max deviation '
      f'{(rms_norm(x) - rmsnorm(x)).abs().max():.1e}')
for name, v in (('x', x), ('3x', 3 * x), ('x + 10', x + 10)):
    print(f'{name:>6}: RMSNorm {show(rms_norm(v))} | '
          f'LayerNorm {show(layernorm(v))}')
```

```{.python .input #transformer-block-rmsnorm}
%%tab jax
def rms_norm(X, weight=1, eps=1e-5):
    return X / jnp.sqrt((X * X).mean(-1, keepdims=True) + eps) * weight

x = jnp.array([1.0, 2.0, 3.0, 6.0])
rngs = nnx.Rngs(d2l.get_key())  # both start at unit gain and zero shift
rmsnorm = nnx.RMSNorm(4, epsilon=1e-5, rngs=rngs)
layernorm = nnx.LayerNorm(4, epsilon=1e-5, rngs=rngs)
show = lambda v: ' '.join(f'{a:5.2f}' for a in v.tolist())
print(f'ours vs. nnx.RMSNorm: max deviation '
      f'{jnp.abs(rms_norm(x) - rmsnorm(x)).max():.1e}')
for name, v in (('x', x), ('3x', 3 * x), ('x + 10', x + 10)):
    print(f'{name:>6}: RMSNorm {show(rms_norm(v))} | '
          f'LayerNorm {show(layernorm(v))}')
```

### Query--Key Normalization (QK-Norm)
:label:`subsec_qk-norm`

Pre-norm normalizes the input $\mathbf{X}$ entering the attention sublayer, but inside attention the queries $\mathbf{Q} = \textrm{Norm}(\mathbf{X})\mathbf{W}_q$ and keys $\mathbf{K} = \textrm{Norm}(\mathbf{X})\mathbf{W}_k$ are multiplied by learned projection matrices $\mathbf{W}_q, \mathbf{W}_k$. If the singular values of $\mathbf{W}_q$ and $\mathbf{W}_k$ grow during training, the scaled dot-product logits $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ grow quadratically with weight scale, pushing softmax into one-hot saturation where its gradients vanish :cite:`Henry.Dachapally.Pawar.ea.2020,Dehghani.Djolonga.Mustafa.ea.2023`.

**QK-Norm** applies per-head `RMSNorm` of width $d_h$ to $\mathbf{Q}$ and $\mathbf{K}$ immediately before computing attention scores. By Cauchy--Schwarz, two vectors of unit RMS have dot product bounded by $d_h$, so the scaled logit $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ is bounded in magnitude by $\sqrt{d_h}$ (and has unit variance for random directions) regardless of how large $\|\mathbf{W}_q\|$ and $\|\mathbf{W}_k\|$ grow:

```{.python .input #transformer-block-normalizing-queries-and-keys}
%%tab pytorch
def logit_stats(Q, K):
    """Std of the scaled logits; mean entropy of each query's weights."""
    S = Q @ K.T / math.sqrt(Q.shape[-1])
    entropy = -(F.softmax(S, -1) * F.log_softmax(S, -1)).sum(-1).mean()
    return S.std(), entropy

d, d_h = 256, 64
X = torch.randn(64, d)                                # 64 random tokens
W_q, W_k = torch.randn(2, d, d_h) / math.sqrt(d)      # one head
for lam in (1, 4, 16):
    Q, K = X @ (lam * W_q), X @ (lam * W_k)
    s, h = logit_stats(Q, K)
    s_n, h_n = logit_stats(rms_norm(Q), rms_norm(K))
    print(f'lambda {lam:2d}: logit std {s:5.1f}, entropy {h:.2f} | '
          f'QK-norm: logit std {s_n:.2f}, entropy {h_n:.2f}')
```

```{.python .input #transformer-block-normalizing-queries-and-keys}
%%tab jax
def logit_stats(Q, K):
    """Std of the scaled logits; mean entropy of each query's weights."""
    S = Q @ K.T / math.sqrt(Q.shape[-1])
    entropy = -(jax.nn.softmax(S) * jax.nn.log_softmax(S)).sum(-1).mean()
    return S.std(), entropy

d, d_h = 256, 64
X = jax.random.normal(d2l.get_key(), (64, d))         # 64 random tokens
W_q, W_k = jax.random.normal(d2l.get_key(), (2, d, d_h)) / math.sqrt(d)
for lam in (1, 4, 16):
    Q, K = X @ (lam * W_q), X @ (lam * W_k)
    s, h = logit_stats(Q, K)
    s_n, h_n = logit_stats(rms_norm(Q), rms_norm(K))
    print(f'lambda {lam:2d}: logit std {s:5.1f}, entropy {h:.2f} | '
          f'QK-norm: logit std {s_n:.2f}, entropy {h_n:.2f}')
```

When $Q$ and $K$ are scaled by $\lambda = 4\times$ and $\lambda = 16\times$ without QK-Norm, the logit standard deviation jumps quadratically ($\approx 16$ and $\approx 256$) and softmax entropy collapses toward zero; with `rms_norm` applied to $Q$ and $K$, logit spread and attention entropy remain completely invariant to the weight scale.

## The Feed-Forward Network: From GELU to SwiGLU

The original Transformer feed-forward network is a two-layer MLP that expands channel width from $d$ to $d_{\textrm{ff}} = 4d$, applies an elementwise activation ($\textrm{ReLU}$ in :citet:`Vaswani.Shazeer.Parmar.ea.2017`, $\textrm{GELU}$ in GPT-2 and BERT), and projects back to $d$:

$$
\begin{aligned}
\textrm{FFN}_{\textrm{GELU}}(\mathbf{x}) &= \textrm{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2, \\
\mathbf{W}_1 &\in \mathbb{R}^{d \times 4d}, \qquad \mathbf{W}_2 \in \mathbb{R}^{4d \times d},
\end{aligned}
$$
:eqlabel:`eq_ffn`

using $2 \cdot d \cdot 4d = 8d^2$ weight parameters.

Following :citet:`Dauphin.Fan.Auli.ea.2017` and :citet:`Shazeer.2020`, modern Transformers replace :eqref:`eq_ffn` with a bias-free **SwiGLU** gated linear unit, which takes the elementwise product of a $\textrm{SiLU}$-activated gate projection $\mathbf{x}\mathbf{W}_g$ ($\textrm{SiLU}(z) = z \sigma(z)$) and a linear up-projection $\mathbf{x}\mathbf{W}_1$ before projecting back down to $\mathbb{R}^d$ with $\mathbf{W}_2$:

$$
\textrm{FFN}_{\textrm{SwiGLU}}(\mathbf{x}) = \bigl(\textrm{SiLU}(\mathbf{x}\mathbf{W}_g) \odot (\mathbf{x}\mathbf{W}_1)\bigr)\mathbf{W}_2.
$$
:eqlabel:`eq_swiglu`

Because `SwiGLU` uses *three* matrices of shape $d \times d_{\textrm{ff}}$ ($\mathbf{W}_g, \mathbf{W}_1$) and $d_{\textrm{ff}} \times d$ ($\mathbf{W}_2$), setting $d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$ gives $3 d d_{\textrm{ff}} \approx 8d^2$, matching both the parameter count and the FLOP count of the $4d$ GELU MLP:

```{.python .input #transformer-block-the-feed-forward-network-1}
%%tab pytorch
class FeedForward(nn.Module):  #@save
    """Position-wise FFN: GELU MLP or SwiGLU at matched parameter count."""
    def __init__(self, num_hiddens, act='swiglu', bias=False):
        super().__init__()
        assert act in ('swiglu', 'gelu'), f'unknown act: {act!r}'
        self.act = act
        if act == 'gelu':
            width = 4 * num_hiddens
        else:  # 'swiglu': three matrices; width 8d/3 matches the MLP budget
            width = round(8 * num_hiddens / 3)
            self.W_g = nn.Linear(num_hiddens, width, bias=bias)
        self.W_1 = nn.Linear(num_hiddens, width, bias=bias)
        self.W_2 = nn.Linear(width, num_hiddens, bias=bias)

    def forward(self, X):
        if self.act == 'gelu':  # The tanh approximation, as in GPT-2
            return self.W_2(F.gelu(self.W_1(X), approximate='tanh'))
        return self.W_2(F.silu(self.W_g(X)) * self.W_1(X))
```

```{.python .input #transformer-block-the-feed-forward-network-1}
%%tab jax
class FeedForward(nnx.Module):  #@save
    """Position-wise FFN: GELU MLP or SwiGLU at matched parameter count."""
    def __init__(self, num_hiddens, act='swiglu', bias=False, *, rngs):
        assert act in ('swiglu', 'gelu'), f'unknown act: {act!r}'
        self.act = act
        if act == 'gelu':
            width = 4 * num_hiddens
        else:  # 'swiglu': three matrices; width 8d/3 matches the MLP budget
            width = round(8 * num_hiddens / 3)
            self.W_g = d2l.Linear(num_hiddens, width, use_bias=bias,
                                  rngs=rngs)
        self.W_1 = d2l.Linear(num_hiddens, width, use_bias=bias, rngs=rngs)
        self.W_2 = d2l.Linear(width, num_hiddens, use_bias=bias, rngs=rngs)

    def __call__(self, X):
        if self.act == 'gelu':  # The tanh approximation, as in GPT-2
            return self.W_2(jax.nn.gelu(self.W_1(X), approximate=True))
        return self.W_2(jax.nn.silu(self.W_g(X)) * self.W_1(X))
```

## A Configurable `TransformerBlock`

We now assemble `TransformerBlock` combining pre-norm or post-norm placement (`pre_norm`), `RMSNorm` or `LayerNorm` (`norm`), `SwiGLU` or `GELU` FFN (`act`), bias toggle (`bias`), and pluggable `attn_factory` and `ffn_factory` hooks (which allow upcoming sections to drop in causal RoPE attention in :numref:`sec_gpt` or Mixture-of-Experts layers in :numref:`sec_moe` without rewriting the block):

```{.python .input #transformer-block-a-configurable-block}
%%tab pytorch
class TransformerBlock(nn.Module):  #@save
    """Configurable transformer block: attention + FFN on a residual
    stream."""
    def __init__(self, num_hiddens, num_heads, *, dropout=0, norm='rms',
                 act='swiglu', pre_norm=True, bias=False, attn_factory=None,
                 ffn_factory=None):
        super().__init__()
        assert norm in ('rms', 'layer'), f'unknown norm: {norm!r}'
        assert num_hiddens % num_heads == 0
        self.pre_norm = pre_norm
        make_norm = nn.RMSNorm if norm == 'rms' else nn.LayerNorm
        self.norm1 = make_norm(num_hiddens, eps=1e-5)
        self.norm2 = make_norm(num_hiddens, eps=1e-5)
        self.attention = (d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout, bias=bias)
                          if attn_factory is None else attn_factory())
        self.ffn = (FeedForward(num_hiddens, act, bias=bias)
                    if ffn_factory is None else ffn_factory())
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None, **attn_args):
        # attn_args go on to the attention, e.g. the RoPE tables of d2l.GPT
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.dropout(self.attention(Y, Y, Y, mask, **attn_args))
            return X + self.dropout(self.ffn(self.norm2(X)))
        X = self.norm1(X + self.dropout(self.attention(X, X, X, mask,
                                                       **attn_args)))
        return self.norm2(X + self.dropout(self.ffn(X)))
```

```{.python .input #transformer-block-a-configurable-block}
%%tab jax
class TransformerBlock(nnx.Module):  #@save
    """Configurable transformer block: attention + FFN on a residual
    stream."""
    def __init__(self, num_hiddens, num_heads, *, dropout=0, norm='rms',
                 act='swiglu', pre_norm=True, bias=False, attn_factory=None,
                 ffn_factory=None, rngs):
        assert norm in ('rms', 'layer'), f'unknown norm: {norm!r}'
        assert num_hiddens % num_heads == 0
        self.pre_norm = pre_norm
        make_norm = nnx.RMSNorm if norm == 'rms' else nnx.LayerNorm
        self.norm1 = make_norm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.norm2 = make_norm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.attention = (d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout, bias=bias,
                                                 rngs=rngs)
                          if attn_factory is None else attn_factory(rngs))
        self.ffn = (FeedForward(num_hiddens, act, bias=bias, rngs=rngs)
                    if ffn_factory is None else ffn_factory(rngs))
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, X, mask=None, **attn_args):
        # attn_args go on to the attention, e.g. the RoPE tables of d2l.GPT
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.dropout(self.attention(Y, Y, Y, mask, **attn_args))
            return X + self.dropout(self.ffn(self.norm2(X)))
        X = self.norm1(X + self.dropout(self.attention(X, X, X, mask,
                                                       **attn_args)))
        return self.norm2(X + self.dropout(self.ffn(X)))
```

### Parameter Count

Comparing the default modern block (`RMSNorm`, `SwiGLU`, `bias=False`) against the 2017/GPT-2 variants (`GELU`, `LayerNorm`, `bias=True`) at $d = 256, H = 8$ confirms that every configuration holds $\approx 12 d^2 = 786{,}432$ parameters ($4d^2$ in attention and $8d^2$ in the FFN):

```{.python .input #transformer-block-shapes-and-parameters-1}
%%tab pytorch
blocks = {'default': TransformerBlock(256, 8),
          'GELU': TransformerBlock(256, 8, act='gelu'),
          '2017': TransformerBlock(256, 8, norm='layer', act='gelu',
                                   pre_norm=False, bias=True)}
for name, blk in blocks.items():
    attn, ffn, total = (d2l.num_params(m)
                        for m in (blk.attention, blk.ffn, blk))
    print(f'{name:>7}: attention {attn:,}, FFN {ffn:,}, total {total:,} '
          f'= {total / 256 ** 2:.2f} d^2')
```

```{.python .input #transformer-block-shapes-and-parameters-1}
%%tab jax
rngs = nnx.Rngs(d2l.get_key())
blocks = {'default': TransformerBlock(256, 8, rngs=rngs),
          'GELU': TransformerBlock(256, 8, act='gelu', rngs=rngs),
          '2017': TransformerBlock(256, 8, norm='layer', act='gelu',
                                   pre_norm=False, bias=True, rngs=rngs)}
for name, blk in blocks.items():
    attn, ffn, total = (d2l.num_params(m)
                        for m in (blk.attention, blk.ffn, blk))
    print(f'{name:>7}: attention {attn:,}, FFN {ffn:,}, total {total:,} '
          f'= {total / 256 ** 2:.2f} d^2')
```

## Summary

* A **Transformer block** separates **token mixing** (multi-head attention across sequence positions) from **channel mixing** (a position-wise FFN across the $d$ features), adding both updates to a shared residual stream.
* **Pre-norm** (:eqref:`eq_preln-block`) normalizes only the input to each sublayer, preserving an unattenuated identity path for both forward token diversity and backward gradients, whereas **post-norm** (:eqref:`eq_postln-block`) normalizes the residual stream itself and exponentially attenuates early-layer gradients and inter-token differences (:eqref:`eq_collapse-rate`).
* Scaling the initialization standard deviation of each block's two residual output projections (`W_o` and `W_2`) by **$1/\sqrt{2L}$** keeps pre-norm residual stream variance $O(1)$ across arbitrary depth $L$.
* **RMSNorm** (:eqref:`eq_rmsnorm`) drops LayerNorm's mean subtraction and additive bias, while per-head **QK-Norm** bounds attention logits by Cauchy--Schwarz regardless of projection weight growth.
* **SwiGLU** (:eqref:`eq_swiglu`) replaces the two-matrix $4d$ FFN with a three-matrix gated product at width $d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$, matching the $8d^2$ parameter and FLOP budget of a standard FFN.

## Exercises

1. **Scale invariance of RMSNorm and its Jacobian.** Let $\mathbf{y} = \textrm{RMSNorm}(\mathbf{x})$ act as in :eqref:`eq_rmsnorm` with $\epsilon = 0$ and $\boldsymbol{\gamma} = \mathbf{1}$.
   1. Show that $\textrm{RMSNorm}(c\mathbf{x}) = \textrm{RMSNorm}(\mathbf{x})$ for every scalar $c > 0$.
   2. Differentiate $\mathbf{y} = \sqrt{d}\, \mathbf{x} / \|\mathbf{x}\|_2$ to show that the Jacobian is $J(\mathbf{x}) = \frac{\sqrt{d}}{\|\mathbf{x}\|_2}\bigl(\mathbf{I} - \frac{\mathbf{x}\mathbf{x}^\top}{\|\mathbf{x}\|_2^2}\bigr)$, and verify that $J(\mathbf{x})\mathbf{x} = \mathbf{0}$. Why must any radial perturbation parallel to $\mathbf{x}$ lie in the null space of $J(\mathbf{x})$?

2. **Unrolled residual stream and gradient identity path.** Number the $2L$ sublayers of an $L$-block pre-norm stack as $f_1, f_2, \dots, f_{2L}$, where odd indices are attention sublayers and even indices are FFN sublayers.
   1. Unroll :eqref:`eq_preln-block` to express $\mathbf{X}^{(L)}$ explicitly as $\mathbf{X}^{(0)} + \sum_{m=1}^{2L} f_m(\textrm{Norm}(\mathbf{X}^{(m-1)/2}))$.
   2. Apply the chain rule to write $\partial \mathbf{X}^{(L)} / \partial \mathbf{X}^{(0)}$ as a product of $2L$ Jacobians, expand the leading terms, and contrast the resulting identity term $\mathbf{I}$ with the post-norm product where every factor is premultiplied by $J_{\textrm{Norm}}$.

3. **QK-Norm inside `TransformerBlock`.** Write an `attn_factory` that wraps `d2l.MultiHeadAttention` and applies `nn.RMSNorm(d // h)` (or `nnx.RMSNorm`) to queries and keys per head before computing attention scores. Verify on a 16-block stack with projection weights multiplied by $4\times$ (or $16\times$) that attention entropy remains bounded away from zero.

4. **Stream variance budget across depth.** In the pre-norm arrangement, every sublayer adds an update to the residual stream. Run `signal_stats(pre_norm=True, std=0.02 / math.sqrt(2 * L), num_blks=L)` for $L \in \{8, 16, 32, 64\}$ and compare against unscaled `std=None`. Verify that the final stream standard deviation at block $L$ grows as $\sqrt{L}$ without scaling but remains near a constant independent of $L$ under the $1/\sqrt{2L}$ rule.

5. **Rate of token collapse.** :numref:`subsec_signal-propagation` argued that in a post-norm block, only the attention sublayer mixes across tokens, while the FFN acts position-wise. Modify `MiniBlock` so that the attention sublayer is skipped ($\mathbf{H} = \textrm{Norm}(\mathbf{X})$) and only the post-norm FFN runs, and re-run `signal_stats(pre_norm=False)`. Does token spread still collapse to zero across 32 layers, or does it stabilize? Explain how this confirms each factor in :eqref:`eq_collapse-rate`.
:exlabel:`ex_transformer-block-ffn-width-rounding`

[Discussions](https://d2l.discourse.group/t/36774)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.1]{.kicker}

The Transformer Block: Pre-Norm, SwiGLU, and Signal Propagation<br>**residual stream · pre-norm vs. post-norm · token collapse · RMSNorm & QK-Norm · SwiGLU · 12d² census**
:::
:::

::: {.slide title="Why Stacking Attention Alone Fails at Depth"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **No per-token nonlinearity:** Without a position-wise FFN between attention layers, linear projections collapse to a single low-rank map on each token's channels.
- **Token collapse & starved gradients:** Stacking $L = 32$ post-norm blocks with Xavier initialization collapses token spread to $\sim 10^{-4}$ and drops layer-1 query gradients $\|\partial \mathcal{L} / \partial \mathbf{W}_q^{(1)}\|_F$ to $\sim 10^{-9}$!
:::

::: {.col .fig .big}
@fig:ch11-tf-blk-failure-modes
:::
:::
:::

::: {.slide title="Token Mixing vs. Channel Mixing on a Residual Stream"}
[Block Anatomy]{.kicker}

::: {.cols .vc}
::: {.col}
- **Shared residual stream $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$:** Rather than overwriting representations, each sublayer reads from the stream, computes an update in $\mathbb{R}^{B \times n \times d}$, and adds it back.
- **Multi-head attention (token mixing):** The *only* operator where sequence positions $1, \dots, n$ communicate ($4d^2$ parameters).
- **Position-wise FFN (channel mixing):** Applies an identical MLP to each token vector $\mathbf{x}_i \in \mathbb{R}^d$ independently, transforming the context gathered by attention ($8d^2$ parameters, or two-thirds of the block).
:::

::: {.col .fig}
@fig:mdl-transformers-block-anatomy
:::
:::
:::

::: {.slide title="Normalization Placement: Post-Norm vs. Pre-Norm"}
[Normalization Placement]{.kicker}

::: {.cols .vc}
::: {.col}
- **Post-norm (2017 / BERT):** Places normalization on the main residual stream itself:
  $$\mathbf{H} = \mathrm{Norm}_1(\mathbf{X} + \mathrm{Attn}(\mathbf{X})), \quad \mathbf{Y} = \mathrm{Norm}_2(\mathbf{H} + \mathrm{FFN}(\mathbf{H}))$$
- **Pre-norm (GPT-2 / Llama / Qwen):** Moves normalization onto each sublayer's input branch so the main stream runs uninterrupted:
  $$\mathbf{H} = \mathbf{X} + \mathrm{Attn}(\mathrm{Norm}_1(\mathbf{X})), \quad \mathbf{Y} = \mathbf{H} + \mathrm{FFN}(\mathrm{Norm}_2(\mathbf{H}))$$
:::

::: {.col .fig}
@fig:ch11-tf-blk-post-vs-pre-eq
:::
:::
:::

::: {.slide title="Post-Norm and Pre-Norm Differ by Two Lines of Code"}
[Implementation]{.kicker}

Our diagnostic `MiniBlock` uses identical weights for both placements, switching only whether `norm1` and `norm2` wrap the residual sum or the branch inputs:

```{.python #transformer-block-exercises-1}
def forward(self, X):
    if self.pre_norm:                      # Pre-norm: normalize each branch input
        Y = self.norm1(X)
        X = X + self.attention(Y, Y, Y)
        return X + self.ffn(self.norm2(X))
    X = self.norm1(X + self.attention(X, X, X))  # Post-norm: normalize the stream
    return self.norm2(X + self.ffn(X))
```

::: {.d2l-note}
Because an untrained $L$-block stack is a deterministic random function, we can measure how both placements propagate signal variance, inter-token diversity, and gradients at initialization ($t = 0$).
:::
:::

::: {.slide title="Why Post-Norm Squeezes Tokens Together Across Depth"}
[Mechanism of Token Collapse]{.kicker}

::: {.cols .vc}
::: {.col}
- Decompose each token vector $\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$ into the sequence mean $\boldsymbol{\mu} = \frac{1}{n}\sum_j \mathbf{x}_j$ and token-specific deviation $\boldsymbol{\delta}_i$.
- **Random attention reinforces $\boldsymbol{\mu}$:** Uniform initial weights add $\mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$ to every token while deviations $\boldsymbol{\delta}_i$ cancel.
- **Post-norm rescales downward:** Dividing by $\sqrt{1 + \sigma_{\text{attn}}^2}$ shrinks $\boldsymbol{\delta}_i$ at every layer (`LayerNorm` centers *channels*, not tokens!).
:::

::: {.col .fig .big}
@fig:ch11-tf-blk-token-collapse
:::
:::
:::

::: {.slide title="Predicting the Exponential Rate of Token Collapse"}
[Mathematical Derivation]{.kicker}

::: {.cols .vc}
::: {.col}
- Let $\rho = n\|\boldsymbol{\mu}\|^2 / \sum_{i=1}^n \|\boldsymbol{\delta}_i\|^2$ be the ratio of common-to-distinct token energy, so relative token spread is $s = (1 + \rho)^{-1/2}$.
- With per-projection variance gain $g^2 = d\sigma^2$, each post-norm block amplifies $\rho \mapsto (1 + g^4)\rho$:
  $$s_k \approx \left(1 + \frac{(1 + g^4)^k}{63}\right)^{-1/2}$$
:::

::: {.col .fig}
@fig:ch11-tf-blk-collapse-law
:::
:::
:::

::: {.slide title="Measured Token Spread Matches the Analytical Law"}
[Empirical Verification]{.kicker}

::: {.cols .vc}
::: {.col}
- **Post-norm + Xavier (`C0`):** Token spread decays exponentially across depth, falling to $\sim 10^{-4}$ by block 32 (slightly below the attention-only dotted curve because `GELU`'s positive mean adds a small shared vector in the FFN too).
- **Post-norm + $\mathcal{N}(0, 0.02^2)$ (`C2`):** Small weights keep $g^4 \approx 0.01$, holding token spread at $0.97$ through 32 layers at width $d = 256$.
- **Pre-norm + Xavier (`C1`):** Even with large Xavier weights ($g = 1$), pre-norm preserves substantial token diversity ($s_{32} \approx 0.33$) without exponential collapse.
:::

::: {.col .fig}
@!transformer-block-signal-propagation-at-initialization-3
:::
:::
:::

::: {.slide title="Pitfall: Fixed Weight Scale 0.02 Grows With Model Width"}
[Width Scaling Pitfall]{.kicker}

::: {.cols .vc}
::: {.col}
- For $\mathbf{W} \in \mathbb{R}^{d \times d}$ with fixed entry variance $\sigma^2 = 0.02^2$, the per-projection variance gain grows linearly with width $d$:
  $$g^2 = d \sigma^2 = 0.0004\,d$$
- A fixed $\sigma = 0.02$ is only small for $d \ll 2{,}500$; at frontier widths ($d = 4{,}096$), $g^2 = 1.64 > 1$ collapses post-norm stacks even faster than Xavier!
:::

::: {.col .fig}
@fig:ch11-tf-blk-width-pitfall
:::
:::
:::

::: {.slide title="Token Collapse Starves Query and Key Gradients"}
[Backward Signal Propagation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Vanishing $\mathbf{W}_q, \mathbf{W}_k$ gradients at the top:** When incoming tokens $\mathbf{x}_1, \dots, \mathbf{x}_n$ at block 32 are nearly identical, perturbing $\mathbf{W}_q$ or $\mathbf{W}_k$ shifts all attention scores equally and leaves the softmax weighted average unchanged ($\|\nabla_{\mathbf{W}_q^{(32)}} \mathcal{L}\|_F \sim 10^{-9}$).
- **Exponential attenuation toward layer 1:** In post-norm, backpropagating through $2L$ normalization Jacobians attenuates gradients across depth, whereas **pre-norm** (`C1`) delivers a flat, uniform gradient norm to every block.
:::

::: {.col .fig}
@!transformer-block-signal-propagation-at-initialization-2
:::
:::
:::

::: {.slide title="Pre-Norm Unrolled Stream and 1/√(2L) Residual Scaling"}
[Pre-Norm Variance Control]{.kicker}

::: {.cols .vc}
::: {.col}
- Unrolling $L$ pre-norm blocks reveals an uninterrupted identity highway:
  $$\mathbf{X}^{(L)} = \mathbf{X}^{(0)} + \sum_{\ell=1}^{L} \bigl(\mathrm{Attn}_\ell(\mathrm{Norm}(\mathbf{X}^{(\ell-1)})) + \mathrm{FFN}_\ell(\mathrm{Norm}(\mathbf{H}^{(\ell)}))\bigr)$$
- **GPT-2's $1/\sqrt{2L}$ scaling rule:** Scaling the initial std of the two residual write matrices per block (`W_o` and `W_2`) by $1/\sqrt{2L}$ keeps cumulative stream variance $\sum_{m=1}^{2L} \frac{\sigma^2}{2L} = \sigma^2 = O(1)$ at any depth $L$.
:::

::: {.col .fig}
@fig:ch11-tf-blk-unrolled-highway
:::
:::
:::

::: {.slide title="RMSNorm: Dropping Mean-Centering for Speed"}
[Normalization Form]{.kicker}

Zhang & Sennrich (2019) showed that `LayerNorm`'s stability comes from bounding root-mean-square magnitude rather than centering the channel mean $\mu = \frac{1}{d}\sum_j x_j$:

$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}, \qquad \mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}$$

@!transformer-block-rmsnorm

::: {.d2l-note}
**Scale vs. shift invariance:** Both norms are invariant to positive rescaling ($\mathbf{x} \mapsto 3\mathbf{x}$), whereas adding a constant shift ($\mathbf{x} \mapsto \mathbf{x} + 10$) alters `RMSNorm` but leaves `LayerNorm` unchanged. Dropping $\mu$ and $\boldsymbol{\beta}$ saves one reduction pass over HBM per token.
:::
:::

::: {.slide title="Query–Key Normalization (QK-Norm) Bounds Logits"}
[Attention Stability]{.kicker}

::: {.cols .vc}
::: {.col}
- **Quadratic logit blow-up:** If the spectral norms of $\mathbf{W}_q$ and $\mathbf{W}_k$ grow by $\lambda$ during training, attention logits $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ grow as **$\lambda^2$**, saturating softmax.
- **Per-head `QK-Norm`:** Applying width-$d_h$ `RMSNorm` to queries and keys enforces $\|\hat{\mathbf{q}}_i\|_2 = \|\hat{\mathbf{k}}_j\|_2 = \sqrt{d_h}$, imposing a hard Cauchy–Schwarz bound:
  $$\frac{|\hat{\mathbf{q}}_i^\top \hat{\mathbf{k}}_j|}{\sqrt{d_h}} \le \sqrt{d_h}$$
:::

::: {.col .fig}
@fig:ch11-tf-blk-qknorm-bound
:::
:::
:::

::: {.slide title="Empirical Logit and Entropy Invariance Under QK-Norm"}
[QK-Norm Verification]{.kicker}

Scaling $\mathbf{W}_q$ and $\mathbf{W}_k$ by $\lambda \in \{1, 4, 16\}$ on $n = 64$ random tokens ($d_h = 64$, uniform entropy $\ln 64 \approx 4.16$ nats) demonstrates softmax collapse and `QK-Norm` recovery:

@-transformer-block-normalizing-queries-and-keys
:::

::: {.slide title="QK-Norm Holds Logit Spread and Entropy Fixed"}
[QK-Norm Verification]{.kicker}

@!transformer-block-normalizing-queries-and-keys

::: {.d2l-note}
Without `QK-Norm`, a $16\times$ weight growth inflates logit standard deviation by $16^2 = 256\times$ and collapses attention entropy to $\approx 0.01\text{--}0.02$ nats (one-hot); with `QK-Norm`, logit spread ($\approx 1.0$) and entropy ($\approx 3.7$) remain completely invariant.
:::
:::

::: {.slide title="The Feed-Forward Network as a Key–Value Memory"}
[Channel Mixing]{.kicker}

::: {.cols .vc}
::: {.col}
- Writing the two-layer FFN in terms of rows $\mathbf{w}_{1,j} \in \mathbb{R}^d$ of $\mathbf{W}_1$ and columns $\mathbf{w}_{2,j} \in \mathbb{R}^d$ of $\mathbf{W}_2$ reveals a content-addressed memory bank (Geva et al., 2021):
  $$\mathrm{FFN}_{\mathrm{GELU}}(\mathbf{x}) = \sum_{j=1}^{4d} \mathrm{GELU}(\mathbf{x}^\top \mathbf{w}_{1,j})\, \mathbf{w}_{2,j}$$
- **Dominant parameter share:** Two $d \times 4d$ matrices hold $8d^2$ parameters ($3{,}072$ detector slots per block in GPT-2 Small).
:::

::: {.col .fig}
@fig:ch11-tf-blk-ffn-kv-memory
:::
:::
:::

::: {.slide title="From GELU MLPs to Gated SwiGLU Networks"}
[Gated Feed-Forward Networks]{.kicker}

::: {.cols .vc}
::: {.col}
- Modern Transformers replace the two-matrix `GELU` MLP with a bias-free gated **`SwiGLU`** network (Shazeer, 2020):
  $$\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x}) = \bigl(\mathrm{SiLU}(\mathbf{x}\mathbf{W}_g) \odot (\mathbf{x}\mathbf{W}_1)\bigr)\mathbf{W}_2$$
- **Matching the $8d^2$ budget:** Shrinking hidden width from $4d$ to $d_{\text{ff}} = \operatorname{round}(\frac{8}{3}d)$ keeps the 3-matrix parameter count at $3 \times d \times \frac{8}{3}d = 8d^2$.
:::

::: {.col .fig}
@fig:ch11-tf-blk-gelu-vs-swiglu
:::
:::
:::

::: {.slide title="Both FFN Variants in One Parameter-Matched Class"}
[FeedForward Implementation]{.kicker}

Our `FeedForward` module switches cleanly between a $4d$ `GELU` MLP and an $\frac{8}{3}d$ `SwiGLU` network at the exact same $8d^2$ parameter and FLOP budget:

```{.python #transformer-block-exercises-2}
width = 4 * num_hiddens if act == 'gelu' else round(8 * num_hiddens / 3)

def forward(self, X):
    if self.act == 'gelu':                        # 2 matrices of width 4d -> 8d^2
        return self.W_2(F.gelu(self.W_1(X), approximate='tanh'))
    return self.W_2(F.silu(self.W_g(X)) * self.W_1(X))  # 3 matrices of width 8d/3 -> 8d^2
```

::: {.d2l-note}
At $d = 256$, `act='gelu'` uses $2 \times 256 \times 1024 = 524{,}288$ parameters, while `act='swiglu'` uses $3 \times 256 \times 683 = 524{,}544$ parameters ($8.004\,d^2$ due to rounding $\frac{8}{3}\cdot 256 = 682.67 \to 683$).
:::
:::

::: {.slide title="One Configurable TransformerBlock & Parameter Census"}
[Unified Block & Census]{.kicker}

`TransformerBlock` unifies `norm` (`'rms'` or `'layer'`), `act` (`'swiglu'` or `'gelu'`), `pre_norm`, `bias`, and pluggable `attn_factory` / `ffn_factory` hooks, always holding $\approx 12d^2$ parameters:

@transformer-block-shapes-and-parameters-1

::: {.d2l-note .rule}
**The $12d^2$ rule of thumb:** In every dense block with $H_{\text{kv}} = H$, multi-head attention holds **$4d^2$** parameters ($\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$) and the FFN holds **$8d^2$** parameters, totaling **$12d^2$** per layer ($7.08\text{M}$ per block at GPT-2's $d = 768$).
:::
:::

::: {.slide title="Concept Check: Designing a Transformer Block"}
[Check Your Understanding]{.kicker}

Work out the following three design questions:

1. **Unfair ablation trap:** A researcher replaces a $4d$ `GELU` FFN at $d = 768$ with a `SwiGLU` FFN but forgets to change the hidden width from $4d = 3{,}072$ to $\frac{8}{3}d = 2{,}048$. How many extra parameters per block did they accidentally add?
2. **`QK-Norm` scaling:** Without `QK-Norm`, multiplying $\mathbf{W}_q$ and $\mathbf{W}_k$ by $\lambda = 8$ scales attention logit standard deviation by what factor? What is the factor with `QK-Norm`?
3. **Depth scaling:** In a 32-block pre-norm model, by what factor does GPT-2 scale the initial standard deviation of $\mathbf{W}_o$ and $\mathbf{W}_2$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **$+4d^2 \approx 2.36\text{M}$ extra parameters per block** ($12d^2 = 7.08\text{M}$ vs. $8d^2 = 4.72\text{M}$, a $50\%$ FFN size increase), confounding architecture quality with model capacity.
2. **$\lambda^2 = 64\times$** without `QK-Norm` (driving softmax to one-hot); **$1\times$ (invariant)** with `QK-Norm`.
3. **$1 / \sqrt{2L} = 1 / \sqrt{64} = 1/8 = 0.125$**, keeping the variance sum of all $2L = 64$ residual branches $O(1)$.
:::
:::

::: {.slide title="Summary: Anatomy and Stability of the Transformer Block"}
[Key Takeaways]{.kicker}

::: {.cols .vc}
::: {.col}
- **Token Mixing ($4d^2$) + Channel Mixing ($8d^2$):** Multi-head attention and `SwiGLU` FFN ($d_{\text{ff}} = \frac{8}{3}d$) total $12d^2$ parameters per block.
- **Pre-Norm + $1/\sqrt{2L}$ Output Scaling:** Preserves the identity gradient highway while keeping cumulative stream variance $O(1)$ across depth $L$.
- **`RMSNorm` & `QK-Norm`:** Fast scale normalization and Cauchy–Schwarz logit bound ($\le \sqrt{d_h}$).
:::

::: {.col .fig .big}
@fig:ch11-tf-blk-summary
:::
:::
:::
