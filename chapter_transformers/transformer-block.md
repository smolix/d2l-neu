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
\mathbf{H} = \mathbf{X} + \textrm{Attn}\!\bigl(\textrm{Norm}(\mathbf{X})\bigr), \qquad
\mathbf{Y} = \mathbf{H} + \textrm{FFN}\!\bigl(\textrm{Norm}(\mathbf{H})\bigr).
$$
:eqlabel:`eq_preln-block`

![The transformer block: attention and a position-wise FFN read from the residual stream and add their results back. Left, the 2017 arrangement: normalization sits on the stream itself, after each addition. Right, the modern arrangement: normalization moves onto each branch, and the stream runs uninterrupted from input to output.](../img/mdl-transformers-block-anatomy.svg)
:label:`fig_transformer-block`

## Normalization

### Pre-Norm vs. Post-Norm

The original 2017 Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017` and BERT :cite:`Devlin.Chang.Lee.ea.2018` used **post-norm**, placing normalization *on the residual stream itself* after each addition:

$$
\mathbf{H} = \textrm{Norm}\!\bigl(\mathbf{X} + \textrm{Attn}(\mathbf{X})\bigr), \qquad
\mathbf{Y} = \textrm{Norm}\!\bigl(\mathbf{H} + \textrm{FFN}(\mathbf{H})\bigr).
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

Under pre-norm (:eqref:`eq_preln-block`), the residual stream is never divided by a normalization denominator: unrolling $L$ blocks gives a direct sum $\mathbf{X}^{(L)} = \mathbf{X}^{(0)} + \sum_{\ell=1}^{2L} \mathbf{U}^{(\ell)}$. Because $2L$ independent sublayer outputs add variance to the stream, the unscaled pre-norm stream variance grows linearly with depth ($\textrm{Var}(\mathbf{X}^{(\ell)}) \approx 1 + 2\ell \sigma^2$). To keep the total variance deposited by all $2L$ sublayers $O(1)$ at any depth $L$, GPT-2 :cite:`Radford.Wu.Child.ea.2019` scales the initialization standard deviation of the two residual output projections in each block (`W_o` and `W_2`) by **$1/\sqrt{2L}$** ($\sigma = 0.02 / \sqrt{2L}$). We compare all four configurations across 32 blocks:

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

```{.python .input #transformer-block-signal-propagation-at-initialization-3}
%%tab pytorch, jax
d2l.plot(list(range(1, 33)), [s['spread'] for s in stats.values()]
         + list(predicted.values()), 'block index k', 'token spread',
         legend=list(arms) + ['predicted'], yscale='log', ylim=(1e-6, 2),
         fmts=('C0-', 'C1-', 'C2-', 'k:', 'k:'), figsize=(4.5, 3.3),
         legend_loc='lower left')
```

Under standard post-norm (`post-norm`), token spread collapses to nearly zero by layer 32: all tokens in the sequence have been squeezed into nearly identical vectors. Under pre-norm with $1/\sqrt{2L}$ residual scaling (`pre-norm, 1/sqrt(2L)`), both the stream standard deviation and the token spread remain flat near $1.0$ across all 32 blocks.

### Gradients at Initialization

Plotting the backward gradient norm $\|\partial \mathcal{L} / \partial W_q^{(\ell)}\|_F$ across block index $\ell \in \{1, \dots, 32\}$ reveals the dual problem during backpropagation :cite:`Xiong.Yang.He.ea.2020`:

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
\textrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\textrm{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}, \qquad
\textrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon},
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

When $Q$ and $K$ are scaled by $5\times$ without QK-Norm, the logit standard deviation jumps to $25$ and softmax entropy collapses toward zero; with `rms_norm` applied to $Q$ and $K$, logit spread and attention entropy remain completely invariant to the weight scale.

## The Feed-Forward Network: From GELU to SwiGLU

The original Transformer feed-forward network is a two-layer MLP that expands channel width from $d$ to $d_{\textrm{ff}} = 4d$, applies an elementwise activation ($\textrm{ReLU}$ in :citet:`Vaswani.Shazeer.Parmar.ea.2017`, $\textrm{GELU}$ in GPT-2 and BERT), and projects back to $d$:

$$
\textrm{FFN}_{\textrm{GELU}}(\mathbf{x}) = \textrm{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2, \qquad \mathbf{W}_1 \in \mathbb{R}^{d \times 4d}, \; \mathbf{W}_2 \in \mathbb{R}^{4d \times d},
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

3. **QK-Norm inside `TransformerBlock`.** Write an `attn_factory` that wraps `d2l.MultiHeadAttention` and applies `nn.RMSNorm(d // h)` (or `nnx.RMSNorm`) to queries and keys per head before computing attention scores. Verify on a 16-block stack with projection weights multiplied by $5\times$ that attention entropy remains bounded away from zero.

4. **Stream variance budget across depth.** In the pre-norm arrangement, every sublayer adds an update to the residual stream. Run `signal_stats(pre_norm=True, std=0.02 / math.sqrt(2 * L), num_blks=L)` for $L \in \{8, 16, 32, 64\}$ and compare against unscaled `std=None`. Verify that the final stream standard deviation at block $L$ grows as $\sqrt{L}$ without scaling but remains near a constant independent of $L$ under the $1/\sqrt{2L}$ rule.

5. **Rate of token collapse.** :numref:`subsec_signal-propagation` argued that in a post-norm block, only the attention sublayer mixes across tokens, while the FFN acts position-wise. Modify `MiniBlock` so that the attention sublayer is skipped ($\mathbf{H} = \textrm{Norm}(\mathbf{X})$) and only the post-norm FFN runs, and re-run `signal_stats(pre_norm=False)`. Does token spread still collapse to zero across 32 layers, or does it stabilize? Explain how this confirms each factor in :eqref:`eq_collapse-rate`.
:exlabel:`ex_transformer-block-ffn-width-rounding`
