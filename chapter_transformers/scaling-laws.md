# Scaling Laws
:label:`sec_scaling-laws`

When pretraining a Transformer language model (:numref:`sec_gpt`), two macroscopic quantities determine the compute bill: the number of **non-embedding parameters $N$** in the $L$ Transformer blocks, and the number of **training tokens $D$** streamed through the optimizer. Because each token requires a predictable number of floating-point operations (FLOPs) per parameter, a fixed training compute budget $C$ (in FLOPs) forces a direct trade-off between model size $N$ and data budget $D$: should we train a larger model for fewer steps, or a smaller model on more tokens?

Empirical **neural scaling laws** :cite:`kaplan2020scaling,hoffmann2022training` revealed that validation cross-entropy loss follows a smooth power-law surface $\mathcal{L}(N, D)$ across orders of magnitude in $N$ and $D$, and that minimizing loss at fixed pretraining compute $C \approx 6 N D$ requires scaling model parameters $N^*(C)$ and training tokens $D^*(C)$ in **equal proportion** ($N^* \propto C^{1/2}, D^* \propto C^{1/2}$, with roughly $D^* / N^* \approx 20$ tokens per parameter, :numref:`fig_mdl-transformers-chinchilla`). This section derives the parameter census $N \approx 12 L d^2$ and the FLOP rule $C \approx 6 N D$, verifies both against exact operator counters, runs a 15-model scaling grid across 5 sizes and 3 token budgets on WikiText-103, and explains why accounting for **inference cost** shifts modern open models toward training smaller models far past the Chinchilla ratio.

```{.python .input #scaling-laws-scaling-laws-and-the-modern-recipe}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #scaling-laws-scaling-laws-and-the-modern-recipe}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #scaling-laws-1}
%%tab pytorch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode
```

## Counting Parameters ($N \approx 12 L d^2$) and FLOPs ($C \approx 6 N D$)

### Blocks vs. Embeddings

In a dense $L$-block `GPT` of hidden width $d$ with standard multi-head attention ($H_{\textrm{kv}} = H$) and `SwiGLU` width $d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$ (:numref:`sec_transformer-block`):

- Each attention sublayer holds four $d \times d$ projection matrices ($\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$): **$4 d^2$** parameters.
- Each `SwiGLU` FFN holds three matrices ($\mathbf{W}_g, \mathbf{W}_1 \in \mathbb{R}^{d \times \frac{8}{3}d}$ and $\mathbf{W}_2 \in \mathbb{R}^{\frac{8}{3}d \times d}$): **$8 d^2$** parameters.
- Each block therefore holds **$12 d^2$** matrix parameters (plus $2d$ `RMSNorm` gains), and $L$ blocks hold

  $$N \;=\; 12 L d^2$$

  non-embedding matrix parameters, while the tied token embedding table adds $V d$ parameters:

```{.python .input #scaling-laws-counting-parameters-1}
%%tab pytorch
gpt = d2l.GPT
```

```{.python .input #scaling-laws-counting-parameters-1}
%%tab jax
def gpt(*args, **kwargs):  # Fresh random weights for every model
    return d2l.GPT(*args, **kwargs, rngs=nnx.Rngs(d2l.get_key()))
```

```{.python .input #scaling-laws-a-parameter-census}
%%tab pytorch, jax
def census(model):
    L, d = len(model.blks), model.num_hiddens
    emb = d2l.num_params(model.token_emb) + (
        d2l.num_params(model.pos_emb) if model.pos == 'learned' else 0)
    total = d2l.num_params(model)
    print(f'embedding {emb / 1e6:6.2f}M ({emb / total:5.1%}),  '
          f'blocks {(total - emb) / 1e6:6.2f}M '
          f'= {(total - emb) / (L * d ** 2):.2f} L d^2,  '
          f'total {total / 1e6:.2f}M')

census(gpt(256))
census(gpt(50257, num_hiddens=768, num_heads=12, num_blks=12,
           pos='learned', norm='layer', act='gelu', bias=True))
```

### Six FLOPs per Parameter and Token ($C \approx 6 N D$)

Why does one training token consume approximately $6 N$ FLOPs across the $N = 12 L d^2$ linear-projection parameters?

1. **Forward pass ($2 N$ FLOPs/token):** Multiplying a token vector $\mathbf{x} \in \mathbb{R}^{d_{\textrm{in}}}$ by a weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\textrm{in}} \times d_{\textrm{out}}}$ performs $d_{\textrm{in}} d_{\textrm{out}}$ multiply--adds $= 2 P$ FLOPs (where $P = d_{\textrm{in}} d_{\textrm{out}}$ is the matrix's parameter count). Summing over all $N$ matrix parameters over $D$ tokens gives $2 N D$ forward FLOPs.
2. **Backward pass ($4 N$ FLOPs/token):** Backpropagating through $\mathbf{Y} = \mathbf{X}\mathbf{W}$ requires *two* matrix multiplications of size $P$ per token: $\nabla_{\mathbf{X}}\mathcal{L} = (\nabla_{\mathbf{Y}}\mathcal{L})\mathbf{W}^\top$ ($2P$ FLOPs) and $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top(\nabla_{\mathbf{Y}}\mathcal{L})$ ($2P$ FLOPs), totaling $4 N D$ backward FLOPs.

Adding the forward and backward passes gives the fundamental training compute approximation :cite:`kaplan2020scaling`:

$$
C \;\approx\; 6 N D.
$$
:eqlabel:`eq_six_nd`

(Including the $Q K^\top$ and $A V$ attention score dot products over context window length $T$ adds $6 L T d D$ FLOPs, multiplying $6 N D$ by $1 + \frac{T}{2d}$, which is close to $1$ whenever $T \ll 2d$.)

### Checking the Arithmetic Against Hardware Counters
:label:`subsec_scaling-laws-checking`

We can verify :eqref:`eq_six_nd` directly against exact hardware FLOP counters (`FlopCounterMode` in PyTorch and HLO cost analysis in JAX) on a single training step of `gpt(256, act='gelu')`:

```{.python .input #scaling-laws-checking-the-arithmetic}
%%tab pytorch
model = gpt(256, act='gelu')
X = torch.randint(0, 256, (64, 128))
Y = torch.randint(0, 256, (64, 128))
N = sum(p.numel() for p in model.parameters() if p.ndim == 2)
with sdpa_kernel(SDPBackend.MATH):  # Attention as visible matrix products
    with FlopCounterMode(display=False) as fwd:
        loss = F.cross_entropy(model(X).flatten(0, 1), Y.flatten())
    with FlopCounterMode(display=False) as bwd:
        loss.backward()
f, b = fwd.get_total_flops(), bwd.get_total_flops()
print(f'forward {f:.3e} FLOPs, {f / (2 * N * X.numel()):.3f} x 2ND')
print(f'step    {f + b:.3e} FLOPs, {(f + b) / (6 * N * X.numel()):.3f} x 6ND')
print(f'backward / forward = {b / f:.2f}')
```

```{.python .input #scaling-laws-checking-the-arithmetic}
%%tab jax
model = gpt(256, act='gelu')
key_x, key_y = jax.random.split(d2l.get_key())
X = jax.random.randint(key_x, (64, 128), 0, 256)
Y = jax.random.randint(key_y, (64, 128), 0, 256)
graphdef, params, rest = nnx.split(model, nnx.Param, ...)
N = sum(p.size for p in jax.tree.leaves(params) if p.ndim == 2)
# Count on the CPU: on a GPU the count depends on the kernels XLA picks
params, rest, X, Y = jax.device_put((params, rest, X, Y), d2l.cpu())

def loss_fn(params, X, Y):
    logits = nnx.merge(graphdef, params, rest)(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

f = jax.jit(loss_fn).lower(params, X, Y).compile().cost_analysis()['flops']
s = jax.jit(jax.value_and_grad(loss_fn)).lower(
    params, X, Y).compile().cost_analysis()['flops']
print(f'forward {f:.3e} FLOPs, {f / (2 * N * X.size):.3f} x 2ND')
print(f'step    {s:.3e} FLOPs, {s / (6 * N * X.size):.3f} x 6ND')
print(f'backward / forward = {(s - f) / f:.2f}')
```

The measured backward pass is $2.00\times$ the forward pass, and $6 N D$ (with the tied output head and attention score term $6 L T d D$ accounted for) matches the exact operator count to within $0.1\%$.

## A Miniature Scaling Study on WikiText-103

### Five Model Sizes Across Three Data Budgets
:label:`subsec_scaling-laws-five-sizes`

To observe scaling behavior firsthand, we train a grid of **15 models** on WikiText-103: five architectures spanning $N \in [0.3\textrm{M}, 14\textrm{M}]$ non-embedding parameters (`(d, L, H)` from `(96, 3, 4)` up to `(384, 8, 8)`) crossed with three step budgets $s \in \{500, 1000, 2000\}$ steps (corresponding to $D \in \{4.10\textrm{M}, 8.19\textrm{M}, 16.38\textrm{M}\}$ training bytes):

```{.python .input #scaling-laws-a-corpus-bigger-than-a-novella}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
print(f'{len(data.train) / 1e6:.0f}M training bytes, '
      f'{len(data.val) / 1e6:.2f}M validation bytes, '
      f'vocabulary of {len(data.vocab)}')
```

```{.python .input #scaling-laws-five-sizes-one-diet-1}
%%tab pytorch, jax
sizes = ((96, 3, 4), (128, 3, 4), (192, 4, 6), (256, 6, 8), (384, 8, 8))
budgets = (500, 1000, 2000)  # Steps of 64 windows of 128 bytes
Ns, train, val = [], [], {num_steps: [] for num_steps in budgets}
for d, num_blks, num_heads in sizes:
    for num_steps in budgets:
        model = gpt(len(data.vocab), num_hiddens=d, num_heads=num_heads,
                    num_blks=num_blks)
        optimizer = d2l.LMOptimizer(model, lr=2e-3 * 256 / d,
                                    num_steps=num_steps,
                                    warmup=max(100, num_steps // 10))
        losses = d2l.train_steps(model, data, optimizer, num_steps)
        val[num_steps].append(
            math.log(d2l.lm_perplexity(model, data.val_dataloader())))
    # The size of the models just trained, and the training loss of the
    # longest of their runs
    Ns.append(d2l.num_params(model) - d2l.num_params(model.token_emb))
    train.append(sum(losses[-50:]) / 50)
    print(f'd = {d:3d}, {num_blks} blocks, N = {Ns[-1] / 1e6:5.2f}M: '
          'validation ' + ' / '.join(f'{val[s][-1]:.2f}' for s in budgets)
          + f', training {train[-1]:.2f}')
```

### Loss Against Model Size, Data Budget, and Compute

Plotting validation cross-entropy against non-embedding parameters $N$ on a log axis for each token budget $D$, and against total training compute $C = 6 N D$:

```{.python .input #scaling-laws-five-sizes-one-diet-2}
%%tab pytorch, jax
tokens = {s: s * 64 * 128 for s in budgets}  # Tokens read in s steps
d2l.plot(Ns, [val[s] for s in budgets], 'non-embedding parameters N',
         'validation loss (nats)',
         legend=[f'D = {tokens[s] / 1e6:.1f}M' for s in budgets],
         xscale='log', yscale='log', fmts=('C0o-', 'C1s-', 'C2^-'),
         legend_loc='outside')
```

```{.python .input #scaling-laws-equal-compute}
%%tab pytorch, jax
d2l.plot([[6 * N * tokens[s] for s in budgets] for N in Ns],
         [[val[s][i] for s in budgets] for i in range(len(Ns))],
         'training compute C = 6ND (FLOPs)', 'validation loss (nats)',
         legend=[f'N = {N / 1e6:.2f}M' for N in Ns], xscale='log',
         fmts=('C0o-', 'C1s-', 'C2^-', 'C3v-', 'C4D-'), legend_loc='outside')
for i, s in ((2, 2000), (3, 1000), (4, 500)):  # Larger N, fewer tokens
    print(f'N = {Ns[i] / 1e6:5.2f}M, D = {tokens[s] / 1e6:4.1f}M tokens: '
          f'C = {6 * Ns[i] * tokens[s]:.1e} FLOPs, '
          f'validation {val[s][i]:.2f}')
```

Two patterns stand out:

1. At the largest token budget ($2{,}000$ steps, $D = 16.4\textrm{M}$ bytes), validation loss decreases almost linearly with $\log N$ across the small and medium models before bending flat at the largest model ($N = 14.16\textrm{M}$), where $2{,}000$ steps ($D/N \approx 1.2$ bytes per parameter) is still starved of data relative to $14\textrm{M}$ parameters.
2. When all 15 runs are plotted against total training FLOPs $C = 6 N D$, their lower envelope traces the **Pareto-optimal compute frontier**: at low compute, small models win; at higher compute, larger models overtake them once given enough steps.

## Compute-Optimal Allocation (Chinchilla Scaling)

### The Chinchilla Parametric Law

:citet:`hoffmann2022training` (and the refit of :citet:`Besiroglu.Erdil.Barnett.ea.2024`) model the validation loss surface $\mathcal{L}(N, D)$ as the sum of an irreducible text-entropy floor $L_\infty$ and two power-law penalties in $N$ and $D$:

$$
\mathcal{L}(N, D) \;=\; L_\infty \;+\; \frac{A}{N^\alpha} \;+\; \frac{B}{D^\beta}.
$$
:eqlabel:`eq_chinchilla_law`

### The Compute-Optimal Split ($N^*(C)$ and $D^*(C)$)

Minimizing :eqref:`eq_chinchilla_law` with respect to $N$ and $D$ subject to $C = 6 N D$ (substituting $D = C / (6N)$ and setting $\partial \mathcal{L} / \partial N = 0$) yields power-law scaling for both the optimal model size $N^*(C)$ and optimal token budget $D^*(C)$:

$$
N^*(C) \;=\; G \left(\frac{C}{6}\right)^a, \qquad D^*(C) \;=\; G^{-1} \left(\frac{C}{6}\right)^b,
$$
:eqlabel:`eq_chinchilla_optimum`

where

$$
\begin{aligned}
a &\;=\; \frac{\beta}{\alpha + \beta}, \qquad b \;=\; \frac{\alpha}{\alpha + \beta}, \\
G &\;=\; \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}.
\end{aligned}
$$
:eqlabel:`eq_chinchilla_allocation`

![Loss contours of the Chinchilla law with the refit constants of Besiroglu et al. (2024), introduced below, over parameters and training tokens. Dashed: budgets of equal compute $C = 6ND$; dots: the compute-optimal model of each budget; orange: the compute-optimal path. Chinchilla's square sits on the optimum of its budget, covering its dot; Llama 3 8B, on a budget close to Chinchilla's, lies far from the path at a tenth of the optimal size.](../img/mdl-transformers-chinchilla.svg)
:label:`fig_mdl-transformers-chinchilla`

Because $\alpha \approx \beta \approx 0.34\text{--}0.37$, the exponents satisfy $a \approx b \approx 0.5$: when training compute $C$ increases by $100\times$, both model size $N^*$ and training tokens $D^*$ increase by $\sim 10\times$, keeping the compute-optimal ratio $D^* / N^* \approx 20$ tokens per parameter roughly constant:

```{.python .input #scaling-laws-the-optimal-split}
%%tab pytorch, jax
fits = {'Hoffmann et al.': (1.69, 406.4, 410.7, 0.34, 0.28),
        'Besiroglu et al.': (1.82, 482.0, 2085.4, 0.348, 0.366)}

def optimum(C, L_inf, A, B, alpha, beta):
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

for name, fit in fits.items():
    N, D = optimum(5.76e23, *fit)
    print(f'{name:17s} N* = {N / 1e9:3.0f}B, D* = {D / 1e12:.2f}T, '
          f'{D / N:3.0f} tokens per parameter')
```

### Training Past the Chinchilla Optimum for Inference

Why did Llama 3 8B (:numref:`fig_mdl-transformers-chinchilla`) train an 8B-parameter model on $15\textrm{T}$ tokens ($D / N \approx 1{,}875$ tokens per parameter, nearly $100\times$ the Chinchilla ratio) instead of training a Chinchilla-optimal $70\textrm{B}$ model on $1.4\textrm{T}$ tokens at the same pretraining compute $C \approx 7 \times 10^{23}$ FLOPs?

Because Chinchilla minimizes **pretraining FLOPs $6 N D$ alone**, ignoring the $2 N D_{\textrm{inf}}$ inference FLOPs and KV-cache memory spent serving the model after deployment :cite:`Sardana.Portes.Doubov.ea.2024`! Traveling along a constant-loss contour $\mathcal{L}(N, D) = \mathcal{L}_{\textrm{target}}$ shows that an 8B model over-trained on $15\textrm{T}$ tokens reaches a loss close to a Chinchilla-allocated $30\textrm{B}\text{--}40\textrm{B}$ model while making every subsequent inference token $4\text{--}5\times$ cheaper to serve:

```{.python .input #scaling-laws-training-past-the-optimum}
%%tab pytorch, jax
L_inf, A, B, alpha, beta = fits['Besiroglu et al.']
loss = lambda N, D: L_inf + A / N ** alpha + B / D ** beta
reported = {'Chinchilla': (70e9, 1.4e12), 'Llama 3 8B': (8e9, 15e12),
            'Llama 3 405B': (405e9, 15.6e12), 'OLMo 2 7B': (7e9, 4.05e12),
            'Gemma 3 27B': (27e9, 14e12)}
for name, (N, D) in reported.items():
    N_opt, D_opt = optimum(6 * N * D, L_inf, A, B, alpha, beta)
    print(f'{name:12s} {D / N:5.0f} tokens per parameter, '
          f'{N / N_opt:4.2f} x optimal size, '
          f'loss +{loss(N, D) - loss(N_opt, D_opt):.3f}')
```

## Summary

* A dense $L$-layer Transformer of width $d$ with SwiGLU ($d_{\textrm{ff}} \approx \frac{8}{3}d$) contains **$N \approx 12 L d^2$** non-embedding matrix parameters ($4 L d^2$ in attention and $8 L d^2$ in the FFN) and consumes **$C \approx 6 N D$** training FLOPs over $D$ tokens ($2 N D$ forward and $4 N D$ backward, :eqref:`eq_six_nd`).
* Under the **Chinchilla scaling law** $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta}$ (:eqref:`eq_chinchilla_law`), minimizing validation loss at fixed pretraining compute $C = 6 N D$ scales parameters and tokens with exponents $a = \beta/(\alpha+\beta) \approx 0.5$ and $b = \alpha/(\alpha+\beta) \approx 0.5$ (:eqref:`eq_chinchilla_optimum`--:eqref:`eq_chinchilla_allocation`), yielding roughly $D^* / N^* \approx 20$ tokens per parameter.
* Accounting for **lifetime inference compute** ($2 N D_{\textrm{inf}}$) and serving latency shifts the total-compute optimum along the constant-loss contour toward smaller models $N$ over-trained on hundreds or thousands of tokens per parameter.

## Exercises

1. **Attention-score FLOP correction ($1 + T / (2d)$).** Equation :eqref:`eq_six_nd` omits the attention score/value products $\mathbf{Q}\mathbf{K}^\top$ and $\mathbf{A}\mathbf{V}$, and the output head $\mathbf{H}^{(L)}\mathbf{E}^\top$.
   1. Show that in an $L$-block model of width $d$ and window length $T$, the forward attention dot products perform $4 L T^2 d$ FLOPs per window ($4 L T d$ FLOPs per token), and forward + backward attention dot products perform $12 L T d$ FLOPs per token (or $6 L T d$ when causal symmetry is exploited by FlashAttention).
   2. Take the ratio of $6 L T d$ to $6 N = 72 L d^2$ to show that causal attention adds a relative FLOP fraction of $T / (12 d)$ (or $T / (6 d)$ without causal tile skipping). At what context length $T^*$ does attention match the linear-projection FLOPs of Llama 3 8B ($d = 4096$)?

2. **Total vs. non-embedding parameters in the scaling grid.** Each model in our 5-size WikiText-103 sweep used a byte vocabulary ($V = 256$), where $V d$ is less than $3\%$ of $12 L d^2$. Repeat `census` for the same five `(d, L, H)` backbones if they instead used a BPE vocabulary of $V = 128{,}256$ tokens (Llama 3). What fraction of total parameters would the embedding table represent in the smallest (`d=96, L=3`) versus largest (`d=384, L=8`) model, and why do :citet:`kaplan2020scaling` fit power laws against $N_{\textrm{non-emb}} = 12 L d^2$ rather than $N_{\textrm{total}}$?
:exlabel:`ex_scaling-laws-embedding-count`

3. **Cosine decay horizon vs. fixed step horizon.** In our 15-run scaling study, each budget $s \in \{500, 1000, 2000\}$ annealed its cosine schedule to $0.1 \times \textrm{lr}$ at step $s$ (:citet:`hoffmann2022training`'s crucial correction to :citet:`kaplan2020scaling`). Re-run the three budgets for the medium model `(192, 4, 6)` using a single fixed cosine horizon `num_steps=2000` inside `LMOptimizer` and merely reading off the validation loss at steps 500, 1000, and 2000. By how much does failing to decay the learning rate at $s = 500$ and $s = 1000$ overestimate the loss of short runs?
:exlabel:`ex_scaling-laws-frozen-learning-rate`

4. **Fitting the Chinchilla law to our 15-point grid.** Using `scipy.optimize.minimize` (or grid search + nonlinear least squares), fit the parameters $(L_\infty, A, B, \alpha, \beta)$ of :eqref:`eq_chinchilla_law` to the 15 `(N, D, val_loss)` points measured in our WikiText-103 sweep. What scaling exponents $a = \beta/(\alpha+\beta)$ and $b = \alpha/(\alpha+\beta)$ does the byte-level 15-point grid produce?
:exlabel:`ex_scaling-laws-fit-grid`

5. **Lifetime compute optimization with inference demand $D_{\textrm{inf}}$.** A model of size $N$ trained on $D$ tokens and deployed to serve $D_{\textrm{inf}}$ inference tokens consumes total lifetime FLOPs $C_{\textrm{tot}}(N, D) = 6 N D + 2 N D_{\textrm{inf}}$.
   1. Using Lagrange multipliers on $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta} = \mathcal{L}_{\textrm{target}}$, show that at the lifetime-compute minimum $(N^*, D^*)$, the marginal ratio satisfies

      $$\frac{\beta B (D^*)^{-\beta}}{\alpha A (N^*)^{-\alpha}} \;=\; \frac{6 N^* D^*}{6 N^* D^* + 2 N^* D_{\textrm{inf}}} \;=\; \frac{1}{1 + \frac{D_{\textrm{inf}}}{3 D^*}}.$$

   2. Verify that when $D_{\textrm{inf}} = 0$ this recovers the Chinchilla balance $\alpha A (N^*)^{-\alpha} = \beta B (D^*)^{-\beta}$, and solve numerically for $(N^*, D^*)$ at $\mathcal{L}_{\textrm{target}} = \mathcal{L}(70\textrm{B}, 1.4\textrm{T})$ when $D_{\textrm{inf}} \in \{10\textrm{T}, 100\textrm{T}, 500\textrm{T}\}$ tokens.
:exlabel:`ex_scaling-laws-training-for-inference`

6. **Activation checkpointing and the $8 N D$ rule.** Full block-level activation checkpointing saves only the input $\mathbf{X}^{(\ell-1)}$ to each block during the forward pass ($2 N D$ FLOPs) and recomputes each block's forward pass once during backpropagation before computing the backward matrix products ($4 N D$ FLOPs). Show that activation checkpointing changes the pretraining compute formula from $C \approx 6 N D$ to $C \approx 8 N D$, and find how replacing $6$ with $8$ in :eqref:`eq_chinchilla_optimum` scales $N^*(C)$ and $D^*(C)$ at fixed FLOP budget $C$.

[Discussions](https://d2l.discourse.group/t/36780)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.6]{.kicker}

Neural Scaling Laws and Compute-Optimal Training (Chinchilla)<br>**N ≈ 12Ld² & C ≈ 6ND · 15-model scaling grid · IsoFLOP contours · Chinchilla optimum · over-training for inference**
:::
:::

::: {.slide title="Why a 10²⁵-FLOP Run Must Be Planned Before It Starts"}
[Motivation]{.kicker}

Pretraining a frontier language model is a single, non-repeatable capital expenditure spanning thousands of GPUs for months:

| Pretraining Run | Non-Embedding / Total Params $N$ | Training Tokens $D$ | Total Compute $C \approx 6ND$ |
| :--- | :---: | :---: | :---: |
| **Our §11.2 Byte-Level GPT** (1 GPU, $< 1\text{ min}$) | $4.78\text{M}$ ($4.85\text{M}$) | $8.19\text{M}$ | $2.4 \times 10^{14}\text{ FLOPs}$ |
| **GPT-2 Small** (2019) | $85.0\text{M}$ ($124.4\text{M}$) | $\sim 10\text{B}$ | $7.5 \times 10^{18}\text{ FLOPs}$ |
| **Chinchilla 70B** (Hoffmann et al., 2022) | $70\text{B}$ | $1.4\text{T}$ | $5.9 \times 10^{23}\text{ FLOPs}$ |
| **Llama 3 405B** (2024) | $405\text{B}$ | $15.6\text{T}$ | $3.8 \times 10^{25}\text{ FLOPs}$ |

::: {.d2l-note .warn}
**The fundamental resource allocation problem:** Given a fixed training compute budget $C$ (in FLOPs), how should we split $C$ between model parameters $N$ and training tokens $D$ to achieve the lowest validation cross-entropy $\mathcal{L}(N, D)$?
:::
:::

::: {.slide title="Parameter Census: Blocks (12Ld²) vs. Embeddings (Vd)"}
[Parameter Accounting]{.kicker}

In an $L$-block dense Transformer of width $d$ with $H_{\text{kv}} = H$ and `SwiGLU` ($d_{\text{ff}} \approx \frac{8}{3}d$), each block holds $4d^2$ attention weights and $8d^2$ FFN weights:

$$N_{\mathrm{blocks}} = 12 L d^2, \qquad N_{\mathrm{total}} = 12 L d^2 + V d + \text{norms/pos}$$

@scaling-laws-a-parameter-census

- **Small vocabulary ($V = 256$ bytes):** The embedding table is only $1.3\%$ of our $4.85\text{M}$-parameter model ($N_{\text{blocks}} = 12.00\,L d^2$).
- **Large BPE vocabulary ($V = 50{,}257$ in GPT-2 Small):** The embedding and position tables hold $39.4\text{M}$ ($31.7\%$) of the $124.4\text{M}$ parameters, while the 12 blocks hold $85.0\text{M}$ ($12.02\,L d^2$).
:::

::: {.slide title="Why Training Costs Six FLOPs per Parameter and Token"}
[The C ≈ 6ND Law]{.kicker}

Consider a single linear layer $\mathbf{Y} = \mathbf{X}\mathbf{W}$ with $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ ($P = d_{\text{in}} d_{\text{out}}$ parameters) processing $D$ token rows:

$$C \approx \underbrace{2 N D}_{\text{forward pass}} + \underbrace{4 N D}_{\text{backward pass}} = 6 N D$$

- **Forward pass ($2 P$ FLOPs/token):** Each token row $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ performs $d_{\text{in}} d_{\text{out}}$ multiply–accumulate pairs ($1\text{ multiply} + 1\text{ add} = 2 P$ FLOPs).
- **Backward pass ($4 P$ FLOPs/token):** Given upstream gradient $\nabla_{\mathbf{Y}}\mathcal{L}$, backpropagation computes *two* matrix products of size $P$ per token:
  1. **Activation gradient:** $\nabla_{\mathbf{X}}\mathcal{L} = (\nabla_{\mathbf{Y}}\mathcal{L})\mathbf{W}^\top$ ($2 P$ FLOPs/token) to propagate down the stack.
  2. **Parameter gradient:** $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top(\nabla_{\mathbf{Y}}\mathcal{L})$ ($2 P$ FLOPs/token) for the optimizer step.
:::

::: {.slide title="Accounting for Attention Score Dot Products"}
[The Attention FLOP Term]{.kicker}

What does $C \approx 6 N D$ omit? Inside each block, attention also computes the token-to-token score matrix $\mathbf{Q}\mathbf{K}^\top$ and weighted value sum $\mathbf{A}\mathbf{V}$ over window length $T$:

| Operation per Block and Token (Forward Pass) | FLOPs per Token | Ratio to Linear Projections |
| :--- | :---: | :---: |
| **Linear projections ($\mathbf{W}_{q,k,v,o}$ and FFN, $12 d^2$ params)** | $24 d^2$ | $1.00$ (baseline) |
| **Attention scores ($\mathbf{q}_t^\top \mathbf{k}_s$) + value mix ($\sum_s \alpha_{t,s}\mathbf{v}_s$)** | $4 T d$ (unmasked) | $\frac{4 T d}{24 d^2} = \frac{T}{6 d}$ |

- **Short/medium context ($T \ll 6d$):** In our byte-level GPT ($T = 128, d = 256, N = 4.78\text{M}$ including the tied output head), attention adds $\frac{4 L T d}{2 N} = \mathbf{8.2\%}$ to the linear-projection FLOPs.
- **Long-context crossover ($T \ge 6d$):** At $d = 4{,}096$ (Llama 3 8B), quadratic attention FLOPs equal the linear-projection FLOPs once context length reaches $T = 6 d = 24{,}576$ tokens.
:::

::: {.slide title="Verifying C ≈ 6ND Against Exact Hardware Counters"}
[Empirical FLOP Audit]{.kicker}

Measuring a full forward + backward step of `gpt(256, act='gelu')` on batch $(64, 128)$ ($D_{\text{step}} = 8{,}192$ tokens) with PyTorch's `FlopCounterMode` and JAX's HLO `cost_analysis()` confirms the analytical derivation:

@!scaling-laws-checking-the-arithmetic

::: {.d2l-note .rule}
The exact matrix-multiply FLOP count lands at **$1.082 \times 6 N D$**—matching $1 + \frac{4 L T d}{2 N} = 1.082$ to three decimal places—and the backward-to-forward ratio is **$\mathbf{2.00}$** ($4 N D$ vs. $2 N D$).
:::
:::

::: {.slide title="A 15-Model Empirical Scaling Study on WikiText-103"}
[Scaling Grid Design]{.kicker}

To observe neural scaling laws firsthand, we train a $5 \times 3$ grid of **15 byte-level GPTs** spanning a $43\times$ range in non-embedding parameters $N$ and three data budgets $D$:

| Experimental Axis | Values in Our 15-Run Grid | Scope & Purpose |
| :--- | :--- | :--- |
| **5 Backbone Sizes `(d, L, H)`** | `(96, 3, 4)`, `(128, 3, 4)`, `(192, 4, 6)`, `(256, 6, 8)`, `(384, 8, 8)` | $N \in [0.33\text{M},\; 14.17\text{M}]$ non-embedding params ($43\times$ span) |
| **3 Step Budgets ($s$ steps)** | $500$, $1{,}000$, $2{,}000$ steps ($8{,}192$ bytes/step) | $D \in \{4.1\text{M},\; 8.2\text{M},\; 16.4\text{M}\}$ bytes ($\le 6\%$ of $270\text{M}$ corpus) |
| **Width-Scaled Peak LR** | $\eta_{\max}(d) = 2 \times 10^{-3} \cdot (256 / d)$ | Optimal AdamW learning rate scales inversely with width $d$ |
| **Matched Cosine Horizon** | Warmup $\max(100, s/10)$, cosine decays to $0$ at step $s$ | Every run is evaluated at the end of its own annealed schedule |
:::

::: {.slide title="Pitfall: Fixed Horizons Distort Scaling Exponents"}
[Experimental Methodology Pitfall]{.kicker}

Why did we train three separate runs ($s = 500, 1000, 2000$) for each model size instead of training one $2{,}000$-step run and reading validation loss at steps $500$, $1{,}000$, and $2{,}000$?

- **Hoffmann et al. (2022)'s key correction to Kaplan et al. (2020):** In a $2{,}000$-step cosine schedule, the learning rate at step $500$ is still near its peak ($\eta_{500} \approx 0.9\,\eta_{\max}$), preventing the weights from settling into a local minimum.
- **Systematic small-budget bias:** Reading intermediate checkpoints of an un-annealed schedule artificially inflates the loss of short runs, making extra tokens $D$ look less valuable than extra parameters $N$.
- **Width-dependent warmup:** Small widths ($d = 96$, peak LR $5.3 \times 10^{-3}$) require at least $100$ warmup steps so early Adam updates do not overshoot.
:::

::: {.slide title="Grid Results: Diminishing Returns in N Alone"}
[Empirical Scaling Surface]{.kicker}

::: {.cols .vc}
::: {.col}
- **Increasing $N$ at fixed tokens $D$:** Tripling model size from $0.59\text{M} \to 1.77\text{M}$ drops loss by $0.05\text{--}0.09\text{ nats}$, whereas tripling from $4.72\text{M} \to 14.17\text{M}$ gains only $0.02\text{--}0.04\text{ nats}$ because $500\text{--}2{,}000$ steps starve the $14.17\text{M}$ model of data.
- **Increasing $D$ at fixed size $N$:** Quadrupling tokens ($4.1\text{M} \to 16.4\text{M}$) lowers validation loss by **$0.20\text{--}0.27\text{ nats}$** across every model size.
:::

::: {.col .fig}
@!scaling-laws-five-sizes-one-diet-2
:::
:::
:::

::: {.slide title="IsoFLOP Trade-Off: Smaller Models Trained Longer Win"}
[The Pareto Compute Frontier]{.kicker}

::: {.cols .vc}
::: {.col}
- **Plotting all 15 runs against total FLOPs $C = 6 N D$:** Where curves for different model sizes overlap in compute $C$, the **smaller model trained on more tokens** consistently achieves lower validation loss!
- **Concrete IsoFLOP comparison:**
  - $N = 14.17\text{M}$ on $D = 4.1\text{M}$ ($C = 3.5 \times 10^{14}$): **$1.48\text{ nats}$**
  - $N = 4.72\text{M}$ on $D = 8.2\text{M}$ ($C = 2.3 \times 10^{14}$): **$1.35\text{ nats}$**
  - $N = 1.77\text{M}$ on $D = 16.4\text{M}$ ($C = 1.7 \times 10^{14}$): **$1.28\text{ nats}$** (lowest loss at *half* the FLOPs!).
:::

::: {.col .fig}
@!scaling-laws-equal-compute
:::
:::
:::

::: {.slide title="The Chinchilla Parametric Scaling Law"}
[Parametric Loss Surface]{.kicker}

Hoffmann et al. (2022) decompose validation cross-entropy $\mathcal{L}(N, D)$ into three additive terms fitted across $400+$ training runs ($70\text{M}\text{--}16\text{B}$ parameters, $5\text{B}\text{--}500\text{B}$ tokens):

$$\mathcal{L}(N, D) = L_\infty + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

- **Irreducible entropy floor ($L_\infty \approx 1.69\text{--}1.82\text{ nats/token}$):** The fundamental Shannon entropy of natural text that no finite model or dataset can eliminate.
- **Finite-capacity approximation penalty ($A / N^\alpha$):** Excess loss from approximating the true distribution with a finite $N$-parameter network ($\alpha \approx 0.35$).
- **Finite-data estimation penalty ($B / D^\beta$):** Excess loss from training on a finite sample of $D$ tokens ($\beta \approx 0.37$).
:::

::: {.slide title="Deriving the Compute-Optimal Split N*(C) and D*(C)"}
[Constrained Optimization]{.kicker}

To minimize $\mathcal{L}(N, D)$ subject to a fixed compute budget $C = 6 N D$, substitute $D = C / (6N)$ into $\mathcal{L}(N, C/(6N))$ and set the derivative with respect to $N$ to zero:

$$\frac{d}{dN}\left(L_\infty + A N^{-\alpha} + B (C/6)^{-\beta} N^\beta\right) = -\alpha A N^{-\alpha-1} + \beta B (C/6)^{-\beta} N^{\beta-1} = 0$$

- **Equating marginal returns ($\alpha A (N^*)^{-\alpha} = \beta B (D^*)^{-\beta}$):** At the compute optimum, the finite-model penalty and finite-data penalty shrink at proportional rates:
  $$N^*(C) = G\left(\frac{C}{6}\right)^a, \qquad D^*(C) = G^{-1}\left(\frac{C}{6}\right)^b$$
  where $a = \frac{\beta}{\alpha + \beta}$, $b = \frac{\alpha}{\alpha + \beta}$, and $G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}$.
- **Equal scaling ($a \approx b \approx 0.5$):** Because $\alpha \approx \beta$, scaling compute $C$ by $100\times$ scales **both $N^*$ and $D^*$ by $\sim 10\times$**!
:::

::: {.slide title="IsoFLOP Contours and the Compute-Optimal Path"}
[Geometric Intuition]{.kicker}

::: {.cols .vc}
::: {.col}
- **IsoFLOP budget lines (dashed, slope $-1$ on log–log axes):** Lines of constant training compute $C = 6 N D$.
- **Compute-optimal ridge (orange line, slope $\approx +1$):** Connects the tangency points $(N^*(C), D^*(C))$ where each budget line touches the lowest possible loss contour.
- **The 20-tokens-per-parameter rule:** Along the orange path, $D^* / N^* \approx 20$ tokens per parameter (`Chinchilla` 70B on $1.4\text{T}$ tokens).

@scaling-laws-the-optimal-split
:::

::: {.col .fig}
@fig:mdl-transformers-chinchilla
:::
:::
:::

::: {.slide title="Reconciling Kaplan (2020) vs. Chinchilla (2022)"}
[Why Exponents Shifted]{.kicker}

Why did Kaplan et al. (2020) conclude $N^* \propto C^{0.73}, D^* \propto C^{0.27}$ (recommending huge models trained on few tokens, like GPT-3 175B on 300B tokens), whereas Chinchilla found $N^* \propto C^{0.50}, D^* \propto C^{0.50}$?

| Methodological Factor | Kaplan et al. (2020) ($a = 0.73$) | Chinchilla (2022) & Refits ($a \approx 0.50$) |
| :--- | :--- | :--- |
| **1. Learning-rate schedule** | Fixed long horizon; intermediate losses read at high LR | Cosine schedule decayed to minimum LR at the end of *every* run |
| **2. Parameter counting ($N$)** | Counted **non-embedding** $N = 12 L d^2$ only | Included embedding/head $V d$ (or corrected small-model distortion) |
| **3. Small-model tuning** | Fixed warmup length across sizes (starving small runs) | Tuned warmup and peak LR $\propto 1/d$ per model size |

::: {.d2l-note}
Subsequent replication studies (Pearce & Song 2024; Porian et al. 2024) confirmed that accounting for embedding FLOPs and per-horizon LR decay moves the exponent directly from $0.73$ to $0.50$.
:::
:::

::: {.slide title="Why Deployed Models Train Far Past the Chinchilla Ratio"}
[Inference-Aware Scaling]{.kicker}

Look at where recent open-weights models sit relative to their Chinchilla pretraining optimum:

@!scaling-laws-training-past-the-optimum

- **Chinchilla optimizes pretraining cost ($6 N D$) alone:** It assigns **zero cost** to the inference FLOPs ($2 N$ per generated token) and VRAM spent serving the model after release!
- **Shallow loss contours near the optimum:** Training `Llama 3 8B` on $15\text{T}$ tokens ($1{,}875\text{ tokens/param}$, $0.11\times$ the Chinchilla-optimal size for that budget) pays only $+0.053\text{ nats}$ over a Chinchilla-optimal $73\text{B}$ model while making **every inference token $9\times$ cheaper** to generate.
:::

::: {.slide title="Lifetime Compute: Balancing Pretraining and Serving"}
[Total Cost of Ownership]{.kicker}

If a model of size $N$ is trained on $D$ tokens and then deployed to generate $D_{\text{inf}}$ tokens over its lifetime, total lifetime compute is:

$$C_{\mathrm{lifetime}}(N, D) = \underbrace{6 N D}_{\text{pretraining}} + \underbrace{2 N D_{\mathrm{inf}}}_{\text{inference serving}}$$

- **Break-even point for Llama 3 8B ($D = 15\text{T}$) vs. an equal-loss Chinchilla model ($N_{\text{Chin}} \approx 34\text{B}, D_{\text{Chin}} \approx 0.64\text{T}$):**
  - Extra pretraining FLOPs paid upfront by the 8B model: $6(8\text{B})(15\text{T}) - 6(34\text{B})(0.64\text{T}) \approx \mathbf{5.9 \times 10^{23}\text{ FLOPs}}$.
  - Inference FLOPs saved on every generated token: $2(34\text{B} - 8\text{B}) = \mathbf{5.2 \times 10^{10}\text{ FLOPs/token}}$.
  - **Payback threshold:** The upfront over-training pays for itself after serving just $D_{\text{inf}} \approx 11.3\text{T}$ generated tokens!
:::

::: {.slide title="Concept Check: FLOP Accounting and Budget Allocation"}
[Check Your Understanding]{.kicker}

Test your grasp of scaling laws with three calculations:

1. **GPT-2 Small compute & attention correction:** GPT-2 Small has $N = 124\text{M}$ total parameters ($d = 768$) and was trained on roughly $D = 10\text{B}$ tokens at context length $T = 1{,}024$. Estimate its $6ND$ training FLOPs and the relative attention score correction $T / (6d)$.
2. **100× compute budget expansion:** If a lab increases its pretraining compute budget $C$ by $100\times$ (from $10^{23}$ to $10^{25}\text{ FLOPs}$), by what factors should it scale model parameters $N^*$ and training tokens $D^*$ to stay on the Chinchilla compute-optimal path ($a \approx b \approx 0.5$)?
3. **Activation checkpointing ($8ND$):** Full block-level activation checkpointing recomputes the forward pass ($2ND$) once during backpropagation ($4ND$). How does this change the per-token training cost formula?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **$C \approx 6 \times (1.24 \times 10^8) \times 10^{10} \approx \mathbf{7.4 \times 10^{18}\text{ FLOPs}}$**, and the attention term adds $T / (6d) = 1024 / (6 \times 768) \approx \mathbf{22.2\%}$ on top of the linear projections.
2. Both $N^*$ and $D^*$ scale by **$100^{0.5} = \mathbf{10\times}$**, keeping the ratio $D^* / N^* \approx 20$ tokens per parameter invariant.
3. Total training compute increases by $33\%$ from $C \approx 6 N D$ to **$C \approx (2 + 2 + 4) N D = \mathbf{8 N D}$** in exchange for $O(1)$ per-layer activation memory.
:::
:::

::: {.slide title="Summary: Parameter, FLOP, and Chinchilla Scaling Laws"}
[Key Takeaways]{.kicker}

| Law / Quantity | Formula | Practical Meaning & Rule of Thumb |
| :--- | :--- | :--- |
| **Non-Embedding Parameters** | $N \approx 12 L d^2$ ($4 L d^2$ attn $+ 8 L d^2$ FFN) | Exact to $<1\%$ for dense models; add $V d$ for tied embeddings |
| **Training FLOPs** | $C \approx 6 N D \cdot \bigl(1 + \frac{T}{6d}\bigr)$ | $2ND$ forward $+ 4ND$ backward; attention dominates only when $T \gtrsim 6d$ |
| **Chinchilla Loss Surface** | $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta}$ | Additive decomposition into entropy floor, capacity error, and data error |
| **Compute-Optimal Split** | $N^*(C) \propto C^{\frac{\beta}{\alpha+\beta}} \approx C^{0.5},\; D^*(C) \propto C^{0.5}$ | Scale $N$ and $D$ equally with $\sqrt{C}$; maintain $D^* / N^* \approx 20\text{ tokens/param}$ |
| **Inference-Optimal Shift** | Minimize $6 N D + 2 N D_{\text{inf}}$ along $\mathcal{L}(N,D)=\mathcal{L}_0$ | Train smaller models to $200\text{--}2{,}000\text{ tokens/param}$ to slash serving cost |
:::
