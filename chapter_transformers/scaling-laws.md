# Scaling Laws
:label:`sec_scaling-laws`

When pretraining a Transformer language model (:numref:`sec_gpt`), two macroscopic quantities determine the compute bill: the number of **non-embedding parameters $N$** in the $L$ Transformer blocks, and the number of **training tokens $D$** streamed through the optimizer. Because each token requires a predictable number of floating-point operations (FLOPs) per parameter, a fixed training compute budget $C$ (in FLOPs) forces a direct trade-off between model size $N$ and data budget $D$: should we train a larger model for fewer steps, or a smaller model on more tokens?

Empirical **neural scaling laws** :cite:`Kaplan.McCandlish.Henighan.ea.2020,Hoffmann.Borgeaud.Mensch.ea.2022` revealed that validation cross-entropy loss follows a smooth power-law surface $\mathcal{L}(N, D)$ across orders of magnitude in $N$ and $D$, and that minimizing loss at fixed pretraining compute $C \approx 6 N D$ requires scaling model parameters $N^*(C)$ and training tokens $D^*(C)$ in **equal proportion** ($N^* \propto C^{1/2}, D^* \propto C^{1/2}$, with roughly $D^* / N^* \approx 20$ tokens per parameter, :numref:`fig_mdl-transformers-chinchilla`). This section derives the parameter census $N \approx 12 L d^2$ and the FLOP rule $C \approx 6 N D$, verifies both against exact operator counters, runs a 15-model scaling grid across 5 sizes and 3 token budgets on WikiText-103, and explains why accounting for **inference cost** shifts modern open models toward training smaller models far past the Chinchilla ratio.

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

Adding the forward and backward passes gives the fundamental training compute approximation :cite:`Kaplan.McCandlish.Henighan.ea.2020`:

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

To observe scaling behavior firsthand, we train a grid of **15 models** on WikiText-103: five architectures spanning $N \in [0.3\textrm{M}, 14\textrm{M}]$ non-embedding parameters (`(d, L, H)` from `(96, 3, 4)` up to `(384, 8, 8)`) crossed with three step budgets $s \in \{250, 500, 1000\}$ steps (corresponding to $D \in \{2.05\textrm{M}, 4.10\textrm{M}, 8.19\textrm{M}\}$ training bytes):

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

1. At the largest token budget ($1{,}000$ steps), validation loss decreases almost linearly with $\log N$ across the small and medium models before bending flat at the largest model ($N = 14.2\textrm{M}$), where $1{,}000$ steps is too few updates for $14\textrm{M}$ parameters.
2. When all 15 runs are plotted against total training FLOPs $C = 6 N D$, their lower envelope traces the **Pareto-optimal compute frontier**: at low compute, small models win; at higher compute, larger models overtake them once given enough steps.

## Compute-Optimal Allocation (Chinchilla Scaling)

### The Chinchilla Parametric Law

:citet:`Hoffmann.Borgeaud.Mensch.ea.2022` (and the refit of :citet:`Besiroglu.Erdil. Barnett.ea.2024`) model the validation loss surface $\mathcal{L}(N, D)$ as the sum of an irreducible text-entropy floor $L_\infty$ and two power-law penalties in $N$ and $D$:

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
a \;=\; \frac{\beta}{\alpha + \beta}, \qquad b \;=\; \frac{\alpha}{\alpha + \beta}, \qquad G \;=\; \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}.
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

2. **Total vs. non-embedding parameters in the scaling grid.** Each model in our 5-size WikiText-103 sweep used a byte vocabulary ($V = 256$), where $V d$ is less than $3\%$ of $12 L d^2$. Repeat `census` for the same five `(d, L, H)` backbones if they instead used a BPE vocabulary of $V = 128{,}256$ tokens (Llama 3). What fraction of total parameters would the embedding table represent in the smallest (`d=96, L=3`) versus largest (`d=384, L=8`) model, and why do :citet:`Kaplan.McCandlish.Henighan.ea.2020` fit power laws against $N_{\textrm{non-emb}} = 12 L d^2$ rather than $N_{\textrm{total}}$?
:exlabel:`ex_scaling-laws-embedding-count`

3. **Cosine decay horizon vs. fixed step horizon.** In our 15-run scaling study, each budget $s \in \{250, 500, 1000\}$ annealed its cosine schedule to $0.1 \times \textrm{lr}$ at step $s$ (:citet:`Hoffmann.Borgeaud.Mensch.ea.2022`'s crucial correction to :citet:`Kaplan.McCandlish.Henighan.ea.2020`). Re-run the three budgets for the medium model `(192, 4, 6)` using a single fixed cosine horizon `num_steps=1000` inside `LMOptimizer` and merely reading off the validation loss at steps 250, 500, and 1000. By how much does failing to decay the learning rate at $s = 250$ and $s = 500$ overestimate the loss of short runs?
:exlabel:`ex_scaling-laws-frozen-learning-rate`

4. **Fitting the Chinchilla law to our 15-point grid.** Using `scipy.optimize.minimize` (or grid search + nonlinear least squares), fit the parameters $(L_\infty, A, B, \alpha, \beta)$ of :eqref:`eq_chinchilla_law` to the 15 `(N, D, val_loss)` points measured in our WikiText-103 sweep. What scaling exponents $a = \beta/(\alpha+\beta)$ and $b = \alpha/(\alpha+\beta)$ does the byte-level 15-point grid produce?
:exlabel:`ex_scaling-laws-fit-grid`

5. **Lifetime compute optimization with inference demand $D_{\textrm{inf}}$.** A model of size $N$ trained on $D$ tokens and deployed to serve $D_{\textrm{inf}}$ inference tokens consumes total lifetime FLOPs $C_{\textrm{tot}}(N, D) = 6 N D + 2 N D_{\textrm{inf}}$.
   1. Using Lagrange multipliers on $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta} = \mathcal{L}_{\textrm{target}}$, show that at the lifetime-compute minimum $(N^*, D^*)$, the marginal ratio satisfies

      $$\frac{\beta B (D^*)^{-\beta}}{\alpha A (N^*)^{-\alpha}} \;=\; \frac{6 N^* D^*}{6 N^* D^* + 2 N^* D_{\textrm{inf}}} \;=\; \frac{1}{1 + \frac{D_{\textrm{inf}}}{3 D^*}}.$$

   2. Verify that when $D_{\textrm{inf}} = 0$ this recovers the Chinchilla balance $\alpha A (N^*)^{-\alpha} = \beta B (D^*)^{-\beta}$, and solve numerically for $(N^*, D^*)$ at $\mathcal{L}_{\textrm{target}} = \mathcal{L}(70\textrm{B}, 1.4\textrm{T})$ when $D_{\textrm{inf}} \in \{10\textrm{T}, 100\textrm{T}, 500\textrm{T}\}$ tokens.
:exlabel:`ex_scaling-laws-training-for-inference`

6. **Activation checkpointing and the $8 N D$ rule.** Full block-level activation checkpointing saves only the input $\mathbf{X}^{(\ell-1)}$ to each block during the forward pass ($2 N D$ FLOPs) and recomputes each block's forward pass once during backpropagation before computing the backward matrix products ($4 N D$ FLOPs). Show that activation checkpointing changes the pretraining compute formula from $C \approx 6 N D$ to $C \approx 8 N D$, and find how replacing $6$ with $8$ in :eqref:`eq_chinchilla_optimum` scales $N^*(C)$ and $D^*(C)$ at fixed FLOP budget $C$.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.7]{.kicker}

Scaling Laws<br>
**the cost of a run · a miniature scaling study · the compute-optimal split · training past the optimum**
:::
:::

::: {.slide title="A 3.8×10²⁵-FLOP Run Must Be Planned Before It Starts"}
[The problem]{.kicker}

| run | training FLOPs |
|:--|--:|
| our 4.8M-parameter GPT, 8.2M tokens | $2 \times 10^{14}$ |
| GPT-2 small, 124M parameters, ~10B tokens (estimate) | $7 \times 10^{18}$ |
| Llama 3, 405B parameters, 15.6T tokens | $3.8 \times 10^{25}$ |

::: {.d2l-note .warn}
The largest run is done once. How large a model, and how many tokens,
must be decided before it starts.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- estimate the training cost of a transformer as $6ND$ and say what it
  leaves out;
- read a grid of small runs along lines of equal compute;
- derive the compute-optimal split of a budget between parameters and
  tokens;
- explain why smaller deployed models train far past 20 tokens per
  parameter.

**Running example:** GPT-2 small (124M parameters), and five small GPTs
on the bytes of WikiText-103.
:::

::: {.slide title="Cost Follows from the Shape, Loss from Size and Data"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN, residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| The KV Cache | reuse past keys and values |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| **Scaling Laws** | **loss against parameters, data, compute** |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | released models as settings |
:::

::: {.col .narrow}
- **Uses:** `d2l.GPT` and its recipe, learning rate
  $\propto 1/\textrm{width}$.
- **Skips:** data curation; repetition beyond four passes (Language Models
  part).
:::
:::
:::

::: {.slide title="GPT-2's 124M Parameters: an 85M Transformer, 39M of Tables"}
Blocks hold $12Ld^2$ ($4d^2$ attention, $8d^2$ FFN); the embedding $Vd$:

@!scaling-laws-a-parameter-census

- The embedding dominates when $V \gtrsim 12Ld$: small models, large
  vocabularies.
- Kaplan et al. (2020), and our grid, count **non-embedding** $N$;
  Chinchilla counts all parameters.
:::

::: {.slide title="Training Costs Six FLOPs per Parameter and Token"}
$$C \approx \underbrace{2ND}_{\textrm{forward}} + \underbrace{4ND}_{\textrm{backward}} = 6ND$$

- Forward: each weight meets each token once, one multiply and one add.
- Backward: given $\boldsymbol{\delta}$, the gradient with respect to the
  layer's output, two products of the same size:
  $\mathbf{W}^\top\boldsymbol{\delta}$ (for the layer below) and
  $\boldsymbol{\delta}\mathbf{x}^\top$ (for the optimizer).
- Generating a token needs only the forward pass: about $2N$.
:::

::: {.slide title="Attention Adds a Fraction n/(6d) of the Counted Work"}
Per token and block, forward:

| work | FLOPs |
|:--|--:|
| products with the $12d^2$ weights | $24d^2$ |
| scores $\mathbf{q}^\top\mathbf{k}$ over $n$ keys, average of $n$ values | $4nd$ |

- Our models: $n = 128$, $d = 256$ → **8%**.
- Width 4,096 at 24,576 tokens → 100%.
- Norms, softmax, activations: a few FLOPs per element, much less.
:::

::: {.slide title="The Framework's Count Confirms 6ND Plus the Attention Term"}
Prediction: $1 + 4nLd/(2N) = 1.082$ times $2ND$
($n = 128$, $d = 256$, $L = 6$, $N = 4.78$M, output layer included).

@!scaling-laws-checking-the-arithmetic

- A counter of matrix products alone should land on **1.082**.
- A counter that also sees elementwise work should land a few percent
  above it.
- Backward / forward $\approx 2$, the 4-to-2 of the derivation.
:::

::: {.slide title="Check: What Did GPT-2 Small Cost?"}
[Check]{.kicker}

GPT-2 small: 124M parameters, roughly 10B training tokens (an estimate;
the paper does not report it), context 1,024, width 768. Estimate the
training FLOPs and the attention correction.

. . .

- $6ND = 6 \cdot 1.24 \times 10^{8} \cdot 10^{10} \approx 7 \times 10^{18}$.
- $n/(6d) = 1024 / 4608 \approx 22\%$: the correction is no longer small.
:::

::: {.slide title="A Grid of Five Sizes and Three Data Budgets"}
Corpus: WikiText-103 bytes, **270M** for training, read at random offsets.

| axis | values |
|:--|:--|
| width, blocks | 96/3, 128/3, 192/4, 256/6, 384/8 |
| $N$ (non-embedding) | 0.33M → 14.2M, a factor of 43 |
| $D$ (tokens) | 4.1M, 8.2M, 16.4M: at most 6% of the text, almost none repeated |

All 15 runs: AdamW, decay 0.1 on matrices, warmup, cosine, clipping;
no dropout.
:::

::: {.slide title="Every Run Gets Its Own Schedule and Its Own Learning Rate"}
- **Schedule:** the cosine decays to zero at the end of *each* run.
  A 500-step loss read off a 2,000-step run is taken at a high learning
  rate and overstates the loss (Hoffmann et al., 2022).
- **Learning rate:** the best Adam rate falls like $1/d$, so the peak
  rate is $2 \times 10^{-3} \cdot 256/d$.
- **Warmup:** a tenth of the steps, at least 100. With 50, the smallest
  model's 500-step run (peak $5.3 \times 10^{-3}$) can stall far above
  its neighbors; with 100 it ends in line.

::: {.d2l-note}
A study is only as good as the tuning of its worst point.
:::
:::

::: {.slide title="Loss Falls with Size and Data; Each Tripling of the Model Gains Less"}
@!scaling-laws-five-sizes-one-diet-1

- Down a column, $N$ grows: 0.59M → 1.77M gains 0.05–0.09 nats,
  4.72M → 14.2M only 0.02–0.04.
- Along a line, $D$ grows: 4× the tokens gains 0.20–0.27 nats.
- Run-to-run spread of one point: about 0.01 nats.
:::

::: {.slide title="Check: Four Times the Data or 43 Times the Parameters?"}
[Check]{.kicker}

Start from the smallest model (0.33M) on the smallest budget (4.1M
tokens). Which lowers the loss more: 4× the tokens, or the 14.2M model on
the same tokens? Which costs more compute?

. . .

- 4× the data: **0.25 nats or more**, at 4× the compute.
- 43× the parameters: 0.21–0.23 nats, at 43× the compute.
- At this scale, data lowers the loss more, at a tenth of the compute.
:::

::: {.slide title="At Equal Compute, the Smaller Model Trained Longer Wins"}
@!scaling-laws-equal-compute

- Gaps of 0.05 and 0.13 nats, 5–13× the run-to-run spread; the winner
  used half the FLOPs of the 14.2M model.
- Wherever two curves overlap in compute, the smaller model lies lower.
:::

::: {.slide title="The Chinchilla Law Separates Three Limits"}
$$\mathcal{L}(N, D) = \mathcal{L}_\infty + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

- $\mathcal{L}_\infty$ ($E$ in the paper): the entropy of the text, which
  no budget removes.
- $A/N^{\alpha}$: the excess loss of finite parameters.
- $B/D^{\beta}$: the excess loss of finite data.
- The sum assumes that size and data do not interact: the bold
  assumption, which our grid already strains.
- A fixed budget trades the second term against the third.

[Fitted to 400+ runs, 70M–16B parameters, 5–500B tokens (Hoffmann et al., 2022); $N$ counts all parameters.]{.small}
:::

::: {.slide title="At the Optimum, Both Shrinkable Terms Fall at Equal Rates"}
Under $C \approx 6ND$, substitute $D = C/(6N)$ and set $\partial \mathcal{L}/\partial N = 0$:

$$\alpha \frac{A}{N^{\alpha}} = \beta \frac{B}{D^{\beta}}
\quad\Rightarrow\quad
N^* \propto C^{\beta/(\alpha+\beta)}, \;\; D^* \propto C^{\alpha/(\alpha+\beta)}$$

- $\alpha \approx \beta$: both grow like $C^{1/2}$.
- $D^*/N^*$ hardly depends on the budget.
:::

::: {.slide title="The Optimum Is Where a Budget Line Touches a Loss Contour"}
::: {.cols .vc}
::: {.col .fig}
@fig:mdl-transformers-chinchilla
:::

::: {.col .narrow}
- Dashed: equal compute $C = 6ND$, slope $-1$.
- Orange: the compute-optimal path, slope $\approx 1$.
- Llama 3 8B: a similar budget at a tenth of the optimal size.
:::
:::
:::

::: {.slide title="Only the Refit Reproduces Chinchilla's 20 Tokens per Parameter"}
@!scaling-laws-the-optimal-split

- Hoffmann et al.: $\alpha = 0.34$, $\beta = 0.28$; refit (Besiroglu et
  al., 2024): $\alpha = 0.348$, $\beta = 0.366$.
- Published constants: **93** tokens per parameter; refit: **18**;
  Chinchilla trained 70B on 1.4T, **20**.

::: {.d2l-note}
The constants come from fits, and the fits disagree.
:::
:::

::: {.slide title="Counting and Tuning Small Runs Moved the Exponent from 0.73 to 0.5"}
| fit | $N^* \propto$ | 10× the budget gives |
|:--|:--|:--|
| Kaplan et al. (2020) | $C^{0.73}$ | 5.4× the model, 1.9× the tokens |
| refit of Chinchilla | $C^{0.51}$ | a little over 3× each |

- Pearce and Song (2024): non-embedding counts at small sizes, where the
  embedding is a large share.
- Porian et al. (2024): output layer left out of the compute, warmup too
  long for small models, optimizer not tuned per size.
- The earlier answer came from how small runs were counted and set up,
  not from a different law.
:::

::: {.slide title="Check: A Budget One Hundred Times Larger"}
[Check]{.kicker}

Under the refit, the budget grows from $10^{23}$ to $10^{25}$ FLOPs. How
much larger is the compute-optimal model, and how many more tokens does it
read?

. . .

- $N^*$ and $D^*$ both grow about like $C^{1/2}$: **about 10× each**
  (exponents 0.51 and 0.49).
- The ratio stays near 20 tokens per parameter.
:::

::: {.slide title="Smaller Models Train Far Past the Optimum to Make Generation Cheap"}
@!scaling-laws-training-past-the-optimum

- Generating a token costs $2N$, for the model's whole life.
- Llama 3 405B: 0.65 of the optimal size costs 0.001 nats; the loss is
  flat near the optimum.
- Llama 3 8B: a tenth of the optimal size, +0.05 nats: reducible loss
  0.20 against 0.15 (refit, nats per token).
:::

::: {.slide title="Check: The Cost of a Generated Token"}
[Check]{.kicker}

Under the refit, the compute-optimal model with Llama 3 8B's predicted
loss has about 34B parameters and reads 0.64T tokens. Compare the training
costs and the cost of a generated token.

. . .

- Training: $6 \cdot 8\textrm{B} \cdot 15\textrm{T}$ against
  $6 \cdot 34\textrm{B} \cdot 0.64\textrm{T}$, **about 5.5×** more for the 8B
  model.
- Generation: $2N$ per token, **about 4×** cheaper with the 8B model.
- The savings exceed the extra training after about 11T generated tokens,
  fewer than the 15T the model read in training.
:::

::: {.slide title="A Scaling Law Predicts the Loss of Dense Models, Nothing More"}
- It predicts one number: the loss on text like the training data.
- Whether a model answers a question, writes a program or follows an
  instruction is measured separately; small differences in loss need
  not show up there in proportion.
- It was fitted to dense models. A mixture of experts trains on the
  parameters each token uses, while its capacity follows all of them.
:::

::: {.slide title="6ND Prices a Run, and About 20 Tokens per Parameter Spends It Best"}
[Wrap-up]{.kicker}

- Training costs $\approx 6ND$; attention adds $n/(6d)$, 8% for our
  models, confirmed by the framework counts.
- At equal compute, the smaller model trained on more tokens reached the
  lower loss, by 0.05 to 0.13 nats.
- Compute-optimal: $N^*, D^* \propto C^{1/2}$, about 20 tokens per
  parameter (refit).
- Deployed models train up to 100× longer per parameter to make
  generation cheaper.

**Next:** a mixture of experts, whose training cost follows only the
parameters each token uses.
:::
