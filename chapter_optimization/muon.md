# Muon
:label:`sec_muon`

Every optimizer examined so far—SGD, Momentum, and AdamW—treats the parameters of a neural network as an unstructured flat vector. Even AdamW maintains two scalar averages $(m_{t,ij}, v_{t,ij})$ independently for each entry $W_{ij}$ of a weight matrix $\mathbf{W}$: if the entries of $\mathbf{W}$ were arbitrarily permuted, AdamW would take the exact same steps.

Yet more than $80\%$ of the parameters in a Transformer reside in two-dimensional hidden projection matrices $\mathbf{W} \in \mathbb{R}^{m \times n}$ that act as linear operators mapping layer activations $\mathbf{x} \in \mathbb{R}^n$ to $\mathbf{y} = \mathbf{W}\mathbf{x} \in \mathbb{R}^m$. What is the direction of steepest descent for a weight matrix *viewed as a linear operator*?

As :citet:`Bernstein.Newhouse.2024` emphasize, steepest descent is defined relative to the norm chosen to bound the update step. Measuring the step in the Euclidean ($\ell_2$) norm yields the raw gradient direction of SGD; measuring it in the coordinate-wise $\ell_\infty$ box norm yields sign descent, the memoryless limit of Adam. If instead we bound the maximum change that an update $\Delta\mathbf{W}$ can induce in the layer output $\|\Delta\mathbf{W}\mathbf{x}\|_2 / \|\mathbf{x}\|_2$—which is the **spectral norm** $\|\Delta\mathbf{W}\|_2$—the direction of steepest descent replaces every nonzero singular value of the gradient matrix with $1$, yielding the **orthogonalized gradient** $\mathbf{U}\mathbf{V}^\top$.

Computing an exact Singular Value Decomposition (SVD) at every step is too slow for GPUs, but a five-step odd matrix polynomial (**Newton–Schulz iteration**) approximates $\mathbf{U}\mathbf{V}^\top$ using only matrix multiplications. Applied to Nesterov momentum buffers on hidden 2D weight matrices (while leaving embeddings, 1D parameters, and the output head to AdamW), this yields **Muon** (*Momentum Orthogonalized by Newton–Schulz*, :citet:`Jordan.Jin.Boza.ea.2024`). In this section, we derive steepest descent across the Euclidean, $\ell_\infty$, and spectral norm balls, analyze the tuned quintic Newton–Schulz iteration, implement the hybrid Muon+AdamW optimizer from scratch, and benchmark it against AdamW on `TinyLM`.

```{.python .input #muon}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #muon}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #muon-1}
%%tab pytorch
import random
```

## Norms and Steepest Descent

### Steepest Descent in a Norm Ball

Given the first-order Taylor expansion $f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) + \langle \mathbf{g}, \mathbf{d} \rangle$ with gradient $\mathbf{g} = \nabla f(\mathbf{x})$, the steepest descent step of radius $\eta > 0$ under a norm $\|\cdot\|$ solves

$$
\mathbf{d}^\star = \operatorname*{argmin}_{\|\mathbf{d}\| \leq \eta} \; \langle \mathbf{g}, \mathbf{d} \rangle.
$$
:eqlabel:`eq_muon-ball`

Different norm geometries yield fundamentally different update directions :cite:`Bernstein.Newhouse.2024` (:numref:`fig_opt_norm_balls` and :numref:`tab_muon_norms`):

1. **Euclidean ball ($\|\mathbf{d}\|_2 \le \eta$):** By Cauchy–Schwarz, the minimizer aligns with the negative gradient, $\mathbf{d}^\star = -\eta\,\mathbf{g}/\|\mathbf{g}\|_2$, decreasing the linearized objective by $\eta\|\mathbf{g}\|_2$ (the direction of SGD).
2. **Coordinate box ($\|\mathbf{d}\|_\infty = \max_i |d_i| \le \eta$):** Minimizing $\sum_i g_i d_i$ subject to $|d_i| \le \eta$ pushes every coordinate to its boundary, $\mathbf{d}^\star = -\eta\,\operatorname{sign}(\mathbf{g})$, decreasing the linearized objective by the dual $\ell_1$ norm $\eta\|\mathbf{g}\|_1$. This is sign descent, the zero-memory ($\beta_1 = \beta_2 = 0$) limit of Adam (:numref:`sec_adam`).

### Matrices and the Spectral Norm

For a hidden weight matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ mapping $\mathbf{x} \mapsto \mathbf{W}\mathbf{x}$, a perturbation $\Delta\mathbf{W}$ shifts the output by $\|\Delta\mathbf{W}\mathbf{x}\|_2 \le \|\Delta\mathbf{W}\|_2 \|\mathbf{x}\|_2$, where $\|\Delta\mathbf{W}\|_2 = \sigma_1(\Delta\mathbf{W})$ is the **spectral norm** (operator norm, :numref:`subsec_lin-alg-eigenvalues`). Let the gradient matrix $\mathbf{G} \in \mathbb{R}^{m \times n}$ have reduced Singular Value Decomposition (:numref:`sec_mdl-svd-low-rank`)

$$
\mathbf{G} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top = \sum_{i=1}^r \sigma_i\, \mathbf{u}_i \mathbf{v}_i^\top,
$$

where $\mathbf{U} \in \mathbb{R}^{m \times r}$ and $\mathbf{V} \in \mathbb{R}^{n \times r}$ have orthonormal columns and $\sigma_1 \ge \cdots \ge \sigma_r > 0$. Notice that the Frobenius norm $\|\mathbf{G}\|_F = \sqrt{\sum_i \sigma_i^2}$ (the Euclidean norm of the flattened matrix) can exceed the spectral norm $\sigma_1$ by up to $\sqrt{\min(m, n)}$ when singular values are comparable.

Solving :eqref:`eq_muon-ball` under the spectral norm constraint $\|\Delta\mathbf{W}\|_2 \le \eta$ yields the **orthogonalized gradient**:

$$
\Delta \mathbf{W}^\star = \operatorname*{argmin}_{\|\Delta \mathbf{W}\|_2 \leq \eta} \; \langle \mathbf{G}, \Delta \mathbf{W} \rangle = -\eta\, \mathbf{U} \mathbf{V}^\top,
$$
:eqlabel:`eq_muon-spectral-step`

which sets every nonzero singular value of $\mathbf{G}$ to $1$. The proof is immediate: for any $\mathbf{A}$ with $\|\mathbf{A}\|_2 \le 1$, cyclic trace invariance gives

$$
\langle \mathbf{G}, \mathbf{A} \rangle = \operatorname{tr}(\mathbf{G}^\top \mathbf{A}) = \operatorname{tr}(\boldsymbol{\Sigma}\, \mathbf{U}^\top \mathbf{A} \mathbf{V}) = \sum_{i=1}^r \sigma_i\, (\mathbf{U}^\top \mathbf{A} \mathbf{V})_{ii} \leq \sum_{i=1}^r \sigma_i,
$$

since $\|(\mathbf{U}^\top \mathbf{A} \mathbf{V})\|_2 \le \|\mathbf{A}\|_2 \le 1$ implies $|(\mathbf{U}^\top \mathbf{A} \mathbf{V})_{ii}| \le 1$ (Exercise :exref:`ex_muon-diagonal-entry`). Setting $\mathbf{A} = -\mathbf{U}\mathbf{V}^\top$ makes $\mathbf{U}^\top\mathbf{A}\mathbf{V} = -\mathbf{I}_r$, achieving the minimum $-\eta\sum_i \sigma_i = -\eta\|\mathbf{G}\|_*$ (the dual **nuclear norm**).

![Steepest descent directions under three norm balls: Euclidean $\ell_2$ (gradient direction), coordinate $\ell_\infty$ box (sign descent), and spectral norm $\|\cdot\|_2$ (orthogonalized gradient $\mathbf{U}\mathbf{V}^{\top}$, where every active singular value equals $1$).](../img/mdl-opt-norm-balls.svg)
:label:`fig_opt_norm_balls`

Whereas Adam equalizes update magnitudes across *individual matrix entries* (which can still collapse onto a rank-$1$ direction, as $\operatorname{sign}\!\begin{bmatrix}3 & 1\\ 1 & 1\end{bmatrix} = \begin{bmatrix}1 & 1\\ 1 & 1\end{bmatrix}$ has singular values $2$ and $0$), spectral steepest descent $-\eta\mathbf{U}\mathbf{V}^\top$ moves with equal step size $\eta$ along every active singular direction $(\mathbf{u}_i, \mathbf{v}_i)$ of the gradient.

:Steepest descent under three norm geometries.
:label:`tab_muon_norms`

| Norm ball on update | Steepest direction | Linearized decrease (dual norm) | Suited parameter group |
|:--|:--|:--|:--|
| Euclidean ($\|\mathbf{d}\|_2 \leq \eta$) | $-\eta\, \mathbf{g} / \|\mathbf{g}\|_2$ | $\eta\, \|\mathbf{g}\|_2$ | Unstructured vector (SGD) |
| Coordinate box ($\|\mathbf{d}\|_\infty \leq \eta$) | $-\eta\, \mathrm{sign}(\mathbf{g})$ | $\eta\, \|\mathbf{g}\|_1$ | Heterogeneous coordinates / embeddings (AdamW) |
| Spectral ($\|\Delta \mathbf{W}\|_2 \leq \eta$) | $-\eta\, \mathbf{U}\mathbf{V}^\top$ | $\eta \|\mathbf{G}\|_* = \eta \sum_i \sigma_i$ | Dense 2D hidden weight matrices (Muon) |

### Connection to Matrix Preconditioners (Shampoo) and Parameter Partitioning

The orthogonalized gradient $\mathbf{U}\mathbf{V}^\top$ also arises as the zero-memory limit of Kronecker-factored matrix preconditioning. **Shampoo** :cite:`Gupta.Koren.Singer.2018` accumulates left and right Gram matrices $\mathbf{L}_t = \sum_{s \le t} \mathbf{G}_s \mathbf{G}_s^\top$ and $\mathbf{R}_t = \sum_{s \le t} \mathbf{G}_s^\top \mathbf{G}_s$ and preconditions the gradient via $\mathbf{L}_t^{-1/4}\mathbf{G}_t\mathbf{R}_t^{-1/4}$. Evaluated on a single gradient $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ without historical accumulation, the singular values cancel identically:

$$
(\mathbf{G}\mathbf{G}^\top)^{-1/4}\, \mathbf{G}\, (\mathbf{G}^\top\mathbf{G})^{-1/4} = \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\mathbf{U}^\top \cdot \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \cdot \mathbf{V}\boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top = \mathbf{U}\mathbf{V}^\top.
$$
:eqlabel:`eq_muon-shampoo`

By applying orthogonalization to a standard first-moment momentum buffer rather than storing and inverting $\mathbf{L}_t$ and $\mathbf{R}_t$, Muon captures full-matrix preconditioning with only a single momentum buffer per matrix (:numref:`subsec_mdl-preconditioning-ladder`).

Because the spectral-norm geometry applies specifically to dense linear maps between activation spaces, Muon is applied **only to hidden 2D weight matrices** (attention `qkv`/`proj` and MLP weights). Embedding tables (which perform one-hot row lookups), 1D biases and LayerNorm scales, and the final vocabulary readout head are optimized with AdamW :cite:`Jordan.Jin.Boza.ea.2024`.

## Orthogonalization by Newton–Schulz
:label:`subsec_muon-newton-schulz`

Computing $\mathbf{U}\mathbf{V}^\top$ via an explicit SVD is prohibitively slow during training. Instead, observe that any **odd matrix polynomial** acts independently on the singular values of $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ while preserving its singular vectors:

$$
p(\mathbf{X}) = a\mathbf{X} + b\,(\mathbf{X}\mathbf{X}^\top)\mathbf{X} + c\,(\mathbf{X}\mathbf{X}^\top)^2\mathbf{X} = \mathbf{U}\, p(\boldsymbol{\Sigma})\, \mathbf{V}^\top,
$$
:eqlabel:`eq_muon-odd-poly`

since $\mathbf{X}\mathbf{X}^\top = \mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$. Normalizing $\mathbf{X} \leftarrow \mathbf{X} / \|\mathbf{X}\|_F$ first ensures that all singular values lie in $[0, 1]$. Iterating an odd scalar polynomial $p(x)$ that maps $(0, 1]$ toward $1$—a **Newton–Schulz iteration** (:numref:`subsec_mdl-orthogonalized-updates`)—therefore orthogonalizes $\mathbf{X}$ using only fast matrix multiplications.

Rather than the classical cubic $p(x) = \frac{3}{2}x - \frac{1}{2}x^3$ (whose small slope $p'(0) = 1.5$ lifts small singular values too slowly over $5$ steps), :citet:`Jordan.Jin.Boza.ea.2024` use a tuned quintic polynomial with a steep initial slope $p'(0) = 3.4445$:

$$
p(x) = 3.4445\,x - 4.7750\,x^3 + 2.0315\,x^5.
$$
:eqlabel:`eq_muon-quintic`

As shown in :numref:`fig_opt_ns_maps`, five compositions of :eqref:`eq_muon-quintic` lift every singular value in $[0.01, 1]$ into the narrow band $[0.68, 1.13]$ (and every singular value above $0.0015$ to at least $0.68$), trading exact asymptotic convergence to $1$ for rapid $5$-step inflation of small singular directions.

![Newton–Schulz polynomials acting on singular values. Left: single application of the classical cubic vs. the tuned quintic :eqref:`eq_muon-quintic`. Right: five composed iterations on $[10^{-3}, 1]$ (alongside the multi-polynomial Polar Express variant :cite:`Amsel.Persson.Musco.ea.2026`).](../img/mdl-opt-ns-maps.svg)
:label:`fig_opt_ns_maps`

```{.python .input #muon-orthogonalization-by-newton-schulz-1}
%%tab pytorch
def newton_schulz(M, num_iters=5, a=3.4445, b=-4.7750, c=2.0315, eps=1e-7):
    """Approximate the orthogonalization U V^T of M by num_iters steps of
    the odd polynomial a x + b x^3 + c x^5 (Jordan et al.'s tuned quintic)."""
    tall = M.shape[0] > M.shape[1]
    X = M.T if tall else M  # Keep the Gram factor X @ X.T small
    X = X / (X.norm() + eps)
    for _ in range(num_iters):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X
```

```{.python .input #muon-orthogonalization-by-newton-schulz-1}
%%tab jax
def newton_schulz(M, num_iters=5, a=3.4445, b=-4.7750, c=2.0315, eps=1e-7):
    """Approximate the orthogonalization U V^T of M by num_iters steps of
    the odd polynomial a x + b x^3 + c x^5 (Jordan et al.'s tuned quintic)."""
    tall = M.shape[0] > M.shape[1]
    X = M.T if tall else M  # Keep the Gram factor X @ X.T small
    X = X / (jnp.linalg.norm(X) + eps)
    for _ in range(num_iters):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X
```

For an $m \times n$ matrix ($m \le n$) and token batch size $B$, five Newton–Schulz iterations cost $5(4m^2n + 2m^3)$ FLOPs compared to $6mnB$ FLOPs for the forward and backward passes—an overhead of $\sim 5m/B$ ($<1\%$ at production batch sizes $B \sim 5 \times 10^5$, and $\sim 12\%$ on `TinyLM` at $B = 4{,}096$). Applying `newton_schulz` to a $96 \times 64$ Gaussian matrix illustrates how the iterations collapse the singular value spectrum: into $[0.68, 1.20]$ after $k = 3$ and $[0.68, 1.13]$ after $k = 5$:

```{.python .input #muon-orthogonalization-by-newton-schulz-2}
%%tab pytorch
G = torch.randn(96, 64)
sigmas = [torch.linalg.svdvals(newton_schulz(G, num_iters=k))
          for k in (0, 1, 3, 5)]
d2l.plot(torch.arange(1, 65), sigmas, 'index', 'singular value',
         legend=[f'k = {k}' for k in (0, 1, 3, 5)], yscale='log')
for k, sigma in zip((3, 5), sigmas[2:]):
    print(f'after {k} iterations: [{sigma.min():.2f}, {sigma.max():.2f}]')
```

```{.python .input #muon-orthogonalization-by-newton-schulz-2}
%%tab jax
G = jax.random.normal(d2l.get_key(), (96, 64))
sigmas = [jnp.linalg.svd(newton_schulz(G, num_iters=k), compute_uv=False)
          for k in (0, 1, 3, 5)]
d2l.plot(jnp.arange(1, 65), sigmas, 'index', 'singular value',
         legend=[f'k = {k}' for k in (0, 1, 3, 5)], yscale='log')
for k, sigma in zip((3, 5), sigmas[2:]):
    print(f'after {k} iterations: [{sigma.min():.2f}, {sigma.max():.2f}]')
```

On real network gradients, two details work in Muon's favor: first, Muon orthogonalizes the Nesterov momentum buffer $\mathbf{G}_t + \beta\mathbf{M}_t$ (averaged over $\sim 1/(1-\beta) = 20$ steps at $\beta = 0.95$) rather than a noisy single-batch gradient; second, any singular value below $0.15\%$ of the Frobenius norm (where $3.4445^5 \approx 485$ is insufficient to reach $0.68$) is amplified linearly rather than pushed all the way to $1$, naturally damping pure noise directions in the trailing spectrum.

## Muon from Scratch

### The Update and RMS Alignment

For an $m \times n$ hidden weight matrix $\mathbf{W}$ with gradient $\mathbf{G}_t$, Muon updates the momentum buffer $\mathbf{M}_t$ ($\beta = 0.95$) and applies five Newton–Schulz iterations $\mathrm{NS}_5$ to the Nesterov look-ahead gradient $\mathbf{G}_t + \beta\mathbf{M}_t$, combined with decoupled weight decay :cite:`Liu.Su.Yao.ea.2025`:

$$
\mathbf{M}_t = \beta\, \mathbf{M}_{t-1} + \mathbf{G}_t,
\qquad
\mathbf{W}_{t+1} = (1 - \eta\lambda)\, \mathbf{W}_t - \eta \cdot 0.2 \sqrt{\max(m, n)}\; \mathrm{NS}_5(\mathbf{G}_t + \beta\, \mathbf{M}_t).
$$
:eqlabel:`eq_muon-update`

Why multiply by $0.2\sqrt{\max(m, n)}$? An orthogonal $m \times n$ matrix $\mathbf{U}\mathbf{V}^\top$ has $\min(m, n)$ unit singular values, Frobenius norm $\sqrt{\min(m, n)}$, and entrywise RMS $\sqrt{\min(m, n)/(mn)} = 1/\sqrt{\max(m, n)}$. Multiplying by $0.2\sqrt{\max(m, n)}$ gives the update an entrywise RMS of $0.2\,\eta$ regardless of matrix dimensions—matching the empirical $\sim 0.2\,\eta$ entrywise RMS step of AdamW measured in :numref:`sec_adam` so that both optimizers can share the same learning rate $\eta$ and weight decay $\lambda$ :cite:`Liu.Su.Yao.ea.2025`.

```{.python .input #muon-the-update}
%%tab pytorch
class Muon(torch.optim.Optimizer):
    """Steepest descent under the spectral norm: orthogonalized momentum."""
    def __init__(self, params, lr, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:  # A parameter that received no gradient
                    continue
                state = self.state[p]
                if not state:  # First step: allocate the momentum buffer
                    state['buf'] = torch.zeros_like(p)
                buf = state['buf'].mul_(group['momentum']).add_(p.grad)
                G = p.grad + group['momentum'] * buf  # Nesterov momentum
                M = G.reshape(len(G), -1)             # Flattens conv kernels
                O = newton_schulz(M).reshape(p.shape)
                rms_scale = 0.2 * math.sqrt(max(M.shape))  # AdamW's RMS
                p.mul_(1 - group['lr'] * group['weight_decay'])  # Decoupled
                p.add_(O, alpha=-group['lr'] * rms_scale)
```

```{.python .input #muon-the-update}
%%tab jax
def scale_by_muon(momentum=0.95):
    """Orthogonalized Nesterov momentum, scaled to AdamW's update RMS."""
    def init(params):
        return jax.tree.map(jnp.zeros_like, params)
    def update(grads, bufs, params=None):
        bufs = jax.tree.map(lambda b, g: momentum * b + g, bufs, grads)
        def direction(g, b):
            G = g + momentum * b            # Nesterov momentum
            M = G.reshape(-1, G.shape[-1])  # Flattens conv kernels
            O = newton_schulz(M).reshape(G.shape)
            return 0.2 * math.sqrt(max(M.shape)) * O  # AdamW's RMS
        return jax.tree.map(direction, grads, bufs), bufs
    return optax.GradientTransformation(init, update)

def scratch_muon(learning_rate, momentum=0.95, weight_decay=0.0):
    return optax.chain(scale_by_muon(momentum),
                       optax.add_decayed_weights(weight_decay),  # Decoupled
                       optax.scale_by_learning_rate(learning_rate))
```

### Parameter Groups and Hybrid Optimizer
:label:`subsec_muon-parameter-groups`

On `TinyLM`, routing the $8$ internal block matrices (`qkv`, `proj`, `mlp1`, `mlp2` across $2$ blocks) to Muon and the remaining $22$ tensors (embeddings, LayerNorm vectors, biases, and output head) to AdamW assigns $393{,}216$ of the $470{,}784$ parameters ($84\%$) to Muon. Because Muon stores only one momentum buffer per parameter instead of AdamW's two moments, the hybrid optimizer reduces total optimizer-state memory by $42\%$.

```{.python .input #muon-dividing-the-census-1}
%%tab pytorch
data = d2l.WikiText103(batch_size=64, num_steps=64)

def split_lm(model):
    hidden, rest = [], []
    for name, p in model.named_parameters():
        is_hidden = p.ndim == 2 and 'emb' not in name and 'head' not in name
        (hidden if is_hidden else rest).append(p)
    return hidden, rest

hidden, rest = split_lm(d2l.TinyLM(len(data.vocab)))
print(f'Muon:  {len(hidden):>2} tensors, '
      f'{sum(p.numel() for p in hidden):>7} parameters')
print(f'AdamW: {len(rest):>2} tensors, '
      f'{sum(p.numel() for p in rest):>7} parameters')
```

```{.python .input #muon-dividing-the-census-1}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=64)

def muon_labels(params):
    """'muon' for the hidden matrices, 'adamw' for every other parameter."""
    def label(path, p):
        name = jax.tree_util.keystr(path)
        is_hidden = p.ndim == 2 and 'emb' not in name and 'head' not in name
        return 'muon' if is_hidden else 'adamw'
    return jax.tree_util.tree_map_with_path(label, params)

model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(d2l.get_key()))
params = nnx.state(model, nnx.Param)
labels = jax.tree.leaves(muon_labels(params))
for group, name in (('muon', 'Muon'), ('adamw', 'AdamW')):
    sizes = [p.size for p, label in zip(jax.tree.leaves(params), labels)
             if label == group]
    print(f'{name + ":":<6} {len(sizes):>2} tensors, {sum(sizes):>7} '
          'parameters')
```

```{.python .input #muon-dividing-the-census-2}
%%tab pytorch
class MultiOptimizer:
    """Apply independent optimizers to disjoint parameter groups."""
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

def muon_adamw(hidden, rest, lr, weight_decay):
    """Muon with decoupled weight decay on hidden, AdamW without it on rest."""
    return MultiOptimizer(Muon(hidden, lr, weight_decay=weight_decay),
                          torch.optim.AdamW(rest, lr, weight_decay=0.0))
```

```{.python .input #muon-dividing-the-census-2}
%%tab jax
def muon_adamw(lr, weight_decay):
    """Muon with decoupled weight decay on the hidden matrices, AdamW without
    it on the rest."""
    return optax.multi_transform(
        {'muon': scratch_muon(lr, weight_decay=weight_decay),
         'adamw': optax.adamw(lr, weight_decay=0.0)}, muon_labels)
```

### Comparison with AdamW
:label:`subsec_muon-comparison`

We benchmark the Muon+AdamW hybrid against standard AdamW across $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}, 3 \times 10^{-2}, 10^{-1}\}$ with $\lambda = 0.1$ on the hidden matrices for $2{,}000$ steps from shared initial weights, evaluating validation loss every $250$ steps:

```{.python .input #muon-the-race-on-the-language-model-1}
%%tab pytorch
def val_loss(model):
    """Mean cross-entropy per byte over the validation split."""
    return math.log(d2l.lm_perplexity(model, data.val_dataloader()))

def run_lm(make_optimizer, lrs, seed):
    """d2l.lr_sweep of TinyLM, validated every 250 steps."""
    return d2l.lr_sweep(lambda: d2l.TinyLM(len(data.vocab)), make_optimizer,
                        data, lrs, seed, evaluate=val_loss,
                        metric='validation loss', every=250)

lrs, wd = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], 0.1

def adamw(model, lr):  # Decay on the hidden matrices only, as for Muon
    hidden, rest = split_lm(model)
    return torch.optim.AdamW([{'params': hidden, 'weight_decay': wd},
                              {'params': rest, 'weight_decay': 0.0}], lr=lr)

seed = random.getrandbits(32)  # One draw, shared by every run below
adamw_lm, adamw_val = run_lm(adamw, lrs, seed)
```

```{.python .input #muon-the-race-on-the-language-model-1}
%%tab jax
def val_loss(model):
    """Mean cross-entropy per byte over the validation split."""
    return math.log(d2l.lm_perplexity(model, data.val_dataloader()))

def run_lm(make_tx, lrs, seed):
    """d2l.lr_sweep of TinyLM, validated every 250 steps."""
    return d2l.lr_sweep(lambda rngs: d2l.TinyLM(len(data.vocab), rngs=rngs),
                        make_tx, data, lrs, seed, evaluate=val_loss,
                        metric='validation loss', every=250)

def hidden_only(params):  # Decay on the hidden matrices only, as for Muon
    return jax.tree.map(lambda label: label == 'muon', muon_labels(params))

seed = d2l.get_seed()  # One draw, shared by every run below
lrs, wd = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], 0.1
adamw_lm, adamw_val = run_lm(
    lambda lr: optax.adamw(lr, weight_decay=wd, mask=hidden_only), lrs, seed)
```

```{.python .input #muon-the-race-on-the-language-model-2}
%%tab pytorch
muon_lm, muon_val = run_lm(
    lambda model, lr: muon_adamw(*split_lm(model), lr, wd), lrs, seed)
```

```{.python .input #muon-the-race-on-the-language-model-2}
%%tab jax
muon_lm, muon_val = run_lm(lambda lr: muon_adamw(lr, wd), lrs, seed)
```

```{.python .input #muon-the-race-on-the-language-model-3}
%%tab pytorch, jax
best_adamw = min(adamw_val, key=lambda lr: adamw_val[lr][-1])
best_muon = min(muon_val, key=lambda lr: muon_val[lr][-1])
d2l.plot([list(range(0, 2000, 25))] * 2 + [list(range(250, 2001, 250))] * 2,
         [d2l.smooth(adamw_lm[best_adamw]), d2l.smooth(muon_lm[best_muon]),
          adamw_val[best_adamw], muon_val[best_muon]], 'step', 'loss',
         legend=['AdamW, training', 'Muon + AdamW, training',
                 'AdamW, validation', 'Muon + AdamW, validation'],
         fmts=('C0-', 'C1-', 'C0o--', 'C1o--'), ylim=(1.4, 2.6),
         figsize=(5, 3.5))
print('validation loss  lr     step 500  step 2000  below 1.6  within 0.15')
for name, lr, vals in [('AdamW', best_adamw, adamw_val),
                       ('Muon+AdamW', best_muon, muon_val)]:
    below = d2l.steps_to_reach(vals[lr], 1.6, k=1)  # Checkpoint 1, 2, ...
    near = [r for r in lrs if vals[r][-1] < vals[lr][-1] + 0.15]
    print(f'{name:<16} {lr:<6g} {vals[lr][1]:<9.3f} {vals[lr][-1]:<10.3f} '
          f'{250 * below if below else "never":<10} '
          f'{near[0]:g} to {near[-1]:g}')
```

On `TinyLM`, Muon+AdamW establishes a $>0.1$-nat validation lead at step $500$, crosses the $1.6$-nat threshold at least one checkpoint ($250$ steps) earlier than AdamW, and expands the usable learning-rate window up to $\eta = 3 \times 10^{-2}$ (where AdamW finishes near or above $2$ nats because rank-$1$ early gradients under entrywise normalization can have spectral norm $\sqrt{\min(m, n)}$ times larger than an orthogonalized update). In large-scale pretraining benchmarks with full learning-rate decay ($0.1\text{B}\text{--}1.2\text{B}$ parameters :cite:`Wen.Hall.Ma.ea.2025` and trillion-parameter runs such as Kimi K2 :cite:`Liu.Su.Yao.ea.2025,Kimi.Team.2025`, often paired with attention-logit clipping), Muon achieves $1.1\times\text{--}1.4\times$ step reductions over heavily tuned AdamW while cutting optimizer state memory nearly in half.

## Summary

* Steepest descent depends on the norm ball bounding the update: the $\ell_2$ ball gives SGD, the $\ell_\infty$ box gives sign descent (smoothed by Adam), and the spectral norm ball $\|\Delta\mathbf{W}\|_2 \le \eta$ gives the orthogonalized gradient $-\eta\mathbf{U}\mathbf{V}^\top$.
* **Muon** approximates $\mathbf{U}\mathbf{V}^\top$ on the Nesterov momentum buffer of each hidden 2D weight matrix via $5$ iterations of a tuned quintic Newton–Schulz polynomial :eqref:`eq_muon-quintic`, scales the update by $0.2\sqrt{\max(m, n)}$ to match AdamW's entrywise RMS step size, and applies decoupled weight decay while leaving embeddings, 1D vectors, and the readout head to AdamW.
* On `TinyLM`, the Muon+AdamW hybrid accelerates early convergence, remains stable at $3\times$ higher learning rates than AdamW, and reduces optimizer-state memory by $42\%$.

## Exercises

1. **Diagonal-entry bound and uniqueness of the spectral step.** Let $\mathbf{U} \in \mathbb{R}^{m \times r}$ and $\mathbf{V} \in \mathbb{R}^{n \times r}$ have orthonormal columns. Show that $\|\mathbf{U}^\top\mathbf{A}\mathbf{V}\|_2 \le \|\mathbf{A}\|_2$ and $|(\mathbf{U}^\top\mathbf{A}\mathbf{V})_{ii}| \le \|\mathbf{A}\|_2$, and determine when the minimizer $\Delta\mathbf{W}^\star = -\eta\mathbf{U}\mathbf{V}^\top$ in :eqref:`eq_muon-spectral-step` is unique.
   :exlabel:`ex_muon-diagonal-entry`

1. [code] **Empirical update RMS and singular value spectrum.** Instrument `Muon` on `TinyLM` to track the entrywise RMS of $\Delta\mathbf{W} / \eta$ across $2{,}000$ steps and compare the singular value spectrum of a single minibatch gradient $\mathbf{G}_t$ against the momentum buffer $\mathbf{M}_t$ at step $0$ and step $500$.
   :exlabel:`ex_muon-step-size`

1. [code] **Newton–Schulz iteration count and polynomial degree.** Compare `num_iters` $\in \{1, 3, 5, 10\}$ with the tuned quintic :eqref:`eq_muon-quintic` against $5$ iterations of the classical cubic $p(x) = 1.5x - 0.5x^3$ on `TinyLM`, reporting validation loss and wall-clock step time.
   :exlabel:`ex_muon-iterations`

1. [code] **Why embeddings remain with AdamW.**
   :exlabel:`ex_muon-embeddings`
    1. If an $m \times n$ gradient matrix $\mathbf{M}$ has only $r \le n$ nonzero linearly independent rows (corresponding to $r$ distinct tokens in a minibatch), show that every active row of $\mathbf{U}\mathbf{V}^\top$ has unit $\ell_2$ norm regardless of whether its token occurred once or $500$ times.
    1. Move `token_emb` into the Muon parameter group on `TinyLM` and verify empirically that validation perplexity degrades compared to keeping `token_emb` in the AdamW group.

[Discussions](https://d2l.discourse.group/t/36764)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.8]{.kicker}

Muon<br>
**steepest descent under the spectral norm · Newton–Schulz · the hybrid with AdamW · long runs**
:::
:::

::: {.slide title="Adam Cannot See That a Weight Matrix Is a Map"}
[The problem]{.kicker}

`TinyLM`, a two-block transformer that predicts the next byte of Wikipedia
text: 393,216 of its 470,784 parameters (84%) sit in eight matrices that
map activations, $\mathbf{y} = \mathbf{W}\mathbf{x}$.

- Adam chooses the step of each entry from that entry's own gradients.
  Shuffle the entries of a matrix and it takes the same steps.
- The network sees only what a step $\Delta\mathbf{W}$ does to
  $\mathbf{y}$.

::: {.d2l-note}
What is the best step for a matrix, taken as a matrix?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive the steepest step under the Euclidean, $\ell_\infty$, and spectral
  norms;
- orthogonalize a matrix with five polynomial steps, and say which
  singular values arrive;
- write Muon's update with its scale and its weight decay;
- split a model's parameters between Muon and AdamW;
- read a comparison of two optimizers for what it shows.

**Idea:** the steepest step depends on how a step is measured.

**Running example:** `TinyLM`, 2,000 steps, Muon + AdamW against AdamW.
:::

::: {.slide title="Muon Is the Chapter's Third Answer to the Question of Direction"}
[Where this fits]{.kicker}

| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · AdamW · **Muon** |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice |

- **Uses:** Nesterov momentum, Adam as a sign method, decoupled weight
  decay, the spectral norm.
- **Skips:** variants of Muon, training on many devices, convergence
  theory.
:::

::: {.slide title="The Steepest Direction Depends on the Norm"}
[Norms and steepest descent]{.kicker}

Linearize, $f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) + \langle
\mathbf{g}, \mathbf{d}\rangle$ ($\mathbf{g}$: gradient), and ask for the
best step of size at most $\eta$:

$$\mathbf{d}^\star = \operatorname*{argmin}_{\|\mathbf{d}\| \leq \eta} \langle \mathbf{g}, \mathbf{d} \rangle.$$

![](../img/mdl-opt-norm-balls.svg){width=60%}

::: {.d2l-note}
Round ball: the **direction of SGD**. Box, $\max_i |d_i| \leq \eta$:
**sign descent**, which Adam smooths. Right: the ball for a matrix, next.
:::
:::

::: {.slide title="A Matrix Step Is Measured by What It Does to the Layer's Output"}
[Norms and steepest descent]{.kicker}

A layer computes $\mathbf{y} = \mathbf{W}\mathbf{x}$. A step
$\Delta\mathbf{W}$ changes its output by $\Delta\mathbf{W}\mathbf{x}$:

$$\|\Delta\mathbf{W}\mathbf{x}\|_2 \leq \|\Delta\mathbf{W}\|_2\,\|\mathbf{x}\|_2$$

- $\|\Delta\mathbf{W}\|_2$: the **spectral norm**, the largest stretch of
  any vector.
- SVD: $\mathbf{G} = \sum_i \sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top$ with
  orthonormal $\mathbf{u}_i$, orthonormal $\mathbf{v}_i$, and singular
  values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$.
- Spectral norm $\sigma_1$; Frobenius norm $\sqrt{\sum_i \sigma_i^2}$.

::: {.d2l-note}
128 singular values of one: no input is stretched, yet the Frobenius norm
is $\sqrt{128} \approx 11$. The Euclidean ball overstates such a step
elevenfold.
:::
:::

::: {.slide title="Under the Spectral Norm the Steepest Step Is UVᵀ"}
[Norms and steepest descent]{.kicker}

Gradient $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$,
inner product $\langle \mathbf{G}, \mathbf{A}\rangle = \sum_{i,j}
G_{ij}A_{ij} = \mathrm{tr}(\mathbf{G}^\top\mathbf{A})$:

$$\operatorname*{argmin}_{\|\Delta\mathbf{W}\|_2 \leq \eta} \langle \mathbf{G}, \Delta\mathbf{W}\rangle = -\eta\, \mathbf{U}\mathbf{V}^\top$$

For any $\mathbf{A}$ with $\|\mathbf{A}\|_2 \leq 1$:

1. $\langle \mathbf{G}, \mathbf{A}\rangle =
   \mathrm{tr}(\boldsymbol{\Sigma}\,\mathbf{U}^\top\mathbf{A}\mathbf{V}) =
   \sum_i \sigma_i\,(\mathbf{U}^\top\mathbf{A}\mathbf{V})_{ii}$.
2. $\|\mathbf{U}^\top\mathbf{A}\mathbf{V}\|_2 \leq 1$, so no entry
   exceeds 1: $\langle \mathbf{G}, \mathbf{A}\rangle \leq \sum_i \sigma_i$.
3. $\mathbf{A} = \mathbf{U}\mathbf{V}^\top$ gives
   $\mathbf{U}^\top\mathbf{A}\mathbf{V} = \mathbf{I}$: attained.

::: {.d2l-note}
The **orthogonalized gradient**: singular vectors kept, every singular
value set to one.
:::
:::

::: {.slide title="Check: Three Balls, Three Steps for One Gradient"}
[Check]{.kicker}

$$\mathbf{G} = \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}, \qquad \sigma_1 \approx 3.41, \quad \sigma_2 \approx 0.59$$

Up to the factor $-\eta$, what are the steepest steps in the round ball,
the box, and the spectral ball, and what are the singular values of each?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Round: $\mathbf{G}/\|\mathbf{G}\|_F$, singular values $0.99$
and $0.17$. Box: the matrix of four ones, $2$ and $0$. Spectral: the
identity, $1$ and $1$. Equal steps per entry are not equal steps per
direction.
:::
:::
:::

::: {.slide title="Weak Directions Get a Full Step, and the Ball Bounds What It Does"}
[Norms and steepest descent]{.kicker}

**Objection.** Small singular values belong to the directions that a
minibatch identifies most weakly. Some are noise. $\mathbf{U}\mathbf{V}^\top$
moves along them at the full rate.

1. **The ball.** $\|\Delta\mathbf{W}\mathbf{x}\|_2 \leq \eta\,\|\mathbf{x}\|_2$
   for every input $\mathbf{x}$: a weak direction disturbs the layer no
   more than the strongest.
2. **Averaging.** Muon orthogonalizes momentum, an average over about 20
   steps.
3. **Approximation.** Five polynomial steps leave the smallest singular
   values small.

::: {.d2l-note}
Answers 2 and 3 belong to the algorithm. We measure them on `TinyLM`.
:::
:::

::: {.slide title="Embeddings, Head, and Vectors Stay with AdamW"}
[Norms and steepest descent]{.kicker}

| ball on the step | steepest step | loss decrease (linearized) | parameters |
|:--|:--|:--|:--|
| Euclidean | $-\eta\, \mathbf{g} / \|\mathbf{g}\|_2$ | $\eta\,\|\mathbf{g}\|_2$ | no structure assumed |
| box ($\ell_\infty$) | $-\eta\, \mathrm{sign}(\mathbf{g})$ | $\eta\,\|\mathbf{g}\|_1$ | coordinates of different scale |
| spectral | $-\eta\, \mathbf{U}\mathbf{V}^\top$ | $\eta\sum_i \sigma_i$ | hidden matrices |

- Hidden matrix: input and output are both activations.
- Embedding table: one-hot input, each row is looked up alone.
- Output head: a dense matrix, kept with AdamW by convention. It trained
  better that way; the norm argument does not say why.
:::

::: {.slide title="Muon Is Shampoo without Memory"}
[Norms and steepest descent]{.kicker}

Shampoo preconditions an $m \times n$ matrix with two accumulated
matrices, $\mathbf{L}_t = \sum_s \mathbf{G}_s\mathbf{G}_s^\top$ and
$\mathbf{R}_t = \sum_s \mathbf{G}_s^\top\mathbf{G}_s$. Its step is
$\mathbf{L}_t^{-1/4}\mathbf{G}_t\mathbf{R}_t^{-1/4}$. Keep only the
current gradient $\mathbf{G} =
\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$:

$$(\mathbf{G}\mathbf{G}^\top)^{-1/4}\mathbf{G}(\mathbf{G}^\top\mathbf{G})^{-1/4} = \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\mathbf{U}^\top\,\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\,\mathbf{V}\boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top = \mathbf{U}\mathbf{V}^\top.$$

::: {.d2l-note}
Adam without its averages: the steepest step in the box. Shampoo without
its sums: the steepest step in the spectral ball.
:::

[Gupta et al., 2018; Bernstein and Newhouse, 2024]{.small}
:::

::: {.slide title="Odd Matrix Polynomials Act on Singular Values Alone"}
[Newton–Schulz]{.kicker}

Wanted: $\mathbf{U}\mathbf{V}^\top$ without an SVD. For numbers $a$, $b$,
$c$ and $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$,

$$p(\mathbf{X}) = a\mathbf{X} + b\,(\mathbf{X}\mathbf{X}^\top)\mathbf{X} + c\,(\mathbf{X}\mathbf{X}^\top)^2\mathbf{X} = \mathbf{U}\,p(\boldsymbol{\Sigma})\,\mathbf{V}^\top$$

because $\mathbf{X}\mathbf{X}^\top =
\mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$. The Newton–Schulz
iteration:

1. $\mathbf{X} \leftarrow \mathbf{G} / \|\mathbf{G}\|_F$, so every
   singular value is in $[0, 1]$.
2. Five times: $\mathbf{A} \leftarrow \mathbf{X}\mathbf{X}^\top$,
   $\mathbf{X} \leftarrow a\mathbf{X} + (b\mathbf{A} +
   c\mathbf{A}^2)\mathbf{X}$.

::: {.d2l-note}
Choose $p$ to push small numbers up and to keep numbers near 1 near 1.
Zero stays zero.
:::
:::

::: {.slide title="The Tuned Quintic Gives Up Convergence for Speed"}
[Newton–Schulz]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
- Cubic $\tfrac{3}{2}x - \tfrac{1}{2}x^3$: slope 1.5 at zero, converges
  to 1.
- Quintic $3.4445x - 4.7750x^3 + 2.0315x^5$: slope 3.44, ends in a band
  from 0.68 to 1.2.
- Polar Express (orange): a different quintic at each step narrows the
  band.
:::

::: {.col}
![](../img/mdl-opt-ns-maps.svg){width=100%}
:::
:::

[Quintic: Jordan et al., 2024. Per-step quintics: Amsel et al., 2026]{.small}
:::

::: {.slide title="Check: How Small a Singular Value Can Five Steps Lift?"}
[Check]{.kicker}

A singular value is $\sigma$ times the Frobenius norm. Near zero the
quintic multiplies it by 3.4445 per step. How large must $\sigma$ be to
reach the band, at 0.68, in five steps? And in three?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Five steps multiply by at most $3.4445^5 \approx 485$, so
$\sigma \geq 0.68/485 \approx 0.15\%$. Three steps multiply by 41:
$\sigma \geq 1.7\%$. Smaller singular values are raised and still fall
short.
:::
:::
:::

::: {.slide title="On a Gaussian Matrix Three Steps Already Reach the Band"}
[Newton–Schulz]{.kicker}

A random $96 \times 64$ Gaussian matrix: singular values from about 2%
to a little over 20% of the Frobenius norm, enough for three steps.
Spectrum after $k$ steps of the quintic:

@!muon-orthogonalization-by-newton-schulz-2
:::

::: {.slide title="On the Gradient of a New TinyLM Many Singular Values Fall Short"}
[Newton–Schulz]{.kicker}

One minibatch, the four hidden matrices of block 1, five quintic steps.

| matrix | $\sigma_1 / \|\mathbf{G}\|_F$ | singular values that reach 0.68 | norm relative to $\mathbf{U}\mathbf{V}^\top$ |
|:--|:--|:--|:--|
| `qkv`, $384 \times 128$ | 0.99 | about half | about 0.7 |
| `proj`, $128 \times 128$ | 0.99 | under a third | under 0.6 |
| `mlp1`, $512 \times 128$ | about 0.7 | about 90% | 0.9 |
| `mlp2`, $128 \times 512$ | over 0.9 | about 60% | about 0.8 |

- $\sigma_1 / \|\mathbf{G}\|_F = 0.99$: by its norm, a gradient of rank
  one.

::: {.d2l-note}
Five steps flatten the top of the spectrum and leave the tail small.
:::
:::

::: {.slide title="Muon Orthogonalizes Nesterov Momentum, Scaled to AdamW's Step"}
[Muon from scratch]{.kicker}

$$\mathbf{M}_t = \beta \mathbf{M}_{t-1} + \mathbf{G}_t, \qquad \mathbf{W}_{t+1} = (1 - \eta\lambda)\,\mathbf{W}_t - \eta\cdot 0.2\sqrt{\max(m,n)}\;\mathrm{NS}_5(\mathbf{G}_t + \beta \mathbf{M}_t)$$

- $\mathbf{W}$: $m \times n$ matrix; $\mathbf{G}_t$: its gradient; $\eta$:
  learning rate; $\mathrm{NS}_5$: five quintic steps.
- $\mathbf{M}_t$: momentum, $\beta = 0.95$. Averaging comes first:
  full-size directions belong to about 20 steps, not one minibatch.
- $0.2\sqrt{\max(m,n)}$: entry RMS $0.2\,\eta$, as for AdamW.
- $\lambda$: decoupled weight decay.

[Scale and decay: Liu et al., 2025]{.small}
:::

::: {.slide title="Check: The RMS-Matching Factor"}
[Check]{.kicker}

`TinyLM`'s first MLP matrix is $512 \times 128$. What is the RMS of an
entry of its exact $\mathbf{U}\mathbf{V}^\top$, and what does Muon's factor
turn it into?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** 128 singular values of one:
$\|\mathbf{U}\mathbf{V}^\top\|_F^2 = 128$ over $65{,}536$ entries, so the
RMS is $1/\sqrt{512} \approx 0.044$. Times $0.2\sqrt{512} \approx 4.5$:
$0.2$.
:::
:::
:::

::: {.slide title="Hidden Matrices Go to Muon, Everything Else to AdamW"}
[Muon from scratch]{.kicker}

Predicted: $2 \cdot 128 \cdot (384 + 128 + 512 + 512) = 393{,}216$
parameters in eight hidden matrices.

@!muon-dividing-the-census-1

- Muon: 84% of the parameters. AdamW: embeddings, head, vectors.
- Muon keeps one buffer, AdamW two: the optimizer state is 58% of
  AdamW's.
- Both halves share one learning rate; decay goes to the hidden matrices.
:::

::: {.slide title="In Mid-Run Five Steps Come Close to UVᵀ"}
[Muon from scratch]{.kicker}

Momentum buffers after 500 steps of the hybrid ($\eta = 0.003$,
$\lambda = 0.1$), against the gradient of the new network:

| matrix | $\sigma_1 / \|\mathbf{G}\|_F$, new network $\to$ step 500 | reach 0.68, new network $\to$ step 500 |
|:--|:--|:--|
| `qkv` | 0.99 $\to$ about a third | about half $\to$ 99% |
| `proj` | 0.99 $\to$ about 0.4 | under a third $\to$ about 90% |
| `mlp1` | about 0.7 $\to$ under 0.3 | about 90% $\to$ 99% |
| `mlp2` | over 0.9 $\to$ about 0.3 | about 60% $\to$ 100% |

- Norm about 0.9 of $\mathbf{U}\mathbf{V}^\top$: a step of about
  $0.18\,\eta$ per entry.

::: {.d2l-note}
Now every direction gets nearly a full step. Of the three answers to the
objection, the ball and the average remain.
:::
:::

::: {.slide title="On TinyLM the Hybrid Leads Early and Ties at Step 2,000"}
[Muon from scratch]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
Same initial weights, constant rate, grid $10^{-3}$ to $10^{-1}$,
$\lambda = 0.1$ on hidden matrices, validation loss in nats per byte.

- Step 500: about 1.76 against 1.88. Two runs of one setting differ by
  about 0.01.
- Below 1.6: at least 250 steps earlier.
- Step 2,000: under 0.05 apart, a tie.
:::

::: {.col}
@!muon-the-race-on-the-language-model-3
:::
:::
:::

::: {.slide title="At Three Times AdamW's Largest Usable Rate the Hybrid Still Trains"}
[Muon from scratch]{.kicker}

::: {.cols}
::: {.col}
AdamW:

@!muon-the-race-on-the-language-model-1
:::

::: {.col}
Muon + AdamW:

@!muon-the-race-on-the-language-model-2
:::
:::

::: {.d2l-note}
At 0.03 AdamW ends over a nat above its best run, the hybrid about a
tenth. A possible reason, untested: at equal RMS a rank-one step has
$\sqrt{128} \approx 11$ times the spectral norm of an orthogonalized one.
:::
:::

::: {.slide title="With Matched Tuning the Gain Is Tens of Percent"}
[Muon in long runs]{.kicker}

Ten optimizers, each tuned separately, at four model sizes. The fastest
precondition a matrix as a matrix, Muon among them:

| parameters | speedup of the fastest over AdamW |
|:--|:--|
| 0.1 billion | about $1.4\times$ |
| 1.2 billion | about $1.1\times$ |

- Speedup: ratio of the steps needed for the same loss.
- `TinyLM`, 0.47 million parameters: over 0.1 nats ahead at step 500, a
  tie at step 2,000.

::: {.d2l-note}
A small model shows the mechanism. It cannot rank optimizers.
:::

[Wen et al., 2025]{.small}
:::

::: {.slide title="The Spectral Ball Bounds One Step of One Matrix, Not a Long Run"}
[Muon in long runs]{.kicker}

- **Many steps.** Muon's step never shrinks. Without weight decay the
  matrices keep growing and the early lead over AdamW fades.
- **Two matrices.** Attention scores are dot products of two vectors,
  each from weights of its own, fed to a softmax. Under Muon they grew
  past 1,000 in trial runs for a trillion-parameter model.
- **Remedy.** After each step, scale both sets of weights down wherever
  the largest score exceeds 100.

::: {.d2l-note}
Our 2,000 steps are too short to show the first: at $\eta\lambda = 3
\times 10^{-4}$ the decay acts over $1/(\eta\lambda) = 3{,}333$ steps.
:::

[Liu et al., 2025; Kimi Team, 2025]{.small}
:::

::: {.slide title="Check: Can a Training Step Afford Fifteen Matrix Products?"}
[Check]{.kicker}

An $m \times m$ hidden matrix, a batch of $B$ tokens. The forward and
backward passes through the matrix cost about $6m^2B$ operations, one
product of two $m \times m$ matrices $2m^3$. Five Newton–Schulz steps take
three products each. What share do they add at $m = 768$,
$B = 524{,}288$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $15 \cdot 2m^3 / (6m^2B) = 5m/B \approx 0.7\%$. On `TinyLM`,
with $m = 128$, rectangular matrices and $B = 4{,}096$, the same count
gives about 12%: visible on a small model, negligible at scale.
:::
:::

[Cost count: Jordan et al., 2024]{.small}
:::

::: {.slide title="Recap: The Norm Chooses the Step, and for a Hidden Matrix It Is UVᵀ"}
[Wrap-up]{.kicker}

- The ball sets the steepest step: gradient, sign, or
  $\mathbf{U}\mathbf{V}^\top$ for a hidden matrix.
- Five quintic steps lift singular values above 0.15% of
  $\|\mathbf{G}\|_F$ to 0.68 or more: under a third to nine tenths of
  them on a new `TinyLM`, nine tenths or more after 500 steps.
- $0.2\sqrt{\max(m, n)}$ matches AdamW's step, so both share one rate.
- `TinyLM`: over 0.1 nats ahead at step 500, a tie at 2,000, one more
  usable rate at the top.
- Matched tuning at scale: $1.4\times$ shrinking to $1.1\times$.

**Next:** how the learning rate should change during a run.
:::
