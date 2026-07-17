# Muon
:label:`sec_muon`

Every optimizer so far has treated the parameters as one long vector. SGD
moves that vector against the gradient; Adam rescales each entry; AdamW
shrinks them all uniformly. Yet the census of :numref:`subsec_tinylm` showed
that a network is not a homogeneous vector: it is a collection of embedding
tables, hidden *matrices*, and normalization vectors, with most of the
parameters in the matrices. This section takes that structure seriously, and
the reward is the first credible challenger to the Adam family's decade-long
hold on large-scale training.

The organizing idea of this chapter says an optimizer begins with a choice of
descent direction, and this section makes the choice explicit: the direction
of steepest descent *depends on the norm* used to measure the step. Under the
Euclidean norm the answer is the gradient, and we recover SGD. Under the
$\ell_\infty$ norm the answer is the sign of the gradient, and we recover, in
essence, Adam. Under the *spectral* norm, the natural way to measure a matrix,
the answer is the gradient with its singular values erased, and we arrive at
Muon :cite:`Jordan.Jin.Boza.ea.2024`, which since 2024 has gone from a
speed-run leaderboard to trillion-parameter training runs. We derive it, build
it in about fifteen lines, race it against AdamW on the testbeds of
:numref:`sec_adam`, place it in the family of preconditioned methods it
belongs to, and finish with an honest look at the evidence.

```{.python .input #muon}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #muon}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import optax
```

## The Norm Decides the Direction

### Steepest Descent Under a Ball

What does it mean for a direction to be "steepest"? Linearize the loss around
the current iterate, $f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) +
\langle \mathbf{g}, \mathbf{d} \rangle$, and ask for the step that decreases
it most among all steps of a given size:

$$
\mathbf{d}^\star = \operatorname*{argmin}_{\|\mathbf{d}\| \leq \eta} \; \langle \mathbf{g}, \mathbf{d} \rangle.
$$
:eqlabel:`eq_muon-ball`

The problem is not fully posed until we say which norm defines the ball
$\|\mathbf{d}\| \leq \eta$, and different norms give genuinely different
answers :cite:`Bernstein.Newhouse.2024`. Under the Euclidean norm the ball is
round, the minimizer points straight along $-\mathbf{g}$, and
:eqref:`eq_muon-ball` returns gradient descent. So "just follow the gradient"
was never norm-free; it is the Euclidean choice, made silently, and every
method in this chapter that we described as a modification of SGD can instead
be read as a different answer to the same question.

Take the $\ell_\infty$ norm, $\|\mathbf{d}\|_\infty = \max_i |d_i|$. The ball
is now a box: a step is "size $\eta$" as long as *no single coordinate* moves
more than $\eta$. To make $\langle \mathbf{g}, \mathbf{d} \rangle$ as negative
as possible we push every coordinate to its wall, and the minimizer is
$\mathbf{d}^\star = -\eta\, \mathrm{sign}(\mathbf{g})$: sign descent. This is
not a new acquaintance. In :numref:`sec_adam` we switched off Adam's two
averages ($\beta_1 = \beta_2 = 0$, $\epsilon \to 0$) and were left with
exactly $\eta\,\mathrm{sign}(\mathbf{g}_t)$, and we saw that Adam's advantage
over SGD on language models tracks the advantage of sign descent
:cite:`Kunstner.Chen.Lavington.ea.2023`. In the geometric reading, Adam is
smoothed steepest descent under $\ell_\infty$: the box says "move every
coordinate the same distance, however lopsided the gradient", which is
precisely the per-coordinate equalization that :numref:`sec_adam` found to be
Adam's real work, and the moment estimates are there to keep the sign stable
under minibatch noise. Choosing this ball is sensible exactly when
coordinates have wildly different gradient scales, which is what the
heterogeneity of language models produces.

### Matrices and the Spectral Norm

Both balls above treat the parameter as a bag of coordinates. But the census
told us where the parameters actually live: in our tiny transformer, about
95% sit in two-dimensional hidden matrices whose job is to *transform
activations*, $\mathbf{y} = \mathbf{W}\mathbf{x}$. For such a parameter, what
we should mean by "a step of size $\eta$" is not how much the entries change
but how much the layer's *behavior* changes, and that is governed by the
spectral norm: $\|\Delta \mathbf{W} \mathbf{x}\|_2 \leq \|\Delta
\mathbf{W}\|_2\, \|\mathbf{x}\|_2$, with equality for the worst-case input.
The spectral norm $\|\Delta \mathbf{W}\|_2$, the largest singular value, is
the largest factor by which the update can stretch any activation vector
passing through the layer. Flattening the matrix and using the Euclidean norm
of the entries (the Frobenius norm) instead adds up energy across all
$\min(m, n)$ singular directions, so it can call an update "large" even when
its effect on every activation is small; controlling the spectral norm of
updates is also exactly the condition under which feature learning survives
scaling the network up :cite:`Yang.Simon.Bernstein.2023`, a thread
:numref:`sec_scaling` picks up.

So let the ball be spectral and solve :eqref:`eq_muon-ball` for a matrix
parameter with gradient $\mathbf{G}$. Write the reduced singular value
decomposition $\mathbf{G} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$.
The steepest step is the *orthogonalized gradient*:

$$
\Delta \mathbf{W}^\star = \operatorname*{argmin}_{\|\Delta \mathbf{W}\|_2 \leq \eta} \; \langle \mathbf{G}, \Delta \mathbf{W} \rangle = -\eta\, \mathbf{U} \mathbf{V}^\top.
$$
:eqlabel:`eq_muon-spectral-step`

The argument is one line of duality. For any $\mathbf{A}$ with
$\|\mathbf{A}\|_2 \leq 1$ we have $\langle \mathbf{G}, \mathbf{A} \rangle =
\mathrm{tr}(\boldsymbol{\Sigma}\, \mathbf{U}^\top \mathbf{A} \mathbf{V}) =
\sum_i \sigma_i\, (\mathbf{U}^\top \mathbf{A} \mathbf{V})_{ii} \leq \sum_i
\sigma_i$, since no entry of a matrix with unit spectral norm exceeds one;
choosing $\mathbf{A} = \mathbf{U}\mathbf{V}^\top$ attains the bound, and
negating it attains the minimum. (Checking the diagonal-entry claim is an
exercise; the surrounding theory, including what this has to do with
preconditioning, lives in :numref:`subsec_mdl-preconditioning-ladder`.)

Look at what :eqref:`eq_muon-spectral-step` does. The gradient
$\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ names a set of directions and
weights them by singular values that, for real training gradients, are
dominated by a few large ones. The steepest spectral step keeps the
directions and *erases the weights*: every direction the gradient identifies
moves at the same rate. This is per-direction equalization, in whatever basis
the gradient supplies, where Adam manages only per-coordinate equalization in
the axis basis it is handed. The reasoning also says where it does *not*
apply. An embedding table never multiplies a dense activation vector; its
input is one-hot, each row is looked up in isolation, and the operator
reading of the matrix collapses :cite:`Bernstein.2025`. Embeddings, the
output head, and the one-dimensional vectors stay with AdamW, and the census
split becomes an optimizer assignment. :numref:`tab_muon_norms` collects the
story so far.

:Steepest descent under three norms. Each ball yields a closed-form step and
suits a different population of the parameter census.
:label:`tab_muon_norms`

| ball on the step | steepest step | natural habitat |
|:--|:--|:--|
| Euclidean, $\|\mathbf{d}\|_2 \leq \eta$ | $-\eta\, \mathbf{g} / \|\mathbf{g}\|_2$ | no structure assumed (SGD) |
| box, $\|\mathbf{d}\|_\infty \leq \eta$ | $-\eta\, \mathrm{sign}(\mathbf{g})$ | coordinates of very different scale (Adam family) |
| spectral, $\|\Delta \mathbf{W}\|_2 \leq \eta$ | $-\eta\, \mathbf{U}\mathbf{V}^\top$ | hidden matrices acting on activations (Muon) |

:numref:`fig_opt_norm_balls` draws the three choices for one gradient.

![Steepest descent depends on the ball. Under $\ell_2$ the best step within the ball follows the gradient; under $\ell_\infty$ it moves to a corner, keeping only the signs; under the spectral norm — shown in singular-value coordinates, where the ball is the unit square — the best matrix update sits at the corner where every singular value equals one, whatever the spectrum $\sigma(\mathbf{G})$ of the gradient: that corner is the orthogonalization $\mathbf{U}\mathbf{V}^{\top}$.](../img/mdl-opt-norm-balls.svg)
:label:`fig_opt_norm_balls`

## Orthogonalization by Newton--Schulz

Equation :eqref:`eq_muon-spectral-step` asks for $\mathbf{U}\mathbf{V}^\top$,
and computing an SVD for every matrix at every step is out of the question:
SVD is expensive, hard to parallelize well on accelerators, and unavailable
in the low-precision arithmetic that training runs in. Muon's answer is a
classical iteration rediscovered for this purpose. Notice that for any *odd*
polynomial applied as a matrix polynomial,

$$
p(\mathbf{X}) = a\mathbf{X} + b\,(\mathbf{X}\mathbf{X}^\top)\mathbf{X} + c\,(\mathbf{X}\mathbf{X}^\top)^2\mathbf{X} = \mathbf{U}\, p(\boldsymbol{\Sigma})\, \mathbf{V}^\top,
$$
:eqlabel:`eq_muon-odd-poly`

because each factor of $\mathbf{X}\mathbf{X}^\top$ contributes
$\mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$ and the orthogonal factors
telescope. A polynomial in the matrix is the same polynomial applied to each
singular value, with $\mathbf{U}$ and $\mathbf{V}$ untouched. So we can drive
all singular values toward $1$, never computing them, by iterating a scalar
polynomial that has $1$ as an attracting fixed point on $(0, 1]$. Dividing
$\mathbf{X}$ by its Frobenius norm first guarantees every singular value
starts in $(0, 1]$.

The classical Newton--Schulz cubic does this with $p(x) = \tfrac{3}{2}x -
\tfrac{1}{2}x^3$, but its progress near zero is slow: a tiny singular value
only grows by a factor of $1.5$ per iteration. :citet:`Jordan.Jin.Boza.ea.2024`
instead tuned the quintic

$$
p(x) = 3.4445\,x - 4.7750\,x^3 + 2.0315\,x^5,
$$
:eqlabel:`eq_muon-quintic`

whose slope at the origin is $3.4445$, so a direction hundreds of times
weaker than the dominant one still reaches order $1$ within five iterations.
The price of the aggressive slope is that the iteration does not converge to
$1$; it oscillates in a band around it. For an optimizer this is a fine
trade: we need "all directions move at roughly the same rate", not
machine-precision orthogonality, and the iteration is stable enough to run
in `bfloat16`. Five iterations of :eqref:`eq_muon-quintic` is a handful of
matrix multiplications, the operation GPUs are best at.

```{.python .input #muon-orthogonalization-by-newton-schulz-1}
%%tab pytorch
def newton_schulz(M, num_iters=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M / (M.norm() + eps)
    for _ in range(num_iters):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X
```

```{.python .input #muon-orthogonalization-by-newton-schulz-1}
%%tab jax
def newton_schulz(M, num_iters=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M / (jnp.linalg.norm(M) + eps)
    for _ in range(num_iters):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X
```

Let's watch it work. We take a random $96 \times 64$ matrix, whose singular
values after normalization span an order of magnitude, and plot the spectrum
after 0, 1, 3, and 5 iterations. (The SVD below is for *measuring* the
result; the iteration itself never computes one.)

```{.python .input #muon-orthogonalization-by-newton-schulz-2}
%%tab pytorch
torch.manual_seed(0)
G = torch.randn(96, 64)
sigmas = [torch.linalg.svdvals(newton_schulz(G, num_iters=k))
          for k in (0, 1, 3, 5)]
d2l.plot(torch.arange(1, 65), sigmas, 'index', 'singular value',
         legend=[f'k = {k}' for k in (0, 1, 3, 5)], yscale='log')
print(f'singular values after 5 iterations: '
      f'[{sigmas[-1].min():.2f}, {sigmas[-1].max():.2f}]')
```

```{.python .input #muon-orthogonalization-by-newton-schulz-2}
%%tab jax
G = jax.random.normal(jax.random.key(0), (96, 64))
sigmas = [jnp.linalg.svd(newton_schulz(G, num_iters=k), compute_uv=False)
          for k in (0, 1, 3, 5)]
d2l.plot(jnp.arange(1, 65), sigmas, 'index', 'singular value',
         legend=[f'k = {k}' for k in (0, 1, 3, 5)], yscale='log')
print(f'singular values after 5 iterations: '
      f'[{sigmas[-1].min():.2f}, {sigmas[-1].max():.2f}]')
```

One iteration lifts the whole spectrum; three compress it to within a factor
of a few; after five, every singular value sits in a band around $1$, from
roughly $0.7$ to $1.2$. That is $\mathbf{U}\mathbf{V}^\top$ for an
optimizer's purposes, at the cost of fifteen matrix multiplications. (A
*nearly singular* direction takes a few more iterations to climb out, at
$3.4$-fold per pass; the exercises probe how much that matters in
training.)

## Muon from Scratch

### The Update

Muon assembles three ingredients we now have in hand. Gradients are noisy, so
we orthogonalize not the raw gradient but a momentum buffer, the leaky
average of :numref:`sec_momentum`, with the same $\mu = 0.95$ used by its
authors. The buffer is orthogonalized by Newton--Schulz. And the result is
rescaled once per matrix shape:

$$
\mathbf{M}_t = \mu\, \mathbf{M}_{t-1} + \mathbf{G}_t,
\qquad
\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \cdot 0.2 \sqrt{\max(m, n)}\; \mathrm{NS}_5(\mathbf{M}_t)
$$
:eqlabel:`eq_muon-update`

for an $m \times n$ matrix. The scale factor deserves a sentence, because it
is what makes Muon a drop-in teammate for AdamW. An orthogonalized matrix has
$\min(m, n)$ singular values equal to $1$, hence Frobenius norm
$\sqrt{\min(m, n)}$ and root-mean-square entry size $1 / \sqrt{\max(m, n)}$:
without correction, wide matrices would take smaller per-entry steps than
square ones. Multiplying by $0.2\sqrt{\max(m, n)}$ makes every matrix's
update have entrywise RMS $0.2\,\eta$ regardless of shape, which matches the
typical RMS of an AdamW update. With this convention, introduced for the
Moonlight model, a learning rate and weight decay tuned for AdamW transfer
directly to Muon :cite:`Liu.Su.Yao.ea.2025`, so the hybrid optimizer below
needs only one learning rate. (The original Muon used a different
shape factor, and the theoretically derived scale is
$\sqrt{\text{fan-out} / \text{fan-in}}$ per :citet:`Bernstein.2025`; the
distinction matters when transferring across model widths, which is
:numref:`sec_scaling`'s subject.)

The implementation is short. `reshape` handles the one wrinkle we will need
later: a convolution kernel is a matrix in disguise, one row per output
channel, and flattening it lets the same code precondition CNNs. On our tiny
matrices the fifteen extra multiplications add a visible fraction to the
step time; at production scale, where the forward and backward passes
dwarf them, the reported overhead is around one percent of the training
FLOPs :cite:`Jordan.Jin.Boza.ea.2024`.

```{.python .input #muon-the-update}
%%tab pytorch
class Muon(torch.optim.Optimizer):
    """Steepest descent under the spectral norm: orthogonalized momentum."""
    def __init__(self, params, lr, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                buf = self.state[p].setdefault('buf', torch.zeros_like(p))
                buf.mul_(group['momentum']).add_(p.grad)
                M = buf.reshape(len(buf), -1)  # flattens conv kernels
                O = newton_schulz(M).reshape(p.shape)
                p.add_(O, alpha=-group['lr'] * 0.2 * math.sqrt(max(M.shape)))
```

```{.python .input #muon-the-update}
%%tab jax
def scratch_muon(learning_rate, momentum=0.95):
    def init(params):
        return jax.tree.map(jnp.zeros_like, params)
    def update(grads, bufs, params=None):
        bufs = jax.tree.map(lambda b, g: momentum * b + g, bufs, grads)
        def step(b):
            M = b.reshape(-1, b.shape[-1])  # flattens conv kernels
            O = newton_schulz(M).reshape(b.shape)
            return -learning_rate * 0.2 * math.sqrt(max(M.shape)) * O
        return jax.tree.map(step, bufs), bufs
    return optax.GradientTransformation(init, update)
```

### Dividing the Census

Muon is an optimizer for hidden matrices only, so a real training run is a
*hybrid*: the parameter-group machinery of :numref:`sec_adamw`, with the
census deciding who goes where. Hidden matrices go to Muon; embeddings, the
output head, and every one-dimensional tensor go to AdamW.

```{.python .input #muon-dividing-the-census-1}
%%tab pytorch
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)

def split_lm(model):
    hidden, rest = [], []
    for name, p in model.named_parameters():
        is_hidden = p.ndim == 2 and 'emb' not in name and 'head' not in name
        (hidden if is_hidden else rest).append(p)
    return hidden, rest

model = d2l.TinyLM(len(data.vocab))
hidden, rest = split_lm(model)
print(f'Muon:  {len(hidden):>2} tensors, '
      f'{sum(p.numel() for p in hidden):>7} parameters')
print(f'AdamW: {len(rest):>2} tensors, '
      f'{sum(p.numel() for p in rest):>7} parameters')
```

```{.python .input #muon-dividing-the-census-1}
%%tab jax
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)

model = d2l.TinyLM(len(data.vocab))
counts = {'muon': [0, 0], 'adamw': [0, 0]}
for path, p in nnx.to_flat_state(nnx.state(model, nnx.Param)):
    name = '.'.join(str(k) for k in path)
    is_hidden = p.ndim == 2 and 'emb' not in name and 'head' not in name
    group = counts['muon' if is_hidden else 'adamw']
    group[0] += 1
    group[1] += p.size
for name, (num, size) in counts.items():
    print(f'{name}: {num:>2} tensors, {size:>7} parameters')
```

The eight hidden matrices hold about 95% of the parameters. Note what the
split does to optimizer state: AdamW carries two buffers per parameter, Muon
one, so the hybrid's state memory is nearly half of all-AdamW's, an
accounting :numref:`sec_adamw` taught us to notice. The factory below builds
the hybrid; with the RMS-matched scale of :eqref:`eq_muon-update`, both
halves share a single learning rate.

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

def muon_adamw(hidden, rest, lr):
    return MultiOptimizer(Muon(hidden, lr=lr),
                          torch.optim.AdamW(rest, lr=lr, weight_decay=0.0))
```

```{.python .input #muon-dividing-the-census-2}
%%tab jax
def muon_adamw(lr, exclude=('emb', 'head')):
    def labels(params):
        def label(path, p):
            name = jax.tree_util.keystr(path)
            is_hidden = (p.ndim >= 2
                         and not any(s in name for s in exclude))
            return 'muon' if is_hidden else 'adamw'
        return jax.tree_util.tree_map_with_path(label, params)
    return optax.multi_transform(
        {'muon': scratch_muon(lr),
         'adamw': optax.adamw(lr, weight_decay=0.0)}, labels)
```

### The Race on the Language Model

The protocol is the one from :numref:`sec_adam`: same model, same
initialization, 2,000 steps at a constant learning rate, a four-point
learning-rate grid per contestant, best final training loss speaks for its
family. Weight decay is switched off in both arms so that the *only*
difference between them is the direction of the update on the hidden
matrices. First the baseline, all parameters on AdamW:

```{.python .input #muon-the-race-on-the-language-model-1}
%%tab pytorch
def final_loss(losses, k=100):
    v = sum(losses[-k:]) / k
    return v if math.isfinite(v) else float('inf')

def smooth(losses, k=25):
    return [sum(losses[i:i + k]) / k
            for i in range(0, len(losses) - k + 1, k)]

def run_lm(make_optimizer, lrs, num_steps=2000):
    curves = {}
    for lr in lrs:
        torch.manual_seed(0)
        model = d2l.TinyLM(len(data.vocab))
        curves[lr] = d2l.train_lm(model, data, make_optimizer(model, lr),
                                  num_steps)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}')
    return curves

adamw_lm = run_lm(
    lambda model, lr: torch.optim.AdamW(model.parameters(), lr,
                                        weight_decay=0.0),
    lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #muon-the-race-on-the-language-model-1}
%%tab jax
def final_loss(losses, k=100):
    v = sum(losses[-k:]) / k
    return v if math.isfinite(v) else float('inf')

def smooth(losses, k=25):
    return [sum(losses[i:i + k]) / k
            for i in range(0, len(losses) - k + 1, k)]

def run_lm(make_tx, lrs, num_steps=2000):
    curves = {}
    for lr in lrs:
        model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        curves[lr] = d2l.train_lm(model, data, optimizer, num_steps)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}')
    return curves

adamw_lm = run_lm(lambda lr: optax.adamw(lr, weight_decay=0.0),
                  lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

Now the hybrid, over the same grid; thanks to the RMS matching, the grid
means the same thing for both:

```{.python .input #muon-the-race-on-the-language-model-2}
%%tab pytorch
muon_lm = run_lm(
    lambda model, lr: muon_adamw(*split_lm(model), lr),
    lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #muon-the-race-on-the-language-model-2}
%%tab jax
muon_lm = run_lm(muon_adamw, lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #muon-the-race-on-the-language-model-3}
best_adamw = min(adamw_lm, key=lambda lr: final_loss(adamw_lm[lr]))
best_muon = min(muon_lm, key=lambda lr: final_loss(muon_lm[lr]))
d2l.plot(list(range(0, 2000, 25)),
         [smooth(adamw_lm[best_adamw]), smooth(muon_lm[best_muon])],
         'step', 'training loss',
         legend=[f'AdamW, lr {best_adamw:g}',
                 f'Muon + AdamW, lr {best_muon:g}'])
print(f'final perplexity: '
      f'AdamW {math.exp(final_loss(adamw_lm[best_adamw])):.2f}, '
      f'Muon+AdamW {math.exp(final_loss(muon_lm[best_muon])):.2f}')
```

:begin_tab:`pytorch`
Tuned against tuned, the hybrid finishes at or slightly below AdamW; in our
runs the two winners land within run-to-run noise of each other, and we
claim no more than parity from a single seed. Parity is not nothing — the
hybrid gets there carrying nearly half the optimizer state — but a
0.4M-parameter model trained for a minute cannot show much more. The
fair-tuning literature discussed at the end of this section measures
Muon-family gains in tens of percent of data efficiency at small scale,
an effect two thousand steps can only hint at. The demo is mechanism, not
benchmark; before drawing conclusions, look at the same race in the JAX tab,
where the identical protocol produces a very different margin.
:end_tab:

:begin_tab:`jax`
Here the tuned hybrid beats tuned AdamW by a wide margin — several tenths of
a nat, a visibly lower curve throughout. Resist the strong conclusion: the
PyTorch tab runs the identical protocol and ends in a near-tie. The two
frameworks differ in details as mundane as default layer initialization,
and races this small are sensitive to all of them. What survives both tabs
is the weaker, honest statement: at matched tuning the hybrid never lost,
while carrying nearly half the optimizer state. That claims about optimizers
are this protocol-sensitive is not an embarrassment to hide but the
section's recurring lesson, and the fair-tuning studies at the end of the
section exist precisely because of it. The demo is mechanism, not
benchmark.
:end_tab:

### The Same Race on a CNN

:numref:`sec_adam` found that Adam's advantage over SGD, dramatic on the
language model, nearly vanished on a CNN. It is natural to ask the same
question of Muon. We reuse the compact Fashion-MNIST CNN from that section,
along with its test-accuracy check; `reshape` in the update flattens each
convolution kernel to a matrix with one row per output channel, and the
output head stays with AdamW.

```{.python .input #muon-the-same-race-on-a-cnn-1}
%%tab pytorch
fashion = d2l.FashionMNIST(batch_size=256)

def make_cnn():
    return nn.Sequential(
        nn.LazyConv2d(32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.LazyConv2d(64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.LazyLinear(128), nn.ReLU(), nn.LazyLinear(10))

def split_cnn(model):
    head = model[-1].weight
    hidden = [p for p in model.parameters()
              if p.ndim >= 2 and p is not head]
    rest = [p for p in model.parameters() if p.ndim < 2 or p is head]
    return hidden, rest

def test_accuracy(model, data):
    device = d2l.try_gpu()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in data.val_dataloader():
            X, y = X.to(device), y.to(device)
            correct += float((model(X).argmax(axis=1) == y).sum())
            total += y.numel()
    return correct / total

def run_cnn(make_optimizer, lrs, num_steps=2000):
    curves, accs = {}, {}
    for lr in lrs:
        torch.manual_seed(0)
        model = make_cnn()
        model(next(iter(fashion.train_dataloader()))[0])
        curves[lr] = d2l.train_lm(model, fashion, make_optimizer(model, lr),
                                  num_steps)
        accs[lr] = test_accuracy(model, fashion)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}, '
              f'test accuracy {accs[lr]:.3f}')
    return curves, accs

adamw_cnn, adamw_acc = run_cnn(
    lambda model, lr: torch.optim.AdamW(model.parameters(), lr,
                                        weight_decay=0.0),
    lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #muon-the-same-race-on-a-cnn-1}
%%tab jax
fashion = d2l.FashionMNIST(batch_size=256)

class FashionCNN(nnx.Module):
    def __init__(self, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.fc1 = nnx.Linear(64 * 7 * 7, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, X):
        X = nnx.max_pool(nnx.relu(self.conv1(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = nnx.max_pool(nnx.relu(self.conv2(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = X.reshape(X.shape[0], -1)
        return self.fc2(nnx.relu(self.fc1(X)))

def test_accuracy(model, data):
    correct = total = 0
    for X, y in data.val_dataloader():
        pred = model(jnp.asarray(X)).argmax(axis=1)
        correct += float((pred == jnp.asarray(y)).sum())
        total += y.shape[0]
    return correct / total

def run_cnn(make_tx, lrs, num_steps=2000):
    curves, accs = {}, {}
    for lr in lrs:
        model = FashionCNN(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        curves[lr] = d2l.train_lm(model, fashion, optimizer, num_steps)
        accs[lr] = test_accuracy(model, fashion)
        print(f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}, '
              f'test accuracy {accs[lr]:.3f}')
    return curves, accs

adamw_cnn, adamw_acc = run_cnn(lambda lr: optax.adamw(lr, weight_decay=0.0),
                               lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #muon-the-same-race-on-a-cnn-2}
%%tab pytorch
muon_cnn, muon_acc = run_cnn(
    lambda model, lr: muon_adamw(*split_cnn(model), lr),
    lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #muon-the-same-race-on-a-cnn-2}
%%tab jax
muon_cnn, muon_acc = run_cnn(lambda lr: muon_adamw(lr, exclude=('fc2',)),
                             lrs=[3e-4, 1e-3, 3e-3, 1e-2])
```

```{.python .input #muon-the-same-race-on-a-cnn-3}
best_adamw_cnn = min(adamw_cnn, key=lambda lr: final_loss(adamw_cnn[lr]))
best_muon_cnn = min(muon_cnn, key=lambda lr: final_loss(muon_cnn[lr]))
d2l.plot(list(range(0, 2000, 25)),
         [smooth(adamw_cnn[best_adamw_cnn]), smooth(muon_cnn[best_muon_cnn])],
         'step', 'training loss',
         legend=[f'AdamW, lr {best_adamw_cnn:g}',
                 f'Muon + AdamW, lr {best_muon_cnn:g}'])
print(f'test accuracy: AdamW {adamw_acc[best_adamw_cnn]:.3f}, '
      f'Muon+AdamW {muon_acc[best_muon_cnn]:.3f}')
```

The result rewards reading both scoreboards. On training loss the hybrid
pulls clearly ahead: orthogonalized updates drive this small CNN into its
memorization regime several times faster. On test accuracy, the number a
vision practitioner actually cares about, the two land within about a point
of each other. Optimizing faster and predicting better are different claims,
and on a small, quickly saturated task the second is the one that resists
improvement — the same compression of differences that :numref:`sec_adam`
found between Adam and SGD here. The general lesson survives translation:
an optimizer comparison is a statement about a workload and a metric, not a
universal ranking.

### Library Implementations

:begin_tab:`pytorch`
PyTorch ships Muon in core since version 2.9. Mirroring our scratch version,
`torch.optim.Muon` accepts only the 2-D hidden matrices (its documentation
directs embeddings, biases, and heads to AdamW), applies Nesterov-style
momentum before orthogonalizing, and offers the RMS-matched scale of
:eqref:`eq_muon-update` as `adjust_lr_fn='match_rms_adamw'`. Its default
weight decay is 0.1, not zero, so we switch it off to match the protocol
above.
:end_tab:

:begin_tab:`jax`
Optax ships Muon in `optax.contrib`. Unlike our scratch version it manages
the split internally: parameters marked with `MuonDimensionNumbers` are
orthogonalized and everything else falls through to Adam. By default every
2-D parameter is treated as a hidden matrix, which would orthogonalize the
embedding tables too, so we pass an explicit spec that sends embeddings and
the head to Adam; `consistent_rms=0.2` selects the RMS-matched scale of
:eqref:`eq_muon-update`.
:end_tab:

```{.python .input #muon-library-implementations}
%%tab pytorch
torch.manual_seed(0)
model = d2l.TinyLM(len(data.vocab))
hidden, rest = split_lm(model)
optimizer = MultiOptimizer(
    torch.optim.Muon(hidden, lr=best_muon, weight_decay=0.0,
                     adjust_lr_fn='match_rms_adamw'),
    torch.optim.AdamW(rest, lr=best_muon, weight_decay=0.0))
losses = d2l.train_lm(model, data, optimizer, 2000)
print(f'final loss {final_loss(losses):.3f}')
```

```{.python .input #muon-library-implementations}
%%tab jax
def muon_spec(params):
    def spec(path, p):
        name = jax.tree_util.keystr(path)
        if p.ndim == 2 and 'emb' not in name and 'head' not in name:
            return optax.contrib.MuonDimensionNumbers()
        return None
    return jax.tree_util.tree_map_with_path(spec, params)

model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(0))
tx = optax.contrib.muon(learning_rate=best_muon, consistent_rms=0.2,
                        weight_decay=0.0,
                        muon_weight_dimension_numbers=muon_spec)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
losses = d2l.train_lm(model, data, optimizer, 2000)
print(f'final loss {final_loss(losses):.3f}')
```

## The Preconditioning Family

Muon looks exotic until it is placed on the family tree, where it turns out
to be the frugal child of a long line. Every branch answers the question
posed in :numref:`sec_gd`: gradient descent assumes round level sets, real
losses have curved ones, and some matrix should reshape the gradient
accordingly. Adam estimates a *diagonal* such matrix from gradient history.
The methods below estimate structure per layer, exploiting the same fact
Muon does: parameters come in matrices.

K-FAC :cite:`Martens.Grosse.2015` is the family's ancestor. It approximates
each layer's block of the Fisher information matrix, curvature measured
between the *distributions* the model defines, as a Kronecker product of two
small matrices, the second moments of the layer's inputs and of its output
gradients. Preconditioning by a Kronecker product costs two small inverses
rather than one enormous one, which made second-order-style updates feasible
for neural networks and connected them to the natural gradient of
:citet:`Amari.1998`.

Shampoo :cite:`Gupta.Koren.Singer.2018` keeps the two-sided structure but
builds the factors the AdaGrad way, from accumulated gradient statistics,
preconditioning each gradient matrix as $\mathbf{L}_t^{-1/4} \mathbf{G}_t
\mathbf{R}_t^{-1/4}$ with $\mathbf{L}_t = \sum_s \mathbf{G}_s
\mathbf{G}_s^\top$ and $\mathbf{R}_t = \sum_s \mathbf{G}_s^\top
\mathbf{G}_s$. A distributed implementation of Shampoo won the
external-tuning track of the AlgoPerf benchmark, finishing its workloads
about 30% faster than the tuned AdamW baseline
:cite:`Dahl.Schneider.Nado.ea.2023,Kasimbeg.Schneider.Eschenhagen.ea.2025` —
the strongest protocol-controlled evidence that matrix preconditioning pays.
SOAP :cite:`Vyas.Morwani.Zhao.ea.2024` refines it further by running Adam
inside Shampoo's slowly refreshed eigenbasis, cutting the overhead between
factor recomputations.

Muon's place in the tree is now easy to state: *it is Shampoo with the
memory removed*. Switch off Shampoo's accumulators, keeping only the current
gradient with SVD $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$,
and

$$
(\mathbf{G}\mathbf{G}^\top)^{-1/4}\, \mathbf{G}\, (\mathbf{G}^\top\mathbf{G})^{-1/4} = \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\mathbf{U}^\top \cdot \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \cdot \mathbf{V}\boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top = \mathbf{U}\mathbf{V}^\top,
$$
:eqlabel:`eq_muon-shampoo`

the orthogonalized gradient again :cite:`Bernstein.Newhouse.2024`. What Muon
discards is the preconditioner state and the factor inverses; what it keeps
is the geometry. On the other side of the tree, the same anatomy identifies
Lion :cite:`Chen.Liang.Huang.ea.2023`, an update found by symbolic program
search, as the lean member of the *sign* branch: one momentum buffer and a
sign, steepest descent under $\ell_\infty$ with even less state than Adam.
You will implement it in the exercises in about six lines. The full ladder,
from diagonal through Kronecker to spectral, with the derivations, is
assembled in :numref:`subsec_mdl-preconditioning-ladder`.

## Muon in the Wild

Muon's rise was unusually public. It debuted in late 2024 not in a paper but
as a record on the NanoGPT speedrun :cite:`Jordan.2024`, a standing
competition to train a fixed GPT-2-class model to a fixed validation loss on
fixed hardware in the least wall-clock time. The speedrun is a small model of
good evidence culture: one variable changes per record, the diff is public,
and anyone can rerun it. Its headline number, a baseline of 45 minutes
driven down to a few minutes in under two years, mixes optimizer,
architecture, and data-schedule improvements and should not be read as an
optimizer benchmark; but individual records isolate single changes, and
Muon's debut cut the then-record by roughly a third.

Production adoption followed within months. Moonshot's Moonlight report
demonstrated Muon at multi-billion-parameter scale, introduced the
RMS-matching convention of :eqref:`eq_muon-update`, and reported matched
losses for roughly half the training compute of its AdamW baseline
:cite:`Liu.Su.Yao.ea.2025`. Kimi K2, a trillion-parameter
mixture-of-experts model, was pretrained on 15.5 trillion tokens with
MuonClip, Muon plus a cap on attention logits (QK-clip) to contain the
instability that surfaced at that scale, and reports zero loss spikes for
the entire run :cite:`Kimi.Team.2025`. GLM-4.5 likewise trained with Muon
:cite:`Zeng.Lv.Zheng.ea.2025`. An optimizer in the training runs of multiple frontier
labs, and in core PyTorch, within two years of its first appearance is a
pace the field had not seen since Adam itself.

Against this, hold the sobering result of the fair-tuning literature.
:citet:`Wen.Hall.Ma.ea.2025` re-benchmarked eleven optimizers under matched,
per-optimizer hyperparameter tuning across model scales, and found that most
published "beats AdamW by 2×" claims deflate badly: the genuine speedups
belong to the matrix-preconditioned family (Muon, SOAP, Kron), but they are
roughly 1.4× at 100M parameters and shrink toward 1.1× by a billion. The
study's most instructive case is Sophia :cite:`Liu.Li.Hall.ea.2023`, a
second-order method that reported a 2× speedup on GPT-2 pretraining and did
not replicate under matched tuning — a finding published by a group
overlapping with Sophia's own authors, which is how self-correction is
supposed to work. None of this contradicts the production story; it bounds
it. Benchmark verdicts are also protocol-dependent: AlgoPerf's fixed tuning
budgets crowned Shampoo, the speedrun's unlimited tinkering crowned Muon,
and a comparison run at one scale with one tuning budget is evidence about
that protocol, not a universal ranking
:cite:`Schmidt.Schneider.Hennig.2021`. The honest summary for a practitioner
in 2026: AdamW remains the default; Muon on hidden matrices is the one
challenger with both a clean derivation and frontier-scale production
mileage, and its advantage is real but measured in tens of percent, not
multiples.

## Summary

Steepest descent is not one algorithm but a family indexed by a norm: the
Euclidean ball yields SGD, the $\ell_\infty$ box yields sign descent and its
smoothed form Adam, and the spectral ball, the right measure for a matrix
that transforms activations, yields the orthogonalized gradient
$\mathbf{U}\mathbf{V}^\top$. Muon computes it without an SVD by a tuned
five-step Newton--Schulz iteration, pure matrix multiplications that run in
low precision, applies it to the momentum buffer of each hidden matrix, and
rescales by $0.2\sqrt{\max(m, n)}$ so one learning rate serves both Muon and
the AdamW that handles embeddings, heads, and vectors. It is Shampoo without
accumulators, K-FAC's grandchild, and the spectral rung of the
preconditioning ladder.

On our tiny testbed the tuned hybrid never lost to tuned AdamW while
carrying nearly half the optimizer state, with a margin that ranged from
parity to substantial across two frameworks running the identical protocol;
on the CNN it optimized faster but generalized the same. All of this is
consistent with the production record: real gains of tens of percent on
transformer pretraining at matched tuning, trillion-parameter runs without
loss spikes, and no revolution. The methodological lesson is worth as much
as the method: optimizer claims deserve matched tuning, stated protocols,
and suspicion of round numbers.

## Exercises

1. Derive the sign-descent limit of Adam. Setting $\beta_1 = \beta_2 = 0$ in
   :eqref:`eq_adam-moments` and :eqref:`eq_adam-update`, show that the
   update becomes $\eta\, \mathbf{g}_t / (|\mathbf{g}_t| + \epsilon)$ and
   hence $\eta\, \mathrm{sign}(\mathbf{g}_t)$ as $\epsilon \to 0$. Which
   norm ball in :eqref:`eq_muon-ball` does this step solve? What do the two
   moving averages restore that the limit lacks?
1. Verify the RMS-matching factor. Show that $\|\mathbf{U}\mathbf{V}^\top\|_F
   = \sqrt{\min(m, n)}$ for an $m \times n$ matrix of rank $\min(m, n)$, so
   the orthogonalized update has entrywise RMS $1/\sqrt{\max(m, n)}$. Then
   instrument a short AdamW run of `TinyLM` to measure the actual RMS of its
   updates, and compare with the constant $0.2$ used in
   :eqref:`eq_muon-update`.
1. Rerun the tuned hybrid with `num_iters=1` and `num_iters=10` in
   `newton_schulz`. Measure final loss and wall-clock time per step. Where
   does the quality saturate, and why does one iteration already capture
   part of the benefit? (Plot the quintic $p(x)$ of :eqref:`eq_muon-quintic`
   to see what a single application does to the spectrum.)
1. Move the embedding tables and the output head into the Muon group and
   rerun the sweep. Explain what you observe using the one-hot-input
   argument: what does orthogonalizing an embedding table's momentum do to
   the update received by the rows of rare tokens?
1. Implement Lion :cite:`Chen.Liang.Huang.ea.2023` in about six lines: with
   buffer $\mathbf{m}_t$, update $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,
   \mathrm{sign}(\beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t)$
   followed by $\mathbf{m}_t = \beta_2 \mathbf{m}_{t-1} + (1 - \beta_2)\,
   \mathbf{g}_t$, with $(\beta_1, \beta_2) = (0.9, 0.99)$. Race it against
   AdamW and the hybrid on `TinyLM` at matched four-point tuning (Lion's
   best learning rate is typically several times smaller than AdamW's).
   How much optimizer state does each method carry per parameter?
1. Complete the proof of :eqref:`eq_muon-spectral-step`: show that if
   $\|\mathbf{A}\|_2 \leq 1$ then every diagonal entry of
   $\mathbf{U}^\top \mathbf{A} \mathbf{V}$ has absolute value at most $1$,
   and identify when equality holds simultaneously for all entries.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.9]{.kicker}

The norm decides the direction<br>
**steepest descent under three balls · Newton–Schulz · Muon vs. AdamW · the preconditioning family**
:::
:::

::: {.slide title="Steepest descent depends on the norm"}
[The chapter's frame, paid off]{.kicker}

Linearize and ask for the best step of size $\eta$:

$$\mathbf{d}^\star = \operatorname*{argmin}_{\|\mathbf{d}\| \leq \eta} \langle \mathbf{g}, \mathbf{d} \rangle.$$

Not fully posed until the **ball** is chosen (Bernstein & Newhouse, 2024):

- Euclidean ball → $-\eta\,\mathbf{g}/\|\mathbf{g}\|_2$: **SGD** was a choice, made silently.
- Box ($\ell_\infty$) → $-\eta\,\mathrm{sign}(\mathbf{g})$: every coordinate moves the
  same distance — the equalization that is Adam's real work (§9.6).
:::

::: {.slide title="Matrices want the spectral norm"}
A hidden matrix transforms activations: $\mathbf{y} = \mathbf{W}\mathbf{x}$.
The honest size of an update is what it does to activations:

$$\|\Delta\mathbf{W}\mathbf{x}\|_2 \leq \|\Delta\mathbf{W}\|_2\, \|\mathbf{x}\|_2.$$

. . .

Steepest descent under $\|\Delta\mathbf{W}\|_2 \leq \eta$, with
$\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$:

$$\Delta\mathbf{W}^\star = -\eta\, \mathbf{U}\mathbf{V}^\top$$

— keep the gradient's **directions**, erase its **weights**. Per-*direction*
equalization where Adam is per-coordinate.

::: {.d2l-note}
Embeddings see one-hot inputs — no operator, no spectral ball. They stay
with AdamW.
:::
:::

::: {.slide title="Orthogonalization without an SVD"}
Odd matrix polynomials act on singular values alone:
$p(\mathbf{X}) = \mathbf{U}\,p(\boldsymbol{\Sigma})\,\mathbf{V}^\top$.
Iterate a polynomial with fixed point 1; the tuned quintic
(Jordan et al., 2024)

$$p(x) = 3.4445\,x - 4.7750\,x^3 + 2.0315\,x^5$$

has slope 3.44 at 0 → five iterations suffice, in `bfloat16`, all matmuls.

@muon-orthogonalization-by-newton-schulz-1
:::

::: {.slide title="Watching the spectrum collapse"}
@!muon-orthogonalization-by-newton-schulz-2

- 0 iterations: an order of magnitude of spread.
- 5 iterations: a band around 1 (~0.7–1.2) — $\mathbf{U}\mathbf{V}^\top$
  for an optimizer's purposes, at the cost of 15 matmuls.
:::

::: {.slide title="Muon in fifteen lines"}
Momentum buffer → Newton–Schulz → shape-scaled step:

$$\mathbf{M}_t = \mu \mathbf{M}_{t-1} + \mathbf{G}_t, \qquad \mathbf{W}_{t+1} = \mathbf{W}_t - \eta\cdot 0.2\sqrt{\max(m,n)}\;\mathrm{NS}_5(\mathbf{M}_t)$$

$0.2\sqrt{\max(m,n)}$ sets every update's RMS to $0.2\,\eta$ — AdamW's
typical RMS, so AdamW-tuned $\eta$ transfers (Moonlight, 2025).

@muon-the-update
:::

::: {.slide title="Dividing the census"}
Hidden matrices → Muon; embeddings, head, vectors → AdamW
(the param-group pattern of §9.7):

@!muon-dividing-the-census-1

- 8 hidden matrices ≈ 95% of parameters.
- One buffer each vs. AdamW's two: state memory nearly halves.
:::

::: {.slide title="The race, same protocol as §9.6"}
Same init, 2,000 steps, constant lr, four-point grid each, weight decay off
— the only difference is the **direction**:

@!muon-the-race-on-the-language-model-3

- At matched tuning the hybrid **never lost** — carrying ~half the
  optimizer state.
- The margin ranged from *parity* (PyTorch) to *substantial* (JAX) under
  the identical protocol: small races are protocol-sensitive. Mechanism,
  not benchmark.
:::

::: {.slide title="On a CNN: two scoreboards"}
@!muon-the-same-race-on-a-cnn-3

- Training loss: the hybrid reaches the memorization regime much faster.
- Test accuracy: within about a point — optimizing faster ≠ predicting
  better on a small, saturated task.
- Same compression Adam-vs-SGD showed here (§9.6): the verdict depends on
  workload **and metric**.
:::

::: {.slide title="The family tree"}
- **K-FAC** (2015): layer-wise Fisher ≈ Kronecker product — two small inverses.
- **Shampoo** (2018): AdaGrad-style two-sided factors; won AlgoPerf's
  external-tuning track (~30% faster than tuned AdamW).
- **SOAP** (2024): Adam inside Shampoo's eigenbasis.
- **Muon = Shampoo without the memory**:

$$(\mathbf{G}\mathbf{G}^\top)^{-1/4}\mathbf{G}(\mathbf{G}^\top\mathbf{G})^{-1/4} = \mathbf{U}\mathbf{V}^\top.$$

- **Lion** (2023): the sign branch's lean member — one buffer, six lines
  (exercise).
:::

::: {.slide title="Adoption — and the honest accounting"}
**In production:** Moonlight (≈½ the compute of its AdamW baseline);
Kimi K2 — 15.5T tokens with MuonClip, zero loss spikes; GLM-4.5;
`torch.optim.Muon` in core.

. . .

**Fair tuning deflates headlines** (Wen et al., 2025): matrix methods are
genuinely fastest, but ~1.4× at 100M params, → ~1.1× at 1B. Sophia's 2×
did not replicate — reported by its own authors' group.

::: {.d2l-note}
AdamW is still the default. Muon is the one challenger with a clean
derivation **and** frontier mileage. Gains: tens of percent, not multiples.
:::
:::
