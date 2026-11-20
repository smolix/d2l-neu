# Scaling Up
:label:`sec_scaling`

Neural networks are developed at small scale and deployed at large scale. Because a multi-billion-parameter pretraining run cannot afford a grid search over learning rates, practitioners tune hyperparameters on a small **proxy model** of the same architectural family and transfer them to the target scale.

Under the **standard parametrization** (SP) used by default in deep learning frameworks—where layer weights are initialized with variance $1/\text{fan-in}$ and all layers share a single global learning rate $\eta$—the optimal learning rate *does not transfer across model width*. As layer width $n$ increases, a single Adam step changes hidden-layer pre-activations by $O(\eta n)$ because the $n$ terms in the matrix-vector product $\Delta\mathbf{W}\,\mathbf{h}$ align in sign with the incoming activations $\mathbf{h}$. Consequently, the optimal learning rate drifts downward as the model widens.

The **Maximal Update Parametrization** ($\mu\text{P}$, :citet:`Yang.Hu.2021,Yang.Hu.Babuschkin.ea.2022`) eliminates this drift by scaling the learning rate and output multiplier of each layer according to its fan-in and fan-out growth, ensuring that every hidden layer experiences $\Theta(1)$ feature updates as width $n \to \infty$. In this section, we measure the learning-rate drift under SP across a $64\times$ range of MLP widths ($n = 128$ to $8{,}192$), derive the first-step scaling rule, implement $\mu\text{P}$ and verify it with the one-step **coordinate check** :cite:`Dey.Anthony.Hestness.2024`, demonstrate zero-shot learning-rate transfer across widths, and extend the rules to Transformers and spectral-norm optimizers.

```{.python .input #scaling-scaling-up}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #scaling-scaling-up}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #scaling-scaling-up-1}
%%tab pytorch, jax
import random
```

## Learning Rate and Width Under Standard Parametrization

### A Family of Widths and Learning-Rate Drift
:label:`subsec_scaling-sweep`

Consider a three-hidden-layer ReLU MLP on Fashion-MNIST whose hidden width $n$ varies across $\{128, 512, 2048, 8192\}$ (from $1.35 \times 10^5$ to $1.41 \times 10^8$ parameters). Note how each weight matrix scales with $n$:
* **Input layer (`fc_in`, $784 \to n$):** fan-in is fixed ($784$), fan-out grows with $n$.
* **Hidden layers (`fc_h1`, `fc_h2`, $n \to n$):** both fan-in and fan-out grow with $n$.
* **Output readout (`fc_out`, $n \to 10$):** fan-in grows with $n$, fan-out is fixed ($10$).

Under standard parametrization (SP), every weight matrix is initialized from $\mathcal{N}(0, 1/\text{fan-in})$, biases are initialized to zero, and all parameters share a single Adam learning rate $\eta$:

```{.python .input #scaling-a-family-of-widths-1}
%%tab pytorch
fashion = d2l.FashionMNIST()
device = d2l.try_gpu()
X = (fashion.train.data.float().reshape(-1, 784) / 255).to(device)
Y = fashion.train.targets.to(device)
```

```{.python .input #scaling-a-family-of-widths-1}
%%tab jax
fashion = d2l.FashionMNIST()
X = jnp.asarray(fashion.train[0], dtype=jnp.float32).reshape(-1, 784) / 255
Y = jnp.asarray(fashion.train[1], dtype=jnp.int32)
```

```{.python .input #scaling-a-family-of-widths-2}
%%tab pytorch
class MLP(nn.Module):
    """A three-hidden-layer ReLU network under standard parametrization."""
    def __init__(self, width):
        super().__init__()
        self.fc_in = nn.Linear(784, width)
        self.fc_h1 = nn.Linear(width, width)
        self.fc_h2 = nn.Linear(width, width)
        self.fc_out = nn.Linear(width, 10)
        for lin in (self.fc_in, self.fc_h1, self.fc_h2, self.fc_out):
            nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)
            nn.init.zeros_(lin.bias)

    def features(self, X):
        h = F.relu(self.fc_h1(F.relu(self.fc_in(X))))
        return F.relu(self.fc_h2(h))

    def forward(self, X):
        return self.fc_out(self.features(X))

    def configure_adam(self, lr):
        return torch.optim.Adam(self.parameters(), lr)
```

```{.python .input #scaling-a-family-of-widths-2}
%%tab jax
class MLP(nnx.Module):
    """A three-hidden-layer ReLU network under standard parametrization."""
    def __init__(self, width, *, rngs):
        init = nnx.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        self.fc_in = nnx.Linear(784, width, kernel_init=init, rngs=rngs)
        self.fc_h1 = nnx.Linear(width, width, kernel_init=init, rngs=rngs)
        self.fc_h2 = nnx.Linear(width, width, kernel_init=init, rngs=rngs)
        self.fc_out = nnx.Linear(width, 10, kernel_init=init, rngs=rngs)

    def features(self, X):
        h = nnx.relu(self.fc_h1(nnx.relu(self.fc_in(X))))
        return nnx.relu(self.fc_h2(h))

    def __call__(self, X):
        return self.fc_out(self.features(X))

    def configure_adam(self, lr):
        return nnx.Optimizer(self, optax.adam(lr), wrt=nnx.Param)
```

We train each width for $400$ steps of Adam ($b = 512$) across a $10$-point log-spaced learning-rate grid $\eta \in \{2^{-14}, \ldots, 2^{-5}\}$ over two seeds, recording the mean training loss over the final $50$ steps (capped at $2.5$ for diverged runs) and the log–log slope of $\eta^\star(n)$ versus $n$:

```{.python .input #scaling-a-family-of-widths-3}
%%tab pytorch
def train_step(model, optimizer, Xb, Yb):
    loss = F.cross_entropy(model(Xb), Yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()

def train_mlp(arch, width, lr, seed, num_steps=400, batch_size=512):
    torch.manual_seed(seed)
    model = arch(width).to(device)
    optimizer = model.configure_adam(lr)
    idx = torch.as_tensor(np.random.default_rng(seed).integers(
        0, X.shape[0], (num_steps, batch_size)), device=device)
    losses = torch.stack([train_step(model, optimizer, X[i], Y[i])
                          for i in idx])
    v = float(losses[-50:].mean())  # Mean minibatch loss of the last 50 steps
    return min(v, 2.5) if math.isfinite(v) else 2.5
```

```{.python .input #scaling-a-family-of-widths-3}
%%tab jax
@nnx.jit
def train_step(model, optimizer, Xb, Yb):
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(Xb), Yb).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def train_mlp(arch, width, lr, seed, num_steps=400, batch_size=512):
    model = arch(width, rngs=nnx.Rngs(seed))
    step = nnx.cached_partial(train_step, model, model.configure_adam(lr))
    idx = jnp.asarray(np.random.default_rng(seed).integers(
        0, X.shape[0], (num_steps, batch_size)))
    losses = jnp.stack([step(X[i], Y[i]) for i in idx])
    v = float(losses[-50:].mean())  # Mean minibatch loss of the last 50 steps
    return min(v, 2.5) if math.isfinite(v) else 2.5
```

```{.python .input #scaling-learning-rate-sweep-1}
%%tab pytorch, jax
def sweep(arch, widths, lrs, seeds):
    """Score of every width and learning rate, one column per seed."""
    return {w: np.array([[train_mlp(arch, w, lr, s) for s in seeds]
                         for lr in lrs]) for w in widths}

def print_optimum(loss, lrs):
    """Best rate of the seed-averaged loss per width; slope in log-log."""
    widths = list(loss)
    best = [lrs[loss[w].mean(1).argmin()] for w in widths]
    slope = np.polyfit(np.log(widths), np.log(best), 1)[0]
    print('best lr:', '  '.join(f'{lr:.1e}' for lr in best),
          f'   slope {slope:+.2f}')

def plot_sweep(loss, lrs):
    d2l.plot(lrs, [l.mean(1) for l in loss.values()], 'learning rate',
             'training loss', xscale='log', ylim=[0.2, 0.8],
             legend=[f'width {w}' for w in loss])
```

```{.python .input #scaling-the-sweep}
%%tab pytorch, jax
widths = [128, 512, 2048, 8192]
lrs = [2.0 ** k for k in range(-14, -4)]
seeds = [random.getrandbits(32) for _ in range(2)]  # Shared by every sweep
sp_loss = sweep(MLP, widths, lrs, seeds)
print_optimum(sp_loss, lrs)
plot_sweep(sp_loss, lrs)
```

Under standard parametrization, the U-shaped loss curves shift steadily to the left as width grows: the optimal learning rate drops $16\times$ (from $\sim 3.9 \times 10^{-3}$ at width $128$ to $\sim 2.4 \times 10^{-4}$ at width $8{,}192$, a log–log slope of $\approx -0.7$). Applying the width-$128$ optimal learning rate directly to the width-$8{,}192$ model pushes it far up the unstable right arm of its curve, wiping out the benefit of the $1{,}000\times$ larger parameter count.

### Why the Optimal Learning Rate Drifts: The First Adam Step

To see why $\eta^\star$ must fall with width $n$, consider a square hidden weight matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$ mapping activation vector $\mathbf{h} \in \mathbb{R}^n$ to $\mathbf{z} = \mathbf{W}\mathbf{h}$. Because $W_{ij} \sim \mathcal{N}(0, 1/n)$ is independent of $\mathbf{h}$ at initialization, each coordinate $z_i$ has variance $\Theta(1)$ independent of $n$.

For a single example with backpropagated output gradient $\boldsymbol{\delta} = \partial \ell / \partial \mathbf{z}$, the weight gradient is the outer product $\mathbf{G} = \boldsymbol{\delta}\mathbf{h}^\top$. On step $1$, Adam's bias-corrected update is $\Delta\mathbf{W} = -\eta\,\operatorname{sign}(\mathbf{G}) = -\eta\,\operatorname{sign}(\boldsymbol{\delta})\operatorname{sign}(\mathbf{h})^\top$. Evaluated on the same activation vector $\mathbf{h}$, the resulting shift in pre-activation $z_i$ is

$$
(\Delta\mathbf{W}\, \mathbf{h})_i
= -\eta \sum_{j=1}^{n} \operatorname{sign}(\delta_i h_j)\, h_j
= -\eta\, \operatorname{sign}(\delta_i) \sum_{j=1}^{n} |h_j|
= -\eta\, \operatorname{sign}(\delta_i)\, n\, \overline{|h|},
$$
:eqlabel:`eq_scaling_first_step`

where $\overline{|h|} = \frac{1}{n}\sum_{j=1}^n |h_j| = \Theta(1)$. Because $\Delta W_{ij}$ is correlated with $h_j$, all $n$ terms in the sum share the same sign and add coherently ($O(n)$) rather than cancelling like random-sign initialization terms ($O(\sqrt{n})$). To keep the feature update $(\Delta\mathbf{W}\,\mathbf{h})_i = \Theta(1)$ invariant to width $n$, the learning rate of a hidden matrix with fan-in $n$ must scale as **$\eta \propto 1/n$** (log–log slope $-1$; the empirical slope $-0.7$ on the full MLP reflects a compromise between the $n \times n$ hidden matrices and the fixed-fan-in input layer and biases).

## Maximal Update Parametrization ($\mu\text{P}$)

### The $\mu\text{P}$ Scaling Rules

Rather than forcing a single global learning rate across layers with different fan-in scaling, **Maximal Update Parametrization** ($\mu\text{P}$, :citet:`Yang.Hu.2021,Yang.Hu.Babuschkin.ea.2022`) scales each layer relative to a reference **base width** $n_{\text{base}}$ via the **width multiplier** $m = n / n_{\text{base}}$ (:numref:`tab_mup-rules`):
1. **Input weights and 1D biases/norms (fixed fan-in):** Keep initial variance $1/\text{fan-in}$ and learning rate $\eta$ unchanged.
2. **Hidden matrices (fan-in and fan-out $\propto n$):** Keep initial variance $1/\text{fan-in} = 1/n$ and scale the Adam learning rate as **$\eta / m$**, so that $(\eta / m)\,n\,\overline{|h|} = \eta\,n_{\text{base}}\,\overline{|h|} = \Theta(1)$ at every width.
3. **Output readout matrix (fan-in $\propto n$, fixed fan-out):** Because both $\Delta\mathbf{W}_{\text{out}}\,\mathbf{h}$ and $\mathbf{W}_{\text{out}}\,\Delta\mathbf{h}$ sum $n$ correlated terms into the output logits, $\mathbf{W}_{\text{out}}$ must have entry magnitude $O(1/n)$ rather than $O(1/\sqrt{n})$. Fixing the unscaled weight initialization variance at $1/n_{\text{base}}$, multiplying the output logits by **$1/m$** in the forward pass, and keeping the Adam learning rate at $\eta$ ensures that both the effective output weights and their updates scale as $O(1/n)$, producing $\Theta(1)$ logit updates at every width.

:Maximal Update Parametrization ($\mu\text{P}$) rules for Adam relative to base width $n_{\textrm{base}}$ with width multiplier $m = n/n_{\textrm{base}}$.
:label:`tab_mup-rules`

| Parameter group | Initial variance | Adam learning rate | Forward multiplier |
|:--|:--|:--|:--|
| Input weights & all biases | $1/\textrm{fan-in}$ (biases $0$) | $\eta$ | $1$ |
| Hidden matrices ($n \times n$) | $1/\textrm{fan-in} = 1/n$ | $\eta / m$ | $1$ |
| Output readout ($n \times d_{\text{out}}$) | $1/n_{\textrm{base}}$ | $\eta$ | $1/m$ |

```{.python .input #scaling-the-rules}
%%tab pytorch
class MuMLP(MLP):
    """The same network under muP, relative to a width-8,192 base."""
    def __init__(self, width, base_width=8192):
        super().__init__(width)
        self.m = width / base_width
        # Rule 1: output init variance 1/base_width at every width
        with torch.no_grad():
            self.fc_out.weight.mul_(self.m ** 0.5)

    def forward(self, X):
        # Rule 2: scale the output matrix's logits; the bias is untouched
        return self.fc_out(self.features(X) / self.m)

    def configure_adam(self, lr):
        hidden = [self.fc_h1.weight, self.fc_h2.weight]
        rest = [p for p in self.parameters()
                if not any(p is q for q in hidden)]
        return torch.optim.Adam([
            {'params': rest, 'lr': lr},
            {'params': hidden, 'lr': lr / self.m}])  # Rule 3: hidden LR / m
```

```{.python .input #scaling-the-rules}
%%tab jax
class MuMLP(MLP):
    """The same network under muP, relative to a width-8,192 base."""
    def __init__(self, width, base_width=8192, *, rngs):
        super().__init__(width, rngs=rngs)
        self.m = width / base_width
        # Rule 1: output init variance 1/base_width at every width
        self.fc_out.kernel[...] *= self.m ** 0.5

    def __call__(self, X):
        # Rule 2: scale the output matrix's logits; the bias is untouched
        return self.fc_out(self.features(X) / self.m)

    def configure_adam(self, lr):
        def label(path, _):
            name = jax.tree_util.keystr(path)
            return 'hidden' if 'fc_h' in name and 'kernel' in name else 'rest'
        tx = optax.multi_transform(
            {'hidden': optax.adam(lr / self.m),  # Rule 3: hidden LR / m
             'rest': optax.adam(lr)},
            lambda params: jax.tree_util.tree_map_with_path(label, params))
        return nnx.Optimizer(self, tx, wrt=nnx.Param)
```

*(Setting `base_width=8192` makes `MuMLP` identical to `MLP` at the largest width $n = 8{,}192$, allowing direct comparison with the SP sweep above; in production, `base_width` is set to the small proxy width and $m > 1$ scales up to the target model.)*

### The Coordinate Check and Learning-Rate Transfer
:label:`subsec_scaling-coord-check`

To verify a $\mu\text{P}$ implementation before launching a sweep, the **coordinate check** :cite:`Dey.Anthony.Hestness.2024` takes a single Adam step across widths $n \in \{128, 256, \ldots, 8192\}$ and plots the mean absolute activation $\overline{|h_\ell|}$ of each layer:

```{.python .input #scaling-the-coordinate-check-1}
%%tab pytorch
def activations(model, Xb):
    with torch.inference_mode():
        h1 = F.relu(model.fc_in(Xb))
        h2 = F.relu(model.fc_h1(h1))
        h3 = F.relu(model.fc_h2(h2))
        return [float(h.abs().mean()) for h in (h1, h2, h3, model(Xb))]

def coord_check(arch, widths, lr, seed, num_steps=1):
    Xb, Yb = X[:256], Y[:256]
    sizes = []
    for width in widths:
        torch.manual_seed(seed)
        model = arch(width).to(device)
        optimizer = model.configure_adam(lr)
        for _ in range(num_steps):
            train_step(model, optimizer, Xb, Yb)
        sizes.append(activations(model, Xb))
    return list(zip(*sizes))
```

```{.python .input #scaling-the-coordinate-check-1}
%%tab jax
def activations(model, Xb):
    h1 = nnx.relu(model.fc_in(Xb))
    h2 = nnx.relu(model.fc_h1(h1))
    h3 = nnx.relu(model.fc_h2(h2))
    return [float(jnp.abs(h).mean()) for h in (h1, h2, h3, model(Xb))]

def coord_check(arch, widths, lr, seed, num_steps=1):
    Xb, Yb = X[:256], Y[:256]
    sizes = []
    for width in widths:
        model = arch(width, rngs=nnx.Rngs(seed))
        optimizer = model.configure_adam(lr)
        for _ in range(num_steps):
            train_step(model, optimizer, Xb, Yb)
        sizes.append(activations(model, Xb))
    return list(zip(*sizes))
```

```{.python .input #scaling-the-coordinate-check-2}
%%tab pytorch, jax
check_widths = [128 * 2 ** k for k in range(7)]  # 128 to 8,192
lr_128 = lrs[sp_loss[128].mean(1).argmin()]  # SP's best rate at width 128
sp_acts = coord_check(MLP, check_widths, lr_128, seeds[0])
d2l.plot(check_widths, list(sp_acts), 'width', 'mean |activation|',
         xscale='log', yscale='log', ylim=[0.05, 1000],
         legend=['layer 1', 'layer 2', 'layer 3', 'logits'])
```

```{.python .input #scaling-the-coordinate-check-3}
%%tab pytorch, jax
lr_base = lrs[sp_loss[8192].mean(1).argmin()]  # Best at the base width
mup_acts = coord_check(MuMLP, check_widths, lr_base, seeds[0])
d2l.plot(check_widths, list(mup_acts), 'width', 'mean |activation|',
         xscale='log', yscale='log', ylim=[0.05, 1000],
         legend=['layer 1', 'layer 2', 'layer 3', 'logits'])
```

Under SP (`MLP`), layer $2$ grows as $O(n)$, layer $3$ compounds to $O(n^2)$, and the output logits explode as $O(n^3)$ (increasing $1{,}000\times$ from width $128$ to width $8{,}192$ after a single step). Under $\mu\text{P}$ (`MuMLP`), every layer's activation and logit scale remains flat across all seven widths.

Repeating the $400$-step learning-rate sweep under $\mu\text{P}$ confirms that the optimal learning rate aligns across the entire $64\times$ width range:

```{.python .input #scaling-learning-rate-transfer}
%%tab pytorch, jax
mup_loss = sweep(MuMLP, widths[:-1], lrs, seeds)
mup_loss[widths[-1]] = sp_loss[widths[-1]]  # At the base width muP is SP
print_optimum(mup_loss, lrs)
plot_sweep(mup_loss, lrs)
```

```{.python .input #scaling-learning-rate-transfer-1}
%%tab pytorch, jax
for name, loss in (('SP', sp_loss), ('muP', mup_loss)):
    mean = {w: l.mean(1) for w, l in loss.items()}
    print(f'{name:3s} best loss per width:',
          '  '.join(f'{m.min():.3f}' for m in mean.values()))
    small, wide = widths[0], widths[-1]
    tuned, own = mean[small].argmin(), mean[wide].argmin()
    for label, i in ((f'the width-{small} lr', tuned), ('its own lr', own)):
        runs = '  '.join(f'{v:.3f}' for v in loss[wide][i])
        print(f'    width {wide} at {label:16s} {mean[wide][i]:.3f}'
              f'  (seeds {runs})')
```

Under $\mu\text{P}$, the log–log slope of $\eta^\star(n)$ collapses from $-0.7$ to $\approx 0$: transferring the learning rate tuned at width $128$ directly to width $8{,}192$ incurs $\le 0.003$ nats of suboptimality (well below seed noise), compared to a $0.10\text{--}0.15$-nat penalty under SP.

## Extensions to Transformers and Spectral Optimizers

### $\mu\text{P}$ Rules for a Transformer

Applying $\mu\text{P}$ to a Transformer such as `TinyLM` (:numref:`subsec_tinylm`) with model dimension $d$ and width multiplier $m = d / d_{\text{base}}$ follows the exact same fan-in/fan-out classification, with one additional rule for attention logits (:numref:`tab_mup-transformer`): because learned query and key vectors align during training, their dot product $\mathbf{q}^\top\mathbf{k}$ over head dimension $d_k \propto d$ grows as $O(d_k)$, requiring attention scaling by $1/d_k$ (an extra factor of $1/\sqrt{m}$ relative to standard $1/\sqrt{d_k}$ scaling :cite:`Yang.Hu.Babuschkin.ea.2022`).

:Maximal Update Parametrization ($\mu\text{P}$) rules for a Transformer under Adam ($m = d/d_{\textrm{base}}$).
:label:`tab_mup-transformer`

| Component | Scaling reason | $\mu\text{P}$ rule |
|:--|:--|:--|
| Embeddings, 1D biases, LayerNorm scales/shifts | Fixed fan-in (lookup or elementwise) | Learning rate $\eta$ |
| Internal block matrices (`qkv`, `proj`, `mlp1`, `mlp2`) | Both fan-in and fan-out $\propto d$ | Learning rate $\eta/m$, init variance $1/\text{fan-in}$ |
| Output readout `head` | Fan-in $\propto d$, fixed vocabulary size $V$ | Init variance $1/d_{\text{base}}$, logits $\times 1/m$, learning rate $\eta$ |
| Attention dot-product scores | Inner product of aligned $d_k$-vectors | Scale scores by $1/d_k$ (extra factor $1/\sqrt{m}$) |

### The Spectral Scaling Condition and Muon

As :citet:`Yang.Simon.Bernstein.2023` show, all $\mu\text{P}$ rules can be unified under a single operator-norm principle: for a layer mapping $\mathbf{h} \in \mathbb{R}^{d_{\text{in}}}$ with $\|\mathbf{h}\|_2 = \Theta(\sqrt{d_{\text{in}}})$ to $\mathbf{z} \in \mathbb{R}^{d_{\text{out}}}$ with $\|\mathbf{z}\|_2 = \Theta(\sqrt{d_{\text{out}}})$, preserving $\Theta(1)$ entrywise activations and updates requires both the weight matrix $\mathbf{W}$ and its update $\Delta\mathbf{W}$ to have spectral norm

$$
\|\Delta\mathbf{W}\|_2 = \Theta\!\left(\sqrt{\frac{d_{\textrm{out}}}{d_{\textrm{in}}}}\right).
$$
:eqlabel:`eq_spectral_condition`

For a square $n \times n$ hidden matrix, :eqref:`eq_spectral_condition` requires $\|\Delta\mathbf{W}\|_2 = \Theta(1)$. Under Adam, a rank-$1$ sign update has $\|\Delta\mathbf{W}\|_2 = \Theta(\eta n)$, requiring $\eta \propto 1/n$. Under Muon (:numref:`sec_muon`), the orthogonalized update $\mathrm{NS}_5(\cdot)$ has unit spectral norm by construction; if scaled by $\sqrt{d_{\text{out}}/d_{\text{in}}}$ :cite:`Bernstein.2025`, its learning rate transfers across widths directly with no $1/m$ correction (whereas the AdamW-matched factor $0.2\sqrt{\max(d_{\text{out}}, d_{\text{in}})}$ of :eqref:`eq_muon-update` has spectral norm $\Theta(\eta\sqrt{n})$ and therefore scales as $\eta \propto 1/\sqrt{n}$; Exercise :exref:`ex_scaling-spectral`).

## Summary

* Under standard parametrization (SP), a single Adam step changes the output of a hidden layer with fan-in $n$ by $\Theta(\eta n)$, causing the optimal learning rate $\eta^\star(n)$ to drift downward as the network widens.
* **Maximal Update Parametrization ($\mu\text{P}$)** scales hidden-matrix learning rates by $1/m$ (where $m = n/n_{\text{base}}$), fixes output-layer initialization variance at $1/n_{\text{base}}$ while scaling logits by $1/m$, and scales Transformer attention scores by $1/d_k$, achieving $\Theta(1)$ feature updates and zero-shot learning-rate transfer across widths.
* The one-step **coordinate check** verifies across a geometric progression of widths that layer activation magnitudes remain flat after an optimizer step.

## Exercises

1. [code] **Diagnosing broken $\mu\text{P}$ rules with the coordinate check.** Set `base_width=128` in `MuMLP` and use `coord_check` at `lr_128` to test three deliberate bugs one at a time: (a) omitting the $1/m$ logit multiplier, (b) omitting the $\eta/m$ hidden learning-rate scaling, and (c) scaling the hidden learning rate by $\eta/m^2$. Identify which errors `coord_check` catches immediately and why over-shrinking updates ($\eta/m^2$) requires a multi-step loss sweep to detect.
   :exlabel:`ex_scaling-breaking-mup`

1. [code] **Effect of the reference base width.** Rerun the $\mu\text{P}$ sweep with `base_width=128` across widths $\{128, 512, 2048, 8192\}$, verify that the log–log slope of $\eta^\star(n)$ remains near zero, and identify which constant ratio (input-to-hidden learning-rate ratio vs. initial logit scale) accounts for the loss difference at width $8{,}192$.
   :exlabel:`ex_scaling-base`

1. **Spectral norm condition across layers and optimizers.** ● Using :eqref:`eq_spectral_condition`, verify that the $\mu\text{P}$ rules of :numref:`tab_mup-rules` give input, hidden, and output layers updates of spectral norm $\Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})$ under a rank-$1$ Adam step, and derive the width scaling of $\eta$ required for Muon :eqref:`eq_muon-update` on the four block matrices of `TinyLM`.
   :exlabel:`ex_scaling-spectral`

1. [code] **Implementing $\mu\text{P}$ and the coordinate check on `TinyLM`.** ● Apply :numref:`tab_mup-transformer` to `d2l.TinyLM` across `d_model` $\in \{64, 128, 256, 512, 1024\}$ (`num_heads=2`, `base_width=64`) and verify with a one-step coordinate check at $\eta = 3 \times 10^{-3}$ that all sublayer output and logit magnitudes remain flat with width.
   :exlabel:`ex_scaling-tinylm`

[Discussions](https://d2l.discourse.group/t/36766)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.11]{.kicker}

Scaling Up<br>
**the drifting optimum · the first Adam step · muP · the coordinate check · transfer across widths**
:::
:::

::: {.slide title="A Learning Rate Tuned at Width 128 Wastes a Network a Thousand Times Larger"}
[The problem]{.kicker}

MLP on Fashion-MNIST, 400 Adam steps, training loss:

| network | learning rate | loss |
|:--|:--|--:|
| width 128, 135 thousand parameters | its own best | 0.32 to 0.33 |
| width 8,192, 141 million parameters | the one tuned at width 128 | 0.31 to 0.38, by seed |
| width 8,192 | its own best | 0.22 to 0.24, by seed |

::: {.d2l-note .warn}
A large run allows no sweep. Its learning rate comes from a small proxy,
and by default that rate is wrong.
:::

With width-aware rules, a 6.7-billion-parameter GPT-3 was tuned on a
40-million-parameter proxy for about 7% of its pretraining compute.

[Yang et al., 2022]{.small}
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- explain why the best learning rate falls with width;
- state the muP rules for Adam and implement them;
- run a coordinate check and say what it cannot detect;
- read a sweep for its optimum and for the cost of transfer;
- compute the spectral norm of a first update.

**Idea:** a layer with $n$ inputs sums $n$ terms of one sign in every
update; per-layer rules divide the width out.

**Running example:** an MLP $784 \to n \to n \to n \to 10$ on
Fashion-MNIST, $n = 128$ to $8{,}192$, 400 Adam steps, two seeds.
:::

::: {.slide title="Scaling Up Moves the Learning Rate with the Model"}
[Where this fits]{.kicker}

| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · AdamW · Muon |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · **Scaling Up** · Practice |

- **Uses:** Adam's first step is $\eta$ times the sign of the gradient;
  the spectral norm and Muon's update; AdamW's settled relative step.
- **Skips:** growth in depth, batch size and training length; the
  exponents for SGD.
:::

::: {.slide title="Only the Two Middle Matrices Grow in Both Dimensions"}
[A family of widths]{.kicker}

MLP $784 \to n \to n \to n \to 10$, $n$ = 128, 512, 2,048, 8,192:

| matrix | shape | fan-in (inputs) | grows with $n$ |
|:--|:--|:--|:--|
| input | $n \times 784$ | 784 | outputs only |
| hidden (two) | $n \times n$ | $n$ | **inputs and outputs** |
| output | $10 \times n$ | $n$ | inputs only |

**Standard parametrization (SP):** weights of variance
$1/\textrm{fan-in}$, zero biases, one learning rate $\eta$ for all layers.

**Score of a run:** mean training loss of the last 50 of 400 steps.
:::

::: {.slide title="Under SP the Best Learning Rate Falls Sixteenfold over a 64-Fold Widening"}
[Learning-rate sweep]{.kicker}

Four widths, ten learning rates a factor of two apart, two seeds:

@!scaling-the-sweep

Slope of log best rate against log width: about $-0.7$. Each end of the
family, run at the other end's best rate, loses about 0.1 nats.
:::

::: {.slide title="Adam's First Step Adds n Terms of One Sign"}
[The first Adam step]{.kicker}

One example. Hidden matrix $\mathbf{W}$ ($n \times n$), input activations
$\mathbf{h}$, output $\mathbf{z} = \mathbf{W}\mathbf{h}$, backpropagated
error $\boldsymbol{\delta}$ at $\mathbf{z}$, learning rate $\eta$.

I. $\;$Gradient: $\mathbf{g} = \boldsymbol{\delta}\mathbf{h}^\top$

II. $\;$First Adam step:
$\Delta\mathbf{W} = -\eta\operatorname{sign}(\mathbf{g})
= -\eta\operatorname{sign}(\boldsymbol{\delta})\operatorname{sign}(\mathbf{h})^\top$

III. Its effect on $\mathbf{z}$:

$$\boxed{(\Delta\mathbf{W}\mathbf{h})_i = -\eta \operatorname{sign}(\delta_i)
\sum_{j=1}^n |h_j| = -\eta \operatorname{sign}(\delta_i)\, n\,
\overline{|h|}}$$

$\overline{|h|}$: mean absolute activation, about the size of $z_i$
itself. One step changes $z_i$ by about $\eta n$ times its size.
:::

::: {.slide title="The Same η Is a Nudge at Width 128 and a Thirtyfold Change at Width 8,192"}
[The first Adam step]{.kicker}

Relative change of a hidden layer in one step, $\eta n$, at
$\eta = 2^{-8} \approx 0.004$:

| width $n$ | 128 | 512 | 2,048 | 8,192 |
|:--|--:|--:|--:|--:|
| $\eta n$ | 0.5 | 2 | 8 | 32 |

- Equal steps at every width need $\eta \propto 1/n$: a slope of $-1$.
- Measured: $-0.7$. The count is crude: one example, one step, and only
  the matrices with $n$ inputs.

::: {.d2l-note .rule}
One step moves a layer's output by about $\eta$ times its fan-in.
:::
:::

::: {.slide title="Check: Which Matrices Take Larger Steps When the Width Doubles?"}
[Check]{.kicker}

The width $n$ of the MLP doubles at a fixed learning rate. By what factor
does the first Adam step change the output of the input matrix
($n \times 784$), of a hidden matrix ($n \times n$), and of the output
matrix ($10 \times n$)?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** A step moves the output by about $\eta$ times the fan-in.
Input matrix: fan-in 784, no change. Hidden and output matrices: fan-in
$n$, twice as far.
:::
:::
:::

::: {.slide title="muP Writes the Width into Three Per-Layer Rules"}
[The rules]{.kicker}

Base width $n_{\textrm{base}}$, width multiplier
$m = n / n_{\textrm{base}}$, Adam:

| parameters | initial variance | learning rate | forward pass |
|:--|:--|:--|:--|
| input weights, all biases | $1/\textrm{fan-in}$; biases 0 | $\eta$ | unchanged |
| hidden matrices | $1/\textrm{fan-in}$ | $\eta/m$ | unchanged |
| output matrix | $1/n_{\textrm{base}}$ at every width | $\eta$ | logits $\times\, 1/m$ |

"Maximal update": the largest updates that do not grow with the width.
Smaller ones leave a wide layer where its initialization put it.

[Yang and Hu, 2021; Yang et al., 2022]{.small}
:::

::: {.slide title="The Hidden Rule Cancels the Sum, and the Output Layer Averages"}
[The rules]{.kicker}

- **Hidden matrices:** first step
  $(\eta/m)\, n\, \overline{|h|} = \eta\, n_{\textrm{base}}\, \overline{|h|}$
  at every width.
- **Output matrix, its own update:** logits
  $= \frac{1}{m}\mathbf{W}\mathbf{h}$, so every update acts $m$ times
  less.
- **Output matrix, updates of the features $\mathbf{h}$:** the error
  reached them through $\mathbf{W}$, so $\mathbf{W}\,\Delta\mathbf{h}$
  sums $n$ terms of one sign. It stays bounded only for weights of size
  $1/n$: variance $1/n_{\textrm{base}}$, times $1/m$.
- **Code:** output weights $\times \sqrt{m}$ once, features $\div\, m$,
  hidden rate $\eta/m$.

::: {.d2l-note}
A standard output layer sums over $n$ hidden units. Under muP it
averages, and the initial logits shrink as $1/\sqrt{m}$.
:::
:::

::: {.slide title="The Base Width Fixes the Constants That the Rules Leave Open"}
[The rules]{.kicker}

- At $n = n_{\textrm{base}}$, $m = 1$: muP is SP.
- The rules give exponents. The base gives constants: the input rate
  relative to the hidden rate, and the size of the initial logits.
- A project puts the base at its proxy. Here it is 8,192, the widest
  network: the same network under SP and muP, already tuned by the sweep.

| width | $m$ | hidden rate | initial logits, relative to SP |
|:--|--:|--:|--:|
| 8,192 | 1 | $\eta$ | 1 |
| 128 | 1/64 | $64\,\eta$ | 8 |
:::

::: {.slide title="One Adam Step under SP: Slopes of One, Two and Three"}
[The coordinate check]{.kicker}

Mean |activation| of each layer after one Adam step on one minibatch, at
the rate tuned for width 128:

@!scaling-the-coordinate-check-2

Layer 1 is flat. Each later layer multiplies the excess by a further
factor of order $\eta n$; the logits end about 1,000 times larger at width
8,192.
:::

::: {.slide title="Under muP All Four Curves Are Flat"}
[The coordinate check]{.kicker}

The same check, at the rate tuned for the base width. At width 128 the
hidden matrices learn at $64\eta$, four times the rate of the SP check:

@!scaling-the-coordinate-check-3

::: {.d2l-note .warn}
Growth with width convicts an implementation. Flat curves do not acquit
it: updates that shrink too fast with width are flat too.
:::
:::

::: {.slide title="Check: Which Curve Shows a Missing Logit Multiplier?"}
[Check]{.kicker}

An implementation of muP with $m \geq 1$ gives the hidden matrices
$\eta/m$ but forgets to divide the logits by $m$. What does the coordinate
check show?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The hidden layers stay flat. The output matrix still sums $n$
terms of one sign in its first step, so the logit curve grows in
proportion to the width.
:::
:::
:::

::: {.slide title="Under muP the Best Learning Rate Stays in One Place"}
[Learning-rate transfer]{.kicker}

The same sweep with the same seeds, under muP:

@!scaling-learning-rate-transfer

Best rate $2.4 \times 10^{-4}$ or one grid step below, at every width.
Slope within 0.2 of zero, against $-0.7$ under SP.
:::

::: {.slide title="muP Leaves a Narrow Network Less Room above Its Optimum"}
[Learning-rate transfer]{.kicker}

A hidden weight has size about $1/\sqrt{n}$; Adam moves it by up to
$\eta/m$ per step. The two are equal at $\eta = \sqrt{n}/n_{\textrm{base}}$:

| width $n$ | 128 | 512 | 2,048 | 8,192 |
|:--|--:|--:|--:|--:|
| step = weight at $\eta$ = | $1.4 \times 10^{-3}$ | $2.8 \times 10^{-3}$ | $5.5 \times 10^{-3}$ | $1.1 \times 10^{-2}$ |
| loss passes 0.5 at $\eta$ = | $2.0 \times 10^{-3}$ | $3.9 \times 10^{-3}$ | $7.8 \times 10^{-3}$ | $1.6 \times 10^{-2}$ or above |

- The three narrower curves pass 0.5 at the first grid point above their
  rate, from about 0.4 one point below it.

::: {.d2l-note}
The safe way round: we tune narrow and run wide, and the wide network has
more room.
:::
:::

::: {.slide title="Transfer Costs a Tenth of a Nat under SP and Nothing under muP"}
[Learning-rate transfer]{.kicker}

Width 8,192 at the rate tuned for width 128, and at its own best rate:

@!scaling-learning-rate-transfer-1

- SP: either seed loses 0.09 to 0.15 nats; two seeds at one rate differ
  by at most 0.05.
- muP: at most 0.003 above its own best.
:::

::: {.slide title="For a Transformer, Ask Whether Inputs and Outputs Grow with the Width"}
[Rules for a transformer]{.kicker}

`TinyLM`, a small transformer language model of width $d$;
$m = d/d_{\textrm{base}}$; Adam:

| part | rule |
|:--|:--|
| embeddings, biases, normalization scales and shifts (no sum over the width) | learning rate $\eta$ |
| the four matrices of a block (inputs and outputs grow with $d$) | learning rate $\eta/m$ |
| output head ($d$ inputs, always 256 outputs) | initial variance fixed, logits $\times\, 1/m$ |
| attention scores (dot product of two learned vectors) | extra factor $1/\sqrt{m}$ |

Training aligns the two vectors, so their dot product grows as the
dimension, not as its square root.
:::

::: {.slide title="Adam's First Update Has Spectral Norm η√(k_out k_in)"}
[Muon and width]{.kicker}

Spectral norm $\|\Delta\mathbf{W}\|_2$: the largest factor by which
$\Delta\mathbf{W}$ stretches a vector.

I. $\;\Delta\mathbf{W} = -\eta\,\mathbf{s}\mathbf{t}^\top$,
$\mathbf{s} = \operatorname{sign}(\boldsymbol{\delta})$,
$\mathbf{t} = \operatorname{sign}(\mathbf{h})$: rank one

II. $\;\|\mathbf{s}\mathbf{t}^\top\|_2 = \|\mathbf{s}\|_2\,\|\mathbf{t}\|_2$

III. With $k_{\textrm{out}}$, $k_{\textrm{in}}$ nonzero entries of
$\boldsymbol{\delta}$, $\mathbf{h}$:

$$\boxed{\|\Delta\mathbf{W}\|_2 = \eta\sqrt{k_{\textrm{out}}\,k_{\textrm{in}}}}$$

ReLU, $n \times n$: half the entries are nonzero on either side, so the
norm is about $\eta n/2$. It grows with the width.
:::

::: {.slide title="Check: How Large Should the Spectral Norm of an Update Be?"}
[Check]{.kicker}

A matrix has $d_{\textrm{in}}$ inputs and $d_{\textrm{out}}$ outputs. An
input with entries of order one has length $\sqrt{d_{\textrm{in}}}$. How
large must $\|\Delta\mathbf{W}\|_2$ be for the update to change every
output entry by an amount of order one?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The change of the output must have length
$\sqrt{d_{\textrm{out}}}$, so
$\|\Delta\mathbf{W}\|_2 \approx \sqrt{d_{\textrm{out}}/d_{\textrm{in}}}$,
the *spectral condition*. For an $n \times n$ matrix this is a constant,
while Adam's $\eta n/2$ grows with $n$.
:::
:::

[Yang, Simon and Bernstein, 2023]{.small}
:::

::: {.slide title="Muon at AdamW's Entry Scale Still Grows as √n"}
[Muon and width]{.kicker}

Muon's update: the averaged gradient with its singular values pushed to
about one, times a shape factor ($d_{\textrm{in}}$ inputs,
$d_{\textrm{out}}$ outputs). For an $n \times n$ matrix:

| update | spectral norm | $\eta$ must scale as |
|:--|:--|:--|
| Adam, first step | $\eta n/2$ | $1/n$ |
| Muon $\times\; 0.2\sqrt{\max(d_{\textrm{out}}, d_{\textrm{in}})}$, entries as large as AdamW's | $0.2\,\eta\sqrt{n}$ | $1/\sqrt{n}$ |
| Muon $\times \sqrt{d_{\textrm{out}}/d_{\textrm{in}}}$ | $\eta$ | constant |

[Bernstein, 2025; Shah et al., 2025 combine Muon with muP]{.small}
:::

::: {.slide title="Check the Family, Move the Rate with the Width, Sweep Once More"}
[Limits of transfer]{.kicker}

- Coordinate check on every family you scale: seconds.
- Never carry $\eta$ across a large change of width unchanged: per-layer
  rules, or a fitted slope under SP ($-0.7$ here; the first-step count
  gives $-1$).
- Sweep the transferred rate once more, a factor of two each way.

::: {.d2l-note .warn}
Open: under AdamW every matrix settles at a relative step
$\sqrt{2\eta\lambda}$ ($\lambda$: weight decay), whatever its width. muP
may then act mainly as a warmup.
:::

[Bi et al., 2024; Kosson et al., 2025]{.small}
:::

::: {.slide title="Check: By What Factor Should the Learning Rate Fall from Width 128 to 2,048?"}
[Check]{.kicker}

A network is widened from 128 to 2,048, so $m = 16$. Give the factor for
(a) the hidden matrices under muP with Adam, (b) Muon at AdamW's entry
scale, (c) the shared rate under SP, from the measured slope of $-0.7$.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** (a) $m = 16$. (b) $\sqrt{m} = 4$. (c) $16^{0.7} \approx 7$;
in the sweep the best rate fell from about $4 \times 10^{-3}$ to about
$5 \times 10^{-4}$, a factor of 8.
:::
:::
:::

::: {.slide title="Wider Layers Sum More Terms, and muP Divides the Sum Out"}
[Wrap-up]{.kicker}

- **Drift:** one Adam step moves a layer by about $\eta$ times its
  fan-in; under SP the best rate fell sixteenfold from width 128 to 8,192.
- **muP:** hidden rate $\eta/m$, logits $\times\, 1/m$; the best rate
  stays within one grid step.
- **Transfer:** the width-128 rate costs 0.1 nats at width 8,192 under
  SP, at most 0.003 under muP.
- **Coordinate check:** growth convicts, flat curves do not acquit.
- **Muon** at AdamW's entry scale: spectral norm $0.2\,\eta\sqrt{n}$, so
  its rate must fall as $1/\sqrt{n}$.

**Next:** configuring, protecting and finishing a run that cannot be
repeated.
:::
