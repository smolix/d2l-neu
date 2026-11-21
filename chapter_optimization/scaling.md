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

Under standard parametrization, the U-shaped loss curves shift steadily to the left as width grows: the optimal learning rate drops $16$–$32\times$ (from $\sim 3.9 \times 10^{-3}$ at width $128$ to $\sim 1$–$2 \times 10^{-4}$ at width $8{,}192$, a log–log slope of $-0.7$ to $-0.8$). Applying the width-$128$ optimal learning rate directly to the width-$8{,}192$ model pushes it far up the unstable right arm of its curve, wiping out the benefit of the $1{,}000\times$ larger parameter count.

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

where $\overline{|h|} = \frac{1}{n}\sum_{j=1}^n |h_j| = \Theta(1)$. Because $\Delta W_{ij}$ is correlated with $h_j$, all $n$ terms in the sum share the same sign and add coherently ($O(n)$) rather than cancelling like random-sign initialization terms ($O(\sqrt{n})$). To keep the feature update $(\Delta\mathbf{W}\,\mathbf{h})_i = \Theta(1)$ invariant to width $n$, the learning rate of a hidden matrix with fan-in $n$ must scale as **$\eta \propto 1/n$** (log–log slope $-1$; the empirical slope of $-0.7$ to $-0.8$ on the full MLP reflects a compromise between the $n \times n$ hidden matrices and the fixed-fan-in input layer and biases).

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

Under $\mu\text{P}$, the log–log slope of $\eta^\star(n)$ collapses from $-0.7$ or steeper under SP to near zero (the optimum moves by at most one grid step across the $64\times$ width range): transferring the learning rate tuned at width $128$ directly to width $8{,}192$ incurs $\le 0.003$ nats of suboptimality (well below seed noise), compared to a $0.10\text{--}0.15$-nat penalty under SP.

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
[Dive into Deep Learning · §9.9]{.kicker}

Width Parameterization and μP Scaling<br>**standard parametrization drift · coherent first Adam step · Maximal Update Parametrization (μP) · coordinate check · spectral scaling**
:::
:::

::: {.slide title="Why Proxy Learning Rates Fail as Networks Widen"}
[Motivation]{.kicker}

Because a multi-billion-parameter model cannot afford a learning-rate grid search, practitioners tune hyperparameters on a small **proxy model** and transfer them to the wide target network. Under default **Standard Parametrization (SP)**, that transferred learning rate fails:

| MLP Width $n$ ($784 \to n \to n \to n \to 10$) | Parameter Count | Learning Rate Policy ($400$ Adam Steps) | Final Training Loss |
| :--- | :---: | :--- | :---: |
| **Narrow proxy ($n = 128$)** | $135{,}000$ | Tuned at width $128$ ($\eta^\star \approx 3.9 \times 10^{-3}$) | $0.32\text{--}0.33$ |
| **Wide target ($n = 8{,}192$, $64\times$ wider)** | $141{,}000{,}000$ | **Transferred from width $128$ ($\eta = 3.9 \times 10^{-3}$)** | **$0.31\text{--}0.38$ (wipes out $1{,}000\times$ scale!)** |
| **Wide target ($n = 8{,}192$)** | $141{,}000{,}000$ | Retuned at width $8{,}192$ ($\eta^\star \approx 2.4 \times 10^{-4}$, $16\times$ smaller) | **$0.22\text{--}0.24$** |

::: {.d2l-note}
**Maximal Update Parametrization ($\mu\text{P}$, Yang & Hu, 2021; Yang et al., 2022)** scales each layer's learning rate and output multiplier by its fan-in/fan-out growth so that the optimal learning rate $\eta^\star$ stays invariant across widths—allowing GPT-3 6.7B to be tuned on a 40M proxy for $7\%$ of pretraining compute!
:::
:::

::: {.slide title="Fan-In and Fan-Out Growth Across Network Widths"}
[Layer Taxonomy]{.kicker}

::: {.cols .vc}
::: {.col}
In our 3-hidden-layer ReLU MLP $784 \to n \to n \to n \to 10$ across $n \in \{128, 512, 2048, 8192\}$, the four linear layers belong to three distinct scaling classes:

- **1. Input layer (`fc_in`, $784 \to n$):** Fan-in is fixed ($784$); only fan-out grows with $n$.
- **2. Hidden layers (`fc_h1`, `fc_h2`, $n \to n$):** **Both fan-in and fan-out grow with $n$**.
- **3. Output readout (`fc_out`, $n \to 10$):** **Fan-in grows with $n$**; fan-out is fixed ($10$).
- **Standard Parametrization (SP):** Initializes every weight matrix from $\mathcal{N}(0, 1/\text{fan-in})$, biases at $0$, and assigns a single global Adam learning rate $\eta$ to all layers.
:::

::: {.col .fig}
@-scaling-a-family-of-widths-2
:::
:::
:::

::: {.slide title="Under SP, Optimal Learning Rate Drifts Downward 16–32×"}
[Measuring SP Drift]{.kicker}

::: {.cols .vc}
::: {.col}
- Sweeping Adam across $10$ log-spaced learning rates $\eta \in \{2^{-14}, \ldots, 2^{-5}\}$ over $400$ steps ($b = 512$, $2$ seeds) for widths $n \in \{128, 512, 2048, 8192\}$:
  - As width $n$ increases $64\times$ ($128 \to 8{,}192$), the U-shaped loss curves shift steadily to the left!
  - The optimal learning rate falls **$16\text{--}32\times$** with a log–log slope of **$-0.7$ to $-0.8$**.
  - At width $8{,}192$, the width-$128$ optimum sits high up on the unstable right wall of the curve.
:::

::: {.col .fig}
@!scaling-the-sweep
:::
:::
:::

::: {.slide title="Why Initialization Is Θ(1) While Adam Step 1 Is Θ(ηn)"}
[Coherent vs. Incoherent Sums]{.kicker}

Why does standard $1/\text{fan-in}$ initialization keep forward activations $\mathbf{z} = \mathbf{W}\mathbf{h}$ at $\Theta(1)$ as width $n \to \infty$, yet the very first Adam update $\Delta\mathbf{W}$ blows up the activation change $\Delta\mathbf{z} = \Delta\mathbf{W}\,\mathbf{h}$ by $\Theta(\eta n)$?

- **At initialization (incoherent random signs $\to \Theta(1)$):** Because $W_{ij} \sim \mathcal{N}(0, 1/n)$ is independent of incoming activations $h_j = \Theta(1)$, the $n$ terms in $z_i = \sum_{j=1}^n W_{ij} h_j$ have random uncorrelated signs and sum in quadrature:
  $$\operatorname{Var}(z_i) = \sum_{j=1}^n \frac{1}{n}\,\mathbb{E}[h_j^2] = \Theta(1)$$
- **After one Adam step (coherent aligned signs $\to \Theta(\eta n)$):** For backpropagated gradient $\boldsymbol{\delta} = \partial\ell/\partial\mathbf{z}$, the weight gradient is $\mathbf{G} = \boldsymbol{\delta}\mathbf{h}^\top$, and Adam's first step is $\Delta\mathbf{W} = -\eta\,\operatorname{sign}(\mathbf{G}) = -\eta\,\operatorname{sign}(\boldsymbol{\delta})\operatorname{sign}(\mathbf{h})^\top$. Multiplying by $\mathbf{h}$ aligns every sign:
  $$(\Delta\mathbf{W}\,\mathbf{h})_i = -\eta\sum_{j=1}^n \operatorname{sign}(\delta_i h_j)\,h_j = -\eta\,\operatorname{sign}(\delta_i)\sum_{j=1}^n |h_j| = -\eta\,\operatorname{sign}(\delta_i)\,n\,\overline{|h|}$$
:::

::: {.slide title="The Fan-In Law: One Adam Step Shifts Outputs by Θ(ηn)"}
[Quantitative Scaling Breakdown]{.kicker}

Because all $n$ terms $|h_j| > 0$ add with the same sign, a single Adam step changes a hidden unit's pre-activation by $\Theta(\eta n)$ relative to its initial $\Theta(1)$ scale:

| Hidden Width $n$ | Fan-In of `fc_h1`, `fc_h2`, `fc_out` | First-Step Feature Shift $\eta n$ at $\eta = 2^{-8} \approx 0.0039$ | Regime at Fixed $\eta = 0.0039$ |
| :--- | :---: | :---: | :--- |
| **$n = 128$** | $128$ | $0.50\times$ activation scale | Stable feature refinement |
| **$n = 512$** | $512$ | $2.00\times$ activation scale | Moderately large step |
| **$n = 2{,}048$** | $2{,}048$ | $8.00\times$ activation scale | Overshoots initial representation |
| **$n = 8{,}192$** | $8{,}192$ | **$32.0\times$ activation scale!** | **Catastrophic layer blow-up** |

::: {.d2l-note .rule}
**Why hidden learning rates must scale as $1/n$:** Keeping $(\Delta\mathbf{W}\,\mathbf{h})_i = \Theta(1)$ invariant across widths requires scaling the Adam learning rate of every hidden matrix with fan-in $n$ as **$\eta_{\text{hidden}} \propto 1/n$** (log–log slope $-1$).
:::
:::

::: {.slide title="Maximal Update Parametrization (μP) Rules for Adam"}
[Three Per-Layer Rules]{.kicker}

Relative to a reference **base width** $n_{\text{base}}$ with **width multiplier** $m = n / n_{\text{base}}$, $\mu\text{P}$ (Yang et al., 2022) ensures every layer's activations and updates remain $\Theta(1)$ as $m \to \infty$:

| Parameter Group | Fan-In / Fan-Out vs. $n$ | Initial Weight Variance | Adam Learning Rate | Forward Pass Multiplier |
| :--- | :--- | :---: | :---: | :---: |
| **1. Input weights & all 1D biases** | Fixed fan-in ($d_{\text{in}}$) | $1 / \text{fan-in}$ (biases $0$) | $\eta$ | $1$ |
| **2. Hidden matrices ($n \times n$)** | Both fan-in & fan-out $\propto n$ | $1 / \text{fan-in} = 1 / n$ | **$\eta / m$** | $1$ |
| **3. Output readout ($n \times d_{\text{out}}$)** | Fan-in $\propto n$, fixed fan-out | **$1 / n_{\text{base}}$** | $\eta$ | **$1 / m$** |

::: {.d2l-note .rule}
**"Maximal" update principle:** Every hidden layer receives the largest possible update ($\Theta(1)$) that does not explode as $n \to \infty$; smaller updates ($o(1)$) freeze wide layers at their random initialization (the neural tangent kernel limit).
:::
:::

::: {.slide title="Why the Output Layer Must Average (1/n) and Base Width"}
[Output Scaling & Reference Anchor]{.kicker}

- **Why the output readout matrix $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d_{\text{out}} \times n}$ needs two changes:**
  1. **Its own weight update $\Delta\mathbf{W}_{\text{out}}\,\mathbf{h}$:** Sums $n$ sign-aligned terms into each logit; dividing the forward logits by **$m = n/n_{\text{base}}$** keeps $\frac{1}{m}\Delta\mathbf{W}_{\text{out}}\,\mathbf{h} = \Theta(1)$ at learning rate $\eta$.
  2. **Incoming feature updates $\mathbf{W}_{\text{out}}\,\Delta\mathbf{h}$:** Backpropagation aligns $\Delta\mathbf{h}$ with the rows of $\mathbf{W}_{\text{out}}$, so $\frac{1}{m}\mathbf{W}_{\text{out}}\,\Delta\mathbf{h}$ sums $n$ aligned terms of size $|\frac{1}{m}W_{\text{out},ij}|$. Keeping this sum $\Theta(1)$ requires effective weights of size $O(1/n)$, achieved by fixing the unscaled initialization variance at **$1/n_{\text{base}}$**!
- **Role of the base width $n_{\text{base}}$:** At $n = n_{\text{base}}$ ($m = 1$), $\mu\text{P}$ coincides identically with Standard Parametrization (SP), fixing the constant ratios between input and hidden learning rates.
:::

::: {.slide title="Implementing μP in 15 Lines of Code"}
[Subclassing Standard Parametrization]{.kicker}

::: {.cols .vc}
::: {.col}
- Setting `base_width=8192` ($m = 1$ at $n = 8{,}192$) makes `MuMLP` identical to `MLP` at the widest network so both sweeps share the exact same anchor at $n = 8{,}192$:
  - **Rule 1 (`__init__`):** Multiply `fc_out` weights (initialized at std $1/\sqrt{n}$) by $\sqrt{m} = \sqrt{n / n_{\text{base}}}$ so variance becomes $\frac{1}{n} \cdot \frac{n}{n_{\text{base}}} = \frac{1}{n_{\text{base}}}$.
  - **Rule 2 (`forward`):** Divide features entering `fc_out` by $m$.
  - **Rule 3 (`configure_adam`):** Set hidden matrix learning rate to `lr / self.m`.
:::

::: {.col .fig}
@-scaling-the-rules
:::
:::
:::

::: {.slide title="The One-Step Coordinate Check: SP Blows Up as O(n³)"}
[Verifying Scaling Before Training]{.kicker}

::: {.cols .vc}
::: {.col}
- The **coordinate check** (Dey et al., 2024) takes **1 Adam step** on a single minibatch across widths $n \in \{128, 256, \ldots, 8192\}$ and plots the mean absolute activation $\overline{|h_\ell|}$ of each layer.
- **Under Standard Parametrization (`MLP`):**
  - **Layer 1 (`fc_in`, fan-in $784$):** Flat across all widths ($\Theta(1)$).
  - **Layer 2 (`fc_h1`, fan-in $n$):** Grows linearly with width as **$O(n)$** (slope $+1$).
  - **Layer 3 (`fc_h2`):** Compounds Layer 2's input blow-up with its own $O(n)$ update to grow as **$O(n^2)$** (slope $+2$).
  - **Output logits (`fc_out`):** Explodes $1{,}000\times$ across the width range as **$O(n^3)$** (slope $+3$)!
:::

::: {.col .fig}
@!scaling-the-coordinate-check-2
:::
:::
:::

::: {.slide title="Under μP, Every Layer's Coordinate Check Is Flat"}
[Width-Invariant Feature Updates]{.kicker}

::: {.cols .vc}
::: {.col}
- Running the exact same 1-step coordinate check on `MuMLP` across $n \in \{128, 256, \ldots, 8192\}$ produces **four completely flat curves** ($\Theta(1)$ at every layer and every width)!
- **Diagnostic asymmetry to remember:**
  - Rising curves on a coordinate check immediately **convict** a broken $\mu\text{P}$ implementation (e.g., forgetting the $1/m$ logit scale or $\eta/m$ hidden LR).
  - Flat curves rule out *exploding* updates ($\gg \Theta(1)$), but an over-shrunk learning rate (such as $\eta/m^2$) also looks flat after 1 step because weights barely move from initialization.
:::

::: {.col .fig}
@!scaling-the-coordinate-check-3
:::
:::
:::

::: {.slide title="Zero-Shot Learning-Rate Transfer Across Widths Under μP"}
[Aligned Optima Across a 64× Width Range]{.kicker}

::: {.cols .vc}
::: {.col}
- Repeating the $400$-step learning-rate sweep across $n \in \{128, 512, 2048, 8192\}$ under $\mu\text{P}$:
  - The U-shaped curves now **align vertically** across the entire $64\times$ width range (135K to 141M parameters)!
  - The log–log slope of $\eta^\star(n)$ collapses from $-0.75$ under SP to **near $0.0$** (the optimum stays at $2.4 \times 10^{-4}$ or within one grid step at every width).
  - Wider networks achieve strictly lower training loss at the exact same shared learning rate.
:::

::: {.col .fig}
@!scaling-learning-rate-transfer
:::
:::
:::

::: {.slide title="Quantifying the Transfer Penalty: SP vs. μP"}
[Cost of Proxy-to-Target Transfer]{.kicker}

::: {.cols .vc}
::: {.col}
- How much loss do we pay at width $8{,}192$ when we use the learning rate tuned on the $1{,}000\times$ smaller width-$128$ proxy instead of retuning at width $8{,}192$?
  - **Under SP:** Loses **$0.09\text{--}0.15$ nats** ($0.343$ vs. $0.228$), worse than the narrow width-$128$ proxy itself!
  - **Under $\mu\text{P}$:** Loses **$\le 0.003$ nats** ($0.228\text{--}0.231$), completely indistinguishable from seed noise!
- Notice also that under $\mu\text{P}$, narrower proxy networks rise sooner on the right of $\eta^\star$ (where step size $\eta/m$ reaches initial weight size $1/\sqrt{n}$ at $\eta = \sqrt{n}/n_{\text{base}}$), so a rate stable on the narrow proxy is guaranteed to be stable on the wide target!
:::

::: {.col .fig}
@!scaling-learning-rate-transfer-1
:::
:::
:::

::: {.slide title="Extending μP to Transformers: The 1/d_k Attention Rule"}
[Transformer Parameterization]{.kicker}

Applying $\mu\text{P}$ to a Transformer of model width $d$ ($m = d / d_{\text{base}}$, head dimension $d_k \propto d$) follows the exact same fan-in/fan-out taxonomy, plus one rule for attention dot products:

| Transformer Component | Fan-In / Fan-Out Scaling | $\mu\text{P}$ Rule under Adam / AdamW |
| :--- | :--- | :--- |
| **Token & position embeddings** | One-hot lookup (fan-in $= 1$) | Init variance $1$, learning rate $\eta$ |
| **1D biases & LayerNorm gains/shifts** | Elementwise (fan-in $= 1$) | Learning rate $\eta$ |
| **Internal 2D block matrices (`qkv`, `proj`, `mlp1`, `mlp2`)** | Both fan-in and fan-out $\propto d$ | Init variance $1/\text{fan-in}$, learning rate **$\eta / m$** |
| **Vocabulary readout `head` ($d \to V$)** | Fan-in $\propto d$, fixed vocab $V$ | Init variance $1/d_{\text{base}}$, forward logits **$\times 1/m$**, learning rate $\eta$ |
| **Attention scores ($\mathbf{q}^\top\mathbf{k}$ over $d_k \propto d$)** | Inner product of two learned vectors | Scale by **$1 / d_k$** instead of $1/\sqrt{d_k}$ (extra factor $1/\sqrt{m}$) |

- **Why $1/d_k$ instead of $1/\sqrt{d_k}$ in attention?** At initialization $\mathbf{q}, \mathbf{k}$ are independent ($\mathbf{q}^\top\mathbf{k} = O(\sqrt{d_k})$), but after one gradient step $\Delta\mathbf{q}$ and $\Delta\mathbf{k}$ become aligned, making $\mathbf{q}^\top\mathbf{k} = \Theta(d_k)$!
:::

::: {.slide title="The Spectral Condition: ||ΔW||₂ = Θ(√(d_out/d_in))"}
[Operator-Norm Unification]{.kicker}

Yang, Simon & Bernstein (2023) showed that all $\mu\text{P}$ rules reduce to a single **spectral norm condition**: for a layer mapping $\mathbf{h} \in \mathbb{R}^{d_{\text{in}}}$ with $\|\mathbf{h}\|_2 = \Theta(\sqrt{d_{\text{in}}})$ to $\mathbf{z} = \mathbf{W}\mathbf{h} \in \mathbb{R}^{d_{\text{out}}}$ with $\|\mathbf{z}\|_2 = \Theta(\sqrt{d_{\text{out}}})$, keeping entrywise activations and updates $\Theta(1)$ requires:

$$\|\mathbf{W}\|_2 = \Theta\!\left(\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\right) \qquad \text{and} \qquad \|\Delta\mathbf{W}\|_2 = \Theta\!\left(\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\right)$$

- **Why Adam's first update has spectral norm $\Theta(\eta\sqrt{d_{\text{out}} d_{\text{in}}})$:**
  - On step 1, $\Delta\mathbf{W} = -\eta\,\mathbf{s}\mathbf{t}^\top$ is a rank-1 outer product of sign vectors $\mathbf{s} = \operatorname{sign}(\boldsymbol{\delta})$ and $\mathbf{t} = \operatorname{sign}(\mathbf{h})$ with $k_{\text{out}} = \Theta(d_{\text{out}})$ and $k_{\text{in}} = \Theta(d_{\text{in}})$ nonzero entries:
    $$\|\Delta\mathbf{W}\|_2 = \eta\,\|\mathbf{s}\|_2\,\|\mathbf{t}\|_2 = \eta\sqrt{k_{\text{out}}\,k_{\text{in}}} = \Theta\!\left(\eta\sqrt{d_{\text{out}}\,d_{\text{in}}}\right)$$
  - Matching $\Theta\!\left(\sqrt{d_{\text{out}}/d_{\text{in}}}\right)$ immediately requires **$\eta \propto 1/d_{\text{in}}$**!
:::

::: {.slide title="Width Scaling for Spectral Optimizers (Muon vs. AdamW)"}
[Comparing Update Spectral Norms]{.kicker}

For a square $n \times n$ hidden matrix, the spectral condition $\|\Delta\mathbf{W}\|_2 = \Theta(\sqrt{n/n}) = \Theta(1)$ reveals how $\eta$ must scale with width $n$ under different matrix optimizers:

| Optimizer Update on an $n \times n$ Hidden Matrix | Spectral Norm $\|\Delta\mathbf{W}\|_2$ of Update | Required $\eta$ Scaling with Width $n$ |
| :--- | :---: | :---: |
| **Adam step 1:** $-\eta\,\operatorname{sign}(\boldsymbol{\delta})\operatorname{sign}(\mathbf{h})^\top$ (rank $1$, $k \approx n/2$ nonzeros) | $\eta\sqrt{k_{\text{out}}k_{\text{in}}} \approx \frac{1}{2}\eta\,n = \Theta(\eta n)$ | **$\eta \propto 1 / n$** ($\mu\text{P}$ rule $\eta / m$) |
| **Muon (AdamW-RMS scale §9.6):** $-0.2\eta\sqrt{n}\,\mathrm{NS}_5(\cdot)$ | $0.2\,\eta\sqrt{n} = \Theta(\eta\sqrt{n})$ | **$\eta \propto 1 / \sqrt{n}$** ($\eta / \sqrt{m}$) |
| **Muon (spectral $\mu\text{P}$ scale, Bernstein 2025):** $-\eta\sqrt{d_{\text{out}}/d_{\text{in}}}\,\mathrm{NS}_5(\cdot)$ | $\eta = \Theta(\eta)$ | **$\eta = \Theta(1)$ (transfers with no $1/m$ factor!)** |

::: {.d2l-note}
Because $\mathrm{NS}_5(\cdot)$ already normalizes the spectral norm of the update to $\approx 1$, scaling Muon by $\sqrt{d_{\text{out}}/d_{\text{in}}}$ satisfies the spectral $\mu\text{P}$ condition automatically across all matrix shapes!
:::
:::

::: {.slide title="Concept Check: Coherent Sums, Coordinate Checks, and μP"}
[Check Your Understanding]{.kicker}

Test your grasp of width parameterization on three concrete problems:

1. **Doubling width under SP:** If you double the hidden width $n$ of our MLP at a fixed Adam learning rate $\eta$, by what factor does the first Adam step change the output of (a) the input matrix `fc_in` ($784 \to n$), (b) the hidden matrix `fc_h1` ($n \to n$), and (c) the readout matrix `fc_out` ($n \to 10$)?
2. **Reading a broken coordinate check:** You implement $\mu\text{P}$ with $m = n/n_{\text{base}} \ge 1$, remembering to set the hidden matrix learning rate to $\eta/m$ but forgetting to divide the output logits by $m$. What will the 4 curves of the 1-step coordinate check look like?
3. **Scaling down learning rates from $n = 128$ to $n = 2{,}048$ ($m = 16$):** By what factor must the hidden-matrix learning rate be divided when widening $16\times$ under (a) Adam with $\mu\text{P}$, (b) Muon with the $0.2\sqrt{\max(m,n)}$ factor, and (c) Muon with the $\sqrt{d_{\text{out}}/d_{\text{in}}}$ factor?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Step 1 shifts a layer's output by $\Theta(\eta \cdot \text{fan-in})$: (a) `fc_in` has fixed fan-in $784$ $\to$ **$1\times$ (unchanged)**; (b) `fc_h1` and (c) `fc_out` both have fan-in $n$ $\to$ **$2\times$ larger shift**.
2. Layers 1, 2, and 3 (`h1`, `h2`, `h3`) will be **completely flat ($\Theta(1)$)**, while the **output logit curve will grow linearly with width as $O(n)$ (slope $+1$)** because $\Delta\mathbf{W}_{\text{out}}\,\mathbf{h}$ sums $n$ aligned terms without the $1/m$ divisor!
3. For a $16\times$ widening ($m = 16$): (a) Adam $\mu\text{P}$ divides by **$m = 16\times$**; (b) AdamW-scaled Muon divides by **$\sqrt{m} = 4\times$**; (c) operator-scaled Muon divides by **$1\times$ (no change)**.
:::
:::

::: {.slide title="Summary: Width Parameterization and μP Scaling"}
[Key Takeaways]{.kicker}

| Concept / Rule | Mathematical Mechanism | Scaling Behavior vs. Width $n$ ($m = n/n_{\text{base}}$) | Practical Engineering Role |
| :--- | :--- | :--- | :--- |
| **SP First-Step Drift** | $(\Delta\mathbf{W}\,\mathbf{h})_i = -\eta\operatorname{sign}(\delta_i)\,n\,\overline{|h|}$ | Coherent sum of $n$ aligned signs grows as $\Theta(\eta n)$ | Causes $\eta^\star(n)$ to drift down $16\text{--}32\times$ from $n=128 \to 8{,}192$ |
| **$\mu\text{P}$ Hidden Layer Rule** | Init variance $1/n$, Adam LR $\eta / m$ | $(\eta/m)\,n\,\overline{|h|} = \eta\,n_{\text{base}}\,\overline{|h|} = \Theta(1)$ | Cancels the $O(n)$ fan-in sum on internal $n \times n$ matrices |
| **$\mu\text{P}$ Output Layer Rule** | Init variance $1/n_{\text{base}}$, forward logits $\times 1/m$, LR $\eta$ | Effective weights & updates scale as $O(1/n)$ | Keeps both $\Delta\mathbf{W}_{\text{out}}\mathbf{h}$ and $\mathbf{W}_{\text{out}}\Delta\mathbf{h}$ at $\Theta(1)$ |
| **1-Step Coordinate Check** | Plot $\overline{|h_\ell|}$ vs. $n \in \{128, \ldots, 8192\}$ after 1 step | SP explodes as $O(n), O(n^2), O(n^3)$; $\mu\text{P}$ is flat | Catches missing $\mu\text{P}$ multipliers in seconds before training |
| **Spectral Condition** | $\|\Delta\mathbf{W}\|_2 = \Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})$ | Unifies Adam ($\eta \propto 1/n$) and Muon ($\eta \propto 1/\sqrt{n}$ or $1$) | Enables zero-shot hyperparameter transfer from small proxies |
:::

