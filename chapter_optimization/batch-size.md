# Batch Size
:label:`sec_batch_size`

In practice, the minibatch size $b$ is often chosen to fill accelerator memory and then left untouched. Yet the moment we scale training across multiple GPUs—or trade wall-clock training time against total compute FLOPs—we face a fundamental question: if we double the minibatch size $b$ (doubling the parallel computation per step), does the run reach our target loss in half as many steps?

In :numref:`sec_sgd`, we saw that a minibatch of $b$ examples reduces the gradient sampling variance to $\operatorname{tr}(\boldsymbol{\Sigma})/b$. This implies two distinct regimes:
1. **Small-batch (noise-dominated) regime:** When sampling noise is larger than the true gradient, doubling $b$ halves the noise variance, allowing a larger learning rate that cuts the required step count $S(b)$ in half while keeping total examples processed $E(b) = b\,S(b)$ constant (*perfect linear scaling*).
2. **Large-batch (curvature-dominated) regime:** Once $b$ is large enough that the minibatch gradient closely approximates the true full-batch gradient, further increasing $b$ yields diminishing reductions in $S(b)$ while multiplying the data and compute cost $E(b)$ linearly.

:citet:`McCandlish.Kaplan.Amodei.ea.2018` showed that the transition between these two regimes occurs at the **critical batch size** $b_{\text{crit}}$, which is predicted by the **gradient-noise scale** $b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$. In this section, we build a two-batch estimator for $b_{\text{noise}}$, derive the step-versus-data trade-off frontier and the learning-rate scaling rules for SGD ($\eta \propto b$) and Adam ($\eta \propto \sqrt{b}$), and test both predictions on `TinyLM` and `FashionCNN`.

```{.python .input #batch-size}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #batch-size}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #batch-size-1}
%%tab pytorch
import random
```

## The Gradient-Noise Scale

### Signal, Noise, and a Two-Batch Estimator
:label:`subsec_batch-size-estimator`

Let $\nabla f(\mathbf{x})$ denote the exact full-dataset gradient and $\hat{\mathbf{g}}_b$ denote a minibatch gradient averaged over $b$ independent examples, with per-example gradient covariance $\boldsymbol{\Sigma}$. Because $\mathbb{E}[\hat{\mathbf{g}}_b] = \nabla f$ and $\mathbb{E}[\|\hat{\mathbf{g}}_b - \nabla f\|^2] = \operatorname{tr}(\boldsymbol{\Sigma})/b$, the expected squared norm of the minibatch gradient decomposes into signal power $\|\nabla f\|^2$ plus noise power $\operatorname{tr}(\boldsymbol{\Sigma})/b$:

$$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \|\nabla f\|^2 + \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$
:eqlabel:`eq_sqnorm-bias`

The signal power and noise power are equal at the **simple gradient-noise scale** :cite:`McCandlish.Kaplan.Amodei.ea.2018`:

$$b_{\textrm{noise}} = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{\|\nabla f\|^2}.$$
:eqlabel:`eq_noise-scale`

We can estimate both $\operatorname{tr}\boldsymbol{\Sigma}$ and $\|\nabla f\|^2$ without a full pass over the dataset by measuring the mean squared gradient norm $N_b = \mathbb{E}[\|\hat{\mathbf{g}}_b\|^2]$ at two batch sizes $b_{\text{s}} < b_{\text{l}}$ and solving the two-by-two linear system :eqref:`eq_sqnorm-bias`:

$$
\operatorname{tr} \boldsymbol{\Sigma} \approx
\frac{N_{b_{\textrm{s}}} - N_{b_{\textrm{l}}}}{1/b_{\textrm{s}} - 1/b_{\textrm{l}}},
\qquad
\|\nabla f\|^2 \approx
\frac{b_{\textrm{l}} N_{b_{\textrm{l}}} - b_{\textrm{s}} N_{b_{\textrm{s}}}}{b_{\textrm{l}} - b_{\textrm{s}}}.
$$

We save the cross-entropy loss helper `ce_loss`, squared gradient norm `grad_sq_norm`, and two-batch estimator `noise_scale` in `d2l`, and measure $b_{\text{noise}}$ on `TinyLM` at initialization and after $500$ steps of Adam ($\eta = 3 \times 10^{-3}$):

```{.python .input #batch-size-a-two-batch-estimator-1}
%%tab pytorch
device = d2l.try_gpu()
data = d2l.WikiText103(batch_size=64, num_steps=64)
fashion = d2l.FashionMNIST(batch_size=256)
X_f = (fashion.train.data.float() / 255).unsqueeze(1).to(device)
y_f = fashion.train.targets.to(device)

def sample_fashion(b):  # b random training images
    i = torch.randint(len(y_f), (b,), device=device)
    return X_f[i], y_f[i]

def sample_lm(b):  # b windows at random offsets in the training text
    starts = torch.randint(len(data.train) - data.num_steps, (b,))
    X, Y = data.windows(data.train, starts.numpy())
    return X.to(device), Y.to(device)
```

```{.python .input #batch-size-a-two-batch-estimator-1}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
fashion = d2l.FashionMNIST(batch_size=256)
X_f = jnp.asarray(fashion.train[0], jnp.float32)[..., None] / 255
y_f = jnp.asarray(fashion.train[1], jnp.int32)
rng = np.random.default_rng(d2l.get_seed())

def sample_fashion(b):  # b random training images
    i = rng.integers(len(y_f), size=b)
    return X_f[i], y_f[i]

def sample_lm(b):  # b windows at random offsets in the training text
    return data.windows(data.train,
                        rng.integers(len(data.train) - data.num_steps, size=b))
```

```{.python .input #batch-size-a-two-batch-estimator-2}
%%tab pytorch
def ce_loss(model, X, Y):  #@save
    """Mean cross-entropy of the logits model(X) for the labels Y."""
    logits = model(X)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           Y.reshape(-1))

def grad_sq_norm(model, X, Y):  #@save
    """Squared norm of the gradient of ce_loss over all parameters."""
    grads = torch.autograd.grad(ce_loss(model, X, Y),
                                list(model.parameters()))
    return sum((g ** 2).sum() for g in grads)

def noise_scale(model, sample, b_small=16, b_big=2048, m=400):  #@save
    """Gradient-noise scale from the mean squared gradient norms of m
    minibatches of size b_small and m // 8 of size b_big from sample(b)."""
    def mean_sq_norm(b, m):  # N_b, averaged over m random minibatches
        return torch.stack([grad_sq_norm(model, *sample(b))
                            for _ in range(m)]).mean()
    n_small, n_big = mean_sq_norm(b_small, m), mean_sq_norm(b_big, m // 8)
    tr_sigma = (n_small - n_big) / (1 / b_small - 1 / b_big)
    sq_norm = (b_big * n_big - b_small * n_small) / (b_big - b_small)
    return (tr_sigma / sq_norm).item()
```

```{.python .input #batch-size-a-two-batch-estimator-2}
%%tab jax
def ce_loss(model, X, Y):  #@save
    """Mean cross-entropy of the logits model(X) for the labels Y."""
    logits = model(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

@nnx.jit  #@save
def grad_sq_norm(model, X, Y):
    """Squared norm of the gradient of ce_loss over all parameters."""
    grads = nnx.grad(ce_loss)(model, X, Y)
    return sum((g ** 2).sum() for g in jax.tree.leaves(grads))

def noise_scale(model, sample, b_small=16, b_big=2048, m=400):  #@save
    """Gradient-noise scale from the mean squared gradient norms of m
    minibatches of size b_small and m // 8 of size b_big from sample(b)."""
    def mean_sq_norm(b, m):  # N_b, averaged over m random minibatches
        return jnp.stack([grad_sq_norm(model, *sample(b))
                          for _ in range(m)]).mean()
    n_small, n_big = mean_sq_norm(b_small, m), mean_sq_norm(b_big, m // 8)
    tr_sigma = (n_small - n_big) / (1 / b_small - 1 / b_big)
    sq_norm = (b_big * n_big - b_small * n_small) / (b_big - b_small)
    return float(tr_sigma / sq_norm)
```

```{.python .input #batch-size-a-two-batch-estimator-4}
%%tab pytorch
lm = d2l.TinyLM(len(data.vocab)).to(device)
print(f'noise scale at initialization: '
      f'{noise_scale(lm, sample_lm, b_small=8, b_big=256):.1f}')
optimizer = torch.optim.Adam(lm.parameters(), lr=0.003)
d2l.train_steps(lm, data, optimizer, num_steps=500)
print(f'after 500 steps of Adam: '
      f'{noise_scale(lm, sample_lm, b_small=8, b_big=256):.0f}')
```

```{.python .input #batch-size-a-two-batch-estimator-4}
%%tab jax
lm = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(d2l.get_key()))
print(f'noise scale at initialization: '
      f'{noise_scale(lm, sample_lm, b_small=8, b_big=256):.1f}')
optimizer = nnx.Optimizer(lm, optax.adam(0.003), wrt=nnx.Param)
d2l.train_steps(lm, data, optimizer, num_steps=500)
print(f'after 500 steps of Adam: '
      f'{noise_scale(lm, sample_lm, b_small=8, b_big=256):.0f}')
```

At initialization, every window demands the same coarse unigram adjustment, so the signal $\|\nabla f\|^2$ is large and $b_{\text{noise}} \approx 2$ sequences. Once common byte statistics are learned (by step $500$), $\|\nabla f\|^2$ drops while inter-sequence variance $\operatorname{tr}\boldsymbol{\Sigma}$ persists, causing $b_{\text{noise}}$ to grow more than tenfold to $\sim 50\text{--}150$ sequences (and on `FashionCNN`, to $\sim 100\text{--}300$ images).

## Batch Size, Step Efficiency, and Learning-Rate Scaling

### Optimal Step Size and the Efficiency Frontier

Taking expectations of the $L$-smooth descent bound $f(\mathbf{x} - \eta\hat{\mathbf{g}}_b) - f(\mathbf{x}) \le -\eta\nabla f^\top \hat{\mathbf{g}}_b + \frac{L}{2}\eta^2\|\hat{\mathbf{g}}_b\|^2$ and substituting :eqref:`eq_sqnorm-bias` gives

$$
\mathbb{E}\big[f(\mathbf{x} - \eta\, \hat{\mathbf{g}}_b)\big] - f(\mathbf{x})
\leq -\eta\, \|\nabla f\|^2
+ \frac{L}{2}\, \eta^2 \Big(\|\nabla f\|^2 + \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}\Big).
$$
:eqlabel:`eq_noisy-step`

Minimizing the right-hand side over $\eta$ yields the optimal step size $\eta_{\text{opt}}(b)$ and single-step guaranteed decrease:

$$
\eta_{\textrm{opt}}(b) = \frac{1}{L} \cdot \frac{1}{1 + b_{\textrm{noise}} / b},
\qquad
f(\mathbf{x}) - \mathbb{E}\big[f(\mathbf{x} - \eta_{\textrm{opt}}\, \hat{\mathbf{g}}_b)\big]
\geq \frac{\|\nabla f\|^2}{2L} \cdot \frac{1}{1 + b_{\textrm{noise}} / b}.
$$
:eqlabel:`eq_best-step`

*(Replacing the scalar smoothness bound $L\mathbf{I}$ with the local Hessian $\mathbf{H}$ yields the exact same expressions with the curvature-weighted noise scale $\operatorname{tr}(\mathbf{H}\boldsymbol{\Sigma}) / (\nabla f^\top\mathbf{H}\nabla f)$ :cite:`McCandlish.Kaplan.Amodei.ea.2018`, for which $b_{\text{noise}}$ in :eqref:`eq_noise-scale` serves as a Hessian-free proxy.)*

Because each step at batch size $b$ makes a fraction $(1 + b_{\text{noise}}/b)^{-1}$ of the progress of an exact full-batch step, reaching a target loss that requires $S_{\min}$ full-batch steps takes $S(b)$ steps and $E(b) = b\,S(b)$ total training examples:

$$
\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\textrm{noise}}}{b},
\qquad
\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\textrm{noise}}},
$$
:eqlabel:`eq_steps-examples`

where $E_{\min} = b_{\text{noise}} S_{\min}$ is the minimal data budget achieved in the small-batch limit $b \ll b_{\text{noise}}$. At the **critical batch size** $b_{\text{crit}} = b_{\text{noise}}$, both the step count and the example count equal twice their theoretical minimum ($S = 2S_{\min}, E = 2E_{\min}$), marking the knee of the Pareto frontier $((E/E_{\min} - 1)(S/S_{\min} - 1) = 1$).

### Linear vs. Square-Root Learning-Rate Scaling
:label:`subsec_batch-size-limits`

When increasing the batch size from $b_0$ to $b$ in the noise-dominated regime ($b_0, b \ll b_{\text{noise}}$), how must the learning rate $\eta$ scale?
* **SGD and Momentum (linear scaling):** From :eqref:`eq_best-step`, when $b \ll b_{\text{noise}}$, $\eta_{\text{opt}}(b) \approx \frac{b}{L b_{\text{noise}}} \propto b$. Equivalently, $k$ steps of size $\eta_0$ on batch size $b_0$ match one step of size $k\eta_0$ on batch size $k b_0$ while preserving the stationary noise floor $\propto \eta / b$ :cite:`Goyal.Dollar.Girshick.ea.2017,Smith.Kindermans.Ying.ea.2018`.
* **Adam and AdamW (square-root scaling):** In the noise-dominated regime where a coordinate gradient has mean $\bar{g}$ and standard deviation $\sigma / \sqrt{b}$, Adam normalizes the gradient by its RMS scale $\sqrt{\hat{v}_t} \approx \sigma / \sqrt{b}$, yielding single-step drift $-\eta(\sqrt{b}/\sigma)\bar{g}$ and step noise $-\eta z$ ($z \sim \mathcal{N}(0, 1)$). Because the denominator $\sigma/\sqrt{b}$ already supplies a factor of $\sqrt{b/b_0}$, matching the drift and cumulative variance of $k = b/b_0$ small steps in a single large step requires scaling $\eta$ by $\sqrt{b/b_0}$ :cite:`Malladi.Lyu.Panigrahi.ea.2022`:

$$
\eta(b) = \frac{b}{b_0}\, \eta_0 \;\; \textrm{(SGD)},
\qquad
\eta(b) = \sqrt{\frac{b}{b_0}}\, \eta_0 \;\; \textrm{(Adam / AdamW)}.
$$
:eqlabel:`eq_lr-rules`

Once $b$ exceeds $b_{\text{crit}}$, gradient noise no longer dominates the denominator and the optimal learning rate saturates at the deterministic curvature limit rather than continuing to grow with $b$ :cite:`Shallue.Lee.Antognini.ea.2019`.

## Empirical Verification: Steps to Target and Scaling Rules
:label:`subsec_batch-size-steps`

We first measure the steps $S(b)$ required to reach a fixed held-out loss target ($1.9$ nats/byte on `TinyLM` for $b \in \{4, 16, 64, 256\}$ sequences; $0.35$ loss on `FashionCNN` for $b \in \{8, 32, 128, 512, 2048\}$ images) under Adam with square-root learning-rate scaling, fit $S(b) = S_{\min}(1 + b_{\text{crit}}/b)$, and plot the normalized runs against the universal Pareto frontier :eqref:`eq_steps-examples`:

```{.python .input #batch-size-steps-to-a-target}
%%tab pytorch
def eval_loss(model, X, Y, chunk=4096):
    with torch.inference_mode():
        return torch.stack([ce_loss(model, X[i:i + chunk], Y[i:i + chunk])
                            for i in range(0, len(Y), chunk)]).mean().item()
```

```{.python .input #batch-size-steps-to-a-target}
%%tab jax
eval_chunk = nnx.jit(ce_loss)

def eval_loss(model, X, Y, chunk=4096):
    return float(jnp.stack([eval_chunk(model, X[i:i + chunk], Y[i:i + chunk])
                            for i in range(0, len(Y), chunk)]).mean())
```

```{.python .input #batch-size-steps-to-a-target-1}
%%tab pytorch, jax
def train_to_target(model, data, optimizer, X_eval, Y_eval, target,
                    max_steps, eval_every=10):
    """Return the number of steps until the eval loss reaches target."""
    def reached(step):
        return (step % eval_every == 0
                and eval_loss(model, X_eval, Y_eval) <= target)
    steps = len(d2l.train_steps(model, data, optimizer, max_steps,
                                stop=reached))
    return steps if reached(steps) else float('inf')
```

```{.python .input #batch-size-the-language-model}
%%tab pytorch
starts = np.arange(0, len(data.val) - data.num_steps, data.num_steps)
X_val, Y_val = (t.to(device) for t in data.windows(data.val,
                                                   starts[::8][:2048]))
lm_bs, lm_steps = [4, 16, 64, 256], []
seed = random.getrandbits(32)  # One initialization for all batch sizes
for b in lm_bs:
    torch.manual_seed(seed)
    data.batch_size = b
    model = d2l.TinyLM(len(data.vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.003 * math.sqrt(b / 64))
    lm_steps.append(train_to_target(model, data, optimizer, X_val, Y_val,
                                    target=1.9, max_steps=8000))
    print(f'b={b:4d}  steps={lm_steps[-1]:6.0f}  '
          f'examples={b * lm_steps[-1]:8.0f}')
d2l.plot(lm_bs, [lm_steps, [lm_steps[0] * lm_bs[0] / b for b in lm_bs]],
         'batch size (sequences)', 'steps to loss 1.9', xscale='log',
         yscale='log', legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-language-model}
%%tab jax
starts = np.arange(0, len(data.val) - data.num_steps, data.num_steps)
X_val, Y_val = data.windows(data.val, starts[::8][:2048])
lm_bs, lm_steps = [4, 16, 64, 256], []
seed = d2l.get_seed()  # One initialization for all batch sizes
for b in lm_bs:
    data.batch_size = b
    model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model,
                              optax.adam(0.003 * math.sqrt(b / 64)),
                              wrt=nnx.Param)
    lm_steps.append(train_to_target(model, data, optimizer, X_val, Y_val,
                                    target=1.9, max_steps=8000))
    print(f'b={b:4d}  steps={lm_steps[-1]:6.0f}  '
          f'examples={b * lm_steps[-1]:8.0f}')
d2l.plot(lm_bs, [lm_steps, [lm_steps[0] * lm_bs[0] / b for b in lm_bs]],
         'batch size (sequences)', 'steps to loss 1.9', xscale='log',
         yscale='log', legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-cnn}
%%tab pytorch
cnn_bs, cnn_steps = [8, 32, 128, 512, 2048], []
seed = random.getrandbits(32)  # One initialization for all batch sizes
for b in cnn_bs:
    torch.manual_seed(seed)
    fashion_b = d2l.FashionMNIST(batch_size=b)
    model = d2l.FashionCNN().to(device)
    lr = 0.004 * math.sqrt(min(b, 512) / 256)  # Held fixed above 512
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cnn_steps.append(train_to_target(model, fashion_b, optimizer,
                                     X_f[:8192], y_f[:8192], target=0.35,
                                     max_steps=8000))
    print(f'b={b:4d}  steps={cnn_steps[-1]:6.0f}  '
          f'examples={b * cnn_steps[-1]:8.0f}')
d2l.plot(cnn_bs, [cnn_steps,
                  [cnn_steps[0] * cnn_bs[0] / b for b in cnn_bs]],
         'batch size', 'steps to loss 0.35', xscale='log', yscale='log',
         legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-cnn}
%%tab jax
cnn_bs, cnn_steps = [8, 32, 128, 512, 2048], []
seed = d2l.get_seed()  # One initialization for all batch sizes
for b in cnn_bs:
    fashion_b = d2l.FashionMNIST(batch_size=b)
    model = d2l.FashionCNN(rngs=nnx.Rngs(seed))
    lr = 0.004 * math.sqrt(min(b, 512) / 256)  # Held fixed above 512
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    cnn_steps.append(train_to_target(model, fashion_b, optimizer,
                                     X_f[:8192], y_f[:8192], target=0.35,
                                     max_steps=8000))
    print(f'b={b:4d}  steps={cnn_steps[-1]:6.0f}  '
          f'examples={b * cnn_steps[-1]:8.0f}')
d2l.plot(cnn_bs, [cnn_steps,
                  [cnn_steps[0] * cnn_bs[0] / b for b in cnn_bs]],
         'batch size', 'steps to loss 0.35', xscale='log', yscale='log',
         legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-critical-batch-size-1}
%%tab pytorch, jax
def fit_steps(bs, steps):
    """Fit S(b) = S_min (1 + b_crit / b) by least squares in 1/b."""
    b, S = np.array(bs, dtype=float), np.array(steps, dtype=float)
    ok = np.isfinite(S)  # Leave out runs that never reached the target
    slope, S_min = np.polyfit(1 / b[ok], S[ok], 1, w=1 / S[ok])
    assert S_min > 0, 'step counts never level off: no floor to fit'
    return S_min, slope / S_min

fits = {'TinyLM': fit_steps(lm_bs, lm_steps),
        'FashionCNN': fit_steps(cnn_bs, cnn_steps)}
for name, (S_min, b_crit) in fits.items():
    print(f'{name:10s}  S_min = {S_min:4.0f} steps  b_crit = {b_crit:4.0f}')
```

```{.python .input #batch-size-the-bill-in-examples}
%%tab pytorch, jax
x = np.logspace(0.005, 1.3, 100)  # E / E_min along the model curve
xs, ys = [x], [1 + 1 / (x - 1)]
for (S_min, b_crit), bs, steps in zip(fits.values(), (lm_bs, cnn_bs),
                                      (lm_steps, cnn_steps)):
    xs.append([b * S / (b_crit * S_min) for b, S in zip(bs, steps)])
    ys.append([S / S_min for S in steps])
d2l.plot(xs, ys, 'examples / minimum', 'steps / minimum', xscale='log',
         yscale='log', fmts=('-', 'o', 's'),
         legend=['model', 'TinyLM', 'FashionCNN'])
```

Both architectures fall near the universal curve $(E/E_{\min} - 1)(S/S_{\min} - 1) = 1$ (a point close to $E = E_{\min}$ sits on the curve's steep wall, where a small error in the measured step count moves it far up or down) with fitted critical batch sizes $b_{\text{crit}} \sim 100$ sequences for `TinyLM` and $b_{\text{crit}} \sim 200\text{--}400$ images for `FashionCNN`, matching the order of magnitude of the two-batch estimator $b_{\text{noise}}$.

Finally, we directly verify the linear ($\eta \propto b$) versus square-root ($\eta \propto \sqrt{b}$) learning-rate scaling rules :eqref:`eq_lr-rules` on `FashionCNN` at a fixed data budget of $12{,}800$ images ($1/5$ epoch). Increasing $b$ by $8\times$ ($8 \to 64$) under SGD with momentum shifts the optimal learning rate by $4\text{--}8\times$ (at $b = 64$ the final loss is nearly flat between $\eta = 0.05$ and $0.1$), whereas increasing $b$ by $64\times$ ($2 \to 128$) under Adam shifts the optimal learning rate by only $4\text{--}8\times$ ($2$ to $3$ grid steps of $2\times$), near $\sqrt{64} = 8\times$ rather than $64\times$:

```{.python .input #batch-size-learning-rate-rules-1}
%%tab pytorch
def final_loss_at(make_optimizer, lrs, b, seed, num_examples=12800):
    """Train the CNN once per learning rate; return the final losses."""
    curves, _ = d2l.lr_sweep(d2l.FashionCNN, make_optimizer,
                             d2l.FashionMNIST(batch_size=b), lrs, seed,
                             num_examples // b, verbose=False)
    return [d2l.final_loss(c, k=len(c) // 4) for c in curves.values()]
```

```{.python .input #batch-size-learning-rate-rules-1}
%%tab jax
def final_loss_at(make_tx, lrs, b, seed, num_examples=12800):
    """Train the CNN once per learning rate; return the final losses."""
    init_key, order_key = jax.random.split(jax.random.key(seed))
    data = d2l.DataModule()  # Every run sees the images in one order
    data.train_dataloader = lambda: d2l.ArrayLoader(
        (X_f, y_f), b, shuffle=True, key=order_key)
    curves, _ = d2l.lr_sweep(d2l.FashionCNN, make_tx, data, lrs, init_key,
                             num_examples // b, verbose=False)
    return [d2l.final_loss(c, k=len(c) // 4) for c in curves.values()]
```

```{.python .input #batch-size-learning-rate-rules-2}
%%tab pytorch
sgd_lrs = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
make_sgd = lambda model, lr: torch.optim.SGD(model.parameters(), lr,
                                             momentum=0.9)
seed = random.getrandbits(32)
sgd_small = final_loss_at(make_sgd, sgd_lrs, b=8, seed=seed)
sgd_big = final_loss_at(make_sgd, sgd_lrs, b=64, seed=seed)
for b, loss in ((8, sgd_small), (64, sgd_big)):
    print(f'b={b:2d}: best lr {sgd_lrs[np.nanargmin(loss)]:g}')
d2l.plot(sgd_lrs, [sgd_small, sgd_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

```{.python .input #batch-size-learning-rate-rules-2}
%%tab jax
sgd_lrs = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
make_sgd = lambda lr: optax.sgd(lr, momentum=0.9)
seed = d2l.get_seed()
sgd_small = final_loss_at(make_sgd, sgd_lrs, b=8, seed=seed)
sgd_big = final_loss_at(make_sgd, sgd_lrs, b=64, seed=seed)
for b, loss in ((8, sgd_small), (64, sgd_big)):
    print(f'b={b:2d}: best lr {sgd_lrs[np.nanargmin(loss)]:g}')
d2l.plot(sgd_lrs, [sgd_small, sgd_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

```{.python .input #batch-size-learning-rate-rules-3}
%%tab pytorch
adam_lrs = [0.0001 * 2 ** k for k in range(13)]  # 0.0001 to 0.4096
make_adam = lambda model, lr: torch.optim.Adam(model.parameters(), lr)
seed = random.getrandbits(32)
adam_small = final_loss_at(make_adam, adam_lrs, b=2, seed=seed)
adam_big = final_loss_at(make_adam, adam_lrs, b=128, seed=seed)
i, j = int(np.nanargmin(adam_small)), int(np.nanargmin(adam_big))
print(f'best lr: {adam_lrs[i]:g} at b=2; {adam_lrs[j]:g} at b=128, '
      f'final loss {adam_big[j]:.3f}')
for rule, k in (('no scaling', 0), ('square root', 3), ('linear', 6)):
    if i + k < len(adam_lrs):
        print(f'{rule:11s}: lr {adam_lrs[i + k]:g} at b=128, '
              f'final loss {adam_big[i + k]:.3f}')
    else:
        print(f'{rule:11s}: lr {adam_lrs[i] * 2 ** k:g}, beyond the grid')
d2l.plot(adam_lrs, [adam_small, adam_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=2', 'b=128'])
```

```{.python .input #batch-size-learning-rate-rules-3}
%%tab jax
adam_lrs = [0.0001 * 2 ** k for k in range(13)]  # 0.0001 to 0.4096
make_adam = optax.adam
seed = d2l.get_seed()
adam_small = final_loss_at(make_adam, adam_lrs, b=2, seed=seed)
adam_big = final_loss_at(make_adam, adam_lrs, b=128, seed=seed)
i, j = int(np.nanargmin(adam_small)), int(np.nanargmin(adam_big))
print(f'best lr: {adam_lrs[i]:g} at b=2; {adam_lrs[j]:g} at b=128, '
      f'final loss {adam_big[j]:.3f}')
for rule, k in (('no scaling', 0), ('square root', 3), ('linear', 6)):
    if i + k < len(adam_lrs):
        print(f'{rule:11s}: lr {adam_lrs[i + k]:g} at b=128, '
              f'final loss {adam_big[i + k]:.3f}')
    else:
        print(f'{rule:11s}: lr {adam_lrs[i] * 2 ** k:g}, beyond the grid')
d2l.plot(adam_lrs, [adam_small, adam_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=2', 'b=128'])
```

### Practical Procedure for Selecting Batch Size

1. **Saturate hardware arithmetic first:** Increase $b$ at least until per-step execution time begins to scale linearly with $b$ (:numref:`sec_sgd`).
2. **Estimate $b_{\text{noise}}$ post-warmup:** Measure `noise_scale` after the initial loss drop (e.g., a few hundred steps into training), as $b_{\text{noise}}$ starts very small at initialization and grows by $1\text{--}2$ orders of magnitude during training (motivating batch-size warmup ramps in large pretraining runs such as GPT-3 :cite:`brown2020language`).
3. **Position $b$ relative to $b_{\text{crit}} \approx b_{\text{noise}}$:** When compute/data is the bottleneck, choose $b \lesssim 0.3\,b_{\text{crit}}$; when wall-clock time across parallel accelerators is the bottleneck, choose $b \approx b_{\text{crit}}$ (where $S = 2S_{\min}$ and $E = 2E_{\min}$), and avoid $b \gg b_{\text{crit}}$.
4. **Scale $\eta$ via :eqref:`eq_lr-rules`:** Scale $\eta \propto b$ for SGD/Momentum and $\eta \propto \sqrt{b}$ for AdamW up to $b_{\text{crit}}$, paired with linear warmup (:numref:`subsec_scheduler-warmup`).

## Summary

* The **gradient-noise scale** $b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$ measures the batch size at which gradient noise power equals signal power, and can be estimated online from squared gradient norms at two batch sizes $b_{\text{s}} < b_{\text{l}}$ via :eqref:`eq_sqnorm-bias`.
* Under optimal learning rates, the steps $S(b)$ and examples $E(b) = b\,S(b)$ required to reach a target loss follow $S(b)/S_{\min} = 1 + b_{\text{crit}}/b$ and $E(b)/E_{\min} = 1 + b/b_{\text{crit}}$, with $b_{\text{crit}} \approx b_{\text{noise}}$.
* Below $b_{\text{crit}}$, the optimal learning rate scales linearly ($\eta \propto b$) for SGD/Momentum and as the square root ($\eta \propto \sqrt{b}$) for Adam/AdamW, saturating once $b \gtrsim b_{\text{crit}}$.

## Exercises

1. **Larger batch size at an unchanged learning rate.** Using :eqref:`eq_noisy-step` and :eqref:`eq_best-step`, show that if $\eta$ is held fixed at $\eta_0 = \eta_{\text{opt}}(b_0)$ while the batch size is increased to $b > b_0$, the guaranteed single-step decrease improves by factor $2 - (1 + b_{\text{noise}}/b)/(1 + b_{\text{noise}}/b_0) < 2$, never more than doubling even as $b \to \infty$.
   :exlabel:`ex_batch-size-fixed-rate`

1. **Time-optimal vs. compute-optimal batch size.** If one step at batch size $b$ takes wall-clock time $t(b) = t_0(1 + b/b_{\text{sat}})$, use :eqref:`eq_steps-examples` to derive the batch size $b^\star$ that minimizes total wall-clock time $T(b) = t(b)S(b)$ and compute its data overhead $E(b^\star)/E_{\min}$.
   :exlabel:`ex_batch-size-time-optimal-compute-optimal`

1. [code] **Evolution of signal and noise over training.** Modify `noise_scale` to return $\operatorname{tr}\boldsymbol{\Sigma}$, $\|\nabla f\|^2$, and their ratio $b_{\text{noise}}$ separately, track all three on `TinyLM` at steps $0, 25, 50, 100, 200, 500, 1000, 2000, 3000$, and verify that the $100\times$ growth in $b_{\text{noise}}$ is driven primarily by the drop in signal $\|\nabla f\|^2$ rather than an increase in $\operatorname{tr}\boldsymbol{\Sigma}$.
   :exlabel:`ex_batch-size-noise-scale-over-training`

1. [code] **Scaling past the critical batch size.** Extend the `TinyLM` step-to-target sweep of :numref:`subsec_batch-size-steps` to $b = 1{,}024$ under the square-root rule $\eta \propto \sqrt{b}$ versus clamping $\eta$ at its $b = 256$ value, and explain why continuing to scale $\eta$ above $b_{\text{crit}}$ degrades convergence.
   :exlabel:`ex_batch-size-past-elbow`

[Discussions](https://d2l.discourse.group/t/36765)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.8]{.kicker}

Batch Size<br>
**the gradient-noise scale · steps against examples · learning-rate rules · choosing a batch size**
:::
:::

::: {.slide title="The Gradient-Noise Scale Grows Over 10× as Common Patterns Are Learned" layout="tight"}
[A two-batch estimator]{.kicker}

@!batch-size-a-two-batch-estimator-4
:::

::: {.slide title="Step Counts Follow Linear Scaling Below b_crit and Bend Toward a Floor Above It" layout="tight"}
[Steps to a target]{.kicker}

@!batch-size-the-language-model
:::

::: {.slide title="TinyLM and FashionCNN Fall Near the Universal Step–Data Frontier" layout="tight"}
[The critical batch size]{.kicker}

$$\Big(\frac{E}{E_{\min}} - 1\Big)\Big(\frac{S}{S_{\min}} - 1\Big) = 1$$

@!batch-size-the-bill-in-examples
:::

::: {.slide title="Optimal Learning Rate Scales as b for SGD and √b for Adam Below b_crit" layout="tight"}
[Tests of the learning-rate rules]{.kicker}

@!batch-size-learning-rate-rules-3
:::
