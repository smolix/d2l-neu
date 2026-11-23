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

We first load WikiText-103 and Fashion-MNIST onto the accelerator and define the random minibatch samplers `sample_fashion(b)` and `sample_lm(b)`:

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

Using these samplers, we save the cross-entropy helper `ce_loss`, the parameter-wide squared gradient norm `grad_sq_norm`, and the two-batch estimator `noise_scale` in `d2l`:

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

We now evaluate `noise_scale` on `TinyLM` ($b_{\text{s}} = 8, b_{\text{l}} = 256$) at initialization and again after $500$ steps of Adam ($\eta = 3 \times 10^{-3}$):

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

Does the noise scale of a convolutional network on images grow in the same way? We repeat the measurement on `FashionCNN`, with the default $b_{\text{s}} = 16$ and $b_{\text{l}} = 2{,}048$ images, at initialization and after $200$ steps of Adam ($\eta = 0.004$, the rate the sweep below uses at $b = 256$):

```{.python .input #batch-size-signal-noise-and-a-two-batch-estimator-1}
%%tab pytorch
cnn = d2l.FashionCNN().to(device)
print(f'noise scale at initialization: {noise_scale(cnn, sample_fashion):.0f}')
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.004)
d2l.train_steps(cnn, fashion, optimizer, num_steps=200)
print(f'after 200 steps of Adam: {noise_scale(cnn, sample_fashion):.0f}')
```

```{.python .input #batch-size-signal-noise-and-a-two-batch-estimator-1}
%%tab jax
cnn = d2l.FashionCNN(rngs=nnx.Rngs(d2l.get_key()))
print(f'noise scale at initialization: {noise_scale(cnn, sample_fashion):.0f}')
optimizer = nnx.Optimizer(cnn, optax.adam(0.004), wrt=nnx.Param)
d2l.train_steps(cnn, fashion, optimizer, num_steps=200)
print(f'after 200 steps of Adam: {noise_scale(cnn, sample_fashion):.0f}')
```

On `TinyLM` at initialization, every window demands the same coarse unigram adjustment, so the signal $\|\nabla f\|^2$ is large and $b_{\text{noise}} \approx 2$ sequences. Once common byte statistics are learned (by step $500$), $\|\nabla f\|^2$ drops while inter-sequence variance $\operatorname{tr}\boldsymbol{\Sigma}$ persists, causing $b_{\text{noise}}$ to grow more than tenfold to $\sim 50\text{--}150$ sequences.

`FashionCNN` behaves differently. Its ten classes are balanced, so an untrained network finds no adjustment that every image agrees on, and the noise scale is of the order of $100$ images from the first step. It hardly moves over the $200$ steps and stays near $100$ images.

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
\begin{aligned}
\eta_{\textrm{opt}}(b) &= \frac{1}{L} \cdot \frac{1}{1 + b_{\textrm{noise}} / b}, \\
f(\mathbf{x}) - \mathbb{E}\big[f(\mathbf{x} - \eta_{\textrm{opt}}\, \hat{\mathbf{g}}_b)\big]
&\geq \frac{\|\nabla f\|^2}{2L} \cdot \frac{1}{1 + b_{\textrm{noise}} / b}.
\end{aligned}
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
\begin{aligned}
\eta(b) &= \frac{b}{b_0}\, \eta_0 \;\; \textrm{(SGD)}, \\
\eta(b) &= \sqrt{\frac{b}{b_0}}\, \eta_0 \;\; \textrm{(Adam / AdamW)}.
\end{aligned}
$$
:eqlabel:`eq_lr-rules`

Once $b$ exceeds $b_{\text{crit}}$, gradient noise no longer dominates the denominator and the optimal learning rate saturates at the deterministic curvature limit rather than continuing to grow with $b$ :cite:`Shallue.Lee.Antognini.ea.2019`.

## Empirical Verification: Steps to Target and Scaling Rules
:label:`subsec_batch-size-steps`

To measure the steps $S(b)$ required to reach a fixed held-out loss target, we first define `eval_loss` to evaluate `ce_loss` over validation tensors in chunks of $4{,}096$ examples:

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

The stopping harness `train_to_target` checks `eval_loss` every `eval_every` steps inside `d2l.train_steps` and returns the step count at which `target` is reached (or `inf` if `max_steps` is exhausted):

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

We first run `train_to_target` on `TinyLM` (target validation loss $1.9$ nats/byte) across $b \in \{4, 16, 64, 256\}$ sequences from a shared initialization, scaling the Adam learning rate as $\eta = 0.003\sqrt{b / 64}$:

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

We repeat the step-to-target sweep on `FashionCNN` (target loss $0.35$) across $b \in \{8, 32, 128, 512, 2048\}$ images, scaling $\eta = 0.004\sqrt{\min(b, 512)/256}$ up to $b = 512$:

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

Because $S(b) = S_{\min}(1 + b_{\text{crit}}/b)$ is linear in $1/b$, `fit_steps` extracts the asymptotic step floor $S_{\min}$ and empirical critical batch size $b_{\text{crit}}$ for both models via weighted least squares:

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

Normalizing each run's example count $E(b) = b\,S(b)$ by $E_{\min} = b_{\text{crit}} S_{\min}$ and step count by $S_{\min}$ plots both models against the universal Pareto frontier :eqref:`eq_steps-examples`:

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

Both architectures fall near the universal curve $(E/E_{\min} - 1)(S/S_{\min} - 1) = 1$ (a point close to $E = E_{\min}$ sits on the curve's steep wall, where a small error in the measured step count moves it far up or down) with fitted critical batch sizes $b_{\text{crit}} \sim 100$ sequences for `TinyLM` and $b_{\text{crit}} \sim 200\text{--}400$ images for `FashionCNN`. For `TinyLM` the fit agrees with the two-batch estimate of $b_{\text{noise}}$ measured above, somewhat under $100$ sequences. For `FashionCNN` it is three to four times the estimate of about $100$ images. The simple noise scale ignores curvature, so it gives the order of magnitude of $b_{\text{crit}}$ and not its value.

Finally, we directly verify the linear ($\eta \propto b$) versus square-root ($\eta \propto \sqrt{b}$) learning-rate scaling rules :eqref:`eq_lr-rules` on `FashionCNN` at a fixed data budget of $12{,}800$ images ($1/5$ epoch). The helper `final_loss_at` runs `d2l.lr_sweep` for `num_examples // b` steps at batch size `b` and returns the terminal losses averaged over the final quarter of each run:

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

Under SGD with momentum ($\beta = 0.9$), increasing the batch size $8\times$ from $b = 8$ to $b = 64$ across `sgd_lrs` shifts the optimal learning rate rightward by $4\text{--}8\times$ (at $b = 64$ the final loss is nearly flat between $\eta = 0.05$ and $0.1$):

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

By contrast, increasing the batch size $64\times$ from $b = 2$ to $b = 128$ under Adam across `adam_lrs` shifts the optimal learning rate by only $4\text{--}8\times$ ($2$ to $3$ grid steps of $2\times$), matching $\sqrt{64} = 8\times$ rather than $64\times$:

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
2. **Estimate $b_{\text{noise}}$ post-warmup:** Measure `noise_scale` after the initial loss drop (e.g., a few hundred steps into training), as $b_{\text{noise}}$ can start very small at initialization and grow by $1\text{--}2$ orders of magnitude during training, as it did for `TinyLM` (motivating batch-size warmup ramps in large pretraining runs such as GPT-3 :cite:`brown2020language`).
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

Batch Size, Gradient Noise Scale, and Critical Batch Size<br>**signal vs. noise power · two-batch estimator · step-vs.-example Pareto frontier · critical batch size · linear & √b LR rules**
:::
:::

::: {.slide title="When Does Doubling the Batch Size Halve the Step Count?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Small batch ($b: 4 \to 16$, $4\times$):** Noise-dominated; averaging cuts sampling variance $\operatorname{tr}(\boldsymbol{\Sigma})/b$, delivering **$\approx 3\times$ fewer steps** at nearly constant data cost ($\approx 1.1\text{--}1.3\times E(4)$).
- **Large batch ($b: 64 \to 256$, $4\times$):** Curvature-dominated; minibatch gradients already match $\nabla f$, yielding only **$\approx 2\times$ fewer steps** while doubling data & FLOP cost.
- **The knee ($b_{\text{crit}} \approx b_{\text{noise}}$):** Governed by the gradient-noise scale $b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$ (McCandlish et al., 2018).
:::

::: {.col .fig .big}
@fig:ch09-opt-bs-two-regimes
:::
:::
:::

::: {.slide title="Signal Power, Noise Power, and Gradient-Noise Scale"}
[The Signal-to-Noise Ratio of a Minibatch]{.kicker}

::: {.cols .vc}
::: {.col}
- **Expected squared norm decomposition:**
  $$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \underbrace{\|\nabla f\|^2}_{\text{Signal power}} + \underbrace{\frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}}_{\text{Noise power}}$$
- **Simple gradient-noise scale:** The batch size where noise power equals signal power:
  $$b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$$
- For $b \ll b_{\text{noise}}$, $\hat{\mathbf{g}}_b$ is mostly sampling noise; for $b \gg b_{\text{noise}}$, $\hat{\mathbf{g}}_b \approx \nabla f$.
:::

::: {.col .fig}
@fig:ch09-opt-bs-snr-crossover
:::
:::
:::

::: {.slide title="A Two-Batch Estimator for b_noise Without Full Passes"}
[Online Noise-Scale Estimation]{.kicker}

::: {.cols .vc}
::: {.col}
- Computing $\|\nabla f\|^2$ directly would require a full pass over the dataset.
- Instead, measure the mean squared gradient norm $N_b = \mathbb{E}[\|\hat{\mathbf{g}}_b\|^2]$ at two minibatch sizes $b_{\text{s}} < b_{\text{l}}$ and solve the $2 \times 2$ linear system $N_b = \|\nabla f\|^2 + \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$:
  $$\operatorname{tr}\boldsymbol{\Sigma} \approx \frac{N_{b_{\text{s}}} - N_{b_{\text{l}}}}{1/b_{\text{s}} - 1/b_{\text{l}}}$$
- The signal norm is then $\|\nabla f\|^2 \approx \frac{b_{\text{l}} N_{b_{\text{l}}} - b_{\text{s}} N_{b_{\text{s}}}}{b_{\text{l}} - b_{\text{s}}}$. For example, if $N_{16} = 0.50$ and $N_{2048} = 0.103$, then $\operatorname{tr}\boldsymbol{\Sigma} \approx 6.4$, $\|\nabla f\|^2 \approx 0.10$, and $b_{\text{noise}} \approx 64$.
:::

::: {.col .fig}
@-batch-size-a-two-batch-estimator-2
:::
:::
:::

::: {.slide title="How b_noise Grows Over Training on TinyLM"}
[Dynamic Evolution of the Noise Scale]{.kicker}

::: {.cols .vc}
::: {.col}
- Measuring `noise_scale` on `TinyLM` ($b_{\text{s}} = 8, b_{\text{l}} = 256$ sequences) before and after $500$ steps of Adam ($\eta = 3 \times 10^{-3}$):
  - **At initialization (step 0):** $b_{\text{noise}} \approx 2$ sequences! Because every untrained window demands the same global byte-frequency shift, the coherent signal $\|\nabla f\|^2$ is huge.
  - **After 500 steps:** $b_{\text{noise}}$ grows **$>30\times$ to $\sim 50\text{--}150$ sequences** as $\|\nabla f\|^2$ falls while inter-sequence disagreement $\operatorname{tr}\boldsymbol{\Sigma}$ persists!
:::

::: {.col .fig}
@!batch-size-a-two-batch-estimator-4
:::
:::
:::

::: {.slide title="Optimal Step Size and Single-Step Progress Under Noise"}
[The Noisy Descent Bound]{.kicker}

::: {.cols .vc}
::: {.col}
- **Noisy $L$-smooth descent bound:**
  $$\mathbb{E}[f(\mathbf{x} - \eta\hat{\mathbf{g}}_b)] - f(\mathbf{x}) \le -\eta\|\nabla f\|^2 + \frac{L}{2}\eta^2\|\nabla f\|^2\Big(1 + \frac{b_{\text{noise}}}{b}\Big)$$
- **Optimal step size & single-step drop:**
  $$\eta_{\text{opt}}(b) = \frac{1}{L\,(1 + b_{\text{noise}}/b)}, \quad \Delta f_{\text{opt}}(b) = \frac{\|\nabla f\|^2}{2L\,(1 + b_{\text{noise}}/b)}$$
- At $b = b_{\text{noise}}$, both $\eta_{\text{opt}}$ and $\Delta f_{\text{opt}}$ equal **exactly half** of their full-batch ($b \to \infty$) values.
:::

::: {.col .fig}
@fig:ch09-opt-bs-noisy-descent
:::
:::
:::

::: {.slide title="The Step–Data Pareto Frontier and Critical Batch Size"}
[Universal Trade-Off Law]{.kicker}

::: {.cols .vc}
::: {.col}
- **Steps $S(b)$ vs. examples $E(b) = b\,S(b)$ ($E_{\min} = b_{\text{crit}}S_{\min}$):**
  $$\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b}, \qquad \frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$$
- **Small batch ($b = 0.1\,b_{\text{crit}}$):** $11\,S_{\min}$ steps, only $1.10\,E_{\min}$ data.
- **Critical knee ($b = b_{\text{crit}}$):** Balanced **$2\,S_{\min}$ steps and $2\,E_{\min}$ data**.
- **Oversized ($b = 10\,b_{\text{crit}}$):** Saves $<50\%$ steps vs. $b_{\text{crit}}$ at $5.5\times$ the data bill ($11\,E_{\min}$)!
:::

::: {.col .fig}
@fig:ch09-opt-bs-pareto-regimes
:::
:::
:::

::: {.slide title="Why SGD Uses Linear LR Scaling (η ∝ b) Below b_crit"}
[Linear Scaling Rule for Unnormalized Optimizers]{.kicker}

::: {.cols .vc}
::: {.col}
- **Small-batch limit ($b \ll b_{\text{noise}}$):**
  $$\eta_{\text{opt}}(b) = \frac{1}{L\,(1 + b_{\text{noise}}/b)} \approx \frac{b}{L\,b_{\text{noise}}} \implies \eta(b) = \frac{b}{b_0}\eta_0$$
- **Trajectory & noise-floor equivalence (Goyal et al., 2017):** $k = b/b_0$ small steps at $(\eta_0, b_0)$ equal one step at $(k\eta_0, kb_0)$ while holding $\eta/b$ (and excess loss $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}}{4b}$) constant.
- **Saturation past $b_{\text{crit}}$:** For $b \gg b_{\text{crit}}$, $\eta_{\text{opt}} \to 1/L$; linear scaling past $b_{\text{crit}}$ violates $\eta < 2/L$.
:::

::: {.col .fig}
@fig:ch09-opt-bs-linear-vs-sqrt
:::
:::
:::

::: {.slide title="Why Adam Uses Square-Root Scaling (η ∝ √b) Below b_crit"}
[Square-Root Scaling Rule for Adaptive Optimizers]{.kicker}

Why does **Adam / AdamW** require **square-root scaling** $\eta(b) = \sqrt{b/b_0}\,\eta_0$ instead of linear scaling in the noise-dominated regime (Malladi et al., 2022)?

- Consider a coordinate whose minibatch gradient has true mean $\bar{g}$ and sampling noise $z\,\frac{\sigma}{\sqrt{b}}$ ($z \sim \mathcal{N}(0, 1)$) with $\frac{\sigma}{\sqrt{b}} \gg |\bar{g}|$.
- **Adam's denominator already scales with $\sqrt{b}$:** Because $\sqrt{\hat{v}_t} \approx \sqrt{\bar{g}^2 + \sigma^2/b} \approx \frac{\sigma}{\sqrt{b}}$, dividing by $\sqrt{\hat{v}_t}$ automatically amplifies the signal drift by $\sqrt{b}$:
  $$\Delta x_t \approx -\eta\,\frac{\bar{g} + z\,\sigma/\sqrt{b}}{\sigma/\sqrt{b}} = \underbrace{-\eta\,\frac{\sqrt{b}}{\sigma}\,\bar{g}}_{\text{Deterministic drift}} \;-\; \underbrace{\eta\,z}_{\text{Unit-variance step noise}}$$
- **Matching $k = b/b_0$ small steps in $1$ large step:** Over $k$ small steps at $(\eta_0, b_0)$, cumulative drift is $k \cdot \eta_0\frac{\sqrt{b_0}}{\sigma}\bar{g}$ and cumulative noise variance is $k\,\eta_0^2$. A single step at batch size $b = k b_0$ matches **both** the cumulative drift and cumulative variance when:
  $$\eta(b) = \sqrt{k}\,\eta_0 = \sqrt{\frac{b}{b_0}}\;\eta_0$$
:::

::: {.slide title="Empirical Step-to-Target Scaling on TinyLM"}
[Language Model Sweep]{.kicker}

::: {.cols .vc}
::: {.col}
- Training `TinyLM` with Adam ($\eta = 0.003\sqrt{b/64}$) to a held-out validation loss of $1.9$ nats across $b \in \{4, 16, 64, 256\}$ sequences:
  - **$b = 4 \to 16$ ($4\times$):** Steps drop about **$3\times$** ($\approx 4{,}000 \to 1{,}200$), closely tracking the dashed perfect-scaling line ($S \propto 1/b$).
  - **$b = 64 \to 256$ ($4\times$):** Steps drop by only about **$2\times$** ($\approx 460 \to 225$), doubling the total sequences processed ($\approx 30{,}000 \to 57{,}000$) as $b$ crosses $b_{\text{crit}} \sim 100$.
:::

::: {.col .fig}
@!batch-size-the-language-model
:::
:::
:::

::: {.slide title="Empirical Step-to-Target Scaling on FashionCNN"}
[Image Classifier Sweep]{.kicker}

::: {.cols .vc}
::: {.col}
- Training `FashionCNN` with Adam to loss $0.35$ across $b \in \{8, 32, 128, 512, 2048\}$ images:
  - **$b = 8 \to 32 \to 128$:** Near-linear step reductions ($\approx 2{,}900 \to 800 \to 250$ steps) with slowly rising total example cost ($\sim 22{,}000\text{--}32{,}000$ images).
  - **$b = 512 \to 2{,}048$ ($4\times$):** Saves only about a third of the steps ($\approx 120 \to 80$) while nearly tripling the data bill to $163{,}840$ images!
:::

::: {.col .fig}
@!batch-size-the-cnn
:::
:::
:::

::: {.slide title="Fitting S_min and b_crit by Least Squares in 1/b"}
[Extracting the Critical Batch Size]{.kicker}

::: {.cols .vc}
::: {.col}
- Because $S(b) = S_{\min}\!\left(1 + \frac{b_{\text{crit}}}{b}\right) = S_{\min} + (S_{\min}b_{\text{crit}})\frac{1}{b}$ is linear in $1/b$, weighted linear regression extracts both asymptotes directly from the step counts:
  - **Intercept:** Minimum full-batch step floor $S_{\min}$.
  - **Slope / Intercept:** Empirical critical batch size $b_{\text{crit}}$.
- The fitted values ($b_{\text{crit}} \sim 80\text{--}110$ sequences on `TinyLM` and $b_{\text{crit}} \sim 200\text{--}350$ images on `FashionCNN`) match the order of magnitude of our two-batch gradient-noise estimator $b_{\text{noise}}$!
:::

::: {.col .fig}
@!batch-size-the-critical-batch-size-1
:::
:::
:::

::: {.slide title="Collapse onto the Universal Normalized Pareto Curve"}
[Steps vs. Examples Frontier]{.kicker}

::: {.cols .vc}
::: {.col}
- Eliminating $b / b_{\text{crit}}$ between $\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b}$ and $\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$ (where $E_{\min} = b_{\text{crit}}S_{\min}$) yields an architecture-independent hyperbola:
  $$\left(\frac{E}{E_{\min}} - 1\right)\left(\frac{S}{S_{\min}} - 1\right) = 1$$
- Rescaling `TinyLM` and `FashionCNN` by their fitted $(S_{\min}, E_{\min})$ collapses both models onto this single curve!
  - **Upper-left arm ($b \ll b_{\text{crit}}$):** Data-efficient ($E \approx E_{\min}$), many serial steps.
  - **Knee ($b = b_{\text{crit}}$):** $(2E_{\min}, 2S_{\min})$.
  - **Lower-right arm ($b \gg b_{\text{crit}}$):** Step count saturates ($S \approx S_{\min}$) while data cost explodes.
:::

::: {.col .fig}
@!batch-size-the-bill-in-examples
:::
:::
:::

::: {.slide title="Verifying the Linear Scaling Rule (η ∝ b) for SGD"}
[SGD with Momentum on FashionCNN]{.kicker}

::: {.cols .vc}
::: {.col}
- At a fixed data budget of $E = 12{,}800$ images ($1/5$ epoch), we sweep SGD with momentum ($\beta = 0.9$) across a factor-of-$2$ learning-rate grid at $b = 8$ ($1{,}600$ steps) and $b = 64$ ($200$ steps):
  - Increasing the batch size by **$8\times$ ($8 \to 64$)** shifts the U-shaped loss curve and its optimal learning rate rightward by **$4\text{--}8\times$** ($0.0125 \to 0.05\text{--}0.10$).
  - Leaving $\eta$ unscaled at $0.0125$ when moving to $b = 64$ severely under-budgets the $200$-step run.
:::

::: {.col .fig}
@!batch-size-learning-rate-rules-2
:::
:::
:::

::: {.slide title="Verifying the Square-Root Rule (η ∝ √b) for Adam"}
[Adam on FashionCNN Across a 64× Batch Increase]{.kicker}

::: {.cols .vc}
::: {.col}
- At the same $12{,}800$-image budget, we increase the batch size by **$64\times$ ($b = 2 \to 128$)** under Adam across a $13$-point factor-of-$2$ grid:
  - **No scaling ($1\times$):** $\eta$ is $4\text{--}8\times$ too small at $b = 128$, losing roughly $0.1$ nats.
  - **Square-root rule ($\sqrt{64} = 8\times = 2^3$, $+3$ grid steps):** Lands at the empirical optimum ($\eta = 0.0064$) or within one grid step of it!
  - **Linear rule ($64\times = 2^6$, $+6$ grid steps):** Overshoots into the unstable right wall: the loss stays at $\log 10 \approx 2.3$, so the network learns nothing.
:::

::: {.col .fig .narrow}
@!batch-size-learning-rate-rules-3
:::
:::
:::

::: {.slide title="Dynamic Batch Size Ramps and Practical Selection Recipe"}
[Engineering Playbook]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why ramp $b$ over training?** Because $b_{\text{noise}}$ can grow $10\text{--}100\times$ from step 0 to mid-training, frontier runs ramp $b$ up dynamically (e.g., Llama 3: $4\text{M} \to 16\text{M}$ tokens).
- **Four-step selection protocol:**
  1. Set $b \ge b_{\text{sat}}$ so GPU tensor cores saturate.
  2. Measure post-warmup $b_{\text{noise}}$ at step $300\text{--}500$.
  3. Choose $b \lesssim 0.3\,b_{\text{crit}}$ (compute-bound) or $b \approx b_{\text{crit}}$ (wall-clock bound).
  4. Scale $\eta \propto b$ (SGD) or $\eta \propto \sqrt{b}$ (AdamW) with linear warmup.
:::

::: {.col .fig .big}
@fig:ch09-opt-bs-ramp-playbook
:::
:::
:::

::: {.slide title="Concept Check: Two-Batch Estimator and Scaling Rules"}
[Check Your Understanding]{.kicker}

Work out the quantitative prediction for each batch-size scenario:

1. **Two-batch estimator:** At step $500$, measuring mean squared gradient norms at $b_{\text{s}} = 10$ and $b_{\text{l}} = 100$ gives $N_{10} = 1.0$ and $N_{100} = 0.19$. What are $\operatorname{tr}\boldsymbol{\Sigma}$, $\|\nabla f\|^2$, and $b_{\text{noise}}$?
2. **Adding GPUs near $b_{\text{crit}}$:** `TinyLM` trains with AdamW at $b = 64$ and $\eta = 3 \times 10^{-3}$ where $b_{\text{crit}} = 100$. If you double the batch size to $b = 128$ across two GPUs, what learning rate does the square-root rule recommend, by what factor does the step count $S(b)$ fall, and by what factor does total data $E(b)$ increase?
3. **Why warmup is mandatory at large $b$:** If $b_{\text{crit}} = 256$ in mid-training, why can't we apply the $\sqrt{256/16} = 4\times$ scaled-up learning rate at step $t = 1$ without warmup?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. $\operatorname{tr}\boldsymbol{\Sigma} = \frac{1.0 - 0.19}{1/10 - 1/100} = \frac{0.81}{0.09} =$ **$9.0$**; $\|\nabla f\|^2 = 0.19 - \frac{9.0}{100} =$ **$0.10$**; and **$b_{\text{noise}} = \frac{9.0}{0.10} = 90$**.
2. Square-root scaling gives **$\eta = \sqrt{2} \times 3 \times 10^{-3} \approx 4.24 \times 10^{-3}$**. Step count drops by $\frac{1 + 100/64}{1 + 100/128} = \frac{2.5625}{1.78125} \approx$ **$1.44\times$** (taking $69\%$ of wall-clock time across 2 GPUs), while total data processed rises by $\frac{1 + 128/100}{1 + 64/100} = \frac{2.28}{1.64} \approx$ **$1.39\times$ ($+39\%$ data)**.
3. At step $t = 0$, $b_{\text{noise}}(0) \approx 2 \ll 256$, so $b = 256$ is initially deep in the curvature-dominated regime ($b \gg b_{\text{noise}}(0)$) where a $4\times$ scaled-up learning rate exceeds the initial stability ceiling!
:::
:::

::: {.slide title="Summary: Gradient-Noise Scale and Critical Batch Size"}
[Key Takeaways]{.kicker}

| Quantity / Law | Mathematical Formula | Regime / Scaling Behavior | Practical Implication |
| :--- | :--- | :--- | :--- |
| **Minibatch Squared Norm** | $\mathbb{E}\|\hat{\mathbf{g}}_b\|^2 = \|\nabla f\|^2 + \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$ | Signal power $+\; 1/b$ noise power | Allows estimating $\operatorname{tr}\boldsymbol{\Sigma}$ and $\|\nabla f\|^2$ from two batch sizes $b_{\text{s}} < b_{\text{l}}$ |
| **Gradient-Noise Scale** | $b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$ | Can grow $10\text{--}100\times$ during training as $\|\nabla f\|^2$ drops | Hessian-free predictor of the critical batch size $b_{\text{crit}}$ |
| **Step–Data Pareto Curve** | $\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b},\; \frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$ | Universal hyperbola $(E/E_{\min}-1)(S/S_{\min}-1) = 1$ | At $b = b_{\text{crit}}$, $S = 2S_{\min}$ and $E = 2E_{\min}$; diminishing returns above $b_{\text{crit}}$ |
| **SGD LR Scaling** | $\eta(b) = \frac{b}{b_0}\,\eta_0$ ($b \ll b_{\text{crit}}$) | Linear growth up to $b_{\text{crit}}$, then saturates at $1/L$ | Matches $k$ small steps to $1$ large step at constant noise floor $\eta/b$ |
| **AdamW LR Scaling** | $\eta(b) = \sqrt{\frac{b}{b_0}}\,\eta_0$ ($b \ll b_{\text{crit}}$) | Square-root growth because $\sqrt{\hat{v}_t} \approx \sigma/\sqrt{b}$ supplies $\sqrt{b}$ | Pair with linear warmup (or batch-size ramp) to handle small initial $b_{\text{noise}}(0)$ |
:::

