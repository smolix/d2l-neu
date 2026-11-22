```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Synthetic Data
:label:`sec_synthetic-regression-data`

Real-world datasets are messy, and worse yet, they never come with an answer key: when you train a new algorithm on real housing records and get a weight vector $\hat{\mathbf{w}}$, you have no way of knowing the true data-generating parameters $\mathbf{w}^*$ or the irreducible noise floor. Before unleashing an implementation on real data, a seasoned researcher first tests it on *synthetic data* generated from a known ground-truth model.

On a synthetic dataset, we control the exact feature distribution, the true weights $\mathbf{w}^*$ and bias $b^*$, the sample size $n$, and the noise level $\sigma$. That gives us two priceless debugging superpowers: we can check whether our learned parameters $(\hat{\mathbf{w}}, \hat{b})$ converge to $(\mathbf{w}^*, b^*)$, and we can compare our final training loss against the exact theoretical noise floor $\frac{1}{2}\sigma^2$. In this section, we build the synthetic dataset that will serve as our testbed throughout the next three sections, inspect its scatter plots, and compute the exact least-squares fit as a reference benchmark.

```{.python .input #synthetic-regression-data}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #synthetic-regression-data}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #synthetic-regression-data}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #synthetic-regression-data}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Generating the Data

To keep visualization effortless while exercising multi-feature matrix algebra, we generate $n = 1{,}000$ training examples with $d = 2$ features. We draw the design matrix $\mathbf{X} \in \mathbb{R}^{1000 \times 2}$ with independent standard normal entries $X_{ij} \sim \mathcal{N}(0, 1)$, and generate the labels from the linear model

$$\mathbf{y} = \mathbf{X} \mathbf{w}^* + b^* + \boldsymbol{\epsilon},$$

using ground-truth parameters

$$\mathbf{w}^* = (2,\, -3.4)^\top, \qquad b^* = 4.2,$$

and additive Gaussian noise $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ with standard deviation $\sigma = 0.01$. Notice that the signal $(\mathbf{X}\mathbf{w}^* + b^*)$ varies on a scale of several units, whereas the noise $\sigma = 0.01$ is hundreds of times smaller—so any healthy optimizer should recover $\mathbf{w}^*$ and $b^*$ to within a few thousandths.

:begin_tab:`jax`
Because JAX uses explicit functional pseudorandom keys (:numref:`sec_prob`), our data generator takes one extra argument, a `key`, which we draw from the tiny stateful key helper `d2l.get_key`.
:end_tab:

```{.python .input #synthetic-regression-data-generating-the-data-1}
%%tab pytorch, mxnet
def synthetic_data(w, b, noise, n):  #@save
    """n examples with standard normal features and labels Xw + b + noise."""
    X = d2l.randn(n, len(w))
    eps = d2l.randn(n, 1) * noise
    return X, d2l.matmul(X, d2l.reshape(w, (-1, 1))) + b + eps
```

```{.python .input #synthetic-regression-data-generating-the-data-1}
%%tab tensorflow
def synthetic_data(w, b, noise, n):  #@save
    """n examples with standard normal features and labels Xw + b + noise."""
    X = tf.random.normal((n, w.shape[0]))
    eps = tf.random.normal((n, 1)) * noise
    return X, d2l.matmul(X, d2l.reshape(w, (-1, 1))) + b + eps
```

```{.python .input #synthetic-regression-data-generating-the-data-1}
%%tab jax
def synthetic_data(w, b, noise, n, key):  #@save
    """n examples with standard normal features and labels Xw + b + noise."""
    key1, key2 = jax.random.split(key)
    X = jax.random.normal(key1, (n, w.shape[0]))
    eps = jax.random.normal(key2, (n, 1)) * noise
    return X, d2l.matmul(X, d2l.reshape(w, (-1, 1))) + b + eps
```

Let us draw $2{,}000$ examples using $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, and
$\sigma = 0.01$: we will reserve the first $1{,}000$ for training and the
remaining $1{,}000$ for validation. Because the validation examples come from the
exact same distribution but are never touched during fitting, evaluating our
loss on them reveals how accurately the model predicts unseen data—a distinction
we examine closely in :numref:`sec_generalization_basics`.

```{.python .input #synthetic-regression-data-generating-the-dataset-2}
%%tab pytorch, tensorflow, mxnet
w_star, b_star, noise = d2l.tensor([2, -3.4]), 4.2, 0.01
X, y = synthetic_data(w_star, b_star, noise, 2000)
```

```{.python .input #synthetic-regression-data-generating-the-dataset-2}
%%tab jax
w_star, b_star, noise = d2l.tensor([2, -3.4]), 4.2, 0.01
X, y = synthetic_data(w_star, b_star, noise, 2000, d2l.get_key())
```

Each row of `X` stores a two-dimensional feature vector and each row of `y`
stores its corresponding scalar label. Inspecting the very first example—and
backing out its noise $\epsilon^{(1)} = y^{(1)} - (\mathbf{w}^{*\top}\mathbf{x}^{(1)} + b^*)$—gives
us an immediate sanity check.

```{.python .input #synthetic-regression-data-generating-the-dataset-3}
x, label = X[0], y[0, 0]
eps = label - (d2l.reduce_sum(w_star * x) + b_star)
print(f'features: ({float(x[0]):.4f}, {float(x[1]):.4f})')
print(f'label: {float(label):.4f}, noise: {float(eps):.4f}')
```

For our first example, verify the math by hand: $2 x_1^{(1)} - 3.4 x_2^{(1)} + 4.2$ matches $y^{(1)}$ up to the tiny $\mathcal{N}(0, 0.01^2)$ perturbation.

## Inspecting the Data

Before fitting any model, let us plot the label $y$ against each feature $x_1$ and $x_2$ individually:

```{.python .input #synthetic-regression-data-inspecting-the-data}
d2l.set_figsize((7, 2.5))
_, axes = d2l.plt.subplots(1, 2, sharey=True)
for j, ax in enumerate(axes):
    ax.scatter(d2l.numpy(X[:1000, j]), d2l.numpy(y[:1000, 0]), s=2)
    ax.set_xlabel(f'$x_{j + 1}$')
    ax.set_ylabel('$y$' if j == 0 else '')
```

Each scatter plot reveals a clear linear trend with the expected sign: $y$ climbs with $x_1$ (slope $+2$) and falls more steeply with $x_2$ (slope $-3.4$). Why does each 2D scatter plot look like a thick band rather than a razor-thin line when $\sigma = 0.01$ is so tiny? Because plotting $y$ against $x_2$ alone ignores each point's variation in $x_1$: the unplotted term $2 x_1$ has standard deviation $2$, which spreads the vertical position of the points across a band several units wide! Viewing marginal 2D projections of a multivariate relationship always folds the other coordinates into apparent vertical scatter.

## Recovering the Parameters
:label:`subsec_synthetic-regression-data-recovery`

Since we derived the closed-form normal equations in :numref:`subsec_linear-regression-analytic-solution`, let us compute the exact least-squares solution $(\hat{\mathbf{w}}, \hat{b})$ on our 1,000 examples right now using `lstsq` on the augmented design matrix $[\mathbf{X}, \mathbf{1}]$. This gives us the gold-standard finite-sample fit that minibatch SGD in :numref:`sec_linear_scratch` will aim to match:

```{.python .input #synthetic-regression-data-recovering-the-parameters-1}
%%tab pytorch
n = 1000                                          # The training examples
X1 = torch.cat((X[:n], torch.ones(n, 1)), 1)      # A column of ones for b
w_hat = torch.linalg.lstsq(X1, y[:n]).solution[:, 0]
```

```{.python .input #synthetic-regression-data-recovering-the-parameters-1}
%%tab tensorflow
n = 1000                                          # The training examples
X1 = tf.concat((X[:n], tf.ones((n, 1))), 1)       # A column of ones for b
w_hat = tf.linalg.lstsq(X1, y[:n], fast=False)[:, 0]
```

```{.python .input #synthetic-regression-data-recovering-the-parameters-1}
%%tab jax
n = 1000                                          # The training examples
X1 = jnp.concatenate((X[:n], jnp.ones((n, 1))), 1)  # A column of ones for b
w_hat = jnp.linalg.lstsq(X1, y[:n])[0][:, 0]
```

```{.python .input #synthetic-regression-data-recovering-the-parameters-1}
%%tab mxnet
n = 1000                                          # The training examples
X1 = mnp.concatenate((X[:n], mnp.ones((n, 1))), 1)  # A column of ones for b
w_hat = mnp.linalg.lstsq(X1, y[:n], rcond=None)[0][:, 0]
```

Now we print the actual estimation errors alongside our predicted standard error
$\sigma / \sqrt{n}$.

```{.python .input #synthetic-regression-data-recovering-the-parameters-2}
e_w = d2l.numpy(w_hat[:2] - w_star)
print(f'error in w: ({e_w[0]:.1e}, {e_w[1]:.1e})')
print(f'error in b: {float(w_hat[2] - b_star):.1e}')
print(f'standard error: {noise / n ** 0.5:.1e}')
```

Even with `lstsq` solving the empirical loss to machine precision, $(\hat{\mathbf{w}}, \hat{b})$ differs from the true $(2, -3.4, 4.2)$ in the fourth decimal place. That tiny discrepancy is *statistical estimation error*, not numerical error: the 1,000 noise draws $\boldsymbol{\epsilon}$ happen to correlate slightly with the columns of $\mathbf{X}$, tilting the least-squares plane by a quantity of order $\sigma / \sqrt{n} = 0.01 / \sqrt{1000} \approx 3 \times 10^{-4}$ (Exercise :exref:`ex_synthetic-regression-data-recovery-under-noise`). No algorithm trained on these 1,000 noisy points can recover $(\mathbf{w}^*, b^*)$ more accurately than this statistical limit. And as Exercise :exref:`ex_synthetic-regression-data-misspecified` explores, when the true data-generating function is nonlinear, least squares finds the best linear approximation to it over the feature distribution.

## Summary

- **Synthetic benchmarks:** Generating data from a known ground-truth model $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ with $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, and $\sigma = 0.01$ lets us verify parameter recovery and loss convergence directly against known truth.
- **Marginal projections:** Plotting $y$ against a single feature $x_j$ folds the variation of the remaining features into the vertical spread, making a nearly noiseless hyperplane look like a wide band in 2D.
- **Statistical estimation floor:** On $n$ examples with noise standard deviation $\sigma$, even the exact least-squares solution deviates from $(\mathbf{w}^*, b^*)$ by $O(\sigma / \sqrt{n})$.

## Exercises

1. [code] **Recovery under noise.** ● Repeat the least-squares fit of
   :numref:`subsec_synthetic-regression-data-recovery` on 1,000 independent
   datasets for each setting below, and measure the standard deviation of
   the estimation error $\hat{w}_1 - w_1^*$ of the first weight.
    1. Vary `noise` over $\{0.001, 0.01, 0.1, 1\}$ with 1,000 training
       examples, and the number $n$ of training examples over
       $\{10, 100, 1{,}000, 10{,}000\}$ with `noise=0.01`. Plot both measurements against the prediction
       $\sigma/\sqrt{n}$ on logarithmic axes.
    1. Find the smallest $n$ at which the measurement agrees with
       $\sigma/\sqrt{n}$ to within 10%, and say which approximation behind
       that prediction fails for fewer examples.
    1. Write $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}]$,
       $\tilde{\mathbf{w}}^* = (\mathbf{w}^*, b^*)$, and
       $\hat{\tilde{\mathbf{w}}} = (\hat{\mathbf{w}}, \hat{b})$. Show from the
       normal equations that
       $\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}$,
       and derive $\textrm{Cov}(\hat{\tilde{\mathbf{w}}}) = \sigma^2(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}$
       from it. Then generate correlated features
       $x_2 = \rho x_1 + \sqrt{1 - \rho^2}\, z$, with $x_1$ and $z$
       independent and standard normal, for $\rho \in \{0, 0.9, 0.99\}$.
       Predict the standard error of $\hat{w}_1$ from this formula, check
       it numerically, and explain why the predictions
       $\hat{\tilde{\mathbf{w}}}^\top \tilde{\mathbf{x}}$, with $\tilde{\mathbf{x}} = (\mathbf{x}, 1)$, on new examples drawn from the same
       distribution remain accurate when the weights do not.
   :exlabel:`ex_synthetic-regression-data-recovery-under-noise`

1. [code] **A model that does not fit.** ● Add the term $c\,x_1^2$ with
   $c = 0.5$ to every label, so that the labels no longer follow a linear
   function of the features, and fit the linear model by least squares as
   in this section.
    1. Using $\mathbb{E}[x_1^2] = 1$ and $\mathbb{E}[x_1^3] = 0$, predict
       which of $\hat{w}_1$, $\hat{w}_2$, and $\hat{b}$ end up away from
       the true values, and by about how much.
    1. Check the prediction for $n \in \{100, 1{,}000, 10{,}000\}$
       training examples. Which estimation errors shrink as $n$ grows, which
       do not, and how do the shrinking ones compare with $\sigma/\sqrt{n}$?
    1. A test that flags any parameter error above $10^{-3}$, about three
       standard errors, would pass a correct implementation on the data of
       this section. Would it declare a correct implementation faulty on
       these data? Explain what passing or failing such a test can tell you
       when the model does not fit the data.
   :exlabel:`ex_synthetic-regression-data-misspecified`

1. **The loss floor and its fluctuation.** The next section judges a
   training run by its validation loss, the average of
   $\frac{1}{2}(\hat{y} - y)^2$ over the 1,000 validation examples.
    1. Show that a model with the true parameters has expected loss
       $\sigma^2/2$ on a new example, and that no other parameters have a
       smaller one. This is the floor that a correct training run levels
       off at.
    1. For $\epsilon \sim \mathcal{N}(0, \sigma^2)$ one has
       $\mathbb{E}[\epsilon^4] = 3\sigma^4$. Compute the standard deviation
       of the loss $\frac{1}{2}\epsilon^2$ of one example, and from it the
       standard error of the validation loss of the true parameters over
       $n' = 1{,}000$ examples, as a fraction of the floor.
    1. [code] Check both numbers by evaluating the true parameters on many
       independent validation sets. Then use the result of part (a) to find
       the smallest parameter error that raises the expected loss by one
       standard error of the validation loss, and compare it with the
       parameter-error threshold of $10^{-3}$ from the previous exercise.
   :exlabel:`ex_synthetic-regression-data-loss-floor`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/6664)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17975)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.2]{.kicker}

Synthetic Regression Data: Ground-Truth Verification<br>
**known parameters · Gaussian generator · noise floor · statistical estimation error**
:::
:::

::: {.slide title="Why Debug Machine Learning Code on Synthetic Data?"}
[The Debugging Dilemma]{.kicker}

::: {.cols .vc}
::: {.col}
- **Real-world ambiguity:** When loss plateaus on real data, is the model misspecified, are the labels corrupted, or does the code contain a silent bug?
- **Controlled ground truth:** Generating $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ with $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, and $\sigma = 0.01$ makes the linear model and squared loss statistically optimal by construction.
- **Quantitative pass/fail oracle:** Any failure to hit the loss floor $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$ or recover $(\mathbf{w}^*, b^*)$ within $\sim 3\sigma/\sqrt{n} \approx 10^{-3}$ convicts the code.
:::

::: {.col .fig .big}
@fig:ch02-lr-synth-oracle
:::
:::
:::

::: {.slide title="The Ground-Truth Data-Generating Process"}
[Mathematical Specification]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}, \qquad X_{ij} \sim \mathcal{N}(0, 1), \quad \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$

- **True parameters & high SNR:** Weights $\mathbf{w}^* = (2, -3.4)^\top$, bias $b^* = 4.2$, and signal standard deviation $\sqrt{2^2 + (-3.4)^2} \approx 3.94$ ($400\times$ larger than noise $\sigma = 0.01$).
- **Train/validation split:** Drawing $2{,}000$ i.i.d. examples provides $1{,}000$ training examples for fitting and $1{,}000$ validation examples for out-of-sample evaluation.
:::

::: {.col .fig}
@fig:ch02-lr-synth-dgp
:::
:::
:::

::: {.slide title="Implementing the Synthetic Generator"}
[Vectorized Data Generation]{.kicker}

::: {.cols .vc}
::: {.col}
@synthetic-regression-data-generating-the-data-1

- **Explicit column shape `(-1, 1)`:** Reshaping `w` to `(d, 1)` ensures `matmul(X, w)` produces an `(n, 1)` column matching `eps` of shape `(n, 1)`, avoiding an `(n, n)` outer-sum broadcasting catastrophe.
:::

::: {.col .fig}
@fig:ch02-lr-synth-shapes
:::
:::
:::

::: {.slide title="Explicit PRNG Key Splitting in JAX" only="jax"}
[Stateless Randomness · JAX]{.kicker}

::: {.cols .vc}
::: {.col}
```{.python #synthetic-regression-data-exercises}
key1, key2 = jax.random.split(key)
X = jax.random.normal(key1, (n, w.shape[0]))
eps = jax.random.normal(key2, (n, 1)) * noise
```

- **Why split keys:** Passing the *same* `key` to both `X` and `eps` would draw identical normal variates; `jax.random.split` deterministically forks independent subkeys `key1` and `key2`.
:::

::: {.col .fig}
@fig:ch02-lr-jax-key-split
:::
:::
:::

::: {.slide title="Inspecting Generated Tensors and Realized Noise"}
[Sanity-Checking the Data]{.kicker}

We draw $2{,}000$ examples and inspect the first training pair $(\mathbf{x}^{(1)}, y^{(1)})$ alongside its realized noise $\epsilon^{(1)} = y^{(1)} - (\mathbf{w}^{*\top}\mathbf{x}^{(1)} + b^*)$:

@synthetic-regression-data-generating-the-dataset-2

@synthetic-regression-data-generating-the-dataset-3

::: {.d2l-note}
Verify the arithmetic directly: $2 x_1^{(1)} - 3.4 x_2^{(1)} + 4.2$ matches the observed label $y^{(1)}$ up to the tiny Gaussian perturbation $\epsilon^{(1)}$ of order $\sigma = 0.01$.
:::
:::

::: {.slide title="Why 2D Marginal Scatter Plots Look Like Thick Bands"}
[Geometric Intuition]{.kicker}

::: {.cols .vc}
::: {.col}
If $\sigma = 0.01$ is tiny, why does plotting $y$ against $x_1$ and $x_2$ show thick vertical clouds several units wide rather than razor-thin lines?

- **Omitted-variable projection:** In 3D space $(x_1, x_2, y)$, all $1{,}000$ points lie within $\pm 0.03$ of the tilted plane $y = 2x_1 - 3.4x_2 + 4.2$.
- **Unplotted feature variance:** Projecting onto $(x_2, y)$ folds each point's unplotted coordinate $2x_1 \sim \mathcal{N}(0, 4)$ into the vertical spread ($\textrm{std} = 2.0$), while projecting onto $(x_1, y)$ folds in $-3.4x_2$ ($\textrm{std} = 3.4$).
:::

::: {.col .fig}
@!synthetic-regression-data-inspecting-the-data
:::
:::

::: {.d2l-note .rule}
Marginal 2D scatter plots of a multivariate function always fold the variation of the unplotted features into the vertical scatter.
:::
:::

::: {.slide title="The Irreducible Loss Floor of the True Parameters"}
[Theoretical Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}\bigl[L(\mathbf{w}^*, b^*)\bigr] = \frac{1}{2}\,\mathbb{E}\bigl[(-\epsilon)^2\bigr] = \frac{\sigma^2}{2} = 5 \times 10^{-5}$$

- **Irreducible noise floor:** Because future noise draws $\epsilon \sim \mathcal{N}(0, \sigma^2)$ are independent of $\mathbf{x}$, *no* predictor can achieve an expected validation loss below $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$.
- **Finite-sample fluctuation:** Over $n' = 1{,}000$ validation examples, the sample average of $\frac{1}{2}\epsilon^2$ fluctuates around $5 \times 10^{-5}$ with standard error $\sigma^2/\sqrt{2n'} \approx 2.2 \times 10^{-6}$.
:::

::: {.col .fig}
@fig:ch02-lr-noise-floor
:::
:::
:::

::: {.slide title="Recovery I: Estimation Error Is Linear in the Noise"}
[Statistical Recovery Analysis]{.kicker}

::: {.cols .vc}
::: {.col}
$$\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* = \bigl(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}\bigr)^{-1}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}$$

- **Signal cancellation:** Substituting $\mathbf{y} = \tilde{\mathbf{X}}\tilde{\mathbf{w}}^* + \boldsymbol{\epsilon}$ into the normal equations cancels the true parameter vector $\tilde{\mathbf{w}}^*$.
- **Unbiasedness & noiseless limit:** Because $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$, $\hat{\tilde{\mathbf{w}}}$ is unbiased ($\mathbb{E}[\hat{\tilde{\mathbf{w}}}] = \tilde{\mathbf{w}}^*$) and recovers $\tilde{\mathbf{w}}^*$ to machine precision when $\sigma = 0$.
:::

::: {.col .fig}
@fig:ch02-lr-recovery-decomp
:::
:::
:::

::: {.slide title="Recovery II: Isotropic Features Make X̃ᵀX̃ ≈ nI"}
[Law of Large Numbers on the Gram Matrix]{.kicker}

::: {.cols .vc}
::: {.col}
$$\frac{1}{n}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} \approx \mathbf{I}_3 \quad\Longrightarrow\quad \hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* \approx \frac{1}{n}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}$$

- **Diagonal concentration:** Sample variances $\frac{1}{n}\sum_{i=1}^n (x_j^{(i)})^2$ concentrate tightly around $\mathbb{E}[x_j^2] = 1$.
- **Off-diagonal decoupling:** Sample cross-products $\frac{1}{n}\sum_{i=1}^n x_j^{(i)} x_k^{(i)}$ concentrate around $\mathbb{E}[x_j x_k] = 0$, decoupling the three parameter errors.
:::

::: {.col .fig}
@fig:ch02-lr-gram-concentration
:::
:::
:::

::: {.slide title="Recovery III: The σ/√n Standard Error Floor"}
[The Statistical Limit]{.kicker}

::: {.cols .vc}
::: {.col}
$$\operatorname{SE}(\hat{w}_j) = \operatorname{SE}(\hat{b}) \approx \frac{\sigma}{\sqrt{n}} = \frac{0.01}{\sqrt{1000}} \approx 3.16 \times 10^{-4}$$

- **Square-root sample scaling:** Reducing estimation error by $10\times$ requires $100\times$ more training examples ($n = 100{,}000$), whereas reducing $\sigma$ by $10\times$ shrinks error by $10\times$ immediately.
- **Exact covariance:** $\operatorname{Cov}(\hat{\tilde{\mathbf{w}}} \mid \tilde{\mathbf{X}}) = \sigma^2 (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}$ inflates when features are collinear.
:::

::: {.col .fig}
@fig:ch02-lr-se-scaling
:::
:::
:::

::: {.slide title="Verifying the Least-Squares Reference Fit in Code"}
[Gold-Standard Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
We solve the normal equations on our $n = 1{,}000$ training examples with `lstsq` and compare actual parameter errors against $\sigma / \sqrt{n} \approx 3 \times 10^{-4}$:

@synthetic-regression-data-recovering-the-parameters-1

@synthetic-regression-data-recovering-the-parameters-2

- **Pass/fail tolerance:** Any converged iterative optimizer in §2.3–§2.5 must land within $\sim 10^{-3}$ ($\approx 3\text{ SE}$) of $(\mathbf{w}^*, b^*)$.
:::

::: {.col .fig}
@fig:ch02-lr-lstsq-reference
:::
:::
:::

::: {.slide title="Concept Check: Loss Floors, Sample Size & Collinearity"}
[Check Your Understanding]{.kicker}

Predict the quantitative effect of each modification to our synthetic benchmark:

1. **Sample size vs. noise:** What is the standard error of $\hat{w}_1$ when (a) $n = 100, \sigma = 0.01$, and (b) $n = 1{,}000, \sigma = 0.1$?
2. **Misspecified quadratic term:** Suppose the true labels are generated by $y = 2x_1 - 3.4x_2 + 4.2 + 0.5x_1^2 + \epsilon$ with $x_1, x_2 \sim \mathcal{N}(0, 1)$, and we fit the linear model $\hat{y} = w_1 x_1 + w_2 x_2 + b$. Which parameter absorbs the unmodeled $0.5x_1^2$ term as $n \to \infty$?
3. **Correlated features:** If $x_2 = 0.99 x_1 + \sqrt{1 - 0.99^2}\,z$, why do individual weight errors $|\hat{w}_j - w_j^*|$ blow up by $\sim 7\times$ while predictions $\hat{\mathbf{w}}^\top\mathbf{x}$ on new test points from the same distribution remain accurate?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. (a) $\sigma / \sqrt{n} = 0.01 / \sqrt{100} = 10^{-3}$; (b) $0.1 / \sqrt{1000} \approx 3.16 \times 10^{-3}$.
2. Because $\mathbb{E}[x_1^2] = 1$ and $\mathbb{E}[x_1 \cdot x_1^2] = \mathbb{E}[x_1^3] = 0$, the quadratic term $0.5x_1^2$ is orthogonal to $x_1$ and $x_2$ but shifts the mean label by $+0.5$, so $\hat{b} \to 4.2 + 0.5 = 4.7$ while $\hat{\mathbf{w}} \to (2, -3.4)^\top$.
3. When $x_1 \approx x_2$, the difference direction $x_1 - x_2$ has tiny variance ($1 - \rho = 0.01$), inflating $\operatorname{Var}(\hat{w}_1 - \hat{w}_2)$, but new test points from the same distribution also have $x_1 \approx x_2$ and only probe the well-determined sum $\hat{w}_1 + \hat{w}_2$.
:::
:::

::: {.slide title="Section 2.2 Summary & Key Takeaways"}
[Summary]{.kicker}

| Benchmark Quantity | Formula / Definition | Value on Running Example | Diagnostic Role in §2.3–§2.5 |
|---|---|---|---|
| **Data Generator** | $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ | $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, $\sigma = 0.01$ | Ground-truth linear testbed ($n = 1{,}000$ train, $1{,}000$ val) |
| **Irreducible Loss Floor** | $\mathbb{E}[L(\mathbf{w}^*, b^*)] = \frac{1}{2}\sigma^2$ | $5 \times 10^{-5}$ (or $\sigma^2 = 10^{-4}$ without $\frac{1}{2}$) | Target plateau for training and validation loss curves |
| **Estimation Error** | $\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}$ | Exact zero when $\sigma = 0$ | Separates statistical noise floor from optimization error |
| **Parameter Standard Error** | $\operatorname{SE}(\hat{w}_j) \approx \sigma / \sqrt{n}$ | $0.01 / \sqrt{1000} \approx 3.2 \times 10^{-4}$ | Sets $3\text{ SE} \approx 10^{-3}$ pass/fail tolerance for recovered weights |
:::
