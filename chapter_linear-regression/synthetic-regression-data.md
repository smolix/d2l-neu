```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Synthetic Data
:label:`sec_synthetic-regression-data`

Machine learning is all about extracting structure from data, so you might
wonder why we would ever bother generating *synthetic* data from a formula we
invented ourselves. Real datasets are certainly what we care about in the end,
yet they make terrible first tests for new code. Suppose we fit a linear model
to real house prices like those in :numref:`sec_linear_regression` and get
disappointing predictions. Three explanations immediately compete: the linear
model may be misspecified for the data, the optimization algorithm or its
implementation may harbor a bug, or the dataset itself may be noisy or corrupted.
On real data, it is notoriously hard to tell those three culprits apart.

Synthetic data eliminates that ambiguity by construction. Because we control the
data-generating process, we know the exact ground-truth weights $\mathbf{w}^*$,
the true bias $b^*$, and the distribution of the observation noise. If we fit a
correctly specified model to these data, any systematic failure to recover
$\mathbf{w}^*$ and $b^*$ beyond what random noise explains points directly to a
flaw in our optimization or code. Testing a new algorithm first on a controlled
synthetic problem whose answer you know—and only then moving to real data—is a
habit that pays dividends throughout deep learning.

In this section, we build just such a benchmark for linear regression. We
generate examples from a known linear function with parameters
$\mathbf{w}^* = (2, -3.4)$ and $b^* = 4.2$ plus small Gaussian noise, and then
derive *how close* a correct fit can come to recovering those parameters despite
the noise. That calculation turns parameter recovery into a quantitative test
with a concrete threshold—which we will hold our from-scratch training loop to
in the next section.

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

To keep our parameters easy to inspect and compare, we work with two-dimensional
features drawn independently from a standard normal distribution $\mathcal{N}(0, 1)$,
stacked as the rows of the design matrix $\mathbf{X}$ (nothing in our derivation
depends on having only two features). We generate the corresponding labels by
evaluating the ground-truth affine function and adding independent Gaussian noise:

$$\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}.$$

Here $b^*$ broadcasts across all rows, and each entry of $\boldsymbol{\epsilon}$
is drawn independently from $\mathcal{N}(0, \sigma^2)$ with $\sigma = 0.01$.
Every label therefore lies close to the true plane
$\mathbf{w}^{*\top}\mathbf{x} + b^*$, typically within a few hundredths. Notice
that this data-generating process matches the Gaussian noise model of
:numref:`subsec_normal_distribution_and_squared_loss` exactly, so minimizing the
squared loss is maximum likelihood estimation. A linear model trained with
squared error is correctly specified by design, ruling out model misspecification
right from the start.

We package this generator into a helper function `synthetic_data` and save it to
the `d2l` library for reuse in upcoming sections. Given the true parameters `w`
and `b`, the noise standard deviation `noise`, and the sample count `n`, it draws
the feature matrix $\mathbf{X}$ first, draws the noise vector $\boldsymbol{\epsilon}$
second, and returns `(X, y)`.

:begin_tab:`jax`
Because JAX has no hidden global random state, `synthetic_data` takes an explicit
PRNG `key`, splits it into two independent keys for `X` and `eps`, and guarantees
that passing the same `key` always reproduces the exact same dataset. Calling
`d2l.get_key()` hands us a fresh key on each invocation from a stream seeded anew
on every run.
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

Just as expected, the label sits within a few hundredths of the noise-free linear
plane.

## Recovering the Parameters
:label:`subsec_synthetic-regression-data-recovery`

Before we can use these data to validate a training loop, we need a clear
pass/fail threshold. Even a flawless implementation cannot recover $\mathbf{w}^*$
and $b^*$ to machine precision, because the random noise $\boldsymbol{\epsilon}$
perturbs the labels and shifts the minimizer of the training loss slightly away
from the true parameters. How large is that unavoidable shift?

The normal equations from :numref:`subsec_linear-regression-analytic-solution`
give us the exact answer. Appending a column of ones to the $n$ training examples
and folding $b^*$ into the weight vector gives the three-column matrix
$\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}]$, the true parameter vector
$\tilde{\mathbf{w}}^* = (2, -3.4, 4.2)$, and
$\mathbf{y} = \tilde{\mathbf{X}}\tilde{\mathbf{w}}^* + \boldsymbol{\epsilon}$.
Substituting $\mathbf{y}$ into the normal equations
$\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} \hat{\tilde{\mathbf{w}}} = \tilde{\mathbf{X}}^\top \mathbf{y}$
yields
$\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} (\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^*) = \tilde{\mathbf{X}}^\top \boldsymbol{\epsilon}$,
and therefore

$$\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}.$$

The *estimation error* $\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^*$ is a
purely linear function of the noise $\boldsymbol{\epsilon}$: when $\sigma = 0$,
least squares recovers $\tilde{\mathbf{w}}^*$ with zero error.

For arbitrary datasets, reasoning about $(\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}})^{-1}$
can be messy, but for our standard normal features the matrix is remarkably
clean. Each entry of $\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} / n$ is the
sample average over $n$ examples of a product of two columns:

* On the diagonal, we average either $1^2$ or the square of a standard normal
  feature $x_j^2$, both of which have expectation $1$.
* Off the diagonal, we average the product of a zero-mean feature with $1$ or
  with another independent feature, both of which have expectation $0$.

Across $n = 1{,}000$ independent examples, these sample averages sit within a few
percent of their expectations, so
$\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} \approx n \mathbf{I}$. Substituting
this approximation turns the estimation error into a simple average over the
training set:

$$\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* \approx \frac{1}{n}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon} = \frac{1}{n}\sum_{i=1}^{n} \epsilon^{(i)}\tilde{\mathbf{x}}^{(i)},$$

where $\tilde{\mathbf{x}}^{(i)} = (\mathbf{x}^{(i)}, 1)$ is the $i$-th row of
$\tilde{\mathbf{X}}$. Each coordinate $j$ averages $n$ independent terms
$\epsilon^{(i)} \tilde{x}_j^{(i)}$, each with mean $0$ and variance
$\sigma^2 \mathbb{E}[(\tilde{x}_j^{(i)})^2] = \sigma^2$ (since $\epsilon^{(i)}$
is independent of the features). By our standard-error formula from
:numref:`sec_prob`, this average has standard error $\sigma / \sqrt{n}$. Plugging
in $\sigma = 0.01$ and $n = 1{,}000$ predicts that every fitted parameter—both
weights and the bias—will miss its true value by roughly
$0.01 / \sqrt{1000} \approx 3 \times 10^{-4}$.

Let us test this theoretical prediction directly by solving the least-squares
problem on our $1{,}000$ training examples using `lstsq` (storing
$\tilde{\mathbf{X}}$ in `X1` and $\hat{\tilde{\mathbf{w}}}$ in `w_hat`).

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

All three errors sit right around a few times $10^{-4}$, matching the predicted
standard error. By our three-sigma rule of thumb (:numref:`sec_prob`), an error
exceeding $10^{-3}$ (more than three standard errors) is unlikely under pure
sampling noise. That gives us the concrete verification threshold for the next
section: although minibatch SGD approaches the least-squares solution iteratively
and hovers in a small noise ball around it (:numref:`sec_linear_regression`), a
bug-free implementation trained to convergence should still land within roughly
$10^{-3}$ of $\mathbf{w}^*$ and $b^*$. By contrast, an error of $0.1$ cannot be
blamed on random noise—it exposes either a bug in the code or an optimizer that
has not converged. Naturally, this threshold scales with the noise level and
sample size (exercise :exref:`ex_synthetic-regression-data-recovery-under-noise`)
and assumes unit-scale features (exercise
:exref:`ex_synthetic-regression-data-feature-scale`).

Keep in mind the boundary of what a synthetic test can guarantee. Passing it
confirms that our optimization code accurately recovers parameters when the model
matches the data-generating process, but it tells us nothing about how the model
will behave when the true relationship is *not* linear—the everyday reality of
real-world datasets (exercise :exref:`ex_synthetic-regression-data-misspecified`).

## Summary

* Poor performance on real data leaves us guessing between model
  misspecification, optimization bugs, and messy data; synthetic data with known
  ground-truth parameters $\mathbf{w}^*$ and $b^*$ isolates the algorithm and
  implementation.
* Generating labels from $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$
  with independent standard normal features and Gaussian noise
  $\epsilon \sim \mathcal{N}(0, \sigma^2)$ ensures that linear regression with
  squared loss is correctly specified.
* Even the exact least-squares estimator deviates from the true parameters by a
  random estimation error whose standard error is $\sigma / \sqrt{n}$ (here
  $0.01 / \sqrt{1000} \approx 3 \times 10^{-4}$). A converged SGD implementation
  should therefore land within roughly $10^{-3}$ of the ground truth—while
  remembering that passing this test validates the code, not the modeling
  assumptions on real data.


## Exercises

:begin_tab:`mxnet`
1. [code] **Reproducibility.** Two calls of `synthetic_data` with the same
   arguments return different `X` and `y`. Make them identical
   by fixing the seed with `npx.random.seed`. State where the call must be
   placed relative to the draws of `X` and `eps`, and explain why seeding
   once at the top of a notebook does not make the second dataset equal to
   the first.
   :exlabel:`ex_synthetic-regression-data-reproducibility`
:end_tab:

:begin_tab:`pytorch`
1. [code] **Reproducibility.** Two calls of `synthetic_data` with the same
   arguments return different `X` and `y`. Make them identical
   by fixing the seed with `torch.manual_seed`. State where the call must be
   placed relative to the draws of `X` and `eps`, and explain why seeding
   once at the top of a notebook does not make the second dataset equal to
   the first.
   :exlabel:`ex_synthetic-regression-data-reproducibility`
:end_tab:

:begin_tab:`tensorflow`
1. [code] **Reproducibility.** Two calls of `synthetic_data` with the same
   arguments return different `X` and `y`. Make them identical
   by fixing the seed with `tf.random.set_seed`. State where the call must
   be placed relative to the draws of `X` and `eps`, and explain why seeding
   once at the top of a notebook does not make the second dataset equal to
   the first.
   :exlabel:`ex_synthetic-regression-data-reproducibility`
:end_tab:

:begin_tab:`jax`
1. [code] **Reproducibility.** `synthetic_data` splits its `key` into
   `key1` for `X` and `key2` for `eps`.
    1. Explain why two calls with the same `key` return identical data
       without any global seed, and why two calls with keys from
       `d2l.get_key()` do not.
    1. Change both draws to use `key` itself instead of `key1` and `key2`,
       and compare `eps / noise` with the entries of `X` numerically. State
       what the split guarantees about `X` and `eps` that this version does
       not.
    1. Construct five datasets that are independent of each other, as
       averaging an error over repeated trials requires, and state how you
       derive their keys.
   :exlabel:`ex_synthetic-regression-data-reproducibility`
:end_tab:

2. [code] **Recovery under noise.** ● Repeat the least-squares fit of
   :numref:`subsec_synthetic-regression-data-recovery` on 1,000 independent
   datasets for each setting below, and measure the standard deviation of
   the estimation error $\hat{w}_1 - w_1^*$ of the first weight.
    1. Vary `noise` over $\{0.001, 0.01, 0.1, 1\}$ with 1,000 training
       examples, and the number $n$ of training examples over
       $\{10, 100, 1{,}000, 10{,}000\}$ with `noise=0.01`. Plot both measurements against the prediction
       $\sigma/\sqrt{n}$ on logarithmic axes.
    1. Find the smallest $n$ at which the measurement agrees with
       $\sigma/\sqrt{n}$ to within 10%, and say which step of the derivation
       fails for fewer examples.
    1. Derive $\textrm{Cov}(\hat{\tilde{\mathbf{w}}}) = \sigma^2(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}$
       from the expression for $\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^*$
       in this section. Then generate correlated features
       $x_2 = \rho x_1 + \sqrt{1 - \rho^2}\, z$, with $x_1$ and $z$
       independent and standard normal, for $\rho \in \{0, 0.9, 0.99\}$.
       Predict the standard error of $\hat{w}_1$ from this formula, check
       it numerically, and explain why the predictions
       $\hat{\tilde{\mathbf{w}}}^\top \tilde{\mathbf{x}}$ on new examples drawn from the same
       distribution remain accurate when the weights do not.
   :exlabel:`ex_synthetic-regression-data-recovery-under-noise`

3. [code] **A feature on another scale.** Draw the first feature with
   standard deviation 10 instead of 1, and set $w_1^* = 0.2$, so that the
   labels are distributed as before.
    1. Predict $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}/n$, say which
       step of the derivation in this section no longer holds, and predict
       the standard errors of $\hat{w}_1$ and $\hat{w}_2$.
    1. Check both predictions by repeating the least-squares fit on many
       independent datasets.
    1. A bug moves $\hat{w}_1$ by $5 \times 10^{-4}$. Does the threshold
       of $10^{-3}$ catch it? Propose a threshold for each weight that does
       not depend on the scale of its feature.
   :exlabel:`ex_synthetic-regression-data-feature-scale`

4. [code] **A model that does not fit.** ● Add the term $c\,x_1^2$ with
   $c = 0.5$ to every label, so that the labels no longer follow a linear
   function of the features, and fit the linear model by least squares as
   in this section.
    1. Using $\mathbb{E}[x_1^2] = 1$ and $\mathbb{E}[x_1^3] = 0$, predict
       which of $\hat{w}_1$, $\hat{w}_2$, and $\hat{b}$ end up away from
       the true values, and by about how much.
    1. Check the prediction for $n \in \{100, 1{,}000, 10{,}000\}$
       training examples. Which estimation errors shrink as $n$ grows, which
       do not, and how do the shrinking ones compare with $\sigma/\sqrt{n}$?
    1. Would the threshold of $10^{-3}$ declare a correct implementation
       faulty on these data? Explain what passing or failing the test of
       this section can tell you when the model does not fit the data.
   :exlabel:`ex_synthetic-regression-data-misspecified`

5. **The loss floor and its fluctuation.** The next section judges a
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
       threshold of $10^{-3}$.
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

A dataset with a known answer<br>**generator · error threshold**
:::
:::

::: {.slide title="A Poor Fit on Real Data Has Three Suspects"}
[The problem]{.kicker}

Suppose a model reaches a poor validation loss on real data. Is the model
wrong, the optimization or its code, or the data?

- **Real data:** the true parameters are unknown, and the three
  explanations are hard to tell apart.
- **Synthetic data:** we choose $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$,
  and the noise, so model and data are right by construction; only the
  optimization or its implementation can fail.
- A fit of $\hat{\mathbf{w}} = (1.9, -3.1)$ is then $0.3$ off, where the
  noise allows about $0.001$.

::: {.d2l-note}
A fit that misses the known $\mathbf{w}^*$ by more than the noise allows
points to a bug, or to an optimization that has not converged.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- generate $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$
  with one function, `synthetic_data`;
- predict how close a correct fit comes to $\mathbf{w}^*$: within a few
  standard errors $\sigma/\sqrt{n}$.

**Running example:** $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, noise
$\sigma = 0.01$; 1,000 training and 1,000 validation examples.
:::

::: {.slide title="Known Parameters Turn Training Into a Test"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Linear Regression | model, loss, closed form, minibatch SGD |
| **Synthetic Data** | **data with known $\mathbf{w}^*, b^*$** |
| Implementation from Scratch | SGD and the training loop by hand |
| Object-Oriented Design | `Module`, `DataModule`, `Trainer` |
| Concise Implementation | framework layers and losses |
| Generalization | training vs. test error |
| Weight Decay | a penalty on large weights |
:::

::: {.col .narrow}
- **Uses:** the normal equations, the standard error
  $\sigma/\sqrt{n}$ of an average.
- **Skips:** minibatches and the training loop (next deck), the classes
  (the deck after).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Generating the data]{.dtitle}

[a known plane · a little noise · one function]{.dsub}
:::
:::

::: {.slide title="Labels Lie on a Known Plane, Up to Noise of About 0.01"}
[Generating the data]{.kicker}

$$\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}, \qquad x_j \sim \mathcal{N}(0, 1), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

- $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, $\sigma = 0.01$; $b^*$ is added
  to every label.
- Gaussian noise makes the squared loss the maximum-likelihood loss, so a
  linear model is correctly specified.
- `synthetic_data` draws 2,000 examples: 1,000 to train on, 1,000 to
  validate with.
:::

::: {.slide title="The First Label Is Off the Plane by the Noise Alone"}
[Generating the data]{.kicker}

@synthetic-regression-data-generating-the-dataset-2

The first example, and its noise $y - (\mathbf{w}^{*\top}\mathbf{x} + b^*)$:

@!synthetic-regression-data-generating-the-dataset-3

::: {.d2l-note}
The noise is at most a few hundredths, since $\sigma = 0.01$.
:::
:::

::: {.slide title="One Key Fixes the Dataset; a New Key Draws a New One" only="jax"}
[Generating the data · JAX]{.kicker}

JAX has no global random state: every draw takes a key.

```{.python #synthetic-regression-data-exercises}
key1, key2 = jax.random.split(key)
X = jax.random.normal(key1, (n, w.shape[0]))
eps = jax.random.normal(key2, (n, 1)) * noise
```

- $\mathbf{X}$ and $\boldsymbol{\epsilon}$ come from two different keys.
- The same key always produces the same dataset; `d2l.get_key()` gives a
  new key on every call, from a stream seeded anew on every run.
:::

::: {.slide title="Check: Which Loss Do the True Parameters Reach?"}
[Check]{.kicker}

A model that predicts with exactly $\mathbf{w}^* = (2, -3.4)$ and
$b^* = 4.2$ is scored with the squared loss
$\tfrac{1}{2}(\hat{y} - y)^2$, averaged over the examples. About which
value does that average settle?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\hat{y} - y = -\epsilon$, so the average approaches
$\tfrac{1}{2}\mathbb{E}[\epsilon^2] = \sigma^2/2 = 5 \times 10^{-5}$. The noise is
unpredictable: no model does better on new examples, and a correct
training run levels off there.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Recovering the parameters]{.dtitle}

[how close a correct fit can come]{.dsub}
:::
:::

::: {.slide title="I. The Estimation Error Is Linear in the Noise"}
[Recovering the parameters · derivation]{.kicker}

Append a column of ones: $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}]$,
$\tilde{\mathbf{w}}^* = (2, -3.4, 4.2)$, and
$\mathbf{y} = \tilde{\mathbf{X}}\tilde{\mathbf{w}}^* + \boldsymbol{\epsilon}$.

Least squares solves
$\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}} = \tilde{\mathbf{X}}^\top\mathbf{y}$.
Substitute $\mathbf{y}$:
$\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}(\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^*) = \tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}$, so

$$\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}.$$

::: {.d2l-note}
Without noise, least squares recovers $\tilde{\mathbf{w}}^*$ exactly. The
noise moves the minimizer of the loss, and we need to know how far.
:::
:::

::: {.slide title="II. Standard Normal Features Make X̃ᵀX̃ ≈ 1,000 · I"}
[Recovering the parameters · derivation]{.kicker}

Each entry of $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}/n$ averages
$n = 1{,}000$ products of two columns:

| entry | product | its mean |
|---|---|---|
| diagonal | $x_j^2$ or $1 \cdot 1$ | 1 |
| off the diagonal | $x_j x_k$ or $x_j \cdot 1$ | 0 |

An average over 1,000 examples lies within a few percent of its mean. So

$$\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* \approx \frac{1}{n}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon} = \frac{1}{n}\sum_{i=1}^{n} \epsilon^{(i)}\tilde{\mathbf{x}}^{(i)}, \qquad \tilde{\mathbf{x}}^{(i)} = (\mathbf{x}^{(i)}, 1).$$
:::

::: {.slide title="III. Each Coordinate Misses by About σ/√n = 3 × 10⁻⁴"}
[Recovering the parameters · derivation]{.kicker}

Coordinate $j$ of the error averages $n$ independent terms
$\epsilon^{(i)} \tilde{x}_j^{(i)}$, each with mean 0 and variance
$\sigma^2 \mathbb{E}[\tilde{x}_j^2] = \sigma^2$. An average of $n$ such terms has standard
deviation $\sigma/\sqrt{n}$, its **standard error**:

::: {.d2l-note .rule}
$$\sigma/\sqrt{n} = 0.01/\sqrt{1000} \approx 3 \times 10^{-4}$$
for each weight and for the bias.
:::

Ten times more noise, ten times the error; a hundred times more examples, a tenth of it.

[Assumes: noise independent of the features, $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} \approx n\mathbf{I}$.]{.dgm-cap}
:::

::: {.slide title="Least Squares Lands Within a Few 10⁻⁴ of the Truth"}
[Recovering the parameters]{.kicker}

`lstsq` on the 1,000 training examples, with a column of ones for $b$:

@!synthetic-regression-data-recovering-the-parameters-2

- Least squares: an error above $10^{-3}$, over three standard errors,
  would be rare.
- SGD fluctuates around this estimate by a similar amount, so a converged
  model lands within about $10^{-3}$.

::: {.d2l-note .rule}
An error of $0.1$ means a bug, or an optimization that has not converged.
:::
:::

::: {.slide title="Check: How Close With Fewer Examples or More Noise?"}
[Check]{.kicker}

What standard error do the weights have with

1. 100 training examples and $\sigma = 0.01$?
2. 1,000 training examples and $\sigma = 0.1$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.**

1. $0.01/\sqrt{100} = 10^{-3}$: ten times fewer examples, a
   $\sqrt{10} \approx 3.2$ times larger error.
2. $0.1/\sqrt{1000} \approx 3 \times 10^{-3}$: the error scales with
   $\sigma$.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| step | running example |
|---|---|
| generate $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ | $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, $\sigma = 0.01$ |
| standard error $\sigma/\sqrt{n}$ | $0.01/\sqrt{1000} \approx 3 \times 10^{-4}$; least-squares errors of that size |
| threshold for training | a converged model within about $10^{-3}$ |
| what passing shows | the code recovers what it should; nothing about data a linear model does not fit |

**Next:** the model, the loss, and minibatch SGD written as plain
functions, trained on these data, their estimates held to $10^{-3}$.
:::
