```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Weight Decay
:label:`sec_weight_decay`

In :numref:`sec_generalization_basics`, we controlled model complexity in the bluntest way possible: by changing the discrete number of polynomial features $k + 1$. Yet discrete feature selection quickly hits a wall. In high-dimensional problems with hundreds or thousands of features, there are $2^d$ possible feature subsets, and worse still, we saw that when a polynomial overfits, its weights $w_j$ explode into the millions and beyond to cancel almost-collinear features against one another.

To see the core pathology in its simplest two-feature form, imagine predicting room temperature from two noisy thermometers placed side by side, $x_1$ and $x_2$, each reading the true temperature plus a tiny independent noise wiggle. Consider two linear models whose weights sum to $2$:

- **Model A** sets $\mathbf{w} = (1, 1)^\top$, averaging the two sensors and cutting the sensor noise variance in half.
- **Model B** sets $\mathbf{w} = (501, -499)^\top$. Whenever the two thermometers agree, $501 x_1 - 499 x_2 = 2 x_1$ gives the same average, yet any tiny discrepancy $x_1 - x_2$ between the two sensors gets amplified by $500$.

On a small training sample where $x_1 - x_2$ happened by chance to correlate slightly with the label noise, unregularized least squares will choose $(501, -499)^\top$ to shave a sliver off the training error, only to fail badly on new readings. How can we keep all $d$ features in the model while preventing the weights from blowing up along poorly determined directions?

The classic remedy, known as *ridge regression* or *Tikhonov regularization* in statistics :cite:`Hoerl.Kennard.1970,tikhonov1977solutions` and *weight decay* in deep learning, adds a smooth penalty on the squared Euclidean norm $\|\mathbf{w}\|_2^2$ directly to the loss function. In this section, we derive weight decay from four complementary perspectives (penalized optimization, per-step multiplicative decay in SGD, Bayesian Gaussian priors, and spectral shrinkage along eigenvectors of $\mathbf{X}^\top\mathbf{X}$) and implement it both from scratch and concisely in all four frameworks.

```{.python .input #weight-decay}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #weight-decay}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #weight-decay}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #weight-decay}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```


## Norm Penalties

### The Penalized Loss

Rather than restricting the number of features, we control a linear model's effective capacity by penalizing the squared $\ell_2$ norm of its weight vector $\|\mathbf{w}\|_2^2 = \sum_{j=1}^d w_j^2$. Adding this penalty to the mean squared error loss $L(\mathbf{w}, b)$ with a nonnegative *regularization constant* $\lambda \geq 0$ yields the regularized objective:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2.$$
:eqlabel:`eq_wd-objective`

Once again, the factor of $\frac{1}{2}$ cancels the $2$ when we differentiate $\|\mathbf{w}\|_2^2$. Notice three basic properties of :eqref:`eq_wd-objective`:

1. **Trade-off via $\lambda$:** Setting $\lambda = 0$ recovers unregularized least squares; increasing $\lambda$ places a steeper tax on large weights, pulling $\mathbf{w}$ toward $\mathbf{0}$ (and driving $\mathbf{w} \to \mathbf{0}$ as $\lambda \to \infty$).
2. **Why we leave the bias $b$ unpenalized:** The bias $b$ simply shifts the average output level to match the mean of the target $y$; shrinking $b$ toward zero would bias every prediction toward zero if $y$ were measured in, say, Kelvin instead of Celsius, without reducing the model's sensitivity to input perturbations at all.
3. **Feature scaling matters:** Because $\sum_j w_j^2$ sums raw squared coefficients across coordinates, re-expressing one feature in millimeters instead of meters divides its weight $w_j$ by $1{,}000$ and cuts its penalty contribution by $10^6$. In practice, one standardizes features to comparable scales (zero mean, unit variance) before applying an isotropic $\|\mathbf{w}\|_2^2$ penalty.

Why penalize the squared $\ell_2$ norm? In our two-thermometer example, $(1, 1)^\top$ has $\|\mathbf{w}\|_2^2 = 2$, whereas $(501, -499)^\top$ has $\|\mathbf{w}\|_2^2 = 500{,}002$, so even a tiny $\lambda > 0$ strongly prefers averaging the two sensors over subtracting them.

### Weight Decay in the Gradient Step

How does the penalty $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ change our minibatch SGD update rule? Since $\nabla_{\mathbf{w}}\left(\frac{\lambda}{2}\|\mathbf{w}\|_2^2\right) = \lambda \mathbf{w}$, the penalized gradient step is

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \left(\frac{\partial L}{\partial \mathbf{w}} + \lambda \mathbf{w}\right) = (1 - \eta \lambda) \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}.$$
:eqlabel:`eq_wd-update`

Look at the first term on the right: **before taking the usual data-driven gradient step $-\eta\,\partial L / \partial \mathbf{w}$, we multiply the weight vector by $(1 - \eta\lambda) < 1$, shrinking ("decaying") it slightly toward zero on every update.** That multiplicative shrinkage factor $(1 - \eta\lambda)$ is why deep learning practitioners call $\ell_2$ regularization *weight decay*. (For plain SGD, adding $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ to the loss and decaying weights by $(1 - \eta\lambda)$ are algebraically identical; for adaptive optimizers like Adam in :numref:`chap_optimization`, decoupling multiplicative weight decay into *AdamW* makes an important difference.)

### The $\ell_1$ Penalty and Sparsity

What if we replaced the squared $\ell_2$ norm $\|\mathbf{w}\|_2^2 = \sum_j w_j^2$ with the $\ell_1$ norm $\|\mathbf{w}\|_1 = \sum_j |w_j|$, a method known in statistics as the *Lasso* :cite:`Tibshirani.1996`? Comparing the geometry of their constraint sets (:numref:`fig_ridge_geometry`) reveals a striking difference. Minimizing the data loss $L(\mathbf{w})$ subject to $\|\mathbf{w}\|_2^2 \leq r^2$ searches inside a smooth round disk (or sphere in $\mathbb{R}^d$), where the elliptical loss contours almost always touch the boundary at a point where *every* coordinate $w_j$ is small but nonzero. Minimizing $L(\mathbf{w})$ subject to $\|\mathbf{w}\|_1 \leq r$, by contrast, searches inside a sharp-cornered diamond (cross-polytope) whose vertices lie directly on the coordinate axes. Elliptical loss contours frequently collide with one of those sharp corners first, setting many coordinates $w_j$ to *exact zero* and performing automatic feature selection (*sparsity*).

![Constraint geometry of $\ell_2$ ridge regression (left) vs. $\ell_1$ Lasso regression (right). Elliptical contours of the data loss $L(\mathbf{w})$ expanding from the unregularized minimum $\hat{\mathbf{w}}$ typically meet the smooth $\ell_2$ circle away from the axes (keeping both weights nonzero), whereas they often hit a sharp corner of the $\ell_1$ diamond right on an axis (setting $w_2 = 0$).](../img/mdl-linreg-ridge-geometry.svg)
:label:`fig_ridge_geometry`

### A Gaussian Prior on the Weights

Just as maximum likelihood under Gaussian observation noise gave us the squared data loss in :numref:`subsec_normal_distribution_and_squared_loss`, Bayesian statistics gives weight decay an equally clean probabilistic interpretation. Suppose that before observing any data, we place an isotropic zero-mean Gaussian *prior distribution* on the weights, $w_j \sim \mathcal{N}(0, \tau^2)$ independently, so $p(\mathbf{w}) \propto \exp(-\|\mathbf{w}\|_2^2 / (2\tau^2))$, while the labels follow $y \mid \mathbf{x}, \mathbf{w}, b \sim \mathcal{N}(\mathbf{w}^\top\mathbf{x} + b, \sigma^2)$.

By Bayes' theorem, the *posterior density* $p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, b)\,p(\mathbf{w})$ is maximized by minimizing its negative logarithm (*maximum a posteriori* or MAP estimation):

$$\frac{1}{2\sigma^2} \sum_{i=1}^n \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2 + \frac{1}{2\tau^2} \|\mathbf{w}\|_2^2.$$

Dividing through by $n / \sigma^2$ recovers our weight-decay objective :eqref:`eq_wd-objective` with $\lambda = \sigma^2 / (n\tau^2)$. Notice how sensible that ratio is: stronger prior belief that weights are small (smaller $\tau^2$) or noisier observations (larger $\sigma^2$) calls for stronger regularization $\lambda$, whereas as the sample size $n \to \infty$, the penalty $\lambda = \sigma^2 / (n\tau^2) \to 0$ because abundant data eventually speak for themselves.

## Weight Decay from Scratch

To see weight decay rescue a model with nearly as many weights as training examples, let us construct a high-dimensional synthetic regression problem where the dimension $d = 200$ comes close to the training sample size $n_{\textrm{train}} = 300$. Specifically, we draw inputs $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{200})$ and generate labels from

$$y = 0.05 + \sum_{j=1}^{200} 0.1\, x_j + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1),$$

with $300$ training examples (in minibatches of 20) and $1{,}000$ validation examples, using `SyntheticRegressionData` from :numref:`sec_oo-design`. The true weights $\mathbf{w}^* = (0.1, \ldots, 0.1)^\top$ have coordinate scale $\tau = 0.1$, and the noise is substantial: its standard deviation $\sigma = 1$ is comparable to that of the noise-free signal, $\|\mathbf{w}^*\| = \sqrt{2} \approx 1.4$. While 300 examples are technically enough to identify all 200 weights, we will soon see that many linear combinations of them are pinned down only weakly.

```{.python .input #weight-decay-high-dimensional-linear-regression}
tau, sigma, n = 0.1, 1.0, 300   # size of the weights, noise, examples
data = d2l.SyntheticRegressionData(w=d2l.ones(200) * tau, b=0.05,
                                   noise=sigma, num_train=n, num_val=1000,
                                   batch_size=20)
```

Before training anything, let us set two benchmarks for the validation loss.
For independent standard normal inputs $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ and noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the expected loss at $(\mathbf{w}, b^*)$ equals $\frac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|_2^2 + \frac{1}{2}\sigma^2$ (:numref:`sec_linear_scratch`). A trivial constant model that ignores all 200 inputs and
predicts $b^*$ everywhere therefore incurs an expected loss of
$\frac{1}{2}(\|\mathbf{w}^*\|^2 + \sigma^2) = \frac{1}{2}(2 + 1) = 1.5$,
while the oracle parameters $(\mathbf{w}^*, b^*)$ achieve
$\sigma^2/2 = 0.5$. Any learned model must land comfortably below 1.5 to
justify its 200 weights.


### The Penalized Model

We define `l2_penalty(w)` to return $\frac{1}{2}\|\mathbf{w}\|_2^2$:

```{.python .input #weight-decay-defining-ell-2-norm-penalty}
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

We subclass `LinearRegressionScratch` from :numref:`sec_oo-design` (our
object-oriented version of the scratch model in
:numref:`sec_linear_scratch`). Only `training_step` needs to change so that
it returns the penalized objective :eqref:`eq_wd-objective` for
differentiation and optimization. Crucially, `validation_step` continues to
report the unpenalized squared loss, ensuring that validation losses across
different values of $\lambda$ remain directly comparable measures of
predictive accuracy.

```{.python .input #weight-decay-defining-the-model-1}
%%tab pytorch, tensorflow, jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def training_step(self, batch):
        loss = super().training_step(batch)
        return loss + self.lambd * l2_penalty(self.w)
```

```{.python .input #weight-decay-defining-the-model-1}
%%tab mxnet
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def training_step(self, batch):
        loss = super().training_step(batch)
        return loss + self.lambd * l2_penalty(self.w.data())
```


### Training with and without the Penalty

Let us write `train_scratch(lambd, data)` to train `WeightDecayScratch` for 50 epochs at learning rate $\eta = 0.05$ (750 SGD steps) while plotting the training and validation curves, and to return the fitted model, the per-epoch validation losses, and the learned weight norm $\|\mathbf{w}\|_2$ (for comparison, the true weight vector $\mathbf{w}^* = (0.1, \ldots, 0.1)^\top \in \mathbb{R}^{200}$ has $\|\mathbf{w}^*\|_2 = \sqrt{200 \times 0.1^2} \approx 1.4$):

```{.python .input #weight-decay-defining-the-model-2}
%%tab pytorch
def train_scratch(lambd, data):
    trainer = d2l.Trainer(max_epochs=50)
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.05)
    trainer.fit(model, data)
    norm = float(torch.linalg.vector_norm(model.w.detach()))
    return model, [float(v) for v in trainer.history['val_loss']], norm
```

```{.python .input #weight-decay-defining-the-model-2}
%%tab tensorflow
def train_scratch(lambd, data):
    trainer = d2l.Trainer(max_epochs=50)
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.05)
    trainer.fit(model, data)
    norm = float(tf.norm(model.w))
    return model, [float(v) for v in trainer.history['val_loss']], norm
```

```{.python .input #weight-decay-defining-the-model-2}
%%tab jax
def train_scratch(lambd, data):
    trainer = d2l.Trainer(max_epochs=50)
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.05)
    trainer.fit(model, data)
    norm = float(jnp.linalg.norm(model.w[...]))
    return model, [float(v) for v in trainer.history['val_loss']], norm
```

```{.python .input #weight-decay-defining-the-model-2}
%%tab mxnet
def train_scratch(lambd, data):
    trainer = d2l.Trainer(max_epochs=50)
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.05)
    trainer.fit(model, data)
    norm = float(mnp.linalg.norm(model.w.data()))
    return model, [float(v) for v in trainer.history['val_loss']], norm
```

We begin with `lambd = 0`, disabling regularization altogether.

```{.python .input #weight-decay-training-without-regularization}
model_0, val_losses, norm = train_scratch(0, data)
```

```{.python .input #weight-decay-training-with-and-without-the-penalty-1}
print(f'validation loss {val_losses[-1]:.3f} (lowest {min(val_losses):.3f}, '
      f'epoch {val_losses.index(min(val_losses)) + 1}), norm of w {norm:.2f}')
```

The training loss plunges to roughly 0.2–0.3, far below the irreducible
noise floor of 0.5 achieved by the true parameters, confirming that the model
is memorizing noise in its 300 training labels. Meanwhile, the validation
loss hits a minimum of 0.8–1.0 after just two or three epochs and then
climbs steadily, typically finishing between 1.1 and 1.9. After 50 epochs,
this 200-weight model has given back much of its early gain and often does
no better than (and sometimes worse than) the trivial constant $b^*$ at 1.5:
a textbook display of overfitting. Why does validation loss bottom out so early, after only 30
to 45 gradient steps? We will uncover the reason once we examine which
directions in weight space gradient descent learns first.

Next, we turn on weight decay. Rather than guessing $\lambda$ blindly, we
consult the Gaussian prior formula: for $\sigma = 1$, $n = 300$, and true
weight scale $\tau = 0.1$, MAP estimation prescribes
$\lambda = \sigma^2/(n\tau^2) = \frac{1}{3}$.

```{.python .input #weight-decay-using-weight-decay}
lambd = sigma**2 / (n * tau**2)   # the value the prior suggests, 1/3
model_lambd, val_losses, norm = train_scratch(lambd, data)
```

```{.python .input #weight-decay-training-with-and-without-the-penalty-2}
print(f'validation loss {val_losses[-1]:.3f}, norm of w {norm:.2f}')
```

Now the training curve tracks the penalized objective, which stays safely
above 0.5: the penalty keeps the model from chasing noise in the training
labels. Validation loss settles at roughly 0.75–1.05 and stays flat, well below the
1.5 baseline of the constant predictor, while the weight norm shrinks from
roughly 2 to about 1.2, actually *shorter* than the true norm
$\|\mathbf{w}^*\| \approx 1.4$. Even though the regularized weights are
shrunk below the true length, they generalize dramatically better than the
unregularized weights.

Let us be honest about what we gained here. Compared with the
unpenalized model trained for all 50 epochs, weight decay slashes the
validation loss by a quarter to a half. Compared with that same unpenalized
model stopped at its best epoch, however, the gain is small: both
reach roughly 0.8–1.0. The practical difference is stability. Because weight
decay changes the objective itself, its solution stays good no matter how
long we train, whereas the unpenalized run must be halted at a sweet spot
that the training loss alone gives no hint of. To understand *why* both
regularization and early stopping behave this way, let us look inside the
weight vector and ask which components the penalty actually removed.

## Shrinkage Along Eigenvectors
:label:`subsec_wd-shrinkage`

Why does adding $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ improve validation error so dramatically instead of just scaling every weight down blindly? We can answer that question in closed form by looking at the exact minimizer of the ridge objective in the eigenbasis of the feature covariance matrix.

### The Ridge Estimator

Suppose the features and labels are centered so $b = 0$, and write the ridge objective :eqref:`eq_wd-objective` in matrix form:

$$L_{\lambda}(\mathbf{w}) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \frac{\lambda}{2}\|\mathbf{w}\|_2^2.$$

Setting the gradient $\nabla_{\mathbf{w}} L_{\lambda}(\mathbf{w}) = \frac{1}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w} = \mathbf{0}$ gives the regularized normal equations:

$$
\begin{aligned}
\left(\frac{1}{n}\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I}\right) \hat{\mathbf{w}}_{\lambda} &= \frac{1}{n}\mathbf{X}^\top\mathbf{y}, \\
\hat{\mathbf{w}}_{\lambda} &= \left(\mathbf{X}^\top\mathbf{X} + n\lambda \mathbf{I}\right)^{-1}\mathbf{X}^\top\mathbf{y}.
\end{aligned}
$$
:eqlabel:`eq_wd-ridge-normal`

Notice the immediate algebraic consequence: even when $\mathbf{X}^\top\mathbf{X}$ is singular (as it is whenever there are fewer examples than features: with $n = 20$ and $d = 200$, $\mathbf{X}^\top\mathbf{X}$ has 180 zero eigenvalues, exercise :exref:`ex_weight-decay-more-weights`), adding $n\lambda\mathbf{I}$ for any $\lambda > 0$ shifts every eigenvalue upward by $n\lambda > 0$, making $\mathbf{X}^\top\mathbf{X} + n\lambda\mathbf{I}$ strictly positive definite and invertible.

### Shrinkage Factors

To see how :eqref:`eq_wd-ridge-normal` acts along each direction in feature space, let $\mathbf{S} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ be the $d \times d$ sample second-moment (covariance) matrix of the features. Since $\mathbf{S}$ is symmetric and positive semidefinite (:numref:`sec_linear-algebra`), it admits an orthonormal eigendecomposition $\mathbf{S} = \sum_{j=1}^d \mu_j \mathbf{v}_j \mathbf{v}_j^\top$ with eigenvalues $\mu_1 \geq \mu_2 \geq \cdots \geq \mu_d \geq 0$ and orthonormal eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_d$. Here $\mu_j = \frac{1}{n}\|\mathbf{X}\mathbf{v}_j\|_2^2$ measures the empirical variance (spread) of the training inputs along direction $\mathbf{v}_j$.

In this eigenbasis, $\mathbf{S}$ acts on $\mathbf{v}_j$ by multiplying it by $\mu_j$, while $(\mathbf{S} + \lambda\mathbf{I})^{-1}$ acts on $\mathbf{v}_j$ by dividing by $\mu_j + \lambda$. Whenever $\mu_j > 0$, comparing the ridge component $\mathbf{v}_j^\top \hat{\mathbf{w}}_{\lambda}$ against the unregularized least-squares component $\mathbf{v}_j^\top \hat{\mathbf{w}}_0$ yields

$$
\begin{aligned}
\mathbf{v}_j^\top \hat{\mathbf{w}}_{\lambda} &= \frac{\mu_j}{\mu_j + \lambda}\, \left(\mathbf{v}_j^\top \hat{\mathbf{w}}_0\right), \\
\mathbf{v}_j^\top \hat{\mathbf{w}}_{\lambda} &= 0 \quad \textrm{whenever } \mu_j = 0.
\end{aligned}
$$

Each eigendirection $\mathbf{v}_j$ is multiplied by its own *shrinkage factor* (:numref:`fig_ridge_shrinkage`):

$$\frac{\mu_j}{\mu_j + \lambda} \in [0, 1).$$

![The ridge shrinkage factor $\mu / (\mu + \lambda)$ along an eigenvector with eigenvalue $\mu$, against the regularization constant $\lambda$ (log scale). A direction well determined by the data is kept nearly untouched while $\lambda \ll \mu$ and is strongly shrunk toward zero once $\lambda \gg \mu$; the factor is $\frac{1}{2}$ at $\lambda = \mu$ (dots). Directions with $\mu = 0$ are set to zero for every $\lambda > 0$ (dashed).](../img/mdl-linreg-ridge-shrinkage.svg)
:label:`fig_ridge_shrinkage`

Here lies the secret of ridge regression: **weight decay is not uniform shrinkage; it is variance-adaptive directional filtering.**

- Along **well-determined directions** where the data vary richly ($\mu_j \gg \lambda$), the slope is tightly pinned down by observations, and the shrinkage factor $\mu_j / (\mu_j + \lambda) \approx 1$ leaves the least-squares estimate nearly untouched.
- Along **poorly determined directions** where the data barely vary ($\mu_j \ll \lambda$), such as the difference $x_1 - x_2$ between our two almost-identical thermometers, unregularized least squares divides noise by a tiny $\mu_j$ and blows up (with coefficient variance $\sigma^2 / (n \mu_j)$), whereas ridge regression multiplies by $\mu_j / (\mu_j + \lambda) \approx \mu_j / \lambda \ll 1$ and damps the noisy component.
- Along **unobserved null-space directions** ($\mu_j = 0$), ridge regression sets the weight component identically to $0$.

### The Directions Our Data Determine

Let us verify this spectral picture on our $d = 200$ dataset `data`. First, we compute the eigenvalues $\mu_j$ of $\mathbf{S} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ for the 300 training inputs (uncentered, which scarcely alters the spectrum) along with the shrinkage factors at $\lambda = \frac{1}{3}$. As in :numref:`sec_linear-algebra`, `eigh` returns the eigenvalues in ascending order with the eigenvectors as the columns of its second output. We also count how many eigenvalues fall below $\lambda$, marking the poorly determined directions where ridge removes more than half of the least-squares coordinate:

```{.python .input #weight-decay-why-shrinkage-helps-the-spectral-view}
%%tab pytorch
X = data.X[:n]                                # the 300 training inputs
mu, V = torch.linalg.eigh(X.T @ X / n)        # increasing eigenvalues
factors = mu / (mu + lambd)                   # shrinkage at lambda = 1/3
k = int((mu < lambd).sum())                   # the poorly determined ones
print(f'mu from {float(mu[0]):.2f} to {float(mu[-1]):.1f}, '
      f'{k} of {len(mu)} below lambda')
print(f'factors from {float(factors[0]):.2f} to {float(factors[-1]):.2f}, '
      f'df = {float(factors.sum()):.0f}')
```

```{.python .input #weight-decay-why-shrinkage-helps-the-spectral-view}
%%tab tensorflow
X = data.X[:n]                                # the 300 training inputs
mu, V = tf.linalg.eigh(tf.transpose(X) @ X / n)  # increasing eigenvalues
factors = mu / (mu + lambd)                   # shrinkage at lambda = 1/3
k = int(tf.reduce_sum(tf.cast(mu < lambd, tf.int32)))  # poorly determined
print(f'mu from {float(mu[0]):.2f} to {float(mu[-1]):.1f}, '
      f'{k} of {len(mu)} below lambda')
print(f'factors from {float(factors[0]):.2f} to {float(factors[-1]):.2f}, '
      f'df = {float(tf.reduce_sum(factors)):.0f}')
```

```{.python .input #weight-decay-why-shrinkage-helps-the-spectral-view}
%%tab jax
X = data.X[:n]                                # the 300 training inputs
mu, V = jnp.linalg.eigh(X.T @ X / n)          # increasing eigenvalues
factors = mu / (mu + lambd)                   # shrinkage at lambda = 1/3
k = int((mu < lambd).sum())                   # the poorly determined ones
print(f'mu from {float(mu[0]):.2f} to {float(mu[-1]):.1f}, '
      f'{k} of {len(mu)} below lambda')
print(f'factors from {float(factors[0]):.2f} to {float(factors[-1]):.2f}, '
      f'df = {float(factors.sum()):.0f}')
```

```{.python .input #weight-decay-why-shrinkage-helps-the-spectral-view}
%%tab mxnet
X = data.X[:n]                                # the 300 training inputs
mu, V = mnp.linalg.eigh(X.T @ X / n)          # increasing eigenvalues
factors = mu / (mu + lambd)                   # shrinkage at lambda = 1/3
k = int((mu < lambd).sum())                   # the poorly determined ones
print(f'mu from {float(mu[0]):.2f} to {float(mu[-1]):.1f}, '
      f'{k} of {len(mu)} below lambda')
print(f'factors from {float(factors[0]):.2f} to {float(factors[-1]):.2f}, '
      f'df = {float(factors.sum()):.0f}')
```

None of the eigenvalues is zero: 300 examples are enough to span all 200
dimensions of weight space. Yet the data constrain some directions far more
tightly than others. The eigenvalues span nearly two orders of magnitude,
from a few hundredths up to about 3.2, with roughly a quarter of them (50 to
60 directions) lying below $\lambda = \frac{1}{3}$. The corresponding
shrinkage factors range from about 0.1 to 0.9. Their sum defines the
*effective number of parameters*
$\textrm{df}(\lambda) = \sum_j \mu_j/(\mu_j + \lambda)$, and here
$\textrm{df}(\frac{1}{3}) \approx 127$: out of 200 raw weights, the
regularized model behaves as though it has roughly 127 effective parameters.

Just how much damage does a tiny eigenvalue inflict on least squares? Along
eigenvector $\mathbf{v}_j$, label noise injects
$\frac{1}{n}(\mathbf{X}\mathbf{v}_j)^\top\boldsymbol{\epsilon}$ into
$\frac{1}{n}\mathbf{v}_j^\top\mathbf{X}^\top\mathbf{y}$, a random
perturbation with standard deviation $\sigma\sqrt{\mu_j/n}$
(because $\|\mathbf{X}\mathbf{v}_j\|^2 = n\mu_j$). Least squares divides
by $\mu_j$, which shifts its coordinate $\mathbf{v}_j^\top\hat{\mathbf{w}}_0$ by roughly
$\sigma/\sqrt{n\mu_j}$: about 0.06 when $\mu_j = 1$, but a whopping 0.3
when $\mu_j = 0.04$. Since the true coordinates are only around
$\|\mathbf{w}^*\|/\sqrt{200} = 0.1$ in magnitude, least squares along the
smallest-eigenvalue directions is swamped by noise. Meanwhile, validation
inputs are isotropic standard normal, so orthogonal invariance of $\|\mathbf{w} - \mathbf{w}^*\|_2^2 = \sum_j c_j^2$ means that an error $c_j$ along
$\mathbf{v}_j$ inflates the expected validation loss by $c_j^2/2$,
regardless of $\mu_j$. Let us verify this directly by projecting the
parameter error $\mathbf{w} - \mathbf{w}^*$ of both trained models onto the
eigenvectors (`V.T @ (w - w_star)`) and comparing the total squared error
accumulated in the $k$ poorly determined directions versus the remaining
ones:

```{.python .input #weight-decay-the-directions-our-data-determine-1}
%%tab pytorch
w_star = d2l.reshape(data.w, (-1, 1))
for name, model in (('lambda = 0', model_0), ('lambda = 1/3', model_lambd)):
    c = V.T @ (model.w.detach() - w_star)     # error along each eigenvector
    print(f'{name:>12}: squared error {float((c[:k] ** 2).sum()):.2f} in the '
          f'{k} poorly determined directions, '
          f'{float((c[k:] ** 2).sum()):.2f} in the rest')
```

```{.python .input #weight-decay-the-directions-our-data-determine-1}
%%tab tensorflow
w_star = d2l.reshape(data.w, (-1, 1))
for name, model in (('lambda = 0', model_0), ('lambda = 1/3', model_lambd)):
    c = tf.transpose(V) @ (model.w - w_star)  # error along each eigenvector
    print(f'{name:>12}: squared error {float(tf.reduce_sum(c[:k] ** 2)):.2f} '
          f'in the {k} poorly determined directions, '
          f'{float(tf.reduce_sum(c[k:] ** 2)):.2f} in the rest')
```

```{.python .input #weight-decay-the-directions-our-data-determine-1}
%%tab jax
w_star = d2l.reshape(data.w, (-1, 1))
for name, model in (('lambda = 0', model_0), ('lambda = 1/3', model_lambd)):
    c = V.T @ (model.w[...] - w_star)         # error along each eigenvector
    print(f'{name:>12}: squared error {float((c[:k] ** 2).sum()):.2f} in the '
          f'{k} poorly determined directions, '
          f'{float((c[k:] ** 2).sum()):.2f} in the rest')
```

```{.python .input #weight-decay-the-directions-our-data-determine-1}
%%tab mxnet
w_star = d2l.reshape(data.w, (-1, 1))
for name, model in (('lambda = 0', model_0), ('lambda = 1/3', model_lambd)):
    c = V.T @ (model.w.data() - w_star)       # error along each eigenvector
    print(f'{name:>12}: squared error {float((c[:k] ** 2).sum()):.2f} in the '
          f'{k} poorly determined directions, '
          f'{float((c[k:] ** 2).sum()):.2f} in the rest')
```

The breakdown tells the whole story. Without regularization, roughly one
quarter of the eigenvectors account for more than half of the total squared
parameter error: typically between 0.8 and 2.2, versus only about 0.5 across
all the remaining directions combined. Turning on weight decay slashes that
poorly determined error by a factor of two to six while trimming the rest
modestly, leaving roughly 0.3 to 0.5 in each group. Even along the
better-determined directions, label noise is nontrivial relative to the
true coordinates, so shrinking them by 0.5–0.9 still discards more noise
than signal. As a quick sanity check, half the total squared parameter
error plus $\sigma^2/2 = 0.5$ roughly matches the validation loss we
observed for both runs. In effect, weight decay performed across some
fifty ill-conditioned directions simultaneously the same selective damping it
performs on the difference direction $x_1 - x_2$ of our two thermometers.

This spectral lens also solves the mystery we left hanging earlier: why did
the unpenalized run hit its lowest validation loss after just two or three
epochs? On the unpenalized loss, each gradient step contracts the gap
between the current coordinate along $\mathbf{v}_j$ and its least-squares
solution by a factor of $1 - \eta\mu_j$. After a single epoch of 15 steps at
$\eta = 0.05$, a large-eigenvalue direction ($\mu_j = 3.2$) has already
closed $1 - 0.84^{15} \approx 93\%$ of its gap, whereas a small-eigenvalue
direction ($\mu_j = 0.04$) has closed a mere $1 - 0.998^{15} \approx 3\%$.
After two epochs, gradient descent has essentially converged along the
well-determined directions while barely moving along the noisy ones—closely
mimicking the ridge estimator. The resemblance can be made quantitative.
Starting from weights near zero, $t$ steps build up the fraction
$1 - (1 - \eta\mu_j)^t$ of the least-squares coordinate along $\mathbf{v}_j$,
and this is close to the ridge factor $\mu_j/(\mu_j + \lambda)$ with
$\lambda \approx 1/(\eta t)$: both are nearly $1$ when $\mu_j \gg 1/(\eta t)$
and nearly $\eta t \mu_j$ when $\mu_j \ll 1/(\eta t)$. Our best stopping point
of 30 to 45 steps at $\eta = 0.05$ gives $1/(\eta t)$ between $0.45$ and
$0.67$, the same order as the $\lambda = \frac{1}{3}$ that the prior
prescribed. Keep training, however, and those poorly
determined directions steadily soak up label noise, driving validation loss
back up. *Early stopping* is therefore a genuine form of regularization in
its own right, and in this experiment it matches weight decay in accuracy.
It is trickier to deploy, however, because the optimal stopping point shifts
with the learning rate and minibatch noise and must be monitored on
validation data throughout every run (exercise
:exref:`ex_weight-decay-early-stopping`).

## Concise Implementation

Because weight decay is ubiquitous in neural network training, every major
deep learning framework lets the optimizer apply the update
:eqref:`eq_wd-update`, so you need not modify the loss function.
Beyond convenience, this integration is computationally clean: the
optimizer already visits every parameter once per step, so applying the
decay factor adds virtually no overhead. Frameworks also support
per-parameter update rules, allowing us to decay weights while leaving
biases untouched.

We simply subclass `d2l.LinearRegression` into `WeightDecay` and pass the weight decay hyperparameter `wd` to the framework's SGD optimizer in `configure_optimizers` (excluding the bias parameter from decay so that only the weights $\mathbf{w}$ are penalized, matching :eqref:`eq_wd-objective`).

:begin_tab:`mxnet`
In Gluon, we pass the weight decay hyperparameter directly as `wd` when
instantiating `Trainer`. Because Gluon scales each parameter's decay by its
own `wd_mult` attribute, setting `wd_mult = 0` on the bias exempts it from
decay.
:end_tab:

:begin_tab:`pytorch`
PyTorch optimizers accept a `weight_decay` argument. By splitting the
parameters into two parameter groups, we apply `weight_decay` only to
`net.weight` while leaving `net.bias` unregularized. Just as in
:numref:`sec_linear_concise`, we initialize the weights from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero by applying `weight_init`
after the first forward pass materializes the lazy layer's parameters.
:end_tab:

:begin_tab:`tensorflow`
Keras optimizers accept a `weight_decay` argument and multiply each
variable by $1 - \eta \cdot$ `weight_decay` prior to every update. Calling
`exclude_from_weight_decay` exempts the bias from shrinkage.
:end_tab:

:begin_tab:`jax`
While `optax.sgd` does not include weight decay directly, Optax composes
optimizers by chaining gradient transformations. Here `add_decayed_weights`
adds `wd` times each parameter to its gradient, and `masked` restricts that
transformation to parameters matching a predicate—namely arrays with more
than one dimension (the weight matrix, excluding the bias vector).
:end_tab:

```{.python .input #weight-decay-concise-implementation-1}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()

    def configure_optimizers(self):
        for p in self.collect_params('.*bias').values():
            p.wd_mult = 0
        return gluon.Trainer(self.collect_params(), 'sgd',
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input #weight-decay-concise-implementation-1}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input #weight-decay-concise-implementation-1}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = tf.keras.optimizers.SGD(self.lr, weight_decay=self.wd)
        optimizer.exclude_from_weight_decay(var_names=['bias'])
        return optimizer
```

```{.python .input #weight-decay-concise-implementation-1}
%%tab jax
def is_matrix(params):
    """Mask that selects the parameters with more than one dimension."""
    return jax.tree.map(lambda p: p.ndim > 1, params)

class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, num_inputs, lr, rngs=None):
        super().__init__(num_inputs, lr, rngs)
        self.save_hyperparameters(ignore=['rngs'])

    def configure_optimizers(self):
        return optax.chain(
            optax.masked(optax.add_decayed_weights(self.wd), is_matrix),
            optax.sgd(self.lr))
```

To match `train_scratch(lambd, data)` exactly, we must rescale both
hyperparameters. Because the built-in squared-error loss omits our factor
of $\frac{1}{2}$, we halve the learning rate to 0.025 just as in
:numref:`sec_linear_concise`. However, the decay term $\lambda\mathbf{w}$
is injected inside the optimizer rather than multiplied by the loss
gradient. With the learning rate cut in half, the optimizer's weight decay
must therefore be doubled to $2\lambda = \frac{2}{3}$ so that every step
retains the exact same contraction factor
$1 - 0.025 \cdot \frac{2}{3} = 1 - \frac{1}{60}$.

```{.python .input #weight-decay-concise-implementation-2}
%%tab pytorch
trainer = d2l.Trainer(max_epochs=50)
model = WeightDecay(wd=2 * lambd, lr=0.025)
model(data.X[:1])  # Creates the parameters of the lazy layer
model.net.apply(model.weight_init)
trainer.fit(model, data)
norm = float(torch.linalg.vector_norm(model.get_w_b()[0]))
```

```{.python .input #weight-decay-concise-implementation-2}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=50)
model = WeightDecay(wd=2 * lambd, lr=0.025)
trainer.fit(model, data)
norm = float(tf.norm(model.get_w_b()[0]))
```

```{.python .input #weight-decay-concise-implementation-2}
%%tab jax
trainer = d2l.Trainer(max_epochs=50)
model = WeightDecay(wd=2 * lambd, num_inputs=200, lr=0.025)
trainer.fit(model, data)
norm = float(jnp.linalg.norm(model.get_w_b()[0]))
```

```{.python .input #weight-decay-concise-implementation-2}
%%tab mxnet
trainer = d2l.Trainer(max_epochs=50)
model = WeightDecay(wd=2 * lambd, lr=0.025)
trainer.fit(model, data)
norm = float(mnp.linalg.norm(model.get_w_b()[0]))
```

```{.python .input #weight-decay-concise-implementation-3}
print(f'validation loss {trainer.history["val_loss"][-1]:.3f}, '
      f'norm of w {norm:.2f}')
```

The resulting weight norm (around 1.2) reproduces our from-scratch run at
$\lambda = \frac{1}{3}$, and the validation loss (about twice the
from-scratch value) agrees too once we account for the factor of 2 in the
framework's mean squared error. Note that the plotted training curve now tracks plain mean squared
error rather than the penalized loss, because the decay lives inside the
optimizer rather than the loss function.

In everyday practice, enabling weight decay is as simple as passing this
single optimizer hyperparameter. Outside synthetic benchmarks we rarely
know the noise level $\sigma$ or prior scale $\tau$ in advance, so
$\lambda$ is typically tuned on validation data across candidate runs. Keep
in mind that our norm-based story applies directly to linear models,
whose weights are the slopes of the function itself. For nonlinear
functions, measuring complexity rigorously is subtler: frameworks such
as reproducing kernel Hilbert spaces (RKHS) extend norm penalties to
nonlinear functions, though classical kernel methods scale poorly to
massive datasets. Even so, applying weight decay to the weights
of neural network layers remains one of the most reliable regularizers
in modern deep learning.

## Summary

Adding a squared $\ell_2$ penalty $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ to the training objective (while leaving the bias $b$ unpenalized) keeps high-dimensional models from amplifying noise along nearly collinear feature combinations. In minibatch SGD, this penalty multiplies $\mathbf{w}$ by $(1 - \eta\lambda)$ before each gradient step—hence the name *weight decay*—and corresponds to maximum a posteriori estimation under an isotropic Gaussian prior $w_j \sim \mathcal{N}(0, \tau^2)$ with $\lambda = \sigma^2/(n\tau^2)$. Whereas the sharp corners of an $\ell_1$ penalty drive weights to exact zero, the smooth $\ell_2$ penalty acts as a directional filter in the eigenbasis of $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$: multiplying each least-squares component by $\mu_j/(\mu_j + \lambda)$ preserves directions well determined by the data ($\mu_j \gg \lambda$), strongly damps noisy directions ($\mu_j \ll \lambda$), and zeroes unobserved null-space directions ($\mu_j = 0$).

## Exercises

1. **The $\ell_1$ update.** Replace the penalty
   $\frac{\lambda}{2}\|\mathbf{w}\|^2$ by $\lambda \sum_j |w_j|$.
    1. Derive the minibatch SGD update for $\mathbf{w}$, using the
       subgradient $\operatorname{sign}(w_j)$ of exercise
       :exref:`ex_linear-regression-laplace-noise` in place of the
       derivative of $|w_j|$.
    1. Compare the change that the penalty alone makes to $w_j$ in one step
       of this update with the change it makes in the update
       :eqref:`eq_wd-update`, once for large and once for small $|w_j|$.
    1. The update of (a) almost never sets $w_j$ exactly to zero. Split it
       into a gradient step on $L$ followed by the penalty step, and modify
       the penalty step so that it sets $w_j$ to zero whenever it would
       change the sign of $w_j$. Relate the result to the feature selection
       that lasso performs. The modified step is *soft-thresholding*, the
       proximal step of the $\ell_1$ norm
       (:numref:`subsec_mdl-proximal-operators`).

1. **Shrinkage versus penalty.** Weight decay can be applied by shrinking
   the weights before the gradient step or by adding
   $\frac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss $L$.
    1. Show that the update
       $\mathbf{w} \leftarrow (1-\eta\lambda)\mathbf{w} - \eta\nabla L$ is
       identical to a gradient step on the penalized loss
       $L + \frac{\lambda}{2}\|\mathbf{w}\|^2$.
    1. Suppose that the gradient is rescaled coordinate-wise by fixed
       constants $a_j > 0$ before the step. Shrinking and then stepping
       gives the first update below, and stepping on the penalized gradient
       gives the second:

       $$w_j \leftarrow (1-\eta\lambda)\, w_j - \eta a_j \,\partial_{w_j} L
       \qquad \textrm{and} \qquad
       w_j \leftarrow w_j - \eta a_j \left(\partial_{w_j} L + \lambda w_j\right).$$

       Determine for which $a_j$ the two coincide, and the factor by which
       the second update shrinks $w_j$ in a step with
       $\partial_{w_j} L = 0$. Many optimizers used in deep learning
       rescale the gradient in this way. Which of the two updates keeps
       $\lambda$ a rate of decay that is the same for every weight?

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problem 9.5.*
   :exlabel:`ex_weight-decay-shrinkage-vs-penalty`

1. [code] **Ridge as augmented least squares.** Center the 300 training
   inputs and labels so that the bias drops out, and compute the ridge
   estimator for $\lambda = \frac{1}{3}$ in two ways.
    1. Solve the linear system :eqref:`eq_wd-ridge-normal` directly, for
       instance with `np.linalg.solve`.
    1. Run ordinary least squares with `lstsq` on $\mathbf{X}$ stacked on
       top of $\sqrt{n\lambda}\,\mathbf{I}$, with the labels $\mathbf{y}$
       followed by $d$ zeros.

    Confirm that the two agree up to rounding errors, and explain why. Then
    compare the norm of the result with the norm that
    `train_scratch(lambd, data)` returned, and name the differences between
    the two procedures that could account for any gap.

    *Adapted from Hastie, Tibshirani, and Friedman,
    [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/),
    Exercise 3.12.*
   :exlabel:`ex_weight-decay-augmented-least-squares`

1. [code] **The $\lambda$ sweep.** Train `WeightDecayScratch` on `data`
   for $\lambda \in \{0.01, 0.1, 0.3, 1, 3, 10, 30\}$.
    1. Plot the final validation loss and the norm of $\mathbf{w}$ against
       $\lambda$ on a logarithmic axis, with the values of
       `train_scratch(0, data)` as horizontal lines, and report the $\lambda$ with the smallest
       validation loss.
    1. Compute $\textrm{df}(\lambda)$ for the same values from the
       eigenvalues of :numref:`subsec_wd-shrinkage`, and plot the
       validation loss against $\textrm{df}(\lambda)$. Compare the shape
       with :numref:`fig_capacity_vs_error`.
    1. Is the value $\lambda = \frac{1}{3}$ that the prior suggests close
       to the best value of the sweep? Generate data with noise
       $\sigma = 0.5$ and with $\sigma = 2$, recompute the prior's
       $\lambda$ for each, and check whether it follows the best value of
       a sweep.

1. [code] **Stopping early versus decay.** The unpenalized run of this
   section reaches its lowest validation loss after two or three epochs,
   about as low as the run with $\lambda = \frac{1}{3}$ ends.
    1. From weights near zero, gradient descent on the unpenalized loss
       multiplies the gap between the coordinate along $\mathbf{v}_j$ and
       its least-squares value by $1 - \eta\mu_j$ per step, so after $t$
       steps it has built up the fraction $1 - (1 - \eta\mu_j)^t$ of that
       coordinate. Plot this factor against $\mu_j$ for the eigenvalues of
       :numref:`subsec_wd-shrinkage`, with $t$ equal to one, two, and fifty
       epochs of 15 steps at $\eta = 0.05$, next to the ridge factor
       $\mu_j/(\mu_j + \lambda)$ at $\lambda = \frac{1}{3}$. Which $t$ comes
       closest to ridge?
    1. Train `WeightDecayScratch` with $\lambda = 0$ for two epochs and
       split its error along the eigenvectors as
       :numref:`subsec_wd-shrinkage` does. Compare the squared error in the
       $k$ poorly determined directions and in the rest with the values
       printed there for 50 epochs at $\lambda = 0$ and at
       $\lambda = \frac{1}{3}$.
    1. Repeat the unpenalized run at $\eta = 0.01$ and at $\eta = 0.1$ and
       read the best epoch from `trainer.history`. Explain how it moves,
       and state what early stopping requires in practice that the penalty
       does not.
   :exlabel:`ex_weight-decay-early-stopping`

1. [code] **More weights than examples.** ● Generate data with 200
   features and only 20 training examples, true weights $0.01$, bias
   $0.05$, and noise $\sigma = 0.01$, in minibatches of 5.
    1. Compute the eigenvalues of $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$
       for the 20 training inputs. How many are zero, and why does
       gradient descent never change the weights along their
       eigenvectors?
    1. Train with $\lambda = 0$ and with $\lambda = 3$ for 30 epochs at
       learning rate 0.01, once from the random initialization of
       `WeightDecayScratch` and once from $\mathbf{w} = \mathbf{0}$ (pass
       `sigma=0`). Before you run, predict the four validation losses from
       the norm of the initial weights in the undetermined directions;
       then compare.
    1. Compare all four losses with that of a model that predicts $b^*$
       everywhere. What did weight decay achieve here, and what could no
       value of $\lambda$ achieve?
   :exlabel:`ex_weight-decay-more-weights`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/236)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17979)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.7]{.kicker}

Weight Decay & Spectral Regularization<br>
**ℓ₂ ridge penalty · multiplicative decay · ℓ₁ lasso sparsity · Gaussian MAP prior · eigenspectrum shrinkage**
:::
:::

::: {.slide title="Why Feature Selection Fails on Collinear Features"}
[Motivation: The Two-Thermometer Pathology]{.kicker}

::: {.cols .vc}
::: {.col}
- **Model A ($\mathbf{w} = (1, 1)^\top$):** Averages two sensors $x_1 \approx x_2$ with $\|\mathbf{w}\|_2^2 = 2$, cutting sensor noise variance in half.
- **Model B ($\mathbf{w} = (501, -499)^\top$):** Has $\|\mathbf{w}\|_2^2 = 500{,}002$; fits tiny noise in $x_1 - x_2$ and amplifies a $2^\circ$ sensor drift by $+1{,}000$.
- **Continuous capacity control:** Penalizing $\|\mathbf{w}\|_2^2$ keeps all $d$ features while continuously damping poorly determined directions.
:::

::: {.col .fig .big}
@fig:ch02-lr-two-thermometers
:::
:::
:::

::: {.slide title="The ℓ₂-Penalized Objective (Ridge Regression)"}
[Penalized Empirical Risk]{.kicker}

::: {.cols .vc}
::: {.col}
$$L_{\lambda}(\mathbf{w}, b) = L(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$

- **Lipschitz sensitivity bound:** Bounding $\|\mathbf{w}\|_2$ bounds $|\mathbf{w}^\top(\mathbf{x} - \mathbf{x}')| \leq \|\mathbf{w}\|_2\|\mathbf{x} - \mathbf{x}'\|_2$.
- **Unpenalized bias & standardized features:** The intercept $b$ only shifts baseline output units and is left unpenalized; features should be standardized to unit variance.
:::

::: {.col .fig}
@fig:ch02-lr-penalized-loss
:::
:::
:::

::: {.slide title="Why ℓ₂ Regularization Is Called 'Weight Decay'"}
[Per-Step Multiplicative Shrinkage]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{w} \leftarrow (1 - \eta\lambda)\,\mathbf{w} - \eta\,\nabla_{\mathbf{w}} L_{\mathcal{B}}(\mathbf{w}, b)$$

- **Multiplicative contraction:** Before applying the data gradient step $-\eta\nabla_{\mathbf{w}} L_{\mathcal{B}}$, each update shrinks $\mathbf{w}$ toward $\mathbf{0}$ by $(1 - \eta\lambda) < 1$.
- **Geometric decay of unreinforced directions:** Where $\nabla_{\mathbf{w}} L_{\mathcal{B}} = \mathbf{0}$, weights decay exponentially as $(1 - \eta\lambda)^t$.
:::

::: {.col .fig}
@fig:ch02-lr-multiplicative-decay
:::
:::
:::

::: {.slide title="Constraint Geometry: ℓ₂ Ridge vs. ℓ₁ Lasso Sparsity"}
[Why Lasso Zeroes Coordinates While Ridge Keeps All Nonzero]{.kicker}

By Lagrangian duality, minimizing $L(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|_p^p$ is equivalent to minimizing $L(\mathbf{w})$ inside a norm ball $\|\mathbf{w}\|_p \leq r$:

::: {.cols .vc}
::: {.col}
- **$\ell_2$ Ridge (smooth disk $\|\mathbf{w}\|_2 \leq r$):** Expanding elliptical contours of $L(\mathbf{w})$ centered at $\hat{\mathbf{w}}$ meet the smooth circle tangentially away from the axes, keeping all coordinates nonzero and preferring $(1, 1)^\top$ over $(2, 0)^\top$.
- **$\ell_1$ Lasso (sharp diamond $\|\mathbf{w}\|_1 \leq r$):** The $\ell_1$ ball $\sum_j |w_j| \leq r$ has sharp corners sitting directly on the coordinate axes; loss contours frequently hit a corner first, setting $w_1 = 0$ (**sparsity** / feature selection).
:::

::: {.col .fig .big}
![](../img/mdl-linreg-ridge-geometry.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Bayesian View: MAP Estimation With a Gaussian Prior"}
[Probabilistic Derivation of λ]{.kicker}

::: {.cols .vc}
::: {.col}
$$-\log p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = \frac{n}{\sigma^2}L(\mathbf{w}, b) + \frac{1}{2\tau^2}\|\mathbf{w}\|_2^2 + \textrm{const} \implies \lambda = \frac{\sigma^2}{n\tau^2}$$

- **Zero-mean Gaussian prior $w_j \sim \mathcal{N}(0, \tau^2)$:** Maximizing the posterior density yields ridge regression with $\lambda = \sigma^2 / (n\tau^2)$ ($1/3$ for $\sigma = 1, n = 300, \tau = 0.1$).
:::

::: {.col .fig}
@fig:ch02-lr-gaussian-prior-map
:::
:::
:::

::: {.slide title="High-Dimensional Testbed: 200 Features, 300 Examples"}
[From-Scratch Experimental Setup]{.kicker}

::: {.cols .vc}
::: {.col}
@weight-decay-high-dimensional-linear-regression

@weight-decay-defining-the-model-1

- **Reference benchmarks:** Oracle $(\mathbf{w}^*, b^*)$ achieves $\frac{1}{2}\sigma^2 = 0.50$, whereas constant predictor $\mathbf{w} = \mathbf{0}$ incurs $\frac{1}{2}(\|\mathbf{w}^*\|_2^2 + \sigma^2) = 1.50$.
:::

::: {.col .fig}
@fig:ch02-lr-highdim-testbed
:::
:::
:::

::: {.slide title="Unregularized (λ = 0) vs. Prior Decay (λ = 1/3)"}
[Overfitting Cured by Weight Decay]{.kicker}

For $\sigma = 1.0, n = 300, \tau = 0.1$, the Gaussian prior formula prescribes $\lambda = \sigma^2 / (n\tau^2) = 1 / (300 \times 0.01) = \frac{1}{3}$:

::: {.cols .vc}
::: {.col}
@!weight-decay-training-with-and-without-the-penalty-1

@!weight-decay-training-with-and-without-the-penalty-2

- **Without penalty ($\lambda = 0$):** Training loss drops to $\sim 0.25$ (far below the $0.50$ noise floor), while validation loss climbs to $1.1$–$1.9$, no better than the trivial constant predictor at $1.50$.
- **With penalty ($\lambda = 1/3$):** Validation loss stabilizes at $0.8$–$1.0$ and $\|\hat{\mathbf{w}}\|_2 \approx 1.2$ stays compact.
:::

::: {.col .fig}
@!weight-decay-using-weight-decay
:::
:::
:::

::: {.slide title="Spectral View: Shrinkage in the Eigenbasis of XᵀX/n"}
[Why Weight Decay Is Directional Filtering, Not Blind Scaling]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{v}_j^\top \hat{\mathbf{w}}_{\lambda} = \frac{\mu_j}{\mu_j + \lambda}\bigl(\mathbf{v}_j^\top \hat{\mathbf{w}}_0\bigr), \qquad \operatorname{df}(\lambda) = \sum_{j=1}^d \frac{\mu_j}{\mu_j + \lambda}$$

- **Variance-adaptive filtering:** Preserves well-determined directions ($\mu_j \gg \lambda$), strongly damps noisy collinear directions ($\mu_j \ll \lambda$), and zeroes null-space directions ($\mu_j = 0$).
:::

::: {.col .fig}
@fig:ch02-lr-eigen-filter
:::
:::
:::

::: {.slide title="Visualizing the Directional Shrinkage Factor μ/(μ + λ)"}
[Variance-Adaptive Filtering]{.kicker}

::: {.cols .vc}
::: {.col}
- **Half-power threshold ($\mu_j = \lambda$):** An eigendirection whose empirical feature variance equals $\lambda$ has its least-squares coefficient halved ($\frac{\mu_j}{\mu_j + \lambda} = \frac{1}{2}$).
- **Null-space annihilation ($\mu_j = 0$):** Any unobserved direction ($d > n$ or exact duplicate features) is set identically to $0$.
- **Two-thermometer resolution:** For $x_1 \approx x_2$, the sum direction $\mathbf{v}_+ = (1, 1)^\top/\sqrt{2}$ has $\mu_+ \approx 2 \gg \lambda$ (preserved), while the difference direction $\mathbf{v}_- = (1, -1)^\top/\sqrt{2}$ has $\mu_- \approx 10^{-6} \ll \lambda$ (zeroed out, forcing $\hat{w}_1 \approx \hat{w}_2 \approx 1$).
:::

::: {.col .fig .big}
![](../img/mdl-linreg-ridge-shrinkage.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Spectrum and Error Breakdown on Our 200 Features"}
[Where the Error Lives]{.kicker}

::: {.cols .vc}
::: {.col}
@!weight-decay-why-shrinkage-helps-the-spectral-view

@!weight-decay-the-directions-our-data-determine-1

- **Selective noise removal:** Roughly $k \approx 55$ eigenvalues fall below $\lambda = 1/3$ ($\operatorname{df} \approx 127$); turning on $\lambda = 1/3$ slashes squared error in those noisy directions by $2\times$–$6\times$.
:::

::: {.col .fig}
@fig:ch02-lr-error-spectrum
:::
:::
:::

::: {.slide title="Why Unregularized SGD Peaked Early: Early Stopping"}
[Spectral Dynamics of Gradient Descent]{.kicker}

::: {.cols .vc}
::: {.col}
- **Eigenvalue-dependent speed:** Along $\mathbf{v}_j$, each step contracts the gap to $\hat{w}_{0,j}$ by $(1 - \eta\mu_j)$—after 1 epoch ($15$ steps at $\eta = 0.05$), $\mu_j = 3.2$ is $93\%$ converged while $\mu_j = 0.04$ is only $3\%$ converged.
- **Early stopping as a spectral filter:** Stopping after $T$ steps applies filter $1 - (1 - \eta\mu_j)^T \approx \mu_j / (\mu_j + 1/(\eta T))$, mimicking ridge without a penalty.
:::

::: {.col .fig .big}
@fig:ch02-lr-early-stopping-filter
:::
:::
:::

::: {.slide title="Concise Implementation: Optimizer-Level Weight Decay"}
[Framework Parameter Groups & Masks]{.kicker}

::: {.cols .vc}
::: {.col}
@weight-decay-concise-implementation-1

- **Exempting the bias:** The optimizer decays the weights while leaving the bias unpenalized.
:::

::: {.col .fig}
@fig:ch02-lr-param-groups-wd
:::
:::
:::

::: {.slide title="Matching Hyperparameters When Framework MSE Omits 1/2"}
[Concise Training Verification]{.kicker}

Because framework MSE omits the $\frac{1}{2}$ factor on the data loss, its data gradient is doubled ($2\nabla L$), whereas optimizer-level weight decay adds `wd` $\cdot\,\mathbf{w}$ directly to the gradient without going through the loss:

::: {.cols .vc}
::: {.col}
@!weight-decay-concise-implementation-3

- **Halve `lr`, double `wd`:** To preserve both the data step $-\eta \nabla L$ and the per-step multiplicative decay factor $(1 - \eta\lambda) = 1 - 0.05 \times \frac{1}{3} = 1 - \frac{1}{60}$, we set `lr = 0.025` ($\eta/2$) and `wd = 2 * lambd` ($2/3$).
- **Identical solution:** The concise run recovers the exact same weight norm $\|\hat{\mathbf{w}}\|_2 \approx 1.2$ and validation accuracy.
:::

::: {.col .fig}
@!weight-decay-concise-implementation-2
:::
:::
:::

::: {.slide title="Concept Check: Priors, Shrinkage & Adaptive Decay"}
[Check Your Understanding]{.kicker}

Work out these three questions:

1. **Cost of shrinking the truth:** In our 200-feature benchmark ($\|\mathbf{w}^*\|_2^2 = 2.0$, expected validation loss $\frac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|_2^2 + 0.5$), what validation loss does $\mathbf{w} = 0.8\,\mathbf{w}^*$ incur?
2. **Effective parameters at $\lambda = \mu_j$:** A $10$-feature design matrix has eigenvalues $\mu = 2.0$ ($5\times$), $0.5$ ($3\times$), and $0$ ($2\times$). What is $\operatorname{df}(\lambda)$ at $\lambda = 0.5$?
3. **Decoupled weight decay (AdamW vs. $\ell_2$ penalty):** If an adaptive optimizer preconditions each coordinate's gradient by $a_j > 0$, compare the decoupled update $w_j \leftarrow (1 - \eta\lambda)w_j - \eta a_j \partial_{w_j} L$ against the coupled update $w_j \leftarrow w_j - \eta a_j (\partial_{w_j} L + \lambda w_j)$. Which one decays all weights at the same rate regardless of $a_j$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Shrinking $\mathbf{w}^*$ by $20\%$ yields $\frac{1}{2}\|(0.8 - 1)\mathbf{w}^*\|_2^2 + 0.5 = \frac{1}{2}(0.04)(2.0) + 0.5 = 0.54$: only $0.04$ above the oracle floor $0.50$, whereas unregularized noise-fitting cost $> 0.70$.
2. $\operatorname{df}(0.5) = 5 \cdot \frac{2.0}{2.0 + 0.5} + 3 \cdot \frac{0.5}{0.5 + 0.5} + 2 \cdot \frac{0}{0 + 0.5} = 5(0.8) + 3(0.5) + 0 = 5.5$ effective parameters out of $10$.
3. The **decoupled update** (`AdamW`) shrinks every weight by the uniform factor $(1 - \eta\lambda)$; the coupled update shrinks coordinate $j$ by $(1 - \eta a_j \lambda)$, under-decaying coordinates with large historical gradients ($a_j \ll 1$).
:::
:::

::: {.slide title="Summary & Key Takeaways"}
[Summary]{.kicker}

| Perspective on Weight Decay | Mathematical Formulation | Key Insight / Mechanism |
|---|---|---|
| **Penalized Optimization** | $\min_{\mathbf{w}, b} L(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$ | Bounds Lipschitz slope $\|\mathbf{w}\|_2$; bias $b$ is never penalized; features must be standardized |
| **SGD Multiplicative Step** | $\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \eta \nabla_{\mathbf{w}} L_{\mathcal{B}}$ | Shrinks $\mathbf{w}$ toward $\mathbf{0}$ by factor $1 - \eta\lambda$ every step before applying data gradient |
| **Constraint Geometry ($\ell_2$ vs. $\ell_1$)** | Smooth sphere $\|\mathbf{w}\|_2 \leq r$ vs. diamond $\|\mathbf{w}\|_1 \leq r$ | $\ell_2$ ridge shares weight across correlated features $(1,1)$; $\ell_1$ lasso hits axis corners (sparsity) |
| **Bayesian MAP Prior** | $w_j \sim \mathcal{N}(0, \tau^2),\, \epsilon \sim \mathcal{N}(0, \sigma^2) \Rightarrow \lambda = \frac{\sigma^2}{n\tau^2}$ | Optimal penalty scales with noise-to-signal ratio $\sigma^2/\tau^2$ and decays as $1/n$ |
| **Spectral Shrinkage** | $\mathbf{v}_j^\top\hat{\mathbf{w}}_\lambda = \frac{\mu_j}{\mu_j + \lambda}(\mathbf{v}_j^\top\hat{\mathbf{w}}_0)$, $\operatorname{df}(\lambda) = \sum_j \frac{\mu_j}{\mu_j+\lambda}$ | Preserves well-determined directions ($\mu_j \gg \lambda$) and quashes ill-conditioned noise ($\mu_j \ll \lambda$) |
:::
