```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Generalization
:label:`sec_generalization_basics`

Imagine two college students, Ellie and Irene, preparing for a statistics final exam by practicing on past exams. Ellie memorizes every past question and its exact numerical answer word for word; scored on those practice exams, she gets 100%. Irene instead works to understand the underlying concepts and formulas; she gets 92% on the practice questions because of occasional arithmetic slips, yet when the real final exam arrives with brand-new problems, Irene scores 90% while Ellie flounders.

Machine learning models face the exact same divide. Our ultimate goal is never simply to score well on the finite *training set* we already have; it is to make accurate predictions on unseen examples drawn from the broader population—a property called *generalization*. In our synthetic linear regression earlier, we fitted 3 parameters to 1,000 examples, so memorization was impossible and training loss tracked validation loss closely. Once a model class becomes rich enough to bend around individual data points, however, driving training error to zero can actually make predictions on new data *worse*. In this section, we formalize the distinction between training error and generalization error, demonstrate *underfitting* and *overfitting* through polynomial curve fitting, unpack the *bias--variance decomposition*, and introduce *validation sets* and *$K$-fold cross-validation* for principled model selection.

```{.python .input #generalization-polynomial-curve-fitting-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #generalization-polynomial-curve-fitting-1}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #generalization-polynomial-curve-fitting-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #generalization-polynomial-curve-fitting-1}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

## Training Error and Generalization Error
:label:`subsec_empirical-risk-and-risk`

In supervised learning, we assume that both our training examples and the future test examples we will encounter at deployment are drawn independently from the same underlying probability distribution $P(\mathbf{x}, y)$—the *independent and identically distributed* (i.i.d.) assumption.

Given a finite training dataset $\mathcal{D} = \{(\mathbf{x}^{(1)}, y^{(1)}), \ldots, (\mathbf{x}^{(n)}, y^{(n)})\}$ of $n$ examples drawn i.i.d. from $P$, the *training error* (or *empirical risk*) of a predictor $f$ under loss $l$ is the sample average over $\mathcal{D}$:

$$R_{\textrm{emp}}[f] = \frac{1}{n} \sum_{i=1}^n l\left(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})\right).$$
:eqlabel:`eq_empirical-risk-min`

By contrast, the *generalization error* (or *true risk* / *population risk*) is the expected loss on a fresh example $(\mathbf{x}, y)$ drawn from $P$:

$$R[f] = \mathbb{E}_{(\mathbf{x}, y) \sim P}\Big[l\big(\mathbf{x}, y, f(\mathbf{x})\big)\Big] = \int l\big(\mathbf{x}, y, f(\mathbf{x})\big)\, dP(\mathbf{x}, y).$$
:eqlabel:`eq_true-risk`

We want to find a function $f$ that minimizes the true risk $R[f]$, yet we never have direct access to the distribution $P$—we only observe the finite sample $\mathcal{D}$, so our training algorithm minimizes $R_{\textrm{emp}}[f]$ instead (*empirical risk minimization*).

Why can $R_{\textrm{emp}}[\hat{f}]$ underestimate $R[\hat{f}]$ so severely? If we picked a *fixed* function $f$ in advance—before looking at $\mathcal{D}$—then by the law of large numbers, $R_{\textrm{emp}}[f]$ would be an unbiased estimate of $R[f]$ with fluctuations of order $1/\sqrt{n}$. However, our trained model $\hat{f}$ is *not* fixed independent of $\mathcal{D}$: it was specifically selected from an entire model class $\mathcal{F}$ to minimize error *on $\mathcal{D}$*! Whenever $\mathcal{F}$ is flexible enough to latch onto accidental noise quirks in $\mathcal{D}$, the training error $R_{\textrm{emp}}[\hat{f}]$ becomes optimistically biased downward relative to the true risk $R[\hat{f}]$ (Exercise :exref:`ex_generalization-optimism`). To measure $R[\hat{f}]$ honestly, we must evaluate $\hat{f}$ on an independent held-out *test set* (or *validation set*) $\mathcal{D}'$ that played no role in fitting its parameters.

## Underfitting and Overfitting

Comparing training error against validation error diagnoses two opposite failure modes:

- **Underfitting (high bias):** The model is too rigid to capture the true underlying trend in the data, so *both* training error and validation error remain high and close together.
- **Overfitting (high variance):** The model is so flexible relative to the dataset size that it fits the random noise in the training examples in addition to the signal; training error becomes tiny while validation error climbs much higher, opening a large *generalization gap* $R[\hat{f}] - R_{\textrm{emp}}[\hat{f}]$.

### Polynomial Curve Fitting
:label:`subsec_polynomial-curve-fitting`

We can watch underfitting, balanced fitting, and catastrophic overfitting unfold without ever leaving linear regression! Suppose a scalar input $x$ is drawn uniformly from $[-1, 1]$ and the true label is generated by a smooth curve plus Gaussian noise,

$$y = f^*(x) + \epsilon, \qquad f^*(x) = \cos\left(\frac{3\pi}{2} x\right), \qquad \epsilon \sim \mathcal{N}(0, \sigma^2),$$

with $\sigma = 0.3$. Note that the irreducible population noise floor for the squared loss $\frac{1}{2}(\hat{y} - y)^2$ is $\frac{1}{2}\sigma^2 = 0.045$: even the true function $f^*(x)$ incurs an expected loss of $0.045$ because of the unpredictable noise $\epsilon$.

To fit a curved function using linear regression, we map the scalar input $x$ to a vector of monomial *polynomial features* $(1, x, x^2, \ldots, x^k)^\top \in \mathbb{R}^{k+1}$. A degree-$k$ polynomial

$$\hat{y} = \sum_{j=0}^k w_j x^j$$

is nonlinear in $x$, yet **completely linear in the weights $\mathbf{w} = (w_0, \ldots, w_k)^\top$**! Thus we can fit any degree-$k$ polynomial on $n$ training points by constructing the Vandermonde design matrix $\mathbf{X} \in \mathbb{R}^{n \times (k+1)}$ with entries $X_{ij} = (x^{(i)})^j$ and calling `lstsq` (just as we did in :numref:`subsec_linear-regression-analytic-solution`). Let us generate $n = 20$ training points and $1{,}000$ validation points, write a small helper `fit_poly(X, y, k)` that constructs the degree-$k$ Vandermonde matrix and solves least squares, and compare three degrees: $k = 1$ (a straight line), $k = 3$ (a cubic), and $k = 19$ (a degree-19 polynomial with $20$ parameters for $20$ points):

```{.python .input #generalization-polynomial-curve-fitting-2}
def target(x):
    """The cubic f* that generates the labels."""
    return 5 + 1.2 * x - 3.4 * x**2 + 5.6 * x**3

def make_data(rng, n, sigma):
    """Draw n inputs uniformly from [-1, 1] and label them by the noisy cubic."""
    x = rng.uniform(-1, 1, n)
    return x, target(x) + rng.normal(0, sigma, n)

sigma = 0.1                               # standard deviation of the noise
rng = np.random.default_rng()             # a new random draw on every run
x_train, y_train = make_data(rng, 20, sigma)  # few, so high degrees overfit
x_test, y_test = make_data(rng, 1000, sigma)
```

Below, `features(x, d)` builds the design matrix $[1, x, \ldots, x^d]$,
`fit_degree` finds the exact least-squares weights, and `mse` computes the plain
mean squared error (omitting the factor of $\frac{1}{2}$ so the noise floor sits
neatly at $\sigma^2 = 0.01$):

```{.python .input #generalization-polynomial-curve-fitting-3}
def features(x, d):
    """Design matrix whose columns are the powers x^0, x^1, ..., x^d."""
    return x.reshape(-1, 1) ** np.arange(d + 1)

def fit_degree(x, y, d):

Look at the three fitted curves and their losses:
- **Degree $k = 1$ (underfitting):** A straight line cannot bend to follow $\cos(\frac{3\pi}{2}x)$; both training loss and validation loss are stuck around $0.2$–$0.3$, five times the $0.045$ noise floor.
- **Degree $k = 3$ (well matched):** A cubic has just enough flexibility to track the cosine wave closely; both training and validation loss land near the $0.045$ noise floor.
- **Degree $k = 19$ (overfitting):** With $k + 1 = 20$ parameters and $n = 20$ distinct training points, the polynomial has enough degrees of freedom to interpolate every single training point (Exercise :exref:`ex_generalization-exact-polynomial`). Between the data points, however, the curve oscillates wildly, sending the validation loss soaring by orders of magnitude!

Let us sweep the polynomial degree $k$ across $0, 1, \ldots, 14$ and plot both training loss and validation loss against $k$ on a logarithmic scale:

```{.python .input #generalization-polynomial-curve-fitting-6}
xs = np.linspace(-1, 1, 200)
_, axes = d2l.plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, d in zip(axes, (1, 3, 19)):
    w = fit_degree(x_train, y_train, d)
    d2l.plot([xs, xs, x_train], [target(xs), features(xs, d) @ w, y_train],
             xlabel='x', ylim=(-7, 11), fmts=('--', '-', 'o'),
             legend=['cubic', 'fit', 'training data'] if d == 1 else None,
             axes=ax)
    ax.set_title(f'degree {d}')
```

Those three panels tell the whole story of the two classic failure modes of
supervised learning:

As $k$ increases, the training loss decreases monotonically—because adding $x^{k+1}$ only enlarges the subspace of functions available to least squares. The validation loss, by contrast, traces a classic **U-shaped curve**: it drops steeply from $k = 0$ to $k = 3$ or $4$ as underfitting disappears, and then climbs relentlessly as extra powers of $x$ fit the random noise in our 20 training points.

What happens to the learned weights $w_j$ themselves when the polynomial overfits? Let us inspect the weight norms $\|\hat{\mathbf{w}}\|_2$ across degrees and print the coefficients of the cubic ($k = 3$) alongside a high-degree fit ($k = 12$):

```{.python .input #generalization-polynomial-curve-fitting-4}
for d in (1, 3, 19):
    w = fit_degree(x_train, y_train, d)
    print(f'degree {d:2d}: train {mse(w, x_train, y_train):.2g}, '
          f'test {mse(w, x_test, y_test):.2g}, '
          f'largest |weight| {np.abs(w).max():.2g}')
```

Although exact numbers vary slightly with each random draw, the contrast is
dramatic. Degree $1$ suffers massive errors on both splits (a test MSE above $1$,
over $100\times$ the $\sigma^2 = 0.01$ noise floor). Degree $3$ pushes both
training and test errors right down to the $\sigma^2 = 0.01$ floor, with its
largest weight right near the true leading coefficient $5.6$. Degree $19$ drives
training error far below the noise floor—essentially to zero—while its test error
explodes by many orders of magnitude! Look at its weights to see the mechanism:
degree $19$ relies on astronomically large positive and negative coefficients that
delicately cancel at the $20$ training inputs and blow up everywhere else.

Sweeping across every degree from $1$ to $19$ maps out the full transition on a
logarithmic scale:

```{.python .input #generalization-polynomial-curve-fitting-5}
degrees = list(range(1, 20))
weights = [fit_degree(x_train, y_train, d) for d in degrees]
train_err = [mse(w, x_train, y_train) for w in weights]
test_err = [mse(w, x_test, y_test) for w in weights]
d2l.plot(degrees, [train_err, test_err], xlabel='polynomial degree',
         ylabel='mean squared error', legend=['train', 'test'],
         yscale='log', ylim=(1e-4, 1e3))
```

As promised, training error decreases monotonically with degree, plunging far

For $k = 3$, the weights stay modest (a few units each). By $k = 12$ or $k = 19$, however, individual coefficients explode into the hundreds or thousands with alternating signs! Because $x^{10}$ and $x^{12}$ look almost identical on $[-1, 1]$, the least-squares fit cancels giant positive multiples of one against giant negative multiples of the other to thread the 20 noisy points—a hallmark instability that we will cure directly with weight decay in :numref:`sec_weight_decay`.

### Bias and Variance

Why does model complexity produce a U-shaped generalization curve? Let us decompose the expected squared error of a model $\hat{f}_{\mathcal{D}}$ trained on a random dataset $\mathcal{D}$ at a test point $x$ where $y = f^*(x) + \epsilon$ (with $\mathbb{E}[\epsilon] = 0$, $\mathrm{Var}(\epsilon) = \sigma^2$, and $\epsilon$ independent of $\mathcal{D}$). Write $\bar{f}(x) = \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)]$ for the average predictor across all possible training sets of size $n$. Adding and subtracting $f^*(x)$ and $\bar{f}(x)$ inside $\mathbb{E}_{\mathcal{D}, \epsilon}[(\hat{f}_{\mathcal{D}}(x) - y)^2]$ and noting that the cross terms vanish yields the famous *bias--variance decomposition*:

$$\mathbb{E}_{\mathcal{D}, \epsilon}\!\left[\big(\hat{f}_{\mathcal{D}}(x) - y\big)^2\right] = \underbrace{\big(\bar{f}(x) - f^*(x)\big)^2}_{\textrm{bias}^2(x)} + \underbrace{\mathbb{E}_{\mathcal{D}}\!\left[\big(\hat{f}_{\mathcal{D}}(x) - \bar{f}(x)\big)^2\right]}_{\textrm{variance}(x)} + \underbrace{\sigma^2}_{\textrm{noise}}.$$
:eqlabel:`eq_generalization-bias-variance`

Every model's expected test error is the sum of three nonnegative pieces:
1. **Squared bias $(\bar{f}(x) - f^*(x))^2$:** How far the *average* fitted model (even averaged over infinitely many training datasets) sits from the true function $f^*(x)$. Rigid models (like a straight line $k = 1$ on a cosine wave) have high bias.
2. **Variance $\mathbb{E}_{\mathcal{D}}[(\hat{f}_{\mathcal{D}}(x) - \bar{f}(x))^2]$:** How sensitively the fitted model fluctuates from one random training sample $\mathcal{D}$ to the next. Overly flexible models ($k = 12$) swing wildly with every new draw of 20 points, suffering huge variance.
3. **Irreducible noise $\sigma^2$:** The variance of the label $y$ around $f^*(x)$ itself, which no predictor can ever beat.

We can visualize bias and variance directly by drawing 50 independent 20-point training sets from our cosine distribution, fitting $k = 1$, $k = 3$, and $k = 10$ to each, and plotting both the individual fits (thin lines) and their average $\bar{f}(x)$ (bold curve) against $f^*(x)$ (dashed curve):

```{.python .input #generalization-bias-variance-decomposition}
noise = rng.normal(0, sigma, (len(x_train), 200))   # 200 draws of the noise
bias2, var = [], []
for d in degrees:
    w_avg = fit_degree(x_train, target(x_train), d)               # average fit
    W = fit_degree(x_train, target(x_train)[:, None] + noise, d)  # fit per draw
    bias2.append(mse(w_avg, x_test, target(x_test)))
    var.append((features(x_test, d) @ W).var(axis=1).mean())
d2l.plot(degrees, [bias2, var, [sigma**2] * len(degrees),
                   np.add(bias2, var) + sigma**2],
         xlabel='polynomial degree', ylabel='mean squared error',
         fmts=('-', 'C1-.', 'g:', 'm--'), yscale='log', ylim=(1e-4, 1e3))
d2l.plt.legend(['bias$^2$', 'variance', r'noise $\sigma^2$',
                'expected test error'], ncol=2, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, 1));  # above the plot
```

Look how cleanly the curves explain the U-shape of test error: squared bias is
huge at degrees $1$ and $2$, then collapses to zero at degree $3$ (the first
family rich enough to contain the cubic). Variance, meanwhile, climbs with every
added degree—starting around $\sigma^2(d+1)/n$ (the noise variance scaled by
parameters per example; see exercise :exref:`ex_generalization-optimism`) and
eventually exploding orders of magnitude past the noise floor. Underfitting is
error dominated by *bias*; overfitting is error dominated by *variance*. The
idealized trade-off shown in :numref:`fig_capacity_vs_error` recurs across
classical statistics and machine learning: training error drops steadily with
complexity, while test error reaches its minimum at an intermediate sweet spot
(never dropping below $\sigma^2$).

![Training and test error against model complexity. To the left of the best complexity the model underfits and both errors are high; to the right it overfits, its training error falls below the noise level, and the generalization gap widens.](../img/mdl-gen-capacity-error.svg)
:label:`fig_capacity_vs_error`


The three panels make :eqref:`eq_generalization-bias-variance` vivid:
- At $k = 1$ (left), all 50 lines cluster tightly together (low variance), but their average $\bar{f}(x)$ completely misses the cosine arch (high bias).
- At $k = 3$ (middle), the 50 cubics form a narrow ribbon (low variance) whose average $\bar{f}(x)$ lies right on top of $f^*(x)$ (low bias).
- At $k = 10$ (right), the average $\bar{f}(x)$ tracks $f^*(x)$ well (low bias), but individual fits flare wildly near the edges of $[-1, 1]$ (huge variance).

### Model Complexity

As :numref:`fig_capacity_vs_error` summarizes schematically, increasing model complexity (capacity) reduces bias while increasing variance. The optimal generalization performance sits at the sweet spot where the sum of squared bias and variance reaches its minimum. In classical parametric models, complexity is governed by the number of parameters $d$ and the magnitude of the weights $\|\mathbf{w}\|$; in deep learning (:numref:`sec_generalization_deep`), we will see how regularization, architecture design, and dataset scale reshape this curve.

### Dataset Size

Besides restricting model complexity, the most direct way to crush variance is to collect *more training data*. For a fixed polynomial degree $k$, fitting to $n = 1{,}000$ points instead of $n = 20$ averages out the random noise and pins down even high-degree coefficients cleanly (with estimation variance decaying as $(k + 1)\sigma^2 / n$). Let us compare our degree-$10$ polynomial fit on $n = 15$, $n = 50$, and $n = 500$ training examples:

```{.python .input #generalization-dataset-size}
for n in (20, 50, 100, 1000):
    x, y = make_data(rng, n, sigma)          # a fresh training set of size n
    errs = [mse(fit_degree(x, y, d), x_test, y_test) for d in (1, 3, 19)]
    print(f'n = {n:4d}: test error of degrees 1, 3, 19:',
          ', '.join(f'{e:.2g}' for e in errs))
```

Look at the contrast. Growing the dataset rescues the overfitting degree-19
polynomial dramatically—its test error plunges by orders of magnitude by
$n = 100$, and by $n = 1{,}000$ it matches the true cubic. Meanwhile, the
underfitting straight line barely budges (its test error remains stuck around
$1.5$ to $2$): collecting more data shrinks *variance*, but it can never cure
*bias*. As a classical rule of thumb, model complexity should scale alongside
dataset size rather than racing ahead of it—one big reason deep learning surged
only once massive digital datasets became commonplace (though deep networks will
add fascinating twists to this rule in :numref:`sec_generalization_deep`).


With $n = 15$ points, a degree-$10$ polynomial overfits badly; by $n = 50$, the wild oscillations vanish; and by $n = 500$, it hugs the true cosine curve almost indistinguishably from $f^*(x)$. A model is never "too complex" in a vacuum—it is only too complex *relative to the amount of signal and data available to constrain it*.

## Model Selection
:label:`subsec_generalization-model-selection`

### Validation Sets

In our synthetic experiment we knew $f^*(x)$ and generated 1,000 validation points at will. In a real problem, how should we choose hyperparameters—such as the polynomial degree $k$, the learning rate $\eta$, or the weight decay strength $\lambda$?

We can never choose hyperparameters by minimizing training error, because training error always favors the most complex, memorization-prone model ($k = 19$). Nor should we tune hyperparameters by repeatedly consulting the final *test set*, which would leak test-set noise into our design choices (:numref:`chap_classification_generalization`). Instead, we partition our available data into three disjoint splits:

1. **Training set:** Used to fit the model parameters $(\mathbf{w}, b)$ for each candidate hyperparameter setting.
2. **Validation set:** Used to compare candidate hyperparameter settings and select the winner (*model selection*).
3. **Test set:** Held in a vault and evaluated *once* at the very end to report an unbiased estimate of the selected model's generalization error.

### Cross-Validation

When a dataset is small (say, $n = 20$ or $n = 100$ clinical records), carving off a fixed 20% validation set leaves too few examples to train on and too few to validate reliably. *$K$-fold cross-validation* (:numref:`fig_kfold_cv`) makes every example do double duty. We partition the dataset randomly into $K$ equal-sized disjoint subsets (*folds*) $\mathcal{D}_1, \ldots, \mathcal{D}_K$ (typically $K = 5$ or $K = 10$). In round $i \in \{1, \ldots, K\}$, we train the model from scratch on the $K - 1$ folds $\mathcal{D} \setminus \mathcal{D}_i$ and evaluate its validation loss on the held-out fold $\mathcal{D}_i$. Averaging the $K$ held-out losses gives the $K$-fold cross-validation score, and once the winning hyperparameter is chosen, we retrain the model on all $n$ examples. (In the extreme case $K = n$, called *leave-one-out cross-validation*, each fold is a single data point; for linear least squares, a remarkable algebraic identity computes all $n$ leave-one-out residuals from a *single* fit without retraining $n$ times—see Exercise :exref:`ex_generalization-leave-one-out-shortcut`.)

![In $K$-fold cross-validation ($K = 5$ shown), the dataset is partitioned into $K$ folds; each round holds out one fold for validation (shaded) and trains on the remaining $K - 1$ folds.](../img/kfold-cv.svg)
:label:`fig_kfold_cv`

Let us implement $K$-fold cross-validation in `kfold_loss(X, y, k, K=5)` and use 5-fold cross-validation on our 20 training points alone to select the polynomial degree $k \in \{0, 1, \ldots, 8\}$—without peeking at the 1,000-point validation set:

```{.python .input #generalization-cross-validation-1}
def kfold_splits(rng, n, K=5, repeats=10):
    """Random partitions of the indices 0, ..., n-1 into K folds."""
    return [np.array_split(rng.permutation(n), K) for _ in range(repeats)]

def cv_error(x, y, d, splits):
    """Validation error of the degree-d fit, averaged over all folds."""
    return np.mean([mse(fit_degree(np.delete(x, f), np.delete(y, f), d),
                        x[f], y[f]) for folds in splits for f in folds])
```

Let us select the polynomial degree using strictly the $20$ training examples—evaluating
every degree on the exact same folds for a fair comparison—and check how closely
that choice matches the true test-set optimum (which cross-validation never sees):

```{.python .input #generalization-cross-validation-2}
splits = kfold_splits(rng, len(x_train))  # the same folds for every degree
cv_err = [cv_error(x_train, y_train, d, splits) for d in degrees]
print('cross-validation picks degree:', degrees[np.argmin(cv_err)])
print('lowest test error at degree:  ', degrees[np.argmin(test_err)])
```

Let us plot the 5-fold cross-validation curve right alongside the test-error
curve across all 19 degrees:

```{.python .input #generalization-cross-validation-3}
d2l.plot(degrees, [test_err, cv_err], xlabel='polynomial degree',
         ylabel='mean squared error', fmts=('m--', 'g-.'), yscale='log',
         ylim=(1e-4, 1e3))
d2l.plt.legend(['test', '5-fold cross-validation'], ncol=2, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, 1));  # above the plot
```

Without ever glancing at the test set, repeated 5-fold cross-validation reliably
locates the bottom of the test-error bowl—usually selecting degree $3$, or
occasionally a nearby degree whose test error is nearly indistinguishable (see
exercise :exref:`ex_generalization-cv-reliability`). Near the minimum, the two
curves track within a small factor, with cross-validation naturally a bit noisier
since it averages over $20$ points instead of $1{,}000$. At high degrees, notice
why cross-validation shoots up even earlier and higher than test error: each
fold trains on only $16$ points rather than $20$, so degree $15$ already
interpolates all $16$ training points, and any validation point near the edges of
$[-1, 1]$ falls outside the fitted range where high-degree polynomials explode!

How should we pick $K$? Because every fold trains on only a fraction
$(K - 1) / K$ of the dataset, its fitted model is slightly worse than a model
trained on all $n$ examples, giving $K$-fold cross-validation a slight pessimistic
bias. Pushing $K = n$ (*leave-one-out* cross-validation) nearly eliminates that

Using only the 20 training examples, 5-fold cross-validation cleanly rejects both the underfitted lines ($k \leq 1$) and the overfitted high-degree polynomials ($k \geq 6$), picking $k = 3$ or $k = 4$ and retraining on all 20 points to yield a smooth fit whose true validation loss sits near the $0.045$ noise floor.

## Summary

- **Empirical vs. population risk:** Training minimizes empirical risk $R_{\textrm{emp}}[f]$ on a finite sample $\mathcal{D}$, whereas deployment performance is governed by true population risk $R[f] = \mathbb{E}[l(f(\mathbf{x}), y)]$. Because training selects $f$ to fit $\mathcal{D}$, $R_{\textrm{emp}}[\hat{f}]$ is optimistically biased downward.
- **Underfitting, overfitting, and bias--variance:** Expected test error decomposes into $\textrm{bias}^2 + \textrm{variance} + \sigma^2$. Rigid models underfit (high bias, low variance); overly flexible models on small datasets overfit (low bias, high variance), producing a U-shaped validation curve.
- **Dataset scale:** Increasing training sample size $n$ shrinks estimation variance as $1/n$, allowing richer model classes to generalize cleanly.
- **Model selection:** Hyperparameters must never be chosen by training error or on the final test set; instead, use a dedicated validation split or, when data are scarce, $K$-fold cross-validation.

## Exercises

1. **When IID fails.** Give three examples in which dependent observations
   make treating a dataset as IID inadvisable. For each, state whether a
   random train/validation split overestimates or underestimates the
   generalization error, and describe a split that avoids the problem.


1. **Exact polynomial regression.** Show that when the $n$ training inputs
   $x^{(1)}, \ldots, x^{(n)}$ are distinct, a polynomial of degree $n-1$
   fits the $n$ labels exactly, and explain where distinctness is used.
   What does least squares return when two inputs coincide but their
   labels differ?
   :exlabel:`ex_generalization-exact-polynomial`

1. [code] **A better basis for degree 19.** The high powers $x^{17}$ and
   $x^{19}$ differ by less than 0.05 anywhere on $[-1, 1]$, so the columns
   of `features(x, 19)` are nearly parallel. The Chebyshev polynomials
   $T_0, \ldots, T_{19}$ span the same functions but stay between $-1$
   and $1$ on the interval.
    1. Predict first: if we fit degree 19 to the 20 training points in the
       Chebyshev basis, which of the training error, the test error, and
       the largest weight will change, and by how much?
    1. Check your prediction over several runs: replace `features(x, 19)`
       by `np.polynomial.chebyshev.chebvander(x, 19)` for both the training
       and the test inputs, and compare the two design matrices'
       condition numbers (`np.linalg.cond`), the training and test errors,
       and the largest weights.
    1. Explain why the two fits are the same polynomial in exact
       arithmetic, and which of your numbers differ in float64 and why. Does
       the better basis rescue degree 19 on 20 points? What does your answer
       say about the size of the weights as a measure of complexity?
   :exlabel:`ex_generalization-chebyshev-basis`

1. **Optimism of the training error.** ● Consider least squares with a
   fixed $n \times p$ design matrix $\mathbf{X}$ of full column rank and
   labels $\mathbf{y} = \mathbf{X}\mathbf{w}^* + \boldsymbol{\epsilon}$,
   whose noise terms are independent with mean zero and variance
   $\sigma^2$. The fitted predictions are
   $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ with
   $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$.
    1. Show that the expected training error is
       $\mathbb{E}[\|\mathbf{y} - \hat{\mathbf{y}}\|^2]/n = \sigma^2 (n - p)/n$.
       Hint: $\mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\boldsymbol{\epsilon}$,
       and the trace of $\mathbf{I} - \mathbf{H}$ is $n - p$.
    1. Draw fresh labels $\mathbf{y}' = \mathbf{X}\mathbf{w}^* + \boldsymbol{\epsilon}'$
       at the same inputs. Show that the expected error of
       $\hat{\mathbf{y}}$ on them is $\sigma^2 (n + p)/n$, so that the
       training error underestimates it by $2\sigma^2 p/n$. Show that
       $\sigma^2 p/n$ is the variance of the predictions, averaged over
       the $n$ inputs.
    1. Evaluate both expressions for the cubic fit of
       :numref:`subsec_polynomial-curve-fitting` ($n = 20$, $p = 4$,
       $\sigma = 0.1$) and compare them with the training and test errors
       of degree 3 over several runs of the experiment. Explain the
       differences: how much does the training error of a single run
       spread around its expectation, and why does the formula of (b),
       which holds at the training inputs, not apply exactly to the test
       inputs?
   :exlabel:`ex_generalization-optimism`

1. **$K$-fold cross-validation.**
    1. Suppose that a model trained on $m$ examples has expected
       generalization error $R(m) = R_\infty + c/m$ with $c > 0$. Each
       fold's model trains on $(K-1)n/K$ examples. Compute the bias of the
       $K$-fold estimate relative to $R(n)$, the error of the model trained
       on all $n$ examples. Express the bias as a fraction of
       $R(n) - R_\infty$ and evaluate this fraction for $K = 2$, $5$, $10$,
       and $n$.
    1. After selecting a hyperparameter by cross-validation, the final
       model is retrained on all $n$ examples. Explain why a hyperparameter
       whose best value depends on the training-set size complicates this
       step, and give an example.
   :exlabel:`ex_generalization-fold-cross-validation`

1. **Leave-one-out shortcut.** ● Leave-one-out cross-validation requires
   $n$ model fits for a general learner. Consider least squares with the
   design matrix $\mathbf{X}$, whose rows are $\mathbf{x}^{(i)\top}$. Let
   $\mathbf{A} = \mathbf{X}^\top\mathbf{X}$ and assume that it remains
   invertible when any one row of $\mathbf{X}$ is removed. Let
   $\hat{y}^{(i)}$ be the prediction at $\mathbf{x}^{(i)}$ of the fit on
   all $n$ examples, let $\hat{y}^{(-i)}$ be the prediction at
   $\mathbf{x}^{(i)}$ of the fit that omits example $i$, and let
   $h_i = \mathbf{x}^{(i)\top}\mathbf{A}^{-1}\mathbf{x}^{(i)}$. Use the
   Sherman--Morrison identity
   $$(\mathbf{A} - \mathbf{u}\mathbf{u}^\top)^{-1} = \mathbf{A}^{-1} + \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{u}^\top\mathbf{A}^{-1}}{1 - \mathbf{u}^\top\mathbf{A}^{-1}\mathbf{u}}$$
   with $\mathbf{u} = \mathbf{x}^{(i)}$ to show that
   $$y^{(i)} - \hat{y}^{(-i)} = \frac{y^{(i)} - \hat{y}^{(i)}}{1 - h_i}.$$
   State what leave-one-out cross-validation then costs for least
   squares compared with a single fit.
   :exlabel:`ex_generalization-leave-one-out-shortcut`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/234)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17978)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.6]{.kicker}

Performance on data the model has not seen<br>
**training and test error · underfitting and overfitting · bias and variance · validation**
:::
:::

::: {.slide title="Twenty Weights Fit Twenty Points and Miss New Ones"}
[The problem]{.kicker}

@!generalization-polynomial-curve-fitting-6

- 20 noisy samples of a cubic. Degree 19, one weight per point: training
  error essentially 0, error on 1,000 new points many orders of magnitude
  above the cubic's.
- Fitting the sample is the means; predicting **new** data is the goal.
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- tell training error from generalization error, and estimate the second
  on held-out data;
- diagnose underfitting and overfitting from the two errors;
- split the expected test error into bias², variance, and noise;
- predict when more data helps and when it does not;
- choose a model with a validation set or $K$-fold cross-validation.

**Running example:** 20 noisy samples of the cubic
$5 + 1.2x - 3.4x^2 + 5.6x^3$, fit by polynomials of degree 1 to 19.
:::

::: {.slide title="Held-Out Data Tells Us Whether a Fit Carries Over"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Linear Regression | model, loss, closed form, minibatch SGD |
| Synthetic Data | data with known $\mathbf{w}^*, b^*$ |
| Implementation from Scratch | SGD and the training loop by hand |
| Object-Oriented Design | `Module`, `DataModule`, `Trainer` |
| Concise Implementation | framework layers and losses |
| **Generalization** | **training vs. test error** |
| Weight Decay | a penalty on large weights |
:::

::: {.col .narrow}
- **Uses:** least squares, and the standard error $\sigma/\sqrt{n}$ of
  an average.
- **Skips:** learning-theory bounds, distribution shift, and why deep
  networks generalize; later chapters take these up.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Two Errors]{.dtitle}

[what we can compute and what we want]{.dsub}
:::
:::

::: {.slide title="Training Error Averages Over the Sample, Risk Over the Population"}
[Two errors]{.kicker}

**IID assumption:** training examples and future examples are drawn
independently from one distribution $P(X, Y)$. For a model $f$, with
$l(\hat{y}, y)$ the loss of predicting $\hat{y}$ for label $y$:

$$R_\textrm{emp}(f) = \frac{1}{n}\sum_{i=1}^n l\big(f(\mathbf{x}^{(i)}), y^{(i)}\big)$$

$$R(f) = \mathbb{E}\big[l(f(\mathbf{x}), y)\big]$$

The first averages over the $n$ training examples, the second over a new
example $(\mathbf{x}, y)$ drawn from $P$.

::: {.d2l-note}
The training error (*empirical risk*) is computable; the generalization
error (*risk*) is what we want. Their difference $R - R_\textrm{emp}$ is
the *generalization gap*.
:::
:::

::: {.slide title="Held-Out Data Estimates the Risk; Training Data Flatters the Model"}
[Two errors]{.kicker}

- **Held-out examples:** $n'$ examples that played no part in training.
  Their average loss is unbiased, with standard error $\sigma_l/\sqrt{n'}$
  ($\sigma_l$: standard deviation of one example's loss). The validation loss of
  the earlier decks is such an estimate; this deck calls them a test set.
- **Training set:** it chose the model. A lookup table of its labels has
  training error 0 and knows nothing about new inputs.

::: {.d2l-note .warn}
The training error is an **optimistic** estimate of the risk.
:::
:::

::: {.slide title="Check: Is the Gap More Than Noise?"}
[Check]{.kicker}

A model has training error $0.020$ and error $0.030$ on 400 held-out
examples. The loss of one example has standard deviation $0.04$. Is the
gap real, or could the held-out estimate be off by that much?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The standard error of the held-out estimate is
$0.04/\sqrt{400} = 0.002$. The gap of $0.010$ is five standard errors:
the model does worse on new data than on its training data.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Underfitting and Overfitting]{.dtitle}

[polynomials of growing degree on 20 points]{.dsub}
:::
:::

::: {.slide title="Polynomial Regression Is Linear Regression on Powers of x"}
[Underfitting and overfitting]{.kicker}

$$\hat{y} = \sum_{j=0}^{d} w_j x^j \qquad \textrm{(features } 1, x, \ldots, x^d\textrm{)}$$

- The degree $d$ sets the flexibility; degree $n - 1$ fits $n$ distinct
  points exactly.
- Data: $y = f^*(x) + \epsilon$, $f^*(x) = 5 + 1.2x - 3.4x^2 + 5.6x^3$,
  $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.1$, $x$
  uniform on $[-1, 1]$.
- 20 training and 1,000 test examples, drawn anew on every run. Least
  squares for each degree; predicting $f^*$ exactly scores about
  $\sigma^2 = 0.01$.
:::

::: {.slide title="Degree 1 Underfits; Degree 19 Overfits With Huge Weights"}
[Underfitting and overfitting]{.kicker}

@!generalization-polynomial-curve-fitting-4

- **Degree 1:** both errors far above $\sigma^2 = 0.01$. Too rigid:
  *underfitting*.
- **Degree 3:** both errors of the order of the noise level; largest
  weight near the true coefficient 5.6.
- **Degree 19:** training error near 0, test error many orders of
  magnitude larger, enormous weights: *overfitting*.
:::

::: {.slide title="Training Error Always Falls; Test Error Is Lowest Near Degree 3"}
[Underfitting and overfitting]{.kicker}

::: {.cols .vc}
::: {.col}
- Training error falls with every degree: a larger class contains the
  smaller one.
- Test error is lowest at or just above degree 3, then climbs by orders
  of magnitude.
- High training error: underfitting. Test error far above a low training
  error: overfitting.
:::

::: {.col .fig .big}
@!generalization-polynomial-curve-fitting-5
:::
:::
:::

::: {.slide title="Check: Select the Degree by Training Error?"}
[Check]{.kicker}

Which degree has the smallest training error in the sweep, and how does
its test error compare with that of degree 3?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Degree 19, the most flexible, as it must be: a larger class
never fits the training data worse. Its test error is many orders of
magnitude above that of degree 3. The training error cannot choose the
flexibility.
:::
:::
:::

::: {.slide title="Expected Test Error Is Bias² Plus Variance Plus Noise"}
[Bias and variance]{.kicker}

Same inputs, fresh training noise: the fit $\hat{f}$ is random. For a
test input $x$ with label $y = f^*(x) + \epsilon$,

$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\big(f^*(x) - \mathbb{E}[\hat{f}(x)]\big)^2}_{\textrm{bias}^2} + \underbrace{\textrm{Var}\big(\hat{f}(x)\big)}_{\textrm{variance}} + \sigma^2.$$

- **Bias:** how far the average fit is from the truth.
- **Variance:** how much the fit moves when the noise is redrawn.
- **Noise** $\sigma^2$: what no model can predict.
:::

::: {.slide title="Bias Vanishes at Degree 3; Variance Grows With Every Degree"}
[Bias and variance]{.kicker}

::: {.cols .vc}
::: {.col}
- Least squares is linear in the labels, so the average fit is the fit
  to the noiseless labels: it gives the bias².
- 200 redraws of the training noise give the variance, roughly
  $\sigma^2 (d+1)/n$ at low degrees.
- The sum is smallest at degree 3.
:::

::: {.col .fig}
@!generalization-bias-variance-decomposition
:::
:::
:::

::: {.slide title="Too Rigid Means Bias; Too Flexible Means Variance"}
[Bias and variance]{.kicker}

::: {.cols .vc}
::: {.col}
For many families of models that grow step by step:

- training error keeps falling, even below the noise level;
- test error is smallest at an intermediate complexity;
- the gap widens as complexity grows.
:::

::: {.col .fig .big}
![](../img/mdl-gen-capacity-error.svg){width=100%}
:::
:::
:::

::: {.slide title="Capacity Is More Than a Parameter Count"}
[Model complexity]{.kicker}

- **Capacity:** the ability to fit many labelings. Degree 19 fits *any*
  20 labels exactly, random ones too, so its perfect fit is no evidence of
  a pattern.
- **Weight size:** that fit uses weights larger by many orders of
  magnitude than the cubic's, whose largest is about 5.6. Small weights cannot swing so wildly: weight decay, next.
- **Deep networks** fit randomly assigned labels on standard image
  datasets, yet generalize when trained on the real ones (Zhang et al.,
  2021).
:::

::: {.slide title="More Data Cures Variance, Not Bias"}
[Dataset size]{.kicker}

Fresh training sets of size $n$, the same 1,000 test examples:

@!generalization-dataset-size

- Degree 19: orders of magnitude better by $n = 100$, as good as the
  cubic at $n = 1{,}000$.
- Degree 1: still 1.5 to 2 at $n = 1{,}000$.

::: {.d2l-note}
When capacity grows with the number of weights, let the complexity grow
**with** the data, not ahead of it. Deep networks complicate this rule.
:::
:::

::: {.slide title="Check: Collect More Data?"}
[Check]{.kicker}

Model A: training error 1.9, test error 2.0. Model B: training error
$10^{-14}$, test error $10^{8}$. For which one does ten times more data
help?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Model B. It overfits, and more data cuts its variance. Model
A underfits: more data leaves its bias, so it needs a more flexible
model instead.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Model Selection]{.dtitle}

[choosing without the test set]{.dsub}
:::
:::

::: {.slide title="Selecting on the Test Set Makes Its Score Optimistic"}
[Model selection]{.kicker}

- Keep the best of 19 test errors and you keep a lucky one: the test set
  has been fit.
- The training error always picks the most flexible model.
- **Three-way split:** training set (fit) · validation set (choose) ·
  test set (report once).

::: {.d2l-note .warn}
Test sets reused across many experiments and papers invite the same bias.
:::
:::

::: {.slide title="K-Fold Cross-Validation Uses Every Example to Train and to Validate"}
[Model selection]{.kicker}

::: {.cols .vc}
::: {.col}
- Train on $K - 1$ folds, validate on the one left out, average the $K$
  errors; retrain the winner on all the data.
- Each fold's model sees only $(K-1)/K$ of the data: a slightly
  **pessimistic** estimate.
- $K = n$ (leave-one-out) removes that bias but costs $n$ fits. $K = 5$
  or $10$ is the usual compromise.
:::

::: {.col .fig}
![](../img/mdl-mlp-kfold.svg){width=100%}
:::
:::
:::

::: {.slide title="Cross-Validation Picks a Low Degree Without the Test Set"}
[Model selection]{.kicker}

::: {.cols .vc}
::: {.col}
@!generalization-cross-validation-2

- 5 random folds of the 20 training examples, averaged over 10 splits.
- Usually degree 3, otherwise a degree a little above it with a test
  error close to degree 3's. The test error itself is lowest at 3 or a
  little above.
- At high degrees it usually rises earlier than the test error: each
  fold fits only 16 points, and degree 15 interpolates them.
:::

::: {.col .fig}
@!generalization-cross-validation-3
:::
:::
:::

::: {.slide title="Check: What Does Cross-Validation Cost?"}
[Check]{.kicker}

We compare 19 degrees on $n = 20$ examples. How many least-squares fits
do 5-fold cross-validation and leave-one-out need, counting the final
refit? How many examples does each fit see?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** 5-fold: $19 \cdot 5 = 95$ fits on 16 examples, plus one refit
on all 20; our ten random splits make it 950. Leave-one-out:
$19 \cdot 20 = 380$ fits on 19 examples, plus the refit: less bias for four
times the cost of one split.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Two errors:** training error is optimistic; held-out error estimates
  the risk, with standard error $\sigma_l/\sqrt{n'}$.
- **Underfitting** is bias (degree 1: both errors far above $\sigma^2$);
  **overfitting** is variance (degree 19: test error many orders of
  magnitude above that of degree 3).
- **Expected test error** = bias² + variance + $\sigma^2$, smallest at
  degree 3.
- **More data** cures variance, not bias.
- **Select** with a validation set or $K$-fold cross-validation, never
  with the test set: 5-fold usually picks degree 3.

**Next:** weight decay, a finer control of capacity than the degree.
:::
