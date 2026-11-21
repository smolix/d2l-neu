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
    """Least-squares weights of the degree-d polynomial fitted to (x, y)."""
    return np.linalg.lstsq(features(x, d), y, rcond=None)[0]

def mse(w, x, y):
    """Mean squared error of the polynomial with weights w on (x, y)."""
    return ((features(x, len(w) - 1) @ w - y) ** 2).mean()
```

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

$$
\begin{aligned}
\mathbb{E}_{\mathcal{D}, \epsilon}\!\left[\big(\hat{f}_{\mathcal{D}}(x) - y\big)^2\right] &= \underbrace{\big(\bar{f}(x) - f^*(x)\big)^2}_{\textrm{bias}^2(x)} \\
&\quad + \underbrace{\mathbb{E}_{\mathcal{D}}\!\left[\big(\hat{f}_{\mathcal{D}}(x) - \bar{f}(x)\big)^2\right]}_{\textrm{variance}(x)} + \underbrace{\sigma^2}_{\textrm{noise}}.
\end{aligned}
$$
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
occasionally a nearby degree whose test error is nearly indistinguishable.
Near the minimum, the two
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
[Dive into Deep Learning · §2.5]{.kicker}

Generalization, Bias–Variance & Model Selection<br>
**empirical vs. population risk · underfitting & overfitting · bias–variance decomposition · cross-validation**
:::
:::

::: {.slide title="Memorizing Samples vs. Learning Population Patterns"}
[The Central Problem of Supervised Learning]{.kicker}

Fitting 20 polynomial weights to 20 noisy points from a cubic curve drives the training loss to zero while failing catastrophically on new inputs:

@!generalization-polynomial-curve-fitting-6

- **Degree 1 (left — underfitting):** A straight line is too rigid to bend with the cubic; it misses both the training points and the true curve.
- **Degree 3 (middle — balanced):** Matches the true cubic family and tracks the dashed ground-truth curve closely across $[-1, 1]$.
- **Degree 19 (right — overfitting):** Has $20$ parameters for $20$ points, threading every noisy sample exactly while oscillating wildly between them.
:::

::: {.slide title="Empirical Risk (Training Error) vs. Population Risk"}
[Statistical Foundations]{.kicker}

Under the **i.i.d. assumption**, both our finite training dataset $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ and future deployment examples are drawn independently from the same joint distribution $P(\mathbf{x}, y)$:

$$R_{\textrm{emp}}[f] = \frac{1}{n} \sum_{i=1}^n l\bigl(f(\mathbf{x}^{(i)}), y^{(i)}\bigr), \qquad R[f] = \mathbb{E}_{(\mathbf{x}, y) \sim P}\Bigl[l\bigl(f(\mathbf{x}), y\bigr)\Bigr]$$

- **Empirical risk ($R_{\textrm{emp}}[f]$):** The average loss on the $n$ observed training examples; this is the quantity our training algorithm minimizes.
- **True / population risk ($R[f]$):** The expected loss on a fresh unseen example drawn from $P$; this is the quantity that governs deployment accuracy.
- **Generalization gap ($R[\hat{f}] - R_{\textrm{emp}}[\hat{f}]$):** Measures how much worse the learned predictor $\hat{f}$ performs on unseen population data than on the sample that selected it.
:::

::: {.slide title="Why Training Error Is Optimistically Biased Downward"}
[The Optimism of Empirical Risk Minimization]{.kicker}

Why can we never trust $R_{\textrm{emp}}[\hat{f}]$ as an estimate of $R[\hat{f}]$?

- **Fixed function vs. data-selected function:** For a predictor $f$ fixed *before* observing $\mathcal{D}$, $R_{\textrm{emp}}[f]$ is an unbiased estimator of $R[f]$ with standard error $\sigma_l / \sqrt{n}$.
- **Selection bias of optimization:** Our trained model $\hat{f} = \operatorname{argmin}_{f \in \mathcal{F}} R_{\textrm{emp}}[f]$ was explicitly chosen to minimize error *on $\mathcal{D}$*, latching onto accidental correlations between features and label noise $\boldsymbol{\epsilon}$.
- **Exact optimism in linear least squares:** For a full-rank $n \times p$ design matrix $\mathbf{X}$ under homoscedastic noise $\sigma^2$, the projection matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ has trace $p$, yielding:
  $$\mathbb{E}\bigl[R_{\textrm{emp}}\bigr] = \sigma^2\Bigl(1 - \frac{p}{n}\Bigr), \qquad \mathbb{E}\bigl[R_{\textrm{in-sample}}\bigr] = \sigma^2\Bigl(1 + \frac{p}{n}\Bigr)$$

::: {.d2l-note .warn}
Each fitted parameter absorbs $\sigma^2 / n$ of training noise: when $p = n = 20$, expected training MSE vanishes ($\sigma^2(1 - 20/20) = 0$) even on pure noise!
:::
:::

::: {.slide title="Polynomial Regression: Linear Fitting on Monomials"}
[Controlled Capacity Benchmark]{.kicker}

We can vary model capacity continuously without leaving linear least squares by mapping a scalar input $x \in [-1, 1]$ to the degree-$d$ Vandermonde feature vector $(1, x, x^2, \ldots, x^d)^\top \in \mathbb{R}^{d+1}$:

$$\hat{y} = \sum_{j=0}^{d} w_j x^j, \qquad y = f^*(x) + \epsilon, \quad f^*(x) = 5 + 1.2x - 3.4x^2 + 5.6x^3$$

@generalization-polynomial-curve-fitting-3

- **Linear in the parameters $\mathbf{w}$:** Although $x^j$ is nonlinear in $x$, the prediction $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ is linear in $\mathbf{w} \in \mathbb{R}^{d+1}$, so `np.linalg.lstsq` finds the exact global minimizer for every degree $d \in \{1, \ldots, 19\}$.
- **Benchmark setup:** We draw $n = 20$ training points and $n' = 1{,}000$ test points on $[-1, 1]$ with Gaussian noise $\sigma = 0.1$ (irreducible MSE floor $\sigma^2 = 0.01$).
:::

::: {.slide title="Degrees 1, 3, and 19: Errors and Weight Explosion"}
[Quantitative Diagnosis]{.kicker}

Evaluating training MSE, held-out test MSE, and the largest coefficient magnitude $\|\hat{\mathbf{w}}\|_\infty$ across degrees $1$, $3$, and $19$ exposes the mechanism of overfitting:

@!generalization-polynomial-curve-fitting-4

- **Degree 1 (underfitting):** Both training and test MSE exceed $1.0$ (over $100\times$ the $\sigma^2 = 0.01$ noise floor) because a line cannot represent $x^2$ or $x^3$.
- **Degree 3 (well-specified):** Both training and test MSE land near the irreducible floor $\sigma^2 = 0.01$, and the largest coefficient stays near the true cubic coefficient $5.6$.
- **Degree 19 (overfitting & weight explosion):** Training MSE plunges near $0$, test MSE explodes by orders of magnitude, and coefficients blow up to $10^4$–$10^6$ with alternating signs to cancel nearly collinear monomials ($x^{17} \approx x^{19}$) at the 20 training points!
:::

::: {.slide title="The U-Shaped Test Error Curve Across Degrees 1 to 19"}
[Training vs. Generalization Trajectory]{.kicker}

Sweeping polynomial degree $d$ from $1$ to $19$ contrasts the monotonic decline of training error with the U-shaped trajectory of test error:

::: {.cols .vc}
::: {.col}
- **Monotonic training error:** Because degree $d+1$ strictly contains degree $d$ (by setting $w_{d+1} = 0$), adding monomials can *never* increase least-squares training error.
- **Test error sweet spot ($d = 3$):** Test MSE drops by two orders of magnitude from $d = 1$ to $d = 3$, then climbs relentlessly as extra parameters fit noise.
- **Diagnostic rule:** High training error signals **underfitting**; low training error paired with a large generalization gap signals **overfitting**.
:::

::: {.col .fig .big}
@!generalization-polynomial-curve-fitting-5
:::
:::
:::

::: {.slide title="The Bias–Variance Decomposition of Expected Test Error"}
[Theoretical Decomposition]{.kicker}

Let $\hat{f}_{\mathcal{D}}(x)$ be the predictor trained on a random dataset $\mathcal{D}$, and let $\bar{f}(x) = \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)]$ be the average predictor across independent draws of $\mathcal{D}$. For a test point $y = f^*(x) + \epsilon$ ($\mathbb{E}[\epsilon] = 0, \operatorname{Var}(\epsilon) = \sigma^2$):

$$\mathbb{E}_{\mathcal{D}, \epsilon}\Bigl[\bigl(y - \hat{f}_{\mathcal{D}}(x)\bigr)^2\Bigr] = \underbrace{\bigl(f^*(x) - \bar{f}(x)\bigr)^2}_{\textrm{bias}^2(x)} + \underbrace{\mathbb{E}_{\mathcal{D}}\Bigl[\bigl(\hat{f}_{\mathcal{D}}(x) - \bar{f}(x)\bigr)^2\Bigr]}_{\textrm{variance}(x)} + \underbrace{\sigma^2}_{\textrm{irreducible noise}}$$

- **Squared bias $\bigl(f^*(x) - \bar{f}(x)\bigr)^2$:** Systematic approximation error of the model family even with infinite training ensembles (dominates in rigid, underfitted models).
- **Variance $\operatorname{Var}_{\mathcal{D}}(\hat{f}_{\mathcal{D}}(x))$:** Sensitivity of the learned curve to random fluctuations in the finite training sample $\mathcal{D}$ (dominates in overparameterized, overfitted models).
- **Irreducible noise $\sigma^2$:** Variance of the label $y$ itself around $f^*(x)$, setting the absolute floor on expected MSE.
:::

::: {.slide title="Empirical Bias² and Variance Across Polynomial Degrees"}
[Measuring the Decomposition]{.kicker}

By redrawing the 20-point training noise $200$ times at fixed inputs, we compute $\textrm{bias}^2$ (from the noiseless average fit) and $\textrm{variance}$ (from the spread across the 200 fits) at every degree:

::: {.cols .vc}
::: {.col}
- **Squared bias collapses at $d = 3$:** Huge at $d = 1, 2$ (where lines and parabolas cannot match a cubic), then drops to zero for all $d \geq 3$ because every family $d \geq 3$ contains $f^*(x)$.
- **Variance climbs monotonically:** Scales as $\approx \sigma^2 (d + 1) / n$ at low degrees before exploding as the Vandermonde matrix becomes nearly singular near $d = 19$.
- **Optimal trade-off:** The sum $\textrm{bias}^2 + \textrm{variance} + \sigma^2$ achieves its minimum at $d = 3$.
:::

::: {.col .fig}
@!generalization-bias-variance-decomposition
:::
:::
:::

::: {.slide title="Model Complexity and the Classical Capacity Curve"}
[Bias–Variance Trade-Off]{.kicker}

Across classical parametric model families, increasing capacity shifts error from bias to variance:

::: {.cols .vc}
::: {.col}
- **Left of optimum (Underfitting):** Model capacity is too low; high bias keeps both training risk and population risk elevated.
- **Right of optimum (Overfitting):** Capacity outstrips the information in $n$ examples; training error drops below the noise floor $\sigma^2$ while variance drives test risk up.
- **Weight norm as continuous capacity:** Even at degree 19, functions with small coefficients $\|\mathbf{w}\|_2$ cannot oscillate wildly—motivating weight decay in §2.6.
:::

::: {.col .fig .big}
![](../img/mdl-gen-capacity-error.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Dataset Scale: More Data Cures Variance, Never Bias"}
[Sample Complexity]{.kicker}

What happens to our underfitted line ($d = 1$), well-specified cubic ($d = 3$), and overfitted degree-19 polynomial ($d = 19$) as we increase training sample size $n$ from $20$ to $1{,}000$?

@!generalization-dataset-size

- **Variance decays as $O(p / n)$:** Increasing $n$ from $20$ to $100$ slashes degree-19 test MSE by many orders of magnitude, and by $n = 1{,}000$ degree $19$ matches the cubic right at the $\sigma^2 = 0.01$ floor!
- **Bias is invariant to $n$:** Even at $n = 1{,}000$, the degree-1 line remains stuck at MSE $\approx 1.5$–$2.0$; collecting more data never helps a model class that cannot represent the truth.
:::

::: {.slide title="Principled Model Selection: The Three-Way Data Split"}
[Preventing Test-Set Leakage]{.kicker}

How do we select hyperparameters—such as polynomial degree $d$, learning rate $\eta$, or regularization strength $\lambda$—without cheating?

- **Why never select on training error:** Training error decreases monotonically with nesting complexity and will always pick the most overfitted model ($d = 19$).
- **Why never tune on the test set:** Choosing the best of $M$ models on the test set fits a single discrete hyperparameter to test-set noise, making the reported minimum test error optimistically biased.
- **The three-way partition discipline:**
  1. **Training split:** Fits model parameters $(\mathbf{w}, b)$ for each candidate hyperparameter.
  2. **Validation split:** Evaluates candidate models and selects the winning hyperparameter.
  3. **Test split:** Kept locked in a vault and evaluated **once** to report final generalization risk.
:::

::: {.slide title="K-Fold Cross-Validation When Data Are Scarce"}
[Sample-Efficient Validation]{.kicker}

When total data are scarce ($n = 20$), carving out a separate validation split wastes precious training examples. **$K$-fold cross-validation** partitions the $n$ examples into $K$ equal folds:

::: {.cols .vc}
::: {.col}
- **Rotate the held-out fold:** In round $k \in \{1, \ldots, K\}$, train from scratch on $K - 1$ folds ($\frac{K-1}{K}n$ examples) and validate on held-out fold $k$; average the $K$ validation scores.
- **Refit the winner:** Once the best hyperparameter $d^*$ is selected, retrain the final model on all $n$ examples.
- **Choosing $K$ ($5$ or $10$ vs. $K = n$):** Smaller $K$ trains on fewer examples per fold (slight pessimistic bias); leave-one-out ($K = n$) is nearly unbiased but costs $n$ fits.
:::

::: {.col .fig}
![](../img/kfold-cv.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Selecting Degree via 5-Fold Cross-Validation"}
[Cross-Validation in Code]{.kicker}

Using strictly the $n = 20$ training examples ($K = 5$, so each fold trains on $16$ points and validates on $4$), 5-fold cross-validation locates the bottom of the test-error curve without ever seeing the test set:

::: {.cols .vc}
::: {.col}
@!generalization-cross-validation-2

- **Accurate model selection:** Cross-validation consistently picks degree $3$ (or a neighboring degree whose test MSE is within a tiny fraction of optimal).
- **Why CV rises earlier at high degrees:** Each fold trains on $16$ points rather than $20$, so degree $15$ already interpolates the fold and explodes on held-out boundary points.
:::

::: {.col .fig}
@!generalization-cross-validation-3
:::
:::
:::

::: {.slide title="Concept Check: Risk Gaps, Sample Size & CV Cost"}
[Check Your Understanding]{.kicker}

Apply the principles of generalization and model selection to these three problems:

1. **Statistical significance of a gap:** A model achieves training loss $0.020$ and validation loss $0.030$ on $n' = 400$ held-out examples, where per-example loss has standard deviation $\sigma_l = 0.04$. Could this $0.010$ gap be a random fluctuation of the validation set?
2. **When to collect more data:** Model A has training error $1.9$ and validation error $2.0$. Model B has training error $10^{-14}$ and validation error $10^4$. For which model will collecting $10\times$ more training examples improve validation performance?
3. **Leave-one-out shortcut:** Comparing $19$ polynomial degrees on $n = 20$ points via naive leave-one-out ($K = 20$) cross-validation requires $19 \times 20 = 380$ fits (plus $1$ refit). Using the Sherman–Morrison formula $y^{(i)} - \hat{y}^{(-i)} = (y^{(i)} - \hat{y}^{(i)}) / (1 - H_{ii})$ where $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$, how many least-squares fits are needed?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. The validation standard error is $\sigma_l / \sqrt{n'} = 0.04 / \sqrt{400} = 0.002$. The observed gap of $0.010$ is $5$ standard errors wide—confirming a genuine generalization gap.
2. **Model B** (which is overfitting due to high variance; $10\times$ more data crushes variance). Model A is underfitting (high bias); collecting more data leaves its error stuck near $2.0$, so it needs a richer model class instead.
3. Only **$19$ fits** (one full $n$-point fit per degree)! From a single fit's residuals $y^{(i)} - \hat{y}^{(i)}$ and hat-matrix diagonal $H_{ii}$, the exact leave-one-out residuals are obtained in $O(n)$ arithmetic without refitting $n$ times.
:::
:::

::: {.slide title="Section 2.5 Summary & Key Takeaways"}
[Summary]{.kicker}

| Concept / Phenomenon | Mathematical Expression | Scaling / Behavior | Practical Action / Remedy |
|---|---|---|---|
| **Empirical vs. True Risk** | $R_{\textrm{emp}}[f] = \frac{1}{n}\sum_i l_i$ vs. $R[f] = \mathbb{E}[l]$ | $\mathbb{E}[R_{\textrm{emp}}] = \sigma^2(1 - p/n) < \mathbb{E}[R]$ | Never evaluate or select models on training error |
| **Underfitting (High Bias)** | $\textrm{bias}^2(x) = (\bar{f}(x) - f^*(x))^2 \gg \sigma^2$ | Both train and val errors high ($d=1$) | Increase model capacity / add features (more data won't help) |
| **Overfitting (High Variance)** | $\operatorname{Var}(\hat{f}_{\mathcal{D}}(x)) \approx \sigma^2 p / n \gg \sigma^2$ | Train error $\to 0$, val error explodes ($d=19$) | Add training data ($n \uparrow$), reduce $p$, or apply weight decay (§2.6) |
| **Bias–Variance Identity** | $\mathbb{E}[(y - \hat{f}(x))^2] = \textrm{bias}^2 + \operatorname{Var} + \sigma^2$ | U-shaped test risk minimized at intermediate $d^*$ | Balance approximation capacity against sample size $n$ |
| **Validation & $K$-Fold CV** | Average loss across $K$ held-out folds $\mathcal{D}_k$ | Trains on $\frac{K-1}{K}n$ points per fold ($K=5$ or $10$) | Tune all hyperparameters on val/CV; touch test set only once |
:::
