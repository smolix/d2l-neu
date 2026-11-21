```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Linear Regression
:label:`sec_linear_regression`

Suppose you want to estimate the sale price of a house from its floor area and age. Unlike classification, which assigns an input to a discrete category, *regression* predicts a continuous numerical quantity. This problem—finding a rule that maps quantitative inputs to a numerical output—has a distinguished two-century pedigree. In 1805, Adrien-Marie Legendre published the *method of least squares* to determine comet orbits from noisy astronomical sightings :cite:`Legendre.1805`; four years later, Carl Friedrich Gauss grounded least squares in probability theory by asking what distribution of measurement errors makes the least-squares fit the most probable one, arriving at the Gaussian distribution :cite:`Gauss.1809` (having already used the method in 1801 to relocate the newly discovered dwarf planet Ceres after it vanished behind the sun's glare).

Today, regression powers everything from clinical risk scoring and demand forecasting to the output heads of object detectors. Better still, *linear regression*—where the prediction is an affine combination of the inputs—introduces nearly every core concept of deep learning in a setting simple enough to analyze completely by hand. Viewed as a computational graph, linear regression is a single artificial neuron: a *fully connected layer* with one output unit. In this section, we formulate the model and the squared loss, solve it both analytically via the normal equations and iteratively via *minibatch stochastic gradient descent* (the workhorse optimizer of deep learning), benchmark vectorized array operations against Python loops, and connect the loss function to maximum-likelihood estimation under different noise distributions.

```{.python .input #linear-regression}
%%tab mxnet
import time
import numpy as np
from mxnet import np as mnp, npx
npx.set_np()
```

```{.python .input #linear-regression}
%%tab pytorch
import time
import numpy as np
import torch
```

```{.python .input #linear-regression}
%%tab tensorflow
import time
import numpy as np
import tensorflow as tf
```

```{.python .input #linear-regression}
%%tab jax
import time
import jax
from jax import numpy as jnp
import numpy as np
```

## The Model and the Loss

### Features and Labels

To estimate a house's sale price from its area (in square meters) and age (in years), we collect a *training dataset* of historical sales. Each house in that record is an *example* (or *data point*, *sample*, or *instance*). The quantity we aim to predict—here, the price—is the *label* (or *target*), while the measurable attributes on which we base the prediction—area and age—are the *features* (or *covariates*).

Suppose our dataset contains $n$ examples. We index them by $i \in \{1, \ldots, n\}$, writing $\mathbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)})^\top \in \mathbb{R}^2$ for the feature vector of the $i^{\textrm{th}}$ house and $y^{(i)} \in \mathbb{R}$ for its observed sale price. More generally, when each example is described by $d$ features, $\mathbf{x}^{(i)} \in \mathbb{R}^d$, and the full dataset is the collection of pairs $\{(\mathbf{x}^{(1)}, y^{(1)}), \ldots, (\mathbf{x}^{(n)}, y^{(n)})\}$.

### The Linear Model
:label:`subsec_linear_model`

Every supervised learning method starts with a *model class*—a parameterized family of functions mapping features to predictions. The linear assumption posits that the target can be approximated by a weighted sum of the features plus an offset:

$$\textrm{price} = w_{\textrm{area}} \cdot \textrm{area} + w_{\textrm{age}} \cdot \textrm{age} + b.$$
:eqlabel:`eq_price-area`

Here $w_{\textrm{area}}$ and $w_{\textrm{age}}$ are the *weights*, and $b$ is the *bias* (also called the *offset* or *intercept*). Each weight sets the marginal influence of its feature—how much the predicted price changes per extra square meter of floor space, holding age fixed—while the bias $b$ anchors the prediction when all features are zero. Even if no real house has zero area, keeping $b$ is essential: without it, the fitted plane is forced to pass through the origin, severely restricting the functions our model can express. Strictly speaking, :eqref:`eq_price-area` is an *affine* function of the features (a linear map plus a constant translation), though machine learning universally refers to it as a linear model.

With $d$ features, the model's prediction $\hat{y}$ (where the hat distinguishes an estimate from the observed label $y$) takes the form

$$\hat{y} = w_1 x_1 + \cdots + w_d x_d + b = \mathbf{w}^\top \mathbf{x} + b,$$
:eqlabel:`eq_linreg-y`

where $\mathbf{w} = (w_1, \ldots, w_d)^\top \in \mathbb{R}^d$ collects the weights into a single column vector. Whenever we evaluate the model across all $n$ examples at once, we stack the feature vectors $\mathbf{x}^{(i)}$ as the rows of the *design matrix* $\mathbf{X} \in \mathbb{R}^{n \times d}$ and the predictions $\hat{y}^{(i)}$ into the column vector $\hat{\mathbf{y}} \in \mathbb{R}^n$:

$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b,$$
:eqlabel:`eq_linreg-y-vec`

where the scalar bias $b$ is added to every component of $\mathbf{X}\mathbf{w}$ via broadcasting (:numref:`subsec_broadcasting`). Searching for the best linear predictor now reduces to finding the parameters $(\mathbf{w}, b) \in \mathbb{R}^d \times \mathbb{R}$ that make $\hat{\mathbf{y}}$ match the observed labels $\mathbf{y} = (y^{(1)}, \ldots, y^{(n)})^\top$ as closely as possible. Even when the true relationship between features and labels is only approximately linear—and contaminated by unmeasured factors and measurement noise—a linear model frequently provides a fast, interpretable, and remarkably competitive baseline.

### The Squared Loss
:label:`subsec_linear-regression-loss-function`

Before we can search for the best parameters, we need a quantitative measure of how badly any candidate $(\mathbf{w}, b)$ misses the observed labels. A *loss function* assigns a nonnegative penalty to the gap between prediction $\hat{y}$ and truth $y$, vanishing when the two agree. For regression, the standard choice is the *squared error*:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
:eqlabel:`eq_mse`

The factor of $\frac{1}{2}$ is purely a cosmetic convenience: it cancels the factor of $2$ that drops down when we differentiate the square, leaving the location of the minimum unchanged. Because the difference $\hat{y}^{(i)} - y^{(i)}$ is squared, overestimates and underestimates of equal magnitude are penalized symmetrically, while large misses are penalized much more heavily than small ones: a prediction off by $3$ incurs nine times the loss of a prediction off by $1$. Geometrically (:numref:`fig_fit_linreg`), each residual $\hat{y}^{(i)} - y^{(i)}$ is the vertical distance between the data point and the regression line, and $2\,l^{(i)}$ is the area of the square built on that vertical segment.

![Fitting a line to data by minimizing the sum of squared vertical residuals.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

To score $(\mathbf{w}, b)$ across the entire training set of $n$ examples, we average the per-example losses to form the empirical *loss* (or *cost* / *objective*):

$$
\begin{aligned}
L(\mathbf{w}, b) &= \frac{1}{n} \sum_{i=1}^n l^{(i)}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2 \\
&= \frac{1}{2n} \|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|_2^2.
\end{aligned}
$$

Dividing by $n$ puts the objective on a per-example scale, so that doubling the dataset size does not double the numerical magnitude of the loss or its gradients. Training the model means solving the optimization problem

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w},\, b}\  L(\mathbf{w}, b).$$

Because errors are squared, a prediction off by $10$ incurs a hundred times the penalty of a prediction off by $1$, so a single corrupted label can dominate the fit. To see how severely, let us place twenty points on the line $y = 2x$, corrupt a single label to $y^{(6)} = 10{,}000$ (as a data-entry typo might), and fit a line through the origin $\hat{y} = wx$ in two ways: first by the closed-form squared-loss minimizer $w = \sum_i x^{(i)} y^{(i)} / \sum_i (x^{(i)})^2$, and second by minimizing the *mean absolute error* $\frac{1}{n}\sum_i |w x^{(i)} - y^{(i)}|$ over a fine grid of candidate slopes:

```{.python .input #linear-regression-loss-function}
x = np.arange(1.0, 21.0)
y = 2 * x
y[5] = 10000                            # corrupt a single label
w_sq = (x * y).sum() / (x * x).sum()    # squared-loss minimizer
w_grid = np.linspace(0, 30, 3001)       # slopes 0, 0.01, ..., 30
mae = np.abs(w_grid[:, None] * x - y).mean(axis=1)
w_mae = w_grid[mae.argmin()]            # MAE minimizer on the grid
print(f'true w: 2.00, squared loss: {w_sq:.2f}, MAE: {w_mae:.2f}')
```

A single corrupted point out of twenty yanks the squared-loss slope from $2.00$
all the way to roughly $23$, while the absolute-error slope does not budge from
$2.00$! Under squared loss, one gigantic error of nearly $10{,}000$ overwhelms
the combined pull of all nineteen clean points; under absolute error, its pull
grows only linearly, so the nineteen clean points easily outvote it. Which
behavior is desirable depends on what kind of noise we expect in our labels—a
question we answer from first principles in
:numref:`subsec_normal_distribution_and_squared_loss`.

## Minimizing the Loss

How do we find $(\mathbf{w}^*, b^*)$ when there are $d$ features and $n$ examples? Linear regression is one of the very few models in machine learning that admits both an exact closed-form formula and a scalable iterative optimizer. Comparing the two reveals why deep learning overwhelmingly favors iterative gradient methods.

### The Normal Equations
:label:`subsec_linear-regression-analytic-solution`

First, we can absorb the scalar bias $b$ into the weight vector by appending a constant column of $1$s to the design matrix: each input becomes $(\mathbf{x}^{(i)}, 1)^\top \in \mathbb{R}^{d+1}$ and the parameter vector becomes $(\mathbf{w}, b)^\top \in \mathbb{R}^{d+1}$ (or equivalently, we can center both $\mathbf{X}$ and $\mathbf{y}$ by their sample means; see Exercise :exref:`ex_linear-regression-centering`). Writing $\mathbf{X}$ and $\mathbf{w}$ in this augmented notation, the loss is simply

$$L(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2.$$

We can find its global minimum by pure algebra—completing the square without even taking a derivative—whenever $\mathbf{X}^\top\mathbf{X}$ is invertible (which holds if and only if the columns of $\mathbf{X}$ are linearly independent, requiring $n \geq d$). Define $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$, so that $\mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^\top\mathbf{y}$. Adding and subtracting $\mathbf{X}\hat{\mathbf{w}}$ inside the norm and expanding the square yields

$$
\begin{aligned}
\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2
&= \|(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}) + \mathbf{X}(\hat{\mathbf{w}} - \mathbf{w})\|_2^2 \\
&= \|\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}\|_2^2 + \|\mathbf{X}(\hat{\mathbf{w}} - \mathbf{w})\|_2^2 + 2(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}})^\top\mathbf{X}(\hat{\mathbf{w}} - \mathbf{w}).
\end{aligned}
$$
:eqlabel:`eq_linreg-pythagoras`

Look at the cross term at the end:

$$(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}})^\top\mathbf{X} = (\mathbf{X}^\top\mathbf{y} - \mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}})^\top = \mathbf{0}^\top.$$

It vanishes identically for *every* $\mathbf{w}$! What remains is a sum of two nonnegative squared norms. The first term, $\|\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}\|_2^2$, does not depend on $\mathbf{w}$ at all; the second term, $\|\mathbf{X}(\hat{\mathbf{w}} - \mathbf{w})\|_2^2$, is zero if and only if $\mathbf{w} = \hat{\mathbf{w}}$ (since $\mathbf{X}$ has full column rank). Thus $\hat{\mathbf{w}}$ is the unique global minimizer. We could reach the exact same equation via calculus by setting the gradient $\nabla_{\mathbf{w}} L(\mathbf{w}) = -\frac{1}{n}\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$ to zero, producing the classic *normal equations*:

$$\mathbf{X}^\top \mathbf{X}\, \mathbf{w}^* = \mathbf{X}^\top \mathbf{y}, \qquad \textrm{or} \qquad \mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}.$$

When the columns of $\mathbf{X}$ are linearly dependent (for instance, when two features are redundant or when $d > n$), $\mathbf{X}^\top\mathbf{X}$ is singular: a least-squares minimizer still exists, but infinitely many weight vectors achieve the exact same minimum loss (Exercise :exref:`ex_linear-regression-singular-design-matrix`).

In code, you should almost never form $(\mathbf{X}^\top\mathbf{X})^{-1}$ explicitly; dedicated least-squares solvers (`lstsq`) use QR or SVD factorizations that are both faster and far more numerically stable. Let us verify the normal equations on the three-point line model from :numref:`sec_calculus`—inputs $0, 1, 2$, labels $1, 1, 0$, and augmented rows $(1, \textrm{input})$ so that $\hat{\mathbf{w}} = (w_1, w_2)^\top = (7/6, -1/2)^\top$ holds the intercept first and the slope second—and check that the residual vector $\mathbf{r} = \mathbf{X}\hat{\mathbf{w}} - \mathbf{y} = (1/6, -1/3, 1/6)^\top$ satisfies $\mathbf{X}^\top\mathbf{r} = \mathbf{0}$ up to `float32` precision:

```{.python .input #linear-regression-the-normal-equations}
%%tab pytorch
X = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # rows (1, input)
y = torch.tensor([[1.0], [1.0], [0.0]])
w_hat = torch.linalg.lstsq(X, y).solution
r = X @ w_hat - y                                       # errors
print('w_hat:', tuple(round(float(v), 4) for v in w_hat[:, 0]))
print('r:', tuple(round(float(v), 4) for v in r[:, 0]))
print(f'max |X^T r|: {float((X.T @ r).abs().max()):.0e}')
```

```{.python .input #linear-regression-the-normal-equations}
%%tab tensorflow
X = tf.constant([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # rows (1, input)
y = tf.constant([[1.0], [1.0], [0.0]])
w_hat = tf.linalg.lstsq(X, y, fast=False)      # do not form X^T X
r = X @ w_hat - y                                      # errors
print('w_hat:', tuple(round(float(v), 4) for v in w_hat[:, 0]))
print('r:', tuple(round(float(v), 4) for v in r[:, 0]))
XTr = tf.transpose(X) @ r
print(f'max |X^T r|: {float(tf.reduce_max(tf.abs(XTr))):.0e}')
```

```{.python .input #linear-regression-the-normal-equations}
%%tab jax
X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # rows (1, input)
y = jnp.array([[1.0], [1.0], [0.0]])
w_hat = jnp.linalg.lstsq(X, y)[0]
r = X @ w_hat - y                                    # errors
print('w_hat:', tuple(round(float(v), 4) for v in w_hat[:, 0]))
print('r:', tuple(round(float(v), 4) for v in r[:, 0]))
print(f'max |X^T r|: {float(jnp.abs(X.T @ r).max()):.0e}')
```

```{.python .input #linear-regression-the-normal-equations}
%%tab mxnet
X = mnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # rows (1, input)
y = mnp.array([[1.0], [1.0], [0.0]])
w_hat = mnp.linalg.lstsq(X, y, rcond=None)[0]
r = X @ w_hat - y                                    # errors
print('w_hat:', tuple(round(float(v), 4) for v in w_hat[:, 0]))
print('r:', tuple(round(float(v), 4) for v in r[:, 0]))
print(f'max |X^T r|: {float(mnp.abs(X.T @ r).max()):.0e}')
```

### Least Squares as Projection

Why are these called the *normal* equations? The identity :eqref:`eq_linreg-pythagoras` with the vanishing cross term is literally the Pythagorean theorem in $\mathbb{R}^n$ (:numref:`fig_linreg_projection`). As $\mathbf{w}$ ranges over $\mathbb{R}^d$, the possible prediction vectors $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ sweep out a $d$-dimensional subspace of $\mathbb{R}^n$—the column space (or span) of $\mathbf{X}$. Usually the observed label vector $\mathbf{y} \in \mathbb{R}^n$ does not lie in that subspace, because no linear combination of the $d$ feature columns reproduces all $n$ noisy observations exactly.

![Least squares as orthogonal projection in $\mathbb{R}^n$. The column space of $\mathbf{X}$ contains every prediction vector $\mathbf{X}\mathbf{w}$; the closest point to $\mathbf{y}$ is the orthogonal projection $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$, whose residual $\mathbf{y} - \hat{\mathbf{y}}$ is perpendicular (normal) to every column of $\mathbf{X}$.](../img/mdl-linreg-projection.svg)
:label:`fig_linreg_projection`

Minimizing $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$ means finding the point $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$ in the column space of $\mathbf{X}$ that lies closest to $\mathbf{y}$ in Euclidean distance. By elementary geometry, the closest point in a flat subspace is the *orthogonal projection* of $\mathbf{y}$ onto that subspace—the unique point whose residual vector $\mathbf{r} = \mathbf{y} - \mathbf{X}\hat{\mathbf{w}}$ is perpendicular (*normal*) to every column of $\mathbf{X}$. Requiring $\mathbf{r}$ to be orthogonal to all columns of $\mathbf{X}$ is written $\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}) = \mathbf{0}$, which is none other than the normal equations!

### Minibatch Stochastic Gradient Descent
:label:`subsec_linear-regression-minibatch-sgd`

If linear regression has a one-line analytic solution, why not stop here? For two reasons that define all of modern deep learning:

1. **Computational scaling:** Forming $\mathbf{X}^\top\mathbf{X} \in \mathbb{R}^{d \times d}$ costs $O(nd^2)$ operations and solving the $d \times d$ system costs $O(d^3)$. When a dataset has millions of examples and a model has millions or billions of parameters, storing or inverting a $d \times d$ matrix is completely out of the question.
2. **Nonlinearity:** As soon as we replace the squared loss with another objective (like cross-entropy in :numref:`chap_classification`) or stack nonlinear hidden layers (:numref:`chap_perceptrons`), no closed-form solution exists at all.

Instead, we train almost every deep learning model iteratively using *gradient descent* and its stochastic variants. Recall from :numref:`sec_calculus` that the gradient $\nabla L(\mathbf{w}, b)$ points in the direction of steepest ascent of the loss. Stepping in the *opposite* direction reduces the loss, except faces one practical bottleneck: evaluating the exact gradient $\nabla L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \nabla l^{(i)}(\mathbf{w}, b)$ requires a full pass over all $n$ training examples just to take a single parameter step! When $n = 10^7$, full-batch gradient descent is glacially slow—and wildly wasteful, because the average gradient over 100 randomly chosen examples already points in almost the exact same direction as the average over 10 million.

*Minibatch stochastic gradient descent* (minibatch SGD) exploits this statistical redundancy. At each step $t$, we draw a small random *minibatch* $\mathcal{B}_t \subset \{1, \ldots, n\}$ of $|\mathcal{B}|$ examples, average the loss gradients over $\mathcal{B}_t$ alone, and update the parameters with a positive *learning rate* (step size) $\eta > 0$:

$$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w}, b)}\, l^{(i)}(\mathbf{w}, b).$$

Because $\mathcal{B}_t$ is sampled uniformly from the training set, the minibatch gradient is an *unbiased estimator* of the true full-batch gradient, with standard error decaying as $1/\sqrt{|\mathcal{B}|}$. For linear regression under the squared loss :eqref:`eq_mse`, differentiating $l^{(i)} = \frac{1}{2}(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)})^2$ explicitly gives the update rule

$$
\begin{aligned}
\mathbf{w} &\leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\
b &\leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$
:eqlabel:`eq_linreg_batch_update`

Look at what each term does: the scalar $\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}$ is the signed prediction error on example $i$. If the model overpredicts ($\hat{y}^{(i)} > y^{(i)}$), the update subtracts a multiple of $\mathbf{x}^{(i)}$ from $\mathbf{w}$ and nudges $b$ downward, pulling the next prediction closer to $y^{(i)}$. Notice also the crucial philosophical distinction: $\mathbf{w}$ and $b$ are *model parameters* learned by optimization, whereas the minibatch size $|\mathcal{B}|$ and learning rate $\eta$ are *hyperparameters* chosen by us before training starts (and tuned on held-out validation data, as we discuss in :numref:`sec_generalization_basics`).

### Gradient Noise and the Learning Rate

How does the trajectory of minibatch SGD compare with full-batch gradient descent? Consider the contours of the empirical loss $L(\mathbf{w})$ around the least-squares solution $\hat{\mathbf{w}}$ (:numref:`fig_linreg_gd_sgd`). Full-batch gradient descent computes the exact gradient of $L$ at every step, marching smoothly across the elliptic contours directly into the minimum $\hat{\mathbf{w}}$ (provided $\eta$ is not so large that it overshoots and diverges).

Minibatch SGD follows a different path. Even right at the global minimum $\hat{\mathbf{w}}$—where the average gradient across all $n$ examples is zero—Individual noisy examples still have nonzero residuals $\hat{y}^{(i)} - y^{(i)} \neq 0$. A random minibatch $\mathcal{B}_t$ will almost never have its per-example gradients cancel out to zero exactly! Consequently, at constant learning rate $\eta$, minibatch SGD quickly descends into a small neighborhood of $\hat{\mathbf{w}}$ and then jitterbugs perpetually inside a *noise ball* around $\hat{\mathbf{w}}$ whose squared radius scales as $\eta / |\mathcal{B}|$.

![Full-batch gradient descent (black) marches smoothly along the gradient of $L$ to the minimum $\hat{\mathbf{w}}$, taking one pass over all $n$ examples per step. Minibatch SGD (color) takes $n/|\mathcal{B}|$ inexpensive steps per pass using noisy gradient estimates, descending rapidly to a small noise ball around $\hat{\mathbf{w}}$; quartering the learning rate (right) shrinks the ball.](../img/mdl-linreg-gd-sgd.svg)
:label:`fig_linreg_gd_sgd`

Halving $\eta$ or doubling $|\mathcal{B}|$ shrinks that terminal noise ball, and decaying $\eta \to 0$ over the course of training lets the iterates settle onto $\hat{\mathbf{w}}$ (:numref:`chap_optimization`). In practice, however, reaching the exact bottom of the training loss to ten decimal places is rarely even desirable: our ultimate goal is low error on *unseen* data, and because $\hat{\mathbf{w}}$ itself deviates from the true population parameters by statistical estimation error of order $1/\sqrt{n}$, a tiny optimization jitter inside that statistical uncertainty costs nothing in generalization.

### Vectorization for Speed

When implementing minibatch updates :eqref:`eq_linreg_batch_update`, you should never loop over the examples $i \in \mathcal{B}_t$ one by one in Python. Instead, we *vectorize* the computation—expressing the entire minibatch forward pass and gradient step as array and matrix operations dispatched to highly optimized C++/CUDA libraries that run in parallel across SIMD vector units or thousands of GPU cores.

To see the difference firsthand, let us benchmark predicting the labels of $n = 1{,}000$ examples with weights $\mathbf{w} = (2, -3.4)^\top$ and bias $b = 4.2$: first one example at a time in a Python `for` loop, and then for all $1{,}000$ examples at once with a single matrix product `X @ w + b`:

```{.python .input #linear-regression-vectorization-for-speed-1}
%%tab pytorch
n = 1000
X = torch.ones(n, 2)                # the values do not affect the timing
w = torch.tensor([[2.0], [-3.4]])
b = 4.2
# Wait until the device has finished (CUDA runs asynchronously)
wait = lambda x: torch.cuda.synchronize() if torch.cuda.is_available() else None
```

```{.python .input #linear-regression-vectorization-for-speed-1}
%%tab tensorflow
n = 1000
X = tf.ones((n, 2))                 # the values do not affect the timing
w = tf.constant([[2.0], [-3.4]])
b = 4.2
# Wait until all devices have finished (a GPU runs asynchronously)
wait = lambda x: tf.test.experimental.sync_devices()
```

```{.python .input #linear-regression-vectorization-for-speed-1}
%%tab jax
n = 1000
X = jnp.ones((n, 2))                # the values do not affect the timing
w = jnp.array([[2.0], [-3.4]])
b = 4.2
wait = jax.block_until_ready        # JAX returns before it has finished
```

```{.python .input #linear-regression-vectorization-for-speed-1}
%%tab mxnet
n = 1000
X = mnp.ones((n, 2))                # the values do not affect the timing
w = mnp.array([[2.0], [-3.4]])
b = 4.2
wait = lambda x: npx.waitall()      # MXNet returns before it has finished
```

Our timing helper `seconds` runs a warm-up call first (to absorb one-time setup
and JIT compilation costs) and calls `wait` before stopping the clock, since GPU
and accelerator frameworks dispatch operations asynchronously and may return to
Python while the hardware is still computing.

```{.python .input #linear-regression-vectorization-for-speed-2}
def seconds(f):
    """Time one call of f, after a warm-up call."""
    wait(f())
    start = time.perf_counter()
    wait(f())
    return time.perf_counter() - start
```

```{.python .input #linear-regression-vectorization-for-speed-3}
loop = seconds(lambda: [X[i:i + 1] @ w + b for i in range(n)])
vectorized = seconds(lambda: X @ w + b)
print(f'one example at a time: {loop * 1e3:8.3f} ms')
print(f'all examples at once:  {vectorized * 1e3:8.3f} ms')
print(f'speedup: {loop / vectorized:.0f}x')
```

Even though both paths perform the exact same arithmetic, computing all $1{,}000$
predictions in a single vectorized call is *hundreds of times* faster! The loop
pays Python interpreter overhead and kernel-launch latency on every single
example, whereas the matrix multiplication pays that overhead once and hands the
entire batch to a single compiled linear-algebra routine. Throughout this book, vectorizing over minibatches—turning $n$ inner products $\mathbf{w}^\top\mathbf{x}^{(i)}$ into one matrix-vector product $\mathbf{X}\mathbf{w}$—is the single most important habit for writing fast deep learning code.

## Squared Loss and Gaussian Noise
:label:`subsec_normal_distribution_and_squared_loss`

So far we motivated the squared error loss $\frac{1}{2}(\hat{y} - y)^2$ by geometric convenience and clean derivatives. Now let us ask Gauss's deeper question: *what probabilistic assumption about the data makes minimizing squared error the statistically optimal thing to do?*

### The Normal Distribution

Recall the normal (or Gaussian) distribution with mean $\mu$ and variance $\sigma^2$ (standard deviation $\sigma$), whose probability density is

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

This density forms a symmetric bell-shaped curve centered at $\mu$, whose width is set by the standard deviation $\sigma$ (about 68% of the probability lies within $\sigma$ of the mean and about 95% within $2\sigma$). Because the *squared* deviation $(x - \mu)^2$ appears inside the negative exponential, extreme
deviations are vanishingly unlikely: a value $5\sigma$ from the mean has density
$e^{-12.5} \approx 4 \times 10^{-6}$ times that at the peak. We write
$\epsilon \sim \mathcal{N}(\mu, \sigma^2)$ to denote a normally distributed random
variable. Because sums of many small independent disturbances converge toward a
normal distribution (:numref:`subsec_mdl-gaussian-clt`), the Gaussian is the
natural default model for well-behaved measurement noise.

Formally, linear regression with additive Gaussian noise assumes that every
observed label is the linear signal plus an independent normal noise term:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Under this model, the conditional mean $\mathbb{E}[y \mid \mathbf{x}]$ is indeed
$\mathbf{w}^\top \mathbf{x} + b$ (matching :numref:`subsec_linear_model`), and
labels scatter symmetrically around the line with the same noise scale $\sigma$
at every input $\mathbf{x}$ (:numref:`fig_linreg_gaussian_noise`).

![The Gaussian noise model. For every input $x$, the label $y$ is drawn from a normal density $p(y \mid x)$ centered on the line $wx + b$, with the same width everywhere.](../img/mdl-linreg-gaussian-noise.svg)
:label:`fig_linreg_gaussian_noise`


### Maximum Likelihood

Under this additive Gaussian noise model, given features $\mathbf{x}$, the conditional distribution of the observed label $y$ is normal with mean $\mathbf{w}^\top\mathbf{x} + b$ and variance $\sigma^2$:

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

According to the principle of *maximum likelihood*, we should choose $(\mathbf{w}, b)$ to maximize the conditional probability density of observing the entire dataset $\{y^{(1)}, \ldots, y^{(n)}\}$ given $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$. Assuming the examples are drawn independently, the dataset likelihood factors into the product $\prod_{i=1}^n P(y^{(i)} \mid \mathbf{x}^{(i)})$. Maximizing a positive product is equivalent to minimizing its negative logarithm—the *negative log-likelihood*:

$$-\log P(\mathbf{y} \mid \mathbf{X}) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \sum_{i=1}^n \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Look closely at the two terms on the right. The first term, $\frac{n}{2}\log(2\pi\sigma^2)$, does not depend on $\mathbf{w}$ or $b$ at all. The second term is simply $\frac{n}{\sigma^2} L(\mathbf{w}, b)$—our mean squared error loss scaled by the positive constant $n/\sigma^2$! Therefore, minimizing the squared error loss produces the *exact maximum-likelihood estimator* of $(\mathbf{w}, b)$ under independent homoscedastic Gaussian noise, regardless of the value of $\sigma^2$.

### Choosing a Loss from a Noise Model
:label:`subsec_linreg-loss-menu`

This maximum-likelihood connection is far more than a historical footnote: it gives us a principled recipe for designing loss functions whenever Gaussian noise is *not* the right model of reality (:numref:`fig_linreg-loss-menu`):

| Output type | Noise model | Loss (negative log-likelihood) |
|:--|:--|:--|
| real-valued, well-behaved noise | Gaussian | squared error (this section) |
| real-valued, outliers or heavy tails | Laplace | absolute error (exercise :exref:`ex_linear-regression-laplace-noise`) |
| positive, multiplicative fluctuations (prices) | Gaussian on $\log y$ | squared error on $\log y$ (exercise :exref:`ex_linear-regression-beyond-gaussian-noise`) |
| counts (sales, arrivals) | Poisson | $\nu(\mathbf{x}) - y \log \nu(\mathbf{x})$ (exercise :exref:`ex_linear-regression-counting-apples`) |

- **Heavy-tailed outliers (Laplace noise $\rightarrow$ $\ell_1$ loss):** If occasional corrupted measurements produce wild outliers, the quadratic growth of $(\hat{y} - y)^2$ lets a single bad data point yank the fitted line toward itself. Modeling the noise with heavier exponential tails—the Laplace distribution $p(\epsilon) = \frac{1}{2\tau}\exp(-|\epsilon|/\tau)$—turns the negative log-likelihood into the *absolute error* ($\ell_1$) loss $|\hat{y} - y|$ (Exercise :exref:`ex_linear-regression-laplace-noise`), whose gradient has constant magnitude $\pm 1$ no matter how far an outlier sits.
- **Combining smooth optima with outlier robustness (Huber loss):** Because $|\hat{y} - y|$ has a sharp corner at zero where its derivative jumps between $-1$ and $+1$, practitioners often use the *Huber loss*, which splices a quadratic bowl for small residuals ($|\hat{y} - y| \leq \delta$) onto linear slopes for large residuals ($|\hat{y} - y| > \delta$).
- **Count data (Poisson noise):** When the target $y \in \{0, 1, 2, \ldots\}$ is a discrete count (such as daily customer arrivals), a Poisson likelihood leads to its own convex negative log-likelihood (Exercise :exref:`ex_linear-regression-counting-apples`).

![Three regression loss functions (right) and the noise densities $p(\epsilon) \propto \exp(-l(\epsilon))$ that the first two correspond to under maximum likelihood (left): squared error (Gaussian) and absolute error (Laplace), with the Huber loss between them. The log-scale inset shows the heavier Laplace tails.](../img/mdl-linreg-loss-menu.svg)
:label:`fig_linreg-loss-menu`

## Linear Regression as a Neural Network

To connect linear regression to the deep architectures of later chapters, we can draw it as a single-layer neural network (:numref:`fig_single_neuron`). The diagram shows the input features $x_1, \ldots, x_d$ as nodes in the *input layer*, each wired with its own synaptic weight $w_1, \ldots, w_d$ into a single computing unit $o_1$ in the *output layer*, which sums the weighted inputs, adds the bias $b$, and emits $\hat{y} = o_1$.

![Linear regression is a single-layer neural network (a single artificial neuron).](../img/singleneuron.svg)
:label:`fig_single_neuron`

By convention, we count only layers that hold tunable parameters, so linear regression is a *one-layer* network. Because every input feature connects to the output node, this layer is called a *fully connected layer* (or *dense layer*). Biologists have long observed that individual neurons in the cortex integrate weighted signals from thousands of incoming dendrites—an analogy that inspired early pioneers like McCulloch and Pitts :cite:`McCulloch.Pitts.1943`—though our focus throughout this book is firmly on the mathematics and engineering of composing such layers into deep networks.

## Summary

- **Model and squared loss:** Linear regression predicts a continuous target via the affine map $\hat{y} = \mathbf{w}^\top\mathbf{x} + b$ (or $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$ across a dataset) and measures fit via the mean squared error $L(\mathbf{w}, b) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|_2^2$.
- **Normal equations vs. minibatch SGD:** Setting $\nabla_{\mathbf{w}} L = \mathbf{0}$ yields the normal equations $\mathbf{X}^\top\mathbf{X}\mathbf{w}^* = \mathbf{X}^\top\mathbf{y}$, which orthogonally project $\mathbf{y}$ onto the column space of $\mathbf{X}$ in $O(nd^2 + d^3)$ time. Minibatch SGD instead updates parameters iteratively using the average gradient over small random batches $\mathcal{B}_t$, scaling cleanly to massive datasets and non-analytic deep networks.
- **Vectorization:** Expressing minibatch operations as native tensor and matrix primitives eliminates Python loop overhead and unlocks orders-of-magnitude speedups.
- **Maximum likelihood:** Minimizing squared error is equivalent to maximum-likelihood estimation under additive Gaussian noise; changing the noise distribution (e.g., to Laplace or Poisson) systematically derives alternative loss functions such as $\ell_1$ absolute error.

## Exercises

1. **Intercept and centering.** Appending a constant feature $1$ to every
   input turns the affine model $\mathbf{w}^\top \mathbf{x} + b$ into a
   linear model in $(\mathbf{x}, 1)$. Centering the data removes the bias
   in a different way. Let $\bar{\mathbf{x}}$ and $\bar{y}$ be the averages
   of the features and of the labels over the training set.
    1. Show that every minimizer $(\hat{\mathbf{w}}, \hat{b})$ of the
       squared loss satisfies
       $\hat{b} = \bar{y} - \hat{\mathbf{w}}^\top \bar{\mathbf{x}}$, and
       conclude that the errors $\hat{y}^{(i)} - y^{(i)}$ of a least-squares
       fit with a bias sum to zero.
    1. Show that $\hat{\mathbf{w}}$ is a least-squares fit without bias to
       the centered data
       $(\mathbf{x}^{(i)} - \bar{\mathbf{x}},\, y^{(i)} - \bar{y})$.
   :exlabel:`ex_linear-regression-centering`

1. **Laplace noise.** Assume that the additive noise $\epsilon$ follows
   the Laplace distribution with scale $1$ (see
   :numref:`sec_mdl-distributions`), that is,
   $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    1. Write out the negative log-likelihood of the data under the model
       $-\log p(\mathbf y \mid \mathbf X)$.
    1. Determine whether a closed-form solution exists.
    1. Write the minibatch stochastic gradient descent update for
       $(\mathbf{w}, b)$ under this loss in the form of
       :eqref:`eq_linreg_batch_update`, using $0$ where $|\cdot|$ has no
       derivative, at zero, and elsewhere its slope
       $\operatorname{sign}(\cdot) = \pm 1$. A substitute of this kind is
       called a *subgradient*. Compare how the size of each example's
       contribution depends on its error under this loss and under the
       squared loss.
    1. [code] Apply this update to the corrupted-label data of
       :numref:`subsec_linear-regression-loss-function`, for a line through
       the origin, using the full dataset in every step, the learning rate
       0.002, and 2,000 steps from $w = 0$. The minimizer of the absolute
       error is exactly $w = 2$. Where do the iterates end? Explain why they
       do not settle at the minimizer, and propose a fix.
   :exlabel:`ex_linear-regression-laplace-noise`

1. **Beyond Gaussian noise.** Suppose you use regression to estimate the
   price of houses or of stocks.
    1. Model the price directly as
       $y = \mathbf{w}^\top \mathbf{x} + b + \epsilon$ with
       $\epsilon \sim \mathcal{N}(0, \sigma^2)$, and take
       $\sigma = 100{,}000$ dollars. Compute the probability that the model
       assigns to a negative price for a house whose predicted price is
       $50{,}000$ dollars. Explain why no single $\sigma$ suits both this
       house and a house whose predicted price is $5{,}000{,}000$ dollars.
    1. Now regress on the logarithm of the price,
       $\log \textrm{price} = \mathbf{w}^\top \mathbf{x} + b + \epsilon$,
       where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ now acts on the log
       scale. Compute the conditional mean and standard deviation of the
       price and show that their ratio does not depend on $\mathbf{x}$.
       Determine whether $\exp(\mathbf{w}^\top \mathbf{x} + b)$ is an
       unbiased prediction of the price. The same modeling choice underlies
       the Black--Scholes model of option pricing :cite:`Black.Scholes.1973`.
   :exlabel:`ex_linear-regression-beyond-gaussian-noise`

1. **Counting apples.** Suppose we want to use regression to estimate the
   *number* of apples sold in a grocery store.
    1. Identify the problems with a Gaussian additive noise model for such
       a target.
    1. The [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
       models counts. It is given by
       $p(k \mid \nu) = \nu^k e^{-\nu}/k!$, where $\nu > 0$
       is the rate and $k \in \{0, 1, 2, \ldots\}$ is the observed count.
       Prove that $\mathbb{E}[k] = \nu$.
    1. Derive the negative log-likelihood of a count $k$ under a rate
       $\nu(\mathbf{x})$ that depends on the features, dropping terms
       that do not depend on the parameters, and check it against the table
       in :numref:`subsec_linreg-loss-menu`.
    1. Compare the parameterizations
       $\nu(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ and
       $\log \nu(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$.
       Determine for each whether $\nu(\mathbf{x}) > 0$ holds for all
       inputs and parameters, and state the consequence for the loss of
       part (c). Write that loss in terms of $(\mathbf{w}, b)$ for the
       second parameterization.
   :exlabel:`ex_linear-regression-counting-apples`

1. [code] **The size of the noise ball.** Generate $n = 1{,}000$ examples with
   two features drawn from $\mathcal{N}(0, 1)$ and labels
   $y = \mathbf{x}^\top \mathbf{w}^* + \epsilon$, where
   $\mathbf{w}^* = (2, -3.4)$ and $\epsilon \sim \mathcal{N}(0, 1)$, and
   compute the least-squares solution $\hat{\mathbf{w}}$ with `lstsq`.
    1. Run minibatch SGD without a bias, starting at $\hat{\mathbf{w}}$,
       with minibatches of 10 and $\eta = 0.1$ for 20,000 steps. Estimate
       the mean squared distance $\mathbb{E}\|\mathbf{w} - \hat{\mathbf{w}}\|^2$ of
       the iterates from $\hat{\mathbf{w}}$.
    1. Repeat for $\eta \in \{0.05, 0.025\}$ and for minibatch sizes 20
       and 40. Check that the mean squared distance is roughly
       proportional to $\eta / |\mathcal{B}|$.
    1. Why does the distance not depend on how far $\hat{\mathbf{w}}$ is
       from $\mathbf{w}^*$? What would change if the labels had no noise
       at all?


1. **Singular design matrix.** ● Full column rank makes the least-squares
   minimizer unique. Suppose instead that $\mathbf{X}^\top \mathbf{X}$ is
   singular.
    1. Show that a minimizer of $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$
       still exists and characterize the set of all minimizers.
    1. Write the singular value decomposition as
       $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ with
       orthogonal $\mathbf{U}$ and $\mathbf{V}$ and a rectangular diagonal
       $\boldsymbol{\Sigma}$ of nonnegative singular values. The
       Moore--Penrose pseudoinverse is
       $$\mathbf{X}^{\dagger} = \mathbf{V}\boldsymbol{\Sigma}^{\dagger}\mathbf{U}^\top,$$
       where $\boldsymbol{\Sigma}^{\dagger}$ transposes
       $\boldsymbol{\Sigma}$ and inverts its nonzero entries. Show that
       $\mathbf{X}^{\dagger}\mathbf{y}$ is the minimizer of smallest norm
       within the set of part (a).
    1. Starting from $\mathbf{w} = \mathbf{0}$, which minimizer does
       gradient descent converge to? State whether the answer changes for
       minibatch stochastic gradient descent.
   :exlabel:`ex_linear-regression-singular-design-matrix`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/259)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17973)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.1]{.kicker}

Linear Regression: Theory, Geometry & Stochastic Optimization<br>
**affine model · squared loss · normal equations · orthogonal projection · minibatch SGD · Gaussian MLE**
:::
:::

::: {.slide title="Predicting Continuous Quantities From Features"}
[Motivation & Problem Formulation]{.kicker}

Unlike classification into discrete categories, regression predicts a continuous numerical target from observed covariates:

::: {.cols .vc}
::: {.col}
- **Concrete task:** Estimate the sale price $y \in \mathbb{R}$ of a house ($150\text{ m}^2$, built $20\text{ years}$ ago) from a training record of $n$ historical sales $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$.
- **Features and labels:** Each example $i$ is described by a feature vector $\mathbf{x}^{(i)} = (x_1^{(i)}, \ldots, x_d^{(i)})^\top \in \mathbb{R}^d$ and an observed scalar target $y^{(i)} \in \mathbb{R}$.
- **Core assumption:** On average, the target varies as an affine combination of the features plus unmodeled measurement noise.

::: {.d2l-note}
Fitting a plane to historical sales yields a closed-form pricing rule that generalizes immediately to any unseen house $\mathbf{x}$.
:::
:::

::: {.col .fig .big}
![](../img/mdl-linreg-house-prices.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="The Affine Model: Weights and Bias"}
[The Model Class]{.kicker}

A linear regression model forms its prediction $\hat{y}$ as a weighted sum of the $d$ input features plus a scalar intercept:

$$\textrm{price} = w_{\textrm{area}} \cdot \textrm{area} + w_{\textrm{age}} \cdot \textrm{age} + b$$

- **Marginal slopes ($\mathbf{w} \in \mathbb{R}^d$):** With $w_{\textrm{area}} = 3{,}000\text{ \$/m}^2$ and $w_{\textrm{age}} = -1{,}000\text{ \$/yr}$, each extra square meter adds $\$3{,}000$ while each year of age subtracts $\$1{,}000$.
- **Intercept / bias ($b \in \mathbb{R}$):** With $b = \$50{,}000$, our $150\text{ m}^2$, $20$-year-old house is priced at $\$450{,}000 - \$20{,}000 + \$50{,}000 = \$480{,}000$.
- **Single-example inner product:** Collecting weights into $\mathbf{w} = (w_1, \ldots, w_d)^\top \in \mathbb{R}^d$ writes the prediction compactly as $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$.

::: {.d2l-note .rule}
Strictly speaking, $\mathbf{w}^\top \mathbf{x} + b$ is an **affine** function of $\mathbf{x}$. Without the bias $b$, the fitted hyperplane is forced to pass through the origin $\mathbf{x} = \mathbf{0}$, severely restricting the model even when no real house has zero area.
:::
:::

::: {.slide title="Vectorizing Across a Dataset With the Design Matrix"}
[Matrix Formulation]{.kicker}

To evaluate predictions across all $n$ examples simultaneously, we stack the transpose of each feature vector $\mathbf{x}^{(i)} \in \mathbb{R}^d$ as a row of the design matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b, \qquad \mathbf{X} = \begin{bmatrix} (\mathbf{x}^{(1)})^\top \\ \vdots \\ (\mathbf{x}^{(n)})^\top \end{bmatrix} \in \mathbb{R}^{n \times d}$$

- **Shape alignment:** Multiplying $\mathbf{X}$ of shape $(n, d)$ by column vector $\mathbf{w}$ of shape $(d, 1)$ yields $\mathbf{X}\mathbf{w}$ of shape $(n, 1)$, and broadcasting adds the scalar $b$ to all $n$ rows.
- **Absorbing the bias:** Appending a constant column of ones produces the augmented matrix $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}] \in \mathbb{R}^{n \times (d+1)}$ and parameter vector $\tilde{\mathbf{w}} = (\mathbf{w}^\top, b)^\top \in \mathbb{R}^{d+1}$, reducing the affine map to a pure matrix-vector product $\hat{\mathbf{y}} = \tilde{\mathbf{X}}\tilde{\mathbf{w}}$.
- **Hardware efficiency:** Expressing all $n$ dot products as a single GEMM call dispatches the entire batch to fused SIMD or GPU tensor cores.
:::

::: {.slide title="The Squared Error Loss Function"}
[Measuring Misfit]{.kicker}

We quantify how closely predictions $\hat{y}^{(i)}$ match observed targets $y^{(i)}$ by averaging half the squared vertical residuals across the training set:

::: {.cols .vc}
::: {.col}
$$L(\mathbf{w}, b) = \frac{1}{2n} \sum_{i=1}^{n} \bigl(\hat{y}^{(i)} - y^{(i)}\bigr)^2$$

- **Residual ($r^{(i)} = \hat{y}^{(i)} - y^{(i)}$):** The signed vertical gap between the predicted point on the line and the observed data point.
- **Why the factor of $1/2$:** Purely algebraic convenience—it cancels the factor of $2$ that drops down upon differentiation, leaving $\operatorname{argmin}_{\mathbf{w}, b} L(\mathbf{w}, b)$ unchanged.
- **Why divide by $n$:** Normalizing by sample size keeps the numerical magnitude of the loss and its gradients invariant when dataset size changes.
:::

::: {.col .fig}
![](../img/fit-linreg.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Pitfall: Squared Loss Sensitivity to Outliers"}
[Statistical Failure Mode]{.kicker}

Because squared error penalizes residuals quadratically ($r^2$), a single corrupted label exerts a restoring force proportional to its distance from the line:

@linear-regression-loss-function

::: {.d2l-note .warn}
On 20 points along $y = 2x$, corrupting a single label to $10{,}000$ yanks the least-squares slope from $2.00$ to $23.16$! Under mean absolute error ($\frac{1}{n}\sum_i |\hat{y}^{(i)} - y^{(i)}|$), every point has unit vote $\pm 1$, so the 19 clean points easily outvote the outlier and preserve $w = 2.00$.
:::
:::

::: {.slide title="Closed-Form Solution: The Normal Equations"}
[Analytic Optimization]{.kicker}

In augmented notation $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}] \in \mathbb{R}^{n \times (d+1)}$, the empirical loss is a convex quadratic bowl $L(\tilde{\mathbf{w}}) = \frac{1}{2n}\|\tilde{\mathbf{X}}\tilde{\mathbf{w}} - \mathbf{y}\|_2^2$:

$$\nabla_{\tilde{\mathbf{w}}} L(\tilde{\mathbf{w}}) = \frac{1}{n}\tilde{\mathbf{X}}^\top\bigl(\tilde{\mathbf{X}}\tilde{\mathbf{w}} - \mathbf{y}\bigr) = \mathbf{0} \quad\Longleftrightarrow\quad \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}\, \hat{\tilde{\mathbf{w}}} = \tilde{\mathbf{X}}^\top \mathbf{y}$$

- **Pythagorean decomposition:** For any candidate $\tilde{\mathbf{w}}$, expanding $\|\mathbf{y} - \tilde{\mathbf{X}}\tilde{\mathbf{w}}\|_2^2$ around a solution $\hat{\tilde{\mathbf{w}}}$ yields $\|\mathbf{y} - \tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}}\|_2^2 + \|\tilde{\mathbf{X}}(\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}})\|_2^2$ because the cross term $(\mathbf{y} - \tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}})^\top\tilde{\mathbf{X}}$ vanishes identically.
- **Uniqueness condition:** When the $d + 1$ columns of $\tilde{\mathbf{X}}$ are linearly independent (requiring $n \geq d + 1$), the Gram matrix $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ is strictly positive definite and $\hat{\tilde{\mathbf{w}}} = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\mathbf{y}$ is unique.
- **Collinearity ($d > n$ or duplicate features):** When $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ is singular, infinitely many weight vectors achieve the exact same minimum loss; `lstsq` returns the minimum-norm solution $\tilde{\mathbf{X}}^\dagger \mathbf{y}$ via QR or SVD.
:::

::: {.slide title="Worked Example & Numerical Pitfall of Matrix Inversion"}
[Normal Equations in Code]{.kicker}

Consider fitting a line $w_0 + w_1 x$ to three points $(0, 1), (1, 1), (2, 0)$ with design matrix rows $(1, x^{(i)})$:

$$\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix}, \quad \tilde{\mathbf{X}}^\top \mathbf{y} = \begin{bmatrix} 2 \\ 1 \end{bmatrix} \quad\Longrightarrow\quad \hat{\tilde{\mathbf{w}}} = \begin{bmatrix} 7/6 \\ -1/2 \end{bmatrix} \approx (1.1667, -0.5000)^\top$$

@linear-regression-the-normal-equations

::: {.d2l-note .warn}
**Never compute `inv(X.T @ X) @ X.T @ y` explicitly in production:** Forming $\mathbf{X}^\top\mathbf{X}$ squares the condition number $\kappa(\mathbf{X}^\top\mathbf{X}) = \kappa(\mathbf{X})^2$, doubling the loss of floating-point significant digits. Dedicated solvers (`lstsq`) factorize $\mathbf{X}$ directly via Householder QR or SVD in $O(nd^2)$ flops.
:::
:::

::: {.slide title="Geometric View: Least Squares as Orthogonal Projection"}
[Why They Are Called the "Normal" Equations]{.kicker}

In observation space $\mathbb{R}^n$, varying $\tilde{\mathbf{w}} \in \mathbb{R}^{d+1}$ sweeps out a $(d+1)$-dimensional flat subspace—the column space $\operatorname{col}(\tilde{\mathbf{X}})$:

::: {.cols .vc}
::: {.col}
- **Closest point in a subspace:** The noisy target vector $\mathbf{y} \in \mathbb{R}^n$ almost never lies inside $\operatorname{col}(\tilde{\mathbf{X}})$. Minimizing $\|\mathbf{y} - \tilde{\mathbf{X}}\tilde{\mathbf{w}}\|_2^2$ finds the point $\hat{\mathbf{y}} = \tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}} \in \operatorname{col}(\tilde{\mathbf{X}})$ closest to $\mathbf{y}$ in Euclidean distance.
- **Orthogonality of the residual:** By Euclidean geometry, the shortest segment from $\mathbf{y}$ to $\operatorname{col}(\tilde{\mathbf{X}})$ is perpendicular (*normal*) to the subspace, meaning the residual $\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y}$ satisfies:
  $$\tilde{\mathbf{X}}^\top \mathbf{r} = \mathbf{0}$$
- **Zero-sum residuals:** Because the bias column $\mathbf{1}$ lies in $\operatorname{col}(\tilde{\mathbf{X}})$, $\mathbf{1}^\top\mathbf{r} = \sum_{i=1}^n r^{(i)} = 0$ holds for every least-squares fit with an intercept.
:::

::: {.col .fig .big}
![](../img/mdl-linreg-projection.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Why Deep Learning Uses Minibatch SGD Instead"}
[From Closed Form to Iterative Optimization]{.kicker}

If linear regression has an exact one-line analytic solution, why train models iteratively with stochastic gradients?

1. **Cubic parameter scaling ($O(nd^2 + d^3)$):** Forming $\mathbf{X}^\top\mathbf{X} \in \mathbb{R}^{d \times d}$ takes $O(nd^2)$ flops and factoring it takes $O(d^3)$ flops and $O(d^2)$ memory. For a neural network with $d = 10^8$ parameters, storing a single $d \times d$ matrix would require $40\text{ petabytes}$ of RAM.
2. **Dataset redundancy ($n \gg |\mathcal{B}|$):** Full-batch gradient descent costs $O(nd)$ per step because it averages over all $n = 10^7$ examples before moving once, even though a random sample of $|\mathcal{B}| = 64$ examples already points in nearly the same descent direction.
3. **Non-analytic objectives:** Replace the squared loss with cross-entropy or insert a single nonlinear hidden layer, and no closed-form normal equations exist at all.

::: {.d2l-note}
Minibatch SGD decouples the per-step compute cost from dataset size $n$: each update touches only $|\mathcal{B}|$ examples and costs $O(|\mathcal{B}|d)$ time and $O(d)$ parameter memory.
:::
:::

::: {.slide title="Minibatch Stochastic Gradient Descent Mechanics"}
[The Workhorse Update Rule]{.kicker}

At step $t$, we sample a random subset $\mathcal{B}_t \subset \{1, \ldots, n\}$ of size $|\mathcal{B}|$ and step opposite the minibatch mean gradient with learning rate $\eta > 0$:

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \bigl(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\bigr), \qquad b \leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \bigl(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\bigr)$$

- **Error-driven correction:** Each example $i \in \mathcal{B}_t$ pulls $\mathbf{w}$ along $-\mathbf{x}^{(i)}$ in direct proportion to its signed prediction residual $r^{(i)} = \hat{y}^{(i)} - y^{(i)}$.
- **Unbiased gradient estimator:** Under uniform sampling, $\mathbb{E}[\nabla L_{\mathcal{B}_t}(\mathbf{w}, b)] = \nabla L(\mathbf{w}, b)$, while the estimator's standard error decays as $1 / \sqrt{|\mathcal{B}|}$.
- **Parameters vs. hyperparameters:** The weights $\mathbf{w}$ and bias $b$ are learned during training; the learning rate $\eta$ and batch size $|\mathcal{B}|$ (typically $32$ to $256$) are *hyperparameters* set prior to training and tuned on validation data.
:::

::: {.slide title="Gradient Noise Ball and Learning-Rate Scaling"}
[Stochastic Dynamics]{.kicker}

Even at the exact empirical minimizer $\hat{\mathbf{w}}$ where the full-batch gradient $\nabla L(\hat{\mathbf{w}}) = \mathbf{0}$, individual noisy examples have nonzero residuals $r^{(i)} \neq 0$:

::: {.cols .vc}
::: {.col}
- **Full-batch GD (black):** Computes the exact gradient over all $n$ examples per step, marching smoothly across elliptical contours into $\hat{\mathbf{w}}$.
- **Minibatch SGD (color):** Descends rapidly in $n / |\mathcal{B}|$ cheap steps per epoch, then fluctuates perpetually inside a terminal **noise ball** around $\hat{\mathbf{w}}$:
  $$\mathbb{E}\bigl[\|\mathbf{w}_t - \hat{\mathbf{w}}\|_2^2\bigr] \propto \frac{\eta\,\sigma^2}{|\mathcal{B}|}$$
- **Controlling the radius:** Quartering $\eta$ (right panel) or quadrupling $|\mathcal{B}|$ halves the noise ball radius; decaying $\eta_t \to 0$ converges to $\hat{\mathbf{w}}$.
:::

::: {.col .fig .big}
![](../img/mdl-linreg-gd-sgd.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Vectorization: Python Loops vs. Matrix Kernels"}
[Systems Performance]{.kicker}

Never implement minibatch updates by looping over examples $i \in \mathcal{B}_t$ in Python. Vectorizing turns $n$ scalar inner products into one compiled matrix multiplication $\mathbf{X}\mathbf{w} + b$:

@linear-regression-vectorization-for-speed-3

- **Why Python loops crawl:** Computing `X[i:i+1] @ w + b` 1,000 times pays Python interpreter dispatch, type checking, and kernel launch overhead on every single example.
- **Why `@` flies:** A single `X @ w + b` call pays dispatch overhead once and streams contiguous memory through SIMD vector registers or GPU tensor cores, running hundreds of times faster.
- **Asynchronous timing pitfall:** GPU frameworks (CUDA, JAX, MXNet) enqueue kernels asynchronously and return to Python immediately; timing benchmarks must synchronize the device (`torch.cuda.synchronize()`, `jax.block_until_ready`) before stopping the clock.
:::

::: {.slide title="Probabilistic Foundation: The Gaussian Noise Model"}
[Why Squared Error?]{.kicker}

In 1809, Gauss asked what probability distribution of measurement errors $\epsilon$ makes least squares the most probable parameter estimate:

::: {.cols .vc}
::: {.col}
- **Additive Gaussian noise ($y = \mathbf{w}^\top \mathbf{x} + b + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$):**
  $$p(y \mid \mathbf{x}; \mathbf{w}, b) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\Bigl(-\frac{(y - \mathbf{w}^\top \mathbf{x} - b)^2}{2\sigma^2}\Bigr)$$

- **Homoscedastic normal scatter:** Every conditional density $p(y \mid \mathbf{x})$ is a Gaussian bell curve centered on the regression line $\mathbf{w}^\top\mathbf{x} + b$ with constant standard deviation $\sigma$.
- **Light quadratic tails:** Because $(y - \hat{y})^2$ sits inside $\exp(-\cdot / 2\sigma^2)$, a $5\sigma$ residual has density $e^{-12.5} \approx 4 \times 10^{-6}$ of the peak.
:::

::: {.col .fig}
![](../img/mdl-linreg-gaussian-noise.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Gaussian Maximum Likelihood Yields Least Squares"}
[Statistical Derivation]{.kicker}

For $n$ independent observations $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$, the conditional dataset likelihood factors into $\prod_{i=1}^n p(y^{(i)} \mid \mathbf{x}^{(i)}; \mathbf{w}, b)$. Maximizing likelihood is equivalent to minimizing the negative log-likelihood (NLL):

$$-\log p(\mathbf{y} \mid \mathbf{X}; \mathbf{w}, b) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^n \bigl(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\bigr)^2$$

- **Invariance to $\sigma^2$:** The first term $\frac{n}{2}\log(2\pi\sigma^2)$ is a constant independent of $(\mathbf{w}, b)$.
- **Exact equivalence:** The second term is $\frac{n}{\sigma^2} L(\mathbf{w}, b)$—our empirical squared error loss scaled by the positive constant $n / \sigma^2$!

::: {.d2l-note .rule}
**The MLE Loss Recipe:** Minimizing the squared error loss produces the exact maximum-likelihood estimator $(\hat{\mathbf{w}}_{\textrm{MLE}}, \hat{b}_{\textrm{MLE}})$ under additive homoscedastic Gaussian noise, regardless of the noise variance $\sigma^2$.
:::
:::

::: {.slide title="Designing Losses From Alternative Noise Distributions"}
[Beyond Gaussian Noise]{.kicker}

Replacing the Gaussian noise density $p(\epsilon) \propto \exp(-l(\epsilon))$ systematically derives robust and domain-specific regression losses via $-\log p(y \mid \mathbf{x})$:

::: {.cols .vc}
::: {.col}
- **Laplace noise $\to$ $\ell_1$ Absolute Error:** Heavy exponential tails $p(\epsilon) = \frac{1}{2\tau}e^{-|\epsilon|/\tau}$ yield $l(\hat{y}, y) = |\hat{y} - y|$, whose subgradient $\operatorname{sign}(\hat{y} - y) \in \{\pm 1\}$ bounds the pull of any outlier.
- **Huber loss (smooth + robust):** Splices a quadratic bowl $\frac{1}{2}r^2$ for small residuals ($|r| \leq \delta$) onto linear slopes $\delta(|r| - \delta/2)$ for large residuals ($|r| > \delta$).
- **Poisson counts ($y \in \{0, 1, \ldots\}$):** Modeling rate $\log \nu(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b$ yields convex NLL $l = \nu(\mathbf{x}) - y\log\nu(\mathbf{x})$.
:::

::: {.col .fig .big}
![](../img/mdl-linreg-loss-menu.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Linear Regression as a Single-Layer Neural Network"}
[Bridge to Deep Learning]{.kicker}

In the language of neural networks, linear regression is a single artificial neuron—a **fully connected (dense) layer** with $d$ inputs and $1$ linear output unit:

::: {.cols .vc}
::: {.col}
- **Input layer ($x_1, \ldots, x_d$):** Holds the $d$ raw feature coordinates (not counted in layer depth because it holds no tunable parameters).
- **Synaptic weights and output ($o_1$):** Every input node $x_j$ connects to the output node $o_1$ with weight $w_j$, computing $o_1 = \sum_{j=1}^d w_j x_j + b$.
- **Why stack layers later:** Composing multiple affine layers without nonlinearities collapses into a single affine map; inserting elementwise activations in Chapter 4 unlocks universal approximation.
:::

::: {.col .fig}
![](../img/singleneuron.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Concept Check: Orthogonality, Noise Balls & Likelihood"}
[Check Your Understanding]{.kicker}

Work out these three questions from first principles before revealing the solutions:

1. **Zero-sum residuals:** Why do the residuals $r^{(i)} = \hat{y}^{(i)} - y^{(i)}$ of *every* least-squares fit with a bias $b$ satisfy $\sum_{i=1}^n r^{(i)} = 0$?
2. **Batch size vs. noise ball:** If you increase the minibatch size from $|\mathcal{B}| = 32$ to $|\mathcal{B}| = 128$ at a fixed learning rate $\eta$, how does the steady-state squared distance $\mathbb{E}[\|\mathbf{w}_t - \hat{\mathbf{w}}\|_2^2]$ change, and what learning rate restores the original noise ball?
3. **Outlier likelihood ratio:** Under unit-scale noise, how much less likely is a residual of $\epsilon = 10$ than $\epsilon = 1$ under $\mathcal{N}(0, 1)$ versus Laplace $p(\epsilon) = \frac{1}{2}e^{-|\epsilon|}$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. The augmented design matrix $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}]$ contains the all-ones column $\mathbf{1}$, so the normal equations $\tilde{\mathbf{X}}^\top\mathbf{r} = \mathbf{0}$ enforce $\mathbf{1}^\top\mathbf{r} = \sum_{i=1}^n r^{(i)} = 0$.
2. Because $\mathbb{E}[\|\mathbf{w}_t - \hat{\mathbf{w}}\|_2^2] \propto \eta / |\mathcal{B}|$, quadrupling $|\mathcal{B}|$ cuts the squared distance by $4\times$ (halving the ball's radius); scaling the learning rate linearly to $4\eta$ restores the original ratio $\eta / |\mathcal{B}|$.
3. Gaussian ratio: $e^{-(10^2 - 1^2)/2} = e^{-49.5} \approx 3 \times 10^{-22}$ (virtually impossible, forcing the line to tilt). Laplace ratio: $e^{-(10 - 1)} = e^{-9} \approx 1.2 \times 10^{-4}$ (rare but plausible, so the slope ignores the outlier).
:::
:::

::: {.slide title="Section 2.1 Summary & Key Takeaways"}
[Summary]{.kicker}

| Concept | Mathematical Form | Computational Cost / Scaling | Role & Practical Takeaway |
|---|---|---|---|
| **Affine Model** | $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b = \tilde{\mathbf{X}}\tilde{\mathbf{w}}$ | $O(nd)$ flops per full dataset pass | Single fully connected layer; bias $b$ frees the hyperplane from the origin |
| **Squared Error Loss** | $L(\mathbf{w}, b) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|_2^2$ | $O(n)$ reduction over residuals | Exact MLE under $\mathcal{N}(0, \sigma^2)$ noise; sensitive to heavy-tailed outliers |
| **Normal Equations** | $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}} = \tilde{\mathbf{X}}^\top\mathbf{y}$ | $O(nd^2 + d^3)$ via QR/SVD (`lstsq`) | Orthogonal projection of $\mathbf{y}$ onto $\operatorname{col}(\tilde{\mathbf{X}})$; infeasible for large $d$ |
| **Minibatch SGD** | $\tilde{\mathbf{w}} \leftarrow \tilde{\mathbf{w}} - \frac{\eta}{|\mathcal{B}|}\tilde{\mathbf{X}}_{\mathcal{B}}^\top(\tilde{\mathbf{X}}_{\mathcal{B}}\tilde{\mathbf{w}} - \mathbf{y}_{\mathcal{B}})$ | $O(|\mathcal{B}|d)$ flops per step, independent of $n$ | Unbiased gradient with variance $\propto 1/|\mathcal{B}|$; settles in noise ball $\propto \eta/|\mathcal{B}|$ |
| **Robust Alternatives** | $\ell_1$: $\|\hat{\mathbf{y}} - \mathbf{y}\|_1 / n$; Huber $l_\delta$ | $O(|\mathcal{B}|d)$ per subgradient step | Derived from Laplace MLE; bounds each outlier's gradient contribution to $\pm 1$ |
:::
