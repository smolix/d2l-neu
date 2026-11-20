```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Calculus
:label:`sec_calculus`

In Ancient Greece, Archimedes calculated the area of a circle by inscribing regular polygons with increasing numbers of sides $n \to \infty$ (:numref:`fig_circle_area`): as the triangles become thinner, their height approaches $r$ and their combined bases approach the circumference $2\pi r$, giving area $\frac{1}{2} r (2\pi r) = \pi r^2$. Approximating curved quantities by simple linear pieces in the limit of small steps lies at the root of both integral calculus and *differential calculus*.

![Regular polygons with 4, 5, 6, and 8 sides inscribed in a circle. As the number of sides grows, the area of the polygon approaches the area of the circle.](../img/polygon-circle.svg)
:label:`fig_circle_area`

In deep learning, differential calculus is the engine of *optimization*. Training a model means adjusting its parameters to minimize a scalar loss function measuring how poorly the model fits its training examples. With millions or billions of parameters, probing each parameter by trial and error would require millions of forward passes per update. Instead, the *derivative* gives the instantaneous rate at which the loss changes with respect to each parameter; collecting all partial derivatives into the *gradient* vector identifies the direction of steepest ascent, and the *chain rule* assembles gradients across nested layers in a single backward pass.

In this section, we develop derivatives, gradients, and the chain rule on a running linear-regression example small enough that every number can be verified by hand. In :numref:`sec_autograd`, we will see how deep learning frameworks automate these calculations for arbitrary programs.

```{.python .input #calculus}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #calculus}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #calculus}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #calculus}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Derivatives

### The Derivative as a Limit

Suppose a single-parameter model predicts the constant value $w$ for every example, and we evaluate it on three training examples with labels $1$, $1$, and $0$. Summing the squared errors gives

$$L(w) = (w-1)^2 + (w-1)^2 + (w-0)^2 = 3w^2 - 4w + 2.$$

At $w = 1$, the loss is $L(1) = 1$:

```{.python .input #calculus-derivatives-and-differentiation-1}
def loss(w):
    return 3 * w ** 2 - 4 * w + 2
```

To see how $L(w)$ responds to a small perturbation $h \neq 0$, we evaluate the difference quotient $\frac{L(1+h) - L(1)}{h}$ as $h$ shrinks from $10^{-1}$ to $10^{-5}$:

```{.python .input #calculus-derivatives-and-differentiation-2}
for h in 10.0 ** -np.arange(1, 6):
    print(f'h = {h:.0e}   ratio = {(loss(1 + h) - loss(1)) / h:.5f}')
```

Expanding $L(1+h) = 1 + 2h + 3h^2$ shows that $\frac{L(1+h) - L(1)}{h} = 2 + 3h \to 2$ as $h \to 0$. Formally, for $f: \mathbb{R} \rightarrow \mathbb{R}$, the *derivative* of $f$ at $x$ is the limit

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$
:eqlabel:`eq_derivative`

Geometrically, the difference quotient is the slope of the *secant line* through $P = (x, f(x))$ and $Q = (x+h, f(x+h))$; as $h \to 0$, the secant rotates into the *tangent line* at $P$ with slope $f'(x)$ (:numref:`fig_calc_secant_tangent`).

![The difference quotient $(f(x+h)-f(x))/h$ is the slope of the secant through $P$ and $Q$. As $h \to 0$, $Q$ slides toward $P$ and the secant turns into the tangent at $P$, whose slope is $f'(x)$.](../img/calculus-secant-tangent.svg)
:label:`fig_calc_secant_tangent`

Not every function is differentiable everywhere: $|x|$ has a corner at $x = 0$ (slope $-1$ from the left, $+1$ from the right). Even worse, discrete classification accuracy is flat ($f'(x) = 0$) almost everywhere and jumps discontinuously where a prediction flips, providing no gradient signal—which is why classifiers are trained by minimizing smooth surrogate losses such as cross-entropy.

### Rules of Differentiation

Given $y = f(x)$, we write $f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x)$. Common elementary derivatives are

$$
\frac{d}{dx} C = 0, \quad \frac{d}{dx} x^n = n x^{n-1}, \quad
\frac{d}{dx} e^x = e^x, \quad \frac{d}{dx} \log x = \frac{1}{x} \quad (x > 0),
$$

and combinations of differentiable functions $f, g$ obey

$$
\begin{aligned}
\frac{d}{dx} [C f(x)] & = C f'(x), &
\frac{d}{dx} [f(x) + g(x)] & = f'(x) + g'(x), \\
\frac{d}{dx} [f(x) g(x)] & = f'(x) g(x) + f(x) g'(x), &
\frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{f'(x) g(x) - f(x) g'(x)}{g(x)^2}.
\end{aligned}
$$

:numref:`fig_calc_product_rule` illustrates the product rule geometrically via the area of a $f(x) \times g(x)$ rectangle (full proofs appear in :numref:`sec_mdl-single_variable_calculus`). Applying these rules to $L(w) = 3w^2 - 4w + 2$ immediately gives $L'(w) = 6w - 4$, so $L'(1) = 2$.

![The product rule as the area of a rectangle. When $x$ grows by $h$, the sides $f(x)$ and $g(x)$ grow by about $f'(x) h$ and $g'(x) h$, and the area grows by two thin strips and a corner of area $f'(x) g'(x) h^2$, which is negligible for small $h$.](../img/calculus-product-rule.svg)
:label:`fig_calc_product_rule`

### First-Order Linear Approximation and Minima

Multiplying $\frac{f(x+h) - f(x)}{h} \approx f'(x)$ by $h$ gives the *first-order linear approximation*

$$f(x + h) \approx f(x) + f'(x)\, h.$$
:eqlabel:`eq_calculus-linear-1d`

Because $L'(1) = 2 > 0$, decreasing $w$ reduces the loss: stepping opposite the derivative via $w \leftarrow w - \eta L'(w)$ with step size $\eta = 0.1$ moves $w$ from $1$ to $1 - 0.1(2) = 0.8$, lowering $L$ from $1$ to $L(0.8) = 0.72$ (close to the tangent prediction $1 + 2(-0.2) = 0.60$, with the $0.12$ difference equal to the quadratic curvature $3h^2$). At any smooth local minimum, the derivative must vanish ($f'(x) = 0$); setting $L'(w) = 6w - 4 = 0$ yields the unique global minimizer $w^* = 2/3$ (the mean of the labels $1, 1, 0$).

```{.python .input #calculus-visualization-utilities-5}
w = np.linspace(0, 2, 101)
d2l.plot(w, [loss(w), 2 * w - 1], 'w', 'L(w)',
         legend=['L(w)', 'tangent at w = 1'])
```

## Partial Derivatives and Gradients
:label:`subsec_calculus-grad`

### Partial Derivatives

Now give our three-example problem scalar inputs $0$, $1$, and $2$ paired with labels $1$, $1$, and $0$, and fit an affine line $w_1 + w_2 \cdot \textrm{input}$:

$$L(w_1, w_2) = (w_1 - 1)^2 + (w_1 + w_2 - 1)^2 + (w_1 + 2 w_2 - 0)^2.$$

For a multivariate function $y = f(x_1, \ldots, x_n)$, the *partial derivative* with respect to $x_i$ perturbs only coordinate $i$ while holding all other variables fixed:

$$\frac{\partial f}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

At $(w_1, w_2) = (1, 0)$, holding $w_2 = 0$ fixed gives $L(w_1, 0) = 3w_1^2 - 4w_1 + 2$, so $\frac{\partial L}{\partial w_1}(1, 0) = 2$; holding $w_1 = 1$ fixed gives $L(1, w_2) = 5w_2^2 + 4w_2 + 1$, with derivative $10w_2 + 4$, so $\frac{\partial L}{\partial w_2}(1, 0) = 4$ (:numref:`fig_calc_partial_slices`).

![The loss $L(w_1, w_2)$ of the line model near $\mathbf{w} = (1, 0)$ (black dot). Holding $w_2 = 0$ cuts out the orange curve, the loss of the constant model; holding $w_1 = 1$ cuts out the green curve. Their tangents at the dot (dashed) have slopes $\partial L / \partial w_1 = 2$ and $\partial L / \partial w_2 = 4$.](../img/calculus-partial-slices.svg)
:label:`fig_calc_partial_slices`

In matrix--vector form (:numref:`sec_linear-algebra`), writing each example as a row $(1, \textrm{input}_i)$ of $\mathbf{X}$ gives

$$\mathbf{X} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2.$$

We can check $(2, 4)$ numerically by perturbing one coordinate at a time by $h = 10^{-6}$ in `float64`:

```{.python .input #calculus-partial-derivatives-1}
X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # rows [1, input]
y = np.array([1.0, 1.0, 0.0])

def line_loss(w):
    return np.sum((X @ w - y) ** 2)

def numerical_gradient(f, w, h=1e-6):
    """Finite differences of f at w, moving one coordinate at a time."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = h
        grad[i] = (f(w + step) - f(w)) / h
    return grad
```

```{.python .input #calculus-partial-derivatives-2}
w = np.array([1.0, 0.0])
numerical_gradient(line_loss, w)
```

Note that `numerical_gradient` requires $n$ extra evaluations of $f$ for $n$ parameters—prohibitive when $n = 10^8$.

### Numerical Precision Limits of Finite Differences

Finite differences also suffer from floating-point cancellation when $h$ is too small. In finite precision (machine epsilon $\epsilon \approx 2.2 \times 10^{-16}$ in `float64`, $\epsilon \approx 1.2 \times 10^{-7}$ in `float32`), subtracting nearly equal numbers $f(x+h) - f(x)$ cancels leading digits (*catastrophic cancellation*) and divides the $\sim \epsilon$ rounding residual by $h$. Thus *truncation error* ($\propto h$) and *rounding error* ($\propto \epsilon / h$) compete:

```{.python .input #calculus-derivatives-and-differentiation-3}
def quotient_error(f, x, df, h):
    """Error of (f(x + h) - f(x)) / h as an estimate of f'(x) = df."""
    return abs((f(x + h) - f(x)) / h - df)

h = np.logspace(-1, -16, 61)  # 61 values of h, evenly spaced on a log scale
```

```{.python .input #calculus-finite-differences-1}
%%tab mxnet
h32 = mnp.array(h, dtype='float32')
```

```{.python .input #calculus-finite-differences-1}
%%tab pytorch
h32 = torch.tensor(h, dtype=torch.float32)
```

```{.python .input #calculus-finite-differences-1}
%%tab tensorflow
h32 = tf.constant(h, dtype=tf.float32)
```

```{.python .input #calculus-finite-differences-1}
%%tab jax
h32 = jnp.asarray(h, dtype=jnp.float32)
```

```{.python .input #calculus-finite-differences-2}
eps32 = np.finfo(np.float32).eps
d2l.plot(h, [quotient_error(loss, 1, 2, h), quotient_error(loss, 1, 2, h32),
             3 * h, eps32 / h], 'h', 'error', xscale='log', yscale='log',
         ylim=(1e-10, 1e1), legend=['float64', 'float32', '3h', 'ε/h'],
         figsize=(5, 3.5))
```

Balancing $3h \approx \epsilon / h$ puts the optimal step at $h^* \approx \sqrt{\epsilon / 3}$, losing half the format's digits (best error $\sim 10^{-8}$ at $h \approx 10^{-8}$ in `float64`, and $\sim 10^{-4}$ at $h \approx 10^{-4}$ in `float32`). Analytical differentiation via the chain rule avoids both the $O(n)$ cost and the $\sqrt{\epsilon}$ precision loss, leaving `float64` finite differences for unit-test gradient checking.

### The Gradient and Steepest Descent

Concatenating all $n$ partial derivatives of $f: \mathbb{R}^n \to \mathbb{R}$ forms the *gradient* vector $\nabla_{\mathbf{x}} f(\mathbf{x}) = (\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n})^\top$ (matching the shape of $\mathbf{x}$), which generalizes :eqref:`eq_calculus-linear-1d` to simultaneous perturbations $\mathbf{h} \in \mathbb{R}^n$:

$$f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{h}.$$
:eqlabel:`eq_calculus-linear`

For a step $\mathbf{h} = t\mathbf{d}$ of length $t > 0$ along unit vector $\mathbf{d}$ ($\|\mathbf{d}\|_2 = 1$), :eqref:`eq_dot_cos` gives $f(\mathbf{x} + t\mathbf{d}) - f(\mathbf{x}) \approx t\,\|\nabla f(\mathbf{x})\|\cos\theta$. Thus $\nabla f(\mathbf{x})$ points in the direction of *steepest ascent* ($\theta = 0^\circ$), $-\nabla f(\mathbf{x})$ points in the direction of *steepest descent* ($\theta = 180^\circ$), and $\nabla f(\mathbf{x})$ is perpendicular to the level set passing through $\mathbf{x}$ ($\theta = 90^\circ$) (:numref:`fig_calc_gradient_field`).

![Level sets of the loss $L(w_1, w_2)$ of the line model. The gradients (red) are perpendicular to the level set through their base points and point uphill. From $\mathbf{w} = (1, 0)$ (black dot), gradient descent with $\eta = 0.1$ (orange) starts along $-\nabla L$, which does not point at the minimum, and zigzags toward it.](../img/calculus-gradient-field.svg)
:label:`fig_calc_gradient_field`

This motivates *gradient descent*, $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$: from $\mathbf{w} = (1, 0)$ with learning rate $\eta = 0.1$, one step moves to $(1, 0) - 0.1(2, 4) = (0.8, -0.4)$, cutting the loss from $1.0$ to $0.4$ (while zigzagging across the elliptical contours toward the global minimum $\mathbf{w}^* = (7/6, -1/2)$; see :numref:`chap_optimization`).

### Matrix Calculus Identities

Three vector gradient identities (:numref:`sec_mdl-matrix-calculus`) allow us to differentiate vector expressions without writing out coordinates:

$$\nabla_{\mathbf{x}}\, (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}, \qquad \nabla_{\mathbf{x}}\, (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}, \qquad \nabla_{\mathbf{x}} \|\mathbf{x}\|^2 = 2\mathbf{x}.$$

Expanding $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = \mathbf{w}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{w} - 2(\mathbf{X}^\top\mathbf{y})^\top\mathbf{w} + \mathbf{y}^\top\mathbf{y}$ and applying these rules gives

$$\nabla L(\mathbf{w}) = 2\mathbf{X}^\top \mathbf{X} \mathbf{w} - 2\mathbf{X}^\top \mathbf{y} = 2\mathbf{X}^\top (\mathbf{X}\mathbf{w} - \mathbf{y}).$$
:eqlabel:`eq_calculus-line-grad`

## The Chain Rule
:label:`subsec_calculus-chain-rule`

Deep networks compose many nested layers $y = f_L(f_{L-1}(\cdots f_1(\mathbf{x})\cdots))$; differentiating compositions without expanding them relies on the *chain rule*.

For scalar functions $z = f(u)$ and $u = g(x)$, rates of change multiply: $\frac{dz}{dx} = \frac{dz}{du}\frac{du}{dx}$. More generally, if $z = f(u_1, \ldots, u_m)$ where each $u_j = g_j(x_1, \ldots, x_n)$, rates multiply along each path $x_i \to u_j \to z$ and sum across all parallel paths:

$$\frac{\partial z}{\partial x_i} = \sum_{j=1}^m \frac{\partial z}{\partial u_j} \frac{\partial u_j}{\partial x_i} \quad \Longrightarrow \quad \nabla_{\mathbf{x}} z = \mathbf{J}^\top \nabla_{\mathbf{u}} z,$$

where $\mathbf{J} \in \mathbb{R}^{m \times n}$ is the *Jacobian matrix* of $\mathbf{u}$ with respect to $\mathbf{x}$, with entries $J_{ji} = \frac{\partial u_j}{\partial x_i}$. For an affine map $\mathbf{u} = \mathbf{A}\mathbf{x} - \mathbf{y}$, the Jacobian is simply $\mathbf{J} = \mathbf{A}$.

Decomposing our line loss $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ into two stages—(1) inner residuals $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ with Jacobian $\mathbf{J} = \mathbf{X}$, and (2) outer squared norm $L = \|\mathbf{r}\|^2$ with gradient $\nabla_{\mathbf{r}} L = 2\mathbf{r}$—the multivariate chain rule immediately recovers $\nabla_{\mathbf{w}} L = \mathbf{X}^\top(2\mathbf{r})$ in two passes:

```{.python .input #calculus-functions-of-many-variables}
r = X @ w - y    # forward: errors, from which the loss is r @ r
grad_r = 2 * r   # backward: gradient of the loss with respect to r
X.T @ grad_r     # backward: gradient with respect to w
```

The *forward pass* computes $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ and $L = \mathbf{r}^\top\mathbf{r}$; the *backward pass* (*backpropagation*) multiplies $2\mathbf{r}$ by $\mathbf{X}^\top$, computing the exact gradient with respect to all parameters at roughly the cost of one forward evaluation. In :numref:`sec_autograd`, we hand this mechanical backward pass over to the framework's automatic differentiation engine.

## Summary

The derivative $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ gives the slope of the tangent line and the local linear approximation $f(x+h) \approx f(x) + f'(x)h$. For $f: \mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla_{\mathbf{x}} f(\mathbf{x})$ collects all partial derivatives, satisfies $f(\mathbf{x}+\mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{h}$, and points in the direction of steepest ascent (perpendicular to level sets), motivating gradient descent $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$. Unlike finite differences—which cost $O(n)$ evaluations and lose precision to catastrophic cancellation ($\sim \sqrt{\epsilon}$)—the multivariate chain rule $\nabla_{\mathbf{x}} z = \mathbf{J}^\top \nabla_{\mathbf{u}} z$ propagates exact gradients backward from the loss to all parameters in a single backward pass (*backpropagation*).

## Exercises

1. **Differentiating $x^x$ when both base and exponent vary.** For $x > 0$, consider the function $f(x) = x^x$, in which $x$ appears in both the base and the exponent.
    1. Explain why applying the power rule alone (treating the exponent as constant to get $x \cdot x^{x-1} = x^x$) and applying the exponential rule alone (treating the base as constant to get $x^x \log x$) are both incorrect.
    1. Rewrite $x^x = e^{x \log x}$ and apply the chain rule and product rule to find $f'(x)$. Notice that the true derivative is the *sum* of the two naive answers from Part 1; explain why this happens using the two-variable chain rule applied to $g(u, v) = u^v$ along $u = x, v = x$.

1. **Why squared error yields the mean and absolute error yields the median.** Suppose a single-parameter model predicts the constant value $w \in \mathbb{R}$ for $n$ training examples with scalar labels $y_1, \ldots, y_n$.
    1. Differentiate the sum of squared errors $L_2(w) = \sum_{i=1}^n (w - y_i)^2$ with respect to $w$, set the derivative to zero, and prove that the unique minimizer is the sample *mean* $w^* = \frac{1}{n}\sum_{i=1}^n y_i$.
    1. Now consider the sum of absolute errors $L_1(w) = \sum_{i=1}^n |w - y_i|$. Show that at any point $w$ not equal to one of the labels $y_i$, the derivative $L_1'(w)$ equals the number of labels strictly less than $w$ minus the number of labels strictly greater than $w$. At which values of $w$ does $L_1'(w)$ fail to exist? Conclude that $L_1(w)$ is minimized by any sample *median* of $\{y_1, \ldots, y_n\}$.
    1. For the four labels $\{1, 1, 0, 10\}$, compute the $L_2$ minimizer (mean) and the $L_1$ minimizer (median). What happens to each minimizer if the outlier label $10$ is corrupted to $100$?

1. **Gradient of the Euclidean norm.** Compute the gradient $\nabla_{\mathbf{x}} f(\mathbf{x})$ of the unsquared Euclidean norm $f(\mathbf{x}) = \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$ for any nonzero vector $\mathbf{x} \neq \mathbf{0}$, and verify that $\|\nabla_{\mathbf{x}} f(\mathbf{x})\|_2 = 1$ everywhere it exists. What happens to the gradient at the origin $\mathbf{x} = \mathbf{0}$, and why?

1. **Step-size stability of gradient descent.** Consider minimizing the isotropic quadratic bowl $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ via the gradient descent update $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)$ with learning rate $\eta > 0$.
    1. Starting from $\mathbf{x}_0 = (1, 1)^\top$, compute $\mathbf{x}_1$ and the new objective value $f(\mathbf{x}_1)$ after a single step for three choices of learning rate: $\eta = 0.1$, $\eta = 1.0$, and $\eta = 2.0$.
    1. Write $\mathbf{x}_{k+1}$ in closed form in terms of $\mathbf{x}_k$ and $\eta$. For what range of learning rates $\eta > 0$ does a gradient descent step strictly decrease $f(\mathbf{x})$ from any nonzero starting point $\mathbf{x}_k \neq \mathbf{0}$? Which special value of $\eta$ jumps directly to the global minimum $\mathbf{x}^* = \mathbf{0}$ in a single step, and what happens when $\eta > 1$?
    1. Characterize the stationary points $\mathbf{x}$ at which a gradient descent step leaves the parameter vector unchanged ($\mathbf{x}_{k+1} = \mathbf{x}_k$), both for $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ and for a general differentiable objective $f$.

1. [code] **Central differences versus forward differences.** ● The one-sided *forward difference* $\frac{f(x+h) - f(x)}{h}$ analyzed in this section has truncation error $O(h)$. The symmetric *central difference*

   $$\frac{f(x+h) - f(x-h)}{2h}$$

   evaluates $f$ at $x+h$ and $x-h$ so that the $O(h)$ quadratic terms cancel out! Write a central-difference version of `quotient_error` and evaluate it across our logarithmic grid of $h$ values in `float64`:
    1. For the quadratic loss $L(w) = 3w^2 - 4w + 2$ at $w = 1$, plot the forward-difference and central-difference error curves on the same log--log axes. Why does the central-difference error drop straight to machine epsilon ($\sim 10^{-16}$) even at large $h = 10^{-1}$ on a quadratic function?
    1. Now repeat the comparison on a non-quadratic function, $f(x) = e^x$ at $x = 0$ (where $f'(0) = 1$). Compare the slope of the large-$h$ truncation branch of the central-difference curve against the slope-$1$ forward-difference branch, identify the optimal step size $h^*$ and minimum achievable error for each method, and explain both using the Taylor expansion $e^{\pm h} = 1 \pm h + \frac{h^2}{2} \pm \frac{h^3}{6} + \cdots$.

1. [code] **Gradient checking in `float64` versus `float32`.** ● Although finite differences are far too slow for training, central finite differences are the gold standard for unit-testing hand-derived or custom-kernel gradients.
    1. Write a function `central_gradient(f, x, h=1e-5)` that estimates $\nabla f(\mathbf{x})$ for $f: \mathbb{R}^n \to \mathbb{R}$ via central differences. Use it in `float64` to test the quadratic-form gradient identity $\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ on a random *non-symmetric* $5 \times 5$ matrix $\mathbf{A}$ at a random vector $\mathbf{x} \in \mathbb{R}^5$, and report the maximum relative error across the five coordinates.
    1. Now run the same check against the *buggy* formula $2\mathbf{A}\mathbf{x}$ (a common mistake made when forgetting that $\mathbf{A}$ is not symmetric). How large is the relative error now?
    1. Finally, repeat the check of the correct formula $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ in 32-bit `float32` using framework tensors across $h \in \{10^{-1}, 10^{-2}, \ldots, 10^{-6}\}$. Which $h$ achieves the smallest relative error in `float32`, why does it differ from the forward-difference optimum $h^* \approx 10^{-4}$, and what relative-error threshold cleanly separates a correct gradient implementation from a buggy one in `float32`?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/197)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17969)
:end_tab:
