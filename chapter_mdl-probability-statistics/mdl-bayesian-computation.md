# Bayesian Computation
:label:`sec_mdl-bayesian-computation`

Every model in the Basics part was trained to a single parameter vector, by
maximum likelihood or, with a penalty, by the maximum a posteriori (MAP)
estimate, the peak of the posterior. The previous section put an error bar on
such a vector. The predictions, however, still came from that one vector,
although many others fit the data almost as well. How much do they disagree,
and what should we predict when they do?

Take logistic regression with an intercept and a slope, fitted to twenty
labeled points whose inputs lie between 0 and 4. At an input of $-3$, far
below the data, the fitted model gives the label 1 a probability of about one
in a few hundred for a typical draw of the data. Vectors that fit the data
almost as well disagree there by orders of magnitude. Their average, with each
vector weighted by how plausible it is given the data, is several times
larger. At an input of 2, inside the data, the two answers agree to within
about $0.02$. For a rare outcome, such as a fraud flag or a diagnosis, a
factor of several in its probability can decide what we do.

The weighted average is the Bayesian prediction, and computing it is as old a
problem as Bayes' rule: the posterior divides by the evidence, an integral
over all parameter vectors that rarely has a closed form. Laplace, in 1774,
replaced the posterior of a proportion by the Gaussian that matches its peak.
Metropolis and his colleagues, in 1953, let a random walk visit states in
proportion to a distribution whose normalizer they did not know
:cite:`Metropolis.Rosenbluth.Rosenbluth.ea.1953`. In the 1990s the integral
became an optimization problem. One idea runs through all of these methods:
we can average over a distribution known only up to a constant, because each
method uses only ratios of it, and the constant cancels in a ratio.

We ask four questions in turn. Can we replace the posterior by the Gaussian
that matches its peak? Can we draw from a simpler distribution and correct
each draw? Can a random walk spend its time in each region in proportion to
the posterior? Can we optimize a simple distribution to be as close to the
posterior as possible? Each answer repairs a failure of the one before, and
none scales cleanly to the posterior of a neural network. All four compute
the **posterior predictive** distribution of a label $y_\star$ at a new input
$\mathbf x_\star$,

$$
p(y_\star\mid\mathbf x_\star,\mathcal D)
=\int p(y_\star\mid\mathbf x_\star,\boldsymbol\theta)\,
       p(\boldsymbol\theta\mid\mathcal D)\,d\boldsymbol\theta,
$$
:eqlabel:`eq_mdl-bayes-predictive`

which averages the predictions of all parameter vectors, each weighted by its
posterior density given the data $\mathcal D$.

```{.python .input #bayesian-computation-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
from mxnet import autograd, np as mxnp
```

```{.python .input #bayesian-computation-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #bayesian-computation-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input #bayesian-computation-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Posterior of Bayesian Logistic Regression
:label:`sec_mdl-bayes-target`

To compare approximations we need a posterior that has no closed form but
that we can still check. A prior is *conjugate* to a likelihood when the
posterior stays in the prior's family, as a Beta prior on a coin stays Beta
after any number of flips (:numref:`sec_mdl-exponential-family`). A Gaussian
prior is not conjugate to the logistic likelihood: the posterior can be
evaluated at any point but not integrated in closed form. With only two
parameters, a fine grid will still give us a reference against which to
check every approximation.

### The Model and Its Unnormalized Posterior

Logistic regression gives each binary label $y_i\in\{0,1\}$ a probability
that depends on the input $x_i$ through a line,

$$
P(y_i=1\mid x_i,\boldsymbol\theta)
=\operatorname{sigmoid}(\theta_0+\theta_1x_i)
=\operatorname{sigmoid}(\mathbf z_i^\top\boldsymbol\theta),
\qquad
\operatorname{sigmoid}(z)=\frac{1}{1+e^{-z}},
$$

where $\mathbf z_i=(1,x_i)^\top$. The probability crosses $\tfrac12$ where the
line crosses zero, at $x=-\theta_0/\theta_1$, and the slope $\theta_1$ sets
how sharply it rises there. We give the intercept and the slope independent
Gaussian priors $\theta_j\sim\mathcal N(0,\tau^2)$, where $\tau$ is the prior
standard deviation. Bayes' rule multiplies the likelihood of the $n$ labels by
the prior:

$$
p(\boldsymbol\theta\mid\mathcal D)
\propto
\left[\prod_{i=1}^n
 \operatorname{sigmoid}(\mathbf z_i^\top\boldsymbol\theta)^{y_i}
 \big(1-\operatorname{sigmoid}(\mathbf z_i^\top\boldsymbol\theta)\big)^{1-y_i}\right]
\exp\!\left(-\frac{\|\boldsymbol\theta\|^2}{2\tau^2}\right).
$$
:eqlabel:`eq_mdl-bayes-logistic-posterior`

The bracket is the likelihood and the exponential is the prior. The missing
constant is the **evidence** $p(\mathcal D)=\int p(\mathcal
D,\boldsymbol\theta)\,d\boldsymbol\theta$, an integral over all parameter
vectors that we cannot do in closed form. We write
$\widetilde p(\boldsymbol\theta)=p(\mathcal D,\boldsymbol\theta)$ for the
product on the right, the *unnormalized posterior*, so that the posterior is
$\widetilde p/p(\mathcal D)$.

The running data have $n=20$ inputs drawn uniformly from $[0,4]$ and labels
drawn from the line with $\boldsymbol\theta^\star=(-3,1.5)$, which crosses
$\tfrac12$ at $x=2$ and rises from $\operatorname{sigmoid}(-3)\approx0.05$ at
$x=0$ to $\operatorname{sigmoid}(3)\approx0.95$ at $x=4$. The prior has
$\tau^2=4$. Every run draws a fresh dataset, and twenty points leave the
posterior wide enough for each effect in this section to be visible. The cell
draws the data and writes $\log\widetilde p$, using `logaddexp(0, z)` for
$\log(1+e^z)$ so that large logits do not overflow.

```{.python .input #bayesian-computation-model}
rng = np.random.default_rng()
n, prior_var = 20, 4.0                        # prior_var is tau^2
theta_star = np.array([-3.0, 1.5])

def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))

x = rng.uniform(0, 4, n)
Z = np.column_stack([np.ones(n), x])          # rows z_i = (1, x_i)
y = rng.binomial(1, sigmoid(Z @ theta_star))

def log_joint(theta, Z=Z, y=y, prior_var=prior_var):
    """log p(D, theta) up to a constant; theta has final dimension 2."""
    logits = np.einsum('...d,nd->...n', theta, Z)
    log_lik = (y * logits - np.logaddexp(0.0, logits)).sum(axis=-1)
    return log_lik - 0.5 * (theta**2).sum(axis=-1) / prior_var

print(f'{n} points, {y.sum()} of them labeled 1')
```

### The Gradient and the Hessian

Several methods below also use the slope and the curvature of
$\log\widetilde p$, so we derive them once. Differentiating
$y_i\,\mathbf z_i^\top\boldsymbol\theta-\log(1+e^{\mathbf z_i^\top\boldsymbol\theta})$
gives $(y_i-p_i)\,\mathbf z_i$ with $p_i=\operatorname{sigmoid}(\mathbf
z_i^\top\boldsymbol\theta)$, and the prior contributes
$-\boldsymbol\theta/\tau^2$. Collecting the rows $\mathbf z_i^\top$ into the
$n\times2$ matrix $\mathbf Z$, the gradient is
$\nabla\log\widetilde p(\boldsymbol\theta)=\mathbf Z^\top(\mathbf y-\mathbf
p)-\boldsymbol\theta/\tau^2$: each point pushes $\boldsymbol\theta$ along
$\mathbf z_i$ by its residual, and the prior pulls toward zero.
Differentiating once more gives the negative Hessian

$$
\mathbf H(\boldsymbol\theta)
=-\nabla^2\log\widetilde p(\boldsymbol\theta)
=\mathbf Z^\top\operatorname{diag}\!\big(p_i(1-p_i)\big)\mathbf Z
+\tau^{-2}\mathbf I.
$$
:eqlabel:`eq_mdl-bayes-hessian`

The data term is a Gram matrix of the inputs, each point weighted by
$p_i(1-p_i)\le\tfrac14$, so the points the model is unsure about count most.
The prior adds $\tau^{-2}$ to every eigenvalue, so $\mathbf H$ is positive
definite for every $\boldsymbol\theta$. The log posterior is therefore
strictly concave, and it has one maximum and no other local maxima. The
maximum is finite even when the data are linearly separable (about one run in
thirty), because the log-likelihood is never positive while the prior term
falls without bound in every direction.

```{.python .input #mdl-bayesian-computation-the-gradient-and-the-hessian-1}
def grad_log_joint(theta, Z=Z, y=y, prior_var=prior_var):
    """Gradient of log p(D, theta); one row per row of theta."""
    theta = np.atleast_2d(theta)
    return (y - sigmoid(theta @ Z.T)) @ Z - theta / prior_var

def neg_hessian(theta, Z=Z, y=y, prior_var=prior_var):
    """H(theta) = Z^T diag(p (1 - p)) Z + I / tau^2."""
    p = sigmoid(Z @ theta)
    return Z.T @ (Z * (p * (1 - p))[:, None]) + np.eye(2) / prior_var
```

For two parameters these derivatives are easy by hand. For a model with many
parameters we would write only $\log\widetilde p$ and let automatic
differentiation produce its gradient and Hessian. The next cell checks that
this gives the same numbers at $\boldsymbol\theta^\star$.

```{.python .input #mdl-bayesian-computation-the-gradient-and-the-hessian-2}
#@tab pytorch
Z_t, y_t = torch.tensor(Z), torch.tensor(y, dtype=torch.float64)

def log_joint_torch(theta):
    logits = Z_t @ theta
    return ((y_t * logits - torch.nn.functional.softplus(logits)).sum()
            - (theta**2).sum() / (2 * prior_var))

theta0 = torch.tensor(theta_star)
grad = torch.func.grad(log_joint_torch)(theta0).numpy()
hess = torch.func.hessian(log_joint_torch)(theta0).numpy()
print('gradient error:', np.abs(grad - grad_log_joint(theta_star)[0]).max())
print('Hessian error :', np.abs(-hess - neg_hessian(theta_star)).max())
```

```{.python .input #mdl-bayesian-computation-the-gradient-and-the-hessian-2}
#@tab jax
def log_joint_jax(theta):
    logits = jnp.asarray(Z) @ theta
    return ((y * logits - jnp.logaddexp(0.0, logits)).sum()
            - (theta**2).sum() / (2 * prior_var))

theta0 = jnp.asarray(theta_star)
grad = np.asarray(jax.grad(log_joint_jax)(theta0))
hess = np.asarray(jax.hessian(log_joint_jax)(theta0))
g_ref, H_ref = grad_log_joint(theta_star)[0], neg_hessian(theta_star)
# JAX computes in single precision: compare relative errors
print('gradient error:', np.abs(grad - g_ref).max() / np.abs(g_ref).max())
print('Hessian error :', np.abs(-hess - H_ref).max() / np.abs(H_ref).max())
```

```{.python .input #mdl-bayesian-computation-the-gradient-and-the-hessian-2}
#@tab tensorflow
Z_tf, y_tf = tf.constant(Z), tf.constant(y, dtype=tf.float64)
theta0 = tf.Variable(theta_star)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        logits = tf.linalg.matvec(Z_tf, theta0)
        log_p = (tf.reduce_sum(y_tf * logits - tf.math.softplus(logits))
                 - tf.reduce_sum(theta0**2) / (2 * prior_var))
    grad = inner.gradient(log_p, theta0)
hess = outer.jacobian(grad, theta0)
print('gradient error:',
      np.abs(grad.numpy() - grad_log_joint(theta_star)[0]).max())
print('Hessian error :', np.abs(-hess.numpy() - neg_hessian(theta_star)).max())
```

```{.python .input #mdl-bayesian-computation-the-gradient-and-the-hessian-2}
#@tab mxnet
Z_mx = mxnp.array(Z, dtype='float64')
y_mx = mxnp.array(y, dtype='float64')
theta0 = mxnp.array(theta_star, dtype='float64')
theta0.attach_grad()
with autograd.record():
    logits = mxnp.dot(Z_mx, theta0)
    log_p = ((y_mx * logits - mxnp.log1p(mxnp.exp(logits))).sum()
             - (theta0**2).sum() / (2 * prior_var))
log_p.backward()
# MXNet's autograd supports second derivatives for few operators, so this
# tab checks the gradient only; |logits| stays small enough for log1p(exp)
print('gradient error:',
      np.abs(theta0.grad.asnumpy() - grad_log_joint(theta_star)[0]).max())
```

The differences are at the level of rounding error. Every method in this
section touches the posterior only through $\widetilde p$: its values, its
gradient, or its Hessian.

### The Posterior Mode and the Laplace Approximation
:label:`subsec_mdl-bayes-laplace`

The cheapest approximation reuses an optimization we already know how to run.
The peak of the posterior is the MAP estimate, the penalized fit of the
previous section, and Newton's method
$\boldsymbol\theta\leftarrow\boldsymbol\theta+\mathbf
H(\boldsymbol\theta)^{-1}\nabla\log\widetilde p(\boldsymbol\theta)$ reaches it
in a few steps. One more Hessian turns the peak into a Gaussian.

We look for a Gaussian centered at the mode. Expanding $\log\widetilde p$ to
second order around $\boldsymbol\theta_{\mathrm{MAP}}$, with $\mathbf
H=\mathbf H(\boldsymbol\theta_{\mathrm{MAP}})$, gives

$$
\log\widetilde p(\boldsymbol\theta)
\approx\log\widetilde p(\boldsymbol\theta_{\mathrm{MAP}})
-\tfrac12(\boldsymbol\theta-\boldsymbol\theta_{\mathrm{MAP}})^\top
\mathbf H(\boldsymbol\theta-\boldsymbol\theta_{\mathrm{MAP}}).
$$
:eqlabel:`eq_mdl-bayes-laplace-taylor`

The linear term is missing because the gradient vanishes at the mode. The
right side is, up to a constant, the log density of a Gaussian with mean
$\boldsymbol\theta_{\mathrm{MAP}}$ and covariance $\mathbf H^{-1}$, so
exponentiating gives the **Laplace approximation**

$$
p(\boldsymbol\theta\mid\mathcal D)
\approx\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},\mathbf H^{-1}).
$$
:eqlabel:`eq_mdl-bayes-laplace`

It needs $\log\widetilde p$ to be twice differentiable at a mode in the
interior of the parameter space, and $\mathbf H$ to be positive definite
there; both hold here. A sharp peak, with large curvature, gives a small
variance. The Gaussian matches the posterior's location, slope and curvature
at the mode, and nothing else.

The smallest case is one-dimensional. Take the posterior
$\mathrm{Gamma}(4,1)$, with density proportional to $\theta^3e^{-\theta}$ on
$\theta>0$. Its log, $3\log\theta-\theta$, has derivative $3/\theta-1$, which
vanishes at the mode $\theta=3$, and curvature $3/\theta^2=\tfrac13$ there. The
Laplace approximation is $\mathcal N(3,3)$, with standard deviation $1.73$.
The true posterior has mean 4 and variance 4, so the Gaussian sits too far
left, is too narrow, and puts $\Phi(-\sqrt3)\approx4.2\%$ of its mass below
$\theta=0$, where the posterior is zero
(:numref:`fig_mdl-prob-bayes-laplace`).

A second small case is a coin that shows 9 heads and 4 tails. Under a uniform
prior its posterior $\mathrm{Beta}(10,5)$ has density proportional to
$\theta^9(1-\theta)^4$. The same computation gives the mode $9/13=0.692$ and
the curvature $9/\theta^2+4/(1-\theta)^2=61$ there. The Laplace approximation
is therefore $\mathcal N(0.692,0.128^2)$, against the exact mean $0.667$ and
standard deviation $0.118$ (Exercise
:exref:`ex_mdl-bayesian-computation-laplace-parameterization`).

![The Laplace approximation for a skewed one-dimensional posterior, the Gamma(4,1) density with mode 3. Left: the quadratic fit to the log density matches its value, slope and curvature at the mode and nothing else. Right: after exponentiating, the Laplace Gaussian $\mathcal{N}(3,3)$ has the right peak but misses the skew. It puts 4.2% of its mass (shaded orange) below the support edge $\theta=0$, where the posterior is zero, and drops tail mass on the right; the true mean is 4.](../img/mdl-prob-bayes-laplace.svg)
:label:`fig_mdl-prob-bayes-laplace`

The evidence never enters: $\log p(\mathcal D)$ is an additive constant in
$\log p(\boldsymbol\theta\mid\mathcal D)$, so it drops out of the curvature.
Integrating the exponential of the right side of
:eqref:`eq_mdl-bayes-laplace-taylor` even estimates that constant,
$p(\mathcal D)\approx\widetilde p(\boldsymbol\theta_{\mathrm{MAP}})(2\pi)^{d/2}\det(\mathbf
H)^{-1/2}$ for $d$ parameters (Exercise
:exref:`ex_mdl-bayesian-computation-evidence`).

The approximation is local. It sees no skew, no heavy tail, no boundary and
no second mode, and its center is the mode, not the mean. For a network with
$d$ parameters $\mathbf H$ has $d^2$ entries, so in practice one keeps only
its diagonal, or only the Hessian of the last layer.

The cell runs ten Newton steps from zero and inverts the negative Hessian at
the mode.

```{.python .input #bayesian-computation-map-predictive}
theta_map = np.zeros(2)
for _ in range(10):                                   # Newton's method
    theta_map += np.linalg.solve(neg_hessian(theta_map),
                                 grad_log_joint(theta_map)[0])
laplace_cov = np.linalg.inv(neg_hessian(theta_map))
laplace_sd = np.sqrt(np.diag(laplace_cov))
laplace_rho = laplace_cov[0, 1] / (laplace_sd[0] * laplace_sd[1])
print('MAP                :', theta_map.round(3))
print('Laplace sd         :', laplace_sd.round(3))
print('Laplace correlation:', round(laplace_rho, 3))
```

The MAP is finite in every run, separable data included. The Laplace standard
deviations are about 1 for the intercept and 0.5 for the slope, and the
correlation is strongly negative, about $-0.85$: a steeper line must start
lower to pass through the same points.

### A Grid Reference

To judge the Laplace approximation we need the exact answer. With two
parameters we can evaluate $\log\widetilde p$ on an $m\times m$ grid,
exponentiate after subtracting the maximum, and normalize the sum. With $m$
points per coordinate a $d$-dimensional grid needs $m^d$ evaluations, so this
works only as a check in low dimension. The methods after it place their
evaluations where the posterior has mass instead. Even the grid must know
where to look. We center it at the MAP and let it extend seven Laplace
standard deviations in each direction, which leaves a negligible part of the
posterior outside.

```{.python .input #bayesian-computation-grid}
k, m = 7, 241                    # half-width in Laplace sds, points per axis
grid_axes = [np.linspace(c - k * s, c + k * s, m)
             for c, s in zip(theta_map, laplace_sd)]
B0, B1 = np.meshgrid(*grid_axes, indexing='ij')
grid_theta = np.stack([B0.ravel(), B1.ravel()], axis=1)
grid_logp = log_joint(grid_theta)
grid_weight = np.exp(grid_logp - grid_logp.max())
grid_weight /= grid_weight.sum()
post_mean = grid_weight @ grid_theta
centered = grid_theta - post_mean
post_cov = (centered * grid_weight[:, None]).T @ centered
post_sd = np.sqrt(np.diag(post_cov))
post_rho = post_cov[0, 1] / (post_sd[0] * post_sd[1])
print('grid mean       :', post_mean.round(3), '  MAP:', theta_map.round(3))
print('grid sd         :', post_sd.round(3), '  Laplace:', laplace_sd.round(3))
print('grid correlation:', round(post_rho, 3))
print('(mean - MAP)/sd :', ((post_mean - theta_map) / post_sd).round(2))
```

In nearly every run the slope's posterior mean lies above its MAP and the
intercept's mean lies below its MAP; the order flips in about one run in a
thousand. For the slope the gap is typically about
a quarter of a posterior standard deviation. The Laplace standard deviations
are a few percent too small, and the grid's correlation is close to the
strongly negative value that the Laplace approximation reported. So even the
center of the Laplace Gaussian is in the wrong place: the posterior is skewed,
and its mean is not its mode.

### The Plug-in and the Posterior Predictive

We can now answer the question of the opening. The *plug-in* prediction
$\operatorname{sigmoid}(\mathbf z_\star^\top\boldsymbol\theta_{\mathrm{MAP}})$
uses the mode alone. The posterior predictive
:eqref:`eq_mdl-bayes-predictive` averages
$\operatorname{sigmoid}(\mathbf z_\star^\top\boldsymbol\theta)$ over the grid,
with the grid weights. The cell computes both at $x_\star\in\{-3,2,7\}$, and
also the range that holds the middle 90% of the posterior's predictions.

```{.python .input #mdl-bayesian-computation-the-plug-in-and-the-posterior-predictive-1}
x_new = np.array([-3.0, 2.0, 7.0])
Z_new = np.column_stack([np.ones(len(x_new)), x_new])
grid_probs = sigmoid(grid_theta @ Z_new.T)     # each grid vector's answer
plug_in = sigmoid(Z_new @ theta_map)
predictive = grid_weight @ grid_probs
for j, x0 in enumerate(x_new):
    order = np.argsort(grid_probs[:, j])
    cdf = np.cumsum(grid_weight[order])
    lo, hi = grid_probs[order[np.searchsorted(cdf, [0.05, 0.95])], j]
    print(f'x={x0:+.0f}: plug-in {plug_in[j]:.4f}, predictive '
          f'{predictive[j]:.4f}, middle 90%: [{lo:.2g}, {hi:.4f}]')
```

At $x_\star=2$ the two predictions agree to within about $0.02$, and the
posterior's predictions there fill a moderate interval. At $x_\star=-3$ the
middle 90% of the predictions span orders of magnitude, and the posterior
predictive is typically three to ten times the plug-in. At
$x_\star=7$ the same holds for the probability of a 0 label. Probabilities
this close to 0 and 1 are easiest to compare as log-odds
$\log\frac{P}{1-P}$, so the plot shows both predictions on that scale over a
wider range than the data, with the training inputs marked along the bottom.

```{.python .input #mdl-bayesian-computation-the-plug-in-and-the-posterior-predictive-2}
def log_odds(p):
    return np.log(p) - np.log1p(-p)

xs = np.linspace(-4, 8, 121)
Z_xs = np.column_stack([np.ones_like(xs), xs])
plug_xs = Z_xs @ theta_map                       # log-odds of the plug-in
pred_xs = log_odds(grid_weight @ sigmoid(grid_theta @ Z_xs.T))
rug = np.full(n, min(plug_xs.min(), pred_xs.min()) - 1)
d2l.plot([xs, xs, x], [plug_xs, pred_xs, rug],
         xlabel='x', ylabel='log-odds of y = 1', fmts=('-', 'm--', 'k|'),
         legend=['plug-in (MAP)', 'posterior predictive', 'training inputs'],
         figsize=(5, 3))
```

On the log-odds scale the plug-in is the straight line
$\theta_0+\theta_1x$. Inside the data the posterior predictive lies on top of
it; outside $[0,4]$ it bends away from the line toward 0. Two effects produce
the gap, and they point in opposite directions. First, the mode is not the
mean. The posterior mean has a steeper slope than the MAP, and a steeper line
moves the prediction away from $\tfrac12$ far out on either side. Second,
averaging $\operatorname{sigmoid}$ over a spread of logits moves the
prediction toward $\tfrac12$, because the sigmoid flattens at its ends. On the
far side of the data the plausible vectors with a shallow slope give
probabilities that are large compared with the typical one, and they dominate
the average.

The spread of the logit $\mathbf z_\star^\top\boldsymbol\theta$ has variance
$\textrm{Var}(\theta_0)+2x_\star\textrm{Cov}(\theta_0,\theta_1)+x_\star^2\textrm{Var}(\theta_1)$,
which grows quadratically with the distance from the data, while the shift
of the mean grows only linearly. So the second effect wins far out, and
inside the data both effects are small (Exercise
:exref:`ex_mdl-bayesian-computation-moderated-predictions`).

The larger number is not necessarily closer to the truth. The line that
generated these data gives $\operatorname{sigmoid}(-3-1.5\cdot3)=
\operatorname{sigmoid}(-7.5)\approx0.0006$ at $x_\star=-3$, below both numbers
in most runs: twenty points cannot pin the slope down, and the prior pulls it
toward zero. That value does lie inside the printed 90% range in about nine
runs out of ten. The posterior predictive is the probability that the model
and the prior assign to the label once the disagreement among plausible
vectors is counted. It is exact whenever the parameters are themselves drawn
from the prior. Every number in this subsection came from the grid. The rest
of the section asks how to compute such averages when no grid is possible.

## Rejection and Importance Sampling
:label:`sec_mdl-bayes-importance`

The Laplace Gaussian is centered at the mode and has no skew. Draws from the
exact posterior would get every moment right at once, because their average
estimates any posterior expectation. The two methods of this section turn draws from a
simple distribution $q$, the *proposal*, into draws or weighted draws from
the posterior, and they use only the ratio $\widetilde p/q$.

### Rejection Sampling
:label:`subsec_mdl-bayes-rejection`

The simplest exact sampler keeps some draws and discards the rest. Choose a
proposal $q$ that we can sample and evaluate, and a constant $M$ with
$Mq(\boldsymbol\theta)\ge\widetilde p(\boldsymbol\theta)$ for every
$\boldsymbol\theta$, so that $Mq$ is an envelope over the unnormalized
posterior. Draw $\boldsymbol\theta\sim q$ and keep it with probability

$$
a(\boldsymbol\theta)=\frac{\widetilde p(\boldsymbol\theta)}{M\,q(\boldsymbol\theta)}.
$$
:eqlabel:`eq_mdl-bayes-rejection`

The acceptance probability is the height of the target as a fraction of the
height of the envelope at the drawn point.

**Proposition.** The kept draws have density $p(\boldsymbol\theta\mid\mathcal
D)$, and a draw is kept with probability $p(\mathcal D)/M$.

**Proof.** A draw lands in a small region $d\boldsymbol\theta$ around
$\boldsymbol\theta$ with probability $q(\boldsymbol\theta)\,d\boldsymbol\theta$
and is kept with probability $a(\boldsymbol\theta)$, so it lands there and is
kept with probability $\widetilde p(\boldsymbol\theta)\,d\boldsymbol\theta/M$.
Integrating over $\boldsymbol\theta$, a draw is kept with probability
$\int\widetilde p/M=p(\mathcal D)/M$. Dividing the first probability by the
second, a kept draw has density $\widetilde p(\boldsymbol\theta)/p(\mathcal
D)=p(\boldsymbol\theta\mid\mathcal D)$. $\blacksquare$

The evidence appears only in the acceptance rate, never in the decision to
keep a draw. With the prior as the proposal, the ratio $\widetilde p/q$ is the
likelihood and $M$ is its maximum, and the rule reads: draw parameters from the
prior, and keep each with probability likelihood over best likelihood.

On the coin with 9 heads and 4 tails under a uniform prior, the likelihood
$\theta^9(1-\theta)^4$ is largest at $\theta=9/13$, where it equals
$M=(9/13)^9(4/13)^4\approx3.3\times10^{-4}$. The evidence is
$p(\mathcal D)=\int_0^1\theta^9(1-\theta)^4\,d\theta=9!\,4!/14!\approx1.0\times10^{-4}$,
so the acceptance rate is $0.305$. With ten times the data, 90 heads and 40
tails, it is $0.10$, and with a hundred times it is $0.03$. The rate falls
like $1/\sqrt n$ because the posterior's width does, while the prior's stays
fixed. In $d$ dimensions the posterior's volume shrinks like $n^{-d/2}$, and
so does the acceptance rate. The cell checks the coin's numbers.

```{.python .input #mdl-bayesian-computation-rejection-sampling}
N = 100_000
theta_coin = rng.random(N)                    # draws from the uniform prior
lik = theta_coin**9 * (1 - theta_coin)**4     # likelihood of 9 heads, 4 tails
M = (9 / 13)**9 * (4 / 13)**4                 # its maximum, at theta = 9/13
kept = theta_coin[rng.random(N) < lik / M]
print(f'acceptance {len(kept) / N:.3f}, mean {kept.mean():.3f}, '
      f'sd {kept.std():.3f}')
```

The acceptance rate is close to $0.305$, and the kept draws have the
$\mathrm{Beta}(10,5)$ mean $2/3$ and standard deviation $0.118$. Two limits
remain: the bound $M$ must be known, which it is not for the posterior of a
network, and for a fixed amount of data per parameter the acceptance rate
falls at least exponentially with the dimension.

### Importance Weights

Rejection sampling discards most of its draws. We can keep every draw instead
and give it the weight $w=\widetilde p/q$ that rejection sampling used to flip
a coin. To see that this works, write a posterior expectation as a ratio of
two integrals and multiply and divide by $q$ inside both:

$$
\mathbb{E}_{p(\cdot\mid\mathcal D)}[h(\boldsymbol\theta)]
=\frac{\int h\,\widetilde p\,d\boldsymbol\theta}{\int\widetilde p\,d\boldsymbol\theta}
=\frac{\int h\,\frac{\widetilde p}{q}\,q\,d\boldsymbol\theta}
      {\int\frac{\widetilde p}{q}\,q\,d\boldsymbol\theta}
=\frac{\mathbb{E}_q[w(\boldsymbol\theta)h(\boldsymbol\theta)]}
     {\mathbb{E}_q[w(\boldsymbol\theta)]},
\qquad
w(\boldsymbol\theta)=\frac{\widetilde p(\boldsymbol\theta)}
                           {q(\boldsymbol\theta)}.
$$
:eqlabel:`eq_mdl-bayes-importance`

The denominator $\mathbb{E}_q[w]$ equals the unknown evidence
$p(\mathcal D)=\int\widetilde p$, and we never need its value, because the
same constant scales the numerator. Replacing both
expectations by averages over draws $\boldsymbol\theta_1,\ldots,\boldsymbol\theta_N\sim q$
gives **self-normalized importance sampling**,
$\sum_sw_sh(\boldsymbol\theta_s)/\sum_sw_s$. By the law of large numbers it
converges to the posterior expectation; at finite $N$ it is slightly biased,
because it is a ratio of two random averages. Products of many likelihood
terms underflow, so the weights are computed in log space with the largest
log-weight subtracted, as in the log-sum-exp trick. With the prior as the
proposal the weight is the likelihood, and the method reads: draw parameters
from the prior and weight each by how well it explains the data.

### The Effective Sample Size

A weighted sample in which one draw carries all the weight is worth one draw,
whatever $N$ is. The **effective sample size** (ESS) of the normalized weights
$\bar w_s=w_s/\sum_rw_r$,

$$
N_{\mathrm{eff}}=\frac{1}{\sum_{s=1}^N\bar w_s^2},
$$
:eqlabel:`eq_mdl-bayes-is-ess`

equals $N$ when all weights are equal and $1$ when a single draw carries all
of them. It estimates the number of independent posterior draws that the
weighted sample is worth.

**Proposition.** As $N\to\infty$,

$$
\frac{N_{\mathrm{eff}}}{N}\to\frac{(\mathbb{E}_q[w])^2}{\mathbb{E}_q[w^2]}
=\frac{1}{1+\chi^2(p\,\|\,q)},
\qquad
\chi^2(p\,\|\,q)=\int\frac{p^2}{q}\,d\boldsymbol\theta-1,
$$
:eqlabel:`eq_mdl-bayes-ess-limit`

where $p$ is the normalized posterior. With the same proposal, this limit is
at least the acceptance rate $p(\mathcal D)/M$ of rejection sampling.

**Proof.** Multiplying numerator and denominator of the definition by $1/N^2$
gives $N_{\mathrm{eff}}/N=(\frac1N\sum_sw_s)^2/(\frac1N\sum_sw_s^2)$, and the
law of large numbers applies to both averages. Writing $w=p(\mathcal D)\,p/q$
gives $\mathbb{E}_q[w^2]=p(\mathcal D)^2\int p^2/q$ and
$\mathbb{E}_q[w]=p(\mathcal D)$, hence the $\chi^2$ form. Finally, $w\le M$
gives $\mathbb{E}_q[w^2]\le M\,\mathbb{E}_q[w]$, so the limit is at least
$\mathbb{E}_q[w]/M=p(\mathcal D)/M$. $\blacksquare$

The ESS is therefore a sample estimate of how far $q$ is from $p$. For large
$N$, importance sampling keeps at least as many effective draws as rejection
sampling would accept, and it needs no bound $M$. On the coin with the uniform prior as the
proposal, $\mathbb{E}_q[w^2]=\int_0^1\theta^{18}(1-\theta)^8\,d\theta$, and the
limit is $0.42$, against the acceptance rate $0.305$. The cell computes both on
the draws of the rejection sampler.

```{.python .input #mdl-bayesian-computation-the-effective-sample-size}
w = lik / lik.sum()                           # normalized weights
post_coin = w @ theta_coin
print(f'ESS fraction {1 / (w @ w) / N:.3f}, mean {post_coin:.3f}, '
      f'sd {np.sqrt(w @ (theta_coin - post_coin)**2):.3f}')
```

The ESS fraction is close to $0.42$, and the weighted mean and standard
deviation match the posterior. The limit also shows how badly a proposal can
fail. For a Gaussian target with standard deviation $s_p$ and a Gaussian
proposal at the same center with standard deviation $s_q=rs_p$, the limit is
$\sqrt{2r^2-1}/r^2$ (Exercise :exref:`ex_mdl-bayesian-computation-weight-ess`).
It is $0.83$ at $r=0.8$, $1$ at $r=1$, $0.66$ at $r=2$ and $0.46$ at $r=3$, and
it is zero for $r\le1/\sqrt2$, where $\mathbb{E}_q[w^2]=\infty$. A proposal
that is too narrow fails far more abruptly than one that is too wide.

Two conditions make the identity :eqref:`eq_mdl-bayes-importance` hold:
$q(\boldsymbol\theta)>0$ wherever $\widetilde p(\boldsymbol\theta)>0$, and
$\mathbb{E}_q|wh|<\infty$. A central limit theorem for the estimator, and
with it error bars of order $1/\sqrt N$, holds when $\mathbb{E}_q[w^2]$ and
$\mathbb{E}_q[w^2h^2]$ are also finite. Even then the ESS can mislead, as
:numref:`fig_mdl-prob-bayes-importance` shows.

![Importance sampling of a two-mode target with two proposals; stems show the normalized weights of 24 draws. Left: a narrow proposal never reaches the right-hand mode. Its weights are nearly equal (ESS 23 of 24), and its estimate is wrong. Right: a broader proposal samples both modes. Its weights are less equal (ESS 17 of 24), yet its estimate is far better. The ESS measures how unequal the weights are, not whether the proposal found the mass.](../img/mdl-prob-bayes-importance.svg)
:label:`fig_mdl-prob-bayes-importance`

### Proposals for the Logistic Posterior

On the logistic posterior we compare two Gaussian proposals: the prior, with
20,000 draws, and the Laplace approximation, with 5,000. The cell prints the
ESS as a fraction of the draws, the largest normalized weight, and the error
of the estimated posterior mean in units of posterior standard deviations,
measured against the grid.

```{.python .input #bayesian-computation-importance}
def gaussian_logpdf(samples, mean, cov):
    diff = samples - mean
    _, logdet = np.linalg.slogdet(cov)
    maha = np.einsum('ni,ij,nj->n', diff, np.linalg.inv(cov), diff)
    return -0.5 * (len(mean) * np.log(2 * np.pi) + logdet + maha)

def importance(log_target, mean, cov, N):
    """Self-normalized importance sampling from a Gaussian proposal."""
    samples = rng.multivariate_normal(mean, cov, N)
    log_w = log_target(samples) - gaussian_logpdf(samples, mean, cov)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    return w @ samples, 1 / (w @ w), w.max()

for name, mean, cov, N in [('prior', np.zeros(2), prior_var * np.eye(2), 20000),
                           ('Laplace', theta_map, laplace_cov, 5000)]:
    estimate, ess, w_max = importance(log_joint, mean, cov, N)
    error = np.abs(estimate - post_mean) / post_sd
    print(f'{name:8s}: ESS {ess / N:.1%} of {N}, largest weight '
          f'{w_max:.4f}, error in sds {error.round(3)}')
```

With the prior as the proposal, the ESS is between about 1% and 10% of the
draws, because most draws land where the likelihood is negligible. The
Laplace proposal, aimed at the posterior, keeps most of its draws as
effective, typically between about 55% and 95%. Both estimates usually agree
with the grid mean to within a few hundredths of a posterior standard
deviation, and rarely miss by more than about $0.15$.

The Laplace proposal has a defect that the printed ESS does not show. Far
out, the posterior's tails are those of the prior: along any ray
$\boldsymbol\theta=t\mathbf v$ the log-likelihood falls at most linearly in
$t$ (Exercise :exref:`ex_mdl-bayesian-computation-posterior-tails`), while the
log prior falls like $t^2/(2\tau^2)$. The integral
$\mathbb{E}_q[w^2]=\int\widetilde p^2/q$ for a Gaussian proposal
$\mathcal N(\mathbf m,\mathbf S)$ is then infinite as soon as one eigenvalue
of $\mathbf S$ lies below $\tau^2/2=2$. The smallest eigenvalue of the Laplace
covariance is far below 2 in every run, a few hundredths in a typical one, so
its weights have infinite variance.

The region where the weights blow up lies many Laplace standard deviations
out, and most runs of 5,000 draws never visit it. In a few runs out of a
hundred a single far draw takes a large share of the weight, and the ESS
drops far below its usual value. A single run cannot reveal the defect;
repeated runs, or the analysis of the tails, can. A Student-$t$ proposal,
whose tails are heavier than any Gaussian's, removes the defect (Exercise
:exref:`ex_mdl-bayesian-computation-posterior-tails`). So does the defensive
mixture $q=0.9\,\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},\mathbf
H^{-1})+0.1\,\text{prior}$: since $q$ is at least a tenth of the prior, every
weight is at most ten times the likelihood, which is at most one.

The evidence cancelled in the ratio of the two averages. What remains is the
need for one global proposal that covers the whole posterior, and in high
dimension no simple distribution does (Exercise
:exref:`ex_mdl-bayesian-computation-higher-dimension`).

## Markov Chain Monte Carlo
:label:`sec_mdl-bayes-mcmc`

Instead of one proposal for the whole posterior, we can make local moves and
let a random walk spend time in each region in proportion to its posterior
mass. The walk is a Markov chain. Its next state depends on the current
state only, through a *transition rule* $T(\boldsymbol\theta'\mid\boldsymbol\theta)$,
the probability of moving from $\boldsymbol\theta$ to $\boldsymbol\theta'$. A
distribution $\pi$ is *stationary* for the chain if

$$
\sum_{\boldsymbol\theta}\pi(\boldsymbol\theta)\,
T(\boldsymbol\theta'\mid\boldsymbol\theta)=\pi(\boldsymbol\theta')
\quad\text{for every }\boldsymbol\theta',
$$
:eqlabel:`eq_mdl-bayes-stationary`

with an integral in place of the sum for continuous parameters. If the states
are distributed as $\pi$, they still are after one step. We want a chain
whose stationary distribution is the posterior. The first states depend on
where the chain started, so the first part of the run, the *warm-up*, is
discarded before averaging.

### The Metropolis Rule

The Metropolis rule builds such a chain from any symmetric proposal
$g(\boldsymbol\theta'\mid\boldsymbol\theta)=g(\boldsymbol\theta\mid\boldsymbol\theta')$,
such as a Gaussian step $\boldsymbol\theta'=\boldsymbol\theta+\mathbf
L\boldsymbol\epsilon$. From the current state $\boldsymbol\theta$, propose
$\boldsymbol\theta'\sim g(\cdot\mid\boldsymbol\theta)$. If
$\widetilde p(\boldsymbol\theta')\ge\widetilde p(\boldsymbol\theta)$, move to
$\boldsymbol\theta'$. Otherwise move with probability
$\widetilde p(\boldsymbol\theta')/\widetilde p(\boldsymbol\theta)$, and stay
at $\boldsymbol\theta$ with the remaining probability. A rejection repeats the current state, and the repeat counts
as a state of the chain (:numref:`fig_mdl-prob-bayes-metropolis`).

![A Metropolis random walk on posterior contours. From the starting square, each step proposes a local move; uphill moves are always accepted (path), downhill moves only sometimes, and a rejected proposal (crosses) leaves the chain in place for another step. In the long run the time the walk spends in a region is proportional to its posterior mass.](../img/mdl-prob-bayes-metropolis.svg)
:label:`fig_mdl-prob-bayes-metropolis`

Hastings :cite:`Hastings.1970` extended the rule to asymmetric proposals. The
only change is a factor,
$g(\boldsymbol\theta\mid\boldsymbol\theta')/g(\boldsymbol\theta'\mid\boldsymbol\theta)$,
that corrects for a proposal that favors one direction. A proposal is then
accepted with probability

$$
\alpha(\boldsymbol\theta,\boldsymbol\theta')
=\min\!\left(1,
\frac{\widetilde p(\boldsymbol\theta')\,
      g(\boldsymbol\theta\mid\boldsymbol\theta')}
     {\widetilde p(\boldsymbol\theta)\,
      g(\boldsymbol\theta'\mid\boldsymbol\theta)}\right).
$$
:eqlabel:`eq_mdl-bayes-mh`

A move toward a state the proposal rarely offers is accepted more readily, and
for a symmetric proposal the factor is 1. The evidence cancels in the ratio of
the two values of $\widetilde p$.

The smallest case has two states, with target $\pi=(\tfrac14,\tfrac34)$, and a
proposal that always names the other state. From state 1 the ratio is 3, so
the chain always moves. From state 2 the ratio is $\tfrac13$, so it moves with
probability $\tfrac13$ and stays otherwise. The transition matrix, with rows
for the current state, and the check that $\pi$ is stationary are

$$
\mathbf T=\begin{pmatrix}0&1\\\tfrac13&\tfrac23\end{pmatrix},
\qquad
\left(\tfrac14,\,\tfrac34\right)\mathbf T
=\left(\tfrac34\cdot\tfrac13,\;\tfrac14+\tfrac34\cdot\tfrac23\right)
=\left(\tfrac14,\,\tfrac34\right).
$$

The chain spends a quarter of its time in state 1 and three quarters in
state 2, as it should. If we counted only accepted states, the sequence would
alternate 2, 1, 2, 1 and give $(\tfrac12,\tfrac12)$. The repeated 2s after
rejections give state 2 its correct weight
(:numref:`fig_mdl-prob-bayes-two-state`).

![The two-state Metropolis chain with target $\pi=(1/4,3/4)$. The flow from state 1 to state 2 is $\frac14\cdot1$ and the flow back is $\frac34\cdot\frac13$; both are $1/4$, so the occupation $(1/4,3/4)$ does not change from step to step. The self-loop carries the rejected moves from state 2; counting only accepted states would give $(1/2,1/2)$.](../img/mdl-prob-bayes-two-state.svg)
:label:`fig_mdl-prob-bayes-two-state`

### Detailed Balance and Stationarity

The figure shows why the chain keeps its distribution: the probability that
flows from state 1 to state 2 in one step equals the probability that flows
back, so no state gains or loses mass. This balance of flows holds for every
pair of states, and it is the reason Metropolis works.

**Proposition.** If

$$
\pi(\boldsymbol\theta)\,T(\boldsymbol\theta'\mid\boldsymbol\theta)
=\pi(\boldsymbol\theta')\,T(\boldsymbol\theta\mid\boldsymbol\theta')
\quad\text{for all }\boldsymbol\theta,\boldsymbol\theta',
$$
:eqlabel:`eq_mdl-bayes-detailed-balance`

a condition called *detailed balance*, then $\pi$ is stationary. The
Metropolis–Hastings chain satisfies detailed balance with
$\pi\propto\widetilde p$.

**Proof.** Sum both sides over $\boldsymbol\theta$. The left side becomes the
left side of :eqref:`eq_mdl-bayes-stationary`, and the right side is
$\pi(\boldsymbol\theta')\sum_{\boldsymbol\theta}T(\boldsymbol\theta\mid\boldsymbol\theta')=\pi(\boldsymbol\theta')$,
because the chain moves somewhere from $\boldsymbol\theta'$. For the second
claim, detailed balance is trivial when $\boldsymbol\theta'=\boldsymbol\theta$.
When $\boldsymbol\theta'\ne\boldsymbol\theta$, a move happens only if it is
proposed and accepted, so

$$
\pi(\boldsymbol\theta)\,T(\boldsymbol\theta'\mid\boldsymbol\theta)
=\pi(\boldsymbol\theta)\,g(\boldsymbol\theta'\mid\boldsymbol\theta)\,
\alpha(\boldsymbol\theta,\boldsymbol\theta')
=\min\big\{\pi(\boldsymbol\theta)\,g(\boldsymbol\theta'\mid\boldsymbol\theta),\;
\pi(\boldsymbol\theta')\,g(\boldsymbol\theta\mid\boldsymbol\theta')\big\},
$$

which is symmetric in the two states. The normalizer of $\pi$ cancels inside
$\alpha$, so $\widetilde p$ may be used in place of $\pi$. $\blacksquare$

The one step that is not routine is multiplying $\pi g$ into the minimum,
which turns the acceptance rule into a symmetric expression. On the two-state
chain the flows are $\tfrac14\cdot1=\tfrac34\cdot\tfrac13$. For continuous
parameters the rejection mass needs care (Exercise
:exref:`ex_mdl-bayesian-computation-mh-acceptance`).

Stationarity alone does not make a chain converge to $\pi$ from any start.
The chain must also be able to reach every region of positive posterior mass,
and it must not cycle deterministically between regions. Under these two
conditions the long-run fraction of time it spends in a region equals the
posterior mass of the region.

The cell implements the rule with a Gaussian step whose shape comes from the
Laplace covariance $\mathbf L\mathbf L^\top=\mathbf H^{-1}$.

```{.python .input #bayesian-computation-metropolis}
def metropolis(log_target, start, proposal_chol, n_steps):
    current = np.array(start, dtype=float)
    current_logp = log_target(current)
    states, accepted = np.empty((n_steps, len(current))), 0
    for t in range(n_steps):
        proposal = current + proposal_chol @ rng.standard_normal(len(current))
        proposal_logp = log_target(proposal)
        if np.log(rng.random()) < proposal_logp - current_logp:   # accept
            current, current_logp = proposal, proposal_logp
            accepted += 1
        states[t] = current            # a rejection repeats the state
    return states, accepted / n_steps
```

We run four chains of 12,000 steps from starts scattered twice as widely as
the Laplace Gaussian, $\boldsymbol\theta_{\mathrm{MAP}}+2\mathbf
L\boldsymbol\epsilon$, propose steps $1.1\,\mathbf L\boldsymbol\epsilon$, and
discard the first 2,000 states of each chain as warm-up.

```{.python .input #mdl-bayesian-computation-detailed-balance-and-stationarity-1}
def run_chains(scale, n_chains=4, n_steps=12000, warmup=2000):
    L = np.linalg.cholesky(laplace_cov)
    chains, rates = [], []
    for _ in range(n_chains):
        start = theta_map + 2 * L @ rng.standard_normal(2)  # over-dispersed
        states, rate = metropolis(log_joint, start, scale * L, n_steps)
        chains.append(states[warmup:])
        rates.append(rate)
    return np.array(chains), np.array(rates)

chains, rates = run_chains(1.1)
print('acceptance rates:', rates.round(3))
```

About half of the proposals are accepted in every chain. The Laplace
covariance only shapes the proposal. The acceptance rule, not the Gaussian,
fixes the stationary distribution, so a poor shape would slow the chain down
but not bias it.

### Diagnostics for Markov Chain Mixing
:label:`subsec_mdl-bayes-mcmc-diagnostics`

Consecutive states of a chain are correlated, and a chain starts wherever we
put it. Before averaging its states we ask three questions and answer each
with one tool.

*Have the chains lost their starts and agreed?* A trace plot of each chain
shows sticking and drift, and $\widehat R$ puts a number on the disagreement.
For several chains of $N$ states each, let $W$ be the average of their
variances and $B$ be $N$ times the variance of their means. Then

$$
\widehat R^2=\frac{\frac{N-1}{N}W+\frac1NB}{W}.
$$
:eqlabel:`eq_mdl-bayes-split-rhat`

If the chains agree, their means differ only by Monte Carlo noise, $B/N$ is
small compared with $W$, and $\widehat R\approx1$. If they disagree, $B$
inflates the numerator. The **split $\widehat R$** applies the formula to the
two halves of every chain, so that a single chain that drifts looks like two
chains that disagree. A value within $0.01$ of 1 is necessary for trusting the
chains, but not sufficient: chains that all miss the same mode agree with
each other.

*How many independent draws are the states worth?* Let $\rho_k$ be the
correlation between states $k$ steps apart. The integrated autocorrelation
time $\kappa=1+2\sum_{k\ge1}\rho_k$ is the number of states that carry as much
information as one independent draw, so $c$ chains of $N$ states each are
worth $N_{\mathrm{eff}}=cN/\kappa$ draws. We use the same symbol as for the
importance weights because both count the independent draws a sample is
worth.

The smallest case is the autoregressive sequence of order one, AR(1),
$x_t=\phi x_{t-1}+\varepsilon_t$ with independent noise $\varepsilon_t$ and
$|\phi|<1$: each state keeps a fraction $\phi$ of the previous one. Its
autocorrelations are $\rho_k=\phi^k$, so

$$
\kappa=1+2\sum_{k\ge1}\phi^k=\frac{1+\phi}{1-\phi}.
$$

With $\phi=0.9$, $\kappa=19$, and 10,000 states are worth about 530
independent draws; with $\phi=0.99$, $\kappa=199$, and they are worth about 50.

In practice the estimated far autocorrelations are mostly noise, so the sum
must be cut off. We sum them in pairs $\rho_{2k}+\rho_{2k+1}$, starting from $\rho_0=1$, stop
at the first negative pair, and use $\kappa=-1+2\sum_k(\rho_{2k}+\rho_{2k+1})$
over the pairs kept. Production software refines both tools, with
rank-normalized $\widehat R$ and separate effective sample sizes for the bulk
and the tails.

*How wrong is the estimated mean?* By the central limit theorem for the
chain, the **Monte Carlo standard error** (MCSE) of a posterior mean is the
posterior standard deviation divided by $\sqrt{N_{\mathrm{eff}}}$. The next
cell defines the tools.

```{.python .input #bayesian-computation-mcmc-diagnostics}
def split_rhat(chains):
    """Split R-hat per coordinate; chains has shape (c, N, d)."""
    half = chains.shape[1] // 2
    halves = np.concatenate([chains[:, :half], chains[:, half:2 * half]])
    W = halves.var(axis=1, ddof=1).mean(axis=0)
    B = half * halves.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((half - 1) / half * W + B / half) / W)

def autocorrelation(x):
    """rho_k of one sequence for k = 0, ..., N - 1, by FFT."""
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * len(x))
    acov = np.fft.irfft(f * np.conj(f))[:len(x)]
    return acov / acov[0]

def chain_ess(chains):
    """c N / kappa per coordinate, summing rho in pairs (Geyer)."""
    c, N, d = chains.shape
    ess = np.empty(d)
    for j in range(d):
        rho = np.mean([autocorrelation(chain[:, j]) for chain in chains],
                      axis=0)
        pairs = rho[:N - 1:2] + rho[1:N:2]        # rho_2k + rho_2k+1
        negative = np.flatnonzero(pairs < 0)
        n_pairs = negative[0] if len(negative) else len(pairs)
        ess[j] = c * N / (-1 + 2 * pairs[:n_pairs].sum())
    return ess
```

The cell applies them to the four chains, compares the MCMC mean with the
grid mean in units of the MCSE, and compares the measured $\kappa$ of the
slope with the AR(1) formula evaluated at the lag-1 autocorrelation $\rho_1$.

```{.python .input #mdl-bayesian-computation-diagnostics-for-markov-chain-mixing-1}
flat = chains.reshape(-1, 2)
ess = chain_ess(chains)
mcmc_mean, mcmc_sd = flat.mean(axis=0), flat.std(axis=0, ddof=1)
mcse = mcmc_sd / np.sqrt(ess)
rho1 = np.mean([autocorrelation(chain[:, 1])[1] for chain in chains])
print('split R-hat       :', split_rhat(chains).round(3))
print('ESS               :', ess.round().astype(int), 'of', len(flat))
print('(MCMC - grid)/MCSE:', ((mcmc_mean - post_mean) / mcse).round(1))
print(f'slope: rho_1 {rho1:.2f}, AR(1) kappa {(1 + rho1) / (1 - rho1):.1f},'
      f' measured kappa {len(flat) / ess[1]:.1f}')
```

The trace plot shows the slope's first 600 states after warm-up in each of
the four chains.

```{.python .input #mdl-bayesian-computation-diagnostics-for-markov-chain-mixing-2}
d2l.plot(np.arange(600), [chain[:600, 1] for chain in chains],
         xlabel='iteration', ylabel='slope', figsize=(6, 2.5))
```

The split $\widehat R$ is within $0.01$ of 1, and the four traces overlap.
The ESS is about a tenth of the 40,000 states, so $\kappa\approx10$: the
lag-1 autocorrelation is about $0.8$, and the AR(1) formula predicts a
$\kappa$ close to the measured one. The MCMC mean differs from the grid mean
by about one MCSE, the size of discrepancy that a correct error bar predicts.
This is the
strongest check in the section, because it tests the error bar against the
truth.

### Proposal Scale and Mixing Failures

The acceptance rate is the number a sampler reports most readily, and on its
own it says little. The cell reruns the chains with the proposal scale
multiplied by 0.05 and by 10.

```{.python .input #mdl-bayesian-computation-proposal-scale-and-mixing-failures}
for scale in [0.05, 10]:
    bad_chains, bad_rates = run_chains(scale)
    print(f'scale {scale:5}: acceptance {bad_rates.mean():.3f}, max R-hat '
          f'{split_rhat(bad_chains).max():.2f}, '
          f'min ESS {chain_ess(bad_chains).min():.0f}')
```

With tiny steps almost every proposal is accepted, about 97%. Yet
$\widehat R$ is well above $1.01$ and the 40,000 states are worth a few dozen
draws, because each accepted step moves the chain only a short distance. With huge steps about
2% of the proposals are accepted, and the chain repeats each state many times,
so the ESS is a few hundred. The best scale lies between (Exercise
:exref:`ex_mdl-bayesian-computation-proposal-scale`). A random walk moves
uphill only by chance, whereas a proposal that follows the gradient of
$\log\widetilde p$ moves uphill by construction. Such proposals come from
Langevin dynamics (:numref:`sec_mdl-score-matching-diffusion-flow`), and their
Metropolis-adjusted version is Exercise
:exref:`ex_mdl-bayesian-computation-mala`. Every sampling method leaves a
Monte Carlo error on every expectation it estimates, and the states of a chain
are worth only one draw in $\kappa$.

## Variational Inference
:label:`sec_mdl-bayes-approximations`

Sampling estimates each expectation anew and carries its Monte Carlo error.
*Variational inference* replaces the integration by one optimization. It
chooses a simple family of distributions $q_\psi$ with parameters $\psi$ and
optimizes $\psi$ so that $q_\psi$ is as close to the posterior as possible.
Expectations are then computed under $q_\psi$, in closed form or by cheap
sampling.

### The Evidence Lower Bound

We need an objective that measures closeness to the posterior although the
posterior's normalizer is unknown. For every $\boldsymbol\theta$, Bayes' rule
gives $\log p(\mathcal D)=\log p(\mathcal D,\boldsymbol\theta)-\log
p(\boldsymbol\theta\mid\mathcal D)$. Take the expectation of both sides under
$q_\psi$; the left side does not depend on $\boldsymbol\theta$, so it is
unchanged. Then add and subtract $\mathbb{E}_{q_\psi}[\log q_\psi]$:

$$
\log p(\mathcal D)
=\underbrace{\mathbb{E}_{q_\psi}\big[\log p(\mathcal D,\boldsymbol\theta)
-\log q_\psi(\boldsymbol\theta)\big]}_{\mathcal L(\psi)}
+D_{\textrm{KL}}\big(q_\psi\,\|\,p(\cdot\mid\mathcal D)\big).
$$
:eqlabel:`eq_mdl-bayes-elbo-gap`

The KL divergence is non-negative, so $\mathcal L$ is a lower bound on the log
evidence, the **evidence lower bound** (ELBO),

$$
\mathcal L(\psi)
=\mathbb{E}_{q_\psi}
 [\log p(\mathcal D,\boldsymbol\theta)-\log q_\psi(\boldsymbol\theta)]
\le\log p(\mathcal D).
$$
:eqlabel:`eq_mdl-bayes-vi-elbo`

Raising $\mathcal L$ lowers the KL divergence by the same amount, because
their sum is the constant $\log p(\mathcal D)$. Maximizing the ELBO over
$\psi$ therefore minimizes the KL divergence, and the unknown normalizer
drops out as an additive constant. Every term inside the
expectation can be evaluated pointwise, and the expectation can be estimated
by sampling from $q_\psi$. This is the ELBO of the latent-variable models of
:numref:`sec_mdl-latent-em-elbo`, with the parameters $\boldsymbol\theta$ in
the role of the latent variable. With $q$ over per-example latent variables and produced by
an encoder network, the same objective trains a variational autoencoder
:cite:`Kingma.Welling.2014`.

### The Direction of the KL Divergence

Variational inference minimizes $D_{\textrm{KL}}(q\|p)=\int q\log(q/p)$, the
*reverse* KL divergence, with $q$ first. Its integrand is large wherever
$q>0$ and $p\approx0$, so $q$ retreats from regions the posterior does not
cover and tends to settle on one mode: reverse KL is *mode-seeking*. The
*forward* divergence $D_{\textrm{KL}}(p\|q)=\int p\log(p/q)$ is large wherever
$p>0$ and $q\approx0$, so it pushes $q$ to cover all of the posterior's mass:
forward KL is *mass-covering*. For a Gaussian family its minimizer matches the
posterior's mean and covariance.
:numref:`fig_mdl-prob-bayes-kl-modes` shows both on a target with two modes.

![The direction of the KL divergence decides how a too-simple approximation fails. For a two-mode target, the best reverse-KL Gaussian, the one variational inference finds, settles on a single mode and understates the uncertainty; the best forward-KL Gaussian covers the mass of both modes and puts its bulk where the target is small.](../img/mdl-prob-bayes-kl-modes.svg)
:label:`fig_mdl-prob-bayes-kl-modes`

### A Mean-Field Gaussian Fit

The simplest family is the *mean-field* Gaussian
$q=\mathcal N(\mathbf m,\operatorname{diag}(\mathbf s^2))$, which treats the
coordinates as independent. To follow the gradient of the ELBO we write a draw
as $\boldsymbol\theta=\mathbf m+\mathbf s\odot\boldsymbol\epsilon$ with
$\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$. The distribution of
$\boldsymbol\epsilon$ does not depend on $\mathbf m$ and $\mathbf s$, so the
gradient moves inside the expectation
(:numref:`subsec_mdl-stochastic-gradient-estimators`). The entropy term
$-\mathbb{E}_q[\log q]$ equals $\sum_j\log s_j$ plus a constant, and the
gradients are

$$
\nabla_{\mathbf m}\mathcal L
=\mathbb{E}[\nabla_{\boldsymbol\theta}\log p(\mathcal D,\boldsymbol\theta)],
\qquad
\nabla_{\log\mathbf s}\mathcal L
=\mathbb{E}[\nabla_{\boldsymbol\theta}\log p(\mathcal D,\boldsymbol\theta)
 \odot\mathbf s\odot\boldsymbol\epsilon]+\mathbf1,
$$
:eqlabel:`eq_mdl-bayes-vi-gradients`

where $\mathbf 1$ is the derivative of the entropy $\sum_j\log s_j$ with
respect to $\log\mathbf s$. The first gradient moves the mean uphill on
average; the second widens $q$ until the entropy's push is balanced by the
posterior's curvature. Before running it, we can predict the answer on a
Gaussian target.

**Proposition.** For the target $p=\mathcal N(\boldsymbol\mu,\boldsymbol\Lambda^{-1})$
with precision matrix $\boldsymbol\Lambda$, the mean-field Gaussian that
minimizes $D_{\textrm{KL}}(q\|p)$ has mean $\boldsymbol\mu$ and variances
$s_j^2=1/\Lambda_{jj}$, the *conditional* variances of the target. For two
coordinates with correlation $\rho$, each $s_j$ is $\sqrt{1-\rho^2}$ times the
marginal standard deviation.

**Proof.** Up to a constant,
$D_{\textrm{KL}}(q\|p)=-\sum_j\log s_j+\tfrac12\mathbb{E}_q[(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Lambda(\boldsymbol\theta-\boldsymbol\mu)]$.
Under $q$ the coordinates are independent with means $m_j$ and variances
$s_j^2$, so the cross terms of the fluctuations have mean zero and the
expectation is $\sum_j\Lambda_{jj}s_j^2+(\mathbf m-\boldsymbol\mu)^\top\boldsymbol\Lambda(\mathbf m-\boldsymbol\mu)$.
The part in $\mathbf m$ is a positive definite quadratic, minimized at
$\mathbf m=\boldsymbol\mu$. Setting the derivative in $s_j$ to zero,
$-1/s_j+\Lambda_{jj}s_j=0$, gives $s_j^2=1/\Lambda_{jj}$. For $d=2$,
inverting the covariance gives
$1/\Lambda_{jj}=\sigma_j^2(1-\rho^2)$, where $\sigma_j$ is the marginal
standard deviation. $\blacksquare$

A factorized $q$ cannot tilt its axes to follow a correlation, and under the
reverse KL it also shrinks to the
width of a slice through the posterior rather than the width of the whole
posterior (:numref:`fig_mdl-prob-bayes-mean-field`). The shrinkage is a
property of the mean-field family under the reverse KL: a full-covariance
Gaussian fitted by the same objective to a Gaussian target is exact. In the
running example $\rho\approx-0.85$, so the mean-field standard deviations
should be about $\sqrt{1-0.85^2}=0.53$ times the true ones, about half.

![Diagonal Gaussians fitted to a correlated Gaussian posterior with standard deviations $(1,0.5)$ and correlation $\rho=-0.85$, drawn as 2-sd ellipses. A factorized family cannot tilt. Under the reverse KL (orange) it also shrinks to the conditional width, $\sqrt{1-\rho^2}\approx0.53$ times the marginal standard deviation; under the forward KL (green) it keeps the marginal width and covers mass the posterior does not have.](../img/mdl-prob-bayes-mean-field.svg)
:label:`fig_mdl-prob-bayes-mean-field`

The cell starts at the Laplace Gaussian and runs 3,000 steps of stochastic
gradient ascent on :eqref:`eq_mdl-bayes-vi-gradients`, with 256 draws per
step and a step size that decays like $1/\sqrt t$. It reports the average of
the last 1,000 iterates, which removes most of the optimization noise, and
the prediction of the proposition from the grid's own $\rho$.

```{.python .input #bayesian-computation-variational}
m_vi, log_s_vi = theta_map.copy(), np.log(laplace_sd)
m_avg, log_s_avg = np.zeros(2), np.zeros(2)
n_steps, n_avg = 3000, 1000
for t in range(1, n_steps + 1):
    eps = rng.standard_normal((256, 2))
    s_vi = np.exp(log_s_vi)
    g = grad_log_joint(m_vi + s_vi * eps)        # at theta = m + s * eps
    lr = 0.05 / np.sqrt(t)
    m_vi += lr * g.mean(axis=0)
    log_s_vi += lr * ((g * eps * s_vi).mean(axis=0) + 1)
    if t > n_steps - n_avg:                      # average the last iterates
        m_avg += m_vi / n_avg
        log_s_avg += log_s_vi / n_avg
s_avg = np.exp(log_s_avg)
print('             mean             sd')
print('grid      ', post_mean.round(3), post_sd.round(3))
print('Laplace   ', theta_map.round(3), laplace_sd.round(3))
print('mean-field', m_avg.round(3), s_avg.round(3))
print('predicted sd, grid sd * sqrt(1 - rho^2):',
      (post_sd * np.sqrt(1 - post_rho**2)).round(3))
```

The mean-field mean lies within about a tenth of a posterior standard
deviation of the grid mean. Its standard deviations are about half the grid's and match
the predicted values to within a few percent in a typical run. The
optimization has converged, and the remaining error comes from the family.
Richer families remove it at a higher cost: full-covariance Gaussians
(Exercise :exref:`ex_mdl-bayesian-computation-variational-families`),
mixtures, and normalizing flows. A converged ELBO shows that the optimization
within the family has stopped; it does not show that the family fits the
posterior.

## Comparison of the Methods
:label:`sec_mdl-bayes-decision-map`

We now have four answers for the same posterior. The cell compares them on one
marginal, the density of the slope $\theta_1$: the exact density from the
grid, the two Gaussians, and a histogram of the Metropolis states.

```{.python .input #bayesian-computation-comparison-plot}
def gauss_pdf(t, mean, sd):
    return np.exp(-0.5 * ((t - mean) / sd)**2) / (sd * np.sqrt(2 * np.pi))

b1 = grid_axes[1]
exact = grid_weight.reshape(m, m).sum(axis=0) / (b1[1] - b1[0])
hist, edges = np.histogram(flat[:, 1], bins=60, density=True)
d2l.plot([b1, b1, b1, (edges[:-1] + edges[1:]) / 2],
         [exact, gauss_pdf(b1, theta_map[1], laplace_sd[1]),
          gauss_pdf(b1, m_avg[1], s_avg[1]), hist],
         xlabel='slope', ylabel='density', figsize=(5, 3),
         xlim=(post_mean[1] - 4 * post_sd[1], post_mean[1] + 4 * post_sd[1]),
         legend=['grid (exact)', 'Laplace', 'mean-field', 'Metropolis'])
```

The Metropolis histogram follows the exact curve, skew included. The Laplace
curve is centered at the mode, to the left of the exact mean, and slightly too
narrow. The mean-field curve is about half as wide as the exact one, and the
mean-field fit also reports a correlation of zero against the grid's
$\rho\approx-0.85$. The table collects what each method represents, how it is
checked, and how it typically fails.

| Method | Representation | Main diagnostic | Characteristic failure |
|---|---|---|---|
| Grid | weighted points on a lattice | mass at the edge of the grid | cost $m^d$ in dimension $d$ |
| Laplace | one Gaussian at the mode | comparison with samples | skew, tails, boundaries, other modes |
| Rejection sampling | independent exact draws | acceptance rate | rate falls like $n^{-d/2}$; needs a bound $M$ |
| Importance sampling | independent weighted draws | weight ESS, largest weight, reruns | light-tailed or misplaced proposal |
| MCMC | correlated posterior draws | traces, split $\widehat R$, ESS, MCSE | slow mixing, unvisited modes |
| Variational inference | optimized member of a family | ELBO, comparison with samples | family too narrow; factorized $q$ too confident |

Choosing among them starts from the quantity we need: a posterior mean, a
predictive probability, or the probability of a rare event. Whatever the
method, its answer for that quantity should be checked against a sampler whose
error bar we trust, as the grid and the MCMC standard error were used here.

We can now estimate any posterior expectation of a small model in four ways.
We can check each estimate against the others and against its own diagnostic
(the weight ESS, $\widehat R$ with $\kappa$ and the MCSE, the ELBO), and
choose a method by the way it fails. None of the four scales cleanly to the posterior
of a neural network. Importance weights degenerate as the dimension grows
(Exercise :exref:`ex_mdl-bayesian-computation-higher-dimension`), a random
walk needs more steps per independent draw, the Hessian has $d^2$ entries,
and a factorized $q$ is too narrow by the factor computed here. A network's
symmetries also copy every mode of its posterior many times (Exercise
:exref:`ex_mdl-bayesian-computation-two-modes`). The sampling methods return
averages over finitely many draws, and the next section asks how far such an
average can be trusted.

## Summary

A model trained to one parameter vector predicts as if that vector were
certain. Far from the data the plausible vectors disagree by orders of
magnitude, and the posterior predictive, which averages them, can give a
rare label several times the probability that the plug-in prediction gives;
inside the data the two agree. Computing the average requires integrals over
a posterior that we can evaluate only up to its normalizer. Four methods
compute it with quantities in which the normalizer cancels: the curvature of
$\log\widetilde p$ at the mode (Laplace), the weight $\widetilde p/q$
(rejection and importance sampling), the acceptance ratio
$\widetilde p(\boldsymbol\theta')/\widetilde p(\boldsymbol\theta)$
(Metropolis), and an objective that differs from a KL divergence by a
constant (variational inference). Each comes with a check: the weight ESS,
the split $\widehat R$ with the autocorrelation time $\kappa$ and the MCSE,
and the ELBO. For small models they agree with an exact grid, and for neural
networks none of them is yet satisfactory.

## Exercises

1. **Metropolis–Hastings acceptance.** The acceptance probability
   :eqref:`eq_mdl-bayes-mh` turns any proposal into a chain that targets the
   posterior.
    1. On a continuous state space the transition kernel has a density part
       $g(\boldsymbol\theta'\mid\boldsymbol\theta)\alpha(\boldsymbol\theta,\boldsymbol\theta')$
       and an atom at the current state $\boldsymbol\theta$ of mass
       $r(\boldsymbol\theta)=1-\int g(\boldsymbol\theta'\mid\boldsymbol\theta)\alpha(\boldsymbol\theta,\boldsymbol\theta')\,d\boldsymbol\theta'$,
       the probability of a rejection. Show that the posterior is
       stationary, accounting for the atom. *Hint:* the atom contributes
       $\pi(\boldsymbol\theta')r(\boldsymbol\theta')$ to the density at
       $\boldsymbol\theta'$ after one step.
    1. An independence sampler proposes $\boldsymbol\theta'\sim q$
       regardless of the current state. Show that its acceptance probability
       is $\min(1,w(\boldsymbol\theta')/w(\boldsymbol\theta))$ with the
       importance weight $w$ of :eqref:`eq_mdl-bayes-importance`. What does
       the chain do after it reaches a state whose weight is far larger than
       the weights of typical proposals?
    1. Barker's rule accepts a symmetric proposal with probability
       $\widetilde p(\boldsymbol\theta')/(\widetilde
       p(\boldsymbol\theta)+\widetilde p(\boldsymbol\theta'))$. Show that it
       satisfies detailed balance. Write its transition matrix for the
       two-state chain of the text, and show that the Metropolis chain moves
       more often from both states. For a two-state chain the autocorrelation
       of any function of the state is $\rho_k=\lambda^k$, where $\lambda$ is
       the second eigenvalue of the transition matrix. Compute $\lambda$ and
       $\kappa=(1+\lambda)/(1-\lambda)$ for both rules. Which gives the smaller
       Monte Carlo error for the same number of steps? *Hint:* Barker gives
       the off-diagonal entries $3/4$ and $1/4$, against Metropolis' $1$ and
       $1/3$.
   :exlabel:`ex_mdl-bayesian-computation-mh-acceptance`

1. [code] **Proposal scale.** The function `run_chains` multiplies the
   Cholesky factor of `laplace_cov` by a scale, $1.1$ in the text.
    1. Sweep the scale over $\{0.05,0.2,0.5,1.1,1.7,2.4,3,10\}$, keeping the
       12,000 steps per chain and the warm-up. For each value, record the
       acceptance rate, the split $\widehat R$, and the smallest ESS of
       `chain_ess` divided by the $48{,}000$ log-joint evaluations of the four
       chains. Plot the ESS per evaluation against the acceptance rate. Over
       which range of acceptance rates does the ESS per evaluation stay
       within a factor of two of its maximum?
    1. For a $d$-dimensional standard Gaussian target and the proposal
       $\mathcal N(\boldsymbol\theta,(\ell^2/d)\mathbf I)$,
       :citet:`Roberts.Gelman.Gilks.1997` show that as $d\to\infty$ the
       efficiency is largest at $\ell\approx2.38$, where the acceptance rate
       is about $0.234$. The logistic posterior is close to its Laplace
       approximation, so the Cholesky factor makes it roughly standard. Which
       scale does this result suggest for $d=2$? How do the best scale of
       your sweep and its acceptance rate compare?
    1. Which of the diagnostics of the text flag the run with scale $0.05$?
       Is the acceptance rate one of them? How far is that run's posterior
       mean from the grid mean, in units of its own MCSE?
   :exlabel:`ex_mdl-bayesian-computation-proposal-scale`

1. [code] **Monte Carlo standard error.** A stationary AR(1) sequence
   $x_t=\phi x_{t-1}+\varepsilon_t$ with $|\phi|<1$ has the autocorrelations
   $\rho_k=\phi^k$ and the autocorrelation time $\kappa=(1+\phi)/(1-\phi)$.
    1. From the ESS that your run prints for the slope, compute $\kappa$ and
       the AR(1) coefficient $\phi$ with this autocorrelation time. Compare
       $\phi$ with the lag-1 autocorrelation $\rho_1$ that the cell prints.
       Is the chain more or less correlated at long lags than an AR(1)
       sequence with the same $\rho_1$?
    1. How many post-warm-up states per chain does the same sampler need to
       bring the MCSE of the slope's posterior mean down to $0.001$?
    1. Batch means estimate the MCSE without autocorrelations. Split each
       chain into batches of $b$ consecutive states, and divide the standard
       deviation of all batch means by the square root of the number of
       batches. Compute this estimate for $b\in\{25,100,500\}$ and compare it
       with the printed MCSE. What goes wrong when $b$ is not much larger
       than $\kappa$?
   :exlabel:`ex_mdl-bayesian-computation-mcse`

1. **Laplace and parameterization.** A coin shows 9 heads and 4 tails. Under
   a uniform prior, the posterior of its heads probability $\theta$ is
   $\mathrm{Beta}(10,5)$, with mean $2/3$, standard deviation $0.118$, and
   central $95\%$ interval $[0.419,0.872]$.
    1. Compute the Laplace approximation :eqref:`eq_mdl-bayes-laplace` in
       the parameter $\theta$. Give its mean, its standard deviation, its
       central $95\%$ interval, and the probability it places outside
       $[0,1]$.
    1. In the log-odds $\eta=\log(\theta/(1-\theta))$ the posterior density
       is proportional to
       $\operatorname{sigmoid}(\eta)^{10}(1-\operatorname{sigmoid}(\eta))^5$,
       because the Jacobian $d\theta/d\eta=\theta(1-\theta)$ adds one count to
       each side. Show that its mode is $\eta=\log2$, compute the curvature of
       the log density there, and map the central $95\%$ interval of the
       Laplace approximation back to $\theta$.
    1. Which of the two approximations is closer to the exact posterior?
       Why does the Laplace approximation depend on the parameterization,
       whereas the exact posterior probability of an event such as
       $\theta>1/2$ does not?
   :exlabel:`ex_mdl-bayesian-computation-laplace-parameterization`

1. [code] **Moderated predictions.** ● Under the Laplace approximation
   :eqref:`eq_mdl-bayes-laplace`, the logit
   $a=\mathbf z_\star^\top\boldsymbol\theta$ at a test input with
   $\mathbf z_\star=(1,x_\star)^\top$ is Gaussian with mean
   $\mu_a=\mathbf z_\star^\top\boldsymbol\theta_{\mathrm{MAP}}$ and variance
   $s_a^2=\mathbf z_\star^\top\mathbf H^{-1}\mathbf z_\star$. The probit
   approximation $\operatorname{sigmoid}(a)\approx\Phi(\lambda a)$ with
   $\lambda^2=\pi/8$, where $\Phi$ is the standard normal c.d.f., turns the
   predictive integral :eqref:`eq_mdl-bayes-predictive` into a closed form
   :cite:`MacKay.1992`.
    1. Show that
       $\mathbb{E}[\Phi(\lambda a)]=\Phi\bigl(\lambda\mu_a/\sqrt{1+\lambda^2s_a^2}\bigr)$
       for $a\sim\mathcal N(\mu_a,s_a^2)$. *Hint:* write $\Phi(\lambda a)$ as
       $P(U\le\lambda a)$ for an independent standard normal $U$. Conclude
       that the moderated prediction
       $\operatorname{sigmoid}\bigl(\mu_a/\sqrt{1+\pi s_a^2/8}\bigr)$ lies
       between $\operatorname{sigmoid}(\mu_a)$ and $1/2$.
    1. Compute the moderated predictions at $x_\star\in\{-3,2,7\}$ from
       `theta_map` and `laplace_cov`, and compare them with the plug-in and
       posterior predictive values that your run prints.
    1. Compute three quantities at the same inputs: the plug-in
       $\operatorname{sigmoid}(\mathbf z_\star^\top\boldsymbol\theta_{\mathrm{MAP}})$,
       the prediction at the posterior mean
       $\operatorname{sigmoid}(\mathbf z_\star^\top\mathbb{E}[\boldsymbol\theta\mid\mathcal D])$,
       and the posterior predictive
       $\mathbb{E}[\operatorname{sigmoid}(\mathbf z_\star^\top\boldsymbol\theta)\mid\mathcal D]$.
       Which of the two steps, from mode to mean and from mean to average,
       moves the prediction away from $1/2$, and which toward it? Why are
       both steps small at $x_\star=2$ but not at $x_\star=-3$ or
       $x_\star=7$?
       What does this imply for using the Laplace approximation to moderate
       overconfident predictions?
   :exlabel:`ex_mdl-bayesian-computation-moderated-predictions`

1. [code] **Predicting the weight ESS.** ● By the proposition of the text,
   $N_{\mathrm{eff}}/N$ approaches $(\mathbb{E}_q[w])^2/\mathbb{E}_q[w^2]$ as
   the number $N$ of draws grows.
    1. Let the normalized target be $\mathcal N(\mu,s_p^2)$ and the proposal
       $\mathcal N(\nu,s_q^2)$ in one dimension, and let $r=s_q/s_p$. Show
       that for $s_q^2>s_p^2/2$

       $$\frac{\mathbb{E}_q[w^2]}{(\mathbb{E}_q[w])^2}=\frac{r^2}{\sqrt{2r^2-1}}\exp\!\left(\frac{(\mu-\nu)^2}{2s_q^2-s_p^2}\right),$$

       and that the ratio is infinite for $s_q^2\le s_p^2/2$. Which $r$
       minimizes the ratio when $\mu=\nu$?
    1. Approximate the posterior by independent Gaussians with the grid means
       and standard deviations, ignoring the correlation, so that the ratio
       factorizes over the coordinates. Predict the ESS of the 20,000
       prior-proposal draws and compare it with the printed value. Repeat for
       the prior proposals with $\tau=0.5$ and $\tau=10$: set `prior_var`
       $=\tau^2$ in the model cell, rerun the cells through the importance
       cell, and compare prediction and measurement. Which factor of the
       formula makes a narrow prior a poor proposal, and which makes a broad
       prior a poor one?
    1. Under the same approximation, for which coordinates of your run does
       the proposal $\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},0.3\,\mathbf I)$
       have infinite weight variance? Run it ten times with 5,000 draws and
       record the ESS, the largest normalized weight, and the estimated
       posterior mean. Why does a run whose estimate looks plausible not show
       that the proposal is adequate?
    1. Rejection sampling from the prior keeps a draw with probability
       $L(\boldsymbol\theta)/\max L$, where $L$ is the likelihood and $\max L$
       its value at the maximum-likelihood estimate (for separable data, its
       supremum 1). Measure the acceptance rate on your data and on data with
       ten times as many points. Explain the decay with $n$ from the volume of the
       posterior, and compare the rate with the ESS fraction of importance
       sampling from the same prior.
   :exlabel:`ex_mdl-bayesian-computation-weight-ess`

1. [code] **Variational families.** ● The variational cell fits a mean-field
   Gaussian by stochastic gradient ascent on the ELBO
   :eqref:`eq_mdl-bayes-vi-elbo`, following the gradients
   :eqref:`eq_mdl-bayes-vi-gradients` with a decaying step.
    1. Predict the mean-field standard deviations from the grid standard
       deviations and correlation, and compare them with the printed values.
       What would the forward-KL diagonal Gaussian give?
    1. Rerun the cell three times. Compare the spread of the final iterate
       `m_vi` across runs with the spread of the averaged iterate `m_avg`.
       How much of the gap between the mean-field mean and the grid mean is
       approximation error, and how much is optimization noise?
    1. Replace the mean-field family by a full-covariance Gaussian
       $\boldsymbol\theta=\mathbf m+\mathbf L\boldsymbol\epsilon$, where
       $\mathbf L$ is lower triangular with diagonal entries $e^{\ell_1}$ and
       $e^{\ell_2}$. Extend :eqref:`eq_mdl-bayes-vi-gradients` to the
       gradients with respect to $\mathbf m$, $\ell_1$, $\ell_2$, and the
       off-diagonal entry of $\mathbf L$. Implement them, and compare the
       fitted standard deviations and correlation with the grid values.
   :exlabel:`ex_mdl-bayesian-computation-variational-families`

1. [code] **Estimating the evidence.** ● Let $q$ be a normalized proposal,
   let $w=p(\mathcal D,\boldsymbol\theta)/q(\boldsymbol\theta)$, and let
   $\hat Z_N=\frac1N\sum_{s=1}^Nw(\boldsymbol\theta_s)$ for independent draws
   $\boldsymbol\theta_s\sim q$.
    1. Show that $\hat Z_N$ is an unbiased estimate of $p(\mathcal D)$ and
       that $\mathbb{E}[\log\hat Z_N]\le\log p(\mathcal D)$. Show that
       $\mathbb{E}[\log\hat Z_1]$ is the ELBO :eqref:`eq_mdl-bayes-vi-elbo` of
       $q$ and that $\mathbb{E}[\log\hat Z_N]$ does not decrease with $N$
       :cite:`Burda.Grosse.Salakhutdinov.2016`.
    1. The function `log_joint` omits the normalizing constant
       $-\log(2\pi\tau^2)$ of the two-dimensional prior. Add it, and compute
       $\log p(\mathcal D)$ in three ways: by summing the density over the
       grid and multiplying by the area of a grid cell, by $\hat Z_N$ with
       5,000 Laplace-proposal draws, and by the Laplace evidence
       $\log\widetilde p(\boldsymbol\theta_{\mathrm{MAP}})+\log2\pi-\tfrac12\log\det\mathbf H$.
       The Laplace value comes out a few percent low in every run. Which
       feature of the posterior, seen in the grid cell, explains the sign?
    1. Estimate the ELBO of three Gaussians: the Laplace approximation, the
       mean-field fit, and the full-covariance fit of Exercise
       :exref:`ex_mdl-bayesian-computation-variational-families`c. Use $10^5$
       draws from each and report the MCSE of each estimate. By
       :eqref:`eq_mdl-bayes-elbo-gap`, the gap between $\log p(\mathcal D)$
       and the ELBO of $q$ is $D_{\textrm{KL}}(q\|p(\cdot\mid\mathcal D))$.
       Rank the three Gaussians by this divergence, and explain where the
       Laplace Gaussian falls in the ranking.
   :exlabel:`ex_mdl-bayesian-computation-evidence`

1. [code] **A posterior with two modes.** ● Fifty observations are drawn as
   $y_i\sim\mathcal N(\theta^2,1)$ with the true value $\theta=1$, and
   $\theta$ has the prior $\mathcal N(0,4)$. The likelihood depends on
   $\theta$ only through $\theta^2$, just as the loss of a neural network
   does not change when the incoming weights, the bias, and the outgoing
   weights of a $\tanh$ hidden unit flip sign together. The posterior is
   therefore symmetric under $\theta\mapsto-\theta$.
    1. Compute the posterior on a one-dimensional grid. Report
       $\mathbb{E}[\theta\mid\mathcal D]$,
       $\mathbb{E}[\theta^2\mid\mathcal D]$, and the ratio of the posterior
       density at $\theta=0$ to its value at a mode.
    1. Show that the Gaussian that minimizes the forward divergence
       $D_{\textrm{KL}}(p\|q)$ matches the mean and the variance of $p$.
       Where does this Gaussian put its mode for the present posterior, and
       how plausible is that point under $p$?
    1. Fit three local approximations: the Laplace approximation at the
       positive mode, a one-dimensional Gaussian fitted by the ELBO ascent of
       the variational cell started at $m=0.5$, and four Metropolis chains
       with proposal standard deviation $0.2$ started at
       $\theta\in\{0.5,1,1.5,2\}$. Then rerun the chains from
       $\theta\in\{-2,-0.5,0.5,2\}$. Which methods report a single mode? For
       each run, which of its diagnostics (split $\widehat R$ and the ESS for
       the chains, the ELBO for the variational fit) reveal the problem?
    1. Which posterior expectations does a single-mode approximation still
       estimate correctly? A network with $h$ hidden $\tanh$ units has the
       sign flip of this problem for every unit and, in addition, the $h!$
       permutations of its hidden units (Exercise
       :exref:`ex_mdl-maximum-likelihood-relu-symmetries`). How many
       equivalent copies of a mode does its posterior have under an isotropic
       Gaussian prior? What does this suggest about approximating the
       posterior of a neural network by one of the methods of this section?
   :exlabel:`ex_mdl-bayesian-computation-two-modes`

1. [code] **Tails of the logistic posterior.** ●● A central limit theorem
   for importance sampling needs a finite second moment of the weights. The
   Laplace proposal of the importance cell usually reaches a high ESS.
   Throughout, $\tau^2=$ `prior_var` $=4$ is the prior variance in
   the posterior :eqref:`eq_mdl-bayes-logistic-posterior`.
    1. Fix a unit vector $\mathbf v$ and let $\boldsymbol\theta=t\mathbf v$.
       Show that the log-likelihood in
       :eqref:`eq_mdl-bayes-logistic-posterior` equals $-c(\mathbf v)t+O(1)$
       as $t\to\infty$, where $c(\mathbf v)$ is the sum of
       $|\mathbf z_i^\top\mathbf v|$ over the observations with $y_i=1$ and
       $\mathbf z_i^\top\mathbf v<0$ or with $y_i=0$ and
       $\mathbf z_i^\top\mathbf v>0$. For which datasets is
       $c(\mathbf v)>0$ for every $\mathbf v$?
    1. Conclude that a Gaussian proposal $\mathcal N(\mathbf m,\mathbf S)$
       has $\mathbb{E}_q[w^2]=\infty$ when some eigenvalue of $\mathbf S$ is
       smaller than $\tau^2/2$, and $\mathbb{E}_q[w^2]<\infty$ when every
       eigenvalue of $\mathbf S$ exceeds $\tau^2/2$. Which of the two
       proposals of the importance cell has weights with finite variance?
       Compare this answer with the one that the formula of Exercise
       :exref:`ex_mdl-bayesian-computation-weight-ess`a gives for the Laplace
       proposal when the posterior is approximated by a Gaussian.
    1. Along each eigenvector of `laplace_cov`, in both directions, find the
       Mahalanobis distance from $\boldsymbol\theta_{\mathrm{MAP}}$ beyond
       which $\widetilde p^2/q$ exceeds its value at
       $\boldsymbol\theta_{\mathrm{MAP}}$. How likely is a single
       Laplace-proposal draw to land that far out? Reconcile your answer
       with the ESS that your run prints.
    1. :citet:`Geweke.1989` builds importance densities from multivariate
       Student-$t$ approximations at the posterior mode. Use a Student-$t$
       proposal with $3$ degrees of freedom, location
       $\boldsymbol\theta_{\mathrm{MAP}}$, and scale matrix `laplace_cov`, and
       show that its weights have finite variance. Compare it with the
       Laplace proposal over 20 runs on the ESS and on the estimate of the
       tail probability $P(\theta_1>\bar\theta_1+3s_1\mid\mathcal D)$, where
       $\bar\theta_1$ and $s_1$ are the grid mean and standard deviation of
       the slope; for a reference value, evaluate the posterior on a grid
       twice as wide. Also try the defensive mixture of the text, and explain
       why its weights are bounded.
   :exlabel:`ex_mdl-bayesian-computation-posterior-tails`

1. [extended] **Approximations in higher dimension.** ●● Extend the running
   example to $d$ parameters: an intercept and $d-1$ features with
   independent standard normal entries, $n=10d$ observations whose labels
   are drawn from a true parameter with independent
   $\mathcal N(0,1.5^2/d)$ entries, and the prior
   $\mathcal N(\mathbf 0,4\mathbf I)$. No grid is available, so a long
   Metropolis run serves as the reference.
    1. For $d\in\{2,5,10,20,50\}$, compute the Laplace approximation and the
       ESS of 5,000 Laplace-proposal draws. Repeat with $n=40d$. Explain how
       $N_{\mathrm{eff}}/N$ depends on $d$ and on $n/d$.
    1. Run four Metropolis chains with the proposal $(2.38/\sqrt d)\mathbf L$,
       where $\mathbf L$ is the Cholesky factor of the Laplace covariance
       (Exercise :exref:`ex_mdl-bayesian-computation-proposal-scale`b).
       Measure the acceptance rate and the smallest ESS per log-joint
       evaluation over the coordinates as functions of $d$. Which power of
       $d$ describes the efficiency?
    1. At $d=50$ and $n=500$, give importance sampling and Metropolis the
       same budget of 20,000 log-joint evaluations, and repeat each run five
       times. Measure their errors in the posterior mean against the
       reference, in units of posterior standard deviations, and compare them
       with the errors that each method's own ESS predicts. Report one row
       per method with the number of evaluations, the ESS, the predicted
       error, and the measured error. Which method would you use at this
       budget?
   :exlabel:`ex_mdl-bayesian-computation-higher-dimension`

1. [code] **Langevin proposals.** ●● The Metropolis-adjusted Langevin
   algorithm proposes
   $\boldsymbol\theta'=\boldsymbol\theta+\tfrac{\epsilon^2}{2}\mathbf
   C\nabla\log\widetilde p(\boldsymbol\theta)+\epsilon\,\mathbf
   C^{1/2}\boldsymbol\xi$ with $\boldsymbol\xi\sim\mathcal N(\mathbf
   0,\mathbf I)$ and a preconditioner $\mathbf C$; take $\mathbf C$ to be the
   Laplace covariance $\mathbf H^{-1}$ and $\mathbf C^{1/2}$ its Cholesky
   factor.
    1. Write down the proposal density
       $g(\boldsymbol\theta'\mid\boldsymbol\theta)$ and the Hastings
       factor of the acceptance rule :eqref:`eq_mdl-bayes-mh`. Why is the proposal not symmetric?
    1. Implement the sampler with `grad_log_joint` and run four chains for
       $\epsilon\in\{0.5,1,1.5\}$. Report the acceptance rate, the split
       $\widehat R$, and the ESS per gradient evaluation, and compare with
       the random walk's ESS per log-joint evaluation.
    1. Without the accept step the chain is the unadjusted Langevin
       algorithm. Run it at $\epsilon=1$ and compare its mean and standard
       deviations with the grid. Which of the two errors, the bias of the
       discretization or the Monte Carlo error, does the Metropolis step
       remove?
   :exlabel:`ex_mdl-bayesian-computation-mala`

[Discussions](https://d2l.discourse.group/t/36833)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §27.6]{.kicker}

Averaging over a posterior known only up to a constant<br>**Bayesian computation**
:::
:::

::: {.slide title="Far From the Data, One Fitted Model Hides Orders of Magnitude of Doubt"}
[The problem]{.kicker}

::: {.cols .vc}
::: {.col}
- Logistic regression, 20 points with $x\in[0,4]$; plug-in uses the MAP.
- At $x=-3$ the plausible vectors disagree by orders of magnitude; their
  average is typically 3–10× the plug-in.
- Inside the data the two agree to about $0.02$.
:::

::: {.col .fig .big}
@!mdl-bayesian-computation-the-plug-in-and-the-posterior-predictive-2
:::
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- fit a Laplace Gaussian at the mode, and say what it misses;
- sample exactly by rejection, and with weights by importance sampling;
- build a Metropolis chain and prove that it targets the posterior;
- read the ESS, $\widehat R$, $\kappa$ and the MCSE;
- fit a mean-field Gaussian by the ELBO and predict its error.

**Idea:** each method uses only ratios of $\widetilde p$, and the constant
cancels. **Running examples:** the 20-point logistic posterior; the coin with
9 heads and 4 tails.
:::

::: {.slide title="From One Parameter Vector to an Average Over All of Them"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Random Variables | densities, moments, covariance |
| Distributions | the named laws and their losses |
| The Exponential Family | one form, convex fits, conjugacy |
| Statistics | error bars, tests, intervals |
| Maximum Likelihood | losses, Fisher information, MAP, EM |
| **Bayesian Computation** | **averaging over a posterior** |
| Concentration Inequalities | exponential tails of averages |
| Uniform Convergence | guarantees after selection |
| Naive Bayes | one classifier, end to end |
:::

::: {.col .narrow}
- **Uses:** Bayes' rule, MAP, the Hessian, KL.
- **Skips:** Gibbs, Hamiltonian Monte Carlo, network posteriors.
:::
:::
:::

::: {.slide title="Newton at the Mode Gives a Gaussian for One More Hessian"}
[Laplace]{.kicker}

::: {.cols .vc}
::: {.col}
$$\log\widetilde p(\boldsymbol\theta)\approx\log\widetilde p(\boldsymbol\theta_{\mathrm{MAP}})-\tfrac12(\boldsymbol\theta-\boldsymbol\theta_{\mathrm{MAP}})^\top\mathbf H(\boldsymbol\theta-\boldsymbol\theta_{\mathrm{MAP}})$$

so $p(\boldsymbol\theta\mid\mathcal D)\approx\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},\mathbf H^{-1})$,
with $\mathbf H$ the negative Hessian.

- Gamma(4,1): mode 3, curvature $\tfrac13$, so $\mathcal N(3,3)$.
- Truth: mean 4, variance 4; the Gaussian puts 4.2% below 0.
:::

::: {.col .fig}
@fig:mdl-prob-bayes-laplace
:::
:::
:::

::: {.slide title="Check: Laplace on the Coin by Hand"}
[Check]{.kicker}

Posterior $\mathrm{Beta}(10,5)\propto\theta^9(1-\theta)^4$ (9 heads, 4 tails,
uniform prior). Find the Laplace mean and standard deviation.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Mode $9/13=0.692$; curvature $9/\theta^2+4/(1-\theta)^2=61$, so
sd $=1/\sqrt{61}=0.128$. Exact: mean $0.667$, sd $0.118$.
:::
:::
:::

::: {.slide title="The Mode Is Not the Mean"}
[A grid reference]{.kicker}

Grid of $241\times241$ points at MAP $\pm7$ Laplace sds:

@!bayesian-computation-grid

- Slope mean above the MAP (about ¼ sd), intercept below, in nearly every run.
- Laplace sds a few percent small; correlation about $-0.85$.
:::

::: {.slide title="Rejection Sampling Is Exact but Wasteful"}
[Rejection sampling]{.kicker}

Envelope $Mq\ge\widetilde p$; keep $\boldsymbol\theta\sim q$ with probability
$\widetilde p/(Mq)$. Kept draws are exact; rate $=p(\mathcal D)/M$.

| coin data | 9 H, 4 T | 90 H, 40 T | 900 H, 400 T |
|---|---|---|---|
| acceptance | 0.305 | 0.10 | 0.03 |

- Rate $\propto1/\sqrt n$ here, $n^{-d/2}$ in $d$ dimensions.
- Needs a known bound $M$.
:::

::: {.slide title="Importance Weights Keep Every Draw"}
[Importance sampling]{.kicker}

I. The posterior is $\widetilde p/p(\mathcal D)$, so
$\mathbb{E}_{p}[h]=\int h\,\widetilde p\,\big/\!\int\widetilde p$.
II. Multiply and divide by $q$ inside both integrals:

$$\boxed{\mathbb{E}_{p}[h]=\frac{\mathbb{E}_q[wh]}{\mathbb{E}_q[w]},\quad w=\widetilde p/q}$$

- $\mathbb{E}_q[w]=p(\mathcal D)$: the constant cancels in the ratio.
- Prior as $q$: weight $=$ likelihood.
- ESS limit $(\mathbb{E}w)^2/\mathbb{E}w^2\ge p(\mathcal D)/M$, since $w\le M$.

@!mdl-bayesian-computation-the-effective-sample-size
:::

::: {.slide title="The ESS Measures Unequal Weights, Not Unseen Mass"}
[Effective sample size]{.kicker}

::: {.cols .vc}
::: {.col}
$N_{\mathrm{eff}}=1/\sum_s\bar w_s^2\ \to\ N/(1+\chi^2(p\|q))$.

Gaussian target with sd $s_p$, Gaussian $q$ at the same center with sd
$rs_p$: limit $\sqrt{2r^2-1}/r^2$, and zero for $r\le1/\sqrt2$.
:::

::: {.col .fig .big}
@fig:mdl-prob-bayes-importance
:::
:::
:::

::: {.slide title="Aim the Proposal: the Prior Keeps a Few Percent, Laplace Most"}
[Importance sampling]{.kicker}

@!bayesian-computation-importance

- Posterior tails are the prior's (variance $\tau^2=4$); a Gaussian $q$
  with a covariance eigenvalue below $\tau^2/2$ has infinite weight variance.
- The Laplace covariance has one far below 2: the ESS collapses in rare runs.
- Fix: Student-$t$, or $0.9\,$Laplace $+\,0.1\,$prior.
:::

::: {.slide title="Metropolis: Uphill Always, Downhill With Probability $\widetilde p'/\widetilde p$"}
[MCMC]{.kicker}

::: {.cols .vc}
::: {.col}
$$\alpha=\min\!\left(1,\frac{\widetilde p(\boldsymbol\theta')\,g(\boldsymbol\theta\mid\boldsymbol\theta')}{\widetilde p(\boldsymbol\theta)\,g(\boldsymbol\theta'\mid\boldsymbol\theta)}\right)$$

- Propose $\boldsymbol\theta'\sim g(\cdot\mid\boldsymbol\theta)$; $\widetilde p'=\widetilde p(\boldsymbol\theta')$.
- A rejection repeats the state, and the repeat counts.
- Symmetric $g$: the Hastings factor $g(\boldsymbol\theta\mid\boldsymbol\theta')/g(\boldsymbol\theta'\mid\boldsymbol\theta)$ is 1.
- Only a ratio of $\widetilde p$: the constant cancels.
:::

::: {.col .fig}
@fig:mdl-prob-bayes-metropolis
:::
:::
:::

::: {.slide title="Check: Two States With $\pi=(1/4,3/4)$"}
[Check]{.kicker}

The proposal always names the other state. Write $\mathbf T$, check
$\pi\mathbf T=\pi$, and say what counting only accepted states gives.

::: {.fragment .fade-up}
::: {.cols .vc}
::: {.col}
$\mathbf T=\begin{pmatrix}0&1\\1/3&2/3\end{pmatrix}$, $\pi\mathbf T=\pi$;
accepted states alone alternate 2, 1, 2, 1: $(1/2,1/2)$.
:::

::: {.col .fig .big}
@fig:mdl-prob-bayes-two-state
:::
:::
:::
:::

::: {.slide title="Detailed Balance Makes the Posterior Stationary"}
[Proof]{.kicker}

I. For $\boldsymbol\theta'\ne\boldsymbol\theta$ a move needs a proposal and an acceptance:
$\pi(\boldsymbol\theta)T(\boldsymbol\theta'\mid\boldsymbol\theta)=\min\{\pi(\boldsymbol\theta)g(\boldsymbol\theta'\mid\boldsymbol\theta),\,\pi(\boldsymbol\theta')g(\boldsymbol\theta\mid\boldsymbol\theta')\}$, symmetric in the two states.

II. Sum $\pi(\boldsymbol\theta)T(\boldsymbol\theta'\mid\boldsymbol\theta)=\pi(\boldsymbol\theta')T(\boldsymbol\theta\mid\boldsymbol\theta')$ over $\boldsymbol\theta$:

$$\boxed{\textstyle\sum_{\boldsymbol\theta}\pi(\boldsymbol\theta)\,T(\boldsymbol\theta'\mid\boldsymbol\theta)=\pi(\boldsymbol\theta')}$$

Convergence also needs: every region reachable, no deterministic cycles.
:::

::: {.slide title="Judge Mixing by $\widehat R$, $\kappa$ and the MCSE"}
[Diagnostics]{.kicker}

- $\widehat R^2=\big(\tfrac{N-1}{N}W+\tfrac1NB\big)/W$ on half-chains: they must agree.
- $\kappa=1+2\sum_k\rho_k$; $c$ chains of $N$: $N_{\mathrm{eff}}=cN/\kappa$.
- MCSE $=$ sd$/\sqrt{N_{\mathrm{eff}}}$.

@!mdl-bayesian-computation-diagnostics-for-markov-chain-mixing-1
:::

::: {.slide title="An AR(1) Chain With $\phi=0.9$ Is Worth One Draw per 19 States"}
[Worked example]{.kicker}

$x_t=\phi x_{t-1}+\varepsilon_t$: $\rho_k=\phi^k$, so

$$\kappa=1+2\sum_{k\ge1}\phi^k=\frac{1+\phi}{1-\phi}.$$

| $\phi$ | $\kappa$ | 10,000 states are worth |
|---|---|---|
| 0.9 | 19 | about 530 draws |
| 0.99 | 199 | about 50 draws |

Logistic chains: $\rho_1\approx0.8$, $\kappa\approx10$.
:::

::: {.slide title="Tiny Steps Accept 97% and Go Nowhere; Huge Steps Accept 2%"}
[Mixing failures]{.kicker}

@!mdl-bayesian-computation-proposal-scale-and-mixing-failures

- Scale 0.05: $\widehat R$ well above 1.01, a few dozen effective draws.
- Scale 10: each state repeated, a few hundred effective draws.
- The acceptance rate alone certifies nothing; aim between the extremes
  (0.234 in high dimension).
:::

::: {.slide title="ELBO = Log Evidence − KL"}
[Variational inference]{.kicker}

I. $\log p(\mathcal D)=\log p(\mathcal D,\boldsymbol\theta)-\log p(\boldsymbol\theta\mid\mathcal D)$ for every $\boldsymbol\theta$.

II. Take $\mathbb{E}_{q}$; add and subtract $\mathbb{E}_q[\log q]$:

$$\boxed{\log p(\mathcal D)=\underbrace{\mathbb{E}_q[\log p(\mathcal D,\boldsymbol\theta)-\log q]}_{\mathcal L}+D_{\textrm{KL}}(q\,\|\,p(\cdot\mid\mathcal D))}$$

Raising $\mathcal L$ lowers the KL; the constant does not move the optimum.
:::

::: {.slide title="A Factorized Gaussian Shrinks by $\sqrt{1-\rho^2}$"}
[Mean field]{.kicker}

::: {.cols .vc}
::: {.col}
Target $\mathcal N(\boldsymbol\mu,\boldsymbol\Lambda^{-1})$: reverse-KL mean
field gives mean $\boldsymbol\mu$, variances $1/\Lambda_{jj}$, the
conditional variances.

- Cannot tilt; width of a slice.
- Full-covariance $q$ would be exact.
:::

::: {.col .fig}
@fig:mdl-prob-bayes-mean-field
:::
:::
:::

::: {.slide title="Check: With $\rho=-0.85$, How Narrow Is the Mean-Field Fit?"}
[Check]{.kicker}

The logistic posterior has correlation about $-0.85$. By what factor is each
mean-field standard deviation too small, and what correlation does the fit
report?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\sqrt{1-0.85^2}=0.53$: about half. It reports correlation 0.
The run agrees:

@!bayesian-computation-variational
:::
:::
:::

::: {.slide title="One Marginal, Four Answers"}
[Comparison]{.kicker}

@!bayesian-computation-comparison-plot

- Metropolis follows the exact curve, skew included.
- Laplace sits at the mode, left of the mean; mean field is half as wide.
:::

::: {.slide title="Choose a Method by How It Fails"}
[Comparison]{.kicker}

| method | gives | check | fails by |
|---|---|---|---|
| Laplace | Gaussian at mode | vs. samples | skew, tails, modes |
| rejection | exact draws | acceptance | rate $\propto n^{-d/2}$, needs $M$ |
| importance | weighted draws | weight ESS | light or misplaced $q$ |
| MCMC | correlated draws | $\widehat R$, ESS, MCSE | slow mixing, lost modes |
| mean-field VI | fitted $q$ | ELBO | too narrow |
:::

::: {.slide title="Four Ratios in Which the Constant Cancels"}
[Wrap-up]{.kicker}

- Laplace: curvature at the mode; Gamma(4,1) gives $\mathcal N(3,3)$.
- Rejection, importance: $\widetilde p/q$; coin keeps 30%, ESS 42%.
- Metropolis: $\widetilde p'/\widetilde p$; two states keep $(1/4,3/4)$.
- Diagnostics: $\kappa=19$ at $\phi=0.9$; error $\approx$ one MCSE.
- VI: ELBO $=$ log evidence $-$ KL; mean field shrinks by $0.53$.

**Next:** how far an average over a finite sample can be trusted.
:::
