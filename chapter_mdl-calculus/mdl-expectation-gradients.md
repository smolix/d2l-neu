# Expectations and Their Gradients
:label:`sec_mdl-expectation-gradients`

Almost every loss we have minimized so far is an average: over a minibatch of
examples, over random dropout masks, or over noise injected into an input. As
long as the source of randomness does not depend on the model parameters,
differentiating such an average is straightforward. The derivative of an
average equals the average of the derivatives, so minibatch gradient descent
computes it directly without any extra machinery. The difficulty arises when
the parameters control the probability distribution itself. A variational
autoencoder draws its latent code from a distribution whose mean and variance
are produced by its encoder; a policy-gradient agent
(:numref:`sec_policygradient`) samples actions from the very policy it is
trying to improve. In both cases, the objective is an expectation over samples
whose distribution shifts with the parameters, yet a sample, once drawn, is
just a number. It carries no memory of the parameter that generated it.

Consider the simplest problem of this kind. Let $X\sim\mathcal N(\theta,1)$ be
a normal random variable with mean $\theta$ and unit variance, and take the
expected squared loss

$$
\ell(\theta)=\mathbb{E}\bigl[X^2\bigr]=\theta^2+1 .
$$

The closed form follows immediately by writing $X=\theta+Z$ with $Z$ standard
normal:
$\mathbb{E}[(\theta+Z)^2]=\theta^2+2\theta\,\mathbb{E}[Z]+\mathbb{E}[Z^2]$,
where $\mathbb{E}[Z]=0$ by symmetry and $\mathbb{E}[Z^2]=1$
(:numref:`subsec_mdl-integration-by-parts`). At $\theta=1.5$, the exact
derivative is $\ell'(1.5)=3$. In practice, however, a model can draw samples of
$X$ but cannot evaluate $\ell(\theta)$ in closed form. How, then, can we
estimate $\ell'(\theta)$ from samples alone?

Calculus offers two fundamentally different answers. Both express the
expectation as an integral (:numref:`sec_mdl-integral_calculus`) and pass the
derivative inside it. From there they diverge: the *score-function* estimator
differentiates the probability density while leaving the samples untouched,
whereas the *pathwise* estimator differentiates the samples while leaving the
noise density untouched. Neither method dominates in every setting, and
comparing them (:numref:`subsec_mdl-stochastic-gradient-estimators`) is the
central goal of this section. First, we briefly recall how expectations are
written as integrals and establish when passing a derivative inside an integral
is valid, because that step is where subtle errors can creep in.

```{.python .input #mdl-expectation-gradients-expectations-and-their-gradients}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-expectation-gradients-expectations-and-their-gradients}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-expectation-gradients-expectations-and-their-gradients}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-expectation-gradients-expectations-and-their-gradients}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Expectations as Integrals
:label:`subsec_mdl-integration-in-probability`

Before differentiating anything, let us fix our notation for densities and
expectations. The probability chapter will develop both notions in detail (as
we will see in :numref:`sec_mdl-random_variables`); here we restate them in the
integral language of the preceding section. A probability **density** on a
domain $\mathcal X$ is a nonnegative function whose total integral equals $1$:

$$
p(x)\ge 0, \qquad \int_{\mathcal X} p(x)\,dx = 1 .
$$
:eqlabel:`eq_mdl-density`

The probability that a random variable $X$ with density $p$ lands in an
interval $[a, b]$ is the area under $p$ over that interval,
$P(a\le X\le b)=\int_a^b p(x)\,dx$ (as we will formalize in
:numref:`sec_mdl-random_variables`, :eqref:`eq_mdl-pdf_int_int`). Because a
density measures probability per unit length rather than probability itself,
its value can exceed $1$; for example, the uniform density on $[0,\tfrac12]$
equals $2$ everywhere on its support. Our main running example is the
**standard normal** distribution $\mathcal N(0,1)$, whose density
$p(z)=e^{-z^2/2}/\sqrt{2\pi}$ is normalized by the Gaussian integral of
:numref:`sec_mdl-integral_calculus`.

An **expectation** computes the weighted average of $X$, or of a function
$g(X)$, weighting each outcome by its probability density:

$$
\mathbb{E}[X] = \int_{\mathcal X} x\,p(x)\,dx,
\qquad
\mathbb{E}[g(X)] = \int_{\mathcal X} g(x)\,p(x)\,dx .
$$
:eqlabel:`eq_mdl-expectation`

The expectation $\mathbb{E}[g(X)]$ is well defined whenever the integral
converges absolutely, $\int_{\mathcal X}|g(x)|\,p(x)\,dx<\infty$. Heavy-tailed
distributions can violate this condition; the Cauchy density in exercise
:exref:`ex_mdl-integral-calculus-heavy-tails`, for instance, has no finite
mean.

In deep learning, expectations rarely have closed-form expressions, and they
often live in thousands of dimensions where numerical quadrature on a grid is
out of the question. Instead, we draw independent samples $x_1,\ldots,x_n$ from
$p$ and form the sample average $\frac1n\sum_{i=1}^n g(x_i)$, known as the
*Monte Carlo* estimate. If a single sample $g(X)$ has standard deviation
$\sigma$, the Monte Carlo average has root-mean-square error $\sigma/\sqrt n$,
regardless of the dimension (:numref:`subsec_mdl-monte-carlo`). Both gradient
estimators in this section are Monte Carlo averages of this form. Because both
are unbiased for the same true gradient, they differ only in the spread of a
single sample, making the single-sample standard deviation $\sigma$ the natural
yardstick for comparing them.

## Differentiation under the Integral Sign

### The Swap

Our toy loss $\ell(\theta)=\mathbb{E}[X^2]$ is an integral that varies with a
parameter $\theta$. In general, a parameter can affect an integral in two
places: through the limits of integration, which the fundamental theorem of
calculus :eqref:`eq_mdl-ftc` handles, or through the integrand itself.

When the limits of integration are fixed and only the integrand depends on
$\theta$, we differentiate by passing the derivative inside the integral:

$$
\frac{d}{d\theta} \int f(x, \theta)\,dx
= \int \frac{\partial f}{\partial\theta}(x, \theta)\,dx .
$$
:eqlabel:`eq_mdl-diff-under-integral`

Geometrically, the rate of change of the total area equals the integral of the
pointwise rate of change of the height. We will refer to this exchange of
derivative and integral as the *swap*; both gradient estimators depend on it.
At first glance the identity looks harmless, since a derivative is a limit of
difference quotients, an integral is a limit of sums, and finite sums and
differences always commute. Yet exchanging two limits is never automatic and
requires regularity conditions. On a bounded interval of $x$, continuity of
$\partial_\theta f$ in both variables is enough. On an unbounded domain, we
also need an integrable function of $x$ that dominates
$|\partial_\theta f(x,\theta')|$ for all $\theta'$ near $\theta$ (an immediate
consequence of the dominated convergence theorem, quoted without proof from
:cite:`Folland.1999`). For our running example $f(x,\theta)=x^2p_\theta(x)$
with a normal density, such a dominating bound always exists because the
Gaussian tail decays faster than any polynomial grows.

### The Leibniz Rule

When the limits of integration also move with $\theta$, the plain swap is
incomplete and gives the wrong answer. The Leibniz rule supplies the missing
boundary terms: the interior change and the sweeping boundaries simply add
(:numref:`fig_mdl-leibniz-boundary`).

**Proposition (Leibniz rule).** *Let $f$ and $\partial_\theta f$ be continuous
on a region containing $\{(x,\theta):x\in[a(\theta),b(\theta)]\}$ for $\theta$
in an open interval, and let $a$ and $b$ be differentiable. Then*

$$
\frac{d}{d\theta} \int_{a(\theta)}^{b(\theta)} f(x, \theta)\,dx
= \int_{a(\theta)}^{b(\theta)} \frac{\partial f}{\partial\theta}(x,\theta)\,dx
+ f\bigl(b(\theta), \theta\bigr)\,b'(\theta)
- f\bigl(a(\theta), \theta\bigr)\,a'(\theta) .
$$
:eqlabel:`eq_mdl-leibniz`

**Proof.** Define $H(\alpha,\beta,\theta)=\int_\alpha^\beta f(x,\theta)\,dx$.
By the fundamental theorem of calculus, $\partial_\beta H=f(\beta,\theta)$ and
$\partial_\alpha H=-f(\alpha,\theta)$, while the swap on the bounded interval
$[\alpha,\beta]$ gives $\partial_\theta H=\int_\alpha^\beta\partial_\theta f\,dx$.
Since all three partial derivatives are continuous, the multivariable chain
rule (:numref:`sec_mdl-chain-rule-graphs`) applies to the composition
$\theta\mapsto H(a(\theta),b(\theta),\theta)$ and combines the three terms
weighted by $a'(\theta)$, $b'(\theta)$, and $1$. $\blacksquare$

![The Leibniz rule for $f(x,\theta)=x/\theta$ on $[0,\theta]$, the uniform example. When $\theta$ increases by $d\theta$, the integrand curve shifts to $f(\cdot,\theta+d\theta)$ (dashed; here it drops, so the hatched region represents lost area), while the right boundary moves outward and sweeps in a strip of height $f(b,\theta)$ and width $b'\,d\theta$.](../img/mdl-cal-leibniz-boundary.svg)
:label:`fig_mdl-leibniz-boundary`

Ignoring the boundary term is not a minor inaccuracy; it can flip the sign of
the derivative. Suppose $X$ is uniform on $[0,\theta]$, with density $1/\theta$
on that interval. Its mean is
$\mathbb{E}[X]=\int_0^\theta\frac x\theta\,dx=\frac\theta2$, whose derivative
is $+\tfrac12$. Differentiating only the integrand gives
$\int_0^\theta-\frac{x}{\theta^2}\,dx=-\tfrac12$, which has the wrong sign.
Adding the upper boundary contribution
$f(\theta,\theta)\,b'(\theta)=\frac\theta\theta\cdot1=1$ restores the true
derivative, $-\tfrac12+1=+\tfrac12$. Whenever the support of a distribution
depends on the parameter, the plain swap alone is not valid (exercise
:exref:`ex_mdl-integral-calculus-moving-support` works out $\mathbb{E}[X^2]$).

### Log-Integral-Exp

The swap also produces the continuous counterpart of the log-sum-exp function.
Applying the chain rule and :eqref:`eq_mdl-diff-under-integral` to
$\log\int e^{\theta x}p(x)\,dx$ yields

$$
\frac{d}{d\theta}\log\int e^{\theta x}p(x)\,dx=\frac{\int x\,e^{\theta x}p(x)\,dx}{\int e^{\theta x}p(x)\,dx}.
$$

The right-hand side is the mean of $x$ under the exponentially *tilted* density
proportional to $e^{\theta x}p(x)$. For the standard normal density, completing
the square via $\theta z-\tfrac12z^2=\tfrac12\theta^2-\tfrac12(z-\theta)^2$
shows that the tilted density is $\mathcal N(\theta,1)$ and that
$\log\mathbb{E}[e^{\theta Z}]=\theta^2/2$, whose derivative $\theta$ is indeed
the tilted mean. Just as the gradient of log-sum-exp is the softmax vector, the
gradient of log-integral-exp is an expectation. We will encounter this identity
again as the derivative $A'(\eta)$ of the log-partition function in
:numref:`sec_mdl-exponential-family`.

## Score-Function and Pathwise Gradients
:label:`subsec_mdl-stochastic-gradient-estimators`

Applying the swap to an expected loss gives the integral
$\int g(x)\,\partial_\theta p_\theta(x)\,dx$. Unfortunately, we cannot estimate
this integral directly by Monte Carlo sampling: $\partial_\theta p_\theta$ is
not a probability density (it takes both positive and negative values and
integrates to zero), so we cannot draw samples from it. Two simple identities
rewrite this integral back into an expectation that samples can estimate.

### The Score-Function Estimator

Let $p_{\boldsymbol\theta}$ be a density parameterized by
$\boldsymbol\theta$. Applying the chain rule to $\log p_{\boldsymbol\theta}(x)$
yields the **log-derivative identity**:

$$
\nabla_{\boldsymbol\theta}\,p_{\boldsymbol\theta}(x)=p_{\boldsymbol\theta}(x)\,\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(x).
$$
:eqlabel:`eq_mdl-log-derivative`

In words, the gradient of a density equals the density itself multiplied by the
gradient of its logarithm, a vector known as the **score**
$\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}$. Whenever the swap is
valid and the support of $p_{\boldsymbol\theta}$ does not move with
$\boldsymbol\theta$, the score has expectation zero:
$\mathbb{E}[\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]=\int\nabla_{\boldsymbol\theta}p_{\boldsymbol\theta}\,dx=\nabla_{\boldsymbol\theta}\int p_{\boldsymbol\theta}\,dx=\nabla_{\boldsymbol\theta}1=\mathbf 0$.
The uniform density on $[0,\theta]$ shows why fixed support is essential: its
score is $\partial_\theta\log(1/\theta)=-1/\theta$, whose expectation is
$-1/\theta\ne0$, because differentiating $\int_0^\theta p_\theta(x)\,dx=1$ also
picks up the boundary term $p_\theta(\theta)=1/\theta$. The mean-zero property
of the score is also the starting point for Fisher information
(:numref:`sec_mdl-maximum_likelihood`).

Multiplying :eqref:`eq_mdl-log-derivative` by a function $g(x)$ that does not
depend on $\boldsymbol\theta$ and integrating gives the **score-function
identity**:

$$
\nabla_{\boldsymbol{\theta}}\,\mathbb{E}_{X\sim p_{\boldsymbol{\theta}}}[g(X)]
=\int g(x)\,\nabla_{\boldsymbol\theta}p_{\boldsymbol\theta}(x)\,dx
= \mathbb{E}\bigl[g(X)\,\nabla_{\boldsymbol{\theta}}\log
p_{\boldsymbol{\theta}}(X)\bigr].
$$
:eqlabel:`eq_mdl-score-function-gradient`

This identity turns the gradient of an expected loss back into an expectation
under $p_{\boldsymbol\theta}$: each sample's loss $g(x)$ is weighted by how
strongly $\boldsymbol\theta$ increases that sample's log-probability. To
estimate it by Monte Carlo, we draw samples $x_i\sim p_{\boldsymbol\theta}$ and
average $g(x_i)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(x_i)$. In
reinforcement learning, this is the REINFORCE estimator of policy-gradient
methods :cite:`Williams.1992` (as we will see in :numref:`sec_policygradient`
and :numref:`sec_baselines`). (If $g$ also depends directly on
$\boldsymbol\theta$, the product rule simply adds the term
$\mathbb{E}[\nabla_{\boldsymbol\theta}g(X,\boldsymbol\theta)]$.)

Because the score has mean zero, we may subtract any constant **baseline** $b$
(independent of $x$) from $g(X)$ without biasing the estimator
(:numref:`sec_baselines`):
$\mathbb{E}[(g(X)-b)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]=\mathbb{E}[g(X)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]$.
Choosing $b$ well can substantially reduce the variance of the estimate.

### The Pathwise Estimator

The second identity restores the sample's memory of $\boldsymbol\theta$. Often,
a sample from $p_{\boldsymbol\theta}$ can be generated by applying a
deterministic transformation $x=T_{\boldsymbol\theta}(\mathbf z)$ to an
auxiliary noise variable $\mathbf z\sim q$ whose distribution does not depend
on $\boldsymbol\theta$. A normal random variable $\mathcal N(\mu,\sigma^2)$,
for instance, is simply $\mu+\sigma z$ with $z\sim\mathcal N(0,1)$. Rewriting
the expectation as an integral over $\mathbf z$ removes $\boldsymbol\theta$
from the density entirely, so the swap yields the **pathwise identity**:

$$
\nabla_{\boldsymbol\theta}\,
\mathbb{E}_{X\sim p_{\boldsymbol\theta}}[g(X)]
=\nabla_{\boldsymbol\theta}\int g\bigl(T_{\boldsymbol\theta}(\mathbf z)\bigr)\,q(\mathbf z)\,d\mathbf z
=\mathbb{E}_{\mathbf z\sim q}
\Bigl[\nabla_{\boldsymbol\theta}\,
 g\bigl(T_{\boldsymbol\theta}(\mathbf z)\bigr)\Bigr].
$$
:eqlabel:`eq_mdl-pathwise-gradient`

Instead of differentiating the density, this estimator differentiates right
through the sample: automatic differentiation traces the path from
$\boldsymbol\theta$ through $T_{\boldsymbol\theta}$ into $g$. Known as the
*reparameterization trick*, this is how variational autoencoders backpropagate
through Gaussian latent variables :cite:`Kingma.Welling.2014`. Valid use
requires $T_{\boldsymbol\theta}$ to be differentiable in $\boldsymbol\theta$,
$g$ to be differentiable along the path (a Lipschitz $g$ suffices when $X$ has
a continuous density), and an integrable dominating bound for the swap. Notice
that no boundary term ever appears, even if the support of $X$ moves with
$\boldsymbol\theta$, because the integral is taken over $\mathbf z$ whose
density $q$ stays fixed.

The differentiability requirement on $g$ is not a technicality we can ignore.
Consider the step function $g(x)=\mathbf{1}(x>0)$, which equals $1$ for
positive $x$ and $0$ otherwise, with $X=\theta+z$. The true expectation
$\mathbb{E}[g(X)]=P(z>-\theta)$ has derivative
$\varphi(\theta)=e^{-\theta^2/2}/\sqrt{2\pi}$, which evaluates to about $0.13$
at $\theta=1.5$. Yet the pathwise sample $\partial_\theta\mathbf{1}(\theta+z>0)$
is $0$ for every $z$ except the single point $z=-\theta$ (an event of
probability zero), so the pathwise estimator returns $0$ with probability $1$.
By contrast, the score-function sample $\mathbf{1}(x>0)\,z$ has expectation
$\int_{-\theta}^\infty z\,\varphi(z)\,dz=\bigl[-\varphi(z)\bigr]_{-\theta}^\infty=\varphi(\theta)$,
recovering the exact derivative. Because the score-function estimator never
differentiates through $g$ or the sampling step, it applies equally well to
discontinuous losses and to discrete distributions such as Bernoulli or
categorical choices, where no differentiable path from $\boldsymbol\theta$ to
the sample exists.

### The Two Estimators Compared

We now have two unbiased estimators of the same derivative, and our running
example $\ell(\theta)=\mathbb{E}[X^2]$ with $X=\theta+z$ is simple enough to
compute the variance of both by hand. For the pathwise estimator, a single
sample contributes $\partial_\theta(\theta+z)^2 = 2X$, whose variance is
$\textrm{Var}(2X)=4$. For the score-function estimator, the score of the normal
location family is $\partial_\theta\log p_\theta(X)=X-\theta=z$, so one sample
contributes $X^2z$, or $(X^2-b)z$ when using a baseline $b$. To find the
variance of $X^2z$, we use the standard normal moments $\mathbb{E}[z^2]=1$,
$\mathbb{E}[z^4]=3$, and $\mathbb{E}[z^6]=15$, all of which follow from Stein's
identity $\mathbb{E}[g'(Z)]=\mathbb{E}[Z\,g(Z)]$
(:numref:`subsec_mdl-integration-by-parts`). By symmetry, all odd powers of $z$
vanish in the second moment:

$$
\mathbb{E}\bigl[(\theta+z)^4z^2\bigr]=\theta^4\,\mathbb{E}[z^2]+6\theta^2\,\mathbb{E}[z^4]+\mathbb{E}[z^6]=\theta^4+18\theta^2+15.
$$

Subtracting the squared mean $(2\theta)^2$ leaves a variance of
$\theta^4+14\theta^2+15$. The variance-minimizing constant baseline turns out
to be $b^\star=\theta^2+3$, reducing the variance to $8\theta^2+6$ (exercise
:exref:`ex_mdl-integral-calculus-quartic-estimators`). At $\theta=1.5$, the
three single-sample standard deviations are $2$ for the pathwise estimator,
$7.18$ for the plain score-function estimator, and $4.90$ for the
score-function estimator with the optimal baseline. Although the baseline
helps noticeably, it cannot close the gap for this loss, since $8\theta^2+6>4$
for every $\theta$.

Let us test all three closed-form predictions in a single Monte Carlo
simulation. Because the pathwise derivative is simply $2x_i$, we do not even
need automatic differentiation here. If our algebra is right, the three sample
standard deviations should land near $2$, $7.18$, and $4.90$, and each sample
mean should fall within a few standard errors $\mathrm{sd}/\sqrt n$ of the
true derivative $3$.

```{.python .input #mdl-integral-calculus-score-function-and-pathwise-gradients-1}
theta, n = 1.5, 200_000
z = np.random.default_rng().standard_normal(n)     # noise, independent of theta
x = theta + z                                      # samples of N(theta, 1)
pathwise = 2 * x                                   # d/dtheta (theta + z_i)^2
score = x**2 * z                                   # g(x) times the score x - theta
baseline = (x**2 - (theta**2 + 3)) * z             # with the optimal baseline
exact_sd = {'pathwise': 2, 'score': math.sqrt(theta**4 + 14 * theta**2 + 15),
            'score+baseline': math.sqrt(8 * theta**2 + 6)}
print(f'true derivative 2*theta = {2 * theta}')
for name, s in zip(exact_sd, [pathwise, score, baseline]):
    print(f'{name:15s} mean={s.mean():.4f}  sd={s.std():.3f} '
          f'(exact {exact_sd[name]:.3f})  se={s.std() / math.sqrt(n):.4f}')
```

In typical runs, all three means lie within a couple of standard errors of $3$,
and the empirical standard deviations match the closed-form predictions to
within a few percent. For a fixed sample budget, the pathwise estimate is about
$3.6$ times more accurate than the plain score-function estimate; equivalently,
the plain score-function estimator needs roughly thirteen times as many samples
to reach the same precision.

It would be a mistake, however, to conclude that the pathwise estimator always
has lower variance. Consider an oscillating loss $g(x)=\sin(\omega x)$, for
which the true derivative of the expectation is
$\omega\cos(\omega\theta)e^{-\omega^2/2}$. The pathwise sample
$\omega\cos(\omega X)$ scales with the frequency $\omega$ and has standard
deviation close to $\omega/\sqrt2$, whereas the score-function sample
$\sin(\omega X)\,z$ has standard deviation at most
$\sqrt{\mathbb{E}[z^2]}=1$ because $|\sin|\le1$. Whenever an integrand
oscillates rapidly, its derivative can be huge even while its values remain
bounded. Let us compare the two estimators at $\omega=1$ and $\omega=3$,
reusing the samples already drawn.

```{.python .input #integral-score-wins}
for omega in [1, 3]:
    true = omega * math.cos(omega * theta) * math.exp(-omega**2 / 2)
    path, score = omega * np.cos(omega * x), np.sin(omega * x) * z
    print(f'omega={omega}  true={true:+.4f}  pathwise {path.mean():+.4f} '
          f'(sd {path.std():.2f})  score {score.mean():+.4f} (sd {score.std():.2f})')
```

Even at $\omega=1$, the score-function estimator already has the smaller spread
(about $0.5$ versus $0.7$), and at $\omega=3$ it wins by a factor of three
(about $0.7$ versus $2.1$). Both sample means agree with the exact derivatives
within a few standard errors.

Neither estimator dominates in every problem. The pathwise estimator
differentiates $g$, so its variance grows whenever $g'$ is large; the
score-function estimator multiplies $g$ by the score, so its variance grows
whenever $g$ itself is large. In both cases, the single-sample standard
deviation $\sigma$ governs how many samples are needed for a given accuracy.
Because $\sigma$ also depends on the distribution from which we draw samples,
choosing a proposal distribution that makes $\sigma$ small leads directly to
importance sampling, which we take up in :numref:`sec_mdl-bayesian-computation`.

## Summary

When a model's parameters govern the distribution of its samples, the gradient
of an expected loss must be rewritten as an expectation before Monte Carlo
sampling can estimate it. Both ways of doing so pass the derivative inside the
integral, a step that requires an integrable dominating bound and, whenever the
limits of integration move with the parameter, the boundary terms of the
Leibniz rule. The score-function identity differentiates the log-density; it
works for arbitrary losses $g$ and discrete random variables, though its
samples scale with the magnitude of $g$ and can be noisy without a good
baseline. The pathwise identity differentiates through a reparameterized
sample; it requires a differentiable $g$ and often, though not always, achieves
much lower variance. Because both estimators have root-mean-square error
$\sigma/\sqrt n$ (:numref:`subsec_mdl-monte-carlo`), the single-sample standard
deviation $\sigma$ decides which one is preferable for a given problem.

## Exercises

1. **A moving support.** Let $X$ be uniform on $[0,\theta]$ with
   $\theta>0$, so that
   $\ell(\theta)=\mathbb{E}[X^2]=\int_0^\theta \tfrac{x^2}{\theta}\,dx=\tfrac{\theta^2}{3}$
   and $\ell'(\theta)=\tfrac{2\theta}{3}$.
    1. Differentiate $\ell$ with the Leibniz rule :eqref:`eq_mdl-leibniz`.
       Which part of the result does the plain swap
       :eqref:`eq_mdl-diff-under-integral` produce, and which part does it
       miss?
    1. On the support, $\log p_\theta(x)=-\log\theta$. Compute the
       expectation of the score-function estimator
       $X^2\,\partial_\theta\log p_\theta(X)$
       of :eqref:`eq_mdl-score-function-gradient`. Then let $g$ be continuous
       on $[0,\infty)$. Show that the mean of
       $g(X)\,\partial_\theta\log p_\theta(X)$ equals the plain swap
       :eqref:`eq_mdl-diff-under-integral` applied to
       $\mathbb{E}[g(X)]=\int_0^\theta g(x)\,p_\theta(x)\,dx$. What is its
       bias as an estimate of $\frac{d}{d\theta}\mathbb{E}[g(X)]$, in terms
       of $g(\theta)$ and $\theta$?
    1. Write $X=\theta U$ with $U$ uniform on $[0,1]$ and compute the
       expectation of the pathwise estimator
       of :eqref:`eq_mdl-pathwise-gradient`. Why does it need no boundary term,
       although the support of $X$ moves with $\theta$?
   :exlabel:`ex_mdl-integral-calculus-moving-support`

1. [code] **Estimator variance for a quartic.** ● Let
   $X\sim\mathcal N(\theta,1)$, write $X=\theta+z$, and let
   $s(X)=\partial_\theta\log p_\theta(X)=z$ be the score.
    1. Take $g(X)=X^4$, so that
       $\frac{d}{d\theta}\mathbb{E}[X^4]=4\theta^3+12\theta$. Derive the
       single-sample pathwise estimator from
       :eqref:`eq_mdl-pathwise-gradient` and the single-sample score-function
       estimator from :eqref:`eq_mdl-score-function-gradient`. Use the
       identity of exercise
       :exref:`ex_mdl-integral-calculus-gaussian-normalizers` (b) to show
       that both have mean $4\theta^3+12\theta$.
    1. For a constant baseline $b$, show that the variance of
       $(g(X)-b)\,s(X)$ is minimized by
       $b^\star=\mathbb{E}[g(X)s(X)^2]/\mathbb{E}[s(X)^2]$. For $g(X)=X^2$ show
       that $b^\star=\theta^2+3$, and use $\mathbb{E}[z^2]=1$,
       $\mathbb{E}[z^4]=3$, $\mathbb{E}[z^6]=15$ to derive the three variances
       $4$, $\theta^4+14\theta^2+15$ and $8\theta^2+6$ that the text quotes.
    1. Evaluate $b^\star$ for $g(X)=X^4$. With $10^6$ samples at
       $\theta\in\{0,1,2\}$, estimate the standard deviations of the
       pathwise estimator, the score-function estimator, and the
       score-function estimator with the baseline. For which $\theta$ is the
       baseline's gain largest, and does it ever close the gap to the
       pathwise estimator? Explain the ordering from the sizes of $g'$ and
       of $g$.
    1. Take $g(x)=\mathbf{1}(x>0)$ at $\theta=1.5$. The text shows that the
       pathwise estimator is $0$ for almost every sample, while the
       score-function estimator $\mathbf{1}(x>0)\,z$ has the correct mean
       $\varphi(\theta)=0.1295$. Verify both by simulation.
   :exlabel:`ex_mdl-integral-calculus-quartic-estimators`

1. [extended] **Gradients through a coin flip.** ●● When $x$ is a discrete
   choice there is no differentiable path from $\theta$ to the sample, so
   the pathwise estimator does not exist. Continuous relaxations such as
   Gumbel-softmax :cite:`Jang.Gu.Poole.2017,Maddison.Mnih.Teh.2017` create a
   differentiable surrogate, and straight-through estimators
   :cite:`Bengio.Leonard.Courville.2013` use the hard sample in the forward
   pass and the surrogate in the backward pass; both are biased for the
   original objective. This problem measures that bias.
   :citet:`Tucker.Mnih.Maddison.ea.2017` illustrate it on a toy problem. A
   single Bernoulli variable $b\in\{0,1\}$ has $P(b=1)=\sigma(\theta)$ for a
   logit $\theta$, and the objective is $\ell(\theta)=\mathbb{E}[(b-t)^2]$ with
   target $t=0.45$. Let $U$ be uniform on $(0,1)$, let
   $\zeta=\log U-\log(1-U)$, and, for a temperature $\tau>0$, let
   $z=\sigma\bigl((\theta+\zeta)/\tau\bigr)$, the two-class case of the
   Gumbel-softmax relaxation.
    1. Compute $\ell(\theta)$ and $\ell'(\theta)$ in closed form. Which probability
       $\sigma(\theta)$ minimizes $\ell$, and does a finite $\theta$ attain it?
    1. Show that $b=\mathbf{1}(z>\tfrac12)$ satisfies $P(b=1)=\sigma(\theta)$
       for every $\tau$.
    1. Implement three single-sample estimators of $\ell'(\theta)$: REINFORCE
       :eqref:`eq_mdl-score-function-gradient` with the baseline
       $\ell(\theta)$; the relaxed estimator $\frac{d}{d\theta}(z-t)^2$; and the
       straight-through estimator $2(b-t)\frac{dz}{d\theta}$, which uses the
       hard sample in the loss and the relaxed sample in the derivative.
       Tabulate the mean and the standard deviation of each estimator
       against $\ell'(\theta)$ for $\theta\in\{-2,0,2\}$ and
       $\tau\in\{1,0.5,0.1\}$. How does the bias of each estimator depend on
       $\theta$ and $\tau$?
    1. Run stochastic gradient descent on $\theta$ from $\theta=0$ with step
       size $1$ for $5000$ steps with REINFORCE and with the relaxed and
       straight-through estimators at each $\tau$, replacing the REINFORCE
       baseline by an exponential moving average of past losses.
       Plot $\ell(\theta)$ along each run. Compute the relaxed objective
       $R_\tau(\theta)=\mathbb{E}[(z-t)^2]$ by quadrature over $U$, and compare
       its minimizer with the values of $\theta$ at which the relaxed runs
       settle. How does lowering $\tau$ trade the bias of the relaxed
       estimator against its variance?
   :exlabel:`ex_mdl-integral-calculus-coin-flip`

[Discussions](https://d2l.discourse.group/t/36878)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §34.7]{.kicker}

Expectations and Their Gradients<br>**differentiation under the integral sign · Leibniz rule · score-function (REINFORCE) · pathwise reparameterization · variance trade-offs**
:::
:::

::: {.slide title="When Parameters Govern the Sampling Distribution"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Where Parameterized Sampling Arises:** VAEs sample latent codes $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}, \operatorname{diag}(\boldsymbol{\sigma}_{\boldsymbol{\theta}}^2))$ and RL policies sample actions $a \sim \pi_{\boldsymbol{\theta}}(a \mid s)$.
- **The Graph-Severing Obstacle:** A raw sample $x_i \sim p_{\boldsymbol{\theta}}$ is just a number with no computation graph linking it to $\boldsymbol{\theta}$.
- **Running Benchmark:** For $X \sim \mathcal{N}(\theta, 1)$ and $\ell(\theta) = \mathbb{E}[X^2] = \theta^2 + 1$, true slope at $\theta = 1.5$ is $\ell'(1.5) = 2\theta = 3$.
:::

::: {.col .fig .big}
@fig:ch34-calc-param-dependent-sampling
:::
:::
:::

::: {.slide title="Expectations as Integrals and Monte Carlo Error"}
[Foundations]{.kicker}

::: {.cols .vc}
::: {.col}
- **Expectation as Integral:** $\mathbb{E}_{X \sim p}[g(X)] = \int_{\mathcal{X}} g(x)p(x)\,dx$, approximated by $\hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n g(x_i)$.
- **Dimension-Free RMSE ($\sigma/\sqrt{n}$):** Root-mean-square error scales as $\sigma/\sqrt{n}$ regardless of dimension $d$.
- **The Yardstick:** Sample efficiency of any unbiased gradient estimator is governed by its single-sample standard deviation $\sigma$.
:::

::: {.col .fig .big}
@fig:ch34-calc-mc-rmse-scaling
:::
:::
:::

::: {.slide title="Differentiating Under the Integral Sign: The Swap"}
[Exchanging Limits]{.kicker}

::: {.cols .vc}
::: {.col}
- **The Interior Swap:** On fixed support $\mathcal{X}$, $\frac{d}{d\theta}\int_{\mathcal{X}} f(x, \theta)\,dx = \int_{\mathcal{X}} \frac{\partial f}{\partial\theta}(x, \theta)\,dx$.
- **Regularity Conditions:** Joint continuity of $\partial_\theta f$ on bounded $[a, b]$, plus an integrable dominating bound $|\partial_\theta f(x, \theta')| \le M(x)$ on unbounded $\mathbb{R}$ (satisfied by Gaussian tails).
:::

::: {.col .fig .big}
@fig:ch34-calc-integral-swap
:::
:::
:::

::: {.slide title="Log-Integral-Exp and Exponential Family Tilting"}
[Continuous Softmax]{.kicker}

::: {.cols .vc}
::: {.col}
- **Log-Partition Derivative:** Differentiating $\log\int e^{\theta x}p(x)\,dx$ via the swap yields the tilted expectation $\mathbb{E}_{X \sim \tilde{p}_\theta}[X]$ under $\tilde{p}_\theta(x) \propto e^{\theta x}p(x)$.
- **Standard Normal Check:** For $Z \sim \mathcal{N}(0, 1)$, $\log\mathbb{E}[e^{\theta Z}] = \tfrac{1}{2}\theta^2$ has derivative $\theta$, matching the mean of $\tilde{p}_\theta = \mathcal{N}(\theta, 1)$.
:::

::: {.col .fig .big}
@fig:ch34-calc-exponential-tilting
:::
:::
:::

::: {.slide title="The Leibniz Rule: Moving Integration Boundaries"}
[Moving Support]{.kicker}

$$\frac{d}{d\theta}\int_{a(\theta)}^{b(\theta)} f(x, \theta)\,dx = \int_{a(\theta)}^{b(\theta)} \frac{\partial f}{\partial \theta}(x, \theta)\,dx + f\bigl(b(\theta), \theta\bigr)b'(\theta) - f\bigl(a(\theta), \theta\bigr)a'(\theta)$$

::: {.cols .vc}
::: {.col}
- **Interior shift + sweeping boundaries:** When $a(\theta)$ and $b(\theta)$ move with $\theta$, the total area changes both because the integrand roof shifts vertically ($\int_a^b \partial_\theta f\,dx$) and because the endpoints sweep horizontally ($f(b,\theta)b' - f(a,\theta)a'$).
- **Proof via multivariable chain rule (§{sec_mdl-chain-rule-graphs}):** Set $H(\alpha, \beta, \theta) = \int_\alpha^\beta f(x, \theta)\,dx$. By FTC, $\partial_\beta H = f(\beta, \theta)$ and $\partial_\alpha H = -f(\alpha, \theta)$, while the bounded swap gives $\partial_\theta H = \int_\alpha^\beta \partial_\theta f\,dx$. Evaluating $\frac{d}{d\theta}H(a(\theta), b(\theta), \theta)$ sums all three path products.
:::

::: {.col .fig}
![](../img/mdl-cal-leibniz-boundary.svg){width=100%}
:::
:::
:::

::: {.slide title="Why Ignoring Boundary Terms Flips the Sign"}
[Moving-Support Pitfall]{.kicker}

::: {.cols .vc}
::: {.col}
- **Uniform$(0, \theta)$ Mean:** $\mathbb{E}[X] = \int_0^\theta \frac{x}{\theta}\,dx = \frac{\theta}{2}$ has true derivative $+\frac{1}{2}$.
- **Naive Swap Gives Wrong Sign:** Differentiating only inside gives $\int_0^\theta -\frac{x}{\theta^2}\,dx = -\frac{1}{2}$.
- **Leibniz Boundary Term:** Adding the upper wall sweep $f(\theta, \theta)b'(\theta) = 1 \cdot 1 = +1$ restores $-\frac{1}{2} + 1 = +\frac{1}{2}$.
:::

::: {.col .fig .big}
@fig:ch34-calc-moving-support-sign
:::
:::
:::

::: {.slide title="Why Raw Density Derivatives Cannot Be Sampled"}
[The Monte Carlo Obstacle]{.kicker}

::: {.cols .vc}
::: {.col}
- **$\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}(x)$ Is Not a Density:** In $\int g(x)\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}(x)\,dx$, $\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}$ takes negative values and integrates to $\mathbf{0}$.
- **Two Escape Routes Back to an Expectation:**
  1. *Score function:* Multiply and divide by $p_{\boldsymbol{\theta}}(x)$ ($\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}$).
  2. *Pathwise:* Change variables to $X = T_{\boldsymbol{\theta}}(\mathbf{z})$ with $\mathbf{z} \sim q(\mathbf{z})$.
:::

::: {.col .fig .big}
@fig:ch34-calc-two-escape-routes
:::
:::
:::

::: {.slide title="The Log-Derivative Trick and the Score Function"}
[Score-Function Identity]{.kicker}

::: {.cols .vc}
::: {.col}
- **REINFORCE Identity:** Using $\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}} = p_{\boldsymbol{\theta}}\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}$ gives $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)]$.
- **Zero-Mean Score:** On fixed support, $\mathbb{E}[\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)] = \mathbf{0}$ (for $\mathcal{N}(\theta, 1)$, $s_\theta(X) = X - \theta = Z$).
- **Generality:** Never differentiates $g(x)$, so it works for discrete actions and black-box rewards.
:::

::: {.col .fig .big}
@fig:ch34-calc-score-log-deriv
:::
:::
:::

::: {.slide title="Variance Reduction via Control-Variate Baselines"}
[Baseline Subtraction]{.kicker}

::: {.cols .vc}
::: {.col}
- **Unbiased Baseline Subtraction:** Since $\mathbb{E}[s_\theta(X)] = 0$, replacing $g(X)$ by $g(X) - b$ leaves the mean unchanged for any constant $b$.
- **Optimal Constant Baseline:** $b^\star = \frac{\mathbb{E}[g(X)s_\theta(X)^2]}{\mathbb{E}[s_\theta(X)^2]}$ ($= \theta^2 + 3$ for $g(X) = X^2$), cutting variance from $O(\theta^4)$ to $8\theta^2 + 6 = O(\theta^2)$.
:::

::: {.col .fig .big}
@fig:ch34-calc-baseline-variance
:::
:::
:::

::: {.slide title="The Pathwise (Reparameterization) Estimator"}
[Differentiating Through Samples]{.kicker}

::: {.cols .vc}
::: {.col}
- **Reparameterization ($X = T_{\boldsymbol{\theta}}(\mathbf{z})$, $\mathbf{z} \sim q$):** Changing variables to parameter-free noise $\mathbf{z}$ gives $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{\mathbf{z} \sim q}[\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))]$.
- **Direct Chain Rule Through $g$:** For $\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\mathbf{z}$, backprop differentiates $g(\boldsymbol{\mu} + \boldsymbol{\sigma}\odot\mathbf{z})$ directly with low variance.
:::

::: {.col .fig .big}
@fig:ch34-calc-pathwise-reparam
:::
:::
:::

::: {.slide title="When Pathwise Fails: Step Losses & Discrete Choices"}
[Differentiability Boundary]{.kicker}

::: {.cols .vc}
::: {.col}
- **Step Loss $g(x) = \mathbf{1}(x > 0)$ at $\theta = 1.5$:** True derivative of $\Phi(\theta)$ is $\varphi(1.5) \approx 0.1295$.
- **Pathwise Returns $0$ Almost Surely:** The step $\theta \mapsto \mathbf{1}(\theta + z > 0)$ is flat everywhere except the zero-probability jump $z = -\theta$.
- **Score Function Succeeds:** $\mathbb{E}[\mathbf{1}(\theta + Z > 0)Z] = \varphi(\theta) \approx 0.1295$.
:::

::: {.col .fig .big}
@fig:ch34-calc-step-loss-failure
:::
:::
:::

::: {.slide title="Analytical Variance Comparison on E[X²]"}
[Exact Moment Calculation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Exact Variances on $\ell(\theta) = \mathbb{E}[X^2]$ at $\theta = 1.5$:**
  - *Pathwise ($2X$):* $\operatorname{Var} = 4 \implies \sigma = 2.000$.
  - *Score + Baseline ($b^\star = \theta^2+3$):* $\operatorname{Var} = 8\theta^2+6 \implies \sigma = 4.899$.
  - *Plain REINFORCE ($X^2 Z$):* $\operatorname{Var} = \theta^4+14\theta^2+15 \implies \sigma = 7.181$ ($12.9\times$ higher variance!).
:::

::: {.col .fig .big}
@fig:ch34-calc-estimator-variance-bars
:::
:::
:::

::: {.slide title="Monte Carlo Verification: Pathwise vs. REINFORCE"}
[Empirical Benchmark]{.kicker}

Drawing $n = 200{,}000$ samples at $\theta = 1.5$ confirms that all three estimators are unbiased for $\ell'(1.5) = 3.0000$ and reproduces the exact theoretical standard deviations $2.000$, $7.181$, and $4.899$:

@!mdl-integral-calculus-score-function-and-pathwise-gradients-1
:::

::: {.slide title="When Score Beats Pathwise: Rapidly Oscillating Losses"}
[High-Frequency Regime]{.kicker}

Does the pathwise estimator always have lower variance than the score-function estimator? **No!** Consider the oscillating loss $g(x) = \sin(\omega x)$, where $\frac{d}{d\theta}\mathbb{E}[\sin(\omega X)] = \omega\cos(\omega\theta)e^{-\omega^2/2}$:

- **Pathwise variance scales with $|g'|^2 = O(\omega^2)$:** One pathwise sample is $\omega\cos(\omega X)$, whose standard deviation grows linearly with frequency as $\approx \omega/\sqrt{2}$.
- **Score-function variance is bounded by $|g|^2 \le 1$:** One score sample is $\sin(\omega X)\,Z$; since $|\sin(\omega X)| \le 1$, its second moment is at most $\mathbb{E}[Z^2] = 1$ regardless of how large $\omega$ grows!

@!integral-score-wins
:::

::: {.slide title="Decision Guide: Choosing a Gradient Estimator"}
[Design Trade-Offs]{.kicker}

| Property / Requirement | Score-Function (REINFORCE) | Pathwise (Reparameterization) | Relaxed / Straight-Through (e.g. Gumbel-Softmax) |
| :--- | :--- | :--- | :--- |
| **Core Formula** | $g(X)\,\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)$ | $\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))$ $= \mathbf{J}_{T}^\top \nabla_x g$ | Differentiate smooth surrogate $z_\tau = \sigma((\theta+\zeta)/\tau)$ |
| **Requires Differentiable $g$?** | **No** (works on steps & black-box rewards) | **Yes** ($g$ must be differentiable/Lipschitz) | **Yes** (surrogate loss must be differentiable) |
| **Handles Discrete $X$?** | **Yes** (Bernoulli, categorical actions) | **No** (no continuous path $T_{\boldsymbol{\theta}}(\mathbf{z})$) | **Yes** (approximates discrete choice at temperature $\tau$) |
| **Bias** | **Unbiased** (on fixed support) | **Unbiased** (for smooth $g$) | **Biased** (bias shrinks as $\tau \to 0$, variance blows up) |
| **Single-Sample Variance** | Scales with $\|g(X)\|^2$ (needs baseline $b$) | Scales with $\|\nabla_x g(X)\|^2$ (low for smooth $g$) | Controlled by temperature $\tau$ (bias–variance trade-off) |
:::

::: {.slide title="Concept Check: Differentiating Expectations"}
[Review & Practice]{.kicker}

Work out these three questions on the Leibniz rule, score function, and pathwise estimators:

1. **Moving support bias:** For $X \sim \mathrm{Uniform}(0, \theta)$ with density $p_\theta(x) = 1/\theta$ on $[0, \theta]$, the score is $\partial_\theta \log(1/\theta) = -1/\theta$. What does the score-function estimator $X^2 \partial_\theta \log p_\theta(X)$ give in expectation for $\ell(\theta) = \mathbb{E}[X^2] = \theta^2/3$, and why does it miss $\ell'(\theta) = 2\theta/3$?
2. **Reparameterizing Uniform$(0, \theta)$:** How can you reparameterize $X \sim \mathrm{Uniform}(0, \theta)$ using $U \sim \mathrm{Uniform}(0, 1)$ so the pathwise estimator gives the exact unbiased derivative $\ell'(\theta) = 2\theta/3$ without a boundary term?
3. **Variance comparison in high dimensions:** If $g(\mathbf{x}) = \|\mathbf{x}\|_2^2$ for $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\theta}, \mathbf{I}_d)$, how do the magnitudes of $g(\mathbf{X})$ (entering REINFORCE) and $\nabla_{\mathbf{x}} g(\mathbf{X}) = 2\mathbf{X}$ (entering pathwise) scale with dimension $d$?
:::

::: {.slide title="Concept Check: Differentiating Expectations (cont.)"}
[Review & Practice]{.kicker}

::: {.d2l-note}
**Answers:**

1. $\mathbb{E}[X^2(-1/\theta)] = -\frac{1}{\theta}\frac{\theta^2}{3} = -\frac{\theta}{3}$—which equals the naive interior swap $\int_0^\theta x^2 \partial_\theta(1/\theta)\,dx$ and misses the Leibniz upper-boundary term $f(\theta, \theta)b'(\theta) = \theta^2(1/\theta)(1) = +\theta$ ($-\frac{\theta}{3} + \theta = +\frac{2\theta}{3}$).
2. Write $X = T_\theta(U) = \theta U$ on fixed support $U \in [0, 1]$: the pathwise estimator is $\partial_\theta(\theta U)^2 = 2\theta U^2$, whose expectation is $2\theta\mathbb{E}[U^2] = 2\theta(\tfrac{1}{3}) = \frac{2\theta}{3}$!
3. The scalar loss $g(\mathbf{X}) = \sum_{j=1}^d X_j^2 = O(d)$ multiplies all $d$ coordinates of the score vector, making REINFORCE variance grow as $O(d^2)$ (or $O(d)$ with a baseline), whereas the pathwise derivative of coordinate $k$ is simply $2X_k$ with $O(1)$ variance independent of $d$!
:::
:::

::: {.slide title="Summary: Expectation Gradients Toolkit"}
[Key Takeaways]{.kicker}

| Rule / Estimator | Mathematical Identity | Conditions & Variance Behavior |
| :--- | :--- | :--- |
| **Interior Swap** | $\frac{d}{d\theta}\int f(x,\theta)\,dx = \int \frac{\partial f}{\partial \theta}(x,\theta)\,dx$ | Fixed limits; continuous $\partial_\theta f$ + integrable dominating bound |
| **Leibniz Boundary Rule** | $\frac{d}{d\theta}\int_{a(\theta)}^{b(\theta)} f\,dx = \int_a^b \partial_\theta f\,dx + f(b,\theta)b' - f(a,\theta)a'$ | Required whenever integration limits / support move with $\theta$ |
| **Log-Integral-Exp** | $\frac{d}{d\theta}\log\int e^{\theta x}p(x)\,dx = \mathbb{E}_{\tilde{p}_\theta}[X]$ ($\tilde{p}_\theta \propto e^{\theta x}p$) | Continuous counterpart of $\nabla\operatorname{LSE}(\mathbf{z}) = \operatorname{softmax}(\mathbf{z})$ |
| **Score Function (REINFORCE)** | $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{p_{\boldsymbol{\theta}}}\bigl[(g(X)-b)\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)\bigr]$ | Works for discontinuous $g$ & discrete $X$; variance scales with $\|g-b\|^2$ |
| **Optimal Baseline $b^\star$** | $b^\star = \mathbb{E}[g(X)s_\theta(X)^2] / \mathbb{E}[s_\theta(X)^2]$ | Unbiased because $\mathbb{E}[\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)] = \mathbf{0}$ on fixed support |
| **Pathwise (Reparameterized)** | $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{\mathbf{z}\sim q}\bigl[\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))\bigr]$ | Requires smooth $g$ & continuous $X = T_{\boldsymbol{\theta}}(\mathbf{z})$; variance scales with $\|\nabla g\|^2$ |
:::

