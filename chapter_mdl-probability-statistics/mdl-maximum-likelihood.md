# Maximum Likelihood
:label:`sec_mdl-maximum_likelihood`

Every loss function in this book is a disguised probability statement. When we
"minimize the cross-entropy" of a classifier or "minimize the squared error" of
a regressor, we are answering one question: *which parameters make the observed
data most probable?* That is the **maximum likelihood** principle, and it is the
single idea that turns a probabilistic model into a trainable objective. This
section states the principle, proves the three equivalences that make it
operational---likelihood becomes a *negative log-likelihood*, the negative
log-likelihood *is* the cross-entropy to the data, and the Gaussian case *is*
mean squared error---and then shows how putting a prior back in recovers
$L_2$ regularization. We keep the running coin-flip example throughout because it
is small enough to solve by hand, which lets us check every claim against the
answer we already know.

We load the per-framework library so the worked cells below have `d2l` in
scope, plus plain NumPy (as `onp`) for the framework-agnostic simulations. Only
the two small gradient-descent cells branch per framework; everything else is
framework-agnostic.

```{.python .input #maximum-likelihood-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import numpy as onp
npx.set_np()
```

```{.python .input #maximum-likelihood-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as onp
import torch
```

```{.python .input #maximum-likelihood-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as onp
import tensorflow as tf
```

```{.python .input #maximum-likelihood-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as onp
```

## The Maximum Likelihood Principle

Suppose we have a model with parameters $\boldsymbol{\theta}$ and a collection of
observed data $X$. As a concrete picture, let $\boldsymbol{\theta}$ be the single
number giving the probability a coin lands heads, and let $X$ be a sequence of
independent flips. We want the *most probable* parameters given the data, i.e.,

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}}\, P(\boldsymbol{\theta}\mid X).
$$
:eqlabel:`eq_mdl-max_like`

This looks like it requires a prior, and Bayes' rule makes the dependence
explicit:

$$
P(\boldsymbol{\theta}\mid X) = \frac{P(X \mid \boldsymbol{\theta})\,P(\boldsymbol{\theta})}{P(X)}.
$$

Two of the three pieces drop out of the $\mathop{\mathrm{argmax}}$, but for
*different reasons*. The evidence $P(X)$ does not depend on $\boldsymbol{\theta}$,
so it can never change which $\boldsymbol{\theta}$ wins---dropping it is free, no
assumption required. The prior $P(\boldsymbol{\theta})$ is a different matter:
discarding it is a *modelling choice*. Only if we hold *no* prior belief that one
parameter value is better than another---an *uninformative prior*, e.g. that the
coin's bias is equally likely to be any value in $[0,1]$---is
$P(\boldsymbol{\theta})$ constant in $\boldsymbol{\theta}$ and free to drop too.
What remains is the **likelihood** $P(X \mid \boldsymbol{\theta})$, the
probability the data assigns under each parameter, and the principle in its
working form:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}}\, P(X \mid \boldsymbol{\theta}).
$$
:eqlabel:`eq_mdl-mle`

We restore the prior $P(\boldsymbol{\theta})$ deliberately in
:numref:`subsec_mdl-map`; everything until then assumes it is uninformative.

### A Worked Example: The Coin

Take a single parameter $\theta$, the probability of heads, so tails has
probability $1-\theta$. Because the flips are independent, their probabilities
multiply: if $X$ has $n_H$ heads and $n_T$ tails,

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$
:eqlabel:`eq_mdl-coin-like`

Flip $13$ coins and observe the sequence "HHHTHTTHHHHHT", which has $n_H=9$ and
$n_T=4$, giving the likelihood $P(X \mid \theta) = \theta^9(1-\theta)^4$. We know
the answer we *want*: asked "$9$ of $13$ flips were heads, what is the bias?",
everyone replies $9/13$. The value of the maximum-likelihood machinery is that it
*derives* that number from first principles, in a way that scales to models with
billions of parameters where no intuition is available.

The likelihood is a function of $\theta$ alone, so we can plot it and read off
its peak. (This is a *computed* curve that teaches---the likelihood of the data
as the parameter varies---so it lives in code, not as a pre-drawn figure.)

```{.python .input #maximum-likelihood-a-concrete-example}
#@tab mxnet
theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input #maximum-likelihood-a-concrete-example}
#@tab pytorch
theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input #maximum-likelihood-a-concrete-example}
#@tab tensorflow
theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input #maximum-likelihood-a-concrete-example}
#@tab jax
theta = jnp.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

The curve peaks near the expected $9/13 \approx 0.69$. To pin the maximum
exactly we use calculus: at an interior maximum the derivative vanishes. Setting
$\frac{d}{d\theta}P(X\mid\theta)=0$,

$$
0 = \frac{d}{d\theta}\,\theta^9(1-\theta)^4
  = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3
  = \theta^8(1-\theta)^3(9-13\theta).
$$

The roots $\theta=0$ and $\theta=1$ assign probability zero to a sequence that
contains both heads and tails, so they are minima. The remaining root is the
maximum likelihood estimate,

$$
\hat\theta = \frac{9}{13},
$$

exactly the observed fraction of heads. The same calculation on the general
likelihood :eqref:`eq_mdl-coin-like` gives $\hat\theta = n_H/(n_H+n_T)$: the MLE
of a coin's bias is *always* the empirical frequency of heads, our first sign
that maximum likelihood recovers the "obvious" estimator.

## The Negative Log-Likelihood

The closed-form trick above does not survive contact with a real model: with
billions of parameters there is no characteristic-polynomial to factor, and we
optimize numerically instead. But the likelihood itself is a numerically hostile
object. With $n$ independent data points it is a product of $n$ probabilities,

$$
P(X\mid\boldsymbol{\theta}) = \prod_{i=1}^n p(x_i\mid\boldsymbol{\theta}),
$$

each in $[0,1]$. A billion factors near $1/2$ produce roughly $(1/2)^{10^9}$,
underflowing to zero in any floating-point format long before we can
differentiate it.

The logarithm rescues us, turning the product into a sum that stays in range:

$$
\log\!\big((1/2)^{10^9}\big) = 10^9 \cdot \log(1/2) \approx -6.93\times10^{8},
$$

which fits comfortably even in single precision. Since $x \mapsto \log x$ is
strictly increasing, it preserves the location of the maximum, so maximizing the
likelihood is the same as maximizing the *log-likelihood*
$\log P(X\mid\boldsymbol{\theta})$. We use this exact reasoning for the naive
Bayes classifier in :numref:`sec_mdl-naive_bayes`. Finally, because we prefer to
*minimize* losses, we flip the sign and define the **negative log-likelihood**
(NLL):

$$
\ell(\boldsymbol{\theta}) = -\log P(X\mid\boldsymbol{\theta})
  = -\sum_{i=1}^n \log p(x_i\mid\boldsymbol{\theta}).
$$
:eqlabel:`eq_mdl-nll`

For the coin this reads
$\ell(\theta) = -\big(n_H\log\theta + n_T\log(1-\theta)\big)$, which we can
minimize by gradient descent even for billions of flips---no closed form
required. The cell below does exactly this for almost nine million flips and
confirms it recovers $n_H/(n_H+n_T)$.

```{.python .input #maximum-likelihood-numerical-optimization-and-the-negative-log-likelihood}
#@tab mxnet
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our parameters
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input #maximum-likelihood-numerical-optimization-and-the-negative-log-likelihood}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our parameters
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input #maximum-likelihood-numerical-optimization-and-the-negative-log-likelihood}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our parameters
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input #maximum-likelihood-numerical-optimization-and-the-negative-log-likelihood}
#@tab jax
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our parameters
theta = jnp.float32(0.5)

# Define loss function
def nll(theta):
    return -(n_H * jnp.log(theta) + n_T * jnp.log(1 - theta))

grad_fn = jax.grad(nll)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    theta = theta - lr * grad_fn(theta)

# Check output
theta, n_H / (n_H + n_T)
```

Beyond numerical stability, the log gives gradients a clean additive form. The
product rule applied to $\prod_i p(x_i\mid\boldsymbol{\theta})$ produces $n$
terms, each a product of $n-1$ surviving factors; the logarithm collapses this to
a single sum,

$$
\nabla_{\boldsymbol{\theta}}\, \ell(\boldsymbol{\theta})
  = -\sum_{i=1}^n \frac{\nabla_{\boldsymbol{\theta}}\, p(x_i\mid\boldsymbol{\theta})}{p(x_i\mid\boldsymbol{\theta})},
$$

a chain-rule application term by term with no shared subproducts to track. This
is why every minibatch loss in the book is an *average* of per-example NLLs:
the gradient of a sum is the sum of gradients, so we can estimate
$\nabla\ell$ from a random subset of the data.

## Maximum Likelihood Is Minimizing a Loss

We now make precise the claim from the introduction. "Minimize the loss" and "do
maximum likelihood" are the same instruction in different words. The bridge is
one rescaling.

### NLL Is the Cross-Entropy to the Data
:label:`subsec_mdl-nll-crossentropy`

Dividing the NLL :eqref:`eq_mdl-nll` by the number of examples turns it into an
average---the natural per-example loss---and that average has an
information-theoretic name. Let $\hat p_{\textrm{data}}$ be the *empirical
distribution*, which places mass $1/n$ on each observed $x_i$ (working in natural
logarithms, so the units are *nats*; switching to bits merely rescales every
quantity by $\ln 2$ and changes no $\mathop{\mathrm{argmin}}$).

**Proposition (MLE = minimum cross-entropy).** *Maximizing the likelihood is the
same as minimizing the cross-entropy from the empirical distribution
$\hat p_{\textrm{data}}$ to the model $p_{\boldsymbol{\theta}}$:*

$$
\mathop{\mathrm{argmax}}_{\boldsymbol{\theta}} \prod_{i=1}^n p(x_i\mid\boldsymbol{\theta})
  = \mathop{\mathrm{argmin}}_{\boldsymbol{\theta}}\, \textrm{CE}\!\left(\hat p_{\textrm{data}},\, p_{\boldsymbol{\theta}}\right),
\qquad
\textrm{CE}(\hat p_{\textrm{data}}, p_{\boldsymbol{\theta}}) = -\!\!\sum_{x} \hat p_{\textrm{data}}(x)\log p_{\boldsymbol{\theta}}(x).
$$

**Proof.** Take the average negative log-likelihood and rewrite the sum over
data points as a sum over distinct outcomes weighted by their empirical
frequency. Because each $x_i$ carries empirical mass $\hat p_{\textrm{data}}(x_i)=1/n$,

$$
\frac{1}{n}\,\ell(\boldsymbol{\theta})
  = -\frac{1}{n}\sum_{i=1}^n \log p_{\boldsymbol{\theta}}(x_i)
  = -\sum_{x} \hat p_{\textrm{data}}(x)\,\log p_{\boldsymbol{\theta}}(x)
  = \textrm{CE}\!\left(\hat p_{\textrm{data}},\, p_{\boldsymbol{\theta}}\right),
$$

which is exactly the cross-entropy named in the proposition. Scaling by the
positive constant $1/n$ and flipping the sign do not move the optimizer, so the
$\mathop{\mathrm{argmax}}$ of the likelihood equals the $\mathop{\mathrm{argmin}}$
of the cross-entropy. $\blacksquare$

This also explains *why* minimizing cross-entropy is the right thing to do.
Cross-entropy decomposes as $\textrm{CE}(P,Q)=H(P)+D_{\textrm{KL}}(P\|Q)$, the
entropy of $P$ plus the Kullback--Leibler divergence from $P$ to $Q$ (see
:numref:`sec_mdl-information_theory` for that background). Setting
$P=\hat p_{\textrm{data}}$ and $Q=p_{\boldsymbol{\theta}}$, the empirical entropy
$H(\hat p_{\textrm{data}})$ is fixed by the data and does not depend on
$\boldsymbol{\theta}$, so minimizing the cross-entropy is *identical* to
minimizing the KL term alone:

$$
\mathop{\mathrm{argmin}}_{\boldsymbol{\theta}}\, \textrm{CE}\!\left(\hat p_{\textrm{data}},\, p_{\boldsymbol{\theta}}\right)
  = \mathop{\mathrm{argmin}}_{\boldsymbol{\theta}}\, D_{\textrm{KL}}\!\left(\hat p_{\textrm{data}}\,\|\,p_{\boldsymbol{\theta}}\right).
$$

This is the deepest reading of the section: maximum likelihood is the
*KL-projection* of the empirical distribution onto the model family---it picks
the $p_{\boldsymbol{\theta}}$ closest to the data in the only sense that matters
for coding. Training cannot drive the cross-entropy below the floor
$H(\hat p_{\textrm{data}})$, the irreducible cost of the data's own randomness;
all it can remove is the KL gap, as :numref:`fig_mdl-mle-kl` shows. As a special
case, the *categorical* NLL of a classifier with one-hot labels is precisely the
softmax cross-entropy loss of :numref:`sec_softmax`.

![Minimizing the negative log-likelihood is minimizing the KL divergence to the data. The per-example cross-entropy splits into the entropy $H(\hat p_{\textrm{data}})$, an irreducible floor set by the data's own randomness, plus $D_{\textrm{KL}}(\hat p_{\textrm{data}} \| p_{\boldsymbol{\theta}})$, the only part training can remove. As the model improves, the KL slice shrinks and the cross-entropy descends toward---but never below---the floor.](../img/mdl-prob-mle-kl.svg)
:label:`fig_mdl-mle-kl`

### Gaussian NLL Is Mean Squared Error
:label:`subsec_mdl-gaussian-mse`

The most-used regression loss falls out of the same principle by choosing a
Gaussian noise model. Suppose each target is the model's prediction plus
Gaussian noise of *fixed* variance, $y_i \sim \mathcal{N}(\hat y_i, \sigma^2)$
with $\hat y_i = f_{\boldsymbol{\theta}}(\mathbf{x}_i)$.

**Proposition (Gaussian NLL = MSE).** *For fixed-variance Gaussian targets, the
negative log-likelihood equals the mean squared error up to a constant
independent of $\boldsymbol{\theta}$:*

$$
-\log \prod_{i=1}^n \mathcal{N}(y_i;\,\hat y_i,\,\sigma^2)
  = \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\hat y_i)^2 + \frac{n}{2}\log(2\pi\sigma^2).
$$
:eqlabel:`eq_mdl-gaussian-nll`

*Hence $\mathop{\mathrm{argmin}}_{\boldsymbol{\theta}}$ of this NLL is the
$\mathop{\mathrm{argmin}}$ of $\sum_i (y_i-\hat y_i)^2$.*

**Proof.** The Gaussian density :eqref:`eq_mdl-gaussian_pdf` is
$\mathcal{N}(y;\hat y,\sigma^2) = (2\pi\sigma^2)^{-1/2}\exp\!\big(-(y-\hat y)^2/(2\sigma^2)\big)$.
Take its negative log,

$$
-\log \mathcal{N}(y_i;\hat y_i,\sigma^2)
  = \frac{(y_i-\hat y_i)^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2),
$$

then sum over the $n$ independent examples, which is :eqref:`eq_mdl-gaussian-nll`.
The second term is the same for every $\boldsymbol{\theta}$ (the variance is
fixed), so it cannot affect the minimizer, and the leading factor $1/(2\sigma^2)$
is a positive constant. Dropping both leaves $\sum_i (y_i-\hat y_i)^2$, the
sum of squared errors. $\blacksquare$

So squared-error regression is *not* an arbitrary choice---it is maximum
likelihood under the assumption that the residuals are Gaussian, the assumption
the central limit theorem makes plausible whenever the noise is a sum of many
small independent effects (:numref:`sec_mdl-distributions`). The same recipe
turns *every* common loss into the NLL of a chosen conditional
$p(y\mid\mathbf x)$: pick the noise model and its negative log *is* the loss.

| Noise model $p(y\mid\mathbf x)$ | Negative log-likelihood $-\log p(y\mid\mathbf x)$ | Deep-learning loss |
|---|---|---|
| Gaussian, fixed variance | $\tfrac{1}{2\sigma^2}(y-\hat y)^2 + \textrm{const}$ | mean squared error (MSE) |
| Bernoulli | $-\,y\log\hat p - (1-y)\log(1-\hat p)$ | binary cross-entropy (BCE) |
| Categorical (one-hot $y$) | $-\sum_k y_k \log \hat p_k$ | softmax cross-entropy |
| Laplace, fixed scale | $\tfrac{1}{b}\lvert y-\hat y\rvert + \textrm{const}$ | mean absolute error (MAE) |

Each row is one line of algebra---take the density, drop the
$\boldsymbol{\theta}$-free constants---of exactly the kind we just did for the
Gaussian. Picking a loss *is* picking a noise model.

### Why Maximum Likelihood Works

A fair worry is whether $\hat{\boldsymbol{\theta}}$ is any good. The first
reassurance is the same KL picture. Drawing genuinely i.i.d. data from a true
distribution $p_{\boldsymbol{\theta}^\star}$, the average NLL is, by the law of
large numbers, an estimate of the *expected* cross-entropy
$\textrm{CE}(p_{\boldsymbol{\theta}^\star}, p_{\boldsymbol{\theta}})$, which the
proposition above shows is minimized exactly at the truth
$\boldsymbol{\theta}=\boldsymbol{\theta}^\star$ (the KL term vanishes there).
Minimizing the empirical average therefore targets the right minimizer, and as
$n\to\infty$ the empirical objective converges to the population one. This is the
formal content of *consistency*: for a well-specified model the MLE converges in
probability to the true parameter, $\hat{\boldsymbol{\theta}}\xrightarrow{P}\boldsymbol{\theta}^\star$.
The coin made this concrete: $\hat\theta = n_H/(n_H+n_T) \to \theta^\star$ by the
law of large numbers. Consistency, asymptotic unbiasedness, and efficiency are
the estimator-quality notions of :numref:`sec_mdl-statistics`; here they describe
the MLE specifically.

Consistency says only *where* $\hat{\boldsymbol{\theta}}$ lands; the sharper
statement is *how fast* and *how tightly* it concentrates. Under mild regularity
conditions the MLE is **asymptotically normal**: the rescaled error converges in
distribution to a Gaussian,

$$
\sqrt{n}\,\big(\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}^\star\big)
  \;\xrightarrow{d}\; \mathcal{N}\!\big(\mathbf{0},\; I(\boldsymbol{\theta}^\star)^{-1}\big),
$$
:eqlabel:`eq_mdl-asymptotic-normality`

where $I(\boldsymbol{\theta})$ is the **Fisher information** defined just below
:cite:`Bishop.2006,Wasserman.2013`. Two things are worth extracting. First, the
error shrinks at the $1/\sqrt{n}$ rate, so the standard error of each component
falls like $1/\sqrt n$---halving it costs four times the data. Second, the limiting variance is exactly
$I(\boldsymbol{\theta}^\star)^{-1}$, and the Cramér--Rao bound of
:numref:`sec_mdl-statistics` says *no* unbiased estimator can have smaller
variance than $I(\boldsymbol{\theta}^\star)^{-1}/n$. The MLE attains that floor in
the limit: it is **asymptotically efficient**---asymptotically, the best estimator
there is. The qualifier matters. At finite $n$ the MLE is generally only
*asymptotically* unbiased, not exactly unbiased---the maximum-likelihood estimate
of a Gaussian's variance divides the sum of squares by $n$ rather than $n-1$ and
so systematically underestimates $\sigma^2$, the biased estimator worked out in
:numref:`sec_mdl-statistics`. Maximum likelihood is optimal *in the limit*, not a
guarantee of unbiasedness at every sample size.

#### Fisher Information and the Score

To make :eqref:`eq_mdl-asymptotic-normality` quantitative we need the object
$I(\boldsymbol{\theta})$ it contains. Define the **score** as the gradient of the
log-likelihood with respect to the parameters,
$s(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \log p(x\mid\boldsymbol{\theta})$.
The score is the per-example version of the NLL gradient we already met: setting
$\sum_i s(\boldsymbol{\theta}) = \mathbf 0$ is exactly the first-order condition
that locates $\hat{\boldsymbol{\theta}}$. At the true parameter the score has
mean zero, and its variance is the **Fisher information**,

$$
I(\boldsymbol{\theta}) = \operatorname{Var}_{\boldsymbol{\theta}}\!\big[\,s(\boldsymbol{\theta})\,\big]
  = \mathbb{E}_{\boldsymbol{\theta}}\!\big[\,s(\boldsymbol{\theta})\,s(\boldsymbol{\theta})^\top\,\big]
  = -\,\mathbb{E}_{\boldsymbol{\theta}}\!\big[\nabla^2_{\boldsymbol{\theta}} \log p(x\mid\boldsymbol{\theta})\big],
$$
:eqlabel:`eq_mdl-fisher`

the last equality (valid under the same regularity conditions) saying that the
information is the expected *curvature* of the NLL. The reading is intuitive: a
sharply peaked log-likelihood---large curvature, large $I$---pins the parameter
down tightly, so the estimate has small variance; a flat one leaves
$\boldsymbol{\theta}$ poorly determined. That is precisely what
:eqref:`eq_mdl-asymptotic-normality` says, $\operatorname{Var}(\hat{\boldsymbol{\theta}}) \approx I(\boldsymbol{\theta}^\star)^{-1}/n$.

The coin closes the loop on itself. A single flip has log-likelihood
$\log p(x\mid\theta) = x\log\theta + (1-x)\log(1-\theta)$ for $x\in\{0,1\}$, so
the score and its derivative are

$$
s(\theta) = \frac{x}{\theta} - \frac{1-x}{1-\theta},
\qquad
\frac{d}{d\theta}\, s(\theta) = -\frac{x}{\theta^2} - \frac{1-x}{(1-\theta)^2}.
$$

Taking $-\mathbb{E}[\,\cdot\,]$ with $\mathbb{E}[x]=\theta$ collapses the two
fractions to $\tfrac{1}{\theta} + \tfrac{1}{1-\theta}$, so the per-flip Fisher
information and its inverse are

$$
I(\theta) = \frac{1}{\theta} + \frac{1}{1-\theta} = \frac{1}{\theta(1-\theta)},
\qquad
\frac{I(\theta)^{-1}}{n} = \frac{\theta(1-\theta)}{n}.
$$
:eqlabel:`eq_mdl-coin-fisher`

The prediction is therefore that the MLE $\hat\theta = n_H/n$ over $n$ flips has
variance about $\theta(1-\theta)/n$. We can check that this is *exactly* right
here---$\hat\theta$ is a scaled binomial count, whose variance is
$\theta(1-\theta)/n$ on the nose---and confirm it by simulation: run the
$n$-flip experiment many times and compare the spread of the resulting estimates
to the Cramér--Rao floor :eqref:`eq_mdl-coin-fisher`.

```{.python .input #maximum-likelihood-fisher-information}
# A framework-agnostic numpy simulation of the coin MLE's variance.
theta_star, n, trials = 0.3, 200, 20000
rng = onp.random.default_rng(0)
# Each trial: flip the coin n times; the MLE is the observed head fraction.
heads = rng.binomial(n, theta_star, size=trials)
theta_hats = heads / n

empirical_var = theta_hats.var()
cramer_rao = theta_star * (1 - theta_star) / n  # = I(theta)^-1 / n
empirical_var, cramer_rao
```

The empirical variance of the maximum-likelihood estimates matches the
Cramér--Rao floor $\theta(1-\theta)/n$ to two or three digits: the coin's MLE is
not merely consistent, it is as tight as :eqref:`eq_mdl-asymptotic-normality`
permits. This is the concrete content of "asymptotically efficient," and it is
why the curvature of a loss surface---its Fisher information, or empirically its
Hessian---tells us how trustworthy a fitted parameter is.

## MAP Estimation: Priors as Regularizers
:label:`subsec_mdl-map`

We dropped the prior $P(\boldsymbol{\theta})$ by declaring it uninformative.
Keeping it instead turns maximum likelihood into *maximum a posteriori* (MAP)
estimation---and the prior reappears as a regularizer. Maximizing the full
posterior :eqref:`eq_mdl-max_like` and taking negative logs,

$$
\hat{\boldsymbol{\theta}}_{\textrm{MAP}}
  = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}}\, P(X\mid\boldsymbol{\theta})\,P(\boldsymbol{\theta})
  = \mathop{\mathrm{argmin}}_{\boldsymbol{\theta}}\,
    \Big[\underbrace{-\textstyle\sum_i \log p(x_i\mid\boldsymbol{\theta})}_{\textrm{data fit (NLL)}}
       \;\underbrace{-\log P(\boldsymbol{\theta})}_{\textrm{regularizer}}\Big].
$$
:eqlabel:`eq_mdl-map`

The objective is the familiar data-fit term plus a penalty: the negative
log-prior. A prior concentrated near small parameters penalizes large ones, which
is precisely what a regularizer does.

**Proposition (Gaussian prior = $L_2$ / weight decay).** *A Gaussian prior
$\boldsymbol{\theta}\sim\mathcal{N}(\mathbf 0,\tau^2 \mathbf I)$ contributes the
penalty $\tfrac{1}{2\tau^2}\lVert\boldsymbol{\theta}\rVert_2^2$ to the MAP
objective. With Gaussian-noise data (variance $\sigma^2$), MAP estimation is
$L_2$-regularized least squares with weight-decay strength
$\lambda = \sigma^2/\tau^2$.*

**Proof.** The density of $\mathcal{N}(\mathbf 0,\tau^2\mathbf I)$ is proportional
to $\exp\!\big(-\lVert\boldsymbol{\theta}\rVert_2^2/(2\tau^2)\big)$, so its
negative log-prior is $\tfrac{1}{2\tau^2}\lVert\boldsymbol{\theta}\rVert_2^2$ plus
a constant. Insert this and the Gaussian NLL :eqref:`eq_mdl-gaussian-nll` into
the MAP objective :eqref:`eq_mdl-map`:

$$
\hat{\boldsymbol{\theta}}_{\textrm{MAP}}
  = \mathop{\mathrm{argmin}}_{\boldsymbol{\theta}}\;
    \frac{1}{2\sigma^2}\sum_{i}(y_i-\hat y_i)^2
      + \frac{1}{2\tau^2}\lVert\boldsymbol{\theta}\rVert_2^2 .
$$

Multiplying by the positive constant $2\sigma^2$ leaves
$\sum_i(y_i-\hat y_i)^2 + \tfrac{\sigma^2}{\tau^2}\lVert\boldsymbol{\theta}\rVert_2^2$,
which is ridge regression with $\lambda=\sigma^2/\tau^2$. $\blacksquare$

This is the probabilistic origin of *weight decay*: the regularizer we add by
hand in the optimization chapter is the log of a Gaussian belief that weights
should be small. The same construction with a *Laplace* prior
$p(\boldsymbol{\theta})\propto e^{-\lVert\boldsymbol{\theta}\rVert_1/b}$ gives the
$L_1$ penalty $\tfrac{1}{b}\lVert\boldsymbol{\theta}\rVert_1$, whose kink at the
origin produces exact zeros---sparsity. :numref:`fig_mdl-prob-map-prior` shows
the tug-of-war: the NLL bowl wants the MLE, the log-prior bowl wants the prior
mean, and the MAP estimate sits between them.

![Maximum a posteriori as a tug-of-war between data and prior. The negative log-likelihood (blue) is minimized at the MLE; adding a Gaussian log-prior bowl centered at the prior mean (orange, dashed) yields the MAP objective (green), whose minimum is pulled from the MLE toward the prior mean. A tighter prior (smaller $\tau$) pulls harder; as $\tau\to\infty$ the prior flattens and MAP returns to the MLE.](../img/mdl-prob-map-prior.svg)
:label:`fig_mdl-prob-map-prior`

The figure also reads off the two limits. As the prior flattens ($\tau\to\infty$,
or $b\to\infty$) the penalty vanishes and $\hat{\boldsymbol{\theta}}_{\textrm{MAP}}
\to \hat{\boldsymbol{\theta}}_{\textrm{MLE}}$---the uninformative-prior assumption
we started with. And as the data grows the NLL bowl deepens faster than the fixed
prior, so the data term dominates and MAP again approaches the MLE. Regularization
matters most precisely when data is scarce.

One caution before we move on: MAP is *not* the same as "being Bayesian." MAP
restores the prior but still reports a single point---the *mode* of the posterior
$P(\boldsymbol{\theta}\mid X)$, the same $\mathop{\mathrm{argmax}}$ as in
:eqref:`eq_mdl-max_like`. Genuine Bayesian inference keeps the entire posterior
*distribution* and *integrates* over it, predicting with the posterior-averaged
$\int p(x\mid\boldsymbol{\theta})\,P(\boldsymbol{\theta}\mid X)\,d\boldsymbol{\theta}$
rather than plugging in one $\hat{\boldsymbol{\theta}}_{\textrm{MAP}}$. That
average propagates parameter *uncertainty* into the prediction, where the mode
discards it---and the mode is not even reparameterization-invariant, since a
nonlinear change of variables moves the peak of a density but not its integral.
Marginalizing over the posterior is its own subject, beyond our scope here
:cite:`Murphy.2022,mackay2003information`; the point is only that MAP is one more
*point estimate*---maximum likelihood with a penalty---and should not be mistaken
for the full Bayesian treatment.

## Continuous Variables

Everything above was phrased for discrete outcomes, where $P(X\mid\boldsymbol{\theta})$
is a genuine probability. For continuous data we replace probabilities by
densities $p$, and the only worry is that the probability of observing *any
exact* real value is zero, so the naive likelihood is $0$ for every
$\boldsymbol{\theta}$. The resolution is to ask for a match only to within a small
tolerance $\epsilon$, then watch the $\epsilon$ cancel.

For i.i.d. observations $x_1,\ldots,x_N$, the probability that each lands in a
window of width $\epsilon$ is, to first order,

$$
P\big(X_i \in [x_i, x_i+\epsilon]\ \forall i \mid \boldsymbol{\theta}\big)
  \approx \epsilon^N \prod_{i=1}^N p(x_i\mid\boldsymbol{\theta}).
$$

Taking the negative log,

$$
-\log P\big(X_i \in [x_i, x_i+\epsilon]\ \forall i\mid\boldsymbol{\theta}\big)
  \approx -N\log\epsilon - \sum_{i} \log p(x_i\mid\boldsymbol{\theta}).
$$

The tolerance enters only through the additive constant $-N\log\epsilon$, which is
free of $\boldsymbol{\theta}$. Demanding four digits of precision or four hundred
changes that constant but never the minimizer, so we drop it and minimize

$$
-\sum_{i} \log p(x_i\mid\boldsymbol{\theta})
$$

exactly as in the discrete case. Maximum likelihood thus operates on continuous
variables by swapping probabilities for densities and nothing more; the Gaussian
NLL of :numref:`subsec_mdl-gaussian-mse` is precisely this construction.

## Summary

* The **maximum likelihood** principle picks the parameters that make the
  observed data most probable; with an uninformative prior, Bayes' rule reduces
  the posterior mode to $\mathop{\mathrm{argmax}}_{\boldsymbol{\theta}} P(X\mid\boldsymbol{\theta})$.
* We optimize the **negative log-likelihood** $-\sum_i \log p(x_i\mid\boldsymbol{\theta})$:
  the log fixes underflow and makes gradients an additive sum, while the sign
  flip turns "maximize probability" into "minimize a loss."
* The average NLL *is* the **cross-entropy** from the empirical distribution to
  the model (this section is the canonical derivation; see
  :numref:`sec_mdl-information_theory` for the entropy/KL background). Minimizing
  it minimizes the KL divergence to the data---maximum likelihood is the
  KL-projection onto the model family. Categorical NLL is softmax cross-entropy,
  **fixed-variance Gaussian NLL is mean squared error**, and a Laplace model gives
  mean absolute error: picking a loss is picking a noise model.
* The MLE is **consistent** and, under regularity conditions, **asymptotically
  normal** with limiting variance the inverse **Fisher information**
  $I(\boldsymbol{\theta}^\star)^{-1}/n$---so it attains the Cramér--Rao bound and
  is *asymptotically efficient*. At finite $n$ it is generally only
  asymptotically unbiased (:numref:`sec_mdl-statistics`).
* **MAP** estimation restores the prior, adding its negative log as a penalty: a
  Gaussian prior is exactly $L_2$ / weight decay ($\lambda=\sigma^2/\tau^2$), a
  Laplace prior is $L_1$ / sparsity, and MAP $\to$ MLE as the prior flattens or
  the data grows. MAP is still a *point estimate* (the posterior mode), not full
  Bayesian inference, which integrates over the posterior.
* Maximum likelihood extends to continuous variables unchanged---probabilities
  become densities, and the matching-tolerance $\epsilon$ drops out as a constant.

## Exercises
1. A non-negative random variable has density $\alpha e^{-\alpha x}$ for some
   $\alpha>0$. From the single observation $x=3$, find the maximum likelihood
   estimate of $\alpha$. Generalize to a sample $\{x_i\}_{i=1}^N$ and show
   $\hat\alpha = 1/\bar x$.
2. Given a sample $\{x_i\}_{i=1}^N$ from a Gaussian with unknown mean and
   variance $1$, show the maximum likelihood estimate of the mean is the sample
   average $\bar x$. (*Hint:* minimize the Gaussian NLL of
   :numref:`subsec_mdl-gaussian-mse`.)
3. Show directly that a fixed-variance Gaussian NLL equals the mean squared error
   up to an additive constant, and identify the constant.
4. For a $K$-class classifier with one-hot label $\mathbf y$ and softmax
   prediction $\hat{\mathbf p}$, show the categorical NLL $-\sum_k y_k\log\hat p_k$
   is the cross-entropy, and that its gradient with respect to the logits is
   $\hat{\mathbf p} - \mathbf y$.
5. Show that a Gaussian prior on the regression weights recovers ridge
   regression, and read off the weight-decay strength in terms of the noise and
   prior variances.
6. Show that MAP estimation with a Laplace prior gives an $L_1$ penalty, and
   argue why its kink at the origin can force coefficients to be exactly zero.
7. Prove that $\hat{\boldsymbol{\theta}}_{\textrm{MAP}} \to \hat{\boldsymbol{\theta}}_{\textrm{MLE}}$
   as the prior variance $\tau^2\to\infty$, and explain why the same happens as
   the dataset size $n\to\infty$ for fixed $\tau$.
8. For a single Gaussian observation with known variance $\sigma^2$ and unknown
   mean $\mu$, compute the score $\frac{d}{d\mu}\log p(x\mid\mu)$ and show the
   Fisher information is $I(\mu)=1/\sigma^2$. Conclude that the sample-mean MLE
   over $n$ observations has variance $\sigma^2/n$, attaining the Cramér--Rao
   bound exactly. (*Hint:* use :eqref:`eq_mdl-fisher`.)


:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1097)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1097)
:end_tab:

<!-- slides -->

::: {.slide title="Maximum Likelihood"}
**Maximum likelihood**: pick the parameters that make the
observed data most probable.

$$\hat{\boldsymbol\theta} = \arg\max_{\boldsymbol\theta} \prod_i p(x_i \mid \boldsymbol\theta)
            = \arg\min_{\boldsymbol\theta} -\sum_i \log p(x_i \mid \boldsymbol\theta).$$

With an uninformative prior, the posterior mode is just the
likelihood maximizer. The log fixes underflow, makes the
gradient an additive sum, and flips "maximize probability"
into "minimize a loss."
:::

::: {.slide title="A concrete example"}
For 9 heads and 4 tails, the likelihood curve peaks at
$\hat\theta = 9/13$ — the observed fraction of heads. The
MLE of a coin's bias is always the empirical frequency:

@maximum-likelihood-a-concrete-example
:::

::: {.slide title="Numerical optimization (NLL)"}
No closed form for billions of parameters: minimize the NLL
by gradient descent. Sums of logs behave numerically;
the gradient is a per-example sum, so SGD works on minibatches:

@maximum-likelihood-numerical-optimization-and-the-negative-log-likelihood
:::

::: {.slide title="Every loss is an NLL"}
The everyday losses are negative log-likelihoods of a chosen
conditional distribution:

- **Cross-entropy** = NLL of a categorical $p(y \mid x)$;
  the average NLL *is* the cross-entropy to the data, hence
  $\min$ KL to the empirical distribution.
- **MSE** = NLL of a Gaussian $p(y \mid x)$ with fixed
  variance: $-\log\prod \mathcal N(y_i;\hat y_i,\sigma^2)
  = \tfrac{1}{2\sigma^2}\sum_i (y_i-\hat y_i)^2 + \text{const}$.
- **MAE** = NLL of a Laplace $p(y \mid x)$.

So "minimize the loss" is "do MLE," and picking a loss is
picking a noise model.
:::

::: {.slide title="Why it works: KL projection + Fisher"}
Cross-entropy $=H(\hat p_{\textrm{data}})+D_{\textrm{KL}}(\hat p_{\textrm{data}}\|p_{\boldsymbol\theta})$.
The entropy is an irreducible floor; training only removes the
KL gap, so MLE is the **KL-projection** of the data onto the
model family:

@fig:mdl-mle-kl

It is also *consistent* and, under regularity conditions,
asymptotically normal,
$\sqrt n(\hat{\boldsymbol\theta}-\boldsymbol\theta^\star)\to
\mathcal N(\mathbf 0, I(\boldsymbol\theta^\star)^{-1})$, with
**Fisher information** $I(\boldsymbol\theta)$ — attaining the
Cramér–Rao floor (asymptotically efficient). For the coin,
$\operatorname{Var}(\hat\theta)\approx\theta(1-\theta)/n$.
:::

::: {.slide title="MAP: priors are regularizers"}
Keep the prior $P(\boldsymbol\theta)$ and its negative log is
a penalty on the NLL:

$$\hat{\boldsymbol\theta}_{\textrm{MAP}} = \arg\min_{\boldsymbol\theta}
  \big[-\textstyle\sum_i \log p(x_i\mid\boldsymbol\theta) - \log P(\boldsymbol\theta)\big].$$

Gaussian prior $\Rightarrow$ $L_2$ / weight decay
($\lambda=\sigma^2/\tau^2$); Laplace prior $\Rightarrow$ $L_1$
/ sparsity. MAP $\to$ MLE as the prior flattens or data grows:

@fig:mdl-prob-map-prior
:::

::: {.slide title="Recap"}
- MLE: maximize $\sum_i \log p(x_i \mid \boldsymbol\theta)$;
  equivalently minimize the NLL.
- Average NLL = cross-entropy to the data = KL to the
  empirical distribution; consistent and (asymptotically)
  efficient via the Fisher information when well-specified.
- Most "losses" in DL are NLLs of suitable conditional
  distributions; MSE is the Gaussian case.
- Priors become regularizers: Gaussian $\to$ weight decay,
  Laplace $\to$ sparsity.
:::
