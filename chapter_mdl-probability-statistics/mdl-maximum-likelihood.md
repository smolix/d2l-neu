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
mean squared error---then asks how good the resulting estimates are
(consistency, asymptotic normality, and the Fisher information), shows how
putting a prior back in recovers $L_2$ regularization, and ends where modern
generative models begin: when latent variables make the likelihood intractable,
a lower bound---the ELBO---and the EM algorithm rescue the principle. We keep
the running coin-flip example throughout because it
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

(When $\boldsymbol{\theta}$ is continuous, $P(\boldsymbol{\theta}\mid X)$ is a
*density*, and "most probable" means the *mode* of that density---a subtlety
that will matter when we discuss MAP estimation in :numref:`subsec_mdl-map`.
Strictly speaking, then, :eqref:`eq_mdl-max_like` as written is the *posterior
mode*---the MAP objective of that section; the next paragraph identifies the
condition, a flat prior, under which it reduces to the likelihood alone, and it
is that reduced form---not the posterior mode in general---that names this
section.)

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
parameter value is better than another---a *flat (uniform) prior*, e.g. that the
coin's bias is equally likely to be any value in $[0,1]$---is
$P(\boldsymbol{\theta})$ constant in $\boldsymbol{\theta}$ and free to drop too.
(We say "flat" rather than the common "uninformative" deliberately: a prior flat
in $\theta$ is not flat in a transformed parameter such as $\theta^2$, a wrinkle
we return to in :numref:`subsec_mdl-map`.)
What remains is the **likelihood** $P(X \mid \boldsymbol{\theta})$, the
probability the data assigns under each parameter, and the principle in its
working form:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}}\, P(X \mid \boldsymbol{\theta}).
$$
:eqlabel:`eq_mdl-mle`

We restore the prior $P(\boldsymbol{\theta})$ deliberately in
:numref:`subsec_mdl-map`; everything until then assumes it is flat.

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

### The Negative Log-Likelihood

The closed-form trick above does not survive contact with a real model: with
billions of parameters there is no polynomial to factor, and we
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

One hyperparameter deserves a wary eye before moving on: the learning rate
$10^{-9}$ looks absurd, and it works only because this loss is a *sum* over
nearly nine million flips, so its gradient carries that factor of $n$. Shrink
$n_H$ and $n_T$ a thousandfold and the same learning rate barely moves
$\theta$; scale them up and the update diverges. Averaging the loss over
examples instead of summing---as the minibatch losses in this book do---removes
the coupling and returns sensible learning rates to sensible magnitudes. The
lesson generalizes: a learning rate is always tuned *to a loss scale*, and
silently changing one without the other is a classic way to break training.

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
distribution*, which places mass $1/n$ on each *observation*; masses at repeated
values stack, so an outcome $x$ that occurs $n_x$ times in the sample carries
total mass $\hat p_{\textrm{data}}(x) = n_x/n$, its empirical frequency. (We work
in natural logarithms, so the units are *nats*; switching to bits merely rescales
every quantity by $\ln 2$ and changes no $\mathop{\mathrm{argmin}}$.)

**Proposition (MLE = minimum cross-entropy).** *Maximizing the likelihood is the
same as minimizing the cross-entropy from the empirical distribution
$\hat p_{\textrm{data}}$ to the model $p_{\boldsymbol{\theta}}$:*

$$
\mathop{\mathrm{argmax}}_{\boldsymbol{\theta}} \prod_{i=1}^n p(x_i\mid\boldsymbol{\theta})
  = \mathop{\mathrm{argmin}}_{\boldsymbol{\theta}}\, \textrm{CE}\!\left(\hat p_{\textrm{data}},\, p_{\boldsymbol{\theta}}\right),
\qquad
\textrm{CE}(\hat p_{\textrm{data}}, p_{\boldsymbol{\theta}}) = -\!\!\sum_{x} \hat p_{\textrm{data}}(x)\log p_{\boldsymbol{\theta}}(x).
$$

**Proof.** Take the average negative log-likelihood and regroup the sum over
data points into a sum over *distinct outcomes*, pooling the observations with
$x_i = x$. Each observation contributes mass $1/n$, so outcome $x$ receives
total mass $n_x/n = \hat p_{\textrm{data}}(x)$. On the coin this regrouping is
concrete: the nine head-flips pool into nine copies of $\log p_{\boldsymbol{\theta}}(\textrm{H})$,
so heads enters the average with weight $\hat p_{\textrm{data}}(\textrm{H}) = 9/13$,
and tails with weight $4/13$. In general,

$$
\frac{1}{n}\,\ell(\boldsymbol{\theta})
  = -\frac{1}{n}\sum_{i=1}^n \log p_{\boldsymbol{\theta}}(x_i)
  = -\sum_{x} \frac{n_x}{n}\,\log p_{\boldsymbol{\theta}}(x)
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
for coding. The projection reading even survives a *misspecified* model: when no
$p_{\boldsymbol{\theta}}$ equals the data-generating distribution, the MLE
converges not to a "true parameter" (there is none) but to the parameter whose
$p_{\boldsymbol{\theta}}$ is KL-closest to it :cite:`White.1982`. Training
cannot drive the cross-entropy below the floor
$H(\hat p_{\textrm{data}})$, the irreducible cost of the data's own randomness;
all it can remove is the KL gap, as :numref:`fig_mdl-mle-kl` shows. As a special
case, the *categorical* NLL of a classifier with one-hot labels is precisely the
softmax cross-entropy loss of :numref:`sec_softmax`.

![Minimizing the negative log-likelihood is minimizing the KL divergence to the data. The per-example cross-entropy splits into the entropy $H(\hat p_{\textrm{data}})$, an irreducible floor set by the data's own randomness, plus $D_{\textrm{KL}}(\hat p_{\textrm{data}} \| p_{\boldsymbol{\theta}})$, the only part training can remove. As the model improves, the KL slice shrinks and the cross-entropy descends toward---but never below---the floor.](../img/mdl-prob-mle-kl.svg)
:label:`fig_mdl-mle-kl`

### From Probabilities to Densities

Everything so far was phrased for discrete outcomes, where $P(X\mid\boldsymbol{\theta})$
is a genuine probability. For continuous data we replace probabilities by
densities $p$, and the only worry is that the probability of observing *any
exact* real value is zero, so the naive likelihood is $0$ for every
$\boldsymbol{\theta}$. The resolution is to ask for a match only to within a small
tolerance $\epsilon$, then watch the $\epsilon$ cancel.

For i.i.d. observations $x_1,\ldots,x_n$, the probability that each lands in a
window of width $\epsilon$ is, to first order,

$$
P\big(X_i \in [x_i, x_i+\epsilon]\ \forall i \mid \boldsymbol{\theta}\big)
  \approx \epsilon^n \prod_{i=1}^n p(x_i\mid\boldsymbol{\theta}).
$$

Taking the negative log,

$$
-\log P\big(X_i \in [x_i, x_i+\epsilon]\ \forall i\mid\boldsymbol{\theta}\big)
  \approx -n\log\epsilon - \sum_{i} \log p(x_i\mid\boldsymbol{\theta}).
$$

The tolerance enters only through the additive constant $-n\log\epsilon$, which is
free of $\boldsymbol{\theta}$. Demanding four digits of precision or four hundred
changes that constant but never the minimizer, so we drop it and minimize

$$
-\sum_{i} \log p(x_i\mid\boldsymbol{\theta})
$$

exactly as in the discrete case. Maximum likelihood thus operates on continuous
variables by swapping probabilities for densities and nothing more---and the most
important density to swap in is the Gaussian, which we do next.

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

## Estimator Theory: Why Maximum Likelihood Works

A fair worry is whether $\hat{\boldsymbol{\theta}}$ is any good. The first
reassurance is the same KL picture. Drawing genuinely i.i.d. data from a true
distribution $p_{\boldsymbol{\theta}^\star}$, the average NLL is, by the law of
large numbers, an estimate of the *expected* cross-entropy
$\textrm{CE}(p_{\boldsymbol{\theta}^\star}, p_{\boldsymbol{\theta}})$, and the
decomposition $\textrm{CE} = H + D_{\textrm{KL}}$ shows that this population
objective is minimized exactly at the truth
$\boldsymbol{\theta}=\boldsymbol{\theta}^\star$, where the KL term vanishes.
"Exactly at" needs one hypothesis we have so far left implicit: the model must be
**identifiable**---distinct parameters must give distinct distributions,
$\boldsymbol{\theta}\neq\boldsymbol{\theta}'\Rightarrow p_{\boldsymbol{\theta}}\neq p_{\boldsymbol{\theta}'}$---or
else several parameters tie for the minimum and "the" true parameter is not even
well defined. This is no pedantic footnote: neural networks fail identifiability
spectacularly, since permuting the hidden units of a layer (together with their
weights) changes the parameter vector but not the function it computes, so for
such models consistency can only ever be a statement about the fitted
*distribution* $p_{\hat{\boldsymbol{\theta}}}$, never about the parameter vector
itself.

Minimizing the empirical average therefore targets the right minimizer. One more
dose of honesty: the law of large numbers makes the empirical objective converge
to the population one at each *fixed* $\boldsymbol{\theta}$, and pointwise
convergence of objectives does not by itself force their *minimizers* to
converge---for that, the convergence must be *uniform* over the parameter space.
With identifiability and uniform convergence in hand, the conclusion is a genuine
theorem (e.g., Theorem 9.13 of :cite:`Wasserman.2013`): for a well-specified
model the MLE is **consistent**, converging in probability to the true parameter,
$\hat{\boldsymbol{\theta}}\xrightarrow{P}\boldsymbol{\theta}^\star$. The coin
made this concrete: $\hat\theta = n_H/(n_H+n_T) \to \theta^\star$ directly by the
law of large numbers, and the Bernoulli family is identifiable---different biases
give different distributions. Consistency, asymptotic unbiasedness, and
efficiency are the estimator-quality notions of :numref:`sec_mdl-statistics`;
here they describe the MLE specifically.

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
$I(\boldsymbol{\theta}^\star)^{-1}$, and the Cramér--Rao bound stated below
(:eqref:`eq_mdl-cramer-rao`) says *no* unbiased estimator can have smaller
variance than $I(\boldsymbol{\theta}^\star)^{-1}/n$. The MLE attains that floor in
the limit: it is **asymptotically efficient**---asymptotically, the best estimator
there is. The qualifier matters. At finite $n$ the MLE is generally only
*asymptotically* unbiased, not exactly unbiased---the maximum-likelihood estimate
of a Gaussian's variance divides the sum of squares by $n$ rather than $n-1$ and
so systematically underestimates $\sigma^2$. :numref:`sec_mdl-statistics` works
out exactly this bias---identifying the $n$-divided estimator as the Gaussian
MLE---when it derives the $n-1$ correction. Maximum likelihood is optimal *in
the limit*, not a guarantee of unbiasedness at every sample size.

**Watching the theorem happen.** Asymptotic normality is the section's central
limit theorem, and like the CLT proper it can be watched. The cell below returns
to the coin: it draws $20{,}000$ independent datasets of $n=400$ flips with true
bias $\theta^\star=0.7$, computes the MLE $\hat\theta=n_H/n$ on each, and
histograms the rescaled errors $\sqrt n\,(\hat\theta-\theta^\star)$ against the
Gaussian that :eqref:`eq_mdl-asymptotic-normality` names. For the coin the
Fisher information works out below to $I(\theta)=1/(\theta(1-\theta))$, so the
predicted limit is $\mathcal N\bigl(0,\ \theta^\star(1-\theta^\star)\bigr)$ with
variance $0.21$---and the histogram does not merely look bell-shaped, it lands
on that *particular* Gaussian, center, width, and height. (One numerical nicety:
$\hat\theta$ lives on the lattice $k/n$, so we offset the bin edges by half a
lattice step to keep the discrete values off the edges.)

```{.python .input #mdl-maximum-likelihood-estimator-theory-why-maximum-likelihood-works}
rng = onp.random.default_rng(0)
theta_star, n, reps = 0.7, 400, 20000
flips = rng.random((reps, n)) < theta_star             # 20,000 datasets at once
z = onp.sqrt(n) * (flips.mean(axis=1) - theta_star)    # sqrt(n)(theta_hat - theta*)
sigma2 = theta_star * (1 - theta_star)                 # = 1 / I(theta*)
hist, edges = onp.histogram(z, bins=40, range=(-2.025, 1.975), density=True)
x = onp.arange(-2, 2, 0.01)
d2l.plot(x, [onp.interp(x, (edges[:-1] + edges[1:]) / 2, hist),
             onp.exp(-x**2 / (2 * sigma2)) / onp.sqrt(2 * onp.pi * sigma2)],
         xlabel='sqrt(n) * (theta_hat - theta*)', ylabel='density',
         legend=['20,000 replications', 'N(0, theta*(1-theta*))'])
```

### Fisher Information and the Score

To make :eqref:`eq_mdl-asymptotic-normality` quantitative we need the object
$I(\boldsymbol{\theta})$ it contains. Define the **score** as the gradient of the
log-likelihood with respect to the parameters,
$s(x;\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \log p(x\mid\boldsymbol{\theta})$.
The score is the per-example version of the NLL gradient we already met: setting
$\sum_i s(x_i;\boldsymbol{\theta}) = \mathbf 0$ is exactly the first-order condition
that locates $\hat{\boldsymbol{\theta}}$. At the true parameter the score has
mean zero, and its variance is the **Fisher information**,

$$
I(\boldsymbol{\theta}) = \operatorname{Var}_{\boldsymbol{\theta}}\!\big[\,s(x;\boldsymbol{\theta})\,\big]
  = \mathbb{E}_{\boldsymbol{\theta}}\!\big[\,s(x;\boldsymbol{\theta})\,s(x;\boldsymbol{\theta})^\top\,\big]
  = -\,\mathbb{E}_{\boldsymbol{\theta}}\!\big[\nabla^2_{\boldsymbol{\theta}} \log p(x\mid\boldsymbol{\theta})\big],
$$
:eqlabel:`eq_mdl-fisher`

the last equality (valid under the same regularity conditions) saying that the
information is the expected *curvature* of the NLL. The reading is intuitive: a
sharply peaked log-likelihood---large curvature, large $I$---pins the parameter
down tightly, so the estimate has small variance; a flat one leaves
$\boldsymbol{\theta}$ poorly determined. That is precisely what
:eqref:`eq_mdl-asymptotic-normality` says, $\operatorname{Var}(\hat{\boldsymbol{\theta}}) \approx I(\boldsymbol{\theta}^\star)^{-1}/n$.
:numref:`fig_mdl-prob-mle-fisher` draws the picture for a Gaussian's two
parameters: the NLL is a bowl around the MLE, and the inverse of its curvature is
the ellipse within which the estimate scatters.

![Fisher information is the curvature of the negative log-likelihood. Contours show the NLL for a Gaussian's parameters $(\mu, \sigma)$ over $n=30$ draws from $\mathcal{N}(1, 1.5^2)$; the dot marks the maximum-likelihood estimate, and the overlaid ellipse is the inverse-information covariance $I(\hat\mu,\hat\sigma)^{-1}/n$ that asymptotic normality predicts for the estimate's spread. Per observation $I = \operatorname{diag}(1/\sigma^2,\ 2/\sigma^2)$: the bowl is twice as curved in $\sigma$ as in $\mu$, so the data pin down the scale more tightly than the mean and the ellipse is correspondingly wider along the $\mu$-axis.](../img/mdl-prob-mle-fisher.svg)
:label:`fig_mdl-prob-mle-fisher`

The curvature intuition is a theorem about *every* estimator, not just the MLE.
The **Cramér--Rao bound** states that any *unbiased* estimator $\tilde\theta$
built from $n$ i.i.d. observations obeys

$$
\operatorname{Var}\big(\tilde\theta\big) \;\ge\; \frac{1}{n\,I(\theta)}
$$
:eqlabel:`eq_mdl-cramer-rao`

(stated here for a scalar parameter; for vectors the statement reads
$\operatorname{Cov}(\tilde{\boldsymbol{\theta}}) \succeq I(\boldsymbol{\theta})^{-1}/n$
in the positive-semidefinite order) :cite:`Wasserman.2013`. Information caps
precision: no cleverness in constructing the estimator can beat the curvature the
model itself supplies, and comparing with :eqref:`eq_mdl-asymptotic-normality`
shows the MLE's limiting variance meets this floor exactly---the precise content
of "asymptotically efficient."

The coin closes the loop on itself. A single flip has log-likelihood
$\log p(x\mid\theta) = x\log\theta + (1-x)\log(1-\theta)$ for $x\in\{0,1\}$, so
the score and its derivative are

$$
s(x;\theta) = \frac{x}{\theta} - \frac{1-x}{1-\theta},
\qquad
\frac{d}{d\theta}\, s(x;\theta) = -\frac{x}{\theta^2} - \frac{1-x}{(1-\theta)^2}.
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

Note that the per-flip information $1/(\theta(1-\theta))$ is *smallest* at
$\theta = 1/2$: fair coins are the hardest to pin down, each flip carrying the
least information about the bias. The prediction of
:eqref:`eq_mdl-asymptotic-normality` is that the MLE $\hat\theta = n_H/n$ over
$n$ flips has variance about $\theta(1-\theta)/n$---which, by
:eqref:`eq_mdl-coin-fisher`, is also the Cramér--Rao floor
:eqref:`eq_mdl-cramer-rao` for this problem. We can check that this is *exactly*
right here---$\hat\theta$ is a scaled binomial count, whose variance is
$\theta(1-\theta)/n$ on the nose---and confirm it by simulation: run the
$n$-flip experiment many times and compare the spread of the resulting estimates
to the floor.

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
not merely consistent, it is as tight as :eqref:`eq_mdl-cramer-rao`
permits. This is asymptotic efficiency made concrete, and it is
why the curvature of a loss surface---its Fisher information, or empirically its
Hessian---tells us how trustworthy a fitted parameter is.

## MAP Estimation: Priors as Regularizers
:label:`subsec_mdl-map`

We dropped the prior $P(\boldsymbol{\theta})$ by declaring it flat.
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

### Gaussian Priors Are Weight Decay

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
\to \hat{\boldsymbol{\theta}}_{\textrm{MLE}}$---the flat-prior assumption
we started with. And as the data grows the NLL bowl deepens faster than the fixed
prior, so the data term dominates and MAP again approaches the MLE. A bookkeeping
footnote makes the second limit quantitative: deep-learning code minimizes the
*mean* per-example loss rather than the sum, and dividing the MAP objective by
$n$ rescales the weight decay to $\lambda = \sigma^2/(n\tau^2)$---the prior's
pull on the averaged loss literally fades as $1/n$. Regularization matters most
precisely when data is scarce.

### A Beta Prior on the Coin
:label:`subsec_mdl-beta-map`

Weight decay is MAP for regression; the running coin deserves the same
treatment, and it closes the example for good. A prior on a coin's bias must
live on $[0,1]$, and the natural family is the **Beta distribution**
$p(\theta)\propto\theta^{a-1}(1-\theta)^{b-1}$ with shape parameters $a, b > 0$,
the conjugate prior of :numref:`sec_mdl-distributions`: multiplying by the
likelihood :eqref:`eq_mdl-coin-like` just shifts the exponents by the observed
counts, so the posterior is again a Beta (:eqref:`eq_mdl-beta_posterior`),

$$
P(\theta\mid X) \;\propto\; \theta^{\,n_H + a - 1}\,(1-\theta)^{\,n_T + b - 1}.
$$

MAP asks for the mode of this density. That is the same single-variable
maximization we solved for the likelihood itself---the posterior has the shape
$\theta^{\alpha}(1-\theta)^{\beta}$ with $\alpha = n_H+a-1$ and
$\beta = n_T+b-1$, and whenever both exponents are positive (e.g. $a, b \ge 1$
on our data) the maximum is interior---so the answer is the same ratio, "head
exponent over total exponent":

$$
\hat\theta_{\textrm{MAP}} \;=\; \frac{n_H + a - 1}{n + a + b - 2},
\qquad n = n_H + n_T.
$$
:eqlabel:`eq_mdl-beta-map`

The reading is **pseudo-counts**: the prior behaves exactly like $a-1$ phantom
heads and $b-1$ phantom tails appended to the data before taking the empirical
frequency. A $\mathrm{Beta}(2,2)$ prior adds one phantom flip of each kind,
turning our $9$-of-$13$ coin into $(9+1)/(13+2) = 2/3$, pulled from
$9/13\approx 0.692$ toward $1/2$; a $\mathrm{Beta}(30,30)$ prior would all but
pin the estimate near a fair coin until the data outweighed its $58$ phantom
flips. And as $n$ grows, any fixed stock of phantom flips washes out:
$\hat\theta_{\textrm{MAP}}\to\hat\theta_{\textrm{MLE}}$, the data-swamps-the-prior
limit we just met.

Two subtleties in :eqref:`eq_mdl-beta-map` repay attention, because they are
routinely garbled. First, the *flat* prior $\mathrm{Beta}(1,1)$---uniform on
$[0,1]$---gives $\hat\theta_{\textrm{MAP}} = n_H/n$: **MAP under a uniform prior
is exactly the MLE**, zero phantom flips, no smoothing at all. In particular the
"add-one" (Laplace) smoothing rule $(n_H+1)/(n+2)$, despite a common folklore
attribution, is *not* the uniform-prior MAP; read as pseudo-counts, it is the MAP
under $\mathrm{Beta}(2,2)$. Second, the mode is not the mean. The posterior mean
is

$$
\mathbb{E}[\theta\mid X] \;=\; \frac{n_H + a}{n + a + b},
$$

which under the flat prior becomes $(n_H+1)/(n+2)$---**Laplace's rule of
succession**, precisely the add-one rule. So add-one smoothing *is* a Bayes
estimate under the uniform prior, just a different summary: the posterior
**mean**, not the posterior **mode**. (Numerically it coincides with the
$\mathrm{Beta}(2,2)$ mode, which is why the two are so often conflated.) Naive
Bayes (:numref:`sec_mdl-naive_bayes`) leans on exactly this smoothing to keep
never-observed feature--class pairs from annihilating its products; the
pseudo-count accounting built here is what justifies it.

### The Posterior Mode Is Not the Posterior

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
Maximum likelihood does not suffer this wobble: it is **equivariant** under
reparameterization---if $\hat\theta$ is the MLE of $\theta$, then $g(\hat\theta)$
is the MLE of $g(\theta)$ for any one-to-one $g$, so the MLE of the coin's
log-odds is simply $\log\frac{9/13}{4/13} = \log\frac{9}{4}$
:cite:`Wasserman.2013`. The reason is that relabeling the parameter axis moves
no maximum: likelihood values are attached to points, while densities carry a
Jacobian---and MAP inherits the prior density's Jacobian, and with it the
dependence on how the parameter happens to be written down.
Marginalizing over the posterior is its own subject, beyond our scope here
:cite:`Murphy.2022,mackay2003information`; the point is only that MAP is one more
*point estimate*---maximum likelihood with a penalty---and should not be mistaken
for the full Bayesian treatment.

## Latent Variables, EM, and the ELBO
:label:`sec_mdl-latent-em-elbo`

Every likelihood so far has been a product of terms we could log and
differentiate directly. The most interesting generative models are not so
obliging. They posit a **latent variable** $z$---an unobserved cluster identity,
a topic, the "seed" a generator decodes into an image---and define the joint
distribution

$$
p(x, z; \boldsymbol{\theta}) = p(z; \boldsymbol{\theta})\, p(x \mid z; \boldsymbol{\theta}),
$$

of which we observe only $x$. The likelihood of an observation is then a
*marginal*, obtained by summing (for continuous latents, integrating) the latent
out:

$$
p(x;\boldsymbol{\theta}) = \sum_{z} p(z;\boldsymbol{\theta})\, p(x\mid z;\boldsymbol{\theta}).
$$
:eqlabel:`eq_mdl-marginal-lik`

### Why Latent Variables Break the Recipe

The canonical example is the **Gaussian mixture model** (GMM): the latent
$z \in \{1,\ldots,K\}$ picks a component with probability
$P(z = k) = \pi_k$, and the observation is then drawn from that component,
$x \mid z{=}k \sim \mathcal{N}(\mu_k, \sigma_k^2)$. The data's NLL is

$$
\ell(\boldsymbol{\theta})
  = -\sum_{i=1}^n \log\Big(\sum_{k=1}^K \pi_k\, \mathcal{N}(x_i;\, \mu_k, \sigma_k^2)\Big),
\qquad \boldsymbol{\theta} = (\pi_k, \mu_k, \sigma_k)_{k=1}^K.
$$

The trouble is the sum *inside* the logarithm. Until now every log reached an
individual density and the NLL split into clean per-example, per-parameter
pieces; here the log stops at the mixture, every parameter appears inside every
term through the shared sum, and no closed form solves the first-order
conditions. Contrast the *complete-data* problem: if an oracle revealed each
$z_i$, the likelihood would factor by component, and maximum likelihood would
collapse to the estimators we already own---the fraction of points in component
$k$ for $\pi_k$ (the coin), and the per-component sample mean and variance for
$\mu_k, \sigma_k^2$ (the Gaussian). Plain gradient descent on
$\ell(\boldsymbol{\theta})$ remains possible, but the structure of the
problem---easy if complete, hard only because $z$ is missing---suggests
something better: *guess the missing labels probabilistically, then solve the
easy problem.*

### The Evidence Lower Bound

Let $q(z)$ be any distribution over the latent with $q(z) > 0$ wherever
$p(x,z;\boldsymbol{\theta}) > 0$---our probabilistic guess. Insert it into the
marginal and apply Jensen's inequality---$\log$ is concave, so
$\log \mathbb{E}[\cdot] \ge \mathbb{E}[\log(\cdot)]$ (:numref:`sec_convexity`):

$$
\log p(x;\boldsymbol{\theta})
  = \log \sum_z q(z)\, \frac{p(x,z;\boldsymbol{\theta})}{q(z)}
  \;\ge\; \sum_z q(z) \log \frac{p(x,z;\boldsymbol{\theta})}{q(z)}
  \;=:\; \mathcal{L}(q, \boldsymbol{\theta}).
$$
:eqlabel:`eq_mdl-elbo`

The right-hand side is the **evidence lower bound** (**ELBO**). The name comes
from calling the marginal log-likelihood $\log p(x;\boldsymbol{\theta})$ the
*evidence*: it plays for the latent $z$ exactly the role the denominator $P(X)$
played for the parameters in Bayes' rule at the start of this section---the
marginal probability of what we actually observed. Expanding the fraction splits
the bound into two readable pieces,

$$
\mathcal{L}(q,\boldsymbol{\theta})
  = \mathbb{E}_{q}\big[\log p(x,z;\boldsymbol{\theta})\big] + H(q),
$$

the expected *complete-data* log-likelihood---the easy objective, averaged over
the guess---plus the entropy of the guess, which does not involve
$\boldsymbol{\theta}$ at all (:numref:`sec_mdl-information_theory`). How much
does the bound give away? Exactly a KL divergence:

**Proposition (the ELBO gap is a KL).** *For every $q$ and every
$\boldsymbol{\theta}$,*

$$
\log p(x;\boldsymbol{\theta})
  = \mathcal{L}(q,\boldsymbol{\theta})
  + D_{\textrm{KL}}\big(q(z)\,\|\,p(z\mid x;\boldsymbol{\theta})\big).
$$
:eqlabel:`eq_mdl-elbo-gap`

*Hence $\mathcal{L}(q,\boldsymbol{\theta}) \le \log p(x;\boldsymbol{\theta})$,
with equality if and only if $q$ is the posterior
$p(z\mid x;\boldsymbol{\theta})$.*

**Proof.** Expand the KL term using Bayes' rule,
$p(z\mid x;\boldsymbol{\theta}) = p(x,z;\boldsymbol{\theta})/p(x;\boldsymbol{\theta})$:

$$
D_{\textrm{KL}}\big(q \,\|\, p(z\mid x;\boldsymbol{\theta})\big)
  = \mathbb{E}_q\!\left[\log \frac{q(z)\, p(x;\boldsymbol{\theta})}{p(x,z;\boldsymbol{\theta})}\right]
  = \log p(x;\boldsymbol{\theta}) - \mathcal{L}(q,\boldsymbol{\theta}),
$$

where $\log p(x;\boldsymbol{\theta})$ exits the expectation because it does not
involve $z$. Rearranging gives :eqref:`eq_mdl-elbo-gap`; since the KL divergence
is nonnegative and vanishes exactly when its arguments agree, the bound and its
equality condition follow. $\blacksquare$

:numref:`fig_mdl-prob-elbo` draws the resulting geometry: for a fixed guess the
ELBO is a curve running everywhere below the evidence, and choosing $q$ well
pinches the two together at the current parameters.

![The evidence and its lower bound. For a fixed guess $q$, the ELBO $\mathcal{L}(q,\theta)$ runs below the evidence $\log p(x;\theta)$ at every $\theta$, and the gap between the curves is the KL divergence from $q$ to the posterior $p(z\mid x;\theta)$. The E-step of EM picks $q$ to be the posterior at the current iterate $\theta^{(t)}$, closing the gap so the bound touches the evidence there; the M-step climbs the bound to its peak $\theta^{(t+1)}$. Because the bound touches the evidence at $\theta^{(t)}$ and never exceeds it anywhere, the evidence at $\theta^{(t+1)}$ is at least what it was at $\theta^{(t)}$.](../img/mdl-prob-elbo.svg)
:label:`fig_mdl-prob-elbo`

### Expectation--Maximization

The proposition turns the intractable problem into **coordinate ascent** on
$\mathcal{L}(q, \boldsymbol{\theta})$. For a dataset $X = \{x_1,\ldots,x_n\}$ of
independent observations the bound applies example by example and sums, with one
guess $q_i$ per data point. The **expectation--maximization** (EM) algorithm
alternates the two coordinates :cite:`Dempster.Laird.Rubin.1977`:

* **E-step.** Hold $\boldsymbol{\theta}^{(t)}$ fixed and maximize over the
  guesses: by the equality condition of :eqref:`eq_mdl-elbo-gap`, the best
  $q_i$ is the posterior, $q_i^{(t)}(z) = p(z \mid x_i; \boldsymbol{\theta}^{(t)})$.
  This closes every gap, making the bound *tight*:
  $\sum_i \mathcal{L}(q_i^{(t)}, \boldsymbol{\theta}^{(t)}) = \log p(X;\boldsymbol{\theta}^{(t)})$.
* **M-step.** Hold the guesses fixed and maximize over the parameters:
  $\boldsymbol{\theta}^{(t+1)} = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}} \sum_i \mathbb{E}_{q_i^{(t)}}\big[\log p(x_i, z; \boldsymbol{\theta})\big]$
  (the entropy terms do not involve $\boldsymbol{\theta}$). This is the *easy*,
  complete-data objective, with soft assignments standing in for the oracle's
  labels.

**Proposition (EM never decreases the likelihood).** *Each EM iteration
satisfies $\log p(X;\boldsymbol{\theta}^{(t+1)}) \ge \log p(X;\boldsymbol{\theta}^{(t)})$.*

**Proof.** Chain three facts:

$$
\log p(X;\boldsymbol{\theta}^{(t+1)})
  \;\ge\; \sum_i \mathcal{L}\big(q_i^{(t)}, \boldsymbol{\theta}^{(t+1)}\big)
  \;\ge\; \sum_i \mathcal{L}\big(q_i^{(t)}, \boldsymbol{\theta}^{(t)}\big)
  \;=\; \log p(X;\boldsymbol{\theta}^{(t)}).
$$

The first inequality is the ELBO bound :eqref:`eq_mdl-elbo` at the new
parameters; the second is the definition of the M-step; the equality is the
tightness the E-step bought. $\blacksquare$

For the Gaussian mixture both steps are closed-form. The E-step computes
**responsibilities**, the posterior probability that point $i$ came from
component $k$,

$$
r_{ik} = \frac{\pi_k\, \mathcal{N}(x_i;\, \mu_k, \sigma_k^2)}
              {\sum_{j} \pi_j\, \mathcal{N}(x_i;\, \mu_j, \sigma_j^2)},
$$

and the M-step refits each component by responsibility-weighted counting and
averaging: with effective counts $n_k = \sum_i r_{ik}$,

$$
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{1}{n_k}\sum_{i} r_{ik}\, x_i, \qquad
\sigma_k^2 = \frac{1}{n_k}\sum_{i} r_{ik}\,(x_i - \mu_k)^2.
$$
:eqlabel:`eq_mdl-gmm-mstep`

These are the coin's counting estimator and the Gaussian's mean and variance
estimators, softened: instead of voting for one component, each point spreads
its single vote across components in proportion to its responsibilities
(exercise 10 derives the $\mu_k$ update). The cell below runs EM on a
two-component mixture in plain NumPy and prints the log-likelihood as it climbs.

```{.python .input #maximum-likelihood-em-gmm}
# EM for a two-component 1-D Gaussian mixture, in plain numpy.
rng = onp.random.default_rng(42)
# 300 points from N(-2, 0.8^2) and 200 from N(1.5, 0.6^2)
x = onp.concatenate([rng.normal(-2.0, 0.8, 300), rng.normal(1.5, 0.6, 200)])

def gauss(x, mu, sigma):
    return onp.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * onp.sqrt(2 * onp.pi))

# Deliberately poor start: equal weights, overlapping components
pi, mu, sigma = onp.array([0.5, 0.5]), onp.array([-0.5, 0.5]), onp.array([1.0, 1.0])
for step in range(31):
    # E-step: responsibilities r[i, k], the posterior of component k given x_i
    joint = pi * gauss(x[:, None], mu, sigma)
    r = joint / joint.sum(axis=1, keepdims=True)
    if step % 5 == 0:  # the log-likelihood never decreases
        print(f'step {step:2d}, log-likelihood {onp.log(joint.sum(axis=1)).sum():.2f}')
    # M-step: refit each component by responsibility-weighted averaging
    n_k = r.sum(axis=0)
    pi, mu = n_k / len(x), (r * x[:, None]).sum(axis=0) / n_k
    sigma = onp.sqrt((r * (x[:, None] - mu)**2).sum(axis=0) / n_k)
print(f'pi = {pi.round(3)}, mu = {mu.round(3)}, sigma = {sigma.round(3)}')
```

The trace behaves exactly as the proposition demands: the log-likelihood rises
monotonically, gains most of its ground in the first few iterations, and then
flatlines at a stationary point; the recovered weights, means, and scales land
near the generating values $(0.6, 0.4)$, $(-2, 1.5)$, $(0.8, 0.6)$, the small
residual gaps being finite-sample error, not a failure of EM. Two cautions
temper the success. EM is *ascent*, not global optimization: the mixture
likelihood is multimodal---at minimum because relabeling the components permutes
the parameters without changing the model, the same identifiability failure we
met in the estimator theory above---and EM can stall at a poor local maximum, so
practice restarts it from several initializations. And strictly speaking the GMM
likelihood is *unbounded*---let one component collapse onto a single data point
with $\sigma_k \to 0$---so the honest guarantee is convergence to a stationary
point, with degenerate spikes excluded by a floor on $\sigma_k$ or a prior.

The bound, not the closed-form M-step, is the part that scales. A *variational
autoencoder* keeps the ELBO but replaces the per-point E-step with a neural
network $q_{\boldsymbol{\phi}}(z\mid x)$---an *amortized* guess, trained jointly
with the generator by stochastic gradient ascent on the ELBO, because for a deep
decoder the true posterior is no longer available in closed form. Diffusion
models walk further down the same road: their training objective is the ELBO of
a deep chain of latent variables (:numref:`sec_mdl-score-matching-diffusion-flow`).
And the two divergences in this section are deliberately mirrored:
:numref:`sec_mdl-divergences-distances` recasts maximum likelihood itself as
*forward*-KL minimization, while the ELBO gap :eqref:`eq_mdl-elbo-gap` measures
the guess against the posterior in the *reverse* orientation,
$D_{\textrm{KL}}(q\,\|\,p)$---a distinction with real modelling consequences,
taken up in detail there.

## Summary

* The **maximum likelihood** principle picks the parameters that make the
  observed data most probable; with a flat prior, Bayes' rule reduces
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
* The MLE is **consistent** (given identifiability and uniform convergence) and,
  under regularity conditions, **asymptotically normal** with limiting variance
  the inverse **Fisher information** $I(\boldsymbol{\theta}^\star)^{-1}/n$---so
  it attains the Cramér--Rao bound :eqref:`eq_mdl-cramer-rao` and is
  *asymptotically efficient*. At finite $n$ it is generally only
  asymptotically unbiased (:numref:`sec_mdl-statistics`).
* **MAP** estimation restores the prior, adding its negative log as a penalty: a
  Gaussian prior is exactly $L_2$ / weight decay ($\lambda=\sigma^2/\tau^2$; for
  the averaged loss, $\sigma^2/(n\tau^2)$), a Laplace prior is $L_1$ / sparsity,
  and a Beta prior on the coin adds **pseudo-counts**---with MAP under the flat
  Beta(1,1) prior being the plain MLE, while add-one smoothing is the posterior
  *mean* (rule of succession). MAP $\to$ MLE as the prior flattens or the data
  grows, and it is still a *point estimate* (the posterior mode), not full
  Bayesian inference, which integrates over the posterior.
* When the model has **latent variables** the likelihood is a marginal with a
  sum inside the log. The **ELBO** lower-bounds it for every guess $q$, with gap
  exactly $D_{\textrm{KL}}(q \,\|\, p(z\mid x;\boldsymbol{\theta}))$; **EM**
  alternates closing the gap (E-step: $q$ = posterior) with climbing the bound
  (M-step), and never decreases the likelihood. VAEs and diffusion models train
  on the same bound with amortized guesses.
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
   bound :eqref:`eq_mdl-cramer-rao` exactly. (*Hint:* use :eqref:`eq_mdl-fisher`.)
9. Extend the asymptotic-normality demonstration in the text. First rerun it
   with $n = 10$ in place of $400$: how many distinct values can $\hat\theta$
   take, and what does the "histogram" become? Then restore $n = 400$ but move
   $\theta^\star$ to $0.95$ and compare the match to the predicted Gaussian.
   Explain why convergence is slower near the boundary, considering both the
   skew of the binomial there and the shrinking variance
   $\theta^\star(1-\theta^\star)$.
10. Derive the GMM M-step for the means: holding the responsibilities $r_{ik}$
    fixed, show that maximizing the expected complete-data log-likelihood
    $\sum_i \sum_k r_{ik}\big[\log \pi_k + \log \mathcal{N}(x_i;\,\mu_k,\sigma_k^2)\big]$
    over $\mu_k$ yields the responsibility-weighted mean of
    :eqref:`eq_mdl-gmm-mstep`. Then derive the $\pi_k$ update by maximizing over
    the weights subject to $\sum_k \pi_k = 1$. (*Hint:* use a Lagrange
    multiplier for the constraint.)


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

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.3]{.kicker}

The principle that turns a probabilistic model into a trainable loss<br>**maximum likelihood**.
:::
:::

::: {.slide title="Why maximum likelihood?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Pick the parameters that make the observed data most probable. It is the
single principle behind nearly every loss in the book.

- An **operational** recipe (the NLL you minimize).
- A **geometry** (a KL projection onto the model family).
- An **efficiency** guarantee (the Fisher / Cramér–Rao floor).
:::

::: {.col .fig}
@fig:mdl-prob-mle-kl
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The principle and the coin]{.dtitle}

[from Bayes to the likelihood, and the log trick]{.dsub}
:::
:::

::: {.slide title="Likelihood and the flat prior"}
[The principle]{.kicker}

Start from the posterior and drop what does not depend on $\boldsymbol\theta$:

$$\hat{\boldsymbol\theta} = \operatorname*{argmax}_{\boldsymbol\theta}
\frac{P(X\mid\boldsymbol\theta)\,P(\boldsymbol\theta)}{P(X)}.$$

$P(X)$ is constant; a **flat prior** drops $P(\boldsymbol\theta)$ too.

::: {.d2l-note .rule}
$\hat{\boldsymbol\theta}_{\text{MLE}} =
\operatorname*{argmax}_{\boldsymbol\theta} P(X\mid\boldsymbol\theta)$ —
maximum likelihood is MAP with no prior.
:::
:::

::: {.slide title="The coin: a likelihood surface"}
[The coin]{.kicker}

::: {.cols .vc}
::: {.col}
For $9$ heads and $4$ tails, $P(X\mid\theta)=\theta^9(1-\theta)^4$.
Setting the derivative to zero gives

$$\hat\theta = \frac{n_H}{n_H+n_T} = \frac{9}{13}.$$

The MLE of a coin's bias is always the observed frequency.
:::

::: {.col .fig}
@maximum-likelihood-a-concrete-example
:::
:::
:::

::: {.slide title="The negative log-likelihood"}
[The log trick]{.kicker}

A product of probabilities underflows; the log turns it into a sum and the
gradient into a per-example sum (so SGD works):

$$\ell(\boldsymbol\theta) = -\sum_{i=1}^n \log p(x_i\mid\boldsymbol\theta).$$

@!maximum-likelihood-numerical-optimization-and-the-negative-log-likelihood
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Maximum likelihood is minimizing a loss]{.dtitle}

[cross-entropy, mean squared error, and the KL projection]{.dsub}
:::
:::

::: {.slide title="MLE = minimum cross-entropy"}
[Equivalences]{.kicker}

Pool the data by outcome, $\hat p_{\text{data}}(x)=n_x/n$. Then

$$\tfrac1n\,\ell(\boldsymbol\theta) =
\operatorname{CE}\bigl(\hat p_{\text{data}}, p_{\boldsymbol\theta}\bigr)
= -\sum_x \hat p_{\text{data}}(x)\log p_{\boldsymbol\theta}(x).$$

. . .

*Proof.* Group identical observations and divide by $n>0$ — an
argmin-preserving rescaling. $\blacksquare$ So **maximizing likelihood is
minimizing cross-entropy to the data**.
:::

::: {.slide title="MLE is a KL projection"}
[KL geometry]{.kicker}

::: {.cols .vc}
::: {.col}
Cross-entropy splits into an irreducible floor plus a gap:

$$\operatorname{CE}(\hat p_{\text{data}}, p_{\boldsymbol\theta})
= H(\hat p_{\text{data}}) + D_{\mathrm{KL}}\bigl(\hat p_{\text{data}}\,\|\,p_{\boldsymbol\theta}\bigr).$$

Training removes only the KL term, so the MLE is the model **closest in
KL** to the empirical distribution.
:::

::: {.col .fig}
@fig:mdl-prob-mle-kl
:::
:::
:::

::: {.slide title="Gaussian NLL = mean squared error"}
[Loss ↔ noise]{.kicker}

With a fixed-variance Gaussian noise model,

$$-\log\!\prod_i \mathcal N(y_i;\hat y_i,\sigma^2)
= \frac{1}{2\sigma^2}\sum_i (y_i-\hat y_i)^2 + \tfrac{n}{2}\log(2\pi\sigma^2).$$

. . .

The second term is $\boldsymbol\theta$-free, so the argmin is least
squares.

::: {.d2l-note .rule}
Choosing a loss **is** choosing a noise model: Gaussian → MSE,
Bernoulli → BCE, categorical → CE, Laplace → MAE.
:::
:::

::: {.slide title="From probabilities to densities"}
[Continuous data]{.kicker}

For a continuous $x_i$, the probability of the $\epsilon$-window is
$\approx \epsilon\,p(x_i\mid\boldsymbol\theta)$, contributing a constant
$-n\log\epsilon$ to the NLL.

. . .

That constant is $\boldsymbol\theta$-free and drops — the same NLL machine
runs unchanged on densities.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Why maximum likelihood works]{.dtitle}

[consistency, Fisher information, the Cramér–Rao floor]{.dsub}
:::
:::

::: {.slide title="Consistency"}
[Estimator theory]{.kicker}

The population objective is $\operatorname{CE}(p_{\theta^\star},p_{\boldsymbol\theta})$,
minimized where the KL vanishes — at the truth $\theta^\star$. With
identifiability and regularity, $\hat{\boldsymbol\theta}\xrightarrow{P}\theta^\star$.

::: {.d2l-note}
Networks are typically consistent *as distributions*, not as parameter
vectors — many weight settings express the same function.
:::
:::

::: {.slide title="Asymptotic normality, watched"}
[Estimator theory]{.kicker}

Consistency says *where* $\hat{\boldsymbol\theta}$ lands; the sharper
statement is *how tightly*:
$\sqrt{n}\,(\hat{\boldsymbol\theta}-\boldsymbol\theta^\star)
\xrightarrow{d} \mathcal N(\mathbf 0,\, I(\boldsymbol\theta^\star)^{-1})$.
Twenty thousand coin datasets ($n=400$, $\theta^\star=0.7$), rescaled:

@!mdl-maximum-likelihood-estimator-theory-why-maximum-likelihood-works

The histogram does not merely look bell-shaped — it lands on the
*particular* Gaussian the theorem names, $\mathcal N(0,\,0.21)$: center,
width, and height. Halving the error costs four times the data.
:::

::: {.slide title="Fisher information"}
[Curvature = information]{.kicker}

::: {.cols .vc}
::: {.col}
The score is $s = \nabla_{\boldsymbol\theta}\log p$; the Fisher information
is its variance, equivalently the expected NLL curvature:

$$I(\boldsymbol\theta) = \operatorname{Var}[s] =
-\,\mathbb E\bigl[\nabla^2\log p\bigr].$$

A sharper NLL bowl pins the estimate down tighter. For the coin,
$I(\theta)=1/(\theta(1-\theta))$.
:::

::: {.col .fig}
@fig:mdl-prob-mle-fisher
:::
:::
:::

::: {.slide title="Cramér–Rao: the efficiency floor"}
[Curvature = information]{.kicker}

No unbiased estimator beats the inverse Fisher information, and the MLE
attains it asymptotically:

$$\operatorname{Var}(\tilde\theta) \ge \frac{1}{n\,I(\theta)}, \qquad
\sqrt{n}\,(\hat{\boldsymbol\theta}-\theta^\star) \xrightarrow{d}
\mathcal N\bigl(\mathbf 0, I(\theta^\star)^{-1}\bigr).$$

A simulation lands right on the floor:

@!maximum-likelihood-fisher-information
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[MAP: priors as regularizers]{.dtitle}

[weight decay, sparsity, pseudo-counts]{.dsub}
:::
:::

::: {.slide title="A Gaussian prior is weight decay"}
[MAP]{.kicker}

::: {.cols .vc}
::: {.col}
Keep the prior; its negative log is a penalty on the NLL. A
$\mathcal N(\mathbf 0,\tau^2 I)$ prior contributes
$\tfrac{1}{2\tau^2}\|\boldsymbol\theta\|_2^2$:

$$\hat{\boldsymbol\theta}_{\text{MAP}} =
\operatorname*{argmin}_{\boldsymbol\theta}\Bigl[
-\textstyle\sum_i\log p(x_i\mid\boldsymbol\theta) +
\tfrac{1}{2\tau^2}\|\boldsymbol\theta\|_2^2\Bigr].$$

::: {.d2l-note}
With Gaussian data this is **ridge regression**,
$\lambda=\sigma^2/\tau^2$; a Laplace prior gives $\ell_1$ / sparsity.
:::
:::

::: {.col .fig}
@fig:mdl-prob-map-prior
:::
:::
:::

::: {.slide title="A Beta prior on the coin"}
[MAP]{.kicker}

A $\text{Beta}(a,b)$ prior shifts the optimum by pseudo-counts:

$$\hat\theta_{\text{MAP}} = \frac{n_H + a - 1}{n + a + b - 2}.$$

$\text{Beta}(1,1)$ recovers the MLE; as $n$ grows the prior washes out.

::: {.d2l-note .rule}
MAP returns the posterior **mode** — a point estimate, not the full
posterior; the posterior mean differs (Laplace's rule).
:::
:::

::: {.slide}
::: {.divider}
[05]{.dnum}

[Latent variables, the ELBO, and EM]{.dtitle}

[Jensen's bound and coordinate ascent]{.dsub}
:::
:::

::: {.slide title="Why latents break the recipe"}
[Latent variables]{.kicker}

With latents $z$ the likelihood is a marginal,
$p(x;\boldsymbol\theta)=\sum_z p(x,z;\boldsymbol\theta)$ — a **sum inside
the log** (mixtures, VAEs, diffusion) that couples all parameters.

::: {.d2l-note}
The *complete-data* problem (if we knew $z$) would be easy; the plan is to
use that easy problem as a stepping stone.
:::
:::

::: {.slide title="The evidence lower bound"}
[ELBO]{.kicker}

::: {.cols .vc}
::: {.col}
For any $q(z)$, Jensen gives a lower bound whose gap is a KL:

$$\log p(x;\boldsymbol\theta) =
\underbrace{\mathbb E_q[\log p(x,z;\boldsymbol\theta)] + H(q)}_{\mathcal L(q,\boldsymbol\theta)}
+ D_{\mathrm{KL}}\bigl(q\,\|\,p(z\mid x)\bigr).$$

The bound is tight exactly when $q$ is the posterior $p(z\mid x)$.
:::

::: {.col .fig}
@fig:mdl-prob-elbo
:::
:::
:::

::: {.slide title="EM: coordinate ascent on the ELBO"}
[EM]{.kicker}

**E-step** $q_i = p(z\mid x_i;\boldsymbol\theta^{(t)})$ closes the gap;
**M-step** maximizes the expected complete-data log-likelihood. Each step
raises the evidence, so it never decreases:

@!maximum-likelihood-em-gmm

::: {.d2l-note .rule}
The same bound VAEs and diffusion models are trained on — EM is its
exact-posterior special case.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- MLE = maximize $\sum_i\log p(x_i\mid\boldsymbol\theta)$ = minimize the NLL.
- Average NLL = cross-entropy to the data = KL projection; consistent and asymptotically efficient (Fisher / Cramér–Rao).
- Most DL losses are NLLs of a noise model; MSE is the Gaussian case.
:::

::: {.col}
- Priors become regularizers: Gaussian → weight decay, Laplace → sparsity, Beta → pseudo-counts.
- Latents: the ELBO lower-bounds the evidence with a KL gap; EM closes it then climbs.
- One principle threads classifiers, regressors, and generative models.
:::
:::
:::
