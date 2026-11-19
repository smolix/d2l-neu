# Latent Variables, EM, and the ELBO
:label:`sec_mdl-latent-em-elbo`

Every likelihood we met in :numref:`sec_mdl-maximum_likelihood` was a clean
product of factors, one per observation, so taking logarithms turned it into a
sum we could differentiate term by term. That convenience rested on a quiet
assumption: that we get to observe everything the model generates.

In practice, we often miss a crucial piece of the data. Imagine collecting
$500$ measurements drawn from two distinct populations, $300$ from one and
$200$ from the other, only to discover that nobody recorded which population
produced which measurement. If those labels had survived, fitting the model
would take one line per population: a mixing fraction, a mean, and a variance.
Without them, the probability of each measurement becomes a sum over the two
populations it might have come from. Unfortunately, the logarithm of a sum
does not split apart; every parameter ends up entangled in every term, and
setting the gradient to zero leaves equations with no closed-form solution.
The same obstacle appears whenever a model explains observations through
hidden structure: the cluster of a customer, the topic of a document, or the
random code a generator turns into an image. Such unobserved quantities are
called *latent variables*.

The central idea of this section is simple and surprisingly effective: guess
what we cannot see using probabilities rather than hard labels, and then fit
the model as if those soft guesses were real data. What makes this idea work
is that our guess does not even have to be right. For *any* probabilistic
guess, the resulting easy objective lies below the true log-likelihood, and it
touches the log-likelihood precisely when the guess equals the posterior
distribution over the hidden labels. This lower bound is the key result to
remember. Expectation--maximization climbs it directly, while variational
inference and variational autoencoders climb it approximately whenever the
exact posterior is out of reach.

We will develop this one idea from start to finish: first setting up mixture
models, then deriving the lower bound and measuring its gap, and finally
running the resulting algorithm on our $500$ points, once from a starting point
where it recovers both populations and once from a symmetric start where it
gets stuck.


```{.python .input #mdl-latent-em-elbo-latent-variables-em-and-the-elbo}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-latent-em-elbo-latent-variables-em-and-the-elbo}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-latent-em-elbo-latent-variables-em-and-the-elbo}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-latent-em-elbo-latent-variables-em-and-the-elbo}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Mixture Models

Let us translate this missing-label puzzle into symbols. A probabilistic model
with a latent variable $z$ specifies a joint distribution over the hidden and
observed quantities,

$$
p(x, z\mid\boldsymbol{\theta}) = p(z\mid\boldsymbol{\theta})\, p(x \mid z, \boldsymbol{\theta}),
$$

of which we observe only $x$. To compute the likelihood of what we actually
saw, we must sum out all possible values of the hidden variable (or integrate
them out when $z$ is continuous), giving the *marginal* likelihood:

$$
p(x\mid\boldsymbol{\theta}) = \sum_{z} p(z\mid\boldsymbol{\theta})\, p(x\mid z, \boldsymbol{\theta}).
$$
:eqlabel:`eq_mdl-marginal-lik`

In words, the model's probability for $x$ adds up every way the model could
have generated $x$, one term for each value of the hidden variable.

The canonical example is the **Gaussian mixture model** (GMM). Here the latent
variable $z \in \{1,\ldots,K\}$ selects component $k$ with mixing probability
$\pi_k$, and the observation is then drawn from that component's Gaussian,
$x \mid z{=}k \sim \mathcal{N}(\mu_k, \sigma_k^2)$. Putting the $n$
observations together, the negative log-likelihood (NLL) is

$$
\ell(\boldsymbol{\theta})
  = -\sum_{i=1}^n \log\Big(\sum_{k=1}^K \pi_k\, \mathcal{N}(x_i;\, \mu_k, \sigma_k^2)\Big),
\qquad \boldsymbol{\theta} = (\pi_k, \mu_k, \sigma_k)_{k=1}^K.
$$

All the trouble comes from the sum sitting *inside* the logarithm. The
logarithm stops at the mixture and never reaches the individual Gaussian
densities, where it would otherwise cancel the exponential and leave a simple
quadratic. Compare this with the *complete-data* problem: if an oracle handed
us every label $z_i$, the likelihood would separate cleanly by component, and
maximum likelihood would reduce to estimators we already know. Each mixing
weight $\pi_k$ would simply be the fraction of points belonging to component
$k$, just like the coin-bias estimator $\hat\theta = n_H/n$ in
:numref:`sec_mdl-maximum_likelihood`, while $\mu_k$ and $\sigma_k^2$ would be
the ordinary sample mean and variance of those points. Because the problem is
hard only when the labels are missing, a natural strategy suggests itself:
estimate the missing labels as probabilities, and then solve the easy
complete-data problem.

What does such a probabilistic guess look like for a single point? Consider
one observation at $x = 0.5$ under two equally weighted components,
$\mathcal{N}(-1, 1)$ and $\mathcal{N}(1, 1)$. Their densities at $x = 0.5$ are
proportional to $e^{-(1.5)^2/2} = e^{-1.125}$ and
$e^{-(0.5)^2/2} = e^{-0.125}$, so Bayes' rule assigns the left component the
posterior probability

$$
\frac{e^{-1.125}}{e^{-1.125}+e^{-0.125}} = \frac{1}{1+e} = 0.27.
$$

Rather than forcing a hard choice, Bayes' rule splits the point $27\%$ to
$73\%$ between the two components. Move the observation far to the right, say
to $x = 3$, and the posterior becomes $99.8\%$ certain it belongs to the right
component; place it at the midpoint $x = 0$, and the split is an even
$50\%$--$50\%$. The soft guess is confident where the data are clear and
appropriately undecided where the components overlap.

## The Evidence Lower Bound

To turn this intuitive strategy into an algorithm with a mathematical
guarantee, we construct a lower bound on the log-likelihood in which the
logarithm moves back inside the sum. Let $q(z)$ be *any* probability
distribution over the latent variable, representing our soft guess. We
multiply and divide inside the marginal likelihood by $q(z)$ and apply
Jensen's inequality: because the logarithm is concave,
$\log \mathbb{E}[\cdot] \ge \mathbb{E}[\log(\cdot)]$
(:numref:`subsec_mdl-jensen`). This yields

$$
\log p(x\mid\boldsymbol{\theta})
  = \log \sum_z q(z)\, \frac{p(x,z\mid\boldsymbol{\theta})}{q(z)}
  \;\ge\; \sum_z q(z) \log \frac{p(x,z\mid\boldsymbol{\theta})}{q(z)}
  \;=:\; \mathcal{L}(q, \boldsymbol{\theta}).
$$
:eqlabel:`eq_mdl-elbo`

Here the sums run over all $z$ with $q(z) > 0$. If $q$ happens to put zero
probability on some $z$ where the joint distribution is positive, the first
equality simply becomes $\ge$, so the bound still holds. Notice that the
marginal $p(x\mid\boldsymbol{\theta})$ plays the same role for the latent
variable $z$ that the evidence $P(X)$ (the denominator of Bayes' rule in
:numref:`subsec_mdl-map`) plays for model parameters: it is the probability of
the observed data after summing out everything unobserved. For that reason it
is also called the *evidence*, and the right-hand side
$\mathcal{L}(q, \boldsymbol{\theta})$ is called the **evidence lower bound**
(**ELBO**). Splitting the logarithm of the ratio separates the ELBO into two
transparent pieces:

$$
\mathcal{L}(q,\boldsymbol{\theta})
  = \mathbb{E}_{q}\big[\log p(x,z\mid\boldsymbol{\theta})\big] + H(q).
$$

The first term is the expected *complete-data* log-likelihood, our easy
objective averaged over the soft guess $q$. The second term is the entropy of
the guess, which does not depend on $\boldsymbol{\theta}$ at all. Even better,
we can measure the exact gap between the ELBO and the true log-likelihood: it
is a KL divergence, and both EM and variational inference rest on this single
identity.

**Proposition (the ELBO gap is a KL divergence).** *For every
$\boldsymbol{\theta}$ and every distribution $q$ that puts no mass outside the
support of the posterior $p(z\mid x,\boldsymbol{\theta})$,*

$$
\log p(x\mid\boldsymbol{\theta})
  = \mathcal{L}(q,\boldsymbol{\theta})
  + D_{\textrm{KL}}\big(q(z)\,\|\,p(z\mid x,\boldsymbol{\theta})\big).
$$
:eqlabel:`eq_mdl-elbo-gap`

*Hence $\mathcal{L}(q,\boldsymbol{\theta}) \le \log p(x\mid\boldsymbol{\theta})$,
with equality if and only if $q$ equals the posterior
$p(z\mid x,\boldsymbol{\theta})$.*

**Proof.** Substitute Bayes' rule,
$p(z\mid x,\boldsymbol{\theta}) = p(x,z\mid\boldsymbol{\theta})/p(x\mid\boldsymbol{\theta})$,
directly into the definition of the KL divergence:

$$
D_{\textrm{KL}}\big(q \,\|\, p(z\mid x,\boldsymbol{\theta})\big)
  = \mathbb{E}_q\!\left[\log \frac{q(z)\, p(x\mid\boldsymbol{\theta})}{p(x,z\mid\boldsymbol{\theta})}\right]
  = \log p(x\mid\boldsymbol{\theta}) - \mathcal{L}(q,\boldsymbol{\theta}),
$$

where $\log p(x\mid\boldsymbol{\theta})$ pulls straight out of the expectation
because it does not depend on $z$. Rearranging gives
:eqref:`eq_mdl-elbo-gap`. By Gibbs' inequality
:eqref:`eq_mdl-exp_family_gibbs`, the KL divergence is always nonnegative and
vanishes if and only if its two arguments match. $\blacksquare$

Let us check this identity on our concrete example $x = 0.5$ from the previous
subsection. There the true log-evidence is $\log p(x) = -1.42$. If we use the
undecided guess $q = (\tfrac{1}{2}, \tfrac{1}{2})$, the ELBO evaluates to
$-1.54$, and the missing $0.12$ nats are the exact KL divergence from
$(\tfrac{1}{2}, \tfrac{1}{2})$ to the posterior $(0.27, 0.73)$. In other
words, the bound falls short of the log-evidence by the mismatch between our
guess and the true posterior, and by nothing else. :numref:`fig_mdl-prob-elbo`
illustrates how this lower bound behaves as $\boldsymbol{\theta}$ varies.

![The evidence and its lower bound. For a fixed guess $q$, the ELBO $\mathcal{L}(q,\theta)$ lies below the evidence $\log p(x\mid\theta)$ at every $\theta$, and the vertical gap between the curves is the KL divergence from $q$ to the posterior $p(z\mid x,\theta)$. In EM, the E-step sets $q$ to the posterior at the current iterate $\theta^{(t)}$, closing the gap so the bound touches the evidence there; the M-step then climbs the bound to its peak $\theta^{(t+1)}$. Because the bound touches the evidence at $\theta^{(t)}$ and never rises above it, the evidence at $\theta^{(t+1)}$ is guaranteed to be at least as high as at $\theta^{(t)}$.](../img/mdl-prob-elbo.svg)
:label:`fig_mdl-prob-elbo`

## Expectation--Maximization

The ELBO identity turns our difficult optimization problem into **coordinate
ascent** on $\mathcal{L}(q, \boldsymbol{\theta})$: maximize with respect to
one argument while holding the other fixed, swap roles, and repeat. Given a
dataset $X = \{x_1,\ldots,x_n\}$ of independent observations, the bound
applies to each observation separately and sums across the dataset, with one
guess $q_i$ per data point. The **expectation--maximization** (EM) algorithm
alternates between these two coordinates :cite:`Dempster.Laird.Rubin.1977`:

* **E-step.** Hold $\boldsymbol{\theta}^{(t)}$ fixed and maximize over the
  guesses $q_i$. By the equality condition in :eqref:`eq_mdl-elbo-gap`, the
  optimal choice for every point is its exact posterior,
  $q_i^{(t)}(z) = p(z \mid x_i, \boldsymbol{\theta}^{(t)})$. This drives every
  KL gap to zero and makes the bound *tight* at the current parameters:
  $\sum_i \mathcal{L}(q_i^{(t)}, \boldsymbol{\theta}^{(t)}) = \log P(X\mid\boldsymbol{\theta}^{(t)})$.
* **M-step.** Hold the guesses $q_i^{(t)}$ fixed and maximize over the
  parameters $\boldsymbol{\theta}$:

  $$
  \boldsymbol{\theta}^{(t+1)} = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}} \sum_i \mathbb{E}_{q_i^{(t)}}\big[\log p(x_i, z\mid\boldsymbol{\theta})\big],
  $$

  where the entropy terms $H(q_i^{(t)})$ drop out because they do not depend
  on $\boldsymbol{\theta}$. We are back to the easy complete-data objective,
  with soft probabilities standing in for the missing oracle labels.

Why can we trust this alternating procedure? Because the geometry in
:numref:`fig_mdl-prob-elbo` guarantees that every single step improves (or at
worst preserves) the true log-likelihood.

**Proposition (EM never decreases the likelihood).** *Every EM iteration
satisfies $\log P(X\mid\boldsymbol{\theta}^{(t+1)}) \ge \log P(X\mid\boldsymbol{\theta}^{(t)})$.*

**Proof.** Chain together three one-line facts:

$$
\log P(X\mid\boldsymbol{\theta}^{(t+1)})
  \;\ge\; \sum_i \mathcal{L}\big(q_i^{(t)}, \boldsymbol{\theta}^{(t+1)}\big)
  \;\ge\; \sum_i \mathcal{L}\big(q_i^{(t)}, \boldsymbol{\theta}^{(t)}\big)
  \;=\; \log P(X\mid\boldsymbol{\theta}^{(t)}).
$$

The first inequality is the ELBO bound :eqref:`eq_mdl-elbo` evaluated at the
new parameters, the second holds because the M-step maximizes the bound over
$\boldsymbol{\theta}$, and the final equality is the tightness secured by the
E-step. $\blacksquare$

For our Gaussian mixture model, both steps come out in closed form. The E-step
computes the **responsibilities**, the posterior probabilities that point $i$
was generated by component $k$, just as we did by hand for $x = 0.5$:

$$
r_{ik} = \frac{\pi_k\, \mathcal{N}(x_i;\, \mu_k, \sigma_k^2)}
              {\sum_{j} \pi_j\, \mathcal{N}(x_i;\, \mu_j, \sigma_j^2)}.
$$

The M-step then refits each component using responsibility-weighted counting
and averaging. Writing $n_k = \sum_i r_{ik}$ for the effective number of
points assigned to component $k$, we obtain

$$
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{1}{n_k}\sum_{i} r_{ik}\, x_i, \qquad
\sigma_k^2 = \frac{1}{n_k}\sum_{i} r_{ik}\,(x_i - \mu_k)^2.
$$
:eqlabel:`eq_mdl-gmm-mstep`

Notice how familiar these formulas are: they are the ordinary complete-data
estimators (mixing fraction, sample mean, and sample variance), simply softened.
Instead of casting one hard vote for a single component, each data point
divides its vote across the components according to its responsibilities. In
fact, if we force all components to share a single fixed variance $\sigma^2$
and shrink $\sigma^2 \to 0$, the responsibilities harden into $0/1$
assignments to the nearest mean, and EM reduces directly to the K-means
algorithm.

The entire algorithm takes only a dozen lines of code. Let us generate $300$
points from $\mathcal{N}(-2, 0.8^2)$ and $200$ from $\mathcal{N}(1.5, 0.6^2)$
and write a short function `em` for a one-dimensional mixture of $K$ Gaussians
that records the log-likelihood at every step alongside the final fitted
parameters.

```{.python .input #maximum-likelihood-em-gmm}
rng = np.random.default_rng()
x = np.concatenate([rng.normal(-2.0, 0.8, 300), rng.normal(1.5, 0.6, 200)])

def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def em(x, pi, mu, sigma, steps):
    """EM for a 1-D Gaussian mixture: log-likelihood trace and final fit."""
    pi, mu, sigma = (np.array(v, dtype=float) for v in (pi, mu, sigma))
    lls = []
    for _ in range(steps):
        joint = pi * gauss(x[:, None], mu, sigma)  # (n, K)
        lls.append(np.log(joint.sum(axis=1)).sum())
        r = joint / joint.sum(axis=1, keepdims=True)  # E-step
        n_k = r.sum(axis=0)  # M-step
        pi, mu = n_k / len(x), (r * x[:, None]).sum(axis=0) / n_k
        sigma = np.sqrt((r * (x[:, None] - mu)**2).sum(axis=0) / n_k)
    return np.array(lls), pi, mu, sigma
```

Our proposition guarantees that the log-likelihood will never decrease, but it
says nothing about how quickly EM climbs. Let us test it from a rather poor
initial guess, two heavily overlapping unit-variance components centered at
$\pm 0.5$, run $30$ iterations, and inspect the first six log-likelihood values
alongside the final step.

```{.python .input #mdl-maximum-likelihood-expectation-maximization-1}
lls, pi, mu, sigma = em(x, [0.5, 0.5], [-0.5, 0.5], [1, 1], 30)
print('steps 0-5:', lls[:6].round(1), ' step 29:', lls[-1].round(1))
print(f'pi = {pi.round(2)}, mu = {mu.round(2)}, sigma = {sigma.round(2)}')
```

The log-likelihood climbs by more than $400$ nats, rising from roughly
$-1{,}300$ to between $-850$ and $-890$ (the exact figures vary with the
random sample). Remarkably, almost the entire gain happens immediately: after
just six steps the log-likelihood is already within about a nat of its final
value, and the recovered weights, means, and scales land within about $0.2$ of
the true generating parameters $(0.6, 0.4)$, $(-2, 1.5)$, and $(0.8, 0.6)$.

Despite this clean convergence, EM comes with four limitations worth knowing.
The first is harmless: permuting the labels of the $K$ components produces
$K!$ identical copies of every maximum. This *label switching* is the same
non-identifiability we encounter when permuting the hidden units of a neural
network, and it does not make any solution worse than another.

The second limitation is more consequential: EM stops at the first stationary
point it hits, whether or not that point is a global maximum. A symmetric
initialization illustrates this sharply. Before running the next cell, pause
and predict what EM will do if both components start with the exact same mean
and scale.

```{.python .input #mdl-maximum-likelihood-expectation-maximization-2}
lls2, pi2, mu2, sigma2 = em(x, [0.5, 0.5], [0, 0], [1, 1], 30)
print(f'log-likelihood {lls2[0]:.1f} -> {lls2[-1]:.1f}, '
      f'{lls[-1] - lls2[-1]:.1f} below the previous fit')
print(f'pi = {pi2.round(2)}, mu = {mu2.round(2)}, sigma = {sigma2.round(2)}')
print(f'mean of x = {x.mean():.2f}, standard deviation of x = {x.std():.2f}')
```

The two components never break apart. Because they begin identically, they
receive identical responsibilities on every single data point, so the M-step
keeps them locked together. EM settles into the single-Gaussian fit, just the
overall mean and standard deviation of the entire dataset, more than $100$ nats
below the two-cluster solution. Every fixed point of EM is a stationary point
of the likelihood, not necessarily a maximum, which is why practitioners
restart EM from several random initializations. Random initialization breaks
this exact same symmetry among hidden units in a neural network layer
(:numref:`sec_numerical_stability`): units that start equal receive equal
gradients and remain equal forever.

The third limitation is a quirk of continuous mixtures: the likelihood itself
is unbounded. Place the second component directly on top of the first data
point, $\mu_2 = x_1$, with weight $\pi_2 = 1/n$ and standard deviation
$\epsilon$. Its density at $x_1$ is $\tfrac{1}{n}(2\pi)^{-1/2}\epsilon^{-1}$,
while the first component continues to cover the remaining $n - 1$ points,
giving

$$
\ell(\boldsymbol{\theta}) \le \log\big(n\sqrt{2\pi}\big) + \log\epsilon + \textrm{const},
$$

where the constant gathers the contribution of the other $n - 1$ points. As
$\epsilon \to 0$, the negative log-likelihood plunges to $-\infty$: the global
maximum-likelihood "solution" is simply an infinitely sharp spike perched on a
single data point. Imposing a lower floor on the variances, or placing a prior
over them, eliminates these degenerate spikes. Once the set of parameters at
least as good as the starting point is compact, every limit point of EM is
guaranteed to be a stationary point of the regularized problem
:cite:`Wu.1983`.

The fourth limitation is what leads beyond EM altogether. Our monotonicity
proof depended on plugging the *exact* posterior into the E-step; whenever we
must settle for an approximate E-step, that guarantee disappears. A
*variational autoencoder* :cite:`Kingma.Welling.2014`, for example, keeps the
very same ELBO objective but replaces the exact per-point posterior with an
encoder neural network $q_{\boldsymbol{\phi}}(z\mid x)$ trained jointly with
the decoder, because a deep nonlinear generator has no closed-form posterior.
Instead of exact EM steps, it climbs the ELBO using stochastic gradients.

Here the bound outgrows the algorithm that introduced it. Nothing in the ELBO
requires the hidden variable $z$ to be a cluster label: if we put the model
parameters themselves in place of $z$, the posterior that the E-step asks for
is the posterior over parameters, the very object we were left wanting at the
end of :numref:`sec_mdl-maximum_likelihood`. That posterior is rarely
available in closed form either, so :numref:`sec_mdl-bayesian-computation`
turns to the practical methods we use to approximate and sample from it.

## Summary

When a model includes hidden variables, its marginal likelihood traps a sum
inside a logarithm, turning what would have been simple counting and averaging
into an optimization problem with no closed form. Introducing a probabilistic
guess $q$ over the hidden variables restores the easy complete-data objective
as a lower bound: the ELBO falls below the log-evidence by exactly the KL
divergence from $q$ to the true latent posterior. EM alternates between closing
that gap (the E-step) and maximizing the resulting bound over the parameters
(the M-step). Although every iteration is guaranteed never to decrease the
likelihood, EM can stop at non-optimal stationary points, requires a variance
floor or prior to prevent degenerate spikes in continuous mixtures, and needs
an exact posterior to keep its monotonicity guarantee.

## Exercises

1. [code] **EM speed and missing information.** ● Consider the mixture
   $p(x\mid\pi)=\pi f_0(x)+(1-\pi)f_1(x)$ with known components
   $f_0=\mathcal N(\cdot\,;0,1)$ and $f_1=\mathcal N(\cdot\,;\Delta,1)$, in
   which only the mixing weight $\pi$ is unknown.
    1. Show that EM updates the weight by $\pi^{(t+1)}=M(\pi^{(t)})$ with
       $M(\pi)=\tfrac1n\sum_i r_i(\pi)$ and responsibilities
       $r_i(\pi)=\pi f_0(x_i)/p(x_i\mid\pi)$.
    1. Near a fixed point $\hat\pi=M(\hat\pi)$, each iteration multiplies the
       error by approximately $M'(\hat\pi)$. Define the observed information
       $I_{\textrm{obs}}=-\frac{d^2}{d\pi^2}\sum_i\log p(x_i\mid\pi)$ at
       $\hat\pi$, and the complete-data information
       $I_{\textrm{com}}=n/(\hat\pi(1-\hat\pi))$ that the labels would carry
       if they were observed (compare :eqref:`eq_mdl-coin-fisher`). Show
       that

       $$M'(\hat\pi)=1-\frac{I_{\textrm{obs}}}{I_{\textrm{com}}}.$$

       Express both sides in terms of the responsibilities and use
       $\sum_i r_i(\hat\pi)=n\hat\pi$. :citet:`Dempster.Laird.Rubin.1977`
       prove the general form of this result: the convergence rate of EM is
       governed by the fraction of information hidden by the latent variables.
    1. Draw $n=10{,}000$ points with $\pi=0.3$ for each
       $\Delta\in\{0.5,1,2,4\}$, run EM from $\pi^{(0)}=0.5$, and compare the
       measured ratio of successive errors against the prediction from (b).
       Taking $\hat\pi$ from a long run, how many iterations does each
       $\Delta$ need to bring the error below $10^{-6}$?
    1. What does the ratio $I_{\textrm{obs}}/I_{\textrm{com}}$ imply about the
       standard error of $\hat\pi$, compared with a dataset of the same size
       whose labels were observed?
   :exlabel:`ex_mdl-maximum-likelihood-em-missing-information`

1. **The bound by hand.** Take the single observation $x=0.5$ and the two
   equally weighted components $\mathcal N(-1,1)$ and $\mathcal N(1,1)$ from
   the text.
    1. Compute $\log p(x)$, along with the ELBO :eqref:`eq_mdl-elbo` for the
       uniform guess $q=(\tfrac12,\tfrac12)$ and for the posterior
       $q=(0.27,0.73)$, verifying the values quoted in the text.
    1. Verify :eqref:`eq_mdl-elbo-gap` for the uniform guess by computing the
       KL divergence directly.
    1. A hard guess puts all of $q$ on the right component. Compute its ELBO
       and its gap. An algorithm that alternates hard guesses with M-steps
       still never decreases the ELBO; why is its log-likelihood no longer
       guaranteed to increase?
   :exlabel:`ex_mdl-latent-elbo-by-hand`

1. **K-means as a limit.** Let all $K$ components of a Gaussian mixture share
   equal weights and a single fixed variance $\sigma^2$, so that only the means
   are fitted.
    1. Write the responsibilities $r_{ik}$ as a softmax of
       $-(x_i-\mu_k)^2/(2\sigma^2)$ over $k$. What quantity plays the role of
       the temperature? Show that as $\sigma^2\to0$, the responsibility of the
       nearest mean tends to $1$ whenever that nearest mean is unique.
    1. Show that the M-step :eqref:`eq_mdl-gmm-mstep` then sets each mean to
       the average of the points assigned to it, and that the two steps
       together never increase $\sum_i (x_i-\mu_{z_i})^2$, where $z_i$ is the
       component assigned to point $i$.
    1. In the symmetric start from the text, both means begin at the same
       point. What does K-means do from such a start, and what does this imply
       about how an implementation should break ties?
   :exlabel:`ex_mdl-latent-kmeans-limit`

[Discussions](https://d2l.discourse.group/t/36881)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §36.7]{.kicker}

A sum inside the logarithm, and the bound that removes it<br>**latent variables, EM, and the ELBO**
:::
:::

::: {.slide title="500 Points From Two Clusters, Labels Lost: Where Are the Means?"}
[The problem]{.kicker}

- $300$ points from $\mathcal N(-2, 0.8^2)$ and $200$ from
  $\mathcal N(1.5, 0.6^2)$. We see only the values.
- With the labels, the fit is three one-line estimators per cluster.
- Without them, each point's probability is a sum over the two clusters, and
  the log of a sum does not split.

::: {.d2l-note .warn}
How do we maximize a likelihood whose every term is a sum over hidden causes?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write the likelihood of a model with a hidden variable as a marginal;
- bound the log-likelihood from below by the ELBO, and name its gap;
- fit a mixture with EM, and prove that the likelihood never decreases;
- say where EM gets stuck and where the likelihood has no maximum.

**Running example:** two Gaussians on a line; one point at $x=0.5$.
**Idea:** guess the hidden variable in probabilities, then fit as if the
guess were data.
:::

::: {.slide title="After the Fits, Before the Posteriors"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Random Variables | densities, moments, tail bounds |
| Joint Distributions | conditioning, covariance, change of variables |
| Distributions | the named distributions and their losses |
| The Exponential Family | one form, convex fits, conjugacy |
| Statistics | error bars, tests, intervals |
| Maximum Likelihood | losses, Fisher information, MAP |
| **Latent Variables and EM** | **the ELBO, EM** |
| Bayesian Computation | averaging over a posterior |
| Concentration Inequalities | exponential tails of averages |
| Uniform Convergence | guarantees after selection |
| Naive Bayes | one classifier, end to end |
:::

::: {.col .narrow}
- **Uses:** maximum likelihood, Bayes' rule, the KL divergence.
- **Skips:** continuous latent variables, variational autoencoders.
:::
:::
:::

::: {.slide title="With the Labels, the Fit Is Counting and Averaging"}
[Mixtures]{.kicker}

Mixture: a hidden label $z\in\{1,\dots,K\}$ with $P(z=k)=\pi_k$, then
$x\sim\mathcal N(\mu_k,\sigma_k^2)$.

If an oracle revealed every $z_i$:

| parameter | estimator |
|:--|:--|
| weight $\pi_k$ | fraction of points with label $k$ |
| mean $\mu_k$ | average of those points |
| variance $\sigma_k^2$ | their sample variance |

::: {.d2l-note}
The problem is hard only because the labels are missing.
:::
:::

::: {.slide title="Without the Labels, a Sum Sits Inside the Logarithm"}
[Mixtures]{.kicker}

We observe $x$ alone, so its probability sums over the hidden label:

$$\ell(\boldsymbol\theta)=-\sum_{i=1}^n\log\Bigl(\sum_{k=1}^K\pi_k\,\mathcal N(x_i;\mu_k,\sigma_k^2)\Bigr).$$

- The log stops at the sum and never reaches a single Gaussian.
- Every parameter appears in every term.
- Setting the gradient to zero has no closed-form solution.
:::

::: {.slide title="Guess the Label in Probabilities: x = 0.5 Is 27% Left, 73% Right"}
[Mixtures]{.kicker}

One point $x=0.5$; components $\mathcal N(-1,1)$ and $\mathcal N(1,1)$,
equal weights. Bayes' rule:

$$P(\textrm{left}\mid x)=\frac{e^{-1.5^2/2}}{e^{-1.5^2/2}+e^{-0.5^2/2}}=\frac{1}{1+e}=0.27 .$$

- The point is split, not assigned.
- At $x=0$ the split is even; at $x=3$ it is $99.8\%$ right.

::: {.d2l-note}
Plan: guess the labels this way, then solve the easy problem with the guess.
:::
:::

::: {.slide title="I. Any Guess q Gives a Lower Bound on the Log-Likelihood"}
[The ELBO]{.kicker}

$q(z)$: any distribution over the hidden variable. The logarithm is concave,
so $\log\mathbb{E}[\cdot]\ge\mathbb{E}[\log(\cdot)]$ (Jensen):

$$\log p(x\mid\boldsymbol\theta)=\log\sum_z q(z)\frac{p(x,z\mid\boldsymbol\theta)}{q(z)}\;\ge\;\underbrace{\mathbb{E}_q\bigl[\log p(x,z\mid\boldsymbol\theta)\bigr]}_{\textrm{easy objective, averaged}}+\underbrace{H(q)}_{\textrm{entropy of the guess}}.$$

The right side is the evidence lower bound, $\mathcal L(q,\boldsymbol\theta)$.

::: {.d2l-note}
The logarithm reaches the joint density again: the easy problem is back.
:::
:::

::: {.slide title="II. The Gap Is the KL Divergence From the Guess to the Posterior"}
[The ELBO]{.kicker}

::: {.cols .vc}
::: {.col}
$$\log p(x\mid\boldsymbol\theta)=\mathcal L(q,\boldsymbol\theta)+D_{\textrm{KL}}\bigl(q\,\|\,p(z\mid x,\boldsymbol\theta)\bigr)$$

*Proof.* Put $p(z\mid x)=p(x,z)/p(x)$ into the KL divergence; $\log p(x)$
does not depend on $z$ and leaves the average. $\blacksquare$

The bound touches exactly when $q$ is the posterior.
:::

::: {.col .fig}
@fig:mdl-prob-elbo
:::
:::
:::

::: {.slide title="Check: What Does the 50/50 Guess Lose at x = 0.5?"}
[Check]{.kicker}

For $x=0.5$ the log-evidence is $\log p(x)=-1.42$, and the posterior is
$(0.27, 0.73)$. How far below does the ELBO of the guess
$q=(\tfrac12,\tfrac12)$ lie?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** By the KL divergence
$\tfrac12\log\tfrac{0.5}{0.27}+\tfrac12\log\tfrac{0.5}{0.73}=0.31-0.19=0.12$
nats: the ELBO is $-1.54$. The posterior as guess loses nothing.
:::
:::
:::

::: {.slide title="III. EM Alternates Two Maximizations and Never Loses Likelihood"}
[EM]{.kicker}

- **E-step:** fix $\boldsymbol\theta^{(t)}$, set $q_i=p(z\mid x_i,\boldsymbol\theta^{(t)})$. The gap closes.
- **M-step:** fix the $q_i$, maximize $\sum_i\mathbb{E}_{q_i}[\log p(x_i,z\mid\boldsymbol\theta)]$.

$$\log P(X\mid\boldsymbol\theta^{(t+1)})\;\underset{\textrm{bound}}{\ge}\;\sum_i\mathcal L\bigl(q_i,\boldsymbol\theta^{(t+1)}\bigr)\;\underset{\textrm{M-step}}{\ge}\;\sum_i\mathcal L\bigl(q_i,\boldsymbol\theta^{(t)}\bigr)\;\underset{\textrm{E-step}}{=}\;\log P(X\mid\boldsymbol\theta^{(t)})$$

::: {.d2l-note .rule}
Each EM iteration leaves the likelihood at least as high as it was.
:::
:::

::: {.slide title="For a Mixture, Both Steps Are Soft Counting"}
[EM]{.kicker}

**E-step**, the responsibility of component $k$ for point $i$:

$$r_{ik}=\frac{\pi_k\,\mathcal N(x_i;\mu_k,\sigma_k^2)}{\sum_j\pi_j\,\mathcal N(x_i;\mu_j,\sigma_j^2)}$$

**M-step**, with effective counts $n_k=\sum_i r_{ik}$:

$$\pi_k=\frac{n_k}{n},\qquad\mu_k=\frac1{n_k}\sum_i r_{ik}x_i,\qquad\sigma_k^2=\frac1{n_k}\sum_i r_{ik}(x_i-\mu_k)^2$$

::: {.d2l-note}
The oracle's estimators, with each point spreading its one vote. Shared
$\sigma^2\to0$: hard votes, which is K-means.
:::
:::

::: {.slide title="From a Poor Start, EM Finds Both Clusters in About Six Steps"}
[EM]{.kicker}

Start: weights $(0.5, 0.5)$, means $\pm0.5$, scales $1$. Log-likelihood at
steps 0 to 5 and at step 29, then the fit:

@!mdl-maximum-likelihood-expectation-maximization-1

- The trace never goes down, and most of the climb is the first step.
- Truth: weights $(0.6, 0.4)$, means $(-2, 1.5)$, scales $(0.8, 0.6)$.
:::

::: {.slide title="A Symmetric Start Never Separates"}
[Limits]{.kicker}

Both components start at $\mu=0$, $\sigma=1$:

@!mdl-maximum-likelihood-expectation-maximization-2

- Equal responsibilities for every point, so the M-step keeps the components
  equal: EM fits one Gaussian, the mean and standard deviation of all data.
- A stationary point, not a maximum, more than $100$ nats below.

::: {.d2l-note}
Restart from several random initializations.
:::
:::

::: {.slide title="The Mixture Likelihood Has No Maximum Without a Variance Floor"}
[Limits]{.kicker}

Put component 2 on one data point: $\mu_2=x_1$, weight $1/n$, scale
$\epsilon$.

- Its density at $x_1$ is $\dfrac{1}{n\sqrt{2\pi}\,\epsilon}\to\infty$ as $\epsilon\to0$.
- Component 1 still explains the other $n-1$ points.
- So the likelihood is unbounded: the "best" fit is a spike on one point.

::: {.d2l-note}
A floor on the variances, or a prior on them, removes the spike.
:::
:::

::: {.slide title="When the Posterior Is Out of Reach, the Bound Is Climbed Approximately"}
[Beyond EM]{.kicker}

The E-step needs the exact posterior $p(z\mid x,\boldsymbol\theta)$. Without
it the bound still holds, for any $q$:

| method | hidden variable | guess $q$ |
|:--|:--|:--|
| EM | cluster label | exact posterior |
| variational autoencoder | code of an image | a network $q_{\boldsymbol\phi}(z\mid x)$ |
| variational inference | the parameters | a Gaussian, fitted |

::: {.d2l-note}
All three maximize the same ELBO. Only EM closes the gap, so only EM
inherits the monotone likelihood.
:::
:::

::: {.slide title="Check: Does a Hard Guess Lose More Than a 50/50 Guess?"}
[Check]{.kicker}

At $x=0.5$ the posterior is $(0.27, 0.73)$, and the 50/50 guess lost $0.12$
nats. A hard guess puts all of $q$ on the right component. What is its gap?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $D_{\textrm{KL}}=1\cdot\log\tfrac{1}{0.73}=0.31$ nats, more than
the undecided guess loses. Hard assignments still give a valid bound, but it
does not touch the likelihood, so the monotonicity proof is lost.
:::
:::
:::

::: {.slide title="One Bound Turned a Hard Fit Into Repeated Easy Ones"}
[Wrap-up]{.kicker}

- A hidden variable puts a sum inside the logarithm of the likelihood.
- ELBO $=$ easy objective averaged over a guess $q$, plus the entropy of $q$.
- Gap $=$ KL from $q$ to the posterior: $0.12$ nats for the 50/50 guess at $x=0.5$.
- EM: the E-step closes the gap, the M-step climbs; the likelihood never decreases.
- EM stops at stationary points (the symmetric start); an unfloored variance is a spike.

::: {.d2l-note}
Next: the same bound, with the parameters as the hidden variable.
:::
:::
