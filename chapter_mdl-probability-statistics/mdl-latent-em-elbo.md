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
appropriately undecided where the components overlap
(:numref:`fig_mdl-prob-soft-resp`).

![Soft assignment as a ratio of density heights, for two equally weighted components $\mathcal{N}(-1,1)$ and $\mathcal{N}(1,1)$. (a) At $x=0.5$ the densities have heights $h_1$ and $h_2$, and the left component receives the share $r_1=h_1/(h_1+h_2)=0.27$. (b) The two posterior probabilities $r_1(x)$ and $r_2(x)$: an even split at $x=0$, $0.27$ to $0.73$ at $x=0.5$, and $0.998$ for the right component at $x=3$.](../img/mdl-prob-soft-resp.svg)
:label:`fig_mdl-prob-soft-resp`

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

The log-likelihood climbs by roughly $400$ nats, rising from around
$-1{,}280$ to around $-860$ (both figures vary by a few tens of nats with the
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
single data point (:numref:`fig_mdl-prob-gmm-spikes`). Imposing a lower floor on the variances, or placing a prior
over them, eliminates these degenerate spikes. Once the set of parameters at
least as good as the starting point is compact, every limit point of EM is
guaranteed to be a stationary point of the regularized problem
:cite:`Wu.1983`.

![The degenerate maximum of a Gaussian mixture likelihood. The first component (blue) covers $n-1$ of the data points (dots). The second (orange) sits on the remaining point, $\mu_2=x_1$, with standard deviation $\sigma_2=\epsilon$. As $\epsilon$ shrinks from $0.3$ to $0.1$ to $0.03$, its density at $x_1$ grows like $1/\epsilon$, and the likelihood grows with it without bound.](../img/mdl-prob-gmm-spikes.svg)
:label:`fig_mdl-prob-gmm-spikes`

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

Latent Variable Models, EM, and the ELBO<br>**mixture models · marginal log-likelihood · soft responsibilities · evidence lower bound · coordinate ascent · pitfalls**
:::
:::

::: {.slide title="500 Points from Two Populations, Labels Lost"}
[Motivation & Core Problem]{.kicker}

::: {.cols .vc}
::: {.col}
- **With observed labels (complete data):** Fitting $K$ Gaussian populations takes closed-form per-cluster sample fractions, means, and variances.
- **Without labels (latent variables):** Each observation's probability sums over hidden populations, trapping a sum inside the logarithm of the likelihood.
- **Unifying strategy:** Replace missing labels with soft posterior probabilities $q(z)$ and maximize the **Evidence Lower Bound (ELBO)**.
:::

::: {.col .fig .big}
@fig:ch36-prob-lost-labels
:::
:::
:::

::: {.slide title="Joint and Marginal Likelihoods with Latent Variables"}
[Generative Structure]{.kicker}

::: {.cols .vc}
::: {.col}
$$p(x, z \mid \boldsymbol{\theta}) = p(z \mid \boldsymbol{\theta})\,p(x \mid z, \boldsymbol{\theta}), \qquad p(x \mid \boldsymbol{\theta}) = \sum_{z} p(x, z \mid \boldsymbol{\theta})$$

- **Generative story ($z \to x$):** First sample the hidden cause $z \sim p(z \mid \boldsymbol{\theta})$, then generate the observation $x \sim p(x \mid z, \boldsymbol{\theta})$.
- **Marginal likelihood (evidence):** Because only $x$ is observed, $p(x \mid \boldsymbol{\theta})$ sums or integrates over every possible latent state $z$.
:::

::: {.col .fig}
@fig:ch36-prob-latent-dag
:::
:::
:::

::: {.slide title="Complete-Data Oracle vs. Observed-Data Mixture NLL"}
[Gaussian Mixture Models]{.kicker}

::: {.cols .vc}
::: {.col}
$$\ell(\boldsymbol{\theta}) = -\sum_{i=1}^n \log \Biggl(\sum_{k=1}^K \pi_k\,\mathcal{N}(x_i;\, \mu_k, \sigma_k^2)\Biggr)$$

- **Complete-data oracle ($z_i$ known):** The logarithm hits $\mathcal{N}(x_i; \mu_{z_i}, \sigma_{z_i}^2)$ directly, decoupling into $K$ closed-form quadratics.
- **Observed mixture ($z_i$ hidden):** The sum $\sum_{k=1}^K$ sits *inside* $\log(\cdot)$, coupling all $3K$ parameters across every data point.
:::

::: {.col .fig}
@fig:ch36-prob-oracle-vs-mixture
:::
:::
:::

::: {.slide title="Soft Probabilistic Guesses via Bayes' Rule"}
[Posterior Responsibilities]{.kicker}

::: {.cols .vc}
::: {.col}
$$P(z = 1 \mid x = 0.5) = \frac{0.5\,e^{-(0.5 + 1)^2 / 2}}{0.5\,e^{-(0.5 + 1)^2 / 2} + 0.5\,e^{-(0.5 - 1)^2 / 2}} = \frac{1}{1 + e} \approx 0.27$$

- **Fractional responsibility:** Under $0.5\,\mathcal{N}(-1, 1) + 0.5\,\mathcal{N}(1, 1)$, Bayes' rule assigns $x = 0.5$ softly ($27\%$ to component $1$, $73\%$ to component $2$).
- **Smooth uncertainty:** Responsibilities split $50/50$ at $x = 0$ and sharpen to $0.2\% / 99.8\%$ out at $x = 3$.
:::

::: {.col .fig}
@fig:ch36-prob-soft-resp
:::
:::
:::

::: {.slide title="Deriving the Evidence Lower Bound (ELBO)"}
[Jensen's Inequality]{.kicker}

::: {.cols .vc}
::: {.col}
$$\log p(x \mid \boldsymbol{\theta}) = \log \sum_{z} q(z)\,\frac{p(x, z \mid \boldsymbol{\theta})}{q(z)} \ge \sum_{z} q(z) \log \frac{p(x, z \mid \boldsymbol{\theta})}{q(z)} =: \mathcal{L}(q, \boldsymbol{\theta})$$

- **Pushing $\log$ inside the sum:** For *any* distribution $q(z)$ over latent states, concavity of $\log$ ($\log \mathbb{E}_q[Y] \ge \mathbb{E}_q[\log Y]$) yields a global lower bound.
- **Tractable joint log-density:** Inside $\mathcal{L}(q, \boldsymbol{\theta})$, the logarithm acts directly on the complete-data product $p(x, z \mid \boldsymbol{\theta})$.
:::

::: {.col .fig}
@fig:ch36-prob-jensen-elbo
:::
:::
:::

::: {.slide title="Anatomy of the ELBO: Energy Plus Entropy"}
[Decomposing the Lower Bound]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathcal{L}(q, \boldsymbol{\theta}) = \mathbb{E}_{q(z)}\bigl[\log p(x, z \mid \boldsymbol{\theta})\bigr] + H(q)$$

- **Expected complete-data log-likelihood ($\mathbb{E}_q[\log p(x, z \mid \boldsymbol{\theta})]$):** The tractable oracle objective weighted by soft probabilities $q(z)$.
- **Shannon entropy bonus ($H(q) = -\sum_z q(z)\log q(z)$):** Rewards spread-out guesses $q(z)$ and drops out when optimizing over $\boldsymbol{\theta}$.
:::

::: {.col .fig}
@fig:ch36-prob-elbo-anatomy
:::
:::
:::

::: {.slide title="The Fundamental ELBO–KL Decomposition"}
[Exact Gap Identity]{.kicker}

::: {.cols .vc}
::: {.col}
- **Exact identity for the gap:** Substituting $p(x, z \mid \boldsymbol{\theta}) = p(z \mid x, \boldsymbol{\theta})\,p(x \mid \boldsymbol{\theta})$ into $\mathcal{L}(q, \boldsymbol{\theta})$ proves:
  $$\log p(x \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + D_{\mathrm{KL}}\bigl(q(z) \,\|\, p(z \mid x, \boldsymbol{\theta})\bigr)$$
- **Non-negativity and tightness:** Because $D_{\mathrm{KL}} \ge 0$, $\mathcal{L}(q, \boldsymbol{\theta}) \le \log p(x \mid \boldsymbol{\theta})$ everywhere, with equality ($\mathcal{L} = \log p$) if and only if $q(z) = p(z \mid x, \boldsymbol{\theta})$.
- **Geometric tangency:** Setting $q$ to the posterior at $\boldsymbol{\theta}^{(t)}$ makes the lower bound curve touch the true log-likelihood curve tangentially at $\boldsymbol{\theta}^{(t)}$.
:::

::: {.col .fig}
@fig:mdl-prob-elbo
:::
:::
:::

::: {.slide title="Worked Example: Measuring the ELBO Gap by Hand"}
[Numerical Verification at x = 0.5]{.kicker}

::: {.cols .vc}
::: {.col}
- **Exact identity at $x = 0.5$ ($\log p(x) = -1.42$):** For every guess $q(z)$, $\mathcal{L}(q) + D_{\mathrm{KL}}(q \,\|\, p(z \mid x)) = -1.42$ nats.
- **Exact posterior $q = (0.27, 0.73)$:** Achieves $\mathcal{L} = -1.42$ with zero KL gap.
- **Why hard assignment $(0, 1)$ loses twice as much as uniform $(0.5, 0.5)$:** Setting $q = (0, 1)$ forfeits the entropy bonus ($H(q) = 0$), incurring a $0.31$-nat KL gap vs. $0.12$ nats for $(0.5, 0.5)$.
:::

::: {.col .fig .big}
@fig:ch36-prob-elbo-gap-bars
:::
:::
:::

::: {.slide title="Expectation–Maximization as Coordinate Ascent"}
[The EM Algorithm]{.kicker}

::: {.cols .vc}
::: {.col}
- **E-step (Expectation):** Fix $\boldsymbol{\theta}^{(t)}$ and set each $q_i^{(t)}(z) = p(z \mid x_i, \boldsymbol{\theta}^{(t)})$, closing the KL gap to zero so the ELBO touches $\log P(X \mid \boldsymbol{\theta}^{(t)})$.
- **M-step (Maximization):** Fix the responsibilities $q_i^{(t)}$ and maximize the expected complete-data log-likelihood over $\boldsymbol{\theta}$:
  $$\boldsymbol{\theta}^{(t+1)} = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}} \sum_{i=1}^n \mathbb{E}_{q_i^{(t)}(z)}\bigl[\log p(x_i, z \mid \boldsymbol{\theta})\bigr]$$
:::

::: {.col .fig}
@fig:ch36-prob-em-loop
:::
:::
:::

::: {.slide title="Theorem: EM Monotonically Improves Likelihood"}
[Convergence Guarantee]{.kicker}

Why can an EM step never decrease the true observed-data log-likelihood $\log P(X \mid \boldsymbol{\theta})$? Chaining three relations proves monotonicity in one line:

$$\log P(X \mid \boldsymbol{\theta}^{(t+1)}) \ge \sum_{i=1}^n \mathcal{L}\bigl(q_i^{(t)}, \boldsymbol{\theta}^{(t+1)}\bigr) \ge \sum_{i=1}^n \mathcal{L}\bigl(q_i^{(t)}, \boldsymbol{\theta}^{(t)}\bigr) = \log P(X \mid \boldsymbol{\theta}^{(t)})$$

1. **Global lower bound ($\ge$):** The ELBO is a valid lower bound on $\log P(X \mid \boldsymbol{\theta})$ at the new parameters $\boldsymbol{\theta}^{(t+1)}$ for any $q_i^{(t)}$.
2. **M-step optimization ($\ge$):** Choosing $\boldsymbol{\theta}^{(t+1)}$ to maximize the bound guarantees $\sum_i \mathcal{L}(q_i^{(t)}, \boldsymbol{\theta}^{(t+1)}) \ge \sum_i \mathcal{L}(q_i^{(t)}, \boldsymbol{\theta}^{(t)})$.
3. **E-step tightness ($=$):** Because $q_i^{(t)}$ was set to the exact posterior at $\boldsymbol{\theta}^{(t)}$, the KL gap at $\boldsymbol{\theta}^{(t)}$ is zero.

::: {.d2l-note .rule}
**Why exact posteriors matter:** If an algorithm replaces the E-step with hard $0/1$ assignments, it still monotonically climbs the ELBO, but because the third step is no longer an equality, the true log-likelihood $\log P(X \mid \boldsymbol{\theta})$ is no longer guaranteed to increase.
:::
:::

::: {.slide title="Closed-Form EM Updates for Gaussian Mixtures"}
[Soft Counting and Averaging]{.kicker}

Applying the E-step and M-step to a $K$-component Gaussian mixture model yields simple responsibility-weighted sample statistics:

- **E-step (posterior responsibilities $r_{ik} = q_i(z=k)$):**
  $$r_{ik} = \frac{\pi_k\,\mathcal{N}(x_i;\, \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j\,\mathcal{N}(x_i;\, \mu_j, \sigma_j^2)}$$
- **M-step (weighted counts, means, and variances with $n_k = \sum_{i=1}^n r_{ik}$):**
  $$\pi_k \leftarrow \frac{n_k}{n}, \qquad \mu_k \leftarrow \frac{1}{n_k}\sum_{i=1}^n r_{ik}\,x_i, \qquad \sigma_k^2 \leftarrow \frac{1}{n_k}\sum_{i=1}^n r_{ik}(x_i - \mu_k)^2$$
- **K-means as the zero-variance limit:** When $\pi_k = 1/K$ and all components share a fixed variance $\sigma^2 \to 0$, the softmax responsibilities $r_{ik} \propto \exp\bigl(-(x_i-\mu_k)^2/(2\sigma^2)\bigr)$ harden into $0/1$ indicators for the nearest centroid.
:::

::: {.slide title="Implementing 1D Gaussian Mixture EM in NumPy"}
[Code Implementation]{.kicker}

In vectorized NumPy, the entire EM loop for $n = 500$ observations from $0.6\,\mathcal{N}(-2.0, 0.8^2) + 0.4\,\mathcal{N}(1.5, 0.6^2)$ takes a dozen lines:

@maximum-likelihood-em-gmm
:::

::: {.slide title="Implementing 1D Gaussian Mixture EM in NumPy (cont.)"}
[Code Implementation]{.kicker}

::: {.d2l-note}
Broadcasting `x[:, None]` of shape `(n, 1)` against parameter arrays of shape `(K,)` evaluates the `(n, K)` joint density matrix `joint` in one step; normalizing across rows gives `r` (E-step), and summing down columns gives `n_k` (M-step).
:::
:::

::: {.slide title="Rapid Convergence from an Asymmetric Initialization"}
[Empirical Trajectory]{.kicker}

Starting from two heavily overlapping unit-variance components centered at $\mu = (-0.5, 0.5)$ with equal weights $\pi = (0.5, 0.5)$, we run $30$ EM iterations:

@!mdl-maximum-likelihood-expectation-maximization-1

- **Monotonic climb:** The log-likelihood climbs by roughly $400$ nats (from $\approx -1{,}280$ to $\approx -860$), with nearly the entire gain achieved in the first $6$ steps.
- **Missing information governs speed:** Near a fixed point, the linear convergence rate of EM equals $1 - I_{\mathrm{obs}}/I_{\mathrm{com}}$—the fraction of Fisher information hidden by the missing labels. Well-separated clusters ($I_{\mathrm{obs}} \approx I_{\mathrm{com}}$) converge in a few steps.
:::

::: {.slide title="Pitfall 1: Label Switching and Symmetry Traps"}
[Non-Identifiability & Saddle Points]{.kicker}

Permuting the $K$ component indices produces $K!$ identical global maxima (*label switching*). Far more dangerous is initializing two components identically at $\mu = (0, 0)$ and $\sigma = (1, 1)$:

@!mdl-maximum-likelihood-expectation-maximization-2

- **Locked responsibilities:** Because both components start identical, they receive identical responsibilities $r_{i1} = r_{i2} = 0.5$ on every observation $x_i$, so the M-step updates them to the exact same overall mean and standard deviation forever.
- **Stationary point, not a maximum:** EM is trapped at a single-Gaussian saddle point more than $100$ nats below the two-cluster fit—the exact same symmetry trap that occurs if hidden units in a neural network layer are initialized with identical weights.
:::

::: {.slide title="Pitfall 2: Unbounded Likelihood and Variance Spikes"}
[Degenerate Global Maxima]{.kicker}

::: {.cols .vc}
::: {.col}
$$\ell(\boldsymbol{\theta}) \le \log\bigl(n\sqrt{2\pi}\bigr) + \log\epsilon + \mathrm{const} \xrightarrow{\epsilon \to 0} -\infty$$

- **Mechanism of a variance spike:** Centering component $2$ on a single point ($\mu_2 = x_1$) and shrinking $\sigma_2 = \epsilon \to 0$ drives its density $\frac{1}{\sqrt{2\pi}\epsilon} \to \infty$ (just like heteroscedastic regression NLL when $\hat{y}_i = y_i$ and $\hat{\sigma}_i \to 0$).
- **Practical fix:** Enforce a variance floor $\sigma_k^2 \ge \sigma_{\min}^2 > 0$ or place an inverse-gamma prior on $\sigma_k^2$.
:::

::: {.col .fig}
@fig:ch36-prob-gmm-spikes
:::
:::
:::

::: {.slide title="From Exact EM to Variational Autoencoders (VAEs)"}
[Amortized Variational Inference]{.kicker}

::: {.cols .vc}
::: {.col}
- **Intractable deep posterior:** When continuous latent codes $z \in \mathbb{R}^d$ pass through a deep nonlinear decoder $p_{\boldsymbol{\theta}}(x \mid z)$, the exact posterior $p(z \mid x, \boldsymbol{\theta})$ has no closed form.
- **Amortized encoder $q_{\boldsymbol{\phi}}(z \mid x)$:** A neural network predicts $(\boldsymbol{\mu}_{\boldsymbol{\phi}}(x), \log\boldsymbol{\sigma}_{\boldsymbol{\phi}}(x))$ directly from $x$.
- **Joint SGD on the ELBO:** Reparameterizing $z = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ trains $(\boldsymbol{\theta}, \boldsymbol{\phi})$ jointly by backpropagating through $\sum_i \mathcal{L}(q_{\boldsymbol{\phi}}(\cdot \mid x_i), \boldsymbol{\theta})$.
:::

::: {.col .fig .big}
@fig:ch36-prob-vae-amortized
:::
:::
:::

::: {.slide title="Unifying EM, VAEs, and Bayesian Variational Inference"}
[Three Instances of One Bound]{.kicker}

The decomposition $\log \text{Evidence} = \mathrm{ELBO}(q) + D_{\mathrm{KL}}(q \,\|\, \text{Posterior})$ unifies three major algorithms across statistical learning and deep generative modeling:

| Method | Unobserved Variable | Variational Distribution $q$ | How $\mathrm{ELBO}$ Is Maximized | Does KL Gap Close to $0$? |
| :--- | :--- | :--- | :--- | :--- |
| **EM Algorithm** | Per-example discrete cluster $z_i \in \{1,\dots,K\}$ | Exact tabular distribution $q_i(z_i)$ per point | Coordinate ascent (closed-form E & M steps) | **Yes** ($q_i = p(z_i \mid x_i, \boldsymbol{\theta}^{(t)})$ after every E-step) |
| **Variational Autoencoder** | Per-example continuous code $\mathbf{z}_i \in \mathbb{R}^d$ | Amortized encoder network $q_{\boldsymbol{\phi}}(\mathbf{z}_i \mid \mathbf{x}_i)$ | Joint SGD over $(\boldsymbol{\theta}, \boldsymbol{\phi})$ via reparameterization | **No** (bounded by encoder family capacity) |
| **Bayesian VI (§{sec_mdl-bayesian-computation})** | Global model parameters $\boldsymbol{\theta} \in \mathbb{R}^p$ | Parametric family $q_{\boldsymbol{\psi}}(\boldsymbol{\theta})$ (e.g., Gaussian) | Gradient ascent on $\mathcal{L}(\boldsymbol{\psi})$ over variational $\boldsymbol{\psi}$ | **No** (minimizes reverse KL $D_{\mathrm{KL}}(q_{\boldsymbol{\psi}} \,\|\, p(\boldsymbol{\theta}\mid\mathcal{D}))$) |
:::

::: {.slide title="Concept Check: Latent Variables, EM, and the ELBO"}
[Check Your Understanding]{.kicker}

1. **Entropy bonus in the ELBO:** At $x = 0.5$ where the true posterior is $(0.27, 0.73)$, why does a hard guess $q = (0, 1)$ have a larger ELBO gap ($0.31$ nats) than an undecided $50/50$ guess ($0.12$ nats)?
2. **Temperature limit of GMM EM:** If all $K$ Gaussian components have equal mixing weights $\pi_k = 1/K$ and a shared fixed variance $\sigma^2$, what happens to the E-step responsibilities $r_{ik}$ and M-step means $\mu_k$ as $\sigma^2 \to 0$?
3. **Why hard EM can decrease likelihood:** If an algorithm replaces the E-step with hard assignments $q_i(z) = \mathbf{1}\{z = k^*(x_i)\}$, it still monotonically increases the ELBO. Why can the true log-likelihood $\log P(X \mid \boldsymbol{\theta})$ decrease?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. The hard guess puts zero mass on component $1$ ($H(q) = 0$) and incurs $D_{\mathrm{KL}} = 1 \cdot \log\frac{1}{0.73} = 0.31$ nats, whereas the $50/50$ guess retains $H(q) = \log 2 = 0.69$ nats of entropy and sits closer in KL divergence to $(0.27, 0.73)$.
2. The responsibilities $r_{ik} = \frac{\exp(-(x_i-\mu_k)^2/(2\sigma^2))}{\sum_j \exp(-(x_i-\mu_j)^2/(2\sigma^2))}$ form a softmax with temperature $2\sigma^2$; as $\sigma^2 \to 0$, $r_{ik} \to 1$ for the nearest centroid $\mu_k$, reducing EM directly to **K-means**.
3. Because a hard distribution $q_i$ never equals the soft posterior $p(z \mid x_i, \boldsymbol{\theta}^{(t)})$, the KL gap $D_{\mathrm{KL}}(q_i \,\|\, p) > 0$ does not vanish at $\boldsymbol{\theta}^{(t)}$, breaking the tangency equality $\mathcal{L}(q^{(t)}, \boldsymbol{\theta}^{(t)}) = \log P(X \mid \boldsymbol{\theta}^{(t)})$.
:::
:::

::: {.slide title="Summary: Latent Variables, EM, and the ELBO"}
[Key Takeaways]{.kicker}

| Concept | Mathematical Form | Statistical & Deep Learning Role |
| :--- | :--- | :--- |
| **Marginal Log-Likelihood** | $\log p(x \mid \boldsymbol{\theta}) = \log \sum_z p(z \mid \boldsymbol{\theta})\,p(x \mid z, \boldsymbol{\theta})$ | Sum inside $\log$ couples all parameters and blocks closed-form MLE |
| **Evidence Lower Bound** | $\mathcal{L}(q, \boldsymbol{\theta}) = \mathbb{E}_q[\log p(x, z \mid \boldsymbol{\theta})] + H(q)$ | Moves $\log$ back onto the joint density via Jensen's inequality |
| **Exact KL Gap Identity** | $\log p(x \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + D_{\mathrm{KL}}\bigl(q(z) \,\|\, p(z \mid x, \boldsymbol{\theta})\bigr)$ | Shows $\mathcal{L} \le \log p$ everywhere, with equality iff $q$ is the true posterior |
| **EM Coordinate Ascent** | E-step: $q_i^{(t)} = p(z \mid x_i, \boldsymbol{\theta}^{(t)})$; M-step: $\operatorname{argmax}_{\boldsymbol{\theta}} \sum_i \mathcal{L}$ | Monotonically increases $\log P(X \mid \boldsymbol{\theta})$ at every iteration |
| **GMM Soft Counting** | $r_{ik} \propto \pi_k \mathcal{N}(x_i; \mu_k, \sigma_k^2)$; $\mu_k = \frac{1}{n_k}\sum_i r_{ik}x_i$ | Softens oracle estimators; reduces to K-means as shared $\sigma^2 \to 0$ |
| **EM Pitfalls** | Symmetry trap ($\mu_1 = \mu_2$) & variance spikes ($\sigma_k \to 0$) | Requires random restarts and variance floors or priors on $\sigma_k^2$ |
:::

