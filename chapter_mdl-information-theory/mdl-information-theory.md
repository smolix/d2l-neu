# Entropy, Cross-Entropy, and KL Divergence
:label:`sec_mdl-information_theory`

Cross-entropy is one of the most common training losses in this book. It is
also a basic quantity in information theory, introduced by Shannon in 1948
:cite:`Shannon.1948`. Its coding interpretation relates three quantities:
entropy is the expected code length under the data distribution,
cross-entropy is the expected length obtained with a model distribution, and
their difference is the Kullback--Leibler divergence. This section develops
these quantities and Gibbs' inequality, derives their coding interpretation
from the Kraft inequality and Shannon codes, and explains perplexity, label
smoothing, and knowledge distillation in the same framework.

The logarithm base is a pure choice of unit: base $2$ gives *bits*, base $e$
gives *nats*. **We work in nats throughout**, matching deep-learning practice:
the library `log` is natural, so reported training losses are already
in nats. The one place where base $2$ is genuinely natural is coding with
binary symbols, and we flag it explicitly when we get there. Since
$\log_2 x = \ln x / \ln 2$, converting is a fixed rescaling:
$1\textrm{ bit} = \ln 2 \approx 0.693$ nats and
$1\textrm{ nat} = 1/\ln 2 \approx 1.443$ bits. Changing the logarithm base
leaves every argmin and trained model unchanged.

We use the following imports throughout the section.

```{.python .input #information-theory-imports}
#@tab mxnet
import math
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input #information-theory-imports}
#@tab pytorch
import math
import torch
```

```{.python .input #information-theory-imports}
#@tab tensorflow
import math
import tensorflow as tf
```

```{.python .input #information-theory-imports}
#@tab jax
import math
import jax
from jax import numpy as jnp
import optax
```

## Information and Entropy

### Surprise and Self-Information

Consider reports about a shuffled deck of cards. "I see a card" conveys no
new information because the event is certain. "I see a heart" identifies one
of four equally likely suits, while "It is the three of spades" identifies one
of $52$ equally likely cards. The order of the entire deck identifies one of
$52!$ outcomes. Information therefore tracks *surprise*: the less probable an
event, the more is learned by observing it. Shannon formalized this idea. The
*self-information* of an event $x$ with
probability $p(x)$ is

$$
I(x) = -\log p(x),
$$

measured in nats for the natural logarithm. The four card reports carry
$-\ln 1 = 0$, $\ln 4 \approx 1.39$, $\ln 52 \approx 3.95$, and
$\ln 52! \approx 156.4$ nats respectively; in bits (base $2$, the unit of one
fair coin flip) they carry the classic numbers $0$, $2$,
$\log_2 52 \approx 5.70$, and $\log_2 52! \approx 225.6$. The function
$-\log p$ is the unique choice (up to the base, i.e., the unit) satisfying
the properties we implicitly demanded: certain events carry zero information,
rarer events carry more, and independent events add, so that $I$ of a joint
outcome with probability $p_1 p_2$ is $I(p_1) + I(p_2)$, which forces a
logarithm (:citet:`Csiszar.2008` gives the formal axiomatics; Exercise :exref:`ex_mdl-information-theory-show-that-self-information`
walks through the argument).

This surprise is relative to the stated probabilistic model and its event
partition. A confidently incorrect model can assign large self-information to
an ordinary event; the number alone does not measure semantic importance or
anomaly.


![Self-information $I(x) = -\log p(x)$ in nats as a function of the probability $p$. Certain events ($p = 1$) carry zero information, the fair coin ($p = \tfrac{1}{2}$) carries $\ln 2 \approx 0.693$ nats, and the curve diverges as $p \to 0$: rare means surprising.](../img/mdl-it-self-info-curve.svg)
:label:`fig_mdl-self-info-curve`

:numref:`fig_mdl-self-info-curve` plots the curve: decreasing, convex, zero at
$p = 1$, infinite at $p = 0$. The computation is immediate.

```{.python .input #information-theory-self-information}
import numpy as onp
def self_information(p):
    """Self-information -log p, in nats."""
    return -onp.log(p)

self_information(1 / 64)
```

An event of probability $1/64$ carries $\ln 64 \approx 4.16$ nats (exactly
$6$ bits: it takes six fair-coin flips to have probability $1/64$).

### Shannon Entropy

Let $X$ be a discrete random variable with probability mass function $p$.
Its *Shannon entropy* is the expected self-information of its outcome:

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_mdl-ent_def`

Equivalently, $H(X)=-\sum_i p_i\log p_i$, where $p_i=P(X=x_i)$. A point mass has
entropy zero, and a distribution on a fixed finite support has maximal entropy
when it is uniform. These nonnegativity and coding interpretations apply to
probability masses. They must not be transferred to density values, which can
exceed one and change under a coordinate transformation.

For continuous $X$ with density $p$ the sum becomes an integral, and the
result is called the *differential entropy*,

$$
h(X) = -\int p(x) \log p(x) \,dx.
$$
:eqlabel:`eq_mdl-diff-ent-def`

We write lowercase $h$ for differential entropy throughout, to keep it
distinct from the discrete $H$; the two behave differently in ways that
matter. Differential entropy is *not* the limit of the discrete entropy, and
it is *not coordinate-invariant*: under an invertible change of variables
$y = g(x)$ it shifts by $E[\log|\det \partial g/\partial x|]$, so $h$ can
even be *negative* (e.g., a narrow Gaussian with $\sigma < 1/\sqrt{2\pi e}$).
Quantities built from the *ratio* of two densities are invariant under a change
of variables: both densities pick up the same Jacobian factor, which cancels in
$p(x)/q(x)$, so the KL divergence defined below is exactly invariant.
Cross-entropy sits in between. It contains a single lone logarithm,
$-E_P[\log q]$, so it inherits the same Jacobian shift as entropy, but that
shift depends only on the truth $P$ and the map $g$, never on the model $Q$,
so it moves *every* candidate model's loss by the same constant.
Reparameterizing the data changes all the cross-entropy values, but their
differences, their gradients in $Q$, and the argmin do not. That is the
precise sense in which objectives that *compare* distributions are safe under
reparameterization while raw differential entropy is not.

The convention $0\log0=0$ should be implemented by masking only zero probabilities.
A general `nansum` would also hide NaNs caused by negative, unnormalized, or
otherwise invalid input, so the helper validates the vector.

```{.python .input #information-theory-definition}
import numpy as onp
def entropy(p):
    """Entropy of a probability vector, in nats."""
    p = onp.asarray(p, dtype=float)
    if p.ndim != 1 or not onp.all(onp.isfinite(p)):
        raise ValueError("p must be a finite one-dimensional probability vector")
    if onp.any(p < 0):
        raise ValueError("probabilities must be nonnegative")
    if not onp.isclose(p.sum(), 1.0):
        raise ValueError("probabilities must sum to one")
    positive = p > 0
    return float(-onp.dot(p[positive], onp.log(p[positive])))

entropy(onp.array([0.1, 0.5, 0.1, 0.3]))
```

The distribution $(0.1, 0.5, 0.1, 0.3)$ has entropy $\approx 1.168$ nats:
less than the $\ln 4 \approx 1.386$ nats of a uniform distribution on four
outcomes, because the mass is unevenly spread.

### Entropy Is Maximized by the Uniform Distribution

Two basic properties characterize discrete entropy. First, for discrete $X$
every term $-p_i \log p_i$ is non-negative, so $H(X) \geq 0$, with equality
exactly for a point mass (a constant "random" variable). Second, entropy is
largest when the distribution hedges maximally:

**Proposition (uniform maximizes entropy).** *If $X$ takes at most $k$ values,
then*

$$
H(X) \leq \log k,
$$

*with equality if and only if $X$ is uniform on $k$ values.*

**Proof.** Write the entropy as an expectation of a concave function's
argument: $H(X) = E[\log(1/p(X))]$. Since $\log$ is concave, Jensen's
inequality (:numref:`subsec_mdl-jensen`) gives

$$
H(X) = E\!\left[\log \frac{1}{p(X)}\right]
\leq \log E\!\left[\frac{1}{p(X)}\right]
= \log \sum_{i\,:\,p_i > 0} p_i \cdot \frac{1}{p_i}
= \log m
\leq \log k,
$$

where $m \leq k$ is the number of outcomes with $p_i > 0$. Equality requires
both inequalities to be tight. Because $\log$ is *strictly* concave, the
first is an equality only when $1/p(X)$ is constant almost surely, i.e., $X$
is uniform on its $m$ support points; and $\log m = \log k$ forces $m = k$.
Together: $p_i = 1/k$ for all $i$. $\blacksquare$

For continuous $X$ the analogous statements need constraints to be true at all
(on a bounded interval the uniform density again maximizes differential
entropy; under a variance constraint the Gaussian does, as we prove in
:numref:`subsec_mdl-gaussian-max-entropy`).

Consider the coin. A Bernoulli variable with heads-probability $p$ has the
*binary entropy* $H(p) = -p \log p - (1-p)\log(1-p)$ (overloading $H$: the
argument here is a parameter, not a random variable), plotted in
:numref:`fig_mdl-bernoulli-entropy`: zero at the deterministic endpoints
$p \in \{0, 1\}$, concave in between, and maximal at the fair coin
$p = \tfrac{1}{2}$, where it equals $\ln 2 \approx 0.693$ nats ($= 1$ bit,
consistent with the $\log k$ bound for $k = 2$). A biased coin with $p = 0.1$
has entropy $H(0.1) \approx 0.325$ nats because its outcome is comparatively
predictable.

![The binary entropy $H(p) = -p \log p - (1-p) \log (1-p)$ in nats, concave and symmetric about $p = \tfrac{1}{2}$, where it peaks at $\ln 2 \approx 0.693$ nats. At the deterministic endpoints $p = 0$ and $p = 1$ the outcome carries no surprise and the entropy vanishes.](../img/mdl-it-bernoulli-entropy.svg)
:label:`fig_mdl-bernoulli-entropy`

Entropy also extends from one random variable to several: the joint entropy
$H(X, Y)$, the conditional entropy $H(Y \mid X)$, and the *mutual information*
$I(X; Y)$ that measures what $X$ and $Y$ share. Those quantities power
contrastive and self-supervised learning, and we develop them in their own
section, :numref:`sec_mdl-mutual-information`.

## Cross-Entropy and KL Divergence

### The Kullback--Leibler Divergence

In :numref:`sec_linear-algebra` we measured distances between points with
norms. We now want a notion of "distance" between *distributions*: how badly
does a model $Q$ misrepresent a truth $P$? Information theory's answer is the
*Kullback--Leibler (KL) divergence*, also called *relative entropy*
:cite:`Kullback.Leibler.1951`. Given a
random variable $X \sim P$ with p.d.f. or p.m.f. $p(x)$, and a second
distribution $Q$ with p.d.f. or p.m.f. $q(x)$ that we use to approximate $P$,

$$D_{\textrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_mdl-kl_def`

The term inside the expectation is $\log p(x) - \log q(x)$, the difference of
two surprises: how surprised $Q$ is by the outcome $x$, minus how surprised
$P$ is. It is positive where $Q$ underestimates ($q(x) < p(x)$: $Q$ is *more*
surprised than it should be) and negative where $Q$ overestimates. KL averages
this *relative surprise* over outcomes drawn from the truth $P$. Note the
asymmetry baked into the definition: $P$ supplies both the samples and the
numerator. In general $D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P)$, and
the gap can be large (we exhibit it numerically below); the consequences of
*which* direction is optimized are taken up in
:numref:`sec_mdl-divergences-distances`. One more edge case to keep in mind:
if some outcome has $p(x) > 0$ but $q(x) = 0$ (the model declares impossible
an event with positive probability), then $D_{\textrm{KL}}(P\|Q) = \infty$.

Here is the discrete case in code, for `p` and `q` given as probability
vectors over the same finite outcome set. KL divergence is non-negative on
its own (Gibbs' inequality, next), so we do **not** wrap the result in
`abs()`; doing so would teach the false idea that KL needs an absolute value
to stay non-negative. The `nansum` is deliberate: where $p(x) = 0$ the direct
floating-point expression for $0 \cdot \log(0/q)$ yields `nan`,
which `nansum` drops to encode $0 \log 0 = 0$; the *other* edge case,
$p(x) > 0$ with $q(x) = 0$, instead yields $+\infty$ (not `nan`), which
`nansum` keeps, so the code correctly returns $+\infty$ for that divergence.

```{.python .input #information-theory-definition-2}
import numpy as onp
def kl_divergence(p, q):
    """KL(P || Q) for two probability vectors, in nats."""
    p, q = onp.asarray(p, dtype=float), onp.asarray(q, dtype=float)
    kl = p * onp.log(p / q)
    return float(onp.nansum(kl))
```

### Gibbs' Inequality

The KL divergence cannot be negative, an observation going back to Gibbs
:cite:`Gibbs.1902`. Together with the cross-entropy decomposition below, this
result justifies cross-entropy as a loss and supports the later coding and
label-smoothing results.

**Proposition (Gibbs' inequality).** *For any distributions $P$ and $Q$ on
the same space,*

$$
D_{\textrm{KL}}(P\|Q) \geq 0,
$$

*with equality if and only if $P = Q$.*

**Proof.** One application of Jensen's inequality
(:numref:`subsec_mdl-jensen`) to the convex function $-\log$:

$$
\begin{aligned}
D_{\textrm{KL}}(P\|Q)
&= E_{x\sim P}\!\left[-\log\frac{q(x)}{p(x)}\right]
\geq -\log E_{x\sim P}\!\left[\frac{q(x)}{p(x)}\right] \\
&= -\log \!\!\sum_{x\,:\,p(x)>0}\!\! q(x)
= -\log Q(\textrm{supp}\,P)
\geq 0.
\end{aligned}
$$

The expectation runs only over the support of $P$, where the $p(x)$ factors
cancel, and what remains is the $Q$-mass of that support: a number
$Q(\textrm{supp}\,P) \leq 1$, whose negative log is therefore $\geq 0$ (the
same computation runs with integrals for densities, with the same convention
at $q = 0$). For equality
throughout, both inequalities must be tight: because $-\log$ is *strictly*
convex, the first is an equality only when the ratio $q(x)/p(x)$ is constant
$P$-almost surely, and the second requires $Q(\textrm{supp}\,P) = 1$, i.e.,
$Q$ puts no mass outside $P$'s support. Summing $q(x) = c\,p(x)$ over the
support then gives $c = 1$, i.e., $P = Q$.
$\blacksquare$

### Maximum Entropy and the Gaussian
:label:`subsec_mdl-gaussian-max-entropy`

Gibbs' inequality has an immediate consequence for continuous distributions:
among all densities with a given mean and variance, the Gaussian is the
hardest to predict. :numref:`sec_mdl-exponential-family` granted this fact; here
is the proof.

**Proposition (Gaussian maximum entropy).** *Among all densities $p$ on
$\mathbb{R}$ with mean $\mu$ and variance $\sigma^2$,*

$$
h(P) \leq \tfrac{1}{2}\log(2\pi e \sigma^2) = h(\mathcal{N}(\mu, \sigma^2)),
$$
:eqlabel:`eq_mdl-gaussian-max-entropy`

*with equality if and only if $P = \mathcal{N}(\mu, \sigma^2)$.*

**Proof.** Let $\varphi(x) = (2\pi\sigma^2)^{-1/2}
e^{-(x-\mu)^2/(2\sigma^2)}$ be the Gaussian density. The key observation is
that $-\log \varphi(x) = \tfrac{1}{2}\log(2\pi\sigma^2) +
\tfrac{(x-\mu)^2}{2\sigma^2}$ is a quadratic in $x$, so its expectation under
*any* density $p$ with mean $\mu$ and variance $\sigma^2$ depends on $p$ only
through those two moments:

$$
-E_P[\log \varphi(x)]
= \tfrac{1}{2}\log(2\pi\sigma^2) + \frac{E_P\big[(x-\mu)^2\big]}{2\sigma^2}
= \tfrac{1}{2}\log(2\pi\sigma^2) + \tfrac{1}{2}
= \tfrac{1}{2}\log(2\pi e \sigma^2).
$$

Taking $P = \mathcal{N}(\mu, \sigma^2)$ itself, this computes the Gaussian's
own differential entropy en route:
$h(\mathcal{N}(\mu, \sigma^2)) = \tfrac{1}{2}\log(2\pi e \sigma^2)$, the
closed form quoted whenever a Gaussian entropy is needed (for instance in
:numref:`sec_mdl-mutual-information`). Now apply Gibbs' inequality, which
holds for densities as noted above, with $Q = \mathcal{N}(\mu, \sigma^2)$.
Because $E_P[\log \varphi]$ is finite, the divergence splits:

$$
0 \leq D_{\textrm{KL}}\big(P \,\|\, \mathcal{N}(\mu, \sigma^2)\big)
= E_P[\log p(x)] - E_P[\log \varphi(x)]
= -h(P) + \tfrac{1}{2}\log(2\pi e \sigma^2).
$$

Rearranging gives :eqref:`eq_mdl-gaussian-max-entropy`. Equality holds
exactly when the divergence vanishes, which by Gibbs' equality case means
$P = \mathcal{N}(\mu, \sigma^2)$. $\blacksquare$

The bound works because the Gaussian's log-density is a polynomial in exactly
the quantities being constrained, the first two moments, so the cross term
$-E_P[\log \varphi]$ is the same for every competitor $P$.

### Gaussians, in Closed Form

The `kl_divergence` above works on *discrete* probability vectors. For
*continuous* distributions the standard worked example is the KL divergence
between two univariate Gaussians, which has a closed form. For
$P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$,

$$
D_{\textrm{KL}}\!\left(\mathcal{N}(\mu_1,\sigma_1^2) \,\big\|\, \mathcal{N}(\mu_2,\sigma_2^2)\right)
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}.
$$
:eqlabel:`eq_mdl-gaussian_kl`

These logs are natural logs, so the result is in nats. Read two things off
:eqref:`eq_mdl-gaussian_kl` directly. First, it is non-negative
and zero exactly when $\mu_1=\mu_2$ and $\sigma_1=\sigma_2$ (i.e., $P=Q$),
as Gibbs' inequality demands. Second, it is *not symmetric*: swapping the
roles of $P$ and $Q$ changes the value, because the variance ratio and the
mean gap enter asymmetrically.
(Exercise :exref:`ex_mdl-information-theory-gaussian-kl` derives
:eqref:`eq_mdl-gaussian_kl`.)

The following calculation compares the formula with a direct Monte Carlo estimate of
$D_{\textrm{KL}}(P\|Q) = E_{x\sim P}[\log p(x) - \log q(x)]$. The estimate
uses no `abs()`; the divergence is non-negative on its own.

```{.python .input #information-theory-example-1}
import numpy as onp
def gaussian_kl(mu1, sigma1, mu2, sigma2):
    """Closed-form KL(N(mu1, sigma1^2) || N(mu2, sigma2^2)), in nats."""
    return (math.log(sigma2 / sigma1)
            + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
            - 0.5)

def gaussian_log_pdf(x, mu, sigma):
    return (-0.5 * math.log(2 * math.pi * sigma ** 2)
            - (x - mu) ** 2 / (2 * sigma ** 2))

def mc_kl(mu1, sigma1, mu2, sigma2, n=200000):
    """Monte-Carlo estimate of the same KL by sampling x ~ P."""
    x = onp.random.normal(loc=mu1, scale=sigma1, size=(n, ))
    return (gaussian_log_pdf(x, mu1, sigma1)
            - gaussian_log_pdf(x, mu2, sigma2)).mean()
```

Take $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(1, 1)$ (a unit mean shift,
equal variances). The closed form should match the Monte-Carlo estimate to
within sampling noise, and both should equal
$\tfrac{(\mu_1-\mu_2)^2}{2\sigma^2} = \tfrac{1}{2}$ nats here.

```{.python .input #information-theory-example-2}
closed_form = gaussian_kl(0.0, 1.0, 1.0, 1.0)
monte_carlo = mc_kl(0.0, 1.0, 1.0, 1.0)

closed_form, monte_carlo
```

Now make the asymmetry explicit. Compare $D_{\textrm{KL}}(P\|Q)$ with
$D_{\textrm{KL}}(Q\|P)$ when the two Gaussians have *different variances*, say
$P=\mathcal{N}(0,1)$ and $Q=\mathcal{N}(0,4)$ (i.e., $\sigma_2=2$). The two
numbers differ ($\approx 0.318$ vs. $\approx 0.807$ nats): KL is a
divergence, not a distance.

```{.python .input #information-theory-example-3}
kl_pq = gaussian_kl(0.0, 1.0, 0.0, 2.0)
kl_qp = gaussian_kl(0.0, 2.0, 0.0, 1.0)

kl_pq, kl_qp
```

### Cross-Entropy

The loss commonly implemented in classifiers is not the KL divergence but a
close relative. The *cross-entropy* from $P$ to $Q$ scores outcomes drawn from
the truth $P$ by the model's surprise,

$$\textrm{CE}(P, Q) = - E_{x \sim P} [\log q(x)].$$
:eqlabel:`eq_mdl-ce_def`

Adding and subtracting $H(P)$ inside the expectation (legitimate whenever
$H(P)$ is finite) splits the model's surprise into the truth's own surprise
plus the relative surprise:

$$\textrm{CE} (P, Q) = H(P) + D_{\textrm{KL}}(P\|Q).$$
:eqlabel:`eq_mdl-ce_decomp`

Combined with Gibbs' inequality, this identity explains why minimizing
cross-entropy recovers the data distribution when the model family contains it:

$$
\underbrace{D_{\textrm{KL}}(P\|Q) \ge 0}_{\textrm{Gibbs}}
\;\Longrightarrow\;
\underbrace{\textrm{CE}(P,Q) \ge H(P)}_{\textrm{cross-entropy} \ \ge\ \textrm{entropy}},
\quad \textrm{equality iff } P = Q.
$$

Entropy $H(P)$ does not depend on the model $Q$. Consequently, minimizing
cross-entropy in $Q$ is equivalent to minimizing
$D_{\textrm{KL}}(P\|Q)$, with equality at $Q=P$. In the coding view below,
these terms become expected code lengths.
:numref:`fig_mdl-code-length-bars` draws the decomposition to scale
for a four-outcome distribution we will code by hand in that coding view.

![The decomposition :eqref:`eq_mdl-ce_decomp` as code lengths, for the data distribution $P = (\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \tfrac{1}{8})$ and a uniform model $Q$. A code matched to $P$ has expected length $H(P) = 1.75$ bits per symbol. A code based on $Q$ has expected length $\textrm{CE}(P, Q) = 2.0$ bits, and their difference is $D_{\textrm{KL}}(P\|Q) = 0.25$ bits.](../img/mdl-it-code-length-bars.svg)
:label:`fig_mdl-code-length-bars`

The following example verifies :eqref:`eq_mdl-ce_decomp` numerically on two
small categorical distributions, and check the asymmetry of KL while we are at
it.

```{.python .input #information-theory-kl-categorical}
import numpy as onp
p = onp.array([0.6, 0.3, 0.1])
q = onp.array([0.2, 0.5, 0.3])

ce_pq = float(-onp.sum(p * onp.log(q)))
print(f'KL(P||Q) = {kl_divergence(p, q):.4f} nats, '
      f'KL(Q||P) = {kl_divergence(q, p):.4f} nats')
print(f'CE(P, Q) - H(P) = {ce_pq - float(entropy(p)):.4f} nats')
```

The forward and reverse KL disagree ($\approx 0.396$ vs. $\approx 0.365$
nats), and $\textrm{CE} - H$ reproduces the forward KL exactly, as
:eqref:`eq_mdl-ce_decomp` says it must.

### The Classification Loss

In a $k$-class classification problem, the "truth" for one example is the
one-hot distribution that puts all its mass on the correct label $y$, and the
model supplies a predicted distribution
$\hat{\mathbf{y}} = (\hat{y}_1, \ldots, \hat{y}_k)$. Cross-entropy
:eqref:`eq_mdl-ce_def` between them collapses to a single term,
$-\log \hat{y}_{y}$: the model's surprise at the correct class. Averaged over
$n$ examples, the *cross-entropy loss* is

$$
\textrm{CE}(\mathbf{y}, \hat{\mathbf{y}})
= -\frac{1}{n} \sum_{i=1}^n \log \hat{y}_{i, y_i},
$$

in nats, the mathematical quantity a built-in classification loss computes.

This display is the definition. Training code should supply logits to a
from-logits loss, which performs the log-softmax without first rounding small
probabilities to zero; :numref:`subsec_mdl-stable-softmax` derives that
implementation.

```{.python .input #information-theory-formal-definition-1}
import numpy as onp
def cross_entropy(y_hat, y):
    """Mean cross-entropy loss for predicted probabilities, in nats."""
    ce = -onp.log(y_hat[range(len(y_hat)), y])
    return float(ce.mean())
```

Two examples, three classes: the first example's true class is $0$, predicted
with probability $0.3$; the second's is $2$, predicted with probability $0.5$.
The loss is $\tfrac{1}{2}(-\ln 0.3 - \ln 0.5) \approx 0.949$ nats.

```{.python .input #information-theory-formal-definition-2}
import numpy as onp
labels = onp.array([0, 2])
preds = onp.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

The built-in loss computes the same number from *logits* $\mathbf{z}$,
applying log-softmax internally. We construct $\mathbf{z}=\log\mathbf{q}$
only because the small example above starts from normalized probabilities;
in a model, $\mathbf{z}$ is the raw output of the final layer. Softmax inverts
the log for a normalized $\mathbf{q}$:
$\mathrm{softmax}(\log \mathbf{q})_j = e^{\log q_j} / \sum_i e^{\log q_i}
= q_j / \sum_i q_i = q_j$. (More generally, softmax is invariant to adding a
constant to all logits, and $\log$ of a normalized vector is one valid logit
vector among many.) Passing probabilities to a loss intended for logits
applies softmax twice; applying `log` to a computed softmax can underflow.

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab mxnet
preds_mx = np.array(preds.tolist())
labels_mx = np.array(labels.tolist())
logits = np.log(preds_mx)
log_probs = npx.log_softmax(logits, axis=1)
-log_probs[np.arange(len(labels_mx)), labels_mx].mean()
```

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab pytorch
preds_torch = torch.tensor(preds, dtype=torch.float32)
labels_torch = torch.tensor(labels)
logits = torch.log(preds_torch)
torch.nn.CrossEntropyLoss()(logits, labels_torch)
```

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab tensorflow
preds_tf = tf.convert_to_tensor(preds, dtype=tf.float32)
labels_tf = tf.convert_to_tensor(labels)
logits = tf.math.log(preds_tf)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(labels_tf, logits).numpy()
```

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab jax
preds_jax = jnp.asarray(preds)
labels_jax = jnp.asarray(labels)
logits = jnp.log(preds_jax)
optax.softmax_cross_entropy_with_integer_labels(
    logits, labels_jax).mean()
```

The loss also has a decision-theoretic pedigree. A
*scoring rule* assigns a penalty $S(\mathbf{q}, y)$ for having reported the
probability vector $\mathbf{q}$ when class $y$ then occurs. The rule is
*proper* if truthful reporting is optimal: when the data-generating distribution is $\mathbf{p}$, the
expected penalty $E_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$ is minimized by
reporting $\mathbf{q} = \mathbf{p}$, and *strictly proper* if this report is the
*unique* optimum. The cross-entropy loss is the *log score*
$S(\mathbf{q}, y) = -\log q_y$, and it is strictly proper: its expected
penalty is exactly
$\textrm{CE}(\mathbf{p}, \mathbf{q}) = H(\mathbf{p}) +
D_{\textrm{KL}}(\mathbf{p} \| \mathbf{q})$, which Gibbs' inequality minimizes
uniquely at $\mathbf{q} = \mathbf{p}$. Strict propriety of the log score *is*
Gibbs' inequality restated;
Exercise :exref:`ex_mdl-information-theory-scoring-rule-is-strictly` applies the same
test to the Brier score and to an improper rule.

Why insist on *strict* propriety? At population risk, over a model class that
can represent the true conditional distribution, the log score is minimized
by $p(y \mid \mathbf{x})$ itself. That is the basis for calibration
(predicted probabilities matching observed frequencies). Finite data, a
restricted model class, regularization, imperfect optimization, and
distribution shift can still produce miscalibration. Nor is the log score the
only strictly proper rule: the *Brier score*
$S(\mathbf{q}, y) = \sum_j (q_j - \mathbf{1}[j = y])^2$ penalizes the whole
vector by squared error, and its expected penalty exceeds the truthful
reporter's by $\|\mathbf{q} - \mathbf{p}\|^2$: a squared-distance waste term
in place of the KL-flavored one. Different strictly proper rules disagree
about how severely to punish which mistakes (the log score's penalty diverges
on confident errors; the Brier score's stays bounded), but they agree about
where the optimum is: the truth :cite:`Gneiting.Raftery.2007`.

For discrete observations, the average negative log-likelihood is exactly the
cross-entropy from the empirical mass function to the model; its empirical
entropy is constant in the model, so minimizing NLL also minimizes the
corresponding empirical KL. For continuous observations, the atomic empirical
measure is generally singular with respect to a model density, and that KL is
not finite. The sample NLL remains a valid empirical objective. At the population
level, if the data have density $p$, its expectation is the cross-entropy
$-E_p[\log q_{\boldsymbol\theta}]$ and differs from
$D_{\mathrm{KL}}(p\|q_{\boldsymbol\theta})$ by the model-independent differential
entropy of $p$ when these quantities are finite.

## Modern Uses

Two modern training techniques follow directly from these identities; both
are corollaries of Gibbs' inequality.

### Label Smoothing

A classifier trained on one-hot targets is asked to assign probability $1$ to
the true class, which a softmax can approach only by driving the true class's
logit infinitely far above the rest. The result is overconfident models and
ever-growing weights. *Label smoothing* :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`
replaces the one-hot target $\mathbf{e}_y$ with a mixture that reserves a
small probability $\epsilon$ for the other classes:

$$
\mathbf{p}^\epsilon = (1 - \epsilon)\, \mathbf{e}_y + \frac{\epsilon}{k}\, \mathbf{1},
$$
:eqlabel:`eq_mdl-label-smoothing-target`

i.e., target probability $1 - \epsilon + \epsilon/k$ for the true class and
$\epsilon/k$ for each of the other $k - 1$. The training loss is the ordinary
cross-entropy $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ to this softened
target. What prediction minimizes the smoothed loss?

**Proposition (the optimal smoothed prediction).** *Over all probability
vectors $\mathbf{q}$, the cross-entropy
$\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ is minimized uniquely at
$\mathbf{q}^* = \mathbf{p}^\epsilon$, where it equals $H(\mathbf{p}^\epsilon)$.*

**Proof.** By :eqref:`eq_mdl-ce_decomp`,
$\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q}) = H(\mathbf{p}^\epsilon) +
D_{\textrm{KL}}(\mathbf{p}^\epsilon \| \mathbf{q})$. The first term does not
depend on $\mathbf{q}$, and by Gibbs' inequality the second is non-negative
and zero iff $\mathbf{q} = \mathbf{p}^\epsilon$. $\blacksquare$

So the optimum is no longer at infinity: the model is asked to predict
$1 - \epsilon + \epsilon/k$ on the true class, which a softmax reaches with the
*finite* logit gap

$$
z_y - z_j = \log\frac{1 - \epsilon + \epsilon/k}{\epsilon/k}
= \log\!\Big(1 + \frac{k(1-\epsilon)}{\epsilon}\Big)
$$

between the true class and each other class. For $k = 10$ classes and
$\epsilon = 0.1$ the gap is $\ln 91 \approx 4.51$, a concrete, attainable
target instead of a runaway one. (A logit difference is a log-odds ratio, so
we leave it unitless rather than calling it nats.) The loss at the optimum
is the (nonzero) entropy $H(\mathbf{p}^\epsilon)$, which is why a
label-smoothed loss curve plateaus above zero even when the model is doing
what the softened target requests. The following calculation checks both numbers:

```{.python .input #information-theory-label-smoothing}
k, eps = 10, 0.1
p_eps = [1 - eps + eps / k] + [eps / k] * (k - 1)

gap = math.log(p_eps[0] / p_eps[1])              # optimal logit gap
h_p_eps = -sum(t * math.log(t) for t in p_eps)   # CE at the optimum q* = p_eps

z = [10.0] + [0.0] * (k - 1)                     # an overconfident prediction
z_norm = sum(math.exp(v) for v in z)
q_conf = [math.exp(v) / z_norm for v in z]
ce_conf = -sum(t * math.log(s) for t, s in zip(p_eps, q_conf))

print(f'optimal logit gap = ln(91) = {gap:.4f}')
print(f'CE at q* = H(p_eps) = {h_p_eps:.4f} nats')
print(f'CE at the overconfident q = {ce_conf:.4f} nats')
```

The overconfident prediction (logit gap $10$) is *worse* under the smoothed
target than the calibrated one: label smoothing penalizes the very
behavior that one-hot targets demand.

### Knowledge Distillation
:label:`subsec_mdl-distillation`

*Knowledge distillation* :cite:`Hinton.Vinyals.Dean.2015` trains a small
*student* network to imitate a large *teacher*, transferring the teacher's
full distribution over wrong answers rather than its argmax alone (a "2" that
looks somewhat like a "7" is valuable information). Because a trained
teacher's softmax is nearly one-hot, both distributions are first *softened*
with a temperature $T$:

$$
p_j^{(T)} = \mathrm{softmax}(\mathbf{z}^{\textrm{tea}}/T)_j, \qquad
q_j^{(T)} = \mathrm{softmax}(\mathbf{z}^{\textrm{stu}}/T)_j,
$$

and the distillation loss is the KL divergence between the softened
distributions, scaled by $T^2$:

$$
\mathcal{L}_{\textrm{distill}}
= T^2\, D_{\textrm{KL}}\big(\mathbf{p}^{(T)} \,\|\, \mathbf{q}^{(T)}\big).
$$
:eqlabel:`eq_mdl-distill-loss`

Where does the $T^2$ come from? Differentiate the KL term with respect to a
student logit. Since only $-\sum_j p_j^{(T)} \log q_j^{(T)}$ depends on the
student, and the derivative of a log-softmax at temperature $T$ is the usual
softmax-minus-target expression scaled by $1/T$,

$$
\frac{\partial}{\partial z^{\textrm{stu}}_j}
D_{\textrm{KL}}\big(\mathbf{p}^{(T)} \| \mathbf{q}^{(T)}\big)
= \frac{1}{T}\big(q_j^{(T)} - p_j^{(T)}\big).
$$

For large $T$ the softened distributions flatten toward uniform,
$\mathrm{softmax}(\mathbf{z}/T)_j \approx \tfrac{1}{k}\big(1 + (z_j - \bar z)/T\big)$
with $\bar z$ the mean logit, so the difference $q_j^{(T)} - p_j^{(T)}$ itself
shrinks like $1/T$ and the gradient like $1/T^2$:

$$
\frac{1}{T}\big(q_j^{(T)} - p_j^{(T)}\big)
\;\approx\; \frac{1}{k\,T^2}\Big(\big(z^{\textrm{stu}}_j - \bar z^{\textrm{stu}}\big)
- \big(z^{\textrm{tea}}_j - \bar z^{\textrm{tea}}\big)\Big).
$$

Multiplying the loss by $T^2$ therefore keeps the gradient magnitude roughly
constant as the temperature is tuned, so $T$ can be chosen for the *quality*
of the transferred soft targets without silently rescaling the learning rate
against any hard-label loss it is mixed with. The following example verifies both claims
numerically: the closed-form gradient matches autograd, and $T^2$ times the
gradient approaches the limit above instead of vanishing.

```{.python .input #information-theory-distillation}
#@tab mxnet
z_tea = np.array([5.0, 2.0, -1.0])
z_stu = np.array([3.0, 3.0, 0.0])
z_stu.attach_grad()

for T in (1.0, 2.0, 5.0, 10.0):
    with autograd.record():
        p, q = npx.softmax(z_tea / T), npx.softmax(z_stu / T)
        kl = np.sum(p * (np.log(p) - np.log(q)))
    kl.backward()
    closed = (npx.softmax(z_stu / T) - npx.softmax(z_tea / T)) / T
    err = float(np.abs(z_stu.grad - closed).max())
    print(f'T={T:4.1f}  |autograd - closed| = {err:.1e}  '
          f'T^2 grad = {[round(g, 3) for g in (T**2 * z_stu.grad).tolist()]}')
```

```{.python .input #information-theory-distillation}
#@tab pytorch
z_tea = torch.tensor([5.0, 2.0, -1.0])

for T in (1.0, 2.0, 5.0, 10.0):
    z_stu = torch.tensor([3.0, 3.0, 0.0], requires_grad=True)
    p, q = torch.softmax(z_tea / T, 0), torch.softmax(z_stu / T, 0)
    kl = torch.sum(p * (torch.log(p) - torch.log(q)))
    grad, = torch.autograd.grad(kl, z_stu)
    closed = (q - p).detach() / T
    err = (grad - closed).abs().max().item()
    print(f'T={T:4.1f}  |autograd - closed| = {err:.1e}  '
          f'T^2 grad = {[round(g, 3) for g in (T**2 * grad).tolist()]}')
```

```{.python .input #information-theory-distillation}
#@tab tensorflow
z_tea = tf.constant([5.0, 2.0, -1.0])
z_stu = tf.Variable([3.0, 3.0, 0.0])

for T in (1.0, 2.0, 5.0, 10.0):
    with tf.GradientTape() as tape:
        p, q = tf.nn.softmax(z_tea / T), tf.nn.softmax(z_stu / T)
        kl = tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)))
    grad = tape.gradient(kl, z_stu)
    closed = (q - p) / T
    err = float(tf.reduce_max(tf.abs(grad - closed)).numpy())
    print(f'T={T:4.1f}  |autograd - closed| = {err:.1e}  '
          f'T^2 grad = {[round(g, 3) for g in (T**2 * grad).numpy().tolist()]}')
```

```{.python .input #information-theory-distillation}
#@tab jax
z_tea = jnp.array([5.0, 2.0, -1.0])
z_stu = jnp.array([3.0, 3.0, 0.0])

def distill_kl(z_stu, T):
    p, q = jax.nn.softmax(z_tea / T), jax.nn.softmax(z_stu / T)
    return jnp.sum(p * (jnp.log(p) - jnp.log(q)))

for T in (1.0, 2.0, 5.0, 10.0):
    grad = jax.grad(distill_kl)(z_stu, T)
    closed = (jax.nn.softmax(z_stu / T) - jax.nn.softmax(z_tea / T)) / T
    err = jnp.abs(grad - closed).max().item()
    print(f'T={T:4.1f}  |autograd - closed| = {err:.1e}  '
          f'T^2 grad = {[round(g, 3) for g in (T**2 * grad).tolist()]}')
```

The closed form matches autograd to floating-point precision at every
temperature. The raw gradient decays like $1/T^2$, while the $T^2$-scaled
gradient stays order-one: its $T \to \infty$ limit is
$\tfrac{1}{k}\big((\mathbf{z}^{\textrm{stu}} - \bar z^{\textrm{stu}}) -
(\mathbf{z}^{\textrm{tea}} - \bar z^{\textrm{tea}})\big) = (-0.667, 0.333,
0.333)$ here, but the convergence is $O(1/T)$ and still visibly incomplete in
the table: the $T = 10$ row reads $(-0.719, 0.413, 0.306)$, and the first
component even overshoots its limit on the way. The scale-matching holds
regardless, and that is what the $T^2$ factor was designed for. As
$T \to 1$ the loss reduces to the ordinary KL (and, with a one-hot teacher,
to the ordinary cross-entropy loss), recovering standard training as a special
case.

### Relations Among the Objectives

Maximum likelihood, empirical cross-entropy minimization, and KL projection of
the empirical distribution onto the model family are equivalent objectives
(:numref:`subsec_mdl-nll-crossentropy`). Their coding interpretation is an
expected description length, while perplexity is the corresponding
exponentiated per-token loss. Label smoothing and distillation replace the
hard empirical target with different target distributions. The next section
extends KL to other discrepancies between distributions
(:numref:`sec_mdl-divergences-distances`); the final section applies KL to a
joint distribution and the product of its marginals to obtain mutual
information (:numref:`sec_mdl-mutual-information`).

## Summary

* Self-information $I(x) = -\log p(x)$ measures the surprise of an outcome;
  entropy $H(P) = -E_{x\sim P}[\log p(x)]$ is the average surprise of a
  distribution. We measure both in nats (natural log); bits differ by a
  factor of $\ln 2$.

* On $k$ outcomes, $0 \leq H(P) \leq \log k$, with the maximum exactly at the
  uniform distribution (Jensen). Differential entropy $h$ of continuous
  variables is *not* coordinate-invariant and can be negative, one reason deep
  learning prefers relative quantities; under fixed mean and variance it is
  maximized by the Gaussian (:numref:`subsec_mdl-gaussian-max-entropy`).

* The KL divergence $D_{\textrm{KL}}(P\|Q) = E_{x\sim P}[\log p(x)/q(x)]$ is
  asymmetric and, by Gibbs' inequality, non-negative with equality iff
  $P = Q$.

* Cross-entropy decomposes as
  $\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P\|Q)$. Minimizing it in $Q$ is
  minimizing KL, and on empirical data it is maximum likelihood
  (:numref:`subsec_mdl-nll-crossentropy`).

* As a scoring rule, the cross-entropy loss (the log score) is *strictly
  proper*: at population risk, a model class containing the truth is uniquely
  minimized by the true conditional probabilities. Finite data, restricted
  models, and distribution shift can still produce miscalibration.

* Label smoothing and knowledge distillation are corollaries of the same
  identities: the smoothed target makes the cross-entropy optimum finite and
  equal to the deliberately softened target; it does not by itself guarantee
  calibration to the original data distribution, and the $T^2$ factor on the distillation KL keeps
  gradients scale-matched across temperatures.

## Exercises

1. **Gaussian KL divergence.** Let $P = \mathcal{N}(\mu_1, \sigma_1^2)$.
    1. Derive the closed form :eqref:`eq_mdl-gaussian_kl` by writing out
       $E_{x\sim P}[\log p(x) - \log q(x)]$ and using
       $E_{x \sim P}[x] = \mu_1$ and $E_{x\sim P}[(x-\mu_1)^2] = \sigma_1^2$.
    1. Let $Q$ have mean $\mu_1 + \delta$ and standard deviation
       $\sigma_1 e^{\eta}$. Expand both $D_{\textrm{KL}}(P\|Q)$ and
       $D_{\textrm{KL}}(Q\|P)$ to second order in $\delta$ and $\eta$. At
       which order do the two directions first differ?
    1. Evaluate the second-order approximation for $P = \mathcal{N}(0, 1)$
       and $Q = \mathcal{N}(0, 4)$, and compare it with the exact values
       $0.318$ and $0.807$ nats of the two directions.
   :exlabel:`ex_mdl-information-theory-gaussian-kl`

1. **Other scoring rules.** A scoring rule $S(\mathbf{q}, y)$ is strictly
   proper if the expected penalty $E_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$
   is uniquely minimized at $\mathbf{q} = \mathbf{p}$. For the log score,
   this property is Gibbs' inequality.
    1. Show that the expected Brier score, with
       $S(\mathbf{q}, y) = \sum_j (q_j - \mathbf{1}[j = y])^2$, exceeds the
       expected Brier score of the truthful report by exactly
       $\|\mathbf{q} - \mathbf{p}\|^2$, so that the Brier score is strictly
       proper :cite:`Gneiting.Raftery.2007`.
    1. The linear score $S(\mathbf{q}, y) = -q_y$ rewards the probability
       placed on the observed class. Which report minimizes its expected
       penalty? Is the linear score proper? What does a classifier trained
       with it report for an input whose label is class $1$ with probability
       $0.6$ and class $2$ with probability $0.4$?
   :exlabel:`ex_mdl-information-theory-scoring-rule-is-strictly`

1. **Uniqueness of self-information.** ● Show that self-information is
   unique up to scale: if a continuous, decreasing function $I(p)$ on
   $(0, 1]$ satisfies $I(1) = 0$ and $I(p_1 p_2) = I(p_1) + I(p_2)$ for all
   $p_1, p_2 \in (0, 1]$, then $I(p) = -c \log p$ for some constant
   $c > 0$. Work with the function $g(t) = I(e^{-t})$ for $t \geq 0$.
   :exlabel:`ex_mdl-information-theory-show-that-self-information`

1. **Label smoothing on ambiguous inputs.** ● Label smoothing trains with
   the target :eqref:`eq_mdl-label-smoothing-target`.
    1. In the training distribution, an input $\mathbf{x}$ carries the
       label $y$ with probability $p(y \mid \mathbf{x})$. Show that the
       expected smoothed loss for this input is minimized by
       $\mathbf{q}^*(\cdot \mid \mathbf{x}) = (1 - \epsilon)\,
       p(\cdot \mid \mathbf{x}) + \tfrac{\epsilon}{k}\mathbf{1}$. How can
       $p(\cdot \mid \mathbf{x})$ be recovered from $\mathbf{q}^*$? What is
       the largest probability that the minimizer can assign to a class?
    1. For one example with logits $\mathbf{z}$ and
       $\mathbf{q} = \mathrm{softmax}(\mathbf{z})$, the gradient of
       $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ with respect to $z_j$
       is $q_j - p^\epsilon_j$. For $\epsilon > 0$, show that all
       wrong-class logits are equal at every stationary point.
    1. With one-hot targets ($\epsilon = 0$), the gradient with respect to a
       wrong-class logit is $q_j$, and the loss has no stationary point.
       Show that gradient descent on the logits of one example still shrinks
       the gap between two wrong-class logits, and explain why this
       shrinkage stalls as $q_y \to 1$. In a network, the logits of all
       examples are computed from shared features. Which differences between
       wrong-class logits can therefore survive training with
       $\epsilon = 0$? Why does training with $\epsilon > 0$ remove them?
    1. :citet:`Muller.Kornblith.Hinton.2019` observe that teachers trained
       with label smoothing produce worse students under the distillation
       loss :eqref:`eq_mdl-distill-loss`. Explain this observation with the
       results of (b) and (c).
   :exlabel:`ex_mdl-information-theory-label-smoothing`

1. [code] **Distillation at high temperature.** ●
   :citet:`Hinton.Vinyals.Dean.2015` state in their Section 2.1 that, at
   high temperature, distillation reduces to matching zero-mean logits by
   squared error.
    1. Problem :exref:`ex_mdl-information-theory-two-part-coin`a expands
       the KL divergence of two coins to second order. For $k$ classes the
       same expansion reads

       $$
       D_{\textrm{KL}}(\mathbf{p} \| \mathbf{q}) =
       \frac{1}{2}\sum_j \frac{(q_j - p_j)^2}{p_j}
       + O\big(\|\mathbf{q} - \mathbf{p}\|^3\big).
       $$

       Combine this expansion with the large-$T$ approximation of
       $\mathrm{softmax}(\mathbf{z}/T)$ in
       :numref:`subsec_mdl-distillation` to show that, as $T \to \infty$,

       $$
       T^2 D_{\textrm{KL}}\big(\mathbf{p}^{(T)} \,\|\, \mathbf{q}^{(T)}\big)
       \to \frac{1}{2k} \sum_{j=1}^k
       \Big(\big(z^{\textrm{stu}}_j - \bar z^{\textrm{stu}}\big)
       - \big(z^{\textrm{tea}}_j - \bar z^{\textrm{tea}}\big)\Big)^2.
       $$

    1. Evaluate $T^2 D_{\textrm{KL}}(\mathbf{p}^{(T)} \| \mathbf{q}^{(T)})$
       for $T = 1, 2, 5, 10, 100$ with the logits `z_tea` $= (5, 2, -1)$ and
       `z_stu` $= (3, 3, 0)$ of :numref:`subsec_mdl-distillation`, and
       compare it with the limit.
    1. At low temperature, which teacher logits have almost no influence on
       the loss? :citet:`Hinton.Vinyals.Dean.2015` argue that this can be an
       advantage and also a drawback. Give one argument for each side,
       based on what the teacher's own cross-entropy training does to these
       logits.
   :exlabel:`ex_mdl-information-theory-distillation-high-temperature`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/420)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1105)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1105)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §26.1]{.kicker}

The reported loss is a code length<br>**entropy, cross-entropy, and KL divergence**.
:::
:::

::: {.slide title="Three numbers in every loss"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
The scalar reported by a training loop is a *code length* in nats (Shannon,
1948). It splits into three pieces:

- **entropy** $H(P)$: the irreducible expected code length,
- **cross-entropy** $\mathrm{CE}(P,Q)$: the expected length under the model,
- **KL** $D_{\mathrm{KL}}(P\|Q)$: their difference.
:::

::: {.col .fig}
@fig:mdl-it-code-length-bars
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Surprise and entropy]{.dtitle}

[self-information, Shannon's $H$, the uniform maximum]{.dsub}
:::
:::

::: {.slide title="Rare events are surprising"}
[Self-information]{.kicker}

::: {.cols .vc}
::: {.col}
$I(x) = -\log p(x)$ is the *only* measure that is zero for a certain event,
decreasing in $p$, and additive over independent events; additivity is
exactly what forces a logarithm.

The fair coin carries $\ln 2 \approx 0.693$ nats; a one-in-a-million event,
$\approx 13.8$.
:::

::: {.col .fig}
@fig:mdl-it-self-info-curve
:::
:::
:::

::: {.slide title="Entropy averages the surprise"}
[Entropy]{.kicker}

$H(P) = \mathbb{E}_P[-\log p] = -\sum_x p(x)\log p(x)$, with the convention
$0\log 0 = 0$; it is the expected length of the *best possible* code for $P$:

@information-theory-definition

Less than $\ln 4 \approx 1.386$ nats: this distribution is not uniform.
:::

::: {.slide title="Uncertainty peaks at the uniform law"}
[The maximum]{.kicker}

::: {.cols .vc}
::: {.col}
On $k$ outcomes, $0 \le H(P) \le \log k$, with the maximum *exactly* at the
uniform distribution.

::: {.d2l-note .rule}
*Proof.* $H = \mathbb{E}[\log\tfrac{1}{p}] \le \log\mathbb{E}[\tfrac1p]
= \log m \le \log k$, where $m$ counts the outcomes with $p_i>0$, by Jensen
on the *strictly* concave $\log$. Equality forces $1/p$ constant **and** $m=k$:
the uniform law, exactly. $\blacksquare$
:::
:::

::: {.col .fig}
@fig:mdl-it-bernoulli-entropy
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Cross-entropy and KL]{.dtitle}

[the trainable gap: Gibbs' inequality]{.dsub}
:::
:::

::: {.slide title="KL divergence as excess code length"}
[Relative entropy]{.kicker}

$D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_{x\sim P}\bigl[\log p(x)/q(x)\bigr]$: the
extra nats from coding $P$'s data with $Q$'s code. Asymmetric, and $+\infty$
where $q(x)=0$ but $p(x)>0$:

@information-theory-definition-2
:::

::: {.slide title="Gibbs' inequality"}
[The one fact]{.kicker}

::: {.d2l-note .rule}
$D_{\mathrm{KL}}(P\|Q) \ge 0$, with equality **iff** $P=Q$.
:::

*Proof.* $D_{\mathrm{KL}} = \mathbb{E}_P[-\log\tfrac{q}{p}]
\ge -\log\mathbb{E}_P[\tfrac{q}{p}] = -\log Q(\mathrm{supp}\,P) \ge 0$, by
Jensen on the *strictly* convex $-\log$; the sum collects at most all of $Q$'s
mass, never more. Equality needs $q/p$ constant **and**
$Q(\mathrm{supp}\,P)=1$, i.e. $P=Q$. $\blacksquare$

This inequality supports the results developed below.
:::

::: {.slide title="Cross-entropy = entropy + KL"}
[The decomposition]{.kicker}

$\mathrm{CE}(P,Q) = -\mathbb{E}_P[\log q] = H(P) + D_{\mathrm{KL}}(P\|Q)$, so
Gibbs gives $\mathrm{CE} \ge H(P)$, and minimizing CE in $Q$ *is* minimizing KL:

@information-theory-kl-categorical

The forward KL equals $\mathrm{CE}-H$ numerically; KL is asymmetric.
:::

::: {.slide title="Gaussian KL, in closed form"}
[A check]{.kicker}

$D_{\mathrm{KL}}\!\bigl(\mathcal N(\mu_1,\sigma_1^2)\,\|\,\mathcal N(\mu_2,\sigma_2^2)\bigr)
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$
matches Monte Carlo:

@information-theory-example-2

. . .

Swapping the arguments changes the number, $0.318$ vs $0.807$ nats:

@information-theory-example-3
:::

::: {.slide title="Cross-entropy is the classification loss"}
[Maximum likelihood]{.kicker}

A one-hot target collapses CE to $-\log\hat y_{\text{true}}$, the model's
surprise at the correct class, averaged over the batch:

@information-theory-formal-definition-2

This is exactly the negative log-likelihood.
:::

::: {.slide title="The same number the built-in loss prints"}
[Core identity]{.kicker}

The direct cross-entropy calculation equals the built-in loss:

@information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification

::: {.d2l-note}
Minimizing CE $=$ maximum likelihood $=$ KL-projection of the empirical
distribution onto the model family. These are three interpretations of the
same objective.
:::
:::

::: {.slide title="Truthful reporting is the unique optimum"}
[Proper scoring rules]{.kicker}

A scoring rule $S(\mathbf q, y)$ is **strictly proper** if reporting the truth
$\mathbf p$ is the *unique* minimizer of the expected penalty. The log score
$S = -\log q_y$ qualifies: its expected penalty is exactly

$$\mathrm{CE}(\mathbf p,\mathbf q) = H(\mathbf p) + D_{\mathrm{KL}}(\mathbf p\|\mathbf q),$$

which Gibbs minimizes uniquely at $\mathbf q = \mathbf p$.

. . .

::: {.d2l-note}
At population risk, strict propriety makes the true conditional distribution
the unique optimum. Finite data, a restricted model class, or distribution
shift can still leave a classifier miscalibrated. The Brier score
$\sum_j (q_j - \mathbf 1[j{=}y])^2$ is also strictly proper, with the same
optimum and a bounded penalty on confident errors.
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Three corollaries of Gibbs]{.dtitle}

[label smoothing, distillation, and related objectives]{.dsub}
:::
:::

::: {.slide title="Label smoothing softens the target"}
[Calibration]{.kicker}

A smoothed target $(1-\epsilon)\,\mathbf e_y + \tfrac{\epsilon}{k}\mathbf 1$
makes the CE optimum a *finite* logit gap (Gibbs), so the model can no longer
chase infinite confidence:

@!information-theory-label-smoothing

The loss floors at $H(\mathbf p^\epsilon) > 0$, not zero.
:::

::: {.slide title="Distillation matches a soft teacher"}
[Transfer]{.kicker}

The student minimizes $T^2\,D_{\mathrm{KL}}(\text{teacher}_T\,\|\,\text{student}_T)$;
the $T^2$ cancels the $1/T^2$ gradient shrinkage at temperature $T$, keeping
the update scale-matched. Autograd confirms the closed form:

@!information-theory-distillation
:::

::: {.slide title="Relations among training objectives"}
[Synthesis]{.kicker}

::: {.d2l-note .rule}
Maximum likelihood $=$ cross-entropy minimization $=$ KL-projection of the data
onto the model. Label smoothing and distillation follow from the same Gibbs
inequality after replacing a hard target with a softened distribution.
:::

Once the target distribution is specified, Gibbs' inequality identifies both
the optimum and the irreducible loss.
:::

::: {.slide title="Information Measures Link Prediction and Compression"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- Self-information $-\log p$; entropy $H$ averages it and tops out at $\log k$ (uniform).
- KL $\ge 0$ follows from Jensen's inequality and supports the subsequent
  results.
- $\mathrm{CE} = H + \mathrm{KL}$; minimizing it is maximum likelihood.
:::

::: {.col}
- Kraft + Shannon: KL is the excess expected length of a mismatched code.
- Arithmetic coding approaches the surprisal bound: prediction is compression.
- Perplexity $= \exp(\mathrm{CE})$, a base-free branching factor.
- Label smoothing, distillation, proper scoring: corollaries of Gibbs.
:::
:::

::: {.d2l-note}
Next: the wider family of **divergences and distances** that define modern
generative objectives.
:::
:::
