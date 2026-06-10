# Entropy, Cross-Entropy, and KL Divergence
:label:`sec_mdl-information_theory`

Nearly every model in this book is trained by minimizing a cross-entropy. The
number your training loop prints is therefore a quantity from *information
theory*, the field Claude Shannon created in a single 1948 paper
:cite:`Shannon.1948`, and information theory is what tells you what that number
*means*: it is a code length. Entropy is the irreducible floor---the cost of
the data's own randomness; cross-entropy is what your model actually pays; and
the gap between them, the Kullback--Leibler divergence, is the waste you can
train away. This section builds those three quantities and the one inequality
(Gibbs') that relates them, grounds the "extra bits" language in an actual
coding argument (the Kraft inequality and the Shannon code), explains why
language models report *perplexity*, and closes with two modern training tricks
---label smoothing and knowledge distillation---that are one-line consequences
of the same machinery.

A note on units before we start. The logarithm base is a pure choice of unit:
base $2$ gives *bits*, base $e$ gives *nats*. **We work in nats throughout**,
matching deep-learning practice---every framework's `log` is natural, so the
loss your training loop prints is already in nats. The one place where base
$2$ is genuinely natural is coding with binary symbols, and we flag it
explicitly when we get there. Since $\log_2 x = \ln x / \ln 2$, converting is a
fixed rescaling: $1\textrm{ bit} = \ln 2 \approx 0.693$ nats and $1\textrm{
nat} = 1/\ln 2 \approx 1.443$ bits. No argmin in this section---and no trained
model---changes if you switch.

We import each framework once, up front; everything below uses only these
names.

```{.python .input #information-theory-imports}
#@tab mxnet
import math
from mxnet import autograd, np, npx
from mxnet.gluon.metric import CrossEntropy
from mxnet.ndarray import nansum
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

What should a numerical measure of "information" even be? Consider a thought
experiment. A friend shuffles a deck of cards, flips some over, and reports
what they see; we try to assess how much each report tells us.

First they flip a card and say, "I see a card." This tells us nothing---we
were already certain of it---so whatever our measure is, this statement should
score zero. Next: "I see a heart." Four equally likely suits were possible, so
this is mildly informative. Then: "It is the three of spades." Now one of $52$
equally likely outcomes has been pinned down---more information. Finally they
read off the order of the entire shuffled deck: one outcome out of $52!$, a
huge amount of information.

The pattern is that information tracks *surprise*: the less probable the
event, the more we learn from observing it. Shannon turned this into a
definition. The *self-information* of an event $x$ with probability $p(x)$ is

$$
I(x) = -\log p(x),
$$

measured in nats for the natural logarithm. The four card reports above carry
$-\ln 1 = 0$, $\ln 4 \approx 1.39$, $\ln 52 \approx 3.95$, and
$\ln 52! \approx 156.4$ nats respectively. The function $-\log p$ is the
unique choice (up to the base, i.e., the unit) satisfying the properties we
implicitly demanded: certain events carry zero information, rarer events carry
more, and independent events add---$I$ of a joint outcome with probability
$p_1 p_2$ is $I(p_1) + I(p_2)$, which forces a logarithm
(:citet:`Csiszar.2008` gives the formal axiomatics; Exercise 5 walks
through the argument).

**Bits, a sidebar.** Take the logarithm base $2$ instead and the unit is the
*bit*: the information in one fair coin flip, or equivalently one binary
digit, since a uniformly random length-$n$ bit string such as "0010" has
probability $2^{-n}$ and self-information $-\log_2 2^{-n} = n$ bits. In bits
the card-deck reports carry $0$, $2$, $\log_2 52 \approx 5.70$, and
$\log_2 52! \approx 225.6$ bits---the classic numbers. Bits will return when
we talk about actual binary codes in the coding view below; everywhere else we
stay in nats.

![Self-information $I(x) = -\log p(x)$ in nats as a function of the probability $p$. Certain events ($p = 1$) carry zero information, the fair coin ($p = \tfrac{1}{2}$) carries $\ln 2 \approx 0.693$ nats, and the curve diverges as $p \to 0$: rare means surprising.](../img/mdl-it-self-info-curve.svg)
:label:`fig_mdl-self-info-curve`

:numref:`fig_mdl-self-info-curve` plots the curve: decreasing, convex, zero at
$p = 1$, infinite at $p = 0$. Computing it is one line.

```{.python .input #information-theory-self-information}
#@tab mxnet
def self_information(p):
    """Self-information -log p, in nats."""
    return -np.log(p)

self_information(1 / 64)
```

```{.python .input #information-theory-self-information}
#@tab pytorch
def self_information(p):
    """Self-information -log p, in nats."""
    return -torch.log(torch.tensor(p)).item()

self_information(1 / 64)
```

```{.python .input #information-theory-self-information}
#@tab tensorflow
def self_information(p):
    """Self-information -log p, in nats."""
    return -tf.math.log(tf.constant(p)).numpy()

self_information(1 / 64)
```

```{.python .input #information-theory-self-information}
#@tab jax
def self_information(p):
    """Self-information -log p, in nats."""
    return -jnp.log(jnp.array(p)).item()

self_information(1 / 64)
```

An event of probability $1/64$ carries $\ln 64 \approx 4.16$ nats (exactly
$6$ bits: it takes six fair-coin flips to have probability $1/64$).

### Shannon Entropy

Self-information scores a single outcome. To score a *random variable*---a
whole distribution of outcomes---we average. For $X \sim P$ with probability
mass function (p.m.f.) or probability density function (p.d.f.) $p(x)$, the
*entropy* (or *Shannon entropy*) of $X$ is the expected self-information,

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_mdl-ent_def`

For discrete $X$ this reads $H(X) = -\sum_i p_i \log p_i$ with
$p_i = P(X = x_i)$; for continuous $X$ the sum becomes an integral,
$H(X) = -\int p(x) \log p(x)\,dx$, called the *differential entropy*. Each
term weighs the surprise $-\log p(x)$ of an outcome by how often it occurs, so
entropy is the *average surprise* of observing $X$: a distribution
concentrated on one value never surprises us ($H = 0$), while a spread-out
distribution surprises us constantly. Why exactly this form? The logarithm is
forced by additivity over independent observations, the minus sign makes the
measure positive and decreasing in probability, and the expectation is the
only consistent way to aggregate outcome-level surprise into a single number
for the distribution---this is the content of the axiomatic characterizations
mentioned above.

A word of caution about the continuous case. Differential entropy is *not*
the limit of the discrete entropy, and it is *not coordinate-invariant*: under
an invertible change of variables $y = g(x)$ it shifts by
$E[\log|\det \partial g/\partial x|]$, so it can even be *negative* (e.g., a
narrow Gaussian with $\sigma < 1/\sqrt{2\pi e}$). This is one reason deep
learning almost always works with *relative* quantities such as cross-entropy
and KL divergence, which *are* invariant under reparameterization---the
offending Jacobian term cancels between the two densities.

In code, entropy needs one piece of care: the convention $0 \log 0 = 0$ (an
outcome of probability zero contributes nothing). Where $p(x) = 0$ the term
$-p \log p$ has limiting value $0$, but the direct floating-point expression
`0 * -inf` evaluates to `nan`, so we sum with `nansum`, which drops those
terms---exactly the convention we want.

```{.python .input #information-theory-definition}
#@tab mxnet
def entropy(p):
    """Entropy of a probability vector, in nats."""
    ent = -p * np.log(p)
    # `nansum` encodes the convention 0 log 0 = 0
    return nansum(ent.as_nd_ndarray()).asscalar()

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input #information-theory-definition}
#@tab pytorch
def entropy(p):
    """Entropy of a probability vector, in nats."""
    ent = -p * torch.log(p)
    # `nansum` encodes the convention 0 log 0 = 0
    return torch.nansum(ent).item()

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input #information-theory-definition}
#@tab tensorflow
def nansum(x):
    """Sum, skipping nan entries."""
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x))

def entropy(p):
    """Entropy of a probability vector, in nats."""
    # `nansum` encodes the convention 0 log 0 = 0
    return float(nansum(-p * tf.math.log(p)).numpy())

entropy(tf.constant([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input #information-theory-definition}
#@tab jax
def entropy(p):
    """Entropy of a probability vector, in nats."""
    # `nansum` encodes the convention 0 log 0 = 0
    return jnp.nansum(-p * jnp.log(p)).item()

entropy(jnp.array([0.1, 0.5, 0.1, 0.3]))
```

The distribution $(0.1, 0.5, 0.1, 0.3)$ has entropy $\approx 1.168$ nats:
less than the $\ln 4 \approx 1.386$ nats of a uniform distribution on four
outcomes, because the mass is unevenly spread. That comparison is no accident.

### Entropy Is Maximized by the Uniform Distribution

Two basic properties orient everything that follows. First, for discrete $X$
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
\leq \log k.
$$

Because $\log$ is *strictly* concave, the first inequality is an equality only
when $1/p(X)$ is constant almost surely, i.e., $p_i = 1/k$ for all $i$.
$\blacksquare$

For continuous $X$ the analogous statements need constraints to be true at all
(on a bounded interval the uniform density again maximizes differential
entropy; under a variance constraint the Gaussian does---see
:numref:`sec_mdl-distributions`).

The cleanest picture of the proposition is the coin. A Bernoulli variable with
heads-probability $p$ has the *binary entropy*
$H(p) = -p \log p - (1-p)\log(1-p)$, plotted in
:numref:`fig_mdl-bernoulli-entropy`: zero at the deterministic endpoints
$p \in \{0, 1\}$, concave in between, and maximal at the fair coin
$p = \tfrac{1}{2}$, where it equals $\ln 2 \approx 0.693$ nats ($= 1$ bit,
consistent with the $\log k$ bound for $k = 2$). A biased coin with $p = 0.1$
manages only $H(0.1) \approx 0.325$ nats---you are rarely surprised by a coin
that almost always lands tails.

![The binary entropy $H(p) = -p \log p - (1-p) \log (1-p)$ in nats, concave and symmetric about $p = \tfrac{1}{2}$, where it peaks at $\ln 2 \approx 0.693$ nats. At the deterministic endpoints $p = 0$ and $p = 1$ the outcome carries no surprise and the entropy vanishes.](../img/mdl-it-bernoulli-entropy.svg)
:label:`fig_mdl-bernoulli-entropy`

Entropy also extends from one random variable to several---the joint entropy
$H(X, Y)$, the conditional entropy $H(Y \mid X)$, and the *mutual information*
$I(X; Y)$ that measures what $X$ and $Y$ share. Those quantities power
contrastive and self-supervised learning, and we develop them in their own
section, :numref:`sec_mdl-mutual-information`.

## Cross-Entropy and KL Divergence

### The Kullback--Leibler Divergence

In :numref:`sec_linear-algebra` we measured distances between points with
norms. We now want a notion of "distance" between *distributions*: how badly
does a model $Q$ misrepresent a truth $P$? Information theory's answer is the
*Kullback--Leibler (KL) divergence*, also called *relative entropy*. Given a
random variable $X \sim P$ with p.d.f. or p.m.f. $p(x)$, and a second
distribution $Q$ with p.d.f. or p.m.f. $q(x)$ that we use to approximate $P$,

$$D_{\textrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_mdl-kl_def`

The term inside the expectation, $\log p(x) - \log q(x)$, is the difference of
two surprises: how surprised $Q$ is by the outcome $x$, minus how surprised
$P$ is. It is positive where $Q$ underestimates ($q(x) < p(x)$: $Q$ is *more*
surprised than it should be) and negative where $Q$ overestimates. KL averages
this *relative surprise* over outcomes drawn from the truth $P$---note the
asymmetry baked into the definition: $P$ supplies both the samples and the
numerator. In general $D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P)$, and
the gap can be dramatic (we exhibit it numerically below); the consequences of
*which* direction you optimize are taken up in
:numref:`sec_mdl-divergences-distances`. One more edge case to keep in mind:
if some outcome has $p(x) > 0$ but $q(x) = 0$---the model declares impossible
something that actually happens---then $D_{\textrm{KL}}(P\|Q) = \infty$.

Here is the discrete case in code, for `p` and `q` given as probability
vectors over the same finite outcome set. KL divergence is non-negative on
its own (Gibbs' inequality, next), so we do **not** wrap the result in
`abs()`---doing so would teach the false idea that KL needs an absolute value
to stay non-negative. The `nansum` is deliberate: where $p(x) = 0$ the term is
the direct floating-point expression for $0 \cdot \log(0/q)$ yields `nan`,
which `nansum` drops to encode $0 \log 0 = 0$; the *other* edge case,
$p(x) > 0$ with $q(x) = 0$, instead yields $+\infty$ (not `nan`), which
`nansum` keeps---so the code correctly returns $+\infty$ for that divergence.

```{.python .input #information-theory-definition-2}
#@tab mxnet
def kl_divergence(p, q):
    """KL(P || Q) for two probability vectors, in nats."""
    kl = p * np.log(p / q)
    return nansum(kl.as_nd_ndarray()).asscalar()
```

```{.python .input #information-theory-definition-2}
#@tab pytorch
def kl_divergence(p, q):
    """KL(P || Q) for two probability vectors, in nats."""
    return torch.nansum(p * torch.log(p / q)).item()
```

```{.python .input #information-theory-definition-2}
#@tab tensorflow
def kl_divergence(p, q):
    """KL(P || Q) for two probability vectors, in nats."""
    return float(nansum(p * tf.math.log(p / q)).numpy())
```

```{.python .input #information-theory-definition-2}
#@tab jax
def kl_divergence(p, q):
    """KL(P || Q) for two probability vectors, in nats."""
    return jnp.nansum(p * jnp.log(p / q)).item()
```

### Gibbs' Inequality

The single most important fact about the KL divergence is that it cannot be
negative. Everything else in this section---cross-entropy as a sound loss, the
optimality of code lengths, the label-smoothing optimum---follows from it.

**Proposition (Gibbs' inequality).** *For any distributions $P$ and $Q$ on
the same space,*

$$
D_{\textrm{KL}}(P\|Q) \geq 0,
$$

*with equality if and only if $P = Q$.*

**Proof.** One application of Jensen's inequality
(:numref:`subsec_mdl-jensen`) to the convex function $-\log$:

$$
D_{\textrm{KL}}(P\|Q)
= E_{x\sim P}\!\left[-\log\frac{q(x)}{p(x)}\right]
\geq -\log E_{x\sim P}\!\left[\frac{q(x)}{p(x)}\right]
= -\log \sum_x p(x)\,\frac{q(x)}{p(x)}
= -\log 1 = 0,
$$

since $\sum_x q(x) = 1$ (the same computation runs with integrals in the
continuous case). Because $-\log$ is *strictly* convex, equality holds only
when the ratio $q(x)/p(x)$ is constant $P$-almost surely; a constant ratio
between two normalized distributions must be $1$, i.e., $P = Q$.
$\blacksquare$

### Gaussians, in Closed Form

The `kl_divergence` above works on *discrete* probability vectors. For
*continuous* distributions the cleanest worked example is the KL divergence
between two univariate Gaussians, which has a closed form. For
$P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$,

$$
D_{\textrm{KL}}\!\left(\mathcal{N}(\mu_1,\sigma_1^2) \,\big\|\, \mathcal{N}(\mu_2,\sigma_2^2)\right)
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}.
$$
:eqlabel:`eq_mdl-gaussian_kl`

These logs are natural logs, so the result is in nats. Two things are worth
reading off :eqref:`eq_mdl-gaussian_kl` directly. First, it is non-negative
and zero exactly when $\mu_1=\mu_2$ and $\sigma_1=\sigma_2$ (i.e., $P=Q$),
just as Gibbs' inequality demands. Second, it is *not symmetric*: swapping the
roles of $P$ and $Q$ changes the value, because the variance ratio and the
mean gap enter asymmetrically. (Deriving :eqref:`eq_mdl-gaussian_kl` is
Exercise 4.)

Let's verify the formula against a direct Monte-Carlo estimate of
$D_{\textrm{KL}}(P\|Q) = E_{x\sim P}[\log p(x) - \log q(x)]$. (Note: no
`abs()` anywhere---the divergence comes out non-negative on its own.)

```{.python .input #information-theory-example-1}
#@tab mxnet
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
    x = np.random.normal(loc=mu1, scale=sigma1, size=(n, ))
    return (gaussian_log_pdf(x, mu1, sigma1)
            - gaussian_log_pdf(x, mu2, sigma2)).mean()
```

```{.python .input #information-theory-example-1}
#@tab pytorch
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
    x = torch.normal(mu1, sigma1, (n, ))
    return (gaussian_log_pdf(x, mu1, sigma1)
            - gaussian_log_pdf(x, mu2, sigma2)).mean().item()
```

```{.python .input #information-theory-example-1}
#@tab tensorflow
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
    x = tf.random.normal((n, ), mu1, sigma1)
    return tf.reduce_mean(gaussian_log_pdf(x, mu1, sigma1)
                          - gaussian_log_pdf(x, mu2, sigma2)).numpy()
```

```{.python .input #information-theory-example-1}
#@tab jax
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
    key = jax.random.PRNGKey(1)
    x = mu1 + sigma1 * jax.random.normal(key, (n, ))
    return (gaussian_log_pdf(x, mu1, sigma1)
            - gaussian_log_pdf(x, mu2, sigma2)).mean().item()
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
numbers differ ($\approx 0.318$ vs. $\approx 0.807$ nats)---KL is a
divergence, not a distance.

```{.python .input #information-theory-example-3}
kl_pq = gaussian_kl(0.0, 1.0, 0.0, 2.0)
kl_qp = gaussian_kl(0.0, 2.0, 0.0, 1.0)

kl_pq, kl_qp
```

### Cross-Entropy

The loss we actually implement in classifiers is not the KL divergence but a
close relative. The *cross-entropy* from $P$ to $Q$ scores outcomes drawn from
the truth $P$ by the model's surprise,

$$\textrm{CE}(P, Q) = - E_{x \sim P} [\log q(x)].$$
:eqlabel:`eq_mdl-ce_def`

Adding and subtracting $H(P)$ inside the expectation splits the model's
surprise into the truth's own surprise plus the relative surprise---that is,

$$\textrm{CE} (P, Q) = H(P) + D_{\textrm{KL}}(P\|Q).$$
:eqlabel:`eq_mdl-ce_decomp`

This single identity, combined with Gibbs' inequality, gives a clean chain
that explains why cross-entropy is *the* loss to minimize:

$$
\underbrace{D_{\textrm{KL}}(P\|Q) \ge 0}_{\textrm{Gibbs}}
\;\Longrightarrow\;
\underbrace{\textrm{CE}(P,Q) \ge H(P)}_{\textrm{cross-entropy} \ \ge\ \textrm{entropy}},
\quad \textrm{equality iff } P = Q.
$$

Entropy $H(P)$ is the irreducible floor; cross-entropy is what you actually
pay; and the gap $\textrm{CE}(P,Q) - H(P) = D_{\textrm{KL}}(P\|Q)$ is the
waste. Minimizing cross-entropy in $Q$ therefore drives $Q \to P$ and squeezes
the waste to zero---which is the same thing as minimizing KL, since $H(P)$
does not depend on $Q$. (The "floor / payment / waste" language is more than a
metaphor: in the coding view below, all three quantities become literal code
lengths.)

Let's verify the decomposition :eqref:`eq_mdl-ce_decomp` numerically on two
small categorical distributions, and check the asymmetry of KL while we are at
it.

```{.python .input #information-theory-kl-categorical}
#@tab mxnet
p = np.array([0.6, 0.3, 0.1])
q = np.array([0.2, 0.5, 0.3])

ce_pq = float(-np.sum(p * np.log(q)))
print(f'KL(P||Q) = {kl_divergence(p, q):.4f} nats, '
      f'KL(Q||P) = {kl_divergence(q, p):.4f} nats')
print(f'CE(P, Q) - H(P) = {ce_pq - float(entropy(p)):.4f} nats')
```

```{.python .input #information-theory-kl-categorical}
#@tab pytorch
p = torch.tensor([0.6, 0.3, 0.1])
q = torch.tensor([0.2, 0.5, 0.3])

ce_pq = -torch.sum(p * torch.log(q)).item()
print(f'KL(P||Q) = {kl_divergence(p, q):.4f} nats, '
      f'KL(Q||P) = {kl_divergence(q, p):.4f} nats')
print(f'CE(P, Q) - H(P) = {ce_pq - entropy(p):.4f} nats')
```

```{.python .input #information-theory-kl-categorical}
#@tab tensorflow
p = tf.constant([0.6, 0.3, 0.1])
q = tf.constant([0.2, 0.5, 0.3])

ce_pq = float(-tf.reduce_sum(p * tf.math.log(q)).numpy())
print(f'KL(P||Q) = {kl_divergence(p, q):.4f} nats, '
      f'KL(Q||P) = {kl_divergence(q, p):.4f} nats')
print(f'CE(P, Q) - H(P) = {ce_pq - entropy(p):.4f} nats')
```

```{.python .input #information-theory-kl-categorical}
#@tab jax
p = jnp.array([0.6, 0.3, 0.1])
q = jnp.array([0.2, 0.5, 0.3])

ce_pq = -jnp.sum(p * jnp.log(q)).item()
print(f'KL(P||Q) = {kl_divergence(p, q):.4f} nats, '
      f'KL(Q||P) = {kl_divergence(q, p):.4f} nats')
print(f'CE(P, Q) - H(P) = {ce_pq - entropy(p):.4f} nats')
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

in nats, exactly what every framework's classification loss computes.

```{.python .input #information-theory-formal-definition-1}
#@tab mxnet
def cross_entropy(y_hat, y):
    """Mean cross-entropy loss for predicted probabilities, in nats."""
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return float(ce.mean())
```

```{.python .input #information-theory-formal-definition-1}
#@tab pytorch
def cross_entropy(y_hat, y):
    """Mean cross-entropy loss for predicted probabilities, in nats."""
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean().item()
```

```{.python .input #information-theory-formal-definition-1}
#@tab tensorflow
def cross_entropy(y_hat, y):
    """Mean cross-entropy loss for predicted probabilities, in nats."""
    ce = -tf.math.log(tf.gather_nd(y_hat, indices=[[i, j] for i, j in zip(
        range(len(y_hat)), y)]))
    return float(tf.reduce_mean(ce).numpy())
```

```{.python .input #information-theory-formal-definition-1}
#@tab jax
def cross_entropy(y_hat, y):
    """Mean cross-entropy loss for predicted probabilities, in nats."""
    ce = -jnp.log(y_hat[jnp.arange(len(y_hat)), y])
    return ce.mean().item()
```

Two examples, three classes: the first example's true class is $0$, predicted
with probability $0.3$; the second's is $2$, predicted with probability $0.5$.
The loss is $\tfrac{1}{2}(-\ln 0.3 - \ln 0.5) \approx 0.949$ nats.

```{.python .input #information-theory-formal-definition-2}
#@tab mxnet
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input #information-theory-formal-definition-2}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input #information-theory-formal-definition-2}
#@tab tensorflow
labels = tf.constant([0, 2])
preds = tf.constant([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input #information-theory-formal-definition-2}
#@tab jax
labels = jnp.array([0, 2])
preds = jnp.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

The frameworks' built-in losses compute the same number. One subtlety is worth
spelling out, because the built-ins typically expect *logits* $\mathbf{z}$
rather than probabilities, applying softmax internally. Feeding
$\mathbf{z} = \log \mathbf{q}$ for an already-normalized $\mathbf{q}$ is
legitimate because softmax inverts the log exactly there:
$\mathrm{softmax}(\log \mathbf{q})_j = e^{\log q_j} / \sum_i e^{\log q_i}
= q_j / \sum_i q_i = q_j$. (More generally, softmax is invariant to adding a
constant to all logits, and $\log$ of a normalized vector is one valid logit
vector among many.) PyTorch's `NLLLoss` skips the subtlety by taking
log-probabilities directly; TensorFlow's loss accepts probabilities with
`from_logits=False`; `optax` wants logits, so we hand it $\log$-probabilities;
MXNet's metric consumes probabilities.

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab mxnet
nll_loss = CrossEntropy()  # operates on predicted probabilities directly
nll_loss.update([labels], [preds])
nll_loss.get()
```

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab pytorch
# `NLLLoss` consumes log-probabilities; `nn.CrossEntropyLoss` would instead
# take logits and apply log-softmax itself.
nll_loss = torch.nn.NLLLoss()
nll_loss(torch.log(preds), labels)
```

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab tensorflow
# With `from_logits=False` the loss consumes probabilities directly --
# no softmax round-trip needed.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss_fn(labels, preds).numpy()
```

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab jax
# optax expects logits; log-probabilities are valid logits because
# softmax(log q) = q for a normalized q.
optax.softmax_cross_entropy_with_integer_labels(
    jnp.log(preds), labels).mean()
```

Why is *this* the loss that training a probabilistic classifier by maximum
likelihood produces? We proved that already, in
:numref:`subsec_mdl-nll-crossentropy`: the average negative log-likelihood of
any i.i.d. dataset *is* the cross-entropy from the empirical distribution
$\hat p_{\textrm{data}}$ to the model, so maximizing likelihood, minimizing
cross-entropy, and minimizing
$D_{\textrm{KL}}(\hat p_{\textrm{data}} \| p_{\boldsymbol{\theta}})$ are one
and the same optimization---for binary labels, multiclass labels, and
densities alike. We will not re-derive it here; this section's contribution is
the *interpretation* of that loss as a code length, which we build next.

## The Coding View and Perplexity

We have repeatedly promised that entropy is a "floor" and KL the "extra"
cost. This subsection makes those words literal: they are statements about
compressing data with binary codes, and the proofs are short enough to give in
full. Binary codes speak base $2$, so bits briefly take the stage; divide by
$\ln 2$ (i.e., reinterpret every $\log$ below) and every statement holds in
nats verbatim.

### Prefix Codes and the Kraft Inequality

Suppose we must transmit a stream of symbols drawn i.i.d. from a distribution
$P$ over $k$ outcomes, encoding each outcome $x_i$ as a binary string
(*codeword*) of length $l_i$. So that the receiver can split the stream back
into codewords without separators, we require the code to be *prefix-free*: no
codeword is a prefix of another (like telephone numbers---once a dialed string
matches a number, no longer number starts the same way). Short codewords are
precious, and the next proposition says exactly how precious.

**Proposition (Kraft inequality).** *A prefix-free binary code with codeword
lengths $l_1, \ldots, l_k$ exists if and only if*

$$
\sum_{i=1}^k 2^{-l_i} \leq 1.
$$

**Proof.** Identify each codeword $c$ of length $l$ with the dyadic interval
$[0.c,\, 0.c + 2^{-l}) \subseteq [0, 1)$ of all real numbers whose binary
expansion starts with $c$ (e.g., the codeword $10$ owns
$[0.10_2, 0.11_2) = [\tfrac{1}{2}, \tfrac{3}{4})$, of length $2^{-2}$). One
codeword is a prefix of another exactly when the second's interval sits inside
the first's, so a prefix-free code corresponds to *disjoint* intervals, whose
total length $\sum_i 2^{-l_i}$ cannot exceed $1$. Conversely, given lengths
with $\sum_i 2^{-l_i} \leq 1$, sort them increasingly and lay down intervals
of length $2^{-l_i}$ left to right; each starts at a multiple of $2^{-l_i}$,
hence is a dyadic interval, and its binary address is a valid codeword.
$\blacksquare$

The Kraft inequality turns code design into a budget problem: each symbol
buys "address space" $2^{-l_i}$ out of a total budget of $1$, and shorter
codewords cost exponentially more. The optimal spend follows immediately.

**Proposition (entropy bounds the code length).** *Every prefix-free binary
code for $P$ has expected length $E[l] \geq H_2(P)$ (entropy in bits), and
the* Shannon code *with lengths $l_i = \lceil \log_2 (1/p_i) \rceil$ is
prefix-free and achieves $E[l] < H_2(P) + 1$.*

**Proof.** For the lower bound, let $Z = \sum_i 2^{-l_i} \leq 1$ (Kraft) and
define the distribution $q_i = 2^{-l_i}/Z$, the one *implied* by the code
lengths. Then $l_i = -\log_2 q_i - \log_2 Z$, so

$$
E[l] = \sum_i p_i l_i
= -\sum_i p_i \log_2 q_i - \log_2 Z
= H_2(P) + D_{\textrm{KL}}^{(2)}(P \| Q) - \log_2 Z
\geq H_2(P),
$$

since the KL term is non-negative by Gibbs and $-\log_2 Z \geq 0$. For the
upper bound, the Shannon lengths satisfy
$2^{-l_i} = 2^{-\lceil \log_2(1/p_i)\rceil} \leq p_i$, so they pass Kraft's
test ($\sum_i 2^{-l_i} \leq \sum_i p_i = 1$) and a prefix code with these
lengths exists; and $l_i < \log_2(1/p_i) + 1$ gives
$E[l] < H_2(P) + 1$. $\blacksquare$

So entropy is, within one bit, *the* price of communicating draws from
$P$---this is (the source-coding half of) Shannon's theorem, and the
one-bit slack can be driven to zero by coding long blocks of symbols at once
:cite:`Cover.Thomas.1999`. Moreover, look at where the KL divergence appeared
in the proof: if you build your code from the *wrong* distribution $Q$
(spending $-\log_2 q_i$ bits on symbol $i$, as the Shannon recipe would if it
believed $Q$), your expected cost on data that is really $P$ is

$$
-\sum_i p_i \log_2 q_i = \textrm{CE}_2(P, Q) = H_2(P) + D_{\textrm{KL}}^{(2)}(P\|Q).
$$

Cross-entropy is the price of coding with the wrong codebook, and the KL
divergence is *literally the extra bits per symbol* you waste. Every "extra
nats" claim earlier in the section is this statement, rescaled by $\ln 2$.

The bound is tight enough to see in a four-line example. Take
$P = (\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \tfrac{1}{8})$, whose
probabilities are exact powers of two, so the Shannon code wastes nothing; a
codebook built for the *uniform* distribution instead spends $2$ bits on every
symbol.

```{.python .input #information-theory-coding}
p = [1/2, 1/4, 1/8, 1/8]
lengths = [math.ceil(math.log2(1 / p_i)) for p_i in p]
kraft = sum(2 ** -l for l in lengths)
avg_len = sum(p_i * l for p_i, l in zip(p, lengths))
h_bits = -sum(p_i * math.log2(p_i) for p_i in p)

q = [1/4] * 4   # the wrong codebook: 2 bits for every symbol
avg_len_q = sum(p_i * 2 for p_i, l in zip(p, lengths))
kl_bits = sum(p_i * math.log2(p_i / q_i) for p_i, q_i in zip(p, q))

print(f'Shannon lengths {lengths}, Kraft sum = {kraft}')
print(f'E[l] = {avg_len} bits = H(P) = {h_bits} bits: no waste')
print(f'uniform codebook: {avg_len_q} bits = H + KL = {h_bits + kl_bits} bits')
```

A possible Shannon code here is $0,\, 10,\, 110,\, 111$---check that it is
prefix-free and that its lengths $1, 2, 3, 3$ are the ones computed above.

### Perplexity

Language models are evaluated by exactly the quantity this section has been
studying: the per-token cross-entropy of held-out text,
$\textrm{CE} = -\tfrac{1}{N}\sum_{i=1}^N \log q(x_i \mid x_{<i})$ nats, where
$q$ is the model's predicted next-token distribution. The community
exponentiates it and calls the result *perplexity*:

$$
\textrm{PPL} = \exp(\textrm{CE})
= \exp\!\Big(-\frac{1}{N}\sum_{i=1}^N \log q(x_i \mid x_{<i})\Big)
= \Big(\prod_{i=1}^N q(x_i \mid x_{<i})\Big)^{-1/N},
$$

the inverse *geometric mean* of the probabilities the model assigned to what
actually came next. Exponentiating buys an interpretation: a model that is
uniformly undecided among $V$ tokens at every step has
$\textrm{CE} = \ln V$ and hence $\textrm{PPL} = V$, so a perplexity of,
say, $20$ means the model is, on average, as uncertain as if it were choosing
uniformly among $20$ tokens---an *effective branching factor*. A perfect model
has perplexity $1$, and Gibbs' inequality puts the floor at
$\textrm{PPL} \geq e^{H}$, where $H$ is the per-token entropy of the language
itself. Because $\textrm{PPL} = b^{\textrm{CE}_b}$ for *any* log base $b$,
perplexity is also the rare information-theoretic quantity with no unit
ambiguity: nats and bits give the same number (Exercise 6).

A five-token toy stream makes the bookkeeping concrete.

```{.python .input #information-theory-perplexity}
q_tok = [0.2, 0.1, 0.4, 0.25, 0.05]   # model prob. of each observed token
nll = [-math.log(q_i) for q_i in q_tok]
mean_nll = sum(nll) / len(nll)

print(f'per-token NLL (nats): {[round(v, 3) for v in nll]}')
print(f'mean NLL = {mean_nll:.4f} nats, perplexity = {math.exp(mean_nll):.2f}')
```

A mean of $\approx 1.84$ nats per token corresponds to a perplexity of
$\approx 6.3$: this model is as confused as a uniform choice among six or so
tokens. Note how the geometric mean punishes confident mistakes---the single
$0.05$ contributes $3.0$ nats, as much as several good predictions
combined. When you encounter perplexity as the headline metric in
:numref:`sec_language-model` and the Transformer chapters, it is this number.

## Modern Uses

The identities of this section are not museum pieces; they are working parts
of modern training pipelines. We close with two of them---and the punchline
that both are corollaries of Gibbs' inequality.

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

i.e., target probability $1 - \epsilon + \epsilon/k$ for the true class and
$\epsilon/k$ for each of the other $k - 1$. The training loss is the ordinary
cross-entropy $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ to this softened
target. What prediction does the smoothed loss actually ask for?

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
$\epsilon = 0.1$ that is $\ln 91 \approx 4.51$ nats of logit gap---a
concrete, attainable target instead of a runaway one. The loss at the optimum
is the (nonzero) entropy $H(\mathbf{p}^\epsilon)$, which is why a
label-smoothed loss curve plateaus above zero even when the model is doing
everything right. Both numbers are easy to check:

```{.python .input #information-theory-label-smoothing}
k, eps = 10, 0.1
p_eps = [1 - eps + eps / k] + [eps / k] * (k - 1)

gap = math.log(p_eps[0] / p_eps[1])              # optimal logit gap
h_p_eps = -sum(t * math.log(t) for t in p_eps)   # CE at the optimum q* = p_eps

z = [10.0] + [0.0] * (k - 1)                     # an overconfident prediction
z_norm = sum(math.exp(v) for v in z)
q_conf = [math.exp(v) / z_norm for v in z]
ce_conf = -sum(t * math.log(s) for t, s in zip(p_eps, q_conf))

print(f'optimal logit gap = ln(91) = {gap:.4f} nats')
print(f'CE at q* = H(p_eps) = {h_p_eps:.4f} nats')
print(f'CE at the overconfident q = {ce_conf:.4f} nats')
```

The overconfident prediction (logit gap $10$) is *worse* under the smoothed
target than the calibrated one---label smoothing literally penalizes the
behavior that one-hot targets demand.

### Knowledge Distillation

*Knowledge distillation* :cite:`Hinton.Vinyals.Dean.2015` trains a small
*student* network to imitate a large *teacher*, transferring not just the
teacher's argmax but its full distribution over wrong answers (a "2" that
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
against any hard-label loss it is mixed with. Let's verify both claims
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
temperature, and while the raw gradient decays like $1/T^2$, the $T^2$-scaled
gradient settles toward the constant limit
$\tfrac{1}{k}\big((\mathbf{z}^{\textrm{stu}} - \bar z^{\textrm{stu}}) -
(\mathbf{z}^{\textrm{tea}} - \bar z^{\textrm{tea}})\big) = (-0.667, 0.333,
0.333)$ here---the scale-matching the $T^2$ factor was designed for. As
$T \to 1$ the loss reduces to the ordinary KL (and, with a one-hot teacher,
to the ordinary cross-entropy loss), recovering standard training as a special
case.

### One Principle, Many Losses

Stepping back: maximum likelihood *is* cross-entropy minimization *is*
KL-projection of the empirical distribution onto the model family
(:numref:`subsec_mdl-nll-crossentropy`)---and this section has now equipped
that one principle with its operational meaning (code length and waste), its
evaluation metric (perplexity), and two of its modern refinements (smoothed
targets and distilled teachers). The story continues in two directions: KL is
just one member of a whole family of divergences, each inducing a different
generative-modeling objective (:numref:`sec_mdl-divergences-distances`), and
applying KL to a joint distribution versus the product of its marginals yields
mutual information, the engine of contrastive representation learning
(:numref:`sec_mdl-mutual-information`).

## Summary

* Self-information $I(x) = -\log p(x)$ measures the surprise of an outcome;
  entropy $H(P) = -E_{x\sim P}[\log p(x)]$ is the average surprise of a
  distribution. We measure both in nats (natural log); bits differ by a
  factor of $\ln 2$.
* On $k$ outcomes, $0 \leq H(P) \leq \log k$, with the maximum exactly at the
  uniform distribution (Jensen). Differential entropy of continuous variables
  is *not* coordinate-invariant and can be negative---one reason deep learning
  prefers relative quantities.
* The KL divergence $D_{\textrm{KL}}(P\|Q) = E_{x\sim P}[\log p(x)/q(x)]$ is
  asymmetric and, by Gibbs' inequality, non-negative with equality iff
  $P = Q$.
* Cross-entropy decomposes as
  $\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P\|Q)$: an irreducible floor
  plus removable waste. Minimizing it in $Q$ is minimizing KL, and on
  empirical data it is maximum likelihood
  (:numref:`subsec_mdl-nll-crossentropy`).
* The coding view makes this literal: entropy is the optimal expected code
  length (Kraft inequality + Shannon code), cross-entropy is the cost of
  coding with the wrong distribution, and KL is the extra bits wasted.
* Perplexity $\textrm{PPL} = \exp(\textrm{CE})$ is the exponentiated
  per-token cross-entropy of a language model: an effective branching factor,
  independent of the log base.
* Label smoothing and knowledge distillation are corollaries of the same
  identities: the smoothed target makes the cross-entropy optimum a finite,
  calibrated prediction, and the $T^2$ factor on the distillation KL keeps
  gradients scale-matched across temperatures.

## Exercises

1. Verify the card-deck numbers from the first subsection: the four reports
   carry $0$, $\ln 4$, $\ln 52$, and $\ln 52!$ nats. Use Stirling's
   approximation to estimate $\ln 52!$ and check it against the exact value.
1. Give a second proof that the uniform distribution maximizes entropy on $k$
   outcomes, this time from Gibbs' inequality: show that
   $D_{\textrm{KL}}(P \,\|\, U) = \log k - H(P)$ for the uniform $U$, and
   conclude.
1. Let's compute the entropy of a few data sources:
    * Suppose that you are watching the output generated by a monkey at a
      typewriter that hits any of the $44$ keys uniformly at random. How many
      nats (and bits) of randomness per character do you observe?
    * Unhappy with the monkey, you replace it with a drunk typesetter that
      picks a random word out of a vocabulary of $2{,}000$ words, with an
      average word length of $4.5$ letters. How many nats per character now?
    * Still unhappy, you replace the typesetter with a language model that
      achieves a per-word perplexity of $15$. How many nats per character
      does that correspond to?
1. Derive the closed form :eqref:`eq_mdl-gaussian_kl` for the KL divergence
   between two univariate Gaussians by writing
   $E_{x\sim P}[\log p(x) - \log q(x)]$ and using
   $E_{x \sim P}[x] = \mu_1$, $E_{x\sim P}[(x-\mu_1)^2] = \sigma_1^2$.
1. Show that self-information is essentially unique: if a continuous,
   decreasing function $I(p)$ on $(0, 1]$ satisfies $I(1) = 0$ and
   $I(p_1 p_2) = I(p_1) + I(p_2)$ for all $p_1, p_2$, then
   $I(p) = -c \log p$ for some constant $c > 0$.
1. Show that perplexity does not depend on the log base: if
   $\textrm{CE}_b$ denotes the per-token cross-entropy in base-$b$ units,
   then $b^{\textrm{CE}_b}$ is the same for every $b$. Then compute the
   perplexity of a model whose per-token cross-entropy is $\ln 5$ nats.
1. For label smoothing with $k$ classes and smoothing parameter $\epsilon$,
   derive the optimal prediction $\mathbf{q}^*$ and the optimal logit gap,
   and evaluate the gap for $k = 1000$, $\epsilon = 0.1$. What happens to the
   gap as $\epsilon \to 0$, and why does that recover the one-hot pathology?
1. For the distillation loss, derive the gradient formula
   $\partial D_{\textrm{KL}}(\mathbf{p}^{(T)} \| \mathbf{q}^{(T)}) / \partial
   z^{\textrm{stu}}_j = (q_j^{(T)} - p_j^{(T)})/T$, and show that as
   $T \to \infty$ the $T^2$-scaled gradient tends to
   $\tfrac{1}{k}\big((z^{\textrm{stu}}_j - \bar z^{\textrm{stu}}) -
   (z^{\textrm{tea}}_j - \bar z^{\textrm{tea}})\big)$. What does the loss
   reduce to at $T = 1$ with a one-hot teacher?

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

::: {.slide title="Information theory: the language of losses"}
The number your training loop prints is a quantity from information theory
(Shannon, 1948) — a *code length*, in nats:

- **Self-information** $I(x) = -\log p(x)$ — the surprise of observing $x$.
- **Entropy** $H(P) = -\mathbb{E}_P[\log p]$ — average surprise; the
  irreducible floor.
- **Cross-entropy** $\mathrm{CE}(P,Q) = -\mathbb{E}_P[\log q]$ — what your
  model actually pays.
- **KL divergence** $D_{\mathrm{KL}}(P\|Q) = \mathrm{CE}(P,Q) - H(P)$ — the
  waste you can train away.

Units: nats (natural log) throughout; bits are a $\ln 2$ rescaling.
:::

::: {.slide title="Surprise and entropy"}
Rare = surprising: $I(x) = -\log p(x)$ is zero for certain events and
diverges as $p \to 0$:

@fig:mdl-it-self-info-curve

. . .

Entropy averages the surprise, $H(P) = -\sum_x p(x)\log p(x)$:

@information-theory-definition
:::

::: {.slide title="Entropy peaks at maximal uncertainty"}
On $k$ outcomes, $0 \le H \le \log k$, with the maximum exactly at the
uniform distribution (Jensen on $\log$). For the coin, the peak is the fair
coin: $\ln 2 \approx 0.693$ nats.

@fig:mdl-it-bernoulli-entropy
:::

::: {.slide title="KL divergence and Gibbs' inequality"}
$D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_{x\sim P}[\log p(x)/q(x)] \ge 0$, with
equality iff $P = Q$ (Gibbs, via Jensen on $-\log$). Asymmetric: not a
metric.

@information-theory-definition-2

. . .

$\mathrm{CE} = H + \mathrm{KL}$, verified on two categoricals:

@information-theory-kl-categorical
:::

::: {.slide title="Gaussian KL in closed form"}
$D_{\mathrm{KL}}(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2))
= \log\frac{\sigma_2}{\sigma_1} +
\frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$ — matches a
Monte-Carlo estimate, and exposes the asymmetry:

@information-theory-example-2

. . .

@information-theory-example-3
:::

::: {.slide title="Cross-entropy is the classification loss"}
One-hot truth $\Rightarrow$ CE collapses to $-\log \hat y_{\text{true}}$,
the model's surprise at the correct class — same number as every framework
built-in (maximum likelihood, by
the NLL $=$ CE equivalence):

@information-theory-formal-definition-2
:::

::: {.slide title="The coding view and perplexity"}
Kraft: prefix codes satisfy $\sum_i 2^{-l_i} \le 1$; Shannon's code achieves
$H_2 \le \mathbb{E}[l] < H_2 + 1$. Coding with the wrong $Q$ costs
$\mathrm{CE}$ — KL is *literally the extra bits*:

@information-theory-coding

. . .

Language models: $\mathrm{PPL} = \exp(\text{mean NLL})$ — an effective
branching factor:

@information-theory-perplexity
:::

::: {.slide title="Label smoothing and distillation"}
Smoothed target $(1-\epsilon)\,\mathbf{e}_y + \epsilon\,\mathbf{1}/k$ ⟹ the
CE optimum is a *finite* logit gap (Gibbs):

@information-theory-label-smoothing

. . .

Distillation: $T^2\, D_{\mathrm{KL}}(\text{teacher}_T \|\, \text{student}_T)$;
the $T^2$ keeps gradients scale-matched:

@information-theory-distillation
:::

::: {.slide title="Recap"}
- Entropy = floor, cross-entropy = payment, KL = waste — literally, in code
  lengths (Kraft + Shannon).
- Gibbs' inequality ($\mathrm{KL} \ge 0$) powers everything: CE $\ge$ H, the
  coding bound, the label-smoothing optimum.
- Minimizing CE = minimizing KL to the data = maximum likelihood.
- Perplexity = $\exp(\mathrm{CE})$: base-free, an effective branching factor.
- Next: families of divergences (f-divergences, optimal transport) and
  mutual information for representation learning.
:::
