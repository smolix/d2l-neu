# Entropy, Cross-Entropy, and KL Divergence
:label:`sec_mdl-information_theory`

Almost every classifier in this book was trained by pushing a cross-entropy
loss down, and we have grown used to watching that number fall. We have not yet
asked what it measures, and without knowing that, a loss curve cannot be
read. Its starting value already means something. A freshly initialized
classifier with ten classes reports a loss of about $2.30$, and
:numref:`sec_mlp-implementation` used this value to check that an untrained
network is set up correctly. The value is $\ln 10$, the cost in nats of
naming one of ten labels when the model still finds all ten equally likely.
The end of the curve raises harder questions. Is a final loss of $0.6$ good?
Is a plateau a bug or a property of the data? If some of the labels are
wrong, how low can any model go?

The loss is the average of $-\log q(y)$, the model's surprise at the label
$y$ that occurred. That average splits into two parts. The first is the
average surprise of someone who knows the true distribution of the labels. It
is called the *entropy*. The data fix it, and no model can reduce it. On a
finite set of outcomes, entropy also has a concrete meaning that we use
throughout: the average number of yes/no questions needed to identify the
outcome.

The second part is the extra surprise of using the wrong distribution. It is
called the *Kullback--Leibler (KL) divergence*. It is never negative, and it
is zero only for the right model. :numref:`subsec_mdl-nll-crossentropy`
showed that the average negative log-likelihood of a model on a data set is a
cross-entropy, taken from the empirical distribution of the data to the
model. Here we take that quantity apart. Two questions remain open at the
end. We will call the loss a code length without building the code, and the
KL divergence becomes infinite as soon as a model gives zero probability to
an outcome that occurs.

The logarithm base is a choice of unit: base $2$ gives *bits*, base $e$ gives
*nats*. **We work in nats throughout**, matching deep-learning practice: the
library `log` is natural, so reported training losses are already in nats.
Base $2$ suits yes/no questions, and we say so when we use it,
writing $H_2$ for an entropy in bits. Since $\log_2 x = \ln x / \ln 2$,
converting is a fixed rescaling: $1\textrm{ bit} = \ln 2 \approx 0.693$ nats
and $1\textrm{ nat} = 1/\ln 2 \approx 1.443$ bits. Changing the base leaves
every argmin unchanged. It multiplies the loss and its gradients by a
constant, which acts like a change of learning rate for SGD (Adam is nearly
invariant to it).

The chapter uses the symbols of :numref:`tab_mdl-it-notation`; each later
section restates the few it needs.

:Symbols of the information theory chapter. The binary entropy $H_b(p)$ is the entropy of a coin with heads probability $p$, a function of a number rather than of a random variable. The last six rows belong to later sections, which define them where they are first used.
:label:`tab_mdl-it-notation`

| Symbol | Meaning | Unit |
|:--|:--|:--|
| $I(x) = -\log p(x)$ | self-information of the outcome $x$ (by convention the letter is shared with the mutual information $I(X;Y)$) | nats |
| $H(X)$, $H(P)$ | entropy of a discrete random variable or distribution; $H_2$ in bits | nats |
| $H_b(p)$ | binary entropy of a coin with heads probability $p$ | nats |
| $h(X)$ | differential entropy of a continuous random variable | nats, depends on the units of $X$ |
| $D_{\textrm{KL}}(P \Vert Q)$ | KL divergence from $P$ to $Q$ | nats |
| $\textrm{CE}(P, Q)$ | cross-entropy from $P$ to $Q$ | nats |
| $k$ | number of classes (size of the set of outcomes) | |
| $n$ | number of examples | |
| $\mathbf{p}$, $\mathbf{q}$ | probability vectors on $k$ classes (truth, model) | |
| $\mathbf{z}$ | logits, with $\mathbf{q} = \mathrm{softmax}(\mathbf{z})$ | |
| $\epsilon$ | label-smoothing mass (in later sections, a noise variable) | |
| $\tau$ | temperature | |
| $H(X \mid Y)$, $I(X;Y)$ | conditional entropy, mutual information | nats |
| $D_f(P \Vert Q)$ | f-divergence with convex generator $f$ | |
| $T$ | critic: a function whose averages under two distributions bound a divergence | |
| $g$ | test function of an integral probability metric | |
| $\gamma$ | coupling (transport plan) of two distributions | |
| $N$ | number of candidates in a contrastive batch | |

We use the following imports throughout the section.

```{.python .input #information-theory-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, npx
from mxnet import np as mnp
npx.set_np()
```

```{.python .input #information-theory-imports}
#@tab pytorch
import math
import numpy as np
import torch
```

```{.python .input #information-theory-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
```

```{.python .input #information-theory-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
import optax
```

## Information and Entropy

### Self-Information

Consider reports about a shuffled deck of cards. "I see a card" conveys
nothing, because the event is certain. "I see a heart" picks one of four
equally likely suits, "It is the three of spades" one of $52$ equally likely
cards, and the order of the whole deck one of $52!$ orderings. The less
probable an event, the more we learn by observing it. Shannon
:cite:`Shannon.1948` measured this *surprise* by the *self-information* of an
outcome $x$ with probability $p(x)$,

$$
I(x) = -\log p(x).
$$

Thus a certain event carries no information, and halving the probability of
an outcome adds $\ln 2$ nats, one bit, to its surprise. The four card reports
carry $0$, $\ln 4 \approx 1.39$, $\ln 52 \approx 3.95$, and
$\ln 52! \approx 156.4$ nats. In bits, where one bit is the information of
one fair coin flip, they carry $0$, $2$, $5.70$, and $225.6$.

Why a logarithm? We want three properties: a certain event carries zero
information, rarer events carry more, and the information of two independent
events adds up. Since independent probabilities multiply, the third property
says $I(p_1 p_2) = I(p_1) + I(p_2)$, and the only decreasing functions with
this property are multiples of $-\log p$ (:citet:`Csiszar.2008`;
Exercise :exref:`ex_mdl-information-theory-show-that-self-information` gives
the argument). The choice of multiple is the choice of unit. Surprise depends
on the model that assigns the probabilities, so a confidently wrong model
finds ordinary events surprising.

![Self-information $I(x) = -\log p(x)$ in nats as a function of the probability $p$. Certain events ($p = 1$) carry zero information, the fair coin ($p = \tfrac{1}{2}$) carries $\ln 2 \approx 0.693$ nats, and the curve diverges as $p \to 0$.](../img/mdl-it-self-info-curve.svg)
:label:`fig_mdl-self-info-curve`

:numref:`fig_mdl-self-info-curve` plots the curve: decreasing, convex, zero at
$p = 1$, and diverging as $p \to 0$. The first three card reports can be
checked by hand. The fourth needs $\ln 52!$, which the next cell computes
through the log-gamma function, $\ln 52! = \ln \Gamma(53)$, without ever
forming the 68-digit number $52!$.

```{.python .input #information-theory-self-information}
def self_information(p):
    """Self-information -log p = log(1/p), in nats."""
    return math.log(1 / p)

reports = {'a card': 1.0, 'a heart': 1 / 4, 'the three of spades': 1 / 52}
for name, p in reports.items():
    print(f'{name:22s} {self_information(p):7.3f} nats')
deck_order = math.lgamma(53)          # ln 52! without forming 52!
print(f'{"the order of the deck":22s} {deck_order:7.2f} nats '
      f'= {deck_order / math.log(2):.1f} bits')
```

The output reproduces the four numbers of the text. Naming the order of a
shuffled deck carries as much information as $226$ fair coin flips.

### Entropy

How surprised should we expect to be before the outcome arrives? We average
the surprise over the outcomes. Let $X$ be a discrete random variable that
takes the value $x_i$ with probability $p_i$. Its *entropy* is the expected
self-information of its outcome,

$$
H(X) = -\mathbb{E}_{x \sim P}[\log p(x)] = -\sum_i p_i \log p_i,
$$
:eqlabel:`eq_mdl-ent_def`

with the convention $0 \log 0 = 0$, justified by continuity ($t \log t \to 0$
as $t \to 0$): an impossible outcome contributes nothing. Nothing in this
definition is new; it is the self-information averaged over the outcomes.
Since the entropy depends only on the distribution $P$ of $X$, we write
$H(P)$ as well as $H(X)$.

On a finite set, entropy in bits has a concrete meaning: to within one
question, it is the average number of yes/no questions that a good
questioner needs to identify the outcome. Take the distribution $P = (\tfrac12, \tfrac14, \tfrac18,
\tfrac18)$ on four symbols $a, b, c, d$. We carry it through the section and
call it the *dyadic source*, since all its probabilities are powers of
$\tfrac12$. A good questioner first asks "Is it $a$?", which settles the
matter half the time. If not, they ask "Is it $b$?", and then "Is it $c$?",
where "no" means $d$. The expected number of questions is

$$
\tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac18 \cdot 3 + \tfrac18 \cdot 3
= 1.75,
$$

and the entropy in bits is the same number,
$H_2(P) = \tfrac12 \log_2 2 + \tfrac14 \log_2 4 + 2 \cdot \tfrac18 \log_2 8
= 1.75$ bits, or $1.213$ nats. Each outcome is identified with
$\log_2(1/p_i)$ questions, its self-information in bits.

A questioner who ignores the probabilities asks "Is it $a$ or $b$?" and then
"Which of the two?", and always needs $2$ questions. For every distribution
on a finite set, the best questioning strategy needs between $H_2$ and
$H_2 + 1$ questions on average, as :numref:`sec_mdl-coding-compression`
proves.

The entropy in code masks the zero probabilities, which implements
$0 \log 0 = 0$. We check it on the dyadic source, in nats and in bits.

```{.python .input #information-theory-definition}
def entropy(p):
    """Entropy of a probability vector, in nats (0 log 0 = 0 by masking)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.dot(p, np.log(p)))

p_dyadic = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
h = entropy(p_dyadic)
print(f'H = {h:.3f} nats = {h / math.log(2):.2f} bits')
```

The entropy is $1.75$ bits, the number of questions we counted by hand.

Every term $-p_i \log p_i$ is non-negative because $0 \le p_i \le 1$, so
$H(X) \ge 0$, with equality only for a point mass, a variable whose value is
certain. The entropy also depends only on the list of probabilities, not on
the values or names of the outcomes. A variable that takes the values $1$,
$100$, $10^6$, and $10^9$ with the dyadic probabilities also has $1.75$
bits. How large can the entropy of a variable with $k$ outcomes be?

### Maximum Entropy on a Finite Set

We start with the smallest case, a coin. A Bernoulli variable with heads
probability $p$ has the *binary entropy*

$$
H_b(p) = -p \log p - (1-p) \log(1-p),
$$

a function of the number $p$ rather than of a random variable.
:numref:`fig_mdl-bernoulli-entropy` plots it. The entropy is zero at
$p \in \{0, 1\}$, where the outcome is certain, and largest at the fair coin,
$H_b(\tfrac12) = \ln 2 \approx 0.693$ nats, one bit. A coin with $p = 0.1$
has $H_b(0.1) \approx 0.325$ nats, because its outcome is easy to guess.

![The binary entropy $H_b(p)$ in nats, symmetric about $p = \tfrac{1}{2}$, where it peaks at $\ln 2 \approx 0.693$ nats. At the deterministic endpoints $p = 0$ and $p = 1$ the outcome carries no surprise and the entropy vanishes. The curve is concave: any chord lies below it.](../img/mdl-it-bernoulli-entropy.svg)
:label:`fig_mdl-bernoulli-entropy`

The curve is concave: the chord joining any two of its points lies below it.
This chord property is the content of *Jensen's inequality*, which we use twice in
this section. For a concave function $f$ and a random variable $X$ with finite
mean,

$$
\mathbb{E}[f(X)] \le f(\mathbb{E}[X]).
$$

Averaging first and applying $f$ second gives the larger value. For an $X$
with two values, this is the statement that the chord lies below the curve.
For strictly concave $f$, equality holds only when $X$ is constant with
probability one. :numref:`subsec_mdl-jensen` proves the general case. With
Jensen's inequality, the maximum of the coin at $p = \tfrac12$ extends to any
number of outcomes.

**Proposition (uniform maximum).** *If $X$ takes at most $k$ values, then*

$$
H(X) \leq \log k,
$$

*with equality if and only if $X$ is uniform on $k$ values.*

**Proof.** Write the entropy as the expectation of $\log(1/p(X))$, a concave
function of $1/p(X)$. Jensen's inequality for the concave $\log$ gives

$$
H(X) = \mathbb{E}\!\left[\log \frac{1}{p(X)}\right]
\leq \log \mathbb{E}\!\left[\frac{1}{p(X)}\right]
= \log \sum_{i\,:\,p_i > 0} p_i \cdot \frac{1}{p_i}
= \log m
\leq \log k,
$$

where $m \leq k$ is the number of outcomes with $p_i > 0$. The step that
matters is the sum: each outcome in the support contributes
$p_i \cdot (1/p_i) = 1$, so the mean of $1/p(X)$ is finite and counts the
support. Equality requires both inequalities to be tight. Because $\log$ is
strictly concave, the first is an equality only when $1/p(X)$ is constant,
that is, when $X$ is uniform on its $m$ support points; and $\log m = \log k$
forces $m = k$. Together, $p_i = 1/k$ for all $i$. $\blacksquare$

The uniform distribution hedges the most, so it is the hardest to guess. For
four outcomes the bound is $\log_2 4 = 2$ bits, $0.25$ bits above the
$1.75$ bits of the dyadic source. A continuous variable has no such bound
without a constraint: the uniform density on $[0, a]$ has differential
entropy $\log a$, as the next subsection shows, and this grows without limit
with $a$. With the variance fixed, the Gaussian has the largest entropy; we
prove this once cross-entropy is available.

### Differential Entropy
:label:`subsec_mdl-differential-entropy`

For a continuous random variable $X$ with density $p$, the sum in the entropy
becomes an integral. The result is called the *differential entropy*,

$$
h(X) = -\int p(x) \log p(x) \,dx.
$$
:eqlabel:`eq_mdl-diff-ent-def`

It is the average surprise of a density value, and density values are not
probabilities: they carry the units of $1/x$, and they can exceed one. We
write a lowercase $h$ to keep it apart from the discrete $H$, because the two
behave differently.

The smallest case shows how. The uniform density on $[0, a]$ equals $1/a$
there, so $h = -\int_0^a \tfrac1a \log \tfrac1a \,dx = \log a$. For
$a = \tfrac12$ this is $-\ln 2 \approx -0.693$ nats, a negative entropy, which
no discrete distribution can have. Measuring the same variable in centimetres
instead of metres multiplies $a$ by $100$ and adds $\ln 100 \approx 4.61$
nats, although the uncertainty has not changed.

A quantity built from the ratio of two densities behaves better. For two
uniform densities $p_1$ on $[0, a_1]$ and $p_2$ on $[0, a_2]$ with
$a_1 \le a_2$, the average log-ratio is
$\int p_1 \log(p_1/p_2)\,dx = \log(a_2/a_1)$. For $a_1 = \tfrac12$ and
$a_2 = 1$ it is $\ln 2 \approx 0.693$ nats in metres and in centimetres
alike, because the change of unit divides both densities by the same
factor. This average log-ratio is the KL divergence, defined below.

A shift of this kind occurs under any smooth change of variables. If $y = g(x)$ is
invertible with Jacobian $\partial g/\partial x$, the density of $Y$ at $g(x)$
is $p_X(x)/|\det \partial g/\partial x|$, and so

$$
h(Y) = h(X) + \mathbb{E}\big[\log\lvert\det \partial g/\partial x\rvert\big].
$$

Stretching a variable raises its differential entropy, and squeezing it lowers
the entropy without limit. A Gaussian with standard deviation $\sigma$ has
$h = \tfrac12 \log(2\pi e \sigma^2)$, as we derive below, which is negative
once $\sigma < (2\pi e)^{-1/2} \approx 0.24$.

Differential entropy is not the limit of discrete entropy. Divide the line
into cells of width $\Delta$, and let $X_\Delta$ record which cell contains
$X$. Then $P(X_\Delta = i) \approx p(x_i) \Delta$ for a point $x_i$ in cell
$i$, and

$$
H(X_\Delta) + \log \Delta \to h(X) \quad \textrm{as } \Delta \to 0
$$

for densities regular enough that the sums converge to the integrals (for
example, when $p$ is continuous and $p$ and $p \log p$ are
Riemann-integrable). The discrete
entropy itself grows like $\log(1/\Delta)$, since a finer grid takes more
questions, and $h$ is what remains after that growth is subtracted. For the
uniform density on $[0, 1]$ with $\Delta = \tfrac18$, the eight cells are
equally likely, $H_2(X_\Delta) = 3$ bits and $\log_2 \Delta = -3$, which sum
to $0 = h$. Exercise :exref:`ex_mdl-information-theory-quantized-entropy`
proves the limit and checks it on a Gaussian. Differential entropy measures
surprise relative to a unit of length, so it moves with the unit. A quantity
built from a ratio of two densities does not.

## KL Divergence and Cross-Entropy

### The Kullback--Leibler Divergence

We now want to measure how badly a model $Q$ misrepresents a truth $P$.
Information theory answers with the *Kullback--Leibler (KL) divergence*, also
called *relative entropy* :cite:`Kullback.Leibler.1951`: the average extra
surprise of someone who believes $Q$ when the outcomes are drawn from $P$. In
the questions game, the KL divergence in bits is the average number of extra
questions that a strategy designed for $Q$ needs on outcomes drawn from $P$,
up to rounding to whole questions. Let $P$ and $Q$ have mass functions (or
densities) $p$ and $q$. The KL divergence from $P$ to $Q$ is

$$
D_{\textrm{KL}}(P\|Q) = \mathbb{E}_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].
$$
:eqlabel:`eq_mdl-kl_def`

The term inside the expectation is
$\log p(x) - \log q(x) = (-\log q(x)) - (-\log p(x))$: the surprise of the
believer in $Q$ minus the surprise of the believer in $P$. It is positive
where $Q$ underestimates the outcome ($q(x) < p(x)$) and negative where $Q$
overestimates it. KL averages this relative surprise over outcomes drawn from
the truth.

The definition is asymmetric. $P$ supplies both the samples and the
numerator, so in general $D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P)$.
:numref:`sec_mdl-fwd-vs-rev-kl` shows what the choice of direction does to a
fitted model. Outcomes with $p(x) = 0$ contribute nothing, by the convention
$0 \log(0/q) = 0$. If some outcome has $p(x) > 0$ but $q(x) = 0$, however,
then $D_{\textrm{KL}}(P\|Q) = +\infty$.

A small pair makes the definition concrete, and we carry it through the
section. Let $P = (0.6, 0.3, 0.1)$ be the truth and $Q = (0.2, 0.5, 0.3)$ a
model on three outcomes. Then

$$
D_{\textrm{KL}}(P\|Q)
= 0.6 \ln\frac{0.6}{0.2} + 0.3 \ln\frac{0.3}{0.5} + 0.1 \ln\frac{0.1}{0.3}
= 0.659 - 0.153 - 0.110 = 0.396 \textrm{ nats},
$$

and the reverse direction gives $D_{\textrm{KL}}(Q\|P) = 0.365$ nats. In bits
the two directions are $0.571$ and $0.527$. They differ, and the first says
that a code built for $Q$ wastes $0.57$ bits per symbol on long sequences
drawn from $P$.

Here is the discrete case in code, with $0 \log 0 = 0$ implemented by masking
the zero probabilities of $P$, as in `entropy`. We call it on the running
pair in both directions, and on a model $Q' = (0.5, 0.5, 0)$ that declares
the third outcome impossible.

```{.python .input #information-theory-definition-2}
def kl_divergence(p, q):
    """KL(P || Q) for two probability vectors, in nats; inf where q = 0 < p."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    if np.any(q[m] == 0):
        return float('inf')
    return float(np.dot(p[m], np.log(p[m] / q[m])))

p = np.array([0.6, 0.3, 0.1])
q = np.array([0.2, 0.5, 0.3])
print(f'KL(P||Q) = {kl_divergence(p, q):.4f} nats, '
      f'KL(Q||P) = {kl_divergence(q, p):.4f} nats')
print(f"KL(P||Q') = {kl_divergence(p, [0.5, 0.5, 0.0])}")
```

The two directions give $0.3961$ and $0.3653$ nats. The model $Q'$ gives zero
probability to an outcome that occurs one time in ten, and its divergence is
infinite: a model that declares an event impossible is infinitely surprised
when it happens.

### Gibbs' Inequality

Can the extra surprise be negative? That would mean a wrong model is less
surprised, on average, than the truth itself. Gibbs' inequality
:cite:`Gibbs.1902` says that it cannot: no model beats the truth on average.

**Proposition (Gibbs' inequality).** *For any distributions $P$ and $Q$ on
the same space,*

$$
D_{\textrm{KL}}(P\|Q) \geq 0,
$$

*with equality if and only if $P = Q$.*

**Proof.** If $q(x) = 0$ for some $x$ with $p(x) > 0$, the divergence is
$+\infty$ and there is nothing to prove, so assume otherwise. Jensen's
inequality for the convex function $-\log$ reads
$\mathbb{E}[-\log Z] \ge -\log \mathbb{E}[Z]$, and with
$Z = q(x)/p(x)$ it gives

$$
\begin{aligned}
D_{\textrm{KL}}(P\|Q)
&= \mathbb{E}_{x\sim P}\!\left[-\log\frac{q(x)}{p(x)}\right]
\geq -\log \mathbb{E}_{x\sim P}\!\left[\frac{q(x)}{p(x)}\right] \\
&= -\log \!\!\sum_{x\,:\,p(x)>0}\!\! q(x)
= -\log Q(\textrm{supp}\,P)
\geq 0.
\end{aligned}
$$

The step that matters is the second line. The expectation runs over the
support of $P$, the factors $p(x)$ cancel, and what remains is the $Q$-mass of
that support, a number at most $1$, whose negative logarithm is non-negative.
For densities, the same computation runs with integrals. For equality, both
inequalities must be tight. Because $-\log$ is strictly convex, the first is
an equality only when $q(x)/p(x)$ equals a constant $c$ for $P$-almost every
$x$, and the second requires $Q(\textrm{supp}\,P) = 1$. Summing (or
integrating) $q = c\,p$ over the support of $P$ then gives $c = 1$, so $q = p$,
almost everywhere in the case of densities. $\blacksquare$

For the concave logarithm, the log of an average is at least the average of
the logs, with equality only when there is nothing to average. A model
incurs no extra surprise only when it is right. Exercise
:exref:`ex_mdl-information-theory-gibbs-tangent` gives a second proof from the
inequality $\ln u \le u - 1$.

### Cross-Entropy

Classifiers are trained with a close relative of the KL divergence. The
*cross-entropy* from $P$ to $Q$ is the average surprise of the
believer in $Q$ on outcomes drawn from $P$,

$$
\textrm{CE}(P, Q) = -\mathbb{E}_{x \sim P} [\log q(x)].
$$
:eqlabel:`eq_mdl-ce_def`

Adding and subtracting $\log p(x)$ inside the expectation, which is
legitimate whenever $H(P)$ is finite, splits the model's surprise into the
truth's own surprise plus the relative surprise:

$$
\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P\|Q).
$$
:eqlabel:`eq_mdl-ce_decomp`

The cross-entropy is an entropy floor plus a never-negative excess. Gibbs'
inequality turns the identity into a bound, $\textrm{CE}(P,Q) \ge H(P)$ with
equality if and only if $Q = P$. Since $H(P)$ does not depend on $Q$,
minimizing cross-entropy in $Q$ is minimizing KL, and the KL term reaches its
minimum value, zero, at $Q = P$. For the running pair,
$\textrm{CE}(P, Q) = -(0.6 \ln 0.2 + 0.3 \ln 0.5 + 0.1 \ln 0.3) = 1.294$
nats and $H(P) = 0.898$ nats; their difference, $0.396$ nats, is the KL
divergence computed above.

:numref:`fig_mdl-code-length-bars` draws the decomposition in the questions
game, for the dyadic source and the uniform model $U = (\tfrac14, \tfrac14,
\tfrac14, \tfrac14)$. Asking adaptively takes $H_2(P) = 1.75$ questions on
average. The two fixed questions of the uniform questioner take
$\textrm{CE}_2(P, U) = 2$, and the difference, $D_{\textrm{KL}}(P\|U) = 0.25$
bits, is the waste. For this pair the reverse divergence $D_{\textrm{KL}}(U\|P)$
also happens to be $0.25$ bits, which is why we showed the asymmetry of KL on
the other pair.

![The decomposition :eqref:`eq_mdl-ce_decomp` for the dyadic source $P = (\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \tfrac{1}{8})$ and the uniform model $U$, in bits. Asking the questions adapted to $P$ takes $H_2(P) = 1.75$ questions on average, asking two fixed questions takes $\textrm{CE}_2(P, U) = 2.0$, and the waste is $D_{\textrm{KL}}(P\|U) = 0.25$ bits.](../img/mdl-it-code-length-bars.svg)
:label:`fig_mdl-code-length-bars`

Does the decomposition hold on both running examples? The next cell computes
the cross-entropy directly from its definition and compares it with $H + D_{\textrm{KL}}$:
the pair in nats, and the dyadic source against the uniform model in bits.

```{.python .input #information-theory-kl-categorical}
ce_pq = float(-np.dot(p, np.log(q)))
print(f'CE(P,Q) = {ce_pq:.4f}, H(P) = {entropy(p):.4f}, '
      f'CE - H = {ce_pq - entropy(p):.4f}')
u = np.full(4, 1 / 4)
bits = math.log(2)
h_bits = entropy(p_dyadic) / bits
kl_bits = kl_divergence(p_dyadic, u) / bits
ce_bits = float(-np.dot(p_dyadic, np.log2(u)))
print(f'dyadic vs uniform, in bits: H = {h_bits:.2f}, KL = {kl_bits:.2f}, '
      f'CE = {ce_bits:.2f}')
```

In both lines $\textrm{CE} = H + D_{\textrm{KL}}$: for the pair,
$\textrm{CE} - H$ is the $0.3961$ nats of the previous cell, and the second
line is :numref:`fig_mdl-code-length-bars`.

The decomposition also tells which of the three quantities survive a change of
units for continuous data. Let $y = g(x)$ be invertible with Jacobian
determinant $J(x)$, and transport the model with the data, so that both
densities are divided by the same factor, $p_Y = p_X/|J|$ and
$q_Y = q_X/|J|$, each evaluated at $x = g^{-1}(y)$. The factor cancels in the
ratio $p/q$, so the KL divergence does not change. The cross-entropy shifts by
$\mathbb{E}_P[\log |J|]$, a constant that depends on the data and the map but
not on the model. Differences of cross-entropies, their gradients with
respect to the model, and the minimizing model are therefore unchanged. The
differential entropy shifts by the same constant, as we saw above.

The decomposition gives maximum likelihood a second reading. For discrete data, the
average negative log-likelihood of a model $q_{\boldsymbol\theta}$ is the
cross-entropy from the empirical distribution $\hat p_{\textrm{data}}$, which
puts mass $1/n$ on each of the $n$ observations, to the model
(:numref:`subsec_mdl-nll-crossentropy`). Maximum likelihood therefore
minimizes $D_{\textrm{KL}}(\hat p_{\textrm{data}}\|q_{\boldsymbol\theta})$.
For densities the same holds at the population level: when $h(p)$ is finite,
the expected negative log-density of the model is
$h(p) + D_{\textrm{KL}}(p\|q_{\boldsymbol\theta})$.

### KL Divergence Between Gaussians

Gaussians are the continuous distributions we meet most often, and
comparing two of them is a common task. A variational autoencoder, for
instance, penalizes the KL divergence between the Gaussian distribution of
its code and a standard normal prior. For two univariate Gaussians the KL
divergence has a closed form. For
$P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$,

$$
D_{\textrm{KL}}\!\left(\mathcal{N}(\mu_1,\sigma_1^2) \,\big\|\, \mathcal{N}(\mu_2,\sigma_2^2)\right)
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}.
$$
:eqlabel:`eq_mdl-gaussian_kl`

The divergence is a log ratio of standard deviations plus the variance and the
squared mean gap of $P$, both measured in units of the model's variance. It is
non-negative and zero only when $\mu_1 = \mu_2$ and $\sigma_1 = \sigma_2$, as
Gibbs' inequality requires. It is not symmetric, because $\sigma_1$ and
$\sigma_2$ enter differently. Exercise
:exref:`ex_mdl-information-theory-gaussian-kl` derives the formula from the
definition.

The smallest case is a unit mean shift with equal variances,
$P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(1, 1)$, where the formula gives
$(\mu_1 - \mu_2)^2/(2\sigma^2) = \tfrac12$ nat. The definition suggests a
second way to compute it: draw samples from $P$ and average
$\log p(x) - \log q(x)$. The next cell defines the closed form and this
Monte Carlo estimate.

```{.python .input #information-theory-example-1}
def gaussian_kl(mu1, sigma1, mu2, sigma2):
    """Closed-form KL(N(mu1, sigma1^2) || N(mu2, sigma2^2)), in nats."""
    return (math.log(sigma2 / sigma1)
            + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
            - 0.5)

def gaussian_log_pdf(x, mu, sigma):
    """Log-density of N(mu, sigma^2) at x."""
    return (-0.5 * math.log(2 * math.pi * sigma ** 2)
            - (x - mu) ** 2 / (2 * sigma ** 2))

def mc_kl(mu1, sigma1, mu2, sigma2, n=200000):
    """Monte Carlo estimate of the same KL from n samples x ~ P."""
    x = np.random.default_rng().normal(loc=mu1, scale=sigma1, size=n)
    return float((gaussian_log_pdf(x, mu1, sigma1)
                  - gaussian_log_pdf(x, mu2, sigma2)).mean())
```

Does the Monte Carlo estimate agree with the formula for the unit mean shift?

```{.python .input #information-theory-example-2}
closed_form = gaussian_kl(0.0, 1.0, 1.0, 1.0)
monte_carlo = mc_kl(0.0, 1.0, 1.0, 1.0)

closed_form, monte_carlo
```

The closed form gives $0.5$. For this pair $\log p(x) - \log q(x) = \tfrac12 - x$
has variance one, so the average of $2 \cdot 10^5$ samples has standard error
$0.0022$. The estimate usually falls within about $\pm 0.005$ (two standard
errors) of $\tfrac12$ and almost never beyond $\pm 0.01$.

Now let the variances differ: $P = \mathcal{N}(0,1)$ and
$Q = \mathcal{N}(0,4)$, so $\sigma_2 = 2$. The formula gives
$D_{\textrm{KL}}(P\|Q) = \ln 2 + \tfrac18 - \tfrac12 \approx 0.318$ and
$D_{\textrm{KL}}(Q\|P) = -\ln 2 + 2 - \tfrac12 \approx 0.807$ nats, and the
next cell evaluates both directions.

```{.python .input #information-theory-example-3}
kl_pq = gaussian_kl(0.0, 1.0, 0.0, 2.0)
kl_qp = gaussian_kl(0.0, 2.0, 0.0, 1.0)

kl_pq, kl_qp
```

The two directions differ by a factor of $2.5$. When the narrow model has to
explain wide data, the tails dominate: there the narrow model gives almost no
probability to outcomes that occur, and its surprise is large ($0.807$ nats).
When the wide model explains narrow data, it is a little too surprised
everywhere ($0.318$ nats). :numref:`fig_mdl-kl-integrand` shows where each
number comes from. In both directions the integrand changes sign at
$|x| = (8 \ln 2/3)^{1/2} \approx 1.36$, where the two densities cross. In
panel (a) the core $|x| < 1.36$ contributes $+0.424$ and the tails $-0.106$.
In panel (b) the core contributes $-0.240$, and the tails, where the narrow
model is very surprised, contribute $+1.046$. The two directions contrast a
model that spreads its mass to cover the data with one that concentrates it.

![Where the KL divergence between $\mathcal{N}(0,1)$ and $\mathcal{N}(0,4)$ comes from. (a) The narrow distribution coded with the wide model: a positive core and slightly negative tails sum to $0.318$ nats. (b) The wide distribution coded with the narrow model: the tails, where the narrow model assigns almost no probability, contribute $1.05$ nats against a core of $-0.24$, for a total of $0.807$. The sign changes at $|x| = 1.36$ in both directions.](../img/mdl-it-kl-integrand.svg)
:label:`fig_mdl-kl-integrand`

### Maximum Entropy and the Gaussian
:label:`subsec_mdl-gaussian-max-entropy`

On a finite set the uniform distribution is the hardest to predict. On
$\mathbb{R}$ there is no uniform density, and with no constraint the
differential entropy is unbounded. Fix the mean and the variance: which
density is then the hardest to predict? :numref:`sec_mdl-exponential-family`
used the answer. With cross-entropy available, the proof takes two steps.

**Proposition (Gaussian maximum entropy).** *Among all densities $p$ on
$\mathbb{R}$ with mean $\mu$ and variance $\sigma^2$,*

$$
h(P) \leq \tfrac{1}{2}\log(2\pi e \sigma^2) = h(\mathcal{N}(\mu, \sigma^2)),
$$
:eqlabel:`eq_mdl-gaussian-max-entropy`

*with equality if and only if $P = \mathcal{N}(\mu, \sigma^2)$.*

**Proof.** Let $\varphi(x) = (2\pi\sigma^2)^{-1/2}
e^{-(x-\mu)^2/(2\sigma^2)}$ be the Gaussian density and $P$ any density with
mean $\mu$ and variance $\sigma^2$. The first step is the one that matters:
$-\log \varphi(x) = \tfrac{1}{2}\log(2\pi\sigma^2) +
\tfrac{(x-\mu)^2}{2\sigma^2}$ is a quadratic in $x$, so its expectation under
$P$ depends on $P$ only through the two constrained moments,

$$
\textrm{CE}(P, \varphi)
= \tfrac{1}{2}\log(2\pi\sigma^2) + \frac{\mathbb{E}_P\big[(x-\mu)^2\big]}{2\sigma^2}
= \tfrac{1}{2}\log(2\pi\sigma^2) + \tfrac{1}{2}
= \tfrac{1}{2}\log(2\pi e \sigma^2),
$$

the same value for every such $P$. For $P = \varphi$ itself, this is the
Gaussian's own differential entropy,
$h(\mathcal{N}(\mu, \sigma^2)) = \tfrac{1}{2}\log(2\pi e \sigma^2)$, the
closed form quoted whenever a Gaussian entropy is needed. Second, since this
cross-entropy is finite, the decomposition :eqref:`eq_mdl-ce_decomp` holds for
densities, $\textrm{CE}(P, \varphi) = h(P) + D_{\textrm{KL}}(P\|\varphi)$, and
Gibbs' inequality makes the KL term non-negative. Hence
$h(P) \le \tfrac{1}{2}\log(2\pi e \sigma^2)$, with equality if and only if
$D_{\textrm{KL}}(P\|\varphi) = 0$, that is, $P = \mathcal{N}(\mu, \sigma^2)$.
$\blacksquare$

The bound works because the Gaussian's log-density is a linear combination
of $x$ and $x^2$, the statistics whose expectations are constrained, so the
cross-entropy to the Gaussian is the same for every competitor $P$. The same argument with only
the mean fixed, on the outcomes $\{0, 1, 2, \ldots\}$, gives the geometric
distribution (Exercise
:exref:`ex_mdl-information-theory-geometric-max-entropy`). On $\mathbb{R}$
with only the mean fixed there is no maximum: $\mathcal{N}(\mu, \sigma^2)$ has
the right mean for every $\sigma$, and its entropy grows without limit with
$\sigma$.

## The Classification Loss

### Cross-Entropy Loss from Logits

In a $k$-class classification problem, the truth for one example is the
one-hot distribution that puts all its mass on the correct label $y$, and the
model supplies a predicted distribution $\mathbf{q} = (q_1, \ldots, q_k)$.
The cross-entropy :eqref:`eq_mdl-ce_def` between them collapses to a single
term, $-\log q_y$, the model's surprise at the correct class. Averaged over
$n$ examples with labels $y_i$ and predictions $\mathbf{q}_i$, the
*cross-entropy loss* is

$$
\textrm{CE} = -\frac{1}{n} \sum_{i=1}^n \log q_{i, y_i},
$$

in nats, the quantity a built-in classification loss computes. In code it is
one line of indexing.

```{.python .input #information-theory-formal-definition-1}
def cross_entropy(q, y):
    """Mean cross-entropy loss for predicted probabilities q, in nats."""
    return float(-np.log(q[np.arange(len(q)), y]).mean())
```

Take two examples with three classes. The first example's true class is $0$,
predicted with probability $0.3$; the second's is $2$, predicted with
probability $0.5$. The loss should be
$\tfrac{1}{2}(-\ln 0.3 - \ln 0.5) \approx 0.949$ nats.

```{.python .input #information-theory-formal-definition-2}
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

The built-in loss computes the same number from the *logits* $\mathbf{z}$,
the unnormalized scores whose softmax is $\mathbf{q}$. It applies the
log-softmax inside, which is numerically stable, whereas applying `log` to a
computed softmax can underflow (:numref:`subsec_mdl-stable-softmax` derives
the stable form). We build $\mathbf{z} = \log \mathbf{q}$ only because the toy
example starts from probabilities: for a normalized $\mathbf{q}$,
$\mathrm{softmax}(\log \mathbf{q}) = \mathbf{q}$. Passing probabilities to a
loss that expects logits would apply the softmax twice. Does each framework's
built-in loss reproduce $0.949$?

```{.python .input #information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification}
#@tab mxnet
preds_mx = mnp.array(preds.tolist())
labels_mx = mnp.array(labels.tolist(), dtype='int32')
logits = mnp.log(preds_mx)
log_probs = npx.log_softmax(logits, axis=1)
-log_probs[mnp.arange(len(labels_mx)), labels_mx].mean()
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

The built-in loss reports $0.9486$, the value computed by hand.

### The Irreducible Loss
:label:`subsec_mdl-irreducible-loss`

We can now answer the last question of the opening: how low can the loss go
when some labels are wrong? In real data the label is not a function of the
input: the same image can carry different labels, from ambiguity or from
annotation errors. With ten classes and $10\%$ of the labels flipped
uniformly to one of the other nine classes, the answer will turn out to be
$0.545$ nats.

Let the input $\mathbf{x}$ and the label $y \in \{1, \ldots, k\}$ have a
joint distribution, let $p(\cdot \mid \mathbf{x})$ be the true conditional
distribution of the label, and let $q(\cdot \mid \mathbf{x})$ be any model's.

**Proposition (the irreducible classification loss).** *The population
cross-entropy loss of any model satisfies*

$$
\mathbb{E}_{(\mathbf{x}, y)}\big[-\log q(y \mid \mathbf{x})\big]
= \mathbb{E}_{\mathbf{x}}\big[H(p(\cdot \mid \mathbf{x}))\big]
+ \mathbb{E}_{\mathbf{x}}\big[D_{\textrm{KL}}\big(p(\cdot \mid \mathbf{x}) \,\|\, q(\cdot \mid \mathbf{x})\big)\big]
\ge \mathbb{E}_{\mathbf{x}}\big[H(p(\cdot \mid \mathbf{x}))\big],
$$
:eqlabel:`eq_mdl-irreducible-loss`

*with equality if and only if $q(\cdot \mid \mathbf{x}) = p(\cdot \mid
\mathbf{x})$ for almost every $\mathbf{x}$ (with respect to the distribution
of the inputs).*

**Proof.** Condition on $\mathbf{x}$. The inner expectation over $y$ is the
cross-entropy $\textrm{CE}(p(\cdot \mid \mathbf{x}), q(\cdot \mid
\mathbf{x}))$, and the decomposition :eqref:`eq_mdl-ce_decomp` applies at each
$\mathbf{x}$; the entropies are finite, at most $\log k$. Averaging over
$\mathbf{x}$ gives the identity. Gibbs' inequality makes every KL term
non-negative, and an average of non-negative terms is zero only when almost
all of them vanish. $\blacksquare$

The floor is the average entropy of the label given the input, called the
*conditional entropy*. It is a
property of the data: no architecture, data volume, or training time lowers
it.
Three cases show what the floor means, beginning with the $2.30$ of an
untrained classifier.

* A model that predicts the uniform distribution has cross-entropy $\ln k$ at
  every input, whatever the labels are: $2.303$ nats for $k = 10$.
* If the label is a function of the input, every $H(p(\cdot \mid \mathbf{x}))$
  is zero, and a perfect model reports $0$.
* Under symmetric label noise, the label is the clean class with probability
  $1 - \rho$ and each of the other $k - 1$ classes with probability
  $\rho/(k-1)$. Then
  $H = -(1-\rho)\log(1-\rho) - \rho\log\tfrac{\rho}{k-1} = H_b(\rho) + \rho\log(k-1)$,
  which for $\rho = 0.1$ and $k = 10$ is $0.325 + 0.220 = 0.545$ nats. The
  best prediction of the class is the clean one, so the best accuracy is
  $1 - \rho = 90\%$.

The next cell evaluates the floor for the noisy case and the loss of three
predictors: the uniform one, a confident one that puts $0.99$ on the clean
class, and the Bayes predictor $q = p$.

```{.python .input #mdl-information-theory-the-irreducible-loss}
k, rho = 10, 0.1
p_noisy = np.array([1 - rho] + [rho / (k - 1)] * (k - 1))  # clean class first
predictors = {'uniform': np.full(k, 1 / k),
              'confident 0.99': np.array([0.99] + [0.01 / (k - 1)] * (k - 1)),
              'Bayes q = p': p_noisy}
print(f'floor = H(p) = {entropy(p_noisy):.3f} nats')
for name, q_hat in predictors.items():
    ce = float(-np.dot(p_noisy, np.log(q_hat)))
    print(f'{name:15s} CE = {ce:.3f} nats')
```

The uniform predictor reports $\ln 10 = 2.303$, regardless of the data. The
confident predictor reports $0.689$, above the floor, because every flipped
label surprises it. Only the Bayes predictor reaches the floor of $0.545$. A
validation loss below this floor on such data cannot come from a better
model; Exercise :exref:`ex_mdl-information-theory-label-noise-floor` asks
what it signals instead. Label smoothing, treated below, applies this
proposition with a smoothed target in the role of $p$.

### Proper Scoring Rules

The cross-entropy loss also has a decision-theoretic justification. A
*scoring rule* assigns a penalty $S(\mathbf{q}, y)$ for having reported the
probability vector $\mathbf{q}$ when class $y$ then occurs. The rule is
*proper* if truthful reporting is optimal: when the labels follow
$\mathbf{p}$, the expected penalty $\mathbb{E}_{y \sim \mathbf{p}}[S(\mathbf{q},
y)]$ is minimized by reporting $\mathbf{q} = \mathbf{p}$. It is *strictly
proper* if this report is the unique optimum. The cross-entropy loss is the
*log score* $S(\mathbf{q}, y) = -\log q_y$, and it is strictly proper: its
expected penalty is $\textrm{CE}(\mathbf{p}, \mathbf{q}) = H(\mathbf{p}) +
D_{\textrm{KL}}(\mathbf{p} \| \mathbf{q})$, which Gibbs' inequality minimizes
uniquely at $\mathbf{q} = \mathbf{p}$. Strict propriety of the log score is
Gibbs' inequality restated.

At the population optimum over a model class that contains the truth, the
predicted probabilities are therefore the true conditional probabilities. In
particular they are *calibrated*: among the inputs that receive probability
$0.8$ for a class, that class occurs $80\%$ of the time. A finite sample or a
restricted model class can still leave a trained model miscalibrated.

Other strictly proper rules exist. The *Brier score*
$S(\mathbf{q}, y) = \sum_j (q_j - \mathbf{1}(j = y))^2$ is strictly proper
too, with a bounded penalty on confident errors where the log score's
penalty diverges :cite:`Gneiting.Raftery.2007`. Exercise
:exref:`ex_mdl-information-theory-scoring-rule-is-strictly` computes its
excess penalty and tests an improper rule.

## Label Smoothing and Distillation

Label smoothing and knowledge distillation both train against a target other
than the one-hot label. In both, Gibbs' inequality identifies the optimum. In
distillation, a separate argument about the scale of the gradient also fixes
the scale of the loss.

### Label Smoothing

A classifier trained on one-hot targets is asked to assign probability $1$ to
the true class, which a softmax can approach only by driving the true class's
logit infinitely far above the rest. The result is overconfident models and
ever-growing weights. *Label smoothing* :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`
replaces the one-hot target $\mathbf{e}_y$ by a mixture with a fraction
$\epsilon$ of the uniform distribution over all $k$ classes,

$$
\mathbf{p}^\epsilon = (1 - \epsilon)\, \mathbf{e}_y + \frac{\epsilon}{k}\, \mathbf{1}.
$$
:eqlabel:`eq_mdl-label-smoothing-target`

The true class gets target probability $1 - \epsilon + \epsilon/k$ and each
other class $\epsilon/k$, so the other classes receive $\epsilon(k-1)/k$ in
total, $0.09$ for $k = 10$ and $\epsilon = 0.1$. The training loss is the
ordinary cross-entropy $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ to this
softened target. :numref:`sec_training_recipes` writes the same loss as
$-(1-\epsilon)\log q_y - \tfrac{\epsilon}{K}\sum_j \log q_j$, with $K$ for our
$k$. What prediction minimizes it?

**Proposition (the optimal smoothed prediction).** *Over all probability
vectors $\mathbf{q}$, the cross-entropy
$\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ is minimized uniquely at
$\mathbf{q}^* = \mathbf{p}^\epsilon$, where it equals $H(\mathbf{p}^\epsilon)$.*

**Proof.** This is the irreducible-loss proposition with
$\mathbf{p}^\epsilon$ in the role of the label distribution. By
:eqref:`eq_mdl-ce_decomp`,
$\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q}) = H(\mathbf{p}^\epsilon) +
D_{\textrm{KL}}(\mathbf{p}^\epsilon \| \mathbf{q})$. The first term does not
depend on $\mathbf{q}$, and by Gibbs' inequality the second is non-negative
and zero if and only if $\mathbf{q} = \mathbf{p}^\epsilon$. $\blacksquare$

The optimum is no longer at infinity. The model is asked to predict
$1 - \epsilon + \epsilon/k$ on the true class, which a softmax reaches with
the finite logit gap

$$
z_y - z_j = \log\frac{1 - \epsilon + \epsilon/k}{\epsilon/k}
= \log\!\Big(1 + \frac{k(1-\epsilon)}{\epsilon}\Big)
$$

between the true class and each other class. For $k = 10$ classes and
$\epsilon = 0.1$, the gap is $\ln 91 \approx 4.51$, a target the model can
reach. The loss at the optimum is the entropy $H(\mathbf{p}^\epsilon)$, which
is not zero. A label-smoothed loss curve therefore levels off above zero even
when the model predicts the softened target. The next cell checks
both numbers and evaluates an overconfident prediction with logit gap $10$.

```{.python .input #information-theory-label-smoothing}
k, eps = 10, 0.1
p_eps = np.full(k, eps / k)
p_eps[0] += 1 - eps                               # the true class first
gap = math.log(p_eps[0] / p_eps[1])               # optimal logit gap
z = np.array([10.0] + [0.0] * (k - 1))            # an overconfident prediction
q_conf = np.exp(z) / np.exp(z).sum()
print(f'optimal logit gap = ln(91) = {gap:.4f}')
print(f'CE at q* = H(p_eps) = {entropy(p_eps):.4f} nats')
print(f'CE at the overconfident q = {float(-np.dot(p_eps, np.log(q_conf))):.4f} nats')
```

The optimum has loss $0.500$ nats. The overconfident prediction has loss
$0.900$ nats, worse than the optimum, so label smoothing penalizes the
behavior that one-hot targets demand. Exercise
:exref:`ex_mdl-information-theory-label-smoothing` asks what smoothing does to
the logits of the wrong classes, and why teachers trained with it distill
poorly.

### Knowledge Distillation
:label:`subsec_mdl-distillation`

*Knowledge distillation* :cite:`Hinton.Vinyals.Dean.2015` trains a small
*student* network to imitate a large *teacher*. The student learns the
teacher's whole distribution over the classes, not only its most probable
class. That a "2" looks somewhat like a "7" is information a one-hot label
does not carry. A trained teacher's softmax is nearly one-hot, so both
distributions are first *softened*: the logits are divided by a
*temperature* $\tau > 0$ before the softmax,

$$
p_j^{(\tau)} = \mathrm{softmax}(\mathbf{z}^{\textrm{tea}}/\tau)_j, \qquad
q_j^{(\tau)} = \mathrm{softmax}(\mathbf{z}^{\textrm{stu}}/\tau)_j.
$$

A temperature above one shrinks the gaps between logits and moves
probability toward the wrong classes. The distillation loss is the KL divergence between the softened distributions,
scaled by $\tau^2$:

$$
\mathcal{L}_{\textrm{distill}}
= \tau^2\, D_{\textrm{KL}}\big(\mathbf{p}^{(\tau)} \,\|\, \mathbf{q}^{(\tau)}\big).
$$
:eqlabel:`eq_mdl-distill-loss`

By Gibbs' inequality, the student's optimum is to match the softened teacher.
Where does the $\tau^2$ come from? Differentiate the KL term with respect to a
student logit. Only $-\sum_j p_j^{(\tau)} \log q_j^{(\tau)}$ depends on the
student, and the derivative of a log-softmax at temperature $\tau$ is the
usual softmax-minus-target expression scaled by $1/\tau$:

$$
\frac{\partial}{\partial z^{\textrm{stu}}_j}
D_{\textrm{KL}}\big(\mathbf{p}^{(\tau)} \| \mathbf{q}^{(\tau)}\big)
= \frac{1}{\tau}\big(q_j^{(\tau)} - p_j^{(\tau)}\big).
$$

Each student logit is pushed by the gap between the two softened
probabilities of its class, divided by $\tau$.

For large $\tau$ the softened distributions flatten toward uniform. Subtract
the mean logit $\bar z$, which leaves the softmax unchanged, and expand
$e^{(z_j - \bar z)/\tau} \approx 1 + (z_j - \bar z)/\tau$ to first order. The
denominator becomes $k$ because the deviations sum to zero, so
$\mathrm{softmax}(\mathbf{z}/\tau)_j \approx \tfrac{1}{k}\big(1 + (z_j - \bar
z)/\tau\big)$. The difference $q_j^{(\tau)} - p_j^{(\tau)}$ therefore shrinks
like $1/\tau$, and the gradient like $1/\tau^2$:

$$
\frac{1}{\tau}\big(q_j^{(\tau)} - p_j^{(\tau)}\big)
\;\approx\; \frac{1}{k\,\tau^2}\Big(\big(z^{\textrm{stu}}_j - \bar z^{\textrm{stu}}\big)
- \big(z^{\textrm{tea}}_j - \bar z^{\textrm{tea}}\big)\Big).
$$

Multiplying the loss by $\tau^2$ therefore keeps the size of the gradient
roughly constant as the temperature is tuned. The temperature can then be
chosen for the quality of the soft targets without changing the loss's weight
relative to any hard-label loss it is mixed with. The next cell
checks both claims for teacher logits $(5, 2, -1)$ and student logits
$(3, 3, 0)$: the closed-form gradient matches autograd, and $\tau^2$ times the
gradient approaches the limit above instead of vanishing.

```{.python .input #information-theory-distillation}
#@tab mxnet
z_tea = mnp.array([5.0, 2.0, -1.0])
z_stu = mnp.array([3.0, 3.0, 0.0])
z_stu.attach_grad()

for tau in (1.0, 2.0, 5.0, 10.0):
    with autograd.record():
        p_tau, q_tau = npx.softmax(z_tea / tau), npx.softmax(z_stu / tau)
        kl = mnp.sum(p_tau * (mnp.log(p_tau) - mnp.log(q_tau)))
    kl.backward()
    closed = (npx.softmax(z_stu / tau) - npx.softmax(z_tea / tau)) / tau
    err = float(mnp.abs(z_stu.grad - closed).max())
    print(f'tau={tau:4.1f}  |autograd - closed| = {err:.1e}  '
          f'tau^2 grad = {[round(g, 3) for g in (tau**2 * z_stu.grad).tolist()]}')
```

```{.python .input #information-theory-distillation}
#@tab pytorch
z_tea = torch.tensor([5.0, 2.0, -1.0])

for tau in (1.0, 2.0, 5.0, 10.0):
    z_stu = torch.tensor([3.0, 3.0, 0.0], requires_grad=True)
    p_tau = torch.softmax(z_tea / tau, 0)
    q_tau = torch.softmax(z_stu / tau, 0)
    kl = torch.sum(p_tau * (torch.log(p_tau) - torch.log(q_tau)))
    grad, = torch.autograd.grad(kl, z_stu)
    closed = (q_tau - p_tau).detach() / tau
    err = (grad - closed).abs().max().item()
    print(f'tau={tau:4.1f}  |autograd - closed| = {err:.1e}  '
          f'tau^2 grad = {[round(g, 3) for g in (tau**2 * grad).tolist()]}')
```

```{.python .input #information-theory-distillation}
#@tab tensorflow
z_tea = tf.constant([5.0, 2.0, -1.0])
z_stu = tf.Variable([3.0, 3.0, 0.0])

for tau in (1.0, 2.0, 5.0, 10.0):
    with tf.GradientTape() as tape:
        p_tau, q_tau = tf.nn.softmax(z_tea / tau), tf.nn.softmax(z_stu / tau)
        kl = tf.reduce_sum(p_tau * (tf.math.log(p_tau) - tf.math.log(q_tau)))
    grad = tape.gradient(kl, z_stu)
    closed = (q_tau - p_tau) / tau
    err = float(tf.reduce_max(tf.abs(grad - closed)).numpy())
    print(f'tau={tau:4.1f}  |autograd - closed| = {err:.1e}  '
          f'tau^2 grad = {[round(g, 3) for g in (tau**2 * grad).numpy().tolist()]}')
```

```{.python .input #information-theory-distillation}
#@tab jax
z_tea = jnp.array([5.0, 2.0, -1.0])
z_stu = jnp.array([3.0, 3.0, 0.0])

def distill_kl(z_stu, tau):
    p_tau = jax.nn.softmax(z_tea / tau)
    q_tau = jax.nn.softmax(z_stu / tau)
    return jnp.sum(p_tau * (jnp.log(p_tau) - jnp.log(q_tau)))

for tau in (1.0, 2.0, 5.0, 10.0):
    grad = jax.grad(distill_kl)(z_stu, tau)
    closed = (jax.nn.softmax(z_stu / tau) - jax.nn.softmax(z_tea / tau)) / tau
    err = jnp.abs(grad - closed).max().item()
    print(f'tau={tau:4.1f}  |autograd - closed| = {err:.1e}  '
          f'tau^2 grad = {[round(g, 3) for g in (tau**2 * grad).tolist()]}')
```

The closed form matches autograd to floating-point precision at every
temperature. The raw gradient decays like $1/\tau^2$, while the
$\tau^2$-scaled gradient stays of order one. Its limit as $\tau \to \infty$ is
$\tfrac{1}{k}\big((\mathbf{z}^{\textrm{stu}} - \bar z^{\textrm{stu}}) -
(\mathbf{z}^{\textrm{tea}} - \bar z^{\textrm{tea}})\big) = (-0.667, 0.333,
0.333)$ here. The convergence is $O(1/\tau)$ and still incomplete in the
table: the $\tau = 10$ row reads $(-0.719, 0.413, 0.306)$, and the first
component overshoots its limit on the way. The scale stays fixed regardless,
which is what the $\tau^2$ factor is for. At $\tau = 1$ the loss is the
ordinary KL divergence. With a one-hot teacher it equals the ordinary
cross-entropy loss, because a one-hot distribution has zero entropy, so
standard training is a special case.

We can now read any log-loss as an average surprise with a floor, and compute
the floor from the label distribution: $0.545$ nats for $10\%$ symmetric
noise on $10$ classes. We can prove that the KL divergence is never negative,
and say what a softened target asks the model to predict: a logit gap of
$\ln 91 \approx 4.5$ for $\epsilon = 0.1$ and $k = 10$. Two questions remain
open. First, we called the loss a code length and quoted the questions game,
but we have not built a code. :numref:`sec_mdl-coding-compression` builds one
and asks how close a real compressor gets to the entropy floor. Second, KL
needs the model's probability of every outcome the data can produce. It is
infinite as soon as the model gives zero probability to one of them, as
happens for a generator whose samples lie on a thin surface.
:numref:`sec_mdl-divergences-distances` builds comparisons of distributions
that stay finite in that case.

## Summary

Self-information $-\log p$ is the surprise of an outcome, and entropy is its
average: at most $\log k$ on $k$ outcomes, and, in bits, the average number
of yes/no questions needed to identify the outcome, to within one question.
The KL divergence $D_{\textrm{KL}}(P\|Q)$ is the average extra surprise of
using $Q$ when the truth is $P$; by Gibbs' inequality it is never negative and
vanishes only at $P = Q$. The cross-entropy loss is
$H(P) + D_{\textrm{KL}}(P\|Q)$. The population loss of a classifier
therefore never falls below the average entropy of the label given the
input, and it reaches this floor only when the model predicts the true
conditional probabilities. On discrete data, minimizing the cross-entropy is
maximum likelihood. For continuous data,
differential entropy depends on the units and KL does not. A smoothed target
moves the optimum to a finite logit gap, and the $\tau^2$ factor in
distillation keeps the gradient scale fixed as the temperature changes.

## Exercises

1. **Gaussian KL divergence.** Let $P = \mathcal{N}(\mu_1, \sigma_1^2)$.
    1. Derive the closed form :eqref:`eq_mdl-gaussian_kl` by writing out
       $\mathbb{E}_{x\sim P}[\log p(x) - \log q(x)]$ and using
       $\mathbb{E}_{x \sim P}[x] = \mu_1$ and
       $\mathbb{E}_{x\sim P}[(x-\mu_1)^2] = \sigma_1^2$.
    1. Let $Q$ have mean $\mu_1 + \delta$ and standard deviation
       $\sigma_1 e^{\eta}$. Expand both $D_{\textrm{KL}}(P\|Q)$ and
       $D_{\textrm{KL}}(Q\|P)$ to second order in $\delta$ and $\eta$. At
       which order do the two directions first differ?
    1. Evaluate the second-order approximation for $P = \mathcal{N}(0, 1)$
       and $Q = \mathcal{N}(0, 4)$, and compare it with the exact values
       $0.318$ and $0.807$ nats of the two directions.
   :exlabel:`ex_mdl-information-theory-gaussian-kl`

1. **Other scoring rules.** A scoring rule $S(\mathbf{q}, y)$ is strictly
   proper if the expected penalty
   $\mathbb{E}_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$ is uniquely minimized
   at $\mathbf{q} = \mathbf{p}$. For the log score, this property is Gibbs'
   inequality.
    1. Show that the expected Brier score, with
       $S(\mathbf{q}, y) = \sum_j (q_j - \mathbf{1}(j = y))^2$, exceeds the
       expected Brier score of the truthful report by
       $\|\mathbf{q} - \mathbf{p}\|^2$, so that the Brier score is strictly
       proper :cite:`Gneiting.Raftery.2007`.
    1. The linear score $S(\mathbf{q}, y) = -q_y$ rewards the probability
       placed on the observed class. Which report minimizes its expected
       penalty? Is the linear score proper? What does a classifier trained
       with it report for an input whose label is class $1$ with probability
       $0.6$ and class $2$ with probability $0.4$?
   :exlabel:`ex_mdl-information-theory-scoring-rule-is-strictly`

1. **Uniqueness of self-information.** ● Show that self-information is
   unique up to scale: if a strictly decreasing function $I(p)$ on $(0, 1]$
   satisfies $I(1) = 0$ and $I(p_1 p_2) = I(p_1) + I(p_2)$ for all
   $p_1, p_2 \in (0, 1]$, then $I(p) = -c \log p$ for some constant
   $c > 0$. Work with the function $g(t) = I(e^{-t})$ for $t \geq 0$.
   :exlabel:`ex_mdl-information-theory-show-that-self-information`

1. **Gibbs' inequality from the tangent line.** The text proves Gibbs'
   inequality with Jensen's inequality. Here is a second proof.
    1. Show that $\ln u \le u - 1$ for all $u > 0$, with equality only at
       $u = 1$: the tangent line of $\ln$ at $u = 1$ lies above the concave
       graph.
    1. Apply the inequality with $u = q(x)/p(x)$ on the support of $P$ to
       show $D_{\textrm{KL}}(P\|Q) = -\sum_{x} p(x) \ln\frac{q(x)}{p(x)} \ge
       1 - Q(\textrm{supp}\,P) \ge 0$, and read off the equality case.
    1. Explain in one or two sentences why this is the same proof as the one
       in the text. Hint: the standard proof of Jensen's inequality for a
       concave $f$ bounds $f$ by its tangent line at $\mathbb{E}[X]$.
   :exlabel:`ex_mdl-information-theory-gibbs-tangent`

1. **Maximum entropy with a fixed mean.** Among all distributions on
   $\{0, 1, 2, \ldots\}$ with mean $\mu > 0$, show that the geometric
   distribution $g(j) = (1-r)r^j$ with $r = \mu/(1+\mu)$ has the largest
   entropy.
    1. Show that $-\log g(j)$ is affine in $j$, so that $\textrm{CE}(P, g)$
       is the same for every $P$ with mean $\mu$. Finish the proof with
       :eqref:`eq_mdl-ce_decomp`, as in the Gaussian case.
    1. For $\mu = 2$, compute $H(g) = -\log(1-r) - \mu \log r$, and compare
       it with the entropy of the distribution that puts probability
       $\tfrac12$ on $0$ and $\tfrac12$ on $4$, which has the same mean.
    1. Why is there no maximum-entropy density on $\mathbb{R}$ with only the
       mean fixed? Why does the argument of (a) fail there?
   :exlabel:`ex_mdl-information-theory-geometric-max-entropy`

1. [code] **Quantized entropy against differential entropy.** Let $X$ have
   density $p$, and let $X_\Delta$ record which cell
   $[i\Delta, (i+1)\Delta)$ contains $X$.
    1. Show that $H(X_\Delta) + \log \Delta \to h(X)$ as $\Delta \to 0$ when
       $p$ is continuous and $p$ and $p \log p$ are Riemann-integrable. Write
       $P(X_\Delta = i) = p(x_i)\Delta$ for some $x_i$ in the cell, by the
       mean value theorem for integrals.
    1. Check the limit by hand for the uniform density on $[0, 1]$ with
       $\Delta = \tfrac18$. What happens for a cell width that does not
       divide $1$, such as $\Delta = 0.3$?
    1. Compute $H(X_\Delta) + \log \Delta$ numerically for
       $X \sim \mathcal{N}(0, 1)$ with $\Delta = 0.5, 0.1, 0.01$, using the
       Gaussian distribution function for the cell probabilities, and compare
       it with $h = \tfrac12\log(2\pi e) \approx 1.4189$ nats. (At
       $\Delta = 0.5$ you should find $1.4292$.)
    1. Why does $H(X_\Delta)$ itself diverge as $\Delta \to 0$? A regression
       target is binned into cells of width $\Delta$ and predicted with a
       softmax classifier. Which quantity does its loss approach at best,
       and what must be added to it before it can be compared with the
       negative log-density of a continuous model?
   :exlabel:`ex_mdl-information-theory-quantized-entropy`

1. [code] **The label-noise floor.** Labels on $k$ classes are correct with
   probability $1-\rho$ and otherwise uniform over the other $k-1$ classes,
   independently of the input.
    1. Assume that the clean label is a function of the input. Show that the
       population cross-entropy of any classifier is at least
       $H_b(\rho) + \rho\log(k-1)$, that the optimal prediction is the noisy
       conditional distribution itself, and that the optimal accuracy is
       $1-\rho$ whenever $1 - \rho > \rho/(k-1)$.
    1. Evaluate the floor for $k = 10$ and $\rho = 0.05, 0.1, 0.2$.
    1. A colleague reports a validation loss of $0.40$ nats on a 10-class
       data set known to have $10\%$ such noise in every split. List possible
       explanations, such as label leakage, a different noise model, or
       evaluation on cleaned labels, and say which of them the floor rules
       out and which it does not.
    1. Train the softmax regression of :numref:`sec_softmax_scratch` on
       Fashion-MNIST with $10\%$ of the training and test labels flipped in
       this way. Compare the test loss with $0.545$ and the test accuracy with
       $90\%$. The gap is large. Which assumption of (a) does Fashion-MNIST
       violate, what does the linear model's own error add, and how could
       you estimate the floor that does apply?
   :exlabel:`ex_mdl-information-theory-label-noise-floor`

1. [code] **The initial-loss check.** A freshly initialized network produces
   small, random logits.
    1. Let the $k$ logits be independent with mean $0$ and variance $s^2$.
       Expand $\log \sum_j e^{z_j}$ to second order in the $z_j$ to show
       $\mathbb{E}[\textrm{CE}] \approx \ln k + \tfrac{s^2}{2}\big(1 -
       \tfrac1k\big)$.
    1. Check the formula by sampling Gaussian logits for $k = 10$ and
       $s = 0.1, 0.5, 1$. Where does the second-order approximation start to
       fail, and in which direction?
    1. A 10-class model reports a loss of $4.1$ at initialization. What does
       the formula say about the scale of its logits, and what would you
       change in the initialization?
   :exlabel:`ex_mdl-information-theory-initial-loss`

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
    1. Write $q_j = p_j(1 + u_j)$, so that $\sum_j p_j u_j = 0$, and expand
       $D_{\textrm{KL}}(\mathbf{p} \| \mathbf{q}) = -\sum_j p_j \log(1 + u_j)$
       to second order to obtain

       $$
       D_{\textrm{KL}}(\mathbf{p} \| \mathbf{q}) =
       \frac{1}{2}\sum_j \frac{(q_j - p_j)^2}{p_j}
       + O\big(\|\mathbf{q} - \mathbf{p}\|^3\big).
       $$

       Combine this expansion with the large-$\tau$ approximation of
       $\mathrm{softmax}(\mathbf{z}/\tau)$ in
       :numref:`subsec_mdl-distillation` to show that, as $\tau \to \infty$,

       $$
       \tau^2 D_{\textrm{KL}}\big(\mathbf{p}^{(\tau)} \,\|\, \mathbf{q}^{(\tau)}\big)
       \to \frac{1}{2k} \sum_{j=1}^k
       \Big(\big(z^{\textrm{stu}}_j - \bar z^{\textrm{stu}}\big)
       - \big(z^{\textrm{tea}}_j - \bar z^{\textrm{tea}}\big)\Big)^2.
       $$

    1. Evaluate $\tau^2 D_{\textrm{KL}}(\mathbf{p}^{(\tau)} \| \mathbf{q}^{(\tau)})$
       for $\tau = 1, 2, 5, 10, 100$ with the logits `z_tea` $= (5, 2, -1)$
       and `z_stu` $= (3, 3, 0)$ of :numref:`subsec_mdl-distillation`, and
       compare it with the limit.
    1. At low temperature, which teacher logits have almost no influence on
       the loss? :citet:`Hinton.Vinyals.Dean.2015` argue that this can be an
       advantage and also a drawback. Give one argument for each side,
       based on what the teacher's own cross-entropy training does to these
       logits.
   :exlabel:`ex_mdl-information-theory-distillation-high-temperature`

[Discussions](https://d2l.discourse.group/t/1104)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §26.1]{.kicker}

The loss you report is an average surprise<br>**entropy, cross-entropy, and KL divergence**
:::
:::

::: {.slide title="A Fresh 10-Class Model Reports 2.30, a Perfect One 0, a Noisy One at Best 0.545"}
[The problem]{.kicker}

- An untrained classifier, nearly uniform predictions: $\ln 10 \approx 2.30$ nats.
- Clean labels and a perfect model: $0$.
- $10\%$ of the labels flipped to one of the other nine classes: no model,
  however large, gets below $0.545$.

::: {.d2l-note .warn}
What does the number measure, and where is its floor?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute entropy and KL divergence of small distributions by hand;
- prove that KL is never negative (Gibbs' inequality);
- split any log-loss into a floor plus a never-negative excess;
- compute the floor of a noisy classification problem;
- say what label smoothing and distillation ask the model to predict.

**Running examples:** the dyadic source $(\tfrac12, \tfrac14, \tfrac18, \tfrac18)$;
the pair $P = (0.6, 0.3, 0.1)$, $Q = (0.2, 0.5, 0.3)$.
**Idea:** loss $=$ average surprise $=$ entropy $+$ KL.
:::

::: {.slide title="Losses First, Codes and Comparisons Later"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **Entropy, Cross-Entropy, and KL** | **surprise, KL, the loss floor** |
| Codes, Compression, and Perplexity | the code behind the log-loss |
| Divergences and Distances | four ways to compare two laws |
| Mutual Information | dependence and what it limits |
| Estimating Mutual Information | lower bounds from samples |
:::

::: {.col .narrow}
- **Uses:** softmax regression, Jensen's inequality.
- **Skips:** building a code, comparing distributions by samples.
:::
:::
:::

::: {.slide title="Surprise Is $-\log p$, and Independent Surprises Add"}
[Self-information]{.kicker}

::: {.cols .vc}
::: {.col}
$I(x) = -\log p(x)$, in nats. A card, a heart, the three of spades, the
order of the deck: $0$, $1.39$, $3.95$, $156.4$ nats ($0$, $2$, $5.70$,
$225.6$ bits).

Independent probabilities multiply, so a logarithm makes surprises add.
:::

::: {.col .fig}
@fig:mdl-it-self-info-curve
:::
:::
:::

::: {.slide title="Entropy Is the Average Number of Yes/No Questions"}
[Entropy]{.kicker}

$H(P) = -\sum_i p_i \log p_i$, with $0 \log 0 = 0$. Dyadic source: ask
"$a$?", then "$b$?", then "$c$?":
$\tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac18 \cdot 3 + \tfrac18 \cdot 3 = 1.75$.

@information-theory-definition

Two fixed questions would always take $2$.
:::

::: {.slide title="On $k$ Outcomes, Entropy Peaks at $\log k$, at the Uniform Law"}
[The maximum]{.kicker}

::: {.cols .vc}
::: {.col}
Jensen, for concave $f$: $\mathbb{E}[f(X)] \le f(\mathbb{E}[X])$; the chord
lies below the curve.

::: {.d2l-note .rule}
$H = \mathbb{E}[\log\tfrac{1}{p(X)}] \le \log\mathbb{E}[\tfrac{1}{p(X)}]
= \log m \le \log k$, with $m$ the number of outcomes with $p_i > 0$.
Equality: $1/p$ constant and $m = k$, the uniform law. $\blacksquare$
:::
:::

::: {.col .fig}
@fig:mdl-it-bernoulli-entropy
:::
:::
:::

::: {.slide title="Differential Entropy Moves with the Units"}
[Differential entropy]{.kicker}

$h(X) = -\int p(x) \log p(x)\,dx$: density values are not probabilities.

- Uniform on $[0, a]$: $h = \log a$; for $a = \tfrac12$, $h = -0.69$ nats.
- Metres to centimetres: $a \to 100a$, and $h$ gains $\ln 100 = 4.6$ nats.
- A ratio of densities does not move:
  $\int p_1 \log(p_1/p_2)\,dx = \log(a_2/a_1)$ in any unit.
- Cells of width $\Delta$: $H(X_\Delta) + \log \Delta \to h(X)$, while
  $H(X_\Delta)$ itself diverges.
:::

::: {.slide title="Check: What Does an Untrained 3-Class Model Report? A 10-Class One, in Bits?"}
[Check]{.kicker}

An untrained model predicts nearly uniform probabilities. What loss does it
report with $3$ classes, in nats? With $10$ classes, in bits?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\ln 3 \approx 1.10$ nats. $\log_2 10 \approx 3.32$ bits, which
is the familiar $2.30$ nats.
:::
:::
:::

::: {.slide title="KL Averages the Extra Surprise of Believing Q When the Truth Is P"}
[KL divergence]{.kicker}

$D_{\textrm{KL}}(P\|Q) = \mathbb{E}_{x \sim P}[\log p(x)/q(x)]$: surprise
under $Q$ minus surprise under $P$, averaged over $P$.

~~~python
def kl(p, q):
    m = p > 0
    return np.dot(p[m], np.log(p[m] / q[m]))
~~~

@!information-theory-definition-2

Asymmetric ($0.396$ against $0.365$), and $+\infty$ where $q = 0 < p$.
:::

::: {.slide title="KL Is Never Negative, and Zero Only at P = Q"}
[Gibbs' inequality]{.kicker}

::: {.d2l-note .rule}
$D_{\textrm{KL}}(P\|Q) \ge 0$, with equality if and only if $P = Q$.
:::

$$
D_{\textrm{KL}}(P\|Q) = \mathbb{E}_P\Big[-\log\frac{q}{p}\Big]
\ge -\log \mathbb{E}_P\Big[\frac{q}{p}\Big] = -\log Q(\mathrm{supp}\,P) \ge 0
$$

- Jensen for the convex $-\log$; the factors $p$ cancel.
- Equality: $q/p$ constant on the support, and $Q(\mathrm{supp}\,P) = 1$, so $q = p$.
- No model beats the truth on average.
:::

::: {.slide title="Cross-Entropy = Entropy + KL, So the Loss Has a Floor"}
[Cross-entropy]{.kicker}

::: {.cols .vc}
::: {.col}
$\textrm{CE}(P, Q) = -\mathbb{E}_P[\log q] = H(P) + D_{\textrm{KL}}(P\|Q)$

- the pair: $1.294 = 0.898 + 0.396$ nats;
- dyadic source, uniform model: $2.0 = 1.75 + 0.25$ bits;
- $H(P)$ does not depend on $Q$: minimizing CE is minimizing KL.
:::

::: {.col .fig}
@fig:mdl-it-code-length-bars
:::
:::
:::

::: {.slide title="A Narrow Model Is Far More Surprised by Wide Data Than the Reverse"}
[Gaussians]{.kicker}

$D_{\textrm{KL}} = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12$

::: {.cols .vc}
::: {.col .narrow}
$\mathcal{N}(0,1)$ against $\mathcal{N}(0,4)$: $0.318$ one way, $0.807$
the other. The wide tails, where the narrow model is very surprised, carry
the $0.807$.
:::

::: {.col .fig}
@fig:mdl-it-kl-integrand
:::
:::
:::

::: {.slide title="With Mean and Variance Fixed, the Gaussian Has the Largest Entropy"}
[Maximum entropy]{.kicker}

Gaussian density $\varphi$ with mean $\mu$, variance $\sigma^2$; any density
$P$ with the same two moments.

I. $-\log \varphi(x) = \tfrac12\log(2\pi\sigma^2) + \tfrac{(x-\mu)^2}{2\sigma^2}$
is quadratic, so $\textrm{CE}(P, \varphi) = \tfrac12 \log(2\pi e \sigma^2)$
for every such $P$.

II. $\textrm{CE}(P, \varphi) = h(P) + D_{\textrm{KL}}(P\|\varphi) \ge h(P)$,
with equality if and only if $P = \varphi$. $\blacksquare$

::: {.d2l-note}
The same trick with only the mean fixed on $\{0, 1, 2, \ldots\}$ gives the
geometric distribution.
:::
:::

::: {.slide title="Check: KL from P to Q′ = (0.5, 0.5, 0)? And CE(P, P)?"}
[Check]{.kicker}

For the truth $P = (0.6, 0.3, 0.1)$, compute $D_{\textrm{KL}}(P\|Q')$ for
the model $Q' = (0.5, 0.5, 0)$, and the cross-entropy $\textrm{CE}(P, P)$.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $D_{\textrm{KL}}(P\|Q') = +\infty$: the third outcome happens
one time in ten, and $Q'$ calls it impossible.
$\textrm{CE}(P, P) = H(P) = 0.898$ nats, the floor.
:::
:::
:::

::: {.slide title="The One-Hot Cross-Entropy Is the Model's Surprise at the Label"}
[Classification]{.kicker}

- One-hot target on the label $y$: $\textrm{CE} = -\log q_y$.
- Two examples, true classes predicted with $0.3$ and $0.5$:
  $\tfrac12(-\ln 0.3 - \ln 0.5) = 0.949$ nats.
- Built-in losses take logits $\mathbf{z}$ and apply a stable log-softmax
  inside:

@!information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification
:::

::: {.slide title="No Classifier Beats the Label Noise: 10% Noise on 10 Classes Floors the Loss at 0.545"}
[The floor]{.kicker}

$$
\mathbb{E}[-\log q(y \mid \mathbf{x})] = \mathbb{E}_{\mathbf{x}}[H(p(\cdot\mid\mathbf{x}))]
+ \mathbb{E}_{\mathbf{x}}[D_{\textrm{KL}}(p \,\|\, q)] \ge \mathbb{E}_{\mathbf{x}}[H(p(\cdot\mid\mathbf{x}))]
$$

Noise $\rho = 0.1$, $k = 10$:
$H_b(0.1) + 0.1 \ln 9 = 0.325 + 0.220 = 0.545$ nats. Three predictors:

@!mdl-information-theory-the-irreducible-loss
:::

::: {.slide title="Truthful Reporting Is the Unique Optimum of the Log Score"}
[Scoring rules]{.kicker}

- A scoring rule charges $S(\mathbf{q}, y)$ for reporting $\mathbf{q}$ when
  $y$ occurs.
- Strictly proper: $\mathbf{q} = \mathbf{p}$ is the unique minimizer of
  $\mathbb{E}_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$.
- Log score $-\log q_y$: expected penalty $H(\mathbf{p}) +
  D_{\textrm{KL}}(\mathbf{p}\|\mathbf{q})$, strictly proper by Gibbs.
- Brier score $\sum_j (q_j - \mathbf{1}(j = y))^2$: also strictly proper,
  with a bounded penalty on confident errors.
:::

::: {.slide title="A Smoothed Target Makes the Optimum a Finite Logit Gap: ln 91 = 4.51"}
[Label smoothing]{.kicker}

Target $\mathbf{p}^\epsilon = (1-\epsilon)\,\mathbf{e}_y + \tfrac{\epsilon}{k}\mathbf{1}$.
Since $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q}) = H(\mathbf{p}^\epsilon)
+ D_{\textrm{KL}}(\mathbf{p}^\epsilon\|\mathbf{q})$, the optimum is
$\mathbf{q} = \mathbf{p}^\epsilon$, with logit gap
$\log\frac{1-\epsilon+\epsilon/k}{\epsilon/k}$. For $k = 10$, $\epsilon = 0.1$:

@!information-theory-label-smoothing

The loss levels off at $0.500$ nats, not at $0$.
:::

::: {.slide title="The τ² Factor Keeps Distillation Gradients of Order One"}
[Distillation]{.kicker}

$\mathbf{p}^{(\tau)} = \mathrm{softmax}(\mathbf{z}^{\mathrm{tea}}/\tau)$,
$\mathbf{q}^{(\tau)} = \mathrm{softmax}(\mathbf{z}^{\mathrm{stu}}/\tau)$; loss
$\tau^2 D_{\textrm{KL}}(\mathbf{p}^{(\tau)}\|\mathbf{q}^{(\tau)})$.
The gradient $\tfrac1\tau(\mathbf{q}^{(\tau)} - \mathbf{p}^{(\tau)})$ shrinks like $1/\tau^2$:

@!information-theory-distillation

Limit as $\tau \to \infty$: centered logit differences over $k$,
$(-0.667, 0.333, 0.333)$.
:::

::: {.slide title="Check: With ε = 0.1 on 10 Classes, What Loss Does a Perfect Student Report?"}
[Check]{.kicker}

A student trained with label smoothing, $\epsilon = 0.1$ and $k = 10$,
predicts the smoothed target exactly. What loss does it report, and what is
its logit gap?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The entropy of the smoothed target, $H(\mathbf{p}^\epsilon) =
0.500$ nats, not $0$. The gap between the true class and every other class
is $\ln 91 \approx 4.51$.
:::
:::
:::

::: {.slide title="Every Log-Loss Is a Floor Plus a Never-Negative Excess"}
[Wrap-up]{.kicker}

- Dyadic source: $H_2 = 1.75$ bits, the average number of questions.
- The pair: $D_{\textrm{KL}} = 0.396$ one way, $0.365$ the other.
- Gaussians $\mathcal{N}(0,1)$, $\mathcal{N}(0,4)$: $0.318$ against $0.807$.
- $\textrm{CE} = H + D_{\textrm{KL}} \ge H$; on 10 classes with $10\%$
  noise the floor is $0.545$ nats.
- Label smoothing: optimum at a logit gap of $\ln 91$; distillation: the
  $\tau^2$ factor fixes the gradient scale.

::: {.d2l-note}
Next: build the code whose length is this loss, and compare distributions
when KL is infinite.
:::
:::
