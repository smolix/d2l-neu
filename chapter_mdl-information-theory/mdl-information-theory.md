# Entropy, Cross-Entropy, and KL Divergence
:label:`sec_mdl-information_theory`

A loss curve is the first thing we inspect when a classifier trains, and we
quickly learn to read its shape: a falling curve is good news, while a flat one
is suspicious. The numbers on its vertical axis are harder to interpret. So far
we can explain only one of them. A freshly initialized classifier with ten
classes reports a loss of about $2.30$, a value that
:numref:`sec_mlp-implementation` used as a sanity check on an untrained network.
That number is $\ln 10$, the cost of naming one of ten labels when the model
still considers all ten equally likely. The other end of the curve raises deeper
questions. Is a final loss of $0.6$ good? When a curve levels off, is the
plateau a bug in training or an unavoidable property of the data? And if some of
the labels are wrong, how low can any model hope to go?

The cross-entropy loss is the average of $-\log q(y)$, which measures how
surprised the model is by the label $y$ that actually occurred. This average
splits naturally into two pieces. The first is the *entropy*: the average
surprise experienced by someone who already knows the true distribution of the
labels. Because entropy is fixed by the data themselves, no model can reduce it.
The second piece is the *Kullback--Leibler (KL) divergence*: the extra surprise
incurred by using an imperfect model instead of the truth. It is never negative,
and it vanishes only when the model matches the true distribution. A loss value
can therefore be judged only against this irreducible floor.

We already encountered two of these ideas in the previous chapter: maximum
likelihood minimizes a cross-entropy (:numref:`subsec_mdl-nll-crossentropy`),
and the KL divergence is never negative
(:numref:`subsec_mdl-max-entropy-derivation`), a result known as Gibbs'
inequality. Whereas those facts served as tools there, here they are our main
subject. We begin with a deck of cards and a four-symbol source, where entropy
simply counts yes/no questions. From there we compare pairs of distributions,
return to our classifier to find its loss floor, and finally examine what
changes when targets are softened or when outcomes are continuous real numbers.

The base of the logarithm is merely a choice of unit: base $2$ measures
information in *bits*, while base $e$ measures it in *nats*. **We work in nats
throughout**, matching the natural logarithm `log` used in numerical libraries
and in reported training losses. Bits are more convenient when counting yes/no
questions, so we mark quantities in bits with a subscript $2$, as in $H_2$ for
entropy. One bit equals $\ln 2 \approx 0.693$ nats.
:numref:`tab_mdl-it-notation` collects the symbols used across this chapter.

:Symbols of the information theory chapter. The binary entropy $H_b(p)$ is a function of a number, not of a random variable. The mutual information $I(X;Y)$ and the last five rows are defined in later sections.
:label:`tab_mdl-it-notation`

| Symbol | Meaning | Unit |
|:--|:--|:--|
| $I(x) = -\log p(x)$ | self-information (surprisal) of the outcome $x$ (by convention the letter is shared with the mutual information $I(X;Y)$) | nats |
| $H(X)$, $H(P)$ | entropy of a discrete random variable or distribution; a subscript $2$, as in $H_2$ and $\textrm{CE}_2$, marks bits | nats |
| $H_b(p)$ | binary entropy of a coin with heads probability $p$ | nats |
| $h(X)$ | differential entropy of a continuous random variable | nats, depends on the units of $X$ |
| $D_{\textrm{KL}}(P \Vert Q)$ | KL divergence from $P$ to $Q$ | nats |
| $\textrm{CE}(P, Q)$ | cross-entropy from $P$ to $Q$ | nats |
| $k$ | number of classes (size of the set of outcomes) | |
| $n$ | number of examples, or of symbols in a message; the dimension in $\mathbb{R}^n$ | |
| $\mathbf{p}$, $\mathbf{q}$ | probability vectors on $k$ classes (truth, model) | |
| $\mathbf{z}$ | logits, with $\mathbf{q} = \mathrm{softmax}(\mathbf{z})$ | |
| $\epsilon$ | label-smoothing mass (in later sections, a noise variable) | |
| $\tau$ | temperature | |
| $\rho$ | label-noise rate: the probability that a recorded label is not the clean class | |
| $H(X \mid Y)$, $I(X;Y)$ | conditional entropy, mutual information | nats |
| $D_f(P \Vert Q)$ | f-divergence with divergence function $f$ | |
| $T$ | critic: a function whose averages under two distributions bound a divergence | |
| $g$ | test function of an integral probability metric | |
| $\gamma$ | coupling (transport plan) of two distributions | |
| $N$ | number of points in each of two samples that are compared; number of candidates in a contrastive batch | |

```{.python .input #information-theory-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #information-theory-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #information-theory-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #information-theory-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Information and Entropy

### Self-Information

Consider four reports about a freshly shuffled deck of cards. "I see a card"
conveys nothing, because the event is certain. "I see a heart" narrows the
outcome to one of four equally likely suits, "It is the three of spades" singles
out one of $52$ equally likely cards, and reporting the order of the entire deck
identifies one of $52!$ possible orderings. The less probable an event is, the
more we learn by observing it. Shannon :cite:`Shannon.1948` quantified this
idea with the *self-information* of an outcome $x$ that occurs with probability
$p(x)$,

$$
I(x) = -\log p(x).
$$

This quantity is also called the *surprisal* of the outcome, the term used in
:numref:`subsec_info_theory_basics`: the less we expected an outcome, the more
it surprises us. Importantly, the surprise belongs to whoever assigned the
probability, so a confidently wrong model finds ordinary events deeply
surprising. A certain event carries zero information, and halving an outcome's
probability adds $\ln 2$ nats (one bit) to its surprisal. The first three card
reports carry $0$, $\ln 4 \approx 1.39$, and $\ln 52 \approx 3.95$ nats. In
bits, where one bit is the information gained from a single fair coin flip, they
carry $0$, $2$, and $5.70$.

Why choose a logarithm? We want three properties: a certain event carries zero
information, rarer events carry more information, and the information of two
independent events adds up. Because independent probabilities multiply, the
third requirement reads $I(p_1 p_2) = I(p_1) + I(p_2)$. The only decreasing
functions satisfying this identity are positive multiples of $-\log p$
(:citet:`Csiszar.2008`; Exercise
:exref:`ex_mdl-information-theory-show-that-self-information` works through the
proof). Choosing the constant multiple simply chooses the unit.

![Self-information $I(x) = -\log p(x)$ in nats as a function of the probability $p$. Certain events ($p = 1$) carry zero information, a fair coin ($p = \tfrac{1}{2}$) carries $\ln 2 \approx 0.693$ nats, and the curve diverges as $p \to 0$.](../img/mdl-it-self-info-curve.svg)
:label:`fig_mdl-self-info-curve`

:numref:`fig_mdl-self-info-curve` plots the curve: it is decreasing and convex,
hits zero at $p = 1$, and diverges as $p \to 0$. The fourth card report, the
order of the entire deck, sits far out on this curve at $\ln 52!$. We can
compute its value via the log-gamma function, $\ln 52! = \ln \Gamma(53)$,
without ever forming the 68-digit integer $52!$.

```{.python .input #information-theory-self-information}
deck_order = math.lgamma(53)          # ln 52! without forming 52!
print(f'the order of the deck: {deck_order:.2f} nats '
      f'= {deck_order / math.log(2):.1f} bits')
```

Naming the order of a shuffled deck carries $156$ nats, roughly as much
information as $226$ fair coin flips.

### Entropy

We can compute a surprisal only after an outcome arrives. Beforehand, we can ask
how surprised we expect to be on average. Let $X$ be a discrete random variable
taking the value $x_i$ with probability $p_i$. Its *entropy* is the expected
self-information of its outcome,

$$
H(X) = -\mathbb{E}_{x \sim P}[\log p(x)] = -\sum_i p_i \log p_i,
$$
:eqlabel:`eq_mdl-ent_def`

with the convention $0 \log 0 = 0$, justified by continuity ($t \log t \to 0$
as $t \to 0$): an impossible outcome contributes nothing. Because the entropy
depends only on the distribution $P$ of $X$, we write $H(P)$ and $H(X)$
interchangeably.

On a finite set, entropy in bits has a concrete operational meaning: it counts
the average number of yes/no questions a good questioner needs to identify the
outcome. Take the distribution $P = (\tfrac12, \tfrac14, \tfrac18, \tfrac18)$ on
four symbols $a, b, c, d$. We will carry this example through the section and
call it the *dyadic source*, since all of its probabilities are powers of
$\tfrac12$. A good questioner first asks "Is it $a$?", settling the matter half
the time. If not, they ask "Is it $b$?", and then "Is it $c$?", where "no"
leaves only $d$. The expected number of questions is

$$
\tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac18 \cdot 3 + \tfrac18 \cdot 3
= 1.75,
$$

which matches the entropy in bits:
$H_2(P) = \tfrac12 \log_2 2 + \tfrac14 \log_2 4 + 2 \cdot \tfrac18 \log_2 8
= 1.75$ bits, or $1.213$ nats. Each outcome is identified in $\log_2(1/p_i)$
questions, which is its self-information in bits.

By contrast, a questioner who ignores the probabilities asks "Is it $a$ or $b$?"
and then "Which of the two?", always needing $2$ questions. Notice that a
questioning strategy is a binary code in disguise: write $1$ for yes and $0$ for
no, and the sequence of answers that identifies an outcome becomes its codeword,
here $1$, $01$, $001$, and $000$ for $a$, $b$, $c$, and $d$. The average
question count matched the entropy here because every probability was a power of
$\tfrac12$. In the language of codes, :numref:`sec_mdl-coding-compression`
proves that the count is never far off: for any distribution on a finite set,
the best strategy needs between $H_2$ and $H_2 + 1$ questions on average.

In code, the function `entropy` masks out zero probabilities, implementing
$0 \log 0 = 0$. We will reuse it throughout the section, starting with the
dyadic source.

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

It returns the $1.75$ bits that we counted in questions.

Because $0 \le p_i \le 1$, every term $-p_i \log p_i$ is non-negative, so
$H(X) \ge 0$, with equality only for a point mass (a variable whose outcome is
certain). Notice also that entropy depends only on the list of probabilities,
not on the numerical values or names of the outcomes. A variable taking the
values $1$, $100$, $10^6$, and $10^9$ with the dyadic probabilities still has an
entropy of $1.75$ bits.

### Joint and Conditional Entropy
:label:`subsec_mdl-joint-conditional-entropy`

In machine learning, a label always comes paired with an input, and a character
comes after the text preceding it. We therefore need both the entropy of two
variables taken together and the entropy remaining in one variable once the
other is known. The first requires nothing new: a pair $(X, Y)$ with joint
probability mass function $p(x, y)$ is simply a single random variable whose
outcomes are pairs, so its *joint entropy* is
$H(X, Y) = -\mathbb{E}[\log p(X, Y)]$. The *conditional entropy* is the average
surprisal left in $Y$ once $X$ is observed,

$$
H(Y \mid X) = -\mathbb{E}[\log p(Y \mid X)] = \sum_x p(x)\, H(Y \mid X = x),
$$

where $H(Y \mid X = x)$ is the entropy of the conditional distribution
$p(\cdot \mid x)$. Because $\log p(x, y) = \log p(x) + \log p(y \mid x)$, taking
expectations on both sides yields the *chain rule* for entropy:

$$
H(X, Y) = H(X) + H(Y \mid X).
$$
:eqlabel:`eq_mdl-entropy-chain-rule`

In words, the entropy of a pair is the entropy of the first variable plus the
uncertainty that remains in the second once the first is known. Applied
repeatedly, the chain rule gives
$H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_1, \ldots, X_{i-1})$: the
entropy of a sequence is the sum of the entropies of its symbols, each
conditioned on all preceding symbols. A language model predicts text in this
same left-to-right order, one symbol at a time.

The questions game illustrates this identity directly. Let $X$ be the answer to
the first question, "Is it $a$?", and let $Y$ be the outcome of the dyadic
source. Because the answer is a fair coin, $H_2(X) = 1$ bit. After "yes",
nothing remains to ask ($0$ bits). After "no", the outcome is $b$, $c$, or $d$
with probabilities $(\tfrac12, \tfrac14, \tfrac14)$, whose entropy is $1.5$
bits. Averaging over the two answers gives
$H_2(Y \mid X) = \tfrac12 \cdot 0 + \tfrac12 \cdot 1.5 = 0.75$ bits. By the
chain rule, $H_2(X, Y) = 1 + 0.75 = 1.75$ bits: the first answer supplies one
bit and the remaining questions supply the rest. This sum equals $H_2(Y)$, as it
must: knowing the outcome $Y$ completely determines the answer $X$, so
$H(X \mid Y) = 0$, and the chain rule in the reverse order reads
$H(X, Y) = H(Y) + 0$.

### Maximum Entropy on a Finite Set

How large can the entropy of a variable with $k$ outcomes be? Let us start with
the smallest case, a single coin flip. A Bernoulli variable with heads
probability $p$ has the *binary entropy*

$$
H_b(p) = -p \log p - (1-p) \log(1-p),
$$

which is a function of the scalar $p$ rather than of a random variable.
:numref:`fig_mdl-bernoulli-entropy` plots this curve. The entropy vanishes at
$p \in \{0, 1\}$, where the outcome is certain, and reaches its peak at the fair
coin, $H_b(\tfrac12) = \ln 2 \approx 0.693$ nats (one bit). A biased coin with
$p = 0.1$ has $H_b(0.1) \approx 0.325$ nats, because its outcome is much easier
to guess.

![The binary entropy $H_b(p)$ in nats, symmetric about $p = \tfrac{1}{2}$, where it peaks at $\ln 2 \approx 0.693$ nats. At the deterministic endpoints $p = 0$ and $p = 1$, the outcome carries no surprise and the entropy vanishes. The curve is concave: any chord lies below it.](../img/mdl-it-bernoulli-entropy.svg)
:label:`fig_mdl-bernoulli-entropy`

Notice that the curve is concave: the chord joining any two of its points lies
below the curve. It peaks at the uniform coin, and the same principle holds for
any finite number of outcomes. If $X$ takes at most $k$ values, then

$$
H(X) \leq \log k,
$$

with equality if and only if $X$ is uniform over all $k$ values. The uniform
distribution spreads its probability mass as evenly as possible, making the
outcome maximally hard to guess. For four outcomes, the upper bound is
$\log_2 4 = 2$ bits, which sits $0.25$ bits above the $1.75$ bits of the dyadic
source. We postpone the proof for a moment until Gibbs' inequality is in place,
at which point it takes just a single line. Since that inequality compares two
distributions, we turn to comparing distributions next.

## KL Divergence and Cross-Entropy

### The Kullback--Leibler Divergence

Suppose we want to measure how badly a model $Q$ misrepresents a true
distribution $P$. Information theory answers this with the *Kullback--Leibler
(KL) divergence*, also known as *relative entropy* :cite:`Kullback.Leibler.1951`:
the average extra surprisal incurred by someone who believes $Q$ when outcomes
are actually drawn from $P$. In the questions game, the KL divergence in bits is
the average number of extra questions that a strategy designed for $Q$ needs on
outcomes drawn from $P$, up to rounding to whole questions. Let $P$ and $Q$ have
probability mass functions (or densities) $p$ and $q$. The KL divergence from
$P$ to $Q$ is

$$
D_{\textrm{KL}}(P\|Q) = \mathbb{E}_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].
$$
:eqlabel:`eq_mdl-kl_def`

The quantity inside the expectation is
$\log p(x) - \log q(x) = (-\log q(x)) - (-\log p(x))$, which is the surprisal of
the believer in $Q$ minus the surprisal of the believer in $P$. It is positive
where $Q$ underestimates an outcome ($q(x) < p(x)$) and negative where $Q$
overestimates it. The KL divergence averages this difference under the true
distribution $P$.

Because $P$ supplies both the samples and the numerator, the definition is
asymmetric: in general, $D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P)$.
Outcomes with $p(x) = 0$ contribute nothing, following the convention
$0 \log(0/q) = 0$. If any outcome has $p(x) > 0$ while $q(x) = 0$, however, then
$D_{\textrm{KL}}(P\|Q) = +\infty$.

A small numerical example makes the definition concrete, and we will carry it
through the rest of the section. Let $P = (0.6, 0.3, 0.1)$ be the truth and
$Q = (0.2, 0.5, 0.3)$ a model on three outcomes. Then

$$
D_{\textrm{KL}}(P\|Q)
= 0.6 \ln\frac{0.6}{0.2} + 0.3 \ln\frac{0.3}{0.5} + 0.1 \ln\frac{0.1}{0.3}
= 0.659 - 0.153 - 0.110 = 0.396 \textrm{ nats},
$$

whereas the reverse direction gives $D_{\textrm{KL}}(Q\|P) = 0.365$ nats. In
bits, the two directions are $0.571$ and $0.527$. Not only do they differ, but
the first value tells us that a questioner who believes $Q$ asks $0.57$ more
questions per outcome than necessary, averaged over long runs of outcomes drawn
from $P$.

Here is the discrete KL divergence in code, implementing $0 \log 0 = 0$ by
masking out the zero probabilities of $P$, just as we did in `entropy`. We
evaluate it on our running pair in both directions, as well as on a model
$Q' = (0.5, 0.5, 0)$ that declares the third outcome impossible.

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

The second line is the one to remember. The model $Q'$ assigns zero probability
to an outcome that occurs one time in ten, and its KL divergence is infinite: a
model that declares an event impossible is infinitely surprised when that event
happens.

### Gibbs' Inequality

Could the average extra surprisal ever be negative? If it could, a wrong model
would be less surprised on average than the truth itself. Gibbs' inequality
:cite:`Gibbs.1902` rules that out: no model beats the truth on average. Because
everything that follows rests on this result, we prove it in full as a
consequence of averaging a concave function. For any concave function $f$ and any
random variable $X$ with finite mean, *Jensen's inequality* states that

$$
\mathbb{E}[f(X)] \le f(\mathbb{E}[X]).
$$

In words, averaging first and applying $f$ second yields the larger value. When
$X$ takes two values, this is simply the statement that the chord lies below the
curve in :numref:`fig_mdl-bernoulli-entropy`. If $f$ is strictly concave,
equality holds only when $X$ is constant with probability one
(:numref:`subsec_mdl-jensen` proves the general case). We need Jensen's
inequality for just one function, the logarithm, which is strictly concave: the
logarithm of an average is at least the average of the logarithms.

**Proposition (Gibbs' inequality).** *For any distributions $P$ and $Q$ on
the same space,*

$$
D_{\textrm{KL}}(P\|Q) \geq 0,
$$

*with equality if and only if $P = Q$.*

**Proof.** If $q(x) = 0$ for some $x$ with $p(x) > 0$, then the divergence is
$+\infty$ and there is nothing to prove, so assume otherwise. Rewrite the
divergence as minus the average of $\log(q(x)/p(x))$ under $x \sim P$, and apply
Jensen's inequality to the logarithm:

$$
\begin{aligned}
D_{\textrm{KL}}(P\|Q)
&= -\mathbb{E}_{x\sim P}\!\left[\log\frac{q(x)}{p(x)}\right]
\geq -\log \mathbb{E}_{x\sim P}\!\left[\frac{q(x)}{p(x)}\right] \\
&= -\log \!\!\sum_{x\,:\,p(x)>0}\!\! q(x)
= -\log Q(\textrm{supp}\,P)
\geq 0.
\end{aligned}
$$

The leading minus sign flips Jensen's "at most" into "at least". The rest of the
proof happens in the second line. The expectation runs over the support of $P$,
the set of outcomes with $p(x) > 0$. The factors $p(x)$ cancel, leaving the
total $Q$-probability of that support, a number at most $1$, whose negative
logarithm is therefore non-negative. For continuous densities, the same steps
hold with integrals in place of sums. For equality to hold, both inequalities
must be tight. Because the logarithm is strictly concave, the first is an
equality only when $q(x)/p(x)$ equals a constant $c$ for $P$-almost every $x$
(that is, outside a set of $P$-probability zero), and the second requires
$Q(\textrm{supp}\,P) = 1$. Summing (or integrating) $q = c\,p$ over the support
of $P$ then forces $c = 1$, so $q = p$ (again outside a set of probability zero
for densities). $\blacksquare$

The proof rests on a single idea. Averaged over the truth, the ratio $q/p$ is at
most $1$, so its logarithm averages to at most $0$, with equality only when
$q/p$ does not vary at all (meaning $q = p$). A model incurs no extra surprisal
only when it is right. Exercise :exref:`ex_mdl-information-theory-gibbs-tangent`
compares this proof with the tangent-line proof.

Gibbs' inequality immediately settles the uniform upper bound we left unproved.
For any distribution $P$ on $k$ outcomes and the uniform distribution $U$ on the
same set, $D_{\textrm{KL}}(P\|U) = \sum_i p_i \log(k\,p_i) = \log k - H(P)$,
which is non-negative and vanishes only when $P = U$.

It also proves that conditioning cannot increase entropy on average. Compare the
joint distribution of a pair $(X, Y)$ with the product of its marginals. The
log-ratio is $\log \frac{p(x, y)}{p(x)\,p(y)} = \log p(y \mid x) - \log p(y)$,
whose expectation under the joint distribution is $H(Y) - H(Y \mid X)$. Because
that expectation is a KL divergence,

$$
H(Y) - H(Y \mid X) = D_{\textrm{KL}}(p_{X,Y} \,\|\, p_X\, p_Y) \ge 0,
$$

with equality if and only if $X$ and $Y$ are independent. On average, knowing
$X$ never makes $Y$ more uncertain. A *particular* observation, however, can.
Suppose $Y$ is tails for certain when $X = 0$, and a fair coin when $X = 1$,
where $X = 1$ occurs one time in five. Overall, $Y$ lands heads with probability
$0.1$, so $H(Y) = H_b(0.1) \approx 0.325$ nats. Observing $X = 1$ raises our
uncertainty about $Y$ to $\ln 2 \approx 0.693$ nats, yet the conditional entropy
averages over both outcomes of $X$, giving
$H(Y \mid X) = 0.2 \cdot 0.693 \approx 0.139$ nats, less than half of $H(Y)$.

### Cross-Entropy

Classifiers are trained with a close relative of the KL divergence. The
*cross-entropy* from $P$ to $Q$ is the average surprisal of a believer in $Q$ on
outcomes drawn from $P$,

$$
\textrm{CE}(P, Q) = -\mathbb{E}_{x \sim P} [\log q(x)].
$$
:eqlabel:`eq_mdl-ce_def`

Adding and subtracting $\log p(x)$ inside the expectation, which is valid
whenever $H(P)$ is finite, splits the model's surprisal into the truth's own
entropy plus the excess divergence:

$$
\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P\|Q).
$$
:eqlabel:`eq_mdl-ce_decomp`

Cross-entropy is therefore an irreducible entropy floor plus a non-negative
excess: by Gibbs' inequality, $\textrm{CE}(P, Q) \ge H(P)$, with equality if and
only if $Q = P$. For our running pair,
$\textrm{CE}(P, Q) = -(0.6 \ln 0.2 + 0.3 \ln 0.5 + 0.1 \ln 0.3) = 1.294$ nats
and $H(P) = 0.898$ nats; their difference, $0.396$ nats, is the KL divergence we
computed earlier. Because $H(P)$ does not depend on $Q$, minimizing the
cross-entropy with respect to $Q$ is equivalent to minimizing the KL divergence.
Maximum likelihood on discrete data does precisely this: its average negative
log-likelihood is the cross-entropy from the empirical data distribution to the
model (:numref:`subsec_mdl-nll-crossentropy`).

:numref:`fig_mdl-code-length-bars` illustrates this decomposition in the
questions game for the dyadic source $P$ and the uniform model
$U = (\tfrac14, \tfrac14, \tfrac14, \tfrac14)$. Asking questions adapted to $P$
takes $H_2(P) = 1.75$ questions on average. Asking two fixed questions under the
uniform strategy takes $\textrm{CE}_2(P, U) = 2$ questions, so the waste is
$D_{\textrm{KL}}(P\|U) = 0.25$ bits. Since a questioning strategy is a binary
code, these same numbers are average codeword lengths in bits per symbol. (For
this particular pair, the reverse divergence $D_{\textrm{KL}}(U\|P)$ also
happens to equal $0.25$ bits, which is why we used a different pair earlier to
show that KL divergence is asymmetric.)

![The decomposition :eqref:`eq_mdl-ce_decomp` for the dyadic source $P = (\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \tfrac{1}{8})$ and the uniform model $U$, in bits. Asking questions adapted to $P$ takes $H_2(P) = 1.75$ questions on average, asking two fixed questions takes $\textrm{CE}_2(P, U) = 2.0$, and the waste is $D_{\textrm{KL}}(P\|U) = 0.25$ bits.](../img/mdl-it-code-length-bars.svg)
:label:`fig_mdl-code-length-bars`

## The Classification Loss

In a $k$-class classification problem, the target for a single example is the
one-hot distribution placing all its mass on the recorded label $y$, while the
model outputs a predicted distribution $\mathbf{q} = (q_1, \ldots, q_k)$. The
cross-entropy :eqref:`eq_mdl-ce_def` between them collapses to a single term,
$-\log q_y$, which is simply the model's surprisal at the observed label.
Averaged over $n$ examples with labels $y_i$ and predictions $\mathbf{q}_i$,
this gives the *cross-entropy loss*

$$
\textrm{CE} = -\frac{1}{n} \sum_{i=1}^n \log q_{i, y_i},
$$

measured in nats. Deep learning frameworks compute this loss directly from the
*logits* $\mathbf{z}$, the unnormalized scores whose softmax is $\mathbf{q}$.
They apply a numerically stable log-softmax internally, because forming the
probabilities first can overflow for large logits and underflow to $\log 0$ for
very negative ones (:numref:`subsec_softmax-implementation-revisited`;
:numref:`subsec_mdl-stable-softmax` derives the stable form). In the
subsections that follow, we unpack what this loss tells us about noise in the
data, probability calibration, and softened targets.

### The Irreducible Loss
:label:`subsec_mdl-irreducible-loss`

At the start of the section, we asked how low the loss can go when some labels
are wrong. On a fixed training set, the answer is uninformative: a network large
enough to memorize the dataset can fit every recorded label, the wrong ones
included, and drive the training loss toward zero. The question has real content
only on fresh examples that the model has not seen, and there the one-hot target
is not the full truth. In real data, the label is rarely a deterministic
function of the input: the same image may receive different labels due to
genuine ambiguity or annotation errors. The truth for an input $\mathbf{x}$ is
the conditional distribution from which its label is drawn, and the recorded
label is a single sample from it.

Suppose the input $\mathbf{x}$ and label $y \in \{1, \ldots, k\}$ follow a joint
distribution, where $p(\cdot \mid \mathbf{x})$ is the true conditional
distribution of the label and $q(\cdot \mid \mathbf{x})$ is any model's
prediction. As the number of fresh examples grows, their average loss converges
to its expectation over $(\mathbf{x}, y)$, which we call the *population loss*.
No model can push this population loss below the average entropy of the label
given the input.

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
\mathbf{x})$ for almost every $\mathbf{x}$ (that is, for all inputs outside a
set of probability zero).*

**Proof.** Condition on $\mathbf{x}$. The inner expectation over $y$ is the
cross-entropy $\textrm{CE}(p(\cdot \mid \mathbf{x}), q(\cdot \mid \mathbf{x}))$,
so the decomposition :eqref:`eq_mdl-ce_decomp` applies pointwise at every
$\mathbf{x}$ (all entropies are finite, at most $\log k$). Taking the outer
expectation over $\mathbf{x}$ gives the identity. By Gibbs' inequality, every
KL term is non-negative, and an average of non-negative terms vanishes if and
only if almost all of them are zero. $\blacksquare$

This floor is the conditional entropy of the label given the input
(:numref:`subsec_mdl-joint-conditional-entropy`). It is an intrinsic property of
the data: no architecture, extra training data, or longer training schedule can
lower it. Of all the results in this section, this is the one to keep in mind
whenever you read a loss curve. Three cases illustrate what it implies:

* A model that predicts the uniform distribution suffers a cross-entropy of
  $\ln k$ on every input regardless of the labels, giving the familiar $2.30$
  nats for $k = 10$.
* When the label is a deterministic function of the input, every
  $H(p(\cdot \mid \mathbf{x}))$ is zero, and a perfect model achieves a loss of
  $0$.
* Under symmetric label noise, the clean class is determined by the input, while
  the recorded label matches the clean class with probability $1 - \rho$ and
  each of the remaining $k - 1$ classes with probability $\rho/(k-1)$. At every
  input, the conditional entropy is
  $H = -(1-\rho)\log(1-\rho) - \rho\log\tfrac{\rho}{k-1} = H_b(\rho) + \rho\log(k-1)$.
  For $\rho = 0.1$ and $k = 10$, this floor is $0.325 + 0.220 = 0.545$ nats. As
  long as $1 - \rho > \rho/(k-1)$, predicting the clean class remains optimal,
  so the highest achievable accuracy is $1 - \rho$ (here $90\%$). Exercise
  :exref:`ex_mdl-information-theory-noise-ambiguous-input` extends this to
  inputs whose clean class is itself uncertain.

Before running the next cell, pause and guess the loss of a confident predictor
that assigns probability $0.99$ to the clean class in this noisy setting. It is
right nine times out of ten, and nearly certain every time. The code compares it
against the uniform predictor and the *Bayes predictor*, which reports the true
conditional distribution $q = p$.

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

The overconfident predictor suffers a loss of $0.689$, well above the floor:
every flipped label catches it by surprise, and its high confidence costs far
more on the tenth of labels that are flipped than it gains on the rest. Only the
Bayes predictor reaches the floor.

We can now answer the question from the opening: is a final loss of $0.6$ good?
On clean labels, where the floor is zero, $0.6$ still leaves plenty of room for
a better model. On these noisy labels, however, it sits within $0.06$ nats of
the floor and leaves almost none. A plateau at the floor is simply a property of
the data; only a plateau above the floor points to a limitation of the model or
its training. Bear in mind that all of this applies to fresh data. On a
held-out split, a loss *below* the floor cannot come from a better model, and
Exercise :exref:`ex_mdl-information-theory-label-noise-floor` asks what it
signals instead.

So far we derived the loss floor for a single penalty ($-\log q_y$), for
unmodified one-hot labels, and for countable outcomes. In the remainder of this
section we relax each of these three assumptions in turn, examining other
penalties, softened targets, and continuous real-valued outcomes.

### Proper Scoring Rules

The loss floor is achieved only by the true conditional probabilities, which is
just the property we want if we intend to interpret a classifier's outputs as
probabilities. Decision theory gives this property a name. A *scoring rule*
assigns a penalty $S(\mathbf{q}, y)$ for reporting the probability vector
$\mathbf{q}$ when class $y$ actually occurs. The rule is *proper* if truthful
reporting is optimal: when labels follow $\mathbf{p}$, the expected
penalty $\mathbb{E}_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$ is minimized by
reporting $\mathbf{q} = \mathbf{p}$. It is *strictly proper* if this truthful
report is the unique minimizer. The cross-entropy loss is the *log score*
$S(\mathbf{q}, y) = -\log q_y$, and Gibbs' inequality shows that it is strictly
proper: its expected penalty is $\textrm{CE}(\mathbf{p}, \mathbf{q})$, which is
uniquely minimized at $\mathbf{q} = \mathbf{p}$.

Consequently, at the population optimum over a model class rich enough to
contain the truth, a classifier trained with cross-entropy outputs the true
conditional probabilities. In particular, its predictions are *calibrated*:
among all inputs for which the model predicts probability $0.8$ for a given
class, that class actually occurs $80\%$ of the time. (A finite training set or
a restricted model class can still leave a trained network miscalibrated in
practice.)

The log score is not the only strictly proper scoring rule. The *Brier score*,
$S(\mathbf{q}, y) = \sum_j (q_j - \mathbf{1}(j = y))^2$, is strictly proper as
well, and unlike the log score it imposes a bounded penalty on confident errors
where $-\log q_y$ diverges :cite:`Gneiting.Raftery.2007`. Exercise
:exref:`ex_mdl-information-theory-scoring-rule-is-strictly` computes its excess
penalty and examines a seemingly reasonable rule that turns out to be improper.

### Label Smoothing

Until now, our training target has been the one-hot label. Replacing it with a
softer target changes both what the loss asks of the logits and where its floor
lies, and there is a practical reason to do so. One-hot targets demand something
a softmax layer cannot actually produce: assigning probability $1$ to the true
class requires its logit to be infinitely larger than all the others. As a
result, the loss has no finite minimizer, and gradient descent keeps pushing the
logits further apart for as long as training runs. *Label smoothing*
:cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` fixes this by mixing the one-hot target
$\mathbf{e}_y$ with a fraction $\epsilon$ of the uniform distribution over all
$k$ classes:

$$
\mathbf{p}^\epsilon = (1 - \epsilon)\, \mathbf{e}_y + \frac{\epsilon}{k}\, \mathbf{1}.
$$
:eqlabel:`eq_mdl-label-smoothing-target`

Under this target, the true class receives probability $1 - \epsilon +
\epsilon/k$ and every other class receives $\epsilon/k$. The training objective
is the ordinary cross-entropy $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$
against this smoothed target (the same loss used in
:numref:`sec_training_recipes`). It is minimized at
$\mathbf{q} = \mathbf{p}^\epsilon$, which a softmax achieves with a finite gap
between the logit of the true class and each competing logit:

$$
z_y - z_j = \log\frac{1 - \epsilon + \epsilon/k}{\epsilon/k}
= \log\!\Big(1 + \frac{k(1-\epsilon)}{\epsilon}\Big).
$$

For $k = 10$ classes and $\epsilon = 0.1$, this optimal logit gap is
$\ln 91 \approx 4.51$, and the loss at the optimum equals the entropy of the
smoothed target, $H(\mathbf{p}^\epsilon) = 0.500$ nats.

It is instructive to see how quickly the logits drift without smoothing.
Consider a single example whose logits start at zero, updated by gradient
descent with step size $1$. The gradient of the cross-entropy with respect to
the logits is $\mathbf{q} - \mathbf{p}^\epsilon$. With a one-hot target, all
wrong classes stay tied, and each step widens the gap $g$ between the true
class and the rest by $(1 - q_y) + q_j = k/(e^g + k - 1)$, which is
approximately $k e^{-g}$ once $g$ is large. Incrementing $g$ by $k e^{-g}$
multiplies $e^g$ by $e^{k e^{-g}} \approx 1 + k e^{-g}$, thereby adding roughly
$k$ to $e^g$ on each step. After $t$ steps we therefore expect $e^g \approx kt$,
or $g \approx \ln(kt)$. The gap never stops growing, though its growth is
logarithmic: ten times as many steps add only $\ln 10 \approx 2.30$. After
$10{,}000$ steps, we expect $g \approx \ln 10^5 \approx 11.51$. The next cell
runs this gradient descent both without and with smoothing.

```{.python .input #information-theory-label-smoothing}
k, checkpoints = 10, (10, 100, 1000, 10000)
for eps in (0.0, 0.1):
    p_eps = np.full(k, eps / k)
    p_eps[0] += 1 - eps                   # the true class first
    z = np.zeros(k)                       # the logits of one example
    for step in range(1, checkpoints[-1] + 1):
        q = np.exp(z - z.max())
        z -= q / q.sum() - p_eps          # gradient step, step size 1
        if step in checkpoints:
            log_q = z - z.max() - np.log(np.exp(z - z.max()).sum())
            print(f'eps = {eps}  step {step:5d}  gap = {z[0] - z[1]:5.2f}  '
                  f'loss = {-p_eps @ log_q:.4f}')
print(f'optimum for eps = 0.1: gap = ln 91 = {math.log(91):.2f}, '
      f'loss = H(p_eps) = {entropy(p_eps):.4f}')
```

With one-hot targets, the logit gap reaches $4.29$, $6.87$, $9.21$, and $11.51$.
Our approximation $\ln(kt)$ predicts $4.61$, $6.91$, $9.21$, and $11.51$: the
actual gap starts slightly below the estimate (where $k e^{-g}$ is still rough)
and agrees to two decimal places from $1{,}000$ steps onward. Meanwhile, the
loss keeps inching downward without ever reaching zero. In a neural network
large enough to fit its training set, this unbounded drift manifests as weights
that keep growing and predicted probabilities near $1$ on every training
example, mislabeled ones included. With $\epsilon = 0.1$, by contrast, the gap
settles at $4.51$ by step $100$ and stays there, while the loss comes to rest at
$0.500$.

Training now has a finite optimum, though a smoothed model must be interpreted
with care. Even when it predicts the smoothed target without error, its training
loss plateaus at $H(\mathbf{p}^\epsilon)$ rather than zero. Moreover, its
outputs are no longer the true conditional probabilities: no class can receive
probability greater than $1 - \epsilon + \epsilon/k$ (here $0.91$). In practice,
$\epsilon = 0.1$ is the standard choice, and the training recipe of
:numref:`sec_training_recipes` adopts it. Exercise
:exref:`ex_mdl-information-theory-label-smoothing` explores what label smoothing
does to the logits of the incorrect classes.

### Knowledge Distillation
:label:`subsec_mdl-distillation`

Soft targets can also come from another trained model. *Knowledge distillation*
:cite:`Hinton.Vinyals.Dean.2015` trains a compact *student* network to imitate a
larger *teacher* network, an approach applied to language models in
:numref:`sec_lm_distillation`. Instead of training on a one-hot label, the
student trains on the teacher's full probability distribution over classes. The
fact that a handwritten "2" resembles a "7" much more than it resembles a "0" is
information a one-hot label discards; it lives in the teacher's logits for the
wrong classes. (Exercise :exref:`ex_mdl-information-theory-label-smoothing`
asks why a teacher trained with label smoothing passes on less of this
information.)

As with any fixed target distribution, the loss is minimized when the student
reproduces the target itself, so the decomposition :eqref:`eq_mdl-ce_decomp`
adds nothing new here. What does call for explanation is the scale of the loss,
which carries a factor that looks arbitrary at first glance.

Because a well-trained teacher's softmax outputs are nearly one-hot, the
probabilities it assigns to wrong classes are usually too tiny to exert much
pull on the student. We therefore *soften* both distributions first by dividing
the logits by a *temperature* $\tau > 0$ before applying the softmax:

$$
p_j^{(\tau)} = \mathrm{softmax}(\mathbf{z}^{\textrm{tea}}/\tau)_j, \qquad
q_j^{(\tau)} = \mathrm{softmax}(\mathbf{z}^{\textrm{stu}}/\tau)_j.
$$

Setting $\tau > 1$ compresses the gaps between logits and shifts probability
mass toward the wrong classes. The distillation loss is the KL divergence
between these softened distributions, multiplied by $\tau^2$:

$$
\mathcal{L}_{\textrm{distill}}
= \tau^2\, D_{\textrm{KL}}\big(\mathbf{p}^{(\tau)} \,\|\, \mathbf{q}^{(\tau)}\big).
$$
:eqlabel:`eq_mdl-distill-loss`

This loss vanishes if and only if the student matches the softened teacher. To
see why the factor $\tau^2$ is needed, let us differentiate the unscaled KL term
with respect to a student logit. Only the cross-entropy part
$-\sum_j p_j^{(\tau)} \log q_j^{(\tau)}$ depends on the student, and the
derivative of a log-softmax at temperature $\tau$ is the familiar
softmax-minus-target expression scaled by $1/\tau$:

$$
\frac{\partial}{\partial z^{\textrm{stu}}_j}
D_{\textrm{KL}}\big(\mathbf{p}^{(\tau)} \| \mathbf{q}^{(\tau)}\big)
= \frac{1}{\tau}\big(q_j^{(\tau)} - p_j^{(\tau)}\big).
$$

Each student logit is thus nudged by the difference between the student's and
teacher's softened probabilities for that class, divided by $\tau$.

At high temperatures, that probability difference shrinks too. Subtracting the
mean logit $\bar z$ leaves the softmax unchanged, and for large $\tau$ we can
expand $e^{(z_j - \bar z)/\tau} \approx 1 + (z_j - \bar z)/\tau$ to first order.
Because the centered deviations $z_j - \bar z$ sum to zero, the softmax
denominator is approximately $k$, giving
$\mathrm{softmax}(\mathbf{z}/\tau)_j \approx \tfrac{1}{k}\big(1 + (z_j - \bar z)/\tau\big)$.
Consequently, $q_j^{(\tau)} - p_j^{(\tau)}$ shrinks like $1/\tau$, and the
unscaled gradient shrinks like $1/\tau^2$:

$$
\frac{1}{\tau}\big(q_j^{(\tau)} - p_j^{(\tau)}\big)
\;\approx\; \frac{1}{k\,\tau^2}\Big(\big(z^{\textrm{stu}}_j - \bar z^{\textrm{stu}}\big)
- \big(z^{\textrm{tea}}_j - \bar z^{\textrm{tea}}\big)\Big).
$$

Multiplying the loss by $\tau^2$ cancels this decay and keeps the gradient
magnitude roughly constant as $\tau$ varies. That stability is essential in
practice because the distillation loss is rarely used in isolation: it is
typically added, with a fixed weight, to the standard cross-entropy loss on the
hard labels. Without the $\tau^2$ factor, tuning $\tau$ would inadvertently
change the relative weight of the two losses. With it, the temperature controls
only how much of the teacher's small wrong-class probabilities the student sees.

Let us verify this scaling numerically using teacher logits $(5, 2, -1)$ and
student logits $(3, 3, 0)$. Both vectors have mean $2$, so $\tau^2$ times the
gradient should converge to
$\tfrac13\big((1, 1, -2) - (3, 0, -3)\big) = (-0.667, 0.333, 0.333)$, which has
norm $0.816$. In the cell below, we use automatic differentiation to compute the
gradient of the unscaled KL term across five temperatures and compare its norm
before and after multiplying by $\tau^2$.

```{.python .input #information-theory-distillation}
#@tab mxnet
z_tea = mnp.array([5.0, 2.0, -1.0])
z_stu = mnp.array([3.0, 3.0, 0.0])
z_stu.attach_grad()                   # every backward() overwrites the grad
print(' tau  |grad|  |tau^2 grad|  tau^2 grad')
for tau in (1.0, 2.0, 5.0, 10.0, 20.0):
    with autograd.record():
        p_tau = npx.softmax(z_tea / tau)
        kl = (p_tau * (mnp.log(p_tau) - npx.log_softmax(z_stu / tau))).sum()
    kl.backward()
    g = z_stu.grad.asnumpy()
    print(f'{tau:4.0f}  {np.linalg.norm(g):.4f}  '
          f'{np.linalg.norm(tau**2 * g):12.3f}  {np.round(tau**2 * g, 3)}')
limit = (z_stu - z_stu.mean()) - (z_tea - z_tea.mean())
limit = limit.asnumpy() / len(limit)
print(f'limit {"":6s}  {np.linalg.norm(limit):12.3f}  {np.round(limit, 3)}')
```

```{.python .input #information-theory-distillation}
#@tab pytorch
z_tea = torch.tensor([5.0, 2.0, -1.0])
z_stu = torch.tensor([3.0, 3.0, 0.0], requires_grad=True)
print(' tau  |grad|  |tau^2 grad|  tau^2 grad')
for tau in (1.0, 2.0, 5.0, 10.0, 20.0):
    p_tau = torch.softmax(z_tea / tau, 0)
    kl = (p_tau * (torch.log(p_tau) - torch.log_softmax(z_stu / tau, 0))).sum()
    g = torch.autograd.grad(kl, z_stu)[0].numpy()
    print(f'{tau:4.0f}  {np.linalg.norm(g):.4f}  '
          f'{np.linalg.norm(tau**2 * g):12.3f}  {np.round(tau**2 * g, 3)}')
limit = (z_stu - z_stu.mean()) - (z_tea - z_tea.mean())
limit = limit.detach().numpy() / len(limit)
print(f'limit {"":6s}  {np.linalg.norm(limit):12.3f}  {np.round(limit, 3)}')
```

```{.python .input #information-theory-distillation}
#@tab tensorflow
z_tea = tf.constant([5.0, 2.0, -1.0])
z_stu = tf.Variable([3.0, 3.0, 0.0])
print(' tau  |grad|  |tau^2 grad|  tau^2 grad')
for tau in (1.0, 2.0, 5.0, 10.0, 20.0):
    with tf.GradientTape() as tape:
        p_tau = tf.nn.softmax(z_tea / tau)
        kl = tf.reduce_sum(
            p_tau * (tf.math.log(p_tau) - tf.nn.log_softmax(z_stu / tau)))
    g = tape.gradient(kl, z_stu).numpy()
    print(f'{tau:4.0f}  {np.linalg.norm(g):.4f}  '
          f'{np.linalg.norm(tau**2 * g):12.3f}  {np.round(tau**2 * g, 3)}')
limit = (z_stu - tf.reduce_mean(z_stu)) - (z_tea - tf.reduce_mean(z_tea))
limit = limit.numpy() / len(limit)
print(f'limit {"":6s}  {np.linalg.norm(limit):12.3f}  {np.round(limit, 3)}')
```

```{.python .input #information-theory-distillation}
#@tab jax
z_tea = jnp.array([5.0, 2.0, -1.0])
z_stu = jnp.array([3.0, 3.0, 0.0])

def distill_kl(z_stu, z_tea, tau):
    p_tau = jax.nn.softmax(z_tea / tau)
    return (p_tau * (jnp.log(p_tau) - jax.nn.log_softmax(z_stu / tau))).sum()

print(' tau  |grad|  |tau^2 grad|  tau^2 grad')
for tau in (1.0, 2.0, 5.0, 10.0, 20.0):
    g = np.asarray(jax.grad(distill_kl)(z_stu, z_tea, tau))
    print(f'{tau:4.0f}  {np.linalg.norm(g):.4f}  '
          f'{np.linalg.norm(tau**2 * g):12.3f}  {np.round(tau**2 * g, 3)}')
limit = (z_stu - z_stu.mean()) - (z_tea - z_tea.mean())
limit = np.asarray(limit) / len(limit)
print(f'limit {"":6s}  {np.linalg.norm(limit):12.3f}  {np.round(limit, 3)}')
```

Without the $\tau^2$ factor, the gradient collapses: its norm drops from $0.64$
at $\tau = 1$ to $0.0021$ at $\tau = 20$, roughly three hundred times smaller,
and the final doubling of $\tau$ divides it by about four, just as $1/\tau^2$
predicts. With the factor included, the norm stays between $0.64$ and $0.92$. It
slightly overshoots the asymptotic limit $0.816$, peaks near $\tau = 5$, and
then settles back slowly at a rate of order $1/\tau$ (still reading $0.854$ at
$\tau = 20$). Notice also that at $\tau = 1$ the loss reduces to the ordinary KL
divergence, and for a one-hot teacher (whose entropy is zero) it becomes the
ordinary cross-entropy loss, so standard supervised training is a special case.

## Entropy and KL Divergence for Densities

So far every outcome has been discrete and countable. A regression target, by
contrast, is a real number described by a probability density, so sums become
integrals. The KL divergence survives this transition intact, whereas entropy
does not: continuous entropy can become negative and depends on the units in
which the variable is measured. Consequently, the negative log-likelihood loss
of a continuous model can fall below zero, and its floor shifts whenever we
rescale the data. What retains its absolute meaning is the excess above the
floor, and with it the difference in loss between any two models evaluated on
the same data. (On a first reading, you may skip these three subsections and
resume at the final paragraph of the section.)

### Differential Entropy
:label:`subsec_mdl-differential-entropy`

For a continuous random variable $X$ with density $p$, replacing the sum in the
discrete entropy with an integral gives the *differential entropy*:

$$
h(X) = -\int p(x) \log p(x) \,dx.
$$
:eqlabel:`eq_mdl-diff-ent-def`

This is the expected negative log-density, and density values are not
probabilities: they carry units of $1/x$ and can easily exceed $1$. We use a
lowercase $h$ to distinguish differential entropy from the discrete $H$, because
the two behave quite differently.

The simplest example already shows why. The uniform density on $[0, a]$ equals
$1/a$ throughout the interval, so
$h = -\int_0^a \tfrac1a \log \tfrac1a \,dx = \log a$. When $a = \tfrac12$, this
gives $-\ln 2 \approx -0.693$ nats, a negative entropy that no discrete
distribution can ever have. Worse yet, measuring the very same length in
centimetres instead of metres multiplies $a$ by $100$ and adds
$\ln 100 \approx 4.61$ nats to $h$, even though the underlying uncertainty has
not changed at all.

The same kind of shift happens under any smooth change of variables. If
$y = g(x)$ is invertible with derivative $g'$, then the density of $Y$ at $g(x)$
is $p_X(x)/\lvert g'(x)\rvert$, and therefore

$$
h(Y) = h(X) + \mathbb{E}\big[\log\lvert g'(X)\rvert\big].
$$

For a random vector $\mathbf{x} \in \mathbb{R}^n$, both the definition and this
transformation rule keep the exact same form, with an integral over
$\mathbb{R}^n$ and the Jacobian determinant
$\lvert\det \partial g/\partial \mathbf{x}\rvert$ replacing $\lvert g'\rvert$.
Stretching a variable increases its differential entropy, while compressing it
drives the entropy down without bound. As we will derive in
:numref:`subsec_mdl-gaussian-max-entropy`, a Gaussian with standard deviation
$\sigma$ has differential entropy $h = \tfrac12 \log(2\pi e \sigma^2)$, which
turns negative as soon as $\sigma < (2\pi e)^{-1/2} \approx 0.24$.

Fortunately, the KL divergence does not share this sensitivity to units. If we
apply the same change of variables to both the data density $p$ and a model
density $q$, both are divided by the same Jacobian factor, which cancels inside
the ratio $p/q$. The KL divergence is therefore invariant to invertible
reparameterizations. For two uniform densities $p_1$ on $[0, a_1]$ and $p_2$ on
$[0, a_2]$ with $a_1 \le a_2$, for example,
$\int p_1 \log(p_1/p_2)\,dx = \log(a_2/a_1)$: for $a_1 = \tfrac12$ and
$a_2 = 1$, this is $\ln 2 \approx 0.693$ nats whether measured in metres or in
centimetres. The cross-entropy $-\mathbb{E}_P[\log q]$ used to train a
continuous model shifts by the same constant as $h$, and since that constant
does not depend on the model, both the gradients and the minimizer remain
unchanged. Whenever $h(p)$ is finite, the decomposition
:eqref:`eq_mdl-ce_decomp` still holds with $h$ in place of $H$, so maximum
likelihood for densities minimizes $D_{\textrm{KL}}(p\|q_{\boldsymbol\theta})$
at the population level.

Note also that differential entropy is not simply the limit of discrete entropy
under fine binning. Partition the real line into cells of width $\Delta$, and
let $X_\Delta$ record which cell contains $X$. For a point $x_i$ inside cell
$i$, $P(X_\Delta = i) \approx p(x_i)\Delta$, and

$$
H(X_\Delta) + \log \Delta \to h(X) \quad \textrm{as } \Delta \to 0
$$

whenever $p$ is regular enough for the Riemann sums to converge to the integrals
(for instance, when $p$ is continuous and both $p$ and $p \log p$ are
Riemann-integrable). The discrete entropy $H(X_\Delta)$ itself diverges like
$\log(1/\Delta)$, because a finer grid always demands more yes/no questions,
and $h(X)$ is the finite offset that remains after subtracting that growth. For
the uniform
density on $[0, 1]$ with $\Delta = \tfrac18$, all eight cells are equally
likely, so $H_2(X_\Delta) = 3$ bits and $\log_2 \Delta = -3$, which sum to
$0 = h$. Exercise :exref:`ex_mdl-information-theory-quantized-entropy` proves
this limit and tests it numerically on a Gaussian.

### KL Divergence Between Gaussians

When a regression model predicts a real number along with its uncertainty, it
most often outputs a Gaussian: a mean and a variance. Comparing such a model
with
the data, or comparing two Gaussian models against each other, requires the KL
divergence between two Gaussians, which has a clean closed form. For
$P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$,

$$
D_{\textrm{KL}}\!\left(\mathcal{N}(\mu_1,\sigma_1^2) \,\big\|\, \mathcal{N}(\mu_2,\sigma_2^2)\right)
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}.
$$
:eqlabel:`eq_mdl-gaussian_kl`

The formula combines a log ratio of standard deviations with the variance and
squared mean error of $P$, both measured relative to the model's variance
$\sigma_2^2$. As Gibbs' inequality guarantees, it is non-negative and vanishes
only when $\mu_1 = \mu_2$ and $\sigma_1 = \sigma_2$. It is also visibly
asymmetric, because $\sigma_1$ and $\sigma_2$ enter in different ways. Exercise
:exref:`ex_mdl-information-theory-gaussian-kl` derives this expression directly
from the definition.

Start with the simplest case: a unit mean shift with equal variances,
$P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(1, 1)$. Here the formula reduces to
$(\mu_1 - \mu_2)^2/(2\sigma^2) = \tfrac12$ nat in either direction. Now let the
variances differ instead: take $P = \mathcal{N}(0, 1)$ and
$Q = \mathcal{N}(0, 4)$, so $\sigma_2 = 2$. Plugging in gives
$D_{\textrm{KL}}(P\|Q) = \ln 2 + \tfrac18 - \tfrac12 \approx 0.318$ nats,
whereas the reverse direction gives
$D_{\textrm{KL}}(Q\|P) = -\ln 2 + 2 - \tfrac12 \approx 0.807$ nats, more than
$2.5$ times larger.

:numref:`fig_mdl-kl-integrand` reveals where each divergence comes from. In both
directions, the integrand changes sign at $|x| = (8 \ln 2/3)^{1/2} \approx 1.36$,
where the two densities cross. In panel (a), the wide model explains narrow
data: it is too surprised in the core ($|x| < 1.36$, contributing $+0.424$) and
slightly less surprised than the truth in the tails (contributing $-0.106$). In
panel (b), the narrow model must explain wide data: the core contributes
$-0.240$, while the tails, where the narrow model assigns tiny probability to
outcomes that actually occur, contribute a hefty $+1.046$.

![Where the KL divergence between $\mathcal{N}(0,1)$ and $\mathcal{N}(0,4)$ comes from. (a) The narrow distribution explained by the wide model: a positive core and slightly negative tails sum to $0.318$ nats. (b) The wide distribution explained by the narrow model: the tails, where the narrow model assigns little probability, contribute $1.05$ nats against a core of $-0.24$, for a total of $0.807$. The sign changes at $|x| = 1.36$ in both directions.](../img/mdl-it-kl-integrand.svg)
:label:`fig_mdl-kl-integrand`

A narrow model is far more surprised by wide data than a wide model is by narrow
data. In general, KL divergence penalizes a model severely whenever outcomes
occur that it had nearly ruled out, but stays mild when the model merely spreads
extra probability over regions where no data appear. What this asymmetry means
for fitting generative models in one direction versus the other is the subject
of :numref:`sec_mdl-fwd-vs-rev-kl`.

The definition also suggests a general way to estimate a KL divergence when no
closed form exists: draw samples from $P$ and average $\log p(x) - \log q(x)$.
Because we will rely on sample-based estimates later, it is worth checking how
accurate they are on this Gaussian pair. The next cell implements both the
closed form and the Monte Carlo estimator together with its standard error.

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
    """Monte Carlo estimate of the same KL from n samples x ~ P, with its
    standard error."""
    x = np.random.default_rng().normal(loc=mu1, scale=sigma1, size=n)
    r = gaussian_log_pdf(x, mu1, sigma1) - gaussian_log_pdf(x, mu2, sigma2)
    return float(r.mean()), float(r.std() / math.sqrt(n))
```

For the mean shift, we can predict the estimator's accuracy in advance. The
log-ratio is $\log p(x) - \log q(x) = \tfrac12 - x$, which has variance $1$
under $P$, so the average of $n = 2 \cdot 10^5$ samples has standard error
$1/\sqrt{n} = 0.0022$.

For the pair with different variances, however, the two directions are not
equally easy to estimate. Up to sign, both average the same function,
$\ln 2 - \tfrac38 x^2$, but one samples from the narrow Gaussian while the other
samples from the wide one, where $x^2$ fluctuates four times as much. Let us
estimate all three divergences and check.

```{.python .input #information-theory-example-2}
pairs = {'N(0,1) || N(1,1)': (0.0, 1.0, 1.0, 1.0),
         'N(0,1) || N(0,4)': (0.0, 1.0, 0.0, 2.0),
         'N(0,4) || N(0,1)': (0.0, 2.0, 0.0, 1.0)}
print(f'{"P || Q":18s} {"closed":>7s} {"sampled":>8s} {"std err":>8s}')
for name, args in pairs.items():
    estimate, std_err = mc_kl(*args)
    print(f'{name:18s} {gaussian_kl(*args):7.4f} {estimate:8.4f} {std_err:8.4f}')
```

The standard errors come out to $0.0022$, $0.0012$, and $0.0047$, and every
estimate lands within a few standard errors of the closed form. Notice that the
direction sampling from the wide Gaussian is four times noisier than its
reverse, meaning it requires sixteen times as many samples to reach the same
accuracy. Panel (b) of :numref:`fig_mdl-kl-integrand` explains why: most of
this divergence comes from the tails, where the log-ratio is large and samples
are sparse. We will encounter this same sampling difficulty at a much larger
scale in :numref:`sec_mdl-mi-estimation`.

### Maximum Entropy and the Gaussian
:label:`subsec_mdl-gaussian-max-entropy`

On a finite set, the uniform distribution maximizes uncertainty. Which density
on the real line plays the same role? Without any constraint, none does: no
density can be uniform on all of $\mathbb{R}$, and stretching a density raises
its differential entropy without limit. Suppose instead that we fix the two
moments a Gaussian model reports, the mean $\mu$ and variance $\sigma^2$. Among
all densities with those two moments, the Gaussian has the largest differential
entropy. In this precise sense, a Gaussian noise model makes the fewest
assumptions: it commits to a mean and a variance, and to nothing else.

The proof mirrors our one-line argument for the uniform bound, with a single
twist. In the discrete case, $-\log u(x) = \log k$ was constant across all
outcomes, so every distribution $P$ had the same cross-entropy $\log k$ to the
uniform model. Here, let $\varphi$ denote the Gaussian density with mean $\mu$
and variance $\sigma^2$. Its surprisal
$-\log \varphi(x) = \tfrac{1}{2}\log(2\pi\sigma^2) + \tfrac{(x-\mu)^2}{2\sigma^2}$
is no longer constant, but it is a quadratic function of $x$, and the
expectation of a quadratic depends on $P$ solely through the two moments we
fixed. For every density $P$ with mean $\mu$ and variance $\sigma^2$,

$$
\textrm{CE}(P, \varphi)
= \tfrac{1}{2}\log(2\pi\sigma^2) + \frac{\mathbb{E}_P\big[(x-\mu)^2\big]}{2\sigma^2}
= \tfrac{1}{2}\log(2\pi\sigma^2) + \tfrac{1}{2}
= \tfrac{1}{2}\log(2\pi e \sigma^2).
$$

Remarkably, the cross-entropy to $\varphi$ does not depend on which such $P$
generated the data. Setting $P = \varphi$ reduces cross-entropy to entropy, so
this constant is the Gaussian's own differential entropy, the closed form we
quoted earlier. For any other density $P$ with the same moments, the
decomposition :eqref:`eq_mdl-ce_decomp` (with $h$ in place of $H$) and Gibbs'
inequality yield

$$
h(P) = \textrm{CE}(P, \varphi) - D_{\textrm{KL}}(P\|\varphi)
\leq \tfrac{1}{2}\log(2\pi e \sigma^2) = h(\mathcal{N}(\mu, \sigma^2)),
$$
:eqlabel:`eq_mdl-gaussian-max-entropy`

with equality if and only if $P = \mathcal{N}(\mu, \sigma^2)$. (If
$h(P) = -\infty$, the inequality holds trivially.) The gap in this bound,
$D_{\textrm{KL}}(P\|\varphi)$, also has a direct interpretation: it is the
excess loss incurred by the best possible Gaussian model when the data are not
Gaussian (Exercise :exref:`ex_mdl-information-theory-gaussian-excess`).

The uniform and Gaussian distributions are two instances of a single principle.
Whenever a candidate density's surprisal averages to the same constant under
every distribution satisfying the constraints, that candidate maximizes entropy
among all of them. :numref:`subsec_mdl-max-entropy-derivation` establishes this
result in full generality for constraints on the expectations of arbitrary
statistics, where the maximizing distributions form an exponential family. The
geometric distribution on the non-negative integers $\{0, 1, 2, \ldots\}$ is a
third example: its surprisal is affine in the outcome, so it maximizes entropy
among all distributions on the non-negative integers with a given mean.

Let us step back and see what remains open. Earlier, the discrete model $Q'$
assigned zero probability to an outcome that actually occurred, making its KL
divergence infinite; the narrow Gaussian of :numref:`fig_mdl-kl-integrand` came
close, accumulating most of its divergence in tails it had nearly ruled out. A
generative model that maps a few random numbers into an image places all of its
probability mass on a thin surface inside pixel space. If real data do not lie
on that exact surface, the KL divergence is infinite regardless of whether the
surface passes right next to the data or miles away.
:numref:`sec_mdl-divergences-distances` shows that every divergence built from
the density ratio $p/q$ shares this blindness, and
:numref:`sec_mdl-test-functions-transport` develops comparison measures that can
still tell near from far. A second question lies even closer at hand. Earlier we
counted yes/no questions and read them as binary codewords, but only for a
dyadic source whose surprisals were whole numbers of bits.
:numref:`sec_mdl-coding-compression` constructs codes for arbitrary sources and
asks how close a real compressor can get to the entropy.

## Summary

A cross-entropy loss measures a model's average surprisal on the data. It
decomposes into the entropy of the data, which no model can lower, plus the KL
divergence from the data to the model, which by Gibbs' inequality is never
negative and vanishes only for the true distribution. Any loss value must
therefore be judged against its irreducible floor, the conditional entropy of
the label given the input, which on fresh data only the true conditional
probabilities can reach. That floor is zero when the input uniquely determines
the label, and $H_b(\rho) + \rho \log(k-1)$ under symmetric label noise at rate
$\rho$.

In bits, the entropy of a finite source equals the average number of yes/no
questions needed to identify an outcome (to within one question) and is at most
$\log_2 k$ for $k$ outcomes. Softening the targets shifts the optimum: label
smoothing creates a finite optimal logit gap with a positive loss floor, while
knowledge distillation matches a softened teacher, with the factor $\tau^2$
keeping the gradient scale steady as the temperature changes. For continuous
densities, differential entropy can be negative and shifts with the units of
measurement, whereas both the KL divergence and the minimizer of the
cross-entropy remain invariant.

## Exercises

1. **Uniqueness of self-information.** ● Show that self-information is
   unique up to scale: if a strictly decreasing function $I(p)$ on $(0, 1]$
   satisfies $I(1) = 0$ and $I(p_1 p_2) = I(p_1) + I(p_2)$ for all
   $p_1, p_2 \in (0, 1]$, then $I(p) = -c \log p$ for some constant
   $c > 0$. Work with the function $g(t) = I(e^{-t})$ for $t \geq 0$.
   :exlabel:`ex_mdl-information-theory-show-that-self-information`

1. **Two proofs of Gibbs' inequality.** The text proves Gibbs' inequality
   with Jensen's inequality. :numref:`subsec_mdl-max-entropy-derivation`
   proves it from the tangent line $\ln u \le u - 1$.
    1. Show that the tangent-line proof, with $u = q(x)/p(x)$ on the support
       of $P$, gives $D_{\textrm{KL}}(P\|Q) \ge 1 - Q(\textrm{supp}\,P)$,
       where the text obtains $-\log Q(\textrm{supp}\,P)$. Which bound is
       larger?
    1. Find a pair $P \neq Q$ for which the bound of the text holds with
       equality. Can the tangent-line bound hold with equality for
       $P \neq Q$?
    1. Why are the two proofs nevertheless the same argument? Hint: the
       standard proof of Jensen's inequality for a concave $f$ bounds $f$ by
       its tangent line at $\mathbb{E}[X]$.
   :exlabel:`ex_mdl-information-theory-gibbs-tangent`

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
       data set known to have $10\%$ such noise in every split. List
       possible explanations, and say which of them the floor rules out and
       which it does not.
    1. Train the softmax regression of :numref:`sec_softmax_scratch` on
       Fashion-MNIST with $10\%$ of the training and test labels flipped in
       this way. Compare the test loss with $0.545$ and the test accuracy with
       $90\%$. The gap is large. Which assumption of (a) does Fashion-MNIST
       violate, what does the linear model's own error add, and how could
       you estimate the floor that does apply?
   :exlabel:`ex_mdl-information-theory-label-noise-floor`

1. **Label noise on an ambiguous input.** The floor of the previous exercise
   assumes that the input determines the clean class. Fix one input at which
   it does not: the clean class $C$ has the distribution $\mathbf{c}$ on $k$
   classes, and the recorded label $Y$ equals $C$ with probability $1 - \rho$
   and is otherwise uniform over the other $k - 1$ classes.
    1. Show that $H(Y \mid C) = H_b(\rho) + \rho\log(k-1)$, whatever
       $\mathbf{c}$ is. Call this value $H_{\textrm{noise}}$.
    1. Write the chain rule :eqref:`eq_mdl-entropy-chain-rule` for the pair
       $(C, Y)$ in both orders, and conclude that
       $H_{\textrm{noise}} \le H(Y) \le H(C) + H_{\textrm{noise}}$. Which
       result of the text gives the lower bound? For which $\mathbf{c}$ and
       $\rho$ does the upper bound hold with equality?
    1. For $k = 2$ and $\rho = 0.1$, compute $H(Y)$ for
       $\mathbf{c} = (1, 0)$, $(0.9, 0.1)$ and $(\tfrac12, \tfrac12)$, and
       compare each value with the two bounds. How much does the noise add
       to the entropy of the label when the clean class is a fair coin, and
       why?
    1. A data set has $10\%$ label noise on ten classes, and some of its
       inputs are ambiguous. Is the value $0.545$ nats of
       :numref:`subsec_mdl-irreducible-loss` a lower or an upper bound on
       its floor?
   :exlabel:`ex_mdl-information-theory-noise-ambiguous-input`

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
       change in the initialization? Given what you found in (b), is the
       formula still trustworthy at that scale?
   :exlabel:`ex_mdl-information-theory-initial-loss`

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

1. **The excess loss of a Gaussian model.** A regression model reports a
   Gaussian $\mathcal{N}(m, s^2)$ for a target whose density $P$ has mean
   $\mu$, variance $\sigma^2$ and a finite differential entropy.
    1. Show that $\textrm{CE}(P, \mathcal{N}(m, s^2))$ is smallest at
       $m = \mu$ and $s^2 = \sigma^2$, and that this best Gaussian model
       exceeds the floor $h(P)$ by
       $D_{\textrm{KL}}(P\|\varphi) = h(\varphi) - h(P)$, where $\varphi$ is
       the density of $\mathcal{N}(\mu, \sigma^2)$.
    1. Evaluate the excess for the uniform density on $[0, a]$, which has
       variance $a^2/12$, and for the Laplace density
       $\tfrac{1}{2b} e^{-|x|/b}$, which has variance $2b^2$ and
       differential entropy $1 + \log(2b)$. (You should find about $0.176$
       and $0.072$ nats.)
    1. Neither answer depends on the scale $a$ or $b$. Explain why without
       computing, from what :numref:`subsec_mdl-differential-entropy` says
       about a change of units.
    1. The Gaussian has the largest entropy for a given variance, so no
       density has a higher floor. Is there also a largest excess? Consider
       densities that concentrate near the two points $\mu \pm \sigma$.
       What does your answer say about a Gaussian model for a target with
       two well-separated modes?
   :exlabel:`ex_mdl-information-theory-gaussian-excess`

[Discussions](https://d2l.discourse.group/t/1104)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37.1]{.kicker}

The loss you report is an average surprisal<br>**entropy, cross-entropy, and KL divergence**
:::
:::

::: {.slide title="A Fresh 10-Class Model Reports 2.30, a Perfect One 0, a Noisy One at Best 0.545"}
[The problem]{.kicker}

- An untrained classifier, nearly uniform predictions: $\ln 10 \approx 2.30$ nats.
- Clean labels and a perfect model: $0$.
- $10\%$ of the labels flipped to one of the other nine classes: no model,
  however large, gets below $0.545$ on fresh data.

::: {.d2l-note .warn}
Is a final loss of $0.6$ good? What does the number measure, and where is its floor?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute entropy and KL divergence of small distributions by hand;
- prove that KL is never negative (Gibbs' inequality);
- split any log-loss into a floor plus a never-negative excess;
- compute the floor of a noisy classification problem;
- say what label smoothing and distillation ask of the logits.

**Running examples:** the dyadic source $(\tfrac12, \tfrac14, \tfrac18, \tfrac18)$;
the pair $P = (0.6, 0.3, 0.1)$, $Q = (0.2, 0.5, 0.3)$.
**Idea:** loss $=$ average surprisal $=$ entropy $+$ KL.
:::

::: {.slide title="Losses First, Codes and Comparisons Later"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **Entropy, Cross-Entropy, and KL** | **surprise, KL, the loss floor** |
| Codes, Compression, and Perplexity | the code behind the log-loss |
| f-Divergences | one convex function of the ratio |
| Test Functions and Transport | averages and moving mass |
| Score-Based Comparisons | slopes, no normalizer |
| Mutual Information | dependence and what it limits |
| Estimating Mutual Information | what samples can show |
:::

::: {.col .narrow}
- **Uses:** softmax regression, Jensen's inequality.
- **Skips:** building a code, comparing distributions when KL is infinite.
:::
:::
:::

::: {.slide title="Surprisal Is $-\log p$, and Independent Surprisals Add"}
[Self-information]{.kicker}

::: {.cols .vc}
::: {.col}
$I(x) = -\log p(x)$: the *self-information* or *surprisal* of an outcome of
probability $p(x)$.

| report | $p$ | nats | bits |
|---|---|---|---|
| a card | $1$ | $0$ | $0$ |
| a heart | $1/4$ | $1.39$ | $2$ |
| the three of spades | $1/52$ | $3.95$ | $5.70$ |
| the order of the deck | $1/52!$ | $156.4$ | $225.6$ |

Independent probabilities multiply, so a logarithm makes surprisals add.
Natural logarithm: *nats*. Base $2$: *bits*, marked by a subscript $2$;
$1$ bit $= \ln 2 \approx 0.693$ nats.
:::

::: {.col .fig}
@fig:mdl-it-self-info-curve
:::
:::
:::

::: {.slide title="Entropy Is the Average Number of Yes/No Questions"}
[Entropy]{.kicker}

$H(P) = -\sum_i p_i \log p_i$, the expected surprisal, with $0 \log 0 = 0$.
Dyadic source on $a, b, c, d$: ask "$a$?", then "$b$?", then "$c$?":
$\tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac18 \cdot 3 + \tfrac18 \cdot 3 = 1.75$
questions, and $H_2(P) = 1.75$ bits.

@information-theory-definition

Two fixed questions would always take $2$.
:::

::: {.slide title="The First Answer Supplies One Bit, the Later Questions 0.75"}
[Joint and conditional entropy]{.kicker}

- Joint entropy $H(X, Y) = -\mathbb{E}[\log p(X, Y)]$: the entropy of the pair.
- Conditional entropy $H(Y \mid X) = \sum_x p(x)\, H(Y \mid X = x)$: what is
  left in $Y$ once $X$ is known.

::: {.d2l-note .rule}
Chain rule: $H(X, Y) = H(X) + H(Y \mid X)$, since
$\log p(x, y) = \log p(x) + \log p(y \mid x)$.
:::

$X$ answers "Is it $a$?", $Y$ is the outcome of the dyadic source:
$H_2(X) = 1$ bit. After "no", $(\tfrac12, \tfrac14, \tfrac14)$ has $1.5$
bits, so $H_2(Y \mid X) = \tfrac12 \cdot 1.5 = 0.75$ and
$H_2(X, Y) = 1 + 0.75 = 1.75 = H_2(Y)$: the outcome determines the answer.
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

::: {.slide title="KL Averages the Extra Surprisal of Believing Q When the Truth Is P"}
[KL divergence]{.kicker}

$D_{\textrm{KL}}(P\|Q) = \mathbb{E}_{x \sim P}[\log p(x) - \log q(x)]$:
surprisal under $Q$ minus surprisal under $P$, averaged over $P$. In bits:
extra questions.

~~~python
def kl(p, q):
    m = p > 0
    return np.dot(p[m], np.log(p[m] / q[m]))
~~~

@!information-theory-definition-2

Asymmetric, and $+\infty$ for $Q' = (0.5, 0.5, 0)$, which rules out an outcome
that occurs.
:::

::: {.slide title="KL Is Never Negative, and Zero Only at P = Q"}
[Gibbs' inequality]{.kicker}

::: {.d2l-note .rule}
$D_{\textrm{KL}}(P\|Q) \ge 0$, with equality if and only if $P = Q$.
:::

$$
D_{\textrm{KL}}(P\|Q) = -\mathbb{E}_P\Big[\log\frac{q}{p}\Big]
\ge -\log \mathbb{E}_P\Big[\frac{q}{p}\Big] = -\log Q(\mathrm{supp}\,P) \ge 0
$$

- Jensen for the concave $\log$, with $Z = q(x)/p(x)$:
  $\mathbb{E}[\log Z] \le \log\mathbb{E}[Z]$; the factors $p$ cancel.
- $\mathrm{supp}\,P$: the outcomes with $p > 0$. Equality: $q/p$ constant
  there, and $Q(\mathrm{supp}\,P) = 1$, so $q = p$.
- No model beats the truth on average.
:::

::: {.slide title="Entropy Is at Most log k, and Conditioning Cannot Raise It"}
[Two corollaries]{.kicker}

::: {.cols .vc}
::: {.col}
Against the uniform $U$ on $k$ outcomes, equality only at $P = U$:

$$\begin{aligned}
D_{\textrm{KL}}(P\|U) &= \textstyle\sum_i p_i \log(k\,p_i) \\
&= \log k - H(P) \ge 0.
\end{aligned}$$

- Coin with heads probability $p$:
  $H_b(p) = -p\log p - (1-p)\log(1-p) \le \ln 2$.
- Dyadic source: $1.75 < \log_2 4 = 2$ bits.
- Pair against its marginals: the average of
  $\log p(y \mid x) - \log p(y)$ is
  $H(Y) - H(Y \mid X) = D_{\textrm{KL}}(p_{X,Y}\|p_X p_Y) \ge 0$.
:::

::: {.col .fig}
@fig:mdl-it-bernoulli-entropy
:::
:::
:::

::: {.slide title="Cross-Entropy = Entropy + KL, So the Loss Has a Floor"}
[Cross-entropy]{.kicker}

::: {.cols .vc}
::: {.col}
$\textrm{CE}(P, Q) = -\mathbb{E}_P[\log q] = H(P) + D_{\textrm{KL}}(P\|Q)$

- the pair: $1.294 = 0.898 + 0.396$ nats;
- dyadic source, uniform model $U$: $2.0 = 1.75 + 0.25$ bits, two fixed
  questions against $1.75$ adaptive ones;
- $H(P)$ does not depend on $Q$: minimizing CE is minimizing KL.
:::

::: {.col .fig}
@fig:mdl-it-code-length-bars
:::
:::
:::

::: {.slide title="With a One-Hot Target, Cross-Entropy Is the Surprisal at the Label"}
[Classification]{.kicker}

- One-hot target on the label $y$, prediction $\mathbf{q}$ on $k$ classes:
  $\textrm{CE} = -\log q_y$; over $n$ examples,
  $-\tfrac1n \sum_i \log q_{i, y_i}$ nats.
- Uniform $\mathbf{q}$: $\ln k$ whatever the labels, $2.30$ for $k = 10$.
- Logits $\mathbf{z}$: $\mathbf{q} = \mathrm{softmax}(\mathbf{z})$. Built-in
  losses take $\mathbf{z}$ and apply a stable log-softmax.

::: {.d2l-note}
Maximum likelihood on discrete data minimizes the KL divergence from the
data frequencies to the model.
:::
:::

::: {.slide title="No Classifier Beats the Label Noise: 10% Noise on 10 Classes Floors the Loss at 0.545"}
[The floor]{.kicker}

$$
\mathbb{E}[-\log q(y \mid \mathbf{x})] = \mathbb{E}_{\mathbf{x}}[H(p(\cdot\mid\mathbf{x}))]
+ \mathbb{E}_{\mathbf{x}}[D_{\textrm{KL}}(p \,\|\, q)] \ge \mathbb{E}_{\mathbf{x}}[H(p(\cdot\mid\mathbf{x}))]
$$

On fresh data the label $y$ is one draw from the true $p(\cdot \mid \mathbf{x})$;
the model predicts $q(\cdot \mid \mathbf{x})$. Clean class fixed by the input,
label flipped with probability $\rho = 0.1$, $k = 10$:
$H_b(0.1) + 0.1 \ln 9 = 0.325 + 0.220 = 0.545$ nats. Three predictors $q$:

@!mdl-information-theory-the-irreducible-loss

::: {.d2l-note}
On this data a loss of $0.6$ is within $0.06$ nats of the best possible.
:::
:::

::: {.slide title="Truthful Reporting Is the Unique Optimum of the Log Score"}
[Scoring rules]{.kicker}

- A scoring rule charges $S(\mathbf{q}, y)$ for reporting $\mathbf{q}$ when
  $y$ occurs. Strictly proper: $\mathbf{q} = \mathbf{p}$ is the unique
  minimizer of $\mathbb{E}_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$.
- Log score $-\log q_y$: expected penalty $H(\mathbf{p}) +
  D_{\textrm{KL}}(\mathbf{p}\|\mathbf{q})$, strictly proper by Gibbs.
- Brier score $\sum_j (q_j - \mathbf{1}(j = y))^2$: also strictly proper.
- A confident error, $q_y = 0.01$: log score $4.6$, unbounded as
  $q_y \to 0$; Brier score never above $2$.

::: {.d2l-note}
The optimum is *calibrated*: of the inputs given probability $0.8$ for a
class, $80\%$ have it.
:::
:::

::: {.slide title="Check: Where Is the Floor with 20% Label Noise on 10 Classes?"}
[Check]{.kicker}

The clean class is fixed by the input, and $20\%$ of the labels are flipped
to one of the other nine classes. What is the smallest loss on fresh data?
What would a reported $0.40$ mean?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $H_b(0.2) + 0.2 \ln 9 = 0.500 + 0.439 = 0.940$ nats, at best
$80\%$ accuracy. A loss of $0.40$ on fresh data cannot be a better model:
look for leaked or cleaned labels.
:::
:::
:::

::: {.slide title="A Smoothed Target Puts the Optimum at a Finite Logit Gap: ln 91 = 4.51"}
[Label smoothing]{.kicker}

- One-hot target: $q_y = 1$ needs an infinite gap between the logits; the
  loss has no minimizer.
- Smoothed target, $\epsilon$ spread over all $k$ classes:
  $\mathbf{p}^\epsilon = (1-\epsilon)\,\mathbf{e}_y + \tfrac{\epsilon}{k}\mathbf{1}$.
  The loss $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ is smallest at
  $\mathbf{q} = \mathbf{p}^\epsilon$.
- Gap at the optimum: $z_y - z_j = \log\frac{1-\epsilon+\epsilon/k}{\epsilon/k}$.
  For $k = 10$ and the usual $\epsilon = 0.1$:
  $\ln\frac{0.91}{0.01} = \ln 91 = 4.51$.

::: {.d2l-note}
Loss at the optimum: $H(\mathbf{p}^\epsilon) = 0.500$ nats, not $0$.
:::
:::

::: {.slide title="One-Hot Targets Let the Logit Gap Grow Like ln(10t); Smoothing Stops It at 4.51"}
[Label smoothing]{.kicker}

Gradient descent on the logits of one example, $k = 10$, step size $1$,
gradient $\mathbf{q} - \mathbf{p}^\epsilon$. One-hot: a step adds about
$k e^{-\textrm{gap}}$, so the gap is about $\ln(kt)$ after $t$ steps.

@!information-theory-label-smoothing
:::

::: {.slide title="Distillation Trains a Student on the Teacher's Softened Probabilities"}
[Distillation]{.kicker}

- A small *student* imitates a large *teacher*. The teacher's probabilities
  for the wrong classes say that a "2" looks like a "7"; a one-hot label
  does not.
- A trained teacher is nearly one-hot, so divide both sets of logits by a
  *temperature* $\tau > 0$:
  $\mathbf{p}^{(\tau)} = \mathrm{softmax}(\mathbf{z}^{\mathrm{tea}}/\tau)$,
  $\mathbf{q}^{(\tau)} = \mathrm{softmax}(\mathbf{z}^{\mathrm{stu}}/\tau)$.
- Loss: $\tau^2 D_{\textrm{KL}}(\mathbf{p}^{(\tau)}\|\mathbf{q}^{(\tau)})$,
  zero only when the student matches the softened teacher.

::: {.d2l-note .rule}
Gradient of the KL term in the student logit $z^{\mathrm{stu}}_j$: softmax
minus target, over $\tau$:
$\tfrac{1}{\tau}\big(q_j^{(\tau)} - p_j^{(\tau)}\big)$.
:::
:::

::: {.slide title="The τ² Factor Keeps Distillation Gradients of Order One"}
[Distillation]{.kicker}

Large $\tau$, mean logit $\bar z$, $k$ classes:
$\mathrm{softmax}(\mathbf{z}/\tau)_j \approx \tfrac1k\big(1 + (z_j - \bar z)/\tau\big)$.
So $q_j^{(\tau)} - p_j^{(\tau)}$ shrinks like $1/\tau$ and the gradient
like $1/\tau^2$. Logits $(5, 2, -1)$ and $(3, 3, 0)$:

@!information-theory-distillation

Unscaled: $300$ times smaller at $\tau = 20$. Scaled: toward the centered
logit difference over $k$.
:::

::: {.slide title="Differential Entropy Moves with the Units; KL Does Not"}
[Densities]{.kicker}

$h(X) = -\int p(x) \log p(x)\,dx$: density values are not probabilities.

- Uniform on $[0, a]$: $h = \log a$; for $a = \tfrac12$, $h = -0.69$ nats.
- Metres to centimetres: $a \to 100a$, and $h$ gains $\ln 100 = 4.6$ nats.
- A ratio of densities does not move: for uniform $p_1$ on $[0, a_1]$ and
  $p_2$ on $[0, a_2]$, $a_1 \le a_2$,
  $\int p_1 \log(p_1/p_2)\,dx = \log(a_2/a_1)$ in any unit.
- Cells of width $\Delta$: $H(X_\Delta) + \log \Delta \to h(X)$, while
  $H(X_\Delta)$ itself diverges.
:::

::: {.slide title="A Narrow Model Is Far More Surprised by Wide Data Than the Reverse"}
[Gaussians]{.kicker}

$D_{\textrm{KL}}\big(\mathcal{N}(\mu_1, \sigma_1^2)\,\|\,\mathcal{N}(\mu_2, \sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12$

![](../img/mdl-it-kl-integrand.svg){width=82%}

$\mathcal{N}(0,1)$ against $\mathcal{N}(0,4)$: $0.318$ one way, $0.807$ the
other. The tails, where the narrow model gives little probability to
outcomes that occur, contribute $1.05$ nats.
:::

::: {.slide title="The Direction That Lives in the Tails Is Four Times Noisier to Estimate"}
[Gaussians]{.kicker}

Estimate $D_{\textrm{KL}}(P\|Q)$ by averaging $\log p(x) - \log q(x)$ over
$2 \cdot 10^5$ samples $x \sim P$; the standard error is the spread of the
log-ratio over $\sqrt{n}$.

@!information-theory-example-2

::: {.d2l-note}
Both directions average $\pm(\ln 2 - \tfrac38 x^2)$; under $\mathcal{N}(0,4)$,
$x^2$ fluctuates four times as much. Same accuracy: $16$ times the samples.
:::
:::

::: {.slide title="With Mean and Variance Fixed, the Gaussian Has the Largest Entropy"}
[Maximum entropy]{.kicker}

Gaussian density $\varphi$ with mean $\mu$, variance $\sigma^2$; any density
$P$ with the same two moments.

**I.** $-\log \varphi(x) = \tfrac12\log(2\pi\sigma^2) + \tfrac{(x-\mu)^2}{2\sigma^2}$
is quadratic, so $\textrm{CE}(P, \varphi) = \tfrac12 \log(2\pi e \sigma^2)$
for every such $P$.

**II.** $h(P) = \textrm{CE}(P, \varphi) - D_{\textrm{KL}}(P\|\varphi) \le
\tfrac12 \log(2\pi e \sigma^2) = h(\varphi)$, with equality only for
$P = \varphi$. $\blacksquare$

::: {.d2l-note}
The uniform bound is the same argument: $-\log u$ is constant, so
$\textrm{CE}(P, U) = \log k$ for every $P$.
:::
:::

::: {.slide title="Check: Heights Have σ = 0.07 m. Entropy in Metres? In Centimetres?"}
[Check]{.kicker}

Body heights are roughly Gaussian with $\sigma = 0.07$ m. What is their
differential entropy in nats? What is it when heights are recorded in
centimetres, and what happens to the KL divergence from a model?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\tfrac12 \ln(2\pi e) + \ln 0.07 = 1.42 - 2.66 = -1.24$ nats. In
centimetres, add $\ln 100 = 4.61$ to get $3.36$ nats. The KL divergence does
not change.
:::
:::
:::

::: {.slide title="Every Log-Loss Is a Floor Plus a Never-Negative Excess"}
[Wrap-up]{.kicker}

- Dyadic source: $H_2 = 1.75$ bits, the average number of questions.
- The pair: $D_{\textrm{KL}} = 0.396$ one way, $0.365$ the other;
  $\textrm{CE} = H + D_{\textrm{KL}} \ge H$.
- $10$ classes, $10\%$ noise: the floor is $0.545$ nats.
- Label smoothing: optimum at a logit gap of $\ln 91$, loss $0.500$;
  distillation: $\tau^2$ fixes the gradient scale.
- $\mathcal{N}(0,1)$, $\mathcal{N}(0,4)$: $0.318$ against $0.807$; entropy
  of a density depends on units, KL does not.

::: {.d2l-note}
Next: build the code whose length is this loss, and compare distributions
when KL is infinite.
:::
:::
