# Concentration and Generalization
:label:`sec_mdl-concentration-generalization`

Every guarantee this chapter has offered so far decays *polynomially*.
Chebyshev's inequality :eqref:`eq_mdl-chebyshev` caps the miss probability of a
sample mean at $\sigma^2/(nt^2)$---a rate of $1/n$---and the tests and
confidence intervals of :numref:`sec_mdl-statistics` traded that honest bound
away for Gaussian *approximations* that are only exact in the limit. The truth
is far better: when the data are bounded, the probability that an average
strays from its mean decays *exponentially* in $n$, and this section proves
it. The stakes are concrete. The main book bounded the error of a test-set
estimate with Hoeffding's inequality (:numref:`chap_classification_generalization`),
quoting it on faith; here we pay that debt with a proof. The same machinery
then explains the strange geometry of high dimension teased in
:numref:`sec_mdl-distributions`---why a Gaussian is a thin shell, why random
directions are orthogonal, why initialization scales work---and carries us to
the question all of it serves: *generalization*. A learner that picks its
function after seeing the data voids the single-function guarantee, and the
repair---uniform convergence, first over finite classes and then through
Rademacher complexity---is the promised mechanics behind the classical
generalization bounds of :numref:`sec_generalization_deep`. The section closes
where the classical story visibly breaks and something more interesting
appears: interpolation and *double descent*, reproduced from scratch in
twenty-five lines and explained by the very quantity---the norm of the
solution---that the Rademacher calculation says is the true capacity knob.

We load the per-framework library so the computational cells have `d2l` in
scope, plus plain NumPy as `onp`: every computation in this section is a
framework-agnostic matter of tails, norms, and least squares, so all the
worked cells below are shared across frameworks.

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
import numpy as onp  # plain NumPy: the cells below are framework-agnostic
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import numpy as onp  # plain NumPy: the cells below are framework-agnostic
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as onp  # plain NumPy: the cells below are framework-agnostic
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
import numpy as onp  # plain NumPy: the cells below are framework-agnostic
```

## From Chebyshev to Chernoff

### Polynomial Tails Are Not Enough

Recall the ladder built in :numref:`sec_mdl-random_variables`. Markov's
inequality :eqref:`eq_mdl-markov` turns a bare mean into a tail bound; feeding
it the squared deviation sharpens it into Chebyshev's inequality
:eqref:`eq_mdl-chebyshev`. Applied to a sample mean $\bar X$ of $n$ i.i.d.
draws with variance $\sigma^2$, Chebyshev gives

$$
P\bigl(|\bar X - \mu| \ge t\bigr) \le \frac{\sigma^2}{n\,t^2},
$$

and this was enough to prove consistency in :numref:`sec_mdl-statistics`. But
look at what it costs. To certify a miss probability of $10^{-6}$ at fixed $t$
we must grow $n$ by a factor of a *million* relative to certifying $10^{0}$:
the bound decays only like $1/n$. Meanwhile the exact tail of, say, a fair
coin's head-frequency collapses *exponentially* fast---we will compute it
below and watch Chebyshev fall behind by thirty orders of magnitude. The
Gaussian machinery of :numref:`sec_mdl-statistics` (the $z$-test, the
$1.96\,\hat\sigma/\sqrt n$ interval) implicitly knows this, since Gaussian
tails decay like $e^{-t^2/2}$, but it buys the rate with an approximation that
is only exact as $n\to\infty$. What we want is a bound with the *Gaussian
rate* and the *finite-sample honesty* of Chebyshev. The obstruction is easy to
name: Chebyshev sees only the second moment. A tail bound that decays like
$e^{-ct^2}$ must see *every* moment at once, and there is a single function
that packages them all.

### The Chernoff Method

The *moment generating function* (MGF) of a random variable $X$ is
$M(\lambda)=E[e^{\lambda X}]$; expanding the exponential shows its Taylor
coefficients are exactly the moments $E[X^k]/k!$, so bounding $M$ bounds every
moment simultaneously. The *Chernoff method* is Markov's inequality applied
not to $X$ but to $e^{\lambda X}$---a monotone transform, so the event is
unchanged---followed by an optimization over the free parameter: for any
$\lambda>0$,

$$
P(X \ge t)
 = P\bigl(e^{\lambda X} \ge e^{\lambda t}\bigr)
 \le e^{-\lambda t}\,E\bigl[e^{\lambda X}\bigr],
\qquad\textrm{hence}\qquad
P(X \ge t) \le \inf_{\lambda > 0}\, e^{-\lambda t}\, M(\lambda).
$$
:eqlabel:`eq_mdl-chernoff`

That is the whole method. Its power appears the moment $X$ is a *sum* of
independent terms: the MGF of a sum factors into a *product* of MGFs (the
expectation of a product of independent variables is the product of
expectations), so the exponent in :eqref:`eq_mdl-chernoff` grows linearly in
$n$, and optimizing $\lambda$ then turns that linear-in-$n$ exponent into the
exponential decay we are after. Everything below is this one display with a
good bound on the MGF plugged in. For bounded variables the sharpest generic
MGF bound is a lemma of Hoeffding, and it is worth proving carefully because
it is the engine of the whole section.

### Hoeffding's Lemma and Hoeffding's Inequality

**Proposition (Hoeffding's lemma).** *Let $X$ be a random variable with
$E[X]=0$ taking values in $[a,b]$. Then for every $\lambda\in\mathbb{R}$,*

$$
E\bigl[e^{\lambda X}\bigr] \le \exp\!\left(\frac{\lambda^2 (b-a)^2}{8}\right).
$$
:eqlabel:`eq_mdl-hoeffding-lemma`

**Proof.** Let $\psi(\lambda)=\log E[e^{\lambda X}]$, the cumulant generating
function; for bounded $X$ it is finite and smooth, and differentiating under
the expectation is routine. Then $\psi(0)=0$, and

$$
\psi'(\lambda) = \frac{E[X e^{\lambda X}]}{E[e^{\lambda X}]},
\qquad\textrm{so}\qquad
\psi'(0) = E[X] = 0 .
$$

The first derivative has a probabilistic reading that does all the work:
$\psi'(\lambda)$ is the mean of $X$ under the *tilted* distribution
$p_\lambda(x) \propto e^{\lambda x}\,p(x)$---a genuine probability
distribution, still supported inside $[a,b]$. Differentiating once more,

$$
\psi''(\lambda)
 = \frac{E[X^2 e^{\lambda X}]}{E[e^{\lambda X}]}
 - \left(\frac{E[X e^{\lambda X}]}{E[e^{\lambda X}]}\right)^{\!2}
 = \operatorname{Var}_{\lambda}(X),
$$

the variance of $X$ under the tilt. Now use the fact that *any* random
variable $Y$ supported in $[a,b]$ has variance at most $(b-a)^2/4$: variance
is the smallest expected squared deviation over all centers (the mean
minimizes it), and every point of $[a,b]$ lies within $(b-a)/2$ of the
midpoint, so

$$
\operatorname{Var}(Y)
 \le E\!\left[\Bigl(Y - \tfrac{a+b}{2}\Bigr)^{\!2}\right]
 \le \left(\frac{b-a}{2}\right)^{\!2}.
$$

Hence $\psi''(\lambda) \le (b-a)^2/4$ for *every* $\lambda$. Taylor's theorem
with Lagrange remainder finishes it: for some $\xi$ between $0$ and $\lambda$,

$$
\psi(\lambda)
 = \psi(0) + \lambda\,\psi'(0) + \frac{\lambda^2}{2}\,\psi''(\xi)
 \le \frac{\lambda^2 (b-a)^2}{8},
$$

and exponentiating gives :eqref:`eq_mdl-hoeffding-lemma`. $\blacksquare$

The lemma says a bounded, centered variable has an MGF *no worse than a
Gaussian's* with standard deviation $(b-a)/2$---compare the Gaussian identity
$E[e^{\lambda X}]=e^{\lambda^2\sigma^2/2}$ for $X\sim\mathcal N(0,\sigma^2)$,
an integral worth doing once by completing the square. Feeding the lemma into
the Chernoff method yields the headline result, in four lines.

**Proposition (Hoeffding's inequality).** *Let $X_1,\ldots,X_n$ be independent
random variables with $X_i\in[a,b]$, and let $\bar X = \frac1n\sum_i X_i$.
Then for every $t>0$,*

$$
P\bigl(|\bar X - E[\bar X]| \ge t\bigr)
 \le 2\exp\!\left(-\frac{2 n t^2}{(b-a)^2}\right).
$$
:eqlabel:`eq_mdl-hoeffding`

**Proof.** Write $Z_i = X_i - E[X_i]$; each is centered and supported in an
interval of length $b-a$, so the lemma applies. For any $\lambda>0$, Chernoff
:eqref:`eq_mdl-chernoff` plus independence (the MGF of the sum is the product
of MGFs) gives

$$
P(\bar X - E[\bar X] \ge t)
 = P\Bigl(\sum_i Z_i \ge nt\Bigr)
 \le e^{-\lambda n t} \prod_{i=1}^n E\bigl[e^{\lambda Z_i}\bigr]
 \le \exp\!\left(-\lambda n t + \frac{n\lambda^2 (b-a)^2}{8}\right).
$$

The exponent is a parabola in $\lambda$, minimized at
$\lambda = 4t/(b-a)^2$, where its value is $-2nt^2/(b-a)^2$. Applying the same
argument to $-Z_i$ bounds the lower tail identically, and a union bound over
the two tails supplies the factor $2$. $\blacksquare$

This is the inequality of :cite:`Hoeffding.1963`, and it pays the main book's
debt. :numref:`chap_classification_generalization` used exactly
:eqref:`eq_mdl-hoeffding` (with losses in $[0,1]$, so $b-a=1$) to certify a
test-set estimate of the error rate: demanding
$P(|\epsilon_{\mathcal D}(f)-\epsilon(f)|\ge 0.01)\le 0.05$ and solving
$2e^{-2nt^2}\le\delta$ for $n$ gives $n = \log(2/\delta)/(2t^2) \approx
18{,}445$---the "roughly 18,500 examples" quoted there, now a theorem rather
than a citation. Note what the bound does *not* need: no Gaussianity, no
variance estimate, no asymptotics---only boundedness and independence, and it
holds at every finite $n$.

Inverting the inequality gives the form a practitioner actually reaches for.
Set the right-hand side of :eqref:`eq_mdl-hoeffding` equal to $\delta$ and
solve for $t$: with probability at least $1-\delta$,

$$
\bigl|\bar X - E[\bar X]\bigr|
 \le (b-a)\sqrt{\frac{\log(2/\delta)}{2n}} .
$$
:eqlabel:`eq_mdl-hoeffding-interval`

This is a *finite-sample confidence interval* in the exact sense of
:numref:`sec_mdl-statistics`, with the same $1/\sqrt n$ half-width scaling as
the Gaussian interval :eqref:`eq_mdl-gauss_confidence`---but where that
interval's coverage was asymptotic (and, as the coverage audit there measured,
slightly optimistic at finite $n$), this one is a guarantee at every $n$. The
price is the constant: Hoeffding budgets for the most adversarial distribution
on $[a,b]$, so the range $b-a$ stands where the Gaussian interval enjoys the
estimated $\hat\sigma$, and when the true spread is much smaller than the
range the interval is correspondingly conservative---slack that Bernstein's
inequality, below, is designed to recover. Keep
:eqref:`eq_mdl-hoeffding-interval` in view; it is the statement we will
upgrade from one function to entire function classes in the second half of
the section.

### Sub-Gaussian and Sub-Exponential Variables

The proof used only one property of the summands: an MGF bounded by a
Gaussian's. That property deserves a name, because it is the right abstraction
for everything that follows. A random variable $X$ is **sub-Gaussian** with
*variance proxy* $\sigma^2$ if

$$
E\bigl[e^{\lambda (X - E[X])}\bigr] \le \exp\!\left(\frac{\lambda^2\sigma^2}{2}\right)
\qquad\textrm{for all } \lambda\in\mathbb{R}.
$$
:eqlabel:`eq_mdl-subgaussian`

The Chernoff method converts :eqref:`eq_mdl-subgaussian` directly into the
two-sided tail $P(|X-E[X]|\ge t)\le 2e^{-t^2/(2\sigma^2)}$, by the same
optimization as above. Three families matter to us. A *Gaussian*
$\mathcal N(\mu,\sigma^2)$ is sub-Gaussian with proxy exactly $\sigma^2$ (the
MGF identity holds with equality). A *bounded* variable on $[a,b]$ is
sub-Gaussian with proxy $(b-a)^2/4$---that is Hoeffding's lemma. And a
*Rademacher* variable, a uniform random sign $\varepsilon\in\{-1,+1\}$, is
sub-Gaussian with proxy $1$: $E[e^{\lambda\varepsilon}]=\cosh\lambda\le
e^{\lambda^2/2}$, a term-by-term comparison of Taylor series left as exercise
1. Sub-Gaussianity is closed under averaging: if $X_1,\ldots,X_n$ are
independent with proxy $\sigma^2$, then $\bar X$ is sub-Gaussian with proxy
$\sigma^2/n$---the MGF product argument again---so *every* sub-Gaussian
average concentrates at the Hoeffding rate.

Not everything is sub-Gaussian. The *square* of a Gaussian (a $\chi^2$
variable, hence squared norms of Gaussian vectors) has a right tail decaying
like $e^{-t/2}$, too heavy for :eqref:`eq_mdl-subgaussian`; its MGF exists
only for $\lambda$ near zero. Such variables are called *sub-exponential*, and
they concentrate at a Gaussian rate for small deviations and an exponential
(not squared-exponential) rate for large ones. The clean statement is
**Bernstein's inequality**: if the $X_i$ are independent with
$\operatorname{Var}(X_i)\le\sigma^2$ and $|X_i - E[X_i]|\le M$, then

$$
P\bigl(|\bar X - E[\bar X]| \ge t\bigr)
 \le 2\exp\!\left(-\frac{n t^2}{2\bigl(\sigma^2 + M t/3\bigr)}\right),
$$
:eqlabel:`eq_mdl-bernstein`

which we state without proof :cite:`Boucheron.Lugosi.Massart.2013`. Read it as
the variance-aware refinement of Hoeffding: for small $t$ the denominator is
essentially $2\sigma^2$ and the bound matches the CLT's Gaussian tail with the
*true* variance---often far smaller than the worst case $(b-a)^2/4$ that
Hoeffding must assume---while for large $t$ the $Mt/3$ term takes over and the
decay is exponential in $t$, which is the best a bounded-but-skewed sum can
do. We will lean on exactly this small-deviation sharpness when we prove norm
concentration below.

### The Tail Race in Code

Claims about rates deserve a measurement. For the fair coin we can compute the
tail $P(|\hat p - \tfrac12| \ge 0.1)$ *exactly*---it is a finite binomial
sum, which we evaluate in log-space so that $n=5{,}000$ does not underflow---
and race it against Chebyshev ($\sigma^2=\tfrac14$, so the bound is
$25/n$ at $t=0.1$) and Hoeffding ($2e^{-2nt^2}=2e^{-0.02n}$). No sampling is
involved; every number below is deterministic.

```{.python .input #mdl-concentration-generalization-the-tail-race-in-code}
t = 0.1  # deviation of the head-frequency from 1/2

def exact_tail(n):  # P(|p_hat - 1/2| >= t): binomial sum in log-space
    k = onp.arange(n + 1)
    log_pmf = n * onp.log(0.5) + onp.concatenate(
        ([0.0], onp.cumsum(onp.log((n - k[1:] + 1) / k[1:]))))
    in_tail = log_pmf[onp.abs(k / n - 0.5) >= t - 1e-12]  # fp-safe boundary
    m = in_tail.max()
    return onp.exp(m) * onp.exp(in_tail - m).sum()  # stable log-sum-exp

print(f'{"n":>6} {"exact":>10} {"Chebyshev":>10} {"Hoeffding":>10}')
for n in (10, 100, 1000, 5000):
    print(f'{n:6d} {exact_tail(n):10.2e} {0.25 / (n * t**2):10.2e} '
          f'{2 * onp.exp(-2 * n * t**2):10.2e}')

ns = onp.arange(10, 701, 10)
d2l.plot(ns, [onp.array([exact_tail(n) for n in ns]),
              0.25 / (ns * t**2), 2 * onp.exp(-2 * ns * t**2)],
         'n', 'P(|p_hat - 1/2| >= 0.1)',
         legend=['exact', 'Chebyshev', 'Hoeffding'], yscale='log')
```

The table tells the whole story of this section's first half. At $n=10$ both
bounds are vacuous (above $1$) while the exact tail is $0.75$---nothing
concentrates in ten flips. At $n=100$ Chebyshev ($0.25$) is actually a shade
*tighter* than Hoeffding ($0.27$): for small $n$ the fight is between
constants, and Chebyshev's use of the true variance $\tfrac14$ helps it. But
rates always beat constants. By $n=1{,}000$ the exact tail is
$2.7\times10^{-10}$ and Hoeffding certifies $4.1\times10^{-9}$---the right
order of magnitude---while Chebyshev still allows a miss probability of
$2.5\%$, off by *seven orders of magnitude*; at $n=5{,}000$ the gap has grown
to forty orders. The log-scale plot makes the geometry plain: the exact tail
and Hoeffding's bound are parallel straight lines (exponential decay, and
Hoeffding's exponent $2t^2=0.02$ is remarkably close to the optimal rate,
which large-deviations theory identifies as the KL divergence
$\mathrm{KL}(0.6\,\|\,0.5)\approx 0.0201$), while Chebyshev's gentle curve is
polynomial decay that no constant can rescue.

## Probability in High Dimension

Concentration is not only about sample means over datasets. The coordinates of
a random *vector* are also "many independent contributions", so the same
inequalities govern the geometry of high-dimensional space---and that geometry
is where deep learning lives. :numref:`sec_mdl-distributions` stated the two
headline facts while cataloguing the multivariate Gaussian: the norm of a
$d$-dimensional standard Gaussian locks onto $\sqrt d$, and two independent
draws are nearly orthogonal. Here we give both facts their quantitative
treatment.

### Norm Concentration

**Proposition (norm concentration).** *Let
$\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$. For every $0<\varepsilon\le 1$,*

$$
P\!\left(\Bigl|\frac{\|\mathbf x\|}{\sqrt d} - 1\Bigr| \ge \varepsilon\right)
 \le 2\exp\!\left(-\frac{d\,\varepsilon^2}{8}\right).
$$
:eqlabel:`eq_mdl-norm-concentration`

**Proof sketch.** The squared norm $\|\mathbf x\|^2=\sum_{i=1}^d x_i^2$ is a
sum of $d$ i.i.d. variables $x_i^2$ with mean $1$---exactly a sample-mean
problem, except the summands are unbounded, so Hoeffding does not apply
directly. They are, however, sub-exponential, and this is where the
Bernstein-flavored small-$\lambda$ control earns its keep. The MGF is
explicit, $E[e^{\lambda(x_i^2-1)}] = e^{-\lambda}/\sqrt{1-2\lambda}$ for
$\lambda<\tfrac12$, and a one-line series comparison (which we state and skip;
see :cite:`Vershynin.2018`, §2.7) shows

$$
E\bigl[e^{\lambda (x_i^2 - 1)}\bigr] \le e^{2\lambda^2}
\qquad\textrm{for } |\lambda| \le \tfrac14 .
$$

Chernoff with this bound gives, for any $0<\lambda\le\tfrac14$,

$$
P\Bigl(\sum_i (x_i^2 - 1) \ge d\varepsilon\Bigr)
 \le \exp\bigl(-\lambda d \varepsilon + 2 d \lambda^2\bigr),
$$

and the unconstrained minimizer $\lambda=\varepsilon/4$ is admissible
precisely when $\varepsilon\le 1$, yielding $e^{-d\varepsilon^2/8}$; the lower
tail and the factor $2$ follow as in Hoeffding's inequality. Finally convert
squares to norms: writing $z=\|\mathbf x\|/\sqrt d\ge 0$, if $|z-1|\ge\varepsilon$
then $|z^2-1|=|z-1|\,(z+1)\ge\varepsilon$, so the norm event implies the
squared-norm event and inherits its bound. $\blacksquare$

Two readings. First, the *fluctuation of $\|\mathbf x\|$ is of constant
order*: the bound says deviations of $\|\mathbf x\|$ from $\sqrt d$ beyond
$\varepsilon\sqrt d$ are exponentially rare in $d$, and a finer analysis puts
the standard deviation of $\|\mathbf x\|$ near $1/\sqrt2$ *independently of
$d$*. A standard Gaussian in $\mathbb{R}^d$ is therefore not a fuzzy ball
around the origin but a **thin shell** of radius $\sqrt d$ and thickness
$O(1)$. Second, the mode is not the mass: the density is largest at the
origin, yet the volume of a radius-$r$ shell grows like $r^{d-1}$, and the
fight between decaying density and exploding volume is settled overwhelmingly
at $r\approx\sqrt d$. In $d=784$ (an MNIST-sized Gaussian) a typical draw has
norm within a few percent of $28$; a draw with norm below $14$ is, by
:eqref:`eq_mdl-norm-concentration`, rarer than $2e^{-24}$---you will never see
one.

### Near-Orthogonality Revisited

The second headline fact is the cosine. :numref:`sec_mdl-geometry-linear-algebraic-ops`
proved that the cosine between independent random directions has mean $0$ and
standard deviation $\approx 1/\sqrt d$; :numref:`sec_mdl-distributions`
restated it for Gaussians. Concentration puts that "typical" statement into
the same exponential frame as everything else in this section, and the
mechanism is worth seeing because it is *Hoeffding again*. Fix a unit vector
$\mathbf u$ and let $\mathbf x$ have independent, centered coordinates bounded
in $[-1,1]$ (Gaussian coordinates work too, via :eqref:`eq_mdl-subgaussian`
with proxy $1$). The inner product $\langle\mathbf x,\mathbf u\rangle=\sum_i
u_i x_i$ is a sum of independent terms, the $i$-th bounded by $|u_i|$, and the
general Hoeffding inequality for non-identical ranges (exercise 2) gives

$$
P\bigl(|\langle\mathbf x,\mathbf u\rangle| \ge s\bigr)
 \le 2\exp\!\left(-\frac{s^2}{2\sum_i u_i^2}\right) = 2 e^{-s^2/2},
$$

since $\sum_i u_i^2=1$. The inner product with any fixed direction is $O(1)$
while the norm of $\mathbf x$ is $\approx\sqrt d$ by
:eqref:`eq_mdl-norm-concentration`, so the cosine is $O(1/\sqrt d)$ not just
on average but with exponentially high probability: taking
$s=\varepsilon\sqrt d$, a cosine larger than $\approx\varepsilon$ has
probability at most $\approx 2e^{-d\varepsilon^2/2}$. High-dimensional space
holds exponentially many pairwise-nearly-orthogonal directions, and *random
ones come that way by default*.

### What This Buys Deep Learning

Three payoffs, one paragraph each.

**Initialization scales.** LeCun and He initialization draw the weight rows of
a layer with variance $1/d$ (or $2/d$ for ReLU), and the standard telling
(:numref:`sec_numerical_stability`) checks variances: each pre-activation
$\langle\mathbf w,\mathbf x\rangle$ has variance $\|\mathbf x\|^2/d\approx 1$.
Concentration upgrades the telling. A layer output is a $d$-dimensional
random-ish vector of such coordinates, and by exactly the argument of
:eqref:`eq_mdl-norm-concentration` its *norm* concentrates: the claim "unit
activations layer after layer" is not merely true in expectation but holds for
essentially every draw of a wide network's weights, with failure probability
exponentially small in width. That is why a single forward pass at
initialization is a meaningful diagnostic at all---the number it prints is the
number every other draw would have printed.

**Distance concentration and nearest neighbors.** For two independent draws
the difference $\mathbf x-\mathbf y$ is again a Gaussian vector, now with
coordinate variance $2$, so :eqref:`eq_mdl-norm-concentration` applies to it
verbatim: each pairwise distance lands in $(1\pm\varepsilon)\sqrt{2d}$ except
with probability $2e^{-d\varepsilon^2/8}$. Among $n$ unstructured points there
are fewer than $n^2$ pairs, and a union bound puts *every* distance in that
shell as soon as

$$
n^2 \cdot 2e^{-d\varepsilon^2/8} \le \delta,
\qquad\textrm{i.e.}\qquad
d \ \ge\ \frac{8}{\varepsilon^2}\log\frac{2n^2}{\delta} :
$$

a dimension only *logarithmic* in the number of points suffices to flatten the
entire distance matrix into a band of relative width $\varepsilon$. (File the
move away for the next section: a union bound converting one exponential tail
into a simultaneous guarantee over many objects, at logarithmic cost, is
precisely how learning bounds are built.) The nearest and farthest neighbor of
a query then differ by a vanishing relative margin---the contrast
$(d_{\max}-d_{\min})/d_{\min}$ collapses---and nearest-neighbor retrieval
degenerates into noise-ranking. Real embeddings
escape only because they are not unstructured: semantic structure puts data on
a low-dimensional set inside $\mathbb{R}^d$, and the *gap* between the
concentration prediction and observed distance histograms is a working measure
of how much structure an embedding has.

**The thin shell and the mean.** The shell picture corrects a beginner's
mental model that matters in generative modeling: the "most likely point" of a
high-dimensional Gaussian prior---the origin---is nowhere near a *typical*
point. Decoders are trained on inputs of norm $\approx\sqrt d$, so evaluating
one at the mean, or at the midpoint of two latent draws (norm
$\approx\sqrt{d/2}$, well inside the shell), feeds it an input unlike anything
it saw in training; this is why latent-space interpolation is done along the
sphere rather than the chord. The shell is also why the MVN cell of
:numref:`sec_mdl-distributions` saw norm ratios pinned to $1.000$: it was
measuring means; the cell below measures the *whole distribution*.

### Measuring the Shell

The distributions chapter verified the *means* of the norm ratio and the
cosine. Concentration makes distributional claims---what *fraction* of the
mass sits inside the shell $(1\pm\varepsilon)\sqrt d$, and how badly
nearest-neighbor contrast collapses---so those are what we measure, at
$\varepsilon=0.1$.

```{.python .input #mdl-concentration-generalization-measuring-the-shell}
rng = onp.random.default_rng(0)
eps, num_pts = 0.1, 10000
for d in (2, 20, 200, 2000):
    x = rng.standard_normal((num_pts, d))
    ratio = onp.linalg.norm(x, axis=1) / onp.sqrt(d)
    shell = onp.mean(onp.abs(ratio - 1) <= eps)      # mass in the 10% shell
    query, points = x[0], x[1:201]                   # 1 query, 200 neighbors
    dist = onp.linalg.norm(points - query, axis=1)
    contrast = (dist.max() - dist.min()) / dist.min()
    print(f'd={d:5d}:  mass within (1±0.1)·sqrt(d): {shell:6.1%},   '
          f'NN contrast (d_max-d_min)/d_min = {contrast:6.3f}')
```

In $d=2$ the "shell" holds under $15\%$ of the mass---low dimension really is
a fuzzy ball, and the nearest and farthest of $200$ points differ by a factor
of $30$. By $d=200$ the same $\pm10\%$ shell already captures $95\%$ of the
mass, close to what :eqref:`eq_mdl-norm-concentration`'s Gaussian-rate reading
predicts ($\pm 2$ standard deviations at $\sigma\approx 1/\sqrt{2d}=0.05$),
and at $d=2000$ the empirical mass is $100.0\%$: not one of ten thousand draws
left the shell. The nearest-neighbor contrast tells the matching story,
collapsing from $29.4$ to $0.08$---in $d=2000$ the farthest of two hundred
random points is only $8\%$ farther than the nearest, so "nearest" no longer
means much.

## From One Estimate to Uniform Convergence

### The Function Chosen After the Data

Return to learning, where a subtlety voids everything proved so far.
Hoeffding's inequality certifies the empirical risk of **one fixed function**:
choose $f$, *then* draw the sample $S=\{(\mathbf x_i,y_i)\}_{i=1}^n$, and the
empirical risk $\hat R(f)=\frac1n\sum_i \ell(f(\mathbf x_i),y_i)$ sits within
$t$ of the true risk $R(f)=E[\ell(f(\mathbf x),y)]$ except with probability
$2e^{-2nt^2}$. A test set works exactly because it respects this order: the
model was frozen before the test data were drawn. But a *learner* violates the
order by construction---it searches a class $\mathcal F$ and returns the
$\hat f$ that makes $\hat R$ small *on the sample it saw*. The empirical risk
of $\hat f$ is a minimum of fluctuating quantities, selected precisely where
the fluctuation is most favorable, and its optimism is not covered by any
single-function bound. The quantity that must concentrate is the *worst case
over the class*,

$$
\sup_{f\in\mathcal F}\ \bigl|\hat R(f) - R(f)\bigr|,
$$

because if this supremum is at most $t$ then *every* function's empirical risk
is trustworthy to $t$---including whichever one the learner happens to pick.
Bounds on this supremum are called **uniform convergence** bounds, and they
are the classical mechanics that :numref:`sec_generalization_deep` waves at.
The same trap, in different clothes, is the *test-set reuse* discussed in
:numref:`chap_classification_generalization`: an analyst who evaluates many
models on one test set and reports the best is a learner whose "class" is the
set of models tried, and the bound we prove next is the exact price of that
adaptivity.

### Finite Classes: the Union Bound

For a finite class the repair is one line of probability.

**Proposition (finite-class uniform convergence).** *Let $\mathcal F$ be
finite, let the loss take values in $[0,1]$, and let $S$ be an i.i.d. sample
of size $n$. Then with probability at least $1-\delta$,*

$$
\bigl|\hat R(f) - R(f)\bigr|
 \le \sqrt{\frac{\log(2|\mathcal F|/\delta)}{2n}}
\qquad\textrm{simultaneously for every } f\in\mathcal F .
$$
:eqlabel:`eq_mdl-finite-class`

**Proof.** For each fixed $f$, $\hat R(f)$ is an average of $n$ i.i.d. terms
in $[0,1]$, so Hoeffding :eqref:`eq_mdl-hoeffding` gives
$P(|\hat R(f)-R(f)|\ge t)\le 2e^{-2nt^2}$. The probability that *any* of the
$|\mathcal F|$ bad events occurs is at most the sum of their probabilities
(the union bound), i.e. $2|\mathcal F|e^{-2nt^2}$. Setting this to $\delta$
and solving for $t$ gives the claim. $\blacksquare$

Read the bound the way a practitioner should: $\log|\mathcal F|$ **is the
price of choice**. Guaranteeing one function costs $\log(2/\delta)$ in the
numerator; guaranteeing the freedom to pick among $|\mathcal F|$ costs an
extra $\log|\mathcal F|$---that is, *each bit of selection freedom costs one
bit's worth of sample*. Since the deviation shrinks like the square root, the
sample size needed grows only *logarithmically* in the class size, which is
why choosing among even thousands of hyperparameter configurations on a
$10{,}000$-point validation set is tolerable: $t$ grows only from
$\sqrt{\log(2/\delta)/2n}\approx 0.014$ to
$\sqrt{\log(2000/\delta)/2n}\approx 0.023$ at $\delta=0.05$. Run the same
arithmetic on a leaderboard with a million adaptive submissions and the
guarantee has quietly tripled; this is
:numref:`chap_classification_generalization`'s test-set-reuse warning, now
with its constant.

### Rademacher Complexity

Real hypothesis classes are infinite---every $\mathbf w\in\mathbb{R}^d$ is a
different linear classifier---so the union bound stalls. The classical escapes
either *discretize* the class (covering numbers) or count its effective
behaviors on $n$ points (the VC dimension sketched in
:numref:`chap_classification_generalization`). The modern workhorse is more
direct, and it starts from a question a machine learner can act on: **how well
can the class correlate with pure noise?** Let
$\varepsilon_1,\ldots,\varepsilon_n$ be i.i.d. Rademacher variables---fair
random signs---and define the **empirical Rademacher complexity** of a class
$\mathcal F$ on the sample $S=(\mathbf x_1,\ldots,\mathbf x_n)$ as

$$
\widehat{\mathfrak R}_S(\mathcal F)
 = E_{\boldsymbol\varepsilon}\!\left[\,
   \sup_{f\in\mathcal F}\ \frac1n \sum_{i=1}^n \varepsilon_i\, f(\mathbf x_i)
 \right],
$$
:eqlabel:`eq_mdl-rademacher`

with $\mathfrak R_n(\mathcal F)=E_S[\widehat{\mathfrak R}_S(\mathcal F)]$ its
average over samples. The random signs are fictitious labels with *no signal
whatsoever*; the supremum asks how large a sample correlation the class can
manufacture with them anyway. A class that can chase arbitrary coin flips
($\widehat{\mathfrak R}\approx 1$) can chase the noise in real labels too and
its empirical risks mean little; a class that cannot
($\widehat{\mathfrak R}\to 0$) has empirical means that track true means. That
intuition is a theorem:

**Uniform convergence via Rademacher complexity**
:cite:`Bartlett.Mendelson.2002`. *For $\mathcal F$ taking values in $[0,1]$,
with probability at least $1-\delta$ over the sample,*

$$
R(f) \le \hat R(f) + 2\,\mathfrak R_n(\mathcal F)
 + \sqrt{\frac{\log(1/\delta)}{2n}}
\qquad\textrm{simultaneously for every } f\in\mathcal F .
$$
:eqlabel:`eq_mdl-rademacher-bound`

We do not give the full proof, but its central move---*symmetrization*, the
step that conjures coin flips out of a statement that contains none---is short
enough to sketch, and it explains the factor $2$. Introduce a *ghost sample*
$S'$: a second, fictitious dataset of size $n$, independent of $S$, that
exists only inside the analysis. Since $R(f)=E_{S'}[\hat R_{S'}(f)]$, Jensen's
inequality bounds the expected uniform deviation by a comparison of two
concrete samples,

$$
E_S\Bigl[\sup_{f} \bigl(R(f) - \hat R_S(f)\bigr)\Bigr]
 \le E_{S,S'}\Bigl[\sup_{f} \frac1n \sum_{i=1}^n
 \bigl(\ell'_i(f) - \ell_i(f)\bigr)\Bigr],
$$

with $\ell_i,\ell'_i$ the losses on the $i$-th real and ghost points. Now the
trick: $S$ and $S'$ are i.i.d., so swapping the $i$-th real and ghost points
leaves the joint distribution unchanged---which means each difference
$\ell'_i-\ell_i$ can be multiplied by an independent random sign
$\varepsilon_i$ *for free*. Splitting the signed sum into its two halves
leaves $2\,E[\sup_f \frac1n\sum_i \varepsilon_i \ell_i(f)] =
2\,\mathfrak R_n(\ell\circ\mathcal F)$: the unknown truth has vanished, and a
correlation with coin flips stands in its place. The final step from this
statement *in expectation* to the high-probability form
:eqref:`eq_mdl-rademacher-bound` is *McDiarmid's inequality*, the extension of
Hoeffding from sums to any function that no single coordinate can move much
(here, changing one sample point moves the supremum by at most $1/n$); it is
stated in exercise 3, where it also puts an error bar on the bootstrap.
Sanity-check the theorem at its ends. A singleton class has $\widehat{\mathfrak R}=0$ (the expectation of
$\frac1n\sum\varepsilon_i f(\mathbf x_i)$ is $0$ with nothing to optimize) and
:eqref:`eq_mdl-rademacher-bound` collapses to one-sided Hoeffding. A finite
class obeys $\mathfrak R_n \le \sqrt{2\log|\mathcal F|/n}$ (Massart's lemma),
recovering :eqref:`eq_mdl-finite-class` up to constants---and for binary
classes the growth-function/VC machinery is exactly a bound on how "finite"
the class effectively is on $n$ points, giving
$\mathfrak R_n\lesssim\sqrt{d_{\mathrm{VC}}\log n / n}$
:cite:`boucheron2005theory`. At the other end, the class of *all* functions
into $[-1,1]$ matches every sign pattern perfectly,
$\widehat{\mathfrak R}=1$, and the bound is vacuous---as it must be, since
that class can memorize anything.

### The Linear Class in Closed Form

Rademacher complexity would be a definition without teeth if it could not be
*computed*. For the class deep learning cares most about---linear functions
with a norm budget---the computation is four lines, and its conclusion is the
punchline of the section.

**Proposition (Rademacher complexity of a norm-bounded linear class).** *Let
$\mathcal F=\{\mathbf x\mapsto\langle\mathbf w,\mathbf x\rangle :
\|\mathbf w\|_2\le B\}$ and suppose the sample satisfies
$\|\mathbf x_i\|_2\le r$ for all $i$. Then*

$$
\widehat{\mathfrak R}_S(\mathcal F) \le \frac{B\,r}{\sqrt n}.
$$
:eqlabel:`eq_mdl-linear-rademacher`

**Proof.** The supremum is explicit: by linearity and Cauchy--Schwarz (with
equality at $\mathbf w$ aligned to the sum),

$$
\sup_{\|\mathbf w\|\le B}\ \frac1n\sum_{i=1}^n
 \varepsilon_i \langle\mathbf w,\mathbf x_i\rangle
 = \sup_{\|\mathbf w\|\le B}\ \frac1n
   \Bigl\langle\mathbf w,\ \sum_i \varepsilon_i\mathbf x_i\Bigr\rangle
 = \frac Bn\,\Bigl\|\sum_{i=1}^n \varepsilon_i\mathbf x_i\Bigr\| .
$$

Jensen's inequality (the square root is concave) moves the expectation inside
the square:

$$
E_{\boldsymbol\varepsilon}\Bigl\|\sum_i\varepsilon_i\mathbf x_i\Bigr\|
 \le \sqrt{E_{\boldsymbol\varepsilon}\Bigl\|\sum_i\varepsilon_i\mathbf x_i\Bigr\|^2}
 = \sqrt{\sum_{i,j} E[\varepsilon_i\varepsilon_j]\,
   \langle\mathbf x_i,\mathbf x_j\rangle}
 = \sqrt{\sum_i \|\mathbf x_i\|^2}
 \le r\sqrt n,
$$

the cross terms vanishing because independent signs have
$E[\varepsilon_i\varepsilon_j]=0$ for $i\ne j$. Multiplying by $B/n$ gives
$Br/\sqrt n$. $\blacksquare$

Now look at what is *absent* from :eqref:`eq_mdl-linear-rademacher`: the
dimension $d$. A linear class over a million features and over ten features
have the *same* capacity bound if their weight norms and data norms match.
**Norm, not parameter count, controls capacity.** This single line is the
theory behind weight decay (:numref:`sec_weight_decay`): shrinking
$\|\mathbf w\|$ shrinks $B$, which shrinks the one term in
:eqref:`eq_mdl-rademacher-bound` the learner can control. Hold on to it---it
is also the key that unlocks double descent in the final section, where the
*norm* of an interpolating solution will fall even as its parameter count
grows.

One bookkeeping step stitches the two propositions together.
:eqref:`eq_mdl-rademacher-bound` wants the complexity of the *loss* class
$\ell\circ\mathcal F$---the functions whose empirical means we actually
compare---while :eqref:`eq_mdl-linear-rademacher` computes that of the
*predictor* class $\mathcal F$. The bridge is the *contraction principle*
:cite:`Boucheron.Lugosi.Massart.2013`: composing every $f\in\mathcal F$ with
one fixed $L$-Lipschitz function $\phi$ multiplies the Rademacher complexity
by at most $L$, $\widehat{\mathfrak R}_S(\phi\circ\mathcal F)\le
L\,\widehat{\mathfrak R}_S(\mathcal F)$. Margin losses---the hinge, the
clipped square, the logistic---are Lipschitz in the prediction, so the loss
class of the norm-bounded linear model inherits the bound
$L\,Br/\sqrt n$ and the norm-controls-capacity conclusion survives the
composition intact.

### Why the Bounds Go Vacuous---and Why the Language Survives

Honesty requires one more paragraph. Apply this machinery to a modern network
and the numbers are useless. :cite:`zhang2021understanding` trained standard
architectures to zero training error on ImageNet-scale data with *randomly
shuffled labels*: the class realized by "this architecture, trained by SGD"
can correlate perfectly with coin flips, so its Rademacher complexity on such
samples is essentially $1$ and :eqref:`eq_mdl-rademacher-bound` certifies
nothing; norm-based refinements, evaluated at the sizes practitioners use,
yield bounds orders of magnitude above the trivial bound of $1$. It is
important to file this correctly: it is a fact about *these bounds*---about
uniform convergence taken over the entire representable class---not about the
framework. The same experiment shows the same network generalizing on real
labels, so what needs explaining is a property of the *reached* solution, not
the reachable set, and the honest modern program is to shrink the class to
"functions the optimizer actually finds on data like this" and measure *its*
complexity. Uniform convergence remains the right language for saying what a
generalization guarantee even is; what failed is the crude choice of
$\mathcal F$, and the last section of this chapter shows---in a model small
enough to solve---exactly how a giant class can reliably deliver small-norm,
well-generalizing solutions.

### Coin Flips in Code

Both halves of the story are measurable. First we estimate
$\widehat{\mathfrak R}_S$ for the norm-bounded linear class by Monte Carlo:
the proof gave the supremum in closed form,
$\frac Bn\|\sum_i\varepsilon_i\mathbf x_i\|$, so we just average that norm
over random sign draws and compare against $Br/\sqrt n$. Then we exhibit the
Zhang phenomenon in miniature: a class rich enough to *interpolate*---here,
minimum-norm least squares on $p=2n$ random features, via the pseudoinverse of
:numref:`subsec_mdl-pseudoinverse`---achieves correlation $1$ with every coin
flip it is shown.

```{.python .input #mdl-concentration-generalization-coin-flips-in-code}
rng = onp.random.default_rng(0)
n, d, B, num_sigma = 50, 20, 1.0, 2000
X = rng.standard_normal((n, d))
X /= onp.linalg.norm(X, axis=1, keepdims=True)     # ||x_i|| = r = 1
signs = rng.choice([-1.0, 1.0], size=(num_sigma, n))
sup_corr = B * onp.linalg.norm(signs @ X, axis=1) / n  # closed-form supremum
print(f'Monte-Carlo Rademacher complexity  = {sup_corr.mean():.4f}')
print(f'bound B*r/sqrt(n)                  = {B / onp.sqrt(n):.4f}')

# The Zhang phenomenon in miniature: an interpolating class hits corr = 1
Phi = rng.standard_normal((n, 2 * n))        # p = 2n features: interpolation
W = onp.linalg.pinv(Phi) @ signs.T           # min-norm fit to every flip set
corr = onp.mean(signs.T * (Phi @ W), axis=0)     # (1/n) sum_i eps_i f(x_i)
w_norm = onp.linalg.norm(W, axis=0).mean()
r_phi = onp.linalg.norm(Phi, axis=1).mean()
print(f'correlation of min-norm fits with coin flips = {corr.mean():.4f}')
print(f'norm it costs: mean ||w|| = {w_norm:.2f};  implied capacity bound '
      f'B*r/sqrt(n) = {w_norm * r_phi / onp.sqrt(n):.2f}')
```

The Monte-Carlo estimate is $0.1390$ against the bound $0.1414$: the
Cauchy--Schwarz/Jensen argument is nearly tight, losing under $2\%$ (the only
slack is Jensen's step, and $\|\sum\varepsilon_i\mathbf x_i\|$ has small
relative fluctuation---concentration again). So a unit-norm linear class on
fifty unit-norm points can fake a correlation of about $0.14$ with pure noise,
and no more: its empirical risks are meaningful. The interpolating class, by
contrast, scores a correlation of exactly $1.0000$ on every one of the two
thousand sign patterns---it is the all-functions endpoint of the theorem, in
twenty features and fifty points. And the price is printed next to it: the
sign-fitting solutions need norm about $1.0$ on features of norm about $10$,
so the smallest linear class containing them has capacity bound
$Br/\sqrt n\approx 1.41 > 1$---vacuous, exactly as the honest paragraph above
said it must be. The two prints together are this section in one screen:
capacity is not what the class *has* but what the data *forces it to spend*.

## Interpolation and Double Descent

### The U-Curve, Revisited

The classical picture of generalization is the U-curve, and this chapter built
it honestly: the bias--variance decomposition
:eqref:`eq_mdl-bias-variance` of :numref:`sec_mdl-statistics` splits the test
error into a falling squared-bias term and a rising variance term as capacity
grows, with a sweet spot between. Uniform convergence tells the same story in
different units---more capacity means more Rademacher complexity to pay for.
Both tellings tacitly assume the interesting regime is the one where the model
*cannot* fit the training data perfectly. Modern practice lives on the other
side. Networks are routinely trained to (near-)zero training error---they
*interpolate*---and past the **interpolation threshold**, where parameters
suffice to fit every training point, the classical curves have nothing more to
say: empirical risk is identically zero for every model in sight, and the
observed test error *falls again* as capacity keeps growing. This second fall
is **double descent** :cite:`Belkin.Hsu.Ma.ea.2019,nakkiran2021deep`.
:numref:`sec_generalization_deep` surveys the phenomenon across deep models;
this section is its mathematical home: we now build the smallest model that
exhibits it, locate the peak exactly, and explain both descents with the tools
already on the table.

### The Minimum-Norm Mechanism

The smallest such model is **random-features regression**. Fix a feature map
built from randomness---ours will be $\phi(\mathbf x)=\mathrm{ReLU}
(\mathbf V\mathbf x)$ with a random frozen $\mathbf V$, a one-hidden-layer
network whose first layer is never trained---and fit only the linear head
$\mathbf w\in\mathbb{R}^p$ by least squares on $n$ training points. The
feature count $p$ is a capacity dial we can sweep straight through the
threshold $p=n$, and the fitted model changes character three times along the
way. Write $\boldsymbol\Phi\in\mathbb{R}^{n\times p}$ for the feature matrix
and fit $\mathbf w=\boldsymbol\Phi^{+}\mathbf y$ with the pseudoinverse of
:numref:`subsec_mdl-pseudoinverse`, which returns the least-squares solution
for $p<n$ and the **minimum-norm interpolant** for $p\ge n$.

*Below the threshold* ($p<n$) the system is overdetermined: the fit cannot
match all $n$ labels, residual noise is averaged rather than reproduced, and
the classical U-curve logic applies verbatim---bias falls and variance rises
with $p$.

*At the threshold* ($p=n$) the feature matrix is square, and interpolation
requires inverting it: $\mathbf w=\boldsymbol\Phi^{-1}\mathbf y$. A random
square matrix is almost surely invertible but almost never *well*
invertible---its smallest singular value $\sigma_{\min}$ is typically tiny,
and the condition-number lens of :numref:`subsec_mdl-condition-number` says
what happens next: the solution acquires a component of size
$\langle\mathbf u_{\min},\mathbf y\rangle/\sigma_{\min}$ along the worst
direction. The labels' noise, which a regression is supposed to average away,
is instead *divided by a near-zero number*. The solution norm explodes and the
test error spikes---this is the peak of double descent, and it is a
conditioning event, not a mystery.

*Beyond the threshold* ($p>n$) interpolation is easy: infinitely many
$\mathbf w$ fit exactly, and the pseudoinverse picks the smallest. Here is the
decisive fact, and with *nested* features (each model uses the first $p$
columns of one master matrix) it is an exact monotonicity: any interpolant
available at $p$ is still available at $p+1$ by padding with a zero, so the
feasible set only grows and

$$
\bigl\|\mathbf w^{(p+1)}_{\min}\bigr\| \le \bigl\|\mathbf w^{(p)}_{\min}\bigr\| :
$$

**more features let the minimum-norm interpolant get smaller.** Now recall the
punchline of :eqref:`eq_mdl-linear-rademacher`: for linear classes the
capacity that matters is the *norm*, not the parameter count. Past the
threshold, growing $p$ grows the nominal parameter count but *shrinks* the
norm the fit actually spends, so the effective capacity of the learned
predictor falls and the test error follows it down. The second descent is not
a violation of the classical theory; it is the classical theory applied to the
right complexity measure.

### Double Descent in Twenty-Five Lines

The showpiece: a full double-descent curve from scratch. Forty training points
from a noisy linear teacher in fifteen dimensions; ReLU random features with
nested columns; the head fit by `pinv`---least squares below the threshold,
minimum-norm interpolation above it. We sweep $p$ from $2$ to $400$ straight
through $p=n=40$, and we track the two quantities the mechanism says to watch:
the test error and $\|\mathbf w\|$. Medians over $20$ independent runs keep
the curve readable.

```{.python .input #mdl-concentration-generalization-double-descent-in-twenty-five-lines}
rng = onp.random.default_rng(0)
n, n_test, d_in, p_max, trials, noise = 40, 500, 15, 400, 20, 0.1
p_grid = onp.array([2, 5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 45, 50, 60,
                    80, 120, 200, 400])
test_mse, train_mse, w_norm = (onp.zeros((trials, len(p_grid)))
                               for _ in range(3))
for tr in range(trials):
    beta = rng.standard_normal(d_in)
    beta /= onp.linalg.norm(beta)                    # teacher: y = <beta, x>
    X, X_te = (rng.standard_normal((m, d_in)) for m in (n, n_test))
    y = X @ beta + noise * rng.standard_normal(n)    # noisy training labels
    y_te = X_te @ beta
    V = rng.standard_normal((p_max, d_in)) / onp.sqrt(d_in)
    Phi, Phi_te = onp.maximum(X @ V.T, 0), onp.maximum(X_te @ V.T, 0)
    for j, p in enumerate(p_grid):                   # nested feature prefixes
        w = onp.linalg.pinv(Phi[:, :p]) @ y  # least-squares/min-norm solution
        test_mse[tr, j] = onp.mean((Phi_te[:, :p] @ w - y_te) ** 2)
        train_mse[tr, j] = onp.mean((Phi[:, :p] @ w - y) ** 2)
        w_norm[tr, j] = onp.linalg.norm(w)
med = [onp.median(a, axis=0) for a in (test_mse, train_mse, w_norm)]
for j, p in enumerate(p_grid):
    print(f'p={p:3d}  test MSE={med[0][j]:9.4f}  train MSE={med[1][j]:9.2e}'
          f'  ||w||={med[2][j]:7.2f}')
d2l.plot(p_grid, [med[0], med[2]], 'number of random features p',
         'median over 20 runs', legend=['test MSE', '||w|| of the fit'],
         xscale='log', yscale='log')
```

Read the table against the mechanism, feature by feature. *The classical
regime plays out first*: test error falls from $1.04$ at $p=2$ to its
classical sweet spot of about $0.68$ around $p=15$, then creeps up as
variance grows---the familiar U. *The interpolation threshold announces
itself in the train column*: at $p=40=n$ the training error drops from
$5\times10^{-3}$ to $10^{-28}$---exact interpolation, up to floating
point---and precisely there the test error erupts to $33.6$, fifty times its
classical minimum, while $\|\mathbf w\|$ jumps to $18.5$: the near-singular
square system dividing noise by $\sigma_{\min}$, just as promised. *Then the
second descent*: from $p=42$ onward the norm falls monotonically---$4.96$,
$3.64$, $2.42$, $1.70$, $1.18$, down to $0.39$ at $p=400$ (the nesting
guarantees this)---and the test error tracks it down through $0.65$ at
$p=50$ and $0.18$ at $p=80$ to $0.060$ at $p=400$: *ten times better than the
best underparameterized model*. Note also which curve did *not* move: train
error is an identical $0$ everywhere past the threshold, so no
empirical-risk-based criterion can tell these models apart---only the norm
does, exactly as :numref:`sec_generalization_deep`'s survey said of deep
networks and as the Rademacher calculation predicts. The best model in this
entire experiment is the most overparameterized one, fitting noisy data
*exactly*, with ten times more parameters than data points.

### Benign Overfitting

One puzzle remains inside the mechanism. The $p=400$ model interpolates its
noisy labels---it reproduces every corrupted $y_i$ exactly---yet its test
error is the best on the table. When does fitting noise *not* hurt? The
minimum-norm solution splits across the spectrum of the feature matrix: the
*signal* is captured along the few directions with large singular values,
while interpolating the residual *noise* is distributed---because minimizing
the norm spreads it---across the many directions with small singular values,
where each contaminates predictions at new points only faintly. Overfitting is
**benign** when the spectrum has this shape: a few strong directions to carry
the signal, plus a large reservoir of weak directions that absorb and
*average out* the noise, in effect performing implicit regularization without
a regularizer. :cite:`Bartlett.Long.Lugosi.Tsigler.2020` make this exact for
linear regression, characterizing benignity via two effective ranks of the
covariance, and :cite:`Belkin.Hsu.Ma.ea.2019` frame the modern regime it
explains; :cite:`Bartlett.Montanari.Rakhlin.2021` survey the fast-growing
theory. For deep networks the picture is instructive but not settled: features
there are *learned*, reshaping the spectrum during training, and a theory that
predicts a given network's test error from first principles remains open.

## Summary

* The *Chernoff method*---Markov's inequality applied to $e^{\lambda X}$,
  then optimized over $\lambda$---converts a bound on the moment generating
  function into an exponential tail bound; for sums of independent variables
  the MGF factorizes, which is where the exponential-in-$n$ decay comes from.
* *Hoeffding's lemma* bounds the MGF of a centered variable on $[a,b]$ by
  $e^{\lambda^2(b-a)^2/8}$ (proof: the second derivative of the log-MGF is a
  tilted variance, at most $(b-a)^2/4$), and *Hoeffding's inequality* follows:
  $P(|\bar X-E\bar X|\ge t)\le 2e^{-2nt^2/(b-a)^2}$---the finite-sample bound
  behind the main book's test-set arithmetic, exponentially sharper than
  Chebyshev's $1/n$. Inverted, it is a confidence interval with half-width
  $(b-a)\sqrt{\log(2/\delta)/2n}$ whose coverage holds at every $n$.
  *Sub-Gaussian* variables are those obeying such an MGF
  bound; *Bernstein's inequality* refines the constant using the true
  variance and covers the sub-exponential case (squares, norms).
* In high dimension, concentration is geometry: $\|\mathbf x\|/\sqrt d$
  concentrates at $1$ with failure probability $2e^{-d\varepsilon^2/8}$, so a
  Gaussian is a *thin shell*, random directions are *near-orthogonal*
  ($\cos\approx 1/\sqrt d$), pairwise distances concentrate (degrading
  nearest-neighbor contrast), and unit-variance initialization keeps
  activation norms pinned across layers for essentially every draw.
* Hoeffding certifies one *pre-chosen* function; a learner chooses *after*
  seeing data, so guarantees must hold *uniformly* over the class. For finite
  classes the union bound gives deviation
  $\sqrt{\log(2|\mathcal F|/\delta)/2n}$: $\log|\mathcal F|$ is the price of
  choice, and test-set reuse is this bound applied to the analyst.
* *Rademacher complexity*---how well a class can correlate with random
  signs---bounds uniform deviations:
  $R(f)\le\hat R(f)+2\mathfrak R_n(\mathcal F)+\sqrt{\log(1/\delta)/2n}$. For
  the linear class $\{\|\mathbf w\|\le B\}$ on data with $\|\mathbf x\|\le r$
  it computes to $Br/\sqrt n$: *norm, not parameter count, is capacity*. On
  classes that interpolate, the complexity is $\approx 1$ and the bounds are
  vacuous---a fact about the bounds, not the framework.
* *Double descent*: sweeping random features through the interpolation
  threshold $p=n$, test error follows the classical U, spikes at $p=n$ (a
  near-singular system divides noise by $\sigma_{\min}$), then descends again
  as the minimum-norm interpolant's norm---the true capacity---shrinks with
  every added feature. Interpolating noise is *benign* when a few strong
  spectral directions carry the signal and many weak ones absorb the noise.

## Exercises

1. Prove that a Rademacher variable is sub-Gaussian with variance proxy $1$:
   show $\cosh\lambda\le e^{\lambda^2/2}$ by comparing the two Taylor series
   term by term ($(2k)!\ge 2^k k!$). Conclude via the Chernoff method that an
   average of $n$ fair random signs satisfies
   $P(|\bar\varepsilon|\ge t)\le 2e^{-nt^2/2}$.
2. The proof of :eqref:`eq_mdl-hoeffding` assumed a common range $[a,b]$.
   Redo it for independent $X_i\in[a_i,b_i]$ to obtain
   $P(|\bar X-E\bar X|\ge t)\le
   2\exp\bigl(-2n^2t^2/\sum_i(b_i-a_i)^2\bigr)$, and check that it reduces to
   :eqref:`eq_mdl-hoeffding` when the ranges agree. This weighted version is
   the one the near-orthogonality argument used, with weights $u_i$.
3. **McDiarmid's inequality** extends Hoeffding from sums to arbitrary
   functions with *bounded differences*: if changing the $i$-th argument of
   $g(x_1,\ldots,x_n)$ moves its value by at most $c_i$, then
   $P(|g - E[g]|\ge t)\le 2\exp\bigl(-2t^2/\sum_i c_i^2\bigr)$. Take this
   statement on faith (its proof needs martingales). (i) Recover Hoeffding's
   inequality from it. (ii) The bootstrap of :numref:`sec_mdl-statistics`
   computes a statistic of $n$ data points; argue that if that statistic has
   bounded differences $c_i=c/n$, the bootstrap's *ideal* target (the
   statistic's deviation from its mean) already concentrates at rate
   $e^{-2nt^2/c^2}$, and check the median of a sample does *not* have small
   bounded differences in general.
4. Compute the Rademacher complexity of the $\ell_1$-ball class
   $\mathcal F=\{\mathbf x\mapsto\langle\mathbf w,\mathbf x\rangle:
   \|\mathbf w\|_1\le B\}$ on data with $\|\mathbf x_i\|_\infty\le r$. The
   supremum step now uses the duality between the $\ell_1$ and $\ell_\infty$
   norms, giving $\frac Bn E\|\sum_i\varepsilon_i\mathbf x_i\|_\infty$; bound
   the expected max of $d$ sub-Gaussian coordinates to conclude
   $\widehat{\mathfrak R}_S\le Br\sqrt{2\log(2d)/n}$. Why does dimension now
   enter, and only logarithmically?
5. Rerun the double-descent sweep with ridge regression,
   $\mathbf w_\lambda=(\boldsymbol\Phi^\top\boldsymbol\Phi+
   \lambda\mathbf I)^{-1}\boldsymbol\Phi^\top\mathbf y$, for
   $\lambda\in\{10^{-8},10^{-4},10^{-2},1\}$. Watch the interpolation peak
   melt as $\lambda$ grows, and explain why using
   :numref:`sec_mdl-numerical-stability-conditioning`: the ridge term lifts
   $\sigma_{\min}^2$ to $\sigma_{\min}^2+\lambda$, capping the noise
   amplification. Verify numerically that as $\lambda\to0$ the ridge solution
   converges to the pseudoinverse (minimum-norm) solution.
6. In the double-descent experiment, vary the label noise
   ($\textrm{noise}\in\{0,\ 0.1,\ 0.5,\ 1.0\}$) and measure the height and
   location of the peak. Confirm that the peak stays at $p=n$ (its location
   is a rank condition, independent of the labels) while its height grows
   with the noise variance, and explain both observations with the
   $\sigma_{\min}$ mechanism. What happens to the peak when
   $\textrm{noise}=0$---and why is it not entirely gone?
7. Use :eqref:`eq_mdl-norm-concentration` to find the smallest dimension $d$
   at which the shell $(1\pm 0.01)\sqrt d$ is guaranteed to hold at least
   $99\%$ of a standard Gaussian's mass, then estimate the true smallest $d$
   by simulation. How large is the gap, and which side of the story (rate or
   constant) does it come from?
8. An analyst evaluates $m$ models on one validation set of $n$ points
   (losses in $[0,1]$) and reports the best score. Apply
   :eqref:`eq_mdl-finite-class` with $\mathcal F$ the set of tried models to
   bound how *optimistic* the reported score can be, and evaluate the bound
   at $m=10$, $10^3$, and $10^6$ with $n=10{,}000$ and $\delta=0.05$. At what
   $m$ does the guaranteed accuracy degrade past $\pm 0.02$? Compare with the
   Bonferroni discussion of multiple testing in :numref:`sec_mdl-statistics`:
   which bound is doing the same job, and where do the two analyses differ?

[Discussions](https://d2l.discourse.group/t/concentration-and-generalization)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.5]{.kicker}

Why averages can be trusted, and when learners cannot<br>**exponential tails · high dimension · uniform convergence · double descent**.
:::
:::

::: {.slide title="Polynomial tails are not enough"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Everything so far decays polynomially: Chebyshev caps a sample mean's
miss probability at $\sigma^2/(nt^2)$ — certifying $10^{-6}$ costs a
*million* times the data of certifying $10^0$. The truth for bounded
data is **exponential** in $n$, and this section proves it.

The stakes: the main book bounded test-set error with Hoeffding's
inequality *on faith* (§4.6). Here we pay the debt — then follow the
same machinery into high dimension and out to generalization itself.
:::

::: {.col .fig}
@fig:mdl-prob-chebyshev
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[From Chebyshev to Chernoff]{.dtitle}

[one trick: Markov's inequality, after an exponential]{.dsub}
:::
:::

::: {.slide title="The Chernoff method sees every moment at once"}
[Chernoff]{.kicker}

Chebyshev sees only the second moment; a tail decaying like $e^{-ct^2}$
must see them all. The MGF $M(\lambda)=E[e^{\lambda X}]$ packages every
moment, and Markov applied to $e^{\lambda X}$ (a monotone transform)
converts a bound on it into a tail:

$$P(X \ge t) \;\le\; \inf_{\lambda>0}\; e^{-\lambda t}\, M(\lambda).$$

. . .

::: {.d2l-note .rule}
The power move: for a **sum** of independent terms the MGF *factors*, so
the exponent grows linearly in $n$ — optimizing $\lambda$ turns that into
exponential decay. Everything in this section is this display plus a good
MGF bound.
:::
:::

::: {.slide title="Hoeffding's lemma: bounded ⇒ sub-Gaussian"}
[Hoeffding]{.kicker}

For centered $X \in [a,b]$:
$\;E[e^{\lambda X}] \le \exp\bigl(\lambda^2(b-a)^2/8\bigr)$ — an MGF *no
worse than a Gaussian's* with $\sigma = (b-a)/2$.

. . .

*Proof idea.* $\psi(\lambda) = \log E[e^{\lambda X}]$ has
$\psi(0)=\psi'(0)=0$, and $\psi''(\lambda)$ is the variance of $X$ under
an exponentially **tilted** distribution — still supported in $[a,b]$, so
$\psi'' \le (b-a)^2/4$ everywhere. Taylor with remainder gives
$\psi(\lambda) \le \lambda^2(b-a)^2/8$. $\blacksquare$

::: {.d2l-note}
The one-line fact doing the work: any variable supported in $[a,b]$ has
variance at most $\bigl(\tfrac{b-a}{2}\bigr)^2$.
:::
:::

::: {.slide title="Hoeffding's inequality, in four lines"}
[Hoeffding]{.kicker}

Independent $X_i \in [a,b]$, $\bar X$ their average:

$$P\bigl(|\bar X - E[\bar X]| \ge t\bigr)
\;\le\; 2\exp\!\left(-\frac{2nt^2}{(b-a)^2}\right).$$

. . .

*Proof.* Chernoff + independence (MGFs multiply) + the lemma give
exponent $-\lambda n t + n\lambda^2(b-a)^2/8$ — a parabola in $\lambda$,
minimized at $\lambda = 4t/(b-a)^2$ with value $-2nt^2/(b-a)^2$. The
other tail by symmetry; union bound supplies the $2$. $\blacksquare$

::: {.d2l-note .rule}
No Gaussianity, no variance estimate, no asymptotics — only boundedness
and independence, at **every finite $n$**.
:::
:::

::: {.slide title="The Chapter 4 debt, paid"}
[Hoeffding]{.kicker}

Inverted, Hoeffding is a *finite-sample confidence interval*: with
probability $1-\delta$,

$$|\bar X - E[\bar X]| \;\le\; (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}.$$

. . .

§4.6 demanded a test-set error estimate within $t=0.01$ at
$\delta = 0.05$ and quoted "roughly 18,500 examples." Solve
$2e^{-2nt^2} \le \delta$:

$$n = \frac{\log(2/\delta)}{2t^2} \approx 18{,}445.$$

A citation is now a theorem — and unlike the Gaussian interval of §25.4,
its coverage is a guarantee at every $n$, not an asymptote.
:::

::: {.slide title="Names for the well-behaved: sub-Gaussian and friends"}
[The zoo]{.kicker}

$X$ is **sub-Gaussian** with proxy $\sigma^2$ if
$E[e^{\lambda(X-EX)}] \le e^{\lambda^2\sigma^2/2}$ — Chernoff then gives
$P(|X-EX| \ge t) \le 2e^{-t^2/(2\sigma^2)}$. Three families:
Gaussians (proxy $\sigma^2$, exactly), bounded variables (proxy
$(b-a)^2/4$, the lemma), random signs (proxy $1$).

. . .

::: {.d2l-note}
*Squares* of Gaussians — hence squared norms — are only
**sub-exponential**: Gaussian rate for small deviations, $e^{-ct}$ for
large. **Bernstein's inequality** covers them and spends the *true*
variance where Hoeffding must budget for the worst case.
:::
:::

::: {.slide title="The tail race: rates always beat constants" layout="tight"}
[Measured — the fair coin's exact tail vs. both bounds]{.kicker}

@!mdl-concentration-generalization-the-tail-race-in-code

By $n=1000$ Chebyshev is off by **seven orders of magnitude**; by
$n=5000$, forty.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Probability in high dimension]{.dtitle}

[thin shells and orthogonal-by-default directions]{.dsub}
:::
:::

::: {.slide title="A high-dimensional Gaussian is a thin shell"}
[Geometry]{.kicker}

$\|\mathbf x\|^2$ is a sum of $d$ mean-$1$ terms — a sample-mean problem
with sub-exponential summands, and the Bernstein-flavored Chernoff gives

$$P\!\left(\Bigl|\tfrac{\|\mathbf x\|}{\sqrt d} - 1\Bigr| \ge \varepsilon\right)
\;\le\; 2\,e^{-d\varepsilon^2/8}.$$

. . .

The density peaks at the origin, but shell volume grows like $r^{d-1}$:
the fight is settled at $r \approx \sqrt d$, and the shell's thickness is
$O(1)$ — *independent of $d$*.

::: {.d2l-note}
At $d=784$ a typical draw has norm within a few percent of $28$; a draw
below norm $14$ is rarer than $2e^{-24}$. **The mode is not the mass.**
:::
:::

::: {.slide title="Random directions are orthogonal by default"}
[Geometry]{.kicker}

For a unit $\mathbf u$, the inner product
$\langle\mathbf x,\mathbf u\rangle = \sum_i u_i x_i$ is *Hoeffding
again* (ranges $|u_i|$, $\sum u_i^2 = 1$):

$$P\bigl(|\langle\mathbf x,\mathbf u\rangle| \ge s\bigr) \le 2e^{-s^2/2},$$

so the cosine is $O(1/\sqrt d)$ with exponentially high probability.

. . .

::: {.d2l-note .rule}
Among $n$ points there are $< n^2$ pairs, and a **union bound** flattens
*every* pairwise distance into a $(1\pm\varepsilon)$ band once
$d \gtrsim \tfrac{8}{\varepsilon^2}\log\tfrac{2n^2}{\delta}$ — dimension
only logarithmic in $n$. File the move away: one tail $\to$ many objects,
at log cost. It is how learning bounds are built.
:::
:::

::: {.slide title="Measuring the shell"}
[Measured]{.kicker}

What *fraction* of the mass sits in the $\pm10\%$ shell, and what happens
to nearest-neighbor contrast:

@!mdl-concentration-generalization-measuring-the-shell

$d=2$ really is a fuzzy ball (under $15\%$ in the shell, contrast $29$);
at $d=2000$ **not one of ten thousand draws** left the shell and the
farthest of $200$ points is only $8\%$ farther than the nearest.

::: {.d2l-note}
Why $1/\sqrt d$ initialization holds *for essentially every draw*, why
cosine similarity is informative, why latent interpolation follows the
sphere — and why real embeddings must have structure to be searchable.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Uniform convergence]{.dtitle}

[the function chosen after the data]{.dsub}
:::
:::

::: {.slide title="The function chosen after the data"}
[The pivot]{.kicker}

Hoeffding certifies **one fixed function**: choose $f$, *then* draw the
sample. A test set works precisely because it respects this order.

. . .

A *learner* violates it by construction: it searches $\mathcal F$ and
returns the $\hat f$ that looks best *on the sample it saw* — a minimum
of fluctuating quantities, selected exactly where the fluctuation
flatters. The quantity that must concentrate is the worst case,

$$\sup_{f\in\mathcal F}\;\bigl|\hat R(f) - R(f)\bigr|.$$

::: {.d2l-note .warn}
Test-set **reuse** is the same trap in different clothes: an analyst who
tries many models on one test set *is* a learner over the class of models
tried.
:::
:::

::: {.slide title="log |F| is the price of choice"}
[Finite classes]{.kicker}

For finite $\mathcal F$, Hoeffding + a union bound over $|\mathcal F|$
bad events: with probability $1-\delta$, simultaneously for every $f$,

$$\bigl|\hat R(f) - R(f)\bigr|
\;\le\; \sqrt{\frac{\log(2|\mathcal F|/\delta)}{2n}}.$$

. . .

Each bit of selection freedom costs one bit's worth of sample — but under
the square root: $2{,}000$ hyperparameter configs on a $10{,}000$-point
validation set move the guarantee only from $0.014$ to $0.023$. A million
*adaptive* leaderboard submissions quietly triple it.
:::

::: {.slide title="Rademacher complexity: correlate with coin flips"}
[Infinite classes]{.kicker}

Real classes are infinite, so ask a question a learner can act on: how
well can $\mathcal F$ correlate with **pure noise**?

$$\widehat{\mathfrak R}_S(\mathcal F)
= E_{\boldsymbol\varepsilon}\Bigl[\sup_{f\in\mathcal F}
\tfrac1n\textstyle\sum_i \varepsilon_i f(\mathbf x_i)\Bigr],
\qquad
R(f) \le \hat R(f) + 2\,\mathfrak R_n(\mathcal F)
+ \sqrt{\tfrac{\log(1/\delta)}{2n}}.$$

A class that can chase coin flips can chase the noise in real labels; one
that cannot has trustworthy empirical risks.

::: {.d2l-note}
The factor $2$ is **symmetrization**: a ghost sample plus the swap trick
conjures the random signs out of a statement that contains none;
McDiarmid lifts it to high probability.
:::
:::

::: {.slide title="The linear class: capacity is Br/√n"}
[The punchline]{.kicker}

For $\mathcal F = \{\mathbf x \mapsto \langle\mathbf w,\mathbf x\rangle :
\|\mathbf w\| \le B\}$ on data with $\|\mathbf x_i\| \le r$, the supremum
is Cauchy--Schwarz in closed form, and independent signs kill the cross
terms:

$$\widehat{\mathfrak R}_S
= \frac Bn\,E\Bigl\|\sum_i \varepsilon_i \mathbf x_i\Bigr\|
\;\le\; \frac{B\,r}{\sqrt n}.$$

. . .

::: {.d2l-note .rule}
Look at what is **absent**: the dimension $d$. A million features and ten
features have the same capacity if the norms match. **Norm, not parameter
count, controls capacity** — the theory behind weight decay, and the key
to double descent below.
:::
:::

::: {.slide title="Coin flips in code"}
[Measured]{.kicker}

Monte-Carlo Rademacher complexity of the unit-norm linear class — and
the Zhang phenomenon in miniature, an interpolating class fed
$2{,}000$ sets of coin flips:

@!mdl-concentration-generalization-coin-flips-in-code

The bound $0.1414$ is within $2\%$ of the truth. The interpolating class
scores correlation $1.0000$ on *every* flip set — and its own printout
prices the damage: the smallest linear class containing those fits has
$Br/\sqrt n \approx 1.41 > 1$. Vacuous, as it must be.
:::

::: {.slide title="Why the bounds go vacuous — and what survives"}
[Honesty]{.kicker}

Zhang et al. trained standard architectures to zero training error on
**randomly shuffled labels**: the class "this architecture, trained by
SGD" correlates perfectly with coin flips, so its Rademacher complexity
is $\approx 1$ and the bound certifies nothing.

. . .

File it correctly: a fact about *these bounds* — uniform convergence over
the entire representable class — not about the framework. The same
network generalizes on real labels, so what needs explaining is the
**reached solution**, not the reachable set.

::: {.d2l-note}
Capacity is not what the class *has* but what the data *forces it to
spend* — the next act makes that exact in a model small enough to solve.
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Interpolation and double descent]{.dtitle}

[past the threshold, the norm takes over]{.dsub}
:::
:::

::: {.slide title="The U-curve assumes you cannot fit"}
[Double descent]{.kicker}

Bias–variance and uniform convergence both tacitly assume the model
*cannot* fit the training data perfectly. Modern practice lives on the
other side: past the **interpolation threshold** $p = n$, empirical risk
is identically zero for every model in sight — and the observed test
error *falls again* as capacity grows.

. . .

The smallest model that shows it: **random-features regression** —
$\phi(\mathbf x) = \mathrm{ReLU}(\mathbf V\mathbf x)$ with $\mathbf V$
random and frozen, head fit by the pseudoinverse: least squares below the
threshold, the **minimum-norm interpolant** above it.
:::

::: {.slide title="The peak is a conditioning event, not a mystery"}
[Double descent]{.kicker}

At $p = n$ the feature matrix is square and almost never *well*
invertible: interpolation divides the labels' noise by a tiny
$\sigma_{\min}$, the norm explodes, the test error spikes.

. . .

Beyond it, nested features give an exact monotonicity — any interpolant
at $p$ pads with a zero to one at $p+1$, so

$$\bigl\|\mathbf w^{(p+1)}_{\min}\bigr\| \;\le\; \bigl\|\mathbf w^{(p)}_{\min}\bigr\| :$$

**more features let the minimum-norm interpolant get smaller.** Recall
$Br/\sqrt n$: the capacity that matters is the norm, so effective
capacity *falls* as $p$ grows. The second descent is classical theory
applied to the right complexity measure.
:::

::: {.slide title="Double descent in twenty-five lines" layout="tight"}
[Measured — forty noisy points, ReLU random features, swept through p = n]{.kicker}

@!mdl-concentration-generalization-double-descent-in-twenty-five-lines
:::

::: {.slide title="Only the norm can tell them apart"}
[Double descent]{.kicker}

Read the sweep's numbers against the mechanism: at $p = n = 40$ the
train error hits $10^{-28}$ — exact interpolation — and the test error
erupts to **33.6**, fifty times the classical minimum, with
$\|\mathbf w\| = 18.5$. Then the norm falls monotonically to $0.39$ and
the test error follows it down to **0.060** at $p=400$: *ten times
better than the best underparameterized model*, from a model that fits
noisy data exactly.

. . .

Past the threshold the train error is an identical $0$ everywhere: no
empirical-risk criterion distinguishes the spiky $p=42$ model from the
excellent $p=400$ one. The norm does — exactly as the Rademacher
calculation predicts.
:::

::: {.slide title="Benign overfitting: when fitting noise is free"}
[Double descent]{.kicker}

The $p=400$ model reproduces every *corrupted* label exactly, yet tests
best. The min-norm solution splits across the spectrum: a few strong
singular directions carry the *signal*, and minimizing the norm spreads
the interpolated *noise* across the many weak directions, where it
barely contaminates new predictions.

. . .

Implicit regularization without a regularizer: overfitting is **benign**
when the spectrum offers a few strong directions for the signal and a
large reservoir of weak ones to absorb the noise.

::: {.d2l-note}
Made exact for linear regression (two effective ranks of the covariance);
for deep networks — where the features are *learned* and reshape the
spectrum — the theory is instructive but open.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Chernoff:** Markov on $e^{\lambda X}$; independent sums factor the
  MGF → exponential tails.
- **Hoeffding:** $2e^{-2nt^2/(b-a)^2}$ pays §4.6's 18,500-example debt at
  every finite $n$.
- **High dimension:** thin shells, orthogonal-by-default directions.
:::

::: {.col}
- **Choice has a price:** $\log|\mathcal F|$, then Rademacher; the linear
  class costs $Br/\sqrt n$ — **norm, not parameter count**.
- On interpolating classes the bounds are honestly **vacuous** (Zhang).
- **Double descent:** a $\sigma_{\min}$ spike, then the min-norm
  interpolant's norm falls — overfitting can be benign.
:::
:::

::: {.d2l-note}
One inequality, compounded: Hoeffding certifies the test set, the union
bound prices the choice, and the norm is what generalization charges for.
:::
:::
