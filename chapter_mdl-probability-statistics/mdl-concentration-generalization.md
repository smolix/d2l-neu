# Concentration and Generalization
:label:`sec_mdl-concentration-generalization`

Chebyshev's inequality :eqref:`eq_mdl-chebyshev` bounds deviations of a sample
mean with a probability that decreases as $1/n$. For bounded data, Hoeffding's
inequality gives exponential decay in $n$. We derive this inequality and use
related concentration arguments to study norms and angles in high dimensions.
For learning problems, a bound for one fixed function is insufficient because
the function is selected after observing the data. Uniform convergence extends
the guarantee to finite classes and then to infinite classes through
Rademacher complexity. The section concludes by comparing these classical
bounds with interpolation and double descent, emphasizing the role of solution
norms in controlling capacity.

Every computation in this section is a matter of tails, norms, and least
squares, so the worked cells below use plain NumPy, imported as `onp`, with
`d2l` for plotting.

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
import numpy as onp  # the worked cells compute with plain NumPy
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import numpy as np
import numpy as onp  # the worked cells compute with plain NumPy
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import numpy as onp  # the worked cells compute with plain NumPy
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
import numpy as np
import numpy as onp  # the worked cells compute with plain NumPy
```

## From Chebyshev to Chernoff

### Polynomial and Exponential Tail Bounds

Recall the inequalities derived in :numref:`sec_mdl-random_variables`. Markov's
inequality :eqref:`eq_mdl-markov` turns a mean into a tail bound. Applying it
to the squared deviation gives Chebyshev's inequality
:eqref:`eq_mdl-chebyshev`. Applied to a sample mean $\bar X$ of $n$ i.i.d.
draws with variance $\sigma^2$, Chebyshev gives

$$
P\bigl(|\bar X - \mu| \ge t\bigr) \le \frac{\sigma^2}{n\,t^2},
$$

which was enough to prove consistency in :numref:`sec_mdl-statistics`. To reduce
the upper bound from $10^{0}$ to $10^{-6}$ at fixed $t$, however, $n$ must
increase by a factor of one million:
the bound decays only like $1/n$. Meanwhile the exact tail of, say, a fair
coin's head frequency decreases *exponentially* fast. The calculation below
shows that Chebyshev's bound exceeds the exact probability by eight orders of magnitude at
$n=1{,}000$ and about forty-three at $n=5{,}000$. The
Gaussian approximations in :numref:`sec_mdl-statistics`, including the $z$-test and
the $1.96\,\hat\sigma/\sqrt n$ interval, reflect this faster rate because Gaussian
tails decay like $e^{-t^2/2}$, but it gets the rate from an approximation that
is only exact as $n\to\infty$. For bounded variables, we can instead obtain a
Gaussian-shaped rate with finite-sample validity. One standard route applies
Markov's inequality to an exponential transform. This requires the relevant
exponential moments to be finite; martingale and bounded-difference methods
provide other routes in related settings.

### The Chernoff Method

The *moment generating function* (MGF) of a random variable $X$ is
$M(\lambda)=E[e^{\lambda X}]$ at values of $\lambda$ for which this expectation
is finite. If it is finite in a neighborhood of zero, derivatives at zero
recover moments under standard interchange conditions. The *Chernoff method*
is Markov's inequality
applied not to $X$ but to $e^{\lambda X}$ (a monotone transform, so the event
is unchanged), followed by an optimization over the free parameter: for any
$\lambda>0$,

$$
P(X \ge t)
 = P\bigl(e^{\lambda X} \ge e^{\lambda t}\bigr)
 \le e^{-\lambda t}\,E\bigl[e^{\lambda X}\bigr],
\qquad\textrm{hence}\qquad
P(X \ge t) \le \inf_{\lambda > 0}\, e^{-\lambda t}\, M(\lambda).
$$
:eqlabel:`eq_mdl-chernoff`

Its advantage is especially clear when $X$ is a *sum* of
independent terms: the MGF of a sum factors into a *product* of MGFs (the
expectation of a product of independent variables is the product of
expectations), so the exponent in :eqref:`eq_mdl-chernoff` grows linearly in
$n$, and optimizing $\lambda$ then turns that linear-in-$n$ exponent into the
desired exponential decay. The remaining task is to supply a bound on the MGF.
For bounded variables, Hoeffding's lemma provides such a bound.

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
$p_\lambda(x) \propto e^{\lambda x}\,p(x)$, a genuine probability
distribution still supported inside $[a,b]$. Differentiating once more,

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
Gaussian's* with standard deviation $(b-a)/2$: compare the Gaussian identity
$E[e^{\lambda X}]=e^{\lambda^2\sigma^2/2}$ for $X\sim\mathcal N(0,\sigma^2)$,
an integral done by completing the square. Feeding the lemma into
the Chernoff method yields Hoeffding's inequality, in four lines.

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
argument to $-Z_i$ bounds the lower tail identically, and the *union bound*
(the probability that at least one of several events occurs is at most the
sum of their probabilities) applied to the two tails supplies the factor $2$.
$\blacksquare$

This is Hoeffding's inequality :cite:`Hoeffding.1963`.
:numref:`chap_classification_generalization` used
:eqref:`eq_mdl-hoeffding` (with losses in $[0,1]$, so $b-a=1$) to certify a
test-set estimate of the error rate: demanding
$P(|\epsilon_{\mathcal D}(f)-\epsilon(f)|\ge 0.01)\le 0.05$ and solving
$2e^{-2nt^2}\le\delta$ for $n$ gives $n = \log(2/\delta)/(2t^2) \approx
18{,}445$: the "roughly 18,500 examples" quoted there, now a theorem rather
than a citation. Note what the bound does *not* need: no Gaussianity, no
variance estimate, no asymptotics. It needs only boundedness and independence,
and it holds at every finite $n$.

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
the Gaussian interval :eqref:`eq_mdl-gauss_confidence`. But where that
interval's coverage was asymptotic (and, as the coverage audit there measured,
slightly optimistic at finite $n$), this one is a guarantee at every $n$. The
trade-off is the constant: Hoeffding allows for the most adversarial
distribution
on $[a,b]$, so the range $b-a$ stands where the Gaussian interval enjoys the
estimated $\hat\sigma$, and when the true spread is much smaller than the
range the interval is correspondingly conservative. Bernstein's inequality,
below, is designed to recover that slack. Keep
:eqref:`eq_mdl-hoeffding-interval` in view; it is the statement we will
upgrade from one function to entire function classes in the second half of
the section.

### Sub-Gaussian and Sub-Exponential Variables

The proof used only one property of the summands: an MGF bounded by a
Gaussian's. That property is the useful abstraction for the next derivations.
A random variable $X$ is **sub-Gaussian** with
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
sub-Gaussian with proxy $(b-a)^2/4$; that is Hoeffding's lemma. And a
*Rademacher* variable, a uniform random sign $\varepsilon\in\{-1,+1\}$, is
sub-Gaussian with proxy $1$: $E[e^{\lambda\varepsilon}]=\cosh\lambda\le
e^{\lambda^2/2}$, a term-by-term comparison of Taylor series left as
exercise :exref:`ex_mdl-concentration-generalization-random-signs`. Sub-Gaussianity
is closed under averaging: if $X_1,\ldots,X_n$ are independent with proxy
$\sigma^2$, then $\bar X$ is sub-Gaussian with proxy $\sigma^2/n$ (the MGF
product argument again), so *every* sub-Gaussian average concentrates at the
Hoeffding rate.

Not everything is sub-Gaussian. The *square* of a Gaussian (a $\chi^2$
variable, hence squared norms of Gaussian vectors) has a right tail decaying
like $e^{-t/2}$, too heavy for :eqref:`eq_mdl-subgaussian`: its MGF is finite
only for $\lambda<\tfrac12$, so no bound of the form
:eqref:`eq_mdl-subgaussian`, which must hold for *all* $\lambda$, is possible.
Such variables are called **sub-exponential**: $X$ is sub-exponential with
parameters $(\nu,\alpha)$ if

$$
E\bigl[e^{\lambda (X - E[X])}\bigr] \le \exp\!\left(\frac{\lambda^2\nu^2}{2}\right)
\qquad\textrm{for } |\lambda| \le \frac{1}{\alpha},
$$
:eqlabel:`eq_mdl-subexponential`

a Gaussian-type MGF bound restricted to a window of $\lambda$. The Chernoff
optimization then splits at the window's edge: small deviations use an
interior $\lambda$ and get a Gaussian rate, while large deviations clamp
$\lambda$ at $1/\alpha$ and get an exponential (not squared-exponential)
rate. **Bernstein's inequality** packages both regimes: if the $X_i$ are
independent with
$\operatorname{Var}(X_i)\le\sigma^2$ and $|X_i - E[X_i]|\le M$, then

$$
P\bigl(|\bar X - E[\bar X]| \ge t\bigr)
 \le 2\exp\!\left(-\frac{n t^2}{2\bigl(\sigma^2 + M t/3\bigr)}\right),
$$
:eqlabel:`eq_mdl-bernstein`

which we state without proof :cite:`Boucheron.Lugosi.Massart.2013`. Read it as
the variance-aware refinement of Hoeffding: for small $t$ the denominator is
essentially $2\sigma^2$ and the bound matches the CLT's Gaussian tail with the
*true* variance, often far smaller than the worst case $(b-a)^2/4$ that
Hoeffding must assume, while for large $t$ the $Mt/3$ term takes over and the
decay is exponential in $t$, which is the best a bounded-but-skewed sum can
do. We will lean on exactly this small-deviation sharpness when we prove norm
concentration below.

### Comparing Tail Bounds Numerically
:label:`subsec_mdl-tail-bounds-numerical`

For a fair coin, we can compute the
tail $P(|\hat p - \tfrac12| \ge 0.1)$ *exactly* (a finite binomial
sum, which we evaluate in log-space so that $n=5{,}000$ does not underflow)
and compare it with Chebyshev's bound ($\sigma^2=\tfrac14$, so the bound is
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

At $n=10$ both
bounds are vacuous (above $1$) while the exact tail is $0.75$: nothing
concentrates in ten flips. At $n=100$ Chebyshev ($0.25$) is actually a shade
*tighter* than Hoeffding ($0.27$): for small $n$ the comparison is between
constants, and Chebyshev's use of the true variance $\tfrac14$ helps it. But
rates always beat constants. By $n=1{,}000$ the exact tail is
$2.7\times10^{-10}$ and Hoeffding certifies $4.1\times10^{-9}$, within a
factor of about $15$ (one order of magnitude), while Chebyshev still allows a
miss probability of
$2.5\%$, off by *eight orders of magnitude*; at $n=5{,}000$ the gap has grown
to about forty-three orders. The log-scale plot makes the geometry plain: the
exact tail
and Hoeffding's bound are nearly parallel straight lines (exponential decay;
Hoeffding's exponent $2t^2=0.02$ is close to the optimal rate,
which *large-deviations theory* identifies as the Kullback--Leibler divergence
$\mathrm{KL}(0.6\,\|\,0.5)\approx 0.0201$, an information-theoretic measure of
how distinguishable two distributions are; :numref:`sec_mdl-divergences-distances`
develops both), while Chebyshev's gentle curve is
polynomial decay that no constant can rescue.

## Probability in High Dimension

Concentration also governs the geometry of random vectors. Their coordinates
provide many independent contributions, so inequalities for sums yield
high-dimensional geometric results. This phenomenon is called
**concentration of measure**: in high dimension, a sufficiently regular
function of many independent coordinates is close to its expected value with
high probability. This section quantifies two important examples.
:numref:`sec_mdl-distributions` stated two facts
while cataloguing the multivariate Gaussian: the norm of a
$d$-dimensional standard Gaussian locks onto $\sqrt d$, and two independent
draws are nearly orthogonal. Here we give both facts their quantitative
treatment.

**Probability in high dimension.** One more Gaussian fact matters enormously in
deep learning and defies low-dimensional intuition. For
$\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$ the squared norm
$\|\mathbf x\|^2=\sum_{i=1}^d x_i^2$ is a sum of $d$ independent terms with mean
$1$, so $\mathbb E\,\|\mathbf x\|^2=d$, and, by the same averaging that makes
sample means reliable, the sum concentrates near $d$: virtually all the mass
of a high-dimensional Gaussian lives in a *thin shell* of radius
$\approx\sqrt d$, far from the origin where the density is pointwise largest.
Likewise two independent draws $\mathbf x,\mathbf y$ are *nearly orthogonal*:
their cosine has mean $0$ and standard deviation $\approx1/\sqrt d$, a
probabilistic version of the concentration-of-angles proposition of
:numref:`sec_mdl-geometry-linear-algebraic-ops`
:cite:`Vershynin.2018`; both facts get their full quantitative treatment, with
exponential tail bounds, in :numref:`sec_mdl-concentration-generalization`.
Both matter in practice:
weight-initialization schemes scale by $1/\sqrt{d}$ precisely so that
$\|\mathbf W\mathbf x\|$ stays comparable to $\|\mathbf x\|$ layer after layer;
cosine similarity between embeddings is informative because unrelated
high-dimensional vectors sit near cosine $0$ by default; and nearest-neighbor
search loses contrast because the distances between random points all
concentrate around one common value. The check below watches both
concentrations set in as $d$ grows.

```{.python .input #mdl-distributions-multivariate-gaussian-1}
rng = np.random.default_rng(0)
for d in (3, 30, 300):
    x, y = rng.standard_normal((2, 1000, d))
    nx = np.linalg.norm(x, axis=1)
    cos = (x * y).sum(1) / (nx * np.linalg.norm(y, axis=1))
    print(f'd = {d:3d}:  ||x||/sqrt(d) = {(nx / np.sqrt(d)).mean():.3f}'
          f' ± {(nx / np.sqrt(d)).std():.3f},   mean |cos(x,y)| = {np.abs(cos).mean():.3f}')
```

The norm ratio locks onto $1$ with a spread shrinking like $1/\sqrt{2d}$, and
the typical cosine collapses toward $0$ at the promised $1/\sqrt d$ rate: in
high dimension, Gaussian geometry is a thin shell of mutually near-orthogonal
directions.

### Norm Concentration

**Proposition (norm concentration).** *Let
$\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$. For every $0<\varepsilon\le 1$,*

$$
P\!\left(\Bigl|\frac{\|\mathbf x\|}{\sqrt d} - 1\Bigr| \ge \varepsilon\right)
 \le 2\exp\!\left(-\frac{d\,\varepsilon^2}{8}\right).
$$
:eqlabel:`eq_mdl-norm-concentration`

**Proof sketch.** The squared norm $\|\mathbf x\|^2=\sum_{i=1}^d x_i^2$ is a
sum of $d$ i.i.d. variables $x_i^2$ with mean $1$: exactly a sample-mean
problem, except the summands are unbounded, so Hoeffding does not apply
directly. They are, however, sub-exponential, precisely the small-$\lambda$
regime defined above. Its MGF is
explicit, $E[e^{\lambda(x_i^2-1)}] = e^{-\lambda}/\sqrt{1-2\lambda}$ for
$\lambda<\tfrac12$. A series comparison
(exercise :exref:`ex_mdl-concentration-generalization-chi-square-tail`) shows

$$
E\bigl[e^{\lambda (x_i^2 - 1)}\bigr] \le e^{2\lambda^2}
\qquad\textrm{for } |\lambda| \le \tfrac14 .
$$
:eqlabel:`eq_mdl-chi2-mgf`

This is sub-exponentiality with $\nu^2=4$ and $\alpha=4$. Chernoff with this
bound gives, for any $0<\lambda\le\tfrac14$,

$$
P\Bigl(\sum_i (x_i^2 - 1) \ge d\varepsilon\Bigr)
 \le \exp\bigl(-\lambda d \varepsilon + 2 d \lambda^2\bigr),
$$

and the unconstrained minimizer $\lambda=\varepsilon/4$ is admissible
precisely when $\varepsilon\le 1$, yielding $e^{-d\varepsilon^2/8}$; the lower
tail and the factor $2$ follow as in Hoeffding's inequality. Finally convert
squares to norms: writing $z=\|\mathbf x\|/\sqrt d\ge 0$, if $|z-1|\ge\varepsilon$
then $|z^2-1|=|z-1|\,(z+1)\ge\varepsilon$, so the norm event implies the
squared-norm event and inherits its bound.

The bound has two consequences. First, the *fluctuation of $\|\mathbf x\|$ is of
constant order*: deviations of $\|\mathbf x\|$ from $\sqrt d$ beyond
$\varepsilon\sqrt d$ are exponentially rare in $d$, and a finer analysis puts
the standard deviation of $\|\mathbf x\|$ near $1/\sqrt2$ *independently of
$d$*. A standard Gaussian in $\mathbb{R}^d$ is therefore a **thin shell** of
radius $\sqrt d$ and thickness $O(1)$, rather than a ball concentrated near
the origin. Second, the point of highest density is not a typical point:
the density is largest at the
origin, yet the volume of a radius-$r$ shell grows like $r^{d-1}$, and the
product of density and shell volume is largest near
$r\approx\sqrt d$. In $d=784$ (an MNIST-sized Gaussian) a typical draw has
norm within a few percent of $28$. By
:eqref:`eq_mdl-norm-concentration`, a draw with norm below $14$ has probability
at most $2e^{-24}$.

### Near-Orthogonality Revisited

The second fact is the cosine. :numref:`sec_mdl-geometry-linear-algebraic-ops`
proved that the cosine between independent random directions has mean $0$ and
standard deviation $\approx 1/\sqrt d$; :numref:`sec_mdl-distributions`
restated it for Gaussians. Concentration puts that "typical" statement into
the same exponential frame as everything else in this section, and the
mechanism is *Hoeffding again*. Fix a unit vector
$\mathbf u$ and let the coordinates of $\mathbf x$ be independent, centered,
and of unit variance, either uniform random signs $x_i\in\{-1,+1\}$ or
standard Gaussians (the former are bounded, the latter sub-Gaussian with
proxy $1$ via :eqref:`eq_mdl-subgaussian`; either property suffices below).
The inner product $\langle\mathbf x,\mathbf u\rangle=\sum_i
u_i x_i$ is a sum of independent terms, the $i$-th bounded by $|u_i|$, and the
general Hoeffding inequality for non-identical ranges (exercise :exref:`ex_mdl-concentration-generalization-proof-eqref-eq-mdl`) gives

$$
P\bigl(|\langle\mathbf x,\mathbf u\rangle| \ge s\bigr)
 \le 2\exp\!\left(-\frac{s^2}{2\sum_i u_i^2}\right) = 2 e^{-s^2/2},
$$
:eqlabel:`eq_mdl-inner-product-tail`

since $\sum_i u_i^2=1$. The inner product with any fixed direction is $O(1)$,
while the unit-variance coordinates pin the norm of $\mathbf x$ to
$\approx\sqrt d$: exactly $\sqrt d$ for random signs, and within
$(1\pm\varepsilon)\sqrt d$ for Gaussians by
:eqref:`eq_mdl-norm-concentration`. The cosine is therefore $O(1/\sqrt d)$
with exponentially high probability: taking
$s=\varepsilon\sqrt d$, a cosine larger than $\approx\varepsilon$ has
probability at most $\approx 2e^{-d\varepsilon^2/2}$. High-dimensional space
holds exponentially many pairwise-nearly-orthogonal directions: a union bound
over the fewer than $m^2$ pairs among $m$ independent draws keeps every
pairwise cosine below $\approx\varepsilon$ whenever
$2m^2 e^{-d\varepsilon^2/2}<1$, which allows $m$ of order
$e^{d\varepsilon^2/4}$, so *random directions come nearly orthogonal by
default, exponentially many at a time*.

### Three Consequences for Deep Learning
:label:`subsec_mdl-concentration-dl`

**Initialization scales.** LeCun and He initialization
:cite:`He.Zhang.Ren.ea.2015` draw the weight rows of
a layer with variance $1/d$ (or $2/d$ for ReLU), and the standard telling
(:numref:`sec_numerical_stability`) checks variances: each pre-activation
$\langle\mathbf w,\mathbf x\rangle$ has variance $\|\mathbf x\|^2/d\approx 1$.
For a **single wide random layer**, analogous concentration results can make
this variance calculation representative: under independence or weak-dependence
and tail assumptions, the empirical activation norm is close to its expectation
with high probability. Iterating the statement through a deep nonlinear network
is a different problem. Dependencies accumulate across layers, and preserving
mean variance does not guarantee controlled singular values or dynamical
isometry. A forward-pass norm check is therefore a useful diagnostic, not a
proof that all signals remain stable through depth.

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
into a simultaneous guarantee over many objects, with only a logarithmic
increase in the deviation, is
precisely how learning bounds are built.) The nearest and farthest neighbor of
a query then differ by a vanishing relative margin (the contrast
$(d_{\max}-d_{\min})/d_{\min}$ collapses) and nearest-neighbor retrieval
degenerates into noise-ranking. Real embeddings
escape because they are structured: semantic similarity places data near a
low-dimensional set inside $\mathbb{R}^d$, so observed distance histograms are
far broader than the concentration prediction, and the size of that gap is one
gauge of how much structure an embedding carries.

**The thin shell and the mean.** The shell picture corrects a beginner's
mental model that matters in generative modeling: the "most likely point" of a
high-dimensional Gaussian prior (the origin) is nowhere near a *typical*
point. Decoders are trained on inputs of norm $\approx\sqrt d$, so evaluating
one at the mean, or at the midpoint of two latent draws (norm
$\approx\sqrt{d/2}$, well inside the shell), feeds it an input unlike anything
it saw in training; this is why latent-space interpolation is done along the
sphere rather than the chord. The shell is also why the MVN cell of
:numref:`sec_mdl-distributions` saw norm ratios pinned to $1.000$: it was
measuring means; the cell below measures the *whole distribution*.

### Measuring the Shell
:label:`subsec_mdl-measuring-shell`

The distributions chapter verified the *means* of the norm ratio and cosine.
We now estimate the fraction of the distribution inside the shell
$(1\pm\varepsilon)\sqrt d$ and measure the reduction in nearest-neighbor
contrast, using $\varepsilon=0.1$.

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

In $d=2$ the "shell" holds under $15\%$ of the mass: low dimension really is
a fuzzy ball, and the nearest and farthest of $200$ points differ by a factor
of $30$. By $d=200$ the same $\pm10\%$ shell already captures $95\%$ of the
mass, close to what a Gaussian approximation of the norm
predicts ($\pm 2$ standard deviations at $\sigma\approx 1/\sqrt{2d}=0.05$),
and at $d=2000$ all ten thousand sampled points lie within the shell. The
nearest-neighbor contrast decreases from $29.4$ to $0.08$: in $d=2000$ the
farthest of two hundred random points is only $8\%$ farther from the query than
the nearest. Distances therefore discriminate poorly among these points.

## Summary

* The *Chernoff method* (Markov's inequality applied to $e^{\lambda X}$,
  then optimized over $\lambda$) converts a bound on the moment generating
  function into an exponential tail bound; for sums of independent variables
  the MGF factorizes, which is where the exponential-in-$n$ decay comes from.
* *Hoeffding's lemma* bounds the MGF of a centered variable on $[a,b]$ by
  $e^{\lambda^2(b-a)^2/8}$ (proof: the second derivative of the log-MGF is a
  tilted variance, at most $(b-a)^2/4$), and *Hoeffding's inequality* follows:
  $P(|\bar X-E\bar X|\ge t)\le 2e^{-2nt^2/(b-a)^2}$, the finite-sample bound
  behind the main book's test-set arithmetic, exponentially sharper than
  Chebyshev's $1/n$. Inverted, it gives the finite-sample confidence
  interval :eqref:`eq_mdl-hoeffding-interval`. *Sub-Gaussian* variables are those obeying such an MGF
  bound; *Bernstein's inequality* refines the constant using the true
  variance and covers the sub-exponential case (squares, norms).
* In high dimension, concentration is geometry (*concentration of measure*):
  $\|\mathbf x\|/\sqrt d$
  concentrates at $1$ with failure probability $2e^{-d\varepsilon^2/8}$, so a
  Gaussian is a *thin shell*, random directions are *near-orthogonal*
  ($\cos\approx 1/\sqrt d$), pairwise distances concentrate (degrading
  nearest-neighbor contrast). In a single wide random layer, related
  concentration can make the variance calculation representative, but stability
  across many nonlinear layers needs additional dependence and Jacobian control.
## Exercises

1. **Hoeffding with unequal ranges.** The proof of :eqref:`eq_mdl-hoeffding`
   assumed that every $X_i$ takes values in the same interval $[a,b]$.
    1. Redo the proof for independent $X_i\in[a_i,b_i]$ and show that

       $$P\bigl(|\bar X-E[\bar X]|\ge t\bigr)\le 2\exp\left(-\frac{2n^2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\right).$$

       Check that this bound reduces to :eqref:`eq_mdl-hoeffding` when all
       ranges agree.
    1. Let $x_1,\ldots,x_d$ be independent uniform random signs and let
       $\mathbf u\in\mathbb R^d$ be a unit vector. Apply the bound of (a)
       to the sum $\sum_i u_i x_i$ and
       derive :eqref:`eq_mdl-inner-product-tail`.
   :exlabel:`ex_mdl-concentration-generalization-proof-eqref-eq-mdl`

1. **Sub-Gaussian random signs.** A Rademacher variable $\varepsilon$ takes
   the values $-1$ and $+1$ with probability $\tfrac12$ each, so that
   $E[e^{\lambda\varepsilon}]=\cosh\lambda$.
    1. Show that $\cosh\lambda\le e^{\lambda^2/2}$ for every
       $\lambda\in\mathbb R$ by comparing the two Taylor series term by term.
       Use $(2k)!\ge 2^k k!$.
    1. Conclude with the Chernoff method :eqref:`eq_mdl-chernoff` that the
       average $\bar\varepsilon$ of $n$ independent random signs satisfies
       $P(|\bar\varepsilon|\ge t)\le 2e^{-nt^2/2}$.
    1. Hoeffding's lemma :eqref:`eq_mdl-hoeffding-lemma` with $[a,b]=[-1,1]$
       also gives a variance proxy for $\varepsilon$. Compare it with (a).
       Expand $\log\cosh\lambda$ to second order around $\lambda=0$, and
       explain why no constant smaller than $\tfrac18$ can replace the
       $\tfrac18$ in :eqref:`eq_mdl-hoeffding-lemma` for all bounded
       variables.
   :exlabel:`ex_mdl-concentration-generalization-random-signs`

1. **Latent interpolation.** A generative model decodes latent codes
   $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ with $d=512$. Let
   $\mathbf z_1,\mathbf z_2$ be two independent codes and
   $\mathbf z(t)=(1-t)\mathbf z_1+t\mathbf z_2$ for $t\in[0,1]$ their linear
   interpolation.
    1. Find the distribution of $\mathbf z(t)$ and the approximate value of
       $\|\mathbf z(t)\|$. At which $t$ is the norm smallest?
    1. Use :eqref:`eq_mdl-norm-concentration` to bound the probability that
       the midpoint $\mathbf z(\tfrac12)$ has norm at least $0.9\sqrt d$,
       that is, reaches the shell $(1\pm 0.1)\sqrt d$ in which typical codes
       lie.
    1. Spherical interpolation :cite:`White.2016` replaces the chord by the
       arc

       $$\mathbf s(t)=\frac{\sin((1-t)\theta)}{\sin\theta}\,\mathbf z_1+\frac{\sin(t\theta)}{\sin\theta}\,\mathbf z_2,$$

       where $\theta$ is the angle between $\mathbf z_1$ and $\mathbf z_2$.
       Show that $\|\mathbf s(t)\|=\rho$ for all $t$ when
       $\|\mathbf z_1\|=\|\mathbf z_2\|=\rho$. Which value does $\theta$
       take approximately for $d=512$, and why?
   :exlabel:`ex_mdl-concentration-generalization-latent-interpolation`

1. [code] **Chernoff bound for coins.** ● Let $\hat p$ be the fraction of
   heads in $n$ independent flips of a coin with bias $p$, and let $a>p$.
    1. Apply the Chernoff method :eqref:`eq_mdl-chernoff` to $n\hat p$, whose
       moment generating function is $(1-p+pe^\lambda)^n$. Optimize over
       $\lambda$ and show that $P(\hat p\ge a)\le e^{-n\,\mathrm{KL}(a\|p)}$,
       where
       $\mathrm{KL}(a\|p)=a\log\frac ap+(1-a)\log\frac{1-a}{1-p}$.
    1. Show that $\mathrm{KL}(a\|p)\ge 2(a-p)^2$ by examining the second
       derivative in $a$, so that the bound of (a) is never weaker than
       one side of Hoeffding's bound. Evaluate both exponents for
       $p=\tfrac12$ and $a=0.6$.
    1. Add the two-sided bound $2e^{-n\,\mathrm{KL}(0.6\|0.5)}$ as a column
       to the table of :numref:`subsec_mdl-tail-bounds-numerical`, for
       $n\in\{10,100,1000,5000\}$. The ratio of the exact tail to this bound
       still decreases with $n$. How does it scale with $n$, and what does
       the scaling say about the factor that the Chernoff method loses?
   :exlabel:`ex_mdl-concentration-generalization-chernoff-coins`

1. **Chi-square tail.** ● The proof of :eqref:`eq_mdl-norm-concentration`
   rests on the bound :eqref:`eq_mdl-chi2-mgf` for $x^2-1$ with
   $x\sim\mathcal N(0,1)$, which the proof states without derivation.
    1. Show that $E[e^{\lambda(x^2-1)}]=e^{-\lambda}/\sqrt{1-2\lambda}$ for
       $\lambda<\tfrac12$ by completing the square in the Gaussian integral.
    1. Expand $\log E[e^{\lambda(x^2-1)}]$ into a power series in $2\lambda$
       and prove :eqref:`eq_mdl-chi2-mgf`. The series starts at the quadratic
       term.
    1. The proof of :eqref:`eq_mdl-norm-concentration` takes
       $\lambda=\varepsilon/4$, which is admissible only for
       $\varepsilon\le 1$. For $\varepsilon>1$, set $\lambda=\tfrac14$ and
       bound $P(\|\mathbf x\|^2\ge(1+\varepsilon)d)$. Does the result decay
       like $e^{-c\varepsilon^2}$ or like $e^{-c\varepsilon}$ in
       $\varepsilon$, and which regime of :eqref:`eq_mdl-subexponential` does
       it belong to?
   :exlabel:`ex_mdl-concentration-generalization-chi-square-tail`

1. [code] **Activation norms through depth.** ● Consider a ReLU network of
   width $d$ and depth $K$ without biases,
   $\mathbf h_k=\mathrm{ReLU}(\mathbf W_k\mathbf h_{k-1})$, whose weights are
   independent draws from $\mathcal N(0,2/d)$ (He
   initialization, :numref:`sec_numerical_stability`). Let
   $\rho_k=\|\mathbf h_k\|^2/\|\mathbf h_{k-1}\|^2$. How such norms fluctuate
   with depth and width was analyzed by :citet:`Hanin.Nica.2020`.
    1. Show that, given $\mathbf h_{k-1}\ne\mathbf 0$, the ratio $\rho_k$ is
       distributed as $\frac2d\sum_{j=1}^d\mathrm{ReLU}(g_j)^2$ with
       independent $g_j\sim\mathcal N(0,1)$, whatever the direction of
       $\mathbf h_{k-1}$. Conclude that $E[\rho_k]=1$ and
       $\operatorname{Var}(\rho_k)=5/d$, using $E[g^2]=1$ and $E[g^4]=3$.
    1. Write $\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$ as a sum of $K$
       independent terms. Expand $\log\rho$ to second order around $1$ and
       predict the mean and variance of this sum in terms of $K/d$.
    1. Simulate $1{,}000$ networks for each width $d\in\{32,128\}$ and depth
       $K\in\{1,10,50\}$, each with a fresh standard Gaussian input. Compare
       the mean and variance of $\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$
       with (b), and report the mean and the median of the ratio
       $\|\mathbf h_K\|^2/\|\mathbf h_0\|^2$ itself. The variance calculation
       of :numref:`sec_numerical_stability` gives
       $E[\|\mathbf h_K\|^2/\|\mathbf h_0\|^2]=1$ at every depth. Repeat the
       run at $K=50$ and $d=32$ with several random seeds. Why does the sample
       mean over $1{,}000$ networks scatter so widely around $1$? Why does the
       expectation not describe a typical network there?
   :exlabel:`ex_mdl-concentration-generalization-depth-norms`

[Discussions](https://d2l.discourse.group/t/concentration-and-generalization)

[Discussions](https://d2l.discourse.group/t/36834)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.5]{.kicker}

Why averages can be trusted<br>**exponential tails · high dimension**.
:::
:::

::: {.slide title="Polynomial tails are not enough"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Chebyshev caps a sample mean's miss probability at $\sigma^2/(nt^2)$, so its
bound for $10^{-6}$ takes a
*million* times the data of certifying $10^0$. The truth for bounded
independent data admits an **exponential** finite-sample bound in $n$.

The main book bounded test-set error with Hoeffding's
inequality *on faith* (the generalization-in-classification section). Here we
prove it, then follow the same machinery into high dimension and out to
generalization itself.
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

::: {.slide title="The Chernoff method and exponential moments"}
[Chernoff]{.kicker}

Where the MGF $M(\lambda)=E[e^{\lambda X}]$ is finite, Markov's inequality
applied to $e^{\lambda X}$ converts an MGF bound into a tail bound:

$$P(X \ge t) \;\le\; \inf_{\lambda>0}\; e^{-\lambda t}\, M(\lambda).$$

. . .

::: {.d2l-note .rule}
For a **sum** of independent terms the MGF factors, so the exponent grows
linearly in $n$. This is one important concentration route; martingale and
bounded-difference methods provide others.
:::
:::

::: {.slide title="Hoeffding's lemma: bounded ⇒ sub-Gaussian"}
[Hoeffding]{.kicker}

For centered $X \in [a,b]$:
$\;E[e^{\lambda X}] \le \exp\bigl(\lambda^2(b-a)^2/8\bigr)$: an MGF *no
worse than a Gaussian's* with $\sigma = (b-a)/2$.

. . .

*Proof idea.* $\psi(\lambda) = \log E[e^{\lambda X}]$ has
$\psi(0)=\psi'(0)=0$, and $\psi''(\lambda)$ is the variance of $X$ under
an exponentially **tilted** distribution, still supported in $[a,b]$, so
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
exponent $-\lambda n t + n\lambda^2(b-a)^2/8$: a parabola in $\lambda$,
minimized at $\lambda = 4t/(b-a)^2$ with value $-2nt^2/(b-a)^2$. The
other tail by symmetry; union bound supplies the $2$. $\blacksquare$

::: {.d2l-note .rule}
No Gaussianity, no variance estimate, no asymptotics: only boundedness
and independence, at **every finite $n$**.
:::
:::

::: {.slide title="Bounding test error, now a theorem"}
[Hoeffding]{.kicker}

Inverted, Hoeffding is a *finite-sample confidence interval*: with
probability $1-\delta$,

$$|\bar X - E[\bar X]| \;\le\; (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}.$$

. . .

The generalization-in-classification section demanded a test-set error
estimate within $t=0.01$ at $\delta = 0.05$ and quoted "roughly 18,500
examples." Solve $2e^{-2nt^2} \le \delta$:

$$n = \frac{\log(2/\delta)}{2t^2} \approx 18{,}445.$$

A citation is now a theorem, and unlike the Gaussian interval of the
statistics section, its coverage is a guarantee at every $n$, not an
asymptote.
:::

::: {.slide title="Names for the well-behaved: sub-Gaussian and friends"}
[The zoo]{.kicker}

$X$ is **sub-Gaussian** with proxy $\sigma^2$ if
$E[e^{\lambda(X-EX)}] \le e^{\lambda^2\sigma^2/2}$; Chernoff then gives
$P(|X-EX| \ge t) \le 2e^{-t^2/(2\sigma^2)}$. Three families:
Gaussians (proxy $\sigma^2$, exactly), bounded variables (proxy
$(b-a)^2/4$, the lemma), random signs (proxy $1$).

. . .

::: {.d2l-note}
*Squares* of Gaussians (hence squared norms) are only
**sub-exponential**: Gaussian rate for small deviations, $e^{-ct}$ for
large. **Bernstein's inequality** covers them and uses the *true*
variance where Hoeffding must assume the worst case.
:::
:::

::: {.slide title="Numerical comparison of tail bounds" layout="tight"}
[Measured: the fair coin's exact tail vs. both bounds]{.kicker}

@!mdl-concentration-generalization-the-tail-race-in-code

By $n=1000$ Chebyshev is off by **eight orders of magnitude**; by
$n=5000$, about forty-three.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Probability in high dimension]{.dtitle}

[thin shells and orthogonal-by-default directions]{.dsub}
:::
:::

::: {.slide title="In high dimension, the Gaussian is a thin shell"}
[Continuous]{.kicker}

$\|\mathbf x\|^2$ sums $d$ independent mean-$1$ terms, so it concentrates near
$d$. Consequently, most probability lies in a shell of radius $\approx\sqrt d$, *far from
the origin where the density is pointwise largest*; and two independent
draws are nearly orthogonal, cosine $\sim 1/\sqrt d$:

@!mdl-distributions-multivariate-gaussian-1

::: {.d2l-note}
The $1/\sqrt d$ **initialization** scale keeps
$\|\mathbf{Wx}\|\approx\|\mathbf x\|$; cosine similarity is informative
*because* unrelated vectors have cosine similarity near $0$; nearest-neighbor contrast
fades. Exponential tail bounds arrive in the concentration-and-generalization
section.
:::
:::

::: {.slide title="A high-dimensional Gaussian is a thin shell"}
[Geometry]{.kicker}

$\|\mathbf x\|^2$ is a sum of $d$ mean-$1$ terms: a sample-mean problem
with sub-exponential summands, and the Bernstein-flavored Chernoff gives

$$P\!\left(\Bigl|\tfrac{\|\mathbf x\|}{\sqrt d} - 1\Bigr| \ge \varepsilon\right)
\;\le\; 2\,e^{-d\varepsilon^2/8}.$$

. . .

The density peaks at the origin, but shell volume grows like $r^{d-1}$:
the mass ends up at $r \approx \sqrt d$, and the shell's thickness is
$O(1)$, *independent of $d$*.

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
$d \gtrsim \tfrac{8}{\varepsilon^2}\log\tfrac{2n^2}{\delta}$: dimension
only logarithmic in $n$. Remember the move: one tail $\to$ many objects,
for a logarithmic increase. It is how learning bounds are built.
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
This is why $1/\sqrt d$ initialization holds *for essentially every
draw*, and why latent interpolation follows the sphere rather than the
chord.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Chernoff:** Markov on $e^{\lambda X}$; independent sums factor the
  MGF → exponential tails.
- **Hoeffding:** $2e^{-2nt^2/(b-a)^2}$ turns the generalization-in-
  classification section's 18,500-example quote into a theorem, valid at
  every finite $n$.
- **High dimension:** thin shells, orthogonal-by-default directions.
:::
