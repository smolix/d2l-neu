# Random Variables
:label:`sec_mdl-random_variables`

In :numref:`sec_prob` we considered discrete random variables, whose
probabilities are obtained by summing probability masses. Pixel intensities,
network weights, and Gaussian noise are usually modeled as continuous random
variables, for which individual outcomes have probability zero. A probability
density assigns probability to intervals through integration. This section
develops densities and cumulative distribution functions, then introduces the
mean, variance, and standard deviation. Joint, marginal, and conditional
distributions extend the same definitions to several variables, together with
covariance and correlation. The relation in :eqref:`eq_mdl-pdf_def` connects the
probability of a small interval to its width and local density.

```{.python .input #random-variables-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input #random-variables-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import numpy as np
```

```{.python .input #random-variables-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
```

```{.python .input #random-variables-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
from jax import numpy as jnp
import numpy as np
```

## From Discrete to Continuous Probability

For a continuous random variable, probabilities are obtained by integrating
rather than summing. We develop the theory in three steps: first the density
that replaces the mass function,
the cumulative function that turns it back into a genuine probability, and the
derivative relationship between them, which is the fundamental theorem of
calculus restated for probabilities.

### From Bin Probabilities to a Density

Throw a dart at a board and ask for the probability it lands *exactly*
$2\,\textrm{cm}$ from the center. Measure to one digit, with bins for
$0,1,2,\ldots\,\textrm{cm}$: of $100$ throws, perhaps $20$ land in the
"$2\,\textrm{cm}$" bin, suggesting $20\%$. But that bin holds everything between
$1.5$ and $2.5\,\textrm{cm}$, not what we asked. Measure finer, to bins of
$0.1\,\textrm{cm}$: now perhaps $3$ throws land in $[1.95,2.05]$, suggesting
$3\%$. We have only pushed the problem one digit down.

The relevant object is the probability of a short interval, not the behavior
of successive decimal digits. For $\epsilon>0$, consider
$P(x\leq X<x+\epsilon)$. If the local probability-to-width ratio has a limit,
define

$$
p(x)=\lim_{\epsilon\downarrow0}
\frac{P(x\leq X<x+\epsilon)}{\epsilon}.
$$

At points where $p$ is continuous, this definition gives

$$P(x\leq X<x+\epsilon)=\epsilon p(x)+o(\epsilon).$$
:eqlabel:`eq_mdl-pdf_def`

When interval probabilities have the representation
$P(X\in A)=\int_A p(x)\,dx$, the law is *absolutely continuous* and $p$ is its
probability density function (p.d.f.). Not every probability law has a density;
discrete laws have atoms, and mixed laws contain both atomic and density parts.

If $X$ has a density, every singleton has zero probability. Thus, for any fixed
$x$,

$$P(X = x) = 0.$$

This resolves the dartboard example: the probability of landing *exactly*
$2\,\textrm{cm}$ out is zero, even though the dart certainly lands *somewhere*.
The size notion at work here deserves a name. A set has *measure zero* when,
for every $\varepsilon>0$, it can be covered by a countable collection of
intervals of total length at most $\varepsilon$: a single point fits inside one
interval of length $\varepsilon$, and a countable set $\{x_1,x_2,\ldots\}$ fits
inside intervals of lengths $\varepsilon/2,\varepsilon/4,\ldots$, which sum to
$\varepsilon$. For a variable with a density, every measure-zero set carries
zero probability: covering the set by intervals of total length $\delta$ caps
its probability at the integral of $p$ over those intervals, which shrinks to
zero with $\delta$ (immediately when $p$ is bounded, and with a little more
care in general). A point carrying *positive* mass is called an *atom*. A
purely absolutely continuous variable has no atoms, whereas a mixed law may
have atoms and a density on the remainder of its support.

### Densities and Their Two Defining Properties

What must $p(x)$ satisfy to be a density? Two things, both from
:eqref:`eq_mdl-pdf_def`. First, probabilities are never negative, so $p(x)\ge 0$.
Second, slice $\mathbb{R}$ into width-$\epsilon$ pieces $(\epsilon i,
\epsilon(i{+}1)]$; by :eqref:`eq_mdl-pdf_def` each contributes about $\epsilon\,
p(\epsilon i)$ to the total probability, and summing,

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i)
\;\xrightarrow[\epsilon\to0]{}\; \int_{-\infty}^{\infty} p(x)\,dx .
$$

The middle expression is exactly the Riemann sum of :numref:`sec_mdl-integral_calculus`.
Since $X$ must take *some* value, $P(X\in\mathbb{R})=1$, and we recover the
normalization $\int_{-\infty}^{\infty} p(x)\,dx = 1$ of :eqref:`eq_mdl-density`.
The same slicing argument over a finite range gives the rule we actually use:
*probability is area under the density*,

$$P(X\in(a, b]) = \int_{a}^{b} p(x)\,dx.$$
:eqlabel:`eq_mdl-pdf_int_int`

These two properties, non-negativity and total area $1$, describe *exactly* the
space of densities. :numref:`fig_mdl-prob-pdf-area` shows the picture: the total
shaded region has area $1$, and the probability of an interval is the area above
it.

![A probability density $p(x)$. The total area under the curve is $1$; the probability that $X$ lands in the interval from $a$ to $b$ is the area of the shaded slab above it, $\int_a^b p(x)\,dx$. The density itself is not a probability (it may exceed $1$); only its integral over an interval is.](../img/mdl-prob-pdf-area.svg)
:label:`fig_mdl-prob-pdf-area`

We can confirm :eqref:`eq_mdl-pdf_int_int` numerically. Take the two-bump density
$p(x)=0.2\,\mathcal N(x;3,1)+0.8\,\mathcal N(x;-1,1)$ and recover the probability
of an interval by a Riemann sum, the discrete approximation of the integral.

```{.python .input #random-variables-density-to-probability}
import numpy as onp
# Recover P(-2 < X <= 3) from the density by a Riemann sum (numerical integral)
epsilon = 0.01
x = onp.arange(-8, 8, epsilon)
p = 0.2*onp.exp(-(x - 3)**2 / 2)/onp.sqrt(2 * onp.pi) + \
    0.8*onp.exp(-(x + 1)**2 / 2)/onp.sqrt(2 * onp.pi)
print(f'total mass     : {float(onp.sum(epsilon * p)):.4f}')
mask = (x > -2) & (x <= 3)
print(f'P(-2 < X <= 3) : {float(onp.sum(epsilon * p[mask])):.4f}')
```

The total mass is $1.0000$, as :eqref:`eq_mdl-density` requires
(the grid spans $[-8,8]$, wide enough that the truncated Gaussian tails hold
mass under $10^{-7}$), and the interval integral returns a genuine probability
in $[0,1]$. The estimate $0.7725$ is within $6\times10^{-4}$ of the exact value
$0.7731$, the small bias of a right-endpoint Riemann sum at $\epsilon=0.01$. A
catalogue of named densities (Gaussian, exponential, and the rest) follows in
:numref:`sec_mdl-distributions`; here we stay abstract.

### The Cumulative Distribution Function

A density has one awkward feature: its *values* are not probabilities. A density
can exceed $10$, as long as it does so only over an interval shorter than
$1/10$. The remedy is the *cumulative distribution function* (c.d.f.), which by
:eqref:`eq_mdl-pdf_int_int` accumulates the density up to $x$ and so *is* a
probability:

$$
F(x) = \int_{-\infty}^{x} p(t)\,dt = P(X \le x).
$$
:eqlabel:`eq_mdl-cdf_def`

Read off its properties directly: $F(x)\to 0$ as $x\to-\infty$ and $F(x)\to 1$ as
$x\to+\infty$ (the total mass is $1$); $F$ is non-decreasing, since the integrand
$p\ge 0$ only ever adds area; and since $X$ has a density, $F$ is an integral of
it and hence continuous. The
c.d.f. is also what lets discrete and continuous variables share one framework.
For a discrete $X$ taking $0$ and $1$ with probability $\tfrac12$ each,

$$
F(x) = \begin{cases}
0 & x < 0, \\
\tfrac12 & 0 \le x < 1, \\
1 & x \ge 1,
\end{cases}
$$

a *staircase* that jumps by the mass at each atom. Since $F(x)=P(X\le x)$
includes the atom at $x$ itself, $F$ takes the upper value at each jump; in
general every c.d.f. is *right-continuous*, $\lim_{y\downarrow x}F(y)=F(x)$.
The same $F$ thus describes
continuous variables, discrete ones, and mixtures (flip a coin: heads, report a
die roll; tails, a dart distance).

The density and the c.d.f. carry the same information, and the bridge between
them is the fundamental theorem of calculus. Differentiating
:eqref:`eq_mdl-cdf_def` recovers the density.

**Proposition (density is the derivative of the c.d.f.).** *If $X$ has a
continuous density $p$, then $F$ is differentiable and*

$$
F'(x) = p(x).
$$
:eqlabel:`eq_mdl-cdf_deriv`

**Proof.** By definition :eqref:`eq_mdl-cdf_def`, $F(x)=\int_{-\infty}^x p(t)\,dt$
is the area-so-far function of the integrand $p$. The fundamental theorem of
calculus :eqref:`eq_mdl-ftc` says precisely that such an area function is
differentiable with derivative equal to the integrand, $F'(x)=p(x)$, at every
point where $p$ is continuous. $\blacksquare$

So $p=F'$ and $F=\int p$ are two views of one object: the density is the
*instantaneous rate* at which probability accumulates, and the c.d.f. is the
*accumulated area*. :numref:`fig_mdl-prob-pdf-cdf` puts them side by side (the
shaded area $\int_a^b p$ on the left equals the vertical rise $F(b)-F(a)$ on the
right), making :eqref:`eq_mdl-cdf_def` and :eqref:`eq_mdl-cdf_deriv` a single
picture.

![Left: a density $p$ with the area over the interval from $a$ to $b$ shaded. Right: the matching c.d.f. $F(x)=\int_{-\infty}^x p$, an increasing curve from $0$ to $1$; the rise $F(b)-F(a)$ equals the shaded area on the left. The density is the slope of the c.d.f. $\left(F'=p\right)$ and the c.d.f. is the accumulated area under the density.](../img/mdl-prob-pdf-cdf.svg)
:label:`fig_mdl-prob-pdf-cdf`

The inverse $F^{-1}$, which maps a probability level $q\in(0,1)$ back to the
value $x$ with $F(x)=q$, is called the *quantile function*; its values at
$q=0.25,0.5,0.75$ are the quartiles, the middle one the median. The quantile
function is also a *sampler*, by an argument short enough to give in full. The
argument needs one ingredient: a variable $U$ *uniform on $[0,1]$*, the
variable whose density is constant at $1$ there, so that $U$ assigns every
subinterval of $[0,1]$ probability equal to its length
(:numref:`sec_mdl-distributions` names this law $U(0,1)$).

**Proposition (inverse-transform sampling).** *Let $F$ be a continuous, strictly
increasing c.d.f. and let $U$ be uniform on $[0,1]$. Then $X=F^{-1}(U)$ has
c.d.f. exactly $F$.*

**Proof.** Because $F$ is increasing, the events $F^{-1}(U)\le x$ and
$U\le F(x)$ are the same event, and the uniform assigns the latter probability
$F(x)$, the length of the subinterval $[0,F(x)]$:

$$
P\bigl(F^{-1}(U)\le x\bigr) = P\bigl(U\le F(x)\bigr) = F(x). \quad\blacksquare
$$

:numref:`fig_mdl-prob-inverse-transform` illustrates the construction. Place the
uniform level $U$ on the *vertical* axis of the c.d.f. and map it through the
curve to the horizontal axis. Where the density is high, the c.d.f. is steep,
so a uniform range of vertical levels maps to a narrow range of horizontal
values. Thus the transformed samples are concentrated where the density is
large. This proposition gives a general procedure for turning uniform noise
into samples from a one-dimensional distribution: generate $U$, then evaluate
$F^{-1}(U)$. We put it to work when we meet the named distributions in
:numref:`sec_mdl-distributions`, where inverting the exponential's c.d.f. gives
that distribution's standard sampler in closed form.

![Inverse-transform sampling. Uniform levels $U$ on the vertical axis (left) reflect through the c.d.f. $F$ down to values $x=F^{-1}(U)$ on the horizontal axis. Where $F$ is steep, that is, where the density $p=F'$ is large, a uniform spread of levels is compressed into a dense cluster of samples, so the histogram of reflected points reproduces $p$.](../img/mdl-prob-inverse-transform.svg)
:label:`fig_mdl-prob-inverse-transform`

Sampling is also how expectations are computed at scale: given independent
draws $x_1,\ldots,x_n$ of $X$, the *Monte Carlo* estimate
$\tfrac1n\sum_{i=1}^n g(x_i)$ approximates $E[g(X)]$, and its variance is
$\textrm{Var}(g(X))/n$ (Exercise :exref:`ex_mdl-random-variables-using-textrmvar-x-y`a carries out the computation), shrinking with
every additional sample. The guarantee that the estimate converges to the
expectation, the law of large numbers, is stated and proved in
:numref:`sec_mdl-statistics`.

## Summarizing a Distribution

A full distribution is often more than we can interpret at a glance. *Summary
statistics* reduce it to a few numbers: the *mean* locates its center, while the
*variance* and *standard deviation* measure its spread. Each is an expectation,
that is, a density-weighted average as in :eqref:`eq_mdl-expectation`. We derive
their basic algebra once and use it throughout the book.

### The Mean

The *mean* (or *expectation*) is the average value, weighting each outcome by its
probability. For a discrete $X$ taking $x_i$ with probability $p_i$,

$$\mu_X = E[X] = \sum_i x_i p_i,$$
:eqlabel:`eq_mdl-exp_def`

and in the continuum the sum becomes the density-weighted integral
:eqref:`eq_mdl-expectation`, $\mu_X=\int x\,p(x)\,dx$, by the same
slice-and-refine argument that produced the normalization
:eqref:`eq_mdl-density`. The same
weighting averages any *function* of $X$: $E[g(X)]=\sum_i g(x_i)\,p_i$ in the
discrete case, while the continuous form $\int g(x)\,p(x)\,dx$ restates the
second half of :eqref:`eq_mdl-expectation`. We apply the rule so routinely and
so silently that it has earned a name: the *law of the unconscious
statistician*. It applies below whenever we average $X^2$ or a squared
deviation; only the name and the discrete form are new here. The mean describes
the center of a distribution, although it need not be a typical value.

A running example recurs through the whole section. Let $X$ take $a-2$ with
probability $p$, $a+2$ with probability $p$, and $a$ with probability $1-2p$.
Then

$$
\mu_X = (a-2)p + a(1-2p) + (a+2)p = a,
$$

the center of symmetry, exactly as intuition demands.

The most frequently used algebraic property of the mean is *linearity*:
constants may be taken outside an expectation, and the expectation of a sum is
the sum of the expectations. Both statements follow directly from the
definition.

**Proposition (linearity of expectation).** *For random variables $X,Y$ and
constants $a,b$,*

$$
E[aX+b]=a\,E[X]+b,
\qquad
E[X+Y]=E[X]+E[Y].
$$
:eqlabel:`eq_mdl-exp_linear`

**Proof.** Both are linearity of the sum (or integral) defining the expectation.
For the scaling-and-shift, $E[aX+b]=\sum_i(ax_i+b)p_i = a\sum_i x_i p_i + b\sum_i
p_i = a\,E[X]+b$, using $\sum_i p_i=1$. For the sum, write the expectation over
the *joint* distribution of $(X,Y)$, with $p_{ij}$ the probability of the pair
$(x_i,y_j)$ (the discrete joint of :numref:`sec_prob`; its continuous
counterpart, the joint density, arrives later in this section), and split the
inner sum: $E[X+Y]=\sum_{i,j}
(x_i+y_j)p_{ij} = \sum_{i,j} x_i p_{ij} + \sum_{i,j} y_j p_{ij} = E[X]+E[Y]$, the
last step recognizing the marginals $\sum_j p_{ij}=P(X=x_i)$. The continuous case
replaces every sum by an integral verbatim. $\blacksquare$

The second identity needs *no independence assumption*: expectations of sums
always add, however entangled $X$ and $Y$ are. Expected-loss calculations
throughout the book rest on this.

The mean alone is not enough. A profit of $\$10\pm\$1$ per sale and one of
$\$10\pm\$15$ share a mean but carry wildly different risk. We need a measure of
*spread*.

### Variance and Standard Deviation

The *variance* measures how far $X$ strays from its mean. The deviation
$X-\mu_X$ is positive or negative, so we square it before averaging:

$$\sigma_X^2 = \textrm{Var}(X) = E\bigl[(X-\mu_X)^2\bigr].$$
:eqlabel:`eq_mdl-var_def`

Expanding the square gives a formula that is almost always easier to compute
with, and it deserves a proof since we use it constantly.

**Proposition (computational form of the variance).**

$$
\textrm{Var}(X) = E[X^2] - E[X]^2 = E[X^2]-\mu_X^2.
$$
:eqlabel:`eq_mdl-var_comp`

**Proof.** Expand the square inside :eqref:`eq_mdl-var_def` and apply linearity
:eqref:`eq_mdl-exp_linear`, treating $\mu_X$ as the constant it is:

$$
E\bigl[(X-\mu_X)^2\bigr]
 = E\bigl[X^2 - 2\mu_X X + \mu_X^2\bigr]
 = E[X^2] - 2\mu_X\,E[X] + \mu_X^2
 = E[X^2] - \mu_X^2,
$$

since $E[X]=\mu_X$, so $-2\mu_X E[X]+\mu_X^2 = -2\mu_X^2+\mu_X^2=-\mu_X^2$. $\blacksquare$

On the running example $\mu_X=a$, and the computational form does the rest:
$E[X^2]=(a-2)^2p+a^2(1-2p)+(a+2)^2p = a^2+8p$, so

$$
\textrm{Var}(X) = E[X^2]-\mu_X^2 = (a^2+8p)-a^2 = 8p.
$$

This is sensible: at the largest allowed $p=\tfrac12$ the variable is a coin flip
between $a\pm2$, each $2$ away from the mean, giving variance $8\cdot\tfrac12=4=2^2$;
at $p=0$ it is constant at $a$ with no variance at all.

How does variance behave under the affine maps we constantly apply (rescaling
units, centering data)? The shift drops out and the scale comes out squared.

**Proposition (variance under affine maps).** *For constants $a,b$,*

$$
\textrm{Var}(aX+b) = a^2\,\textrm{Var}(X).
$$
:eqlabel:`eq_mdl-var_affine`

**Proof.** By linearity :eqref:`eq_mdl-exp_linear` the mean maps as
$\mu_{aX+b}=a\mu_X+b$, so the deviation is $(aX+b)-\mu_{aX+b}=a(X-\mu_X)$; the
shift $b$ cancels. Squaring and taking the expectation,

$$
\textrm{Var}(aX+b) = E\bigl[a^2(X-\mu_X)^2\bigr] = a^2\,E\bigl[(X-\mu_X)^2\bigr] = a^2\,\textrm{Var}(X). \qquad\blacksquare
$$

Two further facts round out the list: $\textrm{Var}(X)\ge 0$, with equality iff
$X$ is constant (a square has non-negative mean, zero only when $X-\mu_X\equiv0$);
and for *independent* $X,Y$, $\textrm{Var}(X+Y)=\textrm{Var}(X)+\textrm{Var}(Y)$
(we prove the general version, with a covariance correction, in
:numref:`subsec_mdl-covariance`).

The variance has a units problem. If $X$ is in stars, $X-\mu_X$ is in stars but
$(X-\mu_X)^2$ is in *squared stars*, so $\textrm{Var}(X)$ is not comparable to
the data. Taking the square root restores the original units and defines the
*standard deviation*

$$
\sigma_X = \sqrt{\textrm{Var}(X)}.
$$
:eqlabel:`eq_mdl-std_def`

Its properties echo the variance through the square root: $\sigma_X\ge0$;
$\sigma_{aX+b}=|a|\sigma_X$ (the absolute value because $\sqrt{a^2}=|a|$); and for
independent $X,Y$, $\sigma_{X+Y}=\sqrt{\sigma_X^2+\sigma_Y^2}$. On the running
example $\sigma_X=2\sqrt{2p}$, back in units of stars.

### Markov's and Chebyshev's Inequalities

The standard deviation $\sigma_X$ sets the scale of typical fluctuations. Two
inequalities make this statement precise for *any* distribution. The first,
*Markov's inequality*, turns the mean into a tail bound: a non-negative
variable cannot assign much probability to values far above its mean.

**Proposition (Markov's inequality).** *If $X\ge 0$ and $a>0$, then*

$$
P(X \ge a) \le \frac{E[X]}{a}.
$$
:eqlabel:`eq_mdl-markov`

**Proof.** Let $\mathbf{1}_{X\ge a}$ be the indicator variable that is $1$ when
$X\ge a$ and $0$ otherwise. Then $X \ge a\,\mathbf{1}_{X\ge a}$ *pointwise*: on
the event $\{X\ge a\}$ the right-hand side is $a\le X$, and elsewhere it is
$0\le X$. Taking expectations of both sides and using
$E[\mathbf{1}_{X\ge a}]=P(X\ge a)$ gives $E[X]\ge a\,P(X\ge a)$. $\blacksquare$

So at most a tenth of any non-negative population exceeds ten times its
average: true of incomes, file sizes, and gradient norms alike, with no
distributional assumption whatsoever. On its own the bound is crude, because it
knows only the mean. But feed it the *squared deviation* $(X-\mu_X)^2$, whose
mean is by definition the variance, and it sharpens into the statement we are
after, an inequality due to Bienaymé and to Chebyshev :cite:`Chebyshev.1867`.

**Proposition (Chebyshev's inequality).** *For any $X$ with finite variance and
any $\alpha>0$,*

$$P\bigl(|X - \mu_X| \ge \alpha\sigma_X\bigr) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_mdl-chebyshev`

**Proof.** The variable $Z=(X-\mu_X)^2$ is non-negative with
$E[Z]=\textrm{Var}(X)=\sigma_X^2$ by :eqref:`eq_mdl-var_def`, and the event
$\{|X-\mu_X|\ge\alpha\sigma_X\}$ is exactly the event
$\{Z\ge\alpha^2\sigma_X^2\}$. Markov's inequality :eqref:`eq_mdl-markov` with
$a=\alpha^2\sigma_X^2$ gives
$P(|X-\mu_X|\ge\alpha\sigma_X) \le \sigma_X^2/(\alpha^2\sigma_X^2) = 1/\alpha^2$. $\blacksquare$

**Corollary ($\varepsilon$-form).** *For any $X$ with finite variance and any
$\varepsilon>0$,*

$$
P\bigl(|X-\mu_X| \ge \varepsilon\bigr) \le \frac{\textrm{Var}(X)}{\varepsilon^2}.
$$

Setting $\varepsilon=\alpha\sigma_X$ recovers :eqref:`eq_mdl-chebyshev` when
$\sigma_X>0$; applying Markov's inequality to $(X-\mu_X)^2$ with
$a=\varepsilon^2$ proves the corollary directly and covers the edge case
$\sigma_X=0$, where $X$ equals its mean with probability one and the left-hand
side is $0$. This is the form :numref:`sec_mdl-geometry-linear-algebraic-ops`
invoked for the concentration of random angles.

In words at $\alpha=10$: for *any* distribution at most $1\%$ of the mass lies
ten or more standard deviations from the mean; at least $99\%$ lies strictly
within. The standard deviation is thus a universal yardstick for "how far is
far."

The bound is *sharp*: our running example shows why no smaller constant can
hold in general. With $\mu_X=a$ and $\sigma_X=2\sqrt{2p}$, Chebyshev's
inequality at $\alpha=2$ gives
$P\bigl(|X-a|\ge 4\sqrt{2p}\bigr)\le\tfrac14$. The two outlying
atoms $a\pm2$ are at distance exactly $2$, and as $p$ decreases the threshold
$4\sqrt{2p}$ shrinks with it. :numref:`fig_mdl-prob-chebyshev` shows the three
regimes. For $p>\tfrac18$ the threshold exceeds $2$, no atom lies that far out,
and the left-hand side is $0$: the bound holds with room to spare. At
$p=\tfrac18$ the threshold equals $2$ exactly (then $\sigma_X=1$, so $a\pm2$ are
precisely two standard deviations from the mean): both outlying atoms count
toward the event, the probability is $p+p=\tfrac14$, and the bound is attained
with *equality*, so no constant smaller than $1/\alpha^2$ could work for every
distribution. For $p<\tfrac18$ the threshold drops below $2$, both atoms lie
beyond it, and their combined mass $2p<\tfrac14$ keeps the inequality satisfied.

![Chebyshev's bound on the running three-atom example: masses drawn as vertical bars at $a-2,a,a+2$ with the threshold distance $\alpha\sigma_X=4\sqrt{2p}$ for $\alpha=2$ marked as a horizontal bracket about the mean $a$. The three panels show the regimes $p>1/8$, $p=1/8$ (where the bound is attained with equality), and $p<1/8$ that the text walks through.](../img/mdl-prob-chebyshev.svg)
:label:`fig_mdl-prob-chebyshev`

### Means and Variances in the Continuum

The preceding definitions extend to continuous variables by the same limiting
argument: slice $\mathbb{R}$ into width-$\epsilon$ pieces, apply the discrete definition,
and let $\epsilon\to0$ to turn the sum into an integral. The mean is the
density-weighted average :eqref:`eq_mdl-expectation`, and the variance follows
from its computational form :eqref:`eq_mdl-var_comp`:

$$
\mu_X = \int_{-\infty}^\infty x\,p_X(x)\,dx,
\qquad
\sigma^2_X = \int_{-\infty}^\infty x^2 p_X(x)\,dx - \mu_X^2 .
$$
:eqlabel:`eq_mdl-cont_mean_var`

Every property proved above (linearity, the affine rule, Markov and
Chebyshev) carries over unchanged, since each rested only on linearity and
positivity of the averaging operation.
For the uniform density $U(0,1)$ met at inverse-transform sampling ($p(x)=1$ on
$[0,1]$, zero elsewhere),
$\mu_X=\int_0^1 x\,dx=\tfrac12$ and $\sigma_X^2=\int_0^1
x^2\,dx-\tfrac14=\tfrac13-\tfrac14=\tfrac1{12}$, both elementary integrals.

A cautionary example: the *Cauchy distribution* $p(x)=\frac{1}{\pi(1+x^2)}$ is a
perfectly good density (a table of integrals confirms it integrates to $1$), yet
it has *neither* a finite variance *nor* a well-defined mean. The variance
integral $\int x^2 p(x)\,dx$ diverges because $\frac{x^2}{1+x^2}\to1$, so the
integrand has infinite area. The mean is worse: although the integrand
$\frac{x}{\pi(1+x^2)}$ is *odd* and tempts us to declare $0$ by symmetry, the mean
is defined only when $\int|x|p(x)\,dx<\infty$. Substituting $u=1+x^2$,
$\int_0^\infty \frac{x}{\pi(1+x^2)}\,dx=\frac{1}{2\pi}\int_1^\infty
\frac{du}{u}=+\infty$, and the left tail gives $-\infty$, so the mean is a
meaningless $\infty-\infty$ whose value depends on how the limits are taken, a
*stronger* failure than "the mean is infinite." We can watch both integrals
diverge numerically by extending the integration range and observing that the
partial sums do not converge.

```{.python .input #random-variables-cauchy-diverges}
import numpy as onp
# Cauchy second moment integral over growing ranges: it should NOT converge
for R in [10, 100, 1000]:
    x = onp.arange(-R, R, 0.01)
    integrand = x**2 / (onp.pi * (1 + x**2))
    print(f'integral_-{R}^{R} x^2 p(x) dx = {float(onp.sum(0.01*integrand)):.3f}')
```

The "integral" grows without bound as the range widens: it does not converge, so
the variance is infinite. Distributions whose tails are fat enough that moments
diverge are called *heavy-tailed*, and they are endemic inside machine learning:
minibatch gradient noise :cite:`Simsekli.Sagun.Gurbuzbalaban.2019`,
token frequencies, and the spectra of trained weight
matrices all show heavy tails. :numref:`sec_mdl-concentration-generalization`
classifies tails by how fast averages over them concentrate.

## Several Variables

Machine learning rarely involves one variable in isolation. Pixels $R_{i,j}$ in
an image, prices $P_t$ across time: nearby coordinates are correlated, and a
model that ignores this under-performs (:numref:`sec_mdl-naive_bayes` analyzes
exactly such a model). We need a language for *several*, possibly correlated,
continuous variables, and the multiple integrals of :numref:`sec_mdl-integral_calculus`
supply it.

### Joint and Marginal Densities

For two variables $X,Y$, the same $\epsilon$-interval reasoning gives a *joint
density* $p(x,y)$ with

$$
P(X\approx x \text{ and } Y\approx y \text{ in } \epsilon\text{-boxes}) \approx \epsilon^{2}\,p(x, y),
$$

and the one-variable properties carry over: $p(x,y)\ge0$, $\int_{\mathbb{R}^2}
p(x,y)\,dx\,dy=1$, and $P((X,Y)\in\mathcal{D})=\int_{\mathcal{D}} p(x,y)\,dx\,dy$.
For $n$ variables the joint density $p(\mathbf x)=p(x_1,\ldots,x_n)$ obeys the
same non-negativity and unit-integral rules.

Often we hold a joint density but want the distribution of *one* coordinate,
ignoring the rest: its *marginal distribution*. Starting from
$P(X\in[x,x+\epsilon])\approx\epsilon\,p_X(x)$ and noting $Y$ takes *some* value,
we slice in $y$ as well:

$$
\epsilon\,p_X(x) \approx \sum_i P\bigl(X\in[x,x{+}\epsilon],\ Y\in[\epsilon i,\epsilon(i{+}1)]\bigr)
                 \approx \sum_i \epsilon^{2}\,p_{X,Y}(x,\epsilon i).
$$

Geometrically this integrates the joint density up a vertical strip at $x$, as in
:numref:`fig_mdl-prob-marginal`. Cancelling one $\epsilon$ and recognizing the
remaining sum as an integral over $y$,

$$
p_X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y)\,dy.
$$
:eqlabel:`eq_mdl-marginal`

![A continuous joint density $p_{X,Y}(x,y)$ shown by its contours, with one vertical strip at $x$ highlighted. Integrating the joint density up that strip, over all $y$, gives the height of the marginal density $p_X(x)$ plotted along the $\mathit{x}$-axis below. Sweeping the strip across $x$ traces out the whole marginal.](../img/mdl-prob-marginal.svg)
:label:`fig_mdl-prob-marginal`

To marginalize, then, we *integrate out* the variables we do not care about, the
same operation introduced in :numref:`sec_mdl-integral_calculus`, here given its
probabilistic meaning.

### Conditional Densities and Independence

With the joint and the marginal in hand, the last piece is *conditioning*: how the
density of $X$ should update once we learn that $Y=y$. The discrete rule
$P(X\mid Y)=P(X,Y)/P(Y)$ carries over verbatim to densities. We define the
**conditional density**

$$
p_{X\mid Y}(x\mid y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}, \qquad p_Y(y) > 0.
$$
:eqlabel:`eq_mdl-cond_density`

For each fixed $y$ this is a genuine density in $x$: it is non-negative, and
dividing the joint by exactly $p_Y(y)=\int p_{X,Y}(x,y)\,dx$ is precisely what makes
it integrate to one. Geometrically it is a horizontal slice of the joint surface
at height $y$, renormalized to unit area, as :numref:`fig_mdl-prob-conditional-slice`
shows: cut the joint density along $y=y_0$, then rescale that profile so its area
is one. Rearranging :eqref:`eq_mdl-cond_density`
gives the **chain rule** $p_{X,Y}(x,y)=p_{X\mid Y}(x\mid y)\,p_Y(y)$, and writing it
both ways ($p_{X,Y}=p_{X\mid Y}\,p_Y=p_{Y\mid X}\,p_X$) and equating yields **Bayes'
rule for densities**,

$$
p_{X\mid Y}(x\mid y) = \frac{p_{Y\mid X}(y\mid x)\,p_X(x)}{p_Y(y)},
$$
:eqlabel:`eq_mdl-bayes_density`

which is the identity used for every Bayesian update in this book.

An important special case occurs when the distribution of $X$ is unchanged
after observing $Y$.

**Proposition (independence).** *The following are equivalent: (i) the joint
factorizes, $p_{X,Y}(x,y)=p_X(x)\,p_Y(y)$ for all $x,y$; (ii) the conditional equals
the marginal, $p_{X\mid Y}(x\mid y)=p_X(x)$ whenever $p_Y(y)>0$. When either holds we
call $X$ and $Y$ independent, $X\perp Y$.*

**Proof.** If (i) holds, then $p_{X\mid Y}(x\mid y)=p_X(x)p_Y(y)/p_Y(y)=p_X(x)$, which
is (ii). Conversely, if (ii) holds, multiply by $p_Y(y)$ and use the chain rule:
$p_{X,Y}(x,y)=p_{X\mid Y}(x\mid y)\,p_Y(y)=p_X(x)\,p_Y(y)$, which is (i). $\blacksquare$

As a worked example, take the joint $p_{X,Y}(x,y)=4xy$ on the unit square
$[0,1]^2$ (it integrates to one). The marginal is $p_Y(y)=\int_0^1 4xy\,dx=2y$, so
$p_{X\mid Y}(x\mid y)=4xy/2y=2x$, *independent of $y$*. The conditional never
changes as $y$ varies, so by the proposition $X\perp Y$. This is exactly the
visual test of :numref:`fig_mdl-prob-conditional-slice`: independence is the case where
the renormalized slice $p(x\mid y_0)$ has the *same shape* at every height $y_0$,
since the joint factors and the $y$-factor cancels in the renormalization. A joint
that does not factor this way (say one supported on the triangle $x\le y$) has
a conditional whose support and shape shift with $y$, the signature of dependence.

![Reading off a conditional density. Left: the contours of a joint density $p_{X,Y}(x,y)$ with one horizontal slice at $y=y_0$ highlighted. Right: that slice, $x\mapsto p_{X,Y}(x,y_0)$, renormalized to unit area to give the conditional density $p(x\mid y_0)$ of :eqref:`eq_mdl-cond_density`. When the joint factors, every slice has the same shape after renormalization, regardless of $y_0$, which is precisely independence; when it does not, the slice shape drifts with $y_0$.](../img/mdl-prob-conditional-slice.svg)
:label:`fig_mdl-prob-conditional-slice`

Independence is a strong, all-of-the-distribution statement. It is strictly
stronger than being *uncorrelated* (zero covariance), which constrains only the
linear relationship; we make the gap precise in the covariance subsection below.
This factorized structure is also exactly what the naive Bayes classifier of
:numref:`sec_mdl-naive_bayes` assumes across features to make high-dimensional
densities tractable.

### Conditional Expectation and the Tower Property

The conditional density invites a conditional version of the mean. Averaging $X$
against the slice $p_{X\mid Y}(x\mid y)$ rather than its own marginal gives the
**conditional expectation**

$$
E[X\mid Y{=}y] = \int x\,p_{X\mid Y}(x\mid y)\,dx,
$$
:eqlabel:`eq_mdl-cond_exp`

the average value of $X$ *once $Y$ is known to be $y$*. As $y$ ranges it traces a
function of $y$; feeding in the random $Y$ makes $E[X\mid Y]$ itself a random
variable, a function of $Y$ alone. The same slice average defines
$E[g(X,Y)\mid Y{=}y]=\int g(x,y)\,p_{X\mid Y}(x\mid y)\,dx$ for any function
$g$, and a factor $h(Y)$ is constant within each slice, so it comes out:
$E[h(Y)\,g(X,Y)\mid Y]=h(Y)\,E[g(X,Y)\mid Y]$.
The variance gets a conditional version the
same way: $\textrm{Var}(X\mid Y)=E[X^2\mid Y]-E[X\mid Y]^2$, the computational
form :eqref:`eq_mdl-var_comp` applied within each slice. Two identities then say
that conditioning can
only *reorganize* the mean and spread of $X$, never conjure or destroy them.

**Proposition (laws of total expectation and variance).** *Let $X,Y$ be jointly
continuous with finite variance (for discrete pairs, the same identities hold
with every integral replaced by a sum). Then*

$$
E[X] = E\bigl[E[X\mid Y]\bigr],
\qquad
\textrm{Var}(X) = E\bigl[\textrm{Var}(X\mid Y)\bigr] + \textrm{Var}\bigl(E[X\mid Y]\bigr).
$$
:eqlabel:`eq_mdl-tower`

**Proof.** For the first (the **tower property**), expand the inner expectation
and use the chain rule $p_{X\mid Y}(x\mid y)\,p_Y(y)=p_{X,Y}(x,y)$ to recombine
the slices into the joint, then integrate out $y$ via :eqref:`eq_mdl-marginal`:

$$
E\bigl[E[X\mid Y]\bigr]
 = \int\!\Bigl(\int x\,p_{X\mid Y}(x\mid y)\,dx\Bigr)p_Y(y)\,dy
 = \iint x\,p_{X,Y}(x,y)\,dx\,dy
 = E[X].
$$

The set where $p_Y(y)=0$, on which the conditional density
:eqref:`eq_mdl-cond_density` is undefined, contributes nothing to either
integral: $Y$ lands in it with probability zero, and the joint density vanishes
there as well (it integrates to $p_Y$ in $x$), so both sides ignore it.
Replacing $x$ by $g(x,y)$ in this computation gives
$E\bigl[E[g(X,Y)\mid Y]\bigr]=E[g(X,Y)]$ in the same way.
For the variance, apply the definition of $\textrm{Var}(X\mid Y)$ within
each slice, take the
expectation over $Y$, and add and subtract $E\bigl[E[X\mid Y]\bigr]^2=E[X]^2$:

$$
\begin{aligned}
E\bigl[\textrm{Var}(X\mid Y)\bigr] + \textrm{Var}\bigl(E[X\mid Y]\bigr)
 &= \Bigl(E[X^2] - E\bigl[E[X\mid Y]^2\bigr]\Bigr) \\
 &\qquad + \Bigl(E\bigl[E[X\mid Y]^2\bigr] - E[X]^2\Bigr) \\
 &= E[X^2] - E[X]^2,
\end{aligned}
$$

which is $\textrm{Var}(X)$, using the tower property on $E[X^2]$ and on
$E[X\mid Y]$. $\blacksquare$

The variance law (sometimes called **Eve's law**, after the E-V-V-E pattern of
its right-hand side) has a memorable reading: the spread of $X$ splits
into the *average spread within each $Y$-slice* plus the *spread of the
slice-averages themselves*, unexplained variance plus explained variance. That
decomposition underlies the *analysis of variance* (ANOVA), the classical
method that attributes the variation in data to competing explanatory factors,
and it reappears in latent-variable models, models that explain observed data
through an unobserved cause (:numref:`sec_mdl-latent-em-elbo`). We use the tower property
:eqref:`eq_mdl-tower` repeatedly when an expectation is easier one slice at a
time, most heavily in the maximum-likelihood estimator of
:numref:`sec_mdl-maximum_likelihood` and the variational bounds of
:numref:`sec_mdl-mi-variational-bounds`, where a hidden variable is integrated
out one conditional slice at a time.

### Covariance
:label:`subsec_mdl-covariance`

For a *single* extra summary statistic capturing how two variables move together,
we use the *covariance*. For discrete $X,Y$ taking $(x_i,y_j)$ with probability
$p_{ij}$,

$$\sigma_{XY} = \textrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X)(y_j-\mu_Y)\,p_{ij} = E[XY] - E[X]E[Y],$$
:eqlabel:`eq_mdl-cov_def`

the second form being the exact two-variable analogue of the computational
variance formula :eqref:`eq_mdl-var_comp` (and indeed
$\textrm{Cov}(X,X)=\textrm{Var}(X)$). The covariance is positive when $X$ and $Y$
tend to be large together, negative when one is large as the other is small.

An example makes this concrete. Let $X\in\{1,3\}$ and $Y\in\{-1,3\}$ with

$$
\begin{aligned}
P(X{=}1, Y{=}{-}1) &= \tfrac{p}{2}, &\quad P(X{=}1, Y{=}3) &= \tfrac{1-p}{2}, \\
P(X{=}3, Y{=}{-}1) &= \tfrac{1-p}{2}, &\quad P(X{=}3, Y{=}3) &= \tfrac{p}{2},
\end{aligned}
$$

for a parameter $p\in[0,1]$. At $p=1$ the two are large or small *together*; at
$p=0$ they are *opposed*; at $p=\tfrac12$ all four cells are equally likely and
the two are unrelated. With $\mu_X=2$, $\mu_Y=1$, the definition
:eqref:`eq_mdl-cov_def` gives, after summing the four terms,
$\textrm{Cov}(X,Y)=4p-2$: equal to $+2$ at $p=1$, $-2$ at $p=0$, and $0$ at
$p=\tfrac12$, matching every expectation.

Covariance captures only *linear* co-variation. If $Y$ is uniform on
$\{-2,-1,0,1,2\}$ and $X=Y^2$, then $X$ is a deterministic function of $Y$ yet
$\textrm{Cov}(X,Y)=0$, because the symmetric quadratic relationship has no linear
trend. Zero covariance does *not* mean independent.

The continuous version replaces the sum by the density-weighted integral,

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)\,p(x, y)\,dx\,dy .
$$
:eqlabel:`eq_mdl-cov_cont`

:numref:`fig_mdl-prob-covariance` shows clouds of points whose covariance we tune
from negative through zero to positive: the cloud tilts down, rounds out, then
tilts up. Its bottom row replays the *identical* draws with $Y$ restated in
cents instead of dollars: every sample covariance inflates by a factor of $100$
while no cloud changes shape. Covariance carries the units of its arguments;
only its sign is scale-free, a defect the correlation coefficient below will
repair.

![Top row: scatter clouds of $(X,Y)$ pairs with covariance tuned negative, zero, and positive (left to right), $Y$ in dollars: negative covariance tilts the cloud down-right, zero leaves it round, positive tilts it up-right. Bottom row: the identical draws with $Y$ restated in cents. Every sample covariance is multiplied by $100$ while each panel's correlation $\rho$ (gray) is unchanged: the magnitude of covariance carries units, only its sign is scale-free.](../img/mdl-prob-covariance.svg)
:label:`fig_mdl-prob-covariance`

Two properties we use repeatedly: covariance is *linear in each argument*, with
$\textrm{Cov}(aX+b,Y)=a\,\textrm{Cov}(X,Y)$ and symmetrically
$\textrm{Cov}(X,cY+d)=c\,\textrm{Cov}(X,Y)$ (shifts drop out, scales pull
through, in either slot); and independent variables have zero covariance, since
then $E[XY]=E[X]E[Y]$ in :eqref:`eq_mdl-cov_def`. Covariance also completes the
variance-of-a-sum rule.

**Proposition (variance of a sum).** *For any random variables $X,Y$,*

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\,\textrm{Cov}(X, Y).
$$
:eqlabel:`eq_mdl-var_sum`

**Proof.** Write $\bar X=X-\mu_X$, $\bar Y=Y-\mu_Y$, so $X+Y$ has mean
$\mu_X+\mu_Y$ by linearity :eqref:`eq_mdl-exp_linear` and deviation $\bar X+\bar Y$.
Then by :eqref:`eq_mdl-var_def` and linearity,

$$
\begin{aligned}
\textrm{Var}(X+Y) &= E\bigl[(\bar X+\bar Y)^2\bigr]
 = E[\bar X^2] + 2\,E[\bar X\bar Y] + E[\bar Y^2] \\
 &= \textrm{Var}(X) + 2\,\textrm{Cov}(X,Y) + \textrm{Var}(Y),
\end{aligned}
$$

recognizing $E[\bar X\bar Y]=\textrm{Cov}(X,Y)$ from :eqref:`eq_mdl-cov_def`. $\blacksquare$

When $X$ and $Y$ are independent the cross term vanishes and we recover the
additive rule $\textrm{Var}(X+Y)=\textrm{Var}(X)+\textrm{Var}(Y)$ promised
earlier.

### Correlation

Covariance inherits the *units* of $X$ times $Y$ (inches $\times$ dollars), so
its magnitude is hard to read. Switching $Y$ from dollars to cents multiplies it
by $100$. To get a unit-free measure, divide by something that also scales by
$100$: the standard deviations. The *correlation coefficient* is

$$\rho(X, Y) = \frac{\textrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}}.$$
:eqlabel:`eq_mdl-cor_def`

The normalization is calibrated so that $\rho$ never leaves $[-1,1]$, and the
proof is the Cauchy--Schwarz argument of
:numref:`sec_mdl-geometry-linear-algebraic-ops` wearing probabilistic clothes.

**Proposition (correlation bound).** *For any $X,Y$ with finite, nonzero
variances, $-1\le\rho(X,Y)\le 1$, with equality iff $Y=aX+b$ almost surely
for constants $a\ne0,b$ (a perfect linear relationship up to probability-zero
events).*

**Proof.** A variance is never negative, so $\textrm{Var}(tX+Y)\ge 0$ for every
real $t$. Expanding it by the variance-of-a-sum rule :eqref:`eq_mdl-var_sum`
(with the affine rule :eqref:`eq_mdl-var_affine` on the $tX$ term) gives a
quadratic in $t$,

$$
\textrm{Var}(tX+Y) = \textrm{Var}(X)\,t^2 + 2\,\textrm{Cov}(X,Y)\,t + \textrm{Var}(Y) \;\ge\; 0
\quad\textrm{for every } t.
$$

A quadratic with positive leading coefficient $\textrm{Var}(X)$ that stays
non-negative cannot have two distinct real roots, so its discriminant is $\le 0$:

$$
\bigl(2\,\textrm{Cov}(X,Y)\bigr)^2 - 4\,\textrm{Var}(X)\,\textrm{Var}(Y)\le 0,
\qquad\textrm{i.e.}\qquad
\textrm{Cov}(X,Y)^2 \le \textrm{Var}(X)\,\textrm{Var}(Y).
$$

Dividing by $\sigma_X^2\sigma_Y^2$ gives $\rho^2\le 1$, which is the claim.
Equality forces the discriminant to vanish, so the quadratic has a (repeated)
root $t^\star$ with $\textrm{Var}(t^\star X+Y)=0$; a variable of zero variance is constant almost surely, so
$t^\star X+Y=c$ almost surely, i.e. $Y=-t^\star X+c$ almost surely is affine in
$X$.
Conversely, suppose $Y=aX+b$; the nonzero-variance hypothesis forces $a\neq0$,
since $a=0$ would make $Y$ constant. Using
$\textrm{Cov}(X,aX+b)=a\,\textrm{Var}(X)$ (shifts drop out, scales pull
through) and $\sigma_{aX+b}=|a|\sigma_X$,

$$
\rho(X, aX+b) = \frac{a\,\textrm{Var}(X)}{\sigma_X\cdot|a|\,\sigma_X} = \frac{a}{|a|} = \operatorname{sign}(a),
$$

so equality holds. $\blacksquare$

The two extremes thus read off: $\rho=+1$ for a perfect increasing linear
relationship ($a>0$) and $\rho=-1$ for a perfect decreasing one ($a<0$), in
both cases independent of the scale $|a|$. Correlation measures the direction
and tightness of a linear relationship, never its slope.
On the running discrete
example $\sigma_X=1$, $\sigma_Y=2$, so
$\rho=\frac{4p-2}{2}=2p-1$, sweeping from $-1$ to $+1$ as $p$ does from $0$
to $1$.
:numref:`fig_mdl-prob-correlation` shows clouds at correlation $-0.9$, $0$, and
$0.9$; unlike the covariance clouds, the *spread* is comparable across panels and
only the tilt changes.

![Scatter clouds of $(X,Y)$ at correlation $-0.9$, $0$, and $0.9$ (left to right). Correlation rescales covariance to lie between $-1$ and $1$, so unlike raw covariance the comparison is unit-free: only the tilt, not the overall spread, changes across the panels.](../img/mdl-prob-correlation.svg)
:label:`fig_mdl-prob-correlation`

There is also a geometric reading. Centering so $\mu_X=\mu_Y=0$, the
correlation :eqref:`eq_mdl-cor_def` becomes $\sum_{i,j}x_iy_j\,p_{ij}$ divided
by the two root-mean-squares, which is exactly the cosine formula of
:numref:`sec_mdl-geometry-linear-algebraic-ops` with coordinates weighted by
$p_{ij}$. If standard deviations are "lengths" and correlations are "cosines of
angles," that section's geometric intuition transfers wholesale to random
variables: uncorrelated means orthogonal, $\rho=\pm1$ means parallel.

For $n$ variables we collect every pairwise covariance into a single
**covariance matrix** $\boldsymbol\Sigma$ with entries
$\Sigma_{ij}=\textrm{Cov}(X_i,X_j)$, so the diagonal holds the variances and the
off-diagonal the cross-terms. It is symmetric because $\textrm{Cov}$ is. It is
also positive semidefinite, by a two-line computation that generalizes the
variance-of-a-sum rule :eqref:`eq_mdl-var_sum`: writing $\bar X_i=X_i-\mu_{X_i}$,
the deviation of $\sum_i a_iX_i$ is $\sum_i a_i\bar X_i$, and expanding the
square inside the expectation,

$$
\textrm{Var}\Bigl(\sum_i a_i X_i\Bigr)
 = E\Bigl[\Bigl(\sum_i a_i \bar X_i\Bigr)^{\!2}\Bigr]
 = \sum_{i,j} a_i a_j\,E[\bar X_i \bar X_j]
 = \sum_{i,j} a_i a_j\,\Sigma_{ij}
 = \mathbf a^\top \boldsymbol\Sigma\,\mathbf a .
$$

The left-hand side is a variance and hence non-negative, so
$\mathbf a^\top\boldsymbol\Sigma\,\mathbf a\ge 0$ for every $\mathbf a$:
positive semidefiniteness is the $n$-variable face of $\textrm{Var}\ge 0$.
Because $\boldsymbol\Sigma$ is symmetric PSD, the spectral theorem
(:numref:`sec_mdl-eigendecompositions`) diagonalizes it. The Gaussian $\mathcal
N(\boldsymbol\mu,\boldsymbol\Sigma)$ is built from exactly this matrix: its
density contours are ellipses whose axes point along the eigenvectors of
$\boldsymbol\Sigma$ with half-lengths $\propto\sqrt{\lambda_i}$, the standard
deviation along each principal direction, and we draw and sample these contours
when we meet the multivariate Gaussian in :numref:`sec_mdl-distributions`
(:numref:`fig_mdl-prob-mvn-contours`). Finding those principal axes *is*
principal component analysis, PCA (:numref:`sec_mdl-svd-low-rank`), and the same
deviations-as-vectors geometry underlies least squares
:eqref:`eq_mdl-least-squares`.

### Change of Variables for Densities

One final result closes the section. When we push a random variable
through a function $Y=g(X)$, its density is not generally
$p_X\!\big(g^{-1}(y)\big)$: as the map stretches and compresses space the
density must be re-scaled to keep its total mass at $1$. We already computed
the re-scaling factor. The integral change-of-variables theorem of
:numref:`sec_mdl-integral_calculus` already established that a reparametrization
scales volume locally by the absolute Jacobian determinant,
$\int_{\boldsymbol\phi(U)}f(\mathbf x)\,d\mathbf x=\int_U f(\boldsymbol\phi(\mathbf
x))\,|\det D\boldsymbol\phi(\mathbf x)|\,d\mathbf x$
:eqref:`eq_mdl-change_var_nd`. All a density does is read that theorem with $f=p$ and the
constraint that the total integral stays $1$: *probability mass in equals
probability mass out*.

Take the one-dimensional case first. Assume that $g$ is continuously
differentiable, one-to-one and monotone on the support of $X$, with
$g'(x)\ne0$. The
intuition is that the mass
in a tiny interval must survive the map, $p_Y(y)\,|dy| = p_X(x)\,|dx|$ with
$y=g(x)$; solving for the new density and writing $x=g^{-1}(y)$ gives the
formula. A derivation needing no infinitesimals takes two lines with tools we
already have. For increasing $g$ the events $\{g(X)\le y\}$ and
$\{X\le g^{-1}(y)\}$ coincide, so the c.d.f.s satisfy
$F_Y(y)=F_X\bigl(g^{-1}(y)\bigr)$; differentiating with the chain rule and
:eqref:`eq_mdl-cdf_deriv` gives
$p_Y(y)=p_X\bigl(g^{-1}(y)\bigr)\,\tfrac{dg^{-1}}{dy}(y)$, and for decreasing
$g$ the event flips to $\{X\ge g^{-1}(y)\}$, producing a minus sign that the
absolute value absorbs. Either way we obtain the
**one-dimensional change-of-variables formula**

$$
p_Y(y) = p_X\!\big(g^{-1}(y)\big)\,\left|\frac{d g^{-1}}{dy}(y)\right|.
$$
:eqlabel:`eq_mdl-cov_density_1d`

If $g$ has finitely many regular inverse branches, their contributions add:

$$
p_Y(y)=\sum_{x\in g^{-1}(\{y\})}\frac{p_X(x)}{|g'(x)|},
$$

for values $y$ whose preimages satisfy $g'(x)\ne0$. For example, if $Y=X^2$
and $y>0$, then
$p_Y(y)=[p_X(\sqrt y)+p_X(-\sqrt y)]/(2\sqrt y)$. Critical points where
$g'(x)=0$, infinitely many inverse branches, and transformations that create
atoms require additional analysis.

The derivative factor is exactly the local stretch of the map: where $g$ spreads a
small interval out, the density must drop to keep the area fixed. In several
dimensions, assume that $g$ is a continuously differentiable bijection with a
continuously differentiable inverse and nonzero Jacobian determinant. The
scalar stretch $|dg^{-1}/dy|$ becomes the absolute Jacobian determinant, by
instantiating :eqref:`eq_mdl-change_var_nd` with
$\boldsymbol\phi=g^{-1}$ and $f=p_X$:
$P(Y\in A)=P\bigl(X\in g^{-1}(A)\bigr)=\int_{g^{-1}(A)}p_X(\mathbf x)\,d\mathbf x
=\int_A p_X\bigl(g^{-1}(\mathbf y)\bigr)\,\bigl|\det J_{g^{-1}}(\mathbf y)\bigr|\,d\mathbf y$,
so the integrand on the right is the density of $Y$. Conservation of mass thus reads

$$
\begin{aligned}
p_Y(\mathbf y) &= p_X\!\big(g^{-1}(\mathbf y)\big)\,\big|\det J_{g^{-1}}(\mathbf y)\big|, \\
\text{equivalently}\qquad
\log p_Y(\mathbf y) &= \log p_X(\mathbf x) - \log\big|\det J_{g}(\mathbf x)\big|
\quad\text{at }\mathbf x = g^{-1}(\mathbf y).
\end{aligned}
$$
:eqlabel:`eq_mdl-cov_density`

The logarithmic form is convenient in practice. Passing data through an
invertible network adds $-\log|\det J_g|$ to the log-density. Because
$\log|\det(J_2 J_1)| = \log|\det J_1| + \log|\det J_2|$, a composition of
layers contributes the sum of their log-determinants. This additivity permits
efficient likelihood evaluation in **normalizing flows** :cite:`rezende2015variational`
(:numref:`sec_mdl-continuous-normalizing-flows`).

As a worked example, let $X\sim\mathcal N(0,1)$ and $Y=e^X$, so $g^{-1}(y)=\log y$
and $|dg^{-1}/dy| = 1/y$ for $y>0$. Formula :eqref:`eq_mdl-cov_density_1d` gives the
**log-normal** density

$$
p_Y(y) = \frac{1}{y\sqrt{2\pi}}\,\exp\!\Big(-\tfrac12(\log y)^2\Big), \qquad y>0,
$$

whose $1/y$ prefactor is precisely the change-of-variables correction. The
prediction is directly checkable: sample $X\sim\mathcal N(0,1)$, push the
samples through $e^X$, and a histogram of the results should trace out this
density, $1/y$ correction and all.

```{.python .input #random-variables-change-of-variables}
# Push X ~ N(0, 1) through Y = e^X and compare a histogram of the samples
# with the log-normal density predicted by the change-of-variables formula
np.random.seed(0)
y = np.exp(np.random.randn(100000))
edges = np.linspace(0.0, 8.0, 65)
mids, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
hist = np.histogram(y, edges)[0] / (100000 * width)
analytic = np.exp(-np.log(mids)**2 / 2) / (mids * np.sqrt(2 * np.pi))
d2l.plot(mids, [hist, analytic], 'y', 'p(y)',
         legend=['histogram of exp(X)', 'log-normal density'])
```

The two curves coincide: the mass pushed forward through $e^x$ piles up exactly
where :eqref:`eq_mdl-cov_density_1d` says it must, including the mode at
$y=e^{-1}$. The naive guess $p_X(\log y)$ gets that mode wrong, peaking at
$y=1$ instead. A linear map
$\mathbf y = A\mathbf x$ illustrates the multivariate rule in one stroke: $J_g=A$ is
constant, so the density is rescaled uniformly by $1/|\det A|$; stretch space by
$|\det A|$ and the density thins by the same factor.

## Summary

* A continuous random variable is described by a *probability density* $p(x)\ge0$
  with $\int p\,dx=1$ :eqref:`eq_mdl-density`; the density is not itself a
  probability; only its integral over an interval is,
  $P(X\in(a,b])=\int_a^b p$ :eqref:`eq_mdl-pdf_int_int`. Any single point, and
  more generally any set of *measure zero* (coverable by intervals of
  arbitrarily small total length), has probability zero.
* The *cumulative distribution function* $F(x)=\int_{-\infty}^x p=P(X\le x)$ *is*
  a probability and unifies discrete, continuous, and mixed variables. Density and
  c.d.f. are derivative and integral of each other: $F'=p$
  :eqref:`eq_mdl-cdf_deriv`, the fundamental theorem of calculus.
* The *mean* $\mu_X=E[X]$ locates the distribution and is *linear*
  :eqref:`eq_mdl-exp_linear`: sums of expectations add with no independence
  needed. The *variance* $\textrm{Var}(X)=E[X^2]-\mu_X^2$
  :eqref:`eq_mdl-var_comp` measures spread, scales as
  $\textrm{Var}(aX+b)=a^2\textrm{Var}(X)$ :eqref:`eq_mdl-var_affine`, and its root
  is the *standard deviation*, back in the original units.
* *Markov's inequality* :eqref:`eq_mdl-markov` caps the tail of a non-negative
  variable by its mean; applied to $(X-\mu_X)^2$ it yields *Chebyshev's
  inequality* :eqref:`eq_mdl-chebyshev`: at most a $1/\alpha^2$ fraction of any
  distribution lies $\alpha$
  standard deviations or farther from the mean.
* Several variables are handled by a *joint density*; *marginals* come from
  integrating out the unwanted variables :eqref:`eq_mdl-marginal`. *Covariance*
  measures linear co-variation, giving $\textrm{Var}(X+Y)=\textrm{Var}(X)+
  \textrm{Var}(Y)+2\textrm{Cov}(X,Y)$ :eqref:`eq_mdl-var_sum`, and *correlation*
  normalizes it to $[-1,1]$ as a cosine between centered variables. Zero
  covariance does not imply independence.
* Pushing $X$ through an invertible map $g$ rescales its density by the local
  stretch of the map: $p_Y(y)=p_X(g^{-1}(y))\,|dg^{-1}/dy|$
  :eqref:`eq_mdl-cov_density_1d`, with the absolute Jacobian determinant taking
  over in several dimensions :eqref:`eq_mdl-cov_density`, the log-det form that
  powers normalizing flows.

## Exercises

1. **Power-law tail.** Let $X$ have the density $p(x) = 1/x^2$ for
   $x \ge 1$ and $p(x) = 0$ otherwise.
    1. Verify that $p$ is a density, and compute $P(X > 2)$ and the c.d.f.
       $F(x)$ for every real $x$.
    1. Find the quantile function $F^{-1}(q)$ for $q \in (0, 1)$, the
       median, and the other two quartiles. Write down the
       inverse-transform sampler $X = F^{-1}(U)$ of
       :numref:`fig_mdl-prob-inverse-transform` for this law.
    1. Show that $E[X] = \infty$. Clipping $X$ at a level $c > 1$, as
       gradient clipping does for a gradient norm, gives $\min(X, c)$.
       Compute $E[\min(X, c)]$, and describe how it grows with $c$.
    1. For the Cauchy density $q(x) = 1/\bigl(\pi(1+x^2)\bigr)$, compute
       $\int_{-R}^{R} x\,q(x)\,dx$ and $\int_{-R}^{2R} x\,q(x)\,dx$ and
       their limits as $R \to \infty$. What do the two limits say about the
       mean of the Cauchy law? Why can no such ambiguity arise for $E[X]$?
   :exlabel:`ex_mdl-random-variables-power-law-tail`

1. **Rectified Gaussian.** Let $X \sim \mathcal{N}(0, 1)$ have the c.d.f.
   $\Phi$, and let $Y = \max(0, X)$. This is the activation of a ReLU unit
   with a standard normal pre-activation, a common model of a unit at
   initialization. The analysis of :numref:`subsec_he_init` assumes only
   that the pre-activations are symmetric about zero.
    1. Express the c.d.f. $F_Y$ through $\Phi$ and sketch it. Where does
       $Y$ have an atom, and what is its mass? Is the law of $Y$ discrete,
       absolutely continuous, or mixed?
    1. Explain why $Y$ has no density. Which hypothesis of the
       change-of-variables formula :eqref:`eq_mdl-cov_density_1d` fails for
       the ReLU?
    1. Clipping to $[-c, c]$ for a threshold $c > 0$, as the hard-tanh
       activation does for $c = 1$, gives $C = \min(\max(X, -c), c)$. Write
       down the c.d.f. of $C$, and find the threshold $c$ at which the atoms
       of $C$ together hold $10\%$ of the mass. The equation has no
       closed-form solution; use tabulated values of $\Phi$, or `math.erf`
       with $\Phi(z) = \tfrac12\bigl(1 + \operatorname{erf}(z/\sqrt{2})\bigr)$.
   :exlabel:`ex_mdl-random-variables-rectified-gaussian`

1. [code] **Logistic law and calibration.** The logistic distribution has
   the c.d.f. $F(x) = \operatorname{sigmoid}(x) = 1/(1 + e^{-x})$.
    1. Find its density with :eqref:`eq_mdl-cdf_deriv` and its quantile
       function. What are its quartiles?
    1. Let $X$ have a continuous, strictly increasing c.d.f. $F$. Show that
       $F(X)$ is uniform on $[0, 1]$. This is the converse of
       inverse-transform sampling.
    1. A probabilistic regression model predicts for each input $i$ a
       logistic law for the target $y_i$, centered at the correct location
       $\mu_i$ but with a scale $s$, that is, the c.d.f.
       $\operatorname{sigmoid}\bigl((y - \mu_i)/s\bigr)$. The actual errors
       $y_i - \mu_i$ follow the standard logistic law. Draw $10^4$ errors
       with the quantile function from (a), compute
       $u_i = \operatorname{sigmoid}\bigl((y_i - \mu_i)/s\bigr)$ for
       $s \in \{0.5, 1, 2\}$, and plot the three histograms of the $u_i$.
       Which shape reveals predicted laws that are too narrow, and which
       reveals laws that are too wide? Such histograms are a standard check
       of the calibration of probabilistic forecasts
       :cite:`Gneiting.Balabdaoui.Raftery.2007`.
   :exlabel:`ex_mdl-random-variables-logistic-calibration`

1. [code] **Classes missing from a minibatch.** A minibatch holds $B$
   labels drawn independently and uniformly from $K$ classes. Let $M$ be
   the number of classes that do not appear in it, and write
   $M = \sum_{k=1}^K I_k$, where $I_k = 1$ if class $k$ is missing and
   $I_k = 0$ otherwise.
    1. Compute $E[M]$ with linearity :eqref:`eq_mdl-exp_linear`, and
       evaluate it for $K = 1000$ and $B = 256$.
    1. Compute $\textrm{Cov}(I_j, I_k)$ for $j \neq k$. Are the indicators
       independent?
    1. Extend the variance-of-a-sum rule :eqref:`eq_mdl-var_sum` to $K$
       terms, and compute the standard deviation of $M$ for $K = 1000$ and
       $B = 256$. Compare it with the value that independent indicators
       would give, and check both against $10^4$ simulated minibatches. How
       can covariances as small as those in (b) change the answer this much?
   :exlabel:`ex_mdl-random-variables-missing-classes`

1. **Markov versus Chebyshev.** A non-negative random variable $X$ has
   mean $1$ and standard deviation $2$, as a gradient norm with a long
   right tail might.
    1. Bound $P(X \ge 9)$ with Markov's inequality :eqref:`eq_mdl-markov`
       and with Chebyshev's inequality :eqref:`eq_mdl-chebyshev`. For which
       thresholds $t > 1$ does Markov's inequality give the smaller bound on
       $P(X \ge t)$?
    1. Find a non-negative $X$ with mean $1$ and standard deviation $2$ for
       which $P(X \ge 5)$ equals Markov's bound.
    1. A dataset of $n$ numbers has mean $1$ and standard deviation $2$,
       computed with the divisor $n$. Apply Chebyshev's inequality to the
       empirical distribution, which puts mass $1/n$ on each number, to show
       that at most $1/16$ of the numbers exceed $9$.
    1. Now let $1$ and $2$ be the mean and standard deviation of a
       population, and draw $20$ points from it independently. Apply
       Markov's inequality to the number $N$ of sample points above $9$ to
       bound $P(N \ge 5)$. Why does (c) not rule out this event?
   :exlabel:`ex_mdl-random-variables-markov-chebyshev`

1. [code] **Moments of a mixture.** The
   `#random-variables-density-to-probability` cell integrates the mixture
   density $p(x) = 0.2\,\mathcal{N}(x; 3, 1) + 0.8\,\mathcal{N}(x; -1, 1)$.
   Let $Z$ be the component label, with $P(Z = 1) = 0.2$ and
   $X \mid Z{=}1 \sim \mathcal{N}(3, 1)$, and with $P(Z = 2) = 0.8$ and
   $X \mid Z{=}2 \sim \mathcal{N}(-1, 1)$.
    1. Compute $E[X]$ and $\textrm{Var}(X)$ with the laws of total
       expectation and variance :eqref:`eq_mdl-tower`, without integrating.
    1. Extend the cell to compute both quantities as Riemann sums of
       :eqref:`eq_mdl-cont_mean_var`, and compare them with (a).
    1. Compute $P\bigl(|X - \mu_X| \ge \alpha\sigma_X\bigr)$ by a Riemann
       sum for $\alpha \in \{1.5, 2, 3\}$, and compare each value with
       Chebyshev's bound :eqref:`eq_mdl-chebyshev`.
   :exlabel:`ex_mdl-random-variables-mixture-moments`

1. **Uncorrelated but dependent.** Whitening, for example with the
   principal components of :numref:`sec_mdl-svd-low-rank`, makes
   coordinates uncorrelated. The following pair has Gaussian coordinates
   and zero covariance. Let $X \sim \mathcal{N}(0, 1)$, let $S$ be a random
   sign independent of $X$ with $P(S = 1) = P(S = -1) = \tfrac12$, and let
   $Y = SX$.
    1. Show that $Y \sim \mathcal{N}(0, 1)$ by computing its c.d.f., and
       that $\textrm{Cov}(X, Y) = 0$.
    1. Compare $P(|X| \le 1, |Y| \le 1)$ with
       $P(|X| \le 1)\,P(|Y| \le 1)$. Are $X$ and $Y$ independent?
    1. Show that $X + Y$ has an atom at $0$, so that $X + Y$ is not
       Gaussian. On the event $\{X + Y = 0\}$ the point $(X, Y)$ lies on
       the line $y = -x$, and a line in the plane can be covered by
       countably many boxes of arbitrarily small total area. Conclude that the pair $(X, Y)$ has no joint
       density, although each coordinate has one.
   :exlabel:`ex_mdl-random-variables-uncorrelated-dependent`

1. **Averaging correlated estimates.** The Monte Carlo estimate of
   $E[g(X)]$ from draws $x_1, \ldots, x_n$ is
   $\hat g_n = \tfrac1n \sum_{i=1}^n g(x_i)$.
    1. Let the draws be independent. Extend :eqref:`eq_mdl-var_sum` to $n$
       terms and combine it with :eqref:`eq_mdl-var_affine` to show that
       $\textrm{Var}(\hat g_n) = \textrm{Var}(g(X))/n$.
    1. Now let $Y_1, \ldots, Y_n$ each have variance $\sigma^2$, with
       correlation $\rho$ between every pair, as the predictions of $n$
       ensemble members trained on the same data might. Show that

       $$\textrm{Var}\Bigl(\frac1n\sum_{i=1}^n Y_i\Bigr) = \sigma^2\Bigl(\rho + \frac{1-\rho}{n}\Bigr).$$

       What does averaging achieve as $n \to \infty$?
    1. Use $\mathbf{a}^\top \boldsymbol\Sigma\, \mathbf{a} \ge 0$ for the
       covariance matrix $\boldsymbol\Sigma$ of $Y_1, \ldots, Y_n$ to show
       that $\rho \ge -1/(n-1)$. Give variables that attain this bound.
   :exlabel:`ex_mdl-random-variables-using-textrmvar-x-y`

1. [code] **Sigmoid of a Gaussian.** Let $X \sim \mathcal{N}(0, s^2)$ be
   the pre-activation of a sigmoid unit, and let
   $Y = \operatorname{sigmoid}(X)$, whose inverse is
   $\operatorname{logit}(y) = \log\bigl(y/(1-y)\bigr)$.
    1. Use :eqref:`eq_mdl-cov_density_1d` to find the density $p_Y$ on
       $(0, 1)$.
    1. For which $s$ is $y = \tfrac12$ a local minimum of $p_Y$? Writing
       $\log p_Y$ as a function of $z = \operatorname{logit}(y)$ simplifies
       the computation.
    1. Modify the `#random-variables-change-of-variables` cell to compare a
       histogram of $Y$ with $p_Y$ for $s \in \{1, \sqrt{2}, 3\}$. What does
       the histogram at $s = 3$ imply for the derivative
       $\operatorname{sigmoid}'(X) = Y(1 - Y)$, through which
       backpropagation passes the gradient of these units?
   :exlabel:`ex_mdl-random-variables-sigmoid-gaussian`

1. [code] **Sampling from a staircase.** ● The inverse-transform sampler of
   :numref:`fig_mdl-prob-inverse-transform` assumes a continuous, strictly
   increasing c.d.f. A categorical distribution, such as the softmax over
   the vocabulary from which a language model samples its next token, has
   a staircase c.d.f. instead. For any c.d.f. $F$, define the generalized
   inverse $F^{-}(u) = \min\{x : F(x) \ge u\}$ for $u \in (0, 1)$; the
   minimum exists because every c.d.f. is right-continuous.
    1. Show that $F^{-}(u) \le x$ holds exactly when $u \le F(x)$. Conclude
       that $F^{-}(U)$ has the c.d.f. $F$ when $U$ is uniform on $[0, 1]$.
    1. Let $X$ take the value $k \in \{1, \ldots, K\}$ with probability
       $p_k$, and let $c_k = p_1 + \cdots + p_k$. Show that $F^{-}(u)$ is the
       smallest $k$ with $c_k \ge u$. Implement the sampler with NumPy's
       `cumsum` and `searchsorted`, and check the frequencies of $10^5$ draws
       for $\mathbf{p} = (0.1, 0.2, 0.3, 0.25, 0.15)$. What does it cost
       to draw $m$ samples from one distribution over $K$ classes? What does
       each draw cost when the distribution changes at every draw, as in
       autoregressive decoding?
    1. Apply the softmax to $50{,}000$ logits, each $3$ times a standard
       normal draw, store the probabilities in float32, and compute
       $c_K - 1$ from a float32 `cumsum`. How often does a uniform draw
       exceed $c_K$, and what does `searchsorted` return then? Give two ways
       to make the sampler safe.
   :exlabel:`ex_mdl-random-variables-staircase-sampling`

1. **One-sided Chebyshev.** ● Chebyshev's inequality
   :eqref:`eq_mdl-chebyshev` bounds both tails at once. Cantelli's
   inequality bounds one tail: for $X$ with mean $\mu$ and variance
   $\sigma^2$, and for every $t > 0$,

   $$P(X - \mu \ge t) \le \frac{\sigma^2}{\sigma^2 + t^2}.$$

    1. For every $u \ge 0$, apply Markov's inequality :eqref:`eq_mdl-markov`
       to $(X - \mu + u)^2$ to show that
       $P(X - \mu \ge t) \le (\sigma^2 + u^2)/(t + u)^2$. Choose $u$ to prove
       Cantelli's inequality.
    1. Write the bound for $t = \alpha\sigma$, and compare it with
       :eqref:`eq_mdl-chebyshev` at $\alpha = 1$ and $\alpha = 2$.
    1. For given $\mu$, $\sigma$ and $t$, find a distribution on two points
       with mean $\mu$ and variance $\sigma^2$ that attains the bound.
   :exlabel:`ex_mdl-random-variables-one-sided-chebyshev`

1. **Conditional mean as best predictor.** ● A regression model $h$
   predicts a target $X$ from an input $Y$, and training minimizes the
   expected squared error $E\bigl[(X - h(Y))^2\bigr]$ over functions $h$.
    1. Write $X - h(Y) = \bigl(X - E[X \mid Y]\bigr) + \bigl(E[X \mid Y] - h(Y)\bigr)$
       and use the tower property :eqref:`eq_mdl-tower` to show that the
       cross term has mean zero. Conclude that $h^\star(Y) = E[X \mid Y]$
       minimizes the error, and that the minimum equals
       $E\bigl[\textrm{Var}(X \mid Y)\bigr]$.
    1. The fraction of variance that a predictor explains is
       $R^2 = 1 - E\bigl[(X - h(Y))^2\bigr]/\textrm{Var}(X)$. Use (a) and the
       law of total variance to show that every predictor has
       $R^2 \le \textrm{Var}\bigl(E[X \mid Y]\bigr)/\textrm{Var}(X)$.
       Evaluate this ceiling for predicting $X$ from the label $Z$ of
       problem :exref:`ex_mdl-random-variables-mixture-moments`.
    1. Given the input, the target is $-1$ or $+1$ with probability
       $\tfrac12$ each. Which prediction minimizes the expected squared
       error, and how often does the target take that value? Which
       predictions minimize the expected absolute error
       $E\bigl[|X - h(Y)|\bigr]$?
   :exlabel:`ex_mdl-random-variables-best-predictor`

1. [code] **Affine coupling layer.** ● Normalizing flows stack invertible
   maps whose log-determinants in :eqref:`eq_mdl-cov_density` are cheap to
   compute. The affine coupling layer of RealNVP :cite:`dinh2017density`
   maps $\mathbf{x} \in \mathbb{R}^2$ to

   $$y_1 = x_1, \qquad y_2 = x_2\, e^{s(x_1)} + t(x_1),$$

   where $s$ and $t$ are arbitrary functions, for example small neural
   networks.
    1. Write down the inverse map and the Jacobian $J_g(\mathbf{x})$. Show
       that $\log|\det J_g(\mathbf{x})| = s(x_1)$, however complicated $s$
       and $t$ are.
    1. Let $X_1$ and $X_2$ be independent standard normal variables, and
       take $s(x_1) = \tanh(x_1)$ and $t(x_1) = x_1^2/2$. Use
       :eqref:`eq_mdl-cov_density` to write $\log p_{\mathbf{Y}}(\mathbf{y})$
       in closed form. With NumPy, evaluate $p_{\mathbf{Y}}$ on a grid over
       $[-6, 6] \times [-12, 24]$ with spacing $0.02$ and check that its
       Riemann sum is close to $1$. Check the log-determinant against the
       determinant of a finite-difference Jacobian at a few random points.
    1. The first coordinate passes through the layer unchanged. What is the
       law of $Y_1$, whatever $s$ and $t$ are? How must a second coupling
       layer be arranged so that the composition can change both marginals,
       and what is the log-determinant of the composition?
   :exlabel:`ex_mdl-random-variables-affine-coupling`

1. [extended] **Averages of heavy tails.** ●●
   :citet:`Simsekli.Sagun.Gurbuzbalaban.2019` argued that the gradient
   noise of SGD in deep networks is heavy-tailed and better described by an
   $\alpha$-stable law, whose variance is infinite for $\alpha < 2$, than by
   a Gaussian. This project measures how a Monte Carlo average behaves in
   that regime and then tests the premise on a small model. The Pareto
   density $p(x) = \alpha x^{-\alpha-1}$ on $x \ge 1$, with $\alpha > 0$,
   contains the density of problem
   :exref:`ex_mdl-random-variables-power-law-tail` as the case $\alpha = 1$.
    1. Generalize problem :exref:`ex_mdl-random-variables-power-law-tail`b
       and c to this family: show that $E[X^k] < \infty$ exactly when
       $k < \alpha$, compute the mean $\mu$ for $\alpha > 1$, and write down
       an inverse-transform sampler.
    1. Let $1 < \alpha < 2$, so that the mean is finite but the variance is
       not. Show that the largest of $n$ independent draws exceeds $t$ with
       probability close to $n t^{-\alpha}$ when $n t^{-\alpha}$ is small,
       and find the scale $t_n$ that the largest draw typically reaches.
       Argue heuristically that the error $|\bar X_n - \mu|$ of the sample
       mean is of order $n^{1/\alpha - 1}$ rather than $n^{-1/2}$. The
       generalized central limit theorem that Simsekli et al. invoke makes
       this argument precise.
    1. For $\alpha \in \{1.5, 2.5, 3\}$ and $n \in \{10, 10^2, \ldots, 10^5\}$,
       estimate the median of $|\bar X_n - \mu|$ over $400$ repetitions, and
       fit the slope of the logarithm of the median error against $\log n$.
       Compare the slope at $\alpha = 1.5$ with the rate $n^{1/\alpha - 1}$
       from (b). Compare the slopes at $\alpha = 2.5$ and $\alpha = 3$ with
       the rate $n^{-1/2}$ that holds when the variance is finite.
    1. In each repetition, form the interval
       $\bar X_n \pm 1.96\,\hat\sigma_n/\sqrt{n}$, where $\hat\sigma_n$ is
       the sample standard deviation; for Gaussian averages it covers $\mu$
       with probability about $95\%$. Record how often it covers $\mu$.
       Explain the coverage at $\alpha = 1.5$ and $n = 10^5$, given that
       $\hat\sigma_n$ is finite in every run. What does this imply for error
       bars computed from heavy-tailed samples, such as minibatch gradients?
    1. The Hill estimator :cite:`Hill.1975` estimates the tail index from
       the $k$ largest of $m$ positive draws
       $X_{(1)} \ge X_{(2)} \ge \cdots \ge X_{(m)}$ as

       $$\hat\alpha_k = \Bigl(\frac1k \sum_{i=1}^k \log\frac{X_{(i)}}{X_{(k+1)}}\Bigr)^{-1}.$$

       Check it on $10^5$ Pareto draws for $\alpha \in \{1.5, 3\}$ and
       $k \in \{10^2, 10^3, 10^4\}$. Then train the softmax regression of
       :numref:`sec_softmax_concise`. At the trained parameters, collect
       the coordinates of the difference between the minibatch gradient and
       the full-batch gradient over $1000$ minibatches, for batch sizes $32$
       and $256$. Plot $\hat\alpha_k$ of their absolute values against $k$
       for both batch sizes. Is the estimate below $2$? How much does the
       conclusion depend on $k$?
   :exlabel:`ex_mdl-random-variables-heavy-tail-averages`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/415)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1094)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1095)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1095)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.1]{.kicker}

Densities, moments, and joint distributions<br>**the continuous language of deep learning**.
:::
:::

::: {.slide title="Why continuous probability?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Pixels, weights, activations, and noise are commonly modeled as continuous.
A single exact outcome then has probability **zero**; probabilities are
assigned to intervals or regions.

- Density $p$, mean, variance, covariance, the matrix $\boldsymbol\Sigma$.
- The toolkit behind SGD, normalizing flows, and Bayesian inference.

::: {.d2l-note}
The continuous analogue of discrete probability: same ideas, **integration
in place of summation**.
:::
:::

::: {.col .fig .big}
@fig:mdl-prob-pdf-area
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Densities and the c.d.f.]{.dtitle}

[the dartboard argument, $p\ge 0$, $\int p = 1$, and $F' = p$]{.dsub}
:::
:::

::: {.slide title="The density appears"}
[The dartboard]{.kicker}

Shrink the bin around $x$ by $10\times$ and the probability it catches
shrinks by $10\times$. The surviving constant of proportionality is the
**density**:

$$P\bigl(X \in [x, x+\epsilon]\bigr) \approx \epsilon\, p(x).$$

. . .

If $X$ has a density, then $P(X = x) = 0$ for every fixed point, while
$p(x)$ may exceed $1$: a density is a *rate*, not a probability. Discrete
and mixed distributions may instead place positive mass at individual points.
:::

::: {.slide title="Probability is area"}
[Two defining properties]{.kicker}

A density is non-negative and integrates to one; probability of an
interval is the area beneath it:

$$p(x)\ge 0, \qquad \int_{-\infty}^{\infty} p(x)\,dx = 1, \qquad
P\bigl(X\in(a,b]\bigr) = \int_a^b p(x)\,dx.$$

. . .

A Riemann sum over a two-bump mixture confirms the total mass is $1$ and
recovers $P(-2 < X \le 3)$ from the density alone:

@!random-variables-density-to-probability
:::

::: {.slide title="The c.d.f. and the FTC"}
[The c.d.f.]{.kicker}

::: {.cols .vc}
::: {.col}
The cumulative distribution accumulates the density,
$F(x) = \int_{-\infty}^x p = P(X \le x)$, rising monotonically from $0$
to $1$. The same number two ways: the area under $p$ over $(a,b]$ equals
the rise $F(b)-F(a)$.

::: {.d2l-note .rule}
**Proposition.** If $p$ is continuous, $F'(x) = p(x)$: the density is
the *slope* of the c.d.f. (fundamental theorem of calculus).
:::
:::

::: {.col .fig}
@fig:mdl-prob-pdf-cdf
:::
:::
:::

::: {.slide title="The quantile function is a sampler"}
[The c.d.f.]{.kicker}

::: {.cols .vc}
::: {.col}
The c.d.f. covers every kind of variable (staircase, smooth curve,
mixture), and its inverse $F^{-1}(q)$, the **quantile function**, turns
uniform noise into samples from *any* distribution:

::: {.d2l-note .rule}
**Proposition (inverse transform).** If $U\sim\mathrm{Uniform}[0,1]$,
then $X = F^{-1}(U)$ has c.d.f. $F$.
*Proof.* $\{F^{-1}(U)\le x\} = \{U \le F(x)\}$, and the uniform gives
that event probability $F(x)$. $\blacksquare$
:::

Where $F$ is steep (where the density is large), uniform levels land
densely: the slope does the shaping.
:::

::: {.col .fig}
@fig:mdl-prob-inverse-transform
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Mean, variance, and tail bounds]{.dtitle}

[linearity, the computational form, Markov, Chebyshev]{.dsub}
:::
:::

::: {.slide title="The mean is linear"}
[Moments]{.kicker}

The mean is the density-weighted center,
$\mu_X = \mathbb E[X] = \int x\,p(x)\,dx$.

::: {.d2l-note .rule}
**Proposition.** $\mathbb E[aX+b] = a\,\mathbb E[X] + b$ and
$\mathbb E[X+Y] = \mathbb E[X] + \mathbb E[Y]$ (**no independence
needed**).
:::

The sum rule follows by splitting the joint expectation through the
marginals; it is why we can move $\mathbb E$ freely through a network's
losses.
:::

::: {.slide title="Variance and its computational form"}
[Moments]{.kicker}

Variance is the mean squared deviation,
$\sigma_X^2 = \mathbb E[(X-\mu_X)^2]$.

::: {.d2l-note .rule}
**Proposition.** $\operatorname{Var}(X) = \mathbb E[X^2] - \mu_X^2$
and $\operatorname{Var}(aX+b) = a^2\operatorname{Var}(X)$.
:::

. . .

*Proof.* Expand $(X-\mu_X)^2 = X^2 - 2\mu_X X + \mu_X^2$ and take
expectations. A shift $b$ cancels inside the deviation $a(X-\mu_X)$, so
only $a^2$ survives the square. $\blacksquare$
:::

::: {.slide title="Markov's inequality"}
[Tail bounds]{.kicker}

For a **non-negative** $X$ and any $a>0$,

$$P(X \ge a) \le \frac{\mathbb E[X]}{a}.$$

. . .

*Proof.* Pointwise $X \ge a\,\mathbf 1_{\{X\ge a\}}$; take expectations
and use $\mathbb E[\mathbf 1_{\{X\ge a\}}] = P(X\ge a)$. $\blacksquare$

::: {.d2l-note}
No distributional assumption at all: it bounds gradient norms, incomes,
and file sizes alike.
:::
:::

::: {.slide title="Chebyshev: σ sets the scale"}
[Tail bounds]{.kicker}

::: {.cols .vc}
::: {.col}
Standard deviation measures distance in *universal* units:

$$P\bigl(|X-\mu_X| \ge \alpha\,\sigma_X\bigr) \le \frac{1}{\alpha^2}.$$

*Proof.* Apply Markov to $Z = (X-\mu_X)^2$ with $a = \alpha^2\sigma_X^2$.
$\blacksquare$

The bound is **sharp**: the three-atom example attains equality at
$p=\tfrac18$.
:::

::: {.col .fig}
@fig:mdl-prob-chebyshev
:::
:::
:::

::: {.slide title="When moments fail: Cauchy"}
[Counterexample]{.kicker}

For $p(x) = \dfrac{1}{\pi(1+x^2)}$ both summaries fail, at two speeds.
The variance integrand tends to $\tfrac1\pi$, so
$\int_{-R}^{R} x^2 p\,dx = \tfrac{2}{\pi}(R - \arctan R)$ grows
**linearly**, multiplying by ten per decade:

@!random-variables-cauchy-diverges

. . .

The mean fails more subtly: the odd integrand tempts "$0$ by symmetry,"
but a mean needs $\int |x|\,p\,dx < \infty$, and *that* integral
diverges like $\tfrac{1}{\pi}\log(1+R^2)$. The mean is a meaningless
$\infty - \infty$ whose value depends on how the limits are taken.
Absolute integrability is what a finite mean requires; symmetry alone
proves nothing.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Several variables]{.dtitle}

[marginals, conditioning, covariance, change of variables]{.dsub}
:::
:::

::: {.slide title="Joint and marginal densities"}
[Several variables]{.kicker}

::: {.cols .vc}
::: {.col}
A joint density $p(x,y)\ge 0$ integrates to $1$ over the plane.
**Marginalize** by integrating out the variable you do not care about:

$$p_X(x) = \int_{-\infty}^{\infty} p_{X,Y}(x,y)\,dy.$$

Integrate the joint up a vertical strip at $x$ to get the marginal
height there.
:::

::: {.col .fig}
@fig:mdl-prob-marginal
:::
:::
:::

::: {.slide title="Conditioning and independence"}
[Conditioning]{.kicker}

::: {.cols .vc}
::: {.col}
Slice the joint at $Y=y$ and renormalize:

$$p_{X\mid Y}(x\mid y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}.$$

Chain rule $p_{X,Y} = p_{X\mid Y}\,p_Y$ gives **Bayes' rule** for
densities. $X \perp Y$ exactly when the joint factorizes: the slice
shape stops depending on $y$.
:::

::: {.col .fig}
@fig:mdl-prob-conditional-slice
:::
:::
:::

::: {.slide title="Conditional expectation & the tower"}
[Conditioning]{.kicker}

$\mathbb E[X\mid Y=y] = \int x\,p_{X\mid Y}(x\mid y)\,dx$ averages within a
slice.

::: {.d2l-note .rule}
**Tower / Eve's law.**
$\mathbb E[X] = \mathbb E\bigl[\mathbb E[X\mid Y]\bigr]$ and
$\operatorname{Var}(X) = \mathbb E[\operatorname{Var}(X\mid Y)] +
\operatorname{Var}(\mathbb E[X\mid Y])$.
:::

Total variance = **unexplained** (within-slice spread) +
**explained** (spread of slice means): the decomposition behind ANOVA
and latent-variable models.
:::

::: {.slide title="Covariance"}
[Co-variation]{.kicker}

::: {.cols .vc}
::: {.col}
$\operatorname{Cov}(X,Y) = \mathbb E[XY] - \mathbb E[X]\,\mathbb E[Y]$
measures *linear* co-variation, in mixed units:

$$\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)
+ 2\operatorname{Cov}(X,Y).$$

::: {.d2l-note}
Zero covariance is **weaker** than independence: with $Y$ uniform on
$\{-2,\dots,2\}$ and $X=Y^2$, covariance is $0$ yet $X$ is a function
of $Y$.
:::
:::

::: {.col .fig}
@fig:mdl-prob-covariance
:::
:::
:::

::: {.slide title="Correlation is a cosine"}
[Co-variation]{.kicker}

::: {.cols .vc}
::: {.col}
Normalize covariance to kill the units:
$\rho = \operatorname{Cov}(X,Y)/(\sigma_X\sigma_Y) \in [-1,1]$.

*Proof of the bound.* $\operatorname{Var}(tX+Y)\ge 0$ is a non-negative
quadratic in $t$; its discriminant gives
$\operatorname{Cov}^2 \le \operatorname{Var}(X)\operatorname{Var}(Y)$,
with equality iff $Y = aX+b$. $\blacksquare$

So $\rho = \cos\theta$ between centered variables, by the same
Cauchy–Schwarz argument.
:::

::: {.col .fig}
@fig:mdl-prob-correlation
:::
:::
:::

::: {.slide title="The covariance matrix → PCA"}
[Co-variation]{.kicker}

Stack pairwise covariances into $\Sigma_{ij} = \operatorname{Cov}(X_i,X_j)$.
It is symmetric and **positive semidefinite**, since
$\mathbf a^\top\Sigma\,\mathbf a = \operatorname{Var}\bigl(\textstyle\sum_i a_i X_i\bigr) \ge 0$.

::: {.d2l-note .rule}
Gaussian contours are ellipses whose axes are the **eigenvectors** of
$\Sigma$ with half-lengths $\propto\sqrt{\lambda_i}$: finding them is
PCA (the spectral theorem in action).
:::
:::

::: {.slide title="Change of variables for densities"}
[Pushforward]{.kicker}

Suppose $g$ is one-to-one and differentiable, with nonzero derivative, and
set $Y=g(X)$. Conservation of probability gives

$$p_Y(y) = p_X\bigl(g^{-1}(y)\bigr)\,\left|\frac{dg^{-1}}{dy}(y)\right|.$$

If a value $y$ has several regular preimages, sum this contribution over
them. Critical points and atoms require separate care.

. . .

For a differentiable bijection in many dimensions, with a differentiable
inverse and nonsingular Jacobian, this becomes

$$\log p_Y(\mathbf y) = \log p_X(\mathbf x) - \log\bigl|\det J_g(\mathbf x)\bigr|.$$

::: {.d2l-note .rule}
Log-determinants **add** under composition, so a normalizing flow sums one
$-\log|\det J_g|$ term per layer.
:::
:::

::: {.slide title="Change of variables, verified" layout="tight"}
[Pushforward]{.kicker}

For $X\sim\mathcal N(0,1)$ and $Y = e^X$, the formula gives the log-normal
$p_Y(y) = \tfrac{1}{y\sqrt{2\pi}}\exp\!\bigl(-\tfrac12(\log y)^2\bigr)$,
and a histogram of $e^X$ matches it exactly:

@!random-variables-change-of-variables

The naive guess $p_X(\log y)$ peaks at $y=1$; the Jacobian correction
moves the true peak to $y=e^{-1}$.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- Density $p\ge 0$, $\int p = 1$; probability is area; $F'=p$ (FTC).
- $\mathbb E$ is linear (no independence); $\operatorname{Var}=\mathbb E[X^2]-\mu^2$.
- Chebyshev: $P(|X-\mu|\ge\alpha\sigma)\le1/\alpha^2$ for any distribution, and it is sharp.
- Cauchy: a finite mean needs $\int|x|\,p\,dx<\infty$; symmetry alone proves nothing.
:::

::: {.col}
- Marginalize by integrating out; condition by slicing + renormalizing; Bayes for densities.
- Tower / Eve: total = explained + unexplained variance.
- Covariance is linear co-variation; correlation is its cosine in $[-1,1]$; $\Sigma\succeq 0$ → ellipse axes = PCA.
- Change of variables: the Jacobian keeps mass; log-dets sum → normalizing flows.
:::
:::

::: {.d2l-note}
Next: **named distributions**, which instantiate the general definitions above.
:::
:::
