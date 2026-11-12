# Random Variables
:label:`sec_mdl-random_variables`

The probability of :numref:`sec_prob` was built by counting. A die has six faces, a label is
one of ten classes, and the probability of an event is the sum of the
probabilities of the outcomes in it. Most of the numbers a network works with
are not like that. A pixel intensity, a weight drawn at initialization, the
Gaussian noise that a diffusion model adds to an image: each can take any
value in an interval, and there are too many values to list one by one. What
is the probability of a single one of them?

Think of a dart thrown at a round board so carelessly that it is equally
likely to land anywhere on the board, and suppose it comes to rest at half the
radius from the center. Asked how likely that was, we would round the
distance, say to between 0.45 and 0.55 radii, which happens about one time in
ten. Asking for one more digit makes the answer ten times smaller, and another
digit makes it ten times smaller again. With all the digits the probability is
zero, and yet the dart landed somewhere.

The way out is to stop asking for the probability of a point and to ask how
fast probability accumulates near it. Each time we narrow the interval tenfold,
its probability falls tenfold, so the ratio of probability to width stays
fixed. This probability per unit length is the *density*. With it, every tool
of discrete probability carries over to quantities that vary continuously,
with sums replaced by integrals.

Part of what follows is :numref:`sec_prob` again, now with proofs and with
densities: the mean and the variance were motivated there, and we do not
motivate them a second time. New are the density itself, a sampler built from the cumulative distribution
function, the inequalities of Markov and Chebyshev, and a density whose tails
are so heavy that both fail. The two results to remember are that the density
is the derivative of the cumulative distribution function
:eqref:`eq_mdl-cdf_deriv` and Markov's inequality :eqref:`eq_mdl-markov`,
from which the other tail bounds of this chapter descend. We follow the dart
throughout, one number at a time. Its two coordinates, and what happens to a
density when a variable is transformed, wait for
:numref:`sec_mdl-joint-distributions`.

```{.python .input #random-variables-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #random-variables-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #random-variables-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #random-variables-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Densities

We build the density in three steps: at a point, as the limit of ever
narrower bins; over an interval, as an area; and through the function that
accumulates the area, which will also tell us how to draw samples.

### From Bins to a Density

To make "probability per unit length" precise, we first need the probability
that the dart lands within a given range of distances. Let $R$ be the dart's
distance from the center of a board of radius $1$. The dart lands within
distance $r$ of the center when it lands in the disk of radius $r$. Since every
part of the board is equally likely, the probability is that disk's share of
the board's area:

$$
P(R\le r)=\frac{\pi r^2}{\pi}=r^2,\qquad 0\le r\le 1.
$$

Half the radius holds only a quarter of the probability, because it holds only
a quarter of the area. A bin of width $\epsilon$ centered at $0.5$ has
probability $(0.5+\epsilon/2)^2-(0.5-\epsilon/2)^2=\epsilon$, which gives the
numbers of the opening:

| bin | width $\epsilon$ | probability | probability $/\ \epsilon$ |
|:--|--:|--:|--:|
| $[0.45, 0.55]$ | $0.1$ | $0.1$ | $1$ |
| $[0.495, 0.505]$ | $0.01$ | $0.01$ | $1$ |
| $[0.4995, 0.5005]$ | $0.001$ | $0.001$ | $1$ |

Each tenfold narrowing divides the probability by ten, and the ratio does not
move. The value at which the ratio settles is the quantity we name. The
*probability density* of a random variable $X$ at $x$ is the probability per
unit length near $x$,

$$
p(x)=\lim_{\epsilon\downarrow0}\frac{P(x\le X<x+\epsilon)}{\epsilon},
$$

wherever this limit exists. Equivalently, the probability of a short interval
is its length times the density,

$$P(x\leq X<x+\epsilon)\approx\epsilon\, p(x),$$
:eqlabel:`eq_mdl-pdf_def`

with an error that shrinks faster than $\epsilon$. This describes the density
one point at a time, which is all we need for the moment. The definition
proper speaks of intervals and follows in the next subsection. For the dart
at a distance $r$ in $(0,1)$, the interval $[r,r+\epsilon)$ has probability
$(r+\epsilon)^2-r^2=2r\epsilon+\epsilon^2$. The ratio to the width is
$2r+\epsilon$, and letting $\epsilon\to0$,

$$p(r)=2r,\qquad 0\le r\le 1.$$

The density of the distance grows linearly from the center to the rim, as the
circumference of the circle at radius $r$ does. At $r=0.5$ the density is $1$,
the ratio of the table.

A density is not a probability. The dart's density at $r=0.9$ is $1.8$, and a
variable that is equally likely to land anywhere in $[0,0.1]$ has density $10$
there, since it puts probability $1$ on an interval of length $0.1$. A density
can take any non-negative value, but it can exceed $10$ only on a set of total
length less than $1/10$, or the probability of that set would exceed one. A
density is a rate, like a speed. A speed tells how fast distance accumulates,
not how far the car has gone. In the same way, the dart's density of $1.8$ at
$r=0.9$ tells how fast probability accumulates as $r$ grows past $0.9$, not
how much probability sits at $0.9$.

A point has no length, so it has no probability. For every $\epsilon>0$ the
point $x$ lies in $[x,x+\epsilon)$, so $P(X=x)$ is at most about
$\epsilon\,p(x)$, which goes to zero with $\epsilon$. So, for a variable with
a density,

$$P(X=x)=0\qquad\textrm{for every } x.$$

This resolves the dart's puzzle: the dart lands at distance $0.5$ with
probability zero, as it does at every other given distance, and yet it lands
somewhere. Probability zero does not mean impossible.

A point that carries positive probability is called an *atom*. A variable with a density has no
atoms, a discrete variable such as a die roll has only atoms, and some
variables have both. The output of a ReLU unit whose input is standard normal
is $0$ with probability $\tfrac12$, an atom, and spreads the other half over
$(0,\infty)$ with a density (Exercise :exref:`ex_mdl-random-variables-rectified-gaussian`
works it out).

The table came from the area formula. A simulation needs no formula. A
computer throws darts uniformly at the board by throwing them uniformly at the
square $[-1,1]^2$ around it and keeping those that land on the board, which
leaves every part of the board equally likely. We count the fraction of darts
in each of the three bins of the table and divide by the bin's width.

```{.python .input #random-variables-density-to-probability}
rng = np.random.default_rng()
xy = rng.uniform(-1, 1, size=(4 * 10**6, 2))           # darts on the square
on_board = (xy**2).sum(axis=1) <= 1                    # keep those on the board
r_dart = np.linalg.norm(xy[on_board], axis=1)          # distances R
for width in [0.1, 0.01, 0.001]:
    prob = np.mean(np.abs(r_dart - 0.5) <= width / 2)  # fraction in the bin
    print(f'width {width:<5}  probability {prob:.5f}  ratio {prob / width:.3f}')
```

The fraction falls tenfold with each tenfold narrowing, and the ratio stays
near $1=p(0.5)$. What the table could not show is how noisy a narrow bin is.
About $3.1$ million of the four million darts land on the board, and only
about $3{,}000$ of those fall in a bin of width $0.001$, so the last ratio
wobbles by a few percent from run to run.

### Probability as Area

The density was read off from the probabilities of short intervals. We now go
back the other way: given a density, what is the probability of a long
interval? Cut the real line into pieces $[\epsilon i,\epsilon(i+1))$ of width
$\epsilon$. By :eqref:`eq_mdl-pdf_def`, each piece has probability about
$\epsilon\,p(\epsilon i)$, and the probabilities of disjoint pieces add. A sum
of widths times heights is a Riemann sum, and as the width shrinks it tends to
the area under the curve, the integral (:numref:`sec_mdl-integral_calculus`).
Since $X$ takes some value, all pieces together have probability one:

$$
1=P(X\in\mathbb{R})\approx\sum_i \epsilon\, p(\epsilon i)
\;\xrightarrow[\epsilon\to0]{}\; \int_{-\infty}^{\infty} p(x)\,dx .
$$

Summing only the pieces between $a$ and $b$ suggests that the probability of
an interval is the area under the density above it. The argument is a
motivation and not a proof: there are about $1/\epsilon$ pieces, and their
errors, each small compared with $\epsilon$, add up to something small only if
they are small uniformly in $x$. We therefore turn the conclusion into the
definition.

**Definition.** A random variable $X$ *has the density* $p$ when $p\ge0$ and,
for all $a<b$,

$$P(X\in(a, b]) = \int_{a}^{b} p(x)\,dx.$$
:eqlabel:`eq_mdl-pdf_int_int`

Letting $a\to-\infty$ and $b\to\infty$ shows that the total area under a
density is one. These two properties, $p\ge0$ and total area $1$, are all
that a function needs in order to be a density.
:numref:`fig_mdl-prob-pdf-area` shows the definition for a density with two
bumps. The dart passes the test: for $0\le a<b\le1$ the integral
$\int_a^b 2r\,dr=b^2-a^2$ is the probability $P(a<R\le b)$ that the areas
gave, so $2r$ is its density in this sense too. The probability of landing in
the outer half of the radius is $\int_{0.5}^1 2r\,dr=1-0.25=0.75$. Three
quarters of the board lies more than half the radius from the center.

![A probability density $p(x)$. The total area under the curve is $1$; the probability that $X$ lands in the interval from $a$ to $b$ is the area of the shaded slab above it, $\int_a^b p(x)\,dx$. The density itself is not a probability (it may exceed $1$); only its integral over an interval is.](../img/mdl-prob-pdf-area.svg)
:label:`fig_mdl-prob-pdf-area`

### The Cumulative Distribution Function

Often we want a function whose values are probabilities, and one that also
describes variables without a density, such as a die roll. The *cumulative
distribution function* (c.d.f.) of $X$ is the probability that $X$ lands at or below $x$; when $X$ has a
density, it is the area under the density to the left of $x$:

$$
F(x) = P(X \le x) = \int_{-\infty}^{x} p(t)\,dt.
$$
:eqlabel:`eq_mdl-cdf_def`

So $F$ accumulates probability from left to right. It rises from $0$ as
$x\to-\infty$ to $1$ as $x\to\infty$. It never decreases, because the
non-negative $p$ only adds area, and when $X$ has a density it is continuous.
For the dart we have already computed it: $F(r)=0$ for $r<0$, $F(r)=r^2$ for
$0\le r\le1$, and $F(r)=1$ for $r>1$. It is the function we started from, and
the density $2r$ came from it.

The c.d.f. is defined for every random variable. A fair coin that reports $0$
or $1$ has

$$
F(x) = \begin{cases}
0 & x < 0, \\
\tfrac12 & 0 \le x < 1, \\
1 & x \ge 1,
\end{cases}
$$

a *staircase* that jumps by the probability of each atom. Since $F(x)=P(X\le
x)$ includes the atom at $x$ itself, $F$ takes the upper value at each jump.
In general every c.d.f. is *right-continuous*, $\lim_{y\downarrow x}F(y)=F(x)$.
A variable with both kinds of probability has both features: the ReLU output
above has a c.d.f. that is $0$ for $x<0$, jumps to $\tfrac12$ at $0$, and then
rises smoothly to $1$. One function describes all three kinds of variable.

When a density exists, the c.d.f. and the density carry the same information.
Going from $p$ to $F$ is integration; going back is differentiation. This is
where the limit of the bins returns, now as a theorem.

**Proposition (density is the derivative of the c.d.f.).** *If $X$ has a
density $p$ that is continuous at $x$, then $F$ is differentiable at $x$ and*

$$
F'(x) = p(x).
$$
:eqlabel:`eq_mdl-cdf_deriv`

*In particular, the probability per unit length
$P(x\le X<x+\epsilon)/\epsilon$ tends to $p(x)$ as $\epsilon\downarrow0$.*

**Proof.** By :eqref:`eq_mdl-cdf_def`, $F$ is the area-so-far function of $p$.
The fundamental theorem of calculus says that the derivative of an area-so-far
function is the integrand, at every point where the integrand is continuous
(:numref:`sec_mdl-integral_calculus`). For the second statement, the point $x$
has probability at most $\int_{x-\delta}^{x}p(t)\,dt$ for every $\delta>0$,
and these integrals tend to zero with $\delta$. Including or excluding an
endpoint therefore changes nothing,
$P(x\le X<x+\epsilon)=F(x+\epsilon)-F(x)$, and the ratio is a difference
quotient of $F$. $\blacksquare$

For the dart, $F'(r)=2r$ on $(0,1)$ is the density we found from the bins.
:numref:`fig_mdl-prob-pdf-cdf` puts the two functions side by side: the shaded
area $\int_a^b p$ on the left equals the rise $F(b)-F(a)$ on the right.

![Left: a density $p$ with the area over the interval from $a$ to $b$ shaded. Right: the matching c.d.f. $F(x)=\int_{-\infty}^x p$, an increasing curve from $0$ to $1$; the rise $F(b)-F(a)$ equals the shaded area on the left. The density is the slope of the c.d.f. $\left(F'=p\right)$ and the c.d.f. is the accumulated area under the density.](../img/mdl-prob-pdf-cdf.svg)
:label:`fig_mdl-prob-pdf-cdf`

### Quantiles and Inverse-Transform Sampling

A computer has a source of random numbers that are equally likely to fall
anywhere between $0$ and $1$. To simulate the dart, or any other density, we
need to turn these numbers into samples with the right distribution.

The source is the simplest density of all. A variable $U$ is *uniform on
$[0,1]$* when its density is $1$ on $[0,1]$ and $0$ elsewhere. Every
subinterval of $[0,1]$ then has probability equal to its length, and the c.d.f.
is $F_U(u)=u$ for $0\le u\le1$.

The c.d.f. maps a value to a probability. Going back, from a probability level
$u$ to the value below which that fraction of the probability lies, gives the
*quantile function*. When $F$ is continuous, and strictly increasing on the
interval $I$ where $0<F<1$, it maps $I$ one-to-one onto $(0,1)$, and its
inverse $F^{-1}:(0,1)\to I$ is the quantile function. Its values at
$u=0.25,0.5,0.75$ are the *quartiles*, and the middle one is the *median*. For
the dart, $F(r)=r^2$ gives $F^{-1}(u)=\sqrt u$, with quartiles $0.5$, $0.707$
and $0.866$. The median dart lands at $0.707$, farther out than half the
radius, because three quarters of the board lies beyond $0.5$.

The quantile function is also a sampler.

**Proposition (inverse-transform sampling).** *Let $F$ be a continuous c.d.f.
that is strictly increasing on the interval $I$ where $0<F<1$, and let
$U$ be uniform on $[0,1]$. Then $X=F^{-1}(U)$ has c.d.f. $F$.*

**Proof.** With probability one, $U$ lies in $(0,1)$, so $X$ is defined and
lies in $I$. For $x$ in $I$, the events $F^{-1}(U)\le x$ and $U\le F(x)$
coincide, because $F$ is increasing on $I$. The uniform gives the second event
probability $F(x)$, the length of $[0,F(x)]$:

$$
P\bigl(F^{-1}(U)\le x\bigr) = P\bigl(U\le F(x)\bigr) = F(x).
$$

For $x$ to the left of $I$, $F(x)=0$ and $X>x$ always; for $x$ to the right of
$I$, $F(x)=1$ and $X\le x$ always. $\blacksquare$

The hypothesis allows a c.d.f. that is flat outside an interval, as the
dart's is ($F=0$ for $r<0$ and $F=1$ for $r>1$). The *generalized inverse*
$F^{-}(u)=\min\{x:F(x)\ge u\}$ needs no hypothesis at all and samples
staircases too (Exercise :exref:`ex_mdl-random-variables-staircase-sampling`).

For the dart, $R=\sqrt U$. The tempting shortcut $R=U$ is wrong, and we can
say by how much before running anything: it puts half of the darts within
$r<0.5$, where only a quarter belong.
:numref:`fig_mdl-prob-inverse-transform` shows the construction for the
two-bump density. Where the density is high, the c.d.f. is steep, so an even
spread of levels lands in a narrow range of values.

![Inverse-transform sampling. Uniform levels $U$ on the vertical axis (left) reflect through the c.d.f. $F$ down to values $x=F^{-1}(U)$ on the horizontal axis. Where $F$ is steep, that is, where the density $p=F'$ is large, a uniform spread of levels is compressed into a dense cluster of samples, so the histogram of reflected points reproduces $p$.](../img/mdl-prob-inverse-transform.svg)
:label:`fig_mdl-prob-inverse-transform`

We now have two ways to produce the dart's distance, the darts thrown at the
board above and the samples $\sqrt U$. The cell compares them, with the
shortcut alongside.

```{.python .input #mdl-random-variables-quantiles-and-inverse-transform-sampling}
u = rng.uniform(size=10**6)
r = np.sqrt(u)                                         # R = F^{-1}(U)
print(f'P(R < 0.5): darts {np.mean(r_dart < 0.5):.3f}, sqrt(U) '
      f'{np.mean(r < 0.5):.3f}, naive U {np.mean(u < 0.5):.3f}')
print(f'mean distance: darts {r_dart.mean():.3f}, sqrt(U) {r.mean():.3f}')
```

The two agree. Both put a quarter of their distances below $0.5$, and both
have a mean distance close to $0.667$, a value the next subsection derives.
The shortcut puts half of its distances there: it spreads distances evenly,
whereas the board has more area far from the center.

## Mean and Variance

A density describes the whole distribution, but two numbers often say enough:
where the distribution is centered, and how far it spreads.
:numref:`sec_prob` introduced both for discrete variables and said why we
want them. Here we extend them to densities and prove their algebra.

### The Mean

The mean is the average of the values of $X$, each weighted by its
probability. For a variable with a density, slice the line into bins of width $\epsilon$ as before: the bin at $\epsilon
i$ contributes its value $\epsilon i$ times its probability, about $\epsilon\,
p(\epsilon i)$, and the sum tends to an integral. The *mean*, or *expectation*,
of $X$ is

$$
\mu_X=\mathbb{E}[X]=\sum_i x_i\,p_i\quad\textrm{(discrete)},\qquad
\mu_X=\mathbb{E}[X]=\int_{-\infty}^{\infty} x\,p(x)\,dx\quad\textrm{(density)}.
$$
:eqlabel:`eq_mdl-exp_def`

As in the discrete case, the mean is a balance point: a rod with mass density
$p$ along its length balances at $\mu_X$. The definition requires $\sum_i|x_i|\,p_i<\infty$ in the discrete case and
$\int|x|\,p(x)\,dx<\infty$ for a density. Otherwise the mean does not exist,
and below we meet a density for which it fails. The same weighting averages
any function of $X$,

$$
\mathbb{E}[g(X)]=\sum_i g(x_i)\,p_i,\qquad
\mathbb{E}[g(X)]=\int_{-\infty}^{\infty} g(x)\,p(x)\,dx,
$$

again provided the sum or integral of $|g|$ is finite. We weight the values of
$g$ by the probabilities of $X$, without first working out the distribution of
the new variable $g(X)$. The rule is called the *law of the unconscious
statistician*, because it is easy to use without noticing that it is a
theorem rather than the definition of $\mathbb{E}[g(X)]$. In general
$\mathbb{E}[g(X)]$ differs from $g(\mathbb{E}[X])$; for convex $g$, Jensen's
inequality (:numref:`subsec_mdl-jensen`) says which of the two is larger.

For the dart,

$$
\mathbb{E}[R]=\int_0^1 r\cdot 2r\,dr=\tfrac23 .
$$

The mean distance is $0.667$, not $0.5$: the board has more area far from the
center than near it. The mean lies a little below the median $0.707$, because
the few darts near the center pull the average down more than they move the
median.

The rule we use most often is that the mean is linear. :numref:`sec_prob` used
it; here we prove it.

**Proposition (linearity of expectation).** *For random variables $X,Y$ whose
means exist and constants $c,d$,*

$$
\mathbb{E}[cX+d]=c\,\mathbb{E}[X]+d,
\qquad
\mathbb{E}[X+Y]=\mathbb{E}[X]+\mathbb{E}[Y].
$$
:eqlabel:`eq_mdl-exp_linear`

**Proof.** Both follow from the linearity of the sum (or integral) that defines
the mean. For the first,
$\mathbb{E}[cX+d]=\sum_i(cx_i+d)p_i = c\sum_i x_i p_i + d\sum_i p_i =
c\,\mathbb{E}[X]+d$, using $\sum_i p_i=1$. For the second, let $p_{ij}$ be
the probability that $(X,Y)=(x_i,y_j)$, the joint distribution of the pair as
in :numref:`sec_prob`. Split the sum and recognize the marginals
$\sum_j p_{ij}=P(X=x_i)$ and $\sum_i p_{ij}=P(Y=y_j)$:

$$
\mathbb{E}[X+Y]=\sum_{i,j}(x_i+y_j)\,p_{ij}
=\sum_{i,j} x_i\,p_{ij} + \sum_{i,j} y_j\,p_{ij} = \mathbb{E}[X]+\mathbb{E}[Y].
$$

For densities the same computation runs with integrals over a joint density,
the continuous counterpart of $p_{ij}$, which
:numref:`sec_mdl-joint-distributions` defines. $\blacksquare$

The second identity needs no independence: the mean of a sum is the sum of the
means however strongly $X$ and $Y$ depend on each other. The average loss over
a dataset is a sum of this kind.

### Variance and Standard Deviation

The mean says nothing about spread. As in :numref:`sec_prob`, we measure
spread by the average squared distance from the mean, the *variance*

$$\sigma_X^2 = \textrm{Var}(X) = \mathbb{E}\bigl[(X-\mu_X)^2\bigr],$$
:eqlabel:`eq_mdl-var_def`

which is defined when $\mathbb{E}[X^2]<\infty$. Expanding the square and
applying linearity :eqref:`eq_mdl-exp_linear`, with $\mu_X$ treated as the
constant it is, gives the form that is usually easier to compute, the mean
square minus the square of the mean:

$$
\textrm{Var}(X)
 = \mathbb{E}\bigl[X^2 - 2\mu_X X + \mu_X^2\bigr]
 = \mathbb{E}[X^2] - 2\mu_X\,\mathbb{E}[X] + \mu_X^2
 = \mathbb{E}[X^2] - \mu_X^2.
$$
:eqlabel:`eq_mdl-var_comp`

For a variable with a density, the mean and the variance are therefore the two
integrals

$$
\mu_X = \int_{-\infty}^\infty x\,p(x)\,dx,
\qquad
\sigma^2_X = \int_{-\infty}^\infty x^2 p(x)\,dx - \mu_X^2 .
$$
:eqlabel:`eq_mdl-cont_mean_var`

For the dart, $\mathbb{E}[R^2]=\int_0^1 r^2\cdot2r\,dr=\tfrac12$, so
$\textrm{Var}(R)=\tfrac12-\tfrac49=\tfrac1{18}\approx0.056$. For the uniform
$U$ on $[0,1]$, $\mu_U=\int_0^1 u\,du=\tfrac12$ and
$\textrm{Var}(U)=\tfrac13-\tfrac14=\tfrac1{12}$.

Rescaling units and centering data are affine maps. Under them the shift drops
out of the variance and the scale comes out squared. By linearity the mean
maps as $\mu_{cX+d}=c\mu_X+d$, so the deviation is
$(cX+d)-\mu_{cX+d}=c(X-\mu_X)$, and squaring and taking the mean gives

$$
\textrm{Var}(cX+d) = c^2\,\textrm{Var}(X).
$$
:eqlabel:`eq_mdl-var_affine`

Two further facts complete the list. A variance is never negative, and it is
zero if and only if $X$ is constant with probability one: the mean of the
non-negative variable $(X-\mu_X)^2$ is zero only if that variable is zero with
probability one. And variances of independent variables add
(:numref:`sec_prob`); :numref:`subsec_mdl-covariance` proves this together
with the term that replaces independence.

The variance carries the square of the units of $X$, dollars squared for a
profit in dollars. Its square root, the *standard deviation*

$$
\sigma_X = \sqrt{\textrm{Var}(X)},
$$
:eqlabel:`eq_mdl-std_def`

has the units of $X$ again, and $\sigma_{cX+d}=|c|\,\sigma_X$. For the dart,
$\sigma_R=\sqrt{1/18}\approx0.236$ in units of the board's radius.

## Tails

The standard deviation suggests a scale for typical deviations from the mean.
How much probability can lie far out, if all we know is the mean, or the
mean and the variance? Two inequalities answer this for every distribution at
once. They are crude, and they are also the best possible, as one example on
three points will show. A bell-shaped density with neither a mean nor a
variance then shows what the two assumptions were protecting us from.

### Markov's and Chebyshev's Inequalities

The first inequality uses only the mean, and applies to variables that are
never negative. Its proof is one line, and nearly every tail bound in this chapter
is obtained by applying it to a well-chosen function of the variable.

**Proposition (Markov's inequality).** *If $X\ge 0$ has a mean and $t>0$,
then*

$$
P(X \ge t) \le \frac{\mathbb{E}[X]}{t}.
$$
:eqlabel:`eq_mdl-markov`

**Proof.** Let $\mathbf{1}_{X\ge t}$ be the indicator variable that is $1$ when
$X\ge t$ and $0$ otherwise. Then $X \ge t\,\mathbf{1}_{X\ge t}$ for every
outcome: when $X\ge t$ the right-hand side is $t\le X$, and otherwise it is
$0\le X$. Taking means of both sides and using
$\mathbb{E}[\mathbf{1}_{X\ge t}]=P(X\ge t)$ gives
$\mathbb{E}[X]\ge t\,P(X\ge t)$. $\blacksquare$

Geometrically, on $[0,\infty)$ the step function that is $0$ below $t$ and $t$
from $t$ on lies under the line $y=x$, so its mean, $t\,P(X\ge t)$, lies under
the mean of $X$. With $t$ ten times the mean, at most a tenth of any
non-negative population exceeds ten times its average, whether it is incomes,
file sizes or gradient norms.

Markov's inequality uses only the mean, so it is crude. Applied to the squared
deviation $(X-\mu_X)^2$, whose mean is the variance, it bounds both tails at
once. The result is due to Bienaymé and to Chebyshev :cite:`Chebyshev.1867`.

**Proposition (Chebyshev's inequality).** *For any $X$ with finite, nonzero
variance and any $\alpha>0$,*

$$P\bigl(|X - \mu_X| \ge \alpha\sigma_X\bigr) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_mdl-chebyshev`

**Proof.** The variable $Z=(X-\mu_X)^2$ is non-negative with mean
$\textrm{Var}(X)=\sigma_X^2$, and the event $|X-\mu_X|\ge\alpha\sigma_X$ is the
event $Z\ge\alpha^2\sigma_X^2$. Since $\sigma_X>0$, the threshold
$t=\alpha^2\sigma_X^2$ is positive, and Markov's inequality
:eqref:`eq_mdl-markov` gives
$P(|X-\mu_X|\ge\alpha\sigma_X) \le \sigma_X^2/(\alpha^2\sigma_X^2) = 1/\alpha^2$.
$\blacksquare$

So at most $1\%$ of any distribution lies ten or more standard deviations
from its mean. The same argument with a threshold $\epsilon>0$ in place of
$\alpha\sigma_X$ gives a form that needs only a finite variance and also
covers $\sigma_X=0$:

$$
P\bigl(|X-\mu_X| \ge \epsilon\bigr) \le \frac{\textrm{Var}(X)}{\epsilon^2}.
$$

For the dart, $\mu_R=0.667$ and $\sigma_R=0.236$. Two standard deviations below
the mean is $0.195$, and two above is beyond the board, so
$P(|R-\mu_R|\ge2\sigma_R)=P(R\le0.195)=0.195^2\approx0.038$, far below the bound
$\tfrac14$.

Can a bound that holds for every distribution be improved? Not without
further assumptions, and a distribution on three points shows it. Let $X$ take
the values $a-2$ and $a+2$ with probability $q$ each, and $a$ with probability
$1-2q$, where $0<q\le\tfrac12$. We call it the *three-atom example*. By
symmetry its mean is $a$, and
$\textrm{Var}(X)=\mathbb{E}[(X-a)^2]=4q+4q=8q$, so $\sigma_X=2\sqrt{2q}$. Take
$\alpha=2$. The threshold $2\sigma_X=4\sqrt{2q}$ shrinks as $q$ does, while
the outer atoms stay at distance $2$.
:numref:`fig_mdl-prob-chebyshev` shows the three regimes. For $q>\tfrac18$ the
threshold exceeds $2$, no atom lies that far out, and the probability of the
event is $0$. At $q=\tfrac18$ the threshold equals $2$, so $\sigma_X=1$ and
$a\pm2$ lie two standard deviations from the mean. Both outer atoms count, the
probability is $2q=\tfrac14$, and the bound holds with equality. For
$q<\tfrac18$ both atoms lie beyond the threshold, and their total mass $2q$ is
below $\tfrac14$.

![Chebyshev's bound on the three-atom example: masses $q$, $1-2q$, $q$ drawn as vertical bars at $a-2$, $a$, $a+2$, with the threshold distance $\alpha\sigma_X=4\sqrt{2q}$ for $\alpha=2$ marked as a horizontal bracket from $a-\alpha\sigma_X$ to $a+\alpha\sigma_X$. The three panels show the regimes $q>1/8$, $q=1/8$ (where the bound is attained with equality), and $q<1/8$. Orange atoms lie at or beyond the threshold and count toward the event $|X-a|\ge\alpha\sigma_X$.](../img/mdl-prob-chebyshev.svg)
:label:`fig_mdl-prob-chebyshev`

The same example settles every $\alpha\ge1$. Choose $q=1/(2\alpha^2)$, which is
at most $\tfrac12$. Then $\sigma_X=2\sqrt{2q}=2/\alpha$, the threshold
$\alpha\sigma_X=2$ lands on the outer atoms, and
$P(|X-a|\ge\alpha\sigma_X)=2q=1/\alpha^2$. Equality holds, so no constant
smaller than $1/\alpha^2$ works for every distribution. For $\alpha<1$ the
bound exceeds $1$ and says nothing.

Chebyshev's bound is the worst case over all distributions with a given
variance. For the distributions we usually meet it is loose. A Gaussian with
mean $\mu$ and standard deviation $\sigma$, the bell curve with density
$e^{-(x-\mu)^2/(2\sigma^2)}/(\sigma\sqrt{2\pi})$ that
:numref:`subsec_mdl-gaussian-clt` treats in full, puts much less probability
beyond two and three standard deviations than the bound allows:

| distance from the mean | Chebyshev's bound | Gaussian |
|:--|--:|--:|
| $2\sigma$ or more | $25\%$ | $4.6\%$ |
| $3\sigma$ or more | $11\%$ | $0.27\%$ |

The worst case puts all of its tail probability at a single distance, as the
three-atom example does. Typical distributions spread their tails out and thin
them quickly.

### Heavy Tails

Chebyshev's inequality needs a finite variance, and the mean needs
$\int|x|\,p(x)\,dx<\infty$. Both can fail for a density that looks harmless.
The *Cauchy distribution* has the bell-shaped density

$$
p(x)=\frac{1}{\pi(1+x^2)} .
$$

It is a density, since $\int_{-\infty}^\infty\frac{dx}{1+x^2}=[\arctan
x]_{-\infty}^{\infty}=\pi$. Unlike a Gaussian, whose tails fall off like
$e^{-x^2/2}$, its tails fall off only like $1/(\pi x^2)$. The second moment
fails first. Over the range $[-L,L]$,

$$
\int_{-L}^{L} x^2 p(x)\,dx=\frac{2}{\pi}\bigl(L-\arctan L\bigr),
$$

which is $5.43$, $62.7$ and $636$ at $L=10$, $100$ and $1000$ and grows
without bound, because the integrand $x^2/(\pi(1+x^2))$ tends to $1/\pi$. So
$\mathbb{E}[X^2]=\infty$, and the Cauchy distribution has no finite variance.

The mean fails in a subtler way. The integrand $x\,p(x)$ is odd, which tempts
us to declare the mean $0$ by symmetry, and indeed $\int_{-L}^{L}x\,p(x)\,dx=0$
for every $L$. But the definition requires absolute convergence, and

$$
\int_{-L}^{L}|x|\,p(x)\,dx=\frac1\pi\log(1+L^2)
$$

grows without bound. So the mean does not exist. Without absolute convergence
the answer depends on how the limits are taken: over the lopsided range
$[-L,2L]$,

$$
\int_{-L}^{2L}x\,p(x)\,dx=\frac{1}{2\pi}\log\frac{1+4L^2}{1+L^2}
\;\xrightarrow[L\to\infty]{}\;\frac{\log 2}{\pi}\approx0.221 .
$$

With no mean to converge to, what does an average of Cauchy draws do? One
might still hope that an average of $1000$ draws lies closer to $0$ than a
single draw, which lies farther than $1$ from $0$ with probability
$P(|X|>1)=1-\tfrac2\pi\arctan1=\tfrac12$. The cell plots the running mean of
$10^5$ Cauchy draws next to that of $10^5$ standard Gaussian draws, and then
counts how many of $1000$ such averages lie farther than $1$ from $0$.

```{.python .input #random-variables-cauchy-diverges}
n = 10**5
idx = np.unique(np.logspace(0, 5, 500).astype(int)) - 1  # log-spaced n
running_mean = lambda x: (np.cumsum(x) / np.arange(1, n + 1))[idx]
d2l.plot(idx + 1, [running_mean(rng.standard_normal(n)),
                   running_mean(rng.standard_cauchy(n))],
         'n', 'running mean', legend=['Gaussian', 'Cauchy'],
         xscale='log', ylim=[-4, 4], figsize=(5, 3))
averages = rng.standard_cauchy((1000, 1000)).mean(axis=1)
print(f'averages of 1000 Cauchy draws with |average| > 1: '
      f'{np.mean(np.abs(averages) > 1):.2f}')
```

The Gaussian running mean settles near $0$. The Cauchy running mean does not
settle: it moves in jumps, each caused by a single huge draw, and a jump of
order one can come at any $n$. Averaging did not help at all. About half of
the averages of $1000$ draws lie farther than $1$ from $0$, the same fraction
as for single draws. In fact the average of $n$ independent Cauchy draws is
again a standard Cauchy variable, a fact we state without proof.

A distribution is called *heavy-tailed* when its tails decay more slowly than
every exponential $e^{-\lambda t}$ with $\lambda>0$. Power-law tails, with
$P(|X|>t)$ decaying like $t^{-\alpha}$, are the standard example: they make
every moment $\mathbb{E}[|X|^k]$ with $k\ge\alpha$ infinite, and the Cauchy
has $\alpha=1$. A heavy tail does not require infinite moments, though. The
log-normal distribution of :numref:`subsec_mdl-change-of-variables` is
heavy-tailed, yet all of its moments are finite. Heavy tails matter in deep
learning because averages over them converge slowly or not at all. The noise
in minibatch gradients, for example, has been argued to be heavy-tailed
:cite:`Simsekli.Sagun.Gurbuzbalaban.2019`, a claim that
Exercise :exref:`ex_mdl-random-variables-heavy-tail-averages` tests.

All of this concerned one number at a time, and the dart has two. Its
coordinates turn out to be uncorrelated and yet far from independent, and to
say what that means we need densities over the plane, which are the subject
of :numref:`sec_mdl-joint-distributions`.

## Summary

A variable that varies continuously gives every single value
probability zero, so its distribution is described by a density, the
probability per unit length. The probability of an interval is the area under
the density, and the c.d.f. accumulates that area. Markov's and Chebyshev's
inequalities bound the tails of any distribution from its mean or variance;
Chebyshev's bound is attained in the worst case but loose for typical
distributions. The Cauchy distribution has neither a mean nor a variance, and
averaging its draws does nothing. Markov's inequality is the result to carry
forward: the tail bounds of
:numref:`sec_mdl-concentration-generalization` are all obtained from it.

## Exercises

1. **Power-law tail.** Let $X$ have the density $p(x) = 1/x^2$ for
   $x \ge 1$ and $p(x) = 0$ otherwise.
    1. Verify that $p$ is a density, and compute $P(X > 2)$ and the c.d.f.
       $F(x)$ for every real $x$.
    1. Find the quantile function $F^{-1}(u)$ for $u \in (0, 1)$, the
       median, and the other two quartiles. Write down the
       inverse-transform sampler $X = F^{-1}(U)$ of
       :numref:`fig_mdl-prob-inverse-transform` for this distribution.
    1. Show that $\mathbb{E}[X] = \infty$. Clipping $X$ at a level $c > 1$, as
       gradient clipping does for a gradient norm, gives $\min(X, c)$.
       Compute $\mathbb{E}[\min(X, c)]$, and describe how it grows with $c$.
    1. For the Cauchy density $p_C(x) = 1/\bigl(\pi(1+x^2)\bigr)$, derive
       the values of $\int_{-L}^{L} x\,p_C(x)\,dx$ and
       $\int_{-L}^{2L} x\,p_C(x)\,dx$ quoted in the text, and their limits
       as $L \to \infty$. What do the two limits say about the mean of the
       Cauchy distribution? Why can no such ambiguity arise for
       the mean of the non-negative variable $X$ of part (a)?
   :exlabel:`ex_mdl-random-variables-power-law-tail`

1. **Rectified Gaussian.** Let $X \sim \mathcal{N}(0, 1)$ have the c.d.f.
   $\Phi$, and let $Y = \max(0, X)$. This is the output of a ReLU unit
   with a standard normal pre-activation, a common model of a unit at
   initialization.
    1. Express the c.d.f. $F_Y$ through $\Phi$ and sketch it. Where does
       $Y$ have an atom, and what is its mass? Is $Y$ discrete, does it
       have a density, or is it mixed?
    1. Explain why $Y$ has no density. Which hypothesis of the
       change-of-variables proposition :eqref:`eq_mdl-cov_density_1d` fails
       for the ReLU?
    1. Clipping to $[-c, c]$ for a threshold $c > 0$, as the hard-tanh
       activation does for $c = 1$, gives $C = \min(\max(X, -c), c)$. Write
       down the c.d.f. of $C$, and find the threshold $c$ at which the atoms
       of $C$ together hold $10\%$ of the probability. The equation has no
       closed-form solution; use tabulated values of $\Phi$, or `math.erf`
       with $\Phi(z) = \tfrac12\bigl(1 + \operatorname{erf}(z/\sqrt{2})\bigr)$.
    1. Show that $\mathbb{E}[Y^2] = \tfrac12$ and
       $\mathbb{E}[Y] = 1/\sqrt{2\pi}$. Hint: write
       $X^2 = Y^2 + \min(0, X)^2$ and use the symmetry of $X$. The
       He initialization of :numref:`subsec_he_init` multiplies the weight
       variance by $2$; which of these two moments does the factor
       compensate for?
   :exlabel:`ex_mdl-random-variables-rectified-gaussian`

1. [code] **Logistic distribution and calibration.** The logistic
   distribution has the c.d.f.
   $F(x) = \operatorname{sigmoid}(x) = 1/(1 + e^{-x})$.
    1. Find its density with :eqref:`eq_mdl-cdf_deriv` and its quantile
       function. What are its quartiles?
    1. Let $X$ have a continuous, strictly increasing c.d.f. $F$. Show that
       $F(X)$ is uniform on $[0, 1]$. This is the converse of
       inverse-transform sampling.
    1. A probabilistic regression model predicts for each input $i$ a
       logistic distribution for the target $y_i$, centered at the correct
       location $\mu_i$ but with a scale $s$, that is, the c.d.f.
       $\operatorname{sigmoid}\bigl((y - \mu_i)/s\bigr)$. The actual errors
       $y_i - \mu_i$ follow the standard logistic distribution. Draw $10^4$
       errors with the quantile function from (a), compute
       $u_i = \operatorname{sigmoid}\bigl((y_i - \mu_i)/s\bigr)$ for
       $s \in \{0.5, 1, 2\}$, and plot the three histograms of the $u_i$.
       Which shape reveals predicted distributions that are too narrow, and
       which reveals distributions that are too wide? Such histograms are a
       standard check of the calibration of probabilistic forecasts
       :cite:`Gneiting.Balabdaoui.Raftery.2007`.
   :exlabel:`ex_mdl-random-variables-logistic-calibration`

1. [code] **Sampling from a staircase.** ● The inverse-transform sampler of
   :numref:`fig_mdl-prob-inverse-transform` assumes a c.d.f. that is
   continuous and strictly increasing where it lies strictly between $0$
   and $1$. A categorical distribution, such as the softmax over the
   vocabulary from which a language model samples its next token, has a
   staircase c.d.f. instead. For any c.d.f. $F$, define the generalized
   inverse $F^{-}(u) = \min\{x : F(x) \ge u\}$ for $u \in (0, 1)$. The set
   is nonempty and bounded below, because $0 < u < 1$ and $F$ tends to $0$
   and $1$ at $-\infty$ and $\infty$. Its infimum belongs to it, because
   every c.d.f. is right-continuous, so the minimum exists.
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

1. [extended] **Averages of heavy tails.** ●●
   :citet:`Simsekli.Sagun.Gurbuzbalaban.2019` argued that the gradient
   noise of SGD in deep networks is heavy-tailed and better described by an
   $\alpha$-stable distribution, whose variance is infinite for
   $\alpha < 2$, than by a Gaussian. This project measures how a Monte Carlo
   average behaves in that regime and then tests the premise on a small
   model. The Pareto density $p(x) = \alpha x^{-\alpha-1}$ on $x \ge 1$, with
   $\alpha > 0$, contains the density of problem
   :exref:`ex_mdl-random-variables-power-law-tail` as the case $\alpha = 1$.
    1. Generalize problem :exref:`ex_mdl-random-variables-power-law-tail`b
       and c to this family: show that $\mathbb{E}[X^k] < \infty$ exactly
       when $k < \alpha$, compute the mean $\mu$ for $\alpha > 1$, and write
       down an inverse-transform sampler.
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
       $k \in \{10^2, 10^3, 10^4\}$. Then train the softmax regression
       of :numref:`sec_softmax_concise`. At the trained parameters, collect
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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.1]{.kicker}

A dart, a density, and the tools that follow<br>**random variables**
:::
:::

::: {.slide title="A Dart Lands 0.5 From the Center: What Was the Probability?"}
[The problem]{.kicker}

A dart lands anywhere on a round board of radius 1, all places equally likely.
It stops at distance $0.5$ from the center.

| distance rounded to | probability | probability / width |
|:--|--:|--:|
| $[0.45, 0.55]$ | $0.1$ | $1$ |
| $[0.495, 0.505]$ | $0.01$ | $1$ |
| $[0.4995, 0.5005]$ | $0.001$ | $1$ |
| exactly $0.5$ | $0$ | |

::: {.d2l-note .warn}
Every exact value has probability $0$, yet the dart lands somewhere.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute $P(a < X \le b)$ from a density, and sample it from uniform noise;
- compute a mean and a variance, and bound a tail from them alone;
- recognize a heavy tail, for which averaging fails.

**Running example:** the dart, with distance $R$: $P(R\le r)=r^2$.
**Idea:** probability per unit length; sums become integrals.
:::

::: {.slide title="First in the Chapter: the Language of Densities"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **Random Variables** | **densities, moments, tail bounds** |
| Joint Distributions | conditioning, covariance, change of variables |
| Distributions | the named distributions and their losses |
| The Exponential Family | one form, convex fits, conjugacy |
| Statistics | error bars, tests, intervals |
| Maximum Likelihood | losses, Fisher information, MAP |
| Latent Variables and EM | the ELBO, EM |
| Bayesian Computation | averaging over a posterior |
| Concentration Inequalities | exponential tails of averages |
| Uniform Convergence | guarantees after selection |
| Naive Bayes | one classifier, end to end |
:::

::: {.col .narrow}
- **Uses:** discrete probability, integrals.
- **Skips:** named distributions, estimation.
:::
:::
:::

::: {.slide title="Probability per Unit Length Is the Density"}
[Densities]{.kicker}

$$p(x)=\lim_{\epsilon\downarrow 0}\frac{P(x\le X<x+\epsilon)}{\epsilon}$$

- Dart: $P(r\le R<r+\epsilon)=2r\epsilon+\epsilon^2$, so $p(r)=2r$ on $[0,1]$.
- At $r=0.5$: $p=1$, the ratio in the table.
- $P(X=x)\le P(x\le X<x+\epsilon)\approx\epsilon\,p(x)\to0$: no single value has probability.

::: {.d2l-note}
A density is a rate, not a probability: $p(0.9)=1.8$, and the uniform on
$[0,0.1]$ has density $10$.
:::
:::

::: {.slide title="Probability Is Area, and Simulated Darts Agree"}
[Densities]{.kicker}

$$p\ge0,\qquad \int_{-\infty}^{\infty}p(x)\,dx=1,\qquad P(a<X\le b)=\int_a^b p(x)\,dx.$$

Throw $4\cdot10^6$ darts at the square, keep those on the board, count a bin:

@!random-variables-density-to-probability

::: {.d2l-note}
Dart: $\int_{0.5}^1 2r\,dr=0.75$ of the board lies beyond half the radius.
:::
:::

::: {.slide title="The c.d.f. Accumulates Probability; Its Slope Is the Density"}
[Densities]{.kicker}

::: {.cols .vc}
::: {.col}
$F(x)=P(X\le x)=\int_{-\infty}^x p$, and $F'=p$ where $p$ is continuous
(fundamental theorem of calculus).

Dart: $F(r)=r^2$, $F'(r)=2r$. A coin has a staircase $F$.
:::

::: {.col .fig}
@fig:mdl-prob-pdf-cdf
:::
:::
:::

::: {.slide title="The Quantile Function Turns Uniform Noise Into Samples"}
[Sampling]{.kicker}

::: {.cols .vc}
::: {.col}
If $F$ is continuous and strictly increasing where $0<F<1$, and $U$ is
uniform on $[0,1]$, then $F^{-1}(U)$ has c.d.f. $F$:
$P(F^{-1}(U)\le x)=P(U\le F(x))=F(x)$.

Dart: $R=\sqrt U$. Naive $R=U$ puts $50\%$ inside $r<0.5$, not $25\%$.
:::

::: {.col .fig}
@fig:mdl-prob-inverse-transform
:::
:::
:::

::: {.slide title="Check: Can a Density Equal 5? Where Is the Dart's Median?"}
[Check]{.kicker}

Can a density take the value $5$? At which distance does the dart's median
lie?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Yes: the uniform density on $[0,0.2]$ equals $5$. The median
solves $r^2=\tfrac12$, so it is $\sqrt{1/2}\approx0.707$, not $0.5$.
:::
:::
:::

::: {.slide title="The Mean Is Linear, With No Independence Needed"}
[Mean]{.kicker}

$$\mathbb{E}[X]=\sum_i x_i p_i\quad\textrm{or}\quad\int x\,p(x)\,dx,
\qquad\textrm{if }\textstyle\sum|x_i|p_i\textrm{ or }\int|x|\,p<\infty.$$

- Dart: $\mathbb{E}[R]=\int_0^1 r\cdot2r\,dr=\tfrac23$: more area lies far out.
- $\mathbb{E}[cX+d]=c\,\mathbb{E}[X]+d$.
- $\mathbb{E}[X+Y]=\mathbb{E}[X]+\mathbb{E}[Y]$ for any dependence.
:::

::: {.slide title="Variance Is the Mean Square Minus the Squared Mean"}
[Variance]{.kicker}

$\textrm{Var}(X)=\mathbb{E}[(X-\mu)^2]=\mathbb{E}[X^2]-\mu^2$ and
$\textrm{Var}(cX+d)=c^2\,\textrm{Var}(X)$.

- Dart: $\mathbb{E}[R^2]=\tfrac12$, $\textrm{Var}(R)=\tfrac12-\tfrac49=\tfrac1{18}$,
  $\sigma_R\approx0.236$.
- Three atoms: $a-2$, $a$, $a+2$ with probabilities $q$, $1-2q$, $q$
  ($0\le q\le\tfrac12$): mean $a$, $\mathbb{E}[X^2]=a^2+8q$,
  $\textrm{Var}=8q$.
:::

::: {.slide title="The Mean Alone Bounds the Tail of a Non-Negative Variable"}
[Markov]{.kicker}

For $X\ge0$ and $t>0$:

$$P(X\ge t)\le\frac{\mathbb{E}[X]}{t}.$$

*Proof.* $X\ge t\,\mathbf 1_{X\ge t}$ for every outcome; take means. $\blacksquare$

::: {.d2l-note}
At most a tenth of any non-negative population exceeds ten times its mean.
:::
:::

::: {.slide title="Chebyshev: at Most $1/\alpha^2$ Lies $\alpha\sigma$ Away, and This Is Sharp"}
[Chebyshev]{.kicker}

::: {.cols .vc}
::: {.col}
For finite, nonzero variance: $P(|X-\mu|\ge\alpha\sigma)\le1/\alpha^2$.
*Proof:* Markov on $(X-\mu)^2$ with $t=\alpha^2\sigma^2$.

Three atoms with $q=1/(2\alpha^2)$: $\sigma=2/\alpha$, and the outer atoms
give equality.
:::

::: {.col .fig}
@fig:mdl-prob-chebyshev
:::
:::
:::

::: {.slide title="Chebyshev Allows 25% Beyond 2σ; a Gaussian Puts 4.6% There"}
[Chebyshev]{.kicker}

| distance from the mean | Chebyshev | Gaussian | dart |
|:--|--:|--:|--:|
| $2\sigma$ or more | $25\%$ | $4.6\%$ | $3.8\%$ |
| $3\sigma$ or more | $11\%$ | $0.27\%$ | $0\%$ |

::: {.d2l-note}
Tight for the worst case, which puts all its tail at one distance; loose for
the distributions we usually meet.
:::
:::

::: {.slide title="The Cauchy Density Has No Mean, and Averaging Does Not Help"}
[Heavy tails]{.kicker}

$p(x)=1/(\pi(1+x^2))$: $\int_{-L}^{L}|x|\,p=\tfrac1\pi\log(1+L^2)\to\infty$,
and $\int_{-L}^{2L}x\,p\to\log 2/\pi$.

@!random-variables-cauchy-diverges
:::

::: {.slide title="Check: Var(3X + 7)? The Chance of 3σ or More?"}
[Check]{.kicker}

If $\textrm{Var}(X)=2$, what is $\textrm{Var}(3X+7)$? For any $X$ with
$\sigma>0$, how large can $P(|X-\mu|\ge3\sigma)$ be?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $9\cdot2=18$: the shift drops out, the scale squares. At most
$1/9\approx11\%$, by Chebyshev.
:::
:::
:::

::: {.slide title="One Dart Carried Every Tool of the Section"}
[Wrap-up]{.kicker}

- $P(R\le r)=r^2$, density $2r$, median $0.707$, sampler $R=\sqrt U$.
- $\mathbb{E}[R]=\tfrac23$, $\textrm{Var}(R)=\tfrac1{18}$; Chebyshev's $1/\alpha^2$ is sharp but loose.
- Markov and Chebyshev bound a tail from the mean or the variance alone.
- The Cauchy has neither a mean nor a variance: averaging its draws does nothing.

::: {.d2l-note}
Next: two coordinates of the dart at once, and the density of a transformed variable.
:::
:::
