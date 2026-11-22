# Random Variables
:label:`sec_mdl-random_variables`

In :numref:`sec_prob`, we built probability by counting. A die has six faces, a
label belongs to one of ten classes, and the probability of an event is simply
the sum of the probabilities of its outcomes. Most of the numbers a neural
network works with are not like that. A pixel intensity, a weight drawn at
initialization, or the Gaussian noise that a diffusion model adds to an image
can each take any value in an interval, and there are far too many values to
list one by one. What, then, is the probability of a single one of them?

Think of a dart thrown at a round board so carelessly that it is equally
likely to land anywhere on the board, and suppose it comes to rest at half the
radius from the center. Asked how likely that was, we would round the
distance—say, to between $0.45$ and $0.55$ radii—which happens about one time
in ten. Asking for one more digit of precision makes the probability ten times
smaller, and another digit makes it ten times smaller again. With infinitely
many digits, the probability is zero, and yet the dart clearly landed
somewhere.

The way out is to stop asking for the probability of a single point and ask
instead how fast probability accumulates near it. Each time we narrow the
interval tenfold, its probability drops tenfold, so the ratio of probability
to width stays constant. This probability per unit length is the *density*.
With it, every tool of discrete probability carries over to quantities that
vary continuously, with sums replaced by integrals.

Some of what follows revisits :numref:`sec_prob`, now extended to continuous
variables and backed by proofs: we already motivated the mean and variance
there, so we move straight to their continuous forms and algebra. Alongside
them, we introduce the density itself, a general method for drawing samples
from the cumulative distribution function, the tail bounds of Markov and
Chebyshev, and a bell-shaped density whose tails are so heavy that both bounds
fail. If you carry only two results out of this section, make them the
relationship between the density and the cumulative distribution function
:eqref:`eq_mdl-cdf_deriv` and Markov's inequality :eqref:`eq_mdl-markov`, from
which the rest of this chapter's tail bounds descend. We follow the dart
throughout, focusing on a single number, its distance from the center, and
leave its two coordinates and transformations of densities for
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

We develop the idea of a density in three steps: first at a single point, as
the limit of ever narrower bins; next over an interval, as an area under a
curve; and finally through the function that accumulates that area, which also
gives us a practical way to draw random samples.

### From Bins to a Density

To make "probability per unit length" precise, we first need the probability
that the dart lands within a given range of distances. Let $R$ be the dart's
distance from the center of a board of radius $1$. The dart lands within
distance $r$ of the center whenever it hits the disk of radius $r$. Because
every part of the board is equally likely, this probability is simply that
disk's share of the board's area:

$$
P(R\le r)=\frac{\pi r^2}{\pi}=r^2,\qquad 0\le r\le 1.
$$

Half the radius holds only a quarter of the probability, because it holds only
a quarter of the area. A bin of width $\epsilon$ centered at $0.5$ therefore
has probability $(0.5+\epsilon/2)^2-(0.5-\epsilon/2)^2=\epsilon$, which
produces the numbers from our opening example:

| bin | width $\epsilon$ | probability | probability $/\ \epsilon$ |
|:--|--:|--:|--:|
| $[0.45, 0.55]$ | $0.1$ | $0.1$ | $1$ |
| $[0.495, 0.505]$ | $0.01$ | $0.01$ | $1$ |
| $[0.4995, 0.5005]$ | $0.001$ | $0.001$ | $1$ |

Each tenfold narrowing divides the probability by ten, leaving the ratio
unchanged. That limiting ratio is the quantity we want to name. The
*probability density* of a random variable $X$ at $x$ is the probability per
unit length near $x$,

$$
p(x)=\lim_{\epsilon\downarrow0}\frac{P(x\le X<x+\epsilon)}{\epsilon},
$$

wherever this limit exists. Equivalently, for a short interval, the
probability is approximately its length times the density,

$$P(x\leq X<x+\epsilon)\approx\epsilon\, p(x),$$
:eqlabel:`eq_mdl-pdf_def`

with an error that shrinks faster than $\epsilon$. This describes the density
one point at a time, which is all we need for now; we give the formal
definition in terms of intervals in the next subsection. For our dart at any
distance $r$ in $(0,1)$, the interval $[r,r+\epsilon)$ has probability
$(r+\epsilon)^2-r^2=2r\epsilon+\epsilon^2$. Dividing by the width $\epsilon$
gives $2r+\epsilon$, and letting $\epsilon\to 0$ yields

$$p(r)=2r,\qquad 0\le r\le 1.$$

The density grows linearly from the center to the rim, mirroring the
circumference of the circle at radius $r$: there is simply more board farther
out. At $r=0.5$ the density equals $1$, matching the ratio in the table above.

Notice that a density is not a probability. At $r=0.9$, the dart's density is
$1.8$, and a variable that is equally likely to land anywhere in $[0,0.1]$ has
density $10$ on that interval, since it packs a total probability of $1$ into
a length of $0.1$. A density can take any non-negative value, though it can
exceed $10$ only on a set of total length less than $1/10$, or else the
probability of that set would exceed $1$. Think of a density as a rate, much
like speed. A car's speed tells you how fast distance accumulates, not how far
the car has traveled. In the same way, a density of $1.8$ at $r=0.9$ tells you
how fast probability accumulates as $r$ increases past $0.9$, not how much
probability sits at the point $0.9$ itself.

Because a single point has zero length, it also carries zero probability. For
every $\epsilon>0$, the point $x$ lies inside $[x,x+\epsilon)$, so $P(X=x)$ is
at most about $\epsilon\,p(x)$, which vanishes as $\epsilon$ shrinks to zero.
Thus, for any random variable with a density,

$$P(X=x)=0\qquad\textrm{for every } x.$$

This resolves our opening puzzle: the dart lands at distance $0.5$ with
probability zero, just as it does at every other exact distance, and yet it
must land somewhere. In continuous probability, an event of probability zero
is not impossible.

A single point that does carry positive probability is called an *atom*. A
variable with a density has no atoms, whereas a discrete variable such as a
die roll consists entirely of atoms. Some variables have both. For example,
the output of a ReLU unit fed a standard normal input equals $0$ with
probability $\tfrac12$ (an atom) and spreads the remaining half of its
probability over $(0,\infty)$ with a density (Exercise
:exref:`ex_mdl-random-variables-rectified-gaussian` works out the details).

The table above came from an exact area formula, but a simulation needs no
formula at all. To throw darts uniformly at the circular board, a computer can
throw them uniformly at the surrounding square $[-1,1]^2$ and keep only those
that land inside the circle, which leaves every part of the board equally
likely. Below, we count the fraction of darts falling in each of our three
bins and divide by the bin width.

```{.python .input #random-variables-density-to-probability}
rng = np.random.default_rng()
xy = rng.uniform(-1, 1, size=(4 * 10**6, 2))           # darts on the square
on_board = (xy**2).sum(axis=1) <= 1                    # keep those on the board
r_dart = np.linalg.norm(xy[on_board], axis=1)          # distances R
for width in [0.1, 0.01, 0.001]:
    prob = np.mean(np.abs(r_dart - 0.5) <= width / 2)  # fraction in the bin
    print(f'width {width:<5}  probability {prob:.5f}  ratio {prob / width:.3f}')
```

As expected, each tenfold narrowing reduces the fraction tenfold while keeping
the ratio close to $1=p(0.5)$. What the exact table hid, however, is how noisy
a very narrow bin becomes in practice. Roughly $3.1$ million of the four
million darts land on the board, and only about $3{,}000$ of those fall into
the bin of width $0.001$, so the final ratio fluctuates by a few percent from
run to run.

### Probability as Area

So far we have read the density off from the probabilities of tiny intervals.
Now let us reverse the direction: given a density, how do we find the
probability of a full interval? Partition the real line into bins
$[\epsilon i,\epsilon(i+1))$ of width $\epsilon$. By :eqref:`eq_mdl-pdf_def`,
each bin has probability roughly $\epsilon\,p(\epsilon i)$, and because the
bins are disjoint, their probabilities add. Summing widths times heights forms
a Riemann sum, which approaches the area under the curve, namely the integral
(:numref:`sec_mdl-integral_calculus`), as $\epsilon$ shrinks to zero. Since
$X$ must take some real value, all bins together have probability one:

$$
1=P(X\in\mathbb{R})\approx\sum_i \epsilon\, p(\epsilon i)
\;\xrightarrow[\epsilon\to0]{}\; \int_{-\infty}^{\infty} p(x)\,dx .
$$

Summing only the bins between $a$ and $b$ suggests that the probability of an
interval is simply the area under the density above it. Strictly speaking,
this argument is motivation rather than a proof: there are roughly
$1/\epsilon$ bins, and their individual errors, each small compared with
$\epsilon$, sum to a negligible total only if they shrink uniformly across
$x$. To put things on solid footing, we turn this area formula into our formal
definition.

**Definition.** A random variable $X$ *has the density* $p$ when $p\ge0$ and,
for all $a<b$,

$$P(X\in(a, b]) = \int_{a}^{b} p(x)\,dx.$$
:eqlabel:`eq_mdl-pdf_int_int`

Letting $a\to-\infty$ and $b\to\infty$ confirms that the total area under any
density is $1$. In fact, these two conditions, non-negativity ($p\ge 0$) and
unit total area, are all a function needs to qualify as a valid density.
:numref:`fig_mdl-prob-pdf-area` illustrates this definition for a density with
two peaks. Our dart passes the test as well: for $0\le a<b\le 1$, the integral
$\int_a^b 2r\,dr=b^2-a^2$ recovers the area-based probability $P(a<R\le b)$,
confirming that $2r$ is indeed its density. In particular, the probability of
landing in the outer half of the radius is $\int_{0.5}^1 2r\,dr=1-0.25=0.75$:
fully three quarters of the board lies more than half the radius from the
center.

![A probability density $p(x)$. The total area under the curve is $1$; the probability that $X$ lands in the interval from $a$ to $b$ is the area of the shaded slab above it, $\int_a^b p(x)\,dx$. The density itself is not a probability (it may exceed $1$); only its integral over an interval is.](../img/mdl-prob-pdf-area.svg)
:label:`fig_mdl-prob-pdf-area`

### The Cumulative Distribution Function

Often it is convenient to work with a function whose values are genuine
probabilities, and one that handles discrete and continuous variables alike.
The *cumulative distribution function* (c.d.f.) of $X$ gives the probability
that $X$ lands at or below $x$. When $X$ has a density, this is simply the
area under the density to the left of $x$:

$$
F(x) = P(X \le x) = \int_{-\infty}^{x} p(t)\,dt.
$$
:eqlabel:`eq_mdl-cdf_def`

In other words, $F$ sweeps across the real line from left to right,
accumulating probability as it goes. It starts at $0$ as $x\to-\infty$ and
rises to $1$ as $x\to\infty$. It can never decrease, because the non-negative
density $p$ only adds area, and whenever $X$ has a density, $F$ is continuous.
For our dart, we computed it at the very beginning: $F(r)=0$ for $r<0$,
$F(r)=r^2$ for $0\le r\le 1$, and $F(r)=1$ for $r>1$. That function is where we
started, and the density $2r$ came directly from it.

Unlike a density, the c.d.f. is defined for every random variable. A fair coin
that reports $0$ or $1$ has

$$
F(x) = \begin{cases}
0 & x < 0, \\
\tfrac12 & 0 \le x < 1, \\
1 & x \ge 1,
\end{cases}
$$

which is a *staircase* that jumps upward by the probability of each atom.
Because $F(x)=P(X\le x)$ includes any atom sitting right at $x$, the function
takes the upper value at every jump. More generally, every c.d.f. is
*right-continuous*, meaning $\lim_{y\downarrow x}F(y)=F(x)$. A mixed variable
combines both behaviors: the ReLU output mentioned earlier has a c.d.f. that
is $0$ for $x<0$, jumps to $\tfrac12$ at $x=0$, and then climbs smoothly
toward $1$. A single function therefore unifies discrete, continuous, and
mixed random variables.

When a density exists, the c.d.f. and the density carry the same information:
integrating $p$ produces $F$, and differentiating $F$ recovers $p$. This
brings back our opening limit of shrinking bins, now established as a theorem.

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
function is the integrand at every point where the integrand is continuous
(:numref:`sec_mdl-integral_calculus`). For the second statement, the single
point $x$ has probability at most $\int_{x-\delta}^{x}p(t)\,dt$ for every
$\delta>0$, and these integrals tend to zero with $\delta$. Including or
excluding an endpoint therefore changes nothing, so
$P(x\le X<x+\epsilon)=F(x+\epsilon)-F(x)$, and the ratio is simply a
difference quotient of $F$. $\blacksquare$

For the dart, $F'(r)=2r$ on $(0,1)$, recovering the density we found from the
bins. :numref:`fig_mdl-prob-pdf-cdf` places the two functions side by side:
the shaded area $\int_a^b p$ on the left equals the vertical rise $F(b)-F(a)$
on the right.

![Left: a density $p$ with the area over the interval from $a$ to $b$ shaded. Right: the matching c.d.f. $F(x)=\int_{-\infty}^x p$, an increasing curve from $0$ to $1$; the rise $F(b)-F(a)$ equals the shaded area on the left. The density is the slope of the c.d.f. $\left(F'=p\right)$ and the c.d.f. is the accumulated area under the density.](../img/mdl-prob-pdf-cdf.svg)
:label:`fig_mdl-prob-pdf-cdf`

### Quantiles and Inverse-Transform Sampling

Computers give us ready access to random numbers that are equally likely to
fall anywhere between $0$ and $1$. To simulate the dart, or any other
continuous distribution, we need a way to turn those uniform draws into
samples from our target distribution.

Our building block is the simplest density of all. A random variable $U$ is
*uniform on $[0,1]$* if its density is $1$ on $[0,1]$ and $0$ elsewhere. Every
subinterval of $[0,1]$ then has probability equal to its length, and the
c.d.f. is simply $F_U(u)=u$ for $0\le u\le 1$.

The c.d.f. maps a value $x$ to a probability level $u$. Reversing that map,
asking for the value below which a given fraction $u$ of the probability lies,
yields the *quantile function*. Whenever $F$ is continuous and strictly
increasing on the interval $I$ where $0<F<1$, it maps $I$ one-to-one onto
$(0,1)$, and its inverse $F^{-1}:(0,1)\to I$ is the quantile function. Its
values at $u=0.25, 0.5, 0.75$ are the *quartiles*, with the middle quartile
$F^{-1}(0.5)$ called the *median*. For our dart, $F(r)=r^2$ inverts to
$F^{-1}(u)=\sqrt{u}$, giving quartiles of $0.5$, $0.707$, and $0.866$. Notice
that the median dart lands at $0.707$, well past half the radius, because
three quarters of the board lies beyond $0.5$.

Remarkably, the quantile function is also a universal sampler.

**Proposition (inverse-transform sampling).** *Let $F$ be a continuous c.d.f.
that is strictly increasing on the interval $I$ where $0<F<1$, and let
$U$ be uniform on $[0,1]$. Then $X=F^{-1}(U)$ has c.d.f. $F$.*

**Proof.** With probability one, $U$ lies in $(0,1)$, so $X$ is defined and
lies in $I$. For $x$ in $I$, the events $F^{-1}(U)\le x$ and $U\le F(x)$
coincide, because $F$ is increasing on $I$. Since $U$ is uniform on $[0,1]$,
the second event has probability equal to the length of $[0,F(x)]$:

$$
P\bigl(F^{-1}(U)\le x\bigr) = P\bigl(U\le F(x)\bigr) = F(x).
$$

For $x$ to the left of $I$, $F(x)=0$ and $X>x$ always; for $x$ to the right of
$I$, $F(x)=1$ and $X\le x$ always. $\blacksquare$

Notice that this hypothesis allows $F$ to be flat outside an interval, just as
the dart's c.d.f. is ($F=0$ for $r<0$ and $F=1$ for $r>1$). Replacing $F^{-1}$
with the *generalized inverse* $F^{-}(u)=\min\{x:F(x)\ge u\}$ removes the
hypothesis altogether and samples staircase c.d.f.s as well.

For the dart, inverse-transform sampling gives $R=\sqrt{U}$. It is tempting to
take the shortcut $R=U$, and we can see immediately why that fails without
running a single line of code: $R=U$ places half the darts inside $r<0.5$,
where only a quarter belong. :numref:`fig_mdl-prob-inverse-transform` shows
the geometry of the construction on our two-bump density. Wherever the density
is high, the c.d.f. rises steeply, so an evenly spaced set of levels on the
vertical axis is compressed into a tight cluster of samples on the horizontal
axis.

![Inverse-transform sampling. Uniform levels $U$ on the vertical axis (left) reflect through the c.d.f. $F$ down to values $x=F^{-1}(U)$ on the horizontal axis. Where $F$ is steep, that is, where the density $p=F'$ is large, a uniform spread of levels is compressed into a dense cluster of samples, so the histogram of reflected points reproduces $p$.](../img/mdl-prob-inverse-transform.svg)
:label:`fig_mdl-prob-inverse-transform`

We now have two completely different ways to generate the dart's distance:
throwing darts at the square and keeping those on the board, or simply
computing $\sqrt{U}$. The cell below compares the two, alongside the naive
shortcut $R=U$.

```{.python .input #mdl-random-variables-quantiles-and-inverse-transform-sampling}
u = rng.uniform(size=10**6)
r = np.sqrt(u)                                         # R = F^{-1}(U)
print(f'P(R < 0.5): darts {np.mean(r_dart < 0.5):.3f}, sqrt(U) '
      f'{np.mean(r < 0.5):.3f}, naive U {np.mean(u < 0.5):.3f}')
print(f'mean distance: darts {r_dart.mean():.3f}, sqrt(U) {r.mean():.3f}')
```

The two valid methods agree. Both place a quarter of their distances below
$0.5$, and both have a mean distance close to $0.667$, a number we will derive
in the next subsection. The naive shortcut $U$ puts half its samples below
$0.5$ because it spreads distances evenly across $[0,1]$, ignoring the fact
that the board has far more area near the rim than near the center.

## Mean and Variance

A density specifies an entire distribution, yet two summary numbers often tell
us most of what we need to know: where the distribution is centered, and how
widely it spreads. :numref:`sec_prob` introduced both for discrete variables
and motivated their use; here we extend them to densities and prove the
algebraic rules we rely on.

### The Mean

The mean averages the values of $X$, weighting each by its probability. For a
variable with a density, slicing the real line into bins of width $\epsilon$
as before shows that the bin at $\epsilon i$ contributes its value $\epsilon
i$ times its probability, roughly $\epsilon\,p(\epsilon i)$, and the resulting
sum tends to an integral. The *mean*, or *expectation*, of $X$ is therefore

$$
\mu_X=\mathbb{E}[X]=\sum_i x_i\,p_i\quad\textrm{(discrete)},\qquad
\mu_X=\mathbb{E}[X]=\int_{-\infty}^{\infty} x\,p(x)\,dx\quad\textrm{(density)}.
$$
:eqlabel:`eq_mdl-exp_def`

Just as in the discrete case, the mean acts as a balance point: a thin rod
with mass density $p$ along its length balances at $\mu_X$. For this
definition to make sense, we require $\sum_i|x_i|\,p_i<\infty$ in the discrete
case and $\int|x|\,p(x)\,dx<\infty$ for a density. When that condition fails,
the mean does not exist, and we will soon meet a concrete density where this
happens. Using the same probability weights lets us average any function
$g(X)$ directly,

$$
\mathbb{E}[g(X)]=\sum_i g(x_i)\,p_i,\qquad
\mathbb{E}[g(X)]=\int_{-\infty}^{\infty} g(x)\,p(x)\,dx,
$$

again provided the sum or integral of $|g|$ is finite. Notice how convenient
this is: we weight the values of $g$ by the probabilities of $X$, without
first having to work out the distribution of the transformed variable $g(X)$.
This rule is famously called the *law of the unconscious statistician*,
because it is so natural to apply that one easily forgets it is a theorem
rather than the definition of $\mathbb{E}[g(X)]$. Keep in mind that
$\mathbb{E}[g(X)]$ generally differs from $g(\mathbb{E}[X])$; when $g$ is
convex, Jensen's inequality (:numref:`subsec_mdl-jensen`) tells us which of
the two is larger.

For our dart,

$$
\mathbb{E}[R]=\int_0^1 r\cdot 2r\,dr=\tfrac23 .
$$

The mean distance is $0.667$, not $0.5$, again because the board has more area
near the rim than near the center. Notice also that the mean sits slightly
below the median $0.707$: the relatively few darts that land near the center
pull the average downward more than they shift the middle value.

The property of expectation we use most often is linearity. We relied on it in
:numref:`sec_prob`; here we prove it.

**Proposition (linearity of expectation).** *For random variables $X,Y$ whose
means exist and constants $c,d$,*

$$
\mathbb{E}[cX+d]=c\,\mathbb{E}[X]+d,
\qquad
\mathbb{E}[X+Y]=\mathbb{E}[X]+\mathbb{E}[Y].
$$
:eqlabel:`eq_mdl-exp_linear`

**Proof.** Both identities follow from the linearity of the sum (or integral)
that defines the mean. For the first,
$\mathbb{E}[cX+d]=\sum_i(cx_i+d)p_i = c\sum_i x_i p_i + d\sum_i p_i =
c\,\mathbb{E}[X]+d$, using $\sum_i p_i=1$. For the second, let $p_{ij}$ be
the probability that $(X,Y)=(x_i,y_j)$, the joint distribution of the pair as
in :numref:`sec_prob`. Splitting the sum and recognizing the marginals
$\sum_j p_{ij}=P(X=x_i)$ and $\sum_i p_{ij}=P(Y=y_j)$ gives

$$
\mathbb{E}[X+Y]=\sum_{i,j}(x_i+y_j)\,p_{ij}
=\sum_{i,j} x_i\,p_{ij} + \sum_{i,j} y_j\,p_{ij} = \mathbb{E}[X]+\mathbb{E}[Y].
$$

For densities, the exact same computation runs with integrals over a joint
density, the continuous counterpart of $p_{ij}$, which
:numref:`sec_mdl-joint-distributions` defines. $\blacksquare$

Crucially, the second identity requires no independence at all: the mean of a
sum equals the sum of the means no matter how tightly $X$ and $Y$ depend on
each other. Every time we take the expected loss as an average over a dataset,
we are relying on this fact.

### Variance and Standard Deviation

The mean tells us where a distribution sits, but nothing about how tightly it
clusters there. As in :numref:`sec_prob`, we measure spread by the average
squared distance from the mean, called the *variance*,

$$\sigma_X^2 = \textrm{Var}(X) = \mathbb{E}\bigl[(X-\mu_X)^2\bigr],$$
:eqlabel:`eq_mdl-var_def`

which is defined whenever $\mathbb{E}[X^2]<\infty$. Expanding the square and
applying linearity :eqref:`eq_mdl-exp_linear`, treating $\mu_X$ as a constant,
yields a form that is usually much easier to compute: the mean of the square
minus the square of the mean,

$$
\textrm{Var}(X)
 = \mathbb{E}\bigl[X^2 - 2\mu_X X + \mu_X^2\bigr]
 = \mathbb{E}[X^2] - 2\mu_X\,\mathbb{E}[X] + \mu_X^2
 = \mathbb{E}[X^2] - \mu_X^2.
$$
:eqlabel:`eq_mdl-var_comp`

For a variable with a density, the mean and variance are therefore given by
the two integrals

$$
\mu_X = \int_{-\infty}^\infty x\,p(x)\,dx,
\qquad
\sigma^2_X = \int_{-\infty}^\infty x^2 p(x)\,dx - \mu_X^2 .
$$
:eqlabel:`eq_mdl-cont_mean_var`

For the dart, $\mathbb{E}[R^2]=\int_0^1 r^2\cdot 2r\,dr=\tfrac12$, so
$\textrm{Var}(R)=\tfrac12-\tfrac49=\tfrac1{18}\approx 0.056$. For the uniform
variable $U$ on $[0,1]$, $\mu_U=\int_0^1 u\,du=\tfrac12$ and
$\textrm{Var}(U)=\tfrac13-\tfrac14=\tfrac1{12}$.

Rescaling units or centering data are affine transformations. Under an affine
map $cX+d$, the shift $d$ drops out of the variance while the scale factor $c$
comes out squared. Indeed, linearity gives $\mu_{cX+d}=c\mu_X+d$, so the
deviation from the mean is $(cX+d)-\mu_{cX+d}=c(X-\mu_X)$; squaring and taking
the expectation gives

$$
\textrm{Var}(cX+d) = c^2\,\textrm{Var}(X).
$$
:eqlabel:`eq_mdl-var_affine`

Two further properties are worth keeping in mind. First, variance is never
negative, and it equals zero if and only if $X$ is constant with probability
one, since the expectation of the non-negative quantity $(X-\mu_X)^2$ can
vanish only when that quantity is zero with probability one. Second, variances
of independent random variables add (:numref:`sec_prob`); we will prove this
in :numref:`subsec_mdl-covariance`, along with the covariance term that
replaces independence when variables interact.

Because variance squares the deviations, it carries squared units, such as
dollars squared for a profit measured in dollars. Taking its square root
restores the original units and gives the *standard deviation*,

$$
\sigma_X = \sqrt{\textrm{Var}(X)},
$$
:eqlabel:`eq_mdl-std_def`

which scales linearly under affine maps as $\sigma_{cX+d}=|c|\,\sigma_X$. For
our dart, $\sigma_R=\sqrt{1/18}\approx 0.236$, measured in units of the
board's radius.

## Tails

The standard deviation sets a natural scale for typical fluctuations around
the mean. A natural next question is how much probability can sit far out in
the tails when all we know is the mean, or the mean and variance. Two classic
inequalities answer this for every distribution at once. Both bounds are quite
crude, yet surprisingly they are the sharpest possible without extra
assumptions, as a simple three-point distribution will demonstrate. We then
turn to a bell-shaped density that has neither a mean nor a variance, showing
what those two assumptions were protecting us from.

### Markov's and Chebyshev's Inequalities

The first bound uses only the mean and applies to any non-negative random
variable. Its proof takes a single line, yet nearly every tail bound in this
chapter comes from applying it to a well-chosen function of the variable.

**Proposition (Markov's inequality).** *If $X\ge 0$ has a mean and $t>0$,
then*

$$
P(X \ge t) \le \frac{\mathbb{E}[X]}{t}.
$$
:eqlabel:`eq_mdl-markov`

**Proof.** Let $\mathbf{1}_{X\ge t}$ be the indicator variable that equals $1$
when $X\ge t$ and $0$ otherwise. Then $X \ge t\,\mathbf{1}_{X\ge t}$ for every
outcome: when $X\ge t$ the right-hand side is $t\le X$, and otherwise it is
$0\le X$. Taking expectations of both sides and using
$\mathbb{E}[\mathbf{1}_{X\ge t}]=P(X\ge t)$ gives
$\mathbb{E}[X]\ge t\,P(X\ge t)$. $\blacksquare$

Geometrically, on $[0,\infty)$ the step function that is $0$ below $t$ and $t$
from $t$ onward stays underneath the line $y=x$, so its average value
$t\,P(X\ge t)$ must stay underneath the average value of $X$. Setting $t$ to
ten times the mean gives an immediate rule of thumb: at most a tenth of any
non-negative population can exceed ten times its average, whether we are
looking at incomes, file sizes, or gradient norms.

Because Markov's inequality uses only the mean, it is necessarily crude.
Applying it to the squared deviation $(X-\mu_X)^2$, whose mean is the
variance, bounds both tails at once. This result is due to Bienaymé and
Chebyshev :cite:`Chebyshev.1867`.

**Proposition (Chebyshev's inequality).** *For any $X$ with finite, nonzero
variance and any $\alpha>0$,*

$$P\bigl(|X - \mu_X| \ge \alpha\sigma_X\bigr) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_mdl-chebyshev`

**Proof.** The variable $Z=(X-\mu_X)^2$ is non-negative with mean
$\textrm{Var}(X)=\sigma_X^2$, and the event $|X-\mu_X|\ge\alpha\sigma_X$ is
identical to the event $Z\ge\alpha^2\sigma_X^2$. Since $\sigma_X>0$, the
threshold $t=\alpha^2\sigma_X^2$ is positive, so Markov's inequality
:eqref:`eq_mdl-markov` gives
$P(|X-\mu_X|\ge\alpha\sigma_X) \le \sigma_X^2/(\alpha^2\sigma_X^2) = 1/\alpha^2$.
$\blacksquare$

In particular, at most $1\%$ of any distribution can lie ten or more standard
deviations from its mean. Running the same proof with an absolute threshold
$\epsilon>0$ in place of $\alpha\sigma_X$ gives an equivalent form that
requires only finite variance and also covers the degenerate case
$\sigma_X=0$:

$$
P\bigl(|X-\mu_X| \ge \epsilon\bigr) \le \frac{\textrm{Var}(X)}{\epsilon^2}.
$$

For our dart, $\mu_R=0.667$ and $\sigma_R=0.236$. Two standard deviations
below the mean is $0.195$, while two above lies outside the board altogether,
so $P(|R-\mu_R|\ge 2\sigma_R)=P(R\le 0.195)=0.195^2\approx 0.038$, far smaller
than Chebyshev's bound of $\tfrac14$.

Given how loose $\tfrac14$ is for the dart, can a bound valid for every
distribution be sharpened? Not without further assumptions, and a distribution
supported on just three points proves it. Let $X$ take the values $a-2$ and
$a+2$ with probability $q$ each, and $a$ with the remaining probability
$1-2q$, where $0<q\le\tfrac12$. We call this the *three-atom example*. By
symmetry its mean is $a$, and its variance is
$\textrm{Var}(X)=\mathbb{E}[(X-a)^2]=4q+4q=8q$, so $\sigma_X=2\sqrt{2q}$. Take
$\alpha=2$. As $q$ shrinks, the threshold $2\sigma_X=4\sqrt{2q}$ shrinks with
it, while the two outer atoms stay fixed at distance $2$ from $a$.
:numref:`fig_mdl-prob-chebyshev` shows the three resulting regimes. When
$q>\tfrac18$, the threshold exceeds $2$, so neither outer atom falls in the
tail and the probability is $0$. At $q=\tfrac18$, the threshold lands right at
$2$, giving $\sigma_X=1$ and placing $a\pm 2$ two standard deviations from the
mean; both outer atoms now count, their total probability is $2q=\tfrac14$,
and Chebyshev's bound holds with equality. When $q<\tfrac18$, both outer atoms
still lie beyond the threshold, and their combined mass $2q$ falls below
$\tfrac14$.

![Chebyshev's bound on the three-atom example: masses $q$, $1-2q$, $q$ drawn as vertical bars at $a-2$, $a$, $a+2$, with the threshold distance $\alpha\sigma_X=4\sqrt{2q}$ for $\alpha=2$ marked as a horizontal bracket from $a-\alpha\sigma_X$ to $a+\alpha\sigma_X$. The three panels show the regimes $q>1/8$, $q=1/8$ (where the bound is attained with equality), and $q<1/8$. Orange atoms lie at or beyond the threshold and count toward the event $|X-a|\ge\alpha\sigma_X$.](../img/mdl-prob-chebyshev.svg)
:label:`fig_mdl-prob-chebyshev`

The same construction settles every $\alpha\ge 1$. Choosing
$q=1/(2\alpha^2)$, which is at most $\tfrac12$, gives
$\sigma_X=2\sqrt{2q}=2/\alpha$, so the threshold $\alpha\sigma_X=2$ lands
squarely on the outer atoms and $P(|X-a|\ge\alpha\sigma_X)=2q=1/\alpha^2$.
Equality holds, proving that no constant smaller than $1/\alpha^2$ can work
for all distributions. (For $\alpha<1$, the bound exceeds $1$ and says
nothing.)

Chebyshev's inequality therefore captures the worst case over all
distributions with a given variance. For the distributions we usually
encounter, it is quite loose. A Gaussian with mean $\mu$ and standard
deviation $\sigma$, the bell curve with density
$e^{-(x-\mu)^2/(2\sigma^2)}/(\sigma\sqrt{2\pi})$ studied in full in
:numref:`subsec_mdl-gaussian-clt`, puts far less probability beyond two and
three standard deviations than Chebyshev allows:

| distance from the mean | Chebyshev's bound | Gaussian |
|:--|--:|--:|
| $2\sigma$ or more | $25\%$ | $4.6\%$ |
| $3\sigma$ or more | $11\%$ | $0.27\%$ |

The worst-case distribution concentrates all of its tail mass at a single
distance, just as our three-atom example does. Typical distributions spread
their tails out and thin them rapidly.

### Heavy Tails

Chebyshev's inequality requires a finite variance, and the mean itself
requires $\int|x|\,p(x)\,dx<\infty$. Surprisingly, both conditions can fail
for a smooth, harmless-looking bell curve. The *Cauchy distribution* has the
density

$$
p(x)=\frac{1}{\pi(1+x^2)} .
$$

This is a valid density, since $\int_{-\infty}^\infty\frac{dx}{1+x^2}=[\arctan
x]_{-\infty}^{\infty}=\pi$. Unlike a Gaussian, whose tails plummet like
$e^{-x^2/2}$, the Cauchy tails decay only algebraically, like $1/(\pi x^2)$.
The second moment fails first. Integrating over $[-L,L]$ gives

$$
\int_{-L}^{L} x^2 p(x)\,dx=\frac{2}{\pi}\bigl(L-\arctan L\bigr),
$$

which evaluates to $5.43$, $62.7$, and $636$ at $L=10$, $100$, and $1000$,
growing without bound because the integrand $x^2/(\pi(1+x^2))$ approaches the
nonzero constant $1/\pi$. Hence $\mathbb{E}[X^2]=\infty$, and the Cauchy
distribution has no finite variance.

The mean fails in a subtler way. Because the integrand $x\,p(x)$ is odd, it is
tempting to declare the mean to be $0$ by symmetry, and indeed
$\int_{-L}^{L}x\,p(x)\,dx=0$ for every $L$. However, the definition of
expectation requires absolute convergence, and

$$
\int_{-L}^{L}|x|\,p(x)\,dx=\frac1\pi\log(1+L^2)
$$

grows without bound as $L\to\infty$. Consequently, the Cauchy mean does not
exist. Without absolute convergence, the value of the integral depends on how
the limits are taken: over the lopsided interval $[-L,2L]$, for example,

$$
\int_{-L}^{2L}x\,p(x)\,dx=\frac{1}{2\pi}\log\frac{1+4L^2}{1+L^2}
\;\xrightarrow[L\to\infty]{}\;\frac{\log 2}{\pi}\approx0.221 .
$$

With no mean to settle toward, what happens when we average independent Cauchy
draws? One might still hope that averaging $1000$ draws lands closer to $0$
than a single draw, which falls outside $[-1,1]$ with probability
$P(|X|>1)=1-\tfrac2\pi\arctan 1=\tfrac12$. The cell below plots the running
mean of $10^5$ Cauchy draws alongside that of $10^5$ standard Gaussian draws,
and then counts how many of $1000$ separate averages (each over $1000$ Cauchy
draws) still lie farther than $1$ from $0$.

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

The Gaussian running mean settles smoothly toward $0$. The Cauchy running mean
never settles: it lurches in sudden jumps, each triggered by a single extreme
draw, and an order-one jump can strike at any $n$. Averaging brings no
improvement at all. Roughly half of the averages of $1000$ draws still lie
farther than $1$ from $0$, matching the fraction for a single draw. In fact,
as we state without proof, the average of $n$ independent standard Cauchy
variables has the same distribution as a single standard Cauchy variable.

A distribution is called *heavy-tailed* when its tails decay more slowly than
every exponential $e^{-\lambda t}$ with $\lambda>0$. Power-law tails, where
$P(|X|>t)$ decays like $t^{-\alpha}$, are the canonical example: they force
every moment $\mathbb{E}[|X|^k]$ with $k\ge\alpha$ to be infinite, with the
Cauchy distribution corresponding to $\alpha=1$. Heavy tails do not
automatically imply infinite moments, however; the log-normal distribution (which
we will derive in :numref:`subsec_mdl-change-of-variables`) is heavy-tailed yet
has all moments finite. Heavy tails matter in deep learning because sample averages over them
converge slowly or fail to converge altogether. Minibatch gradient noise, for
example, has been argued to be heavy-tailed
:cite:`Simsekli.Sagun.Gurbuzbalaban.2019`, a hypothesis that Exercise
:exref:`ex_mdl-random-variables-heavy-tail-averages` lets you test.

Everything in this section dealt with one number at a time, whereas our dart
lands in a plane and has two coordinates. Those two coordinates turn out to be
uncorrelated yet far from independent, and making sense of that requires
densities over the plane, which we take up in
:numref:`sec_mdl-joint-distributions`.

## Summary

A continuous random variable assigns probability zero to every individual
value, so we describe its distribution through a density, the probability per
unit length. Integrating the density over an interval gives that interval's
probability, the c.d.f. accumulates this area from left to right with $F'=p$,
and inverting the c.d.f. turns uniform noise into samples from the
distribution. Markov's and Chebyshev's inequalities bound the tails of any
distribution using only its mean or variance; Chebyshev's bound is tight in
the worst case yet loose for typical distributions. Heavy-tailed distributions
like the Cauchy have neither a mean nor a variance, and averaging their draws
does nothing to reduce their spread. Above all, carry Markov's inequality
forward: every tail bound in :numref:`sec_mdl-concentration-generalization`
descends from it.

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

1. **Rectified Gaussian.** Let $X \sim \mathcal{N}(0, 1)$ have the standard
   Gaussian density $p_X(x) = \tfrac{1}{\sqrt{2\pi}}e^{-x^2/2}$ and c.d.f.
   $\Phi(x) = \int_{-\infty}^x p_X(t)\,dt$, and let $Y = \max(0, X)$. This is
   the output of a ReLU unit
   with a standard normal pre-activation, a common model of a unit at
   initialization.
    1. Express the c.d.f. $F_Y$ through $\Phi$ and sketch it. Where does
       $Y$ have an atom, and what is its mass? Is $Y$ discrete, does it
       have a density, or is it mixed?
    1. Explain why $Y$ has no density. Under the one-dimensional
       change-of-variables rule $p_Y(y) = p_X(g^{-1}(y))\,|(g^{-1})'(y)|$ (which
       we will prove in :numref:`subsec_mdl-change-of-variables`,
       :eqref:`eq_mdl-cov_density_1d`, for a strictly monotone, continuously
       differentiable transformation $Y = g(X)$), which hypothesis fails
       for the ReLU?
    1. Clipping to $[-c, c]$ for a threshold $c > 0$, as the hard-tanh
       activation does for $c = 1$, gives $C = \min(\max(X, -c), c)$. Write
       down the c.d.f. of $C$, and find the threshold $c$ at which the atoms
       of $C$ together hold $10\%$ of the probability. The equation has no
       closed-form solution; use tabulated values of $\Phi$, or `math.erf`
       with $\Phi(z) = \tfrac12\bigl(1 + \operatorname{erf}(z/\sqrt{2})\bigr)$.
    1. Show that $\mathbb{E}[Y^2] = \tfrac12$ and
       $\mathbb{E}[Y] = 1/\sqrt{2\pi}$. Hint: write
       $X^2 = Y^2 + \min(0, X)^2$ and use the symmetry of $X$ (equivalently,
       this decomposes $\mathbb{E}[Y]$ and $\textrm{Var}(Y)$ by conditioning on
       the sign of $X$, an instance of the law of total expectation
       $\mathbb{E}[Y] = \mathbb{E}[\mathbb{E}[Y \mid X]]$ and law of total
       variance
       $\textrm{Var}(Y) = \mathbb{E}[\textrm{Var}(Y \mid X)] + \textrm{Var}(\mathbb{E}[Y \mid X])$
       developed in full in :numref:`sec_mdl-joint-distributions`). The
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

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1095)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §36.1]{.kicker}

Random Variables: Densities, Cumulative Distributions, Moments, and Tail Bounds<br>**probability density $p(x)$ · CDF $F'(x) = p(x)$ · inverse-transform sampling · expectation & variance · Markov & Chebyshev · heavy tails**
:::
:::

::: {.slide title="A Dart Lands at Distance 0.5: What Was the Probability?"}
[The Continuous Paradox]{.kicker}

::: {.cols .vc}
::: {.col}
- **Continuous outcomes:** A dart landing uniformly on a unit disk has radial distance $R \in [0, 1]$ with disk area share $P(R \le r) = r^2$.
- **Shrinking bins around $r = 0.5$:** As bin width $\epsilon$ shrinks from $0.1 \to 0.01 \to 0.001$, the probability $P(R \in \text{bin})$ shrinks identically while the ratio $P/\epsilon = 1$ stays constant.
- **Zero probability $\ne$ impossibility:** Every single point has $P(R = r) = 0$ because a point has zero width, yet probability accumulates at a finite local rate.
:::

::: {.col .fig .big}
@fig:ch35-prob-dart-bins
:::
:::
:::

::: {.slide title="Probability Density Is Probability per Unit Length"}
[Local Rate of Accumulation]{.kicker}

::: {.cols .vc}
::: {.col}
$$p(x) = \lim_{\epsilon \downarrow 0} \frac{P(x \le X < x + \epsilon)}{\epsilon}$$

- **Deriving the dart's density:** Dividing $P(r \le R < r+\epsilon) = (r+\epsilon)^2 - r^2 = 2r\epsilon + \epsilon^2$ by $\epsilon$ yields $p(r) = 2r$ on $[0, 1]$.
- **Geometric meaning:** Density grows linearly with $r$ because ring circumference $2\pi r$ is larger near the outer rim.
- **Rate, not probability:** A density can exceed $1$ (e.g., $p(0.9) = 1.8$, or $p(x) = 10$ on $[0, 0.1]$).
:::

::: {.col .fig}
@fig:ch35-prob-dart-density
:::
:::
:::

::: {.slide title="Probability as Area Under the Density Curve"}
[Formal Integral Definition]{.kicker}

::: {.cols .vc}
::: {.col}
- Partitioning $(a, b]$ into bins of width $\epsilon$ and summing their probabilities $\sum_i \epsilon\,p(\epsilon i)$ forms a Riemann sum that converges to the definite integral:
  $$P(X \in (a, b]) = \int_a^b p(x)\,dx$$
- **Two conditions define any valid density:**
  1. **Non-negativity:** $p(x) \ge 0$ for all $x \in \mathbb{R}$.
  2. **Unit total area:** $\int_{-\infty}^{\infty} p(x)\,dx = 1$.
- **Dart outer ring:** $\int_{0.5}^1 2r\,dr = [r^2]_{0.5}^1 = 0.75$—fully three quarters of the board lies beyond half the radius.
:::

::: {.col .fig}
@fig:mdl-prob-pdf-area
:::
:::
:::

::: {.slide title="Simulating the Dart Board: From Bins to Ratios"}
[Monte Carlo Verification]{.kicker}

To throw darts uniformly on the unit disk without formulas, we sample uniformly on the bounding square $[-1, 1]^2$, keep points inside $x^2 + y^2 \le 1$, and divide bin frequencies at $r = 0.5$ by width $\epsilon$:

@random-variables-density-to-probability

::: {.d2l-note}
**Finite-sample bin noise:** Narrowing $\epsilon$ tenfold reduces the hit count tenfold (only $\sim 3{,}100$ of $3.14 \times 10^6$ darts fall in the $\epsilon = 0.001$ bin), causing the empirical ratio to fluctuate by a few percent around $p(0.5) = 1$.
:::
:::

::: {.slide title="Atoms, Densities, and Mixed Variables in Deep Learning"}
[Discrete vs. Continuous vs. Mixed]{.kicker}

::: {.cols .vc}
::: {.col}
- **Atoms vs. densities:** A point $x$ with $P(X = x) > 0$ is an **atom** (discrete class labels $Y \in \{1,\dots,K\}$), whereas purely continuous weights $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ have zero atoms.
- **Mixed distributions (Rectified Gaussian / ReLU):** Passing $X \sim \mathcal{N}(0, 1)$ through $Y = \max(0, X)$ collapses all negative inputs onto an **atom $P(Y = 0) = 1/2$**.
- **Positive half-line:** The remaining $50\%$ of the probability spreads continuously over $(0, \infty)$ with Gaussian density $\frac{1}{\sqrt{2\pi}}e^{-y^2/2}$.
:::

::: {.col .fig .big}
@fig:ch35-prob-mixed-relu
:::
:::
:::

::: {.slide title="The Cumulative Distribution Function (CDF)"}
[Unifying Discrete, Continuous, and Mixed Variables]{.kicker}

::: {.cols .vc}
::: {.col}
$$F(x) = P(X \le x) = \int_{-\infty}^x p(t)\,dt$$

- **Universal rules:** Every CDF is monotone non-decreasing, right-continuous ($\lim_{y \downarrow x} F(y) = F(x)$), and runs from $F(-\infty) = 0$ to $F(+\infty) = 1$.
- **Visual signature:** Smooth curves for continuous densities ($F(r) = r^2$), staircases for discrete atoms, and jumps plus smooth climbs for mixed ReLU outputs.
:::

::: {.col .fig}
@fig:ch35-prob-cdf-types
:::
:::
:::

::: {.slide title="The Density Is the Derivative of the CDF"}
[Fundamental Theorem of Calculus]{.kicker}

::: {.cols .vc}
::: {.col}
- **Proposition:** If $X$ has a density $p$ that is continuous at $x$, then $F$ is differentiable at $x$ and
  $$F'(x) = p(x)$$
- **Proof idea:** Since $F(x) = \int_{-\infty}^x p(t)\,dt$ is the area-so-far function, the Fundamental Theorem of Calculus gives $F'(x) = p(x)$. Because single points carry zero mass,
  $$\frac{P(x \le X < x + \epsilon)}{\epsilon} = \frac{F(x + \epsilon) - F(x)}{\epsilon} \xrightarrow{\epsilon \downarrow 0} F'(x)$$
- **Area–slope duality:** Shaded area $\int_a^b p(x)\,dx$ on the left equals vertical rise $F(b) - F(a)$ on the right.
:::

::: {.col .fig}
@fig:mdl-prob-pdf-cdf
:::
:::
:::

::: {.slide title="Quantiles, Quartiles, and Inverse-Transform Sampling"}
[Turning Uniform Noise into Target Samples]{.kicker}

::: {.cols .vc}
::: {.col}
- **Quantile function $F^{-1}(u)$:** While $F$ maps a value $x$ to a probability level $u \in (0, 1)$, its inverse $F^{-1}(u)$ returns the threshold below which fraction $u$ of the distribution lies.
  - Dart ($F(r) = r^2 \implies F^{-1}(u) = \sqrt{u}$): quartiles are $0.50$, **median $\sqrt{0.5} \approx 0.707$**, and $0.866$.
- **Proposition (Inverse-Transform Sampling):** If $U \sim \mathrm{Unif}(0, 1)$, then $X = F^{-1}(U)$ has CDF $F$:
  $$P\bigl(F^{-1}(U) \le x\bigr) = P\bigl(U \le F(x)\bigr) = F(x)$$
- Where $p = F'$ is large, $F$ rises steeply and compresses uniform vertical levels into dense horizontal samples.
:::

::: {.col .fig}
@fig:mdl-prob-inverse-transform
:::
:::
:::

::: {.slide title="Why the Naive Shortcut R = U Fails on the Disk"}
[Sampling Pitfall & Verification]{.kicker}

Comparing $10^6$ inverse-transform draws $R = \sqrt{U}$ against 2D rejection-sampled darts and the naive shortcut $R = U$:

@mdl-random-variables-quantiles-and-inverse-transform-sampling

::: {.d2l-note .warn}
**Geometric distortion of $R = U$:** Setting $R = U$ spreads radial distances uniformly on $[0, 1]$, placing $50\%$ of darts inside $r < 0.5$ instead of $25\%$ and underestimating the mean radius ($0.500$ vs. $0.667$).
:::
:::

::: {.slide title="Expectation: The Center of Mass of a Distribution"}
[First Moment & Law of the Unconscious Statistician]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mu_X = \mathbb{E}[X] = \int_{-\infty}^{\infty} x\,p(x)\,dx, \qquad \mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x)\,p(x)\,dx$$

- **Dart mean vs. median:** $\mathbb{E}[R] = \int_0^1 r(2r)\,dr = \frac{2}{3} \approx 0.667$, sitting slightly below the median $\sqrt{0.5} \approx 0.707$ because the long left tail pulls the center of mass inward.
- **LOTUS & existence:** Any function $g(X)$ averages directly against $p(x)$ without deriving $p_{g(X)}$, provided $\int |g(x)|\,p(x)\,dx < \infty$.
:::

::: {.col .fig}
@fig:ch35-prob-mean-lotus
:::
:::
:::

::: {.slide title="Linearity of Expectation Needs No Independence"}
[Algebra of the Mean]{.kicker}

**Proposition (Linearity of Expectation):** For any random variables $X, Y$ with finite means and constants $c, d \in \mathbb{R}$,

$$\mathbb{E}[cX + d] = c\,\mathbb{E}[X] + d, \qquad \mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$$

::: {.cols}
::: {.col}
- **Proof of additivity:** Weighting $x_i + y_j$ by the joint distribution $p_{ij} = P(X=x_i, Y=y_j)$ and marginalizing each sum separately gives
  $$\sum_{i,j} (x_i + y_j)p_{ij} = \sum_i x_i \underbrace{\sum_j p_{ij}}_{p_X(x_i)} + \sum_j y_j \underbrace{\sum_i p_{ij}}_{p_Y(y_j)}$$
:::

::: {.col}
- **Why this matters in deep learning:**
  - $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ holds **even when $X$ and $Y$ are strongly dependent**.
  - The expected average loss over a correlated minibatch or sequence of tokens always equals the average of the individual expected losses.
:::
:::
:::

::: {.slide title="Variance and Standard Deviation"}
[Second Central Moment]{.kicker}

::: {.cols .vc}
::: {.col}
$$\sigma_X^2 = \mathrm{Var}(X) = \mathbb{E}\bigl[(X - \mu_X)^2\bigr] = \mathbb{E}[X^2] - \mu_X^2$$

- **Affine transformation rule:** Shifting by $d$ leaves spread unchanged, while scaling by $c$ squares the variance ($\mathrm{Var}(cX + d) = c^2\,\mathrm{Var}(X)$ and $\sigma_{cX + d} = |c|\,\sigma_X$).
- **Worked benchmarks:** Dart radius $R$ has $\mathbb{E}[R^2] = 1/2 \implies \mathrm{Var}(R) = 1/2 - 4/9 = 1/18$ ($\sigma_R \approx 0.236$); uniform $U(0, 1)$ has $\mathrm{Var}(U) = 1/3 - 1/4 = 1/12$.
:::

::: {.col .fig}
@fig:ch35-prob-var-affine
:::
:::
:::

::: {.slide title="Markov's Inequality: Tail Bounds from the Mean Alone"}
[The Ancestor of All Concentration Bounds]{.kicker}

::: {.cols .vc}
::: {.col}
$$P(X \ge t) \le \frac{\mathbb{E}[X]}{t} \quad \text{for } X \ge 0,\ t > 0$$

- **One-line proof:** Since $X \ge t\,\mathbf{1}_{X \ge t}$ holds pointwise for every outcome $X \ge 0$, taking expectations gives $\mathbb{E}[X] \ge t\,P(X \ge t)$.
- **Rule of thumb & lineage:** At most $10\%$ of any non-negative population can exceed $10\times$ its average; applying Markov to $(X-\mu)^2$ gives Chebyshev, and to $e^{\lambda X}$ gives Chernoff/Hoeffding.
:::

::: {.col .fig}
@fig:ch35-prob-markov-step
:::
:::
:::

::: {.slide title="Chebyshev's Inequality and Its Three-Atom Sharpness"}
[Two-Sided Variance Bound]{.kicker}

::: {.cols .vc}
::: {.col}
- **Proposition (Chebyshev's Inequality):** For any $X$ with finite variance $\sigma_X^2 > 0$ and any $\alpha > 0$,
  $$P\bigl(|X - \mu_X| \ge \alpha\sigma_X\bigr) \le \frac{1}{\alpha^2}$$
- **Proof:** Apply Markov's inequality to the non-negative variable $Z = (X - \mu_X)^2$ (which has mean $\sigma_X^2$) at threshold $t = \alpha^2\sigma_X^2$.
- **Why $1/\alpha^2$ cannot be improved:** A three-atom distribution at $a - 2, a, a + 2$ with probabilities $q, 1-2q, q$ has $\sigma_X = 2\sqrt{2q}$. Setting $q = \frac{1}{2\alpha^2}$ places the outer atoms right at $\alpha\sigma_X = 2$, achieving exact equality $P(|X - a| \ge \alpha\sigma_X) = 2q = 1/\alpha^2$.
:::

::: {.col .fig}
@fig:mdl-prob-chebyshev
:::
:::
:::

::: {.slide title="Worst-Case vs. Typical Tails"}
[Chebyshev vs. Smooth Distributions]{.kicker}

Chebyshev's inequality must cover the worst-case distribution that piles all of its tail mass right at the threshold $\pm \alpha\sigma$. Smooth distributions thin out much faster:

| Distance from Mean $|X - \mu|$ | Chebyshev Upper Bound ($1/\alpha^2$) | Dart Distance $R$ ($p(r)=2r$) | Gaussian $\mathcal{N}(\mu, \sigma^2)$ |
| :--- | ---: | ---: | ---: |
| $\ge 2\sigma$ ($\alpha = 2$) | $25.0\%$ | $3.8\%$ ($R \le 0.195$) | $4.55\%$ |
| $\ge 3\sigma$ ($\alpha = 3$) | $11.1\%$ | $0.0\%$ (outside $[0,1]$) | $0.27\%$ |
| $\ge 5\sigma$ ($\alpha = 5$) | $4.0\%$ | $0.0\%$ | $5.7 \times 10^{-5}\%$ |
| $\ge 10\sigma$ ($\alpha = 10$) | $1.0\%$ | $0.0\%$ | $1.5 \times 10^{-21}\%$ |

::: {.d2l-note}
**Value of distribution-free bounds:** Although $1/\alpha^2$ is loose for a single Gaussian, applying it to a sample average $\bar{X}_n$ (whose variance is $\sigma^2/n$) suffices to prove the Weak Law of Large Numbers for *any* finite-variance distribution.
:::
:::

::: {.slide title="Heavy Tails and the Cauchy Distribution"}
[When Moments Fail to Exist]{.kicker}

::: {.cols .vc}
::: {.col}
$$p(x) = \frac{1}{\pi(1 + x^2)}, \qquad x \in \mathbb{R}$$

- **Infinite second moment ($\mathbb{E}[X^2] = \infty$):** Because the tail decays algebraically like $1/(\pi x^2)$, $\int_{-L}^L x^2 p(x)\,dx = \frac{2}{\pi}(L - \arctan L) \to \infty$.
- **Undefined mean ($\mathbb{E}[|X|] = \infty$):** Since $\int_{-L}^L |x|\,p(x)\,dx = \frac{1}{\pi}\log(1+L^2) \to \infty$, asymmetric limits $[-L, 2L]$ shift $\int x\,p(x)\,dx \to \frac{\log 2}{\pi} \approx 0.221$.
:::

::: {.col .fig}
@fig:ch35-prob-cauchy-tail
:::
:::
:::

::: {.slide title="When Averaging Fails: Gaussian vs. Cauchy Draws"}
[Breakdown of the Law of Large Numbers]{.kicker}

::: {.cols .vc}
::: {.col}
- **Catastrophic jumps:** As $n$ grows to $10^5$, the Gaussian running mean settles smoothly to $0$ at rate $1/\sqrt{n}$, whereas the Cauchy running mean suffers sudden $O(1)$ jumps whenever a single extreme tail draw arrives.
- **Scale invariance of Cauchy averages:** The average of $n$ independent standard Cauchy draws has the *exact same distribution* as a single Cauchy draw ($P(|\bar{X}_{1000}| > 1) = 0.50$, identical to $P(|X_1| > 1) = 0.50$).
- **Deep learning impact:** When minibatch SGD gradient noise exhibits heavy power-law tails ($P(|X| > t) \sim t^{-\alpha}$ with $\alpha < 2$), naive averaging fails and gradient clipping is required.
:::

::: {.col .fig}
@!random-variables-cauchy-diverges
:::
:::
:::

::: {.slide title="Concept Check: Densities, Quantiles & Tail Bounds"}
[Check Your Understanding]{.kicker}

Test your intuition on three foundational calculations from this section:

1. **Power-law sampler and clipping:** Let $p(x) = 1/x^2$ for $x \ge 1$ ($0$ otherwise). What is its CDF $F(x)$, its inverse-transform sampler $X = F^{-1}(U)$, and its expected value $\mathbb{E}[X]$?
2. **Affine variance and Chebyshev:** If $\mathrm{Var}(X) = 2$, what is $\mathrm{Var}(3X - 7)$? For any random variable with $\sigma_X > 0$, what is the maximum possible probability that $|X - \mu_X| \ge 4\sigma_X$?
3. **He initialization and the Rectified Gaussian:** For a ReLU output $Y = \max(0, X)$ with $X \sim \mathcal{N}(0, 1)$, use symmetry ($X^2 = Y^2 + \min(0, X)^2$) to find $\mathbb{E}[Y^2]$. Why does He initialization multiply weight variance by $2$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. $F(x) = \int_1^x t^{-2}\,dt = 1 - 1/x$ for $x \ge 1$. Solving $1 - 1/x = U$ gives sampler **$X = 1/(1 - U)$** (median $= 2$). Its mean $\int_1^\infty x(1/x^2)\,dx = [\log x]_1^\infty = \boldsymbol{\infty}$ (clipping at $c$ gives finite mean $1 + \log c$).
2. $\mathrm{Var}(3X - 7) = 3^2 \mathrm{Var}(X) = \mathbf{18}$ (shift drops out, scale squares). By Chebyshev, $P(|X - \mu_X| \ge 4\sigma_X) \le 1/4^2 = \mathbf{1/16 = 6.25\%}$.
3. Since $\mathbb{E}[X^2] = 1$ and $Y^2$ has the same distribution as $\min(0, X)^2$ with $Y \cdot \min(0, X) = 0$, we get $2\mathbb{E}[Y^2] = 1 \implies \boldsymbol{\mathbb{E}[Y^2] = 1/2}$. He initialization multiplies weight variance by $2$ to restore unit activation second moment across layers!
:::
:::

::: {.slide title="Summary: Random Variables, Moments, and Tails"}
[Key Takeaways]{.kicker}

| Concept | Mathematical Definition / Rule | Dart Benchmark ($R$ on Unit Disk) | Deep Learning Role |
| :--- | :--- | :--- | :--- |
| **Probability Density $p(x)$** | $P(a < X \le b) = \int_a^b p(x)\,dx$, $p \ge 0$, $\int p = 1$ | $p(r) = 2r$ on $[0, 1]$ ($P(R = r) = 0$) | Continuous representations, weights & noise |
| **CDF & Derivative** | $F(x) = P(X \le x) = \int_{-\infty}^x p(t)\,dt$, $F'(x) = p(x)$ | $F(r) = r^2$, $P(R > 0.5) = 1 - 0.5^2 = 0.75$ | Unifies continuous, discrete & mixed (ReLU) |
| **Inverse-Transform Sampler** | $X = F^{-1}(U)$ for $U \sim \mathrm{Unif}(0, 1)$ has CDF $F$ | $R = \sqrt{U}$ (median $F^{-1}(0.5) \approx 0.707$) | Sampling arbitrary 1D distributions & calibration |
| **Expectation & Linearity** | $\mathbb{E}[g(X)] = \int g(x)p(x)\,dx$; $\mathbb{E}[X+Y] = \mathbb{E}[X]+\mathbb{E}[Y]$ | $\mathbb{E}[R] = \int_0^1 2r^2\,dr = 2/3 \approx 0.667$ | Unbiased empirical risk & SGD gradients |
| **Variance & Affine Scaling** | $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mu^2$; $\mathrm{Var}(cX+d) = c^2\mathrm{Var}(X)$ | $\mathbb{E}[R^2] = 1/2 \implies \mathrm{Var}(R) = 1/18$ | Activation scale control & layer normalization |
| **Markov & Chebyshev Bounds** | $P(X \ge t) \le \mathbb{E}[X]/t$; $P(\|X-\mu\| \ge \alpha\sigma) \le 1/\alpha^2$ | Exact $2\sigma$ tail $3.8\% \le 25\%$ bound | Distribution-free tail & convergence guarantees |
| **Heavy Tails (Cauchy)** | Algebraic decay $p(x) \sim x^{-(\alpha+1)}$ ($\mathbb{E}[\|X\|^k] = \infty, k \ge \alpha$) | Cauchy ($\alpha=1$): no mean/var, $\bar{X}_n \stackrel{d}{=} X_1$ | Gradient clipping under heavy-tailed SGD noise |
:::

