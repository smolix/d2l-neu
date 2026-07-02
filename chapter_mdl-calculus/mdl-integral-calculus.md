# Integral Calculus
:label:`sec_mdl-integral_calculus`

Differentiation answered a local question: how does a function change when we
nudge its input? Integration answers a global one: how much of something is
there in total---the area under a curve, the volume under a surface, the
probability under a density. The two look unrelated, yet the *fundamental
theorem of calculus* welds them into a single subject: integration is
differentiation run backwards. That one fact is what makes integrals
computable at all, and it is the reason a deep-learning reader needs them, since
every continuous probability is an integral and every expectation is an integral
average (:numref:`sec_mdl-random_variables`).

We will not need the full machinery of a calculus course. We need three things:
what an integral *is* (a limit of sums), the theorem that lets us *compute* it
(the fundamental theorem), and the *change-of-variables* rule that powers the
Gaussian normalizer and, later, normalizing flows
(:numref:`sec_mdl-continuous-normalizing-flows`). The worked cells branch per
framework, so we load each framework's library once here.

```{.python .input #integral-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input #integral-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input #integral-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input #integral-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
import jax
from jax import numpy as jnp
```

## The Definite Integral

Take a function $f$ and ask for the area trapped between its graph and the
$x$-axis over an interval $[a, b]$. The area under an entire curve is usually
infinite or undefined, so we always pin it between two endpoints. The recipe is
the same one Archimedes used: chop $[a,b]$ into $N$ thin vertical slices of width
$\epsilon = (b-a)/N$, approximate each slice by a rectangle of height $f(x_i)$,
and add them up. As the slices shrink, the staircase of rectangles squeezes onto
the curve and the sum approaches a definite number, shown in
:numref:`fig_mdl-riemann`.

![Left to right: the area under $f$ from $a$ to $b$ approximated by rectangles of shrinking width $\epsilon$. As $\epsilon\to 0$ the rectangle sum $\sum_i \epsilon\,f(x_i)$ converges to the definite integral, the exact signed area.](../img/mdl-cal-riemann.svg)
:label:`fig_mdl-riemann`

We *define* the **definite integral** as this limit and write it with the
elongated-$S$ symbol,

$$
\int_a^b f(x)\,dx = \lim_{\epsilon\to 0}\ \sum_i \epsilon\, f(x_i),
$$
:eqlabel:`eq_mdl-riemann`

a continuous analogue of a sum: $\sum$ becomes $\int$, the spacing $\epsilon$
becomes $dx$, and the term $f(x_i)$ becomes $f(x)$. The inner variable is a dummy,
exactly like a summation index, so $\int_a^b f(x)\,dx = \int_a^b f(z)\,dz$. For
bounded, piecewise-continuous $f$---all we ever meet in machine learning---the
limit exists and does not depend on how the slices are chosen. (Some hypothesis
is genuinely needed: for the function that is $1$ on the rationals and $0$
elsewhere, the rectangle sums never settle on any value at all. Lebesgue's
integral, built on measure theory rather than on slicing the axis, handles even
that function---its integral is $0$, because the rationals have measure
zero---and it is the foundation probability theory ultimately stands on; the
Riemann picture suffices for every computation in this book.)

The definition is honest but not yet a *computation*: only the simplest
integrands (a line, $\int_a^b x\,dx$) succumb to summing the rectangles by hand.
The next section gives the tool that computes the rest. First let us check that
the limit :eqref:`eq_mdl-riemann` is real by watching it converge. Take
$\int_0^2 \tfrac{x}{1+x^2}\,dx$, whose exact value is $\tfrac12\log 5$ (we will
see why shortly), and refine the partition: the rectangle sum should march toward
the truth, with the error shrinking roughly in proportion to $\epsilon$ for this
left-endpoint rule.

```{.python .input #integral-riemann-converge}
#@tab mxnet
def riemann(eps, a=0., b=2.):
    x = np.arange(a, b, eps)
    return float(np.sum(eps * x / (1 + x**2)))

truth = float(np.log(np.array(5.)) / 2)
for eps in [0.5, 0.1, 0.05, 0.01, 0.001]:
    approx = riemann(eps)
    print(f'eps={eps:<6} sum={approx:.6f}  error={abs(approx - truth):.6f}')
print(f'truth = (1/2) log 5 = {truth:.6f}')
```

```{.python .input #integral-riemann-converge}
#@tab pytorch
def riemann(eps, a=0., b=2.):
    x = torch.arange(a, b, eps)
    return float(torch.sum(eps * x / (1 + x**2)))

truth = float(torch.log(torch.tensor(5.)) / 2)
for eps in [0.5, 0.1, 0.05, 0.01, 0.001]:
    approx = riemann(eps)
    print(f'eps={eps:<6} sum={approx:.6f}  error={abs(approx - truth):.6f}')
print(f'truth = (1/2) log 5 = {truth:.6f}')
```

```{.python .input #integral-riemann-converge}
#@tab tensorflow
def riemann(eps, a=0., b=2.):
    x = tf.range(a, b, eps)
    return float(tf.reduce_sum(eps * x / (1 + x**2)))

truth = float(tf.math.log(5.) / 2)
for eps in [0.5, 0.1, 0.05, 0.01, 0.001]:
    approx = riemann(eps)
    print(f'eps={eps:<6} sum={approx:.6f}  error={abs(approx - truth):.6f}')
print(f'truth = (1/2) log 5 = {truth:.6f}')
```

```{.python .input #integral-riemann-converge}
#@tab jax
def riemann(eps, a=0., b=2.):
    x = jnp.arange(a, b, eps)
    return float(jnp.sum(eps * x / (1 + x**2)))

truth = float(jnp.log(5.) / 2)
for eps in [0.5, 0.1, 0.05, 0.01, 0.001]:
    approx = riemann(eps)
    print(f'eps={eps:<6} sum={approx:.6f}  error={abs(approx - truth):.6f}')
print(f'truth = (1/2) log 5 = {truth:.6f}')
```

The error falls by about a factor of ten each time we cut $\epsilon$ by ten,
confirming the first-order convergence of the left-rule. Refining the partition
*works*, but it is slow and gives no closed form. We need a better idea.

## The Fundamental Theorem of Calculus

The breakthrough is to stop treating the integral as a fixed number and instead
let its upper limit *move*. Define the **area-so-far function**

$$
F(x) = \int_a^x f(y)\,dy,
$$
:eqlabel:`eq_mdl-area-fn`

the signed area accumulated from the left endpoint $a$ out to $x$. This single
function captures *every* definite integral at once: the area over any
sub-interval is a difference of accumulated areas,

$$
\int_c^b f(x)\,dx = F(b) - F(c),
$$

since the area out to $b$ minus the area out to $c$ leaves exactly the area
between them. :numref:`fig_mdl-area-subtract` draws this subtraction.

![The area under $f$ from $a$ to $b$ equals the area-to-the-left of $b$ minus the area-to-the-left of $a$: $F(b) - F(a)$. Every definite integral is a difference of values of the single accumulation function $F$.](../img/mdl-cal-sub-area.svg)
:label:`fig_mdl-area-subtract`

So knowing $F$ solves the whole problem. The astonishing fact---the
*fundamental theorem of calculus*---is that $F$ is determined by a derivative.

**Theorem (Fundamental theorem of calculus).** *Let $f$ be continuous and let
$F(x)=\int_a^x f(y)\,dy$. Then $F$ is differentiable and*

$$
\frac{dF}{dx}(x) = f(x).
$$
:eqlabel:`eq_mdl-ftc`

**Proof.** Differentiate $F$ from the definition: nudge $x$ by $\epsilon$ and
read off how much new area appears. By :eqref:`eq_mdl-area-fn` the increment is a
single thin sliver,

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y)\,dy .
$$

The sliver sits over an interval of width $\epsilon$, and on it the continuous
function $f$ varies by at most $\max_{[x,x+\epsilon]}f - \min_{[x,x+\epsilon]}f$,
a gap that shrinks to $0$ as $\epsilon\to 0$. Bounding the area between the
smallest and largest rectangles of width $\epsilon$,

$$
\epsilon \min_{[x,x+\epsilon]} f
\ \le\ F(x+\epsilon) - F(x)\ \le\
\epsilon \max_{[x,x+\epsilon]} f .
$$

Divide by $\epsilon$. Because $f$ is continuous at $x$, its minimum and maximum
over the shrinking interval $[x,x+\epsilon]$ both converge to the single value
$f(x)$ (equivalently, the mean value theorem for integrals places the average on
the curve), so the squeeze forces
$\lim_{\epsilon\to 0}\frac{F(x+\epsilon)-F(x)}{\epsilon}=f(x)$ (for
$\epsilon<0$ the sliver sits to the left of $x$ and the same bounds apply, with
both sign flips canceling in the quotient). That limit is
exactly $F'(x)$. $\blacksquare$

The sliver argument is the whole story: the rate at which accumulated area grows
is just the current height of the curve. This *reverses* the problem. Finding
areas, hard on its own, becomes the search for an **antiderivative**---a function
whose derivative is $f$---which we can read straight off the derivative table of
:numref:`sec_mdl-derivative_table`. If $G'=f$ then $G$ and $F$ differ by a
constant (they have the same derivative), and since that constant cancels in any
difference,

$$
\int_a^b f(x)\,dx = G(b) - G(a)
$$
:eqlabel:`eq_mdl-ftc-eval`

for *any* antiderivative $G$. The arbitrary constant of integration is real but
irrelevant to definite integrals.

This turns hard sums into easy lookups. The derivative of $x^n$ is $nx^{n-1}$, so
running it backwards,

$$
\int_0^{x} n\,y^{n-1}\,dy = x^n - 0^n = x^n ,
$$

and since $e^x$ is its own derivative,

$$
\int_0^{x} e^{t}\,dt = e^{x} - e^{0} = e^x - 1 .
$$

Every integration rule is a differentiation rule read in reverse, which is why
the running example obeyed $\int_0^2 x/(1+x^2)\,dx=\tfrac12\log 5$: the
antiderivative of $x/(1+x^2)$ is $\tfrac12\log(1+x^2)$, evaluated at the
endpoints. Let us confirm the theorem numerically by checking that the
area-so-far function $F$ has derivative $f$: compute $F$ as a cumulative Riemann
sum, finite-difference it, and compare against $f$ itself.

```{.python .input #integral-ftc-check}
#@tab mxnet
f = lambda t: t / (1 + t**2)
G = lambda t: np.log(1 + t**2) / 2          # antiderivative: F(x) = G(x) - G(0)
eps = 1e-3
x = np.arange(0., 2., eps)
F = np.cumsum(eps * f(x))                    # area-so-far via Riemann sum
dFdx = (F[1:] - F[:-1]) / eps                # telescopes to f(x[1:]) exactly
print('max |dF/dx - f|      :', float(np.abs(dFdx - f(x[1:])).max()))
print('Riemann F(2) vs G(2)-G(0):', float(F[-1]), float(G(np.array(2.))))
```

```{.python .input #integral-ftc-check}
#@tab pytorch
f = lambda t: t / (1 + t**2)
G = lambda t: torch.log(1 + t**2) / 2        # antiderivative: F(x) = G(x) - G(0)
eps = 1e-3
x = torch.arange(0., 2., eps)
F = torch.cumsum(eps * f(x), dim=0)          # area-so-far via Riemann sum
dFdx = (F[1:] - F[:-1]) / eps                # telescopes to f(x[1:]) exactly
print('max |dF/dx - f|      :', float((dFdx - f(x[1:])).abs().max()))
print('Riemann F(2) vs G(2)-G(0):', float(F[-1]), float(G(torch.tensor(2.))))
```

```{.python .input #integral-ftc-check}
#@tab tensorflow
f = lambda t: t / (1 + t**2)
G = lambda t: tf.math.log(1 + t**2) / 2      # antiderivative: F(x) = G(x) - G(0)
eps = 1e-3
x = tf.range(0., 2., eps)
F = tf.cumsum(eps * f(x))                     # area-so-far via Riemann sum
dFdx = (F[1:] - F[:-1]) / eps                 # telescopes to f(x[1:]) exactly
print('max |dF/dx - f|      :', float(tf.reduce_max(tf.abs(dFdx - f(x[1:])))))
print('Riemann F(2) vs G(2)-G(0):', float(F[-1]), float(G(tf.constant(2.))))
```

```{.python .input #integral-ftc-check}
#@tab jax
f = lambda t: t / (1 + t**2)
G = lambda t: jnp.log(1 + t**2) / 2          # antiderivative: F(x) = G(x) - G(0)
eps = 1e-3
x = jnp.arange(0., 2., eps)
F = jnp.cumsum(eps * f(x))                    # area-so-far via Riemann sum
dFdx = (F[1:] - F[:-1]) / eps                 # telescopes to f(x[1:]) exactly
print('max |dF/dx - f|      :', float(jnp.abs(dFdx - f(x[1:])).max()))
print('Riemann F(2) vs G(2)-G(0):', float(F[-1]), float(G(jnp.array(2.))))
```

The first comparison is exact in exact arithmetic: differencing the cumulative
sum *telescopes*, $F(x+\epsilon)-F(x)$ collapsing back to the single term
$\epsilon f(x+\epsilon)$ that the sum just added. The printed error---a few
parts in $10^{5}$---is therefore not discretization error at all but pure
float32 roundoff in the accumulated $F$, amplified by the division by
$\epsilon$. The Riemann total $F(2)$ agrees with the antiderivative value
$G(2)$ to about $2\times 10^{-4}$, the expected first-order bias of the
left-endpoint rule---the fundamental theorem made flesh.

### Improper Integrals

Many densities live on an unbounded domain---the Gaussian, the exponential, any
heavy tail---so we will need to integrate "all the way to infinity," yet
:eqref:`eq_mdl-riemann` only defined integrals over a finite $[a,b]$. We fill the
gap the same way we defined the integral itself: as a limit. The **improper
integral** is

$$
\int_a^\infty f(x)\,dx = \lim_{b\to\infty}\int_a^b f(x)\,dx ,
$$
:eqlabel:`eq_mdl-improper`

with $\int_{-\infty}^\infty$ defined by splitting at any point and taking both
limits. The limit may be finite, in which case the integral *converges*;
otherwise---whether the partial integrals grow without bound or merely fail to
settle---it *diverges*. Whether the tail is "thin enough" is a
genuine question. The cleanest test case is the power law $x^{-p}$, for which the
antiderivative gives

$$
\int_1^\infty x^{-p}\,dx = \lim_{b\to\infty}\frac{b^{1-p}-1}{1-p}
= \begin{cases} \dfrac{1}{p-1}, & p>1\ \text{(converges)},\\[1ex] \infty, & p\le 1\ \text{(diverges)}.\end{cases}
$$

The antiderivative $\tfrac{b^{1-p}-1}{1-p}$ is the $0/0$ indeterminate form at
exactly $p=1$, where the integrand $x^{-1}$ has antiderivative $\log x$ instead, so
$\int_1^\infty x^{-1}\,dx=\lim_{b\to\infty}\log b=\infty$; the case split above
already records this divergent value. So $\int_1^\infty x^{-2}\,dx = 1$ while
$\int_1^\infty x^{-1}\,dx=\infty$: the boundary between convergence and divergence
sits exactly at $p=1$. This single threshold is what decides whether a
heavy-tailed density even has a finite
normalizer or mean, a recurring concern once we reach probability. The cell
watches a convergent improper integral, $\int_0^\infty e^{-x}\,dx = 1$, through
two lenses at once: the *exact* partial integrals
$\int_0^b e^{-x}\,dx = 1-e^{-b}$, read off the antiderivative, and a
left-Riemann approximation of each with $\epsilon = 10^{-3}$.

```{.python .input #integral-improper}
#@tab mxnet
for b in [1., 2., 5., 10., 20.]:
    x = np.arange(0., b, 1e-3)
    left = float(np.sum(1e-3 * np.exp(-x)))
    exact = 1 - float(np.exp(np.array(-b)))
    print(f'b={b:<4}  left-Riemann sum={left:.6f}  exact 1-e^-b={exact:.6f}')
```

```{.python .input #integral-improper}
#@tab pytorch
for b in [1., 2., 5., 10., 20.]:
    x = torch.arange(0., b, 1e-3)
    left = float(torch.sum(1e-3 * torch.exp(-x)))
    exact = 1 - float(torch.exp(torch.tensor(-b)))
    print(f'b={b:<4}  left-Riemann sum={left:.6f}  exact 1-e^-b={exact:.6f}')
```

```{.python .input #integral-improper}
#@tab tensorflow
for b in [1., 2., 5., 10., 20.]:
    x = tf.range(0., b, 1e-3)
    left = float(tf.reduce_sum(1e-3 * tf.exp(-x)))
    exact = 1 - float(tf.exp(-b))
    print(f'b={b:<4}  left-Riemann sum={left:.6f}  exact 1-e^-b={exact:.6f}')
```

```{.python .input #integral-improper}
#@tab jax
for b in [1., 2., 5., 10., 20.]:
    x = jnp.arange(0., b, 1e-3)
    left = float(jnp.sum(1e-3 * jnp.exp(-x)))
    exact = 1 - float(jnp.exp(-b))
    print(f'b={b:<4}  left-Riemann sum={left:.6f}  exact 1-e^-b={exact:.6f}')
```

Read the two columns against each other. The exact partials race up to $1$: the
*truncation error* of stopping at $b$ is $e^{-b}$, already invisible at six
decimals by $b=20$. The left-Riemann column tells a subtler story: it
overshoots and settles near $1.0005$, not $1$. That residue is *discretization
error*. The left rule samples each slice at its left edge, where the decreasing
integrand $e^{-x}$ is largest, so every rectangle overcounts slightly; the
overcounts add up to a bias of
$\approx \tfrac{\epsilon}{2}\bigl(f(0)-f(b)\bigr) \to \tfrac{\epsilon}{2}
= 0.0005$. The two error sources are independent knobs---growing $b$ kills the
first, shrinking $\epsilon$ the second---and no amount of turning one
compensates for neglecting the other.

### Integration by Parts
:label:`subsec_mdl-integration-by-parts`

The fundamental theorem turns every differentiation rule into an integration
rule, and the product rule is the most profitable conversion.

**Proposition (Integration by parts).** *For continuously differentiable $u$
and $v$ on $[a,b]$,*

$$
\int_a^b u(x)\,v'(x)\,dx
= \Bigl[\,u(x)\,v(x)\,\Bigr]_a^b - \int_a^b u'(x)\,v(x)\,dx ,
$$
:eqlabel:`eq_mdl-parts`

*where $[\,h(x)\,]_a^b$ abbreviates $h(b)-h(a)$.*

**Proof.** By the product rule, $uv$ is an antiderivative of $u'v + uv'$, so
:eqref:`eq_mdl-ftc-eval` gives
$\int_a^b (u'v + uv')\,dx = [\,u(x)v(x)\,]_a^b$; subtract $\int_a^b u'v\,dx$
from both sides. $\blacksquare$

The formula is a *trade*, not an evaluation: it moves the derivative from one
factor onto the other, and the trade pays whenever the new integrand
$u'v$ is simpler than the old one $uv'$. The model example takes $u(x)=x$ and
$v'(x)=e^{-x}$, so that $u'=1$ and $v=-e^{-x}$: applying
:eqref:`eq_mdl-parts` on $[0,b]$ and letting $b\to\infty$,

$$
\int_0^\infty x\,e^{-x}\,dx
= \Bigl[\,-x\,e^{-x}\,\Bigr]_0^\infty + \int_0^\infty e^{-x}\,dx
= 0 + 1 = 1 .
$$

The boundary term vanishes because $x e^{-x}\to 0$ at both ends, and what
survives is exactly the improper integral computed above. This particular
value is the mean of the exponential distribution
(:numref:`sec_mdl-distributions`), and it shows the form in which the trick
returns throughout probability: expectations $\int g(x)\,p(x)\,dx$ are
integrals of products, and shifting a derivative from a factor we cannot handle
onto one we can is often the only move available. Its most spectacular reprise
is Hyvärinen's score-matching identity
(:numref:`sec_mdl-score-matching-diffusion-flow`), where a single integration
by parts converts an objective involving the *unknowable* score of the data
distribution into one that can be estimated from samples.

### A Note on Signed Area

The evaluation rule :eqref:`eq_mdl-ftc-eval` cheerfully produces negative numbers,
which can be jarring if "area" is supposed to be positive. The resolution is that
integrals measure *signed* area, governed by two independent sign rules. First,
where $f<0$ the area counts as negative: $\int_0^1 (-1)\,dx = -1$. Second,
integrating right-to-left negates the result: $\int_b^a f = -\int_a^b f$. Each
flip---reflecting across the $x$-axis, or reversing the limits---introduces one
minus sign; :numref:`fig_mdl-signed-area` shows both rules on one curve. Apply
them in turn to a single example: $\int_0^{-1} 1\,dx = -1$ (one
flip, from reversed limits), while $\int_0^{-1}(-1)\,dx = 1$ (two flips, which
cancel). This is the same signed-area bookkeeping the
determinant did for transformed regions in
:numref:`sec_mdl-geometry-linear-algebraic-ops`, and it is exactly what makes the
change-of-variables formula below come out right.

![Signed area. Lobes above the axis count positive and the lobe below counts negative, so $\int_a^b f$ totals the shaded regions with their signs; reversing the limits of integration traverses the same area right-to-left and negates the total.](../img/mdl-cal-signed-area.svg)
:label:`fig_mdl-signed-area`

## Change of Variables

Every differentiation rule has an integration counterpart obtained by running the
fundamental theorem backwards: the sum rule gives linearity, the product rule
gave us integration by parts (:numref:`subsec_mdl-integration-by-parts`),
and---most useful for us---the chain rule gives the
*change-of-variables* formula. It is the substitution that tames otherwise
intractable integrals; carried into higher dimensions in the next section, it
cracks open the Gaussian normalizer. Read for *densities* rather than areas, the
very same formula becomes the engine behind normalizing flows---a specialization
we hand off to :numref:`sec_mdl-random_variables`.

Suppose we reparametrize the variable through a function $u$. The chain rule says
$\frac{d}{dx}F(u(x)) = F'(u(x))\,u'(x)$; feeding this through the fundamental
theorem with $F'=f$ and integrating both sides yields the
**change-of-variables formula**

$$
\int_{u(a)}^{u(b)} f(y)\,dy = \int_a^b f(u(x))\,\frac{du}{dx}\,dx .
$$
:eqlabel:`eq_mdl-change_var`

The picture explains the mysterious factor $\frac{du}{dx}$. Look at one thin
rectangle. On the left side of :eqref:`eq_mdl-change_var`, the sliver from $x$ to
$x+\epsilon$ has area $\approx \epsilon\,f(u(x))$. On the right, the corresponding
sliver runs from $u(x)$ to $u(x+\epsilon)\approx u(x)+\epsilon\,u'(x)$, so its
width is *stretched* by the factor $u'(x)$ and its area is
$\approx \epsilon\,u'(x)\,f(u(x))$. To make the two slivers agree we must multiply
by exactly the local stretch $\frac{du}{dx}$, as :numref:`fig_mdl-rect-transform`
shows.

![A thin rectangle under the substitution $y=u(x)$. The sliver of width $\epsilon$ at $x$ maps to a sliver of width $\epsilon\,u'(x)$; matching the two areas forces the factor $du/dx$ in the change-of-variables formula.](../img/mdl-cal-rect-trans.svg)
:label:`fig_mdl-rect-transform`

With the right $u$ this collapses hard integrals to trivial ones. Taking $f=1$
and $u(x)=e^{-x^2}$ (so $u'(x)=-2x\,e^{-x^2}$),

$$
e^{-1}-1 = \int_{e^{0}}^{e^{-1}} 1\,dy = -2\int_0^{1} x\,e^{-x^2}\,dx
\quad\Longrightarrow\quad
\int_0^{1} x\,e^{-x^2}\,dx = \frac{1-e^{-1}}{2},
$$

a closed form that no amount of rectangle-summing would have revealed. (Keep the
two variables apart: $y$ is the transformed variable on the left of
:eqref:`eq_mdl-change_var`, $x$ the original one on the right; the result is a
statement about the $x$-integral.)

## Multiple Integrals

In one dimension the integral was a limit of rectangle sums, and nothing in that
recipe was attached to the line. This section runs the same recipe over regions
of $\mathbb{R}^n$, establishes the one theorem---Fubini's---that reduces such
integrals to iterated one-dimensional ones, lifts the change-of-variables rule
to $n$ dimensions, and cashes everything in on the Gaussian integral.

### Double Integrals

In higher dimensions we integrate over regions. For $f(x,y)$ on a rectangle
$U=[a,b]\times[c,d]$, the integral is the (signed) volume trapped between the
surface $z=f(x,y)$ and the base plane, written $\int_U f(x,y)\,dx\,dy$, and
Archimedes' recipe survives intact: tile $U$ with $\epsilon\times\epsilon$
squares, stand a box of height $f$ on each, and total the volumes.
:numref:`fig_mdl-bell-surface` shows our running example, the two-dimensional
bell $e^{-x^2-y^2}$ over the box $[-2,2]^2$; the cell totals its boxes.

![The bell surface $z = e^{-x^2-y^2}$ over the box $U = \lbrack -2,2\rbrack^2$. The double integral $\int_U f\,dx\,dy$ is the volume between the surface and the base plane, approximated by tiling the base with $\epsilon\times\epsilon$ squares and adding the volumes of the boxes standing on them.](../img/mdl-cal-bell-surface.svg)
:label:`fig_mdl-bell-surface`

```{.python .input #integral-box-volume}
eps = 1e-2
x, y = np.meshgrid(np.arange(-2., 2., eps), np.arange(-2., 2., eps),
                   indexing='ij')
vol = float(np.sum(eps ** 2 * np.exp(-x**2 - y**2)))
print(f'Riemann volume over the box [-2, 2]^2 : {vol:.5f}')
print('exact: pi*erf(2)^2 = 3.11227 over the box;  pi = 3.14159 over the plane')
```

The comparison line hides two teasers. The exact volume over the box involves
the *error function* $\operatorname{erf}(z)=\tfrac{2}{\sqrt{\pi}}\int_0^z
e^{-t^2}\,dt$---the Gaussian's antiderivative, suitably normalized, which
famously has no elementary formula---and as the box grows to the whole plane
the volume becomes exactly $\pi$. Both facts will fall out of the
change-of-variables machinery below; for now the cell is simply a concrete
two-dimensional Riemann sum.

### Fubini's Theorem

How do we get from grid sums back to the fundamental theorem? Discretize the
region into $\epsilon\times
\epsilon$ squares indexed by integers $i,j$; the integral is approximately
$\sum_{i,j}\epsilon^2 f(\epsilon i, \epsilon j)$. This is just a sum of numbers
on a grid, and **a finite sum can be totalled in any order**. Summing
columns-first---add up each column $i$, then add the column totals---we get

$$
\sum_j \epsilon\!\left(\sum_i \epsilon\, f(\epsilon i, \epsilon j)\right),
$$

illustrated in :numref:`fig_mdl-sum-order`. The inner sum is the Riemann
discretization of $\int_a^b f(x,\epsilon j)\,dx$, and the outer sum then
integrates that over $y$.

![Summing a grid of $\epsilon\times\epsilon$ cells columns-first (1), then adding the column totals together (2). Reordering a finite sum changes nothing---the discrete fact behind iterated integration.](../img/mdl-cal-sum-order.svg)
:label:`fig_mdl-sum-order`

Passing to the limit gives **Fubini's theorem**: a double integral equals either
iterated integral,

$$
\int_U f(x,y)\,dx\,dy
= \int_c^{d}\!\left(\int_a^{b} f(x,y)\,dx\right)dy
= \int_a^{b}\!\left(\int_c^{d} f(x,y)\,dy\right)dx .
$$
:eqlabel:`eq_mdl-fubini`

All we did was reorder a sum, so it may seem like nothing---but Fubini's theorem
is *not* unconditional. It needs $f$ to be absolutely integrable. The standard
counterexample is $f(x,y)=(x^2-y^2)/(x^2+y^2)^2$ on $[0,1]^2$: integrating $x$
first gives $-\tfrac{\pi}{4}$, integrating $y$ first gives $+\tfrac{\pi}{4}$, and
the orders disagree precisely because $f$ has a non-integrable singularity at the
origin. For the continuous, absolutely integrable functions of machine learning
there is no such trouble, and we freely swap orders. When the region is more
complicated than a rectangle we write the integral compactly as $\int_U
f(\mathbf{x})\,d\mathbf{x}$.

### Change of Variables in Many Dimensions

The one-dimensional stretch factor $\frac{du}{dx}$ has a multidimensional
successor, and it is the key that unlocks the Gaussian integral below. Let $U\subseteq\mathbb{R}^n$
be open and let $\boldsymbol{\phi}:U\to\mathbb{R}^n$ be a **$C^1$-diffeomorphism**
onto its image---that is, $\boldsymbol{\phi}$ is continuously differentiable,
injective, and has $\det D\boldsymbol{\phi}(\mathbf{x})\neq 0$ everywhere on $U$.
(Injectivity stops the map folding space onto itself; the nonvanishing Jacobian
keeps it from collapsing a region to lower dimension, and together they make
$\boldsymbol{\phi}$ invertible on $\boldsymbol{\phi}(U)$ with a $C^1$ inverse.)
Then, for absolutely integrable $f$,

$$
\int_{\boldsymbol{\phi}(U)} f(\mathbf{x})\,d\mathbf{x}
= \int_{U} f(\boldsymbol{\phi}(\mathbf{x}))\,
  \bigl|\det D\boldsymbol{\phi}(\mathbf{x})\bigr|\,d\mathbf{x},
$$
:eqlabel:`eq_mdl-change_var_nd`

where $D\boldsymbol{\phi}$ is the **Jacobian**, the matrix of partial derivatives

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi_1}{\partial x_1} & \cdots & \frac{\partial \phi_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi_n}{\partial x_1} & \cdots & \frac{\partial \phi_n}{\partial x_n}
\end{bmatrix}.
$$

Compare :eqref:`eq_mdl-change_var_nd` with the one-dimensional rule
:eqref:`eq_mdl-change_var`: the scalar stretch $\frac{du}{dx}$ has been replaced by
$|\det D\boldsymbol{\phi}|$. The reason is geometry. In one dimension $u'$ said
how much $u$ stretches a tiny interval; in $n$ dimensions the Jacobian is the best
linear approximation of $\boldsymbol{\phi}$ at a point (just as the gradient was
the best linear approximation of a scalar field), and the determinant of a linear
map is exactly the factor by which it scales area or volume---the signed-area
reading of the determinant from
:numref:`sec_mdl-geometry-linear-algebraic-ops`. So a tiny cube of volume
$d\mathbf{x}$ is sent to a tiny parallelepiped of volume
$|\det D\boldsymbol{\phi}|\,d\mathbf{x}$, and we multiply by that factor to keep
the bookkeeping honest. :numref:`fig_mdl-cov-jacobian` puts the one- and
two-dimensional pictures side by side.

![Left: the 1-D substitution scales a segment by $du/dx$. Right: in 2-D a linear $\boldsymbol{\phi}$ sends the unit square to a parallelogram whose area is the local volume-scaling factor $|\det D\boldsymbol{\phi}|$---the multidimensional successor of the stretch factor.](../img/mdl-cal-cov-jacobian.svg)
:label:`fig_mdl-cov-jacobian`

Equation :eqref:`eq_mdl-change_var_nd` is a statement about *area and volume*: it
says how the integral of a fixed function transports under a reparametrization of
the domain. Applied instead to a probability *density*---a function that must keep
integrating to $1$---the very same Jacobian factor becomes the
change-of-variables-for-densities rule of :numref:`sec_mdl-random_variables`, whose
$-\log|\det D\boldsymbol{\phi}|$ correction is the exact mechanism behind
**normalizing flows** (:numref:`sec_mdl-continuous-normalizing-flows`). We state the
area theorem here and defer that density specialization to the probability chapter.

### The Gaussian Integral

The classic payoff of the machinery just assembled is the **Gaussian
integral**, which we will meet again as the normalizer of the normal
distribution. It is the *one-dimensional* integral

$$
\int_{-\infty}^{\infty} e^{-x^2}\,dx
$$

that resists direct attack: $e^{-x^2}$ has no elementary antiderivative (its
antiderivative is the error function from the bell-volume cell, *defined* as
that integral rather than expressed in older functions), so the fundamental
theorem has nothing to grip. The trick is to go *up* a dimension. Square the
integral, rename the dummy variable in the second copy, and let Fubini
:eqref:`eq_mdl-fubini` fuse the iterated integrals into one double integral:

$$
\left(\int_{-\infty}^{\infty} e^{-x^2}\,dx\right)^{\!2}
= \int_{-\infty}^{\infty} e^{-x^2}\,dx \int_{-\infty}^{\infty} e^{-y^2}\,dy
= \int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty} e^{-x^2-y^2}\,dx\,dy .
$$
:eqlabel:`eq_mdl-gauss-square`

The two-dimensional integral has something its one-dimensional parent
lacked---rotational symmetry---and polar coordinates
$\boldsymbol{\phi}(r,\theta)=(r\cos\theta, r\sin\theta)$ are built to exploit
it. (The map fails injectivity only at the origin and
along the seam $\theta=0\equiv 2\pi$, a set of zero area that the theorem safely
ignores---the hypotheses need only hold off such a negligible set.) The Jacobian
determinant is

$$
\bigl|\det D\boldsymbol{\phi}\bigr|
= \left|\det\begin{bmatrix}\cos\theta & -r\sin\theta\\ \sin\theta & r\cos\theta\end{bmatrix}\right|
= r(\cos^2\theta + \sin^2\theta) = r,
$$

so the integrand picks up a factor $r$ that makes it elementary:

$$
\int_0^\infty\!\!\int_0^{2\pi} r\,e^{-r^2}\,d\theta\,dr
= 2\pi\int_0^\infty r\,e^{-r^2}\,dr = \pi .
$$

Taking the square root of :eqref:`eq_mdl-gauss-square` (the positive one, since
the integrand is positive) lands the prize:

$$
\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi} .
$$

The same factorization also redeems the bell-volume cell's comparison line:
over the box, $\bigl(\int_{-2}^{2} e^{-x^2}\,dx\bigr)^2
= \bigl(\sqrt{\pi}\,\operatorname{erf}(2)\bigr)^2
= \pi\operatorname{erf}(2)^2 \approx 3.11227$, shy of the full-plane $\pi$ by
exactly the truncated tails. The cell verifies the two-dimensional integral
numerically by summing the integrand over a fine grid on $[-6,6]^2$---a box
wide enough that the Gaussian's tails beyond it contribute nothing to six
decimals.

```{.python .input #integral-gaussian}
#@tab mxnet
g = np.arange(-6., 6., 0.01)
x, y = np.meshgrid(g, g, indexing='ij')
val = float(np.sum(0.01 ** 2 * np.exp(-x**2 - y**2)))
print('grid integral of e^(-x^2-y^2):', round(val, 6))
print('exact value pi               :', round(float(np.pi), 6))
```

```{.python .input #integral-gaussian}
#@tab pytorch
g = torch.arange(-6., 6., 0.01)
x, y = torch.meshgrid(g, g, indexing='ij')
val = float(torch.sum(0.01 ** 2 * torch.exp(-x**2 - y**2)))
print('grid integral of e^(-x^2-y^2):', round(val, 6))
print('exact value pi               :', round(float(torch.pi), 6))
```

```{.python .input #integral-gaussian}
#@tab tensorflow
g = tf.range(-6., 6., 0.01)
x, y = tf.meshgrid(g, g)
val = float(tf.reduce_sum(0.01 ** 2 * tf.exp(-x**2 - y**2)))
print('grid integral of e^(-x^2-y^2):', round(val, 6))
print('exact value pi               :', round(float(np.pi), 6))
```

```{.python .input #integral-gaussian}
#@tab jax
g = jnp.arange(-6., 6., 0.01)
x, y = jnp.meshgrid(g, g, indexing='ij')
val = float(jnp.sum(0.01 ** 2 * jnp.exp(-x**2 - y**2)))
print('grid integral of e^(-x^2-y^2):', round(val, 6))
print('exact value pi               :', round(float(jnp.pi), 6))
```

## Integration Meets Probability

This is why a deep-learning reader needs integration at all. A continuous
probability **density** is nothing more than a non-negative function that is
*normalized*---its total integral is $1$:

$$
p(x)\ge 0, \qquad \int_{\mathcal X} p(x)\,dx = 1 .
$$
:eqlabel:`eq_mdl-density`

The Gaussian integral above is precisely what supplies the normalizer: we
showed $\int_{-\infty}^\infty e^{-x^2}\,dx=\sqrt\pi$, so the function
$p(x)=\tfrac{1}{\sqrt\pi}e^{-x^2}$ integrates to $1$ and is a bona fide density.
An **expectation** is then an integral average---weighting each value by its
density,

$$
\mathbb{E}[X] = \int_{\mathcal X} x\,p(x)\,dx,
\qquad
\mathbb{E}[g(X)] = \int_{\mathcal X} g(x)\,p(x)\,dx .
$$
:eqlabel:`eq_mdl-expectation`

In several variables a joint density satisfies $\iint p(x,y)\,dx\,dy = 1$, and
*marginals* are obtained by integrating a variable out, $p(x)=\int p(x,y)\,dy$;
Fubini :eqref:`eq_mdl-fubini` guarantees the order of integration does not matter
(densities are absolutely integrable, so the caveat above never bites). The cell
verifies that $\tfrac{1}{\sqrt\pi}e^{-x^2}$ is normalized and computes its mean,
which symmetry sends to $0$.

```{.python .input #integral-density}
#@tab mxnet
x = np.arange(-8., 8., 1e-3)
p = np.exp(-x**2) / np.sqrt(np.array(np.pi))
print('total mass  integral p dx :', round(float(np.sum(1e-3 * p)), 6))
print('mean        integral x p dx:', round(float(np.sum(1e-3 * x * p)), 6))
```

```{.python .input #integral-density}
#@tab pytorch
x = torch.arange(-8., 8., 1e-3)
p = torch.exp(-x**2) / np.sqrt(np.pi)
print('total mass  integral p dx :', round(float(torch.sum(1e-3 * p)), 6))
print('mean        integral x p dx:', round(float(torch.sum(1e-3 * x * p)), 6))
```

```{.python .input #integral-density}
#@tab tensorflow
x = tf.range(-8., 8., 1e-3)
p = tf.exp(-x**2) / np.sqrt(np.pi)
print('total mass  integral p dx :', round(float(tf.reduce_sum(1e-3 * p)), 6))
print('mean        integral x p dx:', round(float(tf.reduce_sum(1e-3 * x * p)), 6))
```

```{.python .input #integral-density}
#@tab jax
x = jnp.arange(-8., 8., 1e-3)
p = jnp.exp(-x**2) / jnp.sqrt(jnp.pi)
print('total mass  integral p dx :', round(float(jnp.sum(1e-3 * p)), 6))
print('mean        integral x p dx:', round(float(jnp.sum(1e-3 * x * p)), 6))
```

When an expectation has no closed form---the rule rather than the exception in
deep learning---we estimate it by **Monte Carlo**: draw samples $x_i\sim p$ and
average,

$$
\mathbb{E}[g(X)] \approx \frac{1}{n}\sum_{i=1}^n g(x_i),
$$
:eqlabel:`eq_mdl-monte-carlo`

a stochastic counterpart of the Riemann sum that ignores the geometry of the
domain. The **law of large numbers** is the guarantee that it works: the sample
average converges to the true expectation, and the central limit theorem pins the
error at order $1/\sqrt{n}$---a rate that depends only on the sample size $n$, not
on the dimension. (Honesty about the fine print: the *constant* in front of
$1/\sqrt{n}$ is the standard deviation of $g(X)$, and *that* can grow with the
dimension; what is dimension-free is the exponent, and it is the exponent that
grids lose.) That dimension-free rate is decisive. A grid laid down to
resolution $\epsilon$ in $d$ dimensions costs $N=\epsilon^{-d}$ evaluations.
The left rule we have used throughout has error $\propto\epsilon$, so it buys
only $N^{-1/d}$; even a *second-order* rule---midpoint or trapezoid, whose
error is $\propto\epsilon^2$---manages just $N^{-2/d}$. Either way the
exponent is devoured by the dimension---the *curse of
dimensionality*---while Monte Carlo keeps its $N^{-1/2}$ in every dimension.
This is why sampling, not
grid quadrature, is how expectations are computed at scale; we develop it
thoroughly when we study random variables in :numref:`sec_mdl-random_variables`.
The cell makes both halves concrete: it estimates $\int_0^1 e^{-x^2}\,dx$ by
Monte Carlo against a midpoint quadrature fine enough to be exact at the
printed precision, then plots the error rates against the number of function
evaluations on log-log axes.

```{.python .input #integral-monte-carlo}
np.random.seed(1)
# E[e^{-X^2}], X ~ U[0, 1], by Monte Carlo vs a fine midpoint quadrature
# (midpoint is second-order: at eps=1e-4 its bias ~ 3e-10, below the print).
quad = float(np.sum(1e-4 * np.exp(-(np.arange(0., 1., 1e-4) + 5e-5) ** 2)))
for n in [10, 100, 1000, 10000, 100000]:
    mc = float(np.mean(np.exp(-np.random.rand(n) ** 2)))
    print(f'n={n:<7} Monte Carlo={mc:.5f}  error={abs(mc - quad):.1e}')
print(f'midpoint quadrature  ={quad:.5f}')

# Error vs work: a second-order grid over d dimensions decays as N^{-2/d}
# (the left rule manages only N^{-1/d}); Monte Carlo gives N^{-1/2} in
# *every* dimension (the law of large numbers).
N = np.logspace(1, 6, 50)
d2l.plot(N, [N ** -0.5, N ** (-2 / 2), N ** (-2 / 6), N ** (-2 / 10)],
         'function evaluations', 'integration error',
         xscale='log', yscale='log',
         legend=['Monte Carlo (any d)', 'grid d=2', 'grid d=6', 'grid d=10'])
```

The Monte-Carlo estimate closes in on the quadrature value as $n$
grows---unevenly, as randomness must: the $1/\sqrt{n}$ rate holds *in
expectation* (about $6\times 10^{-4}$ at $n=10^5$ for this integrand), and an
individual run can land lucky, as the final row here does. The log-log plot reads the rates off as slopes. The $d=2$
grid line falls at slope $-1$ and beats Monte Carlo's $-\tfrac12$; at $d=4$ a
second-order grid would tie exactly ($N^{-2/4}=N^{-1/2}$---we leave that line
out, since it would sit right on top of the Monte-Carlo line); by $d=6$ and
$d=10$ the grid has flattened to slopes $-\tfrac13$ and $-\tfrac15$ and loses
badly. For the first-order left rule the break-even is already $d=2$. Past a
handful of dimensions, sampling is the only practical choice---the quantitative
face of the curse of dimensionality.

**Differentiating under the integral sign.** One more interchange earns its
keep in deep learning: moving a gradient through an integral. When an integrand
depends on a parameter $\boldsymbol{\theta}$, we will routinely write

$$
\nabla_{\boldsymbol{\theta}} \int f(x, \boldsymbol{\theta})\,dx
= \int \nabla_{\boldsymbol{\theta}} f(x, \boldsymbol{\theta})\,dx .
$$

The swap is not free: the standard sufficient condition (via dominated
convergence) is that, near the current $\boldsymbol{\theta}$, the integrands'
derivatives are bounded in magnitude by a single integrable function of $x$.
The smooth, rapidly decaying densities of deep learning almost always comply,
but the caveat is not decorative---it fails exactly when the integrand's
support or its spikes move with $\boldsymbol{\theta}$. When it is the *limits*
that move, the correct derivative is known in closed form---the **Leibniz
rule**,

$$
\frac{d}{d\theta} \int_{a(\theta)}^{b(\theta)} f(x, \theta)\,dx
= \int_{a(\theta)}^{b(\theta)} \frac{\partial f}{\partial\theta}(x,\theta)\,dx
+ f\bigl(b(\theta), \theta\bigr)\,b'(\theta)
- f\bigl(a(\theta), \theta\bigr)\,a'(\theta) .
$$

The two boundary terms are precisely what the naive swap forgets: each moving
endpoint sweeps area in or out at the rate the limit moves, weighted by the
integrand's value there. A density supported on $[0, \theta]$ is the classic
trap---dropping the boundary term produces a confidently wrong gradient. Two
results we lean on
later are nothing but this swap: differentiating the normalization
$\int p_{\boldsymbol{\theta}}(x)\,dx = 1$ under the integral shows the score
$\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}$ has mean zero, the
seed of Fisher information (:numref:`sec_mdl-maximum_likelihood`); and applying
it to an expectation gives
$\nabla_{\boldsymbol{\theta}}\,\mathbb{E}_{x\sim p_{\boldsymbol{\theta}}}[g(x)]
= \mathbb{E}\bigl[g(x)\,\nabla_{\boldsymbol{\theta}}\log
p_{\boldsymbol{\theta}}(x)\bigr]$, the score-function (REINFORCE) identity at
the heart of policy-gradient methods.

## Summary

* The **definite integral** $\int_a^b f\,dx$ is the limit of Riemann rectangle
  sums :eqref:`eq_mdl-riemann`---the signed area under $f$, with negative
  contributions where $f<0$ or when the limits run backwards.
* The **fundamental theorem of calculus** :eqref:`eq_mdl-ftc` says the area-so-far
  function $F(x)=\int_a^x f$ has derivative $F'=f$. Integration is therefore
  differentiation reversed: $\int_a^b f = G(b)-G(a)$ for any antiderivative $G$.
* **Improper integrals** extend the definition to infinite domains as a limit;
  $\int_1^\infty x^{-p}\,dx$ converges exactly when $p>1$.
* **Integration by parts** :eqref:`eq_mdl-parts` trades $\int u\,v'$ for a
  boundary term minus $\int u'\,v$---the product rule run backwards, and the
  standard device for moving a derivative onto the factor that can absorb it
  (score matching, expectations of products).
* **Change of variables** multiplies by the local stretch: $du/dx$ in one
  dimension :eqref:`eq_mdl-change_var`, the Jacobian determinant
  $|\det D\boldsymbol{\phi}|$ for a $C^1$-diffeomorphism in many
  :eqref:`eq_mdl-change_var_nd`---the same volume-scaling read off the
  determinant. Read for densities (:numref:`sec_mdl-random_variables`) it drives
  normalizing flows.
* **Fubini's theorem** evaluates a multiple integral as iterated single integrals
  in either order, for absolutely integrable functions.
* **Integration is the language of continuous probability:** a density is a
  normalized non-negative function $\int p = 1$, and an expectation is the integral
  $\int x\,p\,dx$. When it has no closed form, Monte Carlo estimates it by
  sampling, converging at the dimension-free rate $1/\sqrt{n}$ of the law of large
  numbers---unlike a grid, whose cost explodes with dimension.

## Exercises

1. Evaluate $\int_1^2 \tfrac{1}{x}\,dx$ using an antiderivative, then confirm with
   a Riemann sum.
2. Use the change-of-variables formula to integrate
   $\int_0^{\sqrt{\pi}} x\sin(x^2)\,dx$.
3. Compute $\int_{[0,1]^2} xy\,dx\,dy$ by Fubini's theorem.
4. For which $p$ does $\int_1^\infty x^{-p}\,dx$ converge? Verify the boundary
   $p=1$ numerically by watching the partial integrals as the upper limit grows.
5. Find the constant $c$ that makes $c\,e^{-x^2}$ a probability density on
   $\mathbb{R}$, then compute its mean and $\mathbb{E}[X^2]$.
6. Let $f(x,y) = (x^2-y^2)/(x^2+y^2)^2$. Compute the iterated integrals
   $\int_0^1\!\bigl(\int_0^1 f\,dx\bigr)dy$ and $\int_0^1\!\bigl(\int_0^1
   f\,dy\bigr)dx$ over $[0,1]^2$ and confirm they equal $-\tfrac{\pi}{4}$ and
   $+\tfrac{\pi}{4}$. Why does this not contradict Fubini's theorem? (Hint:
   $\frac{\partial}{\partial x}\frac{-x}{x^2+y^2} = f(x,y)$; the singularity at the
   origin breaks absolute integrability.)
7. Estimate $\int_0^1 e^{-x^2}\,dx$ by Monte Carlo (average $e^{-x_i^2}$ over
   $x_i$ uniform on $[0,1]$) and compare with a Riemann sum.
8. Use integration by parts :eqref:`eq_mdl-parts` twice to evaluate
   $\int_0^\infty u^2 e^{-u}\,du$---the second moment behind the exponential
   distribution's variance in :numref:`sec_mdl-distributions`. The integral you
   have computed is $\Gamma(3)$, an instance of the **Gamma function**
   $\Gamma(t) = \int_0^\infty u^{t-1}e^{-u}\,du$; the same
   integration-by-parts step, run once at general $t$, gives the recursion
   $\Gamma(t+1) = t\,\Gamma(t)$ and hence $\Gamma(n+1) = n!$---the function that
   names the Gamma distribution of :numref:`sec_mdl-distributions`.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/414)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1092)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1093)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1093)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §23.4]{.kicker}

How much of something is there in total<br>**Riemann sums · the fundamental theorem · change of variables · probability**.
:::
:::

::: {.slide title="Why integration"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Differentiation answered a *local* question: how does $f$ change
when we nudge $x$? Integration answers a *global* one, how much is
there in total, and the fundamental theorem welds the two together.

- An **area** under a curve, a **volume** under a surface.
- Every continuous **probability** is an integral, $\int p = 1$.
- Every **expectation** is an integral average, $\int x\,p(x)\,dx$.
- The **change-of-variables** rule powers normalizing flows.
:::

::: {.col .fig}
![](../img/mdl-cal-riemann.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The integral and the theorem]{.dtitle}

[a limit of sums, and how to compute it]{.dsub}
:::
:::

::: {.slide title="The definite integral as a limit"}
[The integral]{.kicker}

::: {.cols .vc}
::: {.col}
Archimedes' recipe: chop $[a,b]$ into $N$ slices of width
$\epsilon$, stand a rectangle of height $f(x_i)$ on each, and add.
As the slices shrink, the staircase squeezes onto the curve:

$$\int_a^b f(x)\,dx = \lim_{\epsilon\to 0}\ \sum_i \epsilon\, f(x_i).$$

The continuous analogue of a sum: $\sum \to \int$, spacing
$\epsilon \to dx$.
:::

::: {.col .fig .big}
![](../img/mdl-cal-riemann.svg)
:::
:::
:::

::: {.slide title="Watching the limit converge"}
[The integral]{.kicker}

The definition is honest but slow. Refine the partition for
$\int_0^2 \tfrac{x}{1+x^2}\,dx = \tfrac12\log 5$ and the left-rule
sum marches toward the truth, the error shrinking in step with
$\epsilon$:

@!integral-riemann-converge

. . .

First-order convergence: cut $\epsilon$ by ten, cut the error by
ten. It works, but it is slow and gives no closed form. We need a
better idea.
:::

::: {.slide title="The fundamental theorem of calculus"}
[The integral]{.kicker}

::: {.cols .vc}
::: {.col}
Let the upper limit *move*: the **area-so-far** function
$F(x)=\int_a^x f$ accumulates signed area out to $x$. Nudge $x$ by
$\epsilon$ and only one thin sliver of new area appears, of height
$f(x)$, so

$$\frac{dF}{dx}(x) = f(x).$$

The rate at which area grows *is* the current height. Finding areas
becomes the search for an **antiderivative**.
:::

::: {.col .fig}
![](../img/mdl-cal-sub-area.svg)
:::
:::
:::

::: {.slide title="Integration is differentiation reversed"}
[The integral]{.kicker}

If $G'=f$ then every definite integral is a difference of two values:

$$\int_a^b f(x)\,dx = G(b) - G(a).$$

Hard sums become table lookups. We check it numerically: build $F$
as a cumulative Riemann sum, finite-difference it, and compare
against $f$ itself.

@!integral-ftc-check

::: {.d2l-note}
The error is float32 roundoff, not discretization: differencing the
cumulative sum *telescopes* back to the term it just added.
:::
:::

::: {.slide title="Improper integrals"}
[The integral]{.kicker}

Densities live on unbounded domains, so we integrate to infinity as
a limit, $\int_a^\infty f = \lim_{b\to\infty}\int_a^b f$. The power
law is the cleanest test:

$$\int_1^\infty x^{-p}\,dx = \begin{cases}\dfrac{1}{p-1}, & p>1\ \text{(converges)},\\[1ex]\infty, & p\le 1\ \text{(diverges)}.\end{cases}$$

::: {.d2l-note .rule}
The threshold $p=1$ decides whether a heavy-tailed density has a
finite normalizer or mean at all.
:::
:::

::: {.slide title="Two errors, two knobs"}
[The integral]{.kicker}

Watch a convergent improper integral, $\int_0^\infty e^{-x}\,dx=1$,
through two lenses: the exact partials $1-e^{-b}$ and a left-Riemann
sum of each:

@!integral-improper

. . .

The partials race to $1$ as $b$ grows (*truncation* error
$e^{-b}$); the left rule settles near $1.0005$, not $1$
(*discretization* error $\approx\tfrac{\epsilon}{2}$). Two
independent knobs: neither fixes the other.
:::

::: {.slide title="Integration by parts"}
[The integral]{.kicker}

The product rule, run backwards through the theorem:

$$\int_a^b u\,v'\,dx = \bigl[\,u\,v\,\bigr]_a^b - \int_a^b u'\,v\,dx.$$

A *trade*, not an evaluation: move the derivative onto the factor
that can absorb it. With $u=x$, $v'=e^{-x}$,

$$\int_0^\infty x\,e^{-x}\,dx = \bigl[\,-x\,e^{-x}\,\bigr]_0^\infty + \int_0^\infty e^{-x}\,dx = 1,$$

the mean of the exponential, and the move behind score matching's
Hyvärinen identity.
:::

::: {.slide title="Area comes with a sign"}
[The integral]{.kicker}

::: {.cols .vc}
::: {.col}
Two independent sign rules govern $\int_a^b f$:

- where $f<0$, area counts **negative**: $\int_0^1 (-1)\,dx = -1$
- reversing the limits negates: $\int_b^a f = -\int_a^b f$

Each flip contributes one minus sign, and two cancel:
$\int_0^{-1}(-1)\,dx = +1$.

::: {.d2l-note}
The same signed bookkeeping the determinant did for transformed areas,
and exactly what makes change of variables come out right.
:::
:::

::: {.col .fig}
![Lobes above the axis count positive, the lobe below counts negative; reversing the limits traverses the same area backwards and negates the total.](../img/mdl-cal-signed-area.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Change of variables]{.dtitle}

[the chain rule, run backwards]{.dsub}
:::
:::

::: {.slide title="Substitution in one dimension"}
[Change of variables]{.kicker}

::: {.cols .vc}
::: {.col}
Reparametrize through $y=u(x)$. A sliver of width $\epsilon$ at $x$
maps to one of width $\epsilon\,u'(x)$, so matching areas forces the
local **stretch** factor:

$$\int_{u(a)}^{u(b)} f(y)\,dy = \int_a^b f(u(x))\,\frac{du}{dx}\,dx.$$

With the right $u$ this collapses hard integrals to trivial ones.
:::

::: {.col .fig}
![](../img/mdl-cal-rect-trans.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Multiple integrals]{.dtitle}

[double integrals, Fubini, the Jacobian, and the Gaussian]{.dsub}
:::
:::

::: {.slide title="Multiple integrals: a volume under a surface"}
[Double integrals]{.kicker}

::: {.cols .vc}
::: {.col}
For $f(x,y)$ on a box, $\int_U f\,d\mathbf{x}$ is the volume between
the surface and the base plane. Archimedes' recipe survives: tile
the base with $\epsilon\times\epsilon$ squares, stand a box on each,
and total the volumes.

Our running example is the bell $e^{-x^2-y^2}$ over $[-2,2]^2$.
:::

::: {.col .fig .big}
@fig:mdl-cal-bell-surface
:::
:::
:::

::: {.slide title="The box volume, numerically"}
[Double integrals]{.kicker}

Total the boxes on a fine grid over $[-2,2]^2$. The exact value
hides the **error function** $\operatorname{erf}$, the Gaussian's
antiderivative, which has no elementary formula:

@!integral-box-volume

::: {.d2l-note}
Over the whole plane this volume becomes exactly $\pi$. Both facts
fall out of the machinery below.
:::
:::

::: {.slide title="Fubini: sum in any order"}
[Fubini]{.kicker}

::: {.cols .vc}
::: {.col}
A grid sum totals in **any order**. Summing columns-first and
passing to the limit splits the double integral into iterated
single ones:

$$\int_U f\,dx\,dy = \int\!\!\left(\int f\,dx\right)dy = \int\!\!\left(\int f\,dy\right)dx.$$

(It needs $f$ absolutely integrable, the rule in machine learning.)
:::

::: {.col .fig}
![](../img/mdl-cal-sum-order.svg)
:::
:::
:::

::: {.slide title="Change of variables in many dimensions"}
[Change of variables]{.kicker}

::: {.cols .vc}
::: {.col}
For a $C^1$-diffeomorphism $\boldsymbol{\phi}$, the scalar stretch
$\tfrac{du}{dx}$ becomes the **Jacobian determinant**:

$$\int_{\boldsymbol{\phi}(U)}\! f\,d\mathbf{x} = \int_{U}\! f(\boldsymbol{\phi}(\mathbf{x}))\,\bigl|\det D\boldsymbol{\phi}\bigr|\,d\mathbf{x}.$$

The determinant is exactly the factor by which a linear map scales
volume, so a tiny cube of volume $d\mathbf{x}$ becomes a
parallelepiped of volume $|\det D\boldsymbol{\phi}|\,d\mathbf{x}$.
:::

::: {.col .fig}
![](../img/mdl-cal-cov-jacobian.svg)
:::
:::

::: {.d2l-note}
Read for *densities* instead of areas, this same factor (as
$-\log|\det D\boldsymbol{\phi}|$) is the engine behind
**normalizing flows**.
:::
:::

::: {.slide title="The Gaussian integral"}
[The payoff]{.kicker}

The one-dimensional $\int_{-\infty}^\infty e^{-x^2}\,dx$ has no
elementary antiderivative. Go *up* a dimension: square it and let
Fubini fuse the copies, then polar coordinates make it elementary,

$$\left(\int e^{-x^2}\,dx\right)^{\!2} = \iint e^{-x^2-y^2}\,dx\,dy = \int_0^\infty\!\!\int_0^{2\pi}\! r\,e^{-r^2}\,d\theta\,dr = \pi.$$

The polar map contributes Jacobian $|\det D\boldsymbol{\phi}|=r$,
the factor that cracks it open. So $\int e^{-x^2}\,dx=\sqrt\pi$:

@!integral-gaussian
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Integration meets probability]{.dtitle}

[densities, expectations, and Monte Carlo]{.dsub}
:::
:::

::: {.slide title="Densities and expectations"}
[Probability]{.kicker}

This is why a deep-learning reader needs integration. A continuous
**density** is a non-negative function that is normalized; an
**expectation** is an integral average:

$$\int_{\mathcal X} p = 1, \qquad \mathbb{E}[g(X)] = \int_{\mathcal X} g(x)\,p(x)\,dx.$$

The Gaussian integral supplies the normalizer: $p(x)=\tfrac{1}{\sqrt\pi}e^{-x^2}$
integrates to $1$, and symmetry sends its mean to $0$.

@!integral-density
:::

::: {.slide title="Monte Carlo beats the curse of dimensionality"}
[Probability]{.kicker}

::: {.cols .vc}
::: {.col}
With no closed form, **Monte Carlo** estimates an expectation by
sampling, $\mathbb{E}[g(X)] \approx \tfrac1n\sum_i g(x_i)$, at rate
$1/\sqrt n$ *in every dimension* (the constant, the standard
deviation of $g(X)$, may grow with $d$; the *exponent* never does).

A grid to resolution $\epsilon$ in $d$ dimensions costs
$N=\epsilon^{-d}$ points and decays only as $N^{-2/d}$: the exponent
is devoured by $d$. Past a handful of dimensions, sampling is the
only practical choice.
:::

::: {.col .fig .big}
@!integral-monte-carlo
:::
:::
:::

::: {.slide title="Differentiating under the integral"}
[Probability]{.kicker}

Training through an expectation means swapping $\nabla_{\boldsymbol\theta}$ and
$\int$; the plain swap is what shows the score $\nabla_{\boldsymbol\theta}\log
p_{\boldsymbol\theta}$ has mean zero, the identity at the heart of policy
gradients. When the *limits* move too, the **Leibniz rule** keeps the books:

$$\frac{d}{d\theta}\!\int_{a(\theta)}^{b(\theta)}\! f(x,\theta)\,dx
= \int_{a(\theta)}^{b(\theta)}\!\frac{\partial f}{\partial\theta}\,dx
+ f(b(\theta),\theta)\,b'(\theta) - f(a(\theta),\theta)\,a'(\theta).$$

. . .

Each moving endpoint sweeps area in or out at the rate it moves, weighted by the
integrand's value there.

::: {.d2l-note .warn}
A density supported on $[0,\theta]$ is the classic trap: drop the boundary term
and you get a confidently wrong gradient.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Integral** = signed area or volume, a limit of Riemann sums.
- **Fundamental theorem**: $F'=f$, so integration is the inverse of
  differentiation, $\int_a^b f = G(b)-G(a)$.
- **By parts**: $\int u\,v' = [\,uv\,] - \int u'\,v$ moves a
  derivative onto the factor that can absorb it.
:::

::: {.col}
- **Change of variables** scales by $du/dx$ in 1-D, the Jacobian
  $|\det D\boldsymbol{\phi}|$ in $n$-D, the engine of normalizing flows.
- **Fubini** evaluates a multiple integral as iterated single ones.
- **Probability**: a density is $\int p = 1$; an expectation is an
  integral, estimated by Monte Carlo when it has no closed form.
:::
:::
:::
