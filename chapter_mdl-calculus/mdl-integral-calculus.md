# Integral Calculus
:label:`sec_mdl-integral_calculus`

The main text used continuous random variables without asking how their
probabilities are computed. A normal distribution was described by its
density (:numref:`subsec_normal_distribution_and_squared_loss`), an expected
loss was an average over that density, and both were taken as given. This
section asks how we add up a quantity that is spread continuously over an
interval.

The standard normal distribution, the bell curve, shows why the question
needs care. A standard normal random variable $Z$ has the density
$p(z)=e^{-z^2/2}/\sqrt{2\pi}$. Its value at $z=0$ is $0.40$, yet $P(Z=0)=0$,
and the same holds at every other single point. Intervals do carry
probability: $P(-1\le Z\le 1)\approx 0.68$. That number is the area under
$p$ between $-1$ and $1$, and the constant $1/\sqrt{2\pi}$ is the one that
makes the total area under $p$ equal to $1$. Where does the constant come
from, and how is $0.68$ computed?

The question is not special to the bell curve. Every probability and every
expected loss over a continuous variable is an area of this kind, and
training a model through random sampling, as variational autoencoders and
policy-gradient methods do, needs the derivative of such an area with respect
to a parameter.

The plan is to cut the area into thin slices, add them, and let the slices
shrink; the limit is the integral. The fundamental theorem of calculus
replaces the adding by the reverse of differentiation. That settles most
integrals but not the bell curve, whose constant $\sqrt{2\pi}$ we find by a
detour through two dimensions. Three results carry the section: the
fundamental theorem :eqref:`eq_mdl-ftc`, the change of variables
:eqref:`eq_mdl-change_var_nd`, and the Gaussian integral
:eqref:`eq_mdl-gaussian-integral`. A reader with first-year calculus at hand
can skim as far as :numref:`subsec_mdl-improper-integrals`. The derivative of
an area with respect to a parameter needs all of these tools and has a
section of its own, :numref:`sec_mdl-expectation-gradients`. Measure theory
and the Lebesgue integral are left out;
:numref:`sec_mdl-calculus-further-reading` points to them.

```{.python .input #integral-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #integral-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #integral-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #integral-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Definite Integral

### Riemann Sums

A density gives probability per unit length, so over a short interval of
width $\epsilon$ the probability is about density times width. Over a long
interval we cut it into thin slices, form density times width on each, and
add. The recipe works for any function.

We start with a case that can be done by hand. Take $f(x)=x^2$ on $[0,1]$
and cut the interval into $N$ slices of width $\epsilon=1/N$. The $i$-th
slice is $[i/N,(i+1)/N]$; give it the height of $f$ at its left end,
$(i/N)^2$. The rectangles add up to

$$
\sum_{i=0}^{N-1}\frac1N\Bigl(\frac iN\Bigr)^2
=\frac{1}{N^3}\sum_{i=0}^{N-1}i^2
=\frac{(N-1)N(2N-1)}{6N^3}
=\frac13-\frac1{2N}+\frac1{6N^2},
$$

where the middle step uses the classical formula
$\sum_{i=0}^{N-1}i^2=(N-1)N(2N-1)/6$. The sums have the limit $\tfrac13$, and
they miss it by about $\tfrac1{2N}$, an error of first order in the slice
width $\epsilon$.

The function `left_sum` implements this **left-endpoint rule** for any $f$:
cut $[a,b]$ into slices of width $\epsilon$ and tag each slice at its left
end, as in :numref:`fig_mdl-riemann`. We use it for every grid sum of the
section, and try it first on $x^2$.

```{.python .input #mdl-integral-calculus-riemann-sums-1}
def left_sum(f, a, b, eps):
    """Left-endpoint Riemann sum of f on [a, b] with slices of width eps."""
    x = a + eps * np.arange(round((b - a) / eps))   # left ends of the slices
    return float(np.sum(eps * f(x)))

for N in [10, 100, 1000]:
    print(f'N={N:<5} left sum={left_sum(lambda x: x**2, 0, 1, 1 / N):.6f}  '
          f'formula={1/3 - 1/(2*N) + 1/(6*N**2):.6f}')
```

The sums agree with the closed form to all printed digits.

**The definition.** Nothing in the recipe needed equal widths or left ends. A
*partition* $P:a=x_0<x_1<\cdots<x_N=b$ cuts $[a,b]$ into slices, a *tag*
$\xi_i\in[x_{i-1},x_i]$ picks a point in each slice, and the *mesh*
$\|P\|=\max_i(x_i-x_{i-1})$ is the width of the widest slice. A bounded
function $f$ is **Riemann integrable** on $[a,b]$ if there is a number $I$ that
every tagged sum approaches as the mesh tends to zero, whatever the tags:

$$
\int_a^b f(x)\,dx = I = \lim_{\|P\|\to0}\sum_{i=1}^N f(\xi_i)(x_i-x_{i-1}).
$$
:eqlabel:`eq_mdl-riemann`

The integral is the value that the slice sums approach as the slices become
thin. The computation above gives
$\int_0^1x^2\,dx=\tfrac13$. Every continuous function on $[a,b]$ is Riemann
integrable, and so is every bounded function with finitely many jumps
:cite:`Rudin.1976`. Some bounded functions, such as the indicator of the
rationals, have no Riemann integral; the Lebesgue integral of
:numref:`sec_mdl-calculus-further-reading` repairs this, and it agrees with
the Riemann integral whenever the latter exists.

Two properties follow from the definition, and the proofs below use both. The
integral is **monotone**: if $f\le g$ on $[a,b]$, then every tagged sum of $f$
is at most the same tagged sum of $g$, and limits preserve $\le$, so
$\int_a^b f\,dx\le\int_a^b g\,dx$. It is **additive over subintervals**:
$\int_a^c f\,dx=\int_a^b f\,dx+\int_b^c f\,dx$ for $a\le b\le c$, because
partitions that include the point $b$ split into a partition of $[a,b]$ and
one of $[b,c]$.

**The running example.** The integral

$$
\int_0^2\frac{x}{1+x^2}\,dx=\tfrac12\log5=0.80472
$$

is our running example as far as the Gaussian integral (the evaluation rule
:eqref:`eq_mdl-ftc-eval` will show why its value is $\tfrac12\log5$).
:numref:`fig_mdl-riemann` shows left-endpoint rectangles under this curve for
three slice widths.

![Left-endpoint rectangles under $f(x)=x/(1+x^2)$ on $[0,2]$ for slice widths $\epsilon=0.5$, $0.2$ and $0.05$. Each slice takes the height of $f$ at its left end; the rectangle sums approach the area $\tfrac12\log 5$ as $\epsilon\to0$.](../img/mdl-cal-riemann.svg)
:label:`fig_mdl-riemann`

How fast do the sums approach $\tfrac12\log5$? For $x^2$ the error was
$\tfrac1{2N}=\tfrac\epsilon2$. If the first order is typical, slices ten
times thinner should give an error ten times smaller, and the table prints
the error beside $0.2\,\epsilon$ for comparison.

```{.python .input #integral-riemann-converge}
f = lambda x: x / (1 + x**2)
truth = math.log(5) / 2
for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
    s = left_sum(f, 0, 2, eps)
    print(f'eps={eps:.0e}  sum={s:.6f}  truth-sum={truth - s:.2e}  '
          f'0.2*eps={0.2 * eps:.0e}')
print(f'truth = (1/2) log 5 = {truth:.6f}')
```

Each time $\epsilon$ shrinks by ten, so does the error, and every sum lies
below the truth. The constant $0.2$ is half of $f(2)-f(0)=0.4$. This is no
accident, and we derive it once the fundamental theorem is available.

### Signed Area

Rectangles below the axis have negative height $f(\xi_i)<0$ and enter the
sum with a minus sign, so the integral measures *signed* area: where $f<0$
the area counts as negative, as in $\int_0^1(-1)\,dx=-1$. Integrating from
right to left negates the result,
$\int_b^a f\,dx=-\int_a^b f\,dx$; this is the convention that keeps
additivity true for every order of $a$, $b$, $c$.
:numref:`fig_mdl-signed-area` shows both rules on one curve.

![Signed area. Lobes above the axis count positive and the lobe below counts negative, so $\int_a^b f$ totals the shaded regions with their signs; reversing the limits traverses the same area right to left and negates the total.](../img/mdl-cal-signed-area.svg)
:label:`fig_mdl-signed-area`

## The Fundamental Theorem

### The Theorem

Adding slices gave $\tfrac13$ only because $\sum i^2$ has a closed form. For
the running example the left sum is
$\sum_i\epsilon\cdot i\epsilon/(1+i^2\epsilon^2)$, and no formula collapses it.
We need a method that does not add slices at all.

The method comes from letting the upper limit vary. Fix a base point $a$ and
define the **area-so-far function**

$$
F(x) = \int_a^x f(y)\,dy,
$$
:eqlabel:`eq_mdl-area-fn`

the signed area accumulated from $a$ out to $x$. One function holds every
definite integral of $f$: by additivity, the area over any interval
$[x_1,x_2]$ is a difference of two accumulated areas,

$$
\int_{x_1}^{x_2} f(x)\,dx = F(x_2) - F(x_1),
$$

as in :numref:`fig_mdl-area-subtract`. So it suffices to find $F$.

![The area under $f$ from $x_1$ to $x_2$ as the difference of two areas accumulated from the base point $a$ at the plot's left edge: $\int_{x_1}^{x_2} f\,dx=\int_a^{x_2} f\,dx-\int_a^{x_1} f\,dx$.](../img/mdl-cal-sub-area.svg)
:label:`fig_mdl-area-subtract`

How does $F$ change when $x$ moves a little to the right?
:numref:`fig_mdl-ftc-sliver` shows the area out to $x$ and the strip of width
$\epsilon$ that moving to $x+\epsilon$ adds. The strip is nearly a rectangle of
height $f(x)$, so $F(x+\epsilon)-F(x)\approx\epsilon f(x)$, and dividing by
$\epsilon$ suggests $F'(x)=f(x)$.

![The area-so-far $F(x)$ under $f(y)=y/(1+y^2)$ from $a=0$ to $x=1.2$ (light), and the strip that moving to $x+\epsilon$ adds (dark). The strip is close to the dashed rectangle of height $f(x)$, so $F(x+\epsilon)-F(x)\approx\epsilon f(x)$.](../img/mdl-cal-ftc-sliver.svg)
:label:`fig_mdl-ftc-sliver`

**Theorem (Fundamental theorem of calculus).** *Let $f$ be Riemann integrable
on $[a,b]$, let $F(x)=\int_a^x f(y)\,dy$, and let $f$ be continuous at a point
$x\in(a,b)$. Then $F$ is differentiable at $x$ and*

$$
\frac{dF}{dx}(x) = f(x).
$$
:eqlabel:`eq_mdl-ftc`

*The rate at which area accumulates is the current height of the curve.*

**Proof.** Nudge $x$ by $\epsilon>0$ and measure the new area. By
:eqref:`eq_mdl-area-fn` and additivity, the increment is a single sliver,

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y)\,dy .
$$

Write $m_\epsilon = \inf_{[x,x+\epsilon]} f$ and
$M_\epsilon = \sup_{[x,x+\epsilon]} f$ for the tightest constant bounds on $f$
over the sliver; they are finite because an integrable function is bounded.
Since $m_\epsilon\le f\le M_\epsilon$ on the sliver, monotonicity of the
integral pins its area between two rectangles of width $\epsilon$:

$$
\epsilon\, m_\epsilon
\ \le\ F(x+\epsilon) - F(x)\ \le\
\epsilon\, M_\epsilon .
$$

Divide by $\epsilon$. Because $f$ is continuous at $x$, for any $\delta>0$ the
whole interval $[x,x+\epsilon]$ eventually lies where $|f(y)-f(x)|<\delta$.
Hence $m_\epsilon$ and $M_\epsilon$ both converge to $f(x)$ as
$\epsilon \to 0$, and the squeeze forces
$\lim_{\epsilon\to 0}\frac{F(x+\epsilon)-F(x)}{\epsilon}=f(x)$. For
$\epsilon<0$ the sliver sits to the left of $x$, and the same bounds apply with
both sign flips canceling in the quotient. That limit is $F'(x)$.
$\blacksquare$

The proof needed only monotonicity and additivity, the two properties proved
from the definition. The theorem turns the search for an area into the search
for an **antiderivative**, a function $G$ with $G'=f$, which we can read off
the derivative table of :numref:`sec_mdl-derivative_table`. Any antiderivative
gives the integral, through the **evaluation rule**: *if $f$ is continuous on
$[a,b]$ and $G'=f$ on all of $[a,b]$, then*

$$
\int_a^b f(x)\,dx = G(b) - G(a).
$$
:eqlabel:`eq_mdl-ftc-eval`

The reason is that $F$ and $G$ have the same derivative $f$ on $(a,b)$, so
$G-F$ has derivative zero there and is constant by the mean value theorem of
:numref:`sec_mdl-mvt`; the constant cancels in $G(b)-G(a)=F(b)-F(a)=\int_a^b
f\,dx$.

The hypothesis "on all of $[a,b]$" matters. Since $(-1/x)'=x^{-2}$, a careless
use gives $\int_{-1}^1x^{-2}\,dx=[-1/x]_{-1}^1=-2$, a negative value for a
positive integrand. The rule does not apply, because $G'=f$ fails at $0$, where
neither function is defined; the integral diverges
(:numref:`subsec_mdl-improper-integrals`).

Each derivative read backwards is an integral: $(y^3)'=3y^2$ gives
$\int_0^1y^2\,dy=\tfrac13$ without a single slice, and $(e^t)'=e^t$ gives
$\int_0^xe^t\,dt=e^x-1$. For the running example take
$G(x)=\tfrac12\log(1+x^2)$. By the chain rule
$G'(x)=\tfrac12\cdot\frac{2x}{1+x^2}=\frac{x}{1+x^2}$, so

$$
\int_0^2\frac{x}{1+x^2}\,dx=G(2)-G(0)=\tfrac12\log5-\tfrac12\log1=\tfrac12\log5,
$$

the value the Riemann sums were approaching.

**The error of the left rule.** The evaluation rule also explains the
constant $0.2$ in the error of those sums. On the slice that starts at $x_i$,
$f(x)\approx f(x_i)+f'(x_i)(x-x_i)$, so the area of the slice exceeds its
rectangle $\epsilon f(x_i)$ by about $\tfrac{\epsilon^2}2f'(x_i)$. Summed over
the slices, these excesses form $\tfrac\epsilon2$ times a left sum of $f'$,
which is close to $\tfrac\epsilon2\int_a^bf'\,dx=\tfrac\epsilon2(f(b)-f(a))$.
To first order in $\epsilon$, then,

$$
\text{left sum}-\int_a^b f(x)\,dx\approx\frac\epsilon2\bigl(f(a)-f(b)\bigr).
$$

For $x^2$ on $[0,1]$ this is $-\tfrac\epsilon2=-\tfrac1{2N}$, the leading
correction found by hand. For the running example, $f(0)=0$ and $f(2)=0.4$
give $-0.2\,\epsilon$. The formula is numerical analysis, not theory, but it
accounts for the residue in three more outputs of this section, and it is the
yardstick against which sampling is measured in
:numref:`subsec_mdl-monte-carlo`.

The theorem says more than $\int_0^2f=G(2)-G(0)$: the area-so-far equals
$G(x)-G(0)$ at every $x$. To test this we build $F$ as a cumulative left sum,
one running total over the slices, and compare it with $G(x)-G(0)$ at every
grid point. The two cannot agree exactly, because the left sum on $[0,x]$
carries the bias $\tfrac\epsilon2\bigl(f(0)-f(x)\bigr)$. It is largest where
$f$ peaks, at $x=1$ with $f(1)=\tfrac12$, so the largest gap should be
$\tfrac\epsilon4$.

```{.python .input #integral-ftc-check}
G = lambda x: np.log(1 + x**2) / 2             # antiderivative of f
for eps in [1e-2, 1e-3, 1e-4]:
    x = eps * np.arange(round(2 / eps))        # left ends of the slices of [0, 2]
    F = np.cumsum(eps * f(x))                  # F[k]: left sum on [0, x_k + eps]
    gap = np.abs(F - (G(x + eps) - G(0))).max()
    print(f'eps={eps:.0e}  max over x of |F(x) - (G(x) - G(0))| = {gap:.2e}')
```

The largest gap is $\tfrac\epsilon4$ at each slice width, as predicted. The
cumulative sum follows the antiderivative at every $x$, not only at $x=2$.

### Improper Integrals
:label:`subsec_mdl-improper-integrals`

The normal, exponential and many other densities live on unbounded domains,
while :eqref:`eq_mdl-riemann` defines integrals over a finite $[a,b]$ only. We
extend the definition the way we made it, by a limit. The **improper
integral** is

$$
\int_a^\infty f(x)\,dx = \lim_{b\to\infty}\int_a^b f(x)\,dx ,
$$
:eqlabel:`eq_mdl-improper`

the area out to $b$, in the limit of large $b$. If the limit exists and is
finite the integral *converges*; otherwise it *diverges*. The integral over the
whole line, $\int_{-\infty}^\infty f\,dx$, is split at any point into two
one-sided improper integrals, and *both* must converge on their own. A
symmetric limit is not enough: $\int_{-b}^bx\,dx=0$ for every $b$, while
$\int_0^\infty x\,dx=\infty$, so $\int_{-\infty}^\infty x\,dx$ does not exist.

The power law $x^{-s}$ with exponent $s>0$ is the standard test of whether a
tail is thin enough to have finite area. The evaluation rule gives the definite
integral over $[1,b]$, and then the limit:

$$
\int_1^b x^{-s}\,dx = \frac{b^{1-s}-1}{1-s}
\ \xrightarrow{\ b\to\infty\ }\
\begin{cases} \dfrac{1}{s-1}, & s>1\ \text{(converges)},\\[1ex] \infty, & s\le 1\ \text{(diverges)}.\end{cases}
$$

(At $s=1$ the formula reads $0/0$; the definite integral is $\log b$, its limit
as $s\to1$, and $\log b\to\infty$.) So $\int_1^\infty x^{-2}\,dx=1$ while
$\int_1^\infty x^{-1}\,dx=\infty$: the boundary between finite and infinite
area sits at $s=1$.

Power laws are the standard example of a **heavy-tailed** distribution
(defined in :numref:`sec_mdl-random_variables`, which also shows what such
tails do to averages). For a density with tail $p(x)\propto x^{-s}$ the
normalizer is finite only if $s>1$, and the mean, by the same test applied to
$x\cdot x^{-s}=x^{1-s}$, only if $s>2$.

The same idea handles an integrand that blows up at an endpoint:
$\int_0^1 f\,dx=\lim_{a\to0^+}\int_a^1 f\,dx$. For the power law the test
reverses,

$$
\int_0^1x^{-s}\,dx=\lim_{a\to0^+}\frac{1-a^{1-s}}{1-s}=\frac1{1-s}\quad\text{for } s<1,
$$

and the integral diverges for $s\ge1$. At $s=\tfrac12$ the spike $x^{-1/2}$ is
infinite at $0$, yet its area is $2$. Near infinity a power must fall faster
than $1/x$; near zero it must grow slower.

An improper integral computed numerically has two errors: we stop at a finite
$b$, and we add slices of finite width. For $\int_0^\infty e^{-x}\,dx = 1$
the two can be told apart. The table prints the exact partial integrals
$\int_0^b e^{-x}\,dx = 1-e^{-b}$, read off the antiderivative $-e^{-x}$, beside
a left sum of each with $\epsilon = 10^{-3}$.

```{.python .input #integral-improper}
for b in [1, 2, 5, 10, 20]:
    left = left_sum(lambda x: np.exp(-x), 0, b, 1e-3)
    print(f'b={b:<3}  left sum={left:.6f}  exact 1-e^-b={1 - math.exp(-b):.6f}')
```

The exact partials approach $1$ quickly: the *truncation error* of stopping at
$b$ is $e^{-b}$, below the sixth decimal by $b=20$. The left sums settle near
$1.0005$, not $1$. That residue is *discretization error*, the left rule's bias
$\tfrac{\epsilon}{2}\bigl(f(0)-f(b)\bigr)\to\tfrac{\epsilon}{2}= 0.0005$. To
leading order the two errors add: a larger $b$ removes the first, a smaller
$\epsilon$ the second.

### Integration by Parts
:label:`subsec_mdl-integration-by-parts`

Every differentiation rule, read through the evaluation rule, becomes an
integration rule. The product rule becomes integration by parts.

For continuously differentiable $u$ and $v$ on $[a,b]$,

$$
\int_a^b u(x)\,v'(x)\,dx
= \Bigl[\,u(x)\,v(x)\,\Bigr]_a^b - \int_a^b u'(x)\,v(x)\,dx ,
$$
:eqlabel:`eq_mdl-parts`

where $[\,h(x)\,]_a^b$ abbreviates $h(b)-h(a)$. The proof is one line: by the
product rule $uv$ is an antiderivative of $u'v + uv'$, so
:eqref:`eq_mdl-ftc-eval` gives
$\int_a^b (u'v + uv')\,dx = [\,u(x)v(x)\,]_a^b$, and we subtract
$\int_a^b u'v\,dx$ from both sides.

The formula moves the derivative from one factor to the other and adds a
boundary term. It helps when the new integrand $u'v$ is simpler than the old
one $uv'$. With $u(x)=x$ and $v'(x)=e^{-x}$, so that $u'=1$ and $v=-e^{-x}$,
apply :eqref:`eq_mdl-parts` on $[0,b]$ and let $b\to\infty$:

$$
\int_0^\infty x\,e^{-x}\,dx
= \Bigl[\,-x\,e^{-x}\,\Bigr]_0^\infty + \int_0^\infty e^{-x}\,dx
= 0 + 1 = 1 .
$$

The boundary term vanishes because $x e^{-x}\to 0$ at both ends, and what
remains is the improper integral computed above. The value is the mean of the
exponential distribution (:numref:`sec_mdl-distributions`).

Expectations are integrals of products, and a common move is to shift a
derivative from a factor we cannot handle onto one we can. The bell curve
shows the move at its simplest. Take the bell weight $w(z)=e^{-z^2/2}$, which
satisfies $w'(z)=-z\,w(z)$. Let $g$ be continuously differentiable with
$g(z)w(z)\to0$ as $z\to\pm\infty$, and let the two integrals below converge.
Integration by parts on $[-a,b]$, followed by the separate limits
$a\to\infty$ and $b\to\infty$, gives

$$
\int_{-\infty}^{\infty}g'(z)\,w(z)\,dz
=\Bigl[\,g(z)\,w(z)\,\Bigr]_{-\infty}^{\infty}-\int_{-\infty}^{\infty}g(z)\,w'(z)\,dz
=\int_{-\infty}^{\infty}z\,g(z)\,w(z)\,dz .
$$
:eqlabel:`eq_mdl-stein-1d`

Under the bell weight, multiplying by $z$ has the same effect as
differentiating. Both conditions hold for every polynomial $g$. With
$g(z)=z$ the identity reads $\int z^2w\,dz=\int w\,dz$: the mean of $z^2$ under
the bell weight is $1$, whatever the normalizer $\int w\,dz$ turns out to be
(it is $\sqrt{2\pi}$, by :eqref:`eq_mdl-normal-normalizer`). With $g(z)=z^3$
it reads $\int z^4w=3\int z^2w=3\int w$, so the mean of $z^4$ is $3$. Once $w$
is normalized into the standard normal density $p=w/\sqrt{2\pi}$, the
identity reads $\mathbb{E}[g'(Z)]=\mathbb{E}[Z\,g(Z)]$, Stein's identity for
the standard normal. Shifting a derivative from an unknown factor onto a
known one is also the step that makes score matching work
(:numref:`sec_mdl-score-matching-diffusion-flow`, :cite:`Hyvarinen.2005`).

### Substitution

The chain rule, read through the evaluation rule, becomes the substitution
rule. Let $u$ be continuously differentiable on $[a,b]$ and $f$ continuous on
an interval containing $u([a,b])$, with antiderivative $G$. The chain rule
gives $(G\circ u)'(x)=f(u(x))\,u'(x)$, so

$$
\int_a^b f(u(x))\,u'(x)\,dx=\int_a^b (G\circ u)'(x)\,dx=G(u(b))-G(u(a))=\int_{u(a)}^{u(b)}f(y)\,dy .
$$

Read from right to left, this is the **change-of-variables formula**

$$
\int_{u(a)}^{u(b)} f(y)\,dy = \int_a^b f(u(x))\,\frac{du}{dx}\,dx .
$$
:eqlabel:`eq_mdl-change_var`

An integral over $y$ can be computed as an integral over $x$, if the
integrand carries the stretch factor $du/dx$. The map $u$ need not be monotone
(exercise :exref:`ex_mdl-integral-calculus-orientation`).

A thin slice explains the factor. Cut $[a,b]$ into slices of width
$\epsilon$. The map $u$ sends the slice at $x$ to a slice of width about
$\epsilon\,u'(x)$ at $y=u(x)$, where the integrand has height $f(y)=f(u(x))$.
Its area, $f(u(x))\,u'(x)\,\epsilon$, is what the $x$-integral must add up
(:numref:`fig_mdl-rect-transform`).

![A thin sliver under the substitution $y=u(x)$. Left, the $x$ side (the right-hand side of the formula): a sliver of width $\epsilon$ at $x$ and height $f(u(x))$. Right, the $y$ side: its image, of width $\epsilon\,u'(x)$ and the same height. Matching the two areas forces the factor $du/dx$.](../img/mdl-cal-rect-trans.svg)
:label:`fig_mdl-rect-transform`

The formula is used by recognizing the pattern $f(u(x))\,u'(x)$ in an
integrand. In $\int_0^1x\,e^{-x^2}dx$, take $u(x)=x^2$ with $u'(x)=2x$: the
integrand is $\tfrac12e^{-u(x)}u'(x)$, so

$$
\int_0^1x\,e^{-x^2}\,dx=\frac12\int_0^1e^{-y}\,dy=\frac{1-e^{-1}}2=0.316 .
$$

The factor $x$ supplied the $u'(x)$ that the rule requires. Without it,
$\int e^{-x^2}dx$ has no elementary antiderivative, and this is the integral
behind the normalizer $1/\sqrt{2\pi}$ of the bell curve. The Gaussian
integral :eqref:`eq_mdl-gaussian-integral` obtains the missing factor from a
change to polar coordinates.

## Multiple Integrals

From here on two theorems are quoted without proof, both from
:cite:`Folland.1999`: Fubini's theorem and the change of variables in many
dimensions.

### Double Integrals and Fubini's Theorem

The road to $\int e^{-x^2}dx$ leads through two dimensions, so we first extend
the integral from intervals to regions of the plane. The bell in two
dimensions, $f(x,y)=e^{-x^2-y^2}$ over the square $[-2,2]^2$, encloses a volume
between its surface and the base plane. We compute it the way we computed
areas: tile the base with $\epsilon\times\epsilon$ squares, stand on each
square a box as tall as the surface, and add the box volumes
(:numref:`fig_mdl-bell-surface`). For $f$ on a rectangle $R=[a,b]\times[c,d]$
the limit of these box sums as $\epsilon\to0$ is the **double integral**
$\iint_R f(x,y)\,dx\,dy$, the signed volume under the surface.

![The bell surface $z=e^{-x^2-y^2}$ over the square $[-2,2]^2$. Tiling the base with $\epsilon\times\epsilon$ squares, $\epsilon=0.5$, and standing a box of height $f$ on each approximates the volume under the surface; the highlighted column of boxes at one fixed $x$ is one inner sum of Fubini's theorem.](../img/mdl-cal-bell-surface.svg)
:label:`fig_mdl-bell-surface`

How do we evaluate such a volume with one-dimensional tools? A finite sum
can be totalled in any order. Index the squares by $i$ along $x$ and $j$
along $y$, and total the boxes column by column
(:numref:`fig_mdl-sum-order`):

$$
\sum_{i,j}\epsilon^2 f(a+\epsilon i,c+\epsilon j)
=\sum_i \epsilon\Bigl(\sum_j \epsilon\, f(a+\epsilon i,c+\epsilon j)\Bigr).
$$

The inner sum is the left sum of $\int_c^d f(a+\epsilon i,y)\,dy$, the area of
the slice of the surface at $x=a+\epsilon i$ (the highlighted column in
:numref:`fig_mdl-bell-surface`), and the outer sum is the left sum of that
slice area over $x$.

![Totalling a grid of $\epsilon\times\epsilon$ cells with corner $(a,c)$, index $i$ along $x$ and $j$ along $y$. (1) Sum each column, at fixed $i$, over $j$. (2) Add the column totals from left to right. Reordering a finite sum changes nothing; this is the discrete fact behind iterated integration.](../img/mdl-cal-sum-order.svg)
:label:`fig_mdl-sum-order`

For finite grids the reordering is trivial. That it survives the limit
$\epsilon\to0$ is a theorem, and it needs a hypothesis. **Fubini's theorem**
states: if $f$ is **absolutely integrable** on $R$, that is
$\iint_R|f(x,y)|\,dx\,dy<\infty$, then the double integral equals either
iterated integral,

$$
\iint_R f(x,y)\,dx\,dy
= \int_c^{d}\!\left(\int_a^{b} f(x,y)\,dx\right)dy
= \int_a^{b}\!\left(\int_c^{d} f(x,y)\,dy\right)dx .
$$
:eqlabel:`eq_mdl-fubini`

A volume can be computed slice by slice, slicing in either direction. For
$f\ge0$ the same conclusion holds with no hypothesis, all three sides possibly
$+\infty$ (*Tonelli's theorem*). Densities, squared errors and every other
nonnegative integrand therefore need no check; a signed integrand needs
$\iint|f|<\infty$ first. The theorem is at work whenever a variable is
integrated out of a joint density and whenever an expectation changes places
with another integral.

The case used most often is a product integrand $f(x,y)=g(x)h(y)$. The inner
integral $\int_c^d g(x)h(y)\,dy=g(x)\int_c^d h(y)\,dy$ is $g(x)$ times a
constant, so

$$
\iint_R g(x)\,h(y)\,dx\,dy=\int_a^b g(x)\,dx\int_c^d h(y)\,dy .
$$
:eqlabel:`eq_mdl-fubini-product`

A product integrand integrates factor by factor. The bell is such a product,
$e^{-x^2-y^2}=e^{-x^2}e^{-y^2}$, and the Gaussian integral below rests on this
fact.

The hypothesis is needed. Take $f(x,y)=(x^2-y^2)/(x^2+y^2)^2$ on
$[0,1]^2$. A short computation gives
$\partial_y\bigl(y/(x^2+y^2)\bigr)=f(x,y)$, so by the evaluation rule
$\int_0^1f(x,y)\,dy=\frac1{1+x^2}$ for $x>0$, and then
$\int_0^1\frac{dx}{1+x^2}=\arctan1=\frac\pi4$. Since $f(x,y)=-f(y,x)$, the
other order gives $-\frac\pi4$. The two orders can disagree only because
$\iint|f|=\infty$: near the origin $|f|$ grows like the inverse square of the
distance, and its volume diverges.

When the region $D$ is not a rectangle, $\int_D f(\mathbf x)\,d\mathbf x$ is
defined by *zero-extension*: set $f$ to $0$ outside $D$ and integrate over any
rectangle containing $D$.

### Change of Variables in Many Dimensions

The one-dimensional stretch factor $du/dx$ has a many-dimensional successor,
and the linear case shows it. A linear map scales every area by the absolute
value of its determinant (:numref:`sec_mdl-linear-maps`); the sign records
orientation and plays no role in area. The map
$\boldsymbol\phi(u,v)=(au,bv)$ with $a,b>0$ has determinant $ab$ and sends the
unit disc, of area $\pi$, to the ellipse $x^2/a^2+y^2/b^2\le1$, whose area is
therefore $\pi ab$.

A nonlinear map is linear on a small scale, with its Jacobian as the linear
part, so the scale factor varies from point to point. Let $\Omega\subseteq\mathbb{R}^n$
be open and let $\boldsymbol{\phi}:\Omega\to\mathbb{R}^n$ be a
**$C^1$-diffeomorphism** onto its image: $\boldsymbol{\phi}$ is continuously
differentiable, injective, and its Jacobian $\mathbf J_{\boldsymbol\phi}$, the
matrix of partial derivatives of :numref:`sec_mdl-matrix-calculus`, has
$\det\mathbf J_{\boldsymbol\phi}(\mathbf{x})\neq 0$ everywhere on $\Omega$.
Injectivity stops the map from folding space onto itself, and the nonzero
determinant keeps it from collapsing a region to lower dimension. Then, for
absolutely integrable $f$,

$$
\int_{\boldsymbol{\phi}(\Omega)} f(\mathbf{y})\,d\mathbf{y}
= \int_{\Omega} f(\boldsymbol{\phi}(\mathbf{x}))\,
  \bigl|\det \mathbf J_{\boldsymbol\phi}(\mathbf{x})\bigr|\,d\mathbf{x}.
$$
:eqlabel:`eq_mdl-change_var_nd`

A tiny cube of volume $d\mathbf x$ at $\mathbf x$ is sent to a tiny
parallelepiped of volume $|\det\mathbf J_{\boldsymbol\phi}(\mathbf x)|\,d\mathbf x$
at $\mathbf y=\boldsymbol\phi(\mathbf x)$, and the factor in the integrand
accounts for that distortion.
:numref:`fig_mdl-cov-jacobian` puts the one- and two-dimensional pictures side
by side.

![Left: the one-dimensional substitution scales a segment of length $\epsilon$ by $u'(x)$. Right: in two dimensions a linear $\boldsymbol\phi$ sends the unit square to a parallelogram whose area is the volume factor $|\det\mathbf J_{\boldsymbol\phi}|$.](../img/mdl-cal-cov-jacobian.svg)
:label:`fig_mdl-cov-jacobian`

The formula uses $|\det\mathbf J_{\boldsymbol\phi}|$ and not
$\det\mathbf J_{\boldsymbol\phi}$ because its left side integrates over a set,
not between oriented limits. In one dimension the order of the limits carries
the sign: for a decreasing $u$ the limits in :eqref:`eq_mdl-change_var` run
backwards and $u'<0$, the two signs cancel, and the rule can be written over
the image interval with $|u'|$. Regions in $\mathbb R^n$ have no order, so
only this unsigned form generalizes (exercise
:exref:`ex_mdl-integral-calculus-orientation`).

Read with $f$ a density, the formula says how a density changes under a map,
which is how a normalizing flow computes the density of its output
(:eqref:`eq_mdl-cov_density` in :numref:`sec_mdl-random_variables`).

### The Gaussian Integral

The normalizer $1/\sqrt{2\pi}$ of the bell curve comes from the
one-dimensional integral $\int_{-\infty}^\infty e^{-x^2}dx$. The fundamental
theorem gives no closed form for it. An antiderivative exists, since
$e^{-x^2}$ is continuous, but it is a new function, not a combination of the
familiar ones. Surprisingly, the value over the whole line can be found all
the same, by a detour through two dimensions. The argument takes three steps
and is the most elegant of the section. (Normalized, the antiderivative is
the **error function**
$\operatorname{erf}(x)=\frac2{\sqrt\pi}\int_0^xe^{-t^2}\,dt$, which tends to
$1$ as $x\to\infty$ by the result below. Libraries provide it as `math.erf`.)

**I. Square and factor.** On the square $[-b,b]^2$ the product rule
:eqref:`eq_mdl-fubini-product` gives
$\iint_{[-b,b]^2}e^{-x^2-y^2}\,dx\,dy=\bigl(\int_{-b}^be^{-x^2}dx\bigr)^2$.
Let $b\to\infty$ on both sides:

$$
\left(\int_{-\infty}^{\infty} e^{-x^2}\,dx\right)^{\!2}
= \lim_{b\to\infty}\iint_{[-b,b]^2} e^{-x^2-y^2}\,dx\,dy
= \iint_{\mathbb R^2} e^{-x^2-y^2}\,dx\,dy .
$$
:eqlabel:`eq_mdl-gauss-square`

The square of the one-dimensional integral is the volume under the
two-dimensional bell, and that bell is rotationally symmetric.

**II. Polar coordinates.** Rotational symmetry calls for the map
$\boldsymbol{\phi}(r,\theta)=(r\cos\theta, r\sin\theta)$ on the open set
$\Omega=(0,\infty)\times(0,2\pi)$, where it is injective. Its Jacobian
determinant is

$$
\bigl|\det \mathbf J_{\boldsymbol\phi}\bigr|
= \left|\det\begin{bmatrix}\cos\theta & -r\sin\theta\\ \sin\theta & r\cos\theta\end{bmatrix}\right|
= r(\cos^2\theta + \sin^2\theta) = r .
$$

A polar cell with sides $dr$ along the ray and $r\,d\theta$ along the arc has
area $r\,dr\,d\theta$ (:numref:`fig_mdl-polar-cell`): cells far from the origin
are larger. This factor $r$ is the missing factor $x$ of the substitution
example $\int_0^1x\,e^{-x^2}dx$.

![A polar grid with one cell filled: radial side $dr$, arc side $r\,d\theta$, so its area is about $r\,dr\,d\theta$. The change to polar coordinates multiplies the integrand by $|\det\mathbf J_{\boldsymbol\phi}|=r$.](../img/mdl-cal-polar-cell.svg)
:label:`fig_mdl-polar-cell`

**III. Integrate.** By :eqref:`eq_mdl-change_var_nd`, and because
$-\tfrac12e^{-r^2}$ is an antiderivative of $re^{-r^2}$,

$$
\iint_{\mathbb{R}^2} e^{-x^2-y^2}\,dx\,dy
= \int_0^{2\pi}\!\!\int_0^\infty r\,e^{-r^2}\,dr\,d\theta
= 2\pi\cdot\frac12 = \pi .
$$

Taking the positive square root of :eqref:`eq_mdl-gauss-square`,

$$
\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi} .
$$
:eqlabel:`eq_mdl-gaussian-integral`

The area under $e^{-x^2}$ is $\sqrt\pi=1.7725$.

Three points make the argument rigorous. For a nonnegative integrand the
integrals over growing squares, or growing discs, increase to one common limit,
because every disc lies inside some square and every square inside some disc;
so the shape of the exhaustion does not matter. The polar map misses the ray
$\theta=0$, which has zero area and contributes nothing. And the change of
variables is applied on the bounded sets $(0,b)\times(0,2\pi)$, where the
integrand is bounded and continuous, before letting $b\to\infty$.

The normalizer of the bell curve now follows by one substitution. With
$z=\sqrt2\,x$, :eqref:`eq_mdl-change_var` gives

$$
\int_{-\infty}^{\infty}e^{-z^2/2}\,dz=\sqrt2\int_{-\infty}^{\infty}e^{-x^2}\,dx=\sqrt{2\pi}.
$$
:eqlabel:`eq_mdl-normal-normalizer`

Dividing the bell $e^{-z^2/2}$ by $\sqrt{2\pi}=2.5066$ makes its total area
$1$. This is where the constant in the standard normal density comes from.

Do the squares exhaust the plane as step I claims, and is the limit $\pi$? The
cell computes the grid sum over $[-b,b]^2$ at $\epsilon=0.01$, the exact square
value $\bigl(\int_{-b}^be^{-x^2}dx\bigr)^2=\pi\operatorname{erf}(b)^2$ from the
product rule, and $\pi$.

```{.python .input #integral-gaussian}
eps = 0.01
for b in [1, 2, 3, 6]:
    g = -b + eps * np.arange(round(2 * b / eps))    # left ends of the slices
    x, y = np.meshgrid(g, g, indexing='ij')
    grid = np.sum(eps**2 * np.exp(-x**2 - y**2))
    print(f'b={b}  grid sum={grid:.6f}  pi*erf(b)^2={math.pi * math.erf(b)**2:.6f}'
          f'  pi={math.pi:.6f}')
```

The grid sums match $\pi\operatorname{erf}(b)^2$ and climb to $\pi$; by $b=3$
the volume outside the square is about $10^{-4}$. The agreement to four or
more decimals at $\epsilon=0.01$ has a reason: for an even integrand on a
symmetric interval the left rule's first-order bias
$\tfrac\epsilon2\bigl(f(-b)-f(b)\bigr)$ vanishes.

The probability $P(-1\le Z\le1)\approx0.68$ of the standard normal is now a
computation as well. The substitution $z=\sqrt2\,t$ turns the area under the
density $p(z)=e^{-z^2/2}/\sqrt{2\pi}$ into an error function:

$$
P(-1\le Z\le1)=\frac{1}{\sqrt{2\pi}}\int_{-1}^1e^{-z^2/2}\,dz
=\frac{1}{\sqrt\pi}\int_{-1/\sqrt2}^{1/\sqrt2}e^{-t^2}\,dt
=\operatorname{erf}\bigl(1/\sqrt2\bigr)=0.683 .
$$

Both questions about the bell curve are thereby answered, and the puzzle
that $p(0)=0.40$ while $P(Z=0)=0$ dissolves: a density is probability per
unit length, and a single point is an interval of width zero.

One question from the beginning of the section is still open. Training
through random sampling needs to know how such an area changes when a
parameter of the density moves, and for that we must differentiate an
integral.

## Summary

The definite integral is the limit of sums of thin slices, a signed area that
the fundamental theorem evaluates: the area-so-far has derivative $F'=f$, so
$\int_a^b f\,dx=G(b)-G(a)$ for any antiderivative $G$. Integration by parts
and substitution are the product and chain rules read backwards, and in many
dimensions the stretch factor becomes $|\det\mathbf J_{\boldsymbol\phi}|$, the
local volume factor; with polar coordinates it gives $\int
e^{-x^2}dx=\sqrt\pi$, hence the normal constant $\sqrt{2\pi}$.

## Exercises

1. [code] **Midpoint versus left rule.** The function `left_sum` evaluates
   the left-endpoint rule. The midpoint rule instead evaluates $f$ at the
   center of each slice of width $\epsilon$. On $[a,b]$ with
   $n=(b-a)/\epsilon$ slices and $x_i=a+i\epsilon$, it returns
   $\epsilon\sum_{i=1}^n f\bigl(x_{i-1}+\tfrac{\epsilon}{2}\bigr)$.
    1. The text derives the error of the left rule on $[a,b]$,
       $\tfrac{\epsilon}{2}\bigl(f(a)-f(b)\bigr)$, from a first-order
       expansion of $f$ on each slice. The left sum of $x^2$ on
       $[0,1]$ has error $-\tfrac1{2N}+\tfrac1{6N^2}$; which term does the
       expansion predict? Use the estimate to predict the errors that the
       text's cell prints at $\epsilon=10^{-1},\dots,10^{-4}$ for
       $\int_0^2 \tfrac{x}{1+x^2}\,dx$. Do the sums lie above or below
       $\tfrac12\log 5$?
    1. Implement both rules for an integrand $f$ on $[a,b]$. Evaluate
       $\int_1^2 \tfrac{1}{x}\,dx$ with an antiderivative, and tabulate the
       error of both rules for $\epsilon\in\{10^{-1},10^{-2},10^{-3}\}$. By
       what factor does each error shrink when $\epsilon$ shrinks by a factor
       of ten?
    1. Expand $f$ to second order about the center of each slice and show
       that the integral minus the midpoint sum is approximately
       $\tfrac{\epsilon^2}{24}\bigl(f'(b)-f'(a)\bigr)$. Compare this
       prediction with your table for $\int_1^2 \tfrac{1}{x}\,dx$. For which
       integrands does the midpoint rule underestimate the integral?

   *Adapted from MIT 18.01SC,
   [Unit 3 exercises](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/eaabc6c74a11402341dbc120fffa5ea5_MIT18_01SC_pset3prb.pdf),
   Problems 3G-1 and 3G-5.*
   :exlabel:`ex_mdl-integral-calculus-midpoint-rule`

1. **Activations as accumulated areas.** Let $H$ be the unit step, with
   $H(t)=1$ for $t>0$ and $H(t)=0$ for $t\le 0$. Let
   $\phi(t)=e^{-t^2/2}/\sqrt{2\pi}$ be the standard Gaussian density and
   $\Phi(x)=\int_{-\infty}^x \phi(t)\,dt$ its distribution function.
    1. Show that $\int_{-\infty}^{x} H(t)\,dt=\mathrm{ReLU}(x)$. Which
       hypothesis of the fundamental theorem :eqref:`eq_mdl-ftc` fails at
       $x=0$? What are the one-sided derivatives of the integral there?
    1. Write $\Phi(x)=\Phi(0)+\int_0^x \phi(t)\,dt$ to conclude that
       $\Phi'=\phi$. Then substitute $t=\sqrt2\,s$ to show that
       $\Phi(x)=\tfrac12\bigl(1+\operatorname{erf}(x/\sqrt2)\bigr)$.
    1. The GELU activation of :numref:`sec_mlp` is $x\Phi(x)$. Derive its
       derivative, find the points where the derivative is largest and
       smallest, and evaluate it there with `math.erf`. In which interval
       does the slope of GELU lie, and how does that interval compare with
       the slopes $0$ and $1$ of ReLU?
   :exlabel:`ex_mdl-integral-calculus-activation-areas`

1. **Heavy tails.** The *Pareto density* with tail index $\alpha>0$ is
   $p(x)=\alpha x^{-\alpha-1}$ for $x\ge 1$ and $p(x)=0$ otherwise. The
   *Cauchy density* is $p(x)=\tfrac{1}{\pi(1+x^2)}$ on $\mathbb R$.
    1. Show that the Pareto density is normalized for every $\alpha>0$.
       For which $\alpha$ are $\mathbb{E}[X]$ and $\mathbb{E}[X^2]$ finite?
       Compute $\mathbb{E}[X]$ where it is finite.
    1. Show that the Cauchy density is normalized, using the antiderivative
       $\arctan x$ of $\tfrac{1}{1+x^2}$.
    1. Show that $\int_{-b}^{b} x\,p(x)\,dx=0$ for the Cauchy density and
       every $b>0$. Why does the Cauchy density nevertheless have no mean
       under the definition of $\int_{-\infty}^{\infty}$ that follows
       :eqref:`eq_mdl-improper`? How fast does $\int_0^b x\,p(x)\,dx$ grow
       with $b$?
   :exlabel:`ex_mdl-integral-calculus-heavy-tails`

1. **Gamma function by parts.** The *Gamma function* is
   $\Gamma(t)=\int_0^\infty u^{t-1}e^{-u}\,du$. For $t<1$ the integrand is
   unbounded at $u=0$, and the integral near $0$ is defined by a limit,
   $\int_0^1=\lim_{a\to 0^+}\int_a^1$, as in
   the subsection :numref:`subsec_mdl-improper-integrals`.
    1. For which $t$ does the integral converge at $u=0$? Compare the
       condition with the test for $\int_0^1x^{-s}\,dx$ in
       the subsection :numref:`subsec_mdl-improper-integrals`.
    1. Integrate by parts :eqref:`eq_mdl-parts` at a general $t>0$ to show
       that $\Gamma(t+1)=t\,\Gamma(t)$, and conclude that $\Gamma(n+1)=n!$
       for every integer $n\ge 0$. Where does the argument use $t>0$? Use
       the result to compute the variance of the exponential density
       $e^{-u}$ on $u\ge 0$, whose mean
       :numref:`subsec_mdl-integration-by-parts` computes.
    1. Substitute $u=x^2$ and use the Gaussian integral
       :eqref:`eq_mdl-gaussian-integral` to evaluate $\Gamma(\tfrac12)$.
    1. The product rule :eqref:`eq_mdl-fubini-product`, applied $d$ times,
       gives $\int_{\mathbb R^d}e^{-\|\mathbf x\|^2}\,d\mathbf x=\pi^{d/2}$. In
       polar form the same integral is
       $S_{d-1}\int_0^\infty r^{d-1}e^{-r^2}\,dr$, where $S_{d-1}$ is the
       surface area of the unit sphere in $\mathbb R^d$. Show that the radial
       integral is $\tfrac12\Gamma(d/2)$ and deduce
       $S_{d-1}=2\pi^{d/2}/\Gamma(d/2)$. Integrate $S_{d-1}r^{d-1}$ over
       $r\in[0,1]$ to show that the unit ball has volume
       $V_d=\pi^{d/2}/\Gamma(d/2+1)$. Evaluate $V_d$ for
       $d\in\{1,2,3,5,10,20\}$ and explain why $V_d\to0$. A midpoint grid with
       three points per axis on the cube $[-1,1]^{10}$ has $3^{10}$ points;
       what fraction of them lies inside the unit ball, and what does that say
       about grids in high dimensions?
   :exlabel:`ex_mdl-integral-calculus-gamma-function`

1. **Orientation in substitution.** The one-dimensional rule
   :eqref:`eq_mdl-change_var` integrates between oriented limits and
   carries the factor $\frac{du}{dx}$. The $n$-dimensional rule
   :eqref:`eq_mdl-change_var_nd` integrates over a set and carries
   $\bigl|\det \mathbf J_{\boldsymbol{\phi}}\bigr|$.
    1. Let $u$ be continuously differentiable and strictly monotone on
       $[a,b]$, either increasing or decreasing. Show that

       $$\int_{u([a,b])} f(y)\,dy=\int_a^b f(u(x))\,|u'(x)|\,dx,$$

       where the left side integrates over the interval $u([a,b])$ from its
       smaller to its larger endpoint. Check the decreasing case with
       $u(x)=e^{-x^2}$ and $f=1$ on $[0,1]$.
    1. Take $u(x)=x^2$ on $[-1,1]$ and $f(y)=y$. Evaluate both sides of
       :eqref:`eq_mdl-change_var`, and both sides of the absolute-value form
       of (a). Which of the two forms remains valid for this $u$, which is
       not injective? Explain the discrepancy in the other form.
   :exlabel:`ex_mdl-integral-calculus-orientation`

1. **Swapping the order.** Fubini's theorem :eqref:`eq_mdl-fubini`, together
   with zero-extension for regions that are not rectangles, lets us choose
   the order of integration.
    1. The inner integral of
       $\int_0^1\bigl(\int_y^1 e^{-x^2}\,dx\bigr)\,dy$ has no elementary
       antiderivative. Sketch the triangular region of integration, reverse
       the order, and evaluate the integral in closed form.
    1. Let $X\ge 0$ have density $p$. Write $x=\int_0^x dt$ inside
       $\mathbb{E}[X]=\int_0^\infty x\,p(x)\,dx$ and swap the two integrals to
       show that $\mathbb{E}[X]=\int_0^\infty P(X>t)\,dt$. Why is the swap
       valid even when $\mathbb{E}[X]=\infty$? Check the formula on the
       density $\lambda e^{-\lambda x}$ for $x\ge 0$.
   :exlabel:`ex_mdl-integral-calculus-order-of-integration`

1. **Gaussian normalizers and moments.** The text normalizes the standard
   normal and computes its moments with the Stein identity
   :eqref:`eq_mdl-stein-1d` for the bell weight. Let
   $p(x)=\tfrac{1}{\sqrt{2\pi\sigma^2}}\exp\bigl(-\tfrac{(x-\mu)^2}{2\sigma^2}\bigr)$
   with $\sigma>0$.
    1. Substitute $x=\mu+\sigma z$ and use
       :eqref:`eq_mdl-normal-normalizer` to show that $p$ is normalized.
       Which $\mu$ and $\sigma^2$ does the density
       $\tfrac{1}{\sqrt\pi}e^{-x^2}$ have?
    1. Let $X$ have density $p$, and let $g$ be continuously differentiable with
       $g(x)\,p(x)\to 0$ as $x\to\pm\infty$. Use
       $p'(x)=-\tfrac{x-\mu}{\sigma^2}\,p(x)$ and integration by parts to
       show that

       $$\mathbb{E}\bigl[(X-\mu)\,g(X)\bigr]=\sigma^2\,\mathbb{E}\bigl[g'(X)\bigr].$$

       Deduce that $\mathbb{E}[(X-\mu)^2]=\sigma^2$ and
       $\mathbb{E}[(X-\mu)^4]=3\sigma^4$.
    1. Let $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ be a $d\times d$
       covariance matrix with invertible $\mathbf L$. Use the change of
       variables $\mathbf x=\mathbf L\mathbf z$ in
       :eqref:`eq_mdl-change_var_nd` and the product rule
       :eqref:`eq_mdl-fubini-product` to show that

       $$\int_{\mathbb R^d}\exp\bigl(-\tfrac12\mathbf x^\top\boldsymbol\Sigma^{-1}\mathbf x\bigr)\,d\mathbf x=(2\pi)^{d/2}\,(\det\boldsymbol\Sigma)^{1/2}.$$
   :exlabel:`ex_mdl-integral-calculus-gaussian-normalizers`

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

Integral Calculus<br>**slices · the fundamental theorem · change of variables · the Gaussian integral**
:::
:::

::: {.slide title="A Density of 0.40 at Zero, yet P(Z = 0) = 0"}
[The problem]{.kicker}

The standard normal has density $p(z)=e^{-z^2/2}/\sqrt{2\pi}$.

- $p(0)=0.40$, yet no single point has positive probability.
- Intervals do: $P(-1\le Z\le 1)\approx 0.68$, the **area** under $p$.
- The constant $1/\sqrt{2\pi}$ makes the total area $1$.
- Training through sampling needs the **derivative** of such an area.

::: {.d2l-note .warn}
Where does $1/\sqrt{2\pi}$ come from, and how is $0.68$ computed?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- evaluate an integral by an antiderivative or by a symmetry;
- predict the error of a grid sum from the slice width;
- change variables in one and in many dimensions, and integrate the bell curve.

**Running examples:** $Z\sim\mathcal N(0,1)$; $\int_0^2\frac{x}{1+x^2}dx$.

**The idea:** thin slices add up; undo a derivative to add them.
:::

::: {.slide title="Integration Turns Rates Back into Totals"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Single Variable Calculus | derivative, Taylor, corners |
| Multivariable Calculus | gradient, Hessian, critical points |
| The Chain Rule on Computational Graphs | sum over paths, backward pass |
| Matrix Calculus | Jacobians, a layer's backward rule |
| Automatic Differentiation | forward and reverse mode |
| **Integral Calculus** | **areas, change of variables, the Gaussian integral** |
| Expectations and Their Gradients | score-function and pathwise estimators |
:::

::: {.col .narrow}
- **Uses:** derivatives, chain rule, determinants.
- **Skips:** measure theory, Stokes' theorem.
:::
:::
:::

::: {.slide title="Thin Slices Add Up: the Integral Is a Limit of Sums"}
::: {.cols .vc}
::: {.col}
$\int_0^1x^2\,dx$ with $N$ slices tagged at the left end:

$$\sum_{i=0}^{N-1}\tfrac1N\bigl(\tfrac iN\bigr)^2=\tfrac13-\tfrac1{2N}+\tfrac1{6N^2}$$

- $N=10$: $0.285$; $100$: $0.3284$; $1000$: $0.3328$
- limit $\tfrac13$, error $\approx\tfrac1{2N}$
:::

::: {.col .fig}
![](../img/mdl-cal-riemann.svg)
:::
:::
:::

::: {.slide title="The Left Rule's Error Falls Like ε"}
$\int_0^2\frac{x}{1+x^2}\,dx=\tfrac12\log5$, slices of width $\epsilon$:

@!integral-riemann-converge

::: {.d2l-note}
Error $\approx\tfrac\epsilon2\bigl(f(a)-f(b)\bigr)=-0.2\,\epsilon$: half the
slice width times the change of $f$ from end to end.
:::
:::

::: {.slide title="Area Grows at the Height of the Curve, so ∫f = G(b) − G(a)"}
::: {.cols .vc}
::: {.col}
- Area-so-far $F(x)=\int_a^xf$; a strip of width $\epsilon$ adds
  $\approx\epsilon f(x)$, so $F'=f$.
- Any $G$ with $G'=f$: $\int_a^bf=G(b)-G(a)$.
- $G=\tfrac12\log(1+x^2)$ gives $\tfrac12\log 5$.
:::

::: {.col .fig}
![](../img/mdl-cal-ftc-sliver.svg)
:::
:::

[Needs $G'=f$ on all of $[a,b]$: $\int_{-1}^1x^{-2}dx\ne[-1/x]_{-1}^1=-2$.]{.small}
:::

::: {.slide title="∫₁^∞ x^(−s) dx Is Finite Only for s > 1"}
$$\int_1^b x^{-s}dx=\frac{b^{1-s}-1}{1-s}\ \to\ \frac1{s-1}\ (s>1),\quad\infty\ (s\le1)$$

$\int_0^\infty e^{-x}dx=1$ by exact partials and by left sums, $\epsilon=10^{-3}$:

@!integral-improper

::: {.d2l-note}
Truncation error $e^{-b}$; discretization error $\tfrac\epsilon2$.
Each knob fixes one error only.
:::
:::

::: {.slide title="Integration by Parts Moves the Derivative to the Other Factor"}
$$\int_a^b u\,v'\,dx = \bigl[\,u\,v\,\bigr]_a^b - \int_a^b u'\,v\,dx$$

- $u=x$, $v'=e^{-x}$: $\int_0^\infty xe^{-x}dx=0+1=1$.
- Bell weight $w=e^{-z^2/2}$, $w'=-zw$:
  $\int g'\,w\,dz=\int z\,g\,w\,dz$.
- $g=z$: the bell-weighted mean of $z^2$ is $1$; $g=z^3$: of $z^4$, $3$.
:::

::: {.slide title="Check: Two Integrals"}
[Check]{.kicker}

1. Is $\int_1^\infty x^{-3/2}\,dx$ finite? If so, what is it?
2. What is $\int_0^1 x\,e^{-x^2}\,dx$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) $s=\tfrac32>1$: the value is $\frac1{s-1}=2$.
(2) Substitute $y=x^2$: $\tfrac12\int_0^1e^{-y}dy=\tfrac{1-e^{-1}}2=0.316$.
:::
:::
:::

::: {.slide title="Substitution Is the Chain Rule Reversed: Match the Slivers"}
$$\int_{u(a)}^{u(b)} f(y)\,dy = \int_a^b f(u(x))\,u'(x)\,dx$$

![](../img/mdl-cal-rect-trans.svg)

Without the factor $x$, $\int e^{-x^2}dx$ has no elementary antiderivative.
:::

::: {.slide title="A Grid Sum Totals in Any Order: Fubini"}
::: {.cols .vc}
::: {.col}
$$\iint_R f = \int_a^b\!\Bigl(\int_c^d f\,dy\Bigr)dx = \int_c^d\!\Bigl(\int_a^b f\,dx\Bigr)dy$$

- products split: $\iint g(x)h(y)=\int g\int h$
- $\frac{x^2-y^2}{(x^2+y^2)^2}$ on $[0,1]^2$: the orders give $\pm\frac\pi4$
:::

::: {.col .fig}
![](../img/mdl-cal-sum-order.svg)
:::
:::

[Needs $\iint|f|<\infty$, or $f\ge0$ (Tonelli).]{.small}
:::

::: {.slide title="The Jacobian Determinant Is the Local Volume Factor"}
$$\int_{\boldsymbol\phi(\Omega)} f(\mathbf y)\,d\mathbf y = \int_{\Omega} f(\boldsymbol\phi(\mathbf x))\,\bigl|\det\mathbf J_{\boldsymbol\phi}(\mathbf x)\bigr|\,d\mathbf x$$

![](../img/mdl-cal-cov-jacobian.svg)

$(u,v)\mapsto(2u,v)$: $|\det|=2$, so the ellipse $\tfrac{x^2}4+y^2\le1$
has area $2\pi$.

[$\boldsymbol\phi$ one-to-one and continuously differentiable on the open set
$\Omega$, with $\det\mathbf J_{\boldsymbol\phi}\ne0$; $f$ absolutely integrable.]{.small}
:::

::: {.slide title="Polar Coordinates Turn ∫ e^(−x²) dx into √π"}
::: {.cols .vc}
::: {.col}
I. $\bigl(\int e^{-x^2}dx\bigr)^2=\iint e^{-x^2-y^2}dx\,dy$ (product rule)

II. $x=r\cos\theta$, $y=r\sin\theta$: $|\det\mathbf J_{\boldsymbol\phi}|=r$

III. $\int_0^{2\pi}\!\int_0^\infty re^{-r^2}dr\,d\theta=2\pi\cdot\tfrac12=\pi$

$$\boxed{\int e^{-x^2}dx=\sqrt\pi,\quad \int e^{-z^2/2}dz=\sqrt{2\pi}}$$
:::

::: {.col .fig}
![](../img/mdl-cal-polar-cell.svg)
:::
:::
:::

::: {.slide title="Squares Exhaust the Plane: the Volume Tends to π"}
Grid sum of $e^{-x^2-y^2}$ over $[-b,b]^2$ at $\epsilon=0.01$:

@!integral-gaussian

::: {.d2l-note}
The square's value is $\pi\operatorname{erf}(b)^2$; by $b=3$ the volume
outside the square is about $10^{-4}$.
:::
:::

::: {.slide title="Probabilities Are Areas Under the Density"}
$$p\ge0,\quad\int p=1,\qquad P(a\le X\le b)=\int_a^bp(x)\,dx,\qquad P(X=x)=0$$

::: {.d2l-note}
$P(|Z|\le1)=\operatorname{erf}(1/\sqrt2)=0.683$: the $68\%$ we set out to compute.
$\mathbb{E}[Z^2]=1$ by the Stein identity.
:::
:::

::: {.slide title="Integration Accumulates Slices"}
[Wrap-up]{.kicker}

- Slices: $0.285\to\tfrac13$; error $\approx\tfrac\epsilon2(f(a)-f(b))$.
- Fundamental theorem: $\int_0^2\frac{x}{1+x^2}dx=\tfrac12\log5=0.805$.
- Polar coordinates: $\sqrt\pi$, so the normal constant is $\sqrt{2\pi}$.
- Densities: $P(|Z|\le1)=0.683$.

::: {.d2l-note}
Next: how such an area changes when a parameter moves, the gradient of an
expectation.
:::
:::
