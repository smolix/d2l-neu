# Further Reading
:label:`sec_mdl-calculus-further-reading`

The sections of this chapter teach the derivative as a local linear model,
its computation by forward and reverse mode, and the integral as a limit of
slice sums. This section collects, in the order of the chapter, topics they
had no room for: the theory of derivatives at kinks, results for vector- and
matrix-valued maps, refinements of automatic differentiation, and the
integration theory and estimators that go beyond what the sections need.
Each topic states the problem it addresses and what it changes relative to
the chapter. Resources for the material the chapter does teach are listed in
its introduction (:numref:`chap_mdl-calculus`). Nothing else in the book
depends on this section.

## Derivatives at Kinks

### Conservative Fields

:numref:`sec_mdl-single_variable_calculus` shows that backpropagation with the
convention $\mathrm{ReLU}'(0)=0$ returns slope $0$ for
$\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$ at $x=0$, a number that is neither the
derivative nor a subgradient of the identity. The subdifferential cannot
describe this, because it does not obey the chain rule for nonconvex
compositions. :citet:`Bolte.Pauwels.2021` replace it by a *conservative
field*, a set-valued derivative that obeys the chain rule by construction and
agrees with the gradient outside a set of measure zero. Backpropagation with
any fixed choice at each kink returns an element of such a field. That the
exceptional set is small is Rademacher's theorem :cite:`Rademacher.1919`: a
Lipschitz function on $\mathbb R^n$ is differentiable almost everywhere, so
the kinks of a ReLU network's loss form a set of measure zero.

The same paper answers whether stochastic gradient descent driven by these
values converges. For losses built from the usual layers (definable
functions), with bounded iterates and the usual conditions on step sizes and
noise, the iterates approach the set of points where the conservative field
contains zero. The result holds in exact arithmetic. In floating point the
kinks are hit: :citet:`Bertoin.Bolte.Gerchinovitz.ea.2021` find that the
value of $\mathrm{ReLU}'(0)$ has no effect in 64-bit arithmetic, changes the
backward pass in about half of their 32-bit experiments, and changes it
systematically in 16-bit arithmetic, where it can also change the accuracy
reached by plain SGD.

### Subgradient Methods

At the minimum of $|x|$ the valid subgradient $\tfrac12$ sends a step uphill,
as :numref:`sec_mdl-single_variable_calculus` shows, so the descent lemma has
no analogue for nonsmooth functions. The subgradient method of convex
optimization converges anyway, by a different mechanism: the step sizes
shrink, with $\sum_t\eta_t=\infty$ and $\sum_t\eta_t^2<\infty$, and the
analysis tracks the distance to the minimizer instead of the function value.
For a convex function with subgradients bounded by $G$, starting at distance
$R$ from a minimizer, the best of $T$ iterates is within $RG/\sqrt T$ of the
minimum. When the dimension exceeds $T$, no method that uses only
subgradients can do better in the worst case :cite:`Nesterov.2018`. :citet:`Rockafellar.1970` develops the calculus of
subdifferentials, including when the sum rule holds with equality.

## Derivatives of Vector and Matrix Maps

### The Mean Value Theorem for Vector-Valued Maps

The one-variable mean value theorem of
:numref:`sec_mdl-single_variable_calculus` writes $f(b)-f(a)=f'(\xi)(b-a)$
for one intermediate point $\xi$. For a curve
$\boldsymbol\gamma:\mathbb R\to\mathbb R^n$ no such point need exist: the
circle $\boldsymbol\gamma(t)=(\cos t,\sin t)$ returns to its start at
$t=2\pi$, yet $\boldsymbol\gamma'(\xi)\neq\mathbf 0$ for every $\xi$. Only the
*mean value inequality*
$\|\boldsymbol\gamma(b)-\boldsymbol\gamma(a)\|\le\sup_\xi\|\boldsymbol\gamma'(\xi)\|\,|b-a|$
survives (Theorem 5.19 of :citet:`Rudin.1976`). The Taylor bounds of
:numref:`sec_mdl-multivariable_calculus` need no vector version of the
theorem, because they restrict a scalar function to a line, where the
one-variable theorem applies.

### Jacobians of Matrix-Valued Maps

:numref:`sec_mdl-matrix-calculus` differentiates scalar losses, so its
gradients have the shape of the variable. The derivative of a matrix-valued
function of a matrix has four indices. Flattening both matrices with
$\operatorname{vec}$, which stacks the columns, turns it into an ordinary
Jacobian: a differential $d\mathbf F=\mathbf A\,d\mathbf X\,\mathbf B$
becomes
$d\operatorname{vec}\mathbf F=(\mathbf B^\top\otimes\mathbf A)\,d\operatorname{vec}\mathbf X$,
and the chain rule multiplies Kronecker products. :citet:`Giles.2008b`
collects the forward and reverse rules for products, inverses, determinants,
and matrix decompositions, derived by the differential method of the
section. The Matrix Cookbook :cite:`Petersen.Pedersen.ea.2008` tabulates
identities for lookup without derivations.

## Automatic Differentiation

### Optimal Jacobian Accumulation

:numref:`sec_mdl-matrix-calculus` shows that the order in which a chain of
Jacobians is multiplied changes the cost by a factor of the layer width. For
a chain, the cheapest bracketing is the matrix-chain dynamic program, which
runs in $O(K^3)$ time for $K$ factors. A computation graph that is not a chain
allows many more orders: each intermediate node can be eliminated by
multiplying the partial derivatives on its incoming and outgoing edges, in
any sequence. Finding the sequence with the fewest multiplications is
NP-complete :cite:`Naumann.2008`. Frameworks therefore offer only forward
mode, reverse mode, and their compositions. :citet:`Griewank.Walther.2008`
treat vertex, edge, and face elimination as the general operations and
discuss the heuristics used in compilers of derivative code.

### Recursive Checkpointing

The one-level schedule of :numref:`sec_mdl-matrix-calculus-autodiff` stores
$T/s+s$ values, at best $2\sqrt T$, for one extra forward pass
:cite:`Chen.Xu.Zhang.ea.2016`. Unrolled solvers and very long sequences need
less memory still. Checkpointing recursively inside each recomputed segment
reaches $O(\log T)$ memory for $O(T\log T)$ recomputation
:cite:`Griewank.1992`, and Griewank's binomial schedule is optimal for a given
number of stored values when every step costs the same
:cite:`Griewank.Walther.2008`. `torch.utils.checkpoint` and `jax.checkpoint`
apply one level per marked block; nesting the marks gives a recursive schedule
by hand.

### Taylor Mode

Forward over reverse gives $\mathbf H\mathbf v$ at the cost of a few
gradients, but nesting forward mode $k$ times to get the $k$-th derivative
along a direction carries $2^k$ components per value. Taylor mode carries the
truncated series $a_0+a_1\varepsilon+\cdots+a_k\varepsilon^k$ with
$\varepsilon^{k+1}=0$ instead of a dual number. Each primitive propagates the
$k+1$ coefficients in one pass: a product convolves the two coefficient
sequences, at $O(k^2)$ arithmetic, and $\exp$, $\log$, and $\sin$ follow from
recurrences of the same cost :cite:`Griewank.Walther.2008`. The exercise on
second derivatives in one forward pass in
:numref:`sec_mdl-matrix-calculus-autodiff` is the case $k=2$;
`jax.experimental.jet` implements the general case.

### Forward Gradients

Reverse mode needs a tape, and forward mode needs one pass per input
direction. :citet:`Baydin.Pearlmutter.Syme.ea.2022` take one forward pass
with a random tangent $\mathbf v\sim\mathcal N(\mathbf 0,\mathbf I)$ and use
$(\nabla\ell^\top\mathbf v)\,\mathbf v$ as the gradient. The estimate is
unbiased, because $\mathbb{E}[\mathbf v\mathbf v^\top]=\mathbf I$, and its
mean squared error is $(n+1)\|\nabla\ell\|^2$ for $n$ parameters. It removes
the tape and multiplies the gradient noise by about $n$, which is acceptable
for small models and prohibitive for large ones.

## Integration

### The Lebesgue Integral

The indicator of the rationals has tagged sums that never settle, so it has
no Riemann integral (:numref:`sec_mdl-integral_calculus`), and a pointwise
limit of Riemann-integrable functions need not be Riemann integrable.
Lebesgue slices the range instead of the domain: it measures the set where
$f$ exceeds each level and adds up these measures. Sets of measure zero, such
as the rationals or the ray $\theta=0$ of polar coordinates, contribute
nothing, so the indicator integrates to $0$. The section quotes three results
of this theory without proof: Fubini–Tonelli, the change of variables in $n$
dimensions, and differentiation under an integral over an infinite range.
All three rest on the monotone and dominated convergence theorems, which
state when a limit passes through an integral: if $f_k\to f$ pointwise and
$|f_k|\le g$ for one integrable $g$, then $\int f_k\to\int f$
:cite:`Folland.1999`.

### Quasi-Monte Carlo

:numref:`sec_mdl-integral_calculus` compares a second-order grid, with error
$N^{-2/d}$, against Monte Carlo, with error $N^{-1/2}$ in every dimension.
Random points leave gaps and clusters by chance. Low-discrepancy point sets
(Sobol and Halton sequences, lattice rules) fill the cube more evenly by
construction, and for integrands of bounded variation (in the sense of Hardy
and Krause) their error is
$O(N^{-1}(\log N)^d)$, which beats $N^{-1/2}$ once $N$ is large compared with
the dimension. A deterministic point set has no error estimate; randomized
quasi-Monte Carlo shifts or scrambles the points so that independent
replicates give one. :citet:`Owen.2013` covers both in Chapters 15–17.

### Gradient Estimators beyond Score and Pathwise

The section derives the score-function and pathwise estimators and reduces
the variance of the first with a constant baseline. Other control variates
use the model itself: REBAR :cite:`Tucker.Mnih.Maddison.ea.2017` subtracts the
pathwise gradient of a continuous relaxation from the score-function
estimator and adds back its expectation, which keeps the estimate unbiased
for discrete variables. The relaxations themselves, Gumbel-softmax
:cite:`Jang.Gu.Poole.2017,Maddison.Mnih.Teh.2017` and straight-through
estimators :cite:`Bengio.Leonard.Courville.2013`, have low variance but are
biased for the original objective. A third family, measure-valued
derivatives, writes the derivative of a density as a scaled difference of
two densities and samples from both. :citet:`Mohamed.Rosca.Figurnov.ea.2020`
compare all three families in one framework, state the conditions each
needs, and recommend which to use for which model.

[Discussions](https://d2l.discourse.group/t/36858)
