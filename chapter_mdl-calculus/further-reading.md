# Further Reading
:label:`sec_mdl-calculus-further-reading`

Throughout this chapter, we proved our main results under clean assumptions
that deep learning practice routinely bends: a ReLU network is full of sharp
kinks, a reverse-mode tape can easily outgrow GPU memory, and a
thousand-dimensional integral cannot be placed on a grid. The short notes
below summarize what happens when those assumptions break down. Two topics
bear directly on everyday training and are worth reading first: conservative
fields, which explain in what sense backpropagation through kinks is still
valid, and recursive checkpointing, which keeps memory manageable on long
computations. The remaining topics can be browsed on demand, and the entire
section can be skipped without losing the thread of the book. Annotated
resources for the material taught in the chapter itself appear in the chapter
introduction (:numref:`chap_mdl-calculus`).

## Derivatives at Kinks

### Conservative Fields

As :numref:`sec_mdl-single_variable_calculus` illustrates, adopting the
standard convention $\mathrm{ReLU}'(0)=0$ leads to a curious anomaly: if we
write the identity function as $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$,
backpropagation returns slope $0$ at $x=0$, a number that is neither the true
derivative nor even a valid subgradient of $x$. Classical subdifferentials
cannot explain this behavior, because their sum and chain rules hold only as
set inclusions once the constituent functions are no longer convex. To close
this gap, :citet:`Bolte.Pauwels.2021` introduce *conservative fields*,
set-valued generalizations of the derivative that obey the chain rule by
construction and agree with the ordinary gradient everywhere outside a set of
measure zero. Backpropagation with any fixed choice of slope at each kink
always returns an element of such a field. And why is the exceptional set of
kinks harmlessly small? By Rademacher's theorem :cite:`Rademacher.1919`, every
Lipschitz function on $\mathbb R^n$ is differentiable almost everywhere, so
the kinks of a ReLU network's loss occupy a set of measure zero.

The same work settles the practical question of whether stochastic gradient
descent driven by these backpropagation values still converges. For losses
built from standard neural network layers, assuming bounded iterates and the
usual conditions on step sizes and gradient noise, the iterates converge to
the set of critical points where the conservative field contains zero. That
guarantee holds in exact arithmetic, whereas in finite-precision floating
point, iterates actually land on kinks.
:numref:`sec_mdl-single_variable_calculus` reports how often this occurs across
three numerical precisions, as measured by
:citet:`Bertoin.Bolte.Gerchinovitz.ea.2021`, and shows how little the choice
of $\mathrm{ReLU}'(0)$ ends up affecting training.

### Subgradient Methods

Smooth optimization relies on the descent lemma, which guarantees that a small
step against the gradient lowers the objective. Nonsmooth functions break that
guarantee: at the minimum of $|x|$, the valid subgradient $\tfrac12$ actually
points uphill, as shown in :numref:`sec_mdl-single_variable_calculus`.
Remarkably, the subgradient method converges anyway, because its analysis
tracks the Euclidean distance to a minimizer rather than the objective value
itself. With diminishing step sizes satisfying $\sum_t\eta_t=\infty$ and
$\sum_t\eta_t^2<\infty$, the best objective value seen so far converges to the
minimum. Obtaining the optimal convergence rate calls for a different schedule:
for a convex function whose subgradients have norm at most $G$, starting at
distance $R$ from a minimizer, a constant step size $R/(G\sqrt T)$ brings the
best of $T$ iterates within $RG/\sqrt T$ of the minimum. When the dimension
exceeds $T$, no method relying solely on subgradients can improve on this rate
by more than a constant factor :cite:`Nesterov.2018`. For a systematic
development of subdifferential calculus, including exact conditions under
which the sum rule holds with equality, see :citet:`Rockafellar.1970`.

## Derivatives of Vector and Matrix Maps

### The Mean Value Theorem for Vector-Valued Maps

In one dimension, the mean value theorem of
:numref:`sec_mdl-single_variable_calculus` guarantees an intermediate point
$\xi$ such that $f(b)-f(a)=f'(\xi)(b-a)$. It is tempting to assume the same
equality holds for vector-valued curves
$\boldsymbol\gamma:\mathbb R\to\mathbb R^n$, but it fails completely. Take the
unit circle $\boldsymbol\gamma(t)=(\cos t,\sin t)$: after a full turn at
$t=2\pi$, the curve returns to its starting point so
$\boldsymbol\gamma(2\pi)-\boldsymbol\gamma(0)=\mathbf 0$, yet the velocity
$\boldsymbol\gamma'(\xi)$ has unit length and is never $\mathbf 0$ for any
$\xi$. What survives in higher dimensions is only the *mean value inequality*,
$\|\boldsymbol\gamma(b)-\boldsymbol\gamma(a)\|\le\sup_\xi\|\boldsymbol\gamma'(\xi)\|\,|b-a|$
(Theorem 5.19 of :citet:`Rudin.1976`). Fortunately, the Taylor bounds in
:numref:`sec_mdl-multivariable_calculus` never need a vector version of the
theorem, because they restrict a scalar-valued loss to a line segment, where
the ordinary one-variable theorem applies directly.

### Jacobians of Matrix-Valued Maps

In :numref:`sec_mdl-matrix-calculus`, we differentiate scalar losses, so every
gradient conveniently has the same shape as its matrix variable. Once we
differentiate a matrix-valued function with respect to a matrix input, however,
the full derivative becomes a four-index array. A standard way to keep the
algebra linear and familiar is to flatten both matrices with the column-stacking
operator $\operatorname{vec}$, turning the four-index tensor into an ordinary
Jacobian matrix. Under this flattening, a matrix differential
$d\mathbf F=\mathbf A\,d\mathbf X\,\mathbf B$ becomes
$d\operatorname{vec}\mathbf F=(\mathbf B^\top\otimes\mathbf A)\,d\operatorname{vec}\mathbf X$,
and the chain rule simply multiplies Kronecker products. :citet:`Giles.2008b`
provides a handy catalog of forward and reverse differentiation rules for
matrix products, inverses, determinants, and standard matrix decompositions.

## Automatic Differentiation

### Optimal Jacobian Accumulation

As :numref:`sec_mdl-matrix-calculus` shows, simply changing the order in which
we multiply a chain of Jacobians can alter the computational cost by roughly a
factor of the layer width. For a straight chain of $K$ factors, finding the
cheapest parenthesization is the classical matrix-chain multiplication problem,
which dynamic programming solves in $O(K^3)$ time. A general computation graph
with branches and skip connections opens up a much larger space of elimination
orders: each intermediate node can be eliminated by multiplying the partial
derivatives on its incoming and outgoing edges, in any sequence. Finding the
elimination order with the fewest multiplications turns out to be NP-complete
:cite:`Naumann.2008`. Deep learning frameworks therefore do not attempt global
optimization of the elimination order; instead, they provide forward mode,
reverse mode, and their compositions, leaving the choice of mode to the user.
:citet:`Griewank.Walther.2008` develop vertex, edge, and face elimination in
full generality and review the heuristics used by derivative compilers.

### Recursive Checkpointing

The one-level checkpointing scheme in
:numref:`sec_mdl-matrix-calculus-autodiff` reduces peak activation memory to
$T/s+s$ stored states, reaching $2\sqrt T$ at the optimal block size $s$ at the
cost of one extra forward pass :cite:`Chen.Xu.Zhang.ea.2016`. For very long
sequences or unrolled iterative solvers, even $O(\sqrt T)$ memory can be too
much. Applying the same idea recursively inside each recomputed segment brings
memory all the way down to $O(\log T)$ while requiring $O(T\log T)$
recomputation :cite:`Griewank.1992`. When every step has the same cost,
Griewank's binomial checkpointing schedule is provably optimal for any fixed
memory budget :cite:`Griewank.Walther.2008`. In practice,
`torch.utils.checkpoint` and `jax.checkpoint` implement one level of
checkpointing per marked block; in JAX, nesting those blocks lets you build a
recursive schedule manually.

### Taylor Mode

Computing a $k$-th directional derivative by nesting forward-mode automatic
differentiation $k$ times is wasteful, because each nesting doubles the state
and carries $2^k$ components per value. *Taylor mode* avoids this exponential
blow-up by replacing dual numbers with truncated power series
$a_0+a_1\varepsilon+\cdots+a_k\varepsilon^k$ subject to $\varepsilon^{k+1}=0$.
Each elementary operation propagates all $k+1$ coefficients in a single pass:
multiplication convolves the two coefficient sequences in $O(k^2)$ operations,
and transcendental functions such as $\exp$, $\log$, and $\sin$ follow from
recurrences of the same $O(k^2)$ cost :cite:`Griewank.Walther.2008`. The
exercise on computing second derivatives in one forward pass in
:numref:`sec_mdl-matrix-calculus-autodiff` is the $k=2$ case, while
`jax.experimental.jet` implements Taylor mode for arbitrary order $k$.

### Forward Gradients

Reverse-mode differentiation requires storing a tape of intermediate values,
whereas forward mode avoids the tape but requires one pass per input parameter.
:citet:`Baydin.Pearlmutter.Syme.ea.2022` sidestep both costs by running a
single forward pass along a random Gaussian tangent
$\mathbf v\sim\mathcal N(\mathbf 0,\mathbf I)$ and taking
$(\nabla\ell^\top\mathbf v)\,\mathbf v$ as a stochastic gradient estimate.
Because $\mathbb{E}[\mathbf v\mathbf v^\top]=\mathbf I$, the estimate is
unbiased, and for $n$ parameters its mean squared error is
$(n+1)\|\nabla\ell\|^2$ (an exercise in
:numref:`sec_mdl-matrix-calculus-autodiff` asks you to derive both facts). The
memory tape disappears completely, but the trade-off is stark: the typical
estimation error is roughly $\sqrt n$ times the norm of the true gradient,
which is manageable for small models yet prohibitive at modern neural network
scales.

## Integration

### The Lebesgue Integral

The Riemann integral partitions the domain into intervals, a strategy that
breaks down on wildly discontinuous functions: as shown in
:numref:`sec_mdl-integral_calculus`, the indicator of the rational numbers has
tagged sums that never settle, and even a pointwise limit of
Riemann-integrable functions need not remain Riemann integrable. Lebesgue's
insight is to slice the range instead of the domain: measure the size of the
set where $f$ exceeds each height level, and add up those measures. Sets of
measure zero, such as the rational numbers or the ray $\theta=0$ in polar
coordinates, contribute nothing at all, so the indicator of the rationals
cleanly integrates to $0$. Both :numref:`sec_mdl-integral_calculus` and
:numref:`sec_mdl-expectation-gradients` invoke three workhorse theorems of
this theory without proof: Fubini–Tonelli, the $n$-dimensional change of
variables, and differentiation under an integral over an infinite domain. All
three rest on the monotone and dominated convergence theorems, which specify
when a limit may be swapped with an integral: if $f_k\to f$ pointwise and
$|f_k|\le g$ for a single integrable function $g$, then $\int f_k\to\int f$
:cite:`Folland.1999`.

### Quasi-Monte Carlo

In :numref:`subsec_mdl-monte-carlo`, we compared a second-order grid, whose
error decays as $N^{-2/d}$, with Monte Carlo integration, whose error decays
as $N^{-1/2}$ regardless of dimension. Purely random samples, however,
inevitably leave accidental clumps and empty gaps. *Low-discrepancy* point
sets (such as Sobol sequences, Halton sequences, and lattice rules) are
constructed to fill the unit cube far more evenly, achieving an error of
$O(N^{-1}(\log N)^d)$ for integrands of bounded variation (in the sense of
Hardy and Krause). Taken purely as a worst-case guarantee, this bound is
sobering: ignoring constants, it drops below $N^{-1/2}$ only past several
thousand points for $d=2$ and past $10^{15}$ points for $d=5$. In practice,
quasi-Monte Carlo still frequently outperforms ordinary Monte Carlo even in
high dimensions, usually because real integrands depend primarily on a small
number of effective coordinates at a time. Because a deterministic point set
provides no built-in error bar, *randomized quasi-Monte Carlo* randomly shifts
or scrambles the points so that independent replicates yield an unbiased
variance estimate. :citet:`Owen.2013` covers both deterministic and randomized
methods in Chapters 15–17.

### Gradient Estimators beyond Score and Pathwise

In :numref:`sec_mdl-expectation-gradients`, we derived the score-function and
pathwise estimators and reduced the variance of the score-function estimator
using a constant baseline. More sophisticated control variates exploit the
structure of the model itself: REBAR :cite:`Tucker.Mnih.Maddison.ea.2017`
subtracts the pathwise gradient of a continuous relaxation from the
score-function estimator and adds back its analytic expectation, lowering
variance while keeping the estimator unbiased for discrete random variables.
The continuous relaxations themselves, such as Gumbel-softmax
:cite:`Jang.Gu.Poole.2017,Maddison.Mnih.Teh.2017` and straight-through
estimators :cite:`Bengio.Leonard.Courville.2013`, offer low variance at the
cost of introducing bias relative to the original discrete objective. A third
approach, *measure-valued derivatives*, expresses the derivative of a
probability density as a scaled difference of two densities and estimates the
gradient by sampling from both. :citet:`Mohamed.Rosca.Figurnov.ea.2020` unite
all three families in a single framework and spell out the conditions under
which each applies.

[Discussions](https://d2l.discourse.group/t/36858)
