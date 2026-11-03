# Further Reading
:label:`sec_mdl-optimization-further-reading`

The sections of this chapter count the steps of gradient methods on smooth
and convex problems, add noise, per-coordinate scaling and constraints, and
ask what rounding does to the answer. This section collects, in the order of
the chapter, topics they had no room for: optimal methods on quadratics and
machine-checked rates, refinements of stochastic and adaptive methods,
solvers for constrained and saddle-point problems, and two results on
floating-point error. Each topic states the problem it addresses and what it
changes relative to the chapter. Resources for the material the chapter does
teach are listed in its introduction (:numref:`chap_mdl-optimization`), and
the practice of training optimizers is surveyed in
:numref:`sec_optimization-further-reading`. Nothing else in the book depends
on this section.

## Gradient Methods

### Optimal Polynomials and Conjugate Gradients

On the quadratic $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$, every method
whose $k$-th step lies in the span of the gradients seen so far leaves the
error $p_k(\mathbf{A})\mathbf{e}_0$, for some polynomial $p_k$ of degree $k$
with $p_k(0) = 1$. Asking which such polynomial is smallest on the whole
interval $[\mu, L]$ gives a scaled Chebyshev polynomial, whose maximum there
is $2\rho^k/(1 + \rho^{2k})$ with $\rho = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$.
In high dimension no method of this kind beats, in the worst case, the
heavy-ball rate of :numref:`subsec_mdl-momentum-acceleration`. Conjugate
gradients picks the best polynomial for the actual eigenvalues of
$\mathbf{A}$, without knowing $\mu$ or $L$, and in exact arithmetic stops
after at most $d$ steps; :citet:`Nocedal.Wright.2006` (ch. 5) derive it and
its preconditioned form.

### Rates Certified by Semidefinite Programs

The rate proofs of :numref:`subsec_mdl-convergence-rates` chain two
inequalities by hand and give the factor $1 - \mu/L$ per step for
$f - f^\star$. :citet:`Drori.Teboulle.2014` write the worst case over all
$L$-smooth convex functions as a small semidefinite program over the Gram
matrix of the iterates and gradients. Its value is the exact worst-case
rate, and a dual solution is a proof. For gradient descent with $\eta = 1/L$
on $\mu$-strongly convex functions this *performance estimation* gives
$(1 - \mu/L)^2$ :cite:`Taylor.Hendrickx.Glineur.2018`.
:citet:`Lessard.Recht.Packard.2016` treat the gradient as a nonlinearity in
a feedback loop, constrained only by $\mu$ and $L$, and certify rates by a
semidefinite program whose size does not grow with the number of steps. The
method recovers Nesterov's rate and finds strongly convex functions on which
heavy ball, tuned for quadratics, fails to converge.

### Self-Stabilization at the Edge of Stability

In :numref:`subsec_mdl-quadratic-model` the sharpness of a network's loss
rises to $2/\eta$ and stays there :cite:`Cohen.Kaur.Li.ea.2021`. The
quadratic model explains why the top mode grows while the sharpness exceeds
$2/\eta$, but not what brings the sharpness back down.
:citet:`Damian.Nichani.Lee.2023` keep the cubic term of the Taylor expansion
along the top eigenvector. The growing oscillation along that direction then
moves the iterate against the gradient of the sharpness, which lowers it, so
the sharpness and the oscillation cycle around $2/\eta$. Without the cubic
term the oscillation simply grows.

## Stochastic and Adaptive Methods

### Averaging with a Constant Step

The noise floor of :numref:`subsec_mdl-stochastic-gradients` is proportional
to $\eta$. A decaying step removes it but makes late progress slow. Averaging
the iterates removes it differently. :citet:`Polyak.Juditsky.1992` show
that the average of iterates with steps decaying more slowly than $1/k$
reaches the asymptotically optimal covariance.
:citet:`Bach.Moulines.2013` show that for least squares a *constant* step
suffices: the averaged iterate reaches an error of order $1/n$ after $n$
examples, without strong convexity. When to start averaging, and over which
window, remain practical choices.

### Interpolation

The floor exists because the per-example gradients do not vanish at the
minimizer; its size is set by the gradient covariance there. When the model
fits every example exactly, as over-parameterized networks and consistent
least squares do, that covariance is zero, and constant-step SGD converges
linearly, as gradient descent does :cite:`Ma.Bassily.Belkin.2018`. The batch
size beyond which larger batches stop helping is then set by the ratio of
per-example to full-batch curvature rather than by the noise. Under a *strong
growth condition*, in which the gradient noise shrinks with the full
gradient, SGD can even be accelerated :cite:`Vaswani.Bach.Schmidt.2019`.

### Mirror Descent

AdaGrad's metric view (:numref:`subsec_mdl-per-coordinate`) measures step
length in a quadratic norm. On the probability simplex the natural steps are
multiplicative. Mirror descent replaces $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$
in the proximal form of the gradient step by a Bregman divergence
$D_\phi(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x}) - \phi(\mathbf{y}) - \nabla\phi(\mathbf{y})^\top(\mathbf{x} - \mathbf{y})$
:cite:`Nemirovski.Yudin.1983,Beck.Teboulle.2003`. With $\phi$ the negative
entropy, the update becomes the exponentiated gradient
$x_i \leftarrow x_i e^{-\eta g_i} / \sum_j x_j e^{-\eta g_j}$. For gradients
bounded coordinatewise, the regret on the simplex in $d$ dimensions then grows
with $\sqrt{\log d}$ instead of the $\sqrt{d}$ of projected gradient steps.

### The Fisher and the Gauss–Newton Matrix

The structured preconditioners of :numref:`subsec_mdl-preconditioning-ladder`
approximate a curvature matrix. K-FAC's is the Fisher
$\mathbf{F} = \mathbb{E}[\nabla\log p\,\nabla\log p^\top]$, with the labels
drawn from the model :cite:`Amari.1998,Martens.Grosse.2015`. The Fisher
equals the Gauss–Newton matrix $\mathbf{J}^\top\mathbf{H}_{\mathrm{out}}\mathbf{J}$,
with $\mathbf{H}_{\mathrm{out}}$ the Hessian of the loss in the network's
output, when that output is the natural parameter of an exponential-family
likelihood, as for softmax cross-entropy and squared loss with Gaussian
noise. For other likelihoods the two differ, and the *empirical* Fisher,
built from the labels of the data, equals neither in general
:cite:`Martens.2020`.

## Constraints and Duality

### Problem Classes and Interior-Point Methods

The worked duals of :numref:`subsec_mdl-worked-duals` are quadratic
programs. Convex problems come in nested classes: linear programs, quadratic
programs, second-order cone programs with constraints
$\|\mathbf{A}\mathbf{x} + \mathbf{b}\| \le \mathbf{c}^\top\mathbf{x} + e$, and
semidefinite programs over a matrix variable $\mathbf{X} \succeq \mathbf{0}$.
Interior-point methods solve each to high accuracy in polynomial time. They
replace the constraints $g_i(\mathbf{x}) \le 0$ by the barrier
$-\tfrac1s\sum_i \log(-g_i(\mathbf{x}))$ and follow the minimizers as $s$
grows, along the *central path*. At each such minimizer the numbers
$-1/(s\,g_i(\mathbf{x}))$ are dual feasible multipliers, so the duality gap
is $m/s$ for $m$ constraints, a certificate of the kind
:numref:`sec_mdl-lagrangian-duality` builds by hand
:cite:`Boyd.Vandenberghe.2004,Nesterov.2018`.

### Operator Splitting and ADMM

The proximal steps of :numref:`subsec_mdl-proximal-operators` handle a
smooth loss plus one simple nonsmooth term. Problems of the form
$\min_{\mathbf{x}} f(\mathbf{x}) + h(\mathbf{A}\mathbf{x})$, with both terms
convex and possibly nonsmooth, have the Fenchel dual
$\max_{\mathbf{y}} -f^*(-\mathbf{A}^\top\mathbf{y}) - h^*(\mathbf{y})$
:cite:`Rockafellar.1970`. Introducing $\mathbf{z} = \mathbf{A}\mathbf{x}$
and a quadratic penalty $\tfrac{\rho}{2}\|\mathbf{A}\mathbf{x} - \mathbf{z}\|^2$
in the Lagrangian gives the alternating direction method of multipliers. It
minimizes over $\mathbf{x}$, then over $\mathbf{z}$ (a proximal step on $h$),
then takes a dual ascent step on $\mathbf{y}$, and it converges for every
$\rho > 0$. :citet:`Boyd.Parikh.Chu.ea.2011` use it to split problems across
machines; :citet:`Parikh.Boyd.2014` list closed-form proximal operators for
common functions.

### The Frank–Wolfe Method

Projected gradient descent needs a projection, which onto a nuclear-norm ball
costs a full SVD. The Frank–Wolfe method :cite:`Frank.Wolfe.1956` minimizes
the linearization over the set instead,
$\mathbf{s}_k = \operatorname{argmin}_{\mathbf{s} \in C} \nabla f(\mathbf{x}_k)^\top\mathbf{s}$,
and moves to $\mathbf{x}_{k+1} = (1 - \gamma_k)\mathbf{x}_k + \gamma_k\mathbf{s}_k$
with $\gamma_k = 2/(k + 2)$. Over the nuclear-norm ball this needs only the
top singular pair of the gradient, and over the simplex only its smallest
coordinate, so the iterates are combinations of few vertices. The rate is
$O(LD^2/k)$ on a set of diameter $D$, and the number
$\nabla f(\mathbf{x}_k)^\top(\mathbf{x}_k - \mathbf{s}_k)$, an upper bound on
$f(\mathbf{x}_k) - f^\star$, comes free with every step :cite:`Jaggi.2013`.

### Saddle-Point Methods

The saddle points of :numref:`subsec_mdl-strong-duality` are also the
solutions of two-player games such as GAN training. Simultaneous gradient
descent and ascent spirals away from the saddle of $xy$ for every step size.
The extragradient method takes a trial step, evaluates the gradient there,
and uses that gradient for the actual step from the original point. On
convex–concave problems with Lipschitz gradients its averaged iterates
converge at the rate $O(1/k)$ :cite:`Nemirovski.2004`. Optimistic methods
reuse the previous gradient in place of the trial one, which saves a gradient
evaluation per step, and :citet:`Daskalakis.Ilyas.Syrgkanis.ea.2018` apply
them to GANs.

## Floating Point

### Probabilistic Rounding-Error Analysis

The summation bounds of :numref:`subsec_mdl-catastrophic-cancellation` grow
like $nu$ for $n$ terms added in order, while measured errors are usually
far smaller. :citet:`Higham.Mary.2019` model the rounding errors as
independent random variables with mean zero and obtain bounds of order
$\sqrt{n}\,u$ that hold with high probability. The model describes typical
data. It fails when the rounding errors are correlated, as when many equal
small terms are added to a large running sum and every addition rounds the
same way.

### Growth Factors and Iterative Refinement

:numref:`subsec_mdl-conditioning-revisited` calls LU factorization with
partial pivoting backward stable in practice. Its backward error is bounded
by a modest multiple of $\rho_n u$, where the *growth factor* $\rho_n$ is the
largest entry produced during elimination divided by the largest entry of
$\mathbf{A}$. Wilkinson's example reaches $\rho_n = 2^{n-1}$, yet $\rho_n$
stays small for almost all matrices met in practice, and QR factorization
removes the worst case at about twice the cost
:cite:`Higham.2002,Trefethen.Bau.1997`. The mixed-precision refinement of
:numref:`sec_mdl-numerical-stability-conditioning` converges only when
$\kappa(\mathbf{A})\,u_{32}$ is well below one. Solving each correction with
GMRES, preconditioned by the low-precision LU factors, extends it to far
worse conditioned matrices, which lets half- and single-precision
factorizations on GPUs deliver double-precision solutions
:cite:`Carson.Higham.2018`.

[Discussions](https://d2l.discourse.group/t/36861)
