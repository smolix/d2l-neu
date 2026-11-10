# Further Reading
:label:`sec_mdl-optimization-further-reading`

This section points past the chapter, and its topics are not of equal
weight. Five bear directly on training networks and come first under their
headings: the edge of stability, iterate averaging, interpolation, the
Fisher matrix, and saddle-point methods. The others are classical
optimization and numerical analysis, for when a problem calls for them.
Textbooks and courses for what the chapter does teach are listed in its
introduction (:numref:`chap_mdl-optimization`), and
:numref:`sec_optimization-further-reading` surveys the practice of training.
Nothing else in the book depends on this section.

## Gradient Methods

### Self-Stabilization at the Edge of Stability

Under full-batch gradient descent the sharpness of a network's loss rises to
$2/\eta$ and stays there :cite:`Cohen.Kaur.Li.ea.2021`. The quadratic model
of :numref:`subsec_mdl-edge-of-stability` explains why the top mode grows
while the sharpness exceeds $2/\eta$, but not what brings the sharpness back
down. :citet:`Damian.Nichani.Lee.2023` keep the cubic term of the Taylor
expansion along the top eigenvector. The growing oscillation along that
direction then moves the iterate against the gradient of the sharpness,
which lowers it, so the sharpness and the oscillation cycle around $2/\eta$.
Without the cubic term the oscillation simply grows.

### Optimal Polynomials and Conjugate Gradients

Take the quadratic $f = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ with
eigenvalues in $[\mu, L]$ and minimizer $\mathbf{0}$. Every method whose
$k$-th step lies in the span of the gradients seen so far arrives at
$\mathbf{x}_k = p_k(\mathbf{A})\mathbf{x}_0$, for some polynomial $p_k$ of
degree at most $k$ with $p_k(0) = 1$. The polynomial that is smallest on
the whole interval is a scaled Chebyshev polynomial. Its maximum there is
$2\rho^k/(1 + \rho^{2k})$, about $2\rho^k$, with the
$\rho = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ of heavy ball
(:numref:`subsec_mdl-momentum-acceleration`). When the dimension $d$ is
large compared with $k$, the eigenvalues can fill the interval, and no
method of this kind then beats that rate in the worst case. Conjugate
gradients finds at every step the polynomial that minimizes $f$ for the
actual $\mathbf{A}$ and $\mathbf{x}_0$, without knowing $\mu$ or $L$, and in
exact arithmetic stops after at most $d$ steps;
:citet:`Nocedal.Wright.2006` (ch. 5) derive it and its preconditioned form.

### Rates Certified by Semidefinite Programs

The hand proof of :numref:`subsec_mdl-convergence-rates` gives the factor
$1 - \mu/L$ per step, and the section quotes the exact $(1 - \mu/L)^2$,
which halves the step count. The exact value comes from *performance
estimation*. :citet:`Drori.Teboulle.2014` write the worst case of a method
over all $L$-smooth convex functions as an optimization problem over the
Gram matrix (all inner products) of its iterates and gradients, and bound it
by a small semidefinite program, whose dual solution is a proof.
:citet:`Taylor.Hendrickx.Glineur.2018` obtain the squared factor in this
way. :citet:`Lessard.Recht.Packard.2016` treat the gradient as an unknown
function inside a linear recursion, constrained only by $\mu$ and $L$, and
certify rates by a semidefinite program whose size does not grow with the
number of steps. This is how they found the function on which heavy ball
fails (:numref:`subsec_mdl-momentum-acceleration`).

## Stochastic and Adaptive Methods

### Averaging with a Constant Step

The noise floor of :numref:`subsec_mdl-stochastic-gradients` is proportional
to $\eta$. A decaying step removes it but makes late progress slow. Averaging
the iterates removes it differently. :citet:`Polyak.Juditsky.1992` show
that the average of iterates with steps decaying more slowly than $1/k$
reaches the asymptotically optimal covariance.
:citet:`Bach.Moulines.2013` show that for least squares a *constant* step
suffices: the averaged iterate reaches an error of order $1/n$ after $n$
examples, without strong convexity.

### Interpolation

When the model fits every example exactly, the per-example gradients vanish
at the minimizer, the floor disappears, and on a strongly convex loss
constant-step SGD converges linearly (:numref:`sec_mdl-variance-reduction`).
:citet:`Ma.Bassily.Belkin.2018` analyze this regime and find, for least
squares, the batch size beyond which larger batches stop helping. It is set
by the ratio of per-example to full-batch curvature rather than by the
noise. Under a *strong growth condition*, in which the gradient noise
shrinks with the full gradient, SGD can even be accelerated
:cite:`Vaswani.Bach.Schmidt.2019`.

### The Fisher and the Gauss–Newton Matrix

K-FAC (:numref:`subsec_mdl-preconditioning-ladder`) approximates the Fisher
matrix $\mathbf{F} = \mathbb{E}[\nabla\log p\,\nabla\log p^\top]$, with the
labels drawn from the model :cite:`Amari.1998,Martens.Grosse.2015`. For
softmax cross-entropy and squared loss, $\mathbf{F}$ equals the Gauss–Newton
matrix $\mathbf{J}^\top\mathbf{H}_{\mathrm{out}}\mathbf{J}$, with
$\mathbf{H}_{\mathrm{out}}$ the Hessian of the loss in the network's output.
In general this needs the output to be the natural parameter of an
exponential-family likelihood (:numref:`sec_mdl-exponential-family`). The
*empirical* Fisher, built from the labels of the data, is cheaper and is
often what code calls the Fisher, but it equals neither matrix in general
:cite:`Martens.2020`.

### Mirror Descent

AdaGrad's metric view (:numref:`subsec_mdl-per-coordinate`) measures step
length in a quadratic norm. On the probability simplex the natural steps are
multiplicative. Mirror descent replaces $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$
in the proximal form of the gradient step by a Bregman divergence
$D_\phi(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x}) - \phi(\mathbf{y}) - \nabla\phi(\mathbf{y})^\top(\mathbf{x} - \mathbf{y})$
:cite:`Nemirovski.Yudin.1983,Beck.Teboulle.2003`. With $\phi$ the negative
entropy, the update becomes the exponentiated gradient
$x_i \leftarrow x_i e^{-\eta g_i} / \sum_j x_j e^{-\eta g_j}$. For gradients
bounded coordinatewise, the regret defined in that subsection then grows on
the simplex in $d$ dimensions with $\sqrt{\log d}$ instead of the $\sqrt{d}$
of projected gradient steps.

## Constraints and Duality

### Saddle-Point Methods

The saddle points of :numref:`subsec_mdl-strong-duality` are also the
solutions of two-player games such as GAN training
(:numref:`sec_basic_gan`). Simultaneous gradient descent and ascent spirals
away from the saddle of $xy$ for every step size (exercise
:exref:`ex_mdl-lagrangian-duality-descent-ascent`). The extragradient method
takes a trial step, evaluates the gradient there, and uses that gradient for
the actual step from the original point. On convex–concave problems with
Lipschitz gradients its averaged iterates converge at the rate $O(1/k)$
:cite:`Nemirovski.2004`. Optimistic methods reuse the previous gradient in
place of the trial one, which saves a gradient evaluation per step, and
:citet:`Daskalakis.Ilyas.Syrgkanis.ea.2018` apply them to GANs.

### Problem Classes and Interior-Point Methods

The worked duals of :numref:`subsec_mdl-worked-duals` are quadratic
programs. Convex problems come in nested classes: linear programs, quadratic
programs, second-order cone programs, and semidefinite programs.
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
and a quadratic penalty $\tfrac{\tau}{2}\|\mathbf{A}\mathbf{x} - \mathbf{z}\|^2$
in the Lagrangian gives the alternating direction method of multipliers. It
minimizes over $\mathbf{x}$, then over $\mathbf{z}$ (a proximal step on $h$),
then takes a dual ascent step on $\mathbf{y}$, and it converges for every
$\tau > 0$. :citet:`Boyd.Parikh.Chu.ea.2011` use it to split problems across
machines; :citet:`Parikh.Boyd.2014` list closed-form proximal operators for
common functions.

### The Frank–Wolfe Method

Projected gradient descent needs a projection, which onto a nuclear-norm ball
(matrices whose singular values sum to at most a bound) costs a full SVD.
The Frank–Wolfe method :cite:`Frank.Wolfe.1956` minimizes
the linearization over the set instead,
$\mathbf{s}_k = \operatorname{argmin}_{\mathbf{s} \in C} \nabla f(\mathbf{x}_k)^\top\mathbf{s}$,
and moves to $\mathbf{x}_{k+1} = (1 - \gamma_k)\mathbf{x}_k + \gamma_k\mathbf{s}_k$
with $\gamma_k = 2/(k + 2)$. Over the nuclear-norm ball this needs only the
top singular pair of the gradient, and over the simplex only its smallest
coordinate. The rate is
$O(LD^2/k)$ on a set of diameter $D$, and the number
$\nabla f(\mathbf{x}_k)^\top(\mathbf{x}_k - \mathbf{s}_k)$, an upper bound on
$f(\mathbf{x}_k) - f^\star$, comes free with every step :cite:`Jaggi.2013`.

## Floating Point

### Probabilistic Rounding-Error Analysis

The summation bounds of :numref:`subsec_mdl-catastrophic-cancellation` grow
like $nu$ for $n$ terms added in order, while measured errors are usually
far smaller. :citet:`Higham.Mary.2019` model the rounding errors as
independent random variables with mean zero and obtain bounds of order
$\sqrt{n}\,u$ that hold with high probability. The model fails when the
rounding errors are correlated, as in that subsection's sum of $10^7$ copies
of $0.1$, where long runs of additions round the same way.

### Growth Factors and Iterative Refinement

:numref:`subsec_mdl-conditioning-revisited` calls LU factorization with
pivoting backward stable in practice. Its backward error is bounded by a
low-degree polynomial in $n$ times $\rho_n u$, where the *growth factor*
$\rho_n$ is the largest entry produced during elimination divided by the
largest entry of $\mathbf{A}$. Wilkinson's example reaches
$\rho_n = 2^{n-1}$, yet $\rho_n$ stays small for almost all matrices met in
practice, and QR factorization removes the worst case at about twice the
cost :cite:`Higham.2002,Trefethen.Bau.1997`. Mixed-precision iterative
refinement factors $\mathbf{A}$ in fp32, computes residuals in fp64, and
solves for corrections with the cheap factors; the last exercise of
:numref:`sec_mdl-numerical-stability-conditioning` finds how ill conditioned
$\mathbf{A}$ may be before it fails. Solving each correction with GMRES, an
iterative solver for general linear systems, preconditioned by the
low-precision factors, extends it to far worse conditioned matrices
:cite:`Carson.Higham.2018`.

[Discussions](https://d2l.discourse.group/t/36861)
