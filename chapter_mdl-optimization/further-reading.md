# Further Reading
:label:`sec_mdl-optimization-further-reading`

This section points beyond the chapter, and its topics are not all equally
urgent. Five of them bear directly on training neural networks, so we place
them first under their respective headings: the edge of stability, iterate
averaging, interpolation, the Fisher matrix, and saddle-point methods. The
remaining topics come from classical optimization and numerical analysis,
useful to have on your radar for when a specific problem calls for them.
Textbooks and courses covering the material the chapter does teach are
listed in its introduction (:numref:`chap_mdl-optimization`), while
:numref:`sec_optimization-further-reading` surveys the practical side of
training. Nothing else in the book depends on this section, so feel free to
browse selectively.

## Gradient Methods

### Self-Stabilization at the Edge of Stability

When you train a neural network with full-batch gradient descent, something
surprising happens: the sharpness of the loss (its largest Hessian
eigenvalue) climbs up to the stability limit $2/\eta$ and then hovers right
around it rather than diverging :cite:`Cohen.Kaur.Li.ea.2021`. The quadratic
model in :numref:`subsec_mdl-edge-of-stability` explains the first half of
this behavior, showing why the top eigenmode grows as soon as the sharpness
crosses $2/\eta$, yet a purely quadratic model predicts unbounded divergence
and cannot explain what pulls the sharpness back down.
:citet:`Damian.Nichani.Lee.2023` resolve the puzzle by keeping just one more
term in the Taylor expansion: the cubic term along the top eigenvector. Once
oscillations build along that direction, this cubic coupling pushes the
iterate against the gradient of the sharpness itself. That lowers the
sharpness back below $2/\eta$, letting the oscillation die down until the
cycle repeats. Without the cubic term, the oscillation would simply grow
unchecked.

### Optimal Polynomials and Conjugate Gradients

Consider minimizing a quadratic $f = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$
whose eigenvalues lie in $[\mu, L]$ and whose minimizer is $\mathbf{0}$. Any
first-order method whose $k$-th step stays inside the span of the gradients
seen so far produces an iterate of the form
$\mathbf{x}_k = p_k(\mathbf{A})\mathbf{x}_0$, where $p_k$ is a polynomial of
degree at most $k$ satisfying $p_k(0) = 1$. Viewed this way, designing a fast
gradient method on quadratics is equivalent to choosing a polynomial that is
as small as possible across the interval $[\mu, L]$. The polynomial that
achieves the smallest maximum over the whole interval is a scaled Chebyshev
polynomial, whose peak value is $2\rho^k/(1 + \rho^{2k}) \approx 2\rho^k$,
with the exact same contraction factor
$\rho = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ that appeared for heavy ball
(:numref:`subsec_mdl-momentum-acceleration`). When the dimension $d$ is
large compared with $k$, the eigenvalues can spread across the entire
interval, so no method in this family can beat that worst-case rate. The
conjugate gradient method does even better in practice: at every step it
automatically finds the polynomial that minimizes $f$ for the *actual*
matrix $\mathbf{A}$ and starting point $\mathbf{x}_0$, without ever needing
to know $\mu$ or $L$, and in exact arithmetic it terminates after at most
$d$ steps. :citet:`Nocedal.Wright.2006` (ch. 5) derive both the standard
method and its preconditioned variant.

### Rates Certified by Semidefinite Programs

Proving tight convergence rates by hand is famously delicate. Our hand proof
in :numref:`subsec_mdl-convergence-rates` yielded a per-step contraction
factor of $1 - \mu/L$, whereas the section quoted the exact factor
$(1 - \mu/L)^2$, which cuts the required step count in half. That tighter
bound comes from a technique called *performance estimation*. Instead of
searching for a clever algebraic trick by hand, :citet:`Drori.Teboulle.2014`
cast the worst-case performance of an algorithm over all $L$-smooth convex
functions as an optimization problem over the Gram matrix (the table of all
pairwise inner products) of its iterates and gradients. Bounding that
problem leads to a small semidefinite program, whose dual solution
literally produces a human-checkable proof.
:citet:`Taylor.Hendrickx.Glineur.2018` obtain the squared factor in just
this way. Taking a control-theoretic perspective,
:citet:`Lessard.Recht.Packard.2016` view the gradient as an unknown
nonlinearity inside a linear dynamical system constrained only by $\mu$ and
$L$, certifying convergence rates with a semidefinite program whose size
stays fixed regardless of the number of steps. Remarkably, the same
machinery also uncovers counterexamples: it is how they discovered the
function on which the heavy-ball method fails to converge
(:numref:`subsec_mdl-momentum-acceleration`).

## Stochastic and Adaptive Methods

### Averaging with a Constant Step

As we saw in :numref:`subsec_mdl-stochastic-gradients`, constant-step SGD
settles into a noise floor proportional to the step size $\eta$. Shrinking
$\eta$ over time eliminates that floor, but it also slows down late progress
and makes the schedule sensitive to tuning. Averaging the iterates offers a
cleaner way out: the individual iterates keep bouncing around the minimum,
while their running average cancels the zero-mean noise.
:citet:`Polyak.Juditsky.1992` show that averaging iterates obtained with
step sizes decaying more slowly than $1/k$ achieves the asymptotically
optimal covariance. For least squares, :citet:`Bach.Moulines.2013` go one
step further and show that a *constant* step size is already enough: the
averaged iterate reaches an error of order $1/n$ after $n$ examples, even
without strong convexity.

### Interpolation

Classical stochastic optimization assumes that individual training examples
pull the parameters in conflicting directions at the optimum, creating an
unavoidable noise floor. Modern overparameterized neural networks often
operate in a very different regime: when a model fits every training example
exactly, every per-example gradient vanishes at the minimizer. The noise
floor then disappears, and on a strongly convex loss constant-step SGD
converges linearly without any variance reduction
(:numref:`sec_mdl-variance-reduction`). :citet:`Ma.Bassily.Belkin.2018`
analyze this interpolation regime for least squares and identify the
critical batch size beyond which larger minibatches stop speeding up
training; intriguingly, that threshold is governed by the ratio of
per-example curvature to full-batch curvature rather than by gradient noise.
Under a related *strong growth condition*, where the gradient variance
shrinks in proportion to the squared norm of the full gradient, SGD can even
be accelerated to match momentum rates :cite:`Vaswani.Bach.Schmidt.2019`.

### The Fisher and the Gauss–Newton Matrix

Second-order approximations such as K-FAC
(:numref:`subsec_mdl-preconditioning-ladder`) aim to approximate the Fisher
information matrix
$\mathbf{F} = \mathbb{E}[\nabla\log p\,\nabla\log p^\top]$, where the
expectation draws labels from the model's own predictive distribution
:cite:`Amari.1998,Martens.Grosse.2015`. For standard losses like softmax
cross-entropy and squared error, $\mathbf{F}$ coincides with the
Gauss–Newton matrix $\mathbf{J}^\top\mathbf{H}_{\mathrm{out}}\mathbf{J}$,
where $\mathbf{H}_{\mathrm{out}}$ is the Hessian of the loss with respect to
the network's output. More generally, this identity holds whenever the
network output parameterizes the natural parameter of an exponential-family
likelihood (:numref:`sec_mdl-exponential-family`). A common pitfall in
practice is substituting the *empirical* Fisher, which averages the same
outer product over the ground-truth dataset labels instead of sampling from
the model. It is cheaper to compute and is often labeled "Fisher" in
codebases, yet in general it equals neither the true Fisher nor the
Gauss–Newton matrix :cite:`Martens.2020`.

### Mirror Descent

In the metric view of AdaGrad (:numref:`subsec_mdl-per-coordinate`), step
lengths are measured in a quadratic Euclidean norm. That geometry fits
unconstrained spaces like $\mathbb{R}^d$, where all directions look alike,
but it is awkward on the probability simplex, where coordinates must stay
nonnegative and sum to one and multiplicative updates are far more natural.
Mirror descent adapts the geometry of gradient descent by replacing the
squared distance $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ in the proximal
form of the gradient step with a Bregman divergence
$D_\phi(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x}) - \phi(\mathbf{y}) - \nabla\phi(\mathbf{y})^\top(\mathbf{x} - \mathbf{y})$
:cite:`Nemirovski.Yudin.1983,Beck.Teboulle.2003`. Choosing $\phi$ to be the
negative entropy turns the update into the exponentiated gradient rule
$x_i \leftarrow x_i e^{-\eta g_i} / \sum_j x_j e^{-\eta g_j}$. When
gradients are bounded coordinatewise, this choice improves the regret
defined in :numref:`subsec_mdl-per-coordinate` on the $d$-dimensional
simplex from the $\sqrt{d}$ scaling of projected gradient descent to a much
gentler $\sqrt{\log d}$.

## Constraints and Duality

### Saddle-Point Methods

The saddle points we met in Lagrangian duality
(:numref:`subsec_mdl-strong-duality`) also arise directly as the equilibria
of two-player games, including GAN training (:numref:`sec_basic_gan`).
Unfortunately, the naive strategy of simultaneous gradient descent on one
player and gradient ascent on the other behaves badly: on the simple
bilinear game $xy$, it spirals outward from the saddle point for every
positive step size (exercise
:exref:`ex_mdl-lagrangian-duality-descent-ascent`). The extragradient method
breaks this spiral by peeking ahead: it takes a trial step, evaluates the
gradient at that lookahead point, and applies that gradient from the
original iterate. On convex–concave problems with Lipschitz gradients, its
averaged iterates converge at the rate $O(1/k)$ :cite:`Nemirovski.2004`.
Optimistic methods achieve a similar stabilizing effect at half the gradient
cost by reusing the previous step's gradient as a cheap proxy for the trial
gradient, an idea that :citet:`Daskalakis.Ilyas.Syrgkanis.ea.2018` apply
successfully to stabilize GAN training.

### Problem Classes and Interior-Point Methods

The dual problems we solved by hand in :numref:`subsec_mdl-worked-duals` are
quadratic programs, which sit inside a standard nested hierarchy of convex
optimization classes: linear programs, quadratic programs, second-order cone
programs, and semidefinite programs. Modern interior-point methods solve all
of these classes to high accuracy in polynomial time. Rather than handling
the inequalities $g_i(\mathbf{x}) \le 0$ directly, they add a smooth
logarithmic barrier $-\tfrac1s\sum_i \log(-g_i(\mathbf{x}))$ to the
objective and track the unconstrained minimizers as $s$ grows, tracing out
the *central path* toward the boundary. At each point along this path, the
quantities $-1/(s\,g_i(\mathbf{x}))$ automatically form dual-feasible
multipliers, certifying a duality gap of $m/s$ for $m$ constraints. This
gives an automated version of the optimality certificates we built by hand in
:numref:`sec_mdl-lagrangian-duality`
:cite:`Boyd.Vandenberghe.2004,Nesterov.2018`.

### Operator Splitting and ADMM

The proximal gradient steps of :numref:`subsec_mdl-proximal-operators`
handle a smooth loss plus a single simple nonsmooth penalty. Many structured
problems instead have the composite form
$\min_{\mathbf{x}} f(\mathbf{x}) + h(\mathbf{A}\mathbf{x})$, where both
terms are convex and possibly nonsmooth; its Fenchel dual is
$\max_{\mathbf{y}} -f^*(-\mathbf{A}^\top\mathbf{y}) - h^*(\mathbf{y})$
:cite:`Rockafellar.1970`. To decouple the two terms so each can be handled
separately, we introduce an auxiliary variable $\mathbf{z} = \mathbf{A}\mathbf{x}$
and add a quadratic penalty $\tfrac{\tau}{2}\|\mathbf{A}\mathbf{x} - \mathbf{z}\|^2$
to the Lagrangian. Minimizing alternately over $\mathbf{x}$, then over
$\mathbf{z}$ (which reduces to a proximal step on $h$), and finally taking a
dual ascent step on $\mathbf{y}$ gives the alternating direction method of
multipliers (ADMM), which converges for every $\tau > 0$.
:citet:`Boyd.Parikh.Chu.ea.2011` show how this splitting naturally
distributes large optimization problems across machines, and
:citet:`Parikh.Boyd.2014` provide a handy catalog of closed-form proximal
operators for common regularizers and constraints.

### The Frank–Wolfe Method

Projected gradient descent requires projecting onto the constraint set at
every iteration. That is effortless for a box or an $\ell_2$ ball, but on a
nuclear-norm ball (matrices whose singular values sum to at most a given
bound) a single projection costs a full SVD. The Frank–Wolfe method
:cite:`Frank.Wolfe.1956` avoids projection altogether by minimizing a linear
approximation of the objective over the constraint set instead: it finds
$\mathbf{s}_k = \operatorname{argmin}_{\mathbf{s} \in C} \nabla f(\mathbf{x}_k)^\top\mathbf{s}$
and takes a convex combination
$\mathbf{x}_{k+1} = (1 - \gamma_k)\mathbf{x}_k + \gamma_k\mathbf{s}_k$
with step size $\gamma_k = 2/(k + 2)$. On the nuclear-norm ball this linear
subproblem needs only the top singular vector pair of the gradient, and on
the probability simplex it reduces to finding the single smallest
coordinate. The method converges at rate $O(LD^2/k)$ on a set of diameter
$D$, and as a bonus the quantity
$\nabla f(\mathbf{x}_k)^\top(\mathbf{x}_k - \mathbf{s}_k)$ provides an
upper bound on $f(\mathbf{x}_k) - f^\star$ at every step with no extra
computation :cite:`Jaggi.2013`.

## Floating Point

### Probabilistic Rounding-Error Analysis

The worst-case summation bounds in
:numref:`subsec_mdl-catastrophic-cancellation` grow linearly as $nu$ when
$n$ terms are added in sequence, yet measured rounding errors in practice
are usually much smaller. The reason is that worst-case analysis assumes
every single rounding error pushes in the same adversarial direction.
:citet:`Higham.Mary.2019` instead model the individual rounding errors as
independent, zero-mean random variables and prove bounds of order
$\sqrt{n}\,u$ that hold with high probability, matching the random-walk
cancellation typically seen in experiments. Just as importantly, their
analysis clarifies when the optimistic $\sqrt{n}\,u$ behavior breaks down:
whenever the rounding errors are strongly correlated, as in that
subsection's sum of $10^7$ copies of $0.1$, where long stretches of
additions all round in the same direction.

### Growth Factors and Iterative Refinement

In :numref:`subsec_mdl-conditioning-revisited`, we noted that LU
factorization with partial pivoting is backward stable in practice. Strictly
speaking, its backward error is bounded by a low-degree polynomial in $n$
times $\rho_n u$, where the *growth factor* $\rho_n$ measures the ratio of
the largest intermediate entry created during elimination to the largest
entry of $\mathbf{A}$. In the worst case, Wilkinson's famous pathological
matrix drives $\rho_n$ all the way to $2^{n-1}$. Yet for almost every matrix
encountered in real computations $\rho_n$ remains modest, and if you need an
ironclad guarantee, QR factorization eliminates exponential growth
altogether at roughly twice the arithmetic cost
:cite:`Higham.2002,Trefethen.Bau.1997`. To recover high accuracy without
paying for double precision everywhere, mixed-precision iterative refinement
factors $\mathbf{A}$ cheaply in fp32, evaluates the residual in fp64, and
solves for a correction using the low-precision factors; the final exercise
of :numref:`sec_mdl-numerical-stability-conditioning` determines just how
ill conditioned $\mathbf{A}$ can get before this basic loop stalls. Using
the low-precision factors instead as a preconditioner inside GMRES, an
iterative solver for general linear systems, pushes mixed-precision
refinement to far more severely conditioned matrices
:cite:`Carson.Higham.2018`.

[Discussions](https://d2l.discourse.group/t/36861)
