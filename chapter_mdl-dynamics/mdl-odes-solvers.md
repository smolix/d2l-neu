# Ordinary Differential Equations
:label:`sec_mdl-odes-solvers`

<!-- WRITER: new opening (structure.md, 38.1). The old opening (the bowl 25x^2, Euler 1768, the three places)
moved to mdl-numerical-solvers.md, and the notation table moved to index.md. The paragraph below is the
old roadmap of the combined section; its "two parts" and the third result now belong to 38.2. -->

The text has two parts. The first asks when a differential equation has
exactly one solution and solves the linear case completely. The second
(:numref:`sec_mdl-euler-runge-kutta`) asks how accurate and how stable a
step-by-step solution is. Three results carry the rest of the chapter: a
Lipschitz field has exactly one trajectory through each point, so its flow
is invertible; the real parts of the eigenvalues of a linear field decide
whether its motion decays; and Euler's method is stable only while $h\lambda$
stays in a disc. The pendulum and transient growth can be skipped on a first
reading.

```{.python .input #odes-solvers-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #odes-solvers-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #odes-solvers-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #odes-solvers-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Vector Fields and Flows
:label:`sec_mdl-vector-fields-trajectories`

A differential equation describes a motion by prescribing a velocity at
every point. Two motions can be solved by hand, and we begin with them.

### Decay, Rotation, and the Spiral

The first is *decay*. The rule "move toward zero at a speed proportional to
the distance" is the equation $\dot x=-x$, where the dot is the derivative in
time. Its solution from $x(0)=x_0$ is

$$
x(t)=e^{-t}x_0,
$$

as differentiating shows: $\dot x=-e^{-t}x_0=-x$. Each unit of time
multiplies the distance to the origin by $e^{-1}\approx0.37$.

The second is *rotation*. In the plane, let the velocity be the position
turned by a right angle,

$$
\dot{\mathbf x}=\mathbf K\mathbf x,
\qquad
\mathbf K=\begin{pmatrix}0&-1\\1&0\end{pmatrix},
$$

that is $\dot x_1=-x_2$ and $\dot x_2=x_1$. The velocity is perpendicular to
the position, because $\mathbf x^\top\mathbf K\mathbf x=-x_1x_2+x_2x_1=0$, so
the length of $\mathbf x$ never changes:

$$
\frac{d}{dt}\|\mathbf x\|^2=2\,\mathbf x^\top\dot{\mathbf x}
=2\,\mathbf x^\top\mathbf K\mathbf x=0 .
$$

A point moves on a circle, and since its speed $\|\mathbf K\mathbf x\|$
equals its radius $\|\mathbf x\|$ it turns at one radian per unit of time:
one revolution takes $2\pi$.

Adding the two motions gives the field we use as the running example of the
chapter,

$$
\dot{\mathbf x}=\mathbf A\mathbf x,
\qquad
\mathbf A=-\tfrac12\mathbf I+\mathbf K
=\begin{pmatrix}-0.5&-1\\1&-0.5\end{pmatrix}.
$$

The rotation part does not change the length and the decay part shrinks it:
$\tfrac{d}{dt}\|\mathbf x\|^2=2\mathbf x^\top\mathbf A\mathbf x
=-\|\mathbf x\|^2$, so $\|\mathbf x(t)\|=e^{-t/2}\|\mathbf x_0\|$. A point
decays at rate $\tfrac12$ while it turns at unit speed: it spirals into the
origin (:numref:`fig_mdl-dyn-ode-field`). The eigenvalues of $\mathbf A$ are
$-0.5\pm\mathrm i$. Their real part gives the decay rate and their imaginary
part the angular speed. :numref:`sec_mdl-linear-odes-stability` shows that
this holds in general: a linear field is a sum of decays, growths and
rotations, one for each eigenvalue.

![An ODE as a velocity field. The faint grid arrows are the field $\mathbf{f}(\mathbf{x})=\mathbf{A}\mathbf{x}$ with $\mathbf{A}=\left(\begin{smallmatrix}-0.5&-1\\1&-0.5\end{smallmatrix}\right)$; the two coloured curves are trajectories that always follow the local arrow, spiralling into the stable fixed point at the origin (eigenvalues $-0.5\pm\mathrm{i}$).](../img/mdl-dyn-ode-field.svg)
:label:`fig_mdl-dyn-ode-field`

### The Initial-Value Problem

Most fields are not linear, so we need a form that covers every rule of
motion. The general problem gives a velocity rule and a starting point:

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), t),
\qquad \mathbf{x}(0) = \mathbf{x}_0 .
$$
:eqlabel:`eq_mdl-ode-ivp`

Here $\mathbf x(t)\in\mathbb R^d$ is the state. The function
$\mathbf f:\mathbb R^d\times\mathbb R\to\mathbb R^d$ is a **vector field**:
it attaches an arrow to every point at every time. A **trajectory** is a
differentiable path whose tangent at each instant is the arrow at its current
position.

The vector $\mathbf x_0$ is the **initial value**, and :eqref:`eq_mdl-ode-ivp`
is called an **initial-value problem**. When $\mathbf f$ does not depend on
$t$, the system is **autonomous**: the arrows stay fixed and only the state
moves. Decay, rotation and the spiral are the autonomous fields
$\mathbf f(\mathbf x)=-\mathbf x$, $\mathbf K\mathbf x$ and
$\mathbf A\mathbf x$.

Integrating both sides from $0$ to $t$ (:numref:`sec_mdl-integral_calculus`)
gives the equivalent **integral form**

$$
\mathbf{x}(t) = \mathbf{x}_0 + \int_0^t \mathbf{f}(\mathbf{x}(s), s)\,ds .
$$
:eqlabel:`eq_mdl-ode-integral-form`

The solution is the start plus the accumulated velocity. The existence proof
below works with the integral form, and Euler's method
(:numref:`sec_mdl-euler-runge-kutta`) is the left-endpoint Riemann sum of
this integral.

### Existence and Uniqueness
:label:`sec_mdl-ode-existence-uniqueness`

Does an initial-value problem have a solution, and only one? For a model
defined by a field, uniqueness is what makes the output a function of the
input, and it can fail.

Consider $\dot x=\sqrt{|x|}$ from $x_0=0$. The constant path $x\equiv0$
solves it. So does, for every waiting time $c\ge0$,

$$
x_c(t) =
\begin{cases}
0, & t \le c, \\
(t - c)^2/4, & t > c,
\end{cases}
$$

since $\dot x_c=(t-c)/2=\sqrt{x_c}$ for $t>c$. A particle at rest may leave
at any time (:numref:`fig_mdl-dyn-uniqueness-fan`), so a model defined by
this field does not assign one output to its input. The cause is the slope of
$\sqrt{|x|}$, which is unbounded at $0$: a small displacement $x$ produces the
much larger velocity $\sqrt{|x|}$.

![Solutions of $\dot{x} = \sqrt{|x|}$ through $x(0) = 0$: the constant solution $x \equiv 0$ and a parabolic solution $x_c(t) = (t-c)^2/4$ for every departure time $c$. Because the slope of $\sqrt{|x|}$ is unbounded at the origin, the Lipschitz condition fails and the solution is not unique.](../img/mdl-dyn-uniqueness-fan.svg)
:label:`fig_mdl-dyn-uniqueness-fan`

For a well-behaved field a solution can be built by iterating the integral
form :eqref:`eq_mdl-ode-integral-form`: guess a path, compute the path the
field would produce from it, and repeat. This is **Picard iteration**. For
$\dot x=x$ with $x_0=1$, start from the constant path $\varphi_0\equiv1$ and
set $\varphi_{k+1}(t)=1+\int_0^t\varphi_k(s)\,ds$:

$$
\varphi_1(t) = 1 + t,
\qquad
\varphi_2(t) = 1 + t + \tfrac{t^2}{2},
\qquad
\varphi_3(t) = 1 + t + \tfrac{t^2}{2} + \tfrac{t^3}{6} .
$$

The iterates converge to the solution $e^t$. Here each iteration adds one
Taylor term of $e^t$; for other fields the iterates look different, but they
still converge, provided the field does not change too abruptly from point to
point. The next proposition makes "too abruptly" precise.

**Proposition (Picard--Lindelöf).** *Let $\mathbf{f}(\mathbf{x}, t)$ be
continuous in $t$ and globally Lipschitz in $\mathbf{x}$: there is an $L$
with*

$$
\|\mathbf{f}(\mathbf{x}, t) - \mathbf{f}(\mathbf{y}, t)\| \le L\,\|\mathbf{x} - \mathbf{y}\|
\qquad \textrm{for all } \mathbf{x}, \mathbf{y}, t.
$$

*Then the initial-value problem :eqref:`eq_mdl-ode-ivp` has exactly one
solution on $[0, T]$, for every $\mathbf{x}_0$ and every $T>0$*
:cite:`Coddington.Levinson.1955`.

This is the result of the first part that everything later leans on. The fan
is what happens when no such $L$ exists near $0$. The proof shows that the
Picard iterates converge, and that they converge to the only solution.

**Proof.** A continuous path $\boldsymbol\varphi$ on $[0,\delta]$ solves the
problem if and only if it is a fixed point of the **Picard operator**

$$
(P\boldsymbol{\varphi})(t) = \mathbf{x}_0 + \int_0^t \mathbf{f}(\boldsymbol{\varphi}(s), s)\,ds .
$$

In words, $P$ replaces a guessed path by the path the field would produce from
it. (A solution satisfies the integral form; conversely, a continuous fixed
point is differentiable by the fundamental theorem of calculus, with the right
derivative and the right start.)

The step that is not routine pulls the Lipschitz bound through the
integral. For two paths on $[0,\delta]$, write
$d(\boldsymbol\varphi,\boldsymbol\psi)=\max_{t\le\delta}\|\boldsymbol\varphi(t)-\boldsymbol\psi(t)\|$.
Then for $t\le\delta$,

$$
\|(P\boldsymbol{\varphi})(t) - (P\boldsymbol{\psi})(t)\|
\le \int_0^t L\,\|\boldsymbol{\varphi}(s) - \boldsymbol{\psi}(s)\|\,ds
\le L\,\delta\; d(\boldsymbol\varphi,\boldsymbol\psi) .
$$

With $\delta=1/(2L)$, every application of $P$ halves the distance between
two candidate paths.

Start from the constant path $\boldsymbol\varphi_0\equiv\mathbf x_0$ and set
$\boldsymbol\varphi_{k+1}=P\boldsymbol\varphi_k$. Consecutive iterates then
differ by at most $2^{-k}d(\boldsymbol\varphi_1,\boldsymbol\varphi_0)$, and
these differences add up like a geometric series. So the iterates converge
uniformly on $[0,\delta]$, to a path $\boldsymbol\varphi$. We take one fact
from analysis on trust here: a uniform limit of continuous paths is
continuous. Since
$d(P\boldsymbol\varphi_k,P\boldsymbol\varphi)\le\tfrac12d(\boldsymbol\varphi_k,\boldsymbol\varphi)\to0$,
the limit satisfies $\boldsymbol\varphi=P\boldsymbol\varphi$: it is a
solution. Two solutions $\boldsymbol\varphi,\boldsymbol\psi$ would satisfy
$d(\boldsymbol\varphi,\boldsymbol\psi)=d(P\boldsymbol\varphi,P\boldsymbol\psi)\le\tfrac12d(\boldsymbol\varphi,\boldsymbol\psi)$,
so they coincide.

Finally restart at $t=\delta$ from $\boldsymbol\varphi(\delta)$. The step
length $1/(2L)$ does not depend on where the solution is, because $L$ is the
same everywhere, so $\lceil 2LT\rceil$ restarts cover $[0,T]$, with one
solution on each piece. $\blacksquare$

The theorem asks for a Lipschitz bound and for that bound to hold globally,
and each requirement rules out a different failure. The fan shows what happens
without a Lipschitz bound. Without the global bound, the solution can blow up.
The field $\dot x=x^2$ is smooth, and from $x_0>0$ separation of variables
gives

$$
x(t) = \frac{x_0}{1 - x_0 t},
$$

which reaches infinity at the finite time $t=1/x_0$. The proof shows why the
theorem does not cover it. On the interval $|x|\le\rho$ the field is
Lipschitz with $L=2\rho$, since $|x^2-y^2|=|x+y|\,|x-y|$. The restart length
$1/(2L)=1/(4\rho)$ therefore shrinks as the solution grows, and the restarts
can add up to a finite time. A field that is Lipschitz only on bounded sets
guarantees a solution only for a while.

### The Flow Map

A generative model moves a whole cloud of starting points at once, so we now
fix the field and vary the start. For an autonomous, globally Lipschitz
field, write

$$
\Phi_t(\mathbf x_0)=\mathbf x(t)
$$

for the state at time $t$ of the one trajectory started at $\mathbf x_0$.
This **flow map** $\Phi_t:\mathbb R^d\to\mathbb R^d$ moves all of space
forward by $t$ along the arrows.

For decay the flow map is $\Phi_t(x)=e^{-t}x$. Flowing for $s$ and then for
$t$ gives $\Phi_t(\Phi_s(x))=e^{-t}e^{-s}x=e^{-(t+s)}x=\Phi_{t+s}(x)$, and
$\Phi_{-t}(x)=e^{t}x$ undoes $\Phi_t$. The same law holds for every
autonomous field:

$$
\Phi_0 = \mathrm{id},
\qquad
\Phi_{t+s} = \Phi_t \circ \Phi_s .
$$
:eqlabel:`eq_mdl-ode-flow-group`

The reason is that the arrows do not change with time: the path
$\tau\mapsto\mathbf x(s+\tau)$ starts at $\Phi_s(\mathbf x_0)$ and follows
the same arrows, so by uniqueness it is the trajectory from that point. The
decay example suggests that the law also holds for negative times. It does,
because running a field backward is the same theorem applied to a second
field.

**Corollary (the flow map is a bijection).** *Let $\mathbf f(\mathbf x)$ be
autonomous and globally Lipschitz. Then $\Phi_t$ exists for every
$t\in\mathbb R$, and each $\Phi_t$ is a bijection of $\mathbb R^d$ with
inverse $\Phi_{-t}$.*

**Proof.** The field $\mathbf g(\mathbf y)=-\mathbf f(\mathbf y)$ is Lipschitz
with the same $L$. A path $\mathbf x(t)$ solves $\dot{\mathbf x}=\mathbf f$ on
$[-T,0]$ if and only if $\mathbf y(s)=\mathbf x(-s)$ solves
$\dot{\mathbf y}=\mathbf g$ on $[0,T]$, since
$\dot{\mathbf y}(s)=-\dot{\mathbf x}(-s)=-\mathbf f(\mathbf y(s))$. The
theorem applied to $\mathbf g$ gives a unique solution backward in time for
every start, so $\Phi_t$ is defined for all real $t$. The composition law
:eqref:`eq_mdl-ode-flow-group`, whose argument did not use the sign of $s$ or
$t$, then gives
$\Phi_{-t}\circ\Phi_t=\Phi_t\circ\Phi_{-t}=\Phi_0=\mathrm{id}$.
$\blacksquare$

One consequence is that two different trajectories never meet: a meeting point
would have two different backward solutions. (For a time-dependent field the
same argument applies to $\mathbf g(\mathbf y,s)=-\mathbf f(\mathbf y,-s)$.)

An invertible map from noise to data whose inverse can be computed is what a
generative flow needs, and the fields of many neural networks qualify. A
field built from linear maps and Lipschitz activations, such as
$\mathbf f(\mathbf x)=\mathbf W_2\tanh(\mathbf W_1\mathbf x+\mathbf b_1)+\mathbf b_2$,
is globally Lipschitz, with $L$ at most the product of the weight norms and
the activations' constants ($\|\mathbf W_2\|\,\|\mathbf W_1\|$ here). The
model of :numref:`sec_mdl-neural-odes-cnf` is therefore well-posed, and by
the corollary its time-$T$ map is a bijection.

## Linear ODEs and Stability
:label:`sec_mdl-linear-odes-stability`

Linear fields are the one family we can solve completely, and they describe
most other fields near a point where the field vanishes.

### The Matrix Exponential

The scalar equation $\dot x=ax$ is solved by $x(t)=e^{at}x_0$. For a linear
system

$$
\dot{\mathbf{x}} = \mathbf A\mathbf{x},
\qquad
\mathbf{x}(0) = \mathbf{x}_0,
\qquad \mathbf A \in \mathbb{R}^{d \times d},
$$

we look for the same form with a matrix in the exponent. For any square
matrix $\mathbf C$, define the **matrix exponential** by the power series that
defines $e^z$ (stated without proof in :numref:`sec_mdl-eigendecompositions`):

$$
e^{\mathbf C} = \sum_{k=0}^{\infty} \frac{\mathbf C^k}{k!}
= \mathbf I + \mathbf C + \frac{\mathbf C^2}{2} + \frac{\mathbf C^3}{6} + \cdots .
$$
:eqlabel:`eq_mdl-ode-matrix-exp-series`

The series converges for every matrix: term by term,
$\|\mathbf C^k\| \le \|\mathbf C\|^k$ (the operator norm is
submultiplicative), so it is dominated by the scalar series for
$e^{\|\mathbf C\|}$, which converges.

**Proposition (the matrix exponential solves linear ODEs).**
*$\mathbf{x}(t) = e^{\mathbf At}\mathbf{x}_0$ is the unique solution of
$\dot{\mathbf{x}} = \mathbf A\mathbf{x}$, $\mathbf{x}(0) = \mathbf{x}_0$.*

**Proof.** Differentiate the series :eqref:`eq_mdl-ode-matrix-exp-series` with
$\mathbf C = \mathbf At$ term by term (legitimate: on any bounded $t$-interval
the differentiated series still converges uniformly, by the same domination):

$$
\frac{d}{dt} e^{\mathbf At}
= \sum_{k=1}^{\infty} \frac{\mathbf A^k\, t^{k-1}}{(k-1)!}
= \mathbf A \sum_{j=0}^{\infty} \frac{(\mathbf At)^j}{j!}
= \mathbf A\, e^{\mathbf At}.
$$

So $\mathbf{x}(t) = e^{\mathbf At}\mathbf{x}_0$ satisfies the ODE, and at
$t = 0$ the series collapses to $\mathbf I$, so $\mathbf{x}(0) = \mathbf{x}_0$.
Uniqueness is Picard--Lindelöf: the field $\mathbf{x} \mapsto \mathbf A\mathbf{x}$
is Lipschitz with $L = \|\mathbf A\|$. $\blacksquare$

An infinite series of matrix powers is not yet a formula we can read. The
eigendecomposition turns it into $d$ scalar exponentials. Recall that if
$\mathbf A$ has $d$ independent eigenvectors $\mathbf v_j$,
$\mathbf A\mathbf v_j=\lambda_j\mathbf v_j$, then with
$\mathbf V=(\mathbf v_1,\ldots,\mathbf v_d)$ and
$\boldsymbol\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_d)$ we have
$\mathbf A=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$
(:numref:`sec_mdl-eigendecompositions`).

Powers then telescope,
$(\mathbf At)^k = \mathbf V (\boldsymbol\Lambda t)^k \mathbf V^{-1}$, because
each adjacent $\mathbf V^{-1} \mathbf V$ cancels, and summing the series
:eqref:`eq_mdl-ode-matrix-exp-series` gives

$$
e^{\mathbf At} = \mathbf V e^{\boldsymbol\Lambda t} \mathbf V^{-1},
\qquad
e^{\boldsymbol\Lambda t} = \mathrm{diag}\!\left(e^{\lambda_1 t}, \ldots, e^{\lambda_d t}\right).
$$
:eqlabel:`eq_mdl-ode-matrix-exp-eig`

The formula is a decoupling statement. Expand the start in the eigenbasis,
$\mathbf{x}_0 = \sum_j c_j \mathbf{v}_j$; then

$$
\mathbf{x}(t) = \sum_{j=1}^{d} c_j\, e^{\lambda_j t}\, \mathbf{v}_j .
$$

Along each eigendirection the system is the scalar equation
$\dot{x} = \lambda_j x$, evolving independently of the others.
Each term $c_je^{\lambda_jt}\mathbf v_j$ is called a **mode** of the system.

For the spiral we can do better than the general formula. Because
$\mathbf A=-\tfrac12\mathbf I+\mathbf K$ and $\mathbf I$ commutes with
$\mathbf K$, the exponential splits, $e^{\mathbf At}=e^{-t/2}e^{\mathbf Kt}$.
Because $\mathbf K^2=-\mathbf I$, the powers of $\mathbf K$ cycle through
$\mathbf I,\mathbf K,-\mathbf I,-\mathbf K$, and the series for
$e^{\mathbf Kt}$ splits into even and odd terms, the series of $\cos t$ and
$\sin t$:

$$
e^{\mathbf At}
=e^{-t/2}\left(\cos t\,\mathbf I+\sin t\,\mathbf K\right)
=e^{-t/2}\begin{pmatrix}\cos t&-\sin t\\\sin t&\cos t\end{pmatrix}.
$$

The flow shrinks by $e^{-t/2}$ and turns by $t$ radians, as we found from the
field itself. From $\mathbf x_0=(2,0)$ at $t=2$
the state is $e^{-1}(2\cos2,\,2\sin2)=(-0.306,\,0.669)$, with length
$2e^{-1}=0.736$. For this $\mathbf A$ the eigenvectors and eigenvalues
$-0.5\pm\mathrm i$ are complex, but the product
$\mathbf Ve^{\boldsymbol\Lambda t}\mathbf V^{-1}$ is real.

A word of caution about the split of the exponential. The rule
$e^{\mathbf P+\mathbf Q}=e^{\mathbf P}e^{\mathbf Q}$ holds when
$\mathbf P\mathbf Q=\mathbf Q\mathbf P$, since then $(\mathbf P+\mathbf Q)^k$
expands by the binomial theorem. For matrices that do not commute it fails
(:exref:`ex_mdl-odes-solvers-symplectic-euler`).

There is a third route to $e^{\mathbf At}$: the compound-interest limit
$(\mathbf I+t\mathbf A/n)^n$, which needs no series and no eigenvectors. The
cell computes all three for the spiral at $t=2$. Before running it, guess how
fast the compounding converges in $n$.
<!-- WRITER: the cell no longer sums the series. It computes e^{At} by the eigendecomposition and the
compounding (I + tA/n)^n, not "all three". -->

```{.python .input #odes-solvers-matrix-exponential}
A = np.array([[-0.5, -1.0], [1.0, -0.5]])      # spiral: eigenvalues -0.5 +- i
t, x0 = 2.0, np.array([2.0, 0.0])
lam, V = np.linalg.eig(A)                      # A = V diag(lam) V^{-1}
expAt = (V @ np.diag(np.exp(lam * t)) @ np.linalg.inv(V)).real
x = expAt @ x0
print('eigenvalues of A:', lam.round(4))
print('x(2) =', x.round(4), f' length {np.linalg.norm(x):.6f}')
for n in [2 ** 10, 2 ** 16, 2 ** 24]:          # compounding (I + tA/n)^n
    err = np.abs(np.linalg.matrix_power(np.eye(2) + (t / n) * A, n) - expAt).max()
    print(f'n = 2^{int(np.log2(n))}: error {err:.1e}, n * error = {n * err:.2f}')
```

<!-- WRITER: the cell prints no series-against-eigenformula line any more; it prints the eigenvalues, the state
x(2) = (-0.3062, 0.669) with length 0.735759 (the hand values), and n * error = 0.90 at n = 2^10, 2^16, 2^24
(errors 8.8e-04, 1.4e-05, 5.4e-08). The first clause below is false now. -->
The series and the eigendecomposition agree to rounding, and the state
$(-0.3062,\,0.6690)$ and its length $0.735759$ are the values computed by
hand. The compounding is far slower: its error times $n$ stays at $0.90$ as
$n$ grows from $2^{10}$ to $2^{24}$, so the error falls only like $1/n$. This
is compound interest with a matrix interest rate, and it is also $n$ steps of
Euler's method, whose error of order $1/n$ is proved in
:numref:`sec_mdl-euler-runge-kutta`.

### Stability from Eigenvalues

The decoupled form tells us at a glance whether a linear system settles down.
Write each eigenvalue as $\lambda_j = a_j + \mathrm i\, b_j$. Its mode has
magnitude $|e^{\lambda_j t}| = e^{a_j t}$, since
$e^{\mathrm i b_jt}=\cos b_jt+\mathrm i\sin b_jt$ has modulus one. The real
part sets growth or decay; the imaginary part sets rotation. For a
diagonalizable $\mathbf A$ this gives the following classification.

* If $\operatorname{Re}\lambda_j < 0$ for all $j$, every mode decays, and
  $\mathbf{x}(t) \to \mathbf{0}$ from every start: the origin is
  **asymptotically stable**. The slowest mode sets the decay rate
  $-\max_j \operatorname{Re}\lambda_j$.
* If $\operatorname{Re}\lambda_j > 0$ for some $j$, that mode grows
  exponentially, and every start with a component along $\mathbf v_j$
  escapes: the origin is **unstable**.

A complex pair $a\pm\mathrm ib$ is the spiral with general rates: a rotation
at angular speed $b$ that decays or grows with the sign of $a$. To see this,
write the eigenvector of $\lambda=a+\mathrm ib$ as
$\mathbf v=\mathbf p+\mathrm i\mathbf q$ with real $\mathbf p,\mathbf q$
(independent, since $b\ne0$). The real and imaginary parts of
$\mathbf A\mathbf v=(a+\mathrm ib)\mathbf v$ are

$$
\mathbf A\mathbf p=a\mathbf p-b\mathbf q,
\qquad
\mathbf A\mathbf q=b\mathbf p+a\mathbf q .
$$

So $\mathbf A$ maps the plane of $\mathbf p$ and $\mathbf q$ to itself, and
in the basis $(\mathbf q,\mathbf p)$ it is the matrix
$a\mathbf I+b\mathbf K$. The computation for the spiral then gives

$$
e^{(a\mathbf I+b\mathbf K)t}=e^{at}\left(\cos bt\,\mathbf I+\sin bt\,\mathbf K\right):
$$

scaling by $e^{at}$ composed with rotation by the angle $bt$. The same block,
iterated in discrete time instead of flowed, appears in
:numref:`subsec_mdl-complex-rotation`.

In two dimensions, and for $\det\mathbf A\ne0$, the signs of the eigenvalues
give six standard pictures.

| eigenvalues of $\mathbf A$ ($2\times 2$)      | portrait        | behavior                    |
|-----------------------------------------------|-----------------|-----------------------------|
| real, both negative                           | stable node     | monotone decay              |
| real, both positive                           | unstable node   | monotone growth             |
| real, opposite signs                          | saddle          | decay in, escape out        |
| complex pair, $\operatorname{Re}\lambda < 0$  | stable spiral   | decaying oscillation        |
| complex pair, $\operatorname{Re}\lambda > 0$  | unstable spiral | growing oscillation         |
| purely imaginary pair                         | center          | closed orbits               |

:numref:`fig_mdl-dyn-phase-portraits` draws one field of each kind; the
stable spiral is the field of :numref:`fig_mdl-dyn-ode-field`.

![Six planar linear systems, one per row of the table; dashed lines in the node and saddle panels are the eigendirections, along which trajectories enter or leave.](../img/mdl-dyn-phase-portraits.svg)
:label:`fig_mdl-dyn-phase-portraits`

When $\mathbf A$ is not diagonalizable, the modes acquire polynomial factors
$t^ke^{\lambda t}$, which change the verdict only for eigenvalues on the
imaginary axis.

### Linear Systems with an Input
:label:`subsec_mdl-ode-driven-linear`

So far nothing pushed the system from outside. The state of a state space
model that reads a sequence (:numref:`sec_ssm`) obeys a linear equation with
an input $\mathbf u(t)$,

$$
\dot{\mathbf x}=\mathbf A\mathbf x+\mathbf B\mathbf u(t),
\qquad \mathbf x(0)=\mathbf x_0 .
$$

The matrix exponential solves this equation as well. Since
$\tfrac{d}{dt}e^{-\mathbf At}=-e^{-\mathbf At}\mathbf A$, the product rule
gives
$\tfrac{d}{dt}\bigl(e^{-\mathbf At}\mathbf x\bigr)
=e^{-\mathbf At}(\dot{\mathbf x}-\mathbf A\mathbf x)
=e^{-\mathbf At}\mathbf B\mathbf u$,
and the right-hand side no longer contains $\mathbf x$. Integrating from $0$
to $t$ and multiplying by $e^{\mathbf At}$ yields the **variation of
constants** formula

$$
\mathbf x(t)=e^{\mathbf At}\mathbf x_0
+\int_0^t e^{\mathbf A(t-s)}\,\mathbf B\mathbf u(s)\,ds .
$$
:eqlabel:`eq_mdl-ode-variation-of-constants`

The state is the decayed start plus every past input, each decayed for the
time since it arrived. Now suppose that the input is constant during a step,
$\mathbf u(s)=\mathbf u_n$ for $s\in[t_n,t_n+h)$, which is called a
*zero-order hold*. The integral over one step can then be done once and for
all, and the states at the grid times obey a linear recurrence without any
approximation:

$$
\mathbf x_{n+1}=\bar{\mathbf A}\,\mathbf x_n+\bar{\mathbf B}\,\mathbf u_n,
\qquad
\bar{\mathbf A}=e^{\mathbf Ah},
\qquad
\bar{\mathbf B}=\int_0^h e^{\mathbf A\tau}\,d\tau\;\mathbf B
=\mathbf A^{-1}\bigl(e^{\mathbf Ah}-\mathbf I\bigr)\mathbf B ,
$$
:eqlabel:`eq_mdl-ode-zoh`

where the last form needs $\mathbf A$ to be invertible. In one dimension
these are $\bar a=e^{h\lambda}$ and $\bar b=(e^{h\lambda}-1)\,b/\lambda$, the
coefficients on which the state space models of :numref:`subsec_zoh` are
built (there the step is $\Delta$ and the state is $h$). The point to
remember is that $|e^{h\lambda}|=e^{h\operatorname{Re}\lambda}<1$ for every
step $h>0$ whenever $\operatorname{Re}\lambda<0$: the exact recurrence is
stable whenever the equation is. On $\dot x=-50x$ with $h=0.05$ the exact
factor is $e^{-2.5}=0.08$, where the opening's Euler step had $-1.5$.
<!-- WRITER: "the opening" (the bowl 25x^2, Euler's factor -1.5 at h = 0.05) is now in :numref:`sec_mdl-euler-runge-kutta`;
this becomes the section's one forward reference to it. -->

<!-- WRITER: lead-in for the new cell. It runs the scalar recurrence of :eqref:`eq_mdl-ode-zoh`,
x_{n+1} = e^{h lam} x_n + (e^{h lam} - 1)/lam * u_n, on dx/dt = lam x + sin t from x(0) = 1 to T = 2, with the
input sampled at the start of each step and held, against the closed form
x(T) = e^{lam T}(1 + 1/(1 + lam^2)) - (lam sin T + cos T)/(1 + lam^2), for lam in {-1, -50} and
h in {0.5, 0.1, 0.02}. Each row prints the exact factor e^{h lam}, Euler's factor 1 + h lam for comparison,
and the error at T. A last line runs a constant input u = 1, which the hold represents exactly. -->

```{.python .input #mdl-odes-zero-order-hold}
def zoh(lam, h, u, x0=1.0, T=2.0):
    """Exact steps of dx/dt = lam x + u(t), u held at u(t_n) over each step."""
    a, b = np.exp(h * lam), np.expm1(h * lam) / lam
    x = x0
    for n in range(round(T / h)):
        x = a * x + b * u(n * h)
    return x

def exact_sin(lam, T=2.0):
    """Solution of dx/dt = lam x + sin t with x(0) = 1, at time T."""
    c = 1 + lam ** 2
    return np.exp(lam * T) * (1 + 1 / c) - (lam * np.sin(T) + np.cos(T)) / c

print(f'{"lam":>5} {"h":>5} {"e^(h lam)":>10} {"1 + h lam":>10}'
      f' {"error at T":>11}')
for lam in [-1.0, -50.0]:
    for h in [0.5, 0.1, 0.02]:
        err = abs(zoh(lam, h, np.sin) - exact_sin(lam))
        print(f'{lam:5.0f} {h:5.2f} {np.exp(h * lam):10.3g}'
              f' {1 + h * lam:10.2f} {err:11.1e}')
```

<!-- WRITER: reading of the zero-order-hold cell (numbers: see the report code-mdl-odes-solvers.md). The recurrence
is exact for a held input (constant input: error at rounding level, about 1e-17 or below); with a moving
input the error comes only from holding it and falls about in proportion to h (lam = -1: 6.8e-02, 9.7e-03,
1.8e-03; lam = -50: 1.6e-03, 5.8e-04, 9.2e-05); and at h = 0.5, lam = -50 (h|lam| = 25) the exact factor is
1.4e-11 while Euler's would be -24, so the recurrence is stable at every step. This is the fact
:numref:`subsec_zoh` builds on. Do not compare errors with forward Euler: for lam = -1 Euler's error is
smaller at every h (both are first order). -->

### Transient Growth
:label:`subsec_mdl-ode-transient-growth`

When every mode decays, the state goes to zero, but its length
$\|\mathbf x(t)\|$ need not decrease all the way. A triangular matrix
(called `B` in the code below) shows it:
<!-- WRITER: the transient-growth cell is cut (its home is :numref:`subsec_mdl-transient-growth`); "(called `B`
in the code below)" is false now. -->

$$
\dot{\mathbf x}=\begin{pmatrix}-1&10\\0&-2\end{pmatrix}\mathbf x,
\qquad \mathbf x_0=(0,1),
$$

with eigenvalues $-1$ and $-2$, both negative. The second row involves only
$x_2$, so $x_2(t)=e^{-2t}$. The first row is then $\dot x_1=-x_1+10e^{-2t}$,
a decay equation with an input, and :eqref:`eq_mdl-ode-variation-of-constants`
gives

$$
x_1(t)=10\int_0^te^{-(t-s)}e^{-2s}\,ds=10\left(e^{-t}-e^{-2t}\right).
$$

The first coordinate starts at $0$, rises to $10(\tfrac12-\tfrac14)=2.5$ at
$t=\log2\approx0.69$, and only then decays. The reason is the one of
:numref:`subsec_mdl-transient-growth`, which treats the discrete-time case in
full: the eigenvectors $\mathbf v_1=(1,0)$ and
$\mathbf v_2=(0.995,\,-0.0995)$ span lines only $5.7^\circ$ apart, so the
short vector $\mathbf x_0=10.0\,\mathbf v_1-10.05\,\mathbf v_2$ is the
difference of two long ones. They decay at different rates, and the
cancellation stops before both have died away. The cell evaluates the closed
form and plots the length of the state against the two modes.
<!-- WRITER: no cell and no plot any more; the sentence above and "The length climbs from 1 to 2.51 at
t = 0.68" below are false (no cell prints them). By hand: x_1 peaks at 2.5 at t = log 2; the length
sqrt(x_1^2 + x_2^2) peaks near 2.51 at t = 0.68 (old store value 2.513), i.e. it follows x_1. -->


The length climbs from $1$ to $2.51$ at $t\approx0.68$ and afterwards decays
like $10e^{-t}$, at the rate of the slower mode. Eigenvalues give the
asymptotic rate of decay, not the size of the transient. This matters when
reading a training curve: a momentum run that converges can grow for its
first few steps (:exref:`ex_mdl-odes-solvers-transient-growth`).

### Linearization at Fixed Points
:label:`subsec_mdl-ode-linearization`

Nonlinear fields have no closed-form solution, but near a point where the
field vanishes they behave like a linear one. A **fixed point** of an
autonomous field is a state $\mathbf x^\star$ with
$\mathbf f(\mathbf x^\star)=\mathbf 0$: a particle placed there never moves.
For a continuously differentiable $\mathbf f$ and a small displacement
$\boldsymbol\delta(t)=\mathbf x(t)-\mathbf x^\star$, Taylor's theorem
(:numref:`sec_mdl-multivariable_calculus`) gives

$$
\dot{\boldsymbol{\delta}}
= \mathbf{f}(\mathbf{x}^\star + \boldsymbol{\delta})
= \mathbf J\,\boldsymbol{\delta} + o(\|\boldsymbol{\delta}\|),
\qquad
\mathbf J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}(\mathbf{x}^\star) .
$$

Near the fixed point the field is the linear field of its Jacobian
$\mathbf J$. A fixed point is **hyperbolic** if no eigenvalue of $\mathbf J$
has zero real part. For a hyperbolic fixed point, the Hartman--Grobman theorem
:cite:`Hartman.1960,Grobman.1959` says that near $\mathbf x^\star$ the
trajectories of the nonlinear field are those of
$\dot{\boldsymbol\delta}=\mathbf J\boldsymbol\delta$, up to a continuous change
of coordinates. The eigenvalues of the Jacobian then decide local stability by
the classification above.

The damped pendulum is the one nonlinear field we work out. Its angle
$\phi$ from the bottom obeys $\ddot\phi=-\sin\phi-c\dot\phi$, with damping
coefficient $c\ge0$. With the angular velocity $\omega=\dot\phi$ it becomes the
first-order system

$$
\dot\phi=\omega,\qquad \dot\omega=-\sin\phi-c\,\omega,
\qquad
\mathbf J(\phi,\omega)=\begin{pmatrix}0&1\\-\cos\phi&-c\end{pmatrix}.
$$

The fixed points have $\omega=0$ and $\sin\phi=0$: the pendulum hanging at
$\phi=0$ and balanced upright at $\phi=\pi$.

* At the bottom, $\mathbf J$ has characteristic polynomial
  $\lambda^2+c\lambda+1$, with roots
  $\lambda=-c/2\pm\mathrm i\sqrt{1-c^2/4}$ for $c<2$. At $c=0.5$ they are
  $-0.25\pm0.97\,\mathrm i$: a stable spiral, the ring-down of a released
  pendulum, which swings with period about $2\pi/0.97$ while its amplitude
  decays like $e^{-t/4}$.
* At the top, the polynomial is $\lambda^2+c\lambda-1$, whose roots
  $(-c\pm\sqrt{c^2+4})/2$ have product $-1$, so one is positive and one
  negative for every $c$. At $c=0.5$ they are $0.78$ and $-1.28$: a saddle.
  Only the starts on one curve approach the top. They are the pendulum swung
  up with just the right speed, which comes to rest balanced like a pencil on
  its tip.

The same computation, a Jacobian and its eigenvalues, classifies the fixed
points of a trained field, with one exception. When an eigenvalue lies on the
imaginary axis, the linearization decides nothing. The fields $\dot x=-x^3$
and $\dot x=x^3$ both have Jacobian $0$ at the origin. The first is stable,
with $x(t)=x_0/\sqrt{1+2x_0^2t}\to0$; the second escapes to infinity at
$t=1/(2x_0^2)$.

<!-- WRITER: ending needed here (structure.md: every result in this section was exact; a computer has only steps;
hand-off to :numref:`sec_mdl-euler-runge-kutta`). -->

## Summary

*Ordinary differential equations.* An ordinary differential equation
prescribes a velocity at every point. When the field is globally Lipschitz,
exactly one trajectory passes through each point, and the flow map is a
bijection. Linear systems are solved by
$e^{\mathbf At}=\mathbf Ve^{\boldsymbol\Lambda t}\mathbf V^{-1}$, so the real
parts of the eigenvalues decide decay and the imaginary parts rotation, and
with an input held constant over a step the exact solution is a linear
recurrence that is stable at every step size. Nearly parallel eigenvectors
allow transient growth, and at a hyperbolic fixed point the Jacobian
classifies a nonlinear field as its eigenvalues would a linear one.

## Exercises

1. **Separation of trajectories.** Consider a field of the form
   $\mathbf{f}(\mathbf{x}) = \mathbf W_2 \tanh(\mathbf W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$,
   the Neural ODE field of the next section.
   <!-- WRITER: two sections ahead now; cite :numref:`sec_mdl-neural-odes-cnf`. -->
    1. Show that $\mathbf{f}$ is Lipschitz with constant
       $L = \|\mathbf W_2\|\,\|\mathbf W_1\|$ in the operator norm. Which
       property of $\tanh$ does the proof use? Does the bound change when
       $\tanh$ is replaced by ReLU?
    1. Let $\mathbf{x}(t)$ and $\mathbf{y}(t)$ be two trajectories of an
       autonomous field with Lipschitz constant $L$. Differentiate
       $\|\mathbf{x}(t) - \mathbf{y}(t)\|^2$ and show that

       $$
       e^{-Lt}\|\mathbf{x}(0) - \mathbf{y}(0)\|
       \le \|\mathbf{x}(t) - \mathbf{y}(t)\|
       \le e^{Lt}\|\mathbf{x}(0) - \mathbf{y}(0)\| .
       $$

       What does the lower bound imply for two trajectories that start at
       distinct points? What do the two bounds imply for the Lipschitz
       constants of $\Phi_T$ and of its inverse?
   :exlabel:`ex_mdl-odes-solvers-trajectory-separation`

1. [code] **Transient growth and nonnormality.** The text shows one matrix
   with transient growth. This exercise finds how the growth depends on the
   angle between the eigenvectors, and meets it again in momentum.
    1. For $\mathbf B_\nu=\left(\begin{smallmatrix}-1&\nu\\0&-2\end{smallmatrix}\right)$
       and $\mathbf x_0=(0,1)$, show that $x_1(t)=\nu(e^{-t}-e^{-2t})$ and find
       the peak of $x_1$ and the time at which it occurs. Find numerically the
       smallest $\nu$ for which $\|\mathbf x(t)\|$ ever exceeds
       $\|\mathbf x_0\|$.
    1. Compute the angle between the two eigenvectors of $\mathbf B_\nu$ as a
       function of $\nu$, express $\mathbf x_0$ in the eigenbasis, and explain
       the growth as the cancellation of two large components.
    1. Compute $\max_t\|e^{\mathbf B_\nu t}\|_2$ for $\nu=1,5,10,20$ numerically
       and compare it with the peak of the trajectory from $(0,1)$. Why can
       the first never be smaller than the second?
    1. The heavy-ball recurrence of
       :exref:`ex_mdl-odes-solvers-heavy-ball-ode` on
       $\mathcal L(x)=\tfrac\lambda2x^2$ is the linear map
       $(x_{k+1},x_k)=\mathbf M(x_k,x_{k-1})$ with
       $\mathbf M=\left(\begin{smallmatrix}1+\beta-\eta\lambda&-\beta\\1&0\end{smallmatrix}\right)$.
       Show that $\mathbf M$ is not normal, and compute its eigenvalues for
       $\beta=0.9$, $\eta\lambda=0.1$. Run it from $(x_0,x_{-1})=(1,0)$: for
       how many steps does $\|(x_{k+1},x_k)\|$ grow before it decays, and to
       what peak? What does this say about reading the first iterations of a
       momentum run as divergence?
   :exlabel:`ex_mdl-odes-solvers-transient-growth`

1. **Limits of linearization.**
    1. Classify the origin of $\dot x=-x^3$, $\dot x=x^3$ and $\dot x=x^2$
       (stable, unstable, or stable from one side only) by solving each
       equation. All three have Jacobian $0$ there. What does
       Hartman--Grobman say about them?
    1. For the damped pendulum, find the damping $c$ at which the bottom rest
       point changes from a spiral to a node, and show that the top is a
       saddle for every $c\ge0$.
    1. Sketch the portrait near the bottom for $c=0.5$, $c=2$ and $c=4$, and
       describe how a released pendulum moves in each case.
   :exlabel:`ex_mdl-odes-solvers-linearization-limits`

[Discussions](https://d2l.discourse.group/t/36837)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §38.1]{.kicker}

A velocity field and the motion it determines<br>**Ordinary differential equations**
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- read decay, rotation and escape from the eigenvalues of a field;
- say when a field has exactly one trajectory through each point;
- solve a linear system with an input held over each step.

**Running example:** the spiral $\mathbf A=\left(\begin{smallmatrix}-0.5&-1\\1&-0.5\end{smallmatrix}\right)$, eigenvalues $-0.5\pm\mathrm i$.

**The idea:** eigenvalues decide the motion.
:::

<!-- WRITER: the "Where this fits" table still names one combined section "ODEs and Solvers"; split its row in two. -->

::: {.slide title="Given Fields, No Noise: The First Step of the Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **ODEs and Solvers** | **exact motion and its numerical steps** |
| Neural ODEs | a learned field, its gradient, its density |
| Stochastic Differential Equations | noise, Itô calculus, the OU process |
| Fokker–Planck and Probability Flow | how the density moves; the score |
| Score Matching and Diffusion | learning the score, sampling with it |
| Flow Matching and Transport | learning a velocity; straight paths |
:::

::: {.col .narrow}
- **Uses:** eigenvalues, Taylor expansion.
- **Skips:** symplectic and exponential integrators, error control.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Vector fields and flows]{.dtitle}

[one trajectory through each point]{.dsub}
:::
:::

::: {.slide title="Decay Plus Rotation Is a Spiral"}
[The running example]{.kicker}

::: {.cols .vc}
::: {.col}
- $\dot x=-x$: $x(t)=e^{-t}x_0$, decay.
- $\dot{\mathbf x}=\mathbf K\mathbf x$, $\mathbf K=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$: $\mathbf x^\top\mathbf K\mathbf x=0$, circles.
- $\mathbf A=-\tfrac12\mathbf I+\mathbf K$: $\|\mathbf x(t)\|=e^{-t/2}\|\mathbf x_0\|$ while turning at unit speed.
:::

::: {.col .fig}
![](../img/mdl-dyn-ode-field.svg)
:::
:::
:::

::: {.slide title="Without a Lipschitz Bound a Particle at Rest May Leave Whenever It Likes"}
[Counterexamples]{.kicker}

::: {.cols .vc}
::: {.col .fig}
![](../img/mdl-dyn-uniqueness-fan.svg)
:::

::: {.col}
- $\dot x=\sqrt{|x|}$, $x_0=0$: $x\equiv0$, and $x_c=(t-c)^2/4$ after any $c$.
- Slope of $\sqrt{|x|}$ is unbounded at $0$.
- $\dot x=x^2$: $x_0/(1-x_0t)$ blows up at $t=1/x_0$.
:::
:::
:::

::: {.slide title="A Lipschitz Field Has Exactly One Trajectory Through Each Point, Forward and Backward"}
[Picard--Lindelöf]{.kicker}

- Picard iteration $\varphi_{k+1}=x_0+\int_0^t f(\varphi_k)\,ds$ on $\dot x=x$: $1,\ 1+t,\ 1+t+\tfrac{t^2}2,\ldots$
- $\|f(x)-f(y)\|\le L\|x-y\|$ gives, on $[0,\delta]$, $\max\|P\varphi-P\psi\|\le L\delta\max\|\varphi-\psi\|$.
- $\delta=1/(2L)$ halves distances, the same everywhere, so restarts cover $[0,T]$.
- Flow map $\Phi_t(\mathbf x_0)=\mathbf x(t)$: $\Phi_{t+s}=\Phi_t\circ\Phi_s$.

::: {.d2l-note .rule}
$-\mathbf f$ is Lipschitz too, so the field runs backward: $\Phi_t^{-1}=\Phi_{-t}$.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Linear systems and stability]{.dtitle}

[one exponential per eigenvalue]{.dsub}
:::
:::

::: {.slide title="e^(At) Splits into One Exponential per Eigenvalue"}
[The matrix exponential]{.kicker}

$e^{\mathbf At}=\sum_k(\mathbf At)^k/k!=\mathbf Ve^{\boldsymbol\Lambda t}\mathbf V^{-1}$; spiral:
$e^{\mathbf At}=e^{-t/2}\left(\begin{smallmatrix}\cos t&-\sin t\\\sin t&\cos t\end{smallmatrix}\right)$,
$\|\mathbf x(2)\|=2e^{-1}=0.736$ from $(2,0)$.

@!odes-solvers-matrix-exponential
:::

::: {.slide title="The Real Part of λ Decides Growth, the Imaginary Part Rotation"}
[Stability]{.kicker}

::: {.cols .vc}
::: {.col .fig .big}
![](../img/mdl-dyn-phase-portraits.svg)
:::

::: {.col}
- $\lambda=a+\mathrm ib$: $|e^{\lambda t}|=e^{at}$.
- All $a<0$: stable; some $a>0$: unstable.
- At a hyperbolic fixed point the Jacobian classifies a nonlinear field the same way.
:::
:::
:::

::: {.slide title="Eigenvalues Give the Asymptotic Rate, Not the Size of the Transient"}
[Transient growth]{.kicker}

$\mathbf B=\left(\begin{smallmatrix}-1&10\\0&-2\end{smallmatrix}\right)$, eigenvalues $-1,-2$, from $(0,1)$:
$x_1=10(e^{-t}-e^{-2t})$ peaks at $2.5$.

::: {.d2l-note .rule}
Eigenvectors $5.7^\circ$ apart: $\mathbf x_0=10\,\mathbf v_1+10.05\,\mathbf v_2$, two large components that stop cancelling.
:::
:::

<!-- WRITER: the transient-growth cell and its plot are cut, so the slide above lost its figure column. A slide
for the driven equation and the zero-order hold (cell #mdl-odes-zero-order-hold) is still to be written. -->

::: {.slide title="Check: The Damped Pendulum's Two Rest Points"}
[Check]{.kicker}

$\dot\phi=\omega$, $\dot\omega=-\sin\phi-0.5\,\omega$, Jacobian $\left(\begin{smallmatrix}0&1\\-\cos\phi&-0.5\end{smallmatrix}\right)$.
What are the eigenvalues at the bottom $(0,0)$ and at the top $(\pi,0)$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Bottom: $\lambda^2+0.5\lambda+1=0$, $\lambda=-0.25\pm0.97\,\mathrm i$, a stable spiral.
Top: $\lambda^2+0.5\lambda-1=0$, $\lambda=0.78$ and $-1.28$, a saddle.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- Lipschitz field: one trajectory per point; $\Phi_t^{-1}=\Phi_{-t}$.
- Spiral: $e^{\mathbf At}=e^{-t/2}\times$rotation, $\|\mathbf x(2)\|=2e^{-1}=0.736$.
- Nonnormal $\mathbf B$: both modes decay, length peaks at $2.5$.

**Next:** following a field step by step, and when the steps are stable.
:::
