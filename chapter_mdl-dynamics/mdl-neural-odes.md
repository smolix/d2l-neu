# Neural ODEs, Adjoints, and Continuous Flows
:label:`sec_mdl-neural-odes-cnf`

Placeholder opening (the section writer replaces this paragraph): this section
trains models defined by an ordinary differential equation, computes their
gradients with the adjoint method, and tracks how such a flow changes a
probability density.

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-2}
#@tab mxnet
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
import numpy as np
npx.set_np()
```

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-2}
#@tab pytorch
import numpy as np
import torch
```

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-2}
#@tab tensorflow
import numpy as np
import tensorflow as tf
```

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-2}
#@tab jax
import jax
from jax import numpy as jnp
import numpy as np
```

The adjoint and density checks below reuse the spiral field $A$ and the
starting point $\mathbf{x}_0$ of :numref:`sec_mdl-odes-solvers`.

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-1}
A = np.array([[-0.5, -1.0], [1.0, -0.5]])      # spiral sink: eigenvalues -0.5 +- i
x0 = np.array([2.0, 0.0])
```

## Neural ODEs and the Adjoint Method
:label:`sec_mdl-neural-odes`

### Residual Blocks as Euler Steps

A residual update has the same algebraic form as a forward-Euler step.

**Proposition (a residual block has the form of an Euler step).** *The residual update
:cite:`He.Zhang.Ren.ea.2016`*

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathbf{f}_\theta(\mathbf{x}_l)
$$

*is exactly one forward-Euler step :eqref:`eq_mdl-ode-euler-update` of the
ODE $\dot{\mathbf{x}} = \mathbf{f}_\theta(\mathbf{x})$ with step size
$h = 1$.*

**Proof.** Set $h = 1$ and $\mathbf{f} = \mathbf{f}_\theta$ in
:eqref:`eq_mdl-ode-euler-update`; the two updates are the same formula.
$\blacksquare$

![Left: a stack of residual blocks, each adding a learned branch to an identity skip. Right: the updates interpreted as forward-Euler steps. This interpretation converges to a smooth flow only for a consistent time-indexed field and shrinking step size; a finite residual stack is not itself the exact ODE flow.](../img/mdl-dyn-resnet-as-euler.svg)
:label:`fig_mdl-dyn-resnet-as-euler`

A residual update can therefore be read as one numerical step. A general
ResNet may use a different branch $\mathbf{f}_{\theta_l}$ at every layer; its
numerical interpretation is a time-dependent field sampled at the layer
times. The interpretation alone gives no invertibility guarantee. For
example, $\dot{x}=-x$ has the bijective exact flow $e^{-t}x$, whereas one Euler
step with $h=1$ maps every $x$ to zero.

A continuous-depth limit requires a consistent family of fields and shrinking
steps. For the shared field
$\mathbf{x}_{l+1}=\mathbf{x}_l+\tfrac{1}{N}\mathbf{f}_\theta(\mathbf{x}_l)$,
Euler convergence implies that the output approaches the exact time-one flow
$\Phi_1(\mathbf{x}_0)$ as $N\to\infty$, under the regularity assumptions of
:numref:`sec_mdl-euler-runge-kutta`. The limiting model is the **Neural ODE**
:cite:`Chen.Rubanova.Bettencourt.ea.2018`:

$$
\dot{\mathbf{x}} = \mathbf{f}_\theta(\mathbf{x}, t),
\qquad
\textrm{output} = \mathbf{x}(T) = \mathbf{x}_0 + \int_0^T \mathbf{f}_\theta(\mathbf{x}(t), t)\,dt .
$$

The exact Neural ODE map is well-posed and invertible when its field satisfies
the existence assumptions in both time directions. A numerical solver
approximates that map. Changing Euler to RK4 or an adaptive method leaves the
field parameters fixed but changes the computed map, its cost, and its error.

### Training Through the Solver

The direct training route is **discretize, then
differentiate**: unroll a fixed solver, say $N$ Euler steps, into the
computation graph, compute the loss on $\mathbf{x}_N$, and apply ordinary
backpropagation flow through the unrolled steps. The "network" this builds is
equivalent to a ResNet with $N$ blocks that share the same weights $\theta$, so
ordinary automatic differentiation already trains it.

Consider a task small enough to inspect directly: learn a planar field
$\mathbf{f}_\theta$ (one hidden layer, $32$ tanh
units) whose time-$1$ flow carries the unit circle onto a shifted, squashed
ellipse. We use $64$ paired points, mean-squared loss on the endpoints, $10$
unrolled Euler steps of size $h = 0.1$, and Adam
(:numref:`sec_mdl-adaptive-stochastic-methods`) as the optimizer. The
underlying ODE describes a smooth invertible deformation of
$\mathbb{R}^2$; the ten-step Euler program is its numerical approximation. We
supervise only the $64$ listed points.

```{.python .input #odes-solvers-neural-ode-train}
#@tab pytorch
theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
X0_ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
X1_ring = np.stack([1.5 + 1.2 * np.cos(theta), 0.4 * np.sin(theta)], axis=1)
torch.manual_seed(0)
X, Y = torch.tensor(X0_ring).float(), torch.tensor(X1_ring).float()
net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(),
                          torch.nn.Linear(32, 2))
N, h = 10, 0.1                                 # unrolled Euler: 10 steps to T = 1

def flow(X):
    for _ in range(N):                         # one Euler step = one residual block
        X = X + h * net(X)
    return X

opt = torch.optim.Adam(net.parameters(), lr=0.05)
for it in range(401):
    loss = ((flow(X) - Y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 100 == 0:
        print(f'iter {it:3d}  loss {loss.item():.5f}')
print(f'mean endpoint error: {(flow(X) - Y).norm(dim=1).mean().item():.4f}')
```

```{.python .input #odes-solvers-neural-ode-train}
#@tab mxnet
theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
X0_ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
X1_ring = np.stack([1.5 + 1.2 * np.cos(theta), 0.4 * np.sin(theta)], axis=1)
npx.random.seed(0)
X, Y = mnp.array(X0_ring), mnp.array(X1_ring)
net = nn.Sequential()
net.add(nn.Dense(32, activation='tanh'), nn.Dense(2))
net.initialize(init.Xavier())
N, h = 10, 0.1                                 # unrolled Euler: 10 steps to T = 1

def flow(X):
    for _ in range(N):                         # one Euler step = one residual block
        X = X + h * net(X)
    return X

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.05})
for it in range(401):
    with autograd.record():
        loss = ((flow(X) - Y) ** 2).mean()
    loss.backward()
    trainer.step(1)
    if it % 100 == 0:
        print(f'iter {it:3d}  loss {float(loss):.5f}')
err = mnp.sqrt(((flow(X) - Y) ** 2).sum(axis=1)).mean()
print(f'mean endpoint error: {float(err):.4f}')
```

```{.python .input #odes-solvers-neural-ode-train}
#@tab tensorflow
theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
X0_ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
X1_ring = np.stack([1.5 + 1.2 * np.cos(theta), 0.4 * np.sin(theta)], axis=1)
tf.random.set_seed(0)
X, Y = tf.constant(X0_ring, tf.float32), tf.constant(X1_ring, tf.float32)
net = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='tanh'),
                           tf.keras.layers.Dense(2)])
N, h = 10, 0.1                                 # unrolled Euler: 10 steps to T = 1

def flow(X):
    for _ in range(N):                         # one Euler step = one residual block
        X = X + h * net(X)
    return X

opt = tf.keras.optimizers.Adam(learning_rate=0.05)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((flow(X) - Y) ** 2)
    opt.apply_gradients(zip(tape.gradient(loss, net.trainable_variables),
                            net.trainable_variables))
    return loss

for it in range(401):
    loss = train_step()
    if it % 100 == 0:
        print(f'iter {it:3d}  loss {float(loss):.5f}')
err = tf.reduce_mean(tf.norm(flow(X) - Y, axis=1))
print(f'mean endpoint error: {float(err):.4f}')
```

```{.python .input #odes-solvers-neural-ode-train}
#@tab jax
theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
X0_ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
X1_ring = np.stack([1.5 + 1.2 * np.cos(theta), 0.4 * np.sin(theta)], axis=1)
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
params = {'W1': 0.25 * jax.random.normal(k1, (2, 32)), 'b1': jnp.zeros(32),
          'W2': 0.25 * jax.random.normal(k2, (32, 2)), 'b2': jnp.zeros(2)}
X, Y = jnp.array(X0_ring), jnp.array(X1_ring)
N, h = 10, 0.1                                 # unrolled Euler: 10 steps to T = 1

def flow(params, X):
    field = lambda X: (jnp.tanh(X @ params['W1'] + params['b1'])
                       @ params['W2'] + params['b2'])
    return jax.lax.fori_loop(0, N, lambda i, X: X + h * field(X), X)

loss_fn = lambda params: ((flow(params, X) - Y) ** 2).mean()

@jax.jit
def adam_step(params, m, v, it):               # hand-rolled Adam, lr = 0.05
    loss, g = jax.value_and_grad(loss_fn)(params)
    m = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, m, g)
    v = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, v, g)
    upd = lambda p, m, v: p - 0.05 * (m / (1 - 0.9 ** it)) / (
        jnp.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return jax.tree.map(upd, params, m, v), m, v, loss

m = jax.tree.map(jnp.zeros_like, params)
v = jax.tree.map(jnp.zeros_like, params)
for it in range(1, 402):
    params, m, v, loss = adam_step(params, m, v, jnp.float32(it))
    if it % 100 == 1:
        print(f'iter {it - 1:3d}  loss {float(loss):.5f}')
err = jnp.linalg.norm(flow(params, X) - Y, axis=1).mean()
print(f'mean endpoint error: {float(err):.4f}')
```

A few hundred full-batch Adam iterations drive the loss to $\sim 10^{-5}$ and
the mean endpoint error to $\sim 10^{-3}$: one tiny vector
field, integrated by ten shared-weight residual updates, learns the prescribed
endpoints. Integrating the learned continuous field backward approximates its
inverse flow. The forward and backward numerical maps agree only up to solver
error; the ten-step Euler map itself is not guaranteed to be invertible.

### The Adjoint Method: Backpropagation in Continuous Time
:label:`sec_mdl-adjoint-method`

Differentiating through the unrolled solver computes the exact gradient of the
discretized program, but ordinarily stores its intermediate states. The
**continuous adjoint method** instead differentiates the exact ODE problem and
integrates a second ODE backward in time. With recomputation it can reduce
stored solver states, at the cost of additional evaluations and possible
numerical mismatch. Its central
object is the **adjoint state**
$\mathbf{a}(t) = \partial L / \partial \mathbf{x}(t)$ ("how would the loss
change if the trajectory were nudged at time $t$?"), the continuous
analogue of the backprop delta of
:numref:`sec_mdl-matrix-calculus-autodiff`.

**Proposition (the adjoint equations).** *Let $\mathbf{f}$ be continuously
differentiable in $\mathbf{x}$ and $\theta$, let $\mathbf{x}(t)$ solve
$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \theta, t)$ on $[0, T]$ from fixed
$\mathbf{x}(0)$, and let $L = \ell(\mathbf{x}(T))$ be a differentiable loss
on the endpoint.
Then the adjoint state satisfies the linear ODE*

$$
\dot{\mathbf{a}}(t) = -\left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right)^{\!\top}\! \mathbf{a}(t),
\qquad
\mathbf{a}(T) = \nabla \ell(\mathbf{x}(T)),
$$
:eqlabel:`eq_mdl-ode-adjoint`

*integrated backward from $T$, and the parameter gradient is the integral*

$$
\frac{\partial L}{\partial \theta}
= \int_0^T \left(\frac{\partial \mathbf{f}}{\partial \theta}\right)^{\!\top}\! \mathbf{a}(t)\, dt,
$$
:eqlabel:`eq_mdl-ode-adjoint-grad`

*with both partial derivatives evaluated along the trajectory
$(\mathbf{x}(t), \theta, t)$* :cite:`Chen.Rubanova.Bettencourt.ea.2018`.

**Proof (variational).** Perturb the parameters by an infinitesimal
$\delta\theta$ and analyze the resulting trajectory perturbation. Because $\mathbf{f}$ is
$C^1$, the flow is differentiable in $\theta$ and the first-order expansion
below is rigorous: the perturbation
$\delta\mathbf{x}(t)$ obeys the *linearization* of the dynamics (differentiate
the ODE in $\theta$ and use the chain rule):

$$
\delta\dot{\mathbf{x}}
= \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\,\delta\mathbf{x}
+ \frac{\partial \mathbf{f}}{\partial \theta}\,\delta\theta,
\qquad
\delta\mathbf{x}(0) = \mathbf{0}.
$$

Define $\mathbf{a}(t)$ by the backward ODE
:eqref:`eq_mdl-ode-adjoint` and examine the pairing
$\mathbf{a}^\top \delta\mathbf{x}$ evolve; the derivative telescopes by
design:

$$
\frac{d}{dt}\left(\mathbf{a}^\top \delta\mathbf{x}\right)
= \dot{\mathbf{a}}^\top \delta\mathbf{x} + \mathbf{a}^\top \delta\dot{\mathbf{x}}
= -\mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\,\delta\mathbf{x}
+ \mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\,\delta\mathbf{x}
+ \mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \theta}\,\delta\theta
= \mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \theta}\,\delta\theta .
$$

The state-coupling terms cancel exactly; that cancellation is *why* the
adjoint ODE has the form it has. Integrate from $0$ to $T$, using
$\delta\mathbf{x}(0) = \mathbf{0}$ on the left endpoint:

$$
\mathbf{a}(T)^\top \delta\mathbf{x}(T)
= \int_0^T \mathbf{a}(t)^\top \frac{\partial \mathbf{f}}{\partial \theta}\, dt \;\delta\theta .
$$

The left side is $\nabla\ell(\mathbf{x}(T))^\top \delta\mathbf{x}(T) = \delta L$,
the first-order change of the loss; the right side is linear in
$\delta\theta$ with coefficient :eqref:`eq_mdl-ode-adjoint-grad`. (The same
telescoping, stopped at an intermediate time $s$, shows
$\delta L = \mathbf{a}(s)^\top \delta\mathbf{x}(s)$ for a nudge injected at
time $s$, so the solution of :eqref:`eq_mdl-ode-adjoint` really is
$\partial L/\partial \mathbf{x}(s)$, earning its name.) $\blacksquare$

One caveat connects back to :numref:`sec_mdl-ode-existence-uniqueness`: the
existence theorem tolerates ReLU fields, but the adjoint equations as stated
need a $C^1$ field. A ReLU field is differentiable only piecewise; in
practice autograd differentiates the unrolled solver piece by piece, and
smooth activations (tanh, GELU) remove the issue entirely.

The product
$-\mathbf{a}^\top \partial\mathbf{f}/\partial\mathbf{x}$ that each
backward-ODE step computes is a **vector--Jacobian product**, the very
primitive that reverse-mode AD is made of
(:numref:`sec_mdl-matrix-calculus-autodiff`). Discretize
:eqref:`eq_mdl-ode-adjoint` with the same Euler scheme as the forward pass
and the result is the backpropagation recursion through the unrolled
network, except that each Jacobian is evaluated at the other end of its
step: backprop **is** the discrete adjoint method, a lineage that runs
from optimal control :cite:`Pontryagin.Boltyanskii.Gamkrelidze.ea.1962`
straight to the backward pass of every deep-learning framework. The continuous formulation offers another implementation:
instead of storing the forward states for the VJPs, one may
*re-integrate* $\mathbf{x}(t)$ backward alongside $\mathbf{a}(t)$, so that
in an idealized implementation the memory does not grow with the number of
solver steps, at the cost of extra compute
and of numerical drift when the reversed dynamics are unstable (a strongly
contracting forward flow is, run backward, strongly expanding; in that regime
checkpointing or plain unrolling is the sturdier choice). The literature
names the two routes **discretize-then-optimize** (differentiate the
unrolled solver, exact for the executed program) versus
**optimize-then-discretize** (discretize the continuous adjoint, using less
trajectory storage but generally producing only an approximation to the
gradient of the discrete forward program)
:cite:`Kidger.2022`; Exercise :exref:`ex_mdl-odes-solvers-re-derive-adjoint-equations` probes exactly this fault line.

The preceding identities can be checked on the linear ODE
$\dot{\mathbf{x}} = A\mathbf{x}$, where every ingredient has a closed form:
the trajectory is $e^{At}\mathbf{x}_0$, the adjoint of
$L = \tfrac12\|\mathbf{x}(T)\|^2$ is
$\mathbf{a}(t) = e^{A^\top (T-t)}\,\mathbf{x}(T)$ (the adjoint ODE
$\dot{\mathbf{a}} = -A^\top\mathbf{a}$ run backward), and with $\theta$ the
entries of $A$ itself, $\partial\mathbf{f}/\partial A$ turns
:eqref:`eq_mdl-ode-adjoint-grad` into
$\partial L/\partial A = \int_0^T \mathbf{a}(t)\,\mathbf{x}(t)^\top dt$. The
cell computes the gradient three ways: hand-written reverse mode through the
unrolled Euler solver (the discrete adjoint: exactly what reverse-mode
autograd builds, written out in six lines), central finite differences on the
same unrolled program, and the continuous adjoint integral by quadrature.

```{.python .input #odes-solvers-adjoint-check}
def expm_eig(B, s):
    """e^{Bs} via the eigendecomposition formula."""
    mu, W = np.linalg.eig(B)
    return (W @ np.diag(np.exp(mu * s)) @ np.linalg.inv(W)).real

def loss_and_grad_unrolled(A, x0, T, n):
    """L = ||x_n||^2 / 2 through n unrolled Euler steps; dL/dA by hand-written
    reverse mode -- the discrete adjoint."""
    h, xs = T / n, [x0]
    for k in range(n):
        xs.append(xs[-1] + h * (A @ xs[-1]))   # forward sweep, tape the states
    a, gA = xs[-1].copy(), np.zeros_like(A)    # a_n = dL/dx_n = x_n
    for k in range(n - 1, -1, -1):             # backward sweep: one VJP per step
        gA += h * np.outer(a, xs[k])           # dL/dA  +=  h a_{k+1} x_k^T
        a = a + h * (A.T @ a)                  # a_k = (I + hA)^T a_{k+1}
    return 0.5 * xs[-1] @ xs[-1], gA

T, n = 2.0, 1000
L, g_disc = loss_and_grad_unrolled(A, x0, T, n)
g_fd = np.zeros_like(A)                        # central finite differences
for i in range(2):
    for j in range(2):
        E = np.zeros_like(A); E[i, j] = 1e-5
        g_fd[i, j] = (loss_and_grad_unrolled(A + E, x0, T, n)[0]
                      - loss_and_grad_unrolled(A - E, x0, T, n)[0]) / 2e-5
print('discrete adjoint vs finite differences:', f'{np.abs(g_disc - g_fd).max():.1e}')

m = 2000                                       # continuous adjoint, trapezoid rule
ts = np.linspace(0.0, T, m + 1)
xT = expm_eig(A, T) @ x0
F = np.stack([np.outer(expm_eig(A.T, T - s) @ xT, expm_eig(A, s) @ x0)
              for s in ts])
g_cont = (T / m) * (F[1:-1].sum(axis=0) + 0.5 * (F[0] + F[-1]))
for n_steps in [100, 1000, 10000]:
    gap = np.abs(loss_and_grad_unrolled(A, x0, T, n_steps)[1] - g_cont).max()
    print(f'n = {n_steps:6d} Euler steps: |discrete - continuous adjoint| = {gap:.1e}')
```

The discrete adjoint matches finite differences to $10^{-10}$ (it *is* the
exact gradient of the unrolled program), and as the solver is refined the
discrete gradient converges to the continuous adjoint integral at exactly the
solver's order, $O(h)$: ten times the steps, one-tenth the gap. Backprop
through a solver and the adjoint method are one algorithm, seen
at two resolutions.

## Continuous Normalizing Flows
:label:`sec_mdl-continuous-normalizing-flows`

### The Instantaneous Change of Variables

Pushing *samples* through a Neural ODE defines a generative model. Maximum-
likelihood training additionally requires the change in *density* along the
flow. The discrete answer we know from :numref:`sec_mdl-random_variables`:
for an invertible map $\mathbf{g}$ with Jacobian $J_{\mathbf{g}}$,

$$
\log p_{\textrm{out}}(\mathbf{g}(\mathbf{x}))
= \log p_{\textrm{in}}(\mathbf{x}) - \log \left|\det J_{\mathbf{g}}(\mathbf{x})\right| ,
$$

and a normalizing flow built from $K$ layers evaluates one log-determinant per
layer. The $O(d^3)$ determinant is the reason discrete flows constrain
their layers to triangular or low-rank Jacobians. In continuous time the
formula *simplifies*: over one infinitesimal step the flow map is
near-identity, and the determinant of a near-identity matrix is governed by
the trace.

**Lemma.** *For any square matrix $J$,
$\det(I + hJ) = 1 + h \operatorname{tr} J + O(h^2)$.*

**Proof.** In the Leibniz expansion of the determinant, any permutation other
than the identity uses at least two off-diagonal entries, each of size
$O(h)$, so contributes $O(h^2)$. The identity permutation contributes
$\prod_i (1 + h J_{ii}) = 1 + h \sum_i J_{ii} + O(h^2)$. $\blacksquare$

**Proposition (instantaneous change of variables).** *Let
$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, t)$ with $\mathbf{f}$ continuously
differentiable and $\mathbf{x}(0) \sim p_0$ for a differentiable density
$p_0$, and let $p_t$ denote the density of $\mathbf{x}(t)$, which then exists
and is differentiable along trajectories. Then along each
trajectory* :cite:`Chen.Rubanova.Bettencourt.ea.2018`

$$
\frac{d}{dt} \log p_t(\mathbf{x}(t))
= -\operatorname{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right)\!(\mathbf{x}(t), t).
$$
:eqlabel:`eq_mdl-ode-instant-cov`

**Proof.** Over a short interval the flow advances by the near-identity map
$\Phi_h(\mathbf{x}) = \mathbf{x} + h\,\mathbf{f}(\mathbf{x}, t) + O(h^2)$,
whose Jacobian is $I + h J + O(h^2)$ with
$J = \partial\mathbf{f}/\partial\mathbf{x}$. Apply the discrete
change-of-variables formula to this one map and expand with the Lemma, which
tolerates the perturbed argument: replacing $J$ by $J + O(h)$ changes
$\det(I + hJ)$ only at order $h^2$. Thus:

$$
\log p_{t+h}(\mathbf{x}(t+h))
= \log p_t(\mathbf{x}(t)) - \log\det\!\left(I + hJ + O(h^2)\right)
= \log p_t(\mathbf{x}(t)) - h \operatorname{tr} J + O(h^2),
$$

using $\log(1 + u) = u + O(u^2)$ (for small $h$ the determinant is positive,
so the absolute value is moot). Subtract, divide by $h$, and let
$h \to 0$. $\blacksquare$

The trace of the Jacobian is the **divergence** of the field: the local
expansion rate of volume. Where the field spreads ($\operatorname{tr} J > 0$),
the cloud of samples dilutes and the log-density along each trajectory falls;
where it compresses, probability concentrates. Integrating
:eqref:`eq_mdl-ode-instant-cov` along the trajectory turns the per-layer
log-det *sum* of a discrete flow into a *trace integral*:

$$
\log p_T(\mathbf{x}(T))
= \log p_0(\mathbf{x}(0)) - \int_0^T \operatorname{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right) dt .
$$
:eqlabel:`eq_mdl-ode-cnf-likelihood`

This yields the **continuous normalizing flow** (CNF). Augment the state with
its log-density and integrate $(\mathbf{x}, \log p)$ together. Under exact
integration, this gives the likelihood for an unconstrained architecture
without requiring triangular Jacobians.

### The Hutchinson Trace Estimator

Directly computing $\operatorname{tr}(\partial\mathbf{f}/\partial\mathbf{x})$
requires all $d$ diagonal entries of the Jacobian, or $d$ derivative passes.
Hutchinson's identity provides an unbiased stochastic estimator
:cite:`Hutchinson.1989`.

**Proposition (Hutchinson estimator).** *For any matrix
$M \in \mathbb{R}^{d \times d}$ and any random
$\boldsymbol{\epsilon} \in \mathbb{R}^d$ with
$\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$ and
$\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top] = I$ (Gaussian
or Rademacher, i.e. uniform $\pm 1$ entries),*

$$
\operatorname{tr}(M) = \mathbb{E}\!\left[\boldsymbol{\epsilon}^\top M \boldsymbol{\epsilon}\right].
$$
:eqlabel:`eq_mdl-ode-hutchinson`

**Proof.** $\mathbb{E}[\boldsymbol{\epsilon}^\top M \boldsymbol{\epsilon}]
= \mathbb{E}[\operatorname{tr}(M \boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top)]
= \operatorname{tr}(M\, \mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top])
= \operatorname{tr}(M)$, using the cyclic property of the trace and its
linearity (which lets it commute with the expectation). $\blacksquare$

The point is *what the estimator touches*: $\boldsymbol{\epsilon}^\top J$ is
a single vector--Jacobian product followed by a dot product: one reverse-mode
pass through $\mathbf{f}$ with cost comparable to one evaluation/backward
pass, and no full Jacobian is ever materialized
(:numref:`sec_mdl-matrix-calculus-autodiff`).
The resulting stochastic likelihood estimate requires one additional backward
pass, as in FFJORD
:cite:`Grathwohl.Chen.Bettencourt.ea.2018`.

For a linear field, the result can be checked directly:
$\dot{\mathbf{x}} = A\mathbf{x}$ has constant Jacobian $A$, so
:eqref:`eq_mdl-ode-cnf-likelihood` says the log-density along every
trajectory falls at the constant rate $\operatorname{tr}(A)$:
$\log p_t(\mathbf{x}(t)) = \log p_0(\mathbf{x}_0) - t \operatorname{tr}(A)$.
The cell flows standard-normal samples through the spiral field and compares
the two answers, since the exact time-$t$ density is the Gaussian
$\mathcal{N}(\mathbf{0},\, e^{At} e^{A^\top t})$, computable in closed form.
Then it verifies the Hutchinson estimator on a random $6 \times 6$ matrix.

```{.python .input #odes-solvers-cnf-trace}
rng = np.random.default_rng(0)
t = 1.5
X0s = rng.standard_normal((4, 2))              # four samples from p_0 = N(0, I)
Et = expm_eig(A, t)
Xts = X0s @ Et.T                               # flow each sample to time t
Sig = Et @ Et.T                                # x(t) ~ N(0, E_t E_t^T)
logp0 = -0.5 * (X0s ** 2).sum(axis=1) - np.log(2 * np.pi)
logpt = (-0.5 * np.einsum('ki,ij,kj->k', Xts, np.linalg.inv(Sig), Xts)
         - 0.5 * np.log(np.linalg.det(Sig)) - np.log(2 * np.pi))
print('log p_t(x(t)), Gaussian formula  :', logpt.round(6))
print('log p_0(x_0) - t tr(A), our rule :', (logp0 - t * np.trace(A)).round(6))

J = rng.standard_normal((6, 6))                # Hutchinson on a bigger matrix
eps = rng.choice([-1.0, 1.0], size=(100000, 6))
est = np.einsum('ki,ij,kj->k', eps, J, eps)
print(f'Hutchinson: tr(J) = {np.trace(J):.4f},  estimate = {est.mean():.4f}'
      f' +- {est.std() / np.sqrt(len(est)):.4f}')
```

The two log-density computations agree digit for digit (the Gaussian formula
subtracts $\log\det e^{At}$ from $\log p_0(\mathbf{x}_0)$, and
$\log\det e^{At} = t\operatorname{tr}A$), and the Hutchinson
estimate brackets the true trace within its standard error. Finally,
:eqref:`eq_mdl-ode-instant-cov` is the
trajectory-wise form of the *continuity equation* governing how a whole
density field is transported;
:numref:`sec_mdl-fokker-planck-probability-flow` starts there and turns
a diffusion's noisy paths into a deterministic probability-flow ODE
integrated by exactly the solvers of this section.

## Summary

* A residual block has the algebraic form of an Euler step. A consistent
  family with shrinking step size can converge to a **Neural ODE**; a finite
  residual stack need not inherit the exact flow's invertibility. Backpropagation
  through the solver is the discrete adjoint and gives the discretized
  program's exact gradient. The continuous adjoint differentiates the ODE
  problem and trades saved states for recomputation, tolerance error, and
  possible reverse-time instability.
* Densities transported by a flow obey the **instantaneous change of
  variables** $\tfrac{d}{dt}\log p_t = -\operatorname{tr}(\partial\mathbf{f}/\partial\mathbf{x})$:
  the log-det-Jacobian becomes a trace integral, estimated unbiasedly by
  **Hutchinson's** estimator $\boldsymbol{\epsilon}^\top J \boldsymbol{\epsilon}$
  at the cost of one VJP per stochastic trace sample: the mathematics of
  continuous normalizing flows.

## Exercises

1. [code] **The learned field at other step counts.** The
   `#odes-solvers-neural-ode-train` cell trains the field with $N = 10$ Euler
   steps of size $h = 0.1$. Integrating the trained field differently tests
   whether it defines a continuous flow or only a ten-step program.
    1. After training, set `N` and `h` with $Nh = 1$ for
       $N = 1, 2, 5, 10, 20, 100, 1000$, and compute the mean endpoint error
       of `flow` on `X` for each. Which quantity does the error at
       $N = 1000$ approximate?
    1. Integrate the learned field backward from `Y` with $N$ steps
       $\mathbf{x} \leftarrow \mathbf{x} - h\,\mathbf{f}_\theta(\mathbf{x})$
       for the same values of $N$, and compare the result with `X`. Why does
       the round trip not return `X` at $N = 10$?
    1. Retrain with $N = 1$ and with $N = 2$, and repeat (a). Which of the
       three trained models comes close to the time-one flow of its own
       field?
   :exlabel:`ex_mdl-odes-solvers-step-count-transfer`

1. [code] **What a flow cannot represent.** ● Trajectories of a Lipschitz
   field never meet (:numref:`sec_mdl-ode-existence-uniqueness`), and
   :citet:`Dupont.Doucet.Teh.2019` draw a consequence for Neural ODEs.
    1. Let $d = 1$. Use exercise
       :exref:`ex_mdl-odes-solvers-trajectory-separation`b to show that the
       time-$T$ map $\Phi_T$ of a Lipschitz field is strictly increasing.
       Conclude that no Neural ODE on $\mathbb{R}$ maps $-1$ to $1$ and $1$
       to $-1$.
    1. Show that the residual block $x + f(x)$ with $f(x) = -2x$ maps every
       $x$ to $-x$. Show that one Euler step $x + hf(x)$ of a field on
       $\mathbb{R}$ with Lipschitz constant $L$ is strictly increasing
       whenever $hL < 1$. Which condition does the block violate?
    1. Embed $\mathbb{R}$ in $\mathbb{R}^2$ as $x \mapsto (x, 0)$. Find a
       linear field on $\mathbb{R}^2$ whose time-one flow maps $(x, 0)$ to
       $(-x, 0)$ for every $x$.
    1. Change the `#odes-solvers-neural-ode-train` cell to fit the map
       $x \mapsto -x$ on $64$ evenly spaced points of $[-1, 1]$, with a field
       of input and output dimension $1$ and $N = 10$ steps. What does the
       trained model output? Repeat with input and output dimension $2$,
       inputs $(x, 0)$, and the loss on the first output coordinate.
    1. Train the one-dimensional model once more with $N = 1$ and $h = 1$,
       and evaluate it with $N = 1$ and with $N = 100$ steps of size
       $h = 0.01$. Relate the two results to (a) and (b).
   :exlabel:`ex_mdl-odes-solvers-flow-expressivity`

1. **Adjoint with a running loss.** ● The adjoint equations
   :eqref:`eq_mdl-ode-adjoint` and :eqref:`eq_mdl-ode-adjoint-grad` treat a
   loss on the endpoint. Regularizers such as the kinetic energy
   $\int_0^T \|\mathbf{f}(\mathbf{x}(t), \theta, t)\|^2\,dt$ of
   :citet:`Finlay.Jacobsen.Nurbekyan.ea.2020` accumulate a cost along the
   trajectory. Let

   $$
   L = \ell_T(\mathbf{x}(T)) + \int_0^T \ell(\mathbf{x}(t), \theta, t)\,dt .
   $$

    1. Extend the variational proof of :eqref:`eq_mdl-ode-adjoint` to show
       that the adjoint acquires a source term,
       $\dot{\mathbf{a}} = -(\partial\mathbf{f}/\partial\mathbf{x})^\top\mathbf{a} - \nabla_{\mathbf{x}}\ell$
       with $\mathbf{a}(T) = \nabla\ell_T(\mathbf{x}(T))$, and that
       $\partial L/\partial\theta = \int_0^T \big((\partial\mathbf{f}/\partial\theta)^\top\mathbf{a} + \nabla_\theta\ell\big)\,dt$.
    1. :citet:`Finlay.Jacobsen.Nurbekyan.ea.2020` instead append a
       coordinate $E$ with $\dot{E} = \ell$ and $E(0) = 0$ to the state, and
       add $E(T)$ to the endpoint loss. Apply :eqref:`eq_mdl-ode-adjoint` to
       the augmented state and show that it reproduces (a).
    1. For the kinetic energy, $\ell = \|\mathbf{f}\|^2$. Write
       $\nabla_{\mathbf{x}}\ell$ as a vector--Jacobian product. How many
       reverse-mode passes through $\mathbf{f}$ does one step of the adjoint
       ODE then require?
   :exlabel:`ex_mdl-odes-solvers-running-loss-adjoint`

1. [code] **Reconstructing the trajectory backward.** ● The continuous
   adjoint of :numref:`sec_mdl-adjoint-method` can avoid storing the forward
   states by integrating $\mathbf{x}(t)$ backward from $\mathbf{x}(T)$
   alongside $\mathbf{a}(t)$.
    1. For $\dot{x} = -\lambda x$ with $\lambda > 0$, an error $\delta$ in
       the stored $x(T)$ causes an error in the reconstructed $x(0)$. Find
       it.
    1. Let $R$ be the rotation by $0.6$ radians and
       $A = R\,\mathrm{diag}(-50, -1)\,R^\top$. Integrate
       $\dot{\mathbf{x}} = A\mathbf{x}$ from $(1, 1)$ to time $T$ with `rk4`
       and $2000$ steps, then integrate the field $-A\mathbf{x}$ from the
       result for the same time. Report the reconstruction error for
       $T = 0.25, 0.5, 1, 2$, and compare it with (a), taking $\delta$ to be
       the double-precision rounding error $\approx 10^{-16}$. Repeat with
       `A_stiff`. Why does the diagonal matrix reconstruct far more
       accurately?
    1. Activation checkpointing (:numref:`subsec_activation_checkpointing`)
       stores the state at the start of each segment of the trajectory. A
       hybrid scheme reconstructs the states within a segment by integrating
       backward from the checkpoint at its end. For the matrix $A$ of (b),
       what is the longest segment duration for which the error of (a)
       stays below $10^{-8}$?
   :exlabel:`ex_mdl-odes-solvers-backward-reconstruction`

1. **Log-determinant of a residual step.** ● The instantaneous change of
   variables :eqref:`eq_mdl-ode-instant-cov` is a limit of infinitesimal
   steps. A finite residual step $F(\mathbf{x}) = \mathbf{x} + \mathbf{g}(\mathbf{x})$
   needs its exact log-determinant, as in the invertible residual networks
   of :citet:`Behrmann.Grathwohl.Chen.ea.2019`.
    1. Let $\mathbf{g}$ be Lipschitz with constant $c < 1$. For $d = 1$ and
       $g = hf$, exercise :exref:`ex_mdl-odes-solvers-flow-expressivity`b
       shows that $F$ is strictly increasing, hence injective. For general
       $d$, write $F(\mathbf{x}) = \mathbf{y}$ as a fixed-point equation for
       $\mathbf{x}$ and use the Banach fixed-point theorem, as in the proof of
       Picard--Lindelöf, to show that $F$ is a bijection of $\mathbb{R}^d$.
       How fast does the fixed-point iteration converge?
    1. Show that $\|J_{\mathbf{g}}\| \le c$, where $J_{\mathbf{g}}$ is the
       Jacobian of $\mathbf{g}$, and use the eigenvalues of $J_{\mathbf{g}}$
       to show that

       $$
       \log\det(I + J_{\mathbf{g}})
       = \sum_{k=1}^\infty \frac{(-1)^{k+1}}{k}\operatorname{tr}\big(J_{\mathbf{g}}^k\big).
       $$

    1. An Euler step has $\mathbf{g} = h\mathbf{f}$ and
       $J_{\mathbf{g}} = hJ$. Compare $\log\det(I + hJ)$ with the term
       $h\operatorname{tr}J$ that :eqref:`eq_mdl-ode-cnf-likelihood`
       accumulates per step. How large is the difference per step, and how
       large is it in total over $T/h$ steps?
    1. The Hutchinson estimator :eqref:`eq_mdl-ode-hutchinson` estimates
       $\operatorname{tr}(J_{\mathbf{g}}^k)$ by
       $\boldsymbol{\epsilon}^\top J_{\mathbf{g}}^k \boldsymbol{\epsilon}$.
       How many vector--Jacobian products per probe does the series
       truncated after $n$ terms cost?
   :exlabel:`ex_mdl-odes-solvers-residual-logdet`

1. [code] **Variance of the Hutchinson estimator.** ● The estimator
   :eqref:`eq_mdl-ode-hutchinson` estimates $\operatorname{tr} M$ by
   $\boldsymbol{\epsilon}^\top M \boldsymbol{\epsilon}$ for a single probe
   $\boldsymbol{\epsilon}$. Let $S = (M + M^\top)/2$, so that
   $\boldsymbol{\epsilon}^\top M \boldsymbol{\epsilon} = \boldsymbol{\epsilon}^\top S \boldsymbol{\epsilon}$.
    1. Show that the variance of
       $\boldsymbol{\epsilon}^\top M \boldsymbol{\epsilon}$ is
       $2\|S\|_F^2$ for Gaussian probes and $2\sum_{i \neq j} S_{ij}^2$ for
       Rademacher probes. For which matrices is a single Rademacher probe
       exact?
    1. Evaluate both variances for the matrix `J` of the
       `#odes-solvers-cnf-trace` cell, and compare the Rademacher value with
       the standard error that the cell prints. How many probes does each
       distribution need for a standard error of $0.01$?
    1. In a CNF, $M$ is the Jacobian $\partial\mathbf{f}/\partial\mathbf{x}$
       in dimension $d$. Suppose that all entries have comparable magnitude.
       How does the relative standard error
       $\sqrt{\operatorname{Var}}/|\operatorname{tr} M|$ of one probe scale
       with $d$ when the diagonal entries share one sign, and when their
       signs are random?
   :exlabel:`ex_mdl-odes-solvers-hutchinson-variance`

1. [code] **Discrete and continuous adjoints.** ●● Backpropagation through
   a solver differentiates the executed program, whereas a solver applied to
   the adjoint ODE :eqref:`eq_mdl-ode-adjoint` approximates the gradient of
   the exact flow :cite:`Kidger.2022`. Compare the two recursions step by
   step for an autonomous field with Jacobian
   $J(\mathbf{x}) = \partial\mathbf{f}/\partial\mathbf{x}$ and forward
   states $\mathbf{x}_k$.
    1. For forward Euler, the backward sweep of `loss_and_grad_unrolled`
       computes $\mathbf{a}_k = (I + hJ(\mathbf{x}_k))^\top\mathbf{a}_{k+1}$.
       An explicit Euler step of the adjoint ODE from $t_{k+1}$ back to
       $t_k$ computes $\mathbf{a}_k = \mathbf{a}_{k+1} + hJ(\mathbf{x}_{k+1})^\top\mathbf{a}_{k+1}$.
       Bound the difference of the two updates for a smooth $\mathbf{f}$. For
       a linear field the two coincide. What then produces the gap that the
       `#odes-solvers-adjoint-check` cell prints?
    1. Differentiate one Heun step
       $\mathbf{x}_{k+1} = \mathbf{x}_k + \tfrac{h}{2}\big[\mathbf{f}(\mathbf{x}_k) + \mathbf{f}(\mathbf{u}_k)\big]$
       with $\mathbf{u}_k = \mathbf{x}_k + h\mathbf{f}(\mathbf{x}_k)$ to
       obtain its discrete adjoint recursion. Show that this recursion is a
       Heun step of the adjoint ODE from $t_{k+1}$ to $t_k$ in which
       $J(\mathbf{x}_{k+1})$ is replaced by $J(\mathbf{u}_k)$. At which order
       in $h$ do the two updates differ?
    1. For a linear field $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$, one RK4
       step multiplies the state by $R(hA)$, with $R$ as in exercise
       :exref:`ex_mdl-odes-solvers-rotating-modes`c. Show that the discrete
       adjoint of RK4 is an RK4 step of the adjoint ODE.
    1. Replace the Euler step of `loss_and_grad_unrolled` by an RK4 step,
       and compute the gradient of the unrolled loss by central finite
       differences instead of a hand-written backward sweep. Measure its gap
       to the continuous-adjoint gradient `g_cont` for
       $n = 5, 10, 20, 40, 80$ steps. The trapezoid rule that computes
       `g_cont` in the `#odes-solvers-adjoint-check` cell uses $m = 2000$
       intervals and is accurate to only about $10^{-7}$, so refine it first.
       Which slope do you measure? Which order does the gradient computed
       through a solver of order $p$ inherit?
   :exlabel:`ex_mdl-odes-solvers-re-derive-adjoint-equations`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §29.2]{.kicker}

**Neural ODEs, Adjoints, and Continuous Flows**
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Neural ODEs and continuous flows]{.dtitle}

[residual updates, discrete and continuous adjoints, density traces]{.dsub}
:::
:::

::: {.slide title="A residual update has Euler's algebraic form"}
[Architecture]{.kicker}

$\mathbf x_{l+1} = \mathbf x_l + \mathbf f_{\theta_l}(\mathbf x_l)$ is one
Euler-shaped update at $h=1$. A continuous-depth limit additionally requires a
consistent time-indexed field and shrinking steps:

@fig:mdl-dyn-resnet-as-euler
:::

::: {.slide title="The Neural ODE limit"}
[Architecture]{.kicker}

Shrink the step and the stack becomes
$\dot{\mathbf x}=\mathbf f_\theta(\mathbf x,t)$,
$\mathbf x(T)=\mathbf x_0+\int_0^T\mathbf f_\theta\,dt$. A 1-hidden-layer net
learns a circle→ellipse flow:

@!odes-solvers-neural-ode-train
::: {.d2l-note}
The exact flow of a Lipschitz field is invertible over intervals with unique
forward and backward solutions. Its finite numerical approximation need not be.
:::
:::

::: {.slide title="Continuous and discrete adjoints optimize different objects"}
[Adjoint]{.kicker}

Run an adjoint $\mathbf a(t)=\partial L/\partial\mathbf x(t)$ backward:

$$\dot{\mathbf a} = -\Bigl(\tfrac{\partial\mathbf f}{\partial\mathbf x}\Bigr)^{\!\top}\mathbf a,
\qquad \frac{\partial L}{\partial\theta} = \int_0^T
\Bigl(\tfrac{\partial\mathbf f}{\partial\theta}\Bigr)^{\!\top}\mathbf a\,dt.$$

@!odes-solvers-adjoint-check
::: {.d2l-note .rule}
Backpropagation through the solver is the exact discrete adjoint. Integrating
the continuous adjoint can save stored states through recomputation, but its
gradient depends on solver tolerances and can drift from the discrete one.
:::
:::

::: {.slide title="Densities along the flow"}
[Continuous flows]{.kicker}

A flow transports density by the **trace** of its Jacobian, replacing a
log-determinant:

$$\frac{d}{dt}\log p_t(\mathbf x(t)) = -\operatorname{tr}\Bigl(\tfrac{\partial\mathbf f}{\partial\mathbf x}\Bigr).$$

. . .

Hutchinson's estimator $\operatorname{tr}(M)=\mathbb E[\boldsymbol\epsilon^\top M\boldsymbol\epsilon]$
computes the trace with one vector–Jacobian product, as used by FFJORD:

@!odes-solvers-cnf-trace
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- A residual update has Euler's form; a consistent shrinking-step family can converge to a Neural ODE.
- Discrete backprop differentiates the solver program; the continuous adjoint trades saved states for recomputation and numerical error.
- Density flows by $-\operatorname{tr}(\partial\mathbf f/\partial\mathbf x)$; each Hutchinson trace sample requires one VJP.

::: {.d2l-note}
Next we add **noise** to the velocity field: stochastic differential
equations.
:::
:::
