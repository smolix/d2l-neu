# Neural ODEs, Adjoints, and Continuous Flows
:label:`sec_mdl-neural-odes-cnf`

A residual block adds a correction to its input, and so does a step of
Euler's method (:numref:`sec_mdl-euler-runge-kutta`). If a residual network
(:numref:`sec_resnet`) is a numerical solution of a differential equation,
what do we gain by taking the equation itself as the model, and what does it
cost?

Generative modelling gives one answer. A normalizing flow is a network that
can be inverted, so that every output comes from exactly one input. The
density of its outputs then follows from the change-of-variables formula,
which asks for the determinant of the network's Jacobian at every sample, a
matrix with $3072$ rows for a small colour image of $32\times32$ pixels.
Discrete flows avoid this cost by restricting the form of their layers. A
flow in continuous time avoids it differently: the determinant turns into a
trace, and the trace can be estimated with one backward pass.

The model is a differential equation
$\dot{\mathbf{x}}=\mathbf{f}_{\boldsymbol\theta}(\mathbf{x},t)$ whose
solution at a final time is the output, a *Neural ODE*
:cite:`Chen.Rubanova.Bettencourt.ea.2018`. Training it and computing its
density each come down to integrating one more equation alongside the state.
For training, that equation is the *adjoint* equation, which optimal control
developed in the 1950s to ask how the cost of a trajectory changes with its
controls :cite:`Pontryagin.Boltyanskii.Gamkrelidze.ea.1962`. For the density,
it is an integral of the trace of the Jacobian, and a model that computes it
is a *continuous normalizing flow* :cite:`Grathwohl.Chen.Bettencourt.ea.2018`.
Compared with a fixed residual stack, the model gains an invertible map, a
computable density, and a step count that is no longer part of the model. It
loses the ability to reverse order on the line, and each gain has a catch
that this section measures: a trained field is tied to the solver it was
trained with, the memory-free gradient drifts, and the cheap trace is noisy.

We train such a model through a solver, derive the adjoint and compare it
with backpropagation, and then follow a density along a flow. Three results
are worth remembering: the adjoint equations, the fact that backpropagation
through a solver discretizes them, and the instantaneous change of variables
:eqref:`eq_mdl-ode-instant-cov`. A reader who wants only the last, on which
the likelihoods of diffusion models rest, can go straight to
:numref:`sec_mdl-continuous-normalizing-flows`. Order preservation and the
subsection on memory and drift can be skipped on a first reading.
Notation: the field
$\mathbf{f}_{\boldsymbol\theta}$ has parameters $\boldsymbol\theta$ and
Jacobian $\mathbf J=\partial\mathbf{f}/\partial\mathbf{x}$, $\Phi_t$ is its flow
map, a solver takes $N$ steps of size $h$ to the horizon $T=Nh$,
$\mathcal{L}$ is a loss, $\mathbf{a}(t)$ the adjoint and $p_t$ the density.

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-2}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-2}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-2}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-2}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

The adjoint and density checks reuse the spiral field $\mathbf A$ and the starting
point $\mathbf{x}_0$ of :numref:`sec_mdl-odes-solvers`, together with two
tools from there: the matrix exponential $e^{\mathbf Bs}=\mathbf Ve^{\boldsymbol\Lambda s}\mathbf V^{-1}$ from
the eigendecomposition $\mathbf B=\mathbf V\boldsymbol\Lambda \mathbf V^{-1}$, and the classical fourth-order
Runge--Kutta solver (RK4).

<!-- WRITER: the cell below now defines rng (the notebook's one generator,
unseeded), A, x0, expm_eig(C, s) (generic matrix renamed B -> C), and three
solver tools: euler_step(f, x, h) and rk4_step(f, x, h), one step each, and
solve(step, f, x, T, N), N steps of size T/N (T < 0 runs the field backward).
Forward Euler is from sec_mdl-euler-runge-kutta as well as RK4. Every later cell
(training in four frameworks, the trace integral) integrates through solve(). -->

```{.python .input #mdl-neural-odes-neural-odes-adjoints-and-continuous-flows-1}
rng = np.random.default_rng()
A = np.array([[-0.5, -1.0], [1.0, -0.5]])      # spiral sink: eigenvalues -0.5 +- i
x0 = np.array([2.0, 0.0])

def expm_eig(C, s):
    """e^{Cs} via the eigendecomposition C = V diag(mu) V^{-1}."""
    mu, V = np.linalg.eig(C)
    return (V @ np.diag(np.exp(mu * s)) @ np.linalg.inv(V)).real

def euler_step(f, x, h):
    """One forward-Euler step of size h for dx/dt = f(x)."""
    return x + h * f(x)

def rk4_step(f, x, h):
    """One classical Runge-Kutta step of size h for dx/dt = f(x)."""
    k1 = f(x)
    k2 = f(x + h / 2 * k1)
    k3 = f(x + h / 2 * k2)
    k4 = f(x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def solve(step, f, x, T, N):
    """N steps of size T/N from time 0 to time T; T < 0 runs the field backward."""
    for _ in range(N):
        x = step(f, x, T / N)
    return x
```

## Neural ODEs
:label:`sec_mdl-neural-odes`

A Neural ODE is what a residual network becomes when its depth turns into
time.

### From Residual Blocks to a Continuous Model

A residual blockWe start from the network we already have, a stack of residual blocks, and
ask which continuous model it approximates. A residual block
:cite:`He.Zhang.Ren.ea.2016` and a forward-Euler step of the ODE
$\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},t)$ read

$$
\mathbf{x}_{l+1}=\mathbf{x}_l+\mathbf{f}_{\boldsymbol\theta_l}(\mathbf{x}_l)
\qquad\textrm{and}\qquad
\mathbf{x}_{n+1}=\mathbf{x}_n+h\,\mathbf{f}(\mathbf{x}_n,t_n).
$$
:eqlabel:`eq_mdl-ode-resnet-euler`

With $h=1$ and $\mathbf{f}(\cdot,t_l)=\mathbf{f}_{\boldsymbol\theta_l}$ the
two updates are the same formula. A ResNet whose blocks have different
weights is therefore a time-dependent field sampled at the layer times and
integrated by Euler's method with unit steps
(:numref:`fig_mdl-dyn-resnet-as-euler` draws the case in which all blocks
share one field).

![Left: a stack of residual blocks that share one field, each adding the learned branch to an identity skip. Right: the same updates read as forward-Euler steps of size $h=1$, beside the exact flow of the same field, from which they drift.](../img/mdl-dyn-resnet-as-euler.svg)
:label:`fig_mdl-dyn-resnet-as-euler`

To take the limit, keep one parameter vector $\boldsymbol\theta$ for all
times and let the steps shrink. (The field may still depend on $t$; a ResNet
with distinct blocks is a field that varies in time.) With $N$ steps of size
$h=T/N$ at the times $t_n=nh$,

$$
\mathbf{x}_{n+1}=\mathbf{x}_n+h\,\mathbf{f}_{\boldsymbol\theta}(\mathbf{x}_n,t_n),
\qquad
\mathbf{x}_N\;\xrightarrow{\;N\to\infty\;}\;
\mathbf{x}(T)=\mathbf{x}_0+\int_0^T\mathbf{f}_{\boldsymbol\theta}(\mathbf{x}(t),t)\,dt .
$$
:eqlabel:`eq_mdl-ode-neural-ode`

The convergence follows from Euler's global error bound in
:numref:`sec_mdl-euler-runge-kutta`: for a smooth field that is Lipschitz in
$\mathbf{x}$, the Euler iterate after $N$ steps is within $O(h)$ of the exact
solution. The **Neural ODE** is the limit object: the model whose output is
the solution $\mathbf{x}(T)$ of
$\dot{\mathbf{x}}=\mathbf{f}_{\boldsymbol\theta}(\mathbf{x},t)$ with
$\mathbf{x}(0)=\mathbf{x}_0$. The parameters define the field. The solver
(Euler, RK4, an adaptive method) is a choice made afterwards, which changes
the computed map, its cost and its error, but not the model.

The exact map and the computed map can differ in kind, not only in accuracy.
The field $\dot{x}=-x$ has the exact flow $\Phi_t(x)=e^{-t}x$, a bijection of
$\mathbb{R}$ for every $t$. One Euler step with $h=1$ computes
$x+1\cdot(-x)=0$ and maps every input to zero. The exact map, by contrast,
is always invertible when the field is continuous in $t$ and globally
Lipschitz in $\mathbf{x}$: the reversed path $\mathbf{y}(s)=\mathbf{x}(T-s)$
solves $\dot{\mathbf{y}}=-\mathbf{f}(\mathbf{y},T-s)$, whose field is again
Lipschitz, so it is the only solution from $\mathbf{x}(T)$
(:numref:`sec_mdl-ode-existence-uniqueness`) and it recovers the start.

### Order Preservation on the Line
:label:`subsec_mdl-flow-order`

Invertibility is a gain, but it is also a constraint. Is a Neural ODE at
least as expressive as the residual stack it came from? On the real line it
is not, and a picture shows why. In the $(t,x)$ plane, trajectories
of a Lipschitz field never cross. Two trajectories that swapped $-1$ and $1$
would have to meet on the way (panel (a) of :numref:`fig_mdl-dyn-flow-order`).

**Proposition (flows on $\mathbb{R}$ preserve order).** *Let $f(x,t)$ be
continuous in $t$ and globally Lipschitz in $x$, and write $\Phi_t(x)$ for
the solution at time $t$ started from $x$ at time $0$. Then for every $T>0$
the map $\Phi_T$ is strictly increasing.*

**Proof.** Let $x<y$ and suppose $\Phi_T(x)\ge\Phi_T(y)$. The function
$g(t)=\Phi_t(y)-\Phi_t(x)$ is continuous with $g(0)>0$ and $g(T)\le0$, so by
the intermediate value theorem $g(s)=0$ for some $s\in(0,T]$: the two
trajectories meet at time $s$ at a point $z$. Now run time backward from $s$.
Both reversed trajectories $\tau\mapsto\Phi_{s-\tau}(x)$ and
$\tau\mapsto\Phi_{s-\tau}(y)$ solve the same initial-value problem
$\dot{w}=-f(w,s-\tau)$, $w(0)=z$, whose field is again Lipschitz. By
Picard--Lindelöf that problem has only one solution, so the two reversed
trajectories coincide, and at $\tau=s$ this says $x=y$, a contradiction.
$\blacksquare$

The proof has one idea, to apply uniqueness backward from the meeting time.
As a consequence, no Neural ODE on $\mathbb{R}$ realizes $x\mapsto-x$: that
map sends $-1$ to $1$ and $1$ to $-1$, which reverses their order.

The residual block $x+f(x)$ with $f(x)=-2x$ does realize it, in one step
(panel (b) of :numref:`fig_mdl-dyn-flow-order`). That block is the Euler step with $h=1$
of $\dot{x}=-2x$, whose exact flow $e^{-2t}x$ never reverses order. The two
statements agree because one Euler step $x+hf(x)$ of a field with Lipschitz
constant $L$ is guaranteed to be increasing when $hL<1$: for $x>y$,

$$
\big(x+hf(x)\big)-\big(y+hf(y)\big)\;\ge\;(x-y)-hL\,(x-y)\;=\;(1-hL)(x-y),
$$

which is positive when $hL<1$. Here $hL=2$, so the step may reverse order.
A finite step is not a flow, and the proposition does not constrain it.

One more dimension removes the obstruction. Let
$\mathbf K=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$, which
satisfies $\mathbf K^2=-\mathbf{I}$, so that the exponential series splits into
$e^{s\mathbf K}=\cos s\,\mathbf{I}+\sin s\,\mathbf K$, the rotation by the angle $s$. The
field $\dot{\mathbf{x}}=\pi \mathbf K\mathbf{x}$ has the time-one map

$$
\Phi_1=e^{\pi \mathbf K}=\cos\pi\,\mathbf{I}+\sin\pi\,\mathbf K=-\mathbf{I}.
$$

It carries $(x,0)$ to $(-x,0)$ along half circles, and trajectories pass each
other in the plane without meeting (panel (c) of :numref:`fig_mdl-dyn-flow-order`).
Adding dimensions in this way is called *augmentation*
:cite:`Dupont.Doucet.Teh.2019`: the input $x$ is embedded as $(x,0)$, the
flow runs in the plane, and the first coordinate is read out. A flow is a
continuous rearrangement of space, and on a line there is no room for two
points to pass.

![(a) On the line, trajectories of a Lipschitz field never cross (here $\dot x=-x$), so the time-$T$ map preserves order; the dashed pair that swaps $-1$ and $1$ would need two solutions through one point. (b) The residual step $x+f(x)$ with $f(x)=-2x$ swaps the points in one step: it is an Euler step with $hL=2$, not a flow. (c) One extra dimension removes the obstruction: the rotation by $\pi$ carries $(x,0)$ to $(-x,0)$ along half circles that pass without meeting (colours as in (b)).](../img/mdl-dyn-flow-order.svg)
:label:`fig_mdl-dyn-flow-order`

### Training Through the Solver

The forward pass of a Neural ODE is a loop of $N$ solver steps. The direct
way to train it is to *discretize, then differentiate*: write the solver into
the computation graph, compute the loss on $\mathbf{x}_N$, and backpropagate
through the unrolled steps. With Euler as the solver the program is a ResNet
of $N$ blocks that share one set of weights, so ordinary automatic
differentiation trains it.

The task is deliberately easy: carry $64$ points of the unit circle to the
matching points of the ellipse centred at $(1.5,0)$ with semi-axes $1.2$ and
$0.4$, a stretch and a shift. The exact flow of the affine field
$\dot{\mathbf{x}}=\mathrm{diag}(\ln1.2,\ln0.4)\,\mathbf{x}+\mathbf{c}$ does
this without error for a suitable $\mathbf{c}$. The question is whether
training finds a flow or only a ten-step program. The field has
one hidden layer of $32$ tanh units, the loss is the mean squared error at
the endpoints, the solver takes $N=10$ Euler steps of size $h=0.1$, and Adam
(:numref:`sec_mdl-adaptive-stochastic-methods`) runs $401$ full-batch
iterations at learning rate $0.05$, lowered to $0.005$ from iteration $300$.

<!-- WRITER: the training now runs twice per tab from one shared initialization
(seed drawn in the shared cell, `solvers = {'Euler': euler_step, 'RK4': rk4_step}`):
once through 10 Euler steps (as before) and once through 10 RK4 steps (four field
evaluations per step, so four times the work per training step). Same field,
loss, Adam schedule and 401 iterations for both. -->

```{.python .input #mdl-neural-odes-training-through-the-solver-2}
angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
X0_ring = np.stack([np.cos(angle), np.sin(angle)], axis=1)            # circle
X1_ring = np.stack([1.5 + 1.2 * np.cos(angle), 0.4 * np.sin(angle)], axis=1)
solvers = {'Euler': euler_step, 'RK4': rk4_step}
seed = int(rng.integers(2**31))                # one initialization for both solvers
```

```{.python .input #odes-solvers-neural-ode-train}
#@tab pytorch
X, Y = torch.tensor(X0_ring).float(), torch.tensor(X1_ring).float()

def train(name, step, seed):
    """Fit the field through 10 steps of one solver, from the given seed."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 2))
    for layer in net[::2]:                     # Glorot-uniform weights, zero biases
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [300], 0.1)  # 0.005 from 300
    losses = []
    for it in range(401):
        loss = ((solve(step, net, X, 1.0, 10) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 200 == 0:
            losses.append(loss.item())
    print(f'{name}-trained: loss', ', '.join(f'{l:.1e}' for l in losses),
          'at iterations 0, 200, 400')
    return net

trained = {name: train(name, step, seed) for name, step in solvers.items()}
```

```{.python .input #odes-solvers-neural-ode-train}
#@tab mxnet
X, Y = mnp.array(X0_ring, dtype='float32'), mnp.array(X1_ring, dtype='float32')

def train(name, step, seed):
    """Fit the field through 10 steps of one solver, from the given seed."""
    npx.random.seed(seed)
    net = nn.HybridSequential()
    net.add(nn.Dense(32, activation='tanh'), nn.Dense(2))
    net.initialize(init.Xavier())              # Glorot-uniform weights, zero biases
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.05})
    losses = []
    for it in range(401):
        if it == 300:
            trainer.set_learning_rate(0.005)   # tenfold smaller for the last 100
        with autograd.record():
            loss = ((solve(step, net, X, 1.0, 10) - Y) ** 2).mean()
        loss.backward()
        trainer.step(1)
        if it % 200 == 0:
            losses.append(float(loss))
    print(f'{name}-trained: loss', ', '.join(f'{l:.1e}' for l in losses),
          'at iterations 0, 200, 400')
    return net

trained = {name: train(name, step, seed) for name, step in solvers.items()}
```

```{.python .input #odes-solvers-neural-ode-train}
#@tab tensorflow
X, Y = tf.constant(X0_ring, tf.float32), tf.constant(X1_ring, tf.float32)

def train(name, step, seed):
    """Fit the field through 10 steps of one solver, from the given seed."""
    tf.keras.utils.set_random_seed(seed)
    net = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='tanh'),
                               tf.keras.layers.Dense(2)])  # Glorot-uniform, zero biases
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([299], [0.05, 0.005])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)  # 0.005 from iteration 300

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((solve(step, net, X, 1.0, 10) - Y) ** 2)
        opt.apply_gradients(zip(tape.gradient(loss, net.trainable_variables),
                                net.trainable_variables))
        return loss

    losses = []
    for it in range(401):
        loss = train_step()
        if it % 200 == 0:
            losses.append(float(loss))
    print(f'{name}-trained: loss', ', '.join(f'{l:.1e}' for l in losses),
          'at iterations 0, 200, 400')
    return tf.function(net)                    # a compiled field evaluates faster

trained = {name: train(name, step, seed) for name, step in solvers.items()}
```

```{.python .input #odes-solvers-neural-ode-train}
#@tab jax
X, Y = jnp.array(X0_ring), jnp.array(X1_ring)
k1, k2 = jax.random.split(jax.random.key(seed))
glorot = jax.nn.initializers.glorot_uniform()  # Glorot-uniform weights, zero biases
params0 = {'W1': glorot(k1, (2, 32)), 'b1': jnp.zeros(32),
           'W2': glorot(k2, (32, 2)), 'b2': jnp.zeros(2)}
opt = optax.adam(optax.piecewise_constant_schedule(0.05, {300: 0.1}))

def net(params, X):
    return jnp.tanh(X @ params['W1'] + params['b1']) @ params['W2'] + params['b2']

def flow(params, step, X, T, N):
    """solve() as a fori_loop: N steps of size T/N along the field net(params)."""
    f = lambda x: net(params, x)
    return jax.lax.fori_loop(0, N, lambda n, X: step(f, X, T / N), X)

@jax.jit(static_argnames='step')
def train_step(params, opt_state, step):
    loss, grads = jax.value_and_grad(
        lambda p: ((flow(p, step, X, 1.0, 10) - Y) ** 2).mean())(params)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

def train(name, step):
    """Fit the field through 10 steps of one solver, from the shared params0."""
    params, opt_state, losses = params0, opt.init(params0), []
    for it in range(401):
        params, opt_state, loss = train_step(params, opt_state, step)
        if it % 200 == 0:
            losses.append(float(loss))
    print(f'{name}-trained: loss', ', '.join(f'{l:.1e}' for l in losses),
          'at iterations 0, 200, 400')
    return params

trained = {name: train(name, step) for name, step in solvers.items()}
```

The loss falls from order one to well below $10^{-4}$, typically a few times
$10^{-6}$.

<!-- WRITER: final losses (iteration 400) in 22 development runs, both arms,
four tabs: 6.8e-7 to 8.4e-6 ("of order 10^-6"); iteration 0: 1.1 to 1.8. In each
run the two arms end within a factor of 1.5 of each other. -->

Two tests tell a flow from a ten-step program. First, integrate the trained field with $N=100$ and $N=1000$ steps
($Nh=1$ throughout); the larger $N$, the closer the computed map is to the
exact time-one flow of the field. Before running the cell, guess what this
does to the endpoint error. Second, integrate forward $N$ steps and
then backward $N$ steps with
$\mathbf{x}\leftarrow\mathbf{x}-h\,\mathbf{f}_{\boldsymbol\theta}(\mathbf{x})$,
and measure how far the round trip lands from where it started.

```{.python .input #mdl-neural-odes-training-through-the-solver-1}
#@tab pytorch
dist = lambda P, Q: torch.linalg.vector_norm(P - Q, dim=1).mean().item()
print('   N   Euler-trained   RK4-trained   Euler round trip')
with torch.inference_mode():
    for N in (10, 100, 1000):
        err = [dist(solve(solvers[k], trained[k], X, 1.0, N), Y) for k in solvers]
        fwd = solve(euler_step, trained['Euler'], X, 1.0, N)
        back = solve(euler_step, trained['Euler'], fwd, -1.0, N)
        print(f'{N:4d}   {err[0]:13.1e}   {err[1]:11.1e}   {dist(back, X):16.1e}')
```

```{.python .input #mdl-neural-odes-training-through-the-solver-1}
#@tab mxnet
dist = lambda P, Q: float(mnp.sqrt(((P - Q) ** 2).sum(axis=1)).mean())
print('   N   Euler-trained   RK4-trained   Euler round trip')
for N in (10, 100, 1000):
    err = [dist(solve(solvers[k], trained[k], X, 1.0, N), Y) for k in solvers]
    fwd = solve(euler_step, trained['Euler'], X, 1.0, N)
    back = solve(euler_step, trained['Euler'], fwd, -1.0, N)
    print(f'{N:4d}   {err[0]:13.1e}   {err[1]:11.1e}   {dist(back, X):16.1e}')
```

```{.python .input #mdl-neural-odes-training-through-the-solver-1}
#@tab tensorflow
dist = lambda P, Q: float(tf.reduce_mean(tf.norm(P - Q, axis=1)))
print('   N   Euler-trained   RK4-trained   Euler round trip')
for N in (10, 100, 1000):
    err = [dist(solve(solvers[k], trained[k], X, 1.0, N), Y) for k in solvers]
    fwd = solve(euler_step, trained['Euler'], X, 1.0, N)
    back = solve(euler_step, trained['Euler'], fwd, -1.0, N)
    print(f'{N:4d}   {err[0]:13.1e}   {err[1]:11.1e}   {dist(back, X):16.1e}')
```

```{.python .input #mdl-neural-odes-training-through-the-solver-1}
#@tab jax
dist = lambda P, Q: float(jnp.linalg.norm(P - Q, axis=1).mean())
print('   N   Euler-trained   RK4-trained   Euler round trip')
for N in (10, 100, 1000):
    err = [dist(flow(trained[k], solvers[k], X, 1.0, N), Y) for k in solvers]
    fwd = flow(trained['Euler'], euler_step, X, 1.0, N)
    back = flow(trained['Euler'], euler_step, fwd, -1.0, N)
    print(f'{N:4d}   {err[0]:13.1e}   {err[1]:11.1e}   {dist(back, X):16.1e}')
```

<!-- WRITER: the cell above prints one table: per N (10, 100, 1000; Nh = 1),
the endpoint error of the Euler-trained field integrated with Euler, of the
RK4-trained field integrated with RK4, and the Euler arm's forward-backward round
trip. 22 development runs, four tabs (3 notebook runs per tab, plus 10 runs of
the same code): Euler-trained 1.1e-3 to 3.5e-3 at N = 10, 1.7e-2 to 1.8e-2 at
100, 1.9e-2 at 1000 in every run; RK4-trained the same value at all three N
within each run (equal to the two printed digits in all 22 runs), 1.1e-3 to
3.7e-3 across runs; Euler round trip 5.4e-2 to 5.6e-2, 5.5e-3 to 5.7e-3,
5.5e-4 to 5.7e-4. At N = 10 the two arms are equally accurate. So: the field trained through RK4 is fitted to its flow, and its step
count is free after training. -->

At the training step count the endpoints lie within $0.01$ of their
targets, typically a few thousandths. With finer steps the endpoint error
*rises* to about $0.02$ and then stops changing. Training fitted the ten-step program. The exact
time-one flow of the same field, which $N=1000$ approximates, lands about
$0.02$ from the targets, under $2\%$ of the ellipse's long semi-axis. The
field defines a flow close to its ten-step program, not equal to it.
For the step count to be free after training, the field must be fitted to
the flow and not to one program: by training through a finer or
higher-order solver, or by a regression that involves no solver at all,
which is how :numref:`sec_mdl-score-matching-diffusion-flow` and
:numref:`sec_mdl-flow-matching-transport` learn their fields.

The round trip misses its start by about $0.055$ at $N=10$, $0.0056$ at
$N=100$ and $0.00056$ at $N=1000$, tenfold less per tenfold more steps. Each
pair of forward and backward Euler steps leaves a residual of order $h^2$,
and $N$ of them add up to $O(h)$, Euler's order. The exact flow is
invertible, whereas the numerical map becomes invertible only in the limit
$h\to0$.

## The Adjoint Method
:label:`sec_mdl-adjoint-method`

Backpropagation through $N$ solver steps gives the gradient of the
$N$-step program. The adjoint method differentiates the exact flow instead,
by integrating one linear ODE backward in time. It is first of all what
backpropagation through a solver converges to as the steps shrink. It is
also an alternative implementation: backpropagation stores the $N$
intermediate states, whereas the adjoint ODE can recompute them on the way
back, at a loss of stability that the last subsection quantifies. Its central object is the
*adjoint state* $\mathbf{a}(t)=\partial\mathcal{L}/\partial\mathbf{x}(t)$,
the gradient of the loss with respect to the state at time $t$. This is the
adjoint $\bar a$ of :eqref:`eq_mdl-adjoint`, the total derivative of the
output with respect to a node, with one node $\mathbf{x}(t)$ for every
instant: reverse mode carries it from layer to layer
(:numref:`sec_mdl-matrix-calculus-autodiff`), and here it is carried
through time.

### The Scalar Case

Before the general statement we work one example to the end, an example in
which every quantity has a closed form. Take $\dot{x}=\theta x$ with $x(0)=x_0$ and the
loss $\mathcal{L}=\tfrac12x(T)^2$ on the endpoint. Directly,
$x(T)=e^{\theta T}x_0$, so $\partial x(T)/\partial\theta=T\,x(T)$ and

$$
\frac{\partial\mathcal{L}}{\partial\theta}=x(T)\,\frac{\partial x(T)}{\partial\theta}=T\,x(T)^2 .
$$

With $\theta=T=x_0=1$ this is $e^2=7.389$.

The adjoint reaches the same number by another route. A nudge $\delta$ to
the state at time $t$ grows to $e^{\theta(T-t)}\delta$ by time $T$ and
changes the loss by $x(T)\,e^{\theta(T-t)}\delta$, so
$a(t)=e^{\theta(T-t)}x(T)$. This adjoint solves the ODE $\dot{a}=-\theta a$
backward from the final value $a(T)=x(T)$, which is the derivative of the
loss $\ell(x)=\tfrac12x^2$ at $x(T)$. A change of $\theta$ pushes the
trajectory at the rate $\partial f/\partial\theta=x$, and the adjoint
converts each push into a change of the loss. Summing over time,

$$
\int_0^T a(t)\,\frac{\partial f}{\partial\theta}\,dt
=\int_0^T e^{\theta(T-t)}x(T)\;e^{\theta t}x_0\,dt
=\int_0^T x(T)^2\,dt
=T\,x(T)^2 .
$$
:eqlabel:`eq_mdl-ode-scalar-adjoint`

The integrand is constant: $x$ grows forward, $a$ shrinks by the same factor
when read forward, and their product stays at $x(T)^2$
(:numref:`fig_mdl-dyn-adjoint-timeline`). The gradient is the area under it.

![The scalar adjoint for $\dot x=\theta x$ at $\theta=1$, $x_0=1$, $T=1$: the state grows forward, the adjoint is integrated backward from the loss gradient $a(1)=x(1)$, and their product is constant; the parameter gradient is the shaded area, $Tx(T)^2=e^2$.](../img/mdl-dyn-adjoint-timeline.svg)
:label:`fig_mdl-dyn-adjoint-timeline`

### The Adjoint Equations

The general case keeps the shape of the scalar one: an adjoint that runs
backward from the gradient of the loss, and a parameter gradient that
integrates the adjoint against $\partial\mathbf{f}/\partial\boldsymbol\theta$.

**Proposition (the adjoint equations).** *Let $\mathbf{f}$ be continuously
differentiable in $\mathbf{x}$ and $\boldsymbol\theta$, let $\mathbf{x}(t)$
solve $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},\boldsymbol\theta,t)$ (the
field $\mathbf{f}_{\boldsymbol\theta}$, with its parameters written as an
argument) on $[0,T]$ from a fixed $\mathbf{x}(0)$, and let
$\mathcal{L}=\ell(\mathbf{x}(T))$ for a differentiable $\ell$. Then the
adjoint state satisfies the linear ODE*

$$
\dot{\mathbf{a}}(t) = -\left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right)^{\!\top}\! \mathbf{a}(t),
\qquad
\mathbf{a}(T) = \nabla \ell(\mathbf{x}(T)),
$$
:eqlabel:`eq_mdl-ode-adjoint`

*integrated backward from $T$, and the parameter gradient is the integral*

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol\theta}
= \int_0^T \left(\frac{\partial \mathbf{f}}{\partial \boldsymbol\theta}\right)^{\!\top}\! \mathbf{a}(t)\, dt,
$$
:eqlabel:`eq_mdl-ode-adjoint-grad`

*with both partial derivatives evaluated along the trajectory
$(\mathbf{x}(t),\boldsymbol\theta,t)$* :cite:`Chen.Rubanova.Bettencourt.ea.2018`.

**Proof (variational).** Perturb the parameters by an infinitesimal
$\delta\boldsymbol\theta$. Because $\mathbf{f}$ is $C^1$, the solution is
differentiable in $\boldsymbol\theta$, and the first-order perturbation
$\delta\mathbf{x}(t)$ of the trajectory obeys the *linearization* of the
dynamics (differentiate the ODE in $\boldsymbol\theta$ and use the chain
rule):

$$
\delta\dot{\mathbf{x}}
= \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\,\delta\mathbf{x}
+ \frac{\partial \mathbf{f}}{\partial \boldsymbol\theta}\,\delta\boldsymbol\theta,
\qquad
\delta\mathbf{x}(0) = \mathbf{0}.
$$

Define $\mathbf{a}(t)$ by the backward ODE :eqref:`eq_mdl-ode-adjoint` and
differentiate the pairing $\mathbf{a}^\top\delta\mathbf{x}$:

$$
\frac{d}{dt}\left(\mathbf{a}^\top \delta\mathbf{x}\right)
= \dot{\mathbf{a}}^\top \delta\mathbf{x} + \mathbf{a}^\top \delta\dot{\mathbf{x}}
= -\mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\,\delta\mathbf{x}
+ \mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\,\delta\mathbf{x}
+ \mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \boldsymbol\theta}\,\delta\boldsymbol\theta
= \mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \boldsymbol\theta}\,\delta\boldsymbol\theta .
$$

The state-coupling terms cancel, and the adjoint ODE was defined so that
they would. Integrate from $0$ to $T$, using
$\delta\mathbf{x}(0)=\mathbf{0}$ at the left endpoint:

$$
\mathbf{a}(T)^\top \delta\mathbf{x}(T)
= \int_0^T \mathbf{a}(t)^\top \frac{\partial \mathbf{f}}{\partial \boldsymbol\theta}\, dt \;\delta\boldsymbol\theta .
$$

The left side is $\nabla\ell(\mathbf{x}(T))^\top\delta\mathbf{x}(T)=\delta\mathcal{L}$,
the first-order change of the loss; the right side is linear in
$\delta\boldsymbol\theta$ with coefficient :eqref:`eq_mdl-ode-adjoint-grad`.
It remains to identify $\mathbf{a}$ with the adjoint state. Run the same
argument with $\delta\boldsymbol\theta=\mathbf{0}$ and a nudge
$\delta\mathbf{x}(s)$ injected at time $s$: the pairing
$\mathbf{a}^\top\delta\mathbf{x}$ is then constant on $[s,T]$, so
$\delta\mathcal{L}=\mathbf{a}(T)^\top\delta\mathbf{x}(T)=\mathbf{a}(s)^\top\delta\mathbf{x}(s)$,
and the solution of :eqref:`eq_mdl-ode-adjoint` is
$\partial\mathcal{L}/\partial\mathbf{x}(s)$. $\blacksquare$

The backward ODE is linear in $\mathbf{a}$, with the transposed Jacobian as
its matrix. Each of its steps computes
$\mathbf{a}^\top\partial\mathbf{f}/\partial\mathbf{x}$, one vector--Jacobian
product, which costs one reverse-mode pass through $\mathbf{f}$
(:numref:`sec_mdl-matrix-calculus-autodiff`). (A ReLU field is only
piecewise $C^1$, so the proposition does not cover it, although autograd
still differentiates the unrolled solver piece by piece. Smooth activations
such as tanh or GELU satisfy the hypothesis.)

### Backpropagation as the Discrete Adjoint

Backpropagation through the Euler program can now be set beside the adjoint
ODE, line by line. Differentiate the program
$\mathbf{x}_{k+1}=\mathbf{x}_k+h\,\mathbf{f}(\mathbf{x}_k,\boldsymbol\theta)$
by the chain rule, with $\mathbf{a}_k=\partial\mathcal{L}/\partial\mathbf{x}_k$.
The Jacobian of one step is $\mathbf{I}+h\mathbf J(\mathbf{x}_k)$, so

$$
\mathbf{a}_k=\big(\mathbf{I}+h\mathbf J(\mathbf{x}_k)\big)^{\!\top}\mathbf{a}_{k+1},
\qquad
\mathbf{a}_N=\nabla\ell(\mathbf{x}_N),
\qquad
\frac{\partial\mathcal{L}}{\partial\boldsymbol\theta}
=\sum_{k=0}^{N-1}h\left(\frac{\partial\mathbf{f}}{\partial\boldsymbol\theta}(\mathbf{x}_k)\right)^{\!\top}\mathbf{a}_{k+1}.
$$
:eqlabel:`eq_mdl-ode-discrete-adjoint`

Compare this with one explicit Euler step of the adjoint ODE
:eqref:`eq_mdl-ode-adjoint` from $t_{k+1}$ back to $t_k$:

$$
\mathbf{a}_k=\mathbf{a}_{k+1}+h\,\mathbf J(\mathbf{x}_{k+1})^\top\mathbf{a}_{k+1}.
$$

The two updates differ only in where the Jacobian is evaluated: at the start
$\mathbf{x}_k$ of the forward step or at its end $\mathbf{x}_{k+1}$. The sum in
:eqref:`eq_mdl-ode-discrete-adjoint` is a rectangle rule for the integral
:eqref:`eq_mdl-ode-adjoint-grad`. So backpropagation through a solver is a
*discrete adjoint*, and a solver applied to the adjoint ODE is the
*continuous adjoint*, discretized. The literature calls the two routes
discretize-then-optimize and optimize-then-discretize :cite:`Kidger.2022`.
The first is the exact gradient of the program, the second approximates the
gradient of the flow, and for Euler they differ at first order in $h$.
 Of
everything in this subsection, this is the statement to keep.

The scalar example shows both at once. For $\dot{x}=\theta x$ the Euler
program is $x_N=(1+h\theta)^Nx_0$, so
$\partial x_N/\partial\theta=Nh(1+h\theta)^{N-1}x_0=T\,x_N/(1+h\theta)$ and
the discrete adjoint gives
$\partial\mathcal{L}/\partial\theta=T\,x_N^2/(1+h\theta)$. With
$\theta=T=x_0=1$ this is $6.116$ at $N=10$, $7.244$ at $N=100$ and $7.374$
at $N=1000$, against the exact $e^2=7.389$. The gaps $1.27$, $0.145$ and
$0.0148$ shrink tenfold per tenfold $N$.

The spiral field of :numref:`sec_mdl-odes-solvers`, $\dot{\mathbf{x}}=\mathbf A\mathbf{x}$
with $\mathbf A=\left(\begin{smallmatrix}-0.5&-1\\1&-0.5\end{smallmatrix}\right)$,
gives the same check in two dimensions with every ingredient in closed form.
Take $\boldsymbol\theta$ to be the entries of $\mathbf A$ and
$\mathcal{L}=\tfrac12\|\mathbf{x}(T)\|^2$. The trajectory is
$e^{\mathbf At}\mathbf{x}_0$, the adjoint is
$\mathbf{a}(t)=e^{\mathbf A^\top(T-t)}\mathbf{x}(T)$ (the solution of
$\dot{\mathbf{a}}=-\mathbf A^\top\mathbf{a}$), and since
$\partial\mathbf{f}/\partial A_{ij}=x_j\mathbf{e}_i$, the gradient
:eqref:`eq_mdl-ode-adjoint-grad` becomes
$\partial\mathcal{L}/\partial \mathbf A=\int_0^T\mathbf{a}(t)\,\mathbf{x}(t)^\top dt$.
The cell writes the backward sweep :eqref:`eq_mdl-ode-discrete-adjoint` by
hand and tests it against central finite differences on the same program,
for $T=2$ and $N=1000$ steps.

```{.python .input #mdl-neural-odes-backpropagation-as-the-discrete-adjoint-1}
def loss_and_grad_unrolled(A, x0, T, N):
    """L = ||x_N||^2 / 2 through N unrolled Euler steps; dL/dA by hand-written
    reverse mode, the discrete adjoint."""
    h, xs = T / N, [x0]
    for _ in range(N):
        xs.append(xs[-1] + h * (A @ xs[-1]))   # forward sweep, tape the states
    a, gA = xs[-1].copy(), np.zeros_like(A)    # a_N = dL/dx_N = x_N
    for n in range(N - 1, -1, -1):             # backward sweep: one VJP per step
        gA += h * np.outer(a, xs[n])           # dL/dA += h a_{n+1} x_n^T
        a = a + h * (A.T @ a)                  # a_n = (I + hA)^T a_{n+1}
    return 0.5 * xs[-1] @ xs[-1], gA

T = 2.0
loss = lambda A: loss_and_grad_unrolled(A, x0, T, 1000)[0]
g_fd = np.array([(loss(A + 1e-5 * E) - loss(A - 1e-5 * E)) / 2e-5
                 for E in np.eye(4).reshape(4, 2, 2)]).reshape(2, 2)
gap_fd = np.abs(loss_and_grad_unrolled(A, x0, T, 1000)[1] - g_fd).max()
print(f'N =  1000: |backward sweep - central differences| = {gap_fd:.1e}')
n_quad = 2000                                  # continuous adjoint, trapezoid rule
ts = np.linspace(0.0, T, n_quad + 1)
xT = expm_eig(A, T) @ x0
AX = np.stack([np.outer(expm_eig(A.T, T - s) @ xT, expm_eig(A, s) @ x0)
               for s in ts])                   # a(s) x(s)^T
g_cont = (T / n_quad) * (AX[1:-1].sum(axis=0) + 0.5 * (AX[0] + AX[-1]))
for N in (100, 1000, 10000):
    gap = np.abs(loss_and_grad_unrolled(A, x0, T, N)[1] - g_cont).max()
    print(f'N = {N:5d}: |discrete - continuous adjoint|     = {gap:.1e}')
```

<!-- WRITER: #odes-solvers-adjoint-check is merged into the cell above (its id
retires). One cell now defines loss_and_grad_unrolled, checks it against central
differences at N = 1000 (first output line, 1.2e-10 in the store, "the sweep is
coded correctly") and then prints the gaps to the continuous adjoint at
N = 100, 1000, 10000 (2.9e-02, 2.8e-03, 2.8e-04). The paragraph between the two
former cells (left below, unchanged) needs to become one lead-in before the
cell and one reading after it. Step count N, step index n (code: `for n in
range(N)`, a_n). -->

The two agree to about $10^{-10}$, the accuracy of the finite differences
themselves, so the sweep is coded correctly. The distance that matters is
the one to the gradient of the exact flow. For it we evaluate the continuous
adjoint integral by the trapezoid rule on $2000$ intervals, with
$\mathbf{a}$ and $\mathbf{x}$ from the matrix exponential, and compare it
with the discrete adjoint for three step counts.

The gap falls from $2.9\times10^{-2}$ to $2.8\times10^{-3}$ to
$2.8\times10^{-4}$, tenfold per tenfold $N$, as in the scalar case.

### Memory and Drift

The adjoint ODE needs $\mathbf{x}(t)$ at every backward step, because the
Jacobian is evaluated there. There are two ways to supply it. Store the
forward states, which costs memory proportional to $N$, or re-integrate
$\dot{\mathbf{x}}=\mathbf{f}$ backward from $\mathbf{x}(T)$ alongside
$\mathbf{a}$, which keeps the memory constant in $N$ at the cost of one more
field evaluation per step. The smallest case shows that the second option
amplifies errors.

For $\dot{x}=-\lambda x$ with $\lambda>0$, the backward integration solves
$\dot{x}=+\lambda x$. An absolute error $\delta$ in the stored $x(T)$
therefore becomes $e^{\lambda T}\delta$ by the time it reaches $t=0$. Take
$\lambda=50$, the fast decay rate of the stiff example of
:numref:`sec_mdl-stiffness-implicit`, as one direction of a state whose
coordinates are of size one, and not along a coordinate axis. Each stored
coordinate is rounded at about $10^{-16}$ in double precision, and the
rounding has a component of that size along the fast direction, so
$\delta=10^{-16}$. The reconstructed start is then off by
$2.7\times10^{-11}$ at $T=0.25$, by $7\times10^{-6}$ at $T=0.5$ and by
$5\times10^{5}$ at $T=1$, larger than the state itself
(:exref:`ex_mdl-odes-solvers-backward-reconstruction` measures this growth).

For the spiral, whose eigenvalues have real part $-0.5$, the factor over
$T=2$ is only $e$. A strongly contracting forward flow is a strongly
expanding backward one, so re-integration saves memory at the cost of drift.
For stiff fields *checkpointing* is the sturdier choice: it stores the state
only at a few checkpoints and recomputes the states in between.

## Continuous Normalizing Flows
:label:`sec_mdl-continuous-normalizing-flows`

A Neural ODE moves points. A generative model must also say how probable
each output is, so we now follow a density along the flow.

### Density Along a Flow

Pushing samples through a flow turns it into a generator: draw
$\mathbf{x}(0)$ from a simple density $p_0$ and integrate. (The generators
of :numref:`sec_mdl-score-matching-diffusion-flow` and
:numref:`sec_mdl-flow-matching-transport` run such a flow backward in time,
from noise at the final time to data at $t = 0$; the density formulas below
hold in either direction.) Training such a
generator by maximum likelihood needs the density of what comes out. A
*normalizing flow* is an invertible network trained in this way, and its
density follows from the change-of-variables formula of
:numref:`sec_mdl-joint-distributions`: for an invertible differentiable map
$\mathbf{g}$ with Jacobian $\mathbf J_{\mathbf{g}}$,

$$
\log p_{\textrm{out}}(\mathbf{g}(\mathbf{x}))
= \log p_{\textrm{in}}(\mathbf{x}) - \log \left|\det \mathbf J_{\mathbf{g}}(\mathbf{x})\right| .
$$
:eqlabel:`eq_mdl-ode-discrete-cov`

The log-density drops by the logarithm of the factor by which $\mathbf{g}$
stretches volume near $\mathbf{x}$. A flow built from many layers needs one
log-determinant per layer and sample. A general determinant costs $O(d^3)$
operations, about $2\times10^{10}$ for the $d=3072$ of a $32\times32$ colour
image. Discrete flows therefore
restrict their layers to triangular Jacobians, whose determinant is the
product of the diagonal, or to identity-plus-low-rank Jacobians, such as
$\det(\mathbf{I}+\mathbf{u}\mathbf{v}^\top)=1+\mathbf{v}^\top\mathbf{u}$.
Both determinants cost $O(d)$.

In continuous time the formula simplifies. Start with one dimension.
The field $\dot{x}=\theta x$ on $\mathbb{R}$ has the flow
$\Phi_t(x)=e^{\theta t}x$, which stretches every interval by $e^{\theta t}$.
A density carried along is divided by the stretch, so
$\log p_t(x(t))=\log p_0(x_0)-\theta t$ and
$\tfrac{d}{dt}\log p_t(x(t))=-\theta$. The number $\theta$ is the
$1\times1$ Jacobian of the field, and therefore also its trace.

The spiral is the running example in two dimensions. Since
$\mathbf A=-\tfrac12\mathbf{I}+\mathbf K$ and $\mathbf{I}$ commutes with $\mathbf K$,
$e^{\mathbf At}=e^{-t/2}e^{t\mathbf K}$, a rotation followed by a shrink by $e^{-t/2}$. A disc
is carried to a disc of $e^{-t/2}$ times its radius, so its area is
multiplied by $e^{-t}$, and the density along each trajectory rises by
$e^{t}$. Here $\operatorname{tr}\mathbf A=-1$, so the rate of change of the
log-density is again minus the trace. :numref:`fig_mdl-dyn-divergence-volume`
draws this beside a field that preserves area and one that expands it.

![A disc of samples carried by the exact flows of three linear fields, shown at $t=0$, $0.6$ and $1.2$ (darker is later); the dotted line is the path of the centre. The density along a trajectory changes by the factor $e^{-t\operatorname{tr}\mathbf A}$: it rises where the flow contracts (left, the spiral), stays constant under the area-preserving saddle although the shape changes (middle), and falls where the flow expands (right).](../img/mdl-dyn-divergence-volume.svg)
:label:`fig_mdl-dyn-divergence-volume`

The general case needs one fact about determinants: over a short step the
flow map is close to the identity, and the determinant of a matrix close to
the identity is governed by its trace.

**Lemma.** *For any square matrix $\mathbf J$,
$\det(\mathbf{I}+h\mathbf J)=1+h\operatorname{tr}\mathbf J+O(h^2)$.*

**Proof.** In the Leibniz expansion of the determinant, any permutation other
than the identity uses at least two off-diagonal entries, each of size
$O(h)$, so it contributes $O(h^2)$. The identity permutation contributes
$\prod_i(1+hJ_{ii})=1+h\sum_iJ_{ii}+O(h^2)$. $\blacksquare$

**Proposition (instantaneous change of variables).** *Let
$\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},t)$ with $\mathbf{f}$ continuously
differentiable in $(\mathbf{x},t)$ and globally Lipschitz in $\mathbf{x}$, so
that each flow map $\Phi_t$ is a $C^1$ bijection with a $C^1$ inverse (the
solution of a $C^1$ field is $C^1$ in its initial value
:cite:`Hairer.Norsett.Wanner.1993`, and the inverse runs the field
backward). Let $\mathbf{x}(0)\sim p_0$ for a positive differentiable density
$p_0$. Then the density $p_t$ of $\mathbf{x}(t)$ exists and is positive, and
along each trajectory* :cite:`Chen.Rubanova.Bettencourt.ea.2018`

$$
\frac{d}{dt} \log p_t(\mathbf{x}(t))
= -\operatorname{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right)\!(\mathbf{x}(t), t).
$$
:eqlabel:`eq_mdl-ode-instant-cov`

**Proof.** The density exists by :eqref:`eq_mdl-ode-discrete-cov` applied to
the bijection $\Phi_t$, and it is positive because $p_0$ is and the Jacobian
of a map with a differentiable inverse is nonsingular. Over a short interval
the flow advances by the map
$\Phi_{t\to t+h}(\mathbf{x})=\mathbf{x}+\int_t^{t+h}\mathbf{f}(\mathbf{x}(s),s)\,ds$.
Differentiating in $\mathbf{x}$ and using the continuity of $\mathbf J$, its Jacobian
is $\mathbf{I}+h\mathbf J+o(h)$, where $\mathbf J=\partial\mathbf{f}/\partial\mathbf{x}$ is
evaluated at $(\mathbf{x}(t),t)$ and $o(h)$ stands for terms that vanish
faster than $h$. Apply :eqref:`eq_mdl-ode-discrete-cov` to this one map and
expand with the Lemma, which tolerates the perturbation: replacing $h\mathbf J$ by
$h\mathbf J+o(h)$ changes the determinant only by $o(h)$. For small $h$ the
determinant is close to $1$, hence positive, so the absolute value can be
dropped:

$$
\log p_{t+h}(\mathbf{x}(t+h))
= \log p_t(\mathbf{x}(t)) - \log\det\!\big(\mathbf{I} + h\mathbf J + o(h)\big)
= \log p_t(\mathbf{x}(t)) - h \operatorname{tr} \mathbf J + o(h),
$$

using $\log(1+u)=u+O(u^2)$. Subtract $\log p_t(\mathbf{x}(t))$, divide by
$h$, and let $h\to0$. $\blacksquare$

The trace of the Jacobian is the *divergence* of the field, the local rate at
which the flow expands volume. Where the field spreads samples apart
($\operatorname{tr}\mathbf J>0$) the log-density along each trajectory falls,
and where it compresses them, it rises. Integrating
:eqref:`eq_mdl-ode-instant-cov` along a trajectory gives

$$
\log p_T(\mathbf{x}(T))
= \log p_0(\mathbf{x}(0)) - \int_0^T \operatorname{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right) dt .
$$
:eqlabel:`eq_mdl-ode-cnf-likelihood`

The per-layer sum of log-determinants of a discrete flow has become an
integral of traces. A model that computes it, by integrating the augmented
state $(\mathbf{x},\log p)$ with any ODE solver, is a **continuous
normalizing flow** (CNF). It needs no restriction on the architecture of
$\mathbf{f}$. To evaluate the density of a given point $\mathbf{x}(T)$,
integrate the field backward from it to $\mathbf{x}(0)$, accumulate the
trace on the way, and read off $p_0$ there. Likelihoods of diffusion models
are computed in this way (:numref:`sec_diffusion-design-space`).

For a linear field the trace integral is $t\operatorname{tr}\mathbf A$, and
:eqref:`eq_mdl-ode-cnf-likelihood` compared with
:eqref:`eq_mdl-ode-discrete-cov` for the map $e^{\mathbf At}$ says
$\log\det e^{\mathbf At}=t\operatorname{tr}\mathbf A$. This identity also
follows in one line from the eigendecomposition $\mathbf A=\mathbf V\boldsymbol\Lambda \mathbf V^{-1}$:

$$
\det e^{\mathbf At}=\det\big(\mathbf Ve^{\boldsymbol\Lambda t}\mathbf V^{-1}\big)=\prod_je^{\Lambda_{jj}t}=e^{t\operatorname{tr}\mathbf A}
$$
:eqlabel:`eq_mdl-ode-jacobi`

for diagonalizable $\mathbf A$, and for every $\mathbf A$ by continuity, since
diagonalizable matrices are dense. For the spiral both sides equal
$e^{-1.5}=0.2231$ at $t=1.5$. The proposition follows individual
trajectories. Stated for the whole density at once instead of along one
trajectory, the same law is the continuity equation of fluid mechanics,
which :numref:`sec_mdl-fokker-planck-probability-flow` derives together with
the term that noise adds.

### Checking the Trace Integral

For the spiral the proposition can be compared with a closed form. Start from $p_0=\mathcal{N}(\mathbf{0},\mathbf{I})$;
a linear map sends it to $p_t=\mathcal{N}(\mathbf{0},e^{\mathbf At}e^{\mathbf A^\top t})$,
which for the spiral is $\mathcal{N}(\mathbf{0},e^{-t}\mathbf{I})$ because
$e^{t\mathbf K}$ is orthogonal. The cell evaluates this Gaussian at the transported
points of four random samples and compares with
$\log p_0(\mathbf{x}_0)-t\operatorname{tr}\mathbf A$.

<!-- WRITER: #odes-solvers-cnf-trace (the Gaussian formula against
log p_0 - t tr A for the spiral at t = 1.5) is cut: a linear field with a
constant trace, it could not fail. The paragraph before this note ("The cell
evaluates this Gaussian ...") and the first sentence after it ("The two rows
agree in every printed digit ...") describe the cut cell; the hand value
e^{-1.5} = 0.2231 stays in the paragraph on eq_mdl-ode-jacobi. Lead straight
into the nonlinear field. -->

The two rows agree in every printed digit, as :eqref:`eq_mdl-ode-jacobi`
says they must; with a constant trace there was little to go wrong. A
nonlinear field is the real test, because its trace
changes along the path. Take the planar field
$\mathbf{f}(\mathbf{x})=\mathbf W_2\tanh(\mathbf W_1\mathbf{x}+\mathbf{b}_1)$ with $8$ hidden
units and random weights. Its Jacobian and trace are explicit:

$$
\mathbf J=\mathbf W_2\,\mathrm{diag}(\mathbf{s})\,\mathbf W_1,
\qquad
\mathbf{s}=1-\tanh^2(\mathbf W_1\mathbf{x}+\mathbf{b}_1),
\qquad
\operatorname{tr}\mathbf J=\sum_is_i\,(\mathbf W_1\mathbf W_2)_{ii},
$$

using the cyclic property
$\operatorname{tr}\big(\mathbf W_2\,\mathrm{diag}(\mathbf{s})\,\mathbf W_1\big)=\operatorname{tr}\big(\mathrm{diag}(\mathbf{s})\,\mathbf W_1\mathbf W_2\big)$.
The cell integrates the augmented
state $(\mathbf{x},\int\operatorname{tr}\mathbf J\,dt)$ with RK4 to $T=1$ and
compares the integral with $\log|\det D\Phi_T|$, computed from central finite
differences of the flow map itself.

<!-- WRITER (code step): the first output line now gives the smallest and the
largest trace along the RK4 path ("tr J along the path: from +0.889 to +1.013"
in PyTorch; -0.172 to +0.067 in TensorFlow, a change of sign; +0.064 to +0.222
in JAX; -0.935 to -0.799 in MXNet), no longer the two ends. "The trace differs at
the two ends of the path" and the slide's "differs at the two ends" should read
the range: the trace varies along the path by a tenth or more in every tab. -->

```{.python .input #mdl-neural-odes-checking-the-trace-integral-1}
n_hidden = 8
W1, b1 = rng.standard_normal((n_hidden, 2)), rng.standard_normal(n_hidden)
W2 = rng.standard_normal((2, n_hidden)) / np.sqrt(n_hidden)
field = lambda x: W2 @ np.tanh(W1 @ x + b1)
trJ = lambda x: np.sum((1 - np.tanh(W1 @ x + b1) ** 2)
                       * np.einsum('ij,ji->i', W1, W2))
aug = lambda z: np.append(field(z[:2]), trJ(z[:2]))  # d/dt (x, int tr J)
xa, N = rng.standard_normal(2), 200
zs = [np.append(xa, 0.0)]                      # RK4 path of (x, int tr J)
for _ in range(N):
    zs.append(rk4_step(aug, zs[-1], 1.0 / N))
zT, tr_path = zs[-1], [trJ(z[:2]) for z in zs]
D = np.zeros((2, 2))                           # D Phi_1 by central differences
for j in range(2):
    e = np.zeros(2); e[j] = 1e-5
    D[:, j] = (solve(rk4_step, field, xa + e, 1.0, N)
               - solve(rk4_step, field, xa - e, 1.0, N)) / 2e-5
logdet = np.log(abs(np.linalg.det(D)))
print(f'tr J along the path: from {min(tr_path):+.3f} to {max(tr_path):+.3f}')
print(f'trace integral {zT[2]:+.6f}  log|det D Phi| {logdet:+.6f}  '
      f'gap {abs(zT[2] - logdet):.1e}')
```

<!-- WRITER: 12 development runs (four tabs): gap 8.7e-12 to 6.1e-11 (the store
of 2026-11-12 has up to 1.6e-9): "to nine digits or better" holds. The cell is
unchanged apart from names (n_hidden for the hidden units, N = 200 RK4 steps
through solve(rk4_step, ...)) and the notebook's rng. -->

The trace generally takes different values at the two ends of the path,
yet the integral and the log-determinant agree to about $10^{-10}$, the
accuracy of the solver and of the finite differences. Nothing in the
proposition used linearity.

### The Hutchinson Trace Estimator

The formula for $\operatorname{tr}\mathbf J$ above used the structure of a
one-layer field. For a general network the trace needs the $d$ diagonal
entries of $\mathbf J$, which cost $d$ reverse-mode passes, one per coordinate.
The trace has already replaced an $O(d^3)$ determinant by $d$ passes. A random
projection replaces the $d$ passes by one, and returns an unbiased but noisy
estimate :cite:`Hutchinson.1989`.

**Proposition (Hutchinson estimator).** *For any matrix
$\mathbf M\in\mathbb{R}^{d\times d}$ and any random vector
$\boldsymbol\epsilon\in\mathbb{R}^d$ with
$\mathbb{E}[\boldsymbol\epsilon\boldsymbol\epsilon^\top]=\mathbf{I}$, such as
a standard Gaussian or a Rademacher vector with independent entries $\pm1$ of
equal probability,*

$$
\operatorname{tr}(\mathbf M) = \mathbb{E}\!\left[\boldsymbol{\epsilon}^\top \mathbf M \boldsymbol{\epsilon}\right].
$$
:eqlabel:`eq_mdl-ode-hutchinson`

**Proof.** $\mathbb{E}[\boldsymbol\epsilon^\top \mathbf M\boldsymbol\epsilon]
=\mathbb{E}[\operatorname{tr}(\mathbf M\boldsymbol\epsilon\boldsymbol\epsilon^\top)]
=\operatorname{tr}(\mathbf M\,\mathbb{E}[\boldsymbol\epsilon\boldsymbol\epsilon^\top])
=\operatorname{tr}(\mathbf M)$, by the cyclic property of the trace and its
linearity, which lets it commute with the expectation. $\blacksquare$

The estimator touches $\mathbf J$ only through $\boldsymbol\epsilon^\top \mathbf J$, one
vector--Jacobian product, which is one reverse-mode pass through $\mathbf{f}$
with $\boldsymbol\epsilon$ as the seed; no Jacobian is formed. For the
one-hidden-layer field the product is explicit,

$$
\boldsymbol\epsilon^\top \mathbf J=\big((\boldsymbol\epsilon^\top \mathbf W_2)\odot\mathbf{s}\big)\,\mathbf W_1,
$$

and it costs about as much as evaluating $\mathbf{f}$. A CNF evaluates one
such product with every evaluation of the field, so the likelihood costs one additional backward
pass per field evaluation (FFJORD, :cite:`Grathwohl.Chen.Bettencourt.ea.2018`).

The proposition says that the average over probes converges to the trace.
It does not say how fast, and that is what we measure. We take a
one-hidden-layer field in $d=50$ with $64$ hidden units, compute its exact
trace from the formula above, and draw $m=10^5$ Rademacher probes.

<!-- WRITER: the experiment is redesigned (NODE-14). The field is now
f(x) = W2 tanh(W1 x + b1) - x in d = 50 with 64 hidden units: the tanh part of
before plus a pull toward the origin, so that its trace is close to -50 in every
draw. m = 1000 Rademacher probes (was 10^5). The first cell (NumPy) draws the
weights, the point x, the exact trace of the tanh part `tr_tanh` and the probes
`E`; it prints nothing. The second (per framework) estimates the trace by
reverse-mode VJPs for the field with the pull and for the tanh part alone, and
prints exact trace, mean of the 1000 probes with its standard error, and the
standard deviation of one probe. A third cell (NumPy, new) integrates the trace
along one Euler trajectory and compares a probe held for the whole solve with a
fresh probe per step. Numbers are in the notes after the cells. -->

```{.python .input #mdl-neural-odes-the-hutchinson-trace-estimator-1}
d, n_hidden, m = 50, 64, 1000
W1 = rng.standard_normal((n_hidden, d)) / np.sqrt(d)
b1 = 0.1 * rng.standard_normal(n_hidden)
W2 = rng.standard_normal((d, n_hidden)) / np.sqrt(n_hidden)
x = rng.standard_normal(d)
s = 1 - np.tanh(W1 @ x + b1) ** 2
tr_tanh = np.sum(s * np.einsum('ij,ji->i', W1, W2))  # trace of W2 diag(s) W1
E = rng.choice([-1.0, 1.0], size=(m, d))         # Rademacher probes, one per row
E_gauss = rng.standard_normal((m, d))            # Gaussian probes, one per row
```

```{.python .input #mdl-neural-odes-the-hutchinson-trace-estimator-2}
#@tab pytorch
W1t, b1t, W2t = (torch.tensor(w) for w in (W1, b1, W2))
field = lambda x: W2t @ torch.tanh(W1t @ x + b1t) - x
_, vjp = torch.func.vjp(field, torch.tensor(x))      # reverse mode at x
for name, P in (('Rademacher', E), ('Gaussian', E_gauss)):
    P = torch.tensor(P)
    rows = torch.func.vmap(vjp)(P)[0]                # row k: eps_k^T J
    est = (rows * P).sum(dim=1)                      # eps_k^T J eps_k
    sd = est.std(correction=0)
    print(f'{name:10s}: exact {tr_tanh - d:+6.2f}, mean of {m} probes '
          f'{est.mean():+6.2f} +- {sd / m ** 0.5:.2f}, one probe sd {sd:.2f}')
```

```{.python .input #mdl-neural-odes-the-hutchinson-trace-estimator-2}
#@tab mxnet
xs = mnp.array(np.tile(x, (m, 1)))             # m copies of x, one per probe
W1m, b1m, W2m = (mnp.array(w) for w in (W1, b1, W2))
field = lambda xs: mnp.tanh(xs @ W1m.T + b1m) @ W2m.T - xs
for name, P in (('Rademacher', E), ('Gaussian', E_gauss)):
    P = mnp.array(P)
    xs.attach_grad()
    with autograd.record():
        ys = field(xs)
    ys.backward(P)                             # seed the backward pass with the probes
    est = (xs.grad * P).sum(axis=1)            # row k of xs.grad: eps_k^T J
    mean, sd = float(est.mean()), float(est.std())
    print(f'{name:10s}: exact {tr_tanh - d:+6.2f}, mean of {m} probes '
          f'{mean:+6.2f} +- {sd / m ** 0.5:.2f}, one probe sd {sd:.2f}')
```

```{.python .input #mdl-neural-odes-the-hutchinson-trace-estimator-2}
#@tab tensorflow
xs = tf.constant(np.tile(x, (m, 1)))           # m copies of x, one per probe
field = lambda xs: tf.tanh(xs @ W1.T + b1) @ W2.T - xs
for name, P in (('Rademacher', E), ('Gaussian', E_gauss)):
    with tf.GradientTape() as tape:
        tape.watch(xs)
        ys = field(xs)
    rows = tape.gradient(ys, xs, output_gradients=tf.constant(P))  # eps_k^T J
    est = tf.reduce_sum(rows * P, axis=1)      # eps_k^T J eps_k
    mean, sd = float(tf.reduce_mean(est)), float(tf.math.reduce_std(est))
    print(f'{name:10s}: exact {tr_tanh - d:+6.2f}, mean of {m} probes '
          f'{mean:+6.2f} +- {sd / m ** 0.5:.2f}, one probe sd {sd:.2f}')
```

```{.python .input #mdl-neural-odes-the-hutchinson-trace-estimator-2}
#@tab jax
W1j, b1j, W2j = (jnp.array(w) for w in (W1, b1, W2))
field = lambda x: W2j @ jnp.tanh(W1j @ x + b1j) - x
_, vjp = jax.vjp(field, jnp.array(x))          # reverse mode at x
for name, P in (('Rademacher', E), ('Gaussian', E_gauss)):
    rows = jax.vmap(vjp)(jnp.array(P))[0]      # row k: eps_k^T J
    est = (rows * P).sum(axis=1)               # eps_k^T J eps_k
    mean, sd = float(est.mean()), float(est.std())
    print(f'{name:10s}: exact {tr_tanh - d:+6.2f}, mean of {m} probes '
          f'{mean:+6.2f} +- {sd / m ** 0.5:.2f}, one probe sd {sd:.2f}')
```

<!-- WRITER: the cell above (changed after lib-review finding 2) now compares
two probe distributions on the field with the pull, not the field with and
without the pull: for Rademacher probes eps^T(-I)eps = -d exactly, so that
second row could not differ from the first. Rows: Rademacher and Gaussian, both
m = 1000 on the same field; exact trace -49 to -51.5 (the tanh part alone is of
order one, either sign). Rademacher one-probe sd about 4 to 5.5 (the
off-diagonal entries only); Gaussian about 11 (adds 2 sum_i J_ii^2, about
2d = 100 to the variance, because the diagonal is near -1). The prose point
"the off-diagonal entries set the noise; Rademacher probes see the diagonal
exactly" is now measured. Values per tab: claims.md. -->

The trace integral of a CNF adds one such estimate per solver step. Whether the
probe is held for the whole solve or drawn afresh at every step changes the
noise of the integral, and the next cell measures both along one Euler
trajectory of the field with the pull, from the same $\mathbf x$, with $N=10$,
$100$ and $1000$ steps and $200$ repetitions, using the explicit product
$\boldsymbol\epsilon^\top\mathbf J\boldsymbol\epsilon
=\big((\boldsymbol\epsilon^\top\mathbf W_2)\odot\mathbf s\big)\,\mathbf W_1\boldsymbol\epsilon
-\boldsymbol\epsilon^\top\boldsymbol\epsilon$.
<!-- WRITER: the paragraph just above is a placeholder lead-in written by the
coder for the new cell; rewrite freely. -->

```{.python .input #mdl-neural-odes-the-hutchinson-trace-estimator-3}
field = lambda x: W2 @ np.tanh(W1 @ x + b1) - x
diag_W1W2 = np.einsum('ij,ji->i', W1, W2)
for N in (10, 100, 1000):
    h, xs = 1 / N, [x]
    for _ in range(N - 1):                     # Euler trajectory x_0, ..., x_{N-1}
        xs.append(xs[-1] + h * field(xs[-1]))
    S = 1 - np.tanh(np.array(xs) @ W1.T + b1) ** 2  # s at every step, (N, n_hidden)
    exact = h * (S @ diag_W1W2).sum() - d      # h * sum_n tr J(x_n)
    held = np.broadcast_to(rng.choice([-1.0, 1.0], (200, 1, d)), (200, N, d))
    fresh = rng.choice([-1.0, 1.0], (200, N, d))
    row = f'N = {N:4d}: exact {exact:+.2f}'
    for name, P in (('held probe', held), ('fresh probes', fresh)):
        est = h * np.einsum('rnh,nh,rnh->r', P @ W2, S, P @ W1.T) - d  # eps^T eps = d
        row += f' | {name} {est.mean():+.2f}, sd {est.std():.2f}'
    print(row)
```

<!-- WRITER: the cell above, 15 development runs (four tabs): exact integral
-48.7 to -51.5 (close to the trace at x, the same for the three N to 0.03);
held probe: mean within 1 of the exact value (200 repetitions, standard error
about 0.4), standard deviation 4.6 to 6.0 at every N, not falling with N; fresh
probes: mean within 0.3, 0.07, 0.02 of it, standard deviation 1.5 to 1.9 at
N = 10, 0.50 to 0.67 at N = 100, 0.16 to 0.19 at N = 1000, i.e. the one-probe
noise over sqrt(N). Unbiased either way. The paragraphs below
("The average lies within ..." with "a few tenths in size", "Our random field
is a hard case ...", "Subtracting x ... would move the trace to about -50") are
now false or done by the cells: rewrite them as the reading of the two tables. -->

The average lies within a few standard errors of the exact trace. But read
the output again: $10^5$ probes locate the trace only to about $0.015$, when
$50$ passes would have given it exactly. The standard error implies a
standard deviation of about $4$ to $5$ for one probe, many times the trace,
which is typically a few tenths in size. For Rademacher probes one probe has
variance $2\sum_{i\ne j}S_{ij}^2$ with
$\mathbf S=(\mathbf J+\mathbf J^\top)/2$: the off-diagonal entries set the
noise, and the trace sees only the diagonal. Our random field is a hard
case, because its diagonal entries have random signs and nearly cancel.
Subtracting $\mathbf{x}$ from the field, a pull toward the origin, would
move the trace to about $-50$ and leave the noise unchanged, a relative
error of a tenth per probe.

Why, then, does a CNF train with one probe per step? The estimate is
unbiased and enters the log-likelihood linearly, so the estimated
log-likelihood and its gradient are unbiased as well. A stochastic gradient
asks for nothing more (:numref:`sec_mdl-stochastic-gradient-methods`): the
noise of the probes is averaged over minibatches and iterations, as the
noise of the minibatch itself is.
What one probe cannot give is the likelihood of a single point. For that,
average many probes or compute the exact trace.

The dynamics of this section were deterministic throughout. A diffusion
model adds fresh noise at every step, and increments that scale like
$\sqrt{h}$ break the Taylor expansions this section relied on.
:numref:`sec_mdl-sdes` asks how to integrate such an equation.

## Summary

A residual block is an Euler step, and one field with shrinking steps
gives a Neural ODE. Its exact flow is invertible and, on $\mathbb{R}$,
order-preserving, so the map $x\mapsto-x$ needs a second dimension. A field
trained through ten Euler steps fits the ten-step program, not the flow. The
gradient of a loss on $\mathbf{x}(T)$ is the integral of
$(\partial\mathbf{f}/\partial\boldsymbol\theta)^\top\mathbf{a}$ along the
trajectory, where the adjoint $\mathbf{a}$ solves a linear ODE backward from
$\nabla\ell(\mathbf{x}(T))$. Backpropagation through Euler is exact for the
program and within $O(h)$ of the gradient of the flow. Re-integrating the
state backward saves memory but amplifies a stored error by $e^{\lambda T}$
for a mode that decays at rate $\lambda$. Along a trajectory
$\tfrac{d}{dt}\log p_t=-\operatorname{tr}\mathbf J$. Hutchinson's estimator
obtains the trace from one vector--Jacobian product per probe, without bias
but with noise that training averages out and a single likelihood does not.

## Exercises

1. [code] **The learned field at other step counts.** The
   `#odes-solvers-neural-ode-train` cell trains the field with $N=10$ Euler
   steps of size $h=0.1$, and the text found that the endpoint error rises
   from a few thousandths to about $0.02$ when the same field is integrated
   with $N=100$ or $N=1000$ steps.
    1. Evaluate the trained `flow` with $N=1$, $2$ and $5$ steps ($Nh=1$).
       Why is the error at $N=1$ large although ten steps of the same field
       are accurate to a few thousandths? Which map does one Euler step of
       the trained field compute?
    1. What does the error at $N=1000$ measure? Name the two maps whose
       distance it is, and explain why it does not shrink as $N$ grows
       further.
    1. Integrate the trained field backward from the targets `Y` with
       $N=10$, $100$ and $1000$ steps of
       $\mathbf{x}\leftarrow\mathbf{x}-h\,\mathbf{f}_{\boldsymbol\theta}(\mathbf{x})$
       and compare the result with `X`. The forward--backward round trip of
       the text went to zero like $1/N$; this error settles at a few
       hundredths. Why? *Hint: at large $N$ the backward integration applies the exact
       inverse flow to the ellipse, not to the ten-step image of the
       circle.*
    1. Retrain with $N=1$ and with $N=2$, and repeat (a). Which of the three
       trained fields comes closest to its own time-one flow?
   :exlabel:`ex_mdl-odes-solvers-step-count-transfer`
<!-- WRITER (exercise): the trained fields are now `trained['Euler']` and
`trained['RK4']` (PyTorch, TensorFlow, MXNet: integrate with
`solve(step, trained[k], X, 1.0, N)`; JAX: `flow(trained[k], step, X, 1.0, N)`);
there is no `flow(X, N, h)` any more, and the backward integration of part (c)
is `solve(euler_step, trained['Euler'], Y, -1.0, N)`. The training cell takes
the solver step and the seed as arguments (`train(name, step, seed)`; JAX
`train(name, step)` from the shared `params0`), so part (d) and the
worklist's new parts (Heun, training at N = 100) need only a new step function
or a changed step count. -->

1. [code] **What a flow cannot represent.** ● Two trajectories of a field
   with Lipschitz constant $L$ satisfy
   $e^{-Lt}\|\mathbf{x}-\mathbf{y}\|\le\|\Phi_t(\mathbf{x})-\Phi_t(\mathbf{y})\|\le e^{Lt}\|\mathbf{x}-\mathbf{y}\|$
   (:exref:`ex_mdl-odes-solvers-trajectory-separation`), and
   :numref:`subsec_mdl-flow-order` showed that no flow on $\mathbb{R}$
   reverses order :cite:`Dupont.Doucet.Teh.2019`.
    1. Let $d=1$. Use the lower bound to give a second proof that $\Phi_T$
       is strictly increasing, and show that
       $\Phi_T(1)-\Phi_T(-1)\ge2e^{-LT}$. Which of the two statements, order
       or distance, does the proposition of the text give, and which needs
       the bound?
    1. The training example carries the circle points $(0,\pm1)$, which are
       $2$ apart, to $(1.5,\pm0.4)$, which are $0.8$ apart. Show that the
       exact time-$T$ flow of a field with Lipschitz constant $L$ can do this
       only if $LT\ge\ln2.5\approx0.92$, whereas the long axis, stretched from
       $2$ to $2.4$, gives only $LT\ge\ln1.2$. A one-hidden-layer tanh field
       has $L\le\|\mathbf W_2\|_2\|\mathbf W_1\|_2$ because $\tanh$ is $1$-Lipschitz.
       Compute this bound for the trained field and check the inequality.
    1. Embed $\mathbb{R}$ in $\mathbb{R}^2$ as $x\mapsto(x,0)$. Show that
       every linear field $\dot{\mathbf{x}}=\mathbf B\mathbf{x}$ whose time-one map
       sends $(x,0)$ to $(-x,0)$ for every $x$ has $e^{\mathbf B}=-\mathbf{I}$.
       *Hint: an eigenvalue $-1$ of $e^{\mathbf B}$ comes from an eigenvalue $\mu$ of
       $\mathbf B$ with $e^{\mu}=-1$, which is not real.* Find such a field whose
       trajectories are ellipses rather than circles.
    1. Change the training cell to fit the map $x\mapsto-x$ on $64$ evenly
       spaced points of $[-1,1]$ with a field of input and output dimension
       $1$ and $N=10$ steps. What does the trained model output, and why does
       the loss settle at a value you can predict before training? Repeat
       with a field of dimension $2$, inputs $(x,0)$, and the loss on the
       first coordinate.
    1. Train the one-dimensional model once more with $N=1$ and $h=1$, and
       evaluate it with $N=1$ and with $N=100$ steps of size $h=0.01$.
       Relate the two results to the proposition and to the condition
       $hL<1$.
   :exlabel:`ex_mdl-odes-solvers-flow-expressivity`

1. **Adjoint with a running loss.** ● The adjoint equations
   :eqref:`eq_mdl-ode-adjoint` and :eqref:`eq_mdl-ode-adjoint-grad` treat a
   loss on the endpoint. Regularizers such as the kinetic energy
   $\int_0^T\|\mathbf{f}(\mathbf{x}(t),\boldsymbol\theta,t)\|^2\,dt$ of
   :citet:`Finlay.Jacobsen.Nurbekyan.ea.2020` accumulate a cost along the
   trajectory. Let

   $$
   \mathcal{L} = \ell_T(\mathbf{x}(T)) + \int_0^T \ell(\mathbf{x}(t), \boldsymbol\theta, t)\,dt .
   $$

    1. Extend the variational proof of :eqref:`eq_mdl-ode-adjoint` to show
       that the adjoint acquires a source term,
       $\dot{\mathbf{a}} = -(\partial\mathbf{f}/\partial\mathbf{x})^\top\mathbf{a} - \nabla_{\mathbf{x}}\ell$
       with $\mathbf{a}(T) = \nabla\ell_T(\mathbf{x}(T))$, and that
       $\partial\mathcal{L}/\partial\boldsymbol\theta = \int_0^T \big((\partial\mathbf{f}/\partial\boldsymbol\theta)^\top\mathbf{a} + \nabla_{\boldsymbol\theta}\ell\big)\,dt$.
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
   adjoint can avoid storing the forward states by integrating
   $\mathbf{x}(t)$ backward from $\mathbf{x}(T)$ alongside $\mathbf{a}(t)$,
   and :numref:`sec_mdl-adjoint-method` quoted the resulting drift
   $e^{\lambda T}\delta$.
    1. For $\dot{x}=-\lambda x$ with $\lambda>0$, an error $\delta$ in the
       stored $x(T)$ causes an error in the reconstructed $x(0)$. Derive it.
    1. Let $\mathbf R$ be the rotation by $0.6$ radians and
       $\mathbf B=\mathbf R\,\mathrm{diag}(-50,-1)\,\mathbf R^\top$. Integrate
       $\dot{\mathbf{x}}=\mathbf B\mathbf{x}$ from $(1,1)$ to time $T$ with `rk4` and
       $2000$ steps, then integrate the field $-\mathbf B\mathbf{x}$ from the result
       for the same time. Report the reconstruction error for
       $T=0.25$, $0.5$, $1$ and $2$, and compare its growth with (a), taking
       $\delta$ to be the double-precision rounding error $\approx10^{-16}$.
       Repeat with $\mathrm{diag}(-50,-1)$ itself. Why does the diagonal
       matrix reconstruct far more accurately?
    1. Activation checkpointing (:numref:`subsec_activation_checkpointing`)
       stores the state at the start of each segment of the trajectory. A
       hybrid scheme reconstructs the states within a segment by integrating
       backward from the checkpoint at its end. For the matrix $\mathbf B$ of (b),
       what is the longest segment duration for which the error of (a)
       stays below $10^{-8}$?
   :exlabel:`ex_mdl-odes-solvers-backward-reconstruction`
<!-- WRITER (exercise): part (b) says "with `rk4` and 2000 steps"; the
notebook now has `solve(rk4_step, f, x, T, 2000)` instead of `rk4`. -->

1. **Log-determinant of a residual step.** ● The instantaneous change of
   variables :eqref:`eq_mdl-ode-instant-cov` is a limit of infinitesimal
   steps. A finite residual step
   $F(\mathbf{x})=\mathbf{x}+\mathbf{g}(\mathbf{x})$ needs its exact
   log-determinant, as in the invertible residual networks of
   :citet:`Behrmann.Grathwohl.Chen.ea.2019`.
    1. Let $\mathbf{g}$ be Lipschitz with constant $c<1$. For $d=1$ the
       one-line argument of :numref:`subsec_mdl-flow-order` (an Euler step
       is increasing when $hL<1$) shows that $F$ is strictly increasing,
       hence injective. For general $d$, write $F(\mathbf{x})=\mathbf{y}$ as
       a fixed-point equation for $\mathbf{x}$ and use the Banach
       fixed-point theorem, as in the proof of Picard--Lindelöf, to show
       that $F$ is a bijection of $\mathbb{R}^d$. How fast does the
       fixed-point iteration converge?
    1. Show that $\|\mathbf J_{\mathbf{g}}\|\le c$, where $\mathbf J_{\mathbf{g}}$ is the
       Jacobian of $\mathbf{g}$, and use the eigenvalues of $\mathbf J_{\mathbf{g}}$
       to show that

       $$
       \log\det(\mathbf{I} + \mathbf J_{\mathbf{g}})
       = \sum_{k=1}^\infty \frac{(-1)^{k+1}}{k}\operatorname{tr}\big(\mathbf J_{\mathbf{g}}^k\big).
       $$

    1. An Euler step has $\mathbf{g}=h\mathbf{f}$ and $\mathbf J_{\mathbf{g}}=h\mathbf J$.
       Compare $\log\det(\mathbf{I}+h\mathbf J)$ with the term $h\operatorname{tr}\mathbf J$
       that :eqref:`eq_mdl-ode-cnf-likelihood` accumulates per step. How
       large is the difference per step, and how large is it in total over
       $T/h$ steps?
    1. The Hutchinson estimator :eqref:`eq_mdl-ode-hutchinson` estimates
       $\operatorname{tr}(\mathbf J_{\mathbf{g}}^k)$ by
       $\boldsymbol{\epsilon}^\top \mathbf J_{\mathbf{g}}^k\boldsymbol{\epsilon}$.
       How many vector--Jacobian products per probe does the series
       truncated after $n$ terms cost?
   :exlabel:`ex_mdl-odes-solvers-residual-logdet`

1. [code] **Variance of the Hutchinson estimator.** ● The estimator
   :eqref:`eq_mdl-ode-hutchinson` estimates $\operatorname{tr}\mathbf M$ by
   $\boldsymbol{\epsilon}^\top \mathbf M\boldsymbol{\epsilon}$ for a single probe
   $\boldsymbol{\epsilon}$. Let $\mathbf S=(\mathbf M+\mathbf M^\top)/2$, so that
   $\boldsymbol{\epsilon}^\top \mathbf M\boldsymbol{\epsilon}=\boldsymbol{\epsilon}^\top \mathbf S\boldsymbol{\epsilon}$.
    1. Show that the variance of
       $\boldsymbol{\epsilon}^\top \mathbf M\boldsymbol{\epsilon}$ is $2\|\mathbf S\|_F^2$
       for Gaussian probes and $2\sum_{i\neq j}S_{ij}^2$ for Rademacher
       probes. For which matrices is a single Rademacher probe exact?
    1. Form the Jacobian $\mathbf J=\mathbf W_2\,\mathrm{diag}(\mathbf{s})\,\mathbf W_1$ of the field
       in the `#mdl-neural-odes-the-hutchinson-trace-estimator-1` cell
       explicitly (it is $50\times50$), evaluate both variances, and compare
       the Rademacher value with $m$ times the square of the standard error
       that the next cell printed. How many probes does each distribution
       need for a standard error of $0.01$?
    1. In a CNF, $\mathbf M$ is the Jacobian $\partial\mathbf{f}/\partial\mathbf{x}$
       in dimension $d$. Suppose that all entries have comparable magnitude.
       How does the relative standard error
       $\sqrt{\textrm{Var}}/|\operatorname{tr}\mathbf M|$ of one probe scale
       with $d$ when the diagonal entries share one sign, and when their
       signs are random?
   :exlabel:`ex_mdl-odes-solvers-hutchinson-variance`
<!-- WRITER (exercise): part (b) still works with the redesigned cells (the
`-1` cell defines W1, b1, W2, x, s and E for d = 50; the Jacobian of the field
with the pull is W2 diag(s) W1 - I). The next cell now prints the one-probe
standard deviation directly, for m = 1000, for Rademacher and for Gaussian
probes on the field with the pull. -->

1. [code] **Discrete and continuous adjoints.** ●● Backpropagation through
   a solver differentiates the executed program, whereas a solver applied to
   the adjoint ODE :eqref:`eq_mdl-ode-adjoint` approximates the gradient of
   the exact flow :cite:`Kidger.2022`. Compare the two recursions step by
   step for an autonomous field with Jacobian
   $\mathbf J(\mathbf{x})=\partial\mathbf{f}/\partial\mathbf{x}$ and forward states
   $\mathbf{x}_k$.
    1. For forward Euler, the discrete adjoint
       :eqref:`eq_mdl-ode-discrete-adjoint` computes
       $\mathbf{a}_k=(\mathbf{I}+h\mathbf J(\mathbf{x}_k))^\top\mathbf{a}_{k+1}$,
       whereas an explicit Euler step of the adjoint ODE from $t_{k+1}$ back
       to $t_k$ computes
       $\mathbf{a}_k=\mathbf{a}_{k+1}+h\mathbf J(\mathbf{x}_{k+1})^\top\mathbf{a}_{k+1}$.
       Bound the difference of the two updates for a smooth $\mathbf{f}$.
       For a linear field the two coincide; what then produces the gap that
       the `#mdl-neural-odes-backpropagation-as-the-discrete-adjoint-1` cell
       prints? For the scalar program $x_N=(1+h\theta)^Nx_0$ the text found
       the gradient $T\,x_N^2/(1+h\theta)$; show that this is what the
       recursion computes.
    1. Differentiate one Heun step
       $\mathbf{x}_{k+1}=\mathbf{x}_k+\tfrac{h}{2}\big[\mathbf{f}(\mathbf{x}_k)+\mathbf{f}(\mathbf{u}_k)\big]$
       with $\mathbf{u}_k=\mathbf{x}_k+h\mathbf{f}(\mathbf{x}_k)$ to obtain
       its discrete adjoint recursion. Show that this recursion is a Heun
       step of the adjoint ODE from $t_{k+1}$ to $t_k$ in which
       $\mathbf J(\mathbf{x}_{k+1})$ is replaced by $\mathbf J(\mathbf{u}_k)$. At which order
       in $h$ do the two updates differ?
    1. For a linear field $\mathbf{f}(\mathbf{x})=\mathbf A\mathbf{x}$, one RK4 step
       multiplies the state by $R(h\mathbf A)$, where
       $R(z)=1+z+z^2/2+z^3/6+z^4/24$ is the RK4 stability polynomial
       (:exref:`ex_mdl-odes-solvers-rotating-modes`). Show that the discrete
       adjoint of RK4 is an RK4 step of the adjoint ODE.
    1. Replace the Euler step of `loss_and_grad_unrolled` by an RK4 step,
       and compute the gradient of the unrolled loss by central finite
       differences instead of a hand-written backward sweep. Measure its gap
       to the continuous-adjoint gradient `g_cont` for $n=5$, $10$, $20$,
       $40$ and $80$ steps. The trapezoid rule that computes `g_cont` uses
       `n_quad = 2000` intervals and is accurate only to about $10^{-7}$, so
       refine it first. Which slope do you measure? Which order does the
       gradient computed through a solver of order $p$ inherit?
   :exlabel:`ex_mdl-odes-solvers-re-derive-adjoint-equations`

1. **Checkpointing arithmetic.** ● A forward pass of $N$ Euler steps is
   differentiated by the discrete adjoint
   :eqref:`eq_mdl-ode-discrete-adjoint`, whose backward sweep needs
   $\mathbf{x}_k$ at step $k$.
    1. Storing every state holds $N$ states and recomputes nothing. Suppose
       instead that only every $k$-th state is stored as a checkpoint, and
       that when the backward sweep enters a segment of $k$ steps, the
       segment is recomputed forward from its checkpoint and its states are
       held until the sweep leaves it. How many states are held at the peak,
       and how many extra forward steps are computed in total?
    1. Minimize the peak over $k$. Show that the minimum is near
       $k=\sqrt N$, and that the extra work is about one forward pass
       whatever $k$ is. For $N=1000$, compare the peak with the $1000$
       states of full storage.
    1. Recursive checkpointing applies the same idea inside each segment and
       holds $O(\log N)$ states at the cost of $O(\log N)$ forward passes
       :cite:`Griewank.1992`. Compare it with the continuous adjoint, whose
       memory is constant but whose backward re-integration amplifies an
       error by $e^{\lambda T}$ on $\dot{x}=-\lambda x$. For which
       $\lambda T$ does re-integration from a state stored in single
       precision ($\delta\approx6\times10^{-8}$) lose all accuracy?
   :exlabel:`ex_mdl-neural-odes-checkpoint-arithmetic`

1. [code] **Liouville's formula.** ●● The instantaneous change of variables
   followed a density along a trajectory. This exercise proves it a second
   way, through the Jacobian $\mathbf M(t)=D\Phi_t(\mathbf{x}_0)$ of the flow map
   itself.
    1. Differentiate
       $\tfrac{d}{dt}\Phi_t(\mathbf{x}_0)=\mathbf{f}(\Phi_t(\mathbf{x}_0),t)$
       with respect to $\mathbf{x}_0$ to obtain the variational equation
       $\dot{\mathbf M}=\mathbf J(\mathbf{x}(t),t)\,\mathbf M$ with $\mathbf M(0)=\mathbf{I}$.
    1. Prove Jacobi's formula
       $\tfrac{d}{dt}\det \mathbf M=\operatorname{tr}(\mathbf M^{-1}\dot{\mathbf M})\det \mathbf M$ for
       invertible $\mathbf M(t)$. *Hint: write
       $\det \mathbf M(t+h)=\det \mathbf M(t)\,\det\big(\mathbf{I}+h\mathbf M^{-1}\dot{\mathbf M}+O(h^2)\big)$
       and use the Lemma.*
    1. Conclude that
       $\log|\det D\Phi_T(\mathbf{x}_0)|=\int_0^T\operatorname{tr}\mathbf J(\mathbf{x}(t),t)\,dt$,
       and derive :eqref:`eq_mdl-ode-cnf-likelihood` from the discrete
       change of variables :eqref:`eq_mdl-ode-discrete-cov` applied to the
       bijection $\Phi_T$. For a linear field the result is the
       formula :eqref:`eq_mdl-ode-jacobi`.
    1. Integrate $\dot{\mathbf M}=\mathbf J\mathbf M$ with RK4 alongside the $8$-unit tanh field of
       the `#mdl-neural-odes-checking-the-trace-integral-1` cell (re-create
       its weights, since the Hutchinson cells reuse the names) and compare
       $\log|\det \mathbf M(1)|$ with the two numbers that cell prints.
   :exlabel:`ex_mdl-neural-odes-liouville`
<!-- WRITER (exercise): part (d) still holds (the trace cell keeps its weights
under W1, b1, W2, and the Hutchinson cells reuse the names); it integrates
with `solve(rk4_step, ...)` and N = 200 steps. -->

[Discussions](https://d2l.discourse.group/t/36868)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §29.2]{.kicker}

What do we gain by taking the ODE itself as the model?<br>**Neural ODEs · adjoints · continuous normalizing flows**
:::
:::

::: {.slide title="A Density on 32×32 Colour Images Needs a 3072×3072 Determinant"}
[The problem]{.kicker}

An invertible network $\mathbf g$ (a normalizing flow) has the density
$$\log p_{\textrm{out}}(\mathbf g(\mathbf x))=\log p_{\textrm{in}}(\mathbf x)-\log\left|\det \mathbf J_{\mathbf g}(\mathbf x)\right|.$$

- $d=3\cdot32\cdot32=3072$: one determinant costs $\approx\tfrac23d^3\approx2\times10^{10}$ operations.
- That is $\approx3000\times$ one layer of width $1024$ ($2\cdot3072\cdot1024$).
- A residual block $\mathbf x+\mathbf f(\mathbf x)$ is an Euler step. In continuous time the
  determinant becomes a **trace**, estimated with one backward pass.
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write a ResNet as an Euler program and train it as a Neural ODE;
- derive the adjoint ODE and see backpropagation as a discretization of it;
- decide between storing states and re-integrating them, from $e^{\lambda T}$;
- compute $\tfrac{d}{dt}\log p=-\operatorname{tr}\mathbf J$ and estimate a trace with one VJP.

**Running examples:** the spiral $\mathbf A=-\tfrac12\mathbf I+\mathbf K$,
$\mathbf K=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$, eigenvalues
$-0.5\pm\mathrm i$; a $32$-unit tanh field that maps the unit circle to an ellipse.
:::

::: {.slide title="From Exact Motion to Learned Motion"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| ODEs and Solvers | exact motion and its numerical steps |
| **Neural ODEs** | **a learned field, its gradient, its density** |
| Stochastic Differential Equations | noise, Itô calculus, the OU process |
| Fokker–Planck and Probability Flow | how the density moves; the score |
| Score Matching and Diffusion | learning the score, sampling with it |
| Flow Matching and Transport | learning a velocity; straight paths |
:::

::: {.col .narrow}
- **Uses:** Euler, RK4, $e^{\mathbf At}$, Picard–Lindelöf, reverse-mode AD.
- **Skips:** noise, adaptive solvers, latent ODEs.
:::
:::
:::

::: {.slide title="A Residual Block Is One Euler Step with $h=1$"}
[From ResNets to ODEs]{.kicker}

::: {.cols .vc}
::: {.col}
- $\mathbf x_{l+1}=\mathbf x_l+\mathbf f_{\boldsymbol\theta_l}(\mathbf x_l)$ is $\mathbf x_{n+1}=\mathbf x_n+h\,\mathbf f(\mathbf x_n,t_n)$ at $h=1$.
- Exact and numerical maps differ in kind: $\dot x=-x$ has the bijection
  $e^{-t}x$; one Euler step with $h=1$ sends every $x$ to $0$.
:::

::: {.col .fig .big}
@fig:mdl-dyn-resnet-as-euler
:::
:::
:::

::: {.slide title="Sharing One Field and Shrinking the Step Gives a Neural ODE"}
[Definition]{.kicker}

$N$ steps of size $h=T/N$ with one field $\mathbf f_{\boldsymbol\theta}$:
$$\mathbf x_{n+1}=\mathbf x_n+h\,\mathbf f_{\boldsymbol\theta}(\mathbf x_n,t_n)
\;\xrightarrow{N\to\infty}\;
\mathbf x(T)=\mathbf x_0+\int_0^T\mathbf f_{\boldsymbol\theta}(\mathbf x(t),t)\,dt.$$

- Parameters define the field; the solver is chosen afterwards.
- Gains: the exact map is **invertible** (run the field backward), its
  **density** is computable, and the **step count** is a dial set after training.
:::

::: {.slide title="No Flow on the Line Can Reverse Order"}
[A limit]{.kicker}

::: {.cols .vc}
::: {.col}
- If $x<y$ but $\Phi_T(x)\ge\Phi_T(y)$, the paths meet at some $s$; uniqueness backward from $s$ gives $x=y$.
- Euler step: $(x+hf(x))-(y+hf(y))\ge(1-hL)(x-y)$; $f=-2x$ has $hL=2$.
- In 2-D, $e^{\pi \mathbf K}=-\mathbf I$ sends $(x,0)$ to $(-x,0)$ along half circles.
:::

::: {.col .fig .big}
@fig:mdl-dyn-flow-order
:::
:::
:::

::: {.slide title="Ten Shared-Weight Euler Steps Learn the Circle-to-Ellipse Map"}
[Training through the solver]{.kicker}

Field: one hidden layer of $32$ tanh units. Target: $64$ circle points to the
ellipse centred at $(1.5,0)$, semi-axes $1.2$ and $0.4$. $N=10$, $h=0.1$,
Adam at $0.05$ then $0.005$, backpropagation through the unrolled steps.

@!odes-solvers-neural-ode-train

::: {.d2l-note}
The program is a ResNet of ten blocks that share one set of weights.
:::
:::

::: {.slide title="The Field Defines a Flow Close to, Not Equal to, Its Ten-Step Program"}
[Flow or program?]{.kicker}

Same field, $Nh=1$: endpoint error, and forward $N$ steps then backward $N$
steps with $\mathbf x\leftarrow\mathbf x-h\,\mathbf f_{\boldsymbol\theta}(\mathbf x)$.

@!mdl-neural-odes-training-through-the-solver-1

::: {.d2l-note}
Training fitted the ten-step program; the exact flow lands $0.02$ away. The
round trip falls tenfold per tenfold $N$: Euler's $O(h)$.
:::
:::

::: {.slide title="Check: Round Trips and Invertibility"}
[Check]{.kicker}

At $N=10{,}000$ steps, how far does the forward–backward round trip land from
its start? Which map is invertible: the exact flow or the ten-step Euler map?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** About $6\times10^{-5}$, another factor of ten. The exact flow is
invertible for a Lipschitz field; the Euler map only in the limit $h\to0$.
:::
:::
:::

::: {.slide title="The Scalar Adjoint: $x$ Grows, $a$ Shrinks, $a\,x$ Is Flat"}
[The adjoint method]{.kicker}

::: {.cols .vc}
::: {.col}
$\dot x=\theta x$, $\mathcal L=\tfrac12x(T)^2$. I: directly $\partial\mathcal L/\partial\theta=Tx(T)^2$.
II: $a(t)=\partial\mathcal L/\partial x(t)=e^{\theta(T-t)}x(T)$, run backward.
III: $\int_0^Ta\,x\,dt=Tx(T)^2$; at $\theta=T=x_0=1$, $\boxed{e^2=7.39}$.
:::

::: {.col .fig .big}
@fig:mdl-dyn-adjoint-timeline
:::
:::
:::

::: {.slide title="The Adjoint ODE Carries the Loss Sensitivity Backward"}
[The adjoint equations]{.kicker}

For $\dot{\mathbf x}=\mathbf f(\mathbf x,\boldsymbol\theta,t)$ and $\mathcal L=\ell(\mathbf x(T))$,
with $\mathbf a(t)=\partial\mathcal L/\partial\mathbf x(t)$:
$$\dot{\mathbf a}=-\Big(\frac{\partial\mathbf f}{\partial\mathbf x}\Big)^{\!\top}\mathbf a,\quad
\mathbf a(T)=\nabla\ell(\mathbf x(T)),\qquad
\frac{\partial\mathcal L}{\partial\boldsymbol\theta}=\int_0^T\Big(\frac{\partial\mathbf f}{\partial\boldsymbol\theta}\Big)^{\!\top}\mathbf a\,dt.$$

- Why: $\tfrac{d}{dt}(\mathbf a^\top\delta\mathbf x)=\mathbf a^\top\tfrac{\partial\mathbf f}{\partial\boldsymbol\theta}\delta\boldsymbol\theta$; the state terms cancel.
- Each backward step is one vector–Jacobian product.
:::

::: {.slide title="Backpropagation Through Euler Is the Discrete Adjoint"}
[Discrete adjoint]{.kicker}

| backprop through the program | Euler step of the adjoint ODE |
|---|---|
| $\mathbf a_k=(\mathbf I+h\mathbf J(\mathbf x_k))^\top\mathbf a_{k+1}$ | $\mathbf a_k=\mathbf a_{k+1}+h\mathbf J(\mathbf x_{k+1})^\top\mathbf a_{k+1}$ |

Spiral, $\mathcal L=\tfrac12\|\mathbf x(T)\|^2$, $T=2$, $1000$ steps: a hand-written
backward sweep against finite differences.

<!-- WRITER (slides): the cell #odes-solvers-adjoint-check is merged into
#mdl-neural-odes-backpropagation-as-the-discrete-adjoint-1 (shown on the next
slide); its first output line is the finite-difference check. -->
:::

::: {.slide title="The Two Gradients Differentiate Different Objects and Agree to $O(h)$"}
[Discrete vs continuous]{.kicker}

Discrete adjoint (the program) against the continuous adjoint (the exact flow),
$\partial\mathcal L/\partial \mathbf A=\int_0^T\mathbf a\,\mathbf x^\top dt$:

@!mdl-neural-odes-backpropagation-as-the-discrete-adjoint-1

Scalar case: $Tx_N^2/(1+h\theta)=6.12,\ 7.24,\ 7.37$ at $N=10,100,1000$, against $e^2=7.39$.
:::

::: {.slide title="Re-Integrating Backward Saves Memory and Amplifies Errors by $e^{\lambda T}$"}
[Memory and drift]{.kicker}

- Store all $N$ states, or re-integrate $\mathbf x$ backward next to $\mathbf a$ (constant memory).
- $\dot x=-\lambda x$ run backward is $\dot x=+\lambda x$: an error $\delta$ in $x(T)$ grows to $e^{\lambda T}\delta$.
- $\lambda=50$, $\delta=10^{-16}$: $2.7\times10^{-11}$ at $T=0.25$, $5\times10^{5}$ at $T=1$.
- Checkpoints every $k=\sqrt N$ steps: $63$ states instead of $1000$, one extra forward pass.
:::

::: {.slide title="Check: How Much Drift?"}
[Check]{.kicker}

$\dot x=-50x$ over $T=0.5$: how large does a $10^{-16}$ error in $x(T)$ become at
$t=0$? And for the spiral, real part $-0.5$, over $T=2$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $e^{25}\cdot10^{-16}\approx7\times10^{-6}$. For the spiral the factor
is $e^{0.5\cdot2}=e$, so the error stays near $3\times10^{-16}$.
:::
:::
:::

::: {.slide title="A Flow Changes Log-Density at the Rate $-\operatorname{tr}\mathbf J$"}
[Instantaneous change of variables]{.kicker}

- 1-D: $\dot x=\theta x$ stretches by $e^{\theta t}$, so $\log p_t(x(t))=\log p_0(x_0)-\theta t$.
- Lemma: $\det(\mathbf I+h\mathbf J)=1+h\operatorname{tr}\mathbf J+O(h^2)$ (off-diagonal terms are $O(h^2)$).
- One short step of the flow has Jacobian $\mathbf I+h\mathbf J+o(h)$, hence
$$\frac{d}{dt}\log p_t(\mathbf x(t))=-\operatorname{tr}\mathbf J,\qquad
\log p_T(\mathbf x(T))=\log p_0(\mathbf x(0))-\int_0^T\operatorname{tr}\mathbf J\,dt.$$
:::

::: {.slide title="The Spiral Shrinks Area by $e^{-t}$ and Raises Density by $e^{t}$"}
[Divergence as volume change]{.kicker}

@fig:mdl-dyn-divergence-volume

$e^{\mathbf At}=e^{-t/2}\times$rotation, $\operatorname{tr}\mathbf A=-1$; Jacobi:
$\det e^{\mathbf At}=e^{t\operatorname{tr}\mathbf A}$, $0.2231$ at $t=1.5$.
:::

::: {.slide title="The Trace Integral Equals $\log\lvert\det D\Phi\rvert$ Even When the Trace Varies"}
[Checking the trace integral]{.kicker}

<!-- WRITER (slides): #odes-solvers-cnf-trace (spiral, Gaussian formula) is
cut; this slide keeps only the nonlinear field. -->

Tanh field $\mathbf W_2\tanh(\mathbf W_1\mathbf x+\mathbf b_1)$, $8$ units: RK4 on $(\mathbf x,\int\operatorname{tr}\mathbf J)$
against finite differences of $\Phi_1$:

@!mdl-neural-odes-checking-the-trace-integral-1
:::

::: {.slide title="Hutchinson Estimates the Trace from One VJP per Probe"}
[The Hutchinson estimator]{.kicker}

$\mathbb{E}[\boldsymbol\epsilon\boldsymbol\epsilon^\top]=\mathbf I$ gives
$\mathbb{E}[\boldsymbol\epsilon^\top \mathbf J\boldsymbol\epsilon]=\operatorname{tr}(\mathbf J\,\mathbb{E}[\boldsymbol\epsilon\boldsymbol\epsilon^\top])=\operatorname{tr}\mathbf J$.

- $\boldsymbol\epsilon^\top \mathbf J$ is one reverse-mode pass seeded with $\boldsymbol\epsilon$; $\mathbf J$ is never formed.
- Tanh field: $\boldsymbol\epsilon^\top \mathbf J=((\boldsymbol\epsilon^\top \mathbf W_2)\odot\mathbf s)\mathbf W_1$, $\mathbf s=1-\tanh^2(\cdot)$.

$d=50$, $64$ hidden units, $10^5$ Rademacher probes:
<!-- WRITER (slides): now the field W2 tanh(W1 x + b1) - x, d = 50, 64 hidden
units, 1000 probes, Rademacher against Gaussian; the held-against-fresh cell is
new and has no slide yet. -->

@!mdl-neural-odes-the-hutchinson-trace-estimator-1

@!mdl-neural-odes-the-hutchinson-trace-estimator-2
:::

::: {.slide title="Check: Sign and Cost"}
[Check]{.kicker}

For the spiral, is $\tfrac{d}{dt}\log p_t(\mathbf x(t))$ positive or negative, and
how large? What does one Hutchinson probe cost?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $-\operatorname{tr}\mathbf A=+1$: the contracting flow concentrates
probability. One probe costs one backward pass through $\mathbf f$.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- A ResNet is an Euler program; the trained field's flow lands $0.02$ from its
  ten-step program, and round trips shrink like $O(h)$.
- On $\mathbb R$ flows preserve order: $x\mapsto-x$ needs a second dimension.
- Backprop through Euler matches finite differences to $10^{-10}$; it is within $O(h)$ of the continuous adjoint.
- Re-integration drifts by $e^{\lambda T}$: $5\times10^5$ for $\lambda=50$, $T=1$.
- $\tfrac{d}{dt}\log p=-\operatorname{tr}\mathbf J$, estimated with one VJP per probe.

**Next:** fresh noise at every step, with increments of size $\sqrt h$.
:::
