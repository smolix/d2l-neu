# Momentum
:label:`sec_momentum`

Gradient descent uses only the gradient at the current point. On an
ill-conditioned objective, a learning rate small enough for directions of
high curvature gives slow progress in directions of low curvature. Momentum
addresses this problem by maintaining a running average of past gradients,
called the *velocity*. It requires one state buffer and a decay parameter,
and is particularly effective on ill-conditioned objectives. The Adam family
in :numref:`sec_adam` also incorporates momentum.

```{.python .input #momentum}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input #momentum}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import optax
```

## An Ill-Conditioned Valley

In :numref:`sec_gd` we minimized $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, a
moderately distorted bowl, and already saw the trajectory bend: the two
coordinates wanted different step sizes. We make the distortion pronounced by
flattening the first direction,

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

The minimum is still at $(0, 0)$, but the curvature is now $0.2$ in the
$x_1$ direction and $4$ in $x_2$ — a ratio of $20$. Gradient descent with
learning rate $0.4$ does what it can:

```{.python .input #momentum-an-ill-conditioned-problem-1}
eta = 0.4
def f_2d(x1, x2):  # Objective
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def f_2d_grad(x1, x2):  # Gradient of the objective
    return (0.2 * x1, 4 * x2)
def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d, f_grad=f_2d_grad))
```

The gradient in the $x_2$ direction is much larger and changes much faster
than in $x_1$, so one learning rate must accommodate both coordinates. A small rate prevents
divergence in $x_2$ but makes slow progress along $x_1$. Raise
it, and progress along $x_1$ improves while $x_2$ starts to oscillate out of
control. Even the slight increase from $0.4$ to $0.6$ tips the balance:

```{.python .input #momentum-an-ill-conditioned-problem-2}
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d, f_grad=f_2d_grad))
```

The trajectory history distinguishes these two directions. Along $x_1$ successive gradients agree — small, but all pointing
the same way. Along $x_2$ they alternate in sign, each step undoing the last.
An average over past gradients would amplify the first and cancel the second.

## The Momentum Method

### Leaky Averages

Minibatches (:numref:`sec_minibatch_sgd`) average gradients across
*examples*. The idea here is to also average across *time*, with an average
that leaks: discount each past gradient by a factor of $\beta$ per step.
Concretely, replace the gradient in the update by a *velocity*
$\mathbf{v}_t$,

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta\, \mathbf{v}_t,
\end{aligned}
$$
:eqlabel:`eq_momentum`

where $\beta \in [0, 1)$, $\mathbf{g}_t$ is the gradient — full-batch,
single-example, or minibatch — evaluated at $\mathbf{x}_{t-1}$, and
$\mathbf{v}_0 = \mathbf{0}$. For $\beta = 0$ we recover plain gradient
descent. Unrolling the recursion shows what the velocity holds:

$$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau},$$

an exponentially weighted sum of all past gradients. The name comes from the
physical picture: a heavy ball rolling down the objective integrates past
forces rather than reacting to the instantaneous slope, and $1 - \beta$
plays the role of friction. This is *heavy-ball momentum*, due to
:citet:`Polyak.1964`; :citet:`Sutskever.Martens.Dahl.ea.2013` document how
much it matters for training deep networks, and the expository article by
:citet:`Goh.2017` develops everything in this section with interactive
animations.

### Back to the Valley

On the valley, the leaky average does exactly what the trajectory history
suggested: the persistent $x_1$ components accumulate while the alternating
$x_2$ components cancel. At learning rate $0.6$, for which plain gradient
descent diverged, momentum $\beta = 0.5$ converges well:

```{.python .input #momentum-the-momentum-method-1}
def momentum_2d(x1, x2, v1, v2, f_grad):
    g1, g2 = f_grad(x1, x2)
    v1, v2 = beta * v1 + g1, beta * v2 + g2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d, f_grad=f_2d_grad))
```

Reducing the momentum to $\beta = 0.25$ weakens the effect, and the
trajectory converges slowly. Plain gradient descent diverged at this learning
rate:

```{.python .input #momentum-the-momentum-method-2}
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d, f_grad=f_2d_grad))
```

Nothing in :eqref:`eq_momentum` requires the gradient to be exact. With
minibatch gradients the same leaky average additionally smooths the sampling
noise across steps. This reduces variance beyond minibatch averaging without
additional gradient evaluations. Momentum therefore addresses both
ill-conditioning and sampling noise.

### The Timescale of $\beta$
:label:`subsec_momentum-timescale`

How much history does the velocity hold? The weights
$1, \beta, \beta^2, \ldots$ sum to $\frac{1}{1-\beta}$ in the limit, so a
useful reading is: **momentum $\beta$ averages over roughly the last
$\frac{1}{1-\beta}$ gradients**. $\beta = 0.9$ — the `momentum=0.9` you
have been passing to optimizers since :numref:`sec_training_recipes` — looks
back about $10$ steps; $\beta = 0.99$ about $100$. The plot shows how sharply
the weights decay for various $\beta$:

```{.python .input #momentum-effective-sample-weight}
d2l.set_figsize()
x = d2l.numpy(d2l.arange(40))
for beta in [0.95, 0.9, 0.6, 0]:
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

The same sum says something about step length. When successive gradients
roughly agree, the velocity builds up to $\frac{1}{1-\beta}$ times a typical
gradient, so momentum takes steps of effective size
$\frac{\eta}{1-\beta}$ in persistent directions. This matters when tuning:
raising $\beta$ without lowering $\eta$ makes the updates larger, not just
smoother, and the two hyperparameters must move together — we will see this
in the experiments below.

### Acceleration and Damping
:label:`subsec_momentum_acceleration`

Momentum does more than stabilize; on ill-conditioned problems it is
provably *faster*. For a quadratic whose Hessian eigenvalues lie between
$\mu$ and $L$, the condition number $\kappa = L/\mu$ governs the worst-case
rate:
gradient descent needs on the order of $\kappa \log \frac{1}{\epsilon}$
iterations to reach precision $\epsilon$, while heavy-ball momentum with
optimally chosen $\eta$ and $\beta$ needs only on the order of
$\sqrt{\kappa} \log \frac{1}{\epsilon}$, achieved at

$$\beta^\star = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{\!2}.$$

For $\kappa = 100$ that is a $10\times$ saving with $\beta^\star \approx
0.67$; for $\kappa = 10^4$, a $100\times$ saving with $\beta^\star \approx
0.96$. Note the trend: as $\kappa$ grows, $\beta^\star$ approaches $1$. In the timescale
interpretation, the optimum averages roughly $\sqrt{\kappa}$ steps.

A damped oscillator provides a useful model for tuning $\beta$. In each
eigendirection of the Hessian, :eqref:`eq_momentum` is a second-order
recurrence — a mass on a spring with friction $1 - \beta$. Too little
momentum and the system is *over-damped*: it creeps down the valley like
gradient descent. Too much and it is *under-damped*: the iterate overshoots
and rings around the minimum. The fastest setting, $\beta^\star$, sits at
critical damping between the two. Our valley has $\kappa = 20$, giving
$\beta^\star \approx 0.4$ — which is close to the earlier successful setting: $\eta = 0.6$ with
$\beta = 0.5$. Push $\beta$ too far and momentum turns against us. Here is
$\beta = 0.8$, well past the fastest-converging $\beta^\star$, at a
learning rate where plain gradient descent would be perfectly stable:

```{.python .input #momentum-acceleration-and-damping}
eta, beta = 0.3, 0.8
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d, f_grad=f_2d_grad))
```

![Convergence rate per step of heavy-ball momentum on a single quadratic mode, as a function of the momentum $\beta$. Below the critical value $\beta^{*}$ the iteration is over-damped and slow; at $\beta^{*}$ it is fastest; beyond it the rate degrades gently as $\sqrt{\beta}$. Worse conditioning (smaller $\eta\lambda$, dashed) pushes $\beta^{*}$ toward one — the reason large momentum values are the common default.](../img/mdl-opt-critical-damping.svg)
:label:`fig_opt_critical_damping`

The trajectory advances along the valley but oscillates around the minimum
before settling. This underdamped behavior differs from the earlier
learning-rate divergence. :numref:`fig_opt_critical_damping`
summarizes the tradeoff on a single quadratic mode: the per-step
convergence rate falls as $\beta$ grows, is best at a critical value
$\beta^{*}$, and degrades gently past it.
:numref:`subsec_mdl-momentum-acceleration` develops the eigenmode analysis
behind this picture, the $\sqrt{\kappa}$ theorem and its matching lower
bound, and the proofs. The scope of the result is important: the
heavy-ball $\sqrt{\kappa}$ rate is a statement about quadratics, and its
practical standing on general objectives rests on the local quadratic
approximation together with empirical evidence
:cite:`Sutskever.Martens.Dahl.ea.2013`.

## Implementation

### From Scratch

Compared with plain minibatch SGD, momentum needs to maintain one auxiliary
buffer per parameter — the velocity, with the same shape as the parameter.
In the harness of :numref:`sec_minibatch_sgd` this is exactly what the
`states` argument is for.

```{.python .input #momentum-implementation-from-scratch-1}
%%tab pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input #momentum-implementation-from-scratch-1}
%%tab jax
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return [v_w, v_b]
```

```{.python .input #momentum-implementation-from-scratch-2}
%%tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.zero_()
```

```{.python .input #momentum-implementation-from-scratch-2}
%%tab jax
def sgd_momentum(params, grads, states, hyperparams):
    for i in range(len(params)):
        states[i] = hyperparams['momentum'] * states[i] + grads[i]
        params[i] = params[i] - hyperparams['lr'] * states[i]
    return params[0], params[1]
```

On the airfoil regression problem, a moderate $\beta = 0.5$ with learning
rate $0.02$ trains without drama:

```{.python .input #momentum-implementation-from-scratch-3}
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Raising the momentum to $\beta = 0.9$ extends the average to roughly
$\frac{1}{1-0.9} = 10$ past gradients — and, by the effective-step reading
above, quintuples the effective step $\frac{\eta}{1-\beta}$ if we leave
$\eta$ alone. We halve the learning rate to $0.01$ to rein it in:

```{.python .input #momentum-implementation-from-scratch-4}
train_momentum(0.01, 0.9)
```

Halving it again to $0.005$ brings the effective step to $0.05$ — the range
of the first experiment — and the loss curve settles accordingly:

```{.python .input #momentum-implementation-from-scratch-5}
train_momentum(0.005, 0.9)
```

### Concise Implementation

Momentum is built into every framework's SGD optimizer as a single argument.
Matching the hyperparameters reproduces the trajectory.

```{.python .input #momentum-concise-implementation}
%%tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input #momentum-concise-implementation}
%%tab jax
trainer = optax.sgd
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Nesterov Momentum

Heavy ball has one characteristic failure, and we have already seen it: with
$\beta$ past the critical value, the accumulated velocity overshoots and the
iterate rings around the minimum. :citet:`Nesterov.1983` proposed a fix of
by evaluating the gradient at a look-ahead point. Evaluate the gradient not at
the current point but at the point the velocity is about to carry you to:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \nabla f(\mathbf{x}_{t-1} - \eta \beta\, \mathbf{v}_{t-1}), \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta\, \mathbf{v}_t.
\end{aligned}
$$
:eqlabel:`eq_nesterov`

If the momentum step is about to overshoot, the gradient at the look-ahead
point already points back, correcting the velocity *before* the mistake
rather than one step after. In code it is a two-line change to the momentum
update — the gradient is taken at the shifted point:

```{.python .input #momentum-nesterov-momentum-1}
def nesterov_2d(x1, x2, v1, v2, f_grad):
    g1, g2 = f_grad(x1 - eta * beta * v1,  # Gradient at the look-ahead
                    x2 - eta * beta * v2)  # point, not at (x1, x2)
    v1, v2 = beta * v1 + g1, beta * v2 + g2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.3, 0.8
d2l.show_trace_2d(f_2d, d2l.train_2d(nesterov_2d, f_grad=f_2d_grad))
```

At the same learning rate and $\beta = 0.8$ used for the oscillatory
heavy-ball run, Nesterov momentum removes the visible oscillation: the look-ahead acts as built-in damping. Nesterov's method also has guarantees beyond this quadratic example that
heavy-ball momentum lacks. On
smooth convex functions it converges as $\mathcal{O}(1/k^2)$ against
gradient descent's $\mathcal{O}(1/k)$, which is optimal for any method built
from gradients; on strongly convex functions it achieves the
$\sqrt{\kappa}$ rate with a proof that is not confined to quadratics
:cite:`Nesterov.2018`. Statements, proofs, and the matching lower bound are
in :numref:`subsec_mdl-momentum-acceleration`.

In frameworks, Nesterov momentum is one flag. Both PyTorch and Optax
implement an equivalent rewrite of :eqref:`eq_nesterov` that evaluates the
gradient at the current iterate (the exercises ask you to verify the
equivalence), so no extra gradient evaluation is needed:

```{.python .input #momentum-nesterov-momentum-2}
%%tab pytorch
d2l.train_concise_ch11(
    torch.optim.SGD,
    {'lr': 0.005, 'momentum': 0.9, 'nesterov': True}, data_iter)
```

```{.python .input #momentum-nesterov-momentum-2}
%%tab jax
d2l.train_concise_ch11(
    optax.sgd,
    {'learning_rate': 0.005, 'momentum': 0.9, 'nesterov': True}, data_iter)
```

On this small, noisy problem the curve is nearly indistinguishable from
plain momentum, as is typical at small batch: the look-ahead correction is
dwarfed by sampling noise.
The look-ahead correction matters most when curvature dominates noise, as in
full-batch or large-batch training with $\beta$ close to $1$. Because it
requires no additional gradient evaluation, implementations often enable it
by default.

## Summary

Momentum replaces the gradient with a leaky average over past gradients. It
adds one buffer and one hyperparameter, $\beta$. On ill-conditioned problems it cures
the zigzag: persistent gradient components accumulate up to
$\frac{1}{1-\beta}$-fold while oscillating ones cancel, and with optimal
tuning the iteration count improves from order $\kappa$ to order
$\sqrt{\kappa}$. The parameter $\beta$ reads as a timescale — an average
over roughly $\frac{1}{1-\beta}$ recent gradients — and as a damping knob,
with too large a value producing ringing rather than progress. Nesterov's
look-ahead variant damps that ringing and carries convergence guarantees
beyond quadratics, at no extra cost per step. With stochastic gradients the
same leaky average also smooths sampling noise, which is why some form of
momentum appears in most optimizers considered in the rest of this chapter.

## Exercises

1. [code] **Momentum and learning rate together.** On the valley `f_2d`
   with `momentum_2d`, run a grid of $\beta \in \{0, 0.25, 0.5, 0.8\}$ and
   $\eta \in \{0.1, 0.3, 0.6, 1.0\}$ for 100 steps each.
    1. Record for every cell the number of steps until
       $\|\mathbf{x}_t\| \leq 10^{-3}$, or the fact that the iterates
       diverged, and display the grid as a table or heatmap.
    1. Compare the fastest $\beta$ at each stable $\eta$ with the
       $\beta^\star$ of :numref:`subsec_momentum_acceleration` for
       $\kappa = 20$.
    1. The effective-step reading of :numref:`subsec_momentum-timescale`
       suggests that raising $\beta$ forces $\eta$ down. Does the largest
       stable $\eta$ in your grid behave that way? Derive the stability
       condition of :eqref:`eq_momentum` for a single eigendirection with
       curvature $\lambda$ and reconcile the two.
1. [code] **Many eigenvalues.** Consider
   $f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^{10} \lambda_i x_i^2$ with
   $\lambda_i = 2^{-i}$, started from $x_i = 1$.
    1. Run gradient descent and heavy-ball momentum at the largest stable
       learning rate of gradient descent and plot $|x_{i,t}|$ for every
       coordinate on a logarithmic scale.
    1. Which coordinate limits the convergence of gradient descent, and by
       what factor does momentum with the $\beta^\star$ of the resulting
       condition number shorten the time for that coordinate?
1. **Two forms of Nesterov momentum.** Library implementations of Nesterov
   momentum evaluate the gradient at the current iterate,
   $\mathbf{g}_t = \nabla f(\mathbf{x}_{t-1})$, and update
   $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t, \qquad
   \mathbf{x}_t = \mathbf{x}_{t-1} - \eta\,(\mathbf{g}_t + \beta \mathbf{v}_t),$$
   so that no extra gradient evaluation is needed.
    1. Show by a change of variables that this recurrence generates the
       same iterates as :eqref:`eq_nesterov`.
    1. Identify the point of :eqref:`eq_nesterov` to which the library's
       parameter vector $\mathbf{x}_t$ corresponds, and state what a
       checkpoint of those parameters therefore stores.
1. [code] **Critical damping.** For $f(x) = \frac{\lambda}{2} x^2$ at a
   fixed learning rate with $\eta\lambda = 0.1$, sweep $\beta$ over
   $[0, 1)$ in steps of $0.02$ and measure the number of iterations until
   $|x_t| \leq 10^{-6} |x_0|$.
    1. Locate the minimizing $\beta$ and compare it with the critical value
       $\beta^\star = (1 - \sqrt{\eta\lambda})^2$ at which the recurrence
       :eqref:`eq_momentum` for this mode is critically damped.
    1. Describe the iterates on either side of $\beta^\star$ and match them
       with the over-damped and under-damped regimes of
       :numref:`fig_opt_critical_damping`.
1. [code] **Momentum under minibatch noise.** With stochastic gradients,
   the leaky average in :eqref:`eq_momentum` also smooths the sampling
   noise.
    1. Let $\mathbf{g}_t = \mathbf{g} + \boldsymbol{\xi}_t$ with
       independent zero-mean noise of variance $\sigma^2$ per coordinate.
       Show that the stationary mean of $\mathbf{v}_t$ is
       $\mathbf{g}/(1-\beta)$ and its stationary variance
       $\sigma^2/(1-\beta^2)$, so that the noise-to-signal ratio of the
       velocity is $\sqrt{(1-\beta)/(1+\beta)}$ times that of a single
       gradient.
    1. On the airfoil harness, rebuild `data_iter` and `feature_dim` with
       `d2l.get_data_ch11(batch_size=b)` for $b \in \{1, 10, 100\}$ and
       run `d2l.train_ch11` with `sgd_momentum` at
       $\beta \in \{0, 0.9\}$, adjusting $\eta$ so that $\eta/(1-\beta)$
       stays at $0.05$. Report the final loss of each run and the spread of
       the recorded losses over its last epoch.
    1. At which batch size does momentum change the outcome most, and is
       the size of the change consistent with the ratio of the first item?

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1070)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1071)
:end_tab:

<!-- slides -->

::: {.slide title="Momentum"}
Gradient descent uses only the current gradient, which is inefficient when
different directions require different step sizes.

- Ill-conditioned valley: steep walls cap $\eta$, flat floor needs big
  $\eta$.
- Fix: a running (leaky) average of past gradients — the **velocity**.
- One buffer, one hyperparameter $\beta$; inside nearly every deep
  learning optimizer.
:::

::: {.slide title="An ill-conditioned valley"}
$f(x_1, x_2) = 0.1 x_1^2 + 2 x_2^2$ — curvatures $0.2$ vs $4$:

@momentum-an-ill-conditioned-problem-1


Raise $\eta$ from 0.4 to 0.6: $x_1$ speeds up, $x_2$ diverges:

@momentum-an-ill-conditioned-problem-2
:::

::: {.slide title="Leaky averages"}
Replace the gradient by a velocity:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t,\qquad
\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \mathbf{v}_t.$$

Unrolled: $\mathbf{v}_t = \sum_{\tau} \beta^{\tau} \mathbf{g}_{t-\tau}$ —
an exponentially weighted sum of the past.

- Components that **agree** accumulate (up to $\tfrac{1}{1-\beta}\times$).
- Components that **alternate** cancel.
- Heavy ball rolling downhill; friction $1-\beta$ (Polyak, 1964).
:::

::: {.slide title="Momentum in the valley"}
At $\eta = 0.6$, adding $\beta = 0.5$ changes the divergent gradient-descent
trajectory to:

@momentum-the-momentum-method-1


With $\beta = 0.25$, the trajectory converges more slowly:

@momentum-the-momentum-method-2
:::

::: {.slide title="The timescale of β"}
Weights sum to $\tfrac{1}{1-\beta}$: momentum $\beta$ ≈ average over the
last $\tfrac{1}{1-\beta}$ gradients. $\beta=0.9$ → ~10 steps;
$\beta=0.99$ → ~100.

@momentum-effective-sample-weight

Effective step in persistent directions: $\eta / (1-\beta)$ —
**raise $\beta$, lower $\eta$**.
:::

::: {.slide title="Acceleration: the √κ law"}
Quadratic with condition number $\kappa$:

- Gradient descent: $\mathcal{O}(\kappa \log \tfrac{1}{\epsilon})$ steps.
- Tuned momentum: $\mathcal{O}(\sqrt{\kappa} \log \tfrac{1}{\epsilon})$,
  at $\beta^\star = \left(\tfrac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$.
- $\kappa = 10^4$: hundreds of steps instead of tens of thousands.

Each eigenmode = damped oscillator; $\beta$ is the damping knob.
Proofs: math appendix (gradient-based optimization).
:::

::: {.slide title="Oscillation with excessive momentum"}
This valley: $\kappa = 20$ → $\beta^\star \approx 0.4$. Now
$\eta = 0.3$ (GD-stable), $\beta = 0.8$ — well past $\beta^\star$, under-damped:

@momentum-acceleration-and-damping

The iterate oscillates around the minimum before settling. Values below
$\beta^\star$ are overdamped; values above it are underdamped;
$\beta^\star$ gives critical damping.
:::

::: {.slide title="From scratch"}
Velocity = one buffer per parameter, carried in `states`:

@momentum-implementation-from-scratch-1


@momentum-implementation-from-scratch-2
:::

::: {.slide title="On the airfoil harness"}
$\beta = 0.5$, $\eta = 0.02$:

@momentum-implementation-from-scratch-3


$\beta = 0.9$ quintuples the effective step $\eta/(1-\beta)$ — so lower
$\eta$ to 0.01, then 0.005:

@momentum-implementation-from-scratch-4


@momentum-implementation-from-scratch-5
:::

::: {.slide title="Framework implementation"}
@momentum-concise-implementation
:::

::: {.slide title="Nesterov momentum"}
Evaluate the gradient at the point the velocity is taking you to:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla f(\mathbf{x}_{t-1} - \eta \beta \mathbf{v}_{t-1}),\qquad
\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \mathbf{v}_t.$$

The look-ahead gradient changes direction before the current iterate reaches
the corresponding overshoot.

@momentum-nesterov-momentum-1

At the same $\eta$ and $\beta$ as the underdamped example, the visible
oscillation is removed.
:::

::: {.slide title="Nesterov in practice"}
One flag; no extra gradient evaluations:

@momentum-nesterov-momentum-2

- Guarantees heavy ball lacks: $\mathcal{O}(1/k^2)$ convex (optimal),
  $\sqrt{\kappa}$ beyond quadratics.
- Small-batch noise dwarfs the correction → curves match plain momentum
  here. Matters when curvature dominates: large batches, $\beta \to 1$.
:::

::: {.slide title="Recap"}
- $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t$,
  $\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \mathbf{v}_t$.
- Persistent components accumulate, oscillating ones cancel; noise
  smooths too.
- $\beta$ = timescale ($\tfrac{1}{1-\beta}$ steps) **and** damping knob;
  tuned momentum: $\kappa \to \sqrt{\kappa}$.
- Nesterov look-ahead: damps ringing, adds guarantees, costs nothing.
- $\beta = 0.9$ is the default; Adam (:numref:`sec_adam`) keeps the idea
  and adds per-coordinate scaling.
:::
