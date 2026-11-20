# Gradient Penalties and Convergence
:label:`sec_gan_convergence`

The garment generator of :numref:`sec_basic_gan` learned to draw tops and shoes, but its losses never settled. After twenty epochs the critic's loss was still climbing toward the value it takes at the equilibrium, and the generator's was still falling toward its own. Is that slow progress, or does the game move around its equilibrium without reaching it? The last three sections cannot say. They computed what the game is worth when the critic answers optimally, and training never gives the critic that chance: both networks take small gradient steps, and each step of one player changes the objective of the other. Knowing where the equilibrium lies does not tell us whether these steps lead there.

:citet:`Mescheder.Geiger.Nowozin.2018` asked which training methods for GANs converge, and they answered with the smallest game there is: one data point, one generated point and a straight-line critic. Plain gradient steps fail even on this game, and a penalty on the critic's slope repairs them. The same penalty, added to the pairing loss of :numref:`sec_gan_relativistic`, gives the R3GAN recipe :cite:`Huang.Gokaslan.Kuleshov.ea.2024` that the rest of the chapter trains.

```{.python .input #convergence-gradient-penalties-and-convergence}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #convergence-gradient-penalties-and-convergence}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Dirac-GAN

The smallest game can be solved exactly, and it shows both obstacles of this section: its value is flat in the generator's position, and gradient steps rotate around the solution instead of approaching it.

### The Game on Two Points

Place the data at the origin and let the generator produce a single point on the real line: $p = \delta_0$ and $q_\theta = \delta_\theta$, so $q = p$ exactly when $\theta = 0$. Let the critic be linear, $D_\psi(x) = \psi x$. :numref:`sec_gan_objectives` scored a real sample $x$ by $\ell(D(x))$ and a generated sample $x'$ by $\ell(-D(x'))$ for a classification loss $\ell$, giving the margin objective

$$
V_\ell(D) = \mathbb{E}_{x \sim p}\big[\ell(D(x))\big] + \mathbb{E}_{x' \sim q}\big[\ell(-D(x'))\big].
$$

For two point masses both expectations are single evaluations:

$$
V(\theta, \psi) \;=\; \ell\big(D_\psi(0)\big) + \ell\big({-D_\psi(\theta)}\big)
\;=\; \ell(0) + \ell(-\psi\theta).
$$
:eqlabel:`eq_gan_dirac_value`

This is the *Dirac-GAN* of :citet:`Mescheder.Geiger.Nowozin.2018`. Any differentiable loss that increases everywhere will do, and we use the logistic loss $\ell = \log\sigma$, whose derivatives are $\ell'(t) = \sigma(-t)$ and $\ell''(t) = -\sigma(t)\sigma(-t)$, so $\ell'(0) = \tfrac12$ and $\ell''(0) = -\tfrac14$.

For every $\theta \neq 0$ the two point masses have disjoint supports. The critic's best response sends $\psi\theta \to -\infty$, so $\sup_\psi V = \ell(0) + \sup_t \ell(t)$, the same number for every $\theta \neq 0$. A value that does not depend on $\theta$ gives the generator no gradient, however close to the data it starts. This is the saturation of :numref:`sec_basic_gan` in its smallest form.

### Circles in Continuous Time

Training does not compute a best response. Both players take gradient steps from where they are. The game has one stationary point, $(\theta, \psi) = (0, 0)$, where the generated point matches the data and the critic is flat. In the simultaneous gradient flow the generator descends $V$ and the critic ascends it:

$$
\dot\theta \;=\; -\partial_\theta V \;=\; \psi\, \ell'(-\psi\theta),
\qquad
\dot\psi \;=\; +\partial_\psi V \;=\; -\theta\, \ell'(-\psi\theta).
$$
:eqlabel:`eq_gan_dirac_flow`

Both components are the same positive scalar $\ell'(-\psi\theta)$ times the vector $(\psi, -\theta)$, which is orthogonal to the position $(\theta, \psi)$. The squared distance to the equilibrium therefore does not change:

$$
\frac{d}{dt}\big(\theta^2 + \psi^2\big)
= 2\theta\psi\,\ell'(-\psi\theta) - 2\psi\theta\,\ell'(-\psi\theta) = 0 .
$$

Every trajectory is a circle around the solution, for every loss that increases everywhere. The generated point passes the data again and again without settling there. (A loss with a flat stretch, such as the hinge beyond its margin, has $\ell' = 0$ there and stops the motion wherever $-\psi\theta$ lands on it.)

The standard local test is linearization. Near a point where a smooth field vanishes, the field behaves like its linear part, the Jacobian $J$ at that point. If every eigenvalue of $J$ has negative real part, nearby trajectories approach the point at a linear rate. If the eigenvalues are purely imaginary, the linear part only rotates, the point is a *center*, and higher-order terms decide whether the orbits close, tighten or open (:numref:`subsec_mdl-ode-linearization`). Here

$$
J = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & 0 \end{pmatrix},
\qquad
\nu_{1,2} = \pm\, i\, \ell'(0),
$$
:eqlabel:`eq_gan_dirac_eigs`

a center. The test agrees with the circles but could not have found them on its own; the conservation law decides what the linear part leaves open, and the orbits close.

Linearized, $V \approx 2\ell(0) - \ell'(0)\,\theta\psi$, so up to a constant, the factor $\ell'(0)$ and the sign of $\psi$ the Dirac-GAN is the *bilinear game* $\min_\theta \max_\psi \theta\psi$. It is the standard example of the min-max literature, where methods such as optimistic gradient steps converge and plain gradient steps do not :cite:`Daskalakis.Ilyas.Syrgkanis.ea.2018`.

The common factor $\ell'(-\psi\theta) = \sigma(D_\psi(\theta))$ is the weight $w_{\textrm{sat}}$ with which the saturating generator of :numref:`sec_basic_gan` scales the update of a generated sample. Because it multiplies both components, it sets how fast the parameters turn and not where they go, so no factor common to both players' updates turns the circle into a spiral. A factor on one player alone can, as the non-saturating generator below shows.

### Outward Spirals in Discrete Time

Discrete steps do worse than the flow. Write $\mathbf{u}(\theta, \psi)$ for the right-hand side of :eqref:`eq_gan_dirac_flow`. A simultaneous step of size $h > 0$ maps $(\theta, \psi)$ to $(\theta, \psi) + h\,\mathbf{u}$. Since $\mathbf{u}$ is orthogonal to $(\theta, \psi)$, the Pythagorean theorem gives

$$
\big\| (\theta, \psi) + h\, \mathbf{u} \big\|^2
\;=\; \big\| (\theta, \psi) \big\|^2 + h^2 \big\| \mathbf{u} \big\|^2 ,
$$

so each step adds $h^2\|\mathbf{u}\|^2$ to the squared distance. For the logistic loss $\mathbf{u}$ vanishes only at the equilibrium, so every step from any other point moves away from the solution, and a smaller step only slows the outward spiral (exercise :exref:`ex_convergence-discrete-simultaneous-update-dirac`).

Every training loop in this chapter, however, alternates: the critic steps first, and the generator then steps against the updated critic. Linearized at the equilibrium, each half of such an update is a shear, which changes one coordinate by an amount that depends only on the other, and a shear preserves area. The composite update therefore has determinant one: no direction can shrink unless another grows, so the iterates cannot approach the solution at a linear rate. For $h\,\ell'(0) \leq 2$ both eigenvalues lie on the unit circle :cite:`Mescheder.Geiger.Nowozin.2018`, so the iterates do not escape at a linear rate either, and in practice they circle in a band around the solution (exercise :exref:`ex_convergence-alternating-updates`). Alternating is better than simultaneous play, and still not convergent.

The two repairs of the earlier sections leave these dynamics in place. The non-saturating generator has the same Jacobian $J$ at the equilibrium; away from it the field gains a slow inward drift, which discrete steps stop on a cycle of radius about $\sqrt{2h}$ (exercise :exref:`ex_convergence-non-saturating-dynamics`). The pairing objective of :numref:`sec_gan_relativistic` scores the pair by $\ell\big(D_\psi(0) - D_\psi(\theta)\big) = \ell(-\psi\theta)$, which differs from :eqref:`eq_gan_dirac_value` only by the constant $\ell(0)$, so its gradient field, its circles and its spirals are the classical ones. A cure has to change the field near the equilibrium, and both repairs leave it as it was.

## Zero-Centered Penalties

The linearization test tells us what to aim for: move both eigenvalues of $J$ off the imaginary axis into the left half-plane, and nearby trajectories converge at a linear rate. (This is sufficient, not necessary: the non-saturating flow converges with imaginary eigenvalues, but only at the rate $r^2 \approx 8/t$ of exercise :exref:`ex_convergence-non-saturating-dynamics`.) The rotation needs friction. Which player should feel it? We do not know where the generator must end, since that is the unknown data distribution. We do know where the critic must end: at the equilibrium it cannot tell $p$ from $q$, so it is flat. Friction that pulls the critic's slope toward zero therefore leaves the equilibrium where it is, and on the Dirac-GAN the critic's slope is $\psi$, the very quantity that keeps the rotation going. Define the two *zero-centered gradient penalties*

$$
R_1 \;=\; \frac{\gamma}{2}\, \mathbb{E}_{x \sim p}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad
R_2 \;=\; \frac{\gamma}{2}\, \mathbb{E}_{x' \sim q}\Big[ \big\| \nabla_{x'} D(x') \big\|^2 \Big],
$$
:eqlabel:`eq_gan_r1r2`

with weight $\gamma > 0$, and let the critic maximize $V_\ell(D) - R_1 - R_2$ while the generator's objective stays unchanged. The penalties act on the critic's gradient with respect to its input, not its parameters: $R_1$ limits how steeply the critic varies at data samples and $R_2$ at generated samples. They are called zero-centered because the flat critic $\nabla_x D = 0$ minimizes both. At $q = p$ the optimal critic $\log(p/q)$ of :numref:`sec_basic_gan` is the constant zero, so the penalties vanish at the equilibrium they are meant to stabilize. Since $\tfrac12(\mathbb{E}_p + \mathbb{E}_q) = \mathbb{E}_m$, their sum is

$$
R_1 + R_2 \;=\; \gamma\, \mathbb{E}_{x \sim m}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad m = \frac{p + q}{2},
$$
:eqlabel:`eq_gan_r1r2_sum`

one smoothness penalty under the mixture $m$, which constrains the critic wherever either distribution puts mass, and nowhere else.

The penalty did not start as a cure for rotation. :citet:`Roth.Lucchi.Nowozin.ea.2017` derived a penalty of this kind as the local effect of adding Gaussian noise to both distributions, with the weights $(1 - \sigma(D))^2$ on real and $\sigma(D)^2$ on generated samples. :citet:`Mescheder.Geiger.Nowozin.2018` dropped the weights, named the two halves $R_1$ and $R_2$, and showed what they do to the dynamics.

### Damping the Dirac-GAN

For the linear critic, $\nabla_x D_\psi = \psi$ at every $x$, so both penalties equal $\tfrac{\gamma}{2}\psi^2$. Subtracting one of them from the critic's objective adds $-\gamma\psi$ to the critic's update:

$$
\dot\theta = \psi\,\ell'(-\psi\theta),
\qquad
\dot\psi = -\theta\,\ell'(-\psi\theta) - \gamma\psi .
$$

Using both penalties replaces $\gamma$ by $2\gamma$. Only the lower-right entry of the Jacobian changes:

$$
J_\gamma = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & -\gamma \end{pmatrix},
\qquad
\nu_{1,2} \;=\; -\frac{\gamma}{2} \pm \sqrt{ \frac{\gamma^2}{4} - \ell'(0)^2 } .
$$
:eqlabel:`eq_gan_dirac_pen`

Both eigenvalues have negative real parts for every $\gamma > 0$. The equilibrium of the Dirac-GAN attracts, and small enough simultaneous or alternating steps converge to it at a linear rate; in the underdamped regime the simultaneous step must satisfy $h < \gamma/\ell'(0)^2$ (exercise :exref:`ex_convergence-discrete-simultaneous-update-dirac`). The characteristic polynomial $\nu^2 + \gamma\nu + \ell'(0)^2$ is that of a damped spring, the damped pendulum of :numref:`subsec_mdl-ode-linearization` with friction $\gamma$, and it has the spring's three regimes. For $\gamma < 2\ell'(0)$ the square root is imaginary and trajectories spiral inward. At $\gamma = 2\ell'(0)$, which is $\gamma = 1$ for the logistic loss, the repeated real eigenvalue gives critical damping, the fastest approach without oscillation. Larger values overdamp the dynamics: the slower eigenvalue, close to $-\ell'(0)^2/\gamma$, makes convergence slower as $\gamma$ grows. More friction is not always better.

### Phase Portraits

Four predictions are now on the table. The flow keeps its distance from the equilibrium, simultaneous steps move away from it at every step, alternating steps stay in a band around it, and the penalty brings the iterates in. We check them by simulating the two equations for the logistic loss. The function `trajectory` repeats one update of step size $h$: `rk4_step` integrates the flow with a fourth-order Runge--Kutta step, `euler_step` takes one simultaneous gradient step, and `alt_step` updates the critic first and then the generator against the new critic.

```{.python .input #convergence-phase-portraits}
%%tab pytorch, jax
def dirac_field(theta, psi, gamma=0.0):
    """Simultaneous gradient field of the Dirac-GAN with the logistic loss."""
    w = 1.0 / (1.0 + np.exp(-psi * theta))      # l'(-psi theta) = w_sat
    return np.array([psi * w, -theta * w - gamma * psi])

def euler_step(x, h, gamma):                     # one simultaneous step
    return x + h * dirac_field(*x, gamma)

def alt_step(x, h, gamma):                       # critic first, then generator
    theta, psi = x
    psi = psi + h * dirac_field(theta, psi, gamma)[1]
    return np.array([theta + h * dirac_field(theta, psi, gamma)[0], psi])

def rk4_step(x, h, gamma):                       # one RK4 step of the ODE
    k1 = dirac_field(*x, gamma)
    k2 = dirac_field(*(x + h / 2 * k1), gamma)
    k3 = dirac_field(*(x + h / 2 * k2), gamma)
    k4 = dirac_field(*(x + h * k3), gamma)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def trajectory(step, x0, h, steps, gamma=0.0):
    xs = [np.array(x0)]
    for _ in range(steps):
        xs.append(step(xs[-1], h, gamma))
    return np.stack(xs)
```

All paths start at the red dot, $(\theta, \psi) = (1, 1)$, at distance $\sqrt 2 \approx 1.41$ from the equilibrium at the black cross. The left panel shows the flow. The middle panel shows 300 simultaneous and 2,000 alternating steps of size $h = 0.2$. The right panel shows 300 simultaneous steps of the same size with the penalty, at $\gamma = 0.3$ in the underdamped regime and at the critical $\gamma = 1$.

```{.python .input #convergence-phase-portraits-1}
%%tab pytorch, jax
x0 = np.array([1.0, 1.0])
panels = {'flow': {'RK4': trajectory(rk4_step, x0, 0.05, 800)},
          'steps': {'simultaneous': trajectory(euler_step, x0, 0.2, 300),
                    'alternating': trajectory(alt_step, x0, 0.2, 2000)},
          'penalized': {
              'gamma = 0.3': trajectory(euler_step, x0, 0.2, 300, gamma=0.3),
              'gamma = 1': trajectory(euler_step, x0, 0.2, 300, gamma=1.0)}}
fig, axes = d2l.plt.subplots(1, 3, figsize=(9.5, 3.4))
for ax, (title, paths) in zip(axes, panels.items()):
    for name, P in paths.items():
        r = np.linalg.norm(P, axis=1)            # distance to the equilibrium
        late = r[len(r) // 2:]
        print(f'{title}, {name}: {r[0]:.3g} -> {r[-1]:.3g} in {len(r) - 1} '
              f'steps; second half in [{late.min():.3g}, {late.max():.3g}]')
    lim = 1.25 * max(np.abs(P).max() for P in paths.values())
    d2l.plot([P[:, 0] for P in paths.values()] + [x0[:1], [0.0]],
             [P[:, 1] for P in paths.values()] + [x0[1:], [0.0]],
             r'$\theta$', r'$\psi$', list(paths) if len(paths) > 1 else None,
             xlim=(-lim, lim), ylim=(-lim, lim), axes=ax,
             fmts=('C0-', 'C1-')[:len(paths)] + ('C3o', 'k+'),
             legend_loc='lower right' if title == 'penalized' else None)
    ax.set_title(title)
    ax.set_aspect('equal')
fig.tight_layout()
```

The integrated flow stays on its circle at 1.41, although it too takes discrete steps. The stability disc $|1 + h\nu| < 1$ of a forward Euler step touches the imaginary axis only at the origin, while RK4's stability region contains a stretch of it, up to $|h\nu| = 2\sqrt2$, so RK4 can follow a rotation that Euler cannot (:numref:`sec_mdl-stiffness-implicit`). The simultaneous iterates move outward at every step; after 300 steps their distance from the equilibrium has doubled, from 1.41 to 2.87. The alternating iterates circle in a band: over their last 1,000 steps the distance stays between 1.34 and 1.42, bounded and not convergent. Both penalized paths converge, the underdamped one along a shrinking spiral to a distance of $6.5 \cdot 10^{-4}$ after 300 steps and the critically damped one directly, to $7 \cdot 10^{-12}$. The penalty turns divergence into convergence and leaves the equilibrium where it was.

### Local Convergence beyond Two Parameters

What carries over to networks? :citet:`Mescheder.Geiger.Nowozin.2018` proved a local theorem for the classical per-sample objective with $R_1$, $R_2$ or any mixture of the two: small enough simultaneous or alternating steps converge at a linear rate to the set of equilibria, if training starts close enough to it and four conditions hold.

1. The generator can represent the data exactly, and at the equilibrium the critic is zero in a neighbourhood of the data's support.
2. $\ell'(0) \neq 0$ and $\ell''(0) < 0$, as for the logistic loss.
3. The critic class detects every first-order departure of the generator from the data, and every change of the critic's parameters that leaves the equilibrium changes the critic or its slope on the data. This is the one place where the critic's expressiveness enters.
4. The parameters that realize the equilibrium form smooth manifolds, since many weight settings represent the same function.

We do not prove this. The theorem is local: it says nothing about how training reaches the neighbourhood, and its first condition is bold, since no finite network reproduces real data exactly. It also covers the classical loss, not the pairing loss of R3GAN; :citet:`Huang.Gokaslan.Kuleshov.ea.2024` prove a local result of the same kind for the pairing loss with $R_1$ or $R_2$ under similar assumptions, which we do not reproduce either. On two parameters we proved convergence; for networks, "the penalties make the equilibrium locally attracting" is a theorem under these conditions and a hope outside them.

### One-Centered versus Zero-Centered

WGAN-GP centers its penalty differently :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017`. To enforce the Lipschitz constraint of :numref:`sec_gan_objectives`, it penalizes $\mathbb{E}\big[(\|\nabla_x D\| - 1)^2\big]$ at points between real and generated samples, because an optimal $W_1$ critic has unit slope along the straight paths on which the optimal transport moves mass. At $q = p$, however, every critic scores zero under the $W_1$ objective, so the penalty alone chooses the critic, and it chooses unit slope. A penalty that prefers unit slope keeps pushing a generator that already matches the data, and on the Dirac-GAN these dynamics do not converge to the data :cite:`Mescheder.Geiger.Nowozin.2018`. With the linear score $\ell(t) = t$ of the Wasserstein critic and $\gamma = 3$, the generator parks at $\theta = 3$, three units from the data. There the critic's drive to tilt against the generated point, of size $\theta$, cancels the penalty's push toward unit slope, of size $\gamma$. The critic stays flat, and a flat critic gives the generator no push (exercise :exref:`ex_convergence-one-centered-dynamics`).

### One Penalty or Two

The Dirac-GAN cannot tell $R_1$ from $R_2$, because a linear critic has the same slope everywhere, and the theorem above holds for either penalty alone. Far from equilibrium the two differ. Keep the data at $0$, put a generated point at $x' = 1$, and consider the critics $D_c(x) = -c\,x^2$, flat at the data and steep at the generated point. $R_1$ measures the slope only at the data, where it is zero, so the critic objective $V_\ell(D_c) - R_1$ keeps rising as $c$ grows, toward a supremum that no finite $c$ attains, while the non-saturating generator's gradient at $x'$ grows with $c$ (exercise :exref:`ex_convergence-one-sided-penalty`). $R_2$ charges the slope $2c$ at the generated point and caps $c$. With $R_1$ alone the critic is free exactly where the generator's samples lie.

Whether that freedom matters in practice depends on the problem. In the StackedMNIST ablation of :numref:`sec_gan_relativistic`, training with $R_1$ alone diverged for both losses, and only with both penalties did either recover most of the modes :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. StyleGAN2, on the other hand, trains faces well with $R_1$ alone :cite:`Karras.Laine.Aittala.ea.2020`, so one ablation does not settle whether one penalty suffices. We use both.

## The Penalized Value Near Equilibrium

The one-centered penalty moved the point where training comes to rest. Does the zero-centered one change what the generator is chasing? At the equilibrium it vanishes, so the target $q = p$ stays where it was. Near the equilibrium, however, the penalty changes the critic's best response, and with it the value the generator descends. Without the penalty that value was a divergence of the density ratio, blind to how far apart separated supports are. What does the penalized game measure?

This is the hardest part of the section. A reader who wants only the recipe can take two facts from it: the penalty shrinks the critic's slope by the factor $1 + 4\gamma$, and $\gamma$ is a squared length, set by the scale of the data. The general formula and its dual Sobolev norm explain the experiments and the exercises. General critics are hard to study, so let us start with a case where a straight line is the best critic, and generalize afterwards.

### A Gaussian Shift

Near $q = p$ the best critic is small, so we expand the payoff to second order in the critic. Write $\langle p - q, D \rangle = \mathbb{E}_p[D] - \mathbb{E}_q[D]$ for the difference of the critic's mean scores. Expanding $\ell(t) \approx \ell(0) + \ell'(0)\,t + \tfrac12 \ell''(0)\,t^2$ in both terms of $V_\ell$ and using $\ell''(0) < 0$ gives

$$
V_\ell(D) \approx 2\ell(0) + \ell'(0)\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 .
$$

The first-order term rewards a critic that is high where $p$ exceeds $q$; the second-order term charges its size. The quadratic term is not negligible: the best critic has size proportional to $p - q$, so all terms are of the same order at the optimum.

Now let $p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\mu, 1)$ with a small shift $\mu$, and try the linear critic $D(x) = c\,x$. Then $\langle p - q, D\rangle = -c\mu$, the penalty $\gamma \int m\, \|\nabla_x D\|^2$ is $\gamma c^2$, and $\int m\, D^2 = c^2(1 + \mu^2/2)$ is $c^2$ to leading order in $\mu$. The penalized objective is a concave parabola in the slope,

$$
-\ell'(0)\,\mu\, c - \big(|\ell''(0)| + \gamma\big)\, c^2 ,
$$

maximized at $c = -\ell'(0)\mu / \big(2(|\ell''(0)| + \gamma)\big)$ with the value

$$
\frac{\ell'(0)^2 \mu^2}{4\,(|\ell''(0)| + \gamma)} \;=\; \frac{\mu^2}{4 + 16\gamma}
\quad \text{for the logistic loss, at the slope } c = -\frac{\mu}{1 + 4\gamma} .
$$

The log ratio $\log(p/q)$ has slope $-\mu$. The penalty shrinks the critic's slope by the factor $1 + 4\gamma$, and the value with it. At $\gamma = 0$ the value is $\mu^2/4 = 2\,\mathrm{JS}(p, q)$ to leading order, the unpenalized value of :numref:`sec_basic_gan`. For large $\gamma$ it is $\mu^2/(16\gamma) = \tfrac{\ell'(0)^2}{4\gamma} W_2(p, q)^2$, since the Wasserstein-2 distance between two Gaussians that differ by a shift is $|\mu|$: a heavily penalized critic measures how far the mass has to move.

We check the formula against the full penalized problem, solved by Newton's method for a critic tabulated on a grid, with no expansion and no assumption on its shape. At $\mu = 0.2$ the formula predicts $0.01$ at $\gamma = 0$, $0.002$ at $\gamma = 1$ and $2.4 \cdot 10^{-4}$ at $\gamma = 10$. The larger shifts test where "near" ends.

```{.python .input #convergence-a-gaussian-shift}
%%tab pytorch, jax
def penalized_value(p, q, gamma, x, newton_steps=30):
    """Value of the logistic game minus gamma E_m[D'^2], on a grid."""
    dx, m = x[1] - x[0], (p + q) / 2
    diff = np.diff(np.eye(len(x)), axis=0) / dx       # D -> D'
    K = 2 * gamma * dx * diff.T @ (((m[1:] + m[:-1]) / 2)[:, None] * diff)
    D = np.zeros_like(x)
    for _ in range(newton_steps):                     # Newton ascent on D
        s = 1 / (1 + np.exp(-D))                      # sigma(D)
        grad = (p * (1 - s) - q * s) * dx - K @ D     # gradient of the value
        neg_hess = np.diag(2 * m * s * (1 - s) * dx) + K  # minus its Hessian
        D = D + np.linalg.solve(neg_hess, grad)
    value = -(p * np.logaddexp(0, -D) + q * np.logaddexp(0, D)).sum() * dx
    return value - D @ K @ D / 2 + 2 * np.log(2)       # minus 2 l(0)

x = np.linspace(-8, 8, 401)
for mu in (0.2, 1, 3):
    p, q = [np.exp(-(x - c) ** 2 / 2) / np.sqrt(2 * np.pi) for c in (0, mu)]
    for gamma in (0, 0.01, 1, 10):
        exact = penalized_value(p, q, gamma, x)
        local = mu ** 2 / (4 + 16 * gamma)
        large = mu ** 2 / (16 * gamma) if gamma else np.inf
        print(f'mu {mu:>3}, gamma {gamma:>4}: exact {exact:.3e}, local '
              f'{local:.3e} ({local / exact:.2f}x), large-gamma {large:.3e} '
              f'({large / exact:.2f}x)')
```

The printed ratios say three things. First, for a small shift the formula is right: at $\mu = 0.2$ it matches the exact value to within one percent at every $\gamma$. Its large-$\gamma$ form, though, holds only when the penalty dominates: at $\gamma = 0.01$ it gives $0.25$ against the exact $0.0096$, 26 times too large, while at $\gamma = 10$ it is within three percent. Second, a large shift breaks the formula. At $\mu = 3$ without a penalty it is 2.1 times too large: it grows like $\mu^2$, but the game's value can never exceed $2\log 2 \approx 1.39$, and the exact value, 1.05, is already bending toward that ceiling. Saturation is what ends the approximation. The penalty postpones it by keeping the critic small: at $\mu = 3$, $\gamma = 1$ the formula is again within 15%, and $\mu = 1$ lies in between. Third, at $\mu = 3$, $\gamma = 10$ the formula errs the other way, at 0.80 times the exact value, for a reason we meet below.

### The Local Value in General

Nothing in the calculation needed a Gaussian except the shape of the best critic. For general $p$ and $q$, write $\mathcal{L}_m D = -\nabla \cdot (m \nabla D)$, a weighted second derivative; integrating by parts, $\int m\,\|\nabla_x D\|^2 = \langle D, \mathcal{L}_m D\rangle$. For the quadratic approximation the best critic solves the linear equation $\big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big) D = \tfrac{\ell'(0)}{2}(p - q)$, and the value is

$$
\sup_{D} \Big\{ \ell'(0)\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 - \gamma \int m\, \| \nabla_x D \|^2 \Big\}
\;=\; \frac{\ell'(0)^2}{4}\, \big\langle p - q,\, \big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big)^{-1} (p - q) \big\rangle .
$$
:eqlabel:`eq_gan_local_value`

The proof is the parabola of the Gaussian shift taken along every direction of critics at once (exercise :exref:`ex_convergence-prove-eqref-eq-gan`). For the Gaussian shift, $p - q \approx -\mu\, x\, m(x)$ and, with $m$ the standard normal density to leading order, $\mathcal{L}_m x = x\, m(x)$, so the straight line solves the equation and no other critic does better. The pairing objective has $\textrm{Var}_m[D]$ in place of $\int m\, D^2$, which gives the same supremum once the critic's free constant is chosen best.

The value is a weighted squared size of $p - q$. As $\gamma \to 0$ it becomes $\tfrac{\ell'(0)^2}{4|\ell''(0)|}\int (p - q)^2/m$, which for the logistic loss is $\tfrac14 \int (p-q)^2/m$, the local form of $2\,\mathrm{JS}(p, q)$. When $\gamma$ is large, the penalty dominates and the value approaches

$$
\frac{\ell'(0)^2}{4\gamma}\, \big\| p - q \big\|^2_{\dot H^{-1}(m)},
\qquad
\|v\|_{\dot H^{-1}(m)} = \sup\Big\{ \langle v, D\rangle : \int m \|\nabla_x D\|^2 \leq 1 \Big\},
$$
:eqlabel:`eq_gan_sobolev`

the largest correlation with $p - q$ that a critic of unit mean squared slope under $m$ can reach; adding a constant to the critic changes neither the correlation nor the slope. This is a *dual Sobolev norm*. For the Gaussian shift it is $|\mu|$, the transport distance found above. Where $m$ is thin, a critic may rise steeply at little cost, so the norm grows when mass must cross such a region. Two bumps three units apart leave $m$ thin in the middle, which is why the formula, built on a single bump, is too small at $\mu = 3$, $\gamma = 10$.

### A Low-Pass Filter

A general $p - q$ has many length scales. Where $m$ is roughly constant, $\mathcal{L}_m$ multiplies a Fourier component $e^{i \omega x}$ by $m \omega^2$, so in :eqref:`eq_gan_local_value` the component with wavenumber $\omega$ enters with weight

$$
\frac{1}{|\ell''(0)| + \gamma \omega^2}
\qquad \text{instead of} \qquad \frac{1}{|\ell''(0)|}.
$$

Components whose length scale $1/\omega$ exceeds $\sqrt{\gamma / |\ell''(0)|}$, which is $2\sqrt\gamma$ for the logistic loss, keep nearly their unpenalized weight; shorter ones are damped by about the factor $|\ell''(0)|/(\gamma \omega^2)$. The penalty is a low-pass filter on $p - q$ with cutoff length $2\sqrt\gamma$: above the cutoff the game measures $2\,\mathrm{JS}$, below it the dual Sobolev norm. Because the critic's input gradient has units of one over length, $\gamma$ is a squared length, set by the scale of the data. R3GAN starts $\gamma$ at 0.05 on CIFAR-10 and at 150 on FFHQ faces at $256 \times 256$ pixels. Part of that factor of 3,000 is resolution, since squared distances between images grow with the number of pixels; the rest is tuned per dataset, and ImageNet at $32 \times 32$, the resolution of CIFAR-10, starts at 0.5.

The filter has one limit that matters more than its cutoff. It acts only where $m$ has mass. Across a gap in $m$ the penalty costs nothing, so a critic free to take any shape can jump there, saturate as in the Dirac-GAN, and leave a generator on the far side with no gradient. For two point masses, the best response of such a critic is worth $2\sup_t \ell(t)$ for every $\theta \neq 0$ and every $\gamma$, exactly as without the penalty (exercise :exref:`ex_convergence-penalized-best-response`). What keeps a gradient across a gap is that a network whose input gradient is small at the samples is usually smooth between them as well. That is an assumption about the critic, not a consequence of the penalty, and it is the same smoothness that gave a generator six units from the data a direction in :numref:`subsec_gan_objectives_separated_start`.

## The R3GAN Recipe

R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` puts the pieces together. Its critic minimizes the pairing loss of :numref:`sec_gan_relativistic` plus both zero-centered penalties, and its generator minimizes the non-saturating form of that loss. Huang et al. report that with this recipe a minimalist convolutional network trains without most of the stabilizers that earlier GANs needed; :numref:`sec_dcgan` tests part of that claim. The penalties damp the rotation, and the non-saturating generator keeps a gradient for rejected samples. Whether pairing also improves mode coverage is the question that :numref:`sec_gan_relativistic` left open, and the grid at the end of this section asks it again.

### The Loss in Code

The implementation keeps the losses and the penalties apart, and writes everything as a loss to minimize: the critic minimizes its loss, which is $-V_\ell$ for the classical game, plus $R_1 + R_2$. The pairing losses are `d2l.rpgan_loss_D` and `d2l.rpgan_loss_G` of :numref:`sec_gan_relativistic`, whose critic loss equals $\log 2$ at the equilibrium. `gan_loss_D` and `gan_loss_G` give the logistic critic loss and the non-saturating generator loss of :numref:`sec_basic_gan` with the same arguments, so a training loop can take either pair; `gan_loss_D` is the sum of two batch means, on the same scale as `update_D`, and equals $2\log 2$ at the equilibrium.

```{.python .input #convergence-the-loss-in-code}
%%tab pytorch
#@save
def gan_loss_D(critic, real, fake):
    """Logistic critic loss -E[log sigma(D(x))] - E[log sigma(-D(x'))]: the
    sum of two batch means, 2 log 2 at equilibrium, as update_D."""
    return F.softplus(-critic(real)).mean() + F.softplus(critic(fake)).mean()

#@save
def gan_loss_G(critic, real, fake):
    """Non-saturating generator loss -E[log sigma(D(x'))]; real is unused."""
    return F.softplus(-critic(fake)).mean()
```

```{.python .input #convergence-the-loss-in-code}
%%tab jax
#@save
def gan_loss_D(critic, real, fake):
    """Logistic critic loss -E[log sigma(D(x))] - E[log sigma(-D(x'))]: the
    sum of two batch means, 2 log 2 at equilibrium, as update_D."""
    return (jax.nn.softplus(-critic(real)).mean()
            + jax.nn.softplus(critic(fake)).mean())

#@save
def gan_loss_G(critic, real, fake):
    """Non-saturating generator loss -E[log sigma(D(x'))]; real is unused."""
    return jax.nn.softplus(-critic(fake)).mean()
```

The penalty function returns the per-sample squared input gradients on both batches, before the factor $\gamma/2$. It differentiates the *sum* of the critic's scores with respect to the input batch: each score depends only on its own row, so one backward pass gives every sample's input gradient (batch normalization, which mixes rows, would break this). The input gradient stays in the autodiff graph, so the optimizer can differentiate the penalty with respect to the critic's parameters. The samples are held constant. The second differentiation makes a penalized critic step cost roughly two backward passes.

:begin_tab:`pytorch`
`create_graph=True` keeps the graph of the input gradient for the second differentiation, and `detach` makes the samples constants.
:end_tab:

:begin_tab:`jax`
`jax.grad` of the summed scores gives the input gradients; differentiating the penalty with respect to the critic nests a second gradient around it, and `jax.lax.stop_gradient` makes the samples constants.
:end_tab:

```{.python .input #convergence-the-loss-in-code-2}
%%tab pytorch
#@save
def r1_r2_penalty(critic, real, fake):
    """Per-sample squared critic input gradients on real (R1) and fake (R2),
    before the gamma/2 scale."""
    def sq_grad_norm(x):
        x = x.detach().requires_grad_(True)
        grad, = torch.autograd.grad(critic(x).sum(), x, create_graph=True)
        return grad.reshape(x.shape[0], -1).pow(2).sum(dim=1)
    return sq_grad_norm(real), sq_grad_norm(fake)
```

```{.python .input #convergence-the-loss-in-code-2}
%%tab jax
#@save
def r1_r2_penalty(critic, real, fake):
    """Per-sample squared critic input gradients on real (R1) and fake (R2),
    before the gamma/2 scale."""
    def sq_grad_norm(x):
        # Rows are independent, so the gradient of the sum is per-sample
        grad = jax.grad(lambda x: critic(x).sum())(jax.lax.stop_gradient(x))
        return (grad.reshape(x.shape[0], -1) ** 2).sum(axis=1)
    return sq_grad_norm(real), sq_grad_norm(fake)
```

### The Training Steps

A training loop alternates one step of each player. `critic_step` takes one optimizer step on the critic loss plus $\gamma/2$ times both penalties, with the generated batch held fixed, and returns the loss and the batch means of the squared input gradients at real and generated samples ($R_1$ and $R_2$ before the factor $\gamma/2$), so that a loop can log the critic's steepness. `generator_step` then takes one step on the generator loss against the updated critic. The optional labels `y` serve the conditional critic of :numref:`sec_gan_conditional`.

```{.python .input #convergence-the-loss-in-code-1}
%%tab pytorch
#@save
def critic_step(net_D, opt_D, loss_D, real, fake, gamma, y=None):
    """One critic step on loss_D + gamma/2 (R1 + R2); return the loss and
    the batch means of R1 and R2."""
    critic = net_D if y is None else lambda x: net_D(x, y)
    fake = fake.detach()  # the generator is held fixed
    r1, r2 = r1_r2_penalty(critic, real, fake)
    loss = loss_D(critic, real, fake)
    opt_D.zero_grad()
    (loss + gamma / 2 * (r1 + r2).mean()).backward()
    opt_D.step()
    return loss.detach(), r1.mean().detach(), r2.mean().detach()

#@save
def generator_step(net_G, opt_G, loss_G, net_D, real, z, y=None):
    """One generator step on loss_G against the critic; return the loss."""
    critic = net_D if y is None else lambda x: net_D(x, y)
    loss = loss_G(critic, real, net_G(z) if y is None else net_G(z, y))
    opt_G.zero_grad()
    loss.backward()
    opt_G.step()
    return loss.detach()
```

```{.python .input #convergence-the-loss-in-code-1}
%%tab jax
#@save
@nnx.jit(static_argnums=2)
def critic_step(net_D, opt_D, loss_D, real, fake, gamma, y=None):
    """One critic step on loss_D + gamma/2 (R1 + R2); return the loss and
    the batch means of R1 and R2."""
    def objective(net_D):
        critic = net_D if y is None else lambda x: net_D(x, y)
        r1, r2 = r1_r2_penalty(critic, real, fake)
        loss = loss_D(critic, real, fake)
        return (loss + gamma / 2 * (r1 + r2).mean(),
                (loss, r1.mean(), r2.mean()))
    (_, logs), grads = nnx.value_and_grad(objective, has_aux=True)(net_D)
    opt_D.update(net_D, grads)
    return logs

#@save
@nnx.jit(static_argnums=2)
def generator_step(net_G, opt_G, loss_G, net_D, real, z, y=None):
    """One generator step on loss_G against the critic; return the loss."""
    def objective(net_G):
        critic = net_D if y is None else lambda x: net_D(x, y)
        return loss_G(critic, real, net_G(z) if y is None else net_G(z, y))
    loss, grads = nnx.value_and_grad(objective)(net_G)
    opt_G.update(net_G, grads)
    return loss
```

:begin_tab:`jax`
The loss functions are static arguments of the compiled steps, so each pair of losses compiles once, while $\gamma$ is traced and can change without recompiling.
:end_tab:

### Mode Coverage on 25 Gaussians
:label:`subsec_gan_25_gaussians`

Does the damping survive in networks? We test it on 25 Gaussians whose centers form a $5 \times 5$ grid with spacing 2, each with standard deviation $0.05$: 25 islands of mass with almost nothing between them. This is the gap case of the low-pass filter. Between the islands the penalty does not act, so any gradient that leads the generator to an empty mode comes from the smoothness of the critic network, not from the penalty.

We score 10,000 generated samples with four statistics. A sample is *on a mode* if it lies within three widths, $0.15$, of the nearest center, where a 2-D Gaussian puts $1 - e^{-9/2} \approx 98.9\%$ of its mass. A mode counts as *covered* if it holds at least a fifth of its fair share of the on-mode samples. We report the number of covered modes, the fraction of samples on a mode, their mean distance to the nearest center (about $0.06$ for a perfect fit), and the critic's steepness $\mathbb{E}_q\|\nabla_x D\|^2$ at the generated samples. Both networks are MLPs with two hidden layers of 256 units, built by `d2l.make_mlp` of :numref:`sec_basic_gan`; the generator maps a 64-dimensional latent vector to the plane, and the critic uses leaky ReLU and no normalization.

```{.python .input #convergence-mode-coverage-on-25-gaussians-1}
%%tab pytorch
def grid_centers(side=5, spacing=2.0):
    coords = (torch.arange(side, dtype=torch.float32) - (side - 1) / 2)
    gx, gy = torch.meshgrid(coords * spacing, coords * spacing, indexing='ij')
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

def sample_modes(centers, n, width=0.05):
    idx = torch.randint(0, centers.shape[0], (n,), device=centers.device)
    return centers[idx] + width * torch.randn(n, 2, device=centers.device)
```

```{.python .input #convergence-mode-coverage-on-25-gaussians-1}
%%tab jax
def grid_centers(side=5, spacing=2.0):
    coords = (jnp.arange(side) - (side - 1) / 2) * spacing
    gx, gy = jnp.meshgrid(coords, coords, indexing='ij')
    return jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

def sample_modes(key, centers, n, width=0.05):
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (n,), 0, centers.shape[0])
    return centers[idx] + width * jax.random.normal(k2, (n, 2))
```

The function `mode_stats` computes the three sample statistics.

```{.python .input #convergence-mode-coverage-on-25-gaussians-5}
%%tab pytorch, jax
def mode_stats(x, centers, width=0.05):
    """Modes covered (each holds a fifth of an even share of the on-mode
    samples), fraction within three widths, mean distance."""
    dist = np.linalg.norm(x[:, None] - centers[None], axis=-1)
    near, idx = dist.min(axis=1), dist.argmin(axis=1)
    on = near <= 3 * width
    counts = np.bincount(idx[on], minlength=len(centers))
    covered = (counts >= on.sum() / len(centers) / 5).sum()
    return covered, on.mean(), near.mean()
```

The training function takes the two losses and $\gamma$ as arguments and alternates `critic_step` and `generator_step`, each on a fresh batch, for 10,000 steps with batches of 1,024. Both networks use Adam with a learning rate that decays from $2 \cdot 10^{-3}$ to $2 \cdot 10^{-4}$ along a cosine, and with $\beta_1 = 0$, as R3GAN does. The reason is the rotation: momentum averages the field over past steps, and when the field turns, the average points where it used to point, which on the underdamped Dirac-GAN shrinks the range of stable step sizes (exercise :exref:`ex_convergence-momentum-rotating-game`). `train_grid` returns the generator and a trace with eight checkpoints, one every 1,250 steps: the on-mode fraction of 2,000 samples, and the critic loss and $\mathbb{E}_q\|\nabla_x D\|^2$ averaged over the steps since the previous checkpoint.

```{.python .input #convergence-mode-coverage-on-25-gaussians-3}
%%tab pytorch
def train_grid(centers, loss_D, loss_G, gamma, steps=10000, lr=2e-3,
               batch=1024, latent=64, seed=0, checkpoints=8):
    """Alternate critic and generator steps; at each checkpoint record the
    on-mode fraction of 2,000 samples, and the critic loss and E_q|grad D|^2
    averaged over the steps since the previous checkpoint."""
    torch.manual_seed(seed)
    dev, grid = centers.device, centers.cpu().numpy()
    net_G = d2l.make_mlp([latent, 256, 256, 2], nn.ReLU()).to(dev)
    net_D = d2l.make_mlp([2, 256, 256, 1], nn.LeakyReLU(0.2)).to(dev)
    opt_G, opt_D = [torch.optim.Adam(net.parameters(),  # lr on the device
                                     torch.tensor(lr, device=dev),
                                     betas=(0.0, 0.99),
                                     capturable=dev.type == 'cuda')
                    for net in (net_G, net_D)]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr / 10)
              for opt in (opt_G, opt_D)]
    logs, every = torch.zeros(2, device=dev), steps // checkpoints

    def step():
        real = sample_modes(centers, batch)
        with torch.no_grad():
            fake = net_G(torch.randn(batch, latent, device=dev))
        loss, _, grad_q = critic_step(net_D, opt_D, loss_D, real, fake, gamma)
        logs.add_(torch.stack([loss, grad_q]))  # summed on the device
        z = torch.randn(batch, latent, device=dev)
        generator_step(net_G, opt_G, loss_G, net_D,
                       sample_modes(centers, batch), z)

    step, trace = d2l.cuda_graphed(step, dev), []
    for t in range(1, steps + 1):
        step()
        for sched in scheds:
            sched.step()
        if t % every == 0:                   # checkpoint
            with torch.no_grad():
                x = net_G(torch.randn(2000, latent, device=dev)).cpu().numpy()
            trace.append((mode_stats(x, grid)[1], *(logs / every).tolist()))
            logs.zero_()
    return net_G, trace
```

```{.python .input #convergence-mode-coverage-on-25-gaussians-3}
%%tab jax
def train_grid(centers, loss_D, loss_G, gamma, steps=10000, lr=2e-3,
               batch=1024, latent=64, seed=0, checkpoints=8):
    """Alternate critic and generator steps; at each checkpoint record the
    on-mode fraction of 2,000 samples, and the critic loss and E_q|grad D|^2
    averaged over the steps since the previous checkpoint."""
    rngs = nnx.Rngs(seed)
    net_G = d2l.make_mlp([latent, 256, 256, 2], nnx.relu, rngs=rngs)
    net_D = d2l.make_mlp([2, 256, 256, 1], lambda x: nnx.leaky_relu(x, 0.2),
                         rngs=rngs)
    adam = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.1),
                      b1=0.0, b2=0.99)
    opt_G, opt_D = (nnx.Optimizer(net, adam, wrt=nnx.Param)
                    for net in (net_G, net_D))

    @nnx.jit
    def step(net_G, net_D, opt_G, opt_D, key, logs):
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1, (2, batch, latent))    # critic, generator
        real = sample_modes(k2, centers, 2 * batch).reshape(2, batch, 2)
        fake = net_G(z[0])
        loss, _, grad_q = critic_step(net_D, opt_D, loss_D, real[0], fake,
                                      gamma)
        generator_step(net_G, opt_G, loss_G, net_D, real[1], z[1])
        return logs + jnp.stack([loss, grad_q])          # summed on the device

    step = nnx.cached_partial(step, net_G, net_D, opt_G, opt_D)
    trace, every, grid = [], steps // checkpoints, np.asarray(centers)
    keys = jax.random.split(rngs(), steps).reshape(checkpoints, every)
    for chunk in keys:                                   # one checkpoint each
        logs = jnp.zeros(2)
        for key in chunk:
            logs = step(key, logs)
        x = np.asarray(net_G(jax.random.normal(rngs(), (2000, latent))))
        trace.append((mode_stats(x, grid)[1], *(logs / every).tolist()))
    return net_G, trace
```

<!-- WRITER: S2 done in code: the CUDA-graph helper moved to the library as `d2l.cuda_graphed(step, device)` (convergence-lib.md, eager off CUDA); cut this note to one sentence, e.g. that `d2l.cuda_graphed` replays each step as one CUDA graph because issuing its few hundred short kernels from Python takes longer than running them (:numref:`subsec_comp-overhead`), which is why the optimizers keep their state on the device (`capturable`). -->
:begin_tab:`pytorch`
One step of these small networks launches a few hundred short kernels, and issuing them from Python takes longer than running them (:numref:`subsec_comp-overhead`). `graphed` records the step as a CUDA graph after three eager warm-up steps and replays it with one launch per step. Replay reuses the recorded memory addresses, so the optimizers keep their learning rate and step count on the GPU (`capturable=True`, with the learning rate passed as a tensor), and the scheduler updates the learning rate in place between replays.
:end_tab:

:begin_tab:`jax`
One compiled function draws the batches and calls both saved steps. `nnx.cached_partial` binds the networks and optimizers to it once, so the loop does not split and merge the model state at every step.
:end_tab:

The experiment separates the two ingredients of the recipe. It crosses two losses, the logistic critic with the non-saturating generator ("NS") and the pairing loss, with two penalties, none and $R_1 + R_2$ at $\gamma = 1$. Each arm trains with seeds 0, 1 and 2, and a seed gives the same initialization in every arm. One further run asks whether more friction helps: the pairing loss with both penalties at $\gamma = 10$. What should we expect? Under alternating steps the slope $\psi$ of the unpenalized Dirac-GAN neither grew nor shrank; it stayed on its band. If that carries over, the unpenalized critics should stay steep at the generated samples, and the penalized ones flat.

```{.python .input #convergence-mode-coverage-on-25-gaussians-4}
%%tab pytorch
centers, latent = grid_centers().to(d2l.try_gpu()), 64
ns, pairing = (gan_loss_D, gan_loss_G), (d2l.rpgan_loss_D, d2l.rpgan_loss_G)
arms = {'NS': (*ns, 0.0, 3), 'NS + R1 + R2': (*ns, 1.0, 3),
        'pairing': (*pairing, 0.0, 3), 'pairing + R1 + R2': (*pairing, 1.0, 3),
        'pairing + R1 + R2, gamma=10': (*pairing, 10.0, 1)}
runs = {name: [] for name in arms}               # per run: samples, trace
for name, (loss_D, loss_G, gamma, n_runs) in arms.items():
    for seed in range(n_runs):
        net_G, trace = train_grid(centers, loss_D, loss_G, gamma,
                                  latent=latent, seed=seed)
        with torch.no_grad():
            z = torch.randn(10000, latent, device=centers.device)
            runs[name].append((net_G(z).cpu().numpy(), trace))
grid = centers.cpu().numpy()
```

```{.python .input #convergence-mode-coverage-on-25-gaussians-4}
%%tab jax
centers, latent = grid_centers(), 64
ns, pairing = (gan_loss_D, gan_loss_G), (d2l.rpgan_loss_D, d2l.rpgan_loss_G)
arms = {'NS': (*ns, 0.0, 3), 'NS + R1 + R2': (*ns, 1.0, 3),
        'pairing': (*pairing, 0.0, 3), 'pairing + R1 + R2': (*pairing, 1.0, 3),
        'pairing + R1 + R2, gamma=10': (*pairing, 10.0, 1)}
runs, key = {name: [] for name in arms}, jax.random.key(1)  # samples, trace
for name, (loss_D, loss_G, gamma, n_runs) in arms.items():
    for seed in range(n_runs):
        net_G, trace = train_grid(centers, loss_D, loss_G, gamma,
                                  latent=latent, seed=seed)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (10000, latent))
        runs[name].append((np.asarray(net_G(z)), trace))
grid = np.asarray(centers)
```

Each run gets one row: the covered modes, the on-mode fraction and the mean distance for 10,000 samples of the trained generator, and the critic's steepness averaged over its last 1,250 steps.

```{.python .input #convergence-mode-coverage-on-25-gaussians-2}
%%tab pytorch, jax
print(f'{"arm":28} run  modes  on-mode  distance  E_q|grad D|^2')
for name, results in runs.items():
    for seed, (x, trace) in enumerate(results):
        modes, on, dist = mode_stats(x, grid)
        print(f'{name:28} {seed:3}  {modes:2}/25  {on:7.2f}  {dist:8.2f}'
              f'  {trace[-1][2]:13.1e}')
```

Numbers alone hide where the off-mode samples go, so we also draw run 0 of each of the four arms: 3,000 generated samples in orange, the 25 centers as blue crosses.

```{.python .input #convergence-mode-coverage-on-25-gaussians-7}
%%tab pytorch, jax
fig, axes = d2l.plt.subplots(1, 4, figsize=(10, 2.9))
for ax, name in zip(axes, list(runs)[:4]):
    x = runs[name][0][0][:3000]               # run 0; centers drawn last
    d2l.plot([x[:, 0], grid[:, 0]], [x[:, 1], grid[:, 1]],
             fmts=('C1.', 'C0x'), xlim=(-5, 5), ylim=(-5, 5), axes=ax)
    ax.set_title(name)
    ax.set_aspect('equal')
fig.tight_layout()
```

The prediction about the critic holds by a wide margin. With the penalty, $\mathbb{E}_q\|\nabla_x D\|^2$ ends below $0.003$ in every run; without it, between about 0.7 and 17. The unpenalized critic stays steep at the generated samples, like the undamped slope $\psi$ of the Dirac-GAN. Coverage follows the penalty, not the loss. Every run at $\gamma = 1$ covers 24 or 25 modes (all 25 in the runs printed here), while the unpenalized runs range from about a dozen modes to all 25: without the penalty, coverage is a matter of luck. Pairing makes no difference beyond this run-to-run range, with or without the penalty: the coverage benefit that R3GAN measured on StackedMNIST is not visible in our runs. Nor does the penalty make the fit sharper. In all four arms 64% to 94% of the samples lie on a mode, against 98.9% for a perfect fit, at a mean distance of 0.08 to 0.24 from the nearest center, against about 0.06. The scatter plots show where the rest go: mostly on lines between neighbouring centers, with a few strays elsewhere. A continuous generator cannot jump from one island to the next, so it draws the bridges. In the unpenalized NS panel some interior modes are nearly empty.

At $\gamma = 10$ the fit is much looser: only 19% to 28% of the samples lie on a mode, at a mean distance two to five times that at $\gamma = 1$, though 23 to 25 modes still hold some mass. The larger penalty flattens the critic further, to about $10^{-4}$, and the fit is slower. More friction is not always better, and the section has offered two reasons why: the overdamped spring approaches its equilibrium more slowly, at a rate near $\ell'(0)^2/\gamma$, and the low-pass filter damps short length scales more. Both give the direction of the effect, neither its size, since both assumed a generator near the data and the filter also a slowly varying $m$ (exercise :exref:`ex_convergence-penalty-length`).

Does the unpenalized game circle, as the Dirac-GAN does? The trace prints, for run 0 of every arm, the on-mode fraction and the critic loss at each checkpoint. The critic loss equals $2\log 2 \approx 1.39$ at the equilibrium for NS and $\log 2 \approx 0.69$ for pairing. Each value averages the critic loss over 1,250 steps, so the trace can show slow swings but not fast ones.

```{.python .input #convergence-mode-coverage-on-25-gaussians-6}
%%tab pytorch, jax
traces = {name: np.array(results[0][1]) for name, results in runs.items()}
for col, stat in enumerate(['on-mode fraction', 'critic loss']):
    print(f'{stat} of run 0 at every 1,250 steps')
    for name, trace in traces.items():         # trace: (checkpoint, stat)
        print(f'  {name:28}', ' '.join(f'{v:4.2f}' for v in trace[:, col]))
```

The on-mode fraction rises over training in every arm, with at most a dip of a hundredth, and no slow swing is visible. The critic losses tell the difference between the arms. Without the penalty the critic wins: its loss falls below the equilibrium value, from about 1.35 to about 1.0 for NS and from 0.68 to about 0.6 for pairing, and stays there. With the penalty it sits at the equilibrium value throughout, 1.36 to 1.38 and 0.67 to 0.69, even at 2,500 steps, when 3% or less of the samples lie on a mode. A critic loss at its equilibrium value therefore certifies nothing about the samples: a flat critic cannot separate $p$ from $q$, whether or not they agree. Keep this in mind whenever a GAN's loss curve looks reassuringly calm.

We can now return to the garment generator of :numref:`subsec_gan_fashion_mnist`. Its critic pulled ahead in the first epoch and then drifted slowly back toward $2\log 2$, ending at about 1.05. The unpenalized NS critics on the grid take much the same course: they fall below $2\log 2$ early, then level off or creep back, and end at about 1.0. On the grid that course goes with samples that improve steadily. So the unsettled garment losses most likely meant slow progress against a critic that stayed ahead and steep, not a visible circle. Whether the parameters also rotate on a faster time scale, as on two points, averages over many steps cannot tell.

On the grid the penalty flattened the critic and made coverage reliable. The rotation of the Dirac-GAN did not show at the resolution of these runs, and the samples were no sharper with the penalty than without it. Two dimensions say little about how either effect scales. Whether this recipe trains an image generator, and how to score the images it produces, is the question of :numref:`sec_dcgan`.

## Summary

Gradient training of an adversarial game need not reach its equilibrium. On the Dirac-GAN, a bilinear game in disguise, the flow circles the solution, every simultaneous step moves away from it, and alternating steps stay in a band around it, for every increasing loss and with or without pairing; the non-saturating generator adds only a slow inward drift, which discrete steps stop on a cycle of radius about $\sqrt{2h}$. The zero-centered penalties $R_1$ and $R_2$ on the critic's input gradient at data and generated samples damp the rotation for every $\gamma > 0$, like friction on a spring, and a local theorem extends this to networks under conditions that include a generator that reproduces the data exactly. Near equilibrium the penalized game measures $p - q$ through a low-pass filter with cutoff length $2\sqrt\gamma$, but only where the samples are; across a gap the gradient comes from the critic's smoothness. On 25 Gaussians the penalties keep the critic flat at the generated samples and make every run at $\gamma = 1$ cover 24 or 25 of the 25 modes, while pairing shows no effect beyond the spread between runs, no arm fits the modes sharply, and a penalized critic's loss sits at its equilibrium value whatever the samples look like.

## Exercises

1. [code] **Discrete-time stability.** A simultaneous step of size $h$
   maps $(\theta, \psi)$ to $(\theta, \psi) + h\, \mathbf{u}(\theta, \psi)$,
   with $\mathbf{u}$ from :eqref:`eq_gan_dirac_flow` plus $-\gamma\psi$ in
   its second component for a penalized critic.
    1. Show that at $\gamma = 0$ the Jacobian of the map at the
       equilibrium has eigenvalues $1 \pm i\, h\, \ell'(0)$, so no step
       size makes the unpenalized game locally convergent.
    1. For $\gamma > 0$ the eigenvalues are $1 + h\nu$, with $\nu$
       from :eqref:`eq_gan_dirac_pen`. Show that in the underdamped regime
       $\gamma < 2\ell'(0)$ the iteration converges locally if and only if
       $h < \gamma / \ell'(0)^2$.
    1. Check the threshold of (b) at $\gamma = 0.3$: run
       `trajectory(euler_step, (0.1, 0.1), h, 3000, gamma=0.3)` for $h = 1.1$ and
       $h = 1.3$ and plot the distance from the equilibrium. What does the
       iteration do above the threshold?
   :exlabel:`ex_convergence-discrete-simultaneous-update-dirac`
1. **Penalized best response.** Let the critic of the Dirac-GAN subtract
   both penalties of :eqref:`eq_gan_r1r2`, which for the linear critic sum
   to $\gamma\psi^2$, take $\ell = \log\sigma$, and write
   $V_\gamma(\theta) = \sup_\psi \big\{ V(\theta, \psi) - \gamma\psi^2 \big\}$.
    1. Show that the supremum is attained for every $\theta$, and that for
       small $\theta$ the maximizer is
       $\psi^\star \approx -\ell'(0)\,\theta / (2\gamma)$ and
       $V_\gamma(\theta) = 2\ell(0) + \tfrac{\ell'(0)^2}{4\gamma}\, \theta^2 + O(\theta^4)$.
    1. Compare (a) with :eqref:`eq_gan_sobolev`, using
       $W_2(\delta_0, \delta_\theta) = |\theta|$. Why does the curvature
       term of :eqref:`eq_gan_local_value` not appear at order $\theta^2$?
       What gradient does the generator receive near $\theta = 0$, and how
       does it depend on $\gamma$?
    1. Now let the critic be any differentiable function, penalized by
       $\gamma \int m\, \|\nabla_x D\|^2$. Prove the claim of the text that
       the penalized best-response value is $2\sup_t \ell(t)$ for every
       $\theta \neq 0$. Which assumption behind :eqref:`eq_gan_local_value`
       fails here, and which property of a trained network restores a
       gradient?
   :exlabel:`ex_convergence-penalized-best-response`
1. **Penalizing one side only.** Let $p = \delta_0$, put the generated
   point at $x' = 1$, take $\ell = \log\sigma$, and consider the critics
   $D_c(x) = -c\, x^2$ with $c \geq 0$, flat at the data and steep at the
   generated point.
    1. Evaluate $R_1$ and $R_2$ for $D_c$. Show that the critic objective
       $V_\ell(D_c) - R_1$ increases with $c$ and has no maximizer.
    1. Show that the gradient of the non-saturating generator loss
       $-\log\sigma(D(x'))$ with respect to $x'$ at $x' = 1$ grows without
       bound in $c$. What does the saturating loss $\log\sigma(-D(x'))$
       give instead? Which observation of
       :citet:`Huang.Gokaslan.Kuleshov.ea.2024` does this reproduce?
    1. Subtract $R_1 + R_2$ instead. Show that the maximizing $c^\star$ is
       finite with $c^\star < 1/(8\gamma)$, and bound the generator's
       gradient at $c^\star$.
   :exlabel:`ex_convergence-one-sided-penalty`
1. [code] **The penalty weight as a length.** In
   :numref:`subsec_gan_25_gaussians`, the pairing loss with the penalties
   at $\gamma = 10$ puts far fewer samples on a mode than at $\gamma = 1$.
    1. For data scaled by $c > 0$ and the critic $D_c(x) = D(x / c)$, show
       that the penalized game at weight $c^2\gamma$ has the same value as
       the original game at weight $\gamma$.
    1. With the low-pass weight and $|\ell''(0)| = \tfrac14$, compute by
       what factor the penalty damps a component whose wavelength is the
       grid spacing 2, and one whose wavelength is the mode width $0.05$,
       at $\gamma = 1$ and at $\gamma = 10$. The $\gamma = 1$ runs put most
       samples within three widths of a center nonetheless. Which
       hypotheses of the Fourier argument does the grid violate?
    1. Train the $\gamma = 10$ run for two and four times as many steps,
       with the cosine schedule stretched to match. Does the on-mode
       fraction approach that of $\gamma = 1$?
   :exlabel:`ex_convergence-penalty-length`
1. [code] **Alternating updates.** ● `train_grid` updates the critic first
   and then the generator against the updated critic. On the Dirac-GAN
   this reads

   $$\psi_{k+1} = \psi_k + h\, u_\psi(\theta_k, \psi_k), \qquad \theta_{k+1} = \theta_k + h\, u_\theta(\theta_k, \psi_{k+1}),$$

   with $(u_\theta, u_\psi)$ the field of :eqref:`eq_gan_dirac_flow`.
    1. Linearize at the equilibrium for $\gamma = 0$ and show that the
       Jacobian of the combined update has determinant one, with
       eigenvalues on the unit circle if and only if $h\, \ell'(0) \leq 2$.
       What does this imply about linear convergence and divergence?
    1. Run the alternating update at $\gamma = 0$,
       `trajectory(alt_step, (0.1, 0.1), h, 10**5)`, with
       $h \in \{0.2, 1, 3, 5\}$. Record the smallest and largest distance
       from the equilibrium over the last $10^4$ steps, and explain the
       wide spread at $h = 3$. Repeat at $h = 0.2$ with $\gamma = 0.3$.
   :exlabel:`ex_convergence-alternating-updates`
1. [code] **The local value.** ● Write
   $\mathcal{A} = |\ell''(0)|\, m + \gamma \mathcal{L}_m$ with
   $\mathcal{L}_m D = -\nabla \cdot (m \nabla D)$.
    1. Integrate by parts, assuming that boundary terms vanish, to show
       that $|\ell''(0)| \int m\, D^2 + \gamma \int m\, \|\nabla_x D\|^2 = \langle D, \mathcal{A} D\rangle$,
       and that $\mathcal{A}$ is symmetric and positive.
    1. Along a ray $\{t D_0 : t \in \mathbb{R}\}$ the objective of
       :eqref:`eq_gan_local_value` is a concave parabola in $t$. Find its
       maximum, then maximize over directions $D_0$ with the
       Cauchy--Schwarz inequality in the inner product
       $\langle \cdot, \mathcal{A}\, \cdot \rangle$ to prove
       :eqref:`eq_gan_local_value`, and show that the maximizer solves
       $\mathcal{A} D = \tfrac{\ell'(0)}{2}(p - q)$.
    1. Let $p = \mathcal{N}(0, 1)$ and $q = p\,(1 + 0.1 \sin \omega x)$. Use
       `penalized_value` to compute the ratio of the values at $\gamma$
       and at $\gamma = 0$ for $\omega \in \{0.5, 2, 8\}$ and
       $\gamma \in \{0.01, 0.1, 1\}$, and compare it with the low-pass
       weight $1/(1 + 4\gamma \omega^2)$. For which $\omega$ does the
       prediction fail, and which assumption of the Fourier argument breaks
       there?
   :exlabel:`ex_convergence-prove-eqref-eq-gan`
1. [code] **One-centered penalty dynamics.** ● Take the linear score
   $\ell(t) = t$ of the Wasserstein critic, so $\ell'(0) = 1$. On the
   Dirac-GAN the one-centered penalty becomes
   $\tfrac{\gamma}{2}(|\psi| - 1)^2$, which the critic subtracts.
    1. Show that the simultaneous gradient field vanishes at no point
       with $\psi \neq 0$, and find where its one-sided limits as
       $\psi \to 0^\pm$ vanish. Where is the generator there?
    1. On $\psi > 0$ the field is affine. Compare its Jacobian with
       $J_\gamma$ of :eqref:`eq_gan_dirac_pen`; for which $\gamma$ are its
       eigenvalues real?
    1. Modify `dirac_field` and run `trajectory` from $(1, 1)$ with
       $h = 0.01$ for 40,000 steps at $\gamma = 1$ and $\gamma = 3$.
       Confirm that at $\gamma = 3$ the generated point comes to rest at
       $\theta = 3$, and describe what happens at $\gamma = 1$. Explain
       both outcomes with (a) and (b).
   :exlabel:`ex_convergence-one-centered-dynamics`
1. [code] **Momentum in a rotating game.** ● `train_grid`, like R3GAN,
   trains with Adam at $\beta_1 = 0$. To isolate the first moment, average
   the field of the penalized Dirac-GAN with the logistic loss,
   $\bar{\mathbf{u}}_k = \beta\, \bar{\mathbf{u}}_{k-1} + (1 - \beta)\, \mathbf{u}(\mathbf{x}_k)$
   and $\mathbf{x}_{k+1} = \mathbf{x}_k + h\, \bar{\mathbf{u}}_k$, with
   $\mathbf{x}_k = (\theta_k, \psi_k)$ and $\bar{\mathbf{u}}_{-1} = \mathbf{0}$.
   Linearize at the equilibrium, write one step as a linear map of
   $(\mathbf{x}_k, \bar{\mathbf{u}}_{k-1}) \in \mathbb{R}^4$, and compute
   numerically the largest stable step size for $\beta \in \{0, 0.5, 0.9\}$
   at $\gamma = 0.3$ and $\gamma = 1$. Check $\beta = 0$ against problem
   :exref:`ex_convergence-discrete-simultaneous-update-dirac`b. In which
   damping regime does momentum shrink the range of stable step sizes?
   :exlabel:`ex_convergence-momentum-rotating-game`
1. [code] **Non-saturating Dirac dynamics.** ●● The non-saturating
   generator replaces $\ell'(-\psi\theta)$ by $\ell'(\psi\theta)$ in the
   first component of :eqref:`eq_gan_dirac_flow`. Take
   $\ell = \log\sigma$, $\gamma = 0$ and $r^2 = \theta^2 + \psi^2$.
    1. Show that
       $\tfrac{d}{dt} r^2 = 2\psi\theta\,\big(\ell'(\psi\theta) - \ell'(-\psi\theta)\big) \leq 0$
       for a concave $\ell$. Why can the approach not be linear?
    1. Average $\tfrac{d}{dt} r^2$ over one turn, at angular rate
       $\ell'(0)$, to leading order in $r$, and show that
       $r(t)^2 \approx 8/t$ for large $t$.
    1. One simultaneous step changes $r^2$ by
       $2h\, \langle \mathbf{x}, \mathbf{u} \rangle + h^2 \|\mathbf{u}\|^2$.
       Average this over one turn and show that it vanishes at
       $(r^\star)^2 = 2h$. Is this cycle stable?
    1. Redefine `dirac_field` with the non-saturating weight, run
       `trajectory` from $(1, 1)$ and from $(0.05, 0.05)$ for $4000/h$
       steps with $h \in \{0.01, 0.05, 0.2\}$, and compare the late
       distance with $r^\star$. Check (b) with `trajectory(rk4_step, ...)`.
   :exlabel:`ex_convergence-non-saturating-dynamics`

[Discussions](https://d2l.discourse.group/t/36811)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.4]{.kicker}

Gradient Penalties and Convergence<br>
**the Dirac-GAN · circles, spirals and bands · zero-centered penalties · the penalized value · R3GAN on 25 Gaussians**
:::
:::

::: {.slide title="Gradient Steps Double the Distance to a Solution That Exists"}
[The problem]{.kicker}

- Garment GAN: after twenty epochs neither loss has reached its
  equilibrium value; both are still drifting.
- Two-point game: data at $0$, generated point at $\theta$, critic
  $D(x) = \psi x$. From $(1, 1)$, 300 gradient steps of size 0.2 take the
  distance to the solution $(0, 0)$ from **1.41 to 2.87**.

::: {.d2l-note .warn}
Knowing where the equilibrium lies does not tell us whether training gets
there. What makes it converge?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- show that the two-point flow circles, simultaneous steps spiral out and
  alternating steps stay in a band;
- compute the eigenvalues of the penalized game and its three damping regimes;
- state what the local theorem for networks assumes;
- compute the penalized value of a Gaussian shift and read $\gamma$ as a
  squared length;
- read a 2 × 2 experiment that separates penalty from pairing.

**The idea:** penalize the critic's slope at real and generated samples.
**Running examples:** the two-point game; a 1-D Gaussian shift; 25 Gaussians.
:::

::: {.slide title="Values Came First; This Deck Is About Dynamics"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | asks |
|---|---|
| Generative Adversarial Networks | what a classifier of samples computes |
| Adversarial Objectives and Divergences | which loss gives which distance |
| Relativistic Objectives | what changes when the critic ranks pairs |
| **Gradient Penalties and Convergence** | **why training circles, and what stops it** |
| Adversarial Image Generation | which network, and how to score images |
| Conditional Generation | how to ask for a class |
| Adversarial Losses Beyond GANs | where a learned comparison remains |
:::

::: {.col .narrow}
- **Uses:** the log ratio $\log(p/q)$ as optimal critic; the weight
  $\sigma(D(x'))$; the pairing loss.
- **Skips:** global convergence; image training (next deck).
:::
:::
:::

::: {.slide title="On Two Points the Best Response Gives No Gradient"}
[The Dirac-GAN]{.kicker}

$p = \delta_0$, $q_\theta = \delta_\theta$, critic $D_\psi(x) = \psi x$, loss
$\ell = \log\sigma$ ($\ell'(0) = \tfrac12$):

$$V(\theta, \psi) = \ell(0) + \ell(-\psi\theta)$$

- Best response: $\psi\theta \to -\infty$, so $\sup_\psi V$ is the same for
  every $\theta \neq 0$.
- A flat value gives the generator no gradient, however close it is.
- Training does not compute best responses: both players take steps.
:::

::: {.slide title="The Gradient Flow Moves on Circles"}
[The Dirac-GAN]{.kicker}

$$\dot\theta = \psi\,\ell'(-\psi\theta), \qquad \dot\psi = -\theta\,\ell'(-\psi\theta)$$

- The field is a positive scalar times $(\psi, -\theta)$, orthogonal to
  $(\theta, \psi)$: $\tfrac{d}{dt}(\theta^2 + \psi^2) = 0$.
- Jacobian eigenvalues $\pm i\,\ell'(0)$: a **center**, a pure rotation.
- Linearized, $V \approx 2\ell(0) - \ell'(0)\theta\psi$: the bilinear game
  $\min_\theta\max_\psi \theta\psi$.

::: {.d2l-note}
The scalar is the saturating weight $\sigma(D(\theta))$. A factor common to
both players sets the speed, not the direction.
:::
:::

::: {.slide title="Every Simultaneous Step Moves Away from the Solution"}
[The Dirac-GAN]{.kicker}

Field $\mathbf{u} \perp (\theta, \psi)$, step size $h$; by Pythagoras

$$\|(\theta, \psi) + h\,\mathbf{u}\|^2 = \|(\theta, \psi)\|^2 + h^2\|\mathbf{u}\|^2$$

- The distance grows at every step, **for every** $h > 0$.
- Update map: eigenvalues $1 \pm i h\ell'(0)$, modulus
  $\sqrt{1 + h^2\ell'(0)^2} > 1$.
- A smaller step only slows the outward spiral.
:::

::: {.slide title="Alternating Steps Preserve Area: Bounded, Not Convergent"}
[The Dirac-GAN]{.kicker}

Critic first, then the generator against the new critic. Linearized with
$a = h\,\ell'(0)$:

$$\begin{pmatrix}\theta\\ \psi\end{pmatrix} \mapsto
\begin{pmatrix} 1 - a^2 & a \\ -a & 1 \end{pmatrix}
\begin{pmatrix}\theta\\ \psi\end{pmatrix}, \qquad \det = 1$$

- Each half is a shear; shears preserve area, so no convergence at a
  linear rate.
- For $a \leq 2$ both eigenvalues lie on the unit circle: no escape at a
  linear rate either.
- From $(1, 1)$ at $h = 0.2$: distance between 1.34 and 1.42 over the last
  1,000 of 2,000 steps.
:::

::: {.slide title="Simulation: Circle, Outward Spiral, Band, and Damping"}
[The Dirac-GAN]{.kicker}

@!convergence-phase-portraits-1
:::

::: {.slide title="Check: Do the Earlier Repairs Change the Two-Point Dynamics?"}
[Check]{.kicker}

The pairing loss scores the pair $(0, \theta)$ by $\ell\big(D(0) - D(\theta)\big)$;
the non-saturating generator reweights its own update. Does either change
the field near the equilibrium?

. . .

- Pairing: $\ell(-\psi\theta)$, the classical value minus the constant
  $\ell(0)$. Same field, same circles, same outward spiral.
- Non-saturating: the same Jacobian at the equilibrium; a slow inward drift
  that discrete steps stop on a cycle of radius about $\sqrt{2h}$.
- The cure must change the field near the equilibrium.
:::

::: {.slide title="Zero-Centered Penalties Charge the Critic's Slope"}
[Penalties]{.kicker}

$$R_1 = \tfrac{\gamma}{2}\,\mathbb{E}_{x \sim p}\|\nabla_x D\|^2, \quad
R_2 = \tfrac{\gamma}{2}\,\mathbb{E}_{x' \sim q}\|\nabla_{x'} D\|^2, \quad
R_1 + R_2 = \gamma\,\mathbb{E}_{m}\|\nabla_x D\|^2$$

- The critic maximizes $V - R_1 - R_2$; $m = (p + q)/2$; gradients with
  respect to the **input**.
- Friction on the critic: we know where it must end (flat at $q = p$),
  not where the generator must end.
- So the penalty vanishes at the equilibrium and leaves it in place.

[Roth et al., 2017: noise-weighted form; Mescheder et al., 2018: $R_1$, $R_2$]{.small}
:::

::: {.slide title="The Penalty Turns the Center into a Damped Spring"}
[Penalties]{.kicker}

On the two-point game $\nabla_x D = \psi$, so one penalty adds $-\gamma\psi$
to $\dot\psi$:

$$J_\gamma = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & -\gamma \end{pmatrix}, \qquad
\nu_{1,2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^2}{4} - \ell'(0)^2}$$

- Negative real parts for **every** $\gamma > 0$.
- $\gamma < 2\ell'(0)$: inward spiral; $\gamma = 2\ell'(0)$ ($= 1$, logistic):
  critical; larger: overdamped and slower (right panel of the simulation).
- More friction is not always better.
:::

::: {.slide title="For Networks, Convergence Is a Local Theorem with Strong Assumptions"}
[Beyond two parameters]{.kicker}

Classical loss with $R_1$, $R_2$ or a mixture: small simultaneous or
alternating steps converge at a linear rate **near** the equilibrium if

- the generator reproduces the data, and the critic is zero near the data;
- $\ell'(0) \neq 0$, $\ell''(0) < 0$;
- the critic detects every first-order departure from the data;
- the equilibria form smooth manifolds.

::: {.d2l-note}
Nothing about reaching the neighbourhood. For pairing, Huang et al. prove
a local result of the same kind.
:::

[Mescheder et al., 2018; Huang et al., 2024 for pairing]{.small}
:::

::: {.slide title="A Unit-Slope Penalty Parks the Generator Away from the Data"}
[Penalties]{.kicker}

| penalty | critic preferred at $q = p$ | two-point game |
|:---|:---|:---|
| one-centered (WGAN-GP), $\mathbb{E}(\|\nabla D\| - 1)^2$ | unit slope | does not reach the data |
| zero-centered, $\mathbb{E}_m\|\nabla D\|^2$ | flat | converges, every $\gamma > 0$ |

- Linear score $\ell(t) = t$, $\gamma = 3$: the generated point comes to
  rest at $\theta = 3$, not at $0$.
- There the critic's tilt against the generated point ($\theta$) cancels
  the penalty's push toward unit slope ($\gamma$); a flat critic leaves the
  generator still.
:::

::: {.slide title="R1 Alone Leaves the Critic Free Where Only Generated Samples Lie"}
[Penalties]{.kicker}

Data at $0$, generated point at $1$, critic $D_c(x) = -c\,x^2$:

- $R_1 = 0$ (flat at the data), so $V(D_c) - R_1$ grows with $c$: no
  maximizer, and the generator's gradient grows without bound.
- $R_2 = 2\gamma c^2$ caps $c$.

::: {.d2l-note}
Whether it matters depends on the problem: $R_1$ alone diverged on
StackedMNIST, yet StyleGAN2 trains faces with it.
:::

[Huang et al., 2024; Karras et al., 2020]{.small}
:::

::: {.slide title="Check: How Steep Is the Penalized Critic for a Gaussian Shift?"}
[The penalized value]{.kicker}

$p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\mu, 1)$, logistic loss, critic
$D(x) = c\,x$. Second order in $D$: maximize
$-\tfrac12\mu c - (\tfrac14 + \gamma)\,c^2$.

. . .

- Slope $c = -\mu/(1 + 4\gamma)$: the log ratio's slope $-\mu$, shrunk by
  $1 + 4\gamma$.
- Value $\mu^2/(4 + 16\gamma)$: $2\,\mathrm{JS}$ at $\gamma = 0$,
  $W_2^2/(16\gamma)$ for large $\gamma$.
- Against the exact value: within 1% at $\mu = 0.2$; 2.1× too large at
  $\mu = 3$, $\gamma = 0$, where the true value bends toward its ceiling
  $2\log 2$.
:::

::: {.slide title="The Penalty Is a Low-Pass Filter, but Only Where the Samples Are"}
[The penalized value]{.kicker}

A Fourier component $e^{i\omega x}$ of $p - q$ enters with weight
$1/(|\ell''(0)| + \gamma\omega^2)$ instead of $1/|\ell''(0)|$.

- Logistic loss: cutoff length $2\sqrt\gamma$; $\gamma$ is a **squared
  length** (R3GAN: 0.05 on CIFAR-10, 150 on 256-pixel faces; partly
  resolution, partly tuning).
- Across a gap in $m = (p+q)/2$ the penalty costs nothing: a free critic
  jumps and saturates.

::: {.d2l-note}
The gradient across a gap comes from the network's smoothness, not from
the penalty.
:::
:::

::: {.slide title="The R3GAN Loss Is Pairing Plus Both Penalties"}
[The R3GAN recipe]{.kicker}

Critic, on aligned real $x$ and generated $x'$:

$$\mathbb{E}\,\mathrm{softplus}\big(D(x') - D(x)\big)
+ \tfrac{\gamma}{2}\big(\mathbb{E}_p\|\nabla_x D\|^2 + \mathbb{E}_q\|\nabla_{x'} D\|^2\big)$$

Generator: $\mathbb{E}\,\mathrm{softplus}\big(D(x) - D(x')\big)$, the
non-saturating form.

- Penalty: gradient of $\sum_i D(x_i)$ with respect to the batch, per row;
  differentiable, samples detached.
- Adam with $\beta_1 = 0$: momentum averages a rotating field.
:::

::: {.slide title="Check: What Should the Penalty Change on 25 Gaussians?"}
[Check]{.kicker}

25 Gaussians on a $5 \times 5$ grid, spacing 2, width 0.05. Losses NS or
pairing; penalty none or $\gamma = 1$; three seeds each. *On a mode*: within
0.15 of a center; *covered*: at least a fifth of a fair share of the
on-mode samples. Predict the critic's steepness
$\mathbb{E}_q\|\nabla_x D\|^2$ at the generated samples.

. . .

- Without the penalty the critic should stay steep, like $\psi$ on its band
  under alternating steps.
- With it, the critic should stay flat.
:::

::: {.slide title="The Penalty Keeps the Critic Flat and Makes Coverage Reliable"}
[25 Gaussians]{.kicker}

@!convergence-mode-coverage-on-25-gaussians-7

- Penalized: $\mathbb{E}_q\|\nabla D\|^2 < 0.003$, 24–25 modes in every run.
  Unpenalized: 0.7 to 17, about a dozen to 25 modes.
- Pairing: no effect beyond the run-to-run range. Sharpness: no arm near a
  perfect fit.
- Off-mode samples sit on bridges between neighbouring centers: a
  continuous generator cannot jump between islands.
:::

::: {.slide title="A Critic Loss at Its Equilibrium Value Certifies Nothing"}
[25 Gaussians]{.kicker}

| run 0, critic loss | start | end | equilibrium |
|:--|--:|--:|--:|
| NS | 1.35 | about 1.0 | 1.39 |
| NS + $R_1$ + $R_2$ | 1.38 | 1.38 | 1.39 |
| pairing | 0.68 | about 0.6 | 0.69 |
| pairing + $R_1$ + $R_2$ | 0.68 | 0.69 | 0.69 |

- Unpenalized critics win and stay steep; penalized ones read the
  equilibrium value even when 3% or less of the samples lie on a mode.
- $\gamma = 10$: 19% to 28% on a mode, slower but not stuck.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Two-point game: the flow circles; simultaneous steps double the distance
  in 300 steps of 0.2; alternating steps stay within 1.34 to 1.42.
- Zero-centered penalties on the critic: negative real parts for every
  $\gamma > 0$; more friction is not always better.
- Networks: a local theorem that assumes a realizable generator.
- Gaussian shift: the penalty shrinks the critic's slope by $1 + 4\gamma$;
  filtering acts only where $m$ has mass.
- 25 Gaussians: penalty, not pairing, gives a flat critic and 24–25 modes;
  no arm fits the modes sharply.
- A critic loss at its equilibrium value certifies nothing about the samples.

**Next:** the same loss on images, and how to score what it generates.
:::

