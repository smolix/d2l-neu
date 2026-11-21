# Gradient Penalties and Convergence
:label:`sec_gan_convergence`

In the Fashion-MNIST experiment of :numref:`sec_basic_gan`, neither the discriminator loss nor the generator loss settled to its theoretical equilibrium value. Does alternating gradient descent converge slowly, or can it orbit an equilibrium indefinitely without reaching it? The population divergence analyses of :numref:`sec_basic_gan`–:numref:`sec_gan_relativistic` characterize the minimax objective when the critic is at its exact optimum $D^\star$, but in practice both networks take small simultaneous or alternating gradient steps, with each update shifting the other network's objective.

To analyze the dynamics of gradient training directly, :citet:`Mescheder.Geiger.Nowozin.2018` introduced the *Dirac-GAN*—a minimal two-parameter problem consisting of a point-mass data distribution, a point-mass generator, and a linear critic. Even on this two-parameter system, unregularized gradient descent fails to converge, whereas penalizing the squared norm of the critic's input gradient ($R_1$ and $R_2$) makes the equilibrium locally asymptotically stable. Combining these zero-centered gradient penalties with the relativistic pairing objective of :numref:`sec_gan_relativistic` yields the R3GAN recipe :cite:`Huang.Gokaslan.Kuleshov.ea.2024` used across the remainder of this chapter.

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

### A Two-Parameter Minimax System

Let the data distribution be a point mass at the origin, $p = \delta_0$, let the generator place a point mass at $\theta \in \mathbb{R}$, $q_\theta = \delta_\theta$ (so $q_\theta = p$ if and only if $\theta = 0$), and let the critic be linear without bias, $D_\psi(x) = \psi x$. Under the symmetric margin objective $V_\ell(D) = \mathbb{E}_{x \sim p}[\ell(D(x))] + \mathbb{E}_{x' \sim q}[\ell(-D(x'))]$ (:numref:`sec_gan_objectives`), both expectations reduce to single evaluations:

$$
V(\theta, \psi) \;=\; \ell\big(D_\psi(0)\big) + \ell\big({-D_\psi(\theta)}\big)
\;=\; \ell(0) + \ell(-\psi\theta).
$$
:eqlabel:`eq_gan_dirac_value`

We take the logistic payoff $\ell(t) = \log \sigma(t)$, which satisfies $\ell'(t) = \sigma(-t) > 0$ and $\ell''(t) = -\sigma(t)\sigma(-t) < 0$, with $\ell'(0) = \tfrac12$ and $\ell''(0) = -\tfrac14$.

For any $\theta \neq 0$, the supports of $p$ and $q_\theta$ are disjoint: maximizing $V(\theta, \psi)$ over $\psi$ sends $\psi\theta \to -\infty$, yielding the constant $\sup_\psi V(\theta, \psi) = \ell(0)$ for all $\theta \neq 0$, which provides zero gradient with respect to $\theta$.

### Undamped Oscillations in Continuous Time

Instead of solving for $\sup_\psi V(\theta, \psi)$, gradient training updates $\theta$ and $\psi$ incrementally from their current values. The unique stationary point of :eqref:`eq_gan_dirac_value` is the equilibrium $(\theta, \psi) = (0, 0)$. Under simultaneous continuous-time gradient flow ($\dot\theta = -\partial_\theta V$, $\dot\psi = +\partial_\psi V$),

$$
\dot\theta \;=\; -\partial_\theta V \;=\; \psi\, \ell'(-\psi\theta),
\qquad
\dot\psi \;=\; +\partial_\psi V \;=\; -\theta\, \ell'(-\psi\theta).
$$
:eqlabel:`eq_gan_dirac_flow`

Because the velocity vector $\mathbf{u}(\theta, \psi) = \ell'(-\psi\theta)\,(\psi, -\theta)$ is everywhere orthogonal to the state vector $(\theta, \psi)$, the squared distance to the origin is conserved:

$$
\frac{d}{dt}\big(\theta^2 + \psi^2\big)
= 2\theta\psi\,\ell'(-\psi\theta) - 2\psi\theta\,\ell'(-\psi\theta) = 0 .
$$

Every trajectory is a closed circle of constant radius around $(0, 0)$ for any strictly increasing payoff $\ell$: the generator parameter $\theta$ oscillates back and forth across $0$ without ever converging.

Linearizing :eqref:`eq_gan_dirac_flow` around $(\theta, \psi) = (0, 0)$ (:numref:`subsec_mdl-ode-linearization`) yields the Jacobian matrix and eigenvalues:

$$
J = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & 0 \end{pmatrix},
\qquad
\nu_{1,2} = \pm\, i\, \ell'(0).
$$
:eqlabel:`eq_gan_dirac_eigs`

Purely imaginary eigenvalues correspond to a linear *center* (the bilinear saddle problem $\min_\theta \max_\psi \theta\psi$, :citet:`Daskalakis.Ilyas.Syrgkanis.ea.2018`). Notice that the scalar factor $\ell'(-\psi\theta) = \sigma(D_\psi(\theta)) = w_{\textrm{sat}}$ multiplies both components simultaneously and therefore only rescales angular velocity along the circle.

### Outward Spirals and Limit Cycles in Discrete Time

In discrete time, simultaneous gradient descent with step size $h > 0$ updates $(\theta, \psi) \mapsto (\theta, \psi) + h\,\mathbf{u}(\theta, \psi)$. Because $\mathbf{u}(\theta, \psi) \perp (\theta, \psi)$, the Pythagorean theorem gives:

$$
\big\| (\theta, \psi) + h\, \mathbf{u} \big\|^2
\;=\; \big\| (\theta, \psi) \big\|^2 + h^2 \big\| \mathbf{u} \big\|^2 .
$$

Thus, every simultaneous discrete step strictly *increases* the distance from the equilibrium for any $h > 0$, causing the iterates to spiral outward (Exercise :exref:`ex_convergence-discrete-simultaneous-update-dirac`).

Alternating gradient updates (updating $\psi$ first, then $\theta$ using the updated $\psi$) fare slightly better: near $(0, 0)$, each half-step is an area-preserving linear shear, so the composite Jacobian has determinant $1$. For $h\,\ell'(0) \leq 2$, both eigenvalues lie on the unit circle :cite:`Mescheder.Geiger.Nowozin.2018`, so alternating iterates orbit within a bounded band around the origin without converging (Exercise :exref:`ex_convergence-alternating-updates`). Neither the non-saturating generator loss (which shares the same linearization $J$ at the origin) nor the relativistic pairing loss ($\ell(D_\psi(0) - D_\psi(\theta)) = \ell(-\psi\theta)$, which has the identical gradient field) alters these local eigenvalues.

## Zero-Centered Gradient Penalties

To make the equilibrium $(\theta, \psi) = (0, 0)$ locally asymptotically stable, we must shift the eigenvalues of the Jacobian from the imaginary axis into the open left half-plane $\operatorname{Re}(\nu) < 0$. Because the equilibrium critic must be flat ($\nabla_x D^\star(x) = 0$ when $q = p$), penalizing the squared norm of the critic's input gradient adds damping to the critic without shifting the equilibrium $q = p$.

:citet:`Mescheder.Geiger.Nowozin.2018` define the two *zero-centered gradient penalties* $R_1$ (evaluated on real data $x \sim p$) and $R_2$ (evaluated on generated samples $x' \sim q$):

$$
R_1 \;=\; \frac{\gamma}{2}\, \mathbb{E}_{x \sim p}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad
R_2 \;=\; \frac{\gamma}{2}\, \mathbb{E}_{x' \sim q}\Big[ \big\| \nabla_{x'} D(x') \big\|^2 \Big],
$$
:eqlabel:`eq_gan_r1r2`

with penalty weight $\gamma > 0$. The critic maximizes $V_\ell(D) - R_1 - R_2$, while the generator objective is unchanged. Writing $m = (p + q)/2$, their sum is the expected squared gradient norm under the mixture distribution:

$$
R_1 + R_2 \;=\; \gamma\, \mathbb{E}_{x \sim m}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad m = \frac{p + q}{2}.
$$
:eqlabel:`eq_gan_r1r2_sum`

### Damping the Dirac-GAN

For the linear critic $D_\psi(x) = \psi x$, we have $\nabla_x D_\psi(x) = \psi$ everywhere, so $R_1 = R_2 = \tfrac{\gamma}{2}\psi^2$. Subtracting $\tfrac{\gamma}{2}\psi^2$ from the critic objective adds a linear damping term $-\gamma\psi$ to $\dot\psi$ (or $-2\gamma\psi$ if both $R_1$ and $R_2$ are included):

$$
\dot\theta = \psi\,\ell'(-\psi\theta),
\qquad
\dot\psi = -\theta\,\ell'(-\psi\theta) - \gamma\psi .
$$

The Jacobian at $(0, 0)$ and its eigenvalues become:

$$
J_\gamma = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & -\gamma \end{pmatrix},
\qquad
\nu_{1,2} \;=\; -\frac{\gamma}{2} \pm \sqrt{ \frac{\gamma^2}{4} - \ell'(0)^2 } .
$$
:eqlabel:`eq_gan_dirac_pen`

For every $\gamma > 0$, both eigenvalues have strictly negative real part ($\operatorname{Re}(\nu_{1,2}) < 0$), so the equilibrium $(0, 0)$ is locally exponentially stable! The characteristic polynomial $\nu^2 + \gamma\nu + \ell'(0)^2 = 0$ is identical to that of a damped harmonic oscillator (:numref:`subsec_mdl-ode-linearization`) with three regimes:

1. *Underdamped* ($\gamma < 2\ell'(0)$): complex-conjugate eigenvalues with real part $-\gamma/2$; trajectories spiral inward (converging under simultaneous Euler steps when $h < \gamma / \ell'(0)^2$, Exercise :exref:`ex_convergence-discrete-simultaneous-update-dirac`).
2. *Critically damped* ($\gamma = 2\ell'(0)$, i.e., $\gamma = 1$ for the logistic loss): repeated real eigenvalue $\nu = -\ell'(0)$, yielding the fastest non-oscillatory decay.
3. *Overdamped* ($\gamma > 2\ell'(0)$): two negative real eigenvalues, where the slow mode $\nu \approx -\ell'(0)^2/\gamma$ decays more slowly as $\gamma$ increases.

### Phase Portraits

We verify these four regimes by integrating :eqref:`eq_gan_dirac_flow` from $(\theta_0, \psi_0) = (1, 1)$ (initial radius $\sqrt{2} \approx 1.41$) using fourth-order Runge–Kutta (`rk4_step`), simultaneous forward Euler (`euler_step`), and alternating gradient updates (`alt_step`).

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

The phase portraits confirm all four predictions: the continuous flow (`RK4`) orbits on the circle of radius $1.41$, simultaneous Euler steps spiral outward to radius $2.87$ after 300 steps, alternating steps orbit inside $[1.34, 1.42]$, and adding the zero-centered penalty drives simultaneous Euler steps directly to the origin ($6.5 \times 10^{-4}$ at underdamped $\gamma = 0.3$, and $7 \times 10^{-12}$ at critically damped $\gamma = 1$).

### Local Convergence for General Networks and Comparison of Penalties

:citet:`Mescheder.Geiger.Nowozin.2018` extended this local stability analysis from the two-parameter Dirac-GAN to general neural networks: near an equilibrium where the generator represents the data distribution $p$ and the critic is flat on $\operatorname{supp}(p)$, simultaneous and alternating gradient updates with $R_1$ or $R_2$ (and a strictly concave loss satisfying $\ell'(0) \neq 0, \ell''(0) < 0$) converge exponentially to the equilibrium manifold, provided the critic has sufficient local capacity to detect first-order generator perturbations (:citet:`Huang.Gokaslan.Kuleshov.ea.2024` establish an analogous local theorem for the relativistic pairing objective).

This analysis also explains two practical design choices:

- **Zero-centered ($R_1, R_2$) vs. one-centered (WGAN-GP) penalties:** WGAN-GP penalizes $(\|\nabla_x D\| - 1)^2$, forcing the critic to maintain unit slope even when $q = p$. On the Dirac-GAN with linear payoff $\ell(t) = t$, the one-centered penalty prevents convergence to $\theta = 0$, instead parking the generator at $|\theta| = \gamma$ where the Wasserstein gradient balances the penalty (Exercise :exref:`ex_convergence-one-centered-dynamics`).
- **Using both $R_1$ and $R_2$ vs. $R_1$ alone:** Near $q = p$, $R_1$ and $R_2$ are equivalent. Far from equilibrium, however, $R_1$ penalizes the critic's slope only at real data points $x \sim p$, leaving its slope at generated points $x' \sim q$ unconstrained (for example, $D_c(x) = -cx^2$ has zero slope at $p = \delta_0$ so $R_1 = 0$, allowing $c \to \infty$ and exploding generator gradients at $x' = 1$; Exercise :exref:`ex_convergence-one-sided-penalty`). Adding $R_2$ bounds the critic's gradient at generated samples as well.

## Effect of Gradient Penalties on the Value Function

How do the zero-centered penalties $R_1 + R_2 = \gamma\,\mathbb{E}_m[\|\nabla_x D\|^2]$ modify the divergence that the generator minimizes near equilibrium?

### Local Quadratic Approximation and Gaussian Shift

Near $q = p$, the optimal critic $D$ is small. Expanding $\ell(t) \approx \ell(0) + \ell'(0)t - \tfrac12|\ell''(0)|t^2$ to second order in $D$ inside $V_\ell(D)$ gives:

$$
V_\ell(D) \approx 2\ell(0) + \ell'(0)\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 ,
$$

where $\langle p - q, D \rangle = \mathbb{E}_p[D] - \mathbb{E}_q[D]$. Consider two unit-variance Gaussians $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$ separated by a small shift $\mu$, and a linear critic $D(x) = cx$. To leading order in $\mu$, $\langle p - q, D \rangle = -c\mu$, $\int m\,D^2 \approx c^2$, and the penalty is $\gamma\int m\,\|\nabla_x D\|^2 = \gamma c^2$. Maximizing the concave quadratic $-\ell'(0)\mu c - (|\ell''(0)| + \gamma)c^2$ over $c$ yields the optimal slope $c^\star = -\frac{\ell'(0)\mu}{2(|\ell''(0)| + \gamma)}$ and value:

$$
\frac{\ell'(0)^2 \mu^2}{4\,(|\ell''(0)| + \gamma)} \;=\; \frac{\mu^2}{4 + 16\gamma}
\quad \text{for the logistic loss, at slope } c^\star = -\frac{\mu}{1 + 4\gamma} .
$$

Thus, the penalty shrinks both the critic's slope and the local value by the factor $1 + 4\gamma$. At $\gamma = 0$, we recover $\mu^2/4 \approx 2\,\mathrm{JS}(p, q)$; for large $\gamma$, the value scales as $\mu^2/(16\gamma) = \tfrac{\ell'(0)^2}{4\gamma} W_2(p, q)^2$. Below, we verify this closed-form approximation against the exact penalized optimum computed via Newton's method on a 1D grid.

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

For small shifts ($\mu = 0.2$), the quadratic formula $\mu^2/(4 + 16\gamma)$ matches the exact numerical solution to within 1% across all $\gamma$.

### Dual Sobolev Norm and Low-Pass Filtering

For general distributions $p$ and $q$ near equilibrium, integrating by parts writes the gradient penalty as $\int m\,\|\nabla_x D\|^2 = \langle D, \mathcal{L}_m D \rangle$, where $\mathcal{L}_m D = -\nabla \cdot (m \nabla D)$ is the $m$-weighted Laplacian. Maximizing the local quadratic approximation over all functions $D$ yields (Exercise :exref:`ex_convergence-prove-eqref-eq-gan`):

$$
\sup_{D} \Big\{ \ell'(0)\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 - \gamma \int m\, \| \nabla_x D \|^2 \Big\}
\;=\; \frac{\ell'(0)^2}{4}\, \big\langle p - q,\, \big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big)^{-1} (p - q) \big\rangle .
$$
:eqlabel:`eq_gan_local_value`

As $\gamma \to 0$, this reduces to $\tfrac{\ell'(0)^2}{4|\ell''(0)|} \int (p - q)^2/m \approx 2\,\mathrm{JS}(p, q)$. As $\gamma \to \infty$, it approaches a multiple of the squared *dual Sobolev norm* $\dot H^{-1}(m)$:

$$
\frac{\ell'(0)^2}{4\gamma}\, \big\| p - q \big\|^2_{\dot H^{-1}(m)},
\qquad
\|v\|_{\dot H^{-1}(m)} = \sup\Big\{ \langle v, D\rangle : \int m \|\nabla_x D\|^2 \leq 1 \Big\}.
$$
:eqlabel:`eq_gan_sobolev`

In regions where $m$ is approximately uniform, $\mathcal{L}_m$ multiplies a Fourier mode of wavenumber $\omega$ by $m\omega^2$, weighting that mode by $(|\ell''(0)| + \gamma\omega^2)^{-1}$ instead of $|\ell''(0)|^{-1}$. Thus, $R_1 + R_2$ acts as a spatial *low-pass filter* on $p - q$ with cutoff length scale $\sqrt{\gamma / |\ell''(0)|}$ ($2\sqrt{\gamma}$ for the logistic loss). Because $\nabla_x D$ has units of inverse length, $\gamma$ has units of squared length and scales with data resolution and norm. Note also that $R_1 + R_2$ penalizes $\nabla_x D$ only on the support of $m = (p+q)/2$; across empty space between disjoint supports, the gradient still relies on the smoothness of the neural network critic (Exercise :exref:`ex_convergence-penalized-best-response`).

## The R3GAN Recipe

R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` combines the relativistic pairing loss (`d2l.rpgan_loss_D` and `d2l.rpgan_loss_G` from :numref:`sec_gan_relativistic`) with both zero-centered gradient penalties $R_1$ and $R_2$.

### Implementation of the Losses and Penalties

For comparison, we also save `gan_loss_D` and `gan_loss_G` implementing the classic logistic discriminator loss and non-saturating generator loss of :numref:`sec_basic_gan` with the same `(critic, real, fake)` signature.

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

### Mode Coverage on a 25-Gaussian Grid
:label:`subsec_gan_25_gaussians`

To test how $R_1 + R_2$ and pairwise ranking affect mode coverage in neural networks, we train a 3-layer MLP generator (64D latent, two hidden layers of 256 ReLU units) and a 3-layer leaky-ReLU MLP critic on a $5 \times 5$ grid of 2D Gaussians (spacing $2.0$, standard deviation $\sigma = 0.05$). We evaluate 10,000 generated samples by four metrics:

1. *Covered modes*: number of modes receiving at least $1/5$ of an equal share ($1/25$) of on-mode samples.
2. *On-mode fraction*: fraction of samples within $3\sigma = 0.15$ of the nearest mode center ($98.9\%$ for the true distribution).
3. *Mean distance* to the nearest center ($\approx 0.06$ for the true distribution).
4. *Critic steepness* $\mathbb{E}_q[\|\nabla_x D(x')\|^2]$ at generated samples.

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

`train_grid` alternates `critic_step` and `generator_step` for 10,000 steps (batch size 1,024) using Adam with $\beta_1 = 0$ (following R3GAN, since positive first-moment momentum in a rotating vector field can reduce the stability region) and cosine learning-rate decay from $2 \times 10^{-3}$ to $2 \times 10^{-4}$.

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

We run a $2 \times 2$ factorial experiment crossing the two objectives (classic non-saturating `NS` vs. `pairing`) with $\gamma = 0$ (unpenalized) and $\gamma = 1$ ($R_1 + R_2$), over three seeds each, plus an overdamped run at $\gamma = 10$.

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

```{.python .input #convergence-mode-coverage-on-25-gaussians-2}
%%tab pytorch, jax
print(f'{"arm":28} run  modes  on-mode  distance  E_q|grad D|^2')
for name, results in runs.items():
    for seed, (x, trace) in enumerate(results):
        modes, on, dist = mode_stats(x, grid)
        print(f'{name:28} {seed:3}  {modes:2}/25  {on:7.2f}  {dist:8.2f}'
              f'  {trace[-1][2]:13.1e}')
```

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

The results isolate the roles of the gradient penalties and the pairing loss:

- **Gradient penalties drive mode coverage:** Without penalties ($\gamma = 0$), critic gradients remain steep ($\mathbb{E}_q\|\nabla_x D\|^2 \in [0.7, 17]$) and mode coverage is unreliable (dropping to 12–25 modes depending on the seed). With $R_1 + R_2$ at $\gamma = 1$, critic gradients are damped by three orders of magnitude ($\sim 10^{-3}$) and every run reliably covers 24–25 modes under both `NS` and `pairing`.
- **Overdamping at large $\gamma$:** Increasing $\gamma$ to $10$ over-flattens the critic ($\sim 10^{-4}$), slowing convergence so that only ~20–25% of samples land within $3\sigma$ of a mode after 10,000 steps.

Finally, inspecting the checkpoint traces shows why a calm critic loss curve must be interpreted with care:

```{.python .input #convergence-mode-coverage-on-25-gaussians-6}
%%tab pytorch, jax
traces = {name: np.array(results[0][1]) for name, results in runs.items()}
for col, stat in enumerate(['on-mode fraction', 'critic loss']):
    print(f'{stat} of run 0 at every 1,250 steps')
    for name, trace in traces.items():         # trace: (checkpoint, stat)
        print(f'  {name:28}', ' '.join(f'{v:4.2f}' for v in trace[:, col]))
```

In the penalized runs, the critic loss sits near its equilibrium value ($2\log 2 \approx 1.39$ for `NS`, $\log 2 \approx 0.69$ for `pairing`) from the very first checkpoint—even when only 1–3% of generated samples lie on a mode—because the gradient penalty keeps $D(x)$ small everywhere. An equilibrium critic loss indicates a flat critic, not necessarily a converged generator.

## Summary

- On the two-parameter Dirac-GAN ($p = \delta_0, q_\theta = \delta_\theta, D_\psi(x) = \psi x$), continuous-time gradient flow orbits on circles of constant radius, simultaneous discrete steps spiral outward, and alternating discrete steps orbit within a bounded band.
- Adding the zero-centered gradient penalties $R_1 = \frac{\gamma}{2}\mathbb{E}_p[\|\nabla_x D(x)\|^2]$ and $R_2 = \frac{\gamma}{2}\mathbb{E}_q[\|\nabla_{x'} D(x')\|^2]$ shifts the Jacobian eigenvalues into the open left half-plane ($\operatorname{Re}(\nu_{1,2}) < 0$) for any $\gamma > 0$, making the equilibrium locally exponentially stable.
- Near equilibrium, $R_1 + R_2$ acts as a low-pass filter on $p - q$ with cutoff length scale $2\sqrt{\gamma}$ (for the logistic loss), transitioning from $2\,\mathrm{JS}(p, q)$ at large scales to the squared dual Sobolev norm $\|p - q\|_{\dot H^{-1}(m)}^2$ at small scales.
- On the 25-Gaussian grid, $R_1 + R_2$ ($\gamma = 1$) damps critic steepness by three orders of magnitude and ensures reliable mode coverage across both classic and relativistic pairing losses.

## Exercises

1. [code] **Discrete-time stability of the Dirac-GAN.** Simultaneous gradient descent with step size $h$ maps $(\theta, \psi) \mapsto (\theta, \psi) + h\,\mathbf{u}(\theta, \psi)$.
    1. Show that at $\gamma = 0$, the Jacobian of the discrete update at $(0, 0)$ has eigenvalues $1 \pm i\,h\,\ell'(0)$, which have modulus $\sqrt{1 + h^2\ell'(0)^2} > 1$ for every $h > 0$.
    1. For $\gamma > 0$, the eigenvalues are $1 + h\nu_{1,2}$ with $\nu_{1,2}$ from :eqref:`eq_gan_dirac_pen`. Show that in the underdamped regime $\gamma < 2\ell'(0)$, the discrete update is locally stable ($|1 + h\nu_{1,2}| < 1$) if and only if $h < \gamma / \ell'(0)^2$.
    1. Verify this threshold at $\gamma = 0.3$ ($\ell'(0) = \tfrac12$, threshold $h = 1.2$) by running `trajectory(euler_step, (0.1, 0.1), h, 3000, gamma=0.3)` at $h = 1.1$ and $h = 1.3$.
   :exlabel:`ex_convergence-discrete-simultaneous-update-dirac`

1. **Penalized best response on the Dirac-GAN.** Consider $V_\gamma(\theta) = \sup_\psi \{V(\theta, \psi) - \gamma\psi^2\}$ on the Dirac-GAN with $\ell = \log \sigma$.
    1. Show that for small $\theta$, the maximizing slope is $\psi^\star \approx -\ell'(0)\theta / (2\gamma)$ and $V_\gamma(\theta) = 2\ell(0) + \frac{\ell'(0)^2}{4\gamma}\theta^2 + O(\theta^4)$, matching :eqref:`eq_gan_sobolev` with $W_2(\delta_0, \delta_\theta) = |\theta|$.
    1. Now let $D$ range over all differentiable functions penalized by $\gamma\int m\,\|\nabla_x D\|^2$. Show that for point masses $p = \delta_0$ and $q_\theta = \delta_\theta$ ($\theta \neq 0$), $\sup_D \{V_\ell(D) - R_1 - R_2\} = 2\sup_t \ell(t)$ is independent of $\theta$. Why does a finite neural network critic still provide a nonzero gradient across the gap?
   :exlabel:`ex_convergence-penalized-best-response`

1. **Penalizing $R_1$ alone versus $R_1 + R_2$.** Let $p = \delta_0$, $q = \delta_1$, $\ell = \log \sigma$, and consider the quadratic critics $D_c(x) = -cx^2$ ($c \geq 0$).
    1. Evaluate $R_1$ and $R_2$ for $D_c$, and show that $V_\ell(D_c) - R_1$ increases without bound as $c \to \infty$.
    1. Show that under the non-saturating generator loss $-\log \sigma(D_c(x'))$, the generator gradient magnitude at $x' = 1$ grows unboundedly with $c$, whereas subtracting $R_1 + R_2$ yields a finite maximizer $c^\star < 1/(8\gamma)$.
   :exlabel:`ex_convergence-one-sided-penalty`

1. [code] **Alternating updates on the Dirac-GAN.** ● Alternating updates apply $\psi_{k+1} = \psi_k + h\,u_\psi(\theta_k, \psi_k)$ followed by $\theta_{k+1} = \theta_k + h\,u_\theta(\theta_k, \psi_{k+1})$.
    1. Linearize the combined update at $(0, 0)$ for $\gamma = 0$ and show that its Jacobian has determinant $1$, with eigenvalues on the unit circle if and only if $h\,\ell'(0) \leq 2$.
    1. Run `trajectory(alt_step, (0.1, 0.1), h, 10**5)` at $\gamma = 0$ for $h \in \{0.2, 1, 3, 5\}$, record the minimum and maximum radius over the final $10^4$ steps, and compare with $\gamma = 0.3$ at $h = 0.2$.
   :exlabel:`ex_convergence-alternating-updates`

1. [code] **Local quadratic value and low-pass filtering.** ● Write $\mathcal{A} = |\ell''(0)|\,m + \gamma\mathcal{L}_m$ with $\mathcal{L}_m D = -\nabla \cdot (m \nabla D)$.
    1. Using integration by parts, show that $|\ell''(0)|\int m\,D^2 + \gamma\int m\,\|\nabla_x D\|^2 = \langle D, \mathcal{A}D \rangle$ and derive :eqref:`eq_gan_local_value` by maximizing along rays $t D_0$ and applying Cauchy–Schwarz.
    1. Let $p = \mathcal{N}(0, 1)$ and $q(x) = p(x)(1 + 0.1\sin(\omega x))$. Using `penalized_value`, compute the ratio of the penalized value at $\gamma \in \{0.01, 0.1, 1\}$ to the unpenalized value at $\gamma = 0$ for $\omega \in \{0.5, 2, 8\}$, and compare against the uniform-$m$ Fourier attenuation factor $(1 + 4\gamma\omega^2)^{-1}$.
   :exlabel:`ex_convergence-prove-eqref-eq-gan`

1. [code] **One-centered penalty dynamics on the Dirac-GAN.** ● Consider the WGAN linear payoff $\ell(t) = t$ ($\ell'(0) = 1$) with the one-centered penalty $\frac{\gamma}{2}(|\psi| - 1)^2$ subtracted from the critic objective.
    1. Show that the simultaneous gradient field never vanishes for $\psi \neq 0$, and find the generator positions $\theta$ where its one-sided limits as $\psi \to 0^\pm$ vanish.
    1. Modify `dirac_field` to use the one-centered penalty and run `trajectory` from $(1, 1)$ with $h = 0.01$ for 40,000 steps at $\gamma = 1$ and $\gamma = 3$, verifying that at $\gamma = 3$ the generator stalls at $\theta = 3$.
   :exlabel:`ex_convergence-one-centered-dynamics`

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

- Penalized: $\mathbb{E}_q\|\nabla D\|^2$ a few thousandths at most, 24–25
  modes in every run.
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
- $\gamma = 10$: about a fifth to a quarter on a mode, slower but not stuck.
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

