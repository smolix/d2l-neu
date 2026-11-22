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
\begin{aligned}
V(\theta, \psi) &= \ell\big(D_\psi(0)\big) + \ell\big({-D_\psi(\theta)}\big) \\
&= \ell(0) + \ell(-\psi\theta).
\end{aligned}
$$
:eqlabel:`eq_gan_dirac_value`

We take the logistic payoff $\ell(t) = \log \sigma(t)$, which satisfies $\ell'(t) = \sigma(-t) > 0$ and $\ell''(t) = -\sigma(t)\sigma(-t) < 0$, with $\ell'(0) = \tfrac12$ and $\ell''(0) = -\tfrac14$.

For any $\theta \neq 0$, the supports of $p$ and $q_\theta$ are disjoint: maximizing $V(\theta, \psi)$ over $\psi$ sends $\psi\theta \to -\infty$, yielding the constant $\sup_\psi V(\theta, \psi) = \ell(0)$ for all $\theta \neq 0$, which provides zero gradient with respect to $\theta$.

### Undamped Oscillations in Continuous Time

Instead of solving for $\sup_\psi V(\theta, \psi)$, gradient training updates $\theta$ and $\psi$ incrementally from their current values. The unique stationary point of :eqref:`eq_gan_dirac_value` is the equilibrium $(\theta, \psi) = (0, 0)$. Under simultaneous continuous-time gradient flow ($\dot\theta = -\partial_\theta V$, $\dot\psi = +\partial_\psi V$),

$$
\begin{aligned}
\dot\theta &= -\partial_\theta V = \psi\, \ell'(-\psi\theta), \\
\dot\psi &= +\partial_\psi V = -\theta\, \ell'(-\psi\theta).
\end{aligned}
$$
:eqlabel:`eq_gan_dirac_flow`

Because the velocity vector $\mathbf{u}(\theta, \psi) = \ell'(-\psi\theta)\,(\psi, -\theta)$ is everywhere orthogonal to the state vector $(\theta, \psi)$, the squared distance to the origin is conserved:

$$
\begin{aligned}
\frac{d}{dt}\big(\theta^2 + \psi^2\big)
&= 2\theta\psi\,\ell'(-\psi\theta) - 2\psi\theta\,\ell'(-\psi\theta) = 0 .
\end{aligned}
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
\begin{aligned}
\big\| (\theta, \psi) + h\, \mathbf{u} \big\|^2
&= \big\| (\theta, \psi) \big\|^2 + h^2 \big\| \mathbf{u} \big\|^2 .
\end{aligned}
$$

Thus, every simultaneous discrete step strictly *increases* the distance from the equilibrium for any $h > 0$, causing the iterates to spiral outward (Exercise :exref:`ex_convergence-discrete-simultaneous-update-dirac`).

Alternating gradient updates (updating $\psi$ first, then $\theta$ using the updated $\psi$) fare slightly better: near $(0, 0)$, each half-step is an area-preserving linear shear, so the composite Jacobian has determinant $1$. For $h\,\ell'(0) \leq 2$, both eigenvalues lie on the unit circle :cite:`Mescheder.Geiger.Nowozin.2018`, so alternating iterates orbit within a bounded band around the origin without converging (Exercise :exref:`ex_convergence-alternating-updates`). Neither the non-saturating generator loss (which shares the same linearization $J$ at the origin) nor the relativistic pairing loss ($\ell(D_\psi(0) - D_\psi(\theta)) = \ell(-\psi\theta)$, which has the identical gradient field) alters these local eigenvalues.

## Zero-Centered Gradient Penalties

To make the equilibrium $(\theta, \psi) = (0, 0)$ locally asymptotically stable, we must shift the eigenvalues of the Jacobian from the imaginary axis into the open left half-plane $\operatorname{Re}(\nu) < 0$. Because the equilibrium critic must be flat ($\nabla_x D^\star(x) = 0$ when $q = p$), penalizing the squared norm of the critic's input gradient adds damping to the critic without shifting the equilibrium $q = p$.

:citet:`Mescheder.Geiger.Nowozin.2018` define the two *zero-centered gradient penalties* $R_1$ (evaluated on real data $x \sim p$) and $R_2$ (evaluated on generated samples $x' \sim q$):

$$
\begin{aligned}
R_1 &= \frac{\gamma}{2}\, \mathbb{E}_{x \sim p}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big], \\
R_2 &= \frac{\gamma}{2}\, \mathbb{E}_{x' \sim q}\Big[ \big\| \nabla_{x'} D(x') \big\|^2 \Big],
\end{aligned}
$$
:eqlabel:`eq_gan_r1r2`

with penalty weight $\gamma > 0$. The critic maximizes $V_\ell(D) - R_1 - R_2$, while the generator objective is unchanged. Writing $m = (p + q)/2$, their sum is the expected squared gradient norm under the mixture distribution:

$$
\begin{aligned}
R_1 + R_2 &= \gamma\, \mathbb{E}_{x \sim m}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad m = \frac{p + q}{2}.
\end{aligned}
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
\begin{aligned}
V_\ell(D) &\approx 2\ell(0) + \ell'(0)\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 ,
\end{aligned}
$$

where $\langle p - q, D \rangle = \mathbb{E}_p[D] - \mathbb{E}_q[D]$. Consider two unit-variance Gaussians $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$ separated by a small shift $\mu$, and a linear critic $D(x) = cx$. To leading order in $\mu$, $\langle p - q, D \rangle = -c\mu$, $\int m\,D^2 \approx c^2$, and the penalty is $\gamma\int m\,\|\nabla_x D\|^2 = \gamma c^2$. Maximizing the concave quadratic $-\ell'(0)\mu c - (|\ell''(0)| + \gamma)c^2$ over $c$ yields the optimal slope $c^\star = -\frac{\ell'(0)\mu}{2(|\ell''(0)| + \gamma)}$ and value:

$$
\begin{aligned}
\frac{\ell'(0)^2 \mu^2}{4\,(|\ell''(0)| + \gamma)} &= \frac{\mu^2}{4 + 16\gamma} \\
&\text{for the logistic loss, at slope } c^\star = -\frac{\mu}{1 + 4\gamma} .
\end{aligned}
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
\begin{aligned}
&\sup_{D} \Big\{ \ell'(0)\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 - \gamma \int m\, \| \nabla_x D \|^2 \Big\} \\
&\qquad = \frac{\ell'(0)^2}{4}\, \big\langle p - q,\, \big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big)^{-1} (p - q) \big\rangle .
\end{aligned}
$$
:eqlabel:`eq_gan_local_value`

As $\gamma \to 0$, this reduces to $\tfrac{\ell'(0)^2}{4|\ell''(0)|} \int (p - q)^2/m \approx 2\,\mathrm{JS}(p, q)$. As $\gamma \to \infty$, it approaches a multiple of the squared *dual Sobolev norm* $\dot H^{-1}(m)$:

$$
\begin{aligned}
&\frac{\ell'(0)^2}{4\gamma}\, \big\| p - q \big\|^2_{\dot H^{-1}(m)}, \\
&\|v\|_{\dot H^{-1}(m)} = \sup\Big\{ \langle v, D\rangle : \int m \|\nabla_x D\|^2 \leq 1 \Big\}.
\end{aligned}
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

To score generated point clouds against the grid `centers`, `mode_stats`
assigns each sample to its nearest mode center and returns the count of covered
modes, the fraction of samples within $3\sigma$, and the mean distance to the
nearest center:

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

We now tabulate `mode_stats` on the 10,000 generated samples from each run
alongside the critic's final squared gradient norm
$\mathbb{E}_q[\|\nabla_x D(x')\|^2]$:

```{.python .input #convergence-mode-coverage-on-25-gaussians-2}
%%tab pytorch, jax
print(f'{"arm":28} run  modes  on-mode  distance  E_q|grad D|^2')
for name, results in runs.items():
    for seed, (x, trace) in enumerate(results):
        modes, on, dist = mode_stats(x, grid)
        print(f'{name:28} {seed:3}  {modes:2}/25  {on:7.2f}  {dist:8.2f}'
              f'  {trace[-1][2]:13.1e}')
```

Plotting 3,000 generated samples from the first run of each of the four main
arms against the 25 mode centers (`grid`) shows the difference in mode coverage
directly:

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

Minimax Game Dynamics and Gradient Penalties<br>**the Dirac-GAN · rotational vector fields · outward spirals & limit cycles · zero-centered $R_1$ & $R_2$ penalties · dual Sobolev filtering · R3GAN**
:::
:::

::: {.slide title="Equilibrium Existence vs. Gradient Convergence"}
[Motivation]{.kicker}

Sections 16.1–16.3 analyzed population divergences assuming the critic sits at its exact optimum $D^\star$. In actual training, both players take incremental gradient steps, and each player's update shifts the other player's objective:

- **Empirical warning from §16.1:** On Fashion-MNIST, neither the critic loss nor the generator loss settled to its theoretical equilibrium value after $20$ epochs.
- **Minimal counterexample (Mescheder et al., 2018):** Even on a two-parameter minimax game with a unique equilibrium at $(0, 0)$, taking $300$ simultaneous gradient steps of size $h = 0.2$ from $(1, 1)$ **doubles** the distance to the equilibrium from $1.41$ to $2.87$!

::: {.d2l-note .warn}
**The dynamical question:** Why do minimax gradient updates orbit or spiral away from equilibrium, and how do zero-centered input-gradient penalties ($R_1, R_2$) make the equilibrium locally exponentially stable?
:::
:::

::: {.slide title="The Dirac-GAN: A Two-Parameter Minimax System"}
[Minimal Analytical Model]{.kicker}

To analyze gradient dynamics in closed form, Mescheder, Geiger, and Nowozin (2018) introduced the **Dirac-GAN**:

- **True data distribution:** A unit point mass at the origin, $p = \delta_0$.
- **Generator distribution:** A unit point mass at $\theta \in \mathbb{R}$, $q_\theta = \delta_\theta$ (so $q_\theta = p \iff \theta = 0$).
- **Linear critic without bias:** $D_\psi(x) = \psi x$ parameterized by slope $\psi \in \mathbb{R}$.
- **Minimax value function:** Under any symmetric margin payoff $\ell(t)$ (e.g., logistic $\ell(t) = \log\sigma(t)$ with $\ell'(0) = \frac{1}{2}$ and $\ell''(0) = -\frac{1}{4}$), both expectations collapse to point evaluations:
  $$V(\theta, \psi) = \ell\big(D_\psi(0)\big) + \ell\big(-D_\psi(\theta)\big) = \ell(0) + \ell(-\psi\theta)$$
  Note that for any $\theta \neq 0$, maximizing over $\psi$ sends $\psi\theta \to -\infty$, yielding a constant best-response value $\sup_\psi V(\theta, \psi) = \ell(0)$ with zero gradient.
:::

::: {.slide title="Continuous-Time Gradient Flow Orbits on Circles"}
[Rotational Vector Field]{.kicker}

Instead of solving for $\sup_\psi V(\theta, \psi)$, simultaneous gradient flow updates $\theta$ and $\psi$ continuously along $\dot{\theta} = -\partial_\theta V$ and $\dot{\psi} = +\partial_\psi V$:

$$\dot{\theta} = -\partial_\theta V = \psi\,\ell'(-\psi\theta), \qquad \dot{\psi} = +\partial_\psi V = -\theta\,\ell'(-\psi\theta)$$

- **Exact conservation of radius:** Because the velocity vector $\mathbf{u}(\theta, \psi) = \ell'(-\psi\theta)\,(\psi, -\theta)$ is everywhere orthogonal to the state vector $(\theta, \psi)$, the squared distance to $(0, 0)$ is conserved:
  $$\frac{d}{dt}\big(\theta^2 + \psi^2\big) = 2\theta\psi\,\ell'(-\psi\theta) - 2\psi\theta\,\ell'(-\psi\theta) = 0$$
- **Purely imaginary Jacobian eigenvalues:** Linearizing around $(0, 0)$ yields $J = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & 0 \end{pmatrix}$ with eigenvalues $\nu_{1,2} = \pm i\,\ell'(0)$—a linear **center** whose trajectories orbit on closed circles of constant radius for *every* strictly increasing payoff $\ell$.
:::

::: {.slide title="Discrete Time: Outward Spirals vs. Limit Bands"}
[Discretization Hazards]{.kicker}

When we discretize the rotational vector field $\mathbf{u}(\theta, \psi)$ with step size $h > 0$, neither simultaneous nor alternating updates converge:

- **1. Simultaneous Euler steps spiral outward for every $h > 0$:** Since $\mathbf{u} \perp (\theta, \psi)$, the Pythagorean theorem shows that every step strictly increases the distance to $(0, 0)$:
  $$\big\|(\theta, \psi) + h\,\mathbf{u}\big\|^2 = \big\|(\theta, \psi)\big\|^2 + h^2\big\|\mathbf{u}\big\|^2 > \big\|(\theta, \psi)\big\|^2$$
  The discrete Jacobian has eigenvalues $1 \pm i\,h\,\ell'(0)$ with modulus $\sqrt{1 + h^2\ell'(0)^2} > 1$.
- **2. Alternating steps preserve area (bounded orbits, no convergence):** Updating $\psi$ first and then $\theta$ against the updated $\psi$ composes two area-preserving linear shears ($a = h\,\ell'(0)$):
  $$\begin{pmatrix} \theta_{k+1} \\ \psi_{k+1} \end{pmatrix} = \begin{pmatrix} 1 - a^2 & a \\ -a & 1 \end{pmatrix} \begin{pmatrix} \theta_k \\ \psi_k \end{pmatrix}, \qquad \det = 1$$
  For $a \le 2$, both eigenvalues lie on the unit circle, trapping iterates inside a bounded annulus around $(0, 0)$.
:::

::: {.slide title="Phase Portraits of Unpenalized and Penalized Flow"}
[Numerical Integration]{.kicker}

Integrating the Dirac-GAN from $(\theta_0, \psi_0) = (1, 1)$ (initial radius $\sqrt{2} \approx 1.41$) confirms all four dynamical regimes:

::: {.cols .vc}
::: {.col}
- **Left (`flow`):** Continuous RK4 integration orbits on the circle of radius $1.41$.
- **Middle (`steps`, $h = 0.2$):** $300$ simultaneous steps spiral out to radius $2.87$, while $2{,}000$ alternating steps orbit in $[1.34, 1.42]$.
- **Right (`penalized`, $h = 0.2$):** Adding a zero-centered critic penalty drives simultaneous Euler steps directly to $(0, 0)$ (reaching radius $6.5 \times 10^{-4}$ at $\gamma = 0.3$ and $7 \times 10^{-12}$ at $\gamma = 1.0$).
:::

::: {.col .fig .big}
@!convergence-phase-portraits-1
:::
:::
:::

::: {.slide title="Zero-Centered Gradient Penalties (R₁ and R₂)"}
[Damping the Critic]{.kicker}

To shift the Jacobian eigenvalues into the open left half-plane ($\operatorname{Re}(\nu) < 0$) without displacing the equilibrium $q = p$, we penalize a critic property that vanishes at $q = p$:

$$R_1 = \frac{\gamma}{2}\,\mathbb{E}_{x \sim p}\Big[\big\|\nabla_x D(x)\big\|^2\Big], \qquad R_2 = \frac{\gamma}{2}\,\mathbb{E}_{x' \sim q}\Big[\big\|\nabla_{x'} D(x')\big\|^2\Big]$$

- **Why penalize the critic's input gradient?** At $q = p$, the optimal log-ratio critic $D^\star(x) = \log(p(x)/q(x)) = 0$ is **completely flat** ($\nabla_x D^\star(x) = \mathbf{0}$), whereas we do not know in advance which parameter vector $\theta^\star$ makes the generator match $p$.
- **Penalized critic objective:** The critic maximizes $V_\ell(D) - R_1 - R_2$, where the combined penalty is the expected squared gradient norm under the balanced mixture $m = \frac{1}{2}(p + q)$:
  $$R_1 + R_2 = \gamma\,\mathbb{E}_{x \sim m}\Big[\big\|\nabla_x D(x)\big\|^2\Big]$$
:::

::: {.slide title="Eigenvalues and Three Oscillator Damping Regimes"}
[Damped Harmonic Oscillator]{.kicker}

On the Dirac-GAN ($D_\psi(x) = \psi x$), $\nabla_x D_\psi(x) = \psi$, so subtracting $\frac{\gamma}{2}\psi^2$ adds linear damping $-\gamma\psi$ to $\dot{\psi}$:

$$J_\gamma = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & -\gamma \end{pmatrix}, \qquad \nu_{1,2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^2}{4} - \ell'(0)^2}$$

For **every** $\gamma > 0$, $\operatorname{Re}(\nu_{1,2}) < 0$, making $(0, 0)$ locally exponentially stable with characteristic polynomial $\nu^2 + \gamma\nu + \ell'(0)^2 = 0$:

| Oscillator Regime | Penalty Range (Logistic $\ell'(0) = \tfrac{1}{2}$) | Eigenvalue Structure | Trajectory Behavior |
| :--- | :--- | :--- | :--- |
| **1. Underdamped** | $0 < \gamma < 2\ell'(0)$ ($0 < \gamma < 1$) | Complex conjugate, $\operatorname{Re}(\nu) = -\gamma/2$ | Spirals inward (Euler converges if $h < \gamma/\ell'(0)^2$) |
| **2. Critically damped** | $\gamma = 2\ell'(0)$ ($\gamma = 1$) | Repeated real $\nu_1 = \nu_2 = -\ell'(0) = -\tfrac{1}{2}$ | Fastest non-oscillatory decay to $(0, 0)$ |
| **3. Overdamped** | $\gamma > 2\ell'(0)$ ($\gamma > 1$) | Two negative real; slow mode $\nu \approx -\ell'(0)^2/\gamma$ | Non-oscillatory, but converges *more slowly* as $\gamma$ grows |
:::

::: {.slide title="Local Network Convergence and Penalty Design"}
[Generalization & Design Rules]{.kicker}

Mescheder et al. (2018) and Huang et al. (2024) extended this stability analysis from the two-parameter Dirac-GAN to general neural networks:

- **Local exponential convergence theorem:** Near an equilibrium manifold where $q_{\theta^\star} = p$ and $D_{\psi^\star}$ is flat on $\operatorname{supp}(p)$, simultaneous and alternating gradient updates with $R_1$/$R_2$ and a strictly concave loss ($\ell'(0) \neq 0, \ell''(0) < 0$) converge exponentially—provided $D$ has sufficient local capacity to detect first-order generator perturbations.
- **Why zero-centered ($R_1, R_2$) beats one-centered (WGAN-GP):** WGAN-GP penalizes $(\|\nabla_x D\| - 1)^2$, forcing unit slope even when $q = p$. On the Dirac-GAN with linear payoff $\ell(t) = t$, WGAN-GP parks the generator away from the data at $|\theta| = \gamma$ where the Wasserstein pull balances the penalty!
- **Why $R_1 + R_2$ beats $R_1$ alone:** Far from equilibrium, $R_1$ constrains $\nabla_x D$ only on $\operatorname{supp}(p)$. For $p = \delta_0, q = \delta_1$, the quadratic critic $D_c(x) = -cx^2$ has $R_1 = 0$ (flat at $0$) and lets $c \to \infty$, exploding generator gradients at $x' = 1$; adding $R_2 = 2\gamma c^2$ caps $c^\star < 1/(8\gamma)$.
:::

::: {.slide title="Effect of Penalties on the Local Value Function"}
[Quadratic Expansion & Gaussian Shift]{.kicker}

How does subtracting $R_1 + R_2 = \gamma\int m\,\|\nabla_x D\|^2$ modify the divergence near equilibrium? Expanding $\ell(t) \approx \ell(0) + \ell'(0)t - \frac{1}{2}|\ell''(0)|t^2$ to second order in $D$ gives:

$$V_\ell(D) - R_1 - R_2 \approx 2\ell(0) + \ell'(0)\,\langle p - q,\, D\rangle - |\ell''(0)|\int m\,D^2 - \gamma\int m\,\|\nabla_x D\|^2$$

- **1D Gaussian shift ($p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\mu, 1)$, linear critic $D(x) = cx$):** For small $\mu$, maximizing $-\ell'(0)\mu c - (|\ell''(0)| + \gamma)c^2$ under the logistic loss ($\ell'(0) = \frac{1}{2}, |\ell''(0)| = \frac{1}{4}$) shrinks both the optimal slope and local value by $1 + 4\gamma$:
  $$c^\star = -\frac{\mu}{1 + 4\gamma}, \qquad V_\gamma(D^\star) - 2\ell(0) \approx \frac{\mu^2}{4 + 16\gamma}$$

@!convergence-a-gaussian-shift
:::

::: {.slide title="Dual Sobolev Norm and Spatial Low-Pass Filtering"}
[Multiscale Interpretation]{.kicker}

Integrating by parts writes $\int m\,\|\nabla_x D\|^2 = \langle D, \mathcal{L}_m D\rangle$, where $\mathcal{L}_m D = -\nabla \cdot (m\nabla D)$ is the $m$-weighted Laplacian. Maximizing the local quadratic approximation over all functions $D$ yields:

$$\sup_D \Big\{ \ell'(0)\,\langle p - q, D\rangle - |\ell''(0)|\int m\,D^2 - \gamma\int m\,\|\nabla_x D\|^2 \Big\} = \frac{\ell'(0)^2}{4}\Big\langle p - q,\; \big(|\ell''(0)|\,m + \gamma\mathcal{L}_m\big)^{-1}(p - q)\Big\rangle$$

- **Small-penalty limit ($\gamma \to 0$):** Recovers $\frac{\ell'(0)^2}{4|\ell''(0)|}\int \frac{(p - q)^2}{m} \approx 2\,\mathrm{JS}(p, q)$.
- **Large-penalty limit ($\gamma \to \infty$):** Approaches $\frac{\ell'(0)^2}{4\gamma}\|p - q\|_{\dot{H}^{-1}(m)}^2$, the squared **dual Sobolev norm**.
- **Spatial low-pass filter:** On a Fourier mode of wavenumber $\omega$, $\mathcal{L}_m$ multiplies by $m\omega^2$, weighting the mode by $(|\ell''(0)| + \gamma\omega^2)^{-1}$ instead of $|\ell''(0)|^{-1}$. Thus $R_1 + R_2$ filters out differences below the cutoff length scale $\sqrt{\gamma / |\ell''(0)|} = 2\sqrt{\gamma}$, meaning $\gamma$ has units of **squared length**.
:::

::: {.slide title="Implementing Differentiable R₁ + R₂ Penalties"}
[Higher-Order Autograd]{.kicker}

Because each sample's critic score $D(x_i)$ depends only on row $x_i$ (provided batch normalization is omitted!), differentiating the sum $\sum_i D(x_i)$ with respect to the input batch $x$ computes every sample's input gradient $\nabla_{x_i} D(x_i)$ in a single backward pass:

@convergence-the-loss-in-code-2

::: {.d2l-note}
**Computational cost:** Keeping the input gradient in the autodiff graph (`create_graph=True` in PyTorch; nested `jax.grad` in JAX) so the optimizer can differentiate $\|\nabla_x D(x)\|^2$ with respect to the critic's parameters makes each penalized critic update cost roughly **two backward passes**.
:::
:::

::: {.slide title="The R3GAN Critic and Generator Step Functions"}
[Training Recipe in Code]{.kicker}

R3GAN (Huang et al., 2024) pairs the relativistic pairing loss (`rpgan_loss_D`, `rpgan_loss_G`) with $R_1 + R_2$. `critic_step` updates `net_D` on `loss_D + gamma / 2 * (r1 + r2).mean()` and returns the unscaled batch means of $R_1$ and $R_2$ for logging:

@convergence-the-loss-in-code-1
:::

::: {.slide title="Mode-Coverage Testbed: 25 Gaussians on a 5×5 Grid"}
[Experimental Setup]{.kicker}

To test whether pairwise ranking or zero-centered gradient penalties drive mode coverage in neural networks, we train a 3-layer ReLU MLP generator ($64 \to 256 \to 256 \to 2$) and leaky-ReLU MLP critic ($2 \to 256 \to 256 \to 1$) on a $5 \times 5$ grid of isotropic 2D Gaussians (spacing $2.0$, width $\sigma = 0.05$):

- **Optimizer settings ($10{,}000$ steps, batch size $1{,}024$):** Adam with **$\beta_1 = 0, \beta_2 = 0.99$** (since positive first-moment momentum averages past orthogonal directions in a rotating vector field) and cosine LR decay from $2 \times 10^{-3}$ to $2 \times 10^{-4}$.
- **$2 \times 2$ factorial design (3 seeds per arm):** Crosses objective (`NS` vs. `pairing`) with penalty ($\gamma = 0$ vs. $\gamma = 1$), plus an overdamped run at $\gamma = 10$.
- **Four metrics on $10{,}000$ generated samples:** (1) covered modes ($\ge \frac{1}{5}$ of an equal share $\frac{1}{25}$ of on-mode samples), (2) on-mode fraction (within $3\sigma = 0.15$ of a center; $98.9\%$ for true $p$), (3) mean distance to nearest center ($\approx 0.06$ for true $p$), and (4) critic steepness $\mathbb{E}_q[\|\nabla_x D(x')\|^2]$.
:::

::: {.slide title="25-Gaussian Results: Penalties Drive Coverage"}
[Quantitative Comparison]{.kicker}

Evaluating $10{,}000$ samples at step $10{,}000$ isolates the role of each ingredient:

@!convergence-mode-coverage-on-25-gaussians-2

- **Gradient penalties ($R_1 + R_2$ at $\gamma = 1$) drive mode coverage:** Without penalties ($\gamma = 0$), critic gradients remain steep ($\mathbb{E}_q\|\nabla_x D\|^2 \in [0.7, 17]$) and mode coverage is unreliable ($12\text{--}25$ modes). With $\gamma = 1$, critic steepness drops by three orders of magnitude ($\sim 10^{-3}$) and **every run covers $24\text{--}25$ modes** under both `NS` and `pairing`.
:::

::: {.slide title="Visualizing Mode Coverage and Overdamping"}
[Spatial Distributions]{.kicker}

@!convergence-mode-coverage-on-25-gaussians-7

- **Penalized vs. unpenalized:** Both $\gamma = 1$ arms (`NS + R1 + R2` and `pairing + R1 + R2`) populate all $25$ grid centers, whereas unpenalized runs drop modes or leave clusters uneven.
- **Why filaments connect neighboring modes:** A continuous ReLU generator maps the connected Gaussian latent space $\mathcal{N}(0, I_{64})$ onto a connected surface, leaving thin bridges between modes.
- **Overdamping ($\gamma = 10$):** Over-flattening the critic ($\mathbb{E}_q\|\nabla_x D\|^2 \sim 10^{-4}$) slows convergence, leaving only $\sim 20\text{--}25\%$ of samples within $3\sigma$ of a mode after $10{,}000$ steps.
:::

::: {.slide title="Diagnostic Pitfall: Calm Losses Certify Nothing"}
[Monitoring Hazard]{.kicker}

Comparing the on-mode sample fraction against the critic loss across all $8$ checkpoints (every $1{,}250$ steps) reveals a critical monitoring hazard:

@!convergence-mode-coverage-on-25-gaussians-6

::: {.d2l-note .warn}
**Why an equilibrium critic loss does not certify convergence:** In every penalized run, the critic loss sits at its exact equilibrium value ($2\log 2 \approx 1.38$ for `NS`; $\log 2 \approx 0.69$ for `pairing`) from the very first checkpoint at step $1{,}250$—when only **$1\text{--}3\%$** of generated samples lie on a mode! Because $R_1 + R_2$ forces $D(x)$ to remain nearly flat everywhere, an equilibrium critic loss only confirms that the critic is damped, never that $q$ matches $p$.
:::
:::

::: {.slide title="Concept Check: Minimax Dynamics and Penalties"}
[Check Your Understanding]{.kicker}

Test your understanding of minimax rotational dynamics and zero-centered gradient penalties:

1. **Discrete Euler stability threshold:** On the penalized Dirac-GAN with logistic loss ($\ell'(0) = \frac{1}{2}$), the continuous Jacobian eigenvalues are $\nu_{1,2} = -\frac{\gamma}{2} \pm i\sqrt{\ell'(0)^2 - \gamma^2/4}$ in the underdamped regime $\gamma < 1$. What is the maximum step size $h$ for which simultaneous discrete Euler steps converge ($|1 + h\nu_{1,2}| < 1$), and what is its numerical value at $\gamma = 0.3$?
2. **Penalized Gaussian shift:** For $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$ with small shift $\mu = 0.2$ under the logistic loss, by what factor does $\gamma = 1$ shrink the optimal linear critic slope $c^\star$ compared to $\gamma = 0$?
3. **Pairing on the Dirac-GAN:** Does replacing the single-sample logistic loss with the relativistic pairing loss $\ell(D_\psi(0) - D_\psi(\theta))$ change the unpenalized Dirac-GAN vector field?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Since $\nu_1 \nu_2 = |\nu|^2 = \ell'(0)^2$ and $2\operatorname{Re}(\nu) = -\gamma$, we have $|1 + h\nu|^2 = 1 - h\gamma + h^2\ell'(0)^2 < 1$ $\iff$ $h < \frac{\gamma}{\ell'(0)^2} = \mathbf{4\gamma}$. At $\gamma = 0.3$, simultaneous Euler steps converge for **$h < 1.2$**.
2. Since $c^\star = -\frac{\mu}{1 + 4\gamma}$, setting $\gamma = 1$ shrinks both the critic slope and the local quadratic value $\frac{\mu^2}{4 + 16\gamma}$ by a factor of $1 + 4(1) = \mathbf{5\times}$.
3. **No.** Because $D_\psi(0) = 0$, the pairing objective $\ell(D_\psi(0) - D_\psi(\theta)) = \ell(-\psi\theta)$ differs from the single-sample objective $\ell(0) + \ell(-\psi\theta)$ only by the constant $\ell(0)$, producing the exact same rotational vector field.
:::
:::

::: {.slide title="Summary: Dynamics, Gradient Penalties, and R3GAN"}
[Key Takeaways]{.kicker}

| Dynamical Regime / Mechanism | Mathematical Form | Jacobian Eigenvalues / Value | Training Behavior & Practical Role |
| :--- | :--- | :--- | :--- |
| **Unpenalized Dirac-GAN Flow** | $\dot{\theta} = \psi\ell'(-\psi\theta)$, $\dot{\psi} = -\theta\ell'(-\psi\theta)$ | $\nu_{1,2} = \pm i\,\ell'(0)$ (linear center) | $\frac{d}{dt}(\theta^2 + \psi^2) = 0$; orbits on constant-radius circles |
| **Discrete Simultaneous Steps** | $(\theta, \psi) \mapsto (\theta, \psi) + h\,\mathbf{u}(\theta, \psi)$ | $1 \pm i\,h\,\ell'(0)$ ($|\lambda| = \sqrt{1 + h^2\ell'(0)^2} > 1$) | Spirals outward for every step size $h > 0$ |
| **Discrete Alternating Steps** | Critic half-step $\to$ generator half-step | $\det = 1$; on unit circle for $h\ell'(0) \le 2$ | Area-preserving shears; orbits in a bounded band |
| **Zero-Centered Penalties ($R_1, R_2$)** | $R_1 + R_2 = \gamma\,\mathbb{E}_m[\|\nabla_x D(x)\|^2]$ | $\nu_{1,2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^2}{4} - \ell'(0)^2}$ ($\operatorname{Re} < 0$) | Exponential local convergence for all $\gamma > 0$ |
| **Local Penalized Value** | $\frac{\ell'(0)^2}{4}\langle p - q, (|\ell''(0)|m + \gamma\mathcal{L}_m)^{-1}(p - q)\rangle$ | Shrinks Gaussian shift by $1 + 4\gamma$ | Low-pass filters $p - q$ below cutoff scale $2\sqrt{\gamma}$ ($\dot{H}^{-1}(m)$) |
| **R3GAN Recipe** | $\operatorname{softplus}(D(x') - D(x)) + \frac{\gamma}{2}(R_1 + R_2)$ | Paired with Adam $\beta_1 = 0$ & no BatchNorm | Damps critic steepness $1000\times$; covers $24\text{--}25$ modes |
:::

