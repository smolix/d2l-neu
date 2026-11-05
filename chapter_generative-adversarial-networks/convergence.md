# Gradient Penalties and Convergence
:label:`sec_gan_convergence`

The garment generator of :numref:`sec_basic_gan` learned to draw shirts, shoes and dresses, but its losses never settled. The critic's loss stayed well below $\log 2$ and the generator's well above it, both still drifting when training stopped. At the equilibrium, where the generated images match the data and the critic can only guess, both losses equal $\log 2$. The last three sections computed what the game is worth when the critic answers optimally. Training never gives the critic that chance: both networks take small gradient steps at the same time, and each step of one player changes the objective of the other. Knowing where the equilibrium lies does not tell us whether these steps lead there.

:citet:`Mescheder.Geiger.Nowozin.2018` asked which training methods for GANs converge. Their answer starts from the smallest game there is: one data point, one generated point and a straight-line critic. Plain gradient steps fail even on this game, and a penalty on the critic's slope at the data and generated samples repairs them. We analyze that game, add the penalty, ask what the penalized game measures, and train the resulting R3GAN loss :cite:`Huang.Gokaslan.Kuleshov.ea.2024` on a grid of 25 Gaussians.

```{.python .input #convergence-gradient-penalties-and-convergence}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #convergence-gradient-penalties-and-convergence}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import numpy as np
import optax
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

This is the *Dirac-GAN* of :citet:`Mescheder.Geiger.Nowozin.2018`. Any differentiable loss with $\ell'(0) > 0$ works. We use the logistic loss $\ell = \log\sigma$, whose derivatives are $\ell'(t) = \sigma(-t)$ and $\ell''(t) = -\sigma(t)\sigma(-t)$, so $\ell'(0) = \tfrac12$ and $\ell''(0) = -\tfrac14$.

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

Every trajectory is a circle around the solution, for every loss in this family. The linearization at the origin gives the same picture locally,

$$
J = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & 0 \end{pmatrix},
\qquad
\nu_{1,2} = \pm\, i\, \ell'(0),
$$
:eqlabel:`eq_gan_dirac_eigs`

with purely imaginary eigenvalues: the equilibrium is a center, not an attractor. The generated point passes the data again and again without settling there. The common factor $\ell'(-\psi\theta) = \sigma(D_\psi(\theta))$ is the weight with which the saturating generator of :numref:`sec_basic_gan` scales the update of a generated sample. Because it multiplies both components, it sets how fast the parameters turn and not where they go, so no reweighting of this kind turns the circle into a spiral toward the solution.

### Outward Spirals in Discrete Time

Discrete steps do worse than the flow. Write $\mathbf{u}(\theta, \psi)$ for the right-hand side of :eqref:`eq_gan_dirac_flow`. A simultaneous step of size $h > 0$ maps $(\theta, \psi)$ to $(\theta, \psi) + h\,\mathbf{u}$. Since $\mathbf{u}$ is orthogonal to $(\theta, \psi)$, the Pythagorean theorem gives

$$
\big\| (\theta, \psi) + h\, \mathbf{u} \big\|^2
\;=\; \big\| (\theta, \psi) \big\|^2 + h^2 \big\| \mathbf{u} \big\|^2 ,
$$

so each step adds $h^2\|\mathbf{u}\|^2$ to the squared distance. For the logistic loss $\mathbf{u}$ vanishes only at the equilibrium, so every step from any other point moves away from the solution, and a smaller step only slows the outward spiral (exercise :exref:`ex_convergence-discrete-simultaneous-update-dirac`). Updating the two players in turn instead of simultaneously does not restore convergence either (exercise :exref:`ex_convergence-alternating-updates`).

The two repairs of the earlier sections leave these dynamics in place. The non-saturating generator has the same Jacobian $J$ at the equilibrium; away from it the field gains a slow inward drift, which discrete steps stop on a cycle of radius about $\sqrt{2h}$ (exercise :exref:`ex_convergence-non-saturating-dynamics`). The pairing objective of :numref:`sec_gan_relativistic` scores the pair by $\ell\big(D_\psi(0) - D_\psi(\theta)\big) = \ell(-\psi\theta)$, which differs from :eqref:`eq_gan_dirac_value` only by the constant $\ell(0)$, so its gradient field, its circles and its spirals are the classical ones. A cure has to change the dynamics themselves.

## Zero-Centered Penalties

The equilibrium attracts the iterates only if the eigenvalues of the linearized dynamics have negative real parts. The rotation needs friction, and the natural place to add it is the critic's slope, the quantity $\psi$ that keeps the rotation going. Define the two *zero-centered gradient penalties*

$$
R_1 \;=\; \frac{\gamma}{2}\, \mathbb{E}_{x \sim p}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad
R_2 \;=\; \frac{\gamma}{2}\, \mathbb{E}_{x' \sim q}\Big[ \big\| \nabla_{x'} D(x') \big\|^2 \Big],
$$
:eqlabel:`eq_gan_r1r2`

with weight $\gamma > 0$, and let the critic maximize $V_\ell(D) - R_1 - R_2$ while the generator's objective stays unchanged :cite:`Roth.Lucchi.Nowozin.ea.2017,Mescheder.Geiger.Nowozin.2018`. The penalties act on the critic's gradient with respect to its *input*, not its parameters: $R_1$ limits how steeply the critic varies at data samples and $R_2$ at generated samples. They are called zero-centered because the flat critic $\nabla_x D = 0$ minimizes both. At $q = p$ the optimal critic $\log(p/q)$ of :numref:`sec_basic_gan` is the constant zero, so the penalties vanish at the equilibrium they are meant to stabilize. Since $\tfrac12(\mathbb{E}_p + \mathbb{E}_q) = \mathbb{E}_m$, their sum is

$$
R_1 + R_2 \;=\; \gamma\, \mathbb{E}_{x \sim m}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad m = \frac{p + q}{2},
$$
:eqlabel:`eq_gan_r1r2_sum`

one smoothness penalty under the mixture $m$, which constrains the critic wherever either distribution puts mass.

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

Both eigenvalues have negative real parts for every $\gamma > 0$. The equilibrium attracts, and small enough gradient steps converge to it at a linear rate :cite:`Mescheder.Geiger.Nowozin.2018`. As for a damped spring, there are three regimes. For $\gamma < 2\ell'(0)$ the square root is imaginary and trajectories spiral inward. At $\gamma = 2\ell'(0)$, which is $\gamma = 1$ for the logistic loss, the repeated real eigenvalue gives critical damping, the fastest approach without oscillation. Larger values overdamp the dynamics: the slower eigenvalue, close to $-\ell'(0)^2/\gamma$, makes convergence slower as $\gamma$ grows.

### Phase Portraits

We check the circles, the outward spiral and the damping by simulating the two equations for the logistic loss. The function `trajectory` either integrates the flow with fourth-order Runge--Kutta steps or applies simultaneous gradient steps of size $h$.

```{.python .input #convergence-phase-portraits}
%%tab pytorch, jax
def dirac_field(theta, psi, gamma=0.0):
    """Simultaneous gradient field of the Dirac-GAN with the logistic loss."""
    w = 1.0 / (1.0 + np.exp(-psi * theta))      # l'(-psi theta) = w_sat
    return np.array([psi * w, -theta * w - gamma * psi])

def trajectory(x0, h, steps, gamma=0.0, flow=False):
    xs, x = [np.array(x0)], np.array(x0)
    for _ in range(steps):
        if flow:                                 # one RK4 step of the ODE
            k1 = dirac_field(*x, gamma)
            k2 = dirac_field(*(x + h / 2 * k1), gamma)
            k3 = dirac_field(*(x + h / 2 * k2), gamma)
            k4 = dirac_field(*(x + h * k3), gamma)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:                                    # one simultaneous step
            x = x + h * dirac_field(*x, gamma)
        xs.append(x)
    return np.stack(xs)
```

All panels start at $(\theta, \psi) = (1, 1)$, in red; the cross marks the equilibrium. They show the flow, 300 simultaneous steps of size $h = 0.2$ against the flow's circle (dashed), and the same steps with the penalty at $\gamma = 0.3$, in the underdamped regime, and at the critical $\gamma = 1$.

```{.python .input #convergence-phase-portraits-1}
%%tab pytorch, jax
def phase_panel(ax, paths, title, lim, legend=None, fmts=('C0-', 'C1-')):
    """Draw paths in the (theta, psi) plane, with start and equilibrium."""
    X = [P[:, 0] for P in paths] + [[paths[0][0, 0]], [0.0]]
    Y = [P[:, 1] for P in paths] + [[paths[0][0, 1]], [0.0]]
    d2l.plot(X, Y, r'$\theta$', r'$\psi$', legend, xlim=(-lim, lim),
             ylim=(-lim, lim), fmts=fmts[:len(paths)] + ('C3o', 'k+'), axes=ax)
    ax.set_title(title), ax.set_aspect('equal')

x0, t = (1.0, 1.0), np.linspace(0, 2 * np.pi, 200)
circle = np.sqrt(2) * np.stack([np.cos(t), np.sin(t)], axis=1)
fig, axes = d2l.plt.subplots(1, 3, figsize=(9.5, 3.4))
phase_panel(axes[0], [trajectory(x0, 0.05, 800, flow=True)], 'flow', 1.6)
phase_panel(axes[1], [trajectory(x0, 0.2, 300), circle], 'steps', 3.2,
            fmts=('C0-', 'k--'))
phase_panel(axes[2], [trajectory(x0, 0.2, 300, gamma=0.3),
                      trajectory(x0, 0.2, 300, gamma=1.0)], 'penalized',
            1.6, legend=[r'$\gamma = 0.3$', r'$\gamma = 1$'])
fig.tight_layout()
```

The integrated flow retraces one circle, as the conservation law requires. The discrete iterates leave the dashed circle at once and move outward on every revolution; after 300 steps their distance from the equilibrium has doubled. Both penalized trajectories converge, the underdamped one along a shrinking spiral and the critically damped one directly. The penalty turns divergence into convergence and leaves the equilibrium where it was.

### One-Centered versus Zero-Centered

WGAN-GP centers its penalty differently :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017`. To enforce the Lipschitz constraint of :numref:`sec_gan_objectives`, it penalizes $\mathbb{E}\big[(\|\nabla_x D\| - 1)^2\big]$ at points between real and generated samples, because optimal $W_1$ critics have unit slope along transport rays. At $q = p$, however, the optimal critic is constant. A penalty that prefers unit slope keeps pushing a generator that already matches the data, and on the Dirac-GAN these dynamics do not converge :cite:`Mescheder.Geiger.Nowozin.2018` (exercise :exref:`ex_convergence-one-centered-dynamics`).

| penalty | critic it prefers at $q = p$ | Dirac-GAN |
|:---|:---|:---|
| one-centered, $\mathbb{E}\big[(\|\nabla_x D\| - 1)^2\big]$ | unit slope | does not converge |
| zero-centered, $\mathbb{E}_m\big[\|\nabla_x D\|^2\big]$ | flat | converges for every $\gamma > 0$ |

### One Penalty or Two

The Dirac-GAN cannot tell $R_1$ from $R_2$, because a linear critic has the same slope everywhere. Near equilibrium the supports of $p$ and $q$ nearly coincide, so either penalty suffices, and the local convergence result of :citet:`Mescheder.Geiger.Nowozin.2018` needs only one. Far from equilibrium the two differ. :citet:`Roth.Lucchi.Nowozin.ea.2017` showed that penalizing $\mathbb{E}_p\big[\|\nabla_x D\|^2\big]$ changes the game, to leading order, as if $p$ were convolved with Gaussian noise of covariance proportional to $\gamma$, and $R_2$ smooths $q$ in the same way. Smoothed, two distributions with disjoint supports overlap, and the divergence responds to the distance between them. With $R_1$ alone the critic is unconstrained where only generated samples lie. R3GAN's ablation on StackedMNIST, stacked MNIST digits with 1,000 known modes :cite:`Metz.Poole.Pfau.ea.2017`, shows the consequence :cite:`Huang.Gokaslan.Kuleshov.ea.2024`:

| loss | modes recovered (of 1,000) | reverse KL |
|:---|---:|---:|
| RpGAN with $R_1$ and $R_2$ | 1,000 | 0.078 |
| GAN with $R_1$ and $R_2$ | 693 | 0.927 |
| RpGAN with $R_1$ only | diverged | --- |
| GAN with $R_1$ only | diverged | --- |

With $R_1$ alone the critic's input gradients on generated samples grew and both losses diverged, for every $\gamma$ from 0.1 to 100 in the paired case. With both penalties both trained, and the paired loss recovered every mode: in this ablation the penalties decided convergence and the pairing decided coverage.

## The Penalized Value Near Equilibrium

The eigenvalues say how training approaches the equilibrium; what does the penalized game measure on the way? Without the penalty the value was a divergence of the density ratio, blind to how far apart separated supports are. With it, the value near equilibrium blends that divergence with a transport distance, and $\gamma$ decides which length scales fall on which side.

### The Local Value

Near $q = p$ the optimal critic is close to zero, so we expand the payoff to second order in the critic. With $\langle p - q, D \rangle = \mathbb{E}_p[D] - \mathbb{E}_q[D]$, and up to third-order terms,

$$
V_\ell(D) = 2\ell(0) + \ell'(0)\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 .
$$

The pairing objective has $\mathrm{Var}_m[D]$ in place of $\int m D^2$, which gives the same supremum once the critic's constant is chosen best. The quadratic term is not negligible: the best critic has size proportional to $p - q$, so all terms are of the same order at the optimum.

**Proposition.** *Let $\mathcal{L}_m D = -\nabla \cdot (m \nabla D)$. Near equilibrium, the penalized critic problem has the value*

$$
\sup_{D} \Big\{ \ell'(0)\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 - \gamma \int m\, \| \nabla_x D \|^2 \Big\}
\;=\; \frac{\ell'(0)^2}{4}\, \big\langle p - q,\, \big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big)^{-1} (p - q) \big\rangle,
$$
:eqlabel:`eq_gan_local_value`

*attained by the critic that solves $\big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big) D = \tfrac{\ell'(0)}{2}(p - q)$.*

**Proof.** Write $\mathcal{A} = |\ell''(0)|\, m + \gamma \mathcal{L}_m$; integrating by parts, the two quadratic terms are $\langle D, \mathcal{A} D \rangle$, and $\mathcal{A}$ is symmetric and positive. Along a ray $\{t D_0\}$ the objective $t\,\ell'(0)\langle p - q, D_0\rangle - t^2 \langle D_0, \mathcal{A} D_0\rangle$ is a concave parabola in $t$ with maximum $\ell'(0)^2 \langle p - q, D_0\rangle^2 / (4\langle D_0, \mathcal{A} D_0 \rangle)$. By the Cauchy--Schwarz inequality in the inner product $\langle \cdot, \mathcal{A}\, \cdot \rangle$, the largest value over directions is $\tfrac{\ell'(0)^2}{4}\langle p - q, \mathcal{A}^{-1}(p - q)\rangle$, reached at $D_0 \propto \mathcal{A}^{-1}(p - q)$. $\blacksquare$

The value is a weighted squared size of $p - q$. As $\gamma \to 0$ it becomes $\tfrac{\ell'(0)^2}{4|\ell''(0)|}\int (p - q)^2/m$, which for the logistic loss is $\tfrac14 \int (p-q)^2/m$, the local form of $2\,\mathrm{JS}(p, q)$, the unpenalized value of :numref:`sec_basic_gan`. When $\gamma$ is large, the penalty dominates and the value approaches

$$
\frac{\ell'(0)^2}{4\gamma}\, \big\| p - q \big\|^2_{\dot H^{-1}(m)},
\qquad
\|v\|_{\dot H^{-1}(m)} = \sup\Big\{ \langle v, D\rangle : \int m \|\nabla_x D\|^2 \leq 1 \Big\},
$$
:eqlabel:`eq_gan_sobolev`

a squared dual Sobolev norm, taken over critics modulo constants.

### A Gaussian Shift

For $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$ with a small shift $\mu$, to first order $p - q \approx -\mu\, x\, m(x)$. Since $\mathcal{L}_m x = x\, m(x)$ for the standard normal density $m$, a straight-line critic solves the equation of the proposition:

$$
D(x) = -\frac{\ell'(0)\,\mu}{2\,(|\ell''(0)| + \gamma)}\; x,
\qquad
\text{value} = \frac{\ell'(0)^2 \mu^2}{4\,(|\ell''(0)| + \gamma)} = \frac{\mu^2}{4 + 16\gamma}
\ \text{for the logistic loss}.
$$

This is the log ratio $-\mu x$ with its slope shrunk by $1 + 4\gamma$. At $\gamma = 0$ the value is $\mu^2/4 = 2\,\mathrm{JS}$ to leading order, and for large $\gamma$ it is $\mu^2/(16\gamma) = \tfrac{\ell'(0)^2}{4\gamma} W_2(p, q)^2$, since the Wasserstein-2 distance of a shift is $|\mu|$. To check both approximations, we solve the full penalized problem by Newton's method for a critic tabulated on a grid.

```{.python .input #convergence-a-gaussian-shift}
%%tab pytorch, jax
def penalized_value(p, q, gamma, x, newton_steps=30):
    """Value of the logistic game minus gamma E_m[D'^2], on a grid."""
    dx, m = x[1] - x[0], (p + q) / 2
    diff = np.diff(np.eye(len(x)), axis=0) / dx       # D -> D'
    K = 2 * gamma * dx * diff.T @ (((m[1:] + m[:-1]) / 2)[:, None] * diff)
    D = np.zeros_like(x)
    for _ in range(newton_steps):
        s = 1 / (1 + np.exp(-D))
        grad = (p * (1 - s) - q * s) * dx - K @ D
        D = D + np.linalg.solve(np.diag(2 * m * s * (1 - s) * dx) + K, grad)
    value = -(p * np.logaddexp(0, -D) + q * np.logaddexp(0, D)).sum() * dx
    return value - D @ K @ D / 2 + 2 * np.log(2)       # minus 2 l(0)

x, mu = np.linspace(-8, 8, 401), 0.2
p, q = [np.exp(-(x - c) ** 2 / 2) / np.sqrt(2 * np.pi) for c in (0, mu)]
for gamma in (0, 0.01, 1, 10):
    large = mu ** 2 / (16 * gamma) if gamma else np.inf
    print(f'gamma {gamma:>5}: exact {penalized_value(p, q, gamma, x):.3e}, '
          f'local {mu ** 2 / (4 + 16 * gamma):.3e}, large-gamma {large:.3e}')
```

The local formula matches the exact value to within one percent at every $\gamma$. The large-$\gamma$ form holds only when the penalty dominates: at $\gamma = 0.01$ it gives $0.25$ against the exact $0.0096$, 26 times too large, and at $\gamma = 10$ it is within three percent.

### A Low-Pass Filter

A general $p - q$ has many length scales. Where $m$ is roughly constant, $\mathcal{L}_m$ multiplies a Fourier component $e^{i \omega x}$ by $m \omega^2$, so in :eqref:`eq_gan_local_value` the component with wavenumber $\omega$ enters with weight

$$
\frac{1}{|\ell''(0)| + \gamma \omega^2}
\qquad \text{instead of} \qquad \frac{1}{|\ell''(0)|}.
$$

Components whose length scale $1/\omega$ exceeds $\sqrt{\gamma / |\ell''(0)|}$, which is $2\sqrt\gamma$ for the logistic loss, keep nearly their unpenalized weight; shorter ones are damped by about the factor $|\ell''(0)|/(\gamma \omega^2)$. The penalty is a low-pass filter on $p - q$ with cutoff length $2\sqrt\gamma$. Above the cutoff the game measures $2\,\mathrm{JS}$; below it, the dual Sobolev norm of :eqref:`eq_gan_sobolev`, which equals $W_2(p, q)$ to leading order as $q \to p$ by the Benamou--Brenier form of optimal transport (:numref:`sec_mdl-ot-connection`). Below the cutoff the generator therefore keeps a signal when nearby supports move apart. Because the critic's input gradient has units of one over length, $\gamma$ is a squared length, set by the scale of the data: R3GAN uses values from 0.05 on CIFAR-10 to 150 on FFHQ at $256 \times 256$ pixels.

## The R3GAN Recipe

R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` puts the pieces together. Its critic minimizes the pairing loss of :numref:`sec_gan_relativistic` plus both zero-centered penalties, and its generator minimizes the non-saturating form of that loss. The penalties make the equilibrium locally attracting; the pairing contributed the coverage measured in the StackedMNIST ablation above. With this loss R3GAN trains a plain convolutional network without most of the stabilizers that earlier GANs needed (:numref:`sec_dcgan`).

### The Loss in Code

The implementation keeps the losses and the penalties apart. For aligned pairs of a real sample $x$ and a generated sample $x'$, the critic minimizes $\mathrm{softplus}(D(x') - D(x)) = -\log \sigma(D(x) - D(x'))$ and the generator the reversed $\mathrm{softplus}(D(x) - D(x'))$. The functions `gan_loss_D` and `gan_loss_G` give the logistic critic loss and the non-saturating generator loss of :numref:`sec_basic_gan` with the same arguments, so a training loop can take either pair.

```{.python .input #convergence-the-loss-in-code}
%%tab pytorch
#@save
def rpgan_loss_D(critic, real, fake):
    """Relativistic pairing critic loss -E[log sigma(D(x) - D(x'))]."""
    return F.softplus(critic(fake) - critic(real)).mean()

#@save
def rpgan_loss_G(critic, real, fake):
    """Non-saturating pairing loss for the generator."""
    return F.softplus(critic(real) - critic(fake)).mean()

#@save
def gan_loss_D(critic, real, fake):
    """Logistic critic loss -E[log sigma(D(x))] - E[log sigma(-D(x'))]: the
    batch means, where update_D reports the halved sum."""
    return F.softplus(-critic(real)).mean() + F.softplus(critic(fake)).mean()

#@save
def gan_loss_G(critic, real, fake):
    """Non-saturating generator loss -E[log sigma(D(x'))]; real is unused."""
    return F.softplus(-critic(fake)).mean()
```

```{.python .input #convergence-the-loss-in-code}
%%tab jax
#@save
def rpgan_loss_D(critic, real, fake):
    """Relativistic pairing critic loss -E[log sigma(D(x) - D(x'))]."""
    return jax.nn.softplus(critic(fake) - critic(real)).mean()

#@save
def rpgan_loss_G(critic, real, fake):
    """Non-saturating pairing loss for the generator."""
    return jax.nn.softplus(critic(real) - critic(fake)).mean()

#@save
def gan_loss_D(critic, real, fake):
    """Logistic critic loss -E[log sigma(D(x))] - E[log sigma(-D(x'))]: the
    batch means, where update_D reports the halved sum."""
    return (jax.nn.softplus(-critic(real)).mean()
            + jax.nn.softplus(critic(fake)).mean())

#@save
def gan_loss_G(critic, real, fake):
    """Non-saturating generator loss -E[log sigma(D(x'))]; real is unused."""
    return jax.nn.softplus(-critic(fake)).mean()
```

The penalty function returns the per-sample squared input gradients on both batches, before the factor $\gamma/2$. It differentiates the *sum* of the critic's scores with respect to the input batch: each score depends only on its own row, so one backward pass gives every sample's input gradient (batch normalization, which mixes rows, would break this). The gradient stays differentiable, so that training can differentiate the penalty with respect to the critic's parameters; the samples are constants. The critic may be any function from a batch of inputs to a batch of scores, including a closure that fixes a condition (:numref:`sec_gan_conditional`).

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

A training loop alternates one step of each player. `critic_step` takes one optimizer step on the critic loss plus $\gamma/2$ times both penalties, with the generated batch held fixed, and returns the loss and the batch means of $R_1$ and $R_2$, so that a loop can log the critic's steepness. `generator_step` then takes one step on the generator loss against the updated critic. Given labels `y`, both score with the conditional critic $D(\cdot, y)$ (:numref:`sec_gan_conditional`).

```{.python .input #convergence-the-loss-in-code-1}
%%tab pytorch
#@save
def critic_step(net_D, opt_D, loss_D, real, fake, gamma, y=None):
    """One critic step on loss_D + gamma/2 (R1 + R2); return the loss and
    the batch means of R1 and R2."""
    critic = net_D if y is None else lambda x: net_D(x, y)
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

We test the recipe on a standard problem with many modes: 25 Gaussians whose centers form a $5 \times 5$ grid with spacing 2, each with standard deviation $0.05$. A generated sample is *on a mode* if it lies within three widths, $0.15$, of the nearest center, where a 2-D Gaussian puts $1 - e^{-9/2} \approx 98.9\%$ of its mass. For 10,000 samples we report how many modes are covered, counting a mode only if it holds at least a fifth of the on-mode samples it would receive if they were spread evenly, then the fraction of samples on a mode, their mean distance to the nearest center (about $0.06$ for a perfect fit), and $\mathbb{E}_q\|\nabla_x D\|^2$ for the final critic. Both networks are MLPs with two hidden layers of 256 units; the generator maps a 64-dimensional latent vector to the plane, and the critic uses leaky ReLU and no normalization.

```{.python .input #convergence-mode-coverage-on-25-gaussians-1}
%%tab pytorch
def grid_centers(side=5, spacing=2.0):
    coords = (torch.arange(side, dtype=torch.float32) - (side - 1) / 2)
    gx, gy = torch.meshgrid(coords * spacing, coords * spacing, indexing='ij')
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

def sample_modes(centers, n, width=0.05):
    idx = torch.randint(0, centers.shape[0], (n,), device=centers.device)
    return centers[idx] + width * torch.randn(n, 2, device=centers.device)

def make_mlp(sizes, act):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        nn.init.zeros_(layers[-1].bias)      # zero offsets
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)
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

def make_mlp(sizes, act, rngs):    # uniform fan-in weights, zero offsets
    init = nnx.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform')
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nnx.Linear(sizes[i], sizes[i + 1], kernel_init=init,
                                 rngs=rngs))
        if i < len(sizes) - 2:
            layers.append(act)
    return nnx.Sequential(*layers)
```

The training function takes the two losses and $\gamma$ as arguments and alternates `critic_step` and `generator_step`, each on a fresh batch. Both networks use Adam with $\beta_1 = 0$ and $\beta_2 = 0.99$, as R3GAN does, and a learning rate that decays from $2 \cdot 10^{-3}$ to $2 \cdot 10^{-4}$ along a cosine over 10,000 steps with batches of 1,024. It returns the generator and $\mathbb{E}_q\|\nabla_x D\|^2$ from the last critic step, which at $\gamma = 0$ is the only use of the penalties.

```{.python .input #convergence-mode-coverage-on-25-gaussians-3}
%%tab pytorch
def train_toy(centers, loss_D, loss_G, gamma, steps=10000, lr=2e-3,
              batch=1024, latent=64, seed=0):
    torch.manual_seed(seed)
    dev = centers.device
    net_G = make_mlp([latent, 256, 256, 2], nn.ReLU).to(dev)
    net_D = make_mlp([2, 256, 256, 1], lambda: nn.LeakyReLU(0.2)).to(dev)
    opt_G, opt_D = [torch.optim.Adam(net.parameters(), lr, betas=(0.0, 0.99))
                    for net in (net_G, net_D)]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr / 10)
              for opt in (opt_G, opt_D)]
    for _ in range(steps):
        real = sample_modes(centers, batch)
        with torch.no_grad():
            fake = net_G(torch.randn(batch, latent, device=dev))
        _, _, grad_q = critic_step(net_D, opt_D, loss_D, real, fake, gamma)
        z = torch.randn(batch, latent, device=dev)
        generator_step(net_G, opt_G, loss_G, net_D,
                       sample_modes(centers, batch), z)
        for sched in scheds:
            sched.step()
    return net_G, grad_q.item()            # E_q ||grad_x D||^2
```

```{.python .input #convergence-mode-coverage-on-25-gaussians-3}
%%tab jax
def train_toy(centers, loss_D, loss_G, gamma, steps=10000, lr=2e-3,
              batch=1024, latent=64, seed=0):
    rngs = nnx.Rngs(seed)
    net_G = make_mlp([latent, 256, 256, 2], nnx.relu, rngs)
    net_D = make_mlp([2, 256, 256, 1], lambda x: nnx.leaky_relu(x, 0.2), rngs)
    adam = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.1),
                      b1=0.0, b2=0.99)
    opt_G, opt_D = (nnx.Optimizer(net, adam, wrt=nnx.Param)
                    for net in (net_G, net_D))

    @nnx.jit
    def step(net_G, net_D, opt_G, opt_D, key):
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1, (2, batch, latent))    # critic, generator
        real = sample_modes(k2, centers, 2 * batch).reshape(2, batch, 2)
        fake = net_G(z[0])
        _, _, grad_q = critic_step(net_D, opt_D, loss_D, real[0], fake, gamma)
        generator_step(net_G, opt_G, loss_G, net_D, real[1], z[1])
        return grad_q                      # E_q ||grad_x D||^2

    step = nnx.cached_partial(step, net_G, net_D, opt_G, opt_D)
    for key in jax.random.split(rngs(), steps):
        grad_q = step(key)
    return net_G, float(grad_q)
```

:begin_tab:`jax`
One compiled function draws the batches and calls both saved steps. `nnx.cached_partial` binds the networks and optimizers to it once, so the loop does not split and merge the model state at every step.
:end_tab:

We train three runs from the same initialization: the classical GAN without a penalty, and the R3GAN loss at $\gamma = 1$ and at $\gamma = 10$. The low-pass filter predicts how the last two differ. The grid repeats every 2 units, a component with wavenumber $\omega = \pi$, which the penalty damps about 40-fold at $\gamma = 1$ and about 400-fold at $\gamma = 10$. With a ten times weaker signal about where the modes sit, the $\gamma = 10$ generator should resolve the grid less sharply within the same budget.

```{.python .input #convergence-mode-coverage-on-25-gaussians-4}
%%tab pytorch
centers = grid_centers().to(d2l.try_gpu())
arms = {'GAN': (gan_loss_D, gan_loss_G, 0.0),
        'R3GAN, gamma=1': (rpgan_loss_D, rpgan_loss_G, 1.0),
        'R3GAN, gamma=10': (rpgan_loss_D, rpgan_loss_G, 10.0)}
samples, grad_q = {}, {}
for name, (loss_D, loss_G, gamma) in arms.items():
    net_G, grad_q[name] = train_toy(centers, loss_D, loss_G, gamma)
    with torch.no_grad():
        z = torch.randn(10000, 64, device=centers.device)
        samples[name] = net_G(z).cpu().numpy()
grid = centers.cpu().numpy()
```

```{.python .input #convergence-mode-coverage-on-25-gaussians-4}
%%tab jax
centers = grid_centers()
arms = {'GAN': (gan_loss_D, gan_loss_G, 0.0),
        'R3GAN, gamma=1': (rpgan_loss_D, rpgan_loss_G, 1.0),
        'R3GAN, gamma=10': (rpgan_loss_D, rpgan_loss_G, 10.0)}
samples, grad_q = {}, {}
for name, (loss_D, loss_G, gamma) in arms.items():
    net_G, grad_q[name] = train_toy(centers, loss_D, loss_G, gamma)
    z = jax.random.normal(jax.random.key(1), (10000, 64))
    samples[name] = np.asarray(net_G(z))
grid = np.asarray(centers)
```

Each panel shows 3,000 generated samples in orange and the 25 centers in blue.

```{.python .input #convergence-mode-coverage-on-25-gaussians-2}
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

fig, axes = d2l.plt.subplots(1, 3, figsize=(9.5, 3.4))
for ax, (name, x) in zip(axes, samples.items()):
    modes, on, dist = mode_stats(x, grid)
    print(f'{name}: {modes}/25 modes, on-mode {on:.2f}, '
          f'distance {dist:.2f}, E_q|grad D|^2 {grad_q[name]:.1e}')
    # Centers drawn last, on top of the samples
    d2l.plot([x[:3000, 0], grid[:, 0]], [x[:3000, 1], grid[:, 1]],
             fmts=('C1.', 'C0x'), xlim=(-5, 5), ylim=(-5, 5), axes=ax)
    ax.set_title(name), ax.set_aspect('equal')
fig.tight_layout()
```

The R3GAN loss at $\gamma = 1$ fits the grid. Over sixteen runs with different seeds it covered all 25 modes in every run but one, which covered 24, and put most of its samples within three widths of a center. The classical GAN also puts most of its samples on a mode but spreads them less evenly: it covered between 20 and 25 modes, dropped at least one mode in about half the runs, and in its weakest runs left a whole line of five modes nearly empty. The two losses differ more reliably in the critic. With the penalty, $\mathbb{E}_q\|\nabla_x D\|^2$ ends about three to four orders of magnitude below its value for the unpenalized critic, which stays steep at the generated samples even when the fit is good, like the undamped slope $\psi$ of the Dirac-GAN. At $\gamma = 10$ the grid is blurred, as predicted: only an eighth to three fifths of the samples lie on a mode, and their mean distance to the nearest center is 1.5 to 6 times that at $\gamma = 1$. The blur also thins out some modes, mostly near the center of the grid, so the strict count finds between 21 and 25 covered. The equilibrium is still $q = p$; the larger $\gamma$ slows the approach to it (exercise :exref:`ex_convergence-penalty-length`).

## The Games of This Chapter

### Three Choices

Four sections have now derived what each adversarial game measures. Three choices describe all of them (:numref:`tab_gan_games`): what the critic estimates, what it does with the estimate, and how the critic is constrained. The last column records whether the generator keeps a gradient when the supports of $p$ and $q$ are disjoint.

:The games of this chapter, described by three choices: what the critic estimates, what it does with the estimate, and how the critic is constrained. The last column records whether the generator keeps a gradient on disjoint supports.
:label:`tab_gan_games`

| objective | the critic estimates | what it does with the estimate | how the critic is constrained | on disjoint supports |
|:--|:--|:--|:--|:--|
| log loss, $f$-GAN (:numref:`sec_basic_gan`, :numref:`sec_gan_objectives`) | $\log(p/q)$, or $f'(p/q)$ for an $f$-GAN | classifies one sample | not at all | saturates |
| MMD (:numref:`sec_gan_objectives`) | a kernel-smoothed $p - q$, the witness | witnesses a difference of means | a fixed kernel ball | smooth |
| Wasserstein-1 (:numref:`sec_gan_objectives`) | a Kantorovich potential | witnesses a difference of means | a Lipschitz ball | smooth |
| pairing, RpGAN (:numref:`sec_gan_relativistic`) | $\log(p/q)$ up to a constant | ranks a real--generated pair | not at all | saturates |
| RpGAN + $R_1$ + $R_2$ (:numref:`sec_gan_convergence`) | $\log(p/q)$ above the length scale $2\sqrt\gamma$, smoothed below | ranks a real--generated pair | a symmetric Sobolev penalty | smooth, near equilibrium |

### Reading the Table

The classifying and the ranking critics estimate the same log ratio; the loss and the pairing change how its errors are weighted, not what is estimated. The MMD and $W_1$ critics estimate a smoothed difference and a potential, and the penalty moves the pairing critic toward them below the length scale $2\sqrt\gamma$. The constraint decides whether a gradient survives on disjoint supports, and the zero-centered penalty also damps the rotation of the Dirac-GAN. Ranking guarantees coverage only for a fixed coupling; for minibatch training the evidence is the StackedMNIST ablation, where the pairing loss recovered all 1,000 modes and the classical loss 693.

Whether this recipe trains an image generator, and how to score the images it produces, is the question of :numref:`sec_dcgan`.

## Summary

Gradient training of an adversarial game need not reach its equilibrium. On the Dirac-GAN the gradient field is orthogonal to the parameters, so the flow circles the solution and every simultaneous step moves away from it, whatever the loss, its weighting or its pairing. The zero-centered penalties $R_1$ and $R_2$ on the critic's input gradient at data and generated samples damp the rotation for every $\gamma > 0$. Near equilibrium the penalized game measures $p - q$ through a low-pass filter with cutoff length $2\sqrt\gamma$: $2\,\mathrm{JS}$ at larger scales, a linearized Wasserstein-2 distance at smaller ones. R3GAN combines the paired loss with both penalties; on 25 Gaussians it fits the grid at $\gamma = 1$ with a critic that stays flat on generated samples, while $\gamma = 10$ blurs the grid within the same budget.

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
       `trajectory((0.1, 0.1), h, 3000, gamma=0.3)` for $h = 1.1$ and
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
       $\gamma \int m\, \|\nabla_x D\|^2$. Show that the penalized
       best-response value is $2\sup_t \ell(t)$ for every $\theta \neq 0$.
       Which assumption behind :eqref:`eq_gan_local_value` fails here?
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
   :numref:`subsec_gan_25_gaussians`, $\gamma = 10$ blurs the grid that
   $\gamma = 1$ fits.
    1. For data scaled by $c > 0$ and the critic $D_c(x) = D(x / c)$, show
       that the penalized game at weight $c^2\gamma$ has the same value as
       the original game at weight $\gamma$.
    1. With the low-pass weight and $|\ell''(0)| = \tfrac14$, compute by
       what factor the penalty damps a component whose wavelength is the
       mode width $0.05$, at $\gamma = 1$ and at $\gamma = 10$. Compare
       the mean distance printed for $\gamma = 1$ with the $0.06$ of a
       perfect fit. Does the damping predict how sharply each mode is
       resolved?
    1. Train the $\gamma = 10$ run for two and four times as many steps,
       with the cosine schedule stretched to match. Does the on-mode
       fraction approach that of $\gamma = 1$?
   :exlabel:`ex_convergence-penalty-length`
1. [code] **Alternating updates.** ● `train_toy` updates the critic first
   and then the generator against the updated critic. On the Dirac-GAN
   this reads

   $$\psi_{k+1} = \psi_k + h\, u_\psi(\theta_k, \psi_k), \qquad \theta_{k+1} = \theta_k + h\, u_\theta(\theta_k, \psi_{k+1}),$$

   with $(u_\theta, u_\psi)$ the field of :eqref:`eq_gan_dirac_flow`.
    1. Linearize at the equilibrium for $\gamma = 0$ and show that the
       Jacobian of the combined update has determinant one, with
       eigenvalues on the unit circle if and only if $h\, \ell'(0) \leq 2$.
       What does this imply about linear convergence and divergence?
    1. Write an alternating `trajectory` and run it at $\gamma = 0$ from
       $(0.1, 0.1)$ for $10^5$ steps with $h \in \{0.2, 1, 3, 5\}$. Record
       the smallest and largest distance from the equilibrium over the
       last $10^4$ steps, and explain the wide spread at $h = 3$. Repeat at
       $h = 0.2$ with $\gamma = 0.3$.
   :exlabel:`ex_convergence-alternating-updates`
1. [code] **The local value.** ●
    1. Perturb the maximizing critic of :eqref:`eq_gan_local_value` to
       $D + s\,\xi$ with a test function $\xi$ and show that the
       first-order condition in $s$ is
       $\big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big) D = \tfrac{\ell'(0)}{2}(p - q)$,
       assuming that boundary terms vanish.
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
       $h = 0.01$ for 40,000 steps at $\gamma = 1$ and $\gamma = 3$. Where
       does each trajectory end? Explain both outcomes with (a) and (b).
   :exlabel:`ex_convergence-one-centered-dynamics`
1. [code] **Momentum in a rotating game.** ● R3GAN trains with Adam at
   $\beta_1 = 0$. To isolate the first moment, average the field of the
   penalized Dirac-GAN with the logistic loss,
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
       distance with $r^\star$. Check (b) with `flow=True`.
   :exlabel:`ex_convergence-non-saturating-dynamics`

[Discussions](https://d2l.discourse.group/t/36811)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.4]{.kicker}

Gradient Penalties and Convergence<br>
**the Dirac-GAN · circles and spirals · zero-centered penalties · the penalized value · the R3GAN loss**
:::
:::

::: {.slide title="Gradient Steps Double the Distance to a Solution That Exists"}
[The problem]{.kicker}

- Garment GAN: when training stops, the critic's loss is well below
  $\log 2$ and the generator's well above; at equilibrium both equal $\log 2$.
- Two-point game: data at $0$, generated point at $\theta$, critic
  $D(x) = \psi x$. From $(1, 1)$, 300 gradient steps of size 0.2
  **double** the distance to the optimum $(0, 0)$.

::: {.d2l-note .warn}
Knowing where the equilibrium lies does not tell us whether training gets
there. What makes it converge?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- show that the gradient flow of the two-point game rotates, and that
  discrete steps spiral outward for every step size;
- compute the eigenvalues of the penalized game and its three damping regimes;
- evaluate the penalized value near equilibrium and read $\gamma$ as a
  squared length;
- write the R3GAN loss and predict what $\gamma = 1$ and $\gamma = 10$ do on
  25 Gaussians.

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
- **Uses:** the log ratio $\log(p/q)$ as optimal critic; the weights
  $\sigma(D(x'))$; pairing.
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
- Linearized: eigenvalues $\nu_{1,2} = \pm i\,\ell'(0)$, a center.
- The scalar $\ell'(-\psi\theta) = \sigma(D(\theta))$ is the generator's
  weight on the generated sample.

::: {.d2l-note}
A weight scales speed, not direction: reweighting cannot stop the rotation.
:::
:::

::: {.slide title="Every Simultaneous Step Moves Away from the Solution"}
[The Dirac-GAN]{.kicker}

Field $\mathbf{u} \perp (\theta, \psi)$, step size $h$; by Pythagoras

$$\|(\theta, \psi) + h\,\mathbf{u}\|^2 = \|(\theta, \psi)\|^2 + h^2\|\mathbf{u}\|^2$$

- The distance grows at every step, **for every** $h > 0$.
- Update map: eigenvalues $1 \pm i h\ell'(0)$, spectral radius
  $\sqrt{1 + h^2\ell'(0)^2} > 1$.
- Non-saturating generator: same Jacobian; a slow inward drift that
  discrete steps stop on a cycle of radius about $\sqrt{2h}$.
:::

::: {.slide title="Check: Does Pairing Change the Two-Point Dynamics?"}
[Check]{.kicker}

The pairing loss scores the pair $(0, \theta)$ by $\ell\big(D(0) - D(\theta)\big)$.
Does it change the field?

. . .

- $\ell(D(0) - D(\theta)) = \ell(-\psi\theta)$: the classical value minus the
  constant $\ell(0)$.
- Same gradient field, same circles, same outward spiral.
- The cure must change the dynamics, not the loss.
:::

::: {.slide title="Zero-Centered Penalties Charge the Critic's Slope"}
[Penalties]{.kicker}

$$R_1 = \tfrac{\gamma}{2}\,\mathbb{E}_{x \sim p}\|\nabla_x D\|^2, \quad
R_2 = \tfrac{\gamma}{2}\,\mathbb{E}_{x' \sim q}\|\nabla_{x'} D\|^2, \quad
R_1 + R_2 = \gamma\,\mathbb{E}_{m}\|\nabla_x D\|^2$$

- The critic maximizes $V - R_1 - R_2$; $m = (p + q)/2$; gradients with
  respect to the **input**.
- At $q = p$ the optimal critic is flat, so the penalty vanishes at the
  equilibrium.
- One-centered (WGAN-GP), $\mathbb{E}(\|\nabla D\| - 1)^2$: prefers unit
  slope at $q = p$, keeps pushing, and does not converge on two points.
:::

::: {.slide title="The Penalty Turns the Center into a Damped Spring"}
[Penalties]{.kicker}

On the two-point game $\nabla_x D = \psi$, so one penalty adds $-\gamma\psi$
to $\dot\psi$:

$$J_\gamma = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & -\gamma \end{pmatrix}, \qquad
\nu_{1,2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^2}{4} - \ell'(0)^2}$$

- Negative real parts for **every** $\gamma > 0$.
- $\gamma < 2\ell'(0)$: inward spiral; $\gamma = 2\ell'(0)$ ($= 1$, logistic):
  critical; larger: overdamped and slower.
:::

::: {.slide title="Simulation Confirms Circles, Outward Spirals and Damping"}
[Penalties]{.kicker}

@!convergence-phase-portraits-1

Start $(1, 1)$, steps of 0.2: unpenalized steps spiral out to twice the
distance in 300 steps; $\gamma = 0.3$ spirals in; $\gamma = 1$ goes in without circling.
:::

::: {.slide title="R1 Alone Leaves the Critic Free Where Only Generated Samples Lie"}
[Penalties]{.kicker}

- Near equilibrium either penalty suffices; far from it, each smooths one
  distribution.
- With $R_1$ only, input gradients on generated samples grow; training
  diverges.

| StackedMNIST, 1,000 modes | modes | reverse KL |
|:--|--:|--:|
| RpGAN + $R_1$ + $R_2$ | 1,000 | 0.078 |
| GAN + $R_1$ + $R_2$ | 693 | 0.927 |
| either, $R_1$ only | diverged | — |

[Huang et al., 2024]{.small}
:::

::: {.slide title="Near Equilibrium the Penalized Value Runs from 2 JS to a Transport Distance"}
[The penalized value]{.kicker}

$\mathcal{L}_m D = -\nabla\cdot(m\nabla D)$; second-order expansion:

$$\sup_D \Big\{ \ell'(0)\langle p - q, D\rangle - |\ell''(0)|\!\int\! m D^2 - \gamma\!\int\! m\|\nabla D\|^2 \Big\}
= \frac{\ell'(0)^2}{4}\big\langle p - q, (|\ell''(0)|\,m + \gamma \mathcal{L}_m)^{-1}(p - q)\big\rangle$$

- $\gamma \to 0$: $2\,\mathrm{JS}(p, q)$; large $\gamma$:
  $\tfrac{\ell'(0)^2}{4\gamma}W_2^2$.
- Gaussian shift $\mu$: value $\mu^2/(4 + 16\gamma)$, within 1% of the exact
  value; the large-$\gamma$ form is 26× too large at $\gamma = 0.01$.
:::

::: {.slide title="The Penalty Is a Low-Pass Filter with Cutoff 2√γ"}
[The penalized value]{.kicker}

A Fourier component $e^{i\omega x}$ of $p - q$ enters with weight

$$\frac{1}{|\ell''(0)| + \gamma \omega^2} \quad\text{instead of}\quad \frac{1}{|\ell''(0)|}$$

- Logistic loss: length scales $1/\omega$ above $2\sqrt\gamma$ are
  measured by $2\,\mathrm{JS}$, shorter ones by a transport distance.
- The signal does not vanish when nearby supports move apart.
- $\gamma$ has units of **length²**: R3GAN uses 0.05 (CIFAR-10) to 150
  (FFHQ-256).
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
- Penalties: convergence. Pairing: all 1,000 StackedMNIST modes.
:::

::: {.slide title="Check: What Should γ = 10 Do on a Grid with Spacing 2?"}
[Check]{.kicker}

25 Gaussians on a $5 \times 5$ grid, spacing 2, width 0.05. Same budget,
$\gamma = 1$ against $\gamma = 10$. Predict.

. . .

- Grid period 2 ($\omega = \pi$): damped about 40× at $\gamma = 1$,
  400× at $\gamma = 10$.
- The signal about mode positions is ten times weaker at $\gamma = 10$, so
  the grid resolves more slowly.
- The equilibrium is still $q = p$: $\gamma$ sets the speed, not the end.
:::

::: {.slide title="γ = 1 Fits the Grid with a Flat Critic; γ = 10 Blurs It"}
[25 Gaussians]{.kicker}

@!convergence-mode-coverage-on-25-gaussians-2

- $\gamma = 1$: all 25 modes in 15 of 16 runs; critic $10^3$–$10^4\times$
  flatter than the GAN's.
- GAN: 20 to 25 modes across seeds; its weakest runs empty a line of the grid.
- $\gamma = 10$: blurred; an eighth to three fifths of samples on a mode.
:::

::: {.slide title="Three Choices Describe the Games of the First Four Sections"}
[The games]{.kicker}

| objective | critic estimates | does | constrained by | disjoint |
|:--|:--|:--|:--|:--|
| log loss | $\log(p/q)$ | classifies | nothing | saturates |
| MMD | smoothed $p - q$ | witnesses | kernel ball | smooth |
| $W_1$ | potential | witnesses | Lipschitz ball | smooth |
| RpGAN | $\log(p/q)$ + const | ranks | nothing | saturates |
| R3GAN | $\log(p/q)$ above $2\sqrt\gamma$ | ranks | Sobolev penalty | smooth |
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Two-point game: the flow circles; simultaneous steps double the distance
  in 300 steps of 0.2; pairing and reweighting do not help.
- Zero-centered penalties: eigenvalues with negative real part for every
  $\gamma > 0$; critical at $\gamma = 1$ (one penalty, logistic loss).
- Near equilibrium: a low-pass filter on $p - q$ with cutoff length $2\sqrt\gamma$;
  the Gaussian shift agrees to 1%.
- 25 Gaussians: $\gamma = 1$ fits with a flat critic; $\gamma = 10$ blurs.

**Next:** the same loss on images, and how to score what it generates.
:::
