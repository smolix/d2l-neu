# Gradient Penalties and Convergence
:label:`sec_gan_convergence`

Anyone who ran the experiment of :numref:`sec_basic_gan` watched the two losses rise and fall around $\log 2$ instead of settling, while the generated points drifted back and forth. The last three sections computed what the game is worth when the critic answers optimally: a Jensen--Shannon divergence, another divergence or metric, and the divergence between the two orderings of a real--generated pair. Training never gives the critic that chance. Both networks take small gradient steps at the same time, and every step of one player changes the objective the other one faces. Knowing where the equilibrium lies says nothing yet about whether these steps reach it.

:citet:`Mescheder.Geiger.Nowozin.2018` asked which training methods for GANs actually converge. Their answer starts from the smallest game there is: the data sit at one point, the generator places its sample at another, and the critic is a straight line. Even this game defeats plain gradient training, and a penalty on the critic's slope at the data and generated samples repairs it. R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` and the image generators of :numref:`sec_dcgan` train with this penalty.

The pairing objective of :numref:`sec_gan_relativistic` does not address this problem. It changes which configurations the generator can get stuck in, but on separated supports its value is constant, and on the two-point game its gradient field equals the classical one. This section analyzes the dynamics of that game, adds the penalty, asks what the penalized game measures near its equilibrium, and then writes the full R3GAN loss in code and trains it on a grid of 25 Gaussians. How the recipe behaves on images, and how to score the result, is the subject of :numref:`sec_dcgan`.

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

The smallest game has one real point, one generated point and a linear critic. It is small enough to solve exactly, and it already shows both obstacles of this section: the value of the game is flat in the generator's position, and the gradient dynamics rotate instead of converging.

### The Game on Two Points

Place the data at the origin and let the generator produce a single point on the real line: $p = \delta_0$ and $q_\theta = \delta_\theta$. The generator's parameter $\theta$ is the position of its point mass, and $q = p$ exactly when $\theta = 0$. Let the critic be linear, $D_\psi(x) = \psi x$, with slope $\psi$. :numref:`sec_gan_objectives` scored a real sample $x$ by $\ell(D(x))$ and a generated sample $x'$ by $\ell(-D(x'))$ for a classification loss $\ell$, giving the margin objective

$$
V_\ell(D) = \mathbb{E}_{x \sim p}\big[\ell(D(x))\big] + \mathbb{E}_{x' \sim q}\big[\ell(-D(x'))\big].
$$

For two point masses both expectations are single evaluations:

$$
V(\theta, \psi) \;=\; \ell\big(D_\psi(0)\big) + \ell\big({-D_\psi(\theta)}\big)
\;=\; \ell(0) + \ell(-\psi\theta).
$$
:eqlabel:`eq_gan_dirac_value`

This is the *Dirac-GAN* of :citet:`Mescheder.Geiger.Nowozin.2018`. Any differentiable loss with $\ell'(0) > 0$ works. We use the logistic loss $\ell = \log\sigma$ as the running example; its derivatives are $\ell'(t) = \sigma(-t)$ and $\ell''(t) = -\sigma(t)\sigma(-t)$, so $\ell'(0) = \tfrac12$ and $\ell''(0) = -\tfrac14$.

For every $\theta \neq 0$ the two point masses have disjoint supports. The critic's best response drives $\ell(-\psi\theta)$ toward its supremum by sending $\psi\theta \to -\infty$, so $\sup_\psi V = \ell(0) + \sup_t \ell(t)$, the same number for every $\theta \neq 0$. A value that does not depend on $\theta$ gives the generator no gradient, however close to the data it starts. This is the saturation of :numref:`sec_basic_gan` in its smallest form.

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

with purely imaginary eigenvalues: the equilibrium is a center, not an attractor. Near it the parameters rotate at angular speed $\ell'(0)$, and the generated point passes the data again and again without settling there.

The common factor has a meaning from earlier sections. For the logistic loss, $\ell'(-\psi\theta) = \sigma(\psi\theta) = \sigma(D_\psi(\theta))$ is the weight $w_{\mathrm{sat}}(x') = \sigma(D(x'))$ with which the saturating generator loss of :numref:`sec_basic_gan` scales the update of a generated sample. Because the one real point scores $D_\psi(0) = 0$, it is also the rank weight $w_{\mathrm{Rp}}(x') = \mathbb{E}_{x \sim p}[\sigma(D(x') - D(x))]$ of :numref:`sec_gan_relativistic`, the probability that the generated point outranks a real one. The weight multiplies both components of the field, so it sets how fast the parameters turn and not where they go. No reweighting of this kind can turn the circle into a spiral toward the solution.

### Outward Spirals in Discrete Time

Discrete steps do worse than the flow. Write $\mathbf{u}(\theta, \psi)$ for the right-hand side of :eqref:`eq_gan_dirac_flow`. A simultaneous step of size $h > 0$ maps $(\theta, \psi)$ to $(\theta, \psi) + h\,\mathbf{u}$. Since $\mathbf{u}$ is orthogonal to $(\theta, \psi)$, the Pythagorean theorem gives

$$
\big\| (\theta, \psi) + h\, \mathbf{u} \big\|^2
\;=\; \big\| (\theta, \psi) \big\|^2 + h^2 \big\| \mathbf{u} \big\|^2 ,
$$

so each step adds $h^2\|\mathbf{u}\|^2$ to the squared distance. For the logistic loss $\ell' > 0$ everywhere, so $\mathbf{u}$ vanishes only at the equilibrium, and every step from any other point moves away from the solution. A smaller step only slows the outward spiral. Exercise :exref:`ex_convergence-discrete-simultaneous-update-dirac`a reaches the same conclusion from the Jacobian of the update map, whose spectral radius $\sqrt{1 + h^2 \ell'(0)^2}$ exceeds one for every $h$. Updating the two players in turn instead of simultaneously does not restore convergence either :cite:`Mescheder.Geiger.Nowozin.2018` (exercise :exref:`ex_convergence-alternating-updates`).

The two repairs of the earlier sections leave these dynamics in place. The non-saturating generator replaces the weight in the first component by $\ell'(\psi\theta)$. Its Jacobian at the equilibrium is still $J$; away from it the field gains a slow inward drift, which for the logistic loss shrinks the squared radius only like $8/t$ in continuous time, and with steps of size $h$ ends on a cycle of radius $\sqrt{2h}$ (exercise :exref:`ex_convergence-non-saturating-dynamics`). The pairing objective scores the pair by $\ell\big(D_\psi(0) - D_\psi(\theta)\big) = \ell(-\psi\theta)$, which differs from :eqref:`eq_gan_dirac_value` only by the constant $\ell(0)$, so its gradient field, its circles and its spirals are the classical ones :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. A cure has to change the dynamics themselves.

## Zero-Centered Penalties

The equilibrium attracts the iterates only if the eigenvalues of the linearized dynamics have negative real parts. The rotation needs friction, and the natural place to add it is the critic's slope, the quantity $\psi$ that keeps the rotation going. Define the two *zero-centered gradient penalties*

$$
R_1 \;=\; \frac{\gamma}{2}\, \mathbb{E}_{x \sim p}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad
R_2 \;=\; \frac{\gamma}{2}\, \mathbb{E}_{x' \sim q}\Big[ \big\| \nabla_{x'} D(x') \big\|^2 \Big],
$$
:eqlabel:`eq_gan_r1r2`

with weight $\gamma > 0$, and let the critic maximize $V_\ell(D) - R_1 - R_2$ while the generator's objective stays unchanged :cite:`Roth.Lucchi.Nowozin.ea.2017,Mescheder.Geiger.Nowozin.2018`. The penalties act on the critic's gradient with respect to its *input*, not its parameters. $R_1$ limits how steeply the critic varies at data samples and $R_2$ at generated samples. They are called zero-centered because the flat critic $\nabla_x D = 0$ minimizes both. At $q = p$ the optimal critic of :numref:`sec_basic_gan` is the constant $D^\star = \log(p/q) = 0$, so the penalties vanish at the equilibrium they are meant to stabilize. Since $\tfrac12(\mathbb{E}_p + \mathbb{E}_q) = \mathbb{E}_m$, their sum is

$$
R_1 + R_2 \;=\; \gamma\, \mathbb{E}_{x \sim m}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad m = \frac{p + q}{2},
$$
:eqlabel:`eq_gan_r1r2_sum`

a single smoothness penalty under the balanced mixture that defines the log-loss game. It constrains the critic wherever either distribution puts mass.

### Damping the Dirac-GAN

For the linear critic, $\nabla_x D_\psi = \psi$ at every $x$, so both penalties equal $\tfrac{\gamma}{2}\psi^2$; $R_1$ and $R_2$ coincide in this example, although they differ for general critics. Subtracting one penalty from the critic's objective adds $-\gamma\psi$ to the critic's update:

$$
\dot\theta = \psi\,\ell'(-\psi\theta),
\qquad
\dot\psi = -\theta\,\ell'(-\psi\theta) - \gamma\psi .
$$

The display uses one penalty; using both replaces $\gamma$ by $2\gamma$. The penalty changes only the lower-right entry of the Jacobian:

$$
J_\gamma = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & -\gamma \end{pmatrix},
\qquad
\nu_{1,2} \;=\; -\frac{\gamma}{2} \pm \sqrt{ \frac{\gamma^2}{4} - \ell'(0)^2 } .
$$
:eqlabel:`eq_gan_dirac_pen`

Both eigenvalues have negative real parts for every $\gamma > 0$. The equilibrium attracts, and gradient steps that are small enough converge to it at a linear rate :cite:`Mescheder.Geiger.Nowozin.2018`; in the underdamped regime "small enough" means $h < \gamma/\ell'(0)^2$ (exercise :exref:`ex_convergence-discrete-simultaneous-update-dirac`b). The eigenvalues distinguish three regimes, exactly as for a damped spring. For $\gamma < 2\ell'(0)$ the square root is imaginary and trajectories spiral inward. At $\gamma = 2\ell'(0)$ the repeated real eigenvalue gives critical damping, the fastest approach without oscillation; for the logistic loss this is $\gamma = 1$. Larger values overdamp the dynamics, and the slower eigenvalue, close to $-\ell'(0)^2/\gamma$, makes convergence slower as $\gamma$ grows.

### Phase Portraits

We check the circles, the outward spiral and the damping by simulating the two coupled scalar equations. For the logistic loss the weight is $\ell'(-\psi\theta) = \sigma(\psi\theta)$. The function `trajectory` either integrates the flow with small fourth-order Runge--Kutta steps or applies simultaneous gradient steps of size $h$.

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

All three panels start at $(\theta, \psi) = (1, 1)$, marked in red; the cross marks the equilibrium. The first panel follows the flow, the second takes 300 simultaneous steps of size $h = 0.2$ with the flow's circle dashed for reference, and the third adds the penalty at $\gamma = 0.3$, in the underdamped regime, and at $\gamma = 1$, the critical value.

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

The portraits agree with the eigenvalues. The integrated flow retraces one circle, as the conservation law requires. The discrete iterates leave the dashed circle at once and move outward on every revolution; after 300 steps their distance from the equilibrium has doubled. Both penalized trajectories converge: the underdamped one along a shrinking spiral, the critically damped one after a single turn without further rotation. The penalty changes divergence into convergence and leaves the equilibrium where it was.

### One-Centered versus Zero-Centered

WGAN-GP centers its penalty differently :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017`. To enforce the Lipschitz constraint of :numref:`sec_gan_objectives` it penalizes $\mathbb{E}\big[(\|\nabla_x D\| - 1)^2\big]$ at points interpolated between real and generated samples. This *one-centered* penalty is motivated by optimal $W_1$ critics, which have unit slope along transport rays. At $q = p$, however, the optimal critic is constant. The zero-centered penalty prefers that critic; the one-centered penalty prefers a unit slope, so the WGAN-GP critic can keep a nonzero gradient at equilibrium and keep pushing a generator that already matches the data. On the Dirac-GAN these dynamics do not converge to the equilibrium :cite:`Mescheder.Geiger.Nowozin.2018` (exercise :exref:`ex_convergence-one-centered-dynamics`).

| penalty | critic it prefers at $q = p$ | Dirac-GAN |
|:---|:---|:---|
| one-centered, $\mathbb{E}\big[(\|\nabla_x D\| - 1)^2\big]$ | unit slope | does not converge |
| zero-centered, $\mathbb{E}_m\big[\|\nabla_x D\|^2\big]$ | flat | converges for every $\gamma > 0$ |

### One Penalty or Two

The Dirac-GAN cannot tell $R_1$ from $R_2$, because a linear critic has the same slope everywhere. Near equilibrium either penalty suffices: when the supports of $p$ and $q$ nearly coincide, controlling the critic under one measure also controls it under the other. The local convergence result of :citet:`Mescheder.Geiger.Nowozin.2018` therefore needs only one penalty.

Far from equilibrium the two are not interchangeable. :citet:`Roth.Lucchi.Nowozin.ea.2017` showed that, up to a weighting factor and a higher-order error, penalizing $\mathbb{E}_p\big[\|\nabla_x D\|^2\big]$ changes the game as if $p$ were convolved with Gaussian noise of covariance proportional to $\gamma$; $R_2$ smooths $q$ in the same way. Smoothing two distributions with disjoint supports makes their densities overlap, so their ratio is finite and the divergence responds to the distance between the supports, without perturbing a single sample. The argument needs both distributions smoothed. With $R_1$ alone the critic is unconstrained where generated samples lie and real ones do not, and :citet:`Huang.Gokaslan.Kuleshov.ea.2024` observe exactly this: the critic's input gradients on generated samples grow and training diverges.

R3GAN's ablation on StackedMNIST, a synthetic dataset of stacked MNIST digits with 1,000 known modes :cite:`Metz.Poole.Pfau.ea.2017`, separates the two effects:

| loss | modes recovered (of 1,000) | reverse KL |
|:---|---:|---:|
| RpGAN with $R_1$ and $R_2$ | 1,000 | 0.078 |
| GAN with $R_1$ and $R_2$ | 693 | 0.927 |
| RpGAN with $R_1$ only | diverged | --- |
| GAN with $R_1$ only | diverged | --- |

With $R_1$ alone both losses diverged early, and for the paired loss no $\gamma$ from 0.1 to 100 prevented it. With both penalties both losses trained, and the paired loss recovered every mode :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. In this ablation the penalties decide convergence and the pairing decides coverage. These are single runs on one synthetic dataset, and StyleGAN2 trained well on faces with $R_1$ alone :cite:`Karras.Laine.Aittala.ea.2020`, so whether one penalty suffices depends on the data and the other training choices. The smoothing argument and the ablation both favor the symmetric penalty, at the cost of one more term in the backward pass.

## The Penalized Value Near Equilibrium

The eigenvalues say how training approaches the equilibrium. What does the penalized game measure on the way? Without the penalty, the answer was a divergence of the density ratio, which is blind to how far apart separated supports are. With it, the answer near equilibrium is a blend of that ratio divergence with a transport distance, and the penalty weight decides which length scales fall on which side.

### The Local Value

Near $q = p$ the optimal critic is close to the constant zero, so we expand the payoffs to second order in the critic. Write $\langle p - q, D \rangle = \mathbb{E}_p[D] - \mathbb{E}_q[D]$, and let $a = \ell'(0) > 0$ and $\ell''(0) < 0$ be the first two derivatives of the loss. Up to third-order terms,

$$
\begin{aligned}
V_\ell(D) &= 2\ell(0) + a\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2, \\
\Phi(D) &= \ell(0) + a\, \langle p - q,\, D \rangle - |\ell''(0)|\, \mathrm{Var}_m[D],
\end{aligned}
$$

where $\Phi$ is the pairing objective of :numref:`sec_gan_relativistic`. Shifting the margin critic by a constant leaves the linear term unchanged and turns $\int m D^2$ into $\mathrm{Var}_m[D]$ at the best shift, so the two payoffs have the same supremum up to their constants. The quadratic term looks small, but it is not: the best penalized critic has size proportional to $p - q$, which makes all three terms of the same order at the optimum.

**Proposition.** *Let $\mathcal{L}_m D = -\nabla \cdot (m \nabla D)$. Near equilibrium, the penalized critic problem has the value*

$$
\sup_{D} \Big\{ a\, \langle p - q,\, D \rangle - |\ell''(0)| \int m\, D^2 - \gamma \int m\, \| \nabla_x D \|^2 \Big\}
\;=\; \frac{a^2}{4}\, \big\langle p - q,\, \big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big)^{-1} (p - q) \big\rangle,
$$
:eqlabel:`eq_gan_local_value`

*attained by the critic that solves $\big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big) D = \tfrac{a}{2}(p - q)$.*

**Proof.** Write $\mathcal{A} = |\ell''(0)|\, m + \gamma \mathcal{L}_m$; integrating by parts, the two quadratic terms are $\langle D, \mathcal{A} D \rangle$, and $\mathcal{A}$ is symmetric and positive. Along a ray $\{t D_0\}$ the objective $t\,a\langle p - q, D_0\rangle - t^2 \langle D_0, \mathcal{A} D_0\rangle$ is a concave parabola in $t$ with maximum $a^2 \langle p - q, D_0\rangle^2 / (4\langle D_0, \mathcal{A} D_0 \rangle)$. By the Cauchy--Schwarz inequality in the inner product $\langle \cdot, \mathcal{A}\, \cdot \rangle$, the largest value over directions is $\tfrac{a^2}{4}\langle p - q, \mathcal{A}^{-1}(p - q)\rangle$, reached at $D_0 \propto \mathcal{A}^{-1}(p - q)$. $\blacksquare$

The value is a weighted squared size of the difference $p - q$, and the weighting operator $\mathcal{A}$ has two parts. Its two limits show what each part contributes. As $\gamma \to 0$, $\mathcal{A}^{-1}$ divides by $|\ell''(0)|\, m$, and the value becomes $\tfrac{a^2}{4|\ell''(0)|}\int (p - q)^2/m$; for the logistic loss this is $\tfrac14 \int (p-q)^2/m$, the local form of $2\,\mathrm{JS}(p, q)$, the unpenalized value of :numref:`sec_basic_gan`. When $\gamma$ is large, the penalty dominates and the value approaches

$$
\frac{a^2}{4\gamma}\, \big\| p - q \big\|^2_{\dot H^{-1}(m)},
\qquad
\|v\|_{\dot H^{-1}(m)} = \sup\Big\{ \langle v, D\rangle : \int m \|\nabla_x D\|^2 \leq 1 \Big\},
$$
:eqlabel:`eq_gan_sobolev`

a squared dual Sobolev norm. The supremum runs over critics modulo constants, which do not matter because $\int (p - q) = 0$ and a constant critic has no gradient.

### A Gaussian Shift

The smallest case is the pair of Gaussians from :numref:`sec_basic_gan`, $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$ with a small shift $\mu$. To first order in $\mu$, $q(x) = p(x - \mu) \approx p(x) + \mu x\, p(x)$, so $p - q \approx -\mu\, x\, m(x)$. A straight-line critic $D(x) = c\,x$ solves the equation of the proposition, because $\mathcal{L}_m x = -(m \cdot 1)' = x\, m(x)$ when $m$ is the standard normal density, as it is to leading order. Substituting gives $c\,(|\ell''(0)| + \gamma)\, x\, m = -\tfrac{a}{2}\mu\, x\, m$, so

$$
D(x) = -\frac{a\,\mu}{2\,(|\ell''(0)| + \gamma)}\; x,
\qquad
\text{value} = \frac{a^2 \mu^2}{4\,(|\ell''(0)| + \gamma)} = \frac{\mu^2}{4 + 16\gamma}
\ \text{for the logistic loss}.
$$

The penalized critic is the unpenalized one, the log ratio $-\mu x$ up to a constant, with its slope shrunk by the factor $1 + 4\gamma$. Both limits are visible: at $\gamma = 0$ the value is $\mu^2/4 = 2\,\mathrm{JS}$ to leading order, and for large $\gamma$ it is $\mu^2/(16\gamma) = \tfrac{a^2}{4\gamma} W_2(p, q)^2$, since the Wasserstein-2 distance of a shift is $|\mu|$. At $\mu = 0.2$ and $\gamma = 0.01$ the value is about $0.0096$, whereas the large-$\gamma$ form :eqref:`eq_gan_sobolev` would give $0.25$.

Both the expansion and the shift approximation drop higher-order terms. To check them, we solve the full penalized problem, with the logistic loss and no expansion, for a critic tabulated on a grid. The objective is concave in the critic values, so Newton's method converges in a few steps.

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

The local formula matches the exact value to within one percent at every $\gamma$, including $\gamma = 0$, where both equal $2\,\mathrm{JS}$. The large-$\gamma$ form is 26 times too large at $\gamma = 0.01$, a quarter too large at $\gamma = 1$, and within three percent at $\gamma = 10$.

### A Low-Pass Filter

The Gaussian shift has a single length scale. A general difference $p - q$ has many, and the operator $\mathcal{A}$ treats them differently. Take $m$ roughly constant over a region and decompose $p - q$ there into Fourier components $e^{i \omega x}$. The operator $\mathcal{L}_m$ multiplies such a component by $m \omega^2$, so in :eqref:`eq_gan_local_value` the component with wavenumber $\omega$ enters with weight

$$
\frac{1}{|\ell''(0)| + \gamma \omega^2}
\qquad \text{instead of} \qquad \frac{1}{|\ell''(0)|}.
$$

Components that vary over lengths $1/\omega$ longer than $\sqrt{\gamma / |\ell''(0)|}$, which is $2\sqrt\gamma$ for the logistic loss, keep their unpenalized weight; shorter ones are damped by a factor $|\ell''(0)|/(\gamma \omega^2)$. The penalty is a low-pass filter on $p - q$ with cutoff length $2\sqrt\gamma$. Above the cutoff the game measures the ratio divergence $2\,\mathrm{JS}$; below it, the transport-type quantity of :eqref:`eq_gan_sobolev`. This makes exact, near equilibrium, the smoothing reading of :citet:`Roth.Lucchi.Nowozin.ea.2017` given above for one penalty or two.

The transport reading of the small scales comes from the dynamic form of optimal transport. The Benamou--Brenier theorem (:numref:`sec_mdl-ot-connection`) writes $W_2(p, q)^2$ as the least kinetic energy $\int m \|\nabla U\|^2$ of a velocity field $\nabla U$ that carries one distribution into the other. Between two nearby distributions this velocity field solves the continuity equation $(p - q) + \nabla \cdot (m \nabla U) = 0$, which is the Poisson equation $\mathcal{L}_m U = p - q$ of the large-$\gamma$ critic, and its kinetic energy is $\|p - q\|^2_{\dot H^{-1}(m)}$. Hence $W_2(p, q) = \|p - q\|_{\dot H^{-1}(m)}$ to leading order as $q \to p$: the dual Sobolev norm is the local metric of optimal transport.

Below the cutoff, then, the game measures a distance that keeps responding when nearby supports move apart, and the generator receives a smooth signal that does not vanish. This is a statement about the value of the game, not a convergence proof; convergence came from the eigenvalues. It is also local, since it expands around a constant critic, and formal, since it assumes that $m$ has a density and that the Poisson equation has a solution.

## The R3GAN Recipe

R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` puts the pieces of the last two sections together. Its critic minimizes the pairing loss of :numref:`sec_gan_relativistic`, its generator uses the non-saturating form of that loss, and the critic subtracts both zero-centered penalties. The two parts address different failures. When each real sample is paired with one generated sample by a fixed coupling, the paired loss has no mode-dropping local minima in the finite-sample best-response landscape :cite:`Sun.Fang.Schwing.2020`; redrawing the pairs at every step averages it into the all-pairs loss, which has such minima again (:numref:`sec_gan_relativistic`). The penalties make the equilibrium locally attracting. With this loss, R3GAN trains a plain convolutional network without most of the stabilizers that earlier GANs relied on, which :numref:`sec_dcgan` takes up.

### The Loss in Code

The implementation keeps the critic loss, the generator loss and the penalties apart. For a real sample $x$ and a generated sample $x'$, the critic minimizes the pairing loss $\mathrm{softplus}(D(x') - D(x)) = -\log \sigma(D(x) - D(x'))$, averaged over aligned pairs of the two batches. The generator reverses the ranking and minimizes $\mathrm{softplus}(D(x) - D(x'))$, the non-saturating form that R3GAN's reference implementation trains with (:numref:`sec_gan_relativistic`). The functions `gan_loss_D` and `gan_loss_G` give the classical logistic critic loss and the non-saturating generator loss of :numref:`sec_basic_gan` with the same arguments, so a training loop can take either pair as an argument.

The penalty function returns the unscaled per-sample squared input gradients on both batches; the caller takes the batch mean and applies $\gamma/2$, so the same function serves one penalty or both. It differentiates the *sum* of the critic's scores with respect to the input batch. Because each score depends only on its own input row, one backward pass gives every sample's input gradient; batch normalization would break this (exercise :exref:`ex_convergence-batch-mixing-critics`). The gradient stays differentiable, so that training can differentiate the penalty with respect to the critic's parameters, and the samples are treated as constants. The critic may be any function from a batch of inputs to a batch of scores, including a closure that fixes a condition (:numref:`sec_gan_conditional`).

:begin_tab:`pytorch`
`create_graph=True` keeps the graph of the input gradient for the second differentiation, and `detach` makes the samples constants.
:end_tab:

:begin_tab:`jax`
`jax.grad` of the summed scores gives the input gradients; differentiating the penalty with respect to the critic nests a second gradient around it, and `jax.lax.stop_gradient` makes the samples constants.
:end_tab:

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

R3GAN evaluates both penalties at every critic step. Applying them only every few steps, as StyleGAN2 does to save compute, made training fail on StackedMNIST and on two-dimensional toy problems in R3GAN's ablation :cite:`Huang.Gokaslan.Kuleshov.ea.2024`.

### The Penalty Weight as a Length

The critic's output is a logit, a pure number, while its input gradient has units of one over length. The weight $\gamma$ in $\gamma\,\mathbb{E}_m\|\nabla_x D\|^2$ therefore has units of length squared, and the low-pass analysis made this concrete: the critic stops resolving $p - q$ below about $2\sqrt\gamma$. A tuned $\gamma$ reflects the scale of the data rather than a property of the method. R3GAN uses values from 0.05 on CIFAR-10 to 150 on FFHQ-256, and its training code starts from these values and anneals them tenfold over training. A $\gamma$ that is too small leaves the dynamics lightly damped; one that is too large hides structure finer than $2\sqrt\gamma$ from the critic, and the generator resolves that structure slowly. The quantity to monitor is $\mathbb{E}_q\|\nabla_x D\|^2$, which $R_2$ penalizes: its growth on generated samples is the signature of the $R_1$-only failure.

### Mode Coverage on 25 Gaussians
:label:`subsec_gan_25_gaussians`

We test the recipe on a standard small problem with many modes: a mixture of 25 Gaussians whose centers form a $5 \times 5$ grid with spacing 2, each with standard deviation $0.05$ in both coordinates. A generated sample is *on a mode* if it lies within three widths, $0.15$, of the nearest center; a 2-D Gaussian puts $1 - e^{-9/2} \approx 98.9\%$ of its mass there. For 10,000 samples we report the number of modes with at least one sample on them, the fraction of samples on a mode, their mean distance to the nearest center (about $0.06$ for a perfect fit), and $\mathbb{E}_q\|\nabla_x D\|^2$ for the final critic.

The networks are small MLPs: the generator maps a 64-dimensional latent vector through two hidden layers of 256 units to the plane, and the critic has the same widths, leaky ReLU activations and no normalization. Both tabs initialize them alike, with PyTorch's default weights and zero offsets.

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
        nn.init.zeros_(layers[-1].bias)      # zero offsets, as in the JAX tab
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

def make_mlp(sizes, act, rngs):    # PyTorch's default weights, zero offsets
    init = nnx.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform')
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nnx.Linear(sizes[i], sizes[i + 1], kernel_init=init,
                                 rngs=rngs))
        if i < len(sizes) - 2:
            layers.append(act)
    return nnx.Sequential(*layers)
```

The training loop takes the two losses and $\gamma$ as arguments, so all three runs below go through one function. Each step updates the critic, with the penalties at weight $\gamma/2$, and then the generator against the updated critic. Both networks use Adam with $\beta_1 = 0$ and $\beta_2 = 0.99$, as R3GAN does, and a learning rate that decays from $2 \cdot 10^{-3}$ to $2 \cdot 10^{-4}$ along a cosine over 10,000 steps with batches of 1,024. The loop returns the generator and $\mathbb{E}_q\|\nabla_x D\|^2$ from the last critic step; with $\gamma = 0$ the penalties are computed only for this measurement.

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
        r1, r2 = r1_r2_penalty(net_D, real, fake)
        loss = loss_D(net_D, real, fake) + gamma / 2 * (r1 + r2).mean()
        opt_D.zero_grad(), loss.backward(), opt_D.step()
        fake = net_G(torch.randn(batch, latent, device=dev))
        loss = loss_G(net_D, sample_modes(centers, batch), fake)
        opt_G.zero_grad(), loss.backward(), opt_G.step()
        for sched in scheds:
            sched.step()
    return net_G, r2.mean().item()         # E_q ||grad_x D||^2
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
    opt_G = nnx.Optimizer(net_G, adam, wrt=nnx.Param)
    opt_D = nnx.Optimizer(net_D, adam, wrt=nnx.Param)

    @nnx.jit
    def step(net_G, net_D, opt_G, opt_D, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        real = sample_modes(k1, centers, batch)
        fake = net_G(jax.random.normal(k2, (batch, latent)))
        def critic_loss(net_D):
            r1, r2 = r1_r2_penalty(net_D, real, fake)
            loss = loss_D(net_D, real, fake) + gamma / 2 * (r1 + r2).mean()
            return loss, r2.mean()         # E_q ||grad_x D||^2
        (_, grad_q), grads = nnx.value_and_grad(critic_loss,
                                                has_aux=True)(net_D)
        opt_D.update(net_D, grads)
        z = jax.random.normal(k3, (batch, latent))
        real = sample_modes(k4, centers, batch)
        opt_G.update(net_G, nnx.grad(
            lambda net_G: loss_G(net_D, real, net_G(z)))(net_G))
        return grad_q

    step = nnx.cached_partial(step, net_G, net_D, opt_G, opt_D)
    key = jax.random.key(seed)
    for _ in range(steps):
        key, subkey = jax.random.split(key)
        grad_q = step(subkey)
    return net_G, float(grad_q)
```

:begin_tab:`jax`
`nnx.cached_partial` binds the networks and optimizers to the compiled step once, so the Python loop pays no per-step cost for splitting and merging the model state.
:end_tab:

We train three runs from the same initialization: the classical GAN of :numref:`sec_basic_gan` without a penalty, and the R3GAN loss at $\gamma = 1$ and at $\gamma = 10$. The low-pass analysis gives a prediction before we run them. At $\gamma = 1$ the cutoff $2\sqrt\gamma$ equals the grid spacing; at $\gamma = 10$ it is about $6.3$, close to the width of the whole grid, so the critic's signal about where the individual modes sit is damped about ten times more, and within the same budget the generator should resolve the grid less sharply.

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

The statistics are the same computation in both tabs, on NumPy arrays. Each panel shows 3,000 generated samples in orange and the 25 centers in blue.

```{.python .input #convergence-mode-coverage-on-25-gaussians-2}
%%tab pytorch, jax
def mode_stats(x, centers, width=0.05):
    """Modes reached, fraction within three widths, mean distance."""
    dist = np.linalg.norm(x[:, None] - centers[None], axis=-1)
    near, idx = dist.min(axis=1), dist.argmin(axis=1)
    on = near <= 3 * width
    return len(np.unique(idx[on])), on.mean(), near.mean()

fig, axes = d2l.plt.subplots(1, 3, figsize=(9.5, 3.4))
for ax, (name, x) in zip(axes, samples.items()):
    modes, on, dist = mode_stats(x, grid)
    print(f'{name}: {modes}/25 modes, on-mode {on:.2f}, '
          f'distance {dist:.2f}, E_q|grad D|^2 {grad_q[name]:.1e}')
    d2l.plot([grid[:, 0], x[:3000, 0]], [grid[:, 1], x[:3000, 1]],
             fmts=('C0x', 'C1.'), xlim=(-5, 5), ylim=(-5, 5), axes=ax)
    ax.set_title(name), ax.set_aspect('equal')
fig.tight_layout()
```

The R3GAN loss at $\gamma = 1$ fits the grid: it reaches all 25 modes and puts roughly three quarters or more of its samples within three widths of a center. On this small grid the classical GAN reaches the modes about as well. What separates the two is the critic. With the penalty, $\mathbb{E}_q\|\nabla_x D\|^2$ ends three to four orders of magnitude below its value for the unpenalized critic, which stays steep at the generated samples even when the fit is good, like the undamped slope $\psi$ of the Dirac-GAN. At $\gamma = 10$ the generator reaches every mode but leaves the grid blurred, with fewer than half of its samples on a mode and at least twice the mean distance, as predicted. The penalized game still has its equilibrium at $q = p$; the larger $\gamma$ slows the resolution of fine structure, and exercise :exref:`ex_convergence-penalty-length` asks whether more steps recover it. A grid this small cannot show the coverage advantage of pairing; that evidence is the StackedMNIST ablation above.

## The Games of This Chapter

Four sections have now derived what each adversarial game measures. Three choices describe all of them (:numref:`tab_gan_games`): what the critic estimates, what it does with the estimate, and how the critic is constrained. The last column records the property that the opening problem of :numref:`sec_basic_gan` tests: whether the generator keeps a gradient when the supports of $p$ and $q$ are disjoint.

:The games of this chapter, described by three choices: what the critic estimates, what it does with the estimate, and how the critic is constrained. The last column records whether the generator keeps a gradient on disjoint supports.
:label:`tab_gan_games`

| objective | the critic estimates | what it does with the estimate | how the critic is constrained | on disjoint supports |
|:--|:--|:--|:--|:--|
| log loss, $f$-GAN (:numref:`sec_basic_gan`, :numref:`sec_gan_objectives`) | $\log(p/q)$, or $f'(p/q)$ for an $f$-GAN | classifies one sample | not at all | saturates |
| MMD (:numref:`sec_gan_objectives`) | a kernel-smoothed $p - q$, the witness | witnesses a difference of means | a fixed kernel ball | smooth |
| Wasserstein-1 (:numref:`sec_gan_objectives`) | a Kantorovich potential | witnesses a difference of means | a Lipschitz ball | smooth |
| pairing, RpGAN (:numref:`sec_gan_relativistic`) | $\log(p/q)$ up to a constant | ranks a real--generated pair | by the data, through the comparison with real scores | saturates |
| RpGAN + $R_1$ + $R_2$ (:numref:`sec_gan_convergence`) | $\log(p/q)$ above the scale $2\sqrt\gamma$, smoothed below | ranks a real--generated pair | a symmetric Sobolev penalty | smooth, near equilibrium |

Read down the first column. The classification and ranking critics estimate the same object, the log ratio $\log(p/q)$; the loss and the pairing change how the estimate is parameterized and how its errors are weighted, not what is estimated. The metric critics estimate a smoothed difference $p - q$ instead, and the penalty moves the pairing critic between the two: a log ratio for features larger than $2\sqrt\gamma$, a smoothed difference below.

The second column decides whether a configuration that drops modes can lower the loss. A classifier scores each generated sample against a fixed zero level, while a ranker scores it against the whole real population, and with one real sample paired to each generated sample the paired loss has no mode-dropping local minima. The third column decides what happens when the supports separate, and whether gradient training converges: an unconstrained critic saturates, a fixed ball or a Sobolev penalty keeps a gradient, and the zero-centered penalty also damps the rotation of the Dirac-GAN. R3GAN estimates a ratio, ranks, and is penalized by a symmetric Sobolev term; its second choice addresses coverage and its third convergence, and the StackedMNIST ablation is consistent with the two being independent. The conditional game of :numref:`sec_gan_conditional` adds a condition to the first column and leaves the other two unchanged.

The reader can now write the R3GAN loss, predict its dynamics on the Dirac-GAN, and read $\gamma$ as a squared length. Whether the recipe trains an image generator, and how to score the images it produces, is the question of :numref:`sec_dcgan`.

## Summary

Gradient training of an adversarial game need not reach its equilibrium: on the Dirac-GAN the gradient field is orthogonal to the parameters, so the flow circles the solution and every simultaneous step moves away from it, whatever the loss, its weighting or its pairing. The zero-centered penalties $R_1$ and $R_2$ penalize the critic's input gradient under the data and generated distributions; they add damping whose eigenvalues have negative real parts for every $\gamma > 0$, with critical damping at $\gamma = 2\ell'(0)$. Near equilibrium the penalized game measures $p - q$ through a low-pass filter with cutoff $2\sqrt\gamma$: the ratio divergence $2\,\mathrm{JS}$ at larger scales and a linearized Wasserstein-2 distance at smaller ones. R3GAN combines the paired loss with both penalties; on 25 Gaussians it fits the grid while keeping the critic flat on generated samples, and a tenfold larger $\gamma$ leaves the grid blurred within the same budget. Whether the recipe carries over to images is the subject of :numref:`sec_dcgan`.

## Exercises

1. [code] **Discrete-time stability.** The simultaneous update of the
   Dirac-GAN with step size $h > 0$ is the map
   $(\theta, \psi) \mapsto (\theta, \psi) + h\, \mathbf{u}(\theta, \psi)$,
   where $\mathbf{u}$ is the right-hand side of :eqref:`eq_gan_dirac_flow`.
   A critic penalized as in :eqref:`eq_gan_dirac_pen` adds the term
   $-\gamma\psi$ to the second component of $\mathbf{u}$.
    1. Compute the Jacobian of the map at the equilibrium $(0, 0)$ for
       $\gamma = 0$ and show that its eigenvalues are
       $1 \pm i\, h\, \ell'(0)$. Conclude that no step size makes the
       unpenalized game locally convergent.
    1. For $\gamma > 0$ the eigenvalues of the Jacobian are $1 + h\nu$,
       with $\nu$ from :eqref:`eq_gan_dirac_pen`. Show that in the
       underdamped regime $\gamma < 2\ell'(0)$ the iteration converges
       locally if and only if $h < \gamma / \ell'(0)^2$.
    1. In the overdamped regime $\gamma > 2\ell'(0)$, show that the
       eigenvalue $\nu$ closer to zero behaves like
       $-\ell'(0)^2/\gamma$ as $\gamma$ grows. For a fixed small $h$,
       write the spectral radius of the step to first order in $h$ as a
       function of $\gamma$ in both regimes, and find the $\gamma$ that
       minimizes it.
    1. Check the threshold of (b) for the logistic loss at
       $\gamma = 0.3$: run `trajectory((0.1, 0.1), h, 3000, gamma=0.3)`
       for $h = 1.1$ and $h = 1.3$ and plot the distance from the
       equilibrium. What does the iteration do above the threshold?
   :exlabel:`ex_convergence-discrete-simultaneous-update-dirac`
1. **Penalized best response.** With $p = \delta_0$ and
   $q = \delta_\theta$, the best-response value $\sup_\psi V(\theta, \psi)$
   of :eqref:`eq_gan_dirac_value` is the same for every $\theta \neq 0$, so
   it gives the generator no gradient. Let the critic subtract both
   penalties of :eqref:`eq_gan_r1r2`, which for the linear critic sum to
   $\gamma\psi^2$, and take $\ell = \log\sigma$. Write
   $V_\gamma(\theta) = \sup_\psi \big\{ V(\theta, \psi) - \gamma\psi^2 \big\}$.
    1. Show that the supremum is attained for every $\theta$. Expand
       $\ell(-\psi\theta)$ to first order around zero and show that, for
       small $\theta$, the maximizer is
       $\psi^\star \approx -\ell'(0)\,\theta / (2\gamma)$ and

       $$V_\gamma(\theta) = 2\ell(0) + \frac{\ell'(0)^2}{4\gamma}\, \theta^2 + O(\theta^4).$$

    1. The supremum in :eqref:`eq_gan_sobolev` runs over all critics; here
       it runs over linear ones. Compare the result of (a) with
       :eqref:`eq_gan_sobolev`, using $W_2(\delta_0, \delta_\theta) = |\theta|$.
       Why does the curvature term of :eqref:`eq_gan_local_value` not
       appear at order $\theta^2$? What gradient does the generator receive
       from $V_\gamma$ near $\theta = 0$, and how does its size depend on
       $\gamma$?
    1. Show that $V_\gamma(\theta) \to \ell(0) + \sup_t \ell(t)$ as
       $|\theta| \to \infty$. What does this imply about the generator's
       gradient far from the data?
    1. Now let the critic be any differentiable function and subtract
       $\gamma \int m\, \|\nabla_x D\|^2$ with
       $m = \tfrac12(\delta_0 + \delta_\theta)$, which is $R_1 + R_2$
       of :eqref:`eq_gan_r1r2_sum`. Show that the penalized best-response
       value is $2\sup_t \ell(t)$ for every $\theta \neq 0$. Which
       assumption behind :eqref:`eq_gan_local_value` fails for these two
       point masses? What does the result say about the role of the critic
       class in (a) to (c)?
   :exlabel:`ex_convergence-penalized-best-response`
1. **Penalizing one side only.** Let $p = \delta_0$, let the generator
   place its mass at $x' = 1$, and consider the critics
   $D_c(x) = -c\, x^2$ with $c \geq 0$, which are flat at the data point
   and steep at the generated one. Take $\ell = \log\sigma$.
    1. Evaluate $R_1$ and $R_2$ of :eqref:`eq_gan_r1r2` for $D_c$. Show
       that the critic objective $V_\ell(D_c) - R_1$ increases with $c$ and
       has no maximizer.
    1. The non-saturating generator minimizes $-\log\sigma(D(x'))$.
       Compute the derivative of this loss with respect to $x'$ at
       $x' = 1$ and show that it grows without bound in $c$. What does the
       saturating generator loss $\log\sigma(-D(x'))$ give instead? Which
       observation of :citet:`Huang.Gokaslan.Kuleshov.ea.2024` does this
       reproduce?
    1. Now subtract $R_1 + R_2$. Show that the maximizing $c^\star$ is
       finite and satisfies $c^\star < 1/(8\gamma)$, and bound the
       generator's gradient at $c^\star$.
   :exlabel:`ex_convergence-one-sided-penalty`
1. [code] **Batch-mixing critics.** The penalties $R_1$ and $R_2$ concern
   the per-sample input gradient $\nabla_x D(x)$. Consider the
   batch-centered critic that maps a batch
   $\mathbf{x}_1, \ldots, \mathbf{x}_B \in \mathbb{R}^2$ to the scores
   $D_i = \mathbf{w}^\top (\mathbf{x}_i - \bar{\mathbf{x}})$, where
   $\bar{\mathbf{x}}$ is the batch mean and $\mathbf{w} \in \mathbb{R}^2$
   is fixed.
    1. Compute $\partial D_i / \partial \mathbf{x}_i$ and the squared norm
       that a per-sample penalty should report for $\mathbf{w} = (3, -4)$
       and $B = 8$.
    1. Implement this critic and evaluate `r1_r2_penalty` on a batch of 8
       real and 8 generated points with $\mathbf{w} = (3, -4)$. Explain the
       returned values from the implementation of `r1_r2_penalty`.
   :exlabel:`ex_convergence-batch-mixing-critics`
1. [code] **The penalty weight as a length.** In
   :numref:`subsec_gan_25_gaussians`, $\gamma = 10$ leaves the grid blurred
   where $\gamma = 1$ fits it.
    1. Let $D_c(x) = D(x / c)$ be a critic for data scaled by $c > 0$.
       Show that $\mathbb{E}\|\nabla_x D_c\|^2$ on the scaled data is
       $c^{-2}$ times $\mathbb{E}\|\nabla_x D\|^2$ on the original data, so
       the penalized game on the scaled data at weight $c^2\gamma$ has the
       same value as the original game at weight $\gamma$.
    1. With the weight $1/(|\ell''(0)| + \gamma \omega^2)$ of the low-pass
       analysis and $|\ell''(0)| = \tfrac14$, compute by what factor the
       penalty damps a component whose wavelength is the grid spacing 2,
       and one whose wavelength is the mode width $0.05$, at $\gamma = 1$
       and at $\gamma = 10$.
    1. Train the $\gamma = 10$ run for two and four times as many steps,
       with the cosine schedule stretched to match. Does the on-mode
       fraction approach that of $\gamma = 1$? Does $\gamma$ change where
       training ends, or how fast it gets there?
    1. Scale the grid by $c = 2$ (spacing 4, width 0.1) and train at
       $\gamma = 4$ and at $\gamma = 1$. Which run matches the original
       $\gamma = 1$ run more closely in on-mode fraction? Which parts of
       `train_toy` are not invariant under the scaling?
   :exlabel:`ex_convergence-penalty-length`
1. [code] **Alternating updates.** ● `train_toy` updates the critic first
   and then takes a generator step against the updated critic. On the
   Dirac-GAN with step size $h$, this alternating update reads

   $$\psi_{k+1} = \psi_k + h\, u_\psi(\theta_k, \psi_k), \qquad \theta_{k+1} = \theta_k + h\, u_\theta(\theta_k, \psi_{k+1}),$$

   where $(u_\theta, u_\psi)$ is the field of :eqref:`eq_gan_dirac_flow`.
    1. Linearize both half-steps at the equilibrium for $\gamma = 0$ and
       show that the Jacobian of the combined update has determinant one.
       Show that its eigenvalues lie on the unit circle if and only if
       $h\, \ell'(0) \leq 2$. What does this imply about linear
       convergence and linear divergence?
    1. Write an alternating variant of `trajectory` and run it at
       $\gamma = 0$ from $(0.1, 0.1)$ for $10^5$ steps with
       $h \in \{0.2, 1, 3, 5\}$. Record the smallest and the largest
       distance from the equilibrium over the last $10^4$ steps. Explain
       the wide spread at $h = 3$ from the shape of the orbits of a
       linear map with determinant one. Then repeat the run at
       $h = 0.2$ with the penalty $\gamma = 0.3$. How does the distance
       evolve?
    1. Compare the run at $h = 5$ with the prediction of (a) for
       $h\, \ell'(0) > 2$. Where does the iterate end up, and why does
       it stop moving there?
   :exlabel:`ex_convergence-alternating-updates`
1. [code] **The local value.** ●
    1. Expand the pairing objective
       $\Phi(D) = \mathbb{E}_{x \sim p,\, x' \sim q}\big[\ell(D(x) - D(x'))\big]$
       to second order in $D$ and show that, up to terms of third order
       in $D$ and $p - q$, the quadratic term is
       $-|\ell''(0)|\, \mathrm{Var}_m[D]$. Show that the margin objective,
       whose quadratic term is $-|\ell''(0)| \int m\, D^2$, reaches the same
       supremum once the critic's constant is optimized.
    1. Perturb the maximizing critic of :eqref:`eq_gan_local_value` to
       $D + s\,\xi$ with a test function $\xi$ and show that the
       first-order condition in $s$ is
       $\big(|\ell''(0)|\, m + \gamma \mathcal{L}_m\big) D = \tfrac{a}{2}(p - q)$.
       Assume that boundary terms vanish.
    1. In one dimension, integrate the large-$\gamma$ equation
       $\mathcal{L}_m D = \tfrac{a}{2\gamma}(p - q)$ once and show that
       $\|p - q\|^2_{\dot H^{-1}(m)} = \int (F_p(x) - F_q(x))^2 / m(x)\, dx$,
       where $F_p$ and $F_q$ are the cumulative distribution functions.
       Evaluate this norm to leading order in $\mu$ for
       $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$, and
       compare it with $W_2(p, q) = |\mu|$.
    1. Let $p = \mathcal{N}(0, 1)$ and $q = p\,(1 + 0.1 \sin \omega x)$. Use
       `penalized_value` to compute the ratio of the values at $\gamma$
       and at $\gamma = 0$ for $\omega \in \{0.5, 2, 8\}$ and
       $\gamma \in \{0.01, 0.1, 1\}$, and compare it with the low-pass
       weight $1/(1 + 4\gamma \omega^2)$. For which $\omega$ does the prediction
       fail, and which assumption of the Fourier argument breaks there?
   :exlabel:`ex_convergence-prove-eqref-eq-gan`
1. [code] **One-centered penalty dynamics.** ●
   :citet:`Mescheder.Geiger.Nowozin.2018` show that WGAN-GP does not
   converge on the Dirac-GAN. Use the linear score $\ell(t) = t$ of the
   Wasserstein critic of
   :numref:`sec_gan_objectives`, so that $\ell'(0) = 1$. The input gradient
   of $D_\psi$ is $\psi$ everywhere, so the one-centered penalty becomes
   $\tfrac{\gamma}{2}(|\psi| - 1)^2$, which the critic subtracts from its
   objective.
    1. Write the simultaneous gradient field and show that it vanishes at
       no point with $\psi \neq 0$. Compute its limits as $\psi \to 0^+$
       and as $\psi \to 0^-$, and find the points at which these one-sided
       limits vanish. Where is the generator at these points, and what
       does the critic look like?
    1. On the half-plane $\psi > 0$, show that the field is affine and
       compare its Jacobian with $J_\gamma$ of :eqref:`eq_gan_dirac_pen`.
       For which $\gamma$ are its eigenvalues real?
    1. Modify `dirac_field` accordingly and run `trajectory` from $(1, 1)$
       with $h = 0.01$ for 40,000 steps, for $\gamma = 1$ and
       $\gamma = 3$. Where does each trajectory end? Explain both outcomes
       with (a) and (b).
   :exlabel:`ex_convergence-one-centered-dynamics`
1. [code] **Momentum in a rotating game.** ● R3GAN trains both networks
   with Adam at $\beta_1 = 0$, one of the differences of its baseline from
   DCGAN listed in :numref:`subsec_gan_r3gan_principles`. To isolate the
   first moment, drop Adam's second-moment normalization and average the
   field of the penalized Dirac-GAN with the logistic loss:

   $$\bar{\mathbf{u}}_k = \beta\, \bar{\mathbf{u}}_{k-1} + (1 - \beta)\, \mathbf{u}(\mathbf{x}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + h\, \bar{\mathbf{u}}_k,$$

   with $\mathbf{x}_k = (\theta_k, \psi_k)$, $\bar{\mathbf{u}}_{-1} = \mathbf{0}$,
   and $\mathbf{u}$ the field of :eqref:`eq_gan_dirac_flow` with the term
   $-\gamma\psi$ in its second component.
    1. Linearize at the equilibrium and write one step as a linear map of
       the state $(\mathbf{x}_k, \bar{\mathbf{u}}_{k-1}) \in \mathbb{R}^4$.
       Compute numerically the largest stable step size for
       $\beta \in \{0, 0.5, 0.9\}$, at $\gamma = 0.3$ and at $\gamma = 1$,
       and check the case $\beta = 0$ against problem
       :exref:`ex_convergence-discrete-simultaneous-update-dirac`b. In
       which damping regime does momentum shrink the range of stable step
       sizes?
    1. Run the nonlinear iteration at $\gamma = 0$ from $(0.1, 0.1)$ for
       20,000 steps with $h = 0.2$ and $\beta \in \{-0.5, 0, 0.5\}$.
       Then give each player its own average and use it inside the
       alternating update of
       exercise :exref:`ex_convergence-alternating-updates`. Which combinations
       converge without any penalty? Compare the outcome with the
       analysis of bilinear games by
       :citet:`Gidel.Askari-Hemmat.Pezeshki.ea.2019`.
   :exlabel:`ex_convergence-momentum-rotating-game`
1. [code] **Non-saturating Dirac dynamics.** ●● The non-saturating
   generator replaces $\ell'(-\psi\theta)$ by $\ell'(\psi\theta)$ in the
   first component of :eqref:`eq_gan_dirac_flow`. The Jacobian at the
   equilibrium stays the same, but the nonlinear dynamics change; the
   appendix of :citet:`Mescheder.Geiger.Nowozin.2018` analyzes the
   continuous case. Take $\ell = \log\sigma$ and $\gamma = 0$, and write
   $r^2 = \theta^2 + \psi^2$.
    1. Show that along the non-saturating flow
       $\tfrac{d}{dt} r^2 = 2\psi\theta\,\big(\ell'(\psi\theta) - \ell'(-\psi\theta)\big)$,
       and that this rate is never positive for a concave $\ell$. Why can
       the resulting approach to the equilibrium not be linear?
    1. Near the equilibrium, the angle of $(\theta, \psi)$ turns at the
       nearly constant rate $\ell'(0)$ while $r$ changes slowly. Average
       $\tfrac{d}{dt} r^2$ over one turn to leading order in $r$, solve the
       averaged equation, and show that $r(t)^2 \approx 8/t$ for large $t$.
    1. For simultaneous steps of size $h$, one step changes $r^2$ by
       $2h\, \langle \mathbf{x}, \mathbf{u} \rangle + h^2 \|\mathbf{u}\|^2$,
       where $\mathbf{x} = (\theta, \psi)$ and $\mathbf{u}$ is the
       non-saturating field. Average this change over one turn to leading
       order and show that it vanishes at a radius $r^\star$ with
       $(r^\star)^2 = 2h$. Is the corresponding cycle stable?
    1. Redefine `dirac_field` with the non-saturating weight. Run
       `trajectory` from $(1, 1)$ and from $(0.05, 0.05)$ for $4000/h$
       steps with $h \in \{0.01, 0.05, 0.2\}$, and compare the late
       distance from the equilibrium with $r^\star$. Check (b) with
       `flow=True`.
   :exlabel:`ex_convergence-non-saturating-dynamics`

[Discussions](https://d2l.discourse.group/t/36811)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.4]{.kicker}

Gradient penalties and convergence<br>
**the Dirac-GAN · circles and spirals · zero-centered penalties · the penalized value · the R3GAN loss**
:::
:::

::: {.slide title="Gradient Steps Double the Distance to a Solution That Exists"}
[The problem]{.kicker}

- Data at $0$, the generator's point at $\theta$, a linear critic
  $D(x) = \psi x$: the optimum $(\theta, \psi) = (0, 0)$ is known.
- 300 simultaneous gradient steps of size 0.2 from $(1, 1)$: the distance
  to it **doubles**.
- The GAN losses of the first experiment oscillated around $\log 2$ for the
  same reason.

::: {.d2l-note .warn}
Knowing the value at the optimum does not tell us whether training gets
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
| Generative Adversarial Networks | what does a classifier of samples compute? |
| Objectives and Divergences | which loss gives which distance? |
| Relativistic Objectives | what changes when the critic ranks pairs? |
| **Penalties and Convergence** | **does gradient training reach the optimum?** |
| Adversarial Image Generation | which network, and how to score images? |
| Conditional Generation | how to ask for a class? |
| Adversarial Losses beyond GANs | where does a learned comparison remain? |
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
  weight, and also the rank weight, since $D(0) = 0$.

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
- Non-saturating generator: same Jacobian; a slow drift ($r^2 \approx 8/t$),
  then a cycle of radius $\sqrt{2h}$.
:::

::: {.slide title="Check: Does Pairing Change the Two-Point Dynamics?"}
[Check]{.kicker}

The pairing loss scores the pair $(0, \theta)$ by $\ell\big(D(0) - D(\theta)\big)$.
Does it change the field?

. . .

- $\ell(D(0) - D(\theta)) = \ell(-\psi\theta)$: the classical value minus the
  constant $\ell(0)$.
- Same gradient field, same circles, same outward spiral.
- Pairing changes which configurations trap the generator, not the
  rotation. The cure must change the dynamics.
:::

::: {.slide title="Zero-Centered Penalties Charge the Critic's Slope"}
[Penalties]{.kicker}

$$R_1 = \tfrac{\gamma}{2}\,\mathbb{E}_{x \sim p}\|\nabla_x D\|^2, \quad
R_2 = \tfrac{\gamma}{2}\,\mathbb{E}_{x' \sim q}\|\nabla_{x'} D\|^2, \quad
R_1 + R_2 = \gamma\,\mathbb{E}_{m}\|\nabla_x D\|^2$$

- The critic maximizes $V - R_1 - R_2$; the generator is unchanged.
- $m = (p + q)/2$; the gradient is with respect to the **input**.
- Zero-centered: the flat critic minimizes both, and at $q = p$ the optimal
  critic is flat, so the penalty vanishes at the equilibrium.
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
distance in 300 steps; $\gamma = 0.3$ spirals in; $\gamma = 1$ turns once.
:::

::: {.slide title="Check: How Large a Step Survives at γ = 0.3?"}
[Check]{.kicker}

Logistic loss, $\ell'(0) = \tfrac12$, one penalty with $\gamma = 0.3$. For
which step sizes $h$ do simultaneous steps converge?

. . .

- Underdamped: $\nu = -\gamma/2 \pm i\omega$ with $\omega^2 = \ell'(0)^2 - \gamma^2/4$.
- $|1 + h\nu|^2 < 1 \iff h\,|\nu|^2 < \gamma$, and $|\nu|^2 = \ell'(0)^2$.
- So $h < \gamma/\ell'(0)^2 = $ **1.2**.
:::

::: {.slide title="A One-Centered Penalty Keeps Pushing at the Optimum"}
[Penalties]{.kicker}

| penalty | critic it prefers at $q = p$ | two-point game |
|:--|:--|:--|
| one-centered (WGAN-GP), $\mathbb{E}(\|\nabla D\| - 1)^2$ | unit slope | does not converge |
| zero-centered, $\mathbb{E}_m\|\nabla D\|^2$ | flat | converges for every $\gamma$ |

- At $q = p$ the optimal critic is constant. A unit slope keeps a gradient
  on a generator that already matches the data.
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

[Huang et al., 2024; single runs]{.small}
:::

::: {.slide title="Near Equilibrium the Curvature Term Is as Large as the Penalty"}
[The penalized value]{.kicker}

Expand to second order, $a = \ell'(0)$, $\mathcal{L}_m D = -\nabla\cdot(m\nabla D)$:

$$\sup_D \Big\{ a\langle p - q, D\rangle - |\ell''(0)|\!\int\! m D^2 - \gamma\!\int\! m\|\nabla D\|^2 \Big\}
= \frac{a^2}{4}\big\langle p - q, (|\ell''(0)|\,m + \gamma \mathcal{L}_m)^{-1}(p - q)\big\rangle$$

- $\gamma \to 0$: $2\,\mathrm{JS}(p, q)$, the unpenalized value.
- Large $\gamma$: $\tfrac{a^2}{4\gamma}\|p - q\|^2_{\dot H^{-1}(m)} \approx \tfrac{a^2}{4\gamma}W_2^2$.
:::

::: {.slide title="A Gaussian Shift Checks the Formula to One Percent"}
[The penalized value]{.kicker}

$p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(0.2, 1)$, logistic loss: the
critic is a line, slope shrunk by $1 + 4\gamma$; value $\mu^2/(4 + 16\gamma)$.

@!convergence-a-gaussian-shift

- Exact (Newton on a grid) and local formula agree within 1%.
- The large-$\gamma$ form is 26× too large at $\gamma = 0.01$.
:::

::: {.slide title="The Penalty Is a Low-Pass Filter with Cutoff 2√γ"}
[The penalized value]{.kicker}

A Fourier component $e^{i\omega x}$ of $p - q$ enters with weight

$$\frac{1}{|\ell''(0)| + \gamma \omega^2} \quad\text{instead of}\quad \frac{1}{|\ell''(0)|}$$

- Logistic loss: features longer than $2\sqrt\gamma$ are measured by
  $2\,\mathrm{JS}$, shorter ones by a transport distance.
- The signal does not vanish when nearby supports move apart.
- $\gamma$ has units of **length²**: R3GAN uses 0.05 (CIFAR-10) to 150
  (FFHQ-256), annealed tenfold during training.
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
- Pairing addresses coverage; the penalties address convergence.
:::

::: {.slide title="Check: What Should γ = 10 Do on a Grid with Spacing 2?"}
[Check]{.kicker}

25 Gaussians on a $5 \times 5$ grid, spacing 2, width 0.05. Same budget,
$\gamma = 1$ against $\gamma = 10$. Predict.

. . .

- Cutoff $2\sqrt\gamma$: 2 at $\gamma = 1$ (the spacing), about 6.3 at
  $\gamma = 10$ (the whole grid).
- The signal about mode positions is damped about ten times more at
  $\gamma = 10$, so the grid resolves more slowly.
- The equilibrium is still $q = p$: $\gamma$ sets the speed, not the end.
:::

::: {.slide title="γ = 1 Fits the Grid with a Flat Critic; γ = 10 Leaves It Blurred"}
[25 Gaussians]{.kicker}

@!convergence-mode-coverage-on-25-gaussians-2

- $\gamma = 1$: all 25 modes, most samples within three widths.
- The plain GAN fits about as well, but its $\mathbb{E}_q\|\nabla_x D\|^2$
  is thousands of times larger: the critic stays steep.
:::

::: {.slide title="Three Choices Describe Every Game of the Chapter"}
[The games]{.kicker}

| objective | critic estimates | does | constrained by | disjoint |
|:--|:--|:--|:--|:--|
| log loss | $\log(p/q)$ | classifies | nothing | saturates |
| MMD | smoothed $p - q$ | witnesses | kernel ball | smooth |
| $W_1$ | potential | witnesses | Lipschitz ball | smooth |
| RpGAN | $\log(p/q)$ + const | ranks | the data | saturates |
| R3GAN | $\log(p/q)$ above $2\sqrt\gamma$ | ranks | Sobolev penalty | smooth |
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Two-point game: the flow circles; simultaneous steps double the distance
  in 300 steps of 0.2; pairing and reweighting do not help.
- Zero-centered penalties: eigenvalues with negative real part for every
  $\gamma > 0$; critical at $\gamma = 1$ for the logistic loss.
- Near equilibrium: a low-pass filter on $p - q$ with cutoff $2\sqrt\gamma$;
  the Gaussian shift agrees to 1%.
- 25 Gaussians: $\gamma = 1$ fits with a flat critic; $\gamma = 10$ blurs.

**Next:** the same loss on images, and how to score what it generates.
:::
