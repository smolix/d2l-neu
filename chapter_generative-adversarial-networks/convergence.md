# Gradient Penalties and Convergence
:label:`sec_gan_convergence`

An adversarial objective can be correct and still untrainable. The preceding sections chose *what* the game evaluates: the divergence at the optimal critic, the payoff, the critic class, the pairing structure. Every result located an equilibrium; none asked whether gradient descent approaches it. This section takes up that question on the smallest adversarial game that exists, two point masses and a linear critic, and the answer is unambiguous: the continuous-time dynamics orbit the solution forever, and discretizing them produces divergence at every step size. The repair is a regularizer on the critic, the zero-centered gradient penalty, and it admits the same exact analysis: the penalized dynamics converge for every penalty weight, and near equilibrium the penalized game measures a linearized optimal-transport distance. Objective and penalty together form the loss of the modern GAN baseline of :citet:`Huang.Gokaslan.Kuleshov.ea.2024`, which the section states in code and tests on a distribution with twenty-five modes.

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

## What the Objective Cannot Buy

:numref:`sec_gan_relativistic` left the chapter with one repair made and one failure standing. The pairing objective removes the mode-dropping basins of the classical loss: its rank weight cannot be satisfied by a generator that pleases a single decision threshold. What it does not remove is the ceiling. The value of the pairing game is a Jensen--Shannon divergence of product measures, and it saturates on disjoint supports exactly as the log-loss game of :numref:`sec_basic_gan` does: once the generator's samples and the data stop overlapping, the value pins at its maximum and the gradient through it is zero. Changing the payoff, :numref:`sec_gan_objectives` showed, moves the problem around rather than solving it, unless the objective is tied to the geometry of the sample space.

The second failure has been invisible because the chapter's method could not see it. Every derivation so far evaluates the game at the critic's best response: an inner optimization solved exactly, then an outer one. Training solves neither. It updates two coupled players by simultaneous or alternating gradient steps, and each player's step changes the landscape the other is descending. Nothing in the analysis of equilibria says that this coupled process approaches the equilibrium, and on the example below it provably does not, even though the equilibrium is unique and the generator starts one step away from it.

Both failures are visible on one tiny example, a single point mass chasing another. The section first analyzes its training dynamics exactly: gradient flow circles the solution, and discretization turns the circles into outward spirals. A pair of penalties on the critic's input gradient then restores convergence, with eigenvalues that can be written down; near equilibrium, the penalized game turns out to measure a linearized Wasserstein-2 distance, connecting the fix back to the transport geometry of :numref:`sec_gan_objectives`. The section ends by assembling objective and penalties into the R3GAN recipe and measuring, on a twenty-five-mode toy distribution, what the penalties repair --- and what a toy of this size cannot show.

## The Dirac-GAN

Fix the data at a single point and let the generator place a single point: $p = \delta_0$ and $q_\theta = \delta_\theta$ on the real line, so the generator's only parameter $\theta$ is the position of its point mass, and $q = p$ exactly at $\theta = 0$. The critic is linear with one parameter, its slope: $D_\psi(x) = \psi x$. Playing the margin game :eqref:`eq_gan_margin` with payoff $\ell$ collapses both expectations to single evaluations,

$$
V(\theta, \psi) \;=\; \ell\big(D_\psi(0)\big) + \ell\big({-D_\psi(\theta)}\big)
\;=\; \ell(0) + \ell(-\psi\theta).
$$
:eqlabel:`eq_gan_dirac_value`

This is the *Dirac-GAN* of :citet:`Mescheder.Geiger.Nowozin.2018`. The payoff is any of the classification payoffs of :numref:`sec_gan_objectives`, differentiable with $\ell'(0) > 0$; the logistic payoff $\ell = \log\sigma$ is the running example, with $\ell'(t) = \sigma(-t)$ and hence $\ell'(0) = \tfrac12$.

The example inherits the chapter's first failure in one line. For any $\theta \neq 0$ the supports are disjoint, and the critic's best response drives $\ell(-\psi\theta)$ toward its supremum by sending $\psi\theta \to -\infty$: the best-response value $\sup_\psi V = \ell(0) + \sup_t \ell(t)$ is the same number for every $\theta \neq 0$, approached but never attained. A value independent of $\theta$ provides no gradient in $\theta$, however close the generator stands to the solution. This is the saturation of :numref:`sec_basic_gan` in miniature.

Training, however, does not play best responses; it takes gradient steps from wherever the two players stand, and this is where the second failure appears. The game has a unique stationary point, $(\theta, \psi) = (0, 0)$: the generator on the data and the critic flat. Write simultaneous gradient descent--ascent as a flow, the generator descending $V$ and the critic ascending it:

$$
\dot\theta \;=\; -\partial_\theta V \;=\; \psi\, \ell'(-\psi\theta),
\qquad
\dot\psi \;=\; +\partial_\psi V \;=\; -\theta\, \ell'(-\psi\theta).
$$
:eqlabel:`eq_gan_dirac_flow`

The two components are the same positive scalar $\ell'(-\psi\theta)$ times the vector $(\psi, -\theta)$, which is orthogonal to the position $(\theta, \psi)$. The consequence is immediate:

$$
\frac{d}{dt}\big(\theta^2 + \psi^2\big)
= 2\theta\psi\,\ell'(-\psi\theta) - 2\psi\theta\,\ell'(-\psi\theta) = 0 .
$$

Every trajectory conserves its distance to the equilibrium: the continuous-time dynamics move on exact circles, for every payoff in the family. The linearization at the origin says the same thing locally,

$$
J = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & 0 \end{pmatrix},
\qquad
\lambda_{1,2} = \pm\, i\, \ell'(0):
$$
:eqlabel:`eq_gan_dirac_eigs`

purely imaginary eigenvalues, a center. The dynamics rotate at angular speed $\ell'(0)$ and attract nothing. The generator is never pulled toward the data: the pair $(\theta, \psi)$ circles the equilibrium at constant distance in parameter space, so the generator sweeps past the data and away again, indefinitely, without settling.

Discrete gradient steps fare worse than the flow they approximate. A simultaneous gradient step with step size $\eta > 0$ replaces the flow by the map $(\theta, \psi) \mapsto (\theta, \psi) + \eta\, v(\theta, \psi)$, with $v$ the right-hand side of :eqref:`eq_gan_dirac_flow`. The orthogonality that conserved the radius now works against the iterate: by the Pythagorean theorem,

$$
\big\| (\theta, \psi) + \eta\, v \big\|^2
\;=\; \big\| (\theta, \psi) \big\|^2 + \eta^2 \big\| v \big\|^2 ,
$$

and for the logistic payoff $\ell' > 0$ everywhere, so $v$ vanishes only at the equilibrium. Every step from every other point strictly increases the distance to the solution, for every step size: the iterates spiral outward. A smaller $\eta$ slows the spiral without changing its direction. Exercise 1 reaches the same conclusion through the update map's Jacobian, whose spectral radius $\sqrt{1 + \eta^2 \ell'(0)^2}$ exceeds one for every $\eta$; alternating the two updates instead of taking them simultaneously does not restore convergence either :cite:`Mescheder.Geiger.Nowozin.2018`.

Neither of the chapter's objective repairs changes this picture. The non-saturating generator weight of :eqref:`eq_gan_weights` replaces $\ell'(-\psi\theta)$ by $\ell'(\psi\theta)$ in the first component of the field, which leaves the Jacobian at the equilibrium, and with it every local conclusion, unchanged. The pairing objective of :numref:`sec_gan_relativistic` collapses on this example to $\ell\big(D_\psi(0) - D_\psi(\theta)\big) = \ell(-\psi\theta)$, which is :eqref:`eq_gan_dirac_value` minus the constant $\ell(0)$: an identical gradient field, hence identical circles and identical spirals :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. The failure lies in the dynamics of the two-player gradient game, so the fix must act on the dynamics.

## Zero-Centered Penalties

What the equilibrium lacks is attraction: eigenvalues with a negative real part. The term that supplies it is a regularizer on the critic. Define the two *zero-centered gradient penalties*

$$
R_1 \;=\; \frac{\gamma}{2}\, E_{x \sim p}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad
R_2 \;=\; \frac{\gamma}{2}\, E_{x' \sim q}\Big[ \big\| \nabla_x D(x') \big\|^2 \Big],
$$
:eqlabel:`eq_gan_r1r2`

with weight $\gamma > 0$, and let the critic maximize $V_\ell(D) - R_1 - R_2$ while the generator's objective is unchanged :cite:`Roth.Lucchi.Nowozin.ea.2017,Mescheder.Geiger.Nowozin.2018`. The gradient being penalized is with respect to the *input* $x$, not the parameters: $R_1$ charges the critic for varying rapidly near data samples, $R_2$ for varying rapidly near generated ones. "Zero-centered" names the penalty's minimizer, the flat critic $\nabla_x D = 0$, and that choice is aimed at the equilibrium: at $q = p$ the optimal critic of :numref:`sec_basic_gan` is the constant $D^\star = \log(p/q) = 0$, so the penalty costs nothing precisely at the point it is meant to stabilize. Adding the two penalties gives, since $\tfrac12(E_p + E_q) = E_m$,

$$
R_1 + R_2 \;=\; \gamma\, E_{x \sim m}\Big[ \big\| \nabla_x D(x) \big\|^2 \Big],
\qquad m = \frac{p + q}{2} :
$$
:eqlabel:`eq_gan_r1r2_sum`

a single smoothness penalty weighted by the balanced mixture that defines the log-loss game. The critic's variation is taxed exactly where the game draws its samples.

### Damping the Dirac-GAN

On the Dirac-GAN the penalties can be evaluated by inspection. The linear critic has $\nabla_x D_\psi = \psi$ at every $x$, so both penalties reduce to the same term $\tfrac{\gamma}{2}\psi^2$; on this example $R_1$ and $R_2$ coincide, a degeneracy worth remembering. Subtracting either one from the critic's objective appends $-\gamma\psi$ to the critic's update (using both doubles it, replacing $\gamma$ by $2\gamma$ in what follows), and the flow becomes

$$
\dot\theta = \psi\,\ell'(-\psi\theta),
\qquad
\dot\psi = -\theta\,\ell'(-\psi\theta) - \gamma\psi .
$$

One entry of the Jacobian changes, and it is the entry that decides convergence:

$$
J_\gamma = \begin{pmatrix} 0 & \ell'(0) \\ -\ell'(0) & -\gamma \end{pmatrix},
\qquad
\lambda_{1,2} \;=\; -\frac{\gamma}{2} \pm \sqrt{ \frac{\gamma^2}{4} - \ell'(0)^2 } .
$$
:eqlabel:`eq_gan_dirac_pen`

Both eigenvalues have negative real part for every $\gamma > 0$: the equilibrium is now attracting, and gradient descent with a small enough step size converges locally at a linear rate :cite:`Mescheder.Geiger.Nowozin.2018`. The formula is the damped oscillator's, and it separates three regimes. For $\gamma < 2\ell'(0)$ the square root is imaginary: trajectories still rotate, but spiral inward. At $\gamma = 2\,|\ell'(0)|$ the root vanishes and with it the rotation: critical damping, the fastest approach without oscillation; for the logistic payoff this critical weight is $\gamma = 1$. Beyond it the dynamics are overdamped, and convergence slows again as $\gamma$ grows, a first sign that more penalty is not always better. (In :citet:`Huang.Gokaslan.Kuleshov.ea.2024` the same formula appears, as their Eq. 12, with $\ell'(0)$ in place of $\ell'(0)^2$ under the root. This is a typographical error: the Jacobian displayed beside it has determinant $\ell'(0)^2$, and the original lemma of :citet:`Mescheder.Geiger.Nowozin.2018` carries the square.)

### What the Penalized Game Measures

Damping explains the mechanics; it does not yet answer the question this chapter asks of every objective, namely what quantity the game evaluates. Near the equilibrium the answer has a closed form. There the critic is close to the constant $D^\star = 0$, so expand the payoffs around $D \equiv 0$. The margin objective :eqref:`eq_gan_margin` and the pairing objective of :numref:`sec_gan_relativistic`, written $\Phi$ there, expand identically up to constants,

$$
V_\ell(D) = 2\ell(0) + \ell'(0)\, \langle p - q,\, D \rangle + O(\|D\|^2),
\qquad
\Phi(D) = \ell(0) + \ell'(0)\, \langle p - q,\, D \rangle + O(\|D\|^2),
$$

where $\langle h, D\rangle = \int h(x)\, D(x)\, dx$, so that $\langle p - q, D\rangle = E_p[D] - E_q[D]$. To leading order, every objective in the family sees the same linear functional $a\,\langle p - q, D\rangle$ with $a = \ell'(0)$, and the penalized critic problem becomes: maximize a linear reward against a quadratic smoothness cost. That problem has a classical value.

**Proposition.** *For $a \in \mathbb{R}$ and $\gamma > 0$,*

$$
\sup_{D} \Big\{ a\, \langle p - q,\, D \rangle \;-\; \gamma \int m\, \| \nabla_x D \|^2 \Big\}
\;=\; \frac{a^2}{4\gamma}\, \big\| p - q \big\|^2_{\dot H^{-1}(m)},
$$
:eqlabel:`eq_gan_sobolev`

*where the dual Sobolev norm is $\|h\|_{\dot H^{-1}(m)} = \sup\big\{ \langle h, D\rangle : \int m \|\nabla_x D\|^2 \leq 1 \big\}$, and the maximizing critic solves the weighted Poisson equation $a\,(p - q) + 2\gamma\, \nabla \cdot (m\, \nabla_x D) = 0$.*

Constants do not enter, since $\int (p - q) = 0$ and a constant critic has zero gradient; the supremum is over critics modulo constants. The computation behind the statement is one-dimensional: along each ray $\{t D_0\}$ the objective is a scalar quadratic in $t$, and optimizing over the direction $D_0$ produces the squared dual norm. Exercise 3 carries it out.

Two readings give the formula its content. First, :eqref:`eq_gan_sobolev` is a squared norm of the *difference* $p - q$, where every unpenalized objective in this chapter evaluated a function of the *ratio* $p/q$. The ratio degenerates the moment the supports separate; a difference of measures does not, and the norm continues to vary as the supports move apart. The factor $\gamma$ sets the scale, with the value inversely proportional to it, and the mixture $m$ sets the local weighting: smoothness is expensive for the critic where the game has samples and free where it has none.

Second, the norm has a geometric name. :eqref:`eq_mdl-w2` defines the Wasserstein-2 distance as the cheapest quadratic-cost transport between two distributions, and the Benamou--Brenier theorem :eqref:`eq_mdl-benamou-brenier` rewrites it as the least kinetic energy of any flow carrying one into the other. In that least-action picture, an infinitesimal perturbation $h$ of the base distribution $m$ is carried by a velocity field $\nabla\phi$ obeying the continuity equation $h + \nabla \cdot (m\, \nabla\phi) = 0$ --- the same Poisson equation the optimal penalized critic solves --- and the kinetic energy of that field is exactly $\|h\|^2_{\dot H^{-1}(m)}$. Consequently $W_2(m,\, m + \epsilon h) = \epsilon\, \|h\|_{\dot H^{-1}(m)} + o(\epsilon)$: the dual Sobolev norm is the local metric of optimal transport. Near its equilibrium, then, the penalized adversarial game measures the squared linearized $W_2$ distance between $p$ and $q$, scaled by $a^2/4\gamma$, to leading order. A transport distance was precisely the property :numref:`sec_gan_objectives` identified as the cure for saturation, and the penalties obtain its quadratic version from two expectations of a squared gradient, with no Lipschitz constraint to enforce. The statement restates the local-convergence results of :citet:`Mescheder.Geiger.Nowozin.2018` and :citet:`Huang.Gokaslan.Kuleshov.ea.2024` in geometric terms, without computing a Jacobian. Both qualifiers carry weight, however. The identification is local, resting on the linearization about a constant critic. It is also formal: it requires $m$ to have a density and the Poisson equation to be solvable, conditions we do not verify here. What the penalty does far from equilibrium is a separate argument, taken up in the next section.

### One-Centered versus Zero-Centered

Zero was not the historical choice of center. WGAN-GP :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017`, enforcing the Lipschitz ball of :numref:`sec_gan_objectives`, penalizes $E\big[(\|\nabla_x D\| - 1)^2\big]$ on points interpolated between real and generated samples: a *one-centered* penalty that pulls the critic's slope toward one, motivated by the fact that optimal $W_1$ potentials have unit slope along transport rays. The two centers disagree exactly at the point that matters. At $q = p$ the optimal critic is constant; the zero-centered penalty is minimized by that critic, while the one-centered penalty rewards unit slope there, so the WGAN-GP critic retains a nonzero gradient at the equilibrium and keeps displacing a generator that has already arrived. On the Dirac-GAN the consequence is exact: the WGAN-GP dynamics do not converge to the equilibrium :cite:`Mescheder.Geiger.Nowozin.2018`.

| penalty | at $q = p$ | induced geometry | Dirac-GAN test |
|:---|:---|:---|:---|
| one-centered, $E\big[(\|\nabla_x D\| - 1)^2\big]$ | rewards unit slope | $W_1$ | does not converge |
| zero-centered, $E_m\big[\|\nabla_x D\|^2\big]$ | rewards the flat critic | linearized $W_2$ | converges for every $\gamma > 0$ |

### Phase Portraits

Circles, outward spiral, contraction: each claim concerns two coupled scalar equations and can be checked by simulating them. For the logistic payoff the field of :eqref:`eq_gan_dirac_flow` is computable directly, since $\ell'(-\psi\theta) = \sigma(\psi\theta)$. The first panel integrates the continuous flow with small fourth-order Runge--Kutta steps; the second takes plain simultaneous gradient steps, with the flow's conserved circle repeated as a dashed line; the third adds the penalty at two weights that :eqref:`eq_gan_dirac_pen` distinguishes, $\gamma = 0.3$ with complex eigenvalues and $\gamma = 1$, the critical damping of the logistic payoff. All three start from the same point $(\theta, \psi) = (1, 1)$, marked in red, with the equilibrium at the cross.

```{.python .input #convergence-phase-portraits}
%%tab pytorch, jax
def dirac_field(theta, psi, gamma=0.0):
    """Simultaneous gradient field of the Dirac-GAN with logistic payoff."""
    s = 1.0 / (1.0 + np.exp(-psi * theta))       # l'(-psi * theta)
    return np.array([psi * s, -theta * s - gamma * psi])

def trajectory(x0, eta, steps, gamma=0.0, flow=False):
    xs, x = [np.array(x0)], np.array(x0)
    for _ in range(steps):
        if flow:                                  # one RK4 step of the ODE
            k1 = dirac_field(*x, gamma)
            k2 = dirac_field(*(x + eta / 2 * k1), gamma)
            k3 = dirac_field(*(x + eta / 2 * k2), gamma)
            k4 = dirac_field(*(x + eta * k3), gamma)
            x = x + eta / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:                                     # one simultaneous GD step
            x = x + eta * dirac_field(*x, gamma)
        xs.append(x)
    return np.stack(xs)

x0 = (1.0, 1.0)
panels = [('gradient flow', [('', trajectory(x0, 0.05, 800, flow=True))]),
          ('simultaneous descent', [('', trajectory(x0, 0.2, 300))]),
          ('descent with penalty',
           [('$\\gamma = 0.3$', trajectory(x0, 0.2, 300, gamma=0.3)),
            ('$\\gamma = 1$', trajectory(x0, 0.2, 300, gamma=1.0))])]
fig, axes = d2l.plt.subplots(1, 3, figsize=(9.5, 3.2))
for ax, (title, paths) in zip(axes, panels):
    for label, path in paths:
        ax.plot(path[:, 0], path[:, 1], lw=1, label=label or None)
    ax.plot(*x0, 'o', ms=4, c='C3')
    ax.plot(0, 0, 'k+', ms=8)
    ax.set_title(title)
    ax.set_xlabel(r'$\theta$')
    ax.set_aspect('equal')
    if len(paths) > 1:
        ax.legend()
axes[0].set_ylabel(r'$\psi$')
t = np.linspace(0, 2 * np.pi, 200)
axes[1].plot(np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), 'k--', lw=0.8)
fig.tight_layout()
```

The portraits agree with the eigenvalues. The integrated flow retraces the same circle revolution after revolution, to visual accuracy, as the conservation law requires. The discrete iterates leave the dashed copy of that circle immediately and cross their own orbit outward on every turn; after 300 small steps they sit at twice the starting distance from the equilibrium. With the penalty, both trajectories fall into the equilibrium: the underdamped one along a shrinking spiral, the critically damped one turning once and then heading in without further rotation. Divergence and convergence here are separated only by the penalty term, which changes nothing about the objective's equilibrium.

## When One Penalty Is Not Enough

The Dirac-GAN cannot distinguish $R_1$ from $R_2$: the linear critic's slope is the same everywhere, so the two penalties coincide there. Its verdict, that either penalty alone stabilizes training, is genuinely correct near equilibrium in general. There the supports of $p$ and $q$ nearly coincide, regularizing the critic under one measure regularizes it under the other, and :eqref:`eq_gan_sobolev` holds with $m$ replaced by either; local convergence needs one penalty, which is the form in which :citet:`Mescheder.Geiger.Nowozin.2018` proved it.

Far from equilibrium the two penalties stop being interchangeable, and a second reading of the penalty, due to :citet:`Roth.Lucchi.Nowozin.ea.2017`, explains why. Up to a weighting factor and a higher-order error term, penalizing $E_p\big[\|\nabla_x D\|^2\big]$ has the same effect on the game as convolving $p$ with Gaussian noise of covariance proportional to $\gamma$, and $R_2$ likewise smooths $q$. Smoothing is exactly what disjoint supports call for: two mutually singular distributions, blurred, overlap everywhere, their density ratio becomes finite, and the divergence between them depends again on how far apart they sit. The penalties implement this convolution analytically, without adding noise to any sample. But the argument needs *both* distributions smoothed: an $R_1$-only critic pays nothing for growing steep in the region where the generator's samples live, so its input gradients there are uncontrolled. :citet:`Huang.Gokaslan.Kuleshov.ea.2024` observe exactly this failure mode: trained with $R_1$ alone, the critic's gradient on generated samples grows without bound and training diverges.

The evidence is empirical and comes with a scope. In R3GAN's ablation, $R_1$-only training diverged early on StackedMNIST, for the classical and the pairing objective alike, and sweeping $\gamma$ from 0.1 to 100 did not rescue it; adding $R_2$ restored convergence in every case :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. Against this stands one prominent counterexample: StyleGAN2 trained successfully at FFHQ scale with $R_1$ alone :cite:`Karras.Laine.Aittala.ea.2020`. Whether one penalty suffices evidently depends on the dataset and on the rest of the recipe. Symmetric smoothing is what the argument above asks for, both penalties are what restored convergence where one failed, and the second penalty costs one additional term in a backward pass that is already being taken.

## The R3GAN Recipe

The chapter's repairs now compose. Take the pairing objective of :numref:`sec_gan_relativistic`, whose landscape has no mode-dropping basins; train its generator on the non-saturating weighting, the lesson of :numref:`sec_basic_gan` that carries over to pairs; and subtract both zero-centered penalties from the critic's objective, which this section has shown makes the equilibrium attracting. The result is the loss that :citet:`Huang.Gokaslan.Kuleshov.ea.2024` train under the name R3GAN, the baseline that made the accumulated GAN trick stack removable.

### The Loss in Code

Three functions implement it. The critic's loss is the pairing objective, written for minimization: $\mathrm{softplus}(D(y) - D(x)) = -\log \sigma(D(x) - D(y))$ for a real sample $x$ and a generated sample $y$, averaged over aligned pairs from the two batches. The generator's loss reverses the ranking rather than negating the critic's loss: it minimizes $\mathrm{softplus}(D(x) - D(y))$, the non-saturating direction, which is what the reference implementation trains even though the paper displays the zero-sum form, the discrepancy :numref:`sec_gan_relativistic` worked out. The penalty function returns the two per-sample squared gradient norms unscaled, and the caller applies $\gamma/2$ and the batch mean, so one function serves any weight and either penalty alone.

The implementations differ only where the frameworks' automatic differentiation differs. PyTorch obtains all per-sample input gradients from one backward pass through the *sum* of the critic's outputs; the sum's gradient with respect to the input batch decomposes row by row because each output depends only on its own input row, which a critic free of batch-mixing layers guarantees (no batch normalization in the critic; the recipe removes normalization layers anyway). The `create_graph=True` flag keeps the gradient differentiable a second time, since the penalty must itself be differentiated with respect to the critic's parameters. JAX writes the same quantity directly as a `vmap` of a per-sample `grad` with the critic module closed over, and its functional autodifferentiation nests the second derivative without any flag. Both versions detach the incoming samples first, with `detach` in PyTorch and `stop_gradient` in JAX, so the penalty's gradient reaches only the critic's parameters and never the process that produced the samples.

```{.python .input #convergence-the-loss-in-code}
%%tab pytorch
#@save
def rpgan_loss_D(critic, real, fake):
    """Relativistic pairing loss for the critic: -E[log sigma(D(x) - D(y))]."""
    return F.softplus(critic(fake) - critic(real)).mean()

#@save
def rpgan_loss_G(critic, real, fake):
    """Non-saturating pairing loss for the generator."""
    return F.softplus(critic(real) - critic(fake)).mean()

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
    """Relativistic pairing loss for the critic: -E[log sigma(D(x) - D(y))]."""
    return jax.nn.softplus(critic(fake) - critic(real)).mean()

#@save
def rpgan_loss_G(critic, real, fake):
    """Non-saturating pairing loss for the generator."""
    return jax.nn.softplus(critic(real) - critic(fake)).mean()

#@save
def r1_r2_penalty(critic, real, fake):
    """Per-sample squared critic input gradients on real (R1) and fake (R2),
    before the gamma/2 scale."""
    def sq_grad_norm(x):
        x = jax.lax.stop_gradient(x)
        grad_fn = jax.grad(lambda xi: critic(xi[None, ...]).squeeze())
        grad = jax.vmap(grad_fn)(x)
        return (grad.reshape(x.shape[0], -1) ** 2).sum(axis=1)
    return sq_grad_norm(real), sq_grad_norm(fake)
```

One scheduling decision is deliberate: the penalties are computed at every critic step. StyleGAN2 amortized its penalty by applying it only every few minibatches, rescaled to compensate, so-called lazy regularization; R3GAN's ablation rejects the trick, finding that it slightly worsened image quality on real datasets and caused outright convergence failure on StackedMNIST and on two-dimensional toy problems :cite:`Huang.Gokaslan.Kuleshov.ea.2024`.

### Principles and Evidence

The loss is the first of six principles that R3GAN distills from its ablations; the rest are optimization and architecture hygiene. In their formulation: (a) a convergent training objective, which the paper states as regularization with $R_1$ and which the trained recipe, like this section, realizes as the pairing loss with both penalties; (b) a small learning rate and no momentum, Adam with $\beta_1 = 0$; (c) no normalization layers in either network; (d) resampling by bilinear interpolation rather than by strided or transposed convolution; (e) leaky ReLU in both networks and no tanh on the generator's output; (f) a modernized residual backbone. They report that violating (a), (b), or (c) often fails outright, while (d) and (e) cost sample quality rather than stability :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. The claim that this list *replaces* the historical trick stack is tested by a stripping experiment on FFHQ-256, roughly matched in parameter count and training budget throughout:

| config | change relative to the previous row | FID |
|:---|:---|---:|
| A | StyleGAN2 baseline | 7.5 |
| B | remove all ten tricks: $z$ normalization, minibatch standard deviation, equalized learning rate, mapping network, style injection, weight demodulation, noise injection, mixing regularization, path-length regularization, lazy regularization | 12.5 |
| C | add the well-behaved loss: RpGAN objective with $R_1$ and $R_2$ | 11.7 |
| D | modernize the backbone: ResNet generator and critic | 10.0 |
| E | widen with grouped convolutions and inverted bottlenecks | 7.0 |

Reading the table from both ends: removing the tricks costs five FID points, and principled loss plus modern architecture wins them all back, ending slightly ahead of the baseline with none of the removed machinery :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. The loss ablation on StackedMNIST, a synthetic benchmark that stacks MNIST digits into a distribution with 1000 known modes :cite:`Metz.Poole.Pfau.ea.2017`, then separates what each component of the loss buys:

| loss | modes recovered (of 1000) | reverse KL |
|:---|---:|---:|
| RpGAN with $R_1$ and $R_2$ | 1000 | 0.078 |
| GAN with $R_1$ and $R_2$ | 693 | 0.927 |
| RpGAN with $R_1$ only | diverged | --- |
| GAN with $R_1$ only | diverged | --- |

The penalties are responsible for convergence, and the pairing objective for coverage: under identical penalties, the classical loss recovers 693 modes at reverse KL 0.927 where the pairing loss recovers all 1000 at 0.078, and without $R_2$ both objectives diverge at every $\gamma$ tried :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. These are single runs on one synthetic dataset; the separation of mechanisms, not its magnitude, is the finding.

Two practical numbers calibrate expectations before the experiment. The penalty weight is not portable: across R3GAN's benchmarks $\gamma$ ranges from 0.05 on CIFAR-10 to 150 on FFHQ-256, scaling with resolution and dataset, so a value tuned on one problem transfers to another only as an order-of-magnitude starting point. And the compute behind the quoted results is substantial: seven hours on eight L40 GPUs for StackedMNIST, four days for CIFAR-10, about three weeks on eight A6000s for FFHQ-256, and about a day on 32 H100s for conditional ImageNet :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. :numref:`sec_dcgan` discusses what changes at that scale; the experiment below runs in minutes.

### Mode Coverage on 25 Gaussians

The closing experiment measures, at toy scale, what the penalties repair. The target is a mixture of 25 Gaussians on a $5 \times 5$ grid, spacing 2 and standard deviation $\sigma = 0.05$, so the modes are far apart relative to their width and coverage is unambiguous: a mode counts as covered if at least one of 10,000 generated samples lands within $3\sigma$ of its center. Three configurations train on identical networks and identical optimization: the non-saturating GAN of :numref:`sec_basic_gan` with no penalty, the same objective with $R_1 + R_2$ added, and the full recipe, RpGAN with $R_1 + R_2$. The generator maps a 64-dimensional latent through two hidden layers of 256 units; the critic mirrors that width with leaky ReLU and, per principle (c), no normalization layers; both use Adam with $\beta_1 = 0$, $\beta_2 = 0.99$, learning rate $2 \cdot 10^{-4}$, batch size 256, and 20,000 generator steps, with $\gamma = 1$ wherever penalties apply.

```{.python .input #convergence-mode-coverage-on-25-gaussians-1}
%%tab pytorch
device = d2l.try_gpu()

def grid_centers(side=5, spacing=2.0):
    coords = (torch.arange(side, dtype=torch.float32) - (side - 1) / 2) * spacing
    gx, gy = torch.meshgrid(coords, coords, indexing='ij')
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

def sample_modes(centers, n, std=0.05):
    idx = torch.randint(0, centers.shape[0], (n,), device=centers.device)
    return centers[idx] + std * torch.randn(n, 2, device=centers.device)

centers = grid_centers().to(device)

def gan_loss_D(critic, real, fake):
    return (F.softplus(-critic(real)) + F.softplus(critic(fake))).mean()

def gan_loss_G(critic, fake):
    return F.softplus(-critic(fake)).mean()

def make_mlp(sizes, act):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)

def train_toy(loss_type, gamma, steps=20000, batch=256, latent=64):
    torch.manual_seed(2)
    net_G = make_mlp([latent, 256, 256, 2], nn.ReLU).to(device)
    net_D = make_mlp([2, 256, 256, 1],
                     lambda: nn.LeakyReLU(0.2)).to(device)
    opt_G = torch.optim.Adam(net_G.parameters(), lr=2e-4, betas=(0.0, 0.99))
    opt_D = torch.optim.Adam(net_D.parameters(), lr=2e-4, betas=(0.0, 0.99))
    for _ in range(steps):
        real = sample_modes(centers, batch)
        with torch.no_grad():
            fake = net_G(torch.randn(batch, latent, device=device))
        if loss_type == 'rp':
            loss_D = rpgan_loss_D(net_D, real, fake)
        else:
            loss_D = gan_loss_D(net_D, real, fake)
        if gamma > 0:
            r1, r2 = r1_r2_penalty(net_D, real, fake)
            loss_D = loss_D + gamma / 2 * (r1 + r2).mean()
        opt_D.zero_grad(), loss_D.backward(), opt_D.step()
        fake = net_G(torch.randn(batch, latent, device=device))
        if loss_type == 'rp':
            loss_G = rpgan_loss_G(net_D, sample_modes(centers, batch), fake)
        else:
            loss_G = gan_loss_G(net_D, fake)
        opt_G.zero_grad(), loss_G.backward(), opt_G.step()
    return net_G

configs = [('GAN', 'gan', 0.0), ('GAN + $R_1 + R_2$', 'gan', 1.0),
           ('RpGAN + $R_1 + R_2$', 'rp', 1.0)]
generators = {name: train_toy(loss_type, gamma)
              for name, loss_type, gamma in configs}
```

```{.python .input #convergence-mode-coverage-on-25-gaussians-1}
%%tab jax
def grid_centers(side=5, spacing=2.0):
    coords = (jnp.arange(side) - (side - 1) / 2) * spacing
    gx, gy = jnp.meshgrid(coords, coords, indexing='ij')
    return jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

def sample_modes(key, centers, n, std=0.05):
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (n,), 0, centers.shape[0])
    return centers[idx] + std * jax.random.normal(k2, (n, 2))

centers = grid_centers()

def gan_loss_D(critic, real, fake):
    return (jax.nn.softplus(-critic(real)) +
            jax.nn.softplus(critic(fake))).mean()

def gan_loss_G(critic, fake):
    return jax.nn.softplus(-critic(fake)).mean()

class ToyMLP(nnx.Module):
    def __init__(self, d_in, d_hidden, d_out, act, rngs):
        self.h1 = nnx.Linear(d_in, d_hidden, rngs=rngs)
        self.h2 = nnx.Linear(d_hidden, d_hidden, rngs=rngs)
        self.out = nnx.Linear(d_hidden, d_out, rngs=rngs)
        self.act = act

    def __call__(self, x):
        return self.out(self.act(self.h2(self.act(self.h1(x)))))

def make_steps(loss_type, gamma, batch=256, latent=64):
    @nnx.jit
    def d_step(net_G, net_D, opt_D, key):
        kr, kz = jax.random.split(key)
        real = sample_modes(kr, centers, batch)
        fake = net_G(jax.random.normal(kz, (batch, latent)))
        def loss_fn(net_D):
            if loss_type == 'rp':
                loss = rpgan_loss_D(net_D, real, fake)
            else:
                loss = gan_loss_D(net_D, real, fake)
            if gamma > 0:
                r1, r2 = r1_r2_penalty(net_D, real, fake)
                loss = loss + gamma / 2 * (r1 + r2).mean()
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(net_D)
        opt_D.update(net_D, grads)
        return loss

    @nnx.jit
    def g_step(net_G, net_D, opt_G, key):
        kr, kz = jax.random.split(key)
        z = jax.random.normal(kz, (batch, latent))
        def loss_fn(net_G):
            fake = net_G(z)
            if loss_type == 'rp':
                real = sample_modes(kr, centers, batch)
                return rpgan_loss_G(net_D, real, fake)
            return gan_loss_G(net_D, fake)
        loss, grads = nnx.value_and_grad(loss_fn)(net_G)
        opt_G.update(net_G, grads)
        return loss
    return d_step, g_step

def train_toy(loss_type, gamma, steps=20000, batch=256, latent=64, seed=2):
    rngs = nnx.Rngs(seed)
    net_G = ToyMLP(latent, 256, 2, nnx.relu, rngs)
    net_D = ToyMLP(2, 256, 1, lambda x: nnx.leaky_relu(x, 0.2), rngs)
    opt_G = nnx.Optimizer(net_G, optax.adam(2e-4, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    opt_D = nnx.Optimizer(net_D, optax.adam(2e-4, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    d_step, g_step = make_steps(loss_type, gamma, batch, latent)
    key = jax.random.PRNGKey(seed)
    for _ in range(steps):
        key, kd, kg = jax.random.split(key, 3)
        d_step(net_G, net_D, opt_D, kd)
        g_step(net_G, net_D, opt_G, kg)
    return net_G

configs = [('GAN', 'gan', 0.0), ('GAN + $R_1 + R_2$', 'gan', 1.0),
           ('RpGAN + $R_1 + R_2$', 'rp', 1.0)]
generators = {name: train_toy(loss_type, gamma)
              for name, loss_type, gamma in configs}
```

The evaluation draws 10,000 samples from each trained generator and reports two groups of statistics. The first group measures reach and evenness of use. A mode counts as covered when at least one sample lands within $3\sigma$ of its center, each such sample is assigned to its nearest center, and the reverse KL divergence of the resulting 25-way histogram from the uniform distribution measures how evenly the covered modes are used. The second group measures how much probability mass those neighborhoods actually receive. The on-mode fraction is the share of samples within $3\sigma$ of some center, printed together with its complement, the off-mode mass, and with the mean and median distance from each sample to its nearest center. A final number aggregates fit into one divergence: the reverse KL of the unconditional 26-bin histogram, the 25 mode bins plus one off-mode bin, from the target mixture's own histogram. In two dimensions a Gaussian places $1 - e^{-9/2} \approx 0.989$ of its mass within $3\sigma$ of its center, so the target histogram is $0.989/25$ per mode bin and $0.011$ in the off-mode bin; a generator that matched the mixture would drive this KL to zero.

```{.python .input #convergence-mode-coverage-on-25-gaussians-2}
%%tab pytorch
def mode_coverage(net_G, n=10000, latent=64, std=0.05):
    torch.manual_seed(1)
    with torch.no_grad():
        x = net_G(torch.randn(n, latent, device=device))
    near, idx = torch.cdist(x, centers).min(dim=1)
    on = near <= 3 * std
    counts = torch.bincount(idx[on], minlength=len(centers)).float().cpu()
    share = counts[counts > 0] / counts.sum()
    rev_kl = float((share * (share * len(centers)).log()).sum())
    on_mass = 1 - np.exp(-4.5)             # P(|x - c| <= 3 sigma) in 2d
    p = torch.cat([torch.full((25,), on_mass / 25),
                   torch.tensor([1 - on_mass])])
    q = torch.cat([counts, (n - counts.sum()).reshape(1)]) / n
    kl26 = float((q[q > 0] * (q[q > 0] / p[q > 0]).log()).sum())
    stats = dict(covered=int((counts > 0).sum()), rev_kl=rev_kl,
                 on_frac=float(on.float().mean()), d_mean=float(near.mean()),
                 d_med=float(near.median()), kl26=kl26)
    return x.cpu(), stats

fig, axes = d2l.plt.subplots(1, 3, figsize=(9.5, 3.4))
for ax, (name, _, _) in zip(axes, configs):
    x, s = mode_coverage(generators[name])
    ax.scatter(x[:3000, 0], x[:3000, 1], s=2, alpha=0.3)
    ax.scatter(centers.cpu()[:, 0], centers.cpu()[:, 1], marker='x', s=30,
               c='C3')
    ax.set_title(f"{name}: {s['covered']}/25 modes")
    ax.set_xlim(-6, 6), ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    print(f"{name}: {s['covered']}/25 modes, reverse KL {s['rev_kl']:.2f}, "
          f"on-mode fraction {s['on_frac']:.2f}")
    print(f"    off-mode mass {1 - s['on_frac']:.2f}, nearest-center "
          f"distance mean {s['d_mean']:.2f} / median {s['d_med']:.2f}, "
          f"26-bin KL {s['kl26']:.2f}")
fig.tight_layout()
```

```{.python .input #convergence-mode-coverage-on-25-gaussians-2}
%%tab jax
def mode_coverage(net_G, n=10000, latent=64, std=0.05):
    x = net_G(jax.random.normal(jax.random.PRNGKey(1), (n, latent)))
    dist = jnp.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
    near, idx = dist.min(axis=1), dist.argmin(axis=1)
    on = near <= 3 * std
    counts = jnp.bincount(idx[on], length=len(centers)).astype(jnp.float32)
    share = counts[counts > 0] / counts.sum()
    rev_kl = float((share * jnp.log(share * len(centers))).sum())
    on_mass = 1 - np.exp(-4.5)             # P(|x - c| <= 3 sigma) in 2d
    p = jnp.append(jnp.full(25, on_mass / 25), 1 - on_mass)
    q = jnp.append(counts, n - counts.sum()) / n
    kl26 = float(jnp.where(q > 0, q * jnp.log(q / p), 0.0).sum())
    stats = dict(covered=int((counts > 0).sum()), rev_kl=rev_kl,
                 on_frac=float(on.mean()), d_mean=float(near.mean()),
                 d_med=float(jnp.median(near)), kl26=kl26)
    return np.asarray(x), stats

fig, axes = d2l.plt.subplots(1, 3, figsize=(9.5, 3.4))
for ax, (name, _, _) in zip(axes, configs):
    x, s = mode_coverage(generators[name])
    ax.scatter(x[:3000, 0], x[:3000, 1], s=2, alpha=0.3)
    ax.scatter(centers[:, 0], centers[:, 1], marker='x', s=30, c='C3')
    ax.set_title(f"{name}: {s['covered']}/25 modes")
    ax.set_xlim(-6, 6), ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    print(f"{name}: {s['covered']}/25 modes, reverse KL {s['rev_kl']:.2f}, "
          f"on-mode fraction {s['on_frac']:.2f}")
    print(f"    off-mode mass {1 - s['on_frac']:.2f}, nearest-center "
          f"distance mean {s['d_mean']:.2f} / median {s['d_med']:.2f}, "
          f"26-bin KL {s['kl26']:.2f}")
fig.tight_layout()
```

The comparison that is stable across reruns and seeds is the left panel against the other two. The plain non-saturating GAN trains without any numerical trouble here: no divergence, no exploding losses. That pathology belongs to the Dirac-GAN analysis. Its failure is one of reach and balance. On no seed we tried does it reach all 25 modes, and its printed reverse KL is markedly higher than either penalized run's. The left panel shows why: mass gathers in dense filaments and clumps while some centers receive none. Both penalized configurations reach all 25 modes on every seed, and their markedly lower reverse KL says the covered modes are used far more evenly.

The second group of printed statistics bounds what that comparison means. The target mixture puts 98.9% of its mass within $3\sigma$ of the centers; every configuration here, penalized or not, places only a small fraction of its mass there --- about a tenth or less in the stored runs --- and the nearest-center distances, whose mean and median run to several times the target's $\sigma = 0.05$, say where the rest sits: the generated per-mode clouds are far wider than the modes they surround. The 26-bin KL is correspondingly dominated by its off-mode bin. The experiment therefore demonstrates support reach and evenness of use --- the penalties turn a generator that misses modes and loads the ones it hits unevenly into one that reaches every mode and spreads its samples across them evenly --- and it does not demonstrate a distributional fit to the mixture, which none of the three configurations achieves at this budget and architecture.

What this experiment cannot show is the difference between the two penalized configurations, and the reason is instructive. At 25 well-separated modes, both sit at the ceiling of the coverage statistic; the advantage of the pairing objective is a claim about many modes under capacity pressure, and it is carried by the StackedMNIST citation above --- 1000 of 1000 modes at reverse KL 0.078 against 693 at 0.927 for the classical loss, with both objectives requiring the penalties to converge at all :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. A two-dimensional toy with 25 modes has neither the mode count nor the starved generator that make that gap visible. The division of labor the toy does state is the scoped one: the penalties buy stable training, full reach, and even use of the modes here, while the pairing objective's additional advantage rests on the cited StackedMNIST evidence. Exercise 4 reconciles the printed statistics with one another.

## Summary

This section answered the question the chapter's analysis had postponed: whether gradient descent finds the equilibria that the objectives define. On the Dirac-GAN, the smallest adversarial game, the answer is exactly no. The simultaneous gradient field is orthogonal to the position, so the continuous flow conserves the distance to the equilibrium and moves on circles, with purely imaginary eigenvalues $\pm i\,\ell'(0)$. Discrete simultaneous descent increases that distance at every step, whatever the step size. Neither the non-saturating weighting nor the pairing objective changes the field near the equilibrium. The zero-centered penalties $R_1$ and $R_2$, with sum $\gamma\, E_m\big[\|\nabla_x D\|^2\big]$, supply the missing damping. The penalized eigenvalues $-\gamma/2 \pm \sqrt{\gamma^2/4 - \ell'(0)^2}$ have negative real part for every $\gamma > 0$, and the dynamics are critically damped at $\gamma = 2\,|\ell'(0)|$. Near equilibrium the penalized game evaluates $\tfrac{a^2}{4\gamma}\, \|p - q\|^2_{\dot H^{-1}(m)}$, the squared linearized Wasserstein-2 distance. That quantity is built from the difference of the distributions rather than their ratio, so locally it is free of the saturation ceiling. The one-centered WGAN-GP penalty instead rewards a sloped critic at the equilibrium and fails the same Dirac test. Far from equilibrium the penalties act by implicitly smoothing the two distributions, which is why one penalty alone can fail. R3GAN's ablation finds $R_1$-only training divergent on StackedMNIST even though StyleGAN2 trained with $R_1$ alone at scale. Objective plus penalties compose into the R3GAN loss, whose ablations attribute convergence to the penalties and mode coverage to the pairing objective. The 25-Gaussians experiment shows the penalties' contribution at toy scale as reach and evenness: the penalized runs cover every mode and use them far more evenly, while all three configurations place only a small fraction of their mass within $3\sigma$ of the centers, so the toy demonstrates support reach, not a fit to the mixture. The pairing objective's advantage is invisible at 25 modes and rests on the cited StackedMNIST result. :numref:`sec_dcgan` carries this loss to images.

## Exercises

1. The discrete simultaneous update of the Dirac-GAN with step size $\eta > 0$ is the map $(\theta, \psi) \mapsto \big(\theta + \eta\,\psi\,\ell'(-\psi\theta),\; \psi - \eta\,\theta\,\ell'(-\psi\theta)\big)$. Compute the Jacobian of this map at the equilibrium $(0, 0)$ and show that its eigenvalues are $1 \pm i\,\eta\,\ell'(0)$, so the spectral radius is $\sqrt{1 + \eta^2\,\ell'(0)^2} > 1$ for every step size. Conclude that no choice of $\eta$ makes the unpenalized game locally convergent. The norm identity in the text shows this growth exactly, at every point of the plane; the eigenvalue computation here recovers the same conclusion in linearized form at the equilibrium.
1. Derive :eqref:`eq_gan_r1r2_sum` from :eqref:`eq_gan_r1r2`. Then explain why the mixture weighting matters: construct a critic and a pair of distributions with separated supports for which $E_p\big[\|\nabla_x D\|^2\big] = 0$ while $E_q\big[\|\nabla_x D\|^2\big]$ is arbitrarily large, and relate your construction to the failure of $R_1$-only training described in the text.
1. Prove :eqref:`eq_gan_sobolev`. Restrict the objective to a ray $\{t D_0 : t \in \mathbb{R}\}$ with $\int m\,\|\nabla_x D_0\|^2 = 1$, maximize the resulting scalar quadratic in $t$, and then take the supremum over directions $D_0$. Verify from the result that the value of the penalized game is proportional to $1/\gamma$, so that doubling the penalty weight halves the value at every fixed pair $(p, q)$.
1. The printed statistics of the mode-coverage experiment appear to conflict: a penalized run covers 25 of 25 modes, yet only a small fraction of its mass lies within $3\sigma$ of any center. (a) Explain how both can be true at once, given that coverage is a threshold statistic over 10,000 samples. (b) From the printed on-mode fraction, compute the off-mode bin's contribution $q_{\textrm{off}} \log(q_{\textrm{off}} / 0.011)$ to the 26-bin KL and compare it with the printed total: how much of the divergence is the off-mode mass alone? (c) Two arrangements are consistent with a small on-mode fraction: per-mode clouds centered on the modes but wider than the target's $\sigma = 0.05$, and mass strewn along paths between modes. Which statistics of the nearest-center distance distribution, beyond the printed mean and median, would distinguish them, and what would each arrangement predict? (d) Without re-running anything, predict from the damping regimes of :eqref:`eq_gan_dirac_pen` which failure each direction of the penalty weight invites: $\gamma = 10$ versus $\gamma = 0.1$.
1. In the middle panel of the phase portrait, sweep the step size over $\eta \in \{0.01, 0.05, 0.2, 0.5\}$ with $\gamma = 0$. Does any step size stabilize the unpenalized game, or change anything other than the rate at which the spiral grows? Reconcile the observation with Exercise 1.

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.4]{.kicker}

Gradient penalties and convergence<br>
**the Dirac-GAN · circles and spirals · zero-centered penalties · the R3GAN recipe**
:::
:::

::: {.slide title="What the Objective Cannot Buy"}
Two failures survive the chapter's objective repairs:

- The pairing objective removed the mode-dropping basins — but still
  **saturates on disjoint supports**: no gradient once samples and data
  separate.
- Every result so far evaluated the game at the critic's **best response**.
  Training takes coupled gradient steps — and nothing yet says those steps
  approach the equilibrium.

. . .

Both failures are exact on the smallest example: one point mass chasing
another.
:::

::: {.slide title="The Dirac-GAN"}
$p = \delta_0$, $q_\theta = \delta_\theta$, linear critic $D_\psi(x) = \psi x$:

$$V(\theta, \psi) = \ell(0) + \ell(-\psi\theta)$$

Simultaneous descent–ascent flow:

$$\dot\theta = \psi\,\ell'(-\psi\theta), \qquad \dot\psi = -\theta\,\ell'(-\psi\theta)$$

. . .

- The field is orthogonal to the position: $\tfrac{d}{dt}(\theta^2 + \psi^2) = 0$.
- **Exact circles**; eigenvalues $\pm i\,\ell'(0)$ — rotation, no attraction.
:::

::: {.slide title="Discretization Turns Circles into Outward Spirals"}
One simultaneous gradient step, by Pythagoras (field $\perp$ position):

$$\|(\theta, \psi) + \eta v\|^2 = \|(\theta, \psi)\|^2 + \eta^2 \|v\|^2$$

- Every step increases the distance to the solution — **for every step size**.
- Update-map eigenvalues $1 \pm i\eta\,\ell'(0)$: spectral radius $> 1$ always.
- Non-saturating weighting and the pairing objective leave the field at the
  equilibrium unchanged — the failure is in the dynamics, not the objective.
:::

::: {.slide title="Zero-Centered Penalties Add Damping"}
$$R_1 = \tfrac{\gamma}{2} E_p\big[\|\nabla_x D\|^2\big], \quad
R_2 = \tfrac{\gamma}{2} E_q\big[\|\nabla_x D\|^2\big], \quad
R_1 + R_2 = \gamma\, E_m\big[\|\nabla_x D\|^2\big]$$

On the Dirac-GAN: $\nabla_x D_\psi = \psi$, so either penalty is
$\tfrac{\gamma}{2}\psi^2$ — the two coincide here, and using both doubles the
damping ($\gamma \to 2\gamma$) — giving

$$\lambda_{1,2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^2}{4} - \ell'(0)^2}$$

- Negative real part for **every** $\gamma > 0$; critically damped at
  $\gamma = 2|\ell'(0)|$.
- (R3GAN's Eq. 12 prints this without the square — a typo; the Jacobian's
  determinant is $\ell'(0)^2$.)
:::

::: {.slide title="The Penalized Game Measures Linearized W2"}
Near equilibrium every payoff linearizes to $a\,\langle p - q, D\rangle$, and

$$\sup_D \Big\{ a \langle p - q, D\rangle - \gamma \int m \|\nabla_x D\|^2 \Big\}
= \frac{a^2}{4\gamma}\, \|p - q\|^2_{\dot H^{-1}(m)}$$

— the squared **linearized $W_2$ distance**: a function of the *difference*
$p - q$, not the ratio. No saturation ceiling, locally.

| penalty | at $q = p$ | geometry | Dirac test |
|:--|:--|:--|:--|
| one-centered (WGAN-GP) | rewards unit slope | $W_1$ | fails |
| zero-centered ($R_1$, $R_2$) | rewards flat critic | linearized $W_2$ | converges |
:::

::: {.slide title="When One Penalty Is Not Enough"}
- Near equilibrium: supports overlap, either penalty regularizes both
  measures — one suffices (that is the theorem).
- Far from equilibrium: the penalty acts by **implicit smoothing**
  (Roth et al.) — and smoothing $p$ alone leaves the critic free to steepen
  on $q$: gradients on fakes grow without bound.
- R3GAN ablation: $R_1$-only **diverges** on StackedMNIST, both objectives,
  $\gamma$ swept 0.1–100. Counterpoint: StyleGAN2 trained with $R_1$ alone
  at FFHQ scale.
:::

::: {.slide title="The Full Loss in Code"}
Pairing objective (non-saturating generator) + both penalties. The penalty
returns per-sample $\|\nabla_x D\|^2$ unscaled; the caller applies
$\gamma/2$ and the mean — every step, since lazy regularization fails on toys:

@convergence-the-loss-in-code
:::

::: {.slide title="Phase Portraits Confirm the Eigenvalues"}
@!convergence-phase-portraits

Same start point $(\theta, \psi) = (1, 1)$ in all three panels. The flow
retraces its circle; the discrete iterates cross it outward on every turn;
with the penalty, $\gamma = 0.3$ spirals in and $\gamma = 1$ is critically
damped — it turns once and heads in without rotation.
:::

::: {.slide title="Mode Coverage on 25 Gaussians"}
@!convergence-mode-coverage-on-25-gaussians-2

- Plain GAN: stable training, but it never reaches all 25 modes and uses
  them unevenly (markedly higher reverse KL).
- Either penalized configuration: all 25 modes on every seed, used far more
  evenly — and no visible difference between the two penalized losses at
  this scale.
- Every configuration puts only a small fraction of mass within $3\sigma$
  of the centers — the per-mode clouds are far wider than $\sigma = 0.05$.
  **Reach and even use, not a fit to the mixture.**
:::

::: {.slide title="What the Toy Cannot Show"}
The RpGAN-vs-GAN coverage gap needs many modes under capacity pressure —
StackedMNIST (cited):

| loss | modes / 1000 | reverse KL |
|:--|--:|--:|
| RpGAN + $R_1$ + $R_2$ | 1000 | 0.078 |
| GAN + $R_1$ + $R_2$ | 693 | 0.927 |
| either + $R_1$ only | diverged | — |

In the toy, penalties buy stable training, full reach, and even use; the
pairing objective's coverage advantage is this cited evidence.
$\gamma$ is dataset-dependent (0.05–150 across R3GAN's benchmarks).
:::

::: {.slide title="Recap"}
- Correct objective $\neq$ trainable: Dirac-GAN flow **circles**; discrete
  descent **spirals out** at every step size.
- Zero-centered penalties damp the game: convergence for every $\gamma > 0$;
  critical damping at $\gamma = 2|\ell'(0)|$.
- Near equilibrium the penalized game $=$ scaled **linearized $W_2^2$** —
  difference, not ratio; one-centered penalties fail the same test.
- Far from equilibrium: implicit smoothing — of **both** distributions.
- RpGAN + $R_1$ + $R_2$ = the R3GAN loss: penalties → convergence (in the
  toy: reach and even use, not a fit), pairing → coverage (StackedMNIST).
  Next: images.
:::
