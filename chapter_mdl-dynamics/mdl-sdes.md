# Stochastic Differential Equations
:label:`sec_mdl-sdes`

Every trajectory in the previous sections was fixed by its start, and a flow
never forgets where it came from: two starts always reach two different
ends. The diffusion models of :numref:`chap_diffusion` are built on
a process that does forget. Their *forward process* mixes new noise into its
state, a little at a time, until nothing of the start remains, and
generation, which runs it backward, has a known place to begin. Such a
process solves a differential equation too, but not one that ordinary
calculus can handle.

Let us begin with a case small enough to follow by hand. Take one number,
$x_0 = 2$. At each step keep $99\%$ of it and add an independent Gaussian
draw of standard deviation $\sqrt{0.02} \approx 0.14$:

$$
x_{k+1} = 0.99\, x_k + \sqrt{0.02}\; \xi_k, \qquad \xi_k \sim \mathcal{N}(0, 1).
$$

After $400$ steps the mean is $2 \cdot 0.99^{400} \approx 0.036$, and the
variance is close to $1$, where the $2\%$ that a step removes balances the
$0.02$ that it adds. Any other start ends in the same place, a draw from
nearly $\mathcal{N}(0, 1)$. One feature of the recurrence looks odd. If it
describes a process in continuous time with time step $h = 0.01$, then the
decay $0.99 = 1 - h$ changes $x$ by an amount of order $h$, but the noise
has size $\sqrt{h}$, not $h$. This choice is forced.
Noise of size $h$ would vanish as $h \to 0$, while noise of size $\sqrt{h}$
survives, and its square is of order $h$, the same order as the drift.
Ordinary calculus discards squares of small changes, so this one fact
requires a new calculus.

The limit of such noise is Brownian motion, named after the botanist Robert
Brown, who in 1827 watched particles from pollen grains jitter in water.
Norbert Wiener made it a mathematical object in 1923, and Kiyoshi Itô built
its calculus in the
1940s :cite:`Ito.1944` on one rule: the squares of short Brownian increments
add up to the elapsed time, so the chain rule gains a second-derivative
term.

We build Brownian motion from coin flips and the calculus from that rule.
Then we solve the process behind the recurrence, the forward process of
:numref:`sec_diffusion-gaussian-paths`. Given the start, its distribution at
every time is a Gaussian, so a diffusion model can noise its data to any
level in one draw. Simulation, and the measurement of its error, can wait
for a second reading. Only the rule for squares is hard to believe, and its
proof takes six lines.

```{.python .input #sdes-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #sdes-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #sdes-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #sdes-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Brownian Motion

### Noise as a Forward Process
:label:`sec_mdl-why-randomness`

Before building any machinery, let us check what a forward process must do.
Starting from data distributed as $p_0 = p_{\textrm{data}}$, it must end at
one known distribution, whatever the data looked like,

$$
p_0 = p_{\textrm{data}} \;\longrightarrow\; p_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

so that generation always knows where to start. No fixed flow does this for
two datasets at once. A flow is an invertible map $\Phi$
(:numref:`sec_mdl-odes-solvers`), so if it carried two distributions $p$ and
$q$ to the same one, applying $\Phi^{-1}$ would show $p = q$.

Let us try three processes on two "datasets" that are as simple as datasets
can be: point masses at $x_0 = 2$ and at $x_0 = -2$.

- **Drift alone**, $\dot{x} = -x$. The solution $x(t) = x_0 e^{-t}$ puts the
  two starts at $\pm 2e^{-3} = \pm 0.0996$ at $t = 3$. They approach the
  point $0$, which is not a Gaussian, and at every finite time they are still
  two distinct points.
- **Noise alone**, $dX = \sqrt{2}\, dW$, which means
  $X_t = x_0 + \sqrt{2}\, W_t$ for a noise $W$ whose variance grows linearly
  in time (the next subsection constructs it). The distribution of $X_t$,
  its *law*, is $\mathcal{N}(x_0, 2t)$: each cloud keeps its center at
  $\pm 2$ while its standard deviation $\sqrt{2t}$ grows without bound ($2.45$ at $t = 3$). The
  start is forgotten only relative to this growing width, and there is no
  limiting law.
- **Both together**, $dX = -X\,dt + \sqrt{2}\, dW$. From either start the law
  at $t = 3$ is $\mathcal{N}(\pm 0.0996,\, 0.9975)$, as
  :numref:`sec_mdl-ornstein-uhlenbeck` will prove below (and as its density
  evolution $\partial_t p_t = -\nabla \cdot (\mathbf{f} p_t) + \tfrac{1}{2}g^2 \Delta p_t$
   attains at stationarity $\partial_t p_t = 0$, previewing the Fokker--Planck
  partial differential equation that :numref:`sec_mdl-fokker-planck-probability-flow`
  develops next). The two laws are nearly
  indistinguishable from each other and from $\mathcal{N}(0, 1)$.

![One failure, one usable process and the one we use, from the two starts $x_0=\pm2$. Drift alone sends every start to the point $0$. Noise alone keeps each start as the center of a cloud whose two-standard-deviation envelope $x_0\pm2\sqrt{2t}$ (dashed) grows without bound. Both together carry every start into the same limiting law $\mathcal N(0,1)$, drawn sideways at the right. Green lines are the means.](../img/mdl-dyn-sde-paths.svg)
:label:`fig_mdl-dyn-sde-paths`

:numref:`fig_mdl-dyn-sde-paths` draws the three. Only the first fails
outright. The second is usable, and is in use. Once its variance $2t$ is far
larger than that of the data, the start hardly matters. Processes of this
kind are called *variance-exploding*, because without a limiting law they
must make their final variance huge. The third is the running example of
this section, and the recurrence of the opening simulates it. Its drift pulls each start toward zero while the noise refills
the variance that the drift removes, and the two balance at variance $1$.

In general, a forward process of this kind is a deterministic vector field
plus a noise term. For data in $\mathbb{R}^d$ we write

$$
d\mathbf{X} \;=\; \underbrace{\mathbf{f}(\mathbf{X}, t)\,dt}_{\textrm{drift}}
\;+\; \underbrace{g(t)\,d\mathbf{W}}_{\textrm{diffusion}},
$$
:eqlabel:`eq_mdl-sde-template`

which reads: over a short time $dt$, the state moves by the velocity
$\mathbf{f}$ times $dt$ and receives a random kick whose size is set by the
noise amplitude $g$. The drift steers the mean, and the diffusion injects
variance. So far the kick $d\mathbf{W}$ is only a symbol. We give it a
meaning in one coordinate, and return to $\mathbb{R}^d$ only at the end of
the Itô calculus.

### From Random Walks to the Wiener Process
:label:`sec_mdl-wiener-process`

We put an object behind the symbol by taking the limit of the simplest random
process there is, a walk driven by coin flips. The limit explains the square
root in our recurrence.

Fix a step duration $\Delta t$ and let a walker take independent steps of size
$\pm\sqrt{\Delta t}$, each sign chosen by a fair coin. After $k$ steps, at time
$t = k\,\Delta t$, the position is a sum of $k$ independent mean-zero steps of
variance $\Delta t$ apiece. Variances of independent terms add
(:numref:`subsec_mdl-covariance`), so

$$
\mathbb{E}\!\left[W^{(\Delta t)}_t\right] = 0,
\qquad
\textrm{Var}\!\left(W^{(\Delta t)}_t\right) = k\,\Delta t = t,
$$

whatever the step size. With $\Delta t = 0.01$, for instance, $100$ steps of
$\pm 0.1$ give variance $100 \cdot 0.01 = 1$ at time $1$. Steps of $\pm 0.01$
on the same clock would give variance $0.01$, and steps of $\pm 1$ would give
$100$. In general, steps of size $c\,\Delta t^{a}$ give variance
$c^2\, t\, \Delta t^{2a - 1}$ at time $t$,
which tends to $0$ for $a > \tfrac12$ and to $\infty$ for $a < \tfrac12$.
Only the square root leaves a finite nonzero variance in the limit, which is
why it returns in the noise term of every numerical scheme below.

As $\Delta t \to 0$ the number of steps before time $t$ grows without bound,
and the central limit theorem (:numref:`subsec_mdl-gaussian-clt`) makes the
position Gaussian: a sum of many independent mean-zero terms with total
variance $t$ is approximately $\mathcal{N}(0, t)$.

The limit process is called **Brownian motion** or the **Wiener process**
:cite:`Wiener.1923`. It is defined by four properties:

1. $W_0 = 0$;
2. **independent increments**: for $s < t$, the increment $W_t - W_s$ is
   independent of the entire path up to time $s$;
3. **Gaussian increments**: $W_t - W_s \sim \mathcal{N}(0,\, t - s)$;
4. continuous sample paths.

That a process with these properties exists is Wiener's theorem, and that the
rescaled walks converge to it is Donsker's theorem; we take both as given.
The process spreads with standard deviation $\sqrt{t}$ forever
(:numref:`fig_mdl-dyn-brownian-paths`), and its increments have the form

$$
\Delta W \;=\; W_{t + \Delta t} - W_t \;=\; \sqrt{\Delta t}\;\xi,
\qquad \xi \sim \mathcal{N}(0, 1),
$$
:eqlabel:`eq_mdl-sde-increment`

a standard normal draw scaled by $\sqrt{\Delta t}$.

![One coin-flip walk of $1000$ steps $\pm\sqrt{0.001}$, viewed on meshes $\Delta t=0.1$, $0.01$ and $0.001$ (faint gray: the full path). Each view is itself a random walk with step variance $\Delta t$; refining the mesh fills in detail without changing the large-scale path. At each fixed time the spread follows the $\pm\sqrt t$ and $\pm2\sqrt t$ envelopes.](../img/mdl-dyn-brownian-paths.svg)
:label:`fig_mdl-dyn-brownian-paths`

Independent increments also fix how the values of $W$ at two times are
related. For $s \le t$ write $W_t = W_s + (W_t - W_s)$. The increment is
independent of $W_s$ and has mean zero, so
$\textrm{Cov}(W_s, W_t) = \mathbb{E}[W_s^2] = s = \min(s, t)$: the walk
carries its position at time $s$ forward and adds noise that is uncorrelated
with it. The formula is worth more than its one-line proof suggests.
Jointly Gaussian values are determined by their means and covariances, so a
process with jointly Gaussian values at any finite set of times, mean zero,
covariance $\min(s, t)$ and continuous paths is a Brownian motion.

The same increments make the path rough. Brownian paths are continuous but
nowhere differentiable, a theorem of Paley, Wiener, and Zygmund
:cite:`Paley.Wiener.Zygmund.1933`. The increment scaling
:eqref:`eq_mdl-sde-increment` gives the heuristic reason:

$$
\left|\frac{\Delta W}{\Delta t}\right|
= \frac{|\xi|}{\sqrt{\Delta t}}
\;\longrightarrow\; \infty
\quad \textrm{in probability as } \Delta t \to 0,
$$

so a difference quotient grows without bound as the step shrinks. This is not
a proof (it rules out a derivative at each fixed time, while "nowhere"
concerns all times at once), but it shows the obstruction.

Rescaling time by a factor $c$ and space by $\sqrt{c}$ returns a Brownian
motion (Exercise :exref:`ex_mdl-sdes-brownian-scaling-exit-times`), so zooming
in on a path shows the same roughness at every scale, where a differentiable
curve would flatten into its tangent. Hence $dW/dt$ does not exist, and $dW$
has to be given a meaning inside integrals. The Itô calculus supplies that
meaning.

### Simulation of Brownian Paths

The increment form :eqref:`eq_mdl-sde-increment` is also a simulation recipe:
take cumulative sums of $\sqrt{\Delta t}\,\xi$ steps. We generate $20{,}000$
paths on $[0, 1]$ in $500$ steps and put a question to them. At any one time
$W_t \sim \mathcal{N}(0, t)$, so $95.4\%$ of the paths lie inside
$\pm 2\sqrt{t}$, the outer envelope of :numref:`fig_mdl-dyn-brownian-paths`.
What fraction stays inside the envelope at all times? It is worth a guess
before running the cell.

```{.python .input #sdes-brownian-paths}
rng = np.random.default_rng()
T_bm, n_bm, n_paths = 1.0, 500, 20000
dt_bm = T_bm / n_bm
dW_bm = np.sqrt(dt_bm) * rng.standard_normal((n_paths, n_bm))
W_bm = np.cumsum(dW_bm, axis=1)              # W at t = dt, 2 dt, ..., 1
band = 2 * np.sqrt(dt_bm * np.arange(1, n_bm + 1))  # 2 sqrt(t) on the grid
inside = np.abs(W_bm) < band
print(f'inside +-2 sqrt(t) at t = 1:   {inside[:, -1].mean():.3f}'
      '   (Gaussian: 0.954)')
for stride in (10, 1):                       # same paths, coarse then fine grid
    print(f'inside at all {n_bm // stride} grid times: '
          f'{inside[:, stride - 1::stride].all(axis=1).mean():.3f}')
```

At the single time $t = 1$ the fraction is about $95\%$, as it must be. But
only about seven paths in ten are inside at each of $50$ equally spaced
times, and only a little over half at all $500$. A band that holds at each
time does not hold for a path: every look is another chance to find the path
outside. Watched at every instant, no path stays inside, since near $t = 0$
a Brownian path crosses $\pm 2\sqrt{t}$ again and again (the law of the
iterated logarithm, which we do not prove). A statement about the law at one time is
much weaker than a statement about whole paths, and the two return as the
weak and the strong error of a simulation
(:numref:`sec_mdl-sde-strong-weak`).

## Itô Calculus

### Quadratic Variation
:label:`sec_mdl-quadratic-variation`

Ordinary calculus stops its Taylor expansions at first order because the
square of a small change is negligible. For Brownian motion this fails, and
the result of this subsection says by how much. It is the one on which the
rest of the section rests.

Take a partition $0 = t_0 < t_1 < \cdots < t_n = t$ with increments
$\Delta t_i = t_{i+1} - t_i$ and mesh $h_{\max} = \max_i \Delta t_i$, and add
up the squared increments of a path along it. For a differentiable path $x$
with $|x'| \le M$, the mean value theorem gives
$|\Delta x_i| \le M\, \Delta t_i$, so

$$
\sum_{i} (\Delta x_i)^2
\;\le\; M^2 \sum_i \Delta t_i^2
\;\le\; M^2\, h_{\max}\, t
\;\longrightarrow\; 0
$$

as the mesh shrinks: a smooth increment squared is of order $\Delta t^2$. A
Brownian increment squared is of order $\Delta t$, since
$\mathbb{E}[(\Delta W_i)^2] = \Delta t_i$, and these add up to $t$ however
fine the mesh (:numref:`fig_mdl-dyn-qv-convergence`). The sum is random, but
its fluctuations vanish in the limit.

**Proposition (quadratic variation of Brownian motion).** *As the mesh
$h_{\max} \to 0$,*

$$
Q_n \;=\; \sum_{i=0}^{n-1} \left(W_{t_{i+1}} - W_{t_i}\right)^2
\;\longrightarrow\; t
\quad \textrm{in mean square:} \quad
\mathbb{E}\!\left[(Q_n - t)^2\right] \to 0.
$$
:eqlabel:`eq_mdl-sde-qv`

**Proof.** Each increment is $\mathcal{N}(0, \Delta t_i)$, so
$\mathbb{E}[(\Delta W_i)^2] = \Delta t_i$ and
$\mathbb{E}[Q_n] = \sum_i \Delta t_i = t$ for every partition. For the
spread, a Gaussian has fourth moment $3\,\Delta t_i^2$, so
$\textrm{Var}\!\left((\Delta W_i)^2\right) = 3\Delta t_i^2 - \Delta t_i^2
= 2\Delta t_i^2$; the increments are independent, so their variances add:

$$
\mathbb{E}\!\left[(Q_n - t)^2\right] = \textrm{Var}(Q_n)
= \sum_i 2\,\Delta t_i^2
\;\le\; 2\,h_{\max} \sum_i \Delta t_i = 2\,h_{\max}\, t \;\longrightarrow\; 0,
$$

and mean-square convergence follows. $\blacksquare$

The statement deserves a second look: every increment is random, and in the
limit the sum of their squares is not. On a uniform grid with $n$ intervals
of $[0, 1]$ the proof gives the sum a standard deviation of $\sqrt{2/n}$,
which is $0.35$ for $n = 16$ and $0.003$ for $n = 2^{18}$.

![Running sums of squared increments along one Brownian path for meshes of $n = 16$, $256$, and $4096$ intervals (darker shades are finer meshes). The sums approach $y = t$, whereas the sums for the smooth path $\sin 2\pi t$ (gray) shrink like $2\pi^2/n$.](../img/mdl-dyn-qv-convergence.svg)
:label:`fig_mdl-dyn-qv-convergence`

The proposition is usually written as the **Itô multiplication table**, the
working rules for products of differentials:

$$
(dW)^2 = dt, \qquad dW\,dt = 0, \qquad (dt)^2 = 0.
$$
:eqlabel:`eq_mdl-sde-ito-table`

The first rule is :eqref:`eq_mdl-sde-qv` in shorthand, and it is a statement
about sums. A single squared increment is $\Delta t\,\xi^2$, which is rarely
close to $\Delta t$: about one time in four it is below a tenth of
$\Delta t$, and about one time in six it is above twice $\Delta t$. Only the
sum over many steps settles, as the proof showed. (For the coin-flip walk
every squared step equals $\Delta t$, and nothing needs to settle.) The other
two rules record that $dW\,dt$, of size $\Delta t^{3/2}$, and $(dt)^2$ vanish
faster than $\Delta t$ and contribute nothing in the limit.

At fine meshes the fluctuations are so small that a single path shows the
limit. We sample a path on $2^{18}$ intervals of $[0, 1]$, view it on coarser
subgrids, and print its sum of squares next to the predicted standard
deviation. For comparison we print the same sum for the smooth path
$x(t) = \sin 2\pi t$ next to $2\pi^2/n$, the mesh $1/n$ times
$\int_0^1 x'(t)^2\,dt = 2\pi^2$.

```{.python .input #sdes-quadratic-variation}
n_fine = 2**18
dW_qv = np.sqrt(1.0 / n_fine) * rng.standard_normal(n_fine)
W_qv = np.concatenate([[0.0], np.cumsum(dW_qv)])
print(f'{"n":>7} {"QV of W":>8} {"sqrt(2/n)":>10} {"QV of sin":>10}'
      f' {"2 pi^2/n":>9}')
for k in range(4, 19, 2):
    n = 2**k
    W_n = W_qv[::n_fine // n]                 # the same path on a coarser grid
    s_n = np.sin(2 * np.pi * np.linspace(0, 1, n + 1))
    print(f'{n:7d} {np.sum(np.diff(W_n)**2):8.4f} {np.sqrt(2 / n):10.4f} '
          f'{np.sum(np.diff(s_n)**2):10.6f} {2 * np.pi**2 / n:9.6f}')
```

The Brownian column scatters around $1$ with the standard deviation
$\sqrt{2/n}$ that the proof predicts. On $16$ intervals it typically lands
between about $0.5$ and $1.6$, and at $n = 2^{18}$ it is within about $0.01$
of $1$. The smooth column is still $1.22$ at $n = 16$, but it shrinks like
$2\pi^2/n$ and agrees with that prediction to three digits from $n = 256$
on.

### The Itô Integral
:label:`sec_mdl-ito-integral`

The rough path forces us to define integrals against $dW$ directly. We start
with the simplest one, $\int_0^t W\,dW$, and find that the usual Riemann-sum
definition does not single out an answer.

For a smooth integrator, Riemann sums with the integrand evaluated at the left
endpoint, the right endpoint, or the midpoint of each interval all converge to
the same limit. Here they do not. With $W$ itself as the integrand, the
right-endpoint sum exceeds the left-endpoint sum by

$$
\sum_i W_{t_{i+1}}\,\Delta W_i - \sum_i W_{t_i}\,\Delta W_i
= \sum_i \left(W_{t_{i+1}} - W_{t_i}\right)\Delta W_i = \sum_i (\Delta W_i)^2
\;\longrightarrow\; t,
$$

by the proposition just proved. The two discretizations of "the same"
integral disagree by a finite, deterministic amount. Let us look at one path
with $1000$ steps.

```{.python .input #sdes-left-right-sums}
n_lr = 1000
dt_lr = 1.0 / n_lr
t_lr = np.linspace(0, 1, n_lr + 1)
dW_lr = np.sqrt(dt_lr) * rng.standard_normal(n_lr)
W_lr = np.concatenate([[0.0], np.cumsum(dW_lr)])
left = np.sum(W_lr[:-1] * dW_lr)              # integrand at the left endpoint
right = np.sum(W_lr[1:] * dW_lr)              # integrand at the right endpoint
print(f'left sum {left:.4f}   right sum {right:.4f}')
print(f'right - left {right - left:.4f}   sum of (dW)^2 {np.sum(dW_lr**2):.4f}')
```

The left sum is a random number that changes from run to run. The right sum
exceeds it by the sum of squared increments in every printed digit, as the
identity above says it must. What the run adds is the size of the gap. It
usually lies within about $0.1$ of $1$ (its standard deviation is
$\sqrt{2/1000} \approx 0.045$), and a finer mesh would not close it.

Both sums have limits, and we can find them with what we have. Their total
telescopes, because
$\left(W_{t_{i+1}} + W_{t_i}\right)\Delta W_i = W_{t_{i+1}}^2 - W_{t_i}^2$
adds up to $W_t^2$ on every partition, and their difference tends to $t$.
Hence the left sum tends to $\tfrac12\left(W_t^2 - t\right)$ and the right
sum to $\tfrac12\left(W_t^2 + t\right)$, in mean square. Ordinary calculus
would answer $\tfrac12 W_t^2$, which is neither limit but their average. We
must choose an evaluation point, and the choice changes the answer. Which of
the two sums should be the integral?

Itô's choice is the left endpoint, so that the integrand never sees the
coming increment. A fair game explains why. Think of
$\Delta W_i$ as the outcome of the $i$-th fair bet and of $H_i$ as the stake
placed on it. A gambler must choose the stake from what is known at time
$t_i$, before the bet is settled, and the total winnings are
$\sum_i H_i\,\Delta W_i$. Because the increment $\Delta W_i$ is independent of
everything known at $t_i$, including $H_i$, each bet has expected winnings
$\mathbb{E}[H_i\,\Delta W_i] = \mathbb{E}[H_i]\,\mathbb{E}[\Delta W_i] = 0$, so
no betting strategy has positive expected winnings. A stake that could peek at
the outcome would make money on average. The right-endpoint sum places such
a stake, and its mean is indeed $t$, not zero.

The definition writes this rule down. An integrand is *predictable* if its
value on each interval is decided before the increment over that interval is
drawn. A **simple predictable process** $H$ is
constant on each interval, $H_s = H_i$ for $s \in (t_i, t_{i+1}]$, where $H_i$
is determined by the path up to time $t_i$ and $\mathbb{E}[H_i^2] < \infty$.
Its **Itô integral** is the sum of the winnings,

$$
\int_0^t H_s \, dW_s
\;=\; \sum_i H_i\left(W_{t_{i+1}}-W_{t_i}\right).
$$
:eqlabel:`eq_mdl-sde-ito-integral`

The fair-game argument shows that it has mean zero,
$\mathbb{E}\!\left[\int_0^t H\,dW\right] = 0$. Its variance is computable as
well.

**Proposition (Itô isometry).** *For a simple predictable $H$,*

$$
\mathbb{E}\!\left[\left(\int_0^t H_s\,dW_s\right)^{\!2}\,\right]
\;=\; \int_0^t \mathbb{E}\!\left[H_s^2\right] ds.
$$
:eqlabel:`eq_mdl-sde-ito-isometry`

**Proof.** Expand the square of the sum in
:eqref:`eq_mdl-sde-ito-integral`. The step that needs care is the cross terms.
A cross term with $i < j$ contains the factor $\Delta W_j$, which is
independent of everything known at time $t_j$, including the product
$H_i\,\Delta W_i\,H_j$. So the expectation factorizes:
$\mathbb{E}\!\left[H_i\,\Delta W_i\, H_j\, \Delta W_j\right] =
\mathbb{E}\!\left[H_i\,\Delta W_i\, H_j\right]\mathbb{E}[\Delta W_j] = 0$.
A diagonal term factorizes the same way into
$\mathbb{E}\!\left[H_i^2\right]\mathbb{E}\!\left[(\Delta W_i)^2\right] =
\mathbb{E}\!\left[H_i^2\right]\Delta t_i$. Summing the diagonal gives
$\sum_i \mathbb{E}[H_i^2]\,\Delta t_i = \int_0^t \mathbb{E}[H_s^2]\,ds$.
$\blacksquare$

The isometry also carries the definition beyond step processes, to
integrands such as $H = W$ itself. It says that $H \mapsto \int H\,dW$
preserves the norm $\|H\|^2 = \int_0^t \mathbb{E}[H_s^2]\,ds$. So if simple
predictable processes converge to $H$ in this norm, their integrals converge
in mean square, and the limit is taken as the definition of $\int_0^t H\,dW$.
The zero mean and the isometry survive the limit and hold for every
predictable $H$ of finite norm, and the left-endpoint sums of a continuous
integrand of this kind converge to its integral. We do not prove these
statements; see :citet:`Oksendal.2003`, chapter 3.

The integral we started from now has a value, the limit of its left-endpoint
sums:

$$
\int_0^t W_s\,dW_s = \tfrac12\left(W_t^2 - t\right).
$$
:eqlabel:`eq_mdl-sde-wdw`

This is the answer of ordinary calculus minus $t/2$. The $-t/2$ gives it the
mean zero that a fair game must have, since $\mathbb{E}[W_t^2] = t$. The
isometry can be checked on it too. It turns the variance of a noise integral
into an ordinary integral of a second moment, here
$\int_0^1 \mathbb{E}[W_s^2]\,ds = \int_0^1 s\,ds = \tfrac12$ at $t = 1$, and
the closed form agrees: $W_1^2$ is a squared standard normal, with variance
$2$, so $\textrm{Var}\big(\tfrac12(W_1^2 - 1)\big) = \tfrac14 \cdot 2 = \tfrac12$.

The other evaluation points define other integrals. The midpoint choice, or
the average of the left and right sums, defines the *Stratonovich integral*
:cite:`Stratonovich.1966`. It obeys the ordinary chain rule, as the average
$\tfrac12 W_t^2$ above suggests, but gives up the zero mean, because its
integrand sees half of the coming increment. This book
uses Itô's convention, for two reasons: the zero mean makes expectations
easy, and the simplest simulation step, which evaluates the coefficients at
the start of the step, is an Itô sum
(Exercise :exref:`ex_mdl-sdes-ito-versus-stratonovich`).

### Drift, Diffusion, and Solutions
:label:`sec_mdl-sde-definition`

With the integral in hand, the forward-process template
:eqref:`eq_mdl-sde-template` acquires a meaning. A **stochastic differential
equation** is written

$$
dX = f(X, t)\,dt + g(X, t)\,dW,
$$
:eqlabel:`eq_mdl-sde-general`

and the differential form is shorthand for the integral equation

$$
X_t = X_0 + \int_0^t f(X_s, s)\,ds + \int_0^t g(X_s, s)\,dW_s,
$$

an ordinary integral plus an Itô integral. Over a short step, with the
coefficients frozen at its start, $\mathbb{E}[dX] = f\,dt$ and
$\textrm{Var}(dX) = g^2\,dt$. So the **drift** $f$ is the conditional mean
velocity, and the **diffusion** coefficient $g$ sets the rate $g^2$ at which
variance is pumped in. With $g \equiv 0$ the equation is the ODE of
:numref:`sec_mdl-odes-solvers`, and with $f \equiv 0$ and a constant $g$ it
is the noise-only process $X_t = X_0 + g\,W_t$ of
:numref:`fig_mdl-dyn-sde-paths`.

What is a solution? For an ODE it is one curve. Here it is one curve for
each Brownian path. If $f$ and $g$ are Lipschitz in $x$ and grow at most
linearly, $|f(x, t)| + |g(x, t)| \le C(1 + |x|)$, and $X_0$ is independent of
$W$ with $\mathbb{E}[X_0^2] < \infty$, then the start and the Brownian path
determine the solution path uniquely (:citet:`Oksendal.2003`, Theorem 5.2.1).
The proof is a stochastic version of the Picard iteration of
:numref:`sec_mdl-ode-existence-uniqueness`, with the isometry in place of the
integral bound. All the randomness therefore sits in $W$. Running the
equation many times from the same start with fresh noise gives a cloud of
paths like those of :numref:`fig_mdl-dyn-sde-paths`, and what a generative
model cares about is the law of the cloud at each time.

One distinction will matter when we simulate. Noise is **additive** if its
amplitude depends on time only, $g = g(t)$, as in the template
:eqref:`eq_mdl-sde-template` and in every diffusion-model SDE of this
chapter, the running example among them. It is **multiplicative** if the
amplitude depends on the state, $g = g(X, t)$, as for a stock price whose
fluctuations are proportional to its level. The difference looks cosmetic,
but it decides how accurately a simulated path follows the true one
(:numref:`sec_mdl-sde-strong-weak`).

### Itô's Lemma
:label:`sec_mdl-ito-lemma`

To compute with an SDE we need the chain rule: if $X$ solves an SDE, what
equation does a function $\phi(X_t, t)$ solve? The ordinary chain rule would
give $d(W^2) = 2W\,dW$, and we know that this is wrong, since
$\int_0^t 2W\,dW = W_t^2 - t$ by :eqref:`eq_mdl-sde-wdw`. Something is
missing.

One step of a walk shows where the missing term comes from, and what it is
for other functions. Take $\phi(x) = x^2$ and one step
$\pm\sqrt{\Delta t}$ from $x$, each sign with probability $\tfrac12$. The
average of $\phi$ after the step is

$$
\tfrac12\left[(x + \sqrt{\Delta t})^2 + (x - \sqrt{\Delta t})^2\right]
= x^2 + \Delta t
= \phi(x) + \tfrac12\phi''(x)\,\Delta t ,
$$

since $\phi'' = 2$. The rise comes from the square of the step, which for a
coin flip is $\Delta t$ every time, and which the ordinary chain rule throws
away.

With $x = 0.7$ and $\Delta t = 0.25$ the step lands at $0.2$ or $1.2$, where
$\phi$ is $0.04$ or $1.44$. The average $0.74$ lies above $\phi(0.7) = 0.49$
by $0.25 = \Delta t$ (:numref:`fig_mdl-dyn-ito-jensen`). A convex function
gains from symmetric noise, and because a squared Brownian increment is of
order $dt$, the gain per unit time, $\tfrac12\phi''$, does not vanish in
the limit.

![The Itô correction as Jensen's gap. One step of $\pm\sqrt{\Delta t}$ from $x$ moves a convex $\phi$ to an average that lies above $\phi(x)$ by $\tfrac12\phi''(x)\,\Delta t$; here $\phi=x^2$, $x=0.7$, $\Delta t=0.25$, and the gap is $0.25$.](../img/mdl-dyn-ito-jensen.svg)
:label:`fig_mdl-dyn-ito-jensen`

The general formula comes from the same computation. Let $X$ solve
$dX = f\,dt + g\,dW$ and let $\phi(x, t)$ be smooth. A small change of
$\phi$, expanded to second order because first order is no longer enough, is

$$
d\phi = \phi_t\,dt + \phi_x\,dX + \tfrac12 \phi_{xx}\,(dX)^2 + \cdots.
$$

Substitute $dX = f\,dt + g\,dW$ and multiply out $(dX)^2$ with the table
:eqref:`eq_mdl-sde-ito-table`:

$$
(dX)^2 = f^2 (dt)^2 + 2 f g\, dt\, dW + g^2 (dW)^2 = g^2\,dt.
$$

The drift and cross terms vanish, and the noise term survives at first
order. Collecting terms gives the chain rule of stochastic calculus. What we
have given is a heuristic derivation, so we state the result as a theorem
without proof.

**Theorem (Itô's lemma).** *Let $X$ solve $dX = f\,dt + g\,dW$ under the
conditions of the previous subsection, and let $\phi \in C^{1,2}$, once
continuously differentiable in time and twice in the state. Then
$Y_t = \phi(X_t, t)$ satisfies*

$$
d\phi \;=\; \left(\phi_t + f\,\phi_x + \tfrac12 g^2\,\phi_{xx}\right) dt
\;+\; g\,\phi_x \, dW.
$$
:eqlabel:`eq_mdl-sde-ito-lemma`

This is the working formula of the subject: the ordinary chain rule, plus
half the second derivative times the rate at which variance arrives. The new
term $\tfrac12 g^2 \phi_{xx}\,dt$ is called the **Itô correction**. The proof
has to control the remainder terms of the Taylor expansion in mean square
(:citet:`Oksendal.2003`, chapter 4), and it needs both derivatives in the
state: the lemma does not apply as stated to a function with a kink, such as
$\phi(x) = |x|$.

**Example 1: $W^2$.** Take $X = W$ (so $f = 0$, $g = 1$) and $\phi(x) = x^2$.
The lemma gives $d(W^2) = 2W\,dW + dt$, with the $dt$ that the ordinary chain
rule dropped. In integrated form this is $W_t^2 = \int_0^t 2W_s\,dW_s + t$,
which is :eqref:`eq_mdl-sde-wdw` again. One line of the lemma has replaced
the telescoping sums.

The correction can be watched as it accumulates. On any grid the gap between
$W_t^2$ and the left-endpoint sum for $\int_0^t 2W\,dW$ telescopes,
$W_t^2 - \sum_{t_i < t} 2W_{t_i}\Delta W_i = \sum_{t_i < t} (\Delta W_i)^2$, so it
is the running quadratic variation and should follow the line $t$. We reuse
the path of the left and right sums.

```{.python .input #sdes-ito-correction}
ito = np.concatenate([[0.0], np.cumsum(2 * W_lr[:-1] * dW_lr)])  # int 2W dW
gap = W_lr**2 - ito
print(f'final gap W_1^2 - int 2W dW = {gap[-1]:.4f}   (Ito correction: t = 1)')
print(f'max |gap_t - t| along the path = {np.abs(gap - t_lr).max():.4f}')
d2l.plot(t_lr, [W_lr**2, ito, ito + t_lr], 't', 'value',
         legend=['$W_t^2$', 'naive $\\int_0^t 2W\\,dW$',
                 '$\\int_0^t 2W\\,dW + t$'], figsize=(5, 3))
```

The curve labeled naive is the Itô integral $\int_0^t 2W\,dW$, which the
ordinary chain rule would take for $W_t^2$. It sags below $W_t^2$, and adding
$t$ puts it back on top. Along the whole path the gap usually stays within
about $0.1$ of $t$, since by the proof of :eqref:`eq_mdl-sde-qv` its standard
deviation at time $t$ is $\sqrt{2t\,\Delta t} \le 0.045$ on this grid. At
$t = 1$ the gap is the sum of squared increments that the earlier cell
printed.

**Example 2: geometric Brownian motion.** The SDE
$dX = \mu X\,dt + \nu X\,dW$ with $X_0 > 0$ models a quantity that grows at
mean rate $\mu$ with noise proportional to its size, such as a stock price
:cite:`Black.Scholes.1973`. Its noise is multiplicative. Apply the lemma to
$\phi(x) = \log x$, with $\phi' = 1/x$, $\phi'' = -1/x^2$, and
$g^2 = \nu^2 x^2$. The logarithm is smooth only for $x > 0$, so assume for
the moment that $X$ stays positive:

$$
d\log X = \left(\mu - \tfrac12\nu^2\right) dt + \nu\, dW,
\qquad\textrm{so}\qquad
X_t = X_0 \exp\!\left(\left(\mu - \tfrac12\nu^2\right) t + \nu W_t\right).
$$
:eqlabel:`eq_mdl-sde-gbm`

The formula we obtain is positive, and the lemma applied to it as a function
of $W_t$ and $t$ confirms that it solves the equation, which removes the
assumption. The $-\tfrac12\nu^2$ is the correction with $\phi'' < 0$: a concave function
loses from symmetric noise. Noise therefore lowers the typical growth rate
$\mu - \tfrac12\nu^2$ below the mean growth rate $\mu$, and
Exercise :exref:`ex_mdl-sdes-geometric-brownian-motion-correction` shows how
far the mean and the median of $X_t$ drift apart. We will measure simulation
errors against this exact solution.

In several dimensions the pattern is the same, and we take the result as
given :cite:`Oksendal.2003`. For
$d\mathbf{X} = \mathbf{f}\,dt + \mathbf{G}\,d\mathbf{W}$
with $\mathbf{X} \in \mathbb{R}^d$, a matrix $\mathbf{G}$, and a Brownian
motion $\mathbf{W}$ whose coordinates are independent scalar Brownian
motions, the multiplication table reads $dW_i\, dW_j = \delta_{ij}\,dt$,
and a twice continuously differentiable $\phi$ satisfies

$$
d\phi(\mathbf{X}) \;=\; \nabla\phi \cdot d\mathbf{X}
\;+\; \tfrac12 \sum_{i,j} \left(\mathbf{G} \mathbf{G}^\top\right)_{ij}
\frac{\partial^2\phi}{\partial x_i\, \partial x_j} \; dt .
$$
:eqlabel:`eq_mdl-ito-multivariate`

The correction weights each second derivative by the covariance rate
$\mathbf{G}\mathbf{G}^\top$ of the noise. The diffusion-model SDEs of this
chapter treat every coordinate alike, $\mathbf{G} = g(t)\,\mathbf{I}$, and for
them it reduces to
$\tfrac12 g(t)^2 \sum_i \partial^2\phi/\partial x_i^2\; dt$, a multiple of the
Laplacian of $\phi$. The next section needs the lemma in this form.

## The Ornstein--Uhlenbeck Process
:label:`sec_mdl-ornstein-uhlenbeck`

The running example now gets its name and its solution. The
Ornstein--Uhlenbeck (OU) process :cite:`Uhlenbeck.Ornstein.1930` is

$$
dX = -\theta X \, dt + g\,dW, \qquad \theta > 0,
$$
:eqlabel:`eq_mdl-sde-ou`

the decay ODE $\dot{x} = -\theta x$ of :numref:`sec_mdl-linear-odes-stability`
with noise of constant amplitude $g$ added. Its authors wrote it for the
velocity of a particle that is kicked by molecules and slowed by friction.
The drift
pulls excursions back toward zero at rate $\theta$, and the noise keeps
producing new ones (:numref:`fig_mdl-dyn-ou-mean-reversion`). The recurrence
of the opening is this process with $\theta = 1$ and $g = \sqrt{2}$, advanced
on a grid with step $h = 0.01$ by freezing the drift over each step: one step
multiplies by $1 - \theta h = 0.99$ and adds noise of standard deviation
$g\sqrt{h} = \sqrt{0.02}$.

![Mean reversion. The drift $-\theta x$ (arrows) pulls every path toward $0$ at rate $\theta=1$ while noise of amplitude $g=\sqrt2$ keeps it moving; paths from three starts forget them within a few multiples of $1/\theta$ and then fluctuate inside the band of two standard deviations of the limiting law, $\pm2\sqrt{g^2/2\theta}=\pm2$ (labeled stationary). Right: its density.](../img/mdl-dyn-ou-mean-reversion.svg)
:label:`fig_mdl-dyn-ou-mean-reversion`

### The Transition Kernel

The OU process is one of the few SDEs whose law is known in closed form at
every time, and this is why diffusion models are built on it. The law of
$X_t$ given the start $X_0 = x_0$, read as a function of the start, is called
the *transition kernel* (no relation to the kernels of a convolution). For
the OU process it is Gaussian. Of all the results of this section, this is
the one the rest of the book uses most.

**Proposition (OU transition kernel).** *For the OU process
:eqref:`eq_mdl-sde-ou`,*

$$
X_t \mid X_0 = x_0 \;\sim\; \mathcal{N}\!\left(
x_0\, e^{-\theta t},\;
\frac{g^2}{2\theta}\left(1 - e^{-2\theta t}\right)
\right).
$$
:eqlabel:`eq_mdl-sde-ou-kernel`

**Proof.** The integrating factor $e^{\theta t}$ solved the decay equation
with an input in :numref:`subsec_mdl-ode-driven-linear` (the variation of
constants formula :eqref:`eq_mdl-ode-variation-of-constants`), and it works
when the input is noise. Apply Itô's lemma :eqref:`eq_mdl-sde-ito-lemma` to
$\phi(x, t) = e^{\theta t} x$, for which
$\phi_t = \theta e^{\theta t} x$, $\phi_x = e^{\theta t}$, and
$\phi_{xx} = 0$, so there is no correction:

$$
d\!\left(e^{\theta t} X_t\right)
= \left(\theta e^{\theta t} X_t - \theta e^{\theta t} X_t\right) dt
+ e^{\theta t} g\,dW_t
= g\, e^{\theta t}\,dW_t.
$$

The drift cancels, and integrating from $0$ to $t$ and multiplying by
$e^{-\theta t}$ gives

$$
X_t = X_0\, e^{-\theta t} + g \int_0^t e^{-\theta (t - s)}\, dW_s .
$$
:eqlabel:`eq_mdl-sde-ou-solution`

The remaining integral has a deterministic integrand, so it is a limit of
weighted sums of independent Gaussian increments and is Gaussian with mean
zero (we grant that mean-square limits of Gaussians are Gaussian). The Itô
isometry :eqref:`eq_mdl-sde-ito-isometry` gives its variance:

$$
\textrm{Var}(X_t \mid X_0)
= g^2 \int_0^t e^{-2\theta (t - s)}\, ds
= \frac{g^2}{2\theta}\left(1 - e^{-2\theta t}\right).
\qquad \blacksquare
$$

The solution :eqref:`eq_mdl-sde-ou-solution` reads as a fading memory. The
start decays at rate $\theta$, and a noise kick from time $s$ survives with
weight $e^{-\theta(t-s)}$: recent noise counts, old noise is forgotten. In the
recurrence a kick from $k$ steps ago survives with weight
$0.99^k \approx e^{-0.01k}$. Every later section,
and :numref:`sec_diffusion-gaussian-paths` in the main text, writes the
kernel with two coefficients, the mean factor and the standard deviation,

$$
\alpha_t = e^{-\theta t}, \qquad
\sigma_t^2 = \frac{g^2}{2\theta}\left(1 - e^{-2\theta t}\right),
\qquad\textrm{so that}\qquad
X_t \mid X_0 = x_0 \sim \mathcal{N}\!\left(\alpha_t x_0,\, \sigma_t^2\right).
$$
:eqlabel:`eq_mdl-sde-kernel-coefficients`

The data enter only through the mean $\alpha_t x_0$, and the noise only
through the variance $\sigma_t^2$. For the running example ($\theta = 1$,
$g = \sqrt{2}$, $\sigma_t^2 = 1 - e^{-2t}$) started at $x_0 = 2$, the kernel is
$\mathcal{N}(1.213,\, 0.795^2)$ at $t = 0.5$, $\mathcal{N}(0.736,\, 0.930^2)$
at $t = 1$, and $\mathcal{N}(0.037,\, 1.000^2)$ at $t = 4$. Because the kernel is
Gaussian at every time and for every start, everything about the noised
state given its start is known in closed form, including the gradient of its
log-density, $-(x-\alpha_t x_0)/\sigma_t^2$. Generation will turn out to
depend on this gradient.

The recurrence is only a discretization of the process, so its cloud should
follow the kernel closely but need not follow it exactly. We run it as a
plain loop on $10{,}000$ copies of $x_0 = 2$, save the cloud at five times,
and print its mean and standard deviation next to $2\alpha_t$ and $\sigma_t$,
together with the fraction of the cloud inside $2\alpha_t \pm 2\sigma_t$.

```{.python .input #sdes-ou-cloud}
theta_ou, g_ou, h_ou = 1.0, np.sqrt(2.0), 0.01
X_ou = np.full(10000, 2.0)                    # 10,000 copies of x_0 = 2
snap_steps = {round(t / h_ou): t for t in (0.5, 1.0, 1.5, 2.0, 4.0)}
X_ou_snap = {}
for k in range(1, 401):
    xi = rng.standard_normal(X_ou.size)
    X_ou = (1 - theta_ou * h_ou) * X_ou + g_ou * np.sqrt(h_ou) * xi
    if k in snap_steps:
        X_ou_snap[snap_steps[k]] = X_ou.copy()
print(f'{"t":>4} {"mean":>6} {"2alpha_t":>8} {"std":>6} {"sigma_t":>7}'
      f' {"inside":>7}')
for t, X in X_ou_snap.items():
    alpha_t = np.exp(-theta_ou * t)
    sigma_t = np.sqrt(g_ou**2 / (2 * theta_ou) * (1 - alpha_t**2))
    inside = np.mean(np.abs(X - 2 * alpha_t) < 2 * sigma_t)
    print(f'{t:4.1f} {X.mean():6.3f} {2 * alpha_t:8.3f} {X.std():6.3f} '
          f'{sigma_t:7.3f} {inside:7.3f}')
```

Means and standard deviations agree with the kernel within two or three
standard errors (about $0.01$ for a mean and $0.007$ for a standard deviation
from $10{,}000$ samples). About $95\%$ of the cloud lies inside the band at
every time, as for any Gaussian. The cloud that began as the single point
$2$ widens quickly and then stops widening. At this sample size the
discretization cannot be seen.

### The Stationary Distribution

A forward process must forget its start, and the kernel
:eqref:`eq_mdl-sde-ou-kernel` shows that the OU process does. As
$t \to \infty$ the mean $x_0 e^{-\theta t}$ decays to zero and the variance
approaches $g^2/(2\theta)$, so

$$
X_t \mid X_0 = x_0 \;\longrightarrow\; \mathcal{N}\!\left(0, \, \frac{g^2}{2\theta}\right)
\quad \textrm{in distribution, for every } x_0 .
$$

This is the endpoint, the same for all data, that
:numref:`sec_mdl-why-randomness` asked for. The limit is more than an
endpoint. A process started in it stays in it, and a law with this property
is called *stationary*. To see it, let $X_0 \sim \mathcal{N}(0, g^2/2\theta)$
be independent of $W$. By :eqref:`eq_mdl-sde-ou-solution`, $X_t$ is then the
sum of $X_0 e^{-\theta t}$ and an independent Gaussian of variance
$\sigma_t^2$, hence Gaussian with mean zero and variance

$$
e^{-2\theta t}\,\frac{g^2}{2\theta}
+ \frac{g^2}{2\theta}\left(1 - e^{-2\theta t}\right)
= \frac{g^2}{2\theta}.
$$

The decay of the start removes as much variance as the noise adds. How long
does the forgetting take? Its time scale is $1/\theta$, the *relaxation
time*, and after a few multiples of it $e^{-\theta t}$ is negligible. For the
running example $g^2/2\theta = 1$ and $1/\theta = 1$, so the stationary law
is $\mathcal{N}(0, 1)$, and by $t = 4$ the mean had fallen from $2$ to
$0.037$.

The recurrence has a stationary law of its own, and it is slightly different.
One step maps a variance $v$ to $(1 - \theta h)^2 v + g^2 h$. From $v = 1$ a
step of the running example gives $0.9801 + 0.02 = 1.0001$, slightly more than
$1$, and the fixed point of the map is

$$
v_\infty = \frac{g^2 h}{1 - (1 - \theta h)^2} = \frac{0.02}{0.0199} = 1.005
\qquad (h = 0.01).
$$

The recurrence therefore settles half a percent above the exact variance $1$.
No cloud of $10{,}000$ points can show half a percent, but the pencil can.
This is the first sign of a discretization error, and
:numref:`sec_mdl-sde-strong-weak` shows that it shrinks in proportion to $h$.

### The Variance-Preserving Normalization

The stationary variance $g^2/2\theta$ is a free parameter, and diffusion
models fix it to $1$ by choosing

$$
g^2 = 2\theta
\qquad \Longrightarrow \qquad
X_t \mid X_0 = x_0 \;\longrightarrow\; \mathcal{N}(0, 1).
$$

The running example makes this choice. With it
$\sigma_t^2 = 1 - \alpha_t^2$, so one coefficient describes the kernel.

So far the start was a single number $x_0$. Let it now be a draw $X_0$ from
the data. In distribution, the solution :eqref:`eq_mdl-sde-ou-solution` then
reads $X_t = \alpha_t X_0 + \sqrt{1 - \alpha_t^2}\,\epsilon$ with a
standard Gaussian $\epsilon$ independent of $X_0$. The law of this $X_t$,
which is the kernel averaged over the data, is called the *marginal* at
time $t$. Most papers write it with another letter. They follow the
denoising diffusion probabilistic model (DDPM) of :citet:`ho2020denoising`,
the discrete-time diffusion model of :numref:`sec_diffusion-gaussian-paths`,
which writes $\bar{\alpha}_t$ for our $\alpha_t^2 = e^{-2\theta t}$. The
main text does the same, and so do we:

$$
X_t \;=\; \sqrt{\bar{\alpha}_t}\; X_0 \;+\; \sqrt{1 - \bar{\alpha}_t}\;
\epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1)
\textrm{ independent of } X_0 .
$$
:eqlabel:`eq_mdl-sde-vp-marginal`

The marginal is a weighted mix of data and fresh noise, with weights whose
squares add to one. Suppose the data have zero mean and unit variance. Taking
variances of the two independent terms gives

$$
\textrm{Var}(X_t)
= \bar{\alpha}_t \textrm{Var}(X_0) + (1 - \bar{\alpha}_t)
= \bar{\alpha}_t + (1 - \bar{\alpha}_t) = 1
\qquad \textrm{for all } t.
$$

This is what **variance-preserving** (VP) means. For unit-variance data the
marginal variance equals $1$ at every time, not only in the limit. The process
moves variance from the data to the noise, a fraction $\bar{\alpha}_t$ still
carried by the data and $1 - \bar{\alpha}_t$ already replaced by noise,
without changing the total. The name is accurate for unit-variance data only.
For data of variance $s^2$ the marginal variance drifts from $s^2$ to $1$
(Exercise :exref:`ex_mdl-sdes-variance-preservation-exactly`), so the scale of
the data matters. Images scaled to $[-1, 1]$, as in
:numref:`sec_diffusion-denoising`, have variance below one, while latent
diffusion models rescale their latents to about unit variance
:cite:`Rombach.Blattmann.Lorenz.ea.2022`.

The variance is preserved; the shape of the distribution is not. Take data
that are $\pm 1$ with probability $\tfrac12$ each, the outcome of a coin
flip. Their fourth moment is $1$, against $3$ for a standard Gaussian, and
along the process it climbs as
$\mathbb{E}[X_t^4] = 3 - 2\bar{\alpha}_t^2$, which is $2.264$ at $t = 0.25$
and $2.963$ at $t = 1$ for the running example (same exercise). The coin
turns into a bell curve, and its variance never moves.

Diffusion models usually let the rate vary with time. The **VP-SDE** of
score-based diffusion :cite:`song2021score`,

$$
dX = -\tfrac12 \beta(t)\, X \, dt + \sqrt{\beta(t)}\; dW,
$$
:eqlabel:`eq_mdl-sde-vp-sde`

is an OU process with time-dependent rate $\theta(t) = \beta(t)/2$ and noise
amplitude $g(t)^2 = \beta(t)$; the running example is the constant schedule
$\beta = 2$. It keeps $g^2 = 2\theta$ at every instant, so
the variance stays at $1$ along the whole schedule, and its marginal is
:eqref:`eq_mdl-sde-vp-marginal` with $\bar{\alpha}_t = e^{-\int_0^t \beta(s)\,ds}$
(Exercise :exref:`ex_mdl-sdes-exact-vp-discretization` proves that its
transitions are exact Gaussians). This is the variance-preserving path of
:numref:`sec_diffusion-gaussian-paths`, the forward process that a diffusion
model learns to reverse.

## Simulation

### The Euler--Maruyama Scheme
:label:`sec_mdl-euler-maruyama`

Most SDEs have no closed-form solution, so we need to simulate them. Our
recurrence already shows how. To simulate :eqref:`eq_mdl-sde-general` on a
grid $t_n = n\,\Delta t$ we freeze the coefficients over each step, as forward Euler does for ODEs
(:numref:`sec_mdl-euler-runge-kutta`), and add a Gaussian kick whose size the
increment form :eqref:`eq_mdl-sde-increment` dictates. The
**Euler--Maruyama** update :cite:`Maruyama.1955` is

$$
\hat{X}_{n+1} = \hat{X}_n + f(\hat{X}_n, t_n)\,\Delta t
+ g(\hat{X}_n, t_n)\,\sqrt{\Delta t}\;\xi_n,
\qquad \xi_n \sim \mathcal{N}(0, 1) \textrm{ i.i.d.},
$$
:eqlabel:`eq_mdl-sde-em`

where $\hat{X}_n$ approximates $X_{t_n}$ and $\hat X_0 = X_0$. The kick
scales as $\sqrt{\Delta t}$, since any other exponent makes the noise vanish
or explode in the limit (:numref:`sec_mdl-wiener-process`), and with $g = 0$
the scheme is forward Euler.

For the OU process the step reads

$$
\hat{X}_{n+1} = (1 - \theta\,\Delta t)\,\hat{X}_n + g\sqrt{\Delta t}\;\xi_n
= 0.99\,\hat{X}_n + 0.1414\,\xi_n ,
$$
:eqlabel:`eq_mdl-sde-em-ou`

where the second form is the running example at $\Delta t = h = 0.01$, the
recurrence we have been running all along.

For this process we could have done better. With $g^2 = 2\theta$ the kernel
:eqref:`eq_mdl-sde-ou-kernel` over one interval is
$\mathcal{N}(e^{-\theta\Delta t}x, 1 - e^{-2\theta\Delta t})$, and sampling
from it is the forward step of DDPM
(:numref:`sec_diffusion-gaussian-paths`),
$x_k = \sqrt{1 - \beta_k}\, x_{k-1} + \sqrt{\beta_k}\,\epsilon_k$ with
$\beta_k = 1 - e^{-2\theta\Delta t}$. DDPM's step is therefore exact at every
step size, and Euler--Maruyama is its first-order approximation. At
$\theta = 1$ and $\Delta t = 0.01$ the exact step multiplies by
$e^{-0.01} = 0.99005$ and adds noise of variance
$\beta_k = 1 - e^{-0.02} = 0.0198$, against $0.99$ and $0.02$. The two betas
should not be confused. The $\beta_k$ of the step is the fraction of the
variance that one step replaces by noise. It is about $\beta(t)\,\Delta t$
for the rate $\beta(t)$ of :eqref:`eq_mdl-sde-vp-sde`, here $0.0198$ against
$2 \cdot 0.01$. Exact steps need a linear equation
(Exercise :exref:`ex_mdl-sdes-exact-vp-discretization`). A general drift has
no transition law in closed form, and the approximate scheme is all we have.

The helper below implements :eqref:`eq_mdl-sde-em` for a whole batch of
paths. It takes the increments $\Delta W$ as an argument, so that two
simulations can be driven by the same noise. We need that at once: with the
noise switched on, "error" itself needs a definition, and there are two.

```{.python .input #sdes-euler-maruyama}
def euler_maruyama(f, g, x0, t, dW):
    """Simulate dX = f dt + g dW on the grid t for a batch of paths."""
    X = np.empty((len(x0), len(t)))
    X[:, 0] = x0
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        X[:, k + 1] = (X[:, k] + f(X[:, k], t[k]) * h
                       + g(X[:, k], t[k]) * dW[:, k])
    return X
```

### Strong and Weak Convergence
:label:`sec_mdl-sde-strong-weak`

A simulated path is random, and so is the true one. The two definitions of
error differ in what they compare: paths or laws, the two kinds of statement
that the band experiment told apart.

Let $N = T/\Delta t$ be the number of steps, so that $\hat{X}_N$ approximates
$X_T$. Drive the scheme and the exact solution with the same Brownian
increments. The **strong error** $\mathbb{E}\,|\hat{X}_N - X_T|$ measures
pathwise tracking: does the simulated trajectory follow the true trajectory
driven by that noise? The **weak error**
$|\mathbb{E}\,\varphi(\hat{X}_N) - \mathbb{E}\,\varphi(X_T)|$, for smooth test
functions $\varphi$, measures only whether the law at time $T$ comes out
right. A generative model needs weak accuracy, since it only uses the samples
at the final time. Strong accuracy matters when paths are coupled across
resolutions, as in multilevel Monte Carlo, which makes most of its estimate
on cheap coarse paths and corrects it with a few expensive fine ones
(Exercise :exref:`ex_mdl-sdes-multilevel-monte-carlo`).
:numref:`fig_mdl-dyn-strong-weak` shows the two notions side by side.

![Strong versus weak convergence of Euler--Maruyama on the running example from $X_0=2$ on $[0,1]$. Left: one coarse path ($16$ steps, orange) against a fine reference ($4096$ steps, blue) driven by the same Brownian increments; the strip below magnifies their gap at the coarse grid times, which is what the strong error measures. Right: the terminal histograms of $20{,}000$ paths at the two step sizes nearly match each other and the exact law (black), although at $16$ steps the variance is about $4\%$ too large.](../img/mdl-dyn-strong-weak.svg)
:label:`fig_mdl-dyn-strong-weak`

A scheme has strong order $p$ if its strong error is at most
$C\,\Delta t^{p}$, and weak order $p$ likewise. The orders of
Euler--Maruyama are classical.

**Theorem (orders of Euler--Maruyama; stated without proof).** *Let $f$, $g$,
and $\varphi$ be smooth, with the growth conditions and the moments of $X_0$
that :citet:`Kloeden.Platen.1992` make precise.*

1. *Euler--Maruyama has strong order $\tfrac12$:
   $\mathbb{E}\,|\hat{X}_N - X_T| \le C\,\Delta t^{1/2}$.*
2. *If the noise is additive, $g = g(t)$, the strong order is $1$:
   $\mathbb{E}\,|\hat{X}_N - X_T| \le C\,\Delta t$.*
3. *The weak order is $1$, for additive and multiplicative noise alike:
   $|\mathbb{E}\,\varphi(\hat{X}_N) - \mathbb{E}\,\varphi(X_T)| \le C\,\Delta t$.*

A count of single-step errors explains the exponents. It is not a proof: it
adds the errors of the steps as if each stayed where it was made, and the
proofs must also control how the later dynamics carry an early error forward.

Euler--Maruyama freezes $g$ over the step, but $g$ moves with the path. At a
time $s$ inside the step the path has moved by about $g\,(W_s - W_{t_n})$, so
the noise coefficient there is about
$g + g\,\partial_x g\,(W_s - W_{t_n})$. The largest term that the scheme
leaves out is the integral of the second part against $dW$, which is
:eqref:`eq_mdl-sde-wdw` for the Brownian motion $W_s - W_{t_n}$ that starts
afresh at $t_n$:

$$
g\,\partial_x g \int_{t_n}^{t_{n+1}} \left(W_s - W_{t_n}\right) dW_s
= \tfrac12\, g\,\partial_x g\left((\Delta W)^2 - \Delta t\right).
$$

Milstein's scheme puts this term back :cite:`Milstein.1975`. Its random
factor is the rule for squares seen a step at a time: $(\Delta W)^2 = \Delta t$
is true in sums, and off by something of mean zero and size $\Delta t$ in
every step. The $N = T/\Delta t$ steps contribute uncorrelated terms of this
kind, which add like a random walk to
$\sqrt{N}\,\Delta t = \sqrt{T\,\Delta t}$. That is order $\tfrac12$.

For additive noise $\partial_x g = 0$ removes this term. The next omitted
stochastic terms have size $\Delta t^{3/2}$ per step and add to
$\sqrt{N}\,\Delta t^{3/2} = O(\Delta t)$, as do the deterministic remainders,
$N\,\Delta t^2 = O(\Delta t)$, and the order is $1$. For the weak error only
means matter. Every omitted stochastic term has mean zero, the bias of one step is
of order $\Delta t^2$, and $N$ steps give $O(\Delta t)$, whatever the noise.

We measure both strong orders on the same noise. Geometric Brownian motion
($\mu = \nu = 1$, $X_0 = 1.5$) is compared with its exact solution
:eqref:`eq_mdl-sde-gbm`, and the running example from $X_0 = 2$ with a
reference on a $4096$-step grid, since the noise integral in its exact
solution :eqref:`eq_mdl-sde-ou-solution` could itself be evaluated only on a
grid. Coarse increments are sums of consecutive fine ones, so every
resolution sees the same Brownian path.

```{.python .input #sdes-em-strong-order}
T, n_ref, n_em = 1.0, 2**12, 2000
mu_g, nu_g = 1.0, 1.0                        # GBM: multiplicative noise
ou_f, ou_g = lambda x, s: -theta_ou * x, lambda x, s: g_ou
gbm_f, gbm_g = lambda x, s: mu_g * x, lambda x, s: nu_g * x
x0_ou, x0_gbm = np.full(n_em, 2.0), np.full(n_em, 1.5)
dW_f = np.sqrt(T / n_ref) * rng.standard_normal((n_em, n_ref))
t_f = np.linspace(0, T, n_ref + 1)
ou_ref = euler_maruyama(ou_f, ou_g, x0_ou, t_f, dW_f)[:, -1]
gbm_exact = 1.5 * np.exp((mu_g - nu_g**2 / 2) * T + nu_g * dW_f.sum(axis=1))
dts, errs_ou, errs_gbm = [], [], []
for n in [2**k for k in range(3, 9)]:        # 8 to 256 steps
    dW = dW_f.reshape(n_em, n, n_ref // n).sum(axis=2)
    t = np.linspace(0, T, n + 1)
    ou = euler_maruyama(ou_f, ou_g, x0_ou, t, dW)[:, -1]
    gbm = euler_maruyama(gbm_f, gbm_g, x0_gbm, t, dW)[:, -1]
    dts.append(T / n)
    errs_ou.append(np.abs(ou - ou_ref).mean())
    errs_gbm.append(np.abs(gbm - gbm_exact).mean())
print('strong-order slopes:  OU (additive) '
      f'{np.polyfit(np.log(dts), np.log(errs_ou), 1)[0]:.2f},'
      '  GBM (multiplicative) '
      f'{np.polyfit(np.log(dts), np.log(errs_gbm), 1)[0]:.2f}')
d2l.plot(dts, [errs_ou, errs_gbm, np.array(dts), 0.5 * np.sqrt(dts)],
         'step size', 'strong error', xscale='log', yscale='log',
         legend=['OU (additive)', 'GBM (multiplicative)',
                 'slope 1', 'slope 1/2'], figsize=(5, 3))
```

The measured slopes are close to $1$ for the additive-noise OU process and
close to $\tfrac12$ for the multiplicative-noise geometric Brownian motion,
the two strong orders of the theorem. The two curves have different heights
because the processes have different scales. What can be compared is how
they fall: each halving of the step halves the OU error and divides the
error of geometric Brownian motion by about $\sqrt{2}$.

For the weak order no sampling is needed. Euler--Maruyama applied to the OU
process is a linear Gaussian recursion: if $\hat{X}_n$ is Gaussian, so is
$\hat{X}_{n+1}$. So the simulated marginal is Gaussian at every step, with mean
and variance

$$
m_{n+1} = (1 - \theta\,\Delta t)\,m_n,
\qquad
v_{n+1} = (1 - \theta\,\Delta t)^2\, v_n + g^2\,\Delta t .
$$

The mean is multiplied by $1 - \theta\Delta t$ per step where the exact
solution multiplies by $e^{-\theta\Delta t}$ ($0.900$ against $0.905$ at
$\Delta t = 0.1$). The variance recursion sums a geometric series,
$v_n = g^2\Delta t\,(1 - a^{2n})/(1 - a^2)$ with $a = 1 - \theta\Delta t$
and $v_0 = 0$, whose limit is the fixed point $v_\infty$ found earlier.
Evaluating these closed forms at $T = 1$ from $X_0 = 2$ for six step counts
gives the errors of the simulated law with no sampling error to blur their
slopes.

```{.python .input #sdes-em-weak-order}
mean_T, var_T = 2 * np.exp(-1), 1 - np.exp(-2)  # exact OU kernel at T = 1
print(f'{"n":>4} {"EM mean":>8} {"EM var":>7}'
      f'   exact {mean_T:.4f}, {var_T:.4f}')
dts, errs = [], []
for n in [10, 20, 40, 80, 160, 320]:
    h = 1 / n
    a_h = 1 - theta_ou * h                   # one-step mean multiplier
    m = 2 * a_h**n                           # EM mean after n steps
    v = g_ou**2 * h * (1 - a_h**(2 * n)) / (1 - a_h**2)   # geometric series
    print(f'{n:4d} {m:8.4f} {v:7.4f}')
    dts.append(h)
    errs.append([abs(m - mean_T), abs(v - var_T)])
s_m, s_v = np.polyfit(np.log(dts), np.log(np.array(errs)), 1)[0]
print(f'weak-order slopes:  mean {s_m:.2f},  variance {s_v:.2f}')
```

At $10$ steps the simulated law has mean $0.697$ and variance $0.925$,
against the exact $0.736$ and $0.865$ (the kernel
$\mathcal{N}(0.736,\, 0.930^2)$ from before), and at $320$ steps it has
$0.735$ and $0.866$. Both fitted slopes are $1.01$: halving the step halves
the error of the simulated law, which is weak order $1$.

The same computation for geometric Brownian motion also gives weak order $1$,
so multiplicative noise costs half an order in the paths and nothing in the
law. A scheme that looks more accurate can even converge to the wrong
equation. Heun's method, which averages the noise coefficient over the step,
converges to the Stratonovich solution instead of the Itô one
(Exercise :exref:`ex_mdl-sdes-ito-versus-stratonovich`).

Simulation produces samples of the forward process, and the kernel gives its
law from a single start. Generation asks for more. Data are not a single
point: the marginal $p_t$ of the noised data is the kernel averaged over all
starts, and for real data, which we know only through samples, it has no
closed form. Yet running the process backward will require the gradient of
$\log p_t$, the quantity that we could write down for one start, and so far
we do not even have an equation that $p_t$ obeys.

## Summary

A forward process must forget its start, which no deterministic flow can do,
so it is driven by noise. Noise that survives the limit of small steps has
size $\sqrt{\Delta t}$, and the squares of its increments add up to the
elapsed time, $(dW)^2 = dt$. An integral against $dW$ therefore depends on
where its integrand is evaluated, and Itô's left endpoint gives it mean zero
and the isometry $\mathbb{E}[(\int H\,dW)^2] = \int \mathbb{E}[H^2]\,ds$.
For the same reason the chain rule gains the correction
$\tfrac12 g^2 \phi_{xx}\,dt$. With these tools the Ornstein--Uhlenbeck
process $dX = -\theta X\,dt + g\,dW$ is solved: given its start $x_0$ it is
Gaussian at every time, with mean $x_0 e^{-\theta t}$ and variance
$\tfrac{g^2}{2\theta}(1 - e^{-2\theta t})$, and with $g^2 = 2\theta$ it is
the variance-preserving forward process of diffusion models. Equations
without a closed form are simulated by Euler--Maruyama, Euler's step plus a
$\sqrt{\Delta t}$ kick, which follows paths at order $1$ for additive noise
and $\tfrac12$ for multiplicative noise and reproduces laws at order $1$ for
both.

## Exercises

1. **Infinite total variation.** The total variation of a path $x$ along a
   partition $0 = t_0 < t_1 < \cdots < t_n = t$ is
   $V_n = \sum_i |x(t_{i+1}) - x(t_i)|$. Pathwise integrals
   $\int H\,dx$, defined as limits of sums $\sum_i H(s_i)\,\Delta x_i$ with
   arbitrary evaluation points $s_i \in [t_i, t_{i+1}]$, exist for every
   continuous $H$ when $V_n$ stays bounded as the mesh shrinks.
    1. Let $x = W$ and use the uniform partition with $\Delta t_i = t/n$.
       Compute $\mathbb{E}[V_n]$ and $\textrm{Var}(V_n)$, using
       $\mathbb{E}|\xi| = \sqrt{2/\pi}$ for $\xi \sim \mathcal{N}(0, 1)$, and
       conclude that $V_n \to \infty$ in probability as $n \to \infty$.
    1. Let $x$ be continuous with $V_n \le V$ for every partition. Show that
       $\max_i |\Delta x_i| \to 0$ as the mesh shrinks, and deduce
       $\sum_i (\Delta x_i)^2 \le \max_i |\Delta x_i| \cdot V \to 0$: such a
       path has quadratic variation zero. Why does :eqref:`eq_mdl-sde-qv`
       then give a second proof that Brownian paths have unbounded total
       variation?
    1. Let $x$ be continuous with bounded total variation. Show that the
       left-endpoint sums $\sum_i x(t_i)\,\Delta x_i$ and the right-endpoint
       sums $\sum_i x(t_{i+1})\,\Delta x_i$ both converge to
       $\tfrac12\left(x(t)^2 - x(0)^2\right)$. Compare with the two limits
       for $x = W$ in :numref:`sec_mdl-ito-integral`. Which step of your
       argument fails for a Brownian path?
   :exlabel:`ex_mdl-sdes-infinite-total-variation`

1. [code] **Brownian scaling and exit times.**
    1. For $c > 0$ let $\tilde{W}_t = W_{ct}/\sqrt{c}$. Check properties 1–4
       of :numref:`sec_mdl-wiener-process` and conclude that $\tilde{W}$ is a
       Brownian motion. Which property of the $\pm\sqrt{\Delta t}$ walk does
       this make exact?
    1. Let $\tau_a$ be the first time $|W_t|$ reaches a level $a > 0$. Use (a)
       to show that $\tau_a$ has the same distribution as $a^2 \tau_1$, so
       that $\mathbb{E}[\tau_a] = a^2\, \mathbb{E}[\tau_1]$.
    1. Granting $\mathbb{E}[\tau_1] = 1$ (it follows from
       $\mathbb{E}[W_{\tau}^2] = \mathbb{E}[\tau]$, a consequence of
       :eqref:`eq_mdl-sde-wdw` and the zero mean of the Itô integral up to a
       stopping time), simulate $4{,}000$ walks with Gaussian steps of
       variance $\Delta t = 10^{-3}$ and estimate $\mathbb{E}[\tau_a]$ for
       $a = 0.5, 1, 2$, with Monte Carlo standard errors. The estimates tend
       to run slightly above $a^2$, most clearly for small $a$. Explain the
       sign of the bias from the way a discrete walk crosses a level, and say
       how it scales with $a$ and $\Delta t$.
    1. In the experiment of the text a little over half of the paths stay
       inside $\pm 2\sqrt{t}$ at all $500$ grid times. Use (a) to show that
       this fraction depends only on the number $n$ of grid times and not on
       the length of the interval. Measure it for $n = 50$, $500$, and
       $5000$. By what factor does it fall each time $n$ grows tenfold, and
       what does that say about the limit $n \to \infty$?
   :exlabel:`ex_mdl-sdes-brownian-scaling-exit-times`

1. **Geometric Brownian motion and the $-\nu^2/2$ correction.** Consider
   $dX = \mu X\,dt + \nu X\,dW$ with $X_0 > 0$ and its solution, given
   by :eqref:`eq_mdl-sde-gbm`.
    1. Using $\mathbb{E}\big[e^{\lambda Z}\big] = e^{\lambda^2 v/2}$ for
       $Z \sim \mathcal{N}(0, v)$, show that
       $\mathbb{E}[X_t] = X_0 e^{\mu t}$ and
       $\mathbb{E}[X_t^2] = X_0^2 e^{(2\mu + \nu^2) t}$. Find the median
       of $X_t$.
    1. Take $\mu = 0.1$ and $\nu = 0.6$. What happens to the mean, to the
       median, and to a typical path as $t \to \infty$? How can these
       statements hold at the same time?
    1. For which $\nu$, at a fixed $\mu > 0$, does the median stop growing?
       Check the condition for the parameters of (b) on $100$ paths
       simulated with `euler_maruyama` to $t = 20$ at $\Delta t = 10^{-2}$:
       what fraction of the paths ends below $X_0$, and what fraction does
       the exact solution :eqref:`eq_mdl-sde-gbm` predict?

   *Adapted from Simo Särkkä and Arno Solin,
   [Applied Stochastic Differential Equations](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf),
   Exercise 4.3.*
   :exlabel:`ex_mdl-sdes-geometric-brownian-motion-correction`

1. [code] **Strong order and Milstein's correction.** As
   :numref:`sec_mdl-sde-strong-weak` explains, Euler--Maruyama omits the
   Milstein term
   $\tfrac12 g\,\partial_x g\left((\Delta W)^2 - \Delta t\right)$, and its
   strong order is $\tfrac12$ unless that term vanishes.
    1. By :eqref:`eq_mdl-sde-gbm`, $Y = \log X$ satisfies an SDE with
       additive noise. Apply `euler_maruyama` to that SDE with the
       increments `dW_f` and the step sizes of the strong-order
       experiment, exponentiate the endpoints, and measure the strong error against
       `gbm_exact`. Explain the result.
    1. Use Itô's lemma to show that for $dX = f(X)\,dt + g(X)\,dW$ with
       $g > 0$, the variable $Y = \psi(X)$ with $\psi' = 1/g$ has unit
       additive noise, and find its drift.
    1. Milstein's scheme :cite:`Milstein.1975` adds that term to the
       Euler--Maruyama step,

       $$\hat X_{n+1} = \hat X_n + f\,\Delta t + g\,\Delta W_n + \tfrac12\, g\,\partial_x g\left((\Delta W_n)^2 - \Delta t\right),$$

       with $f$, $g$, and $\partial_x g$ evaluated at $(\hat X_n, t_n)$.
       Implement it for the geometric Brownian motion of the strong-order
       experiment, and measure its strong-order slope on the same increments
       and step sizes. By what factor does it reduce the
       strong error at $\Delta t = 2^{-8}$?
   :exlabel:`ex_mdl-sdes-milstein-correction`

1. **Variance preservation, exactly.** Let $X_t$ follow
   :eqref:`eq_mdl-sde-vp-marginal` with zero-mean data $X_0$.
    1. Assume unit-variance data and show that
       $\mathbb{E}[X_t^4] = 3 + \bar{\alpha}_t^2\left(\mathbb{E}[X_0^4] - 3\right)$.
       Specialize to Rademacher data $X_0 = \pm 1$ to obtain the prediction
       $3 - 2\bar{\alpha}_t^2$ of the text.
    1. The excess kurtosis $\mathbb{E}[X^4]/\textrm{Var}(X)^2 - 3$ is
       zero for every Gaussian. At what rate in $t$ does it decay for
       $X_t$? Compare this rate with the rate at which the conditional mean
       $\mathbb{E}[X_t \mid X_0]$ forgets the data.
    1. Suppose instead that the data have variance $s^2$. Find
       $\textrm{Var}(X_t)$. For $s = 2$ and $\theta = 1$, how long must
       the process run before $\textrm{Var}(X_T)$ is within $1\%$ of
       the variance of the reference $\mathcal{N}(0, 1)$?
   :exlabel:`ex_mdl-sdes-variance-preservation-exactly`

1. [code] **Itô versus Stratonovich.** ● The left- and right-endpoint sums
   for $\int_0^t W\,dW$ add up to $W_t^2$ on every partition
   (:numref:`sec_mdl-ito-integral`), so their average equals
   $\tfrac12 W_t^2$. Sums of this kind define the Stratonovich integral
   :cite:`Stratonovich.1966`, which obeys the ordinary chain rule.
    1. Heun's method of :numref:`sec_mdl-euler-runge-kutta` carries over to
       SDEs if the noise term is treated like a second slope. The predictor
       is the Euler--Maruyama step
       $\tilde{X} = \hat X_n + f(\hat X_n)\,\Delta t + g(\hat X_n)\,\Delta W_n$,
       and the corrector averages both coefficients:

       $$\hat X_{n+1} = \hat X_n + \tfrac12\left[f(\hat X_n) + f(\tilde{X})\right]\Delta t + \tfrac12\left[g(\hat X_n) + g(\tilde{X})\right]\Delta W_n.$$

       Implement it for the geometric Brownian motion of the strong-order
       experiment, on its increments `dW_f` and step sizes. Compare its endpoints with the exact Itô solution
       :eqref:`eq_mdl-sde-gbm` and with $X_0 \exp(\mu T + \nu W_T)$. To which
       of the two does the scheme converge?
    1. Expand the corrector's noise term to first order in
       $\tilde{X} - \hat X_n$, and use $(\Delta W_n)^2 \approx \Delta t$ to
       show that the scheme behaves like Euler--Maruyama for an Itô SDE with
       a modified drift. Find that drift and check that it explains the limit
       in (a).
    1. Why do the two conventions give the same solution for every SDE of
       the form :eqref:`eq_mdl-sde-template`?
   :exlabel:`ex_mdl-sdes-ito-versus-stratonovich`

1. **Exact transitions of linear SDEs.** ● Consider the linear SDE
   $dX = -\theta(t) X\,dt + g(t)\,dW$ with continuous $\theta$ and
   $g$, and let $\Theta(t) = \int_0^t \theta(u)\,du$.
    1. Apply Itô's lemma to $\phi(x, t) = e^{\Theta(t)} x$ and show that,
       given $X_s$, the state $X_t$ at a later time $t$ is Gaussian with mean
       $e^{-(\Theta(t) - \Theta(s))} X_s$ and variance
       $\int_s^t e^{-2(\Theta(t) - \Theta(u))}\, g(u)^2\, du$.
    1. Specialize (a) to the VP-SDE :eqref:`eq_mdl-sde-vp-sde` and show that
       a transition of any length maps a unit-variance state to a
       unit-variance state. Conversely, assume for the linear SDE of (a) that
       $\textrm{Var}(X_t) = 1$ for every $t \ge 0$ whenever
       $\textrm{Var}(X_0) = 1$, and show that
       $g(t)^2 = 2\,\theta(t)$ at every $t$.
    1. Consider the semilinear SDE
       $dX = \left(-\theta X + q(X)\right)dt + g\,dW$ with constants
       $\theta, g > 0$ and a smooth function $q$. The exponential
       Euler--Maruyama step holds $q$ at the value $q(\hat X_n)$ over a step
       of length $\Delta t$ and solves the remaining linear SDE exactly, by
       the method of (a). Write the step. Show that it agrees with the
       Euler--Maruyama step :eqref:`eq_mdl-sde-em` to first order in
       $\Delta t$ and that it is exact when $q$ is constant. For
       $q \equiv 0$, Euler--Maruyama damps the mean only for
       $\Delta t < 2/\theta$ (:numref:`sec_mdl-stiffness-implicit`). What
       limits the step size of the exponential step?
    1. Exact steps have zero weak error at every step size. Which property
       of the drift and the diffusion of the VP-SDE makes them available?
       The generative sampler of a diffusion model simulates,
       backward in time, an SDE whose drift is
       $-\tfrac12\beta(t)\,x - \beta(t)\,\nabla \log p_t(x)$: the VP drift
       plus a multiple of $\nabla \log p_t$, the *score* of the forward
       marginal, which is a nonlinear function of $x$ for non-Gaussian data.
       Why can its steps not be made exact? Exponential integrators for
       diffusion samplers :cite:`Zhang.Chen.2023` apply the idea of (c) to
       the deterministic counterpart of this sampler, an ODE whose drift
       $f(x, t) - \tfrac12 g(t)^2 \partial_x \log p_t(x)$ has the same two
       parts with half the score term (derived in
       :numref:`sec_mdl-fokker-planck-probability-flow` by rewriting the 1D
       Fokker--Planck PDE
       $\partial_t p_t(x) = -\partial_x\bigl(f(x,t)p_t(x)\bigr) + \tfrac{1}{2}g(t)^2 \partial_x^2 p_t(x)$
       as a continuity equation). Which part of
       such a drift does the step treat exactly, and which part does it
       approximate?
   :exlabel:`ex_mdl-sdes-exact-vp-discretization`

1. [extended] **Multilevel Monte Carlo.** ●● Estimating
   $\mathbb{E}[\varphi(X_T)]$ by averaging over Euler--Maruyama paths has two
   sources of error, the weak bias of the scheme and the sampling noise.
   Multilevel Monte Carlo :cite:`Giles.2008` reduces the cost by coupling
   paths across step sizes, which is one of the uses of strong accuracy named
   in :numref:`sec_mdl-sde-strong-weak`. Throughout, the mean-square strong
   error $\mathbb{E}\big[(\hat X_N - X_T)^2\big]^{1/2}$ has the same
   orders as the strong error of that section.
    1. Plain Monte Carlo averages $\varphi(\hat X_N)$ over $M$
       independent paths at a cost of $MT/\Delta t$ steps. Assume a weak
       bias $c_1\,\Delta t$ and a variance $v$ of $\varphi(\hat X_N)$.
       Show that a mean squared error below $\varepsilon^2$ costs order
       $\varepsilon^{-3}$ steps at the best choice of $M$ and $\Delta t$.
    1. Let $P_l$ be $\varphi$ of the Euler--Maruyama endpoint with step
       $h_l = T\,2^{-l}$, and write
       $\mathbb{E}[P_L] = \mathbb{E}[P_0] + \sum_{l=1}^{L}\mathbb{E}[P_l - P_{l-1}]$.
       Each term is estimated from $M_l$ independent pairs in which the coarse
       path uses the sums of consecutive pairs of fine increments, as in
       the strong-order experiment. For Lipschitz $\varphi$, bound
       $V_l = \textrm{Var}(P_l - P_{l-1})$ by the mean-square strong
       error. Take $\varphi(x) = x$, and predict how $V_l$ scales with $h_l$
       for the OU process and the geometric Brownian motion of that
       experiment, with its parameters. Measure $V_l$
       for $l = 1, \ldots, 8$ with `euler_maruyama`, and fit the slopes on
       the finer levels.
    1. Minimize the total cost $\sum_l M_l / h_l$ subject to
       $\sum_l V_l / M_l \le \varepsilon^2 / 2$ with a Lagrange multiplier.
       Choose the finest level $L$ so that the bias stays below
       $\varepsilon / \sqrt{2}$. Assuming $V_l \propto h_l^{r}$, show that
       the cost is of order $\varepsilon^{-2}$ for $r = 2$ and of order
       $\varepsilon^{-2} (\log \varepsilon)^2$ for $r = 1$.
    1. Implement both estimators for $\mathbb{E}[X_T] = X_0 e^{\mu T}$ of the
       geometric Brownian motion of the strong-order experiment, with $M_l$ from
       (c) and your measured $V_l$. For
       $\varepsilon \in \{0.2, 0.1, 0.05, 0.02\}$, record the total number of
       steps each estimator uses and check its error over a few repetitions.
       Plot $\varepsilon^2$ times the cost against $\varepsilon$. Repeat with
       the Milstein scheme of Exercise :exref:`ex_mdl-sdes-milstein-correction`c
       in place of Euler--Maruyama. `euler_maruyama` stores every path in
       full, so simulate large $M_l$ in batches.
    1. Repeat the measurement of (b) for the discontinuous
       $\varphi(x) = \mathbf{1}(x > X_0)$. Why does $V_l$ decay more slowly,
       and what does the slower decay do to the cost?
   :exlabel:`ex_mdl-sdes-multilevel-monte-carlo`

[Discussions](https://d2l.discourse.group/t/36838)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §38.4]{.kicker}

Stochastic Differential Equations, Brownian Motion, and Itô's Lemma<br>**Wiener process · quadratic variation (dW)² = dt · Itô integral & isometry · Itô's lemma · Ornstein–Uhlenbeck**
:::
:::

::: {.slide title="A Recurrence That Turns Any Start Into N(0, 1)"}
[Motivation]{.kicker}

Start 10,000 copies at $x_0 = 2$ and iterate $x_{k+1} = 0.99\,x_k + \sqrt{0.02}\,\xi_k$ with $\xi_k \sim \mathcal{N}(0, 1)$ ($h = 0.01$, $t = kh$). By 400 steps, every trace of $x_0 = 2$ has vanished into $\mathcal{N}(0, 1)$:

@!sdes-ou-cloud

::: {.d2l-note .warn}
**Why $\sqrt{h}$ Instead of $h$?** Over step $h = 0.01$, the deterministic decay $0.99 = 1 - h$ changes $x$ by $O(h)$, but the noise kick $\sqrt{2h} = \sqrt{0.02} \approx 0.141$ has size $O(\sqrt{h})$. Its **square** is $O(h)$, breaking the ordinary chain rule!
:::
:::

::: {.slide title="Why Deterministic Flows Cannot Forget Their Start"}
[Forward Noising Processes]{.kicker}

A generative diffusion model needs a forward process that maps **every** data distribution $p_0 = p_{\text{data}}$ to the same tractable prior $p_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$. No invertible ODE flow $\Phi_T$ can do this (if $\Phi_T$ mapped two laws $p \ne q$ to the same law, applying $\Phi_T^{-1}$ would force $p = q$):

![](../img/mdl-dyn-sde-paths.svg)

- **Drift Alone ($\dot{x} = -x$):** Collapses $x_0 = \pm 2$ to the point mass $0$ ($\pm 0.0996$ at $t=3$), not a Gaussian.
- **Noise Alone ($dX = \sqrt{2}\,dW$):** Spreads each start as $\mathcal{N}(x_0, 2t)$ (**variance-exploding**); usable once $\sqrt{2t} \gg |x_0|$, but has no stationary limit.
- **Drift Plus Noise ($dX = -X\,dt + \sqrt{2}\,dW$):** Mean decay and noise injection balance at the stationary law $\mathcal{N}(0, 1)$ ($\mathcal{N}(\pm 0.0996, 0.9975)$ by $t = 3$).
:::

::: {.slide title="From Random Walks to the √Δt Scaling Law"}
[Constructing Brownian Motion]{.kicker}

::: {.cols .vc}
::: {.col}
Over $k = t/\Delta t$ independent steps of size $\pm c(\Delta t)^a$, the total variance at time $t$ scales as:

$$\operatorname{Var}\!\left(W_t^{(\Delta t)}\right) = \frac{t}{\Delta t} \cdot c^2 (\Delta t)^{2a} = c^2 t\,(\Delta t)^{2a - 1}$$

- **Unique critical exponent ($a = 1/2$):** If $a > 1/2$, the variance vanishes as $\Delta t \to 0$; if $a < 1/2$, it explodes to $\infty$. Only $\pm\sqrt{\Delta t}$ steps survive in the limit $W_t \sim \mathcal{N}(0, t)$.
:::

::: {.col .fig}
@fig:ch38-dyn-sqrt-dt-scaling-window
:::
:::
:::

::: {.slide title="The Wiener Process: Axioms, Covariance, and Roughness"}
[Brownian Motion]{.kicker}

The continuous-time limit $W_t$ (**Brownian motion** or the **Wiener process**) is characterized by four axioms: (1) $W_0 = 0$; (2) **independent increments** $W_t - W_s$ for $s < t$; (3) **Gaussian increments** $W_t - W_s \sim \mathcal{N}(0, t - s)$; and (4) **continuous sample paths**.

::: {.cols .vc}
::: {.col}
- **Exact Increment Recipe:** Over any interval $\Delta t$,
  $$\Delta W = W_{t + \Delta t} - W_t = \sqrt{\Delta t}\,\xi, \quad \xi \sim \mathcal{N}(0, 1)$$
- **Two-Time Covariance:** For $s \le t$, writing $W_t = W_s + (W_t - W_s)$ gives $\operatorname{Cov}(W_s, W_t) = \mathbb{E}[W_s^2] = \min(s, t)$.
- **Nowhere Differentiable Paths:** The difference quotient blows up as $\Delta t \to 0$:
  $$\left|\frac{\Delta W}{\Delta t}\right| = \frac{|\xi|}{\sqrt{\Delta t}} \;\xrightarrow{\Delta t \to 0}\; \infty$$
:::

::: {.col .fig}
![](../img/mdl-dyn-brownian-paths.svg)
:::
:::
:::

::: {.slide title="Pointwise Marginals vs. Pathwise Envelopes"}
[Simulating Brownian Paths]{.kicker}

At any single fixed time $t$, $W_t \sim \mathcal{N}(0, t)$ lies inside the $\pm 2\sqrt{t}$ two-standard-deviation envelope with probability $95.4\%$. What fraction of $20{,}000$ simulated paths stays inside $\pm 2\sqrt{t}$ across *all* grid times on $[0, 1]$?

@!sdes-brownian-paths

::: {.d2l-note}
**Pointwise vs. Pathwise Laws:** Every additional inspection time gives a jagged path another chance to cross the boundary ($70\%$ stay inside across 50 times, $54\%$ across 500 times, and $0\%$ in continuous time because Brownian paths cross $\pm 2\sqrt{t}$ infinitely often near $t = 0$).
:::
:::

::: {.slide title="Quadratic Variation of Brownian Motion: (dW)² = dt"}
[Foundation of Itô Calculus]{.kicker}

Partition $[0, t]$ into $0 = t_0 < t_1 < \cdots < t_n = t$ with mesh $h_{\max} = \max_i \Delta t_i$. For a smooth path $|x'(t)| \le M$, the sum of squared increments vanishes ($\sum_i (\Delta x_i)^2 \le M^2 h_{\max} t \to 0$). For Brownian motion, the random sum converges to the deterministic time $t$:

::: {.cols .vc}
::: {.col}
- **Mean Equals Elapsed Time:** Since $\Delta W_i \sim \mathcal{N}(0, \Delta t_i)$,
  $$\mathbb{E}[Q_n] = \sum_{i=0}^{n-1} \mathbb{E}\!\left[(\Delta W_i)^2\right] = \sum_{i=0}^{n-1} \Delta t_i = t$$
- **Variance Vanishes as $O(h_{\max})$:** Using $\mathbb{E}[\xi^4] = 3$, $\operatorname{Var}((\Delta W_i)^2) = 2\Delta t_i^2$, so by independence:
  $$\mathbb{E}\!\left[(Q_n - t)^2\right] = \operatorname{Var}(Q_n) = 2\sum_{i=0}^{n-1} \Delta t_i^2 \le 2h_{\max}t \to 0$$
:::

::: {.col .fig}
![](../img/mdl-dyn-qv-convergence.svg)
:::
:::
:::

::: {.slide title="Measuring Quadratic Variation on a Single Path"}
[Numerical Verification]{.kicker}

Viewing a single Brownian realization on $[0, 1]$ across subgrids of $n = 16$ up to $2^{18}$ intervals confirms that $\sum_{i=0}^{n-1} (\Delta W_i)^2 \to 1$ with standard deviation $\sqrt{2/n}$, whereas $\sum (\Delta \sin 2\pi t_i)^2 \approx 2\pi^2 / n \to 0$:

@!sdes-quadratic-variation

::: {.d2l-note .rule}
**The Itô Multiplication Table:** In stochastic differentials, squared Brownian increments survive at first order while higher products vanish:
$$(dW)^2 = dt, \qquad dW\,dt = 0, \qquad (dt)^2 = 0$$
:::
:::

::: {.slide title="Why Riemann Sums Disagree on Stochastic Integrals"}
[The Evaluation-Point Dilemma]{.kicker}

For a smooth curve, left-endpoint and right-endpoint Riemann sums converge to the same integral. For $\int_0^t W_s\,dW_s$, their difference is the quadratic variation:

$$\sum_{i=0}^{n-1} W_{t_{i+1}}\,\Delta W_i - \sum_{i=0}^{n-1} W_{t_i}\,\Delta W_i = \sum_{i=0}^{n-1} (\Delta W_i)^2 \;\xrightarrow{L^2}\; t$$

@!sdes-left-right-sums

- **Two Different Limits:** Because the sum of left and right approximations telescopes ($\sum_i (W_{t_{i+1}} + W_{t_i})\Delta W_i = \sum_i (W_{t_{i+1}}^2 - W_{t_i}^2) = W_t^2$), the left-endpoint sum converges to $\frac{1}{2}(W_t^2 - t)$ while the right-endpoint sum converges to $\frac{1}{2}(W_t^2 + t)$.
:::

::: {.slide title="The Itô Integral and the Itô Isometry"}
[Stochastic Integration]{.kicker}

::: {.cols .vc}
::: {.col}
**Itô's integral** evaluates $H_s$ at the **left endpoint** $t_i$, so $H_i$ is independent of the future kick $\Delta W_i = W_{t_{i+1}} - W_{t_i}$:

$$\mathbb{E}\!\left[\int_0^t H_s\,dW_s\right] = 0, \qquad \mathbb{E}\!\left[\left(\int_0^t H_s\,dW_s\right)^{\!2}\right] = \int_0^t \mathbb{E}\!\left[H_s^2\right]ds$$

- **Martingale & Itô isometry:** Left-endpoint evaluation guarantees zero mean and eliminates all cross terms $i < j$ when squaring the sum.
:::

::: {.col .fig}
@fig:ch38-dyn-ito-isometry-martingale
:::
:::
:::

::: {.slide title="Itô Stochastic Differential Equations: Drift & Noise"}
[Anatomy of an SDE]{.kicker}

An **Itô stochastic differential equation (SDE)** combines a deterministic velocity arrow with a random Brownian kick:

$$dX = f(X, t)\,dt + g(X, t)\,dW \quad \Longleftrightarrow \quad X_t = X_0 + \int_0^t f(X_s, s)\,ds + \int_0^t g(X_s, s)\,dW_s$$

- **Drift $f(X, t)$:** The conditional mean velocity over a short interval, $\mathbb{E}[dX \mid X_t] = f(X_t, t)\,dt$.
- **Diffusion $g(X, t)$:** The noise amplitude setting the local variance injection rate, $\operatorname{Var}(dX \mid X_t) = g(X_t, t)^2\,dt$.
- **Additive vs. Multiplicative Noise:**
  - **Additive Noise ($g = g(t)$):** Noise amplitude depends only on time (used in all standard diffusion models).
  - **Multiplicative Noise ($g = g(X, t)$):** Noise amplitude scales with the state (e.g., geometric Brownian motion $g(X) = \nu X$).
:::

::: {.slide title="Why the Ordinary Chain Rule Fails: Jensen's Gap"}
[Geometric Intuition for Itô's Lemma]{.kicker}

The ordinary chain rule predicts $d(W_t^2) = 2W_t\,dW_t$, which integrates to $\int_0^t 2W_s\,dW_s = W_t^2 - t$—missing the $+t$ term! One step of a coin-flip walk shows why any curved function $\phi(x)$ experiences a deterministic drift from symmetric noise:

::: {.cols .vc}
::: {.col}
- **Symmetric Kick on a Convex Curve:** From $x$, a step $\pm\sqrt{\Delta t}$ lands at $x + \sqrt{\Delta t}$ or $x - \sqrt{\Delta t}$ with probability $\frac{1}{2}$ each.
- **Second-Order Jensen Gap:** For $\phi(x) = x^2$ ($\phi''(x) = 2$), the average value after the step is:
  $$\frac{1}{2}\left[(x + \sqrt{\Delta t})^2 + (x - \sqrt{\Delta t})^2\right] = x^2 + \Delta t = \phi(x) + \frac{1}{2}\phi''(x)\,\Delta t$$
- **Non-Vanishing Rate:** Because $(\Delta W)^2 \approx \Delta t$, the curvature gain $+\frac{1}{2}\phi''(x)$ per unit time survives the $\Delta t \to 0$ limit.
:::

::: {.col .fig}
![](../img/mdl-dyn-ito-jensen.svg)
:::
:::
:::

::: {.slide title="Theorem: Itô's Lemma (The Stochastic Chain Rule)"}
[Core Theorem]{.kicker}

::: {.cols .vc}
::: {.col}
Expanding $\phi(X_t, t)$ to second order in $dX = f\,dt + g\,dW$ and applying $(dX)^2 = g^2(dW)^2 = g^2\,dt$ yields **Itô's lemma**:

$$d\phi(X_t, t) = \left(\phi_t + f\,\phi_x + \frac{1}{2}g^2\,\phi_{xx}\right)dt + g\,\phi_x\,dW_t$$

- **Multivariate isotropic form ($\mathbb{R}^d$):** $d\phi = (\partial_t\phi + \nabla\phi\cdot\mathbf{f} + \tfrac12 g(t)^2\Delta\phi)\,dt + g(t)\nabla\phi\cdot d\mathbf{W}_t$, where $\tfrac12 g^2\Delta\phi$ is the Itô curvature correction.
:::

::: {.col .fig}
@fig:ch38-dyn-ito-lemma-expansion
:::
:::
:::

::: {.slide title="Watching the Itô Correction Accumulate Along One Path"}
[Numerical Verification of Itô's Lemma]{.kicker}

On $X_t = W_t$ ($f = 0, g = 1$) with $\phi(x) = x^2$, Itô's lemma gives $d(W_t^2) = 2W_t\,dW_t + dt$, so $W_t^2 - \int_0^t 2W_s\,dW_s = t$ deterministically along every individual path:

@!sdes-ito-correction

::: {.d2l-note}
The naive chain-rule curve $\int_0^t 2W_s\,dW_s$ sags below $W_t^2$ by the running quadratic variation $\sum (\Delta W_i)^2 \approx t$; adding $+t$ restores exact pathwise agreement.
:::
:::

::: {.slide title="Worked Application: Geometric Brownian Motion"}
[Multiplicative Noise & Concave Drift Shift]{.kicker}

The SDE $dX_t = \mu X_t\,dt + \nu X_t\,dW_t$ ($X_0 > 0$) models a positive quantity whose noise amplitude $g(X) = \nu X$ is proportional to its current level (**multiplicative noise**):

- **Log-Transform via Itô's Lemma:** Apply Itô's lemma to $\phi(x) = \log x$, which has $\phi'(x) = \frac{1}{x}$ and $\phi''(x) = -\frac{1}{x^2}$:
  $$d\log X_t = \frac{1}{X_t}\,dX_t + \frac{1}{2}\left(-\frac{1}{X_t^2}\right)(\nu^2 X_t^2)\,dt = \left(\mu - \frac{1}{2}\nu^2\right)dt + \nu\,dW_t$$
- **Exact Closed-Form Solution:** Integrating from $0$ to $t$ and exponentiating gives:
  $$X_t = X_0 \exp\!\left(\left(\mu - \frac{1}{2}\nu^2\right)t + \nu W_t\right)$$
- **Mean vs. Median Divergence:** Because $\log x$ is strictly concave ($\phi'' < 0$), symmetric noise slows the median growth rate to $\mu - \frac{1}{2}\nu^2$ even though the expectation still grows at the drift rate $\mathbb{E}[X_t] = X_0 e^{\mu t}$.
:::

::: {.slide title="The Ornstein–Uhlenbeck Process and Mean Reversion"}
[The Canonical Diffusion Process]{.kicker}

Adding constant-amplitude Brownian noise $g\,dW$ to the linear decay equation $\dot{x} = -\theta x$ ($\theta > 0$) produces the **Ornstein–Uhlenbeck (OU) process**:

$$dX_t = -\theta X_t\,dt + g\,dW_t$$

::: {.cols .vc}
::: {.col}
- **Exact Solution via Integrating Factor:** Apply Itô's lemma to $\phi(x, t) = e^{\theta t}x$. Because $\phi$ is linear in $x$ ($\phi_{xx} = 0$), the Itô correction is **zero**:
  $$d\!\left(e^{\theta t}X_t\right) = \theta e^{\theta t}X_t\,dt + e^{\theta t}(-\theta X_t\,dt + g\,dW_t) = g\,e^{\theta t}\,dW_t$$
- **Fading-Memory Integral:** Integrating from $0$ to $t$ gives:
  $$X_t = X_0\,e^{-\theta t} + g\int_0^t e^{-\theta(t-s)}\,dW_s$$
:::

::: {.col .fig}
![](../img/mdl-dyn-ou-mean-reversion.svg)
:::
:::
:::

::: {.slide title="Closed-Form OU Gaussian Kernel and Stationary Law"}
[Transition Kernel]{.kicker}

Because the integrand $e^{-\theta(t-s)}$ is deterministic, $\int_0^t e^{-\theta(t-s)}\,dW_s$ is a Gaussian random variable with mean $0$ and variance given directly by the **Itô isometry**:

$$\operatorname{Var}(X_t \mid X_0 = x_0) = g^2 \int_0^t e^{-2\theta(t-s)}\,ds = \frac{g^2}{2\theta}\left(1 - e^{-2\theta t}\right)$$

::: {.d2l-note .rule}
**Proposition (OU Transition Kernel):** Given $X_0 = x_0$, the state at any time $t > 0$ is exact Gaussian:
$$X_t \mid X_0 = x_0 \;\sim\; \mathcal{N}\!\left(\alpha_t x_0,\; \sigma_t^2\right), \qquad \alpha_t = e^{-\theta t}, \qquad \sigma_t^2 = \frac{g^2}{2\theta}\left(1 - e^{-2\theta t}\right)$$
:::

- **Stationary Distribution ($t \to \infty$):** After a few multiples of the relaxation time $1/\theta$, the mean $x_0 e^{-\theta t} \to 0$ and the variance saturates at $\sigma_\infty^2 = \frac{g^2}{2\theta}$, so $X_t \to \mathcal{N}\!\left(0, \frac{g^2}{2\theta}\right)$ from every starting state $x_0$.
:::

::: {.slide title="Variance-Preserving Normalization and the VP-SDE"}
[Foundation of DDPM]{.kicker}

Choosing $g^2 = 2\theta$ fixes the stationary variance to $\frac{g^2}{2\theta} = 1$ and ties the noise variance directly to the signal decay via $\sigma_t^2 = 1 - \alpha_t^2$. Writing $\bar{\alpha}_t = \alpha_t^2 = e^{-2\theta t}$ (DDPM notation):

$$X_t = \sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1) \text{ independent of } X_0$$

- **Why It Is Called Variance-Preserving (VP):** If the data distribution $X_0$ has mean $0$ and unit variance $\operatorname{Var}(X_0) = 1$, then at **every** time $t \ge 0$:
  $$\operatorname{Var}(X_t) = \bar{\alpha}_t \operatorname{Var}(X_0) + (1 - \bar{\alpha}_t) = \bar{\alpha}_t + 1 - \bar{\alpha}_t = 1$$
- **Shape Reshaping:** Higher moments still evolve into a Gaussian—for Rademacher data $X_0 \in \{\pm 1\}$, the fourth moment climbs smoothly from $\mathbb{E}[X_0^4] = 1$ to $\mathbb{E}[X_t^4] = 3 - 2\bar{\alpha}_t^2 \to 3$.
- **Time-Dependent Schedule (VP-SDE):** Allowing the rate $\beta(t) = 2\theta(t) = g(t)^2$ to increase with time gives $dX = -\frac{1}{2}\beta(t)X\,dt + \sqrt{\beta(t)}\,dW$, whose marginal has $\bar{\alpha}_t = \exp\!\big(-\int_0^t \beta(s)\,ds\big)$.
:::

::: {.slide title="Simulating SDEs: Euler–Maruyama vs. Exact Transitions"}
[Numerical SDE Schemes]{.kicker}

::: {.cols .vc}
::: {.col}
**Euler–Maruyama** freezes $f$ and $g$ at the start of each step $\Delta t$ and injects a $\sqrt{\Delta t}$ Gaussian kick:

$$\hat{X}_{n+1} = \hat{X}_n + f(\hat{X}_n, t_n)\,\Delta t + g(\hat{X}_n, t_n)\sqrt{\Delta t}\,\xi_n$$

- **Exact OU vs. Euler–Maruyama ($\theta = 1, g = \sqrt{2}, \Delta t = 0.01$):** Exact DDPM steps use $a = e^{-\theta\Delta t}$ and $b = 1 - e^{-2\theta\Delta t}$ ($v_\infty = 1.000$), whereas Euler–Maruyama's $1 - \theta\Delta t$ inflates $v_\infty = 1.005$ ($+0.5\%$ bias).
:::

::: {.col .fig}
@fig:ch38-dyn-ou-em-vs-exact
:::
:::
:::

::: {.slide title="Implementing Batched Euler–Maruyama With Shared Noise"}
[Clean SDE Simulator]{.kicker}

Passing the Brownian increments `dW` as an explicit argument allows us to drive coarse and fine time grids with the **exact same noise realization** when measuring strong pathwise convergence:

@sdes-euler-maruyama

- **Milstein's Higher-Order Correction ($p_{\text{strong}} = 1$):** Adding the omitted intra-step diffusion integral $\frac{1}{2}g(\hat{X}_n)\,\partial_x g(\hat{X}_n)\big((\Delta W_n)^2 - \Delta t\big)$ to the Euler–Maruyama update restores strong order $1$ even for multiplicative noise.
:::

::: {.slide title="Strong Pathwise Error vs. Weak Distributional Error"}
[Two Notions of Convergence]{.kicker}

Because SDE solutions are random variables, numerical accuracy can be judged either by how closely a simulated path tracks the specific realization of $W_t$, or by how closely the histogram at time $T$ matches the true marginal law:

![](../img/mdl-dyn-strong-weak.svg)

- **Strong Error ($\mathbb{E}\,|\hat{X}_N - X_T| \le C\,\Delta t^p$):** Couples coarse and fine runs with the **exact same Brownian path** $W_t$. Required for multilevel Monte Carlo.
- **Weak Error ($|\mathbb{E}\,\varphi(\hat{X}_N) - \mathbb{E}\,\varphi(X_T)| \le C\,\Delta t^p$):** Compares expectations/histograms at time $T$ across independent noise draws. **Generative diffusion samplers only require weak convergence.**
:::

::: {.slide title="Strong Order: Additive vs. Multiplicative Noise"}
[Pathwise Convergence]{.kicker}

Why does Euler–Maruyama achieve strong order $p = 1$ on the additive-noise OU process, but drop to strong order $p = 1/2$ on multiplicative-noise geometric Brownian motion ($g(X) = \nu X$)?

::: {.cols .vc}
::: {.col}
- **The Omitted Milstein Term:** Freezing $g(X)$ misses its intra-step variation $g(X_s) - g(X_{t_n}) \approx g\,\partial_x g\,(W_s - W_{t_n})$, whose Itô integral over $[t_n, t_{n+1}]$ is:
  $$g\,\partial_x g \int_{t_n}^{t_{n+1}} (W_s - W_{t_n})\,dW_s = \frac{1}{2}g\,\partial_x g\left((\Delta W_n)^2 - \Delta t\right)$$
- **Random Walk of Step Errors:** Each step omits a zero-mean error of size $O(\Delta t)$; summing $N = T/\Delta t$ independent errors gives $\sqrt{N}\,\Delta t = O(\Delta t^{1/2})$ unless $\partial_x g = 0$ (additive noise, where strong order is $O(\Delta t)$).
:::

::: {.col .fig}
@!sdes-em-strong-order
:::
:::
:::

::: {.slide title="Measuring Weak Order 1 via Exact Moment Recursions"}
[Distributional Convergence]{.kicker}

Because every omitted stochastic term has **mean zero**, the one-step distributional bias of Euler–Maruyama is $O(\Delta t^2)$, accumulating over $N = T/\Delta t$ steps to **weak order $1$** ($O(\Delta t)$) for both additive and multiplicative noise:

@!sdes-em-weak-order

::: {.d2l-note .warn}
**Itô vs. Stratonovich Pitfall:** Naive higher-order ODE solvers cannot be plugged directly into multiplicative SDEs—applying Heun's predictor-corrector to $g(X)$ averages the left and right endpoints, converging to the **Stratonovich** solution $X_0 e^{\mu T + \nu W_T}$ instead of the Itô solution $X_0 e^{(\mu - \nu^2/2)T + \nu W_T}$!
:::
:::

::: {.slide title="Concept Check: Itô Calculus and Ornstein–Uhlenbeck"}
[Concept Check]{.kicker}

Work out these three stochastic calculus problems:

1. **Itô's Lemma on a Cubic:** For standard Brownian motion $W_t$, compute the stochastic differential $d(W_t^3)$ and find $\mathbb{E}[W_t^3]$ and $\mathbb{E}[W_t^2]$.
2. **Stationary Variance:** Consider the OU process $dX_t = -3X_t\,dt + \sqrt{3}\,dW_t$ started at $X_0 = 5$. What is its limiting distribution as $t \to \infty$?
3. **Strong vs. Weak Order:** If you quarter the step size $\Delta t \to \Delta t / 4$ in Euler–Maruyama on a multiplicative-noise SDE, by what factor does the **strong error** shrink, and by what factor does the **weak error** shrink?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. With $\phi(x) = x^3$, $\phi' = 3x^2$ and $\phi'' = 6x$, so $d(W_t^3) = 3W_t^2\,dW_t + \frac{1}{2}(1)^2(6W_t)\,dt = 3W_t^2\,dW_t + 3W_t\,dt$. By symmetry $\mathbb{E}[W_t^3] = 0$, while $d(W_t^2) = 2W_t\,dW_t + dt \implies \mathbb{E}[W_t^2] = t$.
2. With $\theta = 3$ and $g^2 = 3$, the stationary distribution is $\mathcal{N}\!\left(0, \frac{g^2}{2\theta}\right) = \mathcal{N}\!\left(0, \frac{1}{2}\right)$.
3. Strong error (order $1/2$) shrinks by $4^{1/2} = 2\times$; weak error (order $1$) shrinks by $4^1 = 4\times$.
:::
:::

::: {.slide title="Summary: Brownian Motion, Itô Calculus, and SDEs"}
[Key Takeaways]{.kicker}

| Concept | Mathematical Formula | Role in Diffusion & Stochastic Modeling |
| :--- | :--- | :--- |
| **Brownian Increment** | $\Delta W = W_{t+\Delta t} - W_t = \sqrt{\Delta t}\,\xi$, $\xi \sim \mathcal{N}(0, 1)$ | Unique scaling with finite, non-zero variance $\operatorname{Var}(W_t) = t$ |
| **Quadratic Variation** | $\sum_i (\Delta W_i)^2 \to t$ in $L^2$ $\iff (dW)^2 = dt$ | Causes second-order Taylor terms to survive at order $dt$ |
| **Itô Integral & Isometry** | $\mathbb{E}\!\left[\int_0^t H_s\,dW_s\right] = 0$, $\mathbb{E}\!\left[\left(\int_0^t H_s\,dW_s\right)^2\right] = \int_0^t \mathbb{E}[H_s^2]\,ds$ | Left-endpoint evaluation preserves zero mean and computes variances |
| **Itô's Lemma** | $d\phi = (\phi_t + f\phi_x + \frac{1}{2}g^2\phi_{xx})\,dt + g\phi_x\,dW$ | Stochastic chain rule; generates the Laplacian $\frac{1}{2}g^2\Delta\phi$ in $\mathbb{R}^d$ |
| **OU / VP-SDE Kernel** | $X_t \mid X_0 \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}X_0, (1-\bar{\alpha}_t)\mathbf{I})$ ($g^2 = 2\theta$) | Closed-form forward noising to $\mathcal{N}(\mathbf{0}, \mathbf{I})$ in a single Gaussian draw |
| **Euler–Maruyama** | $\hat{X}_{n+1} = \hat{X}_n + f\Delta t + g\sqrt{\Delta t}\,\xi_n$ | Strong order $1$ (additive) or $1/2$ (multiplicative); weak order $1$ |
:::

