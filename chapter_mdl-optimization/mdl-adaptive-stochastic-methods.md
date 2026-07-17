# Stochastic and Adaptive Methods
:label:`sec_mdl-adaptive-stochastic-methods`

The optimizer that trains essentially every modern network is AdamW with warmup
and a decaying learning-rate schedule, not the plain gradient descent whose
theory :numref:`sec_mdl-gradient-based-optimization` developed. This section
closes the gap between that theory and that default. Two questions were left
open there. First, the smooth nonconvex benchmark used to reason about
deep networks, the $O(1/K)$ stationarity rate, was proved for the exact
gradient, while training uses noisy minibatch estimates; what survives?
Second, the whole $\kappa$ story was about a *single* step size $\eta$ forced
to respect the stiffest curvature direction. The obvious escape, a *separate*
step size per coordinate, is exactly what AdaGrad, RMSProp, and Adam
implement, and it deserves the same treatment the global $\eta$ received: a
derivation, a rate, and an account of where it fails. Along the way we
derive Adam's bias correction rather than assert it, state and verify the
counterexample showing that Adam can converge to the *worst* point of a convex
problem, work out why weight decay through a preconditioner is not weight
decay, and give learning-rate schedules and warmup the mathematical reading
they usually go without.

We lean on :numref:`sec_mdl-gradient-based-optimization` throughout (the
descent lemma, the condition number $\kappa$, the minibatch variance
$\mathrm{tr}\,\Sigma/b$, and the noise ball are all consumed here) and on
:numref:`sec_mdl-eigendecompositions` for the diagonal-metric view of
preconditioning. The MAP reading of weight decay comes from
:numref:`sec_mdl-maximum_likelihood`, its constraint reading from
:numref:`subsec_mdl-weight-decay-duality`, and the closing preconditioning
ladder reaches back to the SVD and Newton--Schulz material of
:numref:`sec_mdl-svd-low-rank`. Standard references are
:citet:`Bottou.Curtis.Nocedal.2018` for the stochastic theory and the original
papers cited as we go. As everywhere in this chapter, the code is plain NumPy:
every optimizer below is under ten lines written by hand, which is the fastest
way to see there is no magic in any of them.

```{.python .input #adaptive-stochastic-methods-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #adaptive-stochastic-methods-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
```

```{.python .input #adaptive-stochastic-methods-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
```

```{.python .input #adaptive-stochastic-methods-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
```

## SGD Without Convexity
:label:`subsec_mdl-nonconvex-sgd`

### What the Deterministic Theorem Leaves Open

Recall the two bookends from :numref:`sec_mdl-gradient-based-optimization`.
For $L$-smooth $f$ bounded below, *deterministic* gradient descent satisfies
$\min_{k < K} \|\nabla f(\mathbf{x}_k)\|^2 \le 2L(f(\mathbf{x}_0) - f^\star)/K$:
stationarity at rate $O(1/K)$, no convexity needed. And for *stochastic*
gradients on a strongly convex quadratic, a fixed step converges to a noise
ball of squared radius $\approx \eta\sigma^2/(2\lambda)$ around the minimizer
:eqref:`eq_mdl-opt-noise-ball`. Training a
network lives in the intersection of the two hypotheses, nonconvex *and*
noisy, which neither result covers. The theorem that does is due to
:citet:`Ghadimi.Lan.2013`, and its proof is nothing more than the descent
lemma with expectations taken at the right moment.

Model one SGD step as $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\mathbf{g}_k$,
where the stochastic gradient is unbiased with bounded variance:

$$
\mathbb{E}\left[\mathbf{g}_k \mid \mathbf{x}_k\right] = \nabla f(\mathbf{x}_k),
\qquad
\mathbb{E}\left[\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2 \mid \mathbf{x}_k\right] \;\le\; \sigma^2.
$$

The minibatch proposition of :numref:`sec_mdl-gradient-based-optimization`
supplies exactly these hypotheses, with $\sigma^2 = \mathrm{tr}\,\Sigma/b$.

### The Ghadimi--Lan Rate

**Proposition (SGD finds approximate stationary points).** *Let $f$ be
$L$-smooth and bounded below by $f^\star$, write
$\Delta = f(\mathbf{x}_0) - f^\star$, and run SGD with a constant step
$\eta \le 1/L$. Then*

$$
\frac{1}{K} \sum_{k=0}^{K-1} \mathbb{E}\left[\|\nabla f(\mathbf{x}_k)\|^2\right]
\;\le\; \frac{2\Delta}{\eta K} + L\eta\sigma^2,
$$
:eqlabel:`eq_mdl-opt-ghadimi-lan`

*and choosing $\eta = \min\bigl(1/L,\ \sqrt{2\Delta/(L\sigma^2 K)}\bigr)$
balances the two terms:*

$$
\mathbb{E}\left[\|\nabla f(\mathbf{x}_R)\|^2\right]
\;\le\; \frac{2L\Delta}{K} + 2\sigma\,\sqrt{\frac{2L\Delta}{K}},
\qquad R \sim \mathrm{Uniform}\{0, \ldots, K-1\}.
$$

**Proof.** Apply the quadratic ceiling
:eqref:`eq_mdl-opt-quadratic-ceiling` to the step actually taken,
$\mathbf{x}_{k+1} - \mathbf{x}_k = -\eta\,\mathbf{g}_k$:

$$
f(\mathbf{x}_{k+1}) \;\le\; f(\mathbf{x}_k) - \eta\, \nabla f(\mathbf{x}_k)^\top \mathbf{g}_k + \tfrac{L\eta^2}{2}\, \|\mathbf{g}_k\|^2.
$$

Take the expectation conditioned on $\mathbf{x}_k$. Unbiasedness turns the
middle term into $-\eta\,\|\nabla f(\mathbf{x}_k)\|^2$, and the variance
decomposition
$\mathbb{E}\|\mathbf{g}_k\|^2 = \|\nabla f(\mathbf{x}_k)\|^2 + \mathbb{E}\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2$
bounds the last by
$\tfrac{L\eta^2}{2}(\|\nabla f(\mathbf{x}_k)\|^2 + \sigma^2)$. Collecting,

$$
\mathbb{E}\left[f(\mathbf{x}_{k+1}) \mid \mathbf{x}_k\right]
\;\le\; f(\mathbf{x}_k) - \eta\left(1 - \tfrac{L\eta}{2}\right) \|\nabla f(\mathbf{x}_k)\|^2 + \tfrac{L\eta^2 \sigma^2}{2},
$$

the descent lemma with one new term: a *noise tax* of
$\tfrac{L}{2}\eta^2\sigma^2$ per step, paid whether or not the gradient is
large. For $\eta \le 1/L$ the bracket is at least $\tfrac12$. Take total
expectations, telescope over $k = 0, \ldots, K-1$ (the left side collapses to
$\mathbb{E}[f(\mathbf{x}_K)] - f(\mathbf{x}_0) \ge -\Delta$), and divide by
$\eta K/2$ to get :eqref:`eq_mdl-opt-ghadimi-lan`. The first term of the bound
falls in $\eta$, the second grows; the stated $\eta$ equalizes them, and the
average over $k$ *is* the expectation at a uniformly random iterate $R$.
$\blacksquare$

Read the optimized bound against its deterministic ancestor. Without noise
($\sigma = 0$) the second term vanishes and the $2L\Delta/K$ term reproduces
the $O(1/K)$ rate exactly. With noise, the second term dominates for large
$K$, and it decays like $K^{-1/2}$: **noise turns the deterministic $1/K$
into a square root**, the same square root that governed the minibatch
variance ($1/b$ energy, $1/\sqrt{b}$ amplitude). Be careful about what is
bounded: the theorem controls the **expected squared gradient norm**. Running
$100\times$ longer reduces that bound by $10\times$; interpreting it as a bound
on the gradient norm itself gives only a factor $\sqrt{10}$. A tenfold reduction
of that norm would require $10{,}000\times$ the budget under this worst-case
rate. The prescribed step obeys the same
logic as the noise-ball analysis: a longer budget affords a smaller $\eta$,
shrinking the floor the iterates can reach.

One subtlety is a feature of the theorem rather than a defect. The
deterministic result bounded $\min_k \|\nabla f(\mathbf{x}_k)\|^2$: you may
keep the best iterate, because you can *evaluate* which one it is. With
stochastic gradients you never observe $\|\nabla f(\mathbf{x}_k)\|$, only
noisy estimates, so "return the best iterate" is not an executable
instruction. The theorem's output is therefore a *randomly selected* iterate,
whose expected squared gradient norm is the average that
:eqref:`eq_mdl-opt-ghadimi-lan` controls. The cell verifies precisely this
distinction on a nonconvex toy (a one-neuron $\tanh$ regression with noisy
labels, single-example gradients), sweeping the budget $K$ with the
theory-prescribed $\eta \propto 1/\sqrt{K}$ and measuring both the
random-iterate average and the (non-executable) trajectory minimum of
$\|\nabla f\|^2$, each averaged over $20$ seeds.

```{.python .input #adaptive-stochastic-methods-ghadimi-lan}
n, d = 64, 5
rng = np.random.default_rng(0)
Xd = rng.normal(size=(n, d))
yd = np.tanh(Xd @ rng.normal(size=d)) + 0.5 * rng.normal(size=n)

def full_grad(w):                              # exact gradient, for measurement
    t = np.tanh(Xd @ w)
    return 2 * (Xd * ((t - yd) * (1 - t**2))[:, None]).mean(axis=0)

def stoch_grad(w, i):                          # single-example gradient (b = 1)
    t = np.tanh(Xd[i] @ w)
    return 2 * (t - yd[i]) * (1 - t**2) * Xd[i]

Ks, avg_res, min_res = [125, 500, 2000, 8000], [], []
print('    K    eta_K     E|grad(x_R)|^2   min-so-far')
for K in Ks:
    eta = 0.5 / np.sqrt(K)                     # the theorem's eta ~ 1/sqrt(K)
    avgs, mins = [], []
    for seed in range(20):
        rg, w, tot, m = np.random.default_rng(seed), np.zeros(d), 0.0, np.inf
        for k in range(K):
            g2 = (full_grad(w)**2).sum()
            tot, m = tot + g2, min(m, g2)
            w = w - eta * stoch_grad(w, rg.integers(0, n))
        avgs.append(tot / K), mins.append(m)
    avg_res.append(np.mean(avgs)), min_res.append(np.mean(mins))
    print(f'{K:5d}   {eta:.4f}      {avg_res[-1]:.3e}      {min_res[-1]:.3e}')
print(f'log-log slope, random iterate: '
      f'{np.polyfit(np.log(Ks), np.log(avg_res), 1)[0]:.2f}  (theory: -1/2)')
print(f'log-log slope, min-so-far:     '
      f'{np.polyfit(np.log(Ks), np.log(min_res), 1)[0]:.2f}')
```

The random-iterate column scales as promised: a $64\times$ increase in budget
yields a $12\times$ reduction, log-log slope $-0.60$, within reach of the
theoretical $-1/2$. (The residual steepness comes from the average itself:
the trajectory average still contains the early high-gradient iterates, whose
contribution decays like $1/K$. The bound's transient $2\Delta/(\eta K)$ term
is innocent here: with the cell's $\eta \propto 1/\sqrt{K}$ it is of the same
$K^{-1/2}$ order as the noise term, so it cannot steepen the slope.) The
min-so-far column falls much faster,
slope $-1.23$: on this benign toy the best iterate beats the guarantee by a
wide margin, exactly why one would love to return it and exactly what the
noise forbids you from *identifying*. Worst-case bounds are contracts: the
guaranteed rate here is $K^{-1/2}$, and the schedule designs of this section
build on it rather than on the lucky minimum.

## Per-Coordinate Step Sizes
:label:`subsec_mdl-per-coordinate`

### One Global Step Must Respect the Stiffest Mode

In :numref:`sec_mdl-gradient-based-optimization`, stability chained the
single step size to the stiffest mode ($\eta < 2/\lambda_{\max}$), so the
flattest mode advances by only $\eta\lambda_{\min} \approx 2/\kappa$ per
step: one scalar doing a matrix's job, at cost linear in $\kappa$. But
suppose the coordinate axes happen to *be* the eigendirections, as they are
for
$f(\mathbf{x}) = \tfrac12 \mathbf{x}^\top \mathrm{diag}(\boldsymbol{\lambda})\, \mathbf{x}$,
and we allow a *separate* step size $\eta_i$ per coordinate. Then
$\eta_i = 1/\lambda_i$ contracts every mode to zero in one step: a diagonal
matrix of step sizes is a diagonal Newton's method, and the condition number
simply disappears from the problem. This is the entire thesis of adaptive
methods, drawn in :numref:`fig_mdl-opt-per-coordinate`: gradient descent's
zig-zag comes from using one number where $n$ are called for, and it
disappears once each coordinate gets its own.

![The section's thesis, drawn. Both panels show the same elongated quadratic ($\kappa = 20$) from the same start, and both paths are actual runs of the stated update. Left: gradient descent with one global step size near the stability ceiling zig-zags across the stiff axis while crawling along the slow one. Right: Adam's per-coordinate normalization (with $\beta_1 = 0$, to isolate the rescaling from momentum) moves both coordinates at comparable speed and curves smoothly into the minimum: the valley is effectively rounded, at the cost of estimating the rescaling from gradients alone.](../img/mdl-opt-per-coordinate.svg)
:label:`fig_mdl-opt-per-coordinate`

Two obstacles stand between this observation and an algorithm. The
eigenvalues are unknown, and off the quadratic model they vary from place to
place; and with stochastic gradients we never see clean curvature, only noisy
first-order information. Everything in the adaptive family is a scheme for
estimating a useful diagonal rescaling *from the gradients themselves*.

### AdaGrad: Calibrating Steps by Accumulated Evidence

The first such scheme, **AdaGrad** :cite:`Duchi.Hazan.Singer.2011` (shown in
action in :numref:`sec_adam`), keeps a
running sum of squared gradients per coordinate and divides by its square
root:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\; \mathbf{g}_t,
$$
:eqlabel:`eq_mdl-opt-adagrad`

with all operations elementwise and $\epsilon$ a small constant that keeps the
division finite. Two independent derivations land on this rule.

*The metric view.* Exercise 2 of
:numref:`sec_mdl-gradient-based-optimization` showed that steepest descent is
relative to a choice of norm: measuring step length by
$\|\mathbf{d}\|_A = \sqrt{\mathbf{d}^\top A\, \mathbf{d}}$ makes the steepest
direction $-A^{-1}\nabla f$, a *preconditioned* gradient. AdaGrad is steepest
descent in the metric $A_t = \mathrm{diag}(\sqrt{\mathbf{s}_t})$: it declares
a coordinate "expensive to move" in proportion to the evidence
$\sqrt{\sum_{s \le t} g_{s,i}^2}$ that its gradient has been large. Newton
would use the true curvature $\mathrm{diag}(\lambda_i)$; AdaGrad substitutes
the cheapest available proxy, the gradient's own history (Exercise 1 derives
the update from this view).

*The regret view, and why the square root.* AdaGrad was invented for
**online convex optimization**: convex losses $f_1, f_2, \ldots$ arrive one
at a time, the algorithm must play $\mathbf{x}_t$ *before* seeing $f_t$, and
performance is measured by **regret**, the gap
$\sum_{t=1}^{T} f_t(\mathbf{x}_t) - \min_{\mathbf{x}} \sum_{t=1}^{T} f_t(\mathbf{x})$
between the accumulated loss and that of the best *fixed* point chosen in
hindsight; an algorithm learns, in this sense, when its average regret
(regret divided by $T$) vanishes as $T \to \infty$. The problems AdaGrad was
built for have **sparse features**: think of a bag-of-words
model where the coordinate for a rare word receives a nonzero gradient once
in ten thousand steps. A global $\eta_t \propto 1/\sqrt{t}$ decays that rare
coordinate's step size along with everyone else's, so by the time the rare
evidence arrives the step is too small to use it. Per-coordinate
accumulation fixes this: coordinate $i$'s effective step
$\eta/\sqrt{s_{t,i}}$ decays with *its own* activity, not with wall-clock
time, so rare-but-informative coordinates keep large steps. The square root
is what the regret analysis selects: with the per-coordinate rates chosen
this way, the accumulated-gradient terms in the regret bound sum to exactly
$\sum_i \sqrt{\sum_t g_{t,i}^2}$, a quantity that can beat the dimension-free
$G\sqrt{T}$ regret of plain SGD by a factor up to $\sqrt{d}$ when gradients
are sparse :cite:`Duchi.Hazan.Singer.2011`.

The same accumulation is AdaGrad's defect outside the convex setting. Since
$\mathbf{s}_t$ never forgets, the effective step decays like
$\eta/(\sigma\sqrt{t})$ under persistent gradient noise: a
Robbins--Monro-style schedule *hard-wired into the optimizer*. On a convex
problem that is precisely what convergence demands; on a nonconvex landscape
the optimizer may need to cross a plateau or a saddle region long after
$\mathbf{s}_t$ has grown large, and it arrives with steps ground down to
nothing. The method stalls because it remembers too much.

### RMSProp: Forgetting on Purpose

The repair is to replace the all-time sum with an **exponential moving
average**, the same fix that turns a cumulative mean into
something that can track a moving target. **RMSProp**
:cite:`Tieleman.Hinton.2012` (shown in action in :numref:`sec_adam`) keeps

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t} + \epsilon}\; \mathbf{g}_t,
$$
:eqlabel:`eq_mdl-opt-rmsprop`

so $\mathbf{v}_t$ is a weighted average of recent squared gradients with an
effective memory of about $1/(1 - \beta_2)$ steps ($10$ at RMSProp's standard
decay $\beta_2 = 0.9$). The preconditioner now estimates the *current* gradient
scale per coordinate rather than the lifetime total: steps no longer decay to
zero by construction, and the method can keep moving on a nonstationary
landscape. The tradeoff, invisible here but fatal below, is that forgetting
discards exactly the kind of rare evidence AdaGrad was built to retain.

### Adam: Momentum, Second Moments, and Bias Correction

**Adam** :cite:`Kingma.Ba.2014` (shown in action in :numref:`sec_adam`)
completes the recipe by applying the same exponential averaging to the
gradient itself and correcting both averages for their startup bias. The
gradient average $\mathbf{m}_t$ is momentum
(:numref:`subsec_mdl-momentum-acceleration`) in averaged form: where the
heavy-ball buffer of that section *accumulates* gradients, the
$(1 - \beta_1)$ factor here makes $\mathbf{m}_t$ a weighted average of them,
the same method up to a rescaling of the step size:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\, \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t,
\qquad &
\hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t), \\
\mathbf{v}_t &= \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad &
\hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t), \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \eta\; \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-adam`

with defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, and
$\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$. The default $\beta_2 = 0.999$
gives the second-moment estimate an effective memory of
about $1/(1-\beta_2) = 1000$ steps, a hundred times longer than RMSProp's
customary ten. The bias correction is a two-line computation. Unroll the
recursion:

$$
\mathbf{v}_t = (1 - \beta_2) \sum_{s=1}^{t} \beta_2^{\,t-s}\, \mathbf{g}_s^2 .
$$

If the squared-gradient scale is (locally) stationary,
$\mathbb{E}[\mathbf{g}_s^2] = \bar{\mathbf{g}^2}$ for all $s$, then by the
geometric series $\sum_{s=1}^{t} \beta_2^{\,t-s} = (1 - \beta_2^t)/(1 - \beta_2)$,

$$
\mathbb{E}[\mathbf{v}_t]
= (1 - \beta_2)\, \bar{\mathbf{g}^2} \sum_{s=1}^{t} \beta_2^{\,t-s}
= \left(1 - \beta_2^t\right) \bar{\mathbf{g}^2},
$$
:eqlabel:`eq_mdl-opt-bias-correction`

so dividing by $1 - \beta_2^t$ makes $\hat{\mathbf{v}}_t$ an *exactly*
unbiased estimate of $\bar{\mathbf{g}^2}$ at every $t$: the correction
factors cancel the zero initialization identically, at every step, rather
than merely fading with it. The same computation with $\beta_1$ handles
$\hat{\mathbf{m}}_t$ (Exercise 2). The transient being corrected is
large: at the defaults, $\mathbf{v}_{10}$ carries only about $1\%$ of its
stationary value, and the raw ratio
$\mathbf{m}_t/(\sqrt{\mathbf{v}_t} + \epsilon)$ mis-scales early steps by
the factor $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$: already $3.16$ at
$t = 1$, peaking above $6\times$ near $t = 12$, because the momentum
average saturates in about ten steps while the second-moment average needs
about a thousand. :numref:`fig_mdl-opt-bias-correction` plots the transient
and its cancellation.

![Why Adam corrects for bias. Left: with zero initialization, the running average $\mathbf{v}_t$ carries only the fraction $1-\beta_2^t$ of the true squared-gradient scale; at $\beta_2=0.999$ it has recovered just $63\%$ after $1000$ steps, and dividing by $1-\beta_2^t$ cancels the deficit exactly, at every $t$. Right: the resulting mis-scaling of the raw update $\mathbf{m}_t/\sqrt{\mathbf{v}_t}$, the factor $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$: it is $3.16$ at $t=1$, peaks above $6\times$ near $t=12$ (the numerator saturates in about ten steps, the denominator needs about a thousand), and decays back to $1$ only on the $1/(1-\beta_2)$ timescale. Uncorrected Adam takes its *largest* steps precisely when its preconditioner is estimated from the fewest samples.](../img/mdl-opt-bias-correction.svg)
:label:`fig_mdl-opt-bias-correction`

Assembled, Adam is three ideas stacked: RMSProp's per-coordinate scale
estimate, momentum's variance-averaged direction, and an exact startup
correction for both. Near a minimum of a quadratic with diagonal Hessian, the
scale estimate obeys
$\sqrt{\hat{v}_i} \approx |g_i| = \lambda_i |x_i|$, so the update on
coordinate $i$ is
$\eta\, g_i / \sqrt{\hat{v}_i} \approx \eta\,\mathrm{sign}(x_i)$:
constant-magnitude *sign descent*, where a diagonal Newton step would move by
$x_i$ itself. What survives of the opening argument is the *ratio* between
coordinates: the effective step $\eta/(\lambda_i |x_i|)$ scales like
$1/\lambda_i$ wherever the $|x_i|$ are comparable (as at the equal-coordinate
start of the race below), so stiff coordinates take small steps and flat ones
take large steps, estimated from first-order information alone.

### When Adam Fails to Converge

Adam's forgetting has a provable cost, found by
:citet:`Reddi.Kale.Kumar.2019` after the method had already trained thousands
of models: on some convex problems, Adam converges to the *worst* point.

**Proposition (Adam need not converge).** *Consider online optimization over
the interval $x \in [-1, 1]$ with the periodic sequence of convex losses*

$$
f_t(x) =
\begin{cases}
C\,x, & t \bmod 3 = 1, \\
-x, & \textrm{otherwise},
\end{cases}
\qquad C > 2,
$$
:eqlabel:`eq_mdl-opt-reddi`

*whose gradients over one period sum to $C - 2 > 0$, so the best fixed point
is $x^\star = -1$. Adam with $\beta_1 = 0$, $\beta_2 = 1/(1 + C^2)$, and any
step sequence $\eta_t = \eta/\sqrt{t}$ makes* positive *net progress each
period: its iterates converge to $x = +1$, where every third loss is
maximal, and its average regret does not vanish.*
:cite:`Reddi.Kale.Kumar.2019` *carry the computation to general
$\beta_1 < \sqrt{\beta_2}$ (every practical setting): for any fixed
exponential decay there is a $C$ that defeats it.*

The mechanism fits in one sentence: **the effective step on the informative
gradient shrinks faster than its information accrues.** When the rare large
gradient $C$ arrives, it lands in $\mathbf{v}$ *quadratically*: $v_t$
jumps to nearly $C^2$, so the very step that should exploit the information
is normalized by nearly $C$ and moves the iterate by only $O(\eta_t)$. Then
$\beta_2$ forgets: over the next two steps $v_t$ collapses back toward $1$,
and the two small $-1$ gradients each get full-sized steps in the *wrong*
direction. Signal is throttled by its own magnitude, noise is not, and the
mismatch compounds forever. This is a different failure from the noise ball
of :numref:`sec_mdl-gradient-based-optimization` (which shrinks with the step
size): the drift has a *sign*, and the iterate travels to the
worst boundary and stays. The one-line fix, **AMSGrad**
:cite:`Reddi.Kale.Kumar.2019`, replaces $\hat{\mathbf{v}}_t$ by the running
*maximum* $\tilde{\mathbf{v}}_t = \max(\tilde{\mathbf{v}}_{t-1}, \mathbf{v}_t)$,
which makes the per-coordinate step nonincreasing (the large gradient is
never forgotten) and restores the convergence guarantee. The cell runs the
construction with $C = 4$ against AMSGrad and plain projected SGD with the
same $1/\sqrt{t}$ decay:

```{.python .input #adaptive-stochastic-methods-reddi}
def online(method, C=4.0, T=15000, alpha=0.5):
    beta2 = 1 / (1 + C * C)                    # the proposition's beta_2
    x, v, vmax, xs = 0.0, 0.0, 0.0, []
    for t in range(1, T + 1):
        g = C if t % 3 == 1 else -1.0          # rare +C, frequent -1
        v = beta2 * v + (1 - beta2) * g * g
        vmax = max(vmax, v)
        if method == 'sgd':
            step = alpha / np.sqrt(t) * g / C
        else:                                  # adam (beta_1 = 0) or amsgrad
            vv = vmax if method == 'amsgrad' else v
            step = alpha / np.sqrt(t) * g / (np.sqrt(vv) + 1e-8)
        x = min(1.0, max(-1.0, x - step))      # project back onto [-1, 1]
        xs.append(x)
    return xs

print('        x_t at t =    30      300     3000    15000     (x* = -1)')
for method in ['adam', 'amsgrad', 'sgd']:
    xs = online(method)
    print(f'{method:8s}' + ''.join(f'{xs[t - 1]:9.3f}'
                                   for t in (30, 300, 3000, 15000)))
```

Adam marches to $+1$, the maximizer of the periodic loss, and is pinned
there by $t = 300$; AMSGrad and SGD settle onto the optimum $-1$. Nothing
about the problem is pathological from SGD's point of view (it is convex,
one-dimensional, with bounded gradients); the pathology is entirely in the
interaction between exponential forgetting and rare informative gradients. In
practice the construction's fingerprint, occasional large gradients that
matter, is not exotic: rare tokens, rare classes, and loss spikes all
qualify. Practice mostly retains plain Adam anyway (the drift needs the rare
gradients to be *consistently* informative in the same direction, which real
noise usually breaks up), but the theorem establishes the narrower conclusion that vanilla Adam lacks a
general convergence guarantee without additional assumptions or modification,
even for convex problems.

### The Valley, Revisited

The counterpart of that failure is what Adam does to the condition number,
and here the news is good. The cell returns to the ill-conditioned quadratic
of :numref:`sec_mdl-gradient-based-optimization` at $\kappa = 10^3$ and races
tuned gradient descent (the optimal single step
$\eta^\star = 2/(\lambda_{\min} + \lambda_{\max})$) against hand-rolled
Adam, printing Adam's effective per-coordinate steps
$\eta/(\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ as it runs:

```{.python .input #adaptive-stochastic-methods-adam-vs-gd}
lam = np.array([1.0, 1000.0])                  # kappa = 1000
f = lambda x: 0.5 * (lam * x * x).sum()

def race(optimizer, eta, K=20000, tol=1e-8):
    x, m, v, hit = np.array([1.0, 1.0]), np.zeros(2), np.zeros(2), None
    for k in range(1, K + 1):
        g = lam * x
        if optimizer == 'gd':
            x = x - eta * g
        else:                                  # adam, the update of the text
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            step = eta / (np.sqrt(v / (1 - 0.999**k)) + 1e-8)
            if k in (1, 100, 1000):
                print(f'   k = {k:4d}: effective steps {step.round(6)}')
            x = x - step * (m / (1 - 0.9**k))
        if hit is None and f(x) < tol:
            hit = k
    return hit

hit = race('gd', 2 / lam.sum())
print(f'GD, optimal single eta = {2 / lam.sum():.1e}: f < 1e-8 at k = {hit}')
print('Adam, eta = 0.01:')
print(f'Adam reaches f < 1e-8 at k = {race("adam", 0.01)}')
```

Gradient descent with the *optimal* single step needs $6160$ iterations, and
the $(\kappa-1)/(\kappa+1)$ contraction law of
:numref:`sec_mdl-gradient-based-optimization` predicts that count exactly:
with $\eta^\star$ both coordinates contract by $\rho = 999/1001$ per step, so
$f_k = 500.5\,\rho^{2k}$ crosses $10^{-8}$ at $k = 6159.07$, making $6160$
the first iterate under tolerance. Adam gets there in $344$. The printed
effective steps say why: from the
first iteration Adam's step on the stiff coordinate is $10^{-5}$ and on the
flat coordinate $10^{-2}$, the $1000\times$ ratio of the eigenvalues,
reconstructed from gradient magnitudes alone, with no eigendecomposition and
no Hessian. Both coordinates then travel toward zero at comparable speed
(this is the right panel of :numref:`fig_mdl-opt-per-coordinate`): the valley
has been rescaled into a bowl.

Now the counterpoint. On *this* problem, exact preconditioning is
available: Newton's method solves it in one step, and even a fixed
diagonal $\eta_i = 1/\lambda_i$ needs no adaptivity at all. What Adam offers
is an approximation to that oracle built *from noisy first-order data,
continuously, at $O(d)$ cost*. And the approximation is crude:
$\sqrt{\hat{\mathbf{v}}}$
conflates curvature with gradient noise (both inflate squared gradients), the
diagonal ignores every off-axis correlation (rotate this quadratic by
$45°$ and per-coordinate rescaling loses most of its effect), and the Reddi
construction shows the estimate can be steered adversarially. Adaptive
methods occupy a well-chosen point partway along a cost--fidelity curve; the
preconditioning ladder at the end of this section climbs further up it.

## Decoupled Weight Decay
:label:`subsec_mdl-decoupled-weight-decay`

### The Penalty Gradient Goes Through the Preconditioner

Weight decay looks too simple to interact with any of this: add
$\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss, get $\lambda \mathbf{w}$
added to the gradient. Under SGD the two implementations, "penalize the
loss" and "shrink the weights", are *identical*:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{w}_t)
= (1 - \eta\lambda)\, \mathbf{w}_t - \eta\, \mathbf{g}_t .
$$

Under Adam they are not, and the two-line derivation is the whole story.
Feed the penalized gradient through :eqref:`eq_mdl-opt-adam` and look at the
decay term alone (take $\beta_1 = 0$ for clarity, so the update is
$-\eta\,(\mathbf{g}_t + \lambda\mathbf{w}_t)/(\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$):
the shrinkage applied to coordinate $i$ is

$$
\textrm{$\ell_2$ through Adam:}\quad
\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; w_{t,i}
\qquad \textrm{versus} \qquad
\textrm{decoupled:}\quad \eta\,\lambda\, w_{t,i} .
$$
:eqlabel:`eq_mdl-opt-adamw`

The penalty's gradient is divided by the same $\sqrt{\hat{\mathbf{v}}}$ as
everything else, so the *regularization strength is no longer a constant of
the problem*: it varies per coordinate and over time, inversely with the
gradient scale. A weight whose loss gradients are large or noisy (large
$\hat{v}_i$) is barely regularized at all; a weight whose gradients have gone
quiet is regularized hard. Whatever $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ was
supposed to mean (the Gaussian prior of the MAP estimate in
:numref:`sec_mdl-maximum_likelihood`, or the norm constraint
$\|\mathbf{w}\|^2 \le r^2$ whose multiplier it is in
:numref:`subsec_mdl-weight-decay-duality`), that meaning assumed one
$\lambda$ for all coordinates, and Adam's preconditioner silently discards it.

**AdamW** :cite:`Loshchilov.Hutter.2019` restores the intended semantics by
*decoupling*: the loss gradient goes through the preconditioner, the decay
does not,

$$
\mathbf{w}_{t+1} = (1 - \eta\lambda)\, \mathbf{w}_t - \eta\; \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
$$

which is verbatim the SGD shrinkage $(1 - \eta\lambda)$, applied uniformly.
The empirical finding that motivated the fix ($\ell_2$-regularized Adam
generalizing worse than SGD with weight decay, with the gap closed by
decoupling) is downstream of exactly the per-coordinate distortion in
:eqref:`eq_mdl-opt-adamw`. The cell isolates that distortion: two weights
whose loss gradients are pure noise (so decay is the only systematic force)
at very different noise scales, $\sigma = (10, 0.1)$.

```{.python .input #adaptive-stochastic-methods-adamw}
sig, alpha, lam_wd, T = np.array([10.0, 0.1]), 1e-3, 0.1, 4000

def decay_race(decoupled, seed=0):
    rg = np.random.default_rng(seed)
    w, m, v = np.array([1.0, 1.0]), np.zeros(2), np.zeros(2)
    for t in range(1, T + 1):
        g = sig * rg.standard_normal(2)        # loss gradient: pure noise
        if not decoupled:
            g = g + lam_wd * w                 # l2 penalty inside the gradient
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        w = w - alpha * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        if decoupled:
            w = w - alpha * lam_wd * w         # decay outside the preconditioner
    return w

print('per-step decay rate, l2 through Adam (alpha*lam/sigma_i):',
      (alpha * lam_wd / sig).round(6))
print('per-step decay rate, AdamW (uniform):                    ',
      alpha * lam_wd)
print(f'|w| after {T} steps,  Adam + l2: {np.abs(decay_race(False)).round(4)}')
print(f'|w| after {T} steps,  AdamW    : {np.abs(decay_race(True)).round(4)}')
print(f'AdamW prediction (1 - alpha*lam)^T = {(1 - alpha * lam_wd)**T:.4f}')
```

Under $\ell_2$-through-Adam the noisy coordinate has decayed from $1$ to only
$0.968$ after $4000$ steps while the quiet coordinate has collapsed to
$0.024$: effective decay rates of $10^{-5}$ versus $10^{-3}$, the
predicted $\eta\lambda/\sigma_i$, a $100\times$ disparity that nobody chose.
AdamW shrinks both to $\approx 0.66$, within $2\%$ of the pure-decay
prediction $(1 - \eta\lambda)^T = 0.6703$: one $\lambda$ with one meaning.
(A subtlety about fixed points: at a *deterministic* stationary point,
coupled Adam is stationary exactly where SGD with $\ell_2$ is, at
$\nabla L + \lambda\mathbf{w} = \mathbf{0}$, while *AdamW's* stationary
points sit elsewhere; Exercise 4 derives both conditions and explains why the
noisy, never-stationary training regime, where paths matter more than fixed
points, is where decoupling matters.) In the major libraries, `AdamW` and
`Adam` with a `weight_decay` flag implement two different regularizers: the
former is the decoupled update exactly, the latter the coupled
:eqref:`eq_mdl-opt-adamw`.

## Schedules and Warmup
:label:`subsec_mdl-schedules-warmup`

### What Decay Does, and Which Shape

A learning-rate schedule is a sequence $\eta_t$ (the main book surveys the
common ones in :numref:`sec_scheduler`), and the mathematics it must
negotiate was already proved in
:numref:`sec_mdl-gradient-based-optimization`: a constant step parks SGD on a
noise floor proportional to $\eta$ :eqref:`eq_mdl-opt-noise-ball`, decay must
respect Robbins--Monro :eqref:`eq_mdl-opt-robbins-monro` to reach the optimum,
and the textbook $\eta_k = c/k$ carries the trap that undersized $c$ silently
degrades the rate class.

Add to this the Ghadimi--Lan prescription of this section: for a fixed
nonconvex budget $K$, a *constant* $\eta \propto 1/\sqrt{K}$ is already
**minimax-optimal**, meaning no method that sees only the same
unbiased, bounded-variance gradient oracle can guarantee a better worst-case
rate than the $K^{-1/2}$ it achieves; the matching lower bound is due to
:citet:`Arjevani.Carmon.Duchi.ea.2023`. The theory therefore assigns decay a
specific job: it lowers the noise floor, at a cost in transient progress, and
beyond convex problems no theorem ranks one decay *shape* against another.
Which of the statements below are proved and which are read off practice is
marked as we go. The three shapes to know, drawn in
:numref:`fig_mdl-opt-schedule-zoo`, are these:

![The schedule zoo, at equal budget. A constant step never leaves its noise floor (the floor is $\propto \eta$, per the noise-ball analysis); $c/k$ decay reaches the optimum but must get $c$ right; cosine spends a long tail at small steps; warmup--stable--decay (WSD) holds the constant plateau as long as possible and drops the noise floor in a final decay phase. All four begin with the linear warmup that adaptive preconditioners need while $\hat{\mathbf{v}}$ is still estimated from a handful of gradients.](../img/mdl-opt-schedule-zoo.svg)
:label:`fig_mdl-opt-schedule-zoo`

**Cosine decay** :cite:`Loshchilov.Hutter.2016` sets
$\eta_t = \tfrac12 \eta_0 (1 + \cos(\pi t / K))$: a smooth descent from
$\eta_0$ to $0$ with most of its time spent at moderate steps and a long,
gentle tail. It satisfies no optimality theorem; it is popular because it has
one parameter ($\eta_0$), no kinks to tune, and behaves gracefully across
budgets. **Warmup--stable--decay (WSD)** :cite:`Hu.Tu.Han.ea.2024` holds
$\eta_0$ constant for most of the run and decays only in a final fraction
(often the last $10$--$20\%$). Its practical appeal is operational: the
long constant plateau means checkpoints from one run can be re-decayed to any
budget, rather than committing to $K$ at launch. Its loss-landscape
reading is the river-valley picture: during the plateau the iterate travels
fast *along* the valley while rattling in its noise ball across it, and the
final decay shrinks the ball, dropping the loss sharply as the iterate
settles to the valley floor. That reading is this chapter's noise-ball
result used as a design principle, and it is the level
of theory available: the plateau's benefit on real losses (more transient
progress per unit budget) is an empirical regularity, not a theorem. The cell
races the shapes on the noisy quadratic, the miniature where the noise
floor is exact, at equal budget, printing the loss at $80\%$ of the budget
and at the end:

```{.python .input #adaptive-stochastic-methods-schedules}
lam_s, sigma, K, eta0 = 1.0, 1.0, 2000, 0.3

def run_schedule(eta_fn, seeds=200):
    at80, final = [], []
    for s in range(seeds):
        rg, x = np.random.default_rng(s), 3.0
        for k in range(K):
            x = x - eta_fn(k) * (lam_s * x + sigma * rg.standard_normal())
            if k == int(0.8 * K) - 1:
                at80.append(0.5 * lam_s * x * x)
        final.append(0.5 * lam_s * x * x)
    return np.mean(at80), np.mean(final)

schedules = [
    ('constant', lambda k: eta0),
    ('c/k decay', lambda k: eta0 / (1 + k / 50)),
    ('cosine', lambda k: 0.5 * eta0 * (1 + np.cos(np.pi * k / K))),
    ('WSD 80/20', lambda k: eta0 if k < 0.8 * K else eta0 * (K - k) / (0.2 * K)),
]
print(f'         E f at 80% budget   E f at end     (floor ~ '
      f'{eta0 * sigma**2 / (2 * (2 - eta0 * lam_s)):.3f})')
for name, fn in schedules:
    m80, mK = run_schedule(fn)
    print(f'{name:10s}    {m80:9.2e}      {mK:9.2e}')
```

The constant run sits on its floor at both checkpoints ($9.5$ and
$9.2 \times 10^{-2}$, against the printed prediction
$\eta\sigma^2/(2(2 - \eta\lambda)) \approx 0.088$), while every decaying
schedule ends one to two orders lower. The instructive row is WSD: at $80\%$
of the budget it is *still on the constant's floor* ($9.5 \times 10^{-2}$,
equal to the constant row to every printed digit, since the two runs are the
same trajectory until the decay begins), then the final $400$ steps drop it
by a factor of $17$. All of WSD's visible gain
arrives in the decay phase: the loss-curve cliff that WSD training runs
are known for, reproduced here by a one-line quadratic. On this pure
quadratic, cosine's longer tail of small steps wins the endgame
($1.6 \times 10^{-3}$); the reasons to prefer WSD (re-decayable
checkpoints, more time at large steps on losses whose transient is the
bottleneck) live off this toy, in exactly the gap between the noise-ball
theorem and practice.

### Warmup: Do Not Trust an Estimated Preconditioner Cold

Every schedule above begins with **warmup**: $\eta_t$ ramped linearly from
$\approx 0$ over the first fraction of training
:cite:`Goyal.Dollar.Girshick.ea.2017`. For SGD, warmup is mild prudence
(early gradients are large and the landscape unexplored). For adaptive
methods it is close to structural, and this section has already assembled
both reasons.

First, the preconditioner starts out *statistically worthless*. At $t = 1$,
$\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$: the per-coordinate rescaling is
estimated from a single sample, and bias correction makes it *unbiased*, not
*accurate*. The step direction
$\hat{\mathbf{m}}_1/(\sqrt{\hat{\mathbf{v}}_1} + \epsilon) = \mathrm{sign}(\mathbf{g}_1)$
commits to full-magnitude movement on every coordinate, including ones whose
lone observed gradient was pure noise. The estimate needs on the order of
$1/(1 - \beta_2) = 1000$ steps to average over a meaningful window, and
until then Adam is applying its most aggressive, least justified rescaling.
Ramping $\eta$ over that same window is the obvious hedge: keep the steps
small until the preconditioner has data to justify them. (This diagnosis can
also be read off :numref:`fig_mdl-opt-bias-correction`: the correction fixes
the *mean* of the early estimate; nothing fixes its variance but data.)

Second, the target step size is only safe *after* the landscape has adapted
to it. The edge-of-stability measurement in
:numref:`sec_mdl-gradient-based-optimization` showed sharpness equilibrating
onto $2/\eta$, but that equilibration takes time, and a full-size $\eta$
imposed at initialization, before progressive sharpening has anywhere to
settle, simply violates the local stability ceiling and diverges or spikes.
Warmup gives the curvature time to meet the step size it will be asked to
tolerate. Neither argument is a theorem about deep networks; both are this
chapter's instruments (estimator variance and the stability ceiling)
pointed at the first thousand steps, and together they predict the practice
that every large training run has converged on independently.

## Variance Reduction for Finite Sums
:label:`sec_mdl-variance-reduction`

Minibatching reduces variance by averaging more examples, but a fixed-size
minibatch still has noise at the optimum. A finite training set offers another
possibility: reuse information from earlier gradients as a **control variate**.
Write the empirical objective as

$$
F(\mathbf x)=\frac1n\sum_{i=1}^n f_i(\mathbf x).
$$

At a reference point $\widetilde{\mathbf x}$, compute the full gradient
$\widetilde{\boldsymbol\mu}=\nabla F(\widetilde{\mathbf x})$. The **SVRG**
estimator at the current point is

$$
\mathbf v_i(\mathbf x)
=\nabla f_i(\mathbf x)-\nabla f_i(\widetilde{\mathbf x})
 +\widetilde{\boldsymbol\mu}.
$$
:eqlabel:`eq_mdl-opt-svrg`

For a uniformly sampled index,

$$
\mathbb E_i[\mathbf v_i(\mathbf x)]
=\nabla F(\mathbf x)-\nabla F(\widetilde{\mathbf x})
 +\nabla F(\widetilde{\mathbf x})
=\nabla F(\mathbf x),
$$

so the estimator is unbiased. Its advantage is not the mean but the variance:
when $\mathbf x$ is close to the snapshot, the two component gradients are
highly correlated and their difference is much less variable than either one.
After an inner loop moves too far from the snapshot, refresh the full gradient
and repeat. Under smooth strong convexity, this mechanism yields geometric
convergence with a fixed step size, rather than SGD's fixed-step noise floor
:cite:`Bottou.Curtis.Nocedal.2018`.

The cell compares SVRG with SGD on ridge regression. One SVRG epoch pays $n$
component-gradient equivalents for the full gradient and two per inner step;
SGD receives the same total budget of $3n$ component gradients per reported
epoch. Both use a fixed step size, so the comparison isolates the control
variate rather than a schedule.

```{.python .input #mdl-adaptive-svrg}
rng = np.random.default_rng(4)
n_vr, d_vr = 300, 12
X_vr = rng.standard_normal((n_vr, d_vr))
w_true_vr = rng.standard_normal(d_vr)
y_vr = X_vr @ w_true_vr + 0.5 * rng.standard_normal(n_vr)
lam_vr = 0.1
H_vr = X_vr.T @ X_vr / n_vr + lam_vr * np.eye(d_vr)
w_star_vr = np.linalg.solve(H_vr, X_vr.T @ y_vr / n_vr)

def component_grad(w, i):
    return X_vr[i] * (X_vr[i] @ w - y_vr[i]) + lam_vr * w

def full_grad(w):
    return X_vr.T @ (X_vr @ w - y_vr) / n_vr + lam_vr * w

def objective_vr(w):
    return (0.5 * np.mean((X_vr @ w - y_vr)**2)
            + 0.5 * lam_vr * (w @ w))

eta_vr, epochs_vr = 0.05, 20
snapshot, gaps_svrg = np.zeros(d_vr), []
for _ in range(epochs_vr):
    mean_grad = full_grad(snapshot)
    w = snapshot.copy()
    for _ in range(n_vr):
        i = rng.integers(n_vr)
        v = component_grad(w, i) - component_grad(snapshot, i) + mean_grad
        w -= eta_vr * v
    snapshot = w
    gaps_svrg.append(objective_vr(snapshot) - objective_vr(w_star_vr))

rng_sgd = np.random.default_rng(5)
w_sgd, gaps_sgd = np.zeros(d_vr), []
for _ in range(epochs_vr):
    for _ in range(3 * n_vr):               # same component-gradient budget
        i = rng_sgd.integers(n_vr)
        w_sgd -= eta_vr * component_grad(w_sgd, i)
    gaps_sgd.append(objective_vr(w_sgd) - objective_vr(w_star_vr))

d2l.plot(np.arange(1, epochs_vr + 1), [gaps_sgd, gaps_svrg],
         'equal-budget epoch', 'optimality gap',
         legend=['SGD, fixed step', 'SVRG, fixed step'], yscale='log')
```

SGD makes cheaper early progress, but its constant-step iterates continue to
wander around the optimum. SVRG pays for its snapshots and then drives the gap
down geometrically on this finite, strongly convex problem. The result does not
say SVRG universally beats SGD: full passes are expensive on enormous or
streaming datasets, the linear-rate theorem uses finite-sum smoothness and
convexity, and stale control variates are less effective when the objective
changes during training.

**SAGA** removes the periodic full-gradient pass by storing the most recently
seen gradient for every example and using their table average as the control
variate. It spends one new component gradient per step but stores $O(nd)$
numbers in the literal form (structure can reduce this for generalized linear
models). SARAH-style estimators update the control variate recursively. The
shared principle is more important than the acronyms: correlate a noisy new
estimate with a stored estimate whose mean is known, subtract the shared noise,
and add the known mean back.

Variance reduction is distinct from momentum. Momentum averages gradients
across recent iterates and can reduce high-frequency noise, but its moving
average is generally biased for the gradient at the current point. SVRG and
SAGA engineer an estimator that is unbiased at the current point and whose
variance shrinks as the optimization converges.

## Beyond Diagonals: the Preconditioning Ladder
:label:`subsec_mdl-preconditioning-ladder`

Step back and every method in this section is one answer to a single design
question: *which matrix $B_t$ should multiply the gradient*,
$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, B_t\, \mathbf{g}_t$? Gradient
descent answers $B = I$; Newton answers $B = (\nabla^2 f)^{-1}$ at $O(d^3)$;
Adam answers with a diagonal estimated from gradient history at $O(d)$.
Between the diagonal and the full inverse Hessian runs a ladder of structured
compromises, and 2020s practice has climbed it rung by rung. What makes the
climb affordable is one structural fact the "one long vector
$\mathbf{x} \in \mathbb{R}^d$" notation hides: a network's parameters come in
*layers of matrices*, and curvature between layers can be ignored while
curvature within a layer is captured cheaply.

**Kronecker-factored curvature (K-FAC)** :cite:`Martens.Grosse.2015` treats
each layer's weight matrix $W \in \mathbb{R}^{m \times n}$ separately and
approximates its block of the **Fisher information matrix**, the
outer-product curvature
$\mathbb{E}[\nabla \log p\, \nabla \log p^\top]$ that
:numref:`sec_mdl-maximum_likelihood` introduced. The word *curvature* is
precise here: for log-likelihood losses the Fisher equals the Gauss--Newton matrix,
the PSD part of the Hessian obtained by dropping the second-derivative terms
of the network :cite:`Martens.Grosse.2015`. K-FAC's approximation is a
Kronecker
product $A \otimes G$ (the block matrix whose $(i,j)$ block is $a_{ij}\, G$),
where $A = \mathbb{E}[\mathbf{a}\mathbf{a}^\top]$ is
the second moment of the layer's *inputs* and
$G = \mathbb{E}[\boldsymbol{\delta}\boldsymbol{\delta}^\top]$ that of the
gradients at its *outputs*. The factorization is exact if the layer's inputs
and output-gradients are statistically independent under the model's
distribution; that independence, which holds only approximately in trained
networks, is the approximation. The payoff is the
Kronecker inverse identity: writing $\mathrm{vec}(V)$ for the vector obtained
by stacking the columns of the layer's gradient matrix $V$,

$$
(A \otimes G)^{-1}\, \mathrm{vec}(V) \;=\; \mathrm{vec}\left(G^{-1}\, V\, A^{-1}\right),
$$
:eqlabel:`eq_mdl-opt-kronecker`

so preconditioning by a curvature matrix with $(mn)^2$ entries costs two
small inverses, of sizes $n \times n$ and $m \times m$: the whole trick in
one display. Preconditioning by the Fisher has its own name and pedigree:
it is the **natural gradient** :cite:`Amari.1998`, steepest descent when
distance is measured
between the *distributions* the parameters define rather than between
parameter vectors, the metric view of AdaGrad again, with the Fisher as
the metric. **Shampoo** :cite:`Gupta.Koren.Singer.2018` keeps the same
two-sided structure but builds the factors AdaGrad-style from accumulated
gradient statistics, preconditioning $V \mapsto L^{-1/4} V R^{-1/4}$ with
$L = \sum_t V_t V_t^\top$ and $R = \sum_t V_t^\top V_t$, a two-sided matrix
cousin of :eqref:`eq_mdl-opt-adagrad`. It is a cousin rather than a strict
generalization: restricting $L, R$ to their diagonals normalizes entry
$(i,j)$ by the factored $(l_i r_j)^{1/4}$ of its row and column statistics,
not by AdaGrad's per-entry root, and the precise relation is an inequality
(the Kronecker root $L^{1/4} \otimes R^{1/4}$ upper-bounds the full-matrix
AdaGrad preconditioner :cite:`Gupta.Koren.Singer.2018`). **Muon**
:cite:`Jordan.Jin.Boza.ea.2024` takes the limiting view: rather than
estimate second moments at all, it replaces each layer's momentum matrix by
its nearest orthogonal matrix, the polar factor $UV^\top$ of the SVD,
computed by the Newton--Schulz iteration demonstrated in
:numref:`sec_mdl-svd-low-rank`. This whitens the update's spectrum
directly: every singular direction of the update moves at the same rate,
per-*direction* step equalization where Adam manages only per-coordinate.

The ladder, assembled: diagonal (Adam, $O(d)$) $\to$ per-layer Kronecker
factors (K-FAC) $\to$ per-layer full-matrix roots (Shampoo) $\to$ per-layer
spectral normalization (Muon) $\to$ full second order (Newton, unreachable).
Each rung captures geometry the previous one ignored and costs more per-step
arithmetic; where training runs land on the ladder is set by hardware
economics as much as mathematics, which is why the frontier keeps moving.
The same matching of step size to geometry also extends *across model
sizes*: the maximal-update parametrization (muP) :cite:`Yang.Hu.2021` derives
how per-layer learning rates must scale with width so that one tuned
$\eta$ transfers from a small model to a large one.

## Summary

* The **Ghadimi--Lan theorem** extends the no-convexity guarantee to
  SGD: with $\eta \le 1/L$ the descent lemma survives in expectation with a
  noise tax $\tfrac{L}{2}\eta^2\sigma^2$ per step, and at the balanced
  $\eta \propto 1/\sqrt{K}$ a random iterate satisfies
  $\mathbb{E}\|\nabla f\|^2 = O(\sigma\sqrt{L\Delta/K})$. Noise turns the
  deterministic $1/K$ rate into $K^{-1/2}$, and the
  guarantee is for a *random* iterate (the best one cannot be identified
  from noisy evaluations).
* **Per-coordinate step sizes** attack $\kappa$ directly: on a diagonal
  quadratic, $\eta_i = 1/\lambda_i$ removes the condition number entirely.
  **AdaGrad** is steepest descent in the metric
  $\mathrm{diag}(\sqrt{\sum_t \mathbf{g}_t^2})$, built for sparse
  gradients, but its never-forgetting denominator can drive steps toward zero
  and make progress impractically slow on persistent noisy or nonconvex
  problems; **RMSProp** substitutes an EMA; **Adam** adds momentum
  and *exact* bias correction: $\mathbb{E}[\mathbf{v}_t] =
  (1 - \beta_2^t)\,\bar{\mathbf{g}^2}$ under stationarity, so dividing by
  $1 - \beta_2^t$ cancels the zero-initialization deficit identically.
* Vanilla Adam is **not guaranteed to converge in general**, even on convex
  problems: in the Reddi--Kale--Kumar construction the effective step on a rare informative gradient shrinks
  (by $\sqrt{\mathbf{v}}\approx C$) faster than its information accrues, and
  Adam converges to the *worst* point while AMSGrad's running-max fix (a
  nonincreasing per-coordinate step) and plain SGD find the optimum. What
  adaptivity wins back is the valley: on the $\kappa = 10^3$ quadratic Adam
  reconstructs the $1000\times$ eigenvalue ratio from gradient magnitudes
  alone and beats optimally tuned GD by $18\times$.
* **Weight decay through Adam's preconditioner is not weight decay**: the
  shrinkage becomes $\eta\lambda/(\sqrt{\hat{v}_i} + \epsilon)$ per
  coordinate, strong where gradients are quiet, negligible where they are
  large or noisy. **AdamW** decouples the decay from the preconditioner,
  restoring the uniform $(1 - \eta\lambda)$ that the MAP-prior and
  norm-constraint readings of $\lambda$ presuppose.
* **Schedules** negotiate the noise floor ($\propto \eta$) against transient
  progress; beyond convexity no theorem ranks decay shapes. Cosine spends a
  long small-step tail; **WSD** stays on the constant-step floor until a
  final decay phase drops it (the loss cliff, reproduced on a noisy
  quadratic). **Warmup** is the adaptive family's structural need: at small
  $t$ the preconditioner is estimated from a handful of gradients (unbiased
  $\ne$ accurate), and the stability ceiling has not yet adapted to the
  target step.
* On finite-sum objectives, **SVRG** subtracts a component gradient at a
  snapshot and adds the snapshot's full gradient, producing an unbiased
  control-variate estimator whose variance shrinks near the snapshot. **SAGA**
  replaces full-gradient refreshes by a table of stored component gradients.
  These methods can reach linear fixed-step convergence under smooth strong
  convexity, but require repeated access to a finite dataset.
* Between Adam's diagonal and Newton's full inverse runs the
  **preconditioning ladder**: K-FAC's Kronecker-factored Fisher (natural
  gradient at two small inverses via
  $(A \otimes G)^{-1}\mathrm{vec}(V) = \mathrm{vec}(G^{-1}VA^{-1})$),
  Shampoo's two-sided full-matrix roots, Muon's Newton--Schulz
  orthogonalization of momentum, each rung exploiting the fact that
  parameters come in matrices.

## Exercises

1. **AdaGrad from the metric view.** Exercise 2 of
   :numref:`sec_mdl-gradient-based-optimization` showed that the steepest
   descent direction under the norm
   $\|\mathbf{d}\|_A = \sqrt{\mathbf{d}^\top A \mathbf{d}}$ is proportional
   to $-A^{-1}\nabla f$. Set
   $A_t = \mathrm{diag}\bigl(\sqrt{\mathbf{s}_t}\bigr) + \epsilon I$ with
   $\mathbf{s}_t = \sum_{s \le t} \mathbf{g}_s^2$ and recover
   :eqref:`eq_mdl-opt-adagrad`. Why is $A_t$ built from
   $\sqrt{\mathbf{s}_t}$ rather than $\mathbf{s}_t$? (Two answers: match the
   units of a curvature, since squared gradients have the wrong dimensions,
   and check which choice makes the effective step on a coordinate with
   i.i.d. noise decay like the Robbins--Monro-compatible $1/\sqrt{t}$
   rather than the too-fast $1/t$.)
2. **Bias correction for the first moment.** Repeat the computation of
   :eqref:`eq_mdl-opt-bias-correction` for
   $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$ under
   $\mathbb{E}[\mathbf{g}_s] = \bar{\mathbf{g}}$, obtaining
   $\mathbb{E}[\mathbf{m}_t] = (1 - \beta_1^t)\,\bar{\mathbf{g}}$. Then
   compute the scale of *uncorrected* Adam's first step: with
   $\mathbf{m}_1 = (1-\beta_1)\mathbf{g}_1$ and
   $\mathbf{v}_1 = (1-\beta_2)\mathbf{g}_1^2$, show the raw ratio
   $\mathbf{m}_1/\sqrt{\mathbf{v}_1}$ has magnitude
   $(1-\beta_1)/\sqrt{1-\beta_2} \approx 3.16$ at the defaults. Finally,
   find (numerically or by calculus) where the inflation factor
   $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$ *peaks* and how large it gets,
   and explain, from the two averaging timescales $1/(1-\beta_1)$ and
   $1/(1-\beta_2)$, why the peak sits where it does.
3. **The Reddi example by hand.** Take $C = 4$, $\beta_1 = 0$,
   $\beta_2 = 1/17$, a fixed step $\eta_t = \eta$, and an iterate $x$ in the
   interior of $[-1, 1]$ with $v$ at its period-limit value. Compute the
   three Adam updates of one period of :eqref:`eq_mdl-opt-reddi` and show
   the net displacement over the period is positive (toward the wrong
   endpoint) for small $\eta$. Where exactly does the factor
   $\sqrt{v} \approx C$ throttle the informative step?
4. **Coupled versus decoupled fixed points.** For
   $L(\mathbf{w}) = \tfrac12 (\mathbf{w} - \mathbf{w}_0)^\top
   \mathrm{diag}(h_1, h_2) (\mathbf{w} - \mathbf{w}_0)$ with
   $h = (100, 1)$ and deterministic gradients, write the stationarity
   conditions of (a) Adam on $L + \tfrac{\lambda}{2}\|\mathbf{w}\|^2$ and
   (b) AdamW on $L$. Show that (a) demands
   $\nabla L + \lambda\mathbf{w} = \mathbf{0}$ (the *same* point as SGD
   with $\ell_2$), while (b) demands
   $\nabla L_i = -\lambda w_i (\sqrt{\hat{v}_i} + \epsilon)$, a
   *different* point. Reconcile this with the section's cell: in the noisy,
   never-stationary regime, which of the two updates applies the uniform
   per-step shrinkage, and why is that the property that matters during
   training?
5. **Where $\epsilon$ matters.** On the $\kappa = 10^3$ quadratic of the
   `#adaptive-stochastic-methods-adam-vs-gd` cell, sweep
   $\epsilon \in \{10^{-12}, 10^{-8}, 10^{-4}, 10^{-2}, 10^{-1}\}$ and
   record iterations to $f < 10^{-8}$. Explain the two regimes you find:
   $\epsilon$ is invisible while $\sqrt{\hat{v}_i} \gg \epsilon$, and it
   converts Adam into plain (momentum) SGD with step $\eta/\epsilon$ once
   gradients shrink below it. Which coordinate's gradients cross that
   threshold first, and what does $\epsilon$ therefore control near
   convergence?
6. **The optimal constant step, exactly.** For the noisy scalar quadratic of
   the `#adaptive-stochastic-methods-schedules` cell, iterate the noise-ball
   recursion of :numref:`sec_mdl-gradient-based-optimization` to get the
   exact finite-horizon loss
   $\mathbb{E}[x_K^2] = (1-\eta\lambda)^{2K} x_0^2 +
   \bigl(1 - (1-\eta\lambda)^{2K}\bigr)\, \eta\sigma^2/(\lambda(2-\eta\lambda))$.
   For a fixed budget $K$, show the minimizing constant step scales like
   $\eta^\star = \Theta(\log K / K)$ (balance the exponential transient
   against the linear-in-$\eta$ floor), *not* like the $1/\sqrt{K}$ of the
   Ghadimi--Lan prescription, and explain why there is no contradiction
   (strong convexity versus the nonconvex worst case). Verify your
   $\eta^\star$ numerically against a grid search at $K = 2000$.
7. **AMSGrad's monotone steps.** Show that with the running maximum
   $\tilde{\mathbf{v}}_t = \max(\tilde{\mathbf{v}}_{t-1}, \mathbf{v}_t)$ the
   effective per-coordinate step $\eta_t/\sqrt{\tilde{v}_{t,i}}$ is
   nonincreasing in $t$ whenever $\eta_t$ is, and explain in one sentence
   why this property blocks the drift mechanism of
   :eqref:`eq_mdl-opt-reddi`. Then modify the
   `#adaptive-stochastic-methods-reddi` cell to *interpolate*: replace the
   hard max by $\tilde{v}_t = \max(\gamma\tilde{v}_{t-1}, v_t)$ and find the
   largest forgetting factor $\gamma < 1$ at which the drift to $+1$
   reappears.

8. **SVRG unbiasedness and cost.** Prove the conditional unbiasedness of
   :eqref:`eq_mdl-opt-svrg`. Count component-gradient evaluations in one epoch
   of the cell, then vary the inner-loop length from $n/2$ to $4n$. Explain the
   tradeoff between amortizing the snapshot gradient and letting the control
   variate become stale.

## Discussions

This section is the bridge between this chapter's theory and the optimizer
actually running in your training loop. The main book shows the methods in
action (:numref:`sec_sgd` and :numref:`sec_minibatch_sgd` for stochastic
descent, :numref:`sec_momentum` for velocity, :numref:`sec_adam` for the
adaptive family, :numref:`sec_scheduler` for schedules), while the
Ghadimi--Lan rate, the diagonal-metric derivation, the bias-correction
identity, the Reddi counterexample, and the decoupling algebra proved here
are the mathematics those recipes stand on. Within this part,
:numref:`sec_mdl-gradient-based-optimization` supplied every instrument
(descent lemma, $\kappa$, minibatch variance, noise ball);
:numref:`sec_mdl-convexity` explains which guarantees return when convexity
does; and the stationary-distribution view of constant-step SGD, the noise
ball as an invariant measure, is developed properly with the SDE machinery
of :numref:`chap_mdl-dynamics`.

[Discussions](https://d2l.discourse.group/t/adaptive-stochastic-methods)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §24.2]{.kicker}

From the theory to the default optimizer<br>**SGD without convexity · AdaGrad to AdamW · schedules, warmup, and the preconditioning ladder**.
:::
:::

::: {.slide title="Two questions the theory left open"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Every modern network trains with AdamW, warmup, and a decaying schedule,
not the plain gradient descent we analyzed. Two gaps to close:

- The one guarantee deep nets keep was proved for the **exact** gradient;
  training uses noisy minibatch estimates. *What survives?*
- The whole $\kappa$ story chained a **single** $\eta$ to the stiffest
  mode. *What if every coordinate had its own?*

::: {.d2l-note}
Every optimizer below is under ten lines of NumPy, written by hand.
:::
:::

::: {.col .fig}
@fig:mdl-opt-per-coordinate
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[SGD without convexity]{.dtitle}

[the descent lemma survives noise, taxed]{.dsub}
:::
:::

::: {.slide title="The descent lemma pays a noise tax"}
[Ghadimi--Lan]{.kicker}

Feed an unbiased gradient with variance $\sigma^2$ through the quadratic
ceiling and take expectations at the right moment:

$$\mathbb{E}\left[f(\mathbf{x}_{k+1}) \mid \mathbf{x}_k\right]
\;\le\; f(\mathbf{x}_k) - \eta\bigl(1 - \tfrac{L\eta}{2}\bigr)\,\|\nabla f(\mathbf{x}_k)\|^2
+ \tfrac{L\eta^2\sigma^2}{2}.$$

One new term in the descent lemma: a **noise tax** paid every step,
gradient large or not.

. . .

Telescoping (Ghadimi--Lan, 2013) and balancing the two terms with
$\eta \propto 1/\sqrt{K}$:

$$\mathbb{E}\bigl[\|\nabla f(\mathbf{x}_R)\|^2\bigr]
\;\le\; \frac{2L\Delta}{K} + 2\sigma\sqrt{\frac{2L\Delta}{K}},
\qquad R \sim \mathrm{Uniform}\{0, \ldots, K-1\}.$$
:::

::: {.slide title="Noise turns 1/K into a square root"}
[Ghadimi--Lan]{.kicker}

At $\sigma = 0$ the deterministic $O(1/K)$ returns; with noise the
$K^{-1/2}$ term rules: $10\times$ smaller gradients cost $100\times$
the budget. Measured on a nonconvex toy, $20$ seeds per budget:

@!adaptive-stochastic-methods-ghadimi-lan

::: {.d2l-note}
The guarantee is for a **randomly selected** iterate: with noisy
evaluations you can never *identify* the best one, and the min-so-far
column (slope $-1.23$) is exactly the luck you cannot bank on.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Per-coordinate step sizes]{.dtitle}

[AdaGrad → RMSProp → Adam]{.dsub}
:::
:::

::: {.slide title="One number, doing a matrix's job"}
[The thesis]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
On a diagonal quadratic, per-coordinate steps $\eta_i = 1/\lambda_i$
solve the problem in **one step**: a diagonal Newton's method, and
$\kappa$ simply disappears.

The zig-zag comes from using one number where $n$ are called for.
Adaptive methods estimate those $n$ numbers *from the gradients
themselves*.
:::

::: {.col .fig .big}
@fig:mdl-opt-per-coordinate
:::
:::
:::

::: {.slide title="AdaGrad: steepest descent in a learned metric"}
[The family]{.kicker}

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\,\mathbf{g}_t.$$

Two derivations, one rule: steepest descent in the metric
$\mathrm{diag}(\sqrt{\mathbf{s}_t})$ (a coordinate is expensive in
proportion to the evidence its gradients have been large) and the
regret-optimal step for **sparse** features: a rare word's step decays
with *its own* activity, not wall-clock time.

. . .

::: {.d2l-note .warn}
$\mathbf{s}_t$ never forgets, so steps decay like $\eta/(\sigma\sqrt{t})$:
Robbins--Monro hard-wired in. On some nonconvex problems the resulting decay can make progress
impractically slow because the accumulator remembers too much. **RMSProp**
forgets on purpose: an EMA with memory $\approx 1/(1-\beta_2)$ steps
($10$ at its standard $\beta_2 = 0.9$).
:::
:::

::: {.slide title="Bias correction is a two-line theorem"}
[The family]{.kicker}

::: {.cols .vc}
::: {.col}
Unroll $\mathbf{v}_t = (1-\beta_2)\sum_{s\le t} \beta_2^{\,t-s}\mathbf{g}_s^2$
and take expectations under a stationary scale $\bar{\mathbf{g}^2}$:

$$\mathbb{E}[\mathbf{v}_t] = \left(1 - \beta_2^{\,t}\right)\bar{\mathbf{g}^2}$$

by the geometric series. Dividing by $1-\beta_2^t$ is *exactly* unbiased
at every $t$: the correction cancels the zero initialization
identically, no approximation.

::: {.d2l-note}
The transient is large: the raw ratio mis-scales early steps by
$(1-\beta_1^t)/\sqrt{1-\beta_2^t}$, already $3.16$ at $t=1$ and peaking
above $6\times$ near $t=12$.
:::
:::

::: {.col .fig}
@fig:mdl-opt-bias-correction
:::
:::
:::

::: {.slide title="Adam is three ideas stacked"}
[The family]{.kicker}

$$\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t,
\quad
\mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2,
\quad
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

RMSProp's per-coordinate scale, momentum's averaged direction, and the
exact startup correction for both ($\hat{\mathbf{m}}_t$,
$\hat{\mathbf{v}}_t$).

. . .

::: {.d2l-note .rule}
Near a diagonal quadratic minimum, $\sqrt{\hat{v}_i} \approx |g_i| =
\lambda_i|x_i|$: the update is $\approx \eta\,\mathrm{sign}(x_i)$, sign
descent whose per-coordinate steps $\eta/(\lambda_i|x_i|)$ carry the
$1/\lambda_i$ ratio, reconstructed from first-order information alone.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[When Adam fails]{.dtitle}

[a convex counterexample, and the valley revisited]{.dsub}
:::
:::

::: {.slide title="A convex problem built to defeat Adam"}
[The counterexample]{.kicker}

Reddi--Kale--Kumar (2018): on $x \in [-1,1]$, cycle the convex losses

$$f_t(x) = \begin{cases} C\,x, & t \bmod 3 = 1,\\ -x, & \textrm{otherwise,}\end{cases} \qquad C > 2,$$

whose gradients sum to $C - 2 > 0$ per period, so the best point is
$x^\star = -1$.

. . .

::: {.d2l-note .warn}
The mechanism in one sentence: **the effective step on the informative
gradient shrinks faster than its information accrues.** The rare $+C$
lands in $\mathbf{v}$ *quadratically* and throttles its own step by
$\approx C$; then $\beta_2$ forgets, and the two $-1$ steps run at full
size, the wrong way. Signal is throttled by its own magnitude; noise
is not.
:::
:::

::: {.slide title="Adam converges to the worst point"}
[The counterexample]{.kicker}

Running the construction at $C = 4$, against AMSGrad and projected SGD
with the same $1/\sqrt{t}$ decay:

@!adaptive-stochastic-methods-reddi

**AMSGrad**'s one-line fix, replacing $\hat{\mathbf{v}}_t$ by the running
maximum, makes per-coordinate steps nonincreasing, so the large
gradient is never forgotten.

::: {.d2l-note}
The theorem shows that vanilla Adam has **no general convergence guarantee,
even on convex objectives**, without additional assumptions or modification. Practice keeps Adam anyway: real noise
usually breaks up consistently-informative rare gradients.
:::
:::

::: {.slide title="The valley, revisited"}
[The payoff]{.kicker}

The $\kappa = 10^3$ quadratic of the gradient-based-optimization section,
optimally tuned GD versus hand-rolled Adam:

@!adaptive-stochastic-methods-adam-vs-gd

From the first iteration Adam's per-coordinate steps sit in the ratio
$10^{-2} : 10^{-5}$, the eigenvalue ratio, reconstructed from gradient
magnitudes with no Hessian. **GD $6160$, Adam $344$.**

::: {.d2l-note .warn}
The counterpoint: $\sqrt{\hat{\mathbf{v}}}$ conflates curvature
with noise, the diagonal misses every off-axis correlation (rotate the
quadratic $45°$ and the rescaling stops helping), and Reddi steers it
adversarially. A well-chosen point partway along the cost--fidelity curve.
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Decoupled weight decay]{.dtitle}

[coupled versus decoupled regularization]{.dsub}
:::
:::

::: {.slide title="Weight decay through Adam is not weight decay"}
[AdamW]{.kicker}

Under SGD, "penalize the loss" and "shrink the weights" are the *same
update*. Under Adam the penalty gradient rides through the
preconditioner, and the shrinkage on coordinate $i$ becomes

$$\underbrace{\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; w_{t,i}}_{\ell_2\ \textrm{through Adam}}
\qquad \textrm{versus} \qquad
\underbrace{\eta\,\lambda\, w_{t,i}}_{\textrm{decoupled}}.$$

. . .

::: {.d2l-note .warn}
The regularization strength is no longer a constant of the problem:
noisy-gradient weights are barely decayed, quiet ones are decayed hard.
Whatever $\lambda$ *meant* (the Gaussian prior of MAP, the maximum-likelihood
section; the norm constraint it multiplies, the constrained-optimization-and-
duality section) assumed one $\lambda$ for all coordinates. Adam's
preconditioner silently discards it.
:::
:::

::: {.slide title="AdamW: one λ, one meaning"}
[AdamW]{.kicker}

Decouple: the loss gradient goes through the preconditioner, the decay
does not. Two pure-noise weights at scales $\sigma = (10, 0.1)$, where
decay is the only systematic force:

@!adaptive-stochastic-methods-adamw

Coupled decay gave the two weights effective rates $100\times$ apart,
a disparity nobody chose; AdamW shrinks both by the uniform
$(1-\eta\lambda)^T$, within $2\%$ of prediction.

::: {.d2l-note}
In the major libraries, `AdamW` and `Adam` with a `weight_decay` flag
implement **two different regularizers**: the decoupled update and the
coupled one.
:::
:::

::: {.slide}
::: {.divider}
[05]{.dnum}

[Schedules and warmup]{.dtitle}

[what decay does, and why ramps come first]{.dsub}
:::
:::

::: {.slide title="The schedule zoo, at equal budget"}
[Schedules]{.kicker}

::: {.cols .vc}
::: {.col}
The mathematics a schedule negotiates was already proved: a constant
step parks on a **noise floor** $\propto \eta$; Robbins--Monro decay
reaches the optimum; beyond convexity *no theorem ranks decay shapes*.

- **cosine**: one knob, no kinks, a long gentle tail
- **WSD**: hold the plateau, drop the floor in a final decay;
  checkpoints stay re-decayable to any budget
:::

::: {.col .fig .big}
@fig:mdl-opt-schedule-zoo
:::
:::
:::

::: {.slide title="All of WSD's gain arrives in the decay phase"}
[Schedules]{.kicker}

On the noisy quadratic (the miniature where the floor is exact),
at $80\%$ of budget and at the end:

@!adaptive-stochastic-methods-schedules

WSD sits *on the constant's floor* at $80\%$ (same trajectory, to every
digit), then $400$ decay steps drop it $17\times$: the loss-curve cliff
of WSD runs, reproduced by a one-line quadratic. Cosine's longer tail
wins this pure-quadratic endgame; WSD's case lives off the toy, in
re-decayable checkpoints and time spent at large steps.
:::

::: {.slide title="Warmup: never trust an estimated preconditioner cold"}
[Warmup]{.kicker}

For adaptive methods warmup is close to structural, and both reasons are
this chapter's instruments:

- At $t = 1$, $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$: the rescaling is
  estimated from **one sample**, and the first step is
  $\mathrm{sign}(\mathbf{g}_1)$, full-magnitude movement on every
  coordinate, noise included. Bias correction makes it *unbiased*, not
  *accurate*; only $\sim 1/(1-\beta_2)$ steps of data fix the variance.
- The **edge of stability** (the gradient-based-optimization section):
  sharpness equilibrates onto $2/\eta$,
  but that takes time; a full-size $\eta$ at initialization violates a
  ceiling the landscape has not yet adapted to.

::: {.d2l-note}
Ramp $\eta$ while the preconditioner accumulates data and the
curvature meets its step size. Every large run converged on this
independently.
:::
:::

::: {.slide}
::: {.divider}
[06]{.dnum}

[Beyond diagonals]{.dtitle}

[the preconditioning ladder]{.dsub}
:::
:::

::: {.slide title="Every optimizer answers one question"}
[The ladder]{.kicker}

*Which matrix $B_t$ multiplies the gradient?* GD says $I$; Newton says
$(\nabla^2 f)^{-1}$ at $O(d^3)$; Adam says a diagonal at $O(d)$. The
rungs between exploit one structural fact: parameters come in **matrices**.

::: {.d2l-note .rule}
diagonal (**Adam**) → Kronecker-factored Fisher (**K-FAC**, natural
gradient) → two-sided full-matrix roots (**Shampoo**) → spectral
whitening by polar factor (**Muon**) → full Newton (unreachable)
:::

. . .

K-FAC's whole trick in one display, a curvature matrix with $(mn)^2$
entries preconditioned by two small inverses:

$$(A \otimes G)^{-1}\,\mathrm{vec}(V) \;=\; \mathrm{vec}\left(G^{-1}\, V\, A^{-1}\right).$$
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Ghadimi--Lan:** the descent lemma survives noise, taxed; the
  stochastic rate is $K^{-1/2}$.
- **AdaGrad** is steepest descent in a learned metric; **Adam** adds an
  EMA, momentum, and an *exactly* unbiased startup correction.
- **Reddi:** Adam can converge to the worst point of a convex problem;
  AMSGrad's running max restores the guarantee.
:::

::: {.col}
- What adaptivity wins back: the $1000\times$ eigenvalue ratio from
  gradients alone; GD $6160$, Adam $344$.
- **AdamW** decouples decay from the preconditioner: one $\lambda$, one
  meaning.
- **Schedules** trade the noise floor against the transient; **warmup**
  shields a cold preconditioner; beyond the diagonal, the ladder.
:::
:::
:::
