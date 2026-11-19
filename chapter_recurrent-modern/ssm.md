# State Space Models
:label:`sec_ssm`

How long should a memory last? The subject of a sentence matters for a few
words, the topic of a paragraph for a few hundred, and the name of the
murderer in a detective novel for the whole book. The minGRU of
:numref:`sec_linear-recurrence` can in principle learn any of these, but
it starts from gates near one half. A memory of a thousand steps needs a
decay near 0.999, which halves a stored symbol only every 693 steps.
Gradient descent has to get there from one half, and a gradient carried
back a thousand steps through decays of one half is multiplied by
$2^{-1000}$. In single precision that number is not small but zero. Nor
does anything in the layer suggest which lengths to try.

Physics has long described systems that forget at a known rate. A cup of
coffee cools toward room temperature, and the gap shrinks by the same
fraction every minute, at a rate that belongs to the cup. A state space
model treats the memory of a layer as such a system, with many rates, and
reads it at discrete steps. The result is again a linear recurrence, but
every decay now comes from a rate and a step size, so we can choose the
timescales before training starts, give every feature memories of many
lengths at once, and gain a second way to train, as a convolution. We
derive the recurrence, build the S4D layer of
:citet:`Gu.Gupta.Goel.Re.2022`, and test on the delayed recall task of
:numref:`sec_lstm` whether the right starting timescales are what a
recurrence needs to remember for a thousand steps.

```{.python .input #ssm-linear-recurrence-and-state-space-models}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #ssm-linear-recurrence-and-state-space-models}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## From Continuous Time to a Recurrence

### Continuous-Time Dynamics
:label:`subsec_state-space`

General systems of this kind are hard to think about all at once, so let
us start, as with the coffee, with a single number. Let $h(t)$ be a state
that changes at a rate proportional to its own value and is driven by an
input $x(t)$:

$$h'(t) = \lambda\, h(t) + x(t).$$
:eqlabel:`eq_ssm_scalar`

Without input, the solution is $h(t) = e^{\lambda t} h(0)$. For $\lambda <
0$ the state decays to zero, faster for larger $|\lambda|$; with $\lambda =
-1$, a fraction $e^{-1} \approx 0.37$ remains after one unit of time. For
$\lambda > 0$ it grows without bound, which no memory should do. The
parameter $\lambda$ is a rate: it says how fast the state forgets per unit
of time, not per step. This distinction will matter in a moment.

A *state space model* (SSM) generalizes this to a vector state
$\mathbf{h}(t) \in \mathbb{R}^N$ that is driven by a scalar input $x(t)$ and
read out as a scalar output $y(t)$:

$$
\mathbf{h}'(t) = \mathbf{A}\, \mathbf{h}(t) + \mathbf{B}\, x(t),
\qquad
y(t) = \mathbf{C}\, \mathbf{h}(t) + D\, x(t).
$$
:eqlabel:`eq_ssm_cont`

The input enters the state through $\mathbf{B} \in \mathbb{R}^N$, the
*state matrix* $\mathbf{A} \in \mathbb{R}^{N \times N}$ moves the state
along, $\mathbf{C} \in \mathbb{R}^{1 \times N}$ reads it, and $D$ passes the
input straight to the output.

We take $\mathbf{A} = \textrm{diag}(\lambda_1, \ldots, \lambda_N)$
diagonal, so that the model is $N$ independent copies of
:eqref:`eq_ssm_scalar`, each with its own rate, read out together by
$\mathbf{C}$. The reason is cost. With a full $\mathbf{A}$, every step
multiplies the state by an $N \times N$ matrix, and a parallel scan
multiplies such matrices with each other, $N^3$ multiply-adds per combine;
with a diagonal $\mathbf{A}$, a step is $N$ scalar multiplications.

The diagonal is less of a restriction than it looks. In exact arithmetic
almost every matrix can be diagonalized over the complex numbers, and
$\mathbf{B}$ and $\mathbf{C}$ absorb the change of basis, so a complex
diagonal loses almost nothing in what the layer can represent; that such
layers also train well was found by experiment
:cite:`Gu.Gupta.Goel.Re.2022`. A *real* diagonal does give something up,
namely oscillation. A complex rate $\lambda = -\alpha + i\omega$ makes a
state decay while it rotates, whereas a real rate can only decay. The S4D
layer of that paper uses complex rates by default. We build its variant
with real rates, because it is the simplest, and will see below what the
simplification costs; complex rates are left to an exercise.

A layer reads a sequence of $d$-dimensional vectors and runs one such SSM
for each coordinate, or *channel*. Picture channel 7 of the input. It
drives its own 16 states, each a leaky copy of the input that forgets at
its own rate, and channel 7 of the output is a learned combination of those
16 through $\mathbf{C}$. Channel 8 has 16 states of its own, and the two
channels never mix inside the layer. The state of a layer therefore holds
$d \times N$ numbers, $N$ times as many as a minGRU of the same width.

### Discretization
:label:`subsec_zoh`

To process a sequence, we read the continuous system at steps of length
$\Delta$, so that step $t$ ends at time $t\Delta$. Between two readings the
input must have some value, and the simplest choice is to hold it at $x_t$
throughout the step that ends at $t$. With this rule, called the
*zero-order hold*, :eqref:`eq_ssm_scalar` can be solved exactly over one
step. The trick is to shift the state. With the input held at $x_t$, the
quantity $h + x_t/\lambda$ has the derivative $\lambda h + x_t = \lambda(h +
x_t/\lambda)$, so it obeys the equation without input and shrinks by the
factor $e^{\Delta\lambda}$ over the step: $h_t + x_t/\lambda =
e^{\Delta\lambda}(h_{t-1} + x_t/\lambda)$. Rearranged, this is a linear
recurrence:

$$
h_t = \bar{a}\, h_{t-1} + \bar{b}\, x_t, \qquad
\bar{a} = e^{\Delta \lambda}, \qquad
\bar{b} = \frac{e^{\Delta \lambda} - 1}{\lambda}.
$$
:eqlabel:`eq_zoh`

Over one step the state keeps the fraction $\bar{a}$ of its value, and the
input enters with the weight $\bar{b}$. Nothing was approximated: at the
step boundaries, the recurrence and the differential equation agree.

For $\lambda = -1$ the two coefficients are $\bar{a}$ and $1 - \bar{a}$,
so the update reads $h_t = \bar{a}\, h_{t-1} + (1 - \bar{a})\, x_t$: the
blend of the GRU, with $\bar{a}$ in the role of the update gate. The step
size decides the gate. With $\Delta = 0.01$ the state keeps $\bar{a} =
0.990$ of its value, a half-life of 69 steps, and admits 1% of the input;
with $\Delta = 0.1$ it keeps 0.905, a half-life of 7 steps; with $\Delta =
1$ it keeps 0.37 and is mostly overwritten. A small step size means a long
memory.

Is a decay obtained this way better behaved than a gate? Not in the values
it can take. If we store the rate as $\lambda = -e^{\theta}$ and learn
$\theta$, then for every $\theta$ and every $\Delta > 0$ the decay $\bar{a}
= e^{\Delta\lambda}$ lies in $(0, 1)$, and no gradient step can make the
recurrence grow; but the sigmoid gate of the minGRU cannot leave $(0, 1)$
either. What is new is the form of the parameter. A state forgets half of
its content every $\ln 2 / (\Delta|\lambda|)$ steps, and this half-life is
set by a product of two factors with separate jobs. The rates tell the
states of a channel apart; the step size, one number per channel, slows or
speeds all of them at once. A gate has one independent number per unit and
nothing that moves a group of units together.
:numref:`fig_ssm_discretization` shows the map from
$\Delta|\lambda|$ to the decay and its half-life.

![Zero-order hold turns a real rate $\lambda < 0$ into a decay per step $\bar{a} = e^{-\Delta|\lambda|}$, which depends only on the product $\Delta|\lambda|$. The labels give the half-life in steps, $\ln 2 / (\Delta|\lambda|)$. The shaded band is the range that the S4D initialization covers, with $\Delta$ between 0.001 and 0.1 and $|\lambda|$ between 1 and 16.](../img/mdl-modernrnn-discretization.svg)
:label:`fig_ssm_discretization`

With a diagonal $\mathbf{A}$, the zero-order hold applies to each state
separately: $\bar{a}_n = e^{\Delta \lambda_n}$ and $\bar{b}_n =
(e^{\Delta \lambda_n} - 1) b_n / \lambda_n$. (For small $\Delta\lambda_n$,
the difference $e^{\Delta\lambda_n} - 1$ loses digits in floating point, so
the code computes it with `expm1`, which evaluates $e^x - 1$ accurately.)
For a full matrix the step is $\bar{\mathbf{A}} = e^{\Delta\mathbf{A}}$,
derived in :numref:`subsec_mdl-ode-driven-linear`. (The appendix writes the
state as $\mathbf{x}$, the input as $\mathbf{u}$ and the step as $h$, and it
indexes the held input by the step that it starts, one less than ours.)

The rates also tell us how to initialize. S4D, in the variant with real
rates that we build below, sets $\lambda_n = -n$ for $n = 1, \ldots, N$ and
draws one step size per channel, log-uniformly between 0.001 and
0.1. With $N = 16$ states, the half-lives then range from under half a step
($\Delta = 0.1$, $\lambda = -16$) to 693 steps ($\Delta = 0.001$, $\lambda =
-1$), the shaded band of :numref:`fig_ssm_discretization`. Before training,
every channel already holds memories of many lengths, each channel at its
own scale. The rates come from the HiPPO theory of
:citet:`Gu.Dao.Ermon.ea.2020`, which designs $\mathbf{A}$ so that the state
stores the coefficients of a polynomial approximation of the input's
history; the rates above keep the diagonal of that matrix. The theory is
elegant, and for what follows it can be forgotten: what the initialization
contributes is a wide spread of timescales.

### The Convolutional Form
:label:`subsec_ssm-conv`

The parallel scan of :numref:`sec_linear-recurrence` computes all states of
a linear recurrence at once, by composing affine steps in about $\log_2 T$
rounds. Dynamics that do not change along the sequence allow a second way.
When $\bar{a}_n$, $\bar{b}_n$, and $\mathbf{C}$ are the same at every step,
the model is *linear time-invariant*, and unrolling it gives the outputs
directly in terms of the inputs. State $n$ holds $\sum_{k \ge 0}
\bar{a}_n^k \bar{b}_n x_{t-k}$, every input weighted by the decay raised to
its age, and the readout adds the states with weights $c_n$:

$$
y_t = \sum_{k=0}^{t-1} K_k\, x_{t-k} + D x_t, \qquad
K_k = \sum_{n=1}^{N} c_n \bar{a}_n^k \bar{b}_n.
$$
:eqlabel:`eq_ssm_kernel`

The output is a *causal convolution* of the input with the kernel
$\mathbf{K} = (K_0, K_1, \ldots)$: every input contributes with a weight
that depends only on how many steps ago it arrived. The kernel is a sum of
$N$ decaying exponentials, one per state. A single state with
$\lambda = -1$, $\Delta = 0.1$, and $c = 1$ gives $K_k = 0.095 \cdot
0.905^k$, the weights of an exponential moving average.

One model therefore has three equivalent forms
(:numref:`fig_ssm_views`). The differential equation is where we choose the
parameters and read them as timescales. The recurrence is how the model
generates, one step at a time in constant memory. The convolution is a
second way to train in parallel: the convolution of two sequences of
length $T$ can be computed with the fast Fourier transform (FFT) in
$O(T \log T)$ operations, by transforming both, multiplying them, and
transforming back. S4D's predecessor S4 :cite:`Gu.Goel.Re.2022`, whose
state matrix is not diagonal, trained this way, and it was the first model
to do better than chance on Path-X, a benchmark that shows an image of
$128 \times 128$ pixels one pixel at a time, 16,384 steps in all, and asks
whether two marked points are joined by a path. The convolution requires
dynamics that are the same at every step; the scan does not.

![Three forms of one state space model. Left: the continuous-time state responds to an input pulse. Middle: the discretized system is a linear recurrence, one step per token. Right: unrolled, the same system is a causal convolution whose kernel is the response to a single pulse.](../img/mdl-modernrnn-ssm-views.svg)
:label:`fig_ssm_views`

### Timescales and Learned Kernels

A reader may object at this point. The gates of the minGRU are decays in
$(0, 1)$ as well, and nothing stops us from setting their biases so that
they start at the decays of S4D; the experiment at the end of this section
does that. What does the differential equation give that a gate bias does
not? Less than it seems, and one thing that matters. Writing a half-life
down as a rate and a step size is a convenience, and a minor one: for a
gate we solve $\sigma(b) = e^{-\Delta}$ for the bias $b$, which takes one
line. The convolutional form comes from dynamics that stay the same along
the sequence, not from calculus; any linear recurrence whose decays do not
depend on the input has it. The learned kernel comes from the size of the
state. The readout $\mathbf{C}$ combines the $N$ states of a channel, so
the memory of a channel is a weighted sum of $N$ exponentials of its own; a
unit of the minGRU holds a single one, and only the layer after it can mix
units. What does come from writing a decay as a rate times a step is the
step size, one number per channel that stretches or shrinks all $N$
timescales of the channel together. It is the part of this section that
the rest of the chapter builds on.

The kernel is the main gain in what a channel can represent, so let us
see how far it goes. How well can $N$ exponentials imitate a memory that is
not exponential? Suppose a channel should report the input from exactly 20
steps ago, a kernel that is zero except for $K_{20} = 1$. One prediction is
safe before any computation: a single state cannot do it, because one
decaying exponential is largest at $k = 0$. The cell fits the readout
$\mathbf{c}$ of one channel by least squares, with $N$ states at S4D's
rates and a fixed step $\Delta = 0.05$ (half-lives from 14 steps down to
under one; exercise 2 varies $\Delta$), to two targets over 100 steps: the
exact delay, and a smooth bump that remembers "about 20 steps ago, give or
take 4". The question is how close $N$ states come, measured by the
relative error $\|\mathbf{K} - \textrm{target}\| / \|\textrm{target}\|$, and
how large a readout they need to get there.

```{.python .input #ssm-what-the-continuous-time-view-adds}
%%tab pytorch, jax
def fit_kernel(N, target, dt=0.05):
    """Fit the readout c of N states at S4D's rates to a target kernel."""
    rate = -np.arange(1., N + 1)
    a_bar, b_bar = np.exp(dt * rate), np.expm1(dt * rate) / rate
    basis = a_bar ** np.arange(len(target))[:, None] * b_bar  # K = basis @ c
    c = np.linalg.lstsq(basis, target, rcond=None)[0]
    K = basis @ c
    return K, c, np.linalg.norm(K - target) / np.linalg.norm(target)

k = np.arange(100)
spike = (k == 20).astype(float)      # Exactly the input 20 steps back
bump = np.exp(-(k - 20) ** 2 / 32)   # About 20 steps back, give or take 4
print(' N  peak  largest |c|  spike residual  bump residual')
for N in [1, 4, 16]:
    K, c, spike_error = fit_kernel(N, spike)
    bump_error = fit_kernel(N, bump)[2]
    print(f'{N:2d}  {K.argmax():4d}  {np.abs(c).max():11.0e}  '
          f'{spike_error:14.2f}  {bump_error:13.2f}')
```

With one state the kernel peaks at 0, as predicted, and misses both
targets almost entirely. Four states move the peak to 17 and bring the
error on the bump down to 0.55. Sixteen states put the peak at 20 and
match the bump to an error of 0.03, while the sharp delay stays out of
reach at 0.89.

The column of weights tells us not to celebrate yet. The weights of the
spike fit reach $2 \times 10^{11}$, and those of the bump fit are of the
same order. Real exponentials with nearby rates look much alike, and least
squares tells them apart only with enormous weights of opposite sign that
cancel almost perfectly. Gradient descent from small weights has no
reasonable path to such a solution. With weights of ordinary size, the
peak of the kernel can still be moved away from zero, but the kernel stays
broad; exercise 2 measures how much of the bump survives. A sum of decaying
exponentials can remember roughly when something happened, not exactly
when, and with a readout of ordinary size the "roughly" is rough indeed.
For a language model the lost precision matters less than it sounds: a
word usually needs the subject a few words back, not the token exactly 20
positions ago. The poor conditioning is another matter, and it is where
complex rates help. Their oscillating terms differ from one another far
more than decays with nearby rates do, and exercise 3 repeats the fit with
them.

## A Diagonal State Space Layer
:label:`subsec_s4d`

### The S4D Layer

The `S4D` layer runs one diagonal SSM with $N$ states per channel. It
stores the rates as $\lambda_n = -e^{\theta_n}$, so that they stay
negative, and one step size per channel as $\log \Delta$, both initialized
as described above. It fixes $\mathbf{B} = \mathbf{1}$: only the products
$b_n c_n$ enter the kernel :eqref:`eq_ssm_kernel`, so the learned readout
$\mathbf{C}$ can absorb any scale that $\mathbf{B}$ would carry. We start
with the parameters and the method `discretize`, which applies the
zero-order hold :eqref:`eq_zoh` to every state of every channel. Its two
lines are the whole of the continuous-time view in code; the rest of the
class works with $\bar{a}$ and $\bar{b}$ alone, as any linear recurrence
would.

```{.python .input #ssm-the-s4d-layer-1}
%%tab pytorch
class S4D(nn.Module):  #@save
    """A diagonal state space layer: N states per channel, fixed dynamics."""
    def __init__(self, num_hiddens, num_states=16, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        d, N = num_hiddens, num_states
        # Rates lambda_n = -(n + 1), stored as log(-lambda) to stay negative
        self.log_neg_rate = nn.Parameter(
            torch.log(torch.arange(1., N + 1)).repeat(d, 1))
        self.log_dt = nn.Parameter(  # Log-uniform in [dt_min, dt_max]
            torch.rand(d, 1) * math.log(dt_max / dt_min) + math.log(dt_min))
        self.C = nn.Parameter(torch.randn(d, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.ones(d))

    def discretize(self):
        """Zero-order hold with B = 1: decays a_bar and input weights b_bar."""
        rate = -torch.exp(self.log_neg_rate)  # (d, N)
        dt_rate = torch.exp(self.log_dt) * rate
        return torch.exp(dt_rate), torch.expm1(dt_rate) / rate
```

```{.python .input #ssm-the-s4d-layer-1}
%%tab jax
class S4D(nnx.Module):  #@save
    """A diagonal state space layer: N states per channel, fixed dynamics."""
    def __init__(self, num_hiddens, num_states=16, dt_min=1e-3, dt_max=1e-1,
                 *, rngs):
        d, N = num_hiddens, num_states
        # Rates lambda_n = -(n + 1), stored as log(-lambda) to stay negative
        self.log_neg_rate = nnx.Param(
            jnp.tile(jnp.log(jnp.arange(1., N + 1)), (d, 1)))
        self.log_dt = nnx.Param(  # Log-uniform in [dt_min, dt_max]
            rngs.params.uniform((d, 1)) * math.log(dt_max / dt_min)
            + math.log(dt_min))
        self.C = nnx.Param(rngs.params.normal((d, N)) / math.sqrt(N))
        self.D = nnx.Param(jnp.ones(d))

    def discretize(self):
        """Zero-order hold with B = 1: decays a_bar and input weights b_bar."""
        rate = -jnp.exp(self.log_neg_rate[...])  # (d, N)
        dt_rate = jnp.exp(self.log_dt[...]) * rate
        return jnp.exp(dt_rate), jnp.expm1(dt_rate) / rate
```

The forward pass, used in training, holds no new idea, only an efficient
way to convolve. It computes the kernel :eqref:`eq_ssm_kernel` of every
channel from the discretized parameters and convolves it with the input by
the FFT. Both sequences are padded to twice their length, because the FFT
computes a *circular* convolution, in which the end of a sequence would
otherwise wrap around and leak into its beginning.

```{.python .input #ssm-the-s4d-layer-2}
%%tab pytorch
@d2l.add_to_class(S4D)  #@save
def forward(self, X):  # X: (batch_size, num_steps, d)
    T = X.shape[1]
    a_bar, b_bar = self.discretize()
    # All powers a_bar^k for k < T: d * N * T numbers, the price of the kernel
    powers = a_bar.unsqueeze(-1) ** torch.arange(T, device=X.device)
    K = ((self.C * b_bar).unsqueeze(-1) * powers).sum(1)  # Kernel, (d, T)
    Y = torch.fft.irfft(torch.fft.rfft(X, n=2 * T, dim=1) *
                        torch.fft.rfft(K.T, n=2 * T, dim=0), n=2 * T, dim=1)
    return Y[:, :T] + self.D * X
```

```{.python .input #ssm-the-s4d-layer-2}
%%tab jax
@d2l.add_to_class(S4D)  #@save
def __call__(self, X):  # X: (batch_size, num_steps, d)
    T = X.shape[1]
    a_bar, b_bar = self.discretize()
    # All powers a_bar^k for k < T: d * N * T numbers, the price of the kernel
    powers = a_bar[..., None] ** jnp.arange(T)
    K = ((self.C[...] * b_bar)[..., None] * powers).sum(1)  # Kernel, (d, T)
    Y = jnp.fft.irfft(jnp.fft.rfft(X, n=2 * T, axis=1) *
                      jnp.fft.rfft(K.T, n=2 * T, axis=0), n=2 * T, axis=1)
    return Y[:, :T] + self.D[...] * X
```

For generation, `step` advances the recurrence :eqref:`eq_zoh` of every
state by one token and reads the output. Its state has $d \times N$
numbers per sequence, whatever the length of the text generated so far.

```{.python .input #ssm-the-s4d-layer-3}
%%tab pytorch
@d2l.add_to_class(S4D)  #@save
def step(self, x, h):
    """One step: x is (batch_size, d); the state h is (batch_size, d, N)."""
    a_bar, b_bar = self.discretize()
    h = a_bar * h + b_bar * x.unsqueeze(-1)
    return (h * self.C).sum(-1) + self.D * x, h
```

```{.python .input #ssm-the-s4d-layer-3}
%%tab jax
@d2l.add_to_class(S4D)  #@save
def step(self, x, h):
    """One step: x is (batch_size, d); the state h is (batch_size, d, N)."""
    a_bar, b_bar = self.discretize()
    h = a_bar * h + b_bar * x[..., None]
    return (h * self.C[...]).sum(-1) + self.D[...] * x, h
```

We check that the three forms agree on a random layer: the FFT convolution
of the forward pass, the parallel scan of the recurrence, and the recurrence run
one step at a time with `step`.

```{.python .input #ssm-the-s4d-layer-4}
%%tab pytorch
layer = S4D(8, num_states=4)
X = torch.randn(2, 50, 8)  # (batch_size, num_steps, d)
with torch.no_grad():
    a_bar, b_bar = layer.discretize()
    H = d2l.linear_scan(a_bar, b_bar * X.unsqueeze(-1))  # (batch, T, d, N)
    Y_scan = (H * layer.C).sum(-1) + layer.D * X
    h, Y_step = torch.zeros(2, 8, 4), []
    for x in X.unbind(1):
        y, h = layer.step(x, h)
        Y_step.append(y)
    Y_conv = layer(X)
Y_step = torch.stack(Y_step, 1)
print(f'convolution vs scan: {(Y_conv - Y_scan).abs().max():.1e}, '
      f'convolution vs steps: {(Y_conv - Y_step).abs().max():.1e}')
```

```{.python .input #ssm-the-s4d-layer-4}
%%tab jax
layer = S4D(8, num_states=4, rngs=nnx.Rngs(0))
X = jax.random.normal(jax.random.key(1), (2, 50, 8))  # (batch, T, d)
a_bar, b_bar = layer.discretize()
H = d2l.linear_scan(a_bar, b_bar * X[..., None])  # (batch, T, d, N)
Y_scan = (H * layer.C[...]).sum(-1) + layer.D[...] * X
h, Y_step = jnp.zeros((2, 8, 4)), []
for x in jnp.unstack(X, axis=1):
    y, h = layer.step(x, h)
    Y_step.append(y)
Y_conv = layer(X)
Y_step = jnp.stack(Y_step, 1)
print(f'convolution vs scan: {jnp.abs(Y_conv - Y_scan).max():.1e}, '
      f'convolution vs steps: {jnp.abs(Y_conv - Y_step).max():.1e}')
```

The outputs differ by a few parts in ten million, the rounding error of
single precision, so the three forms compute the same function. For
training, the FFT needs less memory than the scan. The scan stores the
$d \times N$ state at every position of every sequence; the FFT stores the
$d \times N \times T$ powers that build the kernels once for the whole
batch.

### Long-Range Recall

Does the range of timescales matter when a model has to remember? We
return to the delayed recall task of :numref:`sec_lstm`: a sequence starts
with one of eight symbols, and after $T - 1$ filler tokens the model must
name it; chance is 1/8. There, the LSTM and the GRU with gates that start
near one half lost the symbol by 50 steps, while the same networks with
their forget and update gates started near one held it for 200. Here the
delays are 100 and 1,000 steps, and the fillers are all the same token, so
the symbol's trace is the only thing that differs from one sequence to the
next. The task then asks about timescales and nothing else. Which tokens
deserve to be kept is a separate question, and :numref:`sec_mamba` takes
it up.

We compare four layers. The minGRU starts with gates near one half. S4D
has 64 channels of 16 states, initialized as above. Two controls separate
the timescales from the rest of the layer. The first is an S4D whose step
sizes start between 0.1 and 1, so that none of its half-lives exceeds 7
steps. The second is a minGRU whose gates start at the decays of the
slowest state of S4D, $e^{-\Delta}$ with $\Delta$ log-uniform between
0.001 and 0.1, half-lives from 7 to 693 steps. A sigmoid gate $\sigma(b)$
equals $e^{-\Delta}$ when $b = -\log(e^\Delta - 1)$, which is what
`init_timescales` computes.

```{.python .input #ssm-long-range-recall-1}
%%tab pytorch
def init_timescales(layer, dt_min=1e-3, dt_max=1e-1):  #@save
    """Start a minGRU's gates at the decays exp(-dt) of S4D."""
    dt = dt_min * (dt_max / dt_min) ** torch.rand(layer.W_z.bias.shape)
    with torch.no_grad():  # sigmoid(b) = exp(-dt)
        layer.W_z.bias.copy_(-torch.log(torch.expm1(dt)))
    return layer
```

```{.python .input #ssm-long-range-recall-1}
%%tab jax
def init_timescales(layer, dt_min=1e-3, dt_max=1e-1, *, rngs):  #@save
    """Start a minGRU's gates at the decays exp(-dt) of S4D."""
    dt = dt_min * (dt_max / dt_min) ** rngs.params.uniform(
        layer.W_z.bias.shape)
    layer.W_z.bias[...] = -jnp.log(jnp.expm1(dt))  # sigmoid(b) = exp(-dt)
    return layer
```

We reuse the model and the training function `d2l.train_delayed_recall` of
:numref:`sec_lstm`: an embedding, the sequence layer, and a classifier read
at the last position, trained with Adam at a learning rate of 0.003 on
1,500 fresh batches of 128 sequences. S4D keeps the width of its input, so
its model embeds tokens in 64 dimensions; the minGRUs map 32 to 64, like
the recurrent layers of :numref:`sec_lstm`. Every model is trained twice
from fresh random weights, and both test accuracies are printed. A
prediction first: if the architecture decides, the two S4D rows will agree
with each other and so will the two minGRU rows; if the timescales decide,
the controls will change sides.

```{.python .input #ssm-long-range-recall-2}
%%tab pytorch
# (layer, embedding width): S4D keeps its width; the minGRUs map 32 to 64
layers = {
    'minGRU': lambda: (d2l.MinGRU(32, 64), 32),
    'S4D': lambda: (S4D(64), 64),
    'S4D, short timescales': lambda: (S4D(64, dt_min=0.1, dt_max=1.0), 64),
    'minGRU, S4D timescales': lambda: (
        init_timescales(d2l.MinGRU(32, 64)), 32)}

def recall(make_layer, T):
    """Test accuracy after training a fresh model at delay T."""
    layer, num_inputs = make_layer()
    model = d2l.RecallModel(layer, 64, num_inputs=num_inputs)
    return d2l.train_delayed_recall(model, T, blank_fillers=True)

print(f"{'':23s} {'T=100':^11s}  {'T=1000':^11s}")
for name, make_layer in layers.items():  # Two runs per entry
    runs = [' '.join(f'{recall(make_layer, T):5.2f}' for _ in range(2))
            for T in [100, 1000]]
    print(f'{name:23s}', '  '.join(runs))
```

```{.python .input #ssm-long-range-recall-2}
%%tab jax
# (layer, embedding width): S4D keeps its width; the minGRUs map 32 to 64
layers = {
    'minGRU': lambda rngs: (d2l.MinGRU(32, 64, rngs=rngs), 32),
    'S4D': lambda rngs: (S4D(64, rngs=rngs), 64),
    'S4D, short timescales': lambda rngs: (
        S4D(64, dt_min=0.1, dt_max=1.0, rngs=rngs), 64),
    'minGRU, S4D timescales': lambda rngs: (init_timescales(
        d2l.MinGRU(32, 64, rngs=rngs), rngs=rngs), 32)}

def recall(make_layer, T):
    """Test accuracy after training a fresh model at delay T."""
    rngs = nnx.Rngs(d2l.get_key())
    layer, num_inputs = make_layer(rngs)
    model = d2l.RecallModel(layer, 64, num_inputs=num_inputs, rngs=rngs)
    return d2l.train_delayed_recall(model, T, blank_fillers=True)

print(f"{'':23s} {'T=100':^11s}  {'T=1000':^11s}")
for name, make_layer in layers.items():  # Two runs per entry
    runs = [' '.join(f'{recall(make_layer, T):5.2f}' for _ in range(2))
            for T in [100, 1000]]
    print(f'{name:23s}', '  '.join(runs))
```

The controls change sides. The minGRU with gates near one half is at
chance already after 100 steps, and S4D recalls the symbol after 1,000
steps in every run. With short half-lives, S4D still recalls it after 100
steps but is at chance after 1,000; given the decays of S4D, the minGRU
recalls it after 1,000 steps too. At 1,000 steps the initialization
decides, as the forget and update gates started near one decided in
:numref:`sec_lstm`, there over two hundred steps of random fillers.

One entry deserves a second look. The short-timescale S4D recalls the
symbol after 100 steps, although none of its half-lives exceeds 7 steps at
the start. After 99 steps its slowest state holds the symbol with a weight
of about $e^{-0.1 \cdot 99} \approx 5 \times 10^{-5}$. That is small, but
it is not nothing: the trace still carries a gradient to the step sizes
and the rates, and training uses it to lengthen the slowest half-lives
until the symbol survives the delay. After 999 steps the weight is
$e^{-100}$, about $10^{-44}$, far below the rounding error of the numbers
it is added to. No gradient reaches the dynamics, and nothing moves.
Exercise 5 lets you watch the half-lives grow, and asks what happens when
they are not allowed to. A memory does not end at its half-life. A faint
trace is enough to teach a layer to remember longer; what ends a memory is
a trace too faint to teach anything.

The experiment answers the objection raised earlier, and not in favor of
the differential equation: to start with long memories, a recurrence needs
the right decays, and a gate bias supplies them as well as a rate does.
What a layer keeps from the state space view is the rest: $N$ states per
channel, read out as a learned kernel, and, from continuous time, the step
size, one number per channel that scales all $N$ decays together.

The fixed dynamics that make S4D a convolution are also its limit. An
input's weight in the output depends only on how many steps ago it
arrived. In a text, whether the character four steps back ended a word or
sat in its middle, its influence on the present is the same fixed number
$K_4$. The minGRU chooses its decays from each input and can hold its
state through some characters and overwrite it at others; S4D cannot.
Whether one layer can have both, the choice of the minGRU and the
timescales of S4D, is the question of :numref:`sec_mamba`, which hands the
step size to the input.

## Summary

A state space model obtains the decays of a linear recurrence from a
continuous-time system with rates $\lambda$, read at steps of size
$\Delta$. Holding the input fixed over each step gives an exact recurrence
with decay $e^{\Delta\lambda}$. A state's half-life is
$\ln 2/(\Delta|\lambda|)$ steps, and one step size per channel scales all
the half-lives of the channel together; S4D starts every layer with
half-lives from under a step to about 700. With $N$ states per channel and
fixed dynamics, each channel's memory is a learned kernel, a sum of $N$
exponentials whose peak can be moved later but which cannot be made sharp,
and the layer trains as a convolution with the FFT. On delayed recall
over 1,000 steps the starting timescales decide: S4D and a
minGRU given the same decays recall the symbol, while gates that start
near one half and an S4D with short half-lives fail. Because its dynamics
are fixed, S4D weighs an input by its distance alone and cannot choose
what to keep.

## Exercises

1. [code] **Discretization rules.** The zero-order hold is one of several
   ways to turn :eqref:`eq_ssm_scalar` into a recurrence. The forward Euler
   rule uses $\bar{a} = 1 + \Delta\lambda$, the backward Euler rule
   $\bar{a} = 1/(1 - \Delta\lambda)$, and the bilinear rule $\bar{a} = (1 +
   \Delta\lambda/2)/(1 - \Delta\lambda/2)$
   (:numref:`sec_mdl-stiffness-implicit` studies the two Euler rules as
   solvers of a differential equation).
    1. For each rule, find the step sizes $\Delta$ for which every
       $\lambda < 0$ gives $|\bar{a}| < 1$. Which rule can make a stable
       continuous system unstable?
    1. Show that backward Euler with $\lambda = -1$ gives the update $h_t =
       z h_{t-1} + (1 - z) x_t$ with $z = \sigma(-\log\Delta)$, the blend of
       a GRU whose update gate has the negative logarithm of the step size as
       its pre-activation.
    1. Simulate the response of each rule to a unit step input for
       $\lambda = -1$ and $\Delta \in \{0.1, 1, 3\}$, and compare with the
       exact solution $1 - e^{-t}$.

1. [code] **Delays and kernels.** Use `fit_kernel` to study which memories
   $N$ real exponentials can represent.
    1. Fit a unit spike at delays 5, 20, and 50 with $N = 16$ states and
       $\Delta = 0.05$. How does the error change with the delay? Relate
       your answer to the half-lives of the slowest and the fastest state.
    1. Fit the spike at delay 20 for step sizes $\Delta$ between 0.01 and
       0.2. Which $\Delta$ fits best? What does this suggest about how a
       trained S4D should set the step size of a channel that needs a
       memory of a particular length?
    1. Fit $\mathbf{c}$ by gradient descent from zero (Adam, a few thousand
       steps) instead of least squares. How close do you come to the
       least-squares error for the bump, and for the spike? What do the
       weights of the least-squares fits predict?

1. [code] **Complex rates.** A state with the complex rate $\lambda =
   -\alpha + i\omega$, $\alpha > 0$, decays while it rotates.
    1. Show that the zero-order hold gives $\bar{a} = e^{-\alpha\Delta}
       e^{i\omega\Delta}$, and that two states with conjugate rates, input
       weights, and readout weights contribute a real kernel $K_k = r\,
       e^{-\alpha\Delta k} \cos(\omega\Delta k + \varphi)$ for some $r \ge 0$
       and $\varphi$. How many real parameters does such a pair have, and
       how many does a pair of real states have?
    1. Which kernels can such pairs represent that real rates cannot? Think
       of an input that repeats with a fixed period.
    1. Change `fit_kernel` to use the complex rates $\lambda_m = -1/2 + i\pi
       m$ for $m = 0, \ldots, N-1$ (the S4D-Lin initialization of
       :citet:`Gu.Gupta.Goel.Re.2022`) and to fit the real part of the
       kernel. For the spike at delay 20, compare the error and the size of
       the weights with the fit of real rates, and explain the difference.

1. [code] **Convolution by the FFT.** The forward pass of `S4D` pads the
   sequence and the kernel to length $2T$ before the FFT.
    1. Remove the padding, so that the FFT has length $T$, and compare the
       outputs with those of `step`. Which outputs are wrong, and what has
       been added to them?
    1. What is the shortest padded length that gives the exact causal
       convolution? Why do implementations often round it up to a power of
       two?

1. [code] **How long must memories start?** `init_timescales` starts the
   minGRU's gates at half-lives of up to about 700 steps.
    1. What bias $b$ gives a gate $\sigma(b)$ with a half-life of $k$ steps?
    1. Initialize the gates with half-lives log-uniform between 1 and
       $k_\textrm{max}$ steps and repeat the long-range recall experiment
       at 1,000 steps for $k_\textrm{max} \in \{10, 30, 100, 300\}$. What
       is the smallest $k_\textrm{max}$ that solves the task? Relate it to
       the weight with which the first token reaches the last position at
       initialization, and to the short-timescale S4D, which recalled after
       100 steps with half-lives of at most 7.
    1. Train the short-timescale S4D at 100 steps and print its longest
       half-life, $\ln 2 / \min_n (\Delta |\lambda_n|)$ over all channels,
       before and after training. Then train it again with `log_dt` and
       `log_neg_rate` held fixed, so that the dynamics cannot change and
       everything else still learns. What does a trace of $5 \times 10^{-5}$
       need in order to be useful?

[Discussions](https://d2l.discourse.group/t/36781)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.4]{.kicker}

State Space Models<br>
**a rate and a step size give a decay · N states make a learned kernel · fixed dynamics make a convolution · timescales decide long recall**
:::
:::

::: {.slide title="Gates That Start at One Half Forget in One Step"}
[The problem]{.kicker}

- Keep a symbol for 1,000 steps: a decay near **0.999**, half-life 693
  steps.
- A learned gate starts near $\sigma(0) = 1/2$: half-life **one step**.
- A gradient carried back 1,000 steps through decays of 1/2 is multiplied
  by $2^{-1000}$: in single precision, zero.

::: {.d2l-note .warn}
Can we choose how long each memory lasts before training starts?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- turn $h' = \lambda h + x$ into an exact recurrence and read the step
  size as a gate;
- set half-lives from under a step to 693 steps by initialization;
- say what $N$ states per channel add: a learned kernel, not one
  exponential;
- compute a time-invariant SSM as a convolution with the FFT;
- read a recall experiment whose controls separate timescales from
  architecture.

**Running example:** delayed recall of one of eight symbols after 100 and
1,000 identical fillers.
:::

::: {.slide title="Decays From a Rate Sit Between Linear Recurrences and Selection"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Gated Recurrent Networks | gates as learned decay |
| Sequence-to-Sequence Learning | one state between two networks |
| Linear Recurrences and Parallel Scans | $h_t = a_t h_{t-1} + b_t$, the scan, the minGRU |
| **State Space Models** | **decays from rates; kernels; convolution** |
| Selective State Space Models | the input sets the step size |
| Matrix-State Recurrences | a state of key and value pairs |
| The Delta Rule | correct a pair instead of adding |
| Hybrid Architectures | a little attention back |
:::

::: {.col .narrow}
- **Uses:** the linear recurrence and its scan, the minGRU, delayed
  recall.
- **Skips:** complex rates, the HiPPO theory behind the rates.
:::
:::
:::

::: {.slide title="Continuous Time Gives a Decay a Rate"}
$$h'(t) = \lambda h(t) + x(t), \qquad h(t) = e^{\lambda t} h(0) \textrm{ without input}$$

- $\lambda = -1$: $e^{-1} \approx 0.37$ remains after one unit of time.
- $\lambda$ is a rate **per unit of time**, not per step.

A state space model with $N$ states:

$$\mathbf{h}'(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t), \qquad y(t) = \mathbf{C}\mathbf{h}(t) + D x(t)$$

$\mathbf{B}$: input in · $\mathbf{A}$: dynamics · $\mathbf{C}$: readout ·
$D$: skip.
:::

::: {.slide title="Each Channel Drives Its Own N Leaky Integrators"}
- $\mathbf{A} = \textrm{diag}(\lambda_1, \ldots, \lambda_N)$: $N$
  independent scalar equations.
- Channel 7 drives its own 16 states; output 7 is a learned mix of them.
  Channels never mix.
- State per layer: $d \times N$, $N$ times a minGRU of width $d$.
- Diagonal: a scan combine costs $N$, not $N^3$.

::: {.d2l-note}
A complex diagonal loses almost nothing; a real one loses oscillation
($-\alpha + i\omega$ decays while it rotates). We build the real variant.
:::
:::

::: {.slide title="Holding the Input Over a Step Gives an Exact Recurrence"}
Hold $x$ at $x_t$ on the step of length $\Delta$ that ends at $t$
(zero-order hold).

I. $h + x_t/\lambda$ obeys the equation without input:
$\;(h + x_t/\lambda)' = \lambda\,(h + x_t/\lambda)$.

II. So it shrinks by $e^{\Delta\lambda}$:
$\;h_t + x_t/\lambda = e^{\Delta\lambda}(h_{t-1} + x_t/\lambda)$.

III. Rearrange:

$$\boxed{h_t = \bar a\, h_{t-1} + \bar b\, x_t, \qquad \bar a = e^{\Delta\lambda},\ \bar b = \frac{e^{\Delta\lambda} - 1}{\lambda}}$$

No approximation at the step boundaries.
:::

::: {.slide title="The Step Size Is a Gate"}
For $\lambda = -1$: $h_t = \bar a\,h_{t-1} + (1 - \bar a)\,x_t$, the GRU's
blend.

| $\lambda = -1$ | $\Delta = 0.01$ | $\Delta = 0.1$ | $\Delta = 1$ |
|---|---|---|---|
| keeps $\bar a$ | 0.990 | 0.905 | 0.37 |
| half-life (steps) | 69 | 7 | 0.7 |

::: {.d2l-note .rule}
A small step size means a long memory.
:::
:::

::: {.slide title="A Rate Makes the Half-Life a Number You Set and Read"}
::: {.cols .vc}
::: {.col .fig .big}
![](../img/mdl-modernrnn-discretization.svg)
:::

::: {.col .narrow}
- $\lambda = -e^\theta$: $\bar a \in (0, 1)$ for every $\theta$, as for a
  sigmoid gate. Nothing new.
- New: half-life $= \ln 2 / (\Delta|\lambda|)$ steps. The rates tell the
  states apart; one $\Delta$ per channel scales all of them at once.
:::
:::
:::

::: {.slide title="S4D Starts With Half-Lives From Under a Step to 693"}
- S4D: a diagonal state space layer, $N = 16$ real states per channel.
- Rates $\lambda_n = -n$, $n = 1, \ldots, 16$.
- One step size per channel, log-uniform in $[0.001, 0.1]$.
- Half-life $\ln 2/(\Delta|\lambda_n|)$: from $\ln 2/(0.1 \cdot 16) =
  0.43$ to $\ln 2/(0.001 \cdot 1) = 693$ steps.

::: {.d2l-note}
The rates are the diagonal of the HiPPO matrix, designed so that the state
holds a polynomial summary of the input's history.
:::

[Gu, Dao, Ermon, Rudra, Ré, 2020 (HiPPO); Gu, Gupta, Goel, Ré, 2022 (S4D)]{.small}
:::

::: {.slide title="Check: A Thousand-Step Memory"}
For $\lambda = -1$, which step size $\Delta$ gives a half-life of 1,000
steps?

::: {.fragment .fade-up}
$e^{-\Delta \cdot 1000} = 1/2$, so $\Delta = \ln 2 / 1000 \approx 0.0007$,
just below S4D's smallest initial step size.
:::
:::

::: {.slide title="Fixed Dynamics Make the Model a Convolution"}
$$y_t = \sum_{k=0}^{t-1} K_k\, x_{t-k} + D x_t, \qquad K_k = \sum_n c_n\, \bar a_n^k\, \bar b_n$$

![](../img/mdl-modernrnn-ssm-views.svg){width=95%}

- One kernel per channel: a sum of $N$ decaying exponentials.
- FFT: $O(T \log T)$. It convolves circularly, so pad to $2T$ and the end
  cannot wrap around.
:::

::: {.slide title="Real Exponentials Peak Late Only by Cancelling"}
Fit the readout $\mathbf{c}$ by least squares, $\Delta = 0.05$, 100 steps.
Targets: a spike at $k = 20$; a bump $e^{-(k-20)^2/32}$.

@!ssm-what-the-continuous-time-view-adds

::: {.d2l-note .rule}
One exponential peaks at $k = 0$. Sixteen match the bump (0.03), not the
spike (0.89), and only with weights up to $2 \times 10^{11}$ that cancel.
With ordinary weights: roughly when, not exactly when.
:::
:::

::: {.slide title="Convolution, Scan, and Steps Agree"}
@!ssm-the-s4d-layer-4

::: {.d2l-note .rule}
Differences of $10^{-7}$ are single-precision rounding: one function,
three forms.
:::

- Train with the FFT: no state stored per position per sequence; the
  $d \times N \times T$ powers once per batch.
- Generate with `step`: a $d \times N$ state, whatever the length.
:::

::: {.slide title="Four Layers Separate Timescales From Architecture"}
Name the first of eight symbols after $T - 1$ identical fillers; chance
1/8. Identical fillers: the symbol's trace is the only difference between
sequences.

| layer | half-lives at the start |
|---|---|
| minGRU | about one step (gates near 1/2) |
| S4D | under one to 693 steps |
| S4D, short timescales | at most 7 steps |
| minGRU, S4D timescales | 7 to 693 steps |

::: {.fragment .fade-up}
Predict: if the timescales decide, the two controls change sides.
:::
:::

::: {.slide title="Timescales, Not Architecture, Decide Long Recall"}
@!ssm-long-range-recall-2

::: {.d2l-note .rule}
The minGRU **with S4D's decays** recalls at 1,000; S4D **with short
half-lives** fails there. At 1,000 steps the initialization decides.
:::
:::

::: {.slide title="Check: Half-Lives of 7 Steps, Recall After 100"}
Why does the short-timescale S4D still recall after 100 steps?

::: {.fragment .fade-up}
- The symbol's weight after 99 steps: $e^{-9.9} \approx 5 \times 10^{-5}$.
- Small, but it carries a gradient to the step sizes and rates: training
  lengthens the slowest half-lives.
- After 999 steps: $e^{-100} \approx 10^{-44}$, below rounding error. No
  gradient, nothing moves.
:::
:::

::: {.slide title="The SSM Adds a Kernel and a Step Size"}
- One $\Delta$ per channel scales all $N$ timescales together: the knob
  the next deck makes input-dependent.
- $N$ states per channel: a learned kernel, not one exponential.
- Fixed dynamics: train as a convolution.
- The decays themselves: a gate bias does as well,
  $b = -\log(e^\Delta - 1)$.
:::

::: {.slide title="S4D Weighs an Input by Its Distance, Not Its Content"}
- $K_4$ is the same number whether the character four steps back ended a
  word or sat in its middle.
- The minGRU chooses its decays from each input; S4D cannot.

::: {.d2l-note .warn}
Can one layer have the minGRU's choice and S4D's timescales?
:::
:::

::: {.slide title="Recap"}
| | decay | trains by | recall at 100 | recall at 1,000 |
|---|---|---|---|---|
| minGRU | from the input, starts near 1/2 | scan | chance | chance |
| minGRU, S4D decays | from the input, long | scan | 1.00 | 1.00 |
| S4D, short timescales | fixed, half-lives ≤ 7 | FFT or scan | 1.00 | chance |
| S4D | fixed, half-lives to 693 | FFT or scan | 1.00 | 1.00 |

**Next:** let the input set the step size, keeping $N$ states and S4D's
timescales.
:::
