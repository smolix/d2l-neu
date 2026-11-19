# Linear Recurrences and Parallel Scans
:label:`sec_linear-recurrence`

The gated networks of :numref:`sec_lstm` can keep information for hundreds
of steps, but they still read a sequence one token after another, and each
update needs the result of the one before. That order is natural when a
model generates text, but during training the whole sequence is known in
advance. Take a sequence of 4,096 tokens. A gated network performs 4,096
updates in a row. A GPU can carry out tens of thousands of multiply-adds at
the same time, but it cannot start step 2,000 until step 1,999 has
finished, so most of it waits. A transformer layer has no such chain: it
processes all 4,096 positions in a few large matrix products. This
difference in training speed was one of the two main reasons why
transformers replaced recurrent networks for large-scale language modeling.
(The other, that attention reads every earlier token directly rather than a
summary of them, returns at the end of the chapter.) Can a recurrent network
be trained on all positions at once?

The chain has a single cause. The gates of :numref:`sec_lstm` read the
previous state, and the state passes through a nonlinearity, so step $t$
cannot be computed until step $t-1$ is known. If the gates read only the
current input, the update becomes linear in the state. Linear updates can be
grouped like the terms of a sum, and the states at all 4,096 positions can
then be computed in $\log_2 4096 = 12$ rounds of batched arithmetic. When
the model generates text, it still updates its state once per token, in
constant memory, like any recurrent network.

We start with a single number that forgets at a fixed rate and add one
thing at a time until we reach a gated layer that trains on all positions
at once, the minimal GRU of :citet:`Feng.Tung.Ahmed.ea.2024`. The trick
that makes it parallel is older than deep learning: it was worked out in
the 1980s, to compute running totals on parallel computers
:cite:`Hillis.Steele.1986`. We then ask what the change costs on text, and
what it saves against attention, which answers a question
:numref:`sec_gpt` left open: at what length does the growing cost of
attention start to matter?

```{.python .input #linear-recurrence-linear-recurrences-and-parallel-scans}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #linear-recurrence-linear-recurrences-and-parallel-scans}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Linear Recurrences
:label:`subsec_linear-recurrences`

### A Single Decaying Number

To see what a linear update gives us, we start with the smallest one.
Consider a state $h_t$, a single number, that keeps a fraction $a$ of its
value at every step and adds the new input $x_t$:

$$h_t = a\, h_{t-1} + x_t, \qquad h_0 = 0.$$
:eqlabel:`eq_lr-scalar`

This is the gated memory of :numref:`sec_lstm` with both gates held fixed.
With $a = 0.9$ and a single input $x_1 = 1$ followed by zeros, the state
reads $1, 0.9, 0.81, 0.729, \ldots$: an input from $k$ steps ago survives
with weight $a^k$. Substituting the recurrence into itself shows that this
holds for any input sequence,

$$h_t = \sum_{s=1}^{t} a^{t-s} x_s.$$
:eqlabel:`eq_lr-unrolled`

The state is a weighted sum of all past inputs, with weights that shrink
geometrically with age. Two consequences follow. First, the sum does not
mention $h_{t-1}$: every $h_t$ is determined by the inputs alone, so in
principle all of them can be computed at once. Second, $a$ sets the length
of the memory, and it sets the length of learning too. The weight of an
input halves every $\ln 2 / \ln(1/a)$ steps, about 7 steps for $a = 0.9$
and 693 for $a = 0.999$, and the gradient of $h_t$ with respect to $x_s$ is
the same weight $a^{t-s}$. The decay that forgets an input also weakens the
signal that would teach the model to use it. This is the vanishing
gradient of :numref:`sec_lstm`, a product of $t-s$ Jacobians, except that
the product is now a power of one number that we can read off directly.

### Input-Dependent Decay

A memory of one number is too small for language. With a state
$\mathbf{h}_t \in \mathbb{R}^d$ we can give each coordinate its own decay,
collected in a vector $\mathbf{a} \in (0, 1)^d$ that multiplies the state
elementwise ($\odot$), and add a vector $\mathbf{b}_t$ at every step:

$$\mathbf{h}_t = \mathbf{a} \odot \mathbf{h}_{t-1} + \mathbf{b}_t.$$

Nothing new happens here. The coordinates do not interact, so this is $d$
copies of :eqref:`eq_lr-scalar` side by side, each with its own half-life.
We call $\mathbf{b}_t$ the *write*: it is what step $t$ adds to the state,
and it will soon be a learned function of the input. Despite its letter,
the write is not a bias: the biases below are subscripted by the gate they
belong to, as in $\mathbf{b}_\textrm{z}$, and the write by the step.

A fixed decay forgets at the same rate whatever the input says. The gates of
:numref:`sec_lstm` exist to make the rate a decision: keep the state through
a run of uninformative tokens, and overwrite it when something important
arrives. We can keep that ability, and keep the update linear in the state,
by letting the decay depend on the current input:

$$
\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t,
\qquad \mathbf{a}_t = \mathbf{a}(\mathbf{x}_t), \quad
\mathbf{b}_t = \mathbf{b}(\mathbf{x}_t).
$$
:eqlabel:`eq_affine_recurrence`

Unrolling as before, the power $a^{t-s}$ becomes the product of the decays
met along the way:

$$
\mathbf{h}_t = \sum_{s=1}^{t} \Big( \prod_{r=s+1}^{t} \mathbf{a}_r \Big)
\odot \mathbf{b}_s.
$$
:eqlabel:`eq_lr-unrolled-gated`

An input survives for as long as the inputs after it choose to keep it.
Because every $\mathbf{a}_r$ lies in $(0, 1)$, the products can shrink but
never grow: no past input is ever amplified, however long the sequence.
The decays and the writes depend only on the inputs, so they can be
computed for all positions in batched matrix products before the
recurrence starts. What remains is the recurrence itself, an elementwise
multiplication followed by an addition. Such an update is *affine* in
$\mathbf{h}_{t-1}$, linear plus a constant; the literature calls these
recurrences linear, and so do we.

### The Minimal GRU
:label:`subsec_mingru`

The GRU of :numref:`sec_gru` almost has this form. Its update gate
$\mathbf{z}_t$, reset gate $\mathbf{r}_t$, and candidate state
$\tilde{\mathbf{h}}_t$ are computed as follows, with the three places where
the previous state enters marked:

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(\mathbf{W}_{\textrm{xz}} \mathbf{x}_t + \underbrace{\mathbf{W}_{\textrm{hz}} \mathbf{h}_{t-1}}_{\textrm{(1)}} + \mathbf{b}_\textrm{z}),\\
\mathbf{r}_t &= \sigma(\mathbf{W}_{\textrm{xr}} \mathbf{x}_t + \underbrace{\mathbf{W}_{\textrm{hr}} \mathbf{h}_{t-1}}_{\textrm{(2)}} + \mathbf{b}_\textrm{r}),\\
\tilde{\mathbf{h}}_t &= \tanh\big(\mathbf{W}_{\textrm{xh}} \mathbf{x}_t + \underbrace{\mathbf{W}_{\textrm{hh}} (\mathbf{r}_t \odot \mathbf{h}_{t-1})}_{\textrm{(3)}} + \mathbf{b}_\textrm{h}\big).
\end{aligned}
$$

The new state is a blend of the old state and the candidate,

$$
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t)
\odot \tilde{\mathbf{h}}_t.
$$

The blend already has the form of :eqref:`eq_affine_recurrence`, with
decay $\mathbf{z}_t$. What stands in the way are the three marked terms,
which make the decay and the write depend on $\mathbf{h}_{t-1}$.
:citet:`Feng.Tung.Ahmed.ea.2024` removed them: the update gate and the
candidate read only the input, and the reset gate, which no longer has a
previous state to filter, disappears. The result is the *minimal GRU*, or
*minGRU* (we keep a $\tanh$ on the candidate, which
:citet:`Feng.Tung.Ahmed.ea.2024` drop; an exercise compares the two):

$$
\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{z} \mathbf{x}_t + \mathbf{b}_\textrm{z}), \qquad
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_\textrm{h} \mathbf{x}_t + \mathbf{b}_\textrm{h}), \qquad
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t.
$$
:eqlabel:`eq_mingru`

This is :eqref:`eq_affine_recurrence` with decay $\mathbf{a}_t =
\mathbf{z}_t$ and write $\mathbf{b}_t = (1 - \mathbf{z}_t) \odot
\tilde{\mathbf{h}}_t$. The update gate is now the decay of a linear
recurrence: a gate value of 0.9 keeps 90% of the state and writes 10% of
the candidate. The layer is also smaller. With input and hidden width $d$,
a GRU layer has $6d^2$ weights, three input matrices and three state
matrices, and the minGRU has $2d^2$.

The minGRU gives something up. A GRU can decide to keep its state because of
what the state contains; the minGRU decides only from the current token, and
between tokens its state can only be scaled and added to. Consider a unit
that should store the first digit it meets in a text and ignore the digits
that follow. A GRU unit can do this: once its state holds a digit, the term
$\mathbf{W}_\textrm{hz} \mathbf{h}_{t-1}$ can push its update gate toward
one, and later digits are not written. A minGRU unit sees a digit arrive and
cannot tell whether its state is empty or full, so whatever it does with
the first digit, it does with the second. Stacking layers gives some of
this back. A unit in the second layer reads the states of the first, so its
gate can see whether a digit has already been stored, and the
nonlinearities between layers do not block parallel evaluation. How much is
lost on real text is a question for an experiment, which we run once we can
train the layer quickly.

## Parallel Scans
:label:`subsec_parallel-scans`

### Composing Affine Steps

The unrolled sum :eqref:`eq_lr-unrolled-gated` no longer mentions
$\mathbf{h}_{t-1}$, but evaluating it term by term costs $t$ terms for
state $t$, about $T^2/2$ in all: the same square in $T$ that makes
attention expensive. We want the parallelism without the square.

Computing $\mathbf{h}_1, \ldots, \mathbf{h}_T$ from the coefficients
$(\mathbf{a}_t, \mathbf{b}_t)$ is a relative of a familiar problem: compute
all running totals $s_t = x_1 + \cdots + x_t$ of a sequence, its *prefix
sums*. Prefix sums look sequential, yet they can be computed in parallel,
because addition is associative: the total $(x_1 + x_2) + (x_3 + x_4)$ can be
split into two halves that are computed independently and combined at the
end. Any associative operation allows the same trick, and composing the
steps of an affine recurrence is one.

Represent step $t$ by its coefficient pair $(\mathbf{a}_t, \mathbf{b}_t)$,
which stands for the map $\mathbf{h} \mapsto \mathbf{a}_t \odot \mathbf{h} +
\mathbf{b}_t$. Applying step 1 and then step 2 gives
$\mathbf{a}_2 \odot (\mathbf{a}_1 \odot \mathbf{h} + \mathbf{b}_1) +
\mathbf{b}_2$, which is again a map of the same form. We write the
composition as

$$
(\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2) =
(\mathbf{a}_1 \odot \mathbf{a}_2,\ \mathbf{a}_2 \odot \mathbf{b}_1 +
\mathbf{b}_2).
$$
:eqlabel:`eq_scan_combine`

Read from left to right, $\bullet$ means "this step, then that one". It is
associative because composing functions is associative, and the composite of
any run of steps is again a single pair. This matters as much as
associativity: a block of a thousand steps costs no more to store or to
combine than a single step. As an example, two steps with $a_1 = a_2 = 0.5$,
$b_1 = 1$, and $b_2 = 2$ combine into $(0.25,\ 0.5 \cdot 1 + 2) = (0.25,
2.5)$. Running them one after the other from $h_0 = 0$ gives $h_1 = 1$ and
$h_2 = 0.5 \cdot 1 + 2 = 2.5$, as the composite says.

Composing the first $t$ pairs gives the map that carries $\mathbf{h}_0$ to
$\mathbf{h}_t$. With $\mathbf{h}_0 = \mathbf{0}$ the state $\mathbf{h}_t$ is
the $\mathbf{b}$-component of that composite, so all states together are the
prefix "sums" of the pairs under $\bullet$. Computing all prefixes of a
sequence under an associative operation is called a *scan*. The
$\mathbf{a}$-component of each prefix is the running product of the
decays. We never need it as a result, but the scan must carry it: to
append a block of steps to what came before, the combine has to know by
how much the block shrinks everything earlier.

### The Doubling Schedule

:numref:`fig_scan_tree` shows a parallel schedule for eight steps. In the
first round, every position combines with its predecessor, so it holds the
composite of two steps. In the second round, every position combines with
the position two back and then holds four steps; in the third, with the
position four back, and then holds eight. After $\lceil \log_2 T \rceil$
rounds every position holds its full prefix. Each round is one elementwise
multiply-add over the whole sequence, the kind of work a GPU does well.

![A parallel scan on eight steps. Each box shows which steps its value
combines, $(1{:}4)$ meaning steps 1 to 4. In every round each position
combines with the position a fixed stride back (blue arrows), and the stride
doubles from round to round; dashed arrows copy a value unchanged. Teal
boxes hold complete prefixes. After $\log_2 8 = 3$ rounds every position
does.](../img/mdl-modernrnn-scan-tree.svg)
:label:`fig_scan_tree`

This schedule, due to :citet:`Hillis.Steele.1986`, performs about
$T \log_2 T$ combines, a factor $\log_2 T$ more than the $T - 1$ steps of a
loop, in exchange for $\lceil \log_2 T \rceil$ rounds instead of $T - 1$.
On a GPU, where the combines of one round run side by side, the rounds
matter more than the combines.

A schedule that does about as many combines as the loop is called
*work-efficient*. The one of :citet:`Blelloch.1990` needs about $2T$
combines in about $2 \log_2 T$ rounds. It first combines neighboring pairs,
then neighboring pairs of pairs, each round on half as many elements as the
round before, and then fills in the missing prefixes on the way back down.
Counted in passes over memory, the doubling schedule reads and writes the
whole sequence in every round, $\lceil \log_2 T \rceil$ times. Because the
rounds of the work-efficient schedule shrink, it touches the sequence only a
few times in all, however long the sequence is. Once the sequence is long
enough to keep the GPU busy, these passes set the time. The doubling
schedule is the simpler of the two, and the one we write by hand; library
scans are work-efficient.

### Implementation

`linear_scan` takes the decays and the writes of
:eqref:`eq_affine_recurrence`, with time along the second axis, and an
optional initial state, and returns all states. For the backward pass,
automatic differentiation keeps the intermediate results of the scan. For
the doubling schedule these are the tensors of every round, so the memory
of training grows with the number of rounds as well as with the length.

:begin_tab:`pytorch`
PyTorch has no scan in its stable interface, so we write one,
`associative_scan`. It takes the combine as a function and a tuple of
tensors with time along the second axis, and runs the doubling schedule,
combining the sequence in every round with a copy of itself shifted by the
stride. `linear_scan` passes it the combine :eqref:`eq_scan_combine`.
:end_tab:

:begin_tab:`jax`
JAX provides an associative scan in its library,
`jax.lax.associative_scan`, which takes the combine :eqref:`eq_scan_combine`
as a function and runs a work-efficient schedule, about $2T$ combines in
all.
:end_tab:

```{.python .input #ssm-implementation-1}
%%tab pytorch
def associative_scan(combine, elems):  #@save
    """All prefixes of elems under an associative combine; time is axis 1."""
    s = 1
    # Autograd keeps the tensors of every round for the backward pass
    while s < elems[0].shape[1]:  # Combine every position with the one s back
        earlier = tuple(x[:, :-s] for x in elems)
        later = tuple(x[:, s:] for x in elems)
        elems = tuple(torch.cat([x[:, :s], y], 1)
                      for x, y in zip(elems, combine(earlier, later)))
        s *= 2
    return elems

def linear_scan(a, b, h0=None):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t; time is the second axis."""
    a = a.expand_as(b)  # Decays may omit axes on which they do not vary
    if h0 is not None:  # Fold the initial state into the first step
        b = torch.cat([a[:, :1] * h0.unsqueeze(1) + b[:, :1], b[:, 1:]], 1)
    def combine(earlier, later):
        (a1, b1), (a2, b2) = earlier, later
        return a1 * a2, a2 * b1 + b2
    return associative_scan(combine, (a, b))[1]
```

```{.python .input #ssm-implementation-1}
%%tab jax
def linear_scan(a, b, h0=None):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t; time is the second axis."""
    a = jnp.broadcast_to(a, b.shape)  # Decays may omit axes; broadcast them
    if h0 is not None:  # Fold the initial state into the first step
        b = b.at[:, 0].add(a[:, 0] * h0)
    def combine(earlier, later):
        (a1, b1), (a2, b2) = earlier, later
        return a1 * a2, a2 * b1 + b2
    # Autograd keeps the intermediate results of the scan for the backward pass
    return jax.lax.associative_scan(combine, (a, b), axis=1)[1]
```

We check the scan against a loop that applies the recurrence one step at a
time, on random decays in $(0, 1)$ and a length that is not a power of two.

```{.python .input #ssm-implementation-2}
%%tab pytorch
def sequential_scan(a, b):
    h, states = torch.zeros_like(b[:, 0]), []
    for a_t, b_t in zip(a.unbind(1), b.unbind(1)):
        h = a_t * h + b_t
        states.append(h)
    return torch.stack(states, 1)

a, b = torch.rand(4, 100, 8), torch.randn(4, 100, 8)
print(f'max deviation: '
      f'{(linear_scan(a, b) - sequential_scan(a, b)).abs().max():.1e}')
```

```{.python .input #ssm-implementation-2}
%%tab jax
def sequential_scan(a, b):
    def step(h, ab):
        h = ab[0] * h + ab[1]
        return h, h
    # lax.scan steps through the leading axis, so time goes first
    ab = (jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1))
    return jnp.swapaxes(jax.lax.scan(step, jnp.zeros_like(b[:, 0]), ab)[1], 0, 1)

key_a, key_b = jax.random.split(d2l.get_key())
a = jax.random.uniform(key_a, (4, 100, 8))
b = jax.random.normal(key_b, (4, 100, 8))
print(f'max deviation: '
      f'{jnp.abs(linear_scan(a, b) - sequential_scan(a, b)).max():.1e}')
```

The largest deviation is below $10^{-6}$, the size of rounding error in
32-bit floating point. The scan adds the same terms as the loop, only
grouped differently, so its results differ in the last digits and nowhere
else.

## A Linear Recurrent Language Model

### The minGRU on Text

The minGRU :eqref:`eq_mingru` is a few lines around `linear_scan`. Its
method `coefficients` computes the decays and writes of all positions with
two matrix products, and the forward pass hands them to the scan. The layer
maps a minibatch of sequences to the states at every step, so it can serve
as the sequence mixer of `d2l.SeqLM`, the residual language model of
:numref:`sec_lstm`. That model is an embedding, a stack of blocks that each
apply a mixer and a position-wise MLP, and a prediction head.

```{.python .input #ssm-a-linear-recurrent-language-model-1}
%%tab pytorch
class MinGRU(nn.Module):  #@save
    """The minimal GRU: its gate and candidate read only the input."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.W_z = nn.Linear(num_inputs, num_hiddens)  # Update gate (decay)
        self.W_h = nn.Linear(num_inputs, num_hiddens)  # Candidate state
        d2l.init_lecun(self)

    def coefficients(self, X):
        """Decays a_t and writes b_t of the linear recurrence."""
        Z = torch.sigmoid(self.W_z(X))
        return Z, (1 - Z) * torch.tanh(self.W_h(X))

    def forward(self, X):  # X: (batch_size, num_steps, num_inputs)
        return linear_scan(*self.coefficients(X))
```

```{.python .input #ssm-a-linear-recurrent-language-model-1}
%%tab jax
class MinGRU(nnx.Module):  #@save
    """The minimal GRU: its gate and candidate read only the input."""
    def __init__(self, num_inputs, num_hiddens, *, rngs):
        self.W_z = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)  # Decay
        self.W_h = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)

    def coefficients(self, X):
        """Decays a_t and writes b_t of the linear recurrence."""
        Z = jax.nn.sigmoid(self.W_z(X))
        return Z, (1 - Z) * jnp.tanh(self.W_h(X))

    def __call__(self, X):  # X: (batch_size, num_steps, num_inputs)
        return linear_scan(*self.coefficients(X))
```

Does taking the state out of the gates cost anything on text? We answer on
the text benchmark of :numref:`sec_lstm`: byte-level WikiText-103 in windows
of 128 bytes, 64 windows per step, trained for 1,000 steps with the recipe
of the GPT of :numref:`sec_gpt`. Every model is a `d2l.SeqLM` stack of
four blocks. `d2l.text_table` trains each model twice and prints its
parameter count, the validation loss of both runs in nats per byte, the
losses at early, middle, and late positions of the window, and the
milliseconds per training step.

A fair answer takes three models. The GRU and the minGRU at width 256
differ only in the mixer: the gates and the candidate of the minGRU read
only the input. The change removes two thirds of the mixer's weights, but
the embedding, the MLPs, and the head are the same in both models, so the
whole minGRU model has 2.76 million parameters against the GRU model's 3.8
million. A smaller model may lose simply for being smaller, so the third
model is the minGRU model at width 302 throughout, which gives it as many
parameters as the GRU model.

:begin_tab:`pytorch`
As in :numref:`sec_gpt`, the training cell lets 32-bit matrix products use
TF32, which rounds their factors to a shorter mantissa and runs faster on
recent GPUs.
:end_tab:

```{.python .input #linear-recurrence-a-linear-recurrent-language-model-1}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32

def text_model(mixer, d):  # 256 byte values; four blocks of width d
    return d2l.SeqLM(256, d, [mixer(d) for _ in range(4)]).to(d2l.try_gpu())

gru = lambda d: d2l.Recurrent(d2l.init_lecun(nn.GRU(d, d, batch_first=True)))
min_gru = lambda d: MinGRU(d, d)
arms = {'GRU': (gru, 256), 'minGRU': (min_gru, 256),
        'minGRU, d=302': (min_gru, 302)}  # As many parameters as the GRU
```

```{.python .input #linear-recurrence-a-linear-recurrent-language-model-1}
%%tab jax
def text_model(mixer, d):  # 256 byte values; four blocks of width d
    rngs = nnx.Rngs(d2l.get_key())
    return d2l.SeqLM(256, d, [mixer(d, rngs) for _ in range(4)], rngs=rngs)

lecun = nnx.initializers.lecun_normal()  # For the recurrent matrices
gru = lambda d, rngs: nnx.RNN(
    nnx.GRUCell(d, d, recurrent_kernel_init=lecun, rngs=rngs), unroll=8)
min_gru = lambda d, rngs: MinGRU(d, d, rngs=rngs)
arms = {'GRU': (gru, 256), 'minGRU': (min_gru, 256),
        'minGRU, d=302': (min_gru, 302)}  # As many parameters as the GRU
```

```{.python .input #linear-recurrence-a-linear-recurrent-language-model-2}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
trained = d2l.text_table(text_model, arms, data)
model = trained['minGRU'][0]  # The model we look inside below
```

Two runs of the same model end within 0.01 nats per byte of each other,
most of them within a few thousandths, so the differences between the
models are not noise. At the GRU's parameter count the minGRU ends about
0.03 nats per byte behind the GRU, and at the same width about 0.05 behind.
The gap is not spread evenly over the window. At equal parameter counts it
is about 0.02 in the first 16 bytes and about 0.03 from byte 16 on. The
minGRU falls behind once there is context to manage, which is where a gate
that can see the state should help.

How bad is that? At width 256 the minGRU has the same 2.76 million
parameters as the plain tanh RNN that :numref:`sec_lstm` trained in the
same stack, and it ends within 0.01 of that network, at about 1.35 nats per
byte. Widened to the GRU's size, the two are again within 0.01 of each
other, and the GRU is about 0.03 ahead of both. On this text, gates that
cannot see the state do no better than a plain recurrence with no gates at
all. Why, then, not use the plain RNN? Its state passes through a $\tanh$
at every step, so it cannot be scanned, and its chain of steps grows with
the length like the GRU's. The minGRU's case is that it trains on all
positions at once. At 128 bytes its training step is already cheaper than
the GRU's, and :numref:`subsec_linear-recurrence-cost` measures how the
difference grows with the length.

### Inside the Trained Model

Because the minGRU's decay is its update gate, we can look inside the
trained model, the minGRU of width 256 from the first run. How much of its
skill is memory? We reset the state of every layer to zero before every
step, so that $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot
\tilde{\mathbf{h}}_t$ and each prediction depends on the current byte
alone, passed through the layers.

```{.python .input #ssm-a-linear-recurrent-language-model-4}
%%tab pytorch
class StateReset(nn.Module):
    """A minGRU layer whose state is reset to zero before every step."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, X):
        return self.layer.coefficients(X)[1]

layers = model.mixers
loss = np.mean(d2l.loss_by_position(model, data))
model.mixers = nn.ModuleList([StateReset(layer) for layer in layers])
reset = np.mean(d2l.loss_by_position(model, data))
model.mixers = layers
print(f'validation loss: {loss:.3f} nats per byte, '
      f'with the state reset at every step: {reset:.3f}')
```

```{.python .input #ssm-a-linear-recurrent-language-model-4}
%%tab jax
class StateReset(nnx.Module):
    """A minGRU layer whose state is reset to zero before every step."""
    def __init__(self, layer):
        self.layer = layer

    def __call__(self, X):
        return self.layer.coefficients(X)[1]

layers = model.mixers
loss = np.mean(d2l.loss_by_position(model, data))
model.mixers = nnx.List([StateReset(layer) for layer in layers])
reset = np.mean(d2l.loss_by_position(model, data))
model.mixers = layers
print(f'validation loss: {loss:.3f} nats per byte, '
      f'with the state reset at every step: {reset:.3f}')
```

Without its memory the model's loss rises from just under 1.35 to about
2.48 nats per byte, from a perplexity of about 3.8 to about 12. The model
was never trained to predict from one byte, so the reset loss overstates
what a one-byte model could reach. It cannot overstate it by much, though.
A model that sees one byte is a bigram model, and the bigram model of
WikiText-103's training bytes, counted in :numref:`sec_gpt`, scores 2.45
nats per byte even on the bytes it was counted on. More than a nat per byte
of the minGRU's skill comes from what it remembers.

How long, then, does the first layer hold its state? A gate value $g$ keeps
a fraction $g$ of the state per step, which by the half-life of
:eqref:`eq_lr-unrolled` is one step for $g = 0.5$, 7 for $0.9$, and 69 for
$0.99$. Before training, the gates spread around one half: their biases
start at zero, so most gates keep the state for a step or two. To read the
gates, of the trained model and of an untrained one for comparison, we need
the input of the first layer: the token embeddings, normalized as the first
block normalizes them. We give `SeqLM` a method for it.

```{.python .input #ssm-a-linear-recurrent-language-model-5}
%%tab pytorch, jax
@d2l.add_to_class(d2l.SeqLM)  #@save
def first_mixer_input(self, X):
    """What the first mixer reads: the normalized token embeddings."""
    return self.norms[0](self.embedding(X))
```

```{.python .input #ssm-a-linear-recurrent-language-model-6}
%%tab pytorch
def first_gates(model, X):  # Gates of the first layer
    return model.mixers[0].coefficients(model.first_mixer_input(X))[0]

X, _ = next(iter(data.val_dataloader()))  # Validation text
X = X.to(next(model.parameters()).device)
with torch.inference_mode():
    Z_init = first_gates(text_model(min_gru, 256), X)  # Untrained
    Z = first_gates(model, X)
print('gate above  half-life in steps  share at initialization  trained')
for g in (0.5, 0.9, 0.99):
    print(f'{g:10}  {math.log(2) / -math.log(g):17.0f}  '
          f'{(Z_init > g).float().mean():23.1%}  '
          f'{(Z > g).float().mean():7.1%}')
```

```{.python .input #ssm-a-linear-recurrent-language-model-6}
%%tab jax
def first_gates(model, X):  # Gates of the first layer
    return model.mixers[0].coefficients(model.first_mixer_input(X))[0]

X = jnp.asarray(next(iter(data.val_dataloader()))[0])  # Validation text
Z_init = first_gates(text_model(min_gru, 256), X)  # Untrained
Z = first_gates(model, X)
print('gate above  half-life in steps  share at initialization  trained')
for g in (0.5, 0.9, 0.99):
    print(f'{g:10}  {math.log(2) / -math.log(g):17.0f}  '
          f'{(Z_init > g).mean():23.1%}  {(Z > g).mean():7.1%}')
```

Training has left the two ends of the table alone. About half of the gate
values exceed 0.5, before training and after, and almost none reaches 0.99.
All the change is in between. At initialization about one gate in a hundred
exceeds 0.9; after training about one in six does, and for those the
memory has grown from a byte or two to a half-life of about 7 bytes or
more. Nothing in training asked for much more: the windows are 128 bytes
long. Each gate value is computed from the current byte, so how long an
input survives depends on the bytes that follow it, as
:eqref:`eq_lr-unrolled-gated` describes; an exercise asks how much the gates
of a unit vary from byte to byte.

## Cost against Attention
:label:`subsec_linear-recurrence-cost`

The minGRU trains in a number of rounds that grows with $\log_2 T$, where
the GRU needs $T$ steps. Attention has no chain at all, but every position
reads every earlier one, so its work grows with the square of the length.
At what length does that square start to matter? :numref:`sec_gpt` left
the question open after comparing a recurrent network with a GPT at 128
bytes. We answer it first by counting and then by measuring.

### Work, Depth, and Memory

:numref:`tab_linear-recurrence-cost` counts, for one layer of each kind,
the training work, the steps that must run one after another, and the
numbers the layer carries from one token to the next when it generates
text.

:Cost of one layer of width $d$ on $T$ tokens, up to constant factors.
:label:`tab_linear-recurrence-cost`

| Layer | Training work | Steps in a row | Memory at generation |
| :-- | :-- | :-- | :-- |
| GRU, one step at a time | $T d^2$ | $T$ | $d$ |
| minGRU, doubling scan | $T d^2 + T d \log_2 T$ | $\log_2 T$ | $d$ |
| minGRU, work-efficient scan | $T d^2 + T d$ | $2 \log_2 T$ | $d$ |
| Causal attention | $T d^2 + T^2 d$ | a few matrix products | $2 T d$ |

Both recurrences do work of order $d^2$ per token. The minGRU's matrix
products all read the input and run for every position at once, which
leaves only the scan in a row. Three of the GRU's six matrix products read
the previous state and must run one step after another: $T$ steps in a
row, against a logarithm for the scan. Attention needs only a few steps in
a row, but its work has a term in $T^2$. For a layer of width $d$,
:numref:`sec_attention-at-scale` counted $8 T d^2$ FLOPs in the projections
and $4 T^2 d$ in the scores and the mixing; a causal mask lets a fused
kernel skip about half of the second term. The last column is the cost at
generation. Every recurrence carries the same few numbers per
channel whatever the length, while attention keeps a key and a value for
every earlier token, the cache of :numref:`sec_kv-cache`. The scan changes
how we train, not how we generate: at inference the minGRU still applies
:eqref:`eq_affine_recurrence` once per token, in constant memory.

### Measured Cost

Let us predict what a measurement should show before we make it. We time
a forward and a backward pass through one layer of each kind, at width 256
and in 16-bit floats, with the `d2l.Benchmark` of
:numref:`subsec_async_dispatch`, which times each call after a warm-up call
and waits for the GPU to finish. Every call processes 65,536 tokens, cut
into sequences of length $T$ from 512 to 16,384: 128 sequences of 512
tokens, down to four of 16,384. With the number of tokens fixed, a layer
whose work is proportional to the number of tokens takes the same time at
every length.

That settles the scan: its time should stay nearly flat, apart from the
extra rounds of a doubling schedule. Attention does $8d^2$ FLOPs per token
in its projections and about $2Td$ in its causal scores and mixing, which
are equal at $T = 4d = 1{,}024$. Below that length the projections dominate
and its time should grow slowly; above it, the time should approach a
doubling with every doubling of $T$. The GRU's work is also proportional to
the number of tokens, but half of it comes in $T$ steps in a row, each on
fewer sequences, and every step has a latency that no amount of parallel
hardware removes. Its time should grow nearly in proportion to $T$.

:begin_tab:`pytorch`
`nn.GRU` runs cuDNN's fused recurrent kernel, which still takes one step
after another. The minGRU is compiled with `torch.compile`, as
:numref:`sec_gpt` compiles the GPT; uncompiled, every round of the doubling
scan would launch several kernels that each read and write the whole
sequence. `d2l.CausalAttention` calls a fused kernel through
`F.scaled_dot_product_attention`.
:end_tab:

:begin_tab:`jax`
The GRU is `nnx.RNN`, unrolled eight steps per loop iteration, as in
:numref:`sec_lstm`. Attention calls `jax.nn.dot_product_attention` with
`implementation='cudnn'`, cuDNN's fused kernel, which never stores the
$T \times T$ attention scores. The default implementation forms them in
32-bit floats: 16 GiB for one call at $T = 16{,}384$.
:end_tab:

```{.python .input #linear-recurrence-cost-against-attention-1}
%%tab pytorch
d, tokens = 256, 65536  # Width; tokens per call, in sequences of length T
device, half = d2l.try_gpu(), torch.float16  # 16-bit floats throughout
timed = {'GRU': d2l.Recurrent(nn.GRU(d, d, batch_first=True)),
         'minGRU': torch.compile(MinGRU(d, d)),
         'attention': d2l.CausalAttention(d, num_heads=4)}
for layer in timed.values():
    layer.to(device, half)

def time_call(layer, T):
    """Milliseconds for a forward and backward pass over tokens / T
    sequences of length T."""
    X = torch.randn(tokens // T, T, d, device=device, dtype=half,
                    requires_grad=True)
    return 1000 * d2l.Benchmark(lambda: layer(X).float().sum().backward(),
                                warmup=1, repeats=3).time
```

```{.python .input #linear-recurrence-cost-against-attention-1}
%%tab jax
class FusedAttention(d2l.CausalAttention):
    """Causal attention through cuDNN's fused kernel."""
    @staticmethod
    def attend(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                            implementation='cudnn')

d, tokens = 256, 65536  # Width; tokens per call, in sequences of length T
half, rngs = jnp.float16, nnx.Rngs(d2l.get_key())  # 16-bit floats
timed = {'GRU': nnx.RNN(nnx.GRUCell(d, d, recurrent_kernel_init=lecun,
                                    param_dtype=half, rngs=rngs), unroll=8),
         'minGRU': MinGRU(d, d, rngs=rngs),
         'attention': FusedAttention(d, num_heads=4, rngs=rngs)}
for layer in timed.values():  # Every weight in 16 bits
    params = nnx.state(layer, nnx.Param)
    nnx.update(layer, jax.tree.map(lambda p: p.astype(half), params))

@nnx.jit
def forward_backward(layer, X):
    loss = lambda layer, X: layer(X).astype(jnp.float32).sum()
    return nnx.grad(loss, argnums=(0, 1))(layer, X)

def time_call(layer, T):
    """Milliseconds for a forward and backward pass over tokens / T
    sequences of length T."""
    X = jax.random.normal(d2l.get_key(), (tokens // T, T, d), half)
    return 1000 * d2l.Benchmark(lambda: forward_backward(layer, X),
                                warmup=1, repeats=3).time
```

```{.python .input #linear-recurrence-cost-against-attention-2}
%%tab pytorch, jax
lengths = [512, 1024, 2048, 4096, 8192, 16384]
times = {name: [time_call(layer, T) for T in lengths]
         for name, layer in timed.items()}
print('T        ', *(f'{T:>7d}' for T in lengths))
for name, ms in times.items():
    print(f'{name:9s}', *(f'{t:7.1f}' for t in ms))
d2l.plot(lengths, list(times.values()), 'sequence length',
         'milliseconds per call', legend=list(times), xscale='log',
         yscale='log', figsize=(4.5, 3))
```

The curves have the shapes the counts predict. The GRU is the slowest layer
at every length, and its time grows by 1.6 to 2.1 times with every doubling
of the length, more than fifteen times from 512 to 16,384 tokens. That is
not quite in proportion, because its input products are part of every call
and cost the same at every length. The scan's time hardly moves; it grows
by less than half over the whole range. Attention's time grows slowly at
first and then, from about 4,096 tokens on, by about 1.7 times with each
doubling of the length, on its way to the factor of two that the quadratic
term predicts. At 16,384 tokens the scan is the fastest of the three and
the GRU more than ten times slower than either of the others.

The count is less exact about where attention bends. It puts the turn at
1,024 tokens and predicts that attention's time grows by a third from 512 to
1,024 tokens. The measured time grows by at most a fifth there, and at every
later doubling it stays a little below the count. The FLOP count leaves out
time that every token costs whatever the length, most likely reading and
writing activations and launching kernels, and that time dilutes the
quadratic term. A FLOP count is right about the shape of a curve and only
roughly right about where it bends.

At short lengths, which of the scan and attention is faster depends on the
scan's schedule. There a doubling scan spends most of its time on its
$\lceil \log_2 T \rceil$ passes over the sequence, while a work-efficient
scan makes only a few.

:begin_tab:`pytorch`
`associative_scan` runs the doubling schedule, 9 rounds at 512 tokens and
14 at 16,384, and its time grows with the number of rounds. It is slower
than attention up to 2,048 tokens; the two cross near 4,096.
:end_tab:

:begin_tab:`jax`
`jax.lax.associative_scan` runs a work-efficient schedule, so the scan's
time stays nearly flat, below attention's at every length measured.
:end_tab:

This answers the question of :numref:`sec_gpt` for a layer of width 256 on
this GPU. Attention's quadratic term matches its projections near 1,024
tokens, and from a few thousand tokens on it sets the cost: every doubling
of the context then nearly doubles the cost of each token, while a token of
the scan costs about the same at any length. At the 128 bytes of our text
benchmark attention is not the expensive layer. In a wider layer the
projections weigh more, $d^2$ against $Td$ per token, so the turn comes
later, in proportion to the width.

:citet:`Feng.Tung.Ahmed.ea.2024` asked whether RNNs were all we needed. On
byte-level text the minGRU gives the quality of a plain recurrent network, a
little below the GRU's, at a training cost per token that hardly grows with
the length. Its memory is another matter. Nothing in the layer says how long
its memories should be: gradient descent starts from gates near one half and
is left to find the rest. In windows of 128 bytes it found memories of tens
of bytes, which is all the windows asked for. Whether it would find a memory
of a thousand steps when a task needs one is a question this run cannot
answer; :numref:`sec_ssm` takes it up, and chooses the timescales before
training starts.

## Summary

A recurrence whose update is linear in its state, $\mathbf{h}_t =
\mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$, keeps a weighted sum
of its past inputs, with weights that are products of decays, and the same
products scale its gradients. Because affine steps compose into affine
steps of the same size, a parallel scan computes all states in about
$\log_2 T$ rounds instead of $T$ steps. The minGRU obtains such a recurrence
from the GRU by letting its gate and candidate read only the input. On
byte-level text it ends about 0.03 nats per byte behind a GRU of the same
size, level with a plain recurrent network. Its advantage over both is
cost, since neither of their updates can be scanned. Per token, the scan's
cost barely changes with the length, while attention's grows in proportion
to the length once the sequence is many times longer than the layer is
wide: from a few thousand tokens on at width 256.

## Exercises

1. **Closure under composition.** A parallel scan needs more than
   associativity: the composite of any run of steps must be described by as
   many numbers as a single step, so that every combine costs the same.
    1. Generalize :eqref:`eq_scan_combine` to the recurrence
       $\mathbf{h}_t = \mathbf{A}_t \mathbf{h}_{t-1} + \mathbf{b}_t$ with
       dense matrices $\mathbf{A}_t \in \mathbb{R}^{d \times d}$, keeping
       the factors of each product in the right order. Count the
       multiply-adds of one combine and of one sequential step. Compare the
       total work of the doubling scan, about $T \log_2 T$ combines, with
       that of the loop at $T = 4096$ and $d = 64$. What does the
       comparison suggest about dense transition matrices?
    1. For the scalar step $h \mapsto \tanh(w_t h + c_t)$, show that the
       composite of two steps with $w_1, w_2 \neq 0$ is not of the form
       $h \mapsto \tanh(w h + c)$ for any $w$ and $c$. What would a scan
       have to store to represent the composite of $k$ steps?

1. [code] **The direct route.** For scalar decays, the unrolled sum
   :eqref:`eq_lr-unrolled-gated` can be evaluated without a scan, from
   cumulative products $P_t = a_1 a_2 \cdots a_t$ and one cumulative sum.
    1. Show that $h_t = P_t \sum_{s=1}^{t} b_s / P_s$. Implement it with a
       library `cumprod` and `cumsum` and compare it with `linear_scan` in
       32-bit floats on decays drawn near 0.9. At what length does it fail,
       and why does the scan not fail there?
    1. :citet:`Feng.Tung.Ahmed.ea.2024` repair the formula by working with
       logarithms of the decays and of the writes, replacing the sum by a
       cumulative log-sum-exp. Implement the repair. What does it require
       of the signs of the writes, and how do the authors arrange it?

1. [code] **Candidate nonlinearity.** `MinGRU` keeps a $\tanh$ on the
   candidate $\tilde{\mathbf{h}}_t$, whereas
   :citet:`Feng.Tung.Ahmed.ea.2024` use a linear candidate.
    1. Starting from $\mathbf{h}_0 = \mathbf{0}$, write each coordinate of
       $\mathbf{h}_t$ as a weighted sum of the candidates
       $\tilde{\mathbf{h}}_1, \ldots, \tilde{\mathbf{h}}_t$. Show that the
       weights are nonnegative and sum to at most one, and derive a bound on
       $|\mathbf{h}_t|$ that holds at every length, with and without the
       $\tanh$.
    1. Remove the $\tanh$, train the language model of this section twice
       with `d2l.train_text`, and compare its validation loss with the two
       runs of `MinGRU`. Is the difference larger than the spread between
       runs? Relate what you observe to your bound.

1. [code] **A minimal LSTM.** Apply the same cut to the LSTM of
   :numref:`sec_lstm`: compute its input, forget, and output gates and its
   input node from $\mathbf{x}_t$ alone, and keep its cell update
   $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot
   \tilde{\mathbf{c}}_t$.
    1. Show that only the cell needs a scan. Implement the layer as a mixer,
       train it twice with `d2l.train_text`, and compare its validation loss
       with the two runs of `MinGRU`, as in the previous exercise.
    1. Unlike the minGRU, this layer does not tie its input gate to its
       forget gate. The exercise on output squashing in :numref:`sec_lstm`
       showed that if $|\tilde{\mathbf{c}}_t| < 1$ and every forget gate
       stays below $f_\textrm{max} < 1$, the cell stays below
       $1 / (1 - f_\textrm{max})$ in magnitude. Give gate values under
       which the cell approaches this bound. What bound holds if the gates
       are replaced by $\mathbf{f}_t / (\mathbf{f}_t + \mathbf{i}_t)$ and
       $\mathbf{i}_t / (\mathbf{f}_t + \mathbf{i}_t)$, as
       :citet:`Feng.Tung.Ahmed.ea.2024` propose?

1. [code] **State passing.** `linear_scan` accepts an initial state, so a
   long sequence can be processed in pieces.
    1. Split a sequence of length $T$ into chunks of length $\tau$, run the
       scan on each chunk starting from the final state of the previous
       one, and check that the concatenated states match a single call on
       the whole sequence.
    1. Now stop gradients through the carried state at every chunk
       boundary. The states do not change. Which contributions to the
       gradient of the loss at step $t$ disappear, and which dependencies
       can the trained model still use when it generates text? Compare with
       truncated backpropagation through time (:numref:`sec_bptt`).

1. [code] **How gates vary.** In the trained language model, measure for
   each unit of the first layer how much its gate varies across the bytes
   of a validation window. Do some units hold their state on some bytes and
   overwrite it on others? Compare the gates at spaces and punctuation with
   the gates inside words.

1. [code] **Work-efficient scans.** ● A recursive schedule does less work
   than the doubling schedule: combine adjacent pairs, which halves the
   sequence; scan the half-length sequence recursively, which yields every
   second prefix; then obtain each remaining prefix with one more combine.
    1. For $T = 2^k$, count the combines and the rounds of combines of each
       schedule, and compare both with the $T - 1$ steps of the loop at
       $T = 16384$.
    1. Implement both schedules with slicing, check both against
       `sequential_scan`, and time them at the lengths of the measurement
       in :numref:`subsec_linear-recurrence-cost`. Does the ratio of times
       follow the ratio of combines? Explain why it might not at short
       lengths.

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.3]{.kicker}

Linear recurrences and parallel scans<br>
**a decaying number · affine steps compose · the minGRU on text · cost against attention**
:::
:::

::: {.slide title="A GRU Waits 4,095 Times on a 4,096-Token Sequence"}
[The problem]{.kicker}

- A GPU does tens of thousands of multiply-adds at once.
- A gated network cannot start step 2,000 before step 1,999 is done.
- A transformer layer processes all 4,096 positions in a few matrix products.

::: {.d2l-note}
The chain has one cause: the gates read $\mathbf{h}_{t-1}$ through a
nonlinearity. Make the update **linear in the state**, and
$\log_2 4096 = 12$ rounds suffice.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- unroll a linear recurrence into a weighted sum of its inputs;
- compose affine steps and compute all states in $\lceil\log_2 T\rceil$
  rounds;
- turn a GRU into a minGRU and say what it loses on text;
- predict how a GRU, a scan, and attention slow down as $T$ grows.

**Running example:** byte-level WikiText-103, windows of 128 bytes, four
blocks of width 256; the GRU reaches 1.30 nats per byte.
:::

::: {.slide title="Linear Recurrences Sit Between Gates and State Space Models"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Gated Recurrent Networks | gates as learned decay |
| Sequence-to-Sequence Learning | one state between two networks |
| **Linear Recurrences and Parallel Scans** | **$h_t = a_t h_{t-1} + b_t$, the scan, the minGRU** |
| State Space Models | decays from rates; kernels; convolution |
| Selective State Space Models | the input sets the step size |
| Matrix-State Recurrences | a state of key and value pairs |
| The Delta Rule | correct a pair instead of adding |
| Hybrid Architectures | a little attention back |
:::

::: {.col .narrow}
- **Uses:** the GRU, `SeqLM` and the byte benchmark, attention's FLOP
  count.
- **Skips:** how fast a state should forget (next deck).
:::
:::
:::

::: {.slide title="A Linear Recurrence Is a Weighted Sum of Its Inputs"}
$$h_t = a\,h_{t-1} + x_t \quad\Longrightarrow\quad h_t = \sum_{s \le t} a^{t-s} x_s$$

- $a = 0.9$, one input $x_1 = 1$: the state reads $1, 0.9, 0.81, \ldots$
- No $h_{t-1}$ on the right: every state follows from the inputs alone.
- Half-life $\ln 2/\ln(1/a)$: 7 steps at $a = 0.9$, 693 at $0.999$.

::: {.d2l-note}
$\partial h_t / \partial x_s = a^{t-s}$: the decay that forgets an input
also silences the gradient that would teach the model to use it.
:::
:::

::: {.slide title="Input-Chosen Decays Keep the Update Affine"}
$$\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t,
\qquad \mathbf{a}_t = \mathbf{a}(\mathbf{x}_t),\ \mathbf{b}_t = \mathbf{b}(\mathbf{x}_t)$$

$$\mathbf{h}_t = \sum_{s \le t} \Big(\prod_{r=s+1}^{t} \mathbf{a}_r\Big) \odot \mathbf{b}_s$$

- Decay $\mathbf{a}_t \in (0,1)^d$; the *write* $\mathbf{b}_t$ is what step
  $t$ adds (not a bias).
- *Affine* in $\mathbf{h}_{t-1}$: multiply, then add.
- An input survives as long as the inputs after it choose to keep it.
- All $\mathbf{a}_t, \mathbf{b}_t$ come from batched matrix products
  before the recurrence starts.
:::

::: {.slide title="The minGRU Is a GRU Whose Gates Read Only the Input"}
GRU: $\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{xz}\mathbf{x}_t + \mathbf{W}_\textrm{hz}\mathbf{h}_{t-1} + \mathbf{b}_\textrm{z})$;
its reset gate and candidate read $\mathbf{h}_{t-1}$ too. minGRU:

$$\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{z}\mathbf{x}_t + \mathbf{b}_\textrm{z}),\quad
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_\textrm{h}\mathbf{x}_t + \mathbf{b}_\textrm{h}),\quad
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$$

- Decay $\mathbf{a}_t = \mathbf{z}_t$, write $\mathbf{b}_t = (1 - \mathbf{z}_t)\odot\tilde{\mathbf{h}}_t$.
- The GRU's three uses of $\mathbf{h}_{t-1}$, in its gates and its
  candidate, are gone, and the reset gate with them.
- $2d^2$ weights per layer instead of the GRU's $6d^2$.
:::

::: {.slide title="A minGRU Unit Cannot Tell an Empty State From a Full One"}
Task for one unit: store the **first** digit of a text, ignore later ones.

| | update gate | first digit | second digit |
|:--|:--|:--|:--|
| GRU | $\sigma(\ldots + \mathbf{W}_\textrm{hz}\mathbf{h}_{t-1})$ | write | keep: the state is full |
| minGRU | $\sigma(\mathbf{W}_\textrm{z}\mathbf{x}_t + \mathbf{b}_\textrm{z})$ | write | write again |

::: {.d2l-note}
Gates that read only the token decide the same way whatever the state
holds. Stacking layers recovers some of this.
:::
:::

::: {.slide title="Affine Steps Compose Into Affine Steps"}
A step is the pair $(\mathbf{a}, \mathbf{b})$: the map
$\mathbf{h} \mapsto \mathbf{a} \odot \mathbf{h} + \mathbf{b}$.

$$(\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2) =
(\mathbf{a}_1 \odot \mathbf{a}_2,\ \mathbf{a}_2 \odot \mathbf{b}_1 + \mathbf{b}_2)$$

Example: $(0.5, 1) \bullet (0.5, 2) = (0.25,\ 2.5)$; stepping from
$h_0 = 0$ gives $h_1 = 1$, $h_2 = 2.5$.

- Summing the unrolled form term by term costs about $T^2/2$; composing
  pairs avoids the square.
- $\mathbf{h}_t$ is the $\mathbf{b}$-part of the first $t$ steps composed.
- The $\mathbf{a}$-part is carried so that a block can shrink what came
  before it.

::: {.d2l-note}
Associative, and a block of a thousand steps is still **one pair**.
:::
:::

::: {.slide title="A Scan Combines With a Doubling Stride"}
![](../img/mdl-modernrnn-scan-tree.svg){width=60%}

```{.python #ssm-exercises}
s = 1
while s < T:  # Combine each position with the one s steps back
    a, b = (cat(a[:s], a[:-s] * a[s:]),
            cat(b[:s], a[s:] * b[:-s] + b[s:]))
    s *= 2
```
A work-efficient scan needs about $2T$ combines in about $2\log_2 T$
rounds.

::: {.d2l-note}
$\lceil\log_2 T\rceil$ rounds instead of $T$, for about $\log_2 T$ times
the work.
:::
:::

::: {.slide title="Check: Rounds and Work"}
$T = 4096$.

1. How many rounds does the doubling scan need?
2. About how many combines, against the loop's steps?

::: {.fragment .fade-up}
1. $\log_2 4096 = 12$ rounds.
2. About $4096 \cdot 12 \approx 49{,}000$ combines against 4,095 steps:
   about 12 times the work, in 340 times fewer rounds.
:::
:::

::: {.slide title="At Equal Size the minGRU Trails the GRU by About 0.03 Nats per Byte"}
@!linear-recurrence-a-linear-recurrent-language-model-2

- Two runs each, nats per byte. At 2.76M parameters the minGRU is level
  with a plain tanh RNN of the same size (about 1.35).
- The gap is about 0.02 in the first 16 bytes, about 0.03 from byte 16 on.

::: {.d2l-note}
Input-only gates do no better than no gates. But the plain RNN's $\tanh$
chain cannot be scanned: the minGRU's case is speed at long lengths.
:::
:::

::: {.slide title="Memory Is Worth More Than a Nat per Byte"}
@!ssm-a-linear-recurrent-language-model-4

- Reset every state before every step: each prediction sees one byte.
- A one-byte model is a bigram model; the bigram model of the training
  bytes scores 2.45, even on those bytes.
- So about 1.1 nats per byte come from what the model remembers.
:::

::: {.slide title="After Training, One Gate in Six Keeps Its State for 7 Bytes or More"}
@!ssm-a-linear-recurrent-language-model-6

- Gate value $g$: half-life $\ln 2/\ln(1/g)$ steps.
- First layer, one validation batch, after 1,000 steps of training.
- Gates start near $1/2$, a half-life of about one step: before training
  about 1% exceed 0.9, after training about 18%.

::: {.d2l-note}
Windows of 128 bytes never ask for a memory of thousands; whether training
would find one is the next deck's question.
:::
:::

::: {.slide title="Attention Needs Few Steps in a Row, but Its Work Grows as T²"}
One layer of width $d$ on $T$ tokens, up to constant factors.

| layer | work | steps in a row | memory at generation |
|:--|:--|:--|:--|
| GRU | $T d^2$ | $T$ | $d$ |
| minGRU, doubling scan | $T d^2 + T d \log_2 T$ | $\log_2 T$ | $d$ |
| minGRU, work-efficient scan | $T d^2 + T d$ | $2\log_2 T$ | $d$ |
| causal attention | $T d^2 + T^2 d$ | a few matmuls | $2Td$ |

Attention per token: $8d^2$ FLOPs in projections, about $2Td$ in causal
scores and mixing.
:::

::: {.slide title="Check: Predict the Curves"}
Width $d = 256$; always 65,536 tokens per call; $T$ from 512 to 16,384.

1. Which layer's time stays flat?
2. Which grows nearly in proportion to $T$, and why?
3. From which $T$ on does attention's time start to double?

::: {.fragment .fade-up}
1. The scan: its work is proportional to the number of tokens.
2. The GRU: $T$ steps in a row, each with a fixed latency.
3. $8d^2 = 2Td$ at $T = 4d = 1{,}024$; beyond that, toward doubling.
:::
:::

::: {.slide title="Beyond a Few Thousand Tokens Attention Grows With T; the Scan Does Not"}
@!linear-recurrence-cost-against-attention-2

Forward and backward, milliseconds per 65,536 tokens. The count's turn is
at 1,024; the measured one comes a little later, because time per token
that the FLOPs leave out dilutes the quadratic term.

::: {.d2l-note}
GRU slowest everywhere; attention about 1.7× per doubling from 4,096 on.
:::
:::

::: {.slide title="The Doubling Scan Makes 9 to 14 Passes Over the Sequence" only="pytorch"}
- Doubling schedule: $\lceil\log_2 T\rceil$ rounds, each reading and
  writing all 65,536 tokens; 9 rounds at 512, 14 at 16,384.
- Its time follows the rounds: slower than attention up to 2,048 tokens,
  level near 4,096, faster beyond.
- A work-efficient scan touches the sequence a few times, whatever $T$.
:::

::: {.slide title="A Work-Efficient Scan Stays Flat and Below Attention" only="jax"}
- `jax.lax.associative_scan`: about $2T$ combines, a few passes over the
  sequence, whatever $T$.
- Its time barely changes from 512 to 16,384 tokens and stays below
  attention's at every length.
- A doubling scan would make $\lceil\log_2 T\rceil$ passes instead.
:::

::: {.slide title="Recap"}
- Linear in the state: $\mathbf{h}_t$ is a sum of writes weighted by
  products of decays; forgetting and vanishing gradients are one number.
- Affine steps compose into one pair: all states in
  $\lceil\log_2 T\rceil$ rounds.
- minGRU on WikiText bytes: 0.03 nats per byte behind the GRU at equal
  size, level with a plain RNN, whose $\tanh$ chain cannot be scanned;
  memory worth more than a nat per byte.
- Cost: the GRU grows with $T$, attention with $T$ beyond a few thousand
  tokens at width 256, the scan hardly at all.

**Next:** the gates start near one half. How fast should a linear state
forget, and where should its decays start?
:::
