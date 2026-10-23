# Linear Recurrences and Parallel Scans
:label:`sec_linear-recurrence`

A gated recurrent network that reads a sequence of 4,096 tokens performs
4,096 updates, and each update needs the result of the one before. A GPU can
carry out tens of thousands of multiply-adds at the same time, but it cannot
start step 2,000 until step 1,999 has finished, so during training most of it
waits. A transformer layer has no such chain: it processes all 4,096
positions in a few large matrix products. This difference in training
speed, more than any difference in what the two kinds of model can
represent, was a main reason why transformers replaced recurrent networks
for large-scale language modeling.

The chain has a single cause. The gates of :numref:`sec_lstm` read the
previous state, and the state passes through a nonlinearity, so step $t$
cannot be computed until step $t-1$ is known. If the gates read only the
current input, the update becomes linear in the state. Linear updates can be
grouped like the terms of a sum, and the states at all 4,096 positions can
then be computed in $\log_2 4096 = 12$ rounds of batched arithmetic. When
the model generates text, it still updates its state once per token, in
constant memory, like any recurrent network.

We begin with the simplest linear recurrence, a single number that forgets at
a fixed rate, and use it to build a gated network whose update is linear in
the state. We then compute all of its states at once and train a language
model with it. The next section asks how such a memory should forget.

We write recurrences for a single sequence, with column vectors; the code
adds a batch dimension.

```{.python .input #linear-recurrence-imports}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #linear-recurrence-imports}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
```

## Linear Recurrences
:label:`subsec_linear-recurrences`

### A Single Decaying Number

Consider a state $h_t$, a single number, that keeps a fraction $a$ of its
value at every step and adds the new input $x_t$:

$$h_t = a\, h_{t-1} + x_t, \qquad h_0 = 0.$$
:eqlabel:`eq_lr-scalar`

With $a = 0.9$ and a single input $x_1 = 1$ followed by zeros, the state
reads $1, 0.9, 0.81, 0.729, \ldots$: an input from $k$ steps ago survives
with weight $a^k$. Substituting the recurrence into itself shows that this
holds for any input sequence,

$$h_t = \sum_{s=1}^{t} a^{t-s} x_s.$$
:eqlabel:`eq_lr-unrolled`

The state is a weighted sum of all past inputs, with weights that shrink
geometrically with age. Two consequences follow. First, the sum does not
mention $h_{t-1}$: every $h_t$ is determined by the inputs alone, so in
principle all of them can be computed at once. Second, the single number
$a$ sets the length of the memory. The weight of an input halves every
$\ln 2 / \ln(1/a)$ steps, which we call the *half-life* of the memory: about
7 steps for $a = 0.9$, 69 for $a = 0.99$, and 693 for $a = 0.999$. For $a$
close to one the half-life is about $0.69/(1-a)$, so each additional 9 in
$a$ makes the memory ten times longer.

The same weights govern learning. The gradient of $h_t$ with respect to
$x_s$ is $a^{t-s}$, so the decay that forgets an input also weakens the
signal that would teach the model to use it. This is the vanishing gradient
of :numref:`subsec_bptt-gradient-pathologies`, where the gradient through
$k$ steps of a plain recurrent network is a product of $k$ Jacobian
matrices. Here the product is a power of one number, and we can read it off
directly.

### Vectors and Input-Dependent Decay

A memory of one number is too small for language. With a state
$\mathbf{h}_t \in \mathbb{R}^d$ we can give each coordinate its own decay,
collected in a vector $\mathbf{a} \in (0, 1)^d$ that multiplies the state
elementwise ($\odot$):

$$\mathbf{h}_t = \mathbf{a} \odot \mathbf{h}_{t-1} + \mathbf{b}_t.$$

Nothing new happens here. The coordinates do not interact, so this is $d$
copies of :eqref:`eq_lr-scalar` side by side, each with its own half-life. We
have also renamed the input term $\mathbf{b}_t$, because it will soon be a
learned function of the input.

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
never grow: no past input is ever amplified, however long the sequence. Because
$\mathbf{a}_t$ and $\mathbf{b}_t$ depend only on $\mathbf{x}_t$, they can be
computed for all positions in one batched matrix product before the
recurrence starts. What remains is the recurrence itself, an update that is
*affine* in $\mathbf{h}_{t-1}$: an elementwise multiplication followed by an
addition.

### The Minimal GRU
:label:`subsec_mingru`

The GRU of :numref:`sec_gru` almost has this form. Written for one sequence,
its update gate $\mathbf{z}_t$, reset gate $\mathbf{r}_t$, and candidate
state $\tilde{\mathbf{h}}_t$ are

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(\mathbf{W}_{\textrm{xz}} \mathbf{x}_t + \mathbf{W}_{\textrm{hz}} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{z}),\\
\mathbf{r}_t &= \sigma(\mathbf{W}_{\textrm{xr}} \mathbf{x}_t + \mathbf{W}_{\textrm{hr}} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{r}),\\
\tilde{\mathbf{h}}_t &= \tanh\big(\mathbf{W}_{\textrm{xh}} \mathbf{x}_t + \mathbf{W}_{\textrm{hh}} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_\textrm{h}\big),
\end{aligned}
$$

and the new state is a blend of the old state and the candidate,

$$
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t)
\odot \tilde{\mathbf{h}}_t.
$$

The blend is already affine in $\mathbf{h}_{t-1}$, with decay
$\mathbf{z}_t$. What stands in the way are the three places where
$\mathbf{h}_{t-1}$ enters a gate or the candidate.
:citet:`Feng.Tung.Ahmed.ea.2024` removed them: the update gate and the
candidate read only the input, and the reset gate, which no longer has a
previous state to filter, disappears. The result is the *minimal GRU*, or
*minGRU*:

$$
\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{z} \mathbf{x}_t + \mathbf{b}_\textrm{z}), \qquad
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_\textrm{h} \mathbf{x}_t + \mathbf{b}_\textrm{h}), \qquad
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t.
$$
:eqlabel:`eq_mingru`

This is :eqref:`eq_affine_recurrence` with decay $\mathbf{a}_t =
\mathbf{z}_t$ and write $\mathbf{b}_t = (1 - \mathbf{z}_t) \odot
\tilde{\mathbf{h}}_t$. The update gate is now the decay of a linear
recurrence: a gate value of 0.9 keeps 90% of the state and admits 10% of
the candidate. The cell is also much smaller. A GRU layer with $d$ inputs
and $h$ hidden units has about $3h(d + h)$ weights, three input matrices and
three state matrices; the minGRU has $2h(d + 1)$ weights and biases. For
$d = 64$ and $h = 128$ that is about 74,000 against 16,640.

The minGRU gives something up. A GRU can decide to keep its state because of
what the state contains; the minGRU decides only from the current token, and
between tokens its state can only be scaled and added to. Stacking layers
gives some of this back: the outputs of one layer, passed through a
nonlinearity, are the inputs of the next, and nonlinearities between layers
do not block parallel evaluation. At the end of this section we measure what
a single layer gives up on language modeling.

## Parallel Scans
:label:`subsec_parallel-scans`

### Composing Affine Steps

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
sequence under an associative operation is called a *scan*.

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
loop, in exchange for $\log_2 T$ rounds instead of $T - 1$. A schedule of
:citet:`Blelloch.1990` needs only about $2T$ combines in about $2 \log_2 T$
rounds. The difference in total work matters once a sequence is long enough
to keep the whole GPU busy, as the measurements below show.

### Implementation

`linear_scan` implements the doubling schedule. Time runs along the first
axis; all other axes are batch and feature axes, and the decays may have
size one along them, in which case they are broadcast. An optional initial
state is folded into the first step, since $\mathbf{h}_1 = \mathbf{a}_1 \odot
\mathbf{h}_0 + \mathbf{b}_1$. In PyTorch, each round combines the sequence
with a copy of itself shifted by the stride. JAX provides scans as a
primitive, `jax.lax.associative_scan`, which takes the combine
:eqref:`eq_scan_combine` as a function and uses a schedule with linear total
work.

```{.python .input #linear-recurrence-scan}
%%tab pytorch
def linear_scan(a, b, h0=None):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t, by a parallel scan."""
    if h0 is not None:  # Fold the initial state into the first step
        b = torch.cat([a[:1] * h0 + b[:1], b[1:]])
    s = 1
    while s < b.shape[0]:  # Combine every position with the one s back
        a, b = (torch.cat([a[:s], a[:-s] * a[s:]]),
                torch.cat([b[:s], a[s:] * b[:-s] + b[s:]]))
        s *= 2
    return b
```

```{.python .input #linear-recurrence-scan}
%%tab jax
def linear_scan(a, b, h0=None):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t, by a parallel scan."""
    if h0 is not None:  # Fold the initial state into the first step
        b = b.at[0].add(a[0] * h0)
    def combine(earlier, later):
        (a1, b1), (a2, b2) = earlier, later
        return a1 * a2, a2 * b1 + b2
    return jax.lax.associative_scan(combine, (a, b))[1]
```

We check the scan against a loop that applies the recurrence one step at a
time, on random decays in $(0, 1)$ and a length that is not a power of two.

```{.python .input #linear-recurrence-check}
%%tab pytorch
def sequential_scan(a, b):
    h, states = torch.zeros_like(b[0]), []
    for a_t, b_t in zip(a, b):
        h = a_t * h + b_t
        states.append(h)
    return torch.stack(states)

a, b = torch.rand(100, 4, 8), torch.randn(100, 4, 8)
print(f'max deviation: '
      f'{(linear_scan(a, b) - sequential_scan(a, b)).abs().max():.1e}')
```

```{.python .input #linear-recurrence-check}
%%tab jax
def sequential_scan(a, b):
    def step(h, ab):
        h = ab[0] * h + ab[1]
        return h, h
    return jax.lax.scan(step, jnp.zeros_like(b[0]), (a, b))[1]

key_a, key_b = jax.random.split(jax.random.key(0))
a = jax.random.uniform(key_a, (100, 4, 8))
b = jax.random.normal(key_b, (100, 4, 8))
print(f'max deviation: '
      f'{jnp.abs(linear_scan(a, b) - sequential_scan(a, b)).max():.1e}')
```

The two agree to the rounding error of 32-bit floating point. The scan
groups the multiplications and additions differently from the loop, so the
results differ in the last digits; they are the same computation.

How much faster is the scan? We time both on a GPU for sequences of 256 to
16,384 steps, with a batch of 32 sequences of 128 coordinates each.
`d2l.Benchmark` calls a function a few times after a warm-up call and
reports the average time, waiting for the GPU to finish each call.

```{.python .input #linear-recurrence-timing}
%%tab pytorch
device = d2l.try_gpu()
lengths, times = [256, 1024, 4096, 16384], [[], []]
for T in lengths:
    a = torch.rand(T, 32, 128, device=device)
    b = torch.randn(T, 32, 128, device=device)
    for i, scan in enumerate([linear_scan, sequential_scan]):
        bench = d2l.Benchmark(lambda: scan(a, b), warmup=1, repeats=3)
        times[i].append(1e3 * bench.time)
d2l.plot(lengths, times, 'sequence length', 'milliseconds per call',
         legend=['parallel scan', 'loop'], xscale='log', yscale='log',
         figsize=(4.5, 3))
```

```{.python .input #linear-recurrence-timing}
%%tab jax
lengths, times = [256, 1024, 4096, 16384], [[], []]
scans = [jax.jit(linear_scan), jax.jit(sequential_scan)]
for T in lengths:
    key_a, key_b = jax.random.split(jax.random.key(T))
    a = jax.random.uniform(key_a, (T, 32, 128))
    b = jax.random.normal(key_b, (T, 32, 128))
    for i, scan in enumerate(scans):
        bench = d2l.Benchmark(lambda: scan(a, b), warmup=1, repeats=3)
        times[i].append(1e3 * bench.time)
d2l.plot(lengths, times, 'sequence length', 'milliseconds per call',
         legend=['parallel scan', 'loop'], xscale='log', yscale='log',
         figsize=(4.5, 3))
```

The loop's time grows in proportion to the length: each step is a separate
small computation that must finish before the next one starts. The scan is
faster at every length, by a factor between several and a hundred depending
on the length and the framework. Its time hardly changes up to about a
thousand steps, where its few rounds of arithmetic are too small to keep the
GPU busy, and then it too grows in proportion to the length, because once the
GPU is full the time is set by the total amount of work. (The compiled JAX
scan, with its linear-work schedule, gains more than the PyTorch version,
which runs every round as separate operations.) In both cases the scan
changes how we train, not how we
generate: at inference the model still applies
:eqref:`eq_affine_recurrence` once per token, in constant memory.

## A Linear Recurrent Language Model
:label:`subsec_mingru-lm`

The minGRU :eqref:`eq_mingru` is a few lines around `linear_scan`. Like the
recurrent layers of :numref:`sec_lstm`, it takes a sequence of inputs and an
optional initial state and returns the outputs at every step together with
the final state, so it can replace them in the language model `d2l.RNNLM`,
which embeds the tokens, runs the recurrent layer, and predicts the next
token from each output.

```{.python .input #linear-recurrence-mingru}
%%tab pytorch
class MinGRU(d2l.Module):  #@save
    """The minimal GRU: its gates and candidate read only the input."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.W_z = nn.Linear(num_inputs, num_hiddens)  # Update gate (decay)
        self.W_h = nn.Linear(num_inputs, num_hiddens)  # Candidate state

    def forward(self, inputs, H=None):
        Z = torch.sigmoid(self.W_z(inputs))  # (num_steps, batch, num_hiddens)
        H_tilde = torch.tanh(self.W_h(inputs))
        outputs = linear_scan(Z, (1 - Z) * H_tilde, H)
        return outputs, outputs[-1]
```

```{.python .input #linear-recurrence-mingru}
%%tab jax
class MinGRU(nnx.Module):  #@save
    """The minimal GRU: its gates and candidate read only the input."""
    def __init__(self, num_inputs, num_hiddens, *, rngs):
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.W_z = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)  # Decay
        self.W_h = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)

    def __call__(self, inputs, H=None):
        Z = jax.nn.sigmoid(self.W_z(inputs))  # (num_steps, batch, num_hiddens)
        H_tilde = jnp.tanh(self.W_h(inputs))
        outputs = linear_scan(Z, (1 - Z) * H_tilde, H)
        return outputs, outputs[-1]
```

To compare language models we need their perplexity on held-out text, the
exponential of the average per-token cross-entropy loss over the validation
set.

```{.python .input #linear-recurrence-perplexity}
%%tab pytorch
def lm_perplexity(model, data):  #@save
    """Perplexity of a language model on the validation set of `data`."""
    device = next(model.parameters()).device
    total, count = 0, 0
    with torch.inference_mode():
        for X, Y in data.val_dataloader():
            losses = model.loss(model(X.to(device)), Y.to(device),
                                averaged=False)
            total, count = total + losses.sum(), count + losses.numel()
    return math.exp(float(total) / count)
```

```{.python .input #linear-recurrence-perplexity}
%%tab jax
def lm_perplexity(model, data):  #@save
    """Perplexity of a language model on the validation set of `data`."""
    total, count = 0, 0
    for X, Y in data.val_dataloader():
        losses = model.loss(model(X), Y, averaged=False)
        total, count = total + losses.sum(), count + losses.size
    return math.exp(float(total) / count)
```

We train a GRU and a minGRU, each with 128 hidden units on 64-dimensional
token embeddings, with the setup of :numref:`sec_lstm`: *The Time Machine*
with a vocabulary of 1,024 tokens, 50,000 windows of 32 tokens, and ten
epochs of minibatch stochastic gradient descent with learning rate 4 and
gradients clipped to norm 1.

```{.python .input #linear-recurrence-data}
%%tab pytorch, jax
data = d2l.TimeMachine(batch_size=1024, num_steps=32, num_train=50000,
                       num_val=5000)
```

```{.python .input #linear-recurrence-train}
%%tab pytorch
layers = {'GRU': d2l.GRU(64, 128), 'minGRU': MinGRU(64, 128)}
models, results = {}, {}
for name, layer in layers.items():
    models[name] = d2l.RNNLM(layer, vocab_size=len(data.vocab), lr=4)
    models[name].board.display = False
    trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
    trainer.fit(models[name], data)
    num_params = sum(p.numel() for p in layer.parameters())
    results[name] = num_params, lm_perplexity(models[name], data)
```

```{.python .input #linear-recurrence-train}
%%tab jax
layers = {'GRU': d2l.GRU(64, 128),
          'minGRU': MinGRU(64, 128, rngs=nnx.Rngs(0))}
models, results = {}, {}
for name, layer in layers.items():
    models[name] = d2l.RNNLM(layer, vocab_size=len(data.vocab), lr=4)
    models[name].board.display = False
    trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
    trainer.fit(models[name], data)
    num_params = sum(p.size for p in jax.tree.leaves(
        nnx.state(layer, nnx.Param)))
    results[name] = num_params, lm_perplexity(models[name], data)
```

```{.python .input #linear-recurrence-results}
%%tab pytorch, jax
for name, (num_params, ppl) in results.items():
    print(f'{name:6s}  {num_params:6d} recurrent parameters  '
          f'perplexity {ppl:.1f}')
```

The minGRU's perplexity is a few points above the GRU's, with less than a
quarter of the recurrent parameters. On this task, taking the state out of
the gates costs little.

### The Trained Gates

Because the minGRU's decay is its update gate, we can look inside the trained
model. Two questions are easy to answer on a batch of validation text. First,
does the model use its memory at all? We compare its perplexity with that of
the same model with its state reset to zero before every step, so that each
prediction sees only the current token: $\mathbf{h}_t = (1 -
\mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$. Second, how often does a gate
hold the state rather than overwrite it?

```{.python .input #linear-recurrence-probe}
%%tab pytorch
model = models['minGRU']
device = next(model.parameters()).device
X, Y = next(iter(data.val_dataloader()))
X, Y = X.to(device), Y.to(device)
with torch.inference_mode():
    E = model.embedding(X)
    Z = torch.sigmoid(model.rnn.W_z(E))  # Decays a_t
    B = (1 - Z) * torch.tanh(model.rnn.W_h(E))  # Writes b_t
    for name, H in [('with memory', linear_scan(Z, B)), ('state reset', B)]:
        loss = model.loss(model.output_layer(H), Y)
        print(f'{name:11s}  perplexity {math.exp(loss):.1f}')
print(f'gates above 0.5: {(Z > 0.5).float().mean():.0%}, '
      f'above 0.9: {(Z > 0.9).float().mean():.0%}')
```

```{.python .input #linear-recurrence-probe}
%%tab jax
model = models['minGRU']
X, Y = next(iter(data.val_dataloader()))
E = model.embedding(X)
Z = jax.nn.sigmoid(model.rnn.W_z(E))  # Decays a_t
B = (1 - Z) * jnp.tanh(model.rnn.W_h(E))  # Writes b_t
for name, H in [('with memory', linear_scan(Z, B)), ('state reset', B)]:
    loss = model.loss(model.output_layer(H), Y)
    print(f'{name:11s}  perplexity {math.exp(loss):.1f}')
print(f'gates above 0.5: {(Z > 0.5).mean():.0%}, '
      f'above 0.9: {(Z > 0.9).mean():.0%}')
```

Resetting the state raises the perplexity by more than ten points, so the
model relies on what it remembers. Yet most of its gates are small: about
30% of the gate values exceed 0.5, and fewer than one in ten exceeds 0.9,
the value at which a held state would halve only every 7 steps.
Most tokens largely overwrite the state, and a few hold it. This is the
input-dependent decay of :eqref:`eq_lr-unrolled-gated` at work: what
survives is decided token by token.

The minGRU shows that a gated recurrence can be linear in its state, and
therefore trained with a scan, at a small cost in perplexity. It leaves two
questions open. Its gates start near $\sigma(0) = 1/2$, a half-life of one
step, so an input 20 steps back initially reaches the state, and sends its
gradient back, with weight $2^{-20}$, about one in a million; nothing in the
model offers longer timescales from the start. And its memory holds only one
number per hidden unit. The next section derives the decay from a
continuous-time model, which gives stable decays with a spread of timescales
from the first step of training, and a larger state at little extra cost.

## Summary

A recurrence whose update is linear in the state, $\mathbf{h}_t =
\mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$, keeps a weighted sum of
its past inputs, and the weights are products of decays that we can compute
and read. Because affine steps compose into affine steps, all states of a
sequence can be computed by a parallel scan in about $\log_2 T$ rounds of
batched arithmetic, instead of $T$ dependent steps. The minGRU obtains such
a recurrence from the GRU by letting its update gate and candidate read only
the input; on our language modeling task it comes within a few points of the
GRU's perplexity with a quarter of the recurrent parameters, and it uses its
memory selectively, holding the state on a few tokens and overwriting it on
most. Its decays start near one half, and its memory holds one number per
hidden unit; the next section addresses both.

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

1. [code] **Candidate nonlinearity.** `MinGRU` keeps a $\tanh$ on the
   candidate $\tilde{\mathbf{h}}_t$, whereas
   :citet:`Feng.Tung.Ahmed.ea.2024` use a linear candidate.
    1. Starting from $\mathbf{h}_0 = \mathbf{0}$, write each coordinate of
       $\mathbf{h}_t$ as a weighted sum of the candidates
       $\tilde{\mathbf{h}}_1, \ldots, \tilde{\mathbf{h}}_t$. Show that the
       weights are nonnegative and sum to at most one, and derive a bound on
       $|\mathbf{h}_t|$ that holds at every length, with and without the
       $\tanh$.
    1. Remove the $\tanh$, retrain with the setup of
       :numref:`subsec_mingru-lm`, and compare the validation perplexity
       with that of `MinGRU`. Relate what you observe to your bound.

1. [code] **A minimal LSTM.** Apply the same two cuts to the LSTM of
   :numref:`sec_lstm`: compute its input, forget, and output gates and its
   input node from $\mathbf{x}_t$ alone, and keep its cell update
   $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot
   \tilde{\mathbf{c}}_t$.
    1. Show that only the cell state needs a scan. Implement the cell with
       the interface of `MinGRU`, train it with the setup of
       :numref:`subsec_mingru-lm`, and compare its perplexity with that of
       `MinGRU`.
    1. Unlike the minGRU, this cell does not tie its input gate to its
       forget gate. Show that if $|\tilde{\mathbf{c}}_t| < 1$ and every
       forget gate stays below $f_\textrm{max} < 1$, the cell state stays
       below $1 / (1 - f_\textrm{max})$ in magnitude, and give gate values
       under which it approaches this bound. What bound holds if the gates
       are replaced by $\mathbf{f}_t / (\mathbf{f}_t + \mathbf{i}_t)$ and
       $\mathbf{i}_t / (\mathbf{f}_t + \mathbf{i}_t)$, as
       :citet:`Feng.Tung.Ahmed.ea.2024` propose?

1. [code] **State passing.** `linear_scan` accepts an initial state, so a
   long sequence can be processed in pieces.
    1. Split a sequence of length $T$ into chunks of length $\tau$, run
       `MinGRU` on each chunk starting from the final state of the previous
       one, and check that the concatenated outputs match a single call on
       the whole sequence.
    1. Now stop gradients through the carried state at every chunk
       boundary. The outputs do not change. Which contributions to the
       gradient of the loss at step $t$ disappear, and which dependencies
       can the trained model still use when it generates text? Compare with
       truncated backpropagation through time (:numref:`sec_bptt`).

1. [code] **Initializing for long memory.** At initialization the minGRU's
   update gates are close to $\sigma(0) = 1/2$.
    1. Choose the gate biases $\mathbf{b}_\textrm{z}$ so that, at
       initialization, the half-lives of the hidden units are spread
       log-uniformly between 1 and 1,000 steps. What bias gives a half-life
       of $k$ steps?
    1. Train this model and the default one on windows of 256 tokens instead
       of 32, and compare their perplexities and the half-lives they learn.

1. [code] **Work-efficient scans.** ● A recursive schedule does less work
   than the doubling schedule: combine adjacent pairs, which halves the
   sequence; scan the half-length sequence recursively, which yields every
   second prefix; then obtain each remaining prefix with one more combine.
    1. For $T = 2^k$, count the combines and the rounds of combines of each
       schedule, and compare both with the $T - 1$ steps of the loop at
       $T = 16384$.
    1. Implement the recursive schedule with slicing, check it against
       `sequential_scan`, and time it against `linear_scan` at the lengths
       used above. Does the ratio of times follow the ratio of combines?
       Explain why it might not at short lengths.

[Discussions](https://d2l.discourse.group/)
