# Backpropagation Through Time
:label:`sec_bptt`

Our experiments with the RNN language model in :numref:`sec_rnn-scratch` left
us with two striking empirical findings: without gradient clipping, most
training runs destabilized; with clipping, training converged reliably, yet
the learned network used only the last few tokens of its 32-token context
window. Both phenomena share the exact same mathematical root. Because a
recurrent network applies the same weight matrix at every time step, any
gradient propagating backward through a sequence is multiplied by a state
Jacobian at each step. Over many steps these factors compound—much like
compound interest—causing the gradient to grow or shrink geometrically.

Even a scalar recurrence $h_t = w_\textrm{h} h_{t-1} + x_t$ shows why:
unrolling gives $h_t = \sum_{s \leq t} w_\textrm{h}^{t-s} x_s$, so
$\partial h_t / \partial h_{t-k} = w_\textrm{h}^k$. Over forty steps,
$w_\textrm{h} = 0.9$ shrinks a gradient to $0.9^{40} \approx 0.015$, whereas
$w_\textrm{h} = 1.1$ inflates it to $1.1^{40} \approx 45$. Exploding gradients
destabilize optimization immediately, forcing us to clip updates; vanishing
gradients are quieter yet more insidious, attenuating error signals from
distant tokens until the model cannot learn long-range dependencies.

Computing gradients in an RNN requires no new differentiation algorithm: we
unroll the recurrence across time into a deep feedforward network with shared
weights and apply standard backpropagation (:numref:`sec_backprop`), a
procedure known as *backpropagation through time* (BPTT) :cite:`Werbos.1990`.
Below, we derive the backward recursion directly in matrix form, measure how
far gradients actually travel inside our trained RNN from
:numref:`sec_rnn-scratch`, and examine truncated backpropagation through time.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input #bptt-backpropagation-through-time}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #bptt-backpropagation-through-time}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #bptt-backpropagation-through-time}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #bptt-backpropagation-through-time}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Vanishing and Exploding Gradients
:label:`subsec_bptt_analysis`
:label:`subsec_vanishing-exploding`

Consider an RNN unrolled across $T$ steps with shared recurrent parameters and
average sequence loss $L = \frac{1}{T}\sum_{t=1}^T l(\mathbf{y}_t,
\mathbf{o}_t)$. Backpropagation handles weight sharing by summing the local
gradient contributions from every time step where a weight matrix appears. To
see the structure of the backward pass transparently, take the RNN of
:numref:`sec_rnn` and temporarily omit biases and the $\tanh$ nonlinearity (we
restore $\tanh$ immediately afterward). For a minibatch of $b$ sequences with
step inputs $\mathbf{X}_t \in \mathbb{R}^{b\times d}$, the linear RNN computes

$$
\mathbf{H}_t = \mathbf{X}_t \mathbf{W}_\textrm{xh} +
\mathbf{H}_{t-1}\mathbf{W}_\textrm{hh}, \qquad
\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_\textrm{hq},
$$
:eqlabel:`eq_bptt_linear_rnn`

with $\mathbf{W}_\textrm{xh}\in\mathbb{R}^{d\times h}$,
$\mathbf{W}_\textrm{hh}\in\mathbb{R}^{h\times h}$, and
$\mathbf{W}_\textrm{hq}\in\mathbb{R}^{h\times q}$. :numref:`fig_rnn_bptt`
depicts three unrolled time steps of this network, showing the forward pass in
grey and the backward pass in red.

![Forward pass (grey) and backward pass (red) of a linear RNN unrolled over three time steps. The gradient that reaches each hidden state except the last is the sum of two red arrows: one from its own output, through $\mathbf{W}_\textrm{hq}^\top$, and one from the next state, through $\mathbf{W}_\textrm{hh}^\top$.](../img/mdl-rnn-bptt.svg)
:label:`fig_rnn_bptt`

Let $\mathbf{G}_t = \partial L/\partial \mathbf{H}_t$ denote the gradient of
the total loss with respect to $\mathbf{H}_t$. Because $\mathbf{H}_t$ feeds
into both its immediate output $\mathbf{O}_t$ and the next hidden state
$\mathbf{H}_{t+1}$, and backpropagating through a right-multiplication
multiplies by the transposed weight matrix (:numref:`sec_backprop`),
$\mathbf{G}_t$ obeys the backward recursion

$$
\mathbf{G}_t = \frac{\partial L}{\partial \mathbf{O}_t}\mathbf{W}_\textrm{hq}^\top +
\mathbf{G}_{t+1}\mathbf{W}_\textrm{hh}^\top, \qquad \mathbf{G}_{T+1} = \mathbf{0},
$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

corresponding to the two incoming red arrows at each hidden state in
:numref:`fig_rnn_bptt`. Unrolling :eqref:`eq_bptt_partial_L_ht_recur` backward
from $\mathbf{G}_T$ down to $\mathbf{G}_t$ yields

$$
\mathbf{G}_t = \sum_{j=t}^{T} \frac{\partial L}{\partial \mathbf{O}_j}\,
\mathbf{W}_\textrm{hq}^\top\big(\mathbf{W}_\textrm{hh}^\top\big)^{j-t}.
$$
:eqlabel:`eq_bptt_partial_L_ht`

Thus, the loss at future step $j$ communicates back to the state $k = j - t$
steps earlier through the $k^\textrm{th}$ power of
$\mathbf{W}_\textrm{hh}^\top$ (for a general nonlinear transition
$\mathbf{H}_t = g(\mathbf{H}_{t-1}, \mathbf{X}_t, \mathbf{W}_\textrm{hh})$,
this matrix power is replaced by the product of the $j - t$ intermediate state
Jacobians $\prod_{i=t+1}^j \partial \mathbf{H}_i / \partial \mathbf{H}_{i-1}$).
Once $\mathbf{G}_1, \ldots, \mathbf{G}_T$ are computed in a single backward
sweep, the parameter gradients simply sum one outer product per step:

$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}} = \sum_{t=1}^{T}
\mathbf{H}_{t-1}^\top \mathbf{G}_t, \qquad
\frac{\partial L}{\partial \mathbf{W}_\textrm{xh}} = \sum_{t=1}^{T}
\mathbf{X}_t^\top \mathbf{G}_t.
$$
:eqlabel:`eq_bptt_param_grads`

As we saw in :numref:`subsec_lin-alg-eigenvalues` and
:numref:`sec_numerical_stability`, raising a matrix to the $k^\textrm{th}$
power scales each eigenvector by $\lambda^k$. Letting $\rho$ denote the
*spectral radius* (largest $|\lambda|$) of $\mathbf{W}_\textrm{hh}$, the norm
of $(\mathbf{W}_\textrm{hh}^\top)^k$ scales asymptotically as $\rho^k$: when
$\rho < 1$, contributions from distant steps vanish exponentially; when
$\rho > 1$, they explode exponentially. Reintroducing the $\tanh$ activation
multiplies each step's backward gradient entrywise by its local derivative
$1 - \mathbf{H}_t^2 \in (0, 1]$ before multiplying by
$\mathbf{W}_\textrm{hh}^\top$. Because $1 - h^2 \leq 1$, $\tanh$ never
amplifies a gradient and pulls strongly toward vanishing whenever hidden units
saturate near $\pm 1$, while still permitting explosion if
$\mathbf{W}_\textrm{hh}$ is large and activations stay near zero.

Let us measure what actually happens inside our trained language model from
:numref:`sec_rnn-scratch` ($d = 512$, $h = 128$, with dropout).

```{.python .input #bptt-vanishing-and-exploding-gradients-1}
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
```

```{.python .input #bptt-vanishing-and-exploding-gradients-2}
%%tab pytorch, mxnet, jax
rnn = d2l.RNNScratch(num_inputs=512, num_hiddens=128)
model = d2l.RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=4, dropout=0.5,
                         emb_dropout=0.1)
```

```{.python .input #bptt-vanishing-and-exploding-gradients-2}
%%tab tensorflow
with d2l.try_gpu():
    rnn = d2l.RNNScratch(num_inputs=512, num_hiddens=128)
    model = d2l.RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=4,
                             dropout=0.5, emb_dropout=0.1)
```

```{.python .input #bptt-vanishing-and-exploding-gradients-3}
trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

We run the trained model forward over 256 validation segments of length
$L = 64$—twice its training window—recording all hidden states and the
final-step logits.

```{.python .input #bptt-vanishing-and-exploding-gradients-4}
L = 64
val = d2l.numpy(data.X[data.num_train:data.num_train + data.num_val, 0])
starts = np.random.default_rng().choice(len(val) - L, 256, replace=False)
seqs = np.stack([val[s:s + L] for s in starts])  # (256, L) token ids
targets = val[starts + L]                        # The token after each stretch
```

```{.python .input #bptt-vanishing-and-exploding-gradients-5}
%%tab mxnet
states, _ = model.rnn(model.embedding(
    d2l.tensor(seqs, dtype=d2l.int32, ctx=d2l.try_gpu())))
logits = model.output_layer(states[-1:])[:, 0]  # The last step only
H, logits = d2l.numpy(d2l.stack(states, 0)), d2l.numpy(logits)
W_hh, W_hq = d2l.numpy(model.rnn.W_hh.data()), d2l.numpy(model.W_hq.data())
```

```{.python .input #bptt-vanishing-and-exploding-gradients-5}
%%tab pytorch
with torch.inference_mode():
    states, _ = model.rnn(model.embedding(
        d2l.tensor(seqs, device=d2l.try_gpu())))
    logits = model.output_layer(states[-1:])[:, 0]  # The last step only
H, logits = d2l.numpy(d2l.stack(states, 0)), d2l.numpy(logits)
W_hh, W_hq = d2l.numpy(model.rnn.W_hh), d2l.numpy(model.W_hq)
```

```{.python .input #bptt-vanishing-and-exploding-gradients-5}
%%tab tensorflow
states, _ = model.rnn(model.embedding(tf.constant(seqs)))
logits = model.output_layer(states[-1:])[:, 0]  # The last step only
H, logits = d2l.numpy(tf.stack(states, 0)), d2l.numpy(logits)
W_hh, W_hq = d2l.numpy(model.rnn.W_hh), d2l.numpy(model.W_hq)
```

```{.python .input #bptt-vanishing-and-exploding-gradients-5}
%%tab jax
states, _ = model.rnn(model.embedding(jnp.array(seqs)))
logits = model.output_layer(states[-1:])[:, 0]  # The last step only
H, logits = d2l.numpy(jnp.stack(states, 0)), d2l.numpy(logits)
W_hh, W_hq = d2l.numpy(model.rnn.W_hh[...]), d2l.numpy(model.W_hq[...])
```

Starting from the final-step loss gradient $\partial L_L / \partial
\mathbf{O}_L = \textrm{softmax}(\mathbf{O}_L) - \mathbf{Y}_L$, we step backward
through time via $\mathbf{G}_{t-1} = (\mathbf{G}_t \odot (1 -
\mathbf{H}_t^2))\mathbf{W}_\textrm{hh}^\top$ and plot the geometric mean of
the relative gradient norm $\|\mathbf{G}_{L-k}\| / \|\mathbf{G}_L\|$ as a
function of the lag $k$.

```{.python .input #bptt-vanishing-and-exploding-gradients-6}
P = np.exp(logits - logits.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)
P[np.arange(len(targets)), targets] -= 1  # dL/dO: softmax minus one-hot
G = (P @ W_hq.T).astype(np.float64)       # dL/dH at the last step
norms = []
for t in range(L - 1, -1, -1):            # Backward through time
    norms.append(np.linalg.norm(G, axis=1))
    G = (G * (1 - H[t] ** 2)) @ W_hh.T    # Through the tanh, then W_hh
rel = np.exp(np.log(np.array(norms) / norms[0]).mean(1))
print(f'spectral radius of W_hh: {np.abs(np.linalg.eigvals(W_hh)).max():.2f}; '
      f'mean tanh derivative: {(1 - H ** 2).mean():.2f}')
print('relative gradient norm at lags 1, 2, 4, 8, 16, 32:',
      [float(f'{rel[k]:.2g}') for k in (1, 2, 4, 8, 16, 32)])
d2l.plot(np.arange(L), rel, 'lag $k$', 'relative gradient norm',
         yscale='log', figsize=(4.5, 3))
```

Even though the trained $\mathbf{W}_\textrm{hh}$ has a spectral radius near
$1.8$—which would explode in a purely linear recurrence—the actual gradient
vanishes rapidly: it drops below 1% of its initial norm within roughly eight
steps and shrinks to a few billionths across 32 tokens. Because the $\tanh$
derivative averages roughly $0.27$, each backward step retains only about
$1.8 \times 0.27 \approx 0.5$ of the gradient norm. This directly explains our
finding in :numref:`sec_rnn-scratch` that per-position validation loss
flattens out after roughly three tokens of context.

### Remedies for Exploding and Vanishing Gradients
:label:`subsec_bptt-gradient-pathologies`

Exploding and vanishing gradients demand fundamentally different remedies.
Because an exploding gradient threatens to corrupt the parameters in a single
oversized update, global gradient clipping (:eqref:`eq_grad_clipping`) provides
a simple safeguard—yet multiplying the entire gradient sum
:eqref:`eq_bptt_partial_L_ht` by a scalar never changes the *relative* weight
of distant versus recent lags. Overcoming vanishing gradients therefore
requires changing the recurrence itself so that state Jacobians stay near unity
along memory directions: either via *gated* recurrent cells (such as LSTMs and
GRUs) that create an additive memory path controlled by learned gates, or via
*structured linear recurrences* and state space models that remove the state
nonlinearity and constrain eigenvalues near the unit circle. Both families are
developed in :numref:`chap_modern_rnn`.

## Truncated Backpropagation Through Time

Evaluating :eqref:`eq_bptt_partial_L_ht_recur` across an entire corpus of $T$
tokens requires retaining all $T$ forward hidden states in memory and
traversing all $T$ steps for a single parameter update. *Truncated
backpropagation through time* caps both memory and compute per update by
slicing long streams into fixed-length segments of length $T_0$ and halting
backpropagation at each segment boundary (:numref:`fig_truncated_bptt`).

![Full and truncated backpropagation through time over a token sequence. Both run the same forward pass (grey). Full backpropagation sends the gradient (red) back along the entire sequence. Truncation cuts the sequence into segments of four tokens and detaches the state at each boundary: its value crosses into the next segment, but no gradient flows back across.](../img/mdl-rnn-truncated-bptt.svg)
:label:`fig_truncated_bptt`

Whether each window resets $\mathbf{H}_0 = \mathbf{0}$ (as in
:numref:`sec_rnn-scratch`) or carries the previous segment's final state
forward while *detaching* it from the computational graph (as in
:numref:`fig_truncated_bptt`), no gradient ever crosses a segment boundary,
imposing a hard learning horizon of at most $T_0$ steps :cite:`Jaeger.2002`. In
a vanilla RNN, however, vanishing gradients already choke off the backward
signal after roughly eight steps—well inside our 32-token window—so lengthening
the window alone cannot extend the model's effective memory.

## Summary

Backpropagation through time applies ordinary backpropagation to an unrolled
RNN, propagating loss sensitivities backward via
:eqref:`eq_bptt_partial_L_ht_recur` and summing parameter gradients across time
steps (:eqref:`eq_bptt_param_grads`). Because a gradient traveling back $k$
steps multiplies $k$ state Jacobians—reducing in a linear RNN to
$(\mathbf{W}_\textrm{hh}^\top)^k \sim \rho^k$—gradients explode when $\rho > 1$
and vanish when $\rho < 1$. Including $\tanh$ multiplies each step by
$1 - \mathbf{H}_t^2 \in (0, 1]$, causing gradients in our trained RNN to fall
below 1% within roughly eight steps despite $\rho \approx 1.8$. Gradient
clipping contains exploding spikes, truncated BPTT bounds memory and compute
per update, and architectural innovations (:numref:`chap_modern_rnn`) are
required to preserve gradients over long horizons.

## Exercises

1. **Backward pass by hand.** One step of
   `RNNScratch` (:numref:`sec_rnn-scratch`) computes
   $$\mathbf{H}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}).$$
   Let $\mathbf{G}_t = \partial L / \partial \mathbf{H}_t$ be the gradient
   arriving from later steps and from the output at step $t$.
    1. Derive the contributions of this step to
       $\partial L / \partial \mathbf{W}_{\textrm{xh}}$,
       $\partial L / \partial \mathbf{W}_{\textrm{hh}}$,
       $\partial L / \partial \mathbf{b}_\textrm{h}$, and
       $\partial L / \partial \mathbf{H}_{t-1}$ as functions of
       $\mathbf{G}_t$, $\mathbf{X}_t$, $\mathbf{H}_t$, $\mathbf{H}_{t-1}$,
       and $\mathbf{W}_{\textrm{hh}}$.
    1. Adding the output contribution at step $t-1$ to the contribution
       to $\partial L / \partial \mathbf{H}_{t-1}$ gives a recursion for
       $\mathbf{G}_{t-1}$. Show that it reduces to
       :eqref:`eq_bptt_partial_L_ht_recur` when the $\tanh$ and the bias
       are dropped, and name the factor that the $\tanh$ adds at every step.
       Explain also how the parameter gradients accumulate over the $T$
       steps of a window, and compare with :eqref:`eq_bptt_param_grads`.

    *Adapted from CMU 11-785,
    [Homework 3 Part 1](https://deeplearning.cs.cmu.edu/F22/document/homework/HW3/HW3P1_F22.pdf).*
   :exlabel:`ex_bptt-backward-pass-hand`

1. **Power iteration and alignment.** Let
   $\mathbf{M} \in \mathbb{R}^{h \times h}$ be symmetric with eigenvalues
   $\lambda_i$ and eigenvectors $\mathbf{v}_i$ for $i = 1, \ldots, h$,
   ordered so that $|\lambda_i| \geq |\lambda_{i+1}|$.
    1. Show that $\mathbf{M}^k$ has eigenvalues $\lambda_i^k$ with the
       same eigenvectors.
    1. Show that for a random vector $\mathbf{x} \in \mathbb{R}^h$ with
       independent standard normal entries, $\mathbf{M}^k \mathbf{x}$
       becomes aligned with $\mathbf{v}_1$ as $k$ grows. State the
       condition on the eigenvalues that this requires and the sense in
       which the alignment holds.
    1. What does this imply for the gradient :eqref:`eq_bptt_partial_L_ht`
       of an RNN, beyond the change in its norm?

1. [code] **Backward pass in code.** ● Train `d2l.RNNLMScratch` on
   `d2l.TimeMachine` as in :numref:`sec_rnn-scratch`. Using the
   single-step gradients and the recursion of problem :exref:`ex_bptt-backward-pass-hand`,
   implement the backward pass of its `RNNScratch` cell over a full
   window with tensor operations only, starting from the gradient of the
   loss with respect to the
   logits (:numref:`subsec_softmax_and_derivatives`). Run it on one training
   minibatch, and compare your gradients of `W_xh`, `W_hh`, and `b_h`
   with those computed by automatic differentiation.

    *Adapted from CMU 11-785,
    [Homework 3 Part 1](https://deeplearning.cs.cmu.edu/F22/document/homework/HW3/HW3P1_F22.pdf).*

1. [code] **Nonlinear Jacobians.** ● Draw a random symmetric
   $100 \times 100$ matrix, rescale it to spectral radius
   $\rho \in \{0.9, 1.0, 1.1\}$, and plot the norm of its $k^\textrm{th}$
   power $\|\mathbf{J}^k\|$ against $k$ on a logarithmic axis: three
   straight lines, as the eigenvalues predict. Then leave the symmetric
   linear case.
    1. Replace the symmetric matrix by a general, nonsymmetric
       $\mathbf{W}_\textrm{hh}$ rescaled to each radius, and plot
       $\|\mathbf{J}^k\|$ against $k$. Report the largest value of
       $\|\mathbf{J}^k\|$ at $\rho = 0.9$ and the lag at which it occurs.
       Explain why the ratio $\|\mathbf{J}^k\| / \rho^k$ can exceed one
       and why it stays bounded as $k$ grows (a random matrix is
       diagonalizable with probability one).
    1. Return to the symmetric $\mathbf{W}_\textrm{hh}$ and run the
       $\tanh$ RNN
       $$\mathbf{H}_t = \tanh(\mathbf{H}_{t-1}\mathbf{W}_\textrm{hh} + \mathbf{X}_t), \qquad \mathbf{H}_0 = \mathbf{0},$$
       for a single sequence ($\mathbf{H}_t \in \mathbb{R}^{1 \times h}$)
       with inputs $\mathbf{X}_t$ whose entries are drawn independently
       from $\mathcal{N}(0, s^2)$.
       Plot the norm of the product of the Jacobians
       $\partial \mathbf{H}_j / \partial \mathbf{H}_{j-1}$ along the
       trajectory for $s = 0.1$ and $s = 1$. How do the nonlinearity and
       the input scale change the growth of the norm, and why?
    1. Explain why a largest singular value of $\mathbf{W}_\textrm{hh}$
       below one is sufficient for the product to shrink, whereas a
       largest singular value above one is necessary but not sufficient
       for it to grow :cite:`bengio1994learning,Pascanu.Mikolov.Bengio.2013`.
       Relate the argument to your $\tanh$ curves at $\rho = 0.9$ and
       $\rho = 1.1$.

1. [code] **Window length and memory horizon.** ● In the model of
   :numref:`subsec_vanishing-exploding`, the gradient of the last loss
   falls below 1% of its size within about eight steps, well inside the
   training window of 32 tokens.
    1. Before running anything: if the model is retrained with
       `num_steps=128`, do you expect this horizon to move? Argue from the
       product of state Jacobians in :eqref:`eq_bptt_partial_L_ht`.
    1. Retrain with `num_steps` set to 128, measure the gradient against
       the lag on stretches of 256 validation tokens, and compare the
       result with your prediction.
    1. Repeat the loss-by-position measurement of
       :numref:`sec_rnn-scratch` for the retrained model. Does the loss
       now keep improving beyond the third token of context?

[Discussions](https://d2l.discourse.group/t/334)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §8.6]{.kicker}

Backpropagation Through Time<br>**unrolled graphs · products of Jacobians · vanishing and exploding gradients · truncation**
:::
:::

::: {.slide title="Forty Steps Shrink a Gradient to 1.5% or Inflate It 45-Fold"}
[The problem]{.kicker}

- Scalar recurrence $h_t = w_\textrm{h} h_{t-1} + x_t$ unrolls to
  $h_t = \sum_{s \leq t} w_\textrm{h}^{t-s} x_s$.
- So $\partial h_t / \partial h_{t-k} = w_\textrm{h}^k$.
- $w_\textrm{h} = 0.9$: $\;0.9^{40} \approx 0.015$.
  $\quad w_\textrm{h} = 1.1$: $\;1.1^{40} \approx 45$.
- Without clipping, most runs of our RNN language model fell apart; with
  it, the model used only about three tokens of context. Both effects
  start here.

::: {.d2l-note .warn}
Where do these factors come from in a real RNN, and which of them can we fix?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- unroll a recurrence and write its gradient as a sum of Jacobian products;
- write the backward recursion of a linear RNN and read off powers of
  $\mathbf{W}_\textrm{hh}^\top$;
- predict vanishing or explosion from the spectral radius $\rho$, and say
  what a $\tanh$ changes;
- measure the gradient against the lag in a trained model;
- say why clipping fixes explosion but not vanishing;
- explain what truncation saves and what it gives up.

**Running example:** the RNN language model on *The Time Machine*
(BPE-1024, 128 hidden units, windows of 32 tokens, clipping at norm 1).
:::

::: {.slide title="The Gradient of the Recurrence Behind the Language Model"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Working with Sequences | autoregression, latent state, rollout error |
| From Text to Tokens | characters, words, bytes, BPE |
| Language Models | $n$-grams, perplexity, bits per byte |
| Recurrent Neural Networks | a hidden state updated at every step |
| Implementing RNNs | an RNN language model, scratch and concise |
| **Backpropagation Through Time** | **gradients through time, truncation** |
| Decoding and Generation | greedy decoding, beam search, sampling |
:::

::: {.col .narrow}
- **Uses:** the chain rule, the RNN update, gradient clipping.
- **Skips:** gated and linear-recurrent cells (State Space Models chapter).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The Unrolled Graph]{.dtitle}

[a deep network with shared weights]{.dsub}
:::
:::

::: {.slide title="Unrolling Turns a Recurrence Into a Deep Network With Shared Weights"}
[Setup]{.kicker}

$$h_t = g(h_{t-1}, x_t, w_\textrm{h}), \qquad o_t = f(h_t, w_\textrm{o}),
\qquad L = \tfrac{1}{T}\textstyle\sum_{t=1}^T l(y_t, o_t)$$

- One layer per time step; every layer uses the **same** $w_\textrm{h}$.
- A weight used in many places gets the **sum** of its gradients there,
  as a convolution kernel sums over positions.
- $w_\textrm{o}$ is easy: it reaches $L$ through each $o_t$ directly.
- $w_\textrm{h}$ is hard: it acts at step $t$ **and** through every earlier
  state.
:::

::: {.slide title="I. The Chain Rule Gives a Recursion Over Time"}
[Derivation]{.kicker}

Write $g_t = g(h_{t-1}, x_t, w_\textrm{h})$:

$$\frac{\partial h_t}{\partial w_\textrm{h}} =
\frac{\partial g_t}{\partial w_\textrm{h}} +
\frac{\partial g_t}{\partial h_{t-1}}\,
\frac{\partial h_{t-1}}{\partial w_\textrm{h}}$$

Same form as $a_t = u_t + c_t a_{t-1}$, $a_0 = 0$, which unrolls to

$$a_t = u_t + \sum_{i=1}^{t-1}\Big(\prod_{j=i+1}^{t} c_j\Big) u_i.$$
:::

::: {.slide title="II. Each Earlier Step Arrives Through a Product of Jacobians"}
[Derivation]{.kicker}

$$\frac{\partial h_t}{\partial w_\textrm{h}} =
\frac{\partial g_t}{\partial w_\textrm{h}} +
\sum_{i=1}^{t-1}\Big(\prod_{j=i+1}^{t}\frac{\partial g_j}{\partial h_{j-1}}\Big)
\frac{\partial g_i}{\partial w_\textrm{h}}$$

- Term $i$: the effect of $w_\textrm{h}$ at step $i$, carried to step $t$
  by $t - i$ state Jacobians.
- For $g = w_\textrm{h} h_{t-1} + x_t$ every Jacobian is $w_\textrm{h}$:
  the term is scaled by $w_\textrm{h}^{t-i}$.
:::

::: {.slide title="Check: How Much of the Gradient Survives Three Steps?"}
[Check]{.kicker}

For $h_t = 0.5\, h_{t-1} + x_t$, what is $\partial h_5 / \partial h_2$?
And for $h_t = 2\, h_{t-1} + x_t$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Three Jacobians of $0.5$: $0.5^3 = 0.125$. With $2$:
$2^3 = 8$. After 20 steps the two would be $10^{-6}$ and $10^{6}$.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Vanishing and Exploding Gradients]{.dtitle}

[the linear RNN, where the product is a matrix power]{.dsub}
:::
:::

::: {.slide title="Each State Collects Gradient From Its Output and From the Next State"}
[Linear RNN]{.kicker}

$\mathbf{H}_t = \mathbf{X}_t\mathbf{W}_\textrm{xh} + \mathbf{H}_{t-1}\mathbf{W}_\textrm{hh}$,
$\;\mathbf{O}_t = \mathbf{H}_t\mathbf{W}_\textrm{hq}$,
$\;\mathbf{G}_t = \partial L/\partial\mathbf{H}_t$:

![Forward pass (grey) and backward pass (red) of a linear RNN over three steps.](../img/mdl-rnn-bptt.svg){width=50%}

$$\mathbf{G}_t = \tfrac{\partial L}{\partial \mathbf{O}_t}\mathbf{W}_\textrm{hq}^\top +
\mathbf{G}_{t+1}\mathbf{W}_\textrm{hh}^\top$$
:::

::: {.slide title="The Loss $k$ Steps Ahead Arrives Through $(\mathbf{W}_\textrm{hh}^\top)^k$"}
[Unrolled]{.kicker}

$$\mathbf{G}_t = \sum_{j=t}^{T} \frac{\partial L}{\partial \mathbf{O}_j}\,
\mathbf{W}_\textrm{hq}^\top\big(\mathbf{W}_\textrm{hh}^\top\big)^{j-t}$$

The weight gradients add one term per step, each containing $\mathbf{G}_t$:

$$\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}} = \sum_{t}
\mathbf{H}_{t-1}^\top \mathbf{G}_t, \qquad
\frac{\partial L}{\partial \mathbf{W}_\textrm{xh}} = \sum_{t}
\mathbf{X}_t^\top \mathbf{G}_t$$
:::

::: {.slide title="The Spectral Radius Decides: $\rho < 1$ Vanishes, $\rho > 1$ Explodes"}
[Eigenvalues]{.kicker}

$\rho$ = largest $|\lambda|$ over the eigenvalues of $\mathbf{W}_\textrm{hh}$
(the same for $\mathbf{W}_\textrm{hh}^\top$).

- Along an eigenvector, the $k^\textrm{th}$ power multiplies by
  $\lambda^k$.
- $\rho < 1$: every distant contribution **vanishes** like $\rho^k$.
- $\rho > 1$: the top eigen-direction **explodes** like $\rho^k$.
- $\rho = 1$: the boundary between the two.

::: {.d2l-note .rule}
The scalar $w_\textrm{h}^k$ again, with $\rho$ in the role of $|w_\textrm{h}|$.
:::
:::

::: {.slide title="A tanh Multiplies Every Step by a Factor Between 0 and 1"}
[The real model]{.kicker}

With $\mathbf{H}_t = \tanh(\cdots)$, one step back multiplies entrywise
($\odot$) by the $\tanh$ derivative, then by the weights:

$$\mathbf{G}_{t-1} \leftarrow
\big(\mathbf{G}_t \odot (1 - \mathbf{H}_t^2)\big)\,\mathbf{W}_\textrm{hh}^\top$$

- $1 - h^2 \in (0, 1]$: never lengthens a gradient; near 0 for a saturated
  unit.
- It pushes toward vanishing. Explosion remains possible when
  $\mathbf{W}_\textrm{hh}$ is large and the states are near zero.
- A nonsymmetric $\mathbf{W}_\textrm{hh}$: powers can grow for a while even
  when $\rho < 1$; $\rho^k$ wins for large $k$.
:::

::: {.slide title="Trained Model: $\rho \approx 1.8$, Yet the Gradient Falls Below 1% in Eight Steps"}
[Experiment]{.kicker}

Loss at the last of 64 validation tokens, backpropagated by hand; gradient
norm at lag $k$ relative to lag 0:

@!bptt-vanishing-and-exploding-gradients-6

- $\rho > 1$: a linear recurrence would explode. The mean $\tanh$
  derivative, about 0.27, leaves about half of the gradient per step.
- It agrees with the loss by position: no gain beyond about three tokens
  of context.
:::

::: {.slide title="Check: What Reaches the Start of a 32-Token Window?"}
[Check]{.kicker}

Our language model backpropagates through windows of 32 tokens. If the
recurrence had $\rho = 0.9$ or $\rho = 1.1$, by roughly what factor would the
loss at the end of a window reach the first state?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** 31 Jacobians lie between them: about $0.9^{31} \approx 0.04$
or $1.1^{31} \approx 19$. Moving $\rho$ from 10% below 1 to 10% above
changes the factor by almost three orders of magnitude.
:::
:::
:::

::: {.slide title="Clipping Bounds the Step but Cannot Fix Vanishing"}
[Remedies]{.kicker}

- **Explosion:** one huge update can wreck the parameters. Clipping bounds
  the step: $\mathbf{g} \leftarrow \min(1, \theta/\|\mathbf{g}\|)\,\mathbf{g}$.
- The gradient is a sum over lags; rescaling multiplies every term by the
  same factor, so the balance between near and far lags stays.
- **Vanishing:** the distant terms are tiny next to the recent ones, and
  there is no large step to limit.

::: {.d2l-note .warn}
Vanishing needs a different recurrence, not a different update rule.
:::
:::

::: {.slide title="Vanishing Needs a Recurrence Whose Jacobians Stay Near One"}
[Remedies]{.kicker}

- **Gating (LSTM, GRU):** the state is copied from step to step, scaled by a
  gate set near 1 when the model must remember; Jacobian $\approx$ identity.
- **Linear recurrences:** no nonlinearity in the state update; the state
  matrix is parameterized so its eigenvalues have magnitude at most 1.
  No explosion, and eigenvalues near 1 keep a long memory.

Both are the subject of the State Space Models chapter.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Truncation]{.dtitle}

[bounded cost, bounded memory]{.dsub}
:::
:::

::: {.slide title="Full BPTT on The Time Machine Would Store 66,000 States per Update"}
[Cost]{.kicker}

- The backward recursion needs every state $\mathbf{H}_1, \ldots,
  \mathbf{H}_T$ of the forward pass: memory grows linearly with $T$.
- Each update needs a forward and a backward pass over the whole sequence.
- *The Time Machine* is about 66,000 BPE tokens: one sequence means one
  update per pass through the book.
:::

::: {.slide title="Truncation Cuts the Gradient at Segment Boundaries, Not the State"}
[Truncated BPTT]{.kicker}

![Full (top) and truncated (bottom) backpropagation: the state crosses each boundary forward, the gradient does not.](../img/mdl-rnn-truncated-bptt.svg){width=88%}

Carry the state across each boundary and **detach** it: same values, no
gradient. Without the detach the graph grows with every segment.
:::

::: {.slide title="Truncation Cuts Each Loss Off at the Start of Its Segment"}
[What it gives up]{.kicker}

- In a 32-token segment, the loss at position $j$ reaches back $j - 1$
  steps: at most 31, about 16 on average.
- No gradient ever connects two segments: no dependency longer than a
  segment can be learned.
- Our model: 32-token windows from a zero state. Its gradient fades
  below 1% after about eight steps, so the fading, not the window, sets
  its limit.
:::

::: {.slide title="Check: Can a 32-Token Window Learn to Close a Quote Opened 50 Tokens Ago?"}
[Check]{.kicker}

The model sees windows of 32 tokens, each from a zero state. A quotation
opens 50 tokens before it closes. Can training teach the model to close it?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Not from that dependency: the opening and the closing quote are
never in the same window, so no gradient links them. The model can only
learn cues that appear inside one window.
:::
:::
:::

::: {.slide title="Gradients Through Time Are Products of Jacobians"}
[Wrap-up]{.kicker}

- BPTT = backpropagation on the unrolled recurrence; shared weights sum
  their gradients over steps.
- The gradient from $k$ steps away carries $k$ Jacobians; in a linear RNN,
  $(\mathbf{W}_\textrm{hh}^\top)^k \sim \rho^k$: $0.9^{40} \approx 0.015$,
  $1.1^{40} \approx 45$.
- Trained model: $\rho \approx 1.8$, mean $\tanh$ derivative 0.27, gradient
  below 1% within about eight steps.
- Clipping tames explosion; vanishing needs gates or stable linear
  recurrences.
- Truncation bounds memory and compute; no dependency longer than a segment.

**Next:** turning the trained model's next-token distributions into text.
:::
