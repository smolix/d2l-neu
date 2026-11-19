# Implementing RNNs
:label:`sec_rnn-scratch`

In :numref:`sec_rnn`, we wrote down the equations for a recurrent language
model and tallied its parameters, but left two essential questions unanswered:
how those equations translate into clean, working code, and how well the
trained hidden state actually performs on *The Time Machine* compared with the
count-based models of :numref:`sec_language-model`. The first question is
straightforward. The entire model rests on three operations—looking up the
embedding of the current token, updating the hidden state from that embedding
and the previous state, and projecting the updated state into logits over the
vocabulary—and its interface is correspondingly simple: at each step, the
model accepts one token per sequence alongside the previous hidden state, and
returns $|\mathcal{V}|$ logits per sequence together with an updated state of
identical shape. Because those tensor shapes never vary across steps, the very
same transition handles sequences of arbitrary length.

We first build this language model from scratch using plain tensor operations
so that every parameter and shape is transparent, and later replace the manual
loop with each framework's built-in recurrent layer. Along the way, training
demands one practical safeguard absent from the forward equations: because
gradients propagate through the same transition matrix at every time step, an
occasional minibatch can produce an enormous gradient that wipes out many
iterations of progress in a single update. We therefore clip the global
gradient norm prior to each step—a simple, standard precaution—and run an
ablation without clipping to observe the instability firsthand. We then
evaluate the trained RNN head to head against interpolated Kneser--Ney
$n$-grams on the exact same held-out tokens, dissect its loss position by
position to see how far back in its context window it actually looks, and
finally generate sample text.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input #rnn-implementation-implementing-rnn-language-models}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #rnn-implementation-implementing-rnn-language-models}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #rnn-implementation-implementing-rnn-language-models}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #rnn-implementation-implementing-rnn-language-models}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #rnn-implementation-implementing-rnns-1}
%%tab jax
import random
```

## An RNN Language Model from Scratch

Our language model breaks cleanly into three components, each implemented as
its own class or method: a *recurrent cell* that advances the hidden state one
time step at a time, an *embedding table* that converts integer token ids into
dense vectors, and an *output layer* that maps each hidden state to vocabulary
logits.

### The Recurrent Cell

For a minibatch of $b$ sequences, let
$\mathbf{X}_t \in \mathbb{R}^{b \times d}$ denote the input vectors at step
$t$ and $\mathbf{H}_t \in \mathbb{R}^{b \times h}$ the corresponding hidden
states. The cell evaluates the recurrence from :numref:`sec_rnn` using a
$\tanh$ nonlinearity,

$$\mathbf{H}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),$$

combining the current input and the previous state via two weight matrices,
$\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$ and
$\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$, plus a bias vector
$\mathbf{b}_\textrm{h} \in \mathbb{R}^h$. The `RNNScratch` class stores
these three parameters, where `num_hiddens` ($h$) is a tunable hyperparameter
and `num_inputs` ($d$) matches the width of the incoming step vectors (shortly
the embedding dimension).

Following :numref:`sec_oo-design`, we register each array as a framework
parameter so that optimizers track them and the trainer can place them on a
GPU. Both weight matrices are drawn from $\mathcal{N}(0, \sigma^2)$ with
$\sigma = 0.01$, and the bias is initialized to zero. At this scale, the
spectral radius of $\mathbf{W}_{\textrm{hh}}$ is roughly
$\sigma\sqrt{h} \approx 0.11$: recalling our scalar example in
:numref:`sec_rnn`, an untrained cell decays its past state by nearly an order
of magnitude per step and forgets its history almost immediately, leaving
training to grow larger recurrent connections where needed.

:begin_tab:`tensorflow`
Each weight gets its own initializer object. An unseeded Keras initializer
returns the same values every time it is called, so one shared initializer
would make the first rows of $\mathbf{W}_{\textrm{hh}}$ a copy of
$\mathbf{W}_{\textrm{xh}}$.
:end_tab:

```{.python .input #rnn-implementation-the-recurrent-cell-1}
%%tab mxnet
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = gluon.Parameter('W_xh', shape=(num_inputs, num_hiddens),
                                    init=init.Normal(sigma))
        self.W_hh = gluon.Parameter('W_hh', shape=(num_hiddens, num_hiddens),
                                    init=init.Normal(sigma))
        self.b_h = gluon.Parameter('b_h', shape=(num_hiddens,),
                                   init=init.Zero())
        self.initialize()
```

```{.python .input #rnn-implementation-the-recurrent-cell-1}
%%tab pytorch
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(
            d2l.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(
            d2l.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
```

```{.python .input #rnn-implementation-the-recurrent-cell-1}
%%tab tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = self.add_weight(  # One initializer per weight
            name='W_xh', shape=(num_inputs, num_hiddens),
            initializer=tf.keras.initializers.RandomNormal(stddev=sigma))
        self.W_hh = self.add_weight(
            name='W_hh', shape=(num_hiddens, num_hiddens),
            initializer=tf.keras.initializers.RandomNormal(stddev=sigma))
        self.b_h = self.add_weight(name='b_h', shape=(num_hiddens,),
                                   initializer='zeros')
```

```{.python .input #rnn-implementation-the-recurrent-cell-1}
%%tab jax
class RNNScratch(nnx.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01, rngs=None):
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.sigma = sigma
        self.W_xh = nnx.Param(
            rngs.params.normal((num_inputs, num_hiddens)) * sigma)
        self.W_hh = nnx.Param(
            rngs.params.normal((num_hiddens, num_hiddens)) * sigma)
        self.b_h = nnx.Param(jnp.zeros(num_hiddens))
```

The forward pass iterates over the leading time axis of `inputs`—which has
shape (`num_steps`, `batch_size`, `num_inputs`)—updating the hidden state at
each step and collecting all intermediate states in a list. Whenever no
initial state is provided, the recurrence starts from an all-zeros tensor.
This explicit Python loop mirrors our mathematical recurrence directly,
prioritizing clarity over speed for now.

:begin_tab:`mxnet`
The MXNet version reads the values of the parameters with `data()`, here and
in the language model below.
:end_tab:

:begin_tab:`jax`
When the training step is compiled, the Python loop is unrolled into one
copy of the cell per time step, so the compilation time grows with the
number of steps. The concise implementation at the end of this section
uses `nnx.RNN`, which runs the cell under `jax.lax.scan` and compiles it
only once.
:end_tab:

```{.python .input #rnn-implementation-the-recurrent-cell-2}
%%tab mxnet
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
        state = d2l.tanh(d2l.matmul(X, self.W_xh.data()) +
                         d2l.matmul(state, self.W_hh.data()) +
                         self.b_h.data())
        outputs.append(state)
    return outputs, state
```

```{.python .input #rnn-implementation-the-recurrent-cell-2}
%%tab pytorch
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input #rnn-implementation-the-recurrent-cell-2}
%%tab tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        state = tf.zeros((tf.shape(inputs)[1], self.num_hiddens))
    outputs = []
    for X in tf.unstack(inputs):  # Shape: (num_steps, batch_size, num_inputs)
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input #rnn-implementation-the-recurrent-cell-2}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        state = jnp.zeros((inputs.shape[1], self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

Let us test the cell on a dummy minibatch of two 100-step sequences.

```{.python .input #rnn-implementation-the-recurrent-cell-3}
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

We expect one hidden state per time step, each matching the
(`batch_size`, `num_hiddens`) shape of the initial state. The `check_shape`
helper below verifies tensor shapes and will be reused in later sections.

```{.python .input #rnn-implementation-the-recurrent-cell-4}
def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_shape(d2l.stack(outputs, 0), (num_steps, batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

### The Embedding Table

Because our recurrent cell expects continuous vectors while the data pipeline
supplies integer token ids, we bridge the two with a learned embedding table
$\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ (:numref:`sec_rnn`),
mapping token $i$ to its $i^\textrm{th}$ row and updating those vectors
jointly with the rest of the network. Classical presentations often represent
token $i$ instead as a *one-hot vector* of length $|\mathcal{V}|$ with a
single 1 in slot $i$ and multiply it by the input weight matrix. As noted in
:numref:`sec_rnn`, multiplying a one-hot vector by a matrix merely selects one
row—the exact same operation as a table lookup, only far more wasteful. With
27 characters the overhead is negligible, but with our 1,027-token BPE
vocabulary it would spend almost all its work multiplying by zeros.

The `RNNLMScratch` class wraps a recurrent cell `rnn` with an embedding table
`E` on the input side and a linear projection (`W_hq`, `b_q`) on the output
side, reading the embedding dimension directly from `rnn.num_inputs`. It also
accepts two dropout rates—left at zero for our first baseline and turned on
shortly to curb overfitting. Embedding rows are initialized at unit scale
(matching the scale of one-hot inputs), while output weights are drawn at
scale $\sigma$ with zero bias. Consequently, an untrained model starts with
near-zero logits, an almost uniform predictive distribution, and an initial
perplexity close to $|\mathcal{V}|$—a clean sanity check we can verify before
running a single gradient step.

Because `RNNLMScratch` inherits from `d2l.Classifier`, it reuses the standard
cross-entropy loss averaged over every token in a minibatch. We override
`validation_step` to return only that loss rather than token accuracy; since
mean cross-entropy is the logarithm of perplexity
(:numref:`subsec_perplexity`), exponentiating the trainer's mean validation
loss immediately yields validation perplexity.

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-1}
%%tab mxnet
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01, dropout=0, emb_dropout=0):
        super().__init__()
        self.save_hyperparameters()
        self.sigma = getattr(rnn, 'sigma', 0.01)  # Scale of the output weights
        self.init_params()

    def init_params(self):
        self.E = gluon.Parameter(
            'E', shape=(self.vocab_size, self.rnn.num_inputs),
            init=init.Normal(1))
        self.W_hq = gluon.Parameter(
            'W_hq', shape=(self.rnn.num_hiddens, self.vocab_size),
            init=init.Normal(self.sigma))
        self.b_q = gluon.Parameter('b_q', shape=(self.vocab_size,),
                                   init=init.Zero())
        for param in (self.E, self.W_hq, self.b_q):  # The cell has its own
            param.initialize()

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-1}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01, dropout=0, emb_dropout=0):
        super().__init__()
        self.save_hyperparameters()
        self.sigma = getattr(rnn, 'sigma', 0.01)  # Scale of the output weights
        self.init_params()

    def init_params(self):
        self.E = nn.Parameter(
            d2l.randn(self.vocab_size, self.rnn.num_inputs))
        self.W_hq = nn.Parameter(
            d2l.randn(self.rnn.num_hiddens, self.vocab_size) * self.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size))

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-1}
%%tab tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01, dropout=0, emb_dropout=0):
        super().__init__()
        self.save_hyperparameters()
        self.sigma = getattr(rnn, 'sigma', 0.01)  # Scale of the output weights
        self.init_params()

    def init_params(self):
        self.E = self.add_weight(
            name='E', shape=(self.vocab_size, self.rnn.num_inputs),
            initializer=tf.keras.initializers.RandomNormal(stddev=1))
        self.W_hq = self.add_weight(
            name='W_hq', shape=(self.rnn.num_hiddens, self.vocab_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma))
        self.b_q = self.add_weight(name='b_q', shape=(self.vocab_size,),
                                   initializer='zeros')

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1], training=False), batch[-1])
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-1}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01, dropout=0, emb_dropout=0,
                 rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rnn', 'rngs'])
        self.rnn = rnn
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        sigma = getattr(rnn, 'sigma', 0.01)  # Scale of the output weights
        self.E = nnx.Param(rngs.params.normal((vocab_size, rnn.num_inputs)))
        self.W_hq = nnx.Param(rngs.params.normal(
            (rnn.num_hiddens, vocab_size)) * sigma)
        self.b_q = nnx.Param(jnp.zeros(vocab_size))
        self.drop = nnx.Dropout(dropout, rngs=rngs)
        self.emb_drop = nnx.Dropout(emb_dropout, rngs=rngs)

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])
```

Because our data loader emits minibatches of shape (`batch_size`,
`num_steps`), the `embedding` method transposes the token indices before
indexing `E`, producing a tensor of shape (`num_steps`, `batch_size`,
`num_inputs`) whose outermost axis aligns with the cell's time loop.

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-2}
%%tab mxnet
@d2l.add_to_class(RNNLMScratch)  #@save
def embedding(self, X):
    # Output shape: (num_steps, batch_size, num_inputs)
    return self.E.data()[X.T]
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-2}
%%tab pytorch
@d2l.add_to_class(RNNLMScratch)  #@save
def embedding(self, X):
    # Output shape: (num_steps, batch_size, num_inputs)
    return self.E[X.T]
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-2}
%%tab tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def embedding(self, X):
    # Output shape: (num_steps, batch_size, num_inputs)
    return tf.gather(self.E, tf.transpose(X))
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-2}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def embedding(self, X):
    # Output shape: (num_steps, batch_size, num_inputs)
    return self.E[X.T]
```

### The Output Layer

The final equation converts each hidden state into vocabulary logits via
$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}$,
where $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times |\mathcal{V}|}$.
Because all time steps share the same output projection, we can compute every
step in a single batched matrix multiplication by stacking the per-step states
into a (`batch_size`, `num_steps`, `num_hiddens`) tensor, producing logits of
shape (`batch_size`, `num_steps`, `vocab_size`) aligned directly with the
target array. The `forward` method chains embedding lookup, recurrent cell,
optional dropout, and output projection together.

```{.python .input #rnn-implementation-the-output-layer-1}
%%tab mxnet
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    # A list of per-step states, or a tensor whose first axis is time
    H = d2l.stack(list(rnn_outputs), 1)  # (batch_size, num_steps, num_hiddens)
    return d2l.matmul(H, self.W_hq.data()) + self.b_q.data()

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    # npx.dropout is active only inside autograd.record, that is, in training
    embs = npx.dropout(self.embedding(X), p=self.emb_dropout)
    rnn_outputs, _ = self.rnn(embs, state)
    H = d2l.stack(list(rnn_outputs), 0)  # (num_steps, batch_size, num_hiddens)
    return self.output_layer(npx.dropout(H, p=self.dropout))
```

```{.python .input #rnn-implementation-the-output-layer-1}
%%tab pytorch
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    # A list of per-step states, or a tensor whose first axis is time
    H = d2l.stack(list(rnn_outputs), 1)  # (batch_size, num_steps, num_hiddens)
    return d2l.matmul(H, self.W_hq) + self.b_q

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = F.dropout(self.embedding(X), self.emb_dropout, self.training)
    rnn_outputs, _ = self.rnn(embs, state)
    H = d2l.stack(list(rnn_outputs), 0)  # (num_steps, batch_size, num_hiddens)
    return self.output_layer(F.dropout(H, self.dropout, self.training))
```

```{.python .input #rnn-implementation-the-output-layer-1}
%%tab tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    # A list of per-step states, or a tensor whose first axis is time
    H = d2l.stack(list(rnn_outputs), 1)  # (batch_size, num_steps, num_hiddens)
    return d2l.matmul(H, self.W_hq) + self.b_q

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.embedding(X)
    if self.training:  # tf.nn.dropout is always active; apply it in training
        embs = tf.nn.dropout(embs, self.emb_dropout)
    rnn_outputs, _ = self.rnn(embs, state)
    if self.training:  # A list of per-step states (scratch) or one tensor
        rnn_outputs = tf.nest.map_structure(
            lambda H: tf.nn.dropout(H, self.dropout), rnn_outputs)
    return self.output_layer(rnn_outputs)
```

```{.python .input #rnn-implementation-the-output-layer-1}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    # A list of per-step states, or a tensor whose first axis is time
    H = d2l.stack(list(rnn_outputs), 1)  # (batch_size, num_steps, num_hiddens)
    return d2l.matmul(H, self.W_hq) + self.b_q

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.emb_drop(self.embedding(X))
    rnn_outputs, _ = self.rnn(embs, state)
    H = d2l.stack(list(rnn_outputs), 0)  # (num_steps, batch_size, num_hiddens)
    return self.output_layer(self.drop(H))
```

Passing a minibatch of shape (`batch_size`, `num_steps`) should therefore
return one logit vector per token position. Let us verify this shape using a
1,024-token vocabulary.

```{.python .input #rnn-implementation-the-output-layer-2}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, vocab_size=1024)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, 1024))
```

```{.python .input #rnn-implementation-the-output-layer-2}
%%tab jax
model = RNNLMScratch(rnn, vocab_size=1024)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, 1024))
```

## Gradient Clipping

Because an unrolled RNN is deep in time, backpropagating across $T$ steps
multiplies $T$ copies of the transition Jacobian, and—just as in deep
feedforward networks—such long products can shrink or grow exponentially
with $T$ (a phenomenon we analyze rigorously in :numref:`sec_bptt`). While
vanishing gradients restrict how far into the past a model can learn without
destabilizing training, exploding gradients can derail optimization
outright, producing occasional gradient spikes orders of magnitude larger than
typical updates.

To understand the danger, consider the size of a single SGD update with
learning rate $\eta$: moving from $\mathbf{w}$ to
$\mathbf{w} - \eta\mathbf{g}$ shifts the parameters by Euclidean distance

$$\|(\mathbf{w} - \eta\mathbf{g}) - \mathbf{w}\| = \eta\|\mathbf{g}\|.$$

Gradient descent succeeds because over sufficiently small steps the loss is
well approximated by its first-order Taylor expansion
$f(\mathbf{w}) - \eta\|\mathbf{g}\|^2$, which is guaranteed to decrease.
When $\|\mathbf{g}\|$ suddenly spikes, however, the update overshoots the
region where that linear approximation holds, and the loss can jump sharply
upward, undoing dozens of productive updates in a single step. Lowering $\eta$
across the board to survive rare spikes would slow training to a crawl.
*Gradient clipping* avoids that penalty by leaving $\eta$ untouched and
instead capping the global gradient norm whenever it exceeds a threshold
$\theta$, projecting $\mathbf{g}$ onto the ball of radius $\theta$:

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$
:eqlabel:`eq_grad_clipping`

Any gradient with $\|\mathbf{g}\| \leq \theta$ passes through untouched,
whereas a larger gradient is rescaled to norm $\theta$ while preserving its
direction (for example, with $\theta = 1$, a gradient of norm 10 is shrunk by
a factor of ten). Here $\|\mathbf{g}\|$ denotes the Euclidean norm computed
jointly across all trainable parameters of the model concatenated into one
vector.

Admittedly, gradient clipping is a somewhat blunt heuristic: whenever it
triggers, the rescaled step no longer equals the true stochastic gradient,
complicating formal convergence analysis. Even so, it is a workhorse of modern
deep learning, relied on routinely when training both recurrent networks and
large Transformers.

Clipping belongs naturally inside the training loop: whenever
`gradient_clip_val` is positive, `d2l.Trainer` (:numref:`sec_oo-design`)
rescales the gradients right after backpropagation and before applying the
optimizer update.

:begin_tab:`mxnet, pytorch`
The method below rescales each parameter's gradient in place.
:end_tab:

:begin_tab:`tensorflow`
The method below returns the (possibly rescaled) gradients, and the compiled
training step hands them to the optimizer.
:end_tab:

:begin_tab:`jax`
The JAX trainer updates the parameters with an optax transformation inside
its compiled training step, so the clipping goes there too:
`prepare_optimizer` turns `clip_gradients` into an optax transformation and
chains it in front of the model's optimizer.
:end_tab:

```{.python .input #rnn-implementation-gradient-clipping}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p.data() for p in model.collect_params().values()
              if p.grad_req != 'null']
    norm = mnp.sqrt(sum((p.grad ** 2).sum() for p in params))
    scale = mnp.minimum(grad_clip_val / norm, 1.0)  # no `if` on the host
    for param in params:
        param.grad[:] *= scale
```

```{.python .input #rnn-implementation-gradient-clipping}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    scale = torch.clamp(grad_clip_val / norm, max=1.0)  # no `if` on the host
    for param in params:
        param.grad.mul_(scale)
```

```{.python .input #rnn-implementation-gradient-clipping}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    scale = tf.minimum(1.0, grad_clip_val / norm)
    return [grad * scale for grad in new_grads]
```

```{.python .input #rnn-implementation-gradient-clipping}
%%tab jax
def clip_gradients(grad_clip_val, grads):  #@save
    """Rescale the gradients to global norm at most grad_clip_val."""
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_optimizer(self, model):
    val = self.gradient_clip_val
    key = (jax.tree.structure(model), type(model).configure_optimizers, val)
    if key not in d2l.optimizer_cache:
        tx = model.configure_optimizers()
        if val > 0:  # The closure holds only val, not the trainer
            clip = lambda grads, params: clip_gradients(val, grads)
            tx = optax.chain(optax.stateless(clip), tx)
        d2l.optimizer_cache.clear()
        d2l.optimizer_cache[key] = tx
    self.optim = nnx.Optimizer(model, d2l.optimizer_cache[key],
                               wrt=nnx.Param)
```

## Training

The `d2l.TimeMachine` data module from :numref:`sec_language-model` downloads
the novella, trains our 1,024-token BPE tokenizer, and yields minibatches of
fixed-length windows paired with their one-step-shifted targets. The full
encoded corpus spans 65,741 BPE tokens, with training window $i$ beginning at
token $i$. To prevent leakage, the loader discards the 32 windows straddling
the train–validation boundary so that no validation token ever appears as a
training target, leaving the first 50,000 windows for training and the next
5,000 for validation.

```{.python .input #rnn-implementation-training-1}
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
```

Each window spans $T = 32$ tokens and resets the hidden state to zero at the
start of every forward pass. Because neither hidden activations nor gradients
cross window boundaries, backpropagation is *truncated* to at most 32 steps;
any longer-range structure in the novella can be learned only to the extent
that it manifests within a 32-token window. In :numref:`sec_bptt`, we examine
what this truncation sacrifices and how carrying states across windows alters
training.

### A Plain Model

We begin by training the exact baseline whose parameters we tallied in
:numref:`sec_rnn`—embedding dimension $d = 64$, hidden width $h = 128$, and no
dropout—for ten epochs of minibatch SGD with learning rate $\eta = 4$ and
gradient clipping at $\theta = 1$. Although $\eta = 4$ looks unusually large
compared with earlier chapters, remember how it interacts with clipping:
whenever clipping activates, the total parameter update has Euclidean length
$\eta\theta = 4$ regardless of the raw gradient magnitude. Training uses
teacher forcing (:numref:`sec_rnn`), conditioning every step on the ground-truth
prefix. Before starting, we save a copy of the initial embedding table so we
can measure later how much SGD actually moves it.

```{.python .input #rnn-implementation-training-2}
%%tab pytorch, mxnet, jax
rnn = RNNScratch(num_inputs=64, num_hiddens=128)
model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=4)
```

```{.python .input #rnn-implementation-training-2}
%%tab tensorflow
with d2l.try_gpu():
    rnn = RNNScratch(num_inputs=64, num_hiddens=128)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=4)
```

```{.python .input #rnn-implementation-a-plain-model-1}
%%tab mxnet
E_init = d2l.numpy(model.E.data()).copy()
```

```{.python .input #rnn-implementation-a-plain-model-1}
%%tab pytorch, tensorflow
E_init = d2l.numpy(model.E).copy()
```

```{.python .input #rnn-implementation-a-plain-model-1}
%%tab jax
E_init = d2l.numpy(model.E[...]).copy()
```

```{.python .input #rnn-implementation-training-3}
trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
ppl_init = math.exp(trainer.evaluate(model, data)['loss'])
print(f'initial validation perplexity {ppl_init:.0f}')
trainer.fit(model, data)
```

Before the first gradient step, validation perplexity sits right near the
vocabulary size of 1,027—just as expected for a near-uniform predictor. As
training proceeds, both curves (plotting mean per-token cross-entropy, i.e.,
log perplexity) drop rapidly over the opening epochs before diverging: the
training loss keeps decreasing while validation loss plateaus, ending after
ten epochs with training perplexity less than half the validation perplexity.
With 222,915 parameters fitted to 50,000 training tokens—over four parameters
per token—this overfitting is unsurprising: the network has begun memorizing
its training passages. Exponentiating the final validation loss gives the
model's validation perplexity.

```{.python .input #rnn-implementation-training-4}
ppl_plain = math.exp(trainer.history['val_loss'][-1])
print(f'validation perplexity {ppl_plain:.1f}')
```

How much of this smooth convergence do we owe to gradient clipping? To test
what clipping actually prevents, we retrain fresh instances of the exact same
model with clipping disabled. Because any single unclipped run might get
lucky or unlucky, we run three independent trials with fresh random
initializations and data shuffles, recording both the worst intermediate
validation perplexity and the final value for each run.

```{.python .input #rnn-implementation-a-plain-model-2}
%%tab pytorch, mxnet, jax
def unclipped_model():
    return RNNLMScratch(RNNScratch(num_inputs=64, num_hiddens=128),
                        vocab_size=len(data.vocab), lr=4)
```

```{.python .input #rnn-implementation-a-plain-model-2}
%%tab tensorflow
def unclipped_model():
    with d2l.try_gpu():
        return RNNLMScratch(RNNScratch(num_inputs=64, num_hiddens=128),
                            vocab_size=len(data.vocab), lr=4)
```

```{.python .input #rnn-implementation-a-plain-model-3}
ppl_runs = []
for run in range(3):  # Three unseeded runs, no clipping
    trainer_unclipped = d2l.Trainer(max_epochs=10, num_gpus=1)
    trainer_unclipped.board.display = False  # One plot for all runs below
    trainer_unclipped.fit(unclipped_model(), data)
    ppls = [math.exp(loss) for loss in trainer_unclipped.history['val_loss']]
    ppl_runs.append(ppls)
    print(f'run {run + 1}: highest {max(ppls):.3g}, final {ppls[-1]:.3g}')
finals = [ppls[-1] for ppls in ppl_runs]
print(f'{sum(f > 2 * ppl_plain for f in finals)} of 3 runs end above twice '
      f'the clipped {ppl_plain:.3g}; finals from {min(finals):.3g} to '
      f'{max(finals):.3g}')
```

Plotting the epoch-by-epoch validation perplexity on a logarithmic axis
alongside the clipped baseline reveals exactly when and how violently the
unclipped runs destabilize.

```{.python .input #rnn-implementation-a-plain-model-4}
ppl_clipped = [math.exp(loss) for loss in trainer.history['val_loss']]
d2l.plot(list(range(1, 11)), [ppl_clipped] + ppl_runs, 'epoch',
         'validation perplexity', yscale='log',
         legend=['clipped', 'run 1', 'run 2', 'run 3'],
         legend_loc='outside', figsize=(5, 2.5))
```

Without clipping, training becomes highly erratic. Mid-training, a gradient
spike drives the loss sharply upward—in some runs sending validation
perplexity past the 1,027 baseline of an untrained model—and subsequent
epochs recover only partially, if at all. None of these spikes appear in the
clipped curve. Across 32 unclipped trials per framework run outside this
notebook, more than eight in ten finished at over twice the clipped model's
perplexity; only an occasional lucky run avoids a spike and matches the
clipped curve. With clipping enabled, by contrast, convergence is completely
reliable, consistently finishing around perplexity 90 to 100. Simple as it
is, gradient clipping is what makes training at learning rate 4 viable.

At first sight, a perplexity of 90 to 100 might seem disappointing next to
the character-level trigram's perplexity of roughly 6 in
:numref:`sec_language-model`. Remember, however, that perplexities across
different vocabularies are not directly comparable: our RNN chooses among
1,027 BPE tokens per step, whereas the character model chooses among just 27
characters. Switching to bits per byte (:numref:`subsec_bits-per-byte`,
:eqref:`eq_bpb`) removes the tokenization dependence by dividing per-token
surprisal ($\log_2 \textrm{ppl}$) by the average number of UTF-8 bytes per
token, measured here across the 5,000 consecutive tokens starting the
validation windows.

```{.python .input #rnn-implementation-training-5}
ids = d2l.numpy(data.X[data.num_train:data.num_train+data.num_val, 0]).tolist()
bytes_per_token = len(data.tokenizer.decode(ids).encode('utf-8')) / len(ids)
print(f'{bytes_per_token:.2f} bytes/token, '
      f'{math.log2(ppl_plain) / bytes_per_token:.2f} bits per byte')
```

Each BPE token spans roughly 2.8 bytes of text, placing our plain RNN at
about 2.3 to 2.4 bits per byte.

### Embeddings under SGD

Overfitting is only the first weakness of this baseline. A second, subtler
issue hides inside the embedding table, and spotting it requires a direct
measurement: how far did each row of $\mathbf{E}$ actually move over ten
epochs, expressed as a fraction of its initial norm?

```{.python .input #rnn-implementation-embeddings-under-sgd-1}
%%tab mxnet
E_final = d2l.numpy(model.E.data())
```

```{.python .input #rnn-implementation-embeddings-under-sgd-1}
%%tab pytorch, tensorflow
E_final = d2l.numpy(model.E)
```

```{.python .input #rnn-implementation-embeddings-under-sgd-1}
%%tab jax
E_final = d2l.numpy(model.E[...])
```

```{.python .input #rnn-implementation-embeddings-under-sgd-2}
change = (((E_final - E_init) ** 2).sum(1) / (E_init ** 2).sum(1)) ** 0.5
print(f'mean change of an embedding row: {change.mean():.1%} of its length')
```

Remarkably, the embedding rows have scarcely budged. Because row $i$ receives
a gradient only from positions where token $i$ appears, and the minibatch
loss averages across all $1{,}024 \times 32 = 32{,}768$ token positions, a
token occurring only a handful of times per batch contributes only a tiny
fraction of the gradient. Plain SGD applies the exact same global learning
rate to every parameter, which is far too small to shift unit-scale embedding
rows meaningfully (and tokens absent from training receive zero updates
altogether). As a result, the table intended to learn rich semantic token
representations behaves here almost like a frozen random projection.

Rather than switching optimizers right away, we can work around this
bottleneck within SGD itself. Because the input weight matrix
$\mathbf{W}_{\textrm{xh}}$ *does* receive dense gradients at every token and
learns readily, widening the embedding from $d = 64$ to $d = 512$ gives
$\mathbf{W}_{\textrm{xh}}$ a far richer set of random token features to
combine. Pairing this wider embedding with dropout (:numref:`sec_dropout`)—at
rate 0.1 on the embeddings and 0.5 on the hidden states feeding the output
layer—directly tackles overfitting as well.

### A Wider Embedding with Dropout

Putting the two dropout arguments of `RNNLMScratch` to work, we now train a
740,355-parameter model (with most parameters sitting in the
$1{,}027 \times 512$ embedding table), keeping all other training settings
identical.

```{.python .input #rnn-implementation-a-wider-embedding-with-dropout-1}
%%tab pytorch, mxnet, jax
rnn = RNNScratch(num_inputs=512, num_hiddens=128)
model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=4, dropout=0.5,
                     emb_dropout=0.1)
```

```{.python .input #rnn-implementation-a-wider-embedding-with-dropout-1}
%%tab tensorflow
with d2l.try_gpu():
    rnn = RNNScratch(num_inputs=512, num_hiddens=128)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=4, dropout=0.5,
                         emb_dropout=0.1)
```

```{.python .input #rnn-implementation-a-wider-embedding-with-dropout-2}
trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input #rnn-implementation-a-wider-embedding-with-dropout-3}
ppl_scratch = math.exp(trainer.history['val_loss'][-1])
print(f'validation perplexity {ppl_scratch:.1f}, '
      f'{math.log2(ppl_scratch) / bytes_per_token:.2f} bits per byte')
```

Validation perplexity drops dramatically—from roughly 90–100 down to around
60, lowering bits per byte from 2.3–2.4 to roughly 2.1. Note that because the
plotted training loss is evaluated with dropout active (which inflates it),
the two curves no longer reflect the raw generalization gap directly, and the
validation loss remains at or near its minimum through the tenth epoch.

### Comparison with Kneser–Ney

How strong is 2.1 bits per byte in absolute terms? The natural benchmark is
our best count-based model from :numref:`sec_language-model`, interpolated
Kneser--Ney smoothing (:numref:`subsec_kneser-ney`). Recall that
`d2l.kneser_ney` counts $n$-grams on a training stream and returns per-token
log-probabilities on an evaluation stream. To keep the comparison strictly
apples-to-apples, we estimate Kneser--Ney on the exact same 50,000 BPE
training tokens seen by the RNN, and score the validation tokens starting from
`val[32:]` so that the RNN has a full 32-token context preceding every
evaluated token.

```{.python .input #rnn-implementation-training-6}
first_tokens = d2l.numpy(data.X[:, 0]).tolist()  # First token of each window
train = first_tokens[:data.num_train]
val = first_tokens[data.num_train:data.num_train + data.num_val]
lp_kns = {n: d2l.kneser_ney(train, val, len(data.vocab), n)[32:]
          for n in (2, 3, 4)}
for n, lp in lp_kns.items():
    bits = -sum(lp) / len(lp) / math.log(2)
    print(f'KN {n}-gram: {bits / bytes_per_token:.2f} bits per byte')
lp_kn = lp_kns[3]
```

Moving from bigrams to trigrams yields a substantial gain, whereas a fourth
context token helps little further. Next, we score those exact same tokens
`val[32:]` with our trained RNN, evaluating each token as the final target of
its 32-token validation window (window $j$ begins at `val[j]` and ends with
target `val[j + 32]`). Note that this is slightly more favorable to the RNN
than the board's validation loss, which averages across all 32 window
positions including the earliest ones with minimal context. Computing the full
per-position log-probability matrix now also lets us inspect every earlier
position right afterward.

```{.python .input #rnn-implementation-comparison-with-kneser-ney-1}
start, stop = data.num_train, data.num_train + data.num_val - 32  # val[32:]
```

```{.python .input #rnn-implementation-comparison-with-kneser-ney-2}
%%tab mxnet
X_val = data.X[start:stop].as_in_ctx(d2l.try_gpu())
Y_val = data.Y[start:stop].as_in_ctx(d2l.try_gpu())
logp = d2l.numpy(mnp.concatenate([  # Log-probability of every target
    npx.pick(npx.log_softmax(model(X_val[i:i + 1024])), Y_val[i:i + 1024])
    for i in range(0, len(X_val), 1024)]))
```

```{.python .input #rnn-implementation-comparison-with-kneser-ney-2}
%%tab pytorch
model.eval()
with torch.inference_mode():  # Log-probability of every target
    logp = d2l.numpy(torch.cat([
        F.log_softmax(model(X.to(d2l.try_gpu())), -1).gather(
            -1, Y.to(d2l.try_gpu())[..., None])[..., 0]
        for X, Y in zip(data.X[start:stop].split(1024),
                        data.Y[start:stop].split(1024))]))
```

```{.python .input #rnn-implementation-comparison-with-kneser-ney-2}
%%tab tensorflow
X_val, Y_val = data.X[start:stop], data.Y[start:stop]
logp = d2l.numpy(tf.concat([  # Log-probability of every target
    tf.gather(tf.nn.log_softmax(model(X_val[i:i + 1024], training=False)),
              Y_val[i:i + 1024], batch_dims=2)
    for i in range(0, len(X_val), 1024)], 0))
```

```{.python .input #rnn-implementation-comparison-with-kneser-ney-2}
%%tab jax
model_eval = nnx.view(model, deterministic=True, raise_if_not_found=False)
X_val, Y_val = data.X[start:stop], data.Y[start:stop]
logp = d2l.numpy(jnp.concatenate([  # Log-probability of every target
    jnp.take_along_axis(jax.nn.log_softmax(model_eval(X_val[i:i + 1024])),
                        Y_val[i:i + 1024, :, None], -1)[..., 0]
    for i in range(0, len(X_val), 1024)]))
```

Before inspecting the final numbers, notice how complementary the two models
are: the RNN can attend to 32 tokens of history through continuous weights
learned over a few hundred SGD steps, whereas Kneser--Ney looks back only two
tokens yet memorizes every trigram in the corpus exactly. Combining them by
linear interpolation,
$\lambda P_\textrm{RNN} + (1 - \lambda) P_\textrm{KN}$—the exact ensemble
strategy used by :citet:`Mikolov.Kombrink.Burget.ea.2011`—produces a valid
next-token distribution whenever $0 \leq \lambda \leq 1$. Here we simply fix
$\lambda = 0.5$, close to the optimal balance for this pair.

```{.python .input #rnn-implementation-comparison-with-kneser-ney-3}
lp_rnn = list(logp[:, -1])  # The last target of each window: val[32:]
lam = 0.5
lp_mix = [math.log(lam * math.exp(a) + (1 - lam) * math.exp(b))
          for a, b in zip(lp_rnn, lp_kn)]
for name, lp in [('RNN', lp_rnn), ('KN trigram', lp_kn), ('mixture', lp_mix)]:
    bits = -sum(lp) / len(lp) / math.log(2)
    print(f'{name}: {bits / bytes_per_token:.2f} bits per byte')
better = np.mean(np.array(lp_rnn) > np.array(lp_kn))
print(f'the RNN is the more confident model on {better:.0%} of the tokens')
```

Even with dropout and a wider embedding, the standalone RNN still falls a few
hundredths of a bit per byte short of the Kneser--Ney trigram. (Both figures
sit slightly higher than in :numref:`subsec_bits-per-byte` because that table
used lowercased, letters-only text, whereas here we model raw, unnormalized
prose where Kneser--Ney achieves roughly 2.1 bits per byte.) Far more
revealing is the mixture: averaging the two distributions beats the
Kneser--Ney trigram by roughly 0.06 bits per byte—impossible unless the two
models make genuinely different predictions. Sure enough, each model assigns
higher probability than the other on roughly half of all validation tokens:
they achieve comparable overall accuracy, yet fail on different passages.

Still, why does a recurrent network reading 32 tokens fail to outperform a
count table that sees only two? Rather than guessing, we can measure directly
how much context the trained RNN actually exploits. Inside each validation
window, position 1 predicts from a single context token, position 2 from two,
and so on up to 32. Averaging surprisal across all validation windows at each
position traces loss as a function of available context, which we plot against
the Kneser--Ney trigram baseline.

```{.python .input #rnn-implementation-comparison-with-kneser-ney-4}
bits = -logp.mean(0) / math.log(2) / bytes_per_token  # One per position
kn_bits = -sum(lp_kn) / len(lp_kn) / math.log(2) / bytes_per_token
positions = np.arange(1, data.num_steps + 1)
d2l.plot(positions, [bits, [kn_bits] * data.num_steps],
         'tokens of context', 'bits per byte', legend=['RNN', 'KN trigram'],
         figsize=(4.5, 3))
print('bits per byte with 1, 2, 3, 4, 8, 32 tokens of context:',
      [round(float(bits[k - 1]), 2) for k in (1, 2, 3, 4, 8, 32)])
```

The diagnostic curve tells the entire story. Predicting from a single token
incurs the highest cost, and adding the second and third tokens slashes
surprisal sharply—yet beyond roughly the fourth token the curve goes
completely flat: having read thirty tokens, the model predicts no more
accurately than after reading three. In other words, although the hidden state
*could* in principle carry information across all 32 steps, training only
taught it to exploit the last few tokens—roughly the same effective horizon as
the trigram, explaining why the two tie.

Part of this gap stems from optimization: as we saw above, vanilla SGD barely
updates the embedding table. Switching to Adam (:numref:`sec_adam`), which
adapts step sizes per parameter, moves embedding rows substantially further
and allows the standalone RNN (with slightly stronger embedding dropout) to
edge past Kneser--Ney—an experiment left to the exercises.

## Text Generation

Once trained, a language model can generate text autoregressively from any
prompt (:numref:`sec_rnn`). The `predict` method below proceeds in two
stages: first it *warms up* the hidden state by stepping through the prefix
tokens one by one without emitting predictions; then it generates `num_tokens`
continuation tokens, converting each step's logit vector $\mathbf{o}$ into a
chosen token and feeding that token right back in as the next input.

By default (`temperature=0.0`), selection is *greedy*, picking the single
highest-scoring token at every step. Setting a positive `temperature` instead
samples from the softmax of the temperature-scaled logits, a family of
decoding methods we study in depth in :numref:`sec_decoding`.

```{.python .input #rnn-implementation-generating-text-1}
%%tab mxnet
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0):
    outputs, state = tok.encode(prefix), None
    for i in range(len(outputs) - 1):  # Warm up on the prefix
        X = d2l.tensor([[outputs[i]]], ctx=device)
        _, state = self.rnn(self.embedding(X), state)
    for _ in range(num_tokens):  # Generate num_tokens continuation tokens
        X = d2l.tensor([[outputs[-1]]], ctx=device)
        rnn_outputs, state = self.rnn(self.embedding(X), state)
        logits = self.output_layer(rnn_outputs)[0, 0]
        if temperature == 0:  # Greedy: the highest-scoring token
            token = logits.argmax()
        else:  # Sample from the softmax of logits / temperature
            token = npx.random.categorical(npx.softmax(logits / temperature))
        outputs.append(int(token))
    return tok.decode(outputs)
```

```{.python .input #rnn-implementation-generating-text-1}
%%tab pytorch
@d2l.add_to_class(RNNLMScratch)  #@save
@torch.inference_mode()  # Generation needs no autograd graph
def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0):
    outputs, state = tok.encode(prefix), None
    for i in range(len(outputs) - 1):  # Warm up on the prefix
        X = d2l.tensor([[outputs[i]]], device=device)
        _, state = self.rnn(self.embedding(X), state)
    for _ in range(num_tokens):  # Generate num_tokens continuation tokens
        X = d2l.tensor([[outputs[-1]]], device=device)
        rnn_outputs, state = self.rnn(self.embedding(X), state)
        logits = self.output_layer(rnn_outputs)[0, 0]
        if temperature == 0:  # Greedy: the highest-scoring token
            token = logits.argmax()
        else:  # Sample from the softmax of logits / temperature
            token = torch.distributions.Categorical(
                logits=logits / temperature).sample()
        outputs.append(int(token))
    return tok.decode(outputs)
```

```{.python .input #rnn-implementation-generating-text-1}
%%tab tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0):
    outputs, state = tok.encode(prefix), None
    for i in range(len(outputs) - 1):  # Warm up on the prefix
        X = d2l.tensor([[outputs[i]]])
        _, state = self.rnn(self.embedding(X), state)
    for _ in range(num_tokens):  # Generate num_tokens continuation tokens
        X = d2l.tensor([[outputs[-1]]])
        rnn_outputs, state = self.rnn(self.embedding(X), state)
        logits = self.output_layer(rnn_outputs)[0]  # Shape: (1, vocab_size)
        if temperature == 0:  # Greedy: the highest-scoring token
            token = tf.argmax(logits[0])
        else:  # Sample from the softmax of logits / temperature
            token = tf.random.categorical(logits / temperature, 1)[0, 0]
        outputs.append(int(token))
    return tok.decode(outputs)
```

```{.python .input #rnn-implementation-generating-text-1}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0):
    model = nnx.view(self, deterministic=True, use_running_average=True,
                     raise_if_not_found=False)
    outputs, state = tok.encode(prefix), None
    for i in range(len(outputs) - 1):  # Warm up on the prefix
        X = d2l.tensor([[outputs[i]]])
        _, state = model.rnn(model.embedding(X), state)
    for _ in range(num_tokens):  # Generate num_tokens continuation tokens
        X = d2l.tensor([[outputs[-1]]])
        rnn_outputs, state = model.rnn(model.embedding(X), state)
        logits = model.output_layer(rnn_outputs)[0, 0]
        if temperature == 0:  # Greedy: the highest-scoring token
            token = logits.argmax()
        else:  # Sample from the softmax of logits / temperature
            token = jax.random.categorical(d2l.get_key(), logits / temperature)
        outputs.append(int(token))
    return tok.decode(outputs)
```

Let us test greedy generation on a real English prompt; `data.tokenizer`
encodes the string on the way in and decodes the generated token ids back into
readable text.

```{.python .input #rnn-implementation-generating-text-2}
%%tab mxnet, pytorch
d2l.print_wrapped(repr(model.predict('The Time Traveller', 50,
                                     data.tokenizer, d2l.try_gpu())))
```

```{.python .input #rnn-implementation-generating-text-2}
%%tab tensorflow, jax
d2l.print_wrapped(repr(model.predict('The Time Traveller', 50,
                                     data.tokenizer)))
```

The generated passage starts out sounding plausibly like English for a few
words before stalling into repetition. Because greedy decoding is purely
deterministic, once the hidden state revisits a phrase it has just emitted, it
cycles through that same phrase indefinitely. Designing decoding algorithms
that escape such degenerate loops is the focus of :numref:`sec_decoding`.

## Concise Implementation
:label:`sec_rnn-concise`

Modern deep learning frameworks provide optimized recurrent layers that fuse
the cell computation and the time loop into fast low-level kernels. The `RNN`
wrapper below exposes the exact same interface as `RNNScratch`—accepting a
(`num_steps`, `batch_size`, `num_inputs`) tensor and optional state, and
returning all hidden states alongside the final state—while matching our
$\mathcal{N}(0, \sigma^2)$ weight initialization and zero biases so that both
implementations train the identical model.

:begin_tab:`mxnet`
Gluon's recurrent layer keeps two bias vectors, one for the input term and
one for the recurrent term. Their sum plays the role of
$\mathbf{b}_\textrm{h}$, and both start at zero. Gluon's default
activation is ReLU, so we ask for tanh.
:end_tab:

:begin_tab:`pytorch`
`nn.RNN` keeps two bias vectors, one for the input term and one for the
recurrent term. Their sum plays the role of $\mathbf{b}_\textrm{h}$, and
both start at zero.
:end_tab:

:begin_tab:`tensorflow`
`SimpleRNN` takes batch-major inputs, so `forward` transposes on the way in
and out. As in `RNNScratch`, each weight gets its own initializer object.
:end_tab:

:begin_tab:`jax`
Flax NNX provides `SimpleCell` for a plain recurrent cell. It consumes one
time step and an explicit hidden state, and `nnx.RNN` scans it over the
input sequence. The cell's input projection carries the bias, and its
recurrent projection has none, so the cell has one bias vector, as
`RNNScratch` does.
:end_tab:

```{.python .input #rnn-implementation-concise-implementation-1}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        # Gluon's default activation is ReLU; use the tanh of RNNScratch
        self.rnn = gluon.rnn.RNN(num_hiddens, activation='tanh')
        self.rnn.initialize(init.Normal(sigma))  # The biases start at zero

    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input #rnn-implementation-concise-implementation-1}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight'):
                nn.init.normal_(param, std=sigma)
            else:  # Both bias vectors start at zero
                nn.init.zeros_(param)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input #rnn-implementation-concise-implementation-1}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        normal = tf.keras.initializers.RandomNormal
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, kernel_initializer=normal(stddev=sigma),
            recurrent_initializer=normal(stddev=sigma),
            return_sequences=True, return_state=True)

    # Keras builds the layer by calling `call` with named arguments, which
    # `Module.call(X, *args)` cannot pass on, so we name the state here
    def call(self, inputs, H=None, training=None):
        return self.forward(inputs, H)

    def forward(self, inputs, H=None):
        # The keras layer is batch-major: transpose in and out
        outputs, H = self.rnn(tf.transpose(inputs, perm=[1, 0, 2]), H)
        return tf.transpose(outputs, perm=[1, 0, 2]), H
```

```{.python .input #rnn-implementation-concise-implementation-1}
%%tab jax
class RNN(nnx.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01, rngs=None):
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.sigma = sigma
        normal = nnx.initializers.normal(sigma)
        cell = nnx.SimpleCell(num_inputs, num_hiddens, kernel_init=normal,
                              recurrent_kernel_init=normal, rngs=rngs)
        self.rnn = nnx.RNN(cell, time_major=True, return_carry=True,
                           rngs=rngs)

    def __call__(self, inputs, H=None):
        H, outputs = self.rnn(inputs, initial_carry=H)
        return outputs, H
```

Because the interface matches `RNNScratch` exactly, our high-level language
model `RNNLM` simply subclasses `RNNLMScratch`, swapping out the raw tensor
embedding and output projection for each framework's built-in embedding and
linear layers while inheriting training, evaluation, and `predict` unchanged.
We initialize both layers to match their scratch counterparts: unit-variance
Gaussian embeddings, $\mathcal{N}(0, \sigma^2)$ output weights, and zero
output biases.

:begin_tab:`pytorch`
`nn.Embedding` draws its rows from $\mathcal{N}(0, 1)$ by default; the
weights of `nn.Linear` are set explicitly.
:end_tab:

```{.python .input #rnn-implementation-concise-implementation-2}
%%tab mxnet
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.emb = nn.Embedding(self.vocab_size, self.rnn.num_inputs,
                                weight_initializer=init.Normal(1))
        self.linear = nn.Dense(self.vocab_size, flatten=False,
                               weight_initializer=init.Normal(self.sigma))
        # The recurrent layer initializes itself (d2l.RNN) or is initialized
        # by its owner; here we initialize only the layers defined here
        self.emb.initialize()
        self.linear.initialize()

    def embedding(self, X):
        return self.emb(X.T)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input #rnn-implementation-concise-implementation-2}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.emb = nn.Embedding(self.vocab_size, self.rnn.num_inputs)
        self.linear = nn.Linear(self.rnn.num_hiddens, self.vocab_size)
        nn.init.normal_(self.linear.weight, std=self.sigma)
        nn.init.zeros_(self.linear.bias)

    def embedding(self, X):
        return self.emb(X.T)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input #rnn-implementation-concise-implementation-2}
%%tab tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        normal = tf.keras.initializers.RandomNormal
        self.emb = tf.keras.layers.Embedding(
            self.vocab_size, self.rnn.num_inputs,
            embeddings_initializer=normal(stddev=1))
        self.linear = tf.keras.layers.Dense(
            self.vocab_size, kernel_initializer=normal(stddev=self.sigma))

    def embedding(self, X):
        return self.emb(tf.transpose(X))

    def output_layer(self, hiddens):
        return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input #rnn-implementation-concise-implementation-2}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def __init__(self, rnn, vocab_size, lr=0.01, dropout=0, emb_dropout=0,
                 rngs=None):
        d2l.Classifier.__init__(self)
        self.save_hyperparameters(ignore=['rnn', 'rngs'])
        self.rnn = rnn
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        sigma = getattr(rnn, 'sigma', 0.01)  # Scale of the output weights
        self.emb = nnx.Embed(vocab_size, rnn.num_inputs,
                             embedding_init=nnx.initializers.normal(1.0),
                             rngs=rngs)
        self.linear = nnx.Linear(
            rnn.num_hiddens, vocab_size,
            kernel_init=nnx.initializers.normal(sigma), rngs=rngs)
        self.drop = nnx.Dropout(dropout, rngs=rngs)
        self.emb_drop = nnx.Dropout(emb_dropout, rngs=rngs)

    def embedding(self, X):
        return self.emb(X.T)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

Let us instantiate the concise model with the exact hyperparameters of our
wider scratch model and train it under the same settings.

```{.python .input #rnn-implementation-concise-implementation-3}
%%tab pytorch, mxnet, jax
rnn = RNN(num_inputs=512, num_hiddens=128)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=4, dropout=0.5,
              emb_dropout=0.1)
```

```{.python .input #rnn-implementation-concise-implementation-3}
%%tab tensorflow
with d2l.try_gpu():
    rnn = RNN(num_inputs=512, num_hiddens=128)
    model = RNNLM(rnn, vocab_size=len(data.vocab), lr=4, dropout=0.5,
                  emb_dropout=0.1)
```

Running `trainer.fit` invokes the exact same training loop as before.

```{.python .input #rnn-implementation-concise-implementation-5}
trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

Given identical initialization, data, and optimization settings, the concise
model converges to essentially the same validation perplexity as `RNNLMScratch`
and exhibits the same greedy repetition behavior.

```{.python .input #rnn-implementation-concise-implementation-6}
%%tab pytorch, mxnet
ppl_concise = math.exp(trainer.history['val_loss'][-1])
pred = model.predict('The Time Traveller', 30, data.tokenizer, d2l.try_gpu())
d2l.print_wrapped(f'perplexity {ppl_concise:.1f}, {pred!r}')
```

```{.python .input #rnn-implementation-concise-implementation-6}
%%tab tensorflow, jax
ppl_concise = math.exp(trainer.history['val_loss'][-1])
pred = model.predict('The Time Traveller', 30, data.tokenizer)
d2l.print_wrapped(f'perplexity {ppl_concise:.1f}, {pred!r}')
```

Because the framework layer executes the exact same recurrence inside a fused
kernel, we will use from-scratch cells in the rest of the book solely for
pedagogical clarity and rely on high-level layers for training.

Still, tying a trigram model falls short of what a recurrent hidden state
seemed to promise. Our position-by-position curve diagnosed the immediate
symptom—the trained network relies on only the most recent few tokens of its
32-token window—pointing toward a deeper obstacle in optimization. For a
network to learn that a token thirty steps earlier matters, the error signal
must survive backpropagation through thirty consecutive recurrent transitions.
In :numref:`sec_bptt`, we derive those gradients analytically and measure them
directly inside this trained model.

## Summary

An RNN language model applies the same three-stage step at every position—an
embedding lookup, a $\tanh$ recurrent cell, and a linear output projection—and
framework-supplied recurrent layers execute this exact computation far more
efficiently than an explicit Python loop. Stable training relies on two key
mechanisms: global gradient-norm clipping bounds any single parameter step,
preventing the erratic perplexity spikes that derail most unclipped runs at
learning rate 4, while resetting the hidden state at each window boundary
truncates backpropagation to the window length. On *The Time Machine*, a wider
embedding paired with dropout brings the SGD-trained RNN to roughly 2.1 bits
per byte—on par with an interpolated Kneser--Ney trigram, while their 50/50
mixture outperforms both. Inspecting surprisal by window position reveals why
the vanilla RNN stops there: it effectively exploits only the last few tokens
of its 32-token context, and greedy generation quickly falls into repetitive
loops.

## Exercises

1. [code] **Gradient clipping ablation.** `d2l.Trainer` applies the
   projection :eqref:`eq_grad_clipping` whenever `gradient_clip_val` is
   positive.
    1. Record the norm of the gradient before clipping at every update
       of the first epoch, and plot it. How often does clipping act, and
       when?
    1. Most unclipped runs of this section jumped at `lr=4`. Halve the
       learning rate until three unclipped runs in a row train without a
       jump in the validation perplexity. How does their validation
       perplexity after ten epochs compare with that of the clipped run at
       `lr=4`? What, then, does clipping make possible?
    1. Replace $\tanh$ by ReLU in `RNNScratch` and train again, with and
       without clipping. Is clipping still needed? Explain the difference
       from the $\tanh$ cell in terms of the range of the hidden state.

1. [code] **Usable history.** Every training window starts from a zero
   state, so the model is never trained to use more than `num_steps`
   tokens of context, and the loss by position showed that it uses far
   fewer. `predict`, by contrast, carries the state across the whole
   prefix and every generated token. Evaluate the trained model
   on 512 consecutive validation tokens in two ways: resetting the state
   to zero every 32 tokens as in training, and carrying the state across
   all 512 tokens. Compare the average cross-entropy of the two. Does the
   carried state help, hurt, or make no difference, and what does the
   answer say about the states the model reaches beyond its training
   horizon?

1. [code] **Hyperparameters and data budget.**
    1. Starting from the model with the wider embedding, adjust the
       number of epochs, hidden units, embedding dimension, `num_steps`,
       learning rate, and dropout rates, or add weight decay, to lower the
       validation perplexity. Can the RNN alone, trained with SGD, beat
       the Kneser–Ney trigram? How does the relative improvement in bits
       per byte compare with the relative improvement in perplexity, and
       why?
    1. Set `num_train=10000` and train for 100
       epochs. Compare the training and validation perplexity curves and
       explain what you see. Consecutive windows overlap by 31 tokens, so
       how many parameters does the model have per distinct training
       token in this regime?

1. [code] **Adam.** Train the model with the wider embedding with the Adam
   optimizer of :numref:`sec_adam` instead of SGD: subclass
   `RNNLMScratch`, override `configure_optimizers` to return Adam with
   learning rate 0.003, and keep clipping at 1.
    1. With `emb_dropout=0.2`, score the tokens `val[32:]` as in the
       comparison above, and compare its bits per byte with the SGD
       model, the Kneser–Ney trigram and 4-gram, and the mixture of the
       RNN and the trigram at $\lambda = 0.5$.
    1. Record the validation loss after every epoch, with
       `emb_dropout=0.1` and with 0.2. At which epoch is it lowest, and what
       happens after that?
    1. Measure again how far the embedding rows move during training. Why
       does Adam move them further than SGD does?

1. [code] **Vocabulary size and bits per byte.** ● Retrain the model with
   the wider embedding for `vocab_size` 512, 1,024, 2,048, and 4,096 in
   `d2l.TimeMachine`. The number of windows depends on the vocabulary size,
   so split by fraction: build each dataset once to read off
   `n = len(data.X)`, then build it again with `num_train=int(0.8 * n)` and
   `num_val=n - int(0.8 * n)`. (Setting `data.num_train` afterwards would
   skip the step that keeps validation tokens out of the training targets.)
   Every model then trains and validates on about the same stretch of text.
   Their per-token perplexities are not comparable, so compare them by bits
   per byte.
    1. Which vocabulary size gives the lowest bits per byte?
    1. How do the parameter counts of the embedding table and the output
       layer, and the length of the corpus in tokens, change with the
       vocabulary size? Which of these effects best explains the trend
       you measured?

1. [code] **Weight tying.** ● Weight tying sets the embedding dimension
   equal to the number of hidden units and uses $\mathbf{E}^\top$ as the
   output projection in place of $\mathbf{W}_{\textrm{hq}}$
   :cite:`Press.Wolf.2017,Inan.Khosravi.Socher.2017`.
    1. How many parameters does tying save at $|\mathcal{V}| = 1{,}027$
       with 128 hidden units? GPT-2 small ties its $50{,}257 \times 768$
       token embedding to its output layer. How many parameters would
       untying add there, and what fraction of the model's roughly 124
       million parameters is that?
    1. Tie the weights in `RNNLMScratch` with an embedding dimension of 128.
       `RNNLMScratch` initializes $\mathbf{E}$ at unit scale
       and $\mathbf{W}_{\textrm{hq}}$ at scale `sigma`. What happens to
       the initial perplexity once the two are tied, and how do you
       restore a near-uniform initial prediction?
    1. Train the tied model and an untied model with a 128-dimensional
       embedding and 128 hidden units, and compare their validation
       perplexities.

1. [code] **Cross-book generalization.** ● Subclass `d2l.TimeMachine` and
   override `_download` to return the text of a different H. G. Wells
   novel, such as *The War of the Worlds* (Project Gutenberg ebook 36).
   Train the tokenizer and the model on it.
    1. Evaluate the model on held-out text of its own novel and on *The
       Time Machine*, encoding both with the new novel's tokenizer.
       Report perplexity and bits per byte for both.
    1. Write $s = \log_2(\textrm{ppl})$ for the per-token surprisal and
       $b$ for the bytes per token of an evaluated text, so that bits per
       byte is $s / b$. The ratio of bits per byte on *The Time Machine*
       (TM) to bits per byte on the held-out text of the new novel (own)
       factors as
       $$\frac{s_\textrm{TM}}{s_\textrm{own}} \cdot \frac{b_\textrm{own}}{b_\textrm{TM}}.$$
       The first factor reflects the model's predictions and the second
       the tokenizer's compression. Measure both factors. Which lies
       further from one?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1052)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/18014)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §8.5]{.kicker}

Implementing RNNs<br>**one step for any length · embeddings · gradient clipping · bits per byte · context used · framework layers**
:::
:::

::: {.slide title="Three Equations Are Not Yet a Trained Language Model"}
[The problem]{.kicker}

- The RNN of the previous deck: embed a token, update the state, map the
  state to $|\mathcal{V}| = 1{,}027$ logits.
- To learn from the 65,741 BPE tokens of *The Time Machine*, the equations
  need code, and training needs a safeguard: an occasional gradient is
  many times longer than the typical one.
- A validation perplexity of 90: good or bad, and against which baseline?
- The state *may* hold the whole prefix. How much does the trained state
  hold?

::: {.d2l-note .warn}
What keeps one update from undoing training, and does the hidden state
beat a count table?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- implement one recurrent step and give the shape of every tensor;
- clip the gradient norm, and show what happens without clipping;
- see why SGD leaves the embeddings almost untouched, and work around it;
- compare the model with Kneser–Ney in bits per byte on the same text;
- measure how many tokens of context the trained model uses;
- generate text, and wrap a framework layer with the same initialization.

**Running example:** *The Time Machine*, BPE-1024, $h = 128$; $d = 64$,
then $d = 512$ with dropout.
:::

::: {.slide title="Training Turns the RNN Into a Language Model"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Working with Sequences | autoregression, latent state, rollout error |
| From Text to Tokens | characters, words, bytes, BPE |
| Language Models | $n$-grams, perplexity, bits per byte |
| Recurrent Neural Networks | a hidden state updated at every step |
| **Implementing RNNs** | **an RNN language model, scratch and concise** |
| Backpropagation Through Time | gradients through time, truncation |
| Decoding and Generation | greedy decoding, beam search, sampling |
:::

::: {.col .narrow}
- **Uses:** `Module`, `Trainer`, the cross-entropy loss, BPE windows with
  shifted targets.
- **Skips:** the gradient computation (next deck), decoding beyond greedy
  (last deck).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The Model from Scratch]{.dtitle}

[one step, applied to any length]{.dsub}
:::
:::

::: {.slide title="Each Step Maps a Token and a State to Logits and a New State"}
[The step]{.kicker}

$$\mathbf{H}_t = \tanh(\mathbf{X}_t \mathbf{W}_\textrm{xh} +
\mathbf{H}_{t-1} \mathbf{W}_\textrm{hh} + \mathbf{b}_\textrm{h}), \qquad
\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_\textrm{hq} + \mathbf{b}_\textrm{q}$$

| tensor | shape |
|---|---|
| token ids | $(b, T)$ = (1,024, 32) |
| embeddings $\mathbf{X}_t$, stacked | $(T, b, d)$ |
| states $\mathbf{H}_t$, stacked | $(T, b, h)$ |
| logits $\mathbf{O}_t$, stacked | $(b, T, \lvert\mathcal{V}\rvert)$ |

The state keeps its shape, so the same step runs for any $T$.
:::

::: {.slide title="The Cell Is a Loop over the Time Axis"}
[The step]{.kicker}

@-rnn-implementation-the-recurrent-cell-2

Weights start from $\mathcal{N}(0, 0.01^2)$: the untrained cell forgets
its state within a few steps, and training has to grow them.
:::

::: {.slide title="An Embedding Lookup Is a One-Hot Product Without the Zeros"}
[Embeddings]{.kicker}

- One-hot input $\mathbf{e}_i$: a 1,027-wide vector with a single 1, at
  position $i$, times a matrix.
- $\mathbf{e}_i^\top \mathbf{E}$ = row $i$ of
  $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$: index instead of
  multiply.
- The one-hot product is the same lookup, computed the expensive way.
- Rows start at unit scale, output weights at 0.01: the untrained model is
  close to uniform, perplexity close to 1,027.
:::

::: {.slide title="Check: Where Do the 222,915 Parameters Sit?"}
[Check]{.kicker}

$|\mathcal{V}| = 1{,}027$, $d = 64$, $h = 128$. Which part is largest?

. . .

| part | parameters |
|---|--:|
| $\mathbf{E}$ | 65,728 |
| $\mathbf{W}_\textrm{xh}$, $\mathbf{W}_\textrm{hh}$, $\mathbf{b}_\textrm{h}$ | 24,704 |
| $\mathbf{W}_\textrm{hq}$, $\mathbf{b}_\textrm{q}$ | 132,483 |

The recurrent layer holds 11%; the embedding table and the output layer
hold 89%.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Gradient Clipping]{.dtitle}

[bound the size of every update]{.dsub}
:::
:::

::: {.slide title="One Large Gradient Can Undo Training in a Single Update"}
[Gradient clipping]{.kicker}

- Backprop through $T$ steps multiplies $T$ Jacobians of the same
  transition: the product can grow exponentially in $T$.
- One step moves the parameters by $\eta\|\mathbf{g}\|$; a long step
  leaves the region where the loss is close to linear.
- Shrinking $\eta$ for the rare spike slows every step.

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right)
\mathbf{g}$$

$\theta = 1$, $\|\mathbf{g}\| = 10$: the update shrinks tenfold and keeps
its direction. $\|\mathbf{g}\|$ is the norm over all parameters at once.
:::

::: {.slide title="The Clip Runs Between the Backward Pass and the Update" except="jax"}
[Gradient clipping]{.kicker}

@rnn-implementation-gradient-clipping

A crude device: whenever it acts, training no longer follows the
gradient. It is standard all the same.
:::

::: {.slide title="The Clip Runs Inside the Compiled Training Step" only="jax"}
[Gradient clipping]{.kicker}

```{.python #rnn-implementation-exercises}
def clip_gradients(grad_clip_val, grads):
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree.leaves(grads)))
    return jax.tree.map(lambda g: jnp.where(
        norm < grad_clip_val, g, g * (grad_clip_val / norm)), grads)

# In Trainer.prepare_optimizer, when gradient_clip_val > 0:
clip = lambda grads, params: clip_gradients(val, grads)
tx = optax.chain(optax.stateless(clip), tx)
```

The clip is chained in front of the model's optimizer.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Training and Evaluation]{.dtitle}

[clipping, bits per byte, baselines, and the context used]{.dsub}
:::
:::

::: {.slide title="A Zero State per 32-Token Window Truncates Backpropagation"}
[Training]{.kicker}

- 50,000 training windows, 5,000 validation windows, $T = 32$, batch 1,024.
- Ten epochs of SGD at learning rate $\eta = 4$, clipping at $\theta = 1$,
  teacher forcing. A clipped update has length $\eta\theta = 4$.
- Plain model first: $d = 64$, $h = 128$, no dropout.
- Every window starts from $\mathbf{H}_0 = \mathbf{0}$: gradients reach at
  most 32 steps back (*truncated* backpropagation through time).
:::

::: {.slide title="The Training Loss Keeps Falling; the Validation Loss Stops Improving"}
[Training]{.kicker}

@!rnn-implementation-training-3

Plain model, $d = 64$: 222,915 parameters for 50,000 training tokens; the
final training perplexity is less than half the validation perplexity.
:::

::: {.slide title="Without Clipping, Most Runs Jump and End Well Above the Clipped Model"}
[Training]{.kicker}

Three fresh copies of the plain model, same learning rate 4, no clipping;
highest and final validation perplexity of each run:

@!rnn-implementation-a-plain-model-3

- With clipping: about 90 to 100, run after run.
- 32 unclipped runs per framework: more than 8 in 10 end above twice the
  clipped perplexity. Now and then a run escapes and matches it.
:::

::: {.slide title="The Clipped Curve Falls; Most Unclipped Curves Spike"}
[Training]{.kicker}

@!rnn-implementation-a-plain-model-4

- At some epoch the perplexity jumps, in some runs past the untrained
  model's 1,027, and falls back only part of the way.

::: {.d2l-note}
Clipping is what makes a learning rate of 4 usable.
:::
:::

::: {.slide title="Bits per Byte Makes Perplexities Comparable Across Tokenizers"}
[Evaluation]{.kicker}

$$\textrm{bits per byte} = \frac{\log_2 \textrm{ppl}}{\textrm{bytes per token}}$$

@!rnn-implementation-training-4

@!rnn-implementation-training-5

A character model chooses among 27 symbols, ours among 1,027 tokens: only
bits per byte compares them.
:::

::: {.slide title="Check: Perplexity 90 at 2.78 Bytes per Token"}
[Check]{.kicker}

How many bits per byte is that?

. . .

$\log_2 90 \approx 6.49$ bits per token, $6.49 / 2.78 \approx 2.34$ bits
per byte.
:::

::: {.slide title="Under SGD the Embedding Rows Barely Move"}
[Evaluation]{.kicker}

@!rnn-implementation-embeddings-under-sgd-2

- A row gets gradient only where its token occurs: a few of the 32,768
  positions of a minibatch.
- SGD's one learning rate is far too small for unit-scale rows; unseen
  tokens' rows do not move at all. The table is little more than a fixed
  random projection.
- Workaround: a wider random table ($d = 512$) for
  $\mathbf{W}_\textrm{xh}$ to read, plus dropout 0.1 on embeddings and 0.5
  on states.
:::

::: {.slide title="A Wider Embedding with Dropout Reaches About 2.1 Bits per Byte"}
[Evaluation]{.kicker}

@!rnn-implementation-a-wider-embedding-with-dropout-3

- 740,355 parameters, same SGD, learning rate, clipping and epochs.
- Validation perplexity: about 90 to 100 $\to$ about 60.
:::

::: {.slide title="The RNN Ties the Kneser–Ney Trigram, and Their Average Beats Both"}
[Evaluation]{.kicker}

Same training tokens, same validation tokens, same tokenizer:

@!rnn-implementation-training-6

@!rnn-implementation-comparison-with-kneser-ney-3

$P = \tfrac12 P_\textrm{RNN} + \tfrac12 P_\textrm{KN}$ (Mikolov et al.,
2011). Each model is the more confident one on about half of the tokens:
equally good, wrong in different places.
:::

::: {.slide title="The Trained RNN Uses About Three Tokens of Its 32-Token Window"}
[Evaluation]{.kicker}

Validation bits per byte by position in the window: the target at
position $k$ is predicted from $k$ tokens of context.

@!rnn-implementation-comparison-with-kneser-ney-4

- One token of context: about 2.3. Three: 2.12. Thirty-two: still about
  2.12. The trigram, with two: 2.09.

::: {.d2l-note .warn}
The state could carry 32 tokens; training did not teach it to. That is
why the RNN ties the trigram.
:::
:::

::: {.slide title="Greedy Decoding Tends to Loop"}
[Generation]{.kicker}

Warm up the state on the prefix, then feed each chosen token back in.
Greedy takes $\arg\max_i o_i$, the highest logit.

@!rnn-implementation-generating-text-2

Nothing in it is random: once the state returns to a phrase it has
produced, the phrase repeats.
:::

::: {.slide title="The Framework Layer Keeps the Interface and the Initialization"}
[Framework layers]{.kicker}

- `RNN` wraps the framework's recurrent layer: (`num_steps`, `batch_size`,
  `num_inputs`) in, all states and the final state out.
- Weights $\mathcal{N}(0, \sigma^2)$, biases zero, as in `RNNScratch`.
- `RNNLM` inherits training, evaluation and `predict` from `RNNLMScratch`;
  only the embedding and output layers change.

@!rnn-implementation-concise-implementation-6
:::

::: {.slide title="Check: Why Does Averaging Help?"}
[Check]{.kicker}

On one token the RNN assigns probability 0.5 and Kneser–Ney 0.02; on
another, 0.02 and 0.5. Bits for each model, and for the average?

. . .

- Each model alone: $-\log_2 0.5 - \log_2 0.02 \approx 1 + 5.6 = 6.6$ bits.
- The average assigns 0.26 to both: $2 \times 1.94 \approx 3.9$ bits.

The mixture gives each token at least half the better model's probability,
so it never costs more than one bit above the better model.
:::

::: {.slide title="An RNN Matches Kneser–Ney, and Their Average Beats Both"}

[Wrap-up]{.kicker}

- One step: token and state in, 1,027 logits and a same-shaped state out.
- Clipping to $\theta = 1$ bounds every update; without it most runs at
  learning rate 4 jump and end well above the clipped model. A zero state
  per window truncates backpropagation at 32 steps.
- Plain model about 2.3 to 2.4 bits per byte; $d = 512$ with dropout about
  2.1, level with Kneser–Ney; their average about 2.0.
- Loss by position: the trained RNN uses about three tokens of context.
- The framework layer: same interface, same initialization, same
  perplexity.

**Next:** the gradients through time, measured on this model: why they
exploded without clipping, and why the state remembers so little.
:::
