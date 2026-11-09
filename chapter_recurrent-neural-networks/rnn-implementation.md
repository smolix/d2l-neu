# Implementing RNNs
:label:`sec_rnn-scratch`

:numref:`sec_rnn` wrote down a recurrent language model and counted its
parameters, but did not train it. Two questions remain: how its equations
become working code, and whether a hidden state predicts *The Time
Machine* better than the counts of :numref:`sec_language-model`. The model fits in three equations: look up the
embedding of the current token, update the hidden state from it and from
the previous state, and map the new state to scores over the vocabulary.
The code for these equations keeps one contract. At each time step the
model consumes one token per sequence and the previous hidden state, and it
returns $|\mathcal{V}|$ logits per sequence and a new state of the same
shape as the old one. Because the shapes do not change from step to step,
the same step applies to a sequence of any length.

We first implement this contract with plain tensor operations, so that
every parameter and every shape is visible, and then hand it to the
recurrent layer that each framework provides. Training needs one addition
that the equations do not show. The gradient passes through the same
transition once per time step and can become large enough to undo much of
the progress of training in a single update. We therefore clip its norm
before every update. We then measure the trained model against the count-based
$n$-gram models of :numref:`sec_language-model` on the same text, and let
it generate text.

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
%%tab mxnet, pytorch, tensorflow
import collections
import time
```

```{.python .input #rnn-implementation-implementing-rnns-1}
%%tab jax
import collections
import random
import time
```

## An RNN Language Model from Scratch

The model has three parts, and we implement each as its own class or method:
a *recurrent cell* that updates the hidden state one time step at a time, an
embedding that turns token ids into the vectors the cell consumes, and an
output layer that maps each hidden state to a vector of logits over the
vocabulary.

### The Recurrent Cell

For a minibatch of $b$ sequences, let $\mathbf{X}_t \in \mathbb{R}^{b
\times d}$ hold their inputs at step $t$ and $\mathbf{H}_t \in
\mathbb{R}^{b \times h}$ their hidden states. The cell computes the
recurrence of :numref:`sec_rnn` with the tanh activation,

$$\mathbf{H}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),$$

so each new state mixes the current input with the previous state through
two weight matrices, $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times
h}$ and $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$, and a
bias $\mathbf{b}_\textrm{h} \in \mathbb{R}^h$. The `RNNScratch` class
holds these three parameters. The number of hidden units `num_hiddens` ($h$)
is a hyperparameter, and `num_inputs` ($d$) is the width of the vectors
arriving at each step, which will shortly be the embedding dimension.

As in :numref:`sec_linear_scratch`, we register the weights with the framework's
parameter type, so that the optimizer finds them and the trainer can move
them to a GPU. Both weight matrices start from $\mathcal{N}(0, \sigma^2)$
with $\sigma = 0.01$, and the bias starts at zero. With weights this
small, the recurrent weight has spectral radius about $\sigma\sqrt{h}
\approx 0.11$, so the untrained cell forgets its state within a few steps,
and training grows the weights from there.

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

The forward computation loops over the outermost axis of `inputs`, whose
shape is (`num_steps`, `batch_size`, `num_inputs`), applies the recurrence
once per time step, and collects the hidden state of every step. When no
initial state is supplied, the state starts at zero. The explicit Python
loop over time steps is the readable form of the recurrence.

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

We feed the cell a minibatch of two sequences of 100 steps each.

```{.python .input #rnn-implementation-the-recurrent-cell-3}
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

The contract asks for one state per step, each with the shape
(`batch_size`, `num_hiddens`) of the state the cell started from. The two
helpers below check lengths and shapes; later sections use them as well.

```{.python .input #rnn-implementation-the-recurrent-cell-4}
def check_len(a, n):  #@save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

### The Embedding Table

The cell consumes vectors, but our data pipeline delivers integer token ids.
Older treatments bridged the gap with a *one-hot encoding*: token $i$
becomes the length-$|\mathcal{V}|$ vector with a single 1 in position $i$,
and the input weights multiply that vector. With a vocabulary of 27
characters this is harmless. With our BPE vocabulary of $|\mathcal{V}| =
1{,}027$ ids it is wasteful, since each step multiplies a 1,027-wide vector
that is zero everywhere but one entry.

As in :numref:`sec_rnn`, we instead use an *embedding lookup*: a trainable
table $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ represents
token $i$ by its $i^\textrm{th}$ row, a $d$-dimensional vector that
gradient descent adjusts along with every other parameter. Multiplying a
one-hot vector by $\mathbf{E}$ selects a row of $\mathbf{E}$, so the lookup
computes the one-hot matrix product without materializing the zeros. A
one-hot encoding is thus an embedding whose table is fixed at the identity.
A trainable table lets the model choose the vector of each token.

The `RNNLMScratch` class assembles the language model around a cell `rnn`:
the embedding table `E` on the input side and an output layer (`W_hq`,
`b_q`) on the output side. The embedding dimension is read off the cell's
`num_inputs`, since the cell consumes what the embedding produces. The
embedding rows start at unit scale, as the 1s of a one-hot encoding would,
while the output weights start at scale $\sigma$ and the output bias at
zero. The logits of the untrained model are then close to zero, its
next-token distribution is close to uniform, and its perplexity is close to
$|\mathcal{V}|$.

The model inherits from `Classifier` the cross-entropy
loss, averaged over all tokens of a minibatch. Its validation step returns
only this loss, since the token accuracy that `Classifier` would add is not
what we monitor. The mean cross-entropy is the logarithm of the perplexity
(:numref:`subsec_perplexity`), so the validation perplexity is the
exponential of the trainer's mean validation loss.

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

The minibatches sampled by our data pipeline have shape (`batch_size`,
`num_steps`). The `embedding` method looks up each id and transposes the
result to (`num_steps`, `batch_size`, `num_inputs`), the layout whose
outermost axis the cell's forward loop walks over.

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

To confirm the equivalence claimed above, we check on a toy five-token
vocabulary that multiplying one-hot vectors into a table picks out the same
rows that indexing does.

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-3}
%%tab mxnet
W, ids = d2l.randn(5, 3), d2l.tensor([0, 2], dtype=d2l.int64)
(d2l.matmul(npx.one_hot(ids, 5), W) == W[ids]).all()
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-3}
%%tab pytorch
W, ids = d2l.randn(5, 3), d2l.tensor([0, 2])
torch.allclose(F.one_hot(ids, 5).float() @ W, W[ids])
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-3}
%%tab tensorflow
W, ids = d2l.normal((5, 3)), d2l.tensor([0, 2])
tf.reduce_all(tf.one_hot(ids, 5) @ W == tf.gather(W, ids))
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-3}
%%tab jax
W, ids = jax.random.normal(d2l.get_key(), (5, 3)), d2l.tensor([0, 2])
jnp.allclose(jax.nn.one_hot(ids, 5) @ W, W[ids])
```

### The Output Layer

The last of the three equations turns each hidden state into scores for
the next token. The output layer computes $\mathbf{O}_t = \mathbf{H}_t
\mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}$ for every time step, with
$\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times |\mathcal{V}|}$. Since
the same weights serve every step, one matrix product handles all of them:
the method stacks the states of all steps into a (`batch_size`,
`num_steps`, `num_hiddens`) tensor and multiplies it by
$\mathbf{W}_{\textrm{hq}}$, which gives logits of shape (`batch_size`,
`num_steps`, `vocab_size`) that line up with the target array. The
`forward` method chains the three parts together.

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

A minibatch of token ids of shape (`batch_size`, `num_steps`) should come
out as one vector of logits per position. We check this with a vocabulary
of 1,024 ids, about the size of the one we train on.

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

A recurrent network is deep in time. Backpropagating through $T$ time
steps multiplies $T$ Jacobians of the same transition, one per step, and
as in a deep feedforward network such a product can shrink or grow
exponentially with its length. :numref:`sec_bptt` analyzes both cases.
Vanishing gradients limit what the model can learn, but they do not break
training. Exploding gradients can: when the product grows, an occasional
gradient is many times longer than the typical one. We do not measure the
gradient norm in this section. The exercises ask you to record it during
the first epoch and to train without clipping.

To see what a large gradient threatens, consider the length of one update.
A gradient step with learning rate $\eta$ moves the parameters
$\mathbf{x}$ to $\mathbf{x} - \eta\mathbf{g}$, a distance

$$\|(\mathbf{x} - \eta\mathbf{g}) - \mathbf{x}\| = \eta\|\mathbf{g}\|.$$

Thus the length of a step grows with the norm of the gradient. Gradient
descent works because, over a short step, the loss is close to its linear
approximation $f(\mathbf{x}) - \eta\|\mathbf{g}\|^2$, which decreases.
When $\|\mathbf{g}\|$ spikes, the step leaves the region where that
approximation holds, and the loss can jump instead of falling. One such
step can undo the progress of many updates. Shrinking $\eta$ until even the
rare spike is safe would slow every step to protect against a few.
*Gradient clipping* instead caps the norm on the occasions it is too large,
projecting $\mathbf{g}$ onto the ball of radius $\theta$:

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$
:eqlabel:`eq_grad_clipping`

A gradient with norm at most $\theta$ passes unchanged; a longer one is
scaled down to norm $\theta$ and keeps its direction. With $\theta = 1$, a
gradient of norm 10 becomes one tenth of itself. The norm
$\|\mathbf{g}\|$ is the Euclidean norm of the gradients of all parameters
of the model, stacked into one vector. Whenever it acts, clipping changes
the update, so training no longer follows the gradient, and its effect on
convergence is hard to analyze. It is standard nevertheless: recurrent
networks and large transformers alike are trained with it.

:begin_tab:`mxnet, pytorch`
The method below attaches to the `d2l.Trainer` class (see
:numref:`sec_linear_scratch`): whenever a positive `gradient_clip_val` is
set, `fit_epoch` invokes it after the backward pass and before the parameter
update, rescaling each parameter's gradient in place. The scale factor
$\min(1, \theta/\|\mathbf{g}\|)$ is computed on the device, so clipping
does not force a synchronization with Python.
:end_tab:

:begin_tab:`tensorflow`
The method below attaches to the `d2l.Trainer` class (see
:numref:`sec_linear_scratch`): whenever a positive `gradient_clip_val` is
set, the compiled training step invokes it between taping the gradients and
applying them, and the returned (possibly rescaled) gradients are handed to
the optimizer.
:end_tab:

:begin_tab:`jax`
The function `clip_gradients` below returns the rescaled gradients. The JAX
trainer of :numref:`sec_linear_scratch` updates the parameters with an optax
transformation inside its compiled training step, so we apply the clipping
there too: whenever a positive `gradient_clip_val` is set,
`prepare_optimizer` wraps `clip_gradients` with `optax.stateless`, which
turns a function of the gradients into an optax transformation, and chains
it in front of the model's optimizer. The scale factor is computed on the
device. The clipping threshold joins the key of `optimizer_cache`, so that a
later `fit` with the same model and threshold reuses the compiled step.
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

The `d2l.TimeMachine` pipeline of :numref:`sec_language-model` downloads the
corpus, trains the BPE tokenizer on it, and serves minibatches of token-id
windows together with their shift-by-one targets. The corpus is 65,741 BPE
tokens long, and training window $i$ starts at token $i$. The pipeline
drops the 32 windows that straddle the split, so that no validation token
is ever a training target. The default split reserves only the first
10,000 windows for training, which suits quick
demonstrations but gives a model too little text (an exercise asks you to
watch what goes wrong). We therefore train on the first 50,000 windows,
most of the novella, and validate on the following 5,000.

```{.python .input #rnn-implementation-training-1}
data = d2l.TimeMachine(batch_size=1024, num_steps=32,
                       num_train=50000, num_val=5000)
```

Every window is $T = 32$ tokens long, and every forward pass starts from a
zero state, so gradients flow backward through at most 32 steps: the chain
of dependencies is *truncated* at the window boundary. The model learns
longer-range structure only insofar as it helps within a window. This
compromise is called truncated backpropagation through time, and
:numref:`sec_bptt` analyzes what it gives up. A common variant for long
sequences carries the final state of one chunk into the next instead of
resetting it. The carried state must then be detached from the computation
graph, so that gradients stop at the chunk boundary; otherwise the graph,
and the cost of each backward pass, would grow with every chunk.

### A Plain Model

We first train the model as built so far: an embedding of dimension $d =
64$ and a cell with $h = 128$ hidden units, 222,915 parameters in all, for
ten epochs of minibatch SGD at learning rate 4, clipping gradients to norm
$\theta = 1$, with no dropout. As in :numref:`sec_rnn`, training is
teacher-forced: every step conditions on the true prefix from the corpus.
We keep a copy of the embedding table, to see later how much training
changes it.

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

Before training, the validation perplexity is close to the vocabulary size
of 1,027, as expected from the near-uniform initial model. The board plots
the mean cross-entropy per token, the logarithm of the perplexity. Both
curves fall fast during the first epochs. Then they separate: the training
loss keeps falling, while the validation loss stops improving over the last
few epochs. After ten epochs the training perplexity is less than half the
validation perplexity: with 222,915 parameters for 50,000 training tokens,
more than four parameters per token, the model has started to memorize its
training text. The final validation perplexity is the exponential of the
last validation loss.

```{.python .input #rnn-implementation-training-4}
ppl_plain = math.exp(trainer.history['val_loss'][-1])
print(f'validation perplexity {ppl_plain:.1f}')
```

A perplexity of about 90 may look poor next to the perplexity of about 6
of the character-level trigram in :numref:`sec_language-model`. The two numbers
cannot be compared: our model chooses among 1,027 BPE tokens per step, a
character model among 27 characters. Bits per byte
(:numref:`subsec_bits-per-byte`, :eqref:`eq_bpb`) removes the dependence on
the tokenizer: it divides the total surprisal of a text by its length in
bytes, which here comes to $\log_2 \textrm{ppl}$ divided by the average
number of bytes per token. We measure the bytes per token on the 5,000
consecutive validation tokens that start the validation windows.

```{.python .input #rnn-implementation-training-5}
ids = d2l.numpy(data.X[data.num_train:data.num_train+data.num_val, 0]).tolist()
bytes_per_token = len(data.tokenizer.decode(ids).encode('utf-8')) / len(ids)
print(f'{bytes_per_token:.2f} bytes/token, '
      f'{math.log2(ppl_plain) / bytes_per_token:.2f} bits per byte')
```

Each token covers about 2.8 bytes of text, and the plain model needs about
2.3 to 2.4 bits per byte.

### Embeddings under SGD

Overfitting is one problem of the plain model. A second one is hidden in
the embedding table. We measure how far each row of $\mathbf{E}$ moved
during training, relative to its initial length.

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

The rows barely move. Each row receives a gradient only from the positions
where its token occurs, and the loss averages over all $1{,}024 \times 32 =
32{,}768$ positions of a minibatch, so a token that occurs a few times
contributes a few parts in 32,768 of the gradient. A step of SGD moves
every parameter by the learning rate times its gradient, the same rate for
all, which is far too small for rows that start at unit scale. The cell
therefore reads its inputs through token vectors that stay close to their
random initial values; rows of tokens that never occur in the training
text do not move at all.

The input weights $\mathbf{W}_{\textrm{xh}}$ do learn, so a wider
embedding helps even if its rows stay close to random: with $d = 512$
instead of 64, $\mathbf{W}_{\textrm{xh}}$ has more random features of each
token to combine. Dropout (:numref:`sec_dropout`), applied to the
embeddings at rate 0.1 and to the hidden states before the output layer at
rate 0.5, counters the overfitting. An optimizer that
adapts the step to each parameter's gradients moves the rows much further,
as an exercise shows with Adam.

### A Wider Embedding with Dropout

`RNNLMScratch` already accepts the two dropout rates, which default to
zero. The new model has 740,355 parameters, most of them in the
$1{,}027 \times 512$ embedding table; everything else in the training
procedure stays the same. We time this run for a comparison at the end of
the section.

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
t0 = time.perf_counter()
trainer.fit(model, data)
t_scratch = time.perf_counter() - t0
```

```{.python .input #rnn-implementation-a-wider-embedding-with-dropout-3}
ppl_scratch = math.exp(trainer.history['val_loss'][-1])
print(f'validation perplexity {ppl_scratch:.1f}, '
      f'{math.log2(ppl_scratch) / bytes_per_token:.2f} bits per byte')
```

The validation perplexity falls to about 60, against about 90 for the
plain model; in bits per byte, about 2.1 against 2.3 to 2.4. The training loss on the board is
measured with dropout active, which raises it, so it no longer shows the
gap between training and validation directly. The validation loss is still
at or near its lowest after ten epochs.

### Comparison with Kneser–Ney

The best count-based model of :numref:`sec_language-model` is Kneser–Ney
smoothing in its interpolated form (:numref:`subsec_kneser-ney`,
:eqref:`eq_kneser_ney`): it subtracts a discount from every observed
$n$-gram count and passes the freed probability to lower orders that count
distinct predecessors. `d2l.kneser_ney` counts the $n$-grams of a training
sequence and returns the natural log-probability of every token of another
sequence, given the tokens before it. We count the $n$-grams of the RNN's
50,000 training tokens, in the same BPE tokenization. We score the
validation tokens from the 33rd on, `val[32:]`, so that the RNN, below, can
read a full window of 32 validation tokens before each of them.

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

The trigram is much better than the bigram, and a fourth token of context
helps it little more. To compare the RNN token by token, we score the same
tokens `val[32:]` with it, each from the 32 validation tokens before it:
validation window $j$ starts at `val[j]`, so its last target is
`val[j + 32]`. (The validation loss on the board averages over all 32
positions of a window, and the first of them see only one or a few tokens
of context.)

```{.python .input #rnn-implementation-comparison-with-kneser-ney-1}
start, stop = data.num_train, data.num_train + data.num_val - 32
targets = [int(y) for y in d2l.numpy(data.Y[start:stop, -1])]  # val[32:]
```

```{.python .input #rnn-implementation-comparison-with-kneser-ney-2}
%%tab mxnet
X_val = data.X[start:stop].as_in_ctx(d2l.try_gpu())
logp = d2l.numpy(mnp.concatenate([  # Log-probabilities at the last position
    npx.log_softmax(model(X_val[i:i + 1024])[:, -1])
    for i in range(0, len(X_val), 1024)]))
```

```{.python .input #rnn-implementation-comparison-with-kneser-ney-2}
%%tab pytorch
model.eval()
with torch.inference_mode():  # Log-probabilities at the last position
    logp = d2l.numpy(torch.cat([
        F.log_softmax(model(X.to(d2l.try_gpu()))[:, -1], -1)
        for X in data.X[start:stop].split(1024)]))
```

```{.python .input #rnn-implementation-comparison-with-kneser-ney-2}
%%tab tensorflow
X_val = data.X[start:stop]
logp = d2l.numpy(tf.concat([  # Log-probabilities at the last position
    tf.nn.log_softmax(model(X_val[i:i + 1024], training=False)[:, -1])
    for i in range(0, len(X_val), 1024)], 0))
```

```{.python .input #rnn-implementation-comparison-with-kneser-ney-2}
%%tab jax
model_eval = nnx.view(model, deterministic=True, raise_if_not_found=False)
X_val = data.X[start:stop]
logp = d2l.numpy(jnp.concatenate([  # Log-probabilities at the last position
    jax.nn.log_softmax(model_eval(X_val[i:i + 1024])[:, -1])
    for i in range(0, len(X_val), 1024)]))
```

The two models make different mistakes. The RNN sees 32 tokens of context
but learns only from what it has seen in a few hundred SGD steps. The
counts see two tokens but remember every trigram of the training text.
Linear interpolation, $\lambda P_\textrm{RNN} + (1 - \lambda)
P_\textrm{KN}$, is how :citet:`Mikolov.Kombrink.Burget.ea.2011` combined
their RNN language models with Kneser–Ney models, choosing $\lambda$ on
held-out data. We fix $\lambda = 0.5$ instead of tuning it, close to the
best weight for this pair. A weighted average of two probability
distributions is a probability distribution, so the mixture is a language
model too.

```{.python .input #rnn-implementation-comparison-with-kneser-ney-3}
lp_rnn = [logp[i, y] for i, y in enumerate(targets)]
lam = 0.5
lp_mix = [math.log(lam * math.exp(a) + (1 - lam) * math.exp(b))
          for a, b in zip(lp_rnn, lp_kn)]
for name, lp in [('RNN', lp_rnn), ('KN trigram', lp_kn), ('mixture', lp_mix)]:
    bits = -sum(lp) / len(lp) / math.log(2)
    print(f'{name}: {bits / bytes_per_token:.2f} bits per byte')
```

The RNN alone comes within a few hundredths of a bit per byte of the
Kneser–Ney trigram, and the mixture beats both by a clear margin, because
the two models are confident on different tokens. Trained with Adam
(:numref:`sec_adam`) instead of SGD and with slightly stronger embedding
dropout, the RNN alone also edges past the trigram; an exercise asks you to
try it. These numbers are higher than those in the table of
:numref:`subsec_bits-per-byte`, which was computed on lowercased text with
letters only. Bits per byte compares tokenizers on the same text, and on
the raw text of this section the Kneser–Ney trigram needs about 2.1.

## Text Generation

A trained language model can continue a prefix (:numref:`sec_rnn`). The
`predict` method below first *warms up* on the prefix, feeding its tokens
one at a time to build up a hidden state without emitting anything. It then
generates `num_tokens` new tokens, each time turning the current logits
$\mathbf{o}$ into a choice of next token and feeding that token back in as
the next input.

The method offers the two simplest ways to make the choice. With
`temperature=0` it is *greedy* and takes the highest-scoring token.
Otherwise it *samples* token $i$ with probability proportional to
$\exp(o_i / \tau)$ for the temperature $\tau > 0$. At $\tau = 1$ this is
the model's own softmax distribution. Values below 1 sharpen the
distribution toward the greedy choice, and values above 1 flatten it toward
the uniform distribution. Each framework's sampler draws a fresh random
token on each call, so the samples differ from run to run.

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

We start with greedy decoding. The prefix is real text, not token ids: the
tokenizer trained by the pipeline (`data.tokenizer`) encodes it on the way
in and decodes the generated ids on the way out.

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

The continuation has the look of English, with real words, punctuation,
and line breaks, but its grammar holds over a few words at most. Greedy
decoding is deterministic, and it often falls into a loop: once the state
reaches a high-probability phrase, the phrase comes back again and again.
Sampling does not loop, but at $\tau = 1$ it also picks unlikely tokens,
which come out as invented words. At $\tau = 0.5$ the samples move back
toward the greedy choice.

```{.python .input #rnn-implementation-generating-text-3}
%%tab mxnet, pytorch
for tau in (1.0, 0.5):
    d2l.print_wrapped(repr(model.predict('The Time Traveller', 30,
                                         data.tokenizer, d2l.try_gpu(),
                                         temperature=tau)))
```

```{.python .input #rnn-implementation-generating-text-3}
%%tab tensorflow, jax
for tau in (1.0, 0.5):
    d2l.print_wrapped(repr(model.predict('The Time Traveller', 30,
                                         data.tokenizer, temperature=tau)))
```

Neither setting removes both defects: repetition and incoherence depend on
how we choose tokens as well as on the model. :numref:`sec_decoding` treats
strategies that choose better.

## Concise Implementation
:label:`sec_rnn-concise`

Frameworks ship recurrent layers that bundle the cell and the loop over
time steps into one optimized call. The `RNN` class below wraps the
framework layer and keeps the contract of `RNNScratch`: it consumes a
(`num_steps`, `batch_size`, `num_inputs`) tensor and an optional state, and
returns the hidden states of all steps together with the final state. So
that the two implementations train the same model, `RNN` also initializes
the layer as `RNNScratch` does, with weights from $\mathcal{N}(0,
\sigma^2)$ and zero biases.

:begin_tab:`mxnet`
Gluon's recurrent layer keeps two bias vectors, one added to the input term
and one to the recurrent term, as cuDNN's kernel expects; their sum plays
the role of $\mathbf{b}_\textrm{h}$. It stores all its weights and biases
in one fused parameter vector. Initializing the layer with
`init.Normal(sigma)` draws the weights from $\mathcal{N}(0, \sigma^2)$ and
sets both biases to zero, so at initialization the layer computes the same
function as `RNNScratch`. Gluon's default activation is ReLU, so we ask for
tanh.
:end_tab:

:begin_tab:`pytorch`
`nn.RNN` keeps two bias vectors, `bias_ih_l0` for the input term and
`bias_hh_l0` for the recurrent term, as cuDNN's kernel expects; their sum
plays the role of $\mathbf{b}_\textrm{h}$. Both start at zero, so at
initialization the layer computes the same function as `RNNScratch`. The
layer stores its weights transposed, as (`num_hiddens`, `num_inputs`) and
(`num_hiddens`, `num_hiddens`) matrices.
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

Because the contract is unchanged, the language model carries over by
inheritance: `RNNLM` subclasses `RNNLMScratch` and swaps the hand-rolled
embedding table and output layer for the framework's embedding and dense
layers. Training, evaluation, and `predict` are all inherited. The new
layers are initialized like their scratch counterparts, with unit-normal
embedding rows, output weights from $\mathcal{N}(0, \sigma^2)$, and a zero
output bias.

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

Before training, we let the untrained model continue a prefix. The
continuation consists of random tokens, with replacement characters where
the generated ids decode to bytes that do not form valid UTF-8. It still
checks the wiring from the tokenizer through the model and back.

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

```{.python .input #rnn-implementation-concise-implementation-4}
model.predict('it has', 20, data.tokenizer)
```

Training is the same call as before, with the same hyperparameters.

```{.python .input #rnn-implementation-concise-implementation-5}
trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
t0 = time.perf_counter()
trainer.fit(model, data)
t_concise = time.perf_counter() - t0
```

With the same initialization, data, and training, the concise model
reaches about the same validation perplexity as the scratch model, and its
samples read the same.

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

### Scratch and Concise Implementations Compared

Both training runs were timed, so we can put the two implementations side
by side. Each time covers the whole call to `fit`: ten epochs of training,
validation after every epoch, and the loading of every minibatch.

```{.python .input #rnn-implementation-scratch-versus-concise-measured}
print(f'{"model":>8} {"time (s)":>9} {"val ppl":>8}')
for name, t, p in [('scratch', t_scratch, ppl_scratch),
                   ('concise', t_concise, ppl_concise)]:
    print(f'{name:>8} {t:>9.1f} {p:>8.1f}')
```

:begin_tab:`mxnet`
The framework layer is faster. `gluon.rnn.RNN` runs the whole recurrence in
one fused cuDNN kernel, while our scratch loop launches several small GPU
operations per time step from Python, and at this model size the cost of
each launch outweighs the arithmetic it starts. Both times also include the
same data loading and validation, so the recurrences alone differ by more
than the totals do.
:end_tab:

:begin_tab:`pytorch`
The framework layer is two to three times faster. `nn.RNN` runs the whole
recurrence in one fused cuDNN kernel, while our scratch loop launches
several small GPU operations per time step from Python, and at this model
size the cost of each launch outweighs the arithmetic it starts. Both times
also include the same data loading and validation, so the recurrences alone
differ by more than the totals do.
:end_tab:

:begin_tab:`tensorflow`
Here the framework layer brings no fused kernel: Keras has none for
`SimpleRNN` (only its LSTM and GRU layers have one), so both versions run
as `tf.function`-compiled graphs of per-step operations, neither has a
structural advantage, and which one finishes first varies from run to run.
Both times also include the same data loading and validation.
:end_tab:

:begin_tab:`jax`
Both versions are compiled by XLA into one program per training step, so
neither launches operations from Python at each time step. Our scratch cell
is unrolled into one copy per time step, while `nnx.RNN` scans a single
compiled copy of the cell over the steps. The difference in time comes
mostly from compiling the unrolled loop, not from the arithmetic, and both
times also include the same data loading and validation.
:end_tab:

At this model size the difference is seconds to tens of seconds. The
overhead of the scratch loop arises once per time step, so it grows with
the length of the sequences and the number of stacked layers. The rest of
the book therefore writes recurrent cells from scratch only to explain them,
and trains with the framework layers.

The model we have built reads a window of 32 tokens and keeps a 128-number
state. With a wider embedding and dropout, it predicts the next token about
as well as a Kneser–Ney trigram, and averaged with that trigram it beats
both. A plain RNN still has trouble carrying information across many
steps, because its gradients pass through the same transition at every
step. :numref:`sec_bptt` computes these gradients and shows when they
vanish or explode.

## Summary

An RNN language model keeps one contract at every time step: it consumes a
token and the previous state and returns logits and a new state of the same
shape. We implemented it from tensor operations, with an embedding lookup in
place of one-hot inputs, a tanh cell, and an output layer shared by all
steps, and then with each framework's recurrent layer, initialized and
trained the same way. Gradients through many steps can explode, so every
update clips the gradient norm, which bounds the change any one update
makes. Each training window starts from a zero state, which truncates
backpropagation at the window length. Under SGD the embedding rows barely
move, and the plain model overfits, reaching 2.3 to 2.4 bits per byte. A wider
embedding with dropout lowers the validation loss to about 2.1 bits per
byte, within a few hundredths of a Kneser–Ney trigram on the same text, and
the average of the two models beats both. Greedy generation falls into
loops, and temperature sampling replaces them with invented words.

## Exercises

1. [code] **Gradient clipping ablation.** `d2l.Trainer` applies the
   projection :eqref:`eq_grad_clipping` whenever `gradient_clip_val` is
   positive.
    1. Record the norm of the gradient before clipping at every update
       of the first epoch, and plot it. How often does clipping act, and
       when?
    1. Train `RNNLMScratch` three times with `gradient_clip_val=0` at
       `lr=4`. How many runs diverge or end far above the validation
       perplexity of the clipped run? Then halve the learning rate until
       three unclipped runs in a row train without a blow-up. How does
       their validation perplexity after ten epochs compare with that of
       the clipped run at `lr=4`?
    1. Replace $\tanh$ by ReLU in `RNNScratch` and train again, with and
       without clipping. Is clipping still needed? Explain the difference
       from the $\tanh$ cell in terms of the range of the hidden state.

1. [code] **Usable history.** Every training window starts from a zero
   state, so the model is never trained to use more than `num_steps`
   tokens of context. `predict`, by contrast, carries the state across
   the whole prefix and every generated token. Evaluate the trained model
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
    1. Restore the pipeline default `num_train=10000` and train for 100
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

1. [code] **Scratch versus concise timing.** `t_scratch` and `t_concise`
   time ten epochs of `RNNLMScratch` and `RNNLM` with 128 hidden units,
   32-token windows, and a batch size of 1,024.
    1. Vary `num_hiddens` over 32, 128, and 512 and `num_steps` over 16,
       32, and 128, and tabulate the ratio `t_scratch / t_concise` for
       each setting. In which direction does the ratio move with each
       hyperparameter?
    1. Repeat the default setting with a batch size of 64. Explain the
       changes you measure: which of per-step launch overhead, arithmetic
       per step, and compilation dominates in each regime?

1. [code] **Vocabulary size and bits per byte.** ● Rebuild the dataset with
   `vocab_size` set to 512, 1,024, 2,048, and 4,096 in `d2l.TimeMachine` and
   retrain with the same architecture and training procedure. The number of
   windows `len(data.X)` depends on the vocabulary size, so replace the
   fixed split of 50,000 training and 5,000 validation windows by a
   fractional one. Build each dataset once to read off the number of windows
   `n = len(data.X)`, then build it again, passing `num_train=int(0.8 * n)`
   and `num_val=n - int(0.8 * n)` to the constructor; setting
   `data.num_train` afterwards would skip the step that keeps validation
   tokens out of the training targets. Every model then trains and validates
   on about the same stretch of text. The per-token perplexities of these
   models are not comparable, so compare them by bits per byte.
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

Implementing RNNs<br>**the step contract · embeddings · gradient clipping · bits per byte · framework layers**
:::
:::

::: {.slide title="Three Equations Are Not Yet a Trained Language Model"}
[The problem]{.kicker}

- The RNN of the previous deck: embed a token, update the state, map the
  state to $|\mathcal{V}| = 1{,}027$ logits.
- To learn from the 65,741 BPE tokens of *The Time Machine*, the equations
  need code with a fixed contract.
- An occasional gradient is many times longer than the typical one.
- A validation perplexity of 90: good or bad, and against which baseline?

::: {.d2l-note .warn}
What keeps one update from undoing training, and what should the result be
compared with?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- implement one recurrent step and give the shape of every tensor;
- clip the gradient norm and say what clipping bounds;
- see why SGD leaves the embeddings almost untouched, and fix the model;
- compare it with Kneser–Ney in bits per byte on the same text;
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
  and temperature.
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
[The contract]{.kicker}

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
[The contract]{.kicker}

@-rnn-implementation-the-recurrent-cell-2
:::

::: {.slide title="An Embedding Lookup Is a One-Hot Product Without the Zeros"}
[Embeddings]{.kicker}

- One-hot input $\mathbf{e}_i$: a 1,027-wide vector with a single 1, at
  position $i$, times a matrix.
- $\mathbf{e}_i^\top \mathbf{E}$ = row $i$ of
  $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$: index instead of
  multiply.
- One-hot = an embedding whose table is fixed at the identity.

@rnn-implementation-from-token-ids-to-embeddings-3
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

The recurrent layer holds 11%; the parts that face the vocabulary hold 89%.
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

The scale factor stays on the device: no `if` on the host.
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

[bits per byte, baselines, and generated text]{.dsub}
:::
:::

::: {.slide title="A Zero State per 32-Token Window Truncates Backpropagation"}
[Training]{.kicker}

- 50,000 training windows, 5,000 validation windows, $T = 32$, batch 1,024.
- Ten epochs of SGD at learning rate 4, $\theta = 1$, teacher forcing.
- Plain model first: $d = 64$, $h = 128$, no dropout.
- Every window starts from $\mathbf{H}_0 = \mathbf{0}$: gradients reach at
  most 32 steps back (truncated backpropagation through time).
- Carrying the state across chunks instead requires detaching it from the
  graph.
:::

::: {.slide title="The Training Loss Keeps Falling; the Validation Loss Stops Improving"}
[Training]{.kicker}

@!rnn-implementation-training-3

Plain model, $d = 64$: 222,915 parameters for 50,000 training tokens; the
final training perplexity is less than half the validation perplexity.
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
  tokens' rows do not move at all.
- Fix: a wider random table ($d = 512$) for $\mathbf{W}_\textrm{xh}$ to
  read, plus dropout 0.1 on embeddings and 0.5 on states.
:::

::: {.slide title="A Wider Embedding with Dropout Reaches About 2.1 Bits per Byte"}
[Evaluation]{.kicker}

@!rnn-implementation-a-wider-embedding-with-dropout-3

- 740,355 parameters, same SGD, learning rate, clipping and epochs.
- Validation perplexity about 90 $\to$ about 60.
- Adam instead of SGD moves the rows much further and does better still
  (exercise).
:::

::: {.slide title="Averaging the RNN with Kneser–Ney Beats Both"}
[Evaluation]{.kicker}

Same training tokens, same validation tokens, same tokenizer:

@!rnn-implementation-training-6

@!rnn-implementation-comparison-with-kneser-ney-3

$P = \tfrac12 P_\textrm{RNN} + \tfrac12 P_\textrm{KN}$: the two models
are confident on different tokens (Mikolov et al., 2011).
:::

::: {.slide title="Greedy Decoding Loops; Temperature Sampling Invents Words"}
[Generation]{.kicker}

Feed each chosen token back in. Greedy takes $\arg\max_i o_i$; sampling
draws token $i$ with probability $\propto \exp(o_i / \tau)$.

@!rnn-implementation-generating-text-2

@!rnn-implementation-generating-text-3
:::

::: {.slide title="The Framework Layer Keeps the Contract and the Initialization"}
[Framework layers]{.kicker}

- `RNN` wraps the framework's recurrent layer: (`num_steps`, `batch_size`,
  `num_inputs`) in, all states and the final state out.
- Weights $\mathcal{N}(0, \sigma^2)$, biases zero, as in `RNNScratch`.
- `RNNLM` inherits training, evaluation and `predict` from `RNNLMScratch`;
  only the embedding and output layers change.

@!rnn-implementation-concise-implementation-6
:::

::: {.slide title="Same Model, Same Perplexity, Less Time" only="mxnet"}
[Framework layers]{.kicker}

@!rnn-implementation-scratch-versus-concise-measured

- `gluon.rnn.RNN` runs the whole recurrence in one fused cuDNN kernel.
- The scratch loop launches several small operations per time step; at
  this size launch overhead outweighs the arithmetic.
- Both times include the same data loading and validation.
:::

::: {.slide title="Same Model, Same Perplexity, Faster Training" only="pytorch"}
[Framework layers]{.kicker}

@!rnn-implementation-scratch-versus-concise-measured

- `nn.RNN` runs the whole recurrence in one fused cuDNN kernel.
- The scratch loop launches several small operations per time step; at
  this size launch overhead outweighs the arithmetic.
- Both times include the same data loading and validation.
:::

::: {.slide title="Same Model, Same Perplexity, No Fused Kernel" only="tensorflow"}
[Framework layers]{.kicker}

@!rnn-implementation-scratch-versus-concise-measured

- `SimpleRNN` has no fused GPU kernel: both versions are compiled graphs
  of per-step operations.
- Neither has a structural speed advantage; which one finishes first varies
  from run to run.
:::

::: {.slide title="Same Model, Same Perplexity, a Smaller Margin in Time" only="jax"}
[Framework layers]{.kicker}

@!rnn-implementation-scratch-versus-concise-measured

- Both versions compile into one XLA program per training step.
- The scratch cell is unrolled per time step; `nnx.RNN` scans one copy.
  The gap comes from compiling the unrolled loop, not from arithmetic.
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
- Clipping to $\theta = 1$ bounds every update; a zero state per window
  truncates backpropagation at 32 steps.
- Plain model about 2.3 to 2.4 bits per byte; $d = 512$ with dropout about
  2.1, level with Kneser–Ney; their average about 2.0.
- Greedy decoding loops; temperature sampling invents words.
- The framework layer: same contract, same initialization, same perplexity.

**Next:** the gradients through time, and when they vanish or explode.
:::
