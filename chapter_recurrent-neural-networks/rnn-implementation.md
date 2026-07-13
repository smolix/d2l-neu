# Implementing RNN Language Models
:label:`sec_rnn-scratch`

We are now ready to build the RNN language model of :numref:`sec_rnn` end to
end: first from raw tensor operations, so that every moving part is visible,
and then again with the recurrent layer that every deep learning framework
ships. We train both on *The Time Machine*, tokenized into the 1,024-token
byte-pair-encoding (BPE) vocabulary of :numref:`sec_text-sequence` and served
as minibatches of shifted input and target windows by the pipeline of
:numref:`sec_language-model`. Along the way we meet gradient clipping, the
standard defense against exploding gradients, and we generate our first text.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input #rnn-implementation-implementing-rnn-language-models}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
import random
import time
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #rnn-implementation-implementing-rnn-language-models}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #rnn-implementation-implementing-rnn-language-models}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import random
import time
import tensorflow as tf
```

```{.python .input #rnn-implementation-implementing-rnn-language-models}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import random
import time
```

## An RNN Language Model from Scratch

The model has three parts, and we implement each as its own class or method:
a *recurrent cell* that updates the hidden state one time step at a time, an
*embedding* that turns token ids into the vectors the cell consumes, and an
*output layer* that maps each hidden state to a vector of logits over the
vocabulary.

### The Recurrent Cell

The `RNNScratch` class holds the three parameters of the recurrence
:eqref:`rnn_h_with_state`: the input-to-hidden weights
$\mathbf{W}_{\textrm{xh}}$, the hidden-to-hidden weights
$\mathbf{W}_{\textrm{hh}}$, and the bias $\mathbf{b}_\textrm{h}$. The number
of hidden units `num_hiddens` is a tunable hyperparameter, and `num_inputs`
is the width of the vectors arriving at each step, which will shortly be the
embedding dimension.

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
%%tab mxnet
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
        self.W_hh = d2l.randn(
            num_hiddens, num_hiddens) * sigma
        self.b_h = d2l.zeros(num_hiddens)
```

```{.python .input #rnn-implementation-the-recurrent-cell-1}
%%tab tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = tf.Variable(d2l.normal(
            (num_inputs, num_hiddens)) * sigma)
        self.W_hh = tf.Variable(d2l.normal(
            (num_hiddens, num_hiddens)) * sigma)
        self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input #rnn-implementation-the-recurrent-cell-1}
%%tab jax
class RNNScratch(nnx.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.sigma = sigma
        self.W_xh = nnx.Param(
            rngs.params.normal((num_inputs, num_hiddens)) * sigma)
        self.W_hh = nnx.Param(
            rngs.params.normal((num_hiddens, num_hiddens)) * sigma)
        self.b_h = nnx.Param(jnp.zeros(num_hiddens))
```

The forward computation loops over the outermost axis of `inputs`, whose
shape is (`num_steps`, `batch_size`, `num_inputs`), applying the recurrence
:eqref:`rnn_h_with_state` once per time step and collecting the hidden state
at every step. When no initial state is supplied we start from zeros. The
explicit Python loop, kept in every tab including JAX, is the deliberately
readable teaching form; the gated cells of :numref:`chap_modern_rnn` swap the
JAX loop for `jax.lax.scan`, trading that clarity for acceptable JIT
compilation time.

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
%%tab mxnet
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
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

We can feed a minibatch of input sequences into the cell as follows.

```{.python .input #rnn-implementation-the-recurrent-cell-3}
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

Let's check that the cell produces results of the correct shapes, so that
the dimensionality of the hidden state indeed remains unchanged from step
to step.

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

### From Token IDs to Embeddings

The cell consumes vectors, but our data pipeline delivers integer token ids.
Older treatments bridged the gap with a *one-hot encoding*: token $i$ becomes
the length-$|\mathcal{V}|$ vector with a single 1 in position $i$, and the
input weights multiply that vector. With a vocabulary of 27 characters this
is harmless. With our BPE vocabulary of over a thousand tokens it is
wasteful: each step would multiply a 1,024-wide vector that is zero
everywhere but one entry, and the "input" would carry no learnable
information about the token itself.

As introduced in :numref:`sec_rnn`, we instead use an *embedding lookup*.
We store a trainable matrix $\mathbf{W}_\textrm{e} \in
\mathbb{R}^{|\mathcal{V}| \times d}$ and represent token $i$ by its $i$-th
row, a dense $d$-dimensional vector that gradient descent shapes along with
every other parameter. Note that multiplying a one-hot vector by
$\mathbf{W}_\textrm{e}$ *selects a row* of $\mathbf{W}_\textrm{e}$, so an
embedding lookup computes exactly a one-hot matrix product, just without
materializing the zeros. One-hot encoding is thus the special case of an
embedding whose table is frozen at the identity; the embedding generalizes
it by letting the model choose what each token looks like.

The `RNNLMScratch` class assembles the full language model around a cell
`rnn`: the embedding table `W_e` on the input side and an output projection
(`W_hq`, `b_q`) on the output side. The embedding dimension is read off the
cell's `num_inputs`, since the cell consumes what the embedding produces. We
initialize the embedding rows at unit scale, taking over the role the one-hot
1s used to play, while the output projection starts near zero, so the initial
next-token distribution is close to uniform, with perplexity near
$|\mathcal{V}|$ before the first update. Since we evaluate with perplexity
(:numref:`subsec_perplexity`), the training and validation steps plot it in
place of raw loss.

:begin_tab:`tensorflow`
The compiled training and validation steps (see :numref:`sec_linear_scratch`)
return the raw loss, so we override the `_report_train` and `_report_val`
hooks that the trainer calls with those values, plotting perplexity instead
of loss.
:end_tab:

:begin_tab:`jax`
The training and validation steps run inside `@nnx.jit` and only return the
mean loss: plotting from inside them would hand the board a JAX tracer.
`Trainer.fit_epoch` plots the materialized loss eagerly outside jit, and we
override `plot` to relabel that value as perplexity for parity with the
other tabs.
:end_tab:

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-1}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_e = nn.Parameter(
            d2l.randn(self.vocab_size, self.rnn.num_inputs))
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-1}
%%tab mxnet
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_e = d2l.randn(self.vocab_size, self.rnn.num_inputs)
        self.W_hq = d2l.randn(
            self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
        self.b_q = d2l.zeros(self.vocab_size)
        for param in self.get_scratch_params():
            param.attach_grad()

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-1}
%%tab tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_e = tf.Variable(d2l.normal(
            (self.vocab_size, self.rnn.num_inputs)))
        self.W_hq = tf.Variable(d2l.normal(
            (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
        self.b_q = tf.Variable(d2l.zeros(self.vocab_size))

    def _report_train(self, loss):
        self.plot('ppl', d2l.exp(loss), train=True)

    def _report_val(self, y_hat, batch):
        self.plot('ppl', d2l.exp(self.loss(y_hat, batch[-1])), train=False)
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-1}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rnn', 'rngs'])
        self.rnn = rnn
        rngs = nnx.Rngs(1) if rngs is None else rngs
        self.W_e = nnx.Param(
            rngs.params.normal((vocab_size, rnn.num_inputs)))
        self.W_hq = nnx.Param(rngs.params.normal(
            (rnn.num_hiddens, vocab_size)) * rnn.sigma)
        self.b_q = nnx.Param(jnp.zeros(vocab_size))

    def training_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])

    def plot(self, key, value, train):
        # The train/val steps run inside `@nnx.jit` and only return the mean
        # loss: plotting a tracer from there would crash the board's drawing
        # thread. `Trainer.fit_epoch` instead calls this with the materialized
        # loss (outside jit), which we relabel as perplexity for parity with
        # the other tabs.
        if key == 'loss':
            key, value = 'ppl', d2l.exp(value)
        super().plot(key, value, train)
```

The minibatches sampled by our data pipeline have shape (`batch_size`,
`num_steps`). The `embedding` method looks up each id and transposes the
result to (`num_steps`, `batch_size`, `num_inputs`), the layout whose
outermost axis the cell's forward loop walks over.

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-2}
%%tab pytorch
@d2l.add_to_class(RNNLMScratch)  #@save
def embedding(self, X):
    # Output shape: (num_steps, batch_size, num_inputs)
    return self.W_e[X.T]
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-2}
%%tab mxnet
@d2l.add_to_class(RNNLMScratch)  #@save
def embedding(self, X):
    # Output shape: (num_steps, batch_size, num_inputs)
    return self.W_e[X.T]
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-2}
%%tab tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def embedding(self, X):
    # Output shape: (num_steps, batch_size, num_inputs)
    return tf.gather(self.W_e, tf.transpose(X))
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-2}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def embedding(self, X):
    # Output shape: (num_steps, batch_size, num_inputs)
    return self.W_e[X.T]
```

To confirm the equivalence claimed above, we check on a toy five-token
vocabulary that multiplying one-hot vectors into a table picks out the same
rows that indexing does.

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-3}
%%tab pytorch
W, ids = d2l.randn(5, 3), d2l.tensor([0, 2])
torch.allclose(F.one_hot(ids, 5).float() @ W, W[ids])
```

```{.python .input #rnn-implementation-from-token-ids-to-embeddings-3}
%%tab mxnet
W, ids = d2l.randn(5, 3), d2l.tensor([0, 2], dtype=d2l.int64)
(d2l.matmul(npx.one_hot(ids, 5), W) == W[ids]).all()
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

The language model applies a fully connected output layer to the hidden
state at every time step, producing one vector of $|\mathcal{V}|$ logits per
step, and stacks them into a (`batch_size`, `num_steps`, `vocab_size`)
tensor that lines up with the target array. The `forward` method chains the
three parts together.

```{.python .input #rnn-implementation-the-output-layer-1}
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.embedding(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Let's check that the forward computation produces outputs with the correct
shape, using a vocabulary of 1,024 tokens like the one we are about to train
on.

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

While you are already used to thinking of neural networks as "deep" in the
sense that many layers separate input from output within a single time step,
the length of the sequence introduces a new notion of depth. Backpropagating
through $T$ time steps produces a chain of matrix products of length
$\mathcal{O}(T)$, and as :numref:`sec_numerical_stability` warned, such
products can make gradients explode or vanish depending on the weight
matrices. We analyze this phenomenon carefully in :numref:`sec_bptt`;
vanishing gradients motivate the architectures of :numref:`chap_modern_rnn`.
Exploding gradients, however, have a blunt and ubiquitous remedy that we
need right now, because even this small model will diverge without it.

To see what a large gradient threatens, suppose the objective $f$ is
$L$-Lipschitz, so that $|f(\mathbf{x}) - f(\mathbf{y})| \leq
L \|\mathbf{x} - \mathbf{y}\|$ for any $\mathbf{x}$ and $\mathbf{y}$. A
gradient step with learning rate $\eta$ then changes the objective by at most

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

If $\|\mathbf{g}\|$ spikes, a single update can be violent enough to undo
thousands of steps of progress, and training diverges or proceeds through
massive loss spikes. Shrinking $\eta$ until even the rare spike is safe would
slow every step to protect against a few. *Gradient clipping* instead caps
the norm on the occasions it is too large, projecting $\mathbf{g}$ onto a
ball of radius $\theta$:

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

The clipped gradient never exceeds norm $\theta$ and keeps its original
direction. It also limits the influence any single minibatch can exert on the
parameters, a crude form of robustness. To be clear, clipping means we do not
always follow the true gradient, and reasoning analytically about its side
effects is hard. It is a hack, but a nearly universal one: virtually every
recurrent (and, at scale, transformer) training loop runs with some form of
gradient clipping. Below we compute the norm over all model parameters
concatenated into one giant vector.

:begin_tab:`pytorch, mxnet`
The method below attaches to the `d2l.Trainer` class (see
:numref:`sec_linear_scratch`): whenever a positive `gradient_clip_val` is
set, `fit_epoch` invokes it after the backward pass and before the parameter
update, rescaling each parameter's gradient in place.
:end_tab:

:begin_tab:`tensorflow`
The method below attaches to the `d2l.Trainer` class (see
:numref:`sec_linear_scratch`): whenever a positive `gradient_clip_val` is
set, the compiled training step invokes it between taping the gradients and
applying them, and the returned (possibly rescaled) gradients are handed to
the optimizer.
:end_tab:

:begin_tab:`jax`
In the JAX trainer, clipping is not a separate method call: `fit` chains
`optax.clip_by_global_norm(gradient_clip_val)` in front of the optimizer, so
every update is clipped inside the jitted training step. The method below
implements the same projection explicitly. The trainer never calls it; we
show it so you can see, in the same style as the other tabs, exactly what
the `optax` transformation computes.
:end_tab:

```{.python .input #rnn-implementation-gradient-clipping}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input #rnn-implementation-gradient-clipping}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
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
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## Training

The `d2l.TimeMachine` pipeline of :numref:`sec_language-model` downloads the
corpus, trains the BPE tokenizer on it, and serves minibatches of token-id
windows together with their shift-by-one targets. The corpus is about 66,000
BPE tokens long; its default split reserves only the first 10,000 windows
for training, which suits quick demonstrations but starves a real model
(an exercise asks you to watch what goes wrong). We therefore train on
50,000 windows, most of the novella, and validate on the following 5,000.

```{.python .input #rnn-implementation-training-1}
data = d2l.TimeMachine(batch_size=1024, num_steps=32,
                       num_train=50000, num_val=5000)
```

One subtlety of this pipeline deserves a name before we train on it. Every
window is `num_steps` tokens long and every forward pass starts from a fresh
zero state, so gradients flow backward through at most 32 steps: the
dependency chain is *truncated* at the window boundary. The model can still
learn longer-range structure only insofar as it helps within a window. This
is the standard practical compromise, called truncated backpropagation
through time, and :numref:`sec_bptt` analyzes what the truncation costs. A
common variant for very long sequences carries the final state across
consecutive minibatch chunks instead of resetting it; in that case the
carried state must be *detached* from the autograd graph (for instance with
`state.detach()` in PyTorch or `jax.lax.stop_gradient` in JAX), since
otherwise the graph, and the cost of the backward pass, would grow with
every chunk processed.

Now we train the model, a cell with 128 hidden units under an embedding of
dimension 64, for ten epochs, clipping gradients to norm 1. Recall from
:numref:`sec_rnn` that training is teacher-forced: every step conditions on
the true prefix from the corpus. We time the run for a comparison to come.

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

```{.python .input #rnn-implementation-training-3}
trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
model.board.yscale = 'log'  # perplexity spans orders of magnitude
t0 = time.time()
trainer.fit(model, data)
t_scratch = time.time() - t0
```

Since perplexity ranges over orders of magnitude during training, we plot it
on a logarithmic axis. The curve starts near the vocabulary size, exactly
the near-uniform initial model we constructed, and falls fast; at this
aggressive learning rate the first few updates can transiently overshoot
(clipping keeps the overshoot from becoming divergence, as an exercise
invites you to verify). Let's read off the final validation perplexity.

```{.python .input #rnn-implementation-training-4}
%%tab pytorch, mxnet, tensorflow
ppl_scratch = float(model.board.data['val_ppl'][-1].y)
print(f'validation perplexity {ppl_scratch:.1f}')
```

```{.python .input #rnn-implementation-training-4}
%%tab jax
total_loss = num_tokens = 0
for X_val, y_val in data.val_dataloader():
    losses = model.loss(model(X_val), y_val, averaged=False)
    total_loss += float(losses.sum())
    num_tokens += losses.size
ppl_scratch = math.exp(total_loss / num_tokens)
print(f'validation perplexity {ppl_scratch:.1f}')
```

A perplexity in the tens-to-hundreds may look alarming next to the
single-digit perplexities of the character-level models earlier in this
chapter. Both numbers are correct, and the comparison is meaningless: our model
chooses among 1,024 BPE tokens per step, a character model among 27
characters, and per-token perplexity cannot compare models with different
tokenizers. This is precisely why :numref:`sec_language-model` introduced
bits per byte. Each of our tokens covers several bytes of text, so we
convert: dividing the per-token surprisal $\log_2(\textrm{ppl})$ by the
average number of bytes per token gives the price in bits of each byte of
validation text.

```{.python .input #rnn-implementation-training-5}
ids = d2l.numpy(data.X[data.num_train:data.num_train+data.num_val, 0]).tolist()
bytes_per_token = len(data.tokenizer.decode(ids).encode('utf-8')) / len(ids)
print(f'{bytes_per_token:.2f} bytes/token, '
      f'{math.log2(ppl_scratch) / bytes_per_token:.2f} bits per byte')
```

Back in :numref:`sec_language-model`, the character-tokenized trigram scored
2.68 bits per byte on this same text. Our BPE-level model, despite a
per-token perplexity more than ten times larger, comes in below that,
compressing the same text *better*. The lesson carries beyond this
comparison: whenever two language models tokenize differently, compare bits
per byte, never perplexity.

## Generating Text

Once trained, the language model can do what :numref:`sec_rnn` promised:
continue a prefix. The `predict` method below first *warms up* on the
user-supplied prefix, feeding its tokens one at a time to build up a hidden
state without emitting anything. It then generates `num_tokens` new tokens,
each time turning the current logits into a choice of next token and feeding
that choice back in as the next input.

How should logits become a choice? The method offers the two simplest
strategies. With `temperature=0` it is *greedy*: always take the
highest-scoring token. Otherwise it *samples* from the model's next-token
distribution after scaling the logits by $1/T$ for temperature $T$: values
below 1 sharpen the distribution toward greedy behavior, values above 1
flatten it toward uniform randomness.

```{.python .input #rnn-implementation-generating-text-1}
%%tab pytorch
@d2l.add_to_class(RNNLMScratch)  #@save
@torch.no_grad()  # inference only: no autograd graph needed
def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0,
            rng=None):
    outputs, state = tok.encode(prefix), None
    for i in range(len(outputs) - 1):  # Warm up on the prefix
        X = d2l.tensor([[outputs[i]]], device=device)
        _, state = self.rnn(self.embedding(X), state)
    rng = random.Random() if rng is None else rng
    for _ in range(num_tokens):  # Generate num_tokens continuation tokens
        X = d2l.tensor([[outputs[-1]]], device=device)
        rnn_outputs, state = self.rnn(self.embedding(X), state)
        logits = d2l.numpy(self.output_layer(rnn_outputs))[0, 0]
        if temperature == 0:
            outputs.append(int(logits.argmax()))
        else:
            weights = [math.exp(l) for l in
                       (logits - logits.max()) / temperature]
            outputs.append(rng.choices(range(len(weights)), weights)[0])
    return tok.decode(outputs)
```

```{.python .input #rnn-implementation-generating-text-1}
%%tab mxnet
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0,
            rng=None):
    outputs, state = tok.encode(prefix), None
    for i in range(len(outputs) - 1):  # Warm up on the prefix
        X = d2l.tensor([[outputs[i]]], ctx=device)
        _, state = self.rnn(self.embedding(X), state)
    rng = random.Random() if rng is None else rng
    for _ in range(num_tokens):  # Generate num_tokens continuation tokens
        X = d2l.tensor([[outputs[-1]]], ctx=device)
        rnn_outputs, state = self.rnn(self.embedding(X), state)
        logits = d2l.numpy(self.output_layer(rnn_outputs))[0, 0]
        if temperature == 0:
            outputs.append(int(logits.argmax()))
        else:
            weights = [math.exp(l) for l in
                       (logits - logits.max()) / temperature]
            outputs.append(rng.choices(range(len(weights)), weights)[0])
    return tok.decode(outputs)
```

```{.python .input #rnn-implementation-generating-text-1}
%%tab tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0,
            rng=None):
    outputs, state = tok.encode(prefix), None
    for i in range(len(outputs) - 1):  # Warm up on the prefix
        X = d2l.tensor([[outputs[i]]])
        _, state = self.rnn(self.embedding(X), state)
    rng = random.Random() if rng is None else rng
    for _ in range(num_tokens):  # Generate num_tokens continuation tokens
        X = d2l.tensor([[outputs[-1]]])
        rnn_outputs, state = self.rnn(self.embedding(X), state)
        logits = d2l.numpy(self.output_layer(rnn_outputs))[0, 0]
        if temperature == 0:
            outputs.append(int(logits.argmax()))
        else:
            weights = [math.exp(l) for l in
                       (logits - logits.max()) / temperature]
            outputs.append(rng.choices(range(len(weights)), weights)[0])
    return tok.decode(outputs)
```

```{.python .input #rnn-implementation-generating-text-1}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0,
            rng=None):
    model = nnx.view(self, deterministic=True, use_running_average=True,
                     raise_if_not_found=False)
    outputs, state = tok.encode(prefix), None
    for i in range(len(outputs) - 1):  # Warm up on the prefix
        X = d2l.tensor([[outputs[i]]])
        _, state = model.rnn(model.embedding(X), state)
    rng = random.Random() if rng is None else rng
    for _ in range(num_tokens):  # Generate num_tokens continuation tokens
        X = d2l.tensor([[outputs[-1]]])
        rnn_outputs, state = model.rnn(model.embedding(X), state)
        logits = d2l.numpy(model.output_layer(rnn_outputs))[0, 0]
        if temperature == 0:
            outputs.append(int(logits.argmax()))
        else:
            weights = [math.exp(l) for l in
                       (logits - logits.max()) / temperature]
            outputs.append(rng.choices(range(len(weights)), weights)[0])
    return tok.decode(outputs)
```

Greedy decoding first. Note that the prefix is real text, not token ids: the
tokenizer trained by the pipeline (`data.tokenizer`) encodes it on the way
in and decodes the generated ids on the way out.

```{.python .input #rnn-implementation-generating-text-2}
%%tab mxnet, pytorch
model.predict('the time traveller', 50, data.tokenizer, d2l.try_gpu())
```

```{.python .input #rnn-implementation-generating-text-2}
%%tab tensorflow, jax
model.predict('the time traveller', 50, data.tokenizer)
```

The continuation is locally plausible Wells, but it does not stay
interesting for long. Greedy decoding is deterministic, and it tends to fall
into repetition: once the hidden state drifts into a high-probability
neighborhood, the same phrases recur again and again. Sampling with
temperature breaks such loops at the price of occasional strange choices,
and lowering the temperature interpolates back toward the greedy behavior.

```{.python .input #rnn-implementation-generating-text-3}
%%tab mxnet, pytorch
for T in (1.0, 0.5):
    print(model.predict('the time traveller', 30, data.tokenizer,
                        d2l.try_gpu(), temperature=T, rng=random.Random(0)))
```

```{.python .input #rnn-implementation-generating-text-3}
%%tab tensorflow, jax
for T in (1.0, 0.5):
    print(model.predict('the time traveller', 30, data.tokenizer,
                        temperature=T, rng=random.Random(0)))
```

Neither knob fixes the underlying problem: repetition, incoherence, and the
trade-off between them are properties of *how we choose tokens*, not just of
the model. Choosing well is a rich topic with a modern toolkit of its own,
and :numref:`sec_decoding` is devoted to it. For now we have what we need: a
trained model and a way to watch it write.

## Concise Implementation
:label:`sec_rnn-concise`

Implementing the RNN from scratch was instructive, but frameworks ship
recurrent layers that bundle the cell and the time loop into one optimized
call. The `RNN` class below wraps the framework layer while keeping exactly
the interface of `RNNScratch`: it consumes a (`num_steps`, `batch_size`,
`num_inputs`) tensor and an optional state, and returns per-step hidden
states plus the final state.

:begin_tab:`jax`
Flax NNX provides `SimpleCell` for a vanilla recurrent cell. Unlike a layer
that processes an entire sequence at once, the cell consumes one time step
and an explicit hidden state; `nnx.RNN` scans it over the input sequence.
:end_tab:

```{.python .input #rnn-implementation-concise-implementation-1}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input #rnn-implementation-concise-implementation-1}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = gluon.rnn.RNN(num_hiddens)

    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input #rnn-implementation-concise-implementation-1}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True)

    def forward(self, inputs, H=None):
        # The keras layer is batch-major: transpose in and out
        outputs, H = self.rnn(tf.transpose(inputs, perm=[1, 0, 2]), H)
        return tf.transpose(outputs, perm=[1, 0, 2]), H
```

```{.python .input #rnn-implementation-concise-implementation-1}
%%tab jax
class RNN(nnx.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.rnn = nnx.RNN(
            nnx.SimpleCell(num_inputs, num_hiddens, rngs=rngs),
            time_major=True, return_carry=True, rngs=rngs)

    def __call__(self, inputs, H=None):
        H, outputs = self.rnn(inputs, initial_carry=H)
        return outputs, H
```

Because the interface is unchanged, the language model wrapper carries over
by inheritance: `RNNLM` subclasses `RNNLMScratch` and merely swaps the
hand-rolled embedding table and output projection for the framework's
embedding and dense layers. Training, evaluation, and `predict` are all
inherited. One detail deserves attention: several frameworks initialize
embedding tables at the small scale appropriate for dense-layer weights,
which starves the recurrence of input signal on this task, so where needed
we request the unit-normal initialization that embeddings conventionally
use (and that our scratch model, and PyTorch's `nn.Embedding`, already
default to).

```{.python .input #rnn-implementation-concise-implementation-2}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.emb = nn.Embedding(self.vocab_size, self.rnn.num_inputs)
        self.linear = nn.LazyLinear(self.vocab_size)

    def embedding(self, X):
        return self.emb(X.T)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input #rnn-implementation-concise-implementation-2}
%%tab mxnet
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.emb = nn.Embedding(self.vocab_size, self.rnn.num_inputs,
                                weight_initializer=init.Normal(1))
        self.linear = nn.Dense(self.vocab_size, flatten=False)
        self.initialize()

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
        self.emb = tf.keras.layers.Embedding(
            self.vocab_size, self.rnn.num_inputs,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=1))
        self.linear = tf.keras.layers.Dense(self.vocab_size)

    def embedding(self, X):
        return self.emb(tf.transpose(X))

    def output_layer(self, hiddens):
        return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input #rnn-implementation-concise-implementation-2}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def __init__(self, rnn, vocab_size, lr=0.01, rngs=None):
        d2l.Classifier.__init__(self)
        self.save_hyperparameters(ignore=['rnn', 'rngs'])
        self.rnn = rnn
        rngs = nnx.Rngs(2) if rngs is None else rngs
        self.emb = nnx.Embed(vocab_size, rnn.num_inputs,
                             embedding_init=nnx.initializers.normal(1.0),
                             rngs=rngs)
        self.linear = nnx.Linear(rnn.num_hiddens, vocab_size, rngs=rngs)

    def embedding(self, X):
        return self.emb(X.T)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

Before training, let's make a prediction with the randomly initialized
model. Since the network has learned nothing, the continuation is byte soup;
the replacement characters appear where generated ids decode to bytes that
do not form valid UTF-8. It is still a useful check that the wiring, from
tokenizer through model and back, is sound.

```{.python .input #rnn-implementation-concise-implementation-3}
%%tab pytorch, mxnet, jax
rnn = RNN(num_inputs=64, num_hiddens=128)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=4)
```

```{.python .input #rnn-implementation-concise-implementation-3}
%%tab tensorflow
with d2l.try_gpu():
    rnn = RNN(num_inputs=64, num_hiddens=128)
    model = RNNLM(rnn, vocab_size=len(data.vocab), lr=4)
```

```{.python .input #rnn-implementation-concise-implementation-4}
model.predict('it has', 20, data.tokenizer)
```

Training is the same call as before, with the same hyperparameters.

```{.python .input #rnn-implementation-concise-implementation-5}
trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1, num_gpus=1)
model.board.yscale = 'log'
t0 = time.time()
trainer.fit(model, data)
t_concise = time.time() - t0
```

The trained model reaches a validation perplexity comparable to the
from-scratch implementation, and its samples read the same.

```{.python .input #rnn-implementation-concise-implementation-6}
%%tab pytorch, mxnet
ppl_concise = float(model.board.data['val_ppl'][-1].y)
pred = model.predict('the time traveller', 30, data.tokenizer, d2l.try_gpu())
print(f'perplexity {ppl_concise:.1f}, {pred!r}')
```

```{.python .input #rnn-implementation-concise-implementation-6}
%%tab tensorflow
ppl_concise = float(model.board.data['val_ppl'][-1].y)
pred = model.predict('the time traveller', 30, data.tokenizer)
print(f'perplexity {ppl_concise:.1f}, {pred!r}')
```

```{.python .input #rnn-implementation-concise-implementation-6}
%%tab jax
total_loss = num_tokens = 0
for X_val, y_val in data.val_dataloader():
    losses = model.loss(model(X_val), y_val, averaged=False)
    total_loss += float(losses.sum())
    num_tokens += losses.size
ppl_concise = math.exp(total_loss / num_tokens)
pred = model.predict('the time traveller', 30, data.tokenizer)
print(f'perplexity {ppl_concise:.1f}, {pred!r}')
```

### Scratch versus Concise, Measured

Since we timed both training runs, we can put the comparison in a small
table rather than assert it.

```{.python .input #rnn-implementation-scratch-versus-concise-measured}
print(f'{"model":>8} {"time (s)":>9} {"val ppl":>8}')
for name, t, p in [('scratch', t_scratch, ppl_scratch),
                   ('concise', t_concise, ppl_concise)]:
    print(f'{name:>8} {t:>9.1f} {p:>8.1f}')
```

:begin_tab:`pytorch`
The framework layer wins severalfold. The reason is kernel fusion: our
scratch loop launches several small GPU operations per time step from
Python, and at this model size the per-launch overhead dwarfs the
arithmetic, while `nn.RNN` executes the whole unrolled recurrence inside one
fused library kernel.
:end_tab:

:begin_tab:`mxnet`
The framework layer wins by a comfortable margin. The reason is kernel
fusion: our scratch loop launches several small GPU operations per time step
from Python, and at this model size the per-launch overhead dwarfs the
arithmetic, while `rnn.RNN` executes the whole unrolled recurrence inside
one fused library kernel.
:end_tab:

:begin_tab:`tensorflow`
The margin here is smaller than in the PyTorch tab, and it moves around
from run to run. The reason is that Keras provides no fused GPU kernel for
the vanilla `SimpleRNN` (only its LSTM and GRU layers have one), so both
versions execute as `tf.function`-compiled graphs of per-step operations,
and neither enjoys a fused advantage. Fused kernels win when they exist;
the LSTM and GRU layers of :numref:`chap_modern_rnn` show the speedup this
layer is missing.
:end_tab:

:begin_tab:`jax`
The gap is modest here because both versions are JIT-compiled end to end:
our scratch loop is unrolled and fused by XLA just like the library cell,
and the remaining difference is mostly compilation and dispatch overhead.
The comparison is starker in eager frameworks, where the scratch loop pays a
Python-level launch cost at every time step.
:end_tab:

At this model size the absolute difference is seconds. At the scale of
:numref:`chap_modern_rnn`'s gated cells, with more parameters per step,
deeper stacks, and longer sequences, the same per-step overhead multiplies,
and the fused layer becomes the only sensible choice. We will keep the
scratch idiom for exposition and the framework layer for every model we
actually train at length.

## Summary

We implemented an RNN language model twice. The from-scratch version makes
the anatomy explicit: an embedding lookup replaces the wasteful one-hot
encoding (it is the same linear map, with trainable rows), a recurrent cell
carries a fixed-size state across time steps, and a shared output layer
produces next-token logits at every step. Training with gradient clipping
bounds the damage any one exploding gradient can do, though it does nothing
for vanishing gradients. Because each minibatch window starts from a fresh
state, backpropagation is truncated at the window length, a compromise that
:numref:`sec_bptt` examines. Perplexities measured over different vocabularies
are incomparable, so we converted to bits per byte to see that our BPE-level
model genuinely improves on a character-level one. Finally, the trained model
generates text: greedy decoding is fluent but circles, temperature sampling
trades repetition for noise, and doing better is the subject of
:numref:`sec_decoding`. The concise implementation collapses all of this into
framework layers with the same interface and, where fused kernels exist,
meaningfully better speed.

## Exercises

1. Does the trained language model ever condition on tokens further back
   than the start of its current window? Which hyperparameter bounds the
   length of the history it can use?
1. Adjust the hyperparameters (number of epochs, hidden units, embedding
   dimension, window length `num_steps`, learning rate) to improve the
   validation perplexity. How low can you go with this architecture? Does
   bits per byte improve by the same factor?
1. Run the training in this section without gradient clipping. What happens?
1. Replace $\tanh$ with ReLU as the activation of the recurrent cell and
   repeat the experiment. Do you still need gradient clipping? Why?
1. Restore the pipeline defaults (`num_train=10000`) and train for 100
   epochs. Compare the training and validation perplexity curves and explain
   what you see. Roughly how many parameters does the model have per
   training token in this regime?
1. Rebuild the dataset with a 2,048-token tokenizer
   (`d2l.TimeMachine(..., vocab_size=2048)`) and retrain. Compare the result
   against this section's model using bits per byte. Why would comparing
   validation perplexities mislead you here?
1. Set the embedding dimension equal to the number of hidden units and *tie
   the weights*: use $\mathbf{W}_\textrm{e}^\top$ as the output projection
   instead of a separate $\mathbf{W}_{\textrm{hq}}$. How many parameters
   does this save? How does it affect perplexity?
1. Show that sampling at temperature $T$ draws token $x$ with probability
   proportional to $P(x)^{1/T}$, where $P$ is the model's next-token
   distribution. What do the limits $T \to 0$ and $T \to \infty$ recover?
1. Train on a different H. G. Wells novel (for example, *The War of the
   Worlds*) and evaluate on *The Time Machine*. Report both perplexity and
   bits per byte. What does the gap relative to in-book validation tell you?

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

::: {.slide title="Implementing RNN Language Models"}
An RNN language model on *The Time Machine*, over the 1,024-token
BPE vocabulary, built twice: from raw tensor ops, then with the
framework's recurrent layer. Four pieces:

1. **RNN cell**: the recurrence
   $\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t +
   \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b})$.
2. **Embedding**: token ids become trainable vectors
   (no more one-hot).
3. **Output head**: hidden state to vocab logits at every step.
4. **Gradient clipping + training + generation.**
:::

::: {.slide title="The RNN cell"}
Parameters: $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{b}$.
Initialize randomly, scaled to keep activations sensible:

@rnn-implementation-implementing-rnn-language-models

@rnn-implementation-the-recurrent-cell-1
:::

::: {.slide title="Forward, unrolled"}
Walk a length-$T$ input one step at a time, carrying the hidden
state forward:

@rnn-implementation-the-recurrent-cell-2

. . .

@rnn-implementation-the-recurrent-cell-3

. . .

Sanity check on output shapes:

@rnn-implementation-the-recurrent-cell-4
:::

::: {.slide title="Embeddings, not one-hot"}
With $|\mathcal{V}| = 1{,}024$, one-hot inputs waste a
1,024-wide multiply per step on a vector of zeros.

- **Embedding lookup**: row $i$ of a trainable
  $\mathbf{W}_e \in \mathbb{R}^{|\mathcal{V}| \times d}$.
- Same map as one-hot $\times$ matrix, but the rows are *learned*.

@rnn-implementation-from-token-ids-to-embeddings-2

. . .

The equivalence, verified:

@rnn-implementation-from-token-ids-to-embeddings-3
:::

::: {.slide title="Wrapping as a language model"}
Embedding in, vocab-sized projection out; plot perplexity
instead of loss:

@rnn-implementation-from-token-ids-to-embeddings-1
:::

::: {.slide title="Output projection"}
Project every hidden state through the shared head, then a
shape smoke test: `(batch, steps)` ids in,
`(batch, steps, vocab)` logits out:

@rnn-implementation-the-output-layer-1

. . .

@rnn-implementation-the-output-layer-2
:::

::: {.slide title="Gradient clipping"}
Backprop through $T$ steps multiplies $T$ Jacobians, one
explosion-prone product. **Clip** the gradient onto a ball of
radius $\theta$ before each update:

$$\mathbf{g} \leftarrow \min\!\left(1, \frac{\theta}{\|\mathbf{g}\|}\right)\mathbf{g}.$$

@rnn-implementation-gradient-clipping

(PyTorch/MXNet: called by `fit_epoch`; TF: inside the compiled
step; JAX: `optax.clip_by_global_norm` does it inside `fit`.)
:::

::: {.slide title="Training"}
50k windows of 32 BPE tokens, batch 1024, 10 epochs, clip at 1.
Fresh zero state per window = **truncated BPTT**:

@rnn-implementation-training-1

. . .

@rnn-implementation-training-2

. . .

@rnn-implementation-training-3
:::

::: {.slide title="Reading the perplexity"}
@rnn-implementation-training-4

. . .

Val ppl ~90–100 over 1,024 tokens vs. char-level ppl ~7 over 27:
**not comparable**. Convert to bits per byte:

@rnn-implementation-training-5

~2.4 bpb beats the char-trigram baseline's 2.68 bpb: the "worse"
perplexity is the better language model.
:::

::: {.slide title="Generating text"}
Warm up on the prefix, then feed each chosen token back in.
Greedy ($T=0$) or temperature sampling:

@rnn-implementation-generating-text-1
:::

::: {.slide title="Greedy vs. temperature"}
@rnn-implementation-generating-text-2

. . .

Greedy is fluent, then **circles**. Sampling breaks the loop at
the price of stranger choices:

@rnn-implementation-generating-text-3

Doing better = decoding strategies, later in this chapter.
:::

::: {.slide title="Concise: the framework layer"}
Same interface as `RNNScratch`, one fused call:

@rnn-implementation-concise-implementation-1

. . .

The LM wrapper is inherited: swap in framework embedding and
dense layers:

@rnn-implementation-concise-implementation-2
:::

::: {.slide title="Sanity check, then train"}
Untrained model generates byte soup, but the wiring
(tokenizer to model and back) is sound:

@rnn-implementation-concise-implementation-3

@rnn-implementation-concise-implementation-4

. . .

Same trainer, same data:

@rnn-implementation-concise-implementation-5

@rnn-implementation-concise-implementation-6
:::

::: {.slide title="Scratch vs. concise, measured"}
@rnn-implementation-scratch-versus-concise-measured

- PyTorch/MXNet: fused kernel wins severalfold; per-step launch
  overhead dominates at this size.
- JAX: both versions JIT-compile; the gap is small.
- TF: `SimpleRNN` has **no** fused GPU kernel; the compiled
  scratch loop matches it.
:::

::: {.slide title="Recap"}
- BPE-token RNN LM: **embedding** + hand-rolled cell + shared
  head + cross-entropy.
- **Gradient clipping** is mandatory for stable RNN training.
- Fresh state per window = **truncated BPTT** through
  `num_steps` tokens.
- Compare models across tokenizers by **bits per byte**, never
  perplexity.
- Greedy decoding loops; temperature trades repetition for
  noise; decoding gets its own section.
- The same scaffold takes any cell (LSTM, GRU): only the
  recurrence changes. Coming next.
:::
