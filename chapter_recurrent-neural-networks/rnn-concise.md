# Concise Implementation of Recurrent Neural Networks
:label:`sec_rnn-concise`

Like most of our from-scratch implementations,
:numref:`sec_rnn-scratch` was designed 
to provide insight into how each component works.
But when you are using RNNs every day 
or writing production code,
you will want to rely more on libraries
that cut down on both implementation time 
(by supplying library code for common models and functions)
and computation time 
(by optimizing the heck out of these library implementations).
This section will show you how to implement 
the same language model more efficiently
using the high-level API provided 
by your deep learning framework.
We begin, as before, by loading 
*The Time Machine* dataset.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input #rnn-concise-concise-implementation-of-recurrent-neural-networks}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input #rnn-concise-concise-implementation-of-recurrent-neural-networks}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #rnn-concise-concise-implementation-of-recurrent-neural-networks}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #rnn-concise-concise-implementation-of-recurrent-neural-networks}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
```

## Defining the Model

We define the following class
using the RNN implemented
by high-level APIs.

:begin_tab:`mxnet`
Specifically, to initialize the hidden state,
we invoke the member method `begin_state`.
This returns a list that contains
an initial hidden state
for each example in the minibatch,
whose shape is
(number of hidden layers, batch size, number of hidden units).
For some models to be introduced later
(e.g., long short-term memory),
this list will also contain other information.
:end_tab:

:begin_tab:`jax`
Flax does not provide an RNNCell for concise implementation of Vanilla RNNs
as of today. There are more advanced variants of RNNs like LSTMs and GRUs
which are available in the Flax `linen` API.
:end_tab:

```{.python .input #rnn-concise-defining-the-model-1}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input #rnn-concise-defining-the-model-1}
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

```{.python .input #rnn-concise-defining-the-model-1}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True)
        
    def forward(self, inputs, H=None):
        # inputs: (time_steps, batch_size, features) -> (batch_size, time_steps, features)
        outputs, H = self.rnn(tf.transpose(inputs, perm=[1, 0, 2]), H)
        return tf.transpose(outputs, perm=[1, 0, 2]), H
```

```{.python .input #rnn-concise-defining-the-model-1}
%%tab jax
class RNN(nn.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        if H is None:
            batch_size = inputs.shape[1]
            H = nn.SimpleCell(features=self.num_hiddens).initialize_carry(
                jax.random.PRNGKey(0), (batch_size, self.num_hiddens))

        SimpleRNN = nn.scan(nn.SimpleCell, variable_broadcast="params",
                            in_axes=0, out_axes=0,
                            split_rngs={"params": False})

        H, outputs = SimpleRNN(features=self.num_hiddens)(H, inputs)
        return outputs, H
```

Inheriting from the `RNNLMScratch` class in :numref:`sec_rnn-scratch`, 
the following `RNNLM` class defines a complete RNN-based language model.
Note that we need to create a separate fully connected output layer.

```{.python .input #rnn-concise-defining-the-model-2}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input #rnn-concise-defining-the-model-2}
%%tab mxnet
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.Dense(self.vocab_size, flatten=False)
        self.initialize()
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)        
```

```{.python .input #rnn-concise-defining-the-model-2}
%%tab tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input #rnn-concise-defining-the-model-2}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## Training and Predicting

Before training the model, let's make a prediction 
with a model initialized with random weights.
Given that we have not trained the network, 
it will generate nonsensical predictions.

```{.python .input #rnn-concise-training-and-predicting-1}
%%tab pytorch
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

```{.python .input #rnn-concise-training-and-predicting-1}
%%tab mxnet
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
rnn = RNN(num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

```{.python .input #rnn-concise-training-and-predicting-1}
%%tab tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
rnn = RNN(num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Next, we train our model, leveraging the high-level API.

```{.python .input #rnn-concise-training-and-predicting-2}
%%tab pytorch
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input #rnn-concise-training-and-predicting-2}
%%tab mxnet
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input #rnn-concise-training-and-predicting-2}
%%tab tensorflow
with d2l.try_gpu():
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

Compared with :numref:`sec_rnn-scratch`,
this model achieves comparable perplexity,
but runs faster due to the optimized implementations.
As before, we can print the final validation perplexity
beside generated tokens following a prefix from the corpus.

```{.python .input #rnn-concise-training-and-predicting-3}
%%tab mxnet, pytorch
ppl = float(model.board.data['val_ppl'][-1].y)
pred = model.predict('time traveller', 20, data.vocab, d2l.try_gpu())
print(f'perplexity {ppl:.1f}, {pred!r}')
```

```{.python .input #rnn-concise-training-and-predicting-3}
%%tab tensorflow
ppl = float(model.board.data['val_ppl'][-1].y)
pred = model.predict('time traveller', 20, data.vocab)
print(f'perplexity {ppl:.1f}, {pred!r}')
```

## Summary

High-level APIs in deep learning frameworks provide implementations of standard RNNs.
These libraries help you to avoid wasting time reimplementing standard models.
Moreover,
framework implementations are often highly optimized, 
  leading to significant (computational) performance gains 
  when compared with implementations from scratch.

## Exercises

1. Can you make the RNN model overfit using the high-level APIs?
1. Implement the autoregressive model of :numref:`sec_sequence` using an RNN.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18015)
:end_tab:

<!-- slides -->

::: {.slide title="Concise RNNs"}
The same character-level LM, using the framework's built-in
`nn.RNN`. The cell + unroll + projection from scratch boil down
to a few lines:

- `nn.RNN(input_size, hidden_size)` handles the recurrence,
  including hardware-accelerated cuDNN kernels on GPU.
- Reuse the `RNNLMScratch` head — it doesn't care whether the
  cell is hand-rolled.
- Same `Trainer`, same gradient clipping, same data.

End result: faster training, ~5× fewer lines of code, identical
mathematics.
:::

::: {.slide title="The model"}
Built-in `RNN` cell + handing off the rest of the LM scaffold to
the from-scratch base class:

@rnn-concise-concise-implementation-of-recurrent-neural-networks

@rnn-concise-defining-the-model-1

. . .

@rnn-concise-defining-the-model-2
:::

::: {.slide title="Sanity check"}
Untrained model still runs — predictions are random characters,
but shapes line up. This check isolates API wiring from
learning quality:

@rnn-concise-training-and-predicting-1
:::

::: {.slide title="Training and decoding"}
Same `Trainer`, with `gradient_clip_val=1` on the optimizer:

@rnn-concise-training-and-predicting-2

. . .

@rnn-concise-training-and-predicting-3

Output looks like simple English-shaped text — same character-
level statistics the from-scratch version learned, in much less
training time.
:::

::: {.slide title="Recap"}
- `nn.RNN` is the cell + unroll + (with cuDNN) GPU kernels in
  one stock layer.
- Reuse the from-scratch LM wrapper — only the cell changes.
- Same scaffold accepts `nn.LSTM`, `nn.GRU`, etc. — drop-in
  replacements with better long-range gradient behavior.
- The framework version trains noticeably faster than the
  from-scratch version on the same hardware.
:::
