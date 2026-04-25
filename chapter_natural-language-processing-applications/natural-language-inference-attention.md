# Natural Language Inference: Using Attention
:label:`sec_natural-language-inference-attention`

We introduced the natural language inference task and the SNLI dataset in :numref:`sec_natural-language-inference-and-dataset`. In view of many models that are based on complex and deep architectures, :citet:`Parikh.Tackstrom.Das.ea.2016` proposed to address natural language inference with attention mechanisms and called it a "decomposable attention model".
This results in a model without recurrent or convolutional layers, achieving the best result at the time on the SNLI dataset with much fewer parameters.
In this section, we will describe and implement this attention-based method (with MLPs) for natural language inference, as depicted in :numref:`fig_nlp-map-nli-attention`.

![This section feeds pretrained GloVe to an architecture based on attention and MLPs for natural language inference.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`


## The Model

Rather than preserving the order of tokens in premises and hypotheses,
we can simply align tokens in one text sequence to every token in the other, and vice versa,
then compare and aggregate such information to predict the logical relationships
between premises and hypotheses.
Similar to alignment of tokens between source and target sentences in machine translation,
the alignment of tokens between premises and hypotheses
can be neatly accomplished by attention mechanisms.

![Natural language inference using attention mechanisms.](../img/nli-attention.svg)
:label:`fig_nli_attention`

:numref:`fig_nli_attention` depicts the natural language inference method using attention mechanisms.
At a high level, it consists of three jointly trained steps: attending, comparing, and aggregating.
We will illustrate them step by step in the following.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
import numpy as np
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import keras
```

### Attending

The first step is to align tokens in one text sequence to each token in the other sequence.
Suppose that the premise is "i do need sleep" and the hypothesis is "i am tired".
Due to semantical similarity,
we may wish to align "i" in the hypothesis with "i" in the premise,
and align "tired" in the hypothesis with "sleep" in the premise.
Likewise, we may wish to align "i" in the premise with "i" in the hypothesis,
and align "need" and "sleep" in the premise with "tired" in the hypothesis.
Note that such alignment is *soft* using weighted average,
where ideally large weights are associated with the tokens to be aligned.
For ease of demonstration, :numref:`fig_nli_attention` shows such alignment in a *hard* way.

Now we describe the soft alignment using attention mechanisms in more detail.
Denote by $\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$
and $\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ the premise and hypothesis,
whose number of tokens are $m$ and $n$, respectively,
where $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) is a $d$-dimensional word vector.
For soft alignment, we compute the attention weights $e_{ij} \in \mathbb{R}$ as

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

where the function $f$ is an MLP defined in the following `mlp` function.
The output dimension of $f$ is specified by the `num_hiddens` argument of `mlp`.

```{.python .input}
#@tab mxnet
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

```{.python .input}
#@tab jax
class MLP(nn.Module):
    num_hiddens: int
    flatten: bool

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dropout(0.2)(x, deterministic=not training)
        x = nn.Dense(self.num_hiddens)(x)
        x = nn.relu(x)
        if self.flatten:
            x = x.reshape((x.shape[0], -1))
        x = nn.Dropout(0.2)(x, deterministic=not training)
        x = nn.Dense(self.num_hiddens)(x)
        x = nn.relu(x)
        if self.flatten:
            x = x.reshape((x.shape[0], -1))
        return x
```

```{.python .input}
#@tab tensorflow
def mlp(num_hiddens, flatten):
    net = keras.Sequential()
    net.add(keras.layers.Dropout(0.2))
    net.add(keras.layers.Dense(num_hiddens, activation='relu'))
    if flatten:
        net.add(keras.layers.Flatten())
    net.add(keras.layers.Dropout(0.2))
    net.add(keras.layers.Dense(num_hiddens, activation='relu'))
    if flatten:
        net.add(keras.layers.Flatten())
    return net
```

It should be highlighted that, in :eqref:`eq_nli_e`
$f$ takes inputs $\mathbf{a}_i$ and $\mathbf{b}_j$ separately rather than taking a pair of them together as input.
This *decomposition* trick leads to only $m + n$ applications (linear complexity) of $f$ rather than $mn$ applications
(quadratic complexity).


Normalizing the attention weights in :eqref:`eq_nli_e`,
we compute the weighted average of all the token vectors in the hypothesis
to obtain representation of the hypothesis that is softly aligned with the token indexed by $i$ in the premise:

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

Likewise, we compute soft alignment of premise tokens for each token indexed by $j$ in the hypothesis:

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

Below we define the `Attend` class to compute the soft alignment of hypotheses (`beta`) with input premises `A` and soft alignment of premises (`alpha`) with input hypotheses `B`.

```{.python .input}
#@tab mxnet
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = npx.batch_dot(npx.softmax(e), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

```{.python .input}
#@tab jax
class Attend(nn.Module):
    num_hiddens: int

    @nn.compact
    def __call__(self, A, B, training=False):
        f = MLP(self.num_hiddens, flatten=False)
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = f(A, training=training)
        f_B = f(B, training=training)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = jnp.matmul(f_A, f_B.transpose(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = jnp.matmul(jax.nn.softmax(e, axis=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = jnp.matmul(jax.nn.softmax(e.transpose(0, 2, 1), axis=-1), A)
        return beta, alpha
```

```{.python .input}
#@tab tensorflow
class Attend(keras.layers.Layer):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def call(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = tf.matmul(f_A, tf.transpose(f_B, perm=[0, 2, 1]))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = tf.matmul(tf.nn.softmax(e, axis=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = tf.matmul(
            tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1]), axis=-1), A)
        return beta, alpha
```

### Comparing

In the next step, we compare a token in one sequence with the other sequence that is softly aligned with that token.
Note that in soft alignment, all the tokens from one sequence, though with probably different attention weights, will be compared with a token in the other sequence.
For ease of demonstration, :numref:`fig_nli_attention` pairs tokens with aligned tokens in a *hard* way.
For example, suppose that the attending step determines that "need" and "sleep" in the premise are both aligned with "tired" in the hypothesis, the pair "tired--need sleep" will be compared.

In the comparing step, we feed the concatenation (operator $[\cdot, \cdot]$) of tokens from one sequence and aligned tokens from the other sequence into a function $g$ (an MLP):

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab`


In :eqref:`eq_nli_v_ab`, $\mathbf{v}_{A,i}$ is the comparison between token $i$ in the premise and all the hypothesis tokens that are softly aligned with token $i$;
while $\mathbf{v}_{B,j}$ is the comparison between token $j$ in the hypothesis and all the premise tokens that are softly aligned with token $j$.
The following `Compare` class defines such a comparing step.

```{.python .input}
#@tab mxnet
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

```{.python .input}
#@tab jax
class Compare(nn.Module):
    num_hiddens: int

    @nn.compact
    def __call__(self, A, B, beta, alpha, training=False):
        g = MLP(self.num_hiddens, flatten=False)
        V_A = g(jnp.concatenate([A, beta], axis=2), training=training)
        V_B = g(jnp.concatenate([B, alpha], axis=2), training=training)
        return V_A, V_B
```

```{.python .input}
#@tab tensorflow
class Compare(keras.layers.Layer):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def call(self, A, B, beta, alpha):
        V_A = self.g(tf.concat([A, beta], axis=2))
        V_B = self.g(tf.concat([B, alpha], axis=2))
        return V_A, V_B
```

### Aggregating

With two sets of comparison vectors $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$) and $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$) on hand,
in the last step we will aggregate such information to infer the logical relationship.
We begin by summing up both sets:

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

Next we feed the concatenation of both summarization results into function $h$ (an MLP) to obtain the classification result of the logical relationship:

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

The aggregation step is defined in the following `Aggregate` class.

```{.python .input}
#@tab mxnet
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

```{.python .input}
#@tab jax
class Aggregate(nn.Module):
    num_hiddens: int
    num_outputs: int

    @nn.compact
    def __call__(self, V_A, V_B, training=False):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = nn.Dense(self.num_outputs)(
            MLP(self.num_hiddens, flatten=True)(
                jnp.concatenate([V_A, V_B], axis=1), training=training))
        return Y_hat
```

```{.python .input}
#@tab tensorflow
class Aggregate(keras.layers.Layer):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.linear = keras.layers.Dense(num_outputs)

    def call(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = tf.reduce_sum(V_A, axis=1)
        V_B = tf.reduce_sum(V_B, axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(tf.concat([V_A, V_B], axis=1)))
        return Y_hat
```

### Putting It All Together

By putting the attending, comparing, and aggregating steps together,
we define the decomposable attention model to jointly train these three steps.

```{.python .input}
#@tab mxnet
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab jax
class DecomposableAttention(nn.Module):
    vocab_size: int
    embed_size: int
    num_hiddens: int

    @nn.compact
    def __call__(self, premises, hypotheses, training=False):
        A = nn.Embed(self.vocab_size, self.embed_size)(premises)
        B = nn.Embed(self.vocab_size, self.embed_size)(hypotheses)
        beta, alpha = Attend(self.num_hiddens)(A, B, training=training)
        V_A, V_B = Compare(self.num_hiddens)(
            A, B, beta, alpha, training=training)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        Y_hat = Aggregate(self.num_hiddens, num_outputs=3)(
            V_A, V_B, training=training)
        return Y_hat
```

```{.python .input}
#@tab tensorflow
class DecomposableAttention(keras.Model):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = keras.layers.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def call(self, inputs, training=False, **kwargs):
        premises, hypotheses = inputs
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## Training and Evaluating the Model

Now we will train and evaluate the defined decomposable attention model on the SNLI dataset.
We begin by reading the dataset.


### Reading the dataset

We download and read the SNLI dataset using the function defined in :numref:`sec_natural-language-inference-and-dataset`. The batch size and sequence length are set to $256$ and $50$, respectively.

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### Creating the Model

We use the pretrained 100-dimensional GloVe embedding to represent the input tokens.
Thus, we predefine the dimension of vectors $\mathbf{a}_i$ and $\mathbf{b}_j$ in :eqref:`eq_nli_e` as 100.
The output dimension of functions $f$ in :eqref:`eq_nli_e` and $g$ in :eqref:`eq_nli_v_ab` is set to 200.
Then we create a model instance, initialize its parameters,
and load the GloVe embedding to initialize vectors of input tokens.

```{.python .input}
#@tab mxnet
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

```{.python .input}
#@tab jax
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(len(vocab), embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
```

```{.python .input}
#@tab tensorflow
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
# Build the embedding layer before assigning pretrained weights.
net.embedding.build((None,))
net.embedding.set_weights([embeds])
```

### Training and Evaluating the Model

In contrast to the `split_batch` function in :numref:`sec_multi_gpu` that takes single inputs such as text sequences (or images),
we define a `split_batch_multi_inputs` function to take multiple inputs such as premises and hypotheses in minibatches.

```{.python .input}
#@tab mxnet
#@save
def split_batch_multi_inputs(X, y, devices):
    """Split multi-input `X` and `y` into multiple devices."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

Now we can train and evaluate the model on the SNLI dataset.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab jax
lr, num_epochs = 0.001, 4
# Initialize model parameters
dummy_premises = jnp.ones((1, 50), dtype=jnp.int32)
dummy_hypotheses = jnp.ones((1, 50), dtype=jnp.int32)
rng = jax.random.PRNGKey(0)
params = net.init(rng, dummy_premises, dummy_hypotheses, training=False)
# Replace the embedding parameters with pretrained GloVe embeddings
params = {**params, 'params': {**params['params'],
    'Embed_0': {'embedding': jnp.array(embeds)}}}

optimizer = optax.adam(lr)
opt_state = optimizer.init(params)

def loss_fn(params, premises, hypotheses, labels, rng):
    logits = net.apply(params, premises, hypotheses, training=True,
                       rngs={'dropout': rng})
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, labels).mean()

@jax.jit
def train_step(params, opt_state, premises, hypotheses, labels, rng):
    loss, grads = jax.value_and_grad(loss_fn)(
        params, premises, hypotheses, labels, rng)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def eval_step(params, premises, hypotheses, labels):
    logits = net.apply(params, premises, hypotheses, training=False)
    return (logits.argmax(axis=-1) == labels).sum()

for epoch in range(num_epochs):
    train_loss, n_train = 0.0, 0
    for batch in train_iter:
        premises, hypotheses, labels = batch[0], batch[1], batch[2]
        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss = train_step(
            params, opt_state, premises, hypotheses, labels, step_rng)
        train_loss += float(loss) * len(labels)
        n_train += len(labels)
    # Evaluate on test set
    n_correct, n_test = 0, 0
    for batch in test_iter:
        premises, hypotheses, labels = batch[0], batch[1], batch[2]
        n_correct += int(eval_step(params, premises, hypotheses, labels))
        n_test += len(labels)
    print(f'epoch {epoch + 1}, loss {train_loss / n_train:.4f}, '
          f'test acc {n_correct / n_test:.4f}')
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.001, 4
# Wrap tf.data batches: each yields (premises, hypotheses, labels);
# Keras model expects (X, y) where X = (premises, hypotheses)
def reformat(premises, hypotheses, labels):
    return (premises, hypotheses), labels

train_iter_tf = train_iter.map(reformat)
test_iter_tf = test_iter.map(reformat)
net.compile(optimizer=keras.optimizers.Adam(lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
net.fit(train_iter_tf, validation_data=test_iter_tf, epochs=num_epochs)
```

### Using the Model

Finally, define the prediction function to output the logical relationship between a pair of premise and hypothesis.

```{.python .input}
#@tab mxnet
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab jax
#@save
def predict_snli(net, params, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    premise = jnp.array(vocab[premise]).reshape((1, -1))
    hypothesis = jnp.array(vocab[hypothesis]).reshape((1, -1))
    label = jnp.argmax(net.apply(params, premise, hypothesis, training=False),
                       axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab tensorflow
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    premise = tf.constant([vocab[premise]], dtype=tf.int32)
    hypothesis = tf.constant([vocab[hypothesis]], dtype=tf.int32)
    label = tf.argmax(net((premise, hypothesis), training=False), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

We can use the trained model to obtain the natural language inference result for a sample pair of sentences.

```{.python .input}
#@tab mxnet, pytorch
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

```{.python .input}
#@tab jax
predict_snli(net, params, vocab, ['he', 'is', 'good', '.'],
             ['he', 'is', 'bad', '.'])
```

```{.python .input}
#@tab tensorflow
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## Summary

* The decomposable attention model consists of three steps for predicting the logical relationships between premises and hypotheses: attending, comparing, and aggregating.
* With attention mechanisms, we can align tokens in one text sequence to every token in the other, and vice versa. Such alignment is soft using weighted average, where ideally large weights are associated with the tokens to be aligned.
* The decomposition trick leads to a more desirable linear complexity than quadratic complexity when computing attention weights.
* We can use pretrained word vectors as the input representation for downstream natural language processing tasks such as natural language inference.


## Exercises

1. Train the model with other combinations of hyperparameters. Can you get better accuracy on the test set?
1. What are major drawbacks of the decomposable attention model for natural language inference?
1. Suppose that we want to get the level of semantical similarity (e.g., a continuous value between 0 and 1) for any pair of sentences. How shall we collect and label the dataset? Can you design a model with attention mechanisms?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1530)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/1530)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1530)
:end_tab:
