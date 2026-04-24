# Pretraining word2vec
:label:`sec_word2vec_pretraining`


We go on to implement the skip-gram
model defined in
:numref:`sec_word2vec`.
Then
we will pretrain word2vec using negative sampling
on the PTB dataset.
First of all,
let's obtain the data iterator
and the vocabulary for this dataset
by calling the `d2l.load_data_ptb`
function, which was described in :numref:`sec_word2vec_data`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
import math
import numpy as np
import optax

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## The Skip-Gram Model

We implement the skip-gram model
by using embedding layers and batch matrix multiplications.
First, let's review
how embedding layers work.


### Embedding Layer

As described in :numref:`sec_seq2seq`,
an embedding layer
maps a token's index to its feature vector.
The weight of this layer
is a matrix whose number of rows equals to
the dictionary size (`input_dim`) and
number of columns equals to
the vector dimension for each token (`output_dim`).
After a word embedding model is trained,
this weight is what we need.

```{.python .input}
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

```{.python .input}
#@tab jax
embed = nn.Embed(num_embeddings=20, features=4)
params = embed.init(jax.random.PRNGKey(0), jnp.ones((1,), dtype=jnp.int32))
print(f'Parameter embedding ({params["params"]["embedding"].shape}, '
      f'dtype={params["params"]["embedding"].dtype})')
```

The input of an embedding layer is the
index of a token (word).
For any token index $i$,
its vector representation
can be obtained from
the $i^\textrm{th}$ row of the weight matrix
in the embedding layer.
Since the vector dimension (`output_dim`)
was set to 4,
the embedding layer
returns vectors with shape (2, 3, 4)
for a minibatch of token indices with shape
(2, 3).

```{.python .input}
#@tab mxnet, pytorch
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

```{.python .input}
#@tab jax
x = jnp.array([[1, 2, 3], [4, 5, 6]])
embed.apply(params, x)
```

### Defining the Forward Propagation

In the forward propagation,
the input of the skip-gram model
includes
the center word indices `center`
of shape (batch size, 1)
and
the concatenated context and noise word indices `contexts_and_negatives`
of shape (batch size, `max_len`),
where `max_len`
is defined
in :numref:`subsec_word2vec-minibatch-loading`.
These two variables are first transformed from the
token indices into vectors via the embedding layer,
then their batch matrix multiplication
(described in :numref:`subsec_batch_dot`)
returns
an output of shape (batch size, 1, `max_len`).
Each element in the output is the dot product of
a center word vector and a context or noise word vector.

```{.python .input}
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

```{.python .input}
#@tab jax
def skip_gram(center, contexts_and_negatives, embed_v, embed_u,
              params_v, params_u):
    v = embed_v.apply(params_v, center)
    u = embed_u.apply(params_u, contexts_and_negatives)
    pred = jnp.matmul(v, jnp.transpose(u, (0, 2, 1)))
    return pred
```

Let's print the output shape of this `skip_gram` function for some example inputs.

```{.python .input}
#@tab mxnet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

```{.python .input}
#@tab jax
skip_gram(jnp.ones((2, 1), dtype=jnp.int32),
          jnp.ones((2, 4), dtype=jnp.int32), embed, embed,
          params, params).shape
```

## Training

Before training the skip-gram model with negative sampling,
let's first define its loss function.


### Binary Cross-Entropy Loss

According to the definition of the loss function
for negative sampling in :numref:`subsec_negative-sampling`, 
we will use 
the binary cross-entropy loss.

```{.python .input}
#@tab mxnet
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

```{.python .input}
#@tab jax
def loss(inputs, target, mask=None):
    """Binary cross-entropy loss with masking."""
    out = optax.sigmoid_binary_cross_entropy(inputs, target)
    if mask is not None:
        out = out * mask
    return out.mean(axis=1)
```

Recall our descriptions
of the mask variable
and the label variable in
:numref:`subsec_word2vec-minibatch-loading`.
The following
calculates the 
binary cross-entropy loss
for the given variables.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Below shows
how the above results are calculated
(in a less efficient way)
using the
sigmoid activation function
in the binary cross-entropy loss.
We can consider 
the two outputs as
two normalized losses
that are averaged over non-masked predictions.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Initializing Model Parameters

We define two embedding layers
for all the words in the vocabulary
when they are used as center words
and context words, respectively.
The word vector dimension
`embed_size` is set to 100.

```{.python .input}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

```{.python .input}
#@tab jax
embed_size = 100
embed_v = nn.Embed(num_embeddings=len(vocab), features=embed_size)
embed_u = nn.Embed(num_embeddings=len(vocab), features=embed_size)
```

### Defining the Training Loop

The training loop is defined below. Because of the existence of padding, the calculation of the loss function is slightly different compared to the previous training functions.

```{.python .input}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab jax
def train(embed_v, embed_u, data_iter, lr, num_epochs):
    key = jax.random.PRNGKey(42)
    key, key_v, key_u = jax.random.split(key, 3)
    # Initialize parameters
    dummy = jnp.ones((1,), dtype=jnp.int32)
    params_v = embed_v.init(key_v, dummy)
    params_u = embed_u.init(key_u, dummy)
    all_params = {'v': params_v, 'u': params_u}
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(all_params)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])

    @jax.jit
    def train_step(all_params, opt_state, center, context_negative,
                   mask, label):
        def compute_loss(all_params):
            pred = skip_gram(center, context_negative, embed_v, embed_u,
                             all_params['v'], all_params['u'])
            l = (loss(pred.reshape(label.shape), label, mask)
                 / mask.sum(axis=1) * mask.shape[1])
            return l.sum(), l.size
        (loss_val, l_size), grads = jax.value_and_grad(
            compute_loss, has_aux=True)(all_params)
        updates, opt_state = optimizer.update(grads, opt_state, all_params)
        all_params = optax.apply_updates(all_params, updates)
        return all_params, opt_state, loss_val, l_size

    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        # Accumulate on device to avoid per-batch host syncs
        loss_sum, count = jnp.array(0.0), jnp.array(0, dtype=jnp.int32)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = batch
            all_params, opt_state, loss_val, l_size = train_step(
                all_params, opt_state, center, context_negative, mask, label)
            loss_sum = loss_sum + loss_val
            count = count + l_size
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (float(loss_sum / count),))
    total_loss = float(loss_sum)
    total_count = int(count)
    print(f'loss {total_loss / total_count:.3f}, '
          f'{total_count / timer.stop():.1f} tokens/sec')
    return all_params
```

Now we can train a skip-gram model using negative sampling.

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

```{.python .input}
#@tab jax
lr, num_epochs = 0.002, 5
all_params = train(embed_v, embed_u, data_iter, lr, num_epochs)
```

## Applying Word Embeddings
:label:`subsec_apply-word-embed`


After training the word2vec model,
we can use the cosine similarity
of word vectors from the trained model
to 
find words from the dictionary
that are most semantically similar
to an input word.

```{.python .input}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab jax
def get_similar_tokens(query_token, k, embed_params):
    W = embed_params['params']['embedding']
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = jnp.dot(W, x) / jnp.sqrt(jnp.sum(W * W, axis=1) *
                                    jnp.sum(x * x) + 1e-9)
    topk = jnp.argsort(-cos)[:k + 1]
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(int(i))}')

get_similar_tokens('chip', 3, all_params['v'])
```

## Summary

* We can train a skip-gram model with negative sampling using embedding layers and the binary cross-entropy loss.
* Applications of word embeddings include finding semantically similar words for a given word based on the cosine similarity of word vectors.


## Exercises

1. Using the trained model, find semantically similar words for other input words. Can you improve the results by tuning hyperparameters?
1. When a training corpus is huge, we often sample context words and noise words for the center words in the current minibatch *when updating model parameters*. In other words, the same center word may have different context words or noise words in different training epochs. What are the benefits of this method? Try to implement this training method.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:
