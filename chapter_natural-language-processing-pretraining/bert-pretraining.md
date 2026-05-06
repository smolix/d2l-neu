# Pretraining BERT
:label:`sec_bert-pretraining`

With the BERT model implemented in :numref:`sec_bert`
and the pretraining examples generated from the WikiText-2 dataset in :numref:`sec_bert-dataset`, we will pretrain BERT on the WikiText-2 dataset in this section.

```{.python .input #bert-pretraining-pretraining-bert-1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```{.python .input #bert-pretraining-pretraining-bert-1}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #bert-pretraining-pretraining-bert-1}
#@tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from flax.training import train_state
```

```{.python .input #bert-pretraining-pretraining-bert-1}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow import keras
```

To start, we load the WikiText-2 dataset as minibatches
of pretraining examples for masked language modeling and next sentence prediction.
The batch size is 512 and the maximum length of a BERT input sequence is 64.
Note that in the original BERT model, the maximum length is 512.

```{.python .input #bert-pretraining-pretraining-bert-2}
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## Pretraining BERT

The original BERT has two versions of different model sizes :cite:`Devlin.Chang.Lee.ea.2018`.
The base model ($\textrm{BERT}_{\textrm{BASE}}$) uses 12 layers (Transformer encoder blocks)
with 768 hidden units (hidden size) and 12 self-attention heads.
The large model ($\textrm{BERT}_{\textrm{LARGE}}$) uses 24 layers
with 1024 hidden units and 16 self-attention heads.
Notably, the former has 110 million parameters while the latter has 340 million parameters.
For demonstration with ease,
we define a small BERT, using 2 layers, 128 hidden units, and 2 self-attention heads.

```{.python .input #bert-pretraining-pretraining-bert-2-1}
#@tab mxnet
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input #bert-pretraining-pretraining-bert-2-1}
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, 
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
# Use reduction='none' so per-token losses can be masked properly below.
loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input #bert-pretraining-pretraining-bert-2-1}
#@tab jax
net = d2l.BERTModel(len(vocab), num_hiddens=128,
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
```

```{.python .input #bert-pretraining-pretraining-bert-2-1}
#@tab tensorflow
net = d2l.BERTModel(len(vocab), num_hiddens=128,
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
```

Before defining the training loop,
we define a helper function `_get_batch_loss_bert`.
Given the shard of training examples,
this function computes the loss for both the masked language modeling and next sentence prediction tasks.
Note that the final loss of BERT pretraining
is just the sum of both the masked language modeling loss
and the next sentence prediction loss.

```{.python .input #bert-pretraining-pretraining-bert-2-2}
#@tab mxnet
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Compute masked language model loss
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```{.python .input #bert-pretraining-pretraining-bert-2-2}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss (mask per-token losses before summing)
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y).mean()
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

```{.python .input #bert-pretraining-pretraining-bert-2-2}
#@tab jax
#@save
def _get_batch_loss_bert(params, net, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net.apply(params, tokens_X, segments_X,
                                        valid_lens_x.reshape(-1),
                                        pred_positions_X, training=True,
                                        rngs={'dropout': jax.random.PRNGKey(0)})
    # Compute masked language model loss
    mlm_l = optax.softmax_cross_entropy_with_integer_labels(
        mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
    mlm_l = (mlm_l * mlm_weights_X.reshape(-1)).sum() / (
        mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = optax.softmax_cross_entropy_with_integer_labels(
        nsp_Y_hat, nsp_y).mean()
    l = mlm_l + nsp_l
    return l, (mlm_l, nsp_l)
```

```{.python .input #bert-pretraining-pretraining-bert-2-2}
#@tab tensorflow
#@save
def _get_batch_loss_bert(net, vocab_size, tokens_X, segments_X,
                         valid_lens_x, pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y, training=True):
    mlm_loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    nsp_loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(
        tokens_X, segments_X, tf.cast(tf.reshape(valid_lens_x, [-1]),
                                      dtype=tf.float32),
        pred_positions_X, training=training)
    # Compute masked language model loss (mask per-token losses before summing)
    mlm_l = mlm_loss_fn(tf.reshape(mlm_Y, [-1]),
                        tf.reshape(mlm_Y_hat, [-1, vocab_size]))
    mlm_l = tf.reduce_sum(mlm_l * tf.reshape(mlm_weights_X, [-1])) / (
        tf.reduce_sum(mlm_weights_X) + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = tf.reduce_mean(nsp_loss_fn(tf.cast(nsp_y, tf.int32), nsp_Y_hat))
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

Invoking the two aforementioned helper functions,
the following `train_bert` function
defines the procedure to pretrain BERT (`net`) on the WikiText-2 (`train_iter`) dataset.
Training BERT can take very long.
Instead of specifying the number of epochs for training
as in the `train_ch13` function (see :numref:`sec_image_augmentation`),
the input `num_steps` of the following function
specifies the number of iteration steps for training.

```{.python .input #bert-pretraining-pretraining-bert-2-3}
#@tab mxnet
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input #bert-pretraining-pretraining-bert-2-3}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input #bert-pretraining-pretraining-bert-2-3}
#@tab jax
def train_bert(train_iter, net, vocab_size, num_steps):
    # Initialize model parameters using a dummy batch
    dummy_tokens = jnp.ones((2, 64), dtype=jnp.int32)
    dummy_segments = jnp.zeros((2, 64), dtype=jnp.int32)
    dummy_valid_lens = jnp.array([64, 64], dtype=jnp.float32)
    dummy_pred_positions = jnp.zeros((2, 10), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    params = net.init(key, dummy_tokens, dummy_segments, dummy_valid_lens,
                      dummy_pred_positions, training=False)
    tx = optax.adam(learning_rate=0.01)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)

    grad_fn = jax.value_and_grad(_get_batch_loss_bert, has_aux=True)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for (tokens_X, segments_X, valid_lens_x, pred_positions_X,
             mlm_weights_X, mlm_Y, nsp_y) in train_iter:
            timer.start()
            (l, (mlm_l, nsp_l)), grads = grad_fn(
                state.params, net, vocab_size, tokens_X, segments_X,
                valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            state = state.apply_gradients(grads=grads)
            metric.add(float(mlm_l), float(nsp_l), tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(jax.devices())}')
    return state
```

```{.python .input #bert-pretraining-pretraining-bert-2-3}
#@tab tensorflow
def train_bert(train_iter, net, vocab_size, devices, num_steps):
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for (tokens_X, segments_X, valid_lens_x, pred_positions_X,
             mlm_weights_X, mlm_Y, nsp_y) in train_iter:
            timer.start()
            with tf.GradientTape() as tape:
                mlm_l, nsp_l, l = _get_batch_loss_bert(
                    net, vocab_size, tokens_X, segments_X, valid_lens_x,
                    pred_positions_X, mlm_weights_X, mlm_Y, nsp_y,
                    training=True)
            grads = tape.gradient(l, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            metric.add(float(mlm_l), float(nsp_l), tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

We can plot both the masked language modeling loss and the next sentence prediction loss
during BERT pretraining.
The 50-step run below is a mechanics-only smoke test for the training loop;
it is intentionally too short to produce a converged BERT model or meaningful downstream representations.

```{.python .input #bert-pretraining-pretraining-bert-2-4}
#@tab mxnet
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

```{.python .input #bert-pretraining-pretraining-bert-2-4}
#@tab pytorch
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

```{.python .input #bert-pretraining-pretraining-bert-2-4}
#@tab jax
state = train_bert(train_iter, net, len(vocab), 50)
```

```{.python .input #bert-pretraining-pretraining-bert-2-4}
#@tab tensorflow
devices = d2l.try_all_gpus()
train_bert(train_iter, net, len(vocab), devices, 50)
```

## Representing Text with BERT

After pretraining BERT,
we can use it to represent single text, text pairs, or any token in them.
The following function returns the BERT (`net`) representations for all tokens
in `tokens_a` and `tokens_b`.

```{.python .input #bert-pretraining-representing-text-with-bert-1}
#@tab mxnet
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input #bert-pretraining-representing-text-with-bert-1}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input #bert-pretraining-representing-text-with-bert-1}
#@tab jax
def get_bert_encoding(net, params, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = jnp.array(vocab[tokens], dtype=jnp.int32)[None, :]
    segments = jnp.array(segments, dtype=jnp.int32)[None, :]
    valid_len = jnp.array([len(tokens)], dtype=jnp.float32)
    encoded_X, _, _ = net.apply(params, token_ids, segments, valid_len,
                                training=False)
    return encoded_X
```

```{.python .input #bert-pretraining-representing-text-with-bert-1}
#@tab tensorflow
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = tf.expand_dims(
        tf.constant(vocab[tokens], dtype=tf.int32), axis=0)
    segments = tf.expand_dims(
        tf.constant(segments, dtype=tf.int32), axis=0)
    valid_len = tf.constant([len(tokens)], dtype=tf.float32)
    encoded_X, _, _ = net(token_ids, segments, valid_len, training=False)
    return encoded_X
```

Consider the sentence "a crane is flying".
Recall the input representation of BERT as discussed in :numref:`subsec_bert_input_rep`.
After inserting special tokens “&lt;cls&gt;” (used for classification)
and “&lt;sep&gt;” (used for separation),
the BERT input sequence has a length of six.
Since zero is the index of the “&lt;cls&gt;” token,
`encoded_text[:, 0, :]` is the BERT representation of the entire input sentence.
To evaluate the polysemy token "crane",
we also print out the first three elements of the BERT representation of the token.

```{.python .input #bert-pretraining-representing-text-with-bert-2}
#@tab mxnet, pytorch
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

```{.python .input #bert-pretraining-representing-text-with-bert-2}
#@tab jax
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, state.params, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

```{.python .input #bert-pretraining-representing-text-with-bert-2}
#@tab tensorflow
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

Now consider a sentence pair
"a crane driver came" and "he just left".
Similarly, `encoded_pair[:, 0, :]` is the encoded result of the entire sentence pair from the pretrained BERT.
Note that the first three elements of the polysemy token "crane" are different from those when the context is different.
This supports that BERT representations are context-sensitive.

```{.python .input #bert-pretraining-representing-text-with-bert-3}
#@tab mxnet, pytorch
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

```{.python .input #bert-pretraining-representing-text-with-bert-3}
#@tab jax
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, state.params, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

```{.python .input #bert-pretraining-representing-text-with-bert-3}
#@tab tensorflow
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

In :numref:`chap_nlp_app`, we will fine-tune a pretrained BERT model
for downstream natural language processing applications.

## Summary

* The original BERT has two versions, where the base model has 110 million parameters and the large model has 340 million parameters.
* After pretraining BERT, we can use it to represent single text, text pairs, or any token in them.
* In the experiment, the same token has different BERT representation when their contexts are different. This supports that BERT representations are context-sensitive.


## Exercises

1. In the experiment, we can see that the masked language modeling loss is significantly higher than the next sentence prediction loss. Why?
2. Set the maximum length of a BERT input sequence to be 512 (same as the original BERT model). Use the configurations of the original BERT model such as $\textrm{BERT}_{\textrm{LARGE}}$. Do you encounter any error when running this section? Why?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1497)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/1497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1497)
:end_tab:

<!-- slides -->

::: {.slide}
With the model (last deck) and the data (deck before
that), we can finally pretrain a small BERT end-to-end.
This deck does it on a tiny scale: 2 layers, 128 hidden
dim, 2 heads. The recipe scales to BERT-Base (12 layers,
768 dim, 12 heads) and BERT-Large by just changing the
config.
:::

::: {.slide title="Setup + tiny BERT"}
@bert-pretraining-pretraining-bert-1

. . .

@bert-pretraining-pretraining-bert-2

. . .

@bert-pretraining-pretraining-bert-2-1
:::

::: {.slide title="Combined loss"}
Two heads, one combined loss:

$$\mathcal{L} = \mathcal{L}_\text{MLM} + \mathcal{L}_\text{NSP}.$$

MLM cross-entropy averaged over masked positions; NSP
binary cross-entropy on the `<cls>` head:

@bert-pretraining-pretraining-bert-2-2
:::

::: {.slide title="Training loop"}
Standard SGD with warmup; on this tiny corpus a few
hundred steps is enough to see both losses drop:

@bert-pretraining-pretraining-bert-2-3

. . .

@bert-pretraining-pretraining-bert-2-4
:::

::: {.slide title="Using the trained encoder"}
After pretraining, the encoder is the *useful* part —
turn token sequences into contextual representations:

@bert-pretraining-representing-text-with-bert-1
:::

::: {.slide title="Single sentence"}
"a crane is flying" → 6 hidden vectors (one per token,
including `<cls>` and `<sep>`). Each is *contextual* — the
representation of "crane" depends on its neighbors:

@bert-pretraining-representing-text-with-bert-2
:::

::: {.slide title="Sentence pair"}
"a crane driver came" / "he just left". Same encoder,
two-segment input — segment IDs distinguish the two halves
inside the same sequence:

@bert-pretraining-representing-text-with-bert-3
:::

::: {.slide title="Recap"}
- BERT pretraining is just two losses (MLM + NSP)
  optimized end-to-end on the encoder + heads.
- Output of pretraining: a *contextual* token encoder.
- For downstream tasks: load encoder weights, attach a
  small head, fine-tune. The next chapter does exactly
  this for sentiment classification, NLI, and SQuAD-style
  QA.
:::
