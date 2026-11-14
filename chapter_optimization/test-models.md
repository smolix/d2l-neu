# Two Test Models
:label:`sec_test_models`

<!-- WRITER: NEW opening from the reader (structure 4, guide 3.1): the valley said what to expect; how do we learn what an optimizer does to a network, and when is a difference between two runs a result? Say that `TinyLM` is a transformer used two chapters before it is explained (:numref:`chap_transformers`) and that an optimizer sees only parameters and gradients. The paragraph below is the old opening of "TinyLM and FashionCNN" in adam.md, moved verbatim. -->

A comparison of optimizers needs models on which the optimizers behave
differently, and models small enough to train many times. We use two of
about the same size: a transformer language model, the kind of model Adam
is used for in practice, and a convolutional image classifier, the kind of
model that SGD with momentum trained for a decade. Both go into the `d2l`
library, and the rest of this chapter trains them.

```{.python .input #test-models-two-test-models-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #test-models-two-test-models-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #test-models-two-test-models-2}
%%tab pytorch
import os
import pandas as pd
import random
import textwrap
```

```{.python .input #test-models-two-test-models-2}
%%tab jax
import os
import pandas as pd
import textwrap
```

## The Language Model
:label:`subsec_tinylm`

<!-- WRITER: ADAM-9: The Time Machine has some 170,000 characters (text-sequence.md; 173,428 on its slide), so each character would be a target about 47 times, not 80. -->

The data is WikiText-103 :cite:`Merity.Xiong.Bradbury.ea.2016`, about 100
million words of Wikipedia articles kept as raw text, with case, digits, and
punctuation. A run of 2,000 steps on minibatches of 64 windows of 64 tokens
makes about 8 million predictions. On the 100,000 characters of *The Time
Machine* each character would be a target about 80 times, and the model
would memorize the text. Here the training windows start at random offsets
in the first of the two training files, about 270 million bytes, so a run
reads about 3% of it and rarely sees the same text twice. The tokens are the
UTF-8 bytes of the text: the byte-level tokenizer of
:numref:`sec_text-sequence` without any merges, a vocabulary of 256 tokens,
so the loss converts directly to bits per byte
(:numref:`sec_language-model`). Validation windows are consecutive,
non-overlapping pieces of the official validation split, about 1.1 million
bytes.

```{.python .input #adam-the-language-model-1}
%%tab pytorch
class WikiText103(d2l.DataModule):  #@save
    """WikiText-103 as UTF-8 bytes, cut into windows of num_steps bytes."""
    def __init__(self, batch_size, num_steps, num_train=None, num_val=None):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = d2l.BPETokenizer(vocab_size=256, specials=())  # Bytes
        self.train = self._read('train-00000-of-00002',
                                '3cbc792dfc0eeedf8d7333acdc15a9f536032d9a')
        self.val = self._read('validation-00000-of-00001',
                              '49d15019b0476afc752bd5f010b5a2aab2de89ca')
        self.train, self.val = self.train[:num_train], self.val[:num_val]

    def _read(self, split, sha1):
        url = ('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
               f'main/wikitext-103-raw-v1/{split}.parquet')
        fname = d2l.download(url, os.path.join(self.root, 'wikitext-103'),
                             sha1)
        lines = pd.read_parquet(fname)['text']  # Each line ends with '\n'
        return np.frombuffer(b''.join(s.encode() for s in lines), np.uint8)

    def windows(self, corpus, starts):
        """Inputs and next-byte targets of the windows at `starts`."""
        idx = np.asarray(starts)[:, None] + np.arange(self.num_steps + 1)
        array = torch.from_numpy(corpus[idx].astype(np.int64))
        return array[:, :-1], array[:, 1:]

    def train_dataloader(self):  # A pass of minibatches at random offsets
        n = len(self.train) - self.num_steps
        for _ in range(n // (self.batch_size * self.num_steps)):
            yield self.windows(self.train,
                               torch.randint(n, (self.batch_size,)).numpy())

    def val_dataloader(self):  # Consecutive, non-overlapping windows
        starts = np.arange(0, len(self.val) - self.num_steps, self.num_steps)
        for i in range(0, len(starts), self.batch_size):
            yield self.windows(self.val, starts[i:i + self.batch_size])
```

```{.python .input #adam-the-language-model-1}
%%tab jax
class WikiText103(d2l.DataModule):  #@save
    """WikiText-103 as UTF-8 bytes, cut into windows of num_steps bytes."""
    def __init__(self, batch_size, num_steps, num_train=None, num_val=None):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = d2l.BPETokenizer(vocab_size=256, specials=())  # Bytes
        self.train = self._read('train-00000-of-00002',
                                '3cbc792dfc0eeedf8d7333acdc15a9f536032d9a')
        self.val = self._read('validation-00000-of-00001',
                              '49d15019b0476afc752bd5f010b5a2aab2de89ca')
        self.train, self.val = self.train[:num_train], self.val[:num_val]

    def _read(self, split, sha1):
        url = ('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
               f'main/wikitext-103-raw-v1/{split}.parquet')
        fname = d2l.download(url, os.path.join(self.root, 'wikitext-103'),
                             sha1)
        lines = pd.read_parquet(fname)['text']  # Each line ends with '\n'
        return np.frombuffer(b''.join(s.encode() for s in lines), np.uint8)

    def windows(self, corpus, starts):
        """Inputs and next-byte targets of the windows at `starts`."""
        idx = np.asarray(starts)[:, None] + np.arange(self.num_steps + 1)
        array = corpus[idx].astype(np.int32)
        return array[:, :-1], array[:, 1:]

    def train_dataloader(self):  # A pass of minibatches at random offsets
        n = len(self.train) - self.num_steps
        rng = np.random.default_rng(d2l.get_seed())
        for _ in range(n // (self.batch_size * self.num_steps)):
            yield self.windows(self.train,
                               rng.integers(n, size=self.batch_size))

    def val_dataloader(self):  # Consecutive, non-overlapping windows
        starts = np.arange(0, len(self.val) - self.num_steps, self.num_steps)
        for i in range(0, len(starts), self.batch_size):
            yield self.windows(self.val, starts[i:i + self.batch_size])
```

Both splits are held in memory as arrays of bytes, one byte per token, and
a minibatch is gathered from them on demand. A training pass draws as many
windows as the text holds non-overlapping ones, each at an independent
random offset, so `train_steps` below can run for any number of steps without
reusing a fixed set of windows.

```{.python .input #adam-a-tiny-language-model-1}
%%tab pytorch, jax
data = WikiText103(batch_size=64, num_steps=64)
print(f'{len(data.train):,} training bytes, '
      f'{len(data.val):,} validation bytes')
text = data.vocab.decode(data.train[:300])
print('\n'.join(textwrap.fill(line, 72) for line in text.splitlines()))
```

### The Model

The model, `TinyLM`, is a *decoder-only transformer*, the architecture of
:numref:`chap_transformers`. We use it two chapters early because an
optimizer does not need to know what attention does: it sees a
differentiable function with parameters of several kinds, and only their
gradients. The model has learned token and position embeddings, two blocks,
and a linear head that predicts the next byte. Each block has an
attention layer and a two-layer MLP, each behind a LayerNorm and each with a
residual connection. The attention computation is a single library call.
Every linear layer draws its weights and biases uniformly from
$\pm 1/\sqrt{d_\textrm{in}}$, and every embedding row from a standard
normal distribution.

```{.python .input #adam-a-tiny-language-model-2}
%%tab pytorch
class TinyLM(nn.Module):  #@save
    """A small decoder-only transformer language model."""
    def __init__(self, vocab_size, d_model=128, num_heads=2, num_blks=2,
                 max_len=64):
        super().__init__()
        self.num_heads = num_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blks = nn.ModuleList([nn.ModuleDict(dict(
            norm1=nn.LayerNorm(d_model),
            qkv=nn.Linear(d_model, 3 * d_model),
            proj=nn.Linear(d_model, d_model),
            norm2=nn.LayerNorm(d_model),
            mlp=nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                              nn.Linear(4 * d_model, d_model))))
            for _ in range(num_blks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def attention(self, blk, X):
        B, T, D = X.shape
        q, k, v = blk['qkv'](X).chunk(3, dim=-1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
                   for u in (q, k, v))
        Y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return blk['proj'](Y.transpose(1, 2).reshape(B, T, D))

    def forward(self, X):
        H = self.token_emb(X) + self.pos_emb(torch.arange(X.shape[1],
                                                          device=X.device))
        for blk in self.blks:
            H = H + self.attention(blk, blk['norm1'](H))
            H = H + blk['mlp'](blk['norm2'](H))
        return self.head(self.norm(H))
```

```{.python .input #adam-a-tiny-language-model-2}
%%tab jax
class TinyLM(nnx.Module):  #@save
    """A small decoder-only transformer language model."""
    def __init__(self, vocab_size, d_model=128, num_heads=2, num_blks=2,
                 max_len=64, *, rngs):
        self.num_heads = num_heads
        def linear(d_in, d_out):
            a = 1 / math.sqrt(d_in)  # Weights and biases ~ U(-a, a)
            init = lambda key, shape, dtype: jax.random.uniform(
                key, shape, dtype, -a, a)
            return nnx.Linear(d_in, d_out, kernel_init=init, bias_init=init,
                              rngs=rngs)
        def embed(n):  # Embedding rows ~ N(0, I)
            return d2l.Embed(n, d_model, rngs=rngs,
                             embedding_init=nnx.initializers.normal(1.0))
        def norm():
            return nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.token_emb, self.pos_emb = embed(vocab_size), embed(max_len)
        self.blks = nnx.List([nnx.Dict(
            norm1=norm(), qkv=linear(d_model, 3 * d_model),
            proj=linear(d_model, d_model), norm2=norm(),
            mlp1=linear(d_model, 4 * d_model),
            mlp2=linear(4 * d_model, d_model))
            for _ in range(num_blks)])
        self.norm = norm()
        self.head = linear(d_model, vocab_size)

    def attention(self, blk, X):
        B, T, D = X.shape
        q, k, v = jnp.split(blk['qkv'](X), 3, axis=-1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1) for u in (q, k, v))
        Y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        return blk['proj'](Y.reshape(B, T, D))

    def __call__(self, X):
        H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = H + self.attention(blk, blk['norm1'](H))
            H = H + blk['mlp2'](jax.nn.gelu(blk['mlp1'](blk['norm2'](H)),
                                            approximate=False))
        return self.head(self.norm(H))
```

<!-- WRITER: the two tab notes on initialization (nn.Linear/nn.Embedding defaults, exact GELU; Flax's initializers passed explicitly, epsilon 1e-5, approximate=False, rngs required, d2l.Embed) are now comments in the count cell below (the saved TinyLM block cannot change by a byte). -->

### Parameter Groups

### Parameter Groups

The parameters of `TinyLM` fall into three groups. *Embeddings* are tables
whose rows receive a gradient only when their token or position occurs,
the sparse features that motivated AdaGrad. Two-dimensional *matrices*, the
weights of the linear layers, hold nearly all of the parameters.
One-dimensional *vectors* are the biases and the LayerNorm scales and
shifts. Weight decay at the end of this section and Muon in
:numref:`sec_muon` treat the three groups differently, so we count them
once here.

```{.python .input #adam-a-tiny-language-model-3}
%%tab pytorch
# nn.Linear and nn.Embedding initialize as described above by default;
# nn.GELU is the exact GELU
model = TinyLM(vocab_size=len(data.vocab))
kinds = {'embeddings': 0, 'matrices': 0, 'vectors': 0}
print(f'{"parameter":<22} {"shape":<14} {"count":>7}')
for name, p in model.named_parameters():
    kind = ('embeddings' if 'emb' in name else
            'matrices' if p.ndim == 2 else 'vectors')
    kinds[kind] += p.numel()
    print(f'{name:<22} {str(tuple(p.shape)):<14} {p.numel():>7}')
for kind, count in kinds.items():
    print(f'{kind:>37} {count:>7}')
print(f'{"total":>37} {sum(kinds.values()):>7}')
```

```{.python .input #adam-a-tiny-language-model-3}
%%tab jax
# Flax's default initializers differ, so TinyLM passes the ones described
# above (and approximate=False for the exact GELU); d2l.Embed computes
# the embedding gradient as one matrix product, far faster than XLA's
# default on frequent bytes such as the space
model = TinyLM(vocab_size=len(data.vocab), rngs=nnx.Rngs(d2l.get_key()))
kinds = {'embeddings': 0, 'matrices': 0, 'vectors': 0}
print(f'{"parameter":<22} {"shape":<14} {"count":>7}')
for path, p in nnx.to_flat_state(nnx.state(model, nnx.Param)):
    name = '.'.join(str(k) for k in path)
    kind = ('embeddings' if 'emb' in name else
            'matrices' if p.ndim == 2 else 'vectors')
    kinds[kind] += p.size
    print(f'{name:<22} {str(tuple(p.shape)):<14} {p.size:>7}')
for kind, count in kinds.items():
    print(f'{kind:>37} {count:>7}')
print(f'{"total":>37} {sum(kinds.values()):>7}')
```

The model has about 0.47 million parameters, 90% of them in the matrices.

<!-- WRITER: "Weight decay at the end of this section" above now lives in :numref:`sec_adamw`. -->

## The Image Classifier
:label:`subsec_fashioncnn`

`FashionCNN` classifies the $28 \times 28$ grayscale images of
Fashion-MNIST (:numref:`sec_fashion_mnist`). Two stages of a $3 \times 3$
convolution, a ReLU, and $2 \times 2$ max-pooling, with 32 and 64 channels,
reduce each image to $64 \times 7 \times 7$ features. Two fully connected
layers map them to 128 hidden units and then to the ten classes. Every
weight and bias is drawn uniformly from $\pm 1/\sqrt{\textrm{fan-in}}$,
where the fan-in of a unit is the number of inputs it sums over.

```{.python .input #adam-the-same-race-on-a-cnn-1}
%%tab pytorch
class FashionCNN(nn.Module):  #@save
    """A compact CNN for Fashion-MNIST: two conv-pool stages, two linear."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), 2)
        X = F.max_pool2d(F.relu(self.conv2(X)), 2)
        return self.fc2(F.relu(self.fc1(X.flatten(1))))
```

```{.python .input #adam-the-same-race-on-a-cnn-1}
%%tab jax
class FashionCNN(nnx.Module):  #@save
    """A compact CNN for Fashion-MNIST: two conv-pool stages, two linear."""
    def __init__(self, *, rngs):
        def init(fan_in):  # Weights and biases ~ U(-a, a), a = 1/sqrt(fan_in)
            a = 1 / math.sqrt(fan_in)
            f = lambda key, shape, dtype: jax.random.uniform(
                key, shape, dtype, -a, a)
            return dict(kernel_init=f, bias_init=f, rngs=rngs)
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), **init(9))
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), **init(9 * 32))
        self.fc1 = nnx.Linear(64 * 7 * 7, 128, **init(64 * 7 * 7))
        self.fc2 = nnx.Linear(128, 10, **init(128))

    def __call__(self, X):
        X = nnx.max_pool(nnx.relu(self.conv1(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = nnx.max_pool(nnx.relu(self.conv2(X)), window_shape=(2, 2),
                         strides=(2, 2))
        return self.fc2(nnx.relu(self.fc1(X.reshape(X.shape[0], -1))))
```

<!-- WRITER: the two tab notes on FashionCNN (initialization, image layout) are now comments in the new count cell below, which prints 421,642 in both tabs. "Since `train_steps` only asks a model for logits" comes before train_steps is introduced; reorder or reword. -->

```{.python .input #test-models-the-image-classifier-1}
%%tab pytorch
# nn.Conv2d and nn.Linear draw from U(-a, a), a = 1/sqrt(fan-in), by default;
# images arrive as (batch, channel, height, width)
print(f'{d2l.num_params(FashionCNN()):,} parameters')
```

```{.python .input #test-models-the-image-classifier-1}
%%tab jax
# Flax's default initializers differ, so FashionCNN passes U(-a, a)
# explicitly; images arrive as (batch, height, width, channel)
print(f'{d2l.num_params(FashionCNN(rngs=nnx.Rngs(d2l.get_key()))):,} '
      'parameters')
```

With 421,642 parameters, `FashionCNN` is about the same size as
`TinyLM`. Since `train_steps` only asks a model for logits, it trains the CNN
unchanged; Fashion-MNIST minibatches of 256 images take the place of the
byte windows. The CNN also needs a test accuracy: `evaluate_accuracy`
returns the fraction of the examples in a set of minibatches that a model
classifies correctly.

```{.python .input #adam-the-image-classifier-1}
%%tab pytorch
fashion = d2l.FashionMNIST(batch_size=256)

def evaluate_accuracy(model, batches):  #@save
    """Fraction of the examples in batches that model classifies correctly."""
    device = next(model.parameters()).device
    training, correct, total = model.training, 0, 0
    model.eval()  # Dropout off, batch norm on running statistics
    with torch.inference_mode():
        for X, y in batches:
            y_hat = model(X.to(device)).argmax(dim=1)
            correct += (y_hat == y.to(device)).sum()
            total += len(y)
    model.train(training)
    return float(correct) / total
```

```{.python .input #adam-the-image-classifier-1}
%%tab jax
fashion = d2l.FashionMNIST(batch_size=256)

@nnx.jit  #@save
def num_correct(model, X, y):
    """Number of examples in (X, y) that model classifies correctly."""
    return (model(X).argmax(axis=1) == y).sum()

def evaluate_accuracy(model, batches):  #@save
    """Fraction of the examples in batches that model classifies correctly."""
    # A view with dropout off and batch norm on running statistics
    model = nnx.view(model, deterministic=True, use_running_average=True,
                     raise_if_not_found=False)
    correct, total = 0, 0
    for X, y in batches:
        correct += num_correct(model, jnp.asarray(X), jnp.asarray(y))
        total += len(y)
    return float(correct) / total
```

## Training and Reading a Run

<!-- WRITER: ADAM-8: present the helpers in a small table (name, what it returns, first use): train_steps, lm_perplexity, evaluate_accuracy, final_loss, smooth, steps_to_reach (now saved), lr_sweep; code shown without narration. ADAM-10: the -100 labels serve the masked language model of chapter 11; say so in a clause. -->

Training is a loop counted in steps rather than epochs: draw a minibatch,
compute the mean cross-entropy of the next-token predictions, take an
optimizer step, record the loss. Nothing in it is specific to language: it
trains any model that maps a minibatch to logits, so we save it for the
rest of the book as `train_steps`. Positions labeled $-100$ add no loss,
which lets a model train on part of its outputs only. An optional function
`stop` of the step count ends a run early once it returns true.

```{.python .input #adam-a-tiny-language-model-4}
%%tab pytorch
def train_steps(model, data, optimizer, num_steps, *, stop=None):  #@save
    """Minimize the mean cross-entropy of the logits model(X) for the labels
    Y over the minibatches (X, Y) of data, leaving out the labels -100, for
    num_steps steps or until stop(step) is true; return every step's loss."""
    device = d2l.try_gpu()
    model.to(device)
    losses, step = [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   Y.reshape(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            step += 1
            if step >= num_steps or (stop is not None and stop(step)):
                return torch.stack(losses).tolist()
```

```{.python .input #adam-a-tiny-language-model-4}
%%tab jax
@nnx.jit  #@save
def _cross_entropy_step(model, optimizer, X, Y):
    """One optimizer step on the mean cross-entropy of the logits model(X)
    for the labels Y; negative labels, such as -100, add no loss."""
    def loss_fn(model):
        logits = model(X)
        logits, y = logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)
        valid = y >= 0
        losses = optax.softmax_cross_entropy_with_integer_labels(
            logits, jnp.where(valid, y, 0))
        return (losses * valid).sum() / valid.sum()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def train_steps(model, data, optimizer, num_steps, *, stop=None):  #@save
    """Minimize the mean cross-entropy of the logits model(X) for the labels
    Y over the minibatches (X, Y) of data, leaving out the labels -100, for
    num_steps steps or until stop(step) is true; return every step's loss."""
    losses, pending, step = [], [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            pending.append(_cross_entropy_step(
                model, optimizer, jnp.asarray(X), jnp.asarray(Y)))
            step += 1
            done = step >= num_steps or (stop is not None and stop(step))
            if len(pending) == 50 or done:  # Sync now and then
                losses.extend(float(l) for l in jax.device_get(pending))
                pending = []
            if done:
                return losses
```

The counterpart for evaluation is `lm_perplexity`. It sums the
cross-entropy of the next-token predictions over a set of minibatches,
usually the validation set, with dropout off, and returns the exponential
of the mean, the perplexity of :numref:`sec_language-model`. Its logarithm
is the mean loss per token.

```{.python .input #adam-training-for-a-fixed-number-of-steps-1}
%%tab pytorch
def lm_perplexity(model, batches):  #@save
    """Perplexity of a language model on the minibatches (X, Y) of batches."""
    device = next(model.parameters()).device
    modes = {m: m.training for m in model.modules()}  # Restored at the end
    model.eval()  # Dropout off while we measure
    total, count = 0, 0
    with torch.inference_mode():
        for X, Y in batches:
            logits = model(X.to(device))
            total += F.cross_entropy(logits.flatten(0, 1),
                                     Y.to(device).flatten(), reduction='sum')
            count += Y.numel()
    for m, training in modes.items():
        m.training = training
    return math.exp(float(total) / count)
```

```{.python .input #adam-training-for-a-fixed-number-of-steps-1}
%%tab jax
@jax.jit(static_argnums=0)  #@save
def lm_loss_sum(graphdef, state, X, Y):
    """Summed next-token cross-entropy of a batch, and the model's state."""
    model = nnx.merge(graphdef, state)
    loss = optax.softmax_cross_entropy_with_integer_labels(model(X), Y).sum()
    return loss, nnx.state(model)

def lm_perplexity(model, batches):  #@save
    """Perplexity of a language model on the minibatches (X, Y) of batches."""
    # A view with dropout off, sharing the weights; one walk through it
    graphdef, state = nnx.split(nnx.view(model, deterministic=True,
                                         raise_if_not_found=False))
    total, count = 0, 0
    for X, Y in batches:
        loss, state = lm_loss_sum(graphdef, state, X, Y)
        total, count = total + loss, count + Y.size
    nnx.update(model, state)  # Keep statistics the model records, if any
    return math.exp(float(total) / count)
```

Three helpers read a run: the final loss, averaged over the last hundred
steps and set to infinity if the run diverged; a curve smoothed over blocks
of 25 steps for plotting; and the first step at which the smoothed curve
reaches a given loss, or `None` if it never does. Later sections use the
first two as well.

```{.python .input #adam-a-tiny-language-model-5}
%%tab pytorch, jax
def final_loss(losses, k=100):  #@save
    """Mean of the last k losses; infinity if the run diverged."""
    v = sum(losses[-k:]) / k
    return v if math.isfinite(v) else float('inf')

def smooth(losses, k=25):  #@save
    """Means over consecutive blocks of k losses, for plotting."""
    return [sum(losses[i:i + k]) / k
            for i in range(0, len(losses) - k + 1, k)]

def steps_to_reach(losses, target, k=25):  #@save
    """First step at which the smoothed losses reach target; None if they
    never do."""
    for i, v in enumerate(smooth(losses, k)):
        if v <= target:
            return k * (i + 1)
    return None
```

The function `lr_sweep` trains one fresh model per learning rate, every
one from the same initialization, and prints its final loss unless
`verbose` is false. Given a function `evaluate`, it also evaluates each
model, at the end of training or every `every` steps, and prints the last
value under the name `metric`. Later sections run their comparisons with
it as well.

```{.python .input #adam-the-race-on-the-language-model-1}
%%tab pytorch
def lr_sweep(make_model, make_optimizer, data, lrs, seed,  #@save
             num_steps=2000, *, evaluate=None, metric='score', every=None,
             verbose=True):
    """Train make_model() with make_optimizer(model, lr) for each lr, from
    one initialization; return the loss curves and the evaluations."""
    every = every or num_steps
    curves, scores = {}, {}
    for lr in lrs:
        torch.manual_seed(seed)  # The same initialization for every lr
        model = make_model()
        optimizer = make_optimizer(model, lr)
        curves[lr], scores[lr] = [], []
        for _ in range(num_steps // every):
            curves[lr] += train_steps(model, data, optimizer, every)
            if evaluate is not None:
                scores[lr].append(evaluate(model))
        line = f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}'
        if scores[lr]:
            line += f', {metric} {scores[lr][-1]:.3f}'
        if verbose:
            print(line)
    return curves, scores
```

```{.python .input #adam-the-race-on-the-language-model-1}
%%tab jax
def lr_sweep(make_model, make_tx, data, lrs, seed,  #@save
             num_steps=2000, *, evaluate=None, metric='score', every=None,
             verbose=True):
    """Train make_model(rngs=...) with the optax transformation make_tx(lr)
    for each lr, from one initialization; return the loss curves and the
    evaluations."""
    every = every or num_steps
    curves, scores = {}, {}
    for lr in lrs:  # The same initialization for every lr
        model = make_model(rngs=nnx.Rngs(seed))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        curves[lr], scores[lr] = [], []
        for _ in range(num_steps // every):
            curves[lr] += train_steps(model, data, optimizer, every)
            if evaluate is not None:
                scores[lr].append(evaluate(model))
        line = f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}'
        if scores[lr]:
            line += f', {metric} {scores[lr][-1]:.3f}'
        if verbose:
            print(line)
    return curves, scores
```

## A First Run with Minibatch SGD

<!-- WRITER: NEW subsection (structure 4.4). The baseline that Momentum, Adam and Muon have to beat: plain minibatch SGD (no momentum) on TinyLM at three rates, then FashionCNN at its good rate. Rates 0.1, 0.5, 2.5 (a factor of 5 apart). Plot: raw per-step losses on a log step axis (the explosion happens in the first ten steps, which a smoothed linear plot hides), legend outside; the dotted line is the uniform guess ln 256 = 5.55, about where every run starts (5.7). Measured, three notebook runs per tab: lr 0.1 ends at 2.473 to 2.493 and is still falling slowly and smoothly (too low); lr 0.5 ends at 2.090 to 2.134, still falling by about 0.02 per 100 steps at step 2,000, so there is no noise floor to see in 2,000 steps (the run is limited by its step count); lr 2.5 prints inf in every run: the loss drops at step 2, is above its start by step 3 and leaves the plot (too high). Development probes, not printed by the notebook (four to five seeds per rate and tab): every run at 1.5, 2, 2.5 and 3 became non-finite within 9 to 16 steps (two JAX exceptions, at steps 52 and 456); at lr 1, 9 of 18 runs diverged (steps 19 to 32), the others ended at 1.91 to 2.06; lr 0.5 and 0.7 never diverged (8 of 8 each; 0.7 ends at 2.01 to 2.14). The good plain-SGD rate thus sits one factor of two below a rate at which half the runs explode; this held in both tabs (no tab difference: JAX at full fp32 matmul precision diverged in 4 of 6 runs at lr 1). FashionCNN, plain SGD at 0.3: final loss 0.220 to 0.232, test accuracy 0.877 to 0.908 (six runs); in the probes lr 1 stayed at ln 10 = 2.304 with test accuracy 0.100 (the uniform-guess symptom, both tabs) and lr 0.1 ended at 0.284 to 0.292. -->

```{.python .input #test-models-a-first-run-with-minibatch-sgd-1}
%%tab pytorch
make_lm = lambda: TinyLM(len(data.vocab))
sgd = lambda model, lr: torch.optim.SGD(model.parameters(), lr)
new_seed = lambda: random.getrandbits(31)
```

```{.python .input #test-models-a-first-run-with-minibatch-sgd-1}
%%tab jax
make_lm = lambda rngs: TinyLM(len(data.vocab), rngs=rngs)
sgd = optax.sgd
new_seed = d2l.get_seed
```

```{.python .input #test-models-a-first-run-with-minibatch-sgd-2}
%%tab pytorch, jax
lm_sgd, _ = lr_sweep(make_lm, sgd, data, lrs=[0.1, 0.5, 2.5], seed=new_seed())
d2l.plot(list(range(1, 2001)),
         list(lm_sgd.values()) + [[math.log(256)] * 2000],
         'step', 'training loss', xscale='log', ylim=[1.5, 6],
         legend=[f'lr {lr:g}' for lr in lm_sgd] + ['uniform guess'],
         legend_loc='outside', figsize=(5, 2.5))
```

```{.python .input #test-models-a-first-run-with-minibatch-sgd-3}
%%tab pytorch, jax
test_acc = lambda model: evaluate_accuracy(model, fashion.val_dataloader())
cnn_sgd, _ = lr_sweep(FashionCNN, sgd, fashion, lrs=[0.3], seed=new_seed(),
                      evaluate=test_acc, metric='test accuracy')
```

## Comparing Two Runs

<!-- WRITER: NEW subsection (structure 4.5). State the protocol once here: lr_sweep gives every arm the same initialization (and, in PyTorch, the same data order); grids by factors of about three; scientific, nuisance and fixed hyperparameters (moved from practice.md "Scientific and Nuisance Hyperparameters" and adam.md "Ours is symmetric and small", both still in place there); which loss is read and why (training loss on single-pass text is a held-out loss); "tuned" means best grid point; the noise rule of lr-scheduler.md. The yardstick setting (author question 4, default): TinyLM with AdamW at 3e-3 and no weight decay, the recipe of :numref:`sec_training_recipes`; FashionCNN with plain SGD at 0.3, the baseline above. Measured, three notebook runs per tab (six batches of five): TinyLM, AdamW 3e-3: training loss 1.579 to 1.608 overall, range within a batch 0.010, 0.029, 0.012 (PyTorch) and 0.016, 0.016, 0.009 (JAX); validation loss 1.557 to 1.587, ranges 0.009, 0.026, 0.010 and 0.007, 0.019, 0.013. FashionCNN, SGD 0.3: training loss 0.214 to 0.236, ranges 0.007, 0.017, 0.018 and 0.016, 0.015, 0.010; test accuracy 0.880 to 0.909, ranges 0.014, 0.018, 0.006 and 0.016, 0.029, 0.016. A usable-range margin of 0.1 nats on TinyLM is about three times the largest printed range; a single-run test-accuracy difference on FashionCNN below about 3 points is within the spread. -->

```{.python .input #test-models-comparing-two-runs-1}
%%tab pytorch
adamw = lambda model, lr: torch.optim.AdamW(model.parameters(), lr,
                                            weight_decay=0)
```

```{.python .input #test-models-comparing-two-runs-1}
%%tab jax
adamw = lambda lr: optax.adamw(lr, weight_decay=0)
```

```{.python .input #test-models-comparing-two-runs-2}
%%tab pytorch, jax
def repeat(make_model, make_optimizer, data, lr, evaluate, seeds):
    """Final training loss and evaluation of one setting, once per seed."""
    runs = [lr_sweep(make_model, make_optimizer, data, [lr], seed,
                     evaluate=evaluate, verbose=False) for seed in seeds]
    return ([final_loss(curves[lr]) for curves, _ in runs],
            [scores[lr][-1] for _, scores in runs])
```

```{.python .input #test-models-comparing-two-runs-3}
%%tab pytorch, jax
val_loss = lambda model: math.log(lm_perplexity(model, data.val_dataloader()))
lm_train, lm_val = repeat(make_lm, adamw, data, 3e-3, val_loss,
                          seeds=[new_seed() for _ in range(5)])
cnn_train, cnn_acc = repeat(FashionCNN, sgd, fashion, 0.3, test_acc,
                            seeds=[new_seed() for _ in range(5)])
for name, values in [('TinyLM, training loss', lm_train),
                     ('TinyLM, validation loss', lm_val),
                     ('FashionCNN, training loss', cnn_train),
                     ('FashionCNN, test accuracy', cnn_acc)]:
    print(f'{name:<26}', ' '.join(f'{v:.3f}' for v in values),
          f'  range {max(values) - min(values):.3f}')
```

## Summary

<!-- WRITER: new Summary. -->

## Exercises

<!-- WRITER: five to six exercises (structure 4): the harness-related Adam exercises (Exercise 4 of adam.md, per-group SGD rates, if it fits better here), "diagnose three curves", a seed-spread exercise. -->

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.4]{.kicker}

Two Test Models<br>**a transformer and a CNN · minibatch SGD at three rates · the spread between two runs**
:::
:::

<!-- WRITER: new deck (slide guide): two models in two slides, the three-curve diagnosis, the spread yardstick. The slide below moved from the Adam deck. -->

::: {.slide title="TinyLM and FashionCNN: Two Models of About 0.45M Parameters"}
[Two models]{.kicker}

| | `TinyLM` | `FashionCNN` |
|---|---|---|
| task | next byte, WikiText-103 | 10 classes, Fashion-MNIST |
| layers | 2 pre-norm transformer blocks, width 128 | 2 conv-pool stages, 2 linear |
| parameters | 470,784 | 421,642 |
| minibatch | 64 windows of 64 bytes | 256 images |

`TinyLM`'s groups: embeddings 9%, matrices 90%, vectors 1%. A run reads
about 3% of the 270M training bytes and sees almost no text twice. Both
models are in `d2l` and train with `d2l.train_steps`.

::: {.d2l-note}
Protocol: same random initialization, 2,000 steps, constant rate, four
rates per method spaced by about 3×; "tuned" is the lowest final loss.
:::
:::
