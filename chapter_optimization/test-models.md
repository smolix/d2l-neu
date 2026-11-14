# Two Test Models
:label:`sec_test_models`

Until now, every claim of this chapter could be checked with a pencil. The
valley has two curvatures, and from them we computed the largest stable
learning rate, the number of steps that gradient descent needs, and the
size of the cloud in which SGD comes to rest. A network offers nothing of
the kind. Does momentum help on a real model? Why are transformers trained
with Adam, when SGD served image classifiers for a decade? No formula
answers such questions. We have to train the network and look.

Looking has a difficulty of its own, which you have met if you ever trained
one model twice. The two runs start from different random weights, see
different minibatches, and end at different losses. If a new optimizer then
ends 0.02 below the old one, was it the optimizer or the draw? The rule we
adopt is as old as measurement: a difference between two settings counts
only if it is larger than the difference between two runs of the same
setting. To apply it we need models small enough to train dozens of times,
and we need to know how far apart two runs of one setting end.

This section supplies both. The models are a language model, `TinyLM`, and
an image classifier, `FashionCNN`, each with a little under half a million
parameters. `TinyLM` is a transformer, an architecture that
:numref:`chap_transformers` explains. We use it two chapters early because
an optimizer does not need to know what attention does: it sees parameters
and their gradients, and nothing else. We train both models with the plain
minibatch SGD of :numref:`sec_sgd`, as a baseline, and then train one
setting five times to measure the spread. Later sections and chapters
reuse the models and the functions defined here.

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

## The Language Model TinyLM
:label:`subsec_tinylm`

A test model for optimizers must not be able to memorize its data, or a low
training loss would measure memory and not optimization. One of our runs
takes 2,000 steps on minibatches of 64 windows of 64 tokens, about 8
million predictions. On *The Time Machine* with its 170,000 characters
(:numref:`sec_text-sequence`), each character would be a target almost 50
times. We therefore use a corpus that a run cannot exhaust: WikiText-103
:cite:`Merity.Xiong.Bradbury.ea.2016`, about 100 million words of Wikipedia
articles kept as raw text, with case, digits, and punctuation. Training
windows start at random offsets in the first of its two training files,
about 270 million bytes, so a run reads about 3% of the file and rarely
sees the same passage twice. Validation windows are consecutive pieces of
the official validation split, about 1.1 million bytes.

The tokens are the UTF-8 bytes of the text, as the byte-level tokenizer
of :numref:`sec_text-sequence` produces them before any merge. A
vocabulary of 256 byte values needs no training of its own, and the loss,
divided by $\ln 2$, is the number of bits that the model spends per byte
(:numref:`subsec_bits-per-byte`).

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

`WikiText103` holds both splits in memory and gathers each minibatch on
demand. A pass over the training data draws as many windows as the file
holds non-overlapping ones, each at an independent random offset, so a
training loop can ask for any number of steps. We load the data and decode
its first 300 bytes.

```{.python .input #adam-a-tiny-language-model-1}
%%tab pytorch, jax
data = WikiText103(batch_size=64, num_steps=64)
print(f'{len(data.train):,} training bytes, '
      f'{len(data.val):,} validation bytes')
text = data.vocab.decode(data.train[:300])
print('\n'.join(textwrap.fill(line, 72) for line in text.splitlines()))
```

The text comes as the authors of the dataset prepared it for word-level
models, with spaces around punctuation and `@-@` for a hyphen. Our tokens
are bytes, not characters: each Japanese character in this passage takes
three of them.

### The Model

`TinyLM` is a *decoder-only transformer*. Read from input to output, it
adds a learned embedding of each byte to a learned embedding of its
position, passes the sum through two blocks, and maps the result through a
layer normalization (:numref:`subsec_layer-normalization-in-bn`) and a
linear layer, the head, to 256 logits for the next byte. Each block has an
attention layer, through which a position reads from the positions before
it, and a two-layer MLP. Each of the two sits behind a layer normalization
and has a residual connection around it. Embeddings and blocks have width
128. Every linear layer draws its weights and biases uniformly from
$\pm 1/\sqrt{d_\textrm{in}}$, where $d_\textrm{in}$ is the number of its
inputs, and every embedding row from a standard normal distribution.

You can read the class below as a list of the layers that hold parameters.
The method `attention` is one library call wrapped in reshapes, and it will
make sense once you have read the transformer chapter. What this chapter
needs from the model are the names and shapes of its parameters.

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

### Parameter Groups

The parameters of `TinyLM` are of three kinds. *Embeddings* are tables, in
which a row receives a gradient only when its byte or its position occurs
in the minibatch. *Matrices* are the two-dimensional weights of the linear
layers. *Vectors* are the biases and the scales and shifts of the layer
normalizations. Later sections treat the three kinds differently (weight
decay in :numref:`sec_adamw`, for instance, applies to the matrices only),
so we count them once here. The four matrices of a block have
$128 \cdot (384 + 128 + 512 + 512) = 196{,}608$ entries, so we expect the
matrices to dominate.

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

The model has 470,784 parameters: 90% of them in the matrices, 9% in the
embeddings and under 1% in the vectors.

## The Image Classifier FashionCNN
:label:`subsec_fashioncnn`

What holds for a transformer need not hold for a convolutional network, so
the second model is one. `FashionCNN` classifies the $28 \times 28$
grayscale images of Fashion-MNIST (:numref:`sec_fashion_mnist`). Two stages
of a $3 \times 3$ convolution, a ReLU, and $2 \times 2$ max-pooling, with
32 and 64 channels, reduce each image to $64 \times 7 \times 7$ features.
Two fully connected layers map them to 128 hidden units and then to the ten
classes. Every weight and bias is drawn uniformly from
$\pm 1/\sqrt{\textrm{fan-in}}$, where the fan-in of a unit is the number of
inputs it sums over.

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

How large is this model? The first fully connected layer alone has
$3{,}136 \times 128$ weights, about 0.4 million, and the two convolutions
are small by comparison.

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

With 421,642 parameters `FashionCNN` is about as large as `TinyLM`, and 95%
of them sit in a single matrix. Its data are the minibatches of
Fashion-MNIST that the Basics part used, 256 images each. A classifier is
judged by its test accuracy as well as by its loss, so we add
`evaluate_accuracy`, which returns the fraction of the examples in a set of
minibatches that a model classifies correctly.

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

Both models map a minibatch to logits and are trained on the cross-entropy
of those logits, so one loop serves both. We count its length in steps and
not in epochs: an epoch has no meaning for a run that reads 3% of its text,
and the predictions of the previous sections were counted in steps.
:numref:`tab_test-models-helpers` lists the loop and the functions that
read off what a run did. All are saved in `d2l`, and the code follows in
the order of the table.

:The functions that train a model and read a run, with the section that
first calls each of them.
:label:`tab_test-models-helpers`

| function | returns | first used |
|:--|:--|:--|
| `train_steps` | the loss of every step of a training run | this section |
| `lm_perplexity` | the perplexity of a language model on a set of minibatches (:numref:`sec_language-model`); its logarithm is the mean loss per token | this section |
| `evaluate_accuracy` | the fraction of examples classified correctly (defined above) | this section |
| `final_loss` | the mean of the last 100 losses of a run, or infinity if the run diverged | this section |
| `smooth` | the means of consecutive blocks of 25 losses, for plotting | :numref:`sec_adam` |
| `steps_to_reach` | the first step at which the smoothed loss is at most a target, or `None` | :numref:`sec_momentum` |
| `lr_sweep` | the loss curves, and on request the evaluations, of one run per learning rate, all from the same initial weights | this section |

Two features of `train_steps` are there for later use. Positions whose
label is $-100$ add no loss, a convention that the masked language model
of :numref:`sec_transformer` relies on; no label in this chapter has that
value. And an optional function `stop` of the step count ends a run early,
for runs that train until they reach a target loss.

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

`lr_sweep` takes two functions, one that builds a new model and one that
builds its optimizer for a given learning rate, and trains once per rate.
Given a function `evaluate`, it also evaluates each model, at the end of
training or every `every` steps, and prints the last value under the name
`metric`.

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

We can now train. The optimizer is the plain minibatch SGD of
:numref:`sec_sgd`, without momentum and at a constant learning rate, for
2,000 steps. Which rate? On the valley the largest curvature $L$ put a
ceiling of $2/L$ on the learning rate (:numref:`sec_gd`). The loss of
`TinyLM` has a ceiling too, but we cannot compute it, so we bracket it with
three rates a factor of five apart: 0.1, 0.5 and 2.5.

One number we can predict, the loss at which the curves start. A model that
knows nothing about its text gives each of the 256 byte values the same
probability, at a loss of $\ln 256 \approx 5.55$ nats per byte. The plot
marks this *uniform guess* with a dotted line. It shows the loss of every
step on a logarithmic step axis, because the three runs part ways within
the first ten steps. In the code, `make_lm` and `sgd` build the model and
the optimizer in the form that `lr_sweep` expects, and `new_seed` returns a
new seed at every call, so each execution of this notebook starts from
weights of its own.

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

All three curves start a little above the dotted line, near 5.7, because
random logits prefer some bytes to others for no reason (Exercise
:exref:`ex_test-models-initial-loss` computes the excess). From there the
runs behave in three ways, and each is worth learning to recognize.

At $\eta = 2.5$ the rate is too high. The first step lowers the loss by
more than a nat, the second throws it above its starting value, and the
run ends as `inf`. This is the divergence of :numref:`sec_gd`, where a
rate above $2/L$ multiplied the error along the steepest direction at
every step. Divergence of this kind is fast: within a handful of steps
the loss has left the plot.

At $\eta = 0.1$ the rate is too low. The curve is smooth from the first
step on, and after 2,000 steps it is still falling, at about 2.5. Nothing
is wrong with this run except that it is slow.

At $\eta = 0.5$ the loss falls much faster at first, and during the first
thirty steps it is thrown back up several times, by up to a nat or so.
Each time it recovers, and the run ends near 2.1, a third of a nat below
the cautious one. A rate that can throw the loss up but not out is close
to the largest one the run tolerates. On the valley the best rate, 0.476,
lay just below the ceiling of 0.5 (:numref:`sec_optimization-intro`), and
the spikes suggest that 0.5 is in the same position here.

Two more things can be read off the plot. The band of useful rates is
narrow: five times below the best of our three rates costs a third of a
nat, and five times above it costs the run. And neither stable run has
reached the state in which :numref:`sec_sgd` left SGD, a loss that has
stopped falling and only fluctuates in its noise ball. After 2,000 steps
both curves still point downward. These runs are short of steps, not stuck
in noise.

`FashionCNN` gets a single run, at a rate of 0.3 that the same bracketing
finds (Exercise :exref:`ex_test-models-diagnosis`), and we print its test
accuracy at the end.

```{.python .input #test-models-a-first-run-with-minibatch-sgd-3}
%%tab pytorch, jax
test_acc = lambda model: evaluate_accuracy(model, fashion.val_dataloader())
cnn_sgd, _ = lr_sweep(FashionCNN, sgd, fashion, lrs=[0.3], seed=new_seed(),
                      evaluate=test_acc, metric='test accuracy')
```

After 2,000 steps the classifier has a training loss of about 0.22 and
classifies about 90% of the test images correctly. These two runs are our
baselines: plain SGD at a constant rate reaches about 2.1 nats per byte on
`TinyLM` and about 90% on `FashionCNN`.

## Comparing Two Runs

The run at $\eta = 0.5$ ended a third of a nat below the run at
$\eta = 0.1$. Before we read anything into a difference between two runs,
we should agree on how the runs are set up, and we should know how much two
runs of the *same* setting differ. Every comparison in this chapter follows
the protocol below.

### The Protocol

The Deep Learning Tuning Playbook :cite:`Godbole.Dahl.Gilmer.ea.2023` sorts
the hyperparameters of an experiment into three classes. The *scientific*
hyperparameter is the one the question is about, such as the optimizer; we
call each of its values an *arm* of the comparison. A *nuisance*
hyperparameter must be set to its best value separately for every arm
before the comparison is fair. The learning rate is the standard case,
since a single shared rate favors whichever method happens to prefer it.
*Fixed* hyperparameters are the same in every arm, and each of them limits
the claim. Ours are the model, the batch size, the 2,000 steps and, unless
a section says otherwise, a constant learning rate without clipping or
weight decay.

We tune the learning rate of an arm with `lr_sweep`, on a grid of rates a
factor of about three apart, or two where a section needs a finer answer,
and all runs of a comparison start from the same initial weights. *Tuned*
means no more than the best point of such a grid. Since a grid cannot
place an optimum more finely than its spacing, we also report the *usable
range* of a method, the rates whose final loss is within 0.15 nats of the
best one.
A coarse grid can tell a method that works over a factor of ten in the
learning rate from one that works at a single grid point, and in practice
that matters as much as the best loss. Claims that one optimizer beats
another depend strongly on how much effort went into tuning each
:cite:`Schmidt.Schneider.Hennig.2021`, so every arm gets a grid of its
own.

:begin_tab:`pytorch`
`torch.manual_seed` also fixes the order of the minibatches, so the runs of
one call of `lr_sweep` share their data as well as their initial weights.
:end_tab:

:begin_tab:`jax`
The seed fixes the initial weights only. Each run of `lr_sweep` draws its
own minibatches.
:end_tab:

The chapter asks how fast a method lowers the loss, so the measure is the
training loss, averaged over the last 100 steps by `final_loss`. On
`TinyLM` this is also a held-out loss: every minibatch is text the model
has not seen, so its loss measures prediction and not memory. Where a
section needs the loss of one fixed set of weights, for instance in the
middle of a run, it uses the validation loss, the logarithm of
`lm_perplexity` on the validation split. `FashionCNN` passes over its
60,000 training images eight and a half times in 2,000 steps. Its training
loss says how well the optimizer optimizes and its test accuracy how well
the result generalizes, and we print both.

### The Spread between Two Runs

What remains is the rule from the beginning of this section, and the
number it needs: how much do two runs of one setting differ? This number
is rarely reported, although it is cheap to measure. `repeat` trains one
setting once per seed, each time with new initial weights and new
minibatches, and returns the final training losses and the evaluations.

We measure the two settings that later sections compare against. For
`FashionCNN` it is the SGD run from above. For `TinyLM` it is AdamW at a
learning rate of 0.003 and without weight decay, the optimizer of the
recipes in :numref:`sec_training_recipes`. This chapter has not explained
AdamW yet. Here we call it as we have called it so far, from the library.

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

Five runs of `TinyLM` end within about a hundredth of a nat of each other,
in the training loss and in the validation loss. `FashionCNN` is somewhat
less repeatable: its training losses lie within about 0.02 and its test
accuracies within about a point. These ranges are the yardstick for the
rest of the chapter. Five runs give a rough range and not a distribution,
so we ask for a margin and call a difference a result only if it is at
least three times the range printed here. On `TinyLM` that is 0.05 nats,
to be on the safe side, and on `FashionCNN` a few points of test accuracy
between two single runs. The 0.15 nats of the usable range are more than
ten times the printed range. Smaller differences we report as ties, and a
section that has to resolve one repeats its runs and prints them.

One detail of the output may puzzle you. The validation loss is lower than
the training loss in every run, by 0.02 to 0.03 nats. Memorization would
produce the opposite sign. Most of the gap is the text itself: the
validation articles are somewhat easier to predict than an average
training window (Exercise :exref:`ex_test-models-held-out`).

Now set the first row of the output beside the baseline. Plain SGD ended
near 2.1 nats, and AdamW ends near 1.59, half a nat lower after the same
2,000 steps. Against a range of a hundredth, nobody will call that a tie.
But by our own protocol it is not yet a comparison. Three rates a factor
of five apart do not tune SGD, and AdamW differs from plain SGD in two
ways at once: it averages its gradients over steps, and it gives every
parameter a step size of its own. :numref:`sec_momentum` and
:numref:`sec_adam` take the two apart, one at a time and by the
protocol.

## Summary

From this section on, a claim about an optimizer is a difference between
training runs of a small network. The networks are `TinyLM`, a transformer
of two blocks and 470,784 parameters that predicts the next byte of
Wikipedia text, and `FashionCNN`, a convolutional classifier of 421,642
parameters. A run takes 2,000 steps and is read through its training loss.
Under plain minibatch SGD on `TinyLM`, a learning rate that was too low
gave a smooth and slow curve, a rate near the largest stable one gave
early spikes from which the run recovered, and a rate that was too high
gave an explosion within a few steps. A comparison tunes the learning rate of every arm on a grid,
starts all arms from the same weights, and reports the usable range of
rates next to the best one. Five runs of one setting ended within about a
hundredth of a nat on `TinyLM` and within about a point of test accuracy
on `FashionCNN`, and a difference between two settings counts as a result
only if it is at least three times as large.

## Exercises

1. **The loss before training.** The curves of the first run start above
   the uniform guess $\ln 256$.
   :exlabel:`ex_test-models-initial-loss`
    1. Let the logits $z_1, \ldots, z_V$ of a prediction be independent
       draws from a normal distribution with mean zero and variance $s^2$,
       and let the target be independent of them. Show that the expected
       cross-entropy is about $\ln V + s^2/2$ when $V$ is large. (A sum
       of many terms $e^{z_i}$ is close to $V$ times their mean
       $e^{s^2/2}$.)
    1. The head of `TinyLM` receives the output of a layer normalization,
       a vector of 128 entries with mean zero and mean square one, and its
       weights and biases are uniform on $\pm 1/\sqrt{128}$. Show that each
       logit has variance $129/384 \approx 1/3$, so that the expected loss
       of the untrained model is about $\ln 256 + 1/6 \approx 5.71$, and
       compare with the start of the curves in the first plot.
    1. What is the loss of the untrained model if the weights and the bias
       of the head start at zero? Which parameters of the model then
       receive a nonzero gradient in the first step?
1. [code] **Diagnosing a run from its curve.** A classifier with ten
   classes is trained by minibatch SGD at a constant learning rate.
   :exlabel:`ex_test-models-diagnosis`
    1. For each of the following loss curves, say what is most likely
       wrong and what you would change first: (a) the loss falls smoothly
       from 2.3 to 1.2 and is still falling at the same pace when the run
       ends; (b) the loss falls to 0.4 within the first tenth of the run
       and then fluctuates around that level without a trend; (c) the loss
       reads 2.3, 1.9, 4.6, 31, `nan`; (d) the loss reaches 2.3026 within a
       few steps and stays there to four digits on every minibatch.
    1. For curve (b), which single additional run tells you whether the
       level is the noise ball of :numref:`sec_sgd` or the best that this
       model can do?
    1. Train `FashionCNN` with plain SGD at the learning rates 0.003, 0.03,
       0.3 and 1, and plot the loss of every step on a logarithmic step
       axis, with a line at $\ln 10$. Which of the four descriptions fits
       each run? If a run matches (d), print its logits for a few images
       and the fraction of its ReLU units that are zero for every image of
       a minibatch, and explain why the loss does not move.
1. [code] **The edge of divergence.** Between the learning rates 0.5 and
   2.5 lies the largest one that plain SGD tolerates on `TinyLM`. Train at
   0.7, 1, 1.4 and 2 with five seeds each (`lr_sweep` with
   `verbose=False`), and record for every run whether it diverged and
   where it ended.
    1. Is there a sharp ceiling, as on the valley, or a range of rates in
       which the outcome depends on the seed?
    1. At which rate is the mean final loss of the runs that survived the
       lowest? What does your answer imply for a statement such as "SGD at
       its best learning rate reaches a loss of 2.0"?
1. [code] **Repeats and the range.** ● The range of five runs is a rough
   measure of spread.
    1. For $n$ independent draws from a normal distribution with standard
       deviation $\sigma$, the expected range is about $2.3\,\sigma$ for
       $n = 5$ and $3.7\,\sigma$ for $n = 20$. Call `repeat` with 20 seeds
       for `TinyLM`, and estimate $\sigma$ of the final training loss once
       from the standard deviation of the 20 values and once from their
       range. Do the two estimates agree?
    1. Two settings are each run $k$ times, and the difference of their
       mean final losses has standard error $\sigma\sqrt{2/k}$. With your
       estimate of $\sigma$, how large must $k$ be for a true difference of
       0.005 nats to be twice the standard error?
    1. Train the setting five times from the same initial weights, with
       different minibatches in every run. Is the range smaller than that
       of five runs that differ in both? How much does a comparison gain by
       starting all arms from the same weights?
1. [code] **Training loss as a held-out loss.** The comparisons of this
   chapter read the training loss of `TinyLM` as a loss on unseen text.
   :exlabel:`ex_test-models-held-out`
    1. In the last cell the validation loss is below the training loss in
       every run. Either `final_loss` is too high, because it averages over
       100 steps during which the model still improves, or the validation
       text is easier to predict. Devise a measurement that separates the
       two, and carry it out.
    1. `WikiText103(batch_size=64, num_steps=64, num_train=100000)` keeps
       the first 100,000 bytes of the training text, so that 2,000 steps
       use every byte as a target about 80 times. Train `TinyLM` on it with
       the AdamW setting of this section and measure the validation loss
       every 250 steps (`lr_sweep` with `every=250`). From which step on do
       training and validation loss part, and how far apart do they end?
       Which of the two would now mislead a comparison of optimizers?
1. [code] **Count models as landmarks.** A loss says more next to the loss
   of a model that we understand.
    1. From the first ten million training bytes, compute the entropy of
       the byte frequencies in nats, the loss of the best model that
       ignores the context.
    1. Count the pairs of consecutive bytes in the same text and compute,
       on the validation split, the mean loss of the model that predicts a
       byte from the one before it, with add-one smoothing
       (:numref:`sec_language-model`).
    1. Place the final losses of this section, about 2.5 and 2.1 under SGD
       and 1.59 under AdamW, among $\ln 256$ and these two numbers. How
       much context can each of the three trained models be using?

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.4]{.kicker}

Two Test Models<br>**a transformer and a CNN · minibatch SGD at three rates · the spread between two runs**
:::
:::

::: {.slide title="A Network Has No Formula, and Two Runs of One Setting Differ"}
[The problem]{.kicker}

On the valley $0.1 x_1^2 + 2 x_2^2$ every answer could be computed:

| question | valley | network |
|:--|:--|:--|
| largest stable learning rate | $2/4 = 0.5$ | unknown |
| steps to 1% at the best rate | 47 | unknown |
| where SGD comes to rest | the noise ball | unknown |

::: {.d2l-note .warn}
We must train and look. But a new optimizer ends 0.02 nats below the old
one: is that the optimizer, or the random seed?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- describe the two test models of this chapter and count their
  parameters;
- predict the loss of a model before training;
- tell from a loss curve whether the learning rate is too low, too high,
  or about right;
- set up a comparison of two optimizers that is fair to both;
- decide whether a difference between two runs is a result.

**Running example:** `TinyLM`, a small language model on Wikipedia bytes.
**The idea:** a difference between two settings counts only if it exceeds
the difference between two runs of one setting.
:::

::: {.slide title="From Here On the Chapter Measures on Two Small Networks"}
[Where this fits]{.kicker}

| movement | sections |
|---|---|
| two difficulties, on a valley we can solve | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| which direction to step in, on two small networks | **Two Test Models** · Momentum · Adam · AdamW · Muon |
| how the step changes over a run, with batch size and width | Schedules · Batch Size · Scaling Up · Practice |

- **Uses:** minibatch SGD, the ceiling $\eta < 2/L$ on the learning rate
  $\eta$ ($L$: largest curvature), the noise ball.
- **Skips:** how a transformer works; an optimizer sees only parameters
  and their gradients.
:::

::: {.slide title="A Run Reads 3% of the Text, So It Cannot Memorize It"}
[TinyLM]{.kicker}

| quantity | value |
|:--|--:|
| corpus: WikiText-103, first training file | 270 million bytes |
| token | one UTF-8 byte, 256 values |
| minibatch | 64 windows of 64 bytes |
| run | 2,000 steps |
| predictions per run | $2{,}000 \times 64 \times 64 \approx 8.2$ million |
| share of the corpus read | 3% |
| on *The Time Machine* (170,000 characters) | each character a target 47 times |

::: {.d2l-note}
Windows start at random offsets, so almost no text is seen twice: the
training loss of a new minibatch is a loss on unseen text.
:::
:::

::: {.slide title="TinyLM Has 470,784 Parameters, 90% of Them in Matrices"}
[TinyLM]{.kicker}

Byte and position embeddings → 2 transformer blocks of width 128 (attention
and a two-layer MLP, each normalized, each with a residual connection) →
linear head with 256 logits for the next byte.

| kind | what | parameters | share |
|:--|:--|--:|--:|
| embeddings | byte and position tables | 40,960 | 9% |
| matrices | weights of the linear layers | 425,984 | 90% |
| vectors | biases, normalization scales and shifts | 3,840 | 1% |

::: {.d2l-note}
Later methods treat the three kinds differently. One block has
$128 \cdot (384 + 128 + 512 + 512) = 196{,}608$ matrix entries.
:::
:::

::: {.slide title="FashionCNN Has 421,642 Parameters, 95% in One Matrix"}
[FashionCNN]{.kicker}

Fashion-MNIST, $28 \times 28$ grayscale images, 10 classes, minibatches of
256 images.

| layer | output | parameters |
|:--|:--|--:|
| $3 \times 3$ convolution, ReLU, $2 \times 2$ max-pooling | $32 \times 14 \times 14$ | 320 |
| $3 \times 3$ convolution, ReLU, $2 \times 2$ max-pooling | $64 \times 7 \times 7$ | 18,496 |
| fully connected, ReLU | 128 | 401,536 |
| fully connected | 10 | 1,290 |

- 2,000 steps are 512,000 images, 8.5 passes over the 60,000.
- Training loss measures optimization, test accuracy generalization.
:::

::: {.slide title="One Loop Trains Both Models, and It Counts Steps"}
[Training and reading a run]{.kicker}

Each step: draw a minibatch, compute the mean cross-entropy of the logits,
take one optimizer step, record the loss.

| function | returns |
|:--|:--|
| `train_steps` | the loss of every step |
| `final_loss` | mean of the last 100 losses; infinity if the run diverged |
| `lm_perplexity` | $\exp$ of the mean loss per token on held-out text |
| `evaluate_accuracy` | fraction of examples classified correctly |
| `lr_sweep` | one run per learning rate, all from the same initial weights |

::: {.d2l-note}
Steps, not epochs: an epoch of 270 million bytes has no meaning for a run
that reads 3% of them.
:::
:::

::: {.slide title="Check: What Is the Loss of a Model That Knows Nothing?"}
[Check]{.kicker}

The loss is the cross-entropy in nats, $-\ln p(\textrm{target})$.

1. `TinyLM` predicts one of 256 byte values. What loss does a uniform
   guess have?
2. What is the same number for `FashionCNN`?
3. Will an untrained `TinyLM` start below, at, or above that level?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\ln 256 \approx 5.55$ and $\ln 10 \approx 2.30$. Above:
random logits of variance $s^2 \approx 1/3$ prefer some bytes for no
reason, which costs about $s^2/2 = 1/6$ nat. Expect a start near 5.7.
:::
:::
:::

::: {.slide title="Plain SGD at Three Rates Ends at 2.5, at 2.1, and at inf"}
[A first run]{.kicker}

`TinyLM`, minibatch SGD without momentum, constant learning rate, 2,000
steps, one initialization.

@!test-models-a-first-run-with-minibatch-sgd-2

Loss of every step; dotted: the uniform guess $\ln 256$.
:::

::: {.slide title="Each of the Three Curves Is a Diagnosis"}
[A first run]{.kicker}

| rate | curve | diagnosis |
|:--|:--|:--|
| 0.1 | smooth from the first step, still falling at step 2,000, ends near 2.5 | too low: raise it |
| 0.5 | thrown up by as much as a nat in the first 30 steps, recovers, ends near 2.1 | near the largest stable rate: keep it |
| 2.5 | down once, then above its start, then `inf` | above the ceiling $2/L$ ($L$: largest curvature): lower it |

- Useful rates form a narrow band: a factor of 5 down costs a third of a
  nat, a factor of 5 up costs the run.
- Neither stable run has flattened into a noise ball: both are short of
  steps, not stuck in noise.
:::

::: {.slide title="The Baselines: 2.1 Nats per Byte and 90% Test Accuracy"}
[A first run]{.kicker}

`FashionCNN`, the same loop, plain SGD at learning rate 0.3:

@!test-models-a-first-run-with-minibatch-sgd-3

| model | plain SGD, 2,000 steps |
|:--|:--|
| `TinyLM`, rate 0.5 | training loss about 2.1 nats per byte |
| `FashionCNN`, rate 0.3 | training loss about 0.22, test accuracy about 90% |
:::

::: {.slide title="A Fair Comparison Tunes the Learning Rate of Every Arm"}
[The protocol]{.kicker}

| class | meaning | in this chapter |
|:--|:--|:--|
| scientific | what the question is about; each value is an *arm* | the optimizer |
| nuisance | set to its best value separately for each arm | the learning rate |
| fixed | the same in every arm; each limits the claim | model, batch size, 2,000 steps, constant rate |

::: {.d2l-note .warn}
One shared learning rate favors whichever method happens to prefer it.
:::

[Godbole et al., Deep Learning Tuning Playbook, 2023]{.small}
:::

::: {.slide title="Tuned Means the Best Point of a Grid"}
[The protocol]{.kicker}

- **Grid:** learning rates a factor of about 3 apart (2 for a finer
  answer); every run of a comparison starts from the same initial weights.
- **Tuned:** the grid point with the lowest final loss, no more.
- **Usable range:** the rates whose final loss is within 0.15 nats of the
  best one. A coarse grid can tell a factor of 10 from a single point.
- **Loss:** the training loss, mean of the last 100 steps. On `TinyLM` it
  is a loss on unseen text.
:::

::: {.slide title="Five Runs of One Setting End Within About 0.01 Nats"}
[The spread]{.kicker}

New initial weights and new minibatches in each run. `TinyLM`: AdamW,
rate 0.003, no weight decay. `FashionCNN`: SGD, rate 0.3.

@!test-models-comparing-two-runs-3

::: {.d2l-note}
**Yardstick:** a difference is a result only if it is at least 3 times
the printed range: 0.05 nats on `TinyLM`, a few points of accuracy on
`FashionCNN`.
:::
:::

::: {.slide title="Check: Result or Tie?"}
[Check]{.kicker}

One run per setting, 2,000 steps each.

1. `TinyLM`: optimizer A ends at 1.57, optimizer B at 1.59.
2. `FashionCNN`: 90.4% against 89.6% test accuracy.
3. `TinyLM`: plain SGD at 2.1, AdamW at 1.59.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** 1: a tie, 0.02 is under 0.05. 2: a tie, the range alone is
about a point. 3: half a nat is no tie, but three rates a factor of 5
apart did not tune SGD, so the comparison is not yet fair.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- `TinyLM`: 470,784 parameters, next byte of Wikipedia, 3% of the text
  per run. `FashionCNN`: 421,642 parameters, Fashion-MNIST.
- Before training the loss is near 5.7, just above
  $\ln 256 \approx 5.55$.
- Plain SGD on `TinyLM`: rate 0.1 ends near 2.5, rate 0.5 near 2.1, rate
  2.5 diverges within a few steps.
- Protocol: tune the learning rate of every arm on a grid, same initial
  weights, report the usable range.
- Five runs agree within about 0.01 nats; a result is at least 3 times
  the range.

**Next:** momentum, the first of the two things that AdamW does
differently from plain SGD.
:::
