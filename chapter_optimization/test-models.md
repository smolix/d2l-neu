# Two Test Models
:label:`sec_test_models`

Until now, almost every claim of this chapter could be checked with a
pencil. The valley has two curvatures, and from them we computed the
largest stable learning rate, the number of steps that gradient descent
needs, and the size of the cloud in which SGD comes to rest. A network
offers nothing of the kind. Does momentum help on a real model? Why are
transformers trained with Adam, when SGD served image classifiers for a
decade? No formula answers such questions. We have to train the network
and look.

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
parameters. Language models trained in earnest are a thousand to a million
times larger, and we shall have to say what models as small as ours can
show and what they cannot. `TinyLM` is a transformer, an architecture that
:numref:`chap_transformers` explains. We use it two chapters early because
an optimizer does not need to know what attention does: it sees parameters
and their gradients, and nothing else. We train both models with the plain
minibatch SGD of :numref:`sec_sgd`, as a baseline, and then train one
setting of each model five times to measure the spread. Later sections and
chapters reuse the models and the functions defined here.

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

`TinyLM` predicts the next byte of a text. Its tokens are the UTF-8 bytes
of the text, as the byte-level tokenizer of :numref:`sec_text-sequence`
produces them before any merge, so the vocabulary has 256 entries and
needs no training of its own. A training example is a *window*, a stretch
of 64 consecutive bytes, and at each of its positions the model predicts
the byte that comes next.

How much text does such a model need? One of our runs takes 2,000 steps on
minibatches of 64 windows, about 8 million predictions. On *The Time
Machine* with its 170,000 characters (:numref:`sec_text-sequence`), each
character would be a target almost 50 times, and a low training loss
would no longer tell us whether the model predicts English or recites
Wells. We therefore use a corpus that a run cannot exhaust: WikiText-103
:cite:`Merity.Xiong.Bradbury.ea.2016`, about 100 million words of Wikipedia
articles kept as raw text, with case, digits, and punctuation. Training
windows start at random offsets in the first of its two training files,
about 270 million bytes, so a run reads about 3% of the file and rarely
sees the same passage twice. Every minibatch is new text, and one number,
the training loss, answers two questions at once: how fast the optimizer
descends, and how well the model predicts. Validation windows are
consecutive pieces of the official validation split, about 1.1 million
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

`WikiText103` holds both splits in memory and gathers each minibatch on
demand: 64 windows, each at an independent random offset. A pass over the
training data is about 66,000 such minibatches, over thirty times as many
as a run uses. We load the data and decode its first 300 bytes.

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

What this chapter needs from `TinyLM` are the names and shapes of its
parameters, and for those a short tour is enough. Let us follow one byte
on its way. The model looks up a learned vector of 128 numbers for the
byte and adds a second learned vector for its position in the window. The
sum passes through two identical *blocks*, then through a layer
normalization (:numref:`subsec_layer-normalization-in-bn`) and a linear
layer, the *head*, which returns 256 logits for the next byte. A block
consists of an attention layer, in which a position reads from the
positions up to its own, and an MLP with one hidden layer of 512 units,
which processes every position by itself. Each of the two sits behind a
layer normalization and has a residual connection around it.

Four weight matrices do the work of a block, and later sections call them
by name. `qkv` maps the 128 inputs of the attention layer to
$3 \times 128 = 384$ numbers, and `proj` maps the 128 numbers that
attention returns back into the block. The two matrices of the MLP lead
from 128 to 512 and from 512 back to 128.

Every linear layer draws its weights and biases uniformly from
$\pm 1/\sqrt{\textrm{fan-in}}$, where the *fan-in* of a layer is the
number of its inputs, and every embedding row is drawn from a standard
normal distribution. The initialization belongs to the model as much as
the layers do: it fixes the loss at which every run starts, a number that
we predict below.

You can read the class as a list of the layers that hold parameters. The
method `attention` is one library call wrapped in reshapes, and it will
make sense once you have read the transformer chapter. Nothing in this
chapter depends on it.

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

To an optimizer the parameters of `TinyLM` are of three kinds.
*Embeddings* are tables, and the row of a byte receives a gradient only
when that byte occurs in the minibatch. *Matrices* are the
two-dimensional weights of the linear layers. *Vectors* are the biases and
the scales and shifts of the layer normalizations. Later sections treat
the three kinds differently (weight decay in :numref:`sec_adamw`, for
instance, applies to the matrices only), so we count them once here. The
four matrices of a block have
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

:begin_tab:`pytorch`
In this output the two matrices of the MLP are `mlp.0.weight` and
`mlp.2.weight`, and a weight of shape `(384, 128)` maps 128 inputs to 384
outputs.
:end_tab:

:begin_tab:`jax`
In this output the two matrices of the MLP are `mlp1.kernel` and
`mlp2.kernel`, and a kernel of shape `(128, 384)` maps 128 inputs to 384
outputs.
:end_tab:

## The Image Classifier FashionCNN
:label:`subsec_fashioncnn`

What holds for a transformer need not hold for a convolutional network, so
the second model is one. `FashionCNN` classifies the $28 \times 28$
grayscale images of Fashion-MNIST (:numref:`sec_fashion_mnist`). Two stages
of a $3 \times 3$ convolution, a ReLU, and $2 \times 2$ max-pooling, with
32 and 64 channels, reduce each image to $64 \times 7 \times 7$ features.
Two fully connected layers map them to 128 hidden units and then to the ten
classes. The initialization follows the rule of `TinyLM`, uniform on
$\pm 1/\sqrt{\textrm{fan-in}}$, where a unit of a $3 \times 3$ convolution
has nine inputs for every input channel.

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

Most of this model is one matrix. The first fully connected layer has
$3{,}136 \times 128$ weights, about 0.4 million, and the two convolutions
are small beside it.

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
of them sit in that one matrix. We call the model convolutional, and its
features are, but by parameter count an optimizer is mostly training one
fully connected layer.

The data are the minibatches of Fashion-MNIST that the Basics part used,
256 images each. A run of 2,000 steps passes over the 60,000 training
images eight and a half times. Unlike `TinyLM`, this model sees its data
again and again, so its training loss does not say how it will do on new
images, and we judge it by its test accuracy as well. `evaluate_accuracy`
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

Both models map a minibatch to logits and are trained on the cross-entropy
of those logits, so one loop serves both. We count its length in steps. An
epoch has no meaning for a run that reads 3% of its text, and the
predictions of the previous sections were counted in steps too.
:numref:`tab_test-models-helpers` lists the loop and the functions that
read off what a run did. All are saved in `d2l`, and the code follows in
the order of the table. Only two of them need your attention:
`train_steps` is the loop, and `lr_sweep` calls it once per learning rate.
The others are a few lines each and can be taken on trust.

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

The loop has more arguments than this section needs. A label of $-100$
marks a position that adds no loss, which the masked language model of
:numref:`sec_transformer` will rely on, and `stop` ends a run before its
last step. Both can be ignored here.

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
builds its optimizer for a given learning rate, and trains once per rate,
each time from the same initial weights. Given a function `evaluate`, it
also evaluates each model, at the end of training or every `every` steps,
and prints the last value under the name `metric`.

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
`TinyLM` has a ceiling too, but we cannot compute it. So we do the crude
thing and bracket it: we start at 0.1 and multiply by five until a run
diverges. That takes three rates, 0.1, 0.5 and 2.5.

One number we can predict, the loss at which the curves start. A model that
knows nothing about its text gives each of the 256 byte values the same
probability, at a loss of $\ln 256 \approx 5.55$ nats per byte. The plot
marks this *uniform guess* with a dotted line.

Further down we shall want two more landmarks, because a loss says more
next to the loss of a model that we understand. A model that knows how
frequent each byte is, and ignores the context altogether, scores about
3.2 nats on this text. A table of byte pairs, which predicts a byte from
the one before it and from nothing else, scores about 2.5. Neither needs
training, only counting, and Exercise :exref:`ex_test-models-landmarks`
does the counting. The plot does not draw these two, so keep the numbers
in mind as you read it.

The plot shows the loss of every step, on a logarithmic step axis because
the three runs part ways within the first ten steps. In the code, `make_lm`
and `sgd` build the model and the optimizer in the form that `lr_sweep`
expects. `new_seed` returns a new seed at every call, so each execution of
this notebook starts from weights of its own.

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
more than a nat, the second throws it above its starting value, and within
a handful of steps the loss has left the plot. The run ends as `inf`. On
the valley, a rate above $2/L$ multiplied the error along the steepest
direction at every step, from the first one on (:numref:`sec_gd`). Here
the first step is a good one and the second is fatal. A quadratic has
the same curvature everywhere, but the curvature of a network depends on
where the run is (:numref:`subsec_gd-edge-of-stability`), and the ceiling
moves with it. A rate need not be too high everywhere to end a run. One
place is enough.

At $\eta = 0.1$ the rate is too low. The curve is smooth from the first
step on, and after 2,000 steps it is still falling, at about 2.5. That is
the level of the pair table: in 2,000 steps this run has learned about as
much as counting pairs of bytes would have told it. Nothing is wrong with
this run except that it is slow.

At $\eta = 0.5$ the loss falls much faster at first, and during the first
thirty steps it is thrown back up several times, once by more than a nat.
Minibatch noise does not do that: at $\eta = 0.1$ no step of the whole
run raises the loss by more than a quarter of a nat. The steps that threw
the loss up overshot, and for those few steps the rate was too high for
the place the run was in. Each time the run recovers, and after about the
thirtieth step it is not thrown again.

For several hundred steps the two stable runs then travel side by side,
the faster one a little lower, both creeping toward the level of the pair
table. The run at 0.5 gets below that level before step 1,000, and from
there it pulls away, to end near 2.1, a third of a nat below the other.
No model that looks at the previous byte alone can do better than the
pair table, so this run must have begun to use more of its context.

The rate 0.5 can throw the loss up but not out, and it ends lowest. It is
the best of our three rates, and we should not claim more for it. The
largest rate that the run tolerates lies somewhere between 0.5 and 2.5,
and a factor of five is a wide gap. Exercise :exref:`ex_test-models-edge`
narrows it.

Now look at the right end of the plot. Neither stable run has reached the
state in which :numref:`sec_sgd` left SGD, a loss that has stopped falling
and only fluctuates in its noise ball. Between steps 1,000 and 2,000 the
run at 0.5 loses another fifth of a nat or more, and the run at 0.1 about
0.05. These runs are short of steps, not stuck in noise. The loss of a
single step, however, scatters over a band about a tenth of a nat wide,
which is more than the slow run gains in a thousand steps. That is why
`final_loss` averages the last hundred steps.

`FashionCNN` gets a single run, at a rate of 0.3, and we print its test
accuracy at the end. Exercise :exref:`ex_test-models-diagnosis` tries the
rates around it.

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

The first run was already a comparison, of one optimizer with itself, and
the learning rate decided it. Five times below the best of our three rates
costs a third of a nat, and five times above it costs the run. Before we
compare two different optimizers we have to settle what we measure, how
much two runs of the *same* setting differ, and how each side of a
comparison comes by its learning rate. Every comparison of optimizers in
this chapter follows the answers given here.

The first is quickly said. The chapter asks how fast a method lowers the
loss, so the measure is the training loss at the end of a run, as
`final_loss` reports it. For `FashionCNN` we print the test accuracy next
to it. One thing `final_loss` is not: the loss of a particular set of
weights, since the weights keep changing during the hundred steps of the
average. Where a section needs the loss of one fixed set of weights, at a
checkpoint in the middle of a run for instance, it evaluates them on the
validation split. The *validation loss* of `TinyLM` is the logarithm of
`lm_perplexity` there.

### The Spread between Two Runs

Now for the rule from the beginning of this section, and the number it
needs: how much do two runs of one setting differ? This number is rarely
reported, although it is cheap to measure. `repeat` trains one setting
once per seed, each time with new initial weights and new minibatches, and
returns the final training losses and the evaluations.

We measure one setting of each model, the one that later sections compare
against. For `FashionCNN` it is the SGD run from above. For `TinyLM` it is
not SGD but AdamW, the optimizer of the compact modern recipe in
:numref:`sec_training_recipes`, here at a constant learning rate of 0.003
and without weight decay. This chapter has not explained AdamW yet. Here
we call it as we have called it so far, from the library.

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
in the training loss and in the validation loss. The training losses of
`FashionCNN` lie within about 0.02, and its test accuracies within about a
point.

These ranges are our yardstick, and we should be clear about how rough it
is. Five runs do not give a distribution. They give one range, and five
more runs would give another, perhaps half as large, perhaps twice. So we
ask for a margin: a difference between two single runs is a result only if
it is at least three times the range printed here. The factor of three is
a convention, and Exercise :exref:`ex_test-models-range` looks at the
statistics behind it. On `TinyLM`, three times the range of the training
losses is 0.03 to 0.04 nats, and we round up to 0.05. A smaller difference
we report as a tie, and a section that has to resolve a tie repeats its
runs and prints them. On `FashionCNN`, three times 0.02 is 0.06 in the
training loss, and three times a point is three points of test accuracy.
A single point of test accuracy between two runs of `FashionCNN`
therefore shows nothing, however good it looks in a table.

The yardstick also belongs to the settings that produced it: an optimizer
at a well-chosen learning rate, read at step 2,000. A run close to the
largest rate that it tolerates, or a loss read early in a run, can spread
more, and a section that compares such settings prints a range of its
own.

One detail of the output may puzzle you. The validation loss is lower than
the training loss in every run, by 0.02 to 0.03 nats. Had the model
memorized its training text, the sign would be the other way round. But
the two numbers are not measured alike. The training loss is a mean over a
hundred steps, during which the model was still improving, on windows
drawn at random. The validation loss belongs to the final weights and to
one fixed text. Either difference could produce a gap of this size, and
Exercise :exref:`ex_test-models-held-out` asks you to find out which one
does.

### The Tuning of the Learning Rate

The measure and the yardstick tell us when two numbers differ. They do not
yet make a comparison fair. In the runs above SGD trained at a rate of 0.5
and AdamW at 0.003. At AdamW's rate, SGD would take steps thirty times
shorter than in the run that we called too slow. Two optimizers compared
at one shared learning rate are ranked mostly by which of them happens to
like that rate.

The Deep Learning Tuning Playbook :cite:`Godbole.Dahl.Gilmer.ea.2023` has
names for the roles that hyperparameters play in such an experiment. The
*scientific* hyperparameter is the one the question is about, here the
optimizer; we call each of its values an *arm* of the comparison. A
*nuisance* hyperparameter is one that the question is not about, but that
every arm needs at its own best value before the comparison is fair. The
learning rate is the standard case.

*Fixed* hyperparameters are the same in every arm, and each of them limits
the claim. Ours, unless a section says otherwise, are the model, the batch
size, the 2,000 steps, and a constant learning rate without clipping or
weight decay. The first and the third are the bold ones.

What can a model of half a million parameters, trained for 2,000 steps,
tell us about one a thousand times larger that trains for weeks? Less than
we would like. We shall trust it for a mechanism, for the order of two
methods that differ by a wide margin, and for the width of the range of
learning rates that a method accepts. That trust is an assumption. An
effect that appears only at scale, or late in a long run, a small model
cannot show at all. What changes with size needs an experiment of its own,
and :numref:`sec_scaling` makes one, on the drift of the best learning
rate as a model grows wider.

We tune the learning rate of an arm with `lr_sweep`, on a grid of rates a
factor of about three apart, or two where a section needs a finer answer.
All runs of a comparison start from the same initial weights, so that the
arms differ in their optimizer and not also in their starting point.
*Tuned* means no more than the best point of such a grid.

:begin_tab:`pytorch`
`torch.manual_seed` also fixes the order of the minibatches, so the runs of
one call of `lr_sweep` share their data as well as their initial weights.
:end_tab:

:begin_tab:`jax`
The seed fixes the initial weights only. Each run of `lr_sweep` draws its
own minibatches.
:end_tab:

A grid cannot place the best rate more finely than its spacing, and the
best loss is uncertain to the same degree. What a coarse grid does measure
reliably is how many of its points come close to the best one. We
therefore also report the *usable range* of a method, the rates whose
final loss is within 0.15 nats of the best. That margin is more than ten
times the range printed above, so the seed does not decide which rates
belong to it. Our SGD sweep makes a small example. Its best rate is 0.5,
and that rate is the whole usable range, since the run at 0.1 ends 0.35
nats higher and the run at 2.5 diverges. A method that works at a single grid
point is harder to live with than one that works over a factor of ten in
the learning rate, and in practice that matters as much as the best loss.
Claims that one optimizer beats another depend strongly on how much effort
went into tuning each :cite:`Schmidt.Schneider.Hennig.2021`, so every arm
gets a grid of its own.

Now set the AdamW runs beside the baseline. Plain SGD ended near 2.1 nats,
and AdamW ends near 1.59, half a nat lower after the same 2,000 steps.
Against a range of a hundredth, nobody will call that a tie, even if the
SGD run should spread several times more than the AdamW runs do. But by
our own protocol it is not yet a comparison. Three rates a factor of five
apart do not tune SGD. And AdamW differs from plain SGD in two ways at
once: it averages its gradients over steps, and it gives every parameter a
step size of its own. The next two sections take up the two changes one
at a time. :numref:`sec_momentum` adds the averaging to SGD.
:numref:`sec_adam` adds the step sizes, and then lets SGD with momentum
and Adam meet on `TinyLM`, each with a grid of its own.

## Summary

From this section on, a claim about an optimizer is a difference between
training runs of a small network. The networks are `TinyLM`, a transformer
of two blocks and 470,784 parameters that predicts the next byte of
Wikipedia text, and `FashionCNN`, a convolutional classifier of 421,642
parameters. A run takes 2,000 steps and is read through its training loss,
which on `TinyLM` is a loss on unseen text. Under plain minibatch SGD on
`TinyLM`, a learning rate that was too low gave a smooth, slow curve that
ended at the level of a table of byte pairs, a rate five times higher
gave early spikes, a recovery, and the lowest loss of the three, and a
rate five times higher again diverged within a few steps. A comparison
tunes the learning rate of every arm on a grid, starts all arms from the
same weights, and reports the usable range of rates next to the best one.
Five runs of one setting ended within about a hundredth of a nat on
`TinyLM` and within about a point of test accuracy on `FashionCNN`, and a
difference between two single runs counts as a result only if it is at
least three times that range, which we round to 0.05 nats on `TinyLM`.

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
1. [code] **The edge of divergence.** The first run left the largest
   learning rate that plain SGD tolerates on `TinyLM` somewhere between
   0.5 and 2.5. Train at 0.7, 1, 1.4 and 2 with five seeds each
   (`lr_sweep` with `verbose=False`), and record for every run whether it
   diverged and where it ended.
   :exlabel:`ex_test-models-edge`
    1. Is there a sharp ceiling, as on the valley, or a range of rates in
       which the outcome depends on the seed?
    1. At which rate is the mean final loss of the runs that survived the
       lowest? What does your answer imply for a statement such as "SGD at
       its best learning rate reaches a loss of 2.0"?
    1. At which step do the runs that diverge leave the plot? Does a run
       that has survived its first hundred steps diverge later? What does
       this say about where on its way a run is most fragile?
1. [code] **Repeats and the range.** ● The range of five runs is a rough
   measure of spread.
   :exlabel:`ex_test-models-range`
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
       different minibatches in every run. (Where the seed of `lr_sweep`
       also fixes the minibatches, build the model yourself, reseed, and
       call `train_steps`.) Is the range smaller than that of five runs
       that differ in both? How much does a comparison gain by starting
       all arms from the same weights?
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
1. [code] **Count models as landmarks.** The text quoted two losses that
   need no training: about 3.2 nats for the byte frequencies and about 2.5
   for a table of byte pairs.
   :exlabel:`ex_test-models-landmarks`
    1. From the first ten million training bytes, compute the entropy of
       the byte frequencies in nats, the loss of the best model that
       ignores the context.
    1. Count the pairs of consecutive bytes in the same text and compute,
       on the validation split, the mean loss of the model that predicts a
       byte from the one before it, with add-one smoothing
       (:numref:`sec_language-model`).
    1. Add a third landmark, a table that predicts a byte from the two
       before it. Then place the final losses of this section, about 2.5
       and 2.1 under SGD and 1.59 under AdamW, among $\ln 256$ and the
       three landmarks. How many bytes of context must each of the three
       trained models be using at least?
    1. A table that conditions on the five previous bytes has
       $256^6 \approx 3 \times 10^{14}$ entries. Why can the series of
       landmarks not be continued that far with the text we have, and
       what does `TinyLM` do instead of counting?

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
| where SGD at a constant rate ends | in a cloud of known size | unknown |

::: {.d2l-note .warn}
We must train and look. But a new optimizer ends at a loss 0.02 below the
old one: is that the optimizer, or the random seed?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- describe the two test models of this chapter and count their
  parameters;
- predict the loss of a model before training, and place a loss among
  landmarks that need no training;
- tell from a loss curve whether the learning rate is too low, too high,
  or workable;
- decide whether a difference between two runs is a result;
- set up a comparison of two optimizers that is fair to both.

**Running example:** `TinyLM`, a small language model on Wikipedia bytes.
**The idea:** a difference between two settings counts only if it exceeds
the difference between two runs of one setting.
:::

::: {.slide title="From Here On the Chapter Measures on Two Small Networks"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch Gradient Descent |
| **2 · Direction:** which way to step | **Two Test Models** · Momentum · Adam · AdamW · Muon |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice |
:::

::: {.col .narrow}
- **Uses:** minibatch SGD; the ceiling $\eta < 2/L$ on the learning rate
  $\eta$ ($L$: largest curvature); the *noise ball*, the cloud in which
  SGD at a constant rate ends.
- **Skips:** how a transformer works. An optimizer sees only parameters
  and their gradients.
:::
:::
:::

::: {.slide title="A Run Reads 3% of the Text, So It Cannot Memorize It"}
[TinyLM]{.kicker}

| quantity | value |
|:--|--:|
| corpus: WikiText-103, first training file | 270 million bytes |
| token | one UTF-8 byte, 256 values |
| minibatch | 64 windows of 64 consecutive bytes |
| run | 2,000 steps |
| predictions per run | $2{,}000 \times 64 \times 64 \approx 8.2$ million |
| share of the corpus read | 3% |
| on *The Time Machine* (170,000 characters) | each character a target 47 times |

::: {.d2l-note}
Windows start at random offsets, so almost no text is seen twice: the
training loss of a new minibatch is a loss on unseen text.
:::

[Merity et al., 2016]{.small}
:::

::: {.slide title="TinyLM Has 470,784 Parameters, 90% of Them in Matrices"}
[TinyLM]{.kicker}

Byte and position embeddings → 2 transformer blocks of width 128 (attention
and an MLP, each normalized, each with a residual connection) → linear
head with 256 logits for the next byte.

| kind | what | parameters | share |
|:--|:--|--:|--:|
| embeddings | byte and position tables | 40,960 | 9% |
| matrices | weights of the linear layers | 425,984 | 90% |
| vectors | biases, normalization scales and shifts | 3,840 | 1% |

::: {.d2l-note}
Four matrices per block: `qkv` (128 → 384) and `proj` (128 → 128) in
attention, 128 → 512 → 128 in the MLP; 196,608 entries. Weight decay will
act on the matrices only.
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

::: {.d2l-note}
The features are convolutional, but by parameter count an optimizer is
mostly training one fully connected layer.
:::
:::

::: {.slide title="A Run Is 2,000 Steps, Read by the Mean of Its Last 100 Losses"}
[Training and reading a run]{.kicker}

Loss of a step: the cross-entropy $-\ln p(\textrm{target})$ on its
minibatch, in nats.

| | `TinyLM` | `FashionCNN` |
|:--|:--|:--|
| minibatch | 64 windows of 64 bytes | 256 images |
| 2,000 steps read | 3% of the text, once | every image 8.5 times |
| the training loss is | a loss on unseen text | a loss on images seen before |
| also reported | validation loss | test accuracy |

::: {.d2l-note}
On `TinyLM` the loss of one step scatters by about 0.1 nats. *Final loss*:
the mean of the last 100 steps, infinity if the run diverged.
:::
:::

::: {.slide title="Check: What Is the Loss of a Model That Knows Nothing?"}
[Check]{.kicker}

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

::: {.slide title="Three Landmarks Need No Training: 5.55, 3.2 and 2.5 Nats"}
[A first run]{.kicker}

A loss says more next to the loss of a model that we understand.

| model | what it knows | loss, nats per byte |
|:--|:--|--:|
| uniform guess | nothing | $\ln 256 \approx 5.55$ |
| byte frequencies | how common each byte is | about 3.2 |
| table of byte pairs | the byte before | about 2.5 |

::: {.d2l-note}
The last two are counts on the Wikipedia text, not trained models. To get
below 2.5, a model must use more than one byte of context.
:::
:::

::: {.slide title="Plain SGD at Three Rates Ends at 2.5, at 2.1, and at inf"}
[A first run]{.kicker}

`TinyLM`, plain minibatch SGD, constant learning rate, 2,000 steps.

@!test-models-a-first-run-with-minibatch-sgd-2

Dotted: the uniform guess. Not drawn: the landmarks 3.2 and 2.5.
:::

::: {.slide title="Each of the Three Curves Is a Diagnosis"}
[A first run]{.kicker}

| rate | curve | diagnosis |
|:--|:--|:--|
| 0.1 | smooth from the first step; still falling at step 2,000; ends near 2.5, the level of the pair table | too low: raise it |
| 0.5 | thrown up by more than a nat in the first 30 steps; recovers; ends near 2.1 | too high for a few early steps; the best of the three |
| 2.5 | down by more than a nat, then above its start, then `inf` | too high where the first step led: lower it |

- The ceiling lies between 0.5 and 2.5, and on a network it depends on
  where the run is.
- Steps 1,000 to 2,000: another 0.2 nats at rate 0.5, about 0.05 at rate
  0.1. Short of steps, not stuck in noise.
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

::: {.slide title="Five Runs of One Setting of TinyLM End Within About 0.01 Nats"}
[The spread]{.kicker}

New initial weights and new minibatches in each run. `TinyLM`: AdamW (a
library optimizer, explained two decks on), rate 0.003, no weight decay.
`FashionCNN`: SGD, rate 0.3.

@!test-models-comparing-two-runs-3

Validation loss is below training loss in every run, the opposite of
memorization.
:::

::: {.slide title="A Result Is a Difference of at Least Three Times the Range"}
[The spread]{.kicker}

| quantity | range of five runs | three times | we ask for |
|:--|:--|:--|:--|
| `TinyLM`, training loss | about 0.01 nats | 0.03 to 0.04 | 0.05 nats |
| `FashionCNN`, training loss | at most about 0.02 | 0.06 | 0.06 |
| `FashionCNN`, test accuracy | at most about 1 point | 3 points | 3 points |

- Five more runs could give a range half or twice as large, hence the
  margin. The factor 3 is a convention.
- A smaller difference is a tie. To resolve it, repeat the runs.
- The range belongs to its setting: a well-chosen rate, read at step
  2,000.
:::

::: {.slide title="A Fair Comparison Tunes the Learning Rate of Every Arm"}
[The protocol]{.kicker}

SGD trained at rate 0.5, AdamW at 0.003: no shared rate suits both.

| class | meaning | in this chapter |
|:--|:--|:--|
| scientific | what the question is about; each value is an *arm* | the optimizer |
| nuisance | set to its best value separately for each arm | the learning rate |
| fixed | the same in every arm; each limits the claim | model, batch size, 2,000 steps, constant rate |

::: {.d2l-note .warn}
Bold: half a million parameters, 2,000 steps. Trust them for mechanisms
and large differences, not for effects of scale.
:::

[Godbole et al., Deep Learning Tuning Playbook, 2023]{.small}
:::

::: {.slide title="Tuned Means the Best Point of a Grid; Usable Means Within 0.15 Nats of It"}
[The protocol]{.kicker}

Our SGD sweep on `TinyLM`:

| rate | final loss | within 0.15 nats of the best? |
|:--|:--|:--|
| 0.1 | about 2.5 | no: 0.35 higher |
| 0.5 | about 2.1 | the best of the grid |
| 2.5 | `inf` | no: diverged |

- **Tuned:** the grid point with the lowest final loss, no more. Grids are
  a factor of about 3 apart; all runs start from the same weights.
- **Usable range:** here 0.5 alone. 0.15 nats is more than 10 times the
  range of five runs.
:::

::: {.slide title="Check: Result or Tie?"}
[Check]{.kicker}

One run per setting, 2,000 steps each.

1. `TinyLM`: optimizer A ends at 1.57, optimizer B at 1.59.
2. `FashionCNN`: 90.4% against 89.6% test accuracy.
3. `TinyLM`: plain SGD at 2.1, AdamW at 1.59.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** 1: a tie, 0.02 is under 0.05. 2: a tie, 0.8 points is under
3. 3: half a nat is no tie, but three rates a factor of 5 apart did not
tune SGD, so the comparison is not yet fair.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- `TinyLM`: 470,784 parameters, next byte of Wikipedia, 3% of the text
  per run. `FashionCNN`: 421,642 parameters, Fashion-MNIST.
- Landmarks: 5.55 for the uniform guess, about 3.2 for byte frequencies,
  about 2.5 for byte pairs. An untrained `TinyLM` starts near 5.7.
- Plain SGD on `TinyLM`: rate 0.1 ends near 2.5, rate 0.5 near 2.1, rate
  2.5 diverges within a few steps.
- Five runs agree within about 0.01 nats; a result is at least 3 times
  the range, 0.05 nats.
- Protocol: tune the learning rate of every arm on a grid, same initial
  weights, report the usable range.

**Next:** momentum, the first of the two things that AdamW does
differently from plain SGD.
:::
