# Two Test Models
:label:`sec_test_models`

Analytical models such as the two-dimensional quadratic valley explain how curvature and gradient noise govern simple update rules, but evaluating modern optimizers requires empirical measurement on real neural networks. Why does AdamW outperform SGD by a wide margin on Transformers while offering a much smaller advantage on convolutional networks? Answering such questions requires training actual models under a controlled experimental protocol.

Because stochastic training runs vary across random initializations and minibatch draws, a difference between two optimizer configurations is meaningful only if it exceeds the run-to-run variation of a single configuration. This section defines the two benchmark architectures used throughout the rest of this chapter: the byte-level Transformer language model `TinyLM` ($470{,}784$ parameters) and the convolutional classifier `FashionCNN` ($421{,}642$ parameters), along with our shared training loop, learning-rate sweep harness, baseline SGD measurements, and significance thresholds.

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

`TinyLM` is an autoregressive byte-level language model that predicts the next UTF-8 byte ($V = 256$) given a context window of $64$ consecutive bytes (:numref:`sec_text-sequence`). A standard $2{,}000$-step run with minibatches of $64$ windows processes $2{,}000 \times 64 \times 64 \approx 8.2 \times 10^6$ target bytes. To avoid repeating training examples (which would conflate optimization speed with memorization), we train on the first shard of WikiText-103 :cite:`Merity.Xiong.Bradbury.ea.2016` ($\sim 2.7 \times 10^8$ UTF-8 bytes of Wikipedia articles). Each training minibatch samples $64$ windows at random byte offsets, so a $2{,}000$-step run sees only $\sim 3\%$ of the corpus and every training step evaluates fresh text. Validation perplexity is measured on consecutive non-overlapping windows of the $1.1 \times 10^6$-byte validation split.

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

Instantiating `WikiText103(batch_size=64, num_steps=64)` downloads the training and validation shards and decodes the opening $300$ bytes of the corpus:

```{.python .input #adam-a-tiny-language-model-1}
%%tab pytorch, jax
data = WikiText103(batch_size=64, num_steps=64)
print(f'{len(data.train):,} training bytes, '
      f'{len(data.val):,} validation bytes')
text = data.vocab.decode(data.train[:300])
print('\n'.join(textwrap.fill(line, 72) for line in text.splitlines()))
```

### Architecture and Parameter Groups

`TinyLM` is a two-block decoder-only Transformer (:numref:`chap_transformers`) with embedding width $d_{\text{model}} = 128$, $2$ causal attention heads, and a two-layer GELU MLP of hidden width $4 d_{\text{model}} = 512$ per block. Each sublayer is preceded by LayerNorm (:numref:`subsec_layer-normalization-in-bn`) and wrapped in a residual connection, followed by a final LayerNorm and linear output head projecting to $256$ byte logits. Embedding rows are initialized from $\mathcal{N}(0, 1)$, and all linear weights and biases are drawn uniformly from $[-1/\sqrt{\text{fan-in}}, 1/\sqrt{\text{fan-in}}]$.

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

For an optimizer, the $470{,}784$ parameters of `TinyLM` divide into three distinct structural groups that later sections treat differently (e.g., for decoupled weight decay in :numref:`sec_adamw` and matrix orthogonalization in :numref:`sec_muon`):

* **Embeddings ($9\%$):** lookup tables (`token_emb`, `pos_emb`) whose rows receive sparse gradients proportional to token frequency.
* **Weight matrices ($90\%$):** dense 2D projection matrices (`qkv`, `proj`, the two MLP layers per block, and `head`).
* **1D vectors ($<1\%$):** linear biases and LayerNorm scale/shift vectors.

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

## The Image Classifier FashionCNN
:label:`subsec_fashioncnn`

Because optimizer rankings on Transformers differ markedly from those on convolutional networks, our second benchmark is `FashionCNN`, a classifier for $28 \times 28$ Fashion-MNIST images (:numref:`sec_fashion_mnist`). Two $3 \times 3$ convolution + ReLU + $2 \times 2$ max-pooling stages ($32$ and $64$ channels) reduce the spatial resolution to $64 \times 7 \times 7$, followed by two linear layers ($3{,}136 \to 128 \to 10$). All layers use uniform $\pm 1/\sqrt{\text{fan-in}}$ initialization, totaling $421{,}642$ parameters.

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

We verify with `d2l.num_params` that `FashionCNN` has $421{,}642$ trainable parameters:

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

With batch size $256$, a $2{,}000$-step run makes $\sim 8.5$ passes over the $60{,}000$ Fashion-MNIST training images, so we report held-out test accuracy alongside training loss via `evaluate_accuracy`:

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

## Training Harness and Evaluation Helpers

Both models map a minibatch to logits and minimize mean cross-entropy loss over a fixed step budget. :numref:`tab_test-models-helpers` summarizes the training loop and evaluation utilities saved in `d2l`.

:Shared training and evaluation functions saved in `d2l`.
:label:`tab_test-models-helpers`

| Function | Purpose |
|:--|:--|
| `train_steps` | Train for `num_steps` steps (or until `stop(step)` holds) and return the per-step cross-entropy losses |
| `lm_perplexity` | Compute token perplexity $\exp(\text{mean cross-entropy})$ over an iterator of batches (:numref:`sec_language-model`) |
| `evaluate_accuracy` | Compute classification accuracy over an iterator of batches |
| `final_loss` | Mean of the last $k=100$ step losses (`inf` if diverged) |
| `smooth` | Block average over consecutive windows of $k=25$ losses for plotting |
| `steps_to_reach` | First step at which the $25$-step smoothed loss drops to `target` (or `None`) |
| `lr_sweep` | Train from identical initial weights across a list of learning rates and return loss curves and evaluation scores |

We first implement `train_steps`, which minimizes mean cross-entropy over `num_steps` minibatches (ignoring padding label `-100`) or until an optional predicate `stop(step)` holds:

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

Next, `lm_perplexity` evaluates a language model in inference mode over an iterator of minibatches and returns $\exp(\text{mean cross-entropy})$ per token:

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

To summarize and plot noisy step-loss histories, `final_loss` averages the last $k = 100$ step losses, `smooth` computes $25$-step block averages, and `steps_to_reach` finds the first step where the smoothed loss reaches `target`:

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

Finally, `lr_sweep` trains a fresh model from the same random `seed` for every learning rate in `lrs`, optionally running `evaluate(model)` every `every` steps:

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

## Baseline Minibatch SGD Runs and Experimental Protocol

We now establish our baseline with constant-rate minibatch SGD ($2{,}000$ steps, no momentum or weight decay). On `TinyLM`, an untrained model starts slightly above the uniform-distribution cross-entropy $\ln 256 \approx 5.55$ nats/byte (at $\sim 5.71$ nats/byte due to initial logit variance; Exercise :exref:`ex_test-models-initial-loss`). As reference landmarks on WikiText-103, a unigram byte-frequency table achieves $\sim 3.2$ nats/byte and a bigram byte-pair table achieves $\sim 2.5$ nats/byte (Exercise :exref:`ex_test-models-landmarks`). :numref:`fig_opt_loss_landmarks` places these landmarks on one scale; every loss in this chapter can be read against it.

![Loss landmarks for byte-level language modeling on WikiText-103. Gray: the untrained `TinyLM` and the uniform guess. Blue: unigram and bigram byte tables. Orange: `TinyLM` after $2{,}000$ steps of SGD, measured below, and after the best tuned runs later in this chapter.](../img/mdl-opt-loss-landmarks.svg)
:label:`fig_opt_loss_landmarks`

We first define the model and plain-SGD factory closures `make_lm`, `sgd`, and `new_seed`:

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

Running `lr_sweep` on `TinyLM` across $\eta \in \{0.1, 0.5, 2.5\}$ and plotting the step losses against the uniform-guess baseline $\ln 256$ exhibits the three canonical learning-rate regimes:

```{.python .input #test-models-a-first-run-with-minibatch-sgd-2}
%%tab pytorch, jax
lm_sgd, _ = lr_sweep(make_lm, sgd, data, lrs=[0.1, 0.5, 2.5], seed=new_seed())
d2l.plot(list(range(1, 2001)),
         list(lm_sgd.values()) + [[math.log(256)] * 2000],
         'step', 'training loss', xscale='log', ylim=[1.5, 6],
         legend=[f'lr {lr:g}' for lr in lm_sgd] + ['uniform guess'],
         legend_loc='outside', figsize=(5, 2.5))
```

* **Too high ($\eta = 2.5$):** Exceeds the local stability threshold $2/L$ after the first step and diverges to `inf` within ten steps.
* **Too low ($\eta = 0.1$):** Descends smoothly but slowly, reaching only $\sim 2.5$ nats/byte (bigram level) at step $2{,}000$.
* **Well-tuned ($\eta = 0.5$):** Experiences brief early transient spikes as the network navigates initial high-curvature regions, then recovers and reaches about $2.1$ nats/byte.

On `FashionCNN`, plain SGD at $\eta = 0.3$ reaches a training loss of $\sim 0.22$ and a held-out test accuracy of $\sim 90\%$:

```{.python .input #test-models-a-first-run-with-minibatch-sgd-3}
%%tab pytorch, jax
test_acc = lambda model: evaluate_accuracy(model, fashion.val_dataloader())
cnn_sgd, _ = lr_sweep(FashionCNN, sgd, fashion, lrs=[0.3], seed=new_seed(),
                      evaluate=test_acc, metric='test accuracy')
```

### Comparison Protocol and Significance Thresholds

Following the Deep Learning Tuning Playbook :cite:`Godbole.Dahl.Gilmer.ea.2023`, comparing two optimizers fairly requires treating the learning rate $\eta$ as a **nuisance hyperparameter** that must be swept independently for each method (`lr_sweep` over a log-spaced grid from shared initial weights) rather than comparing two optimizers at an arbitrary shared learning rate :cite:`Schmidt.Schneider.Hennig.2021`. Alongside the minimum loss achieved at each optimizer's best learning rate, we track its **usable learning-rate range** (grid points finishing within $0.15$ nats of the optimum).

Across $5$ independent random seeds at a well-tuned learning rate, the final $100$-step training loss (`final_loss`) and validation loss of `TinyLM` vary over a range of $\sim 0.01\text{--}0.015$ nats, while `FashionCNN` varies by $\sim 0.02$ in training loss and $\sim 1\%$ in test accuracy. Adopting a conservative $3\times$ range margin, we treat a single-run difference as statistically meaningful only if it exceeds **$0.05$ nats** on `TinyLM` or **$0.06$ training loss ($3\%$ test accuracy)** on `FashionCNN`; smaller differences are treated as ties unless confirmed across multiple seeds.

## Summary

Our empirical comparisons pair two complementary neural network benchmarks: `TinyLM` ($470{,}784$ parameters: $90\%$ 2D projection matrices, $9\%$ embeddings, and $<1\%$ 1D vectors), which trains in a single-pass streaming regime on random $64$-byte windows of WikiText-103 so that a $2{,}000$-step run evaluates fresh text on every step, and `FashionCNN` ($421{,}642$ parameters), which trains for $\sim 8.5$ epochs on $256$-image minibatches of Fashion-MNIST and is evaluated by both training loss and held-out test accuracy. Closed-form uniform, unigram, and bigram byte entropies anchor the `TinyLM` loss scale before any training takes place. To separate genuine algorithmic improvements from hyperparameter and seed sensitivity, every comparison sweeps the learning rate $\eta$ independently for each optimizer from identical initial weights, reporting both the optimal loss and the usable learning-rate range against a multi-seed significance threshold of $0.05$ nats on `TinyLM` ($0.06$ loss or $3\%$ accuracy on `FashionCNN`).

## Exercises

1. **Initial loss before training.**
   :exlabel:`ex_test-models-initial-loss`
    1. If logits $z_1, \ldots, z_V$ are drawn i.i.d. from $\mathcal{N}(0, s^2)$ independently of the target, use $\sum_{i=1}^V e^{z_i} \approx V \mathbb{E}[e^{z_i}] = V e^{s^2/2}$ to show that the expected initial cross-entropy is approximately $\ln V + s^2 / 2$.
    1. Given LayerNorm outputs of mean $0$ and variance $1$ and head weights/biases drawn from $\mathcal{U}(-1/\sqrt{128}, 1/\sqrt{128})$, show that $s^2 = 129/384 \approx 1/3$ and $\ln 256 + 1/6 \approx 5.71$ nats, matching the step-$1$ loss of `TinyLM`. What is the initial loss if the head weights and bias are initialized to zero?

1. [code] **Diagnosing loss curves on `FashionCNN`.** Train `FashionCNN` with plain SGD across $\eta \in \{0.003, 0.03, 0.3, 1.0\}$ and plot the step losses on a logarithmic step axis against the uniform baseline $\ln 10 \approx 2.3026$. For the run that stalls near $\ln 10$, inspect the fraction of dead ReLU activations across a minibatch to explain why gradients vanish.
   :exlabel:`ex_test-models-diagnosis`

1. [code] **The edge of divergence on `TinyLM`.** Sweep plain SGD on `TinyLM` across $\eta \in \{0.7, 1.0, 1.4, 2.0\}$ across $5$ seeds (`lr_sweep(..., verbose=False)`). Record the survival rate, the step at which divergent runs fail, and the lowest surviving loss.
   :exlabel:`ex_test-models-edge`

1. [code] **Multi-seed spread and standard error.** ● Run $20$ seeds of `TinyLM` with AdamW (:numref:`sec_adam`) at $\eta = 3 \times 10^{-3}$ (`weight_decay=0`), compare the sample standard deviation $\sigma$ of `final_loss` against the normal range estimator $\text{range}/3.7$, and compute how many seeds $k$ are required to detect a $0.005$-nat difference at two standard errors ($\sigma\sqrt{2/k}$).
   :exlabel:`ex_test-models-range`

1. [code] **Training loss vs. validation loss and overfitting small corpora.**
   :exlabel:`ex_test-models-held-out`
    1. Explain why `final_loss` (averaged over the last $100$ training steps) is slightly higher ($0.02\text{--}0.03$ nats) than the post-step-$2000$ validation loss on `TinyLM`, and verify by evaluating the final checkpoint on held-out training windows.
    1. Restrict the corpus to $100{,}000$ bytes via `WikiText103(batch_size=64, num_steps=64, num_train=100000)` ($\sim 80$ epochs in $2{,}000$ steps) and plot training vs. validation loss every $250$ steps (`every=250`).

1. [code] **Count-based $n$-gram landmarks.** Compute the empirical byte unigram entropy ($\sim 3.2$ nats) and add-one smoothed bigram ($\sim 2.5$ nats) and trigram cross-entropies from the first $10^7$ training bytes of `WikiText103` evaluated on the validation split, and compare them against the $2{,}000$-step SGD (about $2.1$ nats) and AdamW ($1.59$ nats, :numref:`sec_adam`) losses of `TinyLM`.
   :exlabel:`ex_test-models-landmarks`

[Discussions](https://d2l.discourse.group/t/36884)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.3]{.kicker}

Two Test Models: TinyLM and FashionCNN<br>**byte-level Transformer · convolutional classifier · count-based landmarks · LR sweep protocol · multi-seed significance**
:::
:::

::: {.slide title="From Solvable Quadratics to Neural Network Benchmarks"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why quadratics aren't enough:** Neural networks have evolving sharpness $\lambda_{\max}(\mathbf{x}_t)$, $40\times$ cross-layer gradient disparities, and stochastic seed variance.
- **Three experimental pillars:**
  1. **Two complementary models:** `TinyLM` (single-pass byte Transformer) and `FashionCNN` (multi-epoch image CNN).
  2. **Untrained loss landmarks:** Closed-form uniform, unigram, and bigram byte entropies.
  3. **Fair tuning protocol:** Independent LR sweep per arm from shared seed with $3\times$ multi-seed significance thresholds.
:::

::: {.col .fig .big}
@fig:ch09-opt-test-quad-vs-net
:::
:::
:::

::: {.slide title="Benchmark 1: TinyLM on Single-Pass WikiText-103 Bytes"}
[Language Modeling Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
- **Autoregressive byte prediction ($V = 256$):** Predicts the next UTF-8 byte given a context window of $64$ consecutive bytes.
- **Single-pass streaming regime:**
  - Corpus: First shard of WikiText-103 ($\sim 2.7 \times 10^8$ UTF-8 bytes of Wikipedia articles).
  - Minibatch: $b = 64$ windows of $64$ bytes ($4{,}096$ target bytes/step) sampled at random byte offsets.
  - A $2{,}000$-step run processes $8.2 \times 10^6$ bytes ($\sim 3\%$ of the corpus), so every training step evaluates fresh, unseen text without memorization.
:::

::: {.col .fig}
@!adam-a-tiny-language-model-1
:::
:::
:::

::: {.slide title="Inside TinyLM: Decoder-Only Transformer Architecture"}
[Transformer Structure]{.kicker}

::: {.cols .vc}
::: {.col}
- **Embeddings:** $\mathbf{H}_0 = \mathbf{E}_{\text{tok}}(\mathbf{X}) + \mathbf{E}_{\text{pos}}(0..63) \in \mathbb{R}^{B \times 64 \times 128}$, initialized from $\mathcal{N}(\mathbf{0}, \mathbf{I})$.
- **Two pre-norm residual blocks ($d_{\text{model}} = 128, 2\text{ heads}$):**
  $$\mathbf{H} \leftarrow \mathbf{H} + \text{CausalMHA}(\text{LN}_1(\mathbf{H}))$$
  $$\mathbf{H} \leftarrow \mathbf{H} + \text{MLP}_{\text{GELU}}(\text{LN}_2(\mathbf{H}))$$
- **Readout head:** $\text{Linear}_{128 \to 256}(\text{LN}(\mathbf{H}))$ initialized from $\mathcal{U}(-1/\sqrt{\text{fan-in}}, 1/\sqrt{\text{fan-in}})$.
:::

::: {.col .fig}
@fig:ch09-opt-test-tinylm-arch
:::
:::
:::

::: {.slide title="Parameter Census of TinyLM: Three Structural Groups"}
[Where the Parameters Live]{.kicker}

To an optimizer, `TinyLM`'s $470{,}784$ parameters partition into three distinct structural groups:

- **1. Dense 2D weight matrices ($425{,}984$ params, $90.5\%$):** Four block matrices per layer (`qkv` $128 \to 384$, `proj` $128 \to 128$, `mlp` $128 \to 512 \to 128$) plus output `head` ($128 \to 256$). Decayed in AdamW (§{sec_adam}); internal matrices orthogonalized in Muon (§{sec_muon}).
- **2. Embedding lookup tables ($40{,}960$ params, $8.7\%$):** `token_emb` ($256 \times 128$) and `pos_emb` ($64 \times 128$); receive sparse row gradients proportional to byte frequency.
- **3. 1D vectors ($3{,}840$ params, $0.8\%$):** Linear biases and LayerNorm scales/shifts.
:::

::: {.slide title="Parameter Census of TinyLM: Three Structural Groups (cont.)" layout="tall-output"}
[Where the Parameters Live]{.kicker}

The census lists every parameter tensor with its shape and size, then the three group totals:

@!adam-a-tiny-language-model-3
:::

::: {.slide title="Benchmark 2: FashionCNN on Fashion-MNIST Images"}
[Convolutional Classifier]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why a second architecture?** Optimizer gaps that are large on Transformers often shrink on convolutional networks.
- **Architecture ($421{,}642$ parameters):**
  - `conv1` ($1 \to 32$, $3 \times 3$, ReLU, $2 \times 2$ max-pool $\to 32 \times 14 \times 14$): $320$ params.
  - `conv2` ($32 \to 64$, $3 \times 3$, ReLU, $2 \times 2$ max-pool $\to 64 \times 7 \times 7$): $18{,}496$ params.
  - `fc1` ($3{,}136 \to 128$, ReLU): $401{,}536$ params (**$95\%$ of total**).
  - `fc2` ($128 \to 10$ logits): $1{,}290$ params.
:::

::: {.col .fig}
@-adam-the-same-race-on-a-cnn-1
:::
:::
:::

::: {.slide title="Comparing Our Two Neural Benchmarks Side by Side"}
[Complementary Regimes]{.kicker}

::: {.cols .vc}
::: {.col}
- **Single-pass vs. multi-epoch exposure ($2{,}000$ steps):**
  - `TinyLM` sees $0.03$ epochs ($3\%$ of $270\text{M}$ bytes)—fresh unseen text on every step without memorization.
  - `FashionCNN` sees $8.5$ epochs ($512\text{K}$ evaluations over $60\text{K}$ images) and requires held-out test accuracy to track overfitting.
- **Complementary parameter geometry:**
  - `TinyLM` spreads $90.5\%$ of weights across $9$ Transformer matrices plus sparse embeddings; `FashionCNN` concentrates $95\%$ of weights in `fc1`.
:::

::: {.col .fig .big}
@fig:ch09-opt-test-regime-comparison
:::
:::
:::

::: {.slide title="Shared Training Harness and Evaluation Utilities"}
[Experimental Infrastructure]{.kicker}

::: {.cols .vc}
::: {.col}
- **Controlled sweep & loop (`lr_sweep`, `train_steps`):** Clones one shared random initialization across a log-spaced $\eta$ grid and runs $T = 2{,}000$ minibatches.
- **Smoothed trajectory metrics (`final_loss`, `steps_to_reach`):** Averages the last $k = 100$ step losses and finds the first $25$-step block reaching a target threshold.
- **Held-out evaluation (`lm_perplexity`, `evaluate_accuracy`):** Scores $1.1\text{M}$ validation bytes or $10{,}000$ test images in eval mode.
:::

::: {.col .fig .big}
@fig:ch09-opt-test-harness-flow
:::
:::
:::

::: {.slide title="Information-Theoretic Landmarks Before Any Training"}
[Calibrating the Loss Scale]{.kicker}

::: {.cols .vc}
::: {.col}
- **Untrained step 1 ($\approx 5.71$ nats):** Random head logits $z_i \sim \mathcal{N}(0, s^2)$ with $s^2 \approx 1/3$ add variance penalty $s^2/2 \approx 0.17$ above uniform.
- **Uniform random guess ($5.55$ nats):** $\ln 256$ ($\ln 10 \approx 2.30$ on `FashionCNN`).
- **Unigram byte frequency ($\approx 3.18$ nats):** Marginal UTF-8 counts without context.
- **Bigram table ($\approx 2.50$ nats):** Conditional $p(b_t \mid b_{t-1})$; beating $2.50$ nats demonstrates multi-byte context learning.
:::

::: {.col .fig .big}
@fig:ch09-opt-test-loss-landmarks
:::
:::
:::

::: {.slide title="Baseline Minibatch SGD on TinyLM Across Three Rates"}
[Three Canonical Regimes]{.kicker}

::: {.cols .vc}
::: {.col}
- Sweeping plain minibatch SGD (no momentum, no weight decay, $2{,}000$ steps) across $\eta \in \{0.1, 0.5, 2.5\}$ from identical initial weights reveals three distinct trajectories:
  - **$\eta = 0.1$ (blue):** Smooth, monotonic descent that finishes at **about $2.5$ nats**—barely reaching the bigram table landmark ($2.5$).
  - **$\eta = 0.5$ (orange):** Spikes upward by $>1$ nat in the first $30$ steps, then recovers and reaches **about $2.1$ nats**.
  - **$\eta = 2.5$ (green):** Drops on step 1, explodes above $6$ nats on step 2, and diverges to **`inf`** within $10$ steps.
:::

::: {.col .fig}
@!test-models-a-first-run-with-minibatch-sgd-2
:::
:::
:::

::: {.slide title="Diagnosing Learning-Rate Failure Modes from Loss Curves"}
[Reading Training Logs]{.kicker}

::: {.cols .vc}
::: {.col}
- **$\eta = 0.1$ (Under-budgeted travel):** Smooth descent still dropping at step $2{,}000$; cumulative travel $B_T = \eta T = 200$ is too small $\to$ **raise $\eta$**.
- **$\eta = 0.5$ (Near initial sharpness edge $2/L$):** Early $>1$-nat spike in steps $1\text{--}30$, then fast descent to about $2.1$ nats $\to$ **add warmup (§{sec_scheduler}) or clipping (§{sec_practice})**.
- **$\eta = 2.5$ (Exceeds $2/\lambda_{\max}$):** Step 1 overshoots into a steeper basin and diverges to `inf` $\to$ **lower $\eta$**.
:::

::: {.col .fig .big}
@fig:ch09-opt-test-lr-diagnosis
:::
:::
:::

::: {.slide title="Baseline Minibatch SGD on FashionCNN"}
[Classification Baseline]{.kicker}

::: {.cols .vc}
::: {.col}
- Running the exact same $2{,}000$-step `train_steps` loop on `FashionCNN` ($b = 256$ images) with plain SGD at $\eta = 0.3$ establishes our image classification baseline:
  - **Final training loss:** $\approx 0.22\text{--}0.23$ nats (down from uniform $\ln 10 \approx 2.30$).
  - **Held-out test accuracy:** $\approx 89.5\%\text{--}90.5\%$.
- Together, **`TinyLM` (about $2.1$ nats under plain SGD)** and **`FashionCNN` ($0.22$ loss / $90\%$ test accuracy)** serve as the reference baselines for §{sec_momentum}–§{sec_practice}.
:::

::: {.col .fig}
@!test-models-a-first-run-with-minibatch-sgd-3
:::
:::
:::

::: {.slide title="Fair Comparisons: The Nuisance Parameter Protocol"}
[Experimental Design]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why fixed-$\eta$ comparisons fail:** Plain SGD multiplies raw gradients ($\eta^\star \approx 0.5$), whereas AdamW normalizes step lengths ($\eta^\star \approx 3 \times 10^{-3}$).
- **Nuisance hyperparameter tuning (Godbole et al., 2023):** Sweep $\eta$ independently on a log-spaced grid for every optimizer arm from shared initial weights.
- **Report both peak and width:** Extract the tuned score at $\eta^\star$ *and* the usable LR range finishing within $+0.15$ nats of $\eta^\star$.
:::

::: {.col .fig .big}
@fig:ch09-opt-test-nuisance-protocol
:::
:::
:::

::: {.slide title="Reporting Both Peak Performance and Usable LR Range"}
[Robustness vs. Peak Score]{.kicker}

When evaluating a learning-rate sweep `lr_sweep` on a log-spaced grid (e.g., factors of $\sim 3\times$), we extract two distinct metrics for each optimizer:

- **1. Tuned score at the grid optimum $\eta^\star$:** The lowest `final_loss` (or validation loss) achieved across the grid points tested, starting from identical initial weights.
- **2. Usable learning-rate range:** The span of grid learning rates that finish within **$+0.15$ nats** of the optimum ($>10\times$ the multi-seed spread).
  - *Example on `TinyLM`:* Under plain SGD across $\{0.1, 0.5, 2.5\}$, only a single grid point ($\eta = 0.5$ at about $2.1$ nats) is usable ($\eta = 0.1$ is about $0.35$ nats worse; $\eta = 2.5$ diverges to `inf`).
  - As we will see in §{sec_adam}–§{sec_muon}, adaptive and orthogonalized optimizers widen this usable window across an order of magnitude ($10^{-3}$ to $10^{-2}$ or $3 \times 10^{-2}$).
:::

::: {.slide title="Multi-Seed Spread and the 3× Significance Threshold"}
[Statistical Rigor]{.kicker}

::: {.cols .vc}
::: {.col}
- **Run-to-run seed variation:** Different initial weights and minibatch draws shift 5-seed ranges by $0.010\text{--}0.015$ nats on `TinyLM` and $0.8\%\text{--}1.0\%$ accuracy on `FashionCNN`.
- **Single-run $3\times$ decision threshold:**
  - **`TinyLM`:** Require **$\ge 0.05$ nats/byte** improvement.
  - **`FashionCNN`:** Require **$\ge 0.06$ loss** or **$\ge 3.0\%$ accuracy** improvement.
- Smaller single-run gaps are treated as **statistical ties** unless averaged across multiple seeds.
:::

::: {.col .fig .big}
@fig:ch09-opt-test-seed-threshold
:::
:::
:::

::: {.slide title="Concept Check: Landmarks, Curve Diagnosis, and Seeds"}
[Check Your Understanding]{.kicker}

Apply the experimental protocol of this section to three concrete questions:

1. **Zero-initialized head:** In Exercise 1, an untrained `TinyLM` with default random head weights ($s^2 \approx 1/3$) starts at $\ln 256 + s^2/2 \approx 5.71$ nats. What is its step-1 loss if the output head weights and biases are initialized to exact zeros?
2. **Result or statistical tie?** In a single $2{,}000$-step run from shared initial weights, Optimizer A finishes at $1.57$ nats on `TinyLM` and $90.4\%$ accuracy on `FashionCNN`, while Optimizer B finishes at $1.59$ nats and $89.6\%$ accuracy. Which optimizer wins?
3. **Training vs. validation loss on `TinyLM`:** Why is `TinyLM`'s post-step-$2{,}000$ validation loss typically $0.02\text{--}0.03$ nats *lower* than `final_loss`?
:::

::: {.slide title="Concept Check: Landmarks, Curve Diagnosis, and Seeds (answers)"}
[Check Your Understanding]{.kicker}

::: {.d2l-note}
**Answers:**

1. Zero head weights and biases produce zero logits ($\mathbf{z} = \mathbf{0}$), giving an exact uniform distribution $p_i = 1/256$ and initial loss **$\ln 256 \approx 5.545$ nats** ($0.17$ nats lower from step 1).
2. **Both are ties.** On `TinyLM`, $|1.57 - 1.59| = 0.02 < 0.05$ nats; on `FashionCNN`, $|90.4\% - 89.6\%| = 0.8\% < 3.0\%$. Resolving a $0.02$-nat gap requires averaging across multiple seeds.
3. Because `TinyLM` sees fresh text on every step (no overfitting) and is still improving between steps $1{,}900$ and $2{,}000$, `final_loss` averages steps $1{,}901\text{--}2{,}000$ whereas validation evaluates the final step-$2{,}000$ weights.
:::
:::

::: {.slide title="Summary: Test Models, Landmarks, and Tuning Protocol"}
[Key Takeaways]{.kicker}

| Component | Specification | Key Numbers to Remember | Pedagogical Role |
| :--- | :--- | :--- | :--- |
| **`TinyLM` on WikiText-103** | 2-block Transformer, $d=128$, $2$ heads, $64 \times 64$ byte batches | $470{,}784$ params ($90\%$ matrices, $9\%$ emb, $1\%$ vectors); $3\%$ corpus in $2{,}000$ steps | Single-pass pretraining benchmark without memorization |
| **`FashionCNN` on Fashion-MNIST** | 2 Conv-ReLU-Pool stages + 2 FC layers, $b = 256$ images | $421{,}642$ params ($95\%$ in `fc1`); $8.5$ epochs in $2{,}000$ steps | Multi-epoch classification benchmark with held-out accuracy |
| **Untrained Loss Landmarks** | Closed-form / count-based byte predictors | Untrained $5.71 \to$ Uniform $5.55 \to$ Unigram $3.2 \to$ Bigram $2.5$ nats | Anchors what a given nats/byte value means semantically |
| **Plain SGD Baselines** | $2{,}000$ steps, constant $\eta$, tuned on grid | `TinyLM` at $\eta=0.5$: **$\approx 2.1$ nats**; `FashionCNN` at $\eta=0.3$: **$0.22$ loss / $90\%$ acc** | Reference bar that Momentum, AdamW, and Muon must beat |
| **Comparison & Significance** | Independent LR sweep per arm from shared seed | Single-run threshold: **$0.05$ nats** (`TinyLM`), **$0.06$ loss / $3\%$ acc** (`FashionCNN`) | Separates algorithmic gains from nuisance tuning & seed noise |
:::

