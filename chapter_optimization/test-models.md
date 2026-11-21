# Two Test Models
:label:`sec_test_models`

Analytical models such as the two-dimensional quadratic valley explain how curvature and gradient noise govern simple update rules, but evaluating modern optimizers requires empirical measurement on real neural networks. Why does AdamW outperform SGD by a wide margin on Transformers while offering a much smaller advantage on convolutional networks? Answering such questions requires training actual models under a controlled experimental protocol.

Because stochastic training runs vary across random initializations and minibatch draws, a difference between two optimizer configurations is meaningful only if it exceeds the run-to-run variation of a single configuration. This section defines the two benchmark architectures used throughout the rest of this chapter—the byte-level Transformer language model `TinyLM` ($470{,}784$ parameters) and the convolutional classifier `FashionCNN` ($421{,}642$ parameters)—along with our shared training loop, learning-rate sweep harness, baseline SGD measurements, and significance thresholds.

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

We now establish our baseline with constant-rate minibatch SGD ($2{,}000$ steps, no momentum or weight decay). On `TinyLM`, an untrained model starts slightly above the uniform-distribution cross-entropy $\ln 256 \approx 5.55$ nats/byte (at $\sim 5.71$ nats/byte due to initial logit variance; Exercise :exref:`ex_test-models-initial-loss`). As reference landmarks on WikiText-103, a unigram byte-frequency table achieves $\sim 3.2$ nats/byte and a bigram byte-pair table achieves $\sim 2.5$ nats/byte (Exercise :exref:`ex_test-models-landmarks`).

Sweeping plain SGD across $\eta \in \{0.1, 0.5, 2.5\}$ exhibits the three canonical learning-rate regimes:

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

* **Too high ($\eta = 2.5$):** Exceeds the local stability threshold $2/L$ after the first step and diverges to `inf` within ten steps.
* **Too low ($\eta = 0.1$):** Descends smoothly but slowly, reaching only $\sim 2.5$ nats/byte (bigram level) at step $2{,}000$.
* **Well-tuned ($\eta = 0.5$):** Experiences brief early transient spikes as the network navigates initial high-curvature regions, then recovers and reaches $\sim 2.13$ nats/byte.

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

* `TinyLM` ($470{,}784$ parameters: $90\%$ 2D projection matrices, $9\%$ embeddings, $<1\%$ 1D vectors) trains on random $64$-byte windows of WikiText-103 ($\sim 2.7 \times 10^8$ bytes), so a $2{,}000$-step run evaluates fresh text on every step.
* `FashionCNN` ($421{,}642$ parameters) trains on $256$-image minibatches of Fashion-MNIST ($\sim 8.5$ epochs in $2{,}000$ steps) and is evaluated by both training loss and held-out test accuracy.
* Every optimizer comparison sweeps the learning rate $\eta$ independently for each method from identical initial weights, reporting both the optimal loss and the usable learning-rate range against a multi-seed significance threshold of $0.05$ nats on `TinyLM`.

## Exercises

1. **Initial loss before training.**
   :exlabel:`ex_test-models-initial-loss`
    1. If logits $z_1, \ldots, z_V$ are drawn i.i.d. from $\mathcal{N}(0, s^2)$ independently of the target, use $\sum_{i=1}^V e^{z_i} \approx V \mathbb{E}[e^{z_i}] = V e^{s^2/2}$ to show that the expected initial cross-entropy is approximately $\ln V + s^2 / 2$.
    1. Given LayerNorm outputs of mean $0$ and variance $1$ and head weights/biases drawn from $\mathcal{U}(-1/\sqrt{128}, 1/\sqrt{128})$, show that $s^2 = 129/384 \approx 1/3$ and $\ln 256 + 1/6 \approx 5.71$ nats, matching the step-$1$ loss of `TinyLM`. What is the initial loss if the head weights and bias are initialized to zero?

1. [code] **Diagnosing loss curves on `FashionCNN`.** Train `FashionCNN` with plain SGD across $\eta \in \{0.003, 0.03, 0.3, 1.0\}$ and plot the step losses on a logarithmic step axis against the uniform baseline $\ln 10 \approx 2.3026$. For the run that stalls near $\ln 10$, inspect the fraction of dead ReLU activations across a minibatch to explain why gradients vanish.
   :exlabel:`ex_test-models-diagnosis`

1. [code] **The edge of divergence on `TinyLM`.** Sweep plain SGD on `TinyLM` across $\eta \in \{0.7, 1.0, 1.4, 2.0\}$ across $5$ seeds (`lr_sweep(..., verbose=False)`). Record the survival rate, the step at which divergent runs fail, and the lowest surviving loss.
   :exlabel:`ex_test-models-edge`

1. [code] **Multi-seed spread and standard error.** ● Run $20$ seeds of `TinyLM` with `torch.optim.AdamW` / `optax.adamw` at $\eta = 3 \times 10^{-3}$ (`weight_decay=0`), compare the sample standard deviation $\sigma$ of `final_loss` against the normal range estimator $\text{range}/3.7$, and compute how many seeds $k$ are required to detect a $0.005$-nat difference at two standard errors ($\sigma\sqrt{2/k}$).
   :exlabel:`ex_test-models-range`

1. [code] **Training loss vs. validation loss and overfitting small corpora.**
   :exlabel:`ex_test-models-held-out`
    1. Explain why `final_loss` (averaged over the last $100$ training steps) is slightly higher ($0.02\text{--}0.03$ nats) than the post-step-$2000$ validation loss on `TinyLM`, and verify by evaluating the final checkpoint on held-out training windows.
    1. Restrict the corpus to $100{,}000$ bytes via `WikiText103(batch_size=64, num_steps=64, num_train=100000)` ($\sim 80$ epochs in $2{,}000$ steps) and plot training vs. validation loss every $250$ steps (`every=250`).

1. [code] **Count-based $n$-gram landmarks.** Compute the empirical byte unigram entropy ($\sim 3.2$ nats) and add-one smoothed bigram ($\sim 2.5$ nats) and trigram cross-entropies from the first $10^7$ training bytes of `WikiText103` evaluated on the validation split, and compare them against the $2{,}000$-step SGD ($2.13$ nats) and AdamW ($1.59$ nats) losses of `TinyLM`.
   :exlabel:`ex_test-models-landmarks`

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
