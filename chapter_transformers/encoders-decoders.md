# Encoders, Decoders, and Cross-Attention
:label:`sec_transformer`

Every Transformer stack we have built so far has been a **causal decoder** (:numref:`sec_gpt`): a triangular mask restricts token $i$ to attending to prefix positions $j \le i$, turning a single forward pass into $T$ parallel next-token predictions. Causal masking is essential when generating tokens one by one, since future tokens do not yet exist at inference time. When the *entire* input sequence is already available—for example, when embedding a document for retrieval, classifying tokens, encoding a source sentence for translation, or encoding an image patch grid (:numref:`sec_vision-transformer`)—forbidding position $i$ from reading positions $j > i$ discards half of the surrounding context.

Depending on which tokens are available at the start of the computation and whether the input and output belong to the same sequence, the same `TransformerBlock` (:numref:`sec_transformer-block`) supports three canonical wirings (:numref:`fig_three-wirings`):

1. **Encoder-only (bidirectional self-attention):** Every token attends to every token (`is_causal=False`), computing a context-aware representation of a complete input sequence in one pass (BERT :cite:`Devlin.Chang.Lee.ea.2018`, RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`, ViT :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`).
2. **Decoder-only (causal self-attention):** Each token attends only to itself and earlier positions (`is_causal=True`), supporting both prompt reading and autoregressive generation in a single homogeneous stack (GPT, Llama 3, Qwen3).
3. **Encoder--decoder (bidirectional encoder + causal decoder with cross-attention):** A bidirectional encoder maps a source sequence $\mathbf{X}_{\textrm{src}}$ to contextual representations $\mathbf{H}_{\textrm{enc}}$, and a causal decoder generates the target sequence $\mathbf{Y}_{\textrm{tgt}}$ one token at a time while querying $\mathbf{H}_{\textrm{enc}}$ through **cross-attention** (the original 2017 Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017`, T5 :cite:`raffel2020exploring`, Whisper :cite:`radford2023whisper`).

```{.python .input #encoders-decoders-encoders-decoders-and-cross-attention}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #encoders-decoders-encoders-decoders-and-cross-attention}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Encoder, Decoder, and Encoder--Decoder Architectures

![The three wirings and their attention patterns (query position runs downward, key position rightward; a filled cell means the query may attend to that key). Encoder-only: every token attends to every token. Decoder-only: each token attends only to itself and its past. Encoder--decoder: the source attends to itself bidirectionally, while each target token attends to the full source through cross-attention (light cells) and to its own past causally. The third panel hides one structural fact: its source rows and its target rows belong to two separate stacks, and every decoder block reads the output of the last encoder block.](../img/mdl-transformers-three-wirings.svg)
:label:`fig_three-wirings`

## Bidirectional Context and Masked-Token Prediction

### A Bidirectional Encoder

Removing the causal mask from `d2l.TransformerBlock` turns a causal stack into a **bidirectional Transformer encoder** (`TransformerEncoder`), mapping an input batch of token IDs (and an optional padding mask) to contextual representations of shape $(B, n, d)$:

```{.python .input #encoders-decoders-a-bidirectional-encoder-in-a-dozen-lines}
%%tab pytorch
class TransformerEncoder(nn.Module):
    """Bidirectional encoder: embeddings plus unmasked pre-norm blocks."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=4,
                 max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, num_hiddens)
        self.pos_emb = nn.Embedding(max_len, num_hiddens)
        for emb in (self.token_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        self.blks = nn.ModuleList([
            d2l.TransformerBlock(num_hiddens, num_heads)
            for _ in range(num_blks)])
        self.norm = nn.RMSNorm(num_hiddens, eps=1e-5)

    def forward(self, X, mask=None):
        H = self.token_emb(X) + self.pos_emb(
            torch.arange(X.shape[1], device=X.device))
        for blk in self.blks:
            H = blk(H, mask)  # by default no mask: all tokens see all
        return self.norm(H)
```

```{.python .input #encoders-decoders-a-bidirectional-encoder-in-a-dozen-lines}
%%tab jax
class TransformerEncoder(nnx.Module):
    """Bidirectional encoder: embeddings plus unmasked pre-norm blocks."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=4,
                 max_len=64, *, rngs):
        init = nnx.initializers.normal(0.02)
        self.token_emb = nnx.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, num_hiddens, embedding_init=init,
                                 rngs=rngs)
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, rngs=rngs)
            for _ in range(num_blks)])
        self.norm = nnx.RMSNorm(num_hiddens, epsilon=1e-5, rngs=rngs)

    def __call__(self, X, mask=None):
        H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = blk(H, mask)  # by default no mask: all tokens see all
        return self.norm(H)
```

### The Masked Language Modeling (MLM) Objective

Why can a bidirectional stack not be trained with the next-token prediction objective (:eqref:`eq_gpt-loss`) of :numref:`sec_gpt`? In a stack of $L \ge 2$ unmasked layers, position $t$ at block 1 attends to position $t+1$, and at block 2 reads its own target $x_{t+1}$ directly—collapsing next-token prediction to a trivial identity copy.

**Masked Language Modeling (MLM)**, introduced in **BERT** :cite:`Devlin.Chang.Lee.ea.2018`, replaces next-token prediction with a cloze task: randomly select a subset of positions $M \subset \{1, \dots, n\}$ (typically $p = 15\%$ of tokens), replace each $x_i$ ($i \in M$) with a dedicated `<mask>` token ID in the corrupted input $\tilde{\mathbf{x}}$, and minimize the cross-entropy of reconstructing the original tokens *only at the masked positions* $i \in M$:

$$
\mathcal{L}_{\textrm{MLM}}(\boldsymbol{\theta}) \;=\; -\frac{1}{|M|} \sum_{i \in M} \log P_{\boldsymbol{\theta}}(x_i \mid \tilde{\mathbf{x}}).
$$
:eqlabel:`eq_mlm`

We train a 4-block byte-level `MaskedLM` ($d = 128, H = 4, L = 4$) on the exact same WikiText-103 byte stream as in :numref:`sec_gpt`, reserving token ID `MASK = len(data.vocab)` ($256$) for `<mask>`:

```{.python .input #encoders-decoders-the-masked-token-objective}
%%tab pytorch
data = d2l.WikiText103(batch_size=64, num_steps=64)
MASK = len(data.vocab)  # id of the extra <mask> embedding row

def mask_tokens(X, p=0.15):
    mask = torch.rand(X.shape, device=X.device) < p
    return torch.where(mask, MASK, X), mask

class MaskedBytes:
    """The windows of data with masked bytes; labels -100 add no loss."""
    def __init__(self, data):
        self.data = data

    def train_dataloader(self):
        return self.masked(self.data.train_dataloader())

    def val_dataloader(self):
        return self.masked(self.data.val_dataloader())

    def masked(self, batches):
        for X, _ in batches:
            Xm, mask = mask_tokens(X)
            yield Xm, torch.where(mask, X, -100)

class MaskedLM(nn.Module):
    """An encoder with an output head tied to its token embedding."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, X):
        return F.linear(self.encoder(X), self.encoder.token_emb.weight)

device = d2l.try_gpu()
model = MaskedLM(TransformerEncoder(len(data.vocab) + 1)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                              weight_decay=0.0)
losses = d2l.train_steps(model, MaskedBytes(data), optimizer, 2000)
print('masked loss at step 500/1000/2000: ' + '/'.join(
    f'{sum(losses[k-50:k]) / 50:.2f}' for k in (500, 1000, 2000)))
```

```{.python .input #encoders-decoders-the-masked-token-objective}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
MASK = len(data.vocab)  # id of the extra <mask> embedding row

def mask_tokens(X, key, p=0.15):
    mask = jax.random.uniform(key, X.shape) < p
    return jnp.where(mask, MASK, X), mask

class MaskedBytes:
    """The windows of data with masked bytes; labels -100 add no loss."""
    def __init__(self, data):
        self.data = data

    def train_dataloader(self):
        return self.masked(self.data.train_dataloader())

    def val_dataloader(self):
        return self.masked(self.data.val_dataloader())

    def masked(self, batches):
        key = d2l.get_key()
        for X, _ in batches:
            key, sub = jax.random.split(key)
            Xm, mask = mask_tokens(jnp.asarray(X), sub)
            yield Xm, jnp.where(mask, X, -100)

class MaskedLM(nnx.Module):
    """An encoder with an output head tied to its token embedding."""
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, X):
        return self.encoder.token_emb.attend(self.encoder(X))

model = MaskedLM(TransformerEncoder(len(data.vocab) + 1,
                                    rngs=nnx.Rngs(d2l.get_key())))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=0.0),
                          wrt=nnx.Param)
losses = d2l.train_steps(model, MaskedBytes(data), optimizer, 2000)
print('masked loss at step 500/1000/2000: ' + '/'.join(
    f'{sum(losses[k-50:k]) / 50:.2f}' for k in (500, 1000, 2000)))
```

### Loss by Available Context
:label:`subsec_mlm-loss-by-position`

In :numref:`subsec_gpt-rnn-baseline`, a causal language model's per-position validation loss decreased monotonically from left ($t = 1$, no context) to right ($t = T$, full prefix context). What does `loss_by_position` look like for a bidirectional `MaskedLM` when we evaluate reconstruction loss at each window position $t \in \{0, \dots, n-1\}$?

```{.python .input #encoders-decoders-what-the-second-side-is-worth-1}
%%tab pytorch
def loss_by_position(model, batches):
    """Mean loss at each window position over the labels that are not -100,
    and the fraction of those labels predicted exactly."""
    device = next(model.parameters()).device
    model.eval()
    total = count = correct = 0
    with torch.no_grad():
        for X, Y in batches:
            X, Y = X.to(device), Y.to(device)
            logits, scored = model(X), Y != -100
            total += F.cross_entropy(logits.transpose(1, 2), Y,
                                     reduction='none').sum(0)  # 0 at -100
            count += scored.sum(0)
            correct += ((logits.argmax(-1) == Y) & scored).sum()
    return (total / count).cpu(), float(correct / count.sum())

def show(losses):
    return ' '.join(f'{float(v):.2f}' for v in losses)

masked = MaskedBytes(data)
pos_loss, accuracy = loss_by_position(model, masked.val_dataloader())
interior = slice(16, 48)  # positions 16 to 47 of the 64
print(f'masked accuracy {accuracy:.2f}')
print(f'loss at positions 0-3: {show(pos_loss[:4])}; 60-63: '
      f'{show(pos_loss[-4:])}; interior mean {pos_loss[interior].mean():.2f}')
```

```{.python .input #encoders-decoders-what-the-second-side-is-worth-1}
%%tab jax
@nnx.jit
def scored_losses(model, X, Y):
    logits = model(X)
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits, jnp.maximum(Y, 0))
    return losses, logits.argmax(-1) == Y

def loss_by_position(model, batches):
    """Mean loss at each window position over the labels that are not -100,
    and the fraction of those labels predicted exactly."""
    model.eval()
    total = count = correct = 0
    for X, Y in batches:
        Y = jnp.asarray(Y)
        losses, hit = scored_losses(model, jnp.asarray(X), Y)
        scored = Y != -100
        total += (losses * scored).sum(0)
        count += scored.sum(0)
        correct += (hit & scored).sum()
    return total / count, float(correct / count.sum())

def show(losses):
    return ' '.join(f'{float(v):.2f}' for v in losses)

masked = MaskedBytes(data)
pos_loss, accuracy = loss_by_position(model, masked.val_dataloader())
interior = slice(16, 48)  # positions 16 to 47 of the 64
print(f'masked accuracy {accuracy:.2f}')
print(f'loss at positions 0-3: {show(pos_loss[:4])}; 60-63: '
      f'{show(pos_loss[-4:])}; interior mean {pos_loss[interior].mean():.2f}')
```

The bidirectional encoder traces a symmetric **bathtub (U-shaped) curve**: every interior position enjoys both left and right context (about 1.5–1.6 nats/byte), and loss rises only at the two window edges ($t \approx 0$ and $t \approx n-1$, about 2.4 nats/byte) where one side of the context is cut off by the window boundary. Evaluating `model` on a masked sentence shows how right context disambiguates missing interior bytes:

```{.python .input #encoders-decoders-what-the-second-side-is-worth-2}
%%tab pytorch
snippet = 'Homarus gammarus , known as the European lobster or common '
ids = torch.tensor(data.vocab.encode(snippet), device=device)[None]
for pos in (11, 22, 28):
    Xm = ids.clone()
    Xm[0, pos] = MASK
    with torch.no_grad():
        logits = model(Xm)
    probs = F.softmax(logits[0, pos], -1)
    top = probs.topk(3)
    shown = snippet[:pos] + '_' + snippet[pos + 1:]
    print(f'{shown[:40]!r}... -> '
          + ', '.join(f'{data.vocab.decode([int(i)])!r} {p:.2f}'
                      for p, i in zip(top.values, top.indices)))
```

```{.python .input #encoders-decoders-what-the-second-side-is-worth-2}
%%tab jax
snippet = 'Homarus gammarus , known as the European lobster or common '
ids = jnp.asarray(data.vocab.encode(snippet))[None]
for pos in (11, 22, 28):
    Xm = ids.at[0, pos].set(MASK)
    probs = jax.nn.softmax(model(Xm)[0, pos])
    top = jnp.argsort(probs)[::-1][:3]
    shown = snippet[:pos] + '_' + snippet[pos + 1:]
    print(f'{shown[:40]!r}... -> '
          + ', '.join(f'{data.vocab.decode([int(i)])!r} {probs[i]:.2f}'
                      for i in top))
```

## Cross-Attention in an Encoder--Decoder

### A Sequence-to-Sequence Task with Known Alignment
:label:`subsec_reversal-task`

To study an **encoder--decoder** (:numref:`fig_three-wirings`, right) on a clean sequence-to-sequence alignment problem, we train on string reversal over an alphabet of $V = 16$ letters (`a`--`p`) and length $T = 12$: given source string $\mathbf{x} = (x_1, \dots, x_T)$, the decoder receives $\textrm{dec\_in} = (\texttt{<bos>}, x_T, x_{T-1}, \dots, x_2)$ and predicts the reversed string $\mathbf{y} = (x_T, x_{T-1}, \dots, x_1)$:

```{.python .input #encoders-decoders-a-task-whose-alignment-we-know}
%%tab pytorch
V, T = 16, 12   # 16 letters, strings of length 12
BOS = V         # decoder start token

def sample_batch(batch_size, device):
    src = torch.randint(0, V, (batch_size, T), device=device)
    tgt = src.flip(1)
    dec_in = torch.cat([torch.full((batch_size, 1), BOS, device=device),
                        tgt[:, :-1]], 1)
    return src, dec_in, tgt

def to_str(ids):
    return ''.join(chr(ord('a') + int(i)) for i in ids)

src, dec_in, tgt = sample_batch(1, device)
print(f'source {to_str(src[0])!r} -> target {to_str(tgt[0])!r}')
```

```{.python .input #encoders-decoders-a-task-whose-alignment-we-know}
%%tab jax
V, T = 16, 12   # 16 letters, strings of length 12
BOS = V         # decoder start token

def sample_batch(batch_size, key):
    src = jax.random.randint(key, (batch_size, T), 0, V)
    tgt = src[:, ::-1]
    dec_in = jnp.concatenate(
        [jnp.full((batch_size, 1), BOS), tgt[:, :-1]], 1)
    return src, dec_in, tgt

def to_str(ids):
    return ''.join(chr(ord('a') + int(i)) for i in ids)

src, dec_in, tgt = sample_batch(1, d2l.get_key())
print(f'source {to_str(src[0])!r} -> target {to_str(tgt[0])!r}')
```

### The Decoder Block and `EncoderDecoder`

A `DecoderBlock` inserts a **cross-attention** sublayer (`d2l.MultiHeadAttention`) between causal self-attention and the SwiGLU FFN. In cross-attention, **queries come from the decoder** ($\mathbf{Q} = \textrm{Norm}(\mathbf{Y})\mathbf{W}_q \in \mathbb{R}^{B \times T_{\textrm{dec}} \times d}$) while **keys and values come from the top of the encoder** ($\mathbf{K} = \mathbf{H}_{\textrm{enc}}\mathbf{W}_k \in \mathbb{R}^{B \times T_{\textrm{src}} \times d}$, $\mathbf{V} = \mathbf{H}_{\textrm{enc}}\mathbf{W}_v \in \mathbb{R}^{B \times T_{\textrm{src}} \times d}$). Because the entire encoded source $\mathbf{H}_{\textrm{enc}}$ exists before decoding begins, cross-attention uses **no causal mask**:

```{.python .input #encoders-decoders-the-decoder-block-one-more-sublayer-1}
%%tab pytorch
class DecoderBlock(nn.Module):
    """Pre-norm decoder block: causal self-attention, cross-attention,
    FFN."""
    def __init__(self, num_hiddens, num_heads):
        super().__init__()
        self.norm1 = nn.RMSNorm(num_hiddens, eps=1e-5)
        self.norm2 = nn.RMSNorm(num_hiddens, eps=1e-5)
        self.norm3 = nn.RMSNorm(num_hiddens, eps=1e-5)
        self.self_attention = d2l.MultiHeadAttention(num_hiddens, num_heads)
        self.cross_attention = d2l.MultiHeadAttention(num_hiddens, num_heads)
        self.ffn = d2l.FeedForward(num_hiddens)

    def forward(self, X, enc, mask=None, cross_mask=None):
        if mask is None:  # causal self-attention
            mask = d2l.causal_mask(X.shape[1], device=X.device)
        Y = self.norm1(X)
        X = X + self.self_attention(Y, Y, Y, mask)
        X = X + self.cross_attention(self.norm2(X), enc, enc, cross_mask)
        return X + self.ffn(self.norm3(X))
```

```{.python .input #encoders-decoders-the-decoder-block-one-more-sublayer-1}
%%tab jax
class DecoderBlock(nnx.Module):
    """Pre-norm decoder block: causal self-attention, cross-attention,
    FFN."""
    def __init__(self, num_hiddens, num_heads, *, rngs):
        self.norm1 = nnx.RMSNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.RMSNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.norm3 = nnx.RMSNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.self_attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                     rngs=rngs)
        self.cross_attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                      rngs=rngs)
        self.ffn = d2l.FeedForward(num_hiddens, rngs=rngs)

    def __call__(self, X, enc, mask=None, cross_mask=None):
        if mask is None:  # causal self-attention
            mask = d2l.causal_mask(X.shape[1])
        Y = self.norm1(X)
        X = X + self.self_attention(Y, Y, Y, mask)
        X = X + self.cross_attention(self.norm2(X), enc, enc, cross_mask)
        return X + self.ffn(self.norm3(X))
```

`EncoderDecoder` pairs a `TransformerEncoder` with a stack of `DecoderBlock` layers and a linear output head. Notice that `decode(dec_in, enc)` takes the precomputed encoder output `enc`, so during autoregressive generation the encoder runs **only once** on the source sequence while `decode` steps through the target tokens:

```{.python .input #encoders-decoders-the-decoder-block-one-more-sublayer-2}
%%tab pytorch
class EncoderDecoder(nn.Module):
    """A causal decoder cross-attending into a bidirectional encoder."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4,
                 num_blks=1, max_len=64):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, num_hiddens,
                                          num_heads, num_blks, max_len)
        self.tgt_emb = nn.Embedding(vocab_size, num_hiddens)
        self.pos_emb = nn.Embedding(max_len, num_hiddens)
        for emb in (self.tgt_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        self.blks = nn.ModuleList([DecoderBlock(num_hiddens, num_heads)
                                   for _ in range(num_blks)])
        self.norm = nn.RMSNorm(num_hiddens, eps=1e-5)
        self.head = nn.Linear(num_hiddens, vocab_size, bias=False)

    def decode(self, dec_in, enc):
        H = self.tgt_emb(dec_in) + self.pos_emb(
            torch.arange(dec_in.shape[1], device=dec_in.device))
        for blk in self.blks:
            H = blk(H, enc)
        return self.head(self.norm(H))

    def forward(self, src, dec_in):
        return self.decode(dec_in, self.encoder(src))
```

```{.python .input #encoders-decoders-the-decoder-block-one-more-sublayer-2}
%%tab jax
class EncoderDecoder(nnx.Module):
    """A causal decoder cross-attending into a bidirectional encoder."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4,
                 num_blks=1, max_len=64, *, rngs):
        init = nnx.initializers.normal(0.02)
        self.encoder = TransformerEncoder(vocab_size, num_hiddens,
                                          num_heads, num_blks, max_len,
                                          rngs=rngs)
        self.tgt_emb = nnx.Embed(vocab_size, num_hiddens,
                                 embedding_init=init, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, num_hiddens, embedding_init=init,
                                 rngs=rngs)
        self.blks = nnx.List([DecoderBlock(num_hiddens, num_heads,
                                           rngs=rngs)
                              for _ in range(num_blks)])
        self.norm = nnx.RMSNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.head = d2l.Linear(num_hiddens, vocab_size, use_bias=False,
                               rngs=rngs)

    def decode(self, dec_in, enc):
        H = self.tgt_emb(dec_in) + self.pos_emb(jnp.arange(dec_in.shape[1]))
        for blk in self.blks:
            H = blk(H, enc)
        return self.head(self.norm(H))

    def __call__(self, src, dec_in):
        return self.decode(dec_in, self.encoder(src))
```

### Training and Autoregressive Decoding

Training `EncoderDecoder` for 300 steps on random batches of 128 strings and running greedy autoregressive decoding (`reverse`) achieves near-perfect exact-string reversal:

```{.python .input #encoders-decoders-training-and-decoding-1}
%%tab pytorch
def train_reversal(model, num_steps=300):
    """Train on fresh batches of 128 strings; return the loss of each step."""
    optimizer = d2l.LMOptimizer(model, lr=1e-3, num_steps=num_steps,
                                warmup=0, weight_decay=0.0)
    losses = []
    for step in range(num_steps):
        src, dec_in, tgt = sample_batch(128, device)
        logits = model(src, dec_in)
        loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())  # Stays on the device
    return torch.stack(losses).tolist()  # One transfer at the end

seq2seq = EncoderDecoder(V + 1).to(device)
losses = train_reversal(seq2seq)
print('loss at step 25/50/100/200/300: ' + '/'.join(
    f'{sum(losses[k-10:k]) / 10:.3f}' for k in (25, 50, 100, 200, 300)))
```

```{.python .input #encoders-decoders-training-and-decoding-1}
%%tab jax
@nnx.jit
def s2s_step(model, optimizer, src, dec_in, tgt):
    def loss_fn(model):
        logits = model(src, dec_in)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def train_reversal(model, num_steps=300):
    """Train on fresh batches of 128 strings; return the loss of each step."""
    optimizer = d2l.LMOptimizer(model, lr=1e-3, num_steps=num_steps,
                                warmup=0, weight_decay=0.0)
    key, losses = d2l.get_key(), []
    for step in range(num_steps):
        key, sub = jax.random.split(key)
        src, dec_in, tgt = sample_batch(128, sub)
        losses.append(s2s_step(model, optimizer, src, dec_in, tgt))
    return jax.device_get(losses)  # One transfer at the end

seq2seq = EncoderDecoder(V + 1, rngs=nnx.Rngs(d2l.get_key()))
losses = train_reversal(seq2seq)
print('loss at step 25/50/100/200/300: ' + '/'.join(
    f'{sum(losses[k-10:k]) / 10:.3f}' for k in (25, 50, 100, 200, 300)))
```

At inference time, `reverse(model, src)` encodes the source sequence once and greedily decodes one target token per step from `<bos>`, achieving near-perfect exact-string match on $1{,}000$ held-out random strings:

```{.python .input #encoders-decoders-training-and-decoding-2}
%%tab pytorch
def reverse(model, src):
    """Greedy decoding: the encoder runs once, the decoder once per token."""
    model.eval()
    with torch.no_grad():
        enc = model.encoder(src)
        out = torch.full((len(src), T + 1), BOS, device=src.device)
        for t in range(T):
            logits = model.decode(out[:, :t + 1], enc)
            out[:, t + 1] = logits[:, -1].argmax(-1)
    return out[:, 1:]

src, _, tgt = sample_batch(1000, device)
pred = reverse(seq2seq, src)
print(f'exact match on 1000 fresh strings: '
      f'{(pred == tgt).all(-1).float().mean():.3f}')
print(f'source {to_str(src[0])!r} -> predicted {to_str(pred[0])!r}')
```

```{.python .input #encoders-decoders-training-and-decoding-2}
%%tab jax
@nnx.jit
def next_token(model, enc, buf, t):
    return model.decode(buf, enc)[:, t].argmax(-1)

def reverse(model, src):
    """Greedy decoding: the encoder runs once, the decoder once per token."""
    model.eval()
    enc = model.encoder(src)
    buf = jnp.full((len(src), T + 1), BOS)
    for t in range(T):
        buf = buf.at[:, t + 1].set(next_token(model, enc, buf, t))
    return buf[:, 1:]

src, _, tgt = sample_batch(1000, d2l.get_key())
pred = reverse(seq2seq, src)
print(f'exact match on 1000 fresh strings: '
      f'{(pred == tgt).all(-1).mean():.3f}')
print(f'source {to_str(src[0])!r} -> predicted {to_str(pred[0])!r}')
```

### Reading Cross-Attention Alignment
:label:`subsec_reading-alignment`

In string reversal, target step $t$ outputs source token $T - 1 - t$. Does the cross-attention weight matrix therefore have to concentrate on the anti-diagonal $(t, T - 1 - t)$? Let us compare our trained `EncoderDecoder`, with one encoder block and one decoder block (`with` encoder self-attention), against an otherwise identical model whose encoder has no blocks, only its embeddings (`without`, so encoder position $j$ can only hold information about source token $j$):

```{.python .input #encoders-decoders-reading-the-alignment-1}
%%tab pytorch
def reversal_model(self_attention=True):
    model = EncoderDecoder(V + 1).to(device)
    if not self_attention:  # the encoder keeps embeddings and positions
        model.encoder = TransformerEncoder(V + 1, num_blks=0).to(device)
    train_reversal(model)
    return model

def alignment(model):
    """Exact match on 1000 fresh strings; the head-averaged cross-attention
    of 256 strings, and how often its largest weight in a row falls on the
    true source position."""
    src, _, tgt = sample_batch(1000, device)
    match = (reverse(model, src) == tgt).all(-1).float().mean()
    with torch.no_grad():
        src, dec_in, _ = sample_batch(256, device)
        model(src, dec_in)
    maps = model.blks[0].cross_attention.attention_weights.mean(1)
    want = torch.arange(T - 1, -1, -1, device=device)  # row t: column T-1-t
    return match, (maps.argmax(-1) == want).float().mean(), maps[0].cpu()
```

```{.python .input #encoders-decoders-reading-the-alignment-1}
%%tab jax
def reversal_model(self_attention=True):
    model = EncoderDecoder(V + 1, rngs=nnx.Rngs(d2l.get_key()))
    if not self_attention:  # the encoder keeps embeddings and positions
        model.encoder = TransformerEncoder(V + 1, num_blks=0,
                                           rngs=nnx.Rngs(d2l.get_key()))
    train_reversal(model)
    return model

def alignment(model):
    """Exact match on 1000 fresh strings; the head-averaged cross-attention
    of 256 strings, and how often its largest weight in a row falls on the
    true source position."""
    src, _, tgt = sample_batch(1000, d2l.get_key())
    match = (reverse(model, src) == tgt).all(-1).mean()
    src, dec_in, _ = sample_batch(256, d2l.get_key())
    model(src, dec_in)
    maps = model.blks[0].cross_attention.attention_weights.mean(1)
    want = jnp.arange(T - 1, -1, -1)  # row t: column T-1-t
    return match, (maps.argmax(-1) == want).mean(), maps[0]
```

Running six independent seeds of both `with` and `without` encoder self-attention and plotting the least anti-diagonally aligned cross-attention heatmap from each group contrasts direct positional lookup against multi-layer routing:

```{.python .input #encoders-decoders-reading-the-alignment}
%%tab pytorch, jax
encoders = {'with': [seq2seq] + [reversal_model() for _ in range(5)],
            'without': [reversal_model(False) for _ in range(6)]}
maps, titles = [], []
for name, models in encoders.items():
    runs = [alignment(m) for m in models]
    print(f'encoder {name} self-attention, six runs')
    print('  exact match:', ' '.join(f'{m:.3f}' for m, _, _ in runs))
    print('  rows on the true source position:',
          ' '.join(f'{100 * h:.0f}%' for _, h, _ in runs))
    _, hit, least = min(runs, key=lambda run: run[1])  # the least aligned
    maps.append(least)
    titles.append(f'{name}: {100 * hit:.0f}% of rows')
d2l.show_heatmaps(d2l.stack(maps)[None], xlabel='source position',
                  ylabel='target position', titles=titles, figsize=(6, 3),
                  cmap='Blues')
```

When the encoder has no self-attention (`without`), cross-attention *must* look directly at the anti-diagonal $(t, T - 1 - t)$ to retrieve source token $T - 1 - t$. When the encoder includes bidirectional self-attention (`with`), the encoder layers can already route or reorganize source information across positions before cross-attention reads $\mathbf{H}_{\textrm{enc}}$—a classic reminder :cite:`Jain.Wallace.2019` that raw attention weights in a multi-layer network reflect inter-layer routing rather than a direct token-to-token attribution map.

## Cross-Attention as an Interface: Fixed-Size Latent Bottlenecks

Because cross-attention requires only that queries $\mathbf{Q} \in \mathbb{R}^{B \times M \times d}$ and key--values $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times N \times d}$ share feature width $d$, it can decouple internal processing length $M$ from raw input length $N$ (:numref:`fig_latent-bottleneck`).

![A latent bottleneck with input length $N$, $M$ learned latents, width $d$, and $L$ latent blocks. Every latent reads every input (shaded band); the read of one latent is drawn in full. One cross-attention read has mixing cost $O(MNd)$; each subsequent latent self-attention layer has mixing cost $O(M^2d)$, for $O(LM^2d)$ across the stack. These terms hold batch size, head count, and width fixed and omit projections and FFNs.](../img/mdl-transformers-latent-bottleneck.svg)
:label:`fig_latent-bottleneck`

In **Perceiver** :cite:`Jaegle.Gimeno.Brock.ea.2021` (and multimodal vision--language resamplers :cite:`alayrac2022flamingo`), a learned array of $M \ll N$ latent query vectors $\mathbf{Z}_0 \in \mathbb{R}^{M \times d}$ cross-attends to an $N$-token input $\mathbf{X} \in \mathbb{R}^{B \times N \times d}$ in $O(M N d)$ operations, after which $L$ self-attention blocks process the compact latent array of length $M$ in $O(L M^2 d)$ operations instead of $O(L N^2 d)$:

```{.python .input #encoders-decoders-queries-need-not-come-from-a-sequence}
%%tab pytorch
class PerceiverEncoder(nn.Module):
    """M learned latents cross-attend into the input, then process among
    themselves."""
    def __init__(self, num_latents, num_hiddens, num_heads=4, num_blks=2):
        super().__init__()
        self.latents = nn.Parameter(
            0.02 * torch.randn(num_latents, num_hiddens))
        self.norm_q = nn.RMSNorm(num_hiddens, eps=1e-5)
        self.cross_attention = d2l.MultiHeadAttention(num_hiddens, num_heads)
        self.blks = nn.ModuleList([
            d2l.TransformerBlock(num_hiddens, num_heads)
            for _ in range(num_blks)])
        self.norm = nn.RMSNorm(num_hiddens, eps=1e-5)

    def forward(self, X):
        Z = self.latents.expand(X.shape[0], -1, -1)
        Z = Z + self.cross_attention(self.norm_q(Z), X, X)  # O(MN)
        for blk in self.blks:
            Z = blk(Z)                                       # O(M^2)
        return self.norm(Z)
```

```{.python .input #encoders-decoders-queries-need-not-come-from-a-sequence}
%%tab jax
class PerceiverEncoder(nnx.Module):
    """M learned latents cross-attend into the input, then process among
    themselves."""
    def __init__(self, num_latents, num_hiddens, num_heads=4, num_blks=2,
                 *, rngs):
        self.latents = nnx.Param(
            0.02 * jax.random.normal(rngs.params(),
                                     (num_latents, num_hiddens)))
        self.norm_q = nnx.RMSNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.cross_attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                      rngs=rngs)
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, rngs=rngs)
            for _ in range(num_blks)])
        self.norm = nnx.RMSNorm(num_hiddens, epsilon=1e-5, rngs=rngs)

    def __call__(self, X):
        Z = jnp.broadcast_to(self.latents[...],
                             (X.shape[0],) + self.latents.shape)
        Z = Z + self.cross_attention(self.norm_q(Z), X, X)  # O(MN)
        for blk in self.blks:
            Z = blk(Z)                                       # O(M^2)
        return self.norm(Z)
```

## Choosing Among the Three Wirings
:label:`subsec_encdec-choosing`

* **Encoder-only (`is_causal=False`)** gives every token full bidirectional context in a single pass, making it the standard backbone for dense text embeddings, reranking, token classification, and non-autoregressive vision backbones (:numref:`sec_vision-transformer`).
* **Decoder-only (`is_causal=True`)** trains on $100\%$ of sequence tokens via next-token prediction (versus $15\%$ in MLM) and processes prompts and generated continuations in a single unified parameter stack and KV cache (:numref:`sec_kv-cache`), making it the dominant architecture for general-purpose language models.
* **Encoder--decoder (with cross-attention)** shines when the input and output have different modalities or lengths—such as speech spectrograms to text in Whisper :cite:`radford2023whisper`, or high-resolution visual/sensor arrays compressed through a fixed-size latent cross-attention interface (:numref:`fig_latent-bottleneck`).

## Summary

* The same `TransformerBlock` can be wired as a **bidirectional encoder** (`is_causal=False`), a **causal decoder** (`is_causal=True`), or a **cross-attending encoder--decoder** (:numref:`fig_three-wirings`).
* Because unmasked multi-layer self-attention leaks future tokens under next-token prediction, bidirectional encoders are trained with **Masked Language Modeling (MLM)** (:eqref:`eq_mlm`), producing a symmetric U-shaped `loss_by_position` curve across the window.
* In an **encoder--decoder**, each `DecoderBlock` interleaves causal target self-attention with **cross-attention** (decoder queries attending to encoder keys and values) and a position-wise FFN; during generation, the encoder runs once on the source while the decoder steps autoregressively.
* **Cross-attention** with $M \ll N$ learned query vectors (`PerceiverEncoder`) compresses an $N$-token input into an $M$-token latent bottleneck in $O(M N d + L M^2 d)$ time (:numref:`fig_latent-bottleneck`).

## Exercises

1. **BERT's 80 / 10 / 10 masking recipe.** `mask_tokens` replaces every selected position ($15\%$ of bytes) with `<mask>`. Because `<mask>` never appears at fine-tuning or inference time, :citet:`Devlin.Chang.Lee.ea.2018` replace selected tokens with `<mask>` $80\%$ of the time, with a uniformly random vocabulary token $10\%$ of the time, and leave the original token unchanged $10\%$ of the time. Implement this 80 / 10 / 10 rule in `mask_tokens` and explain why keeping $10\%$ unmasked and $10\%$ random forces the encoder to maintain accurate representations at *unmasked* tokens as well as `<mask>` tokens.
:exlabel:`ex_encdec-masking-recipe`

2. **Padding masks on ragged sequence-to-sequence batches.** When a batch contains variable-length source strings of lengths $(s_1, \dots, s_B) \le S$ and target strings of lengths $(t_1, \dots, t_B) \le T_{\textrm{dec}}$, three boolean masks are needed: (a) a $(B, 1, S, S)$ encoder padding mask, (b) a $(B, 1, T_{\textrm{dec}}, T_{\textrm{dec}})$ decoder causal + padding mask, and (c) a $(B, 1, T_{\textrm{dec}}, S)$ rectangular cross-attention source-padding mask. Write a helper `make_seq2seq_masks(src_len, tgt_len, S, T_dec)` that constructs all three masks and verify that appending padded tokens to `src` in `EncoderDecoder` does not alter the non-padded decoder logits.
:exlabel:`ex_encdec-padding`

3. **Varying the MLM masking fraction $p$.** Train `MaskedLM` for 2,000 steps at masking fractions $p \in \{0.05, 0.15, 0.30, 0.50\}$ and evaluate all four models on the same $p = 0.15$ validation set `masked.val_dataloader()`. How does $p$ trade off the number of supervised gradient signals per batch against the amount of uncorrupted context available around each mask?

4. **Forward vs. backward direction of time in causal and bidirectional models.** A causal language model processes text left-to-right ($x_1 \to x_n$). Reverse every byte window in `WikiText103` so a causal model predicts right-to-left ($x_n \to x_1$), train for 1,000 steps, and compare its average per-byte validation loss and its `loss_by_position` curve against the left-to-right causal model and the bidirectional `MaskedLM`. Why is the total joint log-likelihood $\sum_{t=1}^n \log P(x_t \mid \textrm{past})$ theoretically invariant to time reversal, and where does the bidirectional model gain its advantage over both one-sided directions?
:exlabel:`ex_encdec-times-arrow`

5. **Content-addressed vs. position-addressed cross-attention alignment.** In our string reversal experiment, every letter in a random string can repeat, so position is required to distinguish duplicate letters. Modify `sample_batch` so each length-12 string is a random permutation of 12 *distinct* letters, and train `EncoderDecoder` without positional embeddings in the encoder (`pos_emb.weight.zero_()`). Why does a 1-layer encoder without position embeddings fail on string reversal even when all 12 letters are distinct, whereas a sorting task (outputting the input letters in alphabetical order `a..p`) succeeds without any encoder positional embeddings?
:exlabel:`ex_encdec-depth-heads`

6. **Perceiver IO for sequence-to-sequence reversal.** Perceiver IO :cite:`Jaegle.Borgeaud.Alayrac.ea.2022` pairs a `PerceiverEncoder` ($M$ learned latent queries attending to the $N$-token input) with a cross-attention **decoder** in which $N_{\textrm{out}}$ target position embeddings $\mathbf{Q}_{\textrm{out}} \in \mathbb{R}^{N_{\textrm{out}} \times d}$ cross-attend to the $M$ latent vectors $\mathbf{Z}^{(L)} \in \mathbb{R}^{B \times M \times d}$. Build a `PerceiverAutoencoder` that encodes a 12-letter string into $M \in \{2, 4, 8, 12\}$ latents ($d = 128$) and decodes the reversed string in one parallel cross-attention readout pass. How does reversal accuracy vary with the bottleneck size $M$?
:exlabel:`ex_encdec-perceiver-reversal`

[Discussions](https://d2l.discourse.group/t/36777)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.4]{.kicker}

Bidirectional Encoders, Encoder–Decoders, and Cross-Attention<br>**three block wirings · masked language modeling · sequence-to-sequence cross-attention · Perceiver latent bottlenecks**
:::
:::

::: {.slide title="Why Causal Masking Hurts When Full Input Is Known"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Cloze disambiguation:** In `Homarus gammarus, known as _he European lobster`, the left prefix `known as ` permits hundreds of completions, whereas the right context `he European` uniquely pins the missing byte to `t` ($96\text{--}97\%$ probability).
- **Two-sided context gain:** On WikiText-103 byte windows, reconstructing a masked byte drops from **$\sim 2.4\text{ nats}$** at the 1-sided window boundary to **$\approx 1.55\text{ nats}$** in the 2-sided interior.
:::

::: {.col .fig .big}
@fig:ch11-tf-enc-bidirectional-cloze
:::
:::
:::

::: {.slide title="One Transformer Block, Three Canonical Wirings"}
[Architectural Taxonomy]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Encoder-only (`is_causal=False`):** Every token attends to every token across all $L$ layers in a single bidirectional pass (BERT, RoBERTa, ViT).
- **2. Decoder-only (`is_causal=True`):** Each token attends only to positions $j \le i$, handling prompt reading and generation in one homogeneous stack (GPT, Llama 3, Qwen3).
- **3. Encoder–decoder:** A bidirectional encoder maps source $\mathbf{X}_{\text{src}} \to \mathbf{H}_{\text{enc}}$, and a causal decoder generates $\mathbf{Y}_{\text{tgt}}$ while reading $\mathbf{H}_{\text{enc}}$ via **cross-attention** (2017 Transformer, T5, Whisper).
:::

::: {.col .fig}
@fig:mdl-transformers-three-wirings
:::
:::
:::

::: {.slide title="A Bidirectional Encoder Drops the Causal Mask" only="pytorch"}
[Encoder Implementation]{.kicker}

Removing the triangular mask from `d2l.TransformerBlock` turns a causal stack into `TransformerEncoder`, mapping an $(B, n)$ batch of token IDs to bidirectional contextual representations in $\mathbb{R}^{B \times n \times d}$:

```{.python #encoders-decoders-exercises-1}
def forward(self, X, mask=None):          # X: (batch, n) token ids
    H = self.token_emb(X) + self.pos_emb(
        torch.arange(X.shape[1], device=X.device))
    for blk in self.blks:                 # Pre-norm d2l.TransformerBlock stack
        H = blk(H, mask)                  # mask=None -> all tokens attend to all
    return self.norm(H)                   # (batch, n, d) bidirectional embeddings
```

::: {.d2l-note}
Every output vector $\mathbf{h}_i^{(L)} \in \mathbb{R}^d$ summarizes the entire sequence from position $i$'s vantage point; pooling across positions (mean pooling or reading index 0) produces a single fixed-width embedding per document.
:::
:::

::: {.slide title="A Bidirectional Encoder Drops the Causal Mask" only="jax"}
[Encoder Implementation]{.kicker}

Removing the triangular mask from `d2l.TransformerBlock` turns a causal stack into `TransformerEncoder`, mapping an $(B, n)$ batch of token IDs to bidirectional contextual representations in $\mathbb{R}^{B \times n \times d}$:

```{.python #encoders-decoders-exercises-2}
def __call__(self, X, mask=None):         # X: (batch, n) token ids
    H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
    for blk in self.blks:                 # Pre-norm d2l.TransformerBlock stack
        H = blk(H, mask)                  # mask=None -> all tokens attend to all
    return self.norm(H)                   # (batch, n, d) bidirectional embeddings
```

::: {.d2l-note}
Every output vector $\mathbf{h}_i^{(L)} \in \mathbb{R}^d$ summarizes the entire sequence from position $i$'s vantage point; pooling across positions (mean pooling or reading index 0) produces a single fixed-width embedding per document.
:::
:::

::: {.slide title="Pitfall: Why Encoders Cannot Train on Next-Token Loss"}
[Information Leakage Hazard]{.kicker}

::: {.cols .vc}
::: {.col}
- **Layer 1 direct leakage:** Query $t$ attends directly to key $t+1$, copying target embedding $\mathbf{e}_{x_{t+1}}$ into $\mathbf{h}_t^{(1)}$.
- **Layer 2 multi-hop leakage:** Even if $(t, t+1)$ were masked, query $t$ at layer 2 attends to position $t+2$ (or $t-1$), which already read $x_{t+1}$ in layer 1!
- **Rule:** Any target token $x_i$ ($i \in \mathcal{M}$) must be replaced by `<mask>` in the input before layer 1!
:::

::: {.col .fig .big}
@fig:ch11-tf-enc-multihop-leakage
:::
:::
:::

::: {.slide title="Masked Language Modeling (MLM)"}
[Bidirectional Pretraining]{.kicker}

Introduced in **BERT** (Devlin et al., 2018), **Masked Language Modeling** samples a random subset of positions $\mathcal{M} \subset \{1, \dots, n\}$ (with probability $p = 0.15$), replaces each $x_i$ ($i \in \mathcal{M}$) with a dedicated `<mask>` token ID in $\tilde{\mathbf{x}}$, and scores cross-entropy *only* on the masked subset $\mathcal{M}$:

$$\mathcal{L}_{\mathrm{MLM}}(\boldsymbol{\theta}) = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P_{\boldsymbol{\theta}}(x_i \mid \tilde{\mathbf{x}})$$

@!encoders-decoders-the-masked-token-objective

- **Label masking (`-100`):** Unmasked positions ($i \notin \mathcal{M}$, $85\%$ of tokens) receive target label `-100` so they contribute context without adding trivial copy terms to the loss.
- **Sample efficiency trade-off:** Each training window supervises only $15\%$ of its tokens (around $10$ bytes out of $64$), compared to $100\%$ of tokens in a causal GPT.
:::

::: {.slide title="The U-Shaped Loss Curve Across Window Positions"}
[Value of Two-Sided Context]{.kicker}

Evaluating our trained 4-block byte-level `MaskedLM` across window positions $t \in \{0, \dots, 63\}$ contrasts sharply with a causal model's monotonically decreasing curve:

@!encoders-decoders-what-the-second-side-is-worth-1

- **Symmetric bathtub (U-shaped) profile:** Interior positions ($t \in [16, 47]$) enjoy both left and right context, achieving a flat mean loss of **$\approx 1.55\text{ nats/byte}$** ($\approx 53\%$ exact byte accuracy).
- **Edge penalty at $t = 0$ and $t = 63$:** At the two window boundaries where one side of the context is truncated, loss jumps to **$2.36\text{--}2.44\text{ nats/byte}$**—quantifying the $\sim 0.85\text{-nat}$ advantage of having the second side of context!
:::

::: {.slide title="How Right-Hand Context Resolves Ambiguous Gaps"}
[Qualitative Probing]{.kicker}

::: {.cols .vc}
::: {.col}
- Masking individual bytes in `'Homarus gammarus , known as the European lobster'` shows how bidirectional attention fuses left and right spelling constraints:
- At position 28 (`known as _he European`), the encoder assigns $96\text{--}97\%$ probability to `'t'`; at position 22 (`kno_n as`), it assigns $47\text{--}66\%$ to `'w'`.

@encoders-decoders-what-the-second-side-is-worth-2
:::

::: {.col .fig}
@fig:ch11-tf-enc-bathtub-curve
:::
:::
:::

::: {.slide title="Sequence-to-Sequence Alignment: String Reversal"}
[Encoder–Decoder Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
- **Teacher-forced decoder input (`dec_in`):** Prepends `<bos>` (`id = 16`) and shifts the reversed target right by one slot: `dec_in` $= (\texttt{<bos>}, x_T, \dots, x_2)$ predicts `tgt` $= (x_T, \dots, x_1)$.
- **Exact ground-truth alignment:** To emit target position $t$, the decoder must retrieve source token $x_{T - 1 - t}$ along the **anti-diagonal** $(t, T - 1 - t)$.

@encoders-decoders-a-task-whose-alignment-we-know
:::

::: {.col .fig}
@fig:ch11-tf-enc-string-reversal
:::
:::
:::

::: {.slide title="The Decoder Block: Adding a Cross-Attention Sublayer"}
[Decoder Block Anatomy]{.kicker}

::: {.cols .vc}
::: {.col}
- Each `DecoderBlock` contains **three** pre-norm residual sublayers on target stream $\mathbf{X} \in \mathbb{R}^{B \times T_{\text{dec}} \times d}$ given encoded source $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times T_{\text{src}} \times d}$:
  $$\mathbf{U} = \mathbf{X} + \mathrm{SelfAttn}_{\mathrm{causal}}(\mathrm{Norm}_1(\mathbf{X}))$$
  $$\mathbf{V} = \mathbf{U} + \mathrm{CrossAttn}(\mathrm{Norm}_2(\mathbf{U}), \mathbf{H}_{\mathrm{enc}}, \mathbf{H}_{\mathrm{enc}})$$
  $$\mathbf{Y} = \mathbf{V} + \mathrm{FFN}(\mathrm{Norm}_3(\mathbf{V}))$$
:::

::: {.col .fig}
@fig:ch11-tf-enc-decoder-block
:::
:::
:::

::: {.slide title="Implementing DecoderBlock in Code" only="pytorch"}
[Implementation]{.kicker}

Notice that `self_attention` applies a causal mask over the target prefix, whereas `cross_attention` attends over the entire encoded source `enc` (using only an optional source padding `cross_mask`):

```{.python #encoders-decoders-exercises-3}
def forward(self, X, enc, mask=None, cross_mask=None):
    if mask is None:                      # Causal mask for target self-attention
        mask = d2l.causal_mask(X.shape[1], device=X.device)
    Y = self.norm1(X)
    X = X + self.self_attention(Y, Y, Y, mask)
    X = X + self.cross_attention(self.norm2(X), enc, enc, cross_mask)
    return X + self.ffn(self.norm3(X))
```

::: {.d2l-note}
During autoregressive inference, `enc = self.encoder(src)` and its cross-attention key–value projections depend only on the source sequence—so they are computed **once per prompt** and reused at every target decoding step!
:::
:::

::: {.slide title="Implementing DecoderBlock in Code" only="jax"}
[Implementation]{.kicker}

Notice that `self_attention` applies a causal mask over the target prefix, whereas `cross_attention` attends over the entire encoded source `enc` (using only an optional source padding `cross_mask`):

```{.python #encoders-decoders-exercises-4}
def __call__(self, X, enc, mask=None, cross_mask=None):
    if mask is None:                      # Causal mask for target self-attention
        mask = d2l.causal_mask(X.shape[1])
    Y = self.norm1(X)
    X = X + self.self_attention(Y, Y, Y, mask)
    X = X + self.cross_attention(self.norm2(X), enc, enc, cross_mask)
    return X + self.ffn(self.norm3(X))
```

::: {.d2l-note}
During autoregressive inference, `enc = self.encoder(src)` and its cross-attention key–value projections depend only on the source sequence—so they are computed **once per prompt** and reused at every target decoding step!
:::
:::

::: {.slide title="Training and Greedy Decoding on String Reversal"}
[End-to-End Seq2Seq]{.kicker}

Training a 1-block `EncoderDecoder` on fresh random batches of $128$ strings drives cross-entropy from $\ln 16 \approx 2.77\text{ nats}$ to $0.002\text{ nats}$ in $300$ steps, achieving $100\%$ exact-string match on $1{,}000$ unseen test strings:

@!encoders-decoders-training-and-decoding-1

@!encoders-decoders-training-and-decoding-2
:::

::: {.slide title="Composing Masks Across the Three Attention Layers"}
[Masking Rules]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Encoder self-attention $(S \times S)$:** Source padding mask ($j < s_b$).
- **2. Decoder self-attention $(T_{\text{dec}} \times T_{\text{dec}})$:** Causal lower-triangle $\land$ target padding mask.
- **3. Cross-attention $(T_{\text{dec}} \times S)$:** Source padding mask **only**—never apply a causal mask to cross-attention, since target token $t = 0$ must read source token $S - 1$ immediately!
:::

::: {.col .fig .big}
@fig:ch11-tf-enc-three-masks
:::
:::
:::

::: {.slide title="Pitfall: Do Cross-Attention Weights Show Attribution?"}
[Interpretability Pitfall]{.kicker}

::: {.cols .vc}
::: {.col}
- **Without encoder self-attention (`without`, right heatmap):** Encoder slot $j$ holds only token $x_j$, so cross-attention is forced onto the exact anti-diagonal $(t, T-1-t)$ ($100\%$ of rows).
- **With encoder self-attention (`with`, left heatmap):** Both models achieve $100\%$ reversal accuracy, yet cross-attention aligns with the anti-diagonal on as few as $33\%$ of rows because the encoder's self-attention layer can already permute or fold tokens across positions!
:::

::: {.col .fig}
@!encoders-decoders-reading-the-alignment
:::
:::
:::

::: {.slide title="Cross-Attention as a Fixed-Size Latent Bottleneck"}
[Perceiver Architecture]{.kicker}

::: {.cols .vc}
::: {.col}
- **Decoupling compute from input length $N$:** In `CrossAttn`, the output sequence length equals the number of **queries $M$**, not the number of **keys/values $N$**.
- **Perceiver (Jaegle et al., 2021):** Uses $M \ll N$ learned latent query vectors $\mathbf{Z}_0 \in \mathbb{R}^{M \times d}$ to cross-attend into an $N$-token input in **$O(M N d)$** time, then runs $L$ self-attention blocks on the short latent array in **$O(L M^2 d)$** time—eliminating the $O(L N^2 d)$ bottleneck!
:::

::: {.col .fig}
@fig:mdl-transformers-latent-bottleneck
:::
:::
:::

::: {.slide title="Implementing PerceiverEncoder in Ten Lines" only="pytorch"}
[Latent Bottleneck Code]{.kicker}

::: {.cols .vc}
::: {.col}
```{.python #encoders-decoders-exercises-5}
def forward(self, X):
    Z = self.latents.expand(X.shape[0], -1, -1)
    Z = Z + self.cross_attention(
        self.norm_q(Z), X, X)   # O(M * N) read
    for blk in self.blks:
        Z = blk(Z)              # O(M^2) blocks
    return self.norm(Z)
```
:::

::: {.col .fig}
@fig:ch11-tf-enc-perceiver-flow
:::
:::
:::

::: {.slide title="Implementing PerceiverEncoder in Ten Lines" only="jax"}
[Latent Bottleneck Code]{.kicker}

::: {.cols .vc}
::: {.col}
```{.python #encoders-decoders-exercises-6}
def __call__(self, X):
    Z = jnp.broadcast_to(self.latents[...],
        (X.shape[0],) + self.latents.shape)
    Z = Z + self.cross_attention(
        self.norm_q(Z), X, X)   # O(M * N) read
    for blk in self.blks:
        Z = blk(Z)              # O(M^2) blocks
    return self.norm(Z)
```
:::

::: {.col .fig}
@fig:ch11-tf-enc-perceiver-flow
:::
:::
:::

::: {.slide title="Choosing Among the Three Wirings in Modern Systems"}
[Architectural Decision Guide]{.kicker}

::: {.cols .vc}
::: {.col}
- **Encoder-only:** Full 2-sided context in a single parallel pass (embeddings, rerankers, ViT).
- **Decoder-only:** Scores $100\%$ of tokens per pass with one unified stack and KV cache (general LLMs, code, chat).
- **Encoder–decoder:** Encodes a long source signal once and decodes compact targets via cross-attention (Whisper speech-to-text, Perceiver resamplers).
:::

::: {.col .fig .big}
@fig:ch11-tf-enc-wiring-decision
:::
:::
:::

::: {.slide title="Concept Check: Masking, Complexity, and Wiring Choice"}
[Check Your Understanding]{.kicker}

Work through these three architectural questions:

1. **Perceiver score count:** Suppose an audio or image array has $N = 8{,}192$ tokens and a `PerceiverEncoder` uses $M = 64$ learned latents and $L = 6$ latent self-attention blocks. Compare the number of attention score pairs per head in the Perceiver against a 6-layer full self-attention encoder.
2. **Cross-attention caching:** During autoregressive decoding of $T_{\text{dec}} = 100$ target tokens from an encoded source of length $T_{\text{src}} = 1{,}000$, how many times are the cross-attention key and value projections $\mathbf{H}_{\text{enc}}\mathbf{W}_k, \mathbf{H}_{\text{enc}}\mathbf{W}_v$ computed?
3. **Architecture selection:** You need to embed $50\text{ million}$ product descriptions into $768$-dimensional vectors for nearest-neighbor search. Which wiring do you choose, and why?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Full encoder:** $6 \times 8192^2 \approx \mathbf{4.03 \times 10^8}$ score pairs/head. **Perceiver:** $64 \times 8192 + 6 \times 64^2 = 524{,}288 + 24{,}576 \approx \mathbf{5.49 \times 10^5}$ score pairs/head (**$734\times$ fewer**!).
2. **Exactly once** (right after the encoder finishes), because $\mathbf{H}_{\text{enc}}$ never changes as target tokens are generated.
3. **Encoder-only:** every token benefits from full bidirectional context, and each description is embedded in a single parallel forward pass with no autoregressive decoding loop.
:::
:::

::: {.slide title="Summary: Encoders, Decoders, and Cross-Attention"}
[Key Takeaways]{.kicker}

::: {.cols .vc}
::: {.col}
- **Bidirectional Encoder + MLM:** Unmasked self-attention reads full left and right context; replacing $15\%$ of tokens with `<mask>` prevents multi-layer target leakage.
- **Cross-Attention:** Bridges separate source and target stacks ($\mathbf{Q}$ from target, $\mathbf{K}, \mathbf{V}$ from $\mathbf{H}_{\text{enc}}$).
- **Perceiver Latent Bottleneck:** $M \ll N$ learned queries reduce attention cost from $O(L N^2 d)$ to $O(M N d + L M^2 d)$.
:::

::: {.col .fig .big}
@fig:ch11-tf-enc-summary
:::
:::
:::
