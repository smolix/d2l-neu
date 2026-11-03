# Encoders, Decoders, and Cross-Attention
:label:`sec_transformer`

The GPT of :numref:`sec_gpt` stacks transformer blocks under a causal mask.
The mask makes next-token prediction possible, but it restricts each
representation to the preceding context. In this section the mask, and the
sources of the queries, keys, and values, become design choices.
Removing the mask gives an *encoder*, which reads in both directions and produces
contextual representations. Joining a masked stack to an unmasked one
through the cross-attention of
:numref:`sec_multihead-attention` gives the *encoder--decoder* that
transformers started as :cite:`Vaswani.Shazeer.Parmar.ea.2017`. We build
both from the same `d2l.TransformerBlock` and check whether a learned
cross-attention map reproduces a known alignment. We then turn to learned
queries that do not come from an input sequence. Such queries connect a
computation of fixed size to data of variable size, as in Perceiver-style
and vision--language models.

```{.python .input #encoders-decoders-encoders-decoders-and-cross-attention}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #encoders-decoders-encoders-decoders-and-cross-attention}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import optax
```

## Encoder, Decoder, and Encoder--Decoder Architectures

A transformer block maps a sequence of $d$-dimensional vectors to a
sequence of the same shape. It leaves two questions open: which
positions may attend to which, and where the keys and values come from.
Three common wirings answer these questions.
:numref:`fig_three-wirings` draws them in the convention we will use for
every attention map in this section: queries down the rows, keys along
the columns.

![The three wirings and their attention patterns (query position runs downward, key position rightward; a filled cell means the query may attend to that key). Encoder-only: every token attends to every token. Decoder-only: each token attends only to itself and its past. Encoder--decoder: the source attends to itself bidirectionally, while each target token attends to the full source through cross-attention (light cells) and to its own past causally.](../img/mdl-transformers-three-wirings.svg)
:label:`fig_three-wirings`

**Encoder-only.** Drop the mask. Every token attends to every other, so
each output vector summarizes the *whole* input as seen from its position.
Such a model does not directly implement the left-to-right autoregressive
factorization: with the future visible, next-token prediction reduces to
copying. (Masked models can still generate text by iterative re-masking,
the idea that text diffusion models develop.) But the encoder is the
strongest way to *represent* an input, and one representation per token is
what classification, retrieval, and tagging consume. BERT is this
wiring pretrained on text
:cite:`Devlin.Chang.Lee.ea.2018`; the vision transformer of
:numref:`sec_vision-transformer` is the same wiring over image patches.

**Decoder-only.** Keep the causal mask everywhere: this is the GPT of
:numref:`sec_gpt`. Each token predicts its successor, and generation is the
training objective run forward. One stack is trained with one objective,
next-token prediction on raw text, and the same stack serves generation.
Large enough models of this kind perform tasks that are specified only in
the prompt :cite:`brown2020language`. We do not rebuild this wiring here.
In :numref:`fig_three-wirings` it is the pattern against which the other
two are compared.

**Encoder--decoder.** The original transformer
:cite:`Vaswani.Shazeer.Parmar.ea.2017` combines the two: a bidirectional
encoder reads the source sequence, and a causal decoder generates the
target while *cross-attending* into the encoder's output. This is the
cross-attention of :numref:`sec_multihead-attention`, with queries from the
target stream and keys and values from the source. The pattern in
:numref:`fig_three-wirings` shows the division: a full square for the
source, a full rectangle for target-to-source cross-attention, a triangle
for the target's own past. The wiring fits tasks whose input is fully known
before generation starts. Machine translation was the founding
application. T5 pretrained the architecture on span reconstruction and
recast a broad family of tasks as text-to-text :cite:`raffel2020exploring`,
and BART pretrained it on denoising corrupted text :cite:`lewis2019bart`. In speech
recognition, Whisper's encoder reads an entire audio clip bidirectionally
while a text decoder cross-attends into it :cite:`radford2023whisper`.

The rest of this section builds the two wirings that :numref:`sec_gpt` did
not: first the encoder, then the encoder--decoder.

## Bidirectional Context for Masked-Token Prediction

### A Bidirectional Encoder

The encoder differs from the `CharLM` of :numref:`sec_transformer-block`
in one respect: its blocks receive no mask, so every token attends to
every token. Attention is permutation equivariant
(:numref:`sec_positional-information`), so the model must still supply
positions, and the learned position table stays.

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

    def forward(self, X):
        H = self.token_emb(X) + self.pos_emb(
            torch.arange(X.shape[1], device=X.device))
        for blk in self.blks:
            H = blk(H)  # no mask: every token attends to every token
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

    def __call__(self, X):
        H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = blk(H)  # no mask: every token attends to every token
        return self.norm(H)
```

### The Masked-Token Objective

This model cannot be trained on next-token prediction. With the future
visible, position $t$ can read token $t+1$ directly, and the loss
collapses without teaching anything. The objective must hide what it asks
for. *Masked language modeling* :cite:`Devlin.Chang.Lee.ea.2018` replaces
a random subset of tokens with a special `<mask>` symbol and asks the
model to reconstruct them from everything that remains:

$$
\max \; \sum_{t \in \mathcal{M}} \log p\left(x_t \mid \mathbf{x}_{\setminus \mathcal{M}}\right),
$$
:eqlabel:`eq_mlm`

where $\mathcal{M}$ is the set of masked positions. A prediction at a
masked position draws on context from *both* sides, which is what removing
the causal mask provides. We train this objective in miniature on
`d2l.WikiText103`, the bytes of the Wikipedia articles of
:numref:`subsec_tinylm`. In each window 15% of the bytes are masked, and
`<mask>` gets one extra embedding row after the 256 byte values. The output
head is tied to the embedding, as in :numref:`sec_transformer-block`.

```{.python .input #encoders-decoders-the-masked-token-objective}
%%tab pytorch
data = d2l.WikiText103(batch_size=64, num_steps=64)
MASK = len(data.vocab)  # id of the extra <mask> embedding row

def mask_tokens(X, p=0.15):
    mask = torch.rand(X.shape, device=X.device) < p
    return torch.where(mask, torch.full_like(X, MASK), X), mask

device = d2l.try_gpu()
model = TransformerEncoder(len(data.vocab) + 1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                              weight_decay=0.0)
losses, step = [], 0
while step < 2000:
    for X, _ in data.train_dataloader():
        X = X.to(device)
        Xm, mask = mask_tokens(X)
        logits = F.linear(model(Xm), model.token_emb.weight)
        loss = F.cross_entropy(logits[mask], X[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        step += 1
        if step >= 2000:
            break
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

model = TransformerEncoder(len(data.vocab) + 1,
                           rngs=nnx.Rngs(d2l.get_key()))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=0.0),
                          wrt=nnx.Param)

@nnx.jit
def mlm_step(model, optimizer, Xm, X, mask):
    def loss_fn(model):
        logits = model.token_emb.attend(model(Xm))
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, X)
        return (losses * mask).sum() / mask.sum()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

key, losses, step = d2l.get_key(), [], 0
while step < 2000:
    for X, _ in data.train_dataloader():
        X = jnp.asarray(X)
        key, sub = jax.random.split(key)
        Xm, mask = mask_tokens(X, sub)
        losses.append(float(mlm_step(model, optimizer, Xm, X, mask)))
        step += 1
        if step >= 2000:
            break
print('masked loss at step 500/1000/2000: ' + '/'.join(
    f'{sum(losses[k-50:k]) / 50:.2f}' for k in (500, 1000, 2000)))
```

Two thousand steps bring the masked loss far below 3.17 nats, the unigram
entropy of the byte frequencies, and the model recovers about half of the
masked bytes exactly. Binning this average by position shows how much each
amount of available context contributes.

### Loss by Available Context
:label:`subsec_mlm-loss-by-position`

Our windows are 64 bytes long, so every batch contains a comparison. A
masked byte in the interior has context on both sides, while one at
position 0 or 63 sees only one side. The last position has the context
that a causal language model has at every step. The model and the
objective are the same at every position, so binning the validation loss
by position measures what context from the second side contributes.

```{.python .input #encoders-decoders-what-the-second-side-is-worth-1}
%%tab pytorch
model.eval()
pos_loss = torch.zeros(64, device=device)
pos_cnt = torch.zeros(64, device=device)
correct = total = 0
with torch.no_grad():
    for X, _ in data.val_dataloader():
        X = X.to(device)
        Xm, mask = mask_tokens(X)
        logits = F.linear(model(Xm), model.token_emb.weight)
        loss = F.cross_entropy(logits.transpose(1, 2), X, reduction='none')
        pos_loss += (loss * mask).sum(0)
        pos_cnt += mask.sum(0)
        correct += (logits.argmax(-1)[mask] == X[mask]).sum().item()
        total += int(mask.sum())
pos_loss = (pos_loss / pos_cnt).cpu()
print(f'masked accuracy {correct / total:.2f}')
print(f'loss at position 0: {pos_loss[0]:.2f}, at position 63: '
      f'{pos_loss[63]:.2f}, interior mean: {pos_loss[16:48].mean():.2f}')
d2l.plot(torch.arange(64), pos_loss, 'position in the window',
         'masked loss')
```

```{.python .input #encoders-decoders-what-the-second-side-is-worth-1}
%%tab jax
model.eval()

@nnx.jit
def mlm_eval(model, Xm, X):
    logits = model.token_emb.attend(model(Xm))
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, X)
    return losses, logits.argmax(-1)

pos_loss, pos_cnt = jnp.zeros(64), jnp.zeros(64)
correct = total = 0
key = d2l.get_key()
for X, _ in data.val_dataloader():
    X = jnp.asarray(X)
    key, sub = jax.random.split(key)
    Xm, mask = mask_tokens(X, sub)
    loss, pred = mlm_eval(model, Xm, X)
    pos_loss += (loss * mask).sum(0)
    pos_cnt += mask.sum(0)
    correct += int(((pred == X) & mask).sum())
    total += int(mask.sum())
pos_loss = pos_loss / pos_cnt
print(f'masked accuracy {correct / total:.2f}')
print(f'loss at position 0: {pos_loss[0]:.2f}, at position 63: '
      f'{pos_loss[63]:.2f}, interior mean: {pos_loss[16:48].mean():.2f}')
d2l.plot(jnp.arange(64), pos_loss, 'position in the window',
         'masked loss')
```

The profile is flat across the interior and rises by about half or more at
the two one-sided edges. The difference is the contribution of the second side:
conditioning on both sides of a byte lowers its loss by about a third
relative to one side. The two edges cost about the same, because in text
what follows a gap constrains it about as strongly as what precedes it. We
can also read the predictions directly. We mask one byte of a sentence from
the validation text and list the model's three most probable fills:

```{.python .input #encoders-decoders-what-the-second-side-is-worth-2}
%%tab pytorch
snippet = 'Homarus gammarus , known as the European lobster or common '
ids = torch.tensor(data.vocab.encode(snippet), device=device)[None]
for pos in (11, 22, 28):
    Xm = ids.clone()
    Xm[0, pos] = MASK
    with torch.no_grad():
        logits = F.linear(model(Xm), model.token_emb.weight)
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
    probs = jax.nn.softmax(model.token_emb.attend(model(Xm))[0, pos])
    top = jnp.argsort(probs)[::-1][:3]
    shown = snippet[:pos] + '_' + snippet[pos + 1:]
    print(f'{shown[:40]!r}... -> '
          + ', '.join(f'{data.vocab.decode([int(i)])!r} {probs[i]:.2f}'
                      for i in top))
```

In "known as _he European", the model puts most of its probability on "t".
The left context "known as " allows many words, while the right-hand context
("he European") leaves essentially one, and a left-to-right model never sees
that context. Scaled up to subword tokens, sentence pairs, gigabytes of
text, and a fine-tuning recipe per downstream task, this experiment is BERT
:cite:`Devlin.Chang.Lee.ea.2018`. The Language Models part covers its
pretraining and fine-tuning in full (:numref:`chap_nlp_pretrain`). Encoders
have since adopted the block internals of this chapter: ModernBERT keeps the
encoder-only wiring and uses RoPE, a gated FFN, alternating local and global
attention, and a context of 8,192 tokens
:cite:`Warner.Chaffin.Clavie.ea.2024`.

## Cross-Attention in an Encoder--Decoder

### A Task with Known Alignment
:label:`subsec_reversal-task`

An encoder--decoder is useful when the input and output are different
sequences. Its signature component, cross-attention, is usually
illustrated on machine translation, but a trained translation model gives
us no ground truth to check its attention against. So we choose a task
whose correct alignment is known by construction: *reverse a random
string*. Target position $t$ must copy source position $n-1-t$. The
letters are drawn independently at random, and the only path from the
source to the decoder runs through cross-attention. If the mechanism
works and the model solves the task directly, its attention map should be
the anti-diagonal, a prediction we can check. Random strings also mean
unlimited fresh data: like the GPT of :numref:`sec_gpt`, which almost
never reads the same passage twice, this model never sees the same example
twice.

```{.python .input #encoders-decoders-a-task-whose-alignment-we-know}
%%tab pytorch
V, T = 16, 12   # 16 letters, strings of length 12
BOS = V         # decoder start token

def sample_batch(batch_size):
    src = torch.randint(0, V, (batch_size, T), device=device)
    tgt = src.flip(1)
    dec_in = torch.cat([torch.full((batch_size, 1), BOS, device=device),
                        tgt[:, :-1]], 1)
    return src, dec_in, tgt

def to_str(ids):
    return ''.join(chr(97 + int(i)) for i in ids)

src, dec_in, tgt = sample_batch(1)
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
    return ''.join(chr(97 + int(i)) for i in ids)

src, dec_in, tgt = sample_batch(1, d2l.get_key())
print(f'source {to_str(src[0])!r} -> target {to_str(tgt[0])!r}')
```

As in :numref:`sec_gpt`, training uses teacher forcing: the decoder input
is the target shifted right behind a `<bos>` token, so every position
learns to predict its successor in parallel.

### The Decoder Block

A decoder block is the transformer block plus one sublayer. Between the
causal self-attention and the FFN sits cross-attention: queries from the
target's residual stream, keys and values from the encoder output. It
follows the same pre-norm arrangement as every block in this chapter:
each sublayer reads the stream through a normalization and adds its
result back. The two attention sites are masked differently.
Self-attention stays causal, because the target is being generated, while
cross-attention is unmasked, because the source is fully known before
generation starts. Both masks are optional arguments, so a padded batch
can pass its own (:numref:`subsec_encdec-masks`). By default,
self-attention gets `d2l.causal_mask` and cross-attention no mask.

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

The full model wires a `TransformerEncoder` (the class from the previous
section, reused unchanged) to a stack of decoder blocks with its own
embeddings and output head. `EncoderDecoder` returns the logits; each
attention module keeps the weights of its last call in
`attention_weights`. The method `decode` runs the decoder alone, so that
generation can run the encoder once and then only the decoder for each new
token.

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

### Assembling the Masks
:label:`subsec_encdec-masks`

Our reversal task avoids padding by construction: every string has the
same length, and the only mask in use is the decoder's causal one. A real
batch holds sequences of different lengths, padded on the right to a
rectangle, and it needs one mask per attention site. Each is composed
from the two primitives of :numref:`sec_attention-scoring-functions`, the
padding mask `d2l.padding_mask` built from valid lengths and the causal
triangle `d2l.causal_mask`, combined by logical AND under broadcasting.
Encoder self-attention masks source padding. Decoder self-attention during
teacher-forced training needs the causal triangle *and* target padding on
the key side (padded query rows compute outputs that the loss ignores).
Cross-attention masks source padding again: the target may be
mid-generation, but the source it reads is fully known. The cell below
assembles all three for a toy ragged batch as boolean arrays of shape
(batch, queries, keys). `d2l.MultiHeadAttention` takes such masks
directly, as do the fused kernels of :numref:`sec_attention-at-scale`.

```{.python .input #encoders-decoders-assembling-the-masks}
%%tab pytorch
src_len, tgt_len = torch.tensor([3, 5]), torch.tensor([3, 4])
S, T_dec = int(src_len.max()), int(tgt_len.max())
B = len(src_len)
src_valid = d2l.padding_mask(src_len, S)         # (B, 1, S)
tgt_valid = d2l.padding_mask(tgt_len, T_dec)     # (B, 1, T)
enc_self = src_valid.expand(-1, S, -1)           # (B, S, S): source padding
dec_self = d2l.causal_mask(T_dec) & tgt_valid    # (B, T, T): causal, padding
cross = src_valid.expand(-1, T_dec, -1)          # (B, T, S): source padding
for name, m in (('encoder self', enc_self), ('decoder self', dec_self),
                ('cross', cross)):
    print(f'{name}-attention mask, sequence 0 (1 = may attend):')
    print(m[0].int().numpy())
```

```{.python .input #encoders-decoders-assembling-the-masks}
%%tab jax
src_len, tgt_len = jnp.array([3, 5]), jnp.array([3, 4])
S, T_dec = int(src_len.max()), int(tgt_len.max())
B = len(src_len)
src_valid = d2l.padding_mask(src_len, S)         # (B, 1, S)
tgt_valid = d2l.padding_mask(tgt_len, T_dec)     # (B, 1, T)
enc_self = jnp.broadcast_to(src_valid, (B, S, S))      # source padding
dec_self = d2l.causal_mask(T_dec) & tgt_valid    # (B, T, T): causal, padding
cross = jnp.broadcast_to(src_valid, (B, T_dec, S))     # source padding
for name, m in (('encoder self', enc_self), ('decoder self', dec_self),
                ('cross', cross)):
    print(f'{name}-attention mask, sequence 0 (1 = may attend):')
    print(m[0].astype(int))
```

In the three grids of sequence 0 (source length 3 of 5, target length 3
of 4), the encoder square and the cross rectangle both blank the same two
padded source columns, and the decoder triangle loses its last column to
target padding. These grids are what an encoder--decoder assembles for
every padded batch it trains on. To see the model consume them, we pass
`dec_self` and `cross` to a fresh `DecoderBlock` on random inputs of width
16 and print the cross-attention weights of sequence 0 in one head, in
percent:

```{.python .input #encoders-decoders-assembling-the-masks-1}
%%tab pytorch
blk = DecoderBlock(16, num_heads=2)
X, enc = torch.randn(B, T_dec, 16), torch.randn(B, S, 16)
with torch.no_grad():
    blk(X, enc, dec_self, cross)
weights = blk.cross_attention.attention_weights[0, 0]  # (queries, keys)
print((100 * weights).round().int().numpy())
```

```{.python .input #encoders-decoders-assembling-the-masks-1}
%%tab jax
blk = DecoderBlock(16, num_heads=2, rngs=nnx.Rngs(d2l.get_key()))
X = jax.random.normal(d2l.get_key(), (B, T_dec, 16))
enc = jax.random.normal(d2l.get_key(), (B, S, 16))
blk(X, enc, dec_self, cross)
weights = blk.cross_attention.attention_weights[0, 0]  # (queries, keys)
print((100 * weights).round().astype(int))
```

Each row spreads its weight over the three real source positions, and the
two padded columns receive exactly zero. None of this arises in the
reversal model: `sample_batch` produces no padding, so the decoder keeps its
default causal mask and the cross-attention runs unmasked.

### Training and Decoding

One encoder block, one decoder block, and a few hundred steps on freshly
sampled batches suffice. Over 600 steps of 128 strings the model reads
about a million source characters and never sees a string twice. The
optimizer is `d2l.LMOptimizer` of :numref:`sec_gpt`, with its gradient
clipping but without warmup or weight decay. Once the loss is near zero,
Adam at a constant learning rate makes it jump now and then, to tenths of
a nat or more, for about twenty steps. A jump near the end of the run
leaves a model that fails on many strings. The cosine decay of the
learning rate to zero prevents this.

```{.python .input #encoders-decoders-training-and-decoding-1}
%%tab pytorch
seq2seq = EncoderDecoder(V + 1).to(device)
optimizer = d2l.LMOptimizer(seq2seq, lr=1e-3, num_steps=600, warmup=0,
                            weight_decay=0.0)
losses = []
for step in range(600):
    src, dec_in, tgt = sample_batch(128)
    logits = seq2seq(src, dec_in)
    loss = F.cross_entropy(logits.reshape(-1, V + 1), tgt.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
print('loss at step 100/200/400/600: ' + '/'.join(
    f'{sum(losses[k-20:k]) / 20:.3f}' for k in (100, 200, 400, 600)))
```

```{.python .input #encoders-decoders-training-and-decoding-1}
%%tab jax
seq2seq = EncoderDecoder(V + 1, rngs=nnx.Rngs(d2l.get_key()))
optimizer = d2l.LMOptimizer(seq2seq, lr=1e-3, num_steps=600, warmup=0,
                            weight_decay=0.0)

@nnx.jit
def s2s_step(model, optimizer, src, dec_in, tgt):
    def loss_fn(model):
        logits = model(src, dec_in)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

key, losses = d2l.get_key(), []
for step in range(600):
    key, sub = jax.random.split(key)
    src, dec_in, tgt = sample_batch(128, sub)
    losses.append(float(s2s_step(seq2seq, optimizer, src, dec_in, tgt)))
print('loss at step 100/200/400/600: ' + '/'.join(
    f'{sum(losses[k-20:k]) / 20:.3f}' for k in (100, 200, 400, 600)))
```

Decoding runs the encoder once and the decoder autoregressively: feed
`<bos>`, take the argmax, append, repeat. On a thousand fresh strings the
model reverses essentially every one exactly.

```{.python .input #encoders-decoders-training-and-decoding-2}
%%tab pytorch
seq2seq.eval()
with torch.no_grad():
    src, _, tgt = sample_batch(1000)
    enc = seq2seq.encoder(src)  # the encoder runs once
    out = torch.full((1000, T + 1), BOS, device=device)
    for t in range(T):
        logits = seq2seq.decode(out[:, :t + 1], enc)
        out[:, t + 1] = logits[:, -1].argmax(-1)
pred = out[:, 1:]
print(f'exact match on 1000 fresh strings: '
      f'{(pred == tgt).all(-1).float().mean():.3f}')
print(f'source {to_str(src[0])!r} -> predicted {to_str(pred[0])!r}')
```

```{.python .input #encoders-decoders-training-and-decoding-2}
%%tab jax
seq2seq.eval()

@nnx.jit
def next_token(model, enc, buf, t):
    return model.decode(buf, enc)[:, t].argmax(-1)

src, _, tgt = sample_batch(1000, d2l.get_key())
enc = seq2seq.encoder(src)  # the encoder runs once
buf = jnp.full((1000, T + 1), BOS)
for t in range(T):
    buf = buf.at[:, t + 1].set(next_token(seq2seq, enc, buf, t))
pred = buf[:, 1:]
print(f'exact match on 1000 fresh strings: '
      f'{(pred == tgt).all(-1).mean():.3f}')
print(f'source {to_str(src[0])!r} -> predicted {to_str(pred[0])!r}')
```

:begin_tab:`jax`
As in `generate` of :numref:`sec_gpt`, the decode loop writes into a
buffer of fixed size, overwritten left to right. Every shape stays static,
so the jitted step compiles once. The causal mask hides the
not-yet-written positions from every query at or before $t$.
:end_tab:

### Reading the Alignment
:label:`subsec_reading-alignment`

We run a batch through the model, read the cross-attention weights of the
decoder block, of shape (batch, heads, target positions, source positions),
and compare each target position's attention with the alignment that the
task dictates. Besides the head average we report how much weight each
head puts on the true source position, and we plot the average next to
the head that puts the most weight there.

```{.python .input #encoders-decoders-reading-the-alignment}
%%tab pytorch
with torch.no_grad():
    src, dec_in, tgt = sample_batch(64)
    seq2seq(src, dec_in)
w = seq2seq.blks[0].cross_attention.attention_weights  # (64, 4, T, T)
want = torch.arange(T - 1, -1, -1, device=device)  # row t -> column T-1-t
avg = w.mean(1)                                     # average over heads
cols = avg.argmax(-1)                  # the source column each row reads
hit = (cols == want).float().mean()
same = (cols == cols[0]).float().mean()  # as in the first string
on_target = w[..., torch.arange(T), want].mean((0, 2))  # one per head
print(f'head average: argmax on the true source position in '
      f'{100 * hit:.0f}% of rows, mean weight there {on_target.mean():.2f}')
print('weight on the true position, per head:',
      ' '.join(f'{v:.2f}' for v in on_target))
print(f'argmax in the same column for every string: {100 * same:.0f}% '
      'of rows')
h = int(on_target.argmax())
d2l.show_heatmaps(torch.stack([avg[0], w[0, h]]).cpu()[None],
                  xlabel='source position', ylabel='target position',
                  titles=['head average', f'head {h}'], figsize=(6, 3),
                  cmap='Blues')
```

```{.python .input #encoders-decoders-reading-the-alignment}
%%tab jax
src, dec_in, tgt = sample_batch(64, d2l.get_key())
seq2seq(src, dec_in)
w = seq2seq.blks[0].cross_attention.attention_weights  # (64, 4, T, T)
want = jnp.arange(T - 1, -1, -1)                   # row t -> column T-1-t
avg = w.mean(1)                                    # average over heads
cols = avg.argmax(-1)                  # the source column each row reads
hit = (cols == want).mean()
same = (cols == cols[0]).mean()        # as in the first string
on_target = w[..., jnp.arange(T), want].mean((0, 2))  # one per head
print(f'head average: argmax on the true source position in '
      f'{100 * hit:.0f}% of rows, mean weight there {on_target.mean():.2f}')
print('weight on the true position, per head:',
      ' '.join(f'{v:.2f}' for v in on_target))
print(f'argmax in the same column for every string: {100 * same:.0f}% '
      'of rows')
h = int(on_target.argmax())
d2l.show_heatmaps(jnp.stack([avg[0], w[0, h]])[None],
                  xlabel='source position', ylabel='target position',
                  titles=['head average', f'head {h}'], figsize=(6, 3),
                  cmap='Blues')
```

Whether the maps show the anti-diagonal predicted by the third panel of
:numref:`fig_three-wirings` depends on the run. In some runs they do:
averaged over heads, the argmax lands on the true source position in
almost every row, with most of the softmax mass concentrated there. In
others it lands there in only about half of the rows or fewer, even in
none, although the model still reverses essentially every string. The last line of the output
shows what all runs share. Each target position reads mainly from one
source column, the same for most strings and in many runs for nearly all,
so the decoder does retrieve by position, as the task requires. But the encoder has a position table of its own, and its
self-attention can move every letter to another position before the
decoder reads it. Any such rearrangement, undone by the cross-attention,
reverses the string just as well, and training settles on one of them at
random. The alignment is then split between two attention layers, and the
cross-attention map shows only the second half. As
:numref:`sec_multihead-attention` noted, readable attention patterns are
the exception rather than the rule. Even in this one-block model a single
map can mislead, and with more depth and more heads the maps spread out
further, as exercise :exref:`ex_encdec-depth-heads` shows.

## Cross-Attention as Interface

### Learned Queries

In the decoder, the cross-attention queries came from the target stream.
Nothing in the mechanism requires that source. A set of $M$ query vectors
can instead be *learned parameters*, a fixed array that exists before any
input arrives. Cross-attending it into an input of length $N$ costs
$O(MN)$. The quadratic term of :numref:`sec_attention-at-scale` never
appears, because the $M$ latents only ever self-attend among themselves,
at $O(M^2)$. This is the
*Perceiver* :cite:`Jaegle.Gimeno.Brock.ea.2021`: a fixed-size latent
bottleneck that reads arbitrarily long, arbitrarily structured
inputs through cross-attention, sketched in
:numref:`fig_latent-bottleneck`.

![A latent bottleneck with input length $N$, $M$ learned latents, width $d$, and $L$ latent blocks. Every latent reads every input (shaded band); the read of one latent is drawn in full. One cross-attention read has mixing cost $O(MNd)$; each subsequent latent self-attention layer has mixing cost $O(M^2d)$, for $O(LM^2d)$ across the stack. These terms hold batch size, head count, and width fixed and omit projections and FFNs.](../img/mdl-transformers-latent-bottleneck.svg)
:label:`fig_latent-bottleneck`

A minimal version needs a parameter array, one cross-attention, and a
stack of ordinary transformer blocks over the latents:

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

perceiver = PerceiverEncoder(num_latents=64, num_hiddens=256).to(device)
X = torch.randn(1, 4096, 256, device=device)
print('input', tuple(X.shape), '-> latent summary',
      tuple(perceiver(X).shape))
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

perceiver = PerceiverEncoder(num_latents=64, num_hiddens=256,
                             rngs=nnx.Rngs(d2l.get_key()))
perceiver.eval()
X = jax.random.normal(d2l.get_key(), (1, 4096, 256))
print('input', X.shape, '-> latent summary', perceiver(X).shape)
```

Whatever the input length, the output is $M = 64$ vectors. The input here
is a raw feature sequence rather than token embeddings, because the
encoder does not depend on what the input is: text, audio frames, image
pixels, or a concatenation of all three.

Learned queries can also set the shape of the output. Perceiver IO
:cite:`Jaegle.Borgeaud.Alayrac.ea.2022` adds an array of output queries
that cross-attends *out of* the latent summary, so the output has one
vector per output query, whatever the input: one query for a
classification label, one per pixel for optical flow, one per audio sample
for a waveform. Input length, number of latents, and output shape become
three independent choices, and the latent transformer keeps its fixed
cost.

### The Cost Curve
:label:`subsec_perceiver-cost-curve`

We check the shape of this cost. We time the Perceiver
against the direct alternative (the same two transformer blocks applied
to the full input sequence) as $N$ doubles at fixed $M = 64$. The timer
is `d2l.Benchmark` of :numref:`sec_perf_model`: it runs the function a few
times to warm up, waits for the device to finish, and reports the average
time of further calls.

```{.python .input #encoders-decoders-the-cost-curve}
%%tab pytorch
class SelfAttentionEncoder(nn.Module):
    """The comparison: the same blocks over the full input sequence."""
    def __init__(self, num_hiddens, num_heads=4, num_blks=2):
        super().__init__()
        self.blks = nn.ModuleList([
            d2l.TransformerBlock(num_hiddens, num_heads)
            for _ in range(num_blks)])

    def forward(self, X):
        for blk in self.blks:
            X = blk(X)                                            # O(N^2)
        return X

full = SelfAttentionEncoder(256).to(device).eval()
perceiver.eval()
lengths, t_full, t_perc = [1024, 2048, 4096, 8192], [], []
with torch.no_grad():
    for N in lengths:
        X = torch.randn(1, N, 256, device=device)
        t_full.append(1e3 * d2l.Benchmark(lambda: full(X), repeats=20).time)
        t_perc.append(1e3 * d2l.Benchmark(lambda: perceiver(X),
                                          repeats=20).time)
        print(f'N={N:5d}: self-attention {t_full[-1]:6.2f} ms, '
              f'perceiver {t_perc[-1]:5.2f} ms')
d2l.plot(lengths, [t_full, t_perc], 'input length N', 'forward time (ms)',
         xscale='log', yscale='log',
         legend=['self-attention encoder', 'perceiver encoder'])
d2l.plt.xticks(lengths, [str(N) for N in lengths]);
```

```{.python .input #encoders-decoders-the-cost-curve}
%%tab jax
class SelfAttentionEncoder(nnx.Module):
    """The comparison: the same blocks over the full input sequence."""
    def __init__(self, num_hiddens, num_heads=4, num_blks=2, *, rngs):
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, rngs=rngs)
            for _ in range(num_blks)])

    def __call__(self, X):
        for blk in self.blks:
            X = blk(X)                                            # O(N^2)
        return X

full = SelfAttentionEncoder(256, rngs=nnx.Rngs(d2l.get_key()))
full.eval()
jit_full, jit_perc = nnx.jit(full), nnx.jit(perceiver)
lengths, t_full, t_perc = [1024, 2048, 4096, 8192], [], []
for N in lengths:
    X = jax.random.normal(d2l.get_key(), (1, N, 256))
    # the warmup calls include one compilation per input length
    t_full.append(1e3 * d2l.Benchmark(lambda: jit_full(X), repeats=20).time)
    t_perc.append(1e3 * d2l.Benchmark(lambda: jit_perc(X), repeats=20).time)
    print(f'N={N:5d}: self-attention {t_full[-1]:6.2f} ms, '
          f'perceiver {t_perc[-1]:5.2f} ms')
d2l.plot(lengths, [t_full, t_perc], 'input length N', 'forward time (ms)',
         xscale='log', yscale='log',
         legend=['self-attention encoder', 'perceiver encoder'])
d2l.plt.xticks(lengths, [str(N) for N in lengths]);
```

Each doubling of $N$ eventually multiplies the self-attention encoder's
time by about four, the signature of an $N^2$ term taking over. The
Perceiver's time grows far more slowly over this range: its $O(MN)$
cross-attention grows linearly, but the fixed $O(M^2)$ latent processing
dominates. By $N = 8192$ the gap exceeds an order of magnitude. At
$N = 1024$ the two encoders take comparable time, because the Perceiver's
fixed latent cost is then a large fraction of its total. A latent
bottleneck is the better choice when the input is long and a fixed-size
summary suffices.

The three wirings differ in what they produce. An encoder produces one
representation per input token, which classification and retrieval
consume. An encoder--decoder generates a sequence conditioned on a
separate input that is fully known in advance. A decoder-only model
continues a sequence, with one stack for training and for generation.
Learned queries are a fourth use of cross-attention: they decouple the
cost of attention from the length of the input.

## Summary

A transformer block supports encoder-only, decoder-only, and
encoder--decoder architectures. Removing the causal mask produces an encoder
that represents each token using context from both directions. In our
byte-level experiment, positions with context on both sides have about a
third less loss than positions with only left context. An encoder--decoder
connects an unmasked encoder to a causal decoder through cross-attention; on
a reversal task, it reads each target position mainly from one fixed
source position, the anti-diagonal of the task or, when the encoder has first moved
the letters, a rearrangement of it. Cross-attention can also use a learned
latent array as its queries. Reading a length-$N$ input into $M$ latents
costs $O(MN)$ rather than $O(N^2)$, and for fixed $M$ its measured runtime
grows far more slowly with $N$ than that of full self-attention.
Encoder-only models are common for retrieval and classification,
encoder--decoders for translation and speech, and decoder-only models for
general autoregressive generation.

## Exercises

1. **Prefix language model.** A *prefix language model* wires a single
   stack in a fourth way. The first $m$ positions, which hold the source,
   attend to one another bidirectionally, and each of the $n$ target
   positions that follow attends to the whole source and causally to the
   target positions up to itself. :citet:`raffel2020exploring` compare this
   wiring with the encoder--decoder.
    1. Draw its attention pattern for $m = 3$ and $n = 3$ in the convention
       of :numref:`fig_three-wirings`. Number all $m + n$ positions from 0
       and write the mask entry for query $t$ and key $j$ (true when $t$
       may attend to $j$) as a formula in $t$, $j$, and $m$. Build the mask
       from `d2l.causal_mask` and one block that unmasks the source, and
       check it against your drawing.
    1. In the encoder--decoder, the cross-attention of every decoder block
       reads the output of the last encoder block. Which representations of
       the source does a target token read in block $l$ of a prefix
       language model? On which tokens can a source token's representation
       depend in each design?
    1. Let every block have $c$ parameters, each applied once to every token
       that the block processes, and ignore the cross-attention sublayer and
       the attention scores. Compare the parameter count and the training
       FLOPs per example of an encoder--decoder with $L$ encoder blocks and
       $L$ decoder blocks against prefix language models with $L$ blocks and
       with $2L$ blocks. Which prefix model matches the encoder--decoder in
       parameters, and which in compute?

1. [code] **The masking recipe.** `mask_tokens` masks 15% of the bytes
   and writes `<mask>` at every masked position. Evaluate every variant below
   with the evaluation of :numref:`subsec_mlm-loss-by-position`, so that only
   the training recipe changes.
    1. BERT replaces a chosen token with `<mask>` 80% of the time, with a
       random token 10% of the time, and leaves it unchanged 10% of the time
       :cite:`Devlin.Chang.Lee.ea.2018`, in part because `<mask>` never
       appears when the pretrained encoder reads real text. Implement this
       rule, retrain, and compare the masked accuracy with that of the
       original recipe. Then feed both encoders unmodified validation
       windows and measure the loss of predicting each input byte at
       its own position. Explain what each of the two 10% cases teaches the
       model.
    1. :citet:`Wettig.Gao.Zhong.ea.2023` find that masking rates well above
       15% can work better for large models. Retrain with the rates
       $p \in \{0.05, 0.15, 0.4\}$ for the same number of steps and compare
       the masked loss at evaluation. Explain the result by what a higher
       rate adds to each training window and what it removes.

1. **Padding masks of a ragged batch.** The cell of
   :numref:`subsec_encdec-masks` builds the three masks for a toy batch with
   source lengths $(3, 5)$ and target lengths $(3, 4)$, padded on the right.
    1. Compare `dec_self` with the causal triangle alone and name the rows
       that differ. Would replacing `dec_self` by the causal triangle change
       the training loss of a decoder with any number of blocks? Explain.
       Then decide whether the encoder output at a padded source position
       can reach the loss.
    1. Now pad the batch on the left, so that every sequence ends at the
       last position. Which of the three masks change, and does the causal
       triangle still suffice for decoder self-attention? `d2l.padding_mask`
       marks the first `valid_lens` keys of each sequence as valid. Which of
       the three masks can it still build?
    1. What goes wrong with the learned position tables of `EncoderDecoder`
       under left padding unless the position indices change as well?

1. [code] **Depth, heads, and readable maps.** The alignment check of
   :numref:`subsec_reading-alignment` reads one decoder block with four
   heads.
    1. Retrain `EncoderDecoder` once with `num_blks=2` and once with
       `num_heads=8`, in each case until the exact match on 1000 fresh
       strings exceeds 0.99. Rerun the alignment check for every decoder
       block, and plot the map of every head.
    1. Give `EncoderDecoder` separate depths for its encoder and its
       decoder. Repeat the check with two encoder blocks and one decoder
       block, and with one encoder block and two decoder blocks. Compare
       the maps of both models with those of the one-block model.
    1. Follow each letter back through the encoder: multiply the
       head-averaged cross-attention weights by the head-averaged weights
       of the encoder's self-attention, and compute the hit rate of the
       product. Over several runs, does the product recover the
       anti-diagonal where the cross-attention alone does not? When does
       the residual stream make this tracing fail? Reconcile your findings
       with the cautions of :numref:`sec_multihead-attention` about reading
       attention maps.
   :exlabel:`ex_encdec-depth-heads`

1. [code] **Several masks at once.** ● The objective
   :eqref:`eq_mlm` scores each masked position separately, so the encoder
   returns one distribution per masked position, conditioned only on the
   bytes that remain visible. Filling all masked positions from one
   forward pass therefore chooses each byte independently of the
   others. Iterative schemes such as Mask-Predict
   :cite:`Ghazvininejad.Levy.Liu.ea.2019` keep only their most confident
   predictions and predict the remaining positions again.
    1. Construct a distribution over the two-byte fills of a gap for
       which the pair of per-position argmaxes has probability zero. Show
       that filling the more confident position first and then filling the
       other from its exact conditional distribution always yields a pair
       of positive probability.
    1. In each validation window, mask one run of $k$ adjacent bytes,
       placed at random with at least 8 unmasked bytes on either side,
       and nothing else. For $k \in \{1, 2, 4, 8\}$, measure the byte
       accuracy of the parallel fill and of an iterative fill that commits
       one byte per forward pass, always at the masked position whose
       top probability is largest.
    1. Plot the accuracy of both fills against the offset within the run.
       Explain the shape of the curves by the context that each masked
       byte has lost, and explain where the two fills differ.

1. **Time's arrow.** ● The masked loss of
   :numref:`subsec_mlm-loss-by-position` takes nearly equal values at the two
   edges of the window, although position 0 sees only what follows it and
   position 63 only what precedes it. The conditional entropy
   $H(A \mid B) = -\mathbb{E}\left[\log p(A \mid B)\right]$ is the
   expected loss, in nats, of the best predictor of $A$ from $B$.
    1. Let $X_1, X_2, \ldots$ be a stationary process, so that the joint
       distribution of any $k$ consecutive variables does not depend on
       where the block starts. Show that
       $$H(X_1 \mid X_2, \ldots, X_{k+1}) = H(X_{k+1} \mid X_1, \ldots, X_k).$$
    1. A stationary Markov chain on the states $\{0, 1, 2\}$ moves from
       state $i$ to state $i + 1 \bmod 3$ with probability $0.9$ and stays
       with probability $0.1$. Compute both sides of the identity for
       $k = 1$ in nats. Find the transition matrix of the time-reversed
       chain, and decide whether a long sample path reveals the direction
       of time.
    1. State what the identity predicts for the two edge losses of a model
       whose masked-token predictions equal the true conditional
       distributions. Which differences between the experiment and the
       assumptions of the identity allow the two measured edge losses to
       differ?

1. [code] **Content-addressed alignment.** ● In reversal, target position
   $t$ copies the same source position for every string. Change the line of
   `sample_batch` that builds the target so that the target is the source
   sorted in increasing order.
    1. For one string with a repeated letter, list the source positions that
       each target position may copy. Redefine the hit rate of
       :numref:`subsec_reading-alignment` so that every correct choice
       counts.
    1. Train `EncoderDecoder` on sorting with one block and with two
       blocks, for as many steps as the exact match on fresh strings
       requires or until it stops improving. Report the exact match and
       your hit rate.
    1. Plot the maps for two strings. Describe what the decoder's query at
       step $t$ must encode for sorting, compared with reversal, and which
       information the encoder must supply when a letter occurs more than
       once.

1. [extended] **A Perceiver for reversal.** ● Perceiver IO produces its
   outputs through learned output queries rather than an autoregressive
   decoder. Solve the reversal task of :numref:`subsec_reversal-task` in
   this way, with `num_hiddens=128` throughout.
    1. Embed the source with learned letter and position tables and read it
       with `PerceiverEncoder` into $M$ latents. Let 12 learned output
       queries, one per target position, cross-attend into the latent
       summary, and add each result to its query. Apply `d2l.FeedForward`
       with a second residual connection, and map the result linearly onto
       the 16 letters. Train all outputs in parallel with the cross-entropy
       loss on fresh batches from `sample_batch`, with no decoder input.
    1. Train with $M \in \{1, 2, 4, 12\}$ for a fixed number of steps and
       report the exact match on 1000 fresh strings. Find the smallest $M$
       that solves the task. For a smaller $M$, decide whether the latents'
       read of the source or the output queries' read of the latents limits
       the model.
    1. Plot the latents' cross-attention over the source and the output
       queries' attention over the latents. Describe how the model divides
       the string among the latents.
    1. Explain why predicting all target positions independently given the
       latents is no restriction for reversal, and name a sequence task for
       which it would be one.

[Discussions](https://d2l.discourse.group/t/36777)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.4]{.kicker}

Encoders, decoders, and cross-attention<br>
**three wirings of one block · masked-token prediction · an alignment we can check · learned queries**
:::
:::

::: {.slide title="A Causal Model Sees Only One Side of Every Gap"}
[The problem]{.kicker}

- Fill the gap in `known as _he European`: the left context allows many
  words, the right context (`he European`) fixes the letter.
- Measured on Wikipedia text: a masked byte costs about **half again** as
  much with one side of context as with both sides.
- Translation and speech recognition read a whole input before they write
  a single output token.

::: {.d2l-note .warn}
How do we wire the same transformer block to read both sides, to read a
separate input, and to read inputs too long for $N^2$ attention?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- draw the attention pattern of the encoder, the decoder, and the
  encoder--decoder;
- train an encoder with masked-token prediction and measure what the
  second side of context is worth;
- build a decoder block with cross-attention and compose its masks;
- check a learned cross-attention map against a known alignment;
- estimate the cost of reading $N$ inputs with $M$ learned queries.

**Running example:** GPT-2 small is the decoder-only wiring; we build the
other two at small scale.
:::

::: {.slide title="The Wirings Complete the Block"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN on a residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| Generation and the KV Cache | reuse keys and values while sampling |
| **Encoders, Decoders, and Cross-Attention** | **the other two wirings** |
| Vision Transformer | images as token sequences |
| Mixture of Experts | more parameters per FLOP |
| Scaling Laws | loss against parameters, data, compute |
:::

::: {.col .narrow}
- **Uses:** `TransformerBlock`, cross-attention, padding and causal masks,
  teacher forcing.
- **Skips:** pretraining at scale and fine-tuning (Language Models part),
  beam search.
:::
:::
:::

::: {.slide title="One Block, Three Wirings"}
@fig:mdl-transformers-three-wirings

Which positions may attend to which, and where keys and values come from:
these two choices separate BERT, GPT, and the 2017 translation model.
:::

::: {.slide title="An Encoder Is a Stack of Blocks with No Mask" only="pytorch"}
```{.python #encoders-decoders-exercises-1}
def forward(self, X):                     # X: (batch, n) token ids
    H = self.token_emb(X) + self.pos_emb(torch.arange(X.shape[1]))
    for blk in self.blks:                 # d2l.TransformerBlock
        H = blk(H)                        # no mask: all see all
    return self.norm(H)                   # (batch, n, d)
```

Every output vector summarizes the whole input as seen from its position.
Positions still need a table: attention ignores order.
:::

::: {.slide title="An Encoder Is a Stack of Blocks with No Mask" only="jax"}
```{.python #encoders-decoders-exercises-2}
def __call__(self, X):                    # X: (batch, n) token ids
    H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
    for blk in self.blks:                 # d2l.TransformerBlock
        H = blk(H)                        # no mask: all see all
    return self.norm(H)                   # (batch, n, d)
```

Every output vector summarizes the whole input as seen from its position.
Positions still need a table: attention ignores order.
:::

::: {.slide title="Masked Prediction Hides What It Asks For"}
With the future visible, next-token prediction is copying. Instead, replace
a random set $\mathcal{M}$ of tokens (15%) by `<mask>` and reconstruct them:

$$\max \; \sum_{t \in \mathcal{M}} \log p\left(x_t \mid \mathbf{x}_{\setminus \mathcal{M}}\right)$$

@!encoders-decoders-the-masked-token-objective

Bytes of WikiText-103, windows of 64, 2,000 steps; unigram entropy 3.17
nats.
:::

::: {.slide title="Both Sides of Context Cut the Loss by a Third"}
Positions 0 and 63 of a window see one side only; position 63 sees what
a causal model sees.

@!encoders-decoders-what-the-second-side-is-worth-1

::: {.d2l-note .rule}
The edges cost about 1.5 times the interior; the two edges cost about the
same.
:::
:::

::: {.slide title="The Right-Hand Context Fixes the Missing Letter"}
@!encoders-decoders-what-the-second-side-is-worth-2

- `he European` fixes the `t` in "as _he European"; a left-to-right
  model never sees it.
- Scaled to subwords and gigabytes, this experiment is BERT (2018);
  ModernBERT (2024) keeps the wiring with RoPE, a gated FFN, and 8k
  context.
:::

::: {.slide title="Check: Why Not Train the Encoder on the Next Token?"}
[Check]{.kicker}

Our encoder has no mask. What loss would next-token prediction reach,
and what would the encoder learn from it?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Position $t$ can read token $t+1$ directly, so a copy
attention head drives the loss to zero and teaches nothing about
language. The objective must hide what it asks for.
:::
:::
:::

::: {.slide title="Reversal Fixes the Correct Alignment"}
Target position $t$ must copy source position $n-1-t$. Letters are
random, so content reaches the decoder only through cross-attention.

@!encoders-decoders-a-task-whose-alignment-we-know

- Prediction: if the model solves the task directly, the
  cross-attention map is the anti-diagonal.
- Fresh random strings every step: nothing to memorize.
:::

::: {.slide title="The Decoder Block Adds Cross-Attention"}
On the target's residual stream $\mathbf{x}$, with encoder output
$\mathbf{E}$:

$$\begin{aligned}
\mathbf{x} &\leftarrow \mathbf{x} + \mathrm{Attn}_\text{causal}(\mathrm{Norm}_1(\mathbf{x})) \\
\mathbf{x} &\leftarrow \mathbf{x} + \mathrm{Attn}(\text{q} = \mathrm{Norm}_2(\mathbf{x}),\ \text{k} = \text{v} = \mathbf{E}) \\
\mathbf{x} &\leftarrow \mathbf{x} + \mathrm{FFN}(\mathrm{Norm}_3(\mathbf{x}))
\end{aligned}$$

- Self-attention is causal: the target is being generated.
- Cross-attention is unmasked: the source is known in full.
:::

::: {.slide title="In Code, One Extra Line" only="pytorch"}
```{.python #encoders-decoders-exercises-3}
def forward(self, X, enc, mask=None, cross_mask=None):
    if mask is None:                      # causal self-attention
        mask = d2l.causal_mask(X.shape[1], device=X.device)
    Y = self.norm1(X)
    X = X + self.self_attention(Y, Y, Y, mask)
    Y = self.norm2(X)                     # queries: target stream
    X = X + self.cross_attention(Y, enc, enc, cross_mask)
    return X + self.ffn(self.norm3(X))
```

Keys and values come from `enc`, the encoder output.
:::

::: {.slide title="In Code, One Extra Line" only="jax"}
```{.python #encoders-decoders-exercises-4}
def __call__(self, X, enc, mask=None, cross_mask=None):
    if mask is None:                      # causal self-attention
        mask = d2l.causal_mask(X.shape[1])
    Y = self.norm1(X)
    X = X + self.self_attention(Y, Y, Y, mask)
    Y = self.norm2(X)                     # queries: target stream
    X = X + self.cross_attention(Y, enc, enc, cross_mask)
    return X + self.ffn(self.norm3(X))
```

Keys and values come from `enc`, the encoder output.
:::

::: {.slide title="Three Masks from Two Primitives"}
A padded batch, source lengths (3, 5), target lengths (3, 4):

- `padding_mask(lens, n)`: key $j$ valid if $j <$ length;
- `causal_mask(T)`: key $j$ visible to query $i$ if $j \le i$;
- encoder self: source padding · decoder self: causal ∧ target padding ·
  cross: source padding.

Cross-attention weights of sequence 0, in percent:

@!encoders-decoders-assembling-the-masks-1
:::

::: {.slide title="Check: Which Mask Does Cross-Attention Need?"}
[Check]{.kicker}

The source batch is padded on the right; the target is mid-generation.
Does cross-attention need the causal triangle, source padding, or both?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Source padding only. The causal triangle protects the target
from its own future; every real source token is known before decoding
starts, but padded source positions must receive zero weight.
:::
:::
:::

::: {.slide title="600 Steps Reverse Nearly Every String"}
One encoder block, one decoder block, batches of 128 fresh strings:

@!encoders-decoders-training-and-decoding-1

Greedy decoding: the encoder runs once, the decoder once per token.

@!encoders-decoders-training-and-decoding-2
:::

::: {.slide title="The Map Is the Anti-Diagonal or a Rearrangement of It"}
@!encoders-decoders-reading-the-alignment

Each row reads mainly one source column, the same for most strings. If
the encoder has moved the letters first, the map shows a rearrangement of
the anti-diagonal.
:::

::: {.slide title="Learned Queries Read N Inputs at Cost O(MN)"}
@fig:mdl-transformers-latent-bottleneck

$M$ learned latents cross-attend into the input once, then attend only
among themselves: the $N^2$ term never appears (Perceiver, 2021).
:::

::: {.slide title="A Perceiver Is Latents Plus One Cross-Attention" only="pytorch"}
```{.python #encoders-decoders-exercises-5}
def forward(self, X):                     # X: (batch, N, d)
    Z = self.latents.expand(X.shape[0], -1, -1)   # (batch, M, d)
    Z = Z + self.cross_attention(self.norm_q(Z), X, X)  # O(MN)
    for blk in self.blks:
        Z = blk(Z)                        # O(M^2): latents only
    return self.norm(Z)                   # (batch, M, d)
```

@!encoders-decoders-queries-need-not-come-from-a-sequence

Output queries can set the output shape too (Perceiver IO).
:::

::: {.slide title="A Perceiver Is Latents Plus One Cross-Attention" only="jax"}
```{.python #encoders-decoders-exercises-6}
def __call__(self, X):                    # X: (batch, N, d)
    Z = jnp.broadcast_to(self.latents[...],
                         (X.shape[0],) + self.latents.shape)
    Z = Z + self.cross_attention(self.norm_q(Z), X, X)  # O(MN)
    for blk in self.blks:
        Z = blk(Z)                        # O(M^2): latents only
    return self.norm(Z)                   # (batch, M, d)
```

@!encoders-decoders-queries-need-not-come-from-a-sequence

Output queries can set the output shape too (Perceiver IO).
:::

::: {.slide title="Check: Count the Scores"}
[Check]{.kicker}

$N = 8192$ inputs, $M = 64$ latents. How many attention scores per head
does one layer of full self-attention compute, and how many does the
Perceiver's read compute?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $N^2 \approx 6.7 \times 10^7$ against
$MN \approx 5.2 \times 10^5$: 128 times fewer, plus
$M^2 = 4{,}096$ per latent layer.
:::
:::
:::

::: {.slide title="Doubling N Quadruples Self-Attention Time"}
@!encoders-decoders-the-cost-curve

Self-attention: about $4\times$ per doubling. The Perceiver grows far more
slowly and is over an order of magnitude faster at $N = 8192$; at
$N = 1024$ the two are comparable.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- One block, three wirings: no mask, causal mask, or both joined by
  cross-attention.
- Masked prediction trains the encoder; a masked byte costs about a third
  less with both sides of context than with one.
- On reversal, the encoder--decoder reverses nearly every fresh string;
  its cross-attention reads by position, along the anti-diagonal or a
  rearrangement of it made by the encoder.
- $M$ learned queries read $N$ inputs at $O(MN)$; at $N = 8192$ the
  Perceiver is over $10\times$ faster than full self-attention.

**Next:** the Vision Transformer, the encoder-only wiring over image
patches.
:::
