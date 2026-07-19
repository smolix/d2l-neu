# Encoders, Decoders, and Cross-Attention
:label:`sec_transformer`

The GPT of :numref:`sec_gpt` made one architectural commitment beyond
stacking blocks: the causal mask. That mask is what lets the model factor a
joint distribution into next-token predictions, and it is also a
restriction: every representation is built from the left half of its
context only. This section treats the mask as the design variable it is.
Removing it gives an *encoder*, a model that reads in both directions and
excels at representing rather than generating; combining a masked stack
with an unmasked one, joined by the cross-attention of
:numref:`sec_multihead-attention`, gives the *encoder--decoder* that
transformers started as :cite:`Vaswani.Shazeer.Parmar.ea.2017`. We build
both from the same `d2l.TransformerBlock` as always, watch a
cross-attention map reproduce an alignment we know in advance, and then
push cross-attention past sequence-to-sequence entirely: its queries need
not come from a sequence at all, and making them learned parameters turns
attention into a general interface between fixed-size computation and
variable-size data — the Perceiver idea, alive today inside most
vision--language models.

```{.python .input #encoders-decoders-encoders-decoders-and-cross-attention}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import time
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
import time
```

## Three Wirings of One Block
:label:`sec_large-pretraining-transformers`

A transformer block maps a sequence of $d$-dimensional vectors to a
sequence of the same shape, and it leaves two questions open: which
positions may attend to which, and where the keys and values come from.
Three answers cover essentially every deployed transformer, and
:numref:`fig_three-wirings` draws them in the convention we will use for
every attention map in this section — queries down the rows, keys along
the columns.

![The three wirings and their attention patterns (query position runs downward, key position rightward; a filled cell means the query may attend to that key). Encoder-only: every token attends to every token. Decoder-only: each token attends only to itself and its past. Encoder--decoder: the source attends to itself bidirectionally, while each target token attends to the full source through cross-attention and to its own past causally.](../img/mdl-transformers-three-wirings.svg)
:label:`fig_three-wirings`

**Encoder-only.** Drop the mask. Every token attends to every other, so
each output vector summarizes the *whole* input as seen from its position.
Such a model does not directly implement the left-to-right autoregressive
factorization (with the future visible, next-token prediction is a copying
exercise — though masked models can still be decoded by iterative
re-masking, the idea text diffusion models develop), but it is the
strongest way to *represent* an input, and one
representation per token is exactly what classification, retrieval, and
tagging consume. BERT is this wiring pretrained on text
:cite:`Devlin.Chang.Lee.ea.2018`; the vision transformer of
:numref:`sec_vision-transformer` is the same wiring over image patches.

**Decoder-only.** Keep the causal mask everywhere: the GPT of
:numref:`sec_gpt`, in two sentences. Each token predicts its successor,
generation is the training objective run forward, and one stack serves
both understanding and production. We do not rebuild it here; it appears
in :numref:`fig_three-wirings` as the pattern the other two are measured
against.

**Encoder--decoder.** The original transformer
:cite:`Vaswani.Shazeer.Parmar.ea.2017` combines the two: a bidirectional
encoder reads the source sequence, and a causal decoder generates the
target while *cross-attending* into the encoder's output — queries from
the target stream, keys and values from the source, the second wiring of
:numref:`sec_multihead-attention`. The pattern in
:numref:`fig_three-wirings` shows the division: a full square for the
source, a full rectangle for target-to-source cross-attention, a triangle
for the target's own past. Machine translation was the founding
application; T5 pretrained the architecture on span reconstruction and
recast a broad family of tasks as text-to-text :cite:`raffel2020exploring`,
BART on denoising corrupted text :cite:`lewis2019bart`.

The rest of this section builds the two wirings that :numref:`sec_gpt` did
not, in order.

## An Encoder: Predicting from Both Sides

### A Bidirectional Encoder in a Dozen Lines

The encoder differs from the `CharLM` of :numref:`sec_transformer-block`
by a single argument: we pass no `valid_lens`, so nothing is masked.
Positions are still the model's job, since attention is permutation
equivariant (:numref:`sec_positional-information`), so a learned position
table stays.

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
        self.norm = nn.RMSNorm(num_hiddens)

    def forward(self, X):
        H = self.token_emb(X) + self.pos_emb(
            torch.arange(X.shape[1], device=X.device))
        for blk in self.blks:
            H = blk(H)  # no valid_lens: every token attends everywhere
        return self.norm(H)
```

```{.python .input #encoders-decoders-a-bidirectional-encoder-in-a-dozen-lines}
%%tab jax
class TransformerEncoder(nnx.Module):
    """Bidirectional encoder: embeddings plus unmasked pre-norm blocks."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=4,
                 max_len=64, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        init = nnx.initializers.normal(0.02)
        self.token_emb = nnx.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, num_hiddens, embedding_init=init,
                                 rngs=rngs)
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, rngs=rngs)
            for _ in range(num_blks)])
        self.norm = nnx.RMSNorm(num_hiddens, rngs=rngs)

    def __call__(self, X):
        H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = blk(H)  # no valid_lens: every token attends everywhere
        return self.norm(H)
```

### The Masked-Token Objective

What should this model train on? Not next-token prediction: with the
future visible, position $t$ can read token $t+1$ directly, and the loss
collapses without teaching anything. The objective must hide what it asks
for. *Masked language modeling* :cite:`Devlin.Chang.Lee.ea.2018` replaces
a random subset of tokens with a special `<mask>` symbol and asks the
model to reconstruct them from everything that remains:

$$
\max \; \sum_{t \in \mathcal{M}} \log p\left(x_t \mid \mathbf{x}_{\setminus \mathcal{M}}\right),
$$
:eqlabel:`eq_mlm`

where $\mathcal{M}$ is the masked set. Prediction at a masked position
draws on context from *both* sides, which is the point of dropping the
mask in the first place. We train the recipe in miniature on the
character-level Time Machine corpus of :numref:`sec_text-sequence`,
masking 15% of characters per window and giving `<mask>` one extra
embedding row; the tied output head is the same trick as in
:numref:`sec_transformer-block`.

```{.python .input #encoders-decoders-the-masked-token-objective}
%%tab pytorch
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000, num_val=3000)
MASK = len(data.vocab)  # id of the extra <mask> embedding row

def mask_tokens(X, p=0.15):
    mask = torch.rand(X.shape, device=X.device) < p
    return torch.where(mask, torch.full_like(X, MASK), X), mask

torch.manual_seed(0)
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
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000, num_val=3000)
MASK = len(data.vocab)  # id of the extra <mask> embedding row

def mask_tokens(X, key, p=0.15):
    mask = jax.random.uniform(key, X.shape) < p
    return jnp.where(mask, MASK, X), mask

model = TransformerEncoder(len(data.vocab) + 1, rngs=nnx.Rngs(0))
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

key, losses, step = jax.random.key(0), [], 0
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

A minute of training brings the masked loss to about one nat and the
masked-character accuracy to roughly 65% — against a 28-way vocabulary
whose unigram entropy alone is $2.83$ nats. More interesting than the
average is *where* the model earns it.

### What the Second Side Is Worth

Our windows are 64 characters long, and that finiteness builds a
comparison into every batch: a masked character in the interior has
context on both sides, while one at position 0 or 63 sees only one side
— the final position is exactly the situation a causal language model is
in at every step. Same model, same objective, so binning the validation
loss by position measures directly what the second side of the context is
worth.

```{.python .input #encoders-decoders-what-the-second-side-is-worth-1}
%%tab pytorch
model.eval()
pos_loss = torch.zeros(64, device=device)
pos_cnt = torch.zeros(64, device=device)
correct = total = 0
torch.manual_seed(1)
with torch.no_grad():
    for rep in range(5):
        for X, _ in data.val_dataloader():
            X = X.to(device)
            Xm, mask = mask_tokens(X)
            logits = F.linear(model(Xm), model.token_emb.weight)
            loss = F.cross_entropy(logits.transpose(1, 2), X,
                                   reduction='none')
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
key = jax.random.key(1)
for rep in range(5):
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

The profile is flat across the interior at about $1.1$ nats and roughly
doubles at the two one-sided edges. That gap is the second side's value,
measured: conditioning on both sides of a character cuts its loss by
about half relative to one side, because for text, what follows a gap
narrows it down about as sharply as what precedes it. The predictions are
readable too — mask a character and the model fills it from its
surroundings:

```{.python .input #encoders-decoders-what-the-second-side-is-worth-2}
%%tab pytorch
snippet = 'the time traveller for so it will be convenient to speak of him '
ids = torch.tensor(data.vocab[list(snippet)], device=device)[None]
for pos in (9, 14, 30):
    Xm = ids.clone()
    Xm[0, pos] = MASK
    with torch.no_grad():
        logits = F.linear(model(Xm), model.token_emb.weight)
    probs = F.softmax(logits[0, pos], -1)
    top = probs.topk(3)
    shown = snippet[:pos] + '_' + snippet[pos + 1:]
    print(f'{shown[:40]!r}... -> '
          + ', '.join(f'{data.vocab.to_tokens(int(i))!r} {p:.2f}'
                      for p, i in zip(top.values, top.indices)))
```

```{.python .input #encoders-decoders-what-the-second-side-is-worth-2}
%%tab jax
snippet = 'the time traveller for so it will be convenient to speak of him '
ids = jnp.asarray(data.vocab[list(snippet)])[None]
for pos in (9, 14, 30):
    Xm = ids.at[0, pos].set(MASK)
    probs = jax.nn.softmax(model.token_emb.attend(model(Xm))[0, pos])
    top = jnp.argsort(probs)[::-1][:3]
    shown = snippet[:pos] + '_' + snippet[pos + 1:]
    print(f'{shown[:40]!r}... -> '
          + ', '.join(f'{data.vocab.to_tokens(int(i))!r} {probs[i]:.2f}'
                      for i in top))
```

The doubled consonant in "trave_ler" comes back with high confidence:
only the right-hand context ("ler") pins it down, and a left-to-right
model would never see it. This experiment, scaled up (subword tokens,
sentence pairs, gigabytes of text, a fine-tuning recipe per downstream
task), is BERT :cite:`Devlin.Chang.Lee.ea.2018`, whose pretraining and
fine-tuning the Language Models part covers in full
(:numref:`chap_nlp_pretrain`). Nor did the wiring stop evolving in 2019:
ModernBERT rebuilds the same encoder-only architecture with the modern
block internals of this chapter, RoPE, gated FFN, alternating
local--global attention, and an 8k context, and remains the backbone of
choice for retrieval and classification at small model sizes
:cite:`Warner.Chaffin.Clavie.ea.2024`.

## An Encoder--Decoder: Cross-Attention at Work

### A Task Whose Alignment We Know

The encoder--decoder earns its keep when input and output are different
sequences. Its signature component, cross-attention, is usually
illustrated on machine translation, but a trained translation model gives
us no ground truth to check its attention against. So we choose a task
whose correct alignment is known by construction: *reverse a random
string*. Target position $t$ must copy source position $n-1-t$, the
letters are drawn independently at random, and the only path from source
content to the decoder runs through cross-attention. If the mechanism
works and the model solves the task directly, its attention map should be
the anti-diagonal — a prediction we can check. Random strings also mean
unlimited fresh data: unlike the memorizing GPT of :numref:`sec_gpt`,
this model never sees the same example twice.

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

src, dec_in, tgt = sample_batch(1, jax.random.key(0))
print(f'source {to_str(src[0])!r} -> target {to_str(tgt[0])!r}')
```

As in :numref:`sec_gpt`, training uses teacher forcing: the decoder input
is the target shifted right behind a `<bos>` token, so every position
learns to predict its successor in parallel.

### The Decoder Block: One More Sublayer

A decoder block is the transformer block plus one sublayer. Between the
causal self-attention and the FFN sits cross-attention: queries from the
target's residual stream, keys and values from the encoder output. It
follows the same pre-norm discipline as everything in this chapter — each
sublayer reads the stream through a normalization and adds its result
back. Note the masking asymmetry: self-attention stays causal (the target
is being generated), while cross-attention is unmasked (the source is
fully known before generation starts).

```{.python .input #encoders-decoders-the-decoder-block-one-more-sublayer-1}
%%tab pytorch
class DecoderBlock(nn.Module):
    """Pre-norm decoder block: causal self-attention, cross-attention,
    FFN."""
    def __init__(self, num_hiddens, num_heads):
        super().__init__()
        self.norm1 = nn.RMSNorm(num_hiddens)
        self.norm2 = nn.RMSNorm(num_hiddens)
        self.norm3 = nn.RMSNorm(num_hiddens)
        self.self_attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                     dropout=0)
        self.cross_attention = d2l.MultiHeadAttention(num_hiddens,
                                                      num_heads, dropout=0)
        self.ffn = d2l.FeedForward(num_hiddens)

    def forward(self, X, enc):
        B, T = X.shape[:2]
        causal = torch.arange(1, T + 1, device=X.device).repeat(B, 1)
        Y = self.norm1(X)
        X = X + self.self_attention(Y, Y, Y, causal)
        X = X + self.cross_attention(self.norm2(X), enc, enc, None)
        return X + self.ffn(self.norm3(X))
```

```{.python .input #encoders-decoders-the-decoder-block-one-more-sublayer-1}
%%tab jax
class DecoderBlock(nnx.Module):
    """Pre-norm decoder block: causal self-attention, cross-attention,
    FFN."""
    def __init__(self, num_hiddens, num_heads, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.norm1 = nnx.RMSNorm(num_hiddens, rngs=rngs)
        self.norm2 = nnx.RMSNorm(num_hiddens, rngs=rngs)
        self.norm3 = nnx.RMSNorm(num_hiddens, rngs=rngs)
        self.self_attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                     dropout=0, rngs=rngs)
        self.cross_attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout=0, rngs=rngs)
        self.ffn = d2l.FeedForward(num_hiddens, rngs=rngs)

    def __call__(self, X, enc):
        B, T = X.shape[:2]
        causal = jnp.tile(jnp.arange(1, T + 1), (B, 1))
        Y = self.norm1(X)
        X = X + self.self_attention(Y, Y, Y, causal)[0]
        Y2, cross_weights = self.cross_attention(self.norm2(X), enc, enc,
                                                 None)
        X = X + Y2
        return X + self.ffn(self.norm3(X)), cross_weights
```

The full model wires a `TransformerEncoder` (the class from the previous
section, reused unchanged) to a stack of decoder blocks with its own
embeddings and output head. Following the framework conventions of
:numref:`sec_multihead-attention`, the JAX model returns the
cross-attention weights alongside the logits, while the PyTorch model
stores them on the attention module.

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
        self.norm = nn.RMSNorm(num_hiddens)
        self.head = nn.Linear(num_hiddens, vocab_size, bias=False)

    def forward(self, src, dec_in):
        enc = self.encoder(src)
        H = self.tgt_emb(dec_in) + self.pos_emb(
            torch.arange(dec_in.shape[1], device=dec_in.device))
        for blk in self.blks:
            H = blk(H, enc)
        return self.head(self.norm(H))
```

```{.python .input #encoders-decoders-the-decoder-block-one-more-sublayer-2}
%%tab jax
class EncoderDecoder(nnx.Module):
    """A causal decoder cross-attending into a bidirectional encoder."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4,
                 num_blks=1, max_len=64, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
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
        self.norm = nnx.RMSNorm(num_hiddens, rngs=rngs)
        self.head = nnx.Linear(num_hiddens, vocab_size, use_bias=False,
                               rngs=rngs)

    def __call__(self, src, dec_in):
        enc = self.encoder(src)
        H = self.tgt_emb(dec_in) + self.pos_emb(jnp.arange(dec_in.shape[1]))
        weights = []
        for blk in self.blks:
            H, w = blk(H, enc)
            weights.append(w)
        return self.head(self.norm(H)), weights
```

### Assembling the Masks

Our reversal task ducks one bookkeeping question by construction: every
string has the same length, so nothing is padding, and the only mask in
sight is the decoder's causal one. A real batch — sequences of different
lengths, padded to a rectangle — needs three masks, one per attention
site, each composed from the two primitives of
:numref:`sec_attention-scoring-functions`: a padding mask built from valid
lengths, and the causal triangle, combined by logical AND under
broadcasting. Encoder self-attention masks source padding. Decoder
self-attention during teacher-forced training needs the causal triangle
*and* target padding on the key side (padded query rows compute outputs
the loss ignores). Cross-attention masks source padding again — the target may
be mid-generation, but the source it reads is fully known. The cell below
assembles all three for a toy ragged batch, as boolean arrays of shape
(batch, queries, keys) — the form the fused kernels of
:numref:`sec_attention-at-scale` accept; `d2l.MultiHeadAttention`'s
`valid_lens` argument carries the same information in compressed form.

```{.python .input #encoders-decoders-assembling-the-masks}
%%tab pytorch
src_len, tgt_len = torch.tensor([3, 5]), torch.tensor([3, 4])
S, T_dec = int(src_len.max()), int(tgt_len.max())
src_valid = (torch.arange(S)[None, :] < src_len[:, None])[:, None, :]
tgt_valid = (torch.arange(T_dec)[None, :] < tgt_len[:, None])[:, None, :]
causal = (torch.arange(T_dec)[None, :]
          <= torch.arange(T_dec)[:, None])[None]        # (1, T, T)
enc_self = src_valid.expand(-1, S, -1)     # (B, S, S): source padding
dec_self = causal & tgt_valid              # (B, T, T): causal AND padding
cross = src_valid.expand(-1, T_dec, -1)    # (B, T, S): source padding
for name, m in (('encoder self', enc_self), ('decoder self', dec_self),
                ('cross', cross)):
    print(f'{name}-attention mask, sequence 0 (1 = may attend):')
    print(m[0].int().numpy())
```

```{.python .input #encoders-decoders-assembling-the-masks}
%%tab jax
src_len, tgt_len = jnp.array([3, 5]), jnp.array([3, 4])
S, T_dec = int(src_len.max()), int(tgt_len.max())
src_valid = (jnp.arange(S)[None, :] < src_len[:, None])[:, None, :]
tgt_valid = (jnp.arange(T_dec)[None, :] < tgt_len[:, None])[:, None, :]
causal = (jnp.arange(T_dec)[None, :]
          <= jnp.arange(T_dec)[:, None])[None]          # (1, T, T)
B = len(src_len)
enc_self = jnp.broadcast_to(src_valid, (B, S, S))   # source padding
dec_self = causal & tgt_valid              # (B, T, T): causal AND padding
cross = jnp.broadcast_to(src_valid, (B, T_dec, S))  # source padding
for name, m in (('encoder self', enc_self), ('decoder self', dec_self),
                ('cross', cross)):
    print(f'{name}-attention mask, sequence 0 (1 = may attend):')
    print(m[0].astype(int))
```

Read sequence 0's three grids (source length 3 of 5, target length 3 of
4): the encoder square and the cross rectangle both blank the same two
padded source columns, and the decoder triangle loses its last column to
target padding. In this section's model the composition never surfaces —
`sample_batch` produces no padding, and the cross-attention call passes
`valid_lens=None` — but these three grids are what a production
encoder--decoder assembles for every batch it trains on.

### Training and Decoding

One encoder block, one decoder block, and a few hundred steps of on-line
batches suffice — the model sees about ten million characters, none twice.

```{.python .input #encoders-decoders-training-and-decoding-1}
%%tab pytorch
torch.manual_seed(0)
seq2seq = EncoderDecoder(V + 1).to(device)
optimizer = torch.optim.AdamW(seq2seq.parameters(), lr=1e-3,
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
seq2seq = EncoderDecoder(V + 1, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(seq2seq, optax.adamw(1e-3, weight_decay=0.0),
                          wrt=nnx.Param)

@nnx.jit
def s2s_step(model, optimizer, src, dec_in, tgt):
    def loss_fn(model):
        logits, _ = model(src, dec_in)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

key, losses = jax.random.key(0), []
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
torch.manual_seed(1)
with torch.no_grad():
    src, _, tgt = sample_batch(1000)
    enc = seq2seq.encoder(src)
    out = torch.full((1000, T + 1), BOS, device=device)
    for t in range(T):
        H = seq2seq.tgt_emb(out[:, :t + 1]) + seq2seq.pos_emb(
            torch.arange(t + 1, device=device))
        for blk in seq2seq.blks:
            H = blk(H, enc)
        out[:, t + 1] = seq2seq.head(seq2seq.norm(H))[:, -1].argmax(-1)
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
    H = model.tgt_emb(buf) + model.pos_emb(jnp.arange(buf.shape[1]))
    for blk in model.blks:
        H, _ = blk(H, enc)
    return model.head(model.norm(H))[:, t].argmax(-1)

src, _, tgt = sample_batch(1000, jax.random.key(1))
enc = seq2seq.encoder(src)
buf = jnp.full((1000, T + 1), BOS)
for t in range(T):
    buf = buf.at[:, t + 1].set(next_token(seq2seq, enc, buf, t))
pred = buf[:, 1:]
print(f'exact match on 1000 fresh strings: '
      f'{(pred == tgt).all(-1).mean():.3f}')
print(f'source {to_str(src[0])!r} -> predicted {to_str(pred[0])!r}')
```

:begin_tab:`jax`
The decode loop reuses the fixed-buffer idiom from :numref:`sec_gpt`'s
`generate`: a buffer of the final size, overwritten left to right, keeps
every shape static so the jitted step compiles exactly once. The causal
mask makes the not-yet-written positions invisible to every query at or
before $t$.
:end_tab:

### Reading the Alignment

Now the promised check. We run a batch through the model, pull the
cross-attention weights out of the decoder block, and compare each target
position's attention against the alignment the task dictates.

```{.python .input #encoders-decoders-reading-the-alignment}
%%tab pytorch
with torch.no_grad():
    src, dec_in, tgt = sample_batch(64)
    seq2seq(src, dec_in)
w = seq2seq.blks[0].cross_attention.attention.attention_weights
w = w.reshape(64, 4, T, T)
want = torch.arange(T - 1, -1, -1, device=device)  # row t -> column T-1-t
hit = (w.mean(1).argmax(-1) == want).float().mean()
mass = w.mean(1).gather(-1, want.expand(64, T)[..., None]).mean()
print(f'head-averaged argmax hits the true source position on '
      f'{100 * hit:.0f}% of rows; mean weight there {mass:.2f}')
d2l.show_heatmaps(w[0].cpu()[None], xlabel='source position',
                  ylabel='target position',
                  titles=[f'Head {i}' for i in range(1, 5)],
                  figsize=(9, 2.5), cmap='Blues')
```

```{.python .input #encoders-decoders-reading-the-alignment}
%%tab jax
src, dec_in, tgt = sample_batch(64, jax.random.key(2))
_, cross_weights = seq2seq(src, dec_in)
w = cross_weights[0].reshape(64, 4, T, T)
want = jnp.arange(T - 1, -1, -1)                   # row t -> column T-1-t
hit = (w.mean(1).argmax(-1) == want).mean()
mass = jnp.take_along_axis(w.mean(1),
                           jnp.tile(want, (64, 1))[..., None], -1).mean()
print(f'head-averaged argmax hits the true source position on '
      f'{100 * hit:.0f}% of rows; mean weight there {mass:.2f}')
d2l.show_heatmaps(w[0][None], xlabel='source position',
                  ylabel='target position',
                  titles=[f'Head {i}' for i in range(1, 5)],
                  figsize=(9, 2.5), cmap='Blues')
```

The maps show the anti-diagonal of :numref:`fig_three-wirings`'s third
panel made real: averaged over heads, the argmax lands on the true source
position for well over nine rows in ten, with most of the softmax mass
concentrated there. The model has *learned* the alignment we built into
the task, and cross-attention is where it lives — which is exactly the
kind of readable evidence :numref:`sec_multihead-attention` warned is the
exception rather than the rule. It is readable here because we made the
model small; give the encoder and decoder more depth and heads and the
task stays solved while the maps delocalize, as one of the exercises
demonstrates.

## Cross-Attention as Interface

### Queries Need Not Come from a Sequence

In the decoder, the cross-attention queries came from the target stream.
Nothing in the mechanism requires that. A query is just a vector, and a
set of $M$ query vectors can simply be *learned parameters* — a fixed
array that exists before any input arrives. Cross-attending it into an
input of length $N$ costs $O(MN)$; the quadratic term of
:numref:`sec_attention-at-scale` never appears, because the $M$ latents
only ever self-attend among themselves at $O(M^2)$. This is the
*Perceiver* :cite:`Jaegle.Gimeno.Brock.ea.2021`: a fixed-size latent
bottleneck that reads arbitrarily long, arbitrarily structured
inputs through cross-attention, sketched in
:numref:`fig_latent-bottleneck`.

![The latent bottleneck. A learned array of M latents (M much smaller than the input length N) reads the input once through cross-attention at cost O(MN); all further processing is self-attention and FFN among the latents at cost O(M squared), independent of N.](../img/mdl-transformers-latent-bottleneck.svg)
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
        self.norm_q = nn.RMSNorm(num_hiddens)
        self.cross_attention = d2l.MultiHeadAttention(num_hiddens,
                                                      num_heads, dropout=0)
        self.blks = nn.ModuleList([
            d2l.TransformerBlock(num_hiddens, num_heads)
            for _ in range(num_blks)])
        self.norm = nn.RMSNorm(num_hiddens)

    def forward(self, X):
        Z = self.latents.expand(X.shape[0], -1, -1)
        Z = Z + self.cross_attention(self.norm_q(Z), X, X, None)  # O(MN)
        for blk in self.blks:
            Z = blk(Z)                                            # O(M^2)
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
                 rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.latents = nnx.Param(
            0.02 * jax.random.normal(rngs.params(),
                                     (num_latents, num_hiddens)))
        self.norm_q = nnx.RMSNorm(num_hiddens, rngs=rngs)
        self.cross_attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout=0, rngs=rngs)
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, rngs=rngs)
            for _ in range(num_blks)])
        self.norm = nnx.RMSNorm(num_hiddens, rngs=rngs)

    def __call__(self, X):
        Z = jnp.broadcast_to(self.latents[...],
                             (X.shape[0],) + self.latents.shape)
        Z = Z + self.cross_attention(self.norm_q(Z), X, X, None)[0]  # O(MN)
        for blk in self.blks:
            Z = blk(Z)                                               # O(M^2)
        return self.norm(Z)

perceiver = PerceiverEncoder(num_latents=64, num_hiddens=256)
perceiver.eval()
X = jax.random.normal(jax.random.key(0), (1, 4096, 256))
print('input', X.shape, '-> latent summary', perceiver(X).shape)
```

Whatever the input length, the output is $M = 64$ vectors. The input here
is a raw feature sequence rather than token embeddings, because the whole
point is indifference to what the input is — text, audio frames, image
pixels, or a concatenation of all three.

### The Cost Curve

The claim to verify is the shape of the cost. We time the Perceiver
against the direct alternative (the same two transformer blocks applied
to the full input sequence) as $N$ doubles at fixed $M = 64$.

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

def timed(model, X, reps=20):
    with torch.no_grad():
        for _ in range(3):
            model(X)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            model(X)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1e3

lengths, t_full, t_perc = (1024, 2048, 4096, 8192), [], []
for N in lengths:
    X = torch.randn(1, N, 256, device=device)
    t_full.append(timed(full, X))
    t_perc.append(timed(perceiver, X))
    print(f'N={N:5d}: self-attention {t_full[-1]:6.2f} ms, '
          f'perceiver {t_perc[-1]:5.2f} ms')
d2l.plot(list(lengths), [t_full, t_perc], 'input length N',
         'forward time (ms)', xscale='log', yscale='log',
         legend=['self-attention encoder', 'perceiver encoder'])
```

```{.python .input #encoders-decoders-the-cost-curve}
%%tab jax
class SelfAttentionEncoder(nnx.Module):
    """The comparison: the same blocks over the full input sequence."""
    def __init__(self, num_hiddens, num_heads=4, num_blks=2, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, rngs=rngs)
            for _ in range(num_blks)])

    def __call__(self, X):
        for blk in self.blks:
            X = blk(X)                                            # O(N^2)
        return X

full = SelfAttentionEncoder(256)
full.eval()
jit_full, jit_perc = nnx.jit(full), nnx.jit(perceiver)

def timed(f, X, reps=20):
    f(X).block_until_ready()  # warmup + compile
    t0 = time.time()
    for _ in range(reps):
        Y = f(X)
    Y.block_until_ready()
    return (time.time() - t0) / reps * 1e3

lengths, t_full, t_perc = (1024, 2048, 4096, 8192), [], []
for N in lengths:
    X = jax.random.normal(jax.random.key(0), (1, N, 256))
    t_full.append(timed(jit_full, X))
    t_perc.append(timed(jit_perc, X))
    print(f'N={N:5d}: self-attention {t_full[-1]:6.2f} ms, '
          f'perceiver {t_perc[-1]:5.2f} ms')
d2l.plot(list(lengths), [t_full, t_perc], 'input length N',
         'forward time (ms)', xscale='log', yscale='log',
         legend=['self-attention encoder', 'perceiver encoder'])
```

Each doubling of $N$ eventually multiplies the self-attention encoder's
time by about four, the signature of an $N^2$ term taking over. The
Perceiver's time barely moves (its $O(MN)$ cross-attention grows
linearly but stays dominated by the fixed $O(M^2)$ latent processing),
and by $N = 8192$ the gap exceeds an order of magnitude. The left end of
the plot belongs in the reading too: at short inputs the bottleneck
buys nothing, and full self-attention is as fast or faster. A latent
bottleneck is worth having when the input is long and a fixed-size
summary of it suffices.

### Perceiver IO and the Idea's Descendants

Reading through learned queries has a mirror image: *writing* through
them. Perceiver IO :cite:`Jaegle.Borgeaud.Alayrac.ea.2022` adds an output
query array that cross-attends *out of* the latent summary, so the output
size and shape are set by the queries rather than by the input — one
query for a classification label, one per pixel for optical flow, one per
audio sample for a waveform. Input length, latent width, and output shape
become three independent dials, with everything in between a fixed-cost
transformer.

The pattern's descendants run through today's multimodal systems.
Flamingo's Perceiver resampler compresses a variable number of image and
video features into a fixed handful of visual tokens before a frozen
language model ever sees them :cite:`alayrac2022flamingo`; BLIP-2's
Q-Former is a small stack of learned query tokens that bridges a frozen
vision encoder and a frozen LLM :cite:`Li.Li.Savarese.ea.2023`; learned
query tokens remain one of the two standard interfaces in current
vision--language models — the other is a plain learned projection applied
patch by patch. DETR had already used the query-token device for detection — a
hundred learned object queries cross-attend into image features, each
producing one detection :cite:`Carion.Massa.Synnaeve.ea.2020`. The Image
Models part (:numref:`chap_cv`) takes up DETR and its successors in
depth.

## Which Wiring When

The taxonomy of :numref:`fig_three-wirings` maps cleanly onto current
practice:

| Wiring | Exemplars today | Typical use |
|---|---|---|
| encoder-only | BERT descendants, ModernBERT | embeddings, retrieval, classification |
| encoder--decoder | T5 family, Whisper | translation, speech recognition |
| decoder-only | essentially everything else | generation, chat, in-context everything |

Encoder-only survives where the product *is* the representation: an
embedding model is run once per document, at bidirectional quality, with
no generation loop to pay for — which is why retrieval and reranking
stacks still train BERT-shaped models
:cite:`Warner.Chaffin.Clavie.ea.2024`. The encoder--decoder survives
where the input is fully known before generation starts and deserves its
own tower: T5-style text-to-text :cite:`raffel2020exploring`, and Whisper,
whose encoder reads an entire audio clip bidirectionally while a text
decoder cross-attends into it :cite:`radford2023whisper`.

Everything else went decoder-only, and the reasons compound. One stack
means one pretraining objective: next-token prediction on raw text, with
no masking scheme or span-corruption design to engineer. Every parameter
serves both understanding and generation instead of splitting the budget
between towers. Generation needs no separate machinery, and in-context
learning falls out of it :cite:`brown2020language`: a task description in
the prompt does what a fine-tuned head used to. When one architecture,
trained one way, covers everything from chat to code with the same
serving stack, the coordination costs of maintaining three wirings stop
being worth paying — except in the niches above, where the other two are
simply better tools.

## Summary

One transformer block supports three wirings. Removing the causal mask
gives an encoder-only model: not a left-to-right generator, but the
strongest way to compute one representation per token, trained by masking
tokens and
predicting them from both sides; in our character-level demo, positions
with context on both sides had roughly half the loss of one-sided
positions, which is the bidirectional advantage in one number.
The encoder--decoder joins an unmasked encoder to a causal decoder
through cross-attention; on a reversal task whose true alignment is known
by construction, the learned cross-attention map reproduces the
anti-diagonal almost exactly. Cross-attention also works with queries
that come from no sequence at all: a learned latent array reading a
length-$N$ input costs $O(MN)$ instead of $O(N^2)$, stays nearly constant
in measured time as $N$ grows, and under the names Perceiver, resampler,
and Q-Former is how today's multimodal models feed long perceptual
streams into fixed-size computation. In current practice encoders own
representation (retrieval, classification), encoder--decoders own tasks
with a fully-known input in another modality or length regime
(translation, speech), and decoder-only owns the rest — one model, one
objective, generation for free.

## Exercises

1. Our masking always writes `<mask>`. BERT instead replaces the chosen
   position with `<mask>` 80% of the time, a random token 10%, and the
   original token 10% :cite:`Devlin.Chang.Lee.ea.2018` — partly because
   `<mask>` never appears when the pretrained encoder is used on real
   text. Implement the 80/10/10 rule and compare masked accuracy and the
   loss on inputs containing no `<mask>` at all.
2. Mask two *adjacent* characters instead of one and evaluate the loss at
   both positions. Explain the change using the per-position analysis of
   this section: what context did each masked character lose?
3. Widen the encoder--decoder to `num_blks=2` and rerun the alignment
   check and the heatmaps. The task remains solved; what happens to the
   argmax hit rate and the attention mass on the true source position?
   Reconcile this with the warnings about reading attention maps in
   :numref:`sec_multihead-attention`.
4. Change one line of `sample_batch` to make the task *copy* instead of
   reverse, and predict the heatmap before rerunning. Then try
   reverse-then-copy with `T` even: target = reversed source for the
   first half's length, then the source itself. What alignment do you
   expect in each half of the map?
5. Sweep the number of latents $M \in \{16, 64, 256\}$ in the cost-curve
   experiment. Where does the crossover with full self-attention move,
   and why? Derive the FLOP count of `PerceiverEncoder.forward` as a
   function of $M$, $N$, and $d$, and check which term dominates at each
   $M$.
6. Build a minimal Perceiver IO head: a second learned query array of
   length $K$ that cross-attends into the latent summary and produces
   output shape $(B, K, d)$. Verify the shape, then argue the total cost
   is $O(MN + M^2 + KM)$ and state when this beats attaching the output
   queries to the input directly.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.4]{.kicker}

Encoders, decoders, and cross-attention<br>
**three wirings of one block · predicting from both sides · an alignment you can verify · the latent bottleneck**
:::
:::

::: {.slide title="Three wirings of one block"}
The block leaves two questions open: **which positions may attend to
which**, and **where keys and values come from**.

@fig:mdl-transformers-three-wirings

- **Encoder-only**: no mask — no left-to-right generation, built to
  *represent* (BERT; the ViT is this wiring over patches).
- **Decoder-only**: the GPT of the previous sections — generation is the
  objective run forward.
- **Encoder–decoder**: causal decoder *cross-attends* into a
  bidirectional encoder (the 2017 original).
:::

::: {.slide title="A bidirectional encoder in a dozen lines"}
One difference from `CharLM`: no `valid_lens` — nothing is masked.

@encoders-decoders-a-bidirectional-encoder-in-a-dozen-lines
:::

::: {.slide title="The masked-token objective"}
Next-token prediction is trivial when the future is visible. Hide what
you ask for:

$$\max \; \sum_{t \in \mathcal{M}} \log p\left(x_t \mid \mathbf{x}_{\setminus \mathcal{M}}\right)$$

@!encoders-decoders-the-masked-token-objective
:::

::: {.slide title="What the second side is worth"}
Window edges are one-sided by construction — position 63 is where a
causal LM lives *permanently*. Bin the loss by position:

@!encoders-decoders-what-the-second-side-is-worth-1

Flat interior at ~1.1 nats, roughly **double** at the one-sided edges.
:::

::: {.slide title="Predictions you can read"}
@!encoders-decoders-what-the-second-side-is-worth-2

Only the *right* context pins down "trave_ler".

::: {.d2l-note}
Scaled up — subwords, sentence pairs, gigabytes — this is BERT (2018).
ModernBERT (2024) refits the same wiring with this chapter's block:
RoPE, gated FFN, local–global attention, 8k context.
:::
:::

::: {.slide title="A task whose alignment we know"}
Machine translation gives no ground truth for its attention. Reversal
does: target $t$ **must** copy source $n-1-t$, and content only flows
through cross-attention.

@!encoders-decoders-a-task-whose-alignment-we-know

Random strings = unlimited fresh data — no memorization possible.
:::

::: {.slide title="The decoder block: one more sublayer"}
Causal self-attention → **cross-attention** (queries: target stream;
keys/values: encoder output) → FFN. Self-attention masked,
cross-attention not — the source is fully known.

@encoders-decoders-the-decoder-block-one-more-sublayer-1
:::

::: {.slide title="Three masks, two primitives"}
A ragged real batch composes every mask from a padding mask and the
causal triangle (§10.2): source padding for encoder self-attention,
causal ∧ target padding for decoder self-attention, source padding again
for cross-attention:

@encoders-decoders-assembling-the-masks
:::

::: {.slide title="Train it, decode it"}
One encoder block, one decoder block, 600 steps of on-line batches:

@!encoders-decoders-training-and-decoding-1

@!encoders-decoders-training-and-decoding-2
:::

::: {.slide title="Reading the alignment"}
@!encoders-decoders-reading-the-alignment

The anti-diagonal, learned — and verifiable, because the task fixes the
answer. Deeper, more-headed models solve the task with *delocalized*
maps (exercise): readable attention is the exception.
:::

::: {.slide title="Cross-attention as interface"}
A query is just a vector — it can be a **learned parameter**. $M$
latents read a length-$N$ input:

@fig:mdl-transformers-latent-bottleneck

$O(MN)$ to read, $O(M^2)$ to think — the $N^2$ term never appears
(Perceiver, 2021).
:::

::: {.slide title="A minimal Perceiver"}
@encoders-decoders-queries-need-not-come-from-a-sequence
:::

::: {.slide title="The cost curve"}
@!encoders-decoders-the-cost-curve

Self-attention: ~4× per doubling of $N$. Perceiver: barely moves; over
an order of magnitude ahead by $N = 8192$. At short inputs the
bottleneck buys nothing.
:::

::: {.slide title="Which wiring when"}
- **Encoder-only** — the product is the representation: embeddings,
  retrieval, classification (BERT descendants, ModernBERT).
- **Encoder–decoder** — input fully known, own tower worth it: T5
  text-to-text, Whisper speech recognition.
- **Decoder-only** — everything else: one stack, one objective,
  generation free, in-context learning included.

::: {.d2l-note}
Perceiver descendants live on inside multimodal models: Flamingo's
resampler, BLIP-2's Q-Former, DETR's object queries.
:::
:::

::: {.slide title="Recap"}
- Three wirings, one block: bidirectional / causal / both + cross.
- Masked prediction trains encoders; both-sides context cuts the loss
  roughly in half vs. one side.
- Cross-attention verified: the reversal task's anti-diagonal appears in
  the learned map.
- Learned queries turn attention into an interface: $O(MN)$, flat cost
  curve, Perceiver → resampler → Q-Former.
- Encoders represent, encoder–decoders translate and transcribe,
  decoders do the rest.
:::
