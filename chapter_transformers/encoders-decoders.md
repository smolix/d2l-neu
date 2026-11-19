# Encoders, Decoders, and Cross-Attention
:label:`sec_transformer`

To fill the gap in "the ___ barked at the mail carrier", a reader uses the
words after the gap as much as those before it. To translate a sentence, a
translator reads all of it before writing the first word of the
translation. The GPT of :numref:`sec_gpt`, which the last two sections made
cheaper to run, can do neither: its causal mask lets each position see only
what came before it, which is all that next-token prediction needs. What
does a transformer gain when it may see its whole input, and how does it
produce one sequence from another?

Both questions concern two choices that the GPT fixed: the mask, and the
sources of the queries, keys, and values. Removing the mask gives an
*encoder*, which reads in both directions and produces contextual
representations. Joining a masked stack to an unmasked one through the
cross-attention of :numref:`sec_multihead-attention` gives the
*encoder--decoder*, the form in which transformers began, as a model for
machine translation :cite:`Vaswani.Shazeer.Parmar.ea.2017`. We build both
from the same `d2l.TransformerBlock` and test each on a task small enough
to check. How much does the text after a byte help to predict it? We
measure this against the same encoder restricted to one side. Does
cross-attention look where it should? On a task whose correct alignment
we know in advance, we can simply read the map. Then the queries leave the
sequence altogether and become a handful of learned vectors, which is how
many vision--language models hand an image to a language model. We close
with the question a practitioner brings to this section: which wiring to
use today.

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

A transformer block maps a sequence of $d$-dimensional vectors to a
sequence of the same shape. It leaves two questions open: which
positions may attend to which, and where the keys and values come from.
Three common wirings answer these questions.
:numref:`fig_three-wirings` draws them in the convention we will use for
every attention map in this section: queries down the rows, keys along
the columns.

![The three wirings and their attention patterns (query position runs downward, key position rightward; a filled cell means the query may attend to that key). Encoder-only: every token attends to every token. Decoder-only: each token attends only to itself and its past. Encoder--decoder: the source attends to itself bidirectionally, while each target token attends to the full source through cross-attention (light cells) and to its own past causally. The third panel hides one structural fact: its source rows and its target rows belong to two separate stacks, and every decoder block reads the output of the last encoder block.](../img/mdl-transformers-three-wirings.svg)
:label:`fig_three-wirings`

**Encoder-only.** Drop the mask. Every token attends to every other, so
each output vector summarizes the whole input as seen from its position.
Such a model gives up next-token prediction, for a reason we meet shortly.
It is the wiring to choose when the answer is a property of the input (a
tag per token, a label, a vector) rather than a continuation of it.
Tagging, such as marking the names in a sentence, uses the output at each
token. Classification and retrieval need one vector for the whole input
and obtain it by *pooling*: averaging the outputs over all positions, or
reading the output at one designated position. BERT is this wiring
pretrained on text :cite:`Devlin.Chang.Lee.ea.2018`; the vision
transformer of :numref:`sec_vision-transformer` is the same wiring over
image patches, pooled through a designated position.

**Decoder-only.** Keep the causal mask everywhere: this is the GPT of
:numref:`sec_gpt`. Each token predicts its successor, and generation is the
training objective run forward. Large enough models of this kind, trained
on raw text alone, perform tasks that are specified only in the prompt
:cite:`brown2020language`, including tasks that come with an input of
their own: put the input in the prompt, and the continuation is the
output. We do not rebuild this wiring here; it is the pattern against
which the other two are compared.

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
recast a broad family of tasks as text-to-text :cite:`raffel2020exploring`.
In speech recognition, Whisper's encoder reads an entire audio clip
bidirectionally while a text decoder cross-attends into it
:cite:`radford2023whisper`.

## Bidirectional Context for Masked-Token Prediction

### A Bidirectional Encoder

The encoder differs from the GPT of :numref:`sec_gpt` in one essential
respect: its blocks receive no mask unless one is passed, so every token
attends to every token. It also handles positions in the older way, with a
learned table added to the token embeddings, as the `'learned'` option of
`GPT` does. Some position signal is needed, since attention alone treats
its input as a set (:numref:`sec_positional-information`). We use the
table only because `d2l.TransformerBlock` has no rotary embedding built in;
RoPE would serve as well, and the recent encoders mentioned below use it.

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

### The Masked-Token Objective

This model cannot be trained on next-token prediction. With the future
visible, position $t$ can read token $t+1$ directly, and the loss
collapses without teaching anything. The objective must hide what it asks
for. *Masked language modeling* :cite:`Devlin.Chang.Lee.ea.2018` replaces
a random subset $\mathcal{M}$ of the tokens with a special `<mask>` symbol
and asks the model to reconstruct them from everything that remains. The
masked positions stay in the input, as `<mask>` tokens with their position
embeddings, so the model knows where the gaps are. Writing
$\mathbf{x}_{\setminus \mathcal{M}}$ for this corrupted sequence, training
seeks the parameters $\theta$ that achieve

$$
\max_\theta \; \sum_{t \in \mathcal{M}} \log p_\theta\left(x_t \mid \mathbf{x}_{\setminus \mathcal{M}}\right),
$$
:eqlabel:`eq_mlm`

averaged over texts and over the random choice of $\mathcal{M}$. Only the
masked tokens are scored, and each is predicted from context on *both*
sides, which is what removing the causal mask provides. We mask 15% of the
tokens, as BERT does; Exercise :exref:`ex_encdec-masking-recipe` varies
the recipe.

Let us start small, and on purpose. BERT trained on subword tokens and
billions of words; we train the same objective on `d2l.WikiText103`, the
bytes of the Wikipedia articles of :numref:`subsec_tinylm`, in windows of
64 bytes, with an encoder of four blocks of width 128 and under a million
parameters. What we want to see, the value of the second side, already
shows at this scale. `<mask>` gets one extra embedding row after the 256
byte values. `MaskedBytes` sets the label of every unmasked position to
$-100$, which `d2l.train_steps` (:numref:`sec_test_models`) leaves out of
the loss, and `MaskedLM` adds an output head tied to the token embedding,
as in :numref:`sec_gpt`. One reference point before the run: a model that
knows only how often each byte occurs has a loss of 3.17 nats per byte,
the unigram entropy measured in :numref:`sec_gpt`.

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

At step 500 the model is still near that unigram level: it has learned
the byte frequencies and little else. Then it starts to use the context,
and by step 2,000 the masked loss is down to about 1.6 to 1.8 nats.

### Loss by Available Context
:label:`subsec_mlm-loss-by-position`

Our windows are 64 bytes long, so every batch contains a comparison. A
masked byte in the interior has context on both sides, while one at
position 0 sees only what follows it and one at position 63 only what
precedes it. The model and the objective are the same at every position.
If the second side helps, the validation loss binned by position should be
flat across the interior and rise at both edges. `loss_by_position`
computes this profile, together with the fraction of masked bytes that the
model fills exactly.

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

The model fills about half of the masked bytes exactly, and the profile has
the expected shape: about 1.6 to 1.7 nats across the interior, 2.4 to 2.5 at
the two edges. It also shows how far the model looks around a byte. One byte
on the missing side, at position 1 or 62, already removes most of the gap, and
by position 2 or 3 the loss is back at its interior level: a byte is predicted
mainly from its nearest neighbours. The two edges cost about the same,
although one sees only the future and the other only the past. This is no
accident of English. For any stationary source, predicting a symbol from the
$k$ symbols that follow it is exactly as hard as predicting it from the $k$
that precede it (Exercise :exref:`ex_encdec-times-arrow`).

Is the gap between edge and interior, about 0.8 nats, what the second side
is worth? Not quite, because the model has rarely practised the edges. An
interior byte usually has a visible neighbour on each side, while
position 63 never has one on its right and is one position in 64: in 2,000
steps of 64 windows the model sees about 19,000 masked bytes there, against
1.2 million in all. A fair comparison needs a model that has one side of
context everywhere and trains on it everywhere. We therefore give the same
encoder a causal mask in every block and train it on the same masked
objective for the same 2,000 steps. A third model uses this causal encoder
as a language model, predicting every next byte, so that all 64 positions
of a window add to its loss rather than about 10. The two-sided model is
trained once more and the one-sided masked model twice, so that each
comparison comes with a run-to-run range.

```{.python .input #encoders-decoders-a-causal-control}
%%tab pytorch
class LeftOnlyLM(MaskedLM):
    """MaskedLM with a causal mask in every block: one side of context."""
    def forward(self, X):
        mask = d2l.causal_mask(X.shape[1], device=X.device)
        return F.linear(self.encoder(X, mask), self.encoder.token_emb.weight)

def trained(lm_class, batches):
    lm = lm_class(TransformerEncoder(len(data.vocab) + 1)).to(device)
    d2l.train_steps(lm, batches, torch.optim.AdamW(
        lm.parameters(), lr=1e-3, weight_decay=0.0), 2000)
    return lm

arms = {  # each run is the loss at each position of the window
    'both sides, masked': [pos_loss, loss_by_position(
        trained(MaskedLM, masked), masked.val_dataloader())[0]],
    'left side, masked': [loss_by_position(
        trained(LeftOnlyLM, masked), masked.val_dataloader())[0]
        for _ in range(2)],
    'left side, next byte': [loss_by_position(
        trained(LeftOnlyLM, data), data.val_dataloader())[0]]}
for name, runs in arms.items():
    print(f'{name:21s} interior ' + show(p[interior].mean() for p in runs)
          + '; position 63 ' + show(p[-1] for p in runs))
# Next byte: the loss at position t is that of predicting byte t + 1
d2l.plot([torch.arange(64)] * 2 + [torch.arange(1, 65)],
         [runs[0] for runs in arms.values()], 'position in the window',
         'loss', legend=list(arms))
```

```{.python .input #encoders-decoders-a-causal-control}
%%tab jax
class LeftOnlyLM(MaskedLM):
    """MaskedLM with a causal mask in every block: one side of context."""
    def __call__(self, X):
        mask = d2l.causal_mask(X.shape[1])
        return self.encoder.token_emb.attend(self.encoder(X, mask))

def trained(lm_class, batches):
    lm = lm_class(TransformerEncoder(len(data.vocab) + 1,
                                    rngs=nnx.Rngs(d2l.get_key())))
    d2l.train_steps(lm, batches, nnx.Optimizer(
        lm, optax.adamw(1e-3, weight_decay=0.0), wrt=nnx.Param), 2000)
    return lm

arms = {  # each run is the loss at each position of the window
    'both sides, masked': [pos_loss, loss_by_position(
        trained(MaskedLM, masked), masked.val_dataloader())[0]],
    'left side, masked': [loss_by_position(
        trained(LeftOnlyLM, masked), masked.val_dataloader())[0]
        for _ in range(2)],
    'left side, next byte': [loss_by_position(
        trained(LeftOnlyLM, data), data.val_dataloader())[0]]}
for name, runs in arms.items():
    print(f'{name:21s} interior ' + show(p[interior].mean() for p in runs)
          + '; position 63 ' + show(p[-1] for p in runs))
# Next byte: the loss at position t is that of predicting byte t + 1
d2l.plot([jnp.arange(64)] * 2 + [jnp.arange(1, 65)],
         [runs[0] for runs in arms.values()], 'position in the window',
         'loss', legend=list(arms))
```

Apart from its first few positions, the one-sided masked model sits at
about 2.1 nats, in both of its runs. At position 0 it sees nothing but its
own `<mask>` and starts near the unigram level; a few bytes of left
context bring it down. Against it, the second side is worth 0.4 to 0.6
nats per byte, about a fifth to a quarter of the one-sided loss. The two
runs of the one-sided model agree to within 0.01 nats; the two-sided model
varies more, by up to 0.2 nats between runs, and even the smallest gap, 0.4
nats, is twice that spread. So the gain is large and reliable, though
smaller than the edges suggested. At position 63 the one-sided model,
which practises that situation everywhere, does better than the two-sided
one, about 2.1 nats against 2.4 to 2.5. The edges of the two-sided model
measure a case it has rarely seen, not the value of one side.

The third curve holds a surprise. The language model reads the same side of
the same bytes as the one-sided masked model, yet after the same 2,000
steps it reaches about 1.45 nats, no worse than the two-sided model. It is
scored at all 64 positions of a window rather than about 10, and it never
loses context to masks. At equal steps, next-byte prediction extracts far
more from the same bytes, a fact we return to when choosing a wiring.

These losses are averages over many thousands of bytes. One sentence from
the validation text shows what they average: we mask one byte at a time
and list the model's three most probable fills.

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

In "known as _he European" the model puts more than 0.8 of its probability
on "t". The left context "known as " allows many words, while the
right-hand context "he European" leaves essentially one, and a
left-to-right model never sees it. In "kno_n" both sides point to the same
letter, and the "w" gets about 0.9. In "gam_arus" the model hedges: no
letter gets more than 0.4, and its first choice changes from run to run.
Yet the window holds a strong hint. "Homarus", just before the gap, ends in
the same "marus", and a model that predicts a byte mainly from its nearest
neighbours does not go back for it. It knows English spelling far better
than lobster taxonomy.

Scaled up to subword tokens, sentence pairs, gigabytes of text, and a
fine-tuning recipe per downstream task, this experiment is BERT
:cite:`Devlin.Chang.Lee.ea.2018`. The Language Models part fine-tunes a
pretrained encoder of this kind (:numref:`sec_lm_bert`) and treats
pretraining at scale (:numref:`chap_lm_pretraining`). Encoders
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
string*. For strings of length $T$, target position $t$ must copy source
position $T-1-t$. The letters are drawn independently at random, so the
letters the decoder has already written tell it nothing about the next
one: everything it writes must come through cross-attention. If the
mechanism works and the model solves the task directly, its attention map
should be the anti-diagonal, a prediction we can check. Random strings
also mean unlimited fresh data: like the GPT of :numref:`sec_gpt`, which
almost never reads the same passage twice, this model never sees the same
example twice. `sample_batch` also builds the decoder input for the
teacher forcing of :numref:`subsec_teacher-forcing`: the target shifted
right behind a `<bos>` token, so that every position learns to predict its
successor in parallel.

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

The full model wires a `TransformerEncoder` (the class above, reused
unchanged) to a stack of decoder blocks with its own embeddings and output
head. `EncoderDecoder` returns the logits; each attention module keeps the
weights of its last call in `attention_weights`. The method `decode` runs
the decoder alone, so that generation can run the encoder once and then
only the decoder for each new token.

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

### Training and Decoding

We train one encoder block and one decoder block for 300 steps of 128
freshly sampled strings. The model reads about half a million source
letters and never sees a string twice. The optimizer is `d2l.LMOptimizer`
of :numref:`sec_gpt`, with its gradient clipping but without warmup or
weight decay. Its cosine schedule takes the learning rate to zero; in our
trials at a constant rate, the loss now and then jumped back up after it
had come close to zero. For reference, guessing each letter uniformly
costs $\log 16 \approx 2.77$ nats.

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

By step 25 the loss has fallen to about 2 nats, by step 50 to about 0.5,
and by step 100 below 0.03. The model learns reversal within its first
hundred steps, and the rest of the run refines it.

Decoding runs the encoder once and the decoder autoregressively: feed
`<bos>`, take the argmax, append, repeat.

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

:begin_tab:`jax`
As in `generate` of :numref:`sec_gpt`, the decode loop writes into a
buffer of fixed size, overwritten left to right. Every shape stays static,
so the jitted step compiles once. The causal mask hides the
not-yet-written positions from every query at or before $t$.
:end_tab:

All 1,000 fresh strings come back exactly reversed. Each step of this
loop reruns the decoder over the whole prefix, which at twelve letters
costs nothing worth saving. A decoder that writes long outputs caches the
keys and values of its self-attention, as in :numref:`sec_kv-cache`. Its
cross-attention needs no growing cache at all: the keys and values come
from the encoder output alone, so they are computed once per source and
then read at every step.

### Reading the Alignment
:label:`subsec_reading-alignment`

A perfect score shows that the model reverses strings, not how. The task
dictates which source position each target position needs, so we can
check whether cross-attention looks there. We run a batch through the
model and read the cross-attention weights of the decoder block, of shape
(batch, heads, target positions, source positions). For each target
position we ask whether its largest weight, averaged over heads, falls on
the true source position; we call the fraction of rows where it does the
*hit rate*. If cross-attention alone carries the alignment, the hit rate
is 100% and the map is the anti-diagonal predicted in
:numref:`subsec_reversal-task`.

One run could be typical or not. We therefore train five more models like the
one above, for six in all, and six whose encoder has no transformer blocks at
all: only its embeddings and positions remain. In the second kind, position
$j$ of the encoder output can only carry letter $j$, so the decoder has no
choice: we expect the anti-diagonal in every run. For each run we print the
exact match and the hit rate, and we plot one string's map from the least
aligned run of each kind.

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

Every run of both kinds reverses all 1,000 strings. Without encoder
self-attention every run also has a hit rate of 100%: the map is the
anti-diagonal, as predicted. With encoder self-attention, two or three of
the six runs show the same perfect map. The others reverse just as well,
yet in them the largest weight falls on the true source position in
anywhere from about 30% to 90% of the rows.

The least aligned map shows what happens. Most rows still put their weight
on one source column, but often not on the column whose letter they
output, and some columns are read by more than one target position. The
encoder's self-attention can move a letter to another position, or pack
several letters into one position's vector, before the decoder reads it.
Any such rearrangement, undone by the cross-attention, reverses the string
just as well, and training settles on one of them by chance. The alignment
is then split between two attention layers, and the cross-attention map
shows only the second half. The hit rate adds a smaller blur of its own:
it averages over heads before taking the largest weight, so heads that
read different columns can lower it without any rearrangement.

As :numref:`sec_what-attention-computes` showed, an attention map records
where a head looks and says nothing of what it finds there. Here we wrote
down the alignment ourselves, the model has one block on each side, and
the map we can read is still not always the alignment the model uses.
Exercise :exref:`ex_encdec-depth-heads` asks what more blocks and more
heads do to the maps.

### Assembling the Masks
:label:`subsec_encdec-masks`

Our reversal task avoids padding by construction: every string has the
same length, and the only mask in use is the decoder's causal one. A real
batch holds sequences of different lengths, padded on the right to a
rectangle, and every attention site needs a mask. Each is composed from
the two primitives of :numref:`sec_attention-scoring-functions`, the
padding mask `d2l.padding_mask` built from valid lengths and the causal
triangle `d2l.causal_mask`, combined by logical AND under broadcasting.
Encoder self-attention and cross-attention both mask the source padding,
since the padded source keys carry nothing. Decoder self-attention takes
the causal triangle. With right padding the triangle already hides the
padded target keys from every real query, because every padded position
comes after every real one. Adding the target padding mask changes only the
rows of padded queries, whose outputs the loss ignores. Frameworks add it
anyway, and under left padding it becomes necessary (Exercise
:exref:`ex_encdec-padding`).

We assemble all three masks for a toy ragged batch, as boolean arrays of
shape (batch, queries, keys). Sequence 0 has a source of length
3 padded to 5 and a target of length 3 padded to 4. For it we expect two
blank source columns in the encoder square and in the cross rectangle, and
a causal triangle that differs from the plain one only in its last row.

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

The grids are as predicted. The only row of the decoder mask that the
padding changes belongs to the padded query at position 3, whose output no
loss reads. `d2l.MultiHeadAttention` takes such masks directly and shares
them across heads. The fused kernels of :numref:`sec_attention-at-scale`
broadcast a mask against (batch, heads, queries, keys), so they take it
once a head axis is inserted (`mask[:, None]`).

## Cross-Attention as an Interface

### Learned Queries

An image reaches a model as hundreds or thousands of patch features, and a
minute of audio as thousands of frames. Full self-attention over $N$ such
inputs costs $N^2$ (:numref:`sec_attention-at-scale`), and a language
model that is to read the image as part of its prompt would rather
receive a few dozen vectors than thousands. Cross-attention offers a way
out. In the decoder, its queries came from the target stream, but nothing
in the mechanism requires that source. The queries can instead be $M$
learned vectors: parameters of the model, the same for every input.
Cross-attending with them into an input of length $N$ computes $MN$ scores
rather than $N^2$ and returns $M$ vectors, whatever $N$ is.

The Perceiver :cite:`Jaegle.Gimeno.Brock.ea.2021` builds an encoder on
this idea. Its $M$ learned *latents* read the input through
cross-attention and otherwise self-attend only among themselves, at
$O(M^2)$ per layer. The original interleaves several such reads with the
latent layers; Perceiver IO, like our version below, reads once. Each read
costs $O(MN)$, and the $N^2$ term of full attention never appears.
:numref:`fig_latent-bottleneck` sketches this bottleneck.

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

The input can be any sequence of feature vectors: text, audio frames,
image patches, or a concatenation of all three. Perceiver IO
:cite:`Jaegle.Borgeaud.Alayrac.ea.2022` adds a second array of queries,
built from learned embeddings or from the positions of the desired
outputs, that cross-attends out of the latents. The shape of the output
then becomes a free choice as well: one query for a class label, or one
per pixel or per audio sample.

Learned queries are also a common way to connect an encoder of one kind to
a model of another. A *resampler* turns a variable number of image features
into a fixed number of tokens for a language model. Flamingo
:cite:`alayrac2022flamingo` compresses image and video features into 64
visual tokens, and the Q-Former of BLIP-2 :cite:`Li.Li.Savarese.ea.2023`
places 32 learned queries between a frozen image encoder and a frozen
language model. Detection uses learned queries directly: in DETR
:cite:`Carion.Massa.Synnaeve.ea.2020` each of a hundred learned queries
cross-attends into the image features and returns one object or none. Not
every vision--language model wants the bottleneck, though. Many current
ones skip the resampler and feed the projected patch tokens straight into
the decoder, which lengthens the prompt but keeps every patch.

### The Cost of a Latent Bottleneck
:label:`subsec_perceiver-cost-curve`

How much does the bottleneck save? We time the Perceiver against the
direct alternative, the same two transformer blocks applied to the full
input, as $N$ doubles from 1,024 to 8,192 at $M = 64$. The count predicts
the trend. At $N = 8{,}192$ one layer of full self-attention scores
$N^2 \approx 6.7 \times 10^7$ pairs per head, and the Perceiver's read
scores $MN \approx 5.2 \times 10^5$, 128 times fewer. Each encoder is
compiled before it is timed, so that its operations are not dispatched
from Python one by one. The timer is `d2l.Benchmark` of
:numref:`subsec_async_dispatch`.

```{.python .input #encoders-decoders-the-cost-curve}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32
perceiver = PerceiverEncoder(num_latents=64, num_hiddens=256).to(device)
# The comparison: the same two blocks over the full input, O(N^2)
full = nn.Sequential(*[d2l.TransformerBlock(256, num_heads=4)
                       for _ in range(2)]).to(device)
# Compiled and replayed as one CUDA graph per input length
perceiver, full = (torch.compile(m.eval(), mode='reduce-overhead',
                                 dynamic=False) for m in (perceiver, full))
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
```

```{.python .input #encoders-decoders-the-cost-curve}
%%tab jax
rngs = nnx.Rngs(d2l.get_key())
perceiver = PerceiverEncoder(num_latents=64, num_hiddens=256, rngs=rngs)
perceiver.eval()
# The comparison: the same two blocks over the full input, O(N^2)
full = nnx.Sequential(*[d2l.TransformerBlock(256, num_heads=4, rngs=rngs)
                        for _ in range(2)])
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
```

From $N = 2{,}048$ on, each doubling multiplies the self-attention
encoder's time by about four, the signature of the $N^2$ term taking over.
The first doubling costs less, about 2.5 times, because at 1,024 tokens the
projections and feed-forward networks, whose work grows only linearly in
$N$, still account for much of the time. The Perceiver takes between 0.15
and 0.35 ms at every length, and less than twice as long at $N = 8{,}192$
as at 1,024, although $N$ has grown eightfold. At these sizes its
arithmetic is small, and most of its time is the fixed cost per call of
:numref:`subsec_kv-cache-speedup`, which does not depend on $N$: even a
compiled call costs the host time to check its arguments and launch its
work. At $N = 1{,}024$ the two encoders are within a factor of two of each
other. At $N = 8{,}192$ the Perceiver is more than thirty times faster. The
count of scores promised 128; it leaves out the fixed cost.

A latent bottleneck is the better choice when the input is long and a
summary of fixed size is enough. Whether 64 vectors are enough is the real
question, and no timing answers it. It depends on the task, and Exercise
:exref:`ex_encdec-perceiver-reversal` asks how few latents still reverse a
string.

## Choosing a Wiring
:label:`subsec_encdec-choosing`

We return to the two choices with which this section began: the mask, and
where the keys and values come from. Each wiring is one setting of the two,
and a reader who must choose today will find that one setting does most of
the work.

A decoder-only model can do what the other two do. To translate, write the
source, a separator, and let the model continue: every target token reads
the source through the same causal self-attention that it uses for its own
past. Translation, summarization, and question answering are mostly done
this way today, and the reasons differ with the rival.

Against an encoder, the decoder-only model wins on training signal. Every
position of its training text is a prediction. Our next-byte model was
scored at all 64 positions of a window, against about 10 for the masked
model, and it never read through `<mask>` tokens; after the same 2,000
steps it was no worse than the model that saw both sides. Against an
encoder--decoder that argument fails, because a decoder there also predicts
every target position. In the controlled comparison of
:citet:`raffel2020exploring`, the encoder--decoder did at least as well as
a decoder-only model of the same compute. The decoder-only stack won on
simplicity and reach instead: one stack is pretrained once on raw text, and
the same stack, prompted, serves every task. It gives up little at
inference, since the cache of :numref:`sec_kv-cache` processes the prompt
once, much as an encoder processes its source once.

A variant lets the source positions attend to one another in both
directions as well. This *prefix language model* is a single stack whose
mask is the third pattern of :numref:`fig_three-wirings` drawn as one grid.
Most large models keep the plain causal mask. It scores every position of
the training text, where a prefix language model scores none of its
prefix, and since no position ever sees a later one, a conversation that
alternates between prompt and reply simply extends one cache.

The other two wirings remain the better choice where their structure
matters. An encoder--decoder keeps the source in a network of its own,
which reads it in both directions in every layer and may differ from the
decoder in size and in kind. Whisper's encoder reads audio frames, and its
decoder writes text. The encoder can also be frozen and shared: in Imagen,
a frozen T5 encoder reads the prompt, and the image model cross-attends
into its output :cite:`saharia2022photorealistic`. The wiring suits a
source that is not text in the decoder's vocabulary, or one that is long
and worth reading in both directions.

An encoder is the choice when the output is a vector. It embeds queries
and documents for search, scores query--document pairs for reranking, and
labels millions of documents, and there a small encoder replaces a call to
a large decoder. Even here the decoder has moved in. Several leading
embedding models are decoder-only language models whose causal mask is
removed for fine-tuning
:cite:`BehnamGhader.Adlakha.Mosbach.ea.2024,Lee.Roy.Xu.ea.2025`.

| wiring | output | cost at inference | default today for |
|:--|:--|:--|:--|
| encoder | a vector per token, or one pooled vector | one pass over the input | embeddings for search; reranking; classification |
| encoder--decoder | a sequence, from a separate input | source encoded once; cross-attention keys and values reused at every step | speech recognition; inputs of another kind |
| decoder-only | a continuation of the prompt | prompt processed once and cached; one pass per generated token | generation, chat, and most text-to-text tasks |

Nothing in this table assumes that the tokens are words. The next section
hands an encoder nothing but the patches of an image, in a row, and asks
whether it can learn to read them.

## Summary

Removing the causal mask turns a stack of transformer blocks into an
encoder, trained by reconstructing masked tokens; on Wikipedia bytes the
second side of context lowers the loss of a masked byte by 0.4 to 0.6 nats
against the same encoder trained with one side. An encoder--decoder reads
a separate input through cross-attention, with queries from the target and
keys and values from the encoder. On string reversal its cross-attention
map is the anti-diagonal whenever the encoder cannot rearrange the letters
first. Learned queries let cross-attention read an input of length $N$
into $M$ vectors at cost $O(MN)$, which is how resamplers hand images to
language models. Today a decoder-only model with the source in its prompt
does most text-to-text work: next-token training scores every position,
and one pretrained stack serves every task. Encoders remain the choice for
embeddings and classification, and encoder--decoders for inputs of
another kind, such as speech.

## Exercises

1. **Prefix language model.** The prefix language model of
   :numref:`subsec_encdec-choosing` wires a single stack so that the first
   $m$ positions, which hold the source, attend to one another
   bidirectionally, and each of the $n$ target positions that follow
   attends to the whole source and causally to the target positions up to
   itself. :citet:`raffel2020exploring` compare this wiring with the
   encoder--decoder.
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
   :exlabel:`ex_encdec-masking-recipe`

1. [code] **Training signal per window.** In the causal control of
   :numref:`subsec_mlm-loss-by-position`, a next-byte model with one side
   of context reached a loss far below that of the one-sided masked model,
   and no higher than that of the two-sided one, in the same 2,000 steps.
    1. A masked model scores a fraction $p$ of the $n$ positions of a
       window, a next-token model all $n$ of them. How many scored
       predictions does each receive in 2,000 steps of 64 windows of 64
       bytes at $p = 0.15$? For how many steps must the two-sided masked
       model train to receive as many as the next-byte model?
    1. Train the two-sided masked model for that many steps and compare its
       interior loss with that of the next-byte model after 2,000 steps.
       Does the second side now lower the loss below that of the next-byte
       model, and by how much?
    1. The masked models also read about 15% of their context as `<mask>`.
       Evaluate the two-sided model of the section with a single masked
       byte per window and compare its interior loss with the evaluation
       under the training recipe. How much of its loss comes from the masks
       in its context, and how much is left for the smaller number of
       predictions it was trained on to explain?
    1. The masked model computes a representation at every position,
       scored or not, so equal numbers of scored predictions do not mean
       equal compute. At equal compute, which objective would you choose to
       pretrain an encoder whose only job is to embed documents? Argue from
       your measurements, and say what they leave out.

1. **Padding masks of a ragged batch.** The cell of
   :numref:`subsec_encdec-masks` builds the three masks for a toy batch with
   source lengths $(3, 5)$ and target lengths $(3, 4)$, padded on the right.
    1. With right padding, the causal triangle alone hides every padded
       target key from every real query in one block. Show that replacing
       `dec_self` by the causal triangle leaves the training loss unchanged
       for a decoder with any number of blocks, although the outputs at
       padded positions become the inputs of the next block. Then decide
       whether the encoder output at a padded source position can reach the
       loss.
    1. Now pad the batch on the left, so that every sequence ends at the
       last position. Which of the three masks change, and does the causal
       triangle still suffice for decoder self-attention? `d2l.padding_mask`
       marks the first `valid_lens` keys of each sequence as valid. Which of
       the three masks can it still build?
    1. What goes wrong with the learned position tables of `EncoderDecoder`
       under left padding unless the position indices change as well?
   :exlabel:`ex_encdec-padding`

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
       product. Over several runs of the one-block model, does the product
       recover the anti-diagonal where the cross-attention alone does not?
       When does the residual stream make this tracing fail? Reconcile your
       findings with the cautions of :numref:`sec_what-attention-computes`
       about reading attention maps.
   :exlabel:`ex_encdec-depth-heads`

1. [code] **Several masks at once.** ● The objective
   :eqref:`eq_mlm` scores each masked position separately, so the encoder
   returns one distribution per masked position, conditioned only on the
   bytes that remain visible. Filling all masked positions from one
   forward pass therefore chooses each byte independently of the
   others. Iterative schemes such as Mask-Predict
   :cite:`Ghazvininejad.Levy.Liu.ea.2019` keep only their most confident
   predictions and predict the remaining positions again. Repeated from a
   fully masked window, such steps generate text, which is the idea that
   text diffusion models develop.
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
   :exlabel:`ex_encdec-times-arrow`

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
   :exlabel:`ex_encdec-perceiver-reversal`

[Discussions](https://d2l.discourse.group/t/36777)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.5]{.kicker}

Encoders, Decoders, and Cross-Attention<br>
**three wirings of one block · masked-token prediction · an alignment we can check · learned queries · which wiring today**
:::
:::

::: {.slide title="A Causal Model Sees Only One Side of Every Gap"}
[The problem]{.kicker}

- Fill the gap in `known as _he European`: the left context allows many
  words, the right context (`he European`) fixes the letter.
- Measured on Wikipedia bytes, same encoder and training: a masked byte
  costs about **2.1 nats** with one side of context, **1.5 to 1.7** with
  both.
- Translation and speech recognition read a whole input before they write
  a single output token.

::: {.d2l-note .warn}
How do we wire the same transformer block to read both sides, to read a
separate input, and to read inputs too long for $N^2$ attention? And which
wiring should we use today?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- draw the attention pattern of the encoder, the decoder, and the
  encoder--decoder;
- train an encoder with masked-token prediction and measure what the
  second side of context is worth, against a fair control;
- build a decoder block with cross-attention and compose its masks;
- check a learned cross-attention map against a known alignment;
- estimate the cost of reading $N$ inputs with $M$ learned queries;
- choose a wiring for a task.

**Running examples:** Wikipedia bytes for the encoder, random strings to
reverse for the encoder--decoder; the decoder-only wiring is the GPT we
already have.
:::

::: {.slide title="The Wirings Complete the Block"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN, residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| The KV Cache | reuse past keys and values |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| **Encoders, Decoders, and Cross-Attention** | **the other two wirings** |
| Vision Transformer | images as token sequences |
| Scaling Laws | loss against parameters, data, compute |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | released models as settings |
:::

::: {.col .narrow}
- **Uses:** `TransformerBlock`, cross-attention, padding and causal masks,
  teacher forcing, the KV cache.
- **Skips:** pretraining at scale and fine-tuning (Language Models part),
  beam search.
:::
:::
:::

::: {.slide title="One Block, Three Wirings"}
@fig:mdl-transformers-three-wirings

Which positions may attend to which, and where keys and values come from:
these two choices separate BERT, GPT, and the 2017 translation model.
Drawn as one grid, the third pattern is also a single stack with a
bidirectional prompt.
:::

::: {.slide title="An Encoder Is a Stack of Blocks with No Mask" only="pytorch"}
```{.python #encoders-decoders-exercises-1}
def forward(self, X, mask=None):          # X: (batch, n) token ids
    H = self.token_emb(X) + self.pos_emb(
        torch.arange(X.shape[1], device=X.device))
    for blk in self.blks:                 # d2l.TransformerBlock
        H = blk(H, mask)                  # default: all see all
    return self.norm(H)                   # (batch, n, d)
```

Every output vector summarizes the whole input as seen from its position.
Positions still need a table: attention ignores order. Pooling (average, or
one designated position) gives one vector per input.
:::

::: {.slide title="An Encoder Is a Stack of Blocks with No Mask" only="jax"}
```{.python #encoders-decoders-exercises-2}
def __call__(self, X, mask=None):         # X: (batch, n) token ids
    H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
    for blk in self.blks:                 # d2l.TransformerBlock
        H = blk(H, mask)                  # default: all see all
    return self.norm(H)                   # (batch, n, d)
```

Every output vector summarizes the whole input as seen from its position.
Positions still need a table: attention ignores order. Pooling (average, or
one designated position) gives one vector per input.
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

::: {.slide title="Masked Prediction Hides What It Asks For"}
Replace a random set $\mathcal{M}$ of tokens (15%) by `<mask>`, keep
their positions, and reconstruct them:

$$\max_\theta \; \sum_{t \in \mathcal{M}} \log p_\theta\left(x_t \mid \mathbf{x}_{\setminus \mathcal{M}}\right)$$

@!encoders-decoders-the-masked-token-objective

Bytes of WikiText-103, windows of 64; byte frequencies alone give 3.17
nats, and the loss is still there at step 500.
:::

::: {.slide title="The Loss Is Flat Inside the Window and Jumps at Its Edges"}
Position 0 sees only its right side, position 63 only its left; 15% of
all bytes are masked.

@!encoders-decoders-what-the-second-side-is-worth-1

::: {.d2l-note .rule}
The edges cost 2.4 to 2.5 nats, the interior 1.6 to 1.7; two or three
bytes of the missing side bring the loss back to the interior level.
:::
:::

::: {.slide title="Against a Causal Twin, the Second Side Is Worth 0.4 to 0.6 Nats"}
Same encoder, same 2,000 steps; the third curve also changes the objective.

@!encoders-decoders-a-causal-control

::: {.d2l-note .rule}
One-sided training brings position 63 to about 2.1. Next-byte training
scores all 64 positions, not about 10, and reaches about 1.45.
:::
:::

::: {.slide title="The Right-Hand Context Fixes the Missing Letter"}
@!encoders-decoders-what-the-second-side-is-worth-2

- `he European` fixes the `t` in "as _he European"; a left-to-right
  model never sees it.
- `gam_arus`: the model hedges, although `Homarus` just before the gap
  ends in the same `marus`; it reads mainly its nearest neighbours.
- Scaled to subwords and gigabytes, this experiment is BERT (2018);
  ModernBERT (2024) keeps the wiring with RoPE, a gated FFN, and 8k
  context.
:::

::: {.slide title="Reversal Fixes the Correct Alignment"}
For strings of length $T = 12$, target position $t$ must copy source
position $T-1-t$. Letters are random, so content reaches the decoder only
through cross-attention.

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

- Self-attention is causal and cached while decoding, as in a GPT.
- Cross-attention is unmasked; its keys and values depend only on
  $\mathbf{E}$, so they are computed once per source.
:::

::: {.slide title="Cross-Attention Is One More Sublayer" only="pytorch"}
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

::: {.slide title="Cross-Attention Is One More Sublayer" only="jax"}
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

::: {.slide title="300 Steps Reverse Every Fresh String"}
One encoder block, one decoder block, batches of 128 fresh strings,
cosine decay to zero; uniform guessing costs $\log 16 \approx 2.77$ nats:

@!encoders-decoders-training-and-decoding-1

Greedy decoding: the encoder runs once, the decoder once per token.

@!encoders-decoders-training-and-decoding-2
:::

::: {.slide title="The Anti-Diagonal Appears When the Encoder Cannot Move Letters"}
Six runs each, all reverse. With self-attention the encoder can move or
pack letters first, and the map shows only the second half of the alignment.

@!encoders-decoders-reading-the-alignment
:::

::: {.slide title="Three Masks from Two Primitives"}
A padded batch, source lengths (3, 5), target lengths (3, 4), padded on
the right:

- `padding_mask(lens, n)`: key $j$ valid if $j <$ length;
- `causal_mask(T)`: key $j$ visible to query $i$ if $j \le i$;
- encoder self and cross: source padding · decoder self: causal, which
  already hides right-padded keys from every real query.

::: {.d2l-note .rule}
Masks combine by logical AND and broadcast over heads. Target padding
changes only padded query rows, until the batch is padded on the left.
:::
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

::: {.slide title="Learned Queries Read N Inputs at Cost O(MN)"}
::: {.cols .vc}
::: {.col}
@fig:mdl-transformers-latent-bottleneck
:::

::: {.col .narrow}
- An input of $N$ vectors (patches, frames, bytes): $M$ learned latents
  cross-attend into it at $O(MN)$, then attend only among themselves at
  $O(M^2)$. The $N^2$ term never appears (Perceiver, 2021).
- Resamplers hand images to language models this way (Flamingo: 64
  tokens; BLIP-2: 32 queries); DETR uses one query per object.
:::
:::
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

Whatever $N$, the output is $M$ vectors. Output queries can set the
output shape too (Perceiver IO).
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

Whatever $N$, the output is $M$ vectors. Output queries can set the
output shape too (Perceiver IO).
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

::: {.slide title="Full Attention Nears 4× per Doubling; the Perceiver Barely Moves"}
@!encoders-decoders-the-cost-curve

Both compiled. The Perceiver's 0.15 to 0.35 ms is mostly a fixed cost per
call, so at $N = 8192$ it is over $30\times$ faster, not the $128\times$ of
the score count.
:::

::: {.slide title="Decoder-Only Is the Default for Text Today"}
| wiring | outputs | at inference | default for |
|---|---|---|---|
| encoder | a vector per token, or pooled | one pass | embeddings for search, reranking, classification |
| encoder--decoder | a sequence from a separate input | source encoded once; cross keys and values reused | speech recognition (Whisper) |
| decoder | a continuation of the prompt | prompt cached once, then a token per step | translation, summarization, chat: source in the prompt |

::: {.d2l-note .rule}
Against an encoder: next-byte training scores all 64 positions of a
window, masked training about 10; after 2,000 steps our next-byte model
was no worse than the two-sided masked one.
Against an encoder--decoder: one pretrained stack serves every task
through its prompt.
:::
:::

::: {.slide title="Check: Label Ten Million Support Tickets"}
[Check]{.kicker}

Each ticket must get one of 20 topic labels, at the lowest cost. Which
wiring, and how does it turn a ticket into one label?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** An encoder: one bidirectional pass per ticket, outputs pooled
to one vector (average or a designated token), then a linear layer onto
20 classes. No token is generated, so no decoder is needed.
:::
:::
:::

::: {.slide title="Mask and Key Source Make Three Wirings; Decoder-Only Leads for Text"}
[Wrap-up]{.kicker}

- One block, three wirings: no mask, causal mask, or both joined by
  cross-attention.
- Masked prediction trains the encoder; with the same training a masked
  byte costs 1.5 to 1.7 nats with both sides of context, about 2.1 with
  one.
- On reversal every run reverses every fresh string; the cross-attention
  map is the anti-diagonal whenever the encoder cannot move letters first.
- $M$ learned queries read $N$ inputs at $O(MN)$; at $N = 8192$ the
  Perceiver is over $30\times$ faster than full self-attention.
- Decoder-only is the default for text; encoders serve embeddings,
  encoder--decoders inputs of another kind, such as speech.

**Next:** the Vision Transformer, the encoder-only wiring over image
patches.
:::
