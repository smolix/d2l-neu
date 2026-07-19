# A GPT from Scratch
:label:`sec_gpt`

One block, stacked, is most of a language model. This section adds the
little that remains — an embedding, positions, a causal mask, an output
head — and packages the result as a single `GPT` class whose constructor
arguments span the design space: the block flags of
:numref:`sec_transformer-block` plus a positional scheme. That
one-class-many-configurations shape is the spine of this chapter. The same
class, with the modern flags, trains from scratch on *The Time Machine* in
about a minute; with the 2019 flags, it accepts the released weights of
GPT-2 :cite:`Radford.Wu.Child.ea.2019` and completes English sentences.
Between those two demonstrations sit the practical crafts this section
teaches: reading a loss curve honestly, breaking a model with a
normalization flag, and sampling from a trained distribution.

```{.python .input #gpt-a-gpt-from-scratch}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from safetensors.torch import load_file
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #gpt-a-gpt-from-scratch}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import optax
from safetensors import numpy as safetensors_numpy
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
```

## From Blocks to a Language Model

Four ingredients turn a stack of transformer blocks into a GPT-style
language model.

**A token embedding and a tied head.** A lookup table maps token ids into
the residual stream; after the final block (and one last normalization,
pre-norm convention), the *same* table, transposed, maps the stream back to
logits over the vocabulary. Tying the two matrices saves a
$\textrm{vocab} \times d$ parameter block and consistently helps at small
scale :cite:`Press.Wolf.2017`; GPT-2 ties them too, which will spare us a
tensor when we load its weights.

**A causal mask.** A language model predicts token $t+1$ from tokens $1,
\ldots, t$, so position $t$ must not attend forward. In
:numref:`sec_transformer-block` we got causality by handing
`d2l.MultiHeadAttention` a per-query valid length. Here we build the causal
variant natively, for two reasons. First, efficiency: the fused attention
kernels of :numref:`sec_attention-at-scale`
(`scaled_dot_product_attention` in PyTorch, `dot_product_attention` in
JAX) take the causal mask as a flag and never materialize the $n \times n$
score matrix. Second, positions.

**Positions, inside attention or outside.** The `pos` argument selects
between the two schemes that :numref:`sec_positional-information` found in
deployed models: `'learned'` adds a trained position vector to the
embedding — GPT-2's scheme, capped at the table's length — and `'rope'`
rotates queries and keys inside every attention head, which is what
essentially every current model does. RoPE lives where queries and keys
are made, so the attention module implements it itself (the same
`_rope` we gave `TinyCharLM`); nothing else in the model knows positions
exist.

**The blocks themselves** are `d2l.TransformerBlock`, unchanged: the
causal attention drops in through the `attn_factory` hook, exactly the
seam it was designed to be.

```{.python .input #gpt-from-blocks-to-a-language-model-1}
%%tab pytorch
class GPT(nn.Module):  #@save
    """Decoder-only transformer language model built from configurable
    blocks."""

    class CausalAttention(nn.Module):
        """Multi-head causal self-attention, optionally rotary."""
        def __init__(self, num_hiddens, num_heads, bias=False, rope=False):
            super().__init__()
            self.num_heads, self.rope = num_heads, rope
            self.W_qkv = nn.Linear(num_hiddens, 3 * num_hiddens, bias=bias)
            self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

        def _rope(self, x):
            d = x.shape[-1]
            pos = torch.arange(x.shape[-2], dtype=torch.float32,
                               device=x.device)
            inv_freq = 10000.0 ** (
                -torch.arange(0, d, 2, device=x.device) / d)
            theta = pos[:, None] * inv_freq[None, :]
            cos, sin = torch.cos(theta), torch.sin(theta)
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin,
                                x1 * sin + x2 * cos], -1).flatten(-2)

        def forward(self, X, *_):
            B, T, D = X.shape
            q, k, v = self.W_qkv(X).chunk(3, -1)
            q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
                       for u in (q, k, v))
            if self.rope:
                q, k = self._rope(q), self._rope(k)
            Y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.W_o(Y.transpose(1, 2).reshape(B, T, D))

    def __init__(self, vocab_size, num_hiddens=256, num_heads=8, num_blks=6,
                 max_len=1024, pos='rope', norm='rms', act='swiglu',
                 pre_norm=True, bias=False, dropout=0):
        super().__init__()
        self.pos, self.max_len = pos, max_len
        self.token_emb = nn.Embedding(vocab_size, num_hiddens)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pos == 'learned':
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        attn = lambda: self.CausalAttention(num_hiddens, num_heads, bias,
                                            rope=(pos == 'rope'))
        self.blks = nn.ModuleList([
            d2l.TransformerBlock(num_hiddens, num_heads, dropout, norm, act,
                                 pre_norm, bias, attn_factory=attn)
            for _ in range(num_blks)])
        self.norm = (nn.RMSNorm if norm == 'rms'
                     else nn.LayerNorm)(num_hiddens)

    def forward(self, X):
        H = self.token_emb(X)
        if self.pos == 'learned':
            H = H + self.pos_emb(torch.arange(X.shape[1], device=X.device))
        for blk in self.blks:
            H = blk(H)
        return F.linear(self.norm(H), self.token_emb.weight)
```

```{.python .input #gpt-from-blocks-to-a-language-model-1}
%%tab jax
class GPT(nnx.Module):  #@save
    """Decoder-only transformer language model built from configurable
    blocks."""

    class CausalAttention(nnx.Module):
        """Multi-head causal self-attention, optionally rotary."""
        def __init__(self, num_hiddens, num_heads, bias=False, rope=False,
                     rngs=None):
            rngs = nnx.Rngs(0) if rngs is None else rngs
            self.num_heads, self.rope = num_heads, rope
            self.W_qkv = nnx.Linear(num_hiddens, 3 * num_hiddens,
                                    use_bias=bias, rngs=rngs)
            self.W_o = nnx.Linear(num_hiddens, num_hiddens, use_bias=bias,
                                  rngs=rngs)

        def _rope(self, x):
            # x: (batch, num_steps, num_heads, head_dim)
            d = x.shape[-1]
            pos = jnp.arange(x.shape[1], dtype=jnp.float32)
            inv_freq = 10000.0 ** (-jnp.arange(0, d, 2) / d)
            theta = pos[:, None] * inv_freq[None, :]
            cos = jnp.cos(theta)[:, None, :]  # broadcast over heads
            sin = jnp.sin(theta)[:, None, :]
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return jnp.stack([x1 * cos - x2 * sin,
                              x1 * sin + x2 * cos], -1).reshape(x.shape)

        def __call__(self, X, *_):
            B, T, D = X.shape
            q, k, v = jnp.split(self.W_qkv(X), 3, axis=-1)
            q, k, v = (u.reshape(B, T, self.num_heads, -1)
                       for u in (q, k, v))
            if self.rope:
                q, k = self._rope(q), self._rope(k)
            Y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
            return self.W_o(Y.reshape(B, T, D)), None

    def __init__(self, vocab_size, num_hiddens=256, num_heads=8, num_blks=6,
                 max_len=1024, pos='rope', norm='rms', act='swiglu',
                 pre_norm=True, bias=False, dropout=0, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.pos, self.max_len = pos, max_len
        init = nnx.initializers.normal(0.02)
        self.token_emb = nnx.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        if pos == 'learned':
            self.pos_emb = nnx.Embed(max_len, num_hiddens,
                                     embedding_init=init, rngs=rngs)
        attn = lambda rngs: self.CausalAttention(
            num_hiddens, num_heads, bias, rope=(pos == 'rope'), rngs=rngs)
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, dropout, norm, act,
                                 pre_norm, bias, attn_factory=attn,
                                 rngs=rngs)
            for _ in range(num_blks)])
        self.norm = (nnx.RMSNorm if norm == 'rms'
                     else nnx.LayerNorm)(num_hiddens, rngs=rngs)

    def __call__(self, X):
        H = self.token_emb(X)
        if self.pos == 'learned':
            H = H + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = blk(H)
        return self.token_emb.attend(self.norm(H))
```

A shape check, and the census of the default configuration:

```{.python .input #gpt-from-blocks-to-a-language-model-2}
%%tab pytorch
model = GPT(vocab_size=28)
X = torch.zeros(2, 16, dtype=torch.long)
d2l.check_shape(model(X), (2, 16, 28))
print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters')
```

```{.python .input #gpt-from-blocks-to-a-language-model-2}
%%tab jax
model = GPT(vocab_size=28)
X = jnp.zeros((2, 16), dtype=jnp.int32)
d2l.check_shape(model(X), (2, 16, 28))
n = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
print(f'{n / 1e6:.2f}M parameters')
```

One decision above deserves a defense: the model reads *characters*, not
subwords. Every serious language model tokenizes with byte-pair encoding,
and :numref:`sec_text-sequence` built a full BPE tokenizer — we will even
reuse its GPT-2 pattern verbatim when we load GPT-2 below. But BPE earns
its keep on gigabytes; on a 180 KB novel a subword vocabulary would leave
each token type a handful of training examples. Characters keep the
statistics dense, the vocabulary trivial, and the comparison fair against
the character-level models of previous chapters.

## Training the Modern Configuration

We train the default configuration — pre-norm, RMSNorm, SwiGLU, RoPE:
the same flag settings you would find in a Llama or Qwen checkpoint
:cite:`touvron2023llama` — on the character-level Time Machine corpus,
with one concession to its size: `dropout=0.1`, a regularizer that
frontier models dropped precisely because their corpora outweigh their
parameters, which is the opposite of our situation here. We watch the
validation loss as we go rather than admiring the training curve alone.

```{.python .input #gpt-training-the-modern-configuration}
%%tab pytorch
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char',
                       num_train=100000, num_val=3000)

def val_loss(model, data):
    device = d2l.try_gpu()
    model.to(device).eval()
    with torch.no_grad():
        losses = [F.cross_entropy(
            model(X.to(device)).flatten(0, 1), Y.to(device).flatten())
            for X, Y in data.val_dataloader()]
    model.train()
    return sum(l.item() for l in losses) / len(losses)

torch.manual_seed(0)
model = GPT(len(data.vocab), dropout=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                              weight_decay=0.0)
steps, train_curve, val_curve = [], [], []
for chunk in range(10):
    losses = d2l.train_lm(model, data, optimizer, 200)
    steps.append(200 * (chunk + 1))
    train_curve.append(sum(losses[-50:]) / 50)
    val_curve.append(val_loss(model, data))
d2l.plot(steps, [train_curve, val_curve], 'step', 'loss',
         legend=['train', 'validation'])
print(f'final: train {train_curve[-1]:.2f}, '
      f'validation {val_curve[-1]:.2f}; '
      f'best validation {min(val_curve):.2f} at step '
      f'{steps[val_curve.index(min(val_curve))]}')
```

```{.python .input #gpt-training-the-modern-configuration}
%%tab jax
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char',
                       num_train=100000, num_val=3000)

@nnx.jit
def batch_loss(model, X, Y):
    logits = model(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

def val_loss(model, data):
    model.eval()
    losses = [float(batch_loss(model, jnp.asarray(X), jnp.asarray(Y)))
              for X, Y in data.val_dataloader()]
    model.train()
    return sum(losses) / len(losses)

model = GPT(len(data.vocab), dropout=0.1, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=0.0),
                          wrt=nnx.Param)
steps, train_curve, val_curve = [], [], []
for chunk in range(10):
    losses = d2l.train_lm(model, data, optimizer, 200)
    steps.append(200 * (chunk + 1))
    train_curve.append(sum(losses[-50:]) / 50)
    val_curve.append(val_loss(model, data))
d2l.plot(steps, [train_curve, val_curve], 'step', 'loss',
         legend=['train', 'validation'])
print(f'final: train {train_curve[-1]:.2f}, '
      f'validation {val_curve[-1]:.2f}; '
      f'best validation {min(val_curve):.2f} at step '
      f'{steps[val_curve.index(min(val_curve))]}')
```

Read the two curves separately, because they tell different stories. The
training loss falls smoothly toward a few tenths of a nat — the
architecture works, the optimizer works. The validation loss bottoms out
around $1.5$ nats within the first few hundred steps and then *rises* while
training keeps improving. No bug: our run feeds the model roughly 16
million tokens drawn from a corpus of one hundred thousand characters,
about 160 passes over the book, with 4.7 million parameters to spend —
dozens of parameters per unique character of text. Past the first epochs,
gradient descent has nothing left to learn from this book except the book
itself, verbatim. The cure is not fewer steps or a bigger dropout but
*more data*; how loss actually scales when data and parameters grow
together is the business of this chapter's closing section on scaling laws
:cite:`kaplan2020scaling,hoffmann2022training`.

It is worth locating this run on the cost map. At roughly $6ND$
floating-point operations for training a model of $N$ parameters on $D$
tokens, our minute of GPU time spent about $5 \times 10^{14}$ FLOPs.
GPT-2's training run — same class, sixty times the parameters, a corpus
five orders of magnitude larger — sits near $10^{20}$, and frontier runs
land around $10^{25}$ or beyond. Nothing in the code changes across those
eleven orders of magnitude; that is precisely why the machinery of this
section is worth learning on a novella.

### Breaking It with One Flag

:numref:`sec_transformer-block` predicted, from initialization statistics
alone, that the post-LN arrangement starves its attention layers of
gradient. Now that we own a trainable model, we can watch the prediction
come true. Same model, same data, same 800 steps at learning rate
$3 \times 10^{-3}$ — three times the rate above, still comfortable for
pre-norm — with `pre_norm` flipped:

```{.python .input #gpt-breaking-it-with-one-flag}
%%tab pytorch
for pre_norm in (True, False):
    torch.manual_seed(0)
    ablation = GPT(len(data.vocab), pre_norm=pre_norm)
    losses = d2l.train_lm(ablation, data,
                          torch.optim.AdamW(ablation.parameters(), lr=3e-3,
                                            weight_decay=0.0), 800)
    print(f'pre_norm={pre_norm}: loss at step 100/200/400/800: ' +
          '/'.join(f'{sum(losses[k-50:k]) / 50:.2f}'
                   for k in (100, 200, 400, 800)))
```

```{.python .input #gpt-breaking-it-with-one-flag}
%%tab jax
for pre_norm in (True, False):
    ablation = GPT(len(data.vocab), pre_norm=pre_norm, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(ablation, optax.adamw(3e-3, weight_decay=0.0),
                              wrt=nnx.Param)
    losses = d2l.train_lm(ablation, data, optimizer, 800)
    print(f'pre_norm={pre_norm}: loss at step 100/200/400/800: ' +
          '/'.join(f'{sum(losses[k-50:k]) / 50:.2f}'
                   for k in (100, 200, 400, 800)))
```

The pre-norm model trains as if nothing happened. The post-norm model does
not diverge — no NaNs, no explosion — it does something more telling: it
sticks at about $2.8$ nats and never leaves. That number is exactly the
unigram entropy of this text ($2.83$ nats: predict letter frequencies,
ignore all context), and a language model pinned there is a model whose
attention has learned nothing — the training-time realization of the
starved attention gradients we measured at initialization. At the gentler
learning rate of the previous run, post-LN does train, trailing early
(this is what learning-rate warmup was invented for
:cite:`xiong2020layer`); at the rate pre-norm shrugs off, it fails
outright. GPT-2's quiet move of the normalization, ahead of most of the
field, is one reason its 48-block variant was trainable at all in 2019
:cite:`Radford.Wu.Child.ea.2019`.

## Sampling from the Model

A language model outputs a distribution; text comes from *decoding* it.
:numref:`sec_decoding` covered the search view — greedy, beam,
sampling-with-temperature — for sequence models in general. For open-ended
generation the workhorse is temperature-plus-truncation: divide the logits
by a temperature $\tau$, optionally keep only the $k$ most probable tokens
:cite:`Fan.Lewis.Dauphin.2018` (or the smallest set covering probability
$p$, *nucleus sampling* :cite:`Holtzman.Buys.Du.ea.2020` — an exercise),
and sample. We add `generate` to the class. It is deliberately naive:
every new token reruns the full forward pass over the whole history,
which is quadratic work per token — measuring and then eliminating that
waste is the entire next section.

```{.python .input #gpt-sampling-from-the-model-1}
%%tab pytorch
@d2l.add_to_class(GPT)  #@save
@torch.no_grad()
def generate(self, prefix, num_tokens, temperature=1.0, top_k=None):
    """Sample a continuation of the token-id list prefix."""
    ids = list(prefix)
    device = next(self.parameters()).device
    for _ in range(num_tokens):
        X = torch.tensor(ids[-self.max_len:], device=device)[None]
        logits = self(X)[0, -1] / temperature
        if top_k is not None:
            cutoff = torch.topk(logits, top_k).values[-1]
            logits[logits < cutoff] = -torch.inf
        ids.append(int(torch.multinomial(F.softmax(logits, -1), 1)))
    return ids
```

```{.python .input #gpt-sampling-from-the-model-1}
%%tab jax
@d2l.add_to_class(GPT)  #@save
def generate(self, prefix, num_tokens, seed=0, temperature=1.0,
             top_k=None):
    """Sample a continuation of the token-id list prefix."""
    total = len(prefix) + num_tokens
    buf = jnp.zeros((1, total), dtype=jnp.int32)
    buf = buf.at[0, :len(prefix)].set(jnp.asarray(prefix))

    @nnx.jit
    def logits_at(model, buf, t):
        return model(buf)[0, t - 1]

    ids, key = list(prefix), jax.random.key(seed)
    for t in range(len(prefix), total):
        logits = logits_at(self, buf, t) / temperature
        if top_k is not None:
            logits = jnp.where(logits < jnp.sort(logits)[-top_k],
                               -jnp.inf, logits)
        key, sub = jax.random.split(key)
        ids.append(int(jax.random.categorical(sub, logits)))
        buf = buf.at[0, t].set(ids[-1])
    return ids
```

:begin_tab:`jax`
One JAX-specific choice: rather than growing the input by one token per
step — which would trigger a fresh XLA compilation at every new length —
we allocate a buffer of the final size once and overwrite it left to
right. The causal mask makes the not-yet-written positions invisible to
every query before them, so the logits at position $t-1$ are exact, and
the jitted forward compiles exactly once. Fixed shapes are how real JAX
serving systems work too, as the next section shows.
:end_tab:

Let's hear what 4.7 million parameters trained for a minute sound like:

```{.python .input #gpt-sampling-from-the-model-2}
%%tab pytorch
model.eval()
prefix = data.vocab[list('the time traveller ')]
for temperature, top_k in ((1.0, None), (0.7, 8), (2.0, None)):
    torch.manual_seed(0)
    out = model.generate(prefix, 120, temperature, top_k)
    print(f'T={temperature}, top_k={top_k}: '
          + repr(''.join(data.vocab.to_tokens(out))))
```

```{.python .input #gpt-sampling-from-the-model-2}
%%tab jax
model.eval()
prefix = data.vocab[list('the time traveller ')]
for temperature, top_k in ((1.0, None), (0.7, 8), (2.0, None)):
    out = model.generate(prefix, 120, seed=0, temperature=temperature,
                         top_k=top_k)
    print(f'T={temperature}, top_k={top_k}: '
          + repr(''.join(data.vocab.to_tokens(out))))
```

The samples are fluent pseudo-Wells — and if you search the corpus you
will find long stretches of them are not pseudo at all but verbatim
quotation: the overfitting that the validation curve reported, made
audible. Note how little the temperature changes the early tokens. A
memorizing model is a *confident* model, and dividing near-one-hot logits
by $0.7$ or $2$ barely moves them; the knobs only start to matter once
the model is genuinely uncertain. We need a model whose uncertainty is
real — so let's load one.

## Loading GPT-2

GPT-2 is the type specimen of this section's class: 12 blocks, 768 hidden
units, 12 heads, LayerNorm (with biases), GELU, learned positions, tied
head — and pre-norm, years before that was consensus. Its 124M-parameter
release is a constructor call away; all we owe it is its exact flags, plus
its weights and its tokenizer, both of which we pin by content hash in
`d2l.DATA_HUB` so that one download serves every later run and framework.

```{.python .input #gpt-loading-gpt-2-1}
%%tab pytorch
HF_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/'
d2l.DATA_HUB['gpt2-weights'] = (
    HF_URL + 'model.safetensors',
    '89a76996d7c6ee89b86618a265483aab73e61d50')
d2l.DATA_HUB['gpt2-merges'] = (
    HF_URL + 'merges.txt', '396d4d8ec90cb02f4d56e049e0e4add868bcd943')
d2l.DATA_HUB['gpt2-encoder'] = (
    HF_URL + 'vocab.json', 'f0223209235343bc067d7da838328bced8085ae1')

enc = tiktoken.Encoding(
    'gpt2', explicit_n_vocab=50257,
    pat_str=d2l.BPETokenizer.GPT2_PATTERN,
    mergeable_ranks=data_gym_to_mergeable_bpe_ranks(
        d2l.download('gpt2-merges', '../data/gpt2'),
        d2l.download('gpt2-encoder', '../data/gpt2')),
    special_tokens={'<|endoftext|>': 50256})
ids = enc.encode('Attention is all you need.')
print(ids, [enc.decode([i]) for i in ids])
```

```{.python .input #gpt-loading-gpt-2-1}
%%tab jax
HF_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/'
d2l.DATA_HUB['gpt2-weights'] = (
    HF_URL + 'model.safetensors',
    '89a76996d7c6ee89b86618a265483aab73e61d50')
d2l.DATA_HUB['gpt2-merges'] = (
    HF_URL + 'merges.txt', '396d4d8ec90cb02f4d56e049e0e4add868bcd943')
d2l.DATA_HUB['gpt2-encoder'] = (
    HF_URL + 'vocab.json', 'f0223209235343bc067d7da838328bced8085ae1')

enc = tiktoken.Encoding(
    'gpt2', explicit_n_vocab=50257,
    pat_str=d2l.BPETokenizer.GPT2_PATTERN,
    mergeable_ranks=data_gym_to_mergeable_bpe_ranks(
        d2l.download('gpt2-merges', '../data/gpt2'),
        d2l.download('gpt2-encoder', '../data/gpt2')),
    special_tokens={'<|endoftext|>': 50256})
ids = enc.encode('Attention is all you need.')
print(ids, [enc.decode([i]) for i in ids])
```

The tokenizer cell is the BPE machinery of :numref:`sec_text-sequence`
meeting the real artifact: GPT-2's released merge list and vocabulary,
interpreted with the *same* pre-tokenization pattern our own
`d2l.BPETokenizer` uses, assembled into `tiktoken`'s fast encoder. No
model library, no configuration framework — two data files and a regular
expression.

Now the weights. The checkpoint stores one tensor per parameter under
names like `h.3.attn.c_attn.weight`; our job is a dictionary mapping those
names onto our modules. One historical trap: GPT-2's original code
implemented linear layers as a `Conv1D` class that keeps weights as
$(\textrm{in}, \textrm{out})$ — the transpose of `nn.Linear`'s
$(\textrm{out}, \textrm{in})$ — so every 2-D weight must be transposed on
the way in. The fused `c_attn` matrix concatenates queries, keys, and
values in the same order as our `W_qkv`, and the token embedding doubles
as the output head in both models, so neither needs special handling.

```{.python .input #gpt-loading-gpt-2-2}
%%tab pytorch
gpt2 = GPT(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
           max_len=1024, pos='learned', norm='layer', act='gelu',
           pre_norm=True, bias=True)  # the GPT-2 (124M) configuration
weights = load_file(d2l.download('gpt2-weights', '../data/gpt2'))

with torch.no_grad():
    gpt2.token_emb.weight.copy_(weights['wte.weight'])
    gpt2.pos_emb.weight.copy_(weights['wpe.weight'])
    gpt2.norm.weight.copy_(weights['ln_f.weight'])
    gpt2.norm.bias.copy_(weights['ln_f.bias'])
    for i, blk in enumerate(gpt2.blks):
        modules = {f'h.{i}.ln_1': blk.norm1, f'h.{i}.ln_2': blk.norm2,
                   f'h.{i}.attn.c_attn': blk.attention.W_qkv,
                   f'h.{i}.attn.c_proj': blk.attention.W_o,
                   f'h.{i}.mlp.c_fc': blk.ffn.W_1,
                   f'h.{i}.mlp.c_proj': blk.ffn.W_2}
        for key, module in modules.items():
            W = weights[key + '.weight']
            module.weight.copy_(W.T if W.ndim == 2 else W)  # Conv1D layout
            module.bias.copy_(weights[key + '.bias'])

gpt2.to(d2l.try_gpu()).eval()
print(f'{sum(p.numel() for p in gpt2.parameters()) / 1e6:.1f}M parameters')
```

```{.python .input #gpt-loading-gpt-2-2}
%%tab jax
gpt2 = GPT(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
           max_len=1024, pos='learned', norm='layer', act='gelu',
           pre_norm=True, bias=True)  # the GPT-2 (124M) configuration
weights = safetensors_numpy.load_file(
    d2l.download('gpt2-weights', '../data/gpt2'))

gpt2.token_emb.embedding[...] = jnp.asarray(weights['wte.weight'])
gpt2.pos_emb.embedding[...] = jnp.asarray(weights['wpe.weight'])
gpt2.norm.scale[...] = jnp.asarray(weights['ln_f.weight'])
gpt2.norm.bias[...] = jnp.asarray(weights['ln_f.bias'])
for i, blk in enumerate(gpt2.blks):
    for norm, key in ((blk.norm1, f'h.{i}.ln_1'),
                      (blk.norm2, f'h.{i}.ln_2')):
        norm.scale[...] = jnp.asarray(weights[key + '.weight'])
        norm.bias[...] = jnp.asarray(weights[key + '.bias'])
    for linear, key in ((blk.attention.W_qkv, f'h.{i}.attn.c_attn'),
                        (blk.attention.W_o, f'h.{i}.attn.c_proj'),
                        (blk.ffn.W_1, f'h.{i}.mlp.c_fc'),
                        (blk.ffn.W_2, f'h.{i}.mlp.c_proj')):
        # Conv1D stores (in, out) -- exactly nnx.Linear's kernel layout
        linear.kernel[...] = jnp.asarray(weights[key + '.weight'])
        linear.bias[...] = jnp.asarray(weights[key + '.bias'])

gpt2.eval()
n = sum(p.size for p in jax.tree.leaves(nnx.state(gpt2, nnx.Param)))
print(f'{n / 1e6:.1f}M parameters')
```

:begin_tab:`jax`
Loading takes no torch and no transposes here: the safetensors file opens
directly into NumPy arrays, and the Conv1D $(\textrm{in}, \textrm{out})$
layout that PyTorch users must transpose happens to be exactly the layout
`nnx.Linear` keeps its kernels in.
:end_tab:

Did it work? Weight-loading bugs are notorious for failing silently — a
transposed matrix still multiplies. Two checks, one quantitative and one
you can read. First, the model should assign natural English a
respectable probability:

```{.python .input #gpt-loading-gpt-2-3}
%%tab pytorch
text = ("The Time Machine, by H. G. Wells. The Time Traveller was "
        "expounding a recondite matter to us.")
x = torch.tensor(enc.encode(text), device=d2l.try_gpu())[None]
with torch.no_grad():
    logits = gpt2(x)
loss = F.cross_entropy(logits[0, :-1], x[0, 1:])
print(f'per-token loss {loss.item():.2f}, '
      f'perplexity {loss.exp().item():.0f}')
```

```{.python .input #gpt-loading-gpt-2-3}
%%tab jax
text = ("The Time Machine, by H. G. Wells. The Time Traveller was "
        "expounding a recondite matter to us.")
x = jnp.asarray(enc.encode(text))[None]
loss = optax.softmax_cross_entropy_with_integer_labels(
    gpt2(x)[0, :-1], x[0, 1:]).mean()
print(f'per-token loss {loss:.2f}, perplexity {jnp.exp(loss):.0f}')
```

A per-token perplexity around 50 over a 50,257-way vocabulary — against
50,257 for uniform guessing — says the plumbing is right. Second,
the readable check: greedy decoding of a stock prompt reproduces GPT-2's
well-documented continuation, and sampled continuations respond to the
temperature and top-$k$ knobs the way the char model could not:

```{.python .input #gpt-loading-gpt-2-4}
%%tab pytorch
torch.manual_seed(0)
out = gpt2.generate(enc.encode('Alan Turing theorized that computers '
                               'would one day become'), 16, top_k=1)
print(enc.decode(out))
torch.manual_seed(0)
out = gpt2.generate(enc.encode('The secret of a good deep learning '
                               'textbook is'), 40, temperature=0.8,
                    top_k=50)
print(enc.decode(out))
```

```{.python .input #gpt-loading-gpt-2-4}
%%tab jax
out = gpt2.generate(enc.encode('Alan Turing theorized that computers '
                               'would one day become'), 16, top_k=1)
print(enc.decode(out))
out = gpt2.generate(enc.encode('The secret of a good deep learning '
                               'textbook is'), 40, seed=0, temperature=0.8,
                    top_k=50)
print(enc.decode(out))
```

Take stock of what just happened: a class we wrote in two notebook
sections, with the right five flags, runs a model that in 2019 was
considered too dangerous to release. The gap between our minute of
training and GPT-2 was never architectural — it is five orders of
magnitude of data and compute, and the engineering to spend them. The
rest of this chapter dissects exactly that gap: making generation cheap
(the KV cache, next), other ways of wiring the same block
(encoders and cross-attention), scaling its FFN sideways
(mixture of experts), and the laws that say what all those FLOPs buy
(scaling laws) — where the constructor calls for Llama, Qwen, and
DeepSeek appear as rows of a table, every one of them an argument list
for this class.

## Summary

A GPT is a token embedding, a stack of causally masked transformer
blocks, a final norm, and an output head tied to the embedding; positions
enter either as a learned table added at the bottom (GPT-2) or as
rotations applied inside every attention head (RoPE, the modern default).
Our `GPT` class takes the block's flags plus `pos`, and its causal
attention drops into `d2l.TransformerBlock` through the same
`attn_factory` hook later sections use for cache-friendly attention.
Trained from scratch on a 180 KB novel, the modern configuration reaches
its best validation loss within a few hundred steps and then memorizes —
the val/train gap, and the verbatim quotations in its samples, are the
honest reading of a 30-to-1 parameter-to-data ratio, and the argument for
the scaling laws that close the chapter. Flipping `pre_norm=False` at a
learning rate the pre-norm model tolerates pins training at the unigram
entropy: the initialization-time gradient starvation of the previous
section, realized as a model that never learns to use context. Sampling
is temperature plus truncation, implemented once and reused by the real
GPT-2: the released 124M checkpoint loads into our class with a
name-mapping dictionary (transposing the Conv1D layout in PyTorch;
adopting it unchanged in JAX), its tokenizer reassembles from two pinned
data files and the BPE pattern of ch. 8's tokenizer, and the loaded model
passes both a perplexity check and a readable one — completing English
prompts the way the history books say it should.

## Exercises

1. Account for every one of GPT-2's 124.4M parameters using the block
   census of :numref:`sec_transformer-block` plus the embeddings: 12
   blocks of roughly $12d^2$ at $d = 768$ (plus biases), a
   $50257 \times 768$ token table, a $1024 \times 768$ position table,
   and the final norm. What fraction sits in the embeddings, and why does
   weight tying make the "124M parameters" figure slightly generous as a
   measure of the transformer proper?
2. Instantiate the GPT-2 configuration with one flag changed to
   `norm='rms'` and attempt the weight-loading cell. Which line fails,
   and why must it fail rather than load silently? List every tensor in
   the checkpoint that would have no destination (and every module
   parameter with no source) under the modern flags
   `pos='rope', act='swiglu'`.
3. Add nucleus (top-$p$) sampling :cite:`Holtzman.Buys.Du.ea.2020` to
   `generate`: keep the smallest set of tokens whose probabilities sum to
   at least $p$. On the GPT-2 model, fix the seed and compare
   continuations under top-$k = 50$ and $p = 0.9$ for a prompt where the
   model is confident and one where it is not. Which truncation adapts
   its cutoff to the model's uncertainty, and how?
4. Show that as $\tau \to 0$, sampling with temperature $\tau$ becomes
   greedy decoding, and as $\tau \to \infty$ it converges to the uniform
   distribution over the vocabulary. Then measure the empirical entropy
   of 200-character samples from the char model at
   $\tau \in \{0.5, 1, 2, 4\}$ and plot it against these two limits.
5. Our char model trained with RoPE at context 128 but `max_len` allows
   evaluation at 512. Measure its validation loss at contexts 128, 256,
   and 512 (build the longer-context dataloaders as in
   :numref:`sec_positional-information`). Does the failure of naive RoPE
   extrapolation that section demonstrated reappear inside a full
   transformer? Apply position interpolation — scale every angle by
   $128/512$ — and report what changes.
6. Compare GPT-2 and our char model on the same text on equal footing:
   convert each model's mean loss on a held-out Time Machine passage to
   *bits per character* (GPT-2's per-token loss must be divided by the
   average characters per token of its tokenization). Which model wins,
   by how much, and what does each side of the comparison pay for — a
   28-symbol character vocabulary against 50,257 subwords, and 40 GB of
   WebText against none?

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.2]{.kicker}

A GPT from scratch<br>
**one class, many configurations · train it, break it, sample from it · load the real GPT-2**
:::
:::

::: {.slide title="From blocks to a language model"}
Four ingredients on top of the stacked block:

- **Token embedding + tied head** — the same table in and out
  (Press & Wolf, 2017); GPT-2 ties too.
- **Causal mask** — as a *flag* of the fused attention kernels; the
  $n \times n$ score matrix is never materialized.
- **Positions** — `'learned'` table added at the bottom (GPT-2) or
  `'rope'` rotations inside every head (the modern default).
- **The blocks** — `d2l.TransformerBlock`, unchanged, via `attn_factory`.
:::

::: {.slide title="The GPT class"}
@gpt-from-blocks-to-a-language-model-1
:::

::: {.slide title="Train the modern configuration"}
4.7M parameters, character-level Time Machine, one GPU, about a minute:

@!gpt-training-the-modern-configuration
:::

::: {.slide title="Reading the two curves"}
- Train falls smoothly to ~0.15 nats; validation bottoms near 1.5 within
  a few hundred steps, then **rises**.
- 16M training tokens over a 100k-character book ≈ 160 passes, with
  dozens of parameters per character: past the first epochs there is
  nothing to learn but the book itself, verbatim.
- The cure is **more data**, not more steps — the scaling-laws section's
  subject.

::: {.d2l-note}
Cost anchor: this run ≈ $6ND \approx 5 \times 10^{14}$ FLOPs. GPT-2's
run ≈ $10^{20}$. Frontier ≈ $10^{25}$+. Same code at every scale.
:::
:::

::: {.slide title="Breaking it with one flag"}
`pre_norm=False`, learning rate $3\times10^{-3}$ (fine for pre-norm):

@!gpt-breaking-it-with-one-flag

No divergence — worse: pinned at **2.83 nats = the unigram entropy** of
the text. Attention never learns to use context, exactly as the
at-initialization gradients predicted.
:::

::: {.slide title="Sampling: temperature and truncation"}
@gpt-sampling-from-the-model-1

Deliberately naive: every token reruns the full forward pass — measuring
and fixing that is the next section (KV cache).
:::

::: {.slide title="What a minute of training sounds like"}
@!gpt-sampling-from-the-model-2

Fluent pseudo-Wells — much of it *verbatim* Wells: the validation gap
made audible. Temperature barely matters when a memorizing model is this
confident.
:::

::: {.slide title="Loading GPT-2: config = constructor call"}
GPT-2 (124M) **is** our class with the 2019 flags:
`pos='learned', norm='layer', act='gelu', pre_norm=True, bias=True`.

- Weights + tokenizer files pinned by sha1 in `d2l.DATA_HUB`.
- Tokenizer: GPT-2's merge list + vocabulary + ch. 8's BPE pattern,
  assembled into tiktoken — no model library.

@gpt-loading-gpt-2-1
:::

::: {.slide title="The weight mapping"}
One dictionary from checkpoint names to modules; the one trap is GPT-2's
`Conv1D` layout, $(\textrm{in}, \textrm{out})$ — transpose for
`nn.Linear`, adopt unchanged for `nnx.Linear`:

@gpt-loading-gpt-2-2
:::

::: {.slide title="Did it work?"}
Silent-failure insurance, one number and one sentence:

@!gpt-loading-gpt-2-3

@!gpt-loading-gpt-2-4
:::

::: {.slide title="Recap"}
- GPT = embedding + causal blocks + final norm + tied head; positions
  learned-or-rotary; one class, flags for a decade of designs.
- Honest curves: best validation early, then memorization — data, not
  steps.
- `pre_norm=False` at a healthy learning rate pins training at the
  unigram plateau.
- Sampling = temperature + truncation; naive generation recomputes
  everything (the KV cache fixes this, next).
- The released GPT-2 loads into our class and completes English the way
  the history books say it should.
:::
