# A GPT from Scratch
:label:`sec_gpt`

The block of :numref:`sec_transformer-block` maps a sequence of vectors to
a sequence of the same shape. A language model has a more specific job: it
reads text and says what comes next. How does a stack of blocks become such
a model, and how do we train it?

The training text suggests the answer. A language model solves the same
prediction problem at every position of every training sequence. Given "the
European lobste", it must predict "r"; given "the European lobster", it must
predict a space. One sequence of 128 bytes therefore holds 128 training
examples, and a model can learn from all of them at once if it meets two
requirements. Each position must predict its successor without reading it
or any later token. And the same parameters must serve every position and
every length of context, from one token to the whole window.

The transformer block meets the second requirement already: it applies the
same weights at every position. A causal mask inside attention meets the
first. Adding the remaining parts,
an embedding, positions, and an output head, gives a GPT (*generative
pretrained transformer*), which we write as a `GPT` class. We train it on
the bytes of Wikipedia articles, test where the normalization belongs, and
sample text from it. Finally, we load the released 124-million-parameter
GPT-2 :cite:`Radford.Wu.Child.ea.2019` into the same class, which the rest
of this chapter reuses. Its sampling loop recomputes the whole context for
every new token, a waste that the next section removes.

```{.python .input #gpt-a-gpt-from-scratch}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #gpt-a-gpt-from-scratch}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## From Blocks to a Language Model
:label:`subsec_gpt-blocks-to-lm`

Four ingredients turn a stack of transformer blocks into a language model,
and each solves one part of the problem. :numref:`fig_gpt-stack` shows how
they fit together.

![A GPT from bottom to top. A table $\mathbf{E}$ embeds the token ids into
the residual stream, $L$ pre-norm blocks of causal attention and
feed-forward network add to the stream, and after a final normalization the
same table $\mathbf{E}$ maps the stream back to one logit per vocabulary
entry. Learned positions are added after the embedding; RoPE instead
rotates queries and keys inside each attention layer.](../img/mdl-transformers-gpt-stack.svg)
:label:`fig_gpt-stack`

**A token embedding and a tied head.** A table $\mathbf{E}$ with one row of
width $d$ per vocabulary entry maps token ids into the residual stream.
After the final block and one last normalization, the *same* table maps the
stream back: the logit of token $v$ is the dot product of the stream with
row $v$ of $\mathbf{E}$, so all logits together are
$\mathbf{E}\mathbf{h}$. Sharing, or *tying*, the two matrices saves
$V \times d$ parameters for a vocabulary of size $V$ and helps at small
scale :cite:`Press.Wolf.2017`. GPT-2 ties them too, which spares us a
tensor when we load its weights.

**A causal mask.** Position $t$ predicts token $t+1$ from tokens
$1, \ldots, t$, so its query may attend only to keys at positions up to
$t$. In :numref:`sec_transformer-block` the small language model passed this
lower-triangular mask to every block as a boolean array. Here the attention
module applies it itself: it passes the attention function the flag
`is_causal=True` instead of a mask. The fused attention kernels of
:numref:`sec_attention-at-scale` take the causal mask in this form and
never store the $n \times n$ matrix of scores. With the mask, one forward
pass over a window computes all its next-token predictions at once, each
from its own prefix.

:begin_tab:`jax`
`jax.nn.dot_product_attention` uses XLA's implementation by default, which
builds the mask and the matrix of scores in memory. Wrapped in
`jax.checkpoint`, as in `attend` below, it keeps them only during its own
forward pass: the backward pass stores the queries, keys, and values and
computes the attention weights again, which is what a fused kernel does.
:numref:`sec_fast_transformer` selects cuDNN's fused kernel instead.
:end_tab:

**Positions.** Attention alone treats its inputs as a set: permuting the
tokens permutes the outputs in the same way
(:numref:`sec_positional-information`). The `pos` argument selects one of
the two schemes of that section. `'learned'` adds a trained vector per
position to the embedding, as GPT-2 does, which limits the context to the
length of the table. `'rope'` rotates the queries and keys of every head by
an angle proportional to their position, so that the score of a query and
a key depends on their offset. The attention module calls `d2l.rope` where
it computes the queries and keys. The angles depend only on the positions,
so `GPT` computes their cosines and sines once per forward pass with
`d2l.rope_tables` and passes them to every block.

The attention module below puts the mask and the positions in one place.
One matrix `W_qkv` computes the queries, keys, and values together; `qkv`
splits them into heads and, for RoPE, rotates the queries and the keys;
and the attention call is causal. A block passes its attention keys,
values, and an optional mask, and any keyword arguments of its own call,
here the RoPE tables. This module uses its input as the keys and
the values, and it rejects a mask, because its own mask is always causal.

```{.python .input #gpt-from-blocks-to-a-language-model-4}
%%tab pytorch
class CausalAttention(nn.Module):  #@save
    """Fused causal self-attention, optionally rotary; num_kv_heads <
    num_heads shares each key-value head across a group of queries."""
    def __init__(self, num_hiddens, num_heads, *, bias=False, rope=False,
                 num_kv_heads=None):
        super().__init__()
        self.num_heads, self.rope = num_heads, rope
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
        self.W_qkv = nn.Linear(num_hiddens, num_hiddens + 2 * self.kv_hiddens,
                               bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def qkv(self, X, offset=0, tables=None):
        """Queries, keys, and values of X, each (batch, heads, length, head
        width), at positions offset, offset + 1, ...; tables, if given, are
        the d2l.rope_tables of these positions."""
        B, T, D = X.shape
        q, k, v = self.W_qkv(X).split([D, self.kv_hiddens, self.kv_hiddens],
                                      -1)
        q = q.reshape(B, T, self.num_heads, -1).transpose(1, 2)
        k, v = (u.reshape(B, T, self.num_kv_heads, -1).transpose(1, 2)
                for u in (k, v))
        if self.rope:
            q, k = (d2l.rope(u, offset, tables=tables) for u in (q, k))
        return q, k, v

    def forward(self, X, keys=None, values=None, mask=None, tables=None):
        # Self-attention: X also serves as the keys and the values, and the
        # mask is causal, so a block's mask argument must stay None
        if mask is not None:
            raise ValueError('CausalAttention applies its own mask')
        q, k, v = self.qkv(X, tables=tables)
        Y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            enable_gqa=self.num_kv_heads < self.num_heads)
        return self.W_o(Y.transpose(1, 2).flatten(2))  # (batch, length, d)
```

```{.python .input #gpt-from-blocks-to-a-language-model-4}
%%tab jax
class CausalAttention(nnx.Module):  #@save
    """Causal self-attention, optionally rotary; num_kv_heads < num_heads
    shares each key-value head across a group of queries."""
    def __init__(self, num_hiddens, num_heads, *, bias=False, rope=False,
                 num_kv_heads=None, rngs):
        self.num_heads, self.rope = num_heads, rope
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
        self.W_qkv = d2l.Linear(num_hiddens,
                                num_hiddens + 2 * self.kv_hiddens,
                                use_bias=bias, rngs=rngs)
        self.W_o = d2l.Linear(num_hiddens, num_hiddens, use_bias=bias,
                              rngs=rngs)

    def qkv(self, X, offset=0, tables=None):
        """Queries, keys, and values of X, each (batch, heads, length, head
        width), at positions offset, offset + 1, ...; tables, if given, are
        the d2l.rope_tables of these positions."""
        B, T, D = X.shape
        q, k, v = jnp.split(self.W_qkv(X), [D, D + self.kv_hiddens], axis=-1)
        q = q.reshape(B, T, self.num_heads, -1).swapaxes(1, 2)
        k, v = (u.reshape(B, T, self.num_kv_heads, -1).swapaxes(1, 2)
                for u in (k, v))
        if self.rope:
            q, k = (d2l.rope(u, offset, tables=tables) for u in (q, k))
        return q, k, v

    @staticmethod
    @jax.checkpoint  # The backward pass recomputes the T x T weights
    def attend(q, k, v):
        """Causal attention of (batch, length, heads, head width) queries,
        keys, and values."""
        return jax.nn.dot_product_attention(q, k, v, is_causal=True)

    def __call__(self, X, keys=None, values=None, mask=None, tables=None):
        # Self-attention: X also serves as the keys and the values, and the
        # mask is causal, so a block's mask argument must stay None
        if mask is not None:
            raise ValueError('CausalAttention applies its own mask')
        q, k, v = self.qkv(X, tables=tables)
        Y = self.attend(*(u.swapaxes(1, 2) for u in (q, k, v)))
        return self.W_o(Y.reshape(*X.shape[:2], -1))  # (batch, length, d)
```

**The blocks.** `d2l.TransformerBlock` is used unchanged, with its defaults
of :numref:`sec_transformer-block`: pre-norm, RMSNorm, and a SwiGLU
feed-forward network. The causal attention enters through its
`attn_factory` argument. Its option `num_kv_heads` lets groups of query
heads share one key--value head, which later sections use to make
generation cheaper.

`GPT` keeps the attention module as its class attribute
`CausalAttention`, so that a subclass can replace it, as
:numref:`sec_fast_transformer` does to select a faster kernel.

```{.python .input #gpt-from-blocks-to-a-language-model-1}
%%tab pytorch
class GPT(nn.Module):  #@save
    """Decoder-only transformer language model built from configurable
    blocks."""
    CausalAttention = CausalAttention  # A subclass may replace it

    def __init__(self, vocab_size, *, num_hiddens=256, num_heads=8,
                 num_blks=6, max_len=1024, pos='rope', norm='rms',
                 act='swiglu', pre_norm=True, bias=False, dropout=0,
                 num_kv_heads=None, ffn_factory=None):
        super().__init__()
        assert pos in ('rope', 'learned'), f'unknown pos: {pos!r}'
        self.pos, self.max_len, self.num_hiddens = pos, max_len, num_hiddens
        self.head_width = num_hiddens // num_heads
        self.token_emb = nn.Embedding(vocab_size, num_hiddens)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pos == 'learned':
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        attn = lambda: self.CausalAttention(
            num_hiddens, num_heads, bias=bias, rope=pos == 'rope',
            num_kv_heads=num_kv_heads)
        self.blks = nn.ModuleList([
            d2l.TransformerBlock(num_hiddens, num_heads, dropout=dropout,
                                 norm=norm, act=act, pre_norm=pre_norm,
                                 bias=bias, attn_factory=attn,
                                 ffn_factory=ffn_factory)
            for _ in range(num_blks)])
        self.norm = (nn.RMSNorm if norm == 'rms'
                     else nn.LayerNorm)(num_hiddens, eps=1e-5)
        self.init_weights()

    def forward(self, X):                     # X: (batch, T) token ids
        H, tables = self.token_emb(X), None   # (batch, T, d)
        if self.pos == 'learned':
            H = H + self.pos_emb(torch.arange(X.shape[1], device=X.device))
        else:  # RoPE angles of positions 0, ..., T - 1, for every block
            tables = d2l.rope_tables(X.shape[1], self.head_width,
                                     device=X.device)
        for blk in self.blks:                 # L blocks, causal attention
            H = blk(H, tables=tables)
        return F.linear(self.norm(H), self.token_emb.weight)  # (batch, T, V)
```

```{.python .input #gpt-from-blocks-to-a-language-model-1}
%%tab jax
class GPT(nnx.Module):  #@save
    """Decoder-only transformer language model built from configurable
    blocks."""
    CausalAttention = CausalAttention  # A subclass may replace it

    def __init__(self, vocab_size, *, num_hiddens=256, num_heads=8,
                 num_blks=6, max_len=1024, pos='rope', norm='rms',
                 act='swiglu', pre_norm=True, bias=False, dropout=0,
                 num_kv_heads=None, ffn_factory=None, rngs):
        assert pos in ('rope', 'learned'), f'unknown pos: {pos!r}'
        self.pos, self.max_len, self.num_hiddens = pos, max_len, num_hiddens
        self.head_width = num_hiddens // num_heads
        init = nnx.initializers.normal(0.02)
        self.token_emb = d2l.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        if pos == 'learned':
            self.pos_emb = nnx.Embed(max_len, num_hiddens,
                                     embedding_init=init, rngs=rngs)
        attn = lambda rngs: self.CausalAttention(
            num_hiddens, num_heads, bias=bias, rope=pos == 'rope',
            num_kv_heads=num_kv_heads, rngs=rngs)
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, dropout=dropout,
                                 norm=norm, act=act, pre_norm=pre_norm,
                                 bias=bias, attn_factory=attn,
                                 ffn_factory=ffn_factory, rngs=rngs)
            for _ in range(num_blks)])
        self.norm = (nnx.RMSNorm if norm == 'rms'
                     else nnx.LayerNorm)(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.init_weights(rngs)

    def __call__(self, X):                    # X: (batch, T) token ids
        H, tables = self.token_emb(X), None   # (batch, T, d)
        if self.pos == 'learned':
            H = H + self.pos_emb(jnp.arange(X.shape[1]))
        else:  # RoPE angles of positions 0, ..., T - 1, for every block
            tables = d2l.rope_tables(X.shape[1], self.head_width)
        for blk in self.blks:                 # L blocks, causal attention
            H = blk(H, tables=tables)
        return self.token_emb.attend(self.norm(H))  # (batch, T, V), tied
```

:begin_tab:`jax`
The token embedding is `d2l.Embed` of :numref:`subsec_utils_embed`. Its
gradient is one matrix product of the one-hot rows of the tokens with the
gradients, not the atomic additions of an ordinary lookup, which wait on
one another when a few frequent bytes fill most positions. The product
splits across GPUs like any other: each GPU multiplies its share of the
positions, and the shares are added up with the other gradients.
:end_tab:

`GPT` initializes its weights as GPT-2 does. Every weight matrix is
drawn from $\mathcal{N}(0, 0.02^2)$ and every bias starts at zero, with one
exception. Each block has two matrices that write into the residual stream:
the output projection $\mathbf{W}_o$ of attention and the second matrix
$\mathbf{W}_2$ of the feed-forward network. Their standard deviation is
divided by a further $\sqrt{2L}$, where $2L$ counts the residual branches of
$L$ blocks.

The reason is the growth of the stream. In a pre-norm block each branch
reads the normalized stream, so the size of its output is set by its
weights and not by the stream it reads. The stream is the sum of the
embedding and all branch outputs. These are roughly uncorrelated, so their
mean squares add, and the mean square of the stream grows in proportion to
the number of branches. Dividing the standard deviation of each writing
matrix by $\sqrt{2L}$ divides the mean square of each branch output by
$2L$. The $2L$ contributions together are then about as large as a single
unscaled one, whatever the depth. Without normalization the growth would
compound instead, as :numref:`sec_init_param` showed. This depth scaling
is a common choice rather than a law: OLMo 2 returned to a truncated
$\mathcal{N}(0, 0.02^2)$ for all parameters :cite:`OLMo.2025`.

The constructor draws the embedding tables this way when it creates them.
The method `init_weights`, which it calls last, applies the rule to every
linear layer:

```{.python .input #gpt-from-blocks-to-a-language-model-5}
%%tab pytorch
@d2l.add_to_class(GPT)  #@save
def init_weights(self):
    """GPT-2's initialization of the linear layers: N(0, 0.02^2) weights and
    zero biases; W_o and W_2, which write to the stream, get std
    0.02 / sqrt(2L)."""
    L = len(self.blks)
    for name, m in self.named_modules():
        if isinstance(m, nn.Linear):
            std = 0.02 / (2 * L) ** 0.5 if name.endswith(
                ('W_o', 'W_2')) else 0.02
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

```{.python .input #gpt-from-blocks-to-a-language-model-5}
%%tab jax
@d2l.add_to_class(GPT)  #@save
def init_weights(self, rngs):
    """GPT-2's initialization of the linear layers: N(0, 0.02^2) weights and
    zero biases; W_o and W_2, which write to the stream, get std
    0.02 / sqrt(2L)."""
    L = len(self.blks)
    for path, m in nnx.iter_graph(self):
        if isinstance(m, nnx.Linear):
            std = 0.02 / (2 * L) ** 0.5 if path[-1] in ('W_o', 'W_2') else 0.02
            m.kernel[...] = std * jax.random.normal(rngs.params(),
                                                    m.kernel.shape)
            if m.bias is not None:
                m.bias[...] = jnp.zeros(m.bias.shape)
```

A shape check confirms one logit per vocabulary entry at every position,
and the standard deviations of two matrices of the first block confirm the
initialization:

```{.python .input #gpt-from-blocks-to-a-language-model-2}
%%tab pytorch
model = GPT(vocab_size=256)
X = torch.zeros(2, 16, dtype=torch.long)
d2l.check_shape(model(X), (2, 16, 256))
attention = model.blks[0].attention
print(f'{d2l.num_params(model) / 1e6:.2f}M parameters; initial std: '
      f'W_qkv {attention.W_qkv.weight.std():.4f}, '
      f'W_o {attention.W_o.weight.std():.4f}')
```

```{.python .input #gpt-from-blocks-to-a-language-model-2}
%%tab jax
model = GPT(vocab_size=256, rngs=nnx.Rngs(d2l.get_key()))
X = jnp.zeros((2, 16), dtype=jnp.int32)
d2l.check_shape(model(X), (2, 16, 256))
attention = model.blks[0].attention
print(f'{d2l.num_params(model) / 1e6:.2f}M parameters; initial std: '
      f'W_qkv {attention.W_qkv.kernel[...].std():.4f}, '
      f'W_o {attention.W_o.kernel[...].std():.4f}')
```

The output projection starts with a standard deviation of
$0.02/\sqrt{12} \approx 0.0058$ for the six blocks, the other matrices
with $0.02$. A forward pass through untrained stacks of increasing depth
tests the argument for this scaling. We feed random tokens and measure the
mean square of the stream after the last block, once with every matrix at
$0.02$ and once with the scaled writing matrices:

```{.python .input #gpt-from-blocks-to-a-language-model-3}
%%tab pytorch
def stream_ms(num_blks, scaled):
    """Mean square of the stream after the last block, for random tokens."""
    net = GPT(vocab_size=256, num_blks=num_blks)
    if not scaled:  # Undo the depth scaling of W_o and W_2
        for name, p in net.named_parameters():
            if name.endswith(('W_o.weight', 'W_2.weight')):
                nn.init.normal_(p, std=0.02)
    with torch.no_grad():
        H = net.token_emb(torch.randint(0, 256, (8, 128)))
        for blk in net.blks:
            H = blk(H)
    return H.square().mean().item()

for scaled in (False, True):
    print(f'scaled={scaled}: mean square after 8/16/32 blocks: ' + '/'.join(
        f'{stream_ms(L, scaled):.4f}' for L in (8, 16, 32)))
```

```{.python .input #gpt-from-blocks-to-a-language-model-3}
%%tab jax
def stream_ms(num_blks, scaled):
    """Mean square of the stream after the last block, for random tokens."""
    net = GPT(vocab_size=256, num_blks=num_blks, rngs=nnx.Rngs(d2l.get_key()))
    if not scaled:  # Undo the depth scaling of W_o and W_2
        for path, m in nnx.iter_graph(net):
            if path and path[-1] in ('W_o', 'W_2'):
                m.kernel[...] = 0.02 * jax.random.normal(d2l.get_key(),
                                                         m.kernel.shape)
    H = net.token_emb(jax.random.randint(d2l.get_key(), (8, 128), 0, 256))
    for blk in net.blks:
        H = blk(H)
    return float(jnp.mean(H ** 2))

for scaled in (False, True):
    print(f'scaled={scaled}: mean square after 8/16/32 blocks: ' + '/'.join(
        f'{stream_ms(L, scaled):.4f}' for L in (8, 16, 32)))
```

The embedding alone has a mean square of $0.02^2 = 0.0004$. Without the
scaling, the mean square of the stream about doubles with each doubling of
the depth, as a sum of $2L$ comparable terms should. With the scaling, it
barely grows from 8 blocks to 32.

The parameter count follows from the block. Attention holds
four $d \times d$ matrices, $\mathbf{W}_q$, $\mathbf{W}_k$,
$\mathbf{W}_v$, and $\mathbf{W}_o$, and the feed-forward network holds
$8d^2$, as two matrices of width $4d$ with GELU or three of width $8d/3$
with SwiGLU. A block thus holds about $12d^2$ parameters, 0.79 million at
$d = 256$ and 4.7 million for six blocks, while an embedding of 256 rows
adds only 65,536. GPT-2 small has $d = 768$ and 12 blocks, so its blocks hold
$12 \times 12 \times 768^2 \approx 85$ million parameters and its token
table, $50{,}257 \times 768$, another 38.6 million. An untied head would
add a second table of that size and make the model about 31% larger.

## Training the Model
:label:`subsec_gpt-training`

### The Training Text

A model of 4.8 million parameters needs more text than one novel. The
training run below makes about 8 million predictions. *The Time Machine*
has about 170,000 characters, so the run would pass over the book about
fifty times, often enough for a model of this size to memorize much of it
rather than learn English (an exercise measures how much). We therefore
train on the corpus of the optimization experiments
(:numref:`subsec_tinylm`), `d2l.WikiText103`: Wikipedia articles kept as
raw text, with case, digits, and punctuation, about 270 million bytes for
training. Each training window of 128 bytes starts at a random offset, so
the run reads about 3% of the text and almost never the same passage twice.
The validation windows are consecutive pieces of the official validation
split, about 1.1 million bytes.

```{.python .input #gpt-training-the-model-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
print(f'{len(data.train) / 1e6:.0f}M training bytes, '
      f'{len(data.val) / 1e6:.2f}M validation bytes, '
      f'vocabulary of {len(data.vocab)}')
```

The tokens are the UTF-8 bytes of the text, the byte-level tokenizer of
:numref:`sec_text-sequence` without any merges, rather than its subword
tokens. Bytes have two advantages here. The 256-entry table puts 65,536 of
the model's parameters into embeddings, about 1.4% of the total, so nearly
all of them sit in the blocks; GPT-2's vocabulary of 50,257 tokens would put
12.9 million into the table at width 256, almost three times the rest of
the model. And because each token is one byte, the loss in nats per byte
divided by $\log 2$ is the number of bits per byte of
:numref:`subsec_bits-per-byte`, which does not depend on the tokenizer.

### The Training Recipe

The training recipe is the one :numref:`chap_optimization` assembled, and
each of its four parts addresses a problem met there. The optimizer is
AdamW (:numref:`sec_adamw`), which divides each coordinate's step by a
running estimate of its gradient's magnitude and applies weight decay
separately from that rescaling. The decay, $\lambda = 0.1$, acts on the
weight matrices only; embeddings, normalization gains, and biases are
exempt (:numref:`subsec_adamw-what-not-to-decay`). The learning rate rises
linearly from zero over the first 100 steps, while Adam's estimates still
rest on few gradients, and then falls along a half cosine to zero at the end
of training (`d2l.cosine_warmup`, :eqref:`eq_cosine_warmup`). Gradients are
clipped to a global norm of 1, which shortens the rare large gradient
without changing its direction (:numref:`sec_practice`).

`LMOptimizer` bundles the four parts behind the interface of an ordinary
optimizer, so the training loop `d2l.train_steps` of :numref:`sec_test_models`,
which draws minibatches of windows and takes a fixed number of steps on the
next-byte loss, uses it unchanged. We use no dropout: a model that almost never
reads the same passage twice has nothing to memorize.

```{.python .input #gpt-training-the-model-2}
%%tab pytorch
class LMOptimizer:  #@save
    """AdamW with weight decay on the weight matrices only, a linear warmup
    and a cosine decay of the learning rate, and gradient-norm clipping."""
    def __init__(self, model, lr, num_steps, warmup, weight_decay=0.1,
                 max_norm=1.0):
        named = list(model.named_parameters())
        # Embedding tables are named *emb* (token_emb, pos_emb) by convention
        decay = [p for n, p in named if p.ndim == 2 and 'emb' not in n]
        rest = [p for n, p in named if p.ndim != 2 or 'emb' in n]
        self.optimizer = torch.optim.AdamW(
            [{'params': decay, 'weight_decay': weight_decay},
             {'params': rest, 'weight_decay': 0.0}], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda t: d2l.cosine_warmup(t, num_steps, warmup))
        self.params, self.max_norm = decay + rest, max_norm

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        nn.utils.clip_grad_norm_(self.params, self.max_norm)
        self.optimizer.step()
        self.scheduler.step()
```

```{.python .input #gpt-training-the-model-2}
%%tab jax
def lm_decay_mask(params):  #@save
    """Decay the weight matrices; exempt embeddings, gains, and biases."""
    # Embedding tables are named *emb* (token_emb, pos_emb) by convention
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

@optax.inject_hyperparams  #@save
def lm_adamw(lr=0.0, num_steps=1, warmup=0, weight_decay=0.0, max_norm=1.0):
    """The transformation of LMOptimizer, with its hyperparameters as state."""
    schedule = lambda t: lr * d2l.cosine_warmup(t, num_steps, warmup)
    return optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.adamw(schedule, weight_decay=weight_decay, mask=lm_decay_mask))

class LMOptimizer(nnx.Optimizer):  #@save
    """AdamW with weight decay on the weight matrices only, a linear warmup
    and a cosine decay of the learning rate, and gradient-norm clipping."""
    tx = lm_adamw()  # Shared by all instances: the step compiles once
    def __init__(self, model, lr, num_steps, warmup, weight_decay=0.1,
                 max_norm=1.0):
        super().__init__(model, LMOptimizer.tx, wrt=nnx.Param)
        hyperparams = dict(lr=lr, num_steps=num_steps, warmup=warmup,
                           weight_decay=weight_decay, max_norm=max_norm)
        for name, value in hyperparams.items():
            self.opt_state.hyperparams[name][...] = value
```

:begin_tab:`pytorch`
`LMOptimizer` keeps the two parameter groups of AdamW and a `LambdaLR`
scheduler that multiplies the base rate by `d2l.cosine_warmup`. Its `step`
clips, updates, and advances the schedule, in that order, so the three
always stay in step.
:end_tab:

:begin_tab:`jax`
`LMOptimizer` is an `nnx.Optimizer` whose transformation chains clipping in
front of `optax.adamw`. The schedule is passed as the learning rate, and
`lm_decay_mask` marks the parameters that are decayed.
`optax.inject_hyperparams` keeps the peak rate, the step counts, the decay,
and the clipping norm in the optimizer state rather than in the
transformation. All instances then share one transformation, `tx`, and a
jitted training step compiled for one instance runs the next one too, even
at a different learning rate.
:end_tab:

### Loss Curves and Baselines

A loss in nats means little on its own, so after training we compute
reference levels to compare it with. We train for
1,000 steps of 64 windows and record both losses every 50 steps. The
training loss is the mean over those 50 steps. The validation loss is the
mean cross-entropy per byte on the validation windows, which is the
logarithm of `d2l.lm_perplexity` (:numref:`sec_test_models`).

The experiments of this section build their models and plain optimizers
through two shorthands: `gpt` constructs a `GPT` with fresh random weights,
and `adamw` returns AdamW without weight decay.

```{.python .input #gpt-training-the-model-3}
%%tab pytorch
gpt = GPT

def adamw(model, lr):
    """Plain AdamW without weight decay."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
```

```{.python .input #gpt-training-the-model-3}
%%tab jax
def gpt(*args, **kwargs):  # Fresh random weights for every model
    return GPT(*args, **kwargs, rngs=nnx.Rngs(d2l.get_key()))

def adamw(model, lr):
    """Plain AdamW without weight decay."""
    return nnx.Optimizer(model, optax.adamw(lr, weight_decay=0.0),
                         wrt=nnx.Param)
```

```{.python .input #gpt-training-the-modern-configuration}
%%tab pytorch, jax
model = gpt(len(data.vocab))
num_steps, every = 1000, 50
optimizer = LMOptimizer(model, lr=2e-3, num_steps=num_steps, warmup=100)
steps, train_curve, val_curve = [], [], []
for step in range(every, num_steps + 1, every):
    losses = d2l.train_steps(model, data, optimizer, every)
    steps.append(step)
    train_curve.append(sum(losses) / every)
    val_curve.append(math.log(d2l.lm_perplexity(model, data.val_dataloader())))
d2l.plot(steps, [train_curve, val_curve], 'step', 'loss (nats)',
         legend=['train', 'validation'], fmts=('-', '--'))
print(f'step {num_steps}: train {train_curve[-1]:.2f}, '
      f'validation {val_curve[-1]:.2f} nats per byte')
```

The two curves fall together, and the validation loss never turns upward.
Almost every training window is text the model has not read before, so what
it learns carries over to the held-out articles. At the end, the two losses
lie within a few hundredths of a nat of each other. Most of the decrease
comes in the first few hundred steps, and both losses end near 1.35 nats
per byte, or about 1.9 bits per byte.

Two simple predictors give reference levels for these losses. The
*unigram* model predicts every byte with its frequency in the training
text and ignores all context. Its cross-entropy on that text is the entropy
of the byte frequencies. The *bigram* model predicts each byte from the
previous one, with the conditional frequencies of the training text. Both
are estimated on the first 10 million bytes of the training text, enough
to fix their levels to two decimal places.

```{.python .input #gpt-training-the-model-4}
%%tab pytorch, jax
text = data.train[:10**7].astype(np.int64)  # The first 10M training bytes
V = len(data.vocab)
unigram = np.bincount(text, minlength=V) / len(text)
pairs = np.bincount(text[:-1] * V + text[1:], minlength=V * V).reshape(V, V)
bigram = pairs / np.maximum(pairs.sum(1, keepdims=True), 1)
p = unigram[unigram > 0]
print(f'unigram {-(p * np.log(p)).sum():.2f}, '
      f'bigram {-np.log(bigram[text[:-1], text[1:]]).mean():.2f} '
      f'nats per byte')
```

A model that knows nothing about the text scores $\log 256 \approx 5.55$
nats, close to the loss at initialization. The unigram level is 3.17 nats
and the bigram level 2.45. The trained model ends about a nat below the
bigram level, so it uses much more of the context than the previous byte.

### Training Compute

A rough count of operations places this run among larger ones. In the
forward pass, every parameter takes part in one multiplication and one
addition per token, so a model with $N$ parameters performs about $2N$
floating-point operations per token. The backward pass takes about twice
as many. Training on $D$ tokens therefore takes about

$$C \approx 6ND$$
:eqlabel:`eq_gpt-six-nd`

operations, ignoring the attention scores, which are small at short
contexts. Our run, with $N = 4.8$ million and $D = 8.2$ million, comes to
about $2 \times 10^{14}$. One pass of GPT-2 small, with 124 million
parameters, over roughly ten billion tokens of WebText comes to about
$7 \times 10^{18}$. The same pass of GPT-2 XL, with 1.5 billion parameters,
comes to about $10^{20}$, and the largest current training runs exceed
$10^{25}$. Across
these eleven orders of magnitude, the model definition hardly changes: an
embedding, a stack of blocks, a causal mask, and a head. What changes is
the system around it, from data pipelines to fused kernels and the parallel
hardware of :numref:`chap_performance`.

### Normalization Placement in Training
:label:`subsec_gpt-norm-placement`

:numref:`sec_transformer-block` placed the normalization at the input of
each sublayer (pre-norm) rather than after each residual addition, as the
original transformer did (post-norm). Its argument came from initialization
alone. Post-norm rescales the whole stream after every addition. In a deep
stack at initialization, this drives the tokens toward their average and
leaves the query and key projections with tiny gradients. We now test the
placement in training, with the model and data as above. Two runs differ
only in `pre_norm`. Both take 800 steps of plain AdamW at a constant
learning rate of $10^{-3}$, without warmup, a setting in which
pre-norm trains without trouble. A third run repeats the post-norm model
with a warmup. It uses `LMOptimizer` with the same peak rate, a warmup over
the first 400 steps, half the run, and the cosine decay that follows, but
without weight decay or clipping.

```{.python .input #gpt-breaking-it-with-one-flag}
%%tab pytorch, jax
for pre_norm, warmup in ((True, False), (False, False), (False, True)):
    ablation = gpt(len(data.vocab), pre_norm=pre_norm)
    optimizer = (LMOptimizer(ablation, lr=1e-3, num_steps=800, warmup=400,
                             weight_decay=0.0, max_norm=math.inf)
                 if warmup else adamw(ablation, lr=1e-3))
    losses = d2l.train_steps(ablation, data, optimizer, 800)
    print(f'pre_norm={pre_norm}, warmup={warmup}: loss at step '
          '100/200/400/800: ' + '/'.join(f'{sum(losses[k-50:k]) / 50:.2f}'
                                         for k in (100, 200, 400, 800)))
```

The pre-norm model trains normally. The post-norm model without warmup
does not diverge, but its loss stops at about 3.2 nats and stays there.
That is the unigram level computed above: the model predicts the byte
frequencies of the text and ignores its input, even the current byte.

Attention that has learned nothing would not explain this level by itself.
If every attention sublayer output zero, the feed-forward sublayers would
still see the current byte and could reach the bigram level of 2.45
nats. The tiny query and key gradients at initialization are therefore at
most part of the explanation. :citet:`xiong2020layer` showed that at
initialization the gradients of the parameters near the output of a
post-norm stack are large. A large learning rate without warmup then makes
training unstable, and a warmup avoids the failure. The third run confirms
this: with a warmup, the post-norm model leaves the unigram level and
trains, while pre-norm needs no warmup at this rate. The warmup here is
long, half the run, and an exercise asks how short it can be. GPT-2 already placed
the normalization at the input of each sublayer and added one after the
last block :cite:`Radford.Wu.Child.ea.2019`, the arrangement that `GPT`
uses by default.

## Sampling from the Model
:label:`subsec_gpt-sampling`

A language model outputs a distribution over the next token, and text
comes from decoding it. :numref:`sec_decoding` treated greedy search, beam
search, sampling, and the ways of reshaping the distribution before
sampling. The sampling method below implements the two most common ways.
The temperature $\tau > 0$ of :eqref:`eq_temperature` divides the logits
$z_v$ before the softmax, $p_v \propto \exp(z_v/\tau)$. At $\tau = 1$
this is the model's own distribution. A smaller $\tau$ sharpens it, and
as $\tau \to 0$ all probability goes to the most likely token, which is
greedy decoding. A larger $\tau$ flattens it toward uniform. For the
logits $(2, 1, 0)$, the probabilities are $(0.67, 0.24, 0.09)$ at
$\tau = 1$, $(0.87, 0.12, 0.02)$ at $\tau = 0.5$, and
$(0.51, 0.31, 0.19)$ at $\tau = 2$.

Top-$k$ truncation :cite:`Fan.Lewis.Dauphin.2018` keeps only the $k$
largest logits and sets the others to $-\infty$, which removes the long
tail of unlikely tokens. The nucleus (top-$p$) and min-$p$ rules of
:numref:`sec_decoding` truncate by probability instead and would fit into
the same place of the loop.

We add a `generate` method to the class. It is deliberately simple. For
every new token it runs the forward pass over the whole context again and
recomputes the keys and values of every earlier position, although they
have not changed. :numref:`sec_kv-cache` measures that waste and removes
it.

```{.python .input #gpt-sampling-from-the-model-1}
%%tab pytorch
@d2l.add_to_class(GPT)  #@save
@torch.inference_mode()
def generate(self, prefix, num_tokens, temperature=1.0, top_k=None):
    """Sample a continuation of the token-id list prefix."""
    ids = torch.tensor(prefix, device=next(self.parameters()).device)
    for _ in range(num_tokens):  # The tokens stay on the device
        logits = self(ids[None, -self.max_len:])[0, -1] / temperature
        if top_k is not None:
            cutoff = torch.topk(logits, top_k).values[-1]
            logits[logits < cutoff] = -torch.inf
        ids = torch.cat((ids, torch.multinomial(F.softmax(logits, -1), 1)))
    return ids.tolist()
```

```{.python .input #gpt-sampling-from-the-model-1}
%%tab jax
@nnx.jit(static_argnames='top_k')  #@save
def _gpt_sample_next(model, buf, t, key, temperature, top_k):
    """Sample the token after the first t tokens of buf and append it.

    A full buffer (t equal to its length) drops its oldest token first."""
    logits = model(buf)[0, t - 1] / temperature
    if top_k is not None:
        cutoff = jax.lax.top_k(logits, top_k)[0][-1]
        logits = jnp.where(logits < cutoff, -jnp.inf, logits)
    token = jax.random.categorical(key, logits)
    n = buf.shape[1]
    buf = jnp.where(t == n, jnp.roll(buf, -1, axis=1), buf)
    return buf.at[0, jnp.minimum(t, n - 1)].set(token), token

@d2l.add_to_class(GPT)  #@save
def generate(self, prefix, num_tokens, temperature=1.0, top_k=None):
    """Sample a continuation of the token-id list prefix."""
    n, ctx = self.max_len, list(prefix)[-self.max_len:]
    # One buffer of fixed shape (1, max_len): the step compiles once
    buf, t = jnp.asarray([ctx + [0] * (n - len(ctx))]), len(ctx)
    tokens = []
    for key in jax.random.split(d2l.get_key(), num_tokens):
        buf, token = _gpt_sample_next(self, buf, t, key, temperature, top_k)
        tokens.append(token)
        t = min(t + 1, n)
    return list(prefix) + [int(i) for i in jax.device_get(tokens)]
```

:begin_tab:`pytorch`
`generate` conditions on the last `max_len` tokens: once the text is
longer than the model's context, the oldest tokens are cropped.
:end_tab:

:begin_tab:`jax`
Feeding the jitted forward pass a context that grows by one token per step
would give it a new input shape, and so a new compilation, at every step.
`generate` instead keeps the context in a buffer of fixed shape
`(1, max_len)` and writes each sampled token into it. The causal mask hides
the positions not yet written from every query before them, so the logits
at position $t-1$ are exact. The sampling step `_gpt_sample_next`
therefore compiles once per model shape and value of `top_k`, and the
tokens stay on the device until the end. The drawback is that every step
runs the forward pass over all `max_len` positions of the buffer, however
short the text. Once the buffer is full, each step samples from the logits at
its last position, then drops the oldest token and writes the new one at
the end, so the model conditions on the last `max_len` tokens.
:end_tab:

We sample 120 bytes after the prefix "The history of the " at three
settings: the model's own distribution, a sharpened and truncated one, and
a flattened one.

```{.python .input #gpt-sampling-from-the-model-2}
%%tab pytorch, jax
prefix = data.vocab.encode('The history of the ')
for temperature, top_k in ((1.0, None), (0.7, 8), (2.0, None)):
    out = model.generate(prefix, 120, temperature=temperature, top_k=top_k)
    d2l.print_wrapped(f'T={temperature}, top_k={top_k}: '
                      + repr(data.vocab.decode(out)))
```

At $\tau = 1$ the model writes words and word-like strings in a plausible
order, with the habits of its corpus, which sets punctuation off by spaces
(" , " and " . "). The sharpened and truncated setting keeps to frequent
words, most of them spelled correctly, and at $\tau = 2$ the bytes approach
random strings. A loss near 1.35 nats per byte is a perplexity of about 3.9:
at a typical position the model still hesitates among about four bytes, so
the temperature changes the text visibly. Fluent text needs a much larger
model trained on much more text, such as the one we load next.

## Loading GPT-2
:label:`subsec_gpt-loading`

GPT-2 small is the `GPT` class with five settings
:cite:`Radford.Wu.Child.ea.2019`: 12 blocks of width 768 with 12 heads,
LayerNorm with biases, a GELU feed-forward network, learned positions for
up to 1,024 tokens, and pre-norm with a tied head. Loading it takes a
constructor call with these settings, the released weights, and the
released tokenizer. We register the weight and tokenizer files in
`d2l.DATA_HUB` with their content hashes, so that one download serves every
later run. The tokenizer and the loader both go into the library, as
`gpt2_tokenizer` and `load_gpt2`, because :numref:`sec_kv-cache` studies how
the released model uses its cache.

### The Tokenizer

Our `GPT` reads bytes, but GPT-2 reads subword tokens, so loading it starts
with its tokenizer. The cell registers the three released files and builds
the tokenizer from the two that describe the vocabulary:

```{.python .input #gpt-loading-gpt-2-1}
%%tab pytorch, jax
#@save
GPT2_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/'
d2l.DATA_HUB['gpt2-weights'] = (
    GPT2_URL + 'model.safetensors',
    '89a76996d7c6ee89b86618a265483aab73e61d50')
d2l.DATA_HUB['gpt2-merges'] = (
    GPT2_URL + 'merges.txt', '396d4d8ec90cb02f4d56e049e0e4add868bcd943')
d2l.DATA_HUB['gpt2-encoder'] = (
    GPT2_URL + 'vocab.json', 'f0223209235343bc067d7da838328bced8085ae1')

def gpt2_tokenizer():  #@save
    """GPT-2's byte-level BPE tokenizer, assembled by tiktoken."""
    import tiktoken  # lazy: d2l itself does not require tiktoken
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks
    return tiktoken.Encoding(
        'gpt2', explicit_n_vocab=50257,
        pat_str=d2l.BPETokenizer.GPT2_PATTERN,
        mergeable_ranks=data_gym_to_mergeable_bpe_ranks(
            d2l.download('gpt2-merges', '../data/gpt2'),
            d2l.download('gpt2-encoder', '../data/gpt2')),
        special_tokens={'<|endoftext|>': 50256})

enc = gpt2_tokenizer()
ids = enc.encode('Attention is all you need.')
print(ids, [enc.decode([i]) for i in ids])
```

The tokenizer applies the BPE machinery of :numref:`sec_text-sequence` to
GPT-2's released merge list and vocabulary. It uses the same
pre-tokenization pattern as `d2l.BPETokenizer` and assembles the result
into `tiktoken`'s encoder. The construction needs two data files and a
regular expression, and no model library or configuration framework.

### The Weight Map

The checkpoint stores one tensor per parameter, under the names of GPT-2's
original code. The token table is `wte`, the position table `wpe`, and the
final normalization `ln_f`. Block $i$ is `h.i`. Inside it, `ln_1` and
`ln_2` are the normalizations before attention and before the feed-forward
network, `attn.c_attn` and `attn.c_proj` are our `W_qkv` and `W_o`, and
`mlp.c_fc` and `mlp.c_proj` are our `W_1` and `W_2`. Loading therefore
needs a dictionary of names and one rule about layout. GPT-2's code
implemented its linear layers as a class `Conv1D` that stores each weight
matrix with the input dimension first, as
$(\textrm{in}, \textrm{out})$. The fused `c_attn` matrix concatenates
queries, keys, and values in the same order as our `W_qkv`. The token
embedding doubles as the output head in both models, and our LayerNorm
uses GPT-2's $\epsilon = 10^{-5}$. Nothing else needs special handling.
The function `load_gpt2` builds the configuration, copies every tensor
into place, and returns the model in evaluation mode.

```{.python .input #gpt-loading-gpt-2-2}
%%tab pytorch
def load_gpt2():  #@save
    """GPT-2 small (124M) as a GPT with the released weights, in eval
    mode."""
    from safetensors.torch import load_file  # lazy, like tiktoken above
    gpt2 = GPT(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
               max_len=1024, pos='learned', norm='layer', act='gelu',
               pre_norm=True, bias=True)
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
                W = weights[key + '.weight']  # Conv1D: (in, out)
                module.weight.copy_(W.T if W.ndim == 2 else W)
                module.bias.copy_(weights[key + '.bias'])
    return gpt2.eval()

gpt2 = load_gpt2().to(d2l.try_gpu())
print(f'{d2l.num_params(gpt2) / 1e6:.1f}M parameters')
```

```{.python .input #gpt-loading-gpt-2-2}
%%tab jax
def load_gpt2():  #@save
    """GPT-2 small (124M) as a GPT with the released weights, in eval
    mode."""
    from safetensors.numpy import load_file  # lazy, like tiktoken above
    gpt2 = GPT(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
               max_len=1024, pos='learned', norm='layer', act='gelu',
               pre_norm=True, bias=True, rngs=nnx.Rngs(d2l.get_key()))
    weights = {k: jnp.asarray(v) for k, v in load_file(
        d2l.download('gpt2-weights', '../data/gpt2')).items()}
    gpt2.token_emb.embedding[...] = weights['wte.weight']
    gpt2.pos_emb.embedding[...] = weights['wpe.weight']
    gpt2.norm.scale[...] = weights['ln_f.weight']
    gpt2.norm.bias[...] = weights['ln_f.bias']
    for i, blk in enumerate(gpt2.blks):
        for norm, key in ((blk.norm1, f'h.{i}.ln_1'),
                          (blk.norm2, f'h.{i}.ln_2')):
            norm.scale[...] = weights[key + '.weight']
            norm.bias[...] = weights[key + '.bias']
        for linear, key in ((blk.attention.W_qkv, f'h.{i}.attn.c_attn'),
                            (blk.attention.W_o, f'h.{i}.attn.c_proj'),
                            (blk.ffn.W_1, f'h.{i}.mlp.c_fc'),
                            (blk.ffn.W_2, f'h.{i}.mlp.c_proj')):
            # Conv1D stores (in, out), exactly nnx.Linear's kernel layout
            linear.kernel[...] = weights[key + '.weight']
            linear.bias[...] = weights[key + '.bias']
    gpt2.eval()
    return gpt2

gpt2 = load_gpt2()
print(f'{d2l.num_params(gpt2) / 1e6:.1f}M parameters')
```

:begin_tab:`pytorch`
`nn.Linear` stores its weight as $(\textrm{out}, \textrm{in})$, the
transpose of the `Conv1D` layout, so every 2-D weight is transposed on the
way in. The copies run under `torch.no_grad`, because writing into a
parameter in place is not an operation that autograd should record.
:end_tab:

:begin_tab:`jax`
`nnx.Linear` stores its kernel as $(\textrm{in}, \textrm{out})$, the
`Conv1D` layout, so the weights load unchanged. `safetensors.numpy` opens
the file as NumPy arrays, which become JAX arrays before they are written
into the parameters.
:end_tab:

### Checks of the Loaded Model

Shape checks alone do not validate a loaded checkpoint, because a
transposed square matrix still multiplies. Two further checks test what
the model computes. The first is quantitative: the model should assign a
sentence of English a plausible probability.

```{.python .input #gpt-loading-gpt-2-3}
%%tab pytorch
text = ("The Time Machine, by H. G. Wells. The Time Traveller was "
        "expounding a recondite matter to us.")
x = torch.tensor(enc.encode(text), device=d2l.try_gpu())[None]
with torch.inference_mode():
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

A perplexity around 50 means that the model is, on average, about as
uncertain as a uniform choice among 50 tokens, where uniform guessing over
the vocabulary would score 50,257. An exercise measures how much the
common mapping mistakes raise it. The second check is qualitative: greedy
decoding ($k = 1$) of a familiar prompt should continue it coherently.

```{.python .input #gpt-loading-gpt-2-4}
%%tab pytorch, jax
out = gpt2.generate(enc.encode('Alan Turing theorized that computers '
                               'would one day become'), 16, top_k=1)
d2l.print_wrapped(enc.decode(out))
```

Sampling at $\tau = 0.8$ with $k = 50$ should give varied but grammatical
text:

```{.python .input #gpt-loading-gpt-2-5}
%%tab pytorch, jax
out = gpt2.generate(enc.encode('The secret of a good deep learning '
                               'textbook is'), 40, temperature=0.8,
                    top_k=50)
d2l.print_wrapped(enc.decode(out))
```

The continuations are fluent English, unlike the samples of our own model.
GPT-2 is the same class with about 26 times as many parameters, trained
with tens of thousands of times as much computation on thousands of
times as much text. The rest of this chapter keeps the class and changes
what surrounds it: :numref:`sec_kv-cache` makes generation fast, later
sections add encoders, cross-attention, and mixtures of experts, and
:numref:`sec_scaling-laws` asks how loss falls as parameters, data, and
computation grow together.

## Summary

A GPT predicts every next token of a window in one forward pass. A token
embedding feeds a stack of transformer blocks whose attention is causally
masked, and after a final normalization the same embedding table, used as
the output head, produces the logits. Positions enter as a learned table
added to the embedding, as in GPT-2, or as RoPE rotations inside each
attention layer. GPT-2's initialization scales the projections that write
into the residual stream by $1/\sqrt{2L}$, so that the stream does not grow
with depth. We train with AdamW, weight decay on the matrices, warmup,
cosine decay, and clipping, on a corpus so large that the model almost never
reads the same passage twice, so its validation loss follows its training
loss. Moving the normalization after the residual addition, at a learning
rate that pre-norm tolerates without warmup, leaves the model at the unigram
entropy, where its predictions ignore their context. A slow warmup lets the
post-norm model train. Temperature and top-$k$ truncation reshape the
distribution from which text is sampled, and the released GPT-2 loads into
the same class through a map of parameter names.

## Exercises

1. **Parameter count.** The loading code of :numref:`subsec_gpt-loading`
   reports 124.4M parameters for the GPT-2 configuration, which has width
   $d = 768$, 12 blocks, a vocabulary of $V = 50{,}257$ tokens, and 1,024
   positions.
    1. Count the parameters of one block exactly, including the biases of
       its four linear layers and the gains and shifts of its two
       LayerNorms. Add the token table, the position table, and the final
       normalization, and verify that the total is $124{,}439{,}808$.
    1. Exercise :exref:`ex_transformer-block-ffn-width-rounding` of
       :numref:`sec_transformer-block` gives the exact parameter count of
       the default `TransformerBlock`, which `GPT` builds with its default
       flags. Use it to count the parameters of `GPT(vocab_size=256)`, with
       $d = 256$ and six blocks, and check the total against the count
       printed in :numref:`subsec_gpt-blocks-to-lm`. What fraction of the
       parameters sits in the embedding tables of each of the two models?
    1. Without tying, the output head would be a separate $V \times d$
       matrix. How many parameters would GPT-2 then have? A product of a
       $d_\textrm{out} \times d_\textrm{in}$ matrix with a vector costs
       about $2 d_\textrm{out} d_\textrm{in}$ floating-point operations,
       and a table lookup costs none. Estimate the share of GPT-2's
       per-token matrix work that the head accounts for at short contexts,
       where the attention scores are negligible, and compare it with the
       share of the parameters that the shared table holds.

1. [code] **Unmasked attention.** In `GPT`, the flag `is_causal=True` of
   the attention call in `CausalAttention` is the only mechanism that keeps
   position $t$ from attending to later positions.
    1. Take one validation window of 128 bytes and replace its byte at
       position 64 by a different one. For every position,
       compute the largest absolute change of the logits of the trained
       `model`, and plot the change against position.
    1. Subclass `GPT` with a nested `CausalAttention` whose attention call
       passes `is_causal=False`. Train it as in
       :numref:`subsec_gpt-training`, with `LMOptimizer` set up for 1,000
       steps, but stop after 200 steps. Compare its training and
       validation losses with those that the training of `model` recorded
       at step 200, and explain the numbers.
    1. Sample 100 bytes from the unmasked model with the prefix
       `'The history of the '`. Explain why the validation loss of the
       previous sub-problem and this sample judge the unmasked model so
       differently.

1. [code] **The recipe, one part at a time.** `LMOptimizer` combines four
   choices: the warmup, the cosine decay, the weight decay on the
   matrices, and the clipping.
    1. Train `GPT(len(data.vocab))` as in :numref:`subsec_gpt-training`
       five times: with the full recipe, and with each of the four choices
       removed in turn. Removing the decay of the learning rate means a
       constant rate after the warmup; removing the warmup means starting
       at the peak rate. Repeat each run three times and report the mean
       and the range of the final validation losses. Which removals change
       the loss by more than the run-to-run range?
    1. Repeat at three times the peak learning rate. Which of the four
       choices matter now, and why would you expect clipping and warmup to
       matter more at a high learning rate than at a low one?

1. [code] **How slow a warmup post-norm needs.** The third run of
   :numref:`subsec_gpt-norm-placement` warms the post-norm model up over
   400 of its 800 steps.
    1. Repeat it with warmups of 100, 200, and 300 steps at the peak rate
       $10^{-3}$, and with a warmup of 100 steps at $5 \times 10^{-4}$.
       Which runs leave the unigram level? Run each setting three times.
    1. Do your results depend on the length of the warmup, on the peak
       rate, or on the slope at which the rate rises? Design one more run
       that separates two of these explanations.
    1. Track the token spread of :numref:`subsec_signal-propagation` at
       the last block during the first 100 steps of a run that stalls and
       of one that trains. Does the stalled run lose the distinction
       between its tokens, and if so, when?

1. [code] **Memorizing a novel.** :numref:`subsec_gpt-training` trains on
   WikiText-103 so that the model almost never reads the same passage
   twice. Here it reads one short novel many times.
    1. Train a `GPT` with the same recipe on
       `d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char',
       num_train=100000, num_val=3000)`, whose 100,000 overlapping
       training windows start at the first 100,000 positions of the text
       and whose 28 tokens come from the character pipeline of
       :numref:`sec_text-sequence`, so that the model has a table of 28
       rows. Record both losses every 50 steps. At which step does the
       validation loss reach its minimum, and how far apart are the two
       losses at the end?
    1. A sample that copies its training text contains long strings that
       occur in that text verbatim. Draw ten samples of 100 tokens at
       temperature 1 without truncation, from this model with the prefix
       `'the time traveller '` and from `model` with the prefix
       `'The history of the '`. Compute the *verbatim fraction* of each
       sample, the fraction of its 20-token windows that occur verbatim in
       the corresponding training text.
    1. Retrain on *The Time Machine* with `dropout=0.3`. Does dropout move
       the minimum of the validation loss, lower it, or only slow the rise
       after it?

1. [code] **Checkpoint mismatches.** The GPT-2 checkpoint holds 160 tensors
   with 137,022,720 numbers in all, while `gpt2` has 124,439,808
   parameters.
    1. Identify the checkpoint tensors that `load_gpt2` never reads, and
       explain why `GPT` needs no counterpart for them.
    1. Construct the GPT-2 configuration with `norm='rms'` and run the
       body of `load_gpt2` on it. Which line fails, and why can no mapping
       of the checkpoint onto this configuration reproduce GPT-2? Then
       list, for the GPT-2 configuration with `pos='rope'` and
       `act='swiglu'`, every checkpoint tensor without a destination and
       every model parameter without a source.
    1. Some mistakes load without any error. Introduce three of them, one
       at a time: skip every copy of a normalization shift and load the
       rest into the `norm='rms'` model; load the `attn.c_proj` matrix
       of every block in the wrong orientation, which no shape check
       detects because the matrix is square; and swap the query and key
       thirds of every `c_attn` weight and bias. For each mistake, report
       the perplexity of the sentence that :numref:`subsec_gpt-loading`
       scores and the greedy continuation of the prompt
       `'Alan Turing theorized that computers would one day become'`.
       Which mistake does the perplexity check flag most clearly, and which
       least?
    1. A check that catches every such mistake compares the logits with an
       independent implementation. Write GPT-2's forward pass in NumPy
       directly from the checkpoint tensors, in float64 and with the
       `Conv1D` convention $\mathbf{x}\mathbf{W} + \mathbf{b}$, and
       compare its logits for a short prompt with those of `gpt2`. How
       large is the largest difference relative to the largest logit, and
       what does that ratio say about where the difference comes from?

1. [code] **Tied head without blocks.** ● Consider `GPT` with
   `num_blks=0` and its defaults `pos='rope'` and `norm='rms'`. Let
   $\mathbf{e}_1, \ldots, \mathbf{e}_V \in \mathbb{R}^d$ be the rows of
   the token table, let $\boldsymbol{\gamma}$ be the gain of the final
   RMSNorm :eqref:`eq_rmsnorm`, and let $z_{ij}$ be the logit of token $j$
   when the current token is $i$.
    1. Show that $z_{ij} = s_i M_{ij}$ with scalars $s_i > 0$ and a
       symmetric matrix $\mathbf{M}$, and express $s_i$ and $\mathbf{M}$
       in terms of the table and $\boldsymbol{\gamma}$.
    1. For three distinct tokens $i$, $j$, and $k$, show that

       $$\frac{z_{ij} - z_{ik}}{s_i} + \frac{z_{jk} - z_{ji}}{s_j} + \frac{z_{ki} - z_{kj}}{s_k} = 0.$$

       Conclude that on the periodic text $ijk\,ijk\,ijk \ldots$ the model
       assigns the correct next token a probability of at most $1/2$ in at
       least one of the three contexts, so that its average loss exceeds
       $(\log 2)/3$ nats per token. Which loss can a model with an untied
       head reach on this text?
    1. Show that losses arbitrarily close to $(\log 2)/3$ are attainable.
    1. Train `GPT(len(data.vocab), num_blks=0)` for 1,000 steps with
       `LMOptimizer` at a peak learning rate of $10^{-2}$, and compare its
       final training loss with the bigram level of
       :numref:`subsec_gpt-training`, using the first sub-problem to
       explain the difference.

1. [code] **Bits per byte against GPT-2.** ● GPT-2 and `model` use
   different tokens, so their per-token losses are not comparable. A loss
   of $\ell$ nats per token at $c$ bytes per token corresponds to

   $$\frac{\ell}{c \log 2} \textrm{ bits per byte}.$$

    1. Compute the bits per byte of `model` on the validation split of
       `data`.
    1. Compute the bits per byte of `gpt2` on the same text, in chunks of
       1,024 tokens. WikiText keeps the tokenization of its creators: it
       sets punctuation off by spaces and writes a hyphen inside a word as
       ` @-@ `. Repeat on the text with these marks undone, still dividing
       by the bytes of the original text, as the GPT-2 paper did with
       invertible detokenizers.
    1. Which of the two GPT-2 figures should be compared with that of
       `model`? Discuss what each side brings to the passage: `model` was
       trained on text of the same source and format, sees 128 bytes of
       context, and predicts 256 symbols, whereas GPT-2 was trained on 40 GB
       of web text from which Wikipedia articles were removed, sees up to
       1,024 tokens, and predicts 50,257 symbols.
   :exlabel:`ex_gpt-bits-per-byte`

1. [extended] **Subwords at small scale.** ●
   :numref:`subsec_gpt-training` trains on bytes, which keeps the
   embedding table small and makes the loss a number of bits per byte.
   Subword tokens have other advantages: a window of 128 tokens
   covers several times more text, and each prediction carries more
   information. Test which effect wins at this scale.
    1. Train a `d2l.BPETokenizer` with the GPT-2 pattern on the first
       million bytes of the training text for vocabulary sizes 1,024 and
       4,096; with 256 it learns no merges and gives the bytes of `data`.
       Subclass `d2l.WikiText103` so that it encodes the first 20 million
       bytes of the training split and the whole validation split with a
       given tokenizer.
    1. Train `GPT` with each vocabulary as in
       :numref:`subsec_gpt-training` and convert the final validation
       losses to bits per byte as in :exref:`ex_gpt-bits-per-byte`. Count
       also the parameters in the embedding table of each model.
    1. A larger vocabulary gives each window more bytes of context and
       each step more text, so in 1,000 steps the model reads a larger
       share of its 20 million bytes. Repeat the best vocabulary with
       windows of about 128 bytes, and with the number of steps that reads
       as many bytes as the byte-level run. Which of the three effects
       explains the differences you found?

[Discussions](https://d2l.discourse.group/t/36775)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.2]{.kicker}

A GPT from Scratch<br>
**embedding, causal blocks, and a tied head · GPT-2 initialization · a training recipe · normalization placement · sampling · loading GPT-2**
:::
:::

::: {.slide title="One Window of 128 Bytes Holds 128 Prediction Problems"}
[The problem]{.kicker}

| context | target |
|:--|:--|
| `the European lobste` | `r` |
| `the European lobster` | `␣` |
| `the European lobster␣` | `o` |

- Train on all positions of a window in **one** forward pass.
- Two requirements: position $t$ must not see tokens after $t$, and the
  **same weights** must serve every position and every context length.

::: {.d2l-note .warn}
How do we turn the transformer block into a model that meets both?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- assemble a GPT from an embedding, causal blocks, and a tied head;
- count its parameters ($12d^2$ per block) and its training cost ($6ND$);
- initialize it as GPT-2 does, and say why the residual projections are
  scaled by $1/\sqrt{2L}$;
- train it with the recipe of the optimization chapter and read the two
  loss curves against count-based baselines;
- sample with temperature and top-$k$, and load the released GPT-2.

**Running example:** GPT-2 small (124M parameters), and our 4.8M-parameter
byte-level model.
:::

::: {.slide title="Blocks Become a Language Model"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN on a residual stream |
| **A GPT from Scratch** | **a trained language model; GPT-2 loaded** |
| Generation and the KV Cache | reuse keys and values while sampling |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Mixture of Experts | more parameters per FLOP |
| Scaling Laws | loss against parameters, data, compute |
:::

::: {.col .narrow}
- **Uses:** the block's defaults (pre-norm, RMSNorm, SwiGLU), RoPE,
  AdamW and schedules, byte tokens, perplexity.
- **Skips:** fast generation (next section), training at scale.
:::
:::
:::

::: {.slide title="A GPT Is an Embedding, L Blocks, a Norm, and the Same Table Again"}
::: {.cols .vc}
::: {.col .fig}
@fig:mdl-transformers-gpt-stack
:::

::: {.col}
- Logits $= \mathbf{E}\,\mathrm{Norm}(\mathbf{h})$: one dot product per
  vocabulary entry.
- Tying reuses $\mathbf{E}$ ($V \times d$) as the head: GPT-2 saves
  38.6M parameters.
:::
:::
:::

::: {.slide title="A Causal Mask Lets One Pass Train Every Position"}
Query $t$ may attend to keys $1, \ldots, t$ only:

$$\begin{pmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{pmatrix}
\quad \textrm{(rows: queries, columns: keys)}$$

- The logits at position $t$ depend on $x_1, \ldots, x_t$ only, so the
  target $x_{t+1}$ never leaks.
- Fused kernels take the mask as a flag (`is_causal=True`) and never store
  the $n \times n$ score matrix.
:::

::: {.slide title="Positions Enter as a Table or as a Rotation"}
Attention alone treats its input as a set: permute the tokens, and the
outputs permute the same way.

- **`pos='learned'`** (GPT-2): add a trained vector $\mathbf{p}_t$ to the
  embedding of position $t$; the table caps the context at 1,024 tokens.
- **`pos='rope'`** (default): rotate each query and key by an angle
  proportional to its position; the score of $\mathbf{q}_i$ and
  $\mathbf{k}_j$ then depends on the offset $i - j$.
- RoPE lives inside the attention module; the model computes the angle
  tables once per forward pass and hands them to every block.
:::

::: {.slide title="The Forward Pass Fits in Ten Lines" only="pytorch"}
```{.python #gpt-exercises-1}
def forward(self, X):                   # X: (batch, T) token ids
    H, tables = self.token_emb(X), None # (batch, T, d)
    if self.pos == 'learned':
        H = H + self.pos_emb(torch.arange(X.shape[1], device=X.device))
    else:  # RoPE angles of positions 0, ..., T - 1, for every block
        tables = d2l.rope_tables(X.shape[1], self.head_width,
                                 device=X.device)
    for blk in self.blks:               # L blocks, causal attention
        H = blk(H, tables=tables)
    return F.linear(self.norm(H), self.token_emb.weight)  # tied
```

Output: (batch, T, V), the next-token logits at every position.
:::

::: {.slide title="The Forward Pass Fits in Nine Lines" only="jax"}
```{.python #gpt-exercises-2}
def __call__(self, X):                  # X: (batch, T) token ids
    H, tables = self.token_emb(X), None # (batch, T, d)
    if self.pos == 'learned':
        H = H + self.pos_emb(jnp.arange(X.shape[1]))
    else:  # RoPE angles of positions 0, ..., T - 1, for every block
        tables = d2l.rope_tables(X.shape[1], self.head_width)
    for blk in self.blks:               # L blocks, causal attention
        H = blk(H, tables=tables)
    return self.token_emb.attend(self.norm(H))  # tied head
```

Output: (batch, T, V), the next-token logits at every position.
:::

::: {.slide title="GPT-2's Initialization Keeps the Stream from Growing with Depth"}
- All matrices $\sim \mathcal{N}(0, 0.02^2)$, biases zero; the two that
  write into the stream, $\mathbf{W}_o$ and $\mathbf{W}_2$, get
  $0.02/\sqrt{2L}$.
- Pre-norm: each branch reads the normalized stream and adds a fixed
  amount, so the mean square of the stream grows $\propto 2L$.
- Scaling each writer by $1/\sqrt{2L}$: all $2L$ branches together
  $\approx$ one unscaled branch, whatever the depth.
- Measured from 8 to 16 to 32 blocks: unscaled, it doubles each time;
  scaled, it barely grows.

| model | branches $2L$ | std of $\mathbf{W}_o$, $\mathbf{W}_2$ |
|:--|--:|--:|
| ours, 6 blocks | 12 | 0.0058 |
| GPT-2 small, 12 blocks | 24 | 0.0041 |
:::

::: {.slide title="Each Block Holds 12d² Parameters"}
Attention: $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$ =
$4d^2$. FFN: $2 \cdot d \cdot 4d$ (GELU) or $3 \cdot d \cdot 8d/3$
(SwiGLU) = $8d^2$.

| | $d$ | blocks | $12d^2 \times$ blocks | token table | total |
|:--|--:|--:|--:|--:|--:|
| ours | 256 | 6 | 4.7M | 0.066M | 4.79M |
| GPT-2 small | 768 | 12 | 85M | 38.6M | 124.4M |

GPT-2 adds 0.8M for 1,024 learned positions. Untied, it would have 163M.
:::

::: {.slide title="Check: How Large Is GPT-2 Medium?"}
[Check]{.kicker}

GPT-2 medium: $d = 1024$, 24 blocks, $V = 50{,}257$, 1,024 positions,
tied head. Estimate its parameter count.

. . .

- Blocks: $12 \cdot 1024^2 \cdot 24 \approx 302$M.
- Token table: $50{,}257 \cdot 1024 \approx 51.5$M; positions: 1.0M.
- Total $\approx 355$M, the reported size.
:::

::: {.slide title="A 4.8M-Parameter Model Needs More Than One Novel"}
1,000 steps $\times$ 64 windows $\times$ 128 bytes = 8.2M predictions.

| corpus | bytes for training | passes in 1,000 steps |
|:--|--:|--:|
| *The Time Machine* | 0.17M | about 47 |
| WikiText-103 (Wikipedia articles) | 270M | about 0.03 |

- 256 symbols: the UTF-8 bytes of raw text, with case, digits, punctuation.
- Training windows at random offsets; validation: the official split,
  1.1M bytes.
- Bytes keep the table at 65,536 parameters, and nats per byte
  $/\log 2$ = bits per byte.
:::

::: {.slide title="The Optimization Chapter's Recipe in Four Parts"}
$$\eta_t = \eta \cdot \begin{cases} t/t_w, & t < t_w \\ \tfrac12\left(1 + \cos\left(\pi \frac{t - t_w}{T - t_w}\right)\right), & t \ge t_w \end{cases}$$

- **AdamW**, peak $\eta = 2 \times 10^{-3}$.
- **Weight decay** $0.1$ on the weight matrices only; embeddings, gains,
  biases exempt.
- **Warmup** over $t_w = 100$ steps, then **cosine** to zero at
  $T = 1{,}000$.
- **Clipping** at global norm 1. No dropout.
:::

::: {.slide title="Training and Validation Loss Fall Together"}
@!gpt-training-the-modern-configuration

Almost every window is new text: little to memorize, so the validation loss
tracks the training loss.
:::

::: {.slide title="Count Models Set the Reference Levels"}
@!gpt-training-the-model-4

- **Unigram**: byte frequencies, no context.
- **Bigram**: the previous byte only.
- The GPT ends far below both: it uses much more than the previous
  byte.
:::

::: {.slide title="Training Costs About 6ND Operations"}
Forward: $2N$ per token (a multiply and an add per parameter); backward:
about twice that.

| run | $N$ | $D$ (tokens) | $6ND$ |
|:--|--:|--:|--:|
| ours | 4.8M | 8.2M | $2 \times 10^{14}$ |
| GPT-2 small | 124M | about $10^{10}$ | $7 \times 10^{18}$ |
| GPT-2 XL | 1.5B | about $10^{10}$ | $10^{20}$ |
| largest current runs | | | $> 10^{25}$ |

The model definition hardly changes across 11 orders of magnitude.
:::

::: {.slide title="Post-Norm Without Warmup Sticks at the Unigram Level"}
Plain AdamW at a constant $10^{-3}$ for 800 steps, pre-norm and
post-norm; then post-norm with a 400-step warmup:

@!gpt-breaking-it-with-one-flag

A loss at the unigram entropy means predictions that ignore the input,
even the current byte.
:::

::: {.slide title="Check: Is Starved Attention the Whole Story?"}
[Check]{.kicker}

At initialization, post-norm gives the query and key projections tiny
gradients. Suppose attention learned nothing at all. Which loss could the
post-norm model still reach?

. . .

- The FFNs act on the current byte, so the bigram level is within
  reach.
- The plateau sits at the unigram level instead: starved attention is at
  most part of the explanation.
- Xiong et al. (2020): post-norm's gradients near the output are large
  at initialization; with a warmup, the third run trains.
:::

::: {.slide title="Temperature Sharpens or Flattens; Top-k Cuts the Tail"}
$$p_v = \frac{\exp(z_v/\tau)}{\sum_u \exp(z_u/\tau)}$$

| logits $(2, 1, 0)$ | $p$ |
|:--|:--|
| $\tau = 0.5$ | $(0.87, 0.12, 0.02)$ |
| $\tau = 1$ | $(0.67, 0.24, 0.09)$ |
| $\tau = 2$ | $(0.51, 0.31, 0.19)$ |

- $\tau \to 0$: greedy decoding. Top-$k$: keep the $k$ largest logits,
  set the rest to $-\infty$.
:::

::: {.slide title="Lower Temperature Plays Safe, Higher Temperature Scrambles"}
@!gpt-sampling-from-the-model-2

$\tau = 1$: the model's own distribution. $\tau = 0.7$, $k = 8$: safer,
frequent words, mostly spelled correctly. $\tau = 2$: close to random bytes.
:::

::: {.slide title="Naive Generation Redoes the Whole Context for Every Token" only="pytorch"}
`generate` runs the full forward pass for each new token.

- Token $t$ costs a pass over $t$ positions; $T$ tokens cost
  $1 + 2 + \cdots + T = T(T+1)/2$ position computations.
- $T = 1{,}024$: 524,800 instead of 1,024, about 500 times the work.
- The keys and values of earlier positions never change. Caching them is
  the next section.
:::

::: {.slide title="Naive Generation Redoes the Whole Context for Every Token" only="jax"}
`generate` runs the full forward pass for each new token.

- A buffer of fixed shape $(1, 1{,}024)$: one compilation, but every token
  costs a pass over all 1,024 positions, however short the text.
- $T$ tokens cost $1{,}024\,T$ position computations, where computing
  each position once would cost $T$.
- The keys and values of earlier positions never change. Caching them is
  the next section.
:::

::: {.slide title="GPT-2 Small Is Our Class with Five Settings"}
```{.python #gpt-exercises-3}
gpt2 = GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
           num_blks=12, max_len=1024, pos='learned', norm='layer',
           act='gelu', pre_norm=True, bias=True)
```

- Weights: one safetensors file from the Hugging Face Hub, pinned by its
  SHA-1 hash.
- Tokenizer: GPT-2's merge list and vocabulary with the BPE pattern of
  the tokenization section, assembled by `tiktoken`.
- Saved as `d2l.load_gpt2()` and `d2l.gpt2_tokenizer()`; the KV-cache
  section studies the released model with them.
:::

::: {.slide title="A Name Map and One Layout Rule Load the Checkpoint" only="pytorch"}
GPT-2's `Conv1D` stores weights as (in, out); `nn.Linear` as (out, in):

```{.python #gpt-exercises-4}
for i, blk in enumerate(gpt2.blks):
    modules = {f'h.{i}.ln_1': blk.norm1, f'h.{i}.ln_2': blk.norm2,
               f'h.{i}.attn.c_attn': blk.attention.W_qkv,
               f'h.{i}.attn.c_proj': blk.attention.W_o,
               f'h.{i}.mlp.c_fc': blk.ffn.W_1,
               f'h.{i}.mlp.c_proj': blk.ffn.W_2}
    for key, module in modules.items():
        W = weights[key + '.weight']
        module.weight.copy_(W.T if W.ndim == 2 else W)
        module.bias.copy_(weights[key + '.bias'])
```
:::

::: {.slide title="A Name Map and One Layout Rule Load the Checkpoint" only="jax"}
GPT-2's `Conv1D` stores weights as (in, out), the layout of an
`nnx.Linear` kernel:

```{.python #gpt-exercises-5}
for i, blk in enumerate(gpt2.blks):
    for linear, key in ((blk.attention.W_qkv, f'h.{i}.attn.c_attn'),
                        (blk.attention.W_o, f'h.{i}.attn.c_proj'),
                        (blk.ffn.W_1, f'h.{i}.mlp.c_fc'),
                        (blk.ffn.W_2, f'h.{i}.mlp.c_proj')):
        linear.kernel[...] = weights[key + '.weight']
        linear.bias[...] = weights[key + '.bias']
```

The two LayerNorms per block load the same way (`scale`, `bias`).
:::

::: {.slide title="Perplexity Near 50 and a Coherent Continuation Validate the Load"}
@!gpt-loading-gpt-2-3

@!gpt-loading-gpt-2-4

Uniform guessing over 50,257 tokens scores 50,257. Greedy decoding
continues the prompt coherently.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- GPT = embedding $\mathbf{E}$ + $L$ causal blocks + final norm +
  $\mathbf{E}$ again; positions as a table or a rotation.
- $12d^2$ per block: 4.7M for ours, 85M of GPT-2 small's 124M; training
  costs about $6ND$.
- GPT-2 initialization: $0.02$, and $0.02/\sqrt{2L}$ on the stream
  writers.
- On text it has almost never read before, validation follows training;
  post-norm without warmup stalls at the unigram level.
- Temperature and top-$k$ reshape sampling; GPT-2 loads with a name map.

**Next:** generation recomputes the whole context for every token; the KV
cache removes that.
:::
