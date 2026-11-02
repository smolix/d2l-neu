# A GPT from Scratch
:label:`sec_gpt`

A language model trained on text solves the same prediction problem at
every position of every training sequence. Given "the time travelle", it
must predict "r"; given "the time traveller", it must predict a space. One
sequence of 128 characters therefore holds 128 training examples. A model
can learn from all of them at once if it meets two requirements. Each
position must predict its successor without reading it or any later token.
And the same parameters must serve every position and every length of
context, from one token to the whole window.

The transformer block of :numref:`sec_transformer-block` meets the second
requirement already: it applies the same weights at every position. A
causal mask inside attention meets the first. Adding the remaining parts,
an embedding, positions, and an output head, gives a GPT (*generative
pretrained transformer*), which we write as a `GPT` class. We train it on 5
million characters of English, test where the normalization belongs, and
sample text from it. Finally, we load the released 124-million-parameter
GPT-2 :cite:`Radford.Wu.Child.ea.2019` into the same class, which the rest
of this chapter reuses. Its sampling loop recomputes the whole context for
every new token, a waste that the next section removes.

```{.python .input #gpt-a-gpt-from-scratch}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import numpy as np
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
import math
import numpy as np
import optax
from safetensors import numpy as safetensors_numpy
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
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
$t$. In :numref:`sec_transformer-block` the character model passed this
lower-triangular mask to every block as a boolean array. Here the attention
module applies it itself, because the fused attention kernels of
:numref:`sec_attention-at-scale` take the causal mask as a flag and never
store the $n \times n$ matrix of scores. With the mask, one forward pass
over a window computes all its next-token predictions at once, each from
its own prefix.

**Positions.** Attention alone treats its inputs as a set: permuting the
tokens permutes the outputs in the same way
(:numref:`sec_positional-information`). The `pos` argument selects one of
the two schemes of that section. `'learned'` adds a trained vector per
position to the embedding, as GPT-2 does, which limits the context to the
length of the table. `'rope'` rotates the queries and keys of every head by
an angle proportional to their position, so that the score of a query and
a key depends on their offset. The attention module calls `d2l.rope` where
it computes the queries and keys, and no other part of the model sees
positions.

**The blocks.** `d2l.TransformerBlock` is used unchanged, with its defaults
of :numref:`sec_transformer-block`: pre-norm, RMSNorm, and a SwiGLU
feed-forward network. The causal attention enters through its
`attn_factory` argument. Its option `num_kv_heads` lets groups of query
heads share one key--value head, which later sections use to make
generation cheaper.

The class initializes its weights as GPT-2 does. Every weight matrix is
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

```{.python .input #gpt-from-blocks-to-a-language-model-1}
%%tab pytorch
class GPT(nn.Module):  #@save
    """Decoder-only transformer language model built from configurable
    blocks."""

    class CausalAttention(nn.Module):
        """Fused causal self-attention, optionally rotary; num_kv_heads <
        num_heads shares each key-value head across a group of queries."""
        def __init__(self, num_hiddens, num_heads, bias=False, rope=False,
                     num_kv_heads=None):
            super().__init__()
            self.num_heads, self.rope = num_heads, rope
            self.num_kv_heads = num_kv_heads or num_heads
            self.kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
            self.W_qkv = nn.Linear(num_hiddens,
                                   num_hiddens + 2 * self.kv_hiddens,
                                   bias=bias)
            self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

        def qkv(self, X, offset=0):
            """Queries, keys, and values of X, each (batch, heads, length,
            head width), at positions offset, offset + 1, ..."""
            B, T, D = X.shape
            q, k, v = self.W_qkv(X).split(
                [D, self.kv_hiddens, self.kv_hiddens], -1)
            q = q.reshape(B, T, self.num_heads, -1).transpose(1, 2)
            k, v = (u.reshape(B, T, self.num_kv_heads, -1).transpose(1, 2)
                    for u in (k, v))
            if self.rope:
                q, k = d2l.rope(q, offset), d2l.rope(k, offset)
            return q, k, v

        def forward(self, X, *_):  # X serves as queries, keys, values
            q, k, v = self.qkv(X)
            Y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                enable_gqa=self.num_kv_heads < self.num_heads)
            return self.W_o(Y.transpose(1, 2).flatten(2))

    def __init__(self, vocab_size, num_hiddens=256, num_heads=8, num_blks=6,
                 max_len=1024, pos='rope', norm='rms', act='swiglu',
                 pre_norm=True, bias=False, dropout=0, num_kv_heads=None,
                 ffn_factory=None):
        super().__init__()
        assert pos in ('rope', 'learned'), f'unknown pos: {pos!r}'
        self.pos, self.max_len = pos, max_len
        self.token_emb = nn.Embedding(vocab_size, num_hiddens)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pos == 'learned':
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        attn = lambda: self.CausalAttention(num_hiddens, num_heads, bias,
                                            pos == 'rope', num_kv_heads)
        self.blks = nn.ModuleList([
            d2l.TransformerBlock(num_hiddens, num_heads, dropout, norm, act,
                                 pre_norm, bias, attn_factory=attn,
                                 ffn_factory=ffn_factory)
            for _ in range(num_blks)])
        self.norm = (nn.RMSNorm if norm == 'rms'
                     else nn.LayerNorm)(num_hiddens, eps=1e-5)
        for name, m in self.named_modules():  # GPT-2's initialization
            if isinstance(m, nn.Linear):  # W_o and W_2 write to the stream
                std = 0.02 / (2 * num_blks) ** 0.5 if name.endswith(
                    ('W_o', 'W_2')) else 0.02
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        """Fused causal self-attention, optionally rotary; num_kv_heads <
        num_heads shares each key-value head across a group of queries."""
        def __init__(self, num_hiddens, num_heads, bias=False, rope=False,
                     num_kv_heads=None, *, rngs):
            self.num_heads, self.rope = num_heads, rope
            self.num_kv_heads = num_kv_heads or num_heads
            self.kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
            self.W_qkv = d2l.Linear(num_hiddens,
                                    num_hiddens + 2 * self.kv_hiddens,
                                    use_bias=bias, rngs=rngs)
            self.W_o = d2l.Linear(num_hiddens, num_hiddens, use_bias=bias,
                                  rngs=rngs)

        def qkv(self, X, offset=0):
            """Queries, keys, and values of X, each (batch, heads, length,
            head width), at positions offset, offset + 1, ..."""
            B, T, D = X.shape
            q, k, v = jnp.split(self.W_qkv(X), [D, D + self.kv_hiddens],
                                axis=-1)
            q = q.reshape(B, T, self.num_heads, -1).swapaxes(1, 2)
            k, v = (u.reshape(B, T, self.num_kv_heads, -1).swapaxes(1, 2)
                    for u in (k, v))
            if self.rope:
                q, k = d2l.rope(q, offset), d2l.rope(k, offset)
            return q, k, v

        def __call__(self, X, *_):  # X serves as queries, keys, values
            q, k, v = self.qkv(X)
            # The fused kernel takes (batch, length, heads, head width)
            Y = jax.nn.dot_product_attention(
                q.swapaxes(1, 2), k.swapaxes(1, 2), v.swapaxes(1, 2),
                is_causal=True)
            return self.W_o(Y.reshape(Y.shape[0], Y.shape[1], -1))

    def __init__(self, vocab_size, num_hiddens=256, num_heads=8, num_blks=6,
                 max_len=1024, pos='rope', norm='rms', act='swiglu',
                 pre_norm=True, bias=False, dropout=0, num_kv_heads=None,
                 ffn_factory=None, *, rngs):
        assert pos in ('rope', 'learned'), f'unknown pos: {pos!r}'
        self.pos, self.max_len = pos, max_len
        init = nnx.initializers.normal(0.02)
        self.token_emb = nnx.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        if pos == 'learned':
            self.pos_emb = nnx.Embed(max_len, num_hiddens,
                                     embedding_init=init, rngs=rngs)
        attn = lambda rngs: self.CausalAttention(
            num_hiddens, num_heads, bias, pos == 'rope', num_kv_heads,
            rngs=rngs)
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, dropout, norm, act,
                                 pre_norm, bias, attn_factory=attn,
                                 ffn_factory=ffn_factory, rngs=rngs)
            for _ in range(num_blks)])
        self.norm = (nnx.RMSNorm if norm == 'rms'
                     else nnx.LayerNorm)(num_hiddens, epsilon=1e-5, rngs=rngs)
        for path, m in nnx.iter_graph(self):  # GPT-2's initialization
            if isinstance(m, nnx.Linear):  # W_o and W_2 write to the stream
                std = 0.02 / (2 * num_blks) ** 0.5 if path[-1] in (
                    'W_o', 'W_2') else 0.02
                m.kernel[...] = std * jax.random.normal(rngs.params(),
                                                        m.kernel.shape)
                if m.bias is not None:
                    m.bias[...] = jnp.zeros(m.bias.shape)

    def __call__(self, X):
        H = self.token_emb(X)
        if self.pos == 'learned':
            H = H + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = blk(H)
        return self.token_emb.attend(self.norm(H))
```

A shape check confirms one logit per vocabulary entry at every position,
and the standard deviations of two matrices of the first block confirm the
initialization:

```{.python .input #gpt-from-blocks-to-a-language-model-2}
%%tab pytorch
model = GPT(vocab_size=28)
X = torch.zeros(2, 16, dtype=torch.long)
d2l.check_shape(model(X), (2, 16, 28))
attention = model.blks[0].attention
print(f'{d2l.num_params(model) / 1e6:.2f}M parameters; initial std: '
      f'W_qkv {attention.W_qkv.weight.std():.4f}, '
      f'W_o {attention.W_o.weight.std():.4f}')
```

```{.python .input #gpt-from-blocks-to-a-language-model-2}
%%tab jax
model = GPT(vocab_size=28, rngs=nnx.Rngs(d2l.get_key()))
X = jnp.zeros((2, 16), dtype=jnp.int32)
d2l.check_shape(model(X), (2, 16, 28))
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
    net = GPT(vocab_size=28, num_blks=num_blks)
    if not scaled:  # Undo the depth scaling of W_o and W_2
        for name, p in net.named_parameters():
            if name.endswith(('W_o.weight', 'W_2.weight')):
                nn.init.normal_(p, std=0.02)
    with torch.no_grad():
        H = net.token_emb(torch.randint(0, 28, (8, 128)))
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
    net = GPT(vocab_size=28, num_blks=num_blks, rngs=nnx.Rngs(d2l.get_key()))
    if not scaled:  # Undo the depth scaling of W_o and W_2
        for path, m in nnx.iter_graph(net):
            if path and path[-1] in ('W_o', 'W_2'):
                m.kernel[...] = 0.02 * jax.random.normal(d2l.get_key(),
                                                         m.kernel.shape)
    H = net.token_emb(jax.random.randint(d2l.get_key(), (8, 128), 0, 28))
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
$d = 256$ and 4.7 million for six blocks, while the 28-row embedding adds
only 7,168. GPT-2 small has $d = 768$ and 12 blocks, so its blocks hold
$12 \times 12 \times 768^2 \approx 85$ million parameters and its token
table, $50{,}257 \times 768$, another 38.6 million. An untied head would
add a second table of that size and make the model about 31% larger.

## Training the Model
:label:`subsec_gpt-training`

### The Training Text

A model of 4.7 million parameters needs more text than one novel. The
training run below reads about 8 million characters. *The Time Machine*
has about 170,000, so the run would pass over the book about fifty times,
often enough for a model of this size to memorize much of it rather than
learn English (an exercise measures how much). We therefore append the
training text of the Penn Treebank, about 5 million characters of
newspaper text :cite:`Marcus.Santorini.Marcinkiewicz.1993`, and apply the
character pipeline of :numref:`sec_text-sequence` to both: the 26
lowercase letters, the space, and a token for unknown characters, 28 tokens
in all. The windows of 128 characters do not overlap, because a window
starting at every position, as `d2l.TimeMachine` builds them, would make 5
million rows. The last 2,000 windows, about a quarter of a million
characters, are held out for validation.

```{.python .input #gpt-training-the-model-1}
%%tab pytorch, jax
class TimeMachinePTB(d2l.TimeMachine):  #@save
    """Characters of The Time Machine and of the Penn Treebank training text,
    in non-overlapping windows; the last num_val windows are for validation."""
    def _download(self):
        with open(d2l.download_extract('ptb') + '/ptb.train.txt') as f:
            return super()._download() + ' ' + f.read()

    def __init__(self, batch_size, num_steps, num_val=2000):
        d2l.DataModule.__init__(self)
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        n = (len(corpus) - 1) // num_steps * num_steps
        self.X = d2l.tensor(corpus[:n]).reshape(-1, num_steps)
        self.Y = d2l.tensor(corpus[1:n + 1]).reshape(-1, num_steps)
        self.num_train = len(self.X) - num_val

data = TimeMachinePTB(batch_size=64, num_steps=128)
print(f'{data.num_train * 128 / 1e6:.2f}M training characters, '
      f'vocabulary of {len(data.vocab)}')
```

We keep characters rather than the subword tokens of
:numref:`sec_text-sequence` for two reasons. The 28-entry table puts only
7,168 of the model's parameters into embeddings, so nearly all of them sit
in the blocks; GPT-2's vocabulary of 50,257 tokens would put 12.9 million
into the table at width 256, almost three times the rest of the model. And
because each preprocessed character is one byte, the loss in nats per
character divided by $\log 2$ is the number of bits per byte of
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
optimizer, so the training loop `d2l.train_lm` of :numref:`sec_adam`, which
draws minibatches of windows and takes a fixed number of steps on the
next-character loss, uses it unchanged. We use no dropout: a model that
reads its training text about 1.7 times on average has little opportunity
to memorize.

```{.python .input #gpt-training-the-model-2}
%%tab pytorch
class LMOptimizer:  #@save
    """AdamW with weight decay on the weight matrices only, a linear warmup
    and a cosine decay of the learning rate, and gradient-norm clipping."""
    def __init__(self, model, lr, num_steps, warmup, weight_decay=0.1,
                 max_norm=1.0):
        named = list(model.named_parameters())
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
class LMOptimizer(nnx.Optimizer):  #@save
    """AdamW with weight decay on the weight matrices only, a linear warmup
    and a cosine decay of the learning rate, and gradient-norm clipping."""
    def __init__(self, model, lr, num_steps, warmup, weight_decay=0.1,
                 max_norm=1.0):
        decay = lambda params: jax.tree_util.tree_map_with_path(
            lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)
        schedule = lambda t: lr * d2l.cosine_warmup(t, num_steps, warmup)
        super().__init__(model, optax.chain(
            optax.clip_by_global_norm(max_norm),
            optax.adamw(schedule, weight_decay=weight_decay, mask=decay)),
            wrt=nnx.Param)
```

:begin_tab:`pytorch`
`LMOptimizer` keeps the two parameter groups of AdamW and a `LambdaLR`
scheduler that multiplies the base rate by `d2l.cosine_warmup`. Its `step`
clips, updates, and advances the schedule, in that order, so the three
always stay in step.
:end_tab:

:begin_tab:`jax`
`LMOptimizer` is an `nnx.Optimizer` whose transformation chains clipping in
front of `optax.adamw`. The schedule is passed as the learning rate, and the
`mask` marks the parameters that are decayed.
:end_tab:

### Loss Curves and Baselines

We train for 1,000 steps of 64 windows and record both losses every 50
steps. The training loss is the mean over those 50 steps. The validation
loss is the mean per-character cross-entropy on the held-out windows, which
is the logarithm of `d2l.lm_perplexity` (:numref:`sec_lstm`).

```{.python .input #gpt-training-the-model-3}
%%tab pytorch
model = GPT(len(data.vocab))
```

```{.python .input #gpt-training-the-model-3}
%%tab jax
model = GPT(len(data.vocab), rngs=nnx.Rngs(d2l.get_key()))
```

```{.python .input #gpt-training-the-modern-configuration}
%%tab pytorch, jax
num_steps, every = 1000, 50
optimizer = LMOptimizer(model, lr=2e-3, num_steps=num_steps, warmup=100)
steps, train_curve, val_curve = [], [], []
for step in range(every, num_steps + 1, every):
    losses = d2l.train_lm(model, data, optimizer, every)
    steps.append(step)
    train_curve.append(sum(losses) / every)
    val_curve.append(math.log(d2l.lm_perplexity(model, data)))
d2l.plot(steps, [train_curve, val_curve], 'step', 'loss (nats)',
         legend=['train', 'validation'], fmts=('-', '--'))
print(f'step {num_steps}: train {train_curve[-1]:.2f}, '
      f'validation {val_curve[-1]:.2f} nats per character')
```

The two curves fall together, and the validation loss never turns upward.
With fewer than two passes over the training text, the model has little
chance to memorize it, so what it learns carries over to the held-out end
of the Penn Treebank. Most of the decrease comes in the first few hundred
steps, and both losses end near 1.4 nats per character, or about 2 bits per
byte.

Two simple predictors give reference levels for these losses. The
*unigram* model predicts every character with its frequency in the training
text and ignores all context. Its cross-entropy on that text is the entropy
of the character frequencies. The *bigram* model predicts each character
from the previous one, with the conditional frequencies of the training
text.

```{.python .input #gpt-training-the-model-4}
%%tab pytorch, jax
text = np.asarray(data.X[:data.num_train]).ravel()  # The training text
V = len(data.vocab)
unigram = np.bincount(text, minlength=V) / len(text)
pairs = np.bincount(text[:-1] * V + text[1:], minlength=V * V).reshape(V, V)
bigram = pairs / np.maximum(pairs.sum(1, keepdims=True), 1)
p = unigram[unigram > 0]
print(f'unigram {-(p * np.log(p)).sum():.2f}, '
      f'bigram {-np.log(bigram[text[:-1], text[1:]]).mean():.2f} '
      f'nats per character')
```

A model that knows nothing about the text scores $\log 28 \approx 3.33$
nats, the loss at initialization. The unigram level is 2.85 nats and the
bigram level 2.33. The trained model ends nearly a nat below the bigram
level, so it uses much more of the context than the previous character.

### Training Compute

A rough count of operations places this run among larger ones. In the
forward pass, every parameter takes part in one multiplication and one
addition per token, so a model with $N$ parameters performs about $2N$
floating-point operations per token. The backward pass takes about twice
as many. Training on $D$ tokens therefore takes about

$$C \approx 6ND$$
:eqlabel:`eq_gpt-six-nd`

operations, ignoring the attention scores, which are small at short
contexts. Our run, with $N = 4.7$ million and $D = 8.2$ million, comes to
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
learning rate of $3 \times 10^{-3}$, without warmup, a setting in which
pre-norm trains without trouble. A third run repeats the post-norm model
with a warmup. It uses `LMOptimizer` with the same peak rate, 100 steps of
warmup, and the cosine decay that follows, but without weight decay or
clipping.

```{.python .input #gpt-breaking-it-with-one-flag}
%%tab pytorch
for pre_norm, warmup in ((True, False), (False, False), (False, True)):
    ablation = GPT(len(data.vocab), pre_norm=pre_norm)
    if warmup:
        optimizer = LMOptimizer(ablation, lr=3e-3, num_steps=800, warmup=100,
                                weight_decay=0.0, max_norm=math.inf)
    else:
        optimizer = torch.optim.AdamW(ablation.parameters(), lr=3e-3,
                                      weight_decay=0.0)
    losses = d2l.train_lm(ablation, data, optimizer, 800)
    print(f'pre_norm={pre_norm}, warmup={warmup}: loss at step '
          '100/200/400/800: ' + '/'.join(f'{sum(losses[k-50:k]) / 50:.2f}'
                                         for k in (100, 200, 400, 800)))
```

```{.python .input #gpt-breaking-it-with-one-flag}
%%tab jax
for pre_norm, warmup in ((True, False), (False, False), (False, True)):
    ablation = GPT(len(data.vocab), pre_norm=pre_norm,
                   rngs=nnx.Rngs(d2l.get_key()))
    if warmup:
        optimizer = LMOptimizer(ablation, lr=3e-3, num_steps=800, warmup=100,
                                weight_decay=0.0, max_norm=math.inf)
    else:
        optimizer = nnx.Optimizer(
            ablation, optax.adamw(3e-3, weight_decay=0.0), wrt=nnx.Param)
    losses = d2l.train_lm(ablation, data, optimizer, 800)
    print(f'pre_norm={pre_norm}, warmup={warmup}: loss at step '
          '100/200/400/800: ' + '/'.join(f'{sum(losses[k-50:k]) / 50:.2f}'
                                         for k in (100, 200, 400, 800)))
```

The pre-norm model trains normally. The post-norm model without warmup
does not diverge, but its loss stops at about 2.85 nats and stays there.
That is the unigram level computed above: the model predicts the letter
frequencies of the text and ignores its input, even the current character.

Attention that has learned nothing would not explain this level by itself.
If every attention sublayer output zero, the feed-forward sublayers would
still see the current character and could reach the bigram level of 2.33
nats. The tiny query and key gradients at initialization are therefore at
most part of the explanation. :citet:`xiong2020layer` showed that at
initialization the gradients of the parameters near the output of a
post-norm stack are large. A large learning rate without warmup then makes
training unstable, and a warmup avoids the failure. The third run confirms
this: with a warmup, the post-norm model leaves the unigram level and
trains, while pre-norm needs no warmup at this rate. GPT-2 already placed
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

We sample 120 characters after the prefix "the time traveller " at three
settings: the model's own distribution, a sharpened and truncated one, and
a flattened one.

```{.python .input #gpt-sampling-from-the-model-2}
%%tab pytorch, jax
prefix = data.vocab[list('the time traveller ')]
for temperature, top_k in ((1.0, None), (0.7, 8), (2.0, None)):
    out = model.generate(prefix, 120, temperature=temperature, top_k=top_k)
    d2l.print_wrapped(f'T={temperature}, top_k={top_k}: '
                      + repr(''.join(data.vocab.to_tokens(out))))
```

At $\tau = 1$ the model writes words and word-like strings in a plausible
order, with the habits of its corpus: the Penn Treebank replaces rare words
by `<unk>` and numbers by `N`, which the preprocessing turns into the words
"unk" and "n". The sharpened and truncated setting keeps to frequent words
and repeats itself, and at $\tau = 2$ the letters approach random strings.
A loss near 1.4 nats per character is a perplexity near 4: at a typical
position the model still hesitates among about four characters, so the
temperature changes the text visibly. Fluent text needs a much larger model
trained on much more text, such as the one we load next.

## Loading GPT-2
:label:`subsec_gpt-loading`

GPT-2 small is the `GPT` class with five settings
:cite:`Radford.Wu.Child.ea.2019`: 12 blocks of width 768 with 12 heads,
LayerNorm with biases, a GELU feed-forward network, learned positions for
up to 1,024 tokens, and pre-norm with a tied head. Loading it takes a
constructor call with these settings, the released weights, and the
released tokenizer. We register the weight and tokenizer files in
`d2l.DATA_HUB` with their content hashes, so that one download serves every
later run.

### The Tokenizer

```{.python .input #gpt-loading-gpt-2-1}
%%tab pytorch, jax
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

The tokenizer cell applies the BPE machinery of
:numref:`sec_text-sequence` to GPT-2's released merge list and vocabulary.
It uses the same pre-tokenization pattern as `d2l.BPETokenizer` and
assembles the result into `tiktoken`'s encoder. The construction needs two
data files and a regular expression, and no model library or configuration
framework.

### The Weight Map

The checkpoint stores one tensor per parameter under names such as
`h.3.attn.c_attn.weight`, so loading requires a dictionary that maps these
names onto our modules. GPT-2's original code implemented its linear layers
as a class `Conv1D` that stores each weight matrix with the input dimension
first, as $(\textrm{in}, \textrm{out})$. The fused `c_attn` matrix
concatenates queries, keys, and values in the same order as our `W_qkv`,
the token embedding doubles as the output head in both models, and our
LayerNorm uses GPT-2's $\epsilon = 10^{-5}$, so nothing else needs special
handling.

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
print(f'{d2l.num_params(gpt2) / 1e6:.1f}M parameters')
```

```{.python .input #gpt-loading-gpt-2-2}
%%tab jax
gpt2 = GPT(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
           max_len=1024, pos='learned', norm='layer', act='gelu',
           pre_norm=True, bias=True,  # the GPT-2 (124M) configuration
           rngs=nnx.Rngs(d2l.get_key()))
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
`Conv1D` layout, so the weights load unchanged. The safetensors file opens
directly into NumPy arrays.
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
cosine decay, and clipping, on text that the model reads less than twice
on average, and its validation loss follows its training loss. Moving the
normalization after the residual addition, at a learning rate that
pre-norm tolerates without warmup, leaves the model at the unigram
entropy, where its predictions ignore their context. A warmup lets the
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
       flags. Use it to count the parameters of `GPT(vocab_size=28)`, with
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
    1. Take one validation window of 128 characters and replace its
       character at position 64 by a different one. For every position,
       compute the largest absolute change of the logits of the trained
       `model`, and plot the change against position.
    1. Subclass `GPT` with a nested `CausalAttention` whose attention call
       passes `is_causal=False`. Train it as in
       :numref:`subsec_gpt-training`, with `LMOptimizer` set up for 1,000
       steps, but stop after 200 steps. Compare its training and
       validation losses with those that the training of `model` recorded
       at step 200, and explain the numbers.
    1. Sample 100 characters from the unmasked model with the prefix
       `'the time traveller '`. Explain why the validation loss of the
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

1. [code] **Memorizing a novel.** :numref:`subsec_gpt-training` appends
   the Penn Treebank to *The Time Machine* so that the model reads its
   training text only about 1.7 times.
    1. Train `GPT(len(data.vocab))` with the same recipe on
       `d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char',
       num_train=100000, num_val=3000)` alone, whose 100,000 overlapping
       training windows start at the first 100,000 positions of the text.
       Record both losses every 50 steps. At which step does the validation
       loss reach its minimum, and how far apart are the two losses at the
       end?
    1. A sample that copies its training text contains long strings that
       occur in that text verbatim. Draw ten samples of 100 characters at
       temperature 1 without truncation, with the prefix
       `'the time traveller '`, from this model and from `model`. Compute
       the *verbatim fraction* of each sample, the fraction of its
       20-character windows that occur verbatim in the corresponding
       training text.
    1. Retrain on *The Time Machine* with `dropout=0.3`. Does dropout move
       the minimum of the validation loss, lower it, or only slow the rise
       after it?

1. [code] **Checkpoint mismatches.** The GPT-2 checkpoint holds 160 tensors
   with 137,022,720 numbers in all, while `gpt2` has 124,439,808
   parameters.
    1. Identify the checkpoint tensors that the loading code of
       :numref:`subsec_gpt-loading` never reads, and explain why `GPT`
       needs no counterpart for them.
    1. Construct the GPT-2 configuration with `norm='rms'` and run the
       loading code. Which line fails, and why can no mapping of the
       checkpoint onto this configuration reproduce GPT-2? Then list, for
       the GPT-2 configuration with `pos='rope'` and `act='swiglu'`, every
       checkpoint tensor without a destination and every model parameter
       without a source.
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

    1. Compute the bits per byte of `model` on the text of the validation
       windows of `data`.
    1. Compute the bits per byte of `gpt2` on the same preprocessed text,
       in chunks of 1,024 tokens. Repeat on the raw text that corresponds to
       the validation windows, about the last 256,000 characters of the
       file `ptb.train.txt`, in which rare words appear as `<unk>` and
       numbers as `N`.
    1. Which of the two GPT-2 figures should be compared with that of
       `model`? Discuss what each side brings to the passage: `model` was
       trained on text of the same source, sees 128 characters of context,
       and predicts 28 symbols, whereas GPT-2 was trained on 40 GB of web
       text, sees up to 1,024 tokens, and predicts 50,257 symbols.
   :exlabel:`ex_gpt-bits-per-byte`

1. [extended] **Subwords at small scale.** ●
   :numref:`subsec_gpt-training` trains on characters, which keeps the
   embedding table small and makes the loss a number of bits per byte.
   Subword tokens have the opposite advantages: a window of 128 tokens
   covers several times more text, and each prediction carries more
   information. Test which effect wins at this scale.
    1. Train a `d2l.BPETokenizer` with the GPT-2 pattern on the
       preprocessed text of *The Time Machine* for vocabulary sizes 256,
       1,024, and 4,096; with 256 it learns no merges. Subclass
       `TimeMachinePTB` so that it encodes the preprocessed corpus with a
       given tokenizer, keeping the same split between training and
       validation text.
    1. Train `GPT` with each vocabulary as in
       :numref:`subsec_gpt-training` and convert the final validation
       losses to bits per byte as in :exref:`ex_gpt-bits-per-byte`. Count
       also the parameters in the embedding table of each model.
    1. A larger vocabulary gives each window more characters of context
       and each step more text, and in 1,000 steps the model then reads
       its training text several times instead of about 1.7 times. Repeat
       the best vocabulary with windows of about 128 characters, and with
       as many steps as it takes to read the text about 1.7 times. Which
       of the three effects explains the differences you found?

[Discussions](https://d2l.discourse.group/t/36775)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.2]{.kicker}

A GPT from Scratch<br>
**embedding, causal blocks, and a tied head · GPT-2 initialization · a training recipe · normalization placement · sampling · loading GPT-2**
:::
:::

::: {.slide title="One Window of 128 Characters Holds 128 Prediction Problems"}
[The problem]{.kicker}

| context | target |
|:--|:--|
| `the time travelle` | `r` |
| `the time traveller` | `␣` |
| `the time traveller␣` | `w` |

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

**Running example:** GPT-2 small (124M parameters), and our 4.7M-parameter
character model.
:::

::: {.slide title="Blocks Become a Language Model"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN on a residual stream |
| **A GPT from Scratch** | **embedding, causal mask, tied head; training; loading GPT-2** |
| Generation and the KV Cache | reuse keys and values while sampling |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Mixture of Experts | more parameters per FLOP |
| Scaling Laws | loss against parameters, data, compute |
:::

::: {.col .narrow}
- **Uses:** the block's defaults (pre-norm, RMSNorm, SwiGLU), RoPE,
  AdamW and schedules, character tokens, perplexity.
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
- RoPE lives inside the attention module; no other part of the model sees
  positions.
:::

::: {.slide title="The Forward Pass Fits in Seven Lines" only="pytorch"}
```{.python #gpt-exercises-1}
def forward(self, X):                   # X: (batch, T) token ids
    H = self.token_emb(X)               # (batch, T, d)
    if self.pos == 'learned':
        H = H + self.pos_emb(torch.arange(X.shape[1], device=X.device))
    for blk in self.blks:               # L blocks, causal attention
        H = blk(H)
    return F.linear(self.norm(H), self.token_emb.weight)  # tied
```

Output: (batch, T, V), the next-token logits at every position.
:::

::: {.slide title="The Forward Pass Fits in Seven Lines" only="jax"}
```{.python #gpt-exercises-2}
def __call__(self, X):                  # X: (batch, T) token ids
    H = self.token_emb(X)               # (batch, T, d)
    if self.pos == 'learned':
        H = H + self.pos_emb(jnp.arange(X.shape[1]))
    for blk in self.blks:               # L blocks, causal attention
        H = blk(H)
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
| ours | 256 | 6 | 4.7M | 0.007M | 4.73M |
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

::: {.slide title="A 4.7M-Parameter Model Needs More Than One Novel"}
1,000 steps $\times$ 64 windows $\times$ 128 characters = 8.2M characters.

| corpus | characters | passes in 1,000 steps |
|:--|--:|--:|
| *The Time Machine* | 0.17M | about 47 |
| *The Time Machine* + Penn Treebank | 4.8M for training | about 1.7 |

- 28 symbols: lowercase letters, space, unknown.
- Non-overlapping windows; the last 2,000 (256k characters) for
  validation.
- Characters keep the table at 7,168 parameters, and nats per character
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

The training text is read about 1.7 times: little to memorize, so the
validation loss tracks the training loss.
:::

::: {.slide title="Count Models Set the Reference Levels"}
@!gpt-training-the-model-4

- **Unigram**: letter frequencies, no context.
- **Bigram**: the previous character only.
- The GPT ends far below both: it uses much more than the previous
  character.
:::

::: {.slide title="Training Costs About 6ND Operations"}
Forward: $2N$ per token (a multiply and an add per parameter); backward:
about twice that.

| run | $N$ | $D$ (tokens) | $6ND$ |
|:--|--:|--:|--:|
| ours | 4.7M | 8.2M | $2 \times 10^{14}$ |
| GPT-2 small | 124M | about $10^{10}$ | $7 \times 10^{18}$ |
| GPT-2 XL | 1.5B | about $10^{10}$ | $10^{20}$ |
| largest current runs | | | $> 10^{25}$ |

The model definition hardly changes across 11 orders of magnitude.
:::

::: {.slide title="Post-Norm Without Warmup Sticks at the Unigram Level"}
Plain AdamW at a constant $3 \times 10^{-3}$ for 800 steps, pre-norm and
post-norm; then post-norm with 100 steps of warmup:

@!gpt-breaking-it-with-one-flag

A loss at the unigram entropy means predictions that ignore the input,
even the current character.
:::

::: {.slide title="Check: Is Starved Attention the Whole Story?"}
[Check]{.kicker}

At initialization, post-norm gives the query and key projections tiny
gradients. Suppose attention learned nothing at all. Which loss could the
post-norm model still reach?

. . .

- The FFNs act on the current character, so the bigram level is within
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

::: {.slide title="Lower Temperature Repeats, Higher Temperature Scrambles"}
@!gpt-sampling-from-the-model-2

$\tau = 1$: the model's own distribution. $\tau = 0.7$, $k = 8$: safer and
more repetitive. $\tau = 2$: close to random letters.
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
        linear.kernel[...] = jnp.asarray(weights[key + '.weight'])
        linear.bias[...] = jnp.asarray(weights[key + '.bias'])
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
- On text read about 1.7 times, validation follows training; post-norm
  without warmup stalls at the unigram level.
- Temperature and top-$k$ reshape sampling; GPT-2 loads with a name map.

**Next:** generation recomputes the whole context for every token; the KV
cache removes that.
:::
