# Decoder-Only Language Models (GPT)
:label:`sec_gpt`

A single `TransformerBlock` (:numref:`sec_transformer-block`) maps a sequence of $d$-dimensional token representations to a sequence of the same shape $(B, T, d)$. To turn a stack of causal Transformer blocks into an autoregressive **Generative Pre-trained Transformer (GPT)** :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language`, we wrap $L$ blocks between two boundary layers (:numref:`fig_gpt-stack`):

1. **Input embedding:** A token embedding table $\mathbf{E} \in \mathbb{R}^{V \times d}$ maps discrete token IDs $x_1, \dots, x_T \in \{0, \dots, V-1\}$ to vectors in $\mathbb{R}^d$ (optionally adding learned absolute position embeddings $\mathbf{P} \in \mathbb{R}^{T_{\max} \times d}$ when rotary position embeddings are not used inside attention).
2. **Final normalization and language-modeling head:** After $L$ causal `TransformerBlock` layers, a final normalization layer stabilizes the residual stream $\mathbf{H}^{(L)}$ and a linear head projects each position $t$ to next-token logits $\mathbf{Z} = \textrm{Norm}(\mathbf{H}^{(L)})\mathbf{E}^\top \in \mathbb{R}^{B \times T \times V}$, **tying** the output projection weights to the input embedding table $\mathbf{E}$ :cite:`Press.Wolf.2017,Inan.Khosravi.Socher.2017`.

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

![A GPT from bottom to top. A table $\mathbf{E}$ embeds the token ids into the residual stream; $L$ causal `TransformerBlock` layers read from and write to the stream; a final normalization and the transposed table $\mathbf{E}^\top$ turn each position's vector into logits over the vocabulary. Left, GPT-2 adds learned position vectors once at the bottom; right, the modern configuration leaves the stream alone at the bottom and rotates queries and keys inside each attention layer via RoPE.](../img/mdl-transformers-gpt-stack.svg)
:label:`fig_gpt-stack`

Inside each block, `CausalAttention` fuses the query, key, and value projections, supports **grouped-query attention** (`num_kv_heads` $\le$ `num_heads`, :numref:`sec_multihead-attention`), optionally rotates queries and keys via **RoPE** (`rope=True`, :numref:`sec_positional-information`), and applies the causal mask so position $t$ attends only to positions $s \le t$:

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

Our `GPT` class stacks `num_blks` configurable `TransformerBlock` modules (`norm='rms'`, `act='swiglu'`, `pos='rope'`, `pre_norm=True` by default, while also supporting GPT-2's `norm='layer'`, `act='gelu'`, `pos='learned'`, `bias=True`) and ties the output projection to `token_emb.weight`:

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

Following GPT-2 :cite:`Radford.Wu.Child.ea.2019` and :numref:`subsec_signal-propagation`, `init_weights` draws linear and embedding weights from $\mathcal{N}(0, 0.02^2)$ while scaling the two residual output projections of each block (`attention.W_o` and `ffn.W_2`) by $1 / \sqrt{2L}$:

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

## Training a Byte-Level GPT on WikiText-103
:label:`subsec_gpt-training`

To keep our from-scratch experiments self-contained and free of tokenizer artifacts, we train directly on raw UTF-8 bytes ($V = 256$) from **WikiText-103** :cite:`Merity.Xiong.Bradbury.ea.2016` using mini-batches of $B = 64$ windows of length $T = 128$ bytes ($8{,}192$ tokens per step):

```{.python .input #gpt-training-the-model-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
print(f'{len(data.train) / 1e6:.0f}M training bytes, '
      f'{len(data.val) / 1e6:.2f}M validation bytes, '
      f'vocabulary of {len(data.vocab)}')
```

```{.python .input #gpt-training-the-model-3}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32

def gpt(*args, **kwargs):  # A fresh model, on the GPU, compiled
    return torch.compile(GPT(*args, **kwargs).to(d2l.try_gpu()))
```

```{.python .input #gpt-training-the-model-3}
%%tab jax
def gpt(*args, **kwargs):  # Fresh random weights for every model
    return GPT(*args, **kwargs, rngs=nnx.Rngs(d2l.get_key()))
```

### The Autoregressive Objective
:label:`subsec_gpt-objective`

Given an input window $\mathbf{x} = (x_1, \dots, x_T)$ and shifted target window $\mathbf{y} = (x_2, \dots, x_{T+1})$, the causal mask ensures that row $t$ of the output logits $\mathbf{z}_t \in \mathbb{R}^V$ depends only on $(x_1, \dots, x_t)$. Training minimizes the mean cross-entropy loss across all $T$ positions in parallel:

$$
\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{T}\sum_{t=1}^{T} \log \textrm{softmax}(\mathbf{z}_t)_{x_{t+1}}.
$$
:eqlabel:`eq_gpt-loss`

At initialization, before any learning occurs, the model assigns roughly uniform probability $1/V$ to every byte, so the initial cross-entropy loss is close to $\ln V = \ln 256 \approx 5.55$ nats/byte:

```{.python .input #gpt-the-objective}
%%tab pytorch, jax
start = data.train.tobytes().find(b'the European lobster')
X, Y = data.windows(data.train, [start])  # One window and its targets
print('X[0]:', repr(data.vocab.decode(X[0, :24].tolist())))
print('Y[0]:', repr(data.vocab.decode(Y[0, :24].tolist())))
print(np.asarray(d2l.causal_mask(5)).astype(int))
```

```{.python .input #gpt-from-blocks-to-a-language-model-2}
%%tab pytorch, jax
model = gpt(len(data.vocab), max_len=128)
loss = math.log(d2l.lm_perplexity(model, [next(data.val_dataloader())]))
print(f'{d2l.num_params(model) / 1e6:.2f}M parameters; loss before '
      f'training {loss:.2f} nats per byte, log 256 = {math.log(256):.2f}')
```

### The Training Recipe (`LMOptimizer`)

Modern LLM pretraining pairs **AdamW** ($\beta_1 = 0.9, \beta_2 = 0.95$, :numref:`sec_adam`) with **selective weight decay** (applying weight decay $\lambda = 0.1$ only to 2D linear projection matrices while exempting embeddings, biases, and 1D normalization scales), **linear warmup followed by cosine learning-rate decay** (:numref:`sec_scheduler`), and **global gradient-norm clipping** ($\|\nabla \mathcal{L}\|_2 \le 1.0$):

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

Training our 6-block byte-level `GPT` ($d = 256, H = 8, L = 6$, $\approx 4.8\textrm{M}$ parameters) for 1,000 steps ($\approx 8.2\textrm{M}$ bytes) drives validation cross-entropy well below both the empirical unigram entropy ($3.17$ nats/byte) and the empirical bigram entropy ($2.45$ nats/byte) of English Wikipedia text:

```{.python .input #gpt-training-the-modern-configuration}
%%tab pytorch, jax
model = gpt(len(data.vocab), max_len=128)
num_steps, every = 1000, 50
optimizer = LMOptimizer(model, lr=2e-3, num_steps=num_steps, warmup=100)
steps, train_curve, val_curve, timer = [], [], [], d2l.Timer()
for step in range(every, num_steps + 1, every):
    timer.start()
    losses = d2l.train_steps(model, data, optimizer, every)
    timer.stop()  # Training time only, without the validation passes
    steps.append(step)
    train_curve.append(sum(losses) / every)
    val_curve.append(math.log(d2l.lm_perplexity(model, data.val_dataloader())))
d2l.plot(steps, [train_curve, val_curve], 'step', 'loss (nats)',
         legend=['train', 'validation'], fmts=('-', '--'))
print(f'step {num_steps}: train {train_curve[-1]:.2f}, '
      f'validation {val_curve[-1]:.2f} nats per byte; '
      f'{timer.sum():.0f} s of training')
```

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

## Controlled Comparisons
:label:`subsec_gpt-comparisons`

### Run-to-Run Spread and FFN Activation (`SwiGLU` vs. `GELU`)
:label:`subsec_gpt-run-to-run`

To separate genuine architectural improvements from random seed noise, we train three independent seeds of our default `SwiGLU` model and three seeds of a parameter-matched `GELU` model (`act='gelu'`):

```{.python .input #gpt-run-to-run-spread}
%%tab pytorch, jax
def train_lm(model, data, lr=2e-3, num_steps=1000):
    """Train with the recipe of the main run; return milliseconds per step."""
    optimizer = LMOptimizer(model, lr=lr, num_steps=num_steps, warmup=100)
    # One untimed call of 50 steps warms up (JAX compiles), the rest is timed
    bench = d2l.Benchmark(lambda: d2l.train_steps(model, data, optimizer, 50),
                          warmup=1, repeats=num_steps // 50 - 1)
    return 1000 * bench.time / 50

runs = {'swiglu': [val_curve[-1]], 'gelu': []}  # The main run comes first
times = {'swiglu': [], 'gelu': []}  # Milliseconds per step
for act, repeats in (('swiglu', 2), ('gelu', 3)):
    for _ in range(repeats):
        net = gpt(len(data.vocab), act=act, max_len=128)
        times[act].append(train_lm(net, data))
        runs[act].append(math.log(d2l.lm_perplexity(net,
                                                    data.val_dataloader())))
    print(f'{act:>6} ({d2l.num_params(net):,} parameters): validation '
          + ', '.join(f'{v:.3f}' for v in runs[act])
          + f'; mean {np.mean(runs[act]):.3f}, range {np.ptp(runs[act]):.3f}')
```

Run-to-run variation across random seeds is only $\sim 0.01$ nats/byte, whereas `SwiGLU` consistently beats `GELU` at the same parameter and FLOP budget by $\sim 0.06$ nats/byte. (Similarly, switching `pre_norm=True` to `pre_norm=False` without warmup causes training to stall at the unigram loss of $3.17$ nats/byte due to the early-layer gradient attenuation and token collapse shown in :numref:`subsec_signal-propagation`, and even with warmup post-norm trails pre-norm by a few hundredths of a nat per byte.)

### Comparing Against a Matched Recurrent Baseline
:label:`subsec_gpt-rnn-baseline`

How does our 6-block `GPT` compare against the recurrent language model (`d2l.RNNLM`, :numref:`sec_rnn-scratch`) given the **same parameter count** ($d_{\textrm{rnn}} = 1920$, $\approx 4.7\textrm{M}$ parameters) and training steps? We sweep the RNN learning rate across $\{10^{-3}, 2\times 10^{-3}, 4\times 10^{-3}\}$ and compare validation loss as a function of window position $t \in \{1, \dots, 128\}$ (`loss_by_position`):

```{.python .input #gpt-against-a-recurrent-network-1}
%%tab pytorch
def rnnlm(vocab_size, num_hiddens):  # The tanh RNN LM of the Basics
    return d2l.RNNLM(d2l.RNN(256, num_hiddens), vocab_size).to(d2l.try_gpu())

@torch.inference_mode()
def loss_by_position(model, data):
    """Mean validation loss in nats at each position of the window."""
    device = next(model.parameters()).device
    total, count = 0, 0
    for X, Y in data.val_dataloader():
        logits = model(X.to(device))  # (batch, T, V)
        total += F.cross_entropy(logits.transpose(1, 2), Y.to(device),
                                 reduction='none').sum(0)
        count += len(X)
    return (total / count).tolist()
```

```{.python .input #gpt-against-a-recurrent-network-1}
%%tab jax
def rnnlm(vocab_size, num_hiddens):  # The tanh RNN LM of the Basics
    rngs = nnx.Rngs(d2l.get_key())
    return d2l.RNNLM(d2l.RNN(256, num_hiddens, rngs=rngs), vocab_size,
                     rngs=rngs)

@nnx.jit
def position_loss_sum(model, X, Y):
    """Loss in nats at each position, summed over the batch."""
    return optax.softmax_cross_entropy_with_integer_labels(model(X), Y).sum(0)

def loss_by_position(model, data):
    """Mean validation loss in nats at each position of the window."""
    total, count = 0, 0
    for X, Y in data.val_dataloader():
        total += position_loss_sum(model, jnp.asarray(X), jnp.asarray(Y))
        count += len(X)
    return (total / count).tolist()
```

```{.python .input #gpt-against-a-recurrent-network}
%%tab pytorch, jax
rnn_runs = []  # (validation loss, model, ms per step) of every run
for lr in (1e-3, 2e-3, 4e-3):
    for _ in range(2):
        rnn = rnnlm(len(data.vocab), 1920)
        ms = train_lm(rnn, data, lr=lr)
        rnn_runs.append((np.mean(loss_by_position(rnn, data)), rnn, ms))
    print(f'RNN ({d2l.num_params(rnn):,} parameters), peak rate {lr:g}: '
          'validation ' + ', '.join(f'{r[0]:.3f}' for r in rnn_runs[-2:]))
best = min(rnn_runs, key=lambda run: run[0])
positions = (0, 1, 4, 16, 64, 127)
print('validation loss at positions', ', '.join(map(str, positions)))
for name, net, ms in (('GPT', model, np.mean(times['swiglu'])),
                      ('best RNN', *best[1:])):
    loss = loss_by_position(net, data)
    print(f'{name:>8}: ' + ' '.join(f'{loss[t]:.2f}' for t in positions)
          + f'; {ms:.1f} ms per training step')
```

At position $t = 1$ (where only a single byte of context is available), `GPT` and `RNNLM` start at nearly the same loss. As more prefix context becomes available ($t \to 128$), the Transformer's validation loss drops more steeply than the RNN's because self-attention reads any prior byte in $O(1)$ hops rather than compressing history through $t$ sequential recurrent steps.

## Sampling from the Model
:label:`subsec_gpt-sampling`

Autoregressive generation (`GPT.generate`) appends one token at a time: at each step it feeds the current prefix through the model, divides the final position's logits $\mathbf{z}_t$ by a **temperature** $\tau > 0$, optionally masks all but the **top-$k$** highest logits to $-\infty$ :cite:`Fan.Lewis.Dauphin.2018`, and samples the next token from $\textrm{softmax}(\mathbf{z}_t / \tau)$ (or takes $\arg\max$ when $\tau = 0$):

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

```{.python .input #gpt-sampling-from-the-model-2}
%%tab pytorch, jax
prefix = data.vocab.encode('The history of the ')
for temperature, top_k in ((1.0, None), (0.7, 8), (2.0, None)):
    out = model.generate(prefix, 120, temperature=temperature, top_k=top_k)
    d2l.print_wrapped(f'T={temperature}, top_k={top_k}: '
                      + repr(data.vocab.decode(out)))
```

## Loading Pretrained GPT-2 (`124M`)
:label:`subsec_gpt-loading`

Because our `GPT` class supports `norm='layer'`, `act='gelu'`, `pos='learned'`, and `bias=True`, it can instantiate OpenAI's 124M-parameter **GPT-2 Small** architecture ($V = 50{,}257, d = 768, H = 12, L = 12, T_{\max} = 1024$) and load the official `model.safetensors` weights and byte-level BPE tokenizer (`gpt2_tokenizer`) directly:

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

:Mapping between OpenAI's GPT-2 `safetensors` checkpoint keys and our `GPT` module attributes (`Conv1D` weights have shape `(in, out)` and are transposed on load in PyTorch).
:label:`tab_gpt2-weight-map`

| Checkpoint tensor | `GPT` attribute | Note |
|:--|:--|:--|
| `wte.weight` | `token_emb.weight` | Shared with output head (`V x d`) |
| `wpe.weight` | `pos_emb.weight` | 1,024 learned position vectors |
| `h.i.ln_1`, `h.i.ln_2` | `blks[i].norm1`, `blks[i].norm2` | Pre-attention and pre-FFN LayerNorms |
| `h.i.attn.c_attn` | `blks[i].attention.W_qkv` | Fused QKV (`Conv1D`, transpose in PyTorch) |
| `h.i.attn.c_proj` | `blks[i].attention.W_o` | Output projection (`Conv1D`) |
| `h.i.mlp.c_fc`, `h.i.mlp.c_proj` | `blks[i].ffn.W_1`, `blks[i].ffn.W_2` | Two-layer GELU MLP (`Conv1D`) |
| `ln_f` | `norm` | Final LayerNorm before output head |

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

We verify the loaded checkpoint by computing its cross-entropy on the opening paragraph of *The Time Machine* ($\approx 3.9$ nats/token, or perplexity $\approx 49$) and generating continuations:

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

```{.python .input #gpt-loading-gpt-2-4}
%%tab pytorch, jax
out = gpt2.generate(enc.encode('Alan Turing theorized that computers '
                               'would one day become'), 16, top_k=1)
d2l.print_wrapped(enc.decode(out))
```

```{.python .input #gpt-loading-gpt-2-5}
%%tab pytorch, jax
out = gpt2.generate(enc.encode('The secret of a good deep learning '
                               'textbook is'), 40, temperature=0.8,
                    top_k=50)
d2l.print_wrapped(enc.decode(out))
```

## The Modern Recipe
:label:`sec_modern-recipe`

Comparing the 2019 `load_gpt2()` configuration against our default `GPT` highlights the four core upgrades that define modern open-weights decoder-only language models (:numref:`tab_modern-recipe`):

1. **Normalization:** Pre-norm `RMSNorm` without biases (:numref:`subsec_rmsnorm`) instead of `LayerNorm`, often paired with per-head **QK-Norm** (:numref:`subsec_qk-norm`) or sublayer-output normalization (OLMo 2, Gemma 3).
2. **Positions:** **Rotary Position Embeddings (RoPE)** (:numref:`sec_positional-information`) inside each attention layer instead of a fixed learned position table `wpe`.
3. **Feed-forward network:** Bias-free gated **`SwiGLU`** (or `GeGLU`) instead of a two-layer `GELU` MLP.
4. **Inference cache & capacity scaling:** **Grouped-query attention (GQA)**, **sliding-window attention**, or **low-rank latent compression (MLA)** to shrink the KV cache (:numref:`sec_kv-cache`), and **Mixture-of-Experts (MoE)** layers (:numref:`sec_moe`) to scale total parameters without scaling per-token FLOPs.

:Architectures of seven open-weights model families, as reported in their technical reports, 2023 to 2025. Attention and cache: :numref:`sec_kv-cache-reduction`. Normalization: :numref:`sec_transformer-block`. Positions: :numref:`sec_positional-information`. Mixture of experts: :numref:`sec_moe`.
:label:`tab_modern-recipe`

| Model | Attention and cache | Normalization | Positions | FFN | Dense or MoE |
|:--|:--|:--|:--|:--|:--|
| Mistral 7B (2023) :cite:`Jiang.Sablayrolles.Mensch.ea.2023` | GQA $32{:}8$; sliding window 4096 | RMSNorm, pre | RoPE, base $10^4$ | SwiGLU | Dense |
| Llama 3 (2024) | GQA $32{:}8$ (8B), $64{:}8$ (70B), $128{:}8$ (405B) | RMSNorm, pre | RoPE, base $5 \times 10^5$ | SwiGLU | Dense |
| Qwen3 (2025) :cite:`Yang.Li.Yang.ea.2025` | GQA: $64{:}8$ dense (32B), $64{:}4$ MoE (235B-A22B) | RMSNorm, pre; QK-norm | RoPE, base $10^6$; YaRN | SwiGLU | Dense, and MoE: 128 experts, 8 active |
| OLMo 2/3 (2024/25) :cite:`OLMo.2025,OLMo3.2025` | MHA (7B), GQA (32B); OLMo 3: window 4096 on 3 of 4 layers | RMSNorm on sublayer outputs; QK-norm | RoPE | SwiGLU | Dense |
| DeepSeek-V3 (2024) :cite:`Liu.Feng.Xue.ea.2024` | MLA: KV compressed to a 512-dim latent | RMSNorm, pre | Decoupled RoPE (64-dim shared key) | SwiGLU | MoE: 256 experts + 1 shared, 8 active |
| Gemma 3 (2025) :cite:`Gemma.Team.2025` | GQA; local:global $5{:}1$, window 1024 | RMSNorm, pre and post; QK-norm | RoPE, base $10^6$ global, $10^4$ local | GeGLU | Dense |
| gpt-oss (2025) :cite:`OpenAI.2025` | GQA $64{:}8$; window 128 on alternate layers; learned sinks | RMSNorm, pre | RoPE; YaRN | SwiGLU, clamped | MoE: 128 or 32 experts, 4 active |

## Summary

* A **decoder-only GPT** wraps $L$ causal `TransformerBlock` layers between a token embedding table $\mathbf{E} \in \mathbb{R}^{V \times d}$ and a final normalization + weight-tied output projection $\mathbf{E}^\top$, predicting next-token logits at all $T$ sequence positions in one parallel forward pass (:eqref:`eq_gpt-loss`).
* Training with **AdamW** (selective weight decay on 2D matrices only), linear warmup + cosine decay, and gradient clipping (`LMOptimizer`) drives our 4.8M-parameter byte-level `GPT` well below the bigram entropy of English text in 1,000 steps.
* In controlled comparisons at matched parameters and compute, `SwiGLU` outperforms `GELU` by $\sim 0.06$ nats/byte (versus a $\sim 0.01$-nat seed spread), and `GPT` achieves a steadily widening advantage over a matched `RNNLM` at deeper context positions $t$.
* Modern open-weights LLMs (:numref:`tab_modern-recipe`) share the pre-norm `RMSNorm` + `RoPE` + `SwiGLU`/`GeGLU` recipe of our `GPT` class, differing primarily in how they shrink the KV cache (:numref:`sec_kv-cache`) and whether they replace the dense FFN with a Mixture of Experts (:numref:`sec_moe`).

## Exercises

1. **Ablating the training recipe (`LMOptimizer`).** `LMOptimizer` combines four ingredients: selective weight decay ($\lambda = 0.1$ on 2D weight matrices only), linear warmup (100 steps), cosine decay, and global gradient clipping ($\|\nabla \mathcal{L}\|_2 \le 1.0$). Train our 6-block byte-level `GPT` for 1,000 steps while disabling one ingredient at a time: (a) applying weight decay to *all* parameters including embeddings and RMSNorm gains, (b) no warmup (`warmup=0`), (c) constant learning rate (`lr=2e-3` throughout), and (d) no gradient clipping. Which single change hurts validation loss the most?
:exlabel:`ex_gpt-recipe`

2. **Auditing the GPT-2 checkpoint tensor map.** The GPT-2 `model.safetensors` file holds 148 weight/bias tensors (plus 12 legacy causal mask buffers `h.i.attn.bias`). Write a short check verifying that `sum(p.numel() for p in gpt2.parameters())` equals the total element count of the 148 parameter tensors in the file ($124{,}439{,}808$), and explain why transposing `c_attn.weight` is required in PyTorch (`nn.Linear`) but not in Flax (`nnx.Linear`).
:exlabel:`ex_gpt-checkpoint-mismatches`

3. **Zero-layer tied-embedding baseline ($L = 0$).** Consider `GPT` with `num_blks=0`: input token $x_t$ is embedded as $\mathbf{e}_{x_t}$, normalized by `RMSNorm`, and dotted with $\mathbf{E}^\top$ to produce logits $\mathbf{z}_t \propto \mathbf{e}_{x_t} \mathbf{E}^\top$.
   1. What class of statistical language model ($n$-gram order) can a 0-block model represent?
   2. Why does tying the input and output embeddings ($\mathbf{W}_{\textrm{LM}} = \mathbf{E}^\top$) force the symmetric matrix $\mathbf{E}\mathbf{E}^\top$ to assign its largest diagonal entry to $x_{t+1} = x_t$ if all rows of $\mathbf{E}$ have equal norm, and how does the untied or 1-block model break this symmetry?
:exlabel:`ex_gpt-tied-head`

4. **Comparing bits-per-byte across tokenizers (Byte-level GPT vs. BPE GPT-2).** A byte-level model reports cross-entropy in nats/byte, whereas GPT-2 reports cross-entropy in nats/subword-token.
   1. Show that on any passage of $N_B$ UTF-8 bytes that `enc.encode` tokenizes into $N_T$ subword tokens, the exact **bits-per-byte (bpb)** of a subword model with mean per-token negative log-likelihood $\mathcal{L}_{\textrm{tok}}$ (in nats/token) is $\textrm{bpb} = \frac{N_T \, \mathcal{L}_{\textrm{tok}}}{N_B \ln 2}$.
   2. Evaluate both our 4.8M-parameter byte-level `model` and pretrained 124M-parameter `gpt2` in bits-per-byte on the first 10,000 bytes of the WikiText-103 validation set.
:exlabel:`ex_gpt-bits-per-byte`

5. **Top-$p$ (nucleus) sampling.** Extend `GPT.generate` with an optional `top_p` argument ($0 < p < 1$) :cite:`Holtzman.Buys.Du.ea.2020` that sorts the temperature-scaled probabilities in descending order and masks out tokens beyond the smallest prefix whose cumulative probability exceeds $p$. Compare completions from `gpt2` on the prompt `"Alan Turing theorized that computers would"` using `top_p=0.9, temperature=0.8` versus unfiltered `temperature=1.2`.

6. **Learning-rate stability with QK-Norm.** Train our 6-block byte-level `GPT` across peak learning rates $\eta \in \{2\times 10^{-3}, 6\times 10^{-3}, 1.5\times 10^{-2}\}$ with standard `CausalAttention` versus a variant that applies `RMSNorm(d_h)` to `q` and `k` inside `CausalAttention.qkv` (:numref:`subsec_qk-norm`). How does QK-Norm shift the maximum stable learning rate?
:exlabel:`ex_gpt-qk-norm-sensitivity`
