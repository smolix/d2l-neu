# Vision Transformer
:label:`sec_vision-transformer`

The convolutional networks of :numref:`chap_cnn` were designed for images.
They build in two assumptions (:numref:`sec_why-conv`): *locality*, by
which a unit looks only at a small neighborhood, and *translation
equivariance*, by which a shifted input produces a shifted output. A
transformer block assumes neither. It maps a sequence of vectors to a
sequence of the same shape, and nothing in it says that the vectors stand
for words, or that two of them are neighbors. Can the same blocks, told
nothing about images, learn to classify them?

The first obstacle is the choice of tokens. One token per pixel is far too
expensive, because attention scores every pair of tokens. Even a small
$96 \times 96$ image has 9,216 pixels, so a single attention head in a
single layer would score $9{,}216^2 \approx 85$ million pairs of pixels; at
the $224 \times 224$ resolution of ImageNet the count exceeds 2.5 billion.
Early attention models for vision therefore kept the computation local or
the images small. :citet:`ramachandran2019stand` restricted self-attention
to small windows. :citet:`cordonnier2020relationship` proved that a
self-attention layer with enough heads can express any convolution, and
applied full attention to $2 \times 2$ patches of small images.

The *vision transformer* (ViT) of :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`
takes a coarser route. It cuts the image into patches of $16 \times 16$
pixels, embeds each patch as one token, and passes the sequence to an
unmodified transformer encoder. Our $96 \times 96$ image becomes 36 tokens,
and each head scores 1,296 pairs. Nothing in a ViT tells it that two
patches are neighbors.

This section builds a ViT, trains it on Fashion-MNIST, and checks whether its
position embeddings learn the layout of the patch grid. It then compares the
ViT with a convolutional network of the same size, trained with the same
recipe for the same number of epochs.

```{.python .input #vision-transformer}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #vision-transformer}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Patches as Tokens

Consider an image with height $h$, width $w$, and $c$ channels, and a patch
side $p$ that divides both $h$ and $w$. Cutting the image into a grid of
non-overlapping patches gives $m = hw/p^2$ patches, and each patch flattens
to a vector of length $cp^2$. A linear map then projects every flattened
patch to a vector of width $d$, the width of the transformer. For our
grayscale images, $h = w = 96$, $c = 1$, and $p = 16$, which gives a
$6 \times 6$ grid of $m = 36$ patches of 256 pixels each.

Cutting and projecting can be done in a single operation, a
convolution whose kernel size and stride both equal $p$. At each output
position, the $c \times p \times p$ kernel of output channel $k$ covers
exactly one patch, and the output is the inner product of the kernel with
that patch, which is entry $k$ of the patch's linear projection. The
convolution's weight tensor of shape $(d, c, p, p)$ is the projection
matrix, reshaped. `PatchEmbedding` applies this convolution and flattens
the grid of outputs into a sequence.

```{.python .input #vision-transformer-patches-as-tokens-1}
%%tab pytorch
class PatchEmbedding(nn.Module):  #@save
    """Image-to-sequence stem of the vision transformer."""
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512,
                 num_channels=3):
        super().__init__()
        # A partial trailing patch would silently shrink the token grid
        assert img_size % patch_size == 0, 'patches must tile the image'
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(num_channels, num_hiddens,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, X):  # (batch, c, h, w) -> (batch, patches, d)
        return self.conv(X).flatten(2).transpose(1, 2)
```

```{.python .input #vision-transformer-patches-as-tokens-1}
%%tab jax
class PatchEmbedding(nnx.Module):  #@save
    """Image-to-sequence stem of the vision transformer."""
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512,
                 num_channels=3, *, rngs):
        # A partial trailing patch would silently shrink the token grid
        assert img_size % patch_size == 0, 'patches must tile the image'
        self.num_patches = (img_size // patch_size) ** 2
        # PyTorch's default initialization: U(-a, a), a = 1/sqrt(fan_in)
        a = (num_channels * patch_size ** 2) ** -0.5
        init = lambda key, shape, dtype=jnp.float32: jax.random.uniform(
            key, shape, dtype, -a, a)
        self.conv = nnx.Conv(num_channels, num_hiddens,
                             kernel_size=(patch_size, patch_size),
                             strides=patch_size, padding='VALID',
                             kernel_init=init, bias_init=init, rngs=rngs)

    def __call__(self, X):  # (batch, h, w, c) -> (batch, patches, d)
        X = self.conv(X)
        return X.reshape(X.shape[0], -1, X.shape[-1])
```

A minibatch of four $96 \times 96$ color images, cut into $16 \times 16$
patches, should become four sequences of 36 tokens of width 512:

```{.python .input #vision-transformer-patches-as-tokens-2}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = torch.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input #vision-transformer-patches-as-tokens-2}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens,
                           rngs=nnx.Rngs(d2l.get_key()))
X = jnp.zeros((batch_size, img_size, img_size, 3))
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

From here on the image is a sequence of token vectors, and the rest of the
model is the encoder of :numref:`sec_transformer`: a stack of transformer
blocks without a causal mask, in which every token attends to every other.

## The Vision Transformer

:numref:`fig_vit` shows the whole model. The patch tokens enter at the
bottom, together with one extra token that belongs to no patch. The stack of
blocks processes all of them, and the head reads the final representation of
the extra token alone. We build the pieces in turn: the block, the extra
token, and the position embeddings.

![The vision transformer on a $96 \times 96$ image. A strided convolution turns the $6 \times 6$ grid of $16 \times 16$ patches into 36 tokens. A learned “&lt;cls&gt;” token is prepended and learned position embeddings are added; $L$ pre-norm transformer blocks process all 37 tokens, and a linear head classifies the final representation of the “&lt;cls&gt;” token.](../img/mdl-transformers-vit.svg)
:label:`fig_vit`

### The Encoder Block

The body of the model is the pre-norm transformer block of
:numref:`sec_transformer-block`. Each of its two sublayers, multi-head
self-attention and a two-layer MLP, reads a normalized copy of the residual
stream and adds its output back, so the stream itself is never normalized,
only added to. ViT adopted this arrangement after analyses had shown that
it trains deep stacks more reliably than the post-norm original
:cite:`baevski2018adaptive,wang2019learning,xiong2020layer`, and most
transformers have used it since.

`ViTBlock` is therefore a `d2l.TransformerBlock` with two changes, both
from the original ViT recipe. It normalizes with LayerNorm (`norm='layer'`),
which centers and scales each token vector. And it replaces the FFN through
the `ffn_factory` hook by a two-layer MLP of free width `mlp_num_hiddens`
with the Gaussian error linear unit (GELU), a smooth relative of ReLU
:cite:`Hendrycks.Gimpel.2016`, in its exact form, and dropout after the
hidden layer. Together with the dropout that `TransformerBlock` applies to
the output of each sublayer, this places dropout where the original ViT
does: after attention and after each of the MLP's two linear layers.
Because every patch may attend to every other, the block needs no mask: the
`mask` argument stays `None` throughout this section.

```{.python .input #vision-transformer-the-vision-transformer-block-1}
%%tab pytorch
class ViTBlock(d2l.TransformerBlock):  #@save
    """Pre-norm block with LayerNorm and an exact-GELU MLP."""
    def __init__(self, num_hiddens, mlp_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__(
            num_hiddens, num_heads, dropout=dropout, norm='layer',
            bias=use_bias, ffn_factory=lambda: nn.Sequential(
                nn.Linear(num_hiddens, mlp_num_hiddens),
                nn.GELU(),  # Exact GELU
                nn.Dropout(dropout),
                nn.Linear(mlp_num_hiddens, num_hiddens)))
```

```{.python .input #vision-transformer-the-vision-transformer-block-1}
%%tab jax
class ViTBlock(d2l.TransformerBlock):  #@save
    """Pre-norm block with LayerNorm and an exact-GELU MLP."""
    def __init__(self, num_hiddens, mlp_num_hiddens, num_heads, dropout,
                 use_bias=False, *, rngs):
        super().__init__(
            num_hiddens, num_heads, dropout=dropout, norm='layer',
            bias=use_bias, rngs=rngs, ffn_factory=lambda rngs: nnx.Sequential(
                d2l.Linear(num_hiddens, mlp_num_hiddens, rngs=rngs),
                lambda X: jax.nn.gelu(X, approximate=False),  # Exact GELU
                nnx.Dropout(dropout, rngs=rngs),
                d2l.Linear(mlp_num_hiddens, num_hiddens, rngs=rngs)))
```

As with every transformer block, the output shape equals the input shape,
so the blocks can be stacked:

```{.python .input #vision-transformer-the-vision-transformer-block-2}
%%tab pytorch
X = torch.ones(2, 100, 24)
encoder_blk = ViTBlock(24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input #vision-transformer-the-vision-transformer-block-2}
%%tab jax
X = jnp.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 48, 8, 0.5, rngs=nnx.Rngs(d2l.get_key()))
d2l.check_shape(nnx.view(encoder_blk, deterministic=True)(X), X.shape)
```

### The Class Token and Position Embeddings

Two more pieces turn the stack into a classifier. The first is a learned
vector, the “&lt;cls&gt;” (class) token, that is prepended to the 36 patch
tokens and initialized to zeros. It corresponds to no patch. In every block
it attends to all patches, so its final representation can collect whatever
evidence the classification needs, and the head reads the class from it
alone. Averaging the final representations of all patches works about as
well :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`; the exercises compare the
two readouts.

The second piece tells the model where each patch sits. Without position
information, self-attention is permutation equivariant
(:numref:`subsec_pos-ignores-order`): permuting the input tokens permutes
the outputs in the same way, so a model that reads one output token cannot
distinguish an image from any rearrangement of its patches. A patch has a
row *and* a column, and both must be encoded. A fixed *2-D sinusoidal code*
uses half of each vector for the 1-D sinusoidal code of the row
(:numref:`sec_positional-information`) and the other half for that of the
column :cite:`Beyer.Zhai.Kolesnikov.2022`. A 2-D form of RoPE rotates
queries and keys by angles proportional to the row and to the column
:cite:`Heo.Park.Han.ea.2024`. The original ViT does neither. It learns one
free vector per token position, initialized to small Gaussian noise, and
adds it to the token. We follow the original recipe. At initialization the
model therefore has no notion that patch 7 lies directly to the left of
patch 8 and directly above patch 13; whatever spatial structure it uses, it
must discover during training.

`ViT` assembles the pieces: patch embedding, class token, position
embeddings with dropout, `num_blks` blocks, and a head that normalizes the
class token's final representation and maps it to the logits. The class
also fixes its training recipe in `configure_optimizers`, which
:numref:`subsec_vit-first-run` explains.

```{.python .input #vision-transformer-the-full-model}
%%tab pytorch
class ViT(d2l.Classifier):  #@save
    """Vision transformer."""
    def __init__(self, *, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, num_steps, emb_dropout=0.0,
                 blk_dropout=0.0, lr=1e-3, use_bias=False, num_classes=10,
                 num_channels=1):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens, num_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_tokens = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable, initialized to small noise
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, num_hiddens) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.ModuleList([
            ViTBlock(num_hiddens, mlp_num_hiddens, num_heads, blk_dropout,
                     use_bias=use_bias) for _ in range(num_blks)])
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens, eps=1e-5),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)  # (batch, patches, d)
        cls = self.cls_token.expand(X.shape[0], -1, -1)
        X = self.dropout(torch.cat((cls, X), 1) + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])

    def configure_optimizers(self):
        # AdamW decays weight matrices and kernels; it exempts biases, norm
        # parameters, the class token, and the position embeddings
        named = list(self.named_parameters())
        decay = lambda n, p: p.ndim >= 2 and n not in ('cls_token',
                                                        'pos_embedding')
        optim = torch.optim.AdamW(
            [{'params': [p for n, p in named if decay(n, p)]},
             {'params': [p for n, p in named if not decay(n, p)],
              'weight_decay': 0.0}], lr=self.lr, weight_decay=0.05)
        # Linear warmup over the first tenth of the steps, then cosine
        return optim, torch.optim.lr_scheduler.LambdaLR(optim, lambda t: (
            d2l.cosine_warmup(t, self.num_steps, self.num_steps // 10)))
```

```{.python .input #vision-transformer-the-full-model}
%%tab jax
class ViT(d2l.Classifier):  #@save
    """Vision transformer."""
    def __init__(self, *, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, num_steps, emb_dropout=0.0,
                 blk_dropout=0.0, lr=1e-3, use_bias=False, num_classes=10,
                 num_channels=1, rngs):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens, num_channels, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, num_hiddens)))
        num_tokens = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable, initialized to small noise
        self.pos_embedding = nnx.Param(
            rngs.params.normal((1, num_tokens, num_hiddens)) * 0.02)
        self.dropout = nnx.Dropout(emb_dropout, rngs=rngs)
        self.blks = nnx.List([
            ViTBlock(num_hiddens, mlp_num_hiddens, num_heads, blk_dropout,
                     use_bias=use_bias, rngs=rngs) for _ in range(num_blks)])
        self.head = nnx.Sequential(
            nnx.LayerNorm(num_hiddens, epsilon=1e-5, rngs=rngs),
            d2l.Linear(num_hiddens, num_classes, rngs=rngs))

    def forward(self, X):
        X = self.patch_embedding(X)  # (batch, patches, d)
        cls = jnp.broadcast_to(self.cls_token[...], (len(X), 1, X.shape[2]))
        X = self.dropout(jnp.concatenate((cls, X), 1) + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])

    def configure_optimizers(self):
        # AdamW decays weight matrices and kernels; it exempts biases, norm
        # parameters, the class token, and the position embeddings
        decay = lambda params: jax.tree_util.tree_map_with_path(
            lambda path, p: p.ndim >= 2 and path[0].key not in (
                'cls_token', 'pos_embedding'), params)
        # Linear warmup over the first tenth of the steps, then cosine
        return optax.adamw(lambda t: self.lr * d2l.cosine_warmup(
            t, self.num_steps, self.num_steps // 10), weight_decay=0.05,
            mask=decay)
```

### Register Tokens

The class token is an input with no image content, a place where the model
can collect and store information. Large ViTs turn out to need more such
places. :citet:`Darcet.Oquab.Mairal.ea.2024` found that ViTs of the size of
ViT-Large and above, once trained long enough, take over about 2% of the
patch tokens, mostly in uninformative background regions, and use them for
internal computation. The outputs of these tokens have norms about ten times
those of the others, they carry little information about their own patch,
and they show up as bright spots in the attention maps. This is the vision
counterpart of the attention sinks of :numref:`subsec_kv-cache-sinks`,
positions that receive a large share of the attention whether or not their
content is relevant to the query.

The remedy is to provide the storage explicitly. A few additional learned
tokens, called *registers*, are appended to the input like the class token
and discarded at the output. One register already removes the artifacts,
and four, the number used in most of the paper's experiments, improve
the tasks that read the patch outputs, such as segmentation, depth
estimation, and object discovery, while adding less than 2% to the
computation. Our two-block model is far below the size at which the
artifacts appear, so we train it without registers.

## Training on Fashion-MNIST

### A First Run
:label:`subsec_vit-first-run`

We train a small ViT on Fashion-MNIST, with the images resized to
$96 \times 96$ so that $16 \times 16$ patches give the $6 \times 6$ grid of
36 tokens. The model has two blocks of width 512 with eight heads and an MLP
of width 2,048, about 6.5 million parameters in all.

The training recipe matters more for a transformer than for a CNN. In the
ablation of :citet:`touvron2021training`, replacing AdamW by SGD, with
everything else unchanged, lowered the ImageNet accuracy of a ViT-Base from
81.8% to 74.5%. `ViT` therefore uses the optimizer and schedule of the
compact modern recipe of :numref:`sec_training_recipes`: AdamW with weight
decay 0.05, and a learning rate that rises linearly from zero and then
decays to zero along a half-cosine :eqref:`eq_cosine_warmup`. We tuned each
model's base rate over $\{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$. The
ViT trains best at $3 \times 10^{-4}$ and ends four to five points lower at
$10^{-3}$; the CNN of :numref:`subsec_vit-cnn-baseline` varies by about a
point over the three rates and does equally well at $10^{-3}$ and
$3 \times 10^{-3}$, so we use $10^{-3}$ for it. For both models the warmup
takes the first tenth of the minibatch steps, which over ten epochs is the
first epoch. The decay acts on the weight
matrices and convolution kernels only. Biases and normalization parameters
are exempt, as in :numref:`subsec_adamw-what-not-to-decay`, and so are the
class token and the position embeddings, as in the training code of
:citet:`touvron2021training`. Decaying the position embeddings would also
pull them toward zero while we try to measure what they learn. We leave out the recipe's label smoothing and
Mixup, regularizers that help over long runs, so that the ViT and the CNN
it is compared with later both minimize the plain cross-entropy for ten
epochs.

```{.python .input #vision-transformer-a-first-run-1}
%%tab pytorch, jax
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
num_steps = 10 * len(data.train_dataloader())  # Ten epochs of minibatches
hparams = dict(img_size=96, patch_size=16, num_hiddens=512,
               mlp_num_hiddens=2048, num_heads=8, num_blks=2,
               emb_dropout=0.1, blk_dropout=0.1, num_steps=num_steps)
```

```{.python .input #vision-transformer-a-first-run-3}
%%tab pytorch
model = ViT(**hparams, lr=3e-4)
```

```{.python .input #vision-transformer-a-first-run-3}
%%tab jax
model = ViT(**hparams, lr=3e-4, rngs=nnx.Rngs(d2l.get_key()))
```

```{.python .input #vision-transformer-a-first-run-2}
%%tab pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
vit_history = trainer.history
vit_params, vit_acc = d2l.num_params(model), vit_history['val_acc'][-1]
print(f'ViT: {vit_params/1e6:.1f}M parameters, '
      f'validation accuracy {vit_acc:.2f}')
```

The ViT reaches a validation accuracy of about 90%. Before we ask whether
that is good for a model of its size, we look at what it has learned about
the position of its patches.

### Grid Structure in the Learned Embeddings
:label:`subsec_vit-pos-grid`

The position embeddings started as noise, and nothing in the training
signal ever names a row or a column. In fully trained ViTs they reflect the
grid nonetheless. In the analysis of
:citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`, each position's embedding is
most similar to those of its neighbors and of its own row and column. Does
our small model show this after ten epochs?

We measure the structure with one curve. For every pair of the 36 patch
positions we compute the cosine similarity of their embeddings, after
subtracting the mean embedding over all positions: a component shared by
every position says nothing about where a patch is. We then average the
similarities over all pairs at the same grid distance, counted as the larger
of the row offset and the column offset, so that the eight cells around a
patch are at distance 1. Embeddings that encode the grid give a curve that
falls with distance. Independent random vectors in 512 dimensions give a
flat curve slightly below zero, near $-1/35$: once centered, the 36 vectors
sum to zero, which makes their average pairwise cosine negative. As a
reference we compute the same curve for the fixed
2-D sinusoidal code, which encodes the grid by design.

```{.python .input #vision-transformer-do-the-position-embeddings-discover-the-grid}
%%tab pytorch
def grid_similarity(P):
    """Mean cosine similarity of 36 position embeddings, by grid distance."""
    P = P - P.mean(0)                    # Drop the part shared by all
    P = P / torch.linalg.vector_norm(P, dim=1, keepdim=True)
    sim = P @ P.T                        # (36, 36) cosine similarities
    rows, cols = torch.meshgrid(torch.arange(6), torch.arange(6),
                                indexing='ij')
    ij = torch.stack((rows.flatten(), cols.flatten()), 1)  # Grid cells
    dist = (ij[:, None] - ij[None]).abs().amax(-1)  # Max of the two offsets
    return [float(sim[dist == k].mean()) for k in range(1, 6)]

S = d2l.sinusoidal_encoding(6, 256)      # 1-D codes of the 6 rows/columns
rows, cols = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
fixed = torch.cat((S[rows.flatten()], S[cols.flatten()]), 1)  # (36, 512)
learned = model.pos_embedding[0, 1:].detach().cpu()  # Drop the cls slot
d2l.plot(list(range(1, 6)), [grid_similarity(learned), grid_similarity(fixed)],
         'grid distance', 'mean cosine similarity',
         legend=['learned', '2-D sinusoidal'])
```

```{.python .input #vision-transformer-do-the-position-embeddings-discover-the-grid}
%%tab jax
def grid_similarity(P):
    """Mean cosine similarity of 36 position embeddings, by grid distance."""
    P = P - P.mean(0)                    # Drop the part shared by all
    P = P / jnp.linalg.norm(P, axis=1, keepdims=True)
    sim = P @ P.T                        # (36, 36) cosine similarities
    rows, cols = jnp.meshgrid(jnp.arange(6), jnp.arange(6), indexing='ij')
    ij = jnp.stack((rows.flatten(), cols.flatten()), 1)  # Grid cells
    dist = jnp.abs(ij[:, None] - ij[None]).max(-1)  # Max of the two offsets
    return [float(sim[dist == k].mean()) for k in range(1, 6)]

S = d2l.sinusoidal_encoding(6, 256)      # 1-D codes of the 6 rows/columns
rows, cols = jnp.meshgrid(jnp.arange(6), jnp.arange(6), indexing='ij')
fixed = jnp.concatenate((S[rows.flatten()], S[cols.flatten()]), 1)  # (36, 512)
learned = model.pos_embedding[0, 1:]     # Drop the cls slot
d2l.plot(list(range(1, 6)), [grid_similarity(learned), grid_similarity(fixed)],
         'grid distance', 'mean cosine similarity',
         legend=['learned', '2-D sinusoidal'])
```

The fixed code's curve falls from about 0.76 for neighboring patches to
between $-0.3$ and $-0.5$ at distances 3 to 5, because by construction
patches in nearby rows and columns get similar codes. The learned
embeddings show only a trace of this structure. Their curve stays close to
zero: neighbors are slightly more similar than patches three cells apart,
by about a tenth, and the curve rises again at distance 5, where every pair
joins two opposite edges of the image. Fashion-MNIST items rarely reach the
edges, so the border patches mostly see background, which may make their
embeddings alike; an exercise tests whether the small dip is more than
chance. Ten epochs on 60,000 images teach the ViT little about its own
geometry. The network that we compare with next is given this geometry by
its architecture.

## Comparison with a Convolutional Network

### A Convolutional Baseline at the Same Budget
:label:`subsec_vit-cnn-baseline`

Is 90% good for a model of 6.5 million parameters trained for ten epochs?
We compare the ViT with a CNN built from the residual blocks of
:numref:`sec_resnet`, each of which adds the output of two $3 \times 3$
convolutions with batch normalization to its input. `CompactResNet` uses the
familiar $7 \times 7$ stem and then seven residual blocks whose widths bring
the total to the ViT's parameter count. Everything else is held equal. Both
models see the same data for ten epochs, and both use the same recipe:
`CompactResNet` borrows `configure_optimizers` from `ViT`. The two also do
about the same work, about half a billion floating-point operations per
image in the forward pass (the exercises derive both counts).

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-1}
%%tab pytorch
class CompactResNet(d2l.Classifier):
    """A ResNet-style CNN parameter-matched to the ViT above."""
    configure_optimizers = ViT.configure_optimizers  # The same recipe

    def __init__(self, num_steps, lr=1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            d2l.Residual(64), d2l.Residual(64),
            d2l.Residual(128, strides=2), d2l.Residual(128),
            d2l.Residual(256, strides=2), d2l.Residual(256),
            d2l.Residual(512, strides=2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes))

    def forward(self, X):
        return self.net(X)

    def weight_init(self, module):
        # The Trainer calls this after the first minibatch fixed the shapes
        if type(module) in (nn.Linear, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')  # He
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-1}
%%tab jax
class CompactResNet(d2l.Classifier):
    """A ResNet-style CNN parameter-matched to the ViT above."""
    configure_optimizers = ViT.configure_optimizers  # The same recipe

    def __init__(self, num_steps, lr=1e-3, num_classes=10, *, rngs):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        self.net = nnx.Sequential(
            # Pad 3 on every side: with stride 2 on even sizes, 'same' pads
            # one more row and column at the bottom and right
            nnx.Conv(1, 64, kernel_size=(7, 7), strides=(2, 2), padding=3,
                     use_bias=False, kernel_init=he, rngs=rngs),
            nnx.BatchNorm(64, momentum=0.9, use_fast_variance=False,
                          rngs=rngs), nnx.relu,
            lambda x: nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                   padding=((1, 1), (1, 1))),
            d2l.Residual(64, in_channels=64, rngs=rngs),
            d2l.Residual(64, in_channels=64, rngs=rngs),
            d2l.Residual(128, strides=(2, 2), in_channels=64, rngs=rngs),
            d2l.Residual(128, in_channels=128, rngs=rngs),
            d2l.Residual(256, strides=(2, 2), in_channels=128, rngs=rngs),
            d2l.Residual(256, in_channels=256, rngs=rngs),
            d2l.Residual(512, strides=(2, 2), in_channels=256, rngs=rngs),
            lambda x: x.mean(axis=(1, 2)),
            nnx.Linear(512, num_classes, kernel_init=he, rngs=rngs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-4}
%%tab pytorch
cnn = CompactResNet(num_steps, lr=1e-3)
```

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-4}
%%tab jax
cnn = CompactResNet(num_steps, lr=1e-3, rngs=nnx.Rngs(d2l.get_key()))
```

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-2}
%%tab pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(cnn, data)
cnn_history = trainer.history
cnn_params, cnn_acc = d2l.num_params(cnn), cnn_history['val_acc'][-1]
print(f'CNN: {cnn_params/1e6:.1f}M parameters, '
      f'validation accuracy {cnn_acc:.2f}')
print(f'ViT: {vit_params/1e6:.1f}M parameters, '
      f'validation accuracy {vit_acc:.2f}')
```

Plotting the validation accuracy of the two runs against the epoch shows
how the gap develops over training:

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-3}
%%tab pytorch, jax
d2l.plot(list(range(1, 11)), [vit_history['val_acc'], cnn_history['val_acc']],
         'epoch', 'validation accuracy', legend=['ViT', 'CNN'])
```

At the same parameter count, data, recipe, and number of epochs, and at
about the same computational cost, the CNN reaches about 94%, four to
five points ahead of the ViT. The experiment measures the two models under one
protocol; it does not say which property of the CNN produces the gap. The
CNN builds in locality and translation equivariance, which the ViT must
infer from 60,000 images, and the position embeddings of
:numref:`subsec_vit-pos-grid` show how much of the spatial layout the ViT
has inferred after ten epochs. But the two models also differ in depth, in
normalization, and in how finely they look at the image.

### The Comparison at Other Scales

The patch size alone moves the result by two points on Fashion-MNIST and
by sixteen on CIFAR-10. Training for 300 epochs with heavy augmentation,
:citet:`Hassani.Walton.Shah.ea.2021` report 93.2% on Fashion-MNIST for ViTs
of about 3.7 to 3.9 million parameters with $16 \times 16$ patches, 94.5%
with $8 \times 8$ patches, and 95.2% with $4 \times 4$ patches, slightly
above the 94.8% of a ResNet-18. On CIFAR-10 the same three models span 78%
to 94%, and only the smallest patches beat the ResNet-18. Smaller patches
mean more tokens, so the ViT sees finer detail at a higher cost: halving $p$
quadruples the number of tokens and roughly quadruples the computation,
while the parameter count barely changes.

The ranking also changes with the amount of data. When
:citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` pretrained on the 300 million
images of JFT-300M and fine-tuned on smaller benchmarks, their largest ViTs
exceeded the convolutional baselines they compared with. In a controlled
version, the same models were pretrained on random subsets of JFT-300M of
9 million to 300 million images: the ResNets led on the smaller subsets,
and the ViTs overtook them on the larger ones.
:citet:`touvron2021training` then trained ViTs on the 1.3 million images of
ImageNet alone, with strong augmentation and long training, and made them
competitive with convolutional networks of similar cost. With enough data
or a strong enough recipe, the assumptions that the CNN builds in are no
longer needed for accuracy, and the transformer contributes its generality:
the same block serves text, images, and both together.

## Summary

A vision transformer turns an image into a sequence by cutting it into
patches and embedding each patch with one strided convolution; an unmasked
encoder stack then processes the patch tokens, and a linear head classifies
the final representation of a learned class token. The architecture is the
text transformer's, and only the tokenization and the readout are new.
Position embeddings must encode a row and a column. Learned from noise, as
in the original ViT, they show only a trace of the grid after ten epochs
on Fashion-MNIST, while fixed 2-D sinusoidal codes build it in. Large ViTs also need a few register tokens as internal storage. Matched
with a CNN in parameters, computation, data, and training recipe, our ViT
trails by four to five points; smaller patches and much more data narrow
or reverse such gaps.

## Exercises

1. [code] **Parameters versus compute.** The ViT of
   :numref:`subsec_vit-first-run` reads grayscale images of $h = w = 96$
   pixels with $c = 1$ channel through patches of side $p = 16$. It has
   width $d = 512$, MLP width $4d$, and two blocks. Count one multiply--add
   as two FLOPs, as :eqref:`eq_multihead-flops` does, and write $n = m + 1$
   for the number of tokens.
    1. Express the parameter counts of `PatchEmbedding` and of the position
       embeddings as functions of $p$. By how many parameters does the model
       change when $p$ is halved to 8, and what fraction of its 6.5 million
       parameters is this change?
    1. Show that the stem costs the same number of FLOPs per image for every
       $p$. Derive the FLOPs of one `ViTBlock` as a function of $n$ and $d$,
       and compute the factor by which the forward FLOPs of the model grow
       when $p$ is halved to 8.
    1. Find the number of tokens at which the term of your block count that
       is quadratic in $n$ equals the remaining terms. Which patch sizes put
       this model beyond that point, and how far below it does ViT-B/16
       operate, with $d = 768$ at a resolution of $224 \times 224$?
    1. `CompactResNet` was matched to the ViT by parameter count. Count its
       convolution FLOPs per image, ignoring normalization, activations, and
       pooling, and compare them with those of the ViT.
    1. Train the ViT with $p = 8$, with the protocol of
       :numref:`subsec_vit-first-run`, and compare its validation accuracy
       with `vit_acc` and `cnn_acc`. Which of the two comparisons with the
       CNN, at equal parameters or at equal computation, does this run
       inform?
   :exlabel:`ex_vision-transformer-parameters-compute`

1. [code] **Choice of readout.** `ViT` classifies from the final
   representation of the “&lt;cls&gt;” token.
    1. Under this readout, part of the computation of the last `ViTBlock`
       cannot influence the logits. Identify it, and compute the fraction of
       the FLOPs of that block that inference could skip in the
       configuration of :numref:`subsec_vit-first-run`, counting as in
       problem :exref:`ex_vision-transformer-parameters-compute`.
    1. Implement the alternative readout that averages the final
       representations of the 36 patches, and remove the “&lt;cls&gt;” token
       together with its position embedding. Train the model with the
       protocol of :numref:`subsec_vit-first-run`, and compare its
       validation accuracy with `vit_acc`.
    1. :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` report that the two
       readouts work about equally well but need different learning rates.
       Train both readouts at the learning rates $10^{-4}$ and $10^{-3}$
       as well. Does the ranking of the two readouts
       depend on the learning rate?

1. [code] **Position information.** By the proposition of
   :numref:`subsec_pos-ignores-order`, self-attention without positional
   information is equivariant to permutations of its input tokens.
    1. Suppose that the 36 patch position embeddings of a `ViT` are all
       equal. Show that its logits are then unchanged when the patch tokens
       are permuted, and that permuting the $16 \times 16$ pixel blocks of
       the input image has the same effect as permuting the tokens.
    1. Write a variant of `ViT.forward` that applies a fixed permutation to
       the tokens produced by `PatchEmbedding`, before the position
       embeddings are added. Measure the validation accuracy of the trained
       `model` under five random permutations.
    1. Train a `ViT` whose position embeddings are fixed at zero, with the
       protocol of :numref:`subsec_vit-first-run`. Compare its validation
       accuracy with `vit_acc`, and confirm that the five permutations leave
       its predictions unchanged.
    1. Train a `ViT` whose patch position embeddings are fixed to the 2-D
       sinusoidal code `fixed` of :numref:`subsec_vit-pos-grid`, with a
       learned embedding for the “&lt;cls&gt;” slot. Compare its validation
       accuracy with `vit_acc` and with the model of the previous part.
    1. Compare the accuracy drop under permutation with the gaps between the
       three trained models. What does each of these numbers measure?
   :exlabel:`ex_vision-transformer-position-information`

1. [code] **Grid permutation test.** Let $\Delta$ be the difference
   between the first and the third value that `grid_similarity` returns for
   the learned embeddings, the mean similarity at grid distance 1 minus that
   at grid distance 3.
    1. Permute the rows of `learned` at random, which reassigns the
       embeddings to grid cells, recompute $\Delta$, and repeat 1,000 times.
       In what fraction of the permutations is $\Delta$ at least as large as
       for the trained `model`? What does this fraction say about the
       evidence for grid structure after ten epochs?
    1. Train a `ViT` with the protocol of :numref:`subsec_vit-first-run`
       for 30 epochs instead of 10, with `num_steps` set to 30 epochs of
       minibatches, and repeat the test. How do $\Delta$ and
       its position within the permutation distribution change with longer
       training?

1. **Shift equivariance.** Shift a $96 \times 96$ image by $s$ pixels to
   the right, and fill the vacated columns with zeros. Suppose that the $s$
   rightmost columns of the original image are zero.
    1. For $s = 16$, describe the tokens that `PatchEmbedding` produces for
       the shifted image, and show that they are a rearrangement of those of
       the original image. Give an image for which this fails at $s = 8$.
    1. Use this and problem
       :exref:`ex_vision-transformer-position-information` to show that a
       `ViT` whose patch position embeddings are all equal assigns the same
       logits to both images when $s = 16$. Why does the argument not extend
       to the trained `model`?
    1. Compute the stride of the final feature map of `CompactResNet`
       relative to its input. Show that, for cells whose receptive field
       lies inside the image, a shift of the input by a multiple of this
       stride moves the final feature map by whole cells, and explain why
       no such statement holds for a shift by fewer pixels. Compute the
       side length of the receptive field of a cell of the final feature
       map with :eqref:`eq_receptive_field`. Does any cell of the final
       feature map of a $96 \times 96$ input meet the condition?
    1. Neither model is therefore built to be invariant to a shift by one
       pixel. Name a property that every convolution of `CompactResNet`
       builds in and the attention layers of the ViT lack, and a property
       that both models share.

1. [code] **Attention distance.** ● For one head, let $\alpha_{ij}$ be the
   attention weight of query patch $i$ on key patch $j$ in an image, and
   let $\mathbf{u}_i$ be the pixel coordinates of the center of patch $i$.
   The *mean attention distance* of the head
   :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` averages

   $$\bar{r} = \frac{1}{36} \sum_{i=1}^{36} \sum_{j=1}^{36} \frac{\alpha_{ij}}{\sum_{k=1}^{36} \alpha_{ik}}\, \|\mathbf{u}_i - \mathbf{u}_j\|$$

   over images, with the “&lt;cls&gt;” token excluded as query and key. In
   large pretrained ViTs, some heads of the lowest layers attend locally
   while others attend across the whole image, and the distance grows with
   depth.
    1. Evaluate $\bar{r}$ for a head that attends uniformly to all 36
       patches, as a reference value.
    1. After a forward pass of the trained `model` in evaluation mode,
       `blk.attention.attention_weights` holds the attention weights of
       block `blk`, of shape (batch size, 8, 37, 37). Compute $\bar{r}$ for
       each of the 16 heads on 512 validation images, and plot it against
       the block index. Do the two blocks show either pattern of the
       pretrained ViTs?
    1. Repeat the measurement for the ViT of problem
       :exref:`ex_vision-transformer-position-information` whose position
       embeddings are fixed at zero. What other than position embeddings
       can make a head attend to nearby patches, and how do the results of
       the two models separate the two causes?

1. **Shifted windows.** ● Swin transformers :cite:`liu2021swin` restrict
   attention to non-overlapping windows of $M \times M$ patches. Successive
   blocks alternate between a regular partition of the patch grid into
   windows and a partition displaced by $\lfloor M/2 \rfloor$ patches along
   both axes. Normalization and MLPs act on each token separately. Call
   the set of input patches that can influence the output of a token after
   $L$ blocks its *reach*. A window is the product of a row interval and a
   column interval, so the reach is the product of its one-dimensional
   counterparts, and it suffices to study one row of patches far from the
   border of the grid.
    1. Show that without the displaced partition, the reach of every token
       remains its own window at every depth.
    1. Show that with the alternation, the reach after $L$ blocks is an
       interval of $ML$ patches. Compare this growth with that of a stack
       of $3 \times 3$ convolutions of stride 1 and with a single layer of
       global attention. On a $56 \times 56$ grid with $M = 7$, after how
       many blocks can the token in row 28 and column 28, counting from 0,
       be influenced by every patch?
    1. Show that the length $ML$ stays the same for every displacement
       $s$ with $0 < s < M$. After two blocks, let $g(s)$ be the smallest
       distance from a token to the nearer end of its reach, taken over all
       tokens. Compute $g(s)$, and show that $s = \lfloor M/2 \rfloor$
       maximizes it.

1. [extended] **Distillation from the CNN.** ●● Knowledge distillation trains
   a student model to match the predictions of a trained teacher model as
   well as the labels :cite:`Hinton.Vinyals.Dean.2015`.
   :citet:`touvron2021training` find that ViTs distilled from a
   convolutional teacher outperform ViTs distilled from a transformer
   teacher, and suggest that the student inherits the inductive bias of
   the teacher. Use the trained `cnn` as the teacher and a fresh `ViT` as the
   student. Let $\mathbf{z}_s$ and $\mathbf{z}_t$ be the logits of student
   and teacher for an image with label $y$, let $\mathrm{CE}$ be the
   cross-entropy of the softmax of its first argument, and let $y_t$ be the
   class with the largest teacher logit. Hard-label distillation minimizes

   $$\mathcal{L}_\textrm{hard} = \tfrac{1}{2}\,\mathrm{CE}(\mathbf{z}_s, y) + \tfrac{1}{2}\,\mathrm{CE}(\mathbf{z}_s, y_t),$$

   and soft distillation at temperature $\tau$ and weight $\lambda$
   minimizes

   $$\mathcal{L}_\textrm{soft} = (1 - \lambda)\,\mathrm{CE}(\mathbf{z}_s, y) + \lambda \tau^2\, \mathrm{KL}\big(\mathrm{softmax}(\mathbf{z}_t / \tau) \,\|\, \mathrm{softmax}(\mathbf{z}_s / \tau)\big).$$

    1. Compute the teacher's logits on all training images once, and
       deliver them with each minibatch, so that the training step of the
       student needs no forward pass of the teacher. On what fraction of
       the training images is $y_t \neq y$? Explain what this fraction
       implies for the difference between $\mathcal{L}_\textrm{hard}$ and
       plain cross-entropy in this setting, and why the implication does
       not carry over to the augmented training of
       :citet:`touvron2021training`.
    1. Override `training_step` of `ViT` to minimize
       $\mathcal{L}_\textrm{hard}$, and train with the protocol
       of :numref:`subsec_vit-first-run`. Train again with
       $\mathcal{L}_\textrm{soft}$ at $\tau = 3$ for
       $\lambda \in \{0.1, 0.5\}$, and compare the three students with
       `vit_acc`.
    1. For the undistilled `model` and for each student, compute the
       fraction of validation images on which its prediction differs from
       the teacher's, as in the disagreement analysis of
       :citet:`touvron2021training`. Among the validation images that it
       misclassifies, compute also the fraction on which it predicts the
       same wrong class as the teacher. Which of the two numbers separates
       resemblance to the teacher from accuracy?
    1. The teacher is both more accurate than the student and
       convolutional, so a gain from distillation does not by itself show
       that an inductive bias was transferred. Construct a control that
       separates the two explanations, run it, and state which outcome
       would favor the inductive-bias explanation.

[Discussions](https://d2l.discourse.group/t/36778)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.5]{.kicker}

Vision transformer<br>
**patches as tokens · class token and position embeddings · registers · a matched comparison with a CNN**
:::
:::

::: {.slide title="Attention over Pixels Scores 85 Million Pairs per Head"}
[The problem]{.kicker}

| image | tokens | scores per head and layer |
|:--|--:|--:|
| $96 \times 96$, one token per pixel | 9,216 | 85 million |
| $224 \times 224$, one token per pixel | 50,176 | 2.5 billion |
| $96 \times 96$, $16 \times 16$ patches | 36 | 1,296 |

A transformer block maps a sequence to a sequence and never assumes words.

::: {.d2l-note .warn}
How do we turn an image into a short sequence, and what does the model lose?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- turn an image into tokens with one strided convolution;
- count the tokens, parameters, and FLOPs of a ViT;
- explain the class token, the position embeddings, and register tokens;
- train a ViT with AdamW, warmup, and cosine decay;
- read a ViT-versus-CNN comparison matched in parameters, FLOPs, data,
  and recipe.

**Running example:** Fashion-MNIST at $96 \times 96$, a $6 \times 6$ grid of
patches, a 6.5M-parameter ViT against a 6.5M-parameter CNN.
:::

::: {.slide title="Images Are One More Kind of Token Sequence"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN on a residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| Generation and the KV Cache | reuse keys and values while sampling |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| **Vision Transformer** | **images as token sequences** |
| Mixture of Experts | more parameters per FLOP |
| Scaling Laws | loss against parameters, data, compute |
:::

::: {.col .narrow}
- **Uses:** the encoder wiring (no mask), the pre-norm block, multi-head
  attention, residual CNN blocks, AdamW and the cosine schedule.
- **Skips:** detection, segmentation, and self-supervised pretraining
  (Image Models part).
:::
:::
:::

::: {.slide title="A CNN Builds In Locality; a ViT Does Not"}
- **Locality:** a convolution unit sees a small neighborhood.
- **Translation equivariance:** shift the input, and the feature maps
  shift with it.
- Local self-attention restricted attention to small windows
  (Ramachandran et al., 2019).

. . .

**ViT** (Dosovitskiy et al., 2021): $16 \times 16$ patches, one token per
patch, an unmodified transformer encoder. Nothing tells it that two patches
are neighbors.
:::

::: {.slide title="A ViT Is a Patchifier, an Encoder, and a Head on One Token"}
::: {.cols .vc}
::: {.col .fig}
@fig:mdl-transformers-vit
:::

::: {.col}
- 36 patch tokens + 1 learned class token, plus learned positions.
- $L$ pre-norm blocks, no mask.
- The head reads the class token only.
:::
:::
:::

::: {.slide title="Cutting into Patches and Projecting Is One Strided Convolution"}
Image $h \times w$ with $c$ channels, patch side $p$:
$m = hw/p^2$ patches, each a vector of length $cp^2$.

- A convolution with **kernel size = stride = $p$** covers each patch
  exactly once.
- Output channel $k$ at a position $=$ inner product of kernel $k$ with
  that patch $=$ entry $k$ of the patch's linear projection.
- Weights $(d, c, p, p)$: the $d \times cp^2$ projection matrix, reshaped.

::: {.d2l-note}
$96 \times 96 \times 1$, $p = 16$: 36 tokens of length 256, projected to
$d = 512$.
:::
:::

::: {.slide title="Four Images Become Four Sequences of 36 Tokens"}
`PatchEmbedding` = the strided convolution, then flatten the
$6 \times 6$ grid into a sequence:

@vision-transformer-patches-as-tokens-2
:::

::: {.slide title="Check: Tokens and Scores at ImageNet Resolution"}
A $224 \times 224$ image with $16 \times 16$ patches: how many tokens, and
how many attention scores per head? And with $8 \times 8$ patches?

. . .

- $p = 16$: $14 \times 14 = 196$ tokens, $196^2 = 38{,}416$ scores.
- $p = 8$: $28 \times 28 = 784$ tokens, $784^2 = 614{,}656$ scores, 16
  times as many.
- The model's parameter count barely changes: the stem shrinks
  $4\times$, the position table grows $4\times$.
:::

::: {.slide title="The ViT Block Is the Pre-Norm Encoder Block with a GELU MLP"}
$$\mathbf{X} \leftarrow \mathbf{X} + \mathrm{MHA}(\mathrm{LN}(\mathbf{X})), \qquad
\mathbf{X} \leftarrow \mathbf{X} + \mathrm{MLP}(\mathrm{LN}(\mathbf{X}))$$

- LN: LayerNorm; MHA: multi-head self-attention, **no mask**.
- `ViTBlock` is `d2l.TransformerBlock` with `norm='layer'` and its FFN
  swapped for the MLP: Linear $d \to 4d$, exact GELU, Linear $4d \to d$.
- Dropout after attention and after each linear layer of the MLP.
- The residual stream is only added to, never normalized; output
  shape = input shape, so blocks stack.
:::

::: {.slide title="A Learned Class Token Collects the Evidence for the Head"}
- Prepend one learned vector, initialized to zeros: 37 tokens.
- In every block it attends to all 36 patches.
- The head (LayerNorm + Linear) reads only its final representation.
- Averaging the 36 patch outputs instead works about as well.
:::

::: {.slide title="The Forward Pass Reads One Token" only="pytorch"}
```{.python #vision-transformer-exercises-1}
def forward(self, X):                  # X: (batch, 1, 96, 96)
    X = self.patch_embedding(X)        # (batch, 36, 512)
    cls = self.cls_token.expand(X.shape[0], -1, -1)
    X = self.dropout(torch.cat((cls, X), 1) + self.pos_embedding)
    for blk in self.blks:              # (batch, 37, 512) throughout
        X = blk(X)
    return self.head(X[:, 0])          # the cls token only
```
:::

::: {.slide title="The Forward Pass Reads One Token" only="jax"}
```{.python #vision-transformer-exercises-2}
def forward(self, X):                  # X: (batch, 96, 96, 1)
    X = self.patch_embedding(X)        # (batch, 36, 512)
    cls = jnp.broadcast_to(self.cls_token[...], (len(X), 1, X.shape[2]))
    X = self.dropout(jnp.concatenate((cls, X), 1) + self.pos_embedding)
    for blk in self.blks:              # (batch, 37, 512) throughout
        X = blk(X)
    return self.head(X[:, 0])          # the cls token only
```
:::

::: {.slide title="Position Embeddings Must Encode a Row and a Column"}
Without positions, attention is permutation equivariant: an image and any
rearrangement of its patches look the same to the head.

| code | how |
|:--|:--|
| learned (original ViT) | one free vector per position, from noise |
| 2-D sinusoidal | half the vector codes the row, half the column |
| 2-D RoPE | rotate queries and keys by row and column angles |

With the learned table, the grid must be discovered from data.
:::

::: {.slide title="Large ViTs Take Over Background Patches; Registers Give Them Storage"}
- ViT-Large and above, trained long: about 2% of patch tokens get output
  norms about 10× the rest and bright spots in attention maps.
- They hold global, not local, information: storage for computation.
- The vision counterpart of attention sinks in language models.
- **Registers:** a few extra learned tokens, discarded at the output. One
  removes the artifacts; four improve dense tasks at under 2% extra FLOPs.

[Darcet et al., ICLR 2024]{.small}
:::

::: {.slide title="A Transformer Needs AdamW: SGD Costs ViT-Base Seven Points"}
DeiT on ImageNet, everything else equal: AdamW 81.8%, SGD 74.5%.

Our recipe (both models):

- AdamW, weight decay 0.05 on weight matrices and kernels only (no
  decay on biases, norms, class token, position embeddings);
- learning rate from 0 up to a tuned peak linearly over the first epoch,
  then a half-cosine to 0; peak $3 \times 10^{-4}$ for the ViT, $10^{-3}$
  for the CNN;
- ten epochs, minibatch 128, no augmentation.

[Touvron et al., ICML 2021]{.small}
:::

::: {.slide title="Ten Epochs Take the 6.5M-Parameter ViT to About 90%"}
@!vision-transformer-a-first-run-2
:::

::: {.slide title="The Learned Embeddings Show Only a Trace of the Grid"}
Mean cosine similarity of centered position embeddings, by grid distance
(max of row and column offset):

@!vision-transformer-do-the-position-embeddings-discover-the-grid

The fixed 2-D code falls from 0.76 to about $-0.4$; the learned one stays
near zero, a little higher for neighbors and for opposite edges.
:::

::: {.slide title="Check: What Would Random Embeddings Give?"}
Independent random vectors in 512 dimensions: what curve, and how far from
zero?

. . .

- A flat curve, slightly below zero: centered, the 36 vectors sum to
  zero, so their average pairwise cosine is about $-1/35 \approx -0.03$.
- One pair's cosine scatters by about $1/\sqrt{512} \approx 0.044$;
  averaging the many pairs at each distance shrinks that noise.
- Only a curve that falls with distance is evidence of a learned grid.
:::

::: {.slide title="The CNN Baseline Matches Parameters, FLOPs, Data, and Recipe"}
| | ViT | CompactResNet |
|:--|--:|--:|
| parameters | 6.46M | 6.45M |
| forward GFLOPs per image | 0.48 | 0.55 |
| data, epochs | Fashion-MNIST, 10 | same |
| recipe | AdamW + warmup + cosine | same (borrowed method) |

$7 \times 7$ stem, then seven residual blocks of widths 64 to 512.
:::

::: {.slide title="At Equal Budget the CNN Is Four to Five Points Ahead"}
@!vision-transformer-a-convolutional-baseline-at-the-same-budget-3

::: {.d2l-note}
One protocol, one dataset: the gap does not say which property of the CNN
causes it.
:::
:::

::: {.slide title="Patch Size and Data Scale Change the Ranking"}
- Fashion-MNIST, 300 epochs, heavy augmentation (Hassani et al., 2021):
  ViT 93.2% ($p = 16$), 94.5% ($p = 8$), 95.2% ($p = 4$); ResNet-18 94.8%.
  On CIFAR-10 the same ViTs span 78% to 94%.
- JFT-300M pretraining: the largest ViTs beat the cited ResNets; on
  9M–300M subsets, ResNets lead when data is small, ViTs when it is large.
- ImageNet-1k alone, strong augmentation: DeiT is competitive.
:::

::: {.slide title="Check: Halve the Patch Size"}
Our ViT with $p = 8$ instead of 16: what happens to the tokens, the
parameters, and the forward FLOPs?

. . .

- Tokens: $36 \to 144$ (plus the class token).
- Parameters: $6.46\text{M} \to 6.41\text{M}$, under 1% fewer.
- FLOPs: $0.48 \to 1.92$ GFLOPs per image, four times as many.
:::

::: {.slide title="Wrap-up"}
- ViT = strided-convolution patchify, the pre-norm encoder, a head on the
  class token; only tokenization and readout are new.
- Learned position embeddings start as noise; after ten epochs they show
  only a trace of the $6 \times 6$ grid, which a 2-D sinusoidal code
  builds in.
- Large ViTs need register tokens as internal storage.
- Matched in parameters, FLOPs, data, and recipe, the CNN leads on
  Fashion-MNIST by four to five points; smaller patches and more data move
  the ranking.

**Next:** Mixture of Experts, more parameters per FLOP.
:::
