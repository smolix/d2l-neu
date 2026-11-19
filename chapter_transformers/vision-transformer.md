# Vision Transformer
:label:`sec_vision-transformer`

We have classified Fashion-MNIST garments before, with LeNet
(:numref:`sec_lenet`) and with a ResNet (:numref:`sec_resnet`). Both were
built for images. They rest on two assumptions (:numref:`sec_why-conv`):
*locality*, by which a unit looks only at a small neighborhood, and
*translation equivariance*, by which a shifted input produces a shifted
output. A transformer block assumes neither. It maps a sequence of vectors
to a sequence of the same shape, and nothing in it says that the vectors
stand for words, or that two of them are neighbors. Let us take the block
we built for text, tell it nothing about images, and see whether it can
learn to sort the same shirts, sneakers and bags.

The first obstacle is the choice of tokens. One token per pixel is far too
expensive, because attention scores every pair of tokens. Even a small
$96 \times 96$ image has 9,216 pixels, so a single attention head in a
single layer would score $9{,}216^2 \approx 85$ million pairs of pixels; at
the $224 \times 224$ resolution of ImageNet the count exceeds 2.5 billion.
Early attention models for vision therefore kept the computation local or
the images small. :citet:`ramachandran2019stand` restricted self-attention
to small windows. :citet:`cordonnier2020relationship` proved that a
self-attention layer with enough heads and relative position encodings can
express any convolution, and applied full attention to $2 \times 2$ patches
of small images.

The *vision transformer* (ViT) of :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`
takes a coarser route. It cuts the image into patches of $16 \times 16$
pixels, embeds each patch as one token, and passes the sequence to an
unmodified transformer encoder. Our $96 \times 96$ image becomes 36 tokens,
and each head scores 1,296 pairs instead of 85 million.

We build a ViT, train it on Fashion-MNIST, and then ask two questions of
it. Does the trained model use the layout of its patches, which nobody told
it about? And how does it fare against a ResNet of the same size, trained
with the same recipe for the same number of epochs?

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
$6 \times 6$ grid of $m = 36$ patches of 256 pixels each. Fashion-MNIST
images have only $28 \times 28$ pixels; we upsample them to $96 \times 96$
so that the patch size of the original ViT applies. Each patch therefore
covers under five of the original pixels per side, and the model sees a
garment as 36 small tiles. The upsampling is a convenience. It adds no
information, yet the stem now reads about twelve times as many pixels as the
original image holds; an exercise asks what that costs.

Cutting and projecting can be done in a single operation, a convolution
whose kernel size and stride both equal $p$. At each output position, the
$c \times p \times p$ kernel of output channel $k$ covers exactly one patch,
and the output is the inner product of the kernel with that patch, which is
entry $k$ of the patch's linear projection. The convolution's $d$ kernels,
each of shape $c \times p \times p$, are the rows of the $d \times cp^2$
projection matrix, reshaped. Since the windows never overlap, this
convolution inherits none of the locality of a CNN beyond the patch
boundary: it is simply a fast way to apply one matrix to every patch.
`PatchEmbedding` applies it and flattens the grid of outputs into a
sequence.

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

We check the shapes on a minibatch of four blank grayscale images, which
should come out as four sequences of 36 tokens of width 512:

```{.python .input #vision-transformer-patches-as-tokens-2}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens, num_channels=1)
X = torch.zeros(batch_size, 1, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input #vision-transformer-patches-as-tokens-2}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens, num_channels=1,
                           rngs=nnx.Rngs(d2l.get_key()))
X = jnp.zeros((batch_size, img_size, img_size, 1))
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

The body of the model is the pre-norm block of
:numref:`sec_transformer-block`, an arrangement that ViT took from earlier
language and translation models :cite:`baevski2018adaptive,wang2019learning`.
`ViTBlock` departs from the defaults of `d2l.TransformerBlock` in two
places, both taken from the original paper. It normalizes with LayerNorm,
which centers and scales each token vector, instead of RMSNorm. And its MLP
is the plain two-layer kind, with the Gaussian error linear unit (GELU), a
smooth relative of ReLU :cite:`Hendrycks.Gimpel.2016`, instead of SwiGLU.
Nothing about images calls for either; they are what transformers used in
2020, and we keep them so that our block stays close to the published one.
It is not identical: our attention projections carry no biases, and our
blocks also apply dropout to the attention weights, which the original does
not. Elsewhere dropout sits where the original puts it, after the
embeddings, after attention, and after each of the MLP's two linear layers.
The `mask` argument stays `None` throughout this section, because every
patch may attend to every other.

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

### The Class Token and Position Embeddings

Two more pieces turn the stack into a classifier. The first is a learned
vector, the “&lt;cls&gt;” (class) token, that is prepended to the 36 patch
tokens, initialized to zeros, and corresponds to no patch. In every block it
attends to all patches, so its final representation can collect whatever
evidence the classification needs, and the head reads the class from it
alone. The choice is a convention rather than a discovery: averaging the
final representations of the 36 patches, the usual way to read one vector
out of an encoder (:numref:`sec_transformer`), works about as well
:cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`, and the exercises compare the
two.

A token without image content turns out to have a second use, which large
models find for themselves. Large ViTs trained for long take over a few
patches of plain background and use them to hold global information that
no single patch holds, much as GPT-2 parks attention on the first token of
a passage (:numref:`subsec_kv-cache-sinks`). The outputs of those patches
then no longer describe their patch, which hurts every use that reads patch
outputs, such as segmentation or maps of attention. A few extra input-free
tokens, called *registers*, give the model room for such information and
leave the patches to describe themselves :cite:`Darcet.Oquab.Mairal.ea.2024`.

The second piece tells the model where each patch sits. Without position
information, self-attention is permutation equivariant
(:numref:`subsec_pos-ignores-order`): permuting the input tokens permutes
the outputs in the same way, so a model that reads one output token cannot
distinguish an image from any rearrangement of its patches. A shirt cut
into 36 tiles and shuffled would be the same shirt to it.

A patch has a row *and* a column, so a fixed code must be built in two
dimensions. The 1-D sinusoidal code of :numref:`sec_positional-information`,
applied to the patches in raster order, would give patch 6, at the right end
of the first row, and patch 7, at the left end of the second, neighboring
codes, although the two lie at opposite edges of the image. A *2-D
sinusoidal code* instead uses half of each vector for the 1-D code of the
row and the other half for that of the column
:cite:`Beyer.Zhai.Kolesnikov.2022`. A 2-D form of RoPE rotates queries and
keys by angles proportional to the row and to the column
:cite:`Heo.Park.Han.ea.2024`. A learned table need not be built in two
dimensions, because a free vector per position can come to reflect any
layout. The original ViT learns one such vector per token position,
initialized to small Gaussian noise, and adds it to the token. In the
paper's own ablation, learned 1-D and 2-D tables and relative encodings did
about equally well, while leaving out position information altogether cost
two to three points of accuracy in a few-shot evaluation on ImageNet
:cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`. We follow the original
recipe. At initialization the
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

## Training on Fashion-MNIST

### A First Run
:label:`subsec_vit-first-run`

We train a deliberately small ViT on the upsampled Fashion-MNIST images. It
has two blocks of width 512 with eight heads and an MLP of width 2,048,
about 6.5 million parameters in all. ViT-Base, the smallest model of the
original paper, has twelve blocks of width 768 and 86 million parameters;
two blocks keep each training run of this section short.

A transformer is also more particular about its optimizer than a CNN.
ResNets are routinely trained with SGD and momentum, but in the DeiT study
of :citet:`touvron2021training`, replacing AdamW by SGD, with everything
else unchanged, lowered the ImageNet accuracy of a ViT-Base from 81.8% to
74.5%. `ViT` therefore uses the optimizer and schedule of the compact modern
recipe of :numref:`sec_training_recipes`: AdamW with weight decay 0.05. The
learning rate rises linearly from zero during the first tenth of the
minibatch steps, which over ten epochs is the first epoch, and then falls to
zero along a half-cosine. We tuned the peak rate for each model over three
values and use the best: $3 \times 10^{-4}$ for the ViT and $10^{-3}$ for
the CNN of :numref:`subsec_vit-cnn-baseline`. An exercise repeats the
sweep.

The decay acts on the weight matrices and convolution kernels only. Biases
and normalization parameters are exempt, as in
:numref:`subsec_adamw-what-not-to-decay`, and so are the class token and
the position embeddings, as in the training code of
:citet:`touvron2021training`. Decaying the position embeddings would also
pull them toward zero while we try to measure what they learn. We leave out
the recipe's label smoothing and Mixup, which soften the training targets
and help over long runs, so that the ViT and the CNN it is compared with
later both minimize the plain cross-entropy for ten epochs.

```{.python .input #vision-transformer-a-first-run-1}
%%tab pytorch, jax
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
num_steps = 10 * len(data.train_dataloader())  # Ten epochs of minibatches
hparams = dict(img_size=96, patch_size=16, num_hiddens=512,
               mlp_num_hiddens=2048, num_heads=8, num_blks=2,
               emb_dropout=0.1, blk_dropout=0.1, num_steps=num_steps)
```

Along with a fresh model we keep a copy of its initial position table, the
noise that training starts from, for a comparison later in the section.

```{.python .input #vision-transformer-a-first-run-3}
%%tab pytorch
vit = lambda: ViT(**hparams, lr=3e-4)  # A fresh model at the tuned rate
model = vit()
pos_init = model.pos_embedding[0, 1:].detach().clone()  # The initial noise
```

```{.python .input #vision-transformer-a-first-run-3}
%%tab jax
vit = lambda: ViT(**hparams, lr=3e-4, rngs=nnx.Rngs(d2l.get_key()))
model = vit()  # A fresh model at the tuned rate
pos_init = model.pos_embedding[0, 1:]  # The initial noise, a new array
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

The ViT reaches a validation accuracy of about 89%. Before we ask whether
that is good for a model of its size, we look at what it has made of the
positions of its patches.

### The Role of Position
:label:`subsec_vit-pos-grid`

The model was never told that its 36 tokens form a grid. Has it learned to
use the layout anyway? The most direct test does not look inside the model
at all. We cut every validation image into its 36 blocks of
$16 \times 16$ pixels and rearrange the blocks by one fixed random
permutation of the grid cells. The model's position embeddings stay where
they are, so each block now arrives at a position that is not its own. The
test is a coarse relative of the shuffled pixels of :numref:`sec_lenet`,
with one difference: LeNet was trained on its shuffled images, while our
ViT is trained on intact ones and meets the shuffled blocks only at
validation. `shuffle_patches` does the rearrangement, and
`shuffled_accuracy` evaluates the model on the rearranged images.

```{.python .input #vision-transformer-does-the-vit-use-position-1}
%%tab pytorch
def shuffle_patches(X, perm, p=16):
    """Move the p x p blocks of images X: block perm[k] goes to grid cell k."""
    n, c, h, w = X.shape
    B = X.reshape(n, c, h // p, p, w // p, p).transpose(3, 4)  # Grid of blocks
    B = B.reshape(n, c, -1, p, p)[:, :, perm]  # Blocks in raster order, moved
    B = B.reshape(n, c, h // p, w // p, p, p).transpose(3, 4)
    return B.reshape(X.shape)

def shuffled_accuracy(model, data, perm):
    """Validation accuracy on images whose blocks are moved by perm."""
    return d2l.evaluate_accuracy(model, ((shuffle_patches(X, perm), y)
                                         for X, y in data.val_dataloader()))
```

```{.python .input #vision-transformer-does-the-vit-use-position-1}
%%tab jax
def shuffle_patches(X, perm, p=16):
    """Move the p x p blocks of images X: block perm[k] goes to grid cell k."""
    n, h, w, c = X.shape
    B = X.reshape(n, h // p, p, w // p, p, c).swapaxes(2, 3)  # Grid of blocks
    B = B.reshape(n, -1, p, p, c)[:, perm]  # Blocks in raster order, moved
    B = B.reshape(n, h // p, w // p, p, p, c).swapaxes(2, 3)
    return B.reshape(X.shape)

def shuffled_accuracy(model, data, perm):
    """Validation accuracy on images whose blocks are moved by perm."""
    return d2l.evaluate_accuracy(model, ((shuffle_patches(X, perm), y)
                                         for X, y in data.val_dataloader()))
```

Before running the test, consider the two extremes. A model that ignored
position would not notice the shuffle at all, by the permutation
equivariance above. A model that rested every decision on the layout would
fall to 10%, the rate of guessing. We try five permutations, since one
alone might move only unimportant blocks:

```{.python .input #vision-transformer-does-the-vit-use-position-2}
%%tab pytorch, jax
rng = np.random.default_rng(0)
perms = [rng.permutation(36) for _ in range(5)]  # Five fixed shuffles
accs = [shuffled_accuracy(model, data, perm) for perm in perms]
print(f'trained ViT: {vit_acc:.3f} as is, '
      f'{min(accs):.2f} to {max(accs):.2f} with its patches shuffled')
```

Shuffling drops the ViT from 89% to below one half under every one of the
five permutations, although it stays well above chance, and the printed
range shows that the damage depends on which blocks are moved. The trained
model does use the layout of its patches. Does it need it? A model that was
given positions has had no reason to learn to do without them, so the
shuffle test cannot answer this. A second model can: a ViT whose position
table is set to zero and kept out of training, so that it never sees a
position at all.

```{.python .input #vision-transformer-does-the-vit-use-position-3}
%%tab pytorch
def zero_positions(model):
    """Set the position table to zero and keep it out of training."""
    model.pos_embedding.data.zero_()
    model.pos_embedding.requires_grad_(False)  # No gradient: AdamW skips it
    return model
```

```{.python .input #vision-transformer-does-the-vit-use-position-3}
%%tab jax
def zero_positions(model):
    """Set the position table to zero and keep it out of training."""
    # The trainer differentiates and updates nnx.Param only; a plain
    # Variable stays as it is
    model.pos_embedding = nnx.Variable(jnp.zeros_like(model.pos_embedding))
    return model
```

We train two such models with the recipe of the first run, and one more ViT
with learned positions, so that each arm has two runs and their spread
gives a run-to-run range. One number can be predicted before the cell
prints it. With all positions equal, the proposition of
:numref:`subsec_pos-ignores-order` leaves the logits of the zero-position
model unchanged when its patches are shuffled, so its accuracy under the
first permutation must equal its accuracy on the intact images.

```{.python .input #vision-transformer-does-the-vit-use-position-4}
%%tab pytorch, jax
def final_accuracy(model, data):
    """Validation accuracy after the ten epochs of the first run."""
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    trainer.board.display = False  # No progress plot for the repeats
    trainer.fit(model, data)
    return trainer.history['val_acc'][-1]

learned_runs = [vit_acc, final_accuracy(vit(), data)]
zero_model = zero_positions(vit())
zero_runs = [final_accuracy(zero_model, data),
             final_accuracy(zero_positions(vit()), data)]
print('learned positions:', ', '.join(f'{a:.3f}' for a in learned_runs))
print('zero positions:   ', ', '.join(f'{a:.3f}' for a in zero_runs),
      f'(shuffled: {shuffled_accuracy(zero_model, data, perms[0]):.3f})')
```

The prediction holds: shuffled or not, the zero-position model scores the
same. Like the MLP of :numref:`sec_lenet` on shuffled pixels, it cannot
tell the difference. Each zero-position run lies one to two points below
each learned-position run, while the two runs of an arm agree to within
half a point. Learned positions help, but only a little. A ViT that has never
known where any patch sits still reaches about 87 to 88%, and the original
paper's ablation, cited above, found a loss of similar size at a far larger
scale.

So shuffling costs the trained ViT more than 40 points, yet the layout is
worth only one to two points of its accuracy. Using a piece of information
is not the same as needing it. A model that has positions builds features
that combine where with what, and a shuffled image feeds those features
nonsense. A model without positions learns to classify the unordered set of
36 tiles, and for Fashion-MNIST that set already says a great deal: a heel,
a strap, a collar. At this scale most of the accuracy comes from what the
patches contain, not from where they sit.

The shuffle test asked the model; the cosines we compute next only describe
its table. A table need not look like a grid to be used as one: 36
distinct random vectors already identify every position, and attention can
learn any relation between two positions through its query and key
projections. Still, the table shows what ten epochs wrote into it, and
fully trained ViTs give a reference. In the analysis of
:citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`, each position's embedding is
most similar to those of its neighbors and of its own row and column. Ten
epochs on Fashion-MNIST are far from full training, and our table still
carries much of its initial noise, so we look at it twice: as learned, and
as `learned - pos_init`, its change from initialization, which holds only
what training wrote.

One curve summarizes a table. We center the 36 patch embeddings, since a
component shared by every position says nothing about where a patch is,
and average the cosine similarity of all pairs at the same grid distance.
The distance is the larger of the row offset and the column offset, so
that all eight cells around a patch lie at distance 1. A table that encodes
the grid gives a curve that falls with distance. Two references frame the
result: the fixed 2-D sinusoidal code, which encodes the grid by design,
and independent random vectors, which give a flat curve slightly below zero
(centered vectors sum to zero, which puts their mean cosine near $-1/35$).

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
noise = torch.randn(36, 512)
curves = {'learned': grid_similarity(learned),
          'learned - initial': grid_similarity(learned - pos_init),
          '2-D sinusoidal': grid_similarity(fixed),
          'random': grid_similarity(noise)}
for name, curve in curves.items():
    print(f'{name:>17}:', '  '.join(f'{c:+.2f}' for c in curve))
d2l.plot(list(range(1, 6)), list(curves.values())[:3], 'grid distance',
         'mean cosine similarity', legend=list(curves)[:3])
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
noise = jax.random.normal(d2l.get_key(), (36, 512))
curves = {'learned': grid_similarity(learned),
          'learned - initial': grid_similarity(learned - pos_init),
          '2-D sinusoidal': grid_similarity(fixed),
          'random': grid_similarity(noise)}
for name, curve in curves.items():
    print(f'{name:>17}:', '  '.join(f'{c:+.2f}' for c in curve))
d2l.plot(list(range(1, 6)), list(curves.values())[:3], 'grid distance',
         'mean cosine similarity', legend=list(curves)[:3])
```

The fixed code's curve falls from about 0.76 for neighboring patches to
between $-0.3$ and $-0.5$ at distances 3 to 5, because by construction
patches in nearby rows and columns get similar codes. The learned table
shows the same tendency on a far smaller scale: the mean cosine of
neighbors is about 0.1 higher than that of patches two to four cells apart.
Small does not mean doubtful. The random curve is flat at $-0.03$, its five
values within about 0.01 of one another, so a difference of 0.1 is ten
times their spread. The change from initialization favors neighbors in the
same way, and then it turns up again: at distance 5, where every pair joins
two opposite edges of the grid, it climbs back to the level of neighbors or
a little above. Why the far edges should move alike, the curve does not
say. Fashion-MNIST garments fill the image along their longer side, so many
touch the top and the bottom, but the outer columns and above all the
corners mostly show background. A plausible reason is that training moves
the embeddings of such patches alike, and exercise
:exref:`ex_vision-transformer-grid-test` tests it.

## Comparison with a Convolutional Network

### A Convolutional Baseline at the Same Budget
:label:`subsec_vit-cnn-baseline`

Is 89% good for a model of 6.5 million parameters trained for ten epochs?
We compare the ViT with a CNN built from the residual blocks of
:numref:`sec_resnet`, each of which adds the output of two $3 \times 3$
convolutions with batch normalization to its input (or, where the block
downsamples, to a $1 \times 1$ projection of it). `CompactResNet` uses the
familiar $7 \times 7$ stem and then seven residual blocks whose widths bring
the total to the ViT's parameter count. The training conditions are held
equal. Both models see the same data for ten epochs, and both use the same
recipe:
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

A comparison needs a yardstick for how much a run can move by chance, so we
train the CNN twice from fresh weights, as we trained each ViT arm twice.
`compact_cnn` builds a fresh model, and the training cell after it prints
both CNN runs next to the two learned-position ViT runs.

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-4}
%%tab pytorch
compact_cnn = lambda: CompactResNet(num_steps, lr=1e-3)  # A fresh CNN
cnn = compact_cnn()
```

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-4}
%%tab jax
compact_cnn = lambda: CompactResNet(num_steps, lr=1e-3,
                                    rngs=nnx.Rngs(d2l.get_key()))
cnn = compact_cnn()  # A fresh CNN
```

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-2}
%%tab pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(cnn, data)
cnn_history = trainer.history
cnn_params, cnn_acc = d2l.num_params(cnn), cnn_history['val_acc'][-1]
cnn_runs = [cnn_acc, final_accuracy(compact_cnn(), data)]  # Run-to-run range
print(f'CNN: {cnn_params/1e6:.1f}M parameters, validation accuracy',
      ', '.join(f'{a:.3f}' for a in cnn_runs))
print(f'ViT: {vit_params/1e6:.1f}M parameters, validation accuracy',
      ', '.join(f'{a:.3f}' for a in learned_runs))
```

The CNN reaches about 94%, five points ahead of the ViT. The two runs of
each model agree to a tenth of a point, and against that yardstick a gap of five points is not an accident of
the draw. The validation curves of the first run of each show how the gap
develops during training:

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-3}
%%tab pytorch, jax
d2l.plot(list(range(1, 11)), [vit_history['val_acc'], cnn_history['val_acc']],
         'epoch', 'validation accuracy', legend=['ViT', 'CNN'])
```

The CNN leads from the first epoch. The gap is widest in the first three
epochs, where it reaches seven to ten points, and narrows to about five by
the end. Both curves flatten at the end because the cosine schedule takes
the learning rate to zero.

The losses tell more than the accuracies. In the progress plot of the
CNN's first run, above, the training loss falls almost to zero, and the
validation loss stops falling after four to six epochs and then creeps up
while the accuracy still rises, a sign that the network grows more
confident in all its answers, the wrong ones included. Ten epochs
are enough for the CNN to fit its 60,000 training images almost perfectly.
The ViT's training loss in :numref:`subsec_vit-first-run` ends well above
zero and not far below its validation loss, although dropout, which only
the ViT uses, accounts for part of that. A longer schedule would leave the
CNN little to fit and the ViT probably a good deal. How much of that would
reach the validation set, and whether the ViT would catch up, these runs
cannot say; the published results below can.

At the same parameter count, data, recipe, and number of epochs, and at
about the same computational cost, the CNN wins. The experiment measures
the two models under one protocol; it does not say which property of the
CNN produces the gap. The position table cannot explain most of it:
knowing where each patch sits was worth only one to two points to our ViT.
A CNN uses the layout in another way, by building locality into every
layer, and it differs in more than that: it looks at the image at a finer
grain than a 16-pixel patch, and it differs in depth and normalization. Nor
is attention short of expressive power in principle: with enough heads and
relative positions it can express a convolution, as the opening noted.
What a model can represent and what it learns from 60,000 images in ten
epochs are different things, and this comparison measures the second.

### The Comparison at Other Scales

With long training and augmentation, the gap closes.
:citet:`Hassani.Walton.Shah.ea.2021` trained compact ViTs, seven blocks and
under 4 million parameters, for 300 epochs with heavy augmentation on small
images. On Fashion-MNIST they report 93% with $16 \times 16$ patches, 94.5%
with $8 \times 8$ patches, and 95% with $4 \times 4$ patches, against 94.8%
for a ResNet-18 with about three times as many parameters. With one run of
each model, the two finer settings, within half a point of the ResNet-18,
are level with it, and the coarsest trails it by a point and a half. Only
on CIFAR-10 does the ViT pull ahead, and only with the finest patches: the
three ViTs span 78% to 94% against the ResNet-18's 90%, and
at $p = 16$ a $32 \times 32$ photograph leaves the ViT just four tokens.

What do these numbers say about our run? Even their coarsest patches reach
93% on Fashion-MNIST, so the cut into patches is not what holds our ViT at
89%. Their ViTs differ from ours in depth and recipe at once, seven blocks
trained for 300 epochs against two blocks trained for ten, and the
comparison cannot say which of the two lifts them four to six points above
ours. It does suggest that the transformer has far more to gain from a long
recipe than the CNN, whose ten-epoch run here already ends about a point
below their ResNet-18.

More data reverses the ranking, and this is the result that changed the
field. When :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` pretrained on
random subsets of the 300 million images of JFT-300M, from 9 million images
up, ResNets led on the smaller subsets and ViTs overtook them on the larger
ones. :citet:`touvron2021training` then trained ViTs on the 1.3 million
images of ImageNet alone, with strong augmentation and long training, and
made them competitive with convolutional networks of similar cost.

To train an image classifier from scratch on tens of thousands of images,
use a CNN. To use a transformer on such data, fine-tune a pretrained ViT,
or budget for hundreds of epochs and heavy augmentation. Most ViTs in use
today are pretrained: trained on far larger collections of images, often
paired with text, they serve as the image encoders of vision--language
models, which pass the encoder's patch tokens to a language model much as
if they were words. There the transformer's generality counts, since one
block serves text, images, and both together. A from-scratch run on 60,000
small images, as in this section, is the hardest case for the
architecture, and the CNN's built-in assumptions win it. How much data and
how large a model would be needed to turn the result around is a question
of scale.

## Summary

A vision transformer turns an image into a sequence by cutting it into
patches and embedding each patch with one strided convolution; an unmasked
encoder stack processes the patch tokens, and a linear head classifies the
final representation of a learned class token. Only the tokenization and
the readout are new; the block is the text transformer's. Position
embeddings learned from noise pick up a small grid structure in ten epochs
on Fashion-MNIST, and the trained model uses the layout: shuffling its
patches drops it from about 89% to below one half. Yet a ViT with no
position information at all ends only one to two points lower, so at this
scale most of the accuracy comes from what the patches contain. Matched
with a CNN in parameters, computation, data, and recipe, the ViT trails by
about five points, while two runs of either model agree to a tenth of a
point. Long training with augmentation closes the gap, and
pretraining on far more data reverses it, which is why ViTs are mostly used
pretrained.

## Exercises

1. [code] **Parameters and compute.** The ViT of
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
       with `learned_runs` and `cnn_runs`. Which of the two comparisons with the
       CNN, at equal parameters or at equal computation, does this run
       inform? Then resize the images to $48 \times 48$ instead and keep
       $p = 8$, which restores the $6 \times 6$ grid with each patch covering
       the same part of the original image. Predict the accuracy and the
       time per epoch before you train this model, and explain what the
       result says about the upsampling to $96 \times 96$.
   :exlabel:`ex_vision-transformer-parameters-compute`

1. [code] **Learning rates.** The text uses the best of three peak learning
   rates, $\{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$, for each model.
   Repeat the sweep for the ViT and for `CompactResNet`. Which model is more
   sensitive to the rate? Would the comparison have come out differently if
   both had used the CNN's rate, and which protocol is the fairer one?

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
       validation accuracy with the two runs in `learned_runs`.
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
    1. Train a `ViT` whose patch position embeddings are fixed to the 2-D
       sinusoidal code `fixed` of :numref:`subsec_vit-pos-grid`, with a
       learned embedding for the “&lt;cls&gt;” slot. Compare its validation
       accuracy with `learned_runs` and `zero_runs`, and measure its
       accuracy under the five permutations `perms`.
    1. A random permutation moves almost every block. Measure instead the
       accuracy of the trained `model` when only two blocks trade places:
       two horizontal neighbors, and two blocks at opposite corners. Then
       swap $k$ random pairs of blocks for $k = 1, 2, 4, 8$, with several
       draws for each $k$, and plot the accuracy against $k$. Does the
       model rely on the local layout or on the arrangement as a whole?
   :exlabel:`ex_vision-transformer-position-information`

1. [code] **Grid permutation test.** Let $\Delta$ be the difference
   between the first and the third value that `grid_similarity` returns for
   the update `learned - pos_init`, the mean similarity at grid distance 1
   minus that at grid distance 3.
    1. Permute the rows of `learned - pos_init` at random, which reassigns
       the updates to grid cells, recompute $\Delta$, and repeat 1,000 times.
       In what fraction of the permutations is $\Delta$ at least as large as
       for the trained `model`? Compare the spread of $\Delta$ under
       permutation with the spread of the `random` curve.
    1. $\Delta$ ignores distance 5. The text guesses that training moves
       the embeddings of patches that mostly show background, those of the
       outer columns and the corners, alike. Measure for each grid cell how
       often it holds garment pixels in the training images, design a
       statistic for the guess, and test it in the same way.
    1. Train a `ViT` with the protocol of :numref:`subsec_vit-first-run`
       for 30 epochs instead of 10, with `num_steps` set to 30 epochs of
       minibatches, and repeat the test. How do $\Delta$ and the accuracy
       under `perms` change with longer training?
   :exlabel:`ex_vision-transformer-grid-test`

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
    1. Repeat the measurement for `zero_model`, whose position embeddings
       are fixed at zero. What other than position embeddings can make a
       head attend to nearby patches, and how do the results of the two
       models separate the two causes?

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
       the two runs in `learned_runs`.
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
[Dive into Deep Learning · §11.6]{.kicker}

Vision transformer<br>
**patches as tokens · class token and position embeddings · does it use position? · a matched comparison with a CNN**
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
- test whether a trained ViT uses the layout of its patches, and whether
  it needs it;
- read a ViT-versus-CNN comparison matched in parameters, FLOPs, data,
  and recipe;
- decide when to train a CNN and when to fine-tune a pretrained ViT.

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
| The KV Cache | reuse keys and values while sampling |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| **Vision Transformer** | **images as token sequences** |
| Scaling Laws | loss against parameters, data, compute |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | the settings of released models |
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
  exactly once; windows never overlap.
- Output channel $k$ at a position $=$ inner product of kernel $k$ with
  that patch $=$ entry $k$ of the patch's linear projection.
- The $d$ kernels, each $c \times p \times p$: the rows of the
  $d \times cp^2$ projection matrix, reshaped.

::: {.d2l-note}
Fashion-MNIST's $28 \times 28$ upsampled to $96 \times 96$, $p = 16$: 36
tokens, each covering under five original pixels per side. The upsampling
adds no information.
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
:::

::: {.slide title="The ViT Block Is the Original Pre-Norm Encoder Block"}
$$\mathbf{X} \leftarrow \mathbf{X} + \mathrm{MHA}(\mathrm{LN}(\mathbf{X})), \qquad
\mathbf{X} \leftarrow \mathbf{X} + \mathrm{MLP}(\mathrm{LN}(\mathbf{X}))$$

- LN: LayerNorm; MHA: multi-head self-attention, **no mask**.
- MLP: Linear $d \to 4d$, GELU, Linear $4d \to d$.
- Kept from the original paper: LayerNorm, not RMSNorm; GELU, not SwiGLU.
  Nothing about images calls for either.
- Output shape = input shape, so blocks stack.
:::

::: {.slide title="A Learned Class Token Collects the Evidence for the Head"}
- Prepend one learned vector, initialized to zeros: 37 tokens.
- In every block it attends to all 36 patches.
- The head (LayerNorm + Linear) reads only its final representation.
- Averaging the 36 patch outputs instead works about as well: the class
  token is a convention, not a discovery.
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

::: {.slide title="A Fixed Code Needs Rows and Columns; a Learned Table Can Find Them"}
Without positions, attention is permutation equivariant: an image and any
rearrangement of its patches look the same to the head.

| code | how |
|:--|:--|
| learned (original ViT) | one free vector per position, from noise |
| 2-D sinusoidal | half the vector codes the row, half the column |
| 2-D RoPE | rotate queries and keys by row and column angles |

A 1-D code in raster order would make patch 6 (end of row 1) and patch 7
(start of row 2) neighbors. The learned table must discover the grid from
data.

::: {.d2l-note}
Original ablation: learned 1-D, 2-D, and relative codes about equal; no
positions at all, two to three points lower.
:::
:::

::: {.slide title="SGD Costs a ViT-Base Seven Points on ImageNet"}
ViT-Base (12 blocks, width 768, 86M parameters) in the DeiT study,
everything else equal: AdamW 81.8%, SGD 74.5%. ResNets train well with SGD.

Our recipe (both models):

- AdamW, weight decay 0.05 on weight matrices and kernels only (no
  decay on biases, norms, class token, position embeddings);
- learning rate from 0 up to a tuned peak linearly over the first epoch,
  then a half-cosine to 0; peak $3 \times 10^{-4}$ for the ViT, $10^{-3}$
  for the CNN;
- ten epochs, minibatch 128, no augmentation.

[Touvron et al., ICML 2021]{.small}
:::

::: {.slide title="Ten Epochs Take the 6.5M-Parameter ViT to About 89%"}
@!vision-transformer-a-first-run-2
:::

::: {.slide title="Check: Shuffle the Patches of Every Test Image"}
Rearrange the 36 blocks of $16 \times 16$ pixels by a fixed random
permutation. What accuracy if the ViT ignores position? If every decision
rests on the layout?

. . .

- Ignores position: unchanged, by permutation equivariance.
- Rests on the layout: 10%, the rate of guessing.
- A ViT whose position table is zero: unchanged, for certain.
:::

::: {.slide title="The ViT Uses the Layout of Its Patches but Barely Needs It"}
Five fixed shuffles of the trained ViT:

@!vision-transformer-does-the-vit-use-position-2

Two runs per arm, position table learned or held at zero:

@!vision-transformer-does-the-vit-use-position-4

::: {.d2l-note}
Using the layout is not needing it: most of the accuracy comes from what
the patches contain.
:::
:::

::: {.slide title="Check: What Would Random Embeddings Give?"}
We center the 36 position embeddings and average the cosine of all pairs at
each grid distance (max of row and column offset). Independent random
vectors in 512 dimensions: what curve, and how far from zero?

. . .

- A flat curve, slightly below zero: centered, the 36 vectors sum to
  zero, so their average pairwise cosine is about $-1/35 \approx -0.03$.
- One pair's cosine scatters by about $1/\sqrt{512} \approx 0.044$;
  averaging the many pairs at each distance shrinks that noise.
- Only a curve that falls with distance is evidence of a learned grid.
:::

::: {.slide title="Neighboring Position Embeddings Are About 0.1 More Alike in Cosine"}
@!vision-transformer-do-the-position-embeddings-discover-the-grid

- learned − initial: what training wrote, with the starting noise removed.
- The cosines describe the table; only the shuffle test asks the model.
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

::: {.slide title="Check: Halve the Patch Size"}
Our ViT with $p = 8$ instead of 16: what happens to the tokens, the
parameters, and the forward FLOPs?

. . .

- Tokens: $36 \to 144$ (plus the class token).
- Parameters: $6.46\text{M} \to 6.41\text{M}$, under 1% fewer.
- FLOPs: $0.48 \to 1.92$ GFLOPs per image, four times as many.
:::

::: {.slide title="The CNN Leads from the First Epoch and Ends About Five Points Ahead"}
@!vision-transformer-a-convolutional-baseline-at-the-same-budget-3

Two runs per model agree to a tenth of a point; the gap is about five.

::: {.d2l-note}
One protocol, one dataset: the gap does not say which property of the CNN
causes it. The position table is worth only one to two points.
:::
:::

::: {.slide title="Check: Who Gains More from 300 Epochs?"}
Both curves flatten because the learning rate reaches zero. Train compact
ViTs and a ResNet-18 for 300 epochs with heavy augmentation: which gains
more?

. . .

- Fashion-MNIST: ViTs under 4M parameters reach 93% ($p = 16$), 94.5%
  ($p = 8$), 95% ($p = 4$); ResNet-18 (11M) 94.8%.
- Our CNN reached about 94% in ten epochs, our ViT about 89%: the
  transformer gains far more, and the gap closes.
- Our runs hint at it: after ten epochs the CNN's training loss is near
  zero, the ViT's is not.

[Hassani et al., 2021]{.small}
:::

::: {.slide title="More Data Reverses the Ranking"}
- CIFAR-10, 300 epochs: the same ViTs 78% to 94%, ResNet-18 90%; only the
  finest patches win.
- JFT-300M subsets: ResNets lead on small ones, ViTs on large ones.

::: {.d2l-note}
Small data, from scratch: train a CNN, or fine-tune a pretrained ViT.
:::

[Hassani et al., 2021; Dosovitskiy et al., 2021]{.small}
:::

::: {.slide title="Wrap-up"}
- ViT = strided-convolution patchify, the pre-norm encoder, a head on the
  class token; only tokenization and readout are new.
- Shuffling the patches drops the trained ViT from about 89% to below one
  half; a ViT with no positions at all reaches about 87 to 88%.
- Learned position embeddings: the cosine of neighbors about 0.1 higher,
  against a flat random reference.
- Matched in parameters, FLOPs, data, and recipe, the CNN leads by about
  five points, against a run-to-run range of a tenth of a point; long
  recipes close the gap, and far more data reverses it.

**Next:** Scaling Laws, sizing a model before the run.
:::
