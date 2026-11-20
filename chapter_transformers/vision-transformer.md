# Vision Transformer (ViT)
:label:`sec_vision-transformer`

Convolutional neural networks (:numref:`chap_cnn`, :numref:`chap_modern_cnn`) bake two geometric priors about images directly into their architecture: **spatial locality** (a $k \times k$ kernel couples only neighboring pixels) and **translation equivariance** (the same kernel weights slide across every $(r, c)$ location). Can a standard bidirectional Transformer encoder (:numref:`sec_transformer`)—which has *no* built-in notion of 2D grid neighbors or shift equivariance—classify images competitively?

Applying self-attention directly to individual pixels is quadratic in pixel count: even a $96 \times 96$ image has $9{,}216$ pixels ($9216^2 \approx 8.5 \times 10^7$ attention pairs per head). :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` introduced the **Vision Transformer (ViT)** (:numref:`fig_vit`) by tokenizing an image into a grid of non-overlapping $p \times p$ **patches**: a $96 \times 96$ image with patch size $p = 16$ becomes a short sequence of $m = (96 / 16)^2 = 36$ patch tokens. Each patch is projected to width $d$, a learnable `<cls>` token is prepended, learned 1D positional embeddings are added, and the 37 tokens pass through $L$ bidirectional pre-norm Transformer blocks.

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

## Patches as Tokens (`PatchEmbedding`)

Partitioning an image tensor of shape $(B, C, H, W)$ into non-overlapping $p \times p$ patches and applying a shared linear projection $\mathbb{R}^{C p^2} \to \mathbb{R}^d$ to each patch is identical to a **2D convolution with kernel size $p \times p$, stride $p$, and $d$ output channels**, followed by flattening the 2D spatial grid of height $H/p$ and width $W/p$ into a sequence of $m = (H/p)(W/p)$ token vectors:

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

## The Vision Transformer Architecture

![The vision transformer on a $96 \times 96$ image. A strided convolution turns the $6 \times 6$ grid of $16 \times 16$ patches into 36 tokens. A learned “&lt;cls&gt;” token is prepended and learned position embeddings are added; $L$ pre-norm transformer blocks process all 37 tokens, and a linear head classifies the final representation of the “&lt;cls&gt;” token.](../img/mdl-transformers-vit.svg)
:label:`fig_vit`

### The Encoder Block (`ViTBlock`)

Following :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`, `ViTBlock` subclasses our configurable `d2l.TransformerBlock` (:numref:`sec_transformer-block`) with pre-norm `LayerNorm`, biases enabled, and a two-layer `GELU` MLP of custom hidden width `mlp_num_hiddens`:

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

### The `<cls>` Token, Position Embeddings, and `ViT`

To aggregate the $m$ patch representations into a single image-level vector without favoring any spatial location, `ViT` prepends a learnable parameter vector `cls_token` of shape $(1, 1, d)$ at index $0$, adds a learnable positional embedding table `pos_embedding` of shape $(1, m + 1, d)$ across the $m + 1$ positions, runs $L$ `ViTBlock` layers, normalizes with `LayerNorm`, and feeds index $0$ (`X[:, 0]`) into a linear classification head:

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

We train a 2-block `ViT` ($d = 512, d_{\textrm{mlp}} = 2048, H = 8, L = 2, p = 16$, $\approx 6.4\textrm{M}$ parameters) on $96 \times 96$ Fashion-MNIST for 10 epochs using AdamW ($\lambda = 0.05$) with linear warmup and cosine learning-rate decay:

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

### The Role of Spatial Position and Learned 2D Grid Topology
:label:`subsec_vit-pos-grid`

Without `pos_embedding`, `ViT` is **permutation-invariant** over the $m = 36$ patches: any permutation of the 36 patches simply permutes rows $1 \dots 36$ of every attention matrix and leaves row $0$ (`<cls>`) literally unchanged. With learned `pos_embedding`, patch $k$ receives positional vector $\mathbf{p}_k$ according to the slot $k$ where it lands. We can test how much the trained `ViT` relies on spatial layout by permuting the 36 patches of each validation image across 5 random permutations at test time (`shuffled_accuracy`):

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

```{.python .input #vision-transformer-does-the-vit-use-position-2}
%%tab pytorch, jax
rng = np.random.default_rng(0)
perms = [rng.permutation(36) for _ in range(5)]  # Five fixed shuffles
accs = [shuffled_accuracy(model, data, perm) for perm in perms]
print(f'trained ViT: {vit_acc:.3f} as is, '
      f'{min(accs):.2f} to {max(accs):.2f} with its patches shuffled')
```

Scrambling the $6 \times 6$ patch arrangement drops validation accuracy by roughly 25 percentage points (whereas retraining `ViT` from scratch with `pos_embedding` frozen to zero loses only $\sim 1.5$ percentage points on Fashion-MNIST because local $16 \times 16$ patch textures—like shoe heels or collar shapes—already identify many coarse garment classes as a "bag of patches").

Does the unstructured 1D positional table `pos_embedding` ($36 \times d$) actually discover the 2D geometry of the $6 \times 6$ image grid? Computing the mean cosine similarity between patch position vectors as a function of their 2D Manhattan grid distance $|r - r'| + |c - c'|$ on the $6 \times 6$ grid (`grid_similarity`) confirms that nearby 2D patches learn positively correlated position vectors while distant patches become nearly orthogonal:

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

## Comparison with a Parameter-Matched Convolutional Network
:label:`subsec_vit-cnn-baseline`

How does our 6.4M-parameter `ViT` compare on this 60,000-image dataset against a parameter-matched ResNet-style convolutional baseline (`CompactResNet`, $\approx 6.4\textrm{M}$ parameters) trained for 10 epochs with the exact same cosine AdamW schedule?

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

```{.python .input #vision-transformer-a-convolutional-baseline-at-the-same-budget-3}
%%tab pytorch, jax
d2l.plot(list(range(1, 11)), [vit_history['val_acc'], cnn_history['val_acc']],
         'epoch', 'validation accuracy', legend=['ViT', 'CNN'])
```

On a small 60,000-image dataset without data augmentation or large-scale pretraining, `CompactResNet` converges faster and finishes roughly 2–3 percentage points ahead of `ViT`. This illustrates the fundamental trade-off identified by :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`: a CNN's hard-coded 2D locality and translation equivariance provide a strong **inductive bias** in the small-data regime, whereas a Vision Transformer learns spatial relationships flexibly from data—trailing CNNs on small unaugmented datasets, but matching or surpassing them when pretrained on tens of millions of images (or trained with strong augmentation, distillation :cite:`Touvron.Cordova.Douze.ea.2021`, or masked patch reconstruction :cite:`He.Chen.Xie.ea.2022`).

## Summary

* **Vision Transformers (ViT)** convert an image of shape $(C, H, W)$ into a sequence of $m = (H/p)(W/p)$ tokens via a stride-$p$ convolution (`PatchEmbedding`), prepend a `<cls>` token, add learned 1D positional embeddings, and apply a bidirectional pre-norm Transformer encoder (:numref:`fig_vit`).
* Without positional embeddings, a ViT is permutation-invariant over its patches; with learned 1D positional embeddings, gradient descent spontaneously recovers the 2D Euclidean distance structure of the patch grid.
* Because ViT lacks the built-in 2D locality and translation equivariance of convolutions, a parameter-matched ResNet outperforms ViT when trained from scratch on small unaugmented datasets, while ViT excels at large pretraining scales and unified multimodal architectures.

## Exercises

1. **Patch size $p$, parameter count, and attention FLOPs.** In our $96 \times 96$ `ViT`, the patch size is $p = 16$ ($m = 36$ patches).
   1. How do the parameter count of `PatchEmbedding`, the parameter count of `pos_embedding`, and the parameter count of the $L$ `ViTBlock` layers change when you halve the patch size to $p = 8$ ($m = 144$) or quarter it to $p = 4$ ($m = 576$)?
   2. By what factor does the per-layer attention score matrix size $(m + 1)^2$ grow from $p = 16$ to $p = 8$ and $p = 4$? Train the $p = 8$ `ViT` for 10 epochs and compare its wall-clock time and validation accuracy against $p = 16$.
:exlabel:`ex_vision-transformer-parameters-compute`

2. **`<cls>` readout vs. global average pooling.** Replace the `<cls>` token readout in `ViT` with global average pooling across the $m = 36$ patch tokens (`self.head(self.ln(X[:, 1:].mean(dim=1)))`, or removing `cls_token` altogether and averaging all 36 tokens). Train for 10 epochs on Fashion-MNIST and compare validation accuracy against `<cls>` readout.

3. **Exact permutation invariance without positional embeddings.** Set `model.pos_embedding` to all zeros and switch `model` to evaluation mode (disabling dropout). Prove algebraically—and verify numerically on a validation batch to within $10^{-6}$—that applying `shuffle_patches(X, perm)` for any permutation `perm` of $\{0, \dots, 35\}$ leaves the output logits `model(X)` completely unchanged.
:exlabel:`ex_vision-transformer-position-information`

4. **Statistical permutation test for 2D grid discovery.** Let $\Delta$ be the difference in `grid_similarity` between adjacent patches (grid distance 1) and patches at grid distance $\ge 4$ for the trained `model.pos_embedding`. Compute the null distribution of $\Delta$ across 1,000 random permutations of the 36 rows of `pos_embedding`, and report the empirical $p$-value showing that the learned positional embeddings align with the 2D grid topology.
:exlabel:`ex_vision-transformer-grid-test`

5. **Sub-patch vs. whole-patch translation sensitivity.** In a CNN, shifting an input image by $s$ pixels shifts the internal feature maps by $s$ pixels (up to pooling stride). In `ViT` ($p = 16$), shifting an image by $s = 16$ pixels (one full patch width) merely shifts each patch into the adjacent patch slot, whereas shifting by $s = 8$ pixels (half a patch) cuts across every $16 \times 16$ patch boundary. Evaluate the validation accuracy of our trained `ViT` and `CompactResNet` on validation images shifted right by $s \in \{0, 2, 4, 8, 12, 16\}$ pixels (zero-padding the left border). Why does `ViT` show a non-monotonic dip at $s = 8$ relative to $s = 16$?

6. **Mean attention distance across depth.** :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` define a head's **mean attention distance** as the attention-weighted average 2D Euclidean pixel distance $\sum_{i,j=1}^{36} \alpha_{ij} \, d_{\textrm{pixel}}(i, j) / 36$ between query patch $i$ and key patch $j$. Compute the mean attention distance for each of the 8 heads in blocks 1 and 2 of our trained `ViT`. Do early-layer heads attend more locally than deeper-layer heads, analogous to the growing receptive field of a CNN?

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.6]{.kicker}

Vision Transformer<br>
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
| The Transformer Block | attention + FFN, residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| The KV Cache | reuse past keys and values |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| **Vision Transformer** | **images as token sequences** |
| Scaling Laws | loss against parameters, data, compute |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | released models as settings |
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

::: {.slide title="Patches Make an Image a Sequence; at Small Scale the CNN Still Wins"}
[Wrap-up]{.kicker}

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
