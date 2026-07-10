```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# ConvNeXt: A ConvNet for the 2020s
:label:`sec_convnext`

By 2021, vision transformers :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` had taken over the top of the ImageNet leaderboard, and hierarchical variants such as the Swin transformer :cite:`liu2021swin` were displacing convnets as the default backbone for detection and segmentation. It was tempting to conclude that attention had made convolution obsolete. But a transformer differs from a 2015 ResNet in many ways besides attention: its training recipe, its stem, its activation function, its normalization layers, the shape and depth of its stages. Which of these differences actually carried the improvement?

:citet:`liu2022convnet` answered by experiment. Starting from a ResNet-50, they applied one transformer-inspired design change at a time, measured ImageNet accuracy after each, kept what helped, and never added attention. The end point, named ConvNeXt, reaches 82.0% top-1 accuracy where the Swin-T transformer of the same computational cost reaches 81.3%. The exercise is the best guided tour of modern architecture design we know of, because every step uses a concept this book has already taught: training recipes (:numref:`sec_training_recipes`), depthwise convolutions (:numref:`sec_depthwise_separable`), the inverted bottleneck (:numref:`sec_efficient_cnns`), receptive fields (:eqref:`eq_receptive_field`), and normalization layers (:numref:`sec_batch_norm`). In this section we walk the roadmap, implement the result, and train a scaled-down ConvNeXt with the modern recipe of :numref:`sec_training_recipes`.

```{.python .input #convnext-imports}
%%tab mxnet
from d2l import mxnet as d2l
import math
import mxnet as mx
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #convnext-imports}
%%tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #convnext-imports}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #convnext-imports}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## The Modernization Roadmap

:numref:`tab_convnext_roadmap` shows the whole journey. Each row keeps every change above it, so the accuracies are cumulative; the computational cost is held near that of Swin-T (about 4.5 GFLOPs) throughout.

:The ConvNeXt modernization roadmap :cite:`liu2022convnet`. ImageNet-1K top-1 accuracy of a ResNet-50 as design changes accumulate; each row includes all rows above it, at roughly constant cost. Swin-T, the transformer of the same cost, reaches 81.3%.
:label:`tab_convnext_roadmap`

| Change | What it means | Top-1 |
|:--|:--|:--|
| (starting point) | ResNet-50 with its 2015 recipe | 76.1% |
| Modern training recipe | 300 epochs, AdamW, Mixup, label smoothing, stochastic depth (:numref:`sec_training_recipes`) | 78.8% |
| Stage ratio | blocks per stage $(3, 4, 6, 3) \to (3, 3, 9, 3)$ | 79.4% |
| Patchify stem | $7 \times 7$ stride-2 convolution plus pooling $\to$ one $4 \times 4$ stride-4 convolution | 79.5% |
| Depthwise convolutions | the $3 \times 3$ convolution becomes depthwise; width grows from 64 to 96 channels | 80.5% |
| Inverted bottleneck | expand $4\times$ inside the block instead of compressing | 80.6% |
| Large kernels | depthwise convolution moved to the front and enlarged to $7 \times 7$ | 80.6% |
| ReLU $\to$ GELU | a smooth activation, as in transformers | 80.6% |
| Fewer activations | one activation per block instead of three | 81.3% |
| Fewer normalizations | one normalization per block instead of three | 81.4% |
| BN $\to$ LN | layer normalization over channels at each position | 81.5% |
| Separate downsampling | an LN plus $2 \times 2$ stride-2 convolution between stages | 82.0% |

The first row is the largest single jump, and it is not an architecture change at all. The 2.7 points it contributes are the recipe effect of :numref:`sec_training_recipes` (the "ResNet strikes back" study pushed the same network to 80.4% with a still longer 600-epoch procedure :cite:`wightman2021resnet`). Keep this split in mind for everything below: of the 5.9 points separating the 2015 ResNet-50 from ConvNeXt, 2.7 are training and 3.2 are architecture. Papers that compared a well-tuned transformer against a 2015-recipe ResNet were, in large part, comparing recipes.

### Macro design

The next two rows change the network's silhouette. Swin-T distributes its blocks across the four stages in the ratio $1{:}1{:}3{:}1$, spending most of its depth where the feature maps are small and each block is cheap; adopting the same ratio, $(3, 3, 9, 3)$ blocks instead of ResNet-50's $(3, 4, 6, 3)$, adds 0.6 points. The stem changes more. ResNet begins with a $7 \times 7$ stride-2 convolution followed by max-pooling, a fourfold reduction computed with overlapping windows. Vision transformers instead slice the image into non-overlapping patches, which is just a strided convolution whose kernel size equals its stride (:numref:`sec_padding`): a $4 \times 4$ convolution with stride 4. This *patchify* stem replaces the ResNet stem with no loss (79.5%), and its output, one feature vector per $4 \times 4$ patch, is the same object a transformer calls a patch embedding.

### Depthwise convolutions and the inverted bottleneck

The self-attention layer of a transformer mixes information *across positions* but treats each channel separately; the MLP that follows it mixes *across channels* but treats each position separately. Convnets have owned this factorization since MobileNet: it is exactly a depthwise convolution followed by $1 \times 1$ convolutions (:numref:`sec_depthwise_separable`). Making the ResNet bottleneck's $3 \times 3$ convolution depthwise, and spending the savings on width (64 to 96 channels, matching Swin-T), brings 80.5%. Inverting the bottleneck, so that the block expands its thin residual stream by $4\times$, works in the wide space, and projects back, adds a little more (80.6%). We saw in :numref:`sec_efficient_cnns` why this shape wins once the spatial convolution is depthwise: wide tensors are cheap to filter depthwise, and the tensors that persist across blocks stay thin. The transformer's MLP block, which expands by the same factor of four, has the identical structure.

With the expensive dense convolution gone, kernel size stops being a luxury. Moving the depthwise convolution to the front of the block (so the cheap operation sees the unexpanded stream; this alone temporarily costs 0.7 points) and enlarging it from $3 \times 3$ to $7 \times 7$ recovers the loss at lower cost. By :eqref:`eq_receptive_field`, a $7 \times 7$ kernel grows the receptive field as fast as three stacked $3 \times 3$ layers, the trade VGG once resolved in favor of depth (:numref:`sec_vgg`) when kernels were dense and their cost quadratic. Depthwise, a $7 \times 7$ kernel costs only $49/9 \approx 5\times$ a $3 \times 3$ one on the depthwise term alone, a small fraction of the block. The authors report that accuracy saturates at $7 \times 7$: kernels of $9 \times 9$ and $11 \times 11$ buy nothing further at this scale.

### Micro design

The remaining rows are small and, cumulatively, decisive. Replacing ReLU by GELU, the smooth activation used in transformers and in essentially every large language model, changes nothing by itself (80.6%). What matters is *how many* activations and normalizations the block contains. A ResNet bottleneck interleaves a normalization and an activation after every convolution: three of each. A transformer block applies one activation, inside its MLP, and one or two normalizations. Deleting all but one GELU (between the two $1 \times 1$ convolutions) adds 0.7 points, more than any other micro-design change; deleting all but one normalization adds a little more, and swapping that survivor from batch normalization to layer normalization, applied over the channels at each spatial position, another tenth (81.5%). The lesson of :numref:`sec_batch_norm` reappears: normalization is scaffolding for optimization, and less of it, placed well, can be better. :numref:`fig_resnet_vs_convnext_block` shows the before and after.

![The ResNet-50 bottleneck block and the ConvNeXt block that the roadmap turns it into. Three normalizations and three activations become one of each; the compress-process-expand bottleneck becomes an inverted bottleneck led by a $7 \times 7$ depthwise convolution.](../img/arch-resnet-vs-convnext-block.svg)
:label:`fig_resnet_vs_convnext_block`

Finally, ResNet downsamples inside the first block of each stage, by giving its $3 \times 3$ convolution a stride of 2. Swin instead uses a separate patch-merging layer between stages. Adopting the same separation, a layer normalization followed by a $2 \times 2$ stride-2 convolution between stages (plus a normalization after the stem and one before the classifier head, which the authors found necessary for stable training), completes the roadmap at 82.0%. :numref:`fig_convnext` shows the assembled ConvNeXt-T: a patchify stem, four stages of $(3, 3, 9, 3)$ blocks at widths $(96, 192, 384, 768)$, downsampling layers between them, and a global-average-pooling head, the stem-body-head layout this chapter has used since :numref:`sec_googlenet`.

![ConvNeXt-T. A patchify stem replaces convolution-plus-pooling; stages of 3, 3, 9, and 3 blocks are joined by separate LN plus $2 \times 2$ stride-2 downsampling layers; there is no pooling anywhere in the body.](../img/arch-convnext.svg)
:label:`fig_convnext`

Where does this leave the convnet-versus-transformer question? At this scale, even: a pure convnet, given the transformer's recipe and design sensibilities, matches or slightly beats the transformer, and the paper shows the result persists for larger variants and transfers to COCO detection and ADE20K segmentation, the benchmarks that :numref:`sec_training_recipes` argued still discriminate. What the roadmap does not show is a convnet *advantage*: the two families, tuned by the same hands, land in the same place. We return to what that means in :numref:`sec_cnn-design`.

## Implementation

ConvNeXt is easy to build precisely because the roadmap removed things. A block is six operations; the network is the familiar arch-tuple loop.

### The ConvNeXt block

The block applies, in order: a $7 \times 7$ depthwise convolution, layer normalization, a $1 \times 1$ convolution expanding the channels $4\times$, a GELU, and a $1 \times 1$ convolution projecting back, with the result added to the input. Two refinements from the paper's training setup come along. *Layer scale* multiplies the branch by a learnable per-channel vector $\gamma$ initialized to $10^{-6}$, so every block starts as a near-identity and the network begins training as a shallow function that gradually deepens; the technique was introduced for very deep vision transformers :cite:`touvron2021cait` and helps stability here too. *Stochastic depth*, :eqref:`eq_stochastic_depth` of :numref:`sec_training_recipes`, randomly drops the whole branch per sample during training.

One implementation detail deserves attention, because it is the layer-normalization placement that the roadmap's LN row depends on. ConvNeXt normalizes over the *channels at each spatial position*, the same normalization a transformer applies to each token. In a channels-last layout (samples, height, width, channels) this is the library default, and a $1 \times 1$ convolution is a plain linear layer applied to the last axis, the observation of :numref:`sec_nin` in reverse. Our PyTorch implementation therefore permutes to channels-last inside the block, uses `nn.LayerNorm` and `nn.Linear` directly, and permutes back; TensorFlow and JAX are channels-last natively; MXNet stays channels-first and tells `nn.LayerNorm` to normalize `axis=1`.

```{.python .input #convnext-block}
%%tab pytorch
def drop_path(Y, p, training):  # Stochastic depth, as in the previous section
    if not training or p == 0:
        return Y
    keep = (torch.rand(Y.shape[0], 1, 1, 1, device=Y.device) > p).float()
    return Y * keep / (1 - p)

class ConvNeXtBlock(nn.Module):
    """Depthwise 7x7, LN, 1x1 expand, GELU, 1x1 project, scaled residual."""
    def __init__(self, dim, drop_prob=0.0, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3,
                                groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 1x1 conv in channels-last
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_prob = drop_prob

    def forward(self, X):
        Y = self.dwconv(X).permute(0, 2, 3, 1)  # to channels-last
        Y = self.pwconv2(F.gelu(self.pwconv1(self.norm(Y))))
        Y = (self.gamma * Y).permute(0, 3, 1, 2)  # back to channels-first
        return X + drop_path(Y, self.drop_prob, self.training)
```

```{.python .input #convnext-block}
%%tab mxnet
class ConvNeXtBlock(nn.Block):
    """Depthwise 7x7, LN, 1x1 expand, GELU, 1x1 project, scaled residual."""
    def __init__(self, dim, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(axis=1, epsilon=1e-6)
        self.pwconv1 = nn.Conv2D(4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2D(dim, kernel_size=1)
        self.gamma = gluon.Parameter('gamma', shape=(1, dim, 1, 1),
                                     init=init.Constant(layer_scale))

    def forward(self, X):
        Y = self.norm(self.dwconv(X))
        Y = self.pwconv2(self.act(self.pwconv1(Y)))
        return X + self.gamma.data() * Y
```

```{.python .input #convnext-block}
%%tab tensorflow
class ConvNeXtBlock(tf.keras.layers.Layer):
    """Depthwise 7x7, LN, 1x1 expand, GELU, 1x1 project, scaled residual."""
    def __init__(self, dim, layer_scale=1e-6):
        super().__init__()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=7,
                                                      padding='same')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = tf.keras.layers.Dense(4 * dim)
        self.pwconv2 = tf.keras.layers.Dense(dim)
        self.gamma = self.add_weight(
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(layer_scale))

    def call(self, X):
        Y = self.norm(self.dwconv(X))
        Y = self.pwconv2(tf.keras.activations.gelu(self.pwconv1(Y)))
        return X + self.gamma * Y
```

```{.python .input #convnext-block}
%%tab jax
class ConvNeXtBlock(nn.Module):
    """Depthwise 7x7, LN, 1x1 expand, GELU, 1x1 project, scaled residual."""
    dim: int
    layer_scale: float = 1e-6

    @nn.compact
    def __call__(self, X):
        Y = nn.Conv(self.dim, kernel_size=(7, 7), padding='same',
                    feature_group_count=self.dim)(X)
        Y = nn.LayerNorm(epsilon=1e-6)(Y)
        Y = nn.Dense(self.dim)(nn.gelu(nn.Dense(4 * self.dim)(Y)))
        gamma = self.param('gamma',
                           nn.initializers.constant(self.layer_scale),
                           (self.dim,))
        return X + gamma * Y
```

### The full network

The network is the arch-tuple pattern of :numref:`sec_vgg` one more time: a tuple of (depth, channels) pairs, a stem, a loop, a head. The stem is the patchify convolution plus a normalization. Between stages sits the separate downsampling layer, a normalization followed by a $2 \times 2$ stride-2 convolution; there is no pooling anywhere in the body. Following the paper, the per-block stochastic-depth probability ramps linearly from zero at the first block to its maximum at the last. Our dimensions are a scaled-down ConvNeXt sized for Fashion-MNIST, near the "atto" end of the family: widths $(40, 80, 160, 320)$ and depths $(2, 2, 6, 2)$, with the third stage deepest as in the full model.

```{.python .input #convnext-model}
%%tab pytorch
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel axis of an NCHW tensor."""
    def forward(self, X):
        return super().forward(X.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ConvNeXt(d2l.Classifier):
    arch = ((2, 40), (2, 80), (6, 160), (2, 320))

    def __init__(self, lr=2e-3, num_classes=10, drop_path_max=0.1):
        super().__init__()
        self.save_hyperparameters()
        depths = [d for d, c in self.arch]
        rates = torch.linspace(0, drop_path_max, sum(depths)).tolist()
        layers = [nn.Conv2d(1, self.arch[0][1], kernel_size=4, stride=4),
                  LayerNorm2d(self.arch[0][1], eps=1e-6)]
        c_prev, b = self.arch[0][1], 0
        for i, (depth, c) in enumerate(self.arch):
            if i > 0:  # separate downsampling layer between stages
                layers += [LayerNorm2d(c_prev, eps=1e-6),
                           nn.Conv2d(c_prev, c, kernel_size=2, stride=2)]
            for _ in range(depth):
                layers.append(ConvNeXtBlock(c, drop_prob=rates[b]))
                b += 1
            c_prev = c
        layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                   nn.LayerNorm(c_prev, eps=1e-6),
                   nn.Linear(c_prev, num_classes)]
        self.net = nn.Sequential(*layers)
```

```{.python .input #convnext-model}
%%tab mxnet
class ConvNeXt(d2l.Classifier):
    def __init__(self, arch=((2, 40), (2, 80), (6, 160), (2, 320)),
                 lr=2e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Conv2D(arch[0][1], kernel_size=4, strides=4),
                     nn.LayerNorm(axis=1, epsilon=1e-6))
        for i, (depth, c) in enumerate(arch):
            if i > 0:  # separate downsampling layer between stages
                self.net.add(nn.LayerNorm(axis=1, epsilon=1e-6),
                             nn.Conv2D(c, kernel_size=2, strides=2))
            for _ in range(depth):
                self.net.add(ConvNeXtBlock(c))
        self.net.add(nn.GlobalAvgPool2D(), nn.Flatten(),
                     nn.LayerNorm(axis=-1, epsilon=1e-6),
                     nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
```

```{.python .input #convnext-model}
%%tab tensorflow
class ConvNeXt(d2l.Classifier):
    def __init__(self, arch=((2, 40), (2, 80), (6, 160), (2, 320)),
                 lr=2e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(arch[0][1], kernel_size=4, strides=4),
            tf.keras.layers.LayerNormalization(epsilon=1e-6)])
        for i, (depth, c) in enumerate(arch):
            if i > 0:  # separate downsampling layer between stages
                self.net.add(tf.keras.layers.LayerNormalization(
                    epsilon=1e-6))
                self.net.add(tf.keras.layers.Conv2D(c, kernel_size=2,
                                                    strides=2))
            for _ in range(depth):
                self.net.add(ConvNeXtBlock(c))
        self.net.add(tf.keras.layers.GlobalAvgPool2D())
        self.net.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))
        self.net.add(tf.keras.layers.Dense(num_classes))
```

```{.python .input #convnext-model}
%%tab jax
class ConvNeXt(d2l.Classifier):
    arch: tuple = ((2, 40), (2, 80), (6, 160), (2, 320))
    lr: float = 2e-3
    num_classes: int = 10

    def setup(self):
        layers = [nn.Conv(self.arch[0][1], kernel_size=(4, 4),
                          strides=(4, 4)),
                  nn.LayerNorm(epsilon=1e-6)]
        for i, (depth, c) in enumerate(self.arch):
            if i > 0:  # separate downsampling layer between stages
                layers += [nn.LayerNorm(epsilon=1e-6),
                           nn.Conv(c, kernel_size=(2, 2), strides=(2, 2))]
            layers += [ConvNeXtBlock(c) for _ in range(depth)]
        layers += [lambda x: x.mean(axis=(1, 2)),  # global average pooling
                   nn.LayerNorm(epsilon=1e-6),
                   nn.Dense(self.num_classes)]
        self.net = nn.Sequential(layers)
```

:begin_tab:`jax`
Unlike every network since :numref:`sec_batch_norm`, this model carries no batch statistics: layer normalization is a pure function of its input. The Flax module therefore needs no `training` flag and no mutable `batch_stats` collection, which is why the class above is shorter than its ResNet counterpart. (The stochastic-depth wrapper, which does need per-sample randomness, appears in the PyTorch tab; see the note in :numref:`sec_training_recipes`.)
:end_tab:

A $96 \times 96$ input leaves the stem as a $24 \times 24$ map, and the three downsampling layers reduce it to $12 \times 12$, $6 \times 6$, and finally $3 \times 3$ before the head pools it away. We check the output shape and count parameters: 3,376,450, about a third of the 11.2 million in the ResNet-18 we trained in :numref:`sec_training_recipes`. The exact count is a stringent correctness check for any reimplementation, ours included, since a single wrongly sized layer changes it.

```{.python .input #convnext-params}
%%tab pytorch
model = ConvNeXt()
X = torch.randn(1, 1, 96, 96)
assert model.net(X).shape == (1, 10)
sum(p.numel() for p in model.parameters())
```

```{.python .input #convnext-params}
%%tab mxnet
model = ConvNeXt()
X = np.random.normal(0, 1, (1, 1, 96, 96))
assert model.net(X).shape == (1, 10)
sum(p.data().size for p in model.collect_params().values()
    if p.grad_req != 'null')
```

```{.python .input #convnext-params}
%%tab tensorflow
model = ConvNeXt()
X = tf.random.normal((1, 96, 96, 1))
assert model.net(X).shape == (1, 10)
sum(int(tf.size(w)) for w in model.net.trainable_weights)
```

```{.python .input #convnext-params}
%%tab jax
model = ConvNeXt()
X = jnp.zeros((1, 96, 96, 1))
params = model.init(d2l.get_key(), X)
assert model.apply(params, X).shape == (1, 10)
sum(p.size for p in jax.tree_util.tree_leaves(params['params']))
```

### Training with the modern recipe

A modernized architecture deserves the modernized recipe, and the roadmap's first row says it *needs* one: ConvNeXt was never trained any other way. We reuse the 2022-era recipe of :numref:`sec_training_recipes` verbatim: AdamW, a cosine schedule with warmup, label smoothing, and Mixup, in the same `RecipeTrainer` harness.

:begin_tab:`mxnet`
The training run appears in the PyTorch tab; the numbers quoted in the text come from that run. The ingredients carry over as described in :numref:`sec_training_recipes`: `mx.optimizer.AdamW` with a `CosineScheduler` for the optimizer and schedule, and the `mixup` function from that section for the data path.
:end_tab:

:begin_tab:`tensorflow`
The training run appears in the PyTorch tab; the numbers quoted in the text come from that run. The ingredients carry over as described in :numref:`sec_training_recipes`: `tf.keras.optimizers.AdamW` with a warmup `CosineDecay` schedule, label smoothing in the loss, and the `mixup` function from that section.
:end_tab:

:begin_tab:`jax`
The training run appears in the PyTorch tab; the numbers quoted in the text come from that run. The ingredients carry over as described in :numref:`sec_training_recipes`: `optax.adamw` chained with `optax.warmup_cosine_decay_schedule`, `optax.smooth_labels` for the loss, and the `mixup` function from that section.
:end_tab:

```{.python .input #convnext-recipe}
%%tab pytorch
def cosine_warmup(epoch, max_epochs, base_lr, warmup=3):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / (max_epochs - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))

def mixup(X, y, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(X.shape[0], device=X.device)
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam

class RecipeTrainer(d2l.Trainer):
    """A Trainer that sets the learning rate from the model's schedule."""
    def fit_epoch(self):
        for group in self.optim.param_groups:
            group['lr'] = cosine_warmup(self.epoch, self.max_epochs,
                                        self.model.lr)
        super().fit_epoch()

class ModernConvNeXt(ConvNeXt):
    """ConvNeXt under the modern recipe of the previous section."""
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr,
                                 weight_decay=0.05)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y, label_smoothing=0.1)

    def training_step(self, batch):
        X, y_a, y_b, lam = mixup(*batch, alpha=0.2)
        y_hat = self(X)
        l = lam * self.loss(y_hat, y_a) + (1 - lam) * self.loss(y_hat, y_b)
        self.plot('loss', l, train=True)
        return l
```

We train for 30 epochs on the full Fashion-MNIST training set at $96 \times 96$, matching the modern-recipe ResNet-18 run of :numref:`sec_training_recipes` epoch for epoch.

```{.python .input #convnext-train}
%%tab pytorch
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
model = ModernConvNeXt(lr=2e-3)
trainer = RecipeTrainer(max_epochs=30, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input #convnext-eval}
%%tab pytorch
model.eval()
with torch.no_grad():
    accs = [model.accuracy(model(X), y) for X, y in
            map(trainer.prepare_batch, data.val_dataloader())]
float(torch.stack(accs).mean())
```

On our machine this run reaches about 92.5% test accuracy (92.2% to 92.6% across three runs) in roughly 16 minutes; the wall time is dominated by the input pipeline, since a 3.4-million-parameter network leaves the GPU mostly idle at this image size. The three-times-larger ResNet-18 of :numref:`sec_training_recipes` reached 94.4% under the identical recipe and budget, so our modernized architecture arrives two points *behind*. The gap is not an artifact of our configuration: in pilot runs, initializing layer scale at $10^{-2}$ instead of $10^{-6}$, removing stochastic depth, and stretching the budget to 45 epochs each moved the result by less than the run-to-run spread. The plain reading is the right one. Every step of the roadmap was validated on ImageNet at 224-pixel resolution, where receptive field and capacity bind; Fashion-MNIST at $96 \times 96$ is upsampled low-resolution data that a ResNet-18 nearly saturates, and on it ResNet's overlapping stem, batch normalization, and sheer size beat a patchify stem and layer normalization tuned for a different regime. Architectures, like recipes, are good *for a regime* rather than good in the abstract, and a two-point reversal on a small dataset is exactly what :numref:`sec_training_recipes` taught you to expect when a method is evaluated far from where it was developed.

## Beyond ConvNeXt

### ConvNeXt V2: pretraining and GRN

ConvNeXt closed the supervised gap, but by 2023 the frontier had moved to self-supervised pretraining, where transformers held an advantage: masked autoencoders :cite:`he2022masked` drop most input patches and train the network to reconstruct them, which is natural for a patch-sequence model and awkward for a convolution that slides across the holes. ConvNeXt V2 :cite:`woo2023convnextv2` adapted the idea with sparse convolutions that skip masked regions during pretraining. Doing so exposed a failure the authors traced to feature *collapse*: with the V1 block, many channels of the pretrained network go dead or redundant. Their fix, *Global Response Normalization* (GRN), is a three-line layer inserted after the block's GELU that computes each channel's global response norm, divides by the mean over channels, and uses the ratio to recalibrate the features, sharpening the contrast between channels and preventing collapse; it also replaces layer scale. The combination of the masked-autoencoder pretraining and GRN lifted the largest model to 88.9% ImageNet top-1 accuracy, at the time the best result trained on public data. Implementing GRN is one of the exercises, and it fits in about five lines.

### Large kernels, honestly

The roadmap's $7 \times 7$ kernel invited an obvious question: why stop there? RepLKNet :cite:`ding2022replknet` pushed depthwise kernels to $31 \times 31$, using the re-parameterization trick of :numref:`sec_efficient_cnns` to train a parallel small kernel that folds away at inference, and showed the effective receptive field widening dramatically, with the gains showing up mostly in detection and segmentation rather than classification. SLaK :cite:`liu2022slak` reached $51 \times 51$ by factorizing the kernel into two thin rectangular stripes plus dynamic sparsity. InternImage :cite:`wang2023internimage` took the opposite route to the same goal, a large *adaptive* receptive field, by building its blocks from deformable convolutions (DCNv3) whose sampling locations are input-dependent, and scaled to 3 billion parameters and state-of-the-art COCO detection. UniRepLKNet :cite:`ding2024unireplknet` carried large-kernel design across modalities, to audio, point clouds, and time series.

The verdict, as of 2026: the *principle* stuck, the specific designs mostly did not. That a backbone should have a large effective receptive field, and that a cheap depthwise operation is the way to buy one, is now uncontroversial; ConvNeXt's own $7 \times 7$ is the everyday embodiment. Giant dense kernels of $31 \times 31$ and beyond did not become defaults: their gains concentrate in dense-prediction tasks, they need re-parameterization and sparsity tricks to train, and hardware support for very large depthwise kernels is uneven. Where an even larger or adaptive receptive field is needed, deformable and other input-dependent operators, and attention itself, carried the idea further than fixed giant kernels did.

### ConvNeXt in 2026

ConvNeXt has aged into infrastructure. It is a standard strong-CNN backbone in detection and segmentation toolkits, a common encoder choice where a transformer's quadratic attention cost is unwelcome at high resolution, and the convolutional tower in several widely used OpenCLIP image-text models :cite:`radford2021learning`, where ConvNeXt encoders trained on billion-scale image-text corpora remain among the strongest public convolutional models. When a practitioner in 2026 says "just use a CNN", the CNN they reach for is more often than not a ConvNeXt or something shaped like one.

## Summary and Discussion

ConvNeXt is a controlled experiment wearing an architecture's name. One change at a time, with accuracy measured at each step, it turned a 2015 ResNet-50 into a network that matches the Swin transformer at equal cost: a modern recipe (the largest single contribution), a Swin-like stage ratio, a patchify stem, depthwise convolutions in an inverted bottleneck led by a $7 \times 7$ kernel, GELU, and radically fewer normalization and activation layers, with the surviving normalization a per-position layer norm. None of these pieces is new to this book, and none is attention. The result settled the 2021-era debate on the terms that matter: at these scales, well-tuned convnets and well-tuned transformers are peers, and most reported gaps between the families were recipe and scale gaps in disguise. Our own experiment applies the same discipline to ConvNeXt itself: on a small, low-resolution task the modernized architecture trails a plain ResNet-18 by two points, a reminder that the roadmap's gains belong to the regime in which they were measured.

The block itself is a piece of convergent evolution. Depthwise convolution mixing space but not channels, a pointwise MLP expanding by four mixing channels but not space, one normalization, one activation, a residual connection: strip the names and the ConvNeXt block and the transformer block are the same design with different spatial mixers. That reading, which :numref:`sec_cnn-design` develops, is more durable than any single leaderboard number, including the ones in this section.

## Exercises

1. Implement Global Response Normalization. For a channels-last feature map $X$, compute per-channel global norms $g_c = \lVert X_{:,:,c} \rVert_2$, normalize them as $n_c = g_c / \bar{g}$ where $\bar{g}$ is the mean over channels, and return $\gamma \odot (X \cdot n) + \beta + X$ with learnable per-channel $\gamma, \beta$ initialized to zero. Insert it after the GELU in `ConvNeXtBlock` (ConvNeXt V2 also removes layer scale when doing so), retrain, and compare.
1. Swap the depthwise kernel size: train the model with $3 \times 3$ and with $11 \times 11$ depthwise convolutions (adjust the padding) and compare accuracy, parameter count, and time per epoch against the $7 \times 7$ baseline. Do you see the saturation that :citet:`liu2022convnet` report?
1. Count where the parameters live: for our ConvNeXt, compute the fraction of parameters in depthwise convolutions, in the $1 \times 1$ expansions and projections, and in the downsampling layers, and compare with the ResNet-18 of :numref:`sec_training_recipes`. Which design decision explains why ConvNeXt is three times smaller at the same depth-times-width feel?
1. Ablate layer scale: train with $\gamma$ initialized to $10^{-6}$ (the default), to $1$, and with the parameter removed entirely. Relate what you observe to stochastic depth, which also shrinks the effective contribution of each residual branch early in training.
1. Our run gives the modern recipe to a modern architecture. Complete the other half of the ablation square: train this ConvNeXt with the 2015 recipe of :numref:`sec_training_recipes` (SGD with momentum, step decay, no Mixup or smoothing). How much of the network's quality survives the recipe downgrade?

<!-- slides -->

::: {.slide title="The question in 2021"}
Vision transformers (ViT, then **Swin**) took over ImageNet and
detection/segmentation backbones.

But a transformer differs from a 2015 ResNet in *many* ways
besides attention: recipe, stem, activations, normalization,
stage shape.

. . .

**Liu et al., 2022**: change a ResNet-50 one step at a time toward
Swin's design, measure after each step, never add attention.

End point: **ConvNeXt, 82.0%** vs. Swin-T's 81.3% at equal FLOPs.
:::

::: {.slide title="The modernization roadmap"}
| Change | Top-1 |
|:--|:--|
| ResNet-50, 2015 recipe | 76.1% |
| modern training recipe | 78.8% |
| stage ratio (3,4,6,3) → (3,3,9,3) | 79.4% |
| patchify stem (4×4, stride 4) | 79.5% |
| depthwise conv + width 64 → 96 | 80.5% |
| inverted bottleneck | 80.6% |
| 7×7 depthwise, moved first | 80.6% |
| GELU; **fewer activations** | 81.3% |
| **fewer norms**; BN → LN | 81.5% |
| separate downsampling | **82.0%** |
:::

::: {.slide title="Read the first row first"}
The largest single jump, **+2.7 points, is the recipe**, not the
architecture (and RSB pushed the same ResNet-50 to 80.4%).

. . .

Of the 5.9 points from 2015 ResNet-50 to ConvNeXt:

- **2.7 training**, 3.2 architecture.

Papers comparing tuned transformers against 2015-recipe ResNets
were largely comparing recipes.
:::

::: {.slide title="Macro design"}
- **Stage ratio**: spend depth where maps are small,
  $(3,3,9,3)$ like Swin-T (+0.6).
- **Patchify stem**: a ViT patch embedding *is* a convolution
  whose kernel equals its stride: 4×4, stride 4 (79.5%).

No pooling, no 7×7 stem: one strided conv slices the image into
patches.
:::

::: {.slide title="The block, before and after"}
![Three norms and three activations become one of each; the bottleneck inverts and leads with a 7×7 depthwise conv.](../img/arch-resnet-vs-convnext-block.svg){width=88%}
:::

::: {.slide title="Why this shape"}
The transformer block factorizes: attention mixes **positions**,
the MLP mixes **channels**.

Convnets have owned that factorization since MobileNet:
depthwise conv + 1×1 convs.

- depthwise 3×3, width to 96: **80.5%**
- inverted bottleneck (expand 4×, like a transformer MLP): 80.6%
- depthwise kernel to **7×7** (cheap once depthwise): 80.6%,
  saturates beyond 7×7
- one GELU, one LayerNorm per block: **81.5%**
:::

::: {.slide title="ConvNeXt-T assembled"}
![Patchify stem, stages 3:3:9:3, LN + 2×2 s2 downsampling between stages.](../img/arch-convnext.svg){width=46%}
:::

::: {.slide title="The block in code"}
LN over channels *at each position* (a transformer's LN): go
channels-last, then 1×1 convs are `Linear` layers:

@convnext-block@pytorch

Layer scale ($\gamma \approx 10^{-6}$): every block starts
near-identity. Stochastic depth: the same `drop_path` as in the
previous section.
:::

::: {.slide title="The network: the arch tuple again"}
Widths (40, 80, 160, 320), depths (2, 2, 6, 2), a scaled-down
"atto" ConvNeXt:

@convnext-model@pytorch
:::

::: {.slide title="Parameter count as a checksum"}
@convnext-params

**3,376,450 parameters**: one wrongly sized layer changes this
number, so matching it validates a reimplementation. A third of
ResNet-18's 11.2M.
:::

::: {.slide title="Train it with the modern recipe"}
ConvNeXt was never trained any other way: AdamW + cosine warmup +
label smoothing + Mixup, from the previous section:

@convnext-recipe@pytorch
:::

::: {.slide title="Results"}
@convnext-train@pytorch

@convnext-eval@pytorch
:::

::: {.slide title="Read the result plainly"}
- ConvNeXt, 3.4M params: **~92.5%** (92.2-92.6% across runs).
- ResNet-18, 11.2M params, same recipe and budget: **94.4%**.

. . .

Not a config artifact: layer-scale init, stochastic depth, and a
45-epoch budget all move it less than run-to-run noise.

The roadmap was validated on ImageNet at 224 px. On upsampled
28 px data, ResNet's overlapping stem + BN + size win.
**Architectures are good for a regime, not in the abstract.**
:::

::: {.slide title="ConvNeXt V2, briefly"}
Masked-autoencoder pretraining for convnets (**FCMAE**): sparse
convolutions skip the masked holes.

. . .

Pretraining exposed **feature collapse**; the fix is
**Global Response Normalization**: normalize each channel's global
response by the mean over channels (~5 lines; exercise 1).

Result: **88.9%** top-1, the best public-data model at the time.
:::

::: {.slide title="Large kernels, honestly"}
- **RepLKNet**: 31×31 depthwise (re-param trick to train).
- **SLaK**: 51×51 via stripes + sparsity.
- **InternImage**: deformable conv (DCNv3), adaptive field, 3B params.
- **UniRepLKNet**: large kernels across audio/points/time series.

. . .

Verdict: the *principle* (large effective receptive field, bought
depthwise) stuck. Giant dense kernels did not become defaults;
deformable/adaptive operators and attention carried it further.
:::

::: {.slide title="Recap"}
- ConvNeXt = a **controlled ablation**: ResNet-50 → 82.0%,
  matching Swin-T, with no attention.
- Largest step: the **recipe**. Then: patchify, depthwise +
  inverted bottleneck, 7×7, one norm + one activation, LN.
- Strip the names: ConvNeXt block ≡ transformer block with a
  depthwise conv as the spatial mixer.
- 2026: a default strong-CNN backbone (OpenCLIP towers,
  detection/segmentation encoders).
:::
