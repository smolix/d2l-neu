# Conditional Generation
:label:`sec_gan_conditional`

The generator of :numref:`sec_dcgan` accepts noise but offers no control over the sample it returns. Many applications instead request an image from a specified class, an image matching a caption, or a restoration consistent with a corrupted input. Conditional generation supplies this control by giving the condition to both the generator and the critic. This section first derives the conditional game, which applies the analysis of :numref:`sec_basic_gan` separately within each condition. It then compares mechanisms for introducing the condition into the networks and derives the projection discriminator from the optimal conditional critic. Finally, it evaluates whether a class-conditional CIFAR-10 generator obeys its requested labels without sacrificing sample quality or diversity. The section concludes by relating class conditioning to paired and unpaired image translation.

```{.python .input #conditional-conditional-generation}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import pickle
import torch
import warnings
from torch import nn
from torch.nn import functional as F
```

```{.python .input #conditional-conditional-generation}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import numpy as np
import optax
import pickle
import warnings
```

## The Conditional Game

A condition is side information the sample must respect. It may be a class label, a caption embedding, or an entire image; for this section's running example it is one of ten CIFAR-10 classes. The data now supplies sample--condition pairs: an image together with its label, drawn from a joint distribution $p(x, c)$. The generator gains an input to match, $x' = G(z, c)$, so that fixing $c$ and varying $z$ draws from a *family* of distributions $q(\cdot \mid c)$, one per condition. We feed the generator conditions drawn from the data's own label marginal $p(c)$: its output pairs $(x', c)$ then have joint distribution $q(x, c) = p(c)\, q(x \mid c)$, and the two players' pair distributions share the condition marginal and differ only in how the sample attached to each condition is produced.

The critic scores sample--condition pairs, $D(x, c)$, and the game is the log-loss game of :numref:`sec_basic_gan` played on those pairs:

$$
V(D) \;=\; E_{(x, c) \sim p}\big[\log \sigma(D(x, c))\big]
+ E_{c \sim p(c),\; x' \sim q(\cdot \mid c)}\big[\log \sigma(-D(x', c))\big].
$$
:eqlabel:`eq_gan_cond_V`

The pointwise argument for the original game applies to any classifier input space. Replacing the sample $x$ by the pair $(x, c)$ makes the optimal critic of :eqref:`eq_gan_dstar` the log ratio of the two joint densities. Their shared condition marginal cancels:

$$
D^\star(x, c)
\;=\; \log \frac{p(x, c)}{q(x, c)}
\;=\; \log \frac{p(c)\, p(x \mid c)}{p(c)\, q(x \mid c)}
\;=\; \log \frac{p(x \mid c)}{q(x \mid c)} .
$$
:eqlabel:`eq_gan_cond_dstar`

The optimal conditional critic is the chapter's log density ratio $\lambda$, computed within the slice of the sample space that the condition selects: for CIFAR-10, the log ratio between real birds and generated birds, with no contribution from any other class. Substituting the best response back into :eqref:`eq_gan_cond_V`, the outer expectation over $c$ survives and each slice contributes the value that :eqref:`eq_gan_js_value` assigns to its own two conditionals:

$$
\max_D V(D)
\;=\; E_{c}\Big[\, 2\, \mathrm{JS}\big(p(\cdot \mid c),\, q(\cdot \mid c)\big) \Big] - 2 \log 2 .
$$
:eqlabel:`eq_gan_cond_value`

The conditional game is an average of per-condition games, weighted by how often each condition occurs. Its value reaches the minimum $-2\log 2$ exactly when $q(\cdot \mid c) = p(\cdot \mid c)$ for every condition that $p(c)$ weights: the generator must match the data within every slice.

That requirement is strictly stronger than the unconditional one, and the gap is easy to exhibit. Consider a generator that produces flawless CIFAR-10 images but shifts every label cyclically by one class, so that $q(\cdot \mid c) = p(\cdot \mid c{+}1)$: asked for a bird, it delivers a perfect cat. Because the ten labels are equally frequent, the cyclic shift leaves the mixture over classes unchanged: the unconditional output distribution is exactly the data marginal, and the game of :numref:`sec_basic_gan` scores this generator as a solved problem. The conditional value :eqref:`eq_gan_cond_value` instead averages the divergences between neighboring class conditionals, which are far from zero. The conditional game charges for disobedience that the unconditional game cannot see; Exercise 2 verifies the decomposition numerically on a toy with known densities.

Because :eqref:`eq_gan_cond_dstar` and :eqref:`eq_gan_cond_value` are the chapter's own identities applied slice by slice, every result built on them transfers the same way. Saturation on disjoint supports, the non-saturating weight of :eqref:`eq_gan_weights`, the pairing objective of :numref:`sec_gan_relativistic`, and the zero-centered penalties of :numref:`sec_gan_convergence` with their damping analysis all hold per condition. Conditioning changes the networks' inputs and the data pipeline; the analysis is inherited. One consequence of the weighting deserves its own note: a condition with small $p(c)$ contributes proportionally little to the value, so nothing in the objective prevents the generator from serving rare conditions badly. The cancellation of $p(c)$ in :eqref:`eq_gan_cond_dstar` also relied on the generator drawing its conditions from the data marginal. Pipelines often sample classes uniformly instead, and when that sampling distribution $r(c)$ differs from $p(c)$ the two joints differ by a prior ratio, which adds the per-slice offset $\log(p(c)/r(c))$ to the optimal critic unless the real batches are re-weighted or re-sampled to match --- the same bookkeeping that class-balanced training performs. Exercise 3 pursues both effects for imbalanced classes.

## How the Condition Enters the Networks

The analysis specifies the function computed by an optimal critic but not how a network should represent $c$. Three common mechanisms are concatenation, condition-dependent modulation of intermediate activations, and a critic head that scores compatibility between the sample and condition. They differ in the structural assumptions built into the network.

The first family is *concatenation* :cite:`Mirza.Osindero.2014`, proposed within months of the original GAN: embed the condition into a vector and concatenate it onto an existing input pathway: for the generator alongside the latent code, for the critic alongside the image or its features at some layer. Concatenation assumes nothing. It works for any condition, discrete or continuous, and leaves the networks to discover how the condition should interact with the sample. That generality is also its weakness: the interaction must be learned from scratch, in a game whose training signal is already indirect.

The second family lets the condition set the scale of computation rather than join its input. Normalize a feature map $h$ and let learned functions of the condition supply the affine coefficients, $h \mapsto \gamma(c) \cdot \mathrm{norm}(h) + \beta(c)$: conditional batch normalization, FiLM, and AdaIN are all instances of this *condition-dependent modulation*, differing in which normalization is modulated and in what computes $\gamma$ and $\beta$, and SPADE is the spatial version, computing $\gamma$ and $\beta$ as maps from a semantic layout so that every location is modulated by its own condition :cite:`Park.Liu.Wang.ea.2019`. Because the modulation reaches every layer it is attached to, it steers global attributes such as class or style more directly than a vector appended once at the input. R3GAN's supplement deliberately omits normalization-based conditioning in the name of minimalism while noting that it improves FID :cite:`Huang.Gokaslan.Kuleshov.ea.2024`.

The third family conditions the critic through a *compatibility head*, and its principal member can be *derived* from the shape of the answer. The target :eqref:`eq_gan_cond_dstar` is a conditional log ratio, and Bayes' rule rewrites each conditional as $p(x \mid c) = p(c \mid x)\, p(x) / p(c)$. Applying this to both numerator and denominator, the shared label marginal cancels again, and

$$
\log \frac{p(x \mid c)}{q(x \mid c)}
\;=\; \log \frac{p(c \mid x)}{q(c \mid x)} + \log \frac{p(x)}{q(x)} .
$$
:eqlabel:`eq_gan_cond_bayes`

The optimal conditional critic splits into two parts with different jobs. The second term is the unconditional $\lambda$ of :numref:`sec_basic_gan`: does $x$ look real at all, condition ignored. The first compares two *label posteriors*: is the pairing of $x$ with $c$ more plausible under the data's labeling rule than under the generator's. A generator producing perfect images with scrambled labels fails only the first term; one producing recognizable blobs of the right class fails only the second.

The first term has a form every classifier in this book has used. Model both posteriors as softmax classifiers over one shared feature map $\varphi$, so that $p(c \mid x) \propto \exp(u_c^\top \varphi(x))$ with one weight vector per class, and $q(c \mid x) \propto \exp(w_c^\top \varphi(x))$ likewise. The two normalizers do not depend on $c$, so the log ratio of the posteriors is the inner product $(u_c - w_c)^\top \varphi(x)$ plus a function of $x$ alone. Absorbing that function, together with the unconditional term of :eqref:`eq_gan_cond_bayes`, into a single head $\psi$, and writing $e_c = u_c - w_c$, the critic becomes

$$
D(x, c) \;=\; e_c^\top \varphi(x) + \psi(x) :
$$
:eqlabel:`eq_gan_cond_projection`

one learned embedding per class, an inner product with the critic's features, and an ordinary unconditional head. This is the *projection discriminator* of :citet:`Miyato.Koyama.2018`. The condition affects the score through one bilinear term. This architecture builds the decomposition in :eqref:`eq_gan_cond_bayes` into the critic instead of requiring a concatenation network to discover it. The Bayes step is an identity, but the log-linear form of the two posteriors in a *shared* feature map is a modeling assumption. Learning $\varphi$ makes the restriction more flexible, since the representation can adapt, but one feature map must still serve both posteriors. Limited capacity can make this restriction consequential; Exercise 1 examines that case. Projection heads are standard at scale, and BigGAN uses one for class-conditional ImageNet :cite:`Brock.Donahue.Simonyan.2019`.

The projection head is not the only compatibility head. The *auxiliary classifier* of :citet:`Odena.Olah.Shlens.2017` conditions the critic by appending a classification loss: a classification head on the critic's features must recover the condition, and the generator is rewarded when its samples classify as theirs. That reward is easiest to earn with exaggerated, class-prototypical samples, so the auxiliary classifier's known pathology is over-prototypical output and reduced within-class diversity. The pathology is this section's gameability discussion, taken up below, built into a training objective: the generator directly optimizes the quantity that the condition-alignment metric measures. For text and other structured conditions the modern mechanism is cross-attention from condition tokens into the feature maps, as in the text-to-image GAN of :citet:`Kang.Zhu.Zhang.ea.2023`; as of 2026, diffusion and autoregressive models dominate general text-to-image synthesis, and adversarial variants persist where single-pass sampling speed matters.

The experiment below picks one mechanism per network, each where its case is strongest: concatenation carries the condition into the generator, whose output must be shaped by $c$ in ways no identity constrains, and the projection head carries it into the critic, whose optimal form :eqref:`eq_gan_cond_projection` is known. The same split, an embedding concatenated into the generator against a projection critic, is what R3GAN's class-conditional runs train :cite:`Huang.Gokaslan.Kuleshov.ea.2024`; Exercise 6 builds the concatenation critic instead and compares the two at a matched budget.

## Class-Conditional CIFAR-10

The test bed is CIFAR-10: 50,000 labeled $32 \times 32$ photographs in ten classes. The dataset appeared in :numref:`sec_dcgan` as the training set of the metric network; here it is the generation target itself. Class-conditional CIFAR-10 is also exactly the setting of R3GAN's conditional benchmark :cite:`Huang.Gokaslan.Kuleshov.ea.2024`, which trains for four days on eight GPUs; our budget is fifteen thousand steps in well under half an hour on one, and the goal is to demonstrate the conditioning machinery, not to compete on sample quality. The registration below repeats :numref:`sec_dcgan`'s, since the entry is local to each section's notebook.

```{.python .input #conditional-class-conditional-cifar-10}
%%tab pytorch
d2l.DATA_HUB['cifar10'] = (
    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    '874905e36347c8536514d0a26261acf3bff89bc7')
cifar_dir = d2l.download_extract('cifar10', 'cifar-10-batches-py')
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def load_cifar_batch(path):
    with open(path, 'rb') as f, warnings.catch_warnings():
        warnings.simplefilter('ignore')   # legacy NumPy pickle format
        batch = pickle.load(f, encoding='bytes')
    X = torch.tensor(batch[b'data'], dtype=torch.float32)
    return X.reshape(-1, 3, 32, 32) / 127.5 - 1, torch.tensor(
        batch[b'labels'])

Xs, ys = zip(*[load_cifar_batch(f'{cifar_dir}/data_batch_{i}')
               for i in range(1, 6)])
device = d2l.try_gpu()
train_X, train_y = torch.cat(Xs).to(device), torch.cat(ys).to(device)
test_X, test_y = load_cifar_batch(f'{cifar_dir}/test_batch')
test_X, test_y = test_X.to(device), test_y.to(device)
first = [int((train_y == c).nonzero()[0]) for c in range(10)]
d2l.show_images(train_X[first].cpu().permute(0, 2, 3, 1) / 2 + 0.5,
                num_rows=2, num_cols=5, titles=classes);
```

```{.python .input #conditional-class-conditional-cifar-10}
%%tab jax
d2l.DATA_HUB['cifar10'] = (
    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    '874905e36347c8536514d0a26261acf3bff89bc7')
cifar_dir = d2l.download_extract('cifar10', 'cifar-10-batches-py')
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def load_cifar_batch(path):
    with open(path, 'rb') as f, warnings.catch_warnings():
        warnings.simplefilter('ignore')   # legacy NumPy pickle format
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return (jnp.asarray(X, dtype=jnp.float32) / 127.5 - 1,
            jnp.asarray(batch[b'labels']))

Xs, ys = zip(*[load_cifar_batch(f'{cifar_dir}/data_batch_{i}')
               for i in range(1, 6)])
train_X, train_y = jnp.concatenate(Xs), jnp.concatenate(ys)
test_X, test_y = load_cifar_batch(f'{cifar_dir}/test_batch')
first = [int(np.nonzero(np.asarray(train_y) == c)[0][0]) for c in range(10)]
d2l.show_images(np.asarray(train_X[np.array(first)]) / 2 + 0.5,
                num_rows=2, num_cols=5, titles=classes);
```

### A Conditional Backbone

The networks use the minimal modern backbone of :numref:`sec_dcgan`, with one resolution stage removed because the images are half the size, and with the condition supplied to both players. The generator retains its learned $4 \times 4$ constant and projected latent. A class embedding is broadcast over the $4 \times 4$ grid and concatenated with both, after which a mixing convolution combines the constant, noise, and condition before the upsampling stack runs $4 \to 8 \to 16 \to 32$. Following :numref:`sec_gan_convergence`, the network uses bilinear resampling, leaky ReLU, no normalization, and no tanh output.

```{.python .input #conditional-a-conditional-backbone-at-32-by-32-1}
%%tab pytorch
class Generator(nn.Module):
    """Learned 4x4 constant + projected latent + class embedding."""
    def __init__(self, latent_dim=100, embed_dim=64, const_ch=128,
                 base_ch=256, num_classes=10):
        super().__init__()
        self.latent_dim, self.embed_dim = latent_dim, embed_dim
        self.const = nn.Parameter(0.02 * torch.randn(1, const_ch, 4, 4))
        self.z_proj = nn.Linear(latent_dim, 16 * latent_dim)
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.mix = nn.Conv2d(const_ch + latent_dim + embed_dim, base_ch,
                             3, padding=1)
        chans = [base_ch // 2 ** i for i in range(4)]    # 256, 128, 64, 32
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nn.Conv2d(chans[-1], 3, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z, y):
        zc = self.z_proj(z).reshape(-1, self.latent_dim, 4, 4)
        ec = self.embed(y)[:, :, None, None].expand(-1, -1, 4, 4)
        const = self.const.expand(z.shape[0], -1, -1, -1)
        x = self.act(self.mix(torch.cat([const, zc, ec], dim=1)))
        for conv in self.stages:                     # 4 -> 8 -> 16 -> 32
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.act(conv(x))
        return self.to_rgb(x)
```

```{.python .input #conditional-a-conditional-backbone-at-32-by-32-1}
%%tab jax
class Generator(nnx.Module):
    """Learned 4x4 constant + projected latent + class embedding."""
    def __init__(self, latent_dim=100, embed_dim=64, const_ch=128,
                 base_ch=256, num_classes=10, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        init = nnx.initializers.normal(0.02)
        self.latent_dim, self.embed_dim = latent_dim, embed_dim
        self.const = nnx.Param(init(rngs.params(), (1, 4, 4, const_ch)))
        self.z_proj = nnx.Linear(latent_dim, 16 * latent_dim,
                                 kernel_init=init, rngs=rngs)
        self.embed = nnx.Embed(num_classes, embed_dim,
                               embedding_init=init, rngs=rngs)
        self.mix = nnx.Conv(const_ch + latent_dim + embed_dim, base_ch,
                            (3, 3), padding='SAME', kernel_init=init,
                            rngs=rngs)
        chans = [base_ch // 2 ** i for i in range(4)]    # 256, 128, 64, 32
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nnx.Conv(chans[-1], 3, (3, 3), padding='SAME',
                               kernel_init=init, rngs=rngs)

    def __call__(self, z, y):
        zc = self.z_proj(z).reshape(-1, 4, 4, self.latent_dim)
        ec = jnp.broadcast_to(self.embed(y)[:, None, None, :],
                              (z.shape[0], 4, 4, self.embed_dim))
        const = jnp.broadcast_to(self.const[...],
                                 (z.shape[0],) + self.const.shape[1:])
        x = nnx.leaky_relu(self.mix(jnp.concatenate([const, zc, ec], -1)),
                           0.2)
        for conv in self.stages:                     # 4 -> 8 -> 16 -> 32
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, 2 * h, 2 * w, c), method='bilinear')
            x = nnx.leaky_relu(conv(x), 0.2)
        return self.to_rgb(x)
```

The critic mirrors the generator down to $4 \times 4$ and then implements :eqref:`eq_gan_cond_projection` literally. Its feature vector $\varphi(x)$ is the final mixed map summed over spatial positions, $\psi$ is a linear head on those features, and the class embeddings $e_c$ live in a table with one row per class; the score is $\psi(x)$ plus the inner product of $\varphi(x)$ with the row the condition selects. One narrowing is deliberate: the derivation leaves $\psi$ an arbitrary function of $x$, and the code realizes it as a linear head on the shared features $\varphi(x)$, following :citet:`Miyato.Koyama.2018`. The three symbols in the code are the three symbols in the equation.

```{.python .input #conditional-a-conditional-backbone-at-32-by-32-2}
%%tab pytorch
class Discriminator(nn.Module):
    """Mirror backbone with a projection head: psi(x) + e_c . phi(x)."""
    def __init__(self, base_ch=64, num_classes=10):
        super().__init__()
        chans = [3] + [base_ch * 2 ** i for i in range(3)]  # 3, 64, ..., 256
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nn.Conv2d(chans[-1], 2 * chans[-1], 3, padding=1)
        self.psi = nn.Linear(2 * chans[-1], 1)
        self.embed = nn.Embedding(num_classes, 2 * chans[-1])
        self.act = nn.LeakyReLU(0.2)

    def phi(self, x):
        for conv in self.stages:                     # 32 -> 16 -> 8 -> 4
            x = F.interpolate(self.act(conv(x)), scale_factor=0.5,
                              mode='bilinear', antialias=True)
        return self.act(self.mix(x)).sum(dim=(2, 3))

    def forward(self, x, y):
        phi = self.phi(x)
        return self.psi(phi).squeeze(-1) + (self.embed(y) * phi).sum(dim=1)

net_G, net_D = Generator(), Discriminator()
print(net_G(torch.zeros(2, 100), torch.zeros(2, dtype=torch.long)).shape)
print(net_D(torch.zeros(2, 3, 32, 32),
            torch.zeros(2, dtype=torch.long)).shape)
```

```{.python .input #conditional-a-conditional-backbone-at-32-by-32-2}
%%tab jax
class Discriminator(nnx.Module):
    """Mirror backbone with a projection head: psi(x) + e_c . phi(x)."""
    def __init__(self, base_ch=64, num_classes=10, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        init = nnx.initializers.normal(0.02)
        chans = [3] + [base_ch * 2 ** i for i in range(3)]  # 3, 64, ..., 256
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nnx.Conv(chans[-1], 2 * chans[-1], (3, 3),
                            padding='SAME', kernel_init=init, rngs=rngs)
        self.psi = nnx.Linear(2 * chans[-1], 1, kernel_init=init,
                              rngs=rngs)
        self.embed = nnx.Embed(num_classes, 2 * chans[-1],
                               embedding_init=init, rngs=rngs)

    def phi(self, x):
        for conv in self.stages:                     # 32 -> 16 -> 8 -> 4
            x = nnx.leaky_relu(conv(x), 0.2)
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, h // 2, w // 2, c),
                                 method='bilinear')
        return nnx.leaky_relu(self.mix(x), 0.2).sum(axis=(1, 2))

    def __call__(self, x, y):
        phi = self.phi(x)
        return self.psi(phi).squeeze(-1) + (self.embed(y) * phi).sum(axis=1)

net_G = Generator(rngs=nnx.Rngs(0))
net_D = Discriminator(rngs=nnx.Rngs(0))
print(net_G(jnp.zeros((2, 100)), jnp.zeros(2, dtype=jnp.int32)).shape)
print(net_D(jnp.zeros((2, 32, 32, 3)), jnp.zeros(2, dtype=jnp.int32)).shape)
```

### Training with the Chapter's Loss

The loss is the one :numref:`sec_gan_convergence` saved to the library: the relativistic pairing objective `d2l.rpgan_loss_D` and `d2l.rpgan_loss_G` with both zero-centered penalties from `d2l.r1_r2_penalty`. One point needs care in the conditional setting. The pairing objective compares a real sample against a generated one, and under :eqref:`eq_gan_cond_value` that comparison is meaningful within a condition: a real bird outranking a generated truck tells the generator nothing about either class. Each real--fake pair therefore shares its condition: we draw a labeled real batch and condition the generator on the same labels, so every comparison is real-versus-generated *of the same class*. The training helpers below otherwise repeat :numref:`sec_dcgan`: horizontal-flip augmentation on real batches, now carrying labels along, and an exponential moving average of the generator's weights with a half-life of 500 steps.

```{.python .input #conditional-training-with-the-chapter-s-loss-1}
%%tab pytorch
class EMA:
    """Exponential moving average of model weights, given as a half-life."""
    def __init__(self, model, half_life):
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        self.decay = 0.5 ** (1 / half_life)

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].lerp_(v, 1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

def sample_real(images, labels, n):
    """Draw a labeled training batch with random horizontal flips."""
    idx = torch.randint(0, len(images), (n,), device=images.device)
    batch = images[idx]
    flip = torch.rand(n, device=images.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), batch.flip(-1),
                       batch), labels[idx]
```

```{.python .input #conditional-training-with-the-chapter-s-loss-1}
%%tab jax
class EMA:
    """Exponential moving average of model weights, given as a half-life."""
    def __init__(self, model, half_life):
        self.shadow = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
        self.decay = 0.5 ** (1 / half_life)

    def update(self, model):
        self.shadow = jax.tree.map(
            lambda s, p: self.decay * s + (1 - self.decay) * p,
            self.shadow, nnx.state(model, nnx.Param))

    def copy_to(self, model):
        nnx.update(model, self.shadow)

def sample_real(key, images, labels, n):
    """Draw a labeled training batch with random horizontal flips."""
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (n,), 0, len(images))
    batch = images[idx]
    flip = jax.random.bernoulli(k2, 0.5, (n,))
    return jnp.where(flip[:, None, None, None], batch[:, :, ::-1, :],
                     batch), labels[idx]
```

The library losses take a critic that maps a batch of images to a batch of scores. A conditional critic becomes one by fixing the labels: in PyTorch a lambda closing over the batch's labels is all that is needed, for both the pairing losses and the penalty. The JAX penalty helper differentiates a per-sample closure under `vmap`, so the condition must be threaded through the map alongside the sample; the training step below therefore carries a small conditional variant of the penalty, identical to `d2l.r1_r2_penalty` except that the per-sample gradient closure receives the sample's label. Training returns the EMA generator together with the trained critic, which the diagnostics at the end of the section interrogate.

```{.python .input #conditional-training-with-the-chapter-s-loss-2}
%%tab pytorch
def train_conditional(train_X, train_y, gamma, num_steps=15000,
                      batch_size=128, lr=0.0002, latent_dim=100,
                      half_life=500, log_every=250, seed=0):
    torch.manual_seed(seed)
    net_G, net_D = Generator().to(device), Discriminator().to(device)
    def init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0, 0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
    for net in (net_G, net_D):
        net.apply(init_weights)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr,
                                 betas=(0.0, 0.99))
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr,
                                 betas=(0.0, 0.99))
    ema, history = EMA(net_G, half_life), []
    for step in range(1, num_steps + 1):
        X, y = sample_real(train_X, train_y, batch_size)
        Z = torch.randn(batch_size, latent_dim, device=device)
        crit = lambda x: net_D(x, y)
        fake = net_G(Z, y).detach()
        r1, r2 = d2l.r1_r2_penalty(crit, X, fake)
        loss_D = (d2l.rpgan_loss_D(crit, X, fake)
                  + gamma / 2 * (r1 + r2).mean())
        trainer_D.zero_grad(), loss_D.backward(), trainer_D.step()
        X2, y2 = sample_real(train_X, train_y, batch_size)
        Z2 = torch.randn(batch_size, latent_dim, device=device)
        crit2 = lambda x: net_D(x, y2)
        loss_G = d2l.rpgan_loss_G(crit2, X2, net_G(Z2, y2))
        trainer_G.zero_grad(), loss_G.backward(), trainer_G.step()
        ema.update(net_G)
        if step % log_every == 0:
            with torch.no_grad():
                Xd, yd = sample_real(train_X, train_y, 256)
                d_real = net_D(Xd, yd).mean()
            history.append((step, float(loss_D.detach()),
                            float(loss_G.detach()),
                            float((r1 + r2).mean().detach()),
                            float(d_real)))
    ema_G = Generator().to(device)
    ema.copy_to(ema_G)
    return ema_G, net_D, torch.tensor(history)
```

```{.python .input #conditional-training-with-the-chapter-s-loss-2}
%%tab jax
def cond_r1_r2_penalty(net_D, real, fake, y):
    """d2l.r1_r2_penalty with the condition threaded through vmap."""
    def sq_grad_norm(x):
        x = jax.lax.stop_gradient(x)
        grad_fn = jax.grad(
            lambda xi, yi: net_D(xi[None, ...], yi[None]).squeeze())
        grad = jax.vmap(grad_fn)(x, y)
        return (grad.reshape(x.shape[0], -1) ** 2).sum(axis=1)
    return sq_grad_norm(real), sq_grad_norm(fake)

@nnx.jit
def cond_step(net_G, net_D, opt_G, opt_D, X, y, Z, X2, y2, Z2, gamma):
    def loss_D_fn(net_D):
        crit = lambda x: net_D(x, y)
        fake = jax.lax.stop_gradient(net_G(Z, y))
        r1, r2 = cond_r1_r2_penalty(net_D, X, fake, y)
        return (d2l.rpgan_loss_D(crit, X, fake)
                + gamma / 2 * (r1 + r2).mean()), (r1 + r2).mean()
    (loss_D, pen), grads = nnx.value_and_grad(
        loss_D_fn, has_aux=True)(net_D)
    opt_D.update(net_D, grads)
    def loss_G_fn(net_G):
        crit = lambda x: net_D(x, y2)
        return d2l.rpgan_loss_G(crit, X2, net_G(Z2, y2))
    loss_G, grads = nnx.value_and_grad(loss_G_fn)(net_G)
    opt_G.update(net_G, grads)
    return loss_D, loss_G, pen

def train_conditional(train_X, train_y, gamma, num_steps=15000,
                      batch_size=128, lr=0.0002, latent_dim=100,
                      half_life=500, log_every=250, seed=0):
    rngs = nnx.Rngs(seed)
    net_G, net_D = Generator(rngs=rngs), Discriminator(rngs=rngs)
    opt_G = nnx.Optimizer(net_G, optax.adam(lr, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    opt_D = nnx.Optimizer(net_D, optax.adam(lr, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    ema, history = EMA(net_G, half_life), []
    key = jax.random.PRNGKey(seed + 1)
    for step in range(1, num_steps + 1):
        key, kx, kz, kx2, kz2, kd = jax.random.split(key, 6)
        X, y = sample_real(kx, train_X, train_y, batch_size)
        Z = jax.random.normal(kz, (batch_size, latent_dim))
        X2, y2 = sample_real(kx2, train_X, train_y, batch_size)
        Z2 = jax.random.normal(kz2, (batch_size, latent_dim))
        loss_D, loss_G, pen = cond_step(net_G, net_D, opt_G, opt_D,
                                        X, y, Z, X2, y2, Z2, gamma)
        ema.update(net_G)
        if step % log_every == 0:
            Xd, yd = sample_real(kd, train_X, train_y, 256)
            history.append((step, float(loss_D), float(loss_G),
                            float(pen), float(net_D(Xd, yd).mean())))
    ema_G = Generator(rngs=nnx.Rngs(seed))
    ema.copy_to(ema_G)
    return ema_G, net_D, np.array(history)
```

The penalty weight was re-tuned rather than inherited. The sweep that chose it was a development-time pilot, run while this section was being built and not reproduced in this notebook: $\gamma \in \{0.05, 0.5, 5\}$ at a 4,000-step budget, in which all three weights trained stably and reached indistinguishable condition alignment. Where they differed is exactly where :eqref:`eq_gan_dirac_pen` says they should, in the penalized quantity $E_m[\|\nabla_x D\|^2]$ itself: at $\gamma = 0.05$ it ran several times larger than at $\gamma = 0.5$, the lightly damped regime, while at $\gamma = 5$ it was pinned nearly flat, the overdamped one. The full runs use $\gamma = 0.5$, the middle setting, as the empirical choice this pilot picked. The number does not travel: :numref:`sec_dcgan` used $\gamma = 10$ on sprites at $64 \times 64$, and R3GAN's tuned CIFAR-10 value is $0.05$, decayed on a schedule to $0.005$ over its four-day run :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. The weight is a property of the dataset, the resolution, the input scaling, the loss reduction conventions, the critic architecture, the optimizer, the batch size, the augmentation, and the budget; a value of $\gamma$ is not portable between implementations that differ in any of these, and a sweep like this pilot is the price of moving to a new one.

```{.python .input #conditional-training-with-the-chapter-s-loss-3}
%%tab pytorch
ema_G, net_D, history = train_conditional(train_X, train_y, gamma=0.5)
print(f'final loss_D {history[-1, 1]:.3f}, loss_G {history[-1, 2]:.3f}, '
      f'R1+R2 {history[-1, 3]:.3f}')
```

```{.python .input #conditional-training-with-the-chapter-s-loss-3}
%%tab jax
ema_G, net_D, history = train_conditional(train_X, train_y, gamma=0.5)
print(f'final loss_D {history[-1, 1]:.3f}, loss_G {history[-1, 2]:.3f}, '
      f'R1+R2 {history[-1, 3]:.3f}')
```

```{.python .input #conditional-training-with-the-chapter-s-loss-4}
%%tab pytorch, jax
fig, axes = d2l.plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(history[:, 0], history[:, 1], label='critic')
axes[0].plot(history[:, 0], history[:, 2], label='generator')
axes[0].axhline(0.693, ls='--', c='gray', lw=1)
axes[0].set_xlabel('step'), axes[0].set_ylabel('loss'), axes[0].legend()
axes[1].plot(history[:, 0], history[:, 4])
axes[1].axhline(0, ls='--', c='gray', lw=1)
axes[1].set_xlabel('step')
axes[1].set_title('mean critic score on real batches')
fig.tight_layout()
```

The traces show a stable game. The dashed line marks $\log 2 \approx 0.693$, the value both pairing losses take when the critic can no longer separate its pairs. The critic's loss settles just below the line and the generator's remains above it, giving the critic a modest edge without either trace drifting over the training budget. Stability does not imply smoothness: each recorded run shows a sharp generator-loss excursion within the first few thousand steps. The excursion is mild in one framework and reaches several times the settled level in the other, but both runs return to their previous bands within a few logging intervals. The penalties yield recoverable excursions and no sustained trend, not an absence of excursions.

After an initial transient, the critic's mean score on real batches settles into a narrow band. This contrasts with the unbounded climb of the unpenalized classic arm in :numref:`sec_dcgan`. The band's absolute level is arbitrary because the pairing objective is invariant to shifts in critic scores.

The next grid tests generation on demand. Each row fixes one class, and each column uses a fresh latent code. As in :numref:`sec_dcgan`, the generator has no tanh. The display and all evaluations below therefore clamp its raw output to $[-1, 1]$; the printed diagnostic shows that only a small fraction of pixel values are affected.

```{.python .input #conditional-training-with-the-chapter-s-loss-5}
%%tab pytorch
torch.manual_seed(42)
z = torch.randn(100, 100, device=device)
y = torch.arange(10, device=device).repeat_interleave(10)
with torch.no_grad():
    imgs = ((ema_G(z, y).clamp(-1, 1) + 1) / 2)
grid = imgs.reshape(10, 10, 3, 32, 32).permute(0, 3, 1, 4, 2)
grid = grid.reshape(320, 320, 3).cpu().numpy()
d2l.set_figsize((6, 6))
d2l.plt.imshow(grid)
d2l.plt.yticks([32 * i + 16 for i in range(10)], classes)
d2l.plt.xticks([]);
```

```{.python .input #conditional-training-with-the-chapter-s-loss-5}
%%tab jax
z = jax.random.normal(jax.random.PRNGKey(42), (100, 100))
y = jnp.repeat(jnp.arange(10), 10)
imgs = (jnp.clip(ema_G(z, y), -1, 1) + 1) / 2
grid = np.asarray(imgs.reshape(10, 10, 32, 32, 3).transpose(0, 2, 1, 3, 4)
                  .reshape(320, 320, 3))
d2l.set_figsize((6, 6))
d2l.plt.imshow(grid)
d2l.plt.yticks([32 * i + 16 for i in range(10)], classes)
d2l.plt.xticks([]);
```

Most rows are class-distinct to the eye, and the sharpness of the obedience varies by class. In the recorded runs the automobile, truck, and ship rows read at a glance: wheeled bodies with windshields, hulls on water; the airplane rows are the most variable of the vehicle classes across runs. The animal rows are blobbier and identify themselves as much by palette and setting as by anatomy. Deer stand as tan shapes on grassy ground and horses keep their silhouette, but the bird, cat, and dog rows blur toward mutually confusable brown shapes that only sometimes resolve into their animal. Within each row the columns vary in pose, color, and background, so conditioning has not visibly cost within-class diversity. This is what the budget buys: the demonstration is conditioning, not photorealism, and the same recipe run class-conditionally for four days on eight GPUs reaches photographic CIFAR-10 samples :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. The eye is also a generous judge, and a grid cannot say how often a row's samples would actually pass for their class, so the next section builds a measurement.

## Measuring Condition Alignment

A conditional generator can fail in a way no unconditional metric detects: its samples may be fine while their assignment to conditions is wrong, the class-shift failure that :eqref:`eq_gan_cond_value` charges for and the marginal game ignores. Measuring obedience needs a referee that knows the classes, and the natural referee is a classifier. We train the small convolutional classifier of :numref:`sec_dcgan` on CIFAR-10 until its test accuracy plateaus, at the native $32 \times 32$ this time since the generated images are already at the data's resolution, and use it twice. Its label for a conditioned sample either matches the condition or does not: the matching fraction, per class and overall, is the *condition alignment*. Its accuracy on real test images, computed the same way, is the *reference level* against which alignment is read: the referee misclassifies real birds too, so an alignment number means little until it is set beside how the referee fares on the real class. The reference is not a bound. A generator whose samples are more class-prototypical than real photographs can score above it, a first hint of the gameability taken up below.

```{.python .input #conditional-measuring-condition-alignment-1}
%%tab pytorch
class FeatureCNN(nn.Module):
    """Three conv blocks, global average pooling, and a linear head."""
    def __init__(self, feature_dim=128):
        super().__init__()
        widths = [3, 32, 64, feature_dim]
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1),
                           nn.BatchNorm2d(c_out), nn.ReLU(),
                           nn.MaxPool2d(2))
             for c_in, c_out in zip(widths[:-1], widths[1:])])
        self.head = nn.Linear(feature_dim, 10)

    def features(self, x):
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.features(x))

torch.manual_seed(0)
cnn = FeatureCNN().to(device)
opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
for epoch in range(15):
    order = torch.randperm(len(train_X), device=device)
    for i in range(0, len(order) - 255, 256):
        idx = order[i:i + 256]
        l = F.cross_entropy(cnn(train_X[idx]), train_y[idx])
        opt.zero_grad(), l.backward(), opt.step()
cnn.eval()
with torch.no_grad():
    pred = torch.cat([cnn(test_X[i:i + 1000]).argmax(1)
                      for i in range(0, len(test_X), 1000)])
cls_acc = [float((pred[test_y == c] == c).float().mean())
           for c in range(10)]
print(f'CIFAR-10 test accuracy: {float((pred == test_y).float().mean()):.3f}')
```

```{.python .input #conditional-measuring-condition-alignment-1}
%%tab jax
class FeatureCNN(nnx.Module):
    """Three conv blocks, global average pooling, and a linear head."""
    def __init__(self, feature_dim=128, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        widths = [3, 32, 64, feature_dim]
        self.convs = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME', rngs=rngs)
             for c_in, c_out in zip(widths[:-1], widths[1:])])
        self.norms = nnx.List([nnx.BatchNorm(c, rngs=rngs)
                               for c in widths[1:]])
        self.head = nnx.Linear(feature_dim, 10, rngs=rngs)

    def features(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = nnx.max_pool(nnx.relu(norm(conv(x))),
                             window_shape=(2, 2), strides=(2, 2))
        return x.mean(axis=(1, 2))

    def __call__(self, x):
        return self.head(self.features(x))

@nnx.jit
def cnn_step(cnn, opt, X, y):
    def loss_fn(cnn):
        return optax.softmax_cross_entropy_with_integer_labels(
            cnn(X), y).mean()
    l, grads = nnx.value_and_grad(loss_fn)(cnn)
    opt.update(cnn, grads)
    return l

cnn = FeatureCNN(rngs=nnx.Rngs(0))
opt = nnx.Optimizer(cnn, optax.adam(0.001), wrt=nnx.Param)
key = jax.random.PRNGKey(3)
for epoch in range(15):
    key, kp = jax.random.split(key)
    order = jax.random.permutation(kp, len(train_X))
    for i in range(0, len(order) - 255, 256):
        idx = order[i:i + 256]
        cnn_step(cnn, opt, train_X[idx], train_y[idx])
cnn.eval()
pred = jnp.concatenate([cnn(test_X[i:i + 1000]).argmax(1)
                        for i in range(0, len(test_X), 1000)])
cls_acc = [float((pred[test_y == c] == c).mean()) for c in range(10)]
print(f'CIFAR-10 test accuracy: {float((pred == test_y).mean()):.3f}')
```

```{.python .input #conditional-measuring-condition-alignment-2}
%%tab pytorch
torch.manual_seed(7)
align = []
for c in range(10):
    z = torch.randn(500, 100, device=device)
    yc = torch.full((500,), c, device=device)
    with torch.no_grad():
        pred_c = cnn(ema_G(z, yc).clamp(-1, 1)).argmax(1)
    align.append(float((pred_c == c).float().mean()))
print(f'{"class":12s}{"alignment":>11s}{"classifier":>12s}')
for c in range(10):
    print(f'{classes[c]:12s}{align[c]:11.2f}{cls_acc[c]:12.2f}')
print(f'{"overall":12s}{np.mean(align):11.2f}{np.mean(cls_acc):12.2f}')
```

```{.python .input #conditional-measuring-condition-alignment-2}
%%tab jax
key = jax.random.PRNGKey(7)
align = []
for c in range(10):
    key, kz = jax.random.split(key)
    z = jax.random.normal(kz, (500, 100))
    yc = jnp.full((500,), c)
    pred_c = cnn(jnp.clip(ema_G(z, yc), -1, 1)).argmax(1)
    align.append(float((pred_c == c).mean()))
print(f'{"class":12s}{"alignment":>11s}{"classifier":>12s}')
for c in range(10):
    print(f'{classes[c]:12s}{align[c]:11.2f}{cls_acc[c]:12.2f}')
print(f'{"overall":12s}{np.mean(align):11.2f}{np.mean(cls_acc):12.2f}')
```

In the recorded runs, overall alignment lands within a few points of the classifier's own test accuracy in each framework: the generator obeys its conditions roughly as often as the referee can verify obedience on real images. The per-class rows describe individual runs. The recorded PyTorch and JAX tables disagree about which classes are weak by margins far beyond counting noise at 500 samples per class. The two tabs therefore demonstrate the measurement rather than replicate a shared classwise pattern. The eye is generous in both cases: the weakest rows look better on the page than they score. The cat row in the recorded PyTorch run and the dog row in the recorded JAX run each fall roughly ten points below the referee's already-weak accuracy on that class. In both runs, the weakest generated class is also the class on which the referee itself is weakest.

The comparison is signed, not bounded. Nothing caps a row at the referee's accuracy on the corresponding real class. Samples that appear more class-prototypical to the referee than real photographs can score above it; the collapse baseline in the diagnostics does exactly that. In the recorded runs, every generated class happens to sit below its per-class reference.

The signed comparison is not a calibration. The referee is another model evaluated under distribution shift, and its errors on generated images need not mirror its errors on real ones. A zero gap can hide two failure modes that happen to cancel, while a positive gap can indicate classifier-prototypical samples rather than genuine obedience. Reading alignment against the reference level is more informative than reading the raw number alone, but it remains a proxy. A stronger classifier sharpens that proxy without changing what it measures.

Alignment says nothing about whether the samples are any good as images, so the second measurement reuses the distribution metrics of :numref:`sec_dcgan` in the feature space of the classifier just trained: the Fréchet distance :eqref:`eq_gan_fid` and the unbiased $\mathrm{MMD}^2$ estimator :eqref:`eq_gan_kid`, printed as `FD (CIFAR-CNN)` and `MMD^2 (CIFAR-CNN)` since the features are the chapter's own rather than Inception's. The comparison scores 1,000 generated images, 100 per class, against 1,000 held-out test images drawn 100 per class by label, so that the real sets share the generated set's exactly balanced class marginal; the floor is computed from a second, disjoint stratified thousand, and the table carries :numref:`sec_dcgan`'s out-of-range column for the generator's raw outputs.

```{.python .input #conditional-measuring-condition-alignment-3}
%%tab pytorch, jax
def fid_score(feat_p, feat_q):
    feat_p, feat_q = np.float64(feat_p), np.float64(feat_q)
    mu_p, mu_q = feat_p.mean(0), feat_q.mean(0)
    C_p = np.cov(feat_p, rowvar=False)
    C_q = np.cov(feat_q, rowvar=False)
    lam, U = np.linalg.eigh(C_p)
    sqrt_C_p = (U * np.sqrt(np.maximum(lam, 0))) @ U.T
    M = sqrt_C_p @ C_q @ sqrt_C_p
    cross = np.linalg.eigvalsh((M + M.T) / 2)   # symmetric up to rounding
    return (((mu_p - mu_q) ** 2).sum() + np.trace(C_p) + np.trace(C_q)
            - 2 * np.sqrt(np.maximum(cross, 0)).sum())

def kid_score(feat_p, feat_q):
    m, n, d = len(feat_p), len(feat_q), feat_p.shape[1]
    K_pp = (feat_p @ feat_p.T / d + 1) ** 3
    K_qq = (feat_q @ feat_q.T / d + 1) ** 3
    K_pq = (feat_p @ feat_q.T / d + 1) ** 3
    return ((K_pp.sum() - np.trace(K_pp)) / (m * (m - 1))
            + (K_qq.sum() - np.trace(K_qq)) / (n * (n - 1))
            - 2 * K_pq.mean())
```

```{.python .input #conditional-measuring-condition-alignment-4}
%%tab pytorch
def features(imgs, bs=500):
    with torch.no_grad():
        return np.concatenate([cnn.features(imgs[i:i + bs]).cpu().numpy()
                               for i in range(0, len(imgs), bs)])

torch.manual_seed(11)
z = torch.randn(1000, 100, device=device)
yg = torch.arange(10, device=device).repeat_interleave(100)
with torch.no_grad():
    raw = ema_G(z, yg)
oob = float(((raw < -1) | (raw > 1)).float().mean())
gen = raw.clamp(-1, 1)
idx = [(test_y == c).nonzero().flatten()[:200] for c in range(10)]
f_real = features(test_X[torch.cat([i[:100] for i in idx])])
f_real2 = features(test_X[torch.cat([i[100:] for i in idx])])
f_gen = features(gen)
print(f'{"comparison":22s}{"FD (CIFAR-CNN)":>16s}{"MMD^2 (CIFAR-CNN)":>19s}'
      f'{"out-of-range":>14s}')
print(f'{"real vs. real":22s}{fid_score(f_real2, f_real):16.2f}'
      f'{kid_score(f_real2, f_real):19.3f}{"--":>14s}')
print(f'{"generated vs. real":22s}{fid_score(f_gen, f_real):16.2f}'
      f'{kid_score(f_gen, f_real):19.3f}{oob:14.3f}')
```

```{.python .input #conditional-measuring-condition-alignment-4}
%%tab jax
def features(imgs, bs=500):
    return np.concatenate([np.asarray(cnn.features(imgs[i:i + bs]))
                           for i in range(0, len(imgs), bs)])

z = jax.random.normal(jax.random.PRNGKey(11), (1000, 100))
yg = jnp.repeat(jnp.arange(10), 100)
raw = ema_G(z, yg)
oob = float(((raw < -1) | (raw > 1)).mean())
gen = jnp.clip(raw, -1, 1)
idx = [jnp.nonzero(test_y == c)[0][:200] for c in range(10)]
f_real = features(test_X[jnp.concatenate([i[:100] for i in idx])])
f_real2 = features(test_X[jnp.concatenate([i[100:] for i in idx])])
f_gen = features(gen)
print(f'{"comparison":22s}{"FD (CIFAR-CNN)":>16s}{"MMD^2 (CIFAR-CNN)":>19s}'
      f'{"out-of-range":>14s}')
print(f'{"real vs. real":22s}{fid_score(f_real2, f_real):16.2f}'
      f'{kid_score(f_real2, f_real):19.3f}{"--":>14s}')
print(f'{"generated vs. real":22s}{fid_score(f_gen, f_real):16.2f}'
      f'{kid_score(f_gen, f_real):19.3f}{oob:14.3f}')
```

The two distances return the verdict of the eye at different sensitivities. In the recorded runs the generated-versus-real FD prints at a small multiple of the real-versus-real floor, a clear but modest gap; the collapsed classic arm of :numref:`sec_dcgan`, for comparison, sat two orders of magnitude above its floor. The unbiased $\mathrm{MMD}^2$ agrees on a compressed scale: its floor scatters around zero as a U-statistic should, and the generated arm lands small and positive, near the floor without touching it. As in :numref:`sec_dcgan`, both values are functions of the chapter's own feature network before they are functions of the generator, and are not comparable to published FID or KID numbers. Both are also *marginal feature-distribution fit* metrics: each compares the pooled feature distribution of the generated images with that of the real ones, one number in which fidelity failures and coverage failures mix, with no separate reading for either and no view of the labels at all. What they support is the qualitative statement that matters here: the conditional generator's pooled output sits close to the data in this feature space, so the alignment scores above were not bought with unusable images.

The literature's established instruments close the gap between marginal fit and conditional obedience in two ways. The Fréchet joint distance embeds image and condition jointly, so pairing good images with the wrong conditions moves the metric that a marginal FD cannot see moving :cite:`DeVries.Romero.Pineda.ea.2019`, and the classification accuracy score trains a classifier on generated labeled data and tests it on real data, so missing or collapsed classes surface as errors on those classes' real test images :cite:`Ravuri.Vinyals.2019`. Both are the established versions of the probes this section builds from its own classifier.

Three checks close the loop between the section's theory and its numbers, and all three run in seconds on artifacts already computed. The first executes the opening counterexample: keep the same thousand generated images and cyclically shift the *requested* labels by one class. Alignment collapses, because the images were not generated for the shifted requests, while the pooled FD and $\mathrm{MMD}^2$ are unchanged, because the images are identical and neither distance ever sees a label. The second confirms that training actually used the projection head: on a batch of real images, the critic's mean score under the correct labels minus its mean score under cyclically shifted labels cancels $\psi$ along with the pairing objective's shift symmetry, leaving only the embedding term, positive iff the head learned to read its condition. The third builds the degenerate generator that the trade-off discussion below describes: one fixed held-out image per class, repeated a hundred times. Its alignment lands in the range of the real-image reference, since the referee merely re-classifies ten real images, while its per-class feature variance sits near zero, far below the trained generator's; variance, not alignment, is the direct detector of within-class collapse.

```{.python .input #conditional-measuring-condition-alignment-5}
%%tab pytorch
y_perm = (yg + 1) % 10
with torch.no_grad():
    pred_gen = cnn(gen).argmax(1)
    d_true = net_D(test_X[:1000], test_y[:1000]).mean()
    d_perm = net_D(test_X[:1000], (test_y[:1000] + 1) % 10).mean()
x0 = torch.stack([test_X[test_y == c][0] for c in range(10)])
fixed = x0.repeat_interleave(100, dim=0)
with torch.no_grad():
    pred_fix = cnn(fixed).argmax(1)
f_fix = features(fixed)
cvar = lambda f: float(np.float64(f).reshape(10, 100, -1).var(1).mean())
print(f'permuted labels: alignment '
      f'{float((pred_gen == yg).float().mean()):.2f} -> '
      f'{float((pred_gen == y_perm).float().mean()):.2f}, '
      f'FD {fid_score(f_gen, f_real):.2f}, '
      f'MMD^2 {kid_score(f_gen, f_real):.3f} (both unchanged)')
print(f'critic label check: mean D(x, y) - D(x, y+1) on real = '
      f'{float(d_true - d_perm):.2f}')
print(f'collapse baseline: alignment '
      f'{float((pred_fix == yg).float().mean()):.2f}, per-class feature '
      f'variance {cvar(f_fix):.3g} (generator {cvar(f_gen):.3g}, '
      f'real {cvar(f_real):.3g})')
```

```{.python .input #conditional-measuring-condition-alignment-5}
%%tab jax
y_perm = (yg + 1) % 10
pred_gen = cnn(gen).argmax(1)
d_true = net_D(test_X[:1000], test_y[:1000]).mean()
d_perm = net_D(test_X[:1000], (test_y[:1000] + 1) % 10).mean()
x0 = jnp.stack([test_X[test_y == c][0] for c in range(10)])
fixed = jnp.repeat(x0, 100, axis=0)
pred_fix = cnn(fixed).argmax(1)
f_fix = features(fixed)
cvar = lambda f: float(np.float64(f).reshape(10, 100, -1).var(1).mean())
print(f'permuted labels: alignment '
      f'{float((pred_gen == yg).mean()):.2f} -> '
      f'{float((pred_gen == y_perm).mean()):.2f}, '
      f'FD {fid_score(f_gen, f_real):.2f}, '
      f'MMD^2 {kid_score(f_gen, f_real):.3f} (both unchanged)')
print(f'critic label check: mean D(x, y) - D(x, y+1) on real = '
      f'{float(d_true - d_perm):.2f}')
print(f'collapse baseline: alignment '
      f'{float((pred_fix == yg).mean()):.2f}, per-class feature '
      f'variance {cvar(f_fix):.3g} (generator {cvar(f_gen):.3g}, '
      f'real {cvar(f_real):.3g})')
```

Three properties now describe the generator: sample fidelity, sample diversity, and condition alignment. No instrument above reports all three separately. The two distances measure marginal feature fit, where fidelity and coverage are mixed together, while alignment measures obedience alone.

The collapse baseline shows an extreme trade-off. One memorized image per class can attain alignment near the reference level even though its within-class diversity is zero. This *within-class collapse* is mode collapse within a conditional slice. The conditional value :eqref:`eq_gan_cond_value` still penalizes it because a point mass is far from $p(\cdot \mid c)$ under every divergence considered in this chapter. The alignment number cannot detect it, however, and the feature-space distances see it only through the marginal mixture. Per-class feature variance, as used in the diagnostics, is a more direct detector.

Incorrect labels are not the only conditional failure. A generator may ignore the condition, obey it through shortcuts such as background or palette, collapse within a condition, leak between conditions, neglect rare conditions, or optimize for classifier prototypes. A sound evaluation therefore considers fidelity, diversity, and alignment together. Exercise 4 extends the collapse diagnosis to per-class distribution metrics. Exercise 5 uses the same instruments to measure the truncation trick of :citet:`Brock.Donahue.Simonyan.2019`, which deliberately exchanges diversity for alignment.

## Translation as Conditioning

Nothing in :eqref:`eq_gan_cond_V` requires the condition to be a label. Let $c$ be an entire image --- a semantic map, an edge sketch, a grayscale photograph --- and the conditional game becomes *paired image-to-image translation*: pix2pix :cite:`Isola.Zhu.Zhou.ea.2017` trains exactly the conditional critic $D(x, c)$ of this section, scoring output-input pairs, alongside an L1 reconstruction loss toward the paired ground truth. The division of labor between the two losses follows from what each can represent: the input image nearly determines the output's low frequencies, which the pointwise L1 term recovers, while the residual ambiguity concentrates in local texture, where a pointwise loss averages over the plausible alternatives and blurs them, and the critic, run as a patch critic over local windows, penalizes texture statistics that no real patch exhibits and keeps the output sharp. :numref:`sec_gan_beyond` develops this capacity argument in full.

Unpaired translation drops the ground truth: two photo collections, say horses and zebras, with no correspondence between them. CycleGAN :cite:`Zhu.Park.Isola.ea.2017` does not extend the conditional critic to this setting. For the direction $G: X \to Y$ its discriminator distinguishes real images $y$ from translations $G(x)$ and never receives $x$: an unconditional critic on the target collection's marginal. The system plays two such *marginal* adversarial games, one per direction, coupled by the two generators and a *cycle-consistency* loss requiring that translating forward and back return approximately the original. The cycle loss is what carries the input--output relation, and it narrows an underidentified mapping without making the critics conditional: nothing in the objective guarantees the semantically intended correspondence, and a pair of mappings can satisfy both cycles while relating images in unintended ways. The contrast fits in two rows:

| | Critic sees the source? | Paired targets? | What carries the input--output relation |
|:--|:--|:--|:--|
| pix2pix | yes: $D(x, c)$ scores output--input pairs | yes | the conditional adversarial loss and the L1 term |
| CycleGAN | no: each critic scores its target marginal | no | the cycle-consistency loss alone |

Only pix2pix plays this section's conditional game; CycleGAN substitutes a reconstruction constraint for a conditional critic. The systems themselves, with their encoder-decoder generators and patch critics, are built in :numref:`chap_cv`.

## Summary

Conditional generation trains on pairs $(x,c)$ and gives the condition to both networks. When real and generated pairs share the same marginal $p(c)$, the optimal critic is the conditional log density ratio $\log[p(x\mid c)/q(x\mid c)]$. The value is the average $E_c[2\,\mathrm{JS}(p(\cdot\mid c),q(\cdot\mid c))]-2\log2$. Saturation, non-saturating weights, relativistic pairing, and zero-centered penalties therefore apply separately within each condition. A label-shifted generator shows why this target is stronger than matching the unconditional marginal.

Conditions can enter through concatenation, condition-dependent modulation, or a compatibility head. Under a shared log-linear representation of the real and generated class posteriors, the Bayes decomposition gives the projection critic $D(x,c)=e_c^\top\varphi(x)+\psi(x)$. The CIFAR-10 experiment uses concatenation in the generator and this projection head in the critic. Within 15,000 steps, most requested classes produce visibly distinct rows, overall classifier alignment is close to the classifier's accuracy on real data, and the pooled feature distances remain near their real-versus-real references. The weakest classes differ between the recorded PyTorch and JAX runs, so classwise conclusions are specific to each run.

Conditional evaluation adds an axis that marginal sample metrics cannot observe. The pooled feature distances combine fidelity and coverage and ignore labels. Classifier alignment measures whether generated images match their requested labels, but it is an uncalibrated proxy because the classifier is itself evaluated under distribution shift. High alignment can coexist with within-class collapse, as the repeated-image baseline demonstrates; per-class variance or distributional metrics are needed to detect that failure. With an image as the condition, pix2pix uses a conditional critic on source--output pairs together with an L1 reconstruction term. CycleGAN instead uses marginal critics and places the input--output constraint in its cycle-consistency loss, which does not guarantee the intended semantic correspondence.

## Exercises

1. The projection derivation modeled both class posteriors as softmax classifiers over a *shared* feature map $\varphi$. Suppose instead that the two posteriors require different feature maps, $p(c \mid x) \propto \exp(u_c^\top \varphi_p(x))$ and $q(c \mid x) \propto \exp(w_c^\top \varphi_q(x))$ with $\varphi_p \neq \varphi_q$. What form does the critic take in place of :eqref:`eq_gan_cond_projection`? Can this two-map critic be written as a single projection head over the concatenated feature map $(\varphi_p(x), \varphi_q(x))$? What does the answer say about whether the shared-map assumption is a limit on capacity or a constraint on structure, and about when a learned $\varphi$ can absorb it?
1. Verify the decomposition :eqref:`eq_gan_cond_value` numerically on a toy with known densities. Let $c \in \{1, 2\}$ with equal probability, $p(\cdot \mid 1) = \mathcal{N}(-2, 1)$, $p(\cdot \mid 2) = \mathcal{N}(2, 1)$, and let the generator be right on one slice and wrong on the other: $q(\cdot \mid 1) = \mathcal{N}(0, 1)$, $q(\cdot \mid 2) = \mathcal{N}(2, 1)$. Compute $E_c[2\,\mathrm{JS}] - 2\log 2$ by numerical integration on a grid. Then train a small critic that takes $(x, c)$, with $c$ one-hot, on samples from the two joints, taking the critic loss as the balanced mean of the real and fake cross-entropy terms (the reduction under which the stated relation holds), and compare its converged per-sample loss with your integral, using the relation from :numref:`sec_basic_gan` that the discriminator's minimal per-sample loss is $\log 2 - \mathrm{JS}$, here with the divergence averaged over the two conditions. Confirm that the critic's scores on the matched slice $c = 2$ are near zero while the mismatched slice carries the whole value.
1. Suppose the label marginal is imbalanced: nine classes share probability $(1 - \epsilon)$ evenly and one rare class has $p(c_{\textrm{rare}}) = \epsilon$. The value :eqref:`eq_gan_cond_value` weights the rare slice's divergence by $\epsilon$, so a generator that serves it arbitrarily badly pays at most $2\epsilon \log 2$. Compare this *across-slice* weighting with the *within-slice* imbalance of :numref:`sec_basic_gan`'s exercises, where unequal priors inside one game produced the divergence $\alpha\, \mathrm{KL}(p \,\|\, m_\alpha) + (1-\alpha)\, \mathrm{KL}(q \,\|\, m_\alpha)$, which we may call an $\alpha$-skewed Jensen--Shannon divergence: which mechanism affects the optimal critic, and which only the value? Predict what happens to the rare class's condition alignment if CIFAR-10 is subsampled so one class is a hundred times rarer; if you have the compute, test the prediction with the code of this section, and a lighter test fits a small budget: keep two classes, subsample one to a few percent of the other, and train at the 4,000-step pilot budget. In either setting compare two remedies: drawing generator conditions from the natural, imbalanced label marginal, versus drawing them uniformly while re-weighting or re-sampling the real batches to match --- the prior-ratio correction from the shared-marginal caveat above.
1. The diagnostics cell detects within-class collapse through per-class feature variance. Extend the diagnosis to per-class distribution metrics: for each class $c$, embed $n$ generated and $n$ real samples of that class with the feature CNN and compute the per-class $\widehat{\mathrm{MMD}}^2$ of :eqref:`eq_gan_kid`, or per-class precision and recall. Score the trained generator of this section, rank the ten classes by within-class distance, and compare that ranking with the alignment table's: which classes are well aligned yet poorly distributed, or the reverse, and what do their samples look like?
1. BigGAN's truncation trick :cite:`Brock.Donahue.Simonyan.2019` draws each latent coordinate from a standard Gaussian truncated to $[-\tau, \tau]$, resampling any coordinate that falls outside; in BigGAN this traded diversity for fidelity, a result that depended on that model's training setup. Implement truncation for the trained generator and sweep $\tau \in \{0.5, 1, 2\} \cup \{\infty\}$, where $\tau = \infty$ recovers ordinary sampling, reporting condition alignment and per-class feature variance for each $\tau$. Does alignment rise and within-class variance fall as $\tau$ shrinks here? A flat or reversed outcome for this small generator is a valid finding; report it and explain what it implies about how this generator uses the latent space away from the prior's mode.
1. Implement the concatenation critic: one-hot encode the label, broadcast it spatially, and append it to the critic's input as ten extra planes (or to an intermediate feature map), leaving the backbone otherwise unchanged. Retrain the game with this critic at a reduced budget of 4,000 steps, retrain the projection version at the same budget, and compare condition alignment and the feature-space distances. The comparison is a diagnostic of what the projection structure buys at a matched small budget, not a quality contest, and either ordering is a reportable result.
1. For pix2pix and for CycleGAN, list every adversarial loss in the system and classify its critic as conditional (it sees the source alongside the candidate output) or marginal (it sees only samples from the target collection). State, for each system, what carries the input--output relation, and explain how a CycleGAN whose cycle losses are near zero can still pair images in unintended ways.

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.6]{.kicker}

Conditional generation<br>
**the game on pairs · how the condition enters · class-conditional CIFAR-10 · measuring obedience · translation**
:::
:::

::: {.slide title="The Game on Sample–Condition Pairs"}
Data supplies sample–condition pairs $(x, c)$; the generator draws $c$ from
the same label marginal and attaches a sample: $x' = G(z, c)$. The critic
scores the pairs:

$$V(D) = E_{(x,c) \sim p}[\log \sigma(D(x, c))]
+ E_{c,\, x' \sim q(\cdot \mid c)}[\log \sigma(-D(x', c))]$$

. . .

Same log-loss game as :numref:`sec_basic_gan` on a richer input space;
the shared marginal $p(c)$ is the assumption every step below uses.
:::

::: {.slide title="The Optimal Critic, Slice by Slice"}
The pointwise argument never cared what the input space was:

$$D^\star(x, c) = \log \frac{p(c)\, p(x \mid c)}{p(c)\, q(x \mid c)}
= \log \frac{p(x \mid c)}{q(x \mid c)}$$

. . .

$$\max_D V = E_c\big[2\, \mathrm{JS}(p(\cdot \mid c), q(\cdot \mid c))\big]
- 2 \log 2$$

- An average of per-condition games, weighted by $p(c)$.
- Minimum iff $q(\cdot \mid c) = p(\cdot \mid c)$ on **every** slice:
  stronger than matching the marginal (a label-shifted generator matches
  the marginal perfectly and fails every slice).
- Every chapter result — saturation, weights, pairing, penalties —
  transfers per condition.
:::

::: {.slide title="How the Condition Enters the Networks"}
- **Concatenation** (Mirza & Osindero, 2014): embed $c$, concatenate onto
  an input pathway. Assumes nothing; learns the interaction from scratch.
- **Modulation**: $h \mapsto \gamma(c) \cdot \mathrm{norm}(h) + \beta(c)$:
  conditional BatchNorm, FiLM, AdaIN; SPADE computes $\gamma, \beta$
  spatially from a layout. (R3GAN omits it for minimalism; it helps FID.)
- **Compatibility heads on the critic**: the projection head (derived
  next) or an auxiliary classifier, whose classification reward invites
  class-prototypical samples.

. . .

Text conditions: cross-attention (GigaGAN).
Our experiment: concatenation into $G$, projection head in $D$.
:::

::: {.slide title="Deriving the Projection Discriminator"}
Bayes flips the target into label consistency plus realness:

$$\log \frac{p(x \mid c)}{q(x \mid c)}
= \underbrace{\log \frac{p(c \mid x)}{q(c \mid x)}}_{\textrm{label consistency}}
+ \underbrace{\log \frac{p(x)}{q(x)}}_{\textrm{unconditional realness}}$$

. . .

Model both label posteriors as softmax classifiers over shared features
$\varphi$; the normalizers are $c$-free, so

$$D(x, c) = e_c^\top \varphi(x) + \psi(x)$$

- One embedding per class, one inner product, one unconditional head.
- Bayes step: identity. Shared log-linear posteriors: an assumption;
  learning $\varphi$ softens it but does not remove it.
- The standard head at scale: BigGAN, R3GAN's conditional runs.
:::

::: {.slide title="Class-Conditional CIFAR-10, the Chapter's Recipe"}
32×32, one fewer stage than :numref:`sec_dcgan`; class embedding
concatenated at the 4×4 stage of $G$; projection head in $D$.
Loss: `d2l.rpgan_loss_D/G` + both zero-centered penalties.
Pairs share their condition: real batch's labels condition the fakes.

. . .

$\gamma$ re-tuned by pilot sweep $\{0.05, 0.5, 5\}$: all stable, ends
under- vs. overdamped exactly as :numref:`sec_gan_convergence` predicts;
we fix the middle, $\gamma = 0.5$. (Sprites took 10; R3GAN's CIFAR value
is 0.05 with a decay schedule. The weight does not travel.)
:::

::: {.slide title="Generation on Demand"}
@!conditional-training-with-the-chapter-s-loss-5

Each row is a request, each column a fresh latent draw. Most vehicle rows
read at a glance; animal rows read by palette and setting, the weakest
barely at all; columns vary within every row.
:::

::: {.slide title="Measuring Obedience"}
@!conditional-measuring-condition-alignment-2

Classify conditioned samples; alignment = fraction matching the request,
read against the classifier's own accuracy on real images — a reference
level, not a bound. Reading against the reference sharpens the proxy but
does not calibrate it: the referee is another model under distribution
shift. Which classes are weakest is a per-run finding; the weakest rows
drop far below the grid's visual impression.
:::

::: {.slide title="Alignment Sees the Permutation; the Pooled Distances Do Not"}
@!conditional-measuring-condition-alignment-5

- Same 1,000 images, requested labels shifted by one class: alignment
  collapses, FD and MMD$^2$ unchanged — the opening counterexample,
  executed.
- Collapse baseline, one fixed image per class: alignment near the
  reference level, per-class feature variance near zero — the direct
  detector of within-class collapse.
:::

::: {.slide title="The Trade-off Triangle"}
Three axes: fidelity, diversity, condition alignment.

- FD / MMD$^2$ in the chapter-trained feature space measure the marginal
  fit; alignment measures obedience.
- Alignment is gameable: one perfect image per class aces it.
  **Within-class collapse**: mode collapse relocated inside a slice.
- The conditional value still charges for it; the metrics need per-class
  versions to see it (Exercises 4–5; truncation trades along this edge).
:::

::: {.slide title="Translation as Conditioning"}
The condition can be an image:

- **pix2pix** (paired): conditional critic on output–input pairs + L1.
  The input pins the low frequencies; pointwise losses average over the
  ambiguous texture, so the patch critic carries it
  (:numref:`sec_gan_beyond` develops the argument).
- **CycleGAN** (unpaired): two *marginal* critics — each judges only its
  target collection and never sees the source. The cycle-consistency loss
  alone carries the relation, without guaranteeing the intended
  correspondence.

Implementations: :numref:`chap_cv`.
:::

::: {.slide title="Recap"}
- Conditioning = the same game on pairs; optimal critic = conditional
  log ratio; value = expected per-slice JS. Analysis inherited, slice by
  slice.
- Matching every conditional is strictly stronger than matching the
  marginal.
- Projection head: derived, not designed — Bayes + log-linear posteriors.
- CIFAR-10: the chapter's loss, conditional, $\gamma$ re-tuned; most grid
  rows read as their class, and which rows are weakest is a per-run
  finding, far below their visual impression.
- New metric, new failure: alignment against a referee's reference level
  — a proxy, not a calibration; within-class collapse invisible to it,
  visible to per-class feature variance.
- Paired translation reuses the conditional critic; unpaired translation
  plays marginal games held together by a cycle loss.
:::
