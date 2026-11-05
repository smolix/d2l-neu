# Conditional Generation
:label:`sec_gan_conditional`

Most image generators that people use take a request: a prompt, a class, a sketch to complete. The generators of the previous sections take none. They return a sample from the whole data distribution, and the user gets whatever comes out. Within months of the original GAN, :citet:`Mirza.Osindero.2014` proposed the obvious repair: give the request to both players. The generator receives it as an extra input, and the critic judges whether a sample is a realistic instance of that request.

Is that one change enough? This section asks what the game then optimizes, how the critic should read the request, and how we can tell whether the generator obeys. The answers reuse what the chapter has built: the log-loss game, the pairing loss and the two penalties apply once per request. The running example is CIFAR-10: ask for a bird, and check that we get one. The check turns out to be the harder half, because a generator can obey every request and still return a single image per class.

```{.python .input #conditional-conditional-generation}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
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
```
## The Conditional Game

### The Optimal Conditional Critic

A condition specifies a property of the requested sample. It may be a class label, a caption embedding, or an entire image. Our running example uses the ten CIFAR-10 class labels. The data consists of image--label pairs drawn from a joint distribution $p(x,c)$. The generator also receives the condition, so $x' = G(z,c)$, and varying $z$ at fixed $c$ samples from a conditional distribution $q(\cdot \mid c)$. We draw the generator's conditions from the data marginal $p(c)$, so the generated pairs follow $q(x,c)=p(c)\,q(x\mid c)$. The real and generated joint distributions then have the same condition marginal and differ only in their conditional image distributions.

The critic scores sample--condition pairs, $D(x, c)$, and the game is the log-loss game of :numref:`sec_basic_gan` played on those pairs:

$$
V(D) \;=\; \mathbb{E}_{(x, c) \sim p}\big[\log \sigma(D(x, c))\big]
+ \mathbb{E}_{c \sim p(c),\; x' \sim q(\cdot \mid c)}\big[\log \sigma(-D(x', c))\big].
$$
:eqlabel:`eq_gan_cond_V`

In the unconditional game the optimal critic is the log density ratio $\lambda = \log(p/q)$, and the value at that critic is $2\,\mathrm{JS}(p, q) - 2\log 2$. The pointwise argument behind both results works for any input space, so it applies with the pair $(x, c)$ in place of $x$. The optimal critic is the log ratio of the two joint densities, and their shared condition marginal cancels:

$$
D^\star(x, c)
\;=\; \log \frac{p(x, c)}{q(x, c)}
\;=\; \log \frac{p(c)\, p(x \mid c)}{p(c)\, q(x \mid c)}
\;=\; \log \frac{p(x \mid c)}{q(x \mid c)} .
$$
:eqlabel:`eq_gan_cond_dstar`

The optimal conditional critic is the log density ratio within the slice of data that the condition selects. For CIFAR-10, the score for the bird condition compares real and generated birds; images of other classes do not enter it. Substituting this critic into :eqref:`eq_gan_cond_V` keeps the outer expectation over $c$, and each condition contributes the value of its own unconditional game:

$$
\max_D V(D)
\;=\; \mathbb{E}_{c}\Big[\, 2\, \mathrm{JS}\big(p(\cdot \mid c),\, q(\cdot \mid c)\big) \Big] - 2 \log 2 .
$$
:eqlabel:`eq_gan_cond_value`

Thus the conditional game is an average of per-condition games, weighted by how often each condition occurs. Its value reaches the minimum $-2\log 2$ exactly when $q(\cdot \mid c) = p(\cdot \mid c)$ for every condition that $p(c)$ weights: the generator must match the data within every slice.

### Stronger than Marginal Matching

This requirement is stronger than matching the distribution of images. Consider a generator that produces accurate CIFAR-10 images but shifts every label cyclically, so that $q(\cdot \mid c) = p(\cdot \mid c{+}1)$: a request for a bird returns a cat. All ten classes occur equally often, so the shift leaves the mixture over classes unchanged. The generator's distribution of images equals the data's, and the unconditional game sits at its minimum, $-2\log 2$; it cannot see the wrong labels. The conditional game compares each slice with its neighbor instead. If the classes had disjoint supports, every slice would contribute $\mathrm{JS} = \log 2$, and :eqref:`eq_gan_cond_value` would equal $2\log 2 - 2\log 2 = 0$, the largest value the game can take. CIFAR-10 classes overlap a little, so the value sits just below that maximum: a perfect image generator that fails every request is close to the worst conditional generator. Exercise :exref:`ex_conditional-verify-decomposition-eqref-eq` computes both values for two Gaussian slices.

### Imbalanced Conditions

The weighting by $p(c)$ has two consequences for imbalanced data. First, a rare condition contributes little to the objective, so the generator can serve it poorly at a small cost. Second, the cancellation in :eqref:`eq_gan_cond_dstar` assumes that the generator's conditions are drawn from the data marginal. If they are drawn from a different distribution $r(c)$, the optimal critic gains the offset $\log(p(c)/r(c))$. Reweighting or resampling the real batches to match $r(c)$ removes this offset, as in class-balanced training. Exercise :exref:`ex_conditional-rare-class` examines the first effect, and Exercise :exref:`ex_conditional-suppose-label-marginal-is` the second.

### Same-Class Pairs

The derivation above used the log loss, but the experiment later in this section trains with the pairing loss of :numref:`sec_gan_relativistic`, and pairing raises a question the log loss did not: which generated sample is compared with which real one? We pair each real image with a generated image of the same class, a real bird with a generated bird. Within one class this is the pairing game of :numref:`sec_gan_relativistic` between $p(\cdot \mid c)$ and $q(\cdot \mid c)$, with the objective $\Phi_c(D) = \mathbb{E}_{x \sim p(\cdot \mid c),\, x' \sim q(\cdot \mid c)}[\log \sigma(D(x, c) - D(x', c))]$. The critic of each class can be optimized separately, so the value of the conditional pairing game is the lifted Jensen--Shannon divergence of :numref:`sec_gan_relativistic`, averaged over classes:

$$
\max_D \mathbb{E}_c\big[\Phi_c(D)\big] + \log 2
\;=\; \mathbb{E}_c\Big[\mathrm{JS}\big(p(\cdot \mid c) \otimes q(\cdot \mid c),\; q(\cdot \mid c) \otimes p(\cdot \mid c)\big)\Big].
$$
:eqlabel:`eq_gan_cond_rp_value`

Each term is zero exactly when its slice is matched, so the fixed point is the same as that of :eqref:`eq_gan_cond_value`. Two properties of the pairing game carry over class by class. First, adding a constant to the critic leaves every score difference unchanged; with same-class pairs this holds for one constant per class, so $D(x, c) + \zeta_c$ has the same loss for any constants $\zeta_1, \ldots, \zeta_{10}$. The critic never has to calibrate its scores across classes, and the diagnostics below must not depend on these constants. Second, the generator's rank weight, the probability that a generated sample outranks a random real one, becomes the probability that a generated bird outranks a real bird. In the finite setting of :numref:`sec_gan_relativistic`, where the critic responds optimally and each real point keeps one generated partner, the paired loss has no local minimum with generated points piled onto a few data points; with same-class pairs this holds within each class. Pairing a real bird with a generated truck would keep the fixed point but would compare scores across classes, where the constants $\zeta_c$ no longer cancel.

The other results of the chapter extend per condition in the same way. Saturation on disjoint supports, the non-saturating generator, and the zero-centered penalties of :numref:`sec_gan_convergence` are statements about one pair of distributions, and conditioning only changes which pair. What remains open is how a network should receive $c$.

## Conditioning Mechanisms

### Three Ways In

The analysis says what the optimal critic computes but not how a network should take the condition in. Three mechanisms are common (:numref:`fig_gan_conditioning`): appending the condition to an input, letting the condition scale and shift intermediate activations, and a critic head that scores how well the sample and the condition fit together. They differ in how much structure they build into the network.

![Three ways a condition $c$ enters a GAN. Left: concatenation appends a class embedding $e_c$ to the latent code (or to an image or feature map in the critic). Middle: modulation scales and shifts a feature map $\phi$ by functions of the condition, $a(c) \odot \phi + b(c)$ (conditional batch normalization normalizes $\phi$ first). Right: the compatibility head of the projection critic scores the inner product of a class embedding with the critic features, $e_c^\top \phi(x)$, and adds an unconditional term $\psi_0(x)$.](../img/mdl-gan-conditioning.svg)
:label:`fig_gan_conditioning`

The first mechanism is *concatenation* :cite:`Mirza.Osindero.2014`. The condition is embedded in a vector $e_c$ and appended to an existing input: beside the latent code in the generator, or beside the image or an intermediate feature map in the critic. Concatenation works for discrete and continuous conditions and imposes no form on their interaction with the sample, so the network must learn that interaction from the adversarial signal alone.

The second mechanism is *modulation*. A feature map $\phi$ becomes $a(c) \odot \phi + b(c)$, where $a(c)$ and $b(c)$ are learned functions of the condition with one entry per channel. Conditional batch normalization normalizes $\phi$ first; BigGAN computes $a(c)$ and $b(c)$ from a class embedding in every block of its generator :cite:`Brock.Donahue.Simonyan.2019`. Papers write $\gamma(c)$ and $\beta(c)$; we write $a$ and $b$ because $\gamma$ is the penalty weight in this chapter. Modulation at several layers gives the condition a direct handle on global attributes such as class or style.

The third mechanism conditions only the critic, through a head that scores the compatibility of sample and condition. Its most common form follows from the optimal critic by Bayes' rule.

### The Projection Head from Bayes' Rule

Bayes' rule gives $p(x \mid c) = p(c \mid x)\, p(x) / p(c)$. Applying it to the numerator and the denominator of :eqref:`eq_gan_cond_dstar` cancels the shared label marginal and yields

$$
\log \frac{p(x \mid c)}{q(x \mid c)}
\;=\; \log \frac{p(c \mid x)}{q(c \mid x)} + \log \frac{p(x)}{q(x)} .
$$
:eqlabel:`eq_gan_cond_bayes`

Thus the optimal conditional critic separates condition compatibility from unconditional sample quality. The second term is the unconditional log density ratio $\lambda$ of :numref:`sec_basic_gan`. The first compares the real and generated *label posteriors* at $x$. A generator that produces realistic images with scrambled labels differs from the data through the first term; a generator that produces poorly formed images of the requested class differs through the second.

The first term has a form every classifier in this book has used. Model both posteriors as softmax classifiers on one shared feature map $\phi$, so that $p(c \mid x) \propto \exp(u_c^\top \phi(x))$ with one weight vector per class, and $q(c \mid x) \propto \exp(w_c^\top \phi(x))$ likewise. The two normalizers do not depend on $c$, so the log ratio of the posteriors is the inner product $(u_c - w_c)^\top \phi(x)$ plus a function of $x$ alone. Absorbing that function, together with the unconditional term of :eqref:`eq_gan_cond_bayes`, into a single head $\psi_0$ and writing $e_c = u_c - w_c$, the critic becomes

$$
D(x, c) \;=\; e_c^\top \phi(x) + \psi_0(x).
$$
:eqlabel:`eq_gan_cond_projection`

This is the *projection discriminator* of :citet:`Miyato.Koyama.2018`: one learned embedding per class, an inner product with the critic's features, and an unconditional head. Bayes' rule is an identity, but the log-linear form of both posteriors in a *shared* feature map is a modeling assumption; learning $\phi$ makes it flexible, though one representation must still serve both posteriors (Exercise :exref:`ex_conditional-projection-derivation-modeled-both`). When $\psi_0$ is linear, $\psi_0(x) = w^\top \phi(x) + w_0$, the score is $(e_c + w)^\top \phi(x) + w_0$: the unconditional head can be absorbed into the class embeddings, so the split into two terms is a reparameterization rather than a constraint, and R3GAN's conditional critic keeps only the inner product :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. The table of class embeddings generalizes to any embedding of the condition. StyleGAN-T conditions its critic by projection onto the CLIP embedding of a caption :cite:`Sauer.Karras.Laine.ea.2023`, which makes the same head the text-conditioned critic of current one-step generators.

The projection head is not the only compatibility mechanism. The *auxiliary classifier* critic of :citet:`Odena.Olah.Shlens.2017` adds a classification head and loss: the critic predicts the condition from its features, and the generator is trained to make its samples receive the requested label. This objective favors exaggerated class prototypes, which are easy to classify, over within-class diversity (Exercise :exref:`ex_conditional-auxiliary-classifier-bias`). The generator also optimizes the quantity later used to measure condition alignment, which makes that measurement misleading.

The experiment uses concatenation in the generator and a projection head in the critic. No identity like :eqref:`eq_gan_cond_projection` says how the generator should transform $c$, so concatenation leaves that interaction free, while the critic gets the structure the derivation motivates. R3GAN uses the same division in its class-conditional experiments. Exercise :exref:`ex_conditional-implement-concatenation-critic-one` replaces the projection head by concatenation and compares the two critics at the same budget.

### Images as Conditions

The condition can also be an entire image: a semantic map, an edge sketch, or a grayscale photograph. Pix2pix :cite:`Isola.Zhu.Zhou.ea.2017` plays the conditional game on output--input pairs, with the source image as $c$, and adds an $L_1$ loss against the paired target. The source fixes the low-frequency structure of the output, which the $L_1$ term recovers. Fine texture is ambiguous given the source, and the $L_1$ term alone blurs it. Without paired examples, CycleGAN :cite:`Zhu.Park.Isola.ea.2017` gives up the conditional critic: each direction plays an unconditional game against its target collection, and a cycle-consistency loss asks that translating forward and back return the input. The cycle loss constrains the map without identifying it, since two maps can satisfy both cycles and both critics while pairing images in an unintended way (Exercise :exref:`ex_conditional-translation-identifiability`).

## Class-Conditional CIFAR-10

We now train a class-conditional generator on CIFAR-10, whose 50,000 labeled $32 \times 32$ training images come from ten classes. :numref:`sec_dcgan` used these images to train a feature network; here they are the target distribution, loaded with `d2l.load_cifar10_arrays` from that section. R3GAN reports a class-conditional CIFAR-10 benchmark trained for four days on eight GPUs :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. Our run uses 15,000 steps and takes well under half an hour on one GPU. Its purpose is to test the conditioning and its measurement, not to match large-scale sample quality.

```{.python .input #conditional-class-conditional-cifar-10}
%%tab pytorch
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
device = d2l.try_gpu()
train_X, train_y, test_X, test_y = d2l.load_cifar10_arrays()
first = [int((train_y == c).nonzero()[0]) for c in range(10)]
d2l.show_images(train_X[first].cpu().permute(0, 2, 3, 1) / 2 + 0.5,
                num_rows=2, num_cols=5, titles=classes);
```
```{.python .input #conditional-class-conditional-cifar-10}
%%tab jax
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
train_X, train_y, test_X, test_y = d2l.load_cifar10_arrays()
first = [int(np.nonzero(np.asarray(train_y) == c)[0][0]) for c in range(10)]
d2l.show_images(np.asarray(train_X[np.array(first)]) / 2 + 0.5,
                num_rows=2, num_cols=5, titles=classes);
```
### A Conditional Backbone

The networks are the minimal backbone of :numref:`sec_dcgan` with one resolution stage removed for the smaller images, and with the condition as an extra input to each. The generator keeps its learned $4 \times 4$ constant and its projected latent code, broadcasts a class embedding over the $4 \times 4$ grid, and concatenates all three. A mixing convolution combines them before the upsampling stages raise the resolution from $4$ to $8$, $16$, and $32$. As in :numref:`sec_dcgan`, the network uses bilinear resampling, leaky ReLU, no normalization and no tanh, and every weight is drawn with the fan-in variance $2 / ((1 + 0.2^2)\, n_{\mathrm{in}})$, which keeps the scale of the activations without normalization layers. We repeat the two networks in full because the two entry points of the class are the point of this subsection.

```{.python .input #conditional-a-conditional-backbone-at-32-by-32-1}
%%tab pytorch
def fan_in_init(module):  # as in the unconditional backbone
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.2)  # leaky-ReLU gain
        nn.init.zeros_(module.bias)

class Generator(nn.Module):
    """Learned 4x4 constant + projected latent + class embedding."""
    def __init__(self, latent_dim=100, embed_dim=64, const_ch=128,
                 base_ch=256, num_classes=10):
        super().__init__()
        self.latent_dim, self.embed_dim = latent_dim, embed_dim
        self.const = nn.Parameter(torch.randn(1, const_ch, 4, 4))
        self.z_proj = nn.Linear(latent_dim, 16 * latent_dim)
        self.embed = nn.Embedding(num_classes, embed_dim)  # unit variance
        self.mix = nn.Conv2d(const_ch + latent_dim + embed_dim, base_ch,
                             3, padding=1)
        chans = [base_ch // 2 ** i for i in range(4)]    # 256, 128, 64, 32
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nn.Conv2d(chans[-1], 3, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.apply(fan_in_init)

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
# Fan-in initialization with the leaky-ReLU gain, as in the unconditional
# backbone
fan_in_init = nnx.initializers.variance_scaling(2 / (1 + 0.2 ** 2),
                                                'fan_in', 'normal')

class Generator(nnx.Module):
    """Learned 4x4 constant + projected latent + class embedding."""
    def __init__(self, latent_dim=100, embed_dim=64, const_ch=128,
                 base_ch=256, num_classes=10, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.latent_dim, self.embed_dim = latent_dim, embed_dim
        self.const = nnx.Param(
            jax.random.normal(rngs.params(), (1, 4, 4, const_ch)))
        self.z_proj = nnx.Linear(latent_dim, 16 * latent_dim,
                                 kernel_init=fan_in_init, rngs=rngs)
        self.embed = nnx.Embed(num_classes, embed_dim, rngs=rngs,
                               embedding_init=nnx.initializers.normal(1.0))
        self.mix = nnx.Conv(const_ch + latent_dim + embed_dim, base_ch,
                            (3, 3), padding='SAME', kernel_init=fan_in_init,
                            rngs=rngs)
        chans = [base_ch // 2 ** i for i in range(4)]    # 256, 128, 64, 32
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=fan_in_init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nnx.Conv(chans[-1], 3, (3, 3), padding='SAME',
                               kernel_init=fan_in_init, rngs=rngs)

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
The critic mirrors the generator down to $4 \times 4$ and then implements :eqref:`eq_gan_cond_projection`. Summing the final feature map over its positions gives $\phi(x)$, a linear head on $\phi(x)$ gives $\psi_0(x)$, and a table holds one embedding $e_c$ per class. The table starts at zero, so the critic begins as an unconditional critic and learns the label term from the data.

```{.python .input #conditional-a-conditional-backbone-at-32-by-32-2}
%%tab pytorch
class Discriminator(nn.Module):
    """Mirror backbone with a projection head: e_c . phi(x) + psi_0(x)."""
    def __init__(self, base_ch=64, num_classes=10):
        super().__init__()
        chans = [3] + [base_ch * 2 ** i for i in range(3)]  # 3, 64, ..., 256
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nn.Conv2d(chans[-1], 2 * chans[-1], 3, padding=1)
        self.psi0 = nn.Linear(2 * chans[-1], 1)
        self.embed = nn.Embedding(num_classes, 2 * chans[-1])
        self.act = nn.LeakyReLU(0.2)
        self.apply(fan_in_init)
        nn.init.zeros_(self.embed.weight)  # start as an unconditional critic

    def phi(self, x):
        for conv in self.stages:                     # 32 -> 16 -> 8 -> 4
            x = F.interpolate(self.act(conv(x)), scale_factor=0.5,
                              mode='bilinear', antialias=True)
        return self.act(self.mix(x)).sum(dim=(2, 3))

    def forward(self, x, y):
        phi = self.phi(x)
        return (self.embed(y) * phi).sum(dim=1) + self.psi0(phi).squeeze(-1)

net_G, net_D = Generator(), Discriminator()
print(net_G(torch.zeros(2, 100), torch.zeros(2, dtype=torch.long)).shape)
print(net_D(torch.zeros(2, 3, 32, 32),
            torch.zeros(2, dtype=torch.long)).shape)
```
```{.python .input #conditional-a-conditional-backbone-at-32-by-32-2}
%%tab jax
class Discriminator(nnx.Module):
    """Mirror backbone with a projection head: e_c . phi(x) + psi_0(x)."""
    def __init__(self, base_ch=64, num_classes=10, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        chans = [3] + [base_ch * 2 ** i for i in range(3)]  # 3, 64, ..., 256
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=fan_in_init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nnx.Conv(chans[-1], 2 * chans[-1], (3, 3),
                            padding='SAME', kernel_init=fan_in_init,
                            rngs=rngs)
        self.psi0 = nnx.Linear(2 * chans[-1], 1, kernel_init=fan_in_init,
                               rngs=rngs)
        self.embed = nnx.Embed(num_classes, 2 * chans[-1], rngs=rngs,
                               embedding_init=nnx.initializers.zeros)

    def phi(self, x):
        for conv in self.stages:                     # 32 -> 16 -> 8 -> 4
            x = nnx.leaky_relu(conv(x), 0.2)
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, h // 2, w // 2, c),
                                 method='bilinear')
        return nnx.leaky_relu(self.mix(x), 0.2).sum(axis=(1, 2))

    def __call__(self, x, y):
        phi = self.phi(x)
        return (self.embed(y) * phi).sum(axis=1) + self.psi0(phi).squeeze(-1)

net_G = Generator(rngs=nnx.Rngs(0))
net_D = Discriminator(rngs=nnx.Rngs(0))
print(net_G(jnp.zeros((2, 100)), jnp.zeros(2, dtype=jnp.int32)).shape)
print(net_D(jnp.zeros((2, 32, 32, 3)), jnp.zeros(2, dtype=jnp.int32)).shape)
```
### Training with the Chapter's Loss

Training uses the pairing losses `d2l.rpgan_loss_D` and `d2l.rpgan_loss_G` of :numref:`sec_gan_convergence`, which evaluate the pairing objective on a batch of aligned real and generated images, together with the per-sample squared input gradients of `d2l.r1_r2_penalty` on both batches. Same-class pairs come from the labels: we draw a labeled real batch and condition the generated batch on the same labels. The other choices follow :numref:`sec_dcgan`: horizontal flips of the real images (`d2l.sample_real`, whose `labels` argument returns the labels with the batch), Adam with $\beta_1 = 0$, and an exponential moving average of the generator with a half-life of 500 steps (`d2l.EMA`), which supplies the generator for evaluation.

The library losses and the penalty expect a critic that maps a batch of images to a batch of scores. Fixing the labels turns the conditional critic into such a function, so a lambda that closes over the labels serves both the losses and the penalty. Every 250 steps the loop also records the *held-out pair accuracy*: on 512 fresh pairs of a test image and a generated image of the same class, the fraction that the critic orders correctly, $\hat P(D(x, c) > D(x', c))$, computed by `d2l.ranking_accuracy`. The per-class constants $\zeta_c$ cancel from it, unlike from the critic's mean score, which drifts with them. A value of one half means that the critic cannot tell real from generated images of a class.

```{.python .input #conditional-training-with-the-chapter-s-loss-2}
%%tab pytorch
def train_conditional(train_X, train_y, test_X, test_y, gamma,
                      num_steps=15000, batch_size=128, lr=0.0002,
                      latent_dim=100, half_life=500, log_every=250, seed=0):
    torch.manual_seed(seed)
    device = train_X.device
    net_G, net_D = Generator().to(device), Discriminator().to(device)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr, betas=(0.0, 0.99))
    trainer_D = torch.optim.Adam(net_D.parameters(), lr, betas=(0.0, 0.99))
    noise = lambda n: torch.randn(n, latent_dim, device=device)
    ema, history = d2l.EMA(net_G, half_life), []
    for step in range(1, num_steps + 1):
        X, y = d2l.sample_real(train_X, batch_size, train_y)
        crit = lambda x: net_D(x, y)  # the critic at the batch's labels
        with torch.no_grad():         # the same labels: same-class pairs
            fake = net_G(noise(batch_size), y)
        r1, r2 = d2l.r1_r2_penalty(crit, X, fake)
        loss_D, pen = d2l.rpgan_loss_D(crit, X, fake), (r1 + r2).mean()
        trainer_D.zero_grad(), (loss_D + gamma / 2 * pen).backward()
        trainer_D.step()
        X2, y2 = d2l.sample_real(train_X, batch_size, train_y)
        loss_G = d2l.rpgan_loss_G(lambda x: net_D(x, y2), X2,
                                  net_G(noise(batch_size), y2))
        trainer_G.zero_grad(), loss_G.backward(), trainer_G.step()
        ema.update(net_G)
        if step % log_every == 0:  # fresh same-class pairs of test images
            Xd, yd = d2l.sample_real(test_X, 512, test_y)
            with torch.no_grad():
                fake = net_G(noise(512), yd)
            acc = d2l.ranking_accuracy(lambda x: net_D(x, yd), Xd, fake)
            history.append((step, loss_D.item(), loss_G.item(), pen.item(),
                            acc))
    return ema.model, net_D, np.array(history)
```
```{.python .input #conditional-training-with-the-chapter-s-loss-2}
%%tab jax
@nnx.jit
def cond_step(net_G, net_D, opt_G, opt_D, X, y, Z, X2, y2, Z2, gamma):
    def loss_D_fn(net_D):
        crit = lambda x: net_D(x, y)  # the critic at the batch's labels
        fake = jax.lax.stop_gradient(net_G(Z, y))  # same-class pairs
        r1, r2 = d2l.r1_r2_penalty(crit, X, fake)
        loss, pen = d2l.rpgan_loss_D(crit, X, fake), (r1 + r2).mean()
        return loss + gamma / 2 * pen, (loss, pen)
    (_, (loss_D, pen)), grads = nnx.value_and_grad(
        loss_D_fn, has_aux=True)(net_D)
    opt_D.update(net_D, grads)
    loss_G, grads = nnx.value_and_grad(lambda net_G: d2l.rpgan_loss_G(
        lambda x: net_D(x, y2), X2, net_G(Z2, y2)))(net_G)
    opt_G.update(net_G, grads)
    return loss_D, loss_G, pen

def train_conditional(train_X, train_y, test_X, test_y, gamma,
                      num_steps=15000, batch_size=128, lr=0.0002,
                      latent_dim=100, half_life=500, log_every=250, seed=0):
    rngs = nnx.Rngs(seed)
    net_G, net_D = Generator(rngs=rngs), Discriminator(rngs=rngs)
    opt_G = nnx.Optimizer(net_G, optax.adam(lr, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    opt_D = nnx.Optimizer(net_D, optax.adam(lr, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    noise = lambda key, n: jax.random.normal(key, (n, latent_dim))
    ema, history = d2l.EMA(net_G, half_life), []
    key = jax.random.key(seed + 1)
    for step in range(1, num_steps + 1):
        key, kx, kz, kx2, kz2 = jax.random.split(key, 5)
        X, y = d2l.sample_real(kx, train_X, batch_size, train_y)
        X2, y2 = d2l.sample_real(kx2, train_X, batch_size, train_y)
        logs = cond_step(net_G, net_D, opt_G, opt_D, X, y,
                         noise(kz, batch_size), X2, y2,
                         noise(kz2, batch_size), gamma)
        ema.update(net_G)
        if step % log_every == 0:  # fresh same-class pairs of test images
            key, kd, kz = jax.random.split(key, 3)
            Xd, yd = d2l.sample_real(kd, test_X, 512, test_y)
            acc = d2l.ranking_accuracy(lambda x: net_D(x, yd), Xd,
                                       net_G(noise(kz, 512), yd))
            history.append((step, *map(float, logs), acc))
    return ema.model, net_D, np.array(history)
```
The runs use the penalty weight $\gamma = 0.5$. Penalty weights do not transfer between experiments: :numref:`sec_dcgan` uses $\gamma = 10$ for $64 \times 64$ sprites, whereas R3GAN decays its CIFAR-10 value from $0.05$ to $0.005$ over its four-day run. The right value depends on the resolution, the input scale, the critic, the batch size and the budget, so a new setup needs its own sweep; Exercise :exref:`ex_conditional-implement-concatenation-critic-one` runs one.

```{.python .input #conditional-training-with-the-chapter-s-loss-3}
%%tab pytorch
gamma = 0.5
ema_G, net_D, history = train_conditional(train_X, train_y, test_X, test_y,
                                          gamma)
loss_D, loss_G, pen, acc = history[-20:, 1:].mean(0)  # last 5,000 steps
print(f'critic loss {loss_D:.2f} (+ penalty {gamma / 2 * pen:.3f}), '
      f'generator loss {loss_G:.2f}, pair accuracy {acc:.2f}')
```
```{.python .input #conditional-training-with-the-chapter-s-loss-3}
%%tab jax
gamma = 0.5
ema_G, net_D, history = train_conditional(train_X, train_y, test_X, test_y,
                                          gamma)
loss_D, loss_G, pen, acc = history[-20:, 1:].mean(0)  # last 5,000 steps
print(f'critic loss {loss_D:.2f} (+ penalty {gamma / 2 * pen:.3f}), '
      f'generator loss {loss_G:.2f}, pair accuracy {acc:.2f}')
```
```{.python .input #conditional-training-with-the-chapter-s-loss-4}
%%tab pytorch, jax
fig, axes = d2l.plt.subplots(1, 2, figsize=(9, 3.2))
d2l.plot(history[:, 0], [history[:, 1], history[:, 2]], 'step', 'loss',
         legend=['critic (pairing part)', 'generator'], yscale='log',
         axes=axes[0])
axes[0].axhline(np.log(2), ls='--', c='gray', lw=1)
d2l.plot(history[:, 0], history[:, 4], 'step', 'held-out pair accuracy',
         ylim=[0.3, 1], axes=axes[1])
axes[1].axhline(0.5, ls='--', c='gray', lw=1);
```
The dashed lines mark $\log 2$, the value of either pairing loss when the critic cannot order a pair, and a pair accuracy of one half. Both losses settle within the first few thousand steps, after an early peak of the generator loss. The generator loss then stays near 1.0, above $\log 2$, and the critic's pairing loss near 0.5, below it, with the penalty adding about 0.06. Both readings say that the critic still wins most comparisons, and the pair accuracy agrees: late in training the critic orders about 80% of fresh same-class pairs correctly, down from about 90% in the first few thousand steps, with occasional brief dips. At this budget the accuracy does not approach one half, so on average over classes the generator has not caught up with the data.

The next grid tests generation on demand. Each row requests one class, and each column uses a fresh latent code. As in :numref:`sec_dcgan`, the generator has no tanh, so the display and every evaluation below clamp its output to $[-1, 1]$.

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
z = jax.random.normal(jax.random.key(42), (100, 100))
y = jnp.repeat(jnp.arange(10), 10)
imgs = (jnp.clip(ema_G(z, y), -1, 1) + 1) / 2
grid = np.asarray(imgs.reshape(10, 10, 32, 32, 3).transpose(0, 2, 1, 3, 4)
                  .reshape(320, 320, 3))
d2l.set_figsize((6, 6))
d2l.plt.imshow(grid)
d2l.plt.yticks([32 * i + 16 for i in range(10)], classes)
d2l.plt.xticks([]);
```
Most rows read as their class. Vehicles are recognizable by shape: cars and trucks as wheeled bodies, ships as hulls on water, airplanes against the sky. The animal rows lean more on palette and background: frogs are green and brown textures, deer and horses brown forms on grass, while birds, cats and dogs are often similar brown shapes. Pose, color and background vary along every row, so the grid shows no obvious collapse within a class. Inspection cannot say how often a generated image would be recognized as its class, or how much variety is missing; the next section measures both.

## Measuring Condition Alignment

### A Stable Referee

An unconditional metric cannot detect a mismatch between generated images and their requested conditions: the cyclic label shift preserves the distribution of images while violating every request. We measure the mismatch with a CIFAR-10 classifier, the *referee*. It is the small convolutional network `d2l.FeatureCNN` of :numref:`sec_dcgan`, trained at the native $32 \times 32$ resolution by `d2l.train_feature_cnn` with augmentation and an annealed rate: random horizontal flips, random shifts of up to four pixels, and a learning rate that rises to $3 \cdot 10^{-3}$ and anneals over 15 epochs. The augmentation and the schedule are there for stability. Trained without them, the same network overfits, and its accuracy on a single class moves by tens of points between two training seeds, more than the differences we want to read; with them, it reaches about 78% and its per-class accuracies move by a few points.

For each generated image, the referee either predicts the requested class or it does not. The fraction of matches, overall and per class, is the *condition alignment*. We compare it with the referee's accuracy on real test images of the same class, because the referee also misclassifies real images. The real accuracy is a reference, not an upper bound: unusually prototypical generated images can be easier to classify than photographs.

```{.python .input #conditional-measuring-condition-alignment-1}
%%tab pytorch
torch.manual_seed(0)
cnn = d2l.FeatureCNN(resolution=32).to(device)
d2l.train_feature_cnn(cnn, train_X, train_y, shift=4)
with torch.no_grad():
    pred = torch.cat([cnn(test_X[i:i + 1000]).argmax(1)
                      for i in range(0, len(test_X), 1000)])
cls_acc = [float((pred[test_y == c] == c).float().mean())
           for c in range(10)]
print(f'CIFAR-10 test accuracy: {float((pred == test_y).float().mean()):.3f}')
```
```{.python .input #conditional-measuring-condition-alignment-1}
%%tab jax
cnn = d2l.FeatureCNN(resolution=32, rngs=nnx.Rngs(0))
d2l.train_feature_cnn(cnn, train_X, train_y, shift=4)
pred = jnp.concatenate([cnn(test_X[i:i + 1000]).argmax(1)
                        for i in range(0, len(test_X), 1000)])
cls_acc = [float((pred[test_y == c] == c).mean()) for c in range(10)]
print(f'CIFAR-10 test accuracy: {float((pred == test_y).mean()):.3f}')
```
### Alignment Gaps per Class

The next cell generates 500 images of each class and reports, per class, the alignment, the referee's accuracy on real images, and their difference, the *gap*.

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
print(f'{"class":12s}{"alignment":>11s}{"real":>7s}{"gap":>7s}')
for c in range(10):
    print(f'{classes[c]:12s}{align[c]:11.2f}{cls_acc[c]:7.2f}'
          f'{align[c] - cls_acc[c]:+7.2f}')
print(f'{"mean":12s}{np.mean(align):11.2f}{np.mean(cls_acc):7.2f}'
      f'{np.mean(align) - np.mean(cls_acc):+7.2f}')
```
```{.python .input #conditional-measuring-condition-alignment-2}
%%tab jax
key = jax.random.key(7)
align = []
for c in range(10):
    key, kz = jax.random.split(key)
    z = jax.random.normal(kz, (500, 100))
    yc = jnp.full((500,), c)
    pred_c = cnn(jnp.clip(ema_G(z, yc), -1, 1)).argmax(1)
    align.append(float((pred_c == c).mean()))
print(f'{"class":12s}{"alignment":>11s}{"real":>7s}{"gap":>7s}')
for c in range(10):
    print(f'{classes[c]:12s}{align[c]:11.2f}{cls_acc[c]:7.2f}'
          f'{align[c] - cls_acc[c]:+7.2f}')
print(f'{"mean":12s}{np.mean(align):11.2f}{np.mean(cls_acc):7.2f}'
      f'{np.mean(align) - np.mean(cls_acc):+7.2f}')
```
Averaged over classes, alignment is about 0.7, five or six points below the referee's accuracy of about 0.78 on real images. No class is recognized much more often than its real counterpart. The hardest real class, cat, is also the hardest generated class. Individual gaps move by several points between runs, so the table does not rank them.

The comparison is not a calibration. The referee is evaluated under a distribution shift, so its errors on generated and real images need not have the same causes, and a zero gap may combine compensating errors. The gap is a proxy for obedience, but with this referee a reproducible one.

### Fidelity and Within-Class Diversity

Alignment does not measure image quality. We therefore also compute the two feature-distribution distances of :numref:`sec_dcgan`: the Fréchet distance between Gaussian fits to the two feature sets (:eqref:`eq_gan_fid`) and the unbiased $\mathrm{MMD}^2$ estimate with a polynomial kernel (:eqref:`eq_gan_kid`). Both use the referee's 128-dimensional features rather than Inception features, so we call them FD and $\mathrm{MMD}^2$ (CIFAR-CNN); they cannot be compared with published FID or KID scores. The evaluation compares 1,000 generated images, 100 per class, with 1,000 held-out test images in the same proportions, and a second disjoint set of 1,000 test images gives the finite-sample reference. The distances ignore the labels, so the cell also reports the *within-class feature variance*: the variance of the features within each class, averaged over features and classes, for the generated set and for both real sets.

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
print(f'{"real vs. real":22s}{d2l.fid_score(f_real2, f_real):16.2f}'
      f'{d2l.kid_score(f_real2, f_real):19.3f}{"--":>14s}')
print(f'{"generated vs. real":22s}{d2l.fid_score(f_gen, f_real):16.2f}'
      f'{d2l.kid_score(f_gen, f_real):19.3f}{oob:14.3f}')
cvar = lambda f: float(np.float64(f).reshape(10, 100, -1).var(1).mean())
print(f'within-class feature variance: generated {cvar(f_gen):.3g}, '
      f'real {cvar(f_real):.3g}, second real set {cvar(f_real2):.3g}')
```
```{.python .input #conditional-measuring-condition-alignment-4}
%%tab jax
def features(imgs, bs=500):
    return np.concatenate([np.asarray(cnn.features(imgs[i:i + bs]))
                           for i in range(0, len(imgs), bs)])

z = jax.random.normal(jax.random.key(11), (1000, 100))
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
print(f'{"real vs. real":22s}{d2l.fid_score(f_real2, f_real):16.2f}'
      f'{d2l.kid_score(f_real2, f_real):19.3f}{"--":>14s}')
print(f'{"generated vs. real":22s}{d2l.fid_score(f_gen, f_real):16.2f}'
      f'{d2l.kid_score(f_gen, f_real):19.3f}{oob:14.3f}')
cvar = lambda f: float(np.float64(f).reshape(10, 100, -1).var(1).mean())
print(f'within-class feature variance: generated {cvar(f_gen):.3g}, '
      f'real {cvar(f_real):.3g}, second real set {cvar(f_real2):.3g}')
```
In FD the generated set lies two to four times as far from the real test images as the second real set does. In $\mathrm{MMD}^2$ the real-versus-real estimate fluctuates around zero, while the generated-versus-real estimate is small and positive. The fit is measurably imperfect but close in this feature space. The variance line adds what the distances cannot separate: the generated images have about 20% less within-class feature variance than the real ones, while a second real set comes within 5%. The generator has lost some of the variety within each class, and alignment cannot see that loss.

### Two Baselines

Two constructed generators show what each measurement can and cannot see. The first keeps the same 1,000 generated images and shifts their *requested* labels cyclically, the counterexample from the start of this section. The second is a degenerate generator that returns, for every request of class $c$, the real test image that the referee classifies as $c$ most confidently.

```{.python .input #conditional-measuring-condition-alignment-5}
%%tab pytorch
y_perm = (yg + 1) % 10                       # same images, shifted requests
bal = torch.cat([i[:100] for i in idx])      # 100 test images per class
ar = torch.arange(1000, device=device)
with torch.no_grad():
    pred_gen = cnn(gen).argmax(1)
    p = cnn(test_X[bal]).softmax(1)[ar, test_y[bal]].reshape(10, 100)
    conf = bal.reshape(10, 100)[torch.arange(10), p.argmax(1)]
    fixed = test_X[conf].repeat_interleave(100, dim=0)  # one image per class
    pred_fix = cnn(fixed).argmax(1)
print(f'permuted labels: alignment '
      f'{float((pred_gen == yg).float().mean()):.2f} -> '
      f'{float((pred_gen == y_perm).float().mean()):.2f}, '
      f'FD {d2l.fid_score(f_gen, f_real):.2f}, '
      f'MMD^2 {d2l.kid_score(f_gen, f_real):.3f} (both unchanged)')
print(f'most confident image per class: alignment '
      f'{float((pred_fix == yg).float().mean()):.2f}, '
      f'within-class variance {cvar(features(fixed)):.3g}')
```
```{.python .input #conditional-measuring-condition-alignment-5}
%%tab jax
y_perm = (yg + 1) % 10                       # same images, shifted requests
bal = jnp.concatenate([i[:100] for i in idx])  # 100 test images per class
ar = jnp.arange(1000)
pred_gen = cnn(gen).argmax(1)
p = jax.nn.softmax(cnn(test_X[bal]))[ar, test_y[bal]].reshape(10, 100)
conf = bal.reshape(10, 100)[jnp.arange(10), p.argmax(1)]
fixed = jnp.repeat(test_X[conf], 100, axis=0)  # one image per class
pred_fix = cnn(fixed).argmax(1)
print(f'permuted labels: alignment '
      f'{float((pred_gen == yg).mean()):.2f} -> '
      f'{float((pred_gen == y_perm).mean()):.2f}, '
      f'FD {d2l.fid_score(f_gen, f_real):.2f}, '
      f'MMD^2 {d2l.kid_score(f_gen, f_real):.3f} (both unchanged)')
print(f'most confident image per class: alignment '
      f'{float((pred_fix == yg).mean()):.2f}, '
      f'within-class variance {cvar(features(fixed)):.3g}')
```
Shifting the requests drops alignment from about 0.7 to a few percent, while FD and $\mathrm{MMD}^2$ do not change, because neither receives a label. The confident-image generator reaches alignment 1.00 with zero within-class variance. It beats real images on alignment and is the most collapsed generator possible: one image per request, a *within-class collapse*, which is mode collapse inside each conditional distribution. The conditional objective :eqref:`eq_gan_cond_value` penalizes it, because a point mass is far from $p(\cdot \mid c)$, and so does the within-class variance; alignment rewards it, and the pooled distances see it only after mixing the classes.

Conditional evaluation thus has three axes: fidelity, diversity, and alignment. The feature distances mix fidelity with coverage and ignore the labels, alignment sees only agreement with the request, and the within-class variance sees only spread. Two established metrics combine sample quality with the condition. The Fréchet joint distance embeds image and condition together, so a realistic image with the wrong condition changes the score :cite:`DeVries.Romero.Pineda.ea.2019`. The classification accuracy score trains a classifier on generated pairs and tests it on real ones, so a collapsed or missing class shows up as errors on real images of that class :cite:`Ravuri.Vinyals.2019`. Exercise :exref:`ex_conditional-diagnostics-cell-detects-within` extends the diagnosis to per-class distances, and Exercise :exref:`ex_conditional-biggan-s-truncation-trick` uses the same instruments to measure the truncation trick of :citet:`Brock.Donahue.Simonyan.2019`, which trades variety for fidelity on purpose.

### The Critic's Label Signal

The critic, too, was trained to read the label. Does it still have something to tell the generator about labels? By :eqref:`eq_gan_cond_bayes`, the label term of the optimal critic is $\log(p(c \mid x)/q(c \mid x))$, which vanishes for every $c$ once the generator obeys every request. We test two things on 100 real test images per class. The first is the fraction of images that score higher under their own label $c$ than under the next one, $D(x, c) > D(x, c{+}1)$; above one half, the critic still prefers the true label and has a label signal to pass on. The second uses the critic as a classifier, $\arg\max_c D(x, c)$. The constants $\zeta_c$ that the pairing loss leaves free distort this rule, so we also remove them, subtracting from $D(x, c)$ the critic's mean score on the generated images of class $c$.

```{.python .input #conditional-the-critic-s-label-signal}
%%tab pytorch
yb = test_y[bal]
with torch.no_grad():
    S = torch.stack([net_D(test_X[bal], torch.full_like(yb, c))
                     for c in range(10)], 1)       # S[i, c] = D(x_i, c)
    zeta = net_D(gen, yg).reshape(10, 100).mean(1)  # per-class constants
hit = lambda s: float((s.argmax(1) == yb).float().mean())
print(f'P(D(x, c) > D(x, c+1)) = '
      f'{float((S[ar, yb] > S[ar, (yb + 1) % 10]).float().mean()):.2f}')
print(f'critic as a classifier: raw {hit(S):.2f}, '
      f'constants removed {hit(S - zeta):.2f}')
```
```{.python .input #conditional-the-critic-s-label-signal}
%%tab jax
yb = test_y[bal]
S = jnp.stack([net_D(test_X[bal], jnp.full_like(yb, c))
               for c in range(10)], 1)             # S[i, c] = D(x_i, c)
zeta = net_D(gen, yg).reshape(10, 100).mean(1)      # per-class constants
hit = lambda s: float((s.argmax(1) == yb).mean())
print(f'P(D(x, c) > D(x, c+1)) = '
      f'{float((S[ar, yb] > S[ar, (yb + 1) % 10]).mean()):.2f}')
print(f'critic as a classifier: raw {hit(S):.2f}, '
      f'constants removed {hit(S - zeta):.2f}')
```
Depending on the run, between half and two thirds of real images score higher under their own label than under the next one, close to the value of one half that the equilibrium would give, so this pairwise test barely registers the label information. The per-class constants hide it. Used as a raw classifier, the critic labels 10% to 25% of the images correctly, against 10% for chance, because the scores of each class carry their own constant. Removing the constants raises its accuracy to between a quarter and a third, two to three times chance, so the critic still has label information to pass to the generator. This is the shift invariance of :numref:`sec_gan_relativistic` at work in a trained network, one constant per class. The corrected critic is still a poor classifier, and it should be: its label term records where the generator still disagrees with the data, not what an image shows (Exercise :exref:`ex_conditional-critic-classifier`).

We can now ask a GAN for a class and check what we got: whether it obeys, how much it varies, and how close it comes to the data. The image-to-image systems of this section ask for more, an output that matches a known target, and pix2pix still adds a critic to its $L_1$ loss. Why a learned comparison helps when a regression target exists is the question of :numref:`sec_gan_beyond`.

## Summary

Conditional generation gives the request to both players and plays the chapter's game on sample--condition pairs. Its optimal critic is the per-class log density ratio $\log(p(x \mid c)/q(x \mid c))$, its value averages the per-class divergences, and so it detects a generator that returns perfect images with the wrong labels, which the unconditional game cannot. With same-class pairs the pairing loss plays the game of :numref:`sec_gan_relativistic` in every class: the critic estimates the per-class log ratio up to one free constant per class, compares a real and a generated image of the same class, and is constrained by $R_1$ and $R_2$ on the images. Bayes' rule splits that ratio into a label term and a realness term, which gives the projection critic $D(x, c) = e_c^\top \phi(x) + \psi_0(x)$. On CIFAR-10 the referee recognizes the requested class in about 70% of the generated images, against 78% of real images, and the generated images have about a fifth less within-class variety than real data. The two baselines show that alignment, feature distances and variance each see a different failure.

## Exercises

1. [code] **Two slices with known densities.** Let $c \in \{1, 2\}$ with
   equal probability, $p(\cdot \mid 1) = \mathcal{N}(-2, 1)$ and
   $p(\cdot \mid 2) = \mathcal{N}(2, 1)$. The generator is correct on one
   slice and wrong on the other: $q(\cdot \mid 1) = \mathcal{N}(0, 1)$ and
   $q(\cdot \mid 2) = \mathcal{N}(2, 1)$.
    1. Compute the value :eqref:`eq_gan_cond_value` by numerical
       integration on a grid.
    1. Train a small critic that receives $x$ together with a one-hot
       encoding of $c$ on samples from the two joint distributions, and
       minimize the balanced loss

       $$\tfrac12\, \mathbb{E}_{(x, c) \sim p}\big[\log(1 + e^{-D(x, c)})\big] + \tfrac12\, \mathbb{E}_{c \sim p(c),\; x' \sim q(\cdot \mid c)}\big[\log(1 + e^{D(x', c)})\big].$$

       Show that the minimum of this loss is $-\tfrac12 \max_D V(D)$ for
       the game :eqref:`eq_gan_cond_V`, and compare the converged loss with
       the value from (a).
    1. Plot the trained critic and the log density ratio
       :eqref:`eq_gan_cond_dstar` against $x$ for each condition, over the
       range where that slice's samples lie. How large is the critic on the
       samples of the matched slice $c = 2$?
    1. Now let the generator swap the labels, $q(\cdot \mid 1) =
       \mathcal{N}(2, 1)$ and $q(\cdot \mid 2) = \mathcal{N}(-2, 1)$.
       Compute the conditional value, and the value of the unconditional
       game of :numref:`sec_basic_gan` on the same samples with their
       labels discarded. Retrain the critic in both settings and compare the
       converged losses with your computed values.
   :exlabel:`ex_conditional-verify-decomposition-eqref-eq`
1. **Separate feature maps.** The derivation of
   :eqref:`eq_gan_cond_projection` modeled both class posteriors with one
   shared feature map $\phi$. Suppose instead that the two posteriors
   require different feature maps,
   $p(c \mid x) \propto \exp(u_c^\top \phi_p(x))$ and
   $q(c \mid x) \propto \exp(w_c^\top \phi_q(x))$ with
   $\phi_p \neq \phi_q$.
    1. Derive the critic that takes the place
       of :eqref:`eq_gan_cond_projection`.
    1. Show that this critic is a projection head over the concatenated
       feature map $(\phi_p(x), \phi_q(x))$, and give its class
       embeddings.
    1. Let there be $K$ classes. Show that every function $D(x, c)$ can be
       written in the form :eqref:`eq_gan_cond_projection` once $\phi$
       has at least $K$ dimensions. How many dimensions does $\phi$ have
       in `Discriminator`?
    1. Given these results, what does the shared-map assumption restrict in
       the CIFAR-10 experiment? When does the projection form restrict the
       set of critics?
   :exlabel:`ex_conditional-projection-derivation-modeled-both`
1. [code] **The critic as a classifier.** The trained critic `net_D`
   scores every pair of an image and a class, so it assigns each image ten
   scores.
    1. Use :eqref:`eq_gan_cond_bayes` to express
       $D^\star(x, c) - D^\star(x, c')$ for the optimal critic in terms of
       the real and the generated label posteriors. Which quantity does the
       rule $\arg\max_c D^\star(x, c)$ maximize? How does it differ from the
       rule $\arg\max_c p(c \mid x)$ of a classifier such as `cnn`? What
       does the difference of scores equal when
       $q(\cdot \mid c) = p(\cdot \mid c)$ for every $c$?
    1. The label check of this section removes the constants $\zeta_c$ and
       classifies with $\arg\max_c (D(x, c) - \hat\zeta_c)$. Using (a),
       explain why this classifier must fall to chance as the generator
       improves, however well `cnn` classifies. Compute its accuracy per
       class on the balanced test images and compare the ranking of the
       classes with the ranking by alignment. Which agreement would you
       expect, and do you find it?
    1. Suppose that after much longer training the printed fraction
       $P(D(x, c) > D(x, c+1))$ on real images fell to one half. Name two
       different states of the game that would produce this. Which
       measurement of this section tells them apart?
   :exlabel:`ex_conditional-critic-classifier`
1. [code] **Per-class distances.** Per-class feature variance detects
   within-class collapse but not other differences between a generated
   class and its real counterpart. The features `f_gen`, `f_real` and
   `f_real2` hold 100 images of each class, in class order.
    1. A covariance matrix estimated from 100 samples of the 128-dimensional
       features of `FeatureCNN` has rank at most 99. How does this affect a
       per-class Fréchet distance :eqref:`eq_gan_fid`? Why does the unbiased
       $\mathrm{MMD}^2$ estimator :eqref:`eq_gan_kid` not suffer in the same
       way?
    1. For each class, compute $\mathrm{MMD}^2$ between the generated and
       the real features, and between the two real sets. Rank the classes by
       the difference of the two values and compare this ranking with the
       ranking by alignment. If a class is well aligned but far from its real
       distribution, or the reverse, display its samples.
    1. Compute the same per-class quantities for the collapse baseline. A
       conditional generator may ignore the condition, obey it through
       shortcuts such as background or palette, collapse within a condition,
       leak between conditions, neglect a rare condition, or optimize for
       classifier prototypes. Which of these failures does each of the three
       per-class diagnostics (alignment, feature variance, and
       $\mathrm{MMD}^2$) detect?
   :exlabel:`ex_conditional-diagnostics-cell-detects-within`
1. [code] **Truncated latents.** The truncation trick of
   :citet:`Brock.Donahue.Simonyan.2019` draws each latent coordinate from a
   standard Gaussian truncated to $[-\tau, \tau]$, resampling any coordinate
   that falls outside. Such a coordinate has variance
   $1 - \sqrt{2/\pi}\,\tau e^{-\tau^2/2} / \operatorname{erf}(\tau/\sqrt{2})$.
    1. For the 100-dimensional latent code of `Generator` and
       $\tau \in \{0.5, 1, 2\}$, compute the probability that an untruncated
       draw lies entirely inside the box $[-\tau, \tau]^{100}$, and the
       typical norm of a truncated draw. Compare this norm with that of the
       latent codes seen in training.
    1. Implement truncated sampling for `ema_G` and sweep
       $\tau \in \{0.5, 1, 2, \infty\}$, where $\tau = \infty$ is ordinary
       sampling. For each $\tau$, report the alignment, the per-class
       feature variance, and the Fréchet distance to `f_real`.
    1. BigGAN reports that only 16% of its models trained without
       orthogonal regularization were amenable to truncation, that is, free
       of saturation artifacts on truncated noise, against 60% with it.
       `ema_G` was trained without orthogonal regularization. Display its
       samples at each $\tau$. Compare the $\tau$ at which alignment,
       feature variance, or appearance changes most with the norms from (a).
       What do these measurements show about how this generator treats
       latent codes with norms it did not see in training?

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problem 15.6.*
   :exlabel:`ex_conditional-biggan-s-truncation-trick`
1. **Mismatched label marginals.** ● Suppose that real pairs follow
   $p(x, c)$, while the generator's conditions are drawn from a distribution
   $r(c)$ that differs from $p(c)$, so that generated pairs follow
   $r(c)\, q(x \mid c)$ and the second expectation of
   :eqref:`eq_gan_cond_V` draws $c$ from $r(c)$. By exercise
   :exref:`ex_gan-unequal-priors` of :numref:`sec_basic_gan`, in a log-loss
   game in which a sample is real with prior probability $\alpha$, the
   optimal critic satisfies
   $\sigma(D^\star) = \alpha p / (\alpha p + (1 - \alpha) q)$, and the value
   is, up to constants, the skewed divergence

   $$\alpha\, D_{\mathrm{KL}}(p \,\|\, m_\alpha) + (1 - \alpha)\, D_{\mathrm{KL}}(q \,\|\, m_\alpha), \qquad m_\alpha = \alpha p + (1 - \alpha) q.$$

    1. Show that the terms of this game with condition $c$ form such a game
       between $p(\cdot \mid c)$ and $q(\cdot \mid c)$,
       with prior $\alpha_c = p(c) / (p(c) + r(c))$ and weight
       $p(c) + r(c)$. Recover the offset $\log(p(c)/r(c))$ of the optimal
       critic from $\alpha_c$.
    1. Does the mismatch change the generator's optimum? What does it
       change instead?
    1. `train_conditional` conditions each generated batch on the labels of
       a real batch and compares same-class pairs. What becomes of the
       offset $\log(p(c)/r(c))$ of (a) in this pairing game? Which label
       distribution weights the slices in this scheme?
   :exlabel:`ex_conditional-suppose-label-marginal-is`
1. [code] **A rare class.** ● Let one of the ten classes, for instance
   automobiles, keep only 50 of its 5,000 training images, so that
   $p(c_{\textrm{rare}}) = \epsilon = 50/45{,}050 \approx 0.0011$, and keep
   all images of the other classes.
    1. Under the natural label marginal, show that a generator that serves
       the rare slice arbitrarily badly raises :eqref:`eq_gan_cond_value` by
       at most $2 \epsilon \log 2$. Class-balanced sampling of the real
       batches raises the weight of the slice to $1/10$. Which risk for the
       rare class does class-balanced sampling introduce, given that its
       slice contains only 50 real images?
    1. Train with `train_conditional` at a budget of 4,000 steps under two
       schemes: real batches drawn uniformly from the reduced training set,
       and class-balanced real batches, whose labels then condition the
       generator uniformly. As a reference, train at the same budget on the
       full training set.
    1. For the rare class, report the alignment, the per-class
       $\mathrm{MMD}^2$ of problem
       :exref:`ex_conditional-diagnostics-cell-detects-within`b, and the
       feature distances of its generated samples to their nearest neighbors
       among its 50 training images, as in the memorization
       check of :numref:`subsec_gan_limited_data`. Which scheme serves the rare class
       better, and what does it cost the other classes?
   :exlabel:`ex_conditional-rare-class`
1. **Auxiliary classifier bias.** ● An auxiliary classifier critic
   :cite:`Odena.Olah.Shlens.2017` contains a classifier $C(c \mid x)$, and
   the generator minimizes, in addition to its adversarial loss, the
   classification loss
   $\mathbb{E}_{c \sim p(c),\, x \sim q(\cdot \mid c)}[-\log C(c \mid x)]$ of its own
   samples. Write $q(x)$ for the marginal of the generated images and
   $q(c \mid x)$ for the label posterior of the generated pairs.
    1. Show that this loss equals

       $$\mathbb{E}_{x \sim q}\Big[H\big(q(\cdot \mid x)\big) + D_{\mathrm{KL}}\big(q(\cdot \mid x) \,\|\, C(\cdot \mid x)\big)\Big],$$

       where $H$ is the entropy.
    1. Hold $C$ fixed and set $q(x) = p(x)$. Let $q(\cdot \mid x)$ vary
       freely at every $x$, ignoring the constraint that the generated labels
       follow $p(c)$. Which label posterior $q(\cdot \mid x)$ minimizes the
       loss? Let $C$ be the real posterior $p(c \mid x)$, and let two
       classes overlap. Show that the generator can lower the loss below its
       value at $q = p$.
    1. Take the expectation of the compatibility term
       $\log [p(c \mid x) / q(c \mid x)]$ of :eqref:`eq_gan_cond_bayes`
       under the generated pairs. Which term of the loss in (a) does the
       factor $q(c \mid x)$ cancel?
    1. :citet:`Gong.Xu.Li.ea.2019` add a second classifier that is trained
       on generated pairs only. Which quantity must it estimate, and with
       which sign does its classification loss enter the generator's
       objective?
   :exlabel:`ex_conditional-auxiliary-classifier-bias`
1. [code] **The critic at a fixed budget.** ● `train_conditional` fixes two
   choices for the critic: the penalty weight $\gamma = 0.5$ and the
   projection head.
    1. Train for 4,000 steps at $\gamma \in \{0.05, 0.5, 5\}$. Report the
       logged sum of the squared input gradients on real and generated
       images (the column `history[:, 3]`), the alignment, and the within-class
       feature variance. Which of these change by an order of magnitude
       across the sweep, and which barely move? $\gamma$ multiplies a squared
       input gradient, so it has the units of a squared distance in pixel
       space (:numref:`sec_gan_convergence`); why does the value that suits
       $32 \times 32$ CIFAR-10 images differ from the $\gamma = 10$ of the
       sprites?
    1. Replace the projection head of `Discriminator` by concatenation:
       one-hot encode the label, broadcast it to ten constant $32 \times 32$
       planes, append these planes to the image as extra input channels, and
       score with the head $\psi_0$ alone. Add the critic class as an
       argument of `train_conditional`, and train both critics for 4,000
       steps with two seeds each. Compare the alignment, the feature
       distances, and the critic's accuracy as a classifier with its
       constants removed.
    1. By problem
       :exref:`ex_conditional-projection-derivation-modeled-both`c, the
       projection head restricts no critic for ten classes, and a network
       that receives the label planes as input is unrestricted in principle
       as well. What can then explain a difference between the two critics
       at a fixed budget?
   :exlabel:`ex_conditional-implement-concatenation-critic-one`
1. **Identifiability in translation.** ● In paired translation the critic
   sees source--output pairs; in unpaired translation each critic sees only
   its target collection.
    1. In the paired setting of pix2pix, replace the conditional critic by a
       marginal critic that sees only the output, and drop the L1 term. Show
       that a generator that ignores its input attains the optimum of this
       game. What do the conditional critic and the L1 term each restore?
    1. For CycleGAN, suppose that $G: X \to Y$ and $F: Y \to X$ have zero
       cycle loss, that $G$ maps $p_X$ to $p_Y$, and that $F$ maps $p_Y$ to
       $p_X$. Let $B: Y \to Y$ be an invertible map that leaves $p_Y$
       unchanged. Show that the pair $(B \circ G,\, F \circ B^{-1})$
       also has zero cycle loss and satisfies both marginal critics. Give
       two such solutions for $p_X = p_Y = \mathcal{N}(0, 1)$ on the real
       line, and count the solutions when $X$ and $Y$ each consist of $n$
       equally likely images.
    1. Suppose that $Y$ carries less information than $X$, as when many
       aerial photographs share one street map. Why can a pair with
       near-zero cycle loss on $X \to Y \to X$ not produce maps that depend
       on the photograph only through its map content? Where must the
       missing information go, and why does the marginal critic of $Y$ not
       object? :citet:`Chu.Zhmoginov.Sandler.2017` examine this behavior in
       a trained CycleGAN.
   :exlabel:`ex_conditional-translation-identifiability`

[Discussions](https://d2l.discourse.group/t/36813)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.6]{.kicker}

Conditional generation<br>
**the game on pairs · same-class pairs · the projection head · class-conditional CIFAR-10 · measuring obedience**
:::
:::

::: {.slide title="Perfect Images with Every Label Wrong Fool the Unconditional Game"}
[The problem]{.kicker}

- A CIFAR-10 generator returns perfect images but shifts every label:
  asked for a bird, it returns a cat.
- The ten classes are equally frequent, so the distribution of images is
  exactly the data's.
- Unconditional game: value $-2\log 2$, its **minimum**.
- Every request fails.

::: {.d2l-note .warn}
To ask for a bird, both players must see the request. What does the game
then optimize, and how do we check that the generator obeys?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive the optimal conditional critic and the value of the game on
  sample--condition pairs;
- explain why real and generated images are paired within a class;
- derive the projection critic from Bayes' rule;
- train class-conditional CIFAR-10 with the pairing loss and both penalties;
- measure fidelity, within-class diversity and alignment separately.

**The idea:** play the chapter's game once per condition.
**Running example:** CIFAR-10, ten classes; ask for a bird, check for a bird.
:::

::: {.slide title="Every Image GAN So Far Ignored Requests"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | asks |
|---|---|
| Generative Adversarial Networks | what a classifier of samples computes |
| Adversarial Objectives and Divergences | which loss gives which distance |
| Relativistic Objectives | what changes when the critic ranks pairs |
| Gradient Penalties and Convergence | why training circles, and what stops it |
| Adversarial Image Generation | which network, and how to score images |
| **Conditional Generation** | **how to ask for a class** |
| Adversarial Losses Beyond GANs | where a learned comparison remains |
:::

::: {.col .narrow}
- **Uses:** the optimal critic $\log(p/q)$; the pairing loss and its
  shift invariance; $R_1$, $R_2$; the image backbone.
- **Skips:** text-to-image systems; translation implementations.
:::
:::
:::

::: {.slide title="The Critic Scores Sample–Condition Pairs"}
[The conditional game]{.kicker}

Data: pairs $(x, c) \sim p(x, c)$. Generator: $c \sim p(c)$, then
$x' = G(z, c) \sim q(\cdot \mid c)$. Critic: $D(x, c)$.

$$V(D) = \mathbb{E}_{(x,c) \sim p}[\log \sigma(D(x, c))]
+ \mathbb{E}_{c,\, x' \sim q(\cdot \mid c)}[\log \sigma(-D(x', c))]$$

- The log-loss game, with the pair $(x, c)$ in place of $x$.
- Both joint distributions share the label marginal $p(c)$.
:::

::: {.slide title="The Optimal Critic Compares Real and Generated Birds Only"}
[The conditional game]{.kicker}

$$D^\star(x, c) = \log \frac{p(c)\, p(x \mid c)}{p(c)\, q(x \mid c)}
= \log \frac{p(x \mid c)}{q(x \mid c)}$$

$$\max_D V = \mathbb{E}_c\big[2\, \mathrm{JS}(p(\cdot \mid c), q(\cdot \mid c))\big]
- 2 \log 2$$

- One game per condition, weighted by $p(c)$.
- Minimum iff $q(\cdot \mid c) = p(\cdot \mid c)$ for **every** $c$.
- A rare class carries a small weight, so it is served cheaply.
:::

::: {.slide title="Check: What Is the Conditional Value of the Label Shift?"}
[Check]{.kicker}

$q(\cdot \mid c) = p(\cdot \mid c{+}1)$. Suppose the ten classes had
disjoint supports. Unconditional value? Conditional value?

. . .

- Unconditional: the image distributions match, value $-2\log 2$, the
  minimum.
- Conditional: every slice has $\mathrm{JS} = \log 2$, so
  $2\log 2 - 2\log 2 = 0$, the **maximum**.
- The perfect image generator that ignores requests is the worst
  conditional generator.
:::

::: {.slide title="Pair a Real Bird with a Generated Bird"}
[Same-class pairs]{.kicker}

Within class $c$: the pairing game between $p(\cdot \mid c)$ and
$q(\cdot \mid c)$. Its value is the lifted JS, averaged over classes:

$$\mathbb{E}_c\big[\mathrm{JS}(p_c \otimes q_c,\; q_c \otimes p_c)\big],
\qquad p_c = p(\cdot \mid c),\; q_c = q(\cdot \mid c)$$

- **One free constant per class:** $D(x, c) + \zeta_c$ has the same loss.
- **Rank weight:** the probability that a generated bird outranks a real
  bird.
- **Paired loss:** with an optimal critic and a fixed one-to-one pairing,
  no piled-up configuration is a local minimum, now within each class.
:::

::: {.slide title="Three Ways for the Condition to Enter"}
[Conditioning mechanisms]{.kicker}

![](../img/mdl-gan-conditioning.svg)

Concatenation (any network) · modulation $a(c) \odot \phi + b(c)$
(generators) · a compatibility head (critics).
:::

::: {.slide title="Bayes' Rule Splits the Optimal Critic in Two"}
[The projection head]{.kicker}

$$\log \frac{p(x \mid c)}{q(x \mid c)}
= \underbrace{\log \frac{p(c \mid x)}{q(c \mid x)}}_{\textrm{label term}}
+ \underbrace{\log \frac{p(x)}{q(x)}}_{\textrm{realness term}}$$

- Wrong labels on good images: the label term is off.
- Poor images of the right class: the realness term is off.
:::

::: {.slide title="Softmax Posteriors on Shared Features Give an Inner Product"}
[The projection head]{.kicker}

$p(c \mid x) \propto e^{u_c^\top \phi(x)}$ and
$q(c \mid x) \propto e^{w_c^\top \phi(x)}$; the normalizers do not depend
on $c$:

$$D(x, c) = e_c^\top \phi(x) + \psi_0(x), \qquad e_c = u_c - w_c$$

- One embedding per class, one inner product, one unconditional head.
- Linear $\psi_0$ folds into the $e_c$: R3GAN keeps only the inner product.
- A caption embedding in place of $e_c$: the text-conditioned critic of
  StyleGAN-T.
:::

::: {.slide title="The Projection Head Is Two Lines of the Forward Pass"}
[The projection head]{.kicker}

```{.python #conditional-exercises}
phi = self.phi(x)                      # summed final feature map
label = (self.embed(y) * phi).sum(-1)  # e_c . phi(x), c = y per image
return label + self.psi0(phi)[:, 0]    # + psi_0(x)
```

- The class table starts at zero: the critic starts unconditional.
- The generator concatenates a class embedding at the $4 \times 4$ stage.
:::

::: {.slide title="Check: Which Training Statistic Can Drift Without the Game Changing?"}
[Check]{.kicker}

The pairing loss is unchanged by $D(x, c) \to D(x, c) + \zeta_c$ for any
constants $\zeta_c$. Which can drift: the mean critic score on real birds,
or the fraction of (real bird, generated bird) pairs ordered correctly?

::: {.fragment .fade-up}
The mean score drifts with $\zeta_c$. The pair fraction compares two
scores of the same class, so $\zeta_c$ cancels: monitor that one.
:::
:::

::: {.slide title="The Recipe Is the Chapter's, with Labels Shared Across Each Pair"}
[Class-conditional CIFAR-10]{.kicker}

- Backbone: the minimal image backbone at $32 \times 32$, fan-in
  initialization, no normalization.
- Loss: pairing loss for both players, $\gamma = 0.5$ on $R_1 + R_2$.
- Pairs: the real batch's labels condition the generated batch.
- Adam $(0, 0.99)$, $2 \cdot 10^{-4}$, batch 128, flips, EMA with
  half-life 500 steps; 15,000 steps on one GPU.
- Monitor: held-out pair accuracy $\hat P(D(x, c) > D(x', c))$, free of
  the constants $\zeta_c$.
:::

::: {.slide title="The Critic Still Orders 80% of Same-Class Pairs"}
[Class-conditional CIFAR-10]{.kicker}

@!conditional-training-with-the-chapter-s-loss-4

- Losses settle within a few thousand steps: generator near 1.0, critic's
  pairing loss near 0.5 ($\log 2 \approx 0.69$ dashed).
- Pair accuracy falls from about 90% to about 80%: the generator improves
  but has not caught up on average.
:::

::: {.slide title="Each Row Is a Request"}
[Class-conditional CIFAR-10]{.kicker}

@!conditional-training-with-the-chapter-s-loss-5

- Vehicles read by shape; animals lean on palette and background.
- Pose, color and background vary along every row.
:::

::: {.slide title="A Referee That Gives the Same Verdict Twice"}
[Measuring alignment]{.kicker}

- Referee: a small CNN classifies each generated image; alignment is the
  fraction labeled as requested.
- Trained plainly, its accuracy on one class moves by tens of points
  between seeds.
- With flips, 4-pixel shifts and a one-cycle rate over 15 epochs: about
  78%, per-class accuracies within a few points.
- Reference: the referee's accuracy on real images of the same class.
:::

::: {.slide title="Generated Images Are Recognized Five or Six Points Less Often"}
[Measuring alignment]{.kicker}

@!conditional-measuring-condition-alignment-2

- Mean alignment about 0.7 against 0.78 on real images.
- No class far above its real reference; cat is hardest in both.
- Single gaps move by several points between runs: do not rank them.
:::

::: {.slide title="The Generator Has About a Fifth Less Variety within a Class"}
[Fidelity and diversity]{.kicker}

@!conditional-measuring-condition-alignment-4

- FD: two to four times the real-versus-real value.
- Within-class variance about 20% below real; a second real set is
  within 5%.
- Alignment cannot see this loss.
:::

::: {.slide title="Each Baseline Fools a Different Measurement"}
[Two baselines]{.kicker}

@!conditional-measuring-condition-alignment-5

- Shifted requests: alignment collapses; FD and MMD$^2$ do not move.
- One confident real image per class: alignment 1.00, variance 0.
:::

::: {.slide title="The Critic Still Reads the Labels, up to One Constant per Class"}
[The critic's label signal]{.kicker}

@!conditional-the-critic-s-label-signal

- Own label above the next label: 50–65% of real images, close to the
  equilibrium's 50%.
- Raw $\arg\max_c D(x, c)$: 10–25%, against 10% chance; the free $\zeta_c$
  distort it.
- Constants removed: a quarter to a third, two to three times chance;
  the label signal is there.
:::

::: {.slide title="Check: Which Measurement Catches Which Failure?"}
[Check]{.kicker}

A generator (a) shifts every label, (b) returns one image per class,
(c) returns blurry images of the right class. Which of alignment,
FD/MMD$^2$, within-class variance flags each?

. . .

- (a) alignment only.
- (b) variance (and FD, after mixing the classes); alignment is perfect.
- (c) FD/MMD$^2$; alignment may stay high if the shapes survive.
:::

::: {.slide title="An Image Can Be the Condition"}
[Images as conditions]{.kicker}

- **pix2pix** (paired): conditional critic on output--input pairs plus an
  $L_1$ loss. $L_1$ recovers the structure but blurs ambiguous texture;
  pix2pix adds the critic for it.
- **CycleGAN** (unpaired): one unconditional critic per direction plus a
  cycle loss. The cycle loss constrains the map but does not identify it.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Conditioning plays the game per class: $D^\star = \log(p(x \mid c)/q(x \mid c))$,
  value $\mathbb{E}_c[2\,\mathrm{JS}] - 2\log 2$; a label shift is near the
  worst value.
- Same-class pairs: lifted JS per class, one free constant per class.
- Bayes' rule gives the projection critic $e_c^\top \phi(x) + \psi_0(x)$.
- CIFAR-10, 15,000 steps: alignment about 0.7 against 0.78 on real
  images; within-class variance about 20% below real.
- Alignment, feature distances and within-class variance each catch a
  different failure.

**Next:** why a learned comparison helps even when a regression target is
known.
:::
