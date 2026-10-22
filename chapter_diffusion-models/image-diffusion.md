# Diffusion Models for Images
:label:`sec_diffusion-images`

Nothing in the previous section depended on the data being two-dimensional.
The forward process, the bound, the noise-prediction loss, and the sampler
apply verbatim to images; what changes is the network, which must map an
image and a step index to an image, and the amount of computation. This
section trains the diffusion model of :numref:`sec_diffusion-ddpm` on
Fashion-MNIST with the U-Net of :numref:`sec_diffusion-annealed`, adds a
class label as a condition, and uses the label to demonstrate
classifier-free guidance, the mechanism by which diffusion models are
steered toward a class or a caption, and its predecessor, classifier
guidance, on the same requests and noise draws. It closes with the
developments that
turned this construction into the image generators in current use. The
presentation follows lecture 13 of :citet:`Kuleshov.2023`.

```{.python .input #image-diffusion-diffusion-models-for-images}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import copy
import torch
from torch import nn
```

```{.python .input #image-diffusion-diffusion-models-for-images}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## Data, Schedule, and Model

The images are scaled to $[-1, 1]$, as in :citet:`ho2020denoising`, so that
the data have roughly unit scale, the scale of the standard Gaussian from
which the reverse process starts; the noise levels of the schedule are
calibrated against that scale. The schedule is the one of
:citet:`ho2020denoising`: $T = 1000$ steps with $\beta_t$ increasing
linearly from $10^{-4}$ to $0.02$.

```{.python .input #image-diffusion-data-schedule-and-model-1}
%%tab pytorch
device = d2l.try_gpu()
fmnist = d2l.FashionMNIST(batch_size=128)
X = (fmnist.train.data.float()[:, None] / 255 * 2 - 1).to(device)
Y = fmnist.train.targets.to(device)
X_test = (fmnist.val.data.float()[:, None] / 255 * 2 - 1).to(device)
Y_test = fmnist.val.targets.to(device)

T = 1000
beta = torch.linspace(1e-4, 0.02, T, device=device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, 0)
```

```{.python .input #image-diffusion-data-schedule-and-model-1}
%%tab jax
fmnist = d2l.FashionMNIST(batch_size=128)
X = jnp.asarray(fmnist.train[0], jnp.float32)[..., None] / 255 * 2 - 1
Y = jnp.asarray(fmnist.train[1], jnp.int32)
X_test = jnp.asarray(fmnist.val[0], jnp.float32)[..., None] / 255 * 2 - 1
Y_test = jnp.asarray(fmnist.val[1], jnp.int32)

T = 1000
beta = jnp.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = jnp.cumprod(alpha)
```

The noise predictor is the U-Net `d2l.UNet` of
:numref:`sec_diffusion-annealed`, with the step index $t$ in place of the
noise-level index. To make the model conditional we add a second embedding
to the time embedding: a learned vector per class $c$, the label $y$ of
:eqref:`eq_diffusion-bayes-score` (`y` in the code). The embedding table
has one extra row, a *null label* $\varnothing$ that carries no class
information, and during training each example's label is replaced by the
null label with probability $0.1$. The network therefore learns two
predictors at once, the conditional
$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, c)$ and the
unconditional $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing)$;
this is the training procedure of classifier-free guidance
:cite:`Ho.Salimans.2022`, and "Sampling and Guidance" below explains why
both predictors are wanted.

```{.python .input #image-diffusion-data-schedule-and-model-2}
%%tab pytorch
NULL = 10  # the label index that carries no class information

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10, emb_dim=128):
        super().__init__()
        self.unet = d2l.UNet(emb_dim=emb_dim)
        self.embed = nn.Embedding(num_classes + 1, emb_dim)  # + the null label

    def forward(self, x, t, y):
        return self.unet(x, t, cond=self.embed(y))
```

```{.python .input #image-diffusion-data-schedule-and-model-2}
%%tab jax
NULL = 10  # the label index that carries no class information

class ConditionalUNet(nnx.Module):
    def __init__(self, num_classes=10, emb_dim=128, rngs=None):
        self.unet = d2l.UNet(emb_dim=emb_dim, rngs=rngs)
        self.embed = nnx.Embed(num_classes + 1, emb_dim,  # + null; unit-variance
                               embedding_init=nnx.initializers.normal(1.0),
                               rngs=rngs)  # init, as in PyTorch

    def __call__(self, x, t, y):
        return self.unet(x, t, cond=self.embed(y))
```

## Training

The loop is the training procedure of :numref:`sec_diffusion-ddpm` with
label dropout added. One further device is standard in diffusion training:
sampling uses an exponential moving average of the weights rather than the
weights themselves, because the parameters at any single step carry
optimization noise that averaging reduces
(:numref:`subsec_practice-weight-averaging`). :citet:`ho2020denoising` use
a decay of $0.9999$ over hundreds of thousands of updates; for a run of
$1500$ updates we let the decay rise as $(1 + s) / (10 + s)$ with the step
$s$, capped at $0.999$, so that the average is not dominated by the
initialization; it reaches $0.994$ by the last update, and the average is
dominated by the last few hundred updates. Each update costs a forward and
backward pass through the U-Net on a batch of $128$ images, and the run
takes about ten minutes on a laptop processor.

```{.python .input #image-diffusion-training}
%%tab pytorch
torch.manual_seed(0)
net = ConditionalUNet().to(device)
ema = copy.deepcopy(net)  # the exponential moving average of the weights
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
for step in range(1500):
    idx = torch.randint(0, len(X), (128,), device=device)
    x0, y = X[idx], Y[idx].clone()
    y[torch.rand(128, device=device) < 0.1] = NULL  # label dropout
    t = torch.randint(0, T, (128,), device=device)
    eps = torch.randn_like(x0)
    a = alpha_bar[t][:, None, None, None]
    loss = ((net(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, y) - eps) ** 2).mean()
    optimizer.zero_grad(), loss.backward(), optimizer.step()
    decay = min(0.999, (1 + step) / (10 + step))
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), net.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1 - decay)
    if (step + 1) % 500 == 0:
        print(f'step {step + 1}: loss {loss.item():.3f}')
```

```{.python .input #image-diffusion-training}
%%tab jax
key = jax.random.PRNGKey(0)
net = ConditionalUNet(rngs=nnx.Rngs(0))
graphdef, ema_state = nnx.split(net)  # the moving average of the weights
optimizer = nnx.Optimizer(net, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(net, optimizer, x0, y, t, eps):
    def loss_fn(model):
        a = alpha_bar[t][:, None, None, None]
        x_t = jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps
        return ((model(x_t, t, y) - eps) ** 2).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(net)
    optimizer.update(net, grads)
    return loss

@jax.jit
def ema_update(ema_state, state, decay):
    return jax.tree_util.tree_map(lambda e, p: decay * e + (1 - decay) * p,
                                  ema_state, state)

for step in range(1500):
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    idx = jax.random.randint(k1, (128,), 0, len(X))
    x0, y = X[idx], Y[idx]
    y = jnp.where(jax.random.uniform(k2, (128,)) < 0.1, NULL, y)  # label dropout
    t = jax.random.randint(k3, (128,), 0, T)
    loss = train_step(net, optimizer, x0, y, t, jax.random.normal(k4, x0.shape))
    ema_state = ema_update(ema_state, nnx.state(net),
                           min(0.999, (1 + step) / (10 + step)))
    if (step + 1) % 500 == 0:
        print(f'step {step + 1}: loss {loss:.3f}')
ema = nnx.merge(graphdef, ema_state)
```

## Sampling and Guidance

The sampler is ancestral sampling from :numref:`sec_diffusion-ddpm`, with
one addition that lets a single function serve every classifier-free
experiment below.
Conditioning a score model on a label follows from Bayes' rule, as
:eqref:`eq_diffusion-bayes-score` showed: the conditional score is the
unconditional score plus the input gradient of $\log p_t(c \mid \mathbf{x})$,
evaluated at the current noise level. *Classifier guidance*
:cite:`Dhariwal.Nichol.2021` supplies that gradient from a classifier trained
on noisy images and scales it by a factor $s > 1$ to strengthen the class
signal. *Classifier-free guidance* :cite:`Ho.Salimans.2022` removes the
classifier by applying Bayes' rule once more: the gradient of
$\log p_t(c \mid \mathbf{x})$ equals the difference between the conditional
and the unconditional score, and a network trained with label dropout
provides both. In the noise-prediction parameterization, where score and
noise differ by a constant factor, the guided prediction is

$$
\tilde{\boldsymbol{\epsilon}}(\mathbf{x}_t, t, c)
= \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing)
+ \gamma\, \big( \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing) \big),
$$
:eqlabel:`eq_diffusion-cfg`

which :numref:`fig_diffusion-guidance` draws as an extrapolation. With
$\gamma = 0$ the label is ignored and the sampler is unconditional; with
$\gamma = 1$ it uses the conditional prediction, which is the model's own
estimate of the class-conditional score; with $\gamma > 1$ it moves past the
conditional prediction along the class direction, which sharpens the class
identity of the samples and reduces their diversity.
:citet:`Ho.Salimans.2022` write the same rule with a guidance weight $w$ as
$(1 + w)\, \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, c) - w\, \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing)$,
so $\gamma = 1 + w$.
:numref:`sec_mdl-score-matching-diffusion-flow` shows that
:eqref:`eq_diffusion-cfg` is :eqref:`eq_mdl-cfg` applied to the noise
prediction and records the caveat that for $\gamma > 1$ the guided field is
in general no longer the score of any noised data distribution; the
practical value of pushing past $\gamma = 1$ is an empirical finding, and
the experiment below measures it.

![Classifier-free guidance as an extrapolation of noise predictions, drawn as arrows from the current state $\mathbf{x}_t$. The network provides the unconditional prediction (grey) and the conditional one (blue); their difference is the class direction, and the guided prediction (orange) moves along it by a factor $\gamma$, past the conditional prediction when $\gamma > 1$.](../img/mdl-diffusion-guidance.svg)
:label:`fig_diffusion-guidance`

The function evaluates the conditional and unconditional predictions in one
batched forward pass, combines them with :eqref:`eq_diffusion-cfg`, and
takes the reverse step.

```{.python .input #image-diffusion-sampling-and-guidance}
%%tab pytorch
@torch.no_grad()
def sample(net, y, gamma, seed=0):
    """Ancestral sampling with classifier-free guidance of scale gamma."""
    torch.manual_seed(seed)
    n, null = len(y), torch.full_like(y, NULL)
    x = torch.randn(n, 1, 28, 28, device=device)
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device)
        eps_c, eps_u = net(torch.cat([x, x]), torch.cat([tt, tt]),
                           torch.cat([y, null])).chunk(2)
        eps = eps_u + gamma * (eps_c - eps_u)
        mean = (x - beta[t] / (1 - alpha_bar[t]).sqrt() * eps) / alpha[t].sqrt()
        x = mean + (beta[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    return x

def show(imgs, rows, cols):
    imgs = imgs.clamp(-1, 1) / 2 + 0.5
    d2l.show_images(imgs.permute(0, 2, 3, 1).repeat(1, 1, 1, 3), rows, cols,
                    scale=0.8)
```

```{.python .input #image-diffusion-sampling-and-guidance}
%%tab jax
@nnx.jit
def guided_step(net, x, t, y, gamma, key):
    n = len(y)
    tt, null = jnp.full((n,), t), jnp.full((n,), NULL)
    eps = net(jnp.concatenate([x, x]), jnp.concatenate([tt, tt]),
              jnp.concatenate([y, null]))
    eps_c, eps_u = eps[:n], eps[n:]  # conditional and unconditional halves
    eps = eps_u + gamma * (eps_c - eps_u)
    mean = (x - beta[t] / jnp.sqrt(1 - alpha_bar[t]) * eps) / jnp.sqrt(alpha[t])
    return mean + jnp.where(t > 0, jnp.sqrt(beta[t]), 0.0) * jax.random.normal(
        key, x.shape)

def sample(net, y, gamma, seed=0):
    """Ancestral sampling with classifier-free guidance of scale gamma."""
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (len(y), 28, 28, 1))
    for t in reversed(range(T)):
        key, subkey = jax.random.split(key)
        x = guided_step(net, x, t, y, gamma, subkey)
    return x

def show(imgs, rows, cols):
    imgs = imgs.clip(-1, 1) / 2 + 0.5
    d2l.show_images(jnp.repeat(imgs, 3, -1), rows, cols, scale=0.8)
```

### Unconditional Samples

With $\gamma = 0$ the label plays no role. Forty-eight samples, each the
result of a thousand reverse steps from Gaussian noise with the averaged
weights, take one to two minutes.

```{.python .input #image-diffusion-unconditional-samples}
%%tab pytorch
samples = sample(ema, torch.full((48,), NULL, device=device), gamma=0.0)
show(samples, 6, 8)
```

```{.python .input #image-diffusion-unconditional-samples}
%%tab jax
samples = sample(ema, jnp.full((48,), NULL), gamma=0.0)
show(samples, 6, 8)
```

Many of the samples are recognizable garments and shoes, generated by a
network of about 340 thousand parameters trained for 1500 updates. Many
others are malformed, with uneven outlines, coarse textures, or shapes that
mix several classes, a consequence of that budget. The CIFAR-10 model of
:citet:`ho2020denoising` has 35.7 million parameters, about a hundred times
as many, and trained for 800 thousand updates.

### Guided Samples
:label:`subsec_diffusion-images-guided`

The guidance experiment requests eight samples of every class at three
guidance scales. The field measures the fidelity and diversity of samples
with feature-space statistics such as the FID and precision--recall of
:numref:`sec_dcgan`, which need a pretrained feature network and thousands
of samples; for $28 \times 28$ images from a ten-minute model, two direct
proxies serve. To measure what guidance does, we train a small
convolutional classifier on the clean training images, which reaches close
to ninety percent test accuracy in seconds, and use it to score the
samples. This classifier only measures the samples; it plays no part in
sampling, unlike the noise-trained classifier of classifier guidance.
Samples are clamped to $[-1, 1]$, the range of the training images, before
they are scored. The three guided runs take several minutes.
Two quantities are reported per scale: the fraction of samples that the
classifier assigns to the requested class, a proxy for class fidelity, and
the mean Euclidean distance between pairs of samples of the same class, a
proxy for diversity.

```{.python .input #image-diffusion-guided-samples-1}
%%tab pytorch
torch.manual_seed(1)
classifier = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
                           nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Flatten(),
                           nn.Linear(64 * 49, 128), nn.ReLU(),
                           nn.Linear(128, 10)).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
for step in range(1000):
    idx = torch.randint(0, len(X), (128,), device=device)
    loss = nn.functional.cross_entropy(classifier(X[idx]), Y[idx])
    optimizer.zero_grad(), loss.backward(), optimizer.step()
with torch.no_grad():
    acc = (classifier(X_test).argmax(1) == Y_test).float().mean()
print(f'classifier test accuracy: {acc:.3f}')
```

```{.python .input #image-diffusion-guided-samples-1}
%%tab jax
class Classifier(nnx.Module):
    def __init__(self, rngs):
        self.c1 = nnx.Conv(1, 32, (3, 3), strides=2, padding='SAME', rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (3, 3), strides=2, padding='SAME', rngs=rngs)
        self.l1 = nnx.Linear(64 * 49, 128, rngs=rngs)
        self.l2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, x):
        h = nnx.relu(self.c2(nnx.relu(self.c1(x)))).reshape(len(x), -1)
        return self.l2(nnx.relu(self.l1(h)))

classifier = Classifier(nnx.Rngs(1))
optimizer = nnx.Optimizer(classifier, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def classifier_step(classifier, optimizer, x, y):
    loss_fn = lambda m: optax.softmax_cross_entropy_with_integer_labels(
        m(x), y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(classifier)
    optimizer.update(classifier, grads)
    return loss

for step in range(1000):
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (128,), 0, len(X))
    classifier_step(classifier, optimizer, X[idx], Y[idx])
acc = (classifier(X_test).argmax(1) == Y_test).mean()
print(f'classifier test accuracy: {acc:.3f}')
```

```{.python .input #image-diffusion-guided-samples-2}
%%tab pytorch
y = torch.arange(10, device=device).repeat(8)  # eight samples per class
guided = {}
print(f'{"gamma":>5} {"class agreement":>16} {"within-class distance":>22}')
for gamma in (1.0, 2.0, 4.0):
    guided[gamma] = sample(ema, y, gamma, seed=2).clamp(-1, 1)
    with torch.no_grad():
        agree = (classifier(guided[gamma]).argmax(1) == y).float().mean()
    flat = guided[gamma].flatten(1)
    dist = torch.stack([torch.pdist(flat[y == c]).mean() for c in range(10)]).mean()
    print(f'{gamma:>5.1f} {agree:>16.3f} {dist:>22.2f}')
```

```{.python .input #image-diffusion-guided-samples-2}
%%tab jax
y = jnp.tile(jnp.arange(10), 8)  # eight samples per class
guided = {}
print(f'{"gamma":>5} {"class agreement":>16} {"within-class distance":>22}')
for gamma in (1.0, 2.0, 4.0):
    guided[gamma] = sample(ema, y, gamma, seed=2).clip(-1, 1)
    agree = (classifier(guided[gamma]).argmax(1) == y).mean()
    flat = guided[gamma].reshape(80, -1)
    dist = jnp.mean(jnp.array([
        jnp.linalg.norm(flat[y == c][:, None] - flat[y == c][None], axis=-1).sum()
        / 56 for c in range(10)]))  # 8 samples: 56 ordered pairs per class
    print(f'{gamma:>5.1f} {agree:>16.3f} {dist:>22.2f}')
```

```{.python .input #image-diffusion-guided-samples-3}
%%tab pytorch
print('first grid: guidance scale 1; second grid: guidance scale 4; '
      'columns are the ten classes')
for gamma in (1.0, 4.0):
    show(guided[gamma][:40], 4, 10)  # the first four samples of each class
```

```{.python .input #image-diffusion-guided-samples-3}
%%tab jax
print('first grid: guidance scale 1; second grid: guidance scale 4; '
      'columns are the ten classes')
for gamma in (1.0, 4.0):
    show(guided[gamma][:40], 4, 10)  # the first four samples of each class
```

Class agreement rises with the guidance scale and within-class distance
falls: stronger guidance produces samples that the classifier assigns to
the requested class more reliably and that resemble one another more. The
grids, which show the first four samples of each class, show the same
effect. At $\gamma = 1$ each column contains varied
instances of its class, some ambiguous; at $\gamma = 4$ the instances are
cleaner and more prototypical, and closer to one another. The cost is
visible too: at $\gamma = 4$ several samples saturate to white, and the
sandal column improves little. The trade-off is the one that
:citet:`Ho.Salimans.2022` report on ImageNet, where guidance improves
fidelity metrics at the cost of diversity, so the guidance scale is a
setting chosen for each application rather than a constant fixed by the
derivation.

### Classifier Guidance
:label:`subsec_diffusion-images-classifier-guidance`

Classifier-free guidance obtains the class direction from the diffusion
network itself. The older route, *classifier guidance*
:cite:`Dhariwal.Nichol.2021`, obtains it from a separate classifier and
asks nothing of the diffusion model beyond an unconditional prediction.
Written for the noise prediction, the Bayes-rule identity
:eqref:`eq_diffusion-bayes-score` becomes

$$
\hat{\boldsymbol{\epsilon}}(\mathbf{x}_t, t, c)
= \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing)
- s\, \sqrt{1 - \bar{\alpha}_t}\; \nabla_{\mathbf{x}_t} \log p_{\boldsymbol{\phi}}(c \mid \mathbf{x}_t, t),
$$
:eqlabel:`eq_diffusion-classifier-guidance`

where $p_{\boldsymbol{\phi}}$ is a classifier of *noisy* images that takes
the step $t$ as an input, and the scale $s$ plays the role of $\gamma$:
$s = 1$ would sample from the Bayes-rule conditional if $p_{\boldsymbol{\phi}}$
were the exact class posterior of the noisy images, and $s > 1$ sharpens it.
The factor $\sqrt{1 - \bar{\alpha}_t}$ converts the score-space gradient
into the noise-prediction convention of :eqref:`eq_diffusion-score-eps`.
The classifier must be trained on noisy images because its gradient is
taken at $\mathbf{x}_t$, an image at noise level $t$; a classifier trained
only on clean images never saw such inputs, so its gradient there does not
estimate $\nabla_{\mathbf{x}_t} \log p_t(c \mid \mathbf{x}_t)$. The cell trains such a
classifier, the architecture of the classifier above with a second input
channel that carries the noise level $\sqrt{1 - \bar{\alpha}_t}$, on noisy
training images at random steps, and reports its accuracy on clean test
images and on test images at the middle of the schedule.

```{.python .input #image-diffusion-classifier-guidance-1}
%%tab pytorch
class NoisyClassifier(nn.Module):
    """A classifier of noisy images; the noise level enters as a second channel."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, 32, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(64 * 49, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x, t):
        level = (1 - alpha_bar[t]).sqrt()[:, None, None, None].expand_as(x)
        return self.net(torch.cat([x, level], 1))

torch.manual_seed(3)
noisy_classifier = NoisyClassifier().to(device)
optimizer = torch.optim.Adam(noisy_classifier.parameters(), lr=1e-3)
for step in range(2000):
    idx = torch.randint(0, len(X), (128,), device=device)
    x0, y_batch = X[idx], Y[idx]
    t = torch.randint(0, T, (128,), device=device)
    a = alpha_bar[t][:, None, None, None]
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)
    loss = nn.functional.cross_entropy(noisy_classifier(x_t, t), y_batch)
    optimizer.zero_grad(), loss.backward(), optimizer.step()
with torch.no_grad():
    t_mid = torch.full((len(X_test),), T // 2, device=device)
    a = alpha_bar[t_mid][:, None, None, None]
    x_mid = a.sqrt() * X_test + (1 - a).sqrt() * torch.randn_like(X_test)
    t_clean = torch.zeros(len(X_test), dtype=torch.long, device=device)
    acc_clean = (noisy_classifier(X_test, t_clean).argmax(1) == Y_test).float().mean()
    acc_mid = (noisy_classifier(x_mid, t_mid).argmax(1) == Y_test).float().mean()
print(f'noisy classifier test accuracy: {acc_clean:.3f} at t = 0, '
      f'{acc_mid:.3f} at t = {T // 2}')
```

```{.python .input #image-diffusion-classifier-guidance-1}
%%tab jax
class NoisyClassifier(nnx.Module):
    """A classifier of noisy images; the noise level enters as a second channel."""
    def __init__(self, rngs):
        self.c1 = nnx.Conv(2, 32, (3, 3), strides=2, padding='SAME', rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (3, 3), strides=2, padding='SAME', rngs=rngs)
        self.l1 = nnx.Linear(64 * 49, 128, rngs=rngs)
        self.l2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, x, t):
        level = jnp.sqrt(1 - alpha_bar[t])[:, None, None, None]
        h = jnp.concatenate([x, jnp.broadcast_to(level, x.shape)], -1)
        h = nnx.relu(self.c2(nnx.relu(self.c1(h)))).reshape(len(x), -1)
        return self.l2(nnx.relu(self.l1(h)))

noisy_classifier = NoisyClassifier(nnx.Rngs(3))
optimizer = nnx.Optimizer(noisy_classifier, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def noisy_classifier_step(classifier, optimizer, x0, y, t, eps):
    def loss_fn(model):
        a = alpha_bar[t][:, None, None, None]
        logits = model(jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps, t)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(classifier)
    optimizer.update(classifier, grads)
    return loss

for step in range(2000):
    key, k1, k2, k3 = jax.random.split(key, 4)
    idx = jax.random.randint(k1, (128,), 0, len(X))
    noisy_classifier_step(noisy_classifier, optimizer, X[idx], Y[idx],
                          jax.random.randint(k2, (128,), 0, T),
                          jax.random.normal(k3, (128, 28, 28, 1)))
key, subkey = jax.random.split(key)
t_mid = jnp.full((len(X_test),), T // 2)
a = alpha_bar[t_mid][:, None, None, None]
x_mid = jnp.sqrt(a) * X_test + jnp.sqrt(1 - a) * jax.random.normal(subkey, X_test.shape)
t_clean = jnp.zeros(len(X_test), jnp.int32)
acc_clean = (noisy_classifier(X_test, t_clean).argmax(1) == Y_test).mean()
acc_mid = (noisy_classifier(x_mid, t_mid).argmax(1) == Y_test).mean()
print(f'noisy classifier test accuracy: {acc_clean:.3f} at t = 0, '
      f'{acc_mid:.3f} at t = {T // 2}')
```

The sampler is the ancestral chain of `sample` with
:eqref:`eq_diffusion-classifier-guidance` in place of the classifier-free
rule: at each step it takes the unconditional prediction of the same U-Net,
computes the gradient of the classifier's log-probability of the requested
class with respect to the current noisy image by automatic differentiation,
and combines the two. We run it for the same eighty requests at three
scales and score the samples with the clean classifier as before, so that
both rules are scored the same way.

```{.python .input #image-diffusion-classifier-guidance-2}
%%tab pytorch
def classifier_gradient(classifier, x, t, y):
    """Gradient of log p(y | x_t) with respect to x_t for every image in the batch."""
    x = x.detach().requires_grad_(True)
    log_prob = torch.log_softmax(classifier(x, t), 1).gather(1, y[:, None]).sum()
    return torch.autograd.grad(log_prob, x)[0]

def sample_classifier_guided(net, classifier, y, scale, seed=0):
    """Ancestral sampling with classifier guidance of scale `scale`."""
    torch.manual_seed(seed)
    n, null = len(y), torch.full_like(y, NULL)
    x = torch.randn(n, 1, 28, 28, device=device)
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device)
        with torch.no_grad():
            eps = net(x, tt, null)  # the unconditional prediction
        eps = eps - scale * (1 - alpha_bar[t]).sqrt() * classifier_gradient(
            classifier, x, tt, y)
        with torch.no_grad():
            mean = (x - beta[t] / (1 - alpha_bar[t]).sqrt() * eps) / alpha[t].sqrt()
            x = mean + (beta[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    return x

classifier_guided = {}
print(f'{"scale":>5} {"class agreement":>16} {"within-class distance":>22}')
for scale in (1.0, 4.0, 10.0):
    classifier_guided[scale] = sample_classifier_guided(
        ema, noisy_classifier, y, scale, seed=2).clamp(-1, 1)
    with torch.no_grad():
        agree = (classifier(classifier_guided[scale]).argmax(1) == y).float().mean()
    flat = classifier_guided[scale].flatten(1)
    dist = torch.stack([torch.pdist(flat[y == c]).mean() for c in range(10)]).mean()
    print(f'{scale:>5.1f} {agree:>16.3f} {dist:>22.2f}')
print('first grid: scale 1; second grid: scale 10; columns are the ten classes')
for scale in (1.0, 10.0):
    show(classifier_guided[scale][:40], 4, 10)
```

```{.python .input #image-diffusion-classifier-guidance-2}
%%tab jax
@nnx.jit
def classifier_guided_step(net, classifier, x, t, y, scale, key):
    n = len(y)
    tt, null = jnp.full((n,), t), jnp.full((n,), NULL)
    eps = net(x, tt, null)  # the unconditional prediction
    log_prob = lambda x: jnp.take_along_axis(  # sum of log p(y_i | x_i), one term per image
        jax.nn.log_softmax(classifier(x, tt), 1), y[:, None], 1).sum()
    eps = eps - scale * jnp.sqrt(1 - alpha_bar[t]) * jax.grad(log_prob)(x)
    mean = (x - beta[t] / jnp.sqrt(1 - alpha_bar[t]) * eps) / jnp.sqrt(alpha[t])
    return mean + jnp.where(t > 0, jnp.sqrt(beta[t]), 0.0) * jax.random.normal(
        key, x.shape)

def sample_classifier_guided(net, classifier, y, scale, seed=0):
    """Ancestral sampling with classifier guidance of scale `scale`."""
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (len(y), 28, 28, 1))
    for t in reversed(range(T)):
        key, subkey = jax.random.split(key)
        x = classifier_guided_step(net, classifier, x, t, y, scale, subkey)
    return x

classifier_guided = {}
print(f'{"scale":>5} {"class agreement":>16} {"within-class distance":>22}')
for scale in (1.0, 4.0, 10.0):
    classifier_guided[scale] = sample_classifier_guided(
        ema, noisy_classifier, y, scale, seed=2).clip(-1, 1)
    agree = (classifier(classifier_guided[scale]).argmax(1) == y).mean()
    flat = classifier_guided[scale].reshape(80, -1)
    dist = jnp.mean(jnp.array([
        jnp.linalg.norm(flat[y == c][:, None] - flat[y == c][None], axis=-1).sum()
        / 56 for c in range(10)]))  # 8 samples: 56 ordered pairs per class
    print(f'{scale:>5.1f} {agree:>16.3f} {dist:>22.2f}')
print('first grid: scale 1; second grid: scale 10; columns are the ten classes')
for scale in (1.0, 10.0):
    show(classifier_guided[scale][:40], 4, 10)
```

The noisy classifier is a few points less accurate than the clean one on
clean images and still right more often than not halfway through the
schedule. With the scale at one, which targets the Bayes-rule conditional,
classifier guidance steers only weakly: about half of the samples land in
the requested class, fewer than under classifier-free guidance at
$\gamma = 1$. :citet:`Dhariwal.Nichol.2021` report a similar weakness on
ImageNet: at scale one their samples did not match the requested classes on
inspection, although their guiding classifier gave those classes about fifty
percent probability; larger scales remedied this. Raising the scale raises agreement and lowers within-class
distance, as before; at a scale of ten the agreement comes within about
$0.1$ of that of classifier-free guidance at $\gamma = 4$, while the
within-class distance stays larger. The grids temper this comparison: at
scale ten, more samples are malformed or faint than under classifier-free
guidance at $\gamma = 4$.
Classifier guidance ascends the log-probability of a classifier with the
same architecture and training data as the one that scores the samples, so
it can raise agreement without producing a recognizable garment, and pixel
distance grows with such artifacts as well as with diversity;
:citet:`Ho.Salimans.2022` raise the same concern about classifier-based
metrics. The other cost is a second network, trained on noisy images, that
steers toward whatever it has learned, including its confusions between
similar garments. :citet:`Ho.Salimans.2022` introduced the classifier-free
rule to remove that network, and with it the need for a classifier of noisy
data.

## Outlook

The model trained here has the structure of the systems that generate
images from text, and the differences are matters of scale and of a few
later ideas, two of which the next sections develop. Sampling with a
thousand network evaluations is the main cost, and
:numref:`sec_diffusion-ddim` shows that a trained noise predictor also
supports a deterministic sampler, which for a well-trained network needs
only a few dozen evaluations :cite:`Song.Meng.Ermon.2020`.
:numref:`sec_diffusion-flow-matching` replaces the noise predictor by a
velocity field regressed on straight noise-to-data segments, the formulation
that
:citet:`Esser.Kulal.Blattmann.ea.2024` scale to large text-to-image models,
and shows how the two views are related. :numref:`sec_diffusion-discrete`
then extends the construction to categorical data such as text.

Four further lines of work act on different parts of the model.

On the sampler, improved schedules and learned reverse variances
:cite:`Nichol.Dhariwal.2021` and a careful redesign of the noise
parameterization and sampler :cite:`Karras.Aittala.Aila.ea.2022` reduce
the step count further, and distillation trains a student to reproduce the
sampler's output in one to four evaluations
:cite:`Salimans.Ho.2022,Song.Dhariwal.Chen.ea.2023`; a later line of
distillation adds an adversarial loss to the student
:cite:`Sauer.Lorenz.Blattmann.ea.2023`, which is where adversarial
training re-enters (:numref:`sec_gan_beyond`).

On the data space, latent
diffusion :cite:`Rombach.Blattmann.Lorenz.ea.2022` runs the diffusion in
the latent space of an autoencoder, so that its U-Net, itself hundreds of
millions of parameters or more, operates on a compressed image, and the
decoder maps each generated latent back to an image in a single pass.
Sampling in the smaller space is much cheaper, but the images are limited
by what the autoencoder preserves: :citet:`Rombach.Blattmann.Lorenz.ea.2022`
find that too strong a compression limits the achievable quality and too
weak a compression slows training. The model conditions on text through
cross-attention layers inside the U-Net that attend to the token embeddings
of a text encoder, a richer injection than the single vector added to the
time embedding here.

On the backbone, transformers replace the U-Net at the largest scales
:cite:`Peebles.Xie.2023`.

On the objective and the theory, the variational bound of
:numref:`sec_diffusion-ddpm` also makes diffusion models competitive
likelihood estimators when trained on it rather than on the simple loss
:cite:`Kingma.Salimans.Poole.ea.2021`, and the continuous-time view of
:numref:`sec_mdl-score-matching-diffusion-flow` places the ladder, the
chain, and the samplers of this chapter in one framework: the ladder and
the chain discretize the variance-exploding and the variance-preserving
stochastic differential equations, the ancestral sampler discretizes the
reverse-time equation, and the deterministic sampler of
:numref:`sec_diffusion-ddim` its probability-flow equation
:cite:`song2021score`; :citet:`Lai.Song.Kim.ea.2025` develop that view in
full.

## Summary

A class-conditional diffusion model for images is the model of
:numref:`sec_diffusion-ddpm` with a U-Net as the noise predictor, a class
embedding added to the time embedding, and a null label that makes the same
network unconditional. Trained by the simple loss with label dropout, and
sampled with an exponential moving average of the weights, a network of
about 340 thousand parameters produces Fashion-MNIST samples, many of them
recognizable, after ten minutes of laptop training.

Classifier-free guidance combines the two predictions the network provides:
the guided noise prediction extrapolates from the unconditional prediction
through the conditional one by a scale $\gamma$. Increasing $\gamma$ raised
the fraction of samples that a classifier assigned to the requested class
and lowered the pixel distance between samples of the same class. These
are the proxies used here for fidelity and diversity, and the scale sets the
trade-off between them. Classifier guidance, which draws the class direction
from a separate classifier trained on noisy images, raised agreement and
lowered within-class distance in the same way on the same requests, at the
cost of that second network and with more malformed samples at the largest
scale. Fast samplers, latent-space diffusion, larger
backbones, and the continuous-time formulation build on the model of this
chapter rather than replacing it.

## Exercises

1. **The null label.** The network learns its unconditional prediction only
   from the training examples whose label was replaced by the null label
   $\varnothing$.
    1. Suppose that label dropout were omitted, so that the null label never
       occurs in training. Which term of :eqref:`eq_diffusion-cfg` would then
       come from an untrained part of the network? What would `sample`
       compute at $\gamma = 0$, and what at $\gamma = 1$?
    1. Why would a dropout probability close to one also harm guided
       sampling, and which of the two predictions would suffer?
    1. Count the forward and backward passes through each network per
       reverse step of `sample` and of `sample_classifier_guided`, for a
       batch of $n$ requests. For which values of $\gamma$ could `sample`
       drop half of its batched forward pass?
   :exlabel:`ex_image-diffusion-null-label`
1. **Equivalent guidance rules.** Suppose that the conditional and the
   unconditional prediction at step $t$ are exact minimizers of the simple
   loss of :numref:`sec_diffusion-ddpm`, so that by
   :eqref:`eq_diffusion-score-eps` they are

   $$\boldsymbol{\epsilon}(\mathbf{x}_t, t, c) = -\sqrt{1 - \bar{\alpha}_t}\, \nabla \log p_t(\mathbf{x}_t \mid c), \qquad \boldsymbol{\epsilon}(\mathbf{x}_t, t, \varnothing) = -\sqrt{1 - \bar{\alpha}_t}\, \nabla \log p_t(\mathbf{x}_t),$$

   and that the classifier is the exact class posterior
   $p_t(c \mid \mathbf{x}_t)$ of the noisy images.
    1. Using :eqref:`eq_diffusion-bayes-score` at step $t$, show that
       classifier guidance :eqref:`eq_diffusion-classifier-guidance` at scale
       $s$ and classifier-free guidance :eqref:`eq_diffusion-cfg` at
       $\gamma = s$ produce the same guided prediction.
    1. A network may predict the clean image instead of the noise,
       $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}) / \sqrt{\bar{\alpha}_t}$,
       which solves :eqref:`eq_diffusion-marginal` for $\mathbf{x}_0$. Show
       that applying the rule :eqref:`eq_diffusion-cfg` to the conditional
       and unconditional clean-image predictions gives the clean-image
       prediction of the guided noise prediction. Which property of the
       weights $1 - \gamma$ and $\gamma$ does the argument use? For which
       kind of prediction target would guiding the target differ from
       guiding the noise?
   :exlabel:`ex_image-diffusion-equivalent-guidance`
1. [code] **Reference values for the proxies.** The class agreement and the
   within-class distance printed in
   :numref:`subsec_diffusion-images-guided` and
   :numref:`subsec_diffusion-images-classifier-guidance` can be read only
   against the values that real images attain.
    1. Draw eight images of each class at random from `X_test`, compute both
       quantities for them as the cells of
       :numref:`subsec_diffusion-images-guided` compute them for the
       samples, and repeat for $200$ independent draws. Report the mean and
       the standard deviation of each quantity.
    1. Compare the means with the printed rows of both guidance rules. Which
       runs produce samples less varied than the test images by this
       measure, and which exceed the test images?
    1. Pixel distance between samples of one class grows with artifacts as
       well as with variety. To separate the two, compute the Euclidean
       distance from every image in `classifier_guided[1.0]`, in
       `guided[1.0]`, and in one draw of test images from (a) to the nearest
       training image in `X` of its class, which for a sample is the
       requested class. Which of the two explanations of the excess
       within-class distance of classifier guidance at scale $1$ do the
       three sets of distances support?
    1. Use the standard deviations as estimates of the sampling noise of a
       run of $80$ samples. What is the standard deviation of the
       difference between the values of two independent runs? Which
       differences between the printed rows exceed twice that standard
       deviation?
   :exlabel:`ex_image-diffusion-proxy-reference`
1. [code] **Guidance scales.** Extend the experiment of
   :numref:`subsec_diffusion-images-guided` to
   $\gamma \in \{0, 0.5, 1, 2, 4, 8, 16\}$, with the same requests `y` and
   the same seed. Define the *saturation* of a set of images as the fraction
   of its pixels that equal $1$, the white end of the range, after clamping.
    1. Plot class agreement, within-class distance, and saturation against
       $\gamma$. Mark the values for test images, computing the first two as
       in exercise :exref:`ex_image-diffusion-proxy-reference`.
    1. Display the samples at $\gamma = 8$ and $\gamma = 16$. Which of the
       three quantities registers their defects? Which rates them as
       improvements over $\gamma = 4$?
   :exlabel:`ex_image-diffusion-guidance-scales`
1. [code] **A clean classifier as guide.** Classifier guidance in
   :numref:`subsec_diffusion-images-classifier-guidance` steers with
   `noisy_classifier`, trained on noisy images, and scores the samples with
   `classifier`, trained on clean ones.
    1. Report the accuracy of `classifier` and of `noisy_classifier` on the
       clean test images and on test images noised by
       :eqref:`eq_diffusion-marginal` at the steps
       $t \in \{100, 250, 500, 750\}$. How does the accuracy of each
       classifier fall with $t$?
    1. Wrap `classifier` in a module that accepts and ignores the step
       argument, and pass it to `sample_classifier_guided` at the scales $1$
       and $10$, with the requests `y` and seed $2$. Score these samples and
       those in `classifier_guided` with two judges, `classifier` and
       `noisy_classifier` at $t = 0$, and arrange the class agreements in a
       table of guide against judge. Display the samples of the clean guide
       at scale $10$.
    1. Which entries of the table measure class fidelity, and which measure
       how far the sampler raised the probability that its own guide assigns
       to the requested class? What does the comparison imply for scoring a
       guided sampler with a classifier?
   :exlabel:`ex_image-diffusion-clean-classifier-guide`
1. [code] **Inpainting.** A trained diffusion model can fill in missing pixels
   without retraining :cite:`song2021score`. Take a test image, treat its
   lower half as unknown, and run the ancestral chain of `sample` from
   Gaussian noise. After each reverse step, replace the upper half of the
   state by the upper half of a fresh noisy copy of the original at the step
   that the state has reached. After the update at loop index `t`, that
   step's marginal :eqref:`eq_diffusion-marginal` uses `alpha_bar[t - 1]`,
   and after the last update the upper half is the clean original.
    1. Implement the procedure with the null label at $\gamma = 0$, and show
       the masked input, the completion, and the original for eight test
       images.
    1. Why does the replacement use a noisy copy of the original rather than
       its clean pixels at every step?
    1. Complete one masked image five times with different seeds. Which
       parts of the lower half vary among the completions, and which does
       the upper half determine?
   :exlabel:`ex_image-diffusion-inpainting`
1. **Resolution and the schedule.** ● The noise levels of the schedule are
   calibrated for $28 \times 28$ images in $[-1, 1]$. Write
   $\mathrm{SNR}_t = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$, the ratio of the
   signal variance to the noise variance in $\mathbf{x}_t$ of
   :eqref:`eq_diffusion-marginal` for data with unit variance per pixel.
    1. Show that multiplying the data by a constant $\kappa$ multiplies the
       ratio of signal to noise variance in $\mathbf{x}_t$ by $\kappa^2$ at
       every step.
    1. Upsample each image to $28 r \times 28 r$ pixels by repeating every
       pixel in an $r \times r$ block, and noise the large image with the
       same schedule. Show that averaging the noisy image over the
       $r \times r$ blocks gives the original image noised with the variance
       $(1 - \bar{\alpha}_t) / r^2$ in place of $1 - \bar{\alpha}_t$, so that
       the averaged image has the ratio $r^2\, \mathrm{SNR}_t$.
    1. For $r = 32$, an $896 \times 896$ image, compute $r^2\, \mathrm{SNR}_t$
       at the last step $t = T$, and find the step of the $28 \times 28$
       chain with the same ratio. What does the result imply for an
       $896 \times 896$ model trained with this schedule, whose sampler
       starts from pure Gaussian noise?
    1. Which shift of $\log \mathrm{SNR}_t$, and which rescaling of the data,
       let the averaged image of the large model pass through the same
       ratios as the $28 \times 28$ model?
       :citet:`Hoogeboom.Heek.Salimans.2023` shift the schedule in this way.
   :exlabel:`ex_image-diffusion-resolution-schedule`
1. [code] **Weight averaging.** ● After the $S = 1500$ updates, the
   averaged parameters `ema` are a weighted sum of the initial parameters
   and of the parameters after each update $s = 0, \ldots, S - 1$, which the
   loop mixes in with the decay $d_s = \min(0.999, (1 + s) / (10 + s))$.
   The *weight* of a set of parameters is its coefficient in this sum.
    1. Suppose that the decay were $0.999$ from the first update. Which
       weight would the average give to the random initialization?
    1. Show that the cap $0.999$ is never reached in this run, that the
       initialization receives the weight $9!\, S! / (S + 9)!$, and that the
       parameters after update $s$ receive the weight

       $$w_s = \frac{9 \prod_{k=2}^{9} (s + k)}{\prod_{k=1}^{9} (S + k)}.$$

    1. Approximate $w_s$ for large $s$ and $S$, and compute the fraction of
       the total weight that falls on the last $300$ updates.
       :citet:`Karras.Aittala.Lehtinen.ea.2024` propose averages whose
       weights grow like a power of the update index. Which power does the
       warm-up implement?
    1. Repeat the training while keeping a second average with the constant
       decay $0.999$, and sample forty-eight unconditional images with the
       same seed from `net`, from `ema`, and from the second average.
       Relate the differences among the three grids to your answers to (a)
       to (c).
   :exlabel:`ex_image-diffusion-weight-averaging`
1. [code] **Thresholding the predicted image.** ● The guided noise
   prediction $\tilde{\boldsymbol{\epsilon}}$ implies the clean image
   $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\, \tilde{\boldsymbol{\epsilon}}) / \sqrt{\bar{\alpha}_t}$,
   which solves :eqref:`eq_diffusion-marginal` for $\mathbf{x}_0$. By
   :eqref:`eq_diffusion-posterior-mean-eps`, the update `mean` of each
   reverse step of `sample` is the posterior mean
   :eqref:`eq_diffusion-posterior` with $\hat{\mathbf{x}}_0$ in place of
   $\mathbf{x}_0$.
   :citet:`saharia2022photorealistic` trace the saturated samples of large
   guidance scales to predictions $\hat{\mathbf{x}}_0$ outside the range of
   the training data.
    1. For $\gamma \in \{1, 4, 16\}$, record at every step of `sample` the
       fraction of entries of $\hat{\mathbf{x}}_0$ outside $[-1, 1]$, and
       plot it against $t$. In which part of the chain does the prediction
       leave the range, and how does $\gamma$ change this?
    1. Rewrite the step so that it computes $\hat{\mathbf{x}}_0$ and forms
       `mean` from :eqref:`eq_diffusion-posterior`, where
       $\bar{\alpha}_{t-1}$ is `alpha_bar[t - 1]` at loop index `t` and $1$
       at `t = 0`. Check that the new step reproduces `sample` up to
       rounding. Then clip $\hat{\mathbf{x}}_0$ to $[-1, 1]$ before forming
       `mean`, the *static thresholding* of
       :citet:`saharia2022photorealistic`.
    1. Their *dynamic thresholding* sets, for each image, the threshold $m$
       to the larger of $1$ and the $99.5$th percentile of
       $|\hat{\mathbf{x}}_0|$, clips $\hat{\mathbf{x}}_0$ to $[-m, m]$, and
       divides it by $m$. Implement it, and compare the class agreement, the
       within-class distance, the saturation of exercise
       :exref:`ex_image-diffusion-guidance-scales`, and the samples of the
       three samplers at $\gamma \in \{4, 8, 16\}$.
   :exlabel:`ex_image-diffusion-thresholding`
1. [code] **Guidance in a limited interval.** ●
   :citet:`Kynkaanniemi.Aittala.Karras.ea.2024` observe that guidance acts
   differently at high, middle, and low noise levels, and they apply it only
   within an interval of noise levels, with $\gamma = 1$ elsewhere. They
   measure the noise level of a step by
   $\sigma^{\textrm{eq}}_t = \sqrt{(1 - \bar{\alpha}_t) / \bar{\alpha}_t}$,
   the noise level of the rescaled state
   $\mathbf{x}_t / \sqrt{\bar{\alpha}_t}$ (the equivalent noise level of
   exercise :exref:`ex_ddpm-equivalent-noise-levels` of
   :numref:`sec_diffusion-ddpm`), and found the interval
   $0.28 < \sigma^{\textrm{eq}}_t \leq 2.9$ best by the FID for one of the
   ImageNet models of :citet:`Karras.Aittala.Lehtinen.ea.2024`.
    1. Find the loop indices `t` of `sample` whose noise level, computed
       from `alpha_bar[t]`, lies in the high-noise interval
       $\sigma^{\textrm{eq}}_t > 2.9$, in the middle interval
       $0.28 < \sigma^{\textrm{eq}}_t \leq 2.9$, and in the low-noise
       interval $\sigma^{\textrm{eq}}_t \leq 0.28$.
    1. Modify `sample` to apply $\gamma$ only at the steps of a given
       interval. With $\gamma = 4$, sample the requests `y` with seed $2$
       for each of the three intervals. Compare the class agreement, the
       within-class distance, and the saturation of exercise
       :exref:`ex_image-diffusion-guidance-scales` with those of guidance at
       every step and of $\gamma = 1$.
    1. Which interval accounts for most of the gain in class agreement, and
       which for most of the loss of within-class distance? Explain the
       answer by what the state of the chain has already settled at each
       noise level, in the sense of the commitment time of exercise
       :exref:`ex_ddpm-commitment-time` of :numref:`sec_diffusion-ddpm`.
   :exlabel:`ex_image-diffusion-guidance-interval`
1. [extended] **Autoguidance.** ●●
   :citet:`Karras.Aittala.Kynkaanniemi.ea.2024` guide a conditional model with a weaker version of itself instead of its
   unconditional prediction,

   $$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{\textrm{weak}}(\mathbf{x}_t, t, c) + \gamma \big( \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_{\textrm{weak}}(\mathbf{x}_t, t, c) \big),$$

   where the weak model is smaller or trained for less time on the same data
   and labels. They report that this improves sample quality without the
   loss of variation that classifier-free guidance causes.
    1. Treat each network as the exact noise predictor of its own densities
       $p_{\boldsymbol{\theta}, t}(\mathbf{x} \mid c)$ and
       $p_{\textrm{weak}, t}(\mathbf{x} \mid c)$, as in exercise
       :exref:`ex_image-diffusion-equivalent-guidance`. Show that the guided
       prediction corresponds to the score of a density proportional to
       $p_{\boldsymbol{\theta}, t}(\mathbf{x} \mid c)\, \big( p_{\boldsymbol{\theta}, t}(\mathbf{x} \mid c) / p_{\textrm{weak}, t}(\mathbf{x} \mid c) \big)^{\gamma - 1}$.
       Classifier-free guidance corresponds in the same way to a density
       proportional to
       $p_{\boldsymbol{\theta}, t}(\mathbf{x} \mid c)\, \big( p_{\boldsymbol{\theta}, t}(\mathbf{x} \mid c) / p_{\boldsymbol{\theta}, t}(\mathbf{x}) \big)^{\gamma - 1}$,
       whose ratio is proportional to the class posterior
       $p_{\boldsymbol{\theta}, t}(c \mid \mathbf{x})$
       (:numref:`sec_mdl-score-matching-diffusion-flow`). Which images does
       each of the two tilts favor? Use the answer to explain why
       classifier-free guidance reduces the variation within a class, while
       autoguidance, according to
       :citet:`Karras.Aittala.Kynkaanniemi.ea.2024`, does not.
    1. Build two weak models: the moving average of the weights as it
       stands after $375$ of the $1500$ updates of the training loop, and a
       model built like `ConditionalUNet` but with `ch=16` passed to
       `d2l.UNet`, trained with label dropout for $375$ updates. A copy of
       the training loop rebinds `y` to its last batch of labels, so define
       the requests `y` of :numref:`subsec_diffusion-images-guided` again
       before sampling. For these requests and seed $2$, sample with
       autoguidance at $\gamma \in \{1.5, 2, 3, 4\}$ for each weak model,
       and with classifier-free guidance at
       $\gamma \in \{1, 1.5, 2, 3, 4\}$. Plot the within-class distance
       against the class agreement for the three samplers, and report the
       saturation of exercise :exref:`ex_image-diffusion-guidance-scales`.
    1. Which sampler reaches a given class agreement with the larger
       within-class distance, and which weak model guides better? Relate
       the answer to the requirement of
       :citet:`Karras.Aittala.Kynkaanniemi.ea.2024` that the weak model make
       the same kinds of errors as the main model.
    1. With the null label in both models, autoguidance applies to
       unconditional sampling as well. Sample forty-eight images with it at
       $\gamma \in \{2, 4\}$ for each weak model, and compare them with the
       samples of `sample` at $\gamma = 0$ with the same seed.
   :exlabel:`ex_image-diffusion-autoguidance`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17.8]{.kicker}

Diffusion models for images<br>
**a U-Net noise predictor · class conditioning with a null label · classifier-free guidance**
:::
:::

::: {.slide title="The Same Model, an Image-Shaped Network"}
- Fashion-MNIST in $[-1, 1]$; $T = 1000$, $\beta_t$ linear from $10^{-4}$ to $0.02$;
- the noise-conditional U-Net of the earlier sections predicts the noise,
  with the step index embedded;
- a class embedding is added to the time embedding, plus a **null label**
  that carries no class;
- label dropout with probability $0.1$: one network, two predictors.

. . .

Sampling uses an exponential moving average of the weights.
:::

::: {.slide title="Unconditional Samples after Ten Minutes of Training"}
@!image-diffusion-unconditional-samples

Many recognizable garments from about 340 thousand parameters; detail
limited by the budget.
:::

::: {.slide title="Classifier-Free Guidance"}
![](../img/mdl-diffusion-guidance.svg){width=55%}

$$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing)
+ \gamma\, \big( \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing) \big)$$

Bayes' rule twice: the class gradient is proportional to the difference of
the two predictions the network already provides.
:::

::: {.slide title="Classifier Guidance on the Same Requests"}
$$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing)
- s\, \sqrt{1 - \bar{\alpha}_t}\; \nabla_{\mathbf{x}_t} \log p_{\boldsymbol{\phi}}(c \mid \mathbf{x}_t, t)$$

- the class direction comes from a classifier trained on noisy images;
- agreement rises and within-class distance falls with $s$, with more malformed samples at large $s$;
- the cost: a second network, trained on noisy data, with its own confusions.
:::

::: {.slide title="Agreement Rises, Within-Class Distance Falls"}
@image-diffusion-guided-samples-2

Measured with a classifier trained on clean images: agreement with the
requested class increases with $\gamma$, within-class distance decreases.
:::

::: {.slide title="The Same Seed at Two Guidance Scales"}
@!image-diffusion-guided-samples-3

Columns are classes. At $\gamma = 4$ the instances are cleaner, more
prototypical, and more alike; some saturate to white.
:::

::: {.slide title="Recap"}
- Images change the network, not the model: a U-Net predicts the noise.
- A null label makes one network conditional and unconditional at once.
- Classifier-free guidance extrapolates between the two; $\gamma$ trades
  diversity for class fidelity. Classifier guidance does the same with a
  noisy-image classifier's gradient.
- Beyond: fast samplers, distillation, latent diffusion, transformer
  backbones, and the continuous-time view of the appendix.
:::
