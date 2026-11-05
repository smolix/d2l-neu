# Generative Adversarial Networks
:label:`sec_basic_gan`

The language models trained earlier in this book assign a probability to every token, and training raises the probability of the observed text. Now picture a network that turns a handful of random numbers into an image in a single forward pass. Nothing in it computes the probability of the image it produces, so there is no likelihood to raise. How can such a network be trained?

We have met the way out before. To correct for covariate shift in :numref:`subsec_covariate-shift-correction`, we trained a classifier to tell training inputs from test inputs and read the ratio of their two densities off its logit, although neither density was ever written down. A *generative adversarial network* (GAN) :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` trains the same kind of classifier on real and generated samples, and then changes the generator until the classifier fails. Its authors compared the two networks to a counterfeiter and the police: each improves by trying to beat the other, and the game ends when the police can no longer tell forged notes from real ones.

This section asks what that game measures and whether it gives the generator a usable signal. We solve the classifier's problem exactly, identify the quantity the generator then minimizes, compare two ways of writing the generator's loss, and check each prediction on a two-dimensional Gaussian whose densities we know.

```{.python .input #gan-generative-adversarial-networks}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #gan-generative-adversarial-networks}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import math
import numpy as np
import optax
```

## From Likelihood to Comparison

Maximum likelihood needs $\log p_\theta(x)$, the model's log-density at each data point (:numref:`subsec_mdl-nll-crossentropy`). The generators of this chapter are *implicit*: they draw a latent variable $z \sim \mathcal{N}(0, I_{d_z})$ and output $x' = G(z)$, where $G$ is a neural network. The density of $x'$ is unavailable. When the latent dimension $d_z$ is smaller than the data dimension, the samples lie on a lower-dimensional surface and have no density at all; when the dimensions match, evaluating it would require an invertible $G$ with a tractable Jacobian.

One response is to restrict $G$ until its likelihood becomes tractable. That helps only if likelihood measures what we want from a sampler. Consider a model that mixes the data density with noise,

$$
\tilde p \;=\; 0.01\, p_{\textrm{data}} + 0.99\, p_{\textrm{noise}}.
$$
:eqlabel:`eq_gan_noise_mixture`

For every $x$ we have $\tilde p(x) \geq 0.01\, p_{\textrm{data}}(x)$, so $\log \tilde p(x) \geq \log p_{\textrm{data}}(x) - \log 100$. The expected log-likelihood of $\tilde p$ is therefore within $\log 100 \approx 4.6$ nats of the best achievable by any model. For images or audio, whose log-likelihoods run to thousands of nats, this shortfall is small. Yet 99 percent of the mixture's samples are noise. A model can thus have nearly optimal likelihood and be useless as a sampler :cite:`Theis.Oord.Bethge.2016`. Judging a sampler requires looking at its samples.

### Two Networks, One Classification Problem

Statistics has a direct way to compare samples. Given real and generated examples, a *two-sample test* asks whether the two sets came from the same distribution, and one way to run it is to train a classifier to tell them apart. A GAN turns this test into a training signal. Draw a label $y \sim \textrm{Bernoulli}(\tfrac12)$. If $y = 1$, draw $x$ from the data density $p$; if $y = 0$, draw $x$ from the generator density $q$. The marginal density of $x$ is the balanced mixture $m = (p + q)/2$, which describes a sample whose source is unknown.

The classifier is a function $D: \mathcal{X} \to \mathbb{R}$ that outputs a *realness logit*: with $\sigma(t) = 1/(1 + e^{-t})$, the value $\sigma(D(x))$ estimates $P(y = 1 \mid x)$, and a larger $D(x)$ means stronger evidence that $x$ came from the data. :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` called $D$ the *discriminator*. We call it the *critic*, the name used for its successors later in the chapter, which score samples without classifying them. Two identities of the sigmoid recur below: $\sigma(-t) = 1 - \sigma(t)$ and $\tfrac{d}{dt} \log \sigma(t) = \sigma(-t)$.

Training alternates between the two networks, as :numref:`fig_gan` shows. The critic is fitted to separate a batch of data from a batch of generated samples. The generator then changes so that its samples become harder to separate, receiving its gradient through the critic. When the generator matches the data, no critic can separate the two batches.

![Adversarial training. A latent draw $z$ passes through the generator $G$ to produce a batch of generated samples. The critic $D$ assigns realness logits to this batch and to a batch of real data. The critic raises its scores on data and lowers them on generated samples; the generator receives gradients through the critic and makes the two batches harder to distinguish.](../img/mdl-gan-architecture.svg)
:label:`fig_gan`

### Notation

The chapter's symbols are collected in :numref:`tab_gan_notation` with the section where each first appears. Each has one meaning throughout the chapter; letters defined inside a single exercise are local to it. We follow the letters of the literature where they do not clash; the last column says where we depart from them.

:Notation of this chapter.
:label:`tab_gan_notation`

| symbol | meaning | first used | papers write |
|:--|:--|:--|:--|
| $p$, $q$, $m = \tfrac12(p + q)$ | data, generator and mixture densities | :numref:`sec_basic_gan` | the same |
| $x \sim p$, $x' \sim q$ | a real and a generated sample | :numref:`sec_basic_gan` | often $y$ or $\hat x$ for the generated sample |
| $y$ | the source label of a sample, $y = 1$ for real and $y = 0$ for generated | :numref:`sec_basic_gan` | the same |
| $z$, $G$, $d_z$ | the latent variable, the generator $x' = G(z)$, the latent dimension | :numref:`sec_basic_gan` | $k$ for the latent dimension, a kernel here |
| $D$, $\sigma(D)$ | the critic's realness logit and its probability | :numref:`sec_basic_gan` | the discriminator; :citet:`Huang.Gokaslan.Kuleshov.ea.2024` use a fakeness logit, our $-D$ |
| $V$, $V_\ell$ | the value function of the log-loss game, and of the game with payoff $\ell$ | :numref:`sec_basic_gan` | the same |
| $\rho = p/q$, $\lambda = \log \rho$ | the density ratio and its logarithm; $D^\star = \lambda$ | :numref:`sec_basic_gan` | the same |
| $\mu$ | a mean: of a Gaussian, hence the separation of $\mathcal{N}(0, 1)$ and $\mathcal{N}(\mu, 1)$; of features; and in :numref:`sec_gan_beyond` the conditional mean $\mu(c)$ | :numref:`sec_basic_gan` | $\delta$ or $d$ for the separation |
| $\delta_\theta$ | the point mass at $\theta$ | :numref:`sec_basic_gan` | the same |
| $b$ | an additive shift: of the critic, $D + b$; of a linear map, $z A + b$ or $z A_q + b_q$; of a feature map, $b(c)$; of a band in :numref:`sec_gan_beyond` | :numref:`sec_basic_gan` | the same |
| $d(p, q)$, $d_x$ | a discrepancy between two distributions; the data dimension | :numref:`sec_basic_gan` | $d$ for the dimension |
| $w_{\textrm{sat}}$, $w_{\textrm{ns}}$, $w_{\textrm{Rp}}$ | per-sample weights of the generator's update | :numref:`sec_basic_gan` | not named |
| $\eta = p/(p + q) = \sigma(\lambda)$ | the class posterior | :numref:`sec_gan_objectives` | the same; the step size is $h$ here, not $\eta$ |
| $\ell$, $L$, $\Delta_\ell$ | the payoff of a score, $\ell = \log \sigma$ for the log loss (the loss is $-\ell$), its conditional Bayes risk $L(\eta)$, and its Bayes-risk gap | :numref:`sec_gan_objectives` | the loss $-\ell$ |
| $T$, $\mathcal{T}$, $\mathcal{F}$, $\ell_p$, $\ell_q$ | a critic, or test function, in the template of :numref:`sec_gan_objectives`, its class ($\mathcal{F}$ for the class of an integral probability metric), and the template's scores of real and generated samples | :numref:`sec_gan_objectives` | $T$ for $f$-GAN critics, $f$ for the test functions of an integral probability metric |
| $f$, $D_f$ | a divergence function and its $f$-divergence | :numref:`sec_gan_objectives` | $f$ is called the generator of $D_f$ |
| $k$, $s$ | a kernel and its bandwidth | :numref:`sec_gan_objectives` | $\sigma$ or $\ell$ for the bandwidth, both taken here |
| $\Phi$, $d_{\textrm{Rp}}$ | the pairing objective and its value at the optimal critic | :numref:`sec_gan_relativistic` | Jolicoeur-Martineau's divergence is $2\, d_{\textrm{Rp}}$ |
| $P = p \otimes q$, $Q = q \otimes p$ | the two orderings of a real--generated pair | :numref:`sec_gan_relativistic` | the same |
| $(a_1, a_2)$, $\pi$ | an ordered pair holding one real and one generated sample; a permutation that pairs real with generated samples | :numref:`sec_gan_relativistic` | the same |
| $\mathcal{E}_{\mathrm{GAN}}$, $\mathcal{E}_{\mathrm{diag}}$, $\mathcal{E}_{\mathrm{all}}$ | the classical, paired and all-pairs losses at the critic's best response, as functions of the generated points | :numref:`sec_gan_relativistic` | not named |
| $v$, $\epsilon$ | the direction and the size of a small change $q = p\,(1 + \epsilon v)$ | :numref:`sec_gan_relativistic` | $h$ for the direction, the step size here |
| $\gamma$, $R_1$, $R_2$ | the penalty weight and the two zero-centered gradient penalties | :numref:`sec_gan_convergence` | the same |
| $h$, $\nu_{1,2}$, $\psi$, $\mathbf{u}$ | the step size, the eigenvalues, the critic's slope and the gradient field of the Dirac-GAN | :numref:`sec_gan_convergence` | :citet:`Mescheder.Geiger.Nowozin.2018` write $\lambda_{1,2}$ for the eigenvalues and $v$ for the field |
| $\omega$ | a wavenumber or angular frequency | :numref:`sec_gan_convergence` | $k$ for the wavenumber |
| $\phi$ | a feature map: of FID and KID, of a modulated layer, of the projection head | :numref:`sec_dcgan` | the same |
| $p_{\mathrm{aug}}$ | the probability of applying an augmentation | :numref:`sec_dcgan` | $p$, the data density here |
| $c$, $e_c$, $a(c)$, $\psi_0$, $\zeta_c$ | a condition, its embedding, a modulation's scale (its shift is $b(c)$), the unconditional head, the critic's free constant per class | :numref:`sec_gan_conditional` | $\gamma(c)$, $\beta(c)$ for the scale and shift |
| $g$ | a predictor of a sample $x$ from a condition $c$ | :numref:`sec_gan_beyond` | $f$, the divergence function here; $x \mapsto y$ for input and output in regression |
| $\xi$ | standard normal noise | :numref:`sec_gan_beyond` | $\epsilon$, a small number here |
| $\vartheta$ | a phase | :numref:`sec_gan_beyond` | $\varphi$, too close to the feature map $\phi$ |

Divergences and entropies follow the book's appendix: $D_{\mathrm{KL}}(p \,\|\, q)$ is the Kullback--Leibler divergence, $D_{\mathrm{J}} = D_{\mathrm{KL}}(p \,\|\, q) + D_{\mathrm{KL}}(q \,\|\, p)$ its symmetrized form, $\mathrm{JS}(p, q)$ the Jensen--Shannon divergence, $H(p)$ the entropy of a distribution and $H_b(t)$ that of a coin with bias $t$, all in nats.

## The Log-Loss Game

The natural loss for a probabilistic classifier is the log loss. On the balanced labeling problem, the critic's expected negative log-likelihood is $-\tfrac12 \mathbb{E}_{x \sim p}[\log \sigma(D(x))] - \tfrac12 \mathbb{E}_{x' \sim q}[\log \sigma(-D(x'))]$. Dropping the factor $\tfrac12$ and flipping the sign gives the objective the critic maximizes,

$$
V(D) \;=\; \mathbb{E}_{x \sim p}\big[\log \sigma(D(x))\big] + \mathbb{E}_{x' \sim q}\big[\log \sigma(-D(x'))\big],
$$
:eqlabel:`eq_gan_V`

the value function of :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`. The first term increases when the critic assigns high probability to real samples, the second when it assigns low probability to generated ones. Both terms are expectations of log-probabilities, so $V(D) \leq 0$. The generator influences only the second term and minimizes $V$, so training solves the *minimax* problem $\min_G \max_D V$. Because maximizing $V$ fits the best classifier, the value $\max_D V$ measures how well $p$ and $q$ can be told apart.

### The Optimal Critic

The maximization over $D$ looks like a problem in function space, but it separates into independent scalar problems. The integrand of :eqref:`eq_gan_V` at a point $x$ is $p(x) \log \sigma(D(x)) + q(x) \log \sigma(-D(x))$, which depends on $D$ only through the single number $D(x)$. Writing $u = \sigma(D(x)) \in (0, 1)$, the problem at each point is to maximize $p \log u + q \log(1-u)$, a strictly concave function of $u$ whose stationarity condition $p/u = q/(1-u)$ has the unique solution

$$
\sigma(D^\star(x)) = \frac{p(x)}{p(x) + q(x)} = P(y = 1 \mid x),
\qquad
D^\star(x) = \log \frac{p(x)}{q(x)} = \lambda(x).
$$
:eqlabel:`eq_gan_dstar`

Thus the optimal critic returns the Bayes posterior of the labeling problem, and its logit is the logarithm $\lambda = \log \rho$ of the *density ratio* $\rho = p/q$. Where one density vanishes, the optimal logit is infinite and is approached only in the limit. Although $q$ has no computable density, a classifier estimates the one function of $q$ the objective needs: an unconstrained critic trained with the log loss is a density-ratio estimator, like the domain classifier of covariate shift. The objective also fixes the critic's additive constant. Shifting $D^\star$ by any $b \neq 0$ strictly decreases $V$ (Exercise :exref:`ex_gan-log-loss-game-identifies`b), so the zero level of the score has a meaning: $D(x) > 0$ says that $x$ is more likely real than generated. Critics that compare pairs of samples, later in the chapter, give up this constant.

### One Dimension by Hand

Take two unit Gaussians a distance $\mu$ apart: $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$. The optimal critic is their log ratio,

$$
\lambda(x) = -\frac{x^2}{2} + \frac{(x - \mu)^2}{2} = \mu \Big(\frac{\mu}{2} - x\Big),
$$

a linear function of $x$, positive left of the midpoint $\mu/2$, where a sample is more likely real. A logistic regression $D(x) = wx + b$ therefore already contains the optimal critic, at $w = -\mu$ and $b = \mu^2/2$, and a deeper critic cannot do better. When the two Gaussians also differ in their covariances, as in the experiment at the end of this section, $\lambda$ is quadratic and a nonlinear critic is needed.

### The Value of the Game

Substituting the best response into the objective identifies the discrepancy minimized by the generator. With $\sigma(D^\star) = p/(p+q) = p/(2m)$ and $\sigma(-D^\star) = q/(2m)$,

$$
V(D^\star) = \int p \log \frac{p}{2m} + \int q \log \frac{q}{2m}
= D_{\mathrm{KL}}(p \,\|\, m) + D_{\mathrm{KL}}(q \,\|\, m) - 2 \log 2 .
$$

The two divergences are the ones that define the Jensen--Shannon divergence (:numref:`sec_mdl-divergences-distances`), $\mathrm{JS}(p, q) = \tfrac12 D_{\mathrm{KL}}(p \,\|\, m) + \tfrac12 D_{\mathrm{KL}}(q \,\|\, m)$, which compares each distribution with their mixture. Hence

$$
\max_D V(D) = 2\,\mathrm{JS}(p, q) - 2 \log 2 .
$$
:eqlabel:`eq_gan_js_value`

Against an optimal critic, the generator minimizes the Jensen--Shannon divergence, which vanishes exactly when $q = p$. Equivalently, the critic's smallest per-sample negative log-likelihood is $\log 2 - \mathrm{JS}(p, q)$: chance level minus the divergence. For the two Gaussians above, $\mathrm{JS}$ is 0.11 nats at $\mu = 1$, 0.34 at $\mu = 2$ and 0.63 at $\mu = 4$. At $\mu = 8$ it equals $\log 2 \approx 0.6931$ to four digits.

Two rewritings explain what kind of quantity the game has produced. Expanding each Kullback--Leibler term as $D_{\mathrm{KL}}(p \,\|\, m) = -H(p) - \int p \log m$ and collecting the mixture terms into $\int m \log m = -H(m)$ gives

$$
\mathrm{JS}(p, q) \;=\; H(m) - \frac{H(p) + H(q)}{2},
$$
:eqlabel:`eq_gan_entropy_gap`

where $H$ is the differential entropy. The first term is the uncertainty of a sample whose source is unknown, the second the average uncertainty once the source is known; by the strict concavity of entropy, the gap is nonnegative and vanishes only when $p = q$. The same two terms appear in the mutual information $I(x; y) = H(x) - H(x \mid y)$ between the source label $y$ and the sample $x \sim m$. Hence $\mathrm{JS}(p, q) = I(x; y)$: the divergence is the number of nats one sample provides about its source, and the optimal classifier improves on a random guess by exactly that amount.

The information reading also gives the bounds. A binary label carries at most one bit, $I(x; y) \leq H(y) = \log 2$, with equality exactly when $x$ determines $y$: when the supports of $p$ and $q$ are disjoint and every sample identifies its source. Together with nonnegativity,

$$
0 \;\leq\; \mathrm{JS}(p, q) \;\leq\; \log 2 .
$$

:numref:`sec_mdl-divergences-distances` derives the same bounds from the Kullback--Leibler form.

### Separated Supports

The ceiling has a cost. The two Gaussians at $\mu = 8$ already sit at $\log 2$ to four digits, and moving them further apart changes the value by less than $10^{-4}$ nats. In the limit of disjoint supports, the value stops depending on the distance altogether. The smallest example is two point masses on the line, $p = \delta_0$ and $q_\theta = \delta_\theta$. For every $\theta \neq 0$ their supports are disjoint, $\mathrm{JS}(p, q_\theta) = \log 2$, and the value of the game does not depend on $\theta$. A constant objective supplies no gradient with respect to $\theta$, even when the generator lies arbitrarily close to the data.

The case is not exotic. A generator that maps a $d_z$-dimensional latent variable into a higher-dimensional space places its samples near a $d_z$-dimensional surface, and natural images also concentrate near a low-dimensional set, so the two supports are typically disjoint or nearly so :cite:`Arjovsky.Bottou.2017`. An optimal critic then classifies perfectly, and the generator learns nothing from the value of the game.

## The Generator's Gradient

The value describes the game at the critic's best response, but training updates the generator by gradient steps with the critic held fixed. We now follow the gradient the generator actually receives.

### Two Weights

The generator influences only the second term of :eqref:`eq_gan_V`, so playing the minimax game literally means minimizing $\mathbb{E}_z[\log \sigma(-D(G(z)))]$ over the generator's parameters $\theta$, with the critic fixed. Differentiating through $x' = G(z)$ with the identity $\tfrac{d}{du} \log \sigma(-u) = -\sigma(u)$ gives

$$
\nabla_\theta \, \mathbb{E}_z\big[\log \sigma(-D(G(z)))\big]
= -\,\mathbb{E}_z\big[\sigma(D(x'))\, \nabla_\theta D(G(z))\big],
$$

where $\nabla_\theta D(G(z))$ is the gradient of the composite map. A descent step therefore moves each generated sample in the direction that raises its critic score, with weight $\sigma(D(x'))$, the critic's probability that the sample is real. Early in training, or for a poorly modeled sample later on, $D(x') \ll 0$ and this weight is exponentially small. The loss has *saturated*: the samples the critic rejects most confidently contribute least to the update.

:citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` therefore substitute a different generator loss in practice. Rather than minimizing the probability of being called fake, the generator maximizes the probability of being called real, minimizing $-\mathbb{E}_z[\log \sigma(D(G(z)))]$. The same differentiation, now with $\tfrac{d}{du} \log \sigma(u) = \sigma(-u)$, gives

$$
\nabla_\theta \Big({-\mathbb{E}_z\big[\log \sigma(D(G(z)))\big]}\Big)
= -\,\mathbb{E}_z\big[\sigma(-D(x'))\, \nabla_\theta D(G(z))\big].
$$

The two losses move each sample in the *same* direction $\nabla_\theta D$ and differ only in the weight attached to it:

$$
w_{\textrm{sat}}(x') = \sigma(D(x')),
\qquad
w_{\textrm{ns}}(x') = \sigma(-D(x')) = 1 - \sigma(D(x')).
$$
:eqlabel:`eq_gan_weights`

The *non-saturating* weight $w_{\textrm{ns}}$ is near one where the saturating weight is near zero, as :numref:`fig_gan_weights` shows. Their ratio is $w_{\textrm{ns}}/w_{\textrm{sat}} = e^{-D(x')}$: a sample with score $-3$ receives 20 times more weight under the non-saturating loss, one with score $-5$ about 148 times more.

![The two generator weights of :eqref:`eq_gan_weights` against the critic score $D(x')$ of a generated sample, on a logarithmic axis. The saturating weight $w_{\textrm{sat}} = \sigma(D)$ falls exponentially for a sample the critic rejects, while the non-saturating weight $w_{\textrm{ns}} = \sigma(-D)$ stays near one. Their ratio is exactly $e^{-D}$: about 20 at $D = -3$ and 148 at $D = -5$.](../img/mdl-gan-weights.svg)
:label:`fig_gan_weights`

### The Same Fixed Point

Changing the weight does not change the population minimizer against an optimal critic. Substituting $D^\star = \lambda$ into the non-saturating loss and using $\sigma(\lambda) = p/(2m)$,

$$
-\mathbb{E}_{x' \sim q}\big[\log \sigma(\lambda(x'))\big]
= \log 2 + D_{\mathrm{KL}}(q \,\|\, p) - D_{\mathrm{KL}}(q \,\|\, m)
\;\geq\; \log 2 + \tfrac12\, D_{\mathrm{KL}}(q \,\|\, p).
$$
:eqlabel:`eq_gan_ns_value`

The inequality follows from the convexity of $D_{\mathrm{KL}}(q \,\|\, \cdot)$ and the fact that $m$ is the midpoint of $p$ and $q$: $D_{\mathrm{KL}}(q \,\|\, m) \leq \tfrac12 D_{\mathrm{KL}}(q \,\|\, p) + \tfrac12 D_{\mathrm{KL}}(q \,\|\, q)$, where the last term is zero. The bound exceeds $\log 2$ unless $q = p$, where the loss equals $\log 2$. With an optimal critic, the non-saturating loss is therefore minimized only at $q = p$.

The bound evaluates the loss at the critic that is optimal for each $q$. The generator's update differentiates with the critic held fixed, which is a different function of $q$. :citet:`Arjovsky.Bottou.2017` computed it: with the critic fixed at $D = \lambda$, the non-saturating update descends

$$
D_{\mathrm{KL}}(q \,\|\, p) - 2\,\mathrm{JS}(p, q),
$$

an $f$-divergence whose function $f$ is strictly convex (Exercise :exref:`ex_gan-generator-weight-divergences` derives it). So $q = p$ is again its only minimizer. A gradient step on the generator's parameters lowers this divergence, but it sees the divergence only where $q$ already has mass. It can therefore stall at a $q \neq p$ where the divergence is flat in those parameters, as the next example shows.

### Mode Dropping
:label:`subsec_gan_mode_dropping`

That divergence inherits the asymmetry of its first term, the reverse Kullback--Leibler divergence $D_{\mathrm{KL}}(q \,\|\, p)$, which charges the generator heavily for placing mass where the data have little, but lightly for failing to place mass where the data have much. Take $p$ an equal mixture of two well-separated modes and $q$ a copy of one of them. On the covered mode $p = q/2$, so $D_{\mathrm{KL}}(q \,\|\, p) = \log 2 \approx 0.69$ nats, and $\mathrm{JS}(p, q) \approx 0.22$ nats, a third of its ceiling. The forward divergence $D_{\mathrm{KL}}(p \,\|\, q)$ is infinite, because $q$ misses half of $p$. A generator that covers only some of the data's modes is said to suffer *mode dropping*.

Gradient steps rarely leave such a state. With the critic at its best response, the score on the covered mode is the constant $\log(p/q) = -\log 2$, so $\nabla_\theta D$ vanishes there and neither weight produces a push toward the missing mode; reaching it would also mean moving mass across the gap, where $p$ is nearly zero. A critic away from its best response shows a second blindness, through the weights. Both classical weights read a sample's own score $D(x')$ against the zero level that the log-loss game fixes, and nothing in them compares the sample with the real data. A generator whose samples all score above that level receives little push, whatever the data's other modes do. Later in the chapter, a critic that ranks a generated sample against a real one replaces this threshold with a comparison.

## Fitting a Gaussian

The analysis makes three predictions. Near the fixed point, the critic's per-sample loss approaches $\log 2 \approx 0.693$. A trained critic approximates $\lambda = \log(p/q)$. And a confidently rejected generator keeps learning with the non-saturating loss but stalls with the saturating one. When the data are Gaussian and the generator is linear, $p$, $q$ and $\lambda$ are known in closed form, so all three can be checked exactly.

### Data and Models

The data are a standard Gaussian pushed through a fixed linear map: $x = z A + b$ with $z \sim \mathcal{N}(0, I_2)$, giving $p = \mathcal{N}(b, A^\top A)$. The chosen $A$ makes the cloud a thin tilted ellipse rather than a round blob, so matching it requires learning an orientation and two very different scales. Training uses a fixed sample of one thousand draws; the comparisons below use the population densities.

```{.python .input #gan-data-and-models-1}
%%tab pytorch
torch.manual_seed(0)
Z = torch.normal(0.0, 1.0, (1000, 2))
A = torch.tensor([[1.0, 2.0], [-0.1, 0.5]])
b = torch.tensor([1.0, 2.0])
data = Z @ A + b
print(f'covariance of the data distribution:\n{A.T @ A}')
d2l.plot(data[:100, 0], data[:100, 1], fmts=('C0.',), figsize=(3.5, 3))
```

```{.python .input #gan-data-and-models-1}
%%tab jax
Z = jax.random.normal(jax.random.key(0), (1000, 2))
A = jnp.array([[1.0, 2.0], [-0.1, 0.5]])
b = jnp.array([1.0, 2.0])
data = Z @ A + b
print(f'covariance of the data distribution:\n{A.T @ A}')
d2l.plot(data[:100, 0], data[:100, 1], fmts=('C0.',), figsize=(3.5, 3))
```

The generator is a single linear layer, so its output distribution is Gaussian and its mean and covariance can be read off the weights. The critic is a multilayer perceptron with two hidden layers of 16 leaky ReLU units, which pass negative inputs with slope 0.2 instead of zeroing them. With leaky units the critic's score surface keeps a slope wherever the units are active, so the generator almost always receives a direction; tanh units can saturate and leave the surface flat (Exercise :exref:`ex_gan-dead-critic`). The critic sees only samples, never the densities used in the analysis. A factory function builds both networks from a fixed seed so that later cells can restart from the same initialization.

```{.python .input #gan-data-and-models-2}
%%tab pytorch
def make_nets():
    torch.manual_seed(7)
    net_G = nn.Sequential(nn.Linear(2, 2))
    net_D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2),
                          nn.Linear(16, 16), nn.LeakyReLU(0.2),
                          nn.Linear(16, 1))
    for net in (net_G, net_D):
        for w in net.parameters():
            nn.init.normal_(w, 0, 0.02)
    return net_G, net_D
```

```{.python .input #gan-data-and-models-2}
%%tab jax
class Generator(nnx.Module):
    def __init__(self, rngs):
        normal = nnx.initializers.normal(0.02)
        self.out = nnx.Linear(2, 2, kernel_init=normal, bias_init=normal,
                              rngs=rngs)

    def __call__(self, x):
        return self.out(x)

class Discriminator(nnx.Module):
    def __init__(self, rngs):
        normal = nnx.initializers.normal(0.02)
        self.h1 = nnx.Linear(2, 16, kernel_init=normal, bias_init=normal,
                             rngs=rngs)
        self.h2 = nnx.Linear(16, 16, kernel_init=normal, bias_init=normal,
                             rngs=rngs)
        self.out = nnx.Linear(16, 1, kernel_init=normal, bias_init=normal,
                              rngs=rngs)

    def __call__(self, x):
        h = nnx.leaky_relu(self.h1(x), 0.2)
        return self.out(nnx.leaky_relu(self.h2(h), 0.2))

def make_nets():
    return Generator(nnx.Rngs(7)), Discriminator(nnx.Rngs(8))
```

### The Update Rules

One training iteration is two half-steps, one per player. The critic's update ascends :eqref:`eq_gan_V` on a minibatch, implemented as logistic regression with label one for data and zero for generated samples; the generator's parameters are held fixed during this half-step. Both update functions are saved to the `d2l` library, since later sections train image networks with the same rules.

```{.python .input #gan-the-update-rules-1}
%%tab pytorch
#@save
def update_D(X, Z, net_D, net_G, trainer_D):
    """Update the discriminator; return the summed log loss, halved."""
    trainer_D.zero_grad()
    real_Y = net_D(X)
    # The generator is the fixed player here: detach its output
    fake_Y = net_D(net_G(Z).detach())
    loss_D = (F.binary_cross_entropy_with_logits(
                  real_Y, torch.ones_like(real_Y), reduction='sum') +
              F.binary_cross_entropy_with_logits(
                  fake_Y, torch.zeros_like(fake_Y), reduction='sum')) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

```{.python .input #gan-the-update-rules-1}
%%tab jax
#@save
@nnx.jit
def update_D(X, Z, net_D, net_G, optimizer_D):
    """Update the discriminator; return the summed log loss, halved."""
    batch_size = X.shape[0]
    fake_X = net_G(Z)  # computed outside the loss: no gradient to net_G
    def loss_D_fn(model_D):
        real_Y = model_D(X).squeeze()
        fake_Y = model_D(fake_X).squeeze()
        return (jnp.sum(optax.sigmoid_binary_cross_entropy(
                    real_Y, jnp.ones(batch_size))) +
                jnp.sum(optax.sigmoid_binary_cross_entropy(
                    fake_Y, jnp.zeros(batch_size)))) / 2
    loss_D, grads_D = nnx.value_and_grad(loss_D_fn)(net_D)
    optimizer_D.update(net_D, grads_D)
    return loss_D
```

The generator's update implements the non-saturating loss of :eqref:`eq_gan_weights`: generated samples are fed to the critic under the label *one*, so that the cross-entropy is $-\mathbb{E}[\log \sigma(D(G(z)))]$. The critic has changed in its own half-step, so the scores are recomputed rather than reused.

```{.python .input #gan-the-update-rules-2}
%%tab pytorch
#@save
def update_G(Z, net_D, net_G, trainer_G):
    """Update the generator on the non-saturating loss; return its sum."""
    trainer_G.zero_grad()
    # net_D changed in its half-step, so recompute the scores
    fake_Y = net_D(net_G(Z))
    loss_G = F.binary_cross_entropy_with_logits(
        fake_Y, torch.ones_like(fake_Y), reduction='sum')
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

```{.python .input #gan-the-update-rules-2}
%%tab jax
#@save
@nnx.jit
def update_G(Z, net_D, net_G, optimizer_G):
    """Update the generator on the non-saturating loss; return its sum."""
    # Both networks are arguments, so critics with mutable state (batch
    # norm) work too; argnums=0 differentiates the generator only
    def loss_G_fn(model_G, model_D):
        fake_Y = model_D(model_G(Z)).squeeze()
        return jnp.sum(optax.sigmoid_binary_cross_entropy(
            fake_Y, jnp.ones(Z.shape[0])))
    loss_G, grads_G = nnx.value_and_grad(loss_G_fn, argnums=0)(net_G, net_D)
    optimizer_G.update(net_G, grads_G)
    return loss_G
```

### Training
:label:`subsec_gan_training`

The loop alternates the two half-steps over minibatches, records each player's per-sample loss, and keeps generated samples at chosen epochs for the comparison at the end of the section. Both players use Adam with $\beta_1 = 0.5$ instead of the default 0.9, the setting :citet:`Radford.Metz.Chintala.2015` chose for the DCGAN; later recipes lower it further, to zero. A shorter momentum lets each player respond sooner to the other's latest move. The critic gets the larger learning rate, so that its scores stay close to their best response to the current generator.

```{.python .input #gan-training-1}
%%tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr_D=0.05, lr_G=0.005,
          step_G=update_G, snapshot_epochs=()):
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D,
                                 betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G,
                                 betas=(0.5, 0.999))
    history, snapshots = [], {}
    for epoch in range(1, num_epochs + 1):
        loss_D, loss_G, n = 0, 0, 0  # summed on the device, read once
        for (X,) in data_iter:
            Zb = torch.normal(0, 1, (X.shape[0], 2))
            loss_D += update_D(X, Zb, net_D, net_G, trainer_D).detach()
            loss_G += step_G(Zb, net_D, net_G, trainer_G).detach()
            n += X.shape[0]
        history.append(torch.stack([loss_D, loss_G]) / n)
        if epoch in snapshot_epochs:
            with torch.no_grad():
                snapshots[epoch] = net_G(torch.normal(0, 1, (200, 2)))
    return torch.stack(history), snapshots
```

```{.python .input #gan-training-1}
%%tab jax
def train(net_D, net_G, data_iter, num_epochs, lr_D=0.05, lr_G=0.005,
          step_G=update_G, snapshot_epochs=(), seed=1):
    key = jax.random.key(seed)
    optimizer_D = nnx.Optimizer(net_D, optax.adam(lr_D, b1=0.5),
                                wrt=nnx.Param)
    optimizer_G = nnx.Optimizer(net_G, optax.adam(lr_G, b1=0.5),
                                wrt=nnx.Param)
    history, snapshots = [], {}
    for epoch in range(1, num_epochs + 1):
        loss_D, loss_G, n = 0.0, 0.0, 0  # summed on the device, read once
        for (X,) in data_iter:
            key, subkey = jax.random.split(key)
            Zb = jax.random.normal(subkey, (X.shape[0], 2))
            loss_D += update_D(X, Zb, net_D, net_G, optimizer_D)
            loss_G += step_G(Zb, net_D, net_G, optimizer_G)
            n += X.shape[0]
        history.append((loss_D / n, loss_G / n))
        if epoch in snapshot_epochs:
            key, subkey = jax.random.split(key)
            snapshots[epoch] = net_G(jax.random.normal(subkey, (200, 2)))
    return np.array(history), snapshots
```

Twenty epochs over the thousand points, in minibatches of eight, suffice for the default run.

```{.python .input #gan-training-2}
%%tab pytorch
data_iter = d2l.load_array((data,), batch_size=8)
net_G, net_D = make_nets()
history, _ = train(net_D, net_G, data_iter, num_epochs=20)
with torch.no_grad():
    fake = net_G(torch.normal(0, 1, (200, 2)))
```

```{.python .input #gan-training-2}
%%tab jax
data_iter = d2l.ArrayLoader((data,), batch_size=8, shuffle=True,
                            key=jax.random.key(0))
net_G, net_D = make_nets()
history, _ = train(net_D, net_G, data_iter, num_epochs=20)
fake = net_G(jax.random.normal(jax.random.key(4), (200, 2)))
```

The left panel tracks both losses against the dotted line at $\log 2$; the right panel overlays generated samples on the data.

```{.python .input #gan-training-3}
%%tab pytorch, jax
fig, axes = d2l.plt.subplots(1, 2, figsize=(9, 3.2))
epochs = list(range(1, len(history) + 1))
d2l.plot(epochs, [history[:, 0], history[:, 1], [math.log(2)] * len(epochs)],
         'epoch', 'per-sample loss', legend=['critic', 'generator', 'log 2'],
         fmts=('-', '--', 'k:'), axes=axes[0])
d2l.plot([data[:100, 0], fake[:100, 0]], [data[:100, 1], fake[:100, 1]],
         legend=['real', 'generated'], fmts=('C0.', 'C1.'), axes=axes[1])
fig.tight_layout()
```

The critic's loss starts below $\log 2$ and rises toward it as the generator improves, consistent with $\mathrm{JS}$ approaching zero, since the best critic's per-sample loss is $\log 2 - \mathrm{JS}(p, q)$. The generator's loss approaches the same level from above: by :eqref:`eq_gan_ns_value`, the non-saturating loss has its minimum $\log 2$ at $q = p$. The generated samples fill the same tilted ellipse as the data. Still, a critic loss of $\log 2$ shows only that this critic can no longer separate the samples, not that $q = p$, so the next two measurements check the fit independently.

### The Fit in Nats
:label:`subsec_gan_fit`

The linear generator lets us measure the fit in nats. A linear layer $G(z) = z A_q + b_q$ applied to $z \sim \mathcal{N}(0, I_2)$ produces the Gaussian $q = \mathcal{N}(b_q, A_q^\top A_q)$. For Gaussians in $d_x$ dimensions the Kullback--Leibler divergence has the closed form :cite:`Kullback.1959`

$$
D_{\mathrm{KL}}\big(\mathcal{N}(\mu_0, \Sigma_0) \,\|\, \mathcal{N}(\mu_1, \Sigma_1)\big)
= \tfrac12 \Big[ \operatorname{tr}(\Sigma_1^{-1} \Sigma_0)
+ (\mu_1 - \mu_0)^\top \Sigma_1^{-1} (\mu_1 - \mu_0)
- d_x + \log \tfrac{\det \Sigma_1}{\det \Sigma_0} \Big],
$$
:eqlabel:`eq_gan_mvn_kl`

the multivariate form of the univariate formula of :numref:`sec_mdl-information_theory`. We report $D_{\mathrm{KL}}(q \,\|\, p)$, the direction that charges the generator for placing mass where the data have little, at initialization and after training.

```{.python .input #gan-quantifying-the-fit}
%%tab pytorch
def generator_gaussian(net_G):
    A_qT, b_q = net_G[0].weight.detach(), net_G[0].bias.detach()
    return b_q, A_qT @ A_qT.T  # nn.Linear stores A_q.T

def kl_gaussians(mu0, S0, mu1, S1):
    diff = mu1 - mu0
    return 0.5 * (torch.trace(torch.linalg.solve(S1, S0))
                  + diff @ torch.linalg.solve(S1, diff) - len(mu0)
                  + torch.logdet(S1) - torch.logdet(S0))

mu_p, Sigma_p = b, A.T @ A
for name, net in [('at initialization', make_nets()[0]),
                  ('after training   ', net_G)]:
    mu_q, Sigma_q = generator_gaussian(net)
    print(f'KL(q || p) {name}: '
          f'{kl_gaussians(mu_q, Sigma_q, mu_p, Sigma_p):.3f} nats')
```

```{.python .input #gan-quantifying-the-fit}
%%tab jax
def generator_gaussian(net_G):
    A_q, b_q = net_G.out.kernel[...], net_G.out.bias[...]
    return b_q, A_q.T @ A_q

def kl_gaussians(mu0, S0, mu1, S1):
    diff = mu1 - mu0
    return 0.5 * (jnp.trace(jnp.linalg.solve(S1, S0))
                  + diff @ jnp.linalg.solve(S1, diff) - len(mu0)
                  + jnp.linalg.slogdet(S1)[1] - jnp.linalg.slogdet(S0)[1])

mu_p, Sigma_p = b, A.T @ A
for name, net in [('at initialization', make_nets()[0]),
                  ('after training   ', net_G)]:
    mu_q, Sigma_q = generator_gaussian(net)
    print(f'KL(q || p) {name}: '
          f'{kl_gaussians(mu_q, Sigma_q, mu_p, Sigma_p):.3f} nats')
```

The divergence falls from about eight nats at initialization to well below one, typically about a hundredth of a nat or less. The training loop never evaluates the generator's density, yet the critic supplies enough information to fit its parameters.

### The Learned Density Ratio
:label:`subsec_gan_verify_critic`

:eqref:`eq_gan_dstar` says that the best response to a *fixed* generator computes $\lambda = \log(p/q)$. To check it we freeze a generator and train a fresh critic against it alone. The mismatch must be large enough to measure: after the run above, $\lambda$ is close to zero everywhere and even a constant critic would look accurate. We therefore set the generator by hand to $G(z) = z A_q + b_q$ with $A_q = [[1, 2], [-0.5, 0.5]]$ and $b_q = (1.5, 1.5)$, a Gaussian $q$ at $D_{\mathrm{KL}}(q \,\|\, p) \approx 3.4$ nats from the data, and train a critic of the same architecture for 2,000 steps.

```{.python .input #gan-verifying-the-optimal-critic-1}
%%tab pytorch
def gaussian_logpdf(x, mu, Sigma):
    diff = x - mu
    sol = torch.linalg.solve(Sigma, diff.T).T
    return -0.5 * ((diff * sol).sum(1)
                   + torch.logdet(2 * torch.pi * Sigma))

net_G_q, net_D_q = make_nets()  # a fresh critic
with torch.no_grad():  # the hand-set generator; nn.Linear stores A_q.T
    net_G_q[0].weight.copy_(torch.tensor([[1.0, -0.5], [2.0, 0.5]]))
    net_G_q[0].bias.copy_(torch.tensor([1.5, 1.5]))
trainer_D = torch.optim.Adam(net_D_q.parameters(), lr=0.01)
torch.manual_seed(2)
for _ in range(2000):  # critic-only steps against the frozen generator
    X = data[torch.randint(0, 1000, (256,))]
    Zb = torch.normal(0, 1, (256, 2))
    update_D(X, Zb, net_D_q, net_G_q, trainer_D)
```

```{.python .input #gan-verifying-the-optimal-critic-1}
%%tab jax
def gaussian_logpdf(x, mu, Sigma):
    diff = x - mu
    sol = jnp.linalg.solve(Sigma, diff.T).T
    return -0.5 * ((diff * sol).sum(1)
                   + jnp.linalg.slogdet(2 * jnp.pi * Sigma)[1])

net_G_q, net_D_q = make_nets()  # a fresh critic
net_G_q.out.kernel[...] = jnp.array([[1.0, 2.0], [-0.5, 0.5]])  # A_q
net_G_q.out.bias[...] = jnp.array([1.5, 1.5])  # b_q
optimizer_D = nnx.Optimizer(net_D_q, optax.adam(0.01), wrt=nnx.Param)
key = jax.random.key(2)
for _ in range(2000):  # critic-only steps against the frozen generator
    key, k1, k2 = jax.random.split(key, 3)
    X = data[jax.random.randint(k1, (256,), 0, 1000)]
    Zb = jax.random.normal(k2, (256, 2))
    update_D(X, Zb, net_D_q, net_G_q, optimizer_D)
```

We compare the critic with the analytic $\lambda$ at two hundred real and two hundred generated samples. Two numbers summarize the agreement: the correlation, and the fraction $R^2$ of the variance of $\lambda$ that the critic explains. A constant critic, the best a critic can do without looking at $x$, has $R^2 = 0$.

```{.python .input #gan-verifying-the-optimal-critic-2}
%%tab pytorch
mu_q, Sigma_q = generator_gaussian(net_G_q)
with torch.no_grad():
    xs = torch.cat([data[:200], net_G_q(torch.normal(0, 1, (200, 2)))])
    lam = (gaussian_logpdf(xs, b, A.T @ A)
           - gaussian_logpdf(xs, mu_q, Sigma_q))
    D_out = net_D_q(xs).squeeze()
r2 = 1 - ((D_out - lam) ** 2).mean() / ((lam - lam.mean()) ** 2).mean()
print(f'correlation {torch.corrcoef(torch.stack([lam, D_out]))[0, 1]:.3f}, '
      f'R^2 {r2:.3f}')
err, tail = (D_out - lam).abs(), lam.abs() >= lam.abs().quantile(0.9)
print(f'mean |D - lambda|: {err[tail].mean():.2f} nats on the extreme '
      f'tenth of |lambda|, {err[~tail].mean():.2f} on the rest')
```

```{.python .input #gan-verifying-the-optimal-critic-2}
%%tab jax
mu_q, Sigma_q = generator_gaussian(net_G_q)
key, subkey = jax.random.split(key)
xs = jnp.concatenate([data[:200],
                      net_G_q(jax.random.normal(subkey, (200, 2)))])
lam = (gaussian_logpdf(xs, b, A.T @ A)
       - gaussian_logpdf(xs, mu_q, Sigma_q))
D_out = net_D_q(xs).squeeze()
r2 = 1 - ((D_out - lam) ** 2).mean() / ((lam - lam.mean()) ** 2).mean()
print(f'correlation {jnp.corrcoef(jnp.stack([lam, D_out]))[0, 1]:.3f}, '
      f'R^2 {r2:.3f}')
err = jnp.abs(D_out - lam)
tail = jnp.abs(lam) >= jnp.quantile(jnp.abs(lam), 0.9)
print(f'mean |D - lambda|: {err[tail].mean():.2f} nats on the extreme '
      f'tenth of |lambda|, {err[~tail].mean():.2f} on the rest')
```

The scatter plot puts the analytic $\lambda$ on the horizontal axis and the critic's output on the vertical one; a perfect critic lies on the dashed diagonal.

```{.python .input #gan-the-learned-density-ratio-1}
%%tab pytorch, jax
lims = [float(lam.min()), float(lam.max())]
d2l.plot([lam[:200], lam[200:], lims], [D_out[:200], D_out[200:], lims],
         r'analytic $\lambda = \log(p/q)$', 'critic output $D$',
         legend=['at real samples', 'at generated samples', r'$D = \lambda$'],
         fmts=('C0.', 'C1.', 'k--'), figsize=(4.2, 3.4))
```

A network trained only to classify has learned the log density ratio, as :eqref:`eq_gan_dstar` predicts: the correlation exceeds 0.97 and the critic explains more than 90 percent of the variance of $\lambda$. The errors concentrate in the tails. On the tenth of points with the most extreme $\lambda$, generated samples where $p$ is nearly zero, the critic misses by one to a few nats; elsewhere by a few tenths or less. The objective constrains the critic only where the mixture $m$ has appreciable mass, so density-ratio estimates are least accurate where the ratio is most extreme.

### Saturating and Non-Saturating Losses Compared
:label:`subsec_gan_saturation`

Testing the two weights of :eqref:`eq_gan_weights` needs a start at which they differ. In the main run the initial generator emits samples near the origin, about one standard deviation from the data mean; the critic never rejects them confidently enough to drive $w_{\textrm{sat}}$ toward zero, and the saturating loss also trains from that start. We therefore move the generator's bias far from the data, where the critic can reject its samples with confidence. Two runs start from this same initialization, see the same minibatches and latent draws, and differ only in the generator loss.

```{.python .input #gan-saturating-versus-non-saturating-training-1}
%%tab pytorch
def update_G_saturating(Z, net_D, net_G, trainer_G):
    trainer_G.zero_grad()
    fake_Y = net_D(net_G(Z))
    # minimize E[log sigma(-D)]: the minimax game taken literally
    loss_G = -F.binary_cross_entropy_with_logits(
        fake_Y, torch.zeros_like(fake_Y), reduction='sum')
    loss_G.backward()
    trainer_G.step()
    return loss_G

runs = {}
for name, step in [('non-saturating', update_G),
                   ('saturating', update_G_saturating)]:
    net_G_ab, net_D_ab = make_nets()
    with torch.no_grad():
        net_G_ab[0].bias += torch.tensor([-6.0, 6.0])
    torch.manual_seed(1)  # the same minibatches and latent draws per arm
    hist, snaps = train(net_D_ab, net_G_ab, data_iter, num_epochs=20,
                        lr_G=0.01, step_G=step, snapshot_epochs=(5, 20))
    runs[name] = (hist, snaps)
    mu_q, Sigma_q = generator_gaussian(net_G_ab)
    print(f'{name}: final KL(q || p) = '
          f'{kl_gaussians(mu_q, Sigma_q, mu_p, Sigma_p):.2f} nats')
```

```{.python .input #gan-saturating-versus-non-saturating-training-1}
%%tab jax
@nnx.jit
def update_G_saturating(Z, net_D, net_G, optimizer_G):
    def loss_G_fn(model_G, model_D):
        fake_Y = model_D(model_G(Z)).squeeze()
        # minimize E[log sigma(-D)]: the minimax game taken literally
        return -jnp.sum(optax.sigmoid_binary_cross_entropy(
            fake_Y, jnp.zeros(Z.shape[0])))
    loss_G, grads_G = nnx.value_and_grad(loss_G_fn, argnums=0)(net_G, net_D)
    optimizer_G.update(net_G, grads_G)
    return loss_G

runs = {}
for name, step in [('non-saturating', update_G),
                   ('saturating', update_G_saturating)]:
    net_G_ab, net_D_ab = make_nets()
    net_G_ab.out.bias[...] += jnp.array([-6.0, 6.0])
    # the same keys per arm: the same minibatches and latent draws
    data_iter_ab = d2l.ArrayLoader((data,), batch_size=8, shuffle=True,
                                   key=jax.random.key(9))
    hist, snaps = train(net_D_ab, net_G_ab, data_iter_ab, num_epochs=20,
                        lr_G=0.01, step_G=step, snapshot_epochs=(5, 20),
                        seed=9)
    runs[name] = (hist, snaps)
    mu_q, Sigma_q = generator_gaussian(net_G_ab)
    print(f'{name}: final KL(q || p) = '
          f'{kl_gaussians(mu_q, Sigma_q, mu_p, Sigma_p):.2f} nats')
```

The first two panels show each run's samples after five and after twenty epochs over the data; the third shows both critics' losses.

```{.python .input #gan-saturating-and-non-saturating-losses-compared-1}
%%tab pytorch, jax
fig, axes = d2l.plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, (name, (hist, snaps)) in zip(axes[:2], runs.items()):
    d2l.plot([data[:150, 0], snaps[5][:, 0], snaps[20][:, 0]],
             [data[:150, 1], snaps[5][:, 1], snaps[20][:, 1]],
             legend=['real', 'epoch 5', 'epoch 20'],
             fmts=('C0.', 'C1+', 'C1.'), axes=ax)
    ax.set_title(name)
epochs = list(range(1, 21))
d2l.plot(epochs, [runs['non-saturating'][0][:, 0],
                  runs['saturating'][0][:, 0], [math.log(2)] * 20],
         'epoch', 'critic loss', legend=['non-saturating', 'saturating',
                                         'log 2'],
         fmts=('-', '--', 'k:'), axes=axes[2])
fig.tight_layout()
```

The saturating generator does not move. Its critic's loss drops to zero within the first epochs, so $\sigma(D(x')) \approx 0$ for every generated sample and every update has a weight near zero. The generator stays put, the critic keeps rejecting it, and the run ends where it started, more than three hundred nats from the data. The non-saturating generator crosses to the data within the first few epochs: by epoch five its samples already lie in the data region, though not yet centered on it. By epoch twenty they fill the ellipse, $D_{\mathrm{KL}}(q \,\|\, p)$ is well below one nat, and its critic's loss has climbed back to $\log 2$.

The non-saturating weight amplifies the factor $\nabla_\theta D$ but cannot create it. Here the critic's score surface slopes toward the data even far from them, and the reweighting turns that slope into progress. An optimal critic on separated supports, as in the point-mass example, offers no slope, and no choice of weight helps.

## Summary

A GAN trains a sampler without a density by replacing the likelihood with a learned comparison: a critic learns to tell real from generated samples, and the generator learns to defeat it. With the log loss, the optimal critic's logit is the log density ratio $\lambda = \log(p/q)$, and the value of the game is $2\,\mathrm{JS}(p,q) - 2\log 2$, where $\mathrm{JS}$ is the information a sample carries about its source. The saturating and non-saturating generator losses share the fixed point $q = p$, but only the second keeps learning from confidently rejected samples, as a two-dimensional Gaussian confirmed. Two limits remain: the value is constant on separated supports, and the divergence that the non-saturating generator descends inherits the asymmetry of the reverse KL, so it charges little for a dropped mode. :numref:`sec_gan_objectives` changes the loss and the critic class to obtain discrepancies that keep a gradient when the supports separate.

## Exercises

1. **Likelihood versus samples.** The noise mixture
   :eqref:`eq_gan_noise_mixture` of :citet:`Theis.Oord.Bethge.2016` has a
   nearly optimal likelihood and poor samples.
    1. Replace the weights $0.01$ and $0.99$ by $\epsilon$ and $1 - \epsilon$.
       The argument after :eqref:`eq_gan_noise_mixture` then bounds the
       shortfall
       $\mathbb{E}_{p_{\textrm{data}}}[\log p_{\textrm{data}}] - \mathbb{E}_{p_{\textrm{data}}}[\log \tilde p]$
       by $\log(1/\epsilon)$. Write the shortfall exactly as an expectation
       under $p_{\textrm{data}}$ of a function of the ratio
       $p_{\textrm{noise}} / p_{\textrm{data}}$. When is the bound nearly
       attained? Is it nearly attained for natural images when
       $p_{\textrm{noise}}$ is the uniform density on the pixel cube?
    1. Image models often report likelihoods in bits per dimension: the
       negative log-likelihood divided by the number of dimensions and by
       $\log 2$. Convert the bound for $\epsilon = 0.01$ into bits per
       dimension for $32 \times 32$ color images, with $d_x = 3072$. Would a
       table of results rounded to two decimals reveal the difference?
    1. Now take the converse case. Let the data distribution be
       $p = \mathcal{N}(0, I_{d_x})$ and let a model be
       $q = \mathcal{N}(0, s^2 I_{d_x})$ with $s < 1$. Each sample of $q$ is then
       a data sample shrunk by the factor $s$, which for images resembles a
       slight loss of contrast. Use :eqref:`eq_gan_mvn_kl` to compute the gap
       $\mathbb{E}_p[\log p] - \mathbb{E}_p[\log q]$ for $d_x = 3072$ and $s = 0.9$. Find the $s$
       at which the gap equals the $\log 100$ nats of the noise mixture. Which
       of the two models wins by likelihood, and which by its samples?
   :exlabel:`ex_gan-likelihood-versus-samples`

1. **Excess loss and the additive constant.** Let
   $\eta(x) = p(x) / (p(x) + q(x))$ be the posterior of
   :eqref:`eq_gan_dstar`, and let
   $D_{\textrm{b}}(a \,\|\, u) = a \log \frac{a}{u} + (1 - a) \log \frac{1 - a}{1 - u}$
   be the Kullback--Leibler divergence between Bernoulli distributions with
   success probabilities $a$ and $u$.
    1. Show that every critic $D$ satisfies

       $$V(D^\star) - V(D) = 2\, \mathbb{E}_{x \sim m}\big[D_{\textrm{b}}\big(\eta(x) \,\|\, \sigma(D(x))\big)\big],$$

       and conclude that $D^\star$ is the unique maximizer of
       :eqref:`eq_gan_V`, up to changes on a set of $m$-probability zero.
    1. Apply the identity to the shifted critic $D^\star + b$ with a constant
       $b \neq 0$, and show that $V(D^\star + b) < V(D^\star)$.
    1. Expand the pointwise objective $p \log \sigma(t) + q \log \sigma(-t)$
       to second order around $t = \lambda(x)$, and show that

       $$V(D^\star + b) - V(D^\star) = -\frac{b^2}{2} \int \frac{p q}{p + q}\, dx + O(b^3).$$

       Show that the coefficient $\int pq/(p+q)\, dx$ lies between $0$ and
       $\tfrac12$. When does it equal $\tfrac12$, and when does it vanish?
       When the critic separates real from generated samples easily, how
       sharply do finite samples determine its additive constant?
   :exlabel:`ex_gan-log-loss-game-identifies`

1. **Unequal priors.** Suppose the labeling problem draws $y = 1$ with
   probability $\alpha$, where $0 < \alpha < 1$, so that the critic maximizes

   $$V_\alpha(D) = \alpha\, \mathbb{E}_{x \sim p}\big[\log \sigma(D(x))\big] + (1 - \alpha)\, \mathbb{E}_{x' \sim q}\big[\log \sigma(-D(x'))\big].$$

    1. Show that the optimal critic is
       $D^\star_\alpha = \lambda + \log \frac{\alpha}{1 - \alpha}$, so that
       $\sigma(D^\star_\alpha) = \alpha p / (\alpha p + (1 - \alpha) q)$.
    1. Let $m_\alpha = \alpha p + (1 - \alpha) q$ and
       $H_b(\alpha) = -\alpha \log \alpha - (1 - \alpha) \log(1 - \alpha)$.
       Show that

       $$\max_D V_\alpha(D) = \alpha\, D_{\mathrm{KL}}(p \,\|\, m_\alpha) + (1 - \alpha)\, D_{\mathrm{KL}}(q \,\|\, m_\alpha) - H_b(\alpha).$$

       Identify the divergence term as the mutual information between a
       sample and its label, as the section does for $\alpha = \tfrac12$, and
       conclude that it lies between $0$ and $H_b(\alpha)$. When is
       the upper end attained?
    1. `update_D` gives the real and the generated batch the same size.
       Suppose that the real batch had twice as many samples as the generated
       one, with the summed loss kept. Which $\alpha$ does this implement? How
       do the critic's best response and the non-saturating weight
       $w_{\textrm{ns}}$ of a generated sample change? Is $q = p$ still a
       stationary point of the generator update?
   :exlabel:`ex_gan-unequal-priors`

1. **Supports of measure zero.** The value calculation
   :eqref:`eq_gan_js_value` treats $p$ and $q$ as densities on the same
   space. Two settings close to this section's experiment violate that
   assumption.
    1. A generator with a one-dimensional latent variable,
       $G(z) = z v + b_q$ with $z \sim \mathcal{N}(0, 1)$ and fixed vectors
       $v \neq 0$ and $b_q$ in the plane, places all its samples on a line.
       Show that $\mathrm{JS}(p, q) = \log 2$ against the data distribution
       $p = \mathcal{N}(b, A^\top A)$ for every choice of $v$ and $b_q$, and
       describe the best response of an unrestricted critic.
    1. The training loop never sees $p$, only the empirical distribution
       $\hat p$ that places mass $1/1000$ on each point of `data`. Show that
       $\mathrm{JS}(\hat p, q) = \log 2$ for every generator distribution $q$
       with a density, $q = p$ included.
    1. By (a) and (b), the unrestricted game assigns the same value to every
       generator in either setting. Yet the run of
       :numref:`subsec_gan_training` trains on $\hat p$ and reduces
       $D_{\mathrm{KL}}(q \,\|\, p)$ from about eight nats to about a
       hundredth of a nat or less. Let $V_{\hat p}(D)$ denote :eqref:`eq_gan_V` with $p$
       replaced by $\hat p$, and restrict the critic to functions with
       $|D(x) - D(\tilde x)| \leq \kappa \|x - \tilde x\|$ for all $x, \tilde x$; the
       critic built by `make_nets` is such a function for some finite
       $\kappa$. Use the Kantorovich--Rubinstein duality
       $W_1(\hat p, p) = \sup_T \{\mathbb{E}_{\hat p}[T] - \mathbb{E}_p[T]\}$, the
       supremum taken over 1-Lipschitz functions $T$
       (:numref:`sec_mdl-optimal-transport`), to show that
       $|V_{\hat p}(D) - V(D)| \leq \kappa\, W_1(\hat p, p)$ for every such
       critic. Why does this bound make the restricted game on $\hat p$
       informative about $q$? What happens to it as $\kappa$ grows?
   :exlabel:`ex_gan-measure-zero-supports`

1. [code] **Critic error over the plane.** The comparison of
   :numref:`subsec_gan_verify_critic` checks the critic `net_D_q` against
   $\lambda$ only at samples from the mixture. Write
   $\eta = p / (p + q)$ for the posterior of :eqref:`eq_gan_dstar`, with
   $q$ the hand-set Gaussian `generator_gaussian(net_G_q)`.
    1. Evaluate $|\sigma(D(x)) - \eta(x)|$ on a regular grid over
       $[-3, 5] \times [-5, 9]$, using `gaussian_logpdf`, and plot it as a
       heat map with contours of the mixture density $m$ on top. Report the
       largest error on the grid and where it occurs.
    1. Draw 10,000 fresh samples from each of $p$ and $q$. Estimate the
       $m$-weighted mean error $\mathbb{E}_m\big[|\sigma(D) - \eta|\big]$, and estimate
       the excess $V(D^\star) - V(D)$ of the trained critic from the identity
       of problem :exref:`ex_gan-log-loss-game-identifies`a. Which of the
       three error measures does the training objective control? Use the
       identity to explain why the largest grid errors cost the critic
       almost nothing.
   :exlabel:`ex_gan-verification-above-compared-critic`

1. [code] **A dead critic.** ● Leaky ReLU units keep the critic's score
   surface sloped everywhere. Replace them by tanh units, and shrink the two
   hidden layers to 5 and 3 units, in `make_nets`.
    1. Rerun the non-saturating arm of the far-start comparison of
       :numref:`subsec_gan_saturation` with this critic for five seeds,
       varying the seed of `make_nets` together with the seed of the
       minibatches and latent draws. Record the final
       $D_{\mathrm{KL}}(q \,\|\, p)$ and the critic's loss in the last epoch.
    1. For a run that stalls far from the data, evaluate the second hidden
       layer's activations on 200 real and 200 generated samples, and the
       norm of $\nabla_x D$ at the same points. What do you find? Explain how
       Adam at the critic's learning rate of $0.05$, while the critic rejects
       a distant generator, can drive the units into this state, and why the
       generator then receives no direction at all.
    1. Show that a constant critic $D \equiv b$ has the per-sample loss
       $\tfrac12 \log(1 + e^{-b}) + \tfrac12 \log(1 + e^{b})$, minimized at
       $b = 0$ with the value $\log 2$. Why does a stalled run report a loss of
       $\log 2$ although $q$ is tens of nats from $p$?
    1. A critic loss of $\log 2$ has two readings: a best-response critic
       against $q = p$, or a critic class that cannot separate the samples.
       Which applies here, and which measurement of this section tells the
       two apart?
   :exlabel:`ex_gan-dead-critic`

1. [code] **Two time scales.** ● `train` gives the critic ten times the
   generator's learning rate, `lr_D=0.05` against `lr_G=0.005`, so that the
   critic stays close to its best response.
    1. Rerun the twenty-epoch training of :numref:`subsec_gan_training` with
       (`lr_D`, `lr_G`) set to $(0.05, 0.005)$, $(0.005, 0.005)$, and
       $(0.005, 0.05)$, five seeds each, keeping the initialization of
       `make_nets` and varying the seed of the minibatches and latent draws.
       Report the median and the range of the final
       $D_{\mathrm{KL}}(q \,\|\, p)$, counting a run without a finite value as
       a failure. Is the spread across seeds small compared with the
       differences between the settings?
    1. :eqref:`eq_gan_js_value` and :eqref:`eq_gan_ns_value` describe the
       generator's objective against a best-response critic. What does the
       generator follow instead when it adapts faster than the critic? The
       convergence theorem for the two time-scale update rule of
       :citet:`Heusel.Ramsauer.Unterthiner.ea.2017` (Theorem 1) assumes,
       among other conditions, generator step sizes $a(n)$ and critic step
       sizes $b(n)$ with $\sum_n a(n) = \sum_n b(n) = \infty$,
       $\sum_n a(n)^2 < \infty$, $\sum_n b(n)^2 < \infty$, and
       $a(n) = o(b(n))$. Which of these do the constant learning rates of
       `train` violate? Does a ratio of ten suffice in your runs?
   :exlabel:`ex_gan-two-time-scales`

1. [code] **Classifier two-sample test.** ● A classifier two-sample test
   :cite:`Lopez-Paz.Oquab.2017` trains a classifier to separate fresh real
   samples from fresh generated ones and asks whether its held-out accuracy
   exceeds chance. Apply it to the trained generator `net_G` of
   :numref:`subsec_gan_fit`, with $\eta = p / (p + q)$ as before.
    1. For balanced classes, the Bayes classifier labels $x$ as real where
       $\eta(x) > \tfrac12$. Show that its accuracy is

       $$\mathbb{E}_{x \sim m}\big[\max(\eta(x), 1 - \eta(x))\big] = \tfrac12 \big(1 + \mathrm{TV}(p, q)\big)$$

       with the total variation distance
       $\mathrm{TV}(p, q) = \tfrac12 \int |p - q|$
       (:numref:`sec_mdl-tv-pinsker`). Use Pinsker's inequality,
       $\mathrm{TV}(p, q) \leq \sqrt{D_{\mathrm{KL}}(q \,\|\, p)/2}$, and the printed
       $D_{\mathrm{KL}}(q \,\|\, p)$ to bound the accuracy that any
       classifier can reach against `net_G`.
    1. Estimate the Bayes accuracy by Monte Carlo with `gaussian_logpdf`
       and `generator_gaussian`. Then train the fresh critic
       `make_nets()[1]` with `update_D` against the frozen `net_G` on fresh
       real samples $z A + b$, and measure its accuracy on 2,000 held-out
       samples from each distribution. How close does it come?
    1. Under $q = p$, the held-out accuracy on $n_{\textrm{te}}$ test points
       is approximately Gaussian with mean $\tfrac12$ and variance
       $1/(4 n_{\textrm{te}})$. Find the smallest $n_{\textrm{te}}$ at which
       an observed accuracy equal to the Bayes accuracy rejects $q = p$ in a
       one-sided test at the 5 percent level. What does a test with fewer
       points say about the generator?
   :exlabel:`ex_gan-classifier-two-sample-test`

1. **Generator weights and their divergences.** ●● A generator loss
   $\mathbb{E}_z[g(D(G(z)))]$, differentiated with the critic held fixed, has the
   gradient $-\mathbb{E}_z\big[w(D(x'))\, \nabla_\theta D(G(z))\big]$ with the weight
   $w = -g'$; :eqref:`eq_gan_weights` lists the two weights of this
   section. Throughout, the critic is the best response $D = \lambda$ to the
   current generator $q_{\theta_0}$ and is held fixed while the loss is
   differentiated at $\theta_0$. Write $\rho = e^{D} = p / q_{\theta_0}$.
    1. Show that the sum of the saturating loss $\mathbb{E}_z[\log \sigma(-D(G(z)))]$
       and the non-saturating loss $-\mathbb{E}_z[\log \sigma(D(G(z)))]$ equals
       $D_{\mathrm{KL}}(q \,\|\, p)$ at $D = \lambda$, and that its weight is
       $w = 1$.
    1. Let $D_f(p \,\|\, q_\theta) = \mathbb{E}_{x \sim q_\theta}\big[f\big(p(x)/q_\theta(x)\big)\big]$
       be the $f$-divergence of :numref:`sec_mdl-f-divergences` with a twice
       differentiable convex function $f$; the subscript distinguishes it
       from the critic $D$. Show that

       $$\nabla_\theta D_f(p \,\|\, q_\theta)\big|_{\theta_0} = -\mathbb{E}_z\big[\rho^2 f''(\rho)\, \nabla_\theta D(G(z))\big],$$

       with $\rho$ evaluated at $x' = G(z)$. Two steps suffice: the product
       rule gives $\int \nabla_\theta q_\theta(x)\, h(x)\, dx$ with
       $h = f(\rho) - \rho f'(\rho)$, and for a fixed $h$ this integral equals
       $\nabla_\theta \mathbb{E}_z[h(G(z))]$. A positive weight $w$ therefore selects
       the divergence whose function $f$ satisfies
       $\rho^2 f''(\rho) = w(\log \rho)$.
    1. Identify the divergences selected by $w_{\textrm{sat}}$, by $w = 1$,
       by $w(t) = e^t$, and by $w_{\textrm{ns}}$. Check the
       $w_{\textrm{sat}}$ case against :eqref:`eq_gan_js_value`: why may the
       critic be held fixed when the maximum over $D$ is differentiated? The
       weight $e^t$ belongs to a generator loss of :citet:`Goodfellow.2016`
       (Section 7.3), and the $w_{\textrm{ns}}$ case is Theorem 2.5 of
       :citet:`Arjovsky.Bottou.2017`.
    1. :citet:`Arjovsky.Bottou.2017` read the negative Jensen--Shannon term in
       the divergence of $w_{\textrm{ns}}$ as pushing the two distributions
       apart. Show that this divergence equals $2\, D_{\mathrm{KL}}(m \,\|\, p)$
       with $m = \tfrac12(p + q)$, and that the function $f$ of this
       divergence is strictly convex with $f(1) = 0$. What does this imply for the fixed points of
       non-saturating training against a best-response critic, and for that
       reading?

   *Adapted from Stefano Ermon and Aditya Grover,
   [CS236: Deep Generative Models](https://cs236.stanford.edu), Homework 3,
   Problem 2.3.*
   :exlabel:`ex_gan-generator-weight-divergences`

1. [extended] **Finite samples and critic capacity.** ●● Problem
   :exref:`ex_gan-measure-zero-supports`b showed that an unrestricted critic
   separates the empirical distribution of the training points from every
   generator with a density. :citet:`Arora.Ge.Liang.ea.2017` (Theorem 3.1)
   show that a critic class with few parameters behaves differently: once the
   number of samples exceeds the number of critic parameters, up to
   logarithmic factors and the required accuracy, the critic's value on the
   samples is close to its value on the populations. Test this at small
   scale with a generator set to compute $z A + b$, so that $q = p$ exactly.
    1. Train critics with two hidden layers of equal width, for the widths
       4, 16, 128 and 1024 and for $n \in \{100, 1000, 10000\}$ training
       points drawn from $p$, using `update_D` with fresh generated samples,
       Adam at learning rate $10^{-3}$ and 4,000 steps of 256 samples per
       class. Plot the gap between the per-sample loss on the $n$ training
       points and on fresh draws from $p$ against the number of critic
       parameters, one curve per $n$.
    1. At which ratio of critic parameters to $n$ does the training loss fall
       visibly below $\log 2$? Where does the critic of `make_nets`, with 337
       parameters, lie at $n = 1000$?
    1. A GAN computes its critic loss on minibatches of the training data.
       For a critic large enough to fall below $\log 2$ at $q = p$, what can a
       critic loss below $\log 2$ during training no longer tell apart?
       Propose a measurement that separates the two causes, and state what it
       costs.
   :exlabel:`ex_gan-finite-sample-critic`

[Discussions](https://d2l.discourse.group/t/36808)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.1]{.kicker}

Generative adversarial networks<br>
**a sampler without a density · the optimal critic · the value is Jensen--Shannon · two generator weights · a Gaussian check**
:::
:::

::: {.slide title="A Model within 4.6 Nats of the Best Can Emit 99% Noise"}
[The problem]{.kicker}

- An implicit generator $x' = G(z)$, $z \sim \mathcal{N}(0, I)$: one forward
  pass per sample, but no density we can evaluate.
- Likelihood would be an unsafe yardstick anyway:
  $\tilde p = 0.01\, p_{\textrm{data}} + 0.99\, p_{\textrm{noise}}$ has
  $\log \tilde p \geq \log p_{\textrm{data}} - \log 100$.
- Within 4.6 nats of the best model, against image log-likelihoods in the
  thousands of nats; yet 99% of its samples are noise.

::: {.d2l-note .warn}
Can a classifier that tells samples from data take the place of the
likelihood?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive the optimal critic and read it as a log density ratio;
- compute the value of the game as a Jensen--Shannon divergence;
- explain why separated supports give the generator no gradient;
- compare the saturating and non-saturating generator weights;
- read a GAN loss curve, and say what a loss of $\log 2$ does not prove.

**The idea:** train a classifier to tell real from generated, then move the
generator until the classifier fails. **Running example:** a 2-D Gaussian
whose densities we know.
:::

::: {.slide title="The First Deck Builds the Game the Others Modify"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| deck | adds |
|---|---|
| **Generative Adversarial Networks** | **the log-loss game and its value** |
| Adversarial Objectives | other losses and critic classes |
| Relativistic Objectives | a critic that ranks pairs |
| Gradient Penalties and Convergence | convergence of the training dynamics |
| Adversarial Image Generation | convolutional networks, scores |
| Conditional Generation | asking for a class |
| Adversarial Losses Beyond GANs | where the comparison survives |
:::

::: {.col .narrow}
- **Uses:** logistic regression; KL and JS divergences; the domain
  classifier for covariate shift.
- **Skips:** other losses (next deck); image networks.
:::
:::
:::

::: {.slide title="A Classifier between Real and Generated Samples Trains the Generator"}
![](../img/mdl-gan-architecture.svg){width=70%}

- Label $y \sim \textrm{Bernoulli}(\tfrac12)$; $x \sim p$ (data) if $y = 1$,
  $x \sim q$ (generator) if $y = 0$.
- The critic's logit $D(x)$; $\sigma(D(x))$ estimates $P(y = 1 \mid x)$.
:::

::: {.slide title="The Critic Maximizes the Log-Likelihood of the Labels"}
$$V(D) = \mathbb{E}_{x \sim p}[\log \sigma(D(x))] + \mathbb{E}_{x' \sim q}[\log \sigma(-D(x'))]$$

- The critic maximizes $V$: it fits the best classifier of real against
  generated.
- The generator enters only the second term and minimizes $V$:
  $\min_G \max_D V$.
- $\max_D V$ measures how well the two distributions can be told apart.
:::

::: {.slide title="The Optimal Critic Is the Log Density Ratio"}
At each $x$, maximize $p \log u + q \log(1 - u)$ over $u = \sigma(D(x))$:

$$\sigma(D^\star) = \frac{p}{p+q},
\qquad
D^\star = \log\frac{p}{q} = \lambda$$

. . .

- The trained critic is a **density-ratio estimator**: it recovers the one
  function of $q$ the game needs, though $q$ has no formula.
- The constant is fixed too: shifting $D^\star$ by $b \neq 0$ lowers $V$, so
  $D > 0$ means "more likely real".
:::

::: {.slide title="Check: Two Unit Gaussians"}
[Check]{.kicker}

$p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(1, 1)$. What is the optimal critic
$D^\star(x)$, and which model class contains it?

. . .

- $\lambda(x) = -\tfrac{x^2}{2} + \tfrac{(x-1)^2}{2} = \tfrac12 - x$.
- Linear in $x$: logistic regression already contains the optimal critic.
- In general $\lambda = \mu(\mu/2 - x)$ for $q = \mathcal{N}(\mu, 1)$.
:::

::: {.slide title="At the Optimal Critic the Game Is Worth 2 JS − 2 log 2"}
With $m = \tfrac12(p+q)$, $\sigma(D^\star) = p/(2m)$, $\sigma(-D^\star) = q/(2m)$:

$$V(D^\star) = \int p \log\frac{p}{2m} + \int q \log\frac{q}{2m}
= D_{\mathrm{KL}}(p\|m) + D_{\mathrm{KL}}(q\|m) - 2\log 2$$

$$\max_D V = 2\,\mathrm{JS}(p, q) - 2\log 2$$

- Against the best critic the generator minimizes Jensen--Shannon, which is
  zero only at $q = p$.
- The best critic's per-sample loss is $\log 2 - \mathrm{JS}$.
:::

::: {.slide title="JS Is the Information One Sample Carries about Its Source"}
$$\mathrm{JS}(p, q) = H(m) - \tfrac12\big(H(p) + H(q)\big) = I(x; y) \leq \log 2$$

| $q = \mathcal{N}(\mu, 1)$ vs. $p = \mathcal{N}(0, 1)$ | $\mu = 1$ | $2$ | $4$ | $8$ |
|---|---|---|---|---|
| $\mathrm{JS}$ (nats) | 0.11 | 0.34 | 0.63 | 0.6931 |

- One binary label holds at most $\log 2$ nats.
- The ceiling is reached on **disjoint supports**, where each sample names its
  source.
:::

::: {.slide title="Separated Supports Freeze the Value at log 2"}
Point masses on the line: $p = \delta_0$, $q_\theta = \delta_\theta$.

- For every $\theta \neq 0$: disjoint supports, $\mathrm{JS} = \log 2$.
- The value does not depend on $\theta$: no gradient, even at
  $\theta = 0.001$.
- In high dimensions a generator from a $d_z$-dimensional latent fills a
  $d_z$-dimensional surface: supports are nearly disjoint.

::: {.d2l-note}
An optimal critic classifies perfectly, and the generator learns nothing
from the value of the game.
:::
:::

::: {.slide title="Both Generator Losses Push Along ∇D, with Different Weights"}
::: {.cols .vc}
::: {.col}
$$w_{\textrm{sat}} = \sigma(D(x')), \qquad w_{\textrm{ns}} = \sigma(-D(x'))$$

- Saturating: minimize $\mathbb{E}[\log \sigma(-D(x'))]$, the game as
  written.
- Non-saturating: minimize $-\mathbb{E}[\log \sigma(D(x'))]$.
- Same direction $\nabla_\theta D$; confidently rejected samples get almost
  no weight under $w_{\textrm{sat}}$.
:::

::: {.col}
![](../img/mdl-gan-weights.svg)
:::
:::
:::

::: {.slide title="Check: A Confidently Rejected Sample"}
[Check]{.kicker}

A generated sample scores $D(x') = -5$. How much more weight does it get
under the non-saturating loss?

. . .

- $w_{\textrm{ns}} / w_{\textrm{sat}} = \sigma(5) / \sigma(-5) = e^{5}
  \approx 148$.
- At $D = -3$: $e^3 \approx 20$.
:::

::: {.slide title="The Non-Saturating Loss Keeps the Fixed Point q = p"}
At the optimal critic:

$$-\mathbb{E}_{q}[\log \sigma(\lambda)] = \log 2 + D_{\mathrm{KL}}(q\|p) - D_{\mathrm{KL}}(q\|m)
\;\geq\; \log 2 + \tfrac12 D_{\mathrm{KL}}(q\|p)$$

- Equality, $\log 2$, only at $q = p$.
- With the critic held fixed at $\lambda$, the update descends
  $D_{\mathrm{KL}}(q\|p) - 2\,\mathrm{JS}(p, q)$: a strictly convex
  $f$-divergence, minimized only at $q = p$.
:::

::: {.slide title="Reverse KL Barely Charges a Missing Mode"}
$p$: two separated modes, equal weight. $q$: one of them.

- $D_{\mathrm{KL}}(q\|p) = \log 2 \approx 0.69$ nats; $\mathrm{JS} \approx
  0.22$; $D_{\mathrm{KL}}(p\|q) = \infty$.
- At the best critic the score on the covered mode is the constant
  $-\log 2$: $\nabla D = 0$, no push toward the other mode.
- Both weights read a sample's own score against the zero level; nothing
  compares it with real samples.

**Mode dropping:** the generator covers only some of the data's modes.
:::

::: {.slide title="A Linear Generator on Gaussian Data Gives Closed Forms"}
::: {.cols .vc}
::: {.col}
- Data $x = zA + b$: $p = \mathcal{N}(b, A^\top A)$, a thin tilted ellipse.
- Generator: one linear layer, so $q$ is Gaussian and $\lambda$,
  $D_{\mathrm{KL}}(q\|p)$ are exact.
- Critic: 16-16 leaky ReLU; Adam with $\beta_1 = 0.5$; critic rate ten times
  the generator's.
:::

::: {.col .narrow}
@!gan-data-and-models-1
:::
:::
:::

::: {.slide title="The Critic's Loss Rises to log 2 as the Fit Improves"}
@!gan-training-3

- Critic loss $= \log 2 - \mathrm{JS}$ at the best response: rising toward
  $\log 2$ means $\mathrm{JS} \to 0$.
- The generated cloud fills the same ellipse.
:::

::: {.slide title="KL(q‖p) Falls from About 8 Nats to Well below 1"}
@!gan-quantifying-the-fit

- The loop never evaluates $q$'s density; the critic alone guides the fit.
- Measured in closed form from the generator's weights.
:::

::: {.slide title="A Trained Critic Tracks log(p/q) on a Hand-Set q"}
::: {.cols .vc}
::: {.col}
Freeze $q$ at $D_{\mathrm{KL}}(q\|p) \approx 3.4$ nats; train a fresh critic
for 2,000 steps; compare with the analytic $\lambda$.

- Correlation above 0.97; the critic explains over 90% of the variance of
  $\lambda$ ($R^2$; a constant critic scores 0).
- Errors of one to a few nats on the tenth with the most extreme $\lambda$,
  a few tenths or less elsewhere: the tails, where samples are rare.
:::

::: {.col .narrow}
@!gan-the-learned-density-ratio-1
:::
:::
:::

::: {.slide title="From a Far Start Only the Non-Saturating Generator Moves"}
@!gan-saturating-and-non-saturating-losses-compared-1

- Saturating: critic loss $\to 0$, $w_{\textrm{sat}} \approx 0$; stays more
  than 300 nats away.
- Non-saturating: reaches the data, well below 1 nat.
:::

::: {.slide title="Check: Does a Loss of log 2 Mean q = p?"}
[Check]{.kicker}

The critic's loss reads $\log 2$ at the end of a run. Has the generator
matched the data?

. . .

- Not necessarily: a critic that outputs a constant also scores $\log 2$.
- A tanh critic whose units saturate does exactly this while $q$ is tens of
  nats from $p$.
- Check the fit independently: here $D_{\mathrm{KL}}(q\|p)$ in closed form.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Optimal critic: $D^\star = \log(p/q)$; value $2\,\mathrm{JS} - 2\log 2$;
  $\mathrm{JS} = I(x; y) \leq \log 2$.
- Disjoint supports: value fixed at $\log 2$, no gradient.
- Weights $\sigma(D)$ and $\sigma(-D)$: same fixed point; a ratio of 148 at
  $D = -5$.
- Mode dropping costs only $\log 2$ nats of reverse KL in the two-mode
  example.
- Gaussian test: KL from about 8 nats to well below 1; the critic explains
  over 90% of $\lambda$'s variance; far start, only non-saturating moves.

**Next:** other losses and critic classes, and which keep a gradient on
separated supports.
:::
