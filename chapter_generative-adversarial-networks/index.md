# Generative Adversarial Networks
:label:`chap_gans`

Suppose we draw a 100-dimensional random vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and pass it through a neural network $G(\mathbf{z})$ to generate an image. Writing down such a generator network takes only a few lines of code; training it is the challenge. Unlike classifiers or autoregressive language models—which maximize the log-likelihood of observed labels or tokens—a direct generator defines an implicit distribution over images without an explicit density $q(\mathbf{x})$ to maximize. How can we train a generative model when we can sample from it, but cannot evaluate its likelihood?

We have already seen how a classifier can compare two distributions from samples alone. When correcting for covariate shift in :numref:`subsec_covariate-shift-correction`, we trained a binary classifier to distinguish training inputs from test inputs, and its optimal logit recovered the log density ratio between the two distributions. :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` turned this two-sample comparison into a generative training principle: in a *generative adversarial network* (GAN), a *discriminator* (or *critic*) $D$ is trained to distinguish real data from generated samples, while the *generator* $G$ is trained to produce samples that the discriminator cannot tell apart from real data. Through the discriminator's gradients, the generator learns where its samples are over- or under-represented relative to the data distribution, without either density ever being written down explicitly.

Although diffusion and flow-matching models (:numref:`chap_diffusion`) have largely superseded standalone GANs for general-purpose image synthesis, adversarial objectives remain a core component of modern generative systems—training the latent autoencoders behind latent diffusion models, neural audio vocoders, and one-step generators distilled from multi-step diffusion teachers.

This chapter is organized in three parts:
1. **Objectives and Divergences (:numref:`sec_basic_gan`–:numref:`sec_gan_relativistic`):** :numref:`sec_basic_gan` derives the optimal discriminator and the Jensen–Shannon divergence for the classic logistic objective, and explains why the non-saturating generator loss is needed when generated samples lie far from the data. :numref:`sec_gan_objectives` generalizes this analysis to $f$-divergences (least-squares, hinge) and integral probability metrics (Maximum Mean Discrepancy and Wasserstein-1 distance). :numref:`sec_gan_relativistic` analyzes relativistic paired objectives (RpGAN) that train the critic to rank real–generated pairs via a Bradley–Terry model.
2. **Training Dynamics and Gradient Penalties (:numref:`sec_gan_convergence`):** Minimax gradient training does not automatically converge to an equilibrium; even on a two-parameter bilinear problem (the Dirac-GAN), unregularized gradient updates rotate or spiral outward. :numref:`sec_gan_convergence` shows how zero-centered input-gradient penalties ($R_1$ and $R_2$) damp these oscillations, completing the modernized R3GAN training recipe :cite:`Huang.Gokaslan.Kuleshov.ea.2024` used in the rest of the chapter.
3. **Architectures, Conditioning, and Modern Applications (:numref:`sec_dcgan`–:numref:`sec_gan_beyond`):** :numref:`sec_dcgan` builds convolutional image generators (both the classic 2015 DCGAN and a normalization-free backbone) and introduces quantitative evaluation metrics—Fréchet Distance (FID), Kernel Inception Distance (KID), and $k$-NN precision and recall. :numref:`sec_gan_conditional` extends training and evaluation to class-conditional generation on CIFAR-10 using feature modulation and projection discriminators. Finally, :numref:`sec_gan_beyond` explains why regression losses ($L_2$ and $L_1$) blur ambiguous targets toward conditional averages, and how adversarial losses prevent this blurring in diffusion distillation, image tokenizers, and speech vocoders.

```toc
:maxdepth: 2

gan
objectives
relativistic
convergence
dcgan
conditional
adversarial-losses
further-reading
```

## Notation {.unnumbered}

:numref:`tab_gan_notation` summarizes the notation used across this chapter. Divergences and entropies are measured in nats and follow :numref:`chap_mdl-information-theory`: $D_{\textrm{KL}}$ denotes the Kullback–Leibler divergence, $\mathrm{JS}$ the Jensen–Shannon divergence, $H$ differential or discrete entropy, and $H_b(t)$ the binary entropy of a Bernoulli distribution with parameter $t$.

:Notation used in this chapter.
:label:`tab_gan_notation`

| symbol | meaning | first used | common alternative |
|:--|:--|:--|:--|
| $p$, $q$, $m = \tfrac12(p + q)$ | data, generator, and balanced mixture densities | :numref:`sec_basic_gan` | $p_{\textrm{data}}$, $p_g$ |
| $x \sim p$, $x' \sim q$; $y \in \{0, 1\}$ | real sample, generated sample, and source label ($1$ = real) | :numref:`sec_basic_gan` | |
| $z \in \mathbb{R}^{d_z}$, $G$ | latent vector and generator network ($x' = G(z)$) | :numref:`sec_basic_gan` | |
| $D$; $V(D)$ | critic/discriminator realness logit; minimax value function | :numref:`sec_basic_gan` | |
| $\rho = p/q$, $\lambda = \log(p/q)$ | density ratio and log density ratio | :numref:`sec_basic_gan` | |
| $w_{\textrm{sat}}$, $w_{\textrm{ns}}$, $w_{\textrm{Rp}}$ | per-sample generator gradient weights (saturating, non-saturating, relativistic) | :numref:`sec_basic_gan`, :numref:`sec_gan_relativistic` | |
| $\eta = p/(p + q)$ | Bayes class posterior $P(y = 1 \mid x)$ | :numref:`sec_gan_objectives` | |
| $\ell$, $L(\eta)$, $\Delta_\ell(p, q)$ | margin payoff, conditional Bayes risk, and Bayes-risk gap | :numref:`sec_gan_objectives` | |
| $D_f(p \,\|\, q)$; $k(x, x')$ | $f$-divergence with convex function $f$; positive-definite kernel | :numref:`sec_gan_objectives` | |
| $\Phi(D)$, $d_{\mathrm{Rp}}(p, q)$ | relativistic pairing objective and pairing divergence | :numref:`sec_gan_relativistic` | |
| $R_1$, $R_2$; $\gamma$ | zero-centered gradient penalties at real and generated samples; penalty weight | :numref:`sec_gan_convergence` | |
| $c$, $e_c$, $\phi(x)$ | conditioning variable, condition embedding, and feature representation | :numref:`sec_dcgan`, :numref:`sec_gan_conditional` | |

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16]{.kicker}

Generative adversarial networks<br>
**a learned comparison in place of the likelihood · what the game measures · whether training converges · images, requests, and what survives**
:::
:::

::: {.slide title="A Hundred Random Numbers Become an Image; Nothing Gives Its Probability"}
[The problem]{.kicker}

::: {.cols .vc}
::: {.col}
@!gan-garments-from-noise-3

- $x' = G(z)$, $z$ a vector of 100 Gaussian numbers: one forward pass
  per image.
- Garments: a perceptron trained on Fashion-MNIST. Sprites (right): a
  convolutional generator.
- A hundred numbers in, 784 pixels out: the garments lie on a thin surface
  and have no density at all.
:::

::: {.col .narrow}
@!dcgan-training-with-the-classic-loss-3
:::
:::

::: {.d2l-note .warn}
Nothing computes the probability of $x'$. Can a learned comparison of
samples with data take the place of the likelihood?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- read a critic's logit as a log density ratio, and the game's value as a
  divergence;
- say which losses and critics keep a gradient far from the data;
- explain why gradient training circles the equilibrium, and damp it with
  a gradient penalty;
- train an image generator, score its quality and its coverage, and
  condition it on a class;
- say why an adversarial term outlived the GAN.

**The idea:** train a classifier to tell real from generated, then move the
generator until the classifier fails.
:::

::: {.slide title="A Learned Comparison Replaces the Likelihood"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| part or chapter | the model learns from |
|---|---|
| Basics | the likelihood of observed data |
| Reinforcement Learning | a reward, in post-training fitted to pairwise preferences |
| **Generative Adversarial Networks** | **a critic that compares samples with data** |
| Diffusion Models | regression against closed-form targets |
:::

::: {.col .narrow}
- **Uses:** logistic regression, the covariate-shift classifier, KL
  divergence; the Bradley--Terry model for pairs.
- **Skips:** a survey of architectures; applications.
- **Survives in:** one-step distilled generators, image autoencoders,
  speech vocoders.
:::
:::
:::

::: {.slide title="A Model within 4.6 Nats of the Best Can Emit 99% Noise"}
[Why not likelihood anyway]{.kicker}

- Mix the data density $p$ with noise:
  $\tilde p = 0.01\, p + 0.99\, p_{\textrm{noise}}$.
- Then $\log \tilde p \geq \log p - \log 100$: within 4.6 nats of the
  best model.
- Image log-likelihoods run to thousands of nats, so 4.6 nats is a rounding
  error; yet 99% of the samples are noise.

::: {.d2l-note}
Judging a sampler requires looking at its samples.
:::
:::

::: {.slide title="A Classifier's Logit Is the Log Density Ratio"}
::: {.cols .vc}
::: {.col .narrow}
![](../img/mdl-gan-architecture.svg){width=100%}
:::

::: {.col}
- Label $y = 1$ for a data sample $x \sim p$, $y = 0$ for a generated
  sample $x' \sim q$, half of each.
- The best log-loss classifier has
  $\sigma(D^\star(x)) = \dfrac{p(x)}{p(x) + q(x)}$, so
  $D^\star(x) = \log \dfrac{p(x)}{q(x)}$.
- The generator gets its gradient through $D$.
:::
:::

::: {.d2l-note}
Neither density is written down, yet the ratio tells the generator where
its samples are too dense and where too sparse.
:::
:::

::: {.slide title="Check: Read a Critic's Output"}
[Check]{.kicker}

An optimal critic returns $D(x) = 1.2$ at a point $x$, and
$D(x) = -2$ at another point.

1. What is $p(x)/q(x)$ at each point?
2. Which point should the generator place fewer samples near?

::: {.fragment .fade-up}
1. $e^{1.2} \approx 3.3$: the data is denser there; $e^{-2} \approx 0.14$:
   the generator is about seven times too dense.
2. The second point: $q$ exceeds $p$ there.
:::
:::

::: {.slide title="The Value Barely Changes Once the Samples Miss the Data"}
- The game's value, against the best critic, is a Jensen--Shannon
  divergence: $\max_D V = 2\,\mathrm{JS}(p, q) - 2 \log 2$.
- $p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\mu, 1)$: JS is 0.11, 0.34 and
  0.63 nats at $\mu = 1, 2, 4$.
- At $\mu = 8$ it equals its ceiling $\log 2 \approx 0.6931$ to four digits.
- Its slope at $\mu = 7$ is more than a hundred times smaller than at
  $\mu = 1$: a generator that descends this value barely moves.

::: {.d2l-note .warn}
Which losses and which critics keep the generator moving?
:::
:::

::: {.slide title="Three Choices Describe Every Game of the Chapter"}
| choice | options |
|---|---|
| what the critic estimates | the log ratio $\log(p/q)$, or the difference $p - q$, smoothed over a neighborhood |
| how it scores | one sample at a time, a real sample against a generated one, or the average score of each set |
| how it is held in check | not at all, a cap on how fast its score may change, a penalty on its gradient |

::: {.d2l-note}
The first section plays the simplest option of each choice; the next three
sections change them.
:::
:::

::: {.slide title="The Chapter Moves from What the Game Measures to What It Builds"}
[Chapter map]{.kicker}

| movement | sections | question |
|---|---|---|
| what the game measures | Generative Adversarial Networks; Adversarial Objectives and Divergences; Relativistic Objectives | what does each loss and critic measure, and does it keep a gradient far from the data? |
| whether training converges | Gradient Penalties and Convergence | do gradient steps reach the equilibrium, and what makes them settle? |
| what it builds | Adversarial Image Generation; Conditional Generation; Adversarial Losses Beyond GANs | images and their scores, requests, and where the critic survives |

::: {.d2l-note}
In a hurry to build: the first section, Gradient Penalties and Convergence,
then the images.
:::
:::

::: {.slide title="Garments First, Then Densities We Know, Then Images We Score"}
[Running examples]{.kicker}

- **Fashion-MNIST, 60,000 garments at $28 \times 28$:** the first image
  generator, two perceptrons.
- **Planar Gaussians and mixtures:** both densities known, so the critic is
  checked against $\log(p/q)$ and the fit is measured in nats.
- **25 Gaussians on a $5 \times 5$ grid:** do the samples cover every mode?
- **40,597 game sprites at $64 \times 64$:** DCGAN (a convolutional
  network from 2015) and three losses on one backbone, scored by Fréchet
  distance and $\mathrm{MMD}^2$ in features of our own, with precision and
  recall for coverage.
- **CIFAR-10 at $32 \times 32$, ten classes:** ask for a bird; a classifier
  checks the request.
:::

::: {.slide title="One Recipe Serves as Reference, and Our Runs Check Its Parts"}
[What we adopt]{.kicker}

- **Recipe (R3GAN, 2024):** a critic that ranks a real against a generated
  sample; penalties $R_1$ (at real samples) and $R_2$ (at generated ones)
  keep its gradient small.
- **Why:** few parts, each with an analysis we can carry out.
- **Its evidence:** its authors' runs beat StyleGAN2 on faces, ImageNet and
  CIFAR-10, with compute far beyond ours.
- **Shown here:** on small problems, each part does what its analysis
  predicts.
- **Not shown:** that it wins. Ranking pairs does not visibly help coverage
  in our runs, and on sprites the 2015 DCGAN scores at least as well.

::: {.d2l-note}
Cost: 20 to 40 minutes per image notebook on one GPU; the penalty adds a
second backward pass to every critic step.
:::
:::

::: {.slide title="Check: Which Section Answers Each Symptom?"}
[Check]{.kicker}

1. The generated points sit far from the data, and the generator's gradient
   is almost zero.
2. Both losses hover near their equilibrium values while the generated
   points drift back and forth.
3. Every request for a bird returns the same bird.

::: {.fragment .fade-up}
1. Adversarial Objectives and Divergences: which games keep a gradient
   when the samples and the data do not overlap.
2. Gradient Penalties and Convergence: alternating gradient steps circle
   the equilibrium instead of settling; a penalty damps them.
3. Conditional Generation: a generator can obey every request with one
   image per class.
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- A generator turns 100 random numbers into a garment or a sprite but gives
  no density.
- A sampler can be within 4.6 nats of the best likelihood and emit 99% noise.
- A classifier of real against generated samples estimates $\log(p/q)$
  without either density.
- Three movements: what the game measures, whether training converges,
  what it builds.
- One 2024 recipe, R3GAN, is our reference; our runs check its parts on
  small problems.

**Next:** the log-loss game, solved exactly, then trained on a Gaussian and
on garments.
:::
