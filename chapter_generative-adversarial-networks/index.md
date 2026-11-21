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

Generative Adversarial Networks<br>**implicit density estimation · minimax two-sample games · $f$-divergences & IPMs · rotational dynamics & gradient penalties · conditional generation & perceptual critics**
:::
:::

::: {.slide title="Training Implicit Generators Without a Likelihood"}
[Central Problem]{.kicker}

An **implicit generator** draws a latent vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_z})$ ($d_z = 100$) and produces an image $\mathbf{x}' = G(\mathbf{z}) \in \mathbb{R}^{d_x}$ in a single forward pass—defining a sampling procedure without a tractable density $q(\mathbf{x})$ to maximize:

::: {.cols .vc}
::: {.col}
- **Low-dimensional manifold support:** Mapping $100$ latent coordinates into $784$ (Fashion-MNIST) or $12{,}288$ ($64 \times 64$ RGB) pixels concentrates generated samples on a thin surface with no Lebesgue density.
- **Why likelihood misleads (Theis et al., 2016):** Mixing the true density with $99\%$ noise, $\tilde{p} = 0.01\,p_{\textrm{data}} + 0.99\,p_{\textrm{noise}}$, satisfies $\log \tilde{p}(\mathbf{x}) \ge \log p_{\textrm{data}}(\mathbf{x}) - \log 100$—within $4.6$ nats of the optimal model (out of thousands of nats on images) while emitting $99\%$ pure noise!
- **Core idea:** Replace likelihood evaluation with a learned **two-sample critic** $D$ that inspects generated samples directly.
:::

::: {.col .fig}
![](../img/mdl-gan-dcgan.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Two-Sample Classification as Density-Ratio Estimation"}
[The Core Mechanism]{.kicker}

Goodfellow et al. (2014) turned two-sample binary classification into a generative training principle by pitting a **generator** $G$ against a **discriminator (critic)** $D$:

::: {.cols .vc}
::: {.col}
- **Balanced two-sample game:** Label real draws $x \sim p$ with $y = 1$ and generated draws $x' = G(z) \sim q$ with $y = 0$, each with prior probability $\frac{1}{2}$.
- **Optimal critic recovers the log density ratio:** Minimizing binary log loss yields the Bayes posterior $\sigma(D^\star(x)) = \frac{p(x)}{p(x) + q(x)}$, so the optimal logit is:
  $$D^\star(x) = \log \frac{p(x)}{q(x)} = \lambda(x)$$
- **Gradients without densities:** Backpropagating through $D$ pushes generated samples $x' = G(z)$ uphill along $\nabla_x D(x')$, moving probability mass from over-represented regions ($q > p \implies D^\star < 0$) toward under-represented regions ($p > q \implies D^\star > 0$).
:::

::: {.col .fig}
![](../img/mdl-gan-architecture.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Three Design Axes of Every Adversarial Game"}
[Unifying Framework]{.kicker}

When generated samples miss the data support, the classic minimax value $\max_D V(D) = 2\,\mathrm{JS}(p, q) - 2\log 2$ saturates at $\mathrm{JS}(p, q) = \log 2 \approx 0.6931$ nats with vanishing slope. Every adversarial formulation in this chapter modifies one of three design axes:

| Design Axis | Classical Minimax Choice (§16.1) | Generalized Alternatives (§16.2–16.4) |
| :--- | :--- | :--- |
| **1. What the critic estimates** | Pointwise log density ratio $\lambda(x) = \log(p(x)/q(x))$ | Monotone link $D^\star(\lambda)$ ($f$-divergences: LSGAN, hinge) or smooth spatial potential / witness ($W_1$, MMD) |
| **2. How samples are scored** | Single-sample classification against fixed threshold $0$ | Pairwise Bradley–Terry ranking $\sigma(D(x) - D(x'))$ (RpGAN) or difference of expectations $\mathbb{E}_p[T] - \mathbb{E}_q[T]$ |
| **3. How the critic is regularized** | Unconstrained network ($\mathrm{JS}$ saturates on disjoint supports) | 1-Lipschitz constraint (spectral norm, WGAN-GP) or zero-centered gradient penalties $R_1 + R_2$ |

::: {.d2l-note}
**The modernized R3GAN reference recipe (Huang et al., 2024):** Combines relativistic pairwise ranking (`RpGAN`) with zero-centered input-gradient penalties on both real ($R_1$) and generated ($R_2$) samples to damp rotational minimax dynamics without batch normalization.
:::
:::

::: {.slide title="Chapter 16 Roadmap: Theory, Dynamics, and Systems"}
[Architectural & Mathematical Roadmap]{.kicker}

The chapter progresses from exact population divergences to dynamical stability, convolutional image synthesis, and modern perceptual discriminators:

| Part | Sections | Core Mathematical & Systems Questions |
| :--- | :--- | :--- |
| **I. Objectives & Divergences** | **§16.1** Generative Adversarial Networks<br>**§16.2** Objectives & Divergences<br>**§16.3** Relativistic Objectives | Why does log loss yield $D^\star = \log(p/q)$ and $\mathrm{JS}(p, q) = I(x; y)$? How do non-saturating weights ($2\,D_{\textrm{KL}}(m \,\|\, p)$), $f$-divergences, IPMs ($W_1$, MMD), and Bradley–Terry pairing ($d_{\mathrm{Rp}}$) behave when supports separate? |
| **II. Dynamics & Convergence** | **§16.4** Gradient Penalties & Convergence | Why do minimax gradient updates orbit or spiral outward on the Dirac-GAN? How do $R_1 + R_2$ shift Jacobian eigenvalues into the left half-plane and act as a dual Sobolev low-pass filter? |
| **III. Images, Conditioning & Applications** | **§16.5** Adversarial Image Generation<br>**§16.6** Conditional Generation<br>**§16.7** Adversarial Losses Beyond GANs | How do DCGAN and normalization-free backbones scale to $64 \times 64$ images? How are FID, KID, precision/recall, and projection critics evaluated? Why do diffusion distillation, VQGAN, and HiFi-GAN still need a critic? |
:::

::: {.slide title="Running Benchmarks and Study Guide"}
[Pedagogical Thread]{.kicker}

Each section tests its theoretical predictions on concrete analytical distributions before scaling to high-dimensional image datasets:

::: {.cols .vc}
::: {.col}
- **Exact-density 2D Gaussians & mixtures (§16.1–16.3):** Check learned critic logits directly against analytical $\lambda(x) = \log(p(x)/q(x))$ and measure generator fit in exact nats of $D_{\textrm{KL}}(q \,\|\, p)$ and $\mathrm{JS}(p, q)$.
- **25-Gaussian $5 \times 5$ grid (§16.4):** Isolate whether pairwise ranking or $R_1 + R_2$ gradient penalties eliminate mode dropping.
- **Image & conditional synthesis (§16.1, §16.5–16.6):** $60{,}000$ Fashion-MNIST garments ($28 \times 28$), $40{,}597$ Pokémon sprites ($64 \times 64$), and 10-class CIFAR-10 ($32 \times 32$).
- **Bimodal regression & distillation (§16.7):** Two-band benchmark $x = c \pm 1 + 0.05\,\xi$ with an exact closed-form denoising teacher.
:::

::: {.col .fig}
![](../img/mdl-gan-weights.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Concept Check: Navigating Adversarial Formulation"}
[Check Your Understanding]{.kicker}

Test your intuition on three foundational scenarios from the chapter overview:

1. **Reading an optimal critic's logit:** An optimal log-loss critic returns $D^\star(x_1) = 1.2$ at point $x_1$ and $D^\star(x_2) = -2.0$ at point $x_2$. What is the density ratio $p(x)/q(x)$ at each point, and near which point should the generator reduce its sample density?
2. **Diagnosing three failure modes:** During training, you observe: (a) generated samples sit far from the data and generator gradients are nearly zero; (b) both losses hover near equilibrium while generated samples orbit in a cycle; (c) requesting a *"bird"* always returns the exact same canonical bird. Which chapter mechanism addresses each symptom?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Since $D^\star(x) = \log(p(x)/q(x))$, we have $p(x_1)/q(x_1) = e^{1.2} \approx \mathbf{3.32}$ (real data is $3.3\times$ denser) and $p(x_2)/q(x_2) = e^{-2.0} \approx \mathbf{0.135}$ (the generator is $\approx 7.4\times$ too dense). The generator must move probability mass away from **$x_2$** toward **$x_1$**.
2. **(a)** Use a **non-saturating generator loss** (§16.1) or smooth/penalized critic (§16.2) so rejected samples ($D(x') \ll 0$) retain unit gradient weight. **(b)** Add **zero-centered gradient penalties $R_1 + R_2$** (§16.4) and exponential moving average (**EMA**, §16.5) to damp rotational minimax dynamics. **(c)** Measure **per-class $K$-NN recall** (§16.6) to catch intra-class mode collapse hidden by high classifier alignment.
:::
:::

::: {.slide title="Summary: Core Notation and Chapter Synthesis"}
[Key Takeaways]{.kicker}

| Symbol / Concept | Mathematical Definition | Equilibrium Value | Role Across Chapter 16 |
| :--- | :--- | :--- | :--- |
| **Densities & Mixture** | $x \sim p$ (data), $x' = G(z) \sim q$ (gen.), $m = \frac{1}{2}(p + q)$ | $q = p = m$ | Two-sample comparison domain |
| **Density Ratio & Log-Ratio** | $\rho(x) = \frac{p(x)}{q(x)}$, $\;\lambda(x) = \log \frac{p(x)}{q(x)}$ | $\rho(x) = 1, \;\lambda(x) = 0$ | Target estimated by optimal critic $D^\star(x)$ |
| **Bayes Class Posterior** | $\eta(x) = \frac{p(x)}{p(x) + q(x)}$ $= \sigma(\lambda(x))$ | $\eta(x) = \frac{1}{2}$ | Pointwise realness probability |
| **Generator Gradient Weights** | $w_{\textrm{sat}} = \sigma(D(x'))$, $w_{\textrm{ns}} = \sigma(-D(x'))$, $w_{\textrm{Rp}} = \mathbb{E}_p[\sigma(D(x') - D(x))]$ | $w = \frac{1}{2}$ | Rescales $\nabla_\theta D(G_\theta(z))$ per sample |
| **Pairing Objective & Divergence** | $\Phi(D) = \mathbb{E}_{p,q}[\log\sigma(D(x) - D(x'))]$, $d_{\mathrm{Rp}} = \mathrm{JS}(p \otimes q, q \otimes p)$ | $\Phi = -\log 2$, $d_{\mathrm{Rp}} = 0$ | Shift-invariant Bradley–Terry ranking (§16.3) |
| **Zero-Centered Penalties** | $R_1 = \frac{\gamma}{2}\mathbb{E}_p[\|\nabla_x D\|^2]$, $R_2 = \frac{\gamma}{2}\mathbb{E}_q[\|\nabla_{x'} D\|^2]$ | $R_1 = R_2 = 0$ ($\nabla_x D^\star = \mathbf{0}$) | Damps rotational vector fields (§16.4–16.6) |
:::
