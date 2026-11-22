# Diffusion Models
:label:`chap_diffusion`

The adversarial generators of :numref:`chap_gans` required a second network—a discriminator or critic—because a generator mapping noise to data has no direct regression target. Diffusion and flow-matching models construct a regression target by progressively corrupting each training example with Gaussian noise. Reversing a small Gaussian noise step requires estimating the conditional expectation of the clean data given the noisy observation.

This perspective unites three lines of work: non-equilibrium thermodynamics and denoising diffusion probabilistic models (DDPM) :cite:`sohl2015deep,ho2020denoising`, score-based generative modeling :cite:`song2019generative,song2021score`, and flow matching or stochastic interpolants :cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022,Albergo.Boffi.VandenEijnden.2023`. In all three formulations, a single neural network learns to predict the clean data (or equivalently the added noise, the score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$, or the velocity along the path) at every noise level $t \in [0, 1]$, and a numerical sampler integrates those predictions from pure noise at $t = 1$ back to data at $t = 0$.

Replacing the adversarial minimax objective with regression resolves three limitations of GANs: training requires only a single network optimized by squared error, the weighted regression loss provides a variational bound on the log-likelihood (and an exact log-likelihood via the probability-flow ODE), and the likelihood objective penalizes missing modes. Throughout this chapter, our default formulation regresses the velocity along the linear path $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$ :cite:`Esser.Kulal.Blattmann.ea.2024`, alongside DDPM's noise predictor on the variance-preserving path.

The first three sections develop the complete mathematical and algorithmic core on a two-dimensional mixture of three Gaussians, where the true density, score, posterior mean, and velocity field are available in closed form. :numref:`sec_diffusion-images` moves to Fashion-MNIST images, introducing the time-conditioned U-Net, Fréchet distance evaluation, and classifier-free guidance. The next four sections examine the sampler (:numref:`sec_diffusion-samplers`), the likelihood bound and design space (:numref:`sec_diffusion-design-space`), latent diffusion and transformer backbones (:numref:`sec_diffusion-latent`), and few-step distillation via flow maps (:numref:`sec_diffusion-few-step`). Finally, :numref:`sec_diffusion-discrete` extends diffusion to discrete tokens by replacing Gaussian noise with token masking. Longer mathematical proofs appear in :numref:`chap_mdl-dynamics`, and external tutorials, courses, and reference implementations are collected in :numref:`sec_diffusion-further-reading`.

```toc
:maxdepth: 2

denoising
gaussian-paths
integration
image-diffusion
samplers
design-space
latent-diffusion
few-step
discrete-diffusion
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17]{.kicker}

Diffusion and Flow-Based Generative Models<br>**denoising & scores · Gaussian probability paths · ODE & SDE integration · U-Net & DiT · fast samplers · discrete masking**
:::
:::

::: {.slide title="Why Diffusion Replaces the Adversarial Minimax Game"}
[Motivation]{.kicker}

A generator mapping random noise to data has no direct supervision target—forcing GANs to train a competing discriminator. Progressive Gaussian corruption creates an explicit regression target at every noise level:

::: {.cols .vc}
::: {.col}
- **Single-network regression:** Corrupting a training example $\mathbf{x}_0$ into $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ lets one network regress directly onto the clean point, added noise, score, or velocity by squared error.
- **Exact likelihoods and bounds:** The weighted denoising loss bounds the negative log-likelihood $-\log p_{\boldsymbol\theta}(\mathbf{x}_0)$, while the deterministic probability-flow ODE yields exact log-densities via change of variables.
- **Mode coverage guarantee:** Minimizing expected negative log-likelihood $\mathbb{E}_p[-\log p_{\boldsymbol\theta}] = H(p) + D_{\textrm{KL}}(p \,\|\, p_{\boldsymbol\theta})$ heavily penalizes dropping any data mode.
:::

::: {.col .fig}
![](../img/mdl-diffusion-forward-reverse.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Gaussian Paths and the Shared Log-SNR Clock"}
[Forward Corruption & Time Conventions]{.kicker}

Every continuous model in this chapter interpolates between clean data at $t = 0$ and standard Gaussian noise $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ at $t = 1$:

::: {.cols .vc}
::: {.col}
$$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$$

- **Linear path (flow matching default):** $\alpha_t = 1 - t$ and $\sigma_t = t$, moving each training pair at constant conditional velocity $\dot{\mathbf{x}}_t = \boldsymbol\epsilon - \mathbf{x}_0$.
- **Variance-preserving (VP) path (DDPM):** $\alpha_t^2 + \sigma_t^2 = 1$, preserving unit variance across $T = 1{,}000$ discrete forward steps.
- **Canonical log-SNR clock:** $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$ decreases from $+\infty$ at $t = 0$ to $-\infty$ at $t = 1$; samplers always integrate backward from $t = 1$ to $t = 0$.
:::

::: {.col .fig}
![](../img/mdl-diffusion-paths.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="One Posterior Mean Unifies Four Regression Targets"}
[The Mathematical Core]{.kicker}

Because squared error is minimized by the conditional expectation given the input, training on per-example targets recovers the posterior mean $\hat{\mathbf{x}}_0(\mathbf{x}_t) = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$:

::: {.cols .vc}
::: {.col}
- **Tweedie's identity:** The posterior mean of the clean point and the posterior mean of the noise $\hat{\boldsymbol\epsilon} = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$ determine the score of the marginal density $p_t$:
  $$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{\hat{\boldsymbol\epsilon}(\mathbf{x}_t)}{\sigma_t}$$
- **Affine target equivalence:** Since $\mathbf{x}_t = \alpha_t \hat{\mathbf{x}}_0 + \sigma_t \hat{\boldsymbol\epsilon}$, the clean prediction $\hat{\mathbf{x}}_0$, noise $\hat{\boldsymbol\epsilon}$, score $\nabla \log p_t$, and marginal velocity $\mathbf{u}_t = \dot\alpha_t \hat{\mathbf{x}}_0 + \dot\sigma_t \hat{\boldsymbol\epsilon}$ are affine reparameterizations of one another.
- **Reverse-time integration:** A numerical ODE or SDE solver integrates $\mathbf{u}_t(\mathbf{x})$ from $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ down to $\mathbf{x}_0$.
:::

::: {.col .fig}
![](../img/mdl-dyn-tweedie.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Translating Time Conventions Across the Literature"}
[Reading Diffusion vs. Flow-Matching Papers]{.kicker}

Diffusion papers place clean data at $t = 0$, whereas many flow-matching papers reverse the clock and place noise at $t = 0$:

::: {.cols .vc}
::: {.col}
- **Our convention (diffusion standard):** Clean data sit at $t = 0$ ($\mathbf{x}_0$), pure noise sits at $t = 1$ ($\mathbf{x}_1 = \boldsymbol\epsilon$), and generation integrates from $t = 1$ down to $t = 0$.
- **Converting flow-matching papers:** Replace their time $t_{\textrm{FM}}$ with $1 - t$, map their initial noise $\mathbf{x}_0^{\textrm{FM}}$ to $\boldsymbol\epsilon$ and target data $\mathbf{x}_1^{\textrm{FM}}$ to $\mathbf{x}_0$, and negate every velocity vector ($\mathbf{v}_t^{\textrm{FM}} = -\mathbf{u}_{1-t}$).
- **Sanity check via log-SNR:** Always verify whether a paper's time $t = 0.8$ corresponds to high SNR ($\lambda > 0$, mostly data) or low SNR ($\lambda < 0$, mostly noise).
:::

::: {.col .fig}
![](../img/mdl-dyn-time-conventions.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Concrete Running Benchmarks Across Chapter §{num:chap_diffusion}"}
[Empirical Grounding]{.kicker}

Every theoretical claim and architectural choice is tested on controlled benchmarks where exact references or calibrated feature metrics exist:

::: {.cols .vc}
::: {.col}
- **2D Three-Mode Gaussian Mixture (§{sec_diffusion-denoising}–§{num:sec_diffusion-integration}, §{sec_diffusion-design-space}, §{sec_diffusion-few-step}):** Weights $(0.5, 0.3, 0.2)$ and width $s = 0.5$ where exact density, score, posterior mean, velocity, and entropy ($\approx 2.5$ nats) are available in closed form.
- **Fashion-MNIST Pixels & Latents (§{sec_diffusion-images}–§{num:sec_diffusion-few-step}):** $28 \times 28$ images in $[-1, 1]$ and $7 \times 7 \times 4$ VAE latents, evaluated by Fréchet distance (FD) against the finite-sample real-vs.-real reference floor.
- **Discrete Token Sequences (§{sec_diffusion-discrete}):** Quantized $32 \times 32$ grid pairs and 784-token binary garments corrupted by token masking $[\textrm{M}]$.
:::

::: {.col .fig}
![](../img/mdl-diffusion-score-field.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Chapter §{num:chap_diffusion} Roadmap: From Score Matching to Tokens"}
[Study Guide]{.kicker}

::: {.cols .vc}
::: {.col}
- **§§{sec_diffusion-denoising}–§{num:sec_diffusion-integration} Core Theory:** Denoising score matching, Tweedie's formula, Gaussian paths ($\lambda_t$), and reverse ODE/SDE transport (trained MLP matches exact 2D oracle weights within $0.004$).
- **§§{sec_diffusion-images}–§{num:sec_diffusion-design-space} Images, Fast Solvers & Design Space:** U-Net + CFG ($\approx 90\%$ class agreement), 20-NFE Heun and multistep solvers, continuous ELBO ($w(\lambda) \equiv 1$), and EDM preconditioning.
- **§§{sec_diffusion-latent}–§{num:sec_diffusion-discrete} Latents, Few-Step & Discrete Tokens:** KL-autoencoders + `adaLN-Zero` DiT ($>10\times$ fewer FLOPs), two-time flow-map distillation ($2$ jumps match $15$ Euler steps), and absorbing masked diffusion ($0.17$ bpp).
:::

::: {.col .fig .big}
@fig:ch17-diff-roadmap
:::
:::
:::
