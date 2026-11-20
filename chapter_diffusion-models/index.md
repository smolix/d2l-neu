# Diffusion Models
:label:`chap_diffusion`

The generators of :numref:`chap_gans` learned to draw garments through a game between two networks, and that chapter closed with four limits of the game: no density, two networks and a tuned penalty, no cost for a missing mode, and instability at scale. Why does a generator need a game at all? A classifier needs none, because every training image comes with the label it is fitted to. A generator has nothing to copy; its image must come out of noise. Regression onto what, then, and how can fitting a target produce something new?

The answer came from physics. Nobody has watched a drop of ink unmix itself in a glass of water. Yet each small step that mixed it adds a little Gaussian noise, and its reverse is again nearly Gaussian, so undoing it means learning mainly a mean. An image noised step by step is such a drop of ink. :citet:`sohl2015deep` trained a network to undo the steps, :citet:`ho2020denoising` found that the target is, of all things, the added noise, and a year later such models drew better images than the best GANs :cite:`Dhariwal.Nichol.2021`. Surprisingly, routes that started from the gradient of the log-density, the *score* :cite:`song2019generative`, and from paths between noise and data :cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022,Albergo.Boffi.VandenEijnden.2023` reach the same place: one network that predicts the clean point from a noisy one at every noise level, and a sampler that integrates its predictions.

This regression answers three of the four limits. One network learns it with no critic, and its loss, suitably weighted, bounds the likelihood, which gives a density and charges a model for every point of a mode it drops. Stability at scale cannot be tested on garments of $28 \times 28$ pixels. The second question is harder than it looks. The best prediction of a heavily noised image is the average of all the images that could have produced it, and a sampler built on such averages draws new images. Why a network trained on a finite set does not simply return its training images is still open (:numref:`sec_diffusion-further-reading`).

Our default is the form many image generators adopted from 2024 on :cite:`Esser.Kulal.Blattmann.ea.2024`: a network predicts the velocity along a straight line from data to noise, and the sampler follows it back. Ho et al.'s noise predictor is a second target with the same information. We start simple: the first three sections build the whole model on three Gaussians in the plane, where every learned quantity can be checked against the truth. A reader who stops there understands diffusion models.

:numref:`sec_diffusion-images` moves to Fashion-MNIST and builds the instruments that judge samples. The next four sections each change one choice (the sampler, the objective, pixels or a latent code, the number of steps) and say what to pick. A hurried reader can take the likelihood bound on trust and leave few-step generation, which needs a second training run, until speed matters. The last section makes the noise an erasure, for tokens. Longer proofs are in the optional :numref:`chap_mdl-dynamics`, and text-to-image and video generation lie outside the chapter (:numref:`sec_diffusion-further-reading`).

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

## Resources and Further Reading {.unnumbered}

These resources accompany particular sections; :numref:`sec_diffusion-further-reading` turns to the methods the chapter does not teach. Start with one of the first three: the monograph for the same material at greater length and with proofs, the course for building each model in code, the interactive article for pictures.

- [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) :cite:`Lai.Song.Kim.ea.2025` derives diffusion models along the variational, score-based and flow-based routes and shows where they meet.
- The [notes, lectures and labs](https://diffusion.csail.mit.edu/) of MIT's course 6.S184 :cite:`Holderrieth.Erives.2025` teach flow matching, latent spaces with transformer backbones, and discrete diffusion, with labs that build a flow-matching model and a latent diffusion transformer from scratch.
- [*Diffusion Meets Flow Matching*](https://diffusionflow.github.io/) :cite:`Gao.Hoogeboom.Heek.ea.2025` shows with interactive figures that the DDIM sampler and the Euler sampler of flow matching are one algorithm under a rescaling of the schedule, and that for a fixed weighting of the noise levels the training schedule leaves the objective unchanged and affects only the variance of its estimate.
- The [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) :cite:`Lipman.Havasi.Holderrieth.ea.2024` and its [library](https://github.com/facebookresearch/flow_matching) give the training loop of :numref:`sec_diffusion-gaussian-paths` in full, with proofs.
- [*Step-by-Step Diffusion*](https://arxiv.org/abs/2406.08929) :cite:`Nakkiran.Bradley.Zhou.ea.2025` builds the deterministic sampler of :numref:`sec_diffusion-integration` up from a single point mass and derives the stochastic and deterministic samplers of :numref:`sec_diffusion-samplers` (DDPM and DDIM) without stochastic calculus.
- Sander Dieleman's [blog](https://sander.ai/) explains how noise acts on image frequencies :cite:`Dieleman.2024`, why generators work in a latent space :cite:`Dieleman.2025`, what flow maps learn :cite:`Dieleman.2026`, and how continuous and discrete diffusion compete for language :cite:`Dieleman.2026b`.
- The [official DDPM implementation](https://github.com/hojonathanho/diffusion) of :citet:`ho2020denoising` and the [score SDE implementation](https://github.com/yang-song/score_sde_pytorch) of :citet:`song2021score` contain the full-scale networks and samplers that the small models of :numref:`sec_diffusion-images` imitate.
- The [MDLM repository](https://github.com/kuleshov-group/mdlm) of :citet:`Sahoo.Arriola.Schiff.ea.2024` trains the masked diffusion language model of :numref:`sec_diffusion-discrete`.
- The [*Deep Generative Models* course](https://kuleshov-group.github.io/dgm-website/) :cite:`Kuleshov.2023` and Yang Song's [blog post](https://yang-song.net/blog/2021/score/) :cite:`Song.2021` tell the score-based history behind :numref:`sec_diffusion-denoising`.
- Lilian Weng's [*What Are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) :cite:`Weng.2021` and the [tutorial](https://arxiv.org/abs/2208.11970) of :citet:`Luo.2022` work through the variational algebra that :numref:`sec_diffusion-design-space` compresses.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17]{.kicker}

Diffusion models<br>
**regression onto the clean point · data at t = 0, noise at t = 1 · samplers that integrate · nine sections**
:::
:::

::: {.slide title="A Generator Has No Target to Copy; Adding Noise Makes One"}
[The problem]{.kicker}

![](../img/mdl-diffusion-forward-reverse.svg){width=85%}

- A classifier is fitted to the label that comes with each example; a
  generator's image must come out of noise, and no example says which image.
- Noise a training example in many small steps (a thousand for DDPM); learn
  to undo each step by regression.

::: {.d2l-note .warn}
Regression onto what, and how can fitting a target produce something new?
:::
:::

::: {.slide title="Regression Answers Three of the Game's Four Limits"}
[Why diffusion]{.kicker}

| limit of the adversarial game | answer in this chapter |
|---|---|
| no density | a likelihood bound, and an exact likelihood by an ODE |
| two networks and a tuned penalty | one network, one regression |
| no cost for a missing mode | bound $\geq H(p) + D_{\textrm{KL}}(p\,\|\,p_{\boldsymbol\theta})$: a dropped mode costs on each of its points |

The fourth limit, instability at scale, needs far larger models than
$28 \times 28$ garments to test.

::: {.d2l-note}
The target of the regression is computed from each training example: no
critic, no game.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- convert a denoiser into a score, a noise predictor and a velocity;
- train one network along a path from data to noise by one regression, and
  sample it by an ODE or a chain;
- read the loss as a weighted likelihood bound;
- pick a sampler and step count, a weighting, a backbone, and a latent space
  or pixels;
- cut sampling to a few steps, and diffuse tokens by masking.

**The idea:** one network predicts the clean point from a noisy one at
every noise level; a sampler integrates its predictions from noise to data.
:::

::: {.slide title="From a Learned Comparison to a Fixed Regression Target"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| chapter or part | the model learns from |
|---|---|
| Basics | the likelihood of observed data |
| Generative Adversarial Networks | a critic that compares samples with data |
| **Diffusion Models** | **regression onto the clean point at every noise level** |
:::

::: {.col .narrow}
- **Uses:** squared error learns a conditional mean; the Fréchet distance;
  U-Nets; transformers on image patches.
- **Skips:** text-to-image and video at scale; long proofs (the optional
  appendix on dynamics).
:::
:::
:::

::: {.slide title="The Clock Runs from Data at t = 0 to Noise at t = 1"}
[Notation]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon,
\qquad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

- Variance-preserving (VP): $\alpha_t^2 + \sigma_t^2 = 1$.
- Linear (flow matching): $\alpha_t = 1 - t$, $\sigma_t = t$.
- Log signal-to-noise ratio $\lambda_t = \log(\alpha_t^2/\sigma_t^2)$ falls
  from $+\infty$ to $-\infty$.
:::

::: {.col .fig .big}
![](../img/mdl-diffusion-paths.svg)
:::
:::

::: {.d2l-note}
Every sampler runs from $t = 1$ down to $t = 0$.
:::
:::

::: {.slide title="The Default: Regress a Velocity on the Linear Path, Then Integrate Back"}
[The chapter's model]{.kicker}

On the linear path a training point moves with velocity
$\boldsymbol\epsilon - \mathbf{x}_0$. The network $\mathbf{u}_{\boldsymbol\theta}$
regresses onto it:

$$\min_{\boldsymbol\theta}\ \mathbb{E}_{t,\, \mathbf{x}_0,\, \boldsymbol\epsilon}
\big\|\mathbf{u}_{\boldsymbol\theta}(\mathbf{x}_t, t) - (\boldsymbol\epsilon - \mathbf{x}_0)\big\|^2$$

Sampling integrates $d\mathbf{x}/dt = \mathbf{u}_{\boldsymbol\theta}(\mathbf{x}, t)$
from noise at $t = 1$ to $t = 0$, for instance by Euler steps.

::: {.d2l-note}
DDPM's noise predictor on the VP path is the second model: another target,
the same information.
:::
:::

::: {.slide title="Flow-Matching Papers Run the Clock the Other Way"}
[Notation]{.kicker}

::: {.cols .vc}
::: {.col .fig .big}
![](../img/mdl-dyn-time-conventions.svg)
:::

::: {.col}
To read such a paper:

- replace its $t$ by $1 - t$;
- its noise $\mathbf{x}_0$ by our $\boldsymbol\epsilon$;
- its data $\mathbf{x}_1$ by our $\mathbf{x}_0$;
- flip the sign of every velocity.
:::
:::
:::

::: {.slide title="Check: Read the Clock"}
[Check]{.kicker}

1. On the linear path at $t = 0.5$, what are $\alpha_t$, $\sigma_t$ and
   $\lambda_t$?
2. A flow-matching paper evaluates its network at its $t = 0.8$. Which of
   our times is that: mostly data or mostly noise?

::: {.fragment .fade-up}
1. $\alpha = \sigma = 0.5$, so $\lambda = 0$: signal and noise are equal.
2. Our $t = 0.2$: $\alpha = 0.8$, $\sigma = 0.2$,
   $\lambda = \log 16 \approx 2.8$, mostly data.
:::
:::

::: {.slide title="The Posterior Mean Gives the Clean Point, the Noise, the Score and the Velocity"}
[The object]{.kicker}

::: {.cols .vc}
::: {.col}
$\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$, the
average clean point that could have produced $\mathbf{x}_t$, and
$\hat{\boldsymbol\epsilon} = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$;
$p_t$ is the density of $\mathbf{x}_t$:

$$\mathbf{x}_t = \alpha_t \hat{\mathbf{x}}_0 + \sigma_t \hat{\boldsymbol\epsilon},
\qquad
\nabla \log p_t(\mathbf{x}_t) = -\frac{\hat{\boldsymbol\epsilon}}{\sigma_t}$$

On the linear path the velocity is
$\mathbf{u}_t = \hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0$.
:::

::: {.col .fig}
![](../img/mdl-dyn-tweedie.svg)
:::
:::

::: {.d2l-note}
A network trained on any one target defines all the others.
:::
:::

::: {.slide title="Three Sections Build the Model; Four Each Change One Choice"}
[Chapter map]{.kicker}

| movement | sections |
|---|---|
| the whole model, on the mixture | Denoising and the Score · Gaussian Paths · Sampling by Integration |
| images, and how to judge them | Diffusion for Images |
| one choice each, with a recommendation | Samplers, Inversion and Editing (which sampler, how many steps) · Likelihood, Weighting and the Design Space (which objective) · Latent Diffusion and Transformers (which space, which network) · Few-Step Generation (one to four evaluations) |
| another kind of noise | Discrete Diffusion (erasure, for tokens) |

::: {.d2l-note}
Stop after the first row and you understand diffusion models.
:::
:::

::: {.slide title="A Mixture Where Everything Is Exact, Then Garments"}
[Running examples]{.kicker}

::: {.cols .vc}
::: {.col}
- **Three Gaussians in the plane**, weights 0.5, 0.3, 0.2, width
  $s = 0.5$: density, score and posterior mean exact at every noise level;
  no model beats about 2.5 nats per point.
- **Fashion-MNIST**, $28 \times 28$ in $[-1, 1]$: pixels, then
  $7 \times 7 \times 4$ latents; each section trains its own models.
- **Tokens:** the mixture quantized, and binary garments.
:::

::: {.col .fig}
![](../img/mdl-diffusion-score-field.svg)
:::
:::
:::

::: {.slide title="What the Chapter Measures"}
[Results ahead]{.kicker}

| section | result |
|---|---|
| Denoising | Tweedie's step with a trained score: within about 1% of the best denoiser |
| Images | the 1,000-step DDPM chain: FD 15 to 25% lower than 50 Euler steps |
| Samplers | Heun (a second-order ODE solver), even steps on the linear path: the chain's FD in about 20 evaluations |
| Likelihood | network within half a hundredth of a nat of the exact field (true value about 2.5 nats per point) |
| Latents | at this scale, the best pixel model beats every latent one |
| Few-Step | a distilled student in four evaluations: FD about 1.5 times that of 50 teacher steps |
| Discrete | binary garments: 0.17 bits per pixel, 0.70 for independent pixels |

FD: Fréchet distance between generated and real images in a classifier's
features; lower is better. Differences smaller than its measured noise are
not read.
:::

::: {.slide title="Check: Which Section Answers Each Symptom?"}
[Check]{.kicker}

1. Langevin chains that follow the exact score of the data keep the mode
   weights they started with.
2. A thousand network evaluations per image are too slow.
3. You asked for a sneaker and got a random garment.
4. The data are tokens, not real numbers.

::: {.fragment .fade-up}
1. Gaussian Paths: walk through every noise level, coarse to fine.
2. Samplers (about twenty to fifty evaluations), then Few-Step (one to
   four).
3. Diffusion for Images: classifier-free guidance.
4. Discrete Diffusion: masking as noise.
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- A generator has no target; noising a training example creates one.
- The clock: $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
  \boldsymbol\epsilon$, data at $t = 0$, noise at $t = 1$, samplers run
  from 1 to 0.
- Default: a velocity on the linear path, integrated back by an ODE.
- One posterior mean gives clean point, noise, score and velocity.
- Nine sections: three build the model, one moves it to images, four change
  one choice each, one changes the noise.

**Next:** what a denoiser learns, and what one noise level cannot do.
:::
