# Diffusion Models
:label:`chap_diffusion`

A generative model is asked for two things: a probability, or a density,
for data like those it was trained on, such as an image of a coat, and new
samples that could have come from the same source, such as coats nobody
photographed. Four established families of generative models pay for these
in one of two ways.
Likelihood-based models keep their density normalized by
construction: they factorize it into conditionals, as the language models of
:numref:`sec_language-model` do, average normalized components over latent
variables and train through a tractable lower bound on the log-likelihood
(:numref:`sec_mdl-latent-em-elbo`), or compose
invertible maps whose volume change is tractable
(:numref:`sec_mdl-continuous-normalizing-flows`). Each of these choices
constrains the architecture. Generative adversarial networks
(:numref:`chap_gans`) leave the generator's architecture unconstrained and
give up the density. They are trained through a two-player game whose
dynamics need not converge and whose samples can collapse onto a few modes.

An energy-based model keeps the density and removes the architectural
constraint, and pays a third price. It writes the density as
$p_{\boldsymbol{\theta}}(\mathbf{x}) = \exp(-E_{\boldsymbol{\theta}}(\mathbf{x})) / Z(\boldsymbol{\theta})$,
where the energy $E_{\boldsymbol{\theta}}$ is any network for which the
normalizing constant $Z(\boldsymbol{\theta})$, the integral of
$\exp(-E_{\boldsymbol{\theta}})$ over the whole input space, is finite. The
network is free, but every likelihood evaluation involves this integral,
which for a general network has no closed form and is expensive to
approximate reliably in high dimension, and every gradient step involves
expectations under the
distribution it normalizes. This chapter shows how to train and sample from
models of this kind, and from their descendants, with methods that never
compute it.

The device is the *score*, the gradient of the log-density with respect to
the input. Differentiating $\log p_{\boldsymbol{\theta}}$ removes
$Z(\boldsymbol{\theta})$, which does not depend on $\mathbf{x}$. A model of
the score can be fitted with objectives computable from data alone, and
approximate samples can be drawn from a score by a noisy gradient ascent
called Langevin dynamics. Both steps become unreliable where the data are
sparse: the score is poorly determined there, and the sampler crosses such
regions slowly. The repair, estimating the score of progressively noisier
copies of the data and then denoising from the noisiest level to the
cleanest, is the idea behind diffusion models.

The sections follow this dependency. :numref:`sec_diffusion-limits` states
what adversarial training leaves unsolved and introduces energy-based models
together with the normalization problem. :numref:`sec_diffusion-ebm-training`
derives the maximum-likelihood gradient of an energy-based model and the
Markov chain Monte Carlo methods that estimate it, which exposes the cost of
drawing samples from the model during training.
:numref:`sec_diffusion-score-matching` replaces the likelihood with the
Fisher divergence and derives score matching objectives that need no samples
from the model. :numref:`sec_diffusion-denoising` obtains the score of
noise-perturbed data from a denoising regression and applies it to images.
:numref:`sec_diffusion-langevin` turns a score into a sampler and identifies
where a score estimated at a single noise level fails.
:numref:`sec_diffusion-annealed` estimates scores at many noise levels and
anneals the sampler through them. :numref:`sec_diffusion-ddpm` defines the
forward noising process and the learned reverse process of a denoising
diffusion probabilistic model, derives its variational objective, and shows
that its terms are weighted noise-prediction regressions, the same
regression as before, whose unweighted version is the loss used in practice.
:numref:`sec_diffusion-images` conditions the U-Net of
:numref:`sec_diffusion-annealed` on the class, trains it on Fashion-MNIST,
and adds classifier-free guidance and, for comparison, classifier
guidance.

Three sections then extend the model along the directions in which the
field has moved. Sampling from the model of :numref:`sec_diffusion-images`
runs the reverse chain through all of its thousand steps;
:numref:`sec_diffusion-ddim` shows that the trained noise predictor also
defines a deterministic sampler that skips most of them and, for a
well-trained predictor, reaches comparable samples in a few dozen.
:numref:`sec_diffusion-flow-matching` recasts the whole construction as
learning a velocity field along a prescribed path from noise to data and
shows, through an exact dictionary between scores and velocities on Gaussian
paths, that diffusion is one choice of path among many.
:numref:`sec_diffusion-discrete` carries the framework to categorical data,
where noise means masking or resampling tokens, and derives the
masked-diffusion objective used by recent diffusion language models.

:numref:`sec_diffusion-ebm-training` through :numref:`sec_diffusion-ddpm`
share one running example, a two-dimensional mixture of three Gaussians with
unequal weights, and the three closing sections return to it, the last one
in a quantized form. Its density, its score, and the density and score of
every noise-perturbed version of it are available in closed form, so each
learned quantity can be compared with the truth rather than judged by eye.
Images enter in :numref:`sec_diffusion-denoising` through Fashion-MNIST
(:numref:`sec_fashion_mnist`) and return in :numref:`sec_diffusion-annealed`
and in every section from :numref:`sec_diffusion-images` on. The
$28 \times 28$ images keep each experiment within about half an hour on a
laptop processor, and most within minutes. The running times quoted in the
chapter were measured on a ten-core Apple M5 and vary with the hardware.

:numref:`tab_diffusion_models` lists the models of the chapter, the quantity
each network represents, the objective that fits it, and the sampler that
turns it into data. The rows are ordered as the chapter is. Through the
fifth row each removes a limitation of the one above it; the last three
extend the fifth in different directions.

:The models of this chapter. Each row names what the network represents, the objective that trains it, and the procedure that produces samples.
:label:`tab_diffusion_models`

| model | the network represents | training objective | sampler | where |
|:--|:--|:--|:--|:--|
| energy-based model | an energy $E_{\boldsymbol{\theta}}(\mathbf{x})$; the density is proportional to $\exp(-E_{\boldsymbol{\theta}})$ | maximum likelihood, whose gradient lowers the energy at the data and raises it at samples from the model | Markov chain Monte Carlo, also needed inside training | :numref:`sec_diffusion-limits`, :numref:`sec_diffusion-ebm-training` |
| score model | the score $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p(\mathbf{x})$ | Fisher divergence, made computable by score matching or sliced score matching | Langevin dynamics | :numref:`sec_diffusion-score-matching`, :numref:`sec_diffusion-langevin` |
| denoising score model | the score of the data after Gaussian perturbation at one noise level | denoising score matching, a regression on the added noise | Langevin dynamics at that noise level, which samples the perturbed data | :numref:`sec_diffusion-denoising`, :numref:`sec_diffusion-langevin` |
| noise-conditional score network | the scores at a ladder of noise levels | denoising score matching, weighted by the squared noise level and summed over the ladder | annealed Langevin dynamics | :numref:`sec_diffusion-annealed` |
| denoising diffusion probabilistic model | the noise $\boldsymbol{\epsilon}$ in $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}$, the accumulated noise of the first $t$ steps | a variational bound whose terms are weighted noise-prediction regressions; trained with the unweighted simple loss | ancestral sampling of the learned reverse chain | :numref:`sec_diffusion-ddpm`, :numref:`sec_diffusion-images` |
| denoising diffusion implicit model | the same noise predictor, without retraining | the same simple loss | a deterministic update on a sparse subset of the steps, with a knob that interpolates back to the stochastic chain | :numref:`sec_diffusion-ddim` |
| flow matching | a velocity field along a path from noise to data | regression onto the conditional velocity of the path; on Gaussian paths the velocity is an affine function of the score, so diffusion is one such path | numerical integration of an ordinary differential equation | :numref:`sec_diffusion-flow-matching` |
| discrete (masked) diffusion | the distribution of each clean token given a partially masked sequence | a variational bound that reduces to a weighted cross-entropy on masked tokens | iterative unmasking | :numref:`sec_diffusion-discrete` |

The chapter assumes the probability and maximum-likelihood material of
:numref:`sec_mdl-maximum_likelihood`, including the Kullback--Leibler
divergence and the evidence lower bound, and it recalls the Markov chain
material it needs. Results from :numref:`chap_gans` are
referenced but not required. Proofs already given in the mathematics appendix
are cited rather than repeated: Hyvärinen's identity, Vincent's theorem, the
stationarity of Langevin dynamics, and the correspondence between the
diffusion chain and a stochastic differential equation all appear in
:numref:`sec_mdl-score-matching-diffusion-flow`, and
:numref:`sec_mdl-fokker-planck-probability-flow` develops the continuous-time
view on which that section builds. Readers who want the
differential-equation perspective behind
:numref:`sec_diffusion-flow-matching` will find it in these two sections.

The order of presentation of :numref:`sec_diffusion-limits` through
:numref:`sec_diffusion-images` follows lectures 10 through 13 of the *Deep
Generative Models* course of :citet:`Kuleshov.2023`, whose slides credit
figures and content adapted from Yang Song, Stefano Ermon, and Lilian Weng.
Each section cites the course where it follows that exposition and cites the
original papers for its results; the three closing sections go beyond those
lectures and cite the original papers directly. The monograph of
:citet:`Lai.Song.Kim.ea.2025` develops the same material from the
variational, score-based, and flow-based viewpoints at greater length.

```toc
:maxdepth: 2

limits
energy-training
score-matching
denoising
langevin
annealed-langevin
ddpm
image-diffusion
ddim
flow-matching
discrete-diffusion
```

## Resources and Further Reading {.unnumbered}

The chapter derives each objective from the one before it and verifies the
derivations on a distribution with a closed-form score. The following
resources provide the original expositions, longer treatments, and reference
implementations.

- The [Deep Generative Models course](https://kuleshov-group.github.io/dgm-website/) :cite:`Kuleshov.2023` supplies the lecture structure that this chapter follows. Its lecture 10 covers the adversarial limitations that open :numref:`sec_diffusion-limits`, and lectures 11 through 13 cover energy-based models, score-based models, and diffusion models in the same order.
- [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) :cite:`Lai.Song.Kim.ea.2025` is a monograph that presents the variational, score-based, and flow-based derivations of diffusion models and their unification through differential equations.
- Yang Song's blog post [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) :cite:`Song.2021` explains score matching, Langevin dynamics, and the multi-scale noise perturbation that its author introduced with Stefano Ermon, with animations of Langevin and annealed Langevin sampling and a figure of the low-density failure analyzed in :numref:`sec_diffusion-langevin`.
- Lilian Weng's blog post [What Are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) :cite:`Weng.2021` walks through the variational derivation of :numref:`sec_diffusion-ddpm` step by step and collects the later extensions in one place.
- :citet:`Luo.2022` gives a self-contained account of the equivalence between the variational and score-based derivations, including the algebra that this chapter compresses.
- The lecture notes of :citet:`Holderrieth.Erives.2025` develop diffusion models from the flow-matching side first, with proofs, and complement :numref:`sec_diffusion-flow-matching`; :citet:`Karras.Aittala.Aila.ea.2022` is the reference for the design of samplers and noise parameterizations that :numref:`sec_diffusion-ddim` only names.
- The [official DDPM implementation](https://github.com/hojonathanho/diffusion) accompanying :citet:`ho2020denoising` and the [score SDE implementation](https://github.com/yang-song/score_sde_pytorch) accompanying :citet:`song2021score` contain the full-scale U-Nets, noise schedules, and samplers that the small models of :numref:`sec_diffusion-images` imitate.
