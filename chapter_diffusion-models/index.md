# Diffusion Models
:label:`chap_diffusion`

The generators of :numref:`chap_gans` learned to draw garments and sprites through a game between two networks, and that chapter closed with four limits the game leaves in place: no density, two networks and a tuned penalty, no cost for a missing mode, and instability at scale. Its last sentence pointed elsewhere: diffusion and flow models are trained by regression on a target computed from each training example. A classifier regresses onto a given label; a generator has nothing to copy, and its image must come out of noise. Regression onto what, then, and how can fitting a target produce something new?

The idea came from physics. In non-equilibrium thermodynamics, structure such as a drop of ink in water is destroyed by many small random steps, and when a step is small, its reversal has the same simple form. :citet:`sohl2015deep` added noise to images in the same way and trained a network to undo each step. :citet:`ho2020denoising` made the undoing a regression, predicting the added noise from the noisy image, and matched the sample quality of adversarial training. Others reached the same model from the gradient of the log-density :cite:`song2019generative` and from the velocity of a path between noise and data :cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022,Albergo.Boffi.VandenEijnden.2023`. All routes arrive at one network that predicts the clean point from a noisy one at every noise level, and a sampler that integrates its predictions.

Eight questions organize the chapter. What does a network that removes noise know about the data (:numref:`sec_diffusion-denoising`)? How do we walk from noise to data, and which single regression trains the walk (:numref:`sec_diffusion-gaussian-paths`)? What changes for images, and how do we steer and judge them (:numref:`sec_diffusion-images`)? Which of a thousand sampling steps can we skip, and what can a sampler do when run backwards (:numref:`sec_diffusion-samplers`)? Is the regression maximum likelihood, and which of its choices matter (:numref:`sec_diffusion-design-space`)? Why do current systems diffuse a compressed code with a transformer (:numref:`sec_diffusion-latent`)? Why does a single step fail, and what fixes it (:numref:`sec_diffusion-few-step`)? What is noise for a token (:numref:`sec_diffusion-discrete`)?

Along the way the four limits find answers. The model has a density: :numref:`sec_diffusion-design-space` computes it for the running example against its true value, and :numref:`sec_diffusion-discrete` bounds it in bits per pixel. One network and one regression replace the game (:numref:`sec_diffusion-gaussian-paths`), and :numref:`sec_diffusion-design-space` sorts the choices that remain, a schedule and a weighting, by how much they matter. The loss bounds the Kullback--Leibler divergence from the data to the model, so every missing mode costs, and :numref:`sec_diffusion-gaussian-paths` recovers a mixture's weights within sampling error. The same regression trains a transformer without a second network (:numref:`sec_diffusion-latent`), and what stays hard at scale is treated where it arises (:numref:`sec_diffusion-design-space`, :numref:`sec_diffusion-few-step`). The mathematics (stochastic differential equations, the probability-flow ODE, the proofs) lives in :numref:`chap_mdl-dynamics`, to which the chapter points instead of repeating it. Image generation at scale, text-to-image and video belong to a later part of this book.

```toc
:maxdepth: 2

denoising
gaussian-paths
image-diffusion
samplers
design-space
latent-diffusion
few-step
discrete-diffusion
further-reading
```

## Resources and Further Reading {.unnumbered}

These resources accompany particular sections; :numref:`sec_diffusion-further-reading` turns to the methods the chapter does not teach.

- [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) :cite:`Lai.Song.Kim.ea.2025` derives diffusion models along the variational, score-based and flow-based routes and shows where they meet, the unification on which the whole chapter rests.
- The [notes, lectures and labs](https://diffusion.csail.mit.edu/) of MIT's course 6.S184 :cite:`Holderrieth.Erives.2025` teach flow matching first, then latent spaces and transformer backbones, then discrete diffusion, the material of :numref:`sec_diffusion-gaussian-paths`, :numref:`sec_diffusion-latent` and :numref:`sec_diffusion-discrete`; its labs build each model from scratch.
- The [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) :cite:`Lipman.Havasi.Holderrieth.ea.2024` and its [library](https://github.com/facebookresearch/flow_matching) give the short training loop of :numref:`sec_diffusion-gaussian-paths` in full, with proofs and a reference implementation.
- [*Step-by-Step Diffusion*](https://arxiv.org/abs/2406.08929) :cite:`Nakkiran.Bradley.Zhou.ea.2025` builds the deterministic sampler from a single point mass up to any distribution, as :numref:`sec_diffusion-gaussian-paths` does, and derives the samplers of :numref:`sec_diffusion-samplers` without stochastic calculus.
- [*Diffusion Meets Flow Matching*](https://diffusionflow.github.io/) :cite:`Gao.Hoogeboom.Heek.ea.2025` shows with interactive figures that a diffusion sampler and a flow sampler are one algorithm under a change of schedule, and that the weighting matters more than the schedule, the themes of :numref:`sec_diffusion-samplers` and :numref:`sec_diffusion-design-space`.
- Sander Dieleman's [blog](https://sander.ai/) explains in long, careful posts how noise levels act on image frequencies :cite:`Dieleman.2024`, why generators work in a latent space :cite:`Dieleman.2025`, how flow maps learn the integral of a sampler :cite:`Dieleman.2026`, and how continuous and discrete diffusion compete for language :cite:`Dieleman.2026b`, companions to :numref:`sec_diffusion-design-space` through :numref:`sec_diffusion-discrete`.
- The [official DDPM implementation](https://github.com/hojonathanho/diffusion) of :citet:`ho2020denoising` and the [score SDE implementation](https://github.com/yang-song/score_sde_pytorch) of :citet:`song2021score` contain the full-scale networks, schedules and samplers that the small models of :numref:`sec_diffusion-images` imitate.
- The [MDLM repository](https://github.com/kuleshov-group/mdlm) of :citet:`Sahoo.Arriola.Schiff.ea.2024` trains the masked diffusion language model whose objective :numref:`sec_diffusion-discrete` derives.
- The [*Deep Generative Models* course](https://kuleshov-group.github.io/dgm-website/) :cite:`Kuleshov.2023` and Yang Song's [blog post](https://yang-song.net/blog/2021/score/) :cite:`Song.2021` tell the score-based history behind :numref:`sec_diffusion-denoising`: energy-based models, score matching and Langevin sampling at many noise levels.
- Lilian Weng's [*What Are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) :cite:`Weng.2021` and the [tutorial](https://arxiv.org/abs/2208.11970) of :citet:`Luo.2022` work through the variational algebra that :numref:`sec_diffusion-design-space` compresses into a few lines.
