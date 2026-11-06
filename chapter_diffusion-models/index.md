# Diffusion Models
:label:`chap_diffusion`

The generators of :numref:`chap_gans` learned to draw garments and sprites through a game between two networks, and that chapter closed with four limits the game leaves in place: no density, two networks and a tuned penalty, no cost for a missing mode, and instability at scale. It pointed instead to diffusion and flow models, trained by regression on a target computed from each training example. A classifier regresses onto a given label; a generator has nothing to copy, and its image must come out of noise. Regression onto what, then, and how can fitting a target produce something new?

The idea came from physics: in non-equilibrium thermodynamics, structure such as a drop of ink in water is destroyed by many small random steps, and a small step's reversal has the same simple form. :citet:`sohl2015deep` noised images in the same way and trained a network to undo each step; :citet:`ho2020denoising` made the undoing a regression onto the added noise and matched the sample quality of adversarial training. Others reached the same model from the gradient of the log-density :cite:`song2019generative` and from the velocity of a path between noise and data :cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022,Albergo.Boffi.VandenEijnden.2023`. All routes arrive at one network that predicts the clean point from a noisy one at every noise level, and a sampler that integrates its predictions.

What does a network that removes noise know about the data (:numref:`sec_diffusion-denoising`)? How do we walk from noise to data, and which regression trains the walk (:numref:`sec_diffusion-gaussian-paths`)? What changes for images, and how do we steer and judge them (:numref:`sec_diffusion-images`)? Which sampling steps can we skip, and what does a sampler do when run backwards (:numref:`sec_diffusion-samplers`)? Is the regression maximum likelihood, and which of its choices matter (:numref:`sec_diffusion-design-space`)? Why do current systems diffuse a compressed code with a transformer (:numref:`sec_diffusion-latent`)? Why does a single step fail, and what fixes it (:numref:`sec_diffusion-few-step`)? What is noise for a token (:numref:`sec_diffusion-discrete`)?

Along the way the four limits of the game find answers: a density, one regression instead of two networks, a loss under which every missing mode costs, and a transformer trained stably at scale. The mathematics lives in :numref:`chap_mdl-dynamics`, to which the chapter points instead of repeating it; image generation at scale, text-to-image and video belong to a later part of this book.

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

- [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) :cite:`Lai.Song.Kim.ea.2025` derives diffusion models along the variational, score-based and flow-based routes and shows where they meet.
- The [notes, lectures and labs](https://diffusion.csail.mit.edu/) of MIT's course 6.S184 :cite:`Holderrieth.Erives.2025` teach flow matching, latent spaces with transformer backbones, and discrete diffusion, with labs that build each model from scratch.
- The [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) :cite:`Lipman.Havasi.Holderrieth.ea.2024` and its [library](https://github.com/facebookresearch/flow_matching) give the training loop of :numref:`sec_diffusion-gaussian-paths` in full, with proofs.
- [*Step-by-Step Diffusion*](https://arxiv.org/abs/2406.08929) :cite:`Nakkiran.Bradley.Zhou.ea.2025` builds the deterministic sampler up from a single point mass and derives the samplers of :numref:`sec_diffusion-samplers` without stochastic calculus.
- [*Diffusion Meets Flow Matching*](https://diffusionflow.github.io/) :cite:`Gao.Hoogeboom.Heek.ea.2025` shows with interactive figures that diffusion and flow samplers are one algorithm under a change of schedule, and that the weighting matters more than the schedule.
- Sander Dieleman's [blog](https://sander.ai/) explains how noise acts on image frequencies :cite:`Dieleman.2024`, why generators work in a latent space :cite:`Dieleman.2025`, what flow maps learn :cite:`Dieleman.2026`, and how continuous and discrete diffusion compete for language :cite:`Dieleman.2026b`.
- The [official DDPM implementation](https://github.com/hojonathanho/diffusion) of :citet:`ho2020denoising` and the [score SDE implementation](https://github.com/yang-song/score_sde_pytorch) of :citet:`song2021score` contain the full-scale networks and samplers that the small models of :numref:`sec_diffusion-images` imitate.
- The [MDLM repository](https://github.com/kuleshov-group/mdlm) of :citet:`Sahoo.Arriola.Schiff.ea.2024` trains the masked diffusion language model of :numref:`sec_diffusion-discrete`.
- The [*Deep Generative Models* course](https://kuleshov-group.github.io/dgm-website/) :cite:`Kuleshov.2023` and Yang Song's [blog post](https://yang-song.net/blog/2021/score/) :cite:`Song.2021` tell the score-based history behind :numref:`sec_diffusion-denoising`.
- Lilian Weng's [*What Are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) :cite:`Weng.2021` and the [tutorial](https://arxiv.org/abs/2208.11970) of :citet:`Luo.2022` work through the variational algebra that :numref:`sec_diffusion-design-space` compresses.
