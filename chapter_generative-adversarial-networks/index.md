# Generative Adversarial Networks
:label:`chap_gans`

A generative model can be used as a sampler: draw a latent variable and pass
it through a network to obtain an image, a waveform, or a row of a table. The
models in this chapter provide this sampling operation but no tractable
density for their outputs. They can use flexible architectures and generate a
sample in one forward pass, but they cannot be trained by maximum likelihood.
As :numref:`subsec_mdl-nll-crossentropy` showed, maximum likelihood requires a
model density with which to evaluate a Kullback--Leibler divergence.

Adversarial training replaces likelihood evaluation with a learned
comparison. A second network distinguishes generated samples from real ones,
and the generator is trained to make that distinction difficult. Three design
choices determine the resulting objective: the comparison loss, the class of
functions available to the critic, and whether the critic scores individual
samples or pairs. These choices determine which divergence the game evaluates
at its optimum and whether the generator receives a useful gradient when its
samples are far from the data.

Diffusion and flow models, developed in :numref:`chap_diffusion`, have largely
replaced stand-alone GANs at the frontier of image synthesis. Adversarial
objectives nevertheless remain important in one-step image generators, the
tokenizers used by latent generative models, and neural vocoders. This chapter
therefore treats both the classical theory and the regularization techniques
that make modern adversarial training substantially more reliable.

The sections follow the logical dependencies of the analysis.
:numref:`sec_basic_gan` analyzes one game
exactly: the original logistic-loss objective, its optimal discriminator, and
the divergence it evaluates. :numref:`sec_gan_objectives` varies the design
choices and maps the resulting space of objectives, including the ones whose
gradients survive when supports separate. :numref:`sec_gan_relativistic`
changes what the critic scores — pairs instead of samples — and computes the
value of the modern relativistic objective in closed form.
:numref:`sec_gan_convergence` turns from objectives to dynamics: why gradient
descent can fail on a correct objective, and the regularization that restores
convergence. :numref:`sec_dcgan` puts the pieces to work on images and
measures the results. :numref:`sec_gan_conditional` extends the game to
conditions: a class, a caption, or an image enters both networks, and the
projection discriminator falls out of the chapter's central identity.
:numref:`sec_gan_beyond` examines the roles that adversarial losses retain in
current systems and reconnects the central density-ratio identity to
likelihood-based models.

The chapter's analytical results fit in one table, :numref:`tab_gan_games`:
each adversarial game, what
its critic scores, and the value of the game once the critic plays its best
response — the quantity the generator actually minimizes. Here $p$ is the data
distribution, $q$ the generator's, and $\mathrm{JS}$ the Jensen--Shannon
divergence; :numref:`sec_basic_gan` defines all three precisely.

:The games of this chapter. Each row gives an objective, what its critic scores, and the value of the game at the optimal critic.
:label:`tab_gan_games`

| objective | the critic scores | value at the optimal critic | where |
|:--|:--|:--|:--|
| log loss, the original GAN | one sample's realness logit | $2\,\mathrm{JS}(p, q) - 2 \log 2$, attained at the logit $\log(p/q)$ | :numref:`sec_basic_gan` |
| non-saturating generator loss | the same critic; the generator maximizes its samples' scores | the same fixed point $q = p$, reached with a gradient that survives a confident critic | :numref:`sec_basic_gan` |
| proper classification losses and $f$-GAN | one sample, under other classification payoffs | an $f$-divergence: an average of a convex function of the ratio $p/q$ | :numref:`sec_gan_objectives` |
| integral probability metrics: MMD, Wasserstein | one sample, by a critic from a constrained class | the largest expectation gap the class can certify — finite and informative even for disjoint supports | :numref:`sec_gan_objectives` |
| relativistic pairing, RpGAN | a real--generated pair, rewarded for ranking the real member higher | $\mathrm{JS}(p \otimes q,\, q \otimes p)$: the information a pair carries about which member is real | :numref:`sec_gan_relativistic` |
| zero-centered gradient penalties $R_1$, $R_2$ | any of the above, with the critic's input gradients penalized | near equilibrium, a squared linearized Wasserstein-2 distance — and dynamics that converge for every penalty weight | :numref:`sec_gan_convergence` |
| the conditional game | a sample together with the condition it claims to satisfy | the average over conditions, $E_c\big[2\,\mathrm{JS}(p(\cdot \mid c), q(\cdot \mid c))\big] - 2\log 2$; its optimal critic yields the projection discriminator | :numref:`sec_gan_conditional` |

:numref:`sec_dcgan` and :numref:`sec_gan_beyond` add no further games: the
first assembles the penalized relativistic loss into a working image generator
and measures it, and the second locates the same objectives inside current
one-step generators, tokenizers, and vocoders.

Readers who want the working recipe first can read :numref:`sec_basic_gan`,
then the recipe and experiment of :numref:`sec_gan_convergence` and
:numref:`sec_dcgan`, returning to :numref:`sec_gan_objectives` and
:numref:`sec_gan_relativistic` for the theory that explains why the recipe
looks the way it does. Otherwise the sections build in order, each on the
results of the last.

```toc
:maxdepth: 2

gan
objectives
relativistic
convergence
dcgan
conditional
adversarial-losses
```

## Resources and Further Reading {.unnumbered}

The chapter derives its objectives in closed form; the resources below supply
the history, the interactive intuition, and the full-scale implementations.

- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160) :cite:`Goodfellow.2016` — the inventor's book-length tutorial: the intuition and the state of the art at the moment the field formed, a useful companion to this chapter's closed-form treatment.
- [GAN Lab](https://poloclub.github.io/ganlab/) :cite:`Kahng.Thorat.Chau.ea.2018` — the two-dimensional games of :numref:`sec_basic_gan` and :numref:`sec_gan_objectives` running live in the browser: watch the critic's decision surface and the generator's samples co-evolve, failure modes included.
- The [GAN stability repository](https://github.com/LMescheder/GAN_stability) accompanying :citet:`Mescheder.Geiger.Nowozin.2018` — reference implementations of the Dirac-GAN analysis and the $R_1$/$R_2$ penalties of :numref:`sec_gan_convergence`.
- The [R3GAN reference implementation](https://github.com/brownvc/R3GAN) :cite:`Huang.Gokaslan.Kuleshov.ea.2024` — the loss and backbone of :numref:`sec_gan_convergence` and :numref:`sec_dcgan` at full scale, with configurations for FFHQ and ImageNet.
- [StyleGAN3](https://github.com/NVlabs/stylegan3) :cite:`Karras.Aittala.Laine.ea.2021` — the engineered peak of the architecture line that :numref:`sec_dcgan` recounts; its training diagnostics and metric tooling repay reading whatever loss you train with.
- The large-scale study of :citet:`Lucic.Kurach.Michalski.ea.2018` — many GAN objectives reach similar scores once tuning budgets are equalized; the empirical argument for this chapter's emphasis on regularization and evaluation over accumulating objectives.
- [ganhacks](https://github.com/soumith/ganhacks) — the folklore-era training checklist, frozen in 2016; read it against the decade-of-patches table of :numref:`sec_dcgan` to see what the closed-form regularization of :numref:`sec_gan_convergence` replaced.
