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
gradients survive when supports separate.
:numref:`sec_gan_relativistic` changes the critic's input from one sample to a
real--generated pair and computes the value of the resulting relativistic
objective in closed form. :numref:`sec_gan_convergence` then explains why
gradient descent can fail even when the objective has the correct optimum, and
which regularizers restore convergence. :numref:`sec_dcgan` applies these
results to image generation and evaluates the trained models.
:numref:`sec_gan_conditional` supplies a class, caption, or image to both
networks and derives the projection discriminator from the conditional density
ratio.
:numref:`sec_gan_beyond` examines the roles that adversarial losses retain in
current systems and reconnects the central density-ratio identity to
likelihood-based models.

Table :numref:`tab_gan_games` summarizes the chapter's analytical results. It
lists each adversarial game, the critic's input, and the value attained by the
optimal critic. This last quantity is what the generator minimizes. Here $p$
denotes the data distribution, $q$ the generator distribution, and
$\mathrm{JS}$ the Jensen--Shannon divergence; :numref:`sec_basic_gan` defines
all three.

:The games of this chapter. Each row gives an objective, what its critic scores, and the value of the game at the optimal critic.
:label:`tab_gan_games`

| objective | the critic scores | value at the optimal critic | where |
|:--|:--|:--|:--|
| log loss, the original GAN | one sample's realness logit | $2\,\mathrm{JS}(p, q) - 2 \log 2$, attained at the logit $\log(p/q)$ | :numref:`sec_basic_gan` |
| non-saturating generator loss | the same critic; the generator maximizes its samples' scores | the same minimizer $q = p$ under an optimal critic; confidently rejected samples receive large weights when the critic retains a nonzero score gradient | :numref:`sec_basic_gan` |
| proper classification losses and $f$-GAN | one sample, under other classification losses | an $f$-divergence: an average of a convex function of the ratio $p/q$ | :numref:`sec_gan_objectives` |
| integral probability metrics: MMD, Wasserstein | one sample, scored by a critic from a constrained class | the largest expectation gap the class can certify; geometry-sensitive classes remain informative as supports separate, while a bounded kernel loses sensitivity beyond its length scale | :numref:`sec_gan_objectives` |
| relativistic pairing, RpGAN | a real--generated pair, rewarded for ranking the real member higher | $\mathrm{JS}(p \otimes q,\, q \otimes p)$: the information a pair carries about which member is real | :numref:`sec_gan_relativistic` |
| zero-centered gradient penalties $R_1$, $R_2$ | any of the above, with a penalty on the critic's input gradients | near equilibrium, a squared linearized Wasserstein-2 distance and dynamics that converge for every positive penalty weight | :numref:`sec_gan_convergence` |
| the conditional game | a sample together with the condition it claims to satisfy | the average over conditions, $E_c\big[2\,\mathrm{JS}(p(\cdot \mid c), q(\cdot \mid c))\big] - 2\log 2$; shared log-linear class posteriors motivate the projection discriminator | :numref:`sec_gan_conditional` |

:numref:`sec_dcgan` and :numref:`sec_gan_beyond` add no further games. The
first assembles the penalized relativistic loss into an image generator and
evaluates it. The second identifies the same objectives in current one-step
generators, tokenizers, and vocoders.

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

The chapter derives its objectives in closed form. The following resources
provide historical context, interactive demonstrations, and full-scale
implementations.

- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160) :cite:`Goodfellow.2016` presents the original intuition and surveys the state of the field in 2016. It complements the closed-form treatment in this chapter.
- [GAN Lab](https://poloclub.github.io/ganlab/) :cite:`Kahng.Thorat.Chau.ea.2018` implements the two-dimensional games of :numref:`sec_basic_gan` and :numref:`sec_gan_objectives` in a browser. It displays the critic's decision surface, the generator distribution, and several failure modes during training.
- The [GAN stability repository](https://github.com/LMescheder/GAN_stability) accompanying :citet:`Mescheder.Geiger.Nowozin.2018` contains reference implementations of the Dirac-GAN analysis and the $R_1$/$R_2$ penalties from :numref:`sec_gan_convergence`.
- The [R3GAN reference implementation](https://github.com/brownvc/R3GAN) :cite:`Huang.Gokaslan.Kuleshov.ea.2024` implements the loss and backbone used in :numref:`sec_gan_convergence` and :numref:`sec_dcgan`, with full-scale configurations for FFHQ and ImageNet.
- [StyleGAN3](https://github.com/NVlabs/stylegan3) :cite:`Karras.Aittala.Laine.ea.2021` provides a mature implementation of the StyleGAN architecture, along with extensive training diagnostics and metric tools.
- The large-scale study of :citet:`Lucic.Kurach.Michalski.ea.2018` shows that many GAN objectives attain similar scores when given equal tuning budgets. This finding motivates the chapter's emphasis on regularization and evaluation rather than a catalogue of objectives.
- [ganhacks](https://github.com/soumith/ganhacks) records a widely used training checklist from 2016. Comparing it with :numref:`sec_dcgan` shows which empirical stabilizers were later replaced by explicit regularization.
