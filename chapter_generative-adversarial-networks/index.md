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
