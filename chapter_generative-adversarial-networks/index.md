# Generative Adversarial Networks
:label:`chap_gans`

A generative model is, operationally, a sampler: draw a latent variable, push
it through a network, and out comes an image, a waveform, a row of a table.
The models of this chapter take that description literally. They consist of
nothing but the sampler. There is no density attached to the samples, and so
no way to evaluate how probable any particular example is. Giving up the
density buys unrestricted architectures and single-pass generation, and it
costs the training principle used everywhere else in this book: maximum
likelihood, which :numref:`subsec_mdl-nll-crossentropy` showed is the
minimization of a Kullback--Leibler divergence that the model can evaluate.

What replaces the likelihood is a comparison. A second network is trained to
distinguish generated samples from real ones, and the generator is trained
against its verdict. Everything that matters about the resulting family of
methods follows from how that comparison is set up: the loss the comparing
network minimizes, the class of functions it may use, and whether it scores
samples one at a time or in pairs. These choices determine which divergence
the game evaluates at its optimum, and — the recurring question of this
chapter — whether the generator receives a gradient it can follow when its
samples are still far from the data.

Stand-alone adversarial generators no longer sit at the frontier of image
synthesis; that place has passed to the diffusion and flow models of
:numref:`chap_diffusion`. The adversarial *objective*, however, ships today
inside one-step image generators, the tokenizers behind latent diffusion
systems, and neural vocoders, and the training instabilities that once made
it a specialist's tool are now understood well enough to be repaired with a
short, analyzable recipe. Both facts are recent, and both are taught here.

The chapter proceeds by dependency. :numref:`sec_basic_gan` analyzes one game
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
:numref:`sec_gan_beyond` closes with where adversarial losses operate in
current systems, and with the identity that connects the chapter back to
maximum likelihood.

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
