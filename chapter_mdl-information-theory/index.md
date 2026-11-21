# Information Theory and Divergences
:label:`chap_mdl-information-theory`

Most models in this book were trained by minimizing a cross-entropy loss. We
chose it because it works: it turns the informal goal "assign high probability
to what actually happened" into a smooth number that gradient descent can push
down. Yet we have never stopped to ask what that number actually means. When a
language model reports a loss of 1.2 nats per character, is that good? How low
could it possibly go? And when we compare two distributions in other ways, such
as with a KL penalty in a variational autoencoder or a discriminator in a
generative adversarial network, what are we really measuring?

Information theory answers these questions, though it began with a rather
different one. In 1948, Claude Shannon asked how efficiently a message can be
sent over a telephone line, and discovered that the answer depends on a single
property of the source: how predictable it is. His measure of
unpredictability, *entropy*, captures the average surprise of an outcome. It
also turns out to be the hard floor below which no model's cross-entropy can
ever fall.

The main text states this floor without proof
(:numref:`subsec_info_theory_basics`). This chapter proves it and develops
three core ideas that are worth carrying back to the rest of the book. First,
a model's cross-entropy exceeds the data's entropy by the *KL divergence*
between the data and the model, a gap that vanishes only when the model
matches the data. Second, if you measure a model's total loss on a message in
bits, you can actually store the message in that many bits plus at most two;
in this sense, the loss of a language model (:numref:`subsec_perplexity`) is
literally a file size. Third, a discriminator provides only a lower bound on
how far apart two distributions are, so to a weak discriminator even distant
distributions can look close. Both adversarial training (:numref:`sec_gan_objectives`) and contrastive
learning rest on bounds of this kind. In contrastive learning, for example,
identifying the true partner of an example among $N$ candidates can certify at
most $\log N$ nats of shared information.

The chapter unfolds in three parts. The first asks what cross-entropy measures
(:numref:`sec_mdl-information_theory`) and why it corresponds to the length of
a compressed file (:numref:`sec_mdl-coding-compression`). The second part
starts from a practical flaw: the KL divergence shoots to infinity the moment
a model assigns zero probability to an event that actually occurs. Even worse,
whenever a model's samples fail to overlap with the data at all, no divergence
built from the density ratio can tell a near miss from a wildly bad one
(:numref:`sec_mdl-divergences-distances`). Two families of alternatives fix
this: comparing averages of test functions under the two distributions, or
measuring the cost of transporting probability mass from one to the other
(:numref:`sec_mdl-test-functions-transport`). Finally, if a model's density is
known only up to an intractable normalizing constant, we can still compare it
to data through the slope of its log-density (:numref:`sec_mdl-scores`).

The third part shifts attention to pairs of random variables: how much one
variable tells us about another (:numref:`sec_mdl-mutual-information`), which
is the quantity controlled by the KL penalty of a variational autoencoder, and
how much of that shared information we can hope to estimate from finite
samples (:numref:`sec_mdl-mi-estimation`). This final part does not depend on
the two sections immediately preceding it, so you can safely save those until
:numref:`chap_gans` or :numref:`chap_diffusion` brings you back to them.

We assume familiarity with Jensen's inequality
(:numref:`chap_mdl-optimization`), conditional distributions, and maximum
likelihood (:numref:`chap_mdl-probability-statistics`). Everything else,
starting with entropy and the KL divergence, is built from scratch. Topics
outside our main path—such as practical error-correcting codes, Kolmogorov
complexity, and information geometry—are gathered in
:numref:`sec_mdl-information-theory-further-reading`.

```toc
:maxdepth: 2

mdl-information-theory
mdl-coding-compression
mdl-divergences-distances
mdl-test-functions-transport
mdl-scores
mdl-mutual-information
mdl-mi-estimation
further-reading
```
