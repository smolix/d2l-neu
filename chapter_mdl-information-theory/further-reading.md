# Further Reading
:label:`sec_mdl-information-theory-further-reading`

The chapter kept to what can be proved in a page and checked on a small
example, and each entry below starts where a section stopped. Books and papers
for what the chapter teaches are listed in its introduction
(:numref:`chap_mdl-information-theory`); these entries are for what it does
not. Read first the three that bear on practice: calibration, the constant of
a scaling law, and the bias and variance of mutual-information estimators.
Bits-back coding is worth reading for its own sake. The entries on capacity
and on information geometry can wait. No other page of the book builds on
this one.

## Entropy and Losses

### Calibration after Training

Over the population the log-loss is smallest at the true conditional
probabilities (:numref:`subsec_mdl-irreducible-loss`), which are calibrated.
A trained network is somewhere else, and
:citet:`Guo.Pleiss.Sun.Weinberger.2017` find that modern networks miss on one
side: they are overconfident. One-hot targets are a reason. They ask for an
infinite gap between the logits (:numref:`sec_mdl-information_theory`), so
once the training set is classified without error, the loss can fall only by
pushing the logits apart. Late in training the same authors see the held-out
log-loss rise again, although the held-out error does not: the network
overfits its probabilities and not its decisions. The temperature scaling of
:numref:`sec_classification` repairs this with one constant, fitted on
held-out data. It is the temperature of distillation, used for a different
purpose.

How much can such a repair achieve? Let $\bar{\mathbf{p}}(\mathbf{q})$ be the
frequencies of the classes among the inputs that receive the prediction
$\mathbf{q}$. A cross-entropy is an entropy plus a KL divergence, and this
holds for each prediction, so the held-out log-loss is

$$
\mathbb{E}\big[-\log q_y\big]
= \mathbb{E}_{\mathbf{q}}\big[D_{\textrm{KL}}\big(\bar{\mathbf{p}}(\mathbf{q}) \,\|\, \mathbf{q}\big)\big]
+ \mathbb{E}_{\mathbf{q}}\big[H\big(\bar{\mathbf{p}}(\mathbf{q})\big)\big].
$$

The first term is the miscalibration, in nats: it is zero only when the
classes occur as often as the predictions say. The second is the uncertainty
that remains about the label once the prediction is known. A map from
predictions to new predictions can remove the first term and cannot lower the
second. Temperature scaling is such a map with one parameter, and one is too
few for a model that is overconfident on some classes and underconfident on
others. Every proper scoring rule splits in the same way, with an entropy and
a divergence of its own :cite:`Gneiting.Raftery.2007`.

### Entropy Relative to a Reference

:numref:`subsec_mdl-differential-entropy` found that differential entropy
grows by $\ln 100$ when metres become centimetres, while a KL divergence
stays where it was, because both of its densities change by the same factor.
Entropy measured against a reference density $m$, as
$-\int p \log(p/m)\,dx$, has the form of $-D_{\textrm{KL}}(p\|m)$ and is
protected in the same way. The ordinary $h$ is this quantity with $m = 1$ in
the chosen units, and one per metre is not one per centimetre, so the
$\ln 100$ was a change of reference.
:citet:`Jaynes.1968` obtains the form as the limit of discrete entropies on
ever finer grids, and with it the maximum-entropy principle reads "stay as
close to the reference as the constraints allow". The KL penalty of a
variational autoencoder (:numref:`subsec_mdl-variational-bottleneck`) is, up
to its sign, such a relative entropy, with the prior as the reference, and so
it does not depend on the coordinates of the latent variable.

## Codes and Compression

### Compression without a Model

Every coder of :numref:`sec_mdl-coding-compression` is driven by a model
that predicts the next symbol. The Lempel–Ziv algorithms need none. LZ77
:cite:`Ziv.Lempel.1977` replaces a repeated substring by a pointer to its
previous occurrence, and LZ78 :cite:`Ziv.Lempel.1978` builds a dictionary of
phrases as it parses. Both are *universal*: on any stationary source whose
statistics one long text reveals (an ergodic source), their rate approaches
the entropy rate as the text grows, although nobody tells them what the
source is. The approach to that rate is slow. On *The Time Machine*, `zlib`,
which combines LZ77 with a Huffman code, needs $2.84$ bits per character, and
the counting model of :numref:`subsec_mdl-lm-compressor`, which predicts each
character from the three before it, needs $2.55$.

### Bits-Back Coding

A model with a latent variable $z$, such as a variational autoencoder, cannot
drive the arithmetic coder of :numref:`subsec_mdl-arithmetic-coding`. The
coder needs $p(x)$, and the model offers only the ELBO of
:numref:`sec_mdl-latent-em-elbo`, a lower bound on $\log p(x)$ computed with
an approximate posterior, which that section writes $q(z \mid x)$.
Surprisingly, the bound is a file length all the same, because decoding is
sampling. A decoder for $q(z \mid x)$ that is fed random bits returns a $z$
drawn from $q$ and uses up $\log_2 1/q(z \mid x)$ of them. In *bits-back
coding* the sender takes these bits from the end of the file that already
holds the earlier data points and that, being compressed, is as good as
random. The sender then appends $x$, coded with $p(x \mid z)$, and $z$, coded
with the prior $p(z)$. The receiver decodes $z$ and $x$ and then *encodes*
$z$ with $q(z \mid x)$, which puts the borrowed bits back. The file has grown
by $\log_2 1/p(x \mid z) + \log_2 1/p(z)$ bits and shrunk by
$\log_2 1/q(z \mid x)$, and on average over $q$ the net growth is the
negative ELBO. The receiver undoes the sender's steps last to first, so the
coder must work as a stack. The arithmetic coder on which
:citet:`Frey.Hinton.1997` built the scheme is not one. Asymmetric numeral
systems :cite:`Duda.2013` are, and with them
:citet:`Townsend.Bird.Barber.2019` compress MNIST with a small variational
autoencoder at a rate close to its negative ELBO.

### Description Length beyond Two-Part Codes

The two-part code of :numref:`subsec_mdl-mdl` spends bits on the parameter at
a chosen precision, so its length depends on the grid. A *prequential* code
sends no parameter. It codes each symbol with the model fitted to the symbols
before it, as the adaptive counting model of
:numref:`subsec_mdl-lm-compressor` did, and the cost of learning appears
inside the data term. :citet:`Blier.Ollivier.2018` code the training labels
of deep networks in this way and find the code far shorter than two-part
codes and than variational codes, which apply bits-back coding to the
weights. A large network, for all its parameters, does compress its labels.
:citet:`Grunwald.2007` develops these codes together with normalized maximum
likelihood, the code whose worst-case excess over the best model of the
class, chosen in hindsight, is smallest. Kolmogorov complexity, the length of
the shortest program that prints the data, is the idealization of all of
them, and it is not computable :cite:`Li.Vitanyi.2019`.

### The Constant of a Scaling Law

:numref:`sec_scaling-laws` fits the held-out loss of language models of many
sizes by a constant $E$ plus two power laws, one falling in the number of
parameters and one in the number of training tokens
:cite:`hoffmann2022training`, and reads $E$ as the entropy of the text. At
best $E$ is the entropy rate per token (:numref:`subsec_mdl-entropy-rate`),
below which no expected loss falls. The fitted value is a regression
constant, extrapolated from losses that all lie above it, and not a
measurement of that rate. :citet:`kaplan2020scaling` fit similar runs with no
constant at all, which must fail somewhere beyond their range, because a loss
that tends to zero ends below the floor. And $E$ is counted per token, so
constants fitted with different tokenizers cannot be compared until both are
converted to bits per byte (:numref:`subsec_bits-per-byte`).

## Comparing Distributions

### Rényi Divergences and Privacy

How much does a randomized algorithm reveal about one record when it is run
many times? Differential privacy measures the leak by a divergence between
the distributions of the output on two data sets that differ in that record.
Differentially private SGD adds noise to every gradient step and takes
thousands of steps, so the divergence must be one that adds up over them. The
Rényi divergence of order $\alpha$ :cite:`Renyi.1961`,

$$
D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\log\sum_x p(x)^\alpha q(x)^{1-\alpha},
$$

does. Its logarithm stands *outside* the sum, where an f-divergence has its
function inside. The sum factorizes over independent components, so
$D_\alpha$ adds over them, as KL does and $\chi^2$ and total variation do
not. Rényi differential privacy :cite:`Mironov.2017` bounds $D_\alpha$ for
one step, and the bound for a training run is the sum over its steps. KL adds
too, but it bounds the log-ratio $\log(p(x)/q(x))$ only on average.
$D_\alpha$ is nondecreasing in $\alpha$, from KL as $\alpha \to 1$ to the
largest value of the log-ratio as $\alpha \to \infty$, so a large order also
protects the record against the worst output.
:citet:`Van-Erven.Harremoes.2014` collect the limits and inequalities, and
Exercise :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`d of
:numref:`sec_mdl-divergences-distances` relates $D_\alpha$ to the
$\alpha$-divergences.

### Information Geometry

To second order, every smooth f-divergence between
$q_{\boldsymbol\theta + \mathbf{h}}$ and $q_{\boldsymbol\theta}$ is the
quadratic
$\tfrac12 f''(1)\, \mathbf{h}^\top \mathbf{I}(\boldsymbol\theta)\, \mathbf{h}$
in the Fisher information (:numref:`sec_mdl-divergences-distances`). That
section found this by a Taylor expansion. Chentsov's theorem says that it had
to come out so. Such a quadratic gives every small step $\mathbf{h}$ a
length, which makes it a Riemannian metric (it measures steps, where the
metrics of :numref:`sec_mdl-what-is-a-divergence` measure the distance
between two distributions). For distributions on finite sets the Fisher
information is, up to scale, the only Riemannian metric that is unchanged
when the data are replaced by a sufficient statistic :cite:`Cencov.1982`. No
f-divergence changes under that replacement: the data can be drawn again from
the statistic without knowing $\boldsymbol\theta$, so the data-processing
inequality of :numref:`subsec_mdl-f-div-data-processing` holds in both
directions. The quadratic of an f-divergence is therefore a multiple of the
Fisher information. Gradient descent with its steps measured in this metric
is the natural gradient, which K-FAC approximates with one Kronecker product
per layer (:numref:`subsec_mdl-preconditioning-ladder`), and the KL trust
region of :numref:`sec_ppo` is, to second order, a ball in it.
:citet:`Amari.2016` develops the geometry, in which maximum likelihood and
both steps of the EM algorithm (:numref:`sec_mdl-latent-em-elbo`) become
projections.

### Sinkhorn Divergences and Sliced Wasserstein

How is a transport distance used as a loss on minibatches in many dimensions?
Entropic regularization (:numref:`subsec_mdl-sinkhorn`) is one answer, and it
has a flaw: the regularized cost from a distribution to itself is not zero,
and a model that minimizes this cost ends up more concentrated than its data.
The Sinkhorn divergence of Exercise
:exref:`ex_mdl-divergences-distances-sinkhorn-divergences` in
:numref:`sec_mdl-test-functions-transport` subtracts the mean of the two
self-costs.
Beyond that exercise lies the theorem that makes the divergence a loss. On a
bounded set, with the Euclidean distance or its square as the cost, it is
nonnegative, zero only between equal distributions, and it tends to zero
if and only if the model converges in distribution to the data
:cite:`Feydy.Sejourne.Vialard.ea.2019`. The *sliced* Wasserstein distance is
the other answer. Projected onto one direction, two samples lie on a line,
where transport is solved by sorting (:numref:`sec_mdl-optimal-transport`),
and the sliced distance averages this cost over random directions
:cite:`Bonneel.Rabin.Peyre.ea.2015`. In many dimensions a random direction
seldom shows where two distributions differ, and the image generators of
:citet:`Deshpande.Zhang.Schwing.2018` therefore train a discriminator and
project its features in place of the pixels.

### Lipschitz Critics in Wasserstein GANs

The Kantorovich–Rubinstein dual of :numref:`sec_mdl-optimal-transport` needs a
1-Lipschitz function, and a network is not one by construction.
:numref:`sec_gan-further-reading` compares the three ways of making it one,
weight clipping, spectral normalization and a gradient penalty, and
:numref:`sec_gan_convergence` treats the penalties. The penalty of
:citet:`Gulrajani.Ahmed.Arjovsky.ea.2017` asks for a gradient of norm one at
points between a real and a generated sample, because an optimal critic has
unit slope along the lines on which transport moves mass (Exercise
:exref:`ex_mdl-test-functions-transport-optimal-critics`d of
:numref:`sec_mdl-test-functions-transport`).

### Stein Variational Gradient Descent

Sampling a model that is known only through its score $s_Q$ takes a long
random walk (:numref:`sec_mdl-scores`). The kernel Stein discrepancy of
:numref:`sec_mdl-ksd` measures how far a sample from $P$ is from such a
model $Q$. Stein variational gradient descent :cite:`Liu.Wang.2016` moves the
sample there instead. Starting from any set of points, with distribution $P$,
each point $y$ takes small steps along the velocity

$$
v(y) = \mathbb{E}_{x \sim P}\big[k(x, y)\, s_Q(x) + \nabla_x k(x, y)\big].
$$

The first term is the kernel-weighted average of the scores near $y$, and it
carries $y$ toward higher density. The second pushes $y$ away from its
neighbours and does the work of the noise in the random walk: without it all
points would gather at a mode. Scaled to the unit ball of the feature space
of :numref:`sec_mdl-ksd`, this velocity decreases $D_{\textrm{KL}}(P\|Q)$
faster than any other in the ball, at the rate
$\|v\| = \mathrm{KSD}(P, Q)$. The same Stein operator serves as a test
statistic and as a velocity field. The points still take many small steps,
but the steps are deterministic and the points arrive together, which suits
a posterior that a few dozen points must represent, as for the weights of a
small network in the original paper.

## Mutual Information

### Codes That Approach Capacity

:numref:`subsec_mdl-channel-capacity` states that every rate below capacity
can be reached, but Shannon's random codes take exponential time to decode.
Low-density parity-check codes, decoded by passing messages on the graph of
the code, come close to capacity in practice :cite:`Richardson.Urbanke.2008`,
and polar codes :cite:`Arikan.2009` provably achieve it on symmetric
binary-input channels, with encoding and decoding in $O(n \log n)$ operations
for block length $n$. A 5G phone uses both.

### Learned Lossy Compression

:numref:`subsec_mdl-rate-distortion` computes the rate–distortion function
of a Gaussian source. Nobody knows that function for photographs, and a
learned image codec is a point somewhere above it.
:citet:`Balle.Laparra.Simoncelli.2017` train three parts together: an
encoder, a decoder, and an *entropy model*, a learned distribution over the
rounded latent. Rounding has no gradient, so uniform noise stands in for it
during training. The rate $R$ in their loss is the arithmetic-coding length
of the latent under the entropy model
(:numref:`sec_mdl-coding-compression`), and $D$ is the distortion of the
decoded image. Minimizing $R + \lambda D$ picks the point of the codec's
curve at which its slope is $-\lambda$.
:citet:`Balle.Minnen.Singh.ea.2018` make the entropy model conditional on a
second latent, a *hyperprior*, which lowers the rate at equal distortion.

## Estimating Mutual Information

### Upper Bounds

Every variational estimator of :numref:`sec_mdl-mi-estimation` is a lower
bound, so none can certify that a representation keeps *little* information,
for instance about a protected attribute that it should have forgotten. An
upper bound needs the conditional density $p(z \mid x)$, which a stochastic
encoder supplies. The variational bottleneck of
:numref:`subsec_mdl-variational-bottleneck` is one such bound, with a fixed
density $r(z)$ in place of the marginal. The contrastive log-ratio upper
bound, CLUB :cite:`Cheng.Hao.Dai.ea.2020`, needs no $r$: it averages
$\log p(z \mid x)$ over the matched pairs of a batch and subtracts its
average over the mismatched pairs. It is an upper bound in expectation when
the density is the encoder's own. Between a representation and an attribute
there is no such density, and a network is fitted in its place. The guarantee
then holds on one condition: the fitted density must lie closer in KL
divergence to the joint distribution than to the product of the marginals.
That cannot be checked without the density that was missing.

### Bias and Variance of the Bounds

With an exact critic NWJ is right on average but noisy, and InfoNCE is steady
but capped at $\log N$ (:numref:`subsec_mdl-exact-critics`).
:citet:`Poole.Oord.Alemi.ea.2019` interpolate between the two with one mixing
parameter, which selects a point of the trade-off. :citet:`Song.Ermon.2020b`
prove that the noise is no accident of one estimator: for the NWJ and DV
bounds the variance grows exponentially with the true information, even with
an exact critic. Their repair clips the critic's density ratio, which lowers
the variance and introduces a bias. Every repair on offer moves error from
variance to bias, and none removes it.

The same authors propose tests that need no known answer: an estimate must
not change when a function of one variable is appended to that variable,
and it must double when the pair is replaced by two independent copies of
itself. On images the estimators they try fail these tests. Check a reported
mutual information of several nats against the logarithm of the batch size,
and trust an estimator on your own data only after it has passed such a
test there.

### Noisy Networks and the Bottleneck View

The debate of :numref:`subsec_mdl-compression-debate` began with the proposal
of :citet:`Tishby.Zaslavsky.2015` to judge each layer against the bottleneck
frontier of :numref:`sec_mdl-information-bottleneck`, the largest $I(Y; Z)$
that can be had at each $I(X; Z)$. That subsection left the compression phase
with nothing to measure: in a deterministic network on continuous inputs,
$I(X; Z)$ is infinite. :citet:`Goldfeld.Berg.Greenewald.ea.2019` make the
question well posed by adding Gaussian noise to the activations, which makes
$I(X; Z)$ finite. It can then fall during training, and they trace the fall
to clustering: the representations of each class draw together. This, they
argue, is also what the binned estimates on deterministic networks had
measured.

[Discussions](https://d2l.discourse.group/t/36867)
