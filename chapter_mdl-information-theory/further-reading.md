# Further Reading
:label:`sec_mdl-information-theory-further-reading`

The main sections of this chapter stick to results that we can prove cleanly
and check on small examples. Each short note below picks up where a section
left off. If you are looking for books and papers on the topics already taught
in the chapter, you will find them in the introduction
(:numref:`chap_mdl-information-theory`); this page covers neighboring ideas
that did not make the core path. If you are short on time, start with the
three topics that show up most directly in everyday practice: model
calibration, the irreducible-loss constant in scaling laws, and the
bias--variance trade-off of mutual-information estimators. Bits-back coding is
also well worth reading simply because the trick is so clever. Channel
capacity and information geometry can easily wait, and nothing later in the
book depends on this page.

## Entropy and Losses

### Calibration after Training

In principle, minimizing the population log-loss recovers the true conditional
class probabilities (:numref:`subsec_mdl-irreducible-loss`), which are
automatically calibrated. Trained neural networks, however, rarely sit at that
optimum. As :citet:`Guo.Pleiss.Sun.Weinberger.2017` famously observed, modern
networks tend to err in one particular direction: they are overconfident.
One-hot training labels are a major culprit. Because they reward pushing the
winning logit infinitely far ahead of the rest
(:numref:`sec_mdl-information_theory`), once a network classifies the training
set perfectly, the only way it can keep lowering the loss is to push its
logits further and further apart. Late in training, the same authors see
held-out log-loss begin to climb even while held-out classification error
stays flat: the network is overfitting its probability estimates rather than
its decisions. Temperature scaling (:numref:`sec_classification`) fixes much
of this with a single scalar fitted on a held-out validation set. It is the
same temperature parameter used in distillation, repurposed here to soften
overconfident predictions.

How far can a post-hoc recalibration like this go? Let
$\bar{\mathbf{p}}(\mathbf{q})$ denote the actual empirical class frequencies
among all inputs for which the model outputs the probability vector
$\mathbf{q}$. Since any cross-entropy decomposes into an entropy plus a KL
divergence, applying that identity conditionally at each predicted vector
$\mathbf{q}$ splits the held-out log-loss into two pieces:

$$
\mathbb{E}\big[-\log q_y\big]
= \mathbb{E}_{\mathbf{q}}\big[D_{\textrm{KL}}\big(\bar{\mathbf{p}}(\mathbf{q}) \,\|\, \mathbf{q}\big)\big]
+ \mathbb{E}_{\mathbf{q}}\big[H\big(\bar{\mathbf{p}}(\mathbf{q})\big)\big].
$$

The first term measures miscalibration in nats: it vanishes if and only if
events actually happen as often as the model's predicted probabilities claim.
The second term is the uncertainty that still remains about the true label
even after we know the model's prediction. Any post-hoc remapping of the
predicted probabilities can drive the first term toward zero, but it can never
reduce the second. Temperature scaling is simply a one-parameter family of
such remappings, which makes it wonderfully simple, though a single scalar
cannot fix a model that is overconfident on some classes and underconfident on
others. More generally, every proper scoring rule admits an analogous
decomposition into a divergence term and an entropy term
:cite:`Gneiting.Raftery.2007`.

### Entropy Relative to a Reference

In :numref:`subsec_mdl-differential-entropy`, we saw an awkward quirk of
differential entropy: switching units from meters to centimeters shifts it up
by $\ln 100$. A KL divergence, by contrast, does not budge, because both
densities inside the ratio rescale by the same factor. We can give entropy the
same coordinate invariance by measuring it against a reference density $m$,
defining $-\int p \log(p/m)\,dx = -D_{\textrm{KL}}(p\|m)$. Ordinary
differential entropy $h$ is just the special case $m = 1$ in whatever units we
happened to pick. Because a uniform density of one per meter is not the same
measure as one per centimeter, that $\ln 100$ shift was really a hidden change
of reference. :citet:`Jaynes.1968` derives this relative form as the limit of
discrete entropies on increasingly fine grids; with it, the maximum-entropy
principle takes the natural form "stay as close to the reference measure as
the constraints allow." Up to a sign, the KL penalty in a variational
autoencoder (:numref:`subsec_mdl-variational-bottleneck`) is an entropy of
this kind with the prior playing the role of $m$, which is why it remains
invariant under smooth reparameterizations of the latent space.

## Codes and Compression

### Compression without a Model

Every compressor in :numref:`sec_mdl-coding-compression` relies on an explicit
probabilistic model to predict the next symbol. The Lempel–Ziv family does
away with that requirement altogether. LZ77 :cite:`Ziv.Lempel.1977` replaces
any repeated substring with a short pointer back to an earlier occurrence,
while LZ78 :cite:`Ziv.Lempel.1978` builds up a dictionary of phrases on the
fly as it scans the file. Remarkably, both algorithms are *universal*: for any
stationary ergodic source (roughly, one whose long-run statistics are revealed
by a single long sample), their average code length approaches the true entropy
rate as the file grows, even though nobody tells them anything about the
source in advance. In practice, however, convergence to that limit is slow. On
*The Time Machine*, `zlib` (which pairs LZ77 with Huffman coding) uses $2.84$
bits per character, whereas the simple counting model of
:numref:`subsec_mdl-lm-compressor`, which predicts each character from the
three characters before it, already gets down to $2.55$.

### Bits-Back Coding

A latent-variable model such as a variational autoencoder cannot directly
drive the arithmetic coder of :numref:`subsec_mdl-arithmetic-coding`. The
coder needs the exact marginal probability $p(x)$, whereas the model only
gives us the ELBO (:numref:`sec_mdl-latent-em-elbo`), a lower bound on
$\log p(x)$ built from an approximate posterior $q(z \mid x)$. Surprisingly,
that lower bound still corresponds to an achievable file length, thanks to a
clever duality: decoding a compressed stream is mathematically the same as
sampling. If you feed random bits into a decoder designed for $q(z \mid x)$,
it consumes roughly $\log_2 1/q(z \mid x)$ bits and outputs a latent sample
$z$ drawn from $q$.

*Bits-back coding* turns this observation into a practical scheme. Suppose the
sender already has a compressed file holding earlier data points; because good
compression removes redundancy, the bits at the end of that file look
indistinguishable from random coin flips. To encode a new data point $x$, the
sender borrows $\log_2 1/q(z \mid x)$ bits from the end of the file and
"decodes" them with $q(z \mid x)$ to draw a latent $z$. The sender then
encodes $x$ using $p(x \mid z)$ and $z$ using the prior $p(z)$, appending both
to the file. To unpack $x$, the receiver reverses the process: it decodes $z$
from the prior, decodes $x$ from $p(x \mid z)$, and then *re-encodes* $z$
using $q(z \mid x)$, which puts the borrowed bits right back where they came
from. Overall, the file grew by $\log_2 1/p(x \mid z) + \log_2 1/p(z)$ bits
and shrank by $\log_2 1/q(z \mid x)$ bits, so its expected net growth under
$q$ is the negative ELBO. Because the receiver must undo the sender's
operations in last-in, first-out order, the underlying coder has to behave
like a stack. Traditional arithmetic coding, on which
:citet:`Frey.Hinton.1997` originally proposed the idea, behaves like a queue
instead. Asymmetric numeral systems :cite:`Duda.2013` are naturally
stack-based, and using them :citet:`Townsend.Bird.Barber.2019` demonstrated
compressing MNIST with a small variational autoencoder at rates right near its
negative ELBO.

### Description Length beyond Two-Part Codes

The two-part code of :numref:`subsec_mdl-mdl` encodes the model parameters
explicitly at some chosen precision, which makes the total description length
sensitive to how finely we discretize the parameter space. A *prequential*
code avoids sending any parameters at all. Instead, it encodes each data point
using the model fitted to the points seen so far, just like the adaptive
counting model in :numref:`subsec_mdl-lm-compressor`, so the cost of learning
the model is absorbed directly into the data code length.
:citet:`Blier.Ollivier.2018` used prequential coding to compress the training
labels of deep neural networks and obtained far shorter codes than either
two-part codes or variational codes (which apply bits-back coding to the
network weights). Even a heavily overparameterized network compresses its
labels when measured this way. :citet:`Grunwald.2007` develops
both prequential codes and normalized maximum likelihood, the code that
minimizes worst-case regret relative to the best model in the class chosen in
hindsight. Pushing this compression viewpoint to its theoretical extreme leads
to Kolmogorov complexity, the length of the shortest computer program that
outputs the data, which unifies all of these ideas, though it is famously
uncomputable :cite:`Li.Vitanyi.2019`.

### The Constant of a Scaling Law

In :numref:`sec_scaling-laws`, the held-out loss of language models across
many scales is modeled as a constant $E$ plus two power-law terms, one
decaying with the number of parameters and the other with the number of
training tokens :cite:`hoffmann2022training`. It is tempting to read $E$ as
the true entropy of natural text, and in the best case $E$ does correspond to
the entropy rate per token (:numref:`subsec_mdl-entropy-rate`), the floor
below which no expected loss can fall. In practice, however, $E$ is only a
fitted regression constant extrapolated from losses that all sit well above
it, not a direct measurement of the entropy rate. Earlier work by
:citet:`kaplan2020scaling` fit similar scaling curves with no additive
constant at all, an approximation that must eventually break down at large
enough scale because the loss cannot fall below the entropy floor. Keep in
mind, too, that $E$ is measured per token: constants fitted under different
tokenizers cannot be compared directly until both are converted to bits per
byte (:numref:`subsec_bits-per-byte`).

## Comparing Distributions

### Rényi Divergences and Privacy

How much does a randomized algorithm leak about a single training record when
we run it repeatedly? Differential privacy formalizes this leak as a
divergence between the algorithm's output distributions on two datasets that
differ in just that one record. Because differentially private SGD injects
noise at every gradient step and runs for thousands of iterations, we need a
divergence that composes cleanly across independent steps. The Rényi
divergence of order $\alpha$ :cite:`Renyi.1961`,

$$
D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\log\sum_x p(x)^\alpha q(x)^{1-\alpha},
$$

has this property. Notice that the logarithm sits *outside* the sum,
whereas an f-divergence applies its convex function inside. Because the sum
factorizes over independent coordinates, the logarithm turns that product into
a sum, so $D_\alpha$ adds cleanly over independent steps, just like KL
divergence does and unlike $\chi^2$ or total variation distance. Rényi
differential privacy :cite:`Mironov.2017` bounds $D_\alpha$ for a single noisy
step and then sums that bound over the entire training run. While KL
divergence also adds across steps, it only controls the log-density ratio
$\log(p(x)/q(x))$ on average. Because $D_\alpha$ increases monotonically with
$\alpha$, interpolating from KL divergence as $\alpha \to 1$ to the worst-case
log-ratio as $\alpha \to \infty$, choosing a larger order $\alpha$ protects
against rare, high-leakage outputs as well.
:citet:`Van-Erven.Harremoes.2014` provide a thorough catalog of these limits
and inequalities, and Exercise
:exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`d of
:numref:`sec_mdl-divergences-distances` connects $D_\alpha$ to the
$\alpha$-divergences.

### Information Geometry

In :numref:`sec_mdl-divergences-distances`, a quick Taylor expansion showed
that every smooth f-divergence between $q_{\boldsymbol\theta + \mathbf{h}}$
and $q_{\boldsymbol\theta}$ reduces, to second order, to the quadratic form
$\tfrac12 f''(1)\, \mathbf{h}^\top \mathbf{I}(\boldsymbol\theta)\, \mathbf{h}$
in the Fisher information matrix. That might look like an algebraic
coincidence, yet Chentsov's theorem shows that it could not have turned out
any other way. A positive-definite quadratic form assigns a length to every
infinitesimal step $\mathbf{h}$, turning the space of distributions into a
Riemannian manifold (measuring local steps, whereas the metrics of
:numref:`sec_mdl-what-is-a-divergence` measure global distances between two
distributions). For distributions on finite sets, Chentsov proved that the
Fisher information is, up to a positive constant, the *only* Riemannian metric
that remains invariant when the raw data are replaced by a sufficient
statistic :cite:`Cencov.1982`. Every f-divergence shares this invariance:
since the data can be resampled from a sufficient statistic without knowing
$\boldsymbol\theta$, the data-processing inequality of
:numref:`subsec_mdl-f-div-data-processing` applies in both directions, forcing
equality. Consequently, the local quadratic approximation of any smooth
f-divergence must be a multiple of the Fisher information. Taking gradient
steps whose length is measured in this geometry yields the natural gradient,
which K-FAC approximates with one Kronecker product per layer
(:numref:`subsec_mdl-preconditioning-ladder`), and the KL trust region of
:numref:`sec_ppo` is, to second order, a ball in this very metric.
:citet:`Amari.2016` develops this geometric perspective in full, showing how
maximum likelihood and both steps of the EM algorithm
(:numref:`sec_mdl-latent-em-elbo`) can be understood as orthogonal
projections.

### Sinkhorn Divergences and Sliced Wasserstein

Turning an optimal transport distance into a practical loss on high-dimensional
minibatches runs into two obstacles: computational cost and statistical bias.
Entropic regularization (:numref:`subsec_mdl-sinkhorn`) solves the
computational problem, yet it introduces a subtle bias: the regularized
transport cost from a distribution to itself is strictly nonzero, so a model
trained to minimize it tends to shrink toward a narrower distribution than the
data. The Sinkhorn divergence of Exercise
:exref:`ex_mdl-divergences-distances-sinkhorn-divergences` in
:numref:`sec_mdl-test-functions-transport` fixes this by subtracting the
average of the two self-transport costs. Beyond that exercise lies the theorem
that makes it a trustworthy training objective: on a bounded domain, using the
Euclidean distance or its square as the ground cost, the Sinkhorn divergence
is nonnegative, equals zero if and only if the two distributions agree, and
converges to zero if and only if the model converges in distribution to the
data :cite:`Feydy.Sejourne.Vialard.ea.2019`.

The *sliced* Wasserstein distance takes a completely different route. If we
project two high-dimensional samples onto a single one-dimensional line,
optimal transport reduces to simply sorting the projected values
(:numref:`sec_mdl-optimal-transport`). Averaging this one-dimensional cost
over many random directions produces the sliced Wasserstein distance
:cite:`Bonneel.Rabin.Peyre.ea.2015`. The catch in high dimensions is that a
random direction rarely aligns with the subspace where two distributions
actually differ. To get around this, :citet:`Deshpande.Zhang.Schwing.2018`
train a discriminator to extract informative features and compute the sliced
Wasserstein distance in that learned feature space instead of raw pixels.

### Lipschitz Critics in Wasserstein GANs

The Kantorovich–Rubinstein dual in :numref:`sec_mdl-optimal-transport`
optimizes over 1-Lipschitz functions, a constraint that an ordinary neural
network does not satisfy out of the box. :numref:`sec_gan-further-reading`
compares the three main ways to enforce it (weight clipping, spectral
normalization, and gradient penalties), while :numref:`sec_gan_convergence`
examines gradient penalties in detail. In particular, the penalty of
:citet:`Gulrajani.Ahmed.Arjovsky.ea.2017` nudges the critic's gradient norm
toward one at random points interpolated between real and generated samples,
reflecting the geometric fact that an optimal critic has unit slope along the
transport rays connecting matched points (Exercise
:exref:`ex_mdl-test-functions-transport-optimal-critics`d of
:numref:`sec_mdl-test-functions-transport`).

### Stein Variational Gradient Descent

Drawing samples from a distribution $Q$ known only through its score $s_Q$
ordinarily requires running a long stochastic random walk
(:numref:`sec_mdl-scores`). In :numref:`sec_mdl-ksd`, we saw that the kernel
Stein discrepancy measures how far a sample distribution $P$ sits from such a
model $Q$. Stein variational gradient descent :cite:`Liu.Wang.2016` turns that
discrepancy into a deterministic algorithm that actively transports a set of
particles toward $Q$. Starting from any initial cloud of points with
distribution $P$, each particle $y$ moves in small steps along the velocity
field

$$
v(y) = \mathbb{E}_{x \sim P}\big[k(x, y)\, s_Q(x) + \nabla_x k(x, y)\big].
$$

The two terms inside the expectation play complementary roles. The first is a
kernel-weighted average of the scores near $y$, which steers particles uphill
toward regions of higher probability density. The second is a repulsive force
that pushes $y$ away from its neighbors, taking over the role of random noise
in Langevin dynamics; without it, every particle would collapse onto a single
mode. When normalized to the unit ball of the reproducing kernel Hilbert space
from :numref:`sec_mdl-ksd`, this velocity field is the steepest descent
direction for $D_{\textrm{KL}}(P\|Q)$, reducing the KL divergence at rate
$\|v\| = \mathrm{KSD}(P, Q)$. Remarkably, the same Stein operator serves both
as a goodness-of-fit test and as a particle transport rule. Although the
particles still require many small steps, the updates are deterministic and
the particles spread out cooperatively, which makes the method well suited to
approximating a posterior with only a few dozen particles, such as the weights
of a small neural network in the original paper.

## Mutual Information

### Codes That Approach Capacity

Shannon's channel coding theorem (:numref:`subsec_mdl-channel-capacity`)
guarantees that reliable communication is possible at every rate below channel
capacity, but his proof relies on random codebooks that take exponential time
to decode. Bridging the gap between that existence theorem and practical
hardware took decades. Low-density parity-check (LDPC) codes, decoded
efficiently by passing messages along a sparse factor graph, come remarkably
close to capacity in practice :cite:`Richardson.Urbanke.2008`. Polar codes
:cite:`Arikan.2009` go a step further: they are the first explicit code family
provably shown to achieve capacity on symmetric binary-input channels, with
both encoding and decoding running in $O(n \log n)$ operations for block
length $n$. Modern 5G wireless standards put both families to work.

### Learned Lossy Compression

In :numref:`subsec_mdl-rate-distortion`, we derived the exact rate--distortion
curve for a Gaussian source. For natural photographs, no closed-form
rate--distortion function is known, so any learned image codec represents a
practical operating point somewhere above that unknown frontier.
:citet:`Balle.Laparra.Simoncelli.2017` train three components jointly: an
encoder, a decoder, and an *entropy model* that defines a learned probability
distribution over the quantized latent representation. Because rounding has
zero gradient almost everywhere, they replace quantization with additive
uniform noise during training. Their objective minimizes $R + \lambda D$,
where the rate $R$ is the arithmetic-coding length of the latent under the
entropy model (:numref:`sec_mdl-coding-compression`) and $D$ is the
reconstruction distortion of the decoded image; varying $\lambda$ traces out
the codec's rate--distortion curve by selecting the point with slope
$-\lambda$. In follow-up work, :citet:`Balle.Minnen.Singh.ea.2018` condition
the entropy model on a second, smaller latent called a *hyperprior*, which
captures spatial variations in image complexity and substantially lowers the
bit rate at the same visual quality.

## Estimating Mutual Information

### Upper Bounds

Every variational estimator developed in :numref:`sec_mdl-mi-estimation` is a
*lower* bound on mutual information. That is useful when you want to maximize
shared information, but it cannot certify that a learned representation keeps
*little* information about a sensitive or protected attribute it is supposed
to discard. Obtaining a genuine upper bound requires access to the conditional
density $p(z \mid x)$, which a stochastic encoder provides. The variational
information bottleneck (:numref:`subsec_mdl-variational-bottleneck`) is one
such upper bound, using a fixed marginal approximation $r(z)$. The contrastive
log-ratio upper bound, or CLUB :cite:`Cheng.Hao.Dai.ea.2020`, removes the need
for $r(z)$ altogether: it averages $\log p(z \mid x)$ over matched pairs in a
minibatch and subtracts the average over mismatched pairs. When $p(z \mid x)$
is the true encoder density, CLUB is an upper bound in expectation. Between a
representation and a protected attribute, however, no analytic conditional
density is available, so one must fit a neural network in its place. The upper
bound then holds only if the fitted conditional is closer in KL divergence to
the true joint distribution than to the product of the marginals, a condition
that cannot be verified without the very density we were missing in the first
place.

### Bias and Variance of the Bounds

Even when given the exact optimal critic, variational bounds face a sharp
trade-off: NWJ is unbiased on average but suffers from high variance, whereas
InfoNCE is low-variance but capped at $\log N$
(:numref:`subsec_mdl-exact-critics`). :citet:`Poole.Oord.Alemi.ea.2019`
introduce a single mixing parameter that smoothly interpolates between the two
bounds, letting you dial along this trade-off. Worse still,
:citet:`Song.Ermon.2020b` prove that the variance explosion is fundamental
rather than an artifact of any one objective: for both NWJ and
Donsker--Varadhan bounds, the estimator variance grows exponentially with the
true mutual information, even with an exact critic. Their proposed fix clips
the critic's density ratio, which tames the variance at the cost of
introducing bias. In short, every available remedy shifts error between
variance and bias rather than eliminating it.

Because ground-truth mutual information is rarely known on real data, the same
authors suggest self-consistency sanity checks that require no known answer: a
reliable estimator should remain unchanged when a deterministic function of
one variable is appended to that variable, and it should double when a single
pair $(X, Y)$ is replaced by two independent copies of itself. On image
datasets, the standard estimators they evaluate fail these basic tests. As a
practical rule of thumb, always compare a reported mutual information estimate
of several nats against the logarithm of the batch size, and trust an
estimator on your own data only after verifying that it passes these
consistency checks there.

### Noisy Networks and the Bottleneck View

The compression debate of :numref:`subsec_mdl-compression-debate` began when
:citet:`Tishby.Zaslavsky.2015` proposed plotting each layer of a neural
network against the information bottleneck frontier
(:numref:`sec_mdl-information-bottleneck`), which tracks the maximum $I(Y; Z)$
achievable for a given $I(X; Z)$. As we saw in that subsection, however, the
idea runs into a mathematical snag: in a deterministic network fed continuous
inputs, $I(X; Z)$ is formally infinite, leaving no finite compression curve
to track. :citet:`Goldfeld.Berg.Greenewald.ea.2019` make the question well
posed by injecting small Gaussian noise into the hidden activations, which
renders $I(X; Z)$ finite. Under this noisy model, $I(X; Z)$ can indeed
decrease during training, and the authors trace that drop to geometric
clustering as hidden representations from the same class pull closer together.
This clustering, they argue, is what earlier binned mutual-information
estimates on deterministic networks were really picking up all along.

[Discussions](https://d2l.discourse.group/t/36867)
