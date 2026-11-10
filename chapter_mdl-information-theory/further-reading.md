# Further Reading
:label:`sec_mdl-information-theory-further-reading`

The chapter kept to what can be proved in a page and checked on a small
example, and each entry below starts where a section stopped. Books and papers
for what the chapter teaches are listed in its introduction
(:numref:`chap_mdl-information-theory`); these entries are for what it does
not. Read first the three that bear on practice: calibration, the caveats on
scaling laws, and the bias and variance of mutual-information estimators. The
entries on capacity and on information geometry are for the curious. Nothing
else in the book depends on this section.

## Entropy and Losses

### Calibration after Training

:numref:`subsec_mdl-irreducible-loss` shows that the population log-loss is
minimized by the true conditional probabilities, so a classifier at that
optimum is calibrated. Trained networks are not at the optimum: with the
reliability diagram of :numref:`subsec_beyond-accuracy`,
:citet:`Guo.Pleiss.Sun.Weinberger.2017` find modern ones systematically
overconfident. Their repair is *temperature scaling*: divide the logits by one
constant $\tau$ chosen to minimize the log-loss on held-out data. The argmax,
and hence the accuracy, does not change. This is the temperature of
distillation used for a different purpose, and one constant cannot help a
model that is overconfident on some classes and underconfident on others.
:citet:`Gneiting.Raftery.2007` give the general theory of proper scoring
rules, of which the log score is one member.

### Entropy Relative to a Reference

:numref:`subsec_mdl-differential-entropy` shows that differential entropy
changes by $\ln 100$ when metres become centimetres. Measuring entropy
relative to a reference density $m$, as
$-\int p \log(p/m)\,dx = -D_{\textrm{KL}}(P\|M)$, removes the units: a change
of variables applied to both $p$ and $m$ leaves the value unchanged. The ordinary $h$ is this quantity
with $m = 1$ in the chosen units, so the $\ln 100$ was a change of reference.
:citet:`Jaynes.1968` obtains the form as the limit of discrete entropies on
ever finer grids, and in this form the maximum-entropy principle of
:citet:`Jaynes.1957` reads as "stay as close to the reference as the
constraints allow". The KL penalty of a variational autoencoder is such a
relative entropy, hence independent of the coordinates of the latent variable.

## Codes and Compression

### Compressing without a Model

Every coder of :numref:`sec_mdl-coding-compression` is driven by a model
that predicts the next symbol. The Lempel–Ziv algorithms need none. LZ77
:cite:`Ziv.Lempel.1977` replaces a repeated substring by a pointer to its
previous occurrence, and LZ78 :cite:`Ziv.Lempel.1978` builds a dictionary of
phrases as it parses. Both are *universal*: on any stationary ergodic source
their rate approaches the entropy rate as the text grows. On a short text
they pay for this generality: `zlib`, which combines LZ77 with a Huffman
code, loses on *The Time Machine* to the small adaptive context model that
opens that section.

### Description Length beyond Two-Part Codes

The two-part code of :numref:`subsec_mdl-mdl` spends bits on the parameter at
a chosen precision, so its length depends on the grid. A *prequential* code
sends no parameter: it codes each symbol with the model fitted to the symbols
before it, and the cost of learning appears inside the data term.
:citet:`Blier.Ollivier.2018` code the training labels of deep networks in this
way and find the code far shorter than two-part and variational codes: a large
network does compress its labels. :citet:`Grunwald.2007` develops these codes
together with normalized maximum likelihood, the code with the smallest
worst-case regret over a model class. Kolmogorov complexity, the length of the
shortest program that prints the data, is the idealization of all of them; it
is defined up to an additive constant and is not computable
:cite:`Li.Vitanyi.2019`.

### Scaling Laws of Held-Out Cross-Entropy

Can the held-out loss of a language model be predicted before training it?
Within the ranges studied, fits of the form $L = L_\infty + a x^{-\alpha}$,
with $x$ the number of parameters, tokens or units of compute, describe the
held-out cross-entropy well :cite:`kaplan2020scaling,hoffmann2022training`.
A power law says that each constant *multiple* of $x$ removes a constant
fraction of the reducible loss $L - L_\infty$. The fitted $L_\infty$ is a
regression constant, not a measured entropy rate of language, and since the
loss is per token (:numref:`subsec_mdl-perplexity`), fits with different
tokenizers cannot be compared directly.

## Divergences

### Rényi Divergences and Privacy

How much does a randomized algorithm reveal about one record when it is run
many times? The Rényi divergence of order $\alpha$ :cite:`Renyi.1961`,

$$
D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\log\sum_x p(x)^\alpha q(x)^{1-\alpha},
$$

has its logarithm *outside* the sum. The sum factorizes over independent
components, so $D_\alpha$ adds over them, as KL does and $\chi^2$ and total
variation do not. It is nondecreasing in $\alpha$ and tends to KL as
$\alpha \to 1$; :citet:`Van-Erven.Harremoes.2014` collect its limits and
inequalities,
and problem :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`d of
:numref:`sec_mdl-f-divergences` relates it to the $\alpha$-divergences. Rényi
differential privacy :cite:`Mironov.2017` bounds $D_\alpha$ between the
outputs on neighboring data sets, and the bounds of repeated runs simply add.

### Sinkhorn Divergences and Sliced Wasserstein

How is a transport distance used as a loss on minibatches in many dimensions?
Entropic regularization (:numref:`subsec_mdl-sinkhorn`) is one answer, and
problem :exref:`ex_mdl-divergences-distances-sinkhorn-divergences` removes its
bias with the Sinkhorn divergence. Beyond that problem lies the theorem that
makes the divergence a loss: for common costs it is nonnegative, zero only for
equal distributions, and metrizes weak convergence
:cite:`Feydy.Sejourne.Vialard.ea.2019`. The *sliced* Wasserstein distance is
the other answer. It averages the one-dimensional closed form of
:numref:`sec_mdl-optimal-transport` over random directions, so each direction
costs a sort :cite:`Bonneel.Rabin.Peyre.ea.2015`, and image generators are
trained on it directly :cite:`Deshpande.Zhang.Schwing.2018`.

### Lipschitz Critics in Wasserstein GANs

The Kantorovich–Rubinstein dual of :numref:`sec_mdl-optimal-transport` needs a
1-Lipschitz function, and a network is not one by construction.
:numref:`sec_gan-further-reading` compares the three ways of making it one,
and :numref:`sec_gan_convergence` treats gradient penalties. One remark
belongs here: the penalty of :citet:`Gulrajani.Ahmed.Arjovsky.ea.2017` asks
for a gradient of norm one between the two distributions because the optimal
function has unit slope along the lines on which transport moves mass.

### Stein Variational Gradient Descent

The kernel Stein discrepancy of :numref:`sec_mdl-stein-discrepancy` *measures*
how far a sample from $P$ is from a distribution $Q$ known only through its
score $s_Q$. Stein variational gradient descent :cite:`Liu.Wang.2016` *moves*
the sample instead, along

$$
\phi^\star(\cdot) = \mathbb{E}_{x \sim P}\big[k(x, \cdot)\, s_Q(x) + \nabla_x k(x, \cdot)\big],
$$

the direction in the unit ball of the kernel's function space that decreases
$D_{\textrm{KL}}(P\|Q)$ fastest. Its norm is the kernel Stein discrepancy
:cite:`Liu.Lee.Jordan.2016`, the square root of :eqref:`eq_mdl-ksd-def`, so
the same Stein operator serves as a test statistic and as a velocity field.

### Information Geometry

On a parametric family every smooth f-divergence has the same local quadratic,
a multiple of the Fisher information $\mathbf{I}(\boldsymbol\theta)$
(:eqref:`eq_mdl-f-div-local` and the text after it). Chentsov's theorem
explains why: up to scale, it is the only metric that is unchanged when the
data are replaced by a sufficient statistic :cite:`Amari.2016`. Gradient
descent in this metric is the natural gradient :cite:`Amari.1998`. K-FAC
:cite:`Martens.Grosse.2015` approximates it layer by layer, and the KL trust
region of :numref:`sec_ppo` is, to second order, a ball in it.

## Mutual Information

### Codes That Approach Capacity

:numref:`subsec_mdl-channel-capacity` states that every rate below capacity is
achievable, but Shannon's random codes take exponential time to decode.
Low-density parity-check codes, decoded by belief propagation on the graph of
the code, come close to capacity in practice :cite:`Richardson.Urbanke.2008`,
and polar codes :cite:`Arikan.2009` provably achieve it on symmetric
binary-input channels, with encoding and decoding in $O(n \log n)$ operations
for block length $n$.

### Learned Lossy Compression

:numref:`subsec_mdl-rate-distortion` computes the rate–distortion function
of a Gaussian source. A learned image codec is a point above the curve of
its own source. :citet:`Balle.Laparra.Simoncelli.2017` train an encoder, a
decoder and an *entropy model* of the quantized latent together, and the
rate in their loss is the arithmetic-coding length of the latent under that
model (:numref:`sec_mdl-coding-compression`). Minimizing $R + \lambda D$
picks the point of the codec's curve at which its slope is $-\lambda$.
:citet:`Balle.Minnen.Singh.ea.2018` make the entropy model conditional on a
second latent, a *hyperprior*, which lowers the rate at equal distortion.

## Estimating Mutual Information

### Upper Bounds

Every variational estimator of :numref:`sec_mdl-mi-estimation` is a lower
bound, so none can certify that a representation keeps *little* information.
An upper bound needs the conditional density $p(z \mid x)$, which a stochastic
encoder supplies. The variational bottleneck of
:numref:`sec_mdl-information-bottleneck` is one such bound, with a fixed
density $r(z)$ in place of the marginal. CLUB :cite:`Cheng.Hao.Dai.ea.2020`
needs no $r$: it averages $\log p(z \mid x)$ over the matched pairs of a batch
and subtracts its average over the mismatched pairs. It is an upper bound in
expectation when the density is the encoder's own; a learned approximation
needs an extra condition.

### Trading Bias for Variance

With an exact critic NWJ is right on average but noisy, and InfoNCE is steady
but capped at $\log N$ (:numref:`subsec_mdl-exact-critics`).
:citet:`Poole.Oord.Alemi.ea.2019` interpolate between the two with one mixing
parameter. :citet:`Song.Ermon.2020b` prove that the variance of estimators of
the NWJ and MINE kind can grow exponentially with the true information, show
that existing estimators fail consistency checks such as additivity over
independent pairs, and reduce the variance by clipping the critic's density
ratio, at the price of a bias.

### Noisy Networks and the Bottleneck View

:numref:`subsec_mdl-compression-debate` leaves the compression phase with
nothing to measure: in a deterministic network on continuous inputs, $I(X; Z)$
is infinite. The view behind the debate, that a layer should be judged against
the bottleneck frontier of :numref:`sec_mdl-information-bottleneck`, goes back
to :citet:`Tishby.Zaslavsky.2015`. :citet:`Goldfeld.Berg.Greenewald.ea.2019`
make the question well posed by adding Gaussian noise to the activations,
which makes $I(X; Z)$ finite. It can then fall during training, and they
trace the fall to the clustering of the representations of each class, which,
they argue, is also what the binned estimates on deterministic networks
measured.

[Discussions](https://d2l.discourse.group/t/36867)
