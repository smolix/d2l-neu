# Further Reading
:label:`sec_mdl-information-theory-further-reading`

The sections of this chapter read a log-loss as a code length, compare
distributions by ratios, test functions, transport and scores, and measure
and bound dependence. This section collects, in the order of the chapter,
topics they had no room for. Each topic states the problem it addresses and
what it changes relative to the chapter. Resources for the material the
chapter does teach are listed in its introduction
(:numref:`chap_mdl-information-theory`). Nothing else in the book depends on
this section.

## Entropy and Losses

### Calibration after Training

:numref:`subsec_mdl-irreducible-loss` shows that the population log-loss is
minimized by the true conditional probabilities, so a classifier at that
optimum is calibrated: among predictions made with confidence $0.9$, a
fraction $0.9$ is right. Trained networks are not at the optimum, and
:citet:`Guo.Pleiss.Sun.Weinberger.2017` find that modern ones are
systematically overconfident. They measure the gap with a reliability
diagram, accuracy binned by confidence, and repair it with *temperature
scaling*: divide the logits by one constant $\tau$ chosen to minimize the
log-loss on held-out data. The argmax, and hence the accuracy, does not
change. This is the temperature of distillation used for a different
purpose. :citet:`Gneiting.Raftery.2007` give the general theory of proper
scoring rules, of which the log score is one member.

### Entropy Relative to a Reference

:numref:`subsec_mdl-differential-entropy` shows that differential entropy
changes by $\ln 100$ when metres become centimetres, so it cannot say on its
own how random a continuous variable is. Measuring entropy relative to a
reference density $m$, as $-\int p \log(p/m)\,dx = -D_{\textrm{KL}}(P\|M)$,
removes the units: a change of variables applied to both $p$ and $m$ leaves
the value unchanged. The ordinary $h$ is this quantity with $m = 1$ in the
chosen units. :citet:`Jaynes.1968` obtains the form as the limit of discrete
entropies on ever finer grids, and in this form the maximum-entropy
principle of :citet:`Jaynes.1957` reads as "stay as close to the reference
as the constraints allow".

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

The two-part code of :numref:`subsec_mdl-mdl` spends bits on the parameter
at a chosen precision, so its length depends on the grid. A *prequential*
code sends no parameter: it codes each symbol with the model fitted to the
symbols before it, and the cost of learning appears inside the data term.
:citet:`Blier.Ollivier.2018` use such codes to measure the description length
of deep networks and find it far shorter than the lengths that two-part and
variational codes give.
:citet:`Grunwald.2007` develops these codes together with normalized maximum
likelihood, the code with the smallest worst-case regret over a model
class. Kolmogorov complexity, the length of the shortest program that
prints the data, is the idealization of all of them; it is defined up to
an additive constant and is not computable :cite:`Li.Vitanyi.2019`.

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

$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\log\sum_x p(x)^\alpha q(x)^{1-\alpha},$$

has its logarithm *outside* the sum. It is therefore not an f-divergence but
an increasing function of one, and because the sum factorizes over
independent components, $D_\alpha$ adds over them, as KL does and $\chi^2$
and total variation do not. It is nondecreasing in $\alpha$ and tends to
KL as $\alpha \to 1$; :citet:`Van-Erven.Harremoes.2014` collect its limits
and inequalities. Rényi differential privacy :cite:`Mironov.2017` bounds
$D_\alpha$ between the outputs on neighboring data sets, and the bounds of
repeated runs simply add.

### Sinkhorn Divergences and Sliced Wasserstein

The entropic transport cost of :numref:`subsec_mdl-sinkhorn` is not zero for
two copies of one distribution, and in many dimensions exact transport
needs a linear program over $n^2$ variables. The *Sinkhorn divergence*
subtracts the two self-terms. It interpolates between the Wasserstein
distance (small $\varepsilon$) and an MMD (large $\varepsilon$)
:cite:`Genevay.Peyre.Cuturi.2018`, and it is nonnegative and metrizes weak
convergence for common costs :cite:`Feydy.Sejourne.Vialard.ea.2019`. The
*sliced* Wasserstein distance avoids the transport problem: it averages the
one-dimensional closed form of :numref:`sec_mdl-optimal-transport` over
random projection directions, so each direction costs a sort
:cite:`Bonneel.Rabin.Peyre.ea.2015`.

### Lipschitz Critics in Wasserstein GANs

The Kantorovich–Rubinstein dual of :numref:`sec_mdl-optimal-transport`
needs a 1-Lipschitz function, and a network is not one by construction.
Weight clipping :cite:`Arjovsky.Chintala.Bottou.2017` and spectral
normalization :cite:`Miyato.Kataoka.Koyama.ea.2018` bound the product of
the layers' Lipschitz constants, a global bound that is often loose. A
gradient penalty :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017` instead asks for
a gradient of norm one at points sampled between the two distributions. It
gives no global guarantee, but it targets the property of the optimal
function, which has unit slope along the lines that transport moves mass
on.

### Stein Variational Gradient Descent

The kernel Stein discrepancy of :numref:`sec_mdl-stein-discrepancy`
*measures* how far a set of particles is from a distribution $P$ known only
through its score. Stein variational gradient descent
:cite:`Liu.Wang.2016` *moves* the particles instead, along

$$\phi^\star(\cdot) = \mathbb{E}_{x \sim Q}\big[k(x, \cdot)\, s_P(x) + \nabla_x k(x, \cdot)\big],$$

the direction in the unit ball of the kernel's function space that
decreases $D_{\textrm{KL}}(Q\|P)$ fastest. The norm of this direction is
the kernel Stein discrepancy :cite:`Liu.Lee.Jordan.2016`, so the same Stein
operator serves as a test statistic and as a velocity field.

### Information Geometry

Near equality every smooth f-divergence is the same quadratic
(:numref:`sec_mdl-f-divergences`). On a parametric family $q_\theta$ this
quadratic is $\tfrac12 f''(1)\, d\boldsymbol\theta^\top
\mathbf{F}(\boldsymbol\theta)\, d\boldsymbol\theta$, with $\mathbf{F}$ the
Fisher information. Up to scale it is the only metric that is unchanged when
the data are replaced by a sufficient statistic, which is why every
f-divergence leads to it. Gradient descent in this metric is the
natural gradient :cite:`Amari.1998`. :citet:`Amari.2016` develops the
geometry further.

## Mutual Information

### Codes That Approach Capacity

:numref:`subsec_mdl-channel-capacity` states that every rate below capacity
is achievable, but Shannon's proof uses random codes that take exponential
time to decode. Low-density parity-check codes decoded by belief
propagation come close to capacity in practice :cite:`Richardson.Urbanke.2008`.
Polar codes :cite:`Arikan.2009` were the first family proved to achieve
the capacity of symmetric binary-input channels, the binary symmetric
channel included, with encoding and decoding in $O(n \log n)$ operations
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

Every estimator of :numref:`sec_mdl-mi-estimation` is a lower bound, so none
can certify that a representation keeps *little* information, which the
compression term of a bottleneck needs. An upper bound needs the
conditional density $p(z \mid x)$, which a stochastic encoder supplies. The
variational information bottleneck :cite:`Alemi.Fischer.Dillon.ea.2017`
uses $I(X; Z) \le \mathbb{E}_x D_{\textrm{KL}}(p(z \mid x)\,\|\,r(z))$ for
any fixed density $r$, loose by $D_{\textrm{KL}}(p(z)\,\|\,r(z))$. CLUB
:cite:`Cheng.Hao.Dai.ea.2020` replaces the fixed $r$ by the batch: it
averages $\log p(z \mid x)$ over matched pairs minus its average over
mismatched pairs.

### Trading Bias for Variance

:numref:`subsec_mdl-exact-critics` shows that with an exact critic NWJ is
correct on average but carried by rare pairs, so it is noisy at large
information, while InfoNCE is steady but capped at $\log N$. :citet:`Poole.Oord.Alemi.ea.2019` interpolate between the two
with one mixing parameter. :citet:`Song.Ermon.2020b` prove that the variance
of estimators of the NWJ and MINE kind can grow exponentially with the true
information, show that existing estimators fail simple consistency checks
such as additivity over independent pairs, and reduce the variance by
clipping the critic's density ratio, at the price of a bias.

### Noisy Networks and the Bottleneck View

:numref:`subsec_mdl-compression-debate` shows why the reported compression
phase of deep classifiers is an artifact of binning: in a deterministic
network on continuous inputs, $I(X; Z)$ is infinite. The claim it tested,
that training drives each layer toward the bottleneck frontier of
:numref:`sec_mdl-information-bottleneck`, was proposed by
:citet:`Tishby.Zaslavsky.2015`. :citet:`Goldfeld.Berg.Greenewald.ea.2019`
make the question well posed by adding Gaussian noise to the activations,
so that $I(X; Z)$ is finite and can be estimated. With that definition the
measured compression depends on the noise scale, and it tracks how the
representations of each class cluster together.

[Discussions](https://d2l.discourse.group/)
