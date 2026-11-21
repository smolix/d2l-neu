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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37]{.kicker}

Information Theory and Divergences<br>**entropy · cross-entropy · source coding · f-divergences · optimal transport · scores · mutual information**
:::
:::

::: {.slide title="Why Do We Train Models with Cross-Entropy?"}
[Central Questions of Chapter 37]{.kicker}

Most models in deep learning minimize cross-entropy or compare distributions through KL penalties and adversarial critics. Information theory gives these objectives exact physical and statistical meaning:

- **What does a loss value actually measure?** When a language model reports $1.2$ nats per character or a classifier plateaus at $0.6$ nats, how low could any model possibly go on that dataset?
- **Why is a language model's loss literally a file size?** How does pairing an autoregressive model with an arithmetic coder store a message in its total log-loss in bits plus at most two extra bits?
- **What happens when KL divergence blows up?** When generated samples miss the data manifold or a model's normalizing constant $Z$ is intractable, how can we compare distributions from samples or slopes alone?

::: {.d2l-note .rule}
**The Fundamental Decomposition:** Every cross-entropy loss splits into an irreducible data floor plus a non-negative divergence gap: $\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P \| Q) \ge H(P)$.
:::
:::

::: {.slide title="Three Core Theorems Every Practitioner Needs"}
[Core Mathematical Pillars]{.kicker}

- **Gibbs' inequality and the irreducible floor (§37.1):** Because $D_{\textrm{KL}}(P \| Q) \ge 0$ with equality if and only if $P = Q$, no classifier can push population cross-entropy below the conditional entropy $\mathbb{E}_{\mathbf{x}}[H(p(\cdot \mid \mathbf{x}))]$ of the labels.
- **Prediction is lossless compression (§37.2):** Kraft's inequality ($\sum_i 2^{-\ell_i} \le 1$) governs symbol codes, while arithmetic coding turns any sequential predictor $q(x_i \mid x_{<i})$ into a stream compressor achieving rate $\textrm{CE}_2(P, Q)$ bits per symbol.
- **Sample-based bounds have logarithmic ceilings (§§37.3–37.7):** Variational critics lower-bound divergences ($f$-GAN) and mutual information (InfoNCE) from samples alone, but identifying one true partner among $N$ candidates can certify at most $\log N$ nats.

::: {.d2l-note .warn}
**The Density-Ratio Blind Spot:** Every $f$-divergence built from $p(\mathbf{x})/q(\mathbf{x})$ freezes at a constant on disjoint supports, giving zero gradient—motivating optimal transport ($W_1$), kernel MMD, and score matching.
:::
:::

::: {.slide title="Chapter 37 Roadmap: Seven Connected Sections"}
[Chapter Architecture]{.kicker}

| Section | Core Mathematical Object | Key Theorem / Identity | Deep Learning Role |
|:--|:---|:---|:---|
| **§37.1 Entropy & KL** | Surprisal $-\log p(x)$, $H(P)$, $D_{\textrm{KL}}(P \| Q)$ | $\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P \| Q) \ge H(P)$ | Classification floor, calibration, smoothing, distillation |
| **§37.2 Coding & Perplexity** | Prefix trees, dyadic intervals, $\textrm{PPL} = e^{\textrm{CE}}$ | Kraft $\sum 2^{-\ell_i} \le 1$; $\ell < -\log_2 q(x_{1:n}) + 2$ | LLM compression, MDL, bits per byte across tokenizers |
| **§37.3 $f$-Divergences** | Density ratio $u = p/q$, convex $f(1)=0$ | Pinsker $\textrm{TV} \le \sqrt{D_{\textrm{KL}}/2}$; Fenchel dual | Forward vs. reverse KL (MLE vs. VI), $f$-GAN critics |
| **§37.4 IPMs & Transport** | Test class $\mathcal{G}$, couplings $\gamma \in \Pi(P, Q)$ | Kantorovich–Rubinstein duality for $W_1$ | Kernel MMD two-sample tests, WGAN, Sinkhorn OT |
| **§37.5 Score Comparisons** | Score $s_Q(\mathbf{x}) = \nabla_{\mathbf{x}} \log q(\mathbf{x})$ | Stein's identity $\mathbb{E}_p[g' + g s_p] = 0$ | Hyvärinen score matching, diffusion noise scales, KSD |
| **§37.6 Mutual Information** | $I(X; Y) = D_{\textrm{KL}}(P_{X,Y} \| P_X \otimes P_Y)$ | Data processing $I(X; Z) \le I(X; Y)$; Fano | Representation limits, rate–distortion, VAE bottleneck |
| **§37.7 MI Estimation** | KSG neighbor boxes, critics $T(x, y)$ | InfoNCE $I(X; Y) \ge \log N - \mathcal{L}_{\textrm{NCE}}$ | SimCLR / CLIP contrastive learning, $\log N$ ceiling |
:::

::: {.slide title="Choosing a Distribution Comparison by Access"}
[Practical Decision Guide]{.kicker}

What your model supplies determines how you can train or evaluate it against data $P$:

- **Tractable normalized density $q_{\boldsymbol{\theta}}(\mathbf{x}) > 0$:** Minimize **forward KL** $D_{\textrm{KL}}(P \| Q_{\boldsymbol{\theta}})$ via maximum likelihood—unbiased from data samples and mode-covering.
- **Unnormalized target $p(\mathbf{z} \mid \mathbf{x}) \propto \tilde{p}(\mathbf{z}, \mathbf{x})$ with tractable model $q_{\boldsymbol{\theta}}$:** Minimize **reverse KL** $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \| P)$ via variational inference (ELBO)—mode-seeking without needing the target normalizer.
- **Implicit generator producing samples only:** Compare expectations over test functions using **kernel MMD** (closed-form U-statistic) or **Wasserstein-1 ($W_1$)** via a 1-Lipschitz critic that provides steady gradients even on disjoint supports.
- **Energy-based model $\tilde{q}_{\boldsymbol{\theta}}(\mathbf{x}) = e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}$ with intractable $Z$:** Match log-density slopes $s_Q(\mathbf{x}) = -\nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x})$ via **Fisher divergence** and Stein integration by parts.
:::

::: {.slide title="Running Benchmarks and Study Paths"}
[How to Navigate This Chapter]{.kicker}

Every theoretical bound in this chapter is tested on concrete, reproducible benchmarks:

- **Discrete benchmarks:** The dyadic source $(\tfrac12, \tfrac14, \tfrac18, \tfrac18)$, the asymmetric pair $P = (0.6, 0.3, 0.1)$ vs. $Q = (0.2, 0.5, 0.3)$, the flipped bit ($\delta = 0.1$), and character-level compression of *The Time Machine*.
- **Continuous benchmarks:** Shifted and variance-scaled Gaussians $\mathcal{N}(0, 1)$ vs. $\mathcal{N}(d, \sigma^2)$, bimodal Gaussian mixtures ($70/30$ split), and the correlated bivariate Gaussian pair at $\rho = 0.9$ and $\rho = 0.99$.
- **Three modular tracks:**
  1. **Log-loss & language models (§§37.1–37.2):** Read straight through for supervised learning and perplexity.
  2. **Generative modeling comparisons (§§37.3–37.5):** Essential background for GANs, flow matching, and diffusion models.
  3. **Dependence & contrastive learning (§§37.6–37.7):** Self-contained after §37.1; read directly for VAEs,SimCLR, and CLIP.
:::

