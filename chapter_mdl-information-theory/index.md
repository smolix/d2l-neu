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
[Central Questions of Chapter §{num:chap_mdl-information-theory}]{.kicker}

::: {.cols .vc}
::: {.col}
- **What does a loss number measure?** Every cross-entropy loss splits into an irreducible data floor plus a non-negative divergence gap: $\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P \| Q) \ge H(P)$.
- **Why is log-loss literally a file size?** Arithmetic coding compresses $n$ symbols under $q(x_i \mid x_{<i})$ into $n\,\textrm{CE}_2(P, Q) + 2$ bits.
- **What when KL blows up?** Optimal transport ($W_1$), kernel MMD, and score matching compare distributions from samples or slopes alone.
:::

::: {.col .fig .big}
@fig:ch37-info-ce-decomposition
:::
:::
:::

::: {.slide title="Three Core Theorems Every Practitioner Needs"}
[Core Mathematical Pillars]{.kicker}

::: {.cols .vc}
::: {.col}
- **Gibbs' inequality (§{sec_mdl-information_theory}):** $D_{\textrm{KL}}(P \| Q) \ge 0$ ($= 0 \iff P = Q$), so no classifier can beat the conditional label entropy $H(Y \mid X)$.
- **Source coding (§{sec_mdl-coding-compression}):** Kraft's inequality ($\sum_i 2^{-\ell_i} \le 1$) and arithmetic coding turn any causal LLM into a lossless compressor.
- **Variational critics (§§{sec_mdl-divergences-distances}–§{num:sec_mdl-mi-estimation}):** $f$-GAN and InfoNCE lower-bound divergences and MI from samples; the InfoNCE bound is capped at $\log N$ nats for batch size $N$.
:::

::: {.col .fig .big}
@fig:ch37-info-three-pillars
:::
:::
:::

::: {.slide title="Chapter §{num:chap_mdl-information-theory} Roadmap: Seven Connected Sections"}
[Chapter Architecture]{.kicker}

::: {.cols .vc}
::: {.col}
- **Track 1 — Log-Loss, Coding & Perplexity (§§{sec_mdl-information_theory}–§{num:sec_mdl-coding-compression}):** Surprisal, Shannon entropy, Gibbs' floor, label smoothing, distillation, Kraft's inequality, arithmetic coding, MDL, AEP, and perplexity.
- **Track 2 — Divergences, Transport & Scores (§§{sec_mdl-divergences-distances}–§{num:sec_mdl-scores}):** $f$-divergences, Pinsker's inequality, $f$-GAN, IPMs, kernel MMD, Wasserstein $W_1$, Sinkhorn OT, Stein's identity, score matching, and KSD.
- **Track 3 — Dependence & Contrastive Bounds (§§{sec_mdl-mutual-information}–§{num:sec_mdl-mi-estimation}):** Mutual information, DPI, Fano's inequality, Information Bottleneck, KSG neighbors, McAllester–Stratos ceiling, and InfoNCE.
:::

::: {.col .fig .big}
@fig:ch37-info-chapter-roadmap
:::
:::
:::

::: {.slide title="Choosing a Distribution Comparison by Access"}
[Practical Decision Guide]{.kicker}

::: {.cols .vc}
::: {.col}
- **Normalized density $q_{\boldsymbol{\theta}}(\mathbf{x}) > 0$:** Minimize **forward KL** $D_{\textrm{KL}}(P \| Q_{\boldsymbol{\theta}})$ via maximum likelihood (mode-covering).
- **Unnormalized target $\tilde{p}(\mathbf{z}, \mathbf{x})$ with tractable $q_{\boldsymbol{\theta}}$:** Minimize **reverse KL** $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \| P)$ via variational inference / ELBO (mode-seeking).
- **Implicit sample generator or EBM:** Use **kernel MMD** / **Wasserstein $W_1$** for sample-only generators, or **Fisher divergence** / **KSD** when only the score $-\nabla_{\mathbf{x}}E_{\boldsymbol{\theta}}$ is tractable.
:::

::: {.col .fig .big}
@fig:ch37-info-access-decision-tree
:::
:::
:::

::: {.slide title="Running Benchmarks and Study Paths"}
[How to Navigate This Chapter]{.kicker}

Every theoretical bound in this chapter is tested on concrete, reproducible benchmarks:

- **Discrete benchmarks:** The dyadic source $(\tfrac12, \tfrac14, \tfrac18, \tfrac18)$, the asymmetric pair $P = (0.6, 0.3, 0.1)$ vs. $Q = (0.2, 0.5, 0.3)$, the flipped bit ($\delta = 0.1$), and character-level compression of *The Time Machine*.
- **Continuous benchmarks:** Shifted and variance-scaled Gaussians $\mathcal{N}(0, 1)$ vs. $\mathcal{N}(d, \sigma^2)$, bimodal Gaussian mixtures ($70/30$ split), and the correlated bivariate Gaussian pair at $\rho = 0.9$ and $\rho = 0.99$.
- **Three modular tracks:**
  1. **Log-loss & language models (§§{sec_mdl-information_theory}–§{num:sec_mdl-coding-compression}):** Read straight through for supervised learning and perplexity.
  2. **Generative modeling comparisons (§§{sec_mdl-divergences-distances}–§{num:sec_mdl-scores}):** Essential background for GANs, flow matching, and diffusion models.
  3. **Dependence & contrastive learning (§§{sec_mdl-mutual-information}–§{num:sec_mdl-mi-estimation}):** Self-contained after §{sec_mdl-information_theory}; read directly for VAEs,SimCLR, and CLIP.
:::

