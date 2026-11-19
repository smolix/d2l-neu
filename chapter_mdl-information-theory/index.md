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

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter, with each entry
noting the section it supports. Topics beyond the main text, from model
calibration to compressors that work without a probabilistic model, are
collected in :numref:`sec_mdl-information-theory-further-reading`.

**Books**

- [Elements of Information Theory — Cover & Thomas](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) — the classic graduate reference that develops entropy, KL divergence (here called relative entropy), mutual information, and their chain rules from first principles, alongside source coding, channel capacity, and rate--distortion theory; accompanies :numref:`sec_mdl-information_theory`, :numref:`sec_mdl-coding-compression`, and :numref:`sec_mdl-mutual-information`.
- [Information Theory, Inference, and Learning Algorithms — David MacKay](https://www.inference.org.uk/mackay/itila/) — freely available online; weaves information theory and machine learning together throughout, with chapters on symbol codes and stream codes (including arithmetic coding) that accompany :numref:`sec_mdl-coding-compression`.
- [Information Theory: From Coding to Learning — Polyanskiy & Wu](https://www.cambridge.org/9781108832908) — a modern graduate text geared toward statistics and machine learning, whose treatment of f-divergences, variational representations, and data processing accompanies :numref:`sec_mdl-divergences-distances` and :numref:`sec_mdl-mutual-information`; the [MIT 6.441 lecture notes](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/) from which the book grew are freely available.
- [Computational Optimal Transport — Peyré & Cuturi](https://optimaltransport.github.io/book/) — covers Wasserstein distances, entropic regularization, and Sinkhorn iterations for data science, supporting the transport half of :numref:`sec_mdl-test-functions-transport`; the full text is also on [arXiv](https://arxiv.org/abs/1803.00567).
- [The Minimum Description Length Principle — Peter Grünwald](https://mitpress.mit.edu/9780262072816/the-minimum-description-length-principle/) — develops statistical learning as data compression through two-part and universal codes, expanding on :numref:`subsec_mdl-mdl`; the core ideas are also presented in a [freely available tutorial](https://arxiv.org/abs/math/0406077).

**Courses, tutorials, and software**

- [Stanford EE376A Information Theory — Tsachy Weissman](https://web.stanford.edu/class/ee376a/) — an accessible course with public, student-scribed [lecture notes](https://web.stanford.edu/class/ee376a/files/scribes/lecture_notes.pdf) covering entropy, prefix codes, and typical sets, accompanying :numref:`sec_mdl-information_theory` and :numref:`sec_mdl-coding-compression`.
- [Visual Information Theory — Christopher Olah](https://colah.github.io/posts/2015-09-Visual-Information/) — an intuitive visual introduction that builds from codeword lengths to entropy, cross-entropy, and mutual information; accompanies :numref:`sec_mdl-information_theory`, :numref:`sec_mdl-coding-compression`, and :numref:`sec_mdl-mutual-information`.
- [From GAN to WGAN — Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/) — a clear walkthrough comparing the KL, Jensen--Shannon, and Wasserstein losses for generative models, paralleling :numref:`sec_mdl-divergences-distances` and :numref:`sec_mdl-test-functions-transport`.
- [POT: Python Optimal Transport — Flamary et al.](https://pythonot.github.io/) — an open-source Python library providing solvers for both exact optimal transport and entropically regularized Sinkhorn iterations, useful for computing the distances of :numref:`sec_mdl-optimal-transport` on your own datasets.
- [Generative Modeling by Estimating Gradients of the Data Distribution — Yang Song](https://yang-song.net/blog/2021/score/) — an accessible tutorial explaining why score functions avoid normalizing constants and how score matching fits them to data (covering :numref:`sec_mdl-scores`) before moving on to sampling and diffusion models.

**Foundational papers**

- [Language Modeling Is Compression — Delétang et al.](https://arxiv.org/abs/2309.10668) — pairs an arithmetic coder with a language model's next-token probabilities, the exact idea demonstrated in :numref:`subsec_mdl-lm-compressor`.
- [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization — Nowozin et al.](https://arxiv.org/abs/1606.00709) — shows that any f-divergence yields an adversarial training objective via its Fenchel conjugate lower bound, as derived in :numref:`sec_mdl-f-gan-dual`.
- [A Kernel Two-Sample Test — Gretton et al.](https://jmlr.org/papers/v13/gretton12a.html) — introduces the maximum mean discrepancy, its unbiased two-sample estimator, and the resulting hypothesis test from :numref:`sec_mdl-ipm-mmd`.
- [Wasserstein GAN — Arjovsky et al.](https://arxiv.org/abs/1701.07875) — replaces the Jensen--Shannon divergence with the Wasserstein-1 distance so that gradients remain informative even when supports do not overlap; accompanies :numref:`sec_mdl-optimal-transport`.
- [Estimation of Non-Normalized Statistical Models by Score Matching — Aapo Hyvärinen](https://jmlr.org/papers/v6/hyvarinen05a.html) — fits unnormalized models by matching the gradient of the log-density to that of the data and eliminating the unknown data score via integration by parts, the two key steps of :numref:`sec_mdl-scores`.
- [A Kernelized Stein Discrepancy for Goodness-of-fit Tests — Liu, Lee & Jordan](https://proceedings.mlr.press/v48/liub16.html) — tests whether a sample comes from a model known only through its score using pairwise kernel evaluations, as presented in :numref:`sec_mdl-ksd`.
- [The Information Bottleneck Method — Tishby, Pereira & Bialek](https://arxiv.org/abs/physics/0004057) — formulates the problem of compressing $X$ while preserving information about $Y$, the framework studied in :numref:`sec_mdl-information-bottleneck`.
- [Estimating Mutual Information — Kraskov, Stögbauer & Grassberger](https://arxiv.org/abs/cond-mat/0305641) — estimates the mutual information of continuous variables from nearest-neighbor distances, the estimator of :numref:`subsec_mdl-ksg` on which scikit-learn's [`mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) is built.
- [On Variational Bounds of Mutual Information — Poole et al.](https://arxiv.org/abs/1905.06922) — unifies the Barber--Agakov, Donsker--Varadhan, NWJ, and InfoNCE bounds in a single framework and analyzes their bias--variance trade-offs, accompanying :numref:`sec_mdl-mi-variational-bounds`.
- [Representation Learning with Contrastive Predictive Coding — van den Oord et al.](https://arxiv.org/abs/1807.03748) — introduces the InfoNCE loss as a lower bound on mutual information for self-supervised representation learning, as developed in :numref:`sec_mdl-infonce`.
