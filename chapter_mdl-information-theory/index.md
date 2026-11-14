# Information Theory and Divergences
:label:`chap_mdl-information-theory`

Most models in this book were trained by minimizing a cross-entropy. We
chose it because it works: it turns "give high probability to what
actually happened" into a number that gradient descent can push down. We
never asked what that number means. When a language model reports a loss of
1.2 nats per character, is that good? How low could it go? And when we
compare two distributions in other ways, with a KL penalty in a variational
autoencoder or with a discriminator in a generative adversarial network, what
exactly are we measuring?

Information theory answers these questions, although it began with a
different one. In 1948 Claude Shannon asked how efficiently a message can be
sent over a telephone line, and found that the answer depends on a single
property of the source: how predictable it is. His measure of
unpredictability, *entropy*, is the average surprise of an outcome. It is
also the floor below which no model's cross-entropy can fall.

The main text states this floor without proof
(:numref:`subsec_info_theory_basics`). This chapter proves it, and three of
its results are worth carrying back there. A cross-entropy exceeds its floor
by the *KL divergence* between the data and the model, which is zero only
for the right model. Count a model's total loss on a message in bits, and
the message can be stored in that many bits plus at most two, so the loss of
a language model (:numref:`subsec_perplexity`) is a file size. And a
discriminator reports only a lower bound on how far apart two distributions
are: to a weak one, distant distributions look close. Adversarial training
(:numref:`sec_gan_objectives`) rests on such bounds, and so does contrastive
learning, where picking the true partner of an example among $N$ candidates
certifies at most $\log N$ nats of shared information.

The first of the chapter's three parts asks what a cross-entropy measures
(:numref:`sec_mdl-information_theory`) and why it is the length of a file
(:numref:`sec_mdl-coding-compression`). The second begins with a defect.
The KL divergence is infinite as soon as a model gives zero probability to
something that occurs. If the model's samples miss the data altogether, no
other function of the ratio of the two densities can tell a near miss from a
distant one (:numref:`sec_mdl-divergences-distances`). Comparing averages of
a function under the two distributions can, and so can the cost of moving
probability mass between them (:numref:`sec_mdl-test-functions-transport`).
A model whose density is known only up to a constant factor is compared
through the slope of its log-density (:numref:`sec_mdl-scores`).

The third part turns to two variables: how much one tells about the other
(:numref:`sec_mdl-mutual-information`), which is what the KL penalty of an
autoencoder limits, and how much of that samples can show
(:numref:`sec_mdl-mi-estimation`). It uses nothing from the two sections
just before it, which can wait until :numref:`chap_gans` or
:numref:`chap_diffusion` sends you there.

We assume Jensen's inequality (:numref:`chap_mdl-optimization`), conditional
distributions and maximum likelihood
(:numref:`chap_mdl-probability-statistics`). Entropy and the KL divergence
are developed from the beginning. Practical error-correcting codes,
Kolmogorov complexity and information geometry are left to
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

The resources below accompany the sections of this chapter; each names the
section it goes with. What the chapter does not teach, from calibration to
compressors that need no model, is collected in
:numref:`sec_mdl-information-theory-further-reading`.

**Books**

- [Elements of Information Theory — Cover & Thomas](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) — the graduate reference that develops entropy, the KL divergence (which it calls relative entropy), mutual information and their chain rules from first principles, with chapters on source coding, channel capacity and rate--distortion; it accompanies :numref:`sec_mdl-information_theory`, :numref:`sec_mdl-coding-compression` and :numref:`sec_mdl-mutual-information`.
- [Information Theory, Inference, and Learning Algorithms — David MacKay](https://www.inference.org.uk/mackay/itila/) — free to read online; treats information theory and machine learning together, and its chapters on symbol codes and stream codes (including arithmetic coding) accompany :numref:`sec_mdl-coding-compression`.
- [Information Theory: From Coding to Learning — Polyanskiy & Wu](https://www.cambridge.org/9781108832908) — a modern graduate text oriented toward statistics and learning, whose treatment of f-divergences, their variational forms and data processing accompanies :numref:`sec_mdl-divergences-distances` and :numref:`sec_mdl-mutual-information`; the lecture notes of [MIT 6.441](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/), from which the book grew, are free.
- [Computational Optimal Transport — Peyré & Cuturi](https://optimaltransport.github.io/book/) — Wasserstein distances, entropic regularization and Sinkhorn iterations for data science, the transport part of :numref:`sec_mdl-test-functions-transport`; the full text is on [arXiv](https://arxiv.org/abs/1803.00567).
- [The Minimum Description Length Principle — Peter Grünwald](https://mitpress.mit.edu/9780262072816/the-minimum-description-length-principle/) — develops learning as data compression through two-part and universal codes, the subject of :numref:`subsec_mdl-mdl`; the core ideas also appear in a [freely available tutorial](https://arxiv.org/abs/math/0406077).

**Courses, tutorials and software**

- [Stanford EE376A Information Theory — Tsachy Weissman](https://web.stanford.edu/class/ee376a/) — an undergraduate-to-graduate course with public, student-scribed [lecture notes](https://web.stanford.edu/class/ee376a/files/scribes/lecture_notes.pdf) on entropy, prefix codes and the typical set, for :numref:`sec_mdl-information_theory` and :numref:`sec_mdl-coding-compression`.
- [Visual Information Theory — Christopher Olah](https://colah.github.io/posts/2015-09-Visual-Information/) — a visual introduction that reaches entropy and cross-entropy through the lengths of codewords and ends with mutual information, for :numref:`sec_mdl-information_theory`, :numref:`sec_mdl-coding-compression` and :numref:`sec_mdl-mutual-information`.
- [From GAN to WGAN — Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/) — compares the KL, Jensen--Shannon and Wasserstein losses of generative models, the comparison of :numref:`sec_mdl-divergences-distances` and :numref:`sec_mdl-test-functions-transport`.
- [POT: Python Optimal Transport — Flamary et al.](https://pythonot.github.io/) — a library with solvers for the linear program of transport and for its entropic regularization by Sinkhorn iterations, for computing the distances of :numref:`sec_mdl-optimal-transport` on your own data.
- [Generative Modeling by Estimating Gradients of the Data Distribution — Yang Song](https://yang-song.net/blog/2021/score/) — a tutorial post on why a score needs no normalizing constant and how score matching fits one to data, the material of :numref:`sec_mdl-scores`, before it goes on to sampling and diffusion models.

**Foundational papers**

- [Language Modeling Is Compression — Delétang et al.](https://arxiv.org/abs/2309.10668) — drives an arithmetic coder with a language model's next-token probabilities, the construction of :numref:`subsec_mdl-lm-compressor`.
- [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization — Nowozin et al.](https://arxiv.org/abs/1606.00709) — shows that every f-divergence yields an adversarial objective through its Fenchel lower bound, the bound of :numref:`sec_mdl-f-gan-dual`.
- [A Kernel Two-Sample Test — Gretton et al.](https://jmlr.org/papers/v13/gretton12a.html) — defines the maximum mean discrepancy, its unbiased estimate from two samples and the test built on it, the comparison of :numref:`sec_mdl-ipm-mmd`.
- [Wasserstein GAN — Arjovsky et al.](https://arxiv.org/abs/1701.07875) — replaces the Jensen--Shannon divergence by the Wasserstein-1 distance, which stays informative when the supports do not overlap; it accompanies :numref:`sec_mdl-optimal-transport`.
- [Estimation of Non-Normalized Statistical Models by Score Matching — Aapo Hyvärinen](https://jmlr.org/papers/v6/hyvarinen05a.html) — fits a model whose normalizing constant is unknown by matching the slope of its log-density to that of the data, and removes the unknown slope of the data by an integration by parts, the two steps of :numref:`sec_mdl-scores`.
- [A Kernelized Stein Discrepancy for Goodness-of-fit Tests — Liu, Lee & Jordan](https://proceedings.mlr.press/v48/liub16.html) — tests a sample against a model that is known only through its score, with a statistic computed from kernel evaluations on pairs of data points, the test of :numref:`sec_mdl-ksd`.
- [The Information Bottleneck Method — Tishby, Pereira & Bialek](https://arxiv.org/abs/physics/0004057) — the original formulation of compressing $X$ while keeping information about $Y$, the problem of :numref:`sec_mdl-information-bottleneck`.
- [Estimating Mutual Information — Kraskov, Stögbauer & Grassberger](https://arxiv.org/abs/cond-mat/0305641) — estimates the mutual information of two continuous variables from the distances between each sample and its nearest neighbors, the estimator of :numref:`subsec_mdl-ksg`; scikit-learn's [`mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) is built on it.
- [On Variational Bounds of Mutual Information — Poole et al.](https://arxiv.org/abs/1905.06922) — derives the Barber--Agakov, Donsker--Varadhan, NWJ and InfoNCE bounds in one framework and compares their bias and variance, the material of :numref:`sec_mdl-mi-variational-bounds`.
- [Representation Learning with Contrastive Predictive Coding — van den Oord et al.](https://arxiv.org/abs/1807.03748) — introduces the InfoNCE loss as a lower bound on mutual information for self-supervised learning, the bound of :numref:`sec_mdl-infonce`.
