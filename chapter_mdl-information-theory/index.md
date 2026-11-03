# Information Theory and Divergences
:label:`chap_mdl-information-theory`

A freshly initialized 10-class classifier reports a loss of $2.30$. A
generative adversarial network whose samples miss the data by $0.01$ sees
the same Jensen--Shannon divergence, $\log 2$, as one whose samples miss by
$100$, so its loss cannot tell a near miss from a far one. A contrastive
model trained with $128$ candidates per example can never report more than
$\ln 128 \approx 4.85$ nats of shared information, however much its two
views share. Each number is the value of a familiar loss, and each makes a
precise statement about probability distributions.

The Basics part used these quantities as losses to minimize. Read only as
numbers to push down, they cannot say whether a plateau is a property of
the data or a defect of the model, whether a comparison of two
distributions can see how far apart they are, or whether a reported value
is a measurement or a bound. Information theory gives each of them a
meaning, a floor and a failure mode. Two small examples recur: the
distributions $(0.6, 0.3, 0.1)$ and $(0.2, 0.5, 0.3)$ on three outcomes,
and a pair of standard normal variables with correlation $\rho$.

:numref:`sec_mdl-information_theory` asks what a log-loss measures and how
low it can go. :numref:`sec_mdl-coding-compression` asks why a log-loss is
the length of a code, and how close a real compressor comes to it.
:numref:`sec_mdl-divergences-distances` asks what else two distributions can
be compared by, and which comparisons see a near miss.
:numref:`sec_mdl-mutual-information` asks how to measure the dependence
between two variables, and what that measure limits in processing,
prediction, communication and compression. :numref:`sec_mdl-mi-estimation`
asks what samples alone can say about dependence.

We assume the probability of :numref:`chap_mdl-probability-statistics` and
the convexity of :numref:`chap_mdl-optimization`. The construction of
practical channel codes, Kolmogorov complexity and the information geometry
of statistical models are left out.

```toc
:maxdepth: 2

mdl-information-theory
mdl-coding-compression
mdl-divergences-distances
mdl-mutual-information
mdl-mi-estimation
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. Topics that the chapter does not teach, such as
universal compressors, Rényi divergences, upper bounds on mutual
information and the information-bottleneck view of deep networks, are
collected in :numref:`sec_mdl-information-theory-further-reading`.

**Books**

- [Elements of Information Theory — Cover & Thomas](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) — the graduate reference that develops entropy, relative entropy, mutual information and their chain rules from first principles, with chapters on source coding, channel capacity and rate–distortion; it accompanies :numref:`sec_mdl-information_theory`, :numref:`sec_mdl-coding-compression` and :numref:`sec_mdl-mutual-information`.
- [Information Theory, Inference, and Learning Algorithms — David MacKay](https://www.inference.org.uk/mackay/itila/) — free to read online; treats information theory and machine learning together, and its chapters on symbol codes and stream codes (including arithmetic coding) accompany :numref:`sec_mdl-coding-compression`.
- [Information Theory: From Coding to Learning — Polyanskiy & Wu](https://www.cambridge.org/9781108832908) — a modern graduate text oriented toward statistics and learning, whose treatment of f-divergences, their variational forms and data processing accompanies :numref:`sec_mdl-divergences-distances` and :numref:`sec_mdl-mutual-information`.
- [Computational Optimal Transport — Peyré & Cuturi](https://optimaltransport.github.io/book/) — Wasserstein distances, entropic regularization and Sinkhorn iterations for data science, the transport part of :numref:`sec_mdl-divergences-distances`; the full text is on [arXiv](https://arxiv.org/abs/1803.00567).
- [The Minimum Description Length Principle — Peter Grünwald](https://mitpress.mit.edu/9780262072816/the-minimum-description-length-principle/) — develops learning as data compression through two-part and universal codes, the subject of :numref:`subsec_mdl-mdl`; the core ideas also appear in a [freely available tutorial](https://arxiv.org/abs/math/0406077).

**Courses and lecture notes**

- [MIT 6.441 Information Theory — Polyanskiy & Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/) — free lecture notes, the precursor of the book above, on information measures, hypothesis testing and their links to statistical learning; they accompany :numref:`sec_mdl-divergences-distances` and :numref:`sec_mdl-mutual-information`.
- [Stanford EE376A Information Theory — Tsachy Weissman](https://web.stanford.edu/class/ee376a/) — an undergraduate-to-graduate course with public, student-scribed [lecture notes](https://web.stanford.edu/class/ee376a/files/scribes/lecture_notes.pdf) on entropy, prefix codes and the typical set, for :numref:`sec_mdl-information_theory` and :numref:`sec_mdl-coding-compression`.

**Tutorials and blogs**

- [Visual Information Theory — Christopher Olah](https://colah.github.io/posts/2015-09-Visual-Information/) — a visual introduction to entropy, cross-entropy, KL divergence and mutual information, for :numref:`sec_mdl-information_theory` and :numref:`sec_mdl-mutual-information`.
- [From GAN to WGAN — Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/) — compares the KL, Jensen--Shannon and Wasserstein losses of generative models, the comparison of :numref:`sec_mdl-divergences-distances`.

**Foundational papers**

- [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization — Nowozin et al.](https://arxiv.org/abs/1606.00709) — shows that every f-divergence yields an adversarial objective through its Fenchel lower bound, the bound of :numref:`sec_mdl-f-gan-dual`.
- [Wasserstein GAN — Arjovsky et al.](https://arxiv.org/abs/1701.07875) — replaces the Jensen--Shannon divergence by the Wasserstein-1 distance, which stays informative when the supports do not overlap; it accompanies :numref:`sec_mdl-optimal-transport`.
- [Language Modeling Is Compression — Delétang et al.](https://arxiv.org/abs/2309.10668) — drives an arithmetic coder with a language model's next-token probabilities, the construction of :numref:`subsec_mdl-lm-compressor`.
- [The Information Bottleneck Method — Tishby, Pereira & Bialek](https://arxiv.org/abs/physics/0004057) — the original formulation of compressing $X$ while keeping information about $Y$, the problem of :numref:`sec_mdl-information-bottleneck`.
- [On Variational Bounds of Mutual Information — Poole et al.](https://arxiv.org/abs/1905.06922) — derives the Barber–Agakov, Donsker–Varadhan, NWJ and InfoNCE bounds in one framework and compares their bias and variance, the material of :numref:`sec_mdl-mi-variational-bounds`.
- [Representation Learning with Contrastive Predictive Coding — van den Oord et al.](https://arxiv.org/abs/1807.03748) — introduces the InfoNCE loss as a lower bound on mutual information for self-supervised learning, the bound of :numref:`sec_mdl-infonce`.
