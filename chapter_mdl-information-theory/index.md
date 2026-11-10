# Information Theory and Divergences
:label:`chap_mdl-information-theory`

Almost every model in this book was trained by minimizing a cross-entropy.
We chose it because it works: it turns "give high probability to what
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

The main text needs little of this
(:numref:`subsec_info_theory_basics`); the chapter is for readers who want
the proofs. Three results matter most. Gibbs' inequality puts that floor
under every log-loss. Arithmetic coding stores a message in its log-loss plus
at most two bits, so the loss behind the perplexity of
:numref:`subsec_perplexity` is a file size. And a critic trained on samples
bounds a divergence from below: adversarial training
(:numref:`sec_gan_objectives`) and contrastive learning rest on such bounds,
and with $N$ candidates a contrastive bound never exceeds $\log N$.

The first part stays with one distribution and one model: what a log-loss
measures and why it is a file length (:numref:`sec_mdl-information_theory`,
:numref:`sec_mdl-coding-compression`). The second, from
:numref:`sec_mdl-divergences-distances` on, asks what to use when the KL
divergence is infinite: other functions of the density ratio, test functions
and transport, and scores. The third turns to two variables: how much one
tells about the other (:numref:`sec_mdl-mutual-information`), and how much of
that samples can show (:numref:`sec_mdl-mi-estimation`). It needs from the
second only data processing and the critic bound, so transport and scores can
wait until :numref:`chap_gans` or :numref:`chap_diffusion` sends you back.

We assume the convexity of :numref:`chap_mdl-optimization` and the
probability of :numref:`chap_mdl-probability-statistics`, where
:numref:`sec_mdl-exponential-family` proves Gibbs' inequality for its own
use; entropy and the KL divergence are developed here from the beginning. We
leave out practical error-correcting codes, Kolmogorov complexity and
information geometry.

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
