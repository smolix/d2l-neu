# Information Theory and Divergences
:label:`chap_mdl-information-theory`

Information theory is the language of losses. This chapter builds entropy,
cross-entropy, and the Kullback--Leibler divergence, shows that minimizing
cross-entropy *is* maximum likelihood, and then broadens to the wider family of
divergences and distances (f-divergences, optimal transport, integral
probability metrics) that define modern generative objectives, and to mutual
information and the contrastive objectives of representation learning.

```toc
:maxdepth: 2

mdl-information-theory
mdl-divergences-distances
mdl-mutual-information
```

## Resources and Further Reading

A short, curated reading list for information theory and divergences as they appear in machine and deep learning: entropy and cross-entropy, the KL and broader $f$-divergences, mutual information, optimal transport, and the information bottleneck.

**Books**

- [Elements of Information Theory — Cover & Thomas](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) — The standard graduate reference; entropy, relative entropy, mutual information, and their chain rules, all developed from first principles.
- [Information Theory, Inference, and Learning Algorithms — David MacKay](https://www.inference.org.uk/mackay/itila/) — Free to read online; the rare book that treats information theory and machine learning as one subject, with a strong Bayesian and coding-theoretic flavor.
- [Information Theory: From Coding to Learning — Polyanskiy & Wu](https://www.cambridge.org/9781108832908) — A modern graduate text deliberately oriented toward statistics and learning; the freely available [MIT 6.441 lecture notes](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/) are its precursor.
- [Computational Optimal Transport — Peyré & Cuturi](https://optimaltransport.github.io/book/) — The standard reference on OT for data science; Wasserstein distances, entropic regularization, and Sinkhorn, with the full text on [arXiv](https://arxiv.org/abs/1803.00567).
- [The Minimum Description Length Principle — Peter Grünwald](https://mitpress.mit.edu/9780262072816/the-minimum-description-length-principle/) — The definitive account of learning as data compression: two-part codes, universal coding, and why shorter descriptions generalize; the core ideas are also in his [freely available tutorial](https://arxiv.org/abs/math/0406077).

**Courses and lecture notes**

- [MIT 6.441 Information Theory — Polyanskiy & Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/) — Free OCW lecture notes (single PDF or per-chapter) covering information measures, hypothesis testing, and the bridge to statistical learning.
- [Stanford EE376A Information Theory — Tsachy Weissman](https://web.stanford.edu/class/ee376a/) — An undergraduate-to-graduate course with public, student-scribed [lecture notes](https://web.stanford.edu/class/ee376a/files/scribes/lecture_notes.pdf); intuition-first, with applications throughout.

**Tutorials, blogs, and surveys**

- [Visual Information Theory — Christopher Olah](https://colah.github.io/posts/2015-09-Visual-Information/) — The clearest visual introduction to entropy, cross-entropy, KL divergence, and mutual information; ideal for building intuition before the formalism.
- [From GAN to WGAN — Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/) — Walks through KL, Jensen--Shannon, and Wasserstein distances and exactly why a generative objective might prefer one over another.

**Foundational papers**

- [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization — Nowozin et al.](https://arxiv.org/abs/1606.00709) — Shows that any $f$-divergence yields a GAN-style objective via its variational (Fenchel) lower bound, unifying many adversarial losses.
- [Wasserstein GAN — Arjovsky et al.](https://arxiv.org/abs/1701.07875) — Replaces JS divergence with the Earth-Mover (Wasserstein-1) distance, giving smoother gradients and meaningful loss curves when supports do not overlap.
- [Language Modeling Is Compression — Delétang et al.](https://arxiv.org/abs/2309.10668) — Drives an arithmetic coder with an LLM's next-token probabilities and obtains a state-of-the-art general-purpose compressor, making "cross-entropy is a code length" literal at scale.
- [Representation Learning with Contrastive Predictive Coding — van den Oord et al.](https://arxiv.org/abs/1807.03748) — Introduces the InfoNCE loss, framing contrastive self-supervision as maximizing a tractable lower bound on mutual information.
- [The Information Bottleneck Method — Tishby, Pereira & Bialek](https://arxiv.org/abs/physics/0004057) — The original formulation: compress $X$ while preserving information about $Y$, generalizing rate--distortion theory.
- [Deep Learning and the Information Bottleneck Principle — Tishby & Zaslavsky](https://arxiv.org/abs/1503.02406) — Recasts deep networks as a sequence of information bottlenecks, an influential (and debated) lens on representation and generalization.
