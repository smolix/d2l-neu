# Information Theory and Divergences
:label:`chap_mdl-information-theory`

Information theory provides a common language for many learning objectives.
This chapter begins with entropy, cross-entropy, and the Kullback--Leibler
divergence, including the equivalence between cross-entropy minimization and
maximum likelihood. It then studies broader families of discrepancies between
distributions, including $f$-divergences, integral probability metrics, and
optimal transport, before turning to mutual information and contrastive
objectives for representation learning.

The quantity should follow the question:

| Question | Quantity | What its notion of error retains |
|:--|:--|:--|
| How many bits are needed under a probabilistic model? | Entropy, cross-entropy, KL | Log-probability and coding regret |
| How should two distributions be compared? | f-divergence, IPM, or transport distance | Density ratios, test-function expectations, or sample-space geometry |
| Do two variables share information? | Mutual information | Departure of the joint law from the product of its marginals |

These quantities are not interchangeable losses. Their support assumptions, invariances, and estimators determine what a small numerical value means.

```toc
:maxdepth: 2

mdl-information-theory
mdl-divergences-distances
mdl-mutual-information
```

## Resources and Further Reading {.unnumbered}

The following references cover information theory and divergences as they
arise in machine learning, including mutual information, optimal transport,
and the information bottleneck.

**Books**

- [Elements of Information Theory — Cover & Thomas](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) — a graduate reference that develops entropy, relative entropy, mutual information, and their chain rules from first principles.
- [Information Theory, Inference, and Learning Algorithms — David MacKay](https://www.inference.org.uk/mackay/itila/) — free to read online; treats information theory and machine learning together, with Bayesian and coding-theoretic perspectives.
- [Information Theory: From Coding to Learning — Polyanskiy & Wu](https://www.cambridge.org/9781108832908) — a modern graduate text oriented toward statistics and learning; the freely available [MIT 6.441 lecture notes](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/) are its precursor.
- [Computational Optimal Transport — Peyré & Cuturi](https://optimaltransport.github.io/book/) — covers Wasserstein distances, entropic regularization, and Sinkhorn algorithms for data science; the full text is on [arXiv](https://arxiv.org/abs/1803.00567).
- [The Minimum Description Length Principle — Peter Grünwald](https://mitpress.mit.edu/9780262072816/the-minimum-description-length-principle/) — develops learning as data compression through two-part codes and universal coding; the core ideas also appear in a [freely available tutorial](https://arxiv.org/abs/math/0406077).

**Courses and lecture notes**

- [MIT 6.441 Information Theory — Polyanskiy & Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/) — free OCW lecture notes covering information measures, hypothesis testing, and connections to statistical learning.
- [Stanford EE376A Information Theory — Tsachy Weissman](https://web.stanford.edu/class/ee376a/) — an undergraduate-to-graduate course with public, student-scribed [lecture notes](https://web.stanford.edu/class/ee376a/files/scribes/lecture_notes.pdf) and applications throughout.

**Tutorials, blogs, and surveys**

- [Visual Information Theory — Christopher Olah](https://colah.github.io/posts/2015-09-Visual-Information/) — a visual introduction to entropy, cross-entropy, KL divergence, and mutual information.
- [From GAN to WGAN — Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/) — compares KL, Jensen--Shannon, and Wasserstein distances for generative objectives.

**Foundational papers**

- [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization — Nowozin et al.](https://arxiv.org/abs/1606.00709) — Shows that any $f$-divergence yields a GAN-style objective via its variational (Fenchel) lower bound, unifying many adversarial losses.
- [Wasserstein GAN — Arjovsky et al.](https://arxiv.org/abs/1701.07875) — Replaces JS divergence with the Earth-Mover (Wasserstein-1) distance, giving smoother gradients and meaningful loss curves when supports do not overlap.
- [Language Modeling Is Compression — Delétang et al.](https://arxiv.org/abs/2309.10668) — uses an LLM's next-token probabilities to drive an arithmetic coder and connects cross-entropy directly to compression performance.
- [Representation Learning with Contrastive Predictive Coding — van den Oord et al.](https://arxiv.org/abs/1807.03748) — Introduces the InfoNCE loss, framing contrastive self-supervision as maximizing a tractable lower bound on mutual information.
- [The Information Bottleneck Method — Tishby, Pereira & Bialek](https://arxiv.org/abs/physics/0004057) — The original formulation: compress $X$ while preserving information about $Y$, generalizing rate--distortion theory.
- [Deep Learning and the Information Bottleneck Principle — Tishby & Zaslavsky](https://arxiv.org/abs/1503.02406) — Recasts deep networks as a sequence of information bottlenecks, an influential (and debated) lens on representation and generalization.
