# Cover-page "Resources and Further Reading" sections — ch. 3, 4, 5

Ready-to-paste blocks for the three chapter cover pages, in the Preliminaries /
math-appendix house style (grouped **Books** / **Courses and video lectures** /
**Foundational papers** / **Tutorials, notes, and documentation**; each entry is
`[Title — Author/Org](url)` + an em-dash annotation on *relevance to this
chapter* and free/paywalled status; freely-accessible sources preferred).

- **Ch. 3** and **Ch. 4** already ship a block — included here with the **one
  required fix** (ch. 4's retired info-theory link) and a short list of optional
  net-new entries.
- **Ch. 5 (MLP)** has **no block today** — the full block below is **new**, ready
  to insert into `chapter_multilayer-perceptrons/index.md` after the `toc`.

Scope discipline applies to resource curation too: entries point readers *deeper
on this chapter's own material*, and forward-looking entries (e.g. KAN as a modern
MLP alternative) are included sparingly as "context," mirroring how the Linear
Algebra cover lists LoRA as a modern payoff.

---

## Ch. 5 — Multilayer Perceptrons  *(NEW — insert into `index.md`)*

Insert after the closing ```` ``` ```` of the `toc` block:

```markdown
## Resources and Further Reading

The references below develop the multilayer perceptron and the core mechanics of
training deep networks that this chapter introduces---hidden layers and
activation functions, the universal approximation theorem, backpropagation,
initialization and numerical stability, modern generalization, and dropout. The
optimizers, normalization layers, and architectures these topics point toward are
developed in their own later parts of the book; the sources here go deeper on the
foundations. All are freely accessible online except where noted.

**Books**

- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) — free HTML; Chapter 6 (Deep Feedforward Networks) is the canonical reference for hidden units, activation functions, the universal approximation theorem, and backpropagation as reverse-mode differentiation; Chapter 7 covers dropout and regularization.
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/) — free online; Chapter 2 derives the four equations of backpropagation from first principles with unusual clarity---the best companion to this chapter's backprop section.
- [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) — free PDF; a modern (2023), figure-rich treatment whose early chapters on shallow and deep networks, initialization, and training map directly onto this chapter.
- [Deep Learning: Foundations and Concepts — Bishop & Bishop](https://www.bishopbook.com/) — free online edition; a comprehensive, probabilistically-flavoured (2024) account of MLPs, backpropagation, initialization, and regularization.

**Courses and video lectures**

- [CMU 11-785 Introduction to Deep Learning — Bhiksha Raj et al.](https://deeplearning.cs.cmu.edu/) — free slides and recordings; devotes multiple lectures to the perceptron, universal approximation, and backpropagation at exactly this chapter's depth.
- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/) — free course notes; the "Neural Networks" and "Backpropagation, Intuitions" modules are the most-cited online treatment of the computational-graph view of backprop and the activation-function comparison.
- [MIT 6.S191: Introduction to Deep Learning — Amini & Amini](https://introtodeeplearning.com/) — free, updated annually; the opening lectures build the MLP and training loop from scratch with polished visuals.
- [NYU Deep Learning — LeCun & Canziani](https://atcold.github.io/NYU-DLSP21/) — free notebooks and videos; gradient descent, backpropagation, and training taught by two of the field's founders.

**Foundational papers**

- [Learning Representations by Back-propagating Errors — Rumelhart, Hinton & Williams (1986), *Nature*](https://www.nature.com/articles/323533a0) — the paper that brought backpropagation and learned hidden representations to a broad audience (paywalled, noted; widely reproduced online).
- [Approximation by Superpositions of a Sigmoidal Function — Cybenko (1989)](https://doi.org/10.1007/BF02551274) — the first proof of the universal approximation theorem for single-hidden-layer sigmoid networks (paywalled, noted).
- [Approximation Capabilities of Multilayer Feedforward Networks — Hornik (1991)](https://doi.org/10.1016/0893-6080(91)90009-T) — extends universal approximation to essentially any non-polynomial activation (paywalled, noted).
- [Understanding the Difficulty of Training Deep Feedforward Networks — Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) — free PDF; derives Xavier/Glorot initialization from variance preservation and explains sigmoid-saturation vanishing gradients---the backbone of this chapter's initialization section.
- [Delving Deep into Rectifiers — He, Zhang, Ren & Sun (2015)](https://arxiv.org/abs/1502.01852) — free; derives He/Kaiming initialization, the current default for ReLU networks.
- [Rectified Linear Units Improve Restricted Boltzmann Machines — Nair & Hinton (2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) — free; the systematic case for ReLU as a hidden-unit nonlinearity.
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting — Srivastava et al. (2014), *JMLR*](https://jmlr.org/papers/v15/srivastava14a.html) — free; the dropout paper, with the ensemble and co-adaptation interpretations this chapter uses.
- [Reconciling Modern Machine-Learning Practice and the Bias–Variance Trade-off — Belkin, Hsu, Ma & Mandal (2019), *PNAS*](https://arxiv.org/abs/1812.11118) — free; introduces the double-descent curve central to this chapter's modern-generalization section.

**Tutorials, notes, and interactive**

- [TensorFlow Playground — Smilkov & Carter](https://playground.tensorflow.org/) — free, zero-install; tune depth, width, activation, and regularization and watch a decision boundary train in real time---the fastest way to build hidden-layer intuition.
- [Backpropagation, Intuitions — Stanford CS231n](https://cs231n.github.io/optimization-2/) — free; the gate/computational-graph view with worked numerical examples, the perfect concrete complement to this chapter's backprop derivation.
- [Double Descent — MLU-Explain](https://mlu-explain.github.io/double-descent/) — free, interactive; animates the interpolation threshold and second descent, the best visual companion to the modern-generalization section.
- [KAN: Kolmogorov–Arnold Networks — Liu et al. (2024)](https://arxiv.org/abs/2404.19756) — free; a recent alternative that places learnable activations on edges rather than fixed activations on nodes---useful context for where the MLP sits among modern parameterizations.
```

---

## Ch. 4 — Linear Neural Networks for Classification  *(EXISTING — apply one fix)*

The current block (`chapter_linear-classification/index.md`) is good. **Required
fix (flagged by the §4.1 review):** the "Information Theory" entry links the
**retired legacy `d2l.ai` URL**; repoint it to the book's own appendix chapter.

**Replace** the current entry:

```markdown
- [Information Theory — *Dive into Deep Learning*](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html) — this book's own chapter ...
```

**with** (in-book pointer to the live appendix chapter):

```markdown
- [Information Theory — *this book's Mathematics for Deep Learning appendix*](../chapter_mdl-information-theory/index.html) — entropy, cross-entropy, and KL divergence developed in full (deck-of-cards hook, nats, Kraft coding, perplexity, label smoothing); the information-theoretic background behind the softmax cross-entropy loss. *(Source link should use `:numref:`sec_mdl-information_theory`` once the cross-ref renders on the cover page.)*
```

**Optional net-new entries** (specialist, non-overlapping with ch. 3's classics):

```markdown
**Books** (add)

- [Understanding Machine Learning: From Theory to Algorithms — Shalev-Shwartz & Ben-David](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) — free PDF; the rigorous reference for the PAC/VC/Rademacher learning theory in this chapter's generalization section, and a full treatment of domain adaptation for the distribution-shift section.
- [Information Theory, Inference, and Learning Algorithms — David MacKay](https://www.inference.org.uk/mackay/itila/) — free PDF; the classic, intuition-first development of entropy and cross-entropy underlying the softmax loss.

**Foundational papers** (add)

- [WILDS: A Benchmark of in-the-Wild Distribution Shifts — Koh et al. (2021)](https://arxiv.org/abs/2012.07421) — free; the standard modern benchmark for the covariate/label/concept shift this chapter introduces, and the bridge to how shift matters for today's foundation models.
```

---

## Ch. 3 — Linear Neural Networks for Regression  *(EXISTING — already strong)*

The current block (`chapter_linear-regression/index.md`) already meets the bar
(ISL, ESL, Bishop PRML, Murphy, VMLS; Ng & Hastie/Tibshirani courses; the
Hoerl–Kennard ridge and Tibshirani lasso origin papers). **No fix required.**

**Optional net-new candidates** (only if the author wants to round it out):

```markdown
**Tutorials, notes, and documentation** (add)

- [Linear Regression — *this book's Probability & Statistics appendix*](../chapter_mdl-probability-statistics/index.html) — the maximum-likelihood and statistical-estimation foundations behind the squared-loss/Gaussian-noise derivation in §3.1 and the MAP/ridge connection in §3.7.
- [Least Squares, Pseudoinverse, and the Four Subspaces — Gilbert Strang (MIT 18.06, OCW)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — free; the geometry-of-least-squares (projection onto the column space) view that complements §3.1's analytic solution.
```

*(These are low priority — the existing ch. 3 block is the strongest of the three
and should not be padded just for symmetry.)*
