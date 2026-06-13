# Multilayer Perceptrons
:label:`chap_perceptrons`

In this chapter, we will introduce your first truly *deep* network.
The simplest deep networks are called *multilayer perceptrons*,
and they consist of multiple layers of neurons
each fully connected to those in the layer below
(from which they receive input)
and those above (which they, in turn, influence).
Although automatic differentiation
significantly simplifies the implementation of deep learning algorithms,
we will dive deep into how these gradients
are calculated in deep networks.
Then we will
be ready to
discuss issues relating to numerical stability and parameter initialization
that are key to successfully training deep networks.
When we train such high-capacity models we run the risk of overfitting. Thus, we will
revisit regularization and generalization
for deep networks.
Throughout, we aim
to give you a firm grasp not just of the concepts but also of the practice of using deep networks.
At the end of this chapter, we apply what we have introduced so far to a real case: house price
prediction. We punt matters relating to the computational performance, scalability, and efficiency
of our models to subsequent chapters.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```

## Resources and Further Reading

The references below develop the multilayer perceptron and the core mechanics of
training deep networks that this chapter introduces: hidden layers and activation
functions, the universal approximation theorem, backpropagation, initialization
and numerical stability, modern generalization, and dropout. The optimizers,
normalization layers, and architectures these topics point toward are developed in
their own later parts of the book; the sources here go deeper on the foundations.
All are freely accessible online except where noted.

**Books**

- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) — free HTML; Chapter 6 (Deep Feedforward Networks) is the canonical reference for hidden units, activation functions, the universal approximation theorem, and backpropagation as reverse-mode differentiation; Chapter 7 covers dropout and regularization.
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/) — free online; Chapter 2 derives the four equations of backpropagation from first principles with unusual clarity, the best companion to this chapter's backprop section.
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
- [Understanding the Difficulty of Training Deep Feedforward Networks — Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) — free PDF; derives Xavier/Glorot initialization from variance preservation and explains sigmoid-saturation vanishing gradients, the backbone of this chapter's initialization section.
- [Delving Deep into Rectifiers — He, Zhang, Ren & Sun (2015)](https://arxiv.org/abs/1502.01852) — free; derives He/Kaiming initialization, the current default for ReLU networks.
- [Rectified Linear Units Improve Restricted Boltzmann Machines — Nair & Hinton (2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) — free; the systematic case for ReLU as a hidden-unit nonlinearity.
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting — Srivastava et al. (2014), *JMLR*](https://jmlr.org/papers/v15/srivastava14a.html) — free; the dropout paper, with the ensemble and co-adaptation interpretations this chapter uses.
- [Reconciling Modern Machine-Learning Practice and the Bias–Variance Trade-off — Belkin, Hsu, Ma & Mandal (2019), *PNAS*](https://arxiv.org/abs/1812.11118) — free; introduces the double-descent curve central to this chapter's modern-generalization section.

**Tutorials, notes, and interactive**

- [TensorFlow Playground — Smilkov & Carter](https://playground.tensorflow.org/) — free, zero-install; tune depth, width, activation, and regularization and watch a decision boundary train in real time, the fastest way to build hidden-layer intuition.
- [Backpropagation, Intuitions — Stanford CS231n](https://cs231n.github.io/optimization-2/) — free; the gate/computational-graph view with worked numerical examples, the perfect concrete complement to this chapter's backprop derivation.
- [Double Descent — MLU-Explain](https://mlu-explain.github.io/double-descent/) — free, interactive; animates the interpolation threshold and second descent, the best visual companion to the modern-generalization section.
- [KAN: Kolmogorov–Arnold Networks — Liu et al. (2024)](https://arxiv.org/abs/2404.19756) — free; a recent alternative that places learnable activations on edges rather than fixed activations on nodes, useful context for where the MLP sits among modern parameterizations.

