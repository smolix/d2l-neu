# Multilayer Perceptron
:label:`chap_perceptrons`

In this chapter, we introduce your first truly *deep* neural networks. In
:numref:`chap_classification`, softmax regression simply assigned a fixed
weight to each of the 784 pixels of a garment image and summed the
evidence—meaning every pixel contributed the exact same vote toward a
class regardless of what surrounded it. Yet a bright pixel might belong to
a sleeve, a handbag strap, or a collar, and only its neighbors can reveal
which. How do we build models where the meaning of one input depends on the
others, and how do we train them effectively?

The failure of linear models shows up long before sleeves and collars: no
straight line—no thresholded weighted sum—can compute even the simple
exclusive-OR (XOR) of two bits. Marvin Minsky and Seymour Papert famously
highlighted this barrier in 1969 for Frank Rosenblatt's 1958 *perceptron*,
a single-layer linear classifier. Hand-crafting derived features such as
the product of the two bits sidesteps the barrier, yet for decades no
widely practical algorithm could *learn* useful features automatically.
*Multilayer perceptrons* (MLPs)—the simplest deep networks—solve this by
stacking fully connected *hidden layers*, each pairing an affine
transformation with a fixed nonlinear activation function. In 1986,
Rumelhart, Hinton, and Williams popularized *backpropagation* to train all
layers jointly by gradient descent, allowing hidden layers to discover
useful internal representations directly from data
:cite:`Rumelhart.Hinton.Williams.ea.1988`.

Happily, the cross-entropy loss, minibatch SGD optimizer, and training loop
of softmax regression carry straight over unchanged. Adding hidden layers,
however, confronts us with three fundamentally new challenges: computing
gradients efficiently across hundreds of thousands of parameters, choosing
parameter initializations carefully so signals neither vanish nor explode
across layers, and understanding why heavily overparameterized networks—capable
of memorizing their entire training set—still generalize so well in
practice. The first two challenges have clean, well-understood solutions;
the third remains one of the deepest open puzzles in modern machine
learning.

We start by building the architecture: :numref:`sec_mlp` explains why
nonlinearity is indispensable and explores the expressive power of hidden
layers, while :numref:`sec_mlp-implementation` implements and trains an MLP
on Fashion-MNIST to measure exactly what it gains over softmax regression
and at what computational cost. Next, we tackle our three deep-learning
challenges in turn—backpropagation (:numref:`sec_backprop`), numerical
stability and weight initialization (:numref:`sec_numerical_stability`), and
generalization in deep networks (:numref:`sec_generalization_deep`). We
then introduce *dropout* (:numref:`sec_dropout`), a regularization method
tailored to hidden units, testing it on a noisy-label memorization
experiment. Finally, :numref:`sec_kaggle_house` puts the entire toolkit to
work end to end on a real-world tabular dataset: predicting house prices in
a Kaggle competition.

This chapter builds on the foundations of overfitting and regularization
from :numref:`sec_generalization_basics` and :numref:`sec_weight_decay`,
focusing exclusively on fully connected layers. Specialized architectures
that exploit spatial or sequential structure—together with advanced
optimizers, normalization layers, and scalability techniques—await in the
chapters ahead.

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

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. All are freely accessible online except where noted.

**Books and notes**

- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) — free HTML; Chapter 6 covers hidden units, activation functions, universal approximation, and backpropagation (:numref:`sec_mlp`, :numref:`sec_backprop`), and Chapter 7 covers dropout (:numref:`sec_dropout`).
- [Neural Networks and Deep Learning, Chapter 2 — Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap2.html) — free online; derives backpropagation step by step from the chain rule (:numref:`sec_backprop`).
- [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) — free PDF with notebooks; its chapters on shallow and deep networks, gradients and initialization, and generalization accompany :numref:`sec_mlp`, :numref:`sec_numerical_stability`, and :numref:`sec_generalization_deep`.
- Matrix calculus and automatic differentiation in this book's mathematics appendix (:numref:`sec_mdl-matrix-calculus` and :numref:`sec_mdl-matrix-calculus-autodiff`) — reverse-mode differentiation with proofs, the background of :numref:`sec_backprop`.
- Concentration and generalization in the mathematics appendix (:numref:`sec_mdl-generalization-bounds`) — uniform convergence, the minimum-norm mechanism behind double descent, and a numerical double-descent example with random features (:numref:`sec_generalization_deep`).

**Courses and tutorials**

- [CS231n notes: Neural Networks, Parts 1 and 2 — Stanford](https://cs231n.github.io/neural-networks-1/) — free; activation functions, weight initialization, and dropout in practice (:numref:`sec_mlp`, :numref:`sec_numerical_stability`, :numref:`sec_dropout`); the companion note [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) works through gradients on small graphs (:numref:`sec_backprop`).
- [micrograd — Andrej Karpathy](https://github.com/karpathy/micrograd) — free; a scalar automatic-differentiation engine in about 100 lines, with a [video walkthrough](https://www.youtube.com/watch?v=VMj-3S1tku0) that builds it from scratch (:numref:`sec_backprop`).
- [TensorFlow Playground — Smilkov & Carter](https://playground.tensorflow.org/) — free, in the browser; vary depth, width, and activation and watch a decision boundary form (:numref:`sec_mlp`).
- [Double Descent — MLU-Explain](https://mlu-explain.github.io/double-descent/) — free, interactive; animates the second descent past the interpolation threshold (:numref:`sec_generalization_deep`).

**Papers**

- [Approximation by Superpositions of a Sigmoidal Function — Cybenko (1989)](https://doi.org/10.1007/BF02551274) — the universal approximation theorem for one hidden layer of sigmoid units (:numref:`sec_mlp`; paywalled).
- [Learning Representations by Back-propagating Errors — Rumelhart, Hinton & Williams (1986)](https://www.nature.com/articles/323533a0) — the paper that made backpropagation and learned hidden representations widely known (:numref:`sec_backprop`; paywalled).
- [Understanding the Difficulty of Training Deep Feedforward Networks — Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) and [Delving Deep into Rectifiers — He, Zhang, Ren & Sun (2015)](https://arxiv.org/abs/1502.01852) — derive Xavier and He initialization from keeping the variance of signals constant across layers (:numref:`sec_numerical_stability`).
- [Understanding Deep Learning Requires Rethinking Generalization — Zhang, Bengio, Hardt, Recht & Vinyals (2017)](https://arxiv.org/abs/1611.03530) — networks that fit random labels, the experiment that opens :numref:`sec_generalization_deep`.
- [Reconciling Modern Machine-Learning Practice and the Bias–Variance Trade-off — Belkin, Hsu, Ma & Mandal (2019)](https://arxiv.org/abs/1812.11118) — the double-descent curve (:numref:`sec_generalization_deep`).
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting — Srivastava et al. (2014)](https://jmlr.org/papers/v15/srivastava14a.html) — the dropout paper (:numref:`sec_dropout`).
- [Ames, Iowa: Alternative to the Boston Housing Data — De Cock (2011)](https://doi.org/10.1080/10691898.2011.11889627) — the dataset behind the house-price competition, and how it was assembled (:numref:`sec_kaggle_house`).

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4]{.kicker}

From linear models to networks with hidden layers<br>
**hidden layers · backpropagation · initialization · generalization · dropout · Kaggle**
:::
:::

::: {.slide title="A Linear Model Cannot Combine Its Inputs"}
[The problem]{.kicker}

- **XOR of two bits:** no line separates the two classes, so every linear
  classifier gets at least one of the four points wrong.
- **Fashion-MNIST:** softmax regression levels off at about 84%; one hidden
  layer of 256 ReLUs reaches about 87% after 30 epochs.
- **The price:** 203,530 parameters, 26 times as many, about 3.4 per
  training image. Such a network could memorize its training set, yet it
  does not.

::: {.d2l-note}
Hidden layers make the model expressive; this chapter shows how to train
them and why they still generalize.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- compute XOR with two hidden units;
- implement an MLP from scratch and with framework layers;
- derive backpropagation and check it against automatic differentiation;
- choose the initial scale (Xavier, He) so that ten layers train;
- show that capacity cannot explain generalization, and what gradient
  descent prefers instead;
- measure what dropout changes when labels are noisy; select a model by
  cross-validation.

**The idea:** a hidden layer computes features, an affine map followed by a
fixed nonlinearity, and softmax regression classifies them. **Running
example:** the Fashion-MNIST models on the next slide.
:::

::: {.slide title="Running Example: 26 Times the Parameters Add Three Points on Fashion-MNIST"}
[Running example]{.kicker}

| model | parameters | multiply-adds per image | validation accuracy |
|---|---:|---:|---:|
| softmax regression | 7,850 | 7,840 | about 84% (30 epochs) |
| MLP, 256 hidden units | 203,530 | 203,264 | about 87% (30 epochs) |
| MLP, 2 × 256 hidden units, dropout | 269,322 | 268,800 | about 87% (20 epochs) |

- Seconds per epoch on a GPU: about the same for softmax regression and the
  MLP, since reading the images and the fixed cost of each step dominate.
- The decks of this chapter measure their models against these rows.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question | adds |
|---|---|---|
| Multilayer Perceptrons | which functions can hidden layers represent? | hidden layers, universal approximation, activation functions |
| Implementation | what does an MLP cost and gain? | an MLP trained on Fashion-MNIST |
| Backpropagation | how does one backward pass compute every gradient? | how the gradients are computed, and at what memory cost |
| Numerical Stability and Initialization | why do gradients vanish or explode? | vanishing gradients, Xavier and He initialization |
| Generalization in Deep Learning | why does a network that can memorize generalize? | memorization, double descent, implicit bias, early stopping |
| Dropout | what does silencing random units change? | random masks, slower memorization of noisy labels |
| Predicting House Prices on Kaggle | how do the pieces work on real data? | the workflow on real data |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Multilayer Perceptrons | XOR computed by two hidden units; at $z = 4$ the sigmoid's slope is 0.018, the ReLU's 1 |
| Implementation | 203,530 parameters; about 87% against about 84% for softmax regression |
| Backpropagation | a minibatch of 256 stores 1.1 MB of activations next to 0.8 MB of weights |
| Numerical Stability and Initialization | ten layers, weights of standard deviation 0.01: the gradient shrinks by more than $10^8$, and the loss never leaves $\log 10$ |
| Generalization in Deep Learning | 100% training accuracy on random labels, chance on test images |
| Dropout | 1,000 images, 40% of the labels redrawn: dropout ends three to five points higher |
| Predicting House Prices on Kaggle | RMSE of log prices: about 0.15 for a linear model, about 0.135 for an MLP |
:::
