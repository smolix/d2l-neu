# Multilayer Perceptron
:label:`chap_perceptrons`

Softmax regression scores each class by a weighted sum of the pixels, so a
bright pixel adds the same evidence for a class whatever its neighbors show.
Yet whether that pixel belongs to a sleeve, a strap, or the background
depends on the pixels around it. The smallest problem of this kind, the
exclusive or of two bits, defeats every linear classifier. A *multilayer
perceptron* (MLP) computes new features before it classifies: a hidden layer
applies an affine map followed by a fixed nonlinear function, and softmax
regression classifies the result. Stacking such layers gives the first deep
networks of this book.

The loss, the optimizer, and the training loop of
:numref:`chap_classification` carry over unchanged, but hidden layers raise
three new problems. Every training step needs the derivative of the loss with respect
to hundreds of thousands of parameters. A deep network can fail to train at
all when its initial weights have the wrong scale. And a network with more
parameters than training examples could memorize its training data, yet
usually does not.

:numref:`sec_mlp` asks which functions networks with hidden layers can
represent, and why they need a nonlinearity.
:numref:`sec_mlp-implementation` asks what an MLP costs and gains over
softmax regression on Fashion-MNIST. :numref:`sec_backprop` asks how one
backward pass computes the gradient of every parameter, and what it costs in
memory. :numref:`sec_numerical_stability` asks why gradients vanish or
explode in deep networks, and how the scale of the initial weights prevents
it. :numref:`sec_generalization_deep` asks why a network that can memorize
its training data generalizes anyway. :numref:`sec_dropout` asks whether
silencing random hidden units during training helps.
:numref:`sec_kaggle_house` asks how these tools carry over to a real
prediction task, the sale prices of houses on Kaggle.

The chapter treats fully connected networks. Layers that exploit the
structure of images and sequences, and the optimizers and normalization
layers that train very deep networks, follow in later chapters.

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
- Matrix calculus and automatic differentiation in this book's mathematics appendix (:numref:`sec_mdl-matrix-calculus-autodiff`) — reverse-mode differentiation with proofs, the background of :numref:`sec_backprop`.
- Concentration and generalization in the mathematics appendix (:numref:`sec_mdl-concentration-generalization`) — uniform convergence, the minimum-norm mechanism behind double descent, and a numerical double-descent example with random features (:numref:`sec_generalization_deep`).

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
- measure what dropout changes; select a model by cross-validation.

**The idea:** a hidden layer computes features, an affine map followed by a
fixed nonlinearity, and softmax regression classifies them. **Running
example:** the Fashion-MNIST scoreboard on the next slide.
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
| Multilayer Perceptrons | which functions can hidden layers represent? | hidden layers, activation functions, universal approximation |
| Implementation of Multilayer Perceptrons | what does an MLP cost and gain? | an MLP trained on Fashion-MNIST |
| Backpropagation | how does one backward pass compute every gradient? | how the gradients are computed, and at what memory cost |
| Numerical Stability and Initialization | why do gradients vanish or explode? | vanishing gradients, Xavier and He initialization |
| Generalization in Deep Learning | why does a network that can memorize generalize? | memorization, double descent, implicit bias, early stopping |
| Dropout | does silencing random units help? | random masks, robustness to missing units |
| Predicting House Prices on Kaggle | how do the pieces work on real data? | the workflow on real data |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Multilayer Perceptrons | XOR computed by two hidden units; at $z = 4$ the sigmoid's slope is 0.018, the ReLU's 1 |
| Implementation of Multilayer Perceptrons | 203,530 parameters; about 87% against about 84% for softmax regression |
| Backpropagation | a minibatch of 256 stores 1.1 MB of activations next to 0.8 MB of weights |
| Numerical Stability and Initialization | ten layers, weights of standard deviation 0.01: the gradient shrinks by more than $10^8$ |
| Generalization in Deep Learning | 100% training accuracy on random labels, chance on test images |
| Dropout | with its masks on, the dropout network loses 1–2 points, the plain one 8–12 |
| Predicting House Prices on Kaggle | RMSE of log prices: about 0.15 for a linear model, about 0.14 for an MLP |
:::
