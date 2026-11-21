# Multilayer Perceptrons
:label:`chap_perceptrons`

In this chapter, we introduce your first truly *deep* neural networks. In :numref:`chap_classification`, softmax regression assigned a fixed weight to each of the 784 pixels of a Fashion-MNIST garment image and summed the evidence—meaning every pixel contributed the exact same vote toward a class regardless of what surrounded it. Yet a bright pixel might belong to a sleeve, a handbag strap, or a collar, and only its neighbors can reveal which. Even simpler, no straight line—no single linear layer—can compute the exclusive-OR (XOR) of two binary bits, a barrier Marvin Minsky and Seymour Papert famously highlighted in 1969 for Frank Rosenblatt's single-layer *perceptron*.

*Multilayer perceptrons* (MLPs) break through this linear barrier by stacking one or more fully connected *hidden layers*—each pairing an affine transformation with a nonlinear activation function—before the output layer, and training all layers jointly end to end via *backpropagation* :cite:`Rumelhart.Hinton.Williams.ea.1988`. While the cross-entropy loss, minibatch SGD optimizer, and `Trainer` loop from :numref:`chap_classification` carry straight over, adding hidden layers introduces three fundamental deep-learning challenges: computing gradients efficiently across hundreds of thousands of parameters, choosing initial weight scales so signals and gradients neither vanish nor explode across depth, and understanding why heavily overparameterized networks generalize instead of merely memorizing their training data.

This chapter unfolds in seven sections:

- **Multilayer Perceptrons (:numref:`sec_mlp`):** We show why linear models fail on feature interactions like XOR, solve XOR with a two-unit ReLU hidden layer, examine universal approximation and the exponential piece-multiplying power of depth, and compare standard activation functions (ReLU, GELU, sigmoid, and $\tanh$).
- **Implementation (:numref:`sec_mlp-implementation`):** We implement a 256-unit ReLU MLP both from scratch and concisely on Fashion-MNIST, cutting roughly a fifth of softmax regression's classification errors ($\sim 87\%$ vs. $\sim 84\%$ accuracy) at virtually identical GPU epoch time and visualizing the localized garment-part detectors learned by the hidden units.
- **Backpropagation (:numref:`sec_backprop`):** We derive the three matrix-calculus local rules of reverse-mode differentiation through an MLP, verify our hand-coded backward pass against `autograd` to $10^{-14}$ relative error, and tally the exact activation memory and FLOP cost of training versus inference.
- **Numerical Stability and Initialization (:numref:`sec_numerical_stability`):** We analyze how activations and gradients scale across $K$ layers, derive *Xavier* and *He* initialization to prevent vanishing and exploding gradients in a 10-layer MLP, and prove why random initialization is required to break hidden-unit permutation symmetry.
- **Generalization in Deep Learning (:numref:`sec_generalization_deep`):** We demonstrate random-label memorization, *double descent* past the interpolation threshold, the *implicit bias* of gradient descent (minimum-norm least squares and maximum-margin classification), and *early stopping* under label noise.
- **Dropout (:numref:`sec_dropout`):** We introduce inverted dropout masking and rescaling, implement it from scratch and concisely in a two-hidden-layer MLP, and test how it slows the memorization of corrupted labels.
- **Predicting House Prices on Kaggle (:numref:`sec_kaggle_house`):** We put the entire toolkit to work end to end on a real tabular regression competition (Ames Housing), covering leakage-free preprocessing, log-RMSE evaluation, and 5-fold cross-validated model selection.

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
