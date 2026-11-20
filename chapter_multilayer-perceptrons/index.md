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
