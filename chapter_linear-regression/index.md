# Linear Regression
:label:`chap_regression`

*Regression* is the workhorse of quantitative prediction whenever the target is a meaningful number rather than a discrete category: estimating a home's sale price from its floor area and age, forecasting a patient's hospital stay from vital signs, or predicting next quarter's electricity demand. It is also the ideal proving ground for deep learning. Every essential ingredient of modern neural network training—parameterized forward passes, loss functions derived from maximum likelihood, minibatch stochastic gradient descent, object-oriented training harnesses, generalization diagnostics, and weight-norm regularization—already appears in its cleanest, most transparent form in *linear regression*.

In this chapter, we build linear regression from first principles all the way to a complete, reusable training pipeline:

- **Theory and Statistical Foundations (:numref:`sec_linear_regression`):** We formulate the affine model and squared error loss, contrast the closed-form normal equations (and their geometric interpretation as orthogonal projection) with minibatch stochastic gradient descent (SGD), show why vectorization is essential on modern hardware, and ground squared loss in maximum-likelihood estimation under Gaussian noise.
- **Controlled Synthetic Data (:numref:`sec_synthetic-regression-data`):** We generate a synthetic regression dataset with known ground-truth parameters $(\mathbf{w}^*, b^*)$ and known noise variance, giving us an exact benchmark for verifying our implementations.
- **From-Scratch and Concise Implementations (:numref:`sec_linear_scratch`):** We first implement the model, squared loss, minibatch iterator, and SGD training loop from raw tensors and automatic differentiation—checking that the training loss lands near the theoretical noise floor—and then re-express the exact same model in a few lines using high-level framework layers, built-in losses, and optimizers.
- **Object-Oriented Design (:numref:`sec_oo-design`):** To avoid rewriting training boilerplate in every chapter, we organize our code around three core abstractions—`Module` (models), `DataModule` (data loaders), and `Trainer` (the training loop)—that power the rest of the book.
- **Generalization and Model Selection (:numref:`sec_generalization_basics`):** Using polynomial curve fitting, we distinguish training error from generalization error, unpack the bias--variance trade-off, and use validation splits and $K$-fold cross-validation to select model complexity.
- **Weight Decay (:numref:`sec_weight_decay`):** Finally, we tame high-dimensional overfitting by adding an $\ell_2$ penalty (ridge regression), showing both how it acts as per-step weight decay in SGD and how it selectively shrinks poorly determined eigendirections of the features.

```toc
:maxdepth: 2

linear-regression
synthetic-regression-data
linear-regression-scratch
oo-design
generalization
weight-decay
```
