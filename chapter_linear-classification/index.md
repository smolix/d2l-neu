# Linear Classification
:label:`chap_classification`

In :numref:`chap_regression`, our goal was to predict a continuous quantity such as the sale price of a house. Many real-world tasks, however, ask for a discrete category instead: an online catalog photo depicts a shirt, a coat, or a sandal—never "5.5" of anything. Simply numbering the categories 1 through 10 and running linear regression fails badly, because it imposes an arbitrary ordering as though a pullover lay halfway between a pair of trousers and a dress, while leaving the model unable to express calibrated uncertainty ("most likely a shirt, possibly a pullover").

Happily, almost all of our regression training machinery—minibatch data loaders, vectorized forward passes, automatic differentiation, and minibatch SGD—carries straight over to *classification*. Only three ingredients change: we represent unordered categories via *one-hot vectors*, compute one linear score (*logit*) per category and normalize them via the *softmax* function into a probability distribution, and replace squared error with the *cross-entropy loss* (derived from maximum likelihood under a categorical label model, just as squared error arose from Gaussian noise). Virtually every neural classifier in this book—right up to today's largest language models—terminates in this exact softmax layer and trains under this exact loss.

Where classification departs sharply from regression is in *evaluation* and *deployment*. A predicted probability is not yet an action; raw accuracy can mislead whenever one class is rare or different mistakes carry asymmetric costs; a finite test set carries statistical error bars that erode under repeated reuse; and real deployment environments often shift away from the training distribution. This chapter unfolds in six sections:

- **Softmax Regression and Cross-Entropy (:numref:`sec_softmax`):** We formulate one-hot encodings, linear logits, the softmax transformation, and the cross-entropy loss, deriving its clean residual gradient $\hat{\mathbf{y}} - \mathbf{y}$ and connecting it to Shannon's information theory and temperature scaling.
- **Image Classification Data (:numref:`sec_fashion_mnist`):** We introduce the Fashion-MNIST benchmark ($60{,}000$ training and $10{,}000$ validation images across 10 garment categories) and build a reusable `FashionMNIST` data module.
- **Softmax Regression Implementation (:numref:`sec_softmax_scratch`):** We implement softmax regression both from scratch and concisely (:numref:`sec_softmax_concise`), introducing the `Classifier` base class, diagnosing misclassifications via the confusion matrix and learned $28 \times 28$ weight templates, and resolving floating-point overflow and underflow via the fused log-sum-exp cross-entropy loss.
- **Measuring Classifiers (:numref:`sec_classification`):** We examine class imbalance, precision, recall, cost-sensitive decision thresholds, reliability diagrams, expected calibration error (ECE), and post-hoc temperature scaling.
- **Generalization in Classification (:numref:`chap_classification_generalization`):** We quantify test-set confidence intervals (normal vs. Hoeffding), selection bias and adaptive overfitting from test-set reuse, and uniform convergence via the Vapnik--Chervonenkis (VC) dimension.
- **Environment and Distribution Shift (:numref:`sec_environment-and-distribution-shift`):** We distinguish covariate, label, and concept shift, implement importance-weighted domain adaptation and confusion-matrix label-shift correction, and examine feedback loops in deployed decision systems.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
softmax-regression-scratch
classification
generalization-classification
environment-and-distribution-shift
```
