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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §3]{.kicker}

Predicting a category, and knowing when to trust the prediction<br>
**softmax · cross-entropy · Fashion-MNIST · metrics · test sets · distribution shift**
:::
:::

::: {.slide title="A Category Is Not a Number"}
[The problem]{.kicker}

- Regress on the codes 1–10 of ten garments and a prediction of 5.5 lands
  between coat and sandal: the number means nothing.
- **Accuracy can mislead:** for a disease 1% of patients have, the rule
  "always healthy" is right 99% of the time and finds no one.
- **A test set vouches only for data like it:** a classifier right on about
  85% of its source data is right on only about half of shifted data, less
  than guessing the most common class.

::: {.d2l-note}
A classifier must output probabilities, be measured by what its decisions
cost, and be checked against the data it will meet.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- derive the softmax and the cross-entropy loss from maximum likelihood;
- train softmax regression on Fashion-MNIST, by hand and with a framework,
  to about 83% validation accuracy;
- measure a classifier by precision, recall, costs, and calibration;
- size a test set for a given error bar (±0.7 points from 10,000 images) and
  explain why reusing it inflates scores;
- detect covariate and label shift, and correct for them by reweighting.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question | adds |
|---|---|---|
| Softmax Regression | how does a linear model output class probabilities? | probabilities over classes, cross-entropy |
| Image Classification Data | which data tell models apart quickly? | Fashion-MNIST in minibatches |
| Softmax Regression from Scratch | what does each part look like by hand? | softmax regression by hand, the `Classifier` class |
| Concise Implementation | how does a framework avoid overflow? | framework layers, a stable loss |
| Measuring Classifiers | what does accuracy hide? | precision, recall, costs, calibration |
| Generalization in Classification | how far can we trust a test score? | error bars, test-set reuse, VC dimension |
| Environment and Distribution Shift | what if deployment data differ? | correcting for shifted data |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Softmax Regression | a 2×2 image; logits (1.0, 2.2, 0.3) for cat, chicken, dog give probabilities (0.21, 0.69, 0.10) |
| Image Classification Data | Fashion-MNIST: 60,000 training and 10,000 validation images, 28×28, 10 classes |
| Softmax Regression from Scratch | 7,850 parameters; about 83% after ten epochs, 84% on the training images; shirts are confused most; the weights are ten templates |
| Concise Implementation | the same model with a fused loss, finite on logits (1000, 0, 0) |
| Measuring Classifiers | a disease 1% have: "always healthy" 99% accurate, a screening test 95%; our model's confidence matches its accuracy within about 2 points |
| Generalization in Classification | error 0.15 on 10,000 images: ±0.7 points; the best of 10,000 coin flips scores 0.56 |
| Environment and Distribution Shift | source data shifted by 2 at deployment: about half right unweighted, about 95% reweighted; shirts from 10% to 50% of the images, estimated without a label |
:::
