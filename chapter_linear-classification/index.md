# Linear Classification
:label:`chap_classification`

Many predictions are categories rather than numbers: whether an email is
spam, which of ten garments a photograph shows, which disease explains a set
of symptoms. A regression model is the wrong tool for them. Its output is an
unbounded number, so it cannot say that the model is torn between two
classes, and its squared loss treats the class index as a quantity, as if a
pullover lay halfway between a trouser and a dress.

The procedure of :numref:`chap_regression` carries over: a model with
parameters, a loss, minibatch stochastic gradient descent, and evaluation on
held-out data. Three parts change. The model outputs one score per class, the
scores become a probability distribution over the classes, and the loss
measures how much probability the model gives the correct class. Maximum
likelihood, which turned Gaussian noise into the squared loss, now gives the
softmax function and the cross-entropy loss. Evaluation also asks more than
it did for regression. A probability is not yet a decision, accuracy can
mislead when one class is rare, and a test set vouches only for data that
resemble it.

:numref:`sec_softmax` asks how a linear model can output a probability
distribution over classes, and which loss trains it.
:numref:`sec_fashion_mnist` asks which dataset is small enough for quick
experiments yet hard enough to tell models apart. :numref:`sec_classification`
asks how to measure a classifier, and what accuracy hides.
:numref:`sec_softmax_scratch` asks what softmax regression looks like when we
write it ourselves, and where the naive version fails numerically.
:numref:`sec_softmax_concise` asks how a framework's loss avoids that failure,
and whether the trained model's probabilities can be read as probabilities.
:numref:`chap_classification_generalization` asks how many test examples a
trustworthy accuracy estimate needs, what reusing a test set does to it, and
why a classifier fitted to its training data should do better on new data than
one that memorizes them. :numref:`sec_environment-and-distribution-shift` asks
what happens when the data at deployment differ from the training data, and
when we can correct for it.

The chapter treats problems with exactly one class per example, and the
classes have no order. The next chapter keeps the loss and replaces the
linear model by a multilayer network.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
classification
softmax-regression-scratch
softmax-regression-concise
generalization-classification
environment-and-distribution-shift
```

## Resources and Further Reading {.unnumbered}

The statistical texts listed with :numref:`chap_regression` cover
classification as well; the resources below add what is specific to this
chapter, and each names the section it accompanies. All are freely
accessible online except where noted.

**Books and notes**

- [CS229 Machine Learning lecture notes — Andrew Ng & Tengyu Ma (Stanford)](https://cs229.stanford.edu/main_notes.pdf) — free PDF; derives logistic and softmax regression and the cross-entropy loss from maximum likelihood, the route of :numref:`sec_softmax`.
- Information theory in this book's mathematics appendix (:numref:`chap_mdl-information-theory`) — entropy, cross-entropy, and the KL divergence with proofs, the background of the cross-entropy loss in :numref:`sec_softmax`.
- [Patterns, Predictions, and Actions — Moritz Hardt & Benjamin Recht](https://mlstory.org/) — free online; a modern text on prediction whose chapters on generalization, datasets, and deployment accompany :numref:`chap_classification_generalization` and :numref:`sec_environment-and-distribution-shift`.
- [Understanding Machine Learning: From Theory to Algorithms — Shai Shalev-Shwartz & Shai Ben-David](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) — free PDF; uniform convergence and the VC dimension with full proofs (:numref:`chap_classification_generalization`).
- [Dataset Shift in Machine Learning — Quiñonero-Candela, Sugiyama, Schwaighofer & Lawrence (eds.), MIT Press](https://mitpress.mit.edu/9780262545877/dataset-shift-in-machine-learning/) — the standard reference on covariate, label, and concept shift (:numref:`sec_environment-and-distribution-shift`; an open-access edition is available from MIT Press).
- [Machine Learning in Non-Stationary Environments: Introduction to Covariate Shift Adaptation — Sugiyama & Kawanabe, MIT Press](https://mitpress.mit.edu/9780262017091/machine-learning-in-non-stationary-environments/) — importance weighting for covariate shift in depth (:numref:`sec_environment-and-distribution-shift`; print only).

**Papers**

- [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms — Xiao, Rasul & Vollgraf (2017)](https://arxiv.org/abs/1708.07747) — the dataset of :numref:`sec_fashion_mnist` and of the rest of the chapter.
- [On Calibration of Modern Neural Networks — Guo, Pleiss, Sun & Weinberger (2017)](https://arxiv.org/abs/1706.04599) — when a classifier's probabilities can be read as probabilities, and how temperature scaling repairs them (:numref:`sec_softmax`, :numref:`sec_classification`).
- [WILDS: A Benchmark of in-the-Wild Distribution Shifts — Koh, Sagawa, Marklund et al. (2021)](https://arxiv.org/abs/2012.07421) — ten datasets with real shifts between hospitals, cameras, regions, and time, each with the in-distribution and shifted test accuracy of standard models (:numref:`sec_environment-and-distribution-shift`).
- [Do ImageNet Classifiers Generalize to ImageNet? — Recht, Roelofs, Schmidt & Shankar (2019)](https://arxiv.org/abs/1902.10811) — rebuilds a test set from scratch and finds accuracy drops that a slight distribution shift explains better than test-set reuse (:numref:`chap_classification_generalization`, :numref:`sec_environment-and-distribution-shift`).

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

- Regress on the codes 0–9 of ten garments and a prediction of 5.5 lands
  between coat and sandal: the number means nothing.
- **Accuracy can mislead:** for a disease 1% of patients have, the rule
  "always healthy" is right 99% of the time and finds no one.
- **A test set vouches only for data like it:** a classifier right on about
  85% of its source data is right on only 40–65% of shifted data.

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
| The Image Classification Dataset | which data tell models apart quickly? | Fashion-MNIST in minibatches |
| Measuring Classifiers | what does accuracy hide? | accuracy, costs, calibration, the `Classifier` class |
| Implementation from Scratch | what does each part look like by hand? | softmax regression by hand |
| Concise Implementation | how does a framework avoid overflow? | framework layers, a stable loss |
| Generalization in Classification | how far can we trust a test score? | error bars, test-set reuse, VC dimension |
| Environment and Distribution Shift | what if deployment data differ? | correcting for shifted data |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Softmax Regression | a 2×2 image; logits (1.0, 2.2, 0.3) for cat, chicken, dog give probabilities (0.21, 0.69, 0.10) |
| The Image Classification Dataset | Fashion-MNIST: 60,000 training and 10,000 validation images, 28×28, 10 classes |
| Measuring Classifiers | a disease 1% have: "always healthy" 99% accurate, a screening test 95% |
| Implementation from Scratch | 7,850 parameters; about 83% after ten epochs; shirts are confused most |
| Concise Implementation | the same model with a fused loss; confidence matches accuracy within about 2 points |
| Generalization in Classification | error 0.15 on 10,000 images: ±0.7 points; the best of 10,000 coin flips scores 0.56 |
| Environment and Distribution Shift | source data shifted by 2 at deployment: 40–65% unweighted, about 95% reweighted |
:::
