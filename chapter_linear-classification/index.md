# Linear Classification
:label:`chap_classification`

Now that you have worked through the end-to-end mechanics of linear
regression, you are ready to apply those same skills to a broader class of
problems. In :numref:`chap_regression`, our goal was to predict a continuous
quantity such as the price of a house. Many real-world tasks, however, ask
for a discrete category instead: a photo in an online catalog depicts a
shirt, a coat, or a sandal—never "5.5" of anything. Simply numbering the
categories and running regression on those integers fails badly, because it
imposes an arbitrary ordering as though a pullover lay halfway between a
pair of trousers and a dress. Worse still, a single scalar prediction cannot
express uncertainty, leaving a classifier unable to say "most likely a
shirt, possibly a pullover." How should a model predict categories, and when
can we trust what it tells us?

Happily, even as we pivot from regression to *classification*, almost all of
our training plumbing carries straight over: loading minibatches, computing
forward passes, evaluating losses, differentiating with respect to
parameters, and updating weights via minibatch stochastic gradient descent.
Only three pieces change to suit the classification setting: the
representation of the targets, the parameterization of the output layer, and
the choice of loss function. Statisticians worked out the core idea long
before neural networks existed: in 1936, Ronald Fisher distinguished species
of iris flowers from four botanical measurements using a linear weighted sum
of features. *Softmax regression*, the centerpiece of this chapter, extends
that idea to arbitrarily many classes by computing one linear score per
class and converting those scores into a valid probability distribution.
Just as maximum likelihood under Gaussian noise led us to the squared loss
in :numref:`chap_regression`, maximum likelihood for categorical outcomes
leads directly to the *cross-entropy loss*. Mastering this model pays
dividends across the entire book: virtually every neural classifier—right up
to today's largest language models—terminates in the exact same softmax
layer and trains under the exact same loss.

Where classification truly departs from regression is in *evaluation* and
*deployment*. A predicted probability is not yet an action or decision; raw
accuracy can paint a deceptively rosy picture whenever one class is rare;
and a test set only certifies performance on data drawn from the same
distribution.

Accordingly, this chapter unfolds in two halves. The first four sections
build and train the classifier: we derive softmax regression and
cross-entropy from first principles (:numref:`sec_softmax`), introduce the
Fashion-MNIST clothing dataset (:numref:`sec_fashion_mnist`), and implement
the model twice. Our from-scratch implementation
(:numref:`sec_softmax_scratch`) reaches roughly 83% accuracy and—through its
errors and learned weight templates—reveals both the strengths and blind
spots of a linear classifier, while also exposing an instructive numerical
overflow pitfall that the concise framework implementation
(:numref:`sec_softmax_concise`) eliminates. The second half asks how much
that 83% figure is really worth. We dissect what accuracy conceals and check
whether predicted probabilities are well calibrated
(:numref:`sec_classification`), quantify the statistical uncertainty of a
10,000-image test set and the danger of reusing it repeatedly
(:numref:`chap_classification_generalization`), and examine how to detect
and adapt when deployment data shift away from training data
(:numref:`sec_environment-and-distribution-shift`).

Throughout this chapter, every example belongs to a single class chosen from
an unordered set of categories. In the next chapter, we will keep the
softmax output and cross-entropy loss untouched while replacing the linear
feature map with deep multilayer networks.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
softmax-regression-scratch
softmax-regression-concise
classification
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
