# Linear Classification
:label:`chap_classification`

The previous chapter taught a linear model to predict a number, such as the
price of a house. Many questions ask for a category instead. A photo in an
online clothing store shows a shirt, a coat or a sandal, not 5.5 of anything.
We might be tempted to number the categories and regress on the numbers
anyway. The model would then treat a name as a quantity, as if a pullover lay
halfway between a trouser and a dress. Nor can a single number express doubt,
and a classifier that cannot say "probably a shirt, possibly a pullover" is
of limited use. How should a model predict a category, and when can we trust
its prediction?

Statisticians had a good answer long before anyone trained a neural network.
In 1936 Ronald Fisher wanted to tell species of iris apart from four
measurements of their flowers, and he did it with a weighted sum of the
measurements, a linear score. Softmax regression, the model of this chapter,
does much the same. It gives each class its own linear score and turns the
scores into a probability distribution over the classes. Pleasantly little
else has to change. Maximum likelihood, which turned Gaussian noise into the
squared loss in :numref:`chap_regression`, now yields the cross-entropy loss,
and minibatch stochastic gradient descent trains the model as before. The
model is worth learning properly: nearly every classifier in this book, up to
the largest language models, ends in the same softmax and is trained with the
same loss.

Evaluation is where classification really differs from regression. A
probability is not yet a decision. Accuracy can flatter a classifier when one
class is rare. And a test set vouches only for data that resemble it.

The chapter therefore has two halves. The first four sections build the
classifier: we derive the model and its loss (:numref:`sec_softmax`), meet
the clothing images on which we try it (:numref:`sec_fashion_mnist`), and
implement the model twice. The version we write by hand
(:numref:`sec_softmax_scratch`) trains to about 83% accuracy and shows us,
in its mistakes and in its weights, what a linear classifier can and cannot
see; it also fails numerically in an instructive way, which the framework's
loss (:numref:`sec_softmax_concise`) repairs. The last three sections ask
what that 83% is worth. Accuracy hides a great deal, and the predicted
probabilities may or may not mean what they say
(:numref:`sec_classification`). A number measured on 10,000 images has an
error bar, which shrinks when the images are fresh and widens when the same
test set is consulted again and again
(:numref:`chap_classification_generalization`). And the data at deployment
differ from the training data, as to some degree they always do
(:numref:`sec_environment-and-distribution-shift`).

Throughout the chapter, each example belongs to exactly one class, and the
classes have no order. The next chapter keeps the loss and replaces the
linear model by a multilayer network.

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
