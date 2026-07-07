# Linear Neural Networks for Regression
:label:`chap_regression`

Before we worry about making our neural networks deep,
it will be helpful to implement some shallow ones,
for which the inputs connect directly to the outputs.
This will prove important for a few reasons.
First, rather than getting distracted by complicated architectures,
we can focus on the basics of neural network training,
including parametrizing the output layer, handling data,
specifying a loss function, and training the model.
Second, this class of shallow networks happens
to comprise the set of linear models,
which subsumes many classical methods of statistical prediction,
including linear and softmax regression.
Understanding these classical tools is pivotal
because they are widely used in many contexts
and we will often need to use them as baselines
when justifying the use of fancier architectures.
This chapter will focus narrowly on linear regression
and the next one will extend our modeling repertoire
by developing linear neural networks for classification.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```

## Resources and Further Reading {.unnumbered}

The references below develop the linear-model family this chapter introduces: the statistical foundations shared with classification (these classical texts are listed here once, and the next chapter builds on them), then material specific to regression, least squares, generalization, and the weight decay we use as $L_2$ regularization. All are freely accessible online except where noted.

**Books**

- [An Introduction to Statistical Learning (ISL) — James, Witten, Hastie & Tibshirani](https://www.statlearning.com/) — free PDF (R and Python editions); the gentlest rigorous treatment of regression, classification, and regularization, with worked labs.
- [The Elements of Statistical Learning (ESL) — Hastie, Tibshirani & Friedman](https://hastie.su.domains/ElemStatLearn/) — free PDF; the comprehensive graduate companion to ISL, with deep coverage of shrinkage, the bias–variance tradeoff, and model selection.
- [Pattern Recognition and Machine Learning — Christopher Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — free PDF from Microsoft Research; a probabilistic, Bayesian-flavoured account of linear models and regularization.
- [Probabilistic Machine Learning: An Introduction — Kevin Murphy](https://probml.github.io/pml-book/book1.html) — free draft PDF; a modern, unifying probabilistic perspective spanning regression through deep learning.
- [Understanding Deep Learning (UDL) — Simon Prince](https://udlbook.github.io/udlbook/) — free PDF; a figure-rich modern treatment whose loss-function chapter generalizes the "match the loss to the noise model" recipe of this chapter into a full menu.
- [Deep Learning: Foundations and Concepts — Bishop & Bishop (2024)](https://www.bishopbook.com/) — free to read online; the successor to PRML, connecting the probabilistic view of linear regression and regularization directly to deep networks.
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares (VMLS) — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/vmls/) — free PDF, slides, and code; an applications-first text built around least squares, the engine of linear regression.

**Courses and video lectures**

- [Supervised Machine Learning: Regression and Classification — Andrew Ng (Stanford / DeepLearning.AI)](https://www.coursera.org/learn/machine-learning) — free to audit; the foundational course covering both settings of this part of the book, starting from linear and logistic regression.
- [Statistical Learning — Hastie & Tibshirani (Stanford Online, edX)](https://www.edx.org/learn/statistics/stanford-university-statistical-learning) — free to audit; lecture series following ISL, including regression, ridge, and the LASSO.

**Foundational papers**

- [Ridge Regression: Biased Estimation for Nonorthogonal Problems — Hoerl & Kennard (1970), *Technometrics*](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) — the origin of $L_2$ regularization, exactly the weight decay introduced in this chapter (paywalled, noted).
- [Regression Shrinkage and Selection via the Lasso — Tibshirani (1996), *JRSS-B*](https://academic.oup.com/jrsssb/article/58/1/267/7027929) — introduces the $L_1$-penalized counterpart, contrasting sparse selection with ridge's shrinkage (paywalled, noted).

