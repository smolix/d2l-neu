# Linear Regression in Neural Networks
:label:`chap_regression`

Linear models provide the simplest setting in which to study the complete
learning procedure: represent a model, define a loss, organize data, optimize
parameters, and evaluate predictions on unseen examples. They are also useful
statistical methods in their own right and provide standard baselines for more
complex models.

This chapter develops linear regression as a one-layer neural network. We
derive the model and its loss, implement its components from first principles,
and then replace those components with framework abstractions. The final
sections introduce generalization and weight decay. The next chapter applies
the same framework to classification.

```toc
:maxdepth: 2

linear-regression
synthetic-regression-data
oo-design
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```

## Resources and Further Reading {.unnumbered}

The references below develop the linear-model family this chapter introduces: the statistical foundations shared with classification (these classical texts are listed here once, and the next chapter builds on them), then material specific to regression, least squares, generalization, and the weight decay we use as $L_2$ regularization. All are freely accessible online except where noted.

**Books**

- [An Introduction to Statistical Learning (ISL) — James, Witten, Hastie & Tibshirani](https://www.statlearning.com/) — free PDF (R and Python editions); the gentlest rigorous treatment of regression, classification, and regularization, with worked labs.
- [The Elements of Statistical Learning (ESL) — Hastie, Tibshirani & Friedman](https://hastie.su.domains/ElemStatLearn/) — free PDF; the graduate companion to ISL, with deep coverage of shrinkage, the bias–variance tradeoff, and model selection.
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
