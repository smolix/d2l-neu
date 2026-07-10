# Main-Text Curriculum Additions

This file records topics proposed for the main textbook but deliberately not
added during the Chapters 2--5 correctness pass. It is a planning document,
not a promise about chapter placement or schedule.

## Numerical linear regression

Expand the brief warning about the normal equations into a worked treatment of
least-squares computation. Cover QR and SVD solvers, the Moore--Penrose
pseudoinverse, rank deficiency, conditioning, and the reason forming
$X^\top X$ squares the condition number. Connect the algorithms to the
mathematics appendix without repeating its proofs.

## Evaluation pipelines and dependent data

Develop preprocessing as fitted state within a validation pipeline. Include
nested cross-validation for extensive hyperparameter search and split designs
for grouped, temporal, spatial, and otherwise dependent observations. The
Kaggle house-price section now avoids fold leakage, but the general abstraction
belongs in a later model-selection treatment.

## Probabilistic classifier evaluation

Add a main-text treatment of decision thresholds, asymmetric error costs,
calibration and reliability diagrams, Brier score, expected calibration error,
and precision--recall analysis. Explain when ROC AUC can obscure performance on
rare positive classes and how evaluation changes when probabilities drive a
decision rather than an argmax benchmark.

## Deep-network generalization

Build a fuller comparison of the main explanatory lenses: margin and norm
bounds, algorithmic stability, compression, PAC--Bayes analyses, implicit bias,
architecture, data augmentation, and scaling behavior. State theorem regimes
separately from empirical observations. The current chapter should remain an
introduction rather than implying that one mechanism has settled the subject.

## Fairness and algorithmic decision-making

Replace the short introductory discussion with a properly scoped treatment, or
move the subject to a chapter where it can receive one. Cover calibration and
group-error criteria, incompatibility results, measurement and label choices,
feedback loops, allocation harms, subgroup evaluation, and the limits of
technical criteria without governance and domain context.

## Numerical formats and modern feedforward blocks

Chapter 6 owns the detailed treatment of floating-point formats, mixed
precision, accumulation, loss scaling, and hardware-dependent numerical
behavior. Later architecture chapters should also develop GELU, SiLU, gated
linear units, and SwiGLU in the context of Transformer feedforward blocks rather
than treating activation choice as a standalone lookup table.
