# Basics

The introduction described programs that learn from examples instead of
following rules written by hand. Suppose we want such a program for a small
photograph of a garment: it should say whether the photograph shows a
shirt, a sneaker, or a bag. Nobody knows how to write rules for this. We
can, however, collect tens of thousands of labeled photographs. How do we
turn those examples into a classifier we can trust on photographs it has
never seen?

Our answer is a recipe with three ingredients: a model with parameters, a
loss that measures the model's errors, and an algorithm that reduces the
loss by changing the parameters. The recipe itself is two centuries old,
and astronomers used it first (:numref:`chap_regression` tells the
story). What has changed is the model. A deep network composes many
simple functions, and automatic differentiation computes the derivatives of
the loss with respect to all of their parameters, so that nobody has to
derive them by hand. Nearly every model in this book, including the largest
language models, is trained this way.

The first three chapters teach the recipe on models small enough to
understand completely. :numref:`chap_preliminaries` collects the tools:
tensors, derivatives, and enough probability to reason about uncertainty.
:numref:`chap_regression` fits a linear model to numbers and asks whether
the fit will hold for new data, and :numref:`chap_classification` does the
same for categories, which is where the garments return. The next two
chapters make the model deep. :numref:`chap_perceptrons` adds hidden layers
and deals with the troubles they bring, and :numref:`chap_computation`
turns to the software that assembles such a model, saves it, and runs it on
a GPU. The last three chapters build what we know about the data into the
model. Pixels have neighbors, which leads to convolutional networks
(:numref:`chap_cnn`) and to the ideas that let them grow to more than a
hundred layers (:numref:`chap_modern_cnn`). Words come in order, which
leads to models of text and other sequences (:numref:`chap_rnn`).

The Advanced part takes the engineering habits of this part
(initialization, numerical stability, checkpointing, GPU memory) for
granted. This part uses minibatch stochastic gradient descent, momentum, and
learning-rate schedules as tools; :numref:`chap_optimization` studies why
they work.
