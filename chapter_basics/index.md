# Basics

The introduction described programs that learn from examples instead of
following rules written by hand. Suppose we want such a program for a small
photograph of a garment: it should say whether the photograph shows a
shirt, a sneaker, or a bag. Nobody knows how to write rules for this, but
we can collect tens of thousands of labeled photographs. How do we turn
those examples into a classifier we can trust on photographs it has never
seen?

The core of the answer is older than computers. Early in the nineteenth
century Legendre and Gauss computed the orbits of comets and asteroids by
least squares, choosing the parameters of a formula to make the sum of its
squared errors as small as possible. Deep learning keeps this recipe: a
model with parameters, a loss that measures its errors, and an algorithm
that reduces the loss by changing the parameters. What has changed is the
model. A deep network composes many simple functions, and automatic
differentiation computes the derivatives of the loss with respect to all of
their parameters. Nearly every model in this book, including the largest
language models, is trained this way. This part teaches the recipe on small
models, then builds knowledge about the data, such as the layout of pixels
or the order of words, into the model.

:numref:`chap_preliminaries` asks how to store data as tensors, compute and
differentiate with them, and reason about uncertainty.
:numref:`chap_regression` asks how a model, a loss, and an optimizer fit
together, and whether a fitted model will predict new data.
:numref:`chap_classification` asks how to predict a category instead of a
number, and how to judge such predictions. :numref:`chap_perceptrons` asks
what hidden layers add to a linear model, how their gradients are computed,
and why a network with many parameters can still generalize.
:numref:`chap_computation` asks how models are assembled from modules,
initialized, saved, and run on a GPU. :numref:`chap_cnn` asks how to build
the spatial structure of images into a layer, and :numref:`chap_modern_cnn`
asks which ideas let convolutional networks grow to more than a hundred
layers. :numref:`chap_rnn` asks how to model text and other sequences of
varying length whose entries depend on one another.

The Advanced part takes the engineering habits of this part
(initialization, numerical stability, checkpointing, GPU memory) for
granted. This part uses minibatch stochastic gradient descent, momentum, and
learning-rate schedules as tools; :numref:`chap_optimization` studies why
they work.
