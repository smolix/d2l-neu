# Builders' Guide
:label:`chap_computation`

Alongside giant datasets and powerful hardware, great software tools have
played an indispensable role in the rapid progress of deep learning. Deep
learning libraries let us recycle standard components while retaining the
ability to modify anything, and over time their abstractions have grown
coarser: from individual neurons, to layers, to the multi-layer *blocks* from
which today's models are assembled.

So far we called upon these libraries without asking how they work. We built
models from layers, initialized and trained them, and treated everything
between `net(X)` and the loss as machinery. This chapter opens the machinery.
In 2016 that meant learning to build, initialize, and save a small model
trained from scratch in 32-bit arithmetic on one device. Those skills remain,
and this chapter teaches them, but the working assumptions around them have
changed: a model is now assembled from a configuration object, measured in
gigabytes, run in reduced precision, checkpointed together with its optimizer
state, and as often as not initialized from someone else's weights rather than
from a random number generator.

Accordingly, we proceed in eight steps. We start with how models are built
from modules and configs (:numref:`sec_model_construction`), what a model's
state is and what it costs in memory (:numref:`sec_parameters`), how that
state is initialized (:numref:`sec_init_param`), and how to write layers the
library does not provide (:numref:`sec_custom_layer`). We then turn to the
numeric formats models compute in (:numref:`sec_numerics`), how state is
saved, restored, and adopted from pretrained models
(:numref:`sec_read_write`), how tensors and models live on GPUs and in GPU
memory (:numref:`sec_use_gpu`), and finally how to make runs repeatable and
inspect a model from the outside (:numref:`sec_repro`). The chapter
introduces no new models or datasets; the advanced modeling chapters that
follow rely on these techniques throughout.

```toc
:maxdepth: 2

model-construction
parameters-state-memory
init
custom-layers
numerics
saving-loading
gpus-devices-memory
reproducibility-inspection
```
