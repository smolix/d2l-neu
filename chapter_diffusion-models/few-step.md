# Few-Step Generation
:label:`sec_diffusion-few-step`

(to be written)

```{.python .input #few-step-few-step-generation}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #few-step-few-step-generation}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## Why One Step Fails

(to be written)

### The Average Where the Data Are Not

(to be written)

### Straighter Paths Help but Do Not Finish

(to be written)

## Flow Maps

(to be written)

### The Integral of the Velocity

(to be written)

### Consistency Models

(to be written)

### Average Velocity and the MeanFlow Identity

(to be written)

## Experiments

(to be written)

### On the Running Example

(to be written)

### On Images

(to be written)

## The Other Route: Matching Distributions

(to be written)

## Summary

(to be written)

## Exercises

(to be written)
