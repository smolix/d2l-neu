# Discrete Diffusion
:label:`sec_diffusion-discrete`

(to be written)

```{.python .input #discrete-diffusion}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #discrete-diffusion}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import math
import jax
from jax import numpy as jnp
from jax.scipy.special import xlogy
from jax.scipy.stats import norm
from flax import nnx
import optax
```

## Corrupting Tokens

(to be written)

### Transition Matrices

(to be written)

### The Absorbing Kernel

(to be written)

## The Masked Bound

(to be written)

### The Same Bound, a Different Target

(to be written)

### The Posterior of the Absorbing Chain

(to be written)

### A Weighted Cross-Entropy

(to be written)

## Sampling by Unmasking

(to be written)

### Parallel Reveals and the Independence Error

(to be written)

### Diffusion Language Models in Practice

(to be written)

### Beyond Masking

(to be written)

## Experiments

(to be written)

### The Quantized Mixture

(to be written)

### Images as Binary Tokens

(to be written)

## Masked Language Models as One Step

(to be written)

## Summary

(to be written)

## Exercises

(to be written)
