# Gaussian Paths: Diffusion and Flow Matching
:label:`sec_diffusion-gaussian-paths`

(to be written)

```{.python .input #gaussian-paths-gaussian-paths-diffusion-and-flow-matching}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #gaussian-paths-gaussian-paths-diffusion-and-flow-matching}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import math
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## One Path from Data to Noise

(to be written)

### Signal, Noise and the Log-SNR Clock

(to be written)

### The Chain View

(to be written)

### Why Many Levels

(to be written)

## One Regression

(to be written)

### Conditional Targets and Marginal Fields

(to be written)

### Four Targets, One Oracle

(to be written)

### The Loss Floor

(to be written)

## Sampling by Integration

(to be written)

### The Probability-Flow ODE and Euler's Method

(to be written)

### The Reverse SDE and the Ancestral Chain

(to be written)

## Experiments on the Running Example

(to be written)

### The Learned Predictor against the Optimal One

(to be written)

### Samples from Both Samplers

(to be written)

### The Dictionary Checked

(to be written)

## Summary

(to be written)

## Exercises

(to be written)
