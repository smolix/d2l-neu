# Denoising and the Score
:label:`sec_diffusion-denoising`

(to be written)

```{.python .input #denoising-denoising-score-matching}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #denoising-denoising-score-matching}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## The Running Example

(to be written)

## A Denoiser Is a Posterior Mean

(to be written)

### Noise and the Regression Target

(to be written)

### Tweedie's Formula

(to be written)

### The Normalizer Never Appears

(to be written)

## What One Noise Level Can and Cannot Do

(to be written)

### The Loss Floor and the Noise Level

(to be written)

### Blindness to Mode Weights

(to be written)

### Sampling with a Score

(to be written)

## Denoising Images

(to be written)

## Summary

(to be written)

## Exercises

(to be written)
