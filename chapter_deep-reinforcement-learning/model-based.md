# Learned Models and Planning
:label:`sec_model_based`

<!-- Phase 1 (structure): skeleton only. The opening, the body, the figure, the summary,
the exercises and the deck are written in Phase 3 and 5 (PLAN 15.6-M1 to M5, OUTLINE 15.6). -->

```{.python .input #model-based-learned-models-and-planning}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import gymnasium as gym
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
```

```{.python .input #model-based-learned-models-and-planning}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import gymnasium as gym
import jax
from jax import numpy as jnp
import numpy as np
import optax
```

## A Dynamics Model from Data

### Regression in Place of Counting

### One Step by Hand

### Error Compounds over a Horizon

## Planning in the Model

### Random Shooting

### Warm Starts

### Three Episodes to Swing Up

### What a Real Step Costs

## Search as Policy Improvement

### Improving the Actor's Plan

### Distilling the Search into the Policy

### What the Distilled Actor Learns

### Imagined Transitions as Data

## Summary

## Exercises

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §15.6]{.kicker}

Learned Models and Planning<br>
**a dynamics model from data · planning in the model · search as policy improvement**
:::
:::
