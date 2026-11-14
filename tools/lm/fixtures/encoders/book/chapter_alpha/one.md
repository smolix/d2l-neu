# Defining Alpha
:label:`sec_alpha_one`

An *alpha map* sends $x$ to $\alpha x$ :cite:`Smith.2020,Jones.2021`.
As :citet:`Smith.2020` show, the map is linear (:eqref:`eq_alpha`).

<!-- A comment
that spans two lines, with ```python inside it -->

$$
f(x) = \alpha x,
$$
:eqlabel:`eq_alpha`

where $\alpha \in \mathbb{R}$ is the scale. The expectation $𝔼[f(x)]$ is
$\alpha\, 𝔼[x]$; in Greek, αβγ, and in Chinese, 深度学习.

```{.python .input #one-imports}
%%tab pytorch
import torch
from d2l import torch as d2l
```

```{.python .input #one-imports-jax}
%%tab jax
import jax
```

## Computing the Map
:label:`subsec_alpha_compute`

:begin_tab:`pytorch`
In PyTorch the map is one multiplication.
:end_tab:

:begin_tab:`jax`
In JAX the map is one multiplication of arrays.
:end_tab:

```{.python .input #one-alpha-map}
%%tab pytorch
#@save
def alpha_map(x, alpha=2.0):
    """Scale x by alpha."""
    return alpha * x

print(alpha_map(torch.ones(3)))
```

```{.python .input #one-alpha-map-jax}
%%tab jax
def alpha_map(x, alpha=2.0):
    return alpha * x
```

::: {.d2l-note}
A fenced div keeps its content and loses its fence lines.
:::

![The alpha map scales every input. A second sentence.](../img/alpha.svg)
:label:`fig_alpha`
:width:`300px`

:Alpha values and their names.
:label:`tab_alpha`

| alpha | name |
|---|---|
| 1 | identity |
| 2 | double |

Table :numref:`tab_alpha` and :numref:`fig_alpha` summarize; see also
:numref:`sec_missing` and exercise :exref:`ex_alpha`, or :exref:`ex_alpha`.

## Exercises

1. Prove that the alpha map is linear.
:exlabel:`ex_alpha`

[Discussions](https://discuss.d2l.ai/t/1)

<!-- slides -->

::: {.slide title="Alpha"}
Slides are cut.
:::
