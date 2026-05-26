```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Documentation
:begin_tab:`mxnet`
While we cannot possibly introduce every single MXNet function and class 
(and the information might become outdated quickly), 
the [API documentation](https://mxnet.apache.org/versions/1.8.0/api) 
and additional [tutorials](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) and examples 
provide such documentation. 
This section provides some guidance for how to explore the MXNet API.
:end_tab:

:begin_tab:`pytorch`
While we cannot possibly introduce every single PyTorch function and class 
(and the information might become outdated quickly), 
the [API documentation](https://pytorch.org/docs/stable/index.html) and additional [tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html) and examples 
provide such documentation.
This section provides some guidance for how to explore the PyTorch API.
:end_tab:

:begin_tab:`tensorflow`
While we cannot possibly introduce every single TensorFlow function and class 
(and the information might become outdated quickly), 
the [API documentation](https://www.tensorflow.org/api_docs) and additional [tutorials](https://www.tensorflow.org/tutorials) and examples 
provide such documentation. 
This section provides some guidance for how to explore the TensorFlow API.
:end_tab:

```{.python .input #lookup-api-documentation}
%%tab mxnet
from mxnet import np
```

```{.python .input #lookup-api-documentation}
%%tab pytorch
import torch
```

```{.python .input #lookup-api-documentation}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input #lookup-api-documentation}
%%tab jax
import jax
```

## Functions and Classes in a Module

To know which functions and classes can be called in a module,
we invoke the `dir` function. For instance, we can
query all properties in the module for generating random numbers:

```{.python .input #lookup-api-functions-and-classes-in-a-module  n=1}
%%tab mxnet
print([name for name in dir(np.random) if not name.startswith('_')][:20])
```

```{.python .input #lookup-api-functions-and-classes-in-a-module  n=1}
%%tab pytorch
print([name for name in dir(torch.distributions)
       if not name.startswith('_')][:20])
```

```{.python .input #lookup-api-functions-and-classes-in-a-module  n=1}
%%tab tensorflow
print([name for name in dir(tf.random) if not name.startswith('_')][:20])
```

```{.python .input #lookup-api-functions-and-classes-in-a-module}
%%tab jax
print([name for name in dir(jax.random) if not name.startswith('_')][:20])
```

Generally, we can ignore functions that start and end with `__` (special objects in Python) 
or functions that start with a single `_`(usually internal functions). 
Based on the remaining function or attribute names, 
we might hazard a guess that this module offers 
various methods for generating random numbers, 
including sampling from the uniform distribution (`uniform`), 
normal distribution (`normal`), and multinomial distribution (`multinomial`).

## Specific Functions and Classes

For specific instructions on how to use a given function or class,
we can invoke the  `help` function. As an example, let's
explore the usage instructions for tensors' `ones` function.

```{.python .input #lookup-api-specific-functions-and-classes-1}
%%tab mxnet
help(np.ones)
```

```{.python .input #lookup-api-specific-functions-and-classes-1}
%%tab pytorch
help(torch.ones)
```

```{.python .input #lookup-api-specific-functions-and-classes-1}
%%tab tensorflow
help(tf.ones)
```

```{.python .input #lookup-api-specific-functions-and-classes-1}
%%tab jax
help(jax.numpy.ones)
```

From the documentation, we can see that the `ones` function 
creates a new tensor with the specified shape 
and sets all the elements to the value of 1. 
Whenever possible, you should run a quick test 
to confirm your interpretation:

```{.python .input #lookup-api-specific-functions-and-classes-2}
%%tab mxnet
np.ones(4)
```

```{.python .input #lookup-api-specific-functions-and-classes-2}
%%tab pytorch
torch.ones(4)
```

```{.python .input #lookup-api-specific-functions-and-classes-2}
%%tab tensorflow
tf.ones(4)
```

```{.python .input #lookup-api-specific-functions-and-classes-2}
%%tab jax
jax.numpy.ones(4)
```

In the Jupyter notebook, we can use `?` to display the document in another
window. For example, `list?` will create content
that is almost identical to `help(list)`,
displaying it in a new browser window.
In addition, if we use two question marks, such as `list??`,
the Python code implementing the function will also be displayed.

The official documentation provides plenty of descriptions and examples that are beyond this book. 
We emphasize important use cases 
that will get you started quickly with practical problems, 
rather than completeness of coverage. 
We also encourage you to study the source code of the libraries 
to see examples of high-quality implementations of production code. 
By doing this you will become a better engineer 
in addition to becoming a better scientist.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/199)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17972)
:end_tab:

<!-- slides -->

::: {.slide title="Finding API Help"}
Every framework has thousands of functions and classes. You
won't memorize them — you'll **look them up**.

Two Python builtins do most of the work:

- `dir(module)` — what's in here?
- `help(thing)` (or `?thing` in Jupyter) — how do I use it?

Plus the official docs: pytorch.org, jax.dev, tensorflow.org,
mxnet.apache.org.
:::

::: {.slide title="`dir`: discovering the API"}
Standard import:

@lookup-api-documentation

. . .

`dir(...)` lists names in a module. Filter private names and
show a small prefix on slides; in a notebook you can inspect
the full list interactively:

@lookup-api-functions-and-classes-in-a-module
:::

::: {.slide title="`help`: usage details"}
Once you have the name, `help(...)` prints the docstring with
arguments, defaults, and a usage example:

@lookup-api-specific-functions-and-classes-1

. . .

Then run a one-liner to confirm the call:

@lookup-api-specific-functions-and-classes-2
:::

::: {.slide title="Recap"}
- `dir(module)` — list contents.
- `help(symbol)` (or `symbol?` in Jupyter) — show the docstring.
- Notebook autocomplete (`Tab`) is your fastest discovery tool.
- For prose-heavy explanations, deep links into the framework's
  official documentation beat the inline help.
:::
