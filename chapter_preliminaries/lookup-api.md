```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Documentation
:label:`sec_lookup_api`

No matter how much of a framework's API we cover here,
there will always be functions, classes, and arguments
we never reach, and the libraries keep changing under us.
So rather than try to memorize the API,
the durable skill is getting good at *looking things up*:
finding what exists, reading how it works,
and confirming that it does what you think.
This short section lays out a small, repeatable loop for exactly that,
using tools built into Python and your notebook.

The official documentation is always the source of truth;
bookmark the reference and tutorial pages for the framework you use.

:begin_tab:`mxnet`
For MXNet these are the
[API reference](https://mxnet.apache.org/versions/1.9.1/api) and the
[tutorials](https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/).
One caveat: these pages document version 1.9.1
(the last release with hosted documentation),
which foregrounds the legacy `mx.nd` interface,
whereas this book uses the NumPy-style `np`/`npx` API of MXNet 2.
For MXNet specifics, the in-notebook loop below
is often the more reliable reference.
:end_tab:

:begin_tab:`pytorch`
For PyTorch these are the
[API reference](https://pytorch.org/docs/stable/index.html) and the
[tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html).
:end_tab:

:begin_tab:`tensorflow`
For TensorFlow these are the
[API reference](https://www.tensorflow.org/api_docs) and the
[tutorials](https://www.tensorflow.org/tutorials).
:end_tab:

:begin_tab:`jax`
For JAX these are the
[API reference](https://jax.readthedocs.io/en/latest/) and the
[tutorials](https://jax.readthedocs.io/en/latest/tutorials.html).
:end_tab:

For most day-to-day questions, though, you do not need to leave your
notebook. Four moves, repeated until the call behaves, cover almost
everything.

![Four moves, repeated until the call does what you want: **discover** the names that exist, **inspect** a candidate's signature, **read** the docs or source when you need the *why*, and **verify** with a quick run.](../img/lookup-api-discovery-loop.svg)
:label:`fig_lookup_loop`

The examples below start from the standard import:

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

## Discovering What Exists: `dir`

When you know roughly *where* a tool should live but not what it is called,
the `dir` function lists everything defined in a module.
For instance, to see what is on offer for random sampling
(we print the first twenty names):

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

We can usually ignore names that begin and end with `__`
(Python's special objects) or that start with a single `_`
(internal helpers). The remaining names already hint
at what the module offers.

:begin_tab:`mxnet`
Here almost every name is a sampler: draws from classical distributions
(`beta`, `gamma`, `multinomial`, `normal`, ...) alongside NumPy-style
conveniences such as `rand`, `randint`, and `randn`.
:end_tab:

:begin_tab:`pytorch`
Here the names are distribution *classes* such as `Bernoulli`,
`Categorical`, and `Gamma`; each can be instantiated and then sampled
from. The `Transform` entries build new distributions by transforming
existing ones.
:end_tab:

:begin_tab:`tensorflow`
Here we can spot samplers such as `gamma`, `normal`, and `poisson`,
next to utilities like `Generator` and `set_seed` that manage the
random state.
:end_tab:

:begin_tab:`jax`
Here the names are samplers (`bernoulli`, `beta`, `cauchy`,
`exponential`, `gamma`, ...) plus `PRNGKey`, which creates the explicit
random key that every JAX sampler takes.
:end_tab:

In a notebook you can get the same list interactively, filtered as you
type, by writing the module name followed by a dot and pressing `Tab`;
this is usually the fastest way to turn up a name.

## Reading the Signature: `help`, `?`, and `??`

Once you have a name, `help` prints its docstring:
the arguments it takes, their defaults, what it returns,
and often a short example. Let us look up the `ones` function,
which we have used to build tensors:

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

The docstring tells us that `ones` creates a new tensor of the requested
shape with every element set to 1.
In a Jupyter notebook, two shortcuts make this quicker still:
`ones?` opens the same docstring in a side pane,
and `ones??` additionally displays the function's *source code*.
The source is the final word when a docstring is terse or ambiguous,
and reading it is one of the better ways to pick up idioms
from high-quality libraries.

## Verifying With a Quick Run

Docstrings can be terse, and they occasionally drift out of date.
The fastest way to be certain is to run a tiny example
and look at the result:

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

The shape and values are exactly what the docstring promised.
Making this `discover → inspect → read → verify` loop a habit
will carry you through the unfamiliar corners of any library,
long after the specific functions in this book have changed.

Coding assistants are often the quickest route to a first answer:
ask "how do I sample from a normal distribution in this framework?"
and you will usually get a function and a working call in seconds.
Treat the suggestion the way you would a knowledgeable colleague's tip:
a good starting point that still goes through the loop above.
Glance at the signature with `help` or `?`, run a small example,
and rely on the suggestion once it survives both.

## Exercises

1. Use `dir` on your framework's random-number module to find the routine
   that samples from a *uniform* distribution. Read its signature with `help`
   (or `?`), then call it to draw a $3 \times 3$ tensor and confirm the values
   lie in $[0, 1)$.
1. You want to reduce a tensor along a single axis but cannot remember the
   keyword. Look up your framework's `sum` (or `reduce_sum`) with `help`,
   identify the argument that selects the axis, and verify on a $2 \times 3$
   tensor that summing over each axis gives the shape you predicted.
1. Ask a coding assistant "how do I concatenate two tensors along a new axis
   in my framework?" Then run its answer through the
   discover&nbsp;&rarr;&nbsp;inspect&nbsp;&rarr;&nbsp;read&nbsp;&rarr;&nbsp;verify
   loop: does the suggested function exist (`dir`), does its signature match
   the claim (`help`/`?`), and does a tiny example do what you expect?

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

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.7]{.kicker}

APIs change; the skill of looking things up doesn't<br>**discover · inspect · read · verify**.
:::
:::

::: {.slide title="You cannot memorize an API, so loop instead"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
No book covers a whole framework, and the libraries keep changing under
us. The durable skill is a four-move loop, run without leaving the
notebook, repeated until the call does what you want.

::: {.d2l-note}
The official docs remain the source of truth: bookmark your
framework's reference and tutorial pages.
:::
:::

::: {.col .fig .big}
@fig:lookup-api-discovery-loop
:::
:::
:::

::: {.slide title="dir discovers what exists"}
[Discover]{.kicker}

Know roughly *where* a tool should live, but not its name? `dir` lists a
module's contents; the names alone sketch what is on offer:

@lookup-api-functions-and-classes-in-a-module

Skip the `_`-prefixed internals. In a notebook, `module.` + `Tab` gives
the same list, filtered as you type, usually the fastest way to turn up
a name.
:::

::: {.slide title="help reads the signature; ?? reads the source"}
[Inspect · read]{.kicker}

`help(...)` prints the docstring: arguments, defaults, return value,
often an example.

@-lookup-api-specific-functions-and-classes-1

. . .

::: {.d2l-note}
In Jupyter, `ones?` opens the docstring in a side pane, and `ones??`
shows the **source code**: the final word when a docstring is terse or
ambiguous.
:::
:::

::: {.slide title="A tiny run settles it"}
[Verify]{.kicker}

Docstrings drift out of date; a running call does not lie:

@lookup-api-specific-functions-and-classes-2

Exactly the promised shape and values. This
**discover → inspect → read → verify** loop outlives every function in
this book.
:::

::: {.slide title="Coding assistants enter the same loop"}
[Assistants]{.kicker}

An assistant usually produces a plausible function and a working call in
seconds. Treat the suggestion like a knowledgeable colleague's tip: a
good starting point that still gets a two-line check before you build on it.

::: {.d2l-note .rule}
Glance at the signature (`help` / `?`), then run a small example. A
suggestion that survives both is one you can rely on.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Discover** with `dir` (or `Tab`-completion).
- **Inspect** with `help` / `?`; **read** the source with `??`.
- **Verify** with a tiny run.
- Assistant answers enter the same loop before you rely on them.
:::
