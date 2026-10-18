```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Documentation
:label:`sec_lookup_api`

A framework API contains more functions, classes, and arguments than any
single text can cover, and it changes across releases. Effective use therefore
depends on being able to find an operation, inspect its interface, read its
documentation, and verify its behavior with a small example. This section
demonstrates that procedure using Python and notebook tools.

The official reference and tutorial pages document the supported interface for
each framework.

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

Many routine questions can be answered without leaving the notebook. The
procedure has four steps: discover, inspect, read, and verify.

![A four-step procedure for using an unfamiliar API: discover available names, inspect a candidate's signature, read its documentation or source, and verify its behavior with a small example.](../img/lookup-api-discovery-loop.svg)
:label:`fig_lookup_loop`

The examples below start from the standard import:

```{.python .input #lookup-api-documentation}
%%tab mxnet
from mxnet import np
from pprint import pprint
```

```{.python .input #lookup-api-documentation}
%%tab pytorch
import torch
from pprint import pprint
```

```{.python .input #lookup-api-documentation}
%%tab tensorflow
import tensorflow as tf
from pprint import pprint
```

```{.python .input #lookup-api-documentation}
%%tab jax
import jax
from pprint import pprint
```

## Discovering What Exists: `dir`

When you know roughly *where* a tool should live but not what it is called,
the `dir` function lists the names defined in a module.
For instance, to inspect the available random-sampling tools
(we print the first twenty names):

```{.python .input #lookup-api-functions-and-classes-in-a-module  n=1}
%%tab mxnet
pprint([name for name in dir(np.random)
        if not name.startswith('_')][:20], compact=True)
```

```{.python .input #lookup-api-functions-and-classes-in-a-module  n=1}
%%tab pytorch
pprint([name for name in dir(torch.distributions)
        if not name.startswith('_')][:20], compact=True)
```

```{.python .input #lookup-api-functions-and-classes-in-a-module  n=1}
%%tab tensorflow
pprint([name for name in dir(tf.random)
        if not name.startswith('_')][:20], compact=True)
```

```{.python .input #lookup-api-functions-and-classes-in-a-module}
%%tab jax
pprint([name for name in dir(jax.random)
        if not name.startswith('_')][:20], compact=True)
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
this often locates a name quickly.

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
When a docstring is terse or ambiguous, the source can clarify the
implementation. Reading library source also reveals established usage idioms.

## Verifying With a Quick Run

Docstrings can be terse, and they occasionally drift out of date.
A small example can verify the behavior directly:

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

The result has the documented shape and values.
Making this `discover → inspect → read → verify` loop a habit
will carry you through the unfamiliar corners of any library,
long after the specific functions in this book have changed.

Coding assistants can propose an API call for a question such as “how do I
sample from a normal distribution in this framework?” Treat the result as an
unverified candidate that still goes through the loop above.
Glance at the signature with `help` or `?`, run a small example,
and accept the suggestion only once it survives both.

## Summary

For an unfamiliar operation, discover candidate names, inspect the signature,
read the relevant documentation, and verify the behavior with the smallest run
that exposes shapes, dtypes, and values. This procedure remains valid as library
APIs change and applies equally to suggestions from search or coding assistants.

## Exercises

:begin_tab:`mxnet`
1. [code] **Scale versus rate.** The gamma distribution with shape
   $k = 2$ and rate $\beta = 3$ has mean $k/\beta = 2/3$ and variance
   $k/\beta^2 = 2/9$. Use `dir` on `np.random` to find its sampler and
   read the signature with `help` (or `?`) to find out how the sampler is
   parameterized. Draw 10,000 samples with $k = 2$ and $\beta = 3$ and
   confirm the mean and variance from the samples.
1. [code] **Standard-deviation conventions.** For $n$ numbers whose
   squared deviations from their mean sum to $S$, a library computes the
   standard deviation either as $\sqrt{S/n}$ or as $\sqrt{S/(n-1)}$.
   :numref:`sec_pandas` standardized columns with the pandas method `std`.
   Compute the standard deviation of the values 1.0, 2.0, 3.0 and 4.0
   once with `pd.Series.std` and once with `np.std` on an MXNet array.
   Use `help` on both functions to find out how each one sets the
   divisor, state the default divisor of each, and check both results
   against a hand computation. Then make the two results agree.
1. [code] **Checking a coding assistant's answer.** Ask a coding
   assistant the two questions below. Check each function it names in
   three steps: `dir` or `hasattr` shows that the name exists, `help` or
   `?` shows that the signature fits the suggested call, and a tiny
   example shows the claimed behavior.
    1. Ask how to split an MXNet tensor into three equal parts along an
       axis.
    1. Ask for a single MXNet function that reorders the axes of a tensor
       so that their lengths increase, an operation no library function
       provides. State which of the three checks, if any, each named
       function fails. Then build the operation from functions that pass
       all three checks and test it on a tensor of shape $(4, 2, 3)$.
:end_tab:

:begin_tab:`pytorch`
1. [code] **Scale versus rate.** The gamma distribution with shape
   $k = 2$ and rate $\beta = 3$ has mean $k/\beta = 2/3$ and variance
   $k/\beta^2 = 2/9$. Use `dir` on `torch.distributions` to find its class
   and read the signature with `help` (or `?`) to find out how the class is
   parameterized. Draw 10,000 samples with $k = 2$ and $\beta = 3$ and
   confirm the mean and variance from the samples.
1. [code] **Standard-deviation conventions.** For $n$ numbers whose
   squared deviations from their mean sum to $S$, a library computes the
   standard deviation either as $\sqrt{S/n}$ or as $\sqrt{S/(n-1)}$.
   :numref:`sec_pandas` standardized columns with the pandas method `std`.
   Compute the standard deviation of the values 1.0, 2.0, 3.0 and 4.0
   with `pd.Series.std`, with `torch.std`, and with `numpy.std`. Use
   `help` on each function to find out how it sets the divisor, state the
   default divisor of each, and check the three results against a hand
   computation. Then make all three results agree.
1. [code] **Checking a coding assistant's answer.** Ask a coding
   assistant the two questions below. Check each function it names in
   three steps: `dir` or `hasattr` shows that the name exists, `help` or
   `?` shows that the signature fits the suggested call, and a tiny
   example shows the claimed behavior.
    1. Ask how to split a PyTorch tensor into three equal parts along an
       axis.
    1. Ask for a single PyTorch function that reorders the axes of a
       tensor so that their lengths increase, an operation no library
       function provides. State which of the three checks, if any, each
       named function fails. Then build the operation from functions
       that pass all three checks and test it on a tensor of shape
       $(4, 2, 3)$.
:end_tab:

:begin_tab:`tensorflow`
1. [code] **Scale versus rate.** The gamma distribution with shape
   $k = 2$ and rate $\beta = 3$ has mean $k/\beta = 2/3$ and variance
   $k/\beta^2 = 2/9$. Use `dir` on `tf.random` to find its sampler and
   read the signature with `help` (or `?`) to find out how the sampler is
   parameterized. Draw 10,000 samples with $k = 2$ and $\beta = 3$ and
   confirm the mean and variance from the samples.
1. [code] **Standard-deviation conventions.** For $n$ numbers whose
   squared deviations from their mean sum to $S$, a library computes the
   standard deviation either as $\sqrt{S/n}$ or as $\sqrt{S/(n-1)}$.
   :numref:`sec_pandas` standardized columns with the pandas method `std`.
   Compute the standard deviation of the values 1.0, 2.0, 3.0 and 4.0
   once with `pd.Series.std` and once with `tf.math.reduce_std`. Use
   `help` on both functions to find out how each one sets the divisor,
   state the default divisor of each, and check both results against a
   hand computation. Then make the two results agree.
1. [code] **Checking a coding assistant's answer.** Ask a coding
   assistant the two questions below. Check each function it names in
   three steps: `dir` or `hasattr` shows that the name exists, `help` or
   `?` shows that the signature fits the suggested call, and a tiny
   example shows the claimed behavior.
    1. Ask how to split a TensorFlow tensor into three equal parts along an
       axis.
    1. Ask for a single TensorFlow function that reorders the axes of a
       tensor so that their lengths increase, an operation no library
       function provides. State which of the three checks, if any, each
       named function fails. Then build the operation from functions
       that pass all three checks and test it on a tensor of shape
       $(4, 2, 3)$.
:end_tab:

:begin_tab:`jax`
1. [code] **Scale versus rate.** The gamma distribution with shape
   $k = 2$ and rate $\beta = 3$ has mean $k/\beta = 2/3$ and variance
   $k/\beta^2 = 2/9$. Use `dir` on `jax.random` to find its sampler and
   read the signature with `help` (or `?`) to find out how the sampler is
   parameterized. Draw 10,000 samples with $k = 2$ and $\beta = 3$, using
   an explicit key, and confirm the mean and variance from the samples.
1. [code] **Standard-deviation conventions.** For $n$ numbers whose
   squared deviations from their mean sum to $S$, a library computes the
   standard deviation either as $\sqrt{S/n}$ or as $\sqrt{S/(n-1)}$.
   :numref:`sec_pandas` standardized columns with the pandas method `std`.
   Compute the standard deviation of the values 1.0, 2.0, 3.0 and 4.0
   once with `pd.Series.std` and once with `jax.numpy.std`. Use `help` on
   both functions to find out how each one sets the divisor, state the
   default divisor of each, and check both results against a hand
   computation. Then make the two results agree.
1. [code] **Checking a coding assistant's answer.** Ask a coding
   assistant the two questions below. Check each function it names in
   three steps: `dir` or `hasattr` shows that the name exists, `help` or
   `?` shows that the signature fits the suggested call, and a tiny
   example shows the claimed behavior.
    1. Ask how to split a JAX array into three equal parts along an axis.
    1. Ask for a single JAX function that reorders the axes of an array
       so that their lengths increase, an operation no library function
       provides. State which of the three checks, if any, each named
       function fails. Then build the operation from functions that pass
       all three checks and test it on an array of shape $(4, 2, 3)$.
:end_tab:

4. [code] **Reading the source.** The docstring of `d2l.plot`, defined in
   :numref:`sec_calculus`, describes the function only as "Plot data
   points." After importing `d2l` as in that section, predict how many
   curves `d2l.plot(x, Y)` draws when `x` is a 1-D tensor of length 50 and
   `Y` is a list of five tensors of the same length. Read the source with
   `d2l.plot??` or your editor's go-to-definition, find the line that
   decides the number of curves, and confirm your reading with the call.
   Then pass an argument that makes all five curves appear.

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

Using an unfamiliar API<br>**discover · inspect · read · verify**.
:::
:::

::: {.slide title="A procedure for consulting an API"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
No book covers an entire framework, and libraries change across releases.
Four steps answer most routine questions from within a notebook: discover
available names, inspect the interface, read the documentation, and verify the
behavior with a small example.

::: {.d2l-note}
Use the official reference and tutorial pages for the supported interface.
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
shows the **source code**, which can clarify a terse or ambiguous docstring.
:::
:::

::: {.slide title="A tiny run settles it"}
[Verify]{.kicker}

Docstrings can drift out of date; verify the current behavior with a small call:

@lookup-api-specific-functions-and-classes-2

The result has the documented shape and values. The
**discover → inspect → read → verify** loop remains useful as APIs change.
:::

::: {.slide title="Coding assistants enter the same loop"}
[Assistants]{.kicker}

An assistant may produce a plausible function and call. Treat the suggestion
as a candidate to check before building on it.

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
