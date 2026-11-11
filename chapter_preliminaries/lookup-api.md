```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Documentation
:label:`sec_lookup_api`

The sections of this chapter explained every function that they used. This
courtesy cannot last. Every project soon needs a function that the book
never mentions, or uses a familiar one in an unfamiliar way, and no book can
keep up: the main module of a deep learning framework defines several
hundred to a thousand public names, and new releases add names and deprecate
others. What the book can teach instead is how to find out for ourselves,
reliably, what a function does.

Reliably is the operative word, because a call can look harmless and still
do something other than what we meant. Suppose that we need samples from a
gamma distribution, a distribution of positive numbers often used for
waiting times, with shape $k = 2$ and rate $\beta = 3$. Its mean is
$k / \beta = 2/3$ and its variance $k / \beta^2 = 2/9$. Libraries disagree
on how such a distribution is described: some take the rate $\beta$, others
the scale $1/\beta$, and some neither. A call that passes the numbers 2 and
3 runs either way, but its samples have mean $2/3$ under one reading and
$2 \cdot 3 = 6$ under the other. Nothing in the call tells us which reading
applies. The function's signature and documentation do.

Instead of a list of functions, this section gives a procedure of four
steps, shown in :numref:`fig_lookup_loop`. We discover the names that a
library offers, inspect the arguments and the documentation of a candidate,
read the reference pages or the source code when the documentation leaves a
question open, and verify the behavior with a small example whose result we
predicted. All four steps work inside a notebook, and by the end of the
section they have drawn our gamma samples, with the right mean.

![Four steps for finding out what a function does: discover the names that a module defines, inspect the signature and docstring of a candidate, read the reference documentation or the source code, and verify the behavior with a small example. A result that contradicts the prediction, or answers a question that the prediction left open, sends us back to read further.](../img/lookup-api-discovery-loop.svg)
:label:`fig_lookup_loop`

Besides the framework, we import three helpers from Python's standard
library: `pprint` prints long lists compactly, `inspect` reads the
signature of a function, and `textwrap` wraps long lines.

:begin_tab:`mxnet`
MXNet's array module is imported under the name `mnp`, so that `np` can keep
meaning NumPy.
:end_tab:

```{.python .input #lookup-api-documentation}
%%tab mxnet
import inspect
import textwrap
from pprint import pprint
from mxnet import np as mnp, npx
npx.set_np()
```

```{.python .input #lookup-api-documentation}
%%tab pytorch
import inspect
import textwrap
from pprint import pprint
import torch
```

```{.python .input #lookup-api-documentation}
%%tab tensorflow
import inspect
import textwrap
from pprint import pprint
import tensorflow as tf
```

```{.python .input #lookup-api-documentation}
%%tab jax
import inspect
import random
import textwrap
from pprint import pprint
import jax
```

## Discovering Names

Our first question is whether the library can sample from a gamma
distribution at all, and under which name. We know roughly where to look,
in the module for random numbers, but not what its functions are called.
The built-in function `dir` lists the names that a module defines. Names that begin with an underscore
belong to the internals of the module, so we drop them and print the first
twenty of the rest:

:begin_tab:`pytorch`
PyTorch keeps its probability distributions in the module
`torch.distributions`.
:end_tab:

```{.python .input #lookup-api-functions-and-classes-in-a-module}
%%tab mxnet
pprint([name for name in dir(mnp.random)
        if not name.startswith('_')][:20], compact=True)
```

```{.python .input #lookup-api-functions-and-classes-in-a-module}
%%tab pytorch
pprint([name for name in dir(torch.distributions)
        if not name.startswith('_')][:20], compact=True)
```

```{.python .input #lookup-api-functions-and-classes-in-a-module}
%%tab tensorflow
pprint([name for name in dir(tf.random)
        if not name.startswith('_')][:20], compact=True)
```

```{.python .input #lookup-api-functions-and-classes-in-a-module}
%%tab jax
pprint([name for name in dir(jax.random)
        if not name.startswith('_')][:20], compact=True)
```

:begin_tab:`mxnet`
Almost every name is a sampler. Draws from classical distributions, such as
`beta`, `gamma`, `multinomial`, and `normal`, stand next to the NumPy
conveniences `rand`, `randint`, and `randn`.
:end_tab:

:begin_tab:`pytorch`
The names are distribution classes, such as `Bernoulli`, `Categorical`, and
`Gamma`. Each class is created with the parameters of its distribution and
then sampled from. Names that end in `Transform` are invertible maps, which
`TransformedDistribution` applies to an existing distribution to build a new
one.
:end_tab:

:begin_tab:`tensorflow`
We see samplers such as `gamma`, `normal`, and `poisson`, next to
`Generator` and `set_seed`, which manage the state of the random number
generator.
:end_tab:

:begin_tab:`jax`
The names are samplers, such as `bernoulli`, `beta`, `cauchy`,
`exponential`, and `gamma`, along with `Key`, the type of a random key, and
`PRNGKey`, an older function that creates the key every JAX sampler takes.
This book uses its successor, `jax.random.key`, which comes later in the
alphabet.
:end_tab:

A notebook offers the same list while we type: writing the name of a module
followed by a dot and pressing Tab lists the names that the module defines,
narrowed to those that match the letters typed next. When we do not know
which module to look in, the search box of the framework's online reference
searches all of them.

## Reading Documentation

The list has an entry for the gamma distribution. To find out how to call
it, the built-in function `help` prints its *docstring*, the documentation
written into its code. A docstring says what the function does, what each
argument means, and what the function returns, and it often ends with
examples. The first lines show the function's *signature*, the list of
arguments that it accepts. Three conventions of Python are all we need to
read one. An argument written as `name=value` has a default value and may be
omitted. Arguments that follow a lone `*` can be passed only by keyword. An
argument written as `*name` collects any number of positional arguments;
our sampler has none of these, but the exercises meet one.

The whole docstring is worth reading once in a notebook, but it is long, and
our question is narrow: does the sampler take a rate or a scale? We
therefore print the signature, which `inspect.signature` returns, and only
those lines of the docstring that mention a rate or a scale. Since we print
single lines, a sentence may arrive cut off; the whole of it is one `help`
call away.

```{.python .input #lookup-api-reading-documentation-1}
def doc_lines(f, *words):
    """Print the signature of f and the lines of its docstring with a word."""
    print(textwrap.fill(f'{f.__name__}{inspect.signature(f)}', 79,
                        subsequent_indent='    '))
    for line in inspect.getdoc(f).splitlines():
        if any(word in line for word in words):
            print('   ', line.strip())
```

```{.python .input #lookup-api-reading-documentation-2}
%%tab mxnet
doc_lines(mnp.random.gamma, 'rate', 'scale')
```

```{.python .input #lookup-api-reading-documentation-2}
%%tab pytorch
doc_lines(torch.distributions.Gamma, 'rate', 'scale')
```

```{.python .input #lookup-api-reading-documentation-2}
%%tab tensorflow
doc_lines(tf.random.gamma, 'rate', 'scale')
```

```{.python .input #lookup-api-reading-documentation-2}
%%tab jax
doc_lines(jax.random.gamma, 'rate', 'scale')
```

With the signature and these lines, we can predict what a call does before
we run it:

:begin_tab:`mxnet`
The signature `gamma(shape, scale=1.0, size=None, ...)` holds a trap. The
argument named `shape` is the shape parameter $k$ of the distribution, not
the shape of the output, which is `size`. The second argument is a scale,
$1/\beta$. We predict that `mnp.random.gamma(2, 1 / 3, size=10000)` draws
10,000 samples with mean $2/3$ and variance $2/9$, and that
`mnp.random.gamma(2, 3)` would draw samples with mean 6.
:end_tab:

:begin_tab:`pytorch`
`Gamma` is a class with the arguments `concentration`, which the docstring
calls the shape parameter $k$, and `rate`, the rate $\beta$ itself, which
it also states as $1 / \textrm{scale}$. One of the printed lines is an
example from the end of the docstring, in which `m` is a `Gamma(1, 1)`; it
confirms the vocabulary. We predict that samples of `Gamma(2.0, 3.0)` have
mean $2/3$ and variance $2/9$.
:end_tab:

:begin_tab:`tensorflow`
The signature `(shape, alpha, beta=None, ...)` holds a trap. (Its name is
printed as `random_gamma`, the name of the function in its own module;
`tf.random.gamma` is the public alias.) Here the first argument, `shape`, is
the shape of the output, while the shape parameter $k$ of the distribution
is `alpha`. The docstring calls `beta` the inverse scale, which is the rate.
We predict that `tf.random.gamma([10000], alpha=2.0, beta=3.0)` draws
10,000 samples with mean $2/3$ and variance $2/9$.
:end_tab:

:begin_tab:`jax`
The signature `gamma(key, a, shape=None, ...)`, printed with its type
hints, has neither a rate nor a scale. The sampler draws from the gamma
distribution with shape parameter `a` and rate 1, and the docstring tells
us to divide the samples by the rate. We predict that
`jax.random.gamma(key, 2.0, (10000,)) / 3` has mean $2/3$ and variance
$2/9$.
:end_tab:

In a notebook, two shortcuts show the same documentation. Appending `?` to
a name shows the docstring in a separate pane, and appending `??` also
shows the source code of the function. The source answers
questions that a docstring leaves open, such as how an argument is converted
or checked. Unfortunately, only functions written in Python have source code
to show. Many core functions are compiled from C++, and for those `??` shows
the docstring alone.

When the docstring is not enough, the framework's reference documentation
describes every public function on a page of its own, and its tutorials show
the functions in use.

:begin_tab:`mxnet`
The Apache Software Foundation retired MXNet in 2023, but the project's
website still serves the documentation of its development version, MXNet
2.0, whose NumPy-style interface this book uses: the
[reference of `mxnet.np`](https://mxnet.apache.org/versions/master/api/python/docs/api/np/index.html)
and the
[tutorials](https://mxnet.apache.org/versions/master/api/python/docs/tutorials/index.html).
The pages of version 1.9.1, the last formal release, describe the older
`mx.nd` interface instead. Where the documentation is thin, the steps of
this section, carried out in a notebook, are the more reliable reference.
:end_tab:

:begin_tab:`pytorch`
For PyTorch, these are the
[API reference](https://docs.pytorch.org/docs/stable/index.html) and the
[tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
:end_tab:

:begin_tab:`tensorflow`
For TensorFlow, these are the
[API reference](https://www.tensorflow.org/api_docs/python/tf) and the
[tutorials](https://www.tensorflow.org/tutorials).
:end_tab:

:begin_tab:`jax`
For JAX, these are the
[API reference](https://docs.jax.dev/en/latest/jax.html) and the
[tutorials](https://docs.jax.dev/en/latest/tutorials.html).
:end_tab:

## Verifying Behavior

Our reading of a docstring can be wrong, and occasionally the docstring
itself is out of date. A small example settles the question, provided that
we predict its result before we run it, so that the run could prove us
wrong. A run that merely completes without an error proves little. Our
prediction can fail: had we read the rate as a scale, the samples would have
mean 6 instead of $2/3$. We draw 10,000 samples and compare their mean and
variance with $2/3 \approx 0.667$ and $2/9 \approx 0.222$. The mean of
10,000 independent draws has standard error $\sqrt{(2/9) / 10{,}000}
\approx 0.005$ (:numref:`sec_prob`), so the sample mean should lie within
about $0.01$ of $2/3$:

```{.python .input #lookup-api-verifying-behavior}
%%tab mxnet
samples = mnp.random.gamma(2, 1 / 3, size=10000)  # k = 2, scale 1/3
samples.mean(), samples.var()
```

```{.python .input #lookup-api-verifying-behavior}
%%tab pytorch
samples = torch.distributions.Gamma(2.0, 3.0).sample((10000,))  # k = 2, rate 3
samples.mean(), samples.var()
```

```{.python .input #lookup-api-verifying-behavior}
%%tab tensorflow
samples = tf.random.gamma([10000], alpha=2.0, beta=3.0)  # k = 2, rate 3
tf.reduce_mean(samples), tf.math.reduce_variance(samples)
```

```{.python .input #lookup-api-verifying-behavior}
%%tab jax
key = jax.random.key(random.getrandbits(32))  # A new seed on each run
samples = jax.random.gamma(key, 2.0, (10000,)) / 3  # k = 2, then rate 3
samples.mean(), samples.var()
```

The prediction holds: the mean lies close to $2/3$ and the variance close to
$2/9$, far from the values that the other reading would give. When a result
contradicts the prediction, or answers a question that the prediction left
open, we go around the loop again: we read further, in the reference
documentation or the source, and test the new prediction.

Coding assistants answer many such questions directly. Asked for gamma
samples with a given rate, an assistant proposes a call, and the call
usually looks plausible. Such a suggestion is a candidate, and it goes
through the same steps as a name we found ourselves: we check that the name
exists, for example with `dir` or `hasattr`, read its signature, predict the
result, and run a small example. The last step matters most, because a
wrong suggestion need not fail: it can name a function that exists, fit
its signature, and run, yet compute something else.

With this procedure the tools of the chapter are complete: tensors, tables,
linear algebra, derivatives, probability, and a way to find what the book
leaves out. :numref:`chap_regression` uses them to build and train a first
model.

## Summary

No text can record the details of every function in a deep learning framework,
and a call that looks harmless may read its arguments differently than we
assume: the same two numbers describe a gamma distribution by its rate in
one library and by its scale in another. Four steps settle such questions from within a notebook. `dir` and
Tab completion discover the names that a module defines, and `help` and `?`
show the docstring of a candidate and its signature, which says what each
argument means. The reference documentation and `??` supply
the details that the docstring omits, and a small example whose result we
predicted verifies our reading. Suggestions from coding assistants are
candidates that go through the same steps.

## Exercises

:begin_tab:`mxnet`
1. [code] **Two integers or one tuple.** `mnp.ones` creates an array of
   ones. Read its signature with `help(mnp.ones)`, which begins
   `ones(shape, dtype=None, ...)`, and predict, before running anything,
   what each of `mnp.ones((2, 3))`, `mnp.ones(2, 3)`,
   `mnp.ones(shape=(2, 3))`, and `mnp.ones(size=(2, 3))` returns or raises.
   Then run the four calls and compare. For each call that fails, state
   which argument received the offending value and whether the error
   message says so.
1. [code] **Standard-deviation conventions.** For $n$ numbers whose
   squared deviations from their mean sum to $S$, a library computes the
   standard deviation either as $\sqrt{S/n}$ or as $\sqrt{S/(n-1)}$.
   :numref:`sec_pandas` standardized columns with the pandas method `std`.
   Compute the standard deviation of the values 1.0, 2.0, 3.0 and 4.0
   once with `pd.Series.std` and once with `mnp.std` on an MXNet array.
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
       so that their lengths increase. Whatever the assistant answers,
       check each named function in the three steps, state which step, if
       any, it fails, and say whether the claimed behavior holds. If no
       suggestion passes, build the operation from functions that do, and
       test it on a tensor of shape $(4, 2, 3)$.
:end_tab:

:begin_tab:`pytorch`
1. [code] **Two integers or one tuple.** `torch.ones` creates a tensor of
   ones. Read its signature with `help(torch.ones)`, which begins
   `ones(*size, *, ...)`, and predict, before running anything, what each
   of `torch.ones(2, 3)`, `torch.ones((2, 3))`, `torch.ones(size=(2, 3))`,
   and `torch.ones(shape=(2, 3))` returns or raises. Then run the four calls
   and compare. Name the convention of this section that decides each
   case.
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
       tensor so that their lengths increase. Whatever the assistant
       answers, check each named function in the three steps, state which
       step, if any, it fails, and say whether the claimed behavior holds.
       If no suggestion passes, build the operation from functions that
       do, and test it on a tensor of shape $(4, 2, 3)$.
:end_tab:

:begin_tab:`tensorflow`
1. [code] **Two integers or one tuple.** `tf.ones` creates a tensor of
   ones. Read its signature with `help(tf.ones)`, which begins
   `ones(shape, dtype=tf.float32, ...)`, and predict, before running
   anything, what each of `tf.ones((2, 3))`, `tf.ones(2, 3)`, and
   `tf.ones(size=(2, 3))` returns or raises. Then run the three calls. One
   of them runs without an error and returns something other than a
   $2 \times 3$ matrix. Read the source with `tf.ones??`, find the line that
   converts the second argument, and explain the result; `tf.as_dtype(3)`
   is a hint. Which of the four steps of this section caught it?
1. [code] **Standard-deviation conventions.** For $n$ numbers whose
   squared deviations from their mean sum to $S$, a library computes the
   standard deviation either as $\sqrt{S/n}$ or as $\sqrt{S/(n-1)}$.
   :numref:`sec_pandas` standardized columns with the pandas method `std`.
   Compute the standard deviation of the values 1.0, 2.0, 3.0 and 4.0
   once with `pd.Series.std` and once with `tf.math.reduce_std`, and
   check both results against a hand computation. Use `help` on both
   functions. State the default divisor of `pd.Series.std` and the
   argument that changes it. `tf.math.reduce_std` has no divisor
   argument: infer its divisor from the example in its docstring. Then
   change the pandas call so that the two results agree.
1. [code] **Checking a coding assistant's answer.** Ask a coding
   assistant the two questions below. Check each function it names in
   three steps: `dir` or `hasattr` shows that the name exists, `help` or
   `?` shows that the signature fits the suggested call, and a tiny
   example shows the claimed behavior.
    1. Ask how to split a TensorFlow tensor into three equal parts along an
       axis.
    1. Ask for a single TensorFlow function that reorders the axes of a
       tensor so that their lengths increase. Whatever the assistant
       answers, check each named function in the three steps, state which
       step, if any, it fails, and say whether the claimed behavior holds.
       If no suggestion passes, build the operation from functions that
       do, and test it on a tensor of shape $(4, 2, 3)$.
:end_tab:

:begin_tab:`jax`
1. [code] **Two integers or one tuple.** `jnp.ones` creates an array of
   ones. Read its signature with `help(jnp.ones)`, which begins
   `ones(shape, dtype=None, *, ...)`, and predict, before running anything,
   what each of `jnp.ones((2, 3))`, `jnp.ones(2, 3)`,
   `jnp.ones(shape=(2, 3))`, and `jnp.ones(size=(2, 3))` returns or raises.
   Then run the four calls and compare. For each call that fails, state
   which argument received the offending value and whether the error
   message says so.
1. [code] **Standard-deviation conventions.** For $n$ numbers whose
   squared deviations from their mean sum to $S$, a library computes the
   standard deviation either as $\sqrt{S/n}$ or as $\sqrt{S/(n-1)}$.
   :numref:`sec_pandas` standardized columns with the pandas method `std`.
   Compute the standard deviation of the values 1.0, 2.0, 3.0 and 4.0
   once with `pd.Series.std` and once with `jnp.std`. Use `help` on both
   functions to find out how each one sets the divisor, state the default
   divisor of each, and check both results against a hand computation.
   Then make the two results agree.
1. [code] **Checking a coding assistant's answer.** Ask a coding
   assistant the two questions below. Check each function it names in
   three steps: `dir` or `hasattr` shows that the name exists, `help` or
   `?` shows that the signature fits the suggested call, and a tiny
   example shows the claimed behavior.
    1. Ask how to split a JAX array into three equal parts along an axis.
    1. Ask for a single JAX function that reorders the axes of an array
       so that their lengths increase. Whatever the assistant answers,
       check each named function in the three steps, state which step, if
       any, it fails, and say whether the claimed behavior holds. If no
       suggestion passes, build the operation from functions that do, and
       test it on an array of shape $(4, 2, 3)$.
:end_tab:

4. [code] **Reading the source.** The plotting function `d2l.plot`,
   defined in :numref:`sec_utils`, accepts its data in several forms:
   a single tensor, a tensor of $x$-values with one tensor of $y$-values,
   or a tensor of $x$-values with several curves. Import `d2l` as at the
   start of this section, read the source of `d2l.plot` with `d2l.plot??` or your
   editor's go-to-definition, and find the lines that decide how many
   curves it draws and which values serve as their $x$-coordinates. Let
   `x` hold 50 evenly spaced values from $-2$ to $2$ (use `linspace`), and
   let `Y` be a matrix of shape $(3, 50)$.
   Predict from the source what `d2l.plot(x)`, `d2l.plot(x, x ** 2)`, and
   `d2l.plot(x, Y)` draw, in particular whether `Y` gives three curves of
   50 points or fifty curves of 3 points. Then run the three calls and
   compare.

1. [code] **Tests that can fail.** ● A verifying run is informative only if
   it would have given a different result had the claim been false.
    1. Find the framework's elementwise rounding function with `dir`, and
       design an input of at most four numbers on which three rules give
       three different results: round halves up, round halves away from
       zero, and round halves to the nearest even integer. Run the function
       on this input and state which rule the framework follows.
    1. A sort is *stable* if elements that compare equal keep their
       original order. Find the framework's `argsort`, and read what its
       docstring promises about stability. Then test stability on the input
       $(1, 0, 1, 0)$ and on 1,000 random integers from $\{0, 1\}$. Explain
       why a passing test on a short input cannot establish stability while
       a single failure refutes it, and state whether the behavior you
       observe agrees with the docstring.

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

A procedure for finding out what a function does<br>**discover · inspect · read · verify**
:::
:::

::: {.slide title="Gamma(2, 3): Mean 2/3 or Mean 6?"}
[The problem]{.kicker}

We need samples from a gamma distribution with shape $k = 2$ and rate
$\beta = 3$: mean $k/\beta = 2/3$, variance $k/\beta^2 = 2/9$.

| the sampler takes | a call with 2 and 3 draws |
|---|---|
| the rate $\beta$ | mean $2/3$ |
| the scale $1/\beta$ | mean $2 \cdot 3 = 6$ |
| neither | mean $2$, to be rescaled by hand |

- Both readings run without an error. Nothing in the call says which applies.
- A framework defines hundreds of public names; releases add and deprecate.

::: {.d2l-note}
Memorizing fails. We need a procedure that finds out what a call does.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- list the names that a module defines;
- read a signature: defaults, keyword-only arguments, `*args`;
- predict a call's result from its signature and docstring, before
  running it;
- find the docstring, the reference page, and the source;
- design a small test that could prove your reading wrong;
- check a coding assistant's suggestion.

**Running example:** gamma samples with $k = 2$ and rate $3$. Does the
sampler take a rate or a scale?
:::

::: {.slide title="Documentation Covers What the Book Leaves Out"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | tool |
|---|---|
| Data Manipulation | tensors |
| Data Preprocessing | pandas |
| Linear Algebra | matrix products, norms, eigenvalues |
| Calculus | derivatives, gradients, chain rule |
| Automatic Differentiation | autograd |
| Probability and Statistics | standard error, Bayes' theorem |
| **Documentation** | **`help`, small tests** |
:::

::: {.col .narrow}
- **Uses:** the random samplers and the standard error of the earlier
  decks.
- **Skips:** debuggers, profilers, and type checkers.
:::
:::
:::

::: {.slide title="Four Steps Find Out What a Function Does"}
[The procedure]{.kicker}

@fig:lookup-api-discovery-loop

1. **Discover** the names a module defines.
2. **Inspect** a candidate's signature and docstring.
3. **Read** the reference page or the source when a question stays open.
4. **Verify** with a small run whose result you predicted.
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Discover and inspect]{.dtitle}

[dir · help · signatures · a prediction]{.dsub}
:::
:::

::: {.slide title="dir Lists the Distributions PyTorch Can Sample From" only="pytorch"}
[Discover]{.kicker}

Which distributions can we sample from? `dir` lists the names in
`torch.distributions`; names starting with `_` are internals:

@lookup-api-functions-and-classes-in-a-module

- The names are **classes**: create `Gamma(...)` with its parameters,
  then sample from it.
- `...Transform`: invertible maps that build new distributions.
- In a notebook: `torch.distributions.` then Tab, filtered as you type.
:::

::: {.slide title="dir Lists the Samplers in tf.random" only="tensorflow"}
[Discover]{.kicker}

Which distributions can we sample from? `dir` lists the names in
`tf.random`; names starting with `_` are internals:

@lookup-api-functions-and-classes-in-a-module

- **Samplers** such as `gamma`, `normal`, `poisson`.
- `Generator` and `set_seed` manage the state of the random numbers.
- In a notebook: `tf.random.` then Tab, filtered as you type.
:::

::: {.slide title="dir Lists the Samplers in jax.random" only="jax"}
[Discover]{.kicker}

Which distributions can we sample from? `dir` lists the names in
`jax.random`; names starting with `_` are internals:

@lookup-api-functions-and-classes-in-a-module

- **Samplers** such as `bernoulli`, `beta`, `gamma`; each takes a key.
- `PRNGKey` makes a key; the book uses its successor `jax.random.key`,
  later in the alphabet.
- In a notebook: `jax.random.` then Tab, filtered as you type.
:::

::: {.slide title="dir Lists the Samplers in mxnet.np.random" only="mxnet"}
[Discover]{.kicker}

Which distributions can we sample from? `dir` lists the names in
`mnp.random`; names starting with `_` are internals:

@lookup-api-functions-and-classes-in-a-module

- Classical distributions: `beta`, `gamma`, `multinomial`, `normal`.
- Next to them, the NumPy conveniences `rand`, `randint`, `randn`.
- In a notebook: `mnp.random.` then Tab, filtered as you type.
:::

::: {.slide title="Three Conventions Decode a Signature"}
[Inspect]{.kicker}

`help(f)` prints the **docstring**, the documentation written into the
code. Its first line is the **signature**, the arguments `f` accepts:

| written | meaning |
|---|---|
| `scale=1.0` | a default: the argument may be omitted |
| after a lone `*` | keyword only |
| `*size` | collects any number of positional arguments |

::: {.d2l-note}
The signature says which argument receives each value of a call, before
we run it. The docstring says what the argument means.
:::
:::

::: {.slide title="A Narrow Question Needs Only the Lines That Mention It"}
[Inspect]{.kicker}

The docstring is long; our question is one word, rate or scale. Print the
signature and only the matching lines:

@-lookup-api-reading-documentation-1

@lookup-api-reading-documentation-2
:::

::: {.slide title="Gamma(concentration, rate): the Rate Itself" only="pytorch"}
[Inspect · prediction]{.kicker}

| argument | meaning, from the docstring |
|---|---|
| `concentration` | the shape parameter $k$ |
| `rate` | the rate $\beta$, "rate = 1 / scale" |

- The example line `m.sample()` comes from the end of the docstring, where
  `m` is a `Gamma(1, 1)`.
- **Prediction:** `Gamma(2.0, 3.0).sample((10000,))` has mean $2/3$ and
  variance $2/9$. Had `rate` been a scale: mean 6.
:::

::: {.slide title="gamma(shape, alpha, beta): shape Is the Output Shape" only="tensorflow"}
[Inspect · prediction]{.kicker}

| argument | meaning, from the docstring |
|---|---|
| `shape` | the shape of the output, **not** $k$ |
| `alpha` | the shape parameter $k$ |
| `beta` | "the inverse scale": the rate $\beta$ |

- The signature prints as `random_gamma`, the function's own name;
  `tf.random.gamma` is the public alias.
- **Prediction:** `tf.random.gamma([10000], alpha=2.0, beta=3.0)` has
  mean $2/3$ and variance $2/9$. Had `beta` been a scale: mean 6.
:::

::: {.slide title="gamma(key, a, shape): Neither Rate nor Scale" only="jax"}
[Inspect · prediction]{.kicker}

| argument | meaning, from the docstring |
|---|---|
| `key` | the random key every JAX sampler takes |
| `a` | the shape parameter $k$ |
| `shape` | the shape of the output |

- "Unit scale/rate parameter": the samples have rate 1. "Dividing the
  sample output by the rate" gives rate $\beta$.
- **Prediction:** `gamma(key, 2.0, (10000,)) / 3` has mean $2/3$ and
  variance $2/9$. Undivided: mean 2.
:::

::: {.slide title="gamma(shape, scale, size): shape Is k, and the Second Argument Is a Scale" only="mxnet"}
[Inspect · prediction]{.kicker}

| argument | meaning, from the docstring |
|---|---|
| `shape` | the shape parameter $k$, **not** the output shape |
| `scale` | the scale $1/\beta$ |
| `size` | the shape of the output |

- **Prediction:** `gamma(2, 1 / 3, size=10000)` has mean $2/3$ and
  variance $2/9$. `gamma(2, 3)` would have mean 6.
:::

::: {.slide title="Check: Three Samplers, Three Conventions"}
[Check]{.kicker}

Three libraries document a gamma sampler as follows. For $k = 2$,
$\beta = 3$, what does each call draw?

1. `gamma(k, rate)`, called as `gamma(2, 3)`
2. `gamma(k, scale)`, called as `gamma(2, 3)`
3. `gamma(k)` with rate 1, called as `gamma(2) * 3`

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) mean $2/3$, what we want. (2) mean $6$: scale 3 is rate
$1/3$. (3) mean $6$ as well: the samples should be **divided** by the rate,
not multiplied. Only the docstring tells (1) from (2); only a run would
catch (3).
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Read and verify]{.dtitle}

[the source · the reference · a test that could fail]{.dsub}
:::
:::

::: {.slide title="? Shows the Docstring, ?? Adds the Source" only="pytorch"}
[Read]{.kicker}

`Gamma?` shows the docstring in a pane; `Gamma??` adds the source, which
answers what a docstring leaves open: how an argument is converted or
checked.

- `torch.distributions.Gamma` is written in Python, so `??` shows how
  `rate` enters the sampler.
- Many core functions are compiled from C++; for those `??` shows the
  docstring only.
- Then the [API reference](https://docs.pytorch.org/docs/stable/index.html)
  has a page per public function; the
  [tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
  show functions in use.

::: {.d2l-note}
Not sure which module holds a function? The reference's search box
searches all of them.
:::
:::

::: {.slide title="? Shows the Docstring, ?? Adds the Source" only="tensorflow"}
[Read]{.kicker}

`tf.random.gamma?` shows the docstring in a pane; `tf.random.gamma??` adds
the source, which answers what a docstring leaves open: how an argument is
converted or checked.

- `tf.random.gamma` is Python source, so `??` shows how `beta` is applied.
- Many core functions are compiled from C++; for those `??` shows the
  docstring only.
- The [API reference](https://www.tensorflow.org/api_docs/python/tf) has a
  page per public function; the [tutorials](https://www.tensorflow.org/tutorials)
  show functions in use.

::: {.d2l-note}
Not sure which module holds a function? The reference's search box
searches all of them.
:::
:::

::: {.slide title="? Shows the Docstring, ?? Adds the Source" only="jax"}
[Read]{.kicker}

`jax.random.gamma?` shows the docstring in a pane; `jax.random.gamma??`
adds the source, which answers what a docstring leaves open: how an
argument is converted or checked.

- `jax.random.gamma` is Python source, so `??` shows it.
- Many core functions are compiled from C++; for those `??` shows the
  docstring only.
- The [API reference](https://docs.jax.dev/en/latest/jax.html) has a page
  per public function; the
  [tutorials](https://docs.jax.dev/en/latest/tutorials.html) show functions
  in use.

::: {.d2l-note}
Not sure which module holds a function? The reference's search box
searches all of them.
:::
:::

::: {.slide title="? Shows the Docstring, ?? Adds the Source" only="mxnet"}
[Read]{.kicker}

`mnp.random.gamma?` shows the docstring in a pane; `mnp.random.gamma??`
adds the source, which answers what a docstring leaves open: how an
argument is converted or checked.

- `mnp.random.gamma` is Python source, so `??` shows it.
- MXNet was retired in 2023. Its site still serves the
  [`mxnet.np` reference](https://mxnet.apache.org/versions/master/api/python/docs/api/np/index.html)
  and [tutorials](https://mxnet.apache.org/versions/master/api/python/docs/tutorials/index.html)
  of version 2.0, which the book uses; the pages of 1.9.1 describe the
  older `mx.nd`.

::: {.d2l-note}
Where the documentation is thin, a small test in a notebook is the more
reliable reference.
:::
:::

::: {.slide title="A Prediction Turns a Run Into a Test"}
[Verify]{.kicker}

Predict first: mean $2/3 \approx 0.667$, variance $2/9 \approx 0.222$. The
mean of 10,000 draws has standard error $\sqrt{(2/9)/10{,}000} \approx
0.005$, so it should land within about $0.01$ of $2/3$:

@lookup-api-verifying-behavior

::: {.d2l-note .rule}
"It ran" proves little. A misread rate would have given mean 6. Only a
result that could have contradicted the prediction is a test.
:::
:::

::: {.slide title="Check: Which Run Could Prove Us Wrong?"}
[Check]{.kicker}

Claim: "this `std` divides the squared deviations by $n - 1$, not by
$n$." Which run could refute it?

1. `std` of 1,000 random normal numbers is close to 1.
2. `std` of $(5, 5, 5)$ is 0.
3. `std` of $(0, 2)$ is 1.41.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Only 3. Mean 1, squared deviations sum to 2: $\sqrt{2/1} = 1.41$
under $n - 1$, $\sqrt{2/2} = 1$ under $n$. In run 1 the two rules differ
by 0.05%, far below the sampling noise; run 2 gives 0 under both.
:::
:::
:::

::: {.slide title="An Assistant's Suggestion Is a Candidate"}
[Assistants]{.kicker}

Asked for gamma samples with rate 3, an assistant proposes a call. Check it
like a name you found yourself:

1. the name exists: `dir`, `hasattr`;
2. the call fits the signature: `help`, `?`;
3. a small run gives the predicted result.

::: {.d2l-note .warn}
A wrong suggestion need not fail: it can name a function that exists, fit
its signature, and run, yet pass the rate where a scale is expected.
:::
:::

::: {.slide title="Check: Which Step Catches It?"}
[Check]{.kicker}

1. The suggested sampler does not exist in your version.
2. The suggestion passes `rate=3` to a sampler whose signature has no
   argument of that name.
3. The suggestion passes 3 as the second positional argument, which the
   sampler reads as a scale.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) Discover: `dir` or `hasattr`. (2) Inspect: the signature
rejects the keyword. (3) Verify: the call runs, and the sample mean is 6,
not $2/3$. Nothing before the run could catch it.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| step | tool | in the example |
|--------|--------------|------------------------------|
| discover | `dir`, Tab | the distributions a library can sample from |
| inspect | `help`, `?` | the signature and the lines that mention rate or scale |
| read | reference pages, `??` | the source answers what the docstring leaves open |
| verify | a predicted small run | 10,000 samples: mean $2/3$ within $0.01$, not 6 |

- A test is a run whose result could have contradicted the prediction.
- An assistant's suggestion goes through the same four steps.

**Next:** a first model, linear regression.
:::
