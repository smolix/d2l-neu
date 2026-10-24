```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Documentation
:label:`sec_lookup_api`

The call `torch.ones(2, 3)` returns a $2 \times 3$ matrix of ones. The call
`tf.ones(2, 3)` returns a vector of two ones of an integer type and raises no
error, while `jax.numpy.ones(2, 3)` and `mxnet.np.ones(2, 3)` raise an
error. The four functions have the same name and the same purpose, yet they
read their arguments differently. No book can record such details for every
function. The main module of PyTorch alone defines about a thousand public
names and `jax.numpy` about four hundred, and new releases add names and
deprecate others. This book explains the functions it uses, but every
project soon needs a function that the book never mentions, or uses a
familiar one in an unfamiliar way.

Instead of a list, we need a procedure that finds out reliably what a
function does. This section describes one with four steps, shown in
:numref:`fig_lookup_loop`. We discover the names that a library offers,
inspect the arguments and the documentation of a candidate, read the
reference pages or the source code when the documentation leaves a question
open, and verify the behavior with a small example whose result we
predicted. All four steps work inside a notebook, and at the end of the
section they explain the four results above.

![Four steps for finding out what a function does: discover the names that a module defines, inspect the signature and docstring of a candidate, read the reference documentation or the source code, and verify the behavior with a small example. A result that contradicts the prediction, or answers a question that the prediction left open, sends us back to read further.](../img/lookup-api-discovery-loop.svg)
:label:`fig_lookup_loop`

We import the framework and the function `pprint`, which prints long lists
compactly.

:begin_tab:`mxnet`
MXNet's array module is imported under the name `mnp`, so that `np` can keep
meaning NumPy.
:end_tab:

```{.python .input #lookup-api-documentation}
%%tab mxnet
from pprint import pprint
from mxnet import np as mnp, npx
npx.set_np()
```

```{.python .input #lookup-api-documentation}
%%tab pytorch
from pprint import pprint
import torch
```

```{.python .input #lookup-api-documentation}
%%tab tensorflow
from pprint import pprint
import tensorflow as tf
```

```{.python .input #lookup-api-documentation}
%%tab jax
from pprint import pprint
import jax
from jax import numpy as jnp
```

## Discovering Names

Suppose that we need random samples from a distribution other than the
normal distribution, and we want to know which distributions the library
can sample from. We know roughly where to look, in the module for random
numbers, but not what its functions are called. The built-in function `dir`
lists the names that a module defines. Names that begin with an underscore
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
`exponential`, and `gamma`, along with `PRNGKey`, an older function that
creates the random key every JAX sampler takes. This book uses its
successor, `jax.random.key`, which comes later in the alphabet.
:end_tab:

A notebook offers the same list while we type: writing the name of a module
followed by a dot and pressing Tab lists the names that the module defines,
narrowed to those that match the letters typed next. When we do not know
which module to look in, the search box of the framework's online reference
searches all of them.

## Reading Documentation

We set the samplers aside, since the first exercise completes the search
for a gamma sampler, and inspect a function that we think we already know:
`ones`, which we have used throughout this chapter. The built-in function
`help` prints the *docstring* of a function, the documentation written into
its code. A docstring says what the function does, what each argument means,
and what the function returns, and it often ends with examples:

```{.python .input #lookup-api-specific-functions-and-classes-1}
%%tab mxnet
help(mnp.ones)
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
help(jnp.ones)
```

The first lines of the output show the function's *signature*, the list of
arguments that it accepts. Three conventions of Python help to read it. An
argument written as `name=value` has a default value and may be omitted.
Arguments that follow a lone `*` can be passed only by keyword. An argument
written as `*name` collects any number of positional arguments.

The signatures of `ones` in the four frameworks begin as follows:

| Framework | Signature |
|---|---|
| PyTorch | `ones(*size, *, out=None, dtype=None, ...)` |
| TensorFlow | `ones(shape, dtype=tf.float32, name=None, layout=None)` |
| JAX | `ones(shape, dtype=None, *, device=None, ...)` |
| MXNet | `ones(shape, dtype=None, order='C', device=None)` |

PyTorch collects all positional arguments into the shape, and it accepts the
data type only by keyword. The other three frameworks take the shape as a
single argument and the data type as the second positional argument, so they
read 2 as the shape and 3 as the data type. We therefore predict a
$2 \times 3$ matrix from `ones(2, 3)` in PyTorch, and a call that takes the
integer 3 for a data type in the other three frameworks.

In a notebook, two shortcuts show the same documentation. Appending `?` to
a name, as in `ones?`, shows the docstring in a separate pane, and appending
`??` also shows the source code of the function. The source answers
questions that a docstring leaves open, such as how an argument is converted
or checked. Only functions written in Python have source code to show. Many
core functions, `torch.ones` among them, are compiled from C++, and for
those `??` shows the docstring alone.

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
wrong. A run that merely completes without an error proves little. We
therefore test the prediction about `ones(2, 3)` by printing the shape and
the data type of the result, or the error if the call fails:

```{.python .input #lookup-api-specific-functions-and-classes-2}
%%tab mxnet
try:
    x = mnp.ones(2, 3)
    print('shape:', x.shape, 'dtype:', x.dtype)
except TypeError as e:
    print('TypeError:', e)
print('ones((2, 3)) has shape', mnp.ones((2, 3)).shape)
```

```{.python .input #lookup-api-specific-functions-and-classes-2}
%%tab pytorch
try:
    x = torch.ones(2, 3)
    print('shape:', x.shape, 'dtype:', x.dtype)
except TypeError as e:
    print('TypeError:', e)
print('ones((2, 3)) has shape', torch.ones((2, 3)).shape)
```

```{.python .input #lookup-api-specific-functions-and-classes-2}
%%tab tensorflow
try:
    x = tf.ones(2, 3)
    print('shape:', x.shape, 'dtype:', x.dtype)
except TypeError as e:
    print('TypeError:', e)
print('ones((2, 3)) has shape', tf.ones((2, 3)).shape)
```

```{.python .input #lookup-api-specific-functions-and-classes-2}
%%tab jax
try:
    x = jnp.ones(2, 3)
    print('shape:', x.shape, 'dtype:', x.dtype)
except TypeError as e:
    print('TypeError:', e)
print('ones((2, 3)) has shape', jnp.ones((2, 3)).shape)
```

The predictions hold in all four frameworks. PyTorch returns a $2 \times 3$
matrix of `float32` ones. JAX and MXNet cannot interpret the integer 3 as a
data type, so the call fails; the message of JAX even names the likely
intent. TensorFlow also takes 3 for the data type, but our prediction left
open which type the integer 3 names: the result is a vector of two `int32`
ones, and no error is raised. The source answers this question. The first
statement in the body of `tf.ones`, visible with `tf.ones??`, passes `dtype`
to `tf.as_dtype`. This function accepts not only type objects but also the
integers that TensorFlow uses internally to number its types, and
`tf.as_dtype(3)` is `tf.int32`. A call written with PyTorch in mind thus
runs in TensorFlow and computes something else.

In all four frameworks, `ones((2, 3))`, with the shape passed as one tuple,
returns a $2 \times 3$ matrix, as the last line of the output shows. More
generally, when a result contradicts the prediction, or answers a question
that the prediction left open, we go around the loop again: we read further,
in the reference documentation or the source, and test the new prediction.

Coding assistants answer many such questions directly. Asked for a
$2 \times 3$ tensor of ones, an assistant proposes a call, and the call
usually looks plausible. Such a suggestion is a candidate, and it goes
through the same steps as a name we found ourselves: we check that the name
exists, for example with `dir` or `hasattr`, read its signature, predict the
result, and run a small example. The last step matters most, because a
wrong suggestion need not fail: `tf.ones(2, 3)` names a function that
exists, fits its signature, and runs, yet computes something else.

The four steps let us find out what a function does without leaving the
notebook, and the last step makes the answer reliable, whether it came from
memory, from the documentation, or from an assistant. They complete the
tools of this chapter: tensors, tables, linear algebra, derivatives,
probability, and a way to find what the book leaves out.
:numref:`chap_regression` uses these tools to build and train a first model.

## Summary

No text can record the details of every function in a deep learning
framework, and functions with the same name differ between frameworks:
`ones(2, 3)` returns a $2 \times 3$ matrix in PyTorch, a vector of two
integer ones in TensorFlow, and an error in JAX and MXNet. Four steps settle
such questions from within a notebook. `dir` and Tab completion discover the
names that a module defines, and `help` and `?` show the signature and the
docstring of a candidate. The reference documentation and `??` supply the
details that the docstring omits, and a small example whose result we
predicted verifies our reading. Suggestions from coding assistants are
candidates that go through the same steps.

## Exercises

:begin_tab:`mxnet`
1. [code] **Scale versus rate.** The gamma distribution with shape
   $k = 2$ and rate $\beta = 3$ has mean $k/\beta = 2/3$ and variance
   $k/\beta^2 = 2/9$. Use `dir` on `mnp.random` to find its sampler and
   read the signature with `help` (or `?`) to find out how the sampler is
   parameterized. Draw 10,000 samples with $k = 2$ and $\beta = 3$ and
   confirm the mean and variance from the samples.
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
       so that their lengths increase, an operation no library function
       provides. State which of the three checks, if any, each named
       function fails. Then build the operation from functions that pass
       all three checks and test it on an array of shape $(4, 2, 3)$.
:end_tab:

4. [code] **Reading the source.** The plotting function `d2l.plot`,
   defined in :numref:`sec_calculus`, accepts its data in several forms:
   a single tensor, a tensor of $x$-values with one tensor of $y$-values,
   or a tensor of $x$-values with several curves. Import `d2l` as in that
   section, read the source of `d2l.plot` with `d2l.plot??` or your
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

::: {.slide title="Same Name, Different Behavior: ones(2, 3)"}
[The problem]{.kicker}

| call | result |
|---|---|
| `torch.ones(2, 3)` | shape (2, 3), `float32` |
| `tf.ones(2, 3)` | shape (2,), `int32`, no error |
| `jax.numpy.ones(2, 3)` | `TypeError` |
| `mxnet.np.ones(2, 3)` | `TypeError` |

- `torch` alone defines about 1,000 public names, `jax.numpy` about 400.
- Releases add names and deprecate others.

::: {.d2l-note}
Memorizing fails. We need a procedure that finds out what a call does.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- list the names that a module defines;
- read a signature: defaults, keyword-only arguments, `*args`;
- find the docstring, the reference page, and the source;
- design a small test that could prove your reading wrong;
- check a coding assistant's suggestion.

**Running example:** `ones` in four frameworks.
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
| Probability and Statistics | Bayes' theorem, standard error |
| **Documentation** | **`help`, small tests** |
:::

::: {.col .narrow}
- **Uses:** `ones` and the random samplers of the earlier decks.
- **Skips:** debuggers and profilers.
:::
:::
:::

::: {.slide title="Four Steps Answer Most API Questions"}
[The procedure]{.kicker}

@fig:lookup-api-discovery-loop

Discover the names, inspect a candidate, read further, and verify with a
small run whose result you predicted.
:::

::: {.slide title="dir Lists What a Module Offers"}
[Discover]{.kicker}

Which distributions can the library sample from? `dir` lists the
candidates:

@lookup-api-functions-and-classes-in-a-module

- The list mixes distributions with helpers: keys, generators, transforms.
- Names starting with `_` are internals; they are dropped here.
- In a notebook: module name, a dot, then Tab, filtered as you type.
:::

::: {.slide title="help Prints the Signature and the Docstring"}
[Inspect]{.kicker}

@-lookup-api-specific-functions-and-classes-1

| library | signature |
|---|---|
| PyTorch | `ones(*size, *, dtype=None, ...)` |
| TensorFlow | `ones(shape, dtype=tf.float32, ...)` |
| JAX, MXNet | `ones(shape, dtype=None, ...)` |

`x=v`: default value. After a lone `*`: keyword only. `*size`: any number
of positional arguments.
:::

::: {.slide title="Check: Which Calls Make a 2 × 3 Matrix Everywhere?"}
[Check]{.kicker}

Use the signatures to predict, for each framework:

1. `ones(2, 3)`
2. `ones((2, 3))`
3. `ones(shape=(2, 3))`
4. `ones([2, 3])`

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) PyTorch only. (2) All four. (3) All but PyTorch, whose
signature has no argument `shape`. (4) All four.
:::
:::
:::

::: {.slide title="When the Docstring Leaves a Question Open, Read Further"}
[Read]{.kicker}

- `ones?` shows the docstring in a pane; `ones??` adds the source code.
- The source answers what a docstring leaves open: how an argument is
  converted or checked.
- Only functions written in Python have source to show: `torch.ones` is
  compiled C++, so `??` shows its docstring alone.

::: {.d2l-note}
Also read the reference page of the function, and the tutorials.
:::
:::

::: {.slide title="A Prediction Turns a Run Into a Test"}
[Verify]{.kicker}

Predict first, then print shape and type, or the error:

@lookup-api-specific-functions-and-classes-2

::: {.d2l-note .rule}
"It ran" proves little. Only a result that could have contradicted the
prediction is a test.
:::
:::

::: {.slide title="tf.ones(2, 3) Runs, and the Source Explains Why" only="tensorflow"}
[Read again]{.kicker}

The first statement of `tf.ones`, seen with `tf.ones??`, is

`dtype = dtypes.as_dtype(dtype).base_dtype`

- `tf.as_dtype` also accepts TensorFlow's internal type numbers.
- `tf.as_dtype(3)` is `tf.int32`.
- Fix: `tf.ones((2, 3))`.
:::

::: {.slide title="An Assistant's Suggestion Is a Candidate"}
[Assistants]{.kicker}

Check every suggested call like a name you found yourself:

1. the name exists: `dir`, `hasattr`;
2. the call fits the signature: `help`, `?`;
3. a small run gives the predicted result.

::: {.d2l-note .warn}
A wrong suggestion need not fail: a call can exist, fit the signature, and
run, yet compute something else.
:::
:::

::: {.slide title="Check: Which Step Catches It?"}
[Check]{.kicker}

1. The suggested function does not exist in your version.
2. The suggestion passes `shape=(2, 3)` to `torch.ones`.
3. `tf.ones(2, 3)` was meant as a $2 \times 3$ matrix.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) Discover: `dir` or `hasattr`. (2) Inspect: the signature
has no argument named `shape`. (3) Only verify: the call runs and returns
two `int32` ones.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| step | tool | in the example |
|---|---|---|
| discover | `dir`, Tab | the distributions a library can sample from |
| inspect | `help`, `?` | `ones(*size, ...)` against `ones(shape, dtype, ...)` |
| read | reference pages, `??` | `tf.ones` passes 3 to `tf.as_dtype` |
| verify | a predicted small run | `ones(2, 3)`: (2, 3), (2,) `int32`, or an error |

**Next:** a first model, linear regression.
:::
