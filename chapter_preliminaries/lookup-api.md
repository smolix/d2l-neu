```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Documentation
:label:`sec_lookup_api`

Throughout this chapter, we have explained every framework function before calling
it. That convenience cannot last forever: in any real project you will quickly
need a function this book never mentions, or need to pass an unfamiliar option to
a function you already know. Nor could any textbook possibly document an entire
framework—the top-level modules of PyTorch, TensorFlow, JAX, and MXNet each expose
hundreds to thousands of public names, with new APIs arriving and old ones being
deprecated in every release. Rather than trying to memorize an ever-shifting API,
what we can learn once and for all is a reliable habit for finding out *ourselves*
what any function actually does.

*Reliably* is the key word here, because a function call can look completely
innocent and run without raising an error while silently doing something other
than what we intended. Suppose we want to draw samples from a gamma distribution
(a continuous distribution over positive numbers often used to model waiting
times) with shape parameter $k = 2$ and rate $\beta = 3$, which has theoretical
mean $k / \beta = 2/3$ and variance $k / \beta^2 = 2/9$. Scientific libraries
notoriously disagree on how they parameterize the gamma distribution: some take
the rate $\beta$, others take the scale $1 / \beta$, and still others fix the rate
to $1$ and expect you to rescale the samples yourself. Passing the numbers `2` and
`3` succeeds under both conventions, yet yields samples with mean $2/3$ under one
and mean $2 \cdot 3 = 6$ under the other! Nothing in the call syntax itself warns
us which convention is in force; only the function's signature, documentation, and
a quick sanity check do.

Instead of listing APIs, this section walks through a four-step workflow
(:numref:`fig_lookup_loop`) that works right inside a notebook:

1. **Discover** what names a module exposes;
2. **Inspect** a candidate's call signature and docstring;
3. **Read** the official API reference or underlying source code when the
   docstring leaves a detail ambiguous;
4. **Verify** your understanding on a small test case whose outcome you predicted
   before running the code.

By the end of the section, these four steps will have given us gamma samples with
the exact mean and variance we wanted.

![Four steps for finding out what a function does: discover the names that a module defines, inspect the signature and docstring of a candidate, read the reference documentation or the source code, and verify the behavior with a small example. A result that contradicts the prediction, or answers a question that the prediction left open, sends us back to read further.](../img/lookup-api-discovery-loop.svg)
:label:`fig_lookup_loop`

Alongside our deep learning framework, we import three handy utilities from
Python's standard library: `pprint` for printing long lists compactly, `inspect`
for extracting function signatures and docstrings, and `textwrap` for wrapping
long signature lines cleanly.

:begin_tab:`mxnet`
As throughout this book, we import MXNet's NumPy-compatible array module under the
alias `mnp`, reserving `np` for standard NumPy.
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

Our first question is simple: does the library provide a gamma sampler at all, and
what is it called? We know roughly where to look—the random-number or
distributions module—even if we do not know the exact name. Python's built-in
`dir` function lists every attribute defined in a module. By convention, names
beginning and ending with `__` are special Python objects and names starting with
a single `_` are private internals, so we filter those out and inspect the first
twenty public names:

:begin_tab:`pytorch`
PyTorch organizes its parametric probability distributions inside the
`torch.distributions` module.
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
Almost every entry here is a random sampler: classical distributions such as
`beta`, `gamma`, `multinomial`, and `normal` sit alongside NumPy-style shorthand
helpers like `rand`, `randint`, and `randn`.
:end_tab:

:begin_tab:`pytorch`
Here the public names are distribution *classes*—such as `Bernoulli`,
`Categorical`, and `Gamma`—which you instantiate with their distribution
parameters before calling `.sample()`. Names ending in `Transform` represent
invertible mappings that `TransformedDistribution` can apply to construct new
distributions from existing ones.
:end_tab:

:begin_tab:`tensorflow`
Here we find samplers such as `gamma`, `normal`, and `poisson` alongside
`Generator` and `set_seed`, which manage random-number generator state.
:end_tab:

:begin_tab:`jax`
Here we find functional samplers such as `bernoulli`, `beta`, `cauchy`,
`exponential`, and `gamma`, alongside `Key` (the random key type) and `PRNGKey`
(an older constructor for the explicit keys JAX samplers require; this book uses
its newer replacement `jax.random.key`, which appears further down the alphabet).
:end_tab:

In an interactive notebook or IDE, Tab completion provides the same discovery
mechanism on the fly: typing a module name followed by a dot and pressing `Tab`
brings up all public attributes, narrowing the list as you type. And when you are
not even sure which module to look in, the search bar on the framework's online
API reference searches across the entire library at once.

## Reading Documentation

Our listing confirms that the framework has a gamma distribution. To learn how to
invoke it properly, Python's built-in `help` function displays its *docstring*—the
documentation embedded directly in the code. A well-written docstring explains
what the function computes, what each parameter represents, what it returns, and
often includes short usage examples.

At the top of the docstring sits the function's *signature*, showing the exact
arguments it accepts. Just three Python signature conventions cover almost every
library call you will ever read:

* `name=value` marks an optional argument with a default value;
* arguments appearing after a bare `*` are *keyword-only* and must be passed by
  name;
* `*name` collects any number of positional arguments into a tuple (our gamma
  sampler does not use this form, though you will encounter it in the exercises).

While reading the full docstring once with `help` is always good practice,
docstrings can span several screens, and right now we have one laser-focused
question: *does the sampler expect a rate or a scale?* Below, we write a tiny
helper that prints the formatted signature alongside only those docstring lines
mentioning `rate` or `scale`. (Individual lines may wrap mid-sentence, of course;
the complete text is always one `help(...)` call away.)

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

Equipped with the signature and these lines, we can form a concrete prediction
*before* running any sampling code:

:begin_tab:`mxnet`
Look closely at `gamma(shape, scale=1.0, size=None, ...)`: there is a subtle trap
in the naming! Here `shape` refers to the distribution's shape parameter $k$—not
the shape of the output array, which is controlled by `size`. Meanwhile, the
second argument `scale` is the scale $1 / \beta$. We therefore predict that
`mnp.random.gamma(2, 1 / 3, size=10000)` will yield $10{,}000$ samples with mean
$2/3$ and variance $2/9$ (whereas calling `mnp.random.gamma(2, 3)` would
mistakenly produce mean $6$).
:end_tab:

:begin_tab:`pytorch`
`Gamma` is a class taking `concentration` (the shape parameter $k$) and `rate`
(the rate $\beta = 1 / \textrm{scale}$ directly). One of the matching lines comes
from the docstring's own usage example (`m = Gamma(1, 1)`), confirming the
convention. We therefore predict that sampling from `Gamma(2.0, 3.0)` will give
mean $2/3$ and variance $2/9$.
:end_tab:

:begin_tab:`tensorflow`
Notice the signature `(shape, alpha, beta=None, ...)` (printed under its internal
name `random_gamma`, of which `tf.random.gamma` is the public alias): here `shape`
*is* the output tensor shape, while the distribution's shape parameter $k$ is
named `alpha`! The docstring specifies that `beta` is the inverse scale—that is,
the rate $\beta$. We therefore predict that
`tf.random.gamma([10000], alpha=2.0, beta=3.0)` will yield $10{,}000$ samples with
mean $2/3$ and variance $2/9$.
:end_tab:

:begin_tab:`jax`
Look at the signature `gamma(key, a, shape=None, ...)` (shown with its type
annotations): it accepts *neither* a rate nor a scale! As the docstring explains,
`jax.random.gamma` always draws from a unit-rate ($\beta = 1$) gamma distribution
with shape parameter `a`, and instructs us to divide the output by the desired
rate. We therefore predict that `jax.random.gamma(key, 2.0, (10000,)) / 3` will
yield samples with mean $2/3$ and variance $2/9$.
:end_tab:

In Jupyter notebooks, two quick shortcuts make inspecting documentation
effortless: appending a single question mark (e.g., `list?`) opens the docstring
in a side pane, while appending two question marks (`list??`) displays the
function's actual Python source code as well. Reading the source resolves subtle
questions a docstring may omit—such as how inputs are cast or validated—and
studying well-crafted library code is one of the best ways to become a stronger
engineer as well as a better scientist. (Note, however, that performance-critical
primitives implemented in C++ or CUDA have no Python body to display, so `??`
falls back to showing the docstring alone.)

When in-notebook docstrings are not enough, the framework's official online
documentation provides dedicated reference pages for every public API alongside
in-depth tutorials:

:begin_tab:`mxnet`
Although the Apache Software Foundation retired MXNet in 2023, the project
website still hosts the documentation for its development branch, MXNet 2.0—the
NumPy-compatible interface used in this book—including the
[`mxnet.np` API reference](https://mxnet.apache.org/versions/master/api/python/docs/api/np/index.html)
and the
[tutorials](https://mxnet.apache.org/versions/master/api/python/docs/tutorials/index.html).
(Be aware that the docs for version 1.9.1, the last formal release, cover the
older `mx.nd` API instead.) Whenever online documentation is sparse, the
interactive notebook workflow in this section is the most dependable guide.
:end_tab:

:begin_tab:`pytorch`
For PyTorch, consult the official
[API reference](https://docs.pytorch.org/docs/stable/index.html) and
[tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
:end_tab:

:begin_tab:`tensorflow`
For TensorFlow, consult the official
[API reference](https://www.tensorflow.org/api_docs/python/tf) and
[tutorials](https://www.tensorflow.org/tutorials).
:end_tab:

:begin_tab:`jax`
For JAX, consult the official
[API reference](https://docs.jax.dev/en/latest/jax.html) and
[tutorials](https://docs.jax.dev/en/latest/tutorials.html).
:end_tab:

## Verifying Behavior

Even careful readers misread docstrings from time to time—and occasionally a
docstring itself has drifted out of sync with the code. The fastest way to settle
any lingering doubt is to run a minimal test whose numerical outcome you predicted
*in advance*. Notice why a prediction is essential: code that merely runs without
crashing proves almost nothing, whereas a test with a falsifiable prediction can
actually catch a mistake! Here, confusing rate with scale would shift the mean
from $2/3 \approx 0.667$ all the way to $6$. Drawing $10{,}000$ independent
samples gives a standard error on the sample mean of
$\sqrt{(2/9) / 10{,}000} \approx 0.005$ (:numref:`sec_prob`), so our empirical
mean should land within roughly $0.01$ of $2/3$ and the variance close to
$2/9 \approx 0.222$:

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

The numbers confirm our prediction: the sample mean sits right near $2/3$ and the
sample variance near $2/9$, ruling out the scale interpretation completely. Whenever
a test contradicts your prediction—or exposes an edge case you had not considered—loop
back to the documentation or source code, refine your hypothesis, and test again.

This same four-step habit is doubly valuable when working with AI coding
assistants. Asked how to draw gamma samples with a given rate, an assistant will
readily suggest a plausible-looking call—but treat every suggestion as an
unverified *candidate*, not ground truth. Check that the function name actually
exists (with `dir` or `hasattr`), inspect its signature, predict what it should
return, and verify it on a tiny input. That final verification step is critical:
a subtly wrong suggestion can invoke a real function with valid argument types
and still silently compute the wrong distribution.

With this workflow in place, we have assembled our complete preliminary toolkit:
tensors, tabular preprocessing, linear algebra, calculus, automatic
differentiation, probability, and a reliable method for navigating framework APIs.
In :numref:`chap_regression`, we put these tools to work to build and train our
first real deep learning models.

## Summary

* No book can cover every function in a rapidly evolving deep learning framework,
  and valid-looking calls can silently follow different conventions across
  libraries (such as rate vs. scale for a gamma distribution).
* A four-step loop settles API questions reliably inside a notebook: (i) *discover*
  public names with `dir` or Tab completion, (ii) *inspect* signatures and
  docstrings with `help` or `?`, (iii) *read* the online API reference or Python
  source code (`??`) when details remain unclear, and (iv) *verify* your reading
  on a small test whose outcome you predicted beforehand.
* Treat code snippets proposed by coding assistants as candidates that must pass
  the exact same discovery, inspection, and verification checks.

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
