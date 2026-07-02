```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Probability and Statistics
:label:`sec_prob`

One way or another,
machine learning is all about uncertainty.
In supervised learning, we want to predict
something unknown (the *target*)
given something known (the *features*).
Depending on our objective,
we might attempt to predict
the most likely value of the target.
Or we might predict the value with the smallest
expected distance from the target.
And sometimes we wish not only
to predict a specific value
but to *quantify our uncertainty*.
For example, given some features
describing a patient,
we might want to know *how likely* they are
to suffer a heart attack in the next year.
In unsupervised learning,
we often care about uncertainty.
To determine whether a set of measurements are anomalous,
it helps to know how likely one is
to observe values in a population of interest.
Furthermore, in reinforcement learning,
we wish to develop agents
that act intelligently in various environments.
This requires reasoning about
how an environment might be expected to change
and what rewards one might expect to encounter
in response to each of the available actions.

*Probability* is the mathematical field
concerned with reasoning under uncertainty.
Given a probabilistic model of some process,
we can reason about the likelihood of various events.
The use of probabilities to describe
the frequencies of repeatable events
(like coin tosses)
is fairly uncontroversial.
In fact, *frequentist* scholars adhere
to an interpretation of probability
that applies *only* to such repeatable events.
By contrast *Bayesian* scholars
use the language of probability more broadly
to formalize reasoning under uncertainty.
Bayesian probability is characterized
by two unique features:
(i) assigning degrees of belief
to non-repeatable events,
e.g., what is the *probability*
that a dam will collapse?;
and (ii) subjectivity. While Bayesian
probability provides unambiguous rules
for how one should update their beliefs
in light of new evidence,
it allows for different individuals
to start off with different *prior* beliefs.
*Statistics* helps us to reason backwards,
starting off with collection and organization of data
and backing out to what inferences
we might draw about the process
that generated the data.
Whenever we analyze a dataset, hunting for patterns
that we hope might characterize a broader population,
we are employing statistical thinking.
Many courses, majors, theses, careers, departments,
companies, and institutions have been devoted
to the study of probability and statistics.
While this section only scratches the surface,
we will provide the foundation
that you need to begin building models.

```{.python .input #probability-probability-and-statistics}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
import random
npx.set_np()
```

```{.python .input #probability-probability-and-statistics}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```

```{.python .input #probability-probability-and-statistics}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

```{.python .input #probability-probability-and-statistics}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import random
import jax
from jax import numpy as jnp
import numpy as np
```

## A Simple Example: Tossing Coins

Imagine that we plan to toss a coin
and want to quantify how likely
we are to see heads (vs. tails).
If the coin is *fair*,
then both outcomes
(heads and tails),
are equally likely.
Moreover if we plan to toss the coin $n$ times
then the fraction of heads
that we *expect* to see
should exactly match
the *expected* fraction of tails.
One intuitive way to see this
is by symmetry:
for every possible outcome
with $n_\textrm{h}$ heads and $n_\textrm{t} = (n - n_\textrm{h})$ tails,
there is an equally likely outcome
with $n_\textrm{t}$ heads and $n_\textrm{h}$ tails.
Note that this is only possible
if on average we expect to see
$1/2$ of tosses come up heads
and $1/2$ come up tails.
Of course, if you conduct this experiment
many times with $n=1000000$ tosses each,
you might never see a trial
where $n_\textrm{h} = n_\textrm{t}$ exactly.


Formally, the quantity $1/2$ is called a *probability*
and here it captures the certainty with which
any given toss will come up heads.
Probabilities assign scores between $0$ and $1$
to outcomes of interest, called *events*.
Here the event of interest is $\textrm{heads}$
and we denote the corresponding probability $P(\textrm{heads})$.
A probability of $1$ indicates absolute certainty
(imagine a trick coin where both sides were heads)
and a probability of $0$ indicates impossibility
(e.g., if both sides were tails).
The frequencies $n_\textrm{h}/n$ and $n_\textrm{t}/n$ are not probabilities
but rather *statistics*.
Probabilities are *theoretical* quantities
that underlie the data generating process.
Here, the probability $1/2$
is a property of the coin itself.
By contrast, statistics are *empirical* quantities
that are computed as functions of the observed data.
Our interests in probabilistic and statistical quantities
are inextricably intertwined.
We often design special statistics called *estimators*
that, given a dataset, produce *estimates*
of model parameters such as probabilities.
Moreover, when those estimators satisfy
a nice property called *consistency*,
our estimates will converge
to the corresponding probability.
In turn, these inferred probabilities
tell us about the likely statistical properties
of data from the same population
that we might encounter in the future.

Suppose that we stumbled upon a real coin
for which we did not know
the true $P(\textrm{heads})$.
To investigate this quantity
with statistical methods,
we need to (i) collect some data;
and (ii) design an estimator.
Data acquisition here is easy;
we can toss the coin many times
and record all the outcomes.
Formally, drawing realizations
from some underlying random process
is called *sampling*.
As you might have guessed,
one natural estimator
is the ratio of
the number of observed *heads*
to the total number of tosses.

Now, suppose that the coin was in fact fair,
i.e., $P(\textrm{heads}) = 0.5$.
To simulate tosses of a fair coin,
we can invoke any random number generator.
There are some easy ways to draw samples
of an event with probability $0.5$.
For example Python's `random.random`
yields numbers in the interval $[0,1]$
where the probability of lying
in any sub-interval $[a, b] \subset [0,1]$
is equal to $b-a$.
Thus we can get out `0` and `1` with probability `0.5` each
by testing whether the returned float number is greater than `0.5`:

```{.python .input #probability-a-simple-example-tossing-coins-1}
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

More generally, we can simulate multiple draws
from any variable with a finite number
of possible outcomes
(like the toss of a coin or roll of a die)
by calling the multinomial function,
setting the first argument
to the number of draws
and the second as a list of probabilities
associated with each of the possible outcomes.
To simulate ten tosses of a fair coin,
we assign probability vector `[0.5, 0.5]`,
interpreting index 0 as heads
and index 1 as tails.
The function returns a vector
with length equal to the number
of possible outcomes (here, 2),
where the first component tells us
the number of occurrences of heads
and the second component tells us
the number of occurrences of tails.

```{.python .input #probability-a-simple-example-tossing-coins-2}
%%tab mxnet
fair_probs = [0.5, 0.5]
multinomial(100, fair_probs)
```

```{.python .input #probability-a-simple-example-tossing-coins-2}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input #probability-a-simple-example-tossing-coins-2}
%%tab tensorflow
fair_probs = tf.ones(2) / 2
tfd.Multinomial(100, fair_probs).sample()
```

```{.python .input #probability-a-simple-example-tossing-coins-2}
%%tab jax
# jax.random has no multinomial sampler, so we draw categorical samples
# with an explicit key and bin them with jnp.bincount
fair_probs = jnp.array([0.5, 0.5])
key = d2l.get_key()
counts = jnp.bincount(jax.random.categorical(key, jnp.log(fair_probs),
                                              shape=(100,)), length=2)
counts
```

Each time you run this sampling process,
you will receive a new random value
that may differ from the previous outcome.
If instead you need *reproducible* draws---to debug,
or so that a figure looks the same on every run---seed
the generator first (`torch.manual_seed`, `np.random.seed`, or,
in JAX, by passing an explicit `key`, as introduced
in :numref:`sec_ndarray`).
Dividing by the number of tosses
gives us the *frequency*
of each outcome in our data.
Note that these frequencies,
just like the probabilities
that they are intended
to estimate, sum to $1$.

```{.python .input #probability-a-simple-example-tossing-coins-3}
%%tab mxnet
multinomial(100, fair_probs) / 100
```

```{.python .input #probability-a-simple-example-tossing-coins-3}
%%tab pytorch
Multinomial(100, fair_probs).sample() / 100
```

```{.python .input #probability-a-simple-example-tossing-coins-3}
%%tab tensorflow
tfd.Multinomial(100, fair_probs).sample() / 100
```

```{.python .input #probability-a-simple-example-tossing-coins-3}
%%tab jax
key = d2l.get_key()
counts = jnp.bincount(jax.random.categorical(key, jnp.log(fair_probs),
                                              shape=(100,)), length=2)
counts / 100
```

Here, even though our simulated coin is fair
(we ourselves set the probabilities `[0.5, 0.5]`),
the counts of heads and tails may not be identical.
That is because we only drew a relatively small number of samples.
If we did not implement the simulation ourselves,
and only saw the outcome,
how would we know if the coin were slightly unfair
or if the possible deviation from $1/2$ was
just an artifact of the small sample size?
Let's see what happens when we simulate 10,000 tosses.

```{.python .input #probability-a-simple-example-tossing-coins-4}
%%tab mxnet
counts = multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

```{.python .input #probability-a-simple-example-tossing-coins-4}
%%tab pytorch
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input #probability-a-simple-example-tossing-coins-4}
%%tab tensorflow
counts = tfd.Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input #probability-a-simple-example-tossing-coins-4}
%%tab jax
key = d2l.get_key()
counts = jnp.bincount(jax.random.categorical(key, jnp.log(fair_probs),
                                              shape=(10000,)),
                      length=2).astype(jnp.float32)
counts / 10000
```

In general, for averages of repeated events (like coin tosses),
as the number of repetitions grows,
our estimates are guaranteed to converge
to the true underlying probabilities.
The mathematical formulation of this phenomenon
is called the *law of large numbers*
and the *central limit theorem*
(developed in :numref:`sec_mdl-distributions`)
tells us that in many situations,
as the sample size $n$ grows,
these errors should go down
at a rate of $(1/\sqrt{n})$ :cite:`Wasserman.2013`.

Where does the rate come from? Each toss is a random variable
taking value $1$ (heads) with probability $p$ and $0$ otherwise,
so its variance is $E[X_i^2] - E[X_i]^2 = p - p^2 = p(1-p)$
(we formally introduce variances later in this section).
Our estimate averages $n$ independent tosses,
and variances of independent variables add,
while scaling a variable by $1/n$ scales its variance by $1/n^2$; hence

$$\textrm{Var}[\hat{p}] = \frac{p(1-p)}{n},$$

and the typical error (the standard deviation)
is $\sqrt{p(1-p)/n}$, which shrinks as $1/\sqrt{n}$.
Rather than taking this on faith, we can watch the law emerge:
below we estimate $p$ from 1000 independent batches
of $n$ tosses each, for growing $n$,
and plot the standard deviation of the estimates on log--log axes.
It hugs the predicted $0.5/\sqrt{n}$ line,
a straight line of slope $-\frac{1}{2}$.

```{.python .input #probability-a-simple-example-tossing-coins-6}
%%tab mxnet
ns = [10, 100, 1000, 10000]
stds = [float((np.random.uniform(size=(1000, n)) < 0.5)
              .astype(np.float32).mean(axis=1).std()) for n in ns]
d2l.plot(ns, [stds, [0.5 / n ** 0.5 for n in ns]], 'n', 'std of estimate',
         legend=['empirical', '0.5/sqrt(n)'], xscale='log', yscale='log')
```

```{.python .input #probability-a-simple-example-tossing-coins-6}
%%tab pytorch
ns = [10, 100, 1000, 10000]
stds = [float((torch.rand(1000, n) < 0.5).float().mean(dim=1).std())
        for n in ns]
d2l.plot(ns, [stds, [0.5 / n ** 0.5 for n in ns]], 'n', 'std of estimate',
         legend=['empirical', '0.5/sqrt(n)'], xscale='log', yscale='log')
```

```{.python .input #probability-a-simple-example-tossing-coins-6}
%%tab tensorflow
ns = [10, 100, 1000, 10000]
stds = [float(tf.math.reduce_std(tf.reduce_mean(
    tf.cast(tf.random.uniform((1000, n)) < 0.5, tf.float32), axis=1)))
    for n in ns]
d2l.plot(ns, [stds, [0.5 / n ** 0.5 for n in ns]], 'n', 'std of estimate',
         legend=['empirical', '0.5/sqrt(n)'], xscale='log', yscale='log')
```

```{.python .input #probability-a-simple-example-tossing-coins-6}
%%tab jax
ns = [10, 100, 1000, 10000]
stds = [float((jax.random.uniform(d2l.get_key(), (1000, n)) < 0.5)
              .astype(jnp.float32).mean(axis=1).std()) for n in ns]
d2l.plot(ns, [stds, [0.5 / n ** 0.5 for n in ns]], 'n', 'std of estimate',
         legend=['empirical', '0.5/sqrt(n)'], xscale='log', yscale='log')
```

Let's get some more intuition by studying
how our estimate evolves as we grow
the number of tosses from 1 to 10,000.

```{.python .input #probability-a-simple-example-tossing-coins-5}
%%tab pytorch
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
x = range(1, len(estimates) + 1)
d2l.plt.plot(x, estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(x, estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
ax = d2l.plt.gca()
ax.set_xscale('log')
ax.set_xlim(1, 10000)
ax.set_xticks([1, 10, 100, 1000, 10000])
ax.set_xticklabels(['1', '10', '100', '1,000', '10,000'])
ax.set_xlabel('Samples')
ax.set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input #probability-a-simple-example-tossing-coins-5}
%%tab mxnet
counts = multinomial(1, fair_probs, size=10000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((4.5, 3.5))
x = range(1, len(estimates) + 1)
d2l.plt.plot(x, estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(x, estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
ax = d2l.plt.gca()
ax.set_xscale('log')
ax.set_xlim(1, 10000)
ax.set_xticks([1, 10, 100, 1000, 10000])
ax.set_xticklabels(['1', '10', '100', '1,000', '10,000'])
ax.set_xlabel('Samples')
ax.set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input #probability-a-simple-example-tossing-coins-5}
%%tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(10000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
x = range(1, len(estimates) + 1)
d2l.plt.plot(x, estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(x, estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
ax = d2l.plt.gca()
ax.set_xscale('log')
ax.set_xlim(1, 10000)
ax.set_xticks([1, 10, 100, 1000, 10000])
ax.set_xticklabels(['1', '10', '100', '1,000', '10,000'])
ax.set_xlabel('Samples')
ax.set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input #probability-a-simple-example-tossing-coins-5}
%%tab jax
key = d2l.get_key()
counts = jnp.eye(2)[jax.random.categorical(key, jnp.log(fair_probs),
                                            shape=(10000,))]
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((4.5, 3.5))
x = range(1, len(estimates) + 1)
d2l.plt.plot(x, estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(x, estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
ax = d2l.plt.gca()
ax.set_xscale('log')
ax.set_xlim(1, 10000)
ax.set_xticks([1, 10, 100, 1000, 10000])
ax.set_xticklabels(['1', '10', '100', '1,000', '10,000'])
ax.set_xlabel('Samples')
ax.set_ylabel('Estimated probability')
d2l.plt.legend();
```

Each solid curve corresponds to one of the two values of the coin
and gives our estimated probability that the coin turns up that value
after each group of experiments.
The dashed black line gives the true underlying probability.
As we get more data by conducting more experiments,
the curves converge towards the true probability.
You might already begin to see the shape
of some of the more advanced questions
that preoccupy statisticians:
How quickly does this convergence happen?
If we had already tested many coins
manufactured at the same plant,
how might we incorporate this information?

## The Formal Language

### A More Formal Treatment

We have already gotten pretty far: posing
a probabilistic model,
generating synthetic data,
running a statistical estimator,
empirically assessing convergence,
and reporting error metrics (checking the deviation).
However, to go much further,
we will need to be more precise.


When dealing with randomness,
we denote the set of possible outcomes $\mathcal{S}$
and call it the *sample space* or *outcome space*.
Here, each element is a distinct possible *outcome*.
In the case of tossing a single coin,
$\mathcal{S} = \{\textrm{heads}, \textrm{tails}\}$.
For a single die, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$.
When flipping two coins, possible outcomes are
$\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails}), (\textrm{tails}, \textrm{heads}),  (\textrm{tails}, \textrm{tails})\}$.
*Events* are subsets of the sample space.
For instance, the event "the first coin toss comes up heads"
corresponds to the set $\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails})\}$.
Whenever the outcome $z$ of a random experiment satisfies
$z \in \mathcal{A}$, then event $\mathcal{A}$ has occurred.
For a single roll of a die, we could define the events
"seeing a $5$" ($\mathcal{A} = \{5\}$)
and "seeing an odd number"  ($\mathcal{B} = \{1, 3, 5\}$).
In this case, if the die came up $5$,
we would say that both $\mathcal{A}$ and $\mathcal{B}$ occurred.
On the other hand, if $z = 3$,
then $\mathcal{A}$ did not occur
but $\mathcal{B}$ did.


A *probability* function maps events
onto real values ${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$.
The probability, denoted $P(\mathcal{A})$, of an event $\mathcal{A}$
in the given sample space $\mathcal{S}$,
has the following properties:

* The probability of any event $\mathcal{A}$ is a nonnegative real number, i.e., $P(\mathcal{A}) \geq 0$;
* The probability of the entire sample space is $1$, i.e., $P(\mathcal{S}) = 1$;
* For any countable sequence of events $\mathcal{A}_1, \mathcal{A}_2, \ldots$ that are *mutually exclusive* (i.e., $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for all $i \neq j$), the probability that any of them happens is equal to the sum of their individual probabilities, i.e., $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

These axioms of probability theory,
proposed by :citet:`Kolmogorov.1933`,
can be applied to rapidly derive a number of important consequences.
For instance, it follows immediately
that the probability of any event $\mathcal{A}$
*or* its complement $\mathcal{A}'$ occurring is 1
(because $\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$).
We can also prove that $P(\emptyset) = 0$
because $1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.
Consequently, the probability of any event $\mathcal{A}$
*and* its complement $\mathcal{A}'$ occurring simultaneously
is $P(\mathcal{A} \cap \mathcal{A}') = 0$.
Informally, this tells us that impossible events
have zero probability of occurring.
The axioms also tell us how to handle events that *overlap*.
Writing $\mathcal{A} \cup \mathcal{B}$ as the disjoint union
of $\mathcal{A}$ and $\mathcal{B} \setminus \mathcal{A}$,
and noting that $P(\mathcal{B} \setminus \mathcal{A})
= P(\mathcal{B}) - P(\mathcal{A} \cap \mathcal{B})$,
we obtain the *inclusion--exclusion* rule

$$P(\mathcal{A} \cup \mathcal{B}) = P(\mathcal{A}) + P(\mathcal{B}) - P(\mathcal{A} \cap \mathcal{B}).$$

Summing the two probabilities counts the overlap twice,
so we subtract it once (:numref:`fig_prob_venn`).

![Events are subsets of a sample space $\mathcal{S}$, and the axioms give the *inclusion–exclusion* rule for their union.](../img/probability-venn.svg)
:label:`fig_prob_venn`



### Random Variables

When we spoke about events like the roll of a die
coming up odds or the first coin toss coming up heads,
we were invoking the idea of a *random variable*.
Formally, random variables are mappings
from an underlying sample space
to a set of (possibly many) values.
You might wonder how a random variable
is different from the sample space,
since both are collections of outcomes.
Importantly, random variables can be much coarser
than the raw sample space.
We can define a binary random variable like "greater than 0.5"
even when the underlying sample space is infinite,
e.g., points on the line segment between $0$ and $1$.
Additionally, multiple random variables
can share the same underlying sample space.
For example "whether my home alarm goes off"
and "whether my house was burgled"
are both binary random variables
that share an underlying sample space.
Consequently, knowing the value taken by one random variable
can tell us something about the likely value of another random variable.
Knowing that the alarm went off,
we might suspect that the house was likely burgled.


Every value taken by a random variable corresponds
to a subset of the underlying sample space.
Thus the occurrence where the random variable $X$
takes value $v$, denoted by $X=v$, is an *event*
and $P(X=v)$ denotes its probability.
Sometimes this notation can get clunky,
and we can abuse notation when the context is clear.
For example, we might use $P(X)$ to refer broadly
to the *distribution* of $X$, i.e.,
the function that tells us the probability
that $X$ takes any given value.
Other times we write an equation between distributions
as shorthand for a statement that holds
for all of the values the random variables can take.
For instance, the equation $P(X,Y) = P(X) P(Y)$
is shorthand for "$P(X=i \textrm{ and } Y=j) = P(X=i)P(Y=j)$
for all $i,j$"---although, as we will see when we discuss
*independence*, that particular equation holds only in special cases.
Other times, we abuse notation by writing
$P(v)$ when the random variable is clear from the context.
Since an event in probability theory is a set of outcomes from the sample space,
we can specify a range of values for a random variable to take.
For example, $P(1 \leq X \leq 3)$ denotes the probability of the event $\{1 \leq X \leq 3\}$.


Note that there is a subtle difference
between *discrete* random variables,
like flips of a coin or tosses of a die,
and *continuous* ones,
like the height of a person sampled at random from the population.
With fine enough measurements, no two people share an exact height,
so there is little point in asking for the probability
that someone is 1.801392782910287192 meters tall---it is zero.
The useful question is whether a height falls into an *interval*,
say between 1.79 and 1.81 meters.
In these cases we work with probability *densities*:
an exact value has no probability but nonzero density,
and the probability assigned to an interval
is the *integral* of the density over that interval
(:numref:`fig_prob_density`).

![Discrete random variables place probability *mass* on individual values; continuous ones spread a *density*, and probabilities arise by integrating it over intervals.](../img/probability-density.svg)
:label:`fig_prob_density`

## Multiple Random Variables

You might have noticed that we could not even
make it through the previous section without
making statements involving interactions
among multiple random variables
(recall our shorthand $P(X,Y)$ for the joint distribution).
Most of machine learning
is concerned with such relationships.
Here, the sample space would be
the population of interest,
say customers who transact with a business,
photographs on the Internet,
or proteins known to biologists.
Each random variable would represent
the (unknown) value of a different attribute.
Whenever we sample an individual from the population,
we observe a realization of each of the random variables.
Because the values taken by random variables
correspond to subsets of the sample space
that could be overlapping, partially overlapping,
or entirely disjoint,
knowing the value taken by one random variable
can cause us to update our beliefs
about which values of another random variable are likely.
If a patient walks into a hospital
and we observe that they
are having trouble breathing
and have lost their sense of smell,
then we believe that they are more likely
to have COVID-19 than we might
if they had no trouble breathing
and a perfectly ordinary sense of smell.


When working with multiple random variables,
we can construct events corresponding
to every combination of values
that the variables can jointly take.
The probability function that assigns
probabilities to each of these combinations
(e.g. $A=a$ and $B=b$)
is called the *joint probability* function
and simply returns the probability assigned
to the intersection of the corresponding subsets
of the sample space.
The *joint probability* assigned to the event
where random variables $A$ and $B$
take values $a$ and $b$, respectively,
is denoted $P(A = a, B = b)$,
where the comma indicates "and".
Note that for any values $a$ and $b$,
it follows that

$$P(A=a, B=b) \leq P(A=a) \textrm{ and } P(A=a, B=b) \leq P(B = b),$$

since for $A=a$ and $B=b$ to happen,
$A=a$ has to happen *and* $B=b$ also has to happen.
Interestingly, the joint probability
tells us all that we can know about these
random variables in a probabilistic sense,
and can be used to derive many other
useful quantities, including recovering the
individual distributions $P(A)$ and $P(B)$.
To recover $P(A=a)$ we simply sum up
$P(A=a, B=v)$ over all values $v$
that the random variable $B$ can take:
$P(A=a) = \sum_v P(A=a, B=v)$.


The ratio $\frac{P(A=a, B=b)}{P(A=a)} \leq 1$
turns out to be extremely important.
It is called the *conditional probability*,
and is denoted via the "$\mid$" symbol:

$$P(B=b \mid A=a) = P(A=a,B=b)/P(A=a).$$

It tells us the new probability
associated with the event $B=b$,
once we condition on the fact $A=a$ took place.
We can think of this conditional probability
as restricting attention only to the subset
of the sample space associated with $A=a$
and then renormalizing so that
all probabilities sum to 1.

![The joint distribution $P(A,B)$ determines everything: summing a row or column gives a *marginal* ($P(A)$ or $P(B)$), and renormalizing one row by its sum gives a *conditional* $P(B \mid A=a)$.](../img/probability-joint-grid.svg)
:label:`fig_prob_joint`

Conditional probabilities
are in fact just ordinary probabilities
and thus respect all of the axioms,
as long as we condition all terms
on the same event and thus
restrict attention to the same sample space.
For instance, for disjoint events
$\mathcal{B}$ and $\mathcal{B}'$, we have that
$P(\mathcal{B} \cup \mathcal{B}' \mid A = a) = P(\mathcal{B} \mid A = a) + P(\mathcal{B}' \mid A = a)$.


Using the definition of conditional probabilities,
we can derive the famous result called *Bayes' theorem*.
By construction, we have that $P(A, B) = P(B\mid A) P(A)$
and $P(A, B) = P(A\mid B) P(B)$.
Combining both equations yields
$P(B\mid A) P(A) = P(A\mid B) P(B)$ and hence

$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B)}.$$

This simple equation has profound implications because
it allows us to reverse the order of conditioning.
If we know how to estimate $P(B\mid A)$, $P(A)$, and $P(B)$,
then we can estimate $P(A\mid B)$.
We often find it easier to estimate one term directly
but not the other and Bayes' theorem can come to the rescue here.
For instance, if we know the prevalence of symptoms for a given disease,
and the overall prevalences of the disease and symptoms, respectively,
we can determine how likely someone is
to have the disease based on their symptoms.
In some cases we might not have direct access to $P(B)$,
such as the prevalence of symptoms.
In this case a simplified version of Bayes' theorem comes in handy:

$$P(A \mid B) \propto P(B \mid A) P(A).$$

Since we know that $P(A \mid B)$ must be normalized to $1$, i.e., $\sum_a P(A=a \mid B) = 1$,
we can use it to compute

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{\sum_a P(B \mid A=a) P(A = a)}.$$

In Bayesian statistics, we think of an observer
as possessing some (subjective) prior beliefs
about the plausibility of the available hypotheses
encoded in the *prior* $P(H)$,
and a *likelihood function* that says how likely
one is to observe any value of the collected evidence
for each of the hypotheses in the class $P(E \mid H)$.
Bayes' theorem is then interpreted as telling us
how to update the initial *prior* $P(H)$
in light of the available evidence $E$
to produce *posterior* beliefs
$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$.
Informally, this can be stated as
"posterior equals prior times likelihood, divided by the evidence".
Now, because the evidence $P(E)$ is the same for all hypotheses,
we can get away with simply normalizing over the hypotheses.

Note that $\sum_a P(A=a \mid B) = 1$ also allows us to *marginalize* over random variables. That is, we can drop variables from a joint distribution such as $P(A, B)$. After all, we have that

$$\sum_a P(B \mid A=a) P(A=a) = \sum_a P(B, A=a) = P(B).$$

Independence is another fundamentally important concept
that forms the backbone of
many important ideas in statistics.
In short, two variables are *independent*
if conditioning on the value of $A$ does not
cause any change to the probability distribution
associated with $B$ and vice versa.
More formally, independence, denoted $A \perp B$,
requires that $P(A \mid B) = P(A)$ and, consequently,
that $P(A,B) = P(A \mid B) P(B) = P(A) P(B)$.
Independence is often an appropriate assumption.
For example, if the random variable $A$
represents the outcome from tossing one fair coin
and the random variable $B$
represents the outcome from tossing another,
then knowing whether $A$ came up heads
should not influence the probability
of $B$ coming up heads.


Independence is especially useful when it holds among the successive
draws of our data from some underlying distribution
(allowing us to make strong statistical conclusions)
or when it holds among various variables in our data,
allowing us to work with simpler models
that encode this independence structure.
On the other hand, estimating the dependencies
among random variables is often the very aim of learning.
We care to estimate the probability of disease given symptoms
specifically because we believe
that diseases and symptoms are *not* independent.


Note that because conditional probabilities are proper probabilities,
the concepts of independence and dependence also apply to them.
Two random variables $A$ and $B$ are *conditionally independent*
given a third variable $C$ if and only if $P(A, B \mid C) = P(A \mid C)P(B \mid C)$.
Interestingly, two variables can be independent in general
but become dependent when conditioning on a third.
This often occurs when the two random variables $A$ and $B$
correspond to causes of some third variable $C$.
For example, broken bones and lung cancer might be independent
in the general population but if we condition on being in the hospital
then we might find that broken bones are negatively correlated with lung cancer.
That is because the broken bone *explains away* why some person is in the hospital
and thus lowers the probability that they are hospitalized because of having lung cancer.


And conversely, two dependent random variables
can become independent upon conditioning on a third.
This often happens when two otherwise unrelated events
have a common cause.
Shoe size and reading level are highly correlated
among elementary school students,
but this correlation disappears if we condition on age.

![Conditioning can both destroy and create dependence. A *common cause* makes $A$ and $B$ dependent until we condition on $C$; a *collider* (common effect) makes independent causes dependent once $C$ is observed — *explaining away*.](../img/probability-explaining-away.svg)
:label:`fig_prob_explaining_away`



## Worked Example: HIV Testing
:label:`subsec_probability_hiv_app`

Let's put our skills to the test.
Assume that a doctor administers an HIV test to a patient.
This test is fairly accurate and fails only with 1% probability
if the patient is healthy but reported as diseased,
i.e., healthy patients test positive in 1% of cases.
Moreover, it never fails to detect HIV if the patient actually has it.
We use $D_1 \in \{0, 1\}$ to indicate the diagnosis
($0$ if negative and $1$ if positive)
and $H \in \{0, 1\}$ to denote the HIV status.

| $P(D_1 \mid H)$ | $H=1$ (HIV) | $H=0$ (healthy) |
|:--|:--:|:--:|
| test positive, $D_1 = 1$ | 1.00 | 0.01 |
| test negative, $D_1 = 0$ | 0.00 | 0.99 |

Note that the column sums are all 1 (but the row sums do not),
since they are conditional probabilities.
Let's compute the probability of the patient having HIV
if the test comes back positive, i.e., $P(H = 1 \mid D_1 = 1)$.
Intuitively this is going to depend on how common the disease is,
since it affects the number of false alarms.
Assume that the population is fairly free of the disease, e.g., $P(H=1) = 0.0015$.
To apply Bayes' theorem, we need to apply marginalization
to determine

$$\begin{aligned}
P(D_1 = 1)
&= P(D_1=1, H=0) + P(D_1=1, H=1)  \\
&= P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
&= 0.011485.
\end{aligned}
$$

This leads us to

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

In other words, there is only a 13.06% chance
that the patient actually has HIV,
despite the test being pretty accurate.

![The same result in *natural frequencies*. Because the disease is rare, the few true positives are swamped by false positives — of roughly 115 positive tests, only 15 are real, so $P(H=1 \mid D_1=1) \approx 13\%$.](../img/probability-natural-frequencies.svg)
:label:`fig_prob_natural_freq`

As we can see, probability can be counterintuitive.
What should a patient do upon receiving such terrifying news?
Likely, the patient would ask the physician
to administer another test to get clarity.
The second test has different characteristics
and it is not as good as the first one.

| $P(D_2 \mid H)$ | $H=1$ (HIV) | $H=0$ (healthy) |
|:--|:--:|:--:|
| test positive, $D_2 = 1$ | 0.98 | 0.03 |
| test negative, $D_2 = 0$ | 0.02 | 0.97 |

Unfortunately, the second test comes back positive, too.
Let's calculate the requisite probabilities to invoke Bayes' theorem
by assuming conditional independence:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1 \mid H = 0)
&= P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0) = 0.0003, \\
P(D_1 = 1, D_2 = 1 \mid H = 1)
&= P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1) = 0.98.
\end{aligned}
$$

Now we can apply marginalization to obtain the probability
that both tests come back positive:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1)\\
&= P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
&= P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
&= 0.00176955.
\end{aligned}
$$

Finally, the probability of the patient having HIV given that both tests are positive is

$$P(H = 1 \mid D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

That is, the second test allowed us to gain much higher confidence that not all is well.
Despite the second test being considerably less accurate than the first one,
it still significantly improved our estimate.

![Each conditionally independent positive test multiplies the evidence, driving the posterior $P(H=1)$ from a 0.15% prior to 13% and then to 83%.](../img/probability-bayes-update.svg)
:label:`fig_prob_update`

Since we posed a complete generative model,
we can also check the whole calculation by brute force:
simulate ten million patients,
apply both tests to each,
and look at the fraction of HIV cases
among those with two positive results.
This is the natural-frequencies picture of
:numref:`fig_prob_natural_freq` made executable,
and the empirical frequency should land close
to the exact posterior of $0.8307$.

```{.python .input #probability-worked-example-hiv-testing-1}
%%tab mxnet
n = 10000000
H = np.random.uniform(size=n) < 0.0015
D1 = np.random.uniform(size=n) < np.where(H, 1.00, 0.01)
D2 = np.random.uniform(size=n) < np.where(H, 0.98, 0.03)
both = np.logical_and(D1, D2).astype(np.float32)
(H.astype(np.float32) * both).sum() / both.sum()  # Exact value: 0.8307
```

```{.python .input #probability-worked-example-hiv-testing-1}
%%tab pytorch
n = 10000000
H = torch.rand(n) < 0.0015
D1 = torch.rand(n) < torch.where(H, 1.00, 0.01)
D2 = torch.rand(n) < torch.where(H, 0.98, 0.03)
H[D1 & D2].float().mean()  # Exact value: 0.8307
```

```{.python .input #probability-worked-example-hiv-testing-1}
%%tab tensorflow
n = 10000000
H = tf.random.uniform((n,)) < 0.0015
D1 = tf.random.uniform((n,)) < tf.where(H, 1.00, 0.01)
D2 = tf.random.uniform((n,)) < tf.where(H, 0.98, 0.03)
tf.reduce_mean(tf.cast(tf.boolean_mask(H, D1 & D2), tf.float32))  # Exact value: 0.8307
```

```{.python .input #probability-worked-example-hiv-testing-1}
%%tab jax
k1, k2, k3 = jax.random.split(d2l.get_key(), 3)
n = 10000000
H = jax.random.uniform(k1, (n,)) < 0.0015
D1 = jax.random.uniform(k2, (n,)) < jnp.where(H, 1.00, 0.01)
D2 = jax.random.uniform(k3, (n,)) < jnp.where(H, 0.98, 0.03)
H[D1 & D2].astype(jnp.float32).mean()  # Exact value: 0.8307
```

The assumption of both tests being conditionally independent of each other
was crucial for our ability to generate a more accurate estimate.
Take the extreme case where we run the same test twice.
In this situation we would expect the same outcome both times,
hence no additional insight is gained from running the same test again.
The astute reader might have noticed that the diagnosis behaved
like a classifier hiding in plain sight
where our ability to decide whether a patient is healthy
increases as we obtain more features (test outcomes).


## Expectations

Often, making decisions requires not just looking
at the probabilities assigned to individual events
but composing them together into useful aggregates
that can provide us with guidance.
For example, when random variables take continuous scalar values,
we often care about knowing what value to expect *on average*.
This quantity is formally called an *expectation*.
If we are making investments,
the first quantity of interest
might be the return we can expect,
averaging over all the possible outcomes
(and weighting by the appropriate probabilities).
For instance, say that with 50% probability,
an investment might fail altogether,
with 40% probability it might provide a 2$\times$ return,
and with 10% probability it might provide a 10$\times$ return.
To calculate the expected return,
we sum over all returns, multiplying each
by the probability that they will occur.
This yields the expectation
$0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$.
Hence the expected return is 1.8$\times$.


In general, the *expectation* (or average)
of the random variable $X$ is defined as

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

Likewise, for densities we obtain $E[X] = \int x \, p(x) \;dx$.
Sometimes we are interested in the expected value
of some function of $x$.
We can calculate these expectations as

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \textrm{ and } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$

for discrete probabilities and densities, respectively.
Returning to the investment example from above,
$f$ might be the *utility* (happiness)
associated with the return.
Behavioral economists have long noted
that people associate greater disutility
with losing money than the utility gained
from earning one dollar relative to their baseline.
Moreover, the value of money tends to be sub-linear.
Possessing 100k dollars versus zero dollars
can make the difference between paying the rent,
eating well, and enjoying quality healthcare
versus suffering through homelessness.
On the other hand, the gains due to possessing
200k versus 100k are less dramatic.
Reasoning like this motivates the cliché
that "the utility of money is logarithmic".


If  the utility associated with a total loss were $-1$,
and the utilities associated with returns of $1$, $2$, and $10$
were $1$, $2$ and $4$, respectively,
then the expected happiness of investing
would be $0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$
(an expected utility of $0.7$, i.e., $0.3$ below the utility of $1$ you would get from a guaranteed return of $1$).
If indeed this were your utility function,
you might be best off keeping the money in the bank.

For financial decisions,
we might also want to measure
how *risky* an investment is.
Here, we care not just about the expected value
but how much the actual values tend to *vary*
relative to this value.
Note that we cannot just take
the expectation of the difference
between the actual and expected values.
This is because the expectation of a difference
is the difference of the expectations,
i.e., $E[X - E[X]] = E[X] - E[E[X]] = 0$.
However, we can look at the expectation
of any non-negative function of this difference.
The *variance* of a random variable is calculated by looking
at the expected value of the *squared* differences:

$$\textrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

Here the equality follows by expanding
$(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$
and taking expectations for each term.
The square root of the variance is another
useful quantity called the *standard deviation*.
While this and the variance
convey the same information (either can be calculated from the other),
the standard deviation has the nice property
that it is expressed in the same units
as the original quantity represented
by the random variable.

Lastly, the variance of a function
of a random variable
is defined analogously as

$$\textrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$

Returning to our investment example,
we can now compute the variance of the investment.
It is given by $0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$.
For all intents and purposes this is a risky investment.
Note that by mathematical convention mean and variance
are often referenced as $\mu$ and $\sigma^2$.
This is particularly the case whenever we use it
to parametrize a Gaussian distribution.

In the same way as we introduced expectations
and variance for *scalar* random variables,
we can do so for vector-valued ones.
Expectations are easy, since we can apply them elementwise.
For instance, $\boldsymbol{\mu} \stackrel{\textrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$
has coordinates $\mu_i = E_{\mathbf{x} \sim P}[x_i]$.
*Covariances* are more complicated.
We define them by taking expectations of the *outer product*
of the difference between random variables and their mean:

$$\boldsymbol{\Sigma} \stackrel{\textrm{def}}{=} \textrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$

This matrix $\boldsymbol{\Sigma}$ is referred to as the covariance matrix.
An easy way to see its effect is to consider some vector $\mathbf{v}$
of the same size as $\mathbf{x}$.
It follows that

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \textrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$

As such, $\boldsymbol{\Sigma}$ allows us to compute the variance
for any linear function of $\mathbf{x}$
by a simple matrix multiplication.
The off-diagonal elements tell us how the coordinates vary together:
a value of 0 means no correlation.
Beware, though, that the magnitude of a covariance is scale-dependent:
it changes whenever we change the units in which a coordinate is measured.
The scale-free measure of the strength of association is the *correlation*
$\rho_{ij} = \Sigma_{ij}/(\sigma_i \sigma_j)$,
which always lies in $[-1, 1]$.

### From Means to Tail Bounds

Knowing an expectation already constrains
how often a random variable can be large.
*Markov's inequality* states that for a *nonnegative*
random variable $X$ and any threshold $a > 0$,

$$P(X \geq a) \leq \frac{E[X]}{a}.$$

The proof takes one line.
Since $X \geq 0$, discarding the outcomes below the threshold
can only decrease the expectation, so
$E[X] \geq E[X \cdot \mathbf{1}_{X \geq a}] \geq a \, E[\mathbf{1}_{X \geq a}] = a \, P(X \geq a)$;
dividing by $a$ gives the claim.
Remarkably, nothing about the distribution of $X$ was used:
the bound is *distribution-free* (:numref:`fig_prob_markov`).

![Markov's inequality: for a nonnegative random variable, the probability of exceeding a threshold $a$ is at most the mean divided by $a$---no matter what the distribution looks like.](../img/probability-markov.svg)
:label:`fig_prob_markov`

The payoff comes from choosing $X$ cleverly.
Applying Markov's inequality to the nonnegative variable
$(X - \mu)^2$, whose expectation is precisely
the variance $\sigma^2$, yields *Chebyshev's inequality*:
for any $k > 0$,

$$P(|X - \mu| \geq k \sigma) = P\left((X - \mu)^2 \geq k^2 \sigma^2\right) \leq \frac{\sigma^2}{k^2 \sigma^2} = \frac{1}{k^2}.$$

For instance, draws from *any* distribution with mean $\mu$
and variance $\sigma^2$ land within
$k = \sqrt{2}$ standard deviations of $\mu$
with probability at least 50%.
Sharper bounds---Chernoff's and Hoeffding's---and
what such concentration results say about generalization
in machine learning are developed in
:numref:`sec_mdl-concentration-generalization`.

## Discussion

In machine learning, there are many things to be uncertain about!
We can be uncertain about the value of a label given an input.
We can be uncertain about the estimated value of a parameter.
We can even be uncertain about whether data arriving at deployment
is even from the same distribution as the training data.

By *aleatoric uncertainty*, we mean uncertainty
that is intrinsic to the problem,
and due to genuine randomness
unaccounted for by the observed variables.
By *epistemic uncertainty*, we mean uncertainty
over a model's parameters, the sort of uncertainty
that we can hope to reduce by collecting more data.
We might have epistemic uncertainty
concerning the probability
that a coin turns up heads,
but even once we know this probability,
we are left with aleatoric uncertainty
about the outcome of any future toss.
No matter how long we watch someone tossing a fair coin,
we will never be more or less than 50% certain
that the next toss will come up heads.
These terms come from mechanical modeling,
(see e.g., :citet:`Der-Kiureghian.Ditlevsen.2009` for a review on this aspect of [uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification)).
It is worth noting, however, that these terms constitute a slight abuse of language.
The term *epistemic* refers to anything concerning *knowledge*
and thus, in the philosophical sense, all uncertainty is epistemic.


We saw that sampling data from some unknown probability distribution
can provide us with information that can be used to estimate
the parameters of the data generating distribution.
That said, the rate at which this is possible can be quite slow.
In our coin tossing example (and many others)
we can do no better than to design estimators
that converge at a rate of $1/\sqrt{n}$,
where $n$ is the sample size (e.g., the number of tosses).
This means that by going from 10 to 1000 observations (usually a very achievable task)
we see a tenfold reduction of uncertainty,
whereas the next 1000 observations help comparatively little,
offering only a 1.41 times reduction.
This is a persistent feature of machine learning:
while there are often easy gains, it takes a very large amount of data,
and often with it an enormous amount of computation, to make further gains.
For an empirical review of this fact for large-scale language models, see :citet:`kaplan2020scaling`.

We also sharpened our language and tools for statistical modeling.
In the process of that we learned about conditional probabilities
and about one of the most important equations in statistics---Bayes' theorem.
It is an effective tool for decoupling information conveyed by data
through a likelihood term $P(B \mid A)$ that addresses
how well observations $B$ match a choice of parameters $A$,
and a prior probability $P(A)$ which governs how plausible
a particular choice of $A$ was in the first place.
In particular, we saw how this rule can be applied
to assign probabilities to diagnoses,
based on the efficacy of the test *and*
the prevalence of the disease itself (i.e., our prior).

Lastly, we introduced a first set of nontrivial questions
about the effect of a specific probability distribution,
namely expectations and variances.
While there are many more than just linear and quadratic
expectations for a probability distribution,
these two already provide a good deal of knowledge
about the possible behavior of the distribution:
we used them to derive our first tail bounds,
Markov's and Chebyshev's inequalities,
which hold for *every* distribution.
Their sharper cousins, and the connection between
concentration and generalization, await in
:numref:`sec_mdl-concentration-generalization`.

This section is only a preview of the probability and statistics
we will use.
:numref:`chap_mdl-probability-statistics` develops it in full---common
distributions such as the Bernoulli and Gaussian
(:numref:`sec_mdl-distributions`), maximum likelihood estimation
(:numref:`sec_mdl-maximum_likelihood`), and the elements of statistics
that turn data into estimates---while
:numref:`chap_mdl-information-theory` introduces entropy and the
divergences between distributions.
For a thorough yet accessible reference, see :citet:`Wasserman.2013`.




## Exercises

1. Give an example where observing more data can reduce the amount of uncertainty about the outcome to an arbitrarily low level.
1. Give an example where observing more data will only reduce the amount of uncertainty up to a point and then no further. Explain why this is the case and where you expect this point to occur.
1. We empirically demonstrated convergence to the mean for the toss of a coin. Calculate the variance of the estimate of the probability that we see a head after drawing $n$ samples.
    1. How does the variance scale with the number of observations?
    1. Use Chebyshev's inequality to bound the deviation from the expectation.
    1. How does it relate to the central limit theorem?
1. Assume that we draw $m$ samples $x_i$ from a probability distribution with zero mean and unit variance. Compute the averages $z_m \stackrel{\textrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$. Can we apply Chebyshev's inequality for every $z_m$ independently? Why not?
1. Given two events with probability $P(\mathcal{A})$ and $P(\mathcal{B})$, compute upper and lower bounds on $P(\mathcal{A} \cup \mathcal{B})$ and $P(\mathcal{A} \cap \mathcal{B})$. Hint: graph the situation using a [Venn diagram](https://en.wikipedia.org/wiki/Venn_diagram).
1. Assume that we have a sequence of random variables, say $A$, $B$, and $C$, where $B$ only depends on $A$, and $C$ only depends on $B$, can you simplify the joint probability $P(A, B, C)$? Hint: this is a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain).
1. In :numref:`subsec_probability_hiv_app`, assume that the outcomes of the two tests are not independent. In particular assume that either test on its own has a false positive rate of 10% and a false negative rate of 1%. That is, assume that $P(D =1 \mid H=0) = 0.1$ and that $P(D = 0 \mid H=1) = 0.01$. Moreover, assume that for $H = 1$ (infected) the test outcomes are conditionally independent, i.e., that $P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$ but that for healthy patients the outcomes are coupled via $P(D_1 = D_2 = 1 \mid H=0) = 0.02$.
    1. Work out the joint probability table for $D_1$ and $D_2$, given $H=0$ based on the information you have so far.
    1. Derive the probability that the patient is diseased ($H=1$) after one test returns positive. You can assume the same baseline probability $P(H=1) = 0.0015$ as before.
    1. Derive the probability that the patient is diseased ($H=1$) after both tests return positive.
1. Assume that you are an asset manager for an investment bank and you have a choice of stocks $s_i$ to invest in. Your portfolio needs to add up to $1$ with weights $\alpha_i$ for each stock. The stocks have an average return $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$ and covariance $\boldsymbol{\Sigma} = \textrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
    1. Compute the expected return for a given portfolio $\boldsymbol{\alpha}$.
    1. If you wanted to maximize the return of the portfolio, how should you choose your investment?
    1. Compute the *variance* of the portfolio.
    1. Formulate an optimization problem of maximizing the return while keeping the variance constrained to an upper bound. This is the Nobel-Prize winning [Markovitz portfolio](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Mangram.2013`. To solve it you will need a quadratic programming solver, something way beyond the scope of this book.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/198)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17971)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.6]{.kicker}

Reasoning under uncertainty<br>**sampling · distributions · Bayes · expectation**
:::
:::

::: {.slide title="Machine learning is inference under uncertainty"}
[Motivation]{.kicker}

- A model rarely returns one answer — it returns a **distribution** over
  answers.
- Training **maximizes likelihood**; most losses are negative
  log-likelihoods.
- Generalization, regularization, and the Bayesian view all rest on the
  same handful of rules.

::: {.d2l-note}
**Probability** reasons *forward*: model → data. **Statistics** reasons
*backward*: data → model. We build both, on one running example.
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[From data to a probability]{.dtitle}

[sampling, frequencies, the law of large numbers]{.dsub}
:::
:::

::: {.slide title="A coin of unknown bias"}
[Estimating from data]{.kicker}

We find a coin and want $P(\text{heads})$ — but nobody tells us its
value. The plan: **toss it many times and count**.

A single batch of 100 tosses with `random.random()` already lands
**near** 50/50 — but never exactly, because sampling has variance:

@probability-a-simple-example-tossing-coins-1
:::

::: {.slide title="Multinomial draws 100 tosses in one call"}
[Estimating from data]{.kicker}

A cleaner tool: a `Multinomial` over `{heads, tails}` with probabilities
`[0.5, 0.5]` returns the **count vector** directly:

@probability-a-simple-example-tossing-coins-2

. . .

Dividing by the number of tosses gives **empirical frequencies** — our
estimates of $P(\text{heads})$ and $P(\text{tails})$:

@probability-a-simple-example-tossing-coins-3
:::

::: {.slide title="More data, tighter estimate"}
[Estimating from data]{.kicker}

With **10,000** tosses the frequencies sit far closer to the true
$\tfrac{1}{2}$:

@probability-a-simple-example-tossing-coins-4

::: {.d2l-note}
The **law of large numbers**: as the number of trials $n \to \infty$,
the empirical frequency converges to the true probability.
:::
:::

::: {.slide title="The estimate locks on — at a 1/√n crawl"}
[Estimating from data]{.kicker}

::: {.cols .vc}
::: {.col}
The running estimate vs. sample count zig-zags inward and locks onto
$0.5$:

@!probability-a-simple-example-tossing-coins-5
:::

::: {.col .narrow}
::: {.d2l-note .rule}
The error shrinks like $1/\sqrt{n}$ — to **halve** it you need
**4×** the data.
:::

A first glimpse of the real question of statistics: not just *what* we
estimate, but *how sure* we are.
:::
:::
:::

::: {.slide title="The 1/√n law, measured: slope −½"}
[Estimating from data]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
Why $1/\sqrt{n}$? Each toss has variance $p(1-p)$; averaging $n$
independent tosses gives

$$\textrm{Var}[\hat{p}] = \frac{p(1-p)}{n}.$$

Estimating $p$ from 1000 batches at each $n$, the standard deviation of
the estimates hugs the predicted $0.5/\sqrt{n}$ line — **slope
$-\tfrac12$** on log–log axes.
:::

::: {.col .fig .big}
@!probability-a-simple-example-tossing-coins-6
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The formal language]{.dtitle}

[sample spaces, events, random variables]{.dsub}
:::
:::

::: {.slide title="Three axioms generate every rule"}
[Formal treatment]{.kicker}

::: {.cols .vc}
::: {.col}
Every outcome lives in a **sample space** $\mathcal{S}$; an **event** is
a subset. A probability assigns each event a number in $[0,1]$ obeying
three rules (Kolmogorov):

- $P(\mathcal{A}) \ge 0$;
- $P(\mathcal{S}) = 1$;
- disjoint events **add**.

::: {.d2l-note .rule}
Everything else follows — e.g. inclusion–exclusion:
$P(\mathcal{A}\cup\mathcal{B}) =
P(\mathcal{A}) + P(\mathcal{B}) - P(\mathcal{A}\cap\mathcal{B})$.
:::
:::

::: {.col .fig .big}
@fig:probability-venn
:::
:::
:::

::: {.slide title="Mass sits on points; density needs intervals"}
[Formal treatment]{.kicker}

::: {.cols .vc}
::: {.col}
A **random variable** maps outcomes to values. Discrete ones (a die)
place **mass** on points; continuous ones (a height) spread **density**
along the line.

::: {.d2l-note}
For a continuous variable an *exact* value has probability **zero** —
only intervals carry probability, obtained by **integrating** the
density.
:::
:::

::: {.col .fig .big}
@fig:probability-density
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Joint, marginal, conditional]{.dtitle}

[how two variables relate — and Bayes' theorem]{.dsub}
:::
:::

::: {.slide title="One table holds everything"}
[Multiple variables]{.kicker}

::: {.cols .vc}
::: {.col}
The **joint** $P(A,B)$ lists every combination. From it:

- **sum** a row or column → a **marginal** $P(A)$ or $P(B)$;
- **renormalize** one row → a **conditional**
  $P(B \mid A{=}a) = \dfrac{P(A{=}a,\, B)}{P(A{=}a)}$.

::: {.d2l-note}
Conditioning = restrict to the slice where $A{=}a$, then rescale so it
sums to 1.
:::
:::

::: {.col .fig .big}
@fig:probability-joint-grid
:::
:::
:::

::: {.slide title="Bayes' theorem reverses the conditioning"}
[Multiple variables]{.kicker}

Write the joint two ways, $P(A,B) = P(B\mid A)\,P(A) = P(A\mid B)\,P(B)$,
and equate:

$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}.$$

. . .

This **flips** a hard direction into an easy one — inferring a cause $A$
from an effect $B$ when only $P(B \mid A)$ is known.

::: {.d2l-note .rule}
posterior $\propto$ likelihood $\times$ prior:
$\;P(H \mid E) \propto P(E \mid H)\,P(H)$.
:::
:::

::: {.slide title="Independence — and explaining away"}
[Multiple variables]{.kicker}

::: {.cols .vc}
::: {.col}
Independence, $A \perp B$, means $P(A,B) = P(A)\,P(B)$. But
**conditioning changes dependence**: a common cause links two variables
until you condition on it; a collider (common effect) makes independent
causes dependent once you do — *explaining away*.
:::

::: {.col .fig .big}
@fig:probability-explaining-away
:::
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Bayes in action: the HIV test]{.dtitle}

[why a "99% accurate" test can still mislead]{.dsub}
:::
:::

::: {.slide title="A test that never misses — and still misleads"}
[Worked example]{.kicker}

The test catches **every** true HIV case but has a **1% false-positive**
rate, and the disease is **rare**:

$$\begin{aligned}
P(D{=}1 \mid H{=}1) &= 1.00 \\
P(D{=}1 \mid H{=}0) &= 0.01 \\
\text{prior } P(H{=}1) &= 0.0015
\end{aligned}$$

We want the posterior $P(H{=}1 \mid D{=}1)$. Intuition says "almost
certainly sick" — but Bayes disagrees. Let us count.
:::

::: {.slide title="Of ~115 positives, only 15 are real"}
[Worked example]{.kicker}

::: {.cols .vc}
::: {.col}
Among 10,000 people only ~15 truly have HIV — but ~100 healthy people
also test positive:

$$P(H{=}1 \mid D{=}1) \approx \tfrac{15}{115} \approx 13\%.$$

The **base rate** dominates a rare-disease test.
:::

::: {.col .fig .big}
@fig:probability-natural-frequencies
:::
:::
:::

::: {.slide title="A second positive: 0.15% → 13% → 83%"}
[Worked example]{.kicker}

::: {.cols .vc}
::: {.col}
A *second*, independent positive test multiplies the evidence: applying
Bayes again drives the posterior from the $0.15\%$ prior to $13\%$,
then to $83\%$ — exactly $0.8307$.

Ten million simulated patients land on the same number:

@!probability-worked-example-hiv-testing-1
:::

::: {.col .fig .big}
@fig:probability-bayes-update
:::
:::
:::

::: {.slide}
::: {.divider}
[05]{.dnum}

[Summarizing a distribution]{.dtitle}

[expectation, variance, covariance]{.dsub}
:::
:::

::: {.slide title="Expectation: the probability-weighted average"}
[Summaries]{.kicker}

::: {.cols .vc}
::: {.col}
$$E[X] = \sum_x x\,P(X{=}x).$$

It is the **balance point** of the distribution. For an investment
paying $0$, $2\times$, or $10\times$ with probabilities $0.5, 0.4, 0.1$,
the expected return is $1.8\times$.
:::

::: {.col .fig .big}
@fig:probability-expectation
:::
:::
:::

::: {.slide title="Variance: same mean, different risk"}
[Summaries]{.kicker}

::: {.cols .vc}
::: {.col}
$$\textrm{Var}[X] = E\big[(X - E[X])^2\big] = E[X^2] - E[X]^2.$$

Two investments can share a mean yet differ wildly in **spread**. The
standard deviation $\sigma = \sqrt{\textrm{Var}[X]}$ reports it in the
original units.
:::

::: {.col .fig .big}
@fig:probability-spread
:::
:::
:::

::: {.slide title="Covariance: the sign, not the size, is the story"}
[Summaries]{.kicker}

Covariance is the expected product of the two centered variables; its
**sign** says whether they move together (magnitude is scale-dependent —
rescale by the standard deviations to get the *correlation*). Stacked
over a vector, it becomes the **covariance matrix**
$\boldsymbol{\Sigma}$ — symmetric, and a workhorse of the chapters
ahead:

@fig:probability-covariance
:::

::: {.slide}
::: {.divider}
[06]{.dnum}

[Uncertainty & guarantees]{.dtitle}

[what kind, and how far can it stray?]{.dsub}
:::
:::

::: {.slide title="Two kinds of uncertainty"}
[Discussion]{.kicker}

::: {.cols .vc}
::: {.col}
**Aleatoric** uncertainty is intrinsic randomness — the next fair-coin
flip stays 50/50 no matter how much data you gather. **Epistemic**
uncertainty is about *unknown parameters*, and it **shrinks** as data
accumulates.
:::

::: {.col .fig .big}
@fig:probability-uncertainty
:::
:::
:::

::: {.slide title="Tail bounds: guarantees without the distribution"}
[Discussion]{.kicker}

::: {.cols .vc}
::: {.col}
Even *without* knowing the distribution, we can bound how often a
nonnegative variable lands far out. **Markov:**
$$P(X \ge a) \le \frac{E[X]}{a}.$$

::: {.d2l-note .rule}
Apply it to $(X-\mu)^2$ to get **Chebyshev**; sharper bounds
(Hoeffding, Bernstein) and their consequences for generalization are
developed in §25.5.
:::
:::

::: {.col .fig .big}
@fig:probability-markov
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Sample → count → estimate**; the LLN converges at $1/\sqrt{n}$
  (slope $-\tfrac12$, measured).
- **Axioms** generate every rule; events combine by
  inclusion–exclusion.
- The **joint** yields marginals (sum) and conditionals (renormalize).
- **Bayes** reverses conditioning: posterior $\propto$ likelihood
  $\times$ prior.
:::

::: {.col}
- **Base rates** rule rare-disease tests: $13\%$ after one positive,
  $0.8307$ after two.
- **Expectation / variance / covariance** summarize a distribution.
- **Tail bounds** (Markov → Chebyshev) guarantee concentration even
  when the distribution is unknown.
:::
:::
:::
