# Generalization in Deep Learning
:label:`sec_generalization_deep`


In :numref:`chap_regression` and :numref:`chap_classification`,
we tackled regression and classification problems
by fitting linear models to training data.
In both cases, we provided practical algorithms
for finding the parameters that maximized
the likelihood of the observed training labels.
And then, towards the end of each chapter,
we recalled that fitting the training data
was only an intermediate goal.
Our real quest all along was to discover *general patterns*
on the basis of which we can make accurate predictions
even on new examples drawn from the same underlying population.
Optimization is merely a means to an end: machine learning researchers
consume optimization algorithms, and sometimes invent new ones,
but always in service of a statistical goal.
At its core, machine learning is a statistical discipline
and we wish to optimize training loss only insofar
as some statistical principle (known or unknown)
leads the resulting models to generalize beyond the training set.


On the bright side, it turns out that deep neural networks
trained by stochastic gradient descent generalize remarkably well
across myriad prediction problems, spanning computer vision;
natural language processing; time series data; recommender systems;
electronic health records; protein folding;
value function approximation in video games
and board games; and numerous other domains.
On the downside, if you were looking
for a straightforward account
of either the optimization story
(why we can fit them to training data)
or the generalization story
(why the resulting models generalize to unseen examples),
then you might want to pour yourself a drink.
While our procedures for optimizing linear models
and the statistical properties of the solutions
are both described well by a comprehensive body of theory,
our understanding of deep learning
still resembles the wild west on both fronts.

Both the theory and practice of deep learning
are rapidly evolving,
with theorists adopting new strategies
to explain what's going on,
even as practitioners continue
to innovate at a blistering pace,
building arsenals of heuristics for training deep networks
and a body of intuitions and folk knowledge
that provide guidance for deciding
which techniques to apply in which situations.

The summary of the present moment is that the theory of deep learning
has produced promising lines of attack and scattered fascinating results,
but still appears far from a comprehensive account
of both (i) why we are able to optimize neural networks
and (ii) how models learned by gradient descent
manage to generalize so well, even on high-dimensional tasks.
However, in practice, (i) is seldom a problem
(we can always find parameters that will fit all of our training data)
and thus understanding generalization is by far the bigger problem.
On the other hand, even absent the comfort of a coherent scientific theory,
practitioners have developed a large collection of techniques
that may help you to produce models that generalize well in practice.
While no pithy summary can possibly do justice
to the vast topic of generalization in deep learning,
and while the overall state of research is far from resolved,
we hope, in this section, to present a broad overview
of the state of research and practice.


## Revisiting Overfitting and Regularization

According to the "no free lunch" theorem of :citet:`wolpert1995no`,
any learning algorithm generalizes better on some data distributions
and worse on others.
Thus, given a finite training set,
a model must rely on assumptions, or *inductive biases*.
To achieve human-level performance
it can help to choose inductive biases
that reflect how humans think about the world.
Such inductive biases show preferences 
for solutions with certain properties.
For example,
a deep MLP has an inductive bias
towards building up a complicated function by the composition of simpler functions.

With machine learning models encoding inductive biases,
our approach to training them
typically consists of two phases: (i) fit the training data;
and (ii) estimate the *generalization error*
(the true error on the underlying population)
by evaluating the model on holdout data.
The difference between our fit on the training data
and our fit on the test data is called the *generalization gap* and when this is large,
we say that our models *overfit* to the training data.
In extreme cases of overfitting,
we might exactly fit the training data,
even when the test error remains significant.
And in the classical view,
the interpretation is that our models are too complex,
requiring that we either shrink the number of features,
the number of nonzero parameters learned,
or the size of the parameters as quantified.
Recall the plot of model complexity compared with loss
(:numref:`fig_capacity_vs_error`)
from :numref:`sec_generalization_basics`.


However deep learning complicates this picture in counterintuitive ways.
First, for classification problems,
our models are typically expressive enough
to perfectly fit every training example,
even in datasets consisting of millions
:cite:`zhang2021understanding`.
In the classical picture, we might think
that this setting lies on the far right extreme
of the model complexity axis,
and that any improvements in generalization error
must come by way of regularization,
either by reducing the complexity of the model class,
or by applying a penalty, severely constraining
the set of values that our parameters might take.
But that is where things start to get weird.

Strangely, for many deep learning tasks
(e.g., image recognition and text classification)
we are typically choosing among model architectures,
all of which can achieve arbitrarily low training loss
(and zero training error).
Because all models under consideration achieve zero training error,
*the only avenue for further gains is to reduce overfitting*.
Even stranger, it is often the case that
despite fitting the training data perfectly,
we can actually *reduce the generalization error*
further by making the model *even more expressive*,
e.g., adding layers, nodes, or training
for a larger number of epochs.
Stranger yet, the pattern relating the generalization gap
to the *complexity* of the model
(as captured, for example, in the depth or width of the networks)
can be non-monotonic,
with greater complexity hurting at first
but subsequently helping in a so-called "double-descent" pattern
:cite:`Belkin.Hsu.Ma.ea.2019,nakkiran2021deep`,
which we examine in detail below.
Thus the deep learning practitioner possesses a bag of tricks,
some of which seemingly restrict the model in some fashion
and others that seemingly make it even more expressive,
and all of which, in some sense, are applied to mitigate overfitting.

Complicating things even further,
while the guarantees provided by classical learning theory
can be conservative even for classical models,
they appear powerless to explain why it is
that deep neural networks generalize in the first place.
Because deep neural networks are capable of fitting
arbitrary labels even for large datasets,
and despite the use of familiar methods such as $\ell_2$ regularization,
traditional complexity-based generalization bounds,
e.g., those based on the VC dimension
or Rademacher complexity of a hypothesis class,
cannot explain why neural networks generalize.
(:numref:`chap_classification_generalization` introduces these ideas;
the mechanics of the classical bounds---concentration of measure,
uniform convergence, and Rademacher complexity, with proofs---are
developed in :numref:`sec_mdl-concentration-generalization`.)

### Double Descent

The most striking of these phenomena deserves its own picture.
Classical theory predicts a *U-shaped* test-error curve:
as we add capacity, error first falls (we stop underfitting)
and then rises (we begin overfitting),
with a sweet spot in between
(recall :numref:`fig_capacity_vs_error` from :numref:`sec_generalization_basics`).
Deep networks do not obey this.
As we keep adding capacity past the *interpolation threshold*---the
point where the model can exactly fit the training set,
roughly where the parameter count matches the number of training
examples---the test error first spikes sharply
and then *descends a second time*,
often dropping below the classical sweet spot.
This non-monotone, two-valley shape is called *double descent*
:cite:`Belkin.Hsu.Ma.ea.2019`,
and it has been observed not only as we grow the model
but also as we train for more epochs or add more data
:cite:`nakkiran2021deep` (:numref:`fig_double_descent`).

![Classical bias--variance theory predicts the U-shaped curve on the left: test error falls, then rises as capacity grows, with a sweet spot in between. Deep networks follow it only up to the *interpolation threshold*, where the model can just fit the training set (here $\#$parameters $\approx \#$examples) and the test error spikes. Beyond it lies the *over-parametrized regime*, where adding still more capacity makes the test error *descend a second time*, often below the classical optimum. The training error (gray) falls monotonically to zero at the threshold and stays there.](../img/mdl-mlp-double-descent.svg)
:label:`fig_double_descent`

Why should bigger be better past the point of perfectly fitting the data?
In brief: at the threshold the model has *just barely* enough capacity,
so it is forced into a single, often jagged, high-variance interpolant;
past it there are *many* parameter settings that interpolate the data,
and the training procedure selects a smooth, small-norm one
that happens to generalize---"more capacity" becomes
"more room for the optimizer to find a simple interpolant."
The mathematics of this mechanism---why the spike sits exactly at the
threshold, why the *norm* of the solution rather than its parameter
count is the capacity that matters, and a reproduction of the entire
double-descent curve in twenty-five lines---is developed
in :numref:`sec_mdl-concentration-generalization`;
we return below to *why* gradient descent prefers such solutions.

Model size, moreover, is only one of three knobs that trace out this curve.
:citet:`nakkiran2021deep` document *model-wise* double descent (grow the
network, the flavor above and the one the appendix analyzes), *epoch-wise*
double descent (fix the network and train longer: test error falls, rises as
the model begins to interpolate noise, then falls again), and *sample-wise*
double descent, the genuinely shocking one: adding training examples can
*hurt* test performance, because more data moves the interpolation threshold
and can push a fixed model back into the high-variance spike. All three are
organized by a single axis that Nakkiran et al. call *effective model
complexity* — roughly, how many examples the full training procedure (model,
optimizer, *and* budget) can fit perfectly — with the error peaking wherever
that quantity crosses the dataset size. This chapter only names the
phenomena; the appendix proves the model-wise case, and the exercises below
let you produce the epoch-wise one yourself.

## Inspiration from Nonparametrics

Approaching deep learning for the first time,
it is tempting to think of them as parametric models.
After all, the models *do* have millions of parameters.
When we update the models, we update their parameters.
When we save the models, we write their parameters to disk.
However, mathematics and computer science are riddled
with counterintuitive changes of perspective,
and surprising isomorphisms between seemingly different problems.
While neural networks clearly *have* parameters,
in some ways it can be more fruitful
to think of them as behaving like nonparametric models.
So what precisely makes a model nonparametric?
While the name covers a diverse set of approaches,
one common theme is that nonparametric methods
tend to have a level of complexity that grows
as the amount of available data grows.

Perhaps the simplest example of a nonparametric model
is the $k$-nearest neighbor algorithm (we will cover more nonparametric models later, for example in :numref:`sec_attention-pooling`).
Here, at training time,
the learner simply memorizes the dataset.
Then, at prediction time,
when confronted with a new point $\mathbf{x}$,
the learner looks up the $k$ nearest neighbors
(the $k$ points $\mathbf{x}_i'$ that minimize
some distance $d(\mathbf{x}, \mathbf{x}_i')$).
When $k=1$, this algorithm is called $1$-nearest neighbors,
and the algorithm will always achieve a training error of zero.
That however, does not mean that the algorithm will not generalize.
In fact, it turns out that under some mild conditions,
the error of the $1$-nearest neighbor rule
comes within a factor of two of the optimal (Bayes) error
as the dataset grows :cite:`Cover.Hart.1967`,
and it is optimal in the noiseless case
where the Bayes error is zero.
(Full consistency — convergence to the optimal predictor —
requires $k$-nearest neighbors with $k \to \infty$
while $k/n \to 0$.)


Note that $1$-nearest neighbor requires that we specify
some distance function $d$, or equivalently,
that we specify some vector-valued basis function $\phi(\mathbf{x})$
for featurizing our data.
For any choice of the distance metric,
we will achieve zero training error
and eventually approach this near-optimal limiting behavior,
but different distance metrics $d$
encode different inductive biases
and with a finite amount of available data
will yield different predictors.
Different choices of the distance metric $d$
represent different assumptions about the underlying patterns
and the performance of the different predictors
will depend on how compatible the assumptions
are with the observed data.

In a sense, because neural networks are over-parametrized,
possessing many more parameters than are needed to fit the training data,
they tend to *interpolate* the training data (fitting it perfectly)
and thus behave, in some ways, more like nonparametric models.
More recent theoretical research has established
deep connection between large neural networks
and nonparametric methods, notably kernel methods.
In particular, :citet:`Jacot.Gabriel.Hongler.2018`
demonstrated that in the limit, as multilayer perceptrons
with randomly initialized weights grow infinitely wide,
they become equivalent to (nonparametric) kernel methods
for a specific choice of the kernel function
(essentially, a distance function),
which they call the neural tangent kernel.
While current neural tangent kernel models may not fully explain
the behavior of modern deep networks,
their success as an analytical tool
underscores the usefulness of nonparametric modeling
for understanding the behavior of over-parametrized deep networks.


## Early Stopping

While deep neural networks are capable of fitting arbitrary labels,
even when labels are assigned incorrectly or randomly
:cite:`zhang2021understanding`,
this capability only emerges over many iterations of training.
A new line of work :cite:`Rolnick.Veit.Belongie.Shavit.2017`
has revealed that in the setting of label noise,
neural networks tend to fit cleanly labeled data first
and only subsequently to interpolate the mislabeled data.
Moreover, it has been established that this phenomenon
translates directly into a guarantee on generalization:
whenever a model has fitted the cleanly labeled data
but not randomly labeled examples included in the training set,
it has in fact generalized :cite:`Garg.Balakrishnan.Kolter.Lipton.2021`.

Together these findings help to motivate *early stopping*,
a classic technique for regularizing deep neural networks.
Here, rather than directly constraining the values of the weights,
one constrains the number of epochs of training.
The most common way to determine the stopping criterion
is to monitor validation error throughout training
(typically by checking once after each epoch)
and to cut off training when the validation error
has not decreased by more than some small amount $\epsilon$
for some number of epochs.
This is sometimes called a *patience criterion*.
As well as the potential to lead to better generalization
in the setting of noisy labels,
another benefit of early stopping is the time saved.
Once the patience criterion is met, one can terminate training.
For large models that might require days of training
simultaneously across eight or more GPUs,
well-tuned early stopping can save researchers days of time
and can save their employers many thousands of dollars.

Notably, when there is no label noise and datasets are *realizable*
(the classes are truly separable, e.g., distinguishing cats from dogs),
early stopping tends not to lead to significant improvements in generalization.
On the other hand, when there is label noise,
or intrinsic variability in the label
(e.g., predicting mortality among patients),
early stopping is crucial.
Training models until they interpolate noisy data is typically a bad idea.


## Classical Regularization Methods for Deep Networks

In :numref:`chap_regression`, we described
several  classical regularization techniques
for constraining the complexity of our models.
In particular, :numref:`sec_weight_decay`
introduced a method called weight decay,
which consists of adding a regularization term to the loss function
in order to penalize large values of the weights.
Depending on which weight norm is penalized
this technique is known either as ridge regularization (for $\ell_2$ penalty)
or lasso regularization (for an $\ell_1$ penalty).
In the classical analysis of these regularizers,
they are considered as sufficiently restrictive on the values
that the weights can take to prevent the model from fitting arbitrary labels.

In deep learning implementations,
weight decay remains a popular tool.
However, researchers have noted
that typical strengths of $\ell_2$ regularization
are insufficient to prevent the networks
from interpolating the data :cite:`zhang2021understanding`.
Their benefit, *if* interpreted as classical regularization,
may therefore make sense only in combination with early stopping.
Absent early stopping, it is possible
that just like the number of layers
or number of nodes (in deep learning)
or the distance metric (in 1-nearest neighbor),
these methods may lead to better generalization
not because they meaningfully constrain
the power of the neural network
but rather because they somehow encode inductive biases
that are better compatible with the patterns
found in datasets of interest.
Thus, classical regularizers remain popular
in deep learning implementations,
even if the theoretical rationale
for their efficacy may be radically different.

### Implicit Regularization

A growing body of work suggests that the strongest regularizer
in deep learning may be one we never write down: the optimizer itself.
Among the many parameter settings that interpolate the training data,
stochastic gradient descent does not pick one at random.
Starting from a small initialization,
it is biased toward solutions with small norm and "flat" minima,
which tend to generalize.
Flatness can even be optimized for directly:
*sharpness-aware minimization* (SAM) minimizes the worst-case loss within a
small ball around the current weights rather than the loss itself, and this
one-line change to the update rule improves generalization across
architectures :cite:`Foret.Kleiner.Mobahi.ea.2021`.
This is not merely empirical.
For linearly separable data,
gradient descent on the logistic loss provably converges
to the *maximum-margin* (minimum-norm) separator,
even with no explicit penalty :cite:`Soudry.Hoffer.Nacson.ea.2018`.
This *implicit bias* helps explain why over-parametrized networks,
which classical theory says should overfit catastrophically,
instead find simple, generalizing solutions,
and why the interventions above (weight decay, early stopping)
help by *nudging* an already-benign bias
rather than by brute-force capacity control.
A striking illustration is *grokking*:
on small algorithmic tasks, networks first memorize the training set
and only much later, after many further steps of training,
suddenly generalize,
a reminder that optimization *dynamics*, not just architecture,
govern generalization :cite:`Power.Burda.Edwards.ea.2022`.
:numref:`fig_grokking` shows the signature:
training accuracy saturates almost immediately,
while validation accuracy sits at chance for orders of magnitude more steps
before snapping to near-perfect,
long after any conventional early-stopping rule would have given up.

![The grokking phenomenon, schematically, after :citet:`Power.Burda.Edwards.ea.2022`: on a small algorithmic task, training accuracy (gray) saturates within a few hundred steps, while validation accuracy (blue) lingers near chance for several further orders of magnitude of training before rising sharply. Between *memorization* and *generalization* (dashed markers) the network interpolates its training set yet has not found the generalizing solution; continued optimization, not additional capacity or data, is what eventually finds it.](../img/mdl-mlp-grokking.svg)
:label:`fig_grokking`

Notably, deep learning researchers have also built
on techniques first popularized
in classical regularization contexts,
such as adding noise to model inputs.
In the next section we will introduce
the famous dropout technique
(invented by :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`),
which has become a mainstay of deep learning,
even as the theoretical basis for its efficacy
remains similarly mysterious.


## Summary

Unlike classical linear models,
which tend to have fewer parameters than examples,
deep networks tend to be over-parametrized,
and for most tasks are capable
of perfectly fitting the training set.
This *interpolation regime* challenges
many hard-and-fast intuitions.
Functionally, neural networks look like parametric models.
But thinking of them as nonparametric models
can sometimes be a more reliable source of intuition.
Because it is often the case that all deep networks under consideration
are capable of fitting all of the training labels,
nearly all gains must come by mitigating overfitting
(closing the *generalization gap*).
Paradoxically, the interventions
that reduce the generalization gap
sometimes appear to increase model complexity
and at other times appear to decrease complexity.
However, these methods seldom decrease complexity
sufficiently for classical theory
to explain the generalization of deep networks,
and *why certain choices lead to improved generalization*
remains for the most part a massive open question
despite the concerted efforts of many brilliant researchers.


## Exercises

1. In what sense do traditional complexity-based measures fail to account for generalization of deep neural networks?
1. Why might *early stopping* be considered a regularization technique?
1. How do researchers typically determine the stopping criterion?
1. What important factor seems to differentiate cases when early stopping leads to big improvements in generalization?
1. Beyond generalization, describe another benefit of early stopping.
1. *Epoch-wise double descent.* Take the MLP of :numref:`sec_mlp-implementation` on Fashion-MNIST, randomly relabel 15% of the training examples, and train far past convergence (several hundred epochs), recording test error after every epoch. Plot test error against the epoch count on a log axis. Do you observe a second descent after the initial overfitting rise? How does the curve change with the label-noise fraction, and how does the epoch of the peak relate to when the model starts fitting the noisy labels? What does this imply for choosing an early-stopping patience?
1. (*) *Grokking.* Reproduce the setup of :citet:`Power.Burda.Edwards.ea.2022`: train a small network (they use a two-layer transformer, but a wide MLP on one-hot pairs also works) to predict $c = (a + b) \bmod 97$ from the pair $(a, b)$, using a random 50% of all pairs for training, with weight decay, for $10^5$ or more steps. Plot training and validation accuracy against the logarithm of the step count, and compare with :numref:`fig_grokking`. How does the delay before generalization change with the training fraction and with the weight-decay strength?

[Discussions](https://d2l.discourse.group/t/7473)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §5.5]{.kicker}

Why over-parametrized networks **generalize**<br>**double descent · the interpolation regime · implicit bias**.
:::
:::

::: {.slide title="Optimization is a means; generalization is the goal"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Fitting the training data is only the intermediate step. The real quest
is to find patterns that **predict well on unseen data**.

- For deep networks, fitting is **easy**: they can interpolate almost
  any training set, even random labels.
- So nearly every gain comes from **closing the generalization gap**, not
  from reducing training error.
- Yet *why* deep networks generalize remains a **major open question**.
:::

::: {.col .fig .big}
![Bigger past the interpolation threshold is *better*, not worse: the deep-learning surprise this section explains.](../img/mdl-mlp-double-descent.svg){width=100%}
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Overfitting, revisited]{.dtitle}

[inductive bias and the interpolation regime]{.dsub}
:::
:::

::: {.slide title="No free lunch: every model needs a bias"}
[Overfitting]{.kicker}

::: {.cols .vc}
::: {.col}
No algorithm generalizes better on **all** distributions. Given finite
data, a model must lean on **inductive biases** that prefer some
solutions over others.

- A deep MLP is biased toward functions built by **composing simpler
  ones**.
- Good biases match how the world is structured.

::: {.d2l-note}
The **generalization gap** is the difference between test and training
error. A large gap means we have **overfit**.
:::
:::

::: {.col .narrow}
::: {.d2l-note .rule}
**Classical recipe** to close the gap: make the model *less* complex.

- fewer features
- fewer nonzero parameters
- smaller parameters
:::
:::
:::
:::

::: {.slide title="Deep networks break the classical picture"}
[Overfitting]{.kicker}

Modern networks are expressive enough to **perfectly fit every training
example**, even millions of them, even random labels.

. . .

So the classical levers behave strangely:

- All candidate architectures already reach **zero training error**, so
  the only avenue left is reducing overfitting.
- Making a model **more** expressive (more layers, nodes, epochs) often
  *lowers* test error.

. . .

::: {.d2l-note .warn}
Because nets fit arbitrary labels, **VC-dimension and Rademacher bounds
cannot explain** why they generalize at all.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Double descent]{.dtitle}

[the U-curve, and what lies past it]{.dsub}
:::
:::

::: {.slide title="The classical bias-variance tradeoff"}
[Double descent]{.kicker}

::: {.cols .vc}
::: {.col}
Classical theory predicts a **U-shaped** test-error curve as capacity
grows:

$$\underbrace{\text{error}}_{\text{test}} \;=\;
\underbrace{\text{bias}^2}_{\downarrow\ \text{with capacity}} \;+\;
\underbrace{\text{variance}}_{\uparrow\ \text{with capacity}}.$$

Too little capacity **underfits**; too much **overfits**; a **sweet
spot** sits in between.
:::

::: {.col .narrow}
::: {.d2l-note}
This is the left half of the picture on the next slide, and it is *all*
the classical story predicts.
:::
:::
:::
:::

::: {.slide title="Double descent: the curve has two valleys" layout="figure"}
![Classical theory predicts the U-shaped curve on the left: test error falls, then rises, with a sweet spot between. Deep networks follow it only up to the *interpolation threshold*, where the model can just fit the training set ($\#$parameters $\approx \#$examples) and the test error spikes. Past it lies the *over-parametrized regime*, where adding still more capacity makes the test error *descend a second time*, often below the classical optimum. Training error (gray) falls to zero at the threshold and stays there.](../img/mdl-mlp-double-descent.svg){width=70%}
:::

::: {.slide title="Why bigger is better past the threshold"}
[Double descent]{.kicker}

::: {.cols .vc}
::: {.col}
At the **interpolation threshold** the model has *just barely* enough
capacity to fit the data, forcing a single, jagged, high-variance
solution.

Past it, **many** parameter settings interpolate the data. Gradient
descent from a small initialization selects among them, favoring
**smooth, small-norm** solutions that generalize.

::: {.d2l-note .rule}
Reframe "more capacity" not as "more overfitting" but as **more room for
the optimizer to find a simple interpolant**.
:::
:::

::: {.col .narrow}
Three knobs trace the same curve (Nakkiran et al. 2021):

- **model-wise** — grow the network;
- **epoch-wise** — train longer;
- **sample-wise** — *more data can hurt*, by pushing a fixed model back
  into the spike.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Why gradient descent finds simple solutions]{.dtitle}

[implicit bias, grokking, and the kernel view]{.dsub}
:::
:::

::: {.slide title="The strongest regularizer is the optimizer"}
[Implicit bias]{.kicker}

Among all settings that interpolate the data, **SGD does not pick one at
random**. From a small start it drifts toward small-norm, **flat**
minima.

. . .

This is provable, not just empirical:

::: {.d2l-note .rule}
On **separable** data, gradient descent on the logistic loss converges to
the **maximum-margin** (minimum-norm) separator, with **no** explicit
penalty.
:::

So explicit tricks (weight decay, early stopping) **nudge** an
already-benign bias rather than brute-forcing capacity down.
:::

::: {.slide title="Grokking: dynamics, not just architecture"}
[Implicit bias]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
On small algorithmic tasks, a network first **memorizes** the training
set, with test accuracy flat near chance.

Then, after **many more** steps at zero training loss, it **suddenly
generalizes**: test accuracy jumps late.

::: {.d2l-note .warn}
Same architecture, same data, same loss: only **time spent training**
changes the outcome. Dynamics, not architecture alone, govern
generalization.
:::
:::

::: {.col .fig .big}
![Training accuracy (gray) saturates early; validation accuracy (blue) lingers near chance for orders of magnitude more steps, then rises sharply (after Power et al., 2022).](../img/mdl-mlp-grokking.svg){width=100%}
:::
:::
:::

::: {.slide title="A nonparametric lens: the neural tangent kernel"}
[Implicit bias]{.kicker}

::: {.cols .vc}
::: {.col}
Networks *have* parameters, yet often behave like **nonparametric**
models, whose complexity grows with the data (think $k$-nearest
neighbors, which **memorizes** and still generalizes).

As a randomly initialized MLP grows **infinitely wide**, it becomes
equivalent to a **kernel method** for a fixed kernel:

::: {.d2l-note .rule}
the **neural tangent kernel**, a distance function the architecture
induces.
:::
:::

::: {.col .narrow}
Over-parametrized nets *interpolate* the data, so the nonparametric
intuition is often the more reliable one.
:::
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Regularization in practice]{.dtitle}

[early stopping and classical penalties, reinterpreted]{.dsub}
:::
:::

::: {.slide title="Early stopping: fit clean data first"}
[Practice]{.kicker}

::: {.cols .vc}
::: {.col}
With label noise, networks fit the **cleanly labeled** examples first and
only later interpolate the **mislabeled** ones.

Stop while the clean data is fit but the noise is not, and you have
provably **generalized**.

::: {.d2l-note}
**Patience criterion:** monitor validation error each epoch; stop when it
fails to improve by $\epsilon$ for a few epochs. It also saves training
time and money.
:::
:::

::: {.col .narrow}
::: {.d2l-note .warn}
Matters most under **label noise** or intrinsic label variability. On
clean, separable data it changes little.
:::
:::
:::
:::

::: {.slide title="Classical penalties, reinterpreted"}
[Practice]{.kicker}

Weight decay adds a norm penalty to the loss:

$$\ell_2\ (\text{ridge}):\ \lambda\lVert\mathbf{w}\rVert_2^2
\qquad
\ell_1\ (\text{lasso}):\ \lambda\lVert\mathbf{w}\rVert_1.$$

. . .

But at typical strengths, $\ell_2$ is **too weak** to stop a network from
interpolating the data.

::: {.d2l-note}
So its benefit may come not from **constraining capacity** but from
**encoding a better inductive bias**, much like the choice of distance
in $k$-NN. Often it helps only **together with early stopping**. (Dropout
is the next such tool.)
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- Deep nets are **over-parametrized**: they interpolate the training set,
  so gains come from **closing the generalization gap**.
- **Double descent:** past the interpolation threshold, more capacity
  makes test error **descend again**; the classical U-curve is only half
  the story.
- Classical complexity bounds **cannot** explain this.
:::

::: {.col}
- **Implicit bias:** SGD prefers smooth, small-norm interpolants (provably
  max-margin on separable data): the optimizer is the regularizer.
- **Nonparametric view:** infinitely wide nets become **kernel methods**.
- **In practice:** early stopping and weight decay help by **nudging**
  that bias, not by brute-force capacity control.
:::
:::

::: {.d2l-note}
Why certain choices improve generalization remains, for the most part, a
**massive open question**.
:::
:::
