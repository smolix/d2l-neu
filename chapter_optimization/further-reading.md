# Further Reading
:label:`sec_optimization-further-reading`

A chapter that teaches one representative of each idea leaves a good deal
out. Ours took one valley, one stochastic method, one adaptive method, one
matrix optimizer, and one rule each for schedules, batch size, and width.
You may ask what lies next to each of them. This section answers, and it
is not meant to be read from top to bottom.

If you follow only three threads, take these. To see what the loss of a
network looks like beyond a valley, read about connected and flat minima
(:numref:`subsec_optimization-fr-connected-minima` and
:numref:`subsec_optimization-fr-flat-minima`). To see what became of
Newton's method on networks, read about quasi-Newton methods and Newton
steps without the Hessian (:numref:`subsec_optimization-fr-quasi-newton`
and :numref:`subsec_optimization-fr-hessian-free`). And before you believe
the next report that some optimizer beats AdamW, read "Benchmarks and
Their Protocols" (:numref:`subsec_optimization-fr-benchmarks`). The rest
can wait for the day you meet the problem that an entry solves. The
entries on Muon describe a field that changed from month to month while we
wrote: read them for the ideas, and expect the numbers to have moved.

Each entry says which problem it addresses and what it changes relative to
the chapter, and each can be read without the others. Resources for what
the chapter itself teaches are in its introduction
(:numref:`chap_optimization`), theory texts in the appendix
(:numref:`sec_mdl-optimization-further-reading`), automated hyperparameter
search in :numref:`chap_hyperopt`, and ways to shrink the memory of an
optimizer in :numref:`sec_performance-further-reading`. Nothing else in
the book depends on this section.

## The Loss Surface

### Pictures of a High-Dimensional Loss

:numref:`sec_optimization-intro` could draw its valley because the valley
has two coordinates. For a network the usual substitute is a plot of the
loss along one or two random directions in the space of weights. Such
plots mislead unless each direction is rescaled to the weights it
perturbs. A ReLU network computes the same function after one layer is
multiplied by $c > 0$ and the next by $1/c$, yet a perturbation of a given
size is large against the layer that shrank. The same minimum looks sharp
in one network and flat in its rescaled twin.

:citet:`Li.Xu.Taylor.ea.2018` therefore rescale a direction piece by
piece: in a convolutional network, each filter of the direction gets the
norm of the filter that it perturbs. Deep networks without skip
connections then show chaotic surfaces, and with residual connections
(:numref:`sec_resnet`) smooth, nearly convex basins. Keep in mind what
such a plot is, two directions out of millions. A saddle point whose
downhill directions lie outside the plane looks like a minimum in it.

### Connected Minima
:label:`subsec_optimization-fr-connected-minima`

The permutation argument of :numref:`sec_optimization-intro` showed that a
network has many minima that compute the same function, and that the
average of all these copies is in general worse than any one of them.
Minima reached from different random starts look isolated in the same way:
on the straight line between two of them the loss rises. They are not
isolated. Surprisingly, a curve as simple as a polygonal chain with one
bend joins two of them with low training and test loss all the way
:cite:`Garipov.Izmailov.Podoprikhin.ea.2018` (see also
:cite:`Draxler.Veschgini.Salmhofer.ea.2018`).

Why, then, the barrier on the straight line?
:citet:`Entezari.Sedghi.Saukh.ea.2022` conjectured that it is largely the
permutation symmetry itself, and that almost no barrier remains once the
hidden units of one network are permuted to match those of the other.
:citet:`Ainsworth.Hayase.Srinivasa.2023` found such permutations. For wide
networks the barrier nearly vanished, for narrow ones it stayed. Methods
that merge separately trained models by averaging their weights rest on
this alignment.

### Flat Minima and Sharpness-Aware Minimization
:label:`subsec_optimization-fr-flat-minima`

:numref:`sec_optimization-intro` noted that the minima of a network do not
all generalize equally well. A long-standing hypothesis says which ones
do: those in flat regions, where a small perturbation of the weights
barely changes the loss. Sharpness-aware minimization
:cite:`Foret.Kleiner.Mobahi.ea.2021` turns the hypothesis into an
optimizer. Before each descent step it takes a step of fixed length up the
gradient, and it applies the gradient computed there to the original
weights, which lowers the largest loss in a small ball around them. The
authors report better test accuracy for image classifiers. A step costs
two gradients, though, so the baseline to beat is a plain run of twice the
length.

The premise itself is disputed. A ReLU network computes the same function
after one layer is multiplied by $c > 0$ and the next by $1/c$.
:citet:`Dinh.Pascanu.Bengio.ea.2017` used this rescaling to make a minimum
arbitrarily sharp without changing the function it computes, so that
sharpness, unlike generalization, depends on how the function is
parametrized.

## Classical Methods

### Quasi-Newton Methods
:label:`subsec_optimization-fr-quasi-newton`

Newton's method (:numref:`sec_gd`) needs the Hessian and a linear solve
with it at every step. Over a short step, however, the gradient changes by
about the Hessian times the step, so every step shows how the Hessian acts
on one direction. Quasi-Newton methods build an approximation of the
inverse Hessian from these observations and need only gradients. BFGS,
the best known and named after its four authors, keeps the approximation
as a $d \times d$ matrix. Its limited-memory form, L-BFGS
:cite:`Liu.Nocedal.1989`, keeps the last $m$ pairs of steps and gradient
changes instead, $O(md)$ numbers. For smooth, deterministic problems of
moderate size, such as logistic regression on data that fits in memory, it
is the method to try first. In deep learning it is rare, because the
difference of two minibatch gradients a short step apart is mostly noise.
:numref:`subsec_mdl-quasi-newton` derives the update, and
:citet:`Nocedal.Wright.2006` treat the whole family.

### Line Search and Trust Regions

The chapter fixes the learning rate or schedules it in advance. A *line
search* chooses it anew at every step: it tries a step along the descent
direction, accepts it if the objective fell by a fixed fraction of what
the gradient predicts, and shrinks it otherwise. Each trial costs an
evaluation of the objective, and on a minibatch the test itself is noisy,
so training does without it.

A *trust-region method* bounds the length of the step instead. It
minimizes a quadratic model of the loss inside a ball and adjusts the
radius by how well the model predicted the decrease. The ball makes
negative curvature harmless. Where the model curves downward it has no
minimum, and the Newton step of :numref:`sec_gd` went to its highest point
instead. Inside a ball the model always has a lowest point, on the
boundary if need be. The appendix treats both methods
(:numref:`subsec_mdl-gd-smoothness` and :numref:`subsec_mdl-trust-region`).

The trust region left a trace in the chapter. The steepest step of
:numref:`sec_muon` minimizes a linear model of the loss inside a ball: a
trust-region step with the simplest of models and a radius fixed in
advance, the learning rate. :citet:`Kovalev.2025` analyzes Muon in this
way.

### Newton Steps Without the Hessian
:label:`subsec_optimization-fr-hessian-free`

:numref:`sec_gd` remarked that Newton's method needs only products of the
Hessian with vectors, each about as costly as two gradients. It is out of
reach for networks all the same: a single step needs many such products,
and curvature estimated on a minibatch is unreliable. Two serious attempts
were made nevertheless.

Hessian-free optimization :cite:`Martens.2010` finds the Newton step by a
few iterations of *conjugate gradients*, a solver for linear systems that
needs only such products. For safety it replaces the Hessian by an
approximation without negative eigenvalues and adds a multiple of the
identity, which shortens the step. Saddle-free Newton
:cite:`Dauphin.Pascanu.Gulcehre.ea.2014` goes further and replaces the
eigenvalues of the Hessian by their sizes, so that a direction of negative
curvature repels the iterate where it attracted Newton's method. For a
network this can be done only on a small subspace, spanned by the gradient
and its repeated products with the Hessian.

Neither method stayed in use. What stayed is the least expensive form of
the same two ideas: a preconditioner that is diagonal and positive, built
from the sizes of gradients. We know it as Adam (:numref:`sec_adam`).

### Rates for Stochastic Gradient Descent

:numref:`sec_sgd` quoted a single rate: on a loss that curves upward in
every direction, learning rates that decay as $c/t$ bring the expected
squared distance to the minimum down as $1/t$. Without that curvature less
can be promised. On a convex loss with bounded gradients, SGD with a
learning rate of order $1/\sqrt{T}$ comes within order $1/\sqrt{T}$ of the
minimal loss after $T$ steps, if its iterates are averaged. Half the error
costs four times the steps, and no method that sees only stochastic
gradients does better in the worst case :cite:`Nemirovski.Yudin.1983`.

Neither rate applies to a network, whose loss is not convex. There only a
small gradient can be promised, not a minimum. On a smooth loss the
expected squared norm of the gradient, at an iterate picked at random from
the run, also falls as $1/\sqrt{T}$ :cite:`Ghadimi.Lan.2013`, and a saddle
point satisfies this promise as well as a minimum does. The appendix
proves the two upper bounds (:numref:`subsec_mdl-fixed-budget-rates` and
:numref:`subsec_mdl-nonconvex-sgd`) and quotes the lower bound.

Averaging has a second, finer use: where the loss does curve upward, the
average of the iterates under slowly decaying learning rates attains the
best possible asymptotic error, constant included
:cite:`Polyak.Juditsky.1992`. The weight averaging of
:numref:`subsec_practice-weight-averaging` descends from this result.

### Variance Reduction

SGD stalls in a noise ball (:numref:`sec_sgd`) because its estimate of the
gradient stays noisy at the minimum, where the full gradient vanishes and
the gradients of single examples do not. On a finite training set this
noise can be removed. SVRG, the stochastic variance-reduced gradient
method :cite:`Johnson.Zhang.2013`, computes the full gradient at a
reference point from time to time and adds to it the change in the
gradient of the current minibatch since that point. The variance of this
estimate shrinks as both points approach the minimum, and a constant
learning rate then converges geometrically on a loss that curves upward in
every direction. The theory is clean
(:numref:`sec_mdl-variance-reduction`), but on deep networks these methods
have not given consistent improvements over SGD with momentum. There the
reference goes stale too quickly, for instance under data augmentation.

## Matrix Optimizers

:numref:`sec_muon` reached its matrix optimizer by two roads, steepest
descent under the spectral norm and Shampoo with its memory switched off.
The first entry below follows the older road. The others are about Muon as
it has developed since 2024. Most of that work has had a modest aim, to
make the orthogonalization cheaper. The question that matters most is
open: how much of Muon's lead survives in the largest models.

### Kronecker-Factored Preconditioners

Adam's preconditioner is diagonal, and Shampoo (:numref:`sec_muon`) keeps
one small matrix per side of each weight matrix. K-FAC
:cite:`Martens.Grosse.2015` had arrived at the same shape earlier, from a
question about distance. How far does a step move a model? Counted in
weights, the answer depends on the parametrization. Counted in predictions
it does not, and the *natural gradient* :cite:`Amari.1998` is the steepest
step under that measure. It preconditions the gradient with the inverse of
the *Fisher information matrix*, the covariance of the gradients of the
log-likelihood, which for a network is as far out of reach as the Hessian.
K-FAC approximates the block of one layer by a Kronecker product of two
small matrices, one for the inputs of the layer and one for the gradients
at its outputs (:numref:`subsec_mdl-preconditioning-ladder`), and only
these have to be inverted.

SOAP :cite:`Vyas.Morwani.Zhao.ea.2024` joins the two families by running
Adam in the basis of eigenvectors of Shampoo's two matrices. In the
matched-tuning study that :numref:`sec_muon` quotes
:cite:`Wen.Hall.Ma.ea.2025`, SOAP, with Shampoo's matrices and Adam's
averages to remember, and Muon, with a momentum buffer alone, are both
among the fastest methods. How much a matrix optimizer has to remember is
not settled.

### Muon's Lead over AdamW

The matched-tuning study that :numref:`sec_muon` quoted found Muon ahead
of AdamW in steps, by a margin that shrank as the models grew
:cite:`Wen.Hall.Ma.ea.2025`. Where does the lead come from?
:citet:`Wang.Zhang.Li.ea.2025` find it concentrated in the value and
output projections of attention and in the MLP weights. In their model of
such a matrix, a linear memory that stores pairs of an input and its
answer, Muon gains on the pairs that occur rarely. :numref:`sec_adam`
reported the same of Adam against SGD: on tokens that are each rare and
together much of the text, gradient descent makes almost no progress and
Adam does :cite:`Kunstner.Yadav.Milligan.ea.2024`. Muon's gain over Adam,
it turns out, is of the same kind as Adam's gain over SGD: progress on
what is rare.

:citet:`Shah.Polloreno.Stratos.ea.2025` report a second advantage, which a
count of steps at one batch size does not show. Muon stays efficient in
its use of data at batch sizes well beyond the critical batch size of
AdamW, the size above which most of the extra work of a larger batch saves
nothing (:numref:`sec_batch_size`). Neither finding answers the question
of scale. The matched study ends at a little over a billion parameters.
Far larger models have been trained with Muon, but we know of no matched
comparison at their size.

### A Norm for Every Layer

:numref:`sec_muon` derived Muon as steepest descent under the spectral
norm, and then handed the embeddings, the output layer, and all vectors
back to AdamW. Scion :cite:`Pethick.Xie.Antonakopoulos.ea.2025` finishes
the argument by assigning a norm to every layer: the spectral norm for
hidden matrices, the largest change of any single embedding vector for
embeddings, and a sign or row-normalized step for the output layer. No
Adam is needed. Decoupled weight decay gets a task of its own in this
view. A step of bounded norm followed by a shrinkage keeps the weights
inside a ball of that norm, of radius proportional to $1/\lambda$. Without
decay nothing bounds them, which is what the long runs of
:numref:`sec_muon` ran into.

### Cheaper Orthogonalization

Newton--Schulz, the iteration with which Muon orthogonalizes, adds about
an eighth to the arithmetic of the forward and backward passes on
`TinyLM`, the small language model of this chapter
(:numref:`subsec_tinylm`). At the batch sizes of large runs it adds under
one percent (:numref:`subsec_muon-newton-schulz`). Work on its cost is
therefore work on the last few percent. Polar Express
:cite:`Amsel.Persson.Musco.ea.2026`, the third curve of
:numref:`fig_opt_ns_maps`, comes closer to an exact orthogonalization in
the same number of steps. It uses a different quintic at each step, chosen
for the interval that the singular values can still occupy. An earlier
heuristic search :cite:`Cesista.You.Jordan.2025` announces the size of
such gains in its title: one to two percent in efficiency. Gram
Newton--Schulz :cite:`Zhang.Amsel.Chen.ea.2026` saves arithmetic instead.
For a matrix with fewer rows than columns it iterates on the small
symmetric matrix of the inner products of the rows.

One cost is of another kind. The weights of a large model are split, or
*sharded*, across many devices. AdamW updates every entry on its own, so
each device can update its part of a matrix alone. Newton--Schulz needs
the whole matrix, which must be gathered on one device, orthogonalized,
and scattered back. Dion :cite:`Ahn.Xu.Abreu.ea.2025` avoids the
gathering: it orthogonalizes only a rank-$r$ approximation of the
momentum, for which the devices exchange $O((m + n)r)$ numbers for an
$m \times n$ matrix, and leaves the remainder in the buffer for later
steps.

How much accuracy may be given up? In theory little: with $q$
Newton--Schulz steps Muon converges at the rate of exact orthogonalization
up to a factor that approaches one doubly exponentially fast in $q$
:cite:`Kim.Oh.2026`. The theorem speaks of the rate, though, and not of
the length of the step that five iterations deliver. On `TinyLM` we
measured about nine tenths of the norm of the exact answer in the middle
of a run, and little more than half for some matrices of a new network
(:numref:`sec_muon`). The inexact iteration thus acts as a smaller
learning rate during the first steps.

## Schedules, Averages, and Batches

### Warm Restarts

Cosine decay (:numref:`subsec_scheduler-decay-shapes`) was proposed
together with periodic restarts :cite:`Loshchilov.Hutter.2016`: at the end
of each cosine wave the learning rate jumps back to its peak. The restarts
were meant to give a run a good model to show at any time. Each wave ends
with a decayed model, so one run yields several finished models. Huang et
al. (2017) made a method of this under the name *snapshot ensembles*: save
the model at the end of each wave and average the predictions of the saved
models. Predictions, not weights. After a jump back to the peak rate the
next wave need not end in the basin of the last, and on the straight line
between two basins the loss rises
(:numref:`subsec_optimization-fr-connected-minima`). Today the cosine wave
is in general use and the restarts are not. Warmup–stable–decay
(:numref:`subsec_scheduler-branching`) gets its several models without
ever returning to a high rate.

### Schedule-Free Optimization

Decay removes the noise ball at the end of training by shrinking the late
steps. Averaging the iterates (:numref:`subsec_practice-weight-averaging`)
removes it from the parameters that are evaluated. Schedule-free SGD and
AdamW :cite:`Defazio.Yang.Mehta.ea.2024` build training around the
average: they keep a constant learning rate, evaluate each gradient at an
interpolation between the current iterate and the running average of all
iterates, and return the average. No training horizon is needed, so
training can stop, or continue, at any step. Schedule-free AdamW won the
self-tuning track of the AlgoPerf competition
(:numref:`subsec_optimization-fr-benchmarks`). The exercise on
schedule-free SGD in :numref:`sec_scheduler` implements the recurrence.

### Model Soups

:numref:`subsec_practice-weight-averaging` averaged the iterates of one
run. Model soups :cite:`Wortsman.Ilharco.Gadre.ea.2022` average the
weights of several models fine-tuned from the same pretrained model with
different hyperparameters, and the soup often beats the best single model.
It works because models fine-tuned from a shared start lie in one basin.
Models trained from different random starts do not, and averaging them
fails unless their units are matched first
(:numref:`subsec_optimization-fr-connected-minima`).

### Scaling Rules and Batch Ramps

The square-root rule of :numref:`sec_batch_size` moved Adam's learning
rate with the batch and nothing else. The analysis behind the rule asks
for more :cite:`Malladi.Lyu.Panigrahi.ea.2022`. When the batch grows
$k$-fold, $1 - \beta_1$ and $1 - \beta_2$ should grow $k$-fold as well, so
that each of Adam's averages still spans the same number of examples, and
$\epsilon$ should shrink by $\sqrt{k}$. Practice mostly ignores these
conditions. They are the first thing to check when a large change of the
batch size goes wrong.

The same section saw the noise scale, the batch size at which a minibatch
gradient holds as much noise as signal, grow during a run. A batch that
is wasteful early in a run can then be useful later.
:citet:`Zhang.Morwani.Vyas.ea.2024` found that the critical batch size,
above which most of the extra work of a larger batch saves nothing, grows
with the data a run has seen and hardly at all with the size of the model.
:citet:`Merrill.Arora.Groeneveld.ea.2025` measure it without a noise
estimate. From a checkpoint they branch short runs at larger batch sizes,
and they double the batch whenever the larger batch still makes the same
progress per example. The batch then ramps up over the run, much as the
learning rate does in a warmup.

## Large Runs and Benchmarks

### Hyperparameter Transfer in Production

:numref:`sec_scaling` named two ways to carry a learning rate to a wider
model: muP, which writes the width into per-layer rules, and a drift
fitted under the standard parametrization. Cerebras-GPT
:cite:`Dey.Gosal.Chen.ea.2023` applied muP to a whole family of
transformers: the hyperparameters were tuned once, on a proxy of about 40
million parameters, and transferred to the larger models. DeepSeek
:cite:`Bi.Chen.Chen.ea.2024` took the second way, with the fitted power
laws that :numref:`sec_scaling` quoted. The third case is less satisfying.
Meta reports that Llama 4
:cite:`Meta.2025` was trained with per-layer learning rates and
initialization scales that transfer across width, depth, batch size, and
training length. Unfortunately the method is not published. Transfer in
production now extends beyond width, the one direction the chapter covers.

### Benchmarks and Their Protocols
:label:`subsec_optimization-fr-benchmarks`

How optimizers are compared decides which one wins. Consider LARS
:cite:`You.Gitman.Ginsburg.2017` and LAMB, its counterpart for Adam, which
rescale the update of each layer by the ratio of its weight norm to its
update norm. They were proposed to make very large batches trainable. When
:citet:`Nado.Gilmer.Shallue.ea.2021` tuned the nuisance hyperparameters of
plain momentum and Adam with the care that the new methods had received,
the old optimizers matched them. This is the protocol of
:numref:`sec_test_models` at work: every method gets a grid of its own.

Organized benchmarks fix the protocol in advance, and still they disagree.
AlgoPerf :cite:`Dahl.Schneider.Nado.ea.2023` measures the time to reach a
target on a fixed set of workloads, with hyperparameters either tuned
within a fixed budget of trials or set by the method itself. In its first
competition :cite:`Kasimbeg.Schneider.Eschenhagen.ea.2025`, distributed
Shampoo won the tuned track and schedule-free AdamW the self-tuning track.
In a matched-tuning study of language models
:cite:`Semenov.Pagliardini.Jaggi.2025`, AdEMAMix, which adds a slow second
momentum average to AdamW, was among the strongest methods throughout. And
Muon was first demonstrated on the NanoGPT speedrun
:cite:`Jordan.2024,Jordan.Jin.Boza.ea.2024`, a public competition to train
a language model to a fixed validation loss on fixed data and fixed
hardware in the least wall-clock time. Muon's first record there cut the
time by about a fifth. Four comparisons, and no two favor the same method.

The speedrun also shows what becomes of a benchmark that fixes too little.
Its headline time has since fallen far further, from changes to the
optimizer, the architecture, and the systems together, so it is a record
of engineering rather than a benchmark of optimizers. Its optimization
track :cite:`Jordan.2026` is the repair. It fixes model, data, and batch,
counts the steps to a target loss, and accepts an entry only if its runs
lie below the target with statistical significance.

The track has already repeated what happened to LARS. An orthogonalized
update moves every singular direction equally far. In a matrix with more
rows than columns it still moves some output units, the rows, much farther
than others. NorMuon :cite:`Li.Liu.Liang.ea.2026` applies Adam's remedy
one level up: it keeps a running average of the squared update per row and
divides each row by its root. Its authors report a gain over Muon. On the
track a retuned plain Muon needs as many steps as NorMuon.

Read a ranking together with its protocol: who tuned the baseline, at what
model size (:numref:`sec_muon`), and whether the learning rate had decayed
when the result was read (:numref:`sec_scheduler`).

### A Training Logbook

Reports of large runs give the final recipe. The work of keeping a run
alive is rarely published. The OPT team :cite:`zhang2022opt` released its
logbook with its models: restarts after loss spikes and hardware failures,
and learning rates lowered in mid-run, day by day. It shows what remedies
such as the rewind of :numref:`sec_practice` cost. No other document in
this section is as close to what training a large model is like from one
day to the next.

[Discussions](https://d2l.discourse.group/t/36851)
