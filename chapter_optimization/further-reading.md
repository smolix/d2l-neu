# Further Reading
:label:`sec_optimization-further-reading`

The sections of this chapter teach one representative of each idea: one
valley, one stochastic method, one way to average over time, one adaptive
method, one matrix optimizer, and one set of rules for batch size and width.
This section points to topics they had no room for: other views of the loss
surface, the classical methods that deep learning rarely uses, further
preconditioners and faster ways to orthogonalize, other schedules and
averages, and the evidence from large training runs and benchmarks. Each
topic states the problem it addresses and what it changes relative to the
chapter. Implementations, tutorials, and courses for the material that the
chapter does teach are listed in its introduction,
:numref:`chap_optimization`. Nothing else in the book depends on this
section.

## The Loss Surface

### Pictures of a High-Dimensional Loss

:numref:`sec_optimization-intro` drew its landscapes in one and two
dimensions, where every direction can be shown. A network has millions, and
the common remedy is to plot the loss along one or two random directions
through the trained weights. Such plots mislead unless the directions are
rescaled to the size of the weights they perturb: a ReLU network computes
the same function after one layer is multiplied by $c$ and the next by
$1/c$, and an unnormalized direction then looks sharp in one network and
flat in its rescaled twin. :citet:`Li.Xu.Taylor.ea.2018` normalize each
random direction filter by filter to the norm of the corresponding filter.
With this normalization, deep networks without skip connections show
chaotic, nonconvex surfaces, and the same networks with the residual
connections of :numref:`sec_resnet` show smooth, nearly convex basins.

### Connected Minima

The permutation argument of :numref:`sec_optimization-intro` showed that a
network has many minima that compute the same function, and the average of
two of them is usually worse than either. Minima reached from different
random starts, not related by a permutation, also look isolated along the
straight line between them, where the loss rises in between.
:citet:`Garipov.Izmailov.Podoprikhin.ea.2018` and
:citet:`Draxler.Veschgini.Salmhofer.ea.2018` found that such minima are
nonetheless joined by simple curves, a polygonal chain with one bend or a
quadratic Bézier curve, along which the training and test loss stay low.
The barrier on the straight line is then largely the permutation symmetry
itself: :citet:`Entezari.Sedghi.Saukh.ea.2022` conjectured that once the
hidden units of one network are permuted to match the other, the straight
line has almost no barrier, and :citet:`Ainsworth.Hayase.Srinivasa.2023`
found such permutations by matching weights or activations and confirmed
the conjecture for wide networks. This alignment is the basis of methods
that merge separately trained models by averaging their weights.

### Flat Minima and Sharpness-Aware Minimization

A long-standing hypothesis holds that minima in flat regions generalize
better than sharp ones, because a small perturbation of the weights barely
changes the loss there. Sharpness-aware minimization
:cite:`Foret.Kleiner.Mobahi.ea.2021` turns the hypothesis into an
optimizer. Before each descent step it takes a step of fixed length up the
gradient, and it applies the gradient computed at that perturbed point to the
original weights. It thereby minimizes the largest loss in a small ball
around the weights rather than the loss at one point, at twice the gradient
cost per step. The reported gains are largest in image classification and
fine-tuning. The premise itself is debated:
:citet:`Dinh.Pascanu.Bengio.ea.2017` showed with the rescaling of the
previous paragraphs that a minimum of a ReLU network can be made arbitrarily
sharp without changing the function it computes, so that sharpness, unlike
generalization, depends on how the function is parametrized.

## Classical Methods

### Quasi-Newton Methods

Newton's method of :numref:`sec_gd` needs the Hessian and a linear solve
with it at every step. Quasi-Newton methods build an approximation of the
inverse Hessian from the differences between successive gradients and
iterates, so they need only gradients. BFGS updates a dense $d \times d$
matrix, and limited-memory BFGS :cite:`Liu.Nocedal.1989` keeps only the last
$m$ pairs of differences and applies the approximation implicitly, at
$\mathcal{O}(md)$ memory instead of $\mathcal{O}(d^2)$. L-BFGS is the
default method for smooth, deterministic problems of moderate size, such as
logistic regression on a dataset that fits in memory. It is rare in deep
learning because minibatch noise corrupts the gradient differences from
which it estimates curvature. :numref:`subsec_mdl-quasi-newton` derives the
BFGS update, and :citet:`Nocedal.Wright.2006` treat the whole family.

### Line Search and Trust Regions

The chapter fixes the learning rate or schedules it in advance. A *line
search* chooses it anew at every step: it tries a step along the descent
direction, accepts it if the objective has fallen by at least a fixed
fraction of what the gradient predicts (the Armijo condition), and shrinks
it otherwise. The step then adapts to the local curvature without knowing
it, but each trial costs an evaluation of the objective, which for a
training loss is a pass over the data, and a minibatch estimate makes the
test itself noisy. A *trust-region method* bounds the length of the step
instead of scaling the gradient: it minimizes the quadratic model of the
loss inside a ball, and grows the ball when the model predicted the
decrease well and shrinks it when it did not. Inside a ball, negative
curvature is harmless. This repairs the failure of the raw Newton step of
:numref:`sec_gd`, which climbed to a local maximum. The appendix treats both
(:numref:`subsec_mdl-gd-smoothness` and :numref:`subsec_mdl-trust-region`),
as do :citet:`Nocedal.Wright.2006` and :citet:`Boyd.Vandenberghe.2004`.

### Newton Steps Without the Hessian

A network's Hessian cannot be stored, but its product with a vector costs
about as much as two gradients. Hessian-free optimization
:cite:`Martens.2010` uses this to take approximate Newton steps: it solves
$\mathbf{H}\boldsymbol{\epsilon} = -\nabla f$ by a few iterations of
conjugate gradients, which need only such products, and damps the system to
keep the step safe. It trained deep autoencoders without the layer-by-layer
pretraining that was common at the time. Saddle-free Newton
:cite:`Dauphin.Pascanu.Gulcehre.ea.2014` addresses the other weakness of
Newton's method, its attraction to saddle points and maxima. It applies the
remedy that :numref:`sec_gd` mentions, the absolute value of the Hessian,
the matrix with the same eigenvectors and the magnitudes of its
eigenvalues, so that a direction of negative curvature repels the iterate
instead of attracting it. For a network it computes this matrix on a small
subspace spanned by Hessian–vector products. The exercise of
:numref:`sec_gd` on absolute-curvature preconditioning is the diagonal
version of the idea.

### Rates for Stochastic Gradient Descent

:numref:`sec_sgd` stated the conditions of Robbins and Monro
:cite:`Robbins.Monro.1951` under which SGD converges and measured its noise
ball, but quoted no rates. For convex objectives with bounded gradients, SGD
with learning rates proportional to $1/\sqrt{t}$ reaches an error of order
$1/\sqrt{T}$ after $T$ steps if its iterates are averaged
:cite:`Polyak.Juditsky.1992`, and no method that sees only stochastic
gradients can do better in the worst case :cite:`Nemirovski.Yudin.1983`. For
smooth nonconvex objectives, where a minimum cannot be guaranteed, SGD
drives the expected squared gradient norm to zero at the same order
:cite:`Ghadimi.Lan.2013`. The appendix proves both results
(:numref:`subsec_mdl-fixed-budget-rates` and
:numref:`subsec_mdl-nonconvex-sgd`). :citet:`Bottou.Curtis.Nocedal.2018`
survey the theory and practice of stochastic methods for large-scale
learning, from noise and averaging to second-order methods.

### Variance Reduction

On a finite training set, the noise of SGD can be removed at the minimum
itself. Methods of the SVRG family store a full gradient at a reference
point and correct each stochastic gradient by the difference between the
gradients of its examples at the current and the reference point. The
corrected estimate is still unbiased, its variance shrinks as both points
approach the minimum, and on a strongly convex objective a constant
learning rate then converges at the linear rate of gradient descent. The
theory is clean, but on deep networks these methods have not given
consistent improvements over SGD with momentum. :numref:`sec_mdl-variance-reduction` derives them and discusses
why.

## Adaptive and Matrix Preconditioners

### A Method Without a Learning Rate

Adadelta :cite:`Zeiler.2012` adds a second exponential average to RMSProp,
of the squared *updates*, and replaces the learning rate by the square root
of that average, so that the numerator and the denominator of the update
carry the same units. The method has no learning rate in name. In practice
its scale on the first steps comes from $\epsilon$, which then acts as one.

### Kronecker-Factored Preconditioners

Adam's preconditioner of :numref:`sec_adam` is diagonal, and Shampoo's in
:numref:`sec_muon` keeps one small matrix per side of each weight matrix.
K-FAC :cite:`Martens.Grosse.2015` arrived at a preconditioner of the same
shape from a different question. The *natural gradient*
:cite:`Amari.1998` preconditions with the inverse of the Fisher information
matrix, the covariance of the gradients of the log-likelihood, which
measures the size of a step by how much it changes the model's predictions
rather than its weights. K-FAC approximates the Fisher block of a layer by
the Kronecker product of two small matrices, the second moments of the
layer's inputs and of the gradients with respect to its outputs, and so
needs the inverses of two small matrices instead of one huge one. SOAP
:cite:`Vyas.Morwani.Zhao.ea.2024` runs Adam in the eigenbasis of Shampoo's
two matrices. Because Adam's statistics adapt within that basis, the
expensive eigendecompositions need to be recomputed only occasionally.
SOAP is among the fastest methods in the matched-tuning studies cited in
:numref:`sec_muon`.

### Steepest Descent Under a Chosen Norm

:numref:`sec_muon` derived Muon as steepest descent under the spectral norm
and gave the remaining parameters to AdamW. Scion
:cite:`Pethick.Xie.Antonakopoulos.ea.2025` assigns a norm to every layer
instead: the spectral norm for hidden matrices, a norm that normalizes each
column for embeddings with one-hot inputs, and a sign or row-normalized step
for the output layer. Each step is then the minimizer of a linear function
over a ball of that norm, and the resulting optimizer needs no Adam at all.
In the same view, decoupled weight decay keeps the weights inside a ball of
the norm, and the learning rates transfer across widths.
:citet:`Kovalev.2025` reads Muon as a trust-region method in a non-Euclidean
norm, which explains why it needs weight decay. On where Muon's gain over
Adam comes from, :citet:`Wang.Zhang.Li.ea.2025` find that it concentrates in
the value and output projections of attention and in the MLP weights, and,
in an associative-memory model, on rare classes, the heavy-tailed regime in
which :numref:`sec_adam` located Adam's own advantage over SGD.

### Faster Orthogonalization

Newton--Schulz in :numref:`sec_muon` adds about an eighth to the arithmetic
of a training step on `TinyLM`, and several lines of work reduce this cost.
Before Polar Express, six quintics found by a heuristic search, one per step
:cite:`Cesista.You.Jordan.2025`, improved the speedrun times of the next
paragraph by one to two percent. CANS :cite:`Grishina.Smirnov.Rakhuba.2025`,
concurrent with Polar Express, uses Chebyshev alternation and the Remez
algorithm to find polynomials that map the spectrum into an interval $[1 -
\delta, 1 + \delta]$ with the largest possible slope at zero. Gram
Newton--Schulz :cite:`Zhang.Amsel.Chen.ea.2026` uses that every odd
polynomial step has the form $h(\mathbf{X}\mathbf{X}^\top)\mathbf{X}$, so
the whole composition can run on the small Gram matrix with symmetric
products and only two multiplications by the rectangular matrix. For
rectangular weights this is up to about twice as fast. Turbo-Muon
:cite:`Boissin.Massena.Mamalet.ea.2025` rescales the rows of the momentum by
a diagonal matrix that makes it nearly orthogonal before the iteration
starts, which saves one of the five steps. On the theory side,
:citet:`Kim.Oh.2026` show that Muon with $q$ Newton--Schulz steps converges
at the rate of exact orthogonalization up to a constant factor that
approaches one doubly exponentially fast in $q$, so that a few steps lose
almost nothing against exact orthogonalization.

### Distributed Orthogonalization

Dion, in :numref:`sec_muon`, avoids gathering a sharded matrix by
orthogonalizing a low-rank approximation. Its successor Dion3
:cite:`Amsel.Zhang.Ahn.ea.2026` orthogonalizes at each step only a
fraction, a quarter or an eighth, of the rows of the momentum, those with
the largest $\ell_1$ norms, keeps the rest in the buffer as Dion does, and
can add NorMuon's per-neuron scaling. On models with a billion parameters
under fully sharded data parallelism, the authors report Muon's loss with up
to six times less time spent in the optimizer. MuonBP
:cite:`Khaled.Ozkara.Yu.ea.2025` orthogonalizes each device's shard on its
own and the full matrix only every few steps, with a separate learning rate
for each kind of step.

### Smaller and Sharded Optimizer State

Adam keeps two numbers per parameter, which for a large model costs more
memory than the weights. Adafactor :cite:`Shazeer.Stern.2018` stores for
each matrix only the row and column sums of $\mathbf{v}$, $m + n$ numbers
instead of $mn$, rebuilds an estimate of $\mathbf{v}$ from their outer
product, and can drop the first moment. 8-bit optimizers
:cite:`Dettmers.Lewis.Shleifer.ea.2022` keep both moments but quantize them
block by block to one byte each, with a scale per block, which preserves
the accuracy of 32-bit Adam on the workloads tested. ZeRO
:cite:`Rajbhandari.Rasley.Ruwase.ea.2020` reduces the state per device
instead of the state in total: it partitions the optimizer state, then the
gradients, then the weights across the devices of a data-parallel group,
rather than replicating them on each. The updates are those of this
chapter. Where their bytes live and when they move is the subject of
:numref:`chap_performance`, in particular :numref:`subsec_mgp-fsdp`.

## Schedules and Averages

### Warm Restarts

Cosine decay, as used in :numref:`sec_scheduler`, was proposed together
with periodic restarts :cite:`Loshchilov.Hutter.2016`: at the end of each
cosine cycle, the learning rate jumps back to its peak, and the cycles may
grow in length. Each cycle ends with a decayed model, so one run yields
several finished models, which can be ensembled. Warmup–stable–decay obtains
models at several budgets from one run differently, by branching decays off
a single plateau without ever returning to a high rate.

### Warmup and Representations

:numref:`sec_scheduler` explained warmup through the sharpness of the loss
at initialization. :citet:`Gotmare.Keskar.Xiong.ea.2018` asked instead what
warmup changes inside the network. They compared the representations of
each layer during training with and without warmup and found that its main
measurable effect is to limit how far the weights of the deepest layers
move while the parameters and the optimizer's statistics are still near
their random initial values.

### Schedule-Free Optimization

Decay removes the noise ball at the end of training by shrinking the late
steps. Averaging the iterates, as in :numref:`sec_practice`, removes it
instead from the parameters that are evaluated. Schedule-free SGD and AdamW
:cite:`Defazio.Yang.Mehta.ea.2024` build training around the average: they
keep a constant learning rate, evaluate each gradient at an interpolation
between the current iterate and the running average of all iterates, and
return the average. The average needs no training horizon, which gives it
the operational advantage of warmup–stable–decay: training can stop, or
continue, at any step. The method won the self-tuning track of the AlgoPerf
benchmark described at the end of this section. An exercise of
:numref:`sec_scheduler` implements its recurrence on the noisy valley.

### Averaging Checkpoints and Models

:numref:`sec_practice` averaged the iterates of one run with an exponential
moving average. Three variants change what is averaged. LAWA
:cite:`Kaddour.2022` averages the latest few saved checkpoints with equal
weights, and on ImageNet and BERT training the average reached the accuracy
or loss of a single checkpoint taken considerably later. Model soups
:cite:`Wortsman.Ilharco.Gadre.ea.2022` average the weights of several models
fine-tuned from the same pretrained model with different hyperparameters.
The soup often beats the best single model and adds no cost at inference.
It works because fine-tuned models from a shared start lie in one basin.
Diffusion models (:numref:`sec_diffusion-images`) evaluate an average of
their weights as a matter of course, and their sample quality depends
strongly on the averaging window. :citet:`Karras.Aittala.Lehtinen.ea.2024`
store a few snapshots during training and reconstruct from them, after
training, an exponential average of any length, so that the window can be
tuned without training again.

## Large Runs and Benchmarks

### Muon from the Speedrun to Production

Muon was first demonstrated on the NanoGPT speedrun
:cite:`Jordan.2024,Jordan.Jin.Boza.ea.2024`, a public competition to train
a fixed GPT-2-sized model to a fixed validation loss in the least
wall-clock time, in which each record changes one thing and publishes the
change. Muon's first record cut the time by about a fifth. The headline time
has since fallen far further, from changes to the optimizer, the
architecture, the systems, and the data together, so it is a record of
engineering rather than a benchmark of optimizers. Moonlight
:cite:`Liu.Su.Yao.ea.2025` was the first Muon run with billions of
parameters. It introduced the RMS-matching factor of :numref:`sec_muon` and
decoupled weight decay for Muon, and found that without decay, Muon falls
behind AdamW late in long runs as the weight norms grow. Kimi K2
:cite:`Kimi.Team.2025` and GLM-4.5 :cite:`Zeng.Lv.Zheng.ea.2025` were
pretrained with Muon.

### Attention-Logit Stability

The logits of attention are dot products of queries and keys, so they grow
with the norms of the vectors that the projections produce, and large
logits saturate the softmax, a common source of loss spikes in
transformers.
QK-norm :cite:`Henry.Dachapally.Pawar.ea.2020` normalizes the queries and
keys just before their dot product, which bounds the logits whatever the
norms upstream. OLMo 2 :cite:`OLMo.2025` adopted it among the stability
changes its report documents. The QK-clip of :numref:`sec_muon` reaches the
same bound differently: it leaves the forward pass alone and rescales the
projection weights after an optimizer step.

### Layerwise Adaptive Learning Rates

LARS and LAMB rescale the update of each layer by the ratio of the norm of
its weights to the norm of its update, a rule proposed to make very large
batches trainable. When :citet:`Nado.Gilmer.Shallue.ea.2021` retuned the
learning rate, warmup, and other nuisance hyperparameters of the baselines
as carefully, plain momentum and Adam matched them at the batch sizes
studied. The comparison illustrates the point of :numref:`sec_practice`:
a new method must be compared against an equally well-tuned baseline.

### Hyperparameter Transfer in Production

:numref:`sec_scaling` transferred a learning rate across widths of one MLP.
The Cerebras-GPT family :cite:`Dey.Gosal.Chen.ea.2023` applied muP,
including its transformer-specific rules, to a whole model family, with
hyperparameters transferred from a proxy of about 40 million parameters,
and published its coordinate checks with the models. DeepSeek
:cite:`Bi.Chen.Chen.ea.2024` kept the standard parametrization and
predicted the drift instead: it fitted power laws for the best learning
rate and batch size as functions of the compute budget across many small
runs and extrapolated them, the approach of an exercise in
:numref:`sec_scaling`. Meta reports that Llama 4 :cite:`Meta.2025` was
trained with per-layer learning rates and initialization scales that
transfer across width, depth, batch size, and training length, without
publishing the method. Transfer in production now extends beyond width,
the one direction the chapter covers.

### Benchmarks and Their Protocols

How optimizers are compared decides which one wins. The AlgoPerf benchmark
:cite:`Dahl.Schneider.Nado.ea.2023` measures the time to reach a target on
a fixed set of workloads, with hyperparameters either tuned within a
fixed budget of trials or set by the method itself. In its first
competition :cite:`Kasimbeg.Schneider.Eschenhagen.ea.2025`, distributed
Shampoo won the externally tuned track and schedule-free AdamW the
self-tuning track. The open-ended tinkering of the speedrun favored Muon
instead. :citet:`Semenov.Pagliardini.Jaggi.2025` ran a second matched-tuning
benchmark on dense and mixture-of-experts language models: AdEMAMix, which
adds a slow second momentum average to AdamW, was consistently strong, Muon
with weight decay was strong especially in short runs, and Muon without
decay on its matrices fell behind. The study of
:citet:`Wen.Hall.Ma.ea.2025` cited in :numref:`sec_practice` also found that
rankings can flip late in a run as the learning rate decays, so optimizers
should be compared at the end of training rather than along the way. The
modded-NanoGPT optimization track :cite:`Jordan.2026` fixes the model, the
data, and the batch, counts steps to a target loss over many seeds, and
accepts a record only if the difference is significant. On it, a tuned
Muon ties NorMuon.

### A Training Logbook

Reports of large runs give the final recipe; the work of keeping a run alive
is rarely published. The OPT team :cite:`zhang2022opt` released its
logbook with its models: two months of restarts after loss spikes and
hardware failures, and of learning rates lowered in mid-run. It
complements the rewind-and-skip remedy of :numref:`sec_practice` with a
day-by-day record of what such remedies cost.

[Discussions](https://d2l.discourse.group/t/36851)
