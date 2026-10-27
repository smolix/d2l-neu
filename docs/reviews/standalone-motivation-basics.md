# Standalone Motivation Review: Basics

## Executive assessment

The Basics part is generally well motivated. Its strongest notebooks begin
with a concrete task or failure: a finite training set may not predict new
examples; a one-megapixel fully connected layer needs billions of weights;
long probability contexts make count tables sparse; and a low-precision square
can overflow. Those openings should be preserved.

The main weaknesses are local rather than pervasive:

1. Several notebooks still begin by saying what the preceding notebook did.
   The local problem is recoverable, but a direct reader must reconstruct it
   from a cross-reference. This occurs most visibly in linear algebra, weight
   decay, concise implementations, convolution mechanics, and RNN
   implementation.
2. The calculus notebook spends substantial space on a historical limit
   construction before stating the optimization problem for which derivatives
   are needed.
3. The Builder's Guide and Modern ConvNets indexes describe an order that does
   not match `_quarto.yml`. Their roadmaps therefore cannot reliably establish
   dependencies for either direct or in-order readers.
4. A few notebooks contain internally duplicated motivation or late catalogue
   passages. Dropout defines and motivates the same random transformation
   twice. AlexNet, efficient ConvNets, ResNet variants, and ConvNeXt each contain
   historical or successor surveys that need a governing comparison question.
5. A small number of implementation notebooks name an abstraction before
   supplying its local meaning. The base classifier calls `accuracy` before
   defining what it measures; concise softmax regression assumes the reader
   remembers the numerical failure of the scratch implementation.

The intended progression of the part is sound:

```text
numerical data and mathematical operations
-> a complete regression training loop
-> categorical prediction and evaluation
-> nonlinear multilayer models
-> reusable, inspectable, device-aware implementations
-> spatial inductive bias and convolution
-> deeper and more efficient convolutional architectures
-> ordered data, autoregressive prediction, recurrence, and decoding
```

A revision should not add generic introductions to every notebook. Most direct
readers need only three local elements: the concrete problem, the limitation of
the simplest available method, and the capability introduced here. An in-order
reader can pass over that bridge in a few sentences.

## Part index: `chapter_basics/index.md`

**Assessment: preserve; low-priority precision edit.** The index explains the
dependency chain from tensors and mathematics through supervised models,
engineering, spatial models, and sequence models
([source](../../chapter_basics/index.md#L3)). It also states the
practical capabilities the part supplies to Advanced
([source](../../chapter_basics/index.md#L12)). A direct reader can
tell both what the part teaches and why its order matters.

**Revision specification.** Preserve the two-paragraph scale and do not expand
it into a notebook catalogue. Replace “the two architectures that dominate
practical work” with a scoped statement: convolution and recurrence are the two
architecture families developed in this part. The current claim is broader
than the book's own subsequent qualification that transformers displaced RNNs
at scale. If useful, add one sentence distinguishing Basics from Advanced:
Basics establishes complete working models and engineering habits; Advanced
removes simplifying assumptions and studies scale.

## Chapter: Preliminaries

### Intended progression

The chapter should let a reader execute and inspect one small learning
calculation:

```text
represent arrays
-> turn imperfect tabular records into arrays
-> express predictions with linear algebra
-> measure local change with calculus
-> compute derivatives automatically
-> reason about finite random samples
-> look up framework behavior that the book cannot enumerate
```

The chapter index already states this progression through a “smallest complete
learning problem” rather than through a list of prerequisites
([source](../../chapter_preliminaries/index.md#L4)). Preserve that
organizing example.

### Notebook assessments

#### `index.md`

**Assessment: preserve.** The opening assigns one indispensable tool to each
step of a learning problem and states the deliberately limited scope
([source](../../chapter_preliminaries/index.md#L4),
[source](../../chapter_preliminaries/index.md#L16)). Do not replace
it with definitions of linear algebra, calculus, or probability. The current
problem-to-tool map is the chapter's most useful orientation.

#### `ndarray.md`

**Assessment: strong; low-priority strengthening.** The opening names tensors
as the common storage object for measurements, features, images, and batches,
then states the two framework capabilities that distinguish them from NumPy
([source](../../chapter_preliminaries/ndarray.md#L9)). Broadcasting
is introduced when equal-shape elementwise operations cease to be sufficient
([source](../../chapter_preliminaries/ndarray.md#L666)), and the
memory section quantifies why allocations matter for gigabytes of frequently
updated parameters ([source](../../chapter_preliminaries/ndarray.md#L769)).

**Revision specification.** Add one compact running operation to the opening,
such as representing a minibatch and computing one elementwise transformation.
This would make “create, inspect, and manipulate” operational without repeating
the chapter index. Replace the “By now” opening of broadcasting with a local
statement: elementwise operations are straightforward for equal shapes; model
code also needs to combine a batch with a feature-wise vector. Preserve the
shape-alignment rule and incompatible-shape counterexample.

#### `pandas.md`

**Assessment: preserve with a low-priority standalone bridge.** Missing values,
categorical encoding, and rescaling establish a concrete reason for a tabular
preprocessing library before pandas is introduced
([source](../../chapter_preliminaries/pandas.md#L9)). The small house
table then drives the complete pipeline. The discussion appropriately marks
the boundary of pandas rather than pretending it covers all modalities and
scales ([source](../../chapter_preliminaries/pandas.md#L259)).

**Revision specification.** Replace “So far, our synthetic data...” with
“A model may receive synthetic data as ready-made tensors, but real tabular
records...” This retains the in-order contrast while giving a direct reader the
missing antecedent. Preserve the pipeline figure and the instruction to inspect
shape and dtypes before transforming data.

#### `linear-algebra.md`

**Medium priority.** The opening says that models require linear algebra and
then lists scalar arithmetic, matrix multiplication, norms, and eigenvalues
([source](../../chapter_preliminaries/linear-algebra.md#L9)). A
direct reader is not shown which learning computation requires those objects.
The later section therefore risks reading as a compact mathematics catalogue.
Norms are introduced as “some of the most useful operators” rather than as a
way to measure prediction, parameter, or perturbation size
([source](../../chapter_preliminaries/linear-algebra.md#L1048)).
Eigenvalues arrive under “A First Look” without a locally stated question.

**Revision specification.** Begin with one batched linear predictor:
`predictions = Xw + b`. Use it to assign roles to scalars, vectors, matrices,
matrix--vector products, and reductions. Introduce norms when asking how large
a parameter vector or prediction error is. Introduce eigenvalues only through
the smallest use developed in this notebook—for example, how repeated linear
maps stretch directions—or explicitly label the subsection as an optional
preview for later dynamics and optimization. Keep the existing elementary
examples, but connect each major operation back to the running predictor in one
sentence.

#### `calculus.md`

**High priority.** The notebook begins with fourteen lines deriving the area of
a circle by polygon limits before it identifies parameter updates and loss
minimization as the deep-learning problem
([source](../../chapter_preliminaries/calculus.md#L9),
[source](../../chapter_preliminaries/calculus.md#L27)). The example
does establish a limit, but a direct reader has no reason to need that limit
until substantial historical detail has passed. The derivative section is
better: it connects rate of change to increasing or decreasing a loss
([source](../../chapter_preliminaries/calculus.md#L75)).

**Revision specification.** Open with a scalar model parameter and a loss curve:
if the parameter changes by a small amount, how much does the loss change, and
which direction lowers it? Introduce the derivative as the limiting answer to
that question. Retain the inscribed-polygon construction as a brief secondary
example of the same limiting idea, or move it to a note after limits are
defined. The progression should then be scalar derivative, partial derivative,
gradient, and chain rule, with each step removing a limitation of the previous
one. Preserve the explicit distinction between optimization on training data
and generalization to unseen data.

#### `autograd.md`

**Assessment: preserve.** The opening states the exact failure of manual
differentiation—complex models make elementary chain-rule applications tedious
and error-prone—and explains the computational graph before naming
backpropagation ([source](../../chapter_preliminaries/autograd.md#L9)).
The control-flow, higher-order derivative, and forward-versus-reverse sections
each state a local need before the mechanism
([source](../../chapter_preliminaries/autograd.md#L718),
[source](../../chapter_preliminaries/autograd.md#L725),
[source](../../chapter_preliminaries/autograd.md#L781)). Preserve
the counting argument that explains why reverse mode fits scalar training
losses.

#### `probability.md`

**Assessment: preserve.** The 56-heads example establishes the inference
problem before defining probability and statistics
([source](../../chapter_preliminaries/probability.md#L9)). The
forward/inverse distinction, diagnostic-testing example, tail bounds, and
uncertainty section all have recoverable questions. In particular, Markov's
inequality is introduced because a mean already constrains large deviations
([source](../../chapter_preliminaries/probability.md#L1121)), not as
an isolated theorem.

**Low-priority revision.** “What the Probability Model Leaves Uncertain” mixes
three tasks: defining aleatoric and epistemic uncertainty, discussing the
sample-size rate, and summarizing Bayes and tail bounds
([source](../../chapter_preliminaries/probability.md#L1161)). Keep
the uncertainty distinction in that section, but move the broader recap to a
separate summary. This will prevent a well-motivated concept from becoming a
late chapter catalogue.

#### `lookup-api.md`

**Assessment: preserve.** API size and release change make memorization
impossible, which creates the need for the discover--inspect--read--verify loop
([source](../../chapter_preliminaries/lookup-api.md#L9),
[source](../../chapter_preliminaries/lookup-api.md#L48)). The MXNet
version caveat is local and consequential. Do not enlarge this into a general
documentation survey; preserve the small verification run as the endpoint.

### Implementation sequence

1. Rewrite the calculus opening around a loss perturbation and relocate the
   polygon example.
2. Give linear algebra a batched-predictor spine, then attach norms and the
   eigenvalue preview to explicit questions.
3. Repair the two prerequisite-dependent phrases in pandas and broadcasting.
4. Separate probability's uncertainty discussion from its chapter recap.
5. Leave the chapter index, autograd, and documentation notebook structurally
   unchanged.

## Chapter: Linear Regression in Neural Networks

### Intended progression

The chapter should expose the complete learning loop in its smallest useful
form:

```text
numerical prediction task
-> linear model, squared loss, and gradient update
-> reusable model/data/trainer interfaces
-> controlled synthetic data
-> explicit implementation
-> framework implementation
-> unseen-data error
-> continuous control of model capacity with weight decay
```

The source order in `_quarto.yml` places object-oriented design before
synthetic data because the data notebook subclasses `DataModule`. The chapter
index's TOC currently reverses those two entries
([source](../../chapter_linear-regression/index.md#L16)).

### Notebook assessments

#### `index.md`

**Medium priority.** The roadmap is inconsistent. The opening correctly presents
linear regression as the simplest complete learning procedure and a useful
baseline ([source](../../chapter_linear-regression/index.md#L4)).
However, the prose says “implement its components from first principles” before
framework abstractions while the actual chapter inserts object-oriented
scaffolding and synthetic data first, and the local TOC orders synthetic data
before object-oriented design even though `_quarto.yml` does the reverse.

**Revision specification.** Make the source and rendered orders agree. Explain
why the engineering abstraction appears early: defining `Module`, `DataModule`,
and `Trainer` once lets every later implementation differ only in the model or
data components under study. Then state why synthetic data precedes fitting:
known ground-truth parameters separate implementation failure from data/model
mismatch. Preserve the claim that linear models are useful baselines.

#### `linear-regression.md`

**Assessment: preserve.** The house-price task introduces features, labels,
unseen prediction, and approximate linearity before formal assumptions
([source](../../chapter_linear-regression/linear-regression.md#L9)).
The notebook then develops model, loss, analytic solution, SGD, prediction,
vectorization, and the Gaussian connection in dependency order. The
noise-model section provides a reason for squared loss rather than merely
naming a standard choice
([source](../../chapter_linear-regression/linear-regression.md#L559)).
Do not add another general definition of supervised learning.

#### `oo-design.md`

**Assessment: preserve.** The repeated training loop and the cost of changing it
in every chapter establish the need for `Module`, `DataModule`, and `Trainer`
before the classes appear
([source](../../chapter_linear-regression/oo-design.md#L9)). The
opening also assigns each class a distinct responsibility. Preserve the figure
and the warning that `add_to_class` is notebook scaffolding rather than a
recommended ordinary-module design
([source](../../chapter_linear-regression/oo-design.md#L67)).

#### `synthetic-regression-data.md`

**Assessment: preserve.** The notebook separates three causes of failure—a
misspecified model, flawed optimization, and pathological data—and explains
how known generating parameters isolate the implementation
([source](../../chapter_linear-regression/synthetic-regression-data.md#L10)).
This is an exemplary standalone opening. Preserve the known-parameter check and
do not replace it with the generic claim that synthetic data is convenient.

#### `linear-regression-scratch.md`

**Assessment: strong; low-priority local context.** The opening lists the four
components and explains why writing them explicitly exposes interfaces
([source](../../chapter_linear-regression/linear-regression-scratch.md#L9)).
It depends on the synthetic-data notebook for the actual generating equation,
however. Add one sentence stating that the examples satisfy
`y = Xw* + b* + noise` and that the final check compares learned with known
parameters. This is enough for a direct reader; do not duplicate the data
loader implementation.

#### `linear-regression-concise.md`

**Low priority.** The opening is framed entirely as a recap of the scratch
implementation ([source](../../chapter_linear-regression/linear-regression-concise.md#L9)).
It identifies the useful difference—library layers, losses, optimizers, and data
loaders—but a direct reader is not told why replacing explicit operations is
desirable until later.

**Revision specification.** Start with the maintenance and correctness problem:
explicit primitives are valuable for inspection, but routine models should use
tested, optimized components. Then map each handwritten component to its
framework counterpart, as the existing paragraph at lines 49--54 already does.
Retain only one cross-reference to scratch as an optional side-by-side
comparison.

#### `generalization.md`

**Assessment: preserve.** The opening distinguishes fitting a finite sample from
predicting new cases and names memorization before defining generalization
([source](../../chapter_linear-regression/generalization.md#L9)).
The IID assumption, training/population error, underfitting/overfitting,
polynomial example, dataset size, and model selection form a coherent
progression. Preserve the finite-sample scale examples and the polynomial
experiment because they create the need for the next notebook.

#### `weight-decay.md`

**Medium priority.** The opening assumes the preceding overfitting and
polynomial experiments (“Now that,” “Recall”) and spends twenty lines
reconstructing their details before identifying the unresolved problem:
discarding features changes capacity too coarsely
([source](../../chapter_linear-regression/weight-decay.md#L9),
[source](../../chapter_linear-regression/weight-decay.md#L20)). A
direct reader can recover the issue, but the need for weight decay arrives late.

**Revision specification.** Open with a self-contained high-dimensional
regression case: many correlated features let a model attain low training loss
with large, unstable coefficients; collecting more data is unavailable and
dropping whole features is too coarse. Introduce a continuous penalty on
coefficient size as the desired capability. Keep the polynomial monomial count
as quantitative support after this problem, not as a prerequisite. When norms
are introduced, state what smaller norm changes operationally in the specific
linear model; avoid implying that parameter norm universally measures function
complexity across parameterizations. Preserve the later spectral explanation
of shrinkage.

### Implementation sequence

1. Reconcile the chapter index, its TOC, and `_quarto.yml`; document why OO
   scaffolding precedes synthetic data.
2. Rewrite the weight-decay opening around coefficient instability and a
   continuous capacity control.
3. Make the concise implementation's maintenance/correctness purpose local.
4. Add one equation-level context sentence to the scratch implementation.
5. Preserve the core linear-regression, OO-design, synthetic-data, and
   generalization openings.

## Chapter: Linear Classification in Neural Networks

### Intended progression

```text
categorical target
-> logits, class probabilities, and cross-entropy
-> a small discriminating image dataset
-> reusable classification metrics and validation
-> explicit softmax implementation
-> stable framework implementation
-> statistical limits of evaluation
-> deployment when the data distribution changes
```

The chapter index states this progression compactly
([source](../../chapter_linear-classification/index.md#L4)).

### Notebook assessments

#### `index.md`

**Assessment: preserve.** The opening identifies exactly which parts of the
regression loop remain and which must change for discrete classes: target
representation, output layer, and loss. Preserve this contrast and the concise
scope statement. No additional list of classification applications is needed
at chapter level.

#### `softmax-regression.md`

**Assessment: strong; low-priority reordering.** The opening establishes the
need for a categorical probability model and loss, and the examples make the
task concrete ([source](../../chapter_linear-classification/softmax-regression.md#L4),
[source](../../chapter_linear-classification/softmax-regression.md#L13)).
However, the detour through multilabel and ordinal classification appears
before the notebook has completed the simple mutually exclusive case
([source](../../chapter_linear-classification/softmax-regression.md#L30)).

**Revision specification.** Keep one sentence delimiting the scope to one
unordered class per example; move ordinal and multilabel variants to a later
boundary note after the softmax model is defined. Preserve the $2\times2$ image
example, the score-to-probability constraint, and the noise-model/cross-entropy
connection.

#### `image-classification-dataset.md`

**Assessment: preserve.** The opening selects Fashion-MNIST through a visible
benchmark-design trade-off: MNIST is too easy to distinguish the models,
ImageNet is too large for interactive notebooks, and Fashion-MNIST retains the
same manageable structure
([source](../../chapter_linear-classification/image-classification-dataset.md#L11)).
Preserve that three-way comparison. Add no generic history of image datasets.

#### `classification.md`

**Medium priority.** The notebook begins by asserting that classification
models require validation loss, accuracy, and an optimizer, then immediately
defines a base class that calls `accuracy`
([source](../../chapter_linear-classification/classification.md#L9),
[source](../../chapter_linear-classification/classification.md#L41)).
Accuracy itself is not defined until line 120. A direct reader can infer the
software goal but not why loss alone is insufficient or what the reported
metric counts. The caveat about averaging batch means is useful but arrives
before the basic quantity it qualifies.

**Revision specification.** Begin with a concrete mismatch: cross-entropy can
change even when the predicted class stays the same, while users also need the
fraction of correct labels. Define accuracy in one sentence before the base
class calls it. Then state why shared validation and optimizer hooks belong in
a base class. Keep the exact-dataset-versus-average-of-batch-means caveat, but
place it after the metric definition. “Beyond Accuracy” should organize its
metrics by the decision problem—class imbalance, asymmetric errors,
probability quality—rather than merely listing alternatives.

#### `softmax-regression-scratch.md`

**Assessment: preserve.** The opening identifies only the components that differ
from regression and says which abstractions are reused
([source](../../chapter_linear-classification/softmax-regression-scratch.md#L9)).
The softmax subsection recalls only the tensor reductions required locally and
then demonstrates the naive overflow before pointing to the stable built-in
operation ([source](../../chapter_linear-classification/softmax-regression-scratch.md#L42),
[source](../../chapter_linear-classification/softmax-regression-scratch.md#L81)).
Preserve this failure demonstration.

#### `softmax-regression-concise.md`

**Medium priority.** The opening says only that frameworks provide a linear
layer and stable cross-entropy and that the notebook is more concise
([source](../../chapter_linear-classification/softmax-regression-concise.md#L11)).
The most important reason for the built-in loss—the fused log-sum-exp
calculation that avoids overflow and needless probabilities—does not become
the local problem until “Softmax Revisited” at line 132.

**Revision specification.** Move a two-sentence version of the numerical
failure to the opening: separately exponentiating logits may overflow, and the
loss only needs the log probability of the target. State that framework
cross-entropy accepts logits and evaluates the combined expression stably.
Then introduce the layer and training code. Keep the full derivation in
“Softmax Revisited”; do not repeat the scratch implementation.

#### `generalization-classification.md`

**Assessment: preserve.** Memorization gives the notebook a concrete failed
solution before it asks about test-set size, reuse, and guarantees
([source](../../chapter_linear-classification/generalization-classification.md#L12),
[source](../../chapter_linear-classification/generalization-classification.md#L23)).
The opening also calibrates the limitations of worst-case guarantees for deep
networks. Preserve the three explicit questions and the separation between a
fresh fixed-model test estimate, repeated reuse, and uniform guarantees.

#### `environment-and-distribution-shift.md`

**Assessment: preserve.** The footwear/loan example shows both passive shift
and policy feedback before the taxonomy appears
([source](../../chapter_linear-classification/environment-and-distribution-shift.md#L9)).
The cat/dog label-flip counterexample proves why assumptions relating source
and target distributions are necessary
([source](../../chapter_linear-classification/environment-and-distribution-shift.md#L44)).
Covariate, label, and concept shift are organized by which distributional term
changes rather than presented as names. Preserve this structure and the
boundary between statistical correction and policy/ethical analysis.

### Implementation sequence

1. Define the role of accuracy before the base classifier uses it, then organize
   “Beyond Accuracy” by decision failure.
2. Move the stable-loss need to the concise softmax opening.
3. Move ordinal and multilabel scope notes until after the base multiclass model.
4. Preserve the dataset-selection, memorization, and deployment-feedback
   openings.

## Chapter: Multilayer Perceptrons

### Intended progression

The chapter should begin with the precise representational failure of a linear
classifier, add hidden nonlinear transformations, implement the resulting
model, explain how reverse-mode differentiation trains it, and then address
the optimization and generalization problems introduced by depth. The final
notebook should demonstrate the complete model-development workflow on a
competitive tabular task. This progression is sound; the main editorial work
is to make the backpropagation bridge more concrete and to remove repetition
from dropout.

#### `index.md`

**Assessment: preserve.** The index explains why a single affine map cannot
represent many useful decision boundaries before introducing hidden layers
([source](../../chapter_multilayer-perceptrons/index.md#L4)). It
also previews both optimization and generalization rather than presenting the
chapter as a list of architectures. Preserve this compact problem-to-remedy
map.

#### `mlp.md`

**Assessment: preserve.** The opening states the limitation of linear models
and uses concrete nonlinear relations before adding a hidden layer
([source](../../chapter_multilayer-perceptrons/mlp.md#L9)). The
XOR construction turns expressivity into a testable claim. Preserve the order:
failed linear model, hidden layer, activation, then capacity discussion.

#### `mlp-implementation.md`

**Low priority.** The notebook clearly says that it will implement the model
from scratch and then with concise framework components
([source](../../chapter_multilayer-perceptrons/mlp-implementation.md#L9)),
but it does not say what the scratch version is meant to verify.

**Revision specification.** Add one sentence naming the observable contract:
the hidden activation must have the expected shape, the output must contain
one logit per class, and both implementations should reach comparable
accuracy. This supplies a reason to inspect the scratch mechanics without
repeating the preceding derivation.

#### `backprop.md`

**Medium priority.** The opening motivates automatic differentiation and the
cost of storing intermediates well
([source](../../chapter_multilayer-perceptrons/backprop.md#L9)).
The discussion then changes to row/column conventions and a generic product
graph ([source](../../chapter_multilayer-perceptrons/backprop.md#L63),
[source](../../chapter_multilayer-perceptrons/backprop.md#L138))
without making explicit how those devices answer the motivating questions.
A direct reader can follow the algebra but may miss why particular values must
be retained and where the memory cost arises.

**Revision specification.** Before the generic graph, state two questions that
the notation will answer: which forward intermediates are needed for each
local derivative, and in what reverse order gradients reach the parameters.
After the graph, identify its nodes with an MLP layer and explicitly connect
retained activations to memory consumption. Keep the generic derivation, but
make it serve the concrete network problem.

#### `numerical-stability-and-init.md`

**Assessment: preserve.** The opening frames exploding and vanishing gradients
as consequences of multiplying many Jacobians and connects initialization to
signal propagation
([source](../../chapter_multilayer-perceptrons/numerical-stability-and-init.md#L10)).
The symmetry argument supplies a distinct failure that variance control alone
does not solve. Preserve both failure modes and their separation.

#### `generalization-deep.md`

**Low priority.** The chapter starts with a careful, bounded account of why
deep-network generalization is difficult
([source](../../chapter_multilayer-perceptrons/generalization-deep.md#L5)).
Its closing transition describes dropout as famous, mysterious, and a mainstay
([source](../../chapter_multilayer-perceptrons/generalization-deep.md#L343)),
which is promotional rather than explanatory.

**Revision specification.** Replace that transition with the unresolved
problem established by the notebook: optimization can prefer interpolating
solutions, so the next section studies a training-time perturbation that may
change this preference. Preserve the distinctions among parameter count,
interpolation, and effective complexity.

#### `dropout.md`

**Medium priority.** The opening gives dropout a useful regularization context
and discusses robustness to missing features
([source](../../chapter_multilayer-perceptrons/dropout.md#L10)).
However, motivation, definition, interpretations, and the exact masking rule
are spread across overlapping passages from lines 10–114. The method is in
effect introduced twice, while ensemble and noise interpretations arrive
before the reader has one stable operational definition.

**Revision specification.** Use one sequence: the overfitting/fragile-
coadaptation problem; the mask-and-rescale operation; where it is inserted in
an MLP; training versus inference behavior; then competing interpretations.
State explicitly that preserving a unit's expectation is a local calculation,
not proof that the nonlinear network equals an ensemble average. Preserve the
caveats about the limits of the ensemble interpretation and avoid claiming a
single settled mechanism.

#### `kaggle-house-price.md`

**Low priority.** The notebook strongly motivates a realistic, end-to-end
tabular prediction task and the need to produce a competition submission
([source](../../chapter_multilayer-perceptrons/kaggle-house-price.md#L9)).
The general explanation of the Kaggle platform is long relative to the local
modeling problem ([source](../../chapter_multilayer-perceptrons/kaggle-house-price.md#L61)).

**Revision specification.** Compress the platform tour and bring the dataset,
log-price metric, preprocessing constraints, and validation strategy forward.
Preserve the notebook's role as an integration exercise rather than turning it
into another exposition of MLP mechanics.

### Implementation sequence

1. Consolidate dropout around a single operational definition and delay its
   interpretations until after the mechanics.
2. Add explicit questions and an MLP mapping to the backpropagation graph.
3. State the scratch implementation's verification goal.
4. Replace promotional dropout language and shorten the Kaggle platform tour.
5. Preserve the nonlinear-expressivity, numerical-stability, and end-to-end
   task openings.

## Chapter: Builder's Guide

### Intended progression

This chapter should move from composing models, to inspecting and initializing
their state, to extending the layer vocabulary, and finally to operating models
reliably across numerical, persistence, device, and reproducibility boundaries.
Each notebook already has a concrete engineering problem. The chapter-level
defect is navigational: the index's title and ordering do not accurately
describe the sequence readers actually encounter.

#### `index.md`

**Medium priority.** The opening makes a convincing transition from using
framework abstractions to understanding their contracts
([source](../../chapter_builders-guide/index.md#L4)). However, the
title “Computation” is broader than the chapter's model-building and operating
scope, and both the prose and local table of contents order saving, numerics,
custom layers, and devices differently from the rendered order
([source](../../chapter_builders-guide/index.md#L18)).

**Revision specification.** Rename the chapter to reflect model construction
and operation. Make the roadmap match the actual order: model construction;
parameters and state; initialization; custom layers; numerics; saving and
loading; GPUs, devices, and memory; reproducibility and inspection. Give each
cluster a governing question rather than a list of APIs.

#### `model-construction.md`

**Assessment: preserve.** The opening identifies the limit of treating a
network as a black-box sequence and motivates custom composition, parameter
access, and control flow
([source](../../chapter_builders-guide/model-construction.md#L9)).
Preserve the progression from sequential containers to arbitrary blocks.

**Low-priority addition.** Motivate the configuration subsection with a
specific persistence problem: source code alone does not record the
constructor choices needed to recreate a trained model.

#### `parameters-state-memory.md`

**Assessment: preserve.** The opening asks concrete questions about locating,
sharing, and updating model state
([source](../../chapter_builders-guide/parameters-state-memory.md#L9)).
The notebook also reinstates its local specimen model before manipulating it
([source](../../chapter_builders-guide/parameters-state-memory.md#L49)),
which makes it usable in isolation. Preserve that local setup and distinguish
parameters from non-parameter state throughout.

#### `init.md`

**Low priority.** The opening clearly explains why default initialization is
not always enough and why matching parameters by type or name matters
([source](../../chapter_builders-guide/init.md#L9)). The examples
show mechanics but do not state the numerical invariant that motivates common
fan-aware schemes.

**Revision specification.** Add one sentence connecting fan-in and fan-out to
keeping activation and gradient variance from changing drastically across
layers. Do not repeat the full initialization theory from the MLP chapter.

#### `custom-layers.md`

**Assessment: preserve.** The opening gives two concrete reasons to write a
layer: a parameter-free transformation and a learned operation absent from the
standard library
([source](../../chapter_builders-guide/custom-layers.md#L9)).
Preserve the small examples and the explicit distinction between layers with
and without parameters.

#### `numerics.md`

**Assessment: preserve.** The notebook begins with the finite range and
precision of machine numbers and names failures that affect training
([source](../../chapter_builders-guide/numerics.md#L9)). The
squaring example makes overflow concrete before prescribing safer computation
([source](../../chapter_builders-guide/numerics.md#L64)). Preserve
the failure-first organization and the measured examples.

#### `saving-loading.md`

**Assessment: preserve.** Training cost, interruption, deployment, and
reproducible comparison establish why state must survive a process boundary
before serialization APIs appear
([source](../../chapter_builders-guide/saving-loading.md#L9)).
Preserve the distinction among tensors, parameter dictionaries, and complete
reconstructible models.

#### `gpus-devices-memory.md`

**Assessment: preserve.** The opening connects accelerator speed to explicit
device placement and finite memory, making both parts of the title necessary
([source](../../chapter_builders-guide/gpus-devices-memory.md#L9)).
Preserve the device-mismatch failures and allocation measurements rather than
replacing them with an API catalogue.

#### `reproducibility-inspection.md`

**Low priority.** The opening correctly treats nondeterminism and hidden state
as practical debugging problems
([source](../../chapter_builders-guide/reproducibility-inspection.md#L9)).
Its two themes can appear adjacent rather than causally connected.

**Revision specification.** Add a transition stating that reproducible runs
are a prerequisite for interpreting changes discovered through hooks,
summaries, and profilers. Preserve the careful qualification that exact
determinism may carry performance costs and may differ across hardware.

### Implementation sequence

1. Correct the index title, roadmap, and notebook order.
2. Add the reconstruction need to model configuration and the variance
   invariant to initialization.
3. Connect reproducibility explicitly to diagnosis and inspection.
4. Preserve the failure-first numerics, persistence, and device notebooks.

## Chapter: Convolutional Neural Networks

### Intended progression

The chapter should derive convolution from the structure of images, define the
operator, control its spatial geometry, extend it across channels, summarize
local evidence with pooling, and assemble the pieces into LeNet. The derivation
and architecture notebook are especially strong. Only the convolution-layer
notebook leans too heavily on the preceding section for its local purpose.

#### `index.md`

**Assessment: preserve.** The index starts from the mismatch between dense
networks and spatially structured data and previews locality and translation
equivariance as responses
([source](../../chapter_convolutional-neural-networks/index.md#L4)).
Its roadmap follows the actual conceptual dependency chain. Preserve it.

#### `why-conv.md`

**Assessment: preserve.** The megapixel-image parameter count makes the dense
model's failure quantitative before invariance assumptions are introduced
([source](../../chapter_convolutional-neural-networks/why-conv.md#L4)).
Locality and translation equivariance are then used to derive, rather than
merely announce, convolution. Preserve this as the chapter's conceptual anchor.

#### `conv-layer.md`

**High priority.** The opening says that the preceding section derived the
convolutional layer and immediately turns to implementation
([source](../../chapter_convolutional-neural-networks/conv-layer.md#L9)).
A reader entering here does not get a local task that explains what the kernel
should detect, what its output represents, or why correlation is useful.

**Revision specification.** Open with one small edge-detection problem: the
same vertical boundary may occur anywhere in an image, so one small pattern
detector should be reused at every location. Show the input, kernel, and output
shape in words before formalizing cross-correlation. Briefly distinguish the
mathematical convolution convention from the operator used by deep-learning
libraries. Then proceed to learning a kernel. This requires only a paragraph
and one tiny example; do not repeat the full invariance derivation from
`why-conv.md`.

#### `padding-and-strides.md`

**Assessment: preserve.** The notebook motivates padding through boundary
information and shrinking spatial dimensions, then motivates stride through
redundant computation and resolution control
([source](../../chapter_convolutional-neural-networks/padding-and-strides.md#L9)).
The formulas answer these concrete shape questions.

**Low-priority correction.** Dilation is also introduced later
([source](../../chapter_convolutional-neural-networks/padding-and-strides.md#L333)).
Rename the section “Padding, Stride, and Dilation,” or explicitly label dilation
as an optional extension in the opening.

#### `channels.md`

**Assessment: preserve.** The opening explains why a single two-dimensional
array is insufficient for color inputs and multiple learned features
([source](../../chapter_convolutional-neural-networks/channels.md#L9)).
Grouped and depthwise convolution later answer the cost of dense
cross-channel mixing rather than arriving as unexplained variants
([source](../../chapter_convolutional-neural-networks/channels.md#L255)).
Preserve that cost-to-structure transition.

#### `pooling.md`

**Assessment: preserve.** Sensitivity to exact feature location and the desire
to reduce spatial resolution motivate pooling before max and average operators
are defined
([source](../../chapter_convolutional-neural-networks/pooling.md#L9)).
Preserve the distinction between local invariance and simple downsampling.

#### `lenet.md`

**Assessment: preserve.** The notebook frames LeNet as the first complete test
of the components just developed and describes why alternating convolution,
pooling, and dense prediction is coherent
([source](../../chapter_convolutional-neural-networks/lenet.md#L9)).
Preserve the layer-by-layer shape inspection and the connection between
architecture and training results.

### Implementation sequence

1. Give `conv-layer.md` a self-contained edge-detection problem and operator
   contract.
2. Bring the padding notebook's title or opening into alignment with dilation.
3. Preserve the quantitative dense-network failure, the grouped-convolution
   cost argument, and the complete LeNet assembly.

## Chapter: Modern Convolutional Neural Networks

### Intended progression

The chapter should explain successive design pressures rather than narrate a
lineage of model names: scale convolutional networks to realistic images;
factor them into reusable blocks; stabilize deep optimization; preserve useful
function classes with residual paths; meet deployment budgets; improve training
without changing architecture; revisit convolution with Transformer-era
choices; and finally extract general design principles. Most notebooks already
follow this logic. The chapter index has an order error, and several late
sections drift into architecture catalogues.

#### `index.md`

**Medium priority.** The opening and prose roadmap describe the pressures that
led from AlexNet through residual and efficient networks
([source](../../chapter_convolutional-modern/index.md#L4)).
The local table of contents puts training recipes before efficient networks,
while the rendered order and surrounding prose put efficient networks first
([source](../../chapter_convolutional-modern/index.md#L27),
[source](../../chapter_convolutional-modern/index.md#L46)).

**Revision specification.** Make the table of contents match the actual order.
In the overview, identify each notebook by its design question, not only its
representative architecture. Preserve the distinction between architectural
change and training-recipe improvement.

#### `alexnet.md`

**Low priority.** The opening clearly explains why LeNet-scale success did not
immediately translate to natural images and identifies data, compute, and
optimization as missing conditions
([source](../../chapter_convolutional-modern/alexnet.md#L10)).
The later “Representation Learning” passage partially repeats that history and
adds a sequence of names
([source](../../chapter_convolutional-modern/alexnet.md#L58)).

**Revision specification.** Retain the opening's three-condition explanation.
Condense the later history to the one contrast needed locally: learned
hierarchies replaced hand-engineered visual pipelines once scale made end-to-
end training viable. Move names that do not affect the AlexNet comparison to a
brief note or timeline.

#### `blocks.md`

**Assessment: preserve.** Repetition and code organization provide a unifying
reason to study blocks rather than a sequence of unrelated architectures
([source](../../chapter_convolutional-modern/blocks.md#L9)).
NiN's pointwise convolutions answer the need for richer channel mixing without
large spatial kernels
([source](../../chapter_convolutional-modern/blocks.md#L290)).
Preserve the explicit mapping from architectural motif to reusable block.

**Low-priority refinement.** Shorten the late lineage of named block designs
unless each is compared along a common axis such as parameter sharing,
multi-scale processing, or computation.

#### `batch-norm.md`

**Assessment: preserve.** The notebook begins with the difficulty of training
very deep networks and makes internal scale drift concrete before defining
normalization
([source](../../chapter_convolutional-modern/batch-norm.md#L9),
[source](../../chapter_convolutional-modern/batch-norm.md#L48)).
Preserve the separation between the operational transformation and unsettled
explanations of why it helps.

**Low-priority refinement.** Organize the “Beyond Batch Normalization”
alternatives by which axes they normalize and which settings they target,
rather than by chronology.

#### `resnet.md`

**Assessment: preserve.** The opening asks whether adding layers should make a
model harder to optimize and introduces nested function classes as the
criterion for a safe extension
([source](../../chapter_convolutional-modern/resnet.md#L9)).
Residual connections then follow as a way to recover the identity mapping.
ResNeXt answers width/cardinality cost, while DenseNet makes concatenation an
explicit alternative to addition
([source](../../chapter_convolutional-modern/resnet.md#L568),
[source](../../chapter_convolutional-modern/resnet.md#L758)).
Preserve the function-class argument; it is stronger than historical acclaim.

**Low-priority refinement.** Introduce ResNeXt and DenseNet under one question:
how should later layers preserve, transform, or reuse earlier information?

#### `efficient-convnets.md`

**Medium priority.** Deployment latency, energy, memory, and hardware behavior
give the notebook a strong governing problem
([source](../../chapter_convolutional-modern/efficient-convnets.md#L9)).
The late scaling and search discussion introduces EfficientNet variants and
newer MobileNet generations in quick succession
([source](../../chapter_convolutional-modern/efficient-convnets.md#L523)).
The shared tradeoff is present but easy to lose among names.

**Revision specification.** Add a compact comparison table with rows for the
design families and columns for manual versus searched design, optimized
resource, principal scaling knob, and whether the cost is analytical or
measured on target hardware. Introduce only details needed to explain those
tradeoffs. Preserve the notebook's warning that FLOPs alone do not determine
latency.

#### `training-recipes.md`

**Assessment: preserve.** Holding the architecture fixed while raising
accuracy from 76.1% to 80.4% makes the experimental question and payoff
specific
([source](../../chapter_convolutional-modern/training-recipes.md#L9)).
The ingredient list is governed by that controlled comparison rather than
presented as arbitrary best practices. Preserve the fixed baseline, cumulative
ablation logic, and distinction between training and inference cost.

#### `convnext.md`

**Medium priority.** The opening poses a clean controlled question: how far can
a standard ConvNet advance by adopting selected Transformer-era design choices
while retaining convolution
([source](../../chapter_convolutional-modern/convnext.md#L9)).
“Beyond ConvNeXt” becomes a rapid catalogue of ConvNeXt V2, RepLKNet, SLaK,
InternImage, and UniRepLKNet
([source](../../chapter_convolutional-modern/convnext.md#L650)).

**Revision specification.** Organize the closing section around two unresolved
problems: avoiding feature collapse under masked pretraining, and acquiring
wide spatial context efficiently. Compare the named methods only insofar as
they choose different solutions. A two-axis table is preferable to one
paragraph per name. Preserve the notebook's controlled modernization sequence.

#### `cnn-design.md`

**Medium priority.** The opening inventories many preceding architectures
before reaching the local contrast between intuition-led design, neural
architecture search, and systematic design-space exploration
([source](../../chapter_convolutional-modern/cnn-design.md#L9)).
For an isolated reader, the names consume attention before the method of
comparison is clear.

**Revision specification.** Open with the local problem: architecture choices
are coupled, exhaustive search is expensive, and isolated anecdotes do not
yield transferable rules. Then compare manual design, search, and controlled
design-space studies. Convert the prior model inventory into a compact table
of axes—stage depth, width, bottleneck, operator, normalization, and downsample
placement—and use it as evidence for the derived principles.

### Implementation sequence

1. Correct the index order and restate its roadmap as design questions.
2. Reorganize the late ConvNeXt and efficient-network catalogues around explicit
   comparison axes.
3. Rewrite the CNN-design opening around the need for controlled design-space
   reasoning.
4. Condense duplicated AlexNet history and minor block lineages.
5. Preserve the residual function-class argument and the fixed-architecture
   training-recipe experiment.

## Chapter: Recurrent Neural Networks and Sequence Models

### Intended progression

The chapter should establish why order and variable context matter, convert
text into a sequence-learning dataset, formulate probabilistic language
modeling, replace fixed windows with recurrent state, implement and train the
model, derive gradient flow through time, and finally separate model scoring
from the search procedure used for generation. The conceptual arc is strong.
The language-model and scratch-implementation notebooks need more local purpose
at their openings.

#### `index.md`

**Low priority.** The index clearly contrasts fixed-size, order-insensitive
prediction with sequence problems and previews state, language modeling, and
decoding
([source](../../chapter_recurrent-neural-networks/index.md#L4)).
Its first two paragraphs repeat the fixed-shape/order limitation, and “worth
learning” is promotional rather than informative.

**Revision specification.** Merge the repeated contrast into one paragraph and
use the saved space to distinguish three questions: how to represent temporal
dependence, how to estimate sequence probabilities, and how to search over
possible outputs. Preserve the roadmap and the explicit qualification that
recurrent networks are not the only modern sequence architecture.

#### `sequence.md`

**Assessment: preserve.** A sequence of observations whose present value
depends on past values establishes the modeling problem immediately
([source](../../chapter_recurrent-neural-networks/sequence.md#L4)).
The notebook then motivates fixed-lag autoregression, latent state, and
multi-step prediction through their distinct information and error-propagation
constraints. Preserve the running data example and the separation of one-step
from recursive forecasting.

#### `text-sequence.md`

**Assessment: preserve.** The opening makes text both a practical sequence
source and a preprocessing problem before tokenization code appears
([source](../../chapter_recurrent-neural-networks/text-sequence.md#L4)).
Character, word, and byte units are compared by vocabulary size, sequence
length, unknown-token behavior, and semantic granularity. Preserve those
tradeoffs and the explicit corpus statistics.

#### `language-model.md`

**Medium priority.** The notebook opens with the definition of a language model
and immediately factorizes a joint probability
([source](../../chapter_recurrent-neural-networks/language-model.md#L4)).
Generation and scoring applications arrive later
([source](../../chapter_recurrent-neural-networks/language-model.md#L28)).
The mathematics is relevant, but an isolated reader meets the solution before
the decision it supports.

**Revision specification.** Begin with two uses of the same next-token
distribution: predict plausible continuations and score which complete
sequence is more plausible. Then state that assigning a probability to an
entire variable-length sequence appears difficult, and use the chain rule to
reduce it to conditional predictions. Introduce n-grams as a finite-context
approximation whose data and memory requirements grow with context. Preserve
the exact factorization and frequency demonstration.

#### `rnn.md`

**Assessment: preserve.** The opening identifies the fixed n-gram window and
its exponential context cost before proposing a state that is updated and
reused
([source](../../chapter_recurrent-neural-networks/rnn.md#L5)).
The recurrent equations therefore answer a visible failure rather than merely
defining a cell. Preserve the path from latent state to the unrolled
computation graph.

#### `rnn-implementation.md`

**Medium priority.** The opening is an itinerary—implement from scratch, then
use concise APIs—and relies on cross-references for the reason
([source](../../chapter_recurrent-neural-networks/rnn-implementation.md#L4)).
The notebook's strongest local motivation appears much later: repeated
Jacobians can make gradients explode, requiring clipping
([source](../../chapter_recurrent-neural-networks/rnn-implementation.md#L497)),
and autoregressive generation needs a warm-up state and feedback loop
([source](../../chapter_recurrent-neural-networks/rnn-implementation.md#L710)).

**Revision specification.** Open with the recurrent implementation contract:
at each time step the model consumes the current token and previous hidden
state, returns logits and a new state, and must preserve compatible shapes
across an arbitrary sequence length. Say that the scratch version exposes this
contract and the concise version delegates it to a framework. Preview gradient
clipping as protection needed when this transition is repeated many times.
Keep the detailed failure and generation discussions where they are.

#### `bptt.md`

**Assessment: preserve.** The opening asks how recurrent parameter sharing and
long computational paths change gradient computation and storage
([source](../../chapter_recurrent-neural-networks/bptt.md#L4)).
The derivation is therefore governed by computational and statistical
questions, not algebra alone. Preserve the comparison among full, truncated,
and randomized truncation and its explicit bias/cost tradeoff.

#### `decoding.md`

**Assessment: preserve.** The notebook immediately distinguishes evaluating
candidate sequences from finding a high-scoring sequence
([source](../../chapter_recurrent-neural-networks/decoding.md#L4)).
Its later tree formulation makes the combinatorial search problem concrete
before greedy, exhaustive, and beam strategies are compared
([source](../../chapter_recurrent-neural-networks/decoding.md#L58)).
Preserve the common scoring objective and accuracy/computation comparison.

### Implementation sequence

1. Reframe the language-model opening around continuation and sequence-scoring
   tasks before the chain-rule solution.
2. Give the RNN implementation a local hidden-state and shape contract and
   preview the repeated-gradient failure.
3. Tighten the index's repeated opening and replace promotional language.
4. Preserve the fixed-context failure, BPTT tradeoffs, and decoding search
   formulation.
